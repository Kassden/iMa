"""Featuretools synthesis over validated past-only historical records."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import warnings
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_discovery_specs import DiscoverySpec, content_id


def history_table(frame: pd.DataFrame, spec: DiscoverySpec, input_metadata=None) -> pd.DataFrame:
    history = pd.DataFrame(index=frame.index)
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    history["available_at"] = dates + pd.Timedelta(days=1)
    if "observed_at" in frame:
        observed = pd.to_datetime(frame["observed_at"], errors="raise")
        if observed.isna().any():
            raise ValueError("Explicit observed_at cannot be missing")
        history["available_at"] = pd.concat([history.available_at, observed], axis=1).max(axis=1)
    for entity in spec.entities:
        key = f"{entity}_id"
        values = frame[key] if key in frame else pd.Series(np.nan,index=frame.index)
        fallback = f"{entity}_key"
        if fallback in frame:
            values = values.fillna(frame[fallback])
        if values.isna().any():
            raise ValueError(f"Missing historical identity: {key}")
        history[key] = values.astype(str)
    for name in spec.measurements:
        if name == "speed_mps":
            if "finish_seconds" not in frame:
                raise ValueError("speed_mps requires validated finish_seconds")
            seconds = pd.to_numeric(frame.finish_seconds, errors="coerce")
            history[name] = pd.to_numeric(frame.distance, errors="coerce") / seconds.where(seconds > 0)
        elif name == "beaten_lengths":
            if "lengths_raw" not in frame:
                raise ValueError("beaten_lengths requires normalized lengths_raw")
            history[name] = pd.to_numeric(frame.lengths_raw, errors="coerce")
        elif name == "carried_weight":
            history[name] = pd.to_numeric(frame.actual_weight, errors="coerce")
        else:
            from .feature_expressions import NumericExpression, InputMetadata, validate_expression
            if spec.schema_version != 2 or input_metadata is None or name not in input_metadata or name not in frame:
                raise ValueError(f"Custom measurement requires validated input metadata: {name}")
            metadata = InputMetadata.model_validate(input_metadata[name]) if isinstance(input_metadata[name],dict) else input_metadata[name]
            validate_expression(NumericExpression(op="ref",ref=name),{name:metadata})
            if not pd.api.types.is_numeric_dtype(frame[name]):
                raise ValueError(f"Custom measurement must be numeric: {name}")
            history[name] = frame[name].to_numpy(dtype=float)
            if metadata.available_at_column:
                available = pd.to_datetime(frame[metadata.available_at_column],errors="raise")
                if available.isna().any():
                    raise ValueError("Custom historical measurements require complete availability")
                history["available_at"] = pd.concat([history.available_at,available],axis=1).max(axis=1)
    history.replace([np.inf, -np.inf], np.nan, inplace=True)
    history["start_id"] = np.arange(len(history))
    return history


def synthesize(frame: pd.DataFrame, spec: DiscoverySpec, *, enumerate_only=False, cutoff_positions=None, input_metadata=None) -> tuple[pd.DataFrame, dict]:
    import featuretools as ft
    from woodwork.logical_types import Double, Categorical

    started = time.monotonic()
    if frame[["race_id", "horse_no"]].duplicated().any():
        raise ValueError("Duplicate runner keys")
    history = history_table(frame, spec, input_metadata)
    included = pd.Series(True,index=frame.index)
    if spec.history_sources:
        if "source" not in frame or not set(spec.history_sources) <= set(frame.source.dropna().astype(str)):
            raise ValueError("Unregistered history source")
        included = frame.source.astype(str).isin(spec.history_sources)
    cutoff_dates = pd.to_datetime(frame.date, errors="raise").dt.normalize()
    output = pd.DataFrame(index=[] if enumerate_only else frame.index)
    catalog = []
    definitions = {}
    deferred = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for entity in sorted(spec.entities):
            key = f"{entity}_id"
            records = history.loc[included,["start_id", "available_at", key, *spec.measurements]].copy()
            parents = pd.DataFrame({key: sorted(history[key].unique())})
            es = ft.EntitySet(id=f"history-{entity}")
            es = es.add_dataframe(dataframe_name=entity, dataframe=parents, index=key, logical_types={key: Categorical})
            es = es.add_dataframe(dataframe_name="starts", dataframe=records, index="start_id", time_index="available_at", logical_types={**{m: Double for m in spec.measurements}, key: Categorical})
            es = es.add_relationship(entity, key, "starts", key)
            es.add_last_time_indexes()
            generated = ft.dfs(entityset=es, target_dataframe_name=entity, agg_primitives=list(spec.aggregates), trans_primitives=[], max_depth=spec.max_depth, features_only=True)
            generated = sorted((f for f in generated if f.get_depth() > 0), key=lambda f: f.get_name())
            cutoffs = pd.DataFrame({key: history[key], "time": cutoff_dates})
            unique = (cutoffs if cutoff_positions is None else cutoffs.iloc[cutoff_positions]).drop_duplicates().reset_index(drop=True)
            for window in spec.windows_days:
                eligible = []
                names = []
                for feature in generated:
                    definition = {"entity": entity, "expression": feature.get_name(), "window_days": window, "availability_rule": spec.availability_rule}
                    fid = "dfs_" + content_id(definition)
                    if len(catalog) >= spec.max_definitions:
                        deferred.append(definition)
                        continue
                    catalog.append({"feature_id": fid, **definition})
                    eligible.append(feature)
                    names.append(fid)
                if not eligible:
                    continue
                definitions[f"{entity}-{window}"] = json.loads(ft.save_features(eligible))
                if enumerate_only:
                    output[names] = pd.DataFrame(columns=names)
                    continue
                matrix = ft.calculate_feature_matrix(features=eligible, entityset=es, cutoff_time=unique, cutoff_time_in_index=True, include_cutoff_time=False, training_window=(f"{window} days" if window else None), approximate=None, n_jobs=1)
                matrix.columns = names
                query = pd.MultiIndex.from_frame(cutoffs[[key, "time"]])
                output[names] = matrix.reindex(query).to_numpy(dtype=float)
                definitions[f"{entity}-{window}"] = json.loads(ft.save_features(eligible))
    for entity in sorted(spec.entities):
        key=f"{entity}_id"
        for window in spec.sequence_windows:
            for measurement in spec.measurements:
                for operation in ("mean","std","count","lag"):
                    definition={"entity":entity,"expression":f"{operation}({measurement})","prior_starts":window,"availability_rule":spec.availability_rule}
                    fid="dfs_"+content_id(definition)
                    if len(catalog)>=spec.max_definitions:
                        deferred.append(definition)
                        continue
                    if enumerate_only:
                        output[fid]=pd.Series(dtype=float)
                        catalog.append({"feature_id":fid,**definition})
                        continue
                    values=pd.Series(np.nan,index=frame.index)
                    for identity,group in history.loc[included].groupby(key,sort=True):
                        group=group.sort_values(["available_at","start_id"],kind="stable")
                        rows=history.index[history[key].eq(identity)]
                        last=np.searchsorted(group.available_at.to_numpy(),cutoff_dates.loc[rows].to_numpy(),side="left")-1
                        if operation=="lag":
                            rolled=group[measurement].to_numpy()
                        else:
                            rolled=getattr(group[measurement].rolling(window,min_periods=1 if operation=="count" else window),operation)().to_numpy()
                        valid=last>=0
                        values.loc[rows[valid]]=rolled[last[valid]]
                    output[fid]=values
                    catalog.append({"feature_id":fid,**definition})
    if spec.domain_history:
        if "horse_id" not in history or "speed_mps" not in history:
            raise ValueError("Domain history requires horse entity and speed_mps measurement")
        domain = history.loc[included].copy()
        domain["distance_band"] = (pd.to_numeric(frame.loc[included,"distance"]) / 400).round()
        current=history.copy()
        current["distance_band"]=(pd.to_numeric(frame.distance)/400).round()
        for context in ("surface","venue"):
            if context in frame:
                domain[context]=frame.loc[included,context]
                current[context]=frame[context]
        domain["carried_weight"]=pd.to_numeric(frame.loc[included,"actual_weight"],errors="coerce")
        if "horse_rating" in frame:
            rating=pd.to_numeric(frame.horse_rating,errors="coerce")
            grouped=rating.groupby(frame.race_id)
            opponent=(grouped.transform("sum")-rating)/(grouped.transform("count")-rating.notna().astype(int)).replace(0,np.nan)
            domain["opponent_rating"]=opponent.loc[included]
        for operation in ("elapsed_days", "recency_weighted_speed", "distance_conditioned_speed", "surface_conditioned_speed", "venue_conditioned_speed", "carried_weight_change", "prior_opponent_strength"):
            definition={"entity":"horse","expression":operation,"recency_decay_days":spec.recency_decay_days,"availability_rule":spec.availability_rule}
            required_context={"surface_conditioned_speed":"surface","venue_conditioned_speed":"venue","prior_opponent_strength":"opponent_rating"}.get(operation)
            if required_context and required_context not in domain:
                deferred.append(dict(definition,reason="missing_context"))
                continue
            fid="dfs_"+content_id(definition)
            if len(catalog)>=spec.max_definitions:
                deferred.append(definition)
                continue
            if enumerate_only:
                output[fid]=pd.Series(dtype=float)
                catalog.append({"feature_id":fid,**definition})
                continue
            values=pd.Series(np.nan,index=frame.index)
            context={"distance_conditioned_speed":"distance_band","surface_conditioned_speed":"surface","venue_conditioned_speed":"venue"}.get(operation)
            group_keys=["horse_id",context] if context else ["horse_id"]
            queries={key: rows.index for key,rows in current.groupby(group_keys,sort=False)}
            for key,group in domain.groupby(group_keys,sort=False):
                rows=queries.get(key)
                if rows is None:
                    continue
                group=group.sort_values(["available_at","start_id"],kind="stable")
                count=np.searchsorted(group.available_at.to_numpy(),cutoff_dates.loc[rows].to_numpy(),side="left")
                valid=count>0
                if operation=="elapsed_days":
                    values.loc[rows[valid]]=(cutoff_dates.loc[rows[valid]].to_numpy()-group.available_at.to_numpy()[count[valid]-1])/np.timedelta64(1,"D")
                elif operation=="carried_weight_change":
                    values.loc[rows[valid]]=pd.to_numeric(frame.loc[rows[valid],"actual_weight"],errors="coerce").to_numpy()-group.carried_weight.to_numpy()[count[valid]-1]
                else:
                    measure="opponent_rating" if operation=="prior_opponent_strength" else "speed_mps"
                    weights=np.exp((group.available_at-group.available_at.max()).dt.total_seconds().to_numpy()/(86400*spec.recency_decay_days)) if operation=="recency_weighted_speed" else np.ones(len(group))
                    weights=np.where(group[measure].notna(),weights,0)
                    sums=np.concatenate([[0],np.cumsum(group[measure].fillna(0).to_numpy()*weights)])
                    support=np.concatenate([[0],np.cumsum(weights)])
                    values.loc[rows[valid]]=np.divide(sums[count[valid]],support[count[valid]],out=np.full(valid.sum(),np.nan),where=support[count[valid]]>0)
            output[fid]=values
            catalog.append({"feature_id":fid,**definition})
    if spec.race_relative:
        for original in list(output):
            for operation in ("rank","center"):
                definition={"entity":"race","expression":operation,"input_feature_id":original}
                fid="dfs_"+content_id(definition)
                if len(catalog)>=spec.max_definitions:
                    deferred.append(definition)
                    continue
                grouped=output[original].groupby(frame.race_id)
                output[fid]=grouped.rank(pct=True) if operation=="rank" else output[original]-grouped.transform("mean")
                catalog.append({"feature_id":fid,**definition})
    prediction_context=frame if cutoff_positions is None else frame.iloc[cutoff_positions]
    if cutoff_positions is not None and not enumerate_only:
        output=output.iloc[cutoff_positions]
    manifest = {"schema_version": 1, "discovery_id": spec.discovery_id(), "engine": "featuretools", "engine_version": ft.__version__, "spec": spec.model_dump(mode="json"), "catalog": catalog, "definitions": definitions, "deferred_count": len(deferred), "deferred": deferred, "availability_rule": "historical results available no earlier than next midnight; explicit observation may delay further; strict cutoff", "runner_keys_hash": content_id(prediction_context[["race_id", "horse_no"]].astype(str).to_dict("records")), "coverage": output.notna().mean().to_dict(), "wall_seconds": time.monotonic() - started}
    return output, manifest


def materialize(frame: pd.DataFrame, spec: DiscoverySpec, source_hash: str, cache: Path, *, shared=False, input_metadata=None):
    if shared:
        return _materialize_shared(frame,spec,source_hash,cache,input_metadata=input_metadata)
    import featuretools as ft
    builder_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    generator_spec=spec.model_dump(mode="json",exclude={"max_selected","missingness_limit","correlation_threshold","selection","seed","adjusted_speed_residuals","residual_shrinkage","diagnostics"})
    identity = content_id({"source_hash": source_hash, "generator_spec": generator_spec, "builder_hash": builder_hash, "engine_version": ft.__version__})
    directory = Path(cache) / identity
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path, meta = directory / "matrix.parquet", directory / "manifest.json"
        if meta.exists():
            manifest = json.loads(meta.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["matrix_sha256"]:
                raise ValueError("Corrupt discovery cache")
            matrix = pd.read_parquet(path)
        else:
            matrix, manifest = synthesize(frame, spec)
            temporary = directory / f"matrix-{os.getpid()}.parquet"
            matrix.to_parquet(temporary)
            os.replace(temporary, path)
            manifest.update(matrix_id=identity, source_hash=source_hash, generator_spec=generator_spec, builder_hash=builder_hash, matrix_path=str(path.resolve()), matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest["catalog_shards"]=[{"shard_id":content_id(manifest["catalog"][start:start+64]),"feature_ids":[item["feature_id"] for item in manifest["catalog"][start:start+64]]} for start in range(0,len(manifest["catalog"]),64)]
            temporary_meta = directory / "manifest.tmp"
            temporary_meta.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            os.replace(temporary_meta, meta)
        if len(matrix) != len(frame):
            raise ValueError("Discovery cache runner count mismatch")
        manifest=dict(manifest,discovery_id=spec.discovery_id(),spec=spec.model_dump(mode="json"))
    return pd.concat([frame.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1), manifest


@dataclass(frozen=True)
class DiscoveryMatrixArtifact:
    """Pickle-safe numeric descriptor; base frame/labels stay with the caller."""
    root: str
    matrix_id: str
    cache_status: str

    @property
    def path(self):
        if len(self.matrix_id) != 24 or any(c not in "0123456789abcdef" for c in self.matrix_id):
            raise ValueError("Invalid discovery matrix identity")
        return Path(self.root)/self.matrix_id

    def manifest(self):
        manifest = json.loads((self.path/"manifest.json").read_text())
        if manifest["matrix_id"] != self.matrix_id or _file_sha256(self.path/"matrix.joblib") != manifest["matrix_sha256"]:
            raise ValueError("Corrupt shared discovery matrix")
        return manifest

    def load(self):
        import joblib
        manifest = self.manifest()
        array = joblib.load(self.path/"matrix.joblib",mmap_mode="r")
        if array.size == 0:
            array.setflags(write=False)
        if list(array.shape) != manifest["shape"] or str(array.dtype) != manifest["dtype"] or array.flags.writeable:
            raise ValueError("Shared discovery shape/dtype/read-only mismatch")
        return array

    @property
    def feature_names(self):
        return tuple(self.manifest()["feature_names"])

    @property
    def shared_artifacts(self):
        return {self.matrix_id:self.manifest()["nbytes"]}

    def frame(self, *, columns=None, positions=None):
        """Explicit compact copy for training-only screening; never attaches base data."""
        names = self.feature_names
        columns = names if columns is None else tuple(columns)
        indices = [names.index(name) for name in columns]
        values = self.load()
        rows = np.arange(len(values)) if positions is None else np.asarray(positions)
        return pd.DataFrame(values[np.ix_(rows,indices)],columns=columns)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_shared(frame,spec,source_hash,cache,*,input_metadata=None):
    import featuretools as ft
    import joblib
    root = Path(cache).absolute()/"shared-numeric-v1"
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o022:
        raise ValueError("Shared discovery cache must be trusted own-user storage")
    excluded = {"max_selected","missingness_limit","correlation_threshold","selection","seed","adjusted_speed_residuals","residual_shrinkage","diagnostics","selection_shortlist","selection_fit_budget"}
    generator_spec = spec.model_dump(mode="json",exclude=excluded)
    builder_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    row_keys = frame[["race_id","horse_no"]].astype(str).to_dict("records")
    metadata = {name:value.model_dump(mode="json") if hasattr(value,"model_dump") else value for name,value in (input_metadata or {}).items()}
    custom = sorted(set(spec.measurements)-{"speed_mps","beaten_lengths","carried_weight"})
    custom_columns = sorted(set(custom) | {metadata[name]["available_at_column"] for name in custom if name in metadata and metadata[name].get("available_at_column")})
    custom_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[custom_columns],index=False).to_numpy().tobytes()).hexdigest() if custom_columns else None
    identity = content_id({"source_hash":source_hash,"runner_keys":row_keys,"generator_spec":generator_spec,"builder_hash":builder_hash,"engine_version":ft.__version__,"numpy_version":np.__version__,"pandas_version":pd.__version__,"joblib_version":joblib.__version__,"input_metadata":metadata,"custom_measurements_hash":custom_hash})
    directory = root/identity
    with open(root/(identity+".lock"),"a+b") as lock:
        waited = False
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            waited = True
            deadline = time.monotonic()+600
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Shared discovery cache lock exceeded 600 seconds")
                    time.sleep(.05)
        artifact = DiscoveryMatrixArtifact(str(root),identity,"wait" if waited else "hit")
        try:
            manifest = artifact.manifest()
            artifact.load()
            if manifest["runner_keys"] != row_keys:
                raise ValueError("Shared matrix runner keys mismatch")
        except (OSError,ValueError,KeyError,EOFError):
            if directory.exists():
                shutil.rmtree(directory)
            temporary = Path(tempfile.mkdtemp(dir=root,prefix=identity+"-"))
            try:
                matrix,manifest = synthesize(frame,spec,input_metadata=input_metadata)
                array = np.ascontiguousarray(matrix.to_numpy(dtype="float64"))
                joblib.dump(array,temporary/"matrix.joblib",compress=0)
                manifest.update(matrix_id=identity,source_hash=source_hash,generator_spec=generator_spec,builder_hash=builder_hash,runner_keys=row_keys,feature_names=list(matrix.columns),shape=list(array.shape),dtype=str(array.dtype),nbytes=array.nbytes,shared_bytes=array.nbytes,matrix_path=str((directory/"matrix.joblib").absolute()),matrix_sha256=_file_sha256(temporary/"matrix.joblib"),storage="joblib-readonly-numeric")
                with open(temporary/"manifest.json","w") as stream:
                    json.dump(manifest,stream,sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                with open(temporary/"matrix.joblib","rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary,directory)
                directory_fd = os.open(root,os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                artifact = DiscoveryMatrixArtifact(str(root),identity,"miss")
                artifact.load()
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = dict(manifest,discovery_id=spec.discovery_id(),spec=spec.model_dump(mode="json"),cache_status=artifact.cache_status,shared_bytes=manifest["nbytes"],artifact_descriptor={"root":artifact.root,"matrix_id":artifact.matrix_id,"cache_status":artifact.cache_status})
        return artifact,manifest
