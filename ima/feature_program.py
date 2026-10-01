"""Featuretools synthesis over validated past-only historical records."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_discovery_specs import DiscoverySpec, content_id


def history_table(frame: pd.DataFrame, spec: DiscoverySpec) -> pd.DataFrame:
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
        else:
            history[name] = pd.to_numeric(frame.actual_weight, errors="coerce")
    history.replace([np.inf, -np.inf], np.nan, inplace=True)
    history["start_id"] = np.arange(len(history))
    return history


def synthesize(frame: pd.DataFrame, spec: DiscoverySpec) -> tuple[pd.DataFrame, dict]:
    import featuretools as ft
    from woodwork.logical_types import Double, Categorical

    started = time.monotonic()
    if frame[["race_id", "horse_no"]].duplicated().any():
        raise ValueError("Duplicate runner keys")
    history = history_table(frame, spec)
    included = pd.Series(True,index=frame.index)
    if spec.history_sources:
        if "source" not in frame or not set(spec.history_sources) <= set(frame.source.dropna().astype(str)):
            raise ValueError("Unregistered history source")
        included = frame.source.astype(str).isin(spec.history_sources)
    cutoff_dates = pd.to_datetime(frame.date, errors="raise").dt.normalize()
    output = pd.DataFrame(index=frame.index)
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
            unique = cutoffs.drop_duplicates().reset_index(drop=True)
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
                matrix = ft.calculate_feature_matrix(features=eligible, entityset=es, cutoff_time=unique, cutoff_time_in_index=True, include_cutoff_time=False, training_window=(f"{window} days" if window else None), approximate=None, n_jobs=1)
                matrix.columns = names
                query = pd.MultiIndex.from_frame(cutoffs[[key, "time"]])
                output[names] = matrix.reindex(query).to_numpy(dtype=float)
                definitions[f"{entity}-{window}"] = json.loads(ft.save_features(eligible))
    for entity in sorted(spec.entities):
        key=f"{entity}_id"
        for window in spec.sequence_windows:
            for measurement in spec.measurements:
                for operation in ("mean","std"):
                    definition={"entity":entity,"expression":f"{operation}({measurement})","prior_starts":window,"availability_rule":spec.availability_rule}
                    fid="dfs_"+content_id(definition)
                    if len(catalog)>=spec.max_definitions:
                        deferred.append(definition)
                        continue
                    values=pd.Series(np.nan,index=frame.index)
                    for identity,group in history.loc[included].groupby(key,sort=True):
                        group=group.sort_values(["available_at","start_id"],kind="stable")
                        rows=history.index[history[key].eq(identity)]
                        last=np.searchsorted(group.available_at.to_numpy(),cutoff_dates.loc[rows].to_numpy(),side="left")-1
                        rolled=getattr(group[measurement].rolling(window,min_periods=1),operation)().to_numpy()
                        valid=last>=0
                        values.loc[rows[valid]]=rolled[last[valid]]
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
    manifest = {"schema_version": 1, "discovery_id": spec.discovery_id(), "engine": "featuretools", "engine_version": ft.__version__, "spec": spec.model_dump(mode="json"), "catalog": catalog, "definitions": definitions, "deferred_count": len(deferred), "deferred": deferred, "availability_rule": "historical results available no earlier than next midnight; explicit observation may delay further; strict cutoff", "runner_keys_hash": content_id(frame[["race_id", "horse_no"]].astype(str).to_dict("records")), "coverage": output.notna().mean().to_dict(), "wall_seconds": time.monotonic() - started}
    return output, manifest


def materialize(frame: pd.DataFrame, spec: DiscoverySpec, source_hash: str, cache: Path) -> tuple[pd.DataFrame, dict]:
    identity = content_id({"source_hash": source_hash, "spec": spec.model_dump(mode="json"), "builder": 1})
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
            manifest.update(matrix_id=identity, source_hash=source_hash, matrix_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            temporary_meta = directory / "manifest.tmp"
            temporary_meta.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            os.replace(temporary_meta, meta)
        if len(matrix) != len(frame):
            raise ValueError("Discovery cache runner count mismatch")
    return pd.concat([frame.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1), manifest
