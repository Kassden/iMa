"""Durable, paired feature experiments sharing the research dataset registry."""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
import stat
from pathlib import Path
import shutil
import sqlite3
import time

from .dataset_registry import _digest, _write_json, file_sha256
from .feature_definitions import FeatureDefinition
from .feature_lifecycle import _parent, build_feature_candidate, record_model_outcome, retry_tracking as retry_build_tracking
from .feature_studies import paired_feature_report
from .research_controller import protocol_spec_parameters
from .research_executor import RecipeExecutionRequest, execute_recipe
from .research_specs import PipelineRecipe, V6_PORTFOLIO_VERSION


_QUEUE_SCHEMA = """CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, source TEXT NOT NULL,
  definitions TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}',
  error TEXT, created REAL NOT NULL, updated REAL NOT NULL)"""


def _queue_growth_bytes(*fields):
    """Conservative per-transaction bound from every bound UTF-8 value."""
    return 128 * 1024 + 3 * sum(len(value.encode("utf-8")) for value in fields
                                if isinstance(value, str))


def _missing_directories(path):
    path = Path(path).absolute()
    count = 0
    current = path
    while not current.exists():
        if current.is_symlink():
            raise ValueError("Symlink directory rejected")
        count += 1
        current = current.parent
    return count


EXPERIMENT = "ima-feature-exploration"


@dataclass(frozen=True)
class ExplorationConfig:
    root: Path
    parent_features: Path
    registry: Path
    main_campaign: Path | None = None
    tracking_uri: str | None = None
    code_revision: str = "local"
    environment_hash: str = "local"
    refill_size: int = 32
    low_watermark: int = 8
    native_threads: int = 2
    fit_timeout: float = 900
    disk_reserve_bytes: int = 10_000_000_000
    model_l2: float = 1.0
    model_max_iter: int = 1000

    def __post_init__(self):
        for name in ("refill_size", "low_watermark", "native_threads", "model_max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Positive integer {name} required")
        if not math.isfinite(self.fit_timeout) or self.fit_timeout <= 0 or self.disk_reserve_bytes < 0:
            raise ValueError("Positive finite runtime and nonnegative disk reserve required")
        if not math.isfinite(self.model_l2) or self.model_l2 <= 0:
            raise ValueError("Positive fixed regularization required")


class ExplorationQueue:
    def __init__(self, path, context_id=""):
        self.path = Path(path)
        self.context_id = context_id
        if self._admission_enabled():
            from .research_sqlite import _peek_journal_mode, admitted_sqlite
            if _peek_journal_mode(self.path) == "wal":
                raise RuntimeError("SQLite WAL mode is unsupported for admitted feature queue writes")
            with admitted_sqlite(self.path, estimated_growth_bytes=_queue_growth_bytes(_QUEUE_SCHEMA),
                                 stage="feature_exploration_queue_schema") as connection:
                connection.execute(_QUEUE_SCHEMA)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self.connect()) as connection, connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(_QUEUE_SCHEMA)

    @staticmethod
    def _admission_enabled():
        import os
        return bool(os.environ.get("IMA_STORAGE_ROOT"))

    def connect(self):
        if self._admission_enabled():
            if self.path.absolute().resolve() != self.path.absolute():
                raise ValueError('Feature queue path must be canonical')
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            # Refuse WAL before opening SQLite so reads cannot create -shm/-wal
            # sidecars or perform recovery against a legacy queue.
            from .research_sqlite import _peek_journal_mode
            if _peek_journal_mode(self.path) == "wal":
                raise RuntimeError("SQLite WAL mode is unsupported for admitted feature queue reads")
            connection = sqlite3.connect(
                self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
            connection.row_factory = sqlite3.Row
            return connection
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _write(self, *fields):
        if self._admission_enabled():
            from .research_sqlite import _peek_journal_mode, admitted_sqlite
            if _peek_journal_mode(self.path) == "wal":
                raise RuntimeError("SQLite WAL mode is unsupported for admitted feature queue writes")
            with admitted_sqlite(self.path, estimated_growth_bytes=_queue_growth_bytes(*fields),
                                 stage="feature_exploration_queue_write") as connection:
                yield connection
        else:
            with closing(self.connect()) as connection, connection:
                yield connection

    def enqueue(self, definitions, source="deterministic"):
        definitions = [d if isinstance(d, FeatureDefinition) else FeatureDefinition.model_validate(d) for d in definitions]
        if not definitions or len(definitions) > 16:
            raise ValueError("Feature batch must contain 1 to 16 definitions")
        definitions.sort(key=lambda d: d.content_id())
        if len({d.content_id() for d in definitions}) != len(definitions):
            raise ValueError("Duplicate feature definitions in batch")
        identifier = "feature-test-" + _digest({"definitions": [d.content_id() for d in definitions],
                                              "context_id": self.context_id})
        now = time.time()
        encoded_definitions = json.dumps([d.model_dump(mode="json") for d in definitions],
                                         ensure_ascii=False, allow_nan=False)
        with self._write(identifier, "queued", source, encoded_definitions, now) as connection:
            connection.execute("INSERT OR IGNORE INTO entries(id,status,source,definitions,created,updated) VALUES (?, 'queued', ?, ?, ?, ?)",
                               (identifier, source, encoded_definitions, now, now))
        return identifier

    def rows(self):
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT * FROM entries ORDER BY CASE source WHEN 'orchestrator' THEN 0 WHEN 'adaptive_bundle' THEN 1 ELSE 2 END,created,id").fetchall()
        return [dict(row) | {"definitions": json.loads(row["definitions"]), "result": json.loads(row["result"])} for row in rows]

    def update(self, identifier, status, result=None, error=None):
        with self._write(status, json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else "",
                          error or "", identifier) as connection:
            changed = connection.execute("UPDATE entries SET status=?,result=COALESCE(?,result),error=?,updated=? WHERE id=?",
                (status, json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else None,
                 error, time.time(), identifier)).rowcount
        if changed != 1:
            raise KeyError(identifier)

    def recover(self):
        with self._write("queued", "Recovered interrupted feature work", "running") as connection:
            connection.execute("UPDATE entries SET status='queued',error='Recovered interrupted feature work',updated=? WHERE status='running'", (time.time(),))


@contextmanager
def exploration_lock(root):
    path = Path(root)
    if not os.environ.get('IMA_STORAGE_ROOT'):
        path.mkdir(parents=True, exist_ok=True)
        with (path/'loop.lock').open('a') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another feature exploration loop owns this queue') from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return
    lock_path = path/"loop.lock"
    from .research_storage import admitted_output, reserved_output_file, as_storage_blocked
    # Admit root creation and the stable lock inode only. The reservation is
    # released before yielding so it never spans a campaign.
    inodes = _missing_directories(path) + (0 if lock_path.exists() else 1)
    retained = None
    try:
        with admitted_output(lock_path, estimated_bytes=max(4096, inodes*65536),
                             stage='feature_exploration_loop_lock', inodes=inodes):
            path.mkdir(parents=True, exist_ok=True)
            with reserved_output_file(lock_path, 'ab') as stream:
                info = os.fstat(stream.fileno())
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError('Another feature exploration loop owns this queue') from exc
                current = lock_path.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                        or not os.path.samestat(info, current)):
                    raise ValueError('Unsafe or changed feature exploration loop lock')
                retained = os.dup(stream.fileno())
    except BaseException as exc:
        if retained is not None:
            os.close(retained)
        converted = as_storage_blocked(exc, stage='feature_exploration_loop_lock', output=lock_path)
        if converted is not None and converted is not exc:
            raise converted from exc
        raise
    primary = None
    try:
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            os.close(retained)
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f'Feature exploration lock cleanup also failed: {cleanup!r}')


def feature_screen_worker(request, native_threads):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=native_threads):
        return execute_recipe(request).serializable()


class FeatureExplorer:
    def __init__(self, config: ExplorationConfig):
        from . import feature_lifecycle, modeling, research_evaluation, research_executor
        self.config = config
        self.root = Path(config.root)
        from .research_storage import admitted_output
        with admitted_output(self.root, estimated_bytes=max(4096,_missing_directories(self.root)*65536),
                             stage="feature_exploration_root",
                             inodes=_missing_directories(self.root)):
            self.root.mkdir(parents=True, exist_ok=True)
        self.parent = _parent(config.parent_features, config.parent_features.parent/"manifest.json")
        self.protocol = protocol_spec_parameters(json.loads((config.parent_features.parent/"protocol.json").read_text()))
        context = {"parent_dataset_id": self.parent["dataset_id"],
                   "parent_manifest_sha256": file_sha256(config.parent_features.parent/"manifest.json"),
                   "protocol": self.protocol, "model": self.recipe(self.parent["dataset_id"], ()).canonical_payload(),
                   "code_revision": config.code_revision, "environment_hash": config.environment_hash,
                   "native_threads": config.native_threads,
                   "implementation_hashes": {module.__name__: file_sha256(module.__file__) for module in
                       (feature_lifecycle, modeling, research_evaluation, research_executor)},
                   "exploration_implementation_sha256": file_sha256(__file__)}
        self.context_id = _digest(context)
        self.queue = ExplorationQueue(self.root/"queue.sqlite", self.context_id)
        identity = self.root/"context.json"
        if identity.exists() and json.loads(identity.read_text()) != context:
            raise ValueError("Feature queue context changed; use a new exploration root")
        _write_json(identity, context)

    def recipe(self, dataset_id, features):
        return PipelineRecipe.model_validate({"schema_version": 3, "feature_schema": "baseline-v1",
            "dataset_ref": dataset_id, "extra_numeric_features": list(features),
            "model": {"kind": "benter_conditional_logit", "parameters": {
                "l2": self.config.model_l2, "max_iter": self.config.model_max_iter}},
            "blend": {"kind": "none"}, "seed": 17})

    def refill(self):
        from .feature_candidates import candidate_definitions
        rows = self.queue.rows()
        queued = sum(row["status"] == "queued" for row in rows)
        excluded = {FeatureDefinition.model_validate(d).content_id() for row in rows for d in row["definitions"]}
        if queued < self.config.low_watermark:
            for definition in candidate_definitions(self.parent["predictor_catalog"], exclude_ids=excluded,
                                                    limit=self.config.refill_size):
                self.queue.enqueue([definition])
        self.ingest_planner()
        self.adaptive_bundle()

    def ingest_planner(self):
        if not self.config.main_campaign:
            return
        path = Path(self.config.main_campaign)/"decisions.sqlite"
        if not path.is_file():
            return
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT DISTINCT d.payload FROM decisions d JOIN receipts r ON d.id=r.decision_id WHERE r.action LIKE 'program:%'").fetchall()
        errors = []
        for raw, in rows:
            for proposal in json.loads(raw).get("programs", []):
                recipe = proposal.get("recipe", {})
                if recipe.get("dataset_ref") == self.parent["dataset_id"] and recipe.get("feature_definitions"):
                    try:
                        self.queue.enqueue(recipe["feature_definitions"], "orchestrator")
                    except ValueError as exc:
                        errors.append({"proposal_id": proposal.get("proposal_id"), "error": str(exc)})
        if errors:
            _write_json(self.root/"planner-ingestion-errors.json", errors)

    def adaptive_bundle(self):
        rows = self.queue.rows()
        screened = [row for row in rows if row["status"] == "screened"]
        if not screened or len(screened) % 8:
            return
        promising = sorted([row for row in screened if len(row["definitions"]) == 1
                            and row["result"].get("paired_delta", 0) < 0],
                           key=lambda row: row["result"]["paired_delta"])[:4]
        if len(promising) >= 2:
            self.queue.enqueue([row["definitions"][0] for row in promising], "adaptive_bundle")

    def _execute(self, request, executor=None):
        result_path = request.output_dir/"result.json"
        if result_path.exists():
            stored = json.loads(result_path.read_text())
            if (stored.get("status") == "completed" and stored.get("recipe_hash") == request.recipe.recipe_hash()
                    and stored.get("lineage", {}).get("code_revision") == request.code_revision):
                for artifact, digest in (("predictions", "prediction_sha256"), ("protocol", "protocol_hash")):
                    path = Path(stored["artifacts"][artifact])
                    if not path.is_file() or file_sha256(path) != stored["lineage"][digest]:
                        raise ValueError("Cached feature fit artifact checksum mismatch: "+artifact)
                return stored
        if executor is None:
            return feature_screen_worker(request, self.config.native_threads)
        # Scientific identity is stable; each failed execution gets a fresh deadline.
        executions = request.output_dir/"executions"
        from .research_storage import admitted_output
        with admitted_output(executions, estimated_bytes=max(4096,_missing_directories(executions)*65536),
                             stage="feature_exploration_executions",
                             inodes=_missing_directories(executions)):
            executions.mkdir(parents=True, exist_ok=True)
        attempt = len(list(executions.glob("execution-*.json")))+1
        future = executor.submit(feature_screen_worker, request, self.config.native_threads,
                                 runtime_path=executions/f"execution-{attempt:06d}.json", timeout=self.config.fit_timeout)
        while not future.done():
            progress = request.output_dir/"progress.json"
            self.status("screening", active_id=request.attempt_id,
                        progress=json.loads(progress.read_text()) if progress.exists() else {})
            time.sleep(2)
        return future.result()

    def _request(self, identifier, manifest, features=()):
        return RecipeExecutionRequest(identifier, "feature-exploration", 0,
            self.recipe(manifest["dataset_id"], features), Path(manifest["features_path"]),
            self.root/"trials"/identifier, self.protocol, manifest["files"]["features.parquet"],
            self.config.code_revision, self.config.environment_hash, V6_PORTFOLIO_VERSION)

    def baseline(self, executor=None):
        identifier = "feature-control-" + self.context_id
        request = self._request(identifier, self.parent)
        result = self._execute(request, executor)
        if result["status"] != "completed":
            raise RuntimeError("Feature baseline failed: " + str(result.get("error")))
        return result

    def _observation(self, result, manifest, definition_ids):
        lineage = result["lineage"]
        return {"attempt_id": result["attempt_id"], "dataset_id": manifest["dataset_id"],
            "model_id": "benter_conditional_logit:"+result["recipe_hash"], "definition_ids": definition_ids,
            "model_context": {"feature_schema": "baseline-v1", "l2": self.config.model_l2,
                              "max_iter": self.config.model_max_iter, "stage": "feature_screen",
                              "claim": "not a rich-model or champion improvement"},
            "score": result["objective_value"], "comparison_contract": {
                "comparison_contract_id": manifest["comparison_contract_id"], "target_contract": "win_probability",
                "evaluation_population_id": lineage["evaluation_population_id"],
                "declared_evaluation_population_id": lineage["declared_evaluation_population_id"],
                "protocol_id": manifest["comparison_contract"]["protocol_id"],
                "protocol_hash": lineage["protocol_hash"], "score_population_sha256": lineage["evaluation_population_hash"],
                "metric": result["objective_name"], "direction": "minimize",
                "probability_basis": lineage["probability_basis"]}}

    def step(self, executor=None):
        if shutil.disk_usage(self.root).free < self.config.disk_reserve_bytes:
            self.status("waiting_disk")
            return None
        self.refill()
        pending = next((row for row in self.queue.rows() if row["status"] == "queued"), None)
        if pending is None:
            self.status("vocabulary_exhausted")
            return None
        identifier = pending["id"]
        self.queue.update(identifier, "running")
        self.status("building", active_id=identifier)
        try:
            manifest = build_feature_candidate(self.config.parent_features,
                self.config.parent_features.parent/"manifest.json", self.config.registry,
                definitions=pending["definitions"], tracking_uri=self.config.tracking_uri)
        except ValueError as exc:
            self.queue.update(identifier, "rejected", error=str(exc))
            self.status("candidate_rejected")
            return {"id": identifier, "status": "rejected", "error": str(exc)}
        except OSError as exc:
            self.queue.update(identifier, "queued", error=str(exc))
            self.status("build_retry")
            return None
        try:
            baseline = self.baseline(executor)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self.queue.update(identifier, "queued", error=str(exc))
            self.status("baseline_retry", error=str(exc))
            raise
        try:
            self.status("screening", active_id=identifier, dataset_id=manifest["dataset_id"])
            result = self._execute(self._request(identifier, manifest, manifest["feature_build"]["feature_names"]), executor)
            if result["status"] != "completed":
                raise ValueError("Candidate fit failed: " + str(result.get("error")))
            paired = paired_feature_report(result["artifacts"]["predictions"], baseline["artifacts"]["predictions"], "win_probability")
            outcome = self._observation(result, manifest, manifest["feature_definition_ids"])
            control = self._observation(baseline, self.parent, [])
            report = record_model_outcome(self.config.registry, manifest["feature_build_id"], outcome, baseline=control)
            if not report["paired"]:
                raise ValueError("Feature/control comparison contracts differ")
            payload = {"result": result, "baseline": baseline, "paired": paired, "paired_delta": paired["paired_delta"],
                "build_id": manifest["feature_build_id"], "dataset_id": manifest["dataset_id"],
                "feature_ids": manifest["feature_build"]["feature_names"],
                "definition_ids": manifest["feature_definition_ids"],
                "build_run_id": manifest["feature_build_receipt"]["tracking"].get("run_id"),
                "source": pending["source"], "tracking_status": "pending" if self.config.tracking_uri else "disabled",
                "scope": "adaptive development screening; model-specific, no automatic promotion"}
            _write_json(self.root/"trials"/identifier/"paired-report.json", paired)
            self.queue.update(identifier, "screened", payload)
            self.publish_tracking(identifier, payload)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            self.queue.update(identifier, "failed", error=f"{type(exc).__name__}: {exc}")
            self.status("candidate_failed")
            return {"id": identifier, "status": "failed", "error": str(exc)}
        self.status("screened")
        return {"id": identifier, "status": "screened", "paired_delta": paired["paired_delta"]}

    def publish_tracking(self, identifier, payload):
        if not self.config.tracking_uri:
            return
        try:
            build_tracking = retry_build_tracking(self.config.registry, payload["build_id"],
                                                 tracking_uri=self.config.tracking_uri)
            if build_tracking["status"] != "published" or not build_tracking.get("run_id"):
                raise RuntimeError("Feature build tracking pending: "+str(build_tracking.get("error")))
            payload["build_run_id"] = build_tracking["run_id"]
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=self.config.tracking_uri)
            experiment = client.get_experiment_by_name(EXPERIMENT)
            experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
            controllers = client.search_runs([experiment_id], filter_string=f"tags.`ima.feature_controller` = '{self.context_id}'", max_results=2)
            if len(controllers) > 1:
                raise ValueError("Duplicate feature controller identity")
            controller = controllers[0] if controllers else client.create_run(experiment_id, tags={
                "mlflow.runName": "feature exploration: "+self.context_id[:12],
                "ima.stage": "feature_exploration_controller", "ima.feature_controller": self.context_id})
            snapshot = self.status("tracking")
            for name, value in {"generated": snapshot["generated"], "screened": snapshot["screened"],
                                "rejected": snapshot["counts"].get("rejected", 0),
                                "failed": snapshot["counts"].get("failed", 0),
                                "queued": snapshot["counts"].get("queued", 0)}.items():
                client.log_metric(controller.info.run_id, name, value)
            for role, result in (("control", payload["baseline"]), ("feature_screen", payload["result"])):
                attempt = result["attempt_id"]
                runs = client.search_runs([experiment_id], filter_string=f"tags.`ima.feature_attempt_id` = '{attempt}'", max_results=2)
                if len(runs) > 1:
                    raise ValueError("Duplicate MLflow feature attempt identity")
                run = runs[0] if runs else client.create_run(experiment_id, tags={
                    "mlflow.runName": f"{role}: {attempt}", "ima.stage": role,
                    "ima.feature_attempt_id": attempt, "ima.scope": payload["scope"],
                    "mlflow.parentRunId": controller.info.run_id})
                run_id = run.info.run_id
                if role == "control":
                    payload["baseline_run_id"] = run_id
                else:
                    payload["run_id"] = run_id
                self.queue.update(identifier, "screened", payload)
                if run.info.status == "FINISHED":
                    continue
                params = {"feature_schema": "baseline-v1", "model_kind": "benter_conditional_logit",
                    "target_kind": "win_probability", "model_l2": self.config.model_l2,
                    "model_max_iter": self.config.model_max_iter, "context_id": self.context_id,
                    "dataset_id": result["lineage"]["dataset_id"], "code_revision": self.config.code_revision,
                    "feature_worker_llm_calls": 0}
                if role != "control":
                    params.update(feature_ids=json.dumps(payload["feature_ids"]),
                        baseline_run_id=payload["baseline_run_id"], feature_build_run_id=payload.get("build_run_id") or "pending",
                        feature_build_id=payload["build_id"], candidate_source=payload["source"])
                for key, value in params.items():
                    client.log_param(run_id, key, value)
                metrics = {"objective": result["objective_value"], "duration_seconds": result["duration_seconds"]}
                if role != "control":
                    metrics.update(paired_delta=payload["paired_delta"], races=payload["paired"]["races"],
                        race_ci95_lower=payload["paired"]["race_bootstrap_ci95"][0],
                        race_ci95_upper=payload["paired"]["race_bootstrap_ci95"][1],
                        definition_count=len(payload["definition_ids"]))
                for key, value in metrics.items():
                    if value is not None and math.isfinite(float(value)):
                        client.log_metric(run_id, key, float(value))
                directory = self.root/"trials"/attempt
                for name in ("result.json", "paired-report.json"):
                    if (directory/name).exists():
                        client.log_artifact(run_id, str(directory/name))
                client.set_terminated(run_id)
                if client.get_run(run_id).info.status != "FINISHED":
                    raise ValueError("Feature run readback failed")
            payload.update(tracking_status="published", tracking_error=None)
        except Exception as exc:
            payload.update(tracking_status="pending", tracking_error=f"{type(exc).__name__}: {exc}",
                           tracking_retry_after=time.time()+60)
        self.queue.update(identifier, "screened", payload)

    def retry_tracking(self):
        for row in self.queue.rows():
            payload = row["result"]
            if (row["status"] == "screened" and payload.get("tracking_status") == "pending"
                    and payload.get("tracking_retry_after", 0) <= time.time()):
                self.publish_tracking(row["id"], payload)

    def status(self, operation="idle", **extra):
        with self.queue.connect() as connection:
            counts = dict(connection.execute("SELECT status,COUNT(*) FROM entries GROUP BY status").fetchall())
            pending_tracking = connection.execute("SELECT COUNT(*) FROM entries WHERE status='screened' AND json_extract(result,'$.tracking_status')='pending'").fetchone()[0]
            best = connection.execute("""SELECT id,json_extract(result,'$.dataset_id') AS dataset_id,
                json_extract(result,'$.paired_delta') AS paired_delta,json_extract(result,'$.run_id') AS run_id,
                json_extract(result,'$.definition_ids') AS definition_ids FROM entries WHERE status='screened'
                ORDER BY json_extract(result,'$.paired_delta'),id LIMIT 8""").fetchall()
            failures = connection.execute("SELECT id,status,error FROM entries WHERE error IS NOT NULL ORDER BY updated DESC LIMIT 8").fetchall()
        snapshot = {"updated_at_epoch": time.time(), "operation": operation, "context_id": self.context_id,
            "parent_dataset_id": self.parent["dataset_id"], "counts": counts, "generated": sum(counts.values()),
            "screened": counts.get("screened", 0), "pending_tracking": pending_tracking,
            "worker_capacity": 1, "native_threads": self.config.native_threads,
            "promising": [dict(row) | {"definition_ids": json.loads(row["definition_ids"])} for row in best],
            "recent_errors": [dict(row) for row in failures],
            "scope": "baseline-specific development evidence, not confirmed cross-model merit", **extra}
        _write_json(self.root/"status.json", snapshot)
        return snapshot

    def run(self, *, max_steps=None, idle_seconds=30):
        from .research_worker_runtime import BoundedFitExecutor
        with exploration_lock(self.root):
            self.queue.recover()
            with BoundedFitExecutor(max_workers=1, max_tasks_per_child=1) as executor:
                steps = 0
                while not (self.root/"STOP").exists() and (max_steps is None or steps < max_steps):
                    self.retry_tracking()
                    result = self.step(executor)
                    if result is None:
                        time.sleep(idle_seconds)
                    else:
                        steps += 1
                        print(json.dumps(result, allow_nan=False), flush=True)
            return self.status("stopped")
