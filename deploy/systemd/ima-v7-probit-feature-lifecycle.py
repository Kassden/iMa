"""Sealed, additive own-user runtime/feature hooks over the qualified paid guard."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/home/imaopt/research-v2")
CAMPAIGN = ROOT/"campaigns/agentic_v7_openrouter_recovery_20261007"
PREDECESSOR = ROOT/"live-releases/ima-planner-guard-32582e5a7a5eeb6823ce6ed81f4801a8097c8da4/deploy/systemd/ima-v7-planner-request-guard.py"
PREDECESSOR_SHA = "d66333a431ebadcdc16dbbea7ecc08d8fb3c30903ed06db2aee5fdcaa29a4bda"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent, _, attribute = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], attribute, module)
    return module


def prepare_environment():
    if os.getuid() != 1001 or Path.home() != Path("/home/imaopt"):
        raise RuntimeError("Requires own imaopt user")
    stage = Path(__file__).resolve().parents[2]
    manifest = json.loads((stage/"lifecycle-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = stage/name
        if path.is_symlink() or not path.resolve().is_relative_to(stage) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Lifecycle release checksum mismatch: "+name)
    if hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() != PREDECESSOR_SHA:
        raise RuntimeError("Qualified predecessor changed")
    import ima
    # Load pinned controller first; new files are additive rather than a second repo.
    from ima import research_expansion
    if str(stage/"ima") not in ima.__path__:
        ima.__path__.insert(0, str(stage/"ima"))
    for module in ("probit_runtime_policy", "performance_probit", "research_worker_runtime",
                   "feature_lifecycle", "research_lifecycle"):
        if module in {"performance_probit", "research_worker_runtime"}:
            load("ima."+module, stage/"ima"/(module+".py"))
        else:
            __import__("ima."+module)
    return stage, manifest


def install_previous_with_hooks():
    stage, manifest = prepare_environment()
    previous = load("ima_lifecycle_predecessor", PREDECESSOR)
    original_load = previous.load

    def composed_load(name, path):
        module = original_load(name, path)
        if name == "ima.planner_request_guard":
            from ima.research_lifecycle import compact_optional_history, install_campaign_policy
            bound, install = module.bounded_context, module.install_guard

            def bounded(value, max_bytes=module.CONTEXT_BYTES):
                return bound(compact_optional_history(value), max_bytes)

            def combined(*args, **kwargs):
                receipt = install(*args, **kwargs)
                receipt["lifecycle"] = install_campaign_policy(CAMPAIGN,
                    release_sha256=hashlib.sha256((stage/"lifecycle-manifest.json").read_bytes()).hexdigest())
                return receipt

            module.bounded_context = bounded
            module.install_guard = combined
        return module

    previous.load = composed_load
    return previous, stage, manifest


def main():
    previous, stage, manifest = install_previous_with_hooks()
    self_test = "--lifecycle-self-test" in sys.argv
    if self_test:
        sys.argv = [sys.argv[0], "--guard-self-test"]
    result = previous.main()
    if self_test:
        from ima import research_expansion as expansion
        from ima.research_lifecycle import prove_local_preflight
        from ima.probit_runtime_policy import resolve_probit_budget
        from ima.research_resources import JobWorkload
        proof = prove_local_preflight(CAMPAIGN, "D000236")
        spend = expansion._planner_spend(CAMPAIGN)
        assert "D000236" not in spend["unresolved_decision_ids"]
        policy = resolve_probit_budget(JobWorkload(stage="fit", family="gaussian_probit", rows=100,
            generated_features=0, selected_features=3, categorical_cardinalities=(), folds=3,
            native_threads=1, native_thread_settings={}, search_settings={}, shared_artifacts={},
            implementation_revision="canary", dependency_versions={}), [], 1200)
        assert policy["initial_seconds"] == 14400
        print(json.dumps({"lifecycle_self_test": "passed", "policy": policy,
            "local_preflight_proof": proof, "spend": spend, "paid_calls": 0, "fits_cancelled": 0}))
    return result


if __name__ == "__mp_main__":
    # Spawned workers must import the same sealed implementation as their parent.
    stage, manifest = prepare_environment()
    from ima.research_lifecycle import install_campaign_policy
    install_campaign_policy(CAMPAIGN,
        release_sha256=hashlib.sha256((stage/"lifecycle-manifest.json").read_bytes()).hexdigest())

if __name__ == "__main__":
    raise SystemExit(main())
