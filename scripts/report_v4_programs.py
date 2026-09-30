"""Summarize matched development win trials without opening the protected holdout."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap_interval(
    candidate: pd.Series, baseline: pd.Series, *, repeats: int = 1000,
) -> tuple[float, float, float]:
    if repeats < 1 or not candidate.index.equals(baseline.index) or len(candidate) == 0:
        raise ValueError("Paired race IDs must be identical and nonempty")
    delta = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(delta), size=(repeats, len(delta)))
    means = delta[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def _winner_losses(path: Path) -> pd.Series:
    frame = pd.read_csv(
        path, usecols=["race_id", "target_win", "model_probability"]
    )
    winners = frame.loc[frame["target_win"].eq(1)]
    if winners["race_id"].duplicated().any() or winners["race_id"].nunique() != frame["race_id"].nunique():
        raise ValueError(f"Expected exactly one winner per scored race: {path}")
    if not np.isfinite(winners["model_probability"]).all():
        raise ValueError(f"Nonfinite winner probability: {path}")
    return pd.Series(
        -np.log(np.clip(winners["model_probability"].to_numpy(dtype=float), 1e-12, 1.0)),
        index=winners["race_id"].astype(str),
    ).sort_index()


def report(campaign: Path, output: Path, *, repeats: int = 1000) -> dict:
    ledger = sqlite3.connect(f"file:{campaign / 'ledger.sqlite'}?mode=ro", uri=True)
    rows = [
        (json.loads(payload), json.loads(result))
        for payload, result in ledger.execute(
            "SELECT payload_json, result_json FROM attempts "
            "WHERE status='completed' ORDER BY rowid"
        )
    ]
    win = [
        (payload, result) for payload, result in rows
        if result["target_kind"] == "win_probability"
        and result["objective_name"] == "development_fundamental_race_log_loss"
    ]
    if not win:
        raise ValueError("No completed fundamental win trials")
    baseline_payload, baseline_result = next(
        ((p, r) for p, r in win if p.get("experiment_id") == "B"), win[0]
    )
    baseline = _winner_losses(Path(baseline_result["artifacts"]["predictions"]))
    protocol_id = baseline_result["lineage"]["protocol_id"]
    dataset_hash = baseline_result["lineage"]["dataset_hash"]
    comparisons = []
    for payload, result in win:
        if result["lineage"]["protocol_id"] != protocol_id or result["lineage"]["dataset_hash"] != dataset_hash:
            raise ValueError("Campaign contains incomparable dataset or protocol identities")
        losses = _winner_losses(Path(result["artifacts"]["predictions"]))
        delta, low, high = paired_bootstrap_interval(losses, baseline, repeats=repeats)
        comparisons.append({
            "attempt_id": result["attempt_id"],
            "proposal_id": payload.get("proposal_id"),
            "experiment_id": payload.get("experiment_id"),
            "feature_program_id": result["lineage"].get("feature_program_id"),
            "model_kind": payload["recipe"]["model"]["kind"],
            "feature_schema": payload["recipe"]["feature_schema"],
            "transforms": payload["recipe"]["transforms"],
            "race_log_loss": result["objective_value"],
            "delta_vs_seed": delta,
            "delta_ci_95": [low, high],
        })
    comparisons.sort(key=lambda item: item["race_log_loss"])
    summary = {
        "status": "development_only_not_promotion_evidence",
        "baseline_attempt_id": baseline_result["attempt_id"],
        "baseline_race_log_loss": baseline_result["objective_value"],
        "race_count": len(baseline),
        "dataset_hash": dataset_hash,
        "protocol_id": protocol_id,
        "bootstrap_repeats": repeats,
        "comparisons": comparisons,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "development-progress.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# V4 Development Progress", "",
        f"Baseline: {baseline_result['attempt_id']} ({baseline_result['objective_value']:.6f})",
        f"Matched races: {len(baseline)}. Reused development folds; not a final test.", "",
        "| Attempt | Model | Feature program | Loss | Delta vs seed | 95% paired CI |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['attempt_id'][:20]} | {item['model_kind']} | "
            f"{item['feature_program_id']} | {item['race_log_loss']:.6f} | "
            f"{item['delta_vs_seed']:+.6f} | "
            f"[{item['delta_ci_95'][0]:+.6f}, {item['delta_ci_95'][1]:+.6f}] |"
        )
    (output / "development-progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    result = report(
        args.campaign, args.output or args.campaign / "reports",
        repeats=args.bootstrap_repeats,
    )
    print(json.dumps({
        "baseline": result["baseline_race_log_loss"],
        "best": result["comparisons"][0]["race_log_loss"],
        "races": result["race_count"],
        "win_trials": len(result["comparisons"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
