"""Offline parent-CLI contracts; safety failures require fixes in the parent.

Only temporary fixture artifacts are written. Base model fitting is prohibited;
the three permitted calibration fits are mocked and their cohorts inspected.
"""

import copy
import hashlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scrapy.http import HtmlResponse

from ima.modeling import MarketBlend, TemperatureCalibrator
from ima.pools import SUPPORTED_POOLS, CombinationProbability, OrderExponents
from ima.season_evaluation import evaluate_season
from scripts import evaluate_season_2026 as cli
from scripts.collect_season_2026 import parse_dividends


DIVIDENDS = dict(zip(SUPPORTED_POOLS, [30, 20, 100, 40, 200, 300, 400, 500]))
VECTORS = {
    "benter_conditional_logit": [0.40, 0.25, 0.15, 0.08, 0.06, 0.04, 0.02],
    "boosted": [0.02, 0.04, 0.06, 0.08, 0.15, 0.25, 0.40],
    "gaussian_probit": [1 / 7] * 7,
    "pool": [0.06, 0.15, 0.04, 0.40, 0.02, 0.25, 0.08],
}


def race(day, number, *, reverse=False, card=False):
    finish = list(range(7, 0, -1)) if reverse else list(range(1, 8))
    combinations = {
        "WIN": [(finish[0],)], "PLACE": [(n,) for n in finish[:3]],
        "QIN": [tuple(finish[:2])], "QPL": list(itertools.combinations(finish[:3], 2)),
        "TRI": [tuple(finish[:3])], "TIERCE": [tuple(finish[:3])],
        "FIRST4": [tuple(finish[:4])], "QUARTET": [tuple(finish[:4])],
    }
    return {
        "race_date": day, "venue": "ST", "race_no": number,
        "race_class": "Class 4", "distance": 1200, "course": "TURF",
        "going": "GOOD", "prize": 1000000, "dead_heat": False,
        "runners": [{
            "horse_no": n, "horse_page_id": f"HK_2020_H{n}",
            "place": None if card else finish.index(n) + 1,
            "win_odds": None if card else float(n + 2),
            "jockey": f"J{n}", "trainer": f"T{n}", "draw": n,
            "actual_weight": 125, "weight": 125, "declared_weight": 1100,
            "rating": 60, "horse_rating": 60, "age": 4, "horse_age": 4,
            "finish_time": None if card else f"1:{9 + n:02d}.00",
            "source_cells": [str(n)],
        } for n in range(1, 8)],
        "dividends": {} if card else {
            ("TRIO" if pool == "TRI" else pool): {
                "status": "payable", "records": [{
                    "combination": list(combination), "dividend_hkd": DIVIDENDS[pool],
                    "dividend_per_hkd_10_decimal": str(DIVIDENDS[pool]),
                    "unit_stake_hkd": 10, "currency": "HKD", "status": "payable",
                } for combination in values],
            } for pool, values in combinations.items()
        },
    }


class SeasonReadinessCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.official = self.root / "official"
        self.output = self.root / "output"
        self.predictions = self.root / "predictions"
        self.official.mkdir()
        self.predictions.mkdir()
        self.history_path = self.root / "history.parquet"
        self.season = [race("2026-09-06", 1), race("2026-09-06", 2, reverse=True)]
        self.cards = [race("2026-10-07", 1, card=True), race("2026-10-07", 2, card=True)]
        self.write_json(self.official / "races.json", self.season)
        self.write_json(self.official / "racecards.json", self.cards)
        self.write_json(self.official / "coverage_manifest.json", {
            "results_complete": True, "racecards_complete": True, "dividends_complete": True,
        })
        history = cli.runner_rows([
            race("2025-01-01", 1), race("2026-08-26", 1), race("2026-08-31", 1, reverse=True),
            race("2026-09-01", 1), race("2026-10-08", 1),
        ])
        history.to_parquet(self.history_path, index=False)
        self.fits = []

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, allow_nan=False))

    def invoke(self, stage, *, prepare_training=False):
        argv = ["evaluate_season_2026", "--stage", stage,
                "--official-dir", str(self.official), "--output", str(self.output),
                "--calibration-races", "2"]
        argv += ["--history", str(self.history_path)] if stage == "prepare" else [
            "--predictions-dir", str(self.predictions),
        ]
        if prepare_training:
            argv.append("--prepare-training")
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            cli.main()

    @contextmanager
    def frozen_calibration(self):
        def fit_temperature(probabilities, frame):
            self.fits.append(("temperature", frame.copy(deep=True)))
            return TemperatureCalibrator(1.25)

        def fit_blend(fundamental, market, frame):
            self.fits.append(("blend", frame.copy(deep=True)))
            return MarketBlend(0.8, 0.2)

        def fit_order(frame, column):
            self.fits.append(("order", frame.copy(deep=True)))
            return OrderExponents(second=0.7, third=0.9)

        with ExitStack() as stack:
            for target in ["ima.modeling.RaceProbabilityModel.fit",
                           "ima.modeling.RaceConditionalLogitModel.fit"]:
                stack.enter_context(patch(target, side_effect=AssertionError("Base retraining forbidden")))
            stack.enter_context(patch.object(cli.TemperatureCalibrator, "fit", side_effect=fit_temperature))
            stack.enter_context(patch.object(cli.MarketBlend, "fit", side_effect=fit_blend))
            stack.enter_context(patch.object(cli, "fit_order_exponents", side_effect=fit_order))
            yield

    def prepare_predictions(self):
        with self.frozen_calibration():
            self.invoke("prepare")
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        for name, vector in VECTORS.items():
            frame = metadata[["race_id", "horse_no", "horse_id"]].copy()
            frame["model_probability"] = frame.horse_no.map(dict(enumerate(vector, 1)))
            frame.to_csv(self.predictions / f"{name}.csv", index=False)
        self.write_readback()
        return metadata

    def write_readback(self):
        receipt = {
            "input_sha256": hashlib.sha256((self.output / "query-features.parquet").read_bytes()).hexdigest(),
            "models": {path.stem: {"output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in self.predictions.glob("*.csv")},
        }
        self.write_json(self.predictions / "readback.json", receipt)

    def evaluate(self):
        with self.frozen_calibration():
            self.invoke("evaluate")
        return json.loads((self.output / "evaluation" / "summary.json").read_text())

    def ledger(self, name):
        return pd.read_csv(self.output / "evaluation" / f"{name}-tickets.csv",
                           dtype={"combination": str})

    def assert_quarantined(self, record, *, pool="WIN", require_refund=False):
        frame = cli.runner_rows([record]).assign(model_probability=VECTORS["benter_conditional_logit"])
        try:
            rows = cli.settlement_rows([record])
            report = evaluate_season(frame, rows, pools=[pool])
        except ValueError:
            return
        self.assertEqual(0, report.summary["nsettled"], "Unsupported settlement became payable")
        self.assertEqual(0, report.summary["stake_hkd"])
        if require_refund:
            self.assertIn("refund", report.ledger.status.iloc[0])

    def training_history(self):
        incomplete = race("2026-01-10", 1)
        incomplete["runners"][-1]["place"] = 6
        boundary_unpriced = race("2026-01-14", 2)
        for runner in boundary_unpriced["runners"]:
            runner["win_odds"] = None
        history = cli.runner_rows([
            race("2024-03-10", 1), race("2024-11-16", 1),
            race("2026-01-07", 1, reverse=True), race("2026-01-07", 2),
            incomplete, race("2026-01-14", 1), boundary_unpriced,
            race("2026-01-21", 1), race("2026-09-01", 1), race("2026-10-08", 1),
        ])
        history.to_parquet(self.history_path, index=False)
        return history

    def test_prepare_training_uses_actual_dates_strict_boundary_and_provenance(self):
        history = self.training_history()
        history_bytes = self.history_path.read_bytes()
        with self.frozen_calibration(), \
                patch.object(cli, "prepare_rich_runner_dataset", wraps=cli.prepare_rich_runner_dataset) as rich, \
                patch.object(cli, "build_speed_features", wraps=cli.build_speed_features) as speed:
            self.invoke("prepare", prepare_training=True)
        self.assertEqual([], self.fits)
        rich.assert_called_once()
        speed.assert_called_once()
        self.assertEqual({"strict_before_meeting": True}, rich.call_args.kwargs)
        boundary = pd.Timestamp("2026-01-14")
        self.assertTrue(rich.call_args.args[0].date.lt(boundary).all())
        self.assertEqual(35, len(rich.call_args.args[0]))
        training_path = self.output / "training-features.parquet"
        training = pd.read_parquet(training_path)
        expected_ids = set(history.loc[history.date.lt(boundary) & history.date.ne("2026-01-10"), "race_id"])
        self.assertEqual(expected_ids, set(training.race_id))
        self.assertEqual(28, len(training))
        self.assertTrue(training.date.lt(boundary).all())
        self.assertEqual(pd.Timestamp("2024-03-10"), training.date.min())
        self.assertEqual(pd.Timestamp("2026-01-07"), training.date.max())
        pd.testing.assert_frame_equal(speed.call_args.args[0][["race_id", "horse_no", "date"]],
                                      training[["race_id", "horse_no", "date"]])
        same_meeting = training[training.date.eq("2026-01-07") & training.horse_no.eq(1)]
        self.assertEqual([1, 1], same_meeting.jockey_last_result.tolist())
        self.assertEqual(1, same_meeting.past_race_speed_support.nunique())
        self.assertTrue(same_meeting.past_race_speed_support.gt(0).all())
        provenance = json.loads((self.output / "training-provenance.json").read_text())
        preparation = json.loads((self.output / "preparation.json").read_text())
        self.assertEqual(28, provenance["rows"])
        self.assertEqual(4, provenance["races"])
        self.assertEqual(training.date.max(), pd.Timestamp(provenance["actual_last_training_date"]))
        self.assertEqual(boundary, pd.Timestamp(provenance["exclusive_calibration_boundary"]))
        self.assertEqual(boundary, pd.Timestamp(preparation["calibration_first_date"]))
        self.assertEqual(hashlib.sha256(training_path.read_bytes()).hexdigest(), provenance["input_sha256"])
        self.assertEqual(hashlib.sha256(history_bytes).hexdigest(), preparation["history_sha256"])
        expected_missing = training.select_dtypes(include="number").isna().mean().to_dict()
        self.assertEqual(expected_missing, provenance["numeric_missing_fraction"])
        self.assertIn("strict prior-meeting", provenance["policy"])
        self.assertEqual(history_bytes, self.history_path.read_bytes())
        self.assertFalse((self.output / "models").exists())
        self.assertFalse((self.output / "evaluation").exists())
        self.assertEqual([], list(self.predictions.iterdir()))

    def test_prepare_training_ignores_calibration_and_future_outcome_changes(self):
        history = self.training_history()
        with self.frozen_calibration():
            self.invoke("prepare", prepare_training=True)
        training_path = self.output / "training-features.parquet"
        original = pd.read_parquet(training_path)
        original_provenance = json.loads((self.output / "training-provenance.json").read_text())
        later = history.date.ge("2026-01-14")
        history.loc[later, "result"] = 8 - history.loc[later, "result"]
        history.loc[later, "finish_time"] = "9:59.99"
        history.loc[later, "horse_rating"] = 999
        history.to_parquet(self.history_path, index=False)
        changed_season = copy.deepcopy(self.season)
        for record in changed_season:
            for runner in record["runners"]:
                runner["place"] = 8 - runner["place"]
                runner["horse_rating"] = 888
        self.write_json(self.official / "races.json", changed_season)
        with self.frozen_calibration():
            self.invoke("prepare", prepare_training=True)
        pd.testing.assert_frame_equal(original, pd.read_parquet(training_path))
        self.assertEqual(original_provenance,
                         json.loads((self.output / "training-provenance.json").read_text()))
        self.assertEqual([], self.fits)

    def test_cli_prepare_and_four_frozen_vectors_end_to_end(self):
        metadata = self.prepare_predictions()
        self.assertFalse((self.output / "training-features.parquet").exists())
        self.assertFalse((self.output / "training-provenance.json").exists())
        self.assertEqual({"calibration": 2, "season": 2, "tomorrow": 2},
                         metadata.groupby("cohort").race_id.nunique().to_dict())
        calibration = metadata[metadata.cohort.eq("calibration")]
        self.assertTrue(calibration.date.lt("2026-09-01").all())
        self.assertEqual({"2026-08-26", "2026-08-31"}, set(calibration.date.dt.strftime("%Y-%m-%d")))
        features = pd.read_parquet(self.output / "query-features.parquet")
        self.assertEqual(42, len(features))
        self.assertTrue(features[["result", "win_odds", "target_win"]].isna().all().all())
        (self.output / "models").mkdir()
        self.write_json(self.output / "models" / "frozen-candidates.json", {"candidates": list(VECTORS)})
        summaries = self.evaluate()
        self.assertEqual(10, len(summaries))
        self.assertTrue(self.fits)
        for kind, frame in self.fits:
            self.assertTrue(frame.date.lt("2026-09-01").all(), kind)
            self.assertEqual(2, frame.race_id.nunique(), kind)
            self.assertEqual({"calibration"}, set(frame.cohort), kind)
        for name, summary in summaries.items():
            ledger = self.ledger(name)
            self.assertEqual(16, len(ledger), name)
            self.assertEqual([80, 80], ledger.groupby("race_id").stake_hkd.sum().tolist(), name)
            self.assertEqual(160, summary["stake_hkd"], name)
            self.assertEqual(16, summary["nsettled"], name)
            self.assertTrue(ledger.estimated_ev_hkd.isna().all(), name)
            self.assertEqual(0, summary["ev_available_tickets"], name)
            published = {(row["race_id"], row["pool"], "/".join(map(str, sorted(row["winning_combination"])
                         if row["pool"] in {"QIN", "QPL", "TRI", "FIRST4"}
                         else row["winning_combination"]))): row["dividend_hkd_per_10"]
                         for row in cli.settlement_rows(self.season)}
            expected = [published.get((row.race_id, row.pool, row.combination), 0)
                        for row in ledger.itertuples()]
            self.assertEqual(expected, ledger.realized_gross_hkd.tolist(), name)
            self.assertEqual(sum(expected), summary["gross_hkd"], name)
            self.assertEqual(sum(expected) - 160, summary["profit_hkd"], name)
            self.assertEqual((sum(expected) - 160) / 160, summary["roi"], name)
            meetings = pd.read_csv(self.output / "evaluation" / f"{name}-meetings.csv")
            self.assertEqual(1, len(meetings), name)
            self.assertEqual(160, meetings.stake_hkd.sum(), name)
            self.assertAlmostEqual(sum(expected), meetings.realized_gross_hkd.sum(), msg=name)
            np.testing.assert_allclose(summary["meeting_cluster_bootstrap_roi_interval_95"],
                                       [summary["roi"], summary["roi"]])
        self.assertEqual(1590, summaries["benter_conditional_logit"]["gross_hkd"])
        self.assertEqual(1430, summaries["benter_conditional_logit"]["profit_hkd"])
        tomorrow = pd.read_csv(self.output / "evaluation" / "tomorrow-combinations.csv")
        self.assertEqual(set(VECTORS), set(tomorrow.model))
        self.assertTrue(tomorrow.quoted_ev_hkd.isna().all())
        self.assertEqual({10}, set(tomorrow.stake_hkd_per_combination))
        for name in VECTORS:
            summary = summaries[name]
            self.assertIn("Not actionable", summary["hindsight_win_price_note"])
            self.assertIn("pre-race", summary["price_basis"])
        for name in summaries.keys() - VECTORS.keys():
            self.assertIn("hindsight", name)

    def test_calibration_quality_keeps_candidates_but_fits_only_whole_rated_fields(self):
        metadata = self.prepare_predictions()
        eligible = "HKJC:2026-08-26:ST:R1"
        excluded = "HKJC:2026-08-31:ST:R1"
        for invalid in (np.nan, np.inf, -np.inf, "not-a-rating"):
            with self.subTest(invalid=invalid):
                changed = metadata.copy(deep=True)
                changed["horse_rating"] = changed.horse_rating.astype(object)
                changed.loc[changed.race_id.eq(excluded) & changed.horse_no.eq(7), "horse_rating"] = invalid
                changed.loc[changed.race_id.eq(eligible) & changed.horse_no.eq(1), "horse_rating"] = 0
                changed.loc[changed.race_id.eq(eligible) & changed.horse_no.eq(2), "horse_rating"] = -10
                if isinstance(invalid, str):
                    changed["horse_rating"] = changed.horse_rating.astype(str)
                changed.to_parquet(self.output / "query-metadata.parquet", index=False)
                self.fits.clear()
                summaries = self.evaluate()
                for kind, fitted in self.fits:
                    self.assertEqual({eligible}, set(fitted.race_id), kind)
                    self.assertEqual(set(range(1, 8)), set(fitted.horse_no), kind)
                    self.assertTrue(fitted.date.lt("2026-09-01").all(), kind)
                    self.assertEqual({"calibration"}, set(fitted.cohort), kind)
                    self.assertTrue(set(fitted.race_id).isdisjoint(set(changed.loc[changed.cohort.ne("calibration"), "race_id"])))
                self.assertTrue(self.fits)
                saved = pd.read_parquet(self.output / "query-metadata.parquet")
                self.assertEqual(2, saved[saved.cohort.eq("calibration")].race_id.nunique())
                self.assertEqual(42, len(saved))
                self.assertTrue(all(summary["nraces"] == 2 for summary in summaries.values()))
                calibration = json.loads((self.output / "evaluation" / "calibration.json").read_text())
                for value in calibration.values():
                    self.assertEqual(1, value["calibration_races"])
                    self.assertIn("whole fields", value["calibration_quality_policy"])
                    self.assertIn("no outcome or score", value["calibration_quality_policy"])

    def test_calibration_quality_population_independent_of_target_outcomes(self):
        metadata = self.prepare_predictions()
        excluded = metadata.race_id.eq("HKJC:2026-08-31:ST:R1") & metadata.horse_no.eq(7)
        metadata.loc[excluded, "horse_rating"] = np.nan
        metadata.to_parquet(self.output / "query-metadata.parquet", index=False)
        self.evaluate()
        original_fits = [(kind, frame.copy(deep=True)) for kind, frame in self.fits]
        calibration = metadata.cohort.eq("calibration")
        metadata.loc[calibration, "result"] = 8 - metadata.loc[calibration, "result"]
        metadata.to_parquet(self.output / "query-metadata.parquet", index=False)
        self.fits.clear()
        self.evaluate()
        self.assertEqual(len(original_fits), len(self.fits))
        for (kind, before), (other_kind, after) in zip(original_fits, self.fits):
            self.assertEqual(kind, other_kind)
            pd.testing.assert_frame_equal(before[["race_id", "horse_no", "horse_id", "horse_rating"]],
                                          after[["race_id", "horse_no", "horse_id", "horse_rating"]])
            self.assertFalse(before.target_win.equals(after.target_win))
            self.assertEqual([7], after.loc[after.target_win.eq(1), "horse_no"].tolist())

    def test_calibration_without_any_complete_rated_field_rejected_before_fit(self):
        metadata = self.prepare_predictions()
        metadata.loc[metadata.cohort.eq("calibration") & metadata.horse_no.eq(7), "horse_rating"] = np.nan
        metadata.to_parquet(self.output / "query-metadata.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "No complete pre-season calibration fields"):
            self.evaluate()
        self.assertEqual([], self.fits)

    def test_top_pick_metrics_follow_win_tickets_for_ties_and_permuted_outcome_rows(self):
        for record in self.season:
            for runner in record["runners"]:
                if runner["horse_no"] == 1:
                    runner["horse_no"] = 10
                runner["win_odds"] = 3 if runner["horse_no"] in {2, 10} else 100
            for pool in record["dividends"].values():
                for settlement in pool["records"]:
                    settlement["combination"] = [10 if n == 1 else n for n in settlement["combination"]]
        self.write_json(self.official / "races.json", self.season)
        metadata = self.prepare_predictions()
        ordered = metadata.sort_values(["race_id", "result"], kind="stable", na_position="last").reset_index(drop=True)
        first_race = ordered[ordered.race_id.eq("HKJC:2026-09-06:ST:R1")]
        self.assertEqual([10, 2], first_race.horse_no.iloc[:2].tolist())
        for noise in (0, 4e-14):
            with self.subTest(noise=noise):
                for name in VECTORS:
                    path = self.predictions / f"{name}.csv"
                    predictions = pd.read_csv(path)
                    season = predictions.race_id.str.contains("2026-09-06", regex=False)
                    predictions.loc[season, "model_probability"] = predictions.loc[season, "horse_no"].map({
                        10: 0.3 + noise, 2: 0.3 - noise, 3: 0.08, 4: 0.08,
                        5: 0.08, 6: 0.08, 7: 0.08,
                    })
                    predictions.to_csv(path, index=False)
                self.write_readback()
                original_metrics = {}
                original_picks = {}
                for permutation, rows in enumerate((ordered, ordered.iloc[::-1].reset_index(drop=True))):
                    rows.to_parquet(self.output / "query-metadata.parquet", index=False)
                    summaries = self.evaluate()
                    for name, summary in summaries.items():
                        win = self.ledger(name)
                        win = win[win.pool.eq("WIN")]
                        self.assertEqual({"2"}, set(win.combination), name)
                        self.assertEqual(summary["by_pool"]["WIN"]["hit_rate"], summary["top_pick_win_hit_rate"], name)
                        self.assertEqual(0, summary["top_pick_win_hit_rate"], name)
                        runners = pd.read_csv(self.output / "evaluation" / f"{name}-runners.csv")
                        picks = win.assign(horse_no=win.combination.astype(int)).merge(
                            runners[["race_id", "horse_no", "result"]], on=["race_id", "horse_no"],
                            validate="one_to_one",
                        )
                        self.assertEqual(0.5, summary["top_pick_top3_hit_rate"], name)
                        self.assertEqual(float(picks.result.le(3).mean()), summary["top_pick_top3_hit_rate"], name)
                        metrics = (summary["top_pick_win_hit_rate"], summary["top_pick_top3_hit_rate"],
                                   summary["by_pool"]["WIN"]["hit_rate"])
                        tickets = win.set_index("race_id").combination.to_dict()
                        if permutation == 0:
                            original_metrics[name], original_picks[name] = metrics, tickets
                        else:
                            self.assertEqual(original_metrics[name], metrics, name)
                            self.assertEqual(original_picks[name], tickets, name)

    def test_fundamental_allocation_ignores_actual_results_and_final_price(self):
        metadata = self.prepare_predictions()
        self.evaluate()
        columns = ["race_id", "pool", "rank", "combination", "stake_hkd", "probability"]
        before = {name: self.ledger(name)[columns] for name in VECTORS}
        mask = metadata.cohort.eq("season")
        metadata.loc[mask, "result"] = 8 - metadata.loc[mask, "result"]
        metadata.loc[mask, "win_odds"] = 1e8
        metadata.to_parquet(self.output / "query-metadata.parquet", index=False)
        self.season = [race("2026-09-06", 1, reverse=True), race("2026-09-06", 2)]
        self.write_json(self.official / "races.json", self.season)
        self.evaluate()
        for name in VECTORS:
            pd.testing.assert_frame_equal(before[name], self.ledger(name)[columns])

    def test_daily_totals_and_seeded_meeting_bootstrap_use_stake_weighted_clusters(self):
        self.season = [race("2026-09-06", 1), race("2026-09-09", 1, reverse=True),
                       race("2026-09-13", 1), race("2026-09-16", 1, reverse=True)]
        self.season[0]["dividends"]["QPL"] = {"status": "unavailable", "records": []}
        self.season[2]["dividends"] = {"WIN": self.season[2]["dividends"]["WIN"]}
        self.write_json(self.official / "races.json", self.season)
        self.prepare_predictions()
        summaries = self.evaluate()
        name = "benter_conditional_logit"
        meetings = pd.read_csv(self.output / "evaluation" / f"{name}-meetings.csv")
        self.assertEqual([70, 80, 10, 80], meetings.stake_hkd.tolist())
        self.assertEqual([1550, 0, 30, 0], meetings.realized_gross_hkd.tolist())
        self.assertEqual([1480, -80, 20, -80], meetings.realized_net_hkd.tolist())
        summary = summaries[name]
        self.assertEqual(240, summary["stake_hkd"])
        self.assertEqual(1580, summary["gross_hkd"])
        self.assertAlmostEqual(1340 / 240, summary["roi"])
        profits = np.array([1480, -80, 20, -80], dtype=float)
        stakes = np.array([70, 80, 10, 80], dtype=float)
        draws = np.random.default_rng(20261007).integers(0, 4, size=(10000, 4))
        expected = np.quantile(profits[draws].sum(axis=1) / stakes[draws].sum(axis=1), [0.025, 0.975])
        wrong_unweighted = np.quantile((profits / stakes)[draws].mean(axis=1), [0.025, 0.975])
        self.assertFalse(np.allclose(expected, wrong_unweighted))
        np.testing.assert_allclose(expected, summary["meeting_cluster_bootstrap_roi_interval_95"])
        self.assertIn("Exploratory", summary["uncertainty_note"])
        repeated = self.evaluate()
        self.assertEqual(summary["meeting_cluster_bootstrap_roi_interval_95"],
                         repeated[name]["meeting_cluster_bootstrap_roi_interval_95"])

    def test_converter_alias_and_explicit_dead_heat(self):
        rows = cli.settlement_rows(self.season)
        self.assertIn("TRI", {row["pool"] for row in rows})
        self.assertNotIn("TRIO", {row["pool"] for row in rows})
        tied = copy.deepcopy(self.season[0])
        tied["dead_heat"] = True
        tied["runners"][1]["place"] = 1
        self.assert_quarantined(tied)

    def test_lower_sixth_place_dead_heat_is_retained_by_preparation_and_settlement(self):
        tied = copy.deepcopy(self.season[0])
        tied["dead_heat"] = True
        tied["runners"][6]["place"] = 6
        self.write_json(self.official / "races.json", [tied, self.season[1]])
        with self.frozen_calibration():
            self.invoke("prepare")
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        self.assertEqual(2, metadata.loc[metadata.cohort.eq("season"), "race_id"].nunique())
        frame = cli.runner_rows([tied]).assign(model_probability=VECTORS["benter_conditional_logit"])
        report = evaluate_season(frame, cli.settlement_rows([tied]))
        self.assertEqual(8, report.summary["nsettled"])
        self.assertEqual(80, report.summary["stake_hkd"])

    def test_known_pulled_up_last_starter_retained_without_fabricating_rank(self):
        record = copy.deepcopy(self.season[0])
        record["runners"][6].update(place=None, finishing_status="PU")
        self.write_json(self.official / "races.json", [record, self.season[1]])
        with self.frozen_calibration():
            self.invoke("prepare")
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        source = metadata[(metadata.cohort.eq("season")) & metadata.race_no.eq(1)]
        self.assertEqual(7, len(source))
        self.assertTrue(source.loc[source.horse_no.eq(7), "result"].isna().all())
        report = evaluate_season(source.assign(model_probability=VECTORS["benter_conditional_logit"]),
                                 cli.settlement_rows([record]))
        self.assertEqual(8, report.summary["nsettled"])

    def test_unknown_missing_last_position_is_quarantined_by_preparation(self):
        record = copy.deepcopy(self.season[0])
        record["runners"][6].update(place=None, finishing_status="UNKNOWN")
        self.write_json(self.official / "races.json", [record, self.season[1]])
        with self.frozen_calibration():
            self.invoke("prepare")
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        self.assertEqual(1, metadata.loc[metadata.cohort.eq("season"), "race_id"].nunique())

    def test_converter_preserves_record_refund_quarantine(self):
        refunded = copy.deepcopy(self.season[0])
        refunded["dividends"]["WIN"]["records"] = [{
            "combination": [], "dividend_hkd": None, "status": "refund",
            "currency": "HKD", "unit_stake_hkd": 10,
        }]
        self.assert_quarantined(refunded, require_refund=True)

    def test_missing_pool_is_excluded_without_counting_loss(self):
        missing = copy.deepcopy(self.season[0])
        missing["dividends"]["QPL"] = {"status": "unavailable", "records": []}
        frame = cli.runner_rows([missing]).assign(model_probability=VECTORS["benter_conditional_logit"])
        report = evaluate_season(frame, cli.settlement_rows([missing]))
        self.assertEqual(70, report.summary["stake_hkd"])
        self.assertEqual(1, report.summary["nmissing"])
        self.assertEqual(8, len(report.ledger))
        self.assertTrue(report.ledger.loc[report.ledger.pool.eq("QPL"), "realized_gross_hkd"].isna().all())

    def test_explicit_one_dollar_dividend_normalized_to_hkd10(self):
        html = b"<table><tr><th>Pool</th><th>Winning Combination</th><th>Dividend (HK$)</th></tr><tr><td>QUARTET</td><td>1,2,3,4</td><td>349,150.00/$1.0</td></tr></table>"
        pools, _ = parse_dividends(HtmlResponse(url="https://racing.hkjc.com/fixture", body=html, encoding="utf-8"))
        record = copy.deepcopy(self.season[0])
        record["dividends"] = {"QUARTET": pools["QUARTET"]}
        converted = cli.settlement_rows([record])
        self.assertEqual(3491500, converted[0]["dividend_hkd_per_10"])
        frame = cli.runner_rows([record]).assign(model_probability=VECTORS["benter_conditional_logit"])
        report = evaluate_season(frame, converted, pools=["QUARTET"])
        self.assertEqual(10, report.summary["stake_hkd"])
        self.assertEqual(3491500, report.summary["gross_hkd"])
        self.assertEqual(3491490, report.summary["profit_hkd"])

    def test_unknown_record_status_cannot_settle_as_payable(self):
        record = copy.deepcopy(self.season[0])
        record["dividends"]["WIN"]["records"][0]["status"] = "pending_review"
        self.assert_quarantined(record)

    def test_pool_refund_status_overrides_payable_looking_child(self):
        record = copy.deepcopy(self.season[0])
        record["dividends"]["WIN"]["status"] = "refund"
        self.assert_quarantined(record, require_refund=True)

    def test_pool_refund_without_child_records_is_explicitly_quarantined(self):
        record = copy.deepcopy(self.season[0])
        record["dividends"]["WIN"] = {"status": "refund", "records": []}
        self.assert_quarantined(record, require_refund=True)

    def test_unknown_pool_status_cannot_settle_as_payable(self):
        record = copy.deepcopy(self.season[0])
        record["dividends"]["WIN"]["status"] = "pending_review"
        self.assert_quarantined(record)

    def test_non_hkd_currency_cannot_settle_as_hkd(self):
        record = copy.deepcopy(self.season[0])
        record["dividends"]["WIN"]["records"][0]["currency"] = "USD"
        self.assert_quarantined(record)

    def test_unknown_unit_cannot_be_assumed_hkd10(self):
        record = copy.deepcopy(self.season[0])
        child = record["dividends"]["WIN"]["records"][0]
        child.pop("unit_stake_hkd")
        child.pop("dividend_per_hkd_10_decimal")
        self.assert_quarantined(record)

    def test_unsupported_pool_is_rejected_or_quarantined(self):
        record = copy.deepcopy(self.season[0])
        record["dividends"] = {"DOUBLE": {"status": "payable", "records": [{
            "combination": [1, 2], "dividend_hkd": 100,
            "unit_stake_hkd": 10, "currency": "HKD", "status": "payable",
        }]}}
        self.assert_quarantined(record)

    def test_evaluate_rejects_calibration_dated_in_september_before_any_fit(self):
        metadata = self.prepare_predictions()
        metadata.loc[metadata.cohort.eq("calibration"), "date"] = pd.Timestamp("2026-09-01")
        metadata.to_parquet(self.output / "query-metadata.parquet", index=False)
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_missing_or_duplicate_prediction_keys_rejected_before_calibration(self):
        self.prepare_predictions()
        path = self.predictions / "boosted.csv"
        original = pd.read_csv(path)
        for broken in [original.iloc[:-1], pd.concat([original, original.iloc[:1]], ignore_index=True)]:
            broken.to_csv(path, index=False)
            self.write_readback()
            with self.assertRaises(ValueError):
                self.evaluate()
        self.assertFalse(self.fits)

    def test_wrong_horse_identity_rejected_before_calibration(self):
        self.prepare_predictions()
        path = self.predictions / "boosted.csv"
        predictions = pd.read_csv(path)
        predictions.loc[0, "horse_id"] = "HK_2020_WRONG_HORSE"
        predictions.to_csv(path, index=False)
        self.write_readback()
        with self.assertRaisesRegex(ValueError, "runner populations"):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_stale_feature_dataset_rejected_before_any_fit(self):
        self.prepare_predictions()
        path = self.output / "query-features.parquet"
        features = pd.read_parquet(path)
        features.loc[0, "horse_rating"] = 999
        features.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_changed_prediction_hash_rejected_before_any_fit(self):
        self.prepare_predictions()
        path = self.predictions / "boosted.csv"
        predictions = pd.read_csv(path)
        predictions.loc[0, "model_probability"] = 0.9
        predictions.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "hash"):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_missing_prediction_csv_rejected_before_any_fit(self):
        self.prepare_predictions()
        (self.predictions / "boosted.csv").unlink()
        with self.assertRaisesRegex(ValueError, "candidate population"):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_missing_readback_rejected_before_any_fit(self):
        self.prepare_predictions()
        (self.predictions / "readback.json").unlink()
        with self.assertRaisesRegex(ValueError, "readback"):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_manifest_candidate_mismatch_rejected_even_with_matching_receipt(self):
        self.prepare_predictions()
        (self.output / "models").mkdir()
        self.write_json(self.output / "models" / "frozen-candidates.json", {
            "candidates": ["boosted", "pool"],
        })
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            self.evaluate()
        self.assertFalse(self.fits)

    def test_optional_history_backfill_and_form_files_can_be_absent(self):
        history = pd.read_parquet(self.history_path)
        original = history.copy(deep=True)
        pd.testing.assert_frame_equal(history, cli.enrich_official_history(history, self.official))
        pd.testing.assert_frame_equal(history, cli.enrich_form_inputs(history, self.official))
        pd.testing.assert_frame_equal(original, history)

    def write_flat_enrichment(self, records, *, complete=True):
        self.write_json(self.official / "season-runner-enrichment.json", records)
        self.write_json(self.official / "season-enrichment-coverage.json", {"complete": complete})

    @staticmethod
    def flat_enrichment_record(**overrides):
        return {
            "race_date": "2026-09-06", "venue": "ST", "race_no": 1,
            "horse_page_id": "HK_2020_H1", "horse_no": 1,
            "rating": 71, "age": 5, "country": "NZ", "gear": "B",
            "source": {"source_url": "https://racing.hkjc.com/archived-declaration.pdf"},
            **overrides,
        }

    def test_flat_enrichment_exact_identity_aliases_and_preserves_known_values(self):
        frame = cli.runner_rows([self.season[0]])
        frame.loc[[0, 2], ["horse_rating", "horse_age"]] = np.nan
        frame["horse_country"] = None
        frame["gear"] = None
        frame.loc[1, "horse_country"] = "IRE"
        frame.loc[1, "gear"] = "TT"
        original = frame.copy(deep=True)
        records = [self.flat_enrichment_record(), self.flat_enrichment_record(
            horse_page_id="HK_2020_H2", horse_no=2, rating=999, age=99, country="AUS", gear="WRONG")]
        for overrides in [
            {"race_date": "2026-09-07"}, {"venue": "HV"}, {"race_no": 2},
            {"horse_page_id": "HK_2021_H3"},
        ]:
            records.append(self.flat_enrichment_record(**{
                "horse_page_id": "HK_2020_H3", "horse_no": 3, **overrides,
            }))
        self.write_flat_enrichment(records)
        enriched = cli.enrich_archived_cards(frame, self.official)
        self.assertEqual([71, 5, "NZ", "B"],
                         enriched.loc[0, ["horse_rating", "horse_age", "horse_country", "gear"]].tolist())
        self.assertEqual([60, 4, "IRE", "TT"],
                         enriched.loc[1, ["horse_rating", "horse_age", "horse_country", "gear"]].tolist())
        self.assertTrue(enriched.loc[2, ["horse_rating", "horse_age", "horse_country", "gear"]].isna().all())
        for name in ("horse_rating", "horse_age", "horse_country", "gear"):
            self.assertEqual(records[0]["source"]["source_url"], enriched.loc[0, name + "_source"])
        pd.testing.assert_frame_equal(original, frame)
        pd.testing.assert_frame_equal(original.drop(columns=["horse_rating", "horse_age", "horse_country", "gear"]),
                                      enriched[original.columns].drop(columns=["horse_rating", "horse_age", "horse_country", "gear"]))

    def test_flat_enrichment_number_mismatch_and_duplicate_identity_rejected(self):
        frame = cli.runner_rows([self.season[0]])
        self.write_flat_enrichment([self.flat_enrichment_record(horse_no=2)])
        with self.assertRaisesRegex(ValueError, "horse number mismatch"):
            cli.enrich_archived_cards(frame, self.official)
        first = self.flat_enrichment_record()
        second = self.flat_enrichment_record(rating=99)
        for records in [[first, second], [second, first]]:
            self.write_flat_enrichment(records)
            with self.assertRaisesRegex(ValueError, "Duplicate archived declaration identity"):
                cli.enrich_archived_cards(frame, self.official)

    def test_flat_enrichment_incomplete_coverage_rejected_before_preparation(self):
        self.write_flat_enrichment([self.flat_enrichment_record()], complete=False)
        frame = cli.runner_rows([self.season[0]])
        with self.assertRaisesRegex(ValueError, "enrichment is incomplete"):
            cli.enrich_archived_cards(frame, self.official)
        with self.frozen_calibration(), self.assertRaisesRegex(ValueError, "enrichment is incomplete"):
            self.invoke("prepare")
        self.assertEqual([], self.fits)
        self.assertFalse(self.output.exists())

    def test_partial_calibration_enrichment_requires_finished_acquisition(self):
        frame = cli.runner_rows([self.season[0]])
        frame.loc[0, "horse_rating"] = np.nan
        path = self.official / "calibration-runner-enrichment.json"
        self.write_json(path, [self.flat_enrichment_record()])
        coverage_path = self.official / "calibration-enrichment-coverage.json"
        for coverage in ({"complete": False}, {"complete": False, "acquisition_finished": False}):
            self.write_json(coverage_path, coverage)
            with self.assertRaisesRegex(ValueError, "enrichment is incomplete"):
                cli.enrich_archived_cards(frame, self.official)
        self.write_json(coverage_path, {"complete": False, "acquisition_finished": True})
        enriched = cli.enrich_archived_cards(frame, self.official)
        self.assertEqual(71, enriched.loc[0, "horse_rating"])
        self.assertTrue(enriched.loc[0, "horse_rating_source"].endswith("archived-declaration.pdf"))
        self.write_flat_enrichment([], complete=False)
        self.write_json(self.official / "season-enrichment-coverage.json", {
            "complete": False, "acquisition_finished": True,
        })
        with self.assertRaisesRegex(ValueError, "enrichment is incomplete"):
            cli.enrich_archived_cards(frame, self.official)

    def test_flat_enrichment_reaches_prepared_metadata_and_query_features(self):
        for name in ("horse_rating", "rating", "horse_age", "age"):
            self.season[0]["runners"][0][name] = None
        self.write_json(self.official / "races.json", self.season)
        record = self.flat_enrichment_record(horse_rating=72, horse_age=6, horse_country="GB")
        self.write_flat_enrichment([record])
        with self.frozen_calibration(), \
                patch.object(cli, "prepare_rich_runner_dataset", side_effect=AssertionError("Training rebuild forbidden")):
            self.invoke("prepare")
        for filename in ("query-metadata.parquet", "query-features.parquet"):
            prepared = pd.read_parquet(self.output / filename)
            row = prepared[prepared.race_id.eq("HKJC:2026-09-06:ST:R1") & prepared.horse_no.eq(1)].iloc[0]
            self.assertEqual([72, 6, "GB"], row[["horse_rating", "horse_age", "horse_country"]].tolist())
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        row = metadata[metadata.race_id.eq("HKJC:2026-09-06:ST:R1") & metadata.horse_no.eq(1)].iloc[0]
        self.assertEqual(record["source"]["source_url"], row.horse_country_source)
        self.assertEqual([], self.fits)
        self.assertFalse((self.output / "training-features.parquet").exists())

    def test_past_country_requires_strictly_earlier_full_horse_identity(self):
        frame = pd.DataFrame({
            "horse_id": ["HK_2020_H1", "HK_2020_H2", "HK_2020_H3", "HK_2020_H4",
                         "HK_2020_H5", "HK_2020_H6", "HK_2020_H7"],
            "date": pd.to_datetime(["2026-10-07"] * 7),
            "horse_country": [None, None, None, None, "UNKNOWN", None, "IRE"],
            "horse_country_source": [None] * 6 + ["official:current-racecard"],
        }, index=[11, 22, 33, 44, 55, 66, 77])
        history = pd.DataFrame({
            "horse_id": ["HK_2020_H1", "HK_2020_H1", "HK_2020_H2", "HK_2020_H3",
                         "HK_2021_H4", "HK_2020_H5", "HK_2020_H6", "HK_2020_H7"],
            "date": pd.to_datetime(["2026-10-06", "2026-09-01", "2026-10-07", "2026-10-08",
                                    "2026-09-01", "2026-09-01", "2026-09-01", "2026-09-01"]),
            "horse_country": ["NZ", "NZ", "AUS", "GB", "FR", "AUS", "UNKNOWN", "NZ"],
        })
        original_frame, original_history = frame.copy(deep=True), history.copy(deep=True)
        enriched = cli.enrich_past_country(frame, history)
        self.assertEqual("NZ", enriched.loc[11, "horse_country"])
        self.assertEqual("AUS", enriched.loc[55, "horse_country"])
        self.assertTrue(enriched.loc[[22, 33, 44, 66], "horse_country"].isna().all())
        self.assertEqual("IRE", enriched.loc[77, "horse_country"])
        self.assertEqual("official:current-racecard", enriched.loc[77, "horse_country_source"])
        self.assertEqual({"official:strict-prior-immutable-country"},
                         set(enriched.loc[[11, 55], "horse_country_source"]))
        pd.testing.assert_frame_equal(frame, original_frame)
        pd.testing.assert_frame_equal(history, original_history)
        pd.testing.assert_frame_equal(enriched, cli.enrich_past_country(frame, history.iloc[::-1]))
        without_column = cli.enrich_past_country(frame.drop(columns="horse_country"), history)
        self.assertEqual("NZ", without_column.loc[11, "horse_country"])
        self.assertTrue(without_column.loc[[22, 33, 44, 66], "horse_country"].isna().all())

    def test_conflicting_immutable_country_evidence_rejected_in_both_orders(self):
        frame = pd.DataFrame({"horse_id": ["HK_2020_H1"],
                              "date": pd.to_datetime(["2026-10-07"]), "horse_country": [None]})
        history = pd.DataFrame({"horse_id": ["HK_2020_H1"] * 2,
                                "date": pd.to_datetime(["2026-01-01", "2026-10-08"]),
                                "horse_country": ["NZ", "AUS"]})
        for ordered in [history, history.iloc[::-1]]:
            with self.assertRaisesRegex(ValueError, "Conflicting immutable horse-country"):
                cli.enrich_past_country(frame, ordered)

    def test_prepare_country_enrichment_preserves_metadata_without_training(self):
        history = pd.read_parquet(self.history_path)
        history["horse_country"] = None
        history.loc[history.horse_id.eq("HK_2020_H1") & history.date.eq("2025-01-01"), "horse_country"] = "NZ"
        history.loc[history.horse_id.eq("HK_2020_H2") & history.date.eq("2026-10-08"), "horse_country"] = "AUS"
        history.to_parquet(self.history_path, index=False)
        for card in self.cards:
            card["runners"][0]["horse_country"] = "IRE"
        self.write_json(self.official / "racecards.json", self.cards)
        with self.frozen_calibration(), \
                patch.object(cli, "prepare_rich_runner_dataset", side_effect=AssertionError("Training rebuild forbidden")), \
                patch.object(cli, "build_speed_features", side_effect=AssertionError("Training rebuild forbidden")):
            self.invoke("prepare")
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        tomorrow = metadata[metadata.cohort.eq("tomorrow")]
        self.assertEqual({"IRE"}, set(tomorrow.loc[tomorrow.horse_no.eq(1), "horse_country"]))
        self.assertTrue(tomorrow.loc[tomorrow.horse_no.eq(2), "horse_country"].isna().all())
        season = metadata[metadata.cohort.eq("season")]
        self.assertEqual({"NZ"}, set(season.loc[season.horse_no.eq(1), "horse_country"]))
        features = pd.read_parquet(self.output / "query-features.parquet")
        self.assertEqual({"IRE"}, set(features.loc[features.race_id.isin(tomorrow.race_id) &
                                                   features.horse_no.eq(1), "horse_country"]))
        self.assertEqual([], self.fits)
        self.assertFalse((self.output / "training-features.parquet").exists())
        self.assertFalse((self.output / "models").exists())

    def test_declared_net_weights_reach_query_inputs_and_preserve_handicap_provenance(self):
        original = copy.deepcopy(self.cards)
        for card in self.cards:
            card["runners"][0].update(net_carried_weight_lbs=122, apprentice_allowance_lbs=3,
                                      over_weight=0)
            card["runners"][1]["net_carried_weight_lbs"] = None
        self.write_json(self.official / "racecards.json", self.cards)
        converted = cli.runner_rows(self.cards, card=True)
        self.assertEqual({122}, set(converted.loc[converted.horse_no.eq(1), "actual_weight"]))
        self.assertEqual({125}, set(converted.loc[converted.horse_no.eq(2), "actual_weight"]))
        self.assertEqual({125}, set(converted.loc[converted.horse_no.eq(1), "weight"]))
        self.assertEqual({3}, set(converted.loc[converted.horse_no.eq(1), "apprentice_allowance_lbs"]))
        result = cli.runner_rows([race("2026-09-06", 1)])
        self.assertEqual({125}, set(result.actual_weight))
        with self.frozen_calibration():
            self.invoke("prepare")
        for filename in ("query-metadata.parquet", "query-features.parquet"):
            prepared = pd.read_parquet(self.output / filename)
            tomorrow = prepared[prepared.race_id.isin(converted.race_id)]
            self.assertEqual({122}, set(tomorrow.loc[tomorrow.horse_no.eq(1), "actual_weight"]))
            self.assertEqual({125}, set(tomorrow.loc[tomorrow.horse_no.eq(2), "actual_weight"]))
        metadata = pd.read_parquet(self.output / "query-metadata.parquet")
        picks = metadata[metadata.cohort.eq("tomorrow") & metadata.horse_no.eq(1)]
        self.assertEqual({125}, set(picks.weight))
        self.assertEqual({3}, set(picks.apprentice_allowance_lbs))
        self.assertEqual([], self.fits)
        self.assertEqual(125, original[0]["runners"][0]["weight"])

    @staticmethod
    def odds_quote_record(**overrides):
        return {
            "meeting_date": "2026-10-07", "venue": "ST", "race_no": 1,
            "pool": "WIN", "combination": "01", "odds_value_raw": "3.7",
            "odds_value_numeric": 3.7, "possible_display_ceiling": False,
            "retrieved_at_utc": "2026-10-06T13:00:00Z",
            "source_last_update": "2026-10-06T12:59:00Z",
            "source_url": "https://racing.hkjc.com/odds-snapshot", **overrides,
        }

    def write_odds_quotes(self, records, units=None):
        self.write_json(self.official / "odds-quotes.json", records)
        self.write_json(self.official / "odds-unit-semantics.json", {
            "pool_unit_conversion": units if units is not None else {
                "WIN": {"displayed_odds_to_D10_factor": 10,
                        "conversion_status": "verified_primary_formula"},
                "PLACE": {"displayed_odds_to_D10_factor": 10,
                          "conversion_status": "unverified_shared_display_inference_only"},
            },
        })

    def test_verified_win_snapshot_ev_uses_gross_factor_without_plus_one_or_takeout(self):
        self.write_odds_quotes([self.odds_quote_record()])
        quotes, units = cli.quote_lookup(self.official)
        ticket = CombinationProbability("WIN", ("1",), 0.4)
        result = cli.ticket_quote(ticket, cli.runner_rows([self.cards[0]], card=True), "WIN", quotes, units)
        self.assertEqual(37, result["quoted_gross_hkd_per_10"])
        self.assertAlmostEqual(4.8, result["quoted_ev_hkd"])
        self.assertEqual("indicative_snapshot_not_final", result["quote_status"])
        self.assertEqual("3.7", result["quoted_odds_raw"])
        self.assertEqual("2026-10-06T13:00:00Z", result["quote_retrieved_at_utc"])
        self.assertEqual("2026-10-06T12:59:00Z", result["quote_source_last_update"])

    def test_unverified_place_and_display_ceiling_have_no_point_ev(self):
        frame = cli.runner_rows([self.cards[0]], card=True)
        for record, pool, status in [
            (self.odds_quote_record(pool="PLACE"), "PLACE", "unit_unverified_no_point_ev"),
            (self.odds_quote_record(odds_value_numeric=999, odds_value_raw="999"),
             "WIN", "possible_display_ceiling_no_point_ev"),
            (self.odds_quote_record(possible_display_ceiling=True),
             "WIN", "possible_display_ceiling_no_point_ev"),
        ]:
            with self.subTest(pool=pool, status=status):
                self.write_odds_quotes([record])
                quotes, units = cli.quote_lookup(self.official)
                result = cli.ticket_quote(CombinationProbability(pool, ("1",), 0.4), frame, pool, quotes, units)
                self.assertEqual(status, result["quote_status"])
                self.assertIsNone(result["quoted_ev_hkd"])
                self.assertIsNone(result["quoted_gross_hkd_per_10"])

    def test_quote_identity_normalizes_zeroes_and_unordered_not_ordered_pools(self):
        records = [self.odds_quote_record(pool=pool, combination="04,03,02,01")
                   for pool in ("FIRST4", "QUARTET")]
        self.write_odds_quotes(records, {pool: {"displayed_odds_to_D10_factor": 10,
                                               "conversion_status": "verified_primary_formula"}
                                        for pool in ("FIRST4", "QUARTET")})
        quotes, units = cli.quote_lookup(self.official)
        frame = cli.runner_rows([self.cards[0]], card=True)
        forward = ("1", "2", "3", "4")
        first4 = cli.ticket_quote(CombinationProbability("FIRST4", forward, 0.1), frame, "FIRST4", quotes, units)
        self.assertAlmostEqual(-6.3, first4["quoted_ev_hkd"])
        quartet = cli.ticket_quote(CombinationProbability("QUARTET", forward, 0.1), frame, "QUARTET", quotes, units)
        self.assertEqual("not_available", quartet["quote_status"])
        reverse = cli.ticket_quote(CombinationProbability("QUARTET", forward[::-1], 0.1), frame, "QUARTET", quotes, units)
        self.assertAlmostEqual(-6.3, reverse["quoted_ev_hkd"])
        for overrides in ({"date": pd.Timestamp("2026-10-08")}, {"venue": "HV"}, {"race_no": 2}):
            changed = frame.assign(**overrides)
            result = cli.ticket_quote(CombinationProbability("FIRST4", forward, 0.1), changed, "FIRST4", quotes, units)
            self.assertEqual("not_available", result["quote_status"])
            self.assertIsNone(result["quoted_ev_hkd"])

    def test_normalized_duplicate_quote_identity_rejected(self):
        for records in (
            [self.odds_quote_record(), self.odds_quote_record(combination="1")],
            [self.odds_quote_record(pool="FIRST4", combination="04/03/02/01"),
             self.odds_quote_record(pool="FIRST4", combination="1,2,3,4")],
            [self.odds_quote_record(pool="TRIO", combination="03/02/01"),
             self.odds_quote_record(pool="TRI", combination="1/2/3")],
        ):
            self.write_odds_quotes(records)
            with self.assertRaisesRegex(ValueError, "Duplicate official quote identity"):
                cli.quote_lookup(self.official)

    def test_missing_and_non_numeric_quotes_have_no_ev(self):
        self.assertEqual(({}, {}), cli.quote_lookup(self.official))
        self.write_json(self.official / "odds-quotes.json", [self.odds_quote_record()])
        self.assertEqual(({}, {}), cli.quote_lookup(self.official))
        frame = cli.runner_rows([self.cards[0]], card=True)
        ticket = CombinationProbability("WIN", ("1",), 0.4)
        missing = cli.ticket_quote(ticket, frame, "WIN", {}, {})
        self.assertEqual("not_available", missing["quote_status"])
        self.assertIsNone(missing["quoted_ev_hkd"])
        for value in (None, 0, -1):
            self.write_odds_quotes([self.odds_quote_record(odds_value_numeric=value)])
            quotes, units = cli.quote_lookup(self.official)
            result = cli.ticket_quote(ticket, frame, "WIN", quotes, units)
            self.assertEqual("non_numeric_no_point_ev", result["quote_status"])
            self.assertIsNone(result["quoted_ev_hkd"])

    def test_cli_snapshot_ev_attaches_to_tomorrow_without_changing_selection_or_settlement(self):
        self.prepare_predictions()
        records = [self.odds_quote_record(), self.odds_quote_record(pool="PLACE"),
                   self.odds_quote_record(combination="07", odds_value_numeric=1e9, odds_value_raw="1000000000")]
        self.write_odds_quotes(records)
        summaries = self.evaluate()
        tomorrow = pd.read_csv(self.output / "evaluation" / "tomorrow-combinations.csv", dtype={"combination": str})
        pick = tomorrow[tomorrow.model.eq("benter_conditional_logit") & tomorrow.race_no.eq(1) &
                        tomorrow.pool.eq("WIN") & tomorrow["rank"].eq(1)].iloc[0]
        self.assertEqual("1", pick.combination)
        self.assertAlmostEqual(4.8, pick.quoted_ev_hkd)
        self.assertEqual(37, pick.quoted_gross_hkd_per_10)
        self.assertEqual("indicative_snapshot_not_final", pick.quote_status)
        self.assertTrue(tomorrow[tomorrow.pool.eq("PLACE")].quoted_ev_hkd.isna().all())
        self.assertEqual(1590, summaries["benter_conditional_logit"]["gross_hkd"])
        self.assertTrue(self.ledger("benter_conditional_logit").estimated_ev_hkd.isna().all())

    def test_history_backfill_requires_ready_for_freeze_and_retains_full_fields(self):
        history = pd.read_parquet(self.history_path)
        original = history.copy(deep=True)
        extra = race("2026-08-29", 1)
        self.write_json(self.official / "history-races.json", [extra])
        self.write_json(self.official / "history-race-coverage.json", {
            "ready_for_freeze": False, "complete": True,
        })
        with self.assertRaises(ValueError):
            cli.enrich_official_history(history, self.official)
        self.write_json(self.official / "history-race-coverage.json", {
            "ready_for_freeze": True, "complete": True,
        })
        enriched = cli.enrich_official_history(history, self.official)
        self.assertEqual(len(history) + 7, len(enriched))
        added = enriched[enriched.race_id.eq("HKJC:2026-08-29:ST:R1")]
        self.assertEqual(set(range(1, 8)), set(added.horse_no))
        self.assertEqual({7}, set(added.field_size))
        pd.testing.assert_frame_equal(history, original)
        pd.testing.assert_frame_equal(enriched, cli.enrich_official_history(enriched, self.official))

    def test_form_inputs_match_exact_race_and_horse_preserve_observed_inputs(self):
        frame = cli.runner_rows([self.season[0]])
        frame["gear"] = None
        frame.loc[0, "horse_rating"] = np.nan
        frame.loc[1, "gear"] = "TT"
        original = frame.copy(deep=True)
        forms = [{"horse_page_id": "HK_2020_H1", "form_records": [
            {"race_date": "2026-09-06", "venue": "ST", "race_no": 1,
             "rating": 71, "gear": "B", "result": 7, "win_odds": 999,
             "result_source_url": "https://racing.hkjc.com/exact"},
            {"race_date": "2026-09-07", "venue": "ST", "race_no": 1,
             "rating": 999, "gear": "WRONG_DATE"},
            {"race_date": "2026-09-06", "venue": "HV", "race_no": 1,
             "rating": 999, "gear": "WRONG_VENUE"},
            {"race_date": "2026-09-06", "venue": "ST", "race_no": 2,
             "rating": 999, "gear": "WRONG_RACE"},
        ]}, {"horse_page_id": "HK_2020_H2", "form_records": [
            {"race_date": "2026-09-06", "venue": "ST", "race_no": 1,
             "rating": 999, "gear": "B"},
        ]}, {"horse_page_id": "HK_2020_NOT_A_STARTER", "form_records": [
            {"race_date": "2026-09-06", "venue": "ST", "race_no": 1,
             "rating": 999, "gear": "WRONG_HORSE"},
        ]}]
        self.write_json(self.official / "horse_histories.json", forms)
        enriched = cli.enrich_form_inputs(frame, self.official)
        self.assertEqual(71, enriched.loc[0, "horse_rating"])
        self.assertEqual("B", enriched.loc[0, "gear"])
        self.assertEqual("official:horse-form:https://racing.hkjc.com/exact",
                         enriched.loc[0, "horse_rating_source"])
        self.assertEqual(60, enriched.loc[1, "horse_rating"])
        self.assertEqual("TT", enriched.loc[1, "gear"])
        for name in ["result", "win_odds", "horse_id", "horse_no", "actual_weight"]:
            pd.testing.assert_series_equal(frame[name], enriched[name])
        self.assertTrue(enriched.loc[2:, "gear"].isna().all())
        pd.testing.assert_frame_equal(frame, original)

    def test_conflicting_form_records_rejected_instead_of_order_dependent_inputs(self):
        frame = cli.runner_rows([self.season[0]])
        frame.loc[0, "horse_rating"] = np.nan
        first = {"race_date": "2026-09-06", "venue": "ST", "race_no": 1,
                 "rating": 71, "gear": "B"}
        second = {**first, "rating": 99, "gear": "TT"}
        for records in [[first, second], [second, first]]:
            self.write_json(self.official / "horse_histories.json", [{
                "horse_page_id": "HK_2020_H1", "form_records": records,
            }])
            with self.assertRaises(ValueError):
                cli.enrich_form_inputs(frame, self.official)

    def test_incomplete_census_is_rejected_before_preparation(self):
        self.write_json(self.official / "coverage_manifest.json", {
            "results_complete": False, "racecards_complete": True,
        })
        with self.assertRaises(ValueError):
            self.invoke("prepare")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
