"""Synthetic CLI guards; no real packages, training, or historical evaluation."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import search_season_pool as cli


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metadata_frame():
    records = []
    populations = [("calibration", "2026-01-14", 10, 2),
                   ("season", "2026-09-06", 1, 978),
                   ("tomorrow", "2026-10-07", 1, 108)]
    for cohort, start, meetings, runners in populations:
        for meeting in range(meetings):
            day = pd.Timestamp(start) + pd.Timedelta(days=meeting)
            race_id = f"{cohort}-{meeting}"
            for horse in range(1, runners + 1):
                records.append(dict(date=day, race_id=race_id, race_no=1,
                                    horse_no=horse, horse_id=f"H{horse}",
                                    horse_name=f"Horse {horse}", field_size=runners,
                                    cohort=cohort, horse_rating=50.,
                                    result=horse if cohort != "tomorrow" else np.nan))
    return pd.DataFrame(records)


class StopBeforeEvaluation(RuntimeError):
    pass


class SeasonPoolSearchCLITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.components = self.root / "components"
        self.metadata = metadata_frame()
        self.source.mkdir()
        self.components.mkdir()
        for name in ("races.json", "odds-quotes.json", "odds-unit-semantics.json"):
            write_json(self.source / "official" / name, {})
        self.metadata.to_parquet(self.source / "query-metadata.parquet", index=False)
        self.metadata[cli.KEYS].to_parquet(self.source / "query-features.parquet", index=False)
        probability = 1 / self.metadata.field_size.to_numpy()
        models = {}
        for family in ("benter_conditional_logit", "boosted", "pool", "gaussian_probit"):
            path = self.source / "fresh-predictions" / f"{family}.csv"
            path.parent.mkdir(exist_ok=True)
            self.metadata[cli.KEYS].assign(model_probability=probability).to_csv(path, index=False)
            models[family] = dict(package_id=f"fixture-{family}", training_cutoff="2026-01-11",
                                  output_sha256=cli.sha(path))
        query_hash = cli.sha(self.source / "query-features.parquet")
        self.fresh = dict(input_sha256=query_hash, models=models)
        self.manifest = {"candidates": {family: dict(package_id=model["package_id"],
                         fit_date_max="2026-01-11") for family, model in models.items()}}
        self.component_receipt = dict(input_sha256=query_hash, package_id="fixture-pool",
                                     training_cutoff="2026-01-11", fit_date_max="2026-01-11",
                                     models={})
        for family in ("pool_benter", "pool_boosted"):
            path = self.components / f"{family}.csv"
            self.metadata[cli.KEYS].assign(model_probability=probability).to_csv(path, index=False)
            self.component_receipt["models"][family] = dict(package_id="fixture-pool",
                                                           output_sha256=cli.sha(path))
        self.save_receipts()

    def save_receipts(self):
        write_json(self.source / "fresh-predictions/readback.json", self.fresh)
        write_json(self.source / "final-fits/fresh-candidates.json", self.manifest)
        write_json(self.components / "readback.json", self.component_receipt)

    def test_aligned_predictions_restore_metadata_order(self):
        path = self.root / "shuffled.csv"
        small = self.metadata.iloc[:4].copy()
        values = np.array([.8, .2, .6, .4])
        small[cli.KEYS].assign(model_probability=values).iloc[::-1].to_csv(path, index=False)
        np.testing.assert_array_equal(cli.aligned_predictions(small, path, cli.sha(path)), values)

    def test_aligned_predictions_reject_duplicates_missing_extra_and_hash_mismatch(self):
        small = self.metadata.iloc[:4].copy()
        frame = small[cli.KEYS].assign(model_probability=.5)
        variants = {"duplicate": pd.concat([frame.iloc[:3], frame.iloc[:1]]),
                    "missing": frame.iloc[:3],
                    "replacement": frame.assign(horse_id=["unknown", *frame.horse_id.iloc[1:]]),
                    "extra": pd.concat([frame, frame.iloc[:1].assign(horse_id="extra")])}
        path = self.root / "bad.csv"
        for name, bad in variants.items():
            with self.subTest(name=name):
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    cli.aligned_predictions(small, path, cli.sha(path))
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            cli.aligned_predictions(small, path, "0" * 64)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cli.aligned_predictions(pd.concat([small, small.iloc[:1]]), path, cli.sha(path))

    def test_load_inputs_positive_control(self):
        metadata, pairs, hashes, receipt = cli.load_inputs(self.source, self.components)
        pd.testing.assert_frame_equal(metadata, self.metadata)
        self.assertEqual(set(pairs), {"original_pair", "standalone_pair"})
        self.assertEqual(receipt, self.fresh)
        self.assertTrue(all(cli.sha(path) == digest for path, digest in hashes.items()))

    def test_load_inputs_rejects_stale_receipts_and_candidate_identities(self):
        original = copy.deepcopy(self.fresh)
        for name in ("fresh", "components"):
            with self.subTest(receipt=name):
                receipt = self.fresh if name == "fresh" else self.component_receipt
                correct = receipt["input_sha256"]
                receipt["input_sha256"] = "0" * 64
                self.save_receipts()
                with self.assertRaisesRegex(ValueError, "different query"):
                    cli.load_inputs(self.source, self.components)
                receipt["input_sha256"] = correct
        self.fresh["models"]["pool"]["package_id"] = "wrong-package"
        self.save_receipts()
        with self.assertRaisesRegex(ValueError, "identity"):
            cli.load_inputs(self.source, self.components)
        self.fresh = original

    def test_load_inputs_rejects_overlapping_fresh_cutoffs(self):
        for target in (self.fresh["models"]["pool"], self.manifest["candidates"]["pool"]):
            key = "training_cutoff" if "training_cutoff" in target else "fit_date_max"
            for value in ("2026-01-14", "2026-09-06"):
                with self.subTest(key=key, value=value):
                    target[key] = value
                    self.save_receipts()
                    with self.assertRaisesRegex(ValueError, "overlaps"):
                        cli.load_inputs(self.source, self.components)
            target[key] = "2026-01-11"

    def test_load_inputs_rejects_null_fresh_cutoffs(self):
        for target, key in ((self.fresh["models"]["pool"], "training_cutoff"),
                            (self.manifest["candidates"]["pool"], "fit_date_max")):
            with self.subTest(key=key):
                target[key] = None
                self.save_receipts()
                with self.assertRaises(ValueError):
                    cli.load_inputs(self.source, self.components)
                target[key] = "2026-01-11"

    def test_load_inputs_rejects_component_parent_identity_and_cutoffs(self):
        original = copy.deepcopy(self.component_receipt)
        for key, value in (("package_id", "wrong-parent"),
                           ("training_cutoff", "2026-01-14"),
                           ("fit_date_max", "2026-09-06"),
                           ("fit_date_max", None)):
            with self.subTest(key=key, value=value):
                self.component_receipt = copy.deepcopy(original)
                self.component_receipt[key] = value
                self.save_receipts()
                with self.assertRaises(ValueError):
                    cli.load_inputs(self.source, self.components)
        self.component_receipt = copy.deepcopy(original)
        self.component_receipt["models"]["pool_benter"]["package_id"] = "wrong-parent"
        self.save_receipts()
        with self.assertRaises(ValueError):
            cli.load_inputs(self.source, self.components)

    def test_run_search_refuses_nonempty_outputs_before_loading(self):
        output = self.root / "already-used"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"original evidence")
        with patch.object(cli, "load_inputs") as loading:
            with self.assertRaisesRegex(ValueError, "new empty"):
                cli.run_search(self.source, self.components, output)
            loading.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"original evidence")

    def test_selection_precedes_evaluation_and_is_invariant_to_season_labels(self):
        original = {str(path): cli.sha(path) for directory in (self.source, self.components)
                    for path in directory.rglob("*") if path.is_file()}
        probabilities = np.column_stack([1 / self.metadata.field_size] * 2)
        probabilities[:20] = np.tile([[.8, .2], [.2, .8]], (10, 1))
        pairs = {"original_pair": probabilities, "standalone_pair": probabilities}
        selections = {False: [], True: []}
        grid = pd.DataFrame({"search_race_log_loss": np.full(588, .7)})
        for adopted, changed in itertools.product((False, True), repeat=2):
            selected_recipe = cli.PoolRecipe("standalone_pair", "geometric",
                                             1. if adopted else .2, .85)
            metadata = self.metadata.copy()
            if changed:
                metadata.loc[metadata.cohort.eq("season"), "result"] = 999
            output = self.root / f"adoption-{adopted}-changed-{changed}"

            def evaluate_spy(args):
                selection_path = args.output / "selection.json"
                self.assertTrue(selection_path.exists())
                selection = json.loads(selection_path.read_text())
                self.assertEqual(cli.sha(selection_path), json.loads(
                    (args.predictions_dir / "readback.json").read_text())["selection_sha256"])
                derived = pd.read_parquet(args.output / "query-metadata.parquet")
                supplied_calibration = derived[derived.cohort.eq("calibration")]
                expected_dates = pd.to_datetime(["2026-01-22", "2026-01-23"])
                self.assertEqual(set(supplied_calibration.date), set(expected_dates))
                self.assertEqual(set(derived.loc[derived.cohort.eq("search"), "date"]),
                                 set(pd.date_range("2026-01-14", periods=6)))
                self.assertEqual(len(supplied_calibration), 4)
                self.assertEqual(selection["adopted"], adopted)
                selections[adopted].append(selection)
                raise StopBeforeEvaluation("Do not score synthetic season")

            def search_spy(frame, supplied_pairs):
                self.assertTrue(frame.cohort.eq("calibration").all())
                self.assertLessEqual(frame.date.max(), pd.Timestamp("2026-01-19"))
                self.assertEqual(len(frame), 12)
                self.assertEqual(set(supplied_pairs), set(pairs))
                return grid, selected_recipe

            with patch.object(cli, "load_inputs", return_value=(metadata, pairs, original, self.fresh)), \
                    patch.object(cli, "search_pool", side_effect=search_spy), \
                    patch.object(cli, "evaluate", side_effect=evaluate_spy) as evaluation:
                with self.assertRaises(StopBeforeEvaluation):
                    cli.run_search(self.source, self.components, output)
                evaluation.assert_called_once()
        for decisions in selections.values():
            self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(original, {path: cli.sha(path) for path in original})

    def test_run_search_rejects_official_source_mutation_during_evaluation(self):
        official = self.source / "official"
        for name in ("races.json", "odds-quotes.json", "odds-unit-semantics.json"):
            write_json(official / name, {})
        probabilities = np.column_stack([1 / self.metadata.field_size] * 2)
        pairs = {"original_pair": probabilities, "standalone_pair": probabilities}
        output = self.root / "source-mutation"

        def evaluate_spy(args):
            self.assertTrue((args.output / "selection.json").exists())
            write_json(official / "races.json", {"changed_after_scoring": True})
            evaluation = args.output / "evaluation"
            evaluation.mkdir()
            write_json(evaluation / "calibration.json", {})
            write_json(evaluation / "summary.json", {})
            for name in ("baseline_pool", "candidate_pool"):
                self.metadata[self.metadata.cohort.eq("tomorrow")].to_csv(
                    evaluation / f"{name}-tomorrow-runners.csv", index=False)

        grid = pd.DataFrame({"search_race_log_loss": np.full(588, .7)})
        with patch.object(cli, "search_pool", return_value=(grid, cli.BASELINE)), \
                patch.object(cli, "evaluate", side_effect=evaluate_spy), \
                patch.object(cli, "quote_lookup", return_value=({}, {})), \
                patch.object(cli, "verify_quote_sources", return_value=[]), \
                patch.object(cli, "build_forecast", return_value=(
                    pd.DataFrame({"synthetic": range(216)}),
                    pd.DataFrame({"synthetic": range(432)}))):
            with self.assertRaisesRegex(ValueError, "source|Source|changed|Changed"):
                cli.run_search(self.source, self.components, output)
        self.assertFalse((output / "result-provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
