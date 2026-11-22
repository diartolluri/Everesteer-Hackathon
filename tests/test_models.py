import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

import hackathon_models as models


class ModelTests(unittest.TestCase):
    def test_walk_forward_purges_by_exped_and_reserves_the_final_window(self):
        calendar = [f"exped_{index:04}" for index in range(90)]
        expeds = pd.Series(list(reversed(calendar)) * 2)
        final = models.chronological_split(expeds, 12, 10)
        folds = models.walk_forward_splits(final, 3, 12, 10)
        self.assertEqual(len(folds), 3)
        observed_evaluations = set()
        for fold in [*folds, final]:
            fit = fold["fit_expeds"]
            embargo = fold["embargo_expeds"]
            evaluate = fold["holdout_expeds"]
            self.assertLess(fit[-1], embargo[0])
            self.assertLess(embargo[-1], evaluate[0])
            self.assertEqual(len(embargo), 10)
            self.assertTrue(set(fit).isdisjoint(evaluate))
            self.assertTrue(observed_evaluations.isdisjoint(evaluate))
            observed_evaluations.update(evaluate)
        for fold in folds:
            self.assertTrue(set(fold["holdout_expeds"]) <= set(final["fit_expeds"]))

    def test_feature_names_and_missing_sentinel_drive_model_input(self):
        frame = pd.DataFrame({"a": [-99, 2], "b": [0, 1], "exped": ["e1", "e1"], "graded": [3, 4]})
        schema = {
            "primary_target": "graded",
            "feature_sets": {"all": ["b", "a"]},
            "feature_encoding": {"missing": -99},
        }
        target, exped, columns, sentinel = models.dataset_columns(frame, schema)
        self.assertEqual((target, exped, columns, sentinel), ("graded", "exped", ["a", "b"], -99))
        matrix = models.select_features(frame.drop(columns="b"), columns, sentinel)
        self.assertTrue(pd.isna(matrix.loc[0, "a"]))
        self.assertEqual(matrix.loc[1, "a"], 2)
        self.assertTrue(matrix["b"].isna().all())
        self.assertEqual(models.ridge_design(matrix).tolist(), [[0, 0, 1, 1], [2, 0, 0, 1]])

    def test_benchmark_requires_the_same_id_and_exped(self):
        index = pd.Index([f"opaque-{i}" for i in range(8)], name="id")
        holdout = pd.DataFrame({"exped": ["e1"] * 4 + ["e2"] * 4}, index=index)
        benchmark = holdout.assign(estimate=range(8)).iloc[::-1].copy()
        benchmark.loc[index[2], "exped"] = "different-exped"
        with contextlib.redirect_stdout(io.StringIO()):
            matched, coverage = models.align_benchmark(benchmark, holdout, "exped")
        self.assertEqual(coverage, 7 / 8)
        self.assertTrue(pd.isna(matched.loc[index[2]]))
        self.assertEqual(matched.loc[index[3]], 3)

    def test_tournament_weights_do_not_get_an_event_score_proxy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                models.event_proxy_scale(
                    {"weights": {"fit": 1, "unq": 1, "inov": 1, "score_multiple_constant": 0.6}}
                ),
                (None, "fit"),
            )
        self.assertEqual(
            models.event_proxy_scale(
                {"weights": {"fit": 1, "unq": 3, "inov": 1, "score_multiple_constant": 0.6}}
            ),
            (0.6, None),
        )

    def test_final_holdout_labels_do_not_change_model_selection(self):
        train, benchmark, schema, scale = models.smoke_inputs()
        expeds = sorted(train["exped"].unique())[-120:]
        train = train.loc[train["exped"].isin(expeds)].copy()
        altered = train.copy()
        held_out = altered["exped"].isin(expeds[-20:])
        altered.loc[held_out, schema["primary_target"]] *= -1
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            results = []
            for name, labeled in (("original", train), ("altered", altered)):
                path = Path(scratch) / name
                args = Namespace(
                    artifacts=path,
                    holdout_expeds=20,
                    embargo_expeds=20,
                    cv_folds=2,
                    fold_expeds=12,
                    feature_set=None,
                    auxiliary_target=None,
                    max_fit_expeds=None,
                    baseline_only=True,
                    smoke_test=True,
                )
                models.train_and_package(labeled, benchmark, schema, scale, args)
                manifest = json.loads((path / "manifest.json").read_text())
                selection = (path / manifest["selection_path"]).read_text()
                comparison = (path / manifest["comparison_path"]).read_text()
                results.append((selection, comparison))
        self.assertEqual(results[0][0], results[1][0])
        self.assertNotEqual(results[0][1], results[1][1])

    def test_cloudpickle_preserves_index_shuffled_rows_and_exped_isolation(self):
        rng = np.random.default_rng(2026)
        size = 144
        frame = pd.DataFrame(
            {"a": rng.integers(0, 5, size), "b": rng.integers(0, 5, size)},
            index=pd.Index([f"opaque-{i}" for i in range(size)], name="id"),
        )
        frame["exped"] = np.repeat([f"e{i}" for i in range(12)], 12)
        frame["graded"] = frame["a"] - 2 * frame["b"] + rng.normal(0, 0.2, size)
        frame.loc[frame.index[::11], "a"] = -99
        fits = models.model_predictions(frame.iloc[:96], ["a", "b"], -99, "graded")
        fits["rank-blend"] = models.blend_prediction(fits, "exped", 0.5)
        sample = frame.iloc[96:].drop(columns="graded")
        with tempfile.TemporaryDirectory() as directory:
            records = models.package(fits, sample, "exped", Path(directory))
            for name, record in records.items():
                with self.subTest(model=name):
                    path = Path(directory) / Path(record["path"]).name
                    predict = cloudpickle.loads(path.read_bytes())
                    expected = models.check_predictions(predict, sample)
                    shuffled = sample.sample(frac=1, random_state=17).loc[:, ["exped", "b", "a"]]
                    shuffled_result = models.check_predictions(predict, shuffled)
                    self.assertTrue(shuffled_result.index.equals(shuffled.index))
                    pd.testing.assert_frame_equal(expected, shuffled_result.reindex(expected.index))
                    self.assertGreater(expected["prediction"].nunique(), 1)
                    models.check_predictions(predict, sample.drop(columns="b"))
                    current = sample[sample["exped"] == sample["exped"].iloc[0]]
                    pd.testing.assert_frame_equal(
                        expected.loc[current.index], models.check_predictions(predict, current)
                    )
