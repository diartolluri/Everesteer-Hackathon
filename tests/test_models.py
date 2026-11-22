import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

import everesteer_hackathon.models as models


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

    def test_stable_feature_selection_finds_residual_and_low_exposure_signal(self):
        days = np.repeat(["e1", "e2", "e3"], 36)
        main = np.tile(np.repeat(np.arange(6), 6), 3)
        residual = np.tile(np.arange(6), 18)
        frame = pd.DataFrame(
            {
                "exped": days,
                "main": main,
                "residual": residual,
                "absent": -99,
                "graded": 3 * main + residual,
            }
        )
        columns = ["main", "residual", "absent"]
        stability = models.feature_stability(
            frame, columns, -99, "graded", "exped", frame["main"], frame["main"]
        )
        self.assertEqual(models.stable_feature_subset(stability, "fit", 1), ["main"])
        self.assertEqual(models.stable_feature_subset(stability, "residual", 1), ["residual"])
        self.assertEqual(models.stable_feature_subset(stability, "low_exposure", 1), ["residual"])
        self.assertAlmostEqual(stability.loc["main", "benchmark_corr"], 1)
        self.assertEqual(stability.loc["absent", "missing_rate"], 1)
        self.assertTrue(pd.isna(stability.loc["absent", "fit_mean"]))

    def test_research_predictors_preserve_named_features_missingness_and_range(self):
        train, _, schema, _ = models.smoke_inputs()
        target, exped, columns, missing = models.dataset_columns(train, schema)
        ordered = sorted(train[exped].unique())
        fit = train.loc[train[exped].isin(ordered[:12]) & train[target].notna()]
        sample = train.loc[train[exped].isin(ordered[-2:]), [exped, *columns]]
        predictors = models.model_predictions(
            fit,
            columns,
            missing,
            target,
            exped=exped,
            configs={
                "ridge": {
                    "family": "ridge",
                    "alpha": 10,
                    "missing_policy": "median-indicator",
                    "feature_selection": "fit",
                    "max_features": 2,
                    "pca_components": 2,
                },
                "tree": {
                    "family": "lightgbm",
                    "n_estimators": 10,
                    "missing_policy": "sentinel",
                },
            },
        )
        self.assertEqual(len(predictors["ridge"].feature_columns), 2)
        weights = {"tree": 3, "ridge": 1}
        blend = models.weighted_rank_prediction(predictors, exped, weights)
        result = models.check_predictions(blend, sample)["prediction"]
        self.assertTrue(result.gt(0).all())
        self.assertTrue(result.le(1).all())
        forecasts = pd.DataFrame(
            {name: predict(sample)["prediction"] for name, predict in predictors.items()}
        )
        pd.testing.assert_series_equal(result, models.rank_blend(forecasts, sample[exped], weights))
        for _, rows in sample.groupby(exped):
            pd.testing.assert_series_equal(result.loc[rows.index], blend(rows)["prediction"])
        models.check_predictions(blend, sample.drop(columns=columns[0]))

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

    def test_surrogate_hedge_round_trips_without_local_code_on_import_path(self):
        train, benchmark, schema, _ = models.smoke_inputs()
        target, exped, columns, missing = models.dataset_columns(train, schema)
        calendar = sorted(train[exped].unique())
        fit = train.loc[train[exped].isin(calendar[:10]) & train[target].notna()].copy()
        fit["benchmark"] = benchmark["reference"].reindex(fit.index)
        config = {"ridge": models.BASE_CONFIGS["ridge"]}
        primary = models.model_predictions(fit, columns, missing, target, configs=config)["ridge"]
        reference = models.model_predictions(fit, columns, missing, "benchmark", configs=config)[
            "ridge"
        ]
        hedge = models.benchmark_surrogate_predictor(primary, reference, exped, 0.75)
        sample = train.loc[train[exped].isin(calendar[-2:]), [exped, *columns]]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            models.package({"hedge": hedge}, sample, exped, folder)
            sample.to_parquet(folder / "sample.parquet")
            script = (
                "import sys, cloudpickle, pandas as pd\n"
                "with open('hedge.pkl', 'rb') as artifact:\n"
                "    predict = cloudpickle.load(artifact)\n"
                "assert 'everesteer_hackathon.scoring' not in sys.modules\n"
                "assert 'everesteer_hackathon.models' not in sys.modules\n"
                "frame = pd.read_parquet('sample.parquet')\n"
                "predict(frame).to_parquet('result.parquet')\n"
            )
            subprocess.run([sys.executable, "-I", "-c", script], cwd=folder, check=True)
            result = pd.read_parquet(folder / "result.parquet")
        expected = hedge(sample)
        pd.testing.assert_frame_equal(result, expected)
        self.assertTrue(result["prediction"].between(0, 1, inclusive="neither").all())
