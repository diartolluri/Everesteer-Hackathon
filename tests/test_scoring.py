import contextlib
import io
import math
import unittest

import numpy as np
import pandas as pd
from everestapi import scoring

from everesteer_hackathon.scoring import (
    align_benchmark,
    benchmark_neutralize,
    core_reference,
    event_metrics,
    resolve_core_features,
    score_expeds,
)


class ScoringTests(unittest.TestCase):
    def test_alignment_checks_both_keys_and_preserves_row_order(self):
        ids = pd.Index(["opaque-z", "opaque-a", "opaque-b", "opaque-c"], name="id")
        holdout = pd.DataFrame({"exped": ["one", "one", "two", "two"]}, index=ids)
        benchmark = holdout.assign(reference=[0.4, 0.9, 0.1, 0.3]).iloc[::-1].copy()
        benchmark.loc["opaque-a", "exped"] = "old"
        benchmark["another_series"] = 7.0
        with contextlib.redirect_stdout(io.StringIO()):
            ambiguous = align_benchmark(benchmark, holdout, "exped")
            matched, coverage = align_benchmark(
                benchmark, holdout, "exped", score_column="reference"
            )
        self.assertEqual(ambiguous, (None, None))
        self.assertEqual(coverage, 0.75)
        pd.testing.assert_series_equal(
            matched, pd.Series([0.4, np.nan, 0.1, 0.3], index=ids, name="reference")
        )
        duplicate = pd.concat([benchmark, benchmark.iloc[[0]]])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                align_benchmark(duplicate, holdout, "exped", score_column="reference"),
                (None, None),
            )

    def test_neutralization_is_cross_sectional_and_retains_baseline_ranks(self):
        ids = pd.Index([f"id-{n}" for n in range(16)], name="id")
        expeds = pd.Series(["a"] * 8 + ["b"] * 8, index=ids)
        reference = pd.Series(list(range(8)) * 2, index=ids, dtype=float)
        predictions = reference.copy()
        predictions.loc[ids[8:]] = [2, 7, 1, 3, 5, 4, 0, 6]
        expected_base = (predictions.groupby(expeds).rank() - 0.5) / 8
        pd.testing.assert_series_equal(
            benchmark_neutralize(predictions, reference, expeds, 0), expected_base
        )

        adjusted = benchmark_neutralize(predictions, reference, expeds, 1.0)
        self.assertEqual(adjusted.loc[ids[:8]].tolist(), [0.5] * 8)
        self.assertTrue(adjusted.between(0, 1, inclusive="neither").all())
        shuffled = predictions.sample(frac=1, random_state=7)
        shuffled.loc[ids[8:]] *= -1
        reference.loc[ids[8:]] = np.nan
        result = benchmark_neutralize(shuffled, reference, expeds, 1.0)
        self.assertTrue(result.index.equals(shuffled.index))
        pd.testing.assert_series_equal(result.loc[ids[:8]], adjusted.loc[ids[:8]])
        pd.testing.assert_series_equal(
            result.loc[ids[8:]], (shuffled.loc[ids[8:]].rank() - 0.5) / 8
        )

    def test_full_projection_does_not_rerank_rounding_noise(self):
        reference = pd.Series(np.arange(91, dtype=float))
        expeds = pd.Series("one", index=reference.index)
        neutralized = benchmark_neutralize(-reference, reference, expeds, 1.0)
        self.assertTrue(neutralized.eq(0.5).all())

    def test_event_components_use_sdk_and_average_before_arctangent(self):
        index = pd.Index([f"id-{n}" for n in range(16)], name="id")
        benchmark = pd.Series(list(range(8)) * 2, index=index, dtype=float)
        prediction = pd.Series([1, 4, 0, 7, 3, 6, 2, 5] * 2, index=index, dtype=float)
        target = prediction - 4 * benchmark
        target.iloc[8:] = prediction.iloc[8:] + 2 * benchmark.iloc[8:]
        frame = pd.DataFrame(
            {
                "exped": ["first"] * 8 + ["second"] * 8,
                "target": target,
                "core1": benchmark,
                "core2": benchmark * 2,
            }
        )
        frame.loc[index[1], "core2"] = -99.0
        cols = ["core1", "core2"]
        by_exped = score_expeds(
            frame,
            "exped",
            "target",
            {"candidate": prediction},
            benchmark,
            core_features=cols,
            missing=-99,
        ).set_index("exped")
        for label, group in frame.groupby("exped"):
            pred = prediction.loc[group.index]
            ref = benchmark.loc[group.index]
            actual = group["target"]
            core = group[cols].replace(-99, np.nan)
            self.assertAlmostEqual(by_exped.loc[label, "fit"], scoring.fit20(pred, actual))
            self.assertAlmostEqual(
                by_exped.loc[label, "unq_covariance"],
                scoring.unq20(pred, ref, actual, zero_rule=False),
            )
            self.assertAlmostEqual(
                by_exped.loc[label, "inov"], scoring.inov(pred, core, actual, zero_rule=False)
            )
            self.assertAlmostEqual(
                by_exped.loc[label, "benchmark_inov"],
                scoring.inov(ref, core, actual, zero_rule=False),
            )
        self.assertLess(by_exped.loc["first", "benchmark_fit"], 0)
        self.assertNotEqual(by_exped.loc["first", "unq_covariance"], 0)
        self.assertEqual(core_reference(frame[cols], missing=-99).loc[index[1]], 1.0)

        metrics = event_metrics(by_exped, 0.6)
        means = by_exped.drop(columns="model").mean()
        want_signal = (
            means["fit"]
            + 3 * means["unq_covariance"]
            + means["inov"]
            - means["benchmark_fit"]
            - means["benchmark_inov"]
        )
        self.assertAlmostEqual(metrics["event_signal"], want_signal)
        self.assertAlmostEqual(metrics["event_score"], 0.6 * math.atan(3 * want_signal / 0.6))

        without_core = score_expeds(frame, "exped", "target", {"candidate": prediction}, benchmark)
        partial = event_metrics(without_core, 0.6)
        self.assertTrue(np.isnan(partial["event_score"]))
        self.assertTrue(np.isnan(partial["event_signal"]))
        self.assertTrue(np.isfinite(partial["round_score_proxy"]))

        copied = score_expeds(
            frame,
            "exped",
            "target",
            {"copy": benchmark},
            benchmark,
            core_features=cols,
            missing=-99,
        )
        self.assertAlmostEqual(event_metrics(copied, 0.6)["event_score"], 0.0)

    def test_missing_benchmark_preserves_fit_and_inov_with_no_event_score(self):
        index = pd.Index([f"id-{n}" for n in range(8)], name="id")
        target = pd.Series([0.5, 0.7, 0.2, 0.3, 0.6, 0.4, 0.8, 0.2], index=index)
        frame = pd.DataFrame({"exped": "single", "target": target, "core": range(8)})
        by_exped = score_expeds(
            frame, "exped", "target", {"candidate": target}, core_features=["core"]
        )
        self.assertGreater(by_exped.loc[0, "fit"], 0)
        self.assertTrue(np.isfinite(by_exped.loc[0, "inov"]))
        self.assertTrue(pd.isna(by_exped.loc[0, "unq_covariance"]))
        self.assertTrue(all(pd.isna(value) for value in event_metrics(by_exped, 0.6).values()))

    def test_core_overlap_never_invents_membership(self):
        schema = {"core_feature_overlap": {"all": 2}, "feature_sets": {"all": ["a", "b", "c"]}}
        self.assertEqual(resolve_core_features(schema), [])
        self.assertEqual(resolve_core_features(schema, ["b", "c"]), ["b", "c"])
        with self.assertRaises(ValueError):
            resolve_core_features(schema, ["b"])
        with self.assertRaises(ValueError):
            resolve_core_features(schema, ["b", "d"])


if __name__ == "__main__":
    unittest.main()
