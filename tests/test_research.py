import unittest

import numpy as np
import pandas as pd

from research.research_experiments import (
    BENCHMARK_MODEL,
    Candidate,
    candidate_predictor,
    choose_finalists,
    predict_candidates,
    rank_candidates,
    single,
)


class ResearchTests(unittest.TestCase):
    def test_offline_blend_and_surrogate_match_deployed_predictions(self):
        rng = np.random.default_rng(53)
        frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["tree", "ridge", BENCHMARK_MODEL])
        frame["exped"] = np.repeat(["a", "b", "c"], 20)
        frame.index = pd.Index([f"opaque-{i}" for i in range(len(frame))], name="id")
        predictors = {
            name: lambda features, name=name: features[name].to_frame("prediction")
            for name in ("tree", "ridge", BENCHMARK_MODEL)
        }
        candidates = [
            single("ridge"),
            Candidate("blend", (("tree", 0.4), ("ridge", 0.6))),
            Candidate("neutralized", (("tree", 0.4), ("ridge", 0.6)), 0.8),
        ]
        reference = frame["ridge"]
        offline = predict_candidates(
            candidates, frame.drop(columns="exped"), frame["exped"], reference
        )
        for candidate in candidates:
            predict = candidate_predictor(candidate, predictors, "exped")
            values = predict(frame)["prediction"]
            pd.testing.assert_series_equal(values, offline[candidate.name], check_names=False)
            self.assertTrue(values.gt(0).all() and values.le(1).all())
            changed = frame.copy()
            changed.loc[changed["exped"].eq("c"), "tree"] *= -1000
            past = frame["exped"].ne("c")
            pd.testing.assert_series_equal(
                values.loc[past], predict(changed)["prediction"].loc[past]
            )

    def test_oracle_and_incomplete_benchmark_coverage_are_ineligible(self):
        portable = single("portable")
        oracle = Candidate("oracle", portable.weights, 0.8, oracle=True)
        incomplete = single("incomplete")
        template = {
            "expeds": 20,
            "fit": 0.02,
            "unq": 0.01,
            "inov": np.nan,
            "event_signal": np.nan,
            "event_score": np.nan,
            "round_score_proxy": 0.06,
            "relative_signal_proxy": 0.02,
            "benchmark_corr": 0.2,
            "benchmark_exped_share": 1.0,
            "core_exped_share": 0.0,
        }
        rows = [
            {**template, "candidate": candidate.name, "fold": i}
            for candidate in (portable, oracle, incomplete)
            for i in range(4)
        ]
        rows[4]["round_score_proxy"] = 0.8
        rows[-1]["benchmark_exped_share"] = 0.3
        ranking = rank_candidates(
            pd.DataFrame(rows), [portable, oracle, incomplete], 4, "round_score_proxy"
        )
        self.assertEqual(ranking.loc[ranking["eligible"], "candidate"].tolist(), ["portable"])
        with self.assertRaisesRegex(ValueError, "cannot be deployed"):
            candidate_predictor(oracle, {}, "exped")

    def test_finalists_skip_repeated_constituents_and_highly_correlated_predictions(self):
        ridge = single("ridge")
        hedged = Candidate("hedged", ridge.weights, 0.25)
        blend = Candidate("blend", (("ridge", 0.9), ("tree", 0.1)))
        tree = single("tree")
        ranking = pd.DataFrame(
            {"candidate": ["ridge", "hedged", "blend", "tree"], "eligible": True}
        )
        correlations = {"ridge": {}, "hedged": {}, "blend": {"ridge": 0.99}, "tree": {"ridge": 0.3}}
        portfolio = choose_finalists(ranking, [ridge, hedged, blend, tree], correlations, 3)
        self.assertEqual(list(portfolio), ["ridge", "tree"])


if __name__ == "__main__":
    unittest.main()
