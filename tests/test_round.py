import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import cloudpickle
import numpy as np
import pandas as pd
from everestapi import EverestAPI

from scripts.run_round import reserve_attempt, run


def event(window="round_1", remaining=4):
    return {
        "mode": "diagnostics_hackathon",
        "uploads_remaining": remaining,
        "cadence": {
            "phase": window or "build",
            "open_window": window,
            "intake_fenced": window is None,
        },
    }


class RoundTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.artifacts = self.root / "models"
        self.artifacts.mkdir()
        self.frame = pd.DataFrame(
            {
                "amber": [0, 2, 1, 3],
                "exped": ["era_001"] * 4,
                "graded_target": [np.nan] * 4,
            },
            index=pd.Index(["q9", "b2", "v5", "s4"], name="id"),
        )
        self.client = Mock(spec=EverestAPI)
        self.client.get_started.return_value = event()
        self.client.get_dataset_schema.return_value = {"primary_target": "graded_target"}
        self.client.download_dataset.side_effect = self.download
        self.client.create_model.return_value = {"id": "fixture-model"}
        self.client.get_models.return_value = {
            "models": [
                {"id": "fixture-model", "client_label": "hackathon-01", "name": "hackathon-01"}
            ]
        }
        self.client.get_diagnostics_run.return_value = {"status": "done"}
        self.client.submit_event_predictions.return_value = {
            "upload_id": "fixture-live",
            "status": "pending",
            "uploads_remaining": 3,
        }
        self.client.submit_validation_diagnostics.return_value = {
            "upload_id": "fixture-practice",
            "status": "pending",
            "uploads_remaining": 4,
        }
        self.save_artifact(lambda scored: scored[["amber"]].rename(columns={"amber": "prediction"}))
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def download(self, *, split, output_path):
        frame = self.frame.assign(data_type=split)
        if split == "validation":
            frame.index = pd.Index(["p4", "f5", "k9", "e2"], name="id")
        frame.to_parquet(output_path)
        return output_path

    def save_artifact(self, predict, *, synthetic=False):
        (self.artifacts / "lightgbm.pkl").write_bytes(cloudpickle.dumps(predict))
        release = sys.version_info
        manifest = {
            "synthetic": synthetic,
            "primary_target": "graded_target",
            "exped_column": "exped",
            "feature_columns": ["amber"],
            "models": {
                "lightgbm": {
                    "path": "lightgbm.pkl",
                    "python_version": f"{release.major}.{release.minor}",
                }
            },
        }
        (self.artifacts / "manifest.json").write_text(json.dumps(manifest))

    def args(self, lane="live", *, submit=False, window=None, candidate=None):
        return Namespace(
            lane=lane,
            artifacts=self.artifacts,
            model=None,
            candidate=candidate,
            submit=submit,
            window=window,
            ledger=self.root / "submissions.sqlite3",
        )

    def research_manifest(self):
        path = self.artifacts / "manifest.json"
        manifest = json.loads(path.read_text())
        artifact = manifest["models"].pop("lightgbm")
        artifact["sha256"] = hashlib.sha256(
            (self.artifacts / artifact["path"]).read_bytes()
        ).hexdigest()
        alternate = cloudpickle.dumps(
            lambda scored: scored[["amber"]].sub(1).pow(2).rename(columns={"amber": "prediction"})
        )
        (self.artifacts / "alternate.pkl").write_bytes(alternate)
        manifest["provenance"] = "walk_forward_research"
        manifest["models"] = {
            "selected-blend": {**artifact, "prevalidated": True},
            "alternate": {
                **artifact,
                "path": "alternate.pkl",
                "sha256": hashlib.sha256(alternate).hexdigest(),
                "prevalidated": True,
            },
            "untested": artifact,
        }
        path.write_text(json.dumps(manifest))

    def reports(self, lane="live"):
        paths = sorted((self.artifacts / "runs").glob(f"{lane}-*/report.json"))
        return [(path, json.loads(path.read_text())) for path in paths]

    def assert_no_writes(self):
        self.client.create_model.assert_not_called()
        self.client.submit_event_predictions.assert_not_called()
        self.client.submit_validation_diagnostics.assert_not_called()

    def test_dry_run_uses_fresh_ids_and_keeps_opaque_order(self):
        run(self.args(), self.client)
        self.assert_no_writes()
        self.assertFalse(self.args().ledger.exists())
        self.assertEqual(self.client.download_dataset.call_args.kwargs["split"], "live")
        path, report = self.reports()[-1]
        record = report["models"]["lightgbm"]
        payload = pd.read_parquet(path.parent / record["predictions"])
        self.assertEqual(list(payload.columns), ["id", "prediction"])
        self.assertEqual(payload.id.tolist(), self.frame.index.tolist())
        for digest in (
            report["dataset_sha256"],
            report["row_ids_sha256"],
            record["predictions_sha256"],
            record["artifact_sha256"],
        ):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_practice_uses_validation_ids_and_its_own_lane_even_during_a_live_round(self):
        run(self.args("practice", submit=True), self.client)
        self.client.submit_event_predictions.assert_not_called()
        self.assertEqual(self.client.download_dataset.call_args.kwargs["split"], "validation")
        upload = self.client.submit_validation_diagnostics.call_args
        self.assertEqual(pd.read_parquet(upload.args[1]).id.tolist(), ["p4", "f5", "k9", "e2"])

    def test_round_turnover_after_predictions_blocks_live_upload(self):
        self.client.get_started.side_effect = [event(), event("round_2")]
        with self.assertRaisesRegex(RuntimeError, "changed"):
            run(self.args(submit=True, window="round_1"), self.client)
        self.assert_no_writes()

    def test_turnover_in_final_submit_guard_blocks_live_upload(self):
        self.client.get_started.side_effect = [event(), event(), event("round_2")]
        with self.assertRaisesRegex(RuntimeError, "changed"):
            run(self.args(submit=True, window="round_1"), self.client)
        self.client.submit_event_predictions.assert_not_called()
        self.client.submit_validation_diagnostics.assert_not_called()

    def test_live_submission_requires_an_approved_window(self):
        with self.assertRaisesRegex(RuntimeError, "window"):
            run(self.args(submit=True), self.client)
        self.assert_no_writes()
        self.client.download_dataset.assert_not_called()

    def test_exhausted_budget_blocks_upload(self):
        self.client.get_started.return_value = event(remaining=0)
        with self.assertRaisesRegex(RuntimeError, "allowance"):
            run(self.args(submit=True, window="round_1"), self.client)
        self.assert_no_writes()

    def test_known_receipt_is_reused_across_restarts(self):
        args = self.args(submit=True, window="round_1")
        run(args, self.client)
        run(args, self.client)
        self.client.submit_event_predictions.assert_called_once()
        self.assertEqual(
            self.reports()[-1][1]["models"]["lightgbm"]["receipt"]["upload_id"], "fixture-live"
        )
        self.client.submit_validation_diagnostics.assert_not_called()

    def test_unknown_receipt_blocks_retry_after_restart(self):
        self.client.submit_event_predictions.side_effect = TimeoutError("simulated lost response")
        args = self.args(submit=True, window="round_1")
        with self.assertRaisesRegex(RuntimeError, "review"):
            run(args, self.client)
        with self.assertRaisesRegex(RuntimeError, "confirmed receipt|review"):
            run(args, self.client)
        self.client.submit_event_predictions.assert_called_once()

    def test_synthetic_artifact_stops_before_download(self):
        self.save_artifact(lambda scored: scored, synthetic=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            run(self.args(), self.client)
        self.client.download_dataset.assert_not_called()
        self.assert_no_writes()

    def test_practice_only_experiment_cannot_enter_a_scored_round(self):
        path = self.artifacts / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["practice_only"] = True
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "practice board only"):
            run(self.args(submit=True, window="round_1"), self.client)
        self.client.download_dataset.assert_not_called()
        self.assert_no_writes()

    def test_round_consumes_only_the_named_prevalidated_candidate(self):
        self.research_manifest()
        run(self.args(submit=True, window="round_1", candidate="selected-blend"), self.client)
        self.client.submit_event_predictions.assert_called_once()
        self.client.submit_validation_diagnostics.assert_not_called()
        _, report = self.reports()[-1]
        self.assertEqual(list(report["models"]), ["selected-blend"])

    def test_round_submits_distinct_candidates_in_requested_order(self):
        self.research_manifest()
        self.client.create_model.side_effect = [{"id": "alternate-id"}, {"id": "blend-id"}]
        self.client.submit_event_predictions.side_effect = [
            {"upload_id": "alternate-upload", "status": "pending", "uploads_remaining": 3},
            {"upload_id": "blend-upload", "status": "pending", "uploads_remaining": 2},
        ]
        args = self.args(
            submit=True,
            window="round_1",
            candidate=["alternate", "selected-blend", "alternate"],
        )
        run(args, self.client)
        self.client.download_dataset.assert_called_once()
        self.client.submit_validation_diagnostics.assert_not_called()
        calls = self.client.submit_event_predictions.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["alternate-id", "blend-id"])
        self.assertEqual(
            [Path(call.args[1]).stem for call in calls], ["alternate", "selected-blend"]
        )
        _, report = self.reports()[-1]
        self.assertEqual(list(report["models"]), ["alternate", "selected-blend"])

    def test_research_manifest_requires_an_explicit_prevalidated_candidate(self):
        self.research_manifest()
        for candidate, error in (
            (None, "--candidate"),
            ("missing", "Unknown research candidate"),
            ("untested", "no verified artifact"),
        ):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, error):
                run(self.args(submit=True, window="round_1", candidate=candidate), self.client)
        self.client.download_dataset.assert_not_called()
        self.assert_no_writes()

    def test_research_candidate_rejects_a_changed_artifact(self):
        self.research_manifest()
        (self.artifacts / "lightgbm.pkl").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            run(self.args(submit=True, window="round_1", candidate="selected-blend"), self.client)
        self.assert_no_writes()

    def test_invalid_predictions_are_never_sent(self):
        for value, error in [(np.inf, "infinite"), (1.0, "constant|variation")]:
            with self.subTest(value=value):
                self.save_artifact(
                    lambda scored, value=value: pd.DataFrame({"p": value}, index=scored.index)
                )
                with self.assertRaisesRegex(ValueError, error):
                    run(self.args(submit=True, window="round_1"), self.client)
        self.assert_no_writes()

    def test_duplicate_ids_are_never_sent(self):
        self.frame.index = pd.Index(["dup"] * len(self.frame), name="id")
        with self.assertRaisesRegex(ValueError, "unique"):
            run(self.args(submit=True, window="round_1"), self.client)
        self.assert_no_writes()

    def test_labeled_data_is_never_sent_to_a_scoring_lane(self):
        self.frame.loc[self.frame.index[0], "graded_target"] = 0.1
        with self.assertRaisesRegex(ValueError, "populated target"):
            run(self.args(submit=True, window="round_1"), self.client)
        self.assert_no_writes()

    def test_concurrent_duplicate_reservations_have_one_winner(self):
        ledger = self.args().ledger
        record = {"state": "submitting"}
        with ThreadPoolExecutor(max_workers=6) as workers:
            results = list(workers.map(lambda _: reserve_attempt(ledger, "same", record), range(6)))
        self.assertEqual(results.count(None), 1)
        self.assertEqual(results.count(record), 5)
