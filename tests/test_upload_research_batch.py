import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import run_round
import upload_research_batch as batch
from everesteer_batch import BatchError, BatchMintRefused


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "runs" / "test" / "report.json"
        self.path.parent.mkdir(parents=True)
        self.args = Namespace(
            lane="live",
            window="round_2",
            submit=True,
            artifacts=self.root,
            candidate=["one"],
            model=None,
            ledger=self.root / "ledger.sqlite3",
        )
        self.client = Mock()
        self.client.get_started.return_value = {
            "mode": "diagnostics_hackathon",
            "event_id": "event",
            "uploads_remaining": 100,
            "cadence": {"open_window": "round_2", "intake_fenced": False},
        }
        self.client.get_models.return_value = {"models": []}
        self.client.get_upload_progress.return_value = {"total": 0, "items": []}
        self.client.create_model.side_effect = lambda name: {"id": f"id-{name}"}

    def make_report(self, names):
        records = {}
        for name in names:
            prediction = self.path.parent / f"{name}.parquet"
            artifact = self.root / f"{name}.pkl"
            prediction.write_bytes(name.encode())
            artifact.write_bytes(name.encode())
            records[name] = {
                "state": "prepared",
                "predictions": prediction.name,
                "predictions_sha256": run_round.file_sha256(prediction),
                "prediction_identity": f"semantic-{name}",
                "artifact": artifact.name,
                "artifact_sha256": run_round.file_sha256(artifact),
                "python_version": "3.12",
                "slot_name": f"slot-{name}",
            }
        run_round.write_report(
            self.path,
            {
                "event_id": "event",
                "open_window": "round_2",
                "lane": "live",
                "target": "graded",
                "manifest_sha256": "manifest",
                "row_ids_sha256": "rows",
                "models": records,
            },
        )

    def invoke(self, submission):
        with (
            patch.object(batch.run_round, "run", return_value=self.path) as prep,
            patch.object(batch, "submit_batch", side_effect=submission) as upload,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            okay = batch.run(self.args, self.client)
        self.assertFalse(prep.call_args.args[0].submit)
        self.assertEqual(prep.call_args.args[0].candidate, self.args.candidate)
        return okay, upload

    @staticmethod
    def accepted(client, items, *, lane, batch_id, guard):
        guard()
        guard()
        assert lane == "round"
        return {
            "all_succeeded": True,
            "items": [
                {"index": index, "status": "pending", "upload_id": f"run-{item['model_id']}"}
                for index, item in enumerate(items)
            ],
        }

    def test_batches_at_most_25_and_stores_each_receipt(self):
        self.make_report([f"m{idx}" for idx in range(26)])
        okay, upload = self.invoke(self.accepted)
        self.assertTrue(okay)
        self.assertEqual([len(call.args[1]) for call in upload.call_args_list], [25, 1])
        report = json.loads(self.path.read_text())
        self.assertTrue(report["submit_requested"])
        self.assertEqual(report["models"]["m0"]["receipt"]["upload_id"], "run-id-slot-m0")
        self.assertEqual(report["models"]["m25"]["state"], "submitted")

    def test_pending_batch_keeps_its_idempotency_key_and_completed_one_is_skipped(self):
        self.make_report(["one"])
        okay, first = self.invoke(BatchError("Unknown response"))
        self.assertFalse(okay)
        record = json.loads(self.path.read_text())["models"]["one"]
        self.assertEqual(record["state"], "review_required")

        self.make_report(["one"])
        okay, retry = self.invoke(self.accepted)
        self.assertTrue(okay)
        self.assertEqual(first.call_args.args[1], retry.call_args.args[1])
        self.assertEqual(first.call_args.kwargs["batch_id"], retry.call_args.kwargs["batch_id"])

        self.make_report(["one"])
        okay, third = self.invoke(self.accepted)
        self.assertTrue(okay)
        third.assert_not_called()
        record = json.loads(self.path.read_text())["models"]["one"]
        self.assertEqual(record["state"], "duplicate_skipped")

    def test_mint_denial_only_releases_reservations_created_by_current_run(self):
        self.make_report(["one"])
        okay, _ = self.invoke(BatchError("Unknown response"))
        self.assertFalse(okay)
        self.make_report(["one", "two"])
        with self.assertRaises(BatchMintRefused):
            self.invoke(BatchMintRefused("MCP tools/call returned HTTP 403."))
        models = json.loads(self.path.read_text())["models"]
        self.assertEqual(models["one"]["state"], "review_required")
        self.assertEqual(models["two"]["state"], "blocked")
        with contextlib.closing(sqlite3.connect(self.args.ledger)) as db:
            (remaining,) = db.execute("SELECT attempt_id FROM uploads").fetchone()
        self.assertEqual(remaining, models["one"]["attempt_id"])

    def test_mint_denial_keeps_reservation_when_server_has_a_recipe(self):
        self.make_report(["one"])
        self.client.get_upload_progress.return_value = {"total": 1, "items": [{}]}
        with self.assertRaises(BatchMintRefused):
            self.invoke(BatchMintRefused("MCP tools/call returned HTTP 403."))
        record = json.loads(self.path.read_text())["models"]["one"]
        self.assertEqual(record["state"], "review_required")
        with contextlib.closing(sqlite3.connect(self.args.ledger)) as db:
            (remaining,) = db.execute("SELECT attempt_id FROM uploads").fetchone()
        self.assertEqual(remaining, record["attempt_id"])


if __name__ == "__main__":
    unittest.main()
