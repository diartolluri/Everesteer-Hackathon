import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from everestapi import EverestAPI

from scripts.event_status import event_snapshot, read_fold_scores, stake_scenarios


class StatusTests(unittest.TestCase):
    def test_offline_status_labels_synthetic_artifacts_and_rejects_their_fold_scores(self):
        release = sys.version_info
        manifest = {
            "synthetic": True,
            "fold_metrics_path": "folds.csv",
            "models": {
                "lightgbm": {
                    "path": "model.pkl",
                    "python_version": f"{release.major}.{release.minor}",
                }
            },
        }
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            (directory / "model.pkl").touch()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            snapshot = event_snapshot(None, directory, staking_scenarios=True)
            self.assertFalse(snapshot["online"])
            self.assertEqual(snapshot["models"]["lightgbm"]["state"], "synthetic only")
            self.assertEqual(read_fold_scores(directory, manifest), {})
            self.assertEqual(snapshot["scenarios"][0]["allocations"], {})

    def test_status_checks_event_boards_without_platform_writes(self):
        client = Mock(spec=EverestAPI)
        client.get_started.return_value = {
            "mode": "diagnostics_hackathon",
            "uploads_remaining": 7,
            "cadence": {"open_window": "round_2"},
            "event_staking": {"money_event": False},
        }
        client.get_diagnostics_standings.return_value = {}
        client.get_diagnostics_leaderboard.return_value = {"rank_metric": "round_score"}
        with tempfile.TemporaryDirectory() as scratch:
            snapshot = event_snapshot(client, Path(scratch), window="round_1")
        self.assertEqual(snapshot["uploads_remaining"], 7)
        self.assertEqual(snapshot["board"]["rank_metric"], "round_score")
        client.get_diagnostics_leaderboard.assert_called_once_with(scoring_window="round_1")
        client.get_event_staking.assert_not_called()
        client.create_model.assert_not_called()
        client.submit_event_predictions.assert_not_called()
        client.submit_validation_diagnostics.assert_not_called()
        client.set_stake_allocation.assert_not_called()
        client.withdraw_stake_allocation.assert_not_called()

    def test_stake_scenarios_use_verified_uploads_fold_evidence_and_contract_units(self):
        position = {
            "staking_armed": True,
            "stake_window_open": True,
            "draft_window": "round_2",
            "forwarder_balance_micro": 20_000_000,
            "min_stake_micro": 1_000_000,
            "max_slots": 2,
            "windows": [],
        }
        models = {"ridge": {"sha256": "a"}, "lightgbm": {"sha256": "b"}}
        receipts = [
            {
                "model": name,
                "model_id": name + "-remote",
                "lane": "live",
                "window": "round_2",
                "state": "pending",
                "platform_verified": True,
                "artifact_sha256": entry["sha256"],
            }
            for name, entry in models.items()
        ]
        folds = {"ridge": {"f1": 0.03, "f2": 0.04}, "lightgbm": {"f1": 0.01, "f2": 0.03}}
        scenarios, _ = stake_scenarios(
            position, models, receipts, folds, {"ridge": {"lightgbm": 0.5}}
        )
        self.assertEqual(scenarios[0]["allocations"], {})
        self.assertEqual(scenarios[1]["allocations"], {"ridge-remote": 2_000_000})
        self.assertEqual(
            scenarios[2]["allocations"], {"ridge-remote": 1_000_000, "lightgbm-remote": 1_000_000}
        )
        position["windows"] = [{"allocations": [{"locked_at": "pending"}]}]
        blocked, _ = stake_scenarios(position, models, receipts, folds, {})
        self.assertEqual([scenario["allocations"] for scenario in blocked], [{}])
