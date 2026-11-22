import contextlib
import io
import os
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from everestapi import EverestError

from hackathon_setup import (
    EVENT_URL,
    download_training,
    make_client,
    orient,
    report_model_runtime,
)


class SetupTests(unittest.TestCase):
    def test_schema_merges_the_primary_target_and_verbose_features(self):
        client = Mock(base_url=EVENT_URL)
        client.get_started.return_value = {
            "mode": "diagnostics_hackathon",
            "cadence": {"phase": "build"},
            "event_staking": {"money_event": False},
        }
        client.get_dataset_schema.side_effect = [
            {"primary_target": "graded", "feature_encoding": {"missing": -99}},
            {"primary_target": None, "feature_sets": {"all": ["amber", "blue"]}},
        ]
        client.explain_scoring.return_value = {"weights": {"score_multiple_constant": 0.1}}
        with contextlib.redirect_stdout(io.StringIO()):
            schema, scoring = orient(client)
        self.assertEqual(schema["primary_target"], "graded")
        self.assertEqual(schema["feature_sets"]["all"], ["amber", "blue"])
        self.assertEqual(schema["feature_encoding"]["missing"], -99)
        self.assertEqual(scoring["weights"]["score_multiple_constant"], 0.1)
        self.assertEqual(
            [call.kwargs for call in client.get_dataset_schema.call_args_list],
            [{}, {"verbose": True}],
        )
        client.get_event_staking.assert_not_called()

    def test_tournament_account_stops_before_schema(self):
        client = Mock()
        client.get_started.return_value = {"mode": "tournament", "cadence": {"phase": "build"}}
        with self.assertRaisesRegex(RuntimeError, "hackathon account"):
            orient(client)
        client.get_dataset_schema.assert_not_called()

    def test_configured_key_is_never_passed_to_an_unexpected_host(self):
        with (
            patch("hackathon_setup.load_dotenv"),
            patch.dict(
                os.environ,
                {"EIQ_API_KEY": "mock-only", "EIQ_BASE_URL": "https://example.invalid"},
                clear=True,
            ),
            patch("hackathon_setup.EverestAPI") as constructor,
            self.assertRaisesRegex(RuntimeError, "official hackathon URL"),
        ):
            make_client()
        constructor.assert_not_called()

    def test_status_reports_matched_and_mismatched_runtime_pins_without_auth(self):
        pin_url = f"{EVENT_URL}/fixture.txt"
        current = sys.version_info
        response = httpx.Response(
            200,
            text=f"cloudpickle=={version('cloudpickle')}\nnumpy==0.0.0\n",
            request=httpx.Request("GET", pin_url),
        )
        started = {
            "model_python_versions": {"library_pins": {f"{current.major}.{current.minor}": pin_url}}
        }
        output = io.StringIO()
        with (
            patch("hackathon_setup.httpx.get", return_value=response) as get,
            contextlib.redirect_stdout(output),
        ):
            report_model_runtime(started, EVENT_URL)
        self.assertIn("Matched 1 of 2", output.getvalue())
        self.assertIn("Sandbox version differs for numpy", output.getvalue())
        self.assertNotIn("headers", get.call_args.kwargs)

    def test_missing_training_benchmark_leaves_no_stale_reference(self):
        client = Mock()
        client.download_dataset.side_effect = lambda **args: Path(args["output_path"]).touch()
        client.download_benchmark.side_effect = EverestError(404, "fixture")
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            (directory / "train_benchmark.parquet").touch()
            download_training(client, {}, {}, directory)
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"train.parquet", "schema.json", "scoring.json"},
            )
        self.assertEqual(client.download_dataset.call_args.kwargs["split"], "train")
        self.assertEqual(client.download_benchmark.call_args.args, ("futures", "train"))
