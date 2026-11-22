import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from everesteer_batch import BatchError, BatchMintRefused, HostedBatch, batch_item


class BatchTests(unittest.TestCase):
    def test_http403_before_transfer_is_distinct_from_finalization_refusal(self):
        def response(request):
            if json.loads(request.content)["params"]["arguments"].get("finalize"):
                return httpx.Response(403, text="<html>Finalization refused</html>")
            return httpx.Response(403, text="<html>Initial request refused</html>")

        with httpx.Client(
            base_url="https://platform.test", transport=httpx.MockTransport(response)
        ) as api:
            batch = HostedBatch(api, Mock())
            with self.assertRaises(BatchMintRefused):
                batch.submit(
                    [{"model_id": "one"}], lane="practice", batch_id="batch", guard=lambda: None
                )

            recipe = {
                "items": [
                    {"index": 0, "upload_id": "up", "predictions": {"url": "https://files.test/p"}}
                ]
            }
            original = batch._call_tool
            with (
                patch.object(batch, "_upload_item"),
                patch.object(
                    batch,
                    "_call_tool",
                    side_effect=lambda name, args: (
                        original(name, args) if args.get("finalize") else recipe
                    ),
                ),
            ):
                with self.assertRaises(BatchError) as raised:
                    batch.submit(
                        [{"model_id": "one"}], lane="practice", batch_id="batch", guard=lambda: None
                    )
                self.assertNotIsInstance(raised.exception, BatchMintRefused)

    def test_batch_transfers_without_credentials_and_finalizes_only_successful_items(self):
        requests = []
        guard_calls = []
        model_path = {}

        def platform(request):
            self.assertEqual(request.headers["x-api-key"], "event-key")
            message = json.loads(request.content)
            method = message["method"]
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "initialize":
                result = {"protocolVersion": "2025-06-18"}
            else:
                self.assertEqual(message["params"]["name"], "eiq_submit_event_predictions_batch")
                args = message["params"]["arguments"]
                requests.append(args)
                if args.get("finalize"):
                    self.assertEqual(guard_calls, ["fresh", "fresh"])
                    self.assertEqual(args["items"], [{**model_path, "upload_id": "first"}])
                    payload = {
                        "all_succeeded": True,
                        "items": [{"index": 0, "status": "pending", "upload_id": "first"}],
                    }
                else:
                    payload = {
                        "all_succeeded": True,
                        "items": [
                            {
                                "index": 0,
                                "upload_id": "first",
                                "model_pkl": {
                                    "url": "https://files.test/model",
                                    "method": "PUT",
                                    "headers": {"x-test-digest": "signed"},
                                },
                                "predictions": {
                                    "url": "https://files.test/predictions",
                                    "fields": {"signed-field": "content"},
                                },
                            },
                            {
                                "index": 1,
                                "upload_id": "second",
                                "predictions": {"url": "https://files.test/error", "fields": {}},
                            },
                        ],
                    }
                result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
            reply = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result})
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, text=f"data: {reply}\n\n"
            )

        def storage(request):
            self.assertNotIn("x-api-key", request.headers)
            if request.url.path == "/model":
                self.assertEqual(request.method, "PUT")
                self.assertEqual(request.headers["x-test-digest"], "signed")
                self.assertEqual(request.content, b"model bytes")
            if request.url.path == "/predictions":
                self.assertEqual(request.method, "POST")
                self.assertLess(
                    request.content.index(b"signed-field"), request.content.index(b'name="file"')
                )
                self.assertIn(b"prediction bytes", request.content)
            return httpx.Response(500 if request.url.path == "/error" else 204)

        with tempfile.TemporaryDirectory() as scratch:
            predictions = Path(scratch) / "predictions.parquet"
            model = Path(scratch) / "model.pkl"
            predictions.write_bytes(b"prediction bytes")
            model.write_bytes(b"model bytes")
            model_path.update(
                batch_item("model-id", predictions, model, "3.12", lane="round", window="round_1")
            )
            with (
                httpx.Client(
                    base_url="https://platform.test",
                    headers={"X-API-Key": "event-key"},
                    transport=httpx.MockTransport(platform),
                ) as api,
                httpx.Client(transport=httpx.MockTransport(storage)) as obj,
            ):
                batch = HostedBatch(api, obj)
                batch.initialize()
                result = batch.submit(
                    [model_path, {**model_path, "model_id": "another"}],
                    lane="round",
                    batch_id="batch",
                    guard=lambda: guard_calls.append("fresh"),
                )
        self.assertEqual(len(requests), 2)
        self.assertFalse(result["all_succeeded"])
        self.assertEqual([row["status"] for row in result["items"]], ["pending", "error"])


if __name__ == "__main__":
    unittest.main()
