"""Three-step submission through Everesteer's hosted MCP batch tools."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path
from urllib.parse import urlsplit

import httpx

TOOLS = {
    "practice": "eiq_submit_diagnostics_batch",
    "round": "eiq_submit_event_predictions_batch",
}
EVENT_URL = "https://hackathon.everesteer.ai"


class BatchError(RuntimeError):
    """A failed batch operation whose details are safe to log."""


class BatchMintRefused(BatchError):
    """The hosted MCP rejected the initial recipe request with HTTP 401 or 403."""


class _BatchHttpError(BatchError):
    def __init__(self, method, status_code):
        self.status_code = status_code
        super().__init__(f"MCP {method} returned HTTP {status_code}.")


def _digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def batch_item(model_id, predictions, model, python_version, *, lane, window, model_sha256=None):
    """Describe local files and give these exact bytes a stable upload identity."""
    predictions, model = Path(predictions), Path(model)
    pred_sha = _digest(predictions)
    model_sha256 = model_sha256 or _digest(model)
    identity = f"{lane}:{window}:{model_id}:{pred_sha}:{model_sha256}".encode()
    return {
        "model_id": model_id,
        "file_path": str(predictions),
        "model_pkl_path": str(model),
        "model_pkl_python_version": python_version,
        "model_pkl_sha256": model_sha256,
        "model_pkl_size_bytes": model.stat().st_size,
        "predictions_sha256": pred_sha,
        "predictions_size_bytes": predictions.stat().st_size,
        "idempotency_key": hashlib.sha256(identity).hexdigest(),
    }


class HostedBatch:
    def __init__(self, client, storage):
        self.client = client
        self.storage = storage
        self.requests = count(1)
        self.protocol_headers = {}

    def _rpc(self, method, params=None, *, notification=False):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notification:
            message["id"] = next(self.requests)
        try:
            response = self.client.post("/mcp", json=message, headers=self.protocol_headers)
            response.raise_for_status()
            if notification:
                return {}, response.headers
            if "text/event-stream" in response.headers.get("content-type", ""):
                lines = (
                    line[5:].lstrip()
                    for line in response.text.splitlines()
                    if line.startswith("data:")
                )
                replies = (json.loads(line) for line in lines)
                reply = next(obj for obj in replies if obj.get("id") == message["id"])
            else:
                reply = response.json()
        except httpx.HTTPStatusError as exc:
            raise _BatchHttpError(method, exc.response.status_code) from None
        except (httpx.RequestError, StopIteration, ValueError):
            raise BatchError(f"MCP {method} returned no usable response.") from None
        if "error" in reply or "result" not in reply:
            raise BatchError(f"MCP {method} returned a JSON-RPC error.")
        return reply["result"], response.headers

    def initialize(self):
        result, headers = self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "Everesteer local batch runner", "version": "1"},
            },
        )
        self.protocol_headers["MCP-Protocol-Version"] = result["protocolVersion"]
        if headers.get("MCP-Session-Id"):
            self.protocol_headers["MCP-Session-Id"] = headers["MCP-Session-Id"]
        self._rpc("notifications/initialized", notification=True)

    def _call_tool(self, name, arguments):
        result, _ = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise BatchError(f"{name} refused the batch.")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        for part in result.get("content", []):
            if part.get("type") == "text":
                try:
                    payload = json.loads(part["text"])
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    return payload
        raise BatchError(f"{name} returned no batch results.")

    def _transfer(self, recipe, path):
        url = recipe["url"]
        destination = urlsplit(url)
        if destination.scheme != "https" or not destination.hostname or destination.username:
            raise BatchError("The storage recipe has an invalid destination.")
        method = recipe.get("method", "POST").upper()
        with Path(path).open("rb") as handle:
            if method == "PUT":
                response = self.storage.put(url, headers=recipe.get("headers", {}), content=handle)
            elif method == "POST":
                parts = [(name, (None, value)) for name, value in recipe["fields"].items()]
                parts.append(("file", (Path(path).name, handle, "application/octet-stream")))
                response = self.storage.post(url, files=parts)
            else:
                raise BatchError("The storage recipe has an unsupported method.")
        if not response.is_success:
            raise BatchError(f"Presigned storage upload returned HTTP {response.status_code}.")

    def _upload_item(self, item, recipe):
        for key, path in (("model_pkl", "model_pkl_path"), ("predictions", "file_path")):
            if recipe.get(key):
                self._transfer(recipe[key], item[path])

    def submit(self, items, *, lane, batch_id, guard):
        """Allocate, upload, then finalize each successful item. Guard both tool calls."""
        if not 1 <= len(items) <= 25:
            raise ValueError("Everesteer batches accept 1–25 items.")
        name = TOOLS[lane]
        args = {"items": items, "batch_id": batch_id}
        guard()
        try:
            recipes = self._call_tool(name, args)
        except _BatchHttpError as exc:
            if exc.status_code in {401, 403}:
                raise BatchMintRefused(str(exc)) from None
            raise
        outcomes = {}
        pending = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for recipe in recipes.get("items", []):
                index = recipe.get("index")
                if not isinstance(index, int) or not 0 <= index < len(items):
                    raise BatchError("The batch recipe returned an invalid item index.")
                outcome = {
                    key: recipe[key]
                    for key in ("index", "model_id", "upload_id", "status", "error")
                    if key in recipe
                }
                outcomes[index] = outcome
                if recipe.get("upload_id") and (
                    recipe.get("model_pkl") or recipe.get("predictions")
                ):
                    pending[index] = pool.submit(self._upload_item, items[index], recipe)
            final_indexes = []
            for index, future in pending.items():
                try:
                    future.result()
                except (BatchError, OSError, KeyError, httpx.HTTPError):
                    outcomes[index].update(status="error", error="Presigned transfer failed.")
                else:
                    final_indexes.append(index)

        if final_indexes:
            final_items = [
                {**items[index], "upload_id": outcomes[index]["upload_id"]}
                for index in final_indexes
            ]
            guard()
            final = self._call_tool(name, {**args, "items": final_items, "finalize": True})
            for index in final_indexes:
                outcomes[index].update(status="error", error="No finalization outcome.")
            for item in final.get("items", []):
                index = item.get("index")
                if not isinstance(index, int) or not 0 <= index < len(final_indexes):
                    raise BatchError("The finalization response returned an invalid item index.")
                original = final_indexes[index]
                outcomes[original] = {**item, "index": original}

        return {
            "items": [
                outcomes.get(
                    index, {"index": index, "status": "error", "error": "No batch outcome."}
                )
                for index in range(len(items))
            ],
            "batch_id": recipes.get("batch_id") or batch_id,
            "all_succeeded": (
                bool(recipes.get("all_succeeded"))
                and len(outcomes) == len(items)
                and all(item.get("status") != "error" for item in outcomes.values())
                and (not final_indexes or bool(final.get("all_succeeded")))
            ),
        }


def submit_batch(api, items, *, lane, batch_id, guard):
    """Use the account API key only for the official MCP host."""
    if api.base_url.rstrip("/") != EVENT_URL:
        raise ValueError("Hosted event batches require the official Everesteer hackathon host.")
    headers = {"X-API-Key": api.api_key, "Accept": "application/json, text/event-stream"}
    with (
        httpx.Client(base_url=api.base_url, headers=headers, timeout=30) as client,
        httpx.Client(timeout=60) as storage,
    ):
        hosted = HostedBatch(client, storage)
        hosted.initialize()
        return hosted.submit(items, lane=lane, batch_id=batch_id, guard=guard)
