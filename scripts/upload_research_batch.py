"""Prepare and upload groups of prevalidated Everesteer models."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import httpx
from everestapi import EverestError

import scripts.run_round as run_round
from scripts.everesteer_batch import BatchError, BatchMintRefused, batch_item, submit_batch
from hackathon_setup import ROOT, make_client, require_event_account

RECEIPT_FIELDS = ("upload_id", "status", "coalesced", "uploads_remaining")
ACCEPTED_STATES = {
    "accepted",
    "already_finalized",
    "complete",
    "completed",
    "done",
    "finalized",
    "nominated",
    "pending",
    "queued",
    "running",
    "submitted",
    "succeeded",
}


def _identity(parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _windows(client, report):
    current = client.get_started()
    require_event_account(current)
    run_round.check_window(
        current,
        report["lane"],
        report["open_window"],
        submitting=True,
        event_id=report["event_id"],
    )
    report["uploads_remaining"] = current.get("uploads_remaining")
    return current


def _receipt(client, outcome):
    nested = outcome.get("result") or outcome.get("receipt")
    if isinstance(nested, dict):
        outcome = {**outcome, **nested}
    upload_id = outcome.get("upload_id")
    if not upload_id:
        return None
    if outcome.get("status") not in ACCEPTED_STATES:
        try:
            remote = client.get_diagnostics_run(upload_id)
        except (EverestError, httpx.HTTPError):
            return None
        if remote.get("status") not in ACCEPTED_STATES | {"failed", "cancelled"}:
            return None
        outcome = {**outcome, "status": remote["status"]}
    return {key: outcome[key] for key in RECEIPT_FIELDS if key in outcome}


def _prepare_group(args, client, path, report, names, known_slots, batch_id):
    lane, window = report["lane"], report["open_window"]
    hosted_lane = "round" if lane == "live" else "practice"
    pending = defaultdict(list)
    for name in names:
        record = report["models"][name]
        if record["state"] == "duplicate_predictions":
            continue
        predictions, artifact = (
            path.parent / record["predictions"],
            args.artifacts / record["artifact"],
        )
        if (
            run_round.file_sha256(predictions) != record["predictions_sha256"]
            or run_round.file_sha256(artifact) != record["artifact_sha256"]
        ):
            raise RuntimeError(f"Prepared files changed for {name}. Generate fresh predictions.")

        slot = record["slot_name"]
        model_id = known_slots.get(slot)
        if model_id is None:
            current = _windows(client, report)
            run_round.check_budget(current, lane)
            model_id = client.create_model(name=slot)["id"]
            known_slots[slot] = model_id
        attempt_id = _identity(
            [
                report["event_id"],
                lane,
                window,
                model_id,
                report["target"],
                record["prediction_identity"],
            ]
        )
        item = batch_item(
            model_id,
            predictions,
            artifact,
            record["python_version"],
            lane=hosted_lane,
            window=window,
            model_sha256=record["artifact_sha256"],
        )
        record.update(
            model_id=model_id,
            attempt_id=attempt_id,
            client_label=f"ev.hack/{attempt_id}",
            batch_id=batch_id,
            batch_idempotency_key=item["idempotency_key"],
        )
        reservation = {**record, "state": "submitting", "report_path": str(path)}
        previous = run_round.reserve_attempt(args.ledger, attempt_id, reservation)
        if previous:
            if previous.get("receipt", {}).get("upload_id"):
                record.update(state="duplicate_skipped", receipt=previous["receipt"])
                run_round.refresh_scoring(client, record)
                print(f"{name}: upload {record['receipt']['upload_id']} already recorded.")
                continue
            if previous.get("batch_idempotency_key") != item["idempotency_key"]:
                record.update(state="review_required", previous_report=previous["report_path"])
                print(f"{name}: prior attempt needs reconciliation. See {previous['report_path']}.")
                continue
            record["batch_id"] = previous["batch_id"]
        record["state"] = "submitting"
        run_round.write_report(path, report)
        pending[record["batch_id"]].append((name, item, previous is None))
    run_round.write_report(path, report)
    return pending


def _block_declined_group(args, path, report, group, *, no_recipes):
    for name, _, is_new in group:
        record = report["models"][name]
        if is_new and no_recipes:
            run_round.release_attempt(args.ledger, record["attempt_id"])
            record.update(state="blocked", error="Hosted MCP refused the initial recipe request.")
        else:
            record.update(state="review_required", error="No confirmed upload outcome.")
    run_round.write_report(path, report)


def _submit_group(args, client, path, report, group, batch_id):
    lane = report["lane"]
    hosted_lane = "round" if lane == "live" else "practice"
    minting = True

    def guard():
        nonlocal minting
        current = _windows(client, report)
        if minting and lane == "live":
            run_round.check_budget(current, lane)
            remaining = current["uploads_remaining"]
            if remaining is not None and len(group) > remaining:
                raise RuntimeError(f"{remaining} scored uploads remain. Request a smaller group.")
        minting = False

    try:
        result = submit_batch(
            client, [item for _, item, _ in group], lane=hosted_lane, batch_id=batch_id, guard=guard
        )
        outcomes = {item["index"]: item for item in result["items"]}
    except BatchMintRefused:
        try:
            progress = client.get_upload_progress(batch_id=batch_id)
            no_recipes = progress.get("total") == 0 and progress.get("items") == []
        except (EverestError, httpx.HTTPError):
            no_recipes = False
        _block_declined_group(args, path, report, group, no_recipes=no_recipes)
        raise
    except (BatchError, EverestError, httpx.HTTPError, RuntimeError) as exc:
        status = (
            str(exc) if isinstance(exc, (BatchError, RuntimeError)) else run_round.safe_error(exc)
        )
        print(f"Batch {batch_id}: {status}", file=sys.stderr)
        outcomes = {}

    success = True
    for i, (name, _, _) in enumerate(group):
        record = report["models"][name]
        receipt = _receipt(client, outcomes.get(i, {}))
        if receipt is None:
            record.update(
                state="review_required", error="No confirmed receipt. Retry the same batch."
            )
            success = False
        else:
            record.update(state="submitted", receipt=receipt, scoring_status=receipt["status"])
            if receipt["status"] in {"failed", "cancelled"}:
                success = False
            print(f"{name}: upload {receipt['upload_id']} status {receipt['status']}.")
        run_round.update_attempt(
            args.ledger, record["attempt_id"], {**record, "report_path": str(path)}
        )
        run_round.write_report(path, report)
    return success


def _submit_prepared(args, client, path):
    report = json.loads(path.read_text())
    report["submit_requested"] = True
    run_round.write_report(path, report)
    _windows(client, report)
    known_slots = {
        label: model["id"]
        for model in client.get_models().get("models", [])
        for label in (model.get("client_label"), model.get("name"))
        if label and model.get("id")
    }
    names = list(report["models"])
    success = True
    for start in range(0, len(names), 25):
        group_names = names[start : start + 25]
        batch_id = _identity(
            [
                report["event_id"],
                report["lane"],
                report["open_window"],
                report["manifest_sha256"],
                report["row_ids_sha256"],
                group_names,
            ]
        )[:32]
        pending = _prepare_group(args, client, path, report, group_names, known_slots, batch_id)
        groups = iter(pending.items())
        for existing_id, group in groups:
            try:
                sent = _submit_group(args, client, path, report, group, existing_id)
                success = sent and success
            except BatchMintRefused:
                for _, unattempted in groups:
                    _block_declined_group(args, path, report, unattempted, no_recipes=True)
                raise
    return success and all(
        record["state"] != "review_required" for record in report["models"].values()
    )


def run(args, client):
    if args.submit and args.lane == "live" and not args.window:
        raise RuntimeError("Live submissions require --window from the event status.")
    options = SimpleNamespace(**{**vars(args), "submit": False})
    path = run_round.run(options, client)
    if not args.submit:
        return True
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.ledger.with_suffix(args.ledger.suffix + ".batch.lock")
    with lock_path.open("a") as lock:
        # Serializing batch writers makes an initial mint refusal safe to unwind.
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _submit_prepared(args, client, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("practice", "live"))
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--candidate", action="append", help="Select a named research candidate."
    )
    selection.add_argument("--model", action="append", help="Select a named baseline model.")
    parser.add_argument("--window", help="Expected round, required with live --submit.")
    parser.add_argument("--ledger", type=Path, default=run_round.DEFAULT_LEDGER)
    parser.add_argument(
        "--submit", action="store_true", help="Register models and upload the batch."
    )
    args = parser.parse_args()
    if args.window and args.lane != "live":
        parser.error("--window applies only to live uploads.")
    try:
        with make_client() as client:
            return 0 if run(args, client) else 1
    except (EverestError, httpx.HTTPError) as exc:
        print(f"{args.lane} failed. {run_round.safe_error(exc)}", file=sys.stderr)
    except (BatchError, OSError, ValueError, RuntimeError) as exc:
        print(f"{args.lane} failed. {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
