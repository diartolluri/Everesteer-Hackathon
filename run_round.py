"""Prepare or explicitly upload predictions for one Everesteer lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
import httpx
import numpy as np
import pandas as pd
from everestapi import EverestError
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from hackathon_setup import ROOT, make_client, require_event_account

DEFAULT_LEDGER = ROOT / "artifacts" / "submissions.sqlite3"
BASELINE_SLOTS = {"lightgbm": "hackathon-01", "ridge": "hackathon-02", "rank-blend": "hackathon-03"}


def check_window(started, lane, expected_window=None, *, submitting=False, event_id=None):
    cadence = started.get("cadence") or {}
    if event_id is not None and started.get("event_id") != event_id:
        raise RuntimeError("The event changed while preparing predictions. Rerun for fresh rows.")
    if cadence.get("phase") == "done" and (lane == "live" or submitting):
        raise RuntimeError("The event has ended.")
    if lane == "practice":
        if submitting and cadence.get("diagnostics_maintenance"):
            raise RuntimeError("The practice board is under maintenance.")
        return None

    window = cadence.get("open_window")
    if expected_window is not None and window != expected_window:
        raise RuntimeError(
            f"The open round changed from {expected_window!r} to {window!r}. Rerun for fresh rows."
        )
    if not window or cadence.get("intake_fenced") is not False:
        raise RuntimeError("Live intake is closed. Check event status for the next open round.")
    return window


def check_budget(started, lane):
    if lane != "live":
        return
    if "uploads_remaining" not in started:
        raise RuntimeError("Everesteer did not report the live upload allowance.")
    remaining = started["uploads_remaining"]
    if remaining is not None and (type(remaining) is not int or remaining <= 0):
        raise RuntimeError("The platform reports no usable live upload allowance remaining.")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ordered_ids_sha256(index):
    digest = hashlib.sha256()
    for row_id in index:
        encoded = row_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def prediction_identity(predictions):
    # Canonicalize row order for local deduplication and retain served order in the upload.
    ordered = predictions.sort_values("id")
    digest = hashlib.sha256(bytes.fromhex(ordered_ids_sha256(ordered["id"])))
    values = ordered["prediction"].to_numpy(dtype=">f8", copy=True)
    values[values == 0] = 0  # Treat IEEE negative zero and zero as the same score.
    digest.update(values.tobytes())
    return digest.hexdigest()


def reserve_attempt(path, attempt_id, record):
    """Commit intent before the request so concurrent and restarted runs cannot resend it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as database, database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS uploads (attempt_id TEXT PRIMARY KEY, record TEXT NOT NULL)"
        )
        inserted = database.execute(
            "INSERT INTO uploads VALUES (?, ?) ON CONFLICT(attempt_id) DO NOTHING",
            (attempt_id, json.dumps(record)),
        )
        if inserted.rowcount == 1:
            return None
        (previous,) = database.execute(
            "SELECT record FROM uploads WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
    return json.loads(previous)


def update_attempt(path, attempt_id, record):
    with closing(sqlite3.connect(path)) as database, database:
        database.execute(
            "UPDATE uploads SET record = ? WHERE attempt_id = ?", (json.dumps(record), attempt_id)
        )


def release_attempt(path, attempt_id):
    # Only called before submitting when the guard establishes that no upload began.
    with closing(sqlite3.connect(path)) as database, database:
        database.execute("DELETE FROM uploads WHERE attempt_id = ?", (attempt_id,))


def candidate_slot(name):
    return BASELINE_SLOTS.get(name) or f"hackathon-{hashlib.sha256(name.encode()).hexdigest()[:12]}"


def safe_error(exc):
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return f"{type(exc).__name__} (HTTP {code})"
    # Network/storage error strings may contain signed request URLs.
    return type(exc).__name__


def read_artifacts(directory, names):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No model manifest at {manifest_path}. Run training first.")
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    manifest["manifest_sha256"] = hashlib.sha256(contents).hexdigest()
    if manifest.get("synthetic") or str(manifest.get("provenance", "")).startswith("synthetic"):
        raise RuntimeError(
            "These artifacts were trained on synthetic data. Train on the event data first."
        )
    models = manifest["models"]
    priority = [manifest.get("primary_model"), *(manifest.get("fallback_models") or []), *models]
    selected = list(dict.fromkeys(names or [name for name in priority if name in models]))
    if not selected:
        raise ValueError("The manifest contains no trained models.")
    unknown = set(selected) - models.keys()
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(sorted(unknown))}. Available: {', '.join(models)}."
        )
    return manifest, {name: models[name] for name in selected}


def check_libraries(manifest, models, started):
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    for name, artifact in models.items():
        if artifact["python_version"] != current_python:
            raise RuntimeError(
                f"{name} was saved with Python {artifact['python_version']}. Use that interpreter."
            )

    for package, trained_version in manifest.get("library_versions", {}).items():
        try:
            local_version = version(package)
        except PackageNotFoundError:
            local_version = "missing"
        if local_version != trained_version:
            print(
                f"Warning: {package} {local_version} locally, {trained_version} at training time."
            )

    sandbox = started.get("model_python_versions") or {}
    pins = sandbox.get("library_pins", {}) if isinstance(sandbox, dict) else {}
    pins = pins if isinstance(pins, dict) else {}
    supported = sandbox.get("supported") if isinstance(sandbox, dict) else None
    if isinstance(supported, list) and current_python not in supported:
        print(f"Warning: Python {current_python} is unavailable for platform model replay.")
    if pins.get(current_python):
        print(f"Sandbox package pins for Python {current_python}: {pins[current_python]}")
    elif pins:
        print(f"Warning: no sandbox pin list is published for Python {current_python}.")


def prepare_predictions(model_path, scored, *, model_bytes=None, exped_column="exped"):
    predict = cloudpickle.loads(model_path.read_bytes() if model_bytes is None else model_bytes)
    if not callable(predict):
        raise ValueError(f"{model_path} does not contain a callable.")
    predictions = predict(scored)
    if not isinstance(predictions, pd.DataFrame) or predictions.shape != (len(scored), 1):
        raise ValueError(f"{model_path} must return one prediction column covering every row.")
    if not predictions.index.equals(scored.index):
        raise ValueError(f"{model_path} returned IDs that differ from the downloaded index.")
    values = predictions.iloc[:, 0]
    if not is_numeric_dtype(values) or is_bool_dtype(values) or is_complex_dtype(values):
        raise ValueError(f"{model_path} must return real-valued predictions.")
    values = values.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_path} returned missing or infinite predictions.")
    if len(values) < 2 or (values == values[0]).all():
        raise ValueError(f"{model_path.name} returned constant predictions.")
    if exped_column in scored:
        distinct = pd.Series(values, index=scored.index).groupby(scored[exped_column]).nunique()
        if distinct.empty or distinct.max() <= 1:
            raise ValueError(f"{model_path.name} returned no within-exped variation.")
    return pd.DataFrame({"id": scored.index, "prediction": values})


def write_report(path, report):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    temporary_path.replace(path)


def refresh_scoring(client, record):
    if not record.get("receipt", {}).get("upload_id"):
        return
    try:
        run_status = client.get_diagnostics_run(record["receipt"]["upload_id"])
    except (EverestError, httpx.HTTPError) as exc:
        record["status_error"] = safe_error(exc)
        return
    record["scoring_status"] = run_status.get("status")
    if run_status.get("error"):
        record["scoring_error"] = "See the official run status for details."


def submit_prepared(args, client, report, report_path, models, current):
    lane, window = report["lane"], report["open_window"]
    event_id = report["event_id"]
    ledger = Path(getattr(args, "ledger", None) or DEFAULT_LEDGER)
    submit = (
        client.submit_event_predictions if lane == "live" else client.submit_validation_diagnostics
    )
    owned_models = client.get_models().get("models", [])
    known_slots = {
        label: model["id"]
        for model in owned_models
        for label in (model.get("client_label"), model.get("name"))
        if label and model.get("id")
    }
    for name, artifact in models.items():
        record = report["models"][name]
        if record["state"] == "duplicate_predictions":
            continue
        prediction_path = report_path.parent / record["predictions"]
        model_bytes = (args.artifacts / artifact["path"]).read_bytes()
        if (
            file_sha256(prediction_path) != record["predictions_sha256"]
            or hashlib.sha256(model_bytes).hexdigest() != record["artifact_sha256"]
        ):
            raise RuntimeError(f"Prepared files changed for {name}. Generate fresh predictions.")

        slot = record["slot_name"]
        # Named registration is idempotent and its returned ID scopes the ledger to the account.
        model_id = known_slots.get(slot)
        if not model_id:
            check_window(current, lane, window, submitting=True, event_id=event_id)
            check_budget(current, lane)
            model_id = client.create_model(name=slot)["id"]
        identity = [
            event_id,
            lane,
            window,
            model_id,
            report["target"],
            record["prediction_identity"],
        ]
        attempt_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        record.update(
            model_id=model_id, attempt_id=attempt_id, client_label=f"ev.hack/{attempt_id}"
        )
        reservation = {**record, "state": "submitting", "report_path": str(report_path)}
        previous = reserve_attempt(ledger, attempt_id, reservation)
        if previous is not None:
            if not previous.get("receipt", {}).get("upload_id"):
                record.update(state="review_required", previous_report=previous["report_path"])
                write_report(report_path, report)
                raise RuntimeError(
                    f"Upload attempt {attempt_id} has no confirmed receipt. Review the platform "
                    f"upload history and the original report at {previous['report_path']} before retrying."
                )
            record.update(state="duplicate_skipped", receipt=previous["receipt"])
            write_report(report_path, report)
            refresh_scoring(client, record)
            write_report(report_path, report)
            print(f"{name}: upload {record['receipt']['upload_id']} already recorded.")
            continue

        record["state"] = "submitting"
        write_report(report_path, report)
        try:
            current = client.get_started()
            report["uploads_remaining"] = current.get("uploads_remaining")
            require_event_account(current)
            check_window(current, lane, window, submitting=True, event_id=event_id)
            check_budget(current, lane)
        except Exception as exc:
            release_attempt(ledger, attempt_id)
            record.update(
                state="blocked",
                error=str(exc) if isinstance(exc, RuntimeError) else safe_error(exc),
            )
            write_report(report_path, report)
            raise

        try:
            receipt = submit(
                model_id,
                str(prediction_path),
                target=report["target"],
                model_pkl=model_bytes,
                model_pkl_python_version=artifact["python_version"],
                client_label=record["client_label"],
            )
            if not isinstance(receipt, dict) or not receipt.get("upload_id"):
                raise RuntimeError("Everesteer did not return an upload ID.")
        except Exception as exc:
            record.update(state="review_required", error=safe_error(exc))
            update_attempt(ledger, attempt_id, {**record, "report_path": str(report_path)})
            write_report(report_path, report)
            raise RuntimeError(
                f"Upload attempt {attempt_id} needs review ({safe_error(exc)})."
            ) from None

        receipt_fields = ("upload_id", "status", "coalesced", "uploads_remaining")
        record.update(
            state="submitted",
            receipt={key: receipt[key] for key in receipt_fields if key in receipt},
        )
        if receipt.get("status"):
            record["scoring_status"] = receipt["status"]
        update_attempt(ledger, attempt_id, {**record, "report_path": str(report_path)})
        write_report(report_path, report)
        print(f"{name}: accepted upload {receipt['upload_id']}.")
        refresh_scoring(client, record)
        if "uploads_remaining" in receipt:
            current = {**current, "uploads_remaining": receipt["uploads_remaining"]}
        report["uploads_remaining"] = current.get("uploads_remaining")
        update_attempt(ledger, attempt_id, {**record, "report_path": str(report_path)})
        write_report(report_path, report)
        print(
            f"{name}: status {record.get('scoring_status', 'unavailable')}. Live uploads remaining: {report['uploads_remaining']}."
        )


def run(args, client):
    started = client.get_started()
    require_event_account(started)
    approved_window = getattr(args, "window", None)
    if args.lane == "live" and args.submit and not approved_window:
        raise RuntimeError(
            "Live uploads require --window with the approved window from event status."
        )
    window = check_window(started, args.lane, approved_window, submitting=args.submit)
    manifest, models = read_artifacts(args.artifacts, args.model)
    check_libraries(manifest, models, started)

    schema = client.get_dataset_schema()
    target = schema.get("primary_target")
    if not target:
        raise RuntimeError("get_dataset_schema() did not declare a primary target.")
    if manifest.get("primary_target", target) != target:
        raise RuntimeError(
            f"The manifest's training target differs from the event target {target!r}."
        )

    split = "live" if args.lane == "live" else "validation"
    with TemporaryDirectory() as directory:
        parquet = client.download_dataset(
            split=split, output_path=str(Path(directory) / f"{split}.parquet")
        )
        dataset_sha256 = file_sha256(parquet)
        scored = pd.read_parquet(parquet)
    if scored.empty or scored.index.nlevels != 1 or not scored.index.is_unique:
        raise ValueError(
            f"The downloaded {split} split must have rows with unique, single-level IDs."
        )
    if not all(isinstance(row_id, str) and row_id for row_id in scored.index):
        raise ValueError(
            f"The downloaded {split} split must have nonempty string IDs in its Parquet index."
        )
    row_ids_sha256 = ordered_ids_sha256(scored.index)
    published_targets = {name for name in (schema.get("targets") or []) if isinstance(name, str)}
    published_targets.add(target)
    target_columns = [
        name for name in scored if name in published_targets or name.startswith("target_")
    ]
    if scored[target_columns].notna().any(axis=None):
        raise ValueError(f"The served {split} split contains populated target columns.")
    scored = scored.drop(columns=target_columns)
    exped = manifest.get("exped_column", "exped")
    if manifest.get("feature_columns"):
        input_columns = [*manifest["feature_columns"]]
        missing_features = set(input_columns) - set(scored.columns)
        if len(missing_features) == len(input_columns):
            raise ValueError("The served split contains none of the model's training features.")
        if missing_features:
            print(f"Filling {len(missing_features)} missing training features with NaN.")
        if exped in scored:
            input_columns.append(exped)
        scored = scored.reindex(columns=input_columns)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.artifacts / "runs" / f"{args.lane}-{timestamp}"
    output.mkdir(parents=True)
    report_path = output / "report.json"
    report = {
        "lane": args.lane,
        "split": split,
        "open_window": window,
        "event_id": started.get("event_id"),
        "target": target,
        "rows": len(scored),
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "submit_requested": args.submit,
        "uploads_remaining": started.get("uploads_remaining"),
        "models": {},
    }
    print(f"{split}: {len(scored):,} rows, target {target}, window {window!r}.")
    print(f"Run files: {output}")
    write_report(report_path, report)
    seen_predictions = {}
    for name, artifact in models.items():
        model_path = args.artifacts / artifact["path"]
        model_bytes = model_path.read_bytes()
        artifact_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if artifact.get("sha256") and artifact["sha256"] != artifact_sha256:
            raise ValueError(f"Checksum mismatch for the manifest artifact {name}.")
        predictions = prepare_predictions(
            model_path, scored, model_bytes=model_bytes, exped_column=exped
        )
        prediction_path = output / f"{name}.parquet"
        # Match the SDK's plain string encoding for the high-cardinality ID column.
        predictions.to_parquet(prediction_path, index=False, use_dictionary=False)
        identity = prediction_identity(predictions)
        record = {
            "state": "prepared",
            "predictions": prediction_path.name,
            "predictions_sha256": file_sha256(prediction_path),
            "prediction_identity": identity,
            "artifact": artifact["path"],
            "artifact_sha256": artifact_sha256,
            "python_version": artifact["python_version"],
            "slot_name": candidate_slot(name),
        }
        if identity in seen_predictions:
            record.update(state="duplicate_predictions", duplicate_of=seen_predictions[identity])
            print(f"{name}: predictions match {seen_predictions[identity]}. Skipping any upload.")
        else:
            seen_predictions[identity] = name
        report["models"][name] = record
        write_report(report_path, report)
        print(f"{name}: checked {len(predictions):,} predictions.")

    # Verify that a round did not turn over during download or inference, including on dry runs.
    current = client.get_started()
    report["uploads_remaining"] = current.get("uploads_remaining")
    try:
        require_event_account(current)
        check_window(
            current, args.lane, window, submitting=args.submit, event_id=report["event_id"]
        )
    except RuntimeError as exc:
        report["window_error"] = str(exc)
        write_report(report_path, report)
        raise
    write_report(report_path, report)
    if not args.submit:
        print(f"Dry run complete. Predictions saved locally to {output}.")
        return
    submit_prepared(args, client, report, report_path, models, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lane", choices=("practice", "live"), help="Choose the practice board or the open round."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory containing manifest.json.",
    )
    parser.add_argument(
        "--model", action="append", help="Model name from the manifest. Repeat to select several."
    )
    parser.add_argument(
        "--submit", action="store_true", help="Register model slots and upload predictions."
    )
    parser.add_argument("--window", help="Approved open round. Required with live --submit.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="Local upload ledger.")
    args = parser.parse_args()
    if args.window and args.lane != "live":
        parser.error("--window applies only to the live lane.")
    if args.submit and args.lane == "live" and not args.window:
        parser.error("live --submit requires --window from the event status.")
    try:
        with make_client() as client:
            run(args, client)
    except (EverestError, httpx.HTTPError) as exc:
        print(f"{args.lane} failed: {safe_error(exc)}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"{args.lane} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
