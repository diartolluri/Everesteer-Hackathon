"""Prepare or explicitly upload predictions for one Everesteer lane."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from hackathon_setup import ROOT, make_client


def check_window(started, lane, expected_window=None, *, submitting=False):
    cadence = started.get("cadence") or {}
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


def read_artifacts(directory, names):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No model manifest at {manifest_path}. Run training first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("synthetic"):
        raise RuntimeError(
            "These artifacts were trained on synthetic data. Train on the event data first."
        )
    models = manifest["models"]
    selected = list(dict.fromkeys(names or models))
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
    if pins.get(current_python):
        print(f"Sandbox package pins for Python {current_python}: {pins[current_python]}")
    elif pins:
        print(f"Warning: no sandbox pin list is published for Python {current_python}.")


def prepare_predictions(model_path, scored):
    with model_path.open("rb") as model_file:
        predict = cloudpickle.load(model_file)
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
    if (values == values[0]).all():
        print(f"Warning: {model_path.name} returned constant predictions.")
    return pd.DataFrame({"id": scored.index, "prediction": values})


def write_report(path, report):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    temporary_path.replace(path)


def run(args, client):
    started = client.get_started()
    window = check_window(started, args.lane, submitting=args.submit)
    manifest, models = read_artifacts(args.artifacts, args.model)
    check_libraries(manifest, models, started)

    target = client.get_dataset_schema().get("primary_target")
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
        scored = pd.read_parquet(parquet)
    if scored.empty or scored.index.nlevels != 1 or not scored.index.is_unique:
        raise ValueError(
            f"The downloaded {split} split must have rows with unique, single-level IDs."
        )
    if not all(isinstance(row_id, str) and row_id for row_id in scored.index):
        raise ValueError(
            f"The downloaded {split} split must have nonempty string IDs in its Parquet index."
        )
    scored = scored.drop(
        columns=[name for name in scored if name == target or name.startswith("target_")]
    )
    if manifest.get("feature_columns"):
        input_columns = [*manifest["feature_columns"]]
        missing_features = set(input_columns) - set(scored.columns)
        if len(missing_features) == len(input_columns):
            raise ValueError("The served split contains none of the model's training features.")
        if missing_features:
            print(f"Filling {len(missing_features)} missing training features with NaN.")
        exped = manifest.get("exped_column", "exped")
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
        "target": target,
        "rows": len(scored),
        "submit_requested": args.submit,
        "uploads_remaining": started.get("uploads_remaining"),
        "models": {},
    }
    print(f"{split}: {len(scored):,} rows, target {target}, window {window!r}.")
    print(f"Run files: {output}")
    write_report(report_path, report)
    slot_names = {
        name: f"hackathon-{number:02d}" for number, name in enumerate(manifest["models"], 1)
    }
    for name, artifact in models.items():
        model_path = args.artifacts / artifact["path"]
        predictions = prepare_predictions(model_path, scored)
        prediction_path = output / f"{name}.parquet"
        # Match the SDK's plain string encoding for the high-cardinality ID column.
        predictions.to_parquet(prediction_path, index=False, use_dictionary=False)
        report["models"][name] = {
            "state": "prepared",
            "predictions": prediction_path.name,
            "slot_name": slot_names[name],
        }
        write_report(report_path, report)
        print(f"{name}: checked {len(predictions):,} predictions.")

    # Verify that a round did not turn over during download or inference, including on dry runs.
    current = client.get_started()
    report["uploads_remaining"] = current.get("uploads_remaining")
    try:
        check_window(current, args.lane, window, submitting=args.submit)
    except RuntimeError as exc:
        report["window_error"] = str(exc)
        write_report(report_path, report)
        raise
    write_report(report_path, report)
    if not args.submit:
        print(f"Dry run complete. Predictions saved locally to {output}.")
        return

    submit = (
        client.submit_event_predictions
        if args.lane == "live"
        else client.submit_validation_diagnostics
    )
    for name, artifact in models.items():
        record = report["models"][name]
        check_window(current, args.lane, window, submitting=True)
        if args.lane == "live" and current.get("uploads_remaining") == 0:
            raise RuntimeError("The platform reports no live upload allowance remaining.")
        # A named create reuses an existing slot according to the SDK's idempotency contract.
        model_id = client.create_model(name=slot_names[name])["id"]
        record.update(model_id=model_id, state="checking_window")
        write_report(report_path, report)

        try:
            current = client.get_started()
            check_window(current, args.lane, window, submitting=True)
            if args.lane == "live" and current.get("uploads_remaining") == 0:
                raise RuntimeError("The platform reports no live upload allowance remaining.")
        except Exception as exc:
            record.update(state="blocked", error=str(exc))
            write_report(report_path, report)
            raise

        try:
            receipt = submit(
                model_id,
                str(output / record["predictions"]),
                target=target,
                model_pkl=str(args.artifacts / artifact["path"]),
                model_pkl_python_version=artifact["python_version"],
                client_label=f"{output.name}/{name}",
            )
        except Exception as exc:
            record.update(state="submission_error", error=str(exc))
            write_report(report_path, report)
            raise
        record.update(state="submitted", receipt=receipt)
        write_report(report_path, report)
        print(f"{name} submission response: {receipt}")
        if receipt.get("upload_id"):
            try:
                run_status = client.get_diagnostics_run(receipt["upload_id"])
                record["scoring_status"] = run_status.get("status")
                if run_status.get("error"):
                    record["scoring_error"] = run_status["error"]
                print(f"{name} scoring status: {record['scoring_status']}")
            except Exception as exc:
                record["status_error"] = str(exc)
                print(f"Could not refresh scoring status for {name}: {exc}")
        try:
            current = client.get_status()
            report["uploads_remaining"] = current.get("uploads_remaining")
            print(f"Live uploads remaining: {report['uploads_remaining']}")
        except Exception as exc:
            print(f"Could not refresh the upload allowance: {exc}")
        write_report(report_path, report)


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
    args = parser.parse_args()
    try:
        with make_client() as client:
            run(args, client)
    except Exception as exc:
        print(f"{args.lane} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
