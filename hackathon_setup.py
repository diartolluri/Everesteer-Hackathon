"""Check the event and download local training inputs without submitting anything."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urljoin

import httpx
from dotenv import load_dotenv
from everestapi import EverestAPI, EverestError
from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parent
EVENT_URL = "https://hackathon.everesteer.ai"
MODEL_LIBRARIES = {"cloudpickle", "lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy"}


def make_client() -> EverestAPI:
    load_dotenv(ROOT / ".env")
    key = (os.getenv("EIQ_API_KEY") or os.getenv("EVEREST_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Add EIQ_API_KEY from Everesteer onboarding to the workspace’s .env file."
        )
    base_url = (os.getenv("EIQ_BASE_URL") or EVENT_URL).strip().rstrip("/")
    if base_url != EVENT_URL:
        raise RuntimeError("Set EIQ_BASE_URL to the official hackathon URL from onboarding.")
    return EverestAPI(
        api_key=key,
        base_url=base_url,
        tournament="futures",
    )


def require_event_account(started: dict) -> None:
    """Confirm scope from authenticated orientation before using the event lanes."""
    mode = started.get("mode")
    if mode == "tournament" or not (
        mode == "diagnostics_hackathon"
        or started.get("scope") == "hackathon"
        or started.get("cadence")
    ):
        raise RuntimeError("The credentials did not identify an Everesteer hackathon account.")


def report_model_runtime(started: dict, base_url: str, *, api_key: str | None = None) -> None:
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    supported = started.get("model_python_versions") or {}
    print(f"Local model Python {python}")
    print("Platform model runtimes", json.dumps(supported, sort_keys=True))
    pins = supported.get("library_pins", {}) if isinstance(supported, dict) else {}
    pin_url = pins.get(python) if isinstance(pins, dict) else None
    if not isinstance(pin_url, str):
        print(f"Platform library pins for Python {python} could not be confirmed.")
        return

    url = urljoin(base_url + "/", pin_url)
    lane_path = f"/api/v1/models/upload/lane-pins/{python}"
    try:
        if (
            isinstance(api_key, str)
            and api_key
            and base_url == EVENT_URL
            and url
            in {
                EVENT_URL + lane_path,
                "https://app.everesteer.ai" + lane_path,
            }
        ):
            # The published lane URL points to the tournament host. Event keys
            # can read the same route on the authenticated hackathon host.
            response = httpx.get(
                EVENT_URL + lane_path,
                headers={"X-API-Key": api_key},
                timeout=10,
                follow_redirects=False,
            )
        else:
            response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Platform library pins could not be checked ({type(exc).__name__}).")
        return

    checked = set()
    mismatches = set()
    for line in response.text.splitlines():
        try:
            pin = Requirement(line.split(" #", 1)[0].strip())
        except InvalidRequirement:
            continue
        name = pin.name.lower().replace("_", "-")
        if name not in MODEL_LIBRARIES or (pin.marker and not pin.marker.evaluate()):
            continue
        checked.add(name)
        try:
            installed = version(name)
        except PackageNotFoundError:
            mismatches.add(name)
            print(f"Sandbox dependency {pin} is missing locally.")
        else:
            if installed not in pin.specifier:
                mismatches.add(name)
                print(
                    f"Sandbox version differs for {name}. Local {installed}, platform {pin.specifier}."
                )
    if checked:
        print(
            f"Matched {len(checked - mismatches)} of {len(checked)} published model library pins."
        )
    unverified = MODEL_LIBRARIES - checked
    if unverified:
        print("No applicable pin for", ", ".join(sorted(unverified)))


def phase_time_remaining(ends_at: str | None) -> str:
    if not ends_at:
        return "unavailable"
    try:
        deadline = datetime.fromisoformat(ends_at)
        remaining = max(0, int((deadline - datetime.now(UTC)).total_seconds()))
    except (TypeError, ValueError):
        return "unavailable"
    minutes, seconds = divmod(remaining, 60)
    return f"{minutes}m {seconds:02d}s"


def usdc(micro: int | None) -> str:
    return "unavailable" if micro is None else f"{Decimal(micro) / 1_000_000:f} USDC"


def orient(client: EverestAPI) -> tuple[dict, dict]:
    started = client.get_started()
    require_event_account(started)
    cadence = started.get("cadence") or {}
    print(f"Connected to Everesteer ({started.get('mode') or started.get('scope') or 'event'})")
    if started.get("event_name"):
        print(f"Event {started['event_name']}")
    print(f"Phase {cadence.get('phase')} | open window {cadence.get('open_window')}")
    print(f"Phase time remaining {phase_time_remaining(cadence.get('phase_ends_at'))}")
    print(f"Round intake fenced {cadence.get('intake_fenced')}")
    print(f"Uploads remaining {started.get('uploads_remaining')}")

    compact = client.get_dataset_schema()
    verbose = client.get_dataset_schema(verbose=True)
    feature_sets = verbose.get("feature_sets")
    if not isinstance(feature_sets, dict) or not feature_sets:
        raise RuntimeError("The verbose dataset schema did not include feature sets.")
    schema = {**verbose, **compact, "feature_sets": feature_sets}
    for field in ("primary_target", "feature_encoding", "target_encoding"):
        if not schema.get(field) and field in verbose:
            schema[field] = verbose[field]
    if not schema.get("primary_target"):
        raise RuntimeError("The event dataset schema did not declare a primary target.")
    scoring = client.explain_scoring()
    print(f"Primary target {schema['primary_target']}")
    print("Feature encoding", json.dumps(schema.get("feature_encoding"), sort_keys=True))
    print("Scoring weights", json.dumps(scoring.get("weights"), sort_keys=True))
    report_model_runtime(started, client.base_url, api_key=client.api_key)

    staking = started.get("event_staking") or {}
    print(f"Money event {staking.get('money_event')}")
    if staking.get("money_event"):
        try:
            position = client.get_event_staking()
        except EverestError as exc:
            print(f"Staking status unavailable (HTTP {exc.status_code}).")
        else:
            balance = "unavailable"
            if not position.get("balance_unavailable"):
                if position.get("forwarder_balance_micro") is not None:
                    balance = usdc(position["forwarder_balance_micro"])
                elif position.get("net_usdc") is not None:
                    balance = f"{position['net_usdc']} USDC (recorded net)"
            minimum = usdc(position.get("min_stake_micro"))
            if (
                position.get("min_stake_micro") is None
                and position.get("min_stake_usdc") is not None
            ):
                minimum = f"{position['min_stake_usdc']} USDC"
            print(f"Staking armed {position.get('staking_armed')} | balance {balance}")
            print(
                f"Draft window {position.get('draft_window')} | open {position.get('stake_window_open')}"
            )
            print(f"Model slots {position.get('max_slots')} | minimum {minimum}")
    return schema, scoring


def download_training(client: EverestAPI, schema: dict, scoring: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as staging:
        stage = Path(staging)
        print("Downloading labeled training data…", flush=True)
        client.download_dataset(
            universe="futures", split="train", output_path=str(stage / "train.parquet")
        )
        benchmark = stage / "train_benchmark.parquet"
        try:
            client.download_benchmark("futures", "train", output_path=str(benchmark))
        except EverestError as exc:
            if exc.status_code != 404:
                raise
            print("Training benchmark unavailable (HTTP 404).")
            benchmark.unlink(missing_ok=True)
            (directory / benchmark.name).unlink(missing_ok=True)
        for name, payload in (("schema.json", schema), ("scoring.json", scoring)):
            (stage / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for path in stage.iterdir():
            path.replace(directory / path.name)
    print(f"Training inputs saved to {directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "download"), nargs="?", default="status")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        with make_client() as client:
            schema, scoring = orient(client)
            if args.command == "download":
                download_training(client, schema, scoring, args.data_dir)
    except EverestError as exc:
        print(f"Everesteer returned HTTP {exc.status_code}.", file=sys.stderr)
        return 1
    except (RuntimeError, httpx.HTTPError, OSError) as exc:
        message = type(exc).__name__ if isinstance(exc, httpx.HTTPError) else str(exc)
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
