"""Inspect Everesteer rounds, uploads, and possible stake allocations using read-only calls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from statistics import fmean, pstdev

import httpx
from everestapi import EverestError

from everesteer_hackathon.setup import (
    ROOT,
    make_client,
    phase_time_remaining,
    require_event_account,
    usdc,
)

FINAL_STATES = {"done", "completed", "failed", "cancelled", "canceled"}
ELIGIBLE_STATES = {"submitted", "pending", "running", "done", "completed"}


def load_object(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def read_optional(warnings, label, call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except EverestError as exc:
        warnings.append(f"{label} unavailable (HTTP {exc.status_code}).")
    except httpx.HTTPError as exc:
        warnings.append(f"{label} unavailable ({type(exc).__name__}).")
    return {}


def local_receipts(artifacts, aliases, warnings, event_id=None):
    records = []
    seen = set()
    paths = (artifacts / "runs").glob("*/report.json")
    for path in sorted(paths, key=lambda item: item.parent.name.rsplit("-", 1)[-1], reverse=True):
        try:
            report = load_object(path)
        except (OSError, ValueError):
            warnings.append(f"Could not read {path.parent.name}/report.json.")
            continue
        if event_id is not None and report.get("event_id") not in (None, event_id):
            continue
        lane, window = report.get("lane"), report.get("open_window")
        for name, model in report.get("models", {}).items():
            if aliases and name not in aliases:
                continue
            receipt = model.get("receipt") or {}
            identifier = receipt.get("upload_id")
            key = (name, lane, window) if identifier is None else identifier
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "model": name,
                    "model_id": model.get("model_id"),
                    "lane": lane,
                    "window": window,
                    "upload_id": identifier,
                    "state": model.get("scoring_status") or model.get("state"),
                    "run": path.parent.name,
                    "artifact_sha256": model.get("artifact_sha256"),
                }
            )
    return records


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def read_fold_scores(artifacts, manifest):
    file_name = manifest.get("fold_metrics_path")
    if (
        not file_name
        or manifest.get("synthetic") is not False
        or str(manifest.get("provenance", "")).startswith("synthetic")
    ):
        return {}
    results = {}
    with (artifacts / file_name).open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            score = finite_number(row.get("round_score_proxy"))
            if score is not None:
                results.setdefault(row["model"], {})[row["fold"]] = score
    return results


def staking_budget(position):
    """Return an illustrative 10% exposure ceiling, after checking reported locks and drafts."""
    if position.get("staking_armed") is not True or position.get("stake_window_open") is not True:
        return 0, "Staking is unavailable in the current window."
    balance = position.get("forwarder_balance_micro")
    minimum, slots = position.get("min_stake_micro"), position.get("max_slots")
    if (
        not position.get("draft_window")
        or position.get("balance_unavailable")
        or any(type(value) is not int or value < 1 for value in (balance, minimum, slots))
    ):
        return 0, "A draft window, positive balance, contract minimum and slot cap are required."
    windows = position.get("windows")
    if not isinstance(windows, list) or any(
        not isinstance(window.get("allocations"), list) for window in windows
    ):
        return (
            0,
            "The platform did not report enough allocation history to check outstanding locks.",
        )
    if any(
        not allocation.get("locked_at") or not allocation.get("lock_tx_hash")
        for window in windows
        for allocation in window["allocations"]
    ):
        return 0, "Existing drafts or locks without a relay hash require manual reconciliation."
    if any(
        window.get("window", window.get("scoring_window")) == position["draft_window"]
        and window["allocations"]
        for window in windows
    ):
        return 0, "Existing allocations in the draft window require manual reconciliation."
    budget = balance // 10
    if budget < minimum:
        return 0, "Ten percent of the reported balance is below the contract minimum."
    return budget, None


def describe_scenario(name, picks, folds):
    total = sum(amount for _, amount, _ in picks)
    values = [sum(amount * scores[fold] for _, amount, scores in picks) / total for fold in folds]
    return {
        "name": name,
        "allocations": {model_id: amount for model_id, amount, _ in picks},
        "mean_proxy": fmean(values),
        "worst_proxy": min(values),
        "std_proxy": pstdev(values),
        "folds": len(folds),
    }


def stake_scenarios(position, models, receipts, fold_scores, correlations):
    scenarios = [{"name": "Keep the balance unallocated", "allocations": {}}]
    budget, reason = staking_budget(position)
    if reason:
        return scenarios, reason

    window = position["draft_window"]
    candidates = {}
    completed = {}
    for item in receipts:
        name, model_id = item["model"], item["model_id"]
        if item["lane"] != "live" or not model_id or name not in models:
            continue
        digest = models[name].get("sha256")
        if digest and digest != item.get("artifact_sha256"):
            continue
        if (
            item["window"] == window
            and item.get("platform_verified")
            and item["state"] in ELIGIBLE_STATES
        ):
            candidates.setdefault(name, model_id)
        elif item["window"] != window and item.get("round_score") is not None:
            completed.setdefault(name, {}).setdefault(item["window"], item["round_score"])

    evidence = {name: fold_scores[name] for name in candidates.keys() & fold_scores.keys()}
    if not evidence:
        return scenarios, "A current-round receipt and real-data walk-forward scores are required."
    folds = sorted(set.intersection(*(set(scores) for scores in evidence.values())))
    if len(folds) < 2:
        return scenarios, "Candidates need at least two comparable walk-forward folds."

    ranked = []
    for name, scores in evidence.items():
        values = [scores[fold] for fold in folds]
        observed = list(completed.get(name, {}).values())
        # Downweight unstable historical edge and negative scores observed at the event.
        penalty = min(0.0, fmean(observed)) * len(observed) / (len(observed) + 4) if observed else 0
        criterion = (fmean(values) + min(values)) / 2 - pstdev(values) / 4 + penalty
        if criterion > 0:
            ranked.append((criterion, name))
    if not ranked:
        return scenarios, "No uploaded model has a positive downside-adjusted score proxy."

    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    primary = ranked[0][1]
    scenarios.append(
        describe_scenario(
            "Small primary", [(candidates[primary], budget, evidence[primary])], folds
        )
    )
    if position["max_slots"] > 1 and budget // 2 >= position["min_stake_micro"]:
        for _, other in ranked[1:]:
            correlation = finite_number(correlations.get(primary, {}).get(other))
            if (
                correlation is None
                or abs(correlation) >= 0.85
                or candidates[other] == candidates[primary]
            ):
                continue
            picks = [
                (candidates[primary], budget - budget // 2, evidence[primary]),
                (candidates[other], budget // 2, evidence[other]),
            ]
            diversified = describe_scenario("Split across two diverse models", picks, folds)
            diversified["prediction_rank_correlation"] = correlation
            scenarios.append(diversified)
            break
    return scenarios, "Scenarios use an illustrative ceiling of 10% of the reported live balance."


def event_snapshot(client, artifacts, model_names=(), window=None, staking_scenarios=False):
    started = client.get_started() if client else {}
    if client:
        require_event_account(started)
    cadence = started.get("cadence") or {}
    staking = started.get("event_staking") or {}
    warnings = []
    manifest = load_object(artifacts / "manifest.json")
    entries = manifest.get("models", {})
    active_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    changed_libraries = False
    for package, artifact_version in manifest.get("library_versions", {}).items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "missing"
        if installed != artifact_version:
            changed_libraries = True
            warnings.append(f"Artifact dependency {package} {artifact_version}, local {installed}.")
    models = {}
    for name in dict.fromkeys(model_names or entries):
        entry = entries.get(name, {})
        if not entry.get("path") or not (artifacts / entry["path"]).is_file():
            state = "missing file"
        elif manifest.get("synthetic") or str(manifest.get("provenance", "")).startswith(
            "synthetic"
        ):
            state = "synthetic only"
        elif entry.get("python_version") != active_python:
            state = "Python version differs"
        elif changed_libraries:
            state = "dependency versions differ"
        else:
            state = "ready locally"
        models[name] = {**entry, "state": state}

    receipts = local_receipts(artifacts, models, warnings, started.get("event_id"))
    if client:
        tracked = [record for record in receipts if record["upload_id"]]
        # Event runs have small model counts; keep a status check fast after repeated rehearsals.
        for record in tracked[:12]:
            details = read_optional(
                warnings, "Receipt", client.get_diagnostics_run, record["upload_id"]
            )
            record["state"] = details.get("status") or record["state"]
            scoring_window = details.get("scoring_window")
            scored_model = details.get("model_id")
            record["platform_verified"] = (
                bool(details.get("status"))
                and (scoring_window is None or scoring_window == record["window"])
                and (scored_model is None or scored_model == record["model_id"])
            )
            if record["platform_verified"] and record["state"] in FINAL_STATES:
                score = details.get(
                    "round_score", (details.get("metrics") or {}).get("round_score")
                )
                record["round_score"] = finite_number(score)

    board_window = window or cadence.get("open_window")
    board = (
        read_optional(
            warnings, "Round board", client.get_diagnostics_leaderboard, scoring_window=board_window
        )
        if client and board_window
        else {}
    )
    standings = (
        read_optional(warnings, "Standings", client.get_diagnostics_standings, limit=1)
        if client
        else {}
    )
    position = (
        read_optional(warnings, "Staking", client.get_event_staking)
        if staking.get("money_event") is True
        else {}
    )
    suggestions, note = [], None
    if staking_scenarios:
        try:
            scores = read_fold_scores(artifacts, manifest)
        except (OSError, KeyError, ValueError):
            warnings.append("The walk-forward score file could not be read.")
            scores = {}
        suggestions, note = stake_scenarios(
            position,
            {name: model for name, model in models.items() if model["state"] == "ready locally"},
            receipts,
            scores,
            manifest.get("model_rank_correlations") or {},
        )

    if board:
        owned = {receipt["model_id"] for receipt in receipts if receipt["model_id"]}
        board = {
            "scoring_window": board.get("scoring_window"),
            "rank_metric": board.get("rank_metric"),
            "total": board.get("total"),
            "sealed": board.get("sealed"),
            "entries": [
                {
                    key: row.get(key)
                    for key in ("model_id", "rank", "round_score", "fit20", "unq", "inov")
                }
                for row in board.get("entries", [])
                if row.get("model_id") in owned
            ],
        }
    if standings:
        standings = {
            "available": standings.get("available"),
            "your_rank": standings.get("your_rank"),
            "your_event_score": next(
                (
                    row.get("event_score")
                    for row in standings.get("entries", [])
                    if row.get("rank") is not None and row.get("rank") == standings.get("your_rank")
                ),
                None,
            ),
        }
    position = {
        key: position[key]
        for key in (
            "staking_armed",
            "stake_window_open",
            "draft_window",
            "forwarder_balance_micro",
            "balance_unavailable",
            "min_stake_micro",
            "max_slots",
            "net_micro",
        )
        if key in position
    }

    return {
        "online": client is not None,
        "cadence": cadence,
        "uploads_remaining": started.get("uploads_remaining", "unavailable"),
        "models": models,
        "receipts": receipts,
        "board": board,
        "standings": standings,
        "money_event": staking.get("money_event"),
        "staking": position,
        "scenarios": suggestions,
        "scenario_note": note,
        "warnings": warnings,
    }


def print_snapshot(snapshot):
    if not snapshot["online"]:
        print("Local artifact and receipt status.")
    cadence = snapshot["cadence"]
    seconds = finite_number(cadence.get("seconds_until_next_phase"))
    if seconds is None:
        remaining = phase_time_remaining(cadence.get("phase_ends_at"))
    else:
        minutes, seconds = divmod(max(0, int(seconds)), 60)
        remaining = f"{minutes}m {seconds:02d}s"
    if cadence.get("clock_paused"):
        remaining += " (clock paused)"
    print(
        f"Phase {cadence.get('phase')} | window {cadence.get('open_window')} | remaining {remaining}"
    )
    uploads = snapshot["uploads_remaining"]
    print(
        f"Round intake fenced {cadence.get('intake_fenced')} | uploads {'uncapped' if uploads is None else uploads}"
    )
    for model, entry in snapshot["models"].items():
        print(f"  {model} | {entry['state']} | Python {entry.get('python_version', 'unavailable')}")
    if not snapshot["models"]:
        print("  No local model manifest is available.")
    for item in snapshot["receipts"][:12]:
        label = item["window"] or item["lane"]
        score = item.get("round_score")
        print(
            f"  {label} {item['model']} | {item['state']} | "
            f"upload {item['upload_id'] or 'none'}"
            + (f" | round score {score:+.6f}" if score is not None else "")
        )
    board = snapshot["board"]
    if board:
        print(
            f"Board {board.get('scoring_window')} | rank metric {board.get('rank_metric')} | entries {board.get('total')}"
        )
        own_ids = {r["model_id"] for r in snapshot["receipts"] if r["model_id"]}
        for entry in board.get("entries", []):
            if entry.get("model_id") in own_ids:
                print(
                    f"  Model {entry['model_id']} | rank {entry.get('rank')} | score {entry.get('round_score')}"
                )
    standings = snapshot["standings"]
    if standings.get("available"):
        print(
            f"Cumulative standings rank {standings.get('your_rank')} | event score {standings.get('your_event_score')}"
        )
    if snapshot["money_event"] is True:
        print("Winner metric is the final recorded staking balance.")
    elif snapshot["money_event"] is False:
        print("Winner metric is the cumulative event standings.")
    position = snapshot["staking"]
    if position:
        balance = (
            None if position.get("balance_unavailable") else position.get("forwarder_balance_micro")
        )
        print(
            f"Staking armed {position.get('staking_armed')} | draft window {position.get('draft_window')} | open {position.get('stake_window_open')}"
        )
        print(
            f"Live balance {usdc(balance)} | minimum {usdc(position.get('min_stake_micro'))} | slots {position.get('max_slots')}"
        )
    if snapshot["scenario_note"]:
        print(snapshot["scenario_note"])
    for scenario in snapshot["scenarios"]:
        allocation = ", ".join(
            f"{name} {usdc(amount)}" for name, amount in scenario["allocations"].items()
        )
        print(f"  {scenario['name']} | {allocation or '0 USDC'}")
        if allocation:
            print(
                f"    Walk-forward score proxy mean {scenario['mean_proxy']:+.6f}, worst {scenario['worst_proxy']:+.6f}, std {scenario['std_proxy']:.6f} across {scenario['folds']} folds"
            )
    for warning in snapshot["warnings"]:
        print(warning, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "stakes"), nargs="?", default="status")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Limit to a local model alias. Repeat to select several.",
    )
    parser.add_argument("--window", help="Read the named round board. Defaults to the open window.")
    parser.add_argument("--json", action="store_true", help="Emit the snapshot as JSON.")
    parser.add_argument(
        "--offline", action="store_true", help="Use local artifacts and receipts only."
    )
    args = parser.parse_args()
    try:
        if args.offline:
            snapshot = event_snapshot(
                None, args.artifacts, args.model, staking_scenarios=args.command == "stakes"
            )
        else:
            with make_client() as client:
                snapshot = event_snapshot(
                    client, args.artifacts, args.model, args.window, args.command == "stakes"
                )
        if args.json:
            print(json.dumps(snapshot, indent=2))
        else:
            print_snapshot(snapshot)
    except EverestError as exc:
        print(f"Everesteer returned HTTP {exc.status_code}.", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"Everesteer unavailable ({type(exc).__name__}).", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
