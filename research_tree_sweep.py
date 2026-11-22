"""Scout many residual LightGBM models and publish diverse, validated artifacts locally."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hackathon_models import (
    LIBRARIES,
    benchmark_residual_target,
    check_predictions,
    chronological_split,
    dataset_columns,
    event_proxy_scale,
    feature_stability,
    fitted_predictor,
    model_input,
    package,
    select_features,
    stable_feature_subset,
    weighted_rank_prediction,
)
from hackathon_scoring import align_benchmark, event_metrics, score_expeds

CAPACITIES = {
    "compact": {"num_leaves": 8, "max_depth": 3, "min_child_samples": 120, "reg_lambda": 7},
    "regular": {"num_leaves": 15, "max_depth": 5, "min_child_samples": 60, "reg_lambda": 5},
    "rich": {"num_leaves": 31, "max_depth": 7, "min_child_samples": 46, "reg_lambda": 3},
}
RECENCIES = (360, 720, 1200, 1800)
VIEWS = ("residual24", "residual48", "residual96", "fit-residual", "low-common")
STRENGTHS = (0.25, 0.5, 0.75, 0.95)
OBJECTIVE = "round_score_proxy"


@dataclass(frozen=True)
class Trial:
    history: int
    view: str
    strength: float
    capacity: str

    @property
    def name(self):
        return f"tree-{self.view}-{self.history}-r{round(self.strength * 100)}-{self.capacity}"


def trial_grid(refine=False):
    initial = [
        Trial(1200, "residual48", 0.5, "regular"),
        Trial(360, "residual24", 0.5, "compact"),
        Trial(720, "residual96", 0.75, "rich"),
        Trial(1200, "fit-residual", 0.95, "regular"),
        Trial(1800, "low-common", 0.25, "compact"),
        Trial(720, "residual24", 0.95, "compact"),
        Trial(360, "low-common", 0.75, "regular"),
        Trial(1200, "residual96", 0.25, "rich"),
        Trial(720, "fit-residual", 0.5, "rich"),
        Trial(1800, "residual48", 0.75, "regular"),
        Trial(360, "residual48", 0.95, "rich"),
        Trial(720, "low-common", 0.5, "compact"),
    ]
    axes = (RECENCIES, VIEWS, STRENGTHS, CAPACITIES)
    if refine:
        axes = (
            (240, 360, 480, 620, 800, 1000),
            ("residual12", "residual18", "residual24", "residual32"),
            (0.8, 0.9, 0.95, 1.0),
            ("compact", "regular"),
        )
        initial = [
            Trial(360, "residual24", 0.95, "compact"),
            Trial(620, "residual24", 0.95, "compact"),
            Trial(800, "residual18", 0.95, "regular"),
            Trial(480, "residual32", 0.9, "compact"),
            Trial(1000, "residual12", 1.0, "compact"),
            Trial(240, "residual18", 0.95, "compact"),
        ]
    remaining = sorted(
        {Trial(*values) for values in product(*axes)} - set(initial), key=lambda trial: trial.name
    )
    random.Random(821).shuffle(remaining)
    return initial + remaining


def feature_view(profiles, view):
    if view == "fit-residual":
        return list(
            dict.fromkeys(
                stable_feature_subset(profiles, "residual", 40)
                + stable_feature_subset(profiles, "fit", 24)
            )
        )
    if view == "low-common":
        return stable_feature_subset(profiles, "low_exposure", 48)
    return stable_feature_subset(profiles, "residual", int(view.removeprefix("residual")))


def row_window(train, exped, target, calendar):
    labeled = np.isfinite(train[target])
    return train.loc[labeled & train[exped].isin(calendar)]


def build_windows(train, calendar, exped, target, count, size, embargo):
    splits, folds = [], []
    for offset in range(count - 1, -1, -1):
        allowed = calendar[: len(calendar) - offset * size]
        split = chronological_split(train.loc[train[exped].isin(allowed), exped], size, embargo)
        splits.append(split)
        folds.append(
            (tuple(split["fit_expeds"]), row_window(train, exped, target, split["holdout_expeds"]))
        )
    return splits, folds


def build_fitter(train, columns, missing, target, exped, benchmark, args):
    select_count = max(RECENCIES)
    exposure = select_features(train, columns, missing).mean(axis=1)

    @lru_cache(maxsize=12)
    def prepare(end, length):
        calendar = end[-min(select_count, length) :]
        fit = row_window(train, exped, target, calendar)
        profile = feature_stability(fit, columns, missing, target, exped, benchmark, exposure)
        return fit, profile

    @lru_cache(maxsize=40)
    def labels(end, length, strength):
        frame, _ = prepare(end, length)
        return benchmark_residual_target(frame[target], benchmark, frame[exped], strength)

    def fit(trial, calendar):
        frame, profile = prepare(calendar, trial.history)
        chosen = feature_view(profile, trial.view)
        policy = "indicator" if trial.view == "fit-residual" else "native"
        matrix = model_input(
            select_features(frame, chosen, missing), "lightgbm", policy, missing, 0.0
        )
        model = lgb.LGBMRegressor(
            **CAPACITIES[trial.capacity],
            n_estimators=args.estimators,
            learning_rate=0.055,
            colsample_bytree=0.75,
            max_bin=31,
            random_state=821,
            n_jobs=args.threads,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(matrix, labels(calendar, trial.history, trial.strength))
        predict = fitted_predictor(model, chosen, missing, "lightgbm", policy)
        return weighted_rank_prediction({trial.name: predict}, exped, {trial.name: 1.0})

    return fit


def evaluate(trial, fit, folds, benchmark, score_scale, exped, target, columns):
    scores, ranks = [], []
    for calendar, heldout in folds:
        predict = fit(trial, calendar)
        values = check_predictions(predict, heldout[[exped, *columns]])["prediction"]
        metrics = score_expeds(heldout, exped, target, {trial.name: values}, benchmark)
        scores.append(
            {
                "fit": metrics["fit"].mean(),
                "unq": metrics["unq_covariance"].mean(),
                "benchmark_corr": metrics["benchmark_rank_corr"].mean(),
                "benchmark_exped_share": metrics["unq_covariance"].notna().mean(),
                **event_metrics(metrics, score_scale),
            }
        )
        for _, group in values.groupby(heldout[exped], sort=False):
            rank = group.rank().to_numpy(copy=True)
            rank -= rank.mean()
            norm = np.linalg.norm(rank)
            ranks.append(rank / norm if norm else rank)
    vector = np.concatenate(ranks) / np.sqrt(len(ranks))
    values = pd.DataFrame(scores)
    summary = {
        **values.mean().to_dict(),
        "fold_sd": values[OBJECTIVE].std(ddof=0),
        "worst_fold": values[OBJECTIVE].min(),
    }
    eligible = bool(
        values[OBJECTIVE].notna().all() and values["benchmark_exped_share"].ge(0.8).all()
    )
    return summary, vector, eligible


def persist(args, manifest, summary, staging):
    ordered = sorted(
        manifest["models"], key=lambda name: manifest["models"][name][OBJECTIVE], reverse=True
    )
    manifest["primary_model"] = next(iter(ordered), None)
    manifest["fallback_order"] = ordered
    board = staging / "leaderboard.csv"
    pd.DataFrame(summary).sort_values(OBJECTIVE, ascending=False).to_csv(board, index=False)
    staged_manifest = staging / "manifest.json"
    staged_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    staged_manifest.replace(args.artifacts / "manifest.json")


def audit(args, train, chronological, fit, benchmark, scale, exped, target, columns):
    """Check packaged candidates on two dates outside the windows that selected them."""
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    source_windows = manifest["windows"]
    args.estimators = manifest.get("estimator_count", args.estimators)
    fold_expeds = len(source_windows[-1]["holdout_expeds"])
    embargo_expeds = len(source_windows[-1]["embargo_expeds"])
    first_scored = min(window["holdout_expeds"][0] for window in source_windows)
    calendar = chronological[: chronological.index(first_scored)]
    _, windows = build_windows(train, calendar, exped, target, 2, fold_expeds, embargo_expeds)
    summaries = []
    for name in manifest["fallback_order"][: args.audit_top]:
        trial = Trial(**manifest["models"][name]["trial"])
        metrics, _, eligible = evaluate(
            trial, fit, windows, benchmark, scale, exped, target, columns
        )
        summaries.append(
            {
                "candidate": name,
                **metrics,
                "eligible": eligible,
                "first_exped": windows[0][1][exped].min(),
                "last_exped": windows[-1][1][exped].max(),
            }
        )
        print(f"Audit {name} {metrics[OBJECTIVE]:+.5f}", flush=True)
    destination = args.artifacts / "audit.csv"
    with tempfile.NamedTemporaryFile(dir=args.artifacts, delete=False) as handle:
        temporary = Path(handle.name)
    pd.DataFrame(summaries).sort_values(OBJECTIVE, ascending=False).to_csv(temporary, index=False)
    temporary.replace(destination)
    return 0


def promote(args, train, fit, chronological, exped, target, columns, benchmark, scale):
    """Package late leaders which improved on the best previously streamed model."""
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    board = pd.read_csv(args.artifacts / manifest["leaderboard"])
    previous = manifest["models"]
    ceiling = max((details[OBJECTIVE] for details in previous.values()), default=-np.inf)
    names = board.loc[
        board["eligible"] & board[OBJECTIVE].gt(ceiling) & ~board["candidate"].isin(previous)
    ].sort_values(OBJECTIVE, ascending=False)
    if names.empty:
        print("The streamed portfolio already contains the highest-scoring candidate.", flush=True)
        return 0
    destination = args.artifacts.with_name(f"{args.artifacts.name}-promoted")
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="promoted-", dir=destination))
    manifest["models"] = {}
    manifest["leaderboard"] = f"{staging.name}/leaderboard.csv"
    args.estimators = manifest.get("estimator_count", args.estimators)
    windows = manifest["windows"]
    folds = [
        (tuple(split["fit_expeds"]), row_window(train, exped, target, split["holdout_expeds"]))
        for split in windows
    ]
    sample = row_window(train, exped, target, windows[-1]["holdout_expeds"][-3:])
    outcomes = []
    for row in names.to_dict("records"):
        if len(outcomes) == args.promote_top:
            break
        trial = Trial(int(row["history"]), row["view"], float(row["strength"]), row["capacity"])
        metrics, ranks, eligible = evaluate(
            trial, fit, folds, benchmark, scale, exped, target, columns
        )
        if not eligible or metrics[OBJECTIVE] <= ceiling:
            continue
        if any(abs(float(ranks @ entry)) >= args.max_rank_corr for entry in outcomes):
            continue
        records = package({trial.name: fit(trial, chronological)}, sample, exped, staging)
        manifest["models"][trial.name] = {
            **records[trial.name],
            "prevalidated": True,
            OBJECTIVE: metrics[OBJECTIVE],
            "trial": asdict(trial),
        }
        outcomes.append(ranks)
        print(f"Promoted {trial.name} to {destination}", flush=True)
    if outcomes:
        args.artifacts = destination
        persist(args, manifest, board.to_dict("records"), staging)
    return 0


def search(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    settings = json.loads((args.data_dir / "scoring.json").read_text())
    scale, fallback = event_proxy_scale(settings)
    if fallback or scale is None:
        raise ValueError("Saved scoring settings need a valid event score scale.")
    train = pd.read_parquet(args.data_dir / "train.parquet")
    target, exped, columns, missing = dataset_columns(train, schema)
    if (
        train.index.nlevels != 1
        or not train.index.is_unique
        or train.index.hasnans
        or not all(isinstance(row_id, str) and row_id for row_id in train.index)
        or ("data_type" in train and set(train["data_type"].dropna().unique()) - {"train"})
    ):
        raise ValueError("The training parquet must have unique opaque IDs from the train split.")
    train = train[[exped, target, *columns]]
    benchmark, _ = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"),
        train,
        exped,
        minimum_coverage=0.8,
    )
    if benchmark is None:
        raise ValueError("An aligned official benchmark is required for residual evaluation.")

    chronological = tuple(sorted(train[exped].unique()))
    fit = build_fitter(train, columns, missing, target, exped, benchmark, args)
    if args.audit_top:
        return audit(args, train, chronological, fit, benchmark, scale, exped, target, columns)
    if args.promote_top:
        return promote(args, train, fit, chronological, exped, target, columns, benchmark, scale)
    splits, folds = build_windows(
        train,
        chronological,
        exped,
        target,
        4 if args.refine else 2,
        args.fold_expeds,
        args.embargo_expeds,
    )
    sample = row_window(train, exped, target, splits[-1]["holdout_expeds"][-3:])

    args.artifacts.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="sweep-", dir=args.artifacts))
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": {library: version(library) for library in LIBRARIES},
        "selection_metric": OBJECTIVE,
        "score_scale": scale,
        "estimator_count": args.estimators,
        "leaderboard": f"{staging.name}/leaderboard.csv",
        "models": {},
        "windows": splits,
    }
    grid = trial_grid(args.refine)[: args.trials]
    rows, pending, packaged_ranks = [], [], []
    for index, trial in enumerate(grid, 1):
        metrics, ranks, eligible = evaluate(
            trial, fit, folds, benchmark, scale, exped, target, columns
        )
        rows.append(
            {
                "candidate": trial.name,
                **asdict(trial),
                **metrics,
                "eligible": eligible,
            }
        )
        print(f"{index}/{len(grid)} {trial.name} {metrics[OBJECTIVE]:+.5f}", flush=True)
        if eligible:
            pending.append((metrics[OBJECTIVE], trial, ranks))
        if index % args.batch and index != len(grid):
            continue

        ordered = sorted(pending, key=lambda result: result[0], reverse=True)
        slots = min(args.packaged - len(packaged_ranks), args.publish_batch)
        best = ordered[0][0] if ordered else -np.inf
        for score, candidate, signature in ordered:
            if slots <= 0 or score < best - args.max_score_gap:
                break
            if candidate.name in manifest["models"]:
                continue
            if any(
                abs(float(signature @ existing)) >= args.max_rank_corr
                for existing in packaged_ranks
            ):
                continue
            predictor = fit(candidate, chronological)
            try:
                record = package({candidate.name: predictor}, sample, exped, staging)[
                    candidate.name
                ]
            except ValueError as error:
                print(f"Artifact check skipped {candidate.name}. {error}", flush=True)
                continue
            manifest["models"][candidate.name] = {
                **record,
                "prevalidated": True,
                OBJECTIVE: score,
                "trial": asdict(candidate),
            }
            packaged_ranks.append(signature)
            slots -= 1
            print(f"Ready {candidate.name}", flush=True)
        persist(args, manifest, rows, staging)
    print(
        f"Completed {len(rows)} trials and packaged {len(manifest['models'])} models.", flush=True
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--trials", type=int, help="Limit the number of configurations evaluated.")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Test narrower residual models across four chronological windows.",
    )
    parser.add_argument("--packaged", type=int, default=24)
    parser.add_argument("--estimators", type=int, default=120)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fold-expeds", type=int, default=70)
    parser.add_argument("--embargo-expeds", type=int, default=40)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--publish-batch", type=int, default=6)
    parser.add_argument("--max-rank-corr", type=float, default=0.96)
    parser.add_argument("--max-score-gap", type=float, default=0.08)
    parser.add_argument(
        "--audit-top", type=int, help="Replay this many packaged candidates on earlier expeds."
    )
    parser.add_argument(
        "--promote-top", type=int, help="Package late leaders into a separate -promoted directory."
    )
    args = parser.parse_args()
    if args.artifacts is None:
        args.artifacts = Path("artifacts/refined-trees" if args.refine else "artifacts/wide-trees")
    if args.trials is None:
        args.trials = len(trial_grid(args.refine))
    if (
        min(
            args.trials,
            args.packaged,
            args.estimators,
            args.threads,
            args.batch,
            args.publish_batch,
        )
        < 1
        or (args.audit_top is not None and args.audit_top < 1)
        or (args.promote_top is not None and (args.promote_top < 1 or args.audit_top))
        or args.embargo_expeds < 20
        or not 0 < args.max_rank_corr < 1
        or args.max_score_gap < 0
    ):
        parser.error("Use positive counts and threshold values, and at least 20 embargo expeds.")
    try:
        return search(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Tree sweep failed. {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
