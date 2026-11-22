"""Test query-grouped LightGBM ranking objectives on fixed historical windows."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib.metadata import version
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from hackathon_models import (
    LIBRARIES,
    benchmark_residual_target,
    check_predictions,
    dataset_columns,
    event_proxy_scale,
    feature_stability,
    fitted_predictor,
    package,
    select_features,
    stable_feature_subset,
    weighted_rank_prediction,
)
from hackathon_scoring import align_benchmark, event_metrics, score_expeds
from research_tree_sweep import (
    CAPACITIES,
    OBJECTIVE,
    Trial,
    build_fitter,
    build_windows,
    row_window,
)

ANCHORS = (
    Trial(360, "residual12", 1.0, "compact"),
    Trial(1000, "residual12", 1.0, "compact"),
)
BLEND_WEIGHTS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class RankTrial:
    objective: str
    history: int
    features: int
    strength: float
    capacity: str
    grade_mode: str = "percentile"

    @property
    def name(self):
        stem = (
            f"ranker-{self.objective}-{self.features}f-{self.history}-"
            f"r{round(self.strength * 100)}-{self.capacity}"
        )
        return f"{stem}-magnitude" if self.grade_mode == "magnitude" else stem


def configurations():
    early = [
        RankTrial("rank_xendcg", 360, 12, 1, "compact"),
        RankTrial("lambdarank", 360, 12, 1, "compact"),
        RankTrial("rank_xendcg", 1000, 12, 1, "compact"),
        RankTrial("lambdarank", 720, 24, 0.95, "regular"),
        RankTrial("rank_xendcg", 360, 24, 0, "compact"),
        RankTrial("lambdarank", 720, 24, 0, "compact"),
    ]
    remaining = sorted(
        {
            RankTrial(*row)
            for row in product(
                ("rank_xendcg", "lambdarank"),
                (240, 360, 720, 1000),
                (8, 12, 24),
                (0.9, 1),
                ("compact", "regular"),
            )
        }
        - set(early),
        key=lambda trial: trial.name,
    )
    random.Random(1122).shuffle(remaining)
    return early + remaining


def relevance(labels, groups, levels, grade_count, mode="percentile"):
    """Keep discrete target ties, or quantize residual rank within each query."""
    if levels is not None:
        result = labels.map({value: i for i, value in enumerate(levels)})
        if result.isna().any():
            raise ValueError("Found a primary label outside the declared target encoding.")
        return result.to_numpy(dtype=np.int32)
    grouped = labels.groupby(groups, sort=False)
    if mode == "magnitude":
        spread = grouped.transform("std")
        centered = labels - grouped.transform("mean")
        scaled = centered.div(spread.where(spread.gt(0))).fillna(0)
        grade_position = ((scaled + 2) / 4).clip(0, 1)
    else:
        grade_position = (grouped.rank() - 0.5) / grouped.transform("count")
    return np.minimum(grade_count - 1, (grade_position * grade_count).astype("int32")).to_numpy()


def ranker_fitter(train, columns, missing, target, exped, benchmark, levels, args):
    @lru_cache(maxsize=30)
    def prepared(calendar, length):
        frame = row_window(train, exped, target, calendar[-length:]).sort_values(
            exped, kind="stable"
        )
        profile = feature_stability(frame, columns, missing, target, exped, benchmark)
        groups = frame.groupby(exped, sort=False).size().to_numpy()
        return frame, profile, groups

    @lru_cache(maxsize=72)
    def grades(calendar, length, strength, mode):
        frame, _, _ = prepared(calendar, length)
        if strength:
            labels = benchmark_residual_target(frame[target], benchmark, frame[exped], strength)
            return relevance(labels, frame[exped], None, args.grades, mode)
        return relevance(frame[target], frame[exped], levels, args.grades, mode)

    def fit(trial, calendar):
        frame, profile, groups = prepared(calendar, trial.history)
        selected = stable_feature_subset(
            profile, "residual" if trial.strength else "fit", trial.features
        )
        features = select_features(frame, selected, missing)
        labels = grades(calendar, trial.history, trial.strength, trial.grade_mode)
        params = {}
        if trial.objective == "lambdarank":
            max_grade = args.grades if trial.strength or levels is None else len(levels)
            params = {
                "label_gain": list(range(max_grade)),
                "lambdarank_truncation_level": int(np.quantile(groups, 0.9)),
            }
        model = lgb.LGBMRanker(
            **CAPACITIES[trial.capacity],
            **params,
            objective=trial.objective,
            n_estimators=args.estimators,
            learning_rate=0.055,
            colsample_bytree=0.75,
            max_bin=31,
            random_state=821,
            n_jobs=args.threads,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(features, labels, group=groups)
        predict = fitted_predictor(model, selected, missing, "lightgbm", "native")
        return weighted_rank_prediction({trial.name: predict}, exped, {trial.name: 1.0})

    return fit


def predictions(fit, trial, folds, exped, columns):
    return [
        check_predictions(fit(trial, calendar), heldout[[exped, *columns]])["prediction"]
        for calendar, heldout in folds
    ]


def summarize(name, vectors, folds, exped, target, benchmark, scale):
    scores = []
    for values, (_, frame) in zip(vectors, folds, strict=True):
        metrics = score_expeds(frame, exped, target, {name: values}, benchmark)
        if metrics["unq_covariance"].notna().mean() < 0.8:
            raise ValueError("A historical fold lacks enough aligned benchmark scoring.")
        scores.append(
            {
                "fit": metrics["fit"].mean(),
                "unq": metrics["unq_covariance"].mean(),
                "benchmark_corr": metrics["benchmark_rank_corr"].mean(),
                **event_metrics(metrics, scale),
            }
        )
    fold_scores = pd.DataFrame(scores)
    record = fold_scores.mean().to_dict()
    return {
        "candidate": name,
        **record,
        "worst_fold": fold_scores[OBJECTIVE].min(),
        "fold_sd": fold_scores[OBJECTIVE].std(ddof=0),
    }


def mixed(first, second, weight):
    return [left * weight + right * (1 - weight) for left, right in zip(first, second, strict=True)]


def write_summary(frame, destination):
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def run(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    scale, fallback = event_proxy_scale(json.loads((args.data_dir / "scoring.json").read_text()))
    if fallback or scale is None:
        raise ValueError("The event score scale could not be validated.")
    frame = pd.read_parquet(args.data_dir / "train.parquet")
    if (
        frame.index.nlevels != 1
        or not frame.index.is_unique
        or frame.index.hasnans
        or not all(isinstance(row_id, str) and row_id for row_id in frame.index)
        or ("data_type" in frame and set(frame["data_type"].dropna().unique()) - {"train"})
    ):
        raise ValueError("Expected opaque unique IDs and the event's labeled train split.")
    target, exped, columns, missing = dataset_columns(frame, schema)
    frame = frame[[exped, target, *columns]]
    calendar = tuple(sorted(frame[exped].unique()))
    benchmark, _ = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"),
        frame,
        exped,
        minimum_coverage=0.8,
    )
    if benchmark is None:
        raise ValueError("The official train benchmark is required.")
    levels = (schema.get("target_encoding") or {}).get("values")
    levels = sorted(levels) if isinstance(levels, list) and len(levels) >= 2 else None

    windows, folds = build_windows(
        frame, calendar, exped, target, 4, args.fold_expeds, args.embargo
    )
    audit_dates = calendar[: calendar.index(windows[0]["holdout_expeds"][0])]
    audit_windows, audit = build_windows(
        frame, audit_dates, exped, target, 2, args.fold_expeds, args.embargo
    )
    fit_ranker = ranker_fitter(frame, columns, missing, target, exped, benchmark, levels, args)
    fit_tree = build_fitter(
        frame,
        columns,
        missing,
        target,
        exped,
        benchmark,
        SimpleNamespace(estimators=105, threads=args.threads),
    )

    def score(name, vectors, splits):
        return summarize(name, vectors, splits, exped, target, benchmark, scale)

    anchor_vectors = {
        trial.name: predictions(fit_tree, trial, folds, exped, columns) for trial in ANCHORS
    }
    anchors = [score(name, vectors, folds) for name, vectors in anchor_vectors.items()]
    baseline = max(anchors, key=lambda record: record[OBJECTIVE])
    template = next(trial for trial in ANCHORS if trial.name == baseline["candidate"])
    print(f"Regression anchor {template.name} {baseline[OBJECTIVE]:+.5f}", flush=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="ranker-", dir=args.artifacts))
    leaderboard = staging / "leaderboard.csv"
    rows, helpful = list(anchors), []
    trials = [
        replace(trial, grade_mode=args.grade_mode)
        for trial in configurations()
        if args.grade_mode == "percentile" or trial.strength
    ][: args.trials]
    for index, trial in enumerate(trials, 1):
        vectors = predictions(fit_ranker, trial, folds, exped, columns)
        scored = [score(trial.name, vectors, folds)]
        for weight in BLEND_WEIGHTS:
            name = f"{trial.name}-blend{round(weight * 100)}"
            blend = mixed(vectors, anchor_vectors[template.name], weight)
            scored.append(score(name, blend, folds))
        for weight, summary in zip((1.0, *BLEND_WEIGHTS), scored, strict=True):
            summary.update(**asdict(trial), ranker_weight=weight)
            if summary[OBJECTIVE] > baseline[OBJECTIVE] + args.min_gain:
                helpful.append((trial, summary))
        rows.extend(scored)
        best = max(scored, key=lambda record: record[OBJECTIVE])
        print(f"{index}/{len(trials)} {best['candidate']} {best[OBJECTIVE]:+.5f}", flush=True)
        if index % 6 == 0 or index == len(trials):
            write_summary(pd.DataFrame(rows).sort_values(OBJECTIVE, ascending=False), leaderboard)

    anchor_audit = predictions(fit_tree, template, audit, exped, columns)
    audit_baseline = score(template.name, anchor_audit, audit)
    audit_rows = [audit_baseline]
    models = {}
    prescreen = sorted(helpful, key=lambda entry: entry[1][OBJECTIVE], reverse=True)
    attempted = set()
    for trial, summary in prescreen:
        if trial in attempted:
            continue
        attempted.add(trial)
        weight = summary["ranker_weight"]
        vectors = predictions(fit_ranker, trial, audit, exped, columns)
        candidate = summary["candidate"]
        metrics = score(candidate, mixed(vectors, anchor_audit, weight), audit)
        extra_gain = metrics[OBJECTIVE] - audit_baseline[OBJECTIVE]
        combined_gain = (2 * (summary[OBJECTIVE] - baseline[OBJECTIVE]) + extra_gain) / 3
        audit_rows.append({**metrics, "six_fold_gain": combined_gain})
        print(f"Audit {candidate} {metrics[OBJECTIVE]:+.5f}; six-fold gain {combined_gain:+.5f}")
        if extra_gain < -args.audit_tolerance or combined_gain <= args.min_gain:
            continue
        predictor = fit_ranker(trial, calendar)
        if weight != 1:
            predictor = weighted_rank_prediction(
                {"ranker": predictor, "tree": fit_tree(template, calendar)},
                exped,
                {"ranker": weight, "tree": 1 - weight},
            )
        sample = row_window(frame, exped, target, windows[-1]["holdout_expeds"][-3:])
        record = package({candidate: predictor}, sample, exped, staging)[candidate]
        models[candidate] = {
            **record,
            "prevalidated": True,
            "trial": asdict(trial),
            OBJECTIVE: float(summary[OBJECTIVE]),
            "audit_round_score_proxy": float(metrics[OBJECTIVE]),
            "ranker_weight": weight,
        }
        if len(models) == args.packaged:
            break

    write_summary(pd.DataFrame(audit_rows), staging / "audit.csv")
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": {library: version(library) for library in LIBRARIES},
        "selection_metric": OBJECTIVE,
        "windows": windows,
        "audit_windows": audit_windows,
        "score_scale": scale,
        "baseline": baseline["candidate"],
        "models": models,
        "primary_model": next(iter(models), None),
        "fallback_order": list(models),
        "leaderboard": f"{staging.name}/leaderboard.csv",
        "audit": f"{staging.name}/audit.csv",
    }
    staged_manifest = staging / "manifest.json"
    staged_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    staged_manifest.replace(args.artifacts / "manifest.json")
    print(f"Completed {len(trials)} ranker configurations; packaged {len(models)}.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/wide-ranker"))
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--estimators", type=int, default=105)
    parser.add_argument("--grades", type=int, default=7)
    parser.add_argument("--grade-mode", choices=("percentile", "magnitude"), default="percentile")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fold-expeds", type=int, default=70)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--packaged", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=0.002)
    parser.add_argument("--audit-tolerance", type=float, default=0.025)
    args = parser.parse_args()
    if (
        min(args.trials, args.estimators, args.threads, args.packaged) < 1
        or not 2 <= args.grades <= 12
        or args.embargo < 20
        or args.min_gain < 0
        or args.audit_tolerance < 0
    ):
        parser.error(
            "Use positive fit counts, 2 to 12 relevance grades, and an embargo of at least 20."
        )
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Ranker sweep failed. {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
