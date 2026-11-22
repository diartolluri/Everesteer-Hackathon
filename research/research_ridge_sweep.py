"""Build a diverse portfolio of auxiliary and residual Ridge candidates."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from hackathon_models import (
    LIBRARIES,
    benchmark_residual_target,
    chronological_split,
    dataset_columns,
    event_proxy_scale,
    feature_stability,
    fitted_predictor,
    package,
    ridge_design,
    select_features,
    stable_feature_subset,
    walk_forward_splits,
)
from hackathon_scoring import align_benchmark, core_reference, resolve_core_features
from research.research_experiments import (
    ResearchData,
    atomic_json,
    evaluate_candidates,
    rank_candidates,
    single,
)


@dataclass(frozen=True)
class Profile:
    name: str
    expeds: int
    alpha: float
    selection: str | None = None
    size: int = 0
    hedge: float = 0.0


PROFILES = (
    Profile("short", 500, 10.0),
    Profile("long", 2600, 60.0),
    Profile("stable", 1400, 30.0, "fit", 72),
    Profile("residual", 1400, 40.0, "residual", 64, 0.65),
)


def candidate_name(profile, target, primary):
    suffix = "graded" if target == primary else target.removeprefix("target_").lower()
    return f"wide-ridge-{profile.name}-{suffix}"


def fit_profile(data, profile, expeds, targets):
    fit = data.labeled_fit(expeds, profile.expeds)
    labels = fit[list(targets)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(labels) < 3:
        raise ValueError(f"No complete target rows in {profile.name!r} fit.")
    fit = fit.loc[labels.index]
    # Orient each target using only this window's labeled training data.
    orientation = labels.corrwith(fit[data.target]).lt(0).replace({True: -1, False: 1})
    labels = labels.mul(orientation)
    columns = data.columns
    if profile.selection:
        stability = feature_stability(
            fit, columns, data.missing, data.target, data.exped, data.benchmark
        )
        columns = stable_feature_subset(stability, profile.selection, profile.size)
    if profile.hedge:
        labels = labels.apply(
            benchmark_residual_target,
            args=(data.benchmark, fit[data.exped], profile.hedge),
        )
    matrix = ridge_design(select_features(fit, columns, data.missing)).astype(np.float64)
    joint = Ridge(alpha=profile.alpha, solver="cholesky", copy_X=False).fit(matrix, labels)
    predictors = {}
    for target, coefficient, intercept in zip(targets, joint.coef_, joint.intercept_, strict=True):
        model = copy(joint)
        model.coef_ = coefficient
        model.intercept_ = intercept
        predictors[candidate_name(profile, target, data.target)] = fitted_predictor(
            model, columns, data.missing, "ridge"
        )
    return predictors


def forecast(data, expeds, heldout, profile_targets):
    forecasts = {}
    features = heldout[[data.exped, *data.columns]]
    for profile, targets in profile_targets.items():
        predictors = fit_profile(data, profile, expeds, targets)
        forecasts.update(
            {name: model(features)["prediction"] for name, model in predictors.items()}
        )
    return pd.DataFrame(forecasts, index=features.index)


def cross_sectional_similarity(windows, exped):
    correlations = (
        frame.groupby(heldout[exped], sort=False).rank().groupby(heldout[exped], sort=False)
        for _, heldout, frame in windows
    )
    return pd.concat([groups.corr().abs() for groups in correlations]).groupby(level=1).mean()


def select_portfolio(ranking, similarities, objective, count, limit):
    selected = []
    for name in ranking.loc[ranking["eligible"] & ranking[objective].notna(), "candidate"]:
        if selected and not similarities.loc[name, selected].lt(limit).all():
            continue
        selected.append(name)
        if len(selected) == count:
            break
    return selected


def score_windows(data, windows, scale, args, stage):
    names = [single(name) for name in windows[0][2]]
    scores, _ = evaluate_candidates(data, names, windows, scale, args, stage)
    objective = "event_score" if data.core_features else "round_score_proxy"
    return scores, rank_candidates(scores, names, len(windows), objective), objective


def run(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    settings = json.loads((args.data_dir / "scoring.json").read_text())
    scale, error = event_proxy_scale(settings)
    if error or scale is None:
        raise ValueError("The saved scoring settings do not specify the event formula and scale.")
    train = pd.read_parquet(args.data_dir / "train.parquet")
    if train.index.nlevels != 1 or not train.index.is_unique or train.index.hasnans:
        raise ValueError("The local training data needs unique nonmissing IDs.")
    target, exped, columns, missing = dataset_columns(train, schema)
    core_features = resolve_core_features(schema)
    benchmark, coverage = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"), train[[exped]], exped
    )
    if benchmark is None:
        raise ValueError("The local train benchmark has insufficient alignment.")
    exposure = core_reference(train[core_features], missing=missing) if core_features else None
    data = ResearchData(train, columns, target, exped, missing, benchmark, core_features, exposure)
    split = chronological_split(train[exped], args.audit_expeds, args.embargo_expeds)
    folds = walk_forward_splits(split, args.folds, args.fold_expeds, args.embargo_expeds)
    screen = data.labeled_fit(folds[0]["fit_expeds"], 2600)
    targets = [target]
    for name in schema.get("targets", []):
        if name != target and name in screen and np.isfinite(screen[name]).mean() >= 0.97:
            targets.append(name)
    if len(targets) < 3:
        raise ValueError("At least two well-covered auxiliary targets are required for the sweep.")
    portfolio_configs = {p: tuple(targets) for p in PROFILES}
    windows = []
    for fold in folds:
        print(f"Fitting {len(targets) * len(PROFILES)} candidates for {fold['name']}.", flush=True)
        heldout = train.loc[train[exped].isin(fold["holdout_expeds"])]
        predictions = forecast(data, fold["fit_expeds"], heldout, portfolio_configs)
        windows.append((fold["name"], heldout, predictions))
    print("Scoring the out-of-fold forecasts and measuring cross-sectional similarity.", flush=True)
    fold_scores, ranking, objective = score_windows(data, windows, scale, args, "walk-forward")
    similarity = cross_sectional_similarity(windows, exped)
    selected = select_portfolio(ranking, similarity, objective, args.finalists, args.max_similarity)
    if not selected:
        raise ValueError("No candidates passed the four-window score and diversity checks.")

    def profiles_for(names):
        return {
            profile: portfolio_configs[profile]
            for profile in PROFILES
            if any(candidate_name(profile, name, target) in names for name in targets)
        }

    print(f"Auditing {len(selected)} diversified finalists on the held-out window.", flush=True)
    audit = train.loc[train[exped].isin(split["holdout_expeds"])]
    audit_pred = forecast(data, split["fit_expeds"], audit, profiles_for(selected))[selected]
    audit_scores, audit_ranking, _ = score_windows(
        data, [("final-holdout", audit, audit_pred)], scale, args, "audit"
    )
    audited = set(audit_ranking.loc[audit_ranking["eligible"], "candidate"])
    selected = [name for name in selected if name in audited]
    if not selected:
        raise ValueError("No selected candidate could be scored on the final holdout.")
    print(f"Refitting and packaging {len(selected)} candidates.", flush=True)
    ordered = sorted(train[exped].unique())
    predictors = {}
    for profile, labels in profiles_for(selected).items():
        predictors.update(fit_profile(data, profile, ordered, labels))
    args.artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sweep-", dir=args.artifacts))
    sample = audit.loc[audit[exped].isin(split["holdout_expeds"][-3:]), [exped, *columns]]
    models = package({name: predictors[name] for name in selected}, sample, exped, output)
    for model in models.values():
        model["prevalidated"] = True
    report = ranking.merge(
        audit_ranking[["candidate", objective]], on="candidate", how="left", suffixes=("", "_audit")
    )
    report["selected"] = report["candidate"].isin(selected)
    report.to_csv(output / "leaderboard.csv", index=False)
    (output / "leaderboard.json").write_text(report.to_json(orient="records", indent=2) + "\n")
    pd.concat([fold_scores, audit_scores]).to_csv(output / "fold_metrics.csv", index=False)
    similarity.loc[selected, selected].to_csv(output / "candidate_similarity.csv")
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "experiment": "ridge_target_and_regime_sweep",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": {library: version(library) for library in LIBRARIES},
        "selection_metric": objective,
        "score_scale": scale,
        "zero_rule": args.zero_rule,
        "benchmark_coverage": coverage,
        "core_features": core_features,
        "primary_model": selected[0],
        "fallback_order": selected,
        "models": models,
        "targets": targets,
        "profiles": [asdict(profile) for profile in PROFILES],
        "embargo_expeds": args.embargo_expeds,
        "max_similarity": args.max_similarity,
        "windows": [*folds, {**split, "name": "final-holdout"}],
        "leaderboard": f"{output.name}/leaderboard.csv",
        "leaderboard_json": f"{output.name}/leaderboard.json",
        "fold_metrics": f"{output.name}/fold_metrics.csv",
        "candidate_similarity": f"{output.name}/candidate_similarity.csv",
    }
    atomic_json(args.artifacts / "manifest.json", manifest, output)
    print(
        report.loc[
            report["selected"], ["candidate", "fit", "unq", objective, f"{objective}_audit"]
        ].to_string(index=False),
        flush=True,
    )
    print(
        f"Ready: {len(selected)} named candidates in {args.artifacts / 'manifest.json'}.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/wide-ridge"))
    parser.add_argument("--folds", type=int, default=4, choices=range(4, 7))
    parser.add_argument("--fold-expeds", type=int, default=30)
    parser.add_argument("--audit-expeds", type=int, default=100)
    parser.add_argument("--embargo-expeds", type=int, default=80)
    parser.add_argument("--finalists", type=int, default=24)
    parser.add_argument("--max-similarity", type=float, default=0.94)
    args = parser.parse_args()
    args.zero_rule = False
    if args.finalists < 1 or not 0 < args.max_similarity < 1:
        parser.error("Finalist count must be positive and similarity must be between zero and one.")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Ridge sweep failed. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
