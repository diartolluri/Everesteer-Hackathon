"""Test categorical additive Ridge and learned bin interactions on sealed-event train data."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from hackathon_models import (
    BASE_CONFIGS,
    LIBRARIES,
    benchmark_residual_target,
    chronological_split,
    dataset_columns,
    event_proxy_scale,
    feature_stability,
    model_predictions,
    package,
    select_features,
    stable_feature_subset,
    walk_forward_splits,
)
from hackathon_scoring import align_benchmark, core_reference, resolve_core_features
from research_experiments import ResearchData, atomic_json
from research_ridge_sweep import cross_sectional_similarity, score_windows, select_portfolio

BASELINE = "ordinal-ridge-reference"
AUXILIARY_TARGETS = (
    "target_Tinzar",
    "target_Tignousti",
    "target_Tasserirt",
    "target_Ouaoulzat",
    "target_Timgarine",
)


@dataclass(frozen=True)
class Profile:
    name: str
    expeds: int
    features: int
    alpha: float
    selection: str
    residual_strength: float = 0.0
    interactions: int = 0


PROFILES = (
    Profile("short-additive", 560, 56, 27.0, "fit"),
    Profile("long-additive", 1400, 72, 54.0, "fit"),
    Profile("residual-additive", 1400, 64, 50.0, "residual", 0.65),
    Profile("residual-interactions", 980, 56, 64.0, "residual", 0.65, 12),
)


def interaction_zones(values, lower, upper):
    return np.where(
        np.isfinite(values), 1 + (values > lower).astype(np.int8) + (values > upper), 0
    ).astype(np.int8)


def choose_interactions(features, labels, count):
    if not count:
        return (), np.empty(0), np.empty(0)
    width = min(16, features.shape[1])
    values = features[:, :width]
    lower, upper = np.nanquantile(values, [1 / 3, 2 / 3], axis=0)
    zones = interaction_zones(values, lower, upper).astype(float)
    zones[zones == 0] = 2
    zones -= zones.mean(axis=0)
    centered = labels - labels.mean()
    label_norm = np.linalg.norm(centered)
    scored = []
    for left, right in combinations(range(width), 2):
        product = zones[:, left] * zones[:, right]
        product -= product.mean()
        norm = np.linalg.norm(product) * label_norm
        if norm > 0:
            score = abs(float(product @ centered)) / norm
            scored.append((-score, left, right))
    return (tuple((left, right) for _, left, right in sorted(scored)[:count]), lower, upper)


def categorical_inputs(features, interactions, lower, upper):
    if not interactions:
        return features
    zones = interaction_zones(features[:, : len(lower)], lower, upper)
    extra = np.column_stack([4 * zones[:, left] + zones[:, right] for left, right in interactions])
    return np.column_stack((features, extra))


def categorical_predictor(model, encoder, columns, missing, interactions, lower, upper):
    def predict(live_features):
        matrix = select_features(live_features, columns, missing).to_numpy()
        inputs = categorical_inputs(matrix, interactions, lower, upper)
        result = model.predict(encoder.transform(inputs))
        return pd.DataFrame({"prediction": result}, index=live_features.index)

    predict.feature_columns = tuple(columns)
    return predict


def candidate_name(profile, name, primary):
    target = "graded" if name == primary else name.removeprefix("target_").lower()
    return f"categorical-{profile.name}-{target}"


def fit_profile(data, profile, expeds, targets):
    fit = data.labeled_fit(expeds, profile.expeds)
    labels = fit[list(targets)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(labels) < 3:
        raise ValueError(f"The {profile.name!r} fit has fewer than three complete target rows.")
    fit = fit.loc[labels.index]
    orientation = labels.corrwith(fit[data.target]).lt(0).replace({True: -1, False: 1})
    labels = labels.mul(orientation)
    stability = feature_stability(
        fit, data.columns, data.missing, data.target, data.exped, data.benchmark
    )
    columns = stable_feature_subset(stability, profile.selection, profile.features)
    if profile.residual_strength:
        labels = labels.apply(
            benchmark_residual_target,
            args=(data.benchmark, fit[data.exped], profile.residual_strength),
        )
    features = select_features(fit, columns, data.missing).to_numpy()
    interactions, lower, upper = choose_interactions(
        features, labels[data.target].to_numpy(), profile.interactions
    )
    inputs = categorical_inputs(features, interactions, lower, upper)
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=24, dtype=np.float32
    )
    matrix = encoder.fit_transform(inputs)
    joint = Ridge(alpha=profile.alpha, solver="lsqr", tol=1e-3, max_iter=80).fit(matrix, labels)
    predictors = {}
    for target, coefficient, intercept in zip(targets, joint.coef_, joint.intercept_, strict=True):
        model = copy(joint)
        model.coef_, model.intercept_ = coefficient, intercept
        predictors[candidate_name(profile, target, data.target)] = categorical_predictor(
            model, encoder, columns, data.missing, interactions, lower, upper
        )
    return predictors


def forecast(data, split, targets, profiles, *, include_baseline):
    heldout = data.train.loc[data.train[data.exped].isin(split["holdout_expeds"])]
    features = heldout[[data.exped, *data.columns]]
    predictions = {}
    if include_baseline:
        fit = data.labeled_fit(split["fit_expeds"], 1400)
        baseline = model_predictions(
            fit, data.columns, data.missing, data.target, configs={BASELINE: BASE_CONFIGS["ridge"]}
        )[BASELINE]
        predictions[BASELINE] = baseline(features)["prediction"]
    for profile in profiles:
        models = fit_profile(data, profile, split["fit_expeds"], targets)
        predictions.update({name: model(features)["prediction"] for name, model in models.items()})
    return split["name"], heldout, pd.DataFrame(predictions, index=features.index)


def run(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    settings = json.loads((args.data_dir / "scoring.json").read_text())
    scale, error = event_proxy_scale(settings)
    if error or scale is None:
        raise ValueError("The saved scoring settings must specify the event formula and scale.")
    train = pd.read_parquet(args.data_dir / "train.parquet")
    if train.index.nlevels != 1 or not train.index.is_unique or train.index.hasnans:
        raise ValueError("Training needs a unique, nonmissing row ID index.")
    target, exped, columns, missing = dataset_columns(train, schema)
    core_features = resolve_core_features(schema)
    benchmark, coverage = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"), train[[exped]], exped
    )
    if benchmark is None:
        raise ValueError("The training benchmark must have aligned rows.")
    exposure = core_reference(train[core_features], missing=missing) if core_features else None
    data = ResearchData(train, columns, target, exped, missing, benchmark, core_features, exposure)
    targets = [
        target,
        *(
            name
            for name in AUXILIARY_TARGETS
            if name in schema.get("targets", ()) and name in train
        ),
    ]
    if len(targets) < 2:
        raise ValueError("The sweep needs the graded target and an available auxiliary target.")
    split = chronological_split(train[exped], args.audit_expeds, args.embargo_expeds)
    folds = walk_forward_splits(split, args.folds, args.fold_expeds, args.embargo_expeds)
    windows = []
    for fold in folds:
        print(
            f"Fitting {len(PROFILES) * len(targets)} alternatives for {fold['name']}.", flush=True
        )
        windows.append(forecast(data, fold, targets, PROFILES, include_baseline=True))
    fold_scores, ranking, objective = score_windows(data, windows, scale, args, "walk-forward")
    ordered = ranking.loc[ranking["candidate"].eq(BASELINE), objective]
    if len(ordered) != 1 or pd.isna(ordered.iloc[0]):
        raise ValueError("The ordinal Ridge baseline must have a finite out-of-fold score.")
    baseline = float(ordered.iloc[0])
    leaders = ranking.copy()
    leaders["eligible"] &= leaders[objective].gt(baseline) & leaders["candidate"].ne(BASELINE)
    similarities = cross_sectional_similarity(windows, exped)
    selected = select_portfolio(
        leaders, similarities, objective, args.finalists, args.max_similarity
    )
    print(
        ranking[["candidate", "fit", "unq", objective]].head(15).to_string(index=False), flush=True
    )
    if not selected:
        raise ValueError("No categorical candidate improved on the ordinal Ridge baseline.")
    tested_profiles = [
        profile
        for profile in PROFILES
        if any(candidate_name(profile, name, target) in selected for name in targets)
    ]
    print(f"Auditing {len(selected)} finalists and the ordinal Ridge reference.", flush=True)
    audit = forecast(
        data, {**split, "name": "final-holdout"}, targets, tested_profiles, include_baseline=True
    )
    audit_scores, audit_ranking, _ = score_windows(
        data, [(audit[0], audit[1], audit[2][[BASELINE, *selected]])], scale, args, "audit"
    )
    audit_eligible = set(audit_ranking.loc[audit_ranking["eligible"], "candidate"])
    selected = [name for name in selected if name in audit_eligible]
    if not selected:
        raise ValueError("No selected candidate completed the separate holdout audit.")
    print(f"Refitting and validating {len(selected)} deployment artifacts.", flush=True)
    ordered_expeds = sorted(train[exped].unique())
    predictors = {}
    for profile in tested_profiles:
        predictors.update(fit_profile(data, profile, ordered_expeds, targets))
    args.artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="categorical-", dir=args.artifacts))
    sample = audit[1].loc[audit[1][exped].isin(split["holdout_expeds"][-3:]), [exped, *columns]]
    if __name__ != "__main__":
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
    models = package({name: predictors[name] for name in selected}, sample, exped, output)
    for model in models.values():
        model["prevalidated"] = True
    report = ranking.merge(
        audit_ranking[["candidate", objective]], on="candidate", how="left", suffixes=("", "_audit")
    )
    report["selected"] = report["candidate"].isin(selected)
    report.to_csv(output / "leaderboard.csv", index=False)
    pd.concat([fold_scores, audit_scores]).to_csv(output / "fold_metrics.csv", index=False)
    similarities.loc[selected, selected].to_csv(output / "candidate_similarity.csv")
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "experiment": "categorical_main_effects_and_pair_interactions",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": {library: version(library) for library in LIBRARIES},
        "selection_metric": objective,
        "score_scale": scale,
        "core_features": core_features,
        "benchmark_coverage": coverage,
        "primary_model": selected[0],
        "fallback_order": selected,
        "models": models,
        "targets": targets,
        "profiles": [asdict(profile) for profile in tested_profiles],
        "embargo_expeds": args.embargo_expeds,
        "windows": [*folds, {**split, "name": "final-holdout"}],
        "leaderboard": f"{output.name}/leaderboard.csv",
        "fold_metrics": f"{output.name}/fold_metrics.csv",
        "candidate_similarity": f"{output.name}/candidate_similarity.csv",
    }
    atomic_json(args.artifacts / "manifest.json", manifest, output)
    print(
        report.loc[
            report["selected"] | report["candidate"].eq(BASELINE),
            ["candidate", "fit", "unq", objective, f"{objective}_audit"],
        ].to_string(index=False),
        flush=True,
    )
    print(
        f"Ready: {len(models)} named categorical candidates in {args.artifacts / 'manifest.json'}.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/wide-alternative"))
    parser.add_argument("--folds", type=int, default=4, choices=range(4, 7))
    parser.add_argument("--fold-expeds", type=int, default=30)
    parser.add_argument("--audit-expeds", type=int, default=100)
    parser.add_argument("--embargo-expeds", type=int, default=80)
    parser.add_argument("--finalists", type=int, default=10)
    parser.add_argument("--max-similarity", type=float, default=0.96)
    args = parser.parse_args()
    args.zero_rule = False
    if args.finalists < 1 or not 0 < args.max_similarity < 1:
        parser.error("Finalist count must be positive and similarity must be between zero and one.")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Categorical sweep failed. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
