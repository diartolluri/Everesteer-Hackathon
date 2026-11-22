"""Refine the residual Ridge family using labeled historical forecasts only."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from hackathon_models import (
    LIBRARIES,
    benchmark_residual_target,
    check_predictions,
    chronological_split,
    dataset_columns,
    event_proxy_scale,
    feature_stability,
    package,
    ridge_design,
    select_features,
    stable_feature_subset,
    walk_forward_splits,
    weighted_rank_prediction,
)
from hackathon_scoring import align_benchmark, core_reference, resolve_core_features
from research_experiments import ResearchData, atomic_json
from research_ridge_sweep import PROFILES as PRIOR_PROFILES
from research_ridge_sweep import cross_sectional_similarity, score_windows, select_portfolio
from research_ridge_sweep import fit_profile as fit_prior

TARGETS = ("target_Tidiquin", "target_Toubkal", "target_Timgarine", "target_Ouaoulzat")
REFERENCE = {
    target: f"wide-ridge-residual-{target.removeprefix('target_').lower()}" for target in TARGETS
}
TARGET_BLEND = {"target_Tidiquin": 0.4, "target_Toubkal": 0.3, "target_Timgarine": 0.3}


@dataclass(frozen=True)
class Profile:
    name: str
    expeds: int
    features: int
    representation: str
    hedge: float
    alpha: float
    selection: str = "residual"
    tail_clip: float = 0.0
    equal_expeds: bool = False


PROFILES = (
    Profile("recent-ordinal", 360, 32, "ordinal", 0.90, 36.0),
    Profile("short-unhedged", 620, 48, "ordinal", 0.20, 40.0),
    Profile("short-ordinal", 620, 48, "ordinal", 1.00, 55.0),
    Profile("middle-narrow", 950, 24, "ordinal", 0.90, 50.0),
    Profile("middle-wide", 1050, 72, "ordinal", 0.92, 70.0),
    Profile("long-dual", 2200, 64, "ordinal", 0.82, 110.0, "dual"),
    Profile("recent-ranks", 460, 36, "rank", 0.90, 44.0),
    Profile("middle-ranks", 850, 48, "rank", 1.00, 67.0, equal_expeds=True),
    Profile("long-ranks", 1600, 64, "rank", 0.86, 98.0, "dual", equal_expeds=True),
    Profile("middle-hinges", 860, 36, "hinge", 0.91, 100.0),
    Profile("long-hinges", 1560, 56, "hinge", 0.82, 145.0, "dual"),
    Profile("specific-ranks", 720, 40, "rank", 0.91, 70.0, "target"),
    Profile("specific-ordinal", 1380, 48, "ordinal", 0.90, 79.0, "target"),
    Profile("robust-hinges", 1040, 48, "hinge", 1.00, 125.0, "dual", 0.035, True),
)
BLEND_PROFILES = {"short-ordinal", "middle-ranks", "middle-hinges", "robust-hinges"}


def model_name(profile, target):
    return f"refined-ridge-{profile.name}-{target.removeprefix('target_').lower()}"


def ridge_features(frame, columns, missing, representation, exped, knots):
    matrix = select_features(frame, columns, missing)
    if representation == "rank":
        matrix = matrix.groupby(frame[exped], sort=False).rank(pct=True).mul(10)
    design = ridge_design(matrix)
    if representation != "hinge":
        return design
    values = matrix.to_numpy(dtype=np.float32)
    hinges = [np.maximum(values - knot, 0) for knot in knots]
    for hinge in hinges:
        np.nan_to_num(hinge, copy=False, nan=0.0)
    return np.column_stack((design, *hinges))


def predictor(model, columns, missing, representation, exped, knots):
    def predict(live_features):
        matrix = ridge_features(live_features, columns, missing, representation, exped, knots)
        return pd.DataFrame({"prediction": model.predict(matrix)}, index=live_features.index)

    predict.feature_columns = tuple(columns)
    return predict


def selected_features(stability, profile):
    if profile.selection != "dual":
        return stable_feature_subset(stability, "residual", profile.features)
    first = stable_feature_subset(stability, "residual", profile.features // 2)
    second = stable_feature_subset(stability, "fit", profile.features)
    return list(dict.fromkeys([*first, *second]))[: profile.features]


def fit_design(data, profile, fit, labels, columns):
    knots = np.empty((0, len(columns)), dtype=np.float32)
    if profile.representation == "hinge":
        values = select_features(fit, columns, data.missing).to_numpy(dtype=np.float32)
        knots = np.nanquantile(values, [1 / 3, 2 / 3], axis=0).astype(np.float32)
    matrix = ridge_features(fit, columns, data.missing, profile.representation, data.exped, knots)
    sample_weight = None
    if profile.equal_expeds:
        counts = fit[data.exped].map(fit[data.exped].value_counts())
        sample_weight = 1 / counts
        sample_weight = sample_weight / sample_weight.mean()
    model = Ridge(alpha=profile.alpha, solver="cholesky", copy_X=False).fit(
        matrix.astype(np.float64), labels, sample_weight=sample_weight
    )
    outputs = {}
    if model.coef_.ndim == 1:
        target = labels.columns[0]
        outputs[model_name(profile, target)] = predictor(
            model, columns, data.missing, profile.representation, data.exped, knots
        )
        return outputs
    for target, coefficient, intercept in zip(
        labels.columns, model.coef_, model.intercept_, strict=True
    ):
        fitted = copy(model)
        fitted.coef_, fitted.intercept_ = coefficient, intercept
        outputs[model_name(profile, target)] = predictor(
            fitted, columns, data.missing, profile.representation, data.exped, knots
        )
    return outputs


def fit_new(data, split, profiles, targets):
    estimators = {}
    for profile in profiles:
        fit = data.labeled_fit(split, profile.expeds)
        labels = fit[list(targets)].replace([np.inf, -np.inf], np.nan).dropna()
        if len(labels) < 3:
            raise ValueError(f"No complete labels in the {profile.name!r} fit.")
        fit = fit.loc[labels.index]
        orientation = labels.corrwith(fit[data.target]).lt(0).replace({True: -1, False: 1})
        labels = labels.mul(orientation).apply(
            benchmark_residual_target, args=(data.benchmark, fit[data.exped], profile.hedge)
        )
        if profile.tail_clip:
            lower = labels.quantile(profile.tail_clip)
            upper = labels.quantile(1 - profile.tail_clip)
            labels = labels.clip(lower=lower, upper=upper, axis=1)
        if profile.selection == "target":
            for target in targets:
                stability = feature_stability(
                    fit, data.columns, data.missing, target, data.exped, data.benchmark
                )
                features = selected_features(stability, profile)
                estimators.update(fit_design(data, profile, fit, labels[[target]], features))
        else:
            stability = feature_stability(
                fit, data.columns, data.missing, data.target, data.exped, data.benchmark
            )
            estimators.update(
                fit_design(data, profile, fit, labels, selected_features(stability, profile))
            )
        if profile.name in BLEND_PROFILES and set(TARGET_BLEND) <= set(targets):
            weights = {
                model_name(profile, target): weight for target, weight in TARGET_BLEND.items()
            }
            estimators[model_name(profile, "multi")] = weighted_rank_prediction(
                estimators, data.exped, weights
            )
    return estimators


def forecast(data, split, profiles, targets, prior_targets):
    evaluation = data.train.loc[data.train[data.exped].isin(split["holdout_expeds"])]
    features = evaluation[[data.exped, *data.columns]]
    reference = next(profile for profile in PRIOR_PROFILES if profile.name == "residual")
    prior = fit_prior(data, reference, split["fit_expeds"], prior_targets)
    models = {REFERENCE[target]: prior[REFERENCE[target]] for target in targets}
    models.update(fit_new(data, split["fit_expeds"], profiles, targets))
    predictions = {
        name: check_predictions(model, features)["prediction"] for name, model in models.items()
    }
    return split["name"], evaluation, pd.DataFrame(predictions, index=features.index)


def compare_priors(ranking, profiles, targets, objective):
    reference_scores = ranking.set_index("candidate")[objective]
    reference_worst = ranking.set_index("candidate")["worst_fold"]
    mapping = {
        model_name(profile, target): REFERENCE[target] for profile in profiles for target in targets
    }
    report = ranking.copy()
    report["reference"] = report["candidate"].map(mapping)
    for name in report.loc[report["candidate"].str.endswith("-multi"), "candidate"]:
        refs = [REFERENCE[target] for target in TARGET_BLEND]
        report.loc[report["candidate"].eq(name), "reference"] = reference_scores.loc[refs].idxmax()
    report["mean_gain"] = report[objective] - report["reference"].map(reference_scores)
    report["worst_gain"] = report["worst_fold"] - report["reference"].map(reference_worst)
    return report


def run(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    settings = json.loads((args.data_dir / "scoring.json").read_text())
    scale, error = event_proxy_scale(settings)
    if error or scale is None:
        raise ValueError("The event score settings are incomplete.")
    train = pd.read_parquet(args.data_dir / "train.parquet")
    target, exped, columns, missing = dataset_columns(train, schema)
    if train.index.nlevels != 1 or not train.index.is_unique or train.index.hasnans:
        raise ValueError("The training split needs a unique nonmissing ID index.")
    if set(TARGETS) - set(train.columns):
        raise ValueError("This refinement needs the four labeled event targets.")
    prior_manifest = json.loads((args.prior / "manifest.json").read_text())
    prior_targets = prior_manifest["targets"]
    core_features = resolve_core_features(schema)
    benchmark, coverage = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"), train[[exped]], exped
    )
    if benchmark is None:
        raise ValueError("The official benchmark does not align with labeled train data.")
    exposure = core_reference(train[core_features], missing=missing) if core_features else None
    data = ResearchData(train, columns, target, exped, missing, benchmark, core_features, exposure)
    audit = chronological_split(train[exped], args.audit_expeds, args.embargo_expeds)
    windows = walk_forward_splits(audit, args.folds, args.fold_expeds, args.embargo_expeds)
    forecasts = []
    for fold in windows:
        print(
            f"Fitting {len(PROFILES)} refinement designs and exact controls for {fold['name']}.",
            flush=True,
        )
        forecasts.append(forecast(data, fold, PROFILES, TARGETS, prior_targets))
    scores, ranking, objective = score_windows(data, forecasts, scale, args, "walk-forward")
    comparison = compare_priors(ranking, PROFILES, TARGETS, objective)
    selected_scores = comparison.copy()
    selected_scores["eligible"] &= selected_scores["reference"].notna() & (
        (selected_scores["mean_gain"].gt(0.001) & selected_scores["worst_gain"].ge(-0.025))
        | (selected_scores["worst_gain"].gt(0.006) & selected_scores["mean_gain"].ge(-0.009))
    )
    similarities = cross_sectional_similarity(forecasts, exped)
    candidates = select_portfolio(
        selected_scores, similarities, objective, args.finalists, args.max_similarity
    )
    args.artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="refinement-", dir=args.artifacts))
    comparison.to_csv(output / "leaderboard.csv", index=False)
    if not candidates:
        scores.to_csv(output / "fold_metrics.csv", index=False)
        raise ValueError(f"No refinement passed the control comparison. See {output}.")
    selected_profiles = [
        profile
        for profile in PROFILES
        if any(name.startswith(f"refined-ridge-{profile.name}-") for name in candidates)
    ]
    print(f"Auditing {len(candidates)} finalists against all four prior models.", flush=True)
    audit_window = forecast(
        data, {**audit, "name": "final-holdout"}, selected_profiles, TARGETS, prior_targets
    )
    audit_scores, audit_ranking, _ = score_windows(
        data,
        [(audit_window[0], audit_window[1], audit_window[2][[*REFERENCE.values(), *candidates]])],
        scale,
        args,
        "audit",
    )
    audited = compare_priors(audit_ranking, selected_profiles, TARGETS, objective)
    audit_fields = ["candidate", objective, "mean_gain", "eligible"]
    report = comparison.merge(
        audited[audit_fields], on="candidate", how="left", suffixes=("", "_audit")
    )
    candidates = [
        name for name in candidates if name in set(audited.loc[audited["eligible"], "candidate"])
    ]
    report["selected"] = report["candidate"].isin(candidates)
    report.to_csv(output / "leaderboard.csv", index=False)
    pd.concat([scores, audit_scores]).to_csv(output / "fold_metrics.csv", index=False)
    if not candidates:
        raise ValueError(f"No selected candidate scored on the final holdout. See {output}.")
    similarities.loc[candidates, candidates].to_csv(output / "candidate_similarity.csv")
    fit_targets = data.labeled_fit(windows[0]["fit_expeds"], 1400)
    fit_targets[list(TARGETS)].corr().to_csv(output / "training_target_correlations.csv")
    print(f"Refitting {len(candidates)} candidates for deployment.", flush=True)
    fitted = fit_new(data, sorted(train[exped].unique()), selected_profiles, TARGETS)
    sample = audit_window[1].loc[
        audit_window[1][exped].isin(audit["holdout_expeds"][-3:]), [exped, *columns]
    ]
    if __name__ != "__main__":
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
    models = package({name: fitted[name] for name in candidates}, sample, exped, output)
    for model in models.values():
        model["prevalidated"] = True
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "experiment": "residual_ridge_round_one_target_refinement",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": {name: version(name) for name in LIBRARIES},
        "selection_metric": objective,
        "score_scale": scale,
        "benchmark_coverage": coverage,
        "core_features": core_features,
        "primary_model": candidates[0],
        "fallback_order": candidates,
        "models": models,
        "profiles": [asdict(profile) for profile in PROFILES],
        "targets": TARGETS,
        "multitarget_blend": TARGET_BLEND,
        "controls": REFERENCE,
        "embargo_expeds": args.embargo_expeds,
        "windows": [*windows, {**audit, "name": "final-holdout"}],
        "leaderboard": f"{output.name}/leaderboard.csv",
        "fold_metrics": f"{output.name}/fold_metrics.csv",
        "candidate_similarity": f"{output.name}/candidate_similarity.csv",
    }
    atomic_json(args.artifacts / "manifest.json", manifest, output)
    fields = [
        "candidate",
        "fit",
        "unq",
        objective,
        "mean_gain",
        "worst_gain",
        f"{objective}_audit",
        "mean_gain_audit",
    ]
    print(
        report.loc[
            report["selected"] | report["candidate"].isin(REFERENCE.values()), fields
        ].to_string(index=False),
        flush=True,
    )
    print(
        f"Ready: {len(candidates)} refined Ridge candidates in {args.artifacts / 'manifest.json'}.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prior", type=Path, default=Path("artifacts/wide-ridge"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/refined-ridge"))
    parser.add_argument("--folds", type=int, default=4, choices=range(4, 7))
    parser.add_argument("--fold-expeds", type=int, default=30)
    parser.add_argument("--audit-expeds", type=int, default=100)
    parser.add_argument("--embargo-expeds", type=int, default=80)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--max-similarity", type=float, default=0.95)
    args = parser.parse_args()
    args.zero_rule = False
    if args.finalists < 1 or not 0 < args.max_similarity < 1:
        parser.error("Use at least one finalist and a similarity limit between zero and one.")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Residual Ridge refinement failed. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
