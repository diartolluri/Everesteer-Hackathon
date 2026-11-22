"""Tune fixed blend weights on early forecasts and validate them on later folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pandas as pd

from hackathon_models import (
    BASE_CONFIGS,
    LIBRARIES,
    check_predictions,
    chronological_split,
    dataset_columns,
    event_proxy_scale,
    model_predictions,
    package,
    walk_forward_splits,
    weighted_rank_prediction,
)
from hackathon_scoring import align_benchmark, core_reference, resolve_core_features
from research_categorical_sweep import PROFILES as CATEGORICAL_PROFILES
from research_categorical_sweep import fit_profile as fit_categorical
from research_experiments import (
    Candidate,
    ResearchData,
    atomic_json,
    evaluate_candidates,
    predict_candidates,
    rank_candidates,
    single,
)
from research_ridge_sweep import PROFILES as RIDGE_PROFILES
from research_ridge_sweep import cross_sectional_similarity
from research_ridge_sweep import fit_profile as fit_ridge
from research_tree_sweep import Trial, build_fitter

PRACTICE_TINZAR = "ridge-residual-tinzar"
PRACTICE_TASSERIRT = "ridge-short-tasserirt"
RIDGE_MEMBERS = {
    "short": {PRACTICE_TASSERIRT: "wide-ridge-short-tasserirt"},
    "residual": {
        PRACTICE_TINZAR: "wide-ridge-residual-tinzar",
        "ridge-residual-tignousti": "wide-ridge-residual-tignousti",
    },
}
CAT_MEMBERS = {
    "cat-tignousti": "categorical-residual-interactions-tignousti",
    "cat-timgarine": "categorical-residual-interactions-timgarine",
    "cat-tinzar": "categorical-residual-interactions-tinzar",
    "cat-tasserirt": "categorical-residual-interactions-tasserirt",
}
TREE_MODELS = {
    "tree-compact": Trial(360, "residual24", 0.95, "compact"),
    "tree-narrow": Trial(360, "residual12", 1.0, "compact"),
}
AUX_BASE = "ridge-direct-tinzar"


def fit_components(data, expeds, ridge_targets, cat_targets, tree_fitter):
    exact = data.labeled_fit(expeds, 1400)
    predictors = model_predictions(
        exact,
        data.columns,
        data.missing,
        data.target,
        configs={AUX_BASE: {**BASE_CONFIGS["ridge"], "auxiliary_target": "target_Tinzar"}},
        exped=data.exped,
    )
    for profile in RIDGE_PROFILES:
        members = RIDGE_MEMBERS.get(profile.name)
        if members is not None:
            fitted = fit_ridge(data, profile, expeds, ridge_targets)
            predictors.update({name: fitted[alias] for name, alias in members.items()})
    profile = next(p for p in CATEGORICAL_PROFILES if p.name == "residual-interactions")
    categorical = fit_categorical(data, profile, expeds, cat_targets)
    predictors.update({name: categorical[alias] for name, alias in CAT_MEMBERS.items()})
    predictors.update(
        {name: tree_fitter(trial, tuple(expeds)) for name, trial in TREE_MODELS.items()}
    )
    return predictors


def candidate(members, weights):
    recipe = tuple(sorted(zip(members, weights, strict=True)))
    digest = hashlib.sha256(json.dumps(recipe).encode()).hexdigest()[:10]
    return Candidate(f"crossfit-{digest}", recipe)


def recipes(base_names, *, broad=False):
    anchors = (PRACTICE_TINZAR, PRACTICE_TASSERIRT, AUX_BASE)
    candidates = {}
    for anchor in anchors:
        for member in base_names:
            if member == anchor:
                continue
            weights = (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9) if broad else (0.25, 0.5, 0.75)
            for weight in weights:
                blend = candidate((anchor, member), (weight, 1 - weight))
                candidates[blend.name] = blend
    for member in base_names:
        if member in {PRACTICE_TINZAR, PRACTICE_TASSERIRT}:
            continue
        for weights in ((0.2, 0.2, 0.6), (0.35, 0.35, 0.3), (0.55, 0.2, 0.25), (0.2, 0.55, 0.25)):
            blend = candidate((PRACTICE_TINZAR, PRACTICE_TASSERIRT, member), weights)
            candidates[blend.name] = blend
    return list(candidates.values())


def constituent_deltas(ranking, blends, objective):
    indexed = ranking.set_index("candidate")
    result = []
    for blend in blends:
        row = indexed.loc[blend.name].to_dict()
        members = indexed.loc[[name for name, _ in blend.weights]]
        row.update(
            candidate=blend.name,
            constituents=json.dumps(dict(blend.weights)),
            mean_delta=row[objective] - members[objective].max(),
            worst_delta=row["worst_fold"] - members["worst_fold"].max(),
        )
        result.append(row)
    return pd.DataFrame(result)


def choose(data, scores, blends, windows, objective, args, count, *, screening=False):
    results = constituent_deltas(scores, blends, objective)
    if screening:
        within_margin = results["mean_delta"].ge(-0.05) & results["worst_delta"].ge(-0.05)
        eligible = results.loc[results["eligible"] & within_margin].copy()
        eligible["relative_quality"] = eligible["mean_delta"] + 0.5 * eligible["worst_delta"]
        sort_columns = ["relative_quality", objective, "candidate"]
    else:
        beats_constituents = (
            results["mean_delta"].gt(args.min_gain) & results["worst_delta"].ge(-0.025)
        ) | (results["worst_delta"].gt(args.min_gain) & results["mean_delta"].ge(-0.025))
        eligible = results.loc[results["eligible"] & beats_constituents]
        sort_columns = [objective, "worst_fold", "candidate"]
    eligible = eligible.sort_values(sort_columns, ascending=[False, False, True])
    lookup = {blend.name: blend for blend in blends}
    finalists = [lookup[name] for name in eligible["candidate"]]
    if not finalists:
        return [], results
    composed = []
    for fold, frame, values in windows:
        expeds = frame[data.exped]
        preds = predict_candidates(finalists, values, expeds, data.benchmark.reindex(frame.index))
        composed.append((fold, frame, pd.DataFrame(preds, index=frame.index)))
    similarity = cross_sectional_similarity(composed, data.exped)
    selected = []
    for recipe in finalists:
        if (
            selected
            and not similarity.loc[recipe.name, [item.name for item in selected]]
            .lt(args.max_similarity)
            .all()
        ):
            continue
        selected.append(recipe)
        if len(selected) >= count:
            break
    return selected, results


def run(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    scoring = json.loads((args.data_dir / "scoring.json").read_text())
    scale, error = event_proxy_scale(scoring)
    if error or scale is None:
        raise ValueError("The saved score scale is unavailable.")
    train = pd.read_parquet(args.data_dir / "train.parquet")
    target, exped, columns, missing = dataset_columns(train, schema)
    if not train.index.is_unique or train.index.hasnans or train.index.nlevels != 1:
        raise ValueError("The labeled data needs a unique, nonmissing ID index.")
    core_features = resolve_core_features(schema)
    benchmark, coverage = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"), train[[exped]], exped
    )
    if benchmark is None:
        raise ValueError("The local training benchmark does not align to the training dataset.")
    exposure = core_reference(train[core_features], missing=missing) if core_features else None
    data = ResearchData(train, columns, target, exped, missing, benchmark, core_features, exposure)
    objective = "event_score" if core_features else "round_score_proxy"
    ridge = json.loads((args.sources / "wide-ridge" / "manifest.json").read_text())
    categorical = json.loads((args.sources / "wide-alternative" / "manifest.json").read_text())
    ridge_targets, cat_targets = ridge["targets"], categorical["targets"]
    tree_fitter = build_fitter(
        train,
        columns,
        missing,
        target,
        exped,
        benchmark,
        SimpleNamespace(threads=args.threads, estimators=args.estimators),
    )
    audit_window = chronological_split(train[exped], args.audit_expeds, args.embargo_expeds)
    development = audit_window["fit_expeds"]
    if args.history_offset:
        development = development[: -args.history_offset]
    validation_folds = walk_forward_splits(
        {"fit_expeds": development}, args.validation_folds, args.fold_expeds, args.embargo_expeds
    )
    # End tuning before the embargo of even the earliest independent validation fold.
    tuning_folds = walk_forward_splits(
        {"fit_expeds": validation_folds[0]["fit_expeds"]},
        args.tuning_folds,
        args.fold_expeds,
        args.embargo_expeds,
    )
    tuning_folds = [{**fold, "name": f"tune-{i}"} for i, fold in enumerate(tuning_folds, 1)]
    validation_folds = [
        {**fold, "name": f"validate-{i}"} for i, fold in enumerate(validation_folds, 1)
    ]

    def window(split):
        print(f"Refitting constituent models for {split['name']}.", flush=True)
        members = fit_components(data, split["fit_expeds"], ridge_targets, cat_targets, tree_fitter)
        frame = train.loc[train[exped].isin(split["holdout_expeds"])]
        features = frame[[exped, *columns]]
        values = {
            name: check_predictions(predict, features)["prediction"]
            for name, predict in members.items()
        }
        return split["name"], frame, pd.DataFrame(values, index=frame.index)

    all_scores = []

    def evaluate(blends, forecasts, stage):
        contestants = [*(single(name) for name in forecasts[0][2]), *blends]
        scores, _ = evaluate_candidates(data, contestants, forecasts, scale, args, stage)
        all_scores.append(scores)
        return rank_candidates(scores, contestants, len(forecasts), objective)

    tune_forecasts = [window(fold) for fold in tuning_folds]
    blends = recipes(tune_forecasts[0][2].columns, broad=bool(args.history_offset))
    print(f"Tuning {len(blends)} fixed positive-weight recipes.", flush=True)
    tune_result = evaluate(blends, tune_forecasts, "tuning")
    frozen, tune_composition = choose(
        data,
        tune_result,
        blends,
        tune_forecasts,
        objective,
        args,
        args.screen,
        screening=bool(args.history_offset),
    )
    if not frozen:
        raise ValueError(
            "No tuning recipe improved mean or worst-window score over every constituent."
        )
    print(f"Frozen {len(frozen)} recipes for separate validation.", flush=True)
    val_forecasts = [window(fold) for fold in validation_folds]
    val_result = evaluate(frozen, val_forecasts, "validation")
    finalists, val_composition = choose(
        data, val_result, frozen, val_forecasts, objective, args, args.finalists
    )
    args.artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="crossfit-", dir=args.artifacts))
    tune_composition.to_csv(output / "tuning_recipes.csv", index=False)
    val_composition.to_csv(output / "validation_recipes.csv", index=False)
    if not finalists:
        pd.concat(all_scores).to_csv(output / "fold_metrics.csv", index=False)
        raise ValueError(
            f"No frozen recipe beat all of its constituents on independent folds. See {output}."
        )
    print(f"Auditing {len(finalists)} validation winners on the untouched tail.", flush=True)
    audit = window({**audit_window, "name": "final-holdout"})
    audit_result = evaluate(finalists, [audit], "audit")
    audit_composition = constituent_deltas(audit_result, finalists, objective)
    audit_composition.to_csv(output / "audit_recipes.csv", index=False)
    pd.concat(all_scores).to_csv(output / "fold_metrics.csv", index=False)
    finalists = [
        candidate
        for candidate in finalists
        if candidate.name in set(audit_composition.loc[audit_composition["eligible"], "candidate"])
    ]
    if not finalists:
        raise ValueError(f"The selected recipes could not be scored on the audit. See {output}.")
    print(f"Fitting deployment constituents and packaging {len(finalists)} blends.", flush=True)
    fitted = fit_components(
        data, sorted(train[exped].unique()), ridge_targets, cat_targets, tree_fitter
    )
    predictors = {
        candidate.name: weighted_rank_prediction(fitted, exped, dict(candidate.weights))
        for candidate in finalists
    }
    sample = audit[1].loc[
        audit[1][exped].isin(audit_window["holdout_expeds"][-3:]), [exped, *columns]
    ]
    cloudpickle.register_pickle_by_value(sys.modules[fit_categorical.__module__])
    models = package(predictors, sample, exped, output)
    for model in models.values():
        model["prevalidated"] = True
    chosen = list(models)
    results = val_composition.loc[val_composition["candidate"].isin(chosen)].merge(
        audit_composition[["candidate", objective, "mean_delta"]],
        on="candidate",
        suffixes=("", "_audit"),
    )
    results.to_csv(output / "selected.csv", index=False)
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "experiment": "chronological_crossfit_rank_blend",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": {name: version(name) for name in LIBRARIES},
        "selection_metric": objective,
        "score_scale": scale,
        "benchmark_coverage": coverage,
        "core_features": core_features,
        "primary_model": chosen[0],
        "fallback_order": chosen,
        "models": models,
        "candidates": {recipe.name: asdict(recipe) for recipe in finalists},
        "constituent_ridge_profiles": RIDGE_MEMBERS,
        "constituent_categorical_profiles": CAT_MEMBERS,
        "constituent_tree_profiles": {name: asdict(trial) for name, trial in TREE_MODELS.items()},
        "constituent_tree_estimators": args.estimators,
        "tuning_windows": tuning_folds,
        "validation_windows": validation_folds,
        "audit_window": audit_window,
        "embargo_expeds": args.embargo_expeds,
        "history_offset": args.history_offset,
        "fold_metrics": f"{output.name}/fold_metrics.csv",
        "results": f"{output.name}/selected.csv",
    }
    atomic_json(args.artifacts / "manifest.json", manifest, output)
    print(
        results[
            [
                "candidate",
                "constituents",
                objective,
                "mean_delta",
                "worst_delta",
                f"{objective}_audit",
                "mean_delta_audit",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(
        f"Ready: {len(models)} cross-fit blends in {args.artifacts / 'manifest.json'}.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/crossfit-blends"))
    parser.add_argument("--tuning-folds", type=int, default=3)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--history-offset", type=int, default=0)
    parser.add_argument("--fold-expeds", type=int, default=30)
    parser.add_argument("--audit-expeds", type=int, default=100)
    parser.add_argument("--embargo-expeds", type=int, default=80)
    parser.add_argument("--estimators", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--screen", type=int, default=30)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--min-gain", type=float, default=0.001)
    parser.add_argument("--max-similarity", type=float, default=0.992)
    args = parser.parse_args()
    args.zero_rule = False
    if (
        args.history_offset < 0
        or args.screen < args.finalists
        or args.finalists < 1
        or not 0 < args.max_similarity < 1
    ):
        parser.error(
            "Choose a positive finalist count, a larger screen and a similarity limit between zero and one."
        )
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Cross-fit blend study failed. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
