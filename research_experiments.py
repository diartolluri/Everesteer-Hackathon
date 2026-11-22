"""Compare Everesteer research candidates on embargoed folds and package the finalists."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from hackathon_models import (
    BASE_CONFIGS,
    LIBRARIES,
    TRIAL_CONFIGS,
    average_rank_correlations,
    benchmark_surrogate_predictor,
    check_predictions,
    chronological_split,
    dataset_columns,
    event_proxy_scale,
    feature_stability,
    model_predictions,
    package,
    rank_blend,
    walk_forward_splits,
    weighted_rank_prediction,
)
from hackathon_scoring import (
    align_benchmark,
    benchmark_neutralize,
    core_reference,
    event_metrics,
    resolve_core_features,
    score_expeds,
)

ALPHAS = tuple(i / 10 for i in range(1, 13))
BENCHMARK_MODEL = "benchmark-surrogate"
PUBLISHED_BENCHMARK = "published-benchmark"
MAX_FINALIST_RANK_CORRELATION = 0.95


@dataclass(frozen=True)
class Candidate:
    name: str
    weights: tuple[tuple[str, float], ...]
    alpha: float = 0
    oracle: bool = False


def single(name):
    return Candidate(name, ((name, 1.0),))


def configurations(feature_count, core_features, auxiliary_target):
    tree = BASE_CONFIGS["lightgbm"]
    ridge = BASE_CONFIGS["ridge"]
    subset = min(48, feature_count)
    configs = {
        **BASE_CONFIGS,
        "lightgbm-regularized": TRIAL_CONFIGS["lightgbm-regularized"],
        "lightgbm-sentinel": {**tree, "missing_policy": "sentinel"},
        "lightgbm-indicator": {**tree, "missing_policy": "indicator"},
        "ridge-median-indicator": {**ridge, "missing_policy": "median-indicator"},
        "ridge-fit-features": {**ridge, "feature_selection": "fit", "max_features": subset},
        "lightgbm-residual-features": {
            **tree,
            "feature_selection": "residual",
            "max_features": subset,
        },
        "lightgbm-residual-target": {
            **TRIAL_CONFIGS["lightgbm-regularized"],
            "feature_selection": "residual",
            "max_features": subset,
            "neutralization": 0.5,
        },
        "ridge-residual-target": {
            "family": "ridge",
            "relative_alpha": 0.01,
            "neutralization": 0.5,
        },
        "ridge-pca": {**ridge, "pca_components": min(24, feature_count)},
    }
    exposure = "ridge-low-core-exposure" if core_features else "ridge-low-common-exposure"
    configs[exposure] = {**ridge, "feature_selection": "low_exposure", "max_features": subset}
    if auxiliary_target:
        configs["ridge-auxiliary"] = {**ridge, "auxiliary_target": auxiliary_target}
    return configs


def fit_components(
    fit,
    configs,
    columns,
    missing,
    target,
    exped,
    embargo,
    benchmark,
    exposure_reference,
    *,
    fit_surrogate=True,
):
    ordinary = model_predictions(
        fit,
        columns,
        missing,
        target,
        configs=configs,
        exped=exped,
        embargo=embargo,
        benchmark=benchmark,
        core=exposure_reference,
    )
    if not fit_surrogate:
        return ordinary
    # The surrogate learns only from the training benchmark and accepts features at inference.
    reference = benchmark.reindex(fit.index).dropna()
    if len(reference) < 3:
        raise ValueError("The fitting window has fewer than three aligned benchmark predictions.")
    proxy_fit = fit.loc[reference.index, [exped, *columns]].assign(_reference_target=reference)
    proxy = model_predictions(
        proxy_fit,
        columns,
        missing,
        "_reference_target",
        configs={BENCHMARK_MODEL: {**BASE_CONFIGS["lightgbm"], "n_estimators": 160}},
        exped=exped,
    )
    return {**ordinary, **proxy}


def candidate_predictor(candidate, predictors, exped):
    if candidate.oracle:
        raise ValueError("A candidate that requires the held-out benchmark cannot be deployed.")
    predict = weighted_rank_prediction(predictors, exped, dict(candidate.weights))
    if candidate.alpha:
        predict = benchmark_surrogate_predictor(
            predict, predictors[BENCHMARK_MODEL], exped, candidate.alpha
        )
    return predict


def predict_candidates(candidates, forecasts, expeds, benchmark):
    predictions = {}
    for candidate in candidates:
        if candidate.name == PUBLISHED_BENCHMARK:
            predictions[candidate.name] = benchmark
            continue
        raw = rank_blend(forecasts, expeds, dict(candidate.weights))
        if candidate.alpha:
            reference = benchmark if candidate.oracle else forecasts[BENCHMARK_MODEL]
            raw = benchmark_neutralize(raw, reference, expeds, candidate.alpha)
        predictions[candidate.name] = raw
    return predictions


@dataclass(frozen=True)
class ResearchData:
    train: pd.DataFrame
    columns: list[str]
    target: str
    exped: str
    missing: int | None
    benchmark: pd.Series
    core_features: list[str] | None
    exposure_reference: pd.Series

    def labeled_fit(self, expeds, maximum):
        # Use the most recent contiguous expeds when a fit is capped.
        ordered = expeds if maximum is None else expeds[-maximum:]
        labels = self.train[self.target]
        return self.train.loc[self.train[self.exped].isin(ordered) & np.isfinite(labels)]


def fit_window(data, configs, split, embargo, maximum, *, fit_surrogate):
    fit = data.labeled_fit(split["fit_expeds"], maximum)
    predictors = fit_components(
        fit,
        configs,
        data.columns,
        data.missing,
        data.target,
        data.exped,
        embargo,
        data.benchmark,
        data.exposure_reference,
        fit_surrogate=fit_surrogate,
    )
    heldout = data.train.loc[data.train[data.exped].isin(split["holdout_expeds"])]
    features = heldout[[data.exped, *data.columns]]
    forecasts = pd.DataFrame(
        {
            name: check_predictions(predict, features)["prediction"]
            for name, predict in predictors.items()
        }
    )
    return split["name"], heldout, forecasts


def evaluate_candidates(data, candidates, windows, scale, args, stage):
    rows, all_expeds = [], []
    for fold, heldout, forecasts in windows:
        reference = data.benchmark.reindex(heldout.index)
        predictions = predict_candidates(candidates, forecasts, heldout[data.exped], reference)
        per_exped = score_expeds(
            heldout,
            data.exped,
            data.target,
            predictions,
            reference,
            core_features=data.core_features,
            missing=data.missing,
            zero_rule=args.zero_rule,
        )
        all_expeds.append(per_exped.assign(stage=stage, fold=fold))
        for name, metrics in per_exped.groupby("model", sort=False):
            rows.append(
                {
                    "stage": stage,
                    "fold": fold,
                    "candidate": name,
                    "expeds": metrics["fit"].count(),
                    "fit": metrics["fit"].mean(),
                    "unq": metrics["unq_covariance"].mean(),
                    "inov": metrics["inov"].mean(),
                    "benchmark_exped_share": metrics["unq_covariance"].notna().mean(),
                    "core_exped_share": metrics["inov"].notna().mean(),
                    "benchmark_corr": metrics["benchmark_rank_corr"].mean(),
                    **event_metrics(metrics, scale),
                }
            )
    return pd.DataFrame(rows), pd.concat(all_expeds, ignore_index=True)


def rank_candidates(scores, candidates, expected_folds, objective):
    oracle = {candidate.name for candidate in candidates if candidate.oracle}
    summary = []
    metrics = (
        "fit",
        "unq",
        "inov",
        "event_signal",
        "event_score",
        "relative_signal_proxy",
        "round_score_proxy",
        "benchmark_corr",
    )
    for name, rows in scores.groupby("candidate", sort=False):
        coverage = rows["benchmark_exped_share"].ge(0.5)
        if objective == "event_score":
            coverage &= rows["core_exped_share"].ge(0.5)
        valid = rows[objective].notna() & rows["expeds"].gt(0) & coverage
        record = {}
        for metric in metrics:
            available = rows[metric].notna() & rows["expeds"].gt(0)
            record[metric] = (
                np.average(rows.loc[available, metric], weights=rows.loc[available, "expeds"])
                if available.any()
                else np.nan
            )
        summary.append(
            {
                "candidate": name,
                "oracle": name in oracle,
                "folds": int(valid.sum()),
                "fold_sd": rows[objective].std(ddof=0),
                "worst_fold": rows[objective].min(),
                "eligible": name not in oracle and valid.sum() == expected_folds,
                **record,
            }
        )
    return pd.DataFrame(summary).sort_values(
        [objective, "worst_fold", "fold_sd", "candidate"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )


def blend_grid(tree, ridge, third):
    blends = [
        Candidate(f"rank-blend-{int(weight * 100)}", ((tree, weight), (ridge, 1 - weight)))
        for weight in (0.25, 0.5, 0.75)
    ]
    if third:
        for weight in (0.2, 0.4, 0.6):
            pair = (1 - weight) / 2
            blends.append(
                Candidate(
                    f"rank-blend-residual-{int(weight * 100)}",
                    ((tree, pair), (ridge, pair), (third, weight)),
                )
            )
    return blends


def neutralization_grid(candidate):
    return [
        Candidate(
            f"{candidate.name}-{kind}-{int(round(alpha * 100))}", candidate.weights, alpha, oracle
        )
        for kind, oracle in (("surrogate", False), ("oracle", True))
        for alpha in ALPHAS
    ]


def choose_finalists(leaderboard, candidates, correlations, limit):
    lookup = {candidate.name: candidate for candidate in candidates}
    selected, constituent_sets = {}, set()
    for name in leaderboard.loc[leaderboard["eligible"], "candidate"]:
        candidate = lookup[name]
        members = frozenset(component for component, _ in candidate.weights)
        if members in constituent_sets:
            continue
        if any(
            abs(correlations[name].get(other) or 0.0) > MAX_FINALIST_RANK_CORRELATION
            for other in selected
        ):
            continue
        selected[name] = candidate
        constituent_sets.add(members)
        if len(selected) == limit:
            break
    return selected


def candidate_correlations(candidates, windows, exped, benchmark):
    blocks = [
        pd.DataFrame(
            predict_candidates(
                candidates, forecasts, heldout[exped], benchmark.reindex(heldout.index)
            )
        )
        for _, heldout, forecasts in windows
    ]
    return average_rank_correlations(
        pd.concat(blocks), pd.concat([heldout[exped] for _, heldout, _ in windows])
    )


def atomic_json(path, payload, staging):
    temporary = staging / path.name
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_experiments(data, configs, scale, args):
    objective = "event_score" if data.core_features else "round_score_proxy"
    print(f"Selection objective {objective}.", flush=True)
    split = chronological_split(data.train[data.exped], args.holdout_expeds, args.embargo_expeds)
    folds = walk_forward_splits(split, args.cv_folds, args.fold_expeds, args.embargo_expeds)
    all_scores, all_expeds = [], []

    def evaluate(candidates, windows, stage):
        scores, per_exped = evaluate_candidates(data, candidates, windows, scale, args, stage)
        all_scores.append(scores)
        all_expeds.append(per_exped)
        return scores

    def windows_for(
        config_set,
        maximum,
        stage,
        *,
        split_windows=folds,
        embargo=args.embargo_expeds,
        fit_surrogate=True,
    ):
        windows = []
        for fold in split_windows:
            print(f"{stage} {fold['name']}, {len(config_set)} candidates.", flush=True)
            windows.append(
                fit_window(data, config_set, fold, embargo, maximum, fit_surrogate=fit_surrogate)
            )
        return windows

    singles = [single(name) for name in configs]
    benchmark_entry = Candidate(PUBLISHED_BENCHMARK, (), oracle=True)
    scouts = windows_for(configs, args.scout_fit_expeds, "Scout", fit_surrogate=False)
    scout_scores = evaluate([*singles, benchmark_entry], scouts, "scout")
    scout_ranking = rank_candidates(
        scout_scores, [*singles, benchmark_entry], len(folds), objective
    )
    ordered = scout_ranking.loc[scout_ranking["eligible"], "candidate"].tolist()
    if not ordered:
        raise ValueError(
            "No candidate scored on every scout fold. Check benchmark and core coverage."
        )
    tree = next((name for name in ordered if configs[name]["family"] == "lightgbm"), None)
    ridge = next((name for name in ordered if configs[name]["family"] == "ridge"), None)
    if tree is None or ridge is None:
        raise ValueError(
            "The blend requires both a LightGBM and a Ridge candidate scored on every fold."
        )
    complementary = (name for name in ordered if name not in {tree, ridge})
    third = next(
        (name for name in complementary if "residual" in name or name == "ridge-auxiliary"), None
    )
    blends = blend_grid(tree, ridge, third)
    blend_scores = evaluate(blends, scouts, "scout")
    blend_ranking = rank_candidates(blend_scores, blends, len(folds), objective)
    chosen_blend = next(
        (
            candidate
            for name in blend_ranking.loc[blend_ranking["eligible"], "candidate"]
            for candidate in blends
            if candidate.name == name
        ),
        None,
    )
    if chosen_blend is None:
        raise ValueError("No blend scored on every scout fold.")

    # Keep the current LightGBM and Ridge as reference points for every scaled score.
    names = dict.fromkeys([*BASE_CONFIGS, *ordered[:3], tree, ridge, third])
    scale_configs = {name: configs[name] for name in names if name is not None}
    scaled = windows_for(scale_configs, args.scale_fit_expeds, "Scale")
    baseline_blend = Candidate("rank-blend-baseline", (("lightgbm", 0.5), ("ridge", 0.5)))
    finals = [
        *(single(name) for name in scale_configs),
        baseline_blend,
        *blends,
        benchmark_entry,
        *(
            variant
            for name in dict.fromkeys(("lightgbm", "ridge", tree))
            for variant in neutralization_grid(single(name))
        ),
        *neutralization_grid(chosen_blend),
    ]
    scale_scores = evaluate(finals, scaled, "scale")
    leaderboard = rank_candidates(scale_scores, finals, len(folds), objective)
    eligible_names = set(leaderboard.loc[leaderboard["eligible"], "candidate"])
    eligible = [candidate for candidate in finals if candidate.name in eligible_names]
    correlations = (
        candidate_correlations(eligible, scaled, data.exped, data.benchmark) if eligible else {}
    )
    selected = choose_finalists(leaderboard, eligible, correlations, args.finalists)
    if not selected:
        raise ValueError("No deployable candidate scored on every scale fold.")
    print(leaderboard.loc[leaderboard["eligible"]].head(10).to_string(index=False), flush=True)

    audit_candidates = {
        item.name: item
        for item in (
            *(single(name) for name in BASE_CONFIGS),
            baseline_blend,
            benchmark_entry,
            *selected.values(),
        )
    }
    needed = {name for candidate in audit_candidates.values() for name, _ in candidate.weights}
    selected_configs = {name: configs[name] for name in configs if name in needed}
    print("Auditing the selected candidates on the final holdout.", flush=True)
    audit = fit_window(
        data,
        selected_configs,
        {**split, "name": "final-holdout"},
        args.embargo_expeds,
        args.scale_fit_expeds,
        fit_surrogate=any(candidate.alpha for candidate in selected.values()),
    )
    audit_scores = evaluate(list(audit_candidates.values()), [audit], "audit")
    audit_ranking = rank_candidates(audit_scores, list(audit_candidates.values()), 1, objective)
    audited = set(audit_ranking.loc[audit_ranking["eligible"], "candidate"])
    selected = {name: candidate for name, candidate in selected.items() if name in audited}
    if not selected:
        raise ValueError("No finalist could be scored on the independent holdout.")

    sensitivity = []
    if args.sensitivity_embargo_expeds:
        extra = args.sensitivity_embargo_expeds - args.embargo_expeds
        for fold in folds:
            fit = fold["fit_expeds"]
            if len(fit) - extra < 2:
                raise ValueError("The larger embargo leaves fewer than two training expeds.")
            sensitivity.append(
                {
                    **fold,
                    "fit_expeds": fit[:-extra],
                    "embargo_expeds": fit[-extra:] + fold["embargo_expeds"],
                }
            )
        shifted = windows_for(
            selected_configs,
            args.scale_fit_expeds,
            "Sensitivity",
            split_windows=sensitivity,
            embargo=args.sensitivity_embargo_expeds,
            fit_surrogate=any(candidate.alpha for candidate in audit_candidates.values()),
        )
        evaluate(list(audit_candidates.values()), shifted, "sensitivity")

    needed = {name for candidate in selected.values() for name, _ in candidate.weights}
    fitting = data.labeled_fit(sorted(data.train[data.exped].unique()), args.refit_expeds)
    print(
        f"Training {len(needed)} finalist components on {len(fitting):,} labeled rows.", flush=True
    )
    components = fit_components(
        fitting,
        {name: configs[name] for name in configs if name in needed},
        data.columns,
        data.missing,
        data.target,
        data.exped,
        args.embargo_expeds,
        data.benchmark,
        data.exposure_reference,
        fit_surrogate=any(candidate.alpha for candidate in selected.values()),
    )
    predictors = {
        name: candidate_predictor(candidate, components, data.exped)
        for name, candidate in selected.items()
    }
    sample = audit[1].loc[
        audit[1][data.exped].isin(split["holdout_expeds"][-3:]), [data.exped, *data.columns]
    ]
    args.artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="research-", dir=args.artifacts))
    models = package(predictors, sample, data.exped, output)
    for model in models.values():
        model["prevalidated"] = True
    scout_summary = rank_candidates(
        pd.concat([scout_scores, blend_scores]),
        [*singles, *blends, benchmark_entry],
        len(folds),
        objective,
    )
    summaries = [
        scout_summary.assign(stage="scout"),
        leaderboard.assign(stage="scale"),
        audit_ranking.assign(stage="audit"),
    ]
    if sensitivity:
        summaries.append(
            rank_candidates(
                all_scores[-1], list(audit_candidates.values()), len(folds), objective
            ).assign(stage="sensitivity")
        )
    full_summary = pd.concat(summaries)
    full_summary.to_csv(output / "leaderboard.csv", index=False)
    (output / "leaderboard.json").write_text(
        full_summary.to_json(orient="records", indent=2) + "\n"
    )
    pd.concat(all_scores).to_csv(output / "fold_metrics.csv", index=False)
    pd.concat(all_expeds).to_csv(output / "per_exped.csv", index=False)
    stability = feature_stability(
        fitting,
        data.columns,
        data.missing,
        data.target,
        data.exped,
        data.benchmark,
        data.exposure_reference,
    )
    if not data.core_features:
        stability = stability.rename(
            columns={"core_corr": "common_corr", "core_abs_corr": "common_abs_corr"}
        )
    stability.to_csv(output / "feature_stability.csv")
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "provenance": "walk_forward_research",
        "primary_target": data.target,
        "exped_column": data.exped,
        "feature_columns": data.columns,
        "library_versions": {library: version(library) for library in LIBRARIES},
        "selection_metric": objective,
        "score_scale": scale,
        "zero_rule": args.zero_rule,
        "core_features": data.core_features,
        "primary_model": next(iter(selected)),
        "fallback_order": list(selected),
        "models": models,
        "model_rank_correlations": {
            name: {other: correlations[name][other] for other in selected} for name in selected
        },
        "candidates": {name: asdict(candidate) for name, candidate in selected.items()},
        "component_configs": {name: configs[name] for name in configs if name in needed},
        "scout_fit_expeds": args.scout_fit_expeds,
        "scale_fit_expeds": args.scale_fit_expeds,
        "refit_expeds": int(fitting[data.exped].nunique()),
        "embargo_expeds": args.embargo_expeds,
        "sensitivity_embargo_expeds": args.sensitivity_embargo_expeds,
        "sensitivity_windows": sensitivity,
        "windows": [*folds, {**split, "name": "final-holdout"}],
        "leaderboard": f"{output.name}/leaderboard.csv",
        "leaderboard_json": f"{output.name}/leaderboard.json",
        "fold_metrics": f"{output.name}/fold_metrics.csv",
        "per_exped_metrics": f"{output.name}/per_exped.csv",
        "feature_stability": f"{output.name}/feature_stability.csv",
    }
    atomic_json(args.artifacts / "manifest.json", manifest, output)
    print(
        f"Primary candidate {manifest['primary_model']}. Artifacts at {args.artifacts}.", flush=True
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--cv-folds", type=int, default=4, choices=range(4, 7))
    parser.add_argument("--fold-expeds", type=int, default=40)
    parser.add_argument("--holdout-expeds", type=int, default=100)
    parser.add_argument("--embargo-expeds", type=int, default=40)
    parser.add_argument(
        "--sensitivity-embargo-expeds",
        type=int,
        help="Reevaluate the finalists on the same folds with a longer embargo.",
    )
    parser.add_argument("--scout-fit-expeds", type=int, default=300)
    parser.add_argument("--scale-fit-expeds", type=int, default=1400)
    parser.add_argument(
        "--refit-expeds", type=int, help="Cap the deployment fit to the most recent expeds."
    )
    parser.add_argument("--finalists", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--feature-set")
    parser.add_argument("--benchmark-column")
    parser.add_argument(
        "--core-feature", action="append", help="Verified core member. Repeat for each member."
    )
    parser.add_argument(
        "--auxiliary-target", help="Train an additional Ridge on this schema-listed target."
    )
    parser.add_argument(
        "--zero-rule", action="store_true", help="Test the alternate zero-rule scoring description."
    )
    args = parser.parse_args()
    if not 2 <= args.scout_fit_expeds <= args.scale_fit_expeds or (
        args.refit_expeds is not None and args.refit_expeds < 2
    ):
        parser.error(
            "Fit caps must be at least two expeds, and the scale cap must cover the scout cap."
        )
    if (
        args.sensitivity_embargo_expeds is not None
        and args.sensitivity_embargo_expeds <= args.embargo_expeds
    ):
        parser.error("The sensitivity embargo must exceed the selection embargo.")
    try:
        return run_cli(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Research failed. {exc}", file=sys.stderr)
        return 1


def run_cli(args):
    schema = json.loads((args.data_dir / "schema.json").read_text())
    scoring = json.loads((args.data_dir / "scoring.json").read_text())
    scale, error = event_proxy_scale(scoring)
    if error or scale is None:
        raise ValueError(
            "The saved scoring response must declare compatible event weights and scale."
        )
    train = pd.read_parquet(args.data_dir / "train.parquet")
    if (
        train.index.nlevels != 1
        or not train.index.is_unique
        or train.index.hasnans
        or not all(isinstance(value, str) and value for value in train.index)
    ):
        raise ValueError("Training Parquet needs a unique index of nonempty string IDs.")
    target, exped, columns, missing = dataset_columns(train, schema, args.feature_set)
    if args.auxiliary_target and (
        args.auxiliary_target == target
        or args.auxiliary_target not in (schema.get("targets") or [])
        or args.auxiliary_target not in train
    ):
        raise ValueError(
            "The auxiliary target must be an available, schema-listed non-primary target."
        )
    core_features = resolve_core_features(schema, args.core_feature)
    benchmark, coverage = align_benchmark(
        pd.read_parquet(args.data_dir / "train_benchmark.parquet"),
        train[[exped]],
        exped,
        score_column=args.benchmark_column,
    )
    if benchmark is None:
        raise ValueError("The official benchmark does not cover enough training IDs and expeds.")
    if set(core_features) - set(train.columns):
        raise ValueError("The labeled training file is missing verified core feature columns.")
    exposure_reference = core_reference(train[core_features or columns], missing=missing)
    print(f"{len(train):,} train rows, benchmark coverage {coverage:.1%}.", flush=True)
    data = ResearchData(
        train, columns, target, exped, missing, benchmark, core_features, exposure_reference
    )
    run_experiments(
        data, configurations(len(columns), core_features, args.auxiliary_target), scale, args
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
