"""Cross-sectional scoring and benchmark transforms for offline event research."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri


def align_benchmark(benchmark, holdout, exped, *, minimum_coverage=0.5, score_column=None):
    """Reindex a published benchmark by opaque id and exped together."""
    if benchmark is None:
        return None, None
    if exped not in benchmark:
        print("Benchmark metrics unavailable because the benchmark has no exped column.")
        return None, None

    columns = benchmark.select_dtypes(include="number").columns.difference([exped, "id"])
    if score_column is None:
        if columns.empty:
            print("Benchmark metrics unavailable because the benchmark has no numeric scores.")
            return None, None
        if len(columns) > 1:
            print("Select the event benchmark score column explicitly when several are available.")
            return None, None
        score_column = columns[0]
    elif score_column not in columns:
        raise ValueError(f"No numeric benchmark score column named {score_column!r}.")

    scores = benchmark[score_column].replace([np.inf, -np.inf], np.nan)
    ids = benchmark["id"] if "id" in benchmark and benchmark.index.name != "id" else benchmark.index
    scores.index = pd.MultiIndex.from_arrays([ids, benchmark[exped]])
    if not scores.index.is_unique:
        print(
            "Benchmark metrics unavailable because the benchmark has duplicate id and exped pairs."
        )
        return None, None

    keys = pd.MultiIndex.from_arrays([holdout.index, holdout[exped]])
    aligned = pd.Series(scores.reindex(keys).to_numpy(), index=holdout.index, name=score_column)
    matches = int(aligned.notna().sum())
    coverage = matches / len(aligned) if len(aligned) else 0.0
    print(f"Benchmark matches {matches:,}/{len(aligned):,} rows ({coverage:.2%}) by id and exped.")
    if not matches or coverage < minimum_coverage:
        return None, coverage
    return aligned, coverage


def resolve_core_features(schema, requested=None):
    """Use an explicit core list. An overlap count cannot supply membership."""
    columns = requested
    if columns is None:
        columns = schema.get("core_features") or schema.get("core_feature_columns")
    if columns is None:
        return []
    if (
        not isinstance(columns, (list, tuple))
        or any(not isinstance(name, str) or not name for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("Core features must be a list of distinct feature names.")
    feature_sets = schema.get("feature_sets") or {}
    published = feature_sets.get("all")
    if isinstance(published, list) and set(columns) - set(published):
        raise ValueError("Every core feature must belong to the dataset's published all set.")
    expected = (schema.get("core_feature_overlap") or {}).get("all")
    if expected is not None and len(columns) != expected:
        raise ValueError(f"The dataset reports {expected} core features in its all set.")
    return list(columns)


def core_reference(features, *, missing=-1):
    """Return the SDK core average, masking a dataset-specific missing sentinel."""
    from everestapi import scoring

    if missing is not None:
        features = features.mask(features == missing)
    return scoring.core_feature_average(features)


def _percentile(values, expeds):
    groups = values.groupby(expeds, sort=False, observed=True)
    return (groups.rank() - 0.5) / groups.transform("count")


def benchmark_neutralize(predictions, benchmark, expeds, alpha):
    """Subtract a fraction of the benchmark projection and rerank within each exped."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Benchmark neutralization strength must be finite and nonnegative.")
    ranks = _percentile(predictions, expeds)
    if benchmark is None or alpha == 0:
        return ranks

    benchmark = benchmark.reindex(predictions.index).replace([np.inf, -np.inf], np.nan)
    common = ranks.notna() & benchmark.notna()
    # Keep the floating-point reductions stable when a caller reorders rows.
    matched_index = ranks.index[common].sort_values()
    labels = expeds.reindex(matched_index)
    ordinary = pd.Series(ndtri(ranks.to_numpy()), index=predictions.index)
    reference = pd.Series(
        ndtri(_percentile(benchmark.loc[matched_index], labels)), index=matched_index
    )
    grouping = {"sort": False, "observed": True}
    reference_groups = reference.groupby(labels, **grouping)
    centered = reference - reference_groups.transform("mean")
    denominator = (centered**2).groupby(labels, **grouping).transform("sum")
    matched = ordinary.loc[matched_index]
    prediction_mean = matched.groupby(labels, **grouping).transform("mean")
    signal = matched - prediction_mean
    numerator = (centered * signal).groupby(labels, **grouping).transform("sum")
    usable = denominator.gt(0) & reference_groups.transform("size").ge(5)
    projection = centered * numerator.div(denominator.where(usable))
    residual = signal - alpha * projection
    # Rank normalization should not amplify floating-point dust after cancellation.
    adjusted = residual.mask(residual.abs().le(1e-12), 0.0) + prediction_mean
    ordinary.loc[adjusted.index] = adjusted.where(usable, matched)
    return _percentile(ordinary, expeds)


def _rank_corr(left, right):
    if left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    return left.rank().corr(right.rank())


def score_expeds(
    holdout,
    exped,
    target,
    predictions,
    benchmark=None,
    *,
    core_features=None,
    missing=-1,
    zero_rule=False,
):
    """Evaluate each candidate with the SDK on each labeled cross-section."""
    from everestapi import scoring

    core_features = list(core_features) if core_features is not None else []
    if set(core_features) - set(holdout.columns):
        raise ValueError("The labeled scoring frame is missing specified core features.")
    reference = None if benchmark is None else benchmark.reindex(holdout.index)
    rows = []
    for label, group in holdout.groupby(exped, sort=True, observed=True):
        actual = group[target].dropna()
        indexes = actual.index
        bench = None if reference is None else reference.loc[indexes].dropna()
        has_benchmark = bench is not None and not bench.empty
        benchmark_fit = scoring.fit20(bench, actual) if has_benchmark else np.nan
        core = group.loc[indexes, core_features] if core_features else None
        if core is not None and missing is not None:
            core = core.mask(core == missing)
        benchmark_inov = (
            scoring.inov(bench, core, actual, zero_rule=zero_rule)
            if has_benchmark and core is not None
            else np.nan
        )
        for name, values in predictions.items():
            pred = values.reindex(indexes).dropna()
            fit = scoring.fit20(pred, actual) if not pred.empty else np.nan
            record = {
                "model": name,
                "exped": label,
                "fit": fit,
                "matched_fit_delta": np.nan,
                "benchmark_fit": benchmark_fit,
                "unq_covariance": np.nan,
                "inov": (
                    scoring.inov(pred, core, actual, zero_rule=zero_rule)
                    if core is not None and not pred.empty
                    else np.nan
                ),
                "benchmark_inov": benchmark_inov,
                "benchmark_rank_corr": np.nan,
            }
            if has_benchmark:
                common = pred.index.intersection(bench.index)
                if len(common):
                    matched_pred, matched_bench = pred, bench
                    matched_fit, matched_benchmark_fit = fit, benchmark_fit
                    if len(common) < len(pred):
                        matched_pred = pred.loc[common]
                        matched_fit = scoring.fit20(matched_pred, actual)
                    if len(common) < len(bench):
                        matched_bench = bench.loc[common]
                        matched_benchmark_fit = scoring.fit20(matched_bench, actual)
                    record.update(
                        matched_fit_delta=matched_fit - matched_benchmark_fit,
                        unq_covariance=scoring.unq20(
                            matched_pred, matched_bench, actual, zero_rule=zero_rule
                        ),
                        benchmark_rank_corr=_rank_corr(matched_pred, matched_bench),
                    )
            rows.append(record)
    return pd.DataFrame(rows)


def event_metrics(per_exped, score_scale):
    """Average the component covariances before applying the event score."""
    from everestapi import scoring

    columns = [
        "fit",
        "inov",
        "benchmark_fit",
        "benchmark_inov",
        "unq_covariance",
        "matched_fit_delta",
    ]
    means = per_exped[columns].mean()
    unq = means["unq_covariance"]
    delta = means["matched_fit_delta"]
    relative_proxy = delta + scoring.EVENT_UNQ_WEIGHT * unq
    terms = means[["fit", "inov", "benchmark_fit", "benchmark_inov"]]
    signal = terms["fit"] + scoring.EVENT_UNQ_WEIGHT * unq + terms["inov"]
    signal -= terms["benchmark_fit"] + terms["benchmark_inov"]
    exact_score = partial_score = np.nan
    if score_scale is not None:
        if not np.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("Event score scale must be finite and positive.")
        if pd.notna(signal):
            exact_score = scoring.event_score(
                terms["fit"],
                unq,
                terms["inov"],
                score_multiple_constant=score_scale,
                benchmark_fit=terms["benchmark_fit"],
                benchmark_inov=terms["benchmark_inov"],
            )
        if pd.notna(relative_proxy):
            partial_score = scoring.event_score(
                delta,
                unq,
                0,
                score_multiple_constant=score_scale,
                benchmark_fit=0,
                benchmark_inov=0,
            )
    return {
        "event_signal": signal,
        "event_score": exact_score,
        "relative_signal_proxy": relative_proxy,
        "round_score_proxy": partial_score,
    }
