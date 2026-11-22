"""Select Everesteer candidates by walk-forward evaluation and package full-fit models."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from importlib.metadata import version
from inspect import signature
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from everestapi import scoring
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BLEND_WEIGHTS = (0.25, 0.5, 0.75)
LIBRARIES = ("cloudpickle", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy")
BASE_CONFIGS = {
    "lightgbm": {
        "family": "lightgbm",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
    },
    "ridge": {"family": "ridge", "alpha": 10.0},
}
TRIAL_CONFIGS = {
    "lightgbm-regularized": {
        "family": "lightgbm",
        "n_estimators": 340,
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 60,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "early_stop": True,
    },
    "ridge-scaled-low": {"family": "ridge", "relative_alpha": 0.001},
    "ridge-scaled-high": {"family": "ridge", "relative_alpha": 0.03},
}
RESIDUAL_RIDGE = {"family": "ridge", "relative_alpha": 0.01, "neutralization": 0.5}


def select_features(frame, columns, missing):
    matrix = frame.reindex(columns=columns).astype("float32")
    return matrix if missing is None else matrix.mask(matrix == missing)


def ridge_design(matrix, fill=0.0):
    """Give missing values a separate coefficient and no numeric bin contribution."""
    values = matrix.to_numpy()
    missing = np.isnan(values)
    n_columns = values.shape[1]
    design = np.empty((len(values), 2 * n_columns), dtype=np.float32)
    numeric = design[:, :n_columns]
    numeric[:, :] = values
    np.copyto(numeric, np.asarray(fill, dtype=np.float32), where=missing)
    design[:, n_columns:] = missing
    return design


def benchmark_residual_target(target, benchmark, expeds, strength):
    """Remove part of the benchmark direction using only labels inside the fit set."""
    target = target.astype(float)
    reference = benchmark.reindex(target.index).replace([np.inf, -np.inf], np.nan)
    matched = reference.notna()
    reference = reference.loc[matched]
    groups = expeds.reindex(target.index).loc[matched]
    ranks = reference.groupby(groups).rank(pct=True)
    centered = ranks - ranks.groupby(groups).transform("mean")
    sums = (centered**2).groupby(groups).transform("sum")
    counts = centered.groupby(groups).transform("size")
    slopes = (centered * target.loc[matched]).groupby(groups).transform("sum")
    correction = (centered * slopes.div(sums.where(sums.gt(0) & counts.ge(5)))).fillna(0.0)
    result = target.copy()
    result.loc[matched] -= strength * correction
    return result


def feature_correlations(values, target):
    """Correlate each column with the target over that column's finite pairs."""
    present = np.isfinite(values) & np.isfinite(target[:, None])
    x = np.where(present, values, 0.0)
    y = np.where(present, target[:, None], 0.0)
    count = present.sum(axis=0)
    divisor = np.maximum(1, count)
    x_sum = x.sum(axis=0)
    y_sum = y.sum(axis=0)
    cov = np.sum(x * y, axis=0) - x_sum * y_sum / divisor
    x_var = np.maximum(0, np.sum(x * x, axis=0) - x_sum**2 / divisor)
    y_var = np.maximum(0, np.sum(y * y, axis=0) - y_sum**2 / divisor)
    scale = np.sqrt(x_var * y_var)
    return np.divide(cov, scale, out=np.full(len(count), np.nan), where=(count >= 3) & (scale > 0))


def feature_stability(fit, columns, missing, target, exped, benchmark=None, core=None):
    """Measure feature relevance and reference exposure on labeled fit expeds only."""
    matrix = select_features(fit, columns, missing).replace([np.inf, -np.inf], np.nan)
    labels = fit[target]
    expeds = fit[exped]
    references = {"fit": labels}
    if benchmark is not None:
        references["benchmark"] = benchmark
        references["residual"] = benchmark_residual_target(labels, benchmark, expeds, 1.0).where(
            np.isfinite(benchmark.reindex(fit.index))
        )
    if core is not None:
        references["core"] = core
    ranks = {
        name: values.reindex(fit.index)
        .replace([np.inf, -np.inf], np.nan)
        .groupby(expeds, sort=False)
        .rank()
        .to_numpy(dtype=float)
        for name, values in references.items()
    }
    records = {name: [] for name in (*references, "missing")}
    for positions in fit.groupby(exped, sort=False).indices.values():
        rows = matrix.iloc[positions]
        ranked = rows.rank().to_numpy()
        for name, values in ranks.items():
            records[name].append(feature_correlations(ranked, values[positions]))
        records["missing"].append(
            feature_correlations(rows.isna().to_numpy(dtype=float), ranks["fit"][positions])
        )

    result = pd.DataFrame(index=pd.Index(columns, name="feature"))
    result["missing_rate"] = matrix.isna().mean(axis=0)
    missing_corr = pd.DataFrame(records["missing"], columns=columns)
    result["missing_fit"] = missing_corr.mean(axis=0)
    for name, values in records.items():
        if name == "missing":
            continue
        correlations = pd.DataFrame(values, columns=columns)
        mean = correlations.mean(axis=0)
        if name in {"benchmark", "core"}:
            result[f"{name}_corr"] = mean
            result[f"{name}_abs_corr"] = correlations.abs().mean(axis=0)
        else:
            std = correlations.std(axis=0, ddof=0)
            consistent = correlations.mul(np.sign(mean), axis=1).gt(0)
            sign_share = consistent.sum(axis=0) / correlations.count(axis=0).replace(0, np.nan)
            result[f"{name}_mean"] = mean
            result[f"{name}_std"] = std
            result[f"{name}_sign_share"] = sign_share
            result[f"{name}_stability"] = mean.abs() * sign_share / (1 + std)
    return result


def stable_feature_subset(stability, selection, max_features):
    if selection == "fit":
        score = stability["fit_stability"]
    elif selection == "residual":
        if "residual_stability" not in stability:
            raise ValueError("Residual feature selection requires the aligned training benchmark.")
        score = stability["residual_stability"]
    elif selection == "low_exposure":
        if "core_abs_corr" not in stability:
            raise ValueError("Low-exposure feature selection requires a core reference series.")
        score = stability["fit_stability"] * (1 - stability["core_abs_corr"].clip(0, 1))
    else:
        raise ValueError(f"Unknown feature selection {selection!r}.")
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features < 1:
        raise ValueError("Feature selection requires a positive integer max_features.")
    selected = score.dropna().sort_index().sort_values(ascending=False, kind="stable")
    if selected.empty:
        raise ValueError(f"No measurable features for {selection} selection in this fit window.")
    return selected.head(max_features).index.tolist()


def fit_tree(matrix, labels, params, expeds, embargo):
    import lightgbm as lgb

    params = params.copy()
    early_stop = params.pop("early_stop", False)
    tree = lgb.LGBMRegressor(**params, verbosity=-1, n_jobs=4, random_state=821)
    if early_stop and expeds is not None:
        ordered = sorted(expeds.unique())
        calibration_count = min(20, max(5, len(ordered) // 5))
        cutoff = len(ordered) - calibration_count - embargo
        if cutoff >= 5:
            inner_fit = expeds.isin(ordered[:cutoff])
            calibration = expeds.isin(ordered[-calibration_count:])
            eval_x, eval_y = matrix.loc[calibration], labels.loc[calibration]
            validation = (
                {"eval_X": eval_x, "eval_y": eval_y}
                if "eval_X" in signature(tree.fit).parameters
                else {"eval_set": [(eval_x, eval_y)]}
            )
            tree.fit(
                matrix.loc[inner_fit],
                labels.loc[inner_fit],
                callbacks=[lgb.early_stopping(25, verbose=False)],
                **validation,
            )
            tree.set_params(n_estimators=tree.best_iteration_)
    return tree.fit(matrix, labels)


def model_input(matrix, family, missing_policy, sentinel, fill):
    if missing_policy == "sentinel":
        return matrix if sentinel is None else matrix.fillna(sentinel)
    if family == "ridge":
        return ridge_design(matrix, fill)
    if missing_policy == "native":
        return matrix
    flags = matrix.isna().astype("float32")
    flags.columns = [f"__eiq_missing_{index}" for index in range(len(matrix.columns))]
    if missing_policy == "median-indicator":
        matrix = matrix.fillna(dict(zip(matrix.columns, fill, strict=True)))
    return pd.concat([matrix, flags], axis=1)


def fitted_predictor(estimator, columns, missing, family, missing_policy=None, fill=0.0):
    missing_policy = missing_policy or ("indicator" if family == "ridge" else "native")

    def predict(live_features):
        matrix = select_features(live_features, columns, missing)
        inputs = model_input(matrix, family, missing_policy, missing, fill)
        return pd.DataFrame({"prediction": estimator.predict(inputs)}, index=live_features.index)

    predict.feature_columns = tuple(columns)
    return predict


def model_predictions(
    fit,
    columns,
    missing,
    target,
    *,
    configs=None,
    exped=None,
    embargo=40,
    benchmark=None,
    core=None,
):
    configs = BASE_CONFIGS if configs is None else configs
    expeds = None if exped is None else fit[exped]
    predictors = {}
    stability = {}
    last_design = None
    matrix = inputs = fill = None
    for name, config in configs.items():
        params = config.copy()
        family = params.pop("family")
        if family not in {"lightgbm", "ridge"}:
            raise ValueError(f"Unknown model family {family!r}.")
        selection = params.pop("feature_selection", None)
        max_features = params.pop("max_features", len(columns))
        auxiliary = params.pop("auxiliary_target", None)
        strength = params.pop("neutralization", None)
        pca_components = params.pop("pca_components", None)
        key = auxiliary or target
        selected = list(columns)
        if selection:
            if exped is None:
                raise ValueError("Stable feature selection requires an exped column.")
            if key not in stability:
                stability[key] = feature_stability(
                    fit, columns, missing, key, exped, benchmark, core
                )
            selected = stable_feature_subset(stability[key], selection, max_features)
        missing_policy = params.pop(
            "missing_policy", "indicator" if family == "ridge" else "native"
        )
        valid_missing = {"sentinel", "native", "indicator", "median-indicator"}
        if missing_policy not in valid_missing or (
            family == "ridge" and missing_policy == "native"
        ):
            raise ValueError(f"Unsupported missing policy {missing_policy!r} for {family}.")
        if missing_policy == "sentinel" and missing is None and family == "ridge":
            raise ValueError("Sentinel Ridge requires the dataset to declare a missing sentinel.")
        if pca_components is not None and family != "ridge":
            raise ValueError("PCA is supported for Ridge models.")
        labels = fit[key]
        available = None
        if auxiliary:
            available = (labels.notna() & np.isfinite(labels)).to_numpy()
            labels = labels.iloc[available]
            if len(labels) < 3:
                raise ValueError(
                    f"Auxiliary target {auxiliary!r} has fewer than three usable fit rows."
                )
            if labels.corr(fit.loc[labels.index, target]) < 0:
                labels = -labels
        if strength is not None:
            if benchmark is None or expeds is None:
                raise ValueError(
                    "Residual targets require an aligned training benchmark and expeds."
                )
            labels = benchmark_residual_target(labels, benchmark, expeds, strength)
        design = (tuple(selected), family, missing_policy, auxiliary)
        if design != last_design:
            matrix = inputs = None
            matrix = select_features(
                fit if available is None else fit.iloc[available], selected, missing
            )
            fill = (
                matrix.median().fillna(0.0).to_numpy(dtype=np.float32)
                if missing_policy == "median-indicator"
                else 0.0
            )
            inputs = model_input(matrix, family, missing_policy, missing, fill)
            matrix = None
            last_design = design
        if family == "lightgbm":
            fit_times = None if expeds is None else expeds.loc[labels.index]
            estimator = fit_tree(inputs, labels, params, fit_times, embargo)
        else:
            steps = []
            if "relative_alpha" in params:
                alpha = params["relative_alpha"] * len(labels)
                steps.append(StandardScaler())
            else:
                alpha = params["alpha"]
                if pca_components is not None:
                    steps.append(StandardScaler())
            if pca_components is not None:
                if (
                    not isinstance(pca_components, int)
                    or isinstance(pca_components, bool)
                    or pca_components < 1
                ):
                    raise ValueError("PCA requires a positive integer pca_components.")
                steps.append(
                    PCA(
                        n_components=min(pca_components, *inputs.shape),
                        svd_solver="randomized",
                        random_state=821,
                    )
                )
            ridge = Ridge(alpha=alpha, solver="lsqr")
            estimator = make_pipeline(*steps, ridge) if steps else ridge
            estimator.fit(inputs, labels)
        predictors[name] = fitted_predictor(
            estimator, selected, missing, family, missing_policy, fill
        )
    return predictors


def ranked_components(tree, linear, expeds):
    """Rank each constituent using only the rows of the current exped."""
    return pd.DataFrame({"tree": tree, "linear": linear}).groupby(expeds).rank(pct=True)


def rank_blend(forecasts, expeds, weights):
    """Blend percentile ranks within each exped using nonnegative normalized weights."""
    weights = pd.Series(weights, dtype=float)
    if (
        weights.empty
        or not np.isfinite(weights).all()
        or weights.lt(0).any()
        or not 0 < weights.sum() < np.inf
    ):
        raise ValueError("A rank blend needs finite nonnegative weights with a positive total.")
    expeds = expeds.reindex(forecasts.index)
    if expeds.isna().any():
        raise ValueError("A rank blend needs an exped label for each row.")
    ranks = forecasts.loc[:, weights.index].groupby(expeds, sort=False).rank(pct=True)
    return ranks.mul(weights / weights.sum()).sum(axis=1, skipna=False).rename("prediction")


def weighted_rank_prediction(predictors, exped_column, weights):
    weights = {name: value for name, value in weights.items() if value != 0}
    members = {name: predictors[name] for name in weights}

    def predict(live_features):
        forecasts = pd.DataFrame(
            {name: model(live_features)["prediction"] for name, model in members.items()},
            index=live_features.index,
        )
        return rank_blend(forecasts, live_features[exped_column], weights).to_frame()

    predict.feature_columns = (
        tuple(sorted({column for member in members.values() for column in member.feature_columns}))
        if members and all(hasattr(member, "feature_columns") for member in members.values())
        else ()
    )
    return predict


def blend_prediction(predictors, exped_column, weight):
    return weighted_rank_prediction(
        predictors, exped_column, {"lightgbm": weight, "ridge": 1 - weight}
    )


def benchmark_surrogate_predictor(predictor, reference_predictor, exped_column, alpha):
    """Hedge against a feature-only benchmark estimate trained on historical data."""
    from hackathon_scoring import benchmark_neutralize

    def predict(live_features):
        forecasts = predictor(live_features)["prediction"]
        reference = reference_predictor(live_features)["prediction"]
        return benchmark_neutralize(
            forecasts, reference, live_features[exped_column], alpha
        ).to_frame("prediction")

    if hasattr(predictor, "feature_columns") and hasattr(reference_predictor, "feature_columns"):
        predict.feature_columns = tuple(
            sorted(set(predictor.feature_columns) | set(reference_predictor.feature_columns))
        )
    return predict


def dataset_columns(train, schema, feature_set=None):
    target = schema.get("primary_target")
    exped = schema.get("exped_column") or "exped"
    if not target or target not in train or exped not in train:
        raise ValueError(
            "The train parquet must include the schema's primary target and exped column."
        )
    feature_sets = schema.get("feature_sets") or {}
    if feature_set is not None:
        declared = feature_sets.get(feature_set)
        if not isinstance(declared, list) or not declared:
            raise ValueError(
                f"Feature set {feature_set!r} is missing or empty in the dataset schema."
            )
    elif "all" in feature_sets:
        declared = feature_sets["all"]
    else:
        declared = [
            col for members in feature_sets.values() if isinstance(members, list) for col in members
        ]
    if isinstance(declared, list) and declared:
        features = sorted(set(declared))
    else:
        # The official starter uses the parquet's feature_ convention when only compact schema is present.
        features = sorted(col for col in train if col.startswith("feature_"))
    excluded = {target, exped, "id", "data_type", *(schema.get("targets") or [])}
    if not features or set(features) & excluded or set(features) - set(train.columns):
        raise ValueError("Training features do not match the parquet and verbose dataset schema.")
    return target, exped, features, (schema.get("feature_encoding") or {}).get("missing")


def chronological_split(expeds, holdout_size, embargo_size):
    if expeds.isna().any():
        raise ValueError("Train has rows without an exped label.")
    # Everesteer training tokens are zero-padded with chronological lexical order.
    ordered = sorted(expeds.unique().tolist())
    fit_end = len(ordered) - holdout_size - embargo_size
    if holdout_size < 2 or embargo_size < 1 or fit_end < 1:
        raise ValueError(
            "The split needs at least one fit exped, one embargo exped and two holdout expeds."
        )
    return {
        "fit_expeds": ordered[:fit_end],
        "embargo_expeds": ordered[fit_end : fit_end + embargo_size],
        "holdout_expeds": ordered[fit_end + embargo_size :],
    }


def walk_forward_splits(split, fold_count, fold_size, embargo_size):
    """Make disjoint expanding-window evaluations inside the pre-holdout history."""
    development = split["fit_expeds"]
    start = len(development) - fold_count * fold_size
    if fold_count < 2 or fold_size < 2 or embargo_size < 1 or start - embargo_size < 2:
        raise ValueError(
            "Insufficient history for two or more walk-forward folds, their embargoes, "
            "and the final holdout. Try smaller --fold-expeds or --holdout-expeds."
        )
    folds = []
    for index in range(fold_count):
        beginning = start + index * fold_size
        cutoff = beginning - embargo_size
        folds.append(
            {
                "name": f"fold-{index + 1}",
                "fit_expeds": development[:cutoff],
                "embargo_expeds": development[cutoff:beginning],
                "holdout_expeds": development[beginning : beginning + fold_size],
            }
        )
    return folds


def align_benchmark(benchmark, holdout, exped, *, minimum_coverage=0.5):
    """Match both opaque id and exped, then validate coverage of the labeled holdout."""
    if benchmark is None:
        return None, None
    columns = [col for col in benchmark.select_dtypes(include="number") if col not in {exped, "id"}]
    if exped not in benchmark or not columns:
        print(
            "Benchmark metrics unavailable because benchmark data lacks an exped or score column."
        )
        return None, None
    scores = benchmark[columns].replace([np.inf, -np.inf], np.nan).mean(axis=1)
    scores.index = pd.MultiIndex.from_arrays([benchmark.index, benchmark[exped]])
    if not scores.index.is_unique:
        print(
            "Benchmark metrics unavailable because the benchmark contains duplicate id–exped pairs."
        )
        return None, None
    ids_and_expeds = pd.MultiIndex.from_arrays([holdout.index, holdout[exped]])
    aligned = pd.Series(scores.reindex(ids_and_expeds).to_numpy(), index=holdout.index)
    coverage = float(aligned.notna().mean())
    print(
        f"Benchmark matches {aligned.notna().sum():,}/{len(aligned):,} labeled rows "
        f"({coverage:.2%}) on both id and exped."
    )
    if coverage < minimum_coverage:
        print(
            "Benchmark metrics unavailable because matching covers less than half of the holdout."
        )
        return None, coverage
    return aligned, coverage


def spearman(left, right):
    if left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    return left.rank().corr(right.rank())


def residual_spearman(pred, benchmark, target):
    """Documented offline UNQ proxy from Everesteer's model implementation guide."""
    design = np.column_stack((benchmark, np.ones(len(benchmark))))
    residual = pred.to_numpy() - design @ (np.linalg.pinv(design, rcond=1e-6) @ pred.to_numpy())
    if np.max(np.abs(residual - residual.mean())) < 1e-12:
        return 0.0
    return spearman(pd.Series(residual, index=pred.index), target)


def evaluate(holdout, exped, target, predictions, benchmark):
    rows = []
    for label, group in holdout.groupby(exped, sort=True):
        actual = group[target]
        reference = None if benchmark is None else benchmark.loc[group.index].dropna()
        for name, values in predictions.items():
            pred = values.loc[group.index]
            record = {
                "model": name,
                "exped": label,
                "fit": scoring.fit20(pred, actual),
                "matched_fit_delta": np.nan,
                "unq_covariance": np.nan,
                "unq_spearman": np.nan,
                "benchmark_rank_corr": np.nan,
            }
            if reference is not None and len(reference) >= 5:
                matched = pred.loc[reference.index]
                targets = actual.loc[reference.index]
                record.update(
                    matched_fit_delta=scoring.fit20(matched, targets)
                    - scoring.fit20(reference, targets),
                    unq_covariance=scoring.unq20(matched, reference, targets, zero_rule=False),
                    unq_spearman=residual_spearman(matched, reference, targets),
                    benchmark_rank_corr=spearman(matched, reference),
                )
            rows.append(record)
    return pd.DataFrame(rows)


def summarize(per_exped, holdout_expeds, score_scale):
    boundary = len(holdout_expeds) // 2
    first_half = holdout_expeds[:boundary]

    def score(group):
        delta = group["matched_fit_delta"].mean()
        unq = group["unq_covariance"].mean()
        if score_scale is None or pd.isna(delta) or pd.isna(unq):
            return np.nan
        # SDK event_score with FIT already benchmark-adjusted. Both INOV terms are unobserved.
        return scoring.event_score(
            delta,
            unq,
            0.0,
            score_multiple_constant=score_scale,
            benchmark_fit=0.0,
            benchmark_inov=0.0,
        )

    summary = []
    for name, group in per_exped.groupby("model", sort=False):
        early = group["exped"].isin(first_half)
        fits = group["fit"]
        run = (fits >= 0).cumsum()
        negative = (fits < 0).astype(int)
        summary.append(
            {
                "model": name,
                "fit": fits.mean(),
                "unq_covariance": group["unq_covariance"].mean(),
                "unq_spearman": group["unq_spearman"].mean(),
                "relative_signal_proxy": (
                    group["matched_fit_delta"].mean()
                    + scoring.EVENT_UNQ_WEIGHT * group["unq_covariance"].mean()
                ),
                "round_score_proxy": score(group),
                "benchmark_rank_corr": group["benchmark_rank_corr"].mean(),
                "benchmark_exped_share": group["unq_covariance"].notna().mean(),
                "fit_std": fits.std(),
                "fit_positive_share": (fits > 0).mean(),
                "longest_negative_fit_run": int(negative.groupby(run).sum().max()),
                "first_half_fit": group.loc[early, "fit"].mean(),
                "second_half_fit": group.loc[~early, "fit"].mean(),
                "first_half_score_proxy": score(group.loc[early]),
                "second_half_score_proxy": score(group.loc[~early]),
            }
        )
    return pd.DataFrame(summary).set_index("model")


def check_predictions(predictor, sample):
    result = predictor(sample)
    if (
        not isinstance(result, pd.DataFrame)
        or result.shape != (len(sample), 1)
        or list(result.columns) != ["prediction"]
        or not result.index.equals(sample.index)
        or result.index.nlevels != 1
        or not result.index.is_unique
        or result.index.hasnans
        or not pd.api.types.is_numeric_dtype(result["prediction"])
        or pd.api.types.is_bool_dtype(result["prediction"])
        or pd.api.types.is_complex_dtype(result["prediction"])
        or not np.isfinite(result["prediction"].to_numpy(dtype=float, na_value=np.nan)).all()
    ):
        raise ValueError("A model returned invalid values or changed the source row IDs.")
    return result


def package(predictors, sample, exped, folder):
    if __name__ != "__main__":
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
    if "hackathon_scoring" in sys.modules:
        cloudpickle.register_pickle_by_value(sys.modules["hackathon_scoring"])
    models = {}
    for name, predictor in predictors.items():
        path = folder / f"{name}.pkl"
        expected = check_predictions(predictor, sample)
        if expected["prediction"].nunique() < 2:
            raise ValueError(f"{name} produced constant predictions on the artifact test sample.")
        with path.open("wb") as handle:
            cloudpickle.dump(predictor, handle)
        with path.open("rb") as handle:
            restored = cloudpickle.load(handle)
        pd.testing.assert_frame_equal(
            expected, check_predictions(restored, sample), check_exact=True
        )
        # Independently preserve named inputs and cross-exped isolation in the saved artifact.
        shuffled = sample.iloc[::-1, ::-1]
        pd.testing.assert_frame_equal(
            expected.loc[shuffled.index],
            check_predictions(restored, shuffled),
            rtol=1e-6,
            atol=1e-9,
        )
        for _, rows in sample.groupby(exped, sort=False):
            actual = check_predictions(restored, rows)
            pd.testing.assert_frame_equal(expected.loc[rows.index], actual, rtol=1e-6, atol=1e-9)
        models[name] = {
            "path": f"{folder.name}/{path.name}",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if getattr(predictor, "feature_columns", ()):
            models[name]["feature_columns"] = list(predictor.feature_columns)
    return models


def eligible_reference(reference, index):
    if reference is None:
        return None, None
    matched = reference.reindex(index)
    coverage = float(matched.notna().mean())
    return (matched if coverage >= 0.5 else None), coverage


def fit_expeds(train, labeled, ordered, exped, maximum=None):
    if maximum is not None and len(ordered) > maximum:
        positions = np.linspace(0, len(ordered) - 1, num=maximum, dtype=int)
        ordered = [ordered[position] for position in positions]
    return labeled & train[exped].isin(ordered)


def forecast_window(train, labeled, split, exped, target, columns, predictors, reference):
    frame = train.loc[train[exped].isin(split["holdout_expeds"]), [exped, *columns]]
    scored = train.loc[frame.index, [exped, target]].loc[labeled]
    if scored[exped].nunique() < 2:
        raise ValueError(f"{split.get('name', 'Final holdout')} has fewer than two labeled expeds.")
    forecasts = pd.DataFrame(
        {
            name: check_predictions(predictor, frame)["prediction"]
            for name, predictor in predictors.items()
        },
        index=frame.index,
    )
    matched, coverage = eligible_reference(reference, scored.index)
    return forecasts, scored, frame[exped], matched, coverage


def fold_summary(metrics, split, score_scale):
    return (
        summarize(metrics, split["holdout_expeds"], score_scale)
        .reset_index()
        .assign(fold=split["name"])
    )


def rank_configurations(fold_metrics, names, metric=None):
    """Rank a common set of folds, discounting unstable mean performance."""
    candidates = fold_metrics.loc[fold_metrics["model"].isin(names)]
    if metric is None:
        benchmark_covers_folds = candidates["benchmark_exped_share"].ge(0.5).all()
        metric = next(
            name
            for name in ("round_score_proxy", "relative_signal_proxy", "fit")
            if (name == "fit" or benchmark_covers_folds) and candidates[name].notna().all()
        )
    ranking = candidates.groupby("model")[metric].agg(["mean", "std", "min"])
    ranking = ranking.rename(columns={"min": "worst_fold"})
    ranking["robust_score"] = ranking["mean"] - 0.25 * ranking["std"]
    ranking["selection_metric"] = metric
    return ranking.sort_values(["robust_score", "worst_fold"], ascending=False)


def average_rank_correlations(forecasts, expeds):
    matrices = [block.rank().corr() for _, block in forecasts.groupby(expeds)]
    result = pd.concat(matrices).groupby(level=0).mean()
    return {
        name: {other: float(value) if pd.notna(value) else None for other, value in values.items()}
        for name, values in result.to_dict().items()
    }


def train_and_package(train, benchmark, schema, score_scale, args, *, selection_metric=None):
    target, exped, columns, missing = dataset_columns(train, schema, args.feature_set)
    auxiliary = args.auxiliary_target
    if auxiliary and (
        auxiliary == target
        or auxiliary not in (schema.get("targets") or [])
        or auxiliary not in train
    ):
        raise ValueError(
            "Choose a declared non-primary auxiliary target present in the train parquet."
        )
    train_columns = [exped, target, *columns]
    if auxiliary:
        train_columns.append(auxiliary)
    if (
        train.index.nlevels != 1
        or not train.index.is_unique
        or train.index.hasnans
        or not all(isinstance(row_id, str) and row_id for row_id in train.index)
    ):
        raise ValueError("Train parquet must use unique, nonempty opaque string IDs as its index.")
    if "data_type" in train and set(train["data_type"].dropna().unique()) - {"train"}:
        raise ValueError("Training requires the labeled train split.")
    split = chronological_split(train[exped], args.holdout_expeds, args.embargo_expeds)
    folds = walk_forward_splits(split, args.cv_folds, args.fold_expeds, args.embargo_expeds)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = args.artifacts / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not (args.smoke_test and previous.get("synthetic")) and (
            previous.get("synthetic") != args.smoke_test
            or previous.get("holdout") != split
            or previous.get("primary_target") != target
            or previous.get("folds", folds) != folds
        ):
            raise ValueError(
                "Use a separate artifact directory for a different dataset or holdout."
            )

    labeled = train[target].notna() & np.isfinite(train[target])
    if labeled.sum() < 3:
        raise ValueError("Train parquet has fewer than three usable primary-target labels.")
    reference, _ = align_benchmark(
        benchmark, train.loc[labeled, [exped]], exped, minimum_coverage=0.0
    )
    configs = BASE_CONFIGS.copy()
    if not args.baseline_only:
        configs.update(TRIAL_CONFIGS)
        if reference is not None and all(
            eligible_reference(
                reference,
                train.index[
                    fit_expeds(train, labeled, fold["fit_expeds"], exped, args.max_fit_expeds)
                ],
            )[0]
            is not None
            for fold in folds
        ):
            configs["ridge-benchmark-residual"] = RESIDUAL_RIDGE
    if auxiliary:
        configs["ridge-auxiliary"] = {
            "family": "ridge",
            "relative_alpha": 0.01,
            "auxiliary_target": auxiliary,
        }
    print(
        f"Target {target!r}, {len(columns)} features, missing sentinel {missing!r}, "
        f"{len(folds)} walk-forward folds and a {args.embargo_expeds}-exped embargo."
    )
    print(
        f"Untouched final holdout spans {split['holdout_expeds'][0]} through {split['holdout_expeds'][-1]}."
    )
    fold_outputs = []
    base_metrics = []
    for fold in folds:
        fit = train.loc[
            fit_expeds(train, labeled, fold["fit_expeds"], exped, args.max_fit_expeds),
            train_columns,
        ]
        if len(fit) < 3:
            raise ValueError(f"{fold['name']} has fewer than three labeled training rows.")
        print(
            f"{fold['name']} fits {len(fit):,} rows and evaluates {len(fold['holdout_expeds'])} expeds.",
            flush=True,
        )
        predictors = model_predictions(
            fit,
            columns,
            missing,
            target,
            configs=configs,
            exped=exped,
            embargo=args.embargo_expeds,
            benchmark=reference,
        )
        forecasts, scored, expeds, matched, coverage = forecast_window(
            train, labeled, fold, exped, target, columns, predictors, reference
        )
        metrics = evaluate(scored, exped, target, forecasts, matched)
        base_metrics.append(fold_summary(metrics, fold, score_scale))
        fold_outputs.append((fold, forecasts, scored, expeds, matched))

    base_metrics = pd.concat(base_metrics, ignore_index=True)
    base_rank = rank_configurations(base_metrics, list(configs), selection_metric)
    metric = base_rank["selection_metric"].iloc[0]
    selected = {
        family: next(name for name in base_rank.index if configs[name]["family"] == family)
        for family in BASE_CONFIGS
    }
    blend_candidates = {f"rank-blend-{weight:.2f}": weight for weight in BLEND_WEIGHTS}
    blend_rows = []
    for fold, forecasts, scored, expeds, matched in fold_outputs:
        ranks = ranked_components(
            forecasts[selected["lightgbm"]], forecasts[selected["ridge"]], expeds
        )
        blend_predictions = {
            name: weight * ranks["tree"] + (1 - weight) * ranks["linear"]
            for name, weight in blend_candidates.items()
        }
        metric_rows = evaluate(scored, exped, target, blend_predictions, matched)
        blend_rows.append(fold_summary(metric_rows, fold, score_scale))

    trial_metrics = pd.concat([base_metrics, *blend_rows], ignore_index=True)
    trial_ranking = rank_configurations(
        trial_metrics, list(configs) + list(blend_candidates), metric
    )
    chosen_blend = next(name for name in trial_ranking.index if name in blend_candidates)
    chosen_weight = blend_candidates[chosen_blend]
    selected["rank-blend"] = chosen_blend
    finalists = trial_metrics.loc[trial_metrics["model"].isin(selected.values())].copy()
    finalists["configuration"] = finalists["model"]
    finalists["model"] = finalists["model"].map({config: name for name, config in selected.items()})
    final_ranking = rank_configurations(finalists, list(selected), metric)
    primary, *fallbacks = final_ranking.index.tolist()
    print(f"Walk-forward selection uses {metric}, discounted by 0.25 × fold standard deviation.")
    print(trial_ranking.to_string(float_format=lambda x: f"{x:+.5f}"))
    print(f"Selected {selected}. Primary candidate {primary}.")
    if metric != "round_score_proxy":
        print("A comparable round-score proxy across every selection fold was unavailable.")

    chosen = {name: configs[config] for name, config in selected.items() if name != "rank-blend"}
    fit = train.loc[
        fit_expeds(train, labeled, split["fit_expeds"], exped, args.max_fit_expeds), train_columns
    ]
    fitted = model_predictions(
        fit,
        columns,
        missing,
        target,
        configs=chosen,
        exped=exped,
        embargo=args.embargo_expeds,
        benchmark=reference,
    )
    fitted["rank-blend"] = blend_prediction(fitted, exped, chosen_weight)
    forecasts, scored, expeds, matched, coverage = forecast_window(
        train, labeled, split, exped, target, columns, fitted, reference
    )
    if matched is not None:
        forecasts["benchmark"] = matched
    per_exped = evaluate(scored, exped, target, forecasts, matched)
    summary = summarize(per_exped, split["holdout_expeds"], score_scale)
    print("Untouched final evaluation. The score proxy omits the unknown INOV difference.")
    print(summary.to_string(float_format=lambda x: f"{x:+.5f}", na_rep="—"))

    selected_forecasts, selected_expeds = [], []
    for _, forecasts, _, expeds, _ in fold_outputs:
        tree = forecasts[selected["lightgbm"]]
        ridge = forecasts[selected["ridge"]]
        ranks = ranked_components(tree, ridge, expeds)
        selected_forecasts.append(
            pd.DataFrame(
                {
                    "lightgbm": tree,
                    "ridge": ridge,
                    "rank-blend": chosen_weight * ranks["tree"]
                    + (1 - chosen_weight) * ranks["linear"],
                }
            )
        )
        selected_expeds.append(expeds)
    correlations = average_rank_correlations(
        pd.concat(selected_forecasts), pd.concat(selected_expeds)
    )

    deploy = fit_expeds(train, labeled, sorted(train[exped].unique()), exped, args.max_fit_expeds)
    print(f"Refitting on {deploy.sum():,} labeled train rows for deployment.")
    final = model_predictions(
        train.loc[deploy, train_columns],
        columns,
        missing,
        target,
        configs=chosen,
        exped=exped,
        embargo=args.embargo_expeds,
        benchmark=reference,
    )
    final["rank-blend"] = blend_prediction(final, exped, chosen_weight)
    sample = train.loc[train[exped].isin(split["holdout_expeds"][-2:]), [exped, *columns]]
    folder = Path(tempfile.mkdtemp(prefix="fit-", dir=args.artifacts))
    try:
        models = package(final, sample, exped, folder)
        summary.to_csv(folder / "comparison.csv")
        per_exped.to_csv(folder / "per_exped.csv", index=False)
        trial_ranking.to_csv(folder / "selection.csv")
        trial_metrics.to_csv(folder / "trial_fold_metrics.csv", index=False)
        finalists.to_csv(folder / "fold_metrics.csv", index=False)
        manifest = {
            "format_version": 2,
            "synthetic": args.smoke_test,
            "provenance": "synthetic_fixture_v2" if args.smoke_test else "labeled_train_files",
            "primary_target": target,
            "exped_column": exped,
            "feature_columns": columns,
            "feature_set": args.feature_set,
            "fit_limit_expeds": args.max_fit_expeds,
            "library_versions": {name: version(name) for name in LIBRARIES},
            "models": models,
            "holdout": split,
            "folds": folds,
            "primary_model": primary,
            "fallback_models": fallbacks,
            "selection_metric": metric,
            "selected_configurations": selected,
            "trial_configurations": configs,
            "score_multiple_constant": score_scale,
            "benchmark_coverage": coverage,
            "lightgbm_ridge_rank_corr": correlations["lightgbm"]["ridge"],
            "model_rank_correlations": correlations,
            "blend_lightgbm_weight": chosen_weight,
            "comparison_path": f"{folder.name}/comparison.csv",
            "per_exped_path": f"{folder.name}/per_exped.csv",
            "selection_path": f"{folder.name}/selection.csv",
            "fold_metrics_path": f"{folder.name}/fold_metrics.csv",
            "trial_fold_metrics_path": f"{folder.name}/trial_fold_metrics.csv",
        }
        if args.smoke_test:
            practice = sample.copy()
            practice.index = pd.Index(
                [f"synthetic-practice-{i}" for i in range(len(practice))], name="id"
            )
            practice.to_parquet(folder / "practice.parquet")
            manifest["synthetic_practice_path"] = f"{folder.name}/practice.parquet"
        staged_manifest = folder / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staged_manifest.replace(manifest_path)
    except Exception:
        shutil.rmtree(folder)
        raise
    print(
        f"Verified and packaged three models in {manifest_path}. See {folder / 'comparison.csv'}."
    )


def smoke_inputs():
    rng = np.random.default_rng(821)
    days, rows = 360, 24
    features = rng.integers(0, 5, size=(days * rows, 4))
    values = features.astype("float32")
    values[rng.random(values.shape) < 0.12] = -99
    index = pd.Index([f"synthetic-{i:05d}" for i in range(days * rows)], name="id")
    names = ["amber", "cyan", "violet", "jade"]
    train = pd.DataFrame(values, columns=names, index=index)
    train["exped"] = np.repeat([f"exped_{i:04d}" for i in range(days)], rows)
    train["data_type"] = "train"
    regime = np.repeat(np.where(np.arange(days) % 120 < 60, 1, -1), rows)
    train["synthetic_target"] = (
        0.003 * (features[:, 0] - features[:, 1])
        + 0.0015 * regime * ((features[:, 2] > 1) & (features[:, 3] < 2))
        + rng.normal(0, 0.007, len(train))
    )
    train["synthetic_auxiliary"] = (
        0.002 * (features[:, 0] - features[:, 1])
        + 0.002 * (features[:, 2] - features[:, 3])
        + rng.normal(0, 0.006, len(train))
    )
    train.loc[train.index[::79], "synthetic_target"] = np.nan
    benchmark = pd.DataFrame(
        {"reference": features[:, 0] + rng.normal(0, 1, len(train))}, index=index
    )
    benchmark["exped"] = train["exped"]
    benchmark.loc[benchmark.index[-1], "exped"] = "exped_0000"
    schema = {
        "primary_target": "synthetic_target",
        "targets": ["synthetic_target", "synthetic_auxiliary"],
        "feature_sets": {"all": names, "synthetic_core": names[:2]},
        "feature_encoding": {"missing": -99},
    }
    return train.sample(frac=1, random_state=821), benchmark.iloc[::-1], schema, 0.1


def event_proxy_scale(settings):
    weights = settings.get("weights") or {}
    expected = {"fit": 1, "unq": 3, "inov": 1}
    if any(
        name in weights and weights[name] != coefficient for name, coefficient in expected.items()
    ):
        print(
            "Saved scoring weights differ from the event formula. Walk-forward ranking will use FIT."
        )
        return None, "fit"
    scale = weights.get("score_multiple_constant")
    score_scale = float(scale) if scale is not None else None
    if score_scale is not None and (not np.isfinite(score_scale) or score_scale <= 0):
        raise ValueError("Expected a positive score_multiple_constant in the scoring response.")
    return score_scale, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--schema", type=Path, default=Path("data/schema.json"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/train_benchmark.parquet"))
    parser.add_argument("--scoring", type=Path, default=Path("data/scoring.json"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument(
        "--holdout-expeds", type=int, default=100, help="Chronological tail, default 100"
    )
    parser.add_argument(
        "--embargo-expeds", type=int, default=40, help="Gap before the holdout, default 40"
    )
    parser.add_argument("--cv-folds", type=int, default=3, help="Walk-forward folds, default 3")
    parser.add_argument("--fold-expeds", type=int, default=30, help="Expeds per fold, default 30")
    parser.add_argument("--feature-set", help="Choose a feature set from the dataset schema")
    parser.add_argument(
        "--auxiliary-target",
        help="Test Ridge on a declared auxiliary target with a known safe horizon",
    )
    parser.add_argument(
        "--max-fit-expeds", type=int, help="Cap each fit by sampling evenly spaced whole expeds"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run walk-forward with the fast reference models",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Exercise the pipeline on synthetic data"
    )
    args = parser.parse_args()
    if args.embargo_expeds < 20:
        parser.error("Use at least 20 embargo expeds.")
    if args.max_fit_expeds is not None and args.max_fit_expeds < 5:
        parser.error("Use at least five fit expeds when setting --max-fit-expeds.")
    if args.artifacts is None:
        args.artifacts = Path("artifacts/smoke" if args.smoke_test else "artifacts")

    try:
        selection_metric = None
        if args.smoke_test:
            if args.artifacts.resolve() == Path("artifacts").resolve():
                parser.error("Use a separate artifact folder for a synthetic smoke test.")
            print(
                "Synthetic smoke test. All scores and model artifacts are for local verification."
            )
            inputs = smoke_inputs()
        else:
            if not args.train.is_file() or not args.schema.is_file():
                parser.error(
                    "Missing train parquet or dataset schema. Run `python hackathon_setup.py download` first."
                )
            train = pd.read_parquet(args.train)
            schema = json.loads(args.schema.read_text())
            benchmark = pd.read_parquet(args.benchmark) if args.benchmark.is_file() else None
            settings = json.loads(args.scoring.read_text()) if args.scoring.is_file() else {}
            score_scale, selection_metric = event_proxy_scale(settings)
            inputs = train, benchmark, schema, score_scale
        train_and_package(*inputs, args, selection_metric=selection_metric)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
