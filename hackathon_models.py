"""Compare three Everesteer candidates on a fixed holdout, then package full-fit models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from everestapi import scoring
from sklearn.linear_model import Ridge

BLEND_WEIGHTS = (0.25, 0.5, 0.75)  # LightGBM weight, fixed before seeing the holdout.
LIBRARIES = ("cloudpickle", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy")


def select_features(frame, columns, missing):
    matrix = frame.reindex(columns=columns).astype("float32")
    return matrix if missing is None else matrix.mask(matrix == missing)


def ridge_design(matrix):
    """Give missing values a separate coefficient and no numeric bin contribution."""
    values = matrix.to_numpy()
    missing = np.isnan(values)
    n_columns = values.shape[1]
    design = np.empty((len(values), 2 * n_columns), dtype=np.float32)
    numeric = design[:, :n_columns]
    numeric[:, :] = values
    numeric[missing] = 0.0
    design[:, n_columns:] = missing
    return design


def model_predictions(fit, columns, missing, target):
    import lightgbm as lgb

    matrix = select_features(fit, columns, missing)
    tree = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31, verbosity=-1, n_jobs=4
    )
    tree.fit(matrix, fit[target])
    linear = Ridge(alpha=10.0, solver="lsqr")
    linear.fit(ridge_design(matrix), fit[target])

    def predict_tree(live_features):
        matrix = select_features(live_features, columns, missing)
        return pd.DataFrame({"prediction": tree.predict(matrix)}, index=live_features.index)

    def predict_linear(live_features):
        matrix = select_features(live_features, columns, missing)
        return pd.DataFrame(
            {"prediction": linear.predict(ridge_design(matrix))}, index=live_features.index
        )

    return {"lightgbm": predict_tree, "ridge": predict_linear}


def ranked_components(tree, linear, expeds):
    """Rank each constituent using only the rows of the current exped."""
    return pd.DataFrame({"tree": tree, "linear": linear}).groupby(expeds).rank(pct=True)


def blend_prediction(predictors, exped_column, weight):
    tree, linear = predictors["lightgbm"], predictors["ridge"]

    def predict(live_features):
        expeds = live_features[exped_column]
        if expeds.isna().any():
            raise ValueError(f"The blend needs an exped label for each row in {exped_column!r}.")
        ranks = ranked_components(
            tree(live_features)["prediction"], linear(live_features)["prediction"], expeds
        )
        return (weight * ranks["tree"] + (1 - weight) * ranks["linear"]).to_frame("prediction")

    return predict


def dataset_columns(train, schema):
    target = schema.get("primary_target")
    exped = schema.get("exped_column") or "exped"
    if not target or target not in train or exped not in train:
        raise ValueError(
            "The train parquet must include the schema's primary target and exped column."
        )
    feature_sets = schema.get("feature_sets") or {}
    if "all" in feature_sets:
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


def align_benchmark(benchmark, holdout, exped):
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
        f"Benchmark matches {aligned.notna().sum():,}/{len(aligned):,} labeled holdout rows "
        f"({coverage:.2%}) on both id and exped."
    )
    if coverage < 0.5:
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
                "round_score_proxy": score(group),
                "benchmark_rank_corr": group["benchmark_rank_corr"].mean(),
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
        or not result.index.is_unique
        or result.index.hasnans
        or not np.isfinite(result["prediction"]).all()
    ):
        raise ValueError("A model returned invalid values or changed the source row IDs.")
    return result


def package(predictors, sample, exped, folder):
    if __name__ != "__main__":
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
    models = {}
    for name, predictor in predictors.items():
        path = folder / f"{name}.pkl"
        expected = check_predictions(predictor, sample)
        with path.open("wb") as handle:
            cloudpickle.dump(predictor, handle)
        with path.open("rb") as handle:
            restored = cloudpickle.load(handle)
        pd.testing.assert_frame_equal(
            expected, check_predictions(restored, sample), check_exact=True
        )
        # Independently preserve named inputs and cross-exped isolation in the saved artifact.
        shuffled = sample.loc[:, list(reversed(sample.columns))]
        pd.testing.assert_frame_equal(
            expected, check_predictions(restored, shuffled), check_exact=True
        )
        for _, rows in sample.groupby(exped, sort=False):
            actual = check_predictions(restored, rows)
            pd.testing.assert_frame_equal(expected.loc[rows.index], actual, rtol=1e-6, atol=1e-9)
        models[name] = {
            "path": f"{folder.name}/{path.name}",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
    return models


def train_and_package(train, benchmark, schema, score_scale, args):
    target, exped, columns, missing = dataset_columns(train, schema)
    if (
        not train.index.is_unique
        or train.index.hasnans
        or not all(isinstance(row_id, str) and row_id for row_id in train.index)
    ):
        raise ValueError("Train parquet must use unique, nonempty opaque string IDs as its index.")
    split = chronological_split(train[exped], args.holdout_expeds, args.embargo_expeds)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = args.artifacts / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("synthetic") != args.smoke_test
            or previous.get("holdout") != split
            or previous.get("primary_target") != target
        ):
            raise ValueError(
                "Use a separate artifact directory for a different dataset or holdout."
            )

    labeled = train[target].notna() & np.isfinite(train[target])
    fit_rows = labeled & train[exped].isin(split["fit_expeds"])
    holdout = train.loc[train[exped].isin(split["holdout_expeds"]), [exped, target, *columns]]
    scored = holdout.loc[labeled, [exped, target]]
    if fit_rows.sum() < 3 or scored[exped].nunique() < 2:
        raise ValueError("The chronological split has insufficient finite target values.")
    print(
        f"Target {target!r}, {len(columns)} features, missing sentinel {missing!r}, {fit_rows.sum():,} fit rows, "
        f"{len(split['embargo_expeds'])} embargo expeds, and {len(scored):,} labeled holdout rows."
    )
    print(f"Holdout spans {split['holdout_expeds'][0]} through {split['holdout_expeds'][-1]}.")
    predictors = model_predictions(
        train.loc[fit_rows, [target, *columns]], columns, missing, target
    )
    predictions = {
        name: check_predictions(predictor, holdout)["prediction"]
        for name, predictor in predictors.items()
    }
    ranks = ranked_components(predictions["lightgbm"], predictions["ridge"], holdout[exped])
    blend_candidates = {f"rank-blend-{weight:.2f}": weight for weight in BLEND_WEIGHTS}
    for name, weight in blend_candidates.items():
        predictions[name] = weight * ranks["tree"] + (1 - weight) * ranks["linear"]
    reference, coverage = align_benchmark(benchmark, scored, exped)
    if reference is not None:
        predictions["benchmark"] = reference
    per_exped = evaluate(scored, exped, target, predictions, reference)
    summary = summarize(per_exped, split["holdout_expeds"], score_scale)
    blend_scores = summary.loc[list(blend_candidates), "round_score_proxy"].dropna()
    chosen_weight = blend_candidates[blend_scores.idxmax()] if len(blend_scores) else 0.5
    if not len(blend_scores):
        print("Round-score proxy is unavailable. Packaging the fixed 50/50 rank blend.")
    else:
        print(
            "Round-score proxy includes benchmark-adjusted FIT and UNQ; the INOV difference is unknown."
        )
    pair_correlations = ranks.loc[scored.index].assign(exped=scored[exped]).groupby("exped")
    pair_corr = pd.Series(
        {label: spearman(rows["tree"], rows["linear"]) for label, rows in pair_correlations},
        dtype=float,
    ).mean()
    print(summary.to_string(float_format=lambda x: f"{x:+.4f}", na_rep="—"))
    print(
        f"Mean per-exped LightGBM–Ridge rank correlation {pair_corr:+.4f}; blend weight {chosen_weight:.0%} LightGBM."
    )

    print(f"Refitting on {labeled.sum():,} labeled train rows for deployment.")
    final = model_predictions(train.loc[labeled, [target, *columns]], columns, missing, target)
    final["rank-blend"] = blend_prediction(final, exped, chosen_weight)
    sample = holdout.loc[holdout[exped].isin(split["holdout_expeds"][-2:]), [exped, *columns]]
    folder = Path(tempfile.mkdtemp(prefix="fit-", dir=args.artifacts))
    try:
        models = package(final, sample, exped, folder)
        summary.to_csv(folder / "comparison.csv")
        per_exped.to_csv(folder / "per_exped.csv", index=False)
        manifest = {
            "synthetic": args.smoke_test,
            "primary_target": target,
            "exped_column": exped,
            "feature_columns": columns,
            "library_versions": {name: version(name) for name in LIBRARIES},
            "models": models,
            "holdout": split,
            "score_multiple_constant": score_scale,
            "benchmark_coverage": coverage,
            "lightgbm_ridge_rank_corr": None if pd.isna(pair_corr) else float(pair_corr),
            "blend_lightgbm_weight": chosen_weight,
            "comparison_path": f"{folder.name}/comparison.csv",
            "per_exped_path": f"{folder.name}/per_exped.csv",
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
    days, rows = 180, 24
    features = rng.integers(0, 5, size=(days * rows, 4))
    values = features.astype("float32")
    values[rng.random(values.shape) < 0.12] = -99
    index = pd.Index([f"synthetic-{i:05d}" for i in range(days * rows)], name="id")
    names = ["amber", "cyan", "violet", "jade"]
    train = pd.DataFrame(values, columns=names, index=index)
    train["exped"] = np.repeat([f"exped_{i:04d}" for i in range(days)], rows)
    train["data_type"] = "train"
    train["synthetic_target"] = 0.003 * (features[:, 0] - features[:, 1]) + rng.normal(
        0, 0.007, len(train)
    )
    train.loc[train.index[::79], "synthetic_target"] = np.nan
    benchmark = pd.DataFrame(
        {"reference": features[:, 0] + rng.normal(0, 1, len(train))}, index=index
    )
    benchmark["exped"] = train["exped"]
    benchmark.loc[benchmark.index[-1], "exped"] = "exped_0000"
    schema = {
        "primary_target": "synthetic_target",
        "feature_sets": {"all": names},
        "feature_encoding": {"missing": -99},
    }
    return train.sample(frac=1, random_state=821), benchmark.iloc[::-1], schema, 0.1


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
    parser.add_argument(
        "--smoke-test", action="store_true", help="Exercise the pipeline on synthetic data"
    )
    args = parser.parse_args()
    if args.artifacts is None:
        args.artifacts = Path("artifacts/smoke" if args.smoke_test else "artifacts")

    if args.smoke_test:
        if args.artifacts.resolve() == Path("artifacts").resolve():
            parser.error("Use a separate artifact folder for a synthetic smoke test.")
        print("Synthetic smoke test. All scores and model artifacts are for local verification.")
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
        scale = (settings.get("weights") or {}).get("score_multiple_constant")
        score_scale = float(scale) if scale is not None else None
        if score_scale is not None and (not np.isfinite(score_scale) or score_scale <= 0):
            parser.error("Expected a positive score_multiple_constant in the scoring response.")
        inputs = train, benchmark, schema, score_scale

    try:
        train_and_package(*inputs, args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
