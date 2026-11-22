"""Package practice-only rank blends from validated Everesteer research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from itertools import combinations, product
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from everesteer_hackathon.models import LIBRARIES, check_predictions, package, weighted_rank_prediction
from research.research_experiments import atomic_json

SOURCES = {
    "ridge": "wide-ridge",
    "tree": "wide-trees",
    "portfolio": "research-portfolio",
}
GROUP_ROTATION = (
    "ridge-tree",
    "anchored-pair",
    "ridge-tree",
    "anchored-triple",
    "ridge-pair",
    "ridge-tree",
    "anchored-pair",
    "anchored-triple",
    "tree-pair",
)


@dataclass(frozen=True)
class Recipe:
    group: str
    weights: tuple[tuple[str, float], ...]
    merit: float

    @property
    def name(self):
        digest = hashlib.sha256(json.dumps(self.weights).encode()).hexdigest()[:12]
        return f"practice-rank-{digest}"


def load_sources(artifacts):
    manifests, predictors, members = {}, {}, {}
    libraries = {name: version(name) for name in LIBRARIES}
    interpreter = f"{sys.version_info.major}.{sys.version_info.minor}"
    for source, folder in SOURCES.items():
        directory = artifacts / folder
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("provenance") != "walk_forward_research" or manifest.get("synthetic"):
            raise ValueError(f"{folder!r} must describe walk-forward event models.")
        if manifest.get("library_versions") != libraries:
            raise ValueError(f"Use the matching pinned package versions for {folder!r}.")
        manifests[source] = manifest
        order = manifest.get("fallback_order") or list(manifest["models"])
        members[source] = []
        for name in order:
            model = manifest["models"][name]
            if not model.get("prevalidated") or model.get("python_version") != interpreter:
                raise ValueError(f"{name!r} needs a verified artifact for Python {interpreter}.")
            payload = (directory / model["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != model["sha256"]:
                raise ValueError(f"Checksum mismatch for {name!r}.")
            if name in predictors:
                raise ValueError(f"Source alias {name!r} appears in multiple portfolios.")
            predict = cloudpickle.loads(payload)
            if not callable(predict):
                raise ValueError(f"{name!r} does not contain a callable model.")
            predictors[name] = predict
            members[source].append(name)
    targets = {manifest["primary_target"] for manifest in manifests.values()}
    expeds = {manifest.get("exped_column", "exped") for manifest in manifests.values()}
    if len(targets) != 1 or len(expeds) != 1:
        raise ValueError("Source manifests disagree on the graded target or time-unit column.")
    columns = sorted({feature for m in manifests.values() for feature in m["feature_columns"]})
    return predictors, members, targets.pop(), expeds.pop(), columns, libraries


def unit_rank_vectors(values, expeds):
    """A dot product between normalized vectors is mean within-exped rank correlation."""
    groups = values.groupby(expeds, sort=False)
    ranks = groups.rank()
    groups = ranks.groupby(expeds, sort=False)
    centered = ranks - groups.transform("mean")
    norm = (centered**2).groupby(expeds, sort=False).transform("sum").pow(0.5)
    return centered.div(norm.mask(norm.eq(0))).fillna(0).to_numpy() / expeds.nunique() ** 0.5


def make_recipes(members, similarities):
    quality = {
        name: index / max(1, len(names) - 1)
        for names in members.values()
        for index, name in enumerate(names)
    }
    groups = {group: [] for group in dict.fromkeys(GROUP_ROTATION)}

    def add(group, names, weights):
        overlap = [abs(similarities.at[left, right]) for left, right in combinations(names, 2)]
        if max(overlap) >= 0.98:
            return
        merit = sum(quality[name] * weight for name, weight in zip(names, weights, strict=True))
        merit += 0.2 * np.mean(overlap)
        groups[group].append(Recipe(group, tuple(sorted(zip(names, weights, strict=True))), merit))

    for ridge, tree in product(members["ridge"], members["tree"]):
        add("ridge-tree", (ridge, tree), (0.5, 0.5))
    for original, member in product(members["portfolio"], members["ridge"] + members["tree"]):
        add("anchored-pair", (original, member), (0.6, 0.4))
    for original, ridge, tree in product(
        members["portfolio"], members["ridge"][:12], members["tree"][:12]
    ):
        add("anchored-triple", (original, ridge, tree), (0.4, 0.3, 0.3))
    for source in ("ridge", "tree"):
        for left, right in combinations(members[source], 2):
            add(f"{source}-pair", (left, right), (0.6, 0.4))
    return {
        group: sorted(recipes, key=lambda recipe: (recipe.merit, recipe.name))
        for group, recipes in groups.items()
    }


def diversified_recipes(recipes, component_ranks, expeds, max_similarity):
    norms = unit_rank_vectors(component_ranks, expeds).T
    selected = [*norms]
    indexes = {group: iter(choices) for group, choices in recipes.items()}
    while indexes:
        for group in GROUP_ROTATION:
            choices = indexes.get(group)
            if choices is None:
                continue
            for recipe in choices:
                series = sum(component_ranks[name] * weight for name, weight in recipe.weights)
                aligned = unit_rank_vectors(series.to_frame(), expeds)[:, 0]
                if np.max(np.abs(np.asarray(selected) @ aligned)) >= max_similarity:
                    continue
                selected.append(aligned)
                yield recipe
                break
            else:
                del indexes[group]


def save_batch(directory, output, models, recipes, target, exped, columns, libraries, args):
    manifest = {
        "format_version": 2,
        "synthetic": False,
        "practice_only": True,
        "provenance": "practice_blend_experiments",
        "primary_target": target,
        "exped_column": exped,
        "feature_columns": columns,
        "library_versions": libraries,
        "primary_model": next(iter(models)),
        "models": models,
        "composition_selection": "constituent walk-forward order and feature-only rank diversity",
        "practice_sample_max_similarity": args.max_similarity,
        "recipes": recipes,
        "sources": {source: str(args.sources / folder) for source, folder in SOURCES.items()},
    }
    atomic_json(directory / "manifest.json", manifest, output)
    print(f"Ready: {len(models)} practice blends in {directory / 'manifest.json'}.", flush=True)


def run(args):
    predictors, members, target, exped, columns, libraries = load_sources(args.sources)
    features = pd.read_parquet(args.train, columns=[exped, *columns])
    sample_labels = sorted(features[exped].unique())[-args.sample_expeds :]
    if len(sample_labels) < 3:
        raise ValueError("The feature-only comparison requires at least three expeds.")
    sample = features.loc[features[exped].isin(sample_labels)]
    del features
    forecasts = pd.DataFrame(
        {
            name: check_predictions(predict, sample)["prediction"]
            for name, predict in predictors.items()
        }
    )
    ranks = forecasts.groupby(sample[exped], sort=False).rank(pct=True)
    vectors = unit_rank_vectors(forecasts, sample[exped])
    similarity = pd.DataFrame(
        vectors.T @ vectors, index=forecasts.columns, columns=forecasts.columns
    )
    recipes = make_recipes(members, similarity)
    candidates = diversified_recipes(recipes, ranks, sample[exped], args.max_similarity)
    print(
        f"Comparing {sum(map(len, recipes.values()))} recipes across {len(predictors)} source models.",
        flush=True,
    )
    total = 0
    batch = 0
    models, batch_recipes = {}, {}
    for recipe in candidates:
        if not models:
            directory = args.artifacts / f"batch-{batch + 1:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="blend-", dir=directory))
        composition = weighted_rank_prediction(predictors, exped, dict(recipe.weights))
        try:
            artifact = package({recipe.name: composition}, sample, exped, output)[recipe.name]
        except (AssertionError, ValueError) as exc:
            print(f"Skipped {recipe.name}. Package check failed: {exc}", file=sys.stderr)
            (output / f"{recipe.name}.pkl").unlink(missing_ok=True)
            continue
        models[recipe.name] = {**artifact, "practice_only": True, "prevalidated": False}
        batch_recipes[recipe.name] = {"group": recipe.group, "weights": dict(recipe.weights)}
        total += 1
        if len(models) == args.batch_size or total == args.count:
            save_batch(
                directory, output, models, batch_recipes, target, exped, columns, libraries, args
            )
            batch += 1
            models, batch_recipes = {}, {}
        if total == args.count:
            break
    if models:
        save_batch(
            directory, output, models, batch_recipes, target, exped, columns, libraries, args
        )
        batch += 1
    if not total:
        raise ValueError("No distinct recipe passed the packaging checks.")
    print(f"Packaged {total} practice blends across {batch} manifests.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/wide-blends"))
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--count", type=int, default=88)
    parser.add_argument("--batch-size", type=int, default=22)
    parser.add_argument("--sample-expeds", type=int, default=12)
    parser.add_argument("--max-similarity", type=float, default=0.992)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 25 or args.count < 1 or args.sample_expeds < 3:
        parser.error(
            "Use 1–25 items per batch, a positive recipe count, and at least three expeds."
        )
    if not 0 < args.max_similarity < 1:
        parser.error("Maximum similarity must be between zero and one.")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Blend sweep failed. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
