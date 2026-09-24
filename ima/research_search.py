"""Persistent Optuna-backed recipe search for agentic optimizer campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from .feature_sets import feature_families_for_schema
from .research_specs import PipelineRecipe


@dataclass(frozen=True)
class RecipeSuggestion:
    trial_id: str
    trial_number: int
    recipe: PipelineRecipe
    hypothesis: str
    changed_axes: tuple[str, ...]

    def serializable(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_number": self.trial_number,
            "recipe": self.recipe.canonical_payload(),
            "recipe_hash": self.recipe.recipe_hash(),
            "hypothesis": self.hypothesis,
            "changed_axes": list(self.changed_axes),
        }


class RecipeSearchController:
    """Owns Optuna ask/tell and recipe persistence for one campaign."""

    def __init__(
        self,
        campaign_dir: Path,
        *,
        study_name: str = "ima-research-v2",
        seed: int = 42,
        search_space_version: str = "recipe-space-v1",
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.study_name = study_name
        self.seed = seed
        self.search_space_version = search_space_version
        self.search_dir = self.campaign_dir / "search"
        self.search_dir.mkdir(parents=True, exist_ok=True)
        storage = JournalStorage(JournalFileBackend(str(self.search_dir / "optuna-journal.log")))
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=TPESampler(seed=seed, constant_liar=True),
            direction="minimize",
            load_if_exists=True,
        )

    def ask(self, count: int) -> list[RecipeSuggestion]:
        if count <= 0:
            raise ValueError("count must be positive")
        suggestions: list[RecipeSuggestion] = []
        seen = self._known_recipe_hashes()
        attempts = 0
        while len(suggestions) < count and attempts < count * 20:
            attempts += 1
            trial = self.study.ask()
            recipe, hypothesis, changed_axes = self._recipe_for_trial(trial)
            recipe_hash = recipe.recipe_hash()
            trial.set_user_attr("recipe", recipe.canonical_payload())
            trial.set_user_attr("recipe_hash", recipe_hash)
            trial.set_user_attr("hypothesis", hypothesis)
            trial.set_user_attr("changed_axes", list(changed_axes))
            trial.set_user_attr("search_space_version", self.search_space_version)
            if recipe_hash in seen:
                self.study.tell(trial, state=TrialState.PRUNED)
                continue
            seen.add(recipe_hash)
            suggestions.append(RecipeSuggestion(
                trial_id=f"recipe-trial-{trial.number:06d}",
                trial_number=trial.number,
                recipe=recipe,
                hypothesis=hypothesis,
                changed_axes=changed_axes,
            ))
        if len(suggestions) < count:
            raise RuntimeError(f"Only produced {len(suggestions)} unique recipes from {attempts} attempts")
        return suggestions

    def tell(self, trial_number: int, value: float, metrics: dict[str, Any] | None = None) -> None:
        if metrics is not None:
            trial = self.study.trials[trial_number]
            self.study._storage.set_trial_user_attr(trial._trial_id, "metrics", metrics)
        self.study.tell(trial_number, float(value))

    def snapshot(self) -> dict[str, Any]:
        trials = self.study.get_trials(deepcopy=False)
        return {
            "study_name": self.study_name,
            "search_space_version": self.search_space_version,
            "trials": len(trials),
            "completed": sum(trial.state == TrialState.COMPLETE for trial in trials),
            "running": sum(trial.state == TrialState.RUNNING for trial in trials),
            "pruned": sum(trial.state == TrialState.PRUNED for trial in trials),
            "recipe_hashes": sorted(
                str(trial.user_attrs["recipe_hash"])
                for trial in trials
                if "recipe_hash" in trial.user_attrs
            ),
        }

    def _known_recipe_hashes(self) -> set[str]:
        return {
            str(trial.user_attrs["recipe_hash"])
            for trial in self.study.get_trials(deepcopy=False)
            if "recipe_hash" in trial.user_attrs and trial.state != TrialState.PRUNED
        }

    def _recipe_for_trial(self, trial: optuna.Trial) -> tuple[PipelineRecipe, str, tuple[str, ...]]:
        seeds = _seed_recipes()
        if trial.number < len(seeds):
            recipe, hypothesis, changed_axes = seeds[trial.number]
            return recipe, hypothesis, changed_axes
        feature_schema = trial.suggest_categorical(
            "feature_schema",
            ["baseline-v1", "benter-rich-v1", "notebook-rich-v2"],
        )
        model_kind = trial.suggest_categorical("model_kind", ["logit", "boosted"])
        train_window = trial.suggest_categorical("train_window", ["all_history", "trailing_3_years"])
        drop_family = trial.suggest_categorical(
            "drop_family",
            ["none", "source_quality", "preferences", "past_performance", "current_condition"],
        )
        if drop_family not in feature_families_for_schema(str(feature_schema)):
            drop_family = "none"
        transform_kind = trial.suggest_categorical(
            "transform_kind",
            ["none", "clip_rating", "rank_rating"],
        )
        parameters: dict[str, Any]
        if model_kind == "logit":
            parameters = {
                "C": trial.suggest_float("logit_c", 0.003, 80.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "max_iter": 1200,
            }
        else:
            parameters = {
                "learning_rate": trial.suggest_float("boost_learning_rate", 0.015, 0.12, log=True),
                "max_iter": trial.suggest_int("boost_iterations", 80, 260),
                "max_leaf_nodes": trial.suggest_categorical("boost_leaves", [7, 15, 31, 63]),
                "l2_regularization": trial.suggest_float("boost_l2", 0.0, 10.0),
            }
        transforms = ()
        if transform_kind == "clip_rating":
            transforms = ({
                "kind": "clip_numeric_quantiles",
                "parameters": {"columns": ["horse_rating"], "lower": 0.01, "upper": 0.99},
            },)
        elif transform_kind == "rank_rating":
            transforms = ({
                "kind": "race_relative_rank",
                "parameters": {"columns": ["horse_rating"]},
            },)
        drop_feature_families = () if drop_family == "none" else (drop_family,)
        recipe = PipelineRecipe(
            feature_schema=feature_schema,
            drop_feature_families=drop_feature_families,
            transforms=transforms,
            train_window=train_window,
            model={"kind": model_kind, "parameters": parameters},
        )
        changed_axes = tuple(
            axis for axis, active in (
                ("feature_schema", feature_schema != "baseline-v1"),
                ("feature_family", bool(drop_feature_families)),
                ("transform", bool(transforms)),
                ("dataset_window", train_window != "all_history"),
                ("model_family", model_kind != "logit"),
                ("hyperparameters", True),
            ) if active
        )
        return (
            recipe,
            f"Optuna {self.search_space_version} explores {', '.join(changed_axes)}.",
            changed_axes,
        )


def _seed_recipes() -> tuple[tuple[PipelineRecipe, str, tuple[str, ...]], ...]:
    return (
        (
            PipelineRecipe(model={"kind": "logit", "parameters": {"C": 0.5, "class_weight": "balanced"}}),
            "Control recipe: baseline schema logistic classifier.",
            ("hyperparameters",),
        ),
        (
            PipelineRecipe(
                feature_schema="benter-rich-v1",
                model={"kind": "boosted", "parameters": {"learning_rate": 0.06, "max_leaf_nodes": 31}},
            ),
            "Anchor recipe: rich features with boosted classifier.",
            ("feature_schema", "model_family"),
        ),
        (
            PipelineRecipe(
                feature_schema="notebook-rich-v2",
                drop_feature_families=("preferences",),
                model={"kind": "logit", "parameters": {"C": 0.1, "class_weight": "balanced"}},
            ),
            "Ablation recipe: remove preferences from notebook-rich schema.",
            ("feature_schema", "feature_family"),
        ),
        (
            PipelineRecipe(
                train_window="trailing_3_years",
                model={"kind": "logit", "parameters": {"C": 1.0, "class_weight": None}},
            ),
            "Window recipe: test recent-history estimator training.",
            ("dataset_window", "hyperparameters"),
        ),
        (
            PipelineRecipe(
                transforms=({
                    "kind": "clip_numeric_quantiles",
                    "parameters": {"columns": ["horse_rating"], "lower": 0.01, "upper": 0.99},
                },),
                model={"kind": "boosted", "parameters": {"learning_rate": 0.04, "max_leaf_nodes": 15}},
            ),
            "Transform recipe: train-fitted clipping before boosted model.",
            ("transform", "model_family"),
        ),
        (
            PipelineRecipe(
                transforms=({
                    "kind": "race_relative_rank",
                    "parameters": {"columns": ["horse_rating"]},
                },),
                model={"kind": "logit", "parameters": {"C": 2.0, "class_weight": "balanced"}},
            ),
            "Transform recipe: add within-race rating rank.",
            ("transform",),
        ),
    )


def recipe_suggestions(
    campaign_dir: Path,
    *,
    count: int,
    study_name: str = "ima-research-v2",
) -> list[dict[str, Any]]:
    controller = RecipeSearchController(campaign_dir, study_name=study_name)
    return [suggestion.serializable() for suggestion in controller.ask(count)]
