"""Persistent Optuna-backed recipe search for agentic optimizer campaigns."""

from __future__ import annotations

import json
import hashlib
import os
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna.trial import TrialState
except Exception:  # pragma: no cover - minimal remote canary without research deps.
    optuna = None
    TPESampler = None
    JournalStorage = None
    JournalFileBackend = None
    TrialState = None

from .feature_sets import feature_families_for_schema
from .research_specs import PipelineRecipe, ResearchProposal, SearchDimension


@dataclass(frozen=True)
class RecipeSuggestion:
    trial_id: str
    trial_number: int
    recipe: PipelineRecipe
    hypothesis: str
    changed_axes: tuple[str, ...]
    program_id: str | None = None
    proposal_id: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_number": self.trial_number,
            "recipe": self.recipe.canonical_payload(),
            "recipe_hash": self.recipe.recipe_hash(),
            "hypothesis": self.hypothesis,
            "changed_axes": list(self.changed_axes),
            "program_id": self.program_id,
            "proposal_id": self.proposal_id,
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
        self.study = None
        if optuna is not None:
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
        if self.study is None:
            return self._fallback_ask(count)
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

    def reserve_recipe(
        self,
        recipe: PipelineRecipe,
        hypothesis: str,
        changed_axes: tuple[str, ...],
    ) -> RecipeSuggestion:
        """Attach a validated planner recipe to an Optuna trial for ask/tell."""
        if self.study is None:
            raise RuntimeError("Optuna is required for executable research campaigns")
        recipe_hash = recipe.recipe_hash()
        if recipe_hash in self._known_recipe_hashes():
            raise ValueError(f"Recipe already reserved: {recipe_hash}")
        trial = self.study.ask()
        trial.set_user_attr("recipe", recipe.canonical_payload())
        trial.set_user_attr("recipe_hash", recipe_hash)
        trial.set_user_attr("hypothesis", hypothesis)
        trial.set_user_attr("changed_axes", list(changed_axes))
        trial.set_user_attr("search_space_version", self.search_space_version)
        trial.set_user_attr("source", "planner")
        return RecipeSuggestion(
            trial_id=f"recipe-trial-{trial.number:06d}",
            trial_number=trial.number,
            recipe=recipe,
            hypothesis=hypothesis,
            changed_axes=changed_axes,
        )

    def tell(self, trial_number: int, value: float, metrics: dict[str, Any] | None = None) -> None:
        if self.study is None:
            return
        if metrics is not None:
            trial = self.study.trials[trial_number]
            self.study._storage.set_trial_user_attr(trial._trial_id, "metrics", metrics)
        self.study.tell(trial_number, float(value))

    def tell_failed(self, trial_number: int) -> None:
        if self.study is None:
            raise RuntimeError("Optuna is required for executable research campaigns")
        trial = self.study.trials[trial_number]
        if trial.state == TrialState.RUNNING:
            self.study.tell(trial_number, state=TrialState.FAIL)

    def trial_state(self, trial_number: int) -> str:
        if self.study is None:
            raise RuntimeError("Optuna is required for executable research campaigns")
        return self.study.trials[trial_number].state.name.lower()

    def snapshot(self) -> dict[str, Any]:
        if self.study is None:
            rows = self._fallback_rows()
            return {
                "study_name": self.study_name,
                "search_space_version": self.search_space_version,
                "trials": len(rows),
                "completed": 0,
                "running": len(rows),
                "pruned": 0,
                "recipe_hashes": sorted(row["recipe_hash"] for row in rows),
            }
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
        if self.study is None:
            return {row["recipe_hash"] for row in self._fallback_rows()}
        return {
            str(trial.user_attrs["recipe_hash"])
            for trial in self.study.get_trials(deepcopy=False)
            if "recipe_hash" in trial.user_attrs and trial.state != TrialState.PRUNED
        }

    def _recipe_for_trial(self, trial) -> tuple[PipelineRecipe, str, tuple[str, ...]]:
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

    def _fallback_ask(self, count: int) -> list[RecipeSuggestion]:
        rows = self._fallback_rows()
        seen = {row["recipe_hash"] for row in rows}
        suggestions: list[RecipeSuggestion] = []
        for index, (recipe, hypothesis, changed_axes) in enumerate(_seed_recipes()):
            recipe_hash = recipe.recipe_hash()
            if recipe_hash in seen:
                continue
            suggestion = RecipeSuggestion(
                trial_id=f"recipe-trial-{len(rows) + len(suggestions):06d}",
                trial_number=len(rows) + len(suggestions),
                recipe=recipe,
                hypothesis=hypothesis,
                changed_axes=changed_axes,
            )
            suggestions.append(suggestion)
            if len(suggestions) >= count:
                break
        if len(suggestions) < count:
            raise RuntimeError("Fallback recipe search exhausted seeded recipes")
        path = self.search_dir / "fallback-recipes.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for suggestion in suggestions:
                handle.write(json.dumps(suggestion.serializable(), sort_keys=True) + "\n")
        return suggestions

    def _fallback_rows(self) -> list[dict[str, Any]]:
        path = self.search_dir / "fallback-recipes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ProgramSearchController:
    """A planner-owned set of bounded, comparable Optuna studies."""

    def __init__(self, campaign_dir: Path, *, seed: int = 42) -> None:
        if optuna is None:
            raise RuntimeError("Executable research programs require Optuna")
        self.campaign_dir = Path(campaign_dir)
        self.search_dir = self.campaign_dir / "search"
        self.search_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.search_dir / "programs.jsonl"
        self.storage = JournalStorage(
            JournalFileBackend(str(self.search_dir / "program-journal.log"))
        )
        self.seed = seed
        self.programs: dict[str, ResearchProposal] = {}
        self.studies: dict[str, Any] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._open(row["program_id"], ResearchProposal.model_validate(row["proposal"]))

    def _open(self, program_id: str, proposal: ResearchProposal) -> None:
        self.programs[program_id] = proposal
        self.studies[program_id] = optuna.create_study(
            study_name=f"ima-program-{program_id}",
            storage=self.storage,
            sampler=TPESampler(seed=self.seed, constant_liar=True),
            direction="minimize",
            load_if_exists=True,
        )

    def register(self, proposal: ResearchProposal) -> str:
        payload = {
            "recipe": proposal.recipe.canonical_payload(),
            "search_space": {k: v.model_dump(mode="json") for k, v in proposal.search_space.items()},
            "evidence_ids": proposal.evidence_ids,
            "target_kind": proposal.recipe.target.kind,
        }
        program_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        if program_id in self.programs:
            return program_id
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "program_id": program_id,
                "proposal": proposal.model_dump(mode="json"),
            }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._open(program_id, proposal)
        return program_id

    def bootstrap(self, batch_size: int) -> list[str]:
        if self.programs:
            return []
        seeds = _seed_recipes()[:3]
        budget = min(32, max(1, math.ceil(batch_size / len(seeds))))
        return [self.register(ResearchProposal(
            proposal_id=f"bootstrap-{index}",
            hypothesis=hypothesis,
            changed_axes=axes,
            recipe=recipe,
            expected_observation="Establish a comparable development baseline.",
            falsification_rule="Retire this direction if it cannot match the market baseline.",
            max_trials=budget,
        )) for index, (recipe, hypothesis, axes) in enumerate(seeds)]

    def has_capacity(self) -> bool:
        return any(
            len(self.studies[program_id].get_trials(deepcopy=False)) < proposal.max_trials
            for program_id, proposal in self.programs.items()
        )

    def ask(self, count: int) -> list[RecipeSuggestion]:
        if count <= 0:
            raise ValueError("count must be positive")
        suggestions: list[RecipeSuggestion] = []
        seen = {
            str(trial.user_attrs["recipe_hash"])
            for study in self.studies.values()
            for trial in study.get_trials(deepcopy=False)
            if "recipe_hash" in trial.user_attrs
        }
        attempts = 0
        while len(suggestions) < count and self.has_capacity() and attempts < count * 20:
            for program_id, proposal in self.programs.items():
                if len(suggestions) >= count:
                    break
                study = self.studies[program_id]
                if len(study.get_trials(deepcopy=False)) >= proposal.max_trials:
                    continue
                attempts += 1
                trial = study.ask()
                space = proposal.search_space or _default_space(proposal.recipe.model.kind)
                parameters = dict(proposal.recipe.model.parameters)
                for name, dimension in space.items():
                    if dimension.kind == "float":
                        value = trial.suggest_float(name, float(dimension.low), float(dimension.high), log=dimension.log)
                    elif dimension.kind == "int":
                        value = trial.suggest_int(name, int(dimension.low), int(dimension.high))
                    else:
                        value = trial.suggest_categorical(name, list(dimension.choices))
                    parameters[name] = value
                try:
                    recipe = proposal.recipe.model_copy(update={
                        "model": proposal.recipe.model.model_copy(update={"parameters": parameters})
                    })
                    recipe = PipelineRecipe.model_validate(recipe.model_dump(mode="json"))
                except Exception:
                    study.tell(trial, state=TrialState.FAIL)
                    continue
                recipe_hash = recipe.recipe_hash()
                trial.set_user_attr("recipe", recipe.canonical_payload())
                trial.set_user_attr("recipe_hash", recipe_hash)
                trial.set_user_attr("program_id", program_id)
                if recipe_hash in seen:
                    study.tell(trial, state=TrialState.PRUNED)
                    continue
                seen.add(recipe_hash)
                suggestions.append(RecipeSuggestion(
                    trial_id=f"{program_id}-{trial.number:06d}",
                    trial_number=trial.number,
                    recipe=recipe,
                    hypothesis=proposal.hypothesis,
                    changed_axes=proposal.changed_axes,
                    program_id=program_id,
                    proposal_id=proposal.proposal_id,
                ))
            if attempts >= count * 20:
                break
        return suggestions

    def trial_state(self, program_id: str, trial_number: int) -> str:
        return self.studies[program_id].trials[trial_number].state.name.lower()

    def tell(
        self, program_id: str, trial_number: int, value: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        study = self.studies[program_id]
        if metrics is not None:
            trial = study.trials[trial_number]
            study._storage.set_trial_user_attr(trial._trial_id, "metrics", metrics)
        study.tell(trial_number, float(value))

    def tell_failed(self, program_id: str, trial_number: int) -> None:
        study = self.studies[program_id]
        if study.trials[trial_number].state == TrialState.RUNNING:
            study.tell(trial_number, state=TrialState.FAIL)

    def snapshot(self) -> dict[str, Any]:
        trials = [trial for study in self.studies.values() for trial in study.get_trials(deepcopy=False)]
        return {
            "search_space_version": "program-space-v1",
            "program_count": len(self.programs),
            "programs": {
                program_id: {
                    "proposal_id": proposal.proposal_id,
                    "target_kind": proposal.recipe.target.kind,
                    "model_kind": proposal.recipe.model.kind,
                    "budget": proposal.max_trials,
                    "trials": len(self.studies[program_id].trials),
                    "completed": sum(t.state == TrialState.COMPLETE for t in self.studies[program_id].trials),
                }
                for program_id, proposal in self.programs.items()
            },
            "trials": len(trials),
            "completed": sum(t.state == TrialState.COMPLETE for t in trials),
            "running": sum(t.state == TrialState.RUNNING for t in trials),
            "pruned": sum(t.state == TrialState.PRUNED for t in trials),
            "recipe_hashes": sorted(
                str(t.user_attrs["recipe_hash"]) for t in trials if "recipe_hash" in t.user_attrs
            ),
        }


def _default_space(model_kind: str) -> dict[str, SearchDimension]:
    if model_kind == "logit":
        return {"C": SearchDimension(kind="float", low=0.003, high=80.0, log=True)}
    if model_kind == "ridge_regressor":
        return {"alpha": SearchDimension(kind="float", low=0.001, high=100.0, log=True)}
    return {
        "learning_rate": SearchDimension(kind="float", low=0.015, high=0.12, log=True),
        "max_iter": SearchDimension(kind="int", low=80, high=260),
        "l2_regularization": SearchDimension(kind="float", low=0.0, high=10.0),
    }


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
