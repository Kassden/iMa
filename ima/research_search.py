"""Persistent Optuna-backed recipe search for agentic optimizer campaigns."""

from __future__ import annotations

import json
import hashlib
import os
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
        self.study = None
        if optuna is not None:
            # Keep the optional-dependency fallback import-free. The adapter
            # must run before JournalFileBackend creates either path component.
            from .research_journal import admitted_journal_backend

            storage = JournalStorage(admitted_journal_backend(self.search_dir / "optuna-journal.log"))
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=TPESampler(seed=seed, constant_liar=True),
                direction="minimize",
                load_if_exists=True,
            )
        else:
            if os.environ.get("IMA_STORAGE_ROOT"):
                from .research_controller import _prepare_campaign_directory
                _prepare_campaign_directory(self.search_dir)
            else:
                self.search_dir.mkdir(parents=True, exist_ok=True)

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
        from .research_controller import _append_jsonl

        path = self.search_dir / "fallback-recipes.jsonl"
        for suggestion in suggestions:
            _append_jsonl(path, suggestion.serializable())
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
        self.index_path = self.search_dir / "programs.jsonl"
        from .research_journal import admitted_journal_backend

        self.storage = JournalStorage(
            admitted_journal_backend(self.search_dir / "program-journal.log")
        )
        from .research_controller import _prepare_campaign_directory

        _prepare_campaign_directory(self.search_dir)
        self.seed = seed
        self.programs: dict[str, ResearchProposal] = {}
        self.studies: dict[str, Any] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._open(row["program_id"], ResearchProposal.model_validate(row["proposal"]))
        self.budget_path = self.search_dir / "budget-overrides.json"
        if self.budget_path.exists():
            for program_id, budget in json.loads(self.budget_path.read_text()).items():
                if program_id not in self.programs:
                    raise ValueError("Budget override references an unknown program")
                self.programs[program_id] = self.programs[program_id].model_copy(update={"max_trials": budget})
        self._reconcile_incomplete_asks()

    def set_budget(self, program_id: str, budget: int) -> None:
        """Idempotent absolute budgets; controller persists the allocation first."""
        if program_id not in self.programs or not isinstance(budget, int) or budget < 1:
            raise ValueError("Invalid program budget override")
        if self.programs[program_id].fixed_parameters and budget != 1:
            raise ValueError("Fixed controls cannot be extended")
        old = json.loads(self.budget_path.read_text()) if self.budget_path.exists() else {}
        old[program_id] = budget
        from .research_controller import _write_json_atomic

        _write_json_atomic(self.budget_path, old)
        self.programs[program_id] = self.programs[program_id].model_copy(update={"max_trials": budget})

    def _open(self, program_id: str, proposal: ResearchProposal) -> None:
        self.programs[program_id] = proposal
        self.studies[program_id] = optuna.create_study(
            study_name=f"ima-program-{program_id}",
            storage=self.storage,
            sampler=TPESampler(
                seed=self.seed ^ int(program_id[:8], 16), constant_liar=True
            ),
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
        if proposal.fixed_parameters:
            payload["fixed_parameters"] = True
        program_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        if program_id in self.programs:
            return program_id
        from .research_controller import _append_jsonl

        _append_jsonl(self.index_path, {
            "program_id": program_id,
            "proposal": proposal.model_dump(mode="json"),
        })
        self._open(program_id, proposal)
        return program_id

    def bootstrap(self, batch_size: int) -> list[str]:
        if self.programs:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        seeds = _seed_recipes()[:min(3, batch_size)]
        per_program, extra = divmod(batch_size, len(seeds))
        return [self.register(ResearchProposal(
            proposal_id=f"bootstrap-{index}",
            hypothesis=hypothesis,
            changed_axes=axes,
            recipe=recipe,
            expected_observation="Establish a comparable development baseline.",
            falsification_rule="Retire this direction if it cannot match the market baseline.",
            max_trials=per_program + (index < extra),
        )) for index, (recipe, hypothesis, axes) in enumerate(seeds)]

    def has_capacity(self) -> bool:
        return self.remaining_capacity() > 0

    def remaining_capacity(self) -> int:
        return sum(
            max(0, proposal.max_trials - sum(
                trial.state != TrialState.PRUNED
                for trial in self.studies[program_id].get_trials(deepcopy=False)
            ))
            for program_id, proposal in self.programs.items()
        )

    def _reconcile_incomplete_asks(self) -> None:
        """Durably retire native RUNNING trials whose ask envelope never completed."""
        from .research_journal import program_ask_lock

        with program_ask_lock(self.search_dir.resolve() / "program-ask.lock"):
            self._reconcile_incomplete_asks_locked()


    def _reconcile_incomplete_asks_locked(self) -> None:
        for program_id, study in self.studies.items():
            for trial in study.get_trials(deepcopy=True):
                if trial.state != TrialState.RUNNING:
                    continue
                attrs = trial.user_attrs
                complete = (
                    "recipe" in attrs
                    and "recipe_hash" in attrs
                    and attrs.get("program_id") == program_id
                )
                if complete:
                    recipe = PipelineRecipe.model_validate(attrs["recipe"])
                    if attrs["recipe_hash"] != recipe.recipe_hash():
                        raise ValueError(
                            f"Optuna trial {trial.number} has a recipe hash mismatch"
                        )
                if not complete:
                    storage = study._storage
                    if attrs.get("recovered_incomplete") is not True:
                        storage.set_trial_user_attr(
                            trial._trial_id, "recovered_incomplete", True
                        )
                    if attrs.get("reason") != "infrastructure_storage/incomplete_ask":
                        storage.set_trial_user_attr(
                            trial._trial_id, "reason",
                            "infrastructure_storage/incomplete_ask",
                        )
                    # Persist the recovery classification before the durable,
                    # idempotent terminal transition. PRUNED does not consume budget.
                    study.tell(trial.number, state=TrialState.PRUNED)


    def ask(
        self, count: int, *, preferred_program_ids: list[str] | None = None,
        allowed_program_ids: list[str] | None = None,
    ) -> list[RecipeSuggestion]:
        if count <= 0:
            raise ValueError("count must be positive")
        suggestions: list[RecipeSuggestion] = []
        preferred = [program_id for program_id in preferred_program_ids or [] if program_id in self.programs]
        allowed = set(allowed_program_ids) if allowed_program_ids is not None else set(self.programs)
        program_order = [program_id for program_id in preferred if program_id in allowed]
        program_order += [
            program_id for program_id in self.programs
            if program_id in allowed and program_id not in program_order
        ]
        if not program_order:
            return []
        from .research_journal import admitted_program_ask, program_ask_lock

        per_attempt = max(
            65536 + 2 * len(json.dumps(
                proposal.recipe.canonical_payload(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")) + 4096 * len(proposal.search_space)
            for program_id, proposal in self.programs.items()
            if program_id in program_order
        )
        estimate = max(65536, count * 20 * per_attempt)
        with program_ask_lock(self.search_dir.resolve() / "program-ask.lock"):
            self._reconcile_incomplete_asks_locked()
            try:
                with admitted_program_ask(
                    self.search_dir / "program-journal.log", estimated_bytes=estimate
                ):
                    return self._ask_admitted(count, program_order)
            except Exception as primary:
                try:
                    self._reconcile_incomplete_asks_locked()
                except Exception as recovery_error:
                    primary.add_note(
                        "Immediate incomplete-ask recovery also failed: "
                        f"{type(recovery_error).__name__}: {recovery_error}"
                    )
                raise


    def _ask_admitted(
        self, count: int, program_order: list[str]
    ) -> list[RecipeSuggestion]:
        suggestions: list[RecipeSuggestion] = []
        seen: set[str] = set()
        for program_id, study in self.studies.items():
            for trial in study.get_trials(deepcopy=False):
                attrs = trial.user_attrs
                if attrs.get("recovered_incomplete") is True:
                    continue
                if (
                    attrs.get("program_id") != program_id
                    or "recipe" not in attrs
                    or "recipe_hash" not in attrs
                ):
                    continue
                try:
                    recipe_hash = PipelineRecipe.model_validate(attrs["recipe"]).recipe_hash()
                except Exception:
                    continue
                if attrs["recipe_hash"] == recipe_hash:
                    seen.add(recipe_hash)
        attempts = 0
        while len(suggestions) < count and any(
            sum(trial.state != TrialState.PRUNED for trial in self.studies[program_id].get_trials(deepcopy=False))
            < self.programs[program_id].max_trials for program_id in program_order
        ) and attempts < count * 20:
            for program_id in program_order:
                if len(suggestions) >= count:
                    break
                proposal = self.programs[program_id]
                study = self.studies[program_id]
                if sum(trial.state != TrialState.PRUNED for trial in study.get_trials(deepcopy=False)) >= proposal.max_trials:
                    continue
                attempts += 1
                trial = study.ask()
                space = {} if proposal.fixed_parameters else (proposal.search_space or _default_space(proposal.recipe.model.kind))
                parameters = dict(proposal.recipe.model.parameters)
                for name, dimension in space.items():
                    if dimension.kind == "float":
                        value = trial.suggest_float(name, float(dimension.low), float(dimension.high), log=dimension.log)
                    elif dimension.kind == "int":
                        value = trial.suggest_int(name, int(dimension.low), int(dimension.high), log=dimension.log)
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
                    "executable_trials": sum(
                        trial.state != TrialState.PRUNED
                        for trial in self.studies[program_id].trials
                    ),
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
    if model_kind == "gaussian_probit":
        return {"l2": SearchDimension(kind="float", low=.001, high=1., log=True)}
    if model_kind == "benter_conditional_logit":
        return {"l2": SearchDimension(kind="float", low=0.001, high=5.0, log=True)}
    if model_kind == "lightgbm_lambdarank":
        return {"learning_rate": SearchDimension(kind="float", low=0.02, high=0.12, log=True)}
    if model_kind in {"catboost_classifier", "catboost_regressor"}:
        return {"depth": SearchDimension(kind="int", low=3, high=7)}
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
