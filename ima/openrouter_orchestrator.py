"""OpenRouter planner boundary for optimizer proposal generation."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .experiments import ExperimentSpec
from .research_specs import (
    MODEL_PARAMETER_CONTRACTS,
    ResearchProposal,
    validate_research_proposal_batch,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_REASONING_EFFORTS = {
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
}


class OpenRouterError(RuntimeError):
    """Raised when the remote optimizer planner cannot produce usable proposals."""


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    model: str
    service_tier: str | None = None
    timeout_seconds: int = 60
    max_output_tokens: int = 2400
    base_url: str = OPENROUTER_BASE_URL
    provider_endpoint: str | None = None
    reasoning_effort: str | None = None
    absolute_deadline_seconds: int | None = None
    response_observer: Callable[[dict[str, Any]], None] | None = None
    transport_observer: Callable[[dict[str, Any]], None] | None = None

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        service_tier: str | None = None,
        max_output_tokens: int = 2400,
        timeout_seconds: int = 300,
        provider_endpoint: str | None = None,
        reasoning_effort: str | None = None,
    ) -> "OpenRouterConfig":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is required for policy='openrouter'")
        chosen_model = model or os.environ.get("IMA_OPTIMIZER_MODEL")
        if not chosen_model:
            raise OpenRouterError("IMA_OPTIMIZER_MODEL or --model is required for policy='openrouter'")
        return cls(
            api_key=api_key,
            model=chosen_model,
            service_tier=service_tier or os.environ.get("IMA_OPTIMIZER_SERVICE_TIER"),
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            provider_endpoint=(
                provider_endpoint
                or os.environ.get("IMA_OPTIMIZER_PROVIDER_ENDPOINT")
            ),
            reasoning_effort=(
                reasoning_effort
                or os.environ.get("IMA_OPTIMIZER_REASONING_EFFORT")
            ),
        )


def _provider_route(config: OpenRouterConfig) -> dict[str, Any]:
    if not config.provider_endpoint:
        return {}
    return {
        "provider": {
            "order": [config.provider_endpoint],
            "allow_fallbacks": False,
        }
    }


def _service_tier_option(config: OpenRouterConfig) -> dict[str, str]:
    return {"service_tier": config.service_tier} if config.service_tier else {}


def _reasoning_options(config: OpenRouterConfig) -> dict[str, Any]:
    if config.reasoning_effort is None:
        return {}
    if config.reasoning_effort not in OPENROUTER_REASONING_EFFORTS:
        allowed = ", ".join(sorted(OPENROUTER_REASONING_EFFORTS))
        raise OpenRouterError(f"reasoning_effort must be one of: {allowed}")
    return {"reasoning_effort": config.reasoning_effort}


def _post_json(url: str, payload: dict[str, Any], config: OpenRouterConfig) -> dict[str, Any]:
    if config.absolute_deadline_seconds:
        import httpx
        from .openrouter_transport import post_json_bounded
        try:
            return post_json_bounded(url,payload,config)
        except (TimeoutError,httpx.HTTPError,ValueError) as exc:
            raise OpenRouterError(f"OpenRouter bounded request failed: {type(exc).__name__}: {exc}") from exc
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except (OSError, http.client.HTTPException, ValueError) as exc:
        detail = getattr(exc, "reason", str(exc))
        raise OpenRouterError(f"OpenRouter request failed: {detail}") from exc


def _get_json(url: str, config: OpenRouterConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {config.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except (OSError, http.client.HTTPException, ValueError) as exc:
        detail = getattr(exc, "reason", str(exc))
        raise OpenRouterError(f"OpenRouter request failed: {detail}") from exc


def planner_messages(available_specs: list[ExperimentSpec], proposal_count: int) -> list[dict[str, str]]:
    specs = [
        {
            "run_id": spec.run_id,
            "kind": spec.kind,
            "parameters": spec.parameters,
        }
        for spec in available_specs
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are the iMa horse-racing model optimizer planner. "
                "Choose bounded ML experiments only from the supplied existing specs. "
                "Return strict JSON only. Do not use leakage features, race results, dividends, "
                "final odds, live execution, credentials, or promotion decisions."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "select_next_optimizer_trials",
                    "proposal_count": proposal_count,
                    "allowed_changed_surfaces": [
                        "hyperparameters",
                        "feature_schema",
                        "feature_family",
                        "transform",
                        "dataset_window",
                        "model_family",
                        "calibration",
                        "market_blend",
                    ],
                    "available_specs": specs,
                    "output_schema": {
                        "proposals": [
                            {
                                "run_id": "must match one available_specs.run_id",
                                "hypothesis": "short falsifiable reason",
                                "changed_surface": "one allowed_changed_surfaces value",
                            }
                        ]
                    },
                },
                sort_keys=True,
            ),
        },
    ]


def _proposal_payload_from_response(response: dict[str, Any]) -> dict[str, Any]:
    parsed = _json_message_from_response(response)
    if isinstance(parsed, list):
        return {"proposals": parsed}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("proposals"), list):
        raise OpenRouterError("OpenRouter planner JSON must contain a proposals list")
    return parsed


def _json_message_from_response(response: dict[str, Any]) -> Any:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("OpenRouter response did not contain a chat message") from exc
    if not isinstance(content, str):
        raise OpenRouterError(
            "OpenRouter response did not contain text content; the completion may have "
            "exhausted its token budget during reasoning"
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"OpenRouter planner returned non-JSON content: {content[:200]}") from exc
    return parsed


def choose_proposals(
    available_specs: list[ExperimentSpec],
    proposal_count: int,
    config: OpenRouterConfig,
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": planner_messages(available_specs, proposal_count),
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    } | _service_tier_option(config) | _provider_route(config) | _reasoning_options(config)
    response = _post_json(f"{config.base_url}/v1/chat/completions", payload, config)
    parsed = _proposal_payload_from_response(response)
    return {
        "raw_response": response,
        "proposal_payload": parsed,
        "service_tier": response.get("service_tier"),
    }


def agentic_planner_messages(evidence_bundle: dict[str, Any], proposal_count: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the iMa research optimizer planner. Return strict JSON only. "
                "Propose bounded research programs: one structural PipelineRecipe v2 and a small "
                "typed model-parameter search space per program. Optuna will select parameters "
                "within each program; you select hypotheses, targets, features, transforms, models "
                "and budgets from the registered capabilities. Cite the development evidence, "
                "compare only compatible objectives, and avoid transform columns listed as "
                "unavailable in feature_profile. Do not request raw runner rows, "
                "labels, credentials, promotion, same-race final odds as features, or live betting. "
                "The entire proposal, including its explanatory text, is rejected if it contains "
                "the literal token final_odds. For recorded odds prediction, describe the target "
                "as 'recorded odds' and never propose it as an input feature. "
                "When v3_assignment is present, propose exactly its target and model; the "
                "controller rejects other contracts. Every recipe must "
                "obey the supplied target/model/calibration/blend compatibility matrix exactly. "
                "For feature-discovery-v4 proposals, you may choose feature schema, "
                "drop_feature_families, train_window, and registered transforms: "
                "clip_numeric_quantiles, race_relative_rank, race_relative_center, "
                "signed_log1p, or numeric_interaction. Every transform requires a columns "
                "array; numeric_interaction requires exactly two distinct numeric columns. "
                "Propose only columns in the chosen schema with observations in all training folds. "
                "Do not describe feature importance as a manually assigned feature weight. "
                "If the assignment says required_feature_change=true, change at least one "
                "of feature_schema, drop_feature_families, transforms, or train_window "
                "relative to reference_feature_recipe. A narrative claim is not sufficient; "
                "the controller compares the actual recipe and rejects identical feature programs. "
                "When discovery_capabilities is present this is v5: you may additionally create "
                "new historical features with recipe.feature_discovery using its supplied strict schema. "
                "Choose entities, measurements, aggregates, history windows, selection strategy and "
                "selected-column budget. At least one program must use feature_discovery. "
                "Initially return programs covering all B/E1/E2/E3/E4 contracts so the 80/20 dispatcher "
                "can stay occupied; split your chosen trial budget mainly toward B. "
                "On later decisions, respect available_program_slots and replenish the requested "
                "next_required_lane; existing programs may already cover other contracts. "
                "You may return retire_program_ids to stop asking unused trials in unpromising "
                "programs. Running trials finish; completed evidence is never deleted. "
                "Null feature_discovery provides a matched existing-feature control. "
                "Set fixed_parameters=true for controlled fixed-model feature ablations; "
                "otherwise Optuna tunes within your search_space. You may enable domain_history, "
                "sequence_windows, race_relative, or adjusted_speed_residuals in DiscoverySpec. "
                "Adjusted residuals are separately fit within each temporal fold, never on the full archive. "
                "Registered transforms may also reference exact dfs_ feature IDs present in "
                "feature_evidence, provided the same discovery spec generates them. Never guess IDs. "
                "All other transform inputs MUST occur in numeric_columns_by_schema for the "
                "chosen feature_schema. Inspect recent_planner_rejections and correct rejected "
                "columns or definitions; do not repeat an already rejected recipe. "
                "With no completed trials yet, parent_trial_ids may be empty; do not fabricate parents."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "propose_agentic_research_programs",
                    "proposal_count": proposal_count,
                    "v3_assignment": evidence_bundle.get("v3_assignment"),
                    "allowed_changed_axes": [
                        "hyperparameters",
                        "feature_schema",
                        "feature_family",
                        "transform",
                        "dataset_window",
                        "model_family",
                        "calibration",
                        "market_blend",
                        "target",
                    ],
                    "required_recipe_object_shape": {
                        "schema_version": 2,
                        "target": {
                            "kind": "one of: win_probability, ranking_strength, placing_top_k, adjusted_finish_time_or_speed, market_odds_forecast, recorded_final_win_odds",
                            "parameters": {},
                        },
                        "feature_schema": "one of: baseline-v1, benter-rich-v1, notebook-rich-v2",
                        "train_window": "one of: all_history, trailing_3_years",
                        "model": {
                            "kind": "one of: logit, boosted, benter_conditional_logit, pairwise_ranker, lightgbm_lambdarank, catboost_classifier, catboost_regressor, hist_gradient_regressor, ridge_regressor",
                            "parameters": {},
                        },
                        "drop_feature_families": [],
                        "transforms": [],
                        "calibration": {"kind": "temperature", "parameters": {}},
                        "blend": {"kind": "market_softmax", "parameters": {}},
                        "seed": 42,
                        "feature_discovery": (
                            {"entities": ["horse"], "measurements": ["speed_mps"], "aggregates": ["count", "mean", "std"], "windows_days": [90, 365], "max_depth": 1, "max_selected": 16, "selection": "mutual_information"}
                            if evidence_bundle.get("discovery_capabilities") else None
                        ),
                    },
                    "discovery_capabilities": evidence_bundle.get("discovery_capabilities"),
                    "requested_trial_budget": evidence_bundle.get("requested_trial_budget"),
                    "next_required_lane": evidence_bundle.get("next_required_lane"),
                    "v5_budget_rule": "When requested_trial_budget is supplied, the sum of all proposal max_trials MUST NOT exceed it. Prioritize next_required_lane if specified. Use Benter for most work, but include executable experimental contracts. Unused programs can carry over.",
                    "target_recipe_compatibility": {
                        "win_probability": {
                            "model": "one of: logit, boosted, benter_conditional_logit",
                            "calibration": "one of: temperature, none",
                            "blend": "one of: market_softmax, none",
                        },
                        "placing_top_k": {
                            "model": "one of: logit, boosted, catboost_classifier",
                            "calibration": "none",
                            "blend": "none",
                        },
                        "ranking_strength": {
                            "model": "one of: pairwise_ranker, lightgbm_lambdarank",
                            "calibration": "none",
                            "blend": "none",
                        },
                        "adjusted_finish_time_or_speed": {
                            "model": "one of: hist_gradient_regressor, ridge_regressor",
                            "calibration": "none",
                            "blend": "none",
                        },
                        "market_odds_forecast": {
                            "model": "one of: hist_gradient_regressor, ridge_regressor",
                            "calibration": "none",
                            "blend": "none",
                        },
                        "recorded_final_win_odds": {
                            "model": "catboost_regressor",
                            "calibration": "none",
                            "blend": "none",
                        },
                    },
                    "allowed_model_parameters": MODEL_PARAMETER_CONTRACTS,
                    "search_space_rules": {
                        "parameters": "Only parameters registered for the recipe model kind.",
                        "numeric": {"kind": "float or int", "low": "valid bound", "high": "valid bound", "log": False},
                        "categorical": {"kind": "categorical", "choices": ["at least two valid values"]},
                        "budget": (
                            "Choose a positive total trial count for this program from the evidence. "
                            "This is not the batch size or concurrency limit. The controller runs "
                            "the program across as many bounded batches as needed; request more "
                            "trials only when the hypothesis and search space justify them."
                        ),
                    },
                    "evidence_bundle": evidence_bundle,
                    "output_schema": {
                        "retire_program_ids": [],
                        "proposals": [{
                            "proposal_id": "stable id",
                            "parent_trial_ids": ["actual trial ids from evidence"],
                            "evidence_ids": ["evidence_bundle.evidence_id"],
                            "hypothesis": "short falsifiable reason",
                            "changed_axes": ["one or more allowed axes"],
                            "recipe": "PipelineRecipe v2 object",
                            "fixed_parameters": False,
                            "search_space": {
                                "model_parameter_name": {
                                    "kind": "float, int, or categorical",
                                    "low": "numeric lower bound when applicable",
                                    "high": "numeric upper bound when applicable",
                                    "choices": "valid choices when categorical",
                                }
                            },
                            "expected_observation": "what should improve",
                            "falsification_rule": "what result rejects the idea",
                            "max_trials": "positive integer total for this program, chosen from evidence",
                        }]
                    },
                },
                sort_keys=True,
            ),
        },
    ]


def research_proposals_from_response(response: dict[str, Any]) -> list[ResearchProposal]:
    payload = _proposal_payload_from_response(response)
    proposals = [ResearchProposal.model_validate(row) for row in payload["proposals"]]
    validate_research_proposal_batch(proposals)
    return proposals


def _validated_research_proposals(
    response: dict[str, Any],
) -> tuple[list[ResearchProposal], list[dict[str, Any]]]:
    payload = _proposal_payload_from_response(response)
    valid: list[ResearchProposal] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(payload["proposals"]):
        try:
            valid.append(ResearchProposal.model_validate(row))
        except Exception as exc:
            proposal_id = row.get("proposal_id") if isinstance(row, dict) else None
            rejected.append({
                "index": index,
                "proposal_id": proposal_id,
                "reason": f"{type(exc).__name__}: {exc}"[:2000],
            })
    if valid:
        validate_research_proposal_batch(valid)
    return valid, rejected


def choose_cycle_trial_budget(
    evidence_bundle: dict[str, Any],
    max_trials: int,
    config: OpenRouterConfig,
) -> dict[str, Any]:
    """Let the planner choose this cycle's work; max_trials is only a ceiling."""
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    summary = {
        "evidence_id": evidence_bundle["evidence_id"],
        "completed_trial_count": len(evidence_bundle.get("completed_trials", [])),
        "best_by_target": {
            key: [{
                "objective_name": row.get("objective_name"),
                "objective_value": row.get("objective_value"),
            } for row in values]
            for key, values in evidence_bundle.get("best_by_target", {}).items()
        },
        "program_outcomes": evidence_bundle.get("program_outcomes", [])[-12:],
        "recent_failures": evidence_bundle.get("recent_failures", [])[-5:],
        "global_champions": evidence_bundle.get("global_champions",{}),
        "next_required_lane": evidence_bundle.get("next_required_lane"),
    }
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You choose the number of training trials for the next iMa research cycle. "
                    "The ceiling is not a target. Choose a count justified by development "
                    "evidence, program uncertainty, and diminishing returns. Optuna and the "
                    "controller will schedule the work; concurrency is independent. "
                    "Do not use holdout results. Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "task": "choose_cycle_trial_budget",
                    "ceiling": max_trials,
                    "evidence": summary,
                    "output_schema": {
                        "evidence_id": summary["evidence_id"],
                        "trial_budget": "integer from 1 through ceiling",
                        "rationale": "brief evidence-based reason for this count",
                    },
                }, sort_keys=True),
            },
        ],
        "temperature": 0,
        "max_tokens": config.max_output_tokens,
        "response_format": {"type": "json_object"},
    } | _service_tier_option(config) | _provider_route(config) | _reasoning_options(config)
    usage_rows: list[dict[str, Any]] = []
    for attempt in range(2):
        response = _post_json(f"{config.base_url}/v1/chat/completions", payload, config)
        usage_rows.append(normalize_openrouter_usage(response))
        try:
            result = _json_message_from_response(response)
            budget = result["trial_budget"]
            if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= max_trials:
                raise ValueError(f"trial_budget must be an integer from 1 through {max_trials}")
            if result.get("evidence_id") != summary["evidence_id"]:
                raise ValueError("stale evidence_id")
            rationale = result.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("rationale must be non-empty")
        except (AttributeError, KeyError, TypeError, ValueError, OpenRouterError) as exc:
            if attempt == 1:
                raise OpenRouterError(f"OpenRouter returned an invalid cycle budget: {exc}") from exc
            payload["messages"].append({
                "role": "user",
                "content": f"The budget response was invalid: {exc}. Correct the JSON within the ceiling.",
            })
            continue
        return {
            "trial_budget": budget,
            "ceiling": max_trials,
            "rationale": rationale.strip(),
            "evidence_id": summary["evidence_id"],
            "usage": _sum_planner_usage(usage_rows),
            "retry_count": attempt,
        }
    raise AssertionError("cycle budget retry loop did not return")


def _sum_planner_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usage = {
        key: sum(row[key] for row in rows)
        if all(row[key] is not None for row in rows) else None
        for key in ("input_tokens", "output_tokens", "total_tokens", "total_cost_usd")
    }
    usage["cost_status"] = "reported" if usage["total_cost_usd"] is not None else "unavailable"
    return usage


def choose_research_proposals(
    evidence_bundle: dict[str, Any],
    proposal_count: int,
    config: OpenRouterConfig,
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": agentic_planner_messages(evidence_bundle, proposal_count),
        "temperature": 0,
        "max_tokens": config.max_output_tokens,
        "response_format": {"type": "json_object"},
    } | _service_tier_option(config) | _provider_route(config) | _reasoning_options(config)
    usage_rows: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    for attempt in range(2):
        response = _post_json(f"{config.base_url}/v1/chat/completions", payload, config)
        usage_rows.append(normalize_openrouter_usage(response))
        try:
            proposals, rejected = _validated_research_proposals(response)
            detail = rejected[0]["reason"] if rejected else "no proposals"
        except Exception as exc:
            proposals, rejected = [], []
            detail = f"{type(exc).__name__}: {exc}"
        all_rejected.extend(rejected)
        if proposals:
            return {
                "raw_response": response,
                "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                "rejected_proposals": all_rejected,
                "service_tier": response.get("service_tier"),
                "usage": _sum_planner_usage(usage_rows),
                "retry_count": attempt,
            }
        if attempt == 0:
            payload["messages"].append({
                "role": "user",
                "content": (
                    "Your last proposal was rejected by validation: " + detail[:1000] +
                    ". Return a corrected proposal with the same assigned target/model. "
                    "Do not use forbidden literal terms in any text or input features."
                ),
            })
    raise OpenRouterError(f"OpenRouter planner returned no valid research proposals: {detail}")


def choose_research_decision(evidence_bundle: dict[str, Any], limits: dict[str, int],
                             config: OpenRouterConfig) -> dict[str, Any]:
    """V6 structural decisions and trial allocations in one costed API call."""
    from .research_expansion import PlannerDecision
    from .dataset_specs import DatasetRequest
    from .feature_definitions import FeatureDefinition
    from .research_specs import PerformanceDistributionSpec
    from .research_planner_context import compact_planner_evidence
    memo = evidence_bundle.get("capabilities", {}).get("research_memo")
    messages = [{"role": "system", "content": (
        "You direct horse-racing ML research. Return one strict JSON decision matching the schema. "
        "Read the entire capabilities.research_memo when supplied; acknowledge its SHA256 in "
        "research_memo_sha256 and use its relevant findings in hypotheses or review_reason. "
        "Select hypotheses, feature formulas, dataset requests, transforms, primary models, typed "
        "pipeline graphs, calibration and trial allocations. Optuna only tunes your frozen model "
        "search spaces. fixed_parameters=true requires max_trials=1 and no search_space. "
        "Numeric search bounds with log=true must be strictly positive: low>0 and high>0. "
        "For kind=int, low and high must be actual integers, not floats or booleans. "
        "Logarithmic integer search is supported natively; keep kind=int with log=true rather "
        "than converting integer parameters to float searches. "
        "Choose a budget from 0 through trial_ceiling, not necessarily the ceiling "
        "or a multiple of worker count. New program budgets plus extensions plus explicitly "
        "unallocated trials MUST equal trial_budget: sum(programs[].max_trials) + "
        "sum(extensions.values()) + unallocated_trials = trial_budget. "
        "Include reasons for unallocated work or review. "
        "Full worker queues do not forbid proposing future programs or reviewing/retiring work. "
        "Pending backlog limits are distinct from workers. Do not request unavailable source data. "
        "Use actual registered predictor metadata and generated feature IDs, not guessed columns. "
        "Registered columns outside the chosen baseline schema must be declared in "
        "recipe.extra_numeric_features before transforms reference them. "
        "Feature definitions are typed numeric ASTs, never executable Python. All learned processing "
        "and stacking is chronological, training-only or forward OOF. Do not use current-race "
        "outcomes, final odds, market probabilities or targets as fundamental input features. "
        "It is fine to say result in explanatory text. Fundamental win log loss is primary; market "
        "blend is a separately calibrated reported endpoint. Classical Benter-only ancestry is "
        "80% of dispatched outer trials; 20% experiments may include boosted, ranking, place, "
        "recorded odds, speed/time, conditional-variance probit or composed models. You decide which "
        "experimental programs; there is no forced E1/E2/E3/E4 rotation. Ensure both lanes have "
        "available programs if their evidence queue is empty. One complex graph trial is valid. "
        "On a fresh campaign with no completed trials, include an inexpensive reference program "
        "in each lane using existing features without discovery, formulas or composed graphs. "
        "These establish matched controls and lane readiness; other programs remain free to explore. "
        "Use schema_version=3 recipes. Use exactly the provided evidence_id in decision and proposal "
        "evidence_ids. The sole valid parent_trial_ids are attempt_id values in the current "
        "campaign's completed_trial_index. references are historical context only, not valid "
        "parents unless their IDs also appear in that completed_trial_index. If the index is "
        "empty, every proposal must use empty parent_trial_ids. Examine latest compatible champions and "
        "negative outcomes. Prior dataset scores are historical context, not comparable champions. "
        "You may select betting_requests independently of trial_budget, including a zero-trial paper-only "
        "decision. Use only paper_research.eligible_attempts from current completed win-probability trials "
        "with identical comparison populations; references are never paper inputs. Paper actions may mix "
        "normalized win probabilities and research Plackett-Luce exotic fair prices or explicitly hypothetical "
        "scenario payouts/Kelly sizing. They do not submit wagers, read arbitrary paths, or make extra paid "
        "planner calls. Choose up to four actions per decision, within eight pending and one active action; "
        "choose fewer when useful. Do not fabricate market quotes, realized profits or independent validation. "
        "No live code mutation, money wagering, credentials or arbitrary sources. Future data or "
        "confirmation races are inaccessible. A failed proposal is not a successful empty decision."
    )}, {"role": "user", "content": json.dumps(compact_planner_evidence({
        "task": "research_decision_v6", "limits": limits,
        "required_identity": {"decision_id": evidence_bundle["decision_id"],
                              "evidence_id": evidence_bundle["evidence_id"],
                              "research_memo_sha256": memo["sha256"] if memo else None},
        "output_budget": {"max_completion_tokens": config.max_output_tokens,
                          "reasoning_shares_budget": True,
                          "format": "Concise JSON; omit optional default/null fields; keep explanations short. Copy required_identity exactly."},
        "decision_schema": PlannerDecision.model_json_schema(),
        "feature_definition_schema": FeatureDefinition.model_json_schema(),
        "dataset_request_schema": DatasetRequest.model_json_schema(),
        "performance_distribution_schema": PerformanceDistributionSpec.model_json_schema(),
        "graph_example": {
            "graph_id": "benter-boosted-pool", "primary_node_id": "benter",
            "fundamental_node_id": "pool", "output_node_id": "pool",
            "date_column": "date", "n_splits": 3, "min_train_dates": 2,
            "nodes": [
                {"node_id": "benter", "kind": "estimator", "parameters": {
                    "model_kind": "benter_conditional_logit"}},
                {"node_id": "boosted", "kind": "estimator", "parameters": {
                    "model_kind": "boosted", "model_parameters": {"max_iter": 100}}},
                {"node_id": "pool", "kind": "weighted_probability_pool",
                 "inputs": ["benter", "boosted"], "parameters": {"weights": [0.6, 0.4]}},
            ],
        },
        "graph_contract": {
            "stages": ["feature_view", "estimator", "race_normalize", "calibrate",
                       "weighted_probability_pool", "log_probability_pool", "meta_estimator",
                       "market_blend", "probabilistic_adapter", "forward_oof_predict", "rank_distribution"],
            "learned_downstream_fit_scope": "forward_oof",
            "unsupported": ["arbitrary Python", "external fitted model references",
                            "learned covariance", "arbitrary transform nodes"],
            "outer_model": "Must match primary_node_id estimator model_kind; tuning binds that node",
            "outer_adapters": "Set recipe.calibration.kind=none and recipe.blend.kind=none; compose these stages explicitly inside the graph",
            "market_output": "Output contracts belong to nodes[].output, never the graph root. Set nodes[].output.market=true only for actual market ancestry; fundamental_node_id must have no market ancestor",
        },
        "model_parameter_contracts": MODEL_PARAMETER_CONTRACTS,
        "evidence": evidence_bundle,
    }), default=str, separators=(",", ":"))}]
    payload = {"model": config.model, "messages": messages, "temperature": .2,
               "max_tokens": config.max_output_tokens, "response_format": {"type": "json_object"}}
    payload |= _service_tier_option(config) | _provider_route(config) | _reasoning_options(config)
    usage_rows = []
    started = time.monotonic()
    for attempt in range(2):
        response = _post_json(f"{config.base_url}/v1/chat/completions", payload, config)
        usage_rows.append(normalize_openrouter_usage(response))
        try:
            raw = _json_message_from_response(response)
            decision = PlannerDecision.model_validate(raw)
            if memo and decision.research_memo_sha256 != memo["sha256"]:
                raise ValueError("Research memo checksum acknowledgement missing or mismatched")
            if decision.decision_id != evidence_bundle["decision_id"] or decision.evidence_id != evidence_bundle["evidence_id"]:
                raise ValueError("Decision identity or evidence watermark mismatch")
            if decision.trial_budget > limits["trial_ceiling"] or len(decision.programs) > limits["max_new_programs"]:
                raise ValueError("Decision exceeds independent operator limits")
            return {"decision": decision.model_dump(mode="json"),
                    "usage": _sum_planner_usage(usage_rows), "raw_response": response,
                    "planning_wall_seconds": time.monotonic()-started,
                    "repair_count": attempt}
        except Exception as exc:
            if attempt == 1:
                raise OpenRouterError(f"Invalid V6 decision: {exc}") from exc
            payload["messages"].append({"role": "user", "content":
                                        f"Decision validation failed: {str(exc)[:2000]}. Correct the JSON. "
                                        "Preserve evidence/decision IDs and exact budget reconciliation."})
    raise AssertionError("Decision retry loop did not return")


def normalize_openrouter_usage(response: dict[str, Any]) -> dict[str, Any]:
    """Return stable token/cost fields without estimating unreported spend."""
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _optional_non_negative_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    output_tokens = _optional_non_negative_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _optional_non_negative_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    total_cost = _optional_non_negative_float(
        usage.get("cost", usage.get("total_cost"))
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "cost_status": "reported" if total_cost is not None else "unavailable",
    }


def _optional_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_non_negative_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def submit_proposal_batch(
    available_specs: list[ExperimentSpec],
    proposal_count: int,
    config: OpenRouterConfig,
    custom_id: str = "optimizer-proposals-0001",
) -> dict[str, Any]:
    request_body = {
        "messages": planner_messages(available_specs, proposal_count),
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    } | _service_tier_option(config) | _provider_route(config) | _reasoning_options(config)
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": config.model,
        "requests": [{"custom_id": custom_id, "body": request_body}],
    }
    return _post_json(f"{config.base_url}/v1/batches", payload, config)


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def get_batch(batch_id: str, config: OpenRouterConfig) -> dict[str, Any]:
    if not batch_id:
        raise OpenRouterError("batch_id is required")
    return _get_json(f"{config.base_url}/v1/batches/{batch_id}", config)


def batch_is_terminal(batch: dict[str, Any]) -> bool:
    return str(batch.get("status")) in TERMINAL_BATCH_STATUSES
