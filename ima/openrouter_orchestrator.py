"""OpenRouter planner boundary for optimizer proposal generation."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

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
    service_tier: str = "flex"
    timeout_seconds: int = 60
    max_output_tokens: int = 2400
    base_url: str = OPENROUTER_BASE_URL
    provider_endpoint: str | None = None
    reasoning_effort: str | None = None

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
            service_tier=service_tier or os.environ.get("IMA_OPTIMIZER_SERVICE_TIER") or "flex",
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


def _reasoning_options(config: OpenRouterConfig) -> dict[str, Any]:
    if config.reasoning_effort is None:
        return {}
    if config.reasoning_effort not in OPENROUTER_REASONING_EFFORTS:
        allowed = ", ".join(sorted(OPENROUTER_REASONING_EFFORTS))
        raise OpenRouterError(f"reasoning_effort must be one of: {allowed}")
    return {"reasoning_effort": config.reasoning_effort}


def _post_json(url: str, payload: dict[str, Any], config: OpenRouterConfig) -> dict[str, Any]:
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
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
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
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
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
    if isinstance(parsed, list):
        return {"proposals": parsed}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("proposals"), list):
        raise OpenRouterError("OpenRouter planner JSON must contain a proposals list")
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
        "service_tier": config.service_tier,
    } | _provider_route(config) | _reasoning_options(config)
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
                "labels, credentials, promotion, final odds or live betting. Every recipe must "
                "obey the supplied target/model/calibration/blend compatibility matrix exactly."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "propose_agentic_research_programs",
                    "proposal_count": proposal_count,
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
                            "kind": "one of: win_probability, ranking_strength, placing_top_k, adjusted_finish_time_or_speed, market_odds_forecast",
                            "parameters": {},
                        },
                        "feature_schema": "one of: baseline-v1, benter-rich-v1, notebook-rich-v2",
                        "train_window": "one of: all_history, trailing_3_years",
                        "model": {
                            "kind": "one of: logit, boosted, pairwise_ranker, hist_gradient_regressor, ridge_regressor",
                            "parameters": {},
                        },
                        "drop_feature_families": [],
                        "transforms": [],
                        "calibration": {"kind": "temperature", "parameters": {}},
                        "blend": {"kind": "market_softmax", "parameters": {}},
                        "seed": 42,
                    },
                    "target_recipe_compatibility": {
                        "win_probability": {
                            "model": "one of: logit, boosted",
                            "calibration": "one of: temperature, none",
                            "blend": "one of: market_softmax, none",
                        },
                        "placing_top_k": {
                            "model": "one of: logit, boosted",
                            "calibration": "none",
                            "blend": "none",
                        },
                        "ranking_strength": {
                            "model": "pairwise_ranker",
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
                    },
                    "allowed_model_parameters": MODEL_PARAMETER_CONTRACTS,
                    "search_space_rules": {
                        "parameters": "Only parameters registered for the recipe model kind.",
                        "numeric": {"kind": "float or int", "low": "valid bound", "high": "valid bound", "log": False},
                        "categorical": {"kind": "categorical", "choices": ["at least two valid values"]},
                        "budget": "3 to 8 trials per program; keep the batch small and falsifiable",
                    },
                    "evidence_bundle": evidence_bundle,
                    "output_schema": {
                        "proposals": [{
                            "proposal_id": "stable id",
                            "parent_trial_ids": ["actual trial ids from evidence"],
                            "evidence_ids": ["evidence_bundle.evidence_id"],
                            "hypothesis": "short falsifiable reason",
                            "changed_axes": ["one or more allowed axes"],
                            "recipe": "PipelineRecipe v2 object",
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
                            "max_trials": "integer from 3 through 8",
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
        "service_tier": config.service_tier,
    } | _provider_route(config) | _reasoning_options(config)
    response = _post_json(f"{config.base_url}/v1/chat/completions", payload, config)
    try:
        proposals, rejected = _validated_research_proposals(response)
    except Exception as exc:
        raise OpenRouterError(
            f"OpenRouter planner returned invalid research proposals: {exc}"
        ) from exc
    if not proposals:
        detail = rejected[0]["reason"] if rejected else "no proposals"
        raise OpenRouterError(
            f"OpenRouter planner returned no valid research proposals: {detail}"
        )
    return {
        "raw_response": response,
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "rejected_proposals": rejected,
        "service_tier": response.get("service_tier"),
        "usage": normalize_openrouter_usage(response),
    }


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
        "service_tier": config.service_tier,
    } | _provider_route(config) | _reasoning_options(config)
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
