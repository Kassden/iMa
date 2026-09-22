"""OpenRouter planner boundary for optimizer proposal generation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .experiments import ExperimentSpec


OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class OpenRouterError(RuntimeError):
    """Raised when the remote optimizer planner cannot produce usable proposals."""


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    model: str
    service_tier: str = "flex"
    timeout_seconds: int = 60
    base_url: str = OPENROUTER_BASE_URL

    @classmethod
    def from_env(cls, model: str | None = None, service_tier: str | None = None) -> "OpenRouterConfig":
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
        )


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
    except urllib.error.URLError as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc.reason}") from exc


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
    }
    response = _post_json(f"{config.base_url}/v1/chat/completions", payload, config)
    parsed = _proposal_payload_from_response(response)
    return {
        "raw_response": response,
        "proposal_payload": parsed,
        "service_tier": response.get("service_tier"),
    }


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
    }
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": config.model,
        "requests": [{"custom_id": custom_id, "body": request_body}],
    }
    return _post_json(f"{config.base_url}/beta/batches", payload, config)
