"""Missing transport is a local startup failure, never an ambiguous paid request."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ima.optimizer import CampaignConfig
from ima.research_runtime_preflight import require_planner_transport


def test_missing_httpx_stops_before_campaign_or_planner_state(tmp_path):
    from ima.research_expansion import run_expansion_campaign
    root = tmp_path / "never-created"
    config = CampaignConfig(campaign_dir=root, policy="agentic", research_policy="expansion_v6",
                            planner_mode="openrouter", model="deepseek/deepseek-v4.1-flash")
    with patch("ima.research_runtime_preflight.import_module", side_effect=ModuleNotFoundError("httpx")):
        with pytest.raises(RuntimeError, match="Planner runtime is incomplete"):
            run_expansion_campaign(config)
    assert not root.exists()


@pytest.mark.parametrize("installed", ["0.27.2", "1.0.0"])
def test_unsupported_transport_version_is_rejected(installed):
    with patch("ima.research_runtime_preflight.import_module", return_value=SimpleNamespace(AsyncClient=lambda: None)), \
            patch("ima.research_runtime_preflight.version", return_value=installed):
        with pytest.raises(RuntimeError, match="qualify the declared httpx"):
            require_planner_transport()


def test_transport_preflight_uses_the_real_consumer_without_network():
    import httpx
    with patch.object(httpx.AsyncClient, "send", side_effect=AssertionError("No HTTP during preflight")):
        result = require_planner_transport()
    assert result["module"] == "httpx" and result["network_request_sent"] is False


def test_fixture_campaign_does_not_require_paid_transport(tmp_path):
    config = CampaignConfig(campaign_dir=tmp_path, policy="agentic", research_policy="expansion_v6",
                            planner_mode="fixture", model="fixture")
    with patch("ima.research_runtime_preflight.import_module", side_effect=AssertionError("not needed")):
        config.validate()
