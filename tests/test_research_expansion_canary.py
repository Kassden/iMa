"""The paper canary must exercise feedback, not merely finish its fit budget."""
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_research_expansion_canary import paper_study_fixture_planner


def test_paper_fixture_waits_for_inputs_and_feedback_before_more_fits():
    observations = {}
    config = SimpleNamespace()
    response = {"decision": {"trial_budget": 5}}
    with patch("ima.research_expansion.plan_decision", return_value=response) as original:
        planner = paper_study_fixture_planner(observations)
        evidence = {"decision_id": "D1", "evidence_id": "E1"}
        assert planner(evidence, config) == response
        waiting = planner(evidence, config)
        assert waiting["decision"]["trial_budget"] == 0
        assert waiting["decision"]["review_reason"]
        evidence["paper_research"] = {"eligible_attempts": [{"attempt_id": "a1"}]}
        paper = planner(evidence, config)
        assert paper["decision"]["betting_requests"][0]["attempt_ids"] == ["a1"]
        assert paper["decision"]["trial_budget"] == 0
        assert "betting_requests" not in planner(evidence, config)["decision"]
        assert original.call_count == 1
        evidence["paper_research"]["by_comparison"] = {"same-population": [{"request_id": "canary-paper-study"}]}
        assert planner(evidence, config) == response
        assert observations["feedback_received"] is True
        assert original.call_count == 2
