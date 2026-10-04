import json
import unittest
from unittest.mock import patch

from ima.openrouter_orchestrator import OpenRouterConfig, choose_research_decision
from ima.research_memo import MEMO_PATH, research_memo_context


class ResearchMemoPromptRulesTests(unittest.TestCase):
    def test_fixed_controls_search_bounds_and_budget_rules_preserve_full_memo(self):
        memo = research_memo_context()
        decision = {"decision_id":"D1","evidence_id":"E1","trial_budget":0,
                    "review_reason":"Review controls and frozen searches",
                    "research_memo_sha256":memo["sha256"]}
        response = {"choices":[{"message":{"content":json.dumps(decision)}}],
                    "usage":{"cost":.001}}
        evidence = {"decision_id":"D1","evidence_id":"E1",
                    "capabilities":{"research_memo":memo}}
        with patch("ima.openrouter_orchestrator._post_json",return_value=response) as post:
            result = choose_research_decision(evidence,{"trial_ceiling":260,"max_new_programs":12},
                                             OpenRouterConfig("unused","fixture/model"))
        self.assertEqual(1,post.call_count)
        messages = post.call_args.args[1]["messages"]
        instructions = messages[0]["content"]
        for rule in ("fixed_parameters=true requires max_trials=1 and no search_space",
                     "log=true must be strictly positive: low>0 and high>0",
                     "kind=int, low and high must be actual integers, not floats or booleans",
                     "Logarithmic integer search is supported natively",
                     "keep kind=int with log=true",
                     "sum(programs[].max_trials) + sum(extensions.values()) + unallocated_trials = trial_budget"):
            with self.subTest(rule=rule):
                self.assertIn(rule,instructions)
        submitted = json.loads(messages[1]["content"])
        transmitted = submitted["evidence"]["capabilities"]["research_memo"]
        self.assertEqual(memo,transmitted)
        self.assertEqual(MEMO_PATH.read_text(),transmitted["content"])
        self.assertEqual(memo["sha256"],result["decision"]["research_memo_sha256"])


if __name__ == "__main__":
    unittest.main()
