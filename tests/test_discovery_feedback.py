import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from ima.feature_discovery_specs import DiscoverySpec
from ima.feature_program import synthesize
from ima.feature_residuals import AdjustedSpeedHistory
from ima.feature_studies import paired_feature_report
from ima.research_search import ProgramSearchController
from ima.research_controller import _v4_seed_proposals
from scripts.optimize import _load_config
from ima.openrouter_orchestrator import OpenRouterConfig, OpenRouterError, _post_json
from ima.research_hypotheses import reference_champions
from ima.research_store import ResearchLedger


def history():
    rows=[]
    for day in range(80):
        for horse in range(3):
            rows.append(dict(race_id=f"r{day}", horse_no=horse+1, horse_id=f"h{horse}", date=pd.Timestamp("2020-01-01")+pd.Timedelta(days=day*2), distance=1200+(day%2)*400, finish_seconds=70+horse+day/10, actual_weight=120+horse, race_class=3, horse_age=4, draw=horse+1))
    return pd.DataFrame(rows)


class DiscoveryFeedbackTests(unittest.TestCase):
    def test_absolute_api_deadline(self):
        import asyncio
        class SlowClient:
            def __init__(self,**kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            async def post(self,*args,**kwargs): await asyncio.sleep(5)
        config=OpenRouterConfig("test","test",absolute_deadline_seconds=1)
        with patch("httpx.AsyncClient",SlowClient),self.assertRaisesRegex(OpenRouterError,"TimeoutError"):
            _post_json("https://example.invalid",{},config)

    def test_reference_evidence_requires_same_population(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=ResearchLedger(Path(directory)/"ledger.sqlite")
            payload={"dataset_hash":"d","protocol_parameters":{"score_races":2},"recipe":{"target":{"kind":"win_probability"},"model":{"kind":"boosted"}}}
            record=ledger.reserve_attempt("a",payload)
            ledger.complete_attempt(record.attempt_id,{"objective_name":"loss","objective_value":2.1,"lineage":{"protocol_id":"p"}})
            self.assertEqual(len(reference_champions(directory,"d",{"score_races":2})),1)
            self.assertEqual(reference_champions(directory,"other",{"score_races":2}),[])
            self.assertEqual(ledger.snapshot()["completed"],1)
    def test_domain_past_only_and_definitions(self):
        frame=history()
        spec=DiscoverySpec(domain_history=True,windows_days=(90,))
        original,catalog=synthesize(frame,spec)
        changed=frame.copy()
        changed.loc[changed.date>=pd.Timestamp("2020-04-01"),"finish_seconds"]=999
        after,_=synthesize(changed,spec)
        pd.testing.assert_frame_equal(original[frame.date<pd.Timestamp("2020-04-01")],after[frame.date<pd.Timestamp("2020-04-01")])
        self.assertTrue(any(f["expression"]=="distance_conditioned_speed" for f in catalog["catalog"]))

    def test_fold_residual_replay_and_future_invariance(self):
        frame=history()
        train=frame.iloc[:180].copy()
        score=frame.iloc[180:].copy()
        spec=DiscoverySpec(adjusted_speed_residuals=True)
        fitted=AdjustedSpeedHistory.fit(train,spec,score.date.min())
        first=fitted.transform(score)
        changed=score.copy()
        changed.loc[changed.date==changed.date.max(),"finish_seconds"]=999
        again=fitted.transform(changed)
        pd.testing.assert_frame_equal(first[list(fitted.columns)],again[list(fitted.columns)])
        for block in fitted.report["oof_blocks"]:
            self.assertLess(pd.Timestamp(block["fit_through"]),pd.Timestamp(block["predict_from"]))
        import joblib
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"residual.joblib"
            joblib.dump(fitted,path)
            pd.testing.assert_frame_equal(first,joblib.load(path).transform(score))

    def test_paired_comparability_and_verdict(self):
        rows=[dict(fold_id="f1",race_id=f"r{r}",date=f"2020-01-{r+1:02d}",horse_no=h+1,target_win=int(h==0),model_probability=.8 if h==0 else .1) for r in range(8) for h in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"a.csv",Path(directory)/"b.csv"
            frame=pd.DataFrame(rows)
            frame.to_csv(a,index=False)
            frame.model_probability=1/3
            frame.to_csv(b,index=False)
            report=paired_feature_report(a,b,"win_probability")
            self.assertEqual(report["verdict"],"keep")
            frame.iloc[3:].to_csv(b,index=False)
            with self.assertRaisesRegex(ValueError,"Noncomparable"):
                paired_feature_report(a,b,"win_probability")

    def test_budget_alias_and_fixed_control(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.json"
            path.write_text(json.dumps({"max_trials_per_decision":104}))
            self.assertEqual(_load_config(path)["proposal_batch_size"],104)
            search=ProgramSearchController(Path(directory)/"campaign")
            proposal=_v4_seed_proposals()[0].model_copy(update={"fixed_parameters":True,"max_trials":1,"search_space":{}})
            search.register(proposal)
            suggestion=search.ask(1)[0]
            self.assertEqual(suggestion.recipe.model.parameters,proposal.recipe.model.parameters)


if __name__=="__main__": unittest.main()
