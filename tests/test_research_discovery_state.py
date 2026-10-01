import tempfile
import unittest
from pathlib import Path
from ima.research_hypotheses import HypothesisMemory, champion_snapshot
from ima.research_scheduler import ResourceAdmission, ResourceRequest, fair_program_order


class DiscoveryStateTests(unittest.TestCase):
    def test_small_probe_is_not_starved(self):
        reservations=[{"payload":{"program_id":"large"}} for _ in range(26)]
        self.assertEqual(fair_program_order(["large","probe"],reservations),["probe","large"])
        reservations.append({"payload":{"program_id":"probe"}})
        self.assertEqual(fair_program_order(["large","probe"],reservations),["large","probe"])
    def test_idempotent_memory(self):
        with tempfile.TemporaryDirectory() as d:
            m=HypothesisMemory(Path(d)/"h.sqlite")
            m.record("e1","h1","proposed",{"feature":"speed"})
            m.record("e1","h1","proposed",{"feature":"speed"})
            self.assertEqual(len(m.retrieve("speed")),1)
            with self.assertRaises(ValueError): m.record("e1","h1","proposed",{})

    def test_admission_replenishment(self):
        a=ResourceAdmission(2,4,8)
        a.reserve("slow",ResourceRequest(2,4))
        a.reserve("fast",ResourceRequest(2,4))
        self.assertFalse(a.admits(ResourceRequest()))
        a.release("fast")
        a.reserve("next-program",ResourceRequest(2,4))
        self.assertIn("slow",a.active)
        self.assertEqual(a.snapshot()["running_jobs"],2)

    def test_family_and_global_champion(self):
        rows=[]
        for i,(model,score) in enumerate((("benter_conditional_logit",2.18),("boosted",2.16))):
            rows.append({"status":"completed","attempt_id":str(i),"payload":{"recipe":{"target":{"kind":"win_probability"},"model":{"kind":model}}},"result":{"objective_name":"fundamental_log_loss","objective_value":score,"lineage":{"protocol_hash":"p"}}})
        view=champion_snapshot(rows)
        self.assertEqual(len(view["family_champions"]),2)
        self.assertEqual(next(iter(view["global_champions"].values()))["objective_value"],2.16)
