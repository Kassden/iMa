from unittest.mock import patch

from ima.research_resources import GIB
from ima.research_scheduler import ResourceAdmission, ResourceRequest


def test_controller_refreshes_headroom_before_debiting_worker_peak(tmp_path):
    from ima import research_controller as core
    from ima.research_expansion import run_expansion_campaign
    from tests.test_research_expansion_resources import ExpansionResourceIntegrationTests

    case=ExpansionResourceIntegrationTests('test_preparation_initial_slots_cover_cheap_benter_and_experimental')
    case.setUp()
    try:
        config=case.config
        core._write_json_atomic(config.campaign_dir/'STOP',{})
        observed=[]
        original_init=ResourceAdmission.__init__
        original_set=ResourceAdmission.set_headroom
        original_peak=ResourceAdmission.observe_private_peak

        def init_with_worker(self,*args,**kwargs):
            original_init(self,*args,**kwargs)
            self.active['fixture-worker']=ResourceRequest(ram_gib=1)
            observed.append(self)

        def set_headroom(self,**kwargs):
            original_set(self,**kwargs)
            observed.append(('headroom',kwargs['memory_available_gib']))

        def observe_peak(self,key,private_peak_bytes):
            original_peak(self,key,private_peak_bytes)
            observed.append(('peak',key,self.memory_headroom_gib))

        partition={'resident_private_bytes':0,'workers':{
            'fixture-worker':{'private_current_bytes':2*GIB,'private_peak_bytes':3*GIB}}}
        with patch.object(ResourceAdmission,'__init__',init_with_worker), \
             patch.object(ResourceAdmission,'set_headroom',set_headroom), \
             patch.object(ResourceAdmission,'observe_private_peak',observe_peak), \
             patch('ima.research_resources.partition_worker_private',return_value=partition):
            run_expansion_campaign(config)

        events=[item for item in observed if isinstance(item,tuple)]
        first_set=next(index for index,item in enumerate(events) if item[0]=='headroom')
        first_peak=next(index for index,item in enumerate(events) if item[0]=='peak')
        assert first_set < first_peak
        sample=events[first_set][1]
        assert events[first_peak][2] == sample-2
    finally:
        case.doCleanups()
