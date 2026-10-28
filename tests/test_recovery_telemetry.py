from ima.research_summary import publish_execution_change


def test_idle_heartbeat_is_not_a_new_execution_trace():
    assert publish_execution_change('counts=87,15;waiting',None)
    assert not publish_execution_change('counts=87,15;waiting','counts=87,15;waiting')
    assert publish_execution_change('counts=88,15;training','counts=87,15;waiting')
