"""Evidence-bound file handoff for an operator-supervised session orchestrator."""
import hashlib
import json
import re
import time
from pathlib import Path


def read_external_decision(evidence, config):
    from .research_expansion import PlannerDecision

    identifier = evidence['decision_id']
    if not re.fullmatch(r'D\d{6}', identifier):
        raise ValueError('Invalid external decision identifier')
    root = Path(config.campaign_dir).resolve()
    path = root / 'planner-inbox' / (identifier + '.json')
    deadline = time.monotonic() + config.planner_timeout_seconds
    while not path.exists():
        if (root / 'STOP').exists():
            raise RuntimeError('External planning stopped')
        if time.monotonic() >= deadline:
            raise TimeoutError('Waiting for an evidence-bound external planner decision')
        time.sleep(0.2)
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root / 'planner-inbox':
        raise ValueError('External decision must be a regular campaign-local file')
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('External decision exceeds one MiB')
    raw = path.read_bytes()
    envelope = json.loads(raw)
    if envelope.get('model') != config.model or envelope.get('reasoning_effort') != config.planner_reasoning_effort:
        raise ValueError('External orchestrator identity mismatch')
    decision = PlannerDecision.model_validate(envelope['decision'])
    if decision.decision_id != identifier or decision.evidence_id != evidence['evidence_id']:
        raise ValueError('External decision references stale evidence')
    return {'decision': decision.model_dump(mode='json'),
            'orchestrator': {'mode': 'external_session', 'model': envelope['model'],
                             'reasoning_effort': envelope['reasoning_effort'],
                             'decision_sha256': hashlib.sha256(raw).hexdigest()},
            'usage': {'cost_status': 'external_session_not_api', 'total_cost_usd': None}}
