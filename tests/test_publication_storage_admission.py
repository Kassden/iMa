import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ima.research_storage import StorageBlocked
from ima.research_publication import deliver_feedback, publish_accepted
from tests.test_fit_publication import accepted_fixture


def test_canonical_result_admission_denial_writes_nothing(tmp_path):
    from tests.test_fit_publication import request_at, write_private_result
    from ima.research_publication import stage_arguments

    request = request_at(tmp_path)
    request.output_dir.mkdir(parents=True)
    args, managed = stage_arguments((request,), request.output_dir / "runtime.json", "a" * 32)
    result = write_private_result(args[0])
    deadline = "2999-01-01T00:00:00+00:00"
    state = {"status": "finished", "task_nonce": "a" * 32, "deadline_at": deadline}

    def deny(path, payload):
        assert Path(path) == request.output_dir / "result.json"
        raise StorageBlocked("publication reservation denied")

    with patch("ima.research_controller._write_json_atomic", side_effect=deny):
        with pytest.raises(StorageBlocked, match="reservation denied"):
            publish_accepted(managed, result, state, 0)

    assert not (request.output_dir / "result.json").exists()


def test_feedback_receipt_admission_denial_preserves_accepted_result(tmp_path):
    request, staged, _, _ = accepted_fixture(tmp_path)
    row = {
        "attempt_id": request.attempt_id,
        "status": "completed",
        "payload": {
            "recipe": request.recipe.model_dump(mode="json"),
            "recipe_hash": request.recipe.recipe_hash(),
            "proposal_id": request.proposal_id,
            "trial_number": request.trial_number,
            "dataset_path": str(request.dataset_path),
            "protocol_parameters": request.protocol_parameters,
            "dataset_hash": request.dataset_hash,
            "code_revision": request.code_revision,
            "environment_hash": request.environment_hash,
        },
    }
    intent = {"attempt_id": request.attempt_id, "task_nonce": staged.managed_episode_nonce}
    (staged.output_dir / "feature-feedback-intent.json").write_text(json.dumps(intent))
    canonical = request.output_dir / "result.json"
    original = canonical.read_bytes()
    delivered = staged.output_dir / "feature-feedback-delivered.json"

    def deny(path, payload):
        assert Path(path) == delivered
        raise StorageBlocked("feedback receipt reservation denied")

    with patch("ima.research_lifecycle.link_feature_outcome", return_value={"registered": True}), \
         patch("ima.research_controller._write_json_atomic", side_effect=deny):
        with pytest.raises(StorageBlocked, match="receipt reservation denied"):
            deliver_feedback(tmp_path, row)

    assert canonical.read_bytes() == original
    assert not delivered.exists()
