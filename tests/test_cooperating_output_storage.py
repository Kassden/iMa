"""Input RAM width must not reserve output disk; actual growth remains guarded."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ima import research_storage as storage
from ima.performance_probit import _read_warmstart, _write_warmstart
from ima.research_dataset_capacity import cooperating_output_reservation, dataset_storage_estimate
from ima.research_executor import execute_recipe
from ima.research_expansion import _prepare_program
from tests.test_fit_storage_admission import _authority, _request


def test_encoded_input_larger_than_headroom_can_prepare_and_fit(tmp_path, monkeypatch):
    from tests.test_research_executor_v6 import dataset
    from ima.research_model_package import load_research_package

    source, frame = dataset(tmp_path, races=600)
    # Irrelevant compressible columns reproduce decoded input over-reservation
    # without increasing the model's predictor contract.
    frame = pd.read_parquet(source) if source.suffix == '.parquet' else pd.read_csv(source)
    padding = pd.DataFrame({f'unused_padding_{index}': 'x' for index in range(512)}, index=frame.index)
    frame = pd.concat([frame, padding], axis=1)
    source = tmp_path / 'wide-input.parquet'
    frame.to_parquet(source, index=False)
    required_before = 6 * dataset_storage_estimate(source)
    available = 256 * 1024**2
    assert required_before > available
    assert cooperating_output_reservation(source) == 64 * 1024**2
    authority = _authority(tmp_path, monkeypatch, free=available)
    request = _request(tmp_path, source)
    prepared = _prepare_program(source, request.recipe, hashlib.sha256(source.read_bytes()).hexdigest(),
                                tmp_path / 'programs')
    assert prepared['status'] == 'base_features_ready'
    result = execute_recipe(request)
    assert result.status == 'completed', result.error
    predictions = pd.read_csv(result.artifacts['predictions'])
    score = frame.loc[frame.race_id.isin(predictions.race_id)].reset_index(drop=True)
    np.testing.assert_allclose(load_research_package(Path(result.artifacts['package'])).predict(score),
                               predictions.model_probability, atol=1e-12, rtol=1e-10)
    assert required_before > cooperating_output_reservation(source)
    assert authority.status()['reserved_bytes'] == 0


def test_warmstart_denial_precedes_directory_or_temp_creation(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch, free=0)
    path = tmp_path / 'new-cache' / 'coefficients.json'
    with pytest.raises(storage.StorageBlocked):
        _write_warmstart(path, 'key', np.array([1., 2.]))
    assert not path.parent.exists()
    assert authority.status()['reserved_bytes'] == 0


def test_warmstart_write_reload_and_denied_growth_preserve_prior(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    path = tmp_path / 'cache' / 'coefficients.json'
    _write_warmstart(path, 'key', np.array([1., 2.]))
    np.testing.assert_array_equal(_read_warmstart(path, 'key', 2), [1., 2.])
    before = path.read_bytes()

    def blocked(*args, **kwargs):
        raise storage.StorageBlocked('injected warmstart growth denied')

    monkeypatch.setattr(authority, 'extend', blocked)
    # Replace this module's serializer, not the shared json module also used by
    # the real storage authority's durable state writer.
    monkeypatch.setattr('ima.performance_probit.json', SimpleNamespace(
        dumps=json.dumps, dump=lambda record, stream, **kwargs: stream.write('x' * (1024**2))))
    with pytest.raises(storage.StorageBlocked, match='growth denied'):
        _write_warmstart(path, 'key', np.array([3., 4.]))
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]
    assert authority.status()['reserved_bytes'] == 0


def test_invalid_delta_preflight_still_precedes_output_admission(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    source = tmp_path / 'features.delta.json'
    source.write_text('{"storage_format":"unknown"}')
    request = _request(tmp_path, source)
    with pytest.raises(ValueError, match='Unknown feature descriptor'):
        execute_recipe(request)
    assert not request.output_dir.exists()
