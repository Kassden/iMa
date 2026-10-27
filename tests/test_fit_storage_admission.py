import errno
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ima import research_storage as storage
from ima.research_executor import RecipeExecutionRequest, execute_recipe, prepare_recipe_folds
from ima.research_model_package import load_research_package
from ima.research_specs import PipelineRecipe


def test_partial_csv_failure_preserves_prior_publication(tmp_path, monkeypatch):
    from ima.research_executor import _write_csv_atomic
    authority = _authority(tmp_path, monkeypatch)
    path = tmp_path/'predictions.csv'
    path.write_text('accepted predictions\n')

    def interrupted(frame, stream, **kwargs):
        stream.write('partial output\n')
        raise OSError(errno.ENOSPC, 'injected native CSV full')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', interrupted)
    with pytest.raises(storage.StorageBlocked):
        _write_csv_atomic(path, pd.DataFrame({'p': [.4,.6]}))
    assert path.read_text() == 'accepted predictions\n'
    assert not path.with_suffix('.csv.tmp').exists()
    assert authority.status()['reserved_bytes'] == 0


def _authority(tmp_path, monkeypatch, free=512 * 1024**2):
    authority = storage.StorageAuthority(
        tmp_path / "authority", headroom_bytes=0, headroom_inodes=0,
        statvfs=lambda _: SimpleNamespace(
            f_bavail=free, f_frsize=1, f_favail=100_000, f_bfree=free,
        ),
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    return authority


def _request(root, dataset_path):
    recipe = PipelineRecipe.model_validate({
        "schema_version": 3,
        "model": {"kind": "benter_conditional_logit"},
        "blend": {"kind": "none"},
    })
    return RecipeExecutionRequest(
        "fit-admission", "fit-storage", 0, recipe, dataset_path,
        root / "trials" / "fit-admission",
        {"min_train_races": 8, "calibration_races": 2, "score_races": 2, "max_folds": 1},
    )


def test_native_growth_denial_propagates_without_scientific_result(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("unused fixture\n")
    request = _request(tmp_path, dataset_path)

    def fitted_output(*args, **kwargs):
        with storage.reserved_output_file(request.output_dir / "oversized.bin") as stream:
            stream.write(b"x" * (64 * 1024**2 + 1))

    monkeypatch.setattr("ima.research_executor._load_recipe_features", lambda request: (None, "hash", {}))
    monkeypatch.setattr("ima.research_executor._execute_frame", fitted_output)
    blocked = storage.StorageBlocked("injected reservation growth denial")
    monkeypatch.setattr(authority, "extend", lambda *args, **kwargs: (_ for _ in ()).throw(blocked))

    with pytest.raises(storage.StorageBlocked, match="growth denial"):
        execute_recipe(request)
    assert not (request.output_dir / "result.json").exists()
    assert authority.status()["reserved_bytes"] == 0


def test_fold_output_admission_denies_before_creating_campaign_directory(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch, free=1024)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("unused fixture\n")
    request = _request(tmp_path, dataset_path)

    with pytest.raises(storage.StorageBlocked):
        prepare_recipe_folds(request)
    assert not request.output_dir.exists()


def test_unexpected_io_error_is_recorded_as_scientific_failure(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("unused fixture\n")
    request = _request(tmp_path, dataset_path)
    monkeypatch.setattr(
        "ima.research_executor._load_recipe_features",
        lambda request: (_ for _ in ()).throw(OSError(errno.EIO, "injected device I/O error")),
    )

    result = execute_recipe(request)
    assert result.status == "failed"
    assert result.error.startswith("OSError:")
    assert result.metrics == {}
    assert (request.output_dir / "result.json").is_file()


def test_real_benter_fit_package_and_predictions_read_back_under_admission(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    from tests.test_research_executor_v6 import dataset

    source, frame = dataset(tmp_path, races=18)
    request = _request(tmp_path, source)
    result = execute_recipe(request)
    assert result.status == "completed", result.error

    package = load_research_package(Path(result.artifacts["package"]))
    predictions = pd.read_csv(result.artifacts["predictions"])
    score = frame.loc[frame.race_id.isin(predictions.race_id)].reset_index(drop=True)
    np.testing.assert_allclose(
        package.predict(score), predictions.model_probability, rtol=1e-10, atol=1e-12,
    )


def test_parquet_metadata_estimate_accounts_for_encoded_numeric_width(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = pd.DataFrame({f"numeric_{index}": np.arange(20_000, dtype=np.float64) % 3
                          for index in range(48)})
    path = tmp_path / "highly-encoded.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path, use_dictionary=True, compression="zstd")

    from ima.research_dataset_capacity import dataset_storage_estimate

    estimate = dataset_storage_estimate(path)
    assert estimate >= frame.shape[0] * frame.shape[1] * 9
    assert estimate > path.stat().st_size
