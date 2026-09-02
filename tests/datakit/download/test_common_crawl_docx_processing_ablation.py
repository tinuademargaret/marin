# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from rigging.filesystem import prefix_join
from zephyr.dataset import GlobSource, resolve_glob

from experiments.datakit.common_crawl_docx_processing_ablation import combined_fetched_input_path, fetched_step_paths


def test_combined_fetched_input_path_discovers_every_batch(tmp_path: Path) -> None:
    batches = (tmp_path / "fetched_aaaa", tmp_path / "fetched_bbbb")
    for batch in batches:
        data = batch / "data"
        data.mkdir(parents=True)
        (data / f"{batch.name}.parquet").touch()

    combined = combined_fetched_input_path(tuple(str(batch) for batch in batches))
    entries = resolve_glob(GlobSource(prefix_join(combined, "data/**/*.parquet")))

    assert {Path(entry.path).name for entry in entries} == {"fetched_aaaa.parquet", "fetched_bbbb.parquet"}


def test_fetched_step_paths_discovers_only_artifact_directories(tmp_path: Path) -> None:
    expected = (tmp_path / "fetched_aaaa", tmp_path / "fetched_bbbb")
    for batch in (*expected, tmp_path / "unrelated"):
        batch.mkdir()
        (batch / ".artifact.json").touch()
        data = batch / "data"
        data.mkdir()
        (data / "part-00000.parquet").touch()
    (tmp_path / "fetched_incomplete").mkdir()

    discovered = fetched_step_paths(str(tmp_path))

    assert discovered == tuple(str(path) for path in expected)
