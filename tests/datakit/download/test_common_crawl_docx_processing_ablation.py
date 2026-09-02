# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from experiments.datakit.common_crawl_docx_processing_ablation import validated_fetched_input_path


def test_validated_fetched_input_path_accepts_profile_fetch_root(tmp_path: Path) -> None:
    fetched = tmp_path / "fetched_deadbeef"
    data = fetched / "data"
    data.mkdir(parents=True)
    (data / "part-00000.parquet").touch()

    assert validated_fetched_input_path(str(fetched)) == str(fetched)
