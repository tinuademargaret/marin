# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from docx import Document
from marin.datakit.download.common_crawl_docx import FETCHED_COMMON_CRAWL_DOCX_SCHEMA
from marin.execution.step_spec import StepSpec
from zephyr.dataset import ShardInfo

from experiments.datakit.common_crawl_docx_profile import (
    PROFILE_ROW_SCHEMA,
    ConverterLifecycle,
    DocxExtractionProfileConfig,
    DocxExtractionProfileVariant,
    VariantRole,
    calibrated_shard_targets,
    common_crawl_docx_profile_steps,
    derive_run_metrics,
    main,
    prepare_extraction_profile_corpus,
    profile_extraction_shard,
    profile_summary,
)


def _docx_payload(text: str) -> bytes:
    stream = io.BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(stream)
    return stream.getvalue()


def _fetched_record(index: int) -> dict[str, object]:
    return {
        "payload": f"payload-{index}".encode(),
        "source_id": f"document-{index}",
        "source": "common_crawl",
        "crawl_id": "CC-MAIN-2026-34",
        "url": f"https://example.com/{index}.docx",
        "warc_filename": "crawl.warc.gz",
        "warc_record_offset": index,
        "warc_record_length": 100,
        "warc_date": None,
        "http_status": 200,
        "http_content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "identified_payload_type": None,
        "content_digest": f"sha1:{index}",
        "index_status": 200,
        "index_content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "index_detected_type": None,
        "selection_reason": "declared_mime",
    }


def test_prepare_extraction_profile_corpus_materializes_nonempty_physical_shards(tmp_path: Path) -> None:
    fetched_data = tmp_path / "fetched" / "data"
    fetched_data.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([_fetched_record(index) for index in range(40)], schema=FETCHED_COMMON_CRAWL_DOCX_SCHEMA),
        fetched_data / "part-00000.parquet",
    )

    corpus = prepare_extraction_profile_corpus(
        str(tmp_path / "prepared"),
        fetched_input_path=str(tmp_path / "fetched"),
        target_shards=4,
        max_workers=4,
    )

    files = sorted((tmp_path / "prepared" / "data").glob("*.parquet"))
    row_counts = [pq.read_metadata(path).num_rows for path in files]
    assert len(files) == 4
    assert sum(row_counts) == 40
    assert all(count > 0 for count in row_counts)
    assert corpus.documents == 40


def test_profile_extraction_shard_separates_initialization_read_and_conversion(tmp_path: Path) -> None:
    input_path = tmp_path / "fetched.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"source_id": "document-1", "payload": _docx_payload("Profile this document with Docling.")}]
        ),
        input_path,
    )
    variant = DocxExtractionProfileVariant(
        name="one-shard",
        worker_count=1,
        target_shards=1,
        role=VariantRole.GENERAL,
    )

    rows = list(
        profile_extraction_shard(
            iter([str(input_path)]),
            ShardInfo(shard_idx=0, total_shards=1),
            variant=variant,
            maximum_zip_entries=10_000,
            maximum_uncompressed_bytes=1 << 20,
        )
    )

    document, shard = rows
    assert document["success"] is True
    assert float(document["conversion_wall_seconds"]) > 0
    assert shard["converter_initializations"] == 1
    assert float(shard["converter_initialization_wall_seconds"]) > 0
    assert float(shard["read_wall_seconds"]) > 0
    assert shard["documents_attempted"] == 1
    assert pa.Table.from_pylist(rows, schema=PROFILE_ROW_SCHEMA).num_rows == 2


def test_derive_run_metrics_calculates_tail_utilization_and_terminal_idle() -> None:
    documents = [
        {
            "row_kind": "document",
            "variant": "scaling-2",
            "variant_role": VariantRole.SCALING.value,
            "worker_count": 2,
            "target_shards": 2,
            "lifecycle": ConverterLifecycle.PER_TASK.value,
            "shard_idx": index % 2,
            "worker_id": f"worker-{index % 2}",
            "success": True,
            "payload_bytes": 100,
            "extracted_bytes": 50,
            "conversion_wall_seconds": 100.0 if index == 99 else 1.0,
            "conversion_cpu_seconds": 90.0 if index == 99 else 0.9,
        }
        for index in range(100)
    ]
    shards = [
        {
            "row_kind": "shard",
            "variant": "scaling-2",
            "variant_role": VariantRole.SCALING.value,
            "worker_count": 2,
            "target_shards": 2,
            "lifecycle": ConverterLifecycle.PER_TASK.value,
            "shard_idx": 0,
            "worker_id": "worker-0",
            "converter_initialization_wall_seconds": 1.0,
            "converter_initialization_cpu_seconds": 0.5,
            "converter_initializations": 1,
            "read_wall_seconds": 1.0,
            "shard_wall_seconds": 10.0,
            "shard_cpu_seconds": 9.0,
            "peak_rss_bytes": 1_000,
            "peak_cpu_percent": 100.0,
            "initial_rss_bytes": 500,
            "final_rss_bytes": 750,
            "started_at": 0.0,
            "finished_at": 10.0,
        },
        {
            "row_kind": "shard",
            "variant": "scaling-2",
            "variant_role": VariantRole.SCALING.value,
            "worker_count": 2,
            "target_shards": 2,
            "lifecycle": ConverterLifecycle.PER_TASK.value,
            "shard_idx": 1,
            "worker_id": "worker-1",
            "converter_initialization_wall_seconds": 1.0,
            "converter_initialization_cpu_seconds": 0.5,
            "converter_initializations": 1,
            "read_wall_seconds": 1.0,
            "shard_wall_seconds": 5.0,
            "shard_cpu_seconds": 4.0,
            "peak_rss_bytes": 2_000,
            "peak_cpu_percent": 90.0,
            "initial_rss_bytes": 1_000,
            "final_rss_bytes": 1_500,
            "started_at": 0.0,
            "finished_at": 5.0,
        },
    ]

    metrics = derive_run_metrics([*documents, *shards])

    assert metrics.top_one_percent_work_share == pytest.approx(100 / 199)
    assert metrics.worker_utilization == pytest.approx(0.75)
    assert metrics.terminal_idle_fraction == pytest.approx(0.25)
    assert metrics.observed_workers == 2
    assert metrics.peak_concurrency == 2
    assert metrics.documents_per_initialization == 50
    assert metrics.peak_rss_bytes == 2_000
    assert profile_summary([metrics])["scaling"] == []


def test_profile_cli_dry_run_builds_fixed_corpus_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetched_path = tmp_path / "fetched"
    data_path = fetched_path / "data"
    data_path.mkdir(parents=True)
    (data_path / "part-00000.parquet").touch()
    (data_path / "part-00001.parquet").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "common_crawl_docx_profile",
            "--fetched-input-path",
            str(fetched_path),
            "--output-path",
            str(tmp_path / "profile"),
            "--dry-run",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert '"name": "scaling-1"' in output
    assert '"name": "scaling-16"' in output
    assert '"name": "persistence-cold"' in output


def test_calibrated_shard_targets_respect_requested_duration_and_input_granularity() -> None:
    targets = calibrated_shard_targets(total_shard_seconds=3_600, input_shards=20, target_minutes=(1, 5, 30))

    assert targets == {"1m": 20, "5m": 12, "30m": 2}


def test_profile_steps_resolve_json_serializable_output_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    fetched = StepSpec(name="fetched", fn=lambda output_path: output_path)
    variants = (
        DocxExtractionProfileVariant(
            name="scaling-1",
            worker_count=1,
            target_shards=4,
            role=VariantRole.SCALING,
        ),
    )

    preparation, runs, report = common_crawl_docx_profile_steps(
        DocxExtractionProfileConfig(
            name="smoke",
            variants=variants,
            preparation_shards=4,
            maximum_zip_entries=10_000,
            maximum_uncompressed_bytes=1 << 20,
        ),
        fetched=fetched,
    )

    assert preparation.deps == [fetched]
    assert len(runs) == 1
    assert runs[0].deps == [preparation]
    assert runs[0].output_path.startswith(str(tmp_path))
    assert report.output_path.startswith(str(tmp_path))
