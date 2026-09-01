# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys
from functools import partial
from pathlib import Path

import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from docx import Document
from marin.datakit.download.common_crawl_docx import (
    FETCHED_COMMON_CRAWL_DOCX_SCHEMA,
    DoclingDocxExtractor,
    reset_docling_converter,
)
from marin.execution.step_spec import StepSpec
from zephyr.dataset import ShardInfo

from experiments.datakit.common_crawl_docx_profile import (
    PROFILE_ROW_SCHEMA,
    ConverterLifecycle,
    DocxExtractionProfileConfig,
    DocxExtractionProfileVariant,
    ProfileLayout,
    RunnerMode,
    VariantRole,
    common_crawl_docx_profile_steps,
    derive_run_metrics,
    main,
    natural_layout_variants,
    prepare_extraction_profile_corpus,
    profile_extraction_shard,
    profile_summary,
    worker_shard_size_variants,
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
            run_token=str(tmp_path / "one-shard"),
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


def test_profile_extraction_shard_reuses_persistent_process_converter(tmp_path: Path) -> None:
    input_path = tmp_path / "fetched.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"source_id": "document-1", "payload": _docx_payload("Persistent converter.")}]),
        input_path,
    )
    variant = DocxExtractionProfileVariant(
        name="persistent-worker",
        worker_count=1,
        target_shards=2,
        role=VariantRole.PERSISTENCE,
        lifecycle=ConverterLifecycle.PER_WORKER,
        runner=RunnerMode.INLINE,
    )
    reset_docling_converter()
    DoclingDocxExtractor().initialize()
    run_token = str(tmp_path / "persistent-run")

    try:
        first = list(
            profile_extraction_shard(
                iter([str(input_path)]),
                ShardInfo(shard_idx=0, total_shards=2),
                run_token=run_token,
                variant=variant,
                maximum_zip_entries=10_000,
                maximum_uncompressed_bytes=1 << 20,
            )
        )
        second = list(
            profile_extraction_shard(
                iter([str(input_path)]),
                ShardInfo(shard_idx=1, total_shards=2),
                run_token=run_token,
                variant=variant,
                maximum_zip_entries=10_000,
                maximum_uncompressed_bytes=1 << 20,
            )
        )
        assert first[-1]["converter_initializations"] == 1
        assert second[-1]["converter_initializations"] == 0
    finally:
        reset_docling_converter()


def test_profile_extraction_shard_serializes_when_script_module_is_pickled_by_value() -> None:
    module = sys.modules[profile_extraction_shard.__module__]
    variant = DocxExtractionProfileVariant(
        name="scaling-1",
        worker_count=1,
        target_shards=32,
        role=VariantRole.SCALING,
    )
    shard_function = partial(
        profile_extraction_shard,
        run_token="profile-run",
        variant=variant,
        maximum_zip_entries=10_000,
        maximum_uncompressed_bytes=1 << 30,
    )

    cloudpickle.register_pickle_by_value(module)
    try:
        serialized = cloudpickle.dumps(shard_function)
    finally:
        cloudpickle.unregister_pickle_by_value(module)

    assert cloudpickle.loads(serialized).keywords["variant"] == variant


def test_derive_run_metrics_calculates_tail_utilization_and_terminal_idle() -> None:
    documents = [
        {
            "row_kind": "document",
            "variant": "scaling-2",
            "variant_role": VariantRole.SCALING.value,
            "worker_count": 2,
            "target_shards": 2,
            "layout": ProfileLayout.PREPARED.value,
            "lifecycle": ConverterLifecycle.PER_SHARD.value,
            "runner": RunnerMode.SUBPROCESS.value,
            "shard_idx": index % 2,
            "success": index >= 3,
            "error_kind": "InvalidDocxError" if index < 2 else ("DocxExtractionError" if index == 2 else None),
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
            "layout": ProfileLayout.PREPARED.value,
            "lifecycle": ConverterLifecycle.PER_SHARD.value,
            "runner": RunnerMode.SUBPROCESS.value,
            "shard_idx": 0,
            "converter_initialization_wall_seconds": 1.0,
            "converter_initialization_cpu_seconds": 0.5,
            "converter_initializations": 1,
            "read_wall_seconds": 1.0,
            "shard_wall_seconds": 150.0,
            "shard_cpu_seconds": 9.0,
            "peak_rss_bytes": 1_000,
            "peak_cpu_percent": 100.0,
            "initial_rss_bytes": 500,
            "final_rss_bytes": 750,
            "started_at": 0.0,
            "finished_at": 150.0,
        },
        {
            "row_kind": "shard",
            "variant": "scaling-2",
            "variant_role": VariantRole.SCALING.value,
            "worker_count": 2,
            "target_shards": 2,
            "layout": ProfileLayout.PREPARED.value,
            "lifecycle": ConverterLifecycle.PER_SHARD.value,
            "runner": RunnerMode.SUBPROCESS.value,
            "shard_idx": 1,
            "converter_initialization_wall_seconds": 1.0,
            "converter_initialization_cpu_seconds": 0.5,
            "converter_initializations": 1,
            "read_wall_seconds": 1.0,
            "shard_wall_seconds": 75.0,
            "shard_cpu_seconds": 4.0,
            "peak_rss_bytes": 2_000,
            "peak_cpu_percent": 90.0,
            "initial_rss_bytes": 1_000,
            "final_rss_bytes": 1_500,
            "started_at": 0.0,
            "finished_at": 75.0,
        },
    ]

    metrics = derive_run_metrics([*documents, *shards])

    assert metrics.top_one_percent_work_share == pytest.approx(100 / 199)
    assert metrics.worker_utilization == pytest.approx(0.75)
    assert metrics.terminal_idle_fraction == pytest.approx(0.25)
    assert metrics.peak_concurrency == 2
    assert metrics.documents_per_initialization == 50
    assert metrics.peak_rss_bytes == 2_000
    assert metrics.failures == 3
    assert metrics.failure_distribution == {"DocxExtractionError": 1, "InvalidDocxError": 2}
    assert metrics.other_wall_seconds == pytest.approx(22.0)
    assert metrics.other_fraction == pytest.approx(22 / 225)
    summary = profile_summary([metrics])
    assert summary["scaling"] == []
    runs = summary["runs"]
    assert isinstance(runs, list)
    run = runs[0]
    assert isinstance(run, dict)
    assert run["failure_distribution"] == {
        "DocxExtractionError": 1,
        "InvalidDocxError": 2,
    }
    assert run["other_wall_seconds"] == pytest.approx(22.0)


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
    assert '"name": "persistence-worker"' in output
    assert '"name": "natural-16"' in output


def test_profile_cli_can_select_only_prepared_scaling_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetched_path = tmp_path / "fetched"
    data_path = fetched_path / "data"
    data_path.mkdir(parents=True)
    (data_path / "part-00000.parquet").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "common_crawl_docx_profile",
            "--fetched-input-path",
            str(fetched_path),
            "--output-path",
            str(tmp_path / "profile"),
            "--worker-counts",
            "1,4",
            "--target-shards",
            "32",
            "--shard-counts",
            "32",
            "--skip-natural-layout",
            "--skip-persistence",
            "--dry-run",
        ],
    )

    main()

    variants = json.loads(capsys.readouterr().out)
    assert [(variant["name"], variant["target_shards"]) for variant in variants] == [
        ("scaling-1", 32),
        ("scaling-4", 32),
    ]


def test_profile_summary_selects_smallest_shard_meeting_initialization_target() -> None:
    rows = []
    for shard_count, initialization in ((2, 0.08), (4, 0.10), (8, 0.20)):
        variant = DocxExtractionProfileVariant(
            name=f"shard-w2-s{shard_count}",
            worker_count=2,
            target_shards=shard_count,
            role=VariantRole.SHARD_SIZE,
        )
        for shard_idx in range(shard_count):
            rows.append(
                {
                    "row_kind": "document",
                    "variant": variant.name,
                    "variant_role": variant.role.value,
                    "worker_count": 2,
                    "target_shards": shard_count,
                    "layout": variant.layout.value,
                    "lifecycle": variant.lifecycle.value,
                    "runner": variant.runner.value,
                    "shard_idx": shard_idx,
                    "success": True,
                    "payload_bytes": 100,
                    "extracted_bytes": 50,
                    "conversion_wall_seconds": 1.0,
                    "conversion_cpu_seconds": 1.0,
                }
            )
            rows.append(
                {
                    "row_kind": "shard",
                    "variant": variant.name,
                    "variant_role": variant.role.value,
                    "worker_count": 2,
                    "target_shards": shard_count,
                    "layout": variant.layout.value,
                    "lifecycle": variant.lifecycle.value,
                    "runner": variant.runner.value,
                    "shard_idx": shard_idx,
                    "converter_initialization_wall_seconds": initialization / (1 - initialization),
                    "converter_initialization_cpu_seconds": 0.0,
                    "converter_initializations": 1,
                    "read_wall_seconds": 0.0,
                    "shard_wall_seconds": 2.0,
                    "shard_cpu_seconds": 1.0,
                    "peak_rss_bytes": 1_000,
                    "peak_cpu_percent": 100.0,
                    "initial_rss_bytes": 500,
                    "final_rss_bytes": 500,
                    "started_at": float(shard_idx),
                    "finished_at": float(shard_idx + 2),
                }
            )
    metrics = [
        derive_run_metrics([row for row in rows if row["variant"] == variant.name])
        for variant in worker_shard_size_variants(worker_counts=(2,), target_shards=(2, 4, 8))
    ]

    optimal = profile_summary(metrics)["optimal_shards"]

    assert optimal == [
        {
            "workers": 2,
            "variant": "shard-w2-s4",
            "shards": 4,
            "documents_per_shard": 1.0,
            "initialization_fraction": pytest.approx(0.10),
            "meets_initialization_target": True,
        }
    ]


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
        natural_layout_variants((1,))[0],
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
    assert len(runs) == 2
    assert runs[0].deps == [preparation]
    assert runs[1].deps == [fetched]
    assert runs[0].output_path.startswith(str(tmp_path))
    assert report.output_path.startswith(str(tmp_path))
