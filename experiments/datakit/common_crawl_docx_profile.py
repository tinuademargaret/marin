# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile Common Crawl DOCX extraction and render an evidence-based decision report."""

import argparse
import json
import math
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import fsspec
import numpy as np
import plotly.graph_objects as go
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.download.common_crawl_docx import (
    DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
    DEFAULT_MAXIMUM_ZIP_ENTRIES,
    FETCHED_COMMON_CRAWL_DOCX_SCHEMA,
    DoclingDocxExtractor,
    DocxExtractionError,
    EmptyDocxTextError,
    InvalidDocxError,
    reset_docling_converter,
    validated_extracted_docx,
)
from marin.execution.artifact import read_artifact, write_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel, ValidationError
from rigging.filesystem import prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner, SubprocessRunner

PROFILE_SCHEMA_VERSION = 2
DEFAULT_WORKER_COUNTS = (1, 2, 4, 8, 16)
DEFAULT_PREPARATION_SHARDS = 128
_MEBIBYTE = 1 << 20
_PERSISTENT_CONVERTER_RUNS: set[str] = set()
_PERSISTENT_CONVERTER_LOCK = threading.Lock()


class ConverterLifecycle(StrEnum):
    """Converter lifetime used by one profiling run."""

    PER_SHARD = "per_shard"
    PER_WORKER = "per_worker"


class RunnerMode(StrEnum):
    """Isolation boundary used to execute extraction shards."""

    SUBPROCESS = "subprocess"
    INLINE = "inline"


class VariantRole(StrEnum):
    """How a profiling variant participates in report comparisons."""

    GENERAL = "general"
    SCALING = "scaling"
    PERSISTENCE = "persistence"
    SHARD_SIZE = "shard_size"
    NATURAL_LAYOUT = "natural_layout"


class ProfileLayout(StrEnum):
    """Physical payload layout consumed by a profiling run."""

    PREPARED = "prepared"
    NATURAL = "natural"


@dataclass(frozen=True)
class DocxExtractionProfileVariant:
    """One fixed-corpus profiling treatment."""

    name: str
    worker_count: int
    target_shards: int | None
    role: VariantRole
    layout: ProfileLayout = ProfileLayout.PREPARED
    lifecycle: ConverterLifecycle = ConverterLifecycle.PER_SHARD
    runner: RunnerMode = RunnerMode.SUBPROCESS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if self.layout is ProfileLayout.PREPARED and (self.target_shards is None or self.target_shards <= 0):
            raise ValueError("prepared variants require positive target_shards")
        if self.layout is ProfileLayout.NATURAL and self.target_shards is not None:
            raise ValueError("natural-layout variants must not set target_shards")
        if self.lifecycle is ConverterLifecycle.PER_WORKER and self.runner is not RunnerMode.INLINE:
            raise ValueError("a persistent worker converter requires the inline runner")

    @property
    def hash_attrs(self) -> dict[str, str | int | None]:
        """Return the JSON-compatible identity used by Marin step hashing."""
        return {
            "name": self.name,
            "worker_count": self.worker_count,
            "target_shards": self.target_shards,
            "role": self.role.value,
            "layout": self.layout.value,
            "lifecycle": self.lifecycle.value,
            "runner": self.runner.value,
        }


@dataclass(frozen=True)
class DocxExtractionProfileConfig:
    """Fixed input corpus and profiling variants."""

    name: str
    variants: tuple[DocxExtractionProfileVariant, ...]
    preparation_shards: int
    maximum_zip_entries: int
    maximum_uncompressed_bytes: int

    def __post_init__(self) -> None:
        if not self.name or not self.variants:
            raise ValueError("name and variants must not be empty")
        if self.preparation_shards <= 0:
            raise ValueError("preparation_shards must be positive")
        if len({variant.name for variant in self.variants}) != len(self.variants):
            raise ValueError("variant names must be unique")
        prepared_targets = [
            variant.target_shards for variant in self.variants if variant.layout is ProfileLayout.PREPARED
        ]
        if any(target is not None and target > self.preparation_shards for target in prepared_targets):
            raise ValueError("prepared variant target_shards cannot exceed preparation_shards")


class DocxExtractionProfileRun(BaseModel):
    """Artifact produced by one extraction profiling run."""

    metrics_dir: str
    variant: str
    worker_count: int
    target_shards: int | None
    layout: str
    lifecycle: str
    runner: str
    counters: dict[str, int | float]


class DocxExtractionProfileCorpus(BaseModel):
    """Fixed physical payload shards reused by every profiling treatment."""

    data_dir: str
    target_shards: int
    documents: int
    counters: dict[str, int | float]


class DocxExtractionProfileReport(BaseModel):
    """Rendered profiling report paths."""

    markdown_path: str
    dashboard_path: str
    summary_path: str


PROFILE_ROW_SCHEMA = pa.schema(
    [
        pa.field("row_kind", pa.string(), nullable=False),
        pa.field("variant", pa.string(), nullable=False),
        pa.field("variant_role", pa.string(), nullable=False),
        pa.field("worker_count", pa.int32(), nullable=False),
        pa.field("target_shards", pa.int32()),
        pa.field("layout", pa.string(), nullable=False),
        pa.field("lifecycle", pa.string(), nullable=False),
        pa.field("runner", pa.string(), nullable=False),
        pa.field("shard_idx", pa.int32(), nullable=False),
        pa.field("source_file", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("success", pa.bool_()),
        pa.field("error_kind", pa.string()),
        pa.field("payload_bytes", pa.int64()),
        pa.field("extracted_bytes", pa.int64()),
        pa.field("conversion_wall_seconds", pa.float64()),
        pa.field("conversion_cpu_seconds", pa.float64()),
        pa.field("read_wall_seconds", pa.float64()),
        pa.field("converter_initialization_wall_seconds", pa.float64()),
        pa.field("converter_initialization_cpu_seconds", pa.float64()),
        pa.field("converter_initializations", pa.int32()),
        pa.field("documents_attempted", pa.int64()),
        pa.field("documents_succeeded", pa.int64()),
        pa.field("input_files", pa.int32()),
        pa.field("shard_wall_seconds", pa.float64()),
        pa.field("shard_cpu_seconds", pa.float64()),
        pa.field("peak_rss_bytes", pa.int64()),
        pa.field("peak_cpu_percent", pa.float64()),
        pa.field("initial_rss_bytes", pa.int64()),
        pa.field("final_rss_bytes", pa.int64()),
        pa.field("started_at", pa.float64()),
        pa.field("finished_at", pa.float64()),
    ]
)


def scaling_variants(
    *, target_shards: int, worker_counts: Sequence[int] = DEFAULT_WORKER_COUNTS
) -> tuple[DocxExtractionProfileVariant, ...]:
    """Build fixed-corpus worker-scaling variants."""
    return tuple(
        DocxExtractionProfileVariant(
            name=f"scaling-{workers}",
            worker_count=workers,
            target_shards=target_shards,
            role=VariantRole.SCALING,
        )
        for workers in worker_counts
    )


def persistence_variants(*, target_shards: int) -> tuple[DocxExtractionProfileVariant, ...]:
    """Build the persistent-worker treatment paired with the one-worker scaling baseline."""
    return (
        DocxExtractionProfileVariant(
            name="persistence-worker",
            worker_count=1,
            target_shards=target_shards,
            role=VariantRole.PERSISTENCE,
            lifecycle=ConverterLifecycle.PER_WORKER,
            runner=RunnerMode.INLINE,
        ),
    )


def natural_layout_variants(worker_counts: Sequence[int]) -> tuple[DocxExtractionProfileVariant, ...]:
    """Build worker-scaling treatments over the fetched corpus's physical files."""
    return tuple(
        DocxExtractionProfileVariant(
            name=f"natural-{workers}",
            worker_count=workers,
            target_shards=None,
            role=VariantRole.NATURAL_LAYOUT,
            layout=ProfileLayout.NATURAL,
        )
        for workers in worker_counts
    )


def worker_shard_size_variants(
    *, worker_counts: Sequence[int], target_shards: Sequence[int], scaling_target_shards: int | None = None
) -> tuple[DocxExtractionProfileVariant, ...]:
    """Build the worker-by-shard-count matrix used to select shard size."""
    return tuple(
        DocxExtractionProfileVariant(
            name=f"shard-w{workers}-s{shards}",
            worker_count=workers,
            target_shards=shards,
            role=VariantRole.SHARD_SIZE,
        )
        for workers in worker_counts
        for shards in target_shards
        if shards != scaling_target_shards
    )


def _profile_partition_key(record: Mapping[str, object]) -> str:
    source_id = record["source_id"]
    if not isinstance(source_id, str):
        raise TypeError("Fetched DOCX source_id must be a string")
    return source_id


def _prepared_profile_records(_source_id: str, records: Iterator[dict[str, object]]) -> Iterator[dict[str, object]]:
    for record in records:
        counters.pipeline.update_counter("common_crawl_docx_profile/prepared_documents", 1)
        yield record


def prepare_extraction_profile_corpus(
    output_path: str,
    *,
    fetched_input_path: str,
    target_shards: int,
    max_workers: int,
) -> DocxExtractionProfileCorpus:
    """Materialize fetched records into deterministic physical profiling shards."""
    if target_shards <= 0 or max_workers <= 0:
        raise ValueError("target_shards and max_workers must be positive")
    pipeline = (
        Dataset.from_files(prefix_join(fetched_input_path, "data/**/*.parquet"))
        .load_parquet()
        .group_by(
            key=_profile_partition_key,
            reducer=_prepared_profile_records,
            num_output_shards=target_shards,
        )
        .write_parquet(
            prefix_join(output_path, "data/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=FETCHED_COMMON_CRAWL_DOCX_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name="common-crawl-docx-profile-preparation",
        resources=ResourceConfig(cpu=1, ram="8g"),
        max_workers=min(max_workers, target_shards),
        chunk_storage_prefix=prefix_join(output_path, "_zephyr"),
    ).execute(pipeline)
    documents = int(outcome.counters.get("common_crawl_docx_profile/prepared_documents", 0))
    return DocxExtractionProfileCorpus(
        data_dir=prefix_join(output_path, "data"),
        target_shards=target_shards,
        documents=documents,
        counters=dict(outcome.counters),
    )


def profile_extraction_shard(
    paths: Iterator[str],
    shard: ShardInfo,
    *,
    run_token: str,
    variant: DocxExtractionProfileVariant,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> Iterator[dict[str, object]]:
    """Read and convert one Zephyr shard while emitting document and shard measurements."""
    started_at = time.time()
    shard_wall_start = time.perf_counter()
    shard_cpu_start = time.process_time()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak_rss = initial_rss
    process.cpu_percent()
    peak_cpu_percent = 0.0
    read_wall = 0.0
    initialization_wall = 0.0
    initialization_cpu = 0.0
    initializations = 0
    documents_attempted = 0
    documents_succeeded = 0
    input_files = 0
    extractor = DoclingDocxExtractor()

    for path in paths:
        input_files += 1
        if initializations == 0 and variant.lifecycle is ConverterLifecycle.PER_SHARD:
            reset_docling_converter()
            wall_seconds, cpu_seconds = _initialize_extractor(extractor)
            initialization_wall += wall_seconds
            initialization_cpu += cpu_seconds
            initializations += 1
        elif variant.lifecycle is ConverterLifecycle.PER_WORKER:
            wall_seconds, cpu_seconds, initialized = _initialize_persistent_extractor(run_token, extractor)
            initialization_wall += wall_seconds
            initialization_cpu += cpu_seconds
            initializations += initialized
        read_start = time.perf_counter()
        with fsspec.open(path, "rb") as stream:
            records = pq.read_table(stream).to_pylist()
        file_read_wall = time.perf_counter() - read_start
        read_wall += file_read_wall
        for record in records:
            documents_attempted += 1
            payload = record["payload"]
            if not isinstance(payload, bytes):
                raise TypeError("Fetched DOCX payload must be bytes")
            conversion_wall_start = time.perf_counter()
            conversion_cpu_start = time.process_time()
            success = True
            error_kind: str | None = None
            extracted_bytes = 0
            try:
                extracted = validated_extracted_docx(
                    payload,
                    extractor=extractor,
                    maximum_zip_entries=maximum_zip_entries,
                    maximum_uncompressed_bytes=maximum_uncompressed_bytes,
                )
                extracted_bytes = len(extracted.text.encode("utf-8"))
                documents_succeeded += 1
            except (
                InvalidDocxError,
                EmptyDocxTextError,
                DocxExtractionError,
                ValidationError,
            ) as error:
                success = False
                error_kind = type(error).__name__
            conversion_wall = time.perf_counter() - conversion_wall_start
            conversion_cpu = time.process_time() - conversion_cpu_start
            peak_rss = max(peak_rss, process.memory_info().rss)
            peak_cpu_percent = max(peak_cpu_percent, process.cpu_percent())
            yield _profile_row(
                row_kind="document",
                variant=variant,
                shard_idx=shard.shard_idx,
                source_file=path,
                source_id=str(record.get("source_id", "")),
                success=success,
                error_kind=error_kind,
                payload_bytes=len(payload),
                extracted_bytes=extracted_bytes,
                conversion_wall_seconds=conversion_wall,
                conversion_cpu_seconds=conversion_cpu,
            )

    if input_files == 0:
        return
    finished_at = time.time()
    yield _profile_row(
        row_kind="shard",
        variant=variant,
        shard_idx=shard.shard_idx,
        converter_initialization_wall_seconds=initialization_wall,
        converter_initialization_cpu_seconds=initialization_cpu,
        converter_initializations=initializations,
        documents_attempted=documents_attempted,
        documents_succeeded=documents_succeeded,
        input_files=input_files,
        read_wall_seconds=read_wall,
        shard_wall_seconds=time.perf_counter() - shard_wall_start,
        shard_cpu_seconds=time.process_time() - shard_cpu_start,
        peak_rss_bytes=peak_rss,
        peak_cpu_percent=peak_cpu_percent,
        initial_rss_bytes=initial_rss,
        final_rss_bytes=process.memory_info().rss,
        started_at=started_at,
        finished_at=finished_at,
    )


def _initialize_extractor(extractor: DoclingDocxExtractor) -> tuple[float, float]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    extractor.initialize()
    return time.perf_counter() - wall_start, time.process_time() - cpu_start


def _initialize_persistent_extractor(run_token: str, extractor: DoclingDocxExtractor) -> tuple[float, float, int]:
    with _PERSISTENT_CONVERTER_LOCK:
        if run_token in _PERSISTENT_CONVERTER_RUNS:
            return 0.0, 0.0, 0
        reset_docling_converter()
        wall_seconds, cpu_seconds = _initialize_extractor(extractor)
        _PERSISTENT_CONVERTER_RUNS.add(run_token)
        return wall_seconds, cpu_seconds, 1


def _profile_row(
    *,
    row_kind: str,
    variant: DocxExtractionProfileVariant,
    shard_idx: int,
    **values: object,
) -> dict[str, object]:
    row: dict[str, object] = {field.name: None for field in PROFILE_ROW_SCHEMA}
    row.update(
        row_kind=row_kind,
        variant=variant.name,
        variant_role=variant.role.value,
        worker_count=variant.worker_count,
        target_shards=variant.target_shards,
        layout=variant.layout.value,
        lifecycle=variant.lifecycle.value,
        runner=variant.runner.value,
        shard_idx=shard_idx,
        **values,
    )
    return row


def run_extraction_profile(
    output_path: str,
    *,
    input_path: str,
    variant: DocxExtractionProfileVariant,
    maximum_zip_entries: int,
    maximum_uncompressed_bytes: int,
) -> DocxExtractionProfileRun:
    """Run one extraction profiling treatment over the fixed fetched corpus."""
    dataset = Dataset.from_files(prefix_join(input_path, "data/**/*.parquet"))
    if variant.layout is ProfileLayout.PREPARED:
        assert variant.target_shards is not None
        dataset = dataset.reshard(variant.target_shards)
    pipeline = dataset.map_shard(
        partial(
            profile_extraction_shard,
            run_token=output_path,
            variant=variant,
            maximum_zip_entries=maximum_zip_entries,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
        )
    ).write_parquet(
        prefix_join(output_path, "metrics/part-{shard:05d}-of-{total:05d}.parquet"),
        schema=PROFILE_ROW_SCHEMA,
        skip_existing=True,
    )
    outcome = ZephyrContext(
        name=f"common-crawl-docx-profile-{variant.name}",
        resources=ResourceConfig(cpu=2, ram="16g"),
        max_workers=variant.worker_count,
        stage_runner_factory=InlineRunner if variant.runner is RunnerMode.INLINE else SubprocessRunner,
    ).execute(pipeline)
    return DocxExtractionProfileRun(
        metrics_dir=prefix_join(output_path, "metrics"),
        variant=variant.name,
        worker_count=variant.worker_count,
        target_shards=variant.target_shards,
        layout=variant.layout.value,
        lifecycle=variant.lifecycle.value,
        runner=variant.runner.value,
        counters=dict(outcome.counters),
    )


@dataclass(frozen=True)
class RunMetrics:
    """Derived measurements for one profile variant."""

    variant: str
    variant_role: str
    worker_count: int
    peak_concurrency: int
    target_shards: int | None
    observed_shards: int
    layout: str
    lifecycle: str
    runner: str
    documents: int
    failures: int
    failure_distribution: Mapping[str, int]
    payload_bytes: int
    extracted_bytes: int
    stage_wall_seconds: float
    conversion_wall_seconds: float
    conversion_cpu_seconds: float
    initialization_wall_seconds: float
    initialization_cpu_seconds: float
    read_wall_seconds: float
    other_wall_seconds: float
    shard_wall_seconds_total: float
    shard_cpu_seconds_total: float
    peak_rss_bytes: int
    average_cpu_percent: float
    peak_cpu_percent: float
    worker_seconds: float
    read_throughput_bytes_per_second: float
    rss_growth_per_1000_documents: float
    initialization_fraction: float
    documents_per_initialization: float
    top_one_percent_work_share: float
    maximum_document_share: float
    maximum_document_share_percentiles: Mapping[str, float]
    worker_utilization: float
    terminal_idle_fraction: float
    read_fraction: float
    conversion_fraction: float
    other_fraction: float
    document_percentiles: Mapping[str, float]
    shard_percentiles: Mapping[str, float]

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def derive_run_metrics(rows: Sequence[Mapping[str, object]]) -> RunMetrics:
    """Calculate one variant's decision metrics from persisted profile rows."""
    documents = [row for row in rows if row["row_kind"] == "document"]
    shards = [row for row in rows if row["row_kind"] == "shard"]
    if not documents or not shards:
        raise ValueError("Profile rows must contain documents and shards")
    conversion_times = np.asarray([float(row["conversion_wall_seconds"]) for row in documents])
    shard_times = np.asarray([float(row["shard_wall_seconds"]) for row in shards])
    starts = np.asarray([float(row["started_at"]) for row in shards])
    finishes = np.asarray([float(row["finished_at"]) for row in shards])
    stage_wall = float(finishes.max() - starts.min())
    worker_count = int(shards[0]["worker_count"])
    peak_concurrency = _observed_peak_concurrency(starts, finishes)
    conversion_total = float(conversion_times.sum())
    initialization_total = sum(float(row["converter_initialization_wall_seconds"]) for row in shards)
    read_total = sum(float(row["read_wall_seconds"]) for row in shards)
    shard_wall_total = float(shard_times.sum())
    other_total = shard_wall_total - initialization_total - read_total - conversion_total
    if other_total < -1e-6:
        raise ValueError("Measured initialization, read, and conversion time exceeds aggregate shard wall time")
    other_total = max(0.0, other_total)
    initializations = sum(int(row["converter_initializations"]) for row in shards)
    table = _per_shard_maximum_document_share(documents)
    failure_distribution = Counter(
        str(row["error_kind"] or "UnknownError") for row in documents if not bool(row["success"])
    )
    return RunMetrics(
        variant=str(shards[0]["variant"]),
        variant_role=str(shards[0]["variant_role"]),
        worker_count=worker_count,
        peak_concurrency=peak_concurrency,
        target_shards=int(shards[0]["target_shards"]) if shards[0]["target_shards"] is not None else None,
        observed_shards=len(shards),
        layout=str(shards[0]["layout"]),
        lifecycle=str(shards[0]["lifecycle"]),
        runner=str(shards[0]["runner"]),
        documents=len(documents),
        failures=sum(failure_distribution.values()),
        failure_distribution=dict(sorted(failure_distribution.items())),
        payload_bytes=sum(int(row["payload_bytes"]) for row in documents),
        extracted_bytes=sum(int(row["extracted_bytes"]) for row in documents),
        stage_wall_seconds=stage_wall,
        conversion_wall_seconds=conversion_total,
        conversion_cpu_seconds=sum(float(row["conversion_cpu_seconds"]) for row in documents),
        initialization_wall_seconds=initialization_total,
        initialization_cpu_seconds=sum(float(row["converter_initialization_cpu_seconds"]) for row in shards),
        read_wall_seconds=read_total,
        other_wall_seconds=other_total,
        shard_wall_seconds_total=shard_wall_total,
        shard_cpu_seconds_total=sum(float(row["shard_cpu_seconds"]) for row in shards),
        peak_rss_bytes=max(int(row["peak_rss_bytes"]) for row in shards),
        average_cpu_percent=100
        * _divide(sum(float(row["shard_cpu_seconds"]) for row in shards), float(shard_times.sum())),
        peak_cpu_percent=max(float(row["peak_cpu_percent"]) for row in shards),
        worker_seconds=peak_concurrency * stage_wall,
        read_throughput_bytes_per_second=_divide(
            sum(int(row["payload_bytes"]) for row in documents),
            sum(float(row["read_wall_seconds"]) for row in shards),
        ),
        rss_growth_per_1000_documents=_divide(
            sum(int(row["final_rss_bytes"]) - int(row["initial_rss_bytes"]) for row in shards) * 1_000,
            len(documents),
        ),
        initialization_fraction=_divide(initialization_total, initialization_total + conversion_total),
        documents_per_initialization=_divide(len(documents), initializations),
        top_one_percent_work_share=_top_work_share(conversion_times, 0.01),
        maximum_document_share=max(table.values(), default=0.0),
        maximum_document_share_percentiles=_percentiles(np.asarray(list(table.values())), (50, 95, 99, 100)),
        worker_utilization=_divide(float(shard_times.sum()), peak_concurrency * stage_wall),
        terminal_idle_fraction=_terminal_slot_idle_fraction(starts, finishes, peak_concurrency),
        read_fraction=_divide(read_total, shard_wall_total),
        conversion_fraction=_divide(conversion_total, shard_wall_total),
        other_fraction=_divide(other_total, shard_wall_total),
        document_percentiles=_percentiles(conversion_times, (50, 90, 95, 99, 99.9, 100)),
        shard_percentiles=_percentiles(shard_times, (50, 95, 99, 100)),
    )


def _per_shard_maximum_document_share(documents: Sequence[Mapping[str, object]]) -> dict[int, float]:
    by_shard: dict[int, list[float]] = {}
    for row in documents:
        by_shard.setdefault(int(row["shard_idx"]), []).append(float(row["conversion_wall_seconds"]))
    return {shard: _divide(max(times), sum(times)) for shard, times in by_shard.items() if times}


def _top_work_share(values: np.ndarray, fraction: float) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return _divide(float(np.sort(values)[-count:].sum()), float(values.sum()))


def _percentiles(values: np.ndarray, percentiles: Sequence[float]) -> dict[str, float]:
    return {f"p{percentile:g}": float(np.percentile(values, percentile)) for percentile in percentiles}


def _terminal_slot_idle_fraction(starts: np.ndarray, finishes: np.ndarray, slot_count: int) -> float:
    """Measure unused concurrency after the final queued shard has started."""
    terminal_start = float(starts.max())
    stage_finish = float(finishes.max())
    terminal_capacity = slot_count * (stage_finish - terminal_start)
    terminal_busy = sum(
        max(0.0, float(finish) - max(float(start), terminal_start))
        for start, finish in zip(starts, finishes, strict=True)
    )
    return _divide(terminal_capacity - terminal_busy, terminal_capacity)


def _observed_peak_concurrency(starts: np.ndarray, finishes: np.ndarray) -> int:
    events = [(float(start), 1) for start in starts] + [(float(finish), -1) for finish in finishes]
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def profile_decisions(metrics: RunMetrics) -> list[tuple[str, str, str]]:
    """Map measurements to concise threshold-based recommendations."""
    shard_ratio = _divide(metrics.shard_percentiles["p99"], metrics.shard_percentiles["p50"])
    shard_decision: tuple[str, str] = (
        ("insufficient", f"Only {metrics.observed_shards} non-empty shard was measured.")
        if metrics.observed_shards < 2
        else (
            "good" if shard_ratio < 2 else "watch" if shard_ratio < 3 else "action",
            f"p99/p50 is {shard_ratio:.2f}x.",
        )
    )
    idle_decision: tuple[str, str] = (
        ("insufficient", "Peak concurrency was one; terminal idle cannot measure worker imbalance.")
        if metrics.peak_concurrency < 2
        else (_idle_rating(metrics.terminal_idle_fraction), _idle_action(metrics))
    )
    return [
        (
            "Converter initialization",
            _initialization_rating(metrics.initialization_fraction),
            _initialization_action(metrics),
        ),
        ("Document tail", _tail_rating(metrics.top_one_percent_work_share), _tail_action(metrics)),
        ("Shard balance", shard_decision[0], shard_decision[1]),
        ("Terminal idle", idle_decision[0], idle_decision[1]),
        ("I/O versus conversion", "context", _service_action(metrics)),
    ]


def _initialization_rating(fraction: float) -> str:
    return "good" if fraction < 0.05 else "acceptable" if fraction < 0.10 else "watch" if fraction < 0.20 else "action"


def _initialization_action(metrics: RunMetrics) -> str:
    return (
        f"Initialization is {metrics.initialization_fraction:.1%}; "
        f"{metrics.documents_per_initialization:.1f} documents per initialization."
    )


def _tail_rating(share: float) -> str:
    return "good" if share < 0.10 else "acceptable" if share < 0.25 else "watch" if share < 0.50 else "action"


def _tail_action(metrics: RunMetrics) -> str:
    return (
        f"The slowest 1% consume {metrics.top_one_percent_work_share:.1%}; "
        f"maximum within-shard document share is {metrics.maximum_document_share:.1%}."
    )


def _idle_rating(fraction: float) -> str:
    return "good" if fraction < 0.10 else "acceptable" if fraction < 0.20 else "watch" if fraction < 0.40 else "action"


def _idle_action(metrics: RunMetrics) -> str:
    return (
        f"Estimated terminal idle fraction is {metrics.terminal_idle_fraction:.1%}; "
        f"aggregate utilization is {metrics.worker_utilization:.1%}."
    )


def _service_action(metrics: RunMetrics) -> str:
    if metrics.conversion_fraction >= 0.60:
        return "Conversion dominates; scale CPU conversion workers before changing architecture."
    if metrics.read_fraction >= 0.50:
        return "Reading dominates; investigate shard size, compression, region, and storage contention."
    return "Read and conversion costs are comparable; direct Zephyr can overlap them across tasks."


def render_profile_report(
    output_path: str,
    *,
    run_paths: tuple[str, ...],
) -> DocxExtractionProfileReport:
    """Render Markdown, JSON, and a self-contained Plotly dashboard."""
    rows_by_variant: dict[str, list[dict[str, object]]] = {}
    counters_by_variant: dict[str, dict[str, int | float]] = {}
    all_rows: list[dict[str, object]] = []
    for run_path in run_paths:
        run = read_artifact(run_path, DocxExtractionProfileRun)
        rows = _parquet_rows(run.metrics_dir)
        rows_by_variant[run.variant] = rows
        counters_by_variant[run.variant] = run.counters
        all_rows.extend(rows)
    metrics = [derive_run_metrics(rows) for rows in rows_by_variant.values()]
    summary = profile_summary(metrics, counters_by_variant=counters_by_variant)
    markdown = _markdown_report(metrics, summary)
    dashboard = _dashboard_html(all_rows, metrics)
    markdown_path = prefix_join(output_path, "report.md")
    dashboard_path = prefix_join(output_path, "dashboard.html")
    summary_path = prefix_join(output_path, "summary.json")
    with fsspec.open(markdown_path, "wt") as stream:
        stream.write(markdown)
    with fsspec.open(dashboard_path, "wt") as stream:
        stream.write(dashboard)
    with fsspec.open(summary_path, "wt") as stream:
        json.dump(summary, stream, indent=2)
    return DocxExtractionProfileReport(
        markdown_path=markdown_path,
        dashboard_path=dashboard_path,
        summary_path=summary_path,
    )


def _parquet_rows(path: str) -> list[dict[str, object]]:
    fs, resolved = url_to_fs(prefix_join(path, "*.parquet"))
    protocol = fsspec.core.split_protocol(path)[0]
    rows: list[dict[str, object]] = []
    for matched_path in sorted(fs.glob(resolved)):
        full_path = f"{protocol}://{matched_path}" if protocol else matched_path
        with fsspec.open(full_path, "rb") as stream:
            rows.extend(pq.read_table(stream).to_pylist())
    return rows


def profile_summary(
    metrics: Sequence[RunMetrics], *, counters_by_variant: Mapping[str, Mapping[str, int | float]] | None = None
) -> dict[str, object]:
    """Build the machine-readable cross-run summary used by the report."""
    baseline = next(
        (metric for metric in metrics if metric.worker_count == 1 and metric.variant_role == VariantRole.SCALING.value),
        None,
    )
    scaling = []
    for metric in metrics:
        if metric.variant_role != VariantRole.SCALING.value or baseline is None:
            continue
        speedup = _divide(baseline.stage_wall_seconds, metric.stage_wall_seconds)
        scaling.append(
            {
                "variant": metric.variant,
                "requested_workers": metric.worker_count,
                "peak_concurrency": metric.peak_concurrency,
                "speedup": speedup,
                "efficiency": _divide(speedup, metric.peak_concurrency),
            }
        )
    cold = next(
        (
            metric
            for metric in metrics
            if metric.variant_role == VariantRole.SCALING.value
            and metric.worker_count == 1
            and metric.runner == RunnerMode.SUBPROCESS.value
        ),
        None,
    )
    warm = next(
        (
            metric
            for metric in metrics
            if metric.variant_role == VariantRole.PERSISTENCE.value
            and metric.lifecycle == ConverterLifecycle.PER_WORKER.value
        ),
        None,
    )
    persistence = None
    if cold and warm:
        speedup = _divide(cold.stage_wall_seconds, warm.stage_wall_seconds)
        persistence = {
            "speedup": speedup,
            "savings_seconds": cold.stage_wall_seconds - warm.stage_wall_seconds,
            "percent_saved": 1 - _divide(warm.stage_wall_seconds, cold.stage_wall_seconds),
        }
    return {
        "runs": [metric.as_dict() for metric in metrics],
        "scaling": scaling,
        "persistence": persistence,
        "optimal_shards": [_optimal_shard_summary(metric) for metric in _optimal_shard_sizes(metrics)],
        "zephyr_counters": dict(counters_by_variant or {}),
    }


def _markdown_report(metrics: Sequence[RunMetrics], summary: Mapping[str, object]) -> str:
    lines = [
        "# DOCX extraction profiling report",
        "",
        "The fixed fetched corpus is reused for every treatment. Wall time describes capacity and tail behavior; "
        "conversion CPU time is the safer signal for code-cost comparisons.",
        "",
        "Open `dashboard.html` for interactive distributions, timelines, scaling, and cost decomposition.",
        "",
        "## Run overview",
        "",
        "| Run | Layout | Runner | Requested workers | Peak concurrency | Target/observed shards | Documents | "
        "Wall time | Docs/s | Init/wall | Read | Conversion | Other | Utilization | Terminal idle |",
        "| " + " | ".join(("---", "---", "---", *("---:",) * 12)) + " |",
    ]
    for metric in metrics:
        lines.append(
            f"| `{metric.variant}` | {metric.layout} | {metric.runner} | {metric.worker_count} | "
            f"{metric.peak_concurrency} | {_target_shard_label(metric)}/{metric.observed_shards} | "
            f"{metric.documents:,} | "
            f"{_duration(metric.stage_wall_seconds)} | {_divide(metric.documents, metric.stage_wall_seconds):.2f} | "
            f"{_divide(metric.initialization_wall_seconds, metric.shard_wall_seconds_total):.1%} | "
            f"{metric.read_fraction:.1%} | {metric.conversion_fraction:.1%} | {metric.other_fraction:.1%} | "
            f"{metric.worker_utilization:.1%} | {metric.terminal_idle_fraction:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Service rates and resource footprint",
            "",
            "| Run | Read MiB/s | Stage payload MiB/s | Extracted MiB/s | Conversion docs/s | "
            "Avg/peak CPU | Peak RSS | Worker-seconds | Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in metrics:
        lines.append(
            f"| `{metric.variant}` | {metric.read_throughput_bytes_per_second / _MEBIBYTE:.2f} | "
            f"{_divide(metric.payload_bytes, metric.stage_wall_seconds) / _MEBIBYTE:.2f} | "
            f"{_divide(metric.extracted_bytes, metric.stage_wall_seconds) / _MEBIBYTE:.2f} | "
            f"{_divide(metric.documents, metric.conversion_wall_seconds):.2f} | "
            f"{metric.average_cpu_percent:.0f}%/{metric.peak_cpu_percent:.0f}% | "
            f"{metric.peak_rss_bytes / _MEBIBYTE:.0f} MiB | {metric.worker_seconds:.1f} | "
            f"{metric.failures} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate shard-time decomposition",
            "",
            "`Other = shard wall - initialization - read - conversion`. It is wall time not attributed to the three "
            "explicitly timed operations.",
            "",
            "| Run | Shard wall | Initialization | Read | Conversion | Other |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in metrics:
        lines.append(
            f"| `{metric.variant}` | {_duration(metric.shard_wall_seconds_total)} | "
            f"{_duration(metric.initialization_wall_seconds)} | {_duration(metric.read_wall_seconds)} | "
            f"{_duration(metric.conversion_wall_seconds)} | {_duration(metric.other_wall_seconds)} |"
        )
    lines.extend(["", "## Extraction failure distribution", ""])
    if any(metric.failure_distribution for metric in metrics):
        lines.extend(
            [
                "| Run | Error kind | Documents | Share of run failures |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for metric in metrics:
            for error_kind, count in metric.failure_distribution.items():
                lines.append(
                    f"| `{metric.variant}` | `{error_kind}` | {count:,} | {_divide(count, metric.failures):.1%} |"
                )
    else:
        lines.append("No document-level extraction failures were recorded.")
    representative = _representative_run(metrics)
    shard_p99_p50 = _divide(
        representative.shard_percentiles["p99"],
        representative.shard_percentiles["p50"],
    )
    shard_max_p50 = _divide(
        representative.shard_percentiles["p100"],
        representative.shard_percentiles["p50"],
    )
    lines.extend(["", "## Decision summary", ""])
    for objective, rating, detail in profile_decisions(representative):
        lines.append(f"- **{objective} — {rating}:** {detail}")
    lines.extend(
        [
            "",
            "## Document and shard tails",
            "",
            f"Representative run: `{representative.variant}`.",
            "",
            f"- Document conversion p50/p90/p95/p99/p99.9/max: {_percentile_line(representative.document_percentiles)}.",
            f"- Shard wall p50/p95/p99/max: {_percentile_line(representative.shard_percentiles)}.",
            f"- Shard p99/p50: {shard_p99_p50:.2f}x; maximum/p50: {shard_max_p50:.2f}x.",
            f"- Slowest 1% work share: {representative.top_one_percent_work_share:.1%}.",
            f"- Largest single-document share within any shard: {representative.maximum_document_share:.1%}.",
            "",
            "## Scaling",
            "",
        ]
    )
    scaling = summary.get("scaling")
    if isinstance(scaling, list) and scaling:
        lines.extend(
            [
                "| Requested | Peak concurrency | Speedup | Parallel efficiency |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in scaling:
            assert isinstance(row, dict)
            lines.append(
                f"| {row['requested_workers']} | {row['peak_concurrency']} | "
                f"{row['speedup']:.2f}x | {row['efficiency']:.1%} |"
            )
    else:
        lines.append("Scaling variants were not included.")
    natural = sorted(
        (metric for metric in metrics if metric.layout == ProfileLayout.NATURAL.value),
        key=lambda metric: metric.worker_count,
    )
    lines.extend(["", "## Natural fetched layout", ""])
    if natural:
        lines.extend(
            [
                "| Workers | Physical shards | Wall time | Initialization | Peak concurrency |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in natural:
            lines.append(
                f"| {metric.worker_count} | {metric.observed_shards} | {_duration(metric.stage_wall_seconds)} | "
                f"{metric.initialization_fraction:.1%} | {metric.peak_concurrency} |"
            )
    else:
        lines.append("Natural-layout variants were not included.")
    lines.extend(["", "## Shard-size choice", ""])
    shard_choices = _optimal_shard_sizes(metrics)
    if not shard_choices:
        lines.append("Shard-size variants were not included.")
    else:
        lines.extend(
            [
                "| Workers | Selected run | Shards | Documents/shard | Initialization | Target met |",
                "| ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for choice in shard_choices:
            lines.append(
                f"| {choice.worker_count} | `{choice.variant}` | {choice.observed_shards} | "
                f"{_divide(choice.documents, choice.observed_shards):.1f} | "
                f"{choice.initialization_fraction:.1%} | "
                f"{'yes' if choice.initialization_fraction <= 0.10 else 'no'} |"
            )
    lines.extend(["", "## Persistent converter experiment", ""])
    persistence = summary.get("persistence")
    if isinstance(persistence, dict):
        lines.append(
            f"Persistent-worker speedup is {persistence['speedup']:.2f}x, saving "
            f"{_duration(float(persistence['savings_seconds']))} ({persistence['percent_saved']:.1%}). "
            f"Warm-process RSS growth is {_representative_persistence_growth(metrics):.1f} MiB per 1,000 documents."
        )
    else:
        lines.append("Cold and warm persistence variants were not both included.")
    lines.extend(
        [
            "",
            "## Measurement caveats",
            "",
            "- Read time includes Parquet access and decoding. Conversion time includes DOCX validation and "
            "Docling conversion.",
            "- Worker utilization and terminal idle are derived from the measured concurrency timeline. Terminal idle "
            "starts when the final queued shard begins; task-process PIDs are not treated as workers.",
            "- Cross-worker stage wall time uses wall clocks and can contain small clock-skew error.",
            "- Peak CPU is sampled at document boundaries; use Zephyr finelog CPU time as the primary A/B cost signal.",
            "- Task retries are not present in successful output rows; obtain them from coordinator logs and "
            "finelog before a final decision.",
            "- The persistent-worker treatment clears any inherited converter before its first shard, then reuses "
            "the process-local converter. It does not repeat Python imports for every shard.",
            "- Output writing occurs after rows are yielded and is represented in Zephyr stage statistics, "
            "not shard conversion rows.",
            "- Other time includes profiling overhead and work outside the three explicit timers; compare it with "
            "Zephyr stage statistics before assigning it to a specific cause.",
            "- Compare CPU-seconds per document across code changes; use wall time for scaling and straggler "
            "analysis only.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _representative_run(metrics: Sequence[RunMetrics]) -> RunMetrics:
    scaling = [metric for metric in metrics if metric.variant_role == VariantRole.SCALING.value]
    return max(scaling, key=lambda metric: metric.worker_count) if scaling else metrics[0]


def _percentile_line(values: Mapping[str, float]) -> str:
    return "/".join(_duration(value) for value in values.values())


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1_000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def _dashboard_html(rows: Sequence[Mapping[str, object]], metrics: Sequence[RunMetrics]) -> str:
    documents = [row for row in rows if row["row_kind"] == "document"]
    shards = [row for row in rows if row["row_kind"] == "shard"]
    figures = [
        _document_distribution_figure(documents),
        _failure_distribution_figure(metrics),
        _shard_distribution_figure(shards),
        _maximum_document_share_figure(documents),
        _timeline_figure(shards),
        _decomposition_figure(metrics),
        _scaling_figure(metrics),
        _natural_layout_figure(metrics),
        _persistence_figure(metrics),
        _shard_size_figure(metrics),
        _optimal_shard_size_figure(metrics),
    ]
    sections: list[str] = []
    for index, (title, figure) in enumerate(figures):
        sections.append(
            f"<section><h2>{title}</h2>{figure.to_html(full_html=False, include_plotlyjs=index == 0)}</section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>DOCX extraction profiling</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1400px}"
        "section{margin:3rem 0}h1,h2{color:#243447}</style>"
        "</head><body><h1>DOCX extraction profiling dashboard</h1>" + "".join(sections) + "</body></html>"
    )


def _document_distribution_figure(rows: Sequence[Mapping[str, object]]) -> tuple[str, go.Figure]:
    figure = go.Figure()
    for variant in sorted({str(row["variant"]) for row in rows}):
        values = [float(row["conversion_wall_seconds"]) for row in rows if row["variant"] == variant]
        figure.add_trace(go.Histogram(x=values, name=variant, opacity=0.65, nbinsx=50))
    figure.update_layout(barmode="overlay", xaxis_title="Document conversion seconds", yaxis_title="Documents")
    return "Per-document conversion distribution", figure


def _failure_distribution_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    error_kinds = sorted({error_kind for metric in metrics for error_kind in metric.failure_distribution})
    figure = go.Figure()
    for error_kind in error_kinds:
        figure.add_trace(
            go.Bar(
                name=error_kind,
                x=[metric.variant for metric in metrics],
                y=[metric.failure_distribution.get(error_kind, 0) for metric in metrics],
            )
        )
    figure.update_layout(barmode="stack", yaxis_title="Failed documents")
    return "Extraction failure distribution", figure


def _shard_distribution_figure(rows: Sequence[Mapping[str, object]]) -> tuple[str, go.Figure]:
    figure = go.Figure()
    for variant in sorted({str(row["variant"]) for row in rows}):
        values = [float(row["shard_wall_seconds"]) for row in rows if row["variant"] == variant]
        figure.add_trace(go.Box(y=values, name=variant, boxpoints="outliers"))
    figure.update_layout(yaxis_title="Shard wall seconds")
    return "Shard duration and stragglers", figure


def _maximum_document_share_figure(rows: Sequence[Mapping[str, object]]) -> tuple[str, go.Figure]:
    figure = go.Figure()
    for variant in sorted({str(row["variant"]) for row in rows}):
        shares = _per_shard_maximum_document_share([row for row in rows if row["variant"] == variant])
        figure.add_trace(go.Box(y=[100 * share for share in shares.values()], name=variant, boxpoints="outliers"))
    figure.update_layout(yaxis_title="Slowest document share of shard conversion time (%)")
    return "Maximum-document share by shard", figure


def _timeline_figure(rows: Sequence[Mapping[str, object]]) -> tuple[str, go.Figure]:
    figure = go.Figure()
    for row in rows:
        figure.add_trace(
            go.Bar(
                x=[float(row["finished_at"]) - float(row["started_at"])],
                y=[f"{row['variant']} / {row['shard_idx']}"],
                base=[float(row["started_at"])],
                orientation="h",
                showlegend=False,
            )
        )
    figure.update_layout(xaxis_title="Unix time", yaxis_title="Variant / shard", height=max(450, 18 * len(rows)))
    return "Shard execution timeline", figure


def _decomposition_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    names = [metric.variant for metric in metrics]
    init = [metric.initialization_wall_seconds for metric in metrics]
    read = [metric.read_wall_seconds for metric in metrics]
    conversion = [metric.conversion_wall_seconds for metric in metrics]
    other = [metric.other_wall_seconds for metric in metrics]
    figure = go.Figure()
    for label, values in (("Initialization", init), ("Read", read), ("Conversion", conversion), ("Other", other)):
        figure.add_trace(go.Bar(name=label, x=names, y=values))
    figure.update_layout(barmode="stack", yaxis_title="Aggregate shard-seconds")
    return "Time decomposition", figure


def _scaling_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    scaling = sorted(
        (metric for metric in metrics if metric.variant_role == VariantRole.SCALING.value),
        key=lambda metric: metric.worker_count,
    )
    figure = go.Figure()
    if scaling:
        baseline_run = next((metric for metric in scaling if metric.worker_count == 1), None)
        if baseline_run is None:
            return "Horizontal scaling (requires a 1-worker baseline)", figure
        baseline = baseline_run.stage_wall_seconds
        speedups = [_divide(baseline, metric.stage_wall_seconds) for metric in scaling]
        efficiencies = [
            _divide(speedup, metric.peak_concurrency) for speedup, metric in zip(speedups, scaling, strict=True)
        ]
        figure.add_trace(
            go.Scatter(x=[metric.worker_count for metric in scaling], y=speedups, name="Speedup", mode="lines+markers")
        )
        figure.add_trace(
            go.Scatter(
                x=[metric.worker_count for metric in scaling],
                y=[100 * value for value in efficiencies],
                name="Efficiency %",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        figure.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "Parallel efficiency %"})
    figure.update_layout(xaxis_title="Workers", yaxis_title="Speedup")
    return "Horizontal scaling", figure


def _persistence_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    persistence = [
        metric
        for metric in metrics
        if metric.variant_role == VariantRole.PERSISTENCE.value
        or (metric.variant_role == VariantRole.SCALING.value and metric.worker_count == 1)
    ]
    figure = go.Figure(
        go.Bar(x=[metric.variant for metric in persistence], y=[metric.stage_wall_seconds for metric in persistence])
    )
    figure.update_layout(yaxis_title="Wall seconds")
    return "Cold versus warm converter", figure


def _natural_layout_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    natural = sorted(
        (metric for metric in metrics if metric.layout == ProfileLayout.NATURAL.value),
        key=lambda metric: metric.worker_count,
    )
    prepared = sorted(
        (metric for metric in metrics if metric.variant_role == VariantRole.SCALING.value),
        key=lambda metric: metric.worker_count,
    )
    figure = go.Figure()
    for label, runs in (("Natural", natural), ("Prepared", prepared)):
        figure.add_trace(
            go.Scatter(
                x=[metric.worker_count for metric in runs],
                y=[metric.stage_wall_seconds for metric in runs],
                name=label,
                mode="lines+markers",
            )
        )
    figure.update_layout(xaxis_title="Requested workers", yaxis_title="Stage wall seconds")
    return "Natural versus prepared layout", figure


def _shard_size_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    shard_runs = sorted(
        (metric for metric in metrics if _is_shard_size_candidate(metric)),
        key=lambda metric: metric.shard_percentiles["p50"],
    )
    figure = go.Figure()
    if shard_runs:
        figure.add_trace(
            go.Scatter(
                x=[metric.shard_percentiles["p50"] / 60 for metric in shard_runs],
                y=[metric.stage_wall_seconds / 60 for metric in shard_runs],
                text=[metric.variant for metric in shard_runs],
                mode="markers+text",
                name="Makespan",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[metric.shard_percentiles["p50"] / 60 for metric in shard_runs],
                y=[100 * metric.initialization_fraction for metric in shard_runs],
                mode="lines+markers",
                name="Initialization %",
                yaxis="y2",
            )
        )
        figure.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "Initialization %"})
    figure.update_layout(xaxis_title="Median shard minutes", yaxis_title="Stage makespan minutes")
    return "Shard-size tradeoff", figure


def _optimal_shard_size_figure(metrics: Sequence[RunMetrics]) -> tuple[str, go.Figure]:
    optimal = _optimal_shard_sizes(metrics)
    figure = go.Figure(
        go.Scatter(
            x=[metric.worker_count for metric in optimal],
            y=[_divide(metric.documents, metric.observed_shards) for metric in optimal],
            text=[
                f"{metric.observed_shards} shards; initialization {metric.initialization_fraction:.1%}"
                for metric in optimal
            ],
            mode="lines+markers",
            marker={"color": ["#2ca02c" if metric.initialization_fraction <= 0.10 else "#d62728" for metric in optimal]},
        )
    )
    figure.update_layout(xaxis_title="Requested workers", yaxis_title="Selected documents per shard")
    return "Workers versus optimal measured shard size", figure


def _representative_persistence_growth(metrics: Sequence[RunMetrics]) -> float:
    warm = next(
        (
            metric
            for metric in metrics
            if metric.variant_role == VariantRole.PERSISTENCE.value
            and metric.lifecycle == ConverterLifecycle.PER_WORKER.value
        ),
        None,
    )
    return 0.0 if warm is None else warm.rss_growth_per_1000_documents / _MEBIBYTE


def _optimal_shard_sizes(metrics: Sequence[RunMetrics]) -> list[RunMetrics]:
    selected: list[RunMetrics] = []
    worker_counts = sorted({metric.worker_count for metric in metrics if _is_shard_size_candidate(metric)})
    for workers in worker_counts:
        candidates = [
            metric for metric in metrics if _is_shard_size_candidate(metric) and metric.worker_count == workers
        ]
        amortized = [metric for metric in candidates if metric.initialization_fraction <= 0.10]
        selected.append(
            max(amortized, key=lambda metric: metric.observed_shards)
            if amortized
            else min(candidates, key=lambda metric: metric.initialization_fraction)
        )
    return selected


def _is_shard_size_candidate(metric: RunMetrics) -> bool:
    return metric.variant_role in {VariantRole.SHARD_SIZE.value, VariantRole.SCALING.value}


def _recommended_shard_size(metrics: Sequence[RunMetrics]) -> RunMetrics | None:
    optimal = _optimal_shard_sizes(metrics)
    return max(optimal, key=lambda metric: metric.worker_count) if optimal else None


def _optimal_shard_summary(metric: RunMetrics) -> dict[str, object]:
    return {
        "workers": metric.worker_count,
        "variant": metric.variant,
        "shards": metric.observed_shards,
        "documents_per_shard": _divide(metric.documents, metric.observed_shards),
        "initialization_fraction": metric.initialization_fraction,
        "meets_initialization_target": metric.initialization_fraction <= 0.10,
    }


def _target_shard_label(metric: RunMetrics) -> str:
    return "natural" if metric.target_shards is None else str(metric.target_shards)


def common_crawl_docx_profile_steps(
    config: DocxExtractionProfileConfig, *, fetched: StepSpec
) -> tuple[StepSpec, tuple[StepSpec, ...], StepSpec]:
    """Build independent profile runs followed by one report step."""
    slug = config.name.lower()
    preparation = StepSpec(
        name=f"profiling/common-crawl-docx/{slug}/prepared",
        fn=remote(
            partial(
                prepare_extraction_profile_corpus,
                fetched_input_path=fetched.output_path,
                target_shards=config.preparation_shards,
                max_workers=max(variant.worker_count for variant in config.variants),
            ),
            resources=ResourceConfig(cpu=1, ram="8g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[fetched],
        hash_attrs={"target_shards": config.preparation_shards, "schema_version": 1},
    )
    runs = tuple(
        StepSpec(
            name=f"profiling/common-crawl-docx/{slug}/{variant.name}",
            fn=remote(
                partial(
                    run_extraction_profile,
                    input_path=(
                        preparation.output_path if variant.layout is ProfileLayout.PREPARED else fetched.output_path
                    ),
                    variant=variant,
                    maximum_zip_entries=config.maximum_zip_entries,
                    maximum_uncompressed_bytes=config.maximum_uncompressed_bytes,
                ),
                resources=ResourceConfig(cpu=1, ram="4g"),
                pip_dependency_groups=["datakit"],
            ),
            deps=[preparation] if variant.layout is ProfileLayout.PREPARED else [fetched],
            hash_attrs={
                "variant": variant.hash_attrs,
                "maximum_zip_entries": config.maximum_zip_entries,
                "maximum_uncompressed_bytes": config.maximum_uncompressed_bytes,
                "schema_version": PROFILE_SCHEMA_VERSION,
            },
        )
        for variant in config.variants
    )
    report = StepSpec(
        name=f"profiling/common-crawl-docx/{slug}/report",
        fn=partial(render_profile_report, run_paths=tuple(run.output_path for run in runs)),
        deps=list(runs),
        hash_attrs={"report_version": 3},
    )
    return preparation, runs, report


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def default_profile_shard_counts(maximum_shards: int) -> tuple[int, ...]:
    """Return logarithmically spaced shard counts up to the prepared maximum."""
    if maximum_shards <= 0:
        raise ValueError("maximum_shards must be positive")
    return tuple(sorted({max(1, maximum_shards // divisor) for divisor in (8, 4, 2, 1)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetched-input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--name", default="docx-extraction-profile")
    parser.add_argument("--worker-counts", type=_comma_separated_ints, default=DEFAULT_WORKER_COUNTS)
    parser.add_argument("--target-shards", type=int)
    parser.add_argument("--shard-counts", type=_comma_separated_ints)
    parser.add_argument("--skip-natural-layout", action="store_true")
    parser.add_argument("--skip-persistence", action="store_true")
    parser.add_argument("--maximum-zip-entries", type=int, default=DEFAULT_MAXIMUM_ZIP_ENTRIES)
    parser.add_argument("--maximum-uncompressed-bytes", type=int, default=DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_glob = prefix_join(args.fetched_input_path, "data/**/*.parquet")
    fs, resolved = url_to_fs(input_glob)
    input_shards = len(fs.glob(resolved))
    if input_shards == 0:
        raise FileNotFoundError(f"No fetched Parquet shards match {input_glob}")
    target_shards = args.target_shards or DEFAULT_PREPARATION_SHARDS
    worker_counts = tuple(args.worker_counts)
    variants = list(scaling_variants(target_shards=target_shards, worker_counts=worker_counts))
    if not args.skip_natural_layout:
        variants.extend(natural_layout_variants(worker_counts))
    shard_counts = tuple(args.shard_counts or default_profile_shard_counts(target_shards))
    variants.extend(
        worker_shard_size_variants(
            worker_counts=worker_counts,
            target_shards=shard_counts,
            scaling_target_shards=target_shards,
        )
    )
    if not args.skip_persistence and target_shards > 1:
        variants.extend(persistence_variants(target_shards=target_shards))
    config = DocxExtractionProfileConfig(
        name=args.name,
        variants=tuple(variants),
        preparation_shards=target_shards,
        maximum_zip_entries=args.maximum_zip_entries,
        maximum_uncompressed_bytes=args.maximum_uncompressed_bytes,
    )
    if args.dry_run:
        print(json.dumps([variant.__dict__ for variant in config.variants], default=str, indent=2))
        return

    prepared_path = prefix_join(args.output_path, "prepared")
    prepared = prepare_extraction_profile_corpus(
        prepared_path,
        fetched_input_path=args.fetched_input_path,
        target_shards=config.preparation_shards,
        max_workers=max(variant.worker_count for variant in config.variants),
    )
    write_artifact(prepared, prepared_path)
    run_paths: list[str] = []
    for variant in config.variants:
        run_path = prefix_join(prefix_join(args.output_path, "runs"), variant.name)
        result = run_extraction_profile(
            run_path,
            input_path=(prepared_path if variant.layout is ProfileLayout.PREPARED else args.fetched_input_path),
            variant=variant,
            maximum_zip_entries=config.maximum_zip_entries,
            maximum_uncompressed_bytes=config.maximum_uncompressed_bytes,
        )
        write_artifact(result, run_path)
        run_paths.append(run_path)
    report_path = prefix_join(args.output_path, "report")
    report = render_profile_report(report_path, run_paths=tuple(run_paths))
    write_artifact(report, report_path)
    print(f"Markdown report: {report.markdown_path}")
    print(f"Interactive dashboard: {report.dashboard_path}")


if __name__ == "__main__":
    main()
