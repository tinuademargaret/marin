# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render matched extraction samples from an existing fetched DOCX stage.

Example::

    uv run --package marin-core --extra datakit python \
      -m experiments.datakit.common_crawl_docx_extraction_sample \
      --fetched-step-path gs://bucket/run/samples/common-crawl-docx/crawl/fetched_HASH \
      --output-path gs://bucket/run/extraction-samples/compare-1 \
      --sample-size 12 --seed 7 \
      --extraction-method docling-plain-inline \
      --extraction-method docling-markdown-inline
"""

import argparse
import hashlib
import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass

import fsspec
import pyarrow.parquet as pq
from marin.datakit.download.common_crawl_docx import (
    DOCLING_IMAGE_PLACEHOLDER,
    CommonCrawlDocxStageResult,
    DocumentBlock,
    DocxExtractionError,
)
from marin.execution.artifact import read_artifact
from rigging.filesystem import prefix_join, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.docx_extraction_methods import (
    DOCX_EXTRACTION_METHODS,
    ExtractionMethod,
    extraction_methods,
    remove_markdown_markers,
)

DEFAULT_MAXIMUM_REPORT_CHARACTERS = 8_000
_BACKTICK_RUN = re.compile(r"`+")
_SAMPLE_COLUMNS = ("payload", "source_id", "crawl_id", "url", "selection_reason")
_FACTORIAL_COMPARISONS = (
    ("Plain table placement", "docling-plain-inline", "docling-plain-tables-at-end"),
    ("Markdown table placement", "docling-markdown-inline", "docling-markdown-tables-at-end"),
    ("Inline serialization", "docling-plain-inline", "docling-markdown-inline"),
    ("Tables-at-end serialization", "docling-plain-tables-at-end", "docling-markdown-tables-at-end"),
)


@dataclass(frozen=True)
class SampledFetchedDocument:
    """One payload selected from the fetched corpus before extraction."""

    source_id: str
    crawl_id: str
    url: str
    selection_reason: str
    payload: bytes


@dataclass(frozen=True)
class ExtractionSample:
    """One extractor's observable result for a sampled fetched document."""

    source_id: str
    crawl_id: str
    url: str
    selection_reason: str
    method: str
    extractor_identity: str
    text: str | None
    word_count: int | None
    table_count: int | None
    image_count: int | None
    substantive_text_digest: str | None
    survives_blank_text_filter: bool
    error: str | None


def _substantive_text_digest(blocks: tuple[DocumentBlock, ...]) -> str:
    prose = "\n\n".join(block.text for block in blocks if not block.is_table)
    prose = remove_markdown_markers(prose.replace(DOCLING_IMAGE_PLACEHOLDER, ""))
    normalized = " ".join(prose.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _full_path(protocol: str | None, path: str) -> str:
    return f"{protocol}://{path}" if protocol else path


def fetched_parquet_paths(fetched_step_path: str) -> tuple[str, ...]:
    """Resolve the Parquet shards produced by a successful fetched stage."""
    fetched = read_artifact(fetched_step_path, CommonCrawlDocxStageResult)
    pattern = prefix_join(fetched.data_dir, "*.parquet")
    fs, resolved = url_to_fs(pattern)
    protocol = fsspec.core.split_protocol(pattern)[0]
    paths = tuple(_full_path(protocol, path) for path in sorted(fs.glob(resolved)))
    if not paths:
        raise FileNotFoundError(f"No fetched Parquet shards matched {pattern}")
    return paths


def sample_fetched_documents(
    paths: tuple[str, ...],
    *,
    sample_size: int,
    seed: int,
) -> tuple[SampledFetchedDocument, ...]:
    """Select a bounded, reproducible sample spread across fetched shards."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    shuffled_paths = list(paths)
    random.Random(seed).shuffle(shuffled_paths)
    paths_to_read = shuffled_paths[: min(sample_size, len(shuffled_paths))]
    selected: list[SampledFetchedDocument] = []
    for path_index, path in enumerate(paths_to_read):
        records_per_path = math.ceil((sample_size - len(selected)) / (len(paths_to_read) - path_index))
        records_from_path = 0
        with fsspec.open(path, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=records_per_path, columns=list(_SAMPLE_COLUMNS)):
                for row in batch.to_pylist():
                    payload = row["payload"]
                    if not isinstance(payload, bytes):
                        raise TypeError("Fetched DOCX payload must be bytes")
                    selected.append(
                        SampledFetchedDocument(
                            source_id=str(row["source_id"]),
                            crawl_id=str(row["crawl_id"]),
                            url=str(row["url"]),
                            selection_reason=str(row["selection_reason"]),
                            payload=payload,
                        )
                    )
                    records_from_path += 1
                    if records_from_path == records_per_path or len(selected) == sample_size:
                        break
                if records_from_path == records_per_path or len(selected) == sample_size:
                    break
        if len(selected) == sample_size:
            break
    if len(selected) != sample_size:
        raise ValueError(f"Requested {sample_size} documents, but selected only {len(selected)}")
    return tuple(selected)


def extract_samples(
    documents: tuple[SampledFetchedDocument, ...],
    methods: tuple[ExtractionMethod, ...],
) -> tuple[ExtractionSample, ...]:
    """Run every method over the same sampled payloads."""
    samples: list[ExtractionSample] = []
    for document in documents:
        for method in methods:
            try:
                extracted = method.extract(document.payload)
            except DocxExtractionError as error:
                samples.append(
                    ExtractionSample(
                        source_id=document.source_id,
                        crawl_id=document.crawl_id,
                        url=document.url,
                        selection_reason=document.selection_reason,
                        method=method.name,
                        extractor_identity=method.identity,
                        text=None,
                        word_count=None,
                        table_count=None,
                        image_count=None,
                        substantive_text_digest=None,
                        survives_blank_text_filter=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            samples.append(
                ExtractionSample(
                    source_id=document.source_id,
                    crawl_id=document.crawl_id,
                    url=document.url,
                    selection_reason=document.selection_reason,
                    method=method.name,
                    extractor_identity=method.identity,
                    text=extracted.text,
                    word_count=extracted.word_count,
                    table_count=extracted.table_count,
                    image_count=extracted.image_count,
                    substantive_text_digest=_substantive_text_digest(extracted.language_blocks),
                    survives_blank_text_filter=bool(extracted.text.strip()),
                    error=None,
                )
            )
    return tuple(samples)


def _fenced_text(text: str) -> str:
    longest_run = max((len(match.group()) for match in _BACKTICK_RUN.finditer(text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{text}\n{fence}"


def _diagnostic_markdown(samples: tuple[ExtractionSample, ...]) -> list[str]:
    by_method: dict[str, dict[str, ExtractionSample]] = {}
    for sample in samples:
        by_method.setdefault(sample.method, {})[sample.source_id] = sample

    lines = [
        "## Diagnostics",
        "",
        "Retention below models the normalizer's missing-or-blank `text` filter. It does not rerun "
        "language identification or deduplication.",
        "",
        "| Method | Sampled | Extracted | Survives blank-text filter | Failed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    retained_by_method: dict[str, set[str]] = {}
    for method, method_samples in by_method.items():
        extracted = sum(sample.error is None for sample in method_samples.values())
        retained = {sample.source_id for sample in method_samples.values() if sample.survives_blank_text_filter}
        retained_by_method[method] = retained
        lines.append(
            f"| `{method}` | {len(method_samples)} | {extracted} | {len(retained)} | {len(method_samples) - extracted} |"
        )

    retained_sets = tuple(retained_by_method.values())
    common_ids = set.intersection(*retained_sets) if retained_sets else set()
    union_ids = set.union(*retained_sets) if retained_sets else set()
    retention_matches = all(ids == retained_sets[0] for ids in retained_sets[1:]) if retained_sets else True
    lines.extend(
        [
            "",
            "### Retained source-ID agreement",
            "",
            f"All methods retain the same source IDs: **{'yes' if retention_matches else 'no'}**. "
            f"Common retained IDs: {len(common_ids)}; union: {len(union_ids)}.",
        ]
    )
    if not retention_matches:
        lines.extend(["", "| Method | IDs not retained by every method |", "| --- | --- |"])
        for method, retained in retained_by_method.items():
            differing = sorted(union_ids - retained)
            lines.append(f"| `{method}` | {', '.join(f'`{source_id}`' for source_id in differing) or 'none'} |")

    lines.extend(
        [
            "",
            "### Factorial integrity",
            "",
            "Substantive prose excludes table blocks, image placeholders, common Markdown presentation markers, "
            "and whitespace-only differences.",
            "",
            "| Comparison | Comparable IDs | Matching prose | Mismatches |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for label, left_method, right_method in _FACTORIAL_COMPARISONS:
        if left_method not in by_method or right_method not in by_method:
            continue
        comparable_ids = sorted(retained_by_method[left_method] & retained_by_method[right_method])
        mismatches = [
            source_id
            for source_id in comparable_ids
            if by_method[left_method][source_id].substantive_text_digest
            != by_method[right_method][source_id].substantive_text_digest
        ]
        mismatch_text = ", ".join(f"`{source_id}`" for source_id in mismatches) or "none"
        lines.append(f"| {label} | {len(comparable_ids)} | {len(comparable_ids) - len(mismatches)} | {mismatch_text} |")
    lines.append("")
    return lines


def extraction_sample_markdown(
    samples: tuple[ExtractionSample, ...],
    *,
    maximum_characters: int,
) -> str:
    """Render extraction results grouped by source document for visual comparison."""
    if maximum_characters <= 0:
        raise ValueError("maximum_characters must be positive")
    lines = [
        "# DOCX extraction samples",
        "",
        "Every method below received the same prefetched payload for a document. "
        "Full text is retained in `samples.jsonl`.",
        "",
    ]
    lines.extend(_diagnostic_markdown(samples))
    current_source_id: str | None = None
    for sample in samples:
        if sample.source_id != current_source_id:
            current_source_id = sample.source_id
            lines.extend(
                [
                    f"## {sample.source_id}",
                    "",
                    f"- URL: {sample.url}",
                    f"- Crawl: `{sample.crawl_id}`",
                    f"- Selection: `{sample.selection_reason}`",
                    "",
                ]
            )
        lines.extend([f"### {sample.method}", "", f"Extractor identity: `{sample.extractor_identity}`", ""])
        if sample.error is not None:
            lines.extend([f"Extraction failed: `{sample.error}`", ""])
            continue
        assert sample.text is not None
        lines.extend(
            [
                f"Words: {sample.word_count}; tables: {sample.table_count}; images in source: {sample.image_count}.",
                "",
                _fenced_text(sample.text[:maximum_characters]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_extraction_samples(
    *,
    fetched_step_path: str,
    output_path: str,
    methods: tuple[ExtractionMethod, ...],
    sample_size: int,
    seed: int,
    maximum_report_characters: int,
) -> tuple[str, str]:
    """Read fetched shards, run matched extraction, and persist JSONL and Markdown."""
    documents = sample_fetched_documents(
        fetched_parquet_paths(fetched_step_path),
        sample_size=sample_size,
        seed=seed,
    )
    samples = extract_samples(documents, methods)
    fs, resolved_output = url_to_fs(output_path)
    fs.makedirs(resolved_output, exist_ok=True)
    jsonl_path = prefix_join(output_path, "samples.jsonl")
    markdown_path = prefix_join(output_path, "report.md")
    with fsspec.open(jsonl_path, "wt") as stream:
        for sample in samples:
            stream.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    with fsspec.open(markdown_path, "wt") as stream:
        stream.write(extraction_sample_markdown(samples, maximum_characters=maximum_report_characters))
    return markdown_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetched-step-path", required=True, help="Fetched stage root containing .artifact.json.")
    parser.add_argument("--output-path", required=True, help="Directory for report.md and samples.jsonl.")
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--extraction-method",
        action="append",
        choices=sorted(DOCX_EXTRACTION_METHODS),
        required=True,
        dest="extraction_methods",
    )
    parser.add_argument(
        "--maximum-report-characters",
        type=int,
        default=DEFAULT_MAXIMUM_REPORT_CHARACTERS,
        help="Maximum displayed characters per method; JSONL retains the full text.",
    )
    args = parser.parse_args()
    configure_logging(logging.INFO)
    markdown_path, jsonl_path = write_extraction_samples(
        fetched_step_path=args.fetched_step_path,
        output_path=args.output_path,
        methods=extraction_methods(args.extraction_methods),
        sample_size=args.sample_size,
        seed=args.seed,
        maximum_report_characters=args.maximum_report_characters,
    )
    print(f"Markdown comparison: {markdown_path}")
    print(f"Full extraction samples: {jsonl_path}")


if __name__ == "__main__":
    main()
