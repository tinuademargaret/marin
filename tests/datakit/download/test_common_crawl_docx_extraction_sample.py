# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.download.common_crawl_docx import DocumentBlock, DocxExtractionError, ExtractedDocument

from experiments.datakit.common_crawl_docx_extraction_sample import (
    SampledFetchedDocument,
    extract_samples,
    extraction_sample_markdown,
    sample_fetched_documents,
)
from experiments.datakit.docx_extraction_methods import ExtractionMethod


def _write_fetched_shard(path: Path, *, source_id: str, payload: bytes) -> None:
    pq.write_table(
        pa.table(
            {
                "payload": [payload],
                "source_id": [source_id],
                "crawl_id": ["CC-MAIN-2026-34"],
                "url": [f"https://example.com/{source_id}.docx"],
                "selection_reason": ["declared_mime"],
            }
        ),
        path,
    )


def _method_a(payload: bytes) -> ExtractedDocument:
    text = f"A:{payload.decode()}"
    return ExtractedDocument(
        text=text,
        word_count=1,
        table_count=0,
        image_count=0,
        language_blocks=(DocumentBlock(text, False),),
    )


def _method_b(payload: bytes) -> ExtractedDocument:
    text = f"B:{payload.decode()}"
    return ExtractedDocument(
        text=text,
        word_count=1,
        table_count=0,
        image_count=0,
        language_blocks=(DocumentBlock(text, False),),
    )


def _tables_at_end(payload: bytes) -> ExtractedDocument:
    prose = f"Body: {payload.decode()}"
    return ExtractedDocument(
        text=f"{prose}\n\n| table |",
        word_count=4,
        table_count=1,
        image_count=0,
        language_blocks=(DocumentBlock(prose, False), DocumentBlock("| table |", True)),
    )


def _tables_inline(payload: bytes) -> ExtractedDocument:
    prose = f"Body: {payload.decode()}"
    return ExtractedDocument(
        text=f"| table |\n\n{prose}",
        word_count=4,
        table_count=1,
        image_count=0,
        language_blocks=(DocumentBlock("| table |", True), DocumentBlock(prose, False)),
    )


def _failed_method(_payload: bytes) -> ExtractedDocument:
    raise DocxExtractionError("bad document")


def test_sample_fetched_documents_spreads_selection_across_shards(tmp_path: Path) -> None:
    first = tmp_path / "part-00000.parquet"
    second = tmp_path / "part-00001.parquet"
    _write_fetched_shard(first, source_id="first", payload=b"first payload")
    _write_fetched_shard(second, source_id="second", payload=b"second payload")

    sampled = sample_fetched_documents((str(first), str(second)), sample_size=2, seed=7)

    assert {document.source_id for document in sampled} == {"first", "second"}


def test_extract_samples_compares_methods_on_identical_payloads() -> None:
    documents = (
        SampledFetchedDocument(
            source_id="document-1",
            crawl_id="CC-MAIN-2026-34",
            url="https://example.com/document-1.docx",
            selection_reason="declared_mime",
            payload=b"shared payload",
        ),
    )
    methods = (
        ExtractionMethod("method-a", "v1", _method_a),
        ExtractionMethod("method-b", "v1", _method_b),
    )

    samples = extract_samples(documents, methods)
    markdown = extraction_sample_markdown(samples, maximum_characters=1_000)

    assert [(sample.method, sample.text) for sample in samples] == [
        ("method-a", "A:shared payload"),
        ("method-b", "B:shared payload"),
    ]
    assert markdown.count("## document-1") == 1
    assert "A:shared payload" in markdown
    assert "B:shared payload" in markdown
    assert "All methods retain the same source IDs: **yes**" in markdown


def test_extraction_report_compares_substantive_text_across_table_placement() -> None:
    documents = (
        SampledFetchedDocument(
            source_id="document-1",
            crawl_id="CC-MAIN-2026-34",
            url="https://example.com/document-1.docx",
            selection_reason="declared_mime",
            payload=b"shared payload",
        ),
    )
    methods = (
        ExtractionMethod("docling-plain-inline", "v1", _tables_inline),
        ExtractionMethod("docling-plain-tables-at-end", "v1", _tables_at_end),
    )

    markdown = extraction_sample_markdown(extract_samples(documents, methods), maximum_characters=1_000)

    assert "| Plain table placement | 1 | 1 | none |" in markdown


def test_extraction_report_identifies_retained_source_id_disagreement() -> None:
    documents = (
        SampledFetchedDocument(
            source_id="document-1",
            crawl_id="CC-MAIN-2026-34",
            url="https://example.com/document-1.docx",
            selection_reason="declared_mime",
            payload=b"shared payload",
        ),
    )
    methods = (
        ExtractionMethod("method-a", "v1", _method_a),
        ExtractionMethod("failed-method", "v1", _failed_method),
    )

    markdown = extraction_sample_markdown(extract_samples(documents, methods), maximum_characters=1_000)

    assert "All methods retain the same source IDs: **no**" in markdown
    assert "| `failed-method` | `document-1` |" in markdown
