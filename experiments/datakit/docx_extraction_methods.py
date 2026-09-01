# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DOCX extraction methods used by downstream data-quality experiments."""

import io
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any

from marin.datakit.download.common_crawl_docx import (
    DoclingDocxExtractor,
    DocumentBlock,
    DocxExtractionError,
    ExtractedDocument,
)

_MARKDOWN_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}[ \t]+")
_MARKDOWN_LINK = re.compile(r"\[([^]\n]+)]\([^\n)]+\)")
_MARKDOWN_STRONG_ASTERISKS = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_MARKDOWN_STRONG_UNDERSCORES = re.compile(r"__(?=\S)(.+?)(?<=\S)__")
_MARKDOWN_STRIKETHROUGH = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_MARKDOWN_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+.!~|-])")
_METHOD_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class ExtractionMethod:
    """A stable experiment name paired with one payload-to-text function."""

    name: str
    revision: str
    function: Callable[[bytes], ExtractedDocument]

    def __post_init__(self) -> None:
        if _METHOD_NAME.fullmatch(self.name) is None:
            raise ValueError(f"Invalid extraction method name: {self.name!r}")

    @property
    def identity(self) -> str:
        return f"{self.name}-{self.revision}-{DoclingDocxExtractor().identity}"

    def extract(self, payload: bytes) -> ExtractedDocument:
        return self.function(payload)


def docling_default(payload: bytes) -> ExtractedDocument:
    """Use the stage-separation branch's production Docling extraction."""
    return DoclingDocxExtractor().extract(payload)


@cache
def _docling_converter() -> Any:
    # Docling is an optional datakit dependency, so importing this experiment
    # module must not require it until extraction is requested.
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.document_converter import DocumentConverter  # noqa: PLC0415

    return DocumentConverter(allowed_formats=[InputFormat.DOCX])


def _convert_docx(payload: bytes) -> Any:
    from docling.datamodel.base_models import DocumentStream  # noqa: PLC0415
    from docling.exceptions import ConversionError, SecurityError  # noqa: PLC0415

    try:
        result = _docling_converter().convert(DocumentStream(name="document.docx", stream=io.BytesIO(payload)))
    except (ConversionError, SecurityError) as error:
        raise DocxExtractionError("Docling failed to extract the DOCX payload") from error
    return result.document


def _table_markdown(document: Any) -> tuple[str, ...]:
    return tuple(table.export_to_markdown(document).strip() for table in document.tables)


def _inline_blocks(text: str, tables: tuple[str, ...]) -> tuple[DocumentBlock, ...]:
    """Split a Docling serialization into prose and table blocks without changing it."""
    blocks: list[DocumentBlock] = []
    cursor = 0
    for table in tables:
        table_start = text.find(table, cursor)
        if table_start < 0:
            raise ValueError("Docling table serialization was not present in document serialization")
        if prose := text[cursor:table_start].strip():
            blocks.append(DocumentBlock(text=prose, is_table=False))
        blocks.append(DocumentBlock(text=table, is_table=True))
        cursor = table_start + len(table)
    if prose := text[cursor:].strip():
        blocks.append(DocumentBlock(text=prose, is_table=False))
    return tuple(blocks)


def _extracted_document(
    document: Any,
    *,
    markdown: bool,
    tables_inline: bool,
    image_placeholder: str = "<!-- image -->",
) -> ExtractedDocument:
    from docling_core.types.doc.labels import DocItemLabel  # noqa: PLC0415

    tables = _table_markdown(document)

    def export(*, labels: set[Any] | None = None) -> str:
        if markdown:
            return document.export_to_markdown(labels=labels, image_placeholder=image_placeholder)
        return document.export_to_text(labels=labels)

    if tables_inline:
        blocks = _inline_blocks(export().strip(), tables)
    else:
        non_table_labels = set(DocItemLabel) - {DocItemLabel.TABLE}
        body = export(labels=non_table_labels).strip()
        body_blocks = (DocumentBlock(text=body, is_table=False),) if body else ()
        blocks = (*body_blocks, *(DocumentBlock(text=table, is_table=True) for table in tables))
    text = "\n\n".join(block.text for block in blocks)
    return ExtractedDocument(
        text=text,
        word_count=len(text.split()),
        table_count=len(document.tables),
        image_count=len(document.pictures),
        language_blocks=blocks,
    )


def docling_plain_text_inline_tables(payload: bytes) -> ExtractedDocument:
    """Serialize plain text with tables at their positions in reading order."""
    return _extracted_document(_convert_docx(payload), markdown=False, tables_inline=True)


def docling_plain_text_tables_at_end(payload: bytes) -> ExtractedDocument:
    """Serialize plain text with tables moved after all non-table content."""
    return _extracted_document(_convert_docx(payload), markdown=False, tables_inline=False)


def docling_markdown_inline_tables(payload: bytes) -> ExtractedDocument:
    """Serialize Markdown with tables at their positions in reading order."""
    return _extracted_document(_convert_docx(payload), markdown=True, tables_inline=True)


def docling_markdown_tables_at_end(payload: bytes) -> ExtractedDocument:
    """Serialize Markdown with tables moved after all non-table content."""
    return _extracted_document(_convert_docx(payload), markdown=True, tables_inline=False)


def docling_markdown_inline_tables_without_image_placeholders(payload: bytes) -> ExtractedDocument:
    """Serialize inline Markdown tables while omitting image placeholders."""
    return _extracted_document(
        _convert_docx(payload),
        markdown=True,
        tables_inline=True,
        image_placeholder="",
    )


def docling_markdown_tables_at_end_without_image_placeholders(payload: bytes) -> ExtractedDocument:
    """Serialize Markdown tables at the end while omitting image placeholders."""
    return _extracted_document(
        _convert_docx(payload),
        markdown=True,
        tables_inline=False,
        image_placeholder="",
    )


def _remove_markdown_markers(text: str) -> str:
    text = _MARKDOWN_HEADING.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _MARKDOWN_STRONG_ASTERISKS.sub(r"\1", text)
    text = _MARKDOWN_STRONG_UNDERSCORES.sub(r"\1", text)
    text = _MARKDOWN_STRIKETHROUGH.sub(r"\1", text)
    text = _MARKDOWN_INLINE_CODE.sub(r"\1", text)
    return _MARKDOWN_ESCAPE.sub(r"\1", text)


def docling_without_markdown_markers(payload: bytes) -> ExtractedDocument:
    """Run Docling and remove common Markdown presentation markers from its text."""
    extracted = docling_default(payload)
    blocks = tuple(
        DocumentBlock(text=_remove_markdown_markers(block.text), is_table=block.is_table)
        for block in extracted.language_blocks
    )
    text = "\n\n".join(block.text for block in blocks)
    return ExtractedDocument(
        text=text,
        word_count=len(text.split()),
        table_count=extracted.table_count,
        image_count=extracted.image_count,
        language_blocks=blocks,
    )


DOCX_EXTRACTION_METHODS = {
    "docling-default": ExtractionMethod("docling-default", "v1", docling_default),
    "docling-without-markdown-markers": ExtractionMethod(
        "docling-without-markdown-markers",
        "v1",
        docling_without_markdown_markers,
    ),
    "docling-plain-inline": ExtractionMethod(
        "docling-plain-inline",
        "v1",
        docling_plain_text_inline_tables,
    ),
    "docling-plain-tables-at-end": ExtractionMethod(
        "docling-plain-tables-at-end",
        "v1",
        docling_plain_text_tables_at_end,
    ),
    "docling-markdown-inline": ExtractionMethod(
        "docling-markdown-inline",
        "v1",
        docling_markdown_inline_tables,
    ),
    "docling-markdown-tables-at-end": ExtractionMethod(
        "docling-markdown-tables-at-end",
        "v1",
        docling_markdown_tables_at_end,
    ),
    "docling-markdown-inline-no-image-placeholders": ExtractionMethod(
        "docling-markdown-inline-no-image-placeholders",
        "v1",
        docling_markdown_inline_tables_without_image_placeholders,
    ),
    "docling-markdown-tables-at-end-no-image-placeholders": ExtractionMethod(
        "docling-markdown-tables-at-end-no-image-placeholders",
        "v1",
        docling_markdown_tables_at_end_without_image_placeholders,
    ),
}


def extraction_methods(names: Iterable[str]) -> tuple[ExtractionMethod, ...]:
    """Resolve CLI method names and reject unknown or repeated treatments."""
    selected_names = tuple(names)
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("Extraction method names must be unique")
    unknown = sorted(set(selected_names) - DOCX_EXTRACTION_METHODS.keys())
    if unknown:
        raise ValueError(f"Unknown DOCX extraction methods: {unknown}")
    return tuple(DOCX_EXTRACTION_METHODS[name] for name in selected_names)
