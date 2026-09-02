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
    DOCLING_IMAGE_PLACEHOLDER,
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
    except (ConversionError, SecurityError, RuntimeError) as error:
        raise DocxExtractionError("Docling failed to extract the DOCX payload") from error
    return result.document


def _has_table_ancestor(item: Any, document: Any, table_type: type[Any]) -> bool:
    parent = item.parent
    while parent is not None:
        parent_item = parent.resolve(document)
        if isinstance(parent_item, table_type):
            return True
        parent = parent_item.parent
    return False


def _coalesce_blocks(blocks: Iterable[DocumentBlock]) -> tuple[DocumentBlock, ...]:
    coalesced: list[DocumentBlock] = []
    for block in blocks:
        if coalesced and coalesced[-1].is_table == block.is_table:
            previous = coalesced[-1]
            coalesced[-1] = DocumentBlock(text=f"{previous.text}\n\n{block.text}", is_table=block.is_table)
        else:
            coalesced.append(block)
    return tuple(coalesced)


def _structured_blocks(
    document: Any,
    *,
    markdown: bool,
    image_placeholder: str,
) -> tuple[DocumentBlock, ...]:
    """Serialize Docling items in reading order without inferring structure from text."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams  # noqa: PLC0415
    from docling_core.transforms.serializer.plain_text import PlainTextDocSerializer, PlainTextParams  # noqa: PLC0415
    from docling_core.types.doc.document import TableItem  # noqa: PLC0415

    if markdown:
        serializer = MarkdownDocSerializer(doc=document, params=MarkdownParams(image_placeholder=image_placeholder))
    else:
        serializer = PlainTextDocSerializer(doc=document, params=PlainTextParams(image_placeholder=image_placeholder))

    blocks: list[DocumentBlock] = []
    serialized_tables: set[str] = set()
    for item, _level in document.iterate_items():
        if _has_table_ancestor(item, document, TableItem):
            continue
        is_table = isinstance(item, TableItem)
        if text := serializer.serialize(item=item).text.strip():
            blocks.append(DocumentBlock(text=text, is_table=is_table))
        if is_table:
            serialized_tables.add(item.self_ref)

    for table in document.tables:
        if table.self_ref not in serialized_tables and (text := serializer.serialize(item=table).text.strip()):
            blocks.append(DocumentBlock(text=text, is_table=True))
    return tuple(blocks)


def _extracted_document(
    document: Any,
    *,
    markdown: bool,
    tables_inline: bool,
    image_placeholder: str = DOCLING_IMAGE_PLACEHOLDER,
) -> ExtractedDocument:
    blocks = _structured_blocks(document, markdown=markdown, image_placeholder=image_placeholder)
    if tables_inline:
        ordered_blocks = blocks
    else:
        body_blocks = tuple(block for block in blocks if not block.is_table)
        table_blocks = tuple(block for block in blocks if block.is_table)
        ordered_blocks = (*body_blocks, *table_blocks)
    language_blocks = _coalesce_blocks(ordered_blocks)
    text = "\n\n".join(block.text for block in language_blocks)
    return ExtractedDocument(
        text=text,
        word_count=len(text.split()),
        table_count=len(document.tables),
        image_count=len(document.pictures),
        language_blocks=language_blocks,
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


def docling_without_image_placeholders(payload: bytes) -> ExtractedDocument:
    """Use the default extraction and remove image position placeholders."""
    extracted = docling_default(payload)
    blocks = tuple(
        DocumentBlock(text=block.text.replace(DOCLING_IMAGE_PLACEHOLDER, "").strip(), is_table=block.is_table)
        for block in extracted.language_blocks
        if block.text.replace(DOCLING_IMAGE_PLACEHOLDER, "").strip()
    )
    text = "\n\n".join(block.text for block in blocks)
    return ExtractedDocument(
        text=text,
        word_count=len(text.split()),
        table_count=extracted.table_count,
        image_count=extracted.image_count,
        language_blocks=blocks,
    )


def remove_markdown_markers(text: str) -> str:
    """Remove common Markdown presentation markers without changing prose."""
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
        DocumentBlock(text=remove_markdown_markers(block.text), is_table=block.is_table)
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
    "docling-without-image-placeholders": ExtractionMethod(
        "docling-without-image-placeholders",
        "v1",
        docling_without_image_placeholders,
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
        "v2",
        docling_markdown_inline_tables,
    ),
    "docling-markdown-tables-at-end": ExtractionMethod(
        "docling-markdown-tables-at-end",
        "v2",
        docling_markdown_tables_at_end,
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
