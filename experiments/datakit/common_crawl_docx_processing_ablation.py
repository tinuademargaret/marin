# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build extraction treatments from an existing Common Crawl DOCX fetch.

The fetched input must be the stage root containing ``.artifact.json`` and a
``data`` child. Use its ``gs://`` path when data and compute are in the same
region. For cross-region data, use a ``mirror://`` source after confirming that
the copy is within the approved transfer budget. Every extraction treatment
reads the same input, then runs language identification and normalization
independently::

    uv run --package marin-core --extra datakit python \
      -m experiments.datakit.common_crawl_docx_processing_ablation \
      --fetched-input-path gs://bucket/run/fetched_HASH \
      --fetched-version 2026.09.02 \
      --crawl-id CC-MAIN-2026-34 \
      --paths-manifest-url https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/cc-index-table.paths.gz \
      --output-prefix gs://bucket/run/extraction-ablation \
      --extraction-method docling-plain-inline \
      --extraction-method docling-markdown-inline

Use ``--dry-run`` to inspect the graph without launching it.
"""

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial

from fray.types import ResourceConfig
from marin.datakit.download.common_crawl_docx import (
    COMMON_CRAWL_DOCX_SCHEMA,
    CommonCrawlDocxConfig,
    LinguaLanguageDetector,
    extract_common_crawl_docx,
    identify_common_crawl_docx_language,
)
from marin.datakit.download.common_crawl_plan import CommonCrawlIndexKind, CommonCrawlSource
from marin.datakit.normalize import DedupMode, NormalizedData, normalize_paths_step, normalize_step
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import prefix_join, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.docx_extraction_methods import (
    DOCX_EXTRACTION_METHODS,
    ExtractionMethod,
    extraction_methods,
)


@dataclass(frozen=True)
class ProcessingVariant:
    """Extraction, language-identification, and normalization steps for one treatment."""

    extraction: StepSpec
    language: StepSpec
    normalized: StepSpec


class ProcessingTerminalStage(StrEnum):
    """Last processing stage to materialize."""

    LANGUAGE = "language"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class LanguageArtifactInput:
    """One existing language-stage artifact assigned to an extraction method."""

    method: str
    version: str
    path: str


def language_artifact_input(value: str) -> LanguageArtifactInput:
    """Parse ``METHOD@VERSION=PATH`` into a normalization input."""
    method_and_version, separator, path = value.partition("=")
    method, version_separator, version = method_and_version.partition("@")
    if not separator or not version_separator or not method or not version or not path:
        raise argparse.ArgumentTypeError(f"{value!r} must have the form METHOD@VERSION=PATH")
    if method not in DOCX_EXTRACTION_METHODS:
        raise argparse.ArgumentTypeError(f"Unknown extraction method {method!r}")
    return LanguageArtifactInput(method=method, version=version, path=path.rstrip("/"))


def validated_fetched_input_path(fetched_input_path: str) -> str:
    """Validate and return a fetched stage root containing Parquet shards."""
    fetched_input_path = fetched_input_path.rstrip("/")
    input_glob = prefix_join(fetched_input_path, "data/**/*.parquet")
    fs, resolved = url_to_fs(input_glob)
    if not fs.glob(resolved):
        raise FileNotFoundError(f"No fetched Parquet shards match {input_glob}")
    return fetched_input_path


def processing_variants(
    config: CommonCrawlDocxConfig,
    *,
    fetched_input_path: str,
    fetched_version: str,
    methods: tuple[ExtractionMethod, ...],
    output_path_prefix: str | None,
) -> dict[str, ProcessingVariant]:
    """Build treatment graphs sharing one previously materialized fetched artifact."""
    fetched_input_path = validated_fetched_input_path(fetched_input_path)
    fetched_handle = ArtifactStep.adopt(
        f"inputs/common-crawl-docx-fetched/{config.name.lower()}",
        fetched_version,
        source=fetched_input_path,
        kind=Artifact,
    )
    fetched = replace(fetched_handle.lower(), output_path_prefix=output_path_prefix)
    detector = LinguaLanguageDetector()
    variants: dict[str, ProcessingVariant] = {}
    for method in methods:
        slug = f"{config.name.lower()}/{method.name}"
        extraction = StepSpec(
            name=f"docx-extraction-ablation/{slug}/extracted",
            fn=remote(
                partial(
                    extract_common_crawl_docx,
                    fetched_input_path=fetched_input_path,
                    config=config,
                    extractor=method,
                ),
                resources=ResourceConfig(cpu=1, ram="4g"),
                pip_dependency_groups=["datakit"],
            ),
            deps=[fetched],
            hash_attrs={
                "maximum_zip_entries": config.maximum_zip_entries,
                "maximum_uncompressed_bytes": config.maximum_uncompressed_bytes,
                "extractor": method.identity,
                "chunk_storage": "output-local",
                "schema_version": 5,
            },
            output_path_prefix=output_path_prefix,
        )
        language = StepSpec(
            name=f"docx-extraction-ablation/{slug}/language",
            fn=remote(
                partial(
                    identify_common_crawl_docx_language,
                    extracted_input_path=extraction.output_path,
                    config=config,
                    detector=detector,
                ),
                resources=ResourceConfig(cpu=1, ram="4g"),
                pip_dependency_groups=["datakit"],
            ),
            deps=[extraction],
            hash_attrs={
                "language_detector": detector.version,
                "chunk_chars": config.language_chunk_chars,
                "sample_chunks": config.language_sample_chunks,
                "minimum_alpha_bytes": config.language_minimum_alpha_bytes,
                "minimum_alpha_ratio": config.language_minimum_alpha_ratio,
                "maximum_table_alpha_bytes": config.language_maximum_table_alpha_bytes,
                "distribution_top_k": config.language_distribution_top_k,
                "minimum_score": config.language_minimum_score,
                "schema_version": 3,
            },
            output_path_prefix=output_path_prefix,
        )
        normalized = normalize_step(
            name=f"docx-extraction-ablation/{slug}/normalized",
            download=language,
            relative_input_path="data",
            file_extensions=(".parquet",),
            id_field="source_id",
            dedup_mode=DedupMode.EXACT,
            output_schema=COMMON_CRAWL_DOCX_SCHEMA,
            max_workers=config.max_workers,
            output_path_prefix=output_path_prefix,
        )
        variants[method.name] = ProcessingVariant(extraction, language, normalized)
    return variants


def processing_terminals(variants: dict[str, ProcessingVariant], stage: ProcessingTerminalStage) -> list[StepSpec]:
    """Select the terminal handles that materialize the requested stage."""
    if stage is ProcessingTerminalStage.LANGUAGE:
        return [variant.language for variant in variants.values()]
    return [variant.normalized for variant in variants.values()]


def combined_normalizations(
    crawl_id: str,
    *,
    inputs: tuple[LanguageArtifactInput, ...],
    output_path_prefix: str,
    max_workers: int,
) -> dict[str, StepSpec]:
    """Build one normalization per method over all of its language artifacts."""
    grouped: dict[str, list[LanguageArtifactInput]] = defaultdict(list)
    for language_input in inputs:
        grouped[language_input.method].append(language_input)

    normalized: dict[str, StepSpec] = {}
    for method, method_inputs in grouped.items():
        dependencies: list[StepSpec] = []
        data_paths: list[str] = []
        for index, language_input in enumerate(method_inputs):
            data_path = prefix_join(language_input.path, "data")
            fs, resolved = url_to_fs(prefix_join(data_path, "**/*.parquet"))
            if not fs.glob(resolved):
                raise FileNotFoundError(f"No language-labeled Parquet shards found under {data_path}")
            adopted = ArtifactStep.adopt(
                f"inputs/common-crawl-docx-language/{crawl_id.lower()}/{method}/{index}",
                language_input.version,
                source=language_input.path,
                kind=Artifact,
            )
            dependencies.append(replace(adopted.lower(), output_path_prefix=output_path_prefix))
            data_paths.append(data_path)

        normalized[method] = normalize_paths_step(
            name=f"docx-extraction-ablation/{crawl_id.lower()}/{method}/combined-normalized",
            downloads=tuple(dependencies),
            input_paths=tuple(data_paths),
            file_extensions=(".parquet",),
            id_field="source_id",
            dedup_mode=DedupMode.EXACT,
            output_schema=COMMON_CRAWL_DOCX_SCHEMA,
            max_workers=max_workers,
            output_path_prefix=output_path_prefix,
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetched-input-path",
        help="Fetched stage root containing .artifact.json and data/.",
    )
    parser.add_argument(
        "--fetched-version",
        help="Artifact version: YYYY.MM.DD[.N], dev, or <label>-dev.",
    )
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--paths-manifest-url", help="Manifest associated with the fetched crawl.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument(
        "--run-through",
        type=ProcessingTerminalStage,
        choices=tuple(ProcessingTerminalStage),
        default=ProcessingTerminalStage.NORMALIZED,
        help="Last stage to materialize from fetched data.",
    )
    parser.add_argument(
        "--language-input",
        action="append",
        type=language_artifact_input,
        default=[],
        help="Existing language artifact as METHOD@VERSION=PATH; repeat to normalize batches together.",
    )
    parser.add_argument(
        "--extraction-method",
        action="append",
        choices=sorted(DOCX_EXTRACTION_METHODS),
        dest="extraction_methods",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    language_inputs = tuple(args.language_input)
    if language_inputs:
        if args.run_through is not ProcessingTerminalStage.NORMALIZED:
            parser.error("--language-input can only be used with --run-through normalized")
        normalized = combined_normalizations(
            args.crawl_id,
            inputs=language_inputs,
            output_path_prefix=args.output_prefix,
            max_workers=args.max_workers,
        )
        StepRunner().run(list(normalized.values()), dry_run=args.dry_run, max_concurrent=args.max_concurrent)
        if not args.dry_run:
            for method, step in normalized.items():
                result = read_artifact(step.output_path, NormalizedData)
                print(f"{method} normalized data: {result.main_output_dir}")
        return

    if not args.fetched_input_path or not args.fetched_version or not args.paths_manifest_url:
        parser.error(
            "--fetched-input-path, --fetched-version, and --paths-manifest-url are required "
            "unless --language-input is provided"
        )
    if not args.extraction_methods:
        parser.error("--extraction-method is required unless --language-input is provided")

    config = CommonCrawlDocxConfig(
        name=args.crawl_id,
        sources=(
            CommonCrawlSource(
                crawl_id=args.crawl_id,
                index_kind=CommonCrawlIndexKind.MAIN,
                paths_manifest_url=args.paths_manifest_url,
            ),
        ),
        max_workers=args.max_workers,
    )
    variants = processing_variants(
        config,
        fetched_input_path=args.fetched_input_path,
        fetched_version=args.fetched_version,
        methods=extraction_methods(args.extraction_methods),
        output_path_prefix=args.output_prefix,
    )
    terminals = processing_terminals(variants, args.run_through)
    StepRunner().run(
        terminals,
        dry_run=args.dry_run,
        max_concurrent=args.max_concurrent,
    )
    if args.dry_run:
        return
    for method, variant in variants.items():
        if args.run_through is ProcessingTerminalStage.LANGUAGE:
            print(f"{method} language data: {prefix_join(variant.language.output_path, 'data')}")
        else:
            normalized = read_artifact(variant.normalized.output_path, NormalizedData)
            print(f"{method} normalized data: {normalized.main_output_dir}")


if __name__ == "__main__":
    main()
