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
      --paths-manifest-url https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/warc.paths.gz \
      --output-prefix gs://bucket/run/extraction-ablation \
      --extraction-method docling-plain-inline \
      --extraction-method docling-markdown-inline

Use ``--dry-run`` to inspect the graph without launching it.
"""

import argparse
import logging
from dataclasses import dataclass, replace
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
from marin.datakit.normalize import DedupMode, NormalizedData, normalize_step
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetched-input-path",
        required=True,
        help="Fetched stage root containing .artifact.json and data/.",
    )
    parser.add_argument(
        "--fetched-version",
        required=True,
        help="Artifact version: YYYY.MM.DD[.N], dev, or <label>-dev.",
    )
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--paths-manifest-url", required=True, help="Manifest associated with the fetched crawl.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument(
        "--extraction-method",
        action="append",
        choices=sorted(DOCX_EXTRACTION_METHODS),
        required=True,
        dest="extraction_methods",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.INFO)
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
    StepRunner().run(
        [variant.normalized for variant in variants.values()],
        dry_run=args.dry_run,
        max_concurrent=args.max_concurrent,
    )
    if args.dry_run:
        return
    for method, variant in variants.items():
        normalized = read_artifact(variant.normalized.output_path, NormalizedData)
        print(f"{method} normalized data: {normalized.main_output_dir}")


if __name__ == "__main__":
    main()
