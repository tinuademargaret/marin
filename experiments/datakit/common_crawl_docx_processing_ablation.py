# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build extraction treatments from existing Common Crawl DOCX fetches.

The fetched parent must contain ``fetched_*`` stage directories with
``.artifact.json`` and ``data`` children. Use its ``gs://`` path when data and
compute are in the same region. For cross-region data, use a ``mirror://``
source after confirming that the copy is within the approved transfer budget.
Every extraction treatment reads the same resolved inputs, then runs language
identification and normalization independently::

    uv run --package marin-core --extra datakit python \
      -m experiments.datakit.common_crawl_docx_processing_ablation \
      --fetched-parent gs://bucket/run \
      --fetched-version 2026.09.02 \
      --crawl-id CC-MAIN-2026-34 \
      --paths-manifest-url https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/warc.paths.gz \
      --output-prefix gs://bucket/run/extraction-ablation \
      --extraction-method docling-plain-inline \
      --extraction-method docling-markdown-inline

Use ``--dry-run`` to inspect the graph without launching it.
"""

import argparse
import hashlib
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


def fetched_step_paths(fetched_parent: str) -> tuple[str, ...]:
    """Discover fetched stage roots containing artifact records under a parent."""
    artifact_pattern = prefix_join(fetched_parent.rstrip("/"), "fetched_*/.artifact.json")
    fs, resolved_pattern = url_to_fs(artifact_pattern)
    protocol = artifact_pattern.partition("://")[0] if "://" in artifact_pattern else ""
    artifact_paths = sorted(fs.glob(resolved_pattern))
    if not artifact_paths:
        raise FileNotFoundError(f"No fetched artifact directories matched {artifact_pattern}")
    suffix = "/.artifact.json"
    fetched_paths: list[str] = []
    for path in artifact_paths:
        root = path.removesuffix(suffix)
        fetched_path = f"{protocol}://{root}" if protocol else root
        data_pattern = prefix_join(fetched_path, "data/**/*.parquet")
        data_fs, resolved_data_pattern = url_to_fs(data_pattern)
        if not data_fs.glob(resolved_data_pattern):
            raise FileNotFoundError(f"No fetched Parquet shards matched {data_pattern}")
        fetched_paths.append(fetched_path)
    return tuple(fetched_paths)


def combined_fetched_input_path(fetched_step_paths: tuple[str, ...]) -> str:
    """Return one brace-expanded root for fetched stages under the same parent."""
    fetched_sources = tuple(sorted({path.rstrip("/") for path in fetched_step_paths}))
    parents = {source.rpartition("/")[0] for source in fetched_sources}
    if len(parents) != 1:
        raise ValueError("Fetched stage roots must share one parent directory")
    if len(fetched_sources) == 1:
        return fetched_sources[0]
    parent = parents.pop()
    basenames = (source.rpartition("/")[2] for source in fetched_sources)
    return f"{parent}/{{{','.join(basenames)}}}"


def processing_variants(
    config: CommonCrawlDocxConfig,
    *,
    fetched_step_paths: tuple[str, ...],
    fetched_version: str,
    methods: tuple[ExtractionMethod, ...],
    output_path_prefix: str | None,
) -> dict[str, ProcessingVariant]:
    """Build treatment graphs sharing previously materialized fetched artifacts."""
    fetched_sources = tuple(sorted({path.rstrip("/") for path in fetched_step_paths}))
    fetched_input_path = combined_fetched_input_path(fetched_sources)
    fetched_handles = tuple(
        ArtifactStep.adopt(
            f"inputs/common-crawl-docx-fetched/{config.name.lower()}/{hashlib.sha256(source.encode()).hexdigest()[:8]}",
            fetched_version,
            source=source,
            kind=Artifact,
        )
        for source in fetched_sources
    )
    fetched = [replace(handle.lower(), output_path_prefix=output_path_prefix) for handle in fetched_handles]
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
            deps=fetched,
            hash_attrs={
                "maximum_zip_entries": config.maximum_zip_entries,
                "maximum_uncompressed_bytes": config.maximum_uncompressed_bytes,
                "extractor": method.identity,
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
        "--fetched-parent",
        required=True,
        help="Parent containing disjoint fetched_* stage directories.",
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
    resolved_fetched_paths = fetched_step_paths(args.fetched_parent)
    print("Fetched inputs:")
    for path in resolved_fetched_paths:
        print(f"- {path}")
    variants = processing_variants(
        config,
        fetched_step_paths=resolved_fetched_paths,
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
