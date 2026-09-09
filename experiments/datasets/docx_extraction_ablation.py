# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize private normalized DOCX extraction variants for ablation training.

Pass one normalizer ``outputs/main`` directory per extraction method. This
module creates private tokenized-cache artifacts only; it does not register the
datasets in the canonical Datakit source registry or publish token counts.
"""

import re
from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.data import tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer

_VARIANT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class NormalizedVariant:
    """One extraction treatment and its normalized Parquet directory."""

    name: str
    path: str

    def __post_init__(self) -> None:
        if _VARIANT_NAME.fullmatch(self.name) is None:
            raise ValueError(f"Invalid extraction method name: {self.name!r}")


def normalized_variants(values: tuple[str, ...]) -> tuple[NormalizedVariant, ...]:
    """Parse repeated ``METHOD=GCS_PATH`` arguments into unique treatments."""
    variants: list[NormalizedVariant] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path.startswith("gs://"):
            raise click.BadParameter(f"{value!r} must have the form METHOD=gs://BUCKET/PATH")
        try:
            variants.append(NormalizedVariant(name=name, path=path.rstrip("/")))
        except ValueError as error:
            raise click.BadParameter(str(error)) from error
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise click.BadParameter("Each extraction method may be specified only once")
    return tuple(variants)


def docx_extraction_datasets(
    variants: tuple[NormalizedVariant, ...],
    *,
    region: str,
    tokenizer: str = marin_tokenizer,
) -> dict[str, ArtifactStep[TokenizedCache]]:
    """Return one tokenized dataset handle per normalized extraction variant."""
    resources = ResourceConfig.with_cpu(cpu=1, disk="32G", ram="10G", regions=[region])
    return {
        variant.name: tokenized(
            f"tokenized/docx-extraction-ablation/{variant.name}",
            tokenizer=tokenizer,
            paths=(f"{variant.path}/**/*.parquet",),
            text_key="text",
            tags=("docx", "extraction-ablation", variant.name),
            resources=resources,
            worker_resources=resources,
        )
        for variant in variants
    }


@click.command(help=__doc__)
@click.option(
    "--normalized",
    "normalized_paths",
    multiple=True,
    required=True,
    metavar="METHOD=GS_PATH",
    help="Normalized outputs/main directory. Repeat once per extraction treatment.",
)
@click.option("--region", required=True, help="Region containing the normalized GCS data and compute.")
@click.option("--only", help="Comma-separated extraction methods to tokenize (default: all supplied methods).")
@build_options
def main(
    normalized_paths: tuple[str, ...],
    region: str,
    only: str | None,
) -> dict[str, ArtifactStep[TokenizedCache]]:
    datasets = docx_extraction_datasets(normalized_variants(normalized_paths), region=region)
    if only is None:
        return datasets

    names = only.split(",")
    unknown = [name for name in names if name not in datasets]
    if unknown:
        raise click.BadParameter(f"unknown methods {unknown}; available: {sorted(datasets)}", param_hint="--only")
    return {name: datasets[name] for name in names}


if __name__ == "__main__":
    main()
