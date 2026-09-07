# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import pytest
from marin.execution.step_spec import StepSpec

from experiments.datakit.common_crawl_docx_processing_ablation import (
    LanguageArtifactInput,
    ProcessingTerminalStage,
    ProcessingVariant,
    combined_normalizations,
    language_artifact_input,
    processing_terminals,
    validated_fetched_input_path,
)


def test_validated_fetched_input_path_accepts_profile_fetch_root(tmp_path: Path) -> None:
    fetched = tmp_path / "fetched_deadbeef"
    data = fetched / "data"
    data.mkdir(parents=True)
    (data / "part-00000.parquet").touch()

    assert validated_fetched_input_path(str(fetched)) == str(fetched)


def test_language_artifact_input_parses_method_version_and_path() -> None:
    assert language_artifact_input("docling-default@2026.09.07=gs://bucket/language") == LanguageArtifactInput(
        method="docling-default",
        version="2026.09.07",
        path="gs://bucket/language",
    )


def test_language_artifact_input_rejects_missing_version() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        language_artifact_input("docling-default=gs://bucket/language")


def test_processing_terminals_stop_at_language_stage() -> None:
    extraction = StepSpec(name="extraction", fn=lambda _: None)
    language = StepSpec(name="language", fn=lambda _: None, deps=[extraction])
    normalized = StepSpec(name="normalized", fn=lambda _: None, deps=[language])
    variants = {"docling-default": ProcessingVariant(extraction, language, normalized)}

    assert processing_terminals(variants, ProcessingTerminalStage.LANGUAGE) == [language]
    assert processing_terminals(variants, ProcessingTerminalStage.NORMALIZED) == [normalized]


def test_combined_normalizations_groups_language_artifacts_by_method(tmp_path: Path) -> None:
    roots = (tmp_path / "old", tmp_path / "new")
    for root in roots:
        data = root / "data"
        data.mkdir(parents=True)
        (data / "part-00000.parquet").touch()

    steps = combined_normalizations(
        "CC-MAIN-2026-30",
        inputs=tuple(LanguageArtifactInput("docling-default", "2026.09.07", str(root)) for root in roots),
        output_path_prefix=str(tmp_path / "output"),
        max_workers=2,
    )

    assert set(steps) == {"docling-default"}
    assert len(steps["docling-default"].deps) == 2
