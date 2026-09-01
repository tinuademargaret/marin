# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluate matched models over normalized DOCX extraction variants.

Pass one normalized GCS directory per extraction treatment. The path must be the
normalizer's ``outputs/main`` directory, not the parent step directory::

    python -m experiments.datakit.common_crawl_docx_ablation \
      --normalized docling-default=gs://bucket/run/.../normalized_HASH/outputs/main \
      --normalized docling-without-markdown-markers=gs://bucket/run/.../normalized_HASH/outputs/main \
      --gpu-type H100 --region us-central1 \
      --wandb-entity my-team --wandb-project docx-ablation --wandb-group run-1 \
      --version dev

The default prints the lazy plan. Add ``--run`` to tokenize, train, and evaluate.
Every treatment uses the same model, optimizer, token budget, evaluation cadence,
and accelerator shape. Normalization still filters and deduplicates each treatment
independently; intersect ``source_id`` values first when the experiment must compare
representations over exactly the same documents rather than end-to-end pipeline yield.
"""

import re
from dataclasses import dataclass, replace

import click
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.data import tokenized
from marin.experiment.evaluation import EvalReport, eval_report, eval_steps
from marin.experiment.train import EvalSuite, train_lm

from experiments.evals.evals import core_evals
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_30m, llama_150m
from experiments.marin_tokenizer import marin_tokenizer

MODELS = {"30m": llama_30m, "150m": llama_150m}
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


def build(
    *,
    variants: tuple[NormalizedVariant, ...],
    model_size: str,
    gpu_type: str,
    gpu_count: int,
    region: str,
    batch_size: int,
    train_steps: int,
    evaluation_every: int,
    wandb_entity: str,
    wandb_project: str,
    wandb_group: str,
) -> dict[str, ArtifactStep[EvalReport]]:
    """Build matched tokenization, training, and post-training evaluation graphs."""
    model = MODELS[model_size]
    training_resources = ResourceConfig.with_gpu(
        gpu_type,
        count=gpu_count,
        cpu=16,
        disk="256G",
        ram="128G",
        regions=[region],
    )
    evaluation_accelerator = AcceleratorChoice(
        platform=Platform.GPU,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        region=region,
    )
    reports: dict[str, ArtifactStep[EvalReport]] = {}
    tokenization_resources = ResourceConfig.with_cpu(cpu=1, disk="32G", ram="10G", regions=[region])
    for variant in variants:
        dataset = tokenized(
            f"tokenized/docx-extraction-ablation/{variant.name}",
            tokenizer=marin_tokenizer,
            paths=(f"{variant.path}/**/*.parquet",),
            text_key="text",
            tags=("docx", "extraction-ablation", variant.name),
            resources=tokenization_resources,
            worker_resources=tokenization_resources,
        )
        checkpoint = train_lm(
            name=f"checkpoints/docx-extraction-ablation/{model_size}/{variant.name}",
            run_id=f"docx-{model_size}-{variant.name}",
            model=model,
            optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
            datasets={dataset: 1.0},
            batch_size=batch_size,
            seq_len=model.max_seq_len,
            num_train_steps=train_steps,
            z_loss_weight=None,
            evals=EvalSuite(CORE_TASKS, every=evaluation_every),
            steps_per_eval=evaluation_every,
            resources=training_resources,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_mode="online",
            tags=("docx", "extraction-ablation", variant.name, model_size),
        )
        evaluation_groups = tuple(
            replace(group, discover_latest_checkpoint=False)
            for group in core_evals(accelerator=evaluation_accelerator)
        )
        results = eval_steps(checkpoint, evaluation_groups)
        reports[variant.name] = eval_report(
            results,
            name=f"docx-extraction-ablation/{model_size}/{variant.name}",
        )
    return reports


@click.command(help=__doc__)
@click.option(
    "--normalized",
    "normalized_paths",
    multiple=True,
    required=True,
    metavar="METHOD=GS_PATH",
    help="Normalized outputs/main directory. Repeat once per extraction treatment.",
)
@click.option("--model-size", type=click.Choice(tuple(MODELS)), default="30m", show_default=True)
@click.option("--gpu-type", required=True, help="GPU variant advertised by the target Iris cluster, such as H100.")
@click.option("--gpu-count", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--region", required=True, help="Region containing both the normalized GCS data and compute.")
@click.option("--batch-size", type=click.IntRange(min=1), required=True)
@click.option("--train-steps", type=click.IntRange(min=1), required=True)
@click.option("--evaluation-every", type=click.IntRange(min=1), required=True)
@click.option("--wandb-entity", required=True, help="W&B user or team that owns the project.")
@click.option("--wandb-project", required=True, help="W&B project receiving all treatment runs.")
@click.option("--wandb-group", required=True, help="Shared W&B group for this extraction comparison.")
@build_options
def main(
    normalized_paths: tuple[str, ...],
    model_size: str,
    gpu_type: str,
    gpu_count: int,
    region: str,
    batch_size: int,
    train_steps: int,
    evaluation_every: int,
    wandb_entity: str,
    wandb_project: str,
    wandb_group: str,
) -> dict[str, ArtifactStep[EvalReport]]:
    return build(
        variants=normalized_variants(normalized_paths),
        model_size=model_size,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        region=region,
        batch_size=batch_size,
        train_steps=train_steps,
        evaluation_every=evaluation_every,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
    )


if __name__ == "__main__":
    main()
