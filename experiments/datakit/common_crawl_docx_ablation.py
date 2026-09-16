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

Pass ``--no-benchmarks`` to skip LM Harness and post-training Evalchemy benchmarks.
Training loss and Paloma validation losses are still logged to W&B.
"""

from dataclasses import replace

import click
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.evaluation.evalchemy.runner import EvalchemyRunConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ServeBackend, ServeConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.evaluation import EvalGroup, EvalReport, eval_report, eval_steps
from marin.experiment.train import EvalSuite, train_lm
from marin.training.training import LevanterCheckpoint

from experiments.datasets.docx_extraction_ablation import (
    NormalizedVariant,
    docx_extraction_datasets,
    normalized_variants,
)
from experiments.datasets.paloma import paloma_datasets
from experiments.evals.evals import wikitablequestions_eval
from experiments.llama import llama_30m, llama_150m
from experiments.marin_tokenizer import marin_tokenizer

MODELS = {"30m": llama_30m, "150m": llama_150m}
ABLATION_BENCHMARK_TASKS = (
    EvalTaskConfig("lambada_openai", 0),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    EvalTaskConfig("arc_easy", 10),
)


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
    benchmark_every: int | None,
    benchmark_max_examples: int,
    wikitablequestions_max_examples: int,
    benchmarks: bool,
    wandb_entity: str,
    wandb_project: str,
    wandb_group: str,
) -> dict[str, ArtifactStep[EvalReport] | ArtifactStep[LevanterCheckpoint]]:
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
    evaluation_serve = ServeConfig(backend=ServeBackend.LEVANTER)
    outputs: dict[str, ArtifactStep[EvalReport] | ArtifactStep[LevanterCheckpoint]] = {}
    datasets = docx_extraction_datasets(variants, region=region)
    validation = tuple(paloma_datasets(tokenizer=marin_tokenizer).values())
    if benchmarks and benchmark_every is None:
        raise ValueError("benchmark_every is required when benchmarks are enabled")
    for variant in variants:
        dataset = datasets[variant.name]
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
            evals=(
                EvalSuite(
                    ABLATION_BENCHMARK_TASKS,
                    every=benchmark_every,
                    max_examples=benchmark_max_examples,
                    run_initial=True,
                )
                if benchmarks
                else None
            ),
            validation=validation,
            steps_per_eval=evaluation_every,
            resources=training_resources,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_mode="online",
            tags=("docx", "extraction-ablation", variant.name, model_size),
        )
        if not benchmarks:
            outputs[variant.name] = checkpoint
            continue

        selected_benchmarks = EvalGroup(
            config=EvalchemyRunConfig(name="docx-selected", tasks=ABLATION_BENCHMARK_TASKS),
            serve=evaluation_serve,
            accelerator=evaluation_accelerator,
        )
        evaluation_groups = tuple(
            replace(group, serve=evaluation_serve, tokenizer=marin_tokenizer)
            for group in (
                selected_benchmarks,
                *wikitablequestions_eval(
                    accelerator=evaluation_accelerator,
                    max_eval_instances=wikitablequestions_max_examples,
                ),
            )
        )
        results = eval_steps(checkpoint, evaluation_groups)
        outputs[variant.name] = eval_report(
            results,
            name=f"docx-extraction-ablation/{model_size}/{variant.name}",
        )
    return outputs


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
@click.option(
    "--benchmark-every",
    type=click.IntRange(min=1),
    help="LM Harness interval. Required with --benchmarks; ignored with --no-benchmarks.",
)
@click.option("--benchmark-max-examples", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--wikitablequestions-max-examples", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option(
    "--benchmarks/--no-benchmarks",
    default=True,
    show_default=True,
    help="Run periodic LM Harness and post-training Evalchemy benchmarks.",
)
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
    benchmark_every: int | None,
    benchmark_max_examples: int,
    wikitablequestions_max_examples: int,
    benchmarks: bool,
    wandb_entity: str,
    wandb_project: str,
    wandb_group: str,
) -> dict[str, ArtifactStep[EvalReport] | ArtifactStep[LevanterCheckpoint]]:
    if benchmarks and benchmark_every is None:
        raise click.UsageError("--benchmark-every is required when benchmarks are enabled")
    return build(
        variants=normalized_variants(normalized_paths),
        model_size=model_size,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        region=region,
        batch_size=batch_size,
        train_steps=train_steps,
        evaluation_every=evaluation_every,
        benchmark_every=benchmark_every,
        benchmark_max_examples=benchmark_max_examples,
        wikitablequestions_max_examples=wikitablequestions_max_examples,
        benchmarks=benchmarks,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
    )


if __name__ == "__main__":
    main()
