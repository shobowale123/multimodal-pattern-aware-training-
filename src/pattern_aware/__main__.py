"""CPU training and retrieval commands, plus the backward-compatible demo."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from .config import ContractError, ExperimentConfig
from .experiment import run_experiment


def _demo_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="JSON configuration; built-in defaults if omitted")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"), help="Output directory")
    parser.add_argument("--seed", type=int, help="Override the data and training seed")
    parser.add_argument("--epochs", type=int, help="Override maximum training epochs")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional figures")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save selected model state dictionaries")


def _demo(arguments: argparse.Namespace) -> None:
    config = ExperimentConfig.from_json(arguments.config) if arguments.config else ExperimentConfig()
    changes = {key: value for key, value in {"seed": arguments.seed, "epochs": arguments.epochs}.items()
               if value is not None}
    config = replace(config, training=replace(config.training, **changes),
                     plots=config.plots and not arguments.no_plots,
                     save_checkpoints=config.save_checkpoints or arguments.save_checkpoints)
    result = run_experiment(config, output=arguments.output)
    for model in result.summary["models"].values():
        print(f"{model['name']}: selected epoch {model['selected_epoch']}; "
              f"validation NDCG@30={model['validation_selection_ndcg_at_30']:.4f}")
    print(f"Synthetic experiment saved to {result.output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and use five-modality retrieval models on precomputed embeddings (CPU).",
        epilog="Legacy invocation without a subcommand runs the synthetic demo.",
    )
    _demo_options(parser)
    commands = parser.add_subparsers(dest="command")
    _demo_options(commands.add_parser("demo", help="Run the reproducible synthetic comparison"))
    validate = commands.add_parser("validate", help="Validate an external data manifest and aligned inputs")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--require-labels", action="store_true")
    train = commands.add_parser("train", help="Train on external inputs and save the validation-selected model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True, help="New run directory")
    encode = commands.add_parser("encode", help="Load a model bundle and encode unlabeled inputs")
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--data", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True, help="New embedding NPZ file")
    encode.add_argument("--batch-size", type=int, default=32)
    retrieve = commands.add_parser("retrieve", help="Rank a gallery using compatible saved embeddings")
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--gallery", type=Path, required=True)
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--output", type=Path, required=True, help="New ranking CSV; sibling report JSON")
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved model with held-out labeled inputs")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", type=Path, required=True, help="New summary JSON; sibling metric CSV files")
    evaluate.add_argument("--k-values", type=int, nargs="+", default=(1, 5, 10, 30))
    evaluate.add_argument("--batch-size", type=int, default=32)
    for command in (retrieve, evaluate):
        command.add_argument("--query-chunk-size", type=int, default=128)
        command.add_argument("--gallery-chunk-size", type=int, default=1024)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in (None, "demo"):
            _demo(arguments)
        elif arguments.command == "validate":
            from .io import load_dataset
            inputs = load_dataset(arguments.data, require_labels=arguments.require_labels)
            audit = {**inputs.audit(), "schema": inputs.schema.to_dict()}
            if hasattr(inputs, "splits"):
                audit["split_counts"] = dict(Counter(inputs.splits))
            print(json.dumps(audit, indent=2, allow_nan=False))
        elif arguments.command == "train":
            from .pipeline import train_from_manifest
            run = train_from_manifest(arguments.data, arguments.config, arguments.output)
            print(f"Selected epoch {run.selected_epoch}; validation metric {run.best_validation_metric:.6f}")
            print(f"Model bundle saved to {arguments.output / 'best'}")
        elif arguments.command == "encode":
            from .inference import encode_from_manifest
            artifact = encode_from_manifest(arguments.checkpoint, arguments.data, arguments.output,
                                            batch_size=arguments.batch_size)
            print(f"Encoded {len(artifact.record_ids)} records; {int(artifact.available.sum())} available")
        elif arguments.command == "retrieve":
            from .retrieval import retrieve_from_files
            result = retrieve_from_files(arguments.queries, arguments.gallery, arguments.output,
                                         top_k=arguments.top_k,
                                         query_chunk_size=arguments.query_chunk_size,
                                         gallery_chunk_size=arguments.gallery_chunk_size)
            print(json.dumps(dict(result.summary), indent=2, allow_nan=False))
        elif arguments.command == "evaluate":
            from .inference import evaluate_checkpoint
            result = evaluate_checkpoint(arguments.checkpoint, arguments.data, arguments.split, arguments.output,
                                         k_values=arguments.k_values, batch_size=arguments.batch_size,
                                         query_chunk_size=arguments.query_chunk_size,
                                         gallery_chunk_size=arguments.gallery_chunk_size)
            print(json.dumps(dict(result.summary), indent=2, allow_nan=False))
    except (ContractError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
