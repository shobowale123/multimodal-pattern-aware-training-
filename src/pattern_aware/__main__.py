"""Command-line entry point: python -m pattern_aware --help."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import ContractError, ExperimentConfig
from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fully synthetic multimodal retrieval simulation.")
    parser.add_argument("--config", type=Path, help="JSON configuration; built-in defaults if omitted")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"), help="Output directory")
    parser.add_argument("--seed", type=int, help="Override the data and training seed")
    parser.add_argument("--epochs", type=int, help="Override maximum training epochs")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional figures")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save selected model state dictionaries")
    arguments = parser.parse_args()
    try:
        config = ExperimentConfig.from_json(arguments.config) if arguments.config else ExperimentConfig()
        changes = {key: value for key, value in {"seed": arguments.seed, "epochs": arguments.epochs}.items()
                   if value is not None}
        config = replace(config, training=replace(config.training, **changes),
                         plots=config.plots and not arguments.no_plots,
                         save_checkpoints=config.save_checkpoints or arguments.save_checkpoints)
        result = run_experiment(config, output=arguments.output)
    except (ContractError, OSError, ValueError) as exc:
        parser.error(str(exc))
    for model in result.summary["models"].values():
        print(f"{model['name']}: selected epoch {model['selected_epoch']}; "
              f"validation NDCG@30={model['validation_selection_ndcg_at_30']:.4f}")
    print(f"Synthetic experiment saved to {result.output_dir}")


if __name__ == "__main__":
    main()
