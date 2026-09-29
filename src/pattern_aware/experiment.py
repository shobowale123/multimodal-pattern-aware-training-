"""Reproducible orchestration and portable outputs for the synthetic comparison."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import platform
import warnings

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, MODALITY_ORDER, V1_CONFIG, V11_CONFIG, V2_CONFIG
from .data import SyntheticDataset, make_synthetic_dataset, split_inputs, unit_rows
from .evaluation import evaluate_retrieval
from .models import FiveModalityConcatFusionEncoder, FiveTokenSelfAttentionFusionEncoder
from .sampling import PatternBalancedBatchSampler
from .training import TrainingRun, encode_all, encode_all_with_diagnostics, train_and_select


VARIANT_NAMES = {
    "v1": "V1 global gates",
    "v11": "V1.1 per-record quality gates",
    "v2": "V2 self-attention",
}


@dataclass
class ExperimentResult:
    summary: dict[str, Any]
    runs: dict[str, TrainingRun]
    dataset: SyntheticDataset
    gates: pd.DataFrame
    rankings: pd.DataFrame
    output_dir: Path


def mask_only_embeddings(content_masks: np.ndarray, source_masks: np.ndarray) -> np.ndarray:
    """A seven-feature negative control that ignores every learned content vector."""
    return unit_rows(np.concatenate([content_masks.astype("float32"),
                                     source_masks.astype("float32")], axis=1))


def _gate_parameters(run: TrainingRun, variant: str) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITY_ORDER:
        gate = run.model.reliability_gates[modality]
        if gate.network is None:
            weight = None
            bias = float(gate.bias.detach())
        else:
            weight = float(gate.network.weight.detach().reshape(-1)[0])
            bias = float(gate.network.bias.detach().reshape(-1)[0])
        rows.append({"model": variant, "modality": modality,
                     "quality_weight": weight, "bias": bias})
    return rows


def _gate_statistics(gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, modality), group in gates.groupby(["model", "modality"], sort=True):
        present = group.loc[group["available"]]
        values = present["gate"].to_numpy()
        quality = present["quality"].to_numpy()
        correlation = None
        if len(values) >= 2 and not np.isclose(values.std(), 0) and not np.isclose(quality.std(), 0):
            correlation = float(np.corrcoef(quality, values)[0, 1])
        rows.append({
            "model": model, "modality": modality, "available_records": len(values),
            "gate_mean": float(values.mean()) if len(values) else None,
            "gate_std": float(values.std()) if len(values) else None,
            "gate_min": float(values.min()) if len(values) else None,
            "gate_median": float(np.median(values)) if len(values) else None,
            "gate_max": float(values.max()) if len(values) else None,
            "quality_gate_correlation": correlation,
            "missing_rows_zero": bool(np.allclose(group.loc[~group["available"], "gate"], 0)),
        })
    return pd.DataFrame(rows)


def _plots(
    output: Path, runs: Mapping[str, TrainingRun], gates: pd.DataFrame,
    attention: tuple[np.ndarray, np.ndarray, str] | None,
) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is unavailable; skipping optional figures", stacklevel=2)
        return []

    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    files = []
    colors = {"v1": "#536879", "v11": "#128277", "v2": "#6955b6"}

    def save(figure: Any, filename: str) -> None:
        figure.tight_layout()
        figure.savefig(destination / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)
        files.append(f"figures/{filename}")

    for column, ylabel, title, filename in (
        ("train_loss", "Supervised contrastive loss", "Training objective", "training.png"),
        ("validation_pattern_macro_ndcg_at_30", "Pattern-macro NDCG@30",
         "Validation-only checkpoint selection", "validation_selection.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        for variant, run in runs.items():
            ax.plot(run.history["epoch"], run.history[column], marker="o", markersize=4,
                    color=colors[variant], label=VARIANT_NAMES[variant])
            if filename == "validation_selection.png":
                ax.axvline(run.selected_epoch, color=colors[variant], alpha=.25, linestyle="--")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False)
        save(fig, filename)

    for variant in runs:
        pivot = gates.loc[gates["model"].eq(variant)].pivot(
            index="record_id", columns="modality", values="gate").reindex(columns=MODALITY_ORDER)
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        chart = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_xticks(range(5), MODALITY_ORDER, rotation=25)
        ax.set_yticks(range(len(pivot)), pivot.index)
        ax.set(title=VARIANT_NAMES[variant], xlabel="Modality", ylabel="Synthetic validation record")
        fig.colorbar(chart, ax=ax, label="Learned gate")
        save(fig, f"gates_{variant}.png")

    if attention is not None:
        values, available, record_id = attention
        visible = np.ma.array(values, mask=~(available[:, None] & available[None, :]))
        fig, ax = plt.subplots(figsize=(6.5, 5.3))
        chart = ax.imshow(visible, cmap="viridis", vmin=0, vmax=max(float(visible.max()), .01))
        ax.set_xticks(range(5), MODALITY_ORDER, rotation=25)
        ax.set_yticks(range(5), MODALITY_ORDER)
        ax.set(title=f"V2 mean attention: {record_id}", xlabel="Key modality", ylabel="Query modality")
        fig.colorbar(chart, ax=ax, label="Mean attention across four heads")
        save(fig, "attention_v2.png")
    return files


def run_experiment(
    config: ExperimentConfig | Mapping[str, Any] | str | Path | None = None,
    output: str | Path = "artifacts/demo",
) -> ExperimentResult:
    """Train selected variants; evaluate each selected checkpoint once on the test split.

    The default CPU run uses the notebook's 48 synthetic records, three architectures,
    AdamW, and validation pattern-macro NDCG@30 selection. Files at ``output`` are
    replaced on subsequent runs; use separate directories to compare experiments.
    """
    if config is None:
        config = ExperimentConfig()
    elif isinstance(config, (str, Path)):
        config = ExperimentConfig.from_json(config)
    elif isinstance(config, Mapping):
        config = ExperimentConfig.from_dict(config)
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig, mapping, JSON path, or None")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    # Tiny matrix operations become needlessly expensive with many CPU worker threads.
    # Restore the caller's thread setting after the experiment.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        return _run(config, destination)
    finally:
        torch.set_num_threads(previous_threads)


def _run(config: ExperimentConfig, output: Path) -> ExperimentResult:
    dataset = make_synthetic_dataset(config.training.seed)
    split_data = {split: split_inputs(dataset.inputs, split)
                  for split in ("train", "validation", "test")}
    factories = {
        "v1": lambda: FiveModalityConcatFusionEncoder(V1_CONFIG),
        "v11": lambda: FiveModalityConcatFusionEncoder(V11_CONFIG),
        "v2": lambda: FiveTokenSelfAttentionFusionEncoder(V2_CONFIG),
    }
    runs: dict[str, TrainingRun] = {}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "data_kind": "synthetic_only",
        "config": config.to_dict(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "torch": str(torch.__version__),
                        "device": "cpu", "cpu_threads": config.cpu_threads,
                        "deterministic_algorithms": True},
        "data_audit": dict(dataset.inputs.audit()),
        "splits": {
            split: {"patterns": int(group["pattern_id"].nunique()), "records": int(len(group)),
                    "eligible_records": len(split_data[split].record_ids)}
            for split, group in dataset.cohort.groupby("split", sort=False)
        },
        "sampler": {},
        "models": {},
        "mask_only": {},
        "selection_metric": "validation pattern-macro NDCG@30",
        "all_missing_record_id": dataset.all_missing_record_id,
    }
    sampler = PatternBalancedBatchSampler(
        split_data["train"].pattern_ids, split_data["train"].eligible,
        patterns_per_batch=config.training.patterns_per_batch,
        members_per_pattern=config.training.members_per_pattern, seed=config.training.seed)
    from dataclasses import asdict
    summary["sampler"] = asdict(sampler.audit)
    gate_rows: list[dict[str, Any]] = []
    gate_parameters: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    query_tables, pattern_tables = [], []
    stored_vectors = {}
    attention = None

    # Complete training/selection for all variants before examining test outcomes.
    for variant in config.variants:
        train_data = split_data["train"]
        validation_data = split_data["validation"]
        if variant == "v1":
            train_data, validation_data = train_data.without_quality(), validation_data.without_quality()
        runs[variant] = train_and_select(VARIANT_NAMES[variant], factories[variant],
                                        train_data, validation_data, config.training)

    for variant, run in runs.items():
        model_summary = {
            "name": run.name, "parameter_count": run.parameter_count,
            "selected_epoch": run.selected_epoch, "epochs_executed": len(run.history),
            "validation_selection_ndcg_at_30": run.best_validation_metric,
        }
        for split in ("validation", "test"):
            data = split_data[split]
            if variant == "v1":
                data = data.without_quality()
            vectors = encode_all(run.model, data)
            stored_vectors[f"{variant}_{split}"] = vectors
            result = evaluate_retrieval(vectors, data.pattern_ids, record_ids=data.record_ids,
                                        available=data.eligible, k_values=config.k_values)
            model_summary[split] = dict(result.summary)
            query_tables.append(result.per_query.assign(model=variant, split=split))
            pattern_tables.append(result.per_pattern.assign(model=variant, split=split))
            if split == "validation":
                identifiers = np.asarray(data.record_ids)
                labels = np.asarray(data.pattern_ids)
                scores = unit_rows(vectors)[0] @ unit_rows(vectors).T
                order = np.lexsort((identifiers, -scores))
                order = order[order != 0][:8]
                for rank, index in enumerate(order, start=1):
                    ranking_rows.append({
                        "model": variant, "split": split,
                        "query_record_id": identifiers[0], "query_pattern_id": labels[0],
                        "rank": rank, "candidate_record_id": identifiers[index],
                        "candidate_pattern_id": labels[index], "cosine_similarity": float(scores[index]),
                        "relevant_same_pattern": bool(labels[index] == labels[0]),
                    })
        summary["models"][variant] = model_summary
        validation_data = split_data["validation"]
        model_data = validation_data.without_quality() if variant == "v1" else validation_data
        _, diagnostics = encode_all_with_diagnostics(run.model, model_data)
        gates = diagnostics.gates.cpu().numpy()
        for row, record_id in enumerate(validation_data.record_ids):
            for column, modality in enumerate(MODALITY_ORDER):
                gate_rows.append({
                    "model": variant, "record_id": record_id,
                    "pattern_id": validation_data.pattern_ids[row], "modality": modality,
                    "available": bool(validation_data.content_masks[row, column]),
                    "quality": float(validation_data.qualities[modality][row, 0]),
                    "gate": float(gates[row, column]),
                })
        gate_parameters.extend(_gate_parameters(run, variant))
        if diagnostics.attention_weights is not None:
            availability_counts = validation_data.content_masks.sum(axis=1)
            row = int(np.argmax(availability_counts))
            values = diagnostics.attention_weights[row].mean(dim=0).cpu().numpy()
            attention = (values, validation_data.content_masks[row], validation_data.record_ids[row])
            pd.DataFrame(values, index=MODALITY_ORDER, columns=MODALITY_ORDER).to_csv(
                output / "attention_v2.csv", index_label="query_modality")
            pd.DataFrame({
                "record_id": validation_data.record_ids[row],
                "modality": MODALITY_ORDER,
                "available": validation_data.content_masks[row],
                "quality": [float(validation_data.qualities[name][row, 0]) for name in MODALITY_ORDER],
                "gate": gates[row],
                "adapted_token_norm": torch.linalg.vector_norm(
                    diagnostics.adapted_tokens[row], dim=1).cpu().numpy(),
                "final_token_norm": torch.linalg.vector_norm(
                    diagnostics.final_tokens[row], dim=1).cpu().numpy(),
            }).to_csv(output / "attention_tokens.csv", index=False)
            summary["attention_example_record_id"] = validation_data.record_ids[row]
        if config.save_checkpoints:
            checkpoints = output / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            torch.save(run.model.state_dict(), checkpoints / f"{variant}.pt")

    for split in ("validation", "test"):
        data = split_data[split]
        vectors = mask_only_embeddings(data.content_masks, data.source_masks)
        result = evaluate_retrieval(vectors, data.pattern_ids, record_ids=data.record_ids,
                                    available=data.eligible, k_values=config.k_values)
        summary["mask_only"][split] = dict(result.summary)
        query_tables.append(result.per_query.assign(model="mask_only", split=split))
        pattern_tables.append(result.per_pattern.assign(model="mask_only", split=split))

    gates_frame, rankings_frame = pd.DataFrame(gate_rows), pd.DataFrame(ranking_rows)
    dataset.cohort.to_csv(output / "cohort.csv", index=False)
    pd.concat([run.history.assign(model=variant) for variant, run in runs.items()],
              ignore_index=True).to_csv(output / "history.csv", index=False)
    gates_frame.to_csv(output / "gates.csv", index=False)
    pd.DataFrame(gate_parameters).to_csv(output / "gate_parameters.csv", index=False)
    _gate_statistics(gates_frame).to_csv(output / "gate_statistics.csv", index=False)
    rankings_frame.to_csv(output / "rankings.csv", index=False)
    pd.concat(query_tables, ignore_index=True).to_csv(output / "per_query_metrics.csv", index=False)
    pd.concat(pattern_tables, ignore_index=True).to_csv(output / "per_pattern_metrics.csv", index=False)
    np.savez_compressed(output / "embeddings.npz", **stored_vectors)
    summary["figures"] = _plots(output, runs, gates_frame, attention) if config.plots else []
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return ExperimentResult(summary, runs, dataset, gates_frame, rankings_frame, output)
