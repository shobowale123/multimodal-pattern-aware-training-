from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from pattern_aware.checkpoints import create_model, load_checkpoint
from pattern_aware.config import ContractError, TrainingConfig, V11_CONFIG
from pattern_aware.data import make_synthetic_dataset, split_inputs
from pattern_aware.pipeline import RunConfig, train_dataset
import pattern_aware.training as training


def tiny_config(**overrides):
    return RunConfig(training=TrainingConfig(epochs=2, patience=2, seed=37),
                     architecture={"adapter_dimension": 8, "fusion_hidden_dimension": 16,
                                   "output_dimension": 8, "attention_feedforward_dimension": 16},
                     selection_k=5, **overrides)


def test_training_changes_parameters_and_persists_selected_model(tmp_path):
    inputs = make_synthetic_dataset(seed=19).inputs
    config = tiny_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.training.seed)
        initial = create_model(config.variant, config.fusion_config(inputs)).state_dict()
    run = train_dataset(inputs, config, tmp_path / "run")
    loaded = load_checkpoint(tmp_path / "run" / "best")
    assert any(not torch.equal(initial[key], value) for key, value in run.model.state_dict().items())
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["test_evaluated"] is False
    assert summary["selection_k"] == 5
    assert loaded.metadata["selection"]["epoch"] == run.selected_epoch
    assert "validation_pattern_macro_ndcg_at_5" in run.history
    assert (tmp_path / "run" / "config.json").exists()
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "run" / "history.csv"), run.history)
    with pytest.raises(ContractError, match="already exists"):
        train_dataset(inputs, config, tmp_path / "run")


def test_test_features_cannot_change_selected_state(tmp_path):
    inputs = make_synthetic_dataset(seed=22).inputs
    positions = np.asarray(inputs.splits) == "test"
    matrices = {name: values.copy() for name, values in inputs.matrices.items()}
    for name, values in matrices.items():
        values[positions] *= -3.0
    altered = replace(inputs, matrices=matrices)
    first = train_dataset(inputs, tiny_config(), tmp_path / "first")
    second = train_dataset(altered, tiny_config(), tmp_path / "second")
    assert first.selected_epoch == second.selected_epoch
    pd.testing.assert_frame_equal(first.history, second.history)
    for key, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)


def test_pipeline_restores_global_rng_threads_and_determinism(tmp_path):
    inputs = make_synthetic_dataset(seed=7).inputs
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    train_dataset(inputs, tiny_config(variant="v1"), tmp_path / "run")
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_best_state_restored_after_forced_validation_decline(monkeypatch):
    inputs = make_synthetic_dataset(seed=27).inputs
    metrics = iter([.4, .8, .7, .6, .9])
    epoch = 0

    def fake_epoch_loss(model, *args, **kwargs):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            next(model.parameters()).fill_(float(epoch))
        return 1.0 / epoch

    def fake_validation(model, inputs, selection_k):
        return next(metrics), SimpleNamespace(summary={})

    monkeypatch.setattr(training, "epoch_loss", fake_epoch_loss)
    monkeypatch.setattr(training, "validation_selection_metric", fake_validation)
    architecture = replace(V11_CONFIG, adapter_dimension=8, fusion_hidden_dimension=16, output_dimension=8)
    run = training.train_and_select("controlled", lambda: create_model("v11", architecture),
                                     split_inputs(inputs, "train"), split_inputs(inputs, "validation"),
                                     TrainingConfig(epochs=5, patience=2), selection_k=5)
    assert run.selected_epoch == 2
    assert run.best_validation_metric == .8
    assert len(run.history) == 4
    assert torch.all(next(run.model.parameters()) == 2)


def test_training_preflight_fails_before_creating_output(tmp_path):
    inputs = make_synthetic_dataset(seed=11).inputs
    no_validation = inputs.subset(np.flatnonzero(np.asarray(inputs.splits) != "validation"))
    with pytest.raises(ContractError, match="validation"):
        train_dataset(no_validation, tiny_config(), tmp_path / "missing_validation")
    assert not (tmp_path / "missing_validation").exists()
    too_few = tiny_config()
    too_few = replace(too_few, training=replace(too_few.training, members_per_pattern=5))
    with pytest.raises(ContractError, match="two eligible"):
        train_dataset(inputs, too_few, tmp_path / "insufficient")
    assert not (tmp_path / "insufficient").exists()


@pytest.mark.parametrize("values", [{"variant": "bad"}, {"selection_k": True},
                                    {"cpu_threads": 0}, {"architecture": {"quality_dimension": 0}},
                                    {"training": {"learning_rate": float("nan")}},
                                    {"unknown": 1}])
def test_run_config_rejects_bad_contract(values):
    with pytest.raises(ContractError):
        RunConfig.from_dict(values)
