from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from pattern_aware.config import TrainingConfig, V11_CONFIG
from pattern_aware.data import make_synthetic_dataset, split_inputs
from pattern_aware.models import FiveModalityConcatFusionEncoder
from pattern_aware.training import train_and_select


def test_two_epoch_training_selects_checkpoint_and_is_repeatable():
    inputs = make_synthetic_dataset(seed=13).inputs
    train = split_inputs(inputs, "train")
    validation = split_inputs(inputs, "validation")
    assert set(train.pattern_ids).isdisjoint(validation.pattern_ids)
    architecture = replace(
        V11_CONFIG, adapter_dimension=16, fusion_hidden_dimension=32, output_dimension=8
    )
    config = TrainingConfig(epochs=2, patience=2, seed=13)

    def factory():
        return FiveModalityConcatFusionEncoder(architecture)

    first = train_and_select("tiny", factory, train, validation, config)
    repeated = train_and_select("tiny", factory, train, validation, config)
    assert len(first.history) == 2
    assert np.isfinite(first.history["train_loss"]).all()
    assert first.selected_epoch in (1, 2)
    assert 0 <= first.best_validation_metric <= 1
    assert first.parameter_count > 0
    assert first.model.training is False
    selected_row = first.history.loc[first.history["epoch"].eq(first.selected_epoch)].iloc[0]
    assert first.best_validation_metric == pytest.approx(
        selected_row["validation_pattern_macro_ndcg_at_30"]
    )
    pd.testing.assert_frame_equal(first.history, repeated.history)
    for key, value in first.model.state_dict().items():
        torch.testing.assert_close(value, repeated.model.state_dict()[key], rtol=0, atol=0)
