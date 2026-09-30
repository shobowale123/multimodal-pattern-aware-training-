from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from pattern_aware.checkpoints import load_checkpoint, save_checkpoint, create_model
from pattern_aware.config import ContractError, TrainingConfig, V1_CONFIG, V11_CONFIG, V2_CONFIG
from pattern_aware.data import make_synthetic_dataset
from pattern_aware.pipeline import split_provenance
from pattern_aware.training import encode_all


def make_bundle(tmp_path, variant="v11"):
    inputs = make_synthetic_dataset(seed=71).inputs
    config = replace({"v1": V1_CONFIG, "v11": V11_CONFIG, "v2": V2_CONFIG}[variant],
                     adapter_dimension=8, fusion_hidden_dimension=16, output_dimension=8,
                     attention_feedforward_dimension=16, input_dimensions=inputs.schema.dimensions)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        model = create_model(variant, config)
    bundle = save_checkpoint(tmp_path / "best", model, variant=variant, schema=inputs.schema,
                             training_config=TrainingConfig(epochs=1), selected_epoch=1,
                             validation_metric=.5, selection_k=5, provenance=split_provenance(inputs))
    return bundle, model, inputs


@pytest.mark.parametrize("variant", ["v1", "v11", "v2"])
def test_bundle_reconstructs_all_variants_and_preserves_rng(tmp_path, variant):
    bundle, original, inputs = make_bundle(tmp_path, variant)
    state = torch.random.get_rng_state().clone()
    loaded = load_checkpoint(bundle)
    assert torch.equal(state, torch.random.get_rng_state())
    assert loaded.schema == inputs.schema
    assert not loaded.model.training
    assert next(loaded.model.parameters()).device.type == "cpu"
    prepared = inputs.without_quality() if variant == "v1" else inputs
    np.testing.assert_array_equal(encode_all(original, prepared), encode_all(loaded.model, prepared))
    with pytest.raises(ContractError, match="already exists"):
        save_checkpoint(bundle, original, variant=variant, schema=inputs.schema,
                        training_config=TrainingConfig(), selected_epoch=1, validation_metric=.5,
                        selection_k=5, provenance=split_provenance(inputs))


@pytest.mark.parametrize("variant", ["v1", "v11", "v2"])
def test_checkpoint_fresh_process_embedding_parity(tmp_path, variant):
    bundle, original, inputs = make_bundle(tmp_path, variant)
    prepared = inputs.without_quality() if variant == "v1" else inputs
    expected = encode_all(original, prepared)
    script = """
import sys
import numpy as np
import torch
from pattern_aware.checkpoints import load_checkpoint
from pattern_aware.data import make_synthetic_dataset
from pattern_aware.training import encode_all
torch.set_num_threads(1)
loaded = load_checkpoint(sys.argv[1])
inputs = make_synthetic_dataset(seed=71).inputs
if loaded.metadata['variant'] == 'v1':
    inputs = inputs.without_quality()
np.save(sys.argv[2], encode_all(loaded.model, inputs))
"""
    target = tmp_path / "fresh.npy"
    result = subprocess.run([sys.executable, "-c", script, str(bundle), str(target)],
                            capture_output=True, text=True, env=os.environ.copy(), timeout=60)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(expected, np.load(target, allow_pickle=False))


@pytest.mark.parametrize("mutation", ["missing_key", "extra_key", "shape", "dtype", "nonfinite"])
def test_invalid_weight_state_is_rejected_even_with_updated_digest(tmp_path, mutation):
    bundle, _, _ = make_bundle(tmp_path)
    weights_path = bundle / "weights.pt"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    key = next(iter(state))
    if mutation == "missing_key":
        del state[key]
    elif mutation == "extra_key":
        state["unexpected"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = torch.zeros(1)
    elif mutation == "dtype":
        state[key] = state[key].double()
    else:
        state[key].view(-1)[0] = float("nan")
    torch.save(state, weights_path)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ContractError, match="keys|shape or dtype|finite"):
        load_checkpoint(bundle)


@pytest.mark.parametrize("mutation", ["version", "variant", "schema", "architecture", "selection", "provenance"])
def test_invalid_metadata_is_rejected(tmp_path, mutation):
    bundle, _, _ = make_bundle(tmp_path)
    path = bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    if mutation == "version":
        metadata["format_version"] = 999
    elif mutation == "variant":
        metadata["variant"] = "adaptive_v3"
    elif mutation == "schema":
        metadata["schema_sha256"] = "0" * 64
    elif mutation == "architecture":
        metadata["fusion_config"]["input_dimensions"]["location"] += 1
    elif mutation == "selection":
        metadata["selection"]["metric_value"] = float("nan")
    else:
        del metadata["provenance"]["split_hashes"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ContractError):
        load_checkpoint(bundle)


def test_corrupt_bytes_and_unsafe_pickle_are_rejected(tmp_path):
    bundle, _, _ = make_bundle(tmp_path)
    path = bundle / "weights.pt"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ContractError, match="digest mismatch"):
        load_checkpoint(bundle)
    # A plain unsupported global tests weights-only behavior without executing code.
    torch.save({"unexpected": Path("not-a-tensor")}, path)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ContractError):
        load_checkpoint(bundle)
