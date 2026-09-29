from dataclasses import replace

import pytest
import torch

from pattern_aware.config import MODALITY_DIMENSIONS, MODALITY_ORDER, V1_CONFIG, V11_CONFIG, V2_CONFIG
from pattern_aware.models import FiveModalityConcatFusionEncoder, FiveTokenSelfAttentionFusionEncoder


@pytest.fixture(params=[V1_CONFIG, V11_CONFIG, V2_CONFIG], ids=["v1", "v11", "v2"])
def encoder_and_inputs(request):
    torch.manual_seed(13)
    config = replace(
        request.param,
        adapter_dimension=16,
        fusion_hidden_dimension=32,
        output_dimension=8,
        attention_feedforward_dimension=32,
    )
    encoder_type = (
        FiveTokenSelfAttentionFusionEncoder
        if request.param == V2_CONFIG else FiveModalityConcatFusionEncoder
    )
    model = encoder_type(config).eval()
    matrices = {name: torch.randn(3, MODALITY_DIMENSIONS[name]) for name in MODALITY_ORDER}
    masks = torch.tensor([[False] * 5, [True, False, True, False, False], [True] * 5])
    sources = torch.tensor([[False, False], [True, False], [True, True]])
    quality = {name: torch.rand(3, config.quality_dimension) for name in MODALITY_ORDER}
    return model, matrices, masks, sources, quality


def test_missing_modalities_have_zero_contribution_and_outputs_are_finite(encoder_and_inputs):
    model, matrices, masks, sources, quality = encoder_and_inputs
    embeddings, diagnostics = model(matrices, masks, sources, quality, return_diagnostics=True)
    assert torch.isfinite(embeddings).all()
    assert torch.equal(embeddings[0], torch.zeros_like(embeddings[0]))
    torch.testing.assert_close(torch.linalg.vector_norm(embeddings[1:], dim=1), torch.ones(2))
    assert torch.equal(diagnostics.embedding_available, torch.tensor([False, True, True]))
    assert torch.equal(diagnostics.gates[~masks], torch.zeros_like(diagnostics.gates[~masks]))
    assert torch.equal(
        diagnostics.final_tokens[~masks], torch.zeros_like(diagnostics.final_tokens[~masks])
    )
    assert torch.all((diagnostics.gates[masks] > 0) & (diagnostics.gates[masks] < 1))
    if diagnostics.attention_weights is not None:
        assert torch.isfinite(diagnostics.attention_weights).all()


def test_absent_content_values_cannot_change_embeddings(encoder_and_inputs):
    model, matrices, masks, sources, quality = encoder_and_inputs
    with torch.no_grad():
        expected = model(matrices, masks, sources, quality)
        altered = {name: values.clone() for name, values in matrices.items()}
        for position, name in enumerate(MODALITY_ORDER):
            altered[name][~masks[:, position]] = 100 * torch.randn_like(
                altered[name][~masks[:, position]]
            )
        actual = model(altered, masks, sources, quality)
    torch.testing.assert_close(actual, expected)


def test_all_missing_rows_do_not_poison_backward_pass(encoder_and_inputs):
    model, matrices, masks, sources, quality = encoder_and_inputs
    model.train()
    embeddings = model(matrices, masks, sources, quality)
    embeddings[:, 0].sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
