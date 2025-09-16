import pytest
import torch
import torch.nn.functional as F
from foqus_ml_task.model import MRIEmbeddingModel


@pytest.mark.parametrize(
    "coils,H,W,embed_dim",
    [
        (4, 128, 128, 128),
        (8, 96, 80, 256),
    ],
)
def test_shape_and_norm(coils, H, W, embed_dim):
    B = 3
    C_in = 2 * coils  # real+imag per coil
    x = torch.randn(B, C_in, H, W)

    m = MRIEmbeddingModel(embed_dim=embed_dim, normalize=True, use_mlp_head=False)
    y = m(x)

    # shape check
    assert y.shape == (B, embed_dim)

    # L2-normalized embeddings by default
    norms = y.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_normalize_toggle():
    B, C_in, H, W = 2, 12, 128, 128
    x = torch.randn(B, C_in, H, W)
    m = MRIEmbeddingModel(embed_dim=64, normalize=False)

    # raw vs normalized outputs should match manual F.normalize
    y_raw = m(x, normalize=False)
    y_norm = m(x, normalize=True)
    assert torch.allclose(F.normalize(y_raw, p=2, dim=1), y_norm, atol=1e-6, rtol=0)


def test_gradients_flow():
    B, C_in, H, W = 2, 16, 128, 128
    x = torch.randn(B, C_in, H, W)
    m = MRIEmbeddingModel(embed_dim=32, normalize=False)

    loss = m(x).sum()
    loss.backward()

    total_grad = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert total_grad > 0.0, "No gradients flowed through the network."


def test_lazyconv_adapts_to_channels():
    B, C_in, H, W = 2, 10, 96, 96  # arbitrary, non-standard C_in
    x = torch.randn(B, C_in, H, W)
    m = MRIEmbeddingModel(embed_dim=16)

    _ = m(x)  # triggers LazyConv2d initialization
    assert hasattr(m.stem[0], "in_channels"), "Stem[0] is not a conv-like layer."
    assert m.stem[0].in_channels == C_in, "LazyConv2d did not adapt to input channels."
