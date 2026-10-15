import pytest
import torch

from m1_quant import pack_codes, quantize_blocks, unpack_codes


def test_known_quantizer_and_zero_block():
    x = torch.tensor([[0., 1., 3., 7.], [0., 0., 0., 0.]])
    q, packed, scale = quantize_blocks(x, block_size=4)
    assert torch.equal(x, q)
    assert torch.equal(scale, torch.ones(2, 1, 1))
    assert packed.numel() * 2 == x.numel()
    assert torch.isfinite(q).all()


def test_rounding_ties_and_negative_values():
    x = torch.tensor([[.5, 1.5, -2.5, 7.]])
    q, _, _ = quantize_blocks(x, 4)
    assert torch.equal(q, torch.tensor([[0., 2., -2., 7.]]))


def test_group_boundary_and_dtype():
    x = torch.tensor([[7., 3., 70., 30.]], dtype=torch.bfloat16)
    q, _, scale = quantize_blocks(x, 2)
    assert scale.flatten().tolist() == [1., 10.]
    assert q.dtype == torch.bfloat16
    assert torch.equal(q, x)
    with pytest.raises(ValueError):
        quantize_blocks(x, 3)


def test_full_signed_code_roundtrip():
    codes = torch.arange(-8, 8, dtype=torch.int8)
    packed = pack_codes(codes)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 8
    assert torch.equal(unpack_codes(packed, codes.shape), codes)
