"""Fixed W4 RTN Q/DQ: actual integer codes, packed-nibble round-trip, BF16 compute."""
from contextlib import contextmanager
import hashlib

import torch

QUANTIZATION = dict(method="symmetric_groupwise_RTN_QDQ", bits=4, block_size=128,
                    scale="FP32 absmax / 7; zero block uses scale 1",
                    codes="signed [-7,7]; round ties to even", clipping="none",
                    zero_point=0, compute_dtype="bfloat16", packed_runtime=False,
                    targets="all 7 attention/MLP linear weights per layer",
                    excluded="embeddings, lm_head, norms", layers_per_group=4)


def pack_codes(codes):
    flat = (codes.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
    if flat.numel() % 2:
        raise ValueError("even code count required")
    return flat[::2] | (flat[1::2] << 4)


def unpack_codes(packed, shape):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(shape)
    return (codes.to(torch.int16) - 8).to(torch.int8)


def quantize_blocks(weight, block_size=128):
    if weight.ndim != 2 or weight.shape[1] % block_size:
        raise ValueError("W4 requires a matrix with input dimension divisible by block size")
    blocks = weight.float().reshape(weight.shape[0], -1, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 7
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round(blocks / scale).clamp(-7, 7).to(torch.int8)
    packed = pack_codes(codes)
    restored = unpack_codes(packed, codes.shape)
    assert torch.equal(restored, codes)
    # Forward uses the decoded 4-bit codes, not the original BF16 weights.
    dequantized = (restored.float() * scale).reshape(weight.shape).to(weight.dtype)
    return dequantized, packed, scale


@torch.inference_mode()
def quantize_weight(weight):
    replacement = torch.empty_like(weight)
    mse_sum, changed = 0.0, 0
    digest = hashlib.sha256()
    for start in range(0, weight.shape[0], 128):
        source = weight[start:start + 128]
        q, packed, scale = quantize_blocks(source)
        replacement[start:start + 128].copy_(q)
        mse_sum += ((source.float() - q.float()) ** 2).sum().item()
        changed += (source != q).sum().item()
        digest.update(packed.cpu().numpy().tobytes())
        digest.update(scale.cpu().numpy().tobytes())
    return replacement, dict(numel=weight.numel(), changed_elements=changed,
                             mse=mse_sum / weight.numel(), packed_scale_sha256=digest.hexdigest())


def group_parameters(model, group):
    if group not in range(8):
        raise ValueError("group must be 0..7")
    found = {}
    for layer in range(4 * group, 4 * group + 4):
        for name, module in model.model.layers[layer].named_modules():
            if isinstance(module, torch.nn.Linear):
                found[f"model.layers.{layer}.{name}.weight"] = module.weight
    assert len(found) == 28, f"expected 4*7 matrices, got {len(found)}"
    return found


@contextmanager
def quantized_group(model, group):
    params = group_parameters(model, group)
    originals = {name: p.detach().cpu().clone() for name, p in params.items()}
    audit = []
    try:
        with torch.no_grad():
            for name, param in params.items():
                q, stats = quantize_weight(param)
                param.copy_(q)
                audit.append(dict(name=name, **stats))
        yield audit
    finally:
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(originals[name])
                assert torch.equal(param.detach().cpu(), originals[name]), f"restore failed: {name}"


def fingerprints(model):
    return {name: hashlib.sha256(p.detach().cpu().view(torch.uint16).numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}
