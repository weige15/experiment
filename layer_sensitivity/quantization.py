"""Individual-layer intervention using the existing tested W4 integer-code Q/DQ."""
from contextlib import contextmanager

import torch

from quantization_sensitivity.quantization import QUANTIZATION, quantize_weight

SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LAYER_QUANTIZATION = {k: v for k, v in QUANTIZATION.items() if k != "layers_per_group"}
LAYER_QUANTIZATION.update(intervention="exactly one Transformer layer at a time",
                          layers_per_intervention=1, simulation=True,
                          group_axis="contiguous input features within each output row")


def layer_parameters(model, layer):
    if type(layer) is not int or layer not in range(32):
        raise ValueError("layer must be an integer in 0..31")
    block = model.model.layers[layer]
    found = {f"model.layers.{layer}.{name}.weight": module.weight
             for name, module in block.named_modules() if isinstance(module, torch.nn.Linear)}
    expected = {f"model.layers.{layer}.{suffix}.weight" for suffix in SUFFIXES}
    if set(found) != expected:
        raise ValueError(f"expected exactly seven named attention/MLP matrices in layer {layer}")
    return found


@contextmanager
def quantized_layers(model, layers):
    """Restore on errors too. The main scan passes exactly one layer, never a group."""
    layers = list(layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("nonempty unique layer indices required")
    params = {name: param for layer in layers for name, param in layer_parameters(model, layer).items()}
    originals = {name: p.detach().cpu().clone() for name, p in params.items()}
    audit = []
    try:
        with torch.no_grad():
            for name, param in params.items():
                replacement, stats = quantize_weight(param)
                if stats["changed_elements"] == 0 or stats["mse"] <= 0:
                    raise ValueError(f"quantization did not change {name}")
                param.copy_(replacement)
                audit.append(dict(name=name, **stats))
        yield audit
    finally:
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(originals[name])
                assert torch.equal(param.detach().cpu(), originals[name]), f"restore failed: {name}"
