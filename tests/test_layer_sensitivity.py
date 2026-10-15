"""Tests for actual single-layer semantics; toy models are tests, never model evidence."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from layer_sensitivity import runner
from layer_sensitivity.quantization import SUFFIXES, layer_parameters, quantized_layers
from layer_sensitivity.validate import check_audit, check_coverage


def toy_model():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    blocks = []
    for _ in range(32):
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.mlp = torch.nn.Module()
        for suffix in SUFFIXES:
            prefix, name = suffix.split(".")
            setattr(getattr(block, prefix), name, torch.nn.Linear(128, 4, bias=False, dtype=torch.bfloat16))
        blocks.append(block)
    model.model.layers = torch.nn.ModuleList(blocks)
    model.lm_head = torch.nn.Linear(128, 4, bias=False, dtype=torch.bfloat16)
    return model


def test_every_individual_layer_and_exception_restore():
    model = toy_model()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    for layer in range(32):
        with quantized_layers(model, [layer]) as audit:
            changed = {name for name, p in model.named_parameters() if not torch.equal(p, original[name])}
            assert changed == set(layer_parameters(model, layer)) and len(changed) == len(audit) == 7
        assert all(torch.equal(p, original[name]) for name, p in model.named_parameters())
    with pytest.raises(RuntimeError, match="interrupted"):
        with quantized_layers(model, [0, 31]):
            raise RuntimeError("interrupted")
    assert all(torch.equal(p, original[name]) for name, p in model.named_parameters())
    for invalid in [-1, 32, True, 1.0]:
        with pytest.raises(ValueError):
            layer_parameters(model, invalid)
    with pytest.raises(ValueError):
        with quantized_layers(model, [1, 1]):
            pass


def test_coverage_rejects_grouped_missing_and_duplicate_measurements():
    frame = pd.DataFrame([dict(request_id=r, layer=i, kl_mean=.01) for r in ["a", "b"] for i in range(32)])
    check_coverage(frame, ["a", "b"])
    for bad in [frame.iloc[:-1], frame[frame.layer < 8], pd.concat([frame.iloc[:-1], frame.iloc[[0]]])]:
        with pytest.raises(AssertionError):
            check_coverage(bad, ["a", "b"])
    bad = frame.copy()
    bad.loc[0, "kl_mean"] = np.nan
    with pytest.raises(AssertionError):
        check_coverage(bad, ["a", "b"])


def test_quantization_audit_rejects_wrong_layer_or_group():
    audit = [dict(name=f"model.layers.7.{s}.weight", numel=n, mse=.001, changed_elements=1)
             for s, n in zip(SUFFIXES, [16777216, 4194304, 4194304, 16777216, 58720256, 58720256, 58720256])]
    check_audit(audit, 7)
    for bad, layer in [(audit, 6), (audit * 4, 7), (audit[:-1], 7)]:
        with pytest.raises(AssertionError):
            check_audit(bad, layer)


def test_precision_bank_changes_joint_complement_and_restores():
    from layer_sensitivity.precision_allocation_check import PrecisionBank
    model = toy_model()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    bank = PrecisionBank(model)
    assert all(torch.equal(p, original[name]) for name, p in model.named_parameters())
    for high in [list(range(8)), list(range(24, 32)), list(range(0, 32, 4))]:
        state = bank.set_high_precision(high)
        changed = {name for name, p in model.named_parameters() if not torch.equal(p, original[name])}
        expected = {name for layer in range(32) if layer not in high for name in layer_parameters(model, layer)}
        assert changed == expected and len(state["quantized_layers"]) == 24
    bank.restore_all()
    assert not bank.active
    assert all(torch.equal(p, original[name]) for name, p in model.named_parameters())


def test_allocation_choices_have_fixed_budget_and_deterministic_ties():
    from layer_sensitivity.selection import random_sets, top_layers
    assert top_layers(np.ones(32)) == list(range(8))
    assert top_layers(np.arange(32)) == list(range(24, 32))
    assert random_sets() == random_sets()
    assert all(len(set(layers)) == 8 for layers in random_sets().values())
    with pytest.raises(ValueError):
        top_layers(np.zeros(8))


@pytest.mark.parametrize("changed", [False, True])
def test_layer_resume_rejects_changed_sources_before_gpu_load(tmp_path, monkeypatch, changed):
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps(dict(dataset=d, role="smoke", request_id=f"{d}:{i}")) + "\n"
                                for d in runner.SOURCES for i in range(8)))
    config = runner.configuration("smoke", requests)
    if changed:
        config["code_sha256"]["layer_sensitivity/quantization.py"] = "changed"
    runner.save_json(tmp_path / "run_config.json", config)
    monkeypatch.setattr(sys, "argv", ["runner", "--role", "smoke", "--requests", str(requests), "--out", str(tmp_path)])

    class Loaded(Exception):
        pass

    def load():
        raise Loaded

    monkeypatch.setattr(runner, "load_model", load)
    with pytest.raises(AssertionError if changed else Loaded):
        runner.main()
