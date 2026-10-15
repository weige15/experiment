"""Direct jointly quantized-model evaluation, not an additive accuracy prediction."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from quantization_sensitivity.data import save_json
from quantization_sensitivity.metrics import continuation_logits, summarize_kl, token_kl
from quantization_sensitivity.model import check_precision, generate, load_model, metadata
from quantization_sensitivity.quantization import fingerprints, quantize_weight
from quantization_sensitivity.runner import file_hash, key, read_json, tensor_hash, write_csv
from . import DEFAULT_RESULTS
from .quantization import LAYER_QUANTIZATION, layer_parameters
from .runner import source_hashes
from .selection import BUDGET, top_layers
from .validate import check_audit, check_coverage


class PrecisionBank:
    """Two CPU copies of the seven targeted matrices; the live model always computes BF16."""
    def __init__(self, model):
        self.model = model
        self.parameters = [layer_parameters(model, i) for i in range(32)]
        self.originals, self.quantized, self.audit = {}, {}, {}
        self.active = set()
        for layer, params in enumerate(self.parameters):
            self.audit[layer] = []
            for name, param in params.items():
                self.originals[name] = param.detach().cpu().clone()
                replacement, stats = quantize_weight(param)
                self.quantized[name] = replacement.cpu()
                self.audit[layer].append(dict(name=name, **stats))
            print(f"prepared layer={layer} Q/DQ CPU cache", flush=True)

    @torch.inference_mode()
    def set_high_precision(self, high_precision_layers):
        high = list(high_precision_layers)
        if len(high) != len(set(high)) or any(type(i) is not int or i not in range(32) for i in high):
            raise ValueError("unique integer layer indices required")
        target = set(range(32)) - set(high)
        changed = sorted(self.active ^ target)
        for layer in changed:
            source = self.quantized if layer in target else self.originals
            for name, param in self.parameters[layer].items():
                expected = source[name].to(param.device)
                param.copy_(expected)
                assert torch.equal(param, expected), f"installed weight differs: {name}"
            if layer in target:
                self.active.add(layer)
            else:
                self.active.remove(layer)
        assert self.active == target
        return dict(high_precision_layers=sorted(high), quantized_layers=sorted(target),
                    changed_layers=changed, copied_weights_exact=True)

    @torch.inference_mode()
    def restore_all(self):
        # Unconditional restoration also handles interruption halfway through a layer copy.
        for params in self.parameters:
            for name, param in params.items():
                expected = self.originals[name].to(param.device)
                param.copy_(expected)
                assert torch.equal(param, expected)
        self.active.clear()


def allocation_sources():
    return dict(source_hashes(), **{name: file_hash(name) for name in
                ["layer_sensitivity/selection.py", "layer_sensitivity/precision_allocation_check.py"]})


def strategies(selection, values):
    return dict(fixed=selection["fixed_high_precision_layers"],
                **selection["random_high_precision_layers"], request_oracle=top_layers(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    root = Path(args.root)
    run = root / "runs/evaluation"
    out = root / "precision_allocation_check"
    out.mkdir(parents=True, exist_ok=True)
    selection = read_json(root / "precision_allocation_selection.json")
    assert selection["budget"] == BUDGET and selection["source_sha256"] == source_hashes()
    frame = pd.read_csv(run / "measurements.csv")
    validation = read_json(root / "evaluation_validation.json")
    assert validation["status"] == "verified" and validation["measurements_sha256"] == file_hash(run / "measurements.csv")
    rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    requests = [r for r in rows if r["role"] == "evaluation"]
    check_coverage(frame, [r["request_id"] for r in requests])
    values = frame.pivot(index="request_id", columns="layer", values="kl_mean").reindex(columns=range(32))
    config = dict(source_sha256=allocation_sources(), selection_sha256=file_hash(root / "precision_allocation_selection.json"),
        evaluation_measurements_sha256=file_hash(run / "measurements.csv"),
        requests_sha256=file_hash(root / "requests.jsonl"), budget=BUDGET,
        quantization=dict(LAYER_QUANTIZATION, intervention="joint W4 complement of exactly 8 BF16 layers",
                          layers_per_intervention=24))
    if (out / "run_config.json").exists():
        assert read_json(out / "run_config.json") == config, "allocation inputs/code changed; new output required"
    else:
        save_json(out / "run_config.json", config)
    started = time.monotonic()
    model, tokenizer = load_model()
    save_json(out / f"attempt_{len(list(out.glob('attempt_*.json')))}.json", dict(metadata(),
        started_utc=datetime.now(timezone.utc).isoformat(), allocation_sources=allocation_sources(),
        run_config_sha256=file_hash(out / "run_config.json")))
    before = fingerprints(model)
    save_json(out / "checks/BF16_parameter_fingerprints.json", before)
    bank = PrecisionBank(model)
    assert fingerprints(model) == before, "preparing CPU cache must leave live BF16 model unchanged"
    for layer in range(32):
        check_audit(bank.audit[layer], layer)
        assert bank.audit[layer] == read_json(run / "quantization" / f"layer_{layer}.json")
        save_json(out / "quantization" / f"layer_{layer}.json", bank.audit[layer])
    try:
        for strategy in ["fixed", "random_0", "random_1", "random_2", "request_oracle"]:
            for i, request in enumerate(requests):
                path = out / strategy / f"{key(request)}.json"
                if path.exists():
                    continue
                base = read_json(run / "baselines" / f"{key(request)}.json")
                high = strategies(selection, values.loc[request["request_id"]].to_numpy())[strategy]
                assert len(high) == BUDGET
                state = bank.set_high_precision(high)
                ids = torch.tensor([base["input_ids"]], device="cuda")
                if i == 0:
                    during = fingerprints(model)
                    expected_names = {name for layer in state["quantized_layers"] for name in bank.parameters[layer]}
                    assert {name for name in before if before[name] != during[name]} == expected_names
                    save_json(out / "checks" / f"{strategy}_isolation.json", dict(
                        changed_parameter_names=sorted(expected_names), state=state,
                        precision=check_precision(model, ids), fingerprints=during))
                reference = torch.load(run / "reference_logits" / f"{key(request)}.pt", weights_only=True)
                assert tensor_hash(reference) == base["reference_logits_sha256"]
                joint = continuation_logits(model, ids, base["generated_ids"])
                kl = token_kl(reference, joint, "cuda")
                answer = generate(model, tokenizer, ids, request)
                additive = float(values.loc[request["request_id"], state["quantized_layers"]].sum())
                save_json(path, dict(request_id=request["request_id"], dataset=request["dataset"],
                    strategy=strategy, budget=BUDGET, gold=request["gold"], input_tokens=base["input_tokens"],
                    bf16_generated_ids=base["generated_ids"], bf16_answer=base["answer"], bf16_correct=base["correct"],
                    reference_logits_sha256=base["reference_logits_sha256"], state=state,
                    sum_individual_layer_kl=additive, **summarize_kl(kl), **answer))
                print(f"{strategy} {i+1}/{len(requests)} {request['request_id']} "
                      f"correct={answer['correct']} joint_KL={np.mean(kl):.7f}", flush=True)
    finally:
        bank.restore_all()
        assert fingerprints(model) == before, "full BF16 parameters not restored"
    for dataset in sorted({r["dataset"] for r in requests}):
        request = next(r for r in requests if r["dataset"] == dataset)
        base = read_json(run / "baselines" / f"{key(request)}.json")
        reference = torch.load(run / "reference_logits" / f"{key(request)}.pt", weights_only=True)
        restored = continuation_logits(model, torch.tensor([base["input_ids"]], device="cuda"), base["generated_ids"])
        assert torch.equal(reference, restored)
    save_json(out / "checks/restoration.json", dict(parameters_exact=True, sentinel_logits_exact=True))
    measurements = [read_json(out / s / f"{key(r)}.json") for s in
                    ["fixed", "random_0", "random_1", "random_2", "request_oracle"] for r in requests]
    write_csv(measurements, out / "measurements.csv")
    pd.DataFrame(measurements).to_parquet(out / "measurements.parquet", index=False)
    save_json(out / "summary.json", dict(status="complete", requests=len(requests), configurations=5,
        measurements=len(measurements), budget=BUDGET, seconds=time.monotonic() - started,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(f"PASS {len(measurements)} directly evaluated joint configurations", flush=True)


if __name__ == "__main__":
    main()
