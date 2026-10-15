"""Real-model gates: exact single-layer isolation, then one request × all 32 layers."""
import argparse
from pathlib import Path
import time

import torch

from quantization_sensitivity.data import save_json
from quantization_sensitivity.metrics import continuation_logits, summarize_kl, token_kl
from quantization_sensitivity.model import check_precision, encode, generate, load_model, metadata
from quantization_sensitivity.quantization import fingerprints
from quantization_sensitivity.runner import file_hash, read_json, write_csv
from . import DEFAULT_RESULTS
from .quantization import LAYER_QUANTIZATION, layer_parameters, quantized_layers
from .runner import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["single_layer", "single_request"])
    parser.add_argument("--baseline", default=f"{DEFAULT_RESULTS}/gates/baseline/baseline.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise RuntimeError("gate output already exists: validate it or use a new directory")
    started = time.monotonic()
    saved = read_json(args.baseline)
    request, demos = saved["request"], saved["demos"]
    model, tokenizer = load_model()
    prompt, ids = encode(tokenizer, request, demos, "a")
    assert prompt == saved["prompt"] and ids[0].cpu().tolist() == saved["input_ids"]
    bf16 = generate(model, tokenizer, ids, request)
    assert bf16["generated_ids"] == saved["result"]["generated_ids"]
    reference = continuation_logits(model, ids, bf16["generated_ids"])
    assert torch.equal(reference, continuation_logits(model, ids, bf16["generated_ids"]))
    assert max(map(abs, token_kl(reference, reference, "cuda"))) == 0
    before = fingerprints(model) if args.stage == "single_layer" else None
    layers = [0] if args.stage == "single_layer" else list(range(32))
    measurements = []
    for layer in layers:
        with quantized_layers(model, [layer]) as audit:
            precision = check_precision(model, ids)
            if before:
                during = fingerprints(model)
                changed = {name for name in before if before[name] != during[name]}
                assert changed == set(layer_parameters(model, layer)) and len(changed) == 7
                save_json(out / "parameter_fingerprints.json", dict(before=before, during=during,
                                                                   changed_parameter_names=sorted(changed)))
            q_logits = continuation_logits(model, ids, bf16["generated_ids"])
            values = token_kl(reference, q_logits, "cuda")
            assert torch.equal(q_logits, continuation_logits(model, ids, bf16["generated_ids"]))
            assert max(values) > 1e-6, "intervention must change actual model probabilities"
            answer = generate(model, tokenizer, ids, request)
        assert torch.equal(reference, continuation_logits(model, ids, bf16["generated_ids"]))
        if before:
            after = fingerprints(model)
            assert before == after
            save_json(out / "restored_parameter_fingerprints.json", after)
        row = dict(request_id=request["request_id"], dataset=request["dataset"], layer=layer,
                   gold=request["gold"], input_tokens=ids.shape[1], bf16_answer=bf16["answer"],
                   bf16_correct=bf16["correct"], bf16_generated_ids=bf16["generated_ids"],
                   **summarize_kl(values), **answer)
        measurements.append(row)
        save_json(out / f"layer_{layer}.json", dict(measurement=row, quantization_audit=audit,
                  precision=precision, independent_W4_replay_exact=True, restored_BF16_logits_exact=True))
        print(f"PASS layer={layer}, changed_matrices={len(audit)}, KL={row['kl_mean']:.9f}, "
              f"correct={row['correct']}", flush=True)
    restored = generate(model, tokenizer, ids, request)
    assert restored["generated_ids"] == bf16["generated_ids"]
    write_csv(measurements, out / "measurements.csv")
    save_json(out / "metadata.json", dict(metadata(), measurement_sources=source_hashes(),
              gate_source_sha256=file_hash(__file__), quantization=LAYER_QUANTIZATION,
              baseline_sha256=file_hash(args.baseline), args=vars(args), seconds=time.monotonic() - started))
    save_json(out / "result.json", dict(status="passed", layers=layers, actual_model=True,
              BF16_replay_exact=True, BF16_generation_restoration_exact=True, self_kl_max=0,
              request_id=request["request_id"], n_measurements=len(measurements)))


if __name__ == "__main__":
    main()
