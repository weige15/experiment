"""Incremental gate 2: prove one W4 group changes only its 28 matrices and restores."""
import argparse
import json
from pathlib import Path

from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.data import save_json
from quantization_sensitivity.model import check_precision, encode, generate, load_model, metadata
from quantization_sensitivity.quantization import QUANTIZATION, fingerprints, group_parameters, quantized_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--baseline", default=f"{DEFAULT_RESULTS}/steps/baseline/baseline.json")
    p.add_argument("--out", default=f"{DEFAULT_RESULTS}/steps/one_group")
    args = p.parse_args()
    out = Path(args.out)
    saved = json.loads(Path(args.baseline).read_text())
    request, demos = saved["request"], saved["demos"]
    model, tokenizer = load_model()
    prompt, ids = encode(tokenizer, request, demos, "a")
    assert ids[0].cpu().tolist() == saved["input_ids"]
    bf16 = generate(model, tokenizer, ids, request)
    assert bf16["generated_ids"] == saved["result"]["generated_ids"]
    before = fingerprints(model)
    with quantized_group(model, args.group) as audit:
        during = fingerprints(model)
        changed = {k for k in before if before[k] != during[k]}
        assert changed == set(group_parameters(model, args.group)), changed
        precision = check_precision(model, ids)
        result = generate(model, tokenizer, ids, request)
    after = fingerprints(model)
    assert before == after
    restored = generate(model, tokenizer, ids, request)
    assert restored["generated_ids"] == bf16["generated_ids"]
    save_json(out / "result.json", dict(group=args.group, bf16=bf16, quantized=result,
              exact_restoration=True, exact_baseline_reproduction=True,
              changed_parameter_names=sorted(changed), quantization_audit=audit,
              before_sha256=before, during_sha256=during, after_sha256=after))
    save_json(out / "metadata.json", dict(metadata(), quantization=QUANTIZATION,
              precision_check=precision, args=vars(args)))
    print(f"PASS group={args.group}, changed={len(changed)}, restored=exact, "
          f"BF16 correct={bf16['correct']}, W4 correct={result['correct']}", flush=True)
    print(result["answer"], flush=True)


if __name__ == "__main__":
    main()
