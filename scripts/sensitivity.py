"""Incremental gates 3/4: one real request, one or all eight W4 groups."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.data import save_json
from quantization_sensitivity.metrics import cached_continuation_logits, continuation_logits, summarize_kl, token_kl
from quantization_sensitivity.model import encode, generate, load_model, metadata
from quantization_sensitivity.quantization import QUANTIZATION, quantized_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, nargs="+", default=[0])
    p.add_argument("--repeat-check", action="store_true")
    p.add_argument("--baseline", default=f"{DEFAULT_RESULTS}/steps/baseline/baseline.json")
    p.add_argument("--out", default=f"{DEFAULT_RESULTS}/steps/single_sensitivity")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    saved = json.loads(Path(args.baseline).read_text())
    request, demos = saved["request"], saved["demos"]
    model, tokenizer = load_model()
    _, ids = encode(tokenizer, request, demos, "a")
    assert ids[0].cpu().tolist() == saved["input_ids"]
    bf16 = generate(model, tokenizer, ids, request)
    assert bf16["generated_ids"] == saved["result"]["generated_ids"]
    reference = continuation_logits(model, ids, bf16["generated_ids"])
    torch.save(reference, out / "reference.pt")
    diagnostics = dict(reference_self_kl=token_kl(reference, reference, "cuda"))
    assert max(map(abs, diagnostics["reference_self_kl"])) == 0
    if args.repeat_check:
        repeated = continuation_logits(model, ids, bf16["generated_ids"])
        assert torch.equal(reference, repeated)
        diagnostics["independent_BF16_replay_exact"] = True
        cached = cached_continuation_logits(model, ids, bf16["generated_ids"])
        diagnostics["cached_vs_full_kl"] = summarize_kl(token_kl(reference, cached, "cuda"))
        diagnostics["cached_vs_full_argmax_agreement"] = (reference.argmax(-1) == cached.argmax(-1)).float().mean().item()
    rows = []
    for group in args.groups:
        with quantized_group(model, group) as audit:
            q_logits = continuation_logits(model, ids, bf16["generated_ids"])
            values = token_kl(reference, q_logits, "cuda")
            answer = generate(model, tokenizer, ids, request)
            if args.repeat_check:
                repeated = continuation_logits(model, ids, bf16["generated_ids"])
                assert torch.equal(q_logits, repeated)
                diagnostics[f"g{group}_independent_W4_replay_exact"] = True
                diagnostics[f"g{group}_repeat_mean_discrepancy"] = abs(
                    sum(values) / len(values) - sum(token_kl(reference, repeated, "cuda")) / len(values))
        restored = continuation_logits(model, ids, bf16["generated_ids"])
        assert torch.equal(reference, restored)
        diagnostics[f"g{group}_BF16_restore_logits_exact"] = True
        row = dict(request_id=request["request_id"], dataset=request["dataset"], group=group,
                   wording="a", input_tokens=ids.shape[1], gold=request["gold"],
                   bf16_answer=bf16["answer"], bf16_correct=bf16["correct"],
                   **summarize_kl(values), **answer)
        rows.append(row)
        save_json(out / f"group_{group}.json", dict(measurement=row, quantization_audit=audit))
        print(f"g{group}: KL={row['kl_mean']:.8f}, correct={row['correct']}, "
              f"restoration=exact", flush=True)
    frame = pd.DataFrame(rows)
    if len(set(args.groups)) == 8:
        frame["sensitivity_rank"] = frame.kl_mean.rank(ascending=False, method="average")
    for col in ["token_kl", "generated_ids"]:
        frame[col] = frame[col].map(json.dumps)
    frame.to_csv(out / "measurements.csv", index=False)
    save_json(out / "diagnostics.json", diagnostics)
    save_json(out / "metadata.json", dict(metadata(), quantization=QUANTIZATION,
              args=vars(args), total_seconds=time.monotonic() - start))
    print(f"PASS {len(rows)} measured configurations", flush=True)


if __name__ == "__main__":
    main()
