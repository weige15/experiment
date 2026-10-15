"""Run one role/wording shard, one request at a time, with immutable resumable records."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from m1_data import SOURCES, save_json
from m1_metrics import continuation_logits, summarize_kl, token_kl
from m1_model import check_precision, encode, generate, load_model, metadata
from m1_quant import QUANTIZATION, quantized_group

MEASUREMENT_SOURCES = ["m1_run.py", "m1_data.py", "m1_model.py", "m1_metrics.py", "m1_quant.py"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def key(request):
    return request["request_id"].replace(":", "_")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(rows, path):
    frame = pd.DataFrame(rows)
    for column in frame:
        if frame[column].map(lambda x: isinstance(x, (list, dict))).any():
            frame[column] = frame[column].map(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    frame.to_csv(path, index=False)


def export_run(out, requests):
    for group in range(8):
        assert read_json(out / "checks" / f"g{group}_restoration.json")["exact"]
    baselines = [read_json(out / "baselines" / f"{key(r)}.json") for r in requests]
    measurements = [read_json(out / f"g{g}" / f"{key(r)}.json") for r in requests for g in range(8)]
    frame = pd.DataFrame(measurements)
    assert len(frame) == 8 * len(requests)
    assert not frame.duplicated(["request_id", "group"]).any()
    assert np.isfinite(frame.kl_mean).all() and (frame.kl_mean >= -1e-6).all()
    frame["sensitivity_rank"] = frame.groupby("request_id").kl_mean.rank(ascending=False, method="average")
    write_csv(baselines, out / "baselines.csv")
    write_csv(frame.to_dict("records"), out / "measurements.csv")
    summary = {}
    for dataset in SOURCES:
        base = [b for b in baselines if b["dataset"] == dataset]
        quant = frame[frame.dataset == dataset]
        summary[dataset] = dict(n=len(base), bf16_correct=sum(b["correct"] for b in base),
            bf16_parse_failures=sum(b["parse_failure"] for b in base),
            bf16_truncated=sum(b["truncated"] for b in base),
            W4_correct_by_group=quant.groupby("group").correct.sum().to_dict(),
            W4_parse_failures=int(quant.parse_failure.sum()), W4_truncated=int(quant.truncated.sum()),
            KL_mean_by_group=quant.groupby("group").kl_mean.mean().to_dict())
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["smoke", "calibration", "evaluation"])
    p.add_argument("--wording", required=True, choices=["a", "b"])
    p.add_argument("--requests", default="results/m1/requests.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    requests = [r for r in rows if r["role"] == args.role]
    demos = {d: [r for r in rows if r["dataset"] == d and r["role"] == "demo"] for d in SOURCES}
    expected = 32 if args.role == "evaluation" else 8
    assert all(n >= expected for n in Counter(r["dataset"] for r in requests).values())
    assert set(r["dataset"] for r in requests) == set(SOURCES)
    config = dict(role=args.role, wording=args.wording, requests_sha256=file_hash(args.requests),
                  code_sha256={name: file_hash(name) for name in MEASUREMENT_SOURCES}, quantization=QUANTIZATION)
    if (out / "run_config.json").exists():
        assert read_json(out / "run_config.json") == config, "inputs/code changed: use a NEW output directory"
    else:
        save_json(out / "run_config.json", config)
    started = time.monotonic()
    model, tokenizer = load_model()
    run_meta = dict(metadata(), args=vars(args), started_utc=datetime.now(timezone.utc).isoformat(),
                    run_config_sha256=file_hash(out / "run_config.json"))
    # Keep every attempt's metadata rather than hiding an interrupted/resumed run.
    attempt = len(list(out.glob("attempt_*.json")))
    save_json(out / f"attempt_{attempt}.json", run_meta)
    (out / "reference_logits").mkdir(exist_ok=True)
    sentinel_ids = {next(r["request_id"] for r in requests if r["dataset"] == d) for d in SOURCES}
    for i, request in enumerate(requests):
        path = out / "baselines" / f"{key(request)}.json"
        logits_path = out / "reference_logits" / f"{key(request)}.pt"
        if path.exists() and logits_path.exists():
            continue
        prompt, ids = encode(tokenizer, request, demos[request["dataset"]], args.wording)
        answer = generate(model, tokenizer, ids, request)
        reference = continuation_logits(model, ids, answer["generated_ids"])
        if request["request_id"] in sentinel_ids:
            precision = check_precision(model, ids)
            repeated = continuation_logits(model, ids, answer["generated_ids"])
            assert torch.equal(reference, repeated)
            save_json(out / "checks" / f"{key(request)}_baseline.json",
                      dict(precision=precision, independent_replay_exact=True,
                           self_kl_max=max(map(abs, token_kl(reference, repeated, "cuda")))))
        temporary = logits_path.with_suffix(".pt.tmp")
        torch.save(reference, temporary)
        temporary.replace(logits_path)
        save_json(path, dict(request_id=request["request_id"], dataset=request["dataset"],
            role=args.role, wording=args.wording, gold=request["gold"], prompt=prompt,
            input_ids=ids[0].cpu().tolist(), input_tokens=ids.shape[1],
            request_tokens=len(tokenizer.encode(request["text"], add_special_tokens=False)),
            reference_logits_sha256=tensor_hash(reference), **answer))
        print(f"BF16 {i+1}/{len(requests)} {request['request_id']} correct={answer['correct']} "
              f"truncated={answer['truncated']}", flush=True)
    for group in range(8):
        pending = [r for r in requests if not (out / f"g{group}" / f"{key(r)}.json").exists()]
        if not pending and (out / "checks" / f"g{group}_restoration.json").exists():
            continue
        with quantized_group(model, group) as audit:
            save_json(out / "quantization" / f"g{group}.json", audit)
            for i, request in enumerate(pending):
                base = read_json(out / "baselines" / f"{key(request)}.json")
                ids = torch.tensor([base["input_ids"]], device="cuda")
                reference = torch.load(out / "reference_logits" / f"{key(request)}.pt", weights_only=True)
                assert tensor_hash(reference) == base["reference_logits_sha256"]
                q_logits = continuation_logits(model, ids, base["generated_ids"])
                values = token_kl(reference, q_logits, "cuda")
                answer = generate(model, tokenizer, ids, request)
                row = dict(request_id=request["request_id"], dataset=request["dataset"],
                    role=args.role, wording=args.wording, group=group, gold=request["gold"],
                    input_tokens=base["input_tokens"], request_tokens=base["request_tokens"],
                    bf16_answer=base["answer"], bf16_correct=base["correct"],
                    bf16_generated_ids=base["generated_ids"], reference_logits_sha256=base["reference_logits_sha256"],
                    **summarize_kl(values), **answer)
                save_json(out / f"g{group}" / f"{key(request)}.json", row)
                print(f"g{group} {i+1}/{len(pending)} {request['request_id']} "
                      f"KL={row['kl_mean']:.7f} correct={answer['correct']}", flush=True)
        for request in requests:
            if request["request_id"] not in sentinel_ids:
                continue
            base = read_json(out / "baselines" / f"{key(request)}.json")
            reference = torch.load(out / "reference_logits" / f"{key(request)}.pt", weights_only=True)
            restored = continuation_logits(model, torch.tensor([base["input_ids"]], device="cuda"), base["generated_ids"])
            assert torch.equal(reference, restored), "BF16 reference changed after restoration"
        save_json(out / "checks" / f"g{group}_restoration.json", dict(sentinels=sorted(sentinel_ids), exact=True))
    summary = export_run(out, requests)
    save_json(out / "summary.json", dict(status="complete", samples=len(requests),
        configurations=8, measured_rows=8 * len(requests), datasets=summary,
        attempt_seconds=time.monotonic() - started))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
