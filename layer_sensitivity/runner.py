"""Resumable real-weight scan: every request × each of 32 individual W4 layers."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from quantization_sensitivity.data import SOURCES, save_json
from quantization_sensitivity.metrics import continuation_logits, summarize_kl, token_kl
from quantization_sensitivity.model import check_precision, encode, generate, load_model, metadata
from quantization_sensitivity.runner import file_hash, key, read_json, tensor_hash, write_csv
from . import DEFAULT_RESULTS
from .quantization import LAYER_QUANTIZATION, quantized_layers

MEASUREMENT_SOURCES = [f"quantization_sensitivity/{name}.py" for name in
                       ["__init__", "data", "model", "metrics", "quantization", "runner"]] + [
                       f"layer_sensitivity/{name}.py" for name in ["__init__", "quantization", "runner"]]


def source_hashes():
    return {name: file_hash(name) for name in MEASUREMENT_SOURCES}


def configuration(role, requests_path):
    return dict(role=role, wording="a", requests_sha256=file_hash(requests_path),
                code_sha256=source_hashes(), quantization=LAYER_QUANTIZATION, layers=list(range(32)))


def export_run(out, requests):
    for layer in range(32):
        assert read_json(out / "checks" / f"layer_{layer}_restoration.json")["exact"]
    baselines = [read_json(out / "baselines" / f"{key(r)}.json") for r in requests]
    measurements = [read_json(out / f"layer_{layer}" / f"{key(r)}.json")
                    for r in requests for layer in range(32)]
    frame = pd.DataFrame(measurements)
    assert len(frame) == 32 * len(requests)
    assert not frame.duplicated(["request_id", "layer"]).any()
    assert np.isfinite(frame.kl_mean).all() and (frame.kl_mean >= -1e-6).all()
    frame["sensitivity_rank"] = frame.groupby("request_id").kl_mean.rank(ascending=False, method="average")
    write_csv(baselines, out / "baselines.csv")
    write_csv(frame.to_dict("records"), out / "measurements.csv")
    frame.to_parquet(out / "measurements.parquet", index=False)
    frame.pivot(index="request_id", columns="layer", values="kl_mean").rename(
        columns=lambda i: f"layer_{i}").to_csv(out / "sensitivity_vectors.csv")
    return {dataset: dict(n=sum(b["dataset"] == dataset for b in baselines),
        bf16_correct=sum(b["correct"] for b in baselines if b["dataset"] == dataset),
        W4_correct_by_layer=frame[frame.dataset == dataset].groupby("layer").correct.sum().to_dict(),
        KL_mean_by_layer=frame[frame.dataset == dataset].groupby("layer").kl_mean.mean().to_dict())
        for dataset in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["smoke", "calibration", "evaluation"])
    parser.add_argument("--requests", default=f"{DEFAULT_RESULTS}/requests.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    requests = [r for r in rows if r["role"] == args.role]
    demos = {d: [r for r in rows if r["dataset"] == d and r["role"] == "demo"] for d in SOURCES}
    counts = Counter(r["dataset"] for r in requests)
    assert set(counts) == set(SOURCES) and min(counts.values()) >= (32 if args.role == "evaluation" else 8)
    assert len({r["request_id"] for r in requests}) == len(requests)
    config = configuration(args.role, args.requests)
    if (out / "run_config.json").exists():
        assert read_json(out / "run_config.json") == config, "inputs/code changed: use a NEW output directory"
    else:
        save_json(out / "run_config.json", config)
    # Gate evaluation on train-only choices being committed to disk first.
    if args.role == "evaluation":
        selection = Path(args.requests).parent / "precision_allocation_selection.json"
        frozen = read_json(selection)
        assert frozen["requests_sha256"] == config["requests_sha256"] and frozen["budget"] == 8
        assert frozen["source_role"] == "calibration" and frozen["source_sha256"] == source_hashes()
    started = time.monotonic()
    model, tokenizer = load_model()
    run_meta = dict(metadata(), args=vars(args), started_utc=datetime.now(timezone.utc).isoformat(),
                    measurement_sources=source_hashes(), run_config_sha256=file_hash(out / "run_config.json"))
    if args.role == "evaluation":
        run_meta["allocation_selection_sha256"] = file_hash(selection)
    attempt = len(list(out.glob("attempt_*.json")))
    save_json(out / f"attempt_{attempt}.json", run_meta)
    (out / "reference_logits").mkdir(exist_ok=True)
    sentinel_ids = {next(r["request_id"] for r in requests if r["dataset"] == d) for d in SOURCES}
    for i, request in enumerate(requests):
        path = out / "baselines" / f"{key(request)}.json"
        logits_path = out / "reference_logits" / f"{key(request)}.pt"
        if path.exists() and logits_path.exists():
            continue
        prompt, ids = encode(tokenizer, request, demos[request["dataset"]], "a")
        answer = generate(model, tokenizer, ids, request)
        reference = continuation_logits(model, ids, answer["generated_ids"])
        if request["request_id"] in sentinel_ids:
            precision = check_precision(model, ids)
            repeated = continuation_logits(model, ids, answer["generated_ids"])
            assert torch.equal(reference, repeated)
            save_json(out / "checks" / f"{key(request)}_baseline.json", dict(precision=precision,
                independent_replay_exact=True, self_kl_max=max(map(abs, token_kl(reference, repeated, "cuda")))))
        temporary = logits_path.with_suffix(".pt.tmp")
        torch.save(reference, temporary)
        temporary.replace(logits_path)
        save_json(path, dict(request_id=request["request_id"], dataset=request["dataset"], role=args.role,
            wording="a", gold=request["gold"], prompt=prompt, input_ids=ids[0].cpu().tolist(),
            input_tokens=ids.shape[1], request_tokens=len(tokenizer.encode(request["text"], add_special_tokens=False)),
            reference_logits_sha256=tensor_hash(reference), **answer))
        print(f"BF16 {i+1}/{len(requests)} {request['request_id']} correct={answer['correct']}", flush=True)
    for layer in range(32):
        pending = [r for r in requests if not (out / f"layer_{layer}" / f"{key(r)}.json").exists()]
        if not pending and (out / "checks" / f"layer_{layer}_restoration.json").exists():
            continue
        with quantized_layers(model, [layer]) as audit:
            save_json(out / "quantization" / f"layer_{layer}.json", audit)
            for i, request in enumerate(pending):
                base = read_json(out / "baselines" / f"{key(request)}.json")
                ids = torch.tensor([base["input_ids"]], device="cuda")
                reference = torch.load(out / "reference_logits" / f"{key(request)}.pt", weights_only=True)
                assert tensor_hash(reference) == base["reference_logits_sha256"]
                q_logits = continuation_logits(model, ids, base["generated_ids"])
                values = token_kl(reference, q_logits, "cuda")
                if request["request_id"] in sentinel_ids:
                    assert torch.equal(q_logits, continuation_logits(model, ids, base["generated_ids"]))
                    save_json(out / "checks" / f"layer_{layer}_{key(request)}.json",
                              dict(independent_W4_replay_exact=True, precision=check_precision(model, ids)))
                answer = generate(model, tokenizer, ids, request)
                row = dict(request_id=request["request_id"], dataset=request["dataset"], role=args.role,
                    wording="a", layer=layer, gold=request["gold"], input_tokens=base["input_tokens"],
                    request_tokens=base["request_tokens"], bf16_answer=base["answer"], bf16_correct=base["correct"],
                    bf16_generated_ids=base["generated_ids"], reference_logits_sha256=base["reference_logits_sha256"],
                    **summarize_kl(values), **answer)
                save_json(out / f"layer_{layer}" / f"{key(request)}.json", row)
                print(f"layer={layer} {i+1}/{len(pending)} {request['request_id']} "
                      f"KL={row['kl_mean']:.8f} correct={answer['correct']}", flush=True)
        for request in requests:
            if request["request_id"] not in sentinel_ids:
                continue
            base = read_json(out / "baselines" / f"{key(request)}.json")
            reference = torch.load(out / "reference_logits" / f"{key(request)}.pt", weights_only=True)
            restored = continuation_logits(model, torch.tensor([base["input_ids"]], device="cuda"), base["generated_ids"])
            assert torch.equal(reference, restored), "BF16 reference changed after restoration"
        save_json(out / "checks" / f"layer_{layer}_restoration.json", dict(sentinels=sorted(sentinel_ids), exact=True))
    summary = export_run(out, requests)
    save_json(out / "summary.json", dict(status="complete", samples=len(requests), configurations=32,
        measured_rows=32 * len(requests), datasets=summary, attempt_seconds=time.monotonic() - started,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
