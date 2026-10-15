"""Incremental gate 1: one real GSM8K BF16 baseline, saved before quantization."""
import argparse
import time
from pathlib import Path

from m1_data import sample_dataset, save_json
from m1_model import check_precision, encode, generate, load_model, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/m1/steps/baseline")
    args = parser.parse_args()
    out = Path(args.out)
    start = time.monotonic()
    rows, exclusions = sample_dataset("gsm8k")
    demos = [r for r in rows if r["role"] == "demo"]
    request = next(r for r in rows if r["role"] == "smoke")
    model, tokenizer = load_model()
    prompt, ids = encode(tokenizer, request, demos, "a")
    precision = check_precision(model, ids)
    result = generate(model, tokenizer, ids, request)
    save_json(out / "baseline.json", dict(request=request, demos=demos, prompt=prompt,
              input_ids=ids[0].cpu().tolist(), input_tokens=ids.shape[1],
              exclusions=exclusions, result=result))
    save_json(out / "metadata.json", dict(metadata(), precision_check=precision,
              total_seconds=time.monotonic() - start))
    print(f"PASS BF16 request={request['request_id']} tokens={ids.shape[1]} "
          f"correct={result['correct']} parsed={result['parsed_answer']} "
          f"truncated={result['truncated']}", flush=True)
    print(result["answer"], flush=True)


if __name__ == "__main__":
    main()
