"""Smoke-only boundary check, before formal analysis; preserve original KL unchanged."""
import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from quantization_sensitivity import DEFAULT_RESULTS
from quantization_sensitivity.analysis import add_content_metrics
from quantization_sensitivity.model import MODEL_ID, REVISION
from quantization_sensitivity.data import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_RESULTS)
    root = Path(parser.parse_args().root)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    frames = [pd.read_csv(root / "runs" / f"smoke_{w}" / "measurements.csv") for w in ["a", "b"]]
    measured = add_content_metrics(pd.concat(frames, ignore_index=True), tokenizer)
    assert measured.kl_content_mean.notna().all(), "inspect empty-content cases before scaling"
    output = root / "steps" / "smoke_content"
    output.mkdir(parents=True, exist_ok=True)
    measured.to_csv(output / "measurements.csv", index=False)
    unique = measured.drop_duplicates(["request_id", "wording"])
    summary = {d: dict(worded_requests=len(g), min_content_tokens=int(g.content_tokens.min()),
        max_content_tokens=int(g.content_tokens.max()), content_segments_available=int((g.content_tokens >= 4).sum()))
        for d, g in unique.groupby("dataset")}
    save_json(output / "summary.json", dict(scope="smoke_only_not_formal_evidence", datasets=summary))
    print(summary)


if __name__ == "__main__":
    main()
