"""Save the entire fixed sample before any formal evaluation; no sample selection by scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from m1_data import CHAT_TEMPLATE, SEED, SOURCES, sample_dataset, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/m1")
    p.add_argument("--n-eval", type=int, default=32)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, exclusions, source_files = [], {}, []
    for dataset, (repo, revision, folder, heldout) in SOURCES.items():
        selected, excluded = sample_dataset(dataset, n_eval=args.n_eval)
        rows.extend(selected)
        exclusions[dataset] = excluded
        for split in ["train", heldout]:
            filename = f"{folder}/{split}-00000-of-00001.parquet"
            path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
            source_files.append(dict(dataset=dataset, repo=repo, revision=revision, file=filename,
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(dataset, dict(Counter(r["role"] for r in selected)), "duplicates excluded", len(excluded), flush=True)
    assert len({r["request_id"] for r in rows}) == len(rows)
    for dataset in SOURCES:
        subset = [r for r in rows if r["dataset"] == dataset]
        assert len({r["question_sha256"] for r in subset}) == len(subset)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path = out / "requests.jsonl"
    if path.exists() and path.read_text() != payload:
        raise RuntimeError("refusing to overwrite a different frozen sample")
    path.write_text(payload)
    (out / "chat_template.jinja").write_text(CHAT_TEMPLATE + "\n")
    save_json(out / "data_manifest.json", dict(seed=SEED, n_eval_per_dataset=args.n_eval,
              counts=dict(Counter(r["role"] for r in rows)), sources=source_files,
              exclusions=exclusions, requests_sha256=hashlib.sha256(payload.encode()).hexdigest()))


if __name__ == "__main__":
    main()
