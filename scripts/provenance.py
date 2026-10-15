"""Verify local model shards/tokenizer against the pinned official Hub file digests."""
import argparse
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from m1_data import save_json
from m1_model import MODEL_ID, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/m1/model_provenance.json")
    args = parser.parse_args()
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    verified = []
    for item in info.siblings:
        name = item.rfilename
        if name.startswith("original/") or not (name.endswith(".safetensors") or name in {
                "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "generation_config.json", "model.safetensors.index.json"}):
            continue
        path = Path(hf_hub_download(MODEL_ID, name, revision=REVISION))
        sha256 = hashlib.sha256()
        git_hash = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as f:
            while block := f.read(8 * 1024 * 1024):
                sha256.update(block)
                git_hash.update(block)
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        actual = sha256.hexdigest() if item.lfs else git_hash.hexdigest()
        assert actual == expected, f"official digest mismatch: {name}"
        verified.append(dict(file=name, bytes=path.stat().st_size, sha256=sha256.hexdigest(),
                             official_digest=expected, verified=True))
        print(f"VERIFIED {name}", flush=True)
    assert sum(v["file"].endswith(".safetensors") for v in verified) == 4
    save_json(args.out, dict(model_id=MODEL_ID,
              revision=REVISION, files=verified))


if __name__ == "__main__":
    main()
