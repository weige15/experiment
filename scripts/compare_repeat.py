"""Compare actual independent-device gate runs; no approximate pass concealed as exact."""
import argparse
from pathlib import Path
import numpy as np
import torch
from m1_data import save_json
from m1_run import read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", default="results/m1/steps/single_sensitivity")
    p.add_argument("--repeat", required=True)
    args = p.parse_args()
    original, repeat = Path(args.original), Path(args.repeat)
    a = torch.load(original / "reference.pt", weights_only=True)
    b = torch.load(repeat / "reference.pt", weights_only=True)
    qa = read_json(original / "group_0.json")["measurement"]
    qb = read_json(repeat / "group_0.json")["measurement"]
    result = dict(BF16_logits_bitwise_equal=torch.equal(a, b),
        W4_KL_max_absolute_difference=float(np.abs(np.array(qa["token_kl"]) - np.array(qb["token_kl"])).max()),
        W4_generated_ids_equal=qa["generated_ids"] == qb["generated_ids"],
        token_count=len(qa["token_kl"]), original=str(original), repeat=str(repeat))
    save_json(repeat / "cross_device_comparison.json", result)
    assert result["BF16_logits_bitwise_equal"] and result["W4_generated_ids_equal"]
    assert result["W4_KL_max_absolute_difference"] == 0
    print("PASS exact cross-device reproduction:", result, flush=True)


if __name__ == "__main__":
    main()
