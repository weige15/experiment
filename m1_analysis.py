"""Offline analysis of measured records; never used to generate experiment values."""
import json
import re

import numpy as np
import pandas as pd


def content_indices(tokenizer, generated_ids):
    """Keep whole non-special tokens before the generated next-question delimiter."""
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    delimiter = re.search(r"\n\s*Q:", text)
    end = delimiter.start() if delimiter else len(text.rstrip())
    boundary = 0
    for n in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:n], skip_special_tokens=True)
        if text.startswith(prefix) and len(prefix) <= end:
            boundary = n
    return [i for i, token in enumerate(generated_ids[:boundary]) if token not in tokenizer.all_special_ids]


def add_content_metrics(frame, tokenizer):
    frame = frame.copy()
    masks = {}
    content_means, first_halves, second_halves, counts = [], [], [], []
    for row in frame.itertuples():
        key = (row.request_id, row.wording)
        ids = json.loads(row.bf16_generated_ids)
        values = np.array(json.loads(row.token_kl))
        if key not in masks:
            masks[key] = content_indices(tokenizer, ids)
        selected = values[masks[key]]
        n, mid = len(selected), len(selected) // 2
        counts.append(n)
        content_means.append(float(selected.mean()) if n else np.nan)
        first_halves.append(float(selected[:mid].mean()) if n >= 4 else np.nan)
        second_halves.append(float(selected[mid:].mean()) if n >= 4 else np.nan)
    frame["content_tokens"] = counts
    frame["kl_content_mean"] = content_means
    frame["kl_content_first_half"] = first_halves
    frame["kl_content_second_half"] = second_halves
    return frame
