"""Pinned benchmark samples and one explicit base-model Q/A chat template."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SEED = 20261015
SOURCES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main", "test"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge", "test"),
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", "data", "validation"),
}
CHAT_TEMPLATE = "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}{{ m['content'] + '\n\n' }}{% elif m['role'] == 'user' %}{{ 'Q: ' + m['content'] + '\n' }}{% else %}{{ 'A: ' + m['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'A:' }}{% endif %}"
WORDING = {
    "gsm8k": ("Solve the following problem.", "Work out the answer to this problem."),
    "arc": ("Choose the correct option.", "Select the best answer."),
    "boolq": ("Answer the question using the passage.", "Based on the passage, respond to the question."),
}
INSTRUCTIONS = {
    "gsm8k": "Solve each math problem. Show your reasoning, then end with #### followed by the numerical answer.",
    "arc": "Answer each multiple-choice question with only the correct option label.",
    "boolq": "Read each passage and answer the question with only yes or no.",
}
MAX_NEW_TOKENS = {"gsm8k": 256, "arc": 16, "boolq": 16}


def load_split(dataset, split):
    repo, revision, folder, _ = SOURCES[dataset]
    path = hf_hub_download(repo, f"{folder}/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=revision)
    return pd.read_parquet(path)


def normalize(row, dataset, split, index):
    question = str(row["question"]).strip()
    if dataset == "gsm8k":
        answer = re.sub(r"<<.*?>>", "", row["answer"])
        gold = answer.split("####")[-1].strip().replace(",", "")
        text, labels = question, []
    elif dataset == "arc":
        labels = list(row["choices"]["label"])
        text = question + "\n" + "\n".join(f"{k}. {v}" for k, v in zip(labels, row["choices"]["text"]))
        answer = gold = str(row["answerKey"])
    else:
        text = f"Passage: {row['passage']}\nQuestion: {question}"
        answer = gold = "yes" if row["answer"] else "no"
        labels = []
    content_hash = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
    question_hash = hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()
    return dict(request_id=f"{dataset}:{split}:{index}", dataset=dataset, split=split,
                source_index=int(index), text=text, demo_answer=answer, gold=gold,
                labels=labels, content_sha256=content_hash, question_sha256=question_hash)


def sample_dataset(dataset, n_eval=32):
    """Split-isolated demos/calibration; disjoint smoke/eval, all without replacement."""
    rng = np.random.default_rng(SEED)
    train, heldout = load_split(dataset, "train"), load_split(dataset, SOURCES[dataset][3])
    seen, sampled, excluded = set(), [], []
    for frame, split, roles in [(train, "train", [("demo", 3), ("calibration", 8)]),
                                (heldout, SOURCES[dataset][3], [("smoke", 8), ("evaluation", n_eval)])]:
        order = iter(rng.permutation(len(frame)))
        for role, count in roles:
            for _ in range(count):
                while True:
                    index = int(next(order))
                    row = normalize(frame.iloc[index], dataset, split, index)
                    if row["question_sha256"] not in seen:
                        break
                    excluded.append(row["request_id"])
                seen.add(row["question_sha256"])
                sampled.append(dict(row, role=role))
    return sampled, excluded


def messages_for(request, demos, wording="a"):
    prefix = WORDING[request["dataset"]][{"a": 0, "b": 1}[wording]]
    messages = [{"role": "system", "content": INSTRUCTIONS[request["dataset"]]}]
    for demo in demos:
        messages += [{"role": "user", "content": prefix + "\n" + demo["text"]},
                     {"role": "assistant", "content": demo["demo_answer"]}]
    return messages + [{"role": "user", "content": prefix + "\n" + request["text"]}]


def parse_answer(text, dataset, labels=()):
    text = re.split(r"\n\s*Q:", text, maxsplit=1)[0].strip()
    if dataset == "gsm8k":
        # Final marked answer, entire first line: never accept a prefix of 1/2 or 1e3.
        if "####" not in text:
            return None
        tail = text.rsplit("####", 1)[1].strip()
        line = tail.splitlines()[0].strip() if tail else ""
        match = re.fullmatch(r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)[.!?]?", line)
        if not match:
            return None
        from decimal import Decimal
        return str(Decimal(match.group(1).replace(",", "")).normalize())
    if dataset == "arc":
        match = re.match(r"\s*([A-E1-5])(?:[.\s):]|$)", text)
        return match.group(1) if match and match.group(1) in labels else None
    match = re.match(r"\s*(yes|no)\b", text, re.I)
    return match.group(1).lower() if match else None


def score_answer(text, request):
    parsed = parse_answer(text, request["dataset"], request["labels"])
    gold = request["gold"]
    if request["dataset"] == "gsm8k":
        from decimal import Decimal
        gold = str(Decimal(gold).normalize())
    return {"parsed_answer": parsed, "correct": parsed == gold, "parse_failure": parsed is None}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
