"""Small, single-request BF16 runner. No serving/router or integer-kernel claims."""
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor

from m1_data import CHAT_TEMPLATE, MAX_NEW_TOKENS, SEED, messages_for, score_answer

MODEL_ID = "meta-llama/Llama-3.1-8B"
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


def load_model():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("M1 requires an available BF16-capable CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    assert tokenizer.chat_template is None, "Recheck protocol if upstream template changes"
    tokenizer.chat_template = CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    assert model.config.num_hidden_layers == 32
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.config._commit_hash == REVISION
    return model, tokenizer


def encode(tokenizer, request, demos, wording):
    prompt = tokenizer.apply_chat_template(messages_for(request, demos, wording),
                                           tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    assert ids[0, 0].item() == tokenizer.bos_token_id
    return prompt, ids.cuda()


class RequireFiniteLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite generation logits")
        return scores


@torch.inference_mode()
def generate(model, tokenizer, ids, request):
    torch.cuda.synchronize()
    start = time.monotonic()
    limit = MAX_NEW_TOKENS[request["dataset"]]
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=limit,
                              use_cache=True, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.eos_token_id,
                              stop_strings=["\nQ:", "\n\nQ:"])
    output = model.generate(ids, attention_mask=torch.ones_like(ids),
                            generation_config=config, tokenizer=tokenizer,
                            logits_processor=[RequireFiniteLogits()])
    tokens = output[0, ids.shape[1]:].cpu().tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    torch.cuda.synchronize()
    return dict(answer=text, generated_ids=tokens, generated_tokens=len(tokens),
                generation_seconds=time.monotonic() - start,
                truncated=len(tokens) == limit and tokens[-1] != tokenizer.eos_token_id
                          and "\nQ:" not in text and "\n\nQ:" not in text,
                **score_answer(text, request))


@torch.inference_mode()
def check_precision(model, ids):
    observed = set()
    def hook(_, args, output):
        observed.add((str(args[0].dtype), str(output.dtype)))
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    try:
        out = model(input_ids=ids[:, :8], use_cache=True)
        kv = {str(t.dtype) for pair in out.past_key_values for t in pair}
        assert observed == {("torch.bfloat16", "torch.bfloat16")}, observed
        assert kv == {"torch.bfloat16"}, kv
        assert torch.isfinite(out.logits).all()
        return {"linear_input_output_dtypes": sorted(observed), "kv_cache_dtypes": sorted(kv)}
    finally:
        for handle in handles:
            handle.remove()


def metadata():
    versions = {}
    for name in ["torch", "transformers", "accelerate", "huggingface-hub", "numpy", "pandas", "pyarrow"]:
        versions[name] = importlib.metadata.version(name)
    return dict(model_id=MODEL_ID, model_revision=REVISION, seed=SEED,
                python=platform.python_version(), platform=platform.platform(),
                hostname=platform.node(), versions=versions,
                cuda_runtime=torch.version.cuda,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu=torch.cuda.get_device_name(),
                nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
                chat_template=CHAT_TEMPLATE,
                chat_template_sha256=hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest(),
                activation_precision="BF16 (standard Llama internal norm/softmax behavior)",
                kv_cache_precision="BF16", attention="sdpa", deterministic=True,
                max_new_tokens=MAX_NEW_TOKENS,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path('.').glob('m1_*.py'))})
