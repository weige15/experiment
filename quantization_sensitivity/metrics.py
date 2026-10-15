"""Full-vocabulary KL on the exact same causal BF16 continuation."""
import torch


def prediction_inputs(prompt_ids, continuation_ids):
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or not continuation_ids:
        raise ValueError("one nonempty prompt and nonempty continuation required")
    continuation = torch.tensor([continuation_ids[:-1]], dtype=torch.long, device=prompt_ids.device)
    return torch.cat((prompt_ids, continuation), dim=1)


@torch.inference_mode()
def continuation_logits(model, prompt_ids, continuation_ids):
    # P prompt tokens, T targets: input length P+T-1, last T logits start at P-1.
    ids = prediction_inputs(prompt_ids, continuation_ids)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(continuation_ids)).logits[0]
    assert logits.shape[0] == len(continuation_ids)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite teacher-forced logits")
    return logits.cpu()


@torch.inference_mode()
def cached_continuation_logits(model, prompt_ids, continuation_ids):
    """Independent cached path used only as a numerical/alignment diagnostic."""
    out = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    rows = [out.logits[0, -1].cpu()]
    cache = out.past_key_values
    for token in continuation_ids[:-1]:
        ids = torch.tensor([[token]], device=prompt_ids.device)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        rows.append(out.logits[0, -1].cpu())
    return torch.stack(rows)


@torch.inference_mode()
def token_kl(reference, candidate, device="cpu", chunk=16):
    if reference.shape != candidate.shape or reference.ndim != 2 or not reference.shape[0]:
        raise ValueError("matching, nonempty [tokens, vocabulary] logits required")
    values = []
    for start in range(0, len(reference), chunk):
        p_logits = reference[start:start + chunk].to(device).float()
        q_logits = candidate[start:start + chunk].to(device).float()
        if not torch.isfinite(p_logits).all() or not torch.isfinite(q_logits).all():
            raise FloatingPointError("nonfinite KL input")
        log_p = torch.log_softmax(p_logits, dim=-1)
        log_q = torch.log_softmax(q_logits, dim=-1)
        # FP32 log probabilities/products, FP64 accumulation avoids cancellation in summation.
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1, dtype=torch.float64)
        values.extend(kl.cpu().tolist())
    if min(values) < -1e-6:
        raise FloatingPointError(f"negative KL outside numeric tolerance: {min(values)}")
    return values


def summarize_kl(values):
    n = len(values)
    if not n:
        raise ValueError("empty KL trace")
    mid = n // 2
    return dict(kl_mean=sum(values) / n, kl_tokens=n, token_kl=values,
                kl_first_half=sum(values[:mid]) / mid if n >= 4 else None,
                kl_second_half=sum(values[mid:]) / (n - mid) if n >= 4 else None,
                kl_first_token=values[0])
