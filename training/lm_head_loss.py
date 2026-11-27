"""Answer-token negative log likelihood for HiveMIND's Qwen3 trainer."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from training import ternary_base


VOCAB_CHUNK = 16384


def _qwen_readout(model, *, batch):
    """Return the directly projectable base and whether the public forward honors an integer logits suffix."""
    # Qwen3ForCausalLM.forward projects this exact state with no logit scaling
    # or soft cap. Only bypass that forward for the architecture we verified.
    from transformers import Qwen3ForCausalLM
    if type(model) is Qwen3ForCausalLM:
        return model, True
    # The trainer's LoRA modules live inside the decoder. Prompt-learning
    # and other PEFT wrappers may instead change their inputs in forward.
    from peft import LoraConfig, PeftModel, PeftModelForCausalLM
    if type(model) in (PeftModel, PeftModelForCausalLM) or (not batch and isinstance(model, PeftModel)):
        base = model.get_base_model()
        if type(base) is Qwen3ForCausalLM:
            config = model.active_peft_config
            plain_lora = type(config) is LoraConfig and config.alora_invocation_tokens is None
            return (base if plain_lora else None), True
    return None, False


def _chunk_base(model, targets, chunk_tokens):
    if 0 < chunk_tokens < targets:
        # Only bypass the public forward when the decoder and head have the
        # score function we implement. An adapted head may change the logits.
        from transformers import Qwen3ForCausalLM
        if type(model) is not Qwen3ForCausalLM:
            from peft import LoraConfig, PeftModel
            if not (isinstance(model, PeftModel) and type(model.active_peft_config) is LoraConfig
                    and model.active_peft_config.alora_invocation_tokens is None):
                return None
            model = model.get_base_model()
        if type(model) is Qwen3ForCausalLM:
            head = model.lm_head
            plain = type(head) is torch.nn.Linear or (type(head) is ternary_base.TernaryLinear and not head.rank)
            if plain and not (head._forward_hooks or head._forward_pre_hooks or "forward" in vars(head)):
                return model
    return None


def _width(vocab, targets, chunk_tokens, vocab_chunk):
    if vocab_chunk < 1:
        raise ValueError("vocab_chunk must be positive")
    width = min(vocab, vocab_chunk)
    # A streamed tile needs at most 8 bytes per logit, versus the guard's 16
    # for dense CE. Keep the old byte ceiling even on very long answers. Halving
    # bounds the number of distinct dequantization scratch sizes across rows.
    while width > 1 and targets * width > 2 * chunk_tokens * vocab:
        width = (width + 1) // 2
    return width


def projection_rows(model, targets, chunk_tokens, *, vocab_chunk=VOCAB_CHUNK):
    """Peak logits in equivalent full-vocabulary rows, rounded up for admission."""
    base = _chunk_base(model, targets, chunk_tokens)
    if base is None:
        return targets + 1
    vocab = base.lm_head.out_features
    width = _width(vocab, targets, chunk_tokens, vocab_chunk)
    return (targets * width + vocab - 1) // vocab


def _project(hidden, weight, bias, scale, start, end):
    """Return the fp32 logits and the same weight tile for immediate reuse."""
    block = weight[start:end]
    if scale is None:
        block = block.to(hidden.dtype)
        b = None if bias is None else bias[start:end].to(hidden.dtype)
        return F.linear(hidden, block, b).float(), block
    # The CPU trainer stores the readout as frozen int8 codes and group scales.
    # Its shared dequantization buffer is only valid until the next tile.
    dtype = torch.float32 if hidden.dtype == torch.bfloat16 and not ternary_base.quick16() else hidden.dtype
    block = ternary_base._deq(block, scale[start:end], dtype)
    logits = F.linear(hidden.to(dtype), block).to(hidden.dtype)
    if bias is not None:  # TernaryLinear adds its bias after rounding the matmul.
        logits = logits + bias[start:end].to(logits.dtype)
    return logits.float(), block


class _VocabCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, weight, bias, scale, width):
        maximum = hidden.new_full((len(labels),), -torch.inf, dtype=torch.float32)
        total = torch.zeros_like(maximum)
        target = torch.empty_like(maximum)  # labels are the input token IDs of the decoder
        for start in range(0, len(weight), width):
            end = min(start + width, len(weight))
            logits, block = _project(hidden, weight, bias, scale, start, end)
            rows = torch.where((labels >= start) & (labels < end))[0]
            target[rows] = logits[rows, labels[rows] - start]
            new_max = torch.maximum(maximum, logits.amax(1))
            total = total * (maximum - new_max).exp() + logits.sub_(new_max[:, None]).exp_().sum(1)
            maximum = new_max
            del logits, block
        ctx.save_for_backward(hidden, labels, weight, bias, scale, maximum, total)
        ctx.width = width
        # Keep the two terms of logsumexp separate even for large common offsets.
        return ((maximum - target) + total.log()).sum()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        hidden, labels, weight, bias, scale, maximum, total = ctx.saved_tensors
        acc_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        dh = torch.zeros_like(hidden, dtype=acc_dtype) if ctx.needs_input_grad[0] else None
        dw = torch.empty_like(weight) if ctx.needs_input_grad[2] else None
        db = torch.empty_like(bias) if ctx.needs_input_grad[3] else None
        # Backward also works when called outside the forward autocast context.
        with torch.autocast(hidden.device.type, enabled=False):
            for start in range(0, len(weight), ctx.width):
                end = min(start + ctx.width, len(weight))
                grad, block = _project(hidden, weight, bias, scale, start, end)
                grad.sub_(maximum[:, None]).exp_().div_(total[:, None])
                rows = torch.where((labels >= start) & (labels < end))[0]
                grad[rows, labels[rows] - start] -= 1
                grad.mul_(grad_output)
                # Keep rounded gradients and fp32 weight conversions local to a
                # sub-tile. Accumulate in fp32 before rounding d_hidden once.
                rows_per_mm = len(block)
                if dh is not None and hidden.dtype != dh.dtype:
                    # The two upcast operands fit in one logits tile and in 8 MiB.
                    elements = min(grad.numel(), (8 << 20) // dh.element_size())
                    rows_per_mm = max(1, elements // (len(hidden) + block.shape[1]))
                for offset in range(0, len(block), rows_per_mm):
                    part = slice(offset, offset + rows_per_mm)
                    g = grad[:, part].to(hidden.dtype)
                    if dh is not None:
                        dh.addmm_(g.to(dh.dtype), block[part].to(dh.dtype))
                    cols = slice(start + offset, min(start + offset + rows_per_mm, end))
                    if dw is not None:
                        dw[cols] = g.T @ hidden
                    if db is not None:
                        db[cols] = g.sum(0)
                del g
                del grad, block
        return dh, None, dw, db, None, None


def batch_projection_rows(model, lengths, targets, chunk_tokens=0):
    """Largest vocabulary projection for right-padded rows with an answer at the end of each."""
    if not lengths or len(lengths) != len(targets) or chunk_tokens < 0 or any(
        not 0 <= k < length for length, k in zip(lengths, targets)
    ):
        raise ValueError("expected one target count smaller than each positive sequence length")
    total = sum(targets)
    base, honors_suffix = _qwen_readout(model, batch=True)
    if base is not None:
        return min(chunk_tokens, total) if chunk_tokens else total
    width = max(lengths)
    if not honors_suffix:
        # An unknown public forward can accept logits_to_keep yet project every position.
        config = getattr(model, "active_peft_config", None)
        return len(lengths) * (width + (getattr(config, "num_virtual_tokens", 0) or 0))
    first = min((length - k - 1 for length, k in zip(lengths, targets) if k), default=width - 1)
    return len(lengths) * (width - first)


def _chunk_losses(base, hidden, labels, chunk_tokens, reduction):
    def project(block, targets):
        return F.cross_entropy(base.lm_head(block).float(), targets, reduction=reduction)

    for block, targets in zip(hidden.split(chunk_tokens), labels.split(chunk_tokens)):
        # Keep only hidden inputs and target IDs. Backward recomputes each chunk's logits.
        if torch.is_grad_enabled():
            yield checkpoint(project, block, targets, use_reentrant=False)
        else:
            yield project(block, targets)


def _masked_nll(model, ids, labels, attention_mask, chunk_tokens):
    if ids.ndim != 2 or ids.shape[1] < 2 or labels.shape != ids.shape or chunk_tokens < 0:
        raise ValueError("expected matching batched token IDs and labels, and a nonnegative chunk size")
    valid = labels[:, 1:] != -100
    kwargs = {}
    if attention_mask is not None:
        if attention_mask.shape != ids.shape:
            raise ValueError("attention_mask must match the token IDs")
        active = attention_mask.bool()
        valid = valid & active[:, :-1] & active[:, 1:]
        kwargs["attention_mask"] = active
    targets = labels[:, 1:][valid]
    base, _ = _qwen_readout(model, batch=True)
    if base is not None and 0 < chunk_tokens < targets.numel():
        states = base.model(input_ids=ids, return_dict=True, **kwargs).last_hidden_state
        hidden = states[:, :-1][valid]
        losses = torch.cat(tuple(_chunk_losses(base, hidden, targets, chunk_tokens, "none")))
    else:
        if base is not None:
            # Pinned Qwen3 accepts a bool mask as logits_to_keep and flattens the
            # selected states. A 2D integer tensor instead means per-row indices.
            keep = F.pad(valid, (0, 1), value=False)
            logits = model(input_ids=ids, logits_to_keep=keep, **kwargs).logits
        else:
            # Preserve the public model's transformations and any PEFT virtual prompt.
            # Take the suffix even if a wrapper returns all positions.
            first = int(valid.any(0).nonzero()[0].item()) if targets.numel() else ids.shape[1] - 1
            keep = ids.shape[1] - first
            logits = model(input_ids=ids, logits_to_keep=keep, **kwargs).logits[:, -keep:-1]
            logits = logits[valid[:, first:]]
        losses = F.cross_entropy(logits.float(), targets, reduction="none")
    return losses.new_zeros(valid.shape).masked_scatter(valid, losses).sum(dim=1)


def answer_nll(model, ids, k=None, chunk_tokens=0, *, attention_mask=None, labels=None, vocab_chunk=VOCAB_CHUNK):
    """Sum the k next-token losses at the end of each sequence, across the batch.

    A positive token threshold enables vocabulary tiling on longer Qwen3 rows (k path): the decoder runs once and
    only normalization statistics survive between scans of the output head. Short answers and other heads use
    their stock forward. With labels, return a separate sum for each row and ignore labels of -100 (the batched
    path, chunked by tokens). Labels are aligned to input IDs; each is predicted by its preceding valid token.
    The original k-token path returns a scalar. Call inside autocast.
    """
    if labels is not None:
        return _masked_nll(model, ids, labels, attention_mask, chunk_tokens)
    if ids.ndim != 2 or k is None or not 0 < k < ids.shape[1] or chunk_tokens < 0:
        raise ValueError("expected batched token IDs, 0 < k < sequence length, and a nonnegative chunk size")
    kwargs = {} if attention_mask is None else {"attention_mask": attention_mask}
    labels = ids[:, -k:].reshape(-1)
    base = _chunk_base(model, labels.numel(), chunk_tokens)
    if base is not None:
        head = base.lm_head
        width = _width(head.out_features, labels.numel(), chunk_tokens, vocab_chunk)
        states = base.model(input_ids=ids, return_dict=True, **kwargs).last_hidden_state
        hidden = states[:, -k - 1:-1].reshape(-1, states.shape[-1])
        # A ternary embedding accepts negative Python indices. Preserve dense
        # CE's ignored label and its rejection of other out-of-range targets.
        if ((labels < 0) | (labels >= head.out_features)).any():
            keep = labels != -100
            hidden, labels = hidden[keep], labels[keep]
            if ((labels < 0) | (labels >= head.out_features)).any():
                raise IndexError("target token is outside the output vocabulary")
        if torch.is_autocast_enabled(hidden.device.type) and hidden.dtype != torch.float64:
            hidden = hidden.to(torch.get_autocast_dtype(hidden.device.type))
        with torch.autocast(hidden.device.type, enabled=False):
            if type(head) is ternary_base.TernaryLinear:
                return _VocabCrossEntropy.apply(hidden, labels, head.codes, head.bias, head.scale, width)
            return _VocabCrossEntropy.apply(hidden, labels, head.weight, head.bias, None, width)
    logits = model(input_ids=ids, logits_to_keep=k + 1, **kwargs).logits[:, :-1, :]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels, reduction="sum")
