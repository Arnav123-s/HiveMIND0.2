import contextlib

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from training import lm_head_loss, ternary_base
from training.lm_head_loss import answer_nll, projection_rows


def small_cell(kind):
    torch.manual_seed(17)
    cfg = Qwen3Config(vocab_size=137, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=32, max_position_embeddings=64, use_cache=False)
    model = Qwen3ForCausalLM(cfg)
    if kind == "peft":
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"]))
        disable = model.disable_adapter
    else:
        model.requires_grad_(False)
        for parent, name in [(model.model.layers[0].self_attn, "q_proj"), (model.model.layers[0].mlp, "down_proj")]:
            setattr(parent, name, ternary_base.TernaryLinear(getattr(parent, name), rank=4, alpha=8))
        model.lm_head = ternary_base.TernaryLinear(model.lm_head)
        disable = ternary_base.Cell(model, [0], 4, 8, 0).off
    # An actual policy/base difference exercises all four scores in DPO.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.03)
    return model, disable


def old_scores(model, ids, k):
    logits = model(input_ids=ids, logits_to_keep=k + 1).logits[:, :-1].float()
    targets = ids[:, -k:]
    plain = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    score = F.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).sum()
    return plain, score


@pytest.mark.parametrize("kind", ["peft", "int8"])
@pytest.mark.parametrize("paired", [False, True])
def test_scores_and_all_adapter_gradients_match_pre_chunk_trainer(kind, paired):
    model, disable = small_cell(kind)
    pos, neg = torch.randint(0, 137, (1, 21)), torch.randint(0, 137, (1, 13))
    def run(chunked):
        def scores(ids, k):
            if not chunked:
                return old_scores(model, ids, k)
            nll = answer_nll(model, ids, k, 3)
            return nll / k, -nll

        loss, chosen = scores(pos, 10)
        if paired:
            _, rejected = scores(neg, 5)
            with torch.no_grad(), disable():
                _, ref_chosen = scores(pos, 10)
                _, ref_rejected = scores(neg, 5)
            margin = 0.1 * ((chosen - ref_chosen) - (rejected - ref_rejected))
            loss = loss - F.logsigmoid(margin)
        model.zero_grad(set_to_none=True)
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters() if p.requires_grad]
    old, old_grad = run(False)
    new, new_grad = run(True)
    torch.testing.assert_close(new, old)
    for actual, expected in zip(new_grad, old_grad):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)


def test_reads_each_vocab_tile_once_per_pass_and_saves_no_tile(monkeypatch):
    model, _ = small_cell("peft")
    base = model.get_base_model()
    base.lm_head.weight.requires_grad_(True)
    ids = torch.randint(0, 137, (2, 19))
    answer_nll(model, ids, 9).backward()
    expected = base.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    projected, saved = [], []
    project = lm_head_loss._project

    def record(hidden, weight, bias, scale, start, end):
        projected.append((len(hidden), start, end))
        return project(hidden, weight, bias, scale, start, end)

    def record_tensor(t):
        saved.append(tuple(t.shape))
        return t

    monkeypatch.setattr(lm_head_loss, "_project", record)
    with torch.autograd.graph.saved_tensors_hooks(record_tensor, lambda t: t):
        loss = answer_nll(model, ids, 9, 4, vocab_chunk=32)
    tiles = [(18, i, min(i + 32, 137)) for i in range(0, 137, 32)]
    assert projected == tiles
    assert not any(s in saved for s in [(18, 32), (18, 9), (18, 137)])
    loss.backward()
    assert projected == tiles * 2
    torch.testing.assert_close(base.lm_head.weight.grad, expected, atol=2e-6, rtol=1e-4)
    with torch.no_grad():
        answer_nll(model, ids, 9, 4, vocab_chunk=32)
    assert projected == tiles * 3
    with base.lm_head.register_forward_pre_hook(lambda m, inputs: projected.append(inputs[0].shape[-2])):
        answer_nll(model, ids[:1], 1, 4).backward()
    assert projected[-1] == 2  # short stock readout, including the discarded last prediction


@pytest.mark.parametrize("batch,k", [(1, 1), (2, 1), (2, 512)])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_vocab_loss_and_gradients_match_dense_ce(batch, k, bf16, backward):
    from transformers.modeling_outputs import BaseModelOutputWithPast
    torch.manual_seed(56)
    vocab, dim = 73, 16
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=dim, num_hidden_layers=0,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg)
    model.lm_head = torch.nn.Linear(dim, vocab, bias=True)
    ids = torch.randint(0, vocab, (batch, k + 3))
    ids[:, -1] = vocab - 1  # target in the last, partial vocabulary tile
    hidden = torch.randn(batch, k + 3, dim, requires_grad=backward)
    model.model.forward = lambda **kw: BaseModelOutputWithPast(last_hidden_state=hidden)
    params = (hidden, model.lm_head.weight, model.lm_head.bias)
    with torch.set_grad_enabled(backward), torch.autocast("cpu", torch.bfloat16, enabled=bf16):
        logits = model.lm_head(hidden[:, -k-1:-1]).float()
        expected = F.cross_entropy(logits.reshape(-1, vocab), ids[:, -k:].reshape(-1), reduction="sum")
        loss = answer_nll(model, ids, k, 64 if k > 1 else 1, vocab_chunk=11)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    assert loss.requires_grad == backward
    if backward:
        reference = torch.autograd.grad(expected * 0.37, params)
        actual = torch.autograd.grad(loss * 0.37, params)
        for got, want in zip(actual, reference):
            torch.testing.assert_close(got, want, atol=1e-5 if bf16 else 3e-6, rtol=0.016 if bf16 else 1e-5)


def test_one_target_with_large_common_offset_and_only_bias_trainable():
    hidden = torch.tensor([[1.0, -2.0]])
    weight = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    bias = torch.full((3,), 1e7, requires_grad=True)
    labels = torch.tensor([2])
    loss = lm_head_loss._VocabCrossEntropy.apply(hidden, labels, weight, bias, None, 2)
    expected = F.cross_entropy(F.linear(hidden, weight, bias), labels, reduction="sum")
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, bias), torch.autograd.grad(expected, bias))


def test_bf16_hidden_gradient_rounds_once_after_opposing_tiles():
    hidden = torch.zeros(1, 1, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.tensor([[0], [8], [0.03125], [0], [-8], [0], [0], [0]], dtype=torch.bfloat16)
    label = torch.tensor([0])
    expected = F.cross_entropy((hidden @ weight.T).float(), label, reduction="sum")
    loss = lm_head_loss._VocabCrossEntropy.apply(hidden, label, weight, None, None, 4)
    (grad,) = torch.autograd.grad(expected, hidden)
    assert grad.item() == 0.00390625  # rounding each tile before adding would give zero
    torch.testing.assert_close(torch.autograd.grad(loss, hidden)[0], grad, rtol=0, atol=0)


@pytest.mark.parametrize("label", [-100, -1])
def test_ternary_embedding_negative_indices_keep_ce_target_semantics(label):
    model, _ = small_cell("int8")
    emb = model.model.embed_tokens.weight.detach()
    codes, scales, _ = ternary_base._codes(emb)
    model.model.embed_tokens = ternary_base.TernaryEmbedding(codes, scales, emb.dtype)
    ids = torch.tensor([[1, 2, 3, 4, label]])
    if label == -100:
        torch.testing.assert_close(answer_nll(model, ids, 3, 1), answer_nll(model, ids, 3))
    else:
        with pytest.raises(IndexError):
            answer_nll(model, ids, 3, 1)


def test_frozen_ternary_head_decodes_each_weight_once_in_each_pass(monkeypatch):
    model, _ = small_cell("int8")
    codes = model.lm_head.codes
    decoded = []
    dequant = ternary_base._deq

    def count(block, scale, dtype):
        if block.untyped_storage().data_ptr() == codes.untyped_storage().data_ptr():
            decoded.append(block.numel())
        return dequant(block, scale, dtype)

    monkeypatch.setattr(ternary_base, "_deq", count)
    loss = answer_nll(model, torch.randint(0, 137, (2, 19)), 9, 4, vocab_chunk=32)
    assert sum(decoded) == codes.numel()
    decoded.clear()
    loss.backward()
    assert sum(decoded) == codes.numel()


def test_guard_counts_one_live_tile_and_caps_its_byte_budget():
    from training.grow import row_head_bytes
    model, _ = small_cell("int8")
    projected = lambda n: projection_rows(model, n, 4, vocab_chunk=16)
    assert projected(1) == 2  # stock readout
    assert projected(20) == 3  # ceil(20 * 16 / 137)
    assert row_head_bytes(137, 20, 20, projected) == 3 * 137 * 8
    assert row_head_bytes(137, 1, 20, projected) == 2 * 137 * 16 * 2
    assert row_head_bytes(137, 512, projection=projected) <= 4 * 137 * 16


def test_adapted_output_head_keeps_its_public_forward():
    model, _ = small_cell("int8")
    model.lm_head = ternary_base.TernaryLinear(torch.nn.Linear(128, 137), rank=2, alpha=2)
    with torch.no_grad():
        model.lm_head.lora_B.normal_()
    ids = torch.randint(0, 137, (1, 12))
    assert projection_rows(model, 9, 4) == 10
    torch.testing.assert_close(answer_nll(model, ids, 9, 4), old_scores(model, ids, 9)[0] * 9)


@pytest.mark.parametrize("cast", [contextlib.nullcontext, lambda: torch.autocast("cpu", torch.bfloat16)])
def test_held_out_and_raw_next_token_scoring(cast):
    from training.grow import kept_loss
    model, _ = small_cell("int8")
    ids = torch.randint(0, 137, (1, 20))
    k = ids.shape[1] - 1
    model.train()
    with torch.no_grad(), cast():
        expected = old_scores(model, ids, k)[0].item()
    actual = kept_loss(model, [("reading", (ids, k))], "cpu", cast, chunk_tokens=4)
    assert actual["reading"][0] == pytest.approx(expected, abs=2e-5)
    assert model.training


def test_other_architecture_keeps_its_public_forward():
    class OtherHead(torch.nn.Module):
        def forward(self, input_ids, logits_to_keep):
            # A model-specific score transformation must survive fallback.
            scores = 2 * F.one_hot(input_ids[:, -logits_to_keep:], 7).float() - 1
            return type("Output", (), {"logits": scores})()
    model = OtherHead()
    ids = torch.tensor([[2, 1, 1, 3, 1]])
    assert projection_rows(model, 4, 2) == 5
    torch.testing.assert_close(answer_nll(model, ids, 4, 2), 4 * old_scores(model, ids, 4)[0])
