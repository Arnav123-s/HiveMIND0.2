"""A micro-batch still trains and measures each row's own answer."""
from types import SimpleNamespace

import peft
import pytest
import torch
from safetensors.torch import load_file

from test_held_out import _lesson, _tiny, _train
from test_lm_head_loss import small_cell
from training import grow
from training.lm_head_loss import answer_nll, batch_projection_rows


@pytest.mark.parametrize("chunk", [0, 3])
def test_masked_answer_sums_and_gradients_match_unpadded_rows(chunk):
    model, _ = small_cell("peft")
    base = model.get_base_model()
    rows = [torch.randint(0, 137, (1, n)) for n in (5, 14, 8, 11)]
    k = [1, 8, 3, 5]
    expected = torch.stack([answer_nll(model, row, n) for row, n in zip(rows, k)])
    (expected / torch.tensor(k)).mean().backward()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad.clone() for p in params]
    model.zero_grad(set_to_none=True)

    ids = torch.randint(0, 137, (len(rows), 14))  # Padding contains valid, unrelated token IDs.
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    lengths = [row.numel() for row in rows]
    for i, (row, end, targets) in enumerate(zip(rows, lengths, k)):
        ids[i, :end], mask[i, :end] = row[0], 1
        labels[i, end - targets:end] = row[0, -targets:]

    decodes, projections = [], []
    def decoder_call(module, args):
        decodes.append(True)
    def head_call(module, args):
        projections.append(args[0].numel() // args[0].shape[-1])

    with base.model.register_forward_pre_hook(decoder_call), base.lm_head.register_forward_pre_hook(head_call):
        result = answer_nll(model, ids, chunk_tokens=chunk, attention_mask=mask, labels=labels)
        torch.testing.assert_close(result, expected)
        (result / torch.tensor(k)).mean().backward()
    assert len(decodes) == 1
    assert max(projections) == batch_projection_rows(model, lengths, k, chunk)
    assert sum(projections) == sum(k) * (2 if chunk else 1)
    assert all(g.abs().max() > 0 for g in grads)
    for p, grad in zip(params, grads):
        torch.testing.assert_close(p.grad, grad, atol=1e-6, rtol=1e-4)


def test_generic_model_can_return_its_whole_virtual_prompt_and_padded_head():
    class FullHead(torch.nn.Module):
        active_peft_config = SimpleNamespace(num_virtual_tokens=2)

        def forward(self, input_ids, logits_to_keep, attention_mask=None):
            requested.append(logits_to_keep)
            logits = 2 * torch.nn.functional.one_hot(input_ids, 5).float()
            prompt = logits.new_zeros(len(input_ids), 2, 5)
            return SimpleNamespace(logits=torch.cat((prompt, logits), dim=1))

    requested = []
    model = FullHead()
    ids = torch.tensor([[2, 3, 1, 1, 2, 2, 2, 1], [1, 2, 3, 2, 2, 4, 0, 0]])
    mask = torch.arange(8)[None, :] < torch.tensor([[8], [6]])
    labels = torch.full_like(ids, -100)
    labels[0, 6:], labels[1, 5] = ids[0, 6:], ids[1, 5]
    losses = answer_nll(model, ids, attention_mask=mask, labels=labels, chunk_tokens=2)
    scores = 2 * torch.nn.functional.one_hot(ids, 5).float()
    ce = torch.nn.functional.cross_entropy
    expected = [ce(scores[0, 5:7], ids[0, 6:8], reduction="sum"), ce(scores[1, 4:5], ids[1, 5:6])]
    torch.testing.assert_close(losses, torch.stack(expected))
    assert requested == [4]
    assert batch_projection_rows(model, [8, 6], [2, 1], 2) == 2 * (8 + 2)


@pytest.fixture
def train(tmp_path, monkeypatch):
    tok = _tiny(tmp_path / "dna")
    monkeypatch.setenv("HIVE_GPU", "0")
    monkeypatch.setenv("HIVE_ROW_CHECKPOINT", "0")
    monkeypatch.setenv("HIVE_PREFIX_DISK_GB", "1")
    monkeypatch.setenv("HIVE_TRAIN_PLATEAU", "off")

    def run(rows, batch, *, accum=5, epochs=1, cache="off", chunk=0, dropout=False, orpo=False,
            room=None, checkpoint=False, layers="last1"):
        calls, frozen, learned, steps, grads = [], [], [], [], []
        def record_tokens(module, args, kwargs):
            ids, mask = kwargs["input_ids"], kwargs.get("attention_mask")
            lengths = mask.sum(1).tolist() if mask is not None else [ids.shape[1]] * len(ids)
            if len(ids) > 1:
                assert mask is not None
                expected_mask = torch.arange(ids.shape[1])[None, :] < torch.tensor(lengths)[:, None]
                torch.testing.assert_close(mask.bool(), expected_mask)
            texts = [tok.decode(row[:n]) for row, n in zip(ids, lengths)]
            calls.append((torch.is_grad_enabled(), texts, lengths))

        build = peft.get_peft_model
        def watch_model(model, config):
            if not dropout:
                config.lora_dropout = 0  # Compare the math without independent random masks per forward.
            model = build(model, config)
            decoder = model.get_base_model().model
            decoder.register_forward_pre_hook(record_tokens, with_kwargs=True)
            decoder.layers[0].self_attn.register_forward_pre_hook(
                lambda *args: frozen.append(torch.is_grad_enabled()))
            return model

        clip = torch.nn.utils.clip_grad_norm_
        def record_update(params, *args, **kwargs):
            steps.append(len(learned))
            grads.append(torch.cat([p.grad.detach().flatten() for p in params]))
            return clip(params, *args, **kwargs)

        original_learned = grow.Study.learned
        def record_row(study, subject, loss):
            learned.append((subject, float(loss)))
            return original_learned(study, subject, loss)

        train_fn = grow.train
        with monkeypatch.context() as patch:
            if batch is None:
                patch.delenv("HIVE_TRAIN_BATCH", raising=False)
            else:
                patch.setenv("HIVE_TRAIN_BATCH", str(batch))
            patch.setenv("HIVE_PREFIX_CACHE", "0" if cache == "off" else "1")
            patch.setenv("HIVE_UNLOAD_PREFIX", "1" if cache == "unload" else "0")
            patch.setenv("HIVE_PAIRS_ORPO", str(int(orpo)))
            if room is not None:
                patch.setenv("HIVE_ROW_CHECKPOINT", str(int(checkpoint)))
                def available(guard):
                    if checkpoint:
                        guard.cost = 10 * guard.floor  # Simulate a previously observed ordinary peak.
                    return int(guard.spare + room * guard.floor)
                patch.setattr(grow.MemoryGuard, "available", available)
            patch.setattr(peft, "get_peft_model", watch_model)
            patch.setattr(torch.nn.utils, "clip_grad_norm_", record_update)
            patch.setattr(grow.Study, "learned", record_row)
            patch.setattr(grow, "train", lambda *a, **kw: train_fn(*a, **(kw | {"accum": accum, "epochs": epochs})))
            meta, _ = _train(tmp_path, patch, rows, "", layers=layers, checkpoint=False, pairs=True,
                             loss_chunk_tokens=chunk)
        adapter = tmp_path / "grow" / "t17+kb" / "adapter_model.safetensors"
        weights = {key: tensor.clone() for key, tensor in load_file(adapter).items()}
        return SimpleNamespace(meta=meta, calls=calls, frozen=frozen, learned=learned, steps=steps,
                               grads=grads, weights=weights)

    return run


def sft_rows(n):
    return [_lesson("batch", i, "member")[0] | {"output": "x" * (i + 1)} for i in range(n)]


@pytest.mark.parametrize(("batch", "cache", "chunk", "widths"), [
    (2, "off", 0, [2, 2, 1, 2, 2, 1, 2]),
    (4, "lazy", 3, [4, 1, 2, 3, 2]),
    (4, "unload", 3, [4, 1, 2, 3, 2]),
])
def test_real_training_grouping_tail_updates_and_frozen_prefixes(train, batch, cache, chunk, widths):
    rows = sft_rows(6) + [_lesson("held", 0)[0]]
    stock = train(rows, 1, epochs=2, chunk=chunk)
    packed = train(rows, batch, epochs=2, cache=cache, chunk=chunk)
    singles = [texts[0] for grad, texts, _ in stock.calls if grad]
    batches = [texts for grad, texts, _ in packed.calls if grad]
    assert [len(group) for group in batches] == widths
    assert packed.meta["sft_batches_by_size"] == {str(n): widths.count(n) for n in set(widths)}
    assert "sft_batches_by_size" not in stock.meta
    assert [text for group in batches for text in group] == singles
    assert len(set(singles)) == 6 and all(singles.count(text) == 2 for text in set(singles))
    assert packed.steps == stock.steps == [5, 10, 12]
    assert packed.meta["rows"] == packed.meta["used"] == 6
    assert packed.meta["passes"] == len(packed.learned) == 12
    assert packed.meta["subjects"] == stock.meta["subjects"]
    assert packed.meta["kept_back"] == stock.meta["kept_back"]
    assert [s for s, loss in packed.learned] == [s for s, loss in stock.learned]
    assert [loss for s, loss in packed.learned] == pytest.approx([loss for s, loss in stock.learned], abs=2e-6)
    torch.testing.assert_close(packed.grads, stock.grads, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(packed.weights, stock.weights, atol=1e-6, rtol=1e-4)
    if cache == "unload":
        assert packed.frozen == [False] * 7  # All six SFT rows and the held-out row, only in Phase 0.
    elif cache == "lazy":
        assert 0 < sum(packed.frozen) < len(batches)  # Cold, mixed and all-hit groups span two epochs.


@pytest.mark.parametrize("orpo", [False, True])
def test_default_and_one_agree_and_preference_rows_stay_single(train, orpo):
    rows = sft_rows(6)
    rows += [_lesson("pair", 0, "member")[0] | {"type": "pair", "output": "CHOSEN", "rejected": "REJECTED"}]
    stock = train(rows, None, orpo=orpo, dropout=True)
    one = train(rows, 1, orpo=orpo, dropout=True)
    assert one.calls == stock.calls and all(len(texts) == 1 for _, texts, _ in one.calls)
    assert one.learned == stock.learned and one.steps == stock.steps
    torch.testing.assert_close(one.grads, stock.grads, atol=0, rtol=0)
    torch.testing.assert_close(one.weights, stock.weights, atol=0, rtol=0)

    packed = train(rows, 4, orpo=orpo, dropout=True)
    assert any(len(texts) > 1 for _, texts, _ in packed.calls)
    for arm in (stock, packed):
        pairs = arm.meta["pairs"]
        assert pairs > 0
        for marker in ("CHOSEN", "REJECTED"):
            calls = [(grad, texts) for grad, texts, _ in arm.calls if any(marker in t for t in texts)]
            assert all(len(texts) == 1 for _, texts in calls)
            assert sum(grad for grad, _ in calls) == pairs
            assert sum(not grad for grad, _ in calls) == (0 if orpo else pairs)
        assert arm.meta["passes"] == len(arm.learned) == len(rows)


def test_incompatible_length_waits_for_its_own_forward(train):
    rows = sft_rows(6)
    rows[4]["output"] = "z" * 200
    run = train(rows, 4)
    lengths = [sizes for grad, _, sizes in run.calls if grad]
    longest = max(n for group in lengths for n in group)
    assert longest > 1.25 * min(n for group in lengths for n in group)
    assert [longest] in lengths and any(len(group) > 1 for group in lengths)
    assert all(max(group) <= 1.25 * min(group) for group in lengths)
    assert sum(map(len, lengths)) == run.meta["passes"] == len(rows)


def test_first_sweep_records_each_subjects_own_answer_loss(train):
    rows = [_lesson(kind, i, "member")[0] | {"output": kind * (i + 1)}
            for i, kind in enumerate(("red", "blue", "lime", "ochre"))]
    single = train(rows, 1, accum=4)
    batch = train(rows, 4, accum=4)
    assert [len(texts) for grad, texts, _ in batch.calls if grad] == [4]
    assert len(batch.learned) == len(dict(batch.learned)) == 4
    assert dict(batch.learned) == pytest.approx(dict(single.learned), abs=2e-6)
    assert all(s["passes"] == 1 for s in batch.meta["subjects"])


@pytest.mark.parametrize("due", ["read", "judged", "fresh"])
def test_ineligible_priority_pick_ends_the_batch_before_another_subject_can_train(due):
    study = grow.Study({due: [0], "other": [1]}, hard=["other"])
    if due == "fresh":
        assert study.pick() == ("other", 1)
        study.learned("other", 2.0)
    passes = study.n
    assert study.pick(lambda index: index == 1) is None
    assert study.n == passes
    assert study.pick() == (due, 0)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_memory_pressure_splits_groups_without_skipping_individually_admissible_rows(train, checkpoint):
    rows = sft_rows(6)
    single = train(rows, 1, chunk=3)
    width = max(n for _, _, lengths in single.calls for n in lengths)
    tight = train(rows, 4, chunk=3, room=2.5 * width, checkpoint=checkpoint,
                  cache="unload" if checkpoint else "off")
    groups = [texts for grad, texts, _ in tight.calls if grad]
    assert max(map(len, groups)) == 2
    assert [t for texts in groups for t in texts] == [texts[0] for grad, texts, _ in single.calls if grad]
    assert tight.meta["memory"]["rows_skipped"] == 0
    if checkpoint:
        assert tight.meta["memory"]["row_modes"] == {"ordinary": 0, "checkpoint": 6}
        assert tight.frozen == [False] * 6
    assert tight.meta["passes"] == len(tight.learned) == len(rows)
    assert tight.steps == single.steps == [5, 6]
    torch.testing.assert_close(tight.weights, single.weights, atol=1e-6, rtol=1e-4)


def test_batch_admission_charges_frozen_layers_above_an_interior_adapter(train):
    result = train(sft_rows(2), 2, layers="0-1")
    assert result.meta["layers"] == [0, 0]
    assert result.meta["sft_batches_by_size"] == {"2": 1}
    # Backpropagation traverses both 32-wide float32 layers of the tiny model.
    assert result.meta["memory"]["formula_per_piece_mb"] == round(2 * 32 * 4 * 46 / 2**20, 3)
