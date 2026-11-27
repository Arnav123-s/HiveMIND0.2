"""Exercise the frozen-prefix pre-pass through grow.train, including its adapter export."""
import tempfile
import weakref
from unittest.mock import Mock

import peft
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from training import grow, ternary_base
from test_held_out import _lesson, _tiny, _train


@pytest.mark.parametrize("native,checkpoint_rows,cache", [
    pytest.param(True, False, "on", id="ternary-tied-head"),
    pytest.param(False, True, "on", id="peft-checkpointed-chunked"),
    pytest.param(False, False, "off", id="no-prefix-cache"),
    pytest.param(False, False, "full", id="too-little-cache-space"),
    pytest.param(False, False, "write-error", id="incomplete-cache-write"),
    pytest.param(False, False, "unwritable", id="cannot-create-cache-directory"),
])
def test_unload_preserves_training_and_kept_losses(tmp_path, monkeypatch, native, checkpoint_rows, cache):
    dna = tmp_path / "dna"
    tok = _tiny(dna)
    hidden = 32
    if native:
        hidden = ternary_base.GROUP
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=hidden // 2, tie_word_embeddings=True,
                          use_sliding_window=True, sliding_window=32, layer_types=["full_attention", "sliding_attention"])
        model = Qwen3ForCausalLM(cfg)
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim == 2:
                    p.copy_(torch.randint_like(p, -1, 2) * 0.01)
        model.save_pretrained(dna)
        del model
    key = "base_model.model.model.layers.1.self_attn.q_proj"
    parent = tmp_path / "parent.safetensors"
    save_file({key + ".lora_A.weight": torch.randn(4, hidden) * 0.1,
               key + ".lora_B.weight": torch.randn(hidden, 4) * 0.1}, parent)
    rows = [_lesson("held", 1)[0], _lesson("train", 2, "member")[0],
            _lesson("pair", 3, "member")[0] | {"type": "pair", "rejected": "print(-3)"}]
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("HIVE_GPU", "0")
    monkeypatch.setenv("HIVE_ROW_CHECKPOINT", str(int(checkpoint_rows)))
    monkeypatch.setenv("HIVE_PREFIX_CACHE", "0" if cache == "off" else "1")
    if cache == "full":
        monkeypatch.setenv("HIVE_PREFIX_DISK_GB", "0.00006")
    if checkpoint_rows:
        monkeypatch.setattr(grow.MemoryGuard, "row_mode", lambda *args: "checkpoint")
    if native:
        def admit(guard, pieces, subject):
            if subject.startswith("pair") and not guard.skips:
                guard.skips[subject] = 1
                return False  # A row skipped on its first training pass must still have both prefixes.
            return True
        monkeypatch.setattr(grow.MemoryGuard, "admit", admit)

    def run(patch, unload):
        if unload:
            patch.setenv("HIVE_UNLOAD_PREFIX", "1")
        else:
            patch.delenv("HIVE_UNLOAD_PREFIX", raising=False)
        if unload and cache == "write-error":
            put = grow._Prefix._put

            def fail_second_write(prefix, key, states):
                if prefix.kept:
                    raise OSError("disk full")
                return put(prefix, key, states)

            patch.setattr(grow._Prefix, "_put", fail_second_write)
        if unload and cache == "unwritable":
            temporary_directory = tempfile.TemporaryDirectory

            def make_directory(*args, **kwargs):
                if kwargs.get("prefix") == "hive-prefix-":
                    raise OSError("cannot create prefix cache")
                return temporary_directory(*args, **kwargs)

            patch.setattr(tempfile, "TemporaryDirectory", make_directory)
        frozen, embedded, suffix, kept, grads, objectives = [], [], [], [], [], []
        factory, name = (ternary_base, "load") if native else (peft, "get_peft_model")
        build = getattr(factory, name)

        def watch_model(*args, **kwargs):
            result = build(*args, **kwargs)
            decoder = result[0].model if native else result.get_base_model().model
            block = decoder.layers[0]
            weight = block.self_attn.q_proj.codes if native else block.self_attn.q_proj.weight
            frozen.extend(weakref.ref(obj) for obj in (block, weight, decoder.embed_tokens))
            decoder.embed_tokens.register_forward_pre_hook(
                lambda module, args: embedded.append((tuple(args[0][0].tolist()), torch.is_grad_enabled())))
            decoder.layers[1].register_forward_hook(lambda *args: suffix.append(True))
            return result

        patch.setattr(factory, name, watch_model)
        kept_loss = grow.kept_loss

        def measure(*args, **kwargs):
            released = [ref() is None for ref in frozen]
            if unload and cache == "on":
                assert all(released)
                assert len(embedded) == len(set(embedded)) == 4  # SFT, both sides of DPO, and the kept lesson
                assert not any(grad for ids, grad in embedded)
                if not kept:
                    assert not suffix  # Phase 0 stopped at the boundary
            else:
                assert not any(released)
            result = kept_loss(*args, **kwargs)
            kept.extend(result["held"])
            return result

        patch.setattr(grow, "kept_loss", measure)
        backward = torch.Tensor.backward

        def record_loss(tensor, *args, **kwargs):
            objectives.append(tensor.item())
            return backward(tensor, *args, **kwargs)

        clip = torch.nn.utils.clip_grad_norm_

        def record_grads(params, *args, **kwargs):
            grads.append(torch.cat([p.grad.detach().flatten() for p in params]))
            return clip(params, *args, **kwargs)

        patch.setattr(torch.Tensor, "backward", record_loss)
        patch.setattr(torch.nn.utils, "clip_grad_norm_", record_grads)
        meta, _ = _train(tmp_path, patch, rows, "", parent=str(parent), layers="last1", checkpoint=False,
                         pairs=True, int8_base=native, loss_chunk_tokens=4 if checkpoint_rows else 0)
        assert meta["used"] == 2 and meta["pairs"] > 0 and meta["kept_back"]["held"]["rows"] == 1
        if native:
            assert meta["memory"]["rows_skipped"] == 1
        assert len(kept) == 2 and len(grads) == meta["steps"] > 0 and len(objectives) == meta["passes"]
        adapter = tmp_path / "grow" / "t17+kb"
        saved = load_file(adapter / "adapter_model.safetensors")
        assert len(saved) == 14 and all(".layers.1." in k for k in saved)
        if unload and cache == "on":
            reloaded = peft.PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(dna), adapter)
            torch.testing.assert_close(peft.get_peft_model_state_dict(reloaded), saved)
        assert not list(tmp_path.rglob("*.bin"))
        return kept, objectives, grads, saved

    with monkeypatch.context() as patch:
        stock = run(patch, False)
    with monkeypatch.context() as patch:
        treatment = run(patch, True)
    torch.testing.assert_close(treatment, stock, rtol=1e-5, atol=1e-6)


def test_phase_zero_deadline_leaves_time_for_an_optimizer_update(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_UNLOAD_PREFIX", "1")
    monkeypatch.setenv("HIVE_PREFIX_CACHE", "1")
    monkeypatch.setenv("HIVE_GPU", "0")
    monkeypatch.setenv("TRAIN_PROFILE", "auto")
    clock = Mock(wraps=grow.time)
    clock.time.return_value = 1000.0
    monkeypatch.setattr(grow, "time", clock)
    prepared = []
    put = grow._Prefix._put

    def slow_prefix(prefix, key, states):
        put(prefix, key, states)
        if not torch.is_grad_enabled():
            prepared.append(key)
            clock.time.return_value = 1599.0  # One second remains in the ten-minute slice.

    monkeypatch.setattr(grow._Prefix, "_put", slow_prefix)
    rows = [_lesson("train", 0, "member")[0], _lesson("held", 1)[0]]
    meta, _ = _train(tmp_path, monkeypatch, rows, "", layers="last1", checkpoint=False, minutes=10)
    assert len(prepared) == 1  # The kept row still needs the full decoder.
    assert meta["steps"] > 0 and meta["kept_back"]["held"]["rows"] == 1
