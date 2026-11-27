import os

import pytest
import torch

from training import ternary_base as tb


def _ternary_linear(out=8, inp=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-1, 2, (out, inp), generator=g).float()
    scale = torch.rand(out, inp // tb.GROUP, generator=g) + 0.5
    lin = torch.nn.Linear(inp, out, bias=False)
    lin.weight.data = (codes.view(out, -1, tb.GROUP) * scale.unsqueeze(-1)).view(out, inp)
    return lin


def test_held_ternary_it_computes_what_the_weights_compute():
    lin = _ternary_linear()
    t = tb.TernaryLinear(lin)
    assert t.codes.dtype == (torch.uint8 if os.environ.get("HIVE_TERNARY_VNNI") == "1" else torch.int8)
    assert t.error(lin) < 0.01
    x = torch.randn(3, 256, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(t(x), lin(x), rtol=0.01, atol=0.1)


def test_the_gradient_goes_through_without_keeping_the_weights():
    lin = _ternary_linear()
    t = tb.TernaryLinear(lin, rank=4, alpha=8)
    t.lora_B.data.normal_()
    x = torch.randn(3, 256, requires_grad=True)
    t(x).sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    ref = lin(x2) + x2 @ t.lora_A.detach().t() @ t.lora_B.detach().t() * 2.0
    ref.sum().backward()
    assert torch.allclose(x.grad, x2.grad, rtol=0.01, atol=0.1)
    assert t.lora_A.grad is not None and t.lora_B.grad is not None


def test_the_edit_can_be_switched_off(monkeypatch):
    lin = _ternary_linear()
    t = tb.TernaryLinear(lin, rank=4, alpha=8)
    t.lora_B.data.normal_()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = t
    cell = tb.Cell(M(), [0], 4, 8, 0.0)
    x = torch.randn(2, 256, generator=torch.Generator().manual_seed(0))
    with cell.off():
        assert torch.allclose(t(x), lin(x), rtol=0.01, atol=0.1)
    assert not torch.allclose(t(x), lin(x), rtol=0.01, atol=0.1)


def test_weights_that_are_not_ternary_are_refused():
    lin = torch.nn.Linear(256, 8, bias=False)
    t = tb.TernaryLinear(lin)
    assert t.error(lin) > 0.1


def test_the_c_routine_gives_the_very_same_numbers():
    if tb.kernel() is None:
        import pytest
        pytest.skip("no C compiler on this machine: the PyTorch way is used")
    g = torch.Generator().manual_seed(1)
    out, inp = 2048, 1024  # big enough for the routine to be used
    codes = torch.randint(-1, 2, (out, inp), generator=g, dtype=torch.int8)
    scale = (torch.rand(out, inp // tb.GROUP, generator=g) + 0.5).to(torch.bfloat16)
    slow = (codes.view(out, -1, tb.GROUP).to(torch.bfloat16) * scale.view(out, -1, 1)).view(out, inp)
    fast = tb._deq(codes, scale, torch.bfloat16)
    assert fast.dtype == torch.bfloat16 and torch.equal(fast.view(torch.int16), slow.view(torch.int16))
    assert tb._deq(codes, scale, torch.bfloat16) is fast  # one buffer for each shape, written again at every call


def test_a_tall_matrix_is_computed_in_slices_and_word_vectors_are_read_by_row(monkeypatch):
    monkeypatch.setattr(tb, "SLICE", 3)
    lin = _ternary_linear(out=8)
    t = tb.TernaryLinear(lin)
    x = torch.randn(2, 5, 256, generator=torch.Generator().manual_seed(0), requires_grad=True)
    y = t(x)
    assert torch.allclose(y, lin(x), rtol=0.01, atol=0.1)
    y.sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    lin(x2).sum().backward()
    assert torch.allclose(x.grad, x2.grad, rtol=0.01, atol=0.1)
    # the table of word vectors held as codes: a row is the same numbers as the matrix's row
    emb = tb.TernaryEmbedding(t.codes, t.scale, torch.float32)
    ids = torch.tensor([[0, 7, 3]])
    assert emb(ids).shape == (1, 3, 256) and torch.allclose(emb(ids)[0, 1], lin.weight[7], rtol=0.01, atol=0.01)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_the_whole_model_is_read_matrix_by_matrix(tmp_path, monkeypatch, flag):
    """A small model saved as the published ones are, with ternary weights: loaded held ternary it gives the logits
    the ordinary loading gives."""
    import safetensors
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    monkeypatch.setenv("HIVE_TERNARY_VNNI", flag)
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=4097, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=64, max_position_embeddings=64, tie_word_embeddings=False)
    ref = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if p.dim() == 2:  # make every matrix ternary in groups of 128, as the published weights are
                g = p.view(p.shape[0], -1, tb.GROUP)
                s = g.abs().mean(-1, keepdim=True)
                p.copy_((torch.sign(g) * (g.abs() > 0.5 * s) * s).view_as(p))
    ref.save_pretrained(tmp_path)
    matrices = {name for name, param in ref.named_parameters() if param.ndim == 2}
    full, tiles = [], []
    safe_open = safetensors.safe_open

    class Reader:
        def __init__(self, *args, **kwargs):
            self.file = safe_open(*args, **kwargs)

        def __getattr__(self, attr):
            return getattr(self.file, attr)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *exc):
            file, self.file = self.file, None
            return file.__exit__(*exc)

        def get_tensor(self, name):
            if name in matrices:
                full.append(name)
            return self.file.get_tensor(name)

        def get_slice(self, name):
            view = self.file.get_slice(name)
            if name != "model.embed_tokens.weight":
                return view

            class TableRows:
                def get_shape(_):
                    return view.get_shape()

                def __getitem__(_, rows):
                    if tiles:
                        assert tiles[-1][0].file is None  # The previous tile reader has closed.
                    tiles.append((self, rows.indices(cfg.vocab_size)))
                    return view[rows]
            return TableRows()

    with monkeypatch.context() as patch:
        patch.setattr(safetensors, "safe_open", Reader)
        model, cell = tb.load(tmp_path, rank=4, alpha=8, layers=[1], dtype=torch.float32, log=lambda *a, **k: None)
    if flag == "1":
        assert not full and [bounds for _, bounds in tiles] == [(0, 4096, 1), (4096, 4097, 1)]
        assert all(reader.file is None for reader, _ in tiles)
    else:
        assert not tiles and sorted(full) == sorted(matrices)
    codes, scale, _ = tb._codes(ref.model.embed_tokens.weight.detach())
    assert torch.equal(model.model.embed_tokens.codes, codes) and torch.equal(model.model.embed_tokens.scale, scale)
    ids = torch.tensor([[0, 9, 1, 4095, 4096]])
    with torch.no_grad():
        a, b = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)(input_ids=ids).logits, model(input_ids=ids).logits
    assert torch.allclose(a, b, rtol=0.02, atol=0.02)
    assert isinstance(model.lm_head, tb.TernaryLinear) and isinstance(model.model.embed_tokens, tb.TernaryEmbedding)
    assert len(cell.params) == 2 * 7 and all(p.requires_grad for p in cell.params)  # the edit on layer 1 only
    assert not any(p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)


def test_on_a_cpu_slow_at_16_bit_the_big_multiplications_are_done_in_32_bit(monkeypatch):
    g = torch.Generator().manual_seed(2)
    out, inp = 2048, 1024
    codes = torch.randint(-1, 2, (out, inp), generator=g, dtype=torch.int8)
    scale = (torch.rand(out, inp // tb.GROUP, generator=g) + 0.5).to(torch.bfloat16)
    x = torch.randn(3, inp, generator=g).to(torch.bfloat16)
    monkeypatch.setattr(tb, "_quick16", True)
    quick = tb._times(x, codes, scale, True)
    monkeypatch.setattr(tb, "_quick16", False)
    slow = tb._times(x, codes, scale, True)
    assert slow.dtype == torch.bfloat16 and torch.allclose(quick.float(), slow.float(), rtol=0.05, atol=0.5)
    w32 = tb._deq(codes, scale, torch.float32)  # the 32-bit numbers are the very same values
    assert w32.dtype == torch.float32 and torch.equal(w32, (codes.view(out, -1, tb.GROUP).float() * scale.float().view(out, -1, 1)).view(out, inp))
    back = tb._times(torch.randn(3, out, generator=g).to(torch.bfloat16), codes, scale, False)
    assert back.shape == (3, inp) and back.dtype == torch.bfloat16


def test_packed_codes_checkpoints_and_embedding(monkeypatch):
    c = torch.tensor([[-1, 0, 1, 0]], dtype=torch.int8).repeat(7, 64)
    s = torch.arange(1, 15).reshape(7, 2).bfloat16()
    monkeypatch.setenv("HIVE_TERNARY_VNNI", "0")
    old = tb.TernaryLinear(None, codes=(c, s, 0))
    monkeypatch.setenv("HIVE_TERNARY_VNNI", "1")
    new = tb.TernaryLinear(None, codes=(c, s, 0))
    assert old.codes.dtype == torch.int8 and new.codes.dtype == torch.uint8
    assert torch.all(new.codes == 0b01100100) and torch.equal(tb._unpack(new.codes), c)
    assert new.codes.nbytes + s.nbytes == c.numel() * 17 // 64
    assert (new.out_features, new.in_features) == c.shape
    old.codes.neg_()
    new.load_state_dict(old.state_dict())
    assert torch.equal(tb._unpack(new.codes), old.codes)
    old.codes.zero_()
    old.load_state_dict(new.state_dict())
    assert torch.equal(old.codes, tb._unpack(new.codes))

    emb = tb.TernaryEmbedding(c, s, torch.float32)
    emb.load_state_dict(old.state_dict())
    ids = torch.tensor([[5, 0, 5], [6, 2, 4]])
    expected = tb._deq(old.codes, old.scale, torch.float32)[ids]
    unpack = tb._unpack

    def unpack_rows(rows):
        assert rows.numel() <= ids.numel() * emb.codes.shape[1]
        return unpack(rows)

    monkeypatch.setattr(tb, "_unpack", unpack_rows)
    assert torch.equal(emb(ids), expected)
    assert emb(ids[:0]).shape == (0, 3, 256)


@pytest.mark.parametrize("backend", ["native", "no-openmp", "no-compiler", "torch"])
def test_packed_dequant_matches_stock_tile_bytes(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("HIVE_TERNARY_VNNI", "1")
    if backend != "native":
        monkeypatch.setattr(tb, "_kernel", None)
        monkeypatch.setattr(tb, "_pool", None)
        monkeypatch.setattr(tb, "_packed_openmp", False)
        monkeypatch.setenv("HIVE_TERNARY_HOME", str(tmp_path))
    if backend == "no-openmp":
        compile_c = tb._compile

        def without_openmp(name, source, flags, **kwargs):
            if "-fopenmp" in flags:
                raise OSError("OpenMP runtime unavailable")
            return compile_c(name, source, flags, **kwargs)

        monkeypatch.setattr(tb, "_compile", without_openmp)
    if backend in ("native", "no-openmp"):
        if tb.kernel() is None:
            pytest.skip("native tile comparison requires a C compiler")
        if backend == "no-openmp":
            assert not tb._packed_openmp

        def reject_unpack(*args):
            pytest.fail("native dequantization expanded the tile to an intermediate int8 matrix")
        monkeypatch.setattr(tb, "_unpack", reject_unpack)
    else:
        if backend == "no-compiler":
            monkeypatch.delenv("HIVE_DEQ", raising=False)
            monkeypatch.setenv("PATH", str(tmp_path))
        else:
            monkeypatch.setenv("HIVE_DEQ", "torch")
        assert tb.kernel() is None

    gen = torch.Generator().manual_seed(5)
    c = torch.randint(-1, 2, (1025, 1024), generator=gen, dtype=torch.int8)
    s = torch.rand(1025, 8, generator=gen).bfloat16()
    c[1:3, :3] = torch.tensor([-1, 0, 1])
    s[1, 0], s[2, 0] = 0.0, -0.0
    layer = tb.TernaryLinear(None, codes=(c, s, 0))
    assert layer.codes.dtype == torch.uint8
    for dtype in (torch.bfloat16, torch.float32):
        # A CCE vocabulary tile starting at a nonzero row. Clone before the scratch tile is reused.
        ref = tb._deq(c[1:], s[1:], dtype).clone()
        packed = tb._deq(layer.codes[1:], layer.scale[1:], dtype)
        assert torch.signbit(ref[0, 0]) and torch.signbit(ref[1, 2])
        assert torch.equal(packed.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head", [False, True], ids=["lora", "sliced-head"])
def test_packing_preserves_stock_outputs_and_gradients(monkeypatch, dtype, head):
    out, inp, rank = (tb.SLICE + 1, tb.GROUP, 0) if head else (33, 2 * tb.GROUP, 4)
    lin = _ternary_linear(out, inp)
    gen = torch.Generator().manual_seed(19)
    inputs = torch.randn(2, 3, inp, generator=gen).to(dtype)
    grad_out = torch.randn(2, 3, out, generator=gen).to(dtype)
    a = torch.randn(rank, inp, generator=gen) * 0.01
    b = torch.randn(out, rank, generator=gen) * 0.01
    deq = tb._deq

    def dequantize_tile(c, s, dtype):
        assert c.shape[0] <= tb.SLICE
        return deq(c, s, dtype)
    monkeypatch.setattr(tb, "_deq", dequantize_tile)

    def run(flag):
        monkeypatch.setenv("HIVE_TERNARY_VNNI", flag)
        layer = tb.TernaryLinear(lin, rank=rank, alpha=2 * rank)
        assert layer.codes.dtype == (torch.uint8 if flag == "1" else torch.int8)
        if rank:
            with torch.no_grad():
                layer.lora_A.copy_(a)
                layer.lora_B.copy_(b)
        x = inputs.detach().requires_grad_(True)
        y = layer(x)
        grads = torch.autograd.grad(y, (x, *layer.parameters()), grad_out)
        assert set(dict(layer.named_buffers())) == {"codes", "scale"}
        assert y.dtype == grads[0].dtype == dtype
        return [value.detach().contiguous().view(torch.uint8) for value in (y, *grads)]

    stock = run("0")
    assert all(torch.equal(actual, expected) for actual, expected in zip(run("1"), stock, strict=True))
