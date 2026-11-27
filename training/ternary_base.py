"""The DNA held in memory as what it is: ternary. Training a cell on a 4B or 8B DNA.

The published unpacked weights of a Ternary-Bonsai model are ternary in groups of 128 along each row: every weight
is -s, 0 or +s for its group's scale s (checked on the 1.7B: within 0.7% of the scale, the rounding of the stored
floats). `grow.py` loaded them as 16-bit numbers, 2 bytes a weight: 16 GB for an 8B DNA, more than a free runner
has. By default a frozen layer keeps int8 codes and one bf16 scale per group. Set HIVE_TERNARY_VNNI=1 before
loading to pack four codes per byte in the decoder, word embeddings and vocabulary head. Forward and backward
dequantize the required weights into the same BF16 or FP32 PyTorch matmul used for int8 storage.
The cell's edit is LoRA, written in the layout PEFT saves (adapter_config.json, adapter_model.safetensors),
so `convert_lora_to_gguf.py` and the ternary runtime take it as before.

Found on the first 8B generations (2026-11-14 and 15): one cell in four died in training, its runner out of memory
(a 16 GB runner; the job "lost communication" or sat for five hours in its first long row). Three things took the
room, and each is gone now:
  - the whole DNA was first loaded as 16-bit numbers (16 GB) and only then turned into codes. Now every matrix is
    read from the file by itself and turned into codes at once: nothing but codes is ever held.
  - the table of word vectors and the output layer stayed 16-bit (1.2 GB each on the 8B, which keeps two). They
    are ternary too and are held as codes; the output layer computes in slices of its rows.
  - the edit's own path copied each layer's input as 32-bit numbers. It computes in the input's own type now.

    model, cell = load(src, rank=16, layers=[...])     # cell.params: what training updates
    with cell.off(): ...                               # the DNA without the edit (the reference for --pairs)
    cell.save(folder)                                  # PEFT layout
"""
import contextlib
import ctypes
import functools
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import torch

GROUP = 128
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SLICE = 16384  # rows of a very tall matrix (the output layer: one row a word) turned into numbers at a time


# Turning a layer's codes back into numbers is most of a training row's time on an 8B DNA (measured 2026-11-15: about
# 9 of 12 seconds of a forward pass, more than the multiplying itself): PyTorch does it in three passes over memory on
# one core. This small C routine does it in one pass on every core: a 16-bit number is the group's scale with its
# sign bit set where the code is -1, and zero where the code is 0. Built once with the machine's compiler; where
# there is none, the PyTorch way is used.
_C = r"""
#include <stdint.h>
void hive_deq16(const int8_t *codes, const uint16_t *scale, uint16_t *out, long r0, long r1, long inp, long group) {
    for (long r = r0; r < r1; r++) {
        const int8_t *c = codes + r * inp;
        const uint16_t *s = scale + r * (inp / group);
        uint16_t *o = out + r * inp;
        for (long g = 0; g < inp / group; g++) {
            const uint16_t sb = s[g];
            for (long i = 0; i < group; i++) {
                const int16_t ci = c[g * group + i];
                o[g * group + i] = (uint16_t)((sb ^ (uint16_t)(ci & 0x8000)) & (uint16_t)(-(ci & 1)));
            }
        }
    }
}
void hive_deq32(const int8_t *codes, const uint32_t *scale, uint32_t *out, long r0, long r1, long inp, long group) {
    for (long r = r0; r < r1; r++) {
        const int8_t *c = codes + r * inp;
        const uint32_t *s = scale + r * (inp / group);
        uint32_t *o = out + r * inp;
        for (long g = 0; g < inp / group; g++) {
            const uint32_t sb = s[g];
            for (long i = 0; i < group; i++) {
                const int32_t ci = c[g * group + i];
                o[g * group + i] = (sb ^ ((uint32_t)ci & 0x80000000u)) & (uint32_t)(-(ci & 1));
            }
        }
    }
}

/* Code 00 is -scale, 01 is zero and 10 is +scale. Copying the bf16
 * bits into the high half of an fp32 word widens the scale exactly. */
static inline void expand_group(const uint8_t *restrict codes, uint16_t scale, void *restrict out, int fp32) {
    for (int j = 0; j < HIVE_GROUP / 4; ++j) {
        unsigned four = codes[j];
        for (int lane = 0; lane < 4; ++lane) {
            unsigned code = four >> (2 * lane);
            uint16_t sign = (~code & 2u) << 14;
            uint16_t nonzero = (code & 1u) - 1u;
            uint16_t value = (scale ^ sign) & nonzero;
            if (fp32) ((uint32_t *)out)[4*j + lane] = (uint32_t)value << 16;
            else ((uint16_t *)out)[4*j + lane] = value;
        }
    }
}
void hive_expand(const uint8_t *restrict codes, const uint16_t *restrict scales, void *restrict out,
                 long first, long end, int threads, int fp32) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    #endif
    for (long g = first; g < end; ++g) {
        const uint8_t *group = codes + g * (HIVE_GROUP / 4);
        if (fp32) expand_group(group, scales[g], (uint32_t *)out + g*HIVE_GROUP, 1);
        else expand_group(group, scales[g], (uint16_t *)out + g*HIVE_GROUP, 0);
    }
}
"""
_kernel, _pool, _scratch = None, None, {}
_packed_openmp = False
_quick16 = None


@functools.lru_cache(maxsize=1)
def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as info:
            return next((line for line in info if line.startswith(("flags", "Features"))), "")
    except OSError:
        return ""


def quick16():
    """Select bf16 on AVX512-BF16 or AMX-BF16, including CPUs with AVX512-BF16 alone.
    This tests instruction availability. HIVE_MATMUL=f32 or bf16 overrides the choice."""
    global _quick16
    if _quick16 is None:
        said = os.environ.get("HIVE_MATMUL", "")
        flags = _cpu_flags()
        _quick16 = said == "bf16" or (said != "f32" and ("avx512_bf16" in flags or "amx_bf16" in flags))
    return _quick16


def _compile(name, source, flags, cpu_key=""):
    """Publish a complete shared object even if several loaders compile it concurrently."""
    home = Path(os.environ.get("HIVE_TERNARY_HOME", Path.home() / ".cache" / "hive" / "ternary"))
    build_key = " ".join([platform.machine(), cpu_key, *flags]).encode()
    digest = hashlib.sha256(source + build_key).hexdigest()[:16]
    so = home / f"{name}_{digest}.so"
    if not so.exists():
        home.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home) as build:
            src = Path(build) / f"{name}.c"
            src.write_bytes(source)
            tmp = Path(build) / so.name
            subprocess.run(["cc", "-O3", *flags, "-shared", "-fPIC", "-o", str(tmp), str(src)],
                           check=True, capture_output=True, timeout=120)
            os.replace(tmp, so)
    return ctypes.CDLL(str(so))


def kernel():
    """The C routine, built on first use (None where it cannot be built: no compiler, or HIVE_DEQ=torch)."""
    global _kernel, _pool, _packed_openmp
    if _kernel is not None:
        return _kernel or None
    _kernel = False
    if os.environ.get("HIVE_DEQ") == "torch":
        return None
    try:
        flags = _cpu_flags()
        cflags = [f"-DHIVE_GROUP={GROUP}"] + (["-march=native"] if flags else [])
        lib = None
        if _packed_storage():
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                lib = _compile("deq", _C.encode(), [*cflags, "-fopenmp"], cpu_key=flags)
        _packed_openmp = lib is not None
        if lib is None:
            lib = _compile("deq", _C.encode(), cflags, cpu_key=flags)
        for f in (lib.hive_deq16, lib.hive_deq32):
            f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long] * 4
            f.restype = None
        lib.hive_expand.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long] * 2 + [ctypes.c_int] * 2
        lib.hive_expand.restype = None
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max(1, torch.get_num_threads()))
        _kernel = {torch.bfloat16: lib.hive_deq16, torch.float32: lib.hive_deq32,
                   (torch.uint8, torch.bfloat16): lib.hive_expand, (torch.uint8, torch.float32): lib.hive_expand}
    except (OSError, subprocess.SubprocessError, AttributeError):
        _kernel = False
    return _kernel or None


def _packed_storage():
    return os.environ.get("HIVE_TERNARY_VNNI", "0") == "1"


def _pack(codes):
    """Four consecutive signed trits as (trit + 1), least significant pair first."""
    c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 4, 4).to(torch.uint8) + 1
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def _unpack(codes):
    return (torch.stack([((codes >> shift) & 3) for shift in (0, 2, 4, 6)], -1)
            .flatten(-2).to(torch.int8) - 1)


def _deq(codes, scale, dtype):
    """The numbers of a layer held as codes, as 16-bit or 32-bit numbers. Written by the C routine into one buffer
    for each shape and type of matrix, used again at every call (the caller multiplies with it at once and keeps
    no reference to it)."""
    packed = codes.dtype == torch.uint8
    out, width = codes.shape
    inp = width * 4 if packed else width
    if inp % GROUP or scale.shape != (out, inp // GROUP):
        raise ValueError("ternary weights must contain whole scale groups")
    native = (codes.device.type == scale.device.type == "cpu" and scale.dtype == torch.bfloat16
              and out * inp >= 1 << 20 and codes.is_contiguous() and scale.is_contiguous())
    fast = (kernel() or {}).get((torch.uint8, dtype) if packed else dtype) if native else None
    if fast is None:
        if packed:
            codes = _unpack(codes)
        return (codes.view(out, inp // GROUP, GROUP).to(dtype) * scale.view(out, inp // GROUP, 1).to(dtype)).view(out, inp)
    buf = _scratch.get((out, inp, dtype))
    if buf is None:
        buf = _scratch[(out, inp, dtype)] = torch.empty(out, inp, dtype=dtype)
    if dtype != torch.bfloat16 and not packed:
        scale = scale.float()  # one number a group: small
    n = _pool._max_workers
    c, sc, o = codes.data_ptr(), scale.data_ptr(), buf.data_ptr()
    if packed and _packed_openmp:
        fast(c, sc, o, 0, scale.numel(), n, dtype == torch.float32)
        return buf
    # Without OpenMP the existing CPU workers divide packed groups or int8 rows.
    count, args = (scale.numel(), (1, dtype == torch.float32)) if packed else (out, (inp, GROUP))
    step = -(-count // n)
    list(_pool.map(lambda start: fast(c, sc, o, start, min(count, start + step), *args), range(0, count, step)))
    return buf


def _times(x, codes, scale, t):
    """x times the layer (t: its transpose), multiplied in the type this CPU is quick in; the result in x's type."""
    if x.dtype == torch.bfloat16 and not quick16():
        w = _deq(codes, scale, torch.float32)
        return (x.float() @ (w.t() if t else w)).to(x.dtype)
    w = _deq(codes, scale, x.dtype)
    return x @ (w.t() if t else w)


def _codes(w):
    """A stored matrix as (codes, scales, how far the worst group is from ternary, relative to its scale)."""
    out, inp = w.shape
    if inp % GROUP:
        raise ValueError(f"{inp} inputs is not a whole number of groups of {GROUP}")
    packed = _packed_storage()
    codes = torch.empty(out, inp // 4 if packed else inp, dtype=torch.uint8 if packed else torch.int8)
    scale = torch.empty(out, inp // GROUP, dtype=torch.bfloat16)
    worst = 0.0
    for i in range(0, out, 4096):  # in slices of rows: a matrix of 150,000 rows is never held twice as 32-bit numbers
        g = w[i:i + 4096].float().view(-1, inp // GROUP, GROUP)
        s = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
        c = (g / s).round().clamp(-1, 1)
        worst = max(worst, ((c * s - g).abs().amax(-1) / s.squeeze(-1)).max().item())
        c = c.to(torch.int8).reshape(-1, inp)
        codes[i:i + 4096] = _pack(c) if packed else c
        scale[i:i + 4096] = s.view(-1, inp // GROUP).to(torch.bfloat16)
    return codes, scale, worst


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scale):
        ctx.save_for_backward(codes, scale)
        if codes.shape[0] <= SLICE:
            return _times(x, codes, scale, True)
        return torch.cat([_times(x, codes[i:i + SLICE], scale[i:i + SLICE], True) for i in range(0, codes.shape[0], SLICE)], -1)  # each slice is used before the next is made

    @staticmethod
    def backward(ctx, g):
        codes, scale = ctx.saved_tensors
        if codes.shape[0] <= SLICE:
            return _times(g, codes, scale, False), None, None
        out = None
        for i in range(0, codes.shape[0], SLICE):
            part = _times(g[..., i:i + SLICE], codes[i:i + SLICE], scale[i:i + SLICE], False)
            out = part if out is None else out + part
        return out, None, None


class _TernaryBuffers(torch.nn.Module):
    def _store(self, codes, scale):
        out, groups = scale.shape
        inp = groups * GROUP
        packed = codes.dtype == torch.uint8
        if codes.dtype not in (torch.uint8, torch.int8) or codes.shape != (out, inp // 4 if packed else inp):
            raise ValueError("ternary codes and group scales have incompatible dtype or shape")
        if not packed and _packed_storage():
            codes = _pack(codes)
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)
        return out, inp

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Both storage modes can read checkpoints written by the other, including
        # int8 checkpoints from before packing. Keep the module's chosen format.
        key = prefix + "codes"
        c = state_dict.get(key)
        if c is not None and c.ndim == 2 and c.shape[0] == self.codes.shape[0]:
            if c.dtype == torch.int8 and self.codes.dtype == torch.uint8 and c.shape[1] == self.codes.shape[1] * 4:
                state_dict[key] = _pack(c)
            elif c.dtype == torch.uint8 and self.codes.dtype == torch.int8 and c.shape[1] * 4 == self.codes.shape[1]:
                state_dict[key] = _unpack(c)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class TernaryLinear(_TernaryBuffers):
    """A frozen ternary layer (codes and group scales) with an optional LoRA edit beside it.
    Made from an ordinary layer, or from the matrix as stored (and its bias)."""

    def __init__(self, linear, rank=0, alpha=0, dropout=0.0, bias=None, codes=None):
        super().__init__()
        if codes is None:
            w = linear.weight.detach() if hasattr(linear, "weight") else linear
            bias = linear.bias if hasattr(linear, "weight") else bias
            codes = _codes(w)
        c, s, self.worst = codes
        out, inp = self._store(c, s)
        self.bias = None if bias is None else torch.nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features, self.rank, self.on = inp, out, rank, True
        if rank:
            self.lora_A = torch.nn.Parameter(torch.empty(rank, inp, dtype=torch.float32))
            self.lora_B = torch.nn.Parameter(torch.zeros(out, rank, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # as PEFT starts an edit: A random, B zero
            self.scaling = alpha / rank
            self.drop = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()

    def error(self, linear):
        """The largest difference, relative to the group's scale, between the layer as held and the weights given."""
        w = linear.weight.detach().float()
        return ((_deq(self.codes, self.scale, torch.float32) - w).abs().view(self.out_features, -1, GROUP).amax(-1)
                / self.scale.float()).max().item()

    def forward(self, x):
        # in the type the pass computes in: the last norm hands 32-bit numbers to the output layer, and turning its
        # 600 million codes into 32-bit numbers took 10 of a row's 14 seconds (profiled 2026-11-15)
        if torch.is_autocast_enabled("cpu") and x.dtype != torch.get_autocast_dtype("cpu"):
            x = x.to(torch.get_autocast_dtype("cpu"))
        y = _Matmul.apply(x, self.codes, self.scale)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        if self.rank and self.on:
            # in the input's own type: the two small matrices are converted, never the layer's whole input
            y = y + (self.drop(x) @ self.lora_A.to(x.dtype).t() @ self.lora_B.to(x.dtype).t()) * self.scaling
        return y


class TernaryEmbedding(_TernaryBuffers):
    """The table of word vectors held as codes: only the rows a text uses are turned into numbers."""

    def __init__(self, codes, scale, dtype):
        super().__init__()
        self.num_embeddings, self.embedding_dim = self._store(codes, scale)
        self.dtype_out = dtype

    def forward(self, ids):
        c, s = self.codes[ids], self.scale[ids]
        if c.dtype == torch.uint8:
            c = _unpack(c)
        return (c.reshape(*s.shape, GROUP).to(self.dtype_out) * s.unsqueeze(-1).to(self.dtype_out)).reshape(*ids.shape, self.embedding_dim)


class Cell:
    """The LoRA edit spread over the TernaryLinear layers: what training updates, and how it is saved."""

    def __init__(self, model, layers, rank, alpha, dropout):
        self.layers = {n: m for n, m in model.named_modules() if isinstance(m, TernaryLinear) and m.rank}
        self.params = [p for m in self.layers.values() for p in (m.lora_A, m.lora_B)]
        self.config = {"r": rank, "lora_alpha": alpha, "lora_dropout": dropout, "layers": layers}

    @contextlib.contextmanager
    def off(self):
        for m in self.layers.values():
            m.on = False
        try:
            yield
        finally:
            for m in self.layers.values():
                m.on = True

    def save(self, folder, base_name):
        from peft import LoraConfig
        from safetensors.torch import save_file
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for n, m in self.layers.items():
            tensors[f"base_model.model.{n}.lora_A.weight"] = m.lora_A.detach().float().contiguous()
            tensors[f"base_model.model.{n}.lora_B.weight"] = m.lora_B.detach().float().contiguous()
        save_file(tensors, str(folder / "adapter_model.safetensors"))
        LoraConfig(r=self.config["r"], lora_alpha=self.config["lora_alpha"], lora_dropout=self.config["lora_dropout"],
                   target_modules=list(TARGETS), layers_to_transform=self.config["layers"], task_type="CAUSAL_LM",
                   base_model_name_or_path=base_name).save_pretrained(str(folder))


def _trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def load(src, rank=16, alpha=32, dropout=0.0, layers=None, dtype=torch.bfloat16, log=print, *, release_source_cache=False):
    """The model with every decoder projection, its table of word vectors and its output layer held ternary, and the
    cell (LoRA on the layers named, all if None). Every matrix is read from the file by itself and turned into codes
    at once, so the most memory used is the codes plus one matrix: an 8B DNA loads inside 9 GB."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    src = Path(src)
    config = AutoConfig.from_pretrained(src)
    with init_empty_weights():  # the shape of the model, with no memory behind any weight yet
        model = AutoModelForCausalLM.from_config(config)
    index = src / "model.safetensors.index.json"
    where = json.loads(index.read_text())["weight_map"] if index.exists() else None
    files = {}
    read_shards = set()

    def opened(f):
        if f not in files:
            files.clear()  # one file open at a time: an open file's pages count as this process's memory
            files[f] = safe_open(str(src / f), "pt")
            read_shards.add(src / f)
        return files[f]

    class MatrixRows:
        # Each tile owns its mapping. As _codes finishes using those rows the
        # kernel can drop them from this process's RSS without evicting file cache.
        def __init__(self, filename, name, shape):
            self.path = str(src / filename)
            self.name = name
            self.shape = shape

        def __getitem__(self, rows):
            with safe_open(self.path, "pt") as reader:
                return reader.get_slice(self.name)[rows]

    def stored(name, *, matrix=False):
        filename = where[name] if where else "model.safetensors"
        reader = opened(filename)
        if matrix and _packed_storage():
            return MatrixRows(filename, name, reader.get_slice(name).get_shape())
        return reader.get_tensor(name)

    def has(name):
        return name in where if where is not None else name in opened("model.safetensors").keys()

    nl = config.num_hidden_layers
    which = list(range(nl)) if layers is None else list(layers)
    worst = 0.0
    for i, block in enumerate(model.model.layers):
        for part in ("self_attn", "mlp"):
            parent = getattr(block, part)
            for name in TARGETS:
                lin = getattr(parent, name, None)
                if lin is None:
                    continue
                key = f"model.layers.{i}.{part}.{name}"
                t = TernaryLinear(stored(key + ".weight", matrix=True), rank if i in which else 0, alpha, dropout,
                                  bias=stored(key + ".bias") if lin.bias is not None and has(key + ".bias") else None)
                worst = max(worst, t.worst)
                setattr(parent, name, t)
        if i % 6 == 5:
            _trim()
    if worst > 0.02:
        raise ValueError(f"these weights are not ternary in groups of {GROUP} (the worst group is {worst:.3f} of its scale off)")
    # the table of word vectors and the output layer: ternary too where the model's are (one table when they are tied)
    emb = _codes(stored("model.embed_tokens.weight", matrix=True))
    tied = bool(getattr(config, "tie_word_embeddings", False)) or not has("lm_head.weight")
    as_codes = emb[2] <= 0.02
    if as_codes:
        model.model.embed_tokens = TernaryEmbedding(emb[0], emb[1], dtype)
    else:
        model.model.embed_tokens.weight = torch.nn.Parameter(stored("model.embed_tokens.weight").to(dtype), requires_grad=False)
    head = emb if tied else _codes(stored("lm_head.weight", matrix=True))
    if head[2] <= 0.02:
        model.lm_head = TernaryLinear(None, codes=head)
    elif tied and not as_codes:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        model.lm_head.weight = torch.nn.Parameter(stored("model.embed_tokens.weight" if tied else "lm_head.weight").to(dtype), requires_grad=False)
    del emb, head
    # what is left is small (the norms): as stored
    for name, p in list(model.named_parameters()):
        if p.is_meta:
            set_module_tensor_to_device(model, name, "cpu", value=stored(name).to(dtype))
    for p in model.parameters():
        p.requires_grad_(False)
    files.clear()
    _trim()
    if release_source_cache and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        # The optimizer uses the converted tensors. Hint only the read-only source shards this load consumed;
        # the kernel decides which clean, unpinned pages it can actually release. No system-wide cache flush.
        advised = 0
        for shard in sorted(read_shards):
            try:
                with shard.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                advised += 1
            except OSError:
                pass  # Unsupported filesystem or pages still in use: admission still measures what is available.
        log(f"released source-cache hint on {advised} of {len(read_shards)} converted model shards", flush=True)
    cell = Cell(model, which, rank, alpha, dropout)
    for p in cell.params:
        p.requires_grad_(True)
    storage = "four weights per byte" if _packed_storage() else "one byte per weight"
    log(f"the DNA held ternary: {storage} and one scale per {GROUP} (the word vectors and the output layer "
        f"{'too' if as_codes else 'as numbers'}); the worst group is {worst:.4f} of its scale off; {held_gb():.1f} GB held", flush=True)
    return model, cell


def memory_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def held_gb():
    """The memory this process holds of its own right now (not the pages of files it has open), in GB."""
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("RssAnon:"):
                return int(ln.split()[1]) / 2**20
    except OSError:
        pass
    return memory_gb()


if __name__ == "__main__":  # python training/ternary_base.py <unpacked folder>: the same logits held both ways
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(src)
    ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
    model, cell = load(src, rank=4, layers=[0])
    with torch.no_grad():
        b = model(input_ids=ids).logits.float()
    held = held_gb()
    del model, cell
    gc.collect()
    ref = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    with torch.no_grad():
        a = ref(input_ids=ids).logits.float()
    print(json.dumps({"max_logit_difference": round((a - b).abs().max().item(), 4), "same_next_word": bool((a[0, -1].argmax() == b[0, -1].argmax()).item()),
                      "next_word": tok.decode(b[0, -1].argmax()), "most_memory_gb_held_ternary": round(held, 2)}))
