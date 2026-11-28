"""A cell's weight edit rebuilt from its GGUF adapter: the reverse of convert_lora_to_gguf.py, for cells whose
.safetensors was lost.

    python3 training/gguf_to_peft.py CELL.gguf OUT.safetensors [--compare REAL.safetensors]

Why: g39code and g44words2 are cells of the g81 body, but only their inference .gguf survived the recovery (no
.safetensors). training/merge.py trunk needs a .safetensors per leaf, and skips a missing one with only a warning,
so g82's trunk would quietly lose two of its 18 leaves (docs/RESUME-G81.md, F5).

What it can and cannot give back:
  - The GGUF holds the very same matrices (lora_a, lora_b), unscaled, with lora_alpha in its metadata: the converter
    copies them; for the Qwen3 family it permutes nothing. grow.py writes cells with --outtype f16, so the values
    come back as their float16 roundings: about 3 significant digits, a relative error near 5e-4. The .safetensors
    held float32 (ternary_base.Cell.save). The edit is the same edit to that precision; it is NOT bit-identical.
  - Written in the layout the hive saves (ternary_base.Cell.save): base_model.model.<module>.lora_A.weight / lora_B
    in float32, with an adapter_config.json beside it (r, lora_alpha, the layers, the target modules).

--compare: measures the rebuild against a real .safetensors of the same cell (the check to run on a cell that has
both, e.g. t80.g81code, before trusting the rebuild of one that does not).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GGUF_PY = ROOT / "training" / "llama_cpp_convert" / "gguf-py"

# GGUF tensor name (Qwen2/Qwen3 family, the hive's DNAs) -> the module's HF name
MODULES = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
           "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
NAME = re.compile(r"blk\.(\d+)\.(\w+?)\.weight\.lora_([ab])$")


def _reader(path):
    sys.path.insert(0, str(GGUF_PY))
    from gguf import GGUFReader
    return GGUFReader(str(path))


def read(path):
    """({hf key: float32 array}, metadata) of a GGUF LoRA adapter."""
    r = _reader(path)
    meta = {}
    for f in r.fields.values():
        if f.name in ("adapter.type", "adapter.lora.alpha", "general.architecture", "general.type"):
            v = f.parts[f.data[0]]
            meta[f.name] = bytes(v).decode() if v.dtype == np.uint8 else float(v[0])
    if meta.get("adapter.type") != "lora":
        raise SystemExit(f"{path}: not a LoRA adapter (adapter.type = {meta.get('adapter.type')!r})")
    out = {}
    for t in r.tensors:
        m = NAME.fullmatch(t.name)
        if not m or m.group(2) not in MODULES:
            raise SystemExit(f"{path}: a tensor this tool does not know how to name: {t.name}")
        layer, mod, ab = int(m.group(1)), MODULES[m.group(2)], m.group(3).upper()
        # GGUFReader gives the data in numpy order (rows, columns), the shape torch had: lora_A (r, in), lora_B (out, r)
        out[f"base_model.model.model.layers.{layer}.{mod}.lora_{ab}.weight"] = np.asarray(t.data, dtype=np.float32).reshape(
            [int(x) for x in reversed(t.shape)])
    return out, meta


def config(tensors, meta):
    """The adapter_config.json of the rebuilt edit, from what the tensors and the metadata say."""
    ranks = {a.shape[0] for k, a in tensors.items() if k.endswith(".lora_A.weight")}
    if len(ranks) != 1:
        raise SystemExit(f"the edit's matrices disagree on the rank: {sorted(ranks)}")
    layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in tensors})
    mods = sorted({k.split(".")[-3] for k in tensors})
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": ranks.pop(), "lora_alpha": meta.get("adapter.lora.alpha"),
            "lora_dropout": 0.0, "target_modules": mods, "layers_to_transform": layers, "bias": "none",
            "rebuilt_from_gguf": True}


def compare(rebuilt, real_path):
    """How far the rebuild is from a real .safetensors of the same cell: per matrix and on the edit B·A itself."""
    from safetensors.numpy import load_file
    real = load_file(str(real_path))
    missing, extra = sorted(set(real) - set(rebuilt)), sorted(set(rebuilt) - set(real))
    worst, worst_ba = 0.0, 0.0
    for k in set(real) & set(rebuilt):
        a, b = real[k].astype(np.float32), rebuilt[k]
        if a.shape != b.shape:
            return {"ok": False, "why": f"{k}: shape {b.shape} against {a.shape}"}
        worst = max(worst, float(np.abs(a - b).max() / (np.abs(a).max() or 1)))
    for k in (k for k in real if k.endswith(".lora_A.weight")):
        kb = k.replace("lora_A", "lora_B")
        if k in rebuilt and kb in rebuilt and kb in real:
            d0 = real[kb].astype(np.float32) @ real[k].astype(np.float32)
            d1 = rebuilt[kb] @ rebuilt[k]
            worst_ba = max(worst_ba, float(np.linalg.norm(d0 - d1) / (np.linalg.norm(d0) or 1)))
    ok = not missing and not extra and worst < 2e-3 and worst_ba < 2e-3
    return {"ok": ok, "missing": missing[:5], "extra": extra[:5], "max_rel_error_matrix": worst, "max_rel_error_edit": worst_ba}


def main(argv):
    if len(argv) < 2 or argv[0].startswith("-"):
        print(__doc__)
        return 2
    tensors, meta = read(argv[0])
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(tensors, str(out))
    cfg = config(tensors, meta)
    (out.parent / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"{out}: {len(tensors)} matrices, rank {cfg['r']}, alpha {cfg['lora_alpha']}, layers {cfg['layers_to_transform'][0]}-{cfg['layers_to_transform'][-1]}")
    if "--compare" in argv:
        rep = compare(tensors, argv[argv.index("--compare") + 1])
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
