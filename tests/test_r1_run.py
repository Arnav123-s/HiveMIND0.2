"""scripts/r1_run.py end to end, offline: a fake recovery folder (tar archives compressed with zstd and split into parts
the way the real ones are, a hard-linked member as the archive's convenience paths are), synthetic cells made by the
real converter, a fake store. The real run differs only in the folder, the archives and the store."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

zstandard = pytest.importorskip("zstandard")

FAKE_RESTORE = '''import argparse, hashlib, json
from pathlib import Path
c = argparse.ArgumentParser(); c.add_argument("--manifest"); c.add_argument("--parts"); c.add_argument("--output"); c.add_argument("--only")
o = c.parse_args(); m = json.loads(Path(o.manifest).read_text())
for a in m["archives"]:
    if a["archive"].startswith(o.only):
        Path(o.output).mkdir(parents=True, exist_ok=True)
        data = b"".join((Path(o.parts) / p["name"]).read_bytes() for p in a["parts"])
        assert hashlib.sha256(data).hexdigest() == a["sha256"]
        (Path(o.output) / a["archive"]).write_bytes(data)
'''


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _cell(tmp, name, seed):
    """(gguf bytes, safetensors bytes) of a synthetic Qwen3-shaped cell in the hive's layout."""
    from safetensors.numpy import save_file
    base, cell = tmp / f"base-{name}", tmp / f"cell-{name}"
    base.mkdir(); cell.mkdir()
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
        "intermediate_size": 192, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 256, "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False}))
    rng, r = np.random.default_rng(seed), 16
    shapes = {"self_attn.q_proj": (64, 64), "self_attn.k_proj": (32, 64), "self_attn.v_proj": (32, 64), "self_attn.o_proj": (64, 64),
              "mlp.gate_proj": (192, 64), "mlp.up_proj": (192, 64), "mlp.down_proj": (64, 192)}
    t = {}
    for m, (o, i) in shapes.items():
        t[f"base_model.model.model.layers.3.{m}.lora_A.weight"] = (rng.standard_normal((r, i)) * 0.02).astype(np.float32)
        t[f"base_model.model.model.layers.3.{m}.lora_B.weight"] = (rng.standard_normal((o, r)) * 0.01).astype(np.float32)
    save_file(t, str(cell / "adapter_model.safetensors"))
    (cell / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": 2 * r,
        "target_modules": [m.split(".")[1] for m in shapes], "layers_to_transform": [3], "base_model_name_or_path": str(base)}))
    conv = ROOT / "training" / "llama_cpp_convert"
    out = tmp / f"{name}.gguf"
    done = subprocess.run([sys.executable, str(conv / "convert_lora_to_gguf.py"), "--base", str(base), "--outfile", str(out), "--outtype", "f16", str(cell)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{conv / 'gguf-py'}:{conv}"})
    assert done.returncode == 0, done.stderr[-600:]
    return out.read_bytes(), (cell / "adapter_model.safetensors").read_bytes()


def _archive(drive, name, members, links=()):
    """A .tar.zst of MEMBERS {path: bytes} (+ hard LINKS (path, target)), split into 2 parts: its manifest entry."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for p, b in members.items():
            ti = tarfile.TarInfo(p); ti.size = len(b); t.addfile(ti, io.BytesIO(b))
        for p, target in links:
            ti = tarfile.TarInfo(p); ti.type = tarfile.LNKTYPE; ti.linkname = target; t.addfile(ti)
    data = zstandard.ZstdCompressor().compress(buf.getvalue())
    half = len(data) // 2
    parts = []
    for k, chunk in enumerate((data[:half], data[half:]), 1):
        pn = f"{name}.part{k:03}"
        (drive / pn).write_bytes(chunk)
        parts.append({"name": pn, "bytes": len(chunk), "sha256": _sha(chunk)})
    return {"archive": name, "bytes": len(data), "sha256": _sha(data), "parts": parts}


def _rows():
    head = [{"role": "system", "content": "orders"}, {"role": "user", "content": "Find the length of the river Platte River"}]
    dump = '{"entities": {"Q655754": {"claims": {"P2043": [{"mainsnak": {"datavalue": {"value": {"amount": "+499"}}}}]}}}}'
    base = {"task": "Find the length of the river Platte River", "kind": "a fact found on the open web", "seed": "h10003306_01", "member": 1, "gen": "g34"}
    rows = [base | {"type": "first", "messages": head, "output": "p"},
            base | {"type": "done", "messages": head + [{"role": "user", "content": f"OUTPUT:\n{dump}\n\nMore?"}], "output": "DONE"},
            {"task": "walk", "kind": "walk a site over several pages", "seed": "s", "member": 1, "gen": "g38", "type": "first", "messages": head, "output": "q"}]
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _drive(tmp, break_validation=False):
    drive = tmp / "drive"; drive.mkdir()
    cells = {n: _cell(tmp, n, s) for n, s in (("g81code", 1), ("g80words", 2), ("g39code", 3), ("g44words2", 4))}
    if break_validation:  # the .safetensors of g81code is another cell's: the rebuild cannot match it
        cells["g81code"] = (cells["g81code"][0], cells["g80words"][1])
    a1 = {"models/g81/t80.g81code.gguf": cells["g81code"][0], "models/g81/t80.g81code.safetensors": cells["g81code"][1],
          "workspace/inputs/t80.g80words.gguf": cells["g80words"][0], "models/g80/t80.g80words.safetensors": cells["g80words"][1]}
    a2 = {"models/g39/t80+g39code.gguf": cells["g39code"][0], "models/g44/t80+g44words2.gguf": cells["g44words2"][0], "datasets/g80/rows.jsonl": _rows()}
    m1 = _archive(drive, "03-a.tar.zst", a1, links=[("models/g80/t80.g80words.gguf", "workspace/inputs/t80.g80words.gguf")])
    m2 = _archive(drive, "05-b.tar.zst", a2)
    (drive / "RECOVERY-MANIFEST.json").write_text(json.dumps({"archives": [m1, m2]}))
    (drive / "RESTORE.py").write_text(FAKE_RESTORE)
    files = {"03-a.tar.zst": [("models/g81/t80.g81code.gguf", cells["g81code"][0]), ("models/g81/t80.g81code.safetensors", cells["g81code"][1]),
                              ("models/g80/t80.g80words.gguf", cells["g80words"][0]), ("models/g80/t80.g80words.safetensors", cells["g80words"][1])],
             "05-b.tar.zst": [("models/g39/t80+g39code.gguf", cells["g39code"][0]), ("models/g44/t80+g44words2.gguf", cells["g44words2"][0]),
                              ("datasets/g80/rows.jsonl", _rows())]}
    spec = {"recovery_manifest_sha256": _sha((drive / "RECOVERY-MANIFEST.json").read_bytes()), "restore_py_sha256": _sha(FAKE_RESTORE.encode()),
            "archives": {a: {"bytes": m["bytes"], "sha256": m["sha256"], "parts": 2,
                             "files": [{"path": p, "sha256": _sha(b), "store_name": "rows.jsonl" if p.endswith("rows.jsonl") else p.rsplit("/", 1)[-1].replace("t80+", "t80.")}
                                       for p, b in files[a]]} for a, m in (("03-a.tar.zst", m1), ("05-b.tar.zst", m2))}}
    (tmp / "spec.json").write_text(json.dumps(spec))
    return drive, tmp / "spec.json"


@pytest.fixture
def store(monkeypatch, tmp_path):
    from training import hf_store, quarantine
    held = {}
    monkeypatch.setattr(hf_store, "put", lambda files, message=None: held.update({hf_store.path_in_store(Path(f).name): Path(f).read_bytes() for f in files}) or 0)

    def get(name, out):
        k = hf_store.path_in_store(name)
        if k not in held:
            return False
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(held[k])
        return True
    monkeypatch.setattr(hf_store, "get", get)
    monkeypatch.setattr(hf_store, "names", lambda prefix="": sorted(held))
    monkeypatch.setattr(quarantine, "FILE", tmp_path / "QUARANTINE.jsonl")
    return held


def test_r1_runs_end_to_end_and_the_gate_lets_the_rebuilds_through(tmp_path, store):
    import r1_run
    drive, spec = _drive(tmp_path)
    run = r1_run.Run(drive, tmp_path / "work", tmp_path / "out", files=spec, log=lambda *a: None)
    for s in r1_run.STEPS:
        getattr(run, s)()
    st = json.loads((tmp_path / "out" / "r1-report.json").read_text())
    assert st["extract"]["files"] == 7  # the hard-linked g80words.gguf included
    assert st["validate"]["gate"] == "PASS" and st["validate"]["rebuilt"] == ["g39code", "g44words2"]
    assert all(r["ok"] for r in st["validate"]["reports"].values())
    assert "g39/t80.g39code.safetensors" in store and "g44/t80.g44words2.safetensors" in store and "shared/rows.jsonl" in store
    assert st["upload"]["all_listed"] and all(g["verified"] for g in st["upload"]["groups"].values())
    a = st["audit"]
    assert a["flagged_rows"] == 2 and a["quarantine_keys_added"] == 2 and a["walk_a_site_g38_rows"] == 1
    assert (tmp_path / "out" / "QUARANTINE.jsonl").read_text().count("sha256") == 2
    assert (tmp_path / "out" / "t80.g39code.adapter_config.json").exists()
    # run again: nothing is done twice, nothing is added twice
    run2 = r1_run.Run(drive, tmp_path / "work", tmp_path / "out", files=spec, log=lambda *a: None)
    run2.audit()
    assert json.loads((tmp_path / "work" / "r1-state.json").read_text())["audit"]["quarantine_keys_added"] == 0


def test_a_failed_validation_blocks_both_rebuilds(tmp_path, store):
    import r1_run
    drive, spec = _drive(tmp_path, break_validation=True)
    run = r1_run.Run(drive, tmp_path / "work", tmp_path / "out", files=spec, log=lambda *a: None)
    for s in ("restore", "extract", "validate"):
        getattr(run, s)()
    v = run.state["validate"]
    assert v["gate"] == "FAIL" and v["rebuilt"] == [] and v["g82_trunk"].startswith("FALLBACK: 16 leaves")
    assert not (tmp_path / "work" / "stage" / "t80.g39code.safetensors").exists()
    assert not (tmp_path / "work" / "stage" / "t80.g44words2.safetensors").exists()


def test_a_tampered_part_or_tool_stops_the_run(tmp_path, store):
    import r1_run
    drive, spec = _drive(tmp_path)
    (drive / "RESTORE.py").write_text(FAKE_RESTORE + "\n# changed\n")
    with pytest.raises(SystemExit, match="not the file the receipt recorded"):
        r1_run.Run(drive, tmp_path / "work", tmp_path / "out", files=spec, log=lambda *a: None).restore()


def test_the_real_file_list_is_whole():
    spec = json.loads((ROOT / "manifests" / "r1-files.json").read_text())
    files = [f for a in spec["archives"].values() for f in a["files"]]
    names = {f["store_name"] for f in files}
    assert len(spec["archives"]) == 13 and len(files) == 84 == len(names)
    for g in ("g73", "g80", "g81"):
        assert {f"t80.{g}trunk.{e}" for e in ("gguf", "json", "safetensors")} <= names
    assert {"t80.g81code.gguf", "t80.g81code.safetensors", "t80.g80words.gguf", "t80.g80words.safetensors",
            "t80.g39code.gguf", "t80.g44words2.gguf", "rows.jsonl"} <= names
    assert all(len(f["sha256"]) == 64 for f in files)
