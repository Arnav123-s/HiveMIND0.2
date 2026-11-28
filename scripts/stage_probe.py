"""Stage probe: time every step of a generation, at small scale, on THIS machine (a Kaggle or Colab session), then
estimate what a whole generation would take here. Generation wall-time has never been measured off GitHub Actions.

    python3 scripts/stage_probe.py [--stages env,store,install,practice,gate,train_cpu,train_gpu,upload]
                                   [--rows ROWS.jsonl] [--out runs/probe/probe-<host>.json] [--budget-hours 11]

Every stage runs as its own process with a timeout, so one that hangs or runs out of memory is recorded as such and
the others still run. Nothing here trains a cell for real or changes the body; it is a measurement.

    env        cores, RAM, disk, CPU flags, GPU (name, memory)
    store      fetch g81trunk and one code cell from the store (HIVE_STORE=hf), timed, MB/s
    install    build the ternary runtime (team.ternary install; about 25 min on 4 cores on Actions)
    practice   2 practice tasks through the body's trunk + one cell (training/grow.py practice)
    gate       2 tasks of a mixed exam sitting (training/exam.py sit), memory empty, as the gate sits them
    train_cpu  6 training rows of t80 on the CPU, DNA held ternary (--int8-base): seconds a row, peak memory
    train_gpu  6 rows of t80 on the GPU (bf16 full weights; needs about 16 GB+ of GPU memory: an OOM is a result too)
    upload     the probe file itself into the store (the checkpoint-out path), timed

The estimate uses g81's shape (docs/reads/g81.md): 96 practice tasks + 24 dream tasks, 8 children trained on
20 slices x 174 min on Actions CPUs at 9-18 s a row, and a gate of 8 seeds x 8 tasks x 2 arms + 28 held-out tasks
+ a 45 min talk exam. Change SHAPE below if the plan changes the generation's size.
"""
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

SHAPE = {
    "practice_tasks": 96, "dream_tasks": 24,
    "train_rows": 20 * 174 * 60 / 13.5,  # g81: 20 slices x 174 min at the middle of 9-18 s a row on Actions = ~15,500 rows
    "children": 8,
    "gate_tasks": 8 * 8 * 2 + 28, "talk_minutes": 45,
}
STAGES = ("env", "store", "install", "practice", "gate", "train_cpu", "train_gpu", "upload")


def run(cmd, timeout, env=None, cwd=ROOT):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=dict(os.environ, **(env or {})), capture_output=True, text=True, timeout=timeout)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timed out"
    return {"cmd": " ".join(map(str, cmd)), "exit": code, "seconds": round(time.time() - t0, 1),
            "tail": (out or "")[-600:], "err": (err or "")[-600:]}


def last_json(text):
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def env_info():
    info = {"host": socket.gethostname(), "python": platform.python_version(), "cores": os.cpu_count(),
            "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")), "colab": "COLAB_RELEASE_TAG" in os.environ}
    try:
        info["ram_gb"] = round(int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 2**20, 1)
        flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        info["avx512"] = any(f.startswith("avx512") for f in flags)
        info["avx512_vnni"] = "avx512_vnni" in flags
    except (OSError, StopIteration):
        pass
    info["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
    try:
        import torch
        if torch.cuda.is_available():
            info["gpus"] = [{"name": torch.cuda.get_device_name(i), "gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                            for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"not usable: {e}"
    return info


def estimate(r):
    """Hours a g81-shaped generation would take on this machine, stage by stage, run one after another."""
    est, why = {}, []
    t_prac = (r.get("practice") or {}).get("per_task_s")
    t_gate = (r.get("gate") or {}).get("per_task_s")
    cpu_row = (r.get("train_cpu") or {}).get("seconds_a_row")
    gpu_row = (r.get("train_gpu") or {}).get("seconds_a_row")
    if t_prac:
        est["practice_and_dream_h"] = round((SHAPE["practice_tasks"] + SHAPE["dream_tasks"]) * t_prac / 3600, 2)
    if t_gate:
        est["gate_h"] = round((SHAPE["gate_tasks"] * t_gate + SHAPE["talk_minutes"] * 60) / 3600, 2)
    if cpu_row:
        est["train_cpu_h"] = round(SHAPE["train_rows"] * cpu_row / 3600, 2)
    if gpu_row:
        est["train_gpu_h"] = round(SHAPE["train_rows"] * gpu_row / 3600, 2)
    setup = sum((r.get(s) or {}).get("seconds", 0) for s in ("store", "install")) / 3600
    est["setup_h"] = round(setup, 2)
    train = est.get("train_gpu_h") or est.get("train_cpu_h")
    parts = [est.get("practice_and_dream_h"), train, est.get("gate_h")]
    if all(p is not None for p in parts):
        est["generation_h_one_machine"] = round(setup + sum(parts), 2)
    else:
        why.append("not every stage measured: no whole-generation figure")
    est["shape"] = SHAPE
    est["notes"] = why
    return est


def main(argv):
    opts = dict(zip(argv[::2], argv[1::2]))
    stages = opts.get("--stages", ",".join(STAGES)).split(",")
    budget = float(opts.get("--budget-hours", 11))
    out = Path(opts.get("--out", ROOT / "runs" / "probe" / f"probe-{socket.gethostname()}.json"))
    rows = opts.get("--rows", "")
    dna = os.environ.get("DNA", "t80")
    members = f"{dna}+g81trunk,{dna}+g81code"
    r = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": stages}
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        r["estimate"] = estimate(r)
        out.write_text(json.dumps(r, indent=1, default=str))

    if "env" in stages:
        r["env"] = env_info(); save()
    if "store" in stages:
        r["store"] = run([PY, "training/grow_record.py", "fetch", members], 1800)
        sizes = sum(f.stat().st_size for f in (ROOT / "cells" / "members").glob(f"{dna}+g81*.gguf")) if (ROOT / "cells" / "members").exists() else 0
        r["store"]["mb"] = round(sizes / 1e6, 1)
        r["store"]["mb_s"] = round(sizes / 1e6 / r["store"]["seconds"], 2) if r["store"]["seconds"] else None
        save()
    if "install" in stages:
        r["install"] = run([PY, "-m", "team.ternary", "install"], 3600)
        r["install_models"] = run([PY, "-m", "team.ternary", "fetch"], 3600, env={"HIVE_MEMBERS": members})
        save()
    inference_env = {"HIVE_MEMBERS": members, "AGENT_TEAM_SOLO": "2", "AGENT_TEAM_SOLO_ALONE": "1", "HIVE_LIGHT": "1",
                     "AGENT_TEAM_ENGINE": "gguf", "HIVE_TASK_SECONDS": "600"}
    if "practice" in stages:
        p = run([PY, "training/grow.py", "practice", "--seed", "19990001", "--tasks", "2", "--level", "mixed",
                 "--out", "runs/probe/rows.jsonl"], 2400, env=inference_env)
        p["per_task_s"] = round(p["seconds"] / 2, 1) if p["exit"] == 0 else None
        r["practice"] = p; save()
    if "gate" in stages:
        g = run([PY, "training/exam.py", "sit", "--level", "mixed", "--tasks", "2", "--seed", "29990001", "--empty-memory"],
                2400, env=dict(inference_env, HIVE_TASK_SECONDS="900"))
        g["per_task_s"] = round(g["seconds"] / 2, 1) if g["exit"] == 0 else None
        r["gate"] = g; save()
    rows_file = rows or str(ROOT / "runs" / "probe" / "rows.jsonl")
    train = [PY, "training/grow.py", "train", "--rows", rows_file, "--name", "probe", "--base", dna, "--limit", "6", "--epochs", "1"]
    if "train_cpu" in stages:
        t = run(train + ["--int8-base"], 5400, env={"HIVE_GPU": "0"})
        t.update({k: (last_json(t["tail"]) or {}).get(k) for k in ("seconds_a_row", "memory_gb", "gpu")})
        r["train_cpu"] = t; save()
    if "train_gpu" in stages:
        if (r.get("env") or env_info()).get("gpus"):
            t = run(train, 5400, env={"HIVE_GPU": "1"})
            t.update({k: (last_json(t["tail"]) or {}).get(k) for k in ("seconds_a_row", "memory_gb", "gpu")})
            t["oom"] = "out of memory" in (t["err"] + t["tail"]).lower()
        else:
            t = {"skipped": "no GPU in this session"}
        r["train_gpu"] = t; save()
    for f in (ROOT / "cells" / "members").glob("*+probe.*") if (ROOT / "cells" / "members").exists() else []:
        f.unlink()
    save()
    if "upload" in stages and os.environ.get("HIVE_STORE") == "hf":
        r["upload"] = run([PY, "training/hf_store.py", "put", str(out)], 900); save()
    e = r["estimate"]
    print(json.dumps(e, indent=1))
    g = e.get("generation_h_one_machine")
    print(f"a g81-shaped generation on this machine, one stage after another: {g} h (budget {budget} h): "
          + ("FITS" if g is not None and g <= budget else "DOES NOT FIT" if g is not None else "NOT MEASURED"))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
