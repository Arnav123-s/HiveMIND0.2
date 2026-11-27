"""The hive is one body: one ternary model (its DNA) and many cells, each the DNA with a small weight edit of its own.

    practice  ->  rows  ->  every cell regrown from the rows  ->  gate  ->  the body's next generation

practice   Fresh tasks that are never an exam's (seeds from ten million up; the exam's held-out kinds are never
           practised). The body's own cells do them as they do any task (team/solo.py). Each cell's own answer is
           put to the task's own check, computed by code (training/exam.each_member).
rows       From each cell whose own answer the check confirms, agreed with or not: the task -> its final program, the
           program's output -> DONE, and, where it first failed, the failure -> the mended program. Nobody labelled
           anything: the label is that the program ran and its answer was checked.
train      Every cell is grown again from the same DNA (Ternary-Bonsai-1.7B, 0.43 GB packed, the smallest ternary
           model whose full weights are published) on all the rows, as a low-rank weight edit (LoRA) on a CPU. A
           cell leans to one organ (team/ternary.ORGANS): it meets that organ's rows three times as often. So the
           cells are the same size, overlap, and each is strongest somewhere. A cell is a file of a few megabytes
           (cells/members/<dna>+<name>.gguf); the DNA is in memory once for all of them.
gate       The same fresh exam sat by the DNA as published, by the body as it was and by the body as regrown; the
           body changes only when the numbers say so (.github/workflows/grow.yml, training/grow_record.py).

    python training/grow.py practice --seed 10000001 --tasks 16 [--level mixed|hard|basic] [--out runs/grow/rows.jsonl]
    python training/grow.py rows RUN_FOLDER ... [--out rows.jsonl]      rows from sittings already done (checked tasks only)
    python training/grow.py train --rows rows.jsonl --name g3code --lean code [--epochs 2] [--minutes 240] [--ternary] [--foresee]
    python training/grow.py fetch                                        the cells of the body, from the release
"""
import contextlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GROW = ROOT / "runs" / "grow"
UNPACKED = {"t17": "prism-ml/Ternary-Bonsai-1.7B-unpacked", "t40": "prism-ml/Ternary-Bonsai-4B-unpacked", "t80": "prism-ml/Ternary-Bonsai-8B-unpacked"}  # members whose full weights are published, so they can be taught
READ_LEN = int(os.environ.get("HIVE_READ_LEN", 256))  # the most pieces of one raw text read in training (train --read)
RELEASE = "hive-grown"  # the GitHub release that holds the rows and the weight edits (never in git: they are big and machine-made)


# ---- rows ------------------------------------------------------------------------------------------------------------
def _home(run):
    """The folder of a task's members' own files (member{n}.json, members.json): run/members, or run itself."""
    return Path(run) / "members" if (Path(run) / "members").is_dir() else Path(run)


def _number(f):
    """The member's number in its file's name: member3.json -> 3."""
    return int(re.sub(r"\D", "", Path(f).stem))


def checked(run, home=None):
    """{member number: ok}: each member's own answer put to the task's own check (exam.json "members", written by
    training/exam.each_member), tied to its member{n}.json by number. None when the task has no such verdicts.
    A verdict written before each was tied to its file (generation 28 and earlier: {"who", "ok"} only) is tied by
    the member's name, and only where exactly one file and one verdict carry that name: never by order, since the
    order the verdicts were written in is not the order of the numbers (member10.json sorts before member2.json)."""
    home = Path(home) if home else _home(run)
    try:
        verdicts = json.loads((Path(run) / "exam.json").read_text()).get("members")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(verdicts, list):
        return None
    files = {}
    for f in home.glob("member[0-9]*.json"):
        try:
            t = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        files[_number(f)] = str(t.get("who") or "")[:80] if isinstance(t, dict) else ""
    named = [v.get("who") for v in verdicts if isinstance(v, dict) and not isinstance(v.get("member"), int)]
    out = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        n = v.get("member")
        if not isinstance(n, int) or isinstance(n, bool):
            who = v.get("who")
            mine = [k for k, w in files.items() if who and w == who]
            n = mine[0] if len(mine) == 1 and named.count(who) == 1 else None
        if n is not None:
            out[n] = out.get(n, False) or v.get("ok") is True
    return out


def taught(run, home=None):
    """The members whose programs become rows. Every member whose own answer the task's check confirms wrote a right
    program, whether or not another agreed with it: agreement is not the check. Generation 28's practice (80 of its
    100 tasks in hand): 3 failed tasks had a member the check confirmed (c10002802_02's first member was right while
    the other two disagreed), 9 right members of passed tasks were left out because two others had agreed, and one
    member taught for agreeing was refused by the check. Without per-member verdicts (a run made before exam.json
    kept them, or a task not sat as an exam: team/engram), the old rule: the members that agreed, on a task the
    check passed."""
    home = Path(home) if home else _home(run)
    said = checked(run, home)
    if said is not None:
        return sorted(n for n, ok in said.items() if ok)
    exam = Path(run) / "exam.json"
    try:
        passed = exam.exists() and json.loads(exam.read_text()).get("ok") is True
    except (OSError, ValueError, AttributeError):
        passed = False
    if exam.exists() and not passed:
        return []
    try:
        return [n for n in json.loads((home / "members.json").read_text()).get("agreeing", []) if isinstance(n, int)]
    except (OSError, ValueError, AttributeError):
        return []


def once(rows):
    """Each row once: two members that wrote the same program for the same task make the same row, and the same row
    twice is the same lesson weighed double."""
    seen, out = set(), []
    for r in rows:
        key = json.dumps([r.get("type"), r.get("messages"), r.get("output"), r.get("rejected")], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def saved_key(name):
    """The key in a saved edit (adapter_model.safetensors) of a trainable parameter of the model being trained. A
    ternary edit's matrices are parametrized (train(ternary_edit=True)), so their parameter is
    "...lora_A.default.parametrizations.weight.original", saved as "...lora_A.weight". Without this, an edit trained
    as ternary could not be started from again: the hive engram's second update found none of its matrices."""
    return name.replace(".default", "").replace(".parametrizations.weight.original", ".weight")


def choose_donor(organ, body_kinds, active_cells):
    """Pick an active cross-organ cell using the last two generation records, oldest first.

    Both paired before/after exams must show no gain and room to improve on this organ's
    own tasks. Prefer a donor organ with successes missing from the recipient's kind profile.
    """
    from training import lineage
    if organ not in lineage.ORGANS or not isinstance(body_kinds, (list, tuple)) or len(body_kinds) < 2:
        return None
    previous, latest = body_kinds[-2:]
    numbers = [re.fullmatch(r"g(\d+)", rec.get("generation", "")) for rec in (previous, latest)]
    if not all(numbers) or int(numbers[1][1]) != int(numbers[0][1]) + 1:
        return None
    if previous.get("dna") != latest.get("dna"):
        return None
    for rec in (previous, latest):
        arms = rec.get("arms", {})
        counts = [arms.get(arm, {}).get("by_lean", {}).get(organ, {}).get("organ", {}).get(organ)
                  for arm in ("before", "after")]
        if rec.get("trial") or any(not isinstance(c, (list, tuple)) or len(c) != 2 for c in counts):
            return None
        (b, bn), (a, an) = counts
        if not (0 <= b <= bn and bn > 0 and 0 <= a < an) or a * bn > b * an:
            return None

    by_lean = latest["arms"]["after"]["by_lean"]
    target = {k: p / n for k, (p, n) in by_lean[organ].get("kind", {}).items() if n > 0}
    best, winner = None, None
    for cell in sorted(active_cells):
        other = lineage.organ_of(cell)
        if other == organ or not other or lineage.born(cell) > int(numbers[1][1]):
            continue
        kinds = by_lean.get(other, {}).get("kind", {})
        advantage = sum(max(0, p / n - target.get(k, 0)) for k, (p, n) in kinds.items() if n > 0)
        score = (advantage, lineage.born(cell))
        if advantage > 0 and (best is None or score > best):
            best, winner = score, cell
    return winner


def _birth_donor(source, organ, base, gen):
    """Match local release-named adapter files to active cells and the recorded exams for this DNA."""
    from training import lineage
    source = Path(source)
    body_file, history = ROOT / "cells/body.json", ROOT / "training/reports/generations.jsonl"
    if not source.exists() or not body_file.exists() or not history.exists():
        return None
    body = json.loads(body_file.read_text())
    if body.get("dna") != base:
        return None
    files = source.glob("*.safetensors") if source.is_dir() else [source]
    active = {}
    for path in files:
        match = re.fullmatch(rf"(?:{re.escape(base)}[.+])?({lineage.CELL})", path.stem)
        if match and match[1] in body.get("cells", []):
            active[match[1]] = path
    if not active:
        return None
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("dna", "t17") == base]
    if gen and (not rows or lineage.born(gen) != lineage.born(rows[-1].get("generation", "")) + 1):
        return None
    return active.get(choose_donor(organ, rows, active))


def _sibling_factors(path, model, cell):
    """Map just the common B factors from a full adapter or B-only safetensors snapshot."""
    from safetensors import safe_open
    if cell is not None:
        current = ((f"base_model.model.{n}.lora_B.weight", layer.lora_B) for n, layer in cell.layers.items())
    else:
        current = ((saved_key(n), p) for n, p in model.named_parameters() if p.requires_grad)
    pairs = []
    with safe_open(str(path), framework="pt") as archive:
        keys = set(archive.keys())
        for key, b in current:
            if key.endswith(".lora_B.weight") and key in keys:
                frozen = archive.get_tensor(key)
                if not frozen.is_floating_point() or frozen.ndim != 2 or frozen.size(0) != b.size(0):
                    raise ValueError(f"incompatible sibling B factor {key}")
                pairs.append((b, frozen.to(device=b.device, dtype=b.dtype)))
    if not pairs:
        raise ValueError(f"no matching trainable LoRA B factors in {path}")
    return pairs


def _add_sibling_gradient(pairs, weight):
    """Add the exact gradient of weight * sum(||B.T @ frozen||_F²) before global clipping."""
    import torch
    with torch.no_grad():
        for b, frozen in pairs:
            if b.grad is None:
                b.grad = torch.zeros_like(b)
            b.grad.addmm_(frozen, frozen.T @ b, alpha=2 * weight)


def rows_of(run, task, kind="", seed=None):
    """The training rows of one task: from every member whose own answer the task's check confirmed (taught), whether
    or not the whole task was right. run: the task's folder."""
    from team import solo
    home = _home(run)
    try:
        agreeing = json.loads((home / "members.json").read_text()).get("agreeing", [])
    except (OSError, ValueError, AttributeError):
        agreeing = []
    out = []
    for n in taught(run, home):
        try:
            t = json.loads((home / f"member{n}.json").read_text())
        except (OSError, ValueError):
            continue
        ms, program = t.get("messages") or [], (t.get("program") or "").strip()
        if len(ms) < 3 or not program or len(program) > 6000:
            continue
        # the taught member's short orders (team/solo.SHORT) with the day the work was done, and the task
        day = re.search(r"Today is (\d{4}-\d\d-\d\d \(\w+\))", ms[0]["content"])
        head = [{"role": "system", "content": solo.SHORT.format(today=day.group(1) if day else time.strftime("%Y-%m-%d (%A)"))}, ms[1]]
        fenced = f"```python\n{program}\n```"
        # agreed: whether this member was one of those that agreed (False: a right member the old rule left out)
        base = {"task": task, "kind": kind, "seed": seed, "who": t.get("who", ""), "member": n, "agreed": n in agreeing}
        out.append(base | {"type": "first", "messages": head, "output": fenced})
        out.append(base | {"type": "done", "messages": head + [{"role": "assistant", "content": fenced},
                                                                 {"role": "user", "content": f"OUTPUT:\n{t['answer'][-1800:]}\n\n{solo.MORE}"}], "output": "DONE"})
        # where it failed first: the failing program and what it was told, then the program that was right
        for i in range(2, len(ms) - 2, 2):
            a, fb, nxt = ms[i], ms[i + 1], ms[i + 2]
            if a["role"] == "assistant" and nxt["role"] == "assistant" and solo.extract_code(nxt["content"]).strip() == program \
                    and solo.extract_code(a["content"]).strip() != program and len(a["content"]) < 5000 and len(fb["content"]) < 3000:
                out.append(base | {"type": "fix", "messages": head + [a, fb], "output": fenced})
                break
    return once(out)


def part_kinds(check):
    """{letter: kind} of the parts of a compound task's check (check.parts: maker, seed, level), {} when unknown."""
    out = {}
    for i, p in enumerate(getattr(check, "parts", None) or []):
        try:
            from training import exam as E
            made = E.remake(*p[:3])
            if made:
                out["ABCDEFGH"[i]] = made[1]
        except Exception:  # noqa: BLE001 - a kind not known is no kind
            pass
    return out


def rows_by_part(run, task, kind="", seed=None, pairs=True, kinds=None):
    """The rows of a task of several parts worked part by part (team/by_part.py): each part's own run, in
    <run>/members/part_X, is a single-kind task (its own TASK text and kind, its members' messages, its own agreement).
    A part the task's check said was not right (exam.json 'why': 'PART B not right') or that no two members agreed
    on gives no row. [] when the run was not worked part by part (rows_of applies as before)."""
    home = _home(run)
    out = []
    try:
        why = json.loads((Path(run) / "exam.json").read_text()).get("why") or ""
    except (OSError, ValueError, AttributeError):
        why = ""
    for d in sorted(home.glob("part_[A-H]")):
        letter = d.name[-1]
        if not d.is_dir() or not (d / "TASK.md").exists() or f"PART {letter} not right" in why:
            continue
        ptask = (d / "TASK.md").read_text().strip()
        k = (kinds or {}).get(letter) or kind
        out += rows_of(d, ptask, k, seed) + (pair_rows(d, ptask, k, seed) if pairs else [])
    return out


def has_parts(run):
    return any(_home(run).glob("part_[A-H]"))


def pair_rows(run, task, kind="", seed=None, most=4):
    """Rows that set a right program beside a wrong one written for the same task. The right one is that of a member
    whose own answer the check confirmed (taught); the wrong ones are the first programs of the other members that
    are none of the right programs (sent back with an error, or a different result). A member the check confirmed is
    never the rejected one: what the check passed is not taught as wrong. Trained with --pairs (Iterative Reasoning
    Preference Optimization: DPO with a likelihood term, Pang et al. 2024), the cell learns what makes a program
    wrong as well as what a right one looks like; training only on right answers tends to memorise them (Chu et al.
    2025, "SFT memorizes, RL generalizes"). Used only with --pairs, until a gate has shown it helps."""
    from team import solo
    home = _home(run)
    tries = {}
    for f in sorted(home.glob("member[0-9]*.json")):
        try:
            tries[_number(f)] = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
    right = [n for n in taught(run, home) if n in tries]
    win = next((tries[n] for n in right if (tries[n].get("program") or "").strip()), None)
    if not win or len(win.get("messages") or []) < 3 or len(win["program"]) > 6000:
        return []
    ms, program = win["messages"], win["program"].strip()
    day = re.search(r"Today is (\d{4}-\d\d-\d\d \(\w+\))", ms[0]["content"])
    head = [{"role": "system", "content": solo.SHORT.format(today=day.group(1) if day else time.strftime("%Y-%m-%d (%A)"))}, ms[1]]
    # every program a right member ended with is right: none of them is set down as the wrong one
    out, seen = [], {(tries[n].get("program") or "").strip() for n in right}
    # with the check's own verdict on each member, only members it did not confirm give wrong programs; without it (a
    # run from before the verdicts were kept), as before: any first program that was not the agreed one
    confirmed = set(right) if checked(run, home) is not None else set()
    for n, t in sorted(tries.items()):
        if n in confirmed:
            continue
        tm = t.get("messages") or []
        if len(tm) < 3 or tm[2].get("role") != "assistant":
            continue
        code = solo.extract_code(tm[2]["content"]).strip()
        if not code or code in seen or len(code) > 6000:
            continue
        seen.add(code)
        # whose wrong program it was, and what that member was shown when it ran (its output or its error): a child
        # of that cell learns the correction from exactly there (train: "mend")
        shown = tm[3]["content"].rsplit("\n\n", 1)[0][-900:] if len(tm) > 3 and tm[3].get("role") == "user" else ""
        out.append({"task": task, "kind": kind, "seed": seed, "type": "pair", "member": n, "messages": head,
                    "output": f"```python\n{program}\n```", "rejected": f"```python\n{code}\n```",
                    "rejected_who": t.get("who", ""), "shown": shown})
        if len(out) >= most:
            break
    return out


MEND = "{shown}\n\nThe task's check says that answer is not right. Reply with the whole corrected program."
FORESEE = ("Here is a program that was written for the TASK. Do not write a program: say exactly what this one prints "
           "when it is run (or the last line of its error if it fails).\n```python\n{program}\n```")


def foresee_rows(run, task, kind="", seed=None, most=4):
    """Rows of another kind, from every program any cell ran on a task, right or wrong: the program -> what it
    printed. A cell that can say what a program will print before it runs knows how programs behave; the hive has
    many times more of these rows than of solved tasks, and running the program was the only label needed.
    (The hive's own form of learning by predicting. Used in training only with --foresee, until a gate has shown
    that cells grown with them do better than cells grown without.)"""
    from team import solo
    home = Path(run) / "members" if (Path(run) / "members").exists() else Path(run)
    out, seen = [], set()
    for prog in sorted(home.rglob("member*_*.py")):
        code = prog.read_text(errors="ignore").strip()
        printed = (prog.parent / f".{prog.name}.out")
        err = (prog.parent / f".{prog.name}.err")
        said = printed.read_text(errors="ignore").strip() if printed.exists() else ""
        if not said and err.exists() and err.read_text(errors="ignore").strip():
            said = err.read_text(errors="ignore").strip().splitlines()[-1]
        if not code or not said or len(code) > 3000 or len(said) > 500 or code in seen or re.search(r"look\(|requests|urllib|http|datetime\.(?:now|today)|date\.today|random", code):  # only what the program alone decides
            continue
        seen.add(code)
        out.append({"task": task, "kind": kind, "seed": seed, "type": "foresee", "output": said,
                    "messages": [{"role": "system", "content": solo.SHORT.format(today=time.strftime("%Y-%m-%d (%A)"))},
                                 {"role": "user", "content": f"TASK: {task}\n\n" + FORESEE.format(program=code)}]})
        if len(out) >= most:
            break
    return out


def _dna():
    """The DNA of the body on this machine (cells/body.json), t80 when it says none."""
    from team import ternary
    try:
        return json.loads(ternary.BODY.read_text()).get("dna") or "t80"
    except (OSError, ValueError, AttributeError):
        return "t80"


def edge_weights(dna="t17", last=3, record=None, body=None):
    """{kind: weight} for practice at the edge: the more a kind sits near half passed, the more it is practised.
    A kind always passed is learned; one never passed gives no right program to learn from and no right-against-
    wrong pair yet. p is the body's pass rate on the kind over the last `last` gates of this DNA (with one pass and
    one failure added, so a kind seen twice is not taken as settled), and the weight is p(1 - p), the most at
    one half. As the body gets better, easy kinds drop out of practice and harder ones come in by themselves.

    Where the body's own practice has been counted kind by kind (cells/body.json "kinds", grow_record.practice_kinds),
    the weight is s(1 - f) instead: s how often some member is right on the kind, f how often the body is. It is high
    exactly where practice makes a new right row. p(1 - p) gave a kind passed 2 of 2 and one passed 0 of 2 the same
    weight, so half the practice went to kinds already learned (agent C, 2026-11-16): each kind is passed either always
    or never, and only a never-passed kind that some member gets right sometimes can move."""
    try:
        b = json.loads(Path(body).read_text()) if body else json.loads((ROOT / "cells" / "body.json").read_text())
    except (OSError, ValueError):
        b = {}
    counts = b.get("kinds") if isinstance(b, dict) and b.get("dna") == dna and isinstance(b.get("kinds"), dict) else {}
    if counts:
        out = {}
        for k, c in counts.items():
            if isinstance(c, list) and len(c) == 3 and all(isinstance(x, (int, float)) for x in c):
                tries, right, some = c
                out[k] = ((some + 1) / (tries + 2)) * (1 - (right + 1) / (tries + 2))
        if out:
            return out
    rec = Path(record or ROOT / "training" / "reports" / "generations.jsonl")
    lines = []
    for ln in (rec.read_text().splitlines() if rec.exists() else []):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if d.get("dna", "t17") == dna and d.get("arms", {}).get("after", {}).get("kinds"):
            lines.append(d)
    seen = {}
    for d in lines[-last:]:
        for k, (p, n) in d["arms"]["after"]["kinds"].items():
            a = seen.setdefault(k, [0, 0])
            a[0], a[1] = a[0] + p, a[1] + n
    return {k: ((p + 1) / (n + 2)) * (1 - (p + 1) / (n + 2)) for k, (p, n) in seen.items()}


def practice(seed, n=16, level="mixed", out=None, log=print, minutes=0):
    """Fresh practice tasks done by the hive's members; the rows of those the check confirmed are appended to out.
    minutes: no new task is begun after so long (a runner's job that runs out of time keeps nothing; what is done
    by then is written task by task)."""
    born = time.time()
    from training import exam as E
    from training import exam_hard
    out = Path(out or GROW / "rows.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENT_TEAM_SOLO", "2")
    os.environ.setdefault("AGENT_TEAM_SOLO_ALONE", "1")
    home = E._memory(True)
    made = right = 0
    # where to practise most: the kinds the body's own exams show at its edge (edge and ladder alike; the ladder sets
    # how hard each is). The weights of the body's own DNA: those of another DNA say nothing about this one.
    weights = edge_weights(_dna()) if level in ("edge", "ladder") else None
    if weights:
        log("practice at the edge: " + ", ".join(f"{k} {w:.2f}" for k, w in sorted(weights.items(), key=lambda x: -x[1])), flush=True)
    levels = body_levels() if level == "ladder" else None
    if levels:
        log("how hard the lessons are now: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items())), flush=True)
    passed = []
    from training import practice_arms
    arm = os.environ.get("HIVE_ORDERS_ARM", "") if os.environ.get("HIVE_ORDERS_ARM", "") in practice_arms.ARMS else ""
    if arm:  # the members' orders follow the arm (team/solo.member, which every task's own process runs)
        os.environ.update(practice_arms.env_for(arm))
        log(f"the orders of this runner: {arm} (training/practice_arms.py)", flush=True)
    for tid, field, kind, task, check in E.exam(seed, n, (), level, without=exam_hard.HELD_OUT, weights=weights, levels=levels):
        if minutes and time.time() - born > minutes * 60:
            log(f"time is up after {len(passed)} of {n} tasks: practice ends here", flush=True)
            break
        r = E.sit_one(f"practice_{seed}", tid, task, check, home, kind=kind)
        # rows from every member the task's check confirmed, even where the hive's answer was not right (taught)
        here = E.RUNS / f"practice_{seed}" / tid
        if has_parts(here):  # worked part by part: the rows are each part's own (rows_by_part)
            new = rows_by_part(here, task, kind, seed, kinds=part_kinds(check))
        else:
            new = rows_of(here, task, kind, seed)
            new += pair_rows(here, task, kind, seed)
        if os.environ.get("HIVE_FORESEE") == "1":
            new += foresee_rows(E.RUNS / f"practice_{seed}" / tid, task, kind, seed)
        with open(out, "a") as f:
            for row in new:
                f.write(json.dumps(row) + "\n")
        made, right = made + sum(x["type"] not in ("foresee", "pair") for x in new), right + r["ok"]
        if not r["ok"]:
            remember_failure(E.RUNS / f"practice_{seed}" / tid, task, kind, check, r, out.with_name("failures.jsonl"))
        # and, kind by kind, whether the body was right and whether some member was (training/grow_record.practice_kinds):
        # practice is drawn where some member is right and the body is not, the only place it makes a new right row
        some = any(isinstance(m, dict) and m.get("ok") is True for m in r.get("members") or [])
        # and, for the record's split by orders and by the kinds' age (training/practice_arms.summary): right at the
        # first try, how long it took, whether it ran out of time, and whether its kind is one K2 added
        maker, parts = getattr(check, "maker", ""), getattr(check, "parts", None) or []
        try:
            first = practice_arms.first_try(E.RUNS / f"practice_{seed}" / tid)
        except Exception:  # noqa: BLE001 - a task whose log cannot be read says nothing of its first try
            first = None
        passed.append({"maker": maker, "level": getattr(check, "level", 0), "ok": bool(r["ok"]), "kind": kind, "some": some,
                       "first": first, "seconds": r.get("seconds"), "timeout": str(r.get("note") or "").startswith("timed out"),
                       "age": practice_arms.age(maker, parts)}
                      | ({"parts": [p[0] for p in parts if isinstance(p, (list, tuple)) and p]} if maker == "compound" else {}))
        # what was passed at which level, for the record to move the levels by (training/grow_record.levels); arm: the
        # orders this runner was given (HIVE_ORDERS_ARM, set by the practice job; "" when practice is not split)
        out.with_name("practice.json").write_text(json.dumps({"seed": seed, "arm": arm, "tasks": passed}))
        beyond = sum(x["type"] == "first" and x.get("agreed") is False for x in new)
        log(f"{'RIGHT' if r['ok'] else 'not right'} {tid} [{kind}] {r['seconds']}s, {len(new)} rows"
            + (f", {beyond} of the right members not among those that agreed" if beyond else ""), flush=True)
    return {"practice": seed, "level": level, "tasks": len(passed), "right": right, "rows": made, "file": str(out)} | ({"arm": arm} if arm else {})


def body_levels():
    """{maker: level, "compound": parts}: how hard the body's lessons are now (cells/body.json, moved by each
    generation's record)."""
    from team import ternary
    try:
        lv = json.loads(ternary.BODY.read_text()).get("levels") or {}
        return {str(k): int(v) for k, v in lv.items() if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9}
    except (OSError, ValueError, AttributeError):
        return {}


def _masked(why):
    """What the check said, with its numbers hidden: the hive is to find its mistake, not copy the answer."""
    return re.sub(r"\d+(?:[.,]\d+)*", "#", why or "")[:400]


def delivered(result):
    """Whether the hive gave an answer (its members agreed on one) and it was wrong, not "Not confirmed" or none."""
    a = str((result or {}).get("answer") or "").strip()
    return bool(a) and not a.startswith(("Not confirmed", "No member"))


def remember_failure(run, task, kind, check, result, path):
    """A task the hive got wrong, kept so that it can come back to it (dream): how to make the task again, its own
    last program and what that printed, and what the check said, numbers hidden."""
    if not getattr(check, "maker", None):
        return
    home = Path(run) / "members"
    program, printed, who = "", "", []
    for f in sorted(home.glob("member[0-9]*.json")):
        try:
            t = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        who.append(str(t.get("who", ""))[:80])
        if program:
            continue
        ms = [m for m in t.get("messages") or [] if m.get("role") == "assistant"]
        program = (t.get("program") or (solo_code(ms[-1]["content"]) if ms else "")).strip()
        printed = (t.get("answer") or "")[-600:]
    if not program or len(program) > 5000:
        return
    with open(path, "a") as f:
        f.write(json.dumps({"maker": check.maker, "rseed": check.rseed, "level": getattr(check, "level", 0), "parts": getattr(check, "parts", None),
                            "task": task, "kind": kind, "program": program, "printed": printed, "who": who,
                            "delivered": delivered(result),
                            "said": _masked(result.get("why")), "when": time.strftime("%Y-%m-%d %H:%M")}) + "\n")


def solo_code(text):
    from team import solo
    return solo.extract_code(text)


DREAM = ("{task}\n\nThis task was done once before, and the answer was wrong. The program written then:\n```python\n{program}\n```\n"
         "It printed: {printed}\nWhat the check said about it (its numbers hidden): {said}\n"
         "Find the mistake in that program, then write the whole corrected program.")


def dream(seed, n=4, failures=None, out=None, log=print, minutes=0, tries=2, weights=None):
    """The hive goes back to tasks it got wrong (failures.jsonl, kept by practice) and corrects itself. Each is made
    again exactly (training/exam.remake), with the old program and what the check said (numbers hidden); where the
    check now passes, the correction is a training row like any other: the old attempt and the verdict lead to the
    right program. Problem solving is the correcting of errors, and the hive's own errors are what it can correct
    (STaR's going back to failed problems, Zelikman et al. 2022; Reflexion, Shinn et al. 2023).
    tries: how many times a task is taken again before it is left for another night, each time with another cell beside
    the trunk (HIVE_ROTATE). Generation 28: one try corrected 5 of 24; a task one cell cannot see through, another may."""
    from training import exam as E
    out = Path(out or GROW / "rows.jsonl")
    src = Path(failures or out.with_name("failures.jsonl"))
    seen = []
    for ln in (src.read_text().splitlines() if src.exists() else []):
        try:
            seen.append(json.loads(ln))
        except ValueError:
            continue
    if not seen:
        return {"dream": seed, "tasks": 0, "right": 0, "rows": 0}
    r = random.Random(f"dream:{seed}")
    recent = seen[-400:]
    # the failures of kinds where some member is right now and then, and the body is not, first (the same weight as
    # practice: edge_weights). Generation 33's dreams took mostly kinds never passed (several parts, modules, facts on
    # the open web) and corrected 0 of 10 in two tries each; lessons start those kinds, dreams finish the ones begun.
    w = weights if weights is not None else edge_weights(_dna())
    pick, pool = [], list(recent)
    while pool and len(pick) < n:
        i = r.choices(range(len(pool)), weights=[max(0.01, w.get(f.get("kind", ""), 0.25)) for f in pool])[0]
        pick.append(pool.pop(i))
    os.environ.setdefault("AGENT_TEAM_SOLO", "2")
    os.environ.setdefault("AGENT_TEAM_SOLO_ALONE", "1")
    home = E._memory(True)
    right = made = 0
    born = time.time()
    for i, f in enumerate(pick):
        if minutes and time.time() - born > minutes * 60:
            log(f"time is up after {i} of {len(pick)} dreams", flush=True)
            break
        again = E.remake(f.get("maker", ""), f.get("rseed", ""), f.get("level", 0), f.get("parts"))
        if not again:
            continue
        field, kind, task, check = again
        text = DREAM.format(task=task, program=f["program"][:5000], printed=(f.get("printed") or "nothing")[:600], said=f.get("said", ""))
        was = os.environ.get("HIVE_ROTATE")
        turn = int(was) if (was or "").isdigit() else 0
        new = []
        for k in range(max(1, int(tries))):
            tid = f"d{seed}_{i:02d}" + (f"_{k}" if k else "")
            if k:  # another cell beside the trunk on each try after the first
                os.environ["HIVE_ROTATE"] = str(turn + k)
            try:
                res = E.sit_one(f"dream_{seed}", tid, text, check, home, kind=kind)
            finally:  # the runner's own turn given back exactly, unset if it was
                if was is None:
                    os.environ.pop("HIVE_ROTATE", None)
                else:
                    os.environ["HIVE_ROTATE"] = was
            # every try's members the check confirmed are rows, the tries that failed as a whole too (taught):
            # generation 28's dreams had 2 failed tries with a member the check confirmed
            here = E.RUNS / f"dream_{seed}" / tid
            new += rows_by_part(here, task, kind, seed) if has_parts(here) else rows_of(here, task, kind, seed) + pair_rows(here, task, kind, seed)
            if res["ok"] or (minutes and time.time() - born > minutes * 60):
                break
        new = once(new)
        for row in new:
            row["dream"] = True
        with open(out, "a") as fo:
            for row in new:
                fo.write(json.dumps(row) + "\n")
        right += res["ok"]
        made += sum(x["type"] != "pair" for x in new)
        log(f"dream {'CORRECTED' if res['ok'] else 'still wrong'} {tid} [{kind}] {res['seconds']}s, try {k + 1}, {len(new)} rows", flush=True)
    return {"dream": seed, "tasks": len(pick), "right": right, "rows": made}


def gather(out, gen, old=(), new=(), log=print):
    """Every checked row so far in one file: those of earlier generations as they are, then this generation's, each
    stamped with the generation it was practised in (training tells new rows from ones the parents already learned).
    The same row is kept once. Returns (rows kept from before, rows added)."""
    from training import grow_record
    lines_old = [ln for f in old for ln in grow_record.lines(f)]
    lines_new = []
    for f in new:
        for ln in grow_record.lines(f):
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if isinstance(row, dict):
                row.setdefault("gen", gen)
                lines_new.append(json.dumps(row))
    kept = grow_record.union([lines_old, lines_new])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(ln + "\n" for ln in kept))
    before = len(grow_record.union([lines_old]))
    return before, len(kept) - before


def rows_from(folders, out):
    """Rows from sittings that were already done: every task folder with an exam.json, with rows from the members
    its check confirmed (taught; in a sitting from before the verdicts were kept, only a task that says ok)."""
    made = 0
    with open(out, "a") as f:
        for folder in folders:
            for ex in sorted(Path(folder).rglob("exam.json")):
                try:
                    r = json.loads(ex.read_text())
                except ValueError:
                    continue
                task = (ex.parent / "members" / "TASK.md")
                if isinstance(r, dict) and task.exists():
                    t = task.read_text().strip()
                    for row in (rows_by_part(ex.parent, t) if has_parts(ex.parent) else rows_of(ex.parent, t) + pair_rows(ex.parent, t)):
                        f.write(json.dumps(row) + "\n")
                        made += 1
    return made


# ---- train -----------------------------------------------------------------------------------------------------------
class _Prefix:
    """The layers below the first trained one are frozen, so for a given row they compute the same thing on every
    pass. Their output (the input to the first trained layer) is kept from the first pass of each row, and on later
    passes those layers are skipped and the kept value is used: the same result, without a third to half of the
    forward work from the second pass on (a cell trains the last 14 layers of 28 or 36). Rows are known by their
    tensor. The kept values live in memory up to HIVE_PREFIX_GB (3 by default), or, given a folder, in files there
    up to HIVE_PREFIX_DISK_GB (6): an 8B DNA on a 16 GB runner has no memory to spare for them.
    HIVE_UNLOAD_PREFIX=1 supplies all training and kept-back sequences up front. The complete cache must fit
    on temporary disk and within HIVE_PREFIX_DISK_GB if explicitly set before the frozen weights can be freed."""

    def __init__(self, model, first, log=print, folder=None, *, rows=None, cast=contextlib.nullcontext, beat=None, until=0.0):
        import tempfile
        import torch
        decoder = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
        self.kept, self.key, self.skip, self.bytes, self.hits, self.misses = {}, None, False, 0, 0, 0
        self.batch = None
        self.unloaded = False
        self.temp = None
        self.folder = Path(folder) if folder else None
        if rows is not None:
            try:
                self.temp = tempfile.TemporaryDirectory(prefix="hive-prefix-")
            except OSError as exc:
                log(f"Phase 0 could not create its temporary cache ({exc}). Keeping the full decoder.", flush=True)
                self.most, self.folder = 0, None
                return
            self.folder = Path(self.temp.name)
        elif self.folder:
            import shutil
            shutil.rmtree(self.folder, ignore_errors=True)
            self.folder.mkdir(parents=True, exist_ok=True)
        self.most = float(os.environ.get("HIVE_PREFIX_DISK_GB", "6") if self.folder else os.environ.get("HIVE_PREFIX_GB", "3")) * 2**30
        if rows is not None:
            started = time.monotonic()
            if self.prepare(decoder, first, rows, cast, beat, until, log):
                self.unload(decoder, first)
                self.unloaded = True
                log(f"Phase 0 cached {len(self.kept)} sequences ({self.bytes / 2**30:.3f} GiB) and unloaded "
                    f"{first} frozen layers in {time.monotonic() - started:.3f} s", flush=True)
                return

        layers = decoder.layers
        for layer in layers[:first]:
            orig = layer.forward

            def passthrough(hidden_states, *a, _orig=orig, **k):
                return hidden_states if self.skip else _orig(hidden_states, *a, **k)
            layer.forward = passthrough

        def pre(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if self.batch is not None:
                h = self._batch_boundary(h.shape[1], None if self.skip else h)
            elif self.skip:
                h = self._get(self.key)
            elif self.key is not None and self.key not in self.kept and torch.is_grad_enabled() and self.bytes < self.most:
                self._put(self.key, h.detach())
            if args:
                return (h,) + tuple(args[1:]), kwargs
            return args, dict(kwargs, hidden_states=h)
        layers[first].register_forward_pre_hook(pre, with_kwargs=True)

    def prepare(self, decoder, first, rows, cast, beat, until, log):
        """Run the existing Qwen3 decoder only up to the boundary, without its final norm or the LM head."""
        import shutil
        import torch
        if decoder.config.model_type != "qwen3" or decoder.config.attention_dropout:
            log("Phase 0 needs a deterministic Qwen3 prefix. Keeping the full decoder.", flush=True)
            return False
        sequences = {id(ids): ids for ids in rows}
        layers, norm = decoder.layers, decoder.norm
        decoder.layers, decoder.norm = layers[:first], torch.nn.Identity()
        try:
            with torch.no_grad():
                for i, (key, ids) in enumerate(sequences.items()):
                    if until and time.time() >= until:
                        log("Phase 0 reached its pre-pass deadline. Keeping the full decoder.", flush=True)
                        return False
                    if beat is not None:
                        beat["t"], beat["what"] = time.time(), f"phase 0 prefix, sequence {i + 1} of {len(sequences)}"
                    with cast():
                        h = decoder(input_ids=ids, use_cache=False).last_hidden_state
                    if i == 0:  # Use the boundary's actual dtype, including under autocast.
                        need = sum(row.numel() for row in sequences.values()) * h.shape[-1] * h.element_size()
                        free = shutil.disk_usage(self.folder).free
                        room = min(self.most, free) if "HIVE_PREFIX_DISK_GB" in os.environ else free
                        if need > room:
                            log(f"Phase 0 needs {need} cache bytes and has {room:.0f} available. Keeping the full decoder.", flush=True)
                            if free < self.most:  # Even the lazy cache budget would exhaust this filesystem.
                                self.most = 0
                            return False
                    self._put(key, h)
                    del h
        except OSError as exc:
            log(f"Phase 0 could not write its cache ({exc}). Keeping the full decoder.", flush=True)
            self.most = 0  # A failed file is never published in kept. Stop writing on this filesystem.
            return False
        finally:
            decoder.layers, decoder.norm = layers, norm
        return bool(sequences)

    def unload(self, decoder, first):
        from collections import OrderedDict
        from copy import copy
        import torch
        from training.ternary_base import _trim

        layers = decoder.layers
        del layers[:first]
        # ModuleList deletion renumbers the survivors. Preserve the full model's adapter keys on export.
        decoder.layers = torch.nn.Sequential(OrderedDict((str(i), layer) for i, layer in enumerate(layers, first)))
        # Qwen3 chooses each mask by its position in the loop. The LM and saved adapter keep the full config.
        decoder.config = copy(decoder.config)
        decoder.config.layer_types = decoder.config.layer_types[first:]
        decoder.config.num_hidden_layers = len(layers)
        decoder.embed_tokens = None  # A tied output head still owns the shared storage.

        def cached_input(module, args, kwargs):
            ids = kwargs.pop("input_ids")
            if self.batch is not None and self.key == id(ids):
                kwargs["inputs_embeds"] = self._batch_boundary(ids.shape[1])
                self.hits += len(self.batch)
            else:
                kwargs["inputs_embeds"] = self._get(id(ids))
                self.hits += 1
            return args, kwargs

        decoder.register_forward_pre_hook(cached_input, with_kwargs=True)
        _trim()  # Collect the old modules and return freed allocator pages before training.

    def _put(self, key, h):
        import torch
        if not self.folder:
            self.kept[key] = h
        else:
            raw = {2: torch.int16, 4: torch.int32}[h.element_size()]  # the numbers' own bytes, whatever their type
            path = self.folder / f"{len(self.kept)}.bin"
            h.contiguous().view(raw).numpy().tofile(str(path))
            self.kept[key] = (path, tuple(h.shape), h.dtype, raw)
        self.bytes += h.numel() * h.element_size()

    def _get(self, key):
        import torch
        v = self.kept[key]
        if not isinstance(v, tuple):
            return v
        path, shape, dtype, raw = v
        return torch.from_file(str(path), shared=False, size=math.prod(shape), dtype=raw).view(dtype).view(shape)

    def _batch_boundary(self, width, computed=None):
        """Assemble hits and freshly computed rows at their original positions in the right-padded batch."""
        import torch
        result = None
        for i, (key, length) in enumerate(self.batch):
            if computed is None or key in self.kept:
                cached = self._get(key)
                if result is None:
                    result = (cached.new_zeros((len(self.batch), width, cached.shape[-1])) if computed is None
                              else computed.clone())
                result[i, :length] = cached[0]
            elif torch.is_grad_enabled():
                row = computed[i:i + 1, :length].detach()
                if self.bytes + row.numel() * row.element_size() <= self.most:
                    # A view in the in-memory cache would pin the entire batch, including its padding.
                    self._put(key, row if self.folder else row.clone())
        return computed if result is None else result

    def done(self):
        if self.temp:
            self.temp.cleanup()
        elif self.folder:
            import shutil
            shutil.rmtree(self.folder, ignore_errors=True)

    @contextlib.contextmanager
    def use(self, ids):
        if self.unloaded:
            yield
            return
        self.key = id(ids)
        self.skip = self.key in self.kept
        self.hits += self.skip
        self.misses += not self.skip
        try:
            yield
        finally:
            self.key, self.skip = None, False

    @contextlib.contextmanager
    def use_batch(self, ids, rows):
        """Key a padded forward by its canonical unpadded row tensors, also usable by single-row forwards."""
        batch = [(id(row), row.size(1)) for row in rows]
        if not batch or len(batch) != ids.size(0) or any(length > ids.size(1) for _, length in batch):
            raise ValueError("prefix rows must fit their right-padded batch")
        self.batch, self.key = batch, id(ids)
        if not self.unloaded:
            hits = sum(key in self.kept for key, _ in batch)
            self.skip = hits == len(batch)
            self.hits += hits
            self.misses += len(batch) - hits
        try:
            yield
        finally:
            self.batch, self.key, self.skip = None, None, False


def shorten(messages, keep):
    """The messages with the middle of the longest earlier program cut out (an assistant turn after the task), keeping
    `keep` of its lines: the first ones and the last ones, and a line that says how many were left out. The messages
    themselves when there is no earlier program to cut. Only for context a row shows before its answer."""
    turns = [i for i, m in enumerate(messages) if i >= 2 and m.get("role") == "assistant"]
    if not turns:
        return messages
    i = max(turns, key=lambda j: len(messages[j]["content"]))
    lines = messages[i]["content"].splitlines()
    n = max(6, int(len(lines) * keep))
    if len(lines) <= n + 2:
        return messages
    head, tail = lines[:n * 2 // 3], lines[len(lines) - (n - n * 2 // 3):]
    cut = len(lines) - len(head) - len(tail)
    fence = ["```"] if lines[0].startswith("```") and not tail[-1].startswith("```") else []
    text = "\n".join(head + [f"# ... ({cut} lines of this program left out) ..."] + tail + fence)
    return [dict(m) for m in messages[:i]] + [dict(messages[i], content=text)] + [dict(m) for m in messages[i + 1:]]


KEEP_BACK = int(os.environ.get("HIVE_KEEP_BACK", 2))  # lessons of every kind never trained on, only measured (0: none)


def kept_back(rows, n=None):
    """{kind: [task, ...]}: the `n` lessons of each kind that are never trained on (a lesson is its task; its rows
    share it). The loss of a lesson trained on says only that its rows were met; one never met says whether the kind
    is learned (generation 35's new hard subjects ended at a loss of 0.4 to 1.6 on their own rows, and nothing said
    what that meant for the tasks to come). Chosen by a hash of the task, so every slice of a child keeps back the
    same ones, whatever order the rows come in; and since the rows only grow from one generation to the next, a
    lesson kept back now was kept back in every generation before (the smallest two hashes of a larger set are new
    ones or were already the smallest two).
    Measured on the small DNA (t17, 160 passes, 2026-11-16): kept-back lessons of the two kinds taught 10 lessons each
    went 0.51 -> 0.19 (a join) and 0.91 -> 0.38 (a module); of two kinds never taught, 1.57 -> 1.17 and 0.34 -> 0.25.
    The untaught fall by about a quarter too (the lessons' common way of writing), the taught by more than half: read
    a kind's fall beside an untaught one's, not alone."""
    import zlib
    n = KEEP_BACK if n is None else n
    by = {}
    for r in rows:
        if r.get("who") == "lesson" and r.get("task"):
            by.setdefault(r.get("kind") or "?", set()).add(r["task"])
    return {k: sorted(ts, key=lambda t: (zlib.crc32(t.encode()), t))[:n] for k, ts in sorted(by.items()) if n > 0}


def row_limit(now, born, minutes, most, late=None):
    """Seconds one row may take before the job ends itself: `most` while the cell's time lasts, and only `late`
    (HIVE_LATE_ROW_SECONDS, 600) once `minutes` have passed since `born`: when the time is up a stuck row has
    minutes, not 25, so a slice never runs on far past its budget (g51: one slice ran 220 minutes of a 170 budget)."""
    if late is None:
        late = float(os.environ.get("HIVE_LATE_ROW_SECONDS", 600))
    return min(most, late) if minutes and now - born > minutes * 60 else most


def gguf_seconds(now, born, minutes, default=900.0, wall=1200.0):
    """Seconds the GGUF conversion may take: HIVE_GGUF_SECONDS (900), but never past `wall` seconds (20 minutes) after the
    cell's budget `minutes` since `born`, with 3 minutes always allowed. The whole slice is then over by born + minutes + 20
    + 3 minutes at the latest, inside the step's limit (g56: two slices ran past the job's 200 minutes with nothing saved)."""
    if not minutes:
        return default
    return max(180.0, min(default, born + minutes * 60 + wall - now))


def precheck_deadline(started, now, minutes, profile):
    """In a short CPU trial reserve at least half the remaining clock for training, apart from a row in flight."""
    if profile not in ("85", "110") or minutes <= 0:
        return 0.0
    return now + max(0.0, started + minutes * 60 - now) / 4


def measure_until(now, minutes, share, floor=120.0):
    """Clock time after which an ungated measuring phase before training (the raw pieces of `read`, the kept-back lessons'
    first measure) stops: `share` of the cell's `minutes` from `now`, at least `floor` seconds; 0.0 with no time given.
    g59: both phases ran with no deadline (57 and 35 minutes of a 170 budget on a swapping runner) and two slices
    reached the job's limit with the budget gone before one row was trained."""
    if not minutes:
        return 0.0
    return now + max(floor, minutes * 60 * share)


def paired_kept_losses(back, before, after):
    """The evaluator visits items in order and may stop between rows. Compare only the common prefix per kind."""
    report = {}
    for kind in sorted(set(back.values())):
        left, right = before.get(kind, []), after.get(kind, [])
        pairs = min(len(left), len(right))
        report[kind] = {
            "lessons": sum(k == kind for k in back.values()),
            "rows": pairs,
            "before": round(sum(left[:pairs]) / pairs, 4) if pairs else None,
            "after": round(sum(right[:pairs]) / pairs, 4) if pairs else None,
        }
    return report


def kept_loss(model, items, dev, cast, beat=None, until=0.0, chunk_tokens=0):
    """{kind: [loss of each row]} of the lessons kept back, with the cell's weights as they are now: the same loss
    as training's (the error on the answer's pieces), one row at a time with no gradient, so nothing is held beyond
    one row's logits (an 8B cell trains in 14.7 of 15.6 GB). items: [(kind, (ids, k))] from train's encode.
    until: a clock time (time.time()) after which no further row is measured: the cell's time is up, and what was
    measured so far stands (a slice must save and upload its record, not run on measuring)."""
    import torch
    from training.lm_head_loss import answer_nll
    was = model.training
    model.eval()  # no dropout: the same rows measured twice give the same number
    out = {}
    with torch.no_grad():
        for j, (kind, (ids, k)) in enumerate(items):
            if until and time.time() > until:
                break
            if beat is not None:
                beat["t"], beat["what"] = time.time(), f"the lessons kept back, row {j + 1} of {len(items)}"
            ids = ids.to(dev)
            with cast():
                loss = answer_nll(model, ids, k, chunk_tokens) / (k * ids.shape[0])
            out.setdefault(kind, []).append(loss.item())
    if was:
        model.train()
    return out


def slice_rows(rows, k, n, policy=None):
    """Slice k of n of the rows: each row in exactly one slice, chosen by its content (the same row in the same slice on
    every runner, whatever order the rows come in)."""
    import zlib
    if not 0 <= k < n:
        raise ValueError(f"slice {k} of {n}: k must be from 0 to n - 1")
    policy = os.environ.get("HIVE_SHARD_POLICY", "hash") if policy is None else policy
    if policy == "cost":
        from training.shard_budget import cost_assignment
        rows = list(rows)
        return [r for r, owner in zip(rows, cost_assignment(rows, n)) if owner == k]
    if policy != "hash":
        raise ValueError(f"Unknown training shard policy {policy!r}")
    return [r for r in rows if zlib.crc32(json.dumps([r["messages"][1:], r["output"]], sort_keys=True).encode()) % n == k]


class Study:
    """Which row a cell trains on next. Each kind of task is passed either always or never, and the rows a cell has
    already learned cost passes and teach nothing (g28code's loss went 0.096 -> 0.048 on rows mostly of kinds it
    passed): the passes go where the error is. The rows are sorted into subjects (kind of task, type of row, new this
    generation or not; see subject()). Each pass picks a subject with odds max(its loss, FLOOR), times up to OWN by
    the share of its rows that are the cell's own organ, and trains on that subject's next row (shuffled, cycled). A subject's loss
    is the mean of its last three. A subject not measured yet counts 1.0 and is met before any measured one, so the
    first sweep meets every subject (rehearsing old work too) without waiting on luck. A row is met at most MOST
    times; a subject with none left drops out. Raw text ("read") and judged pairs ("judged") keep a fixed SHARE of
    the passes each instead: their loss does not say what is left to learn.

    New hard subjects (hard: this generation's rows of a kind of exam_hard, the lessons above all; see new_hard())
    are studied until they are learned. Generation 35 met each lesson row 4 times at the most and its new hard
    subjects ended at a loss of 0.4 to 1.6; walk a site was under GOAL in 4 of 23 slices. So while a new hard
    subject's loss is GOAL or more, its rows may be met up to HARD_MOST times, and FIRST of the passes go to such
    subjects. Those of `web` (the kinds that read the web, and the date sums on what was read) come first: first in
    the first sweep, and WEB_ODDS times the odds among the subjects not learned yet (not all of their passes: a
    slice of an hour or two could not give the 4 web kinds' lessons HARD_MOST meetings a row, and the other hard kinds
    would get none). The other passes are picked by the odds above among the other subjects, so what is already
    known is still rehearsed. Once under GOAL a new hard subject is one like the others again (MOST meetings a row).
    Measured on the small DNA (t17, the same parent and 82 rows, 34 of them lessons, 20 minutes; 2026-11-17): the
    13 new hard subjects ended at 0.35 to 2.04 with 4 meetings a row, and all 13 under 0.3 with this rule (6 to 20
    passes each, 384 passes in all against 316); what was under 0.1 before stayed under 0.1. With HARD_MOST 16 the
    fix row of a fact found on the open web met its limit at 0.76, so the limit is 32."""
    FLOOR, OWN, MOST, SHARE, FIXED = 0.05, 3.0, 4, 0.03, ("read", "judged")  # SHARE was 0.10: g35words gave raw text 30 of 296 passes and its loss went 3.08 -> 2.99
    GOAL, HARD_MOST, FIRST, WEB_ODDS = 0.3, 32, 0.75, 3.0

    def __init__(self, subjects, own=(), rng=None, most=None, weight=None, hard=(), web=(), hard_most=None):
        """own: {subject: the share of its rows that are the cell's own organ}, or subjects that are wholly its own.
        hard: the new hard subjects; web: those of them that come first (both may name subjects that are not here)."""
        self.rng = rng or random.Random(0)
        self.most = self.MOST if most is None else most
        self.hard_most = max(self.most, self.HARD_MOST if hard_most is None else hard_most)
        self.weight = self.OWN if weight is None else float(weight)
        self.own = {s: float(v) for s, v in own.items()} if isinstance(own, dict) else {s: 1.0 for s in own}
        self.hard = {s for s in hard if s in subjects and subjects[s] and s not in self.FIXED}
        self.web = {s for s in web if s in self.hard}
        self.queue, self.rows, self.passes, self.losses, self.first = {}, {}, {}, {}, {}
        for s, idx in subjects.items():
            if not idx:
                continue
            q = []
            for _ in range(self.hard_most if s in self.hard else self.most):  # each round of a subject's rows in its own order
                r = list(idx)
                self.rng.shuffle(r)
                q += r
            self.queue[s], self.rows[s], self.passes[s], self.losses[s] = q[::-1], len(idx), 0, []
        self.size = sum(len(q) for q in self.queue.values())  # the most passes there can be
        self.n = 0
        self.skipped, self.waiting = {}, set()  # rows put back for want of memory (MemoryGuard), by subject

    def skip(self, s, idx):
        """The row just picked is not trained on now (it would not fit in the memory there is): it goes back to the
        end of its subject's queue, is not counted as met, and its subject waits until a pass is made elsewhere."""
        self.queue[s].insert(0, idx)
        self.passes[s] -= 1
        self.n -= 1
        self.skipped[s] = self.skipped.get(s, 0) + 1
        self.waiting.add(s)

    def loss(self, s):
        h = self.losses[s][-3:]
        return sum(h) / len(h) if h else 1.0

    def odds(self, s):
        return max(self.loss(s), self.FLOOR) * (1.0 + (self.weight - 1.0) * self.own.get(s, 0.0))

    def behind(self, s):
        """A new hard subject not learned yet: measured, and its loss GOAL or more."""
        return s in self.hard and bool(self.losses[s]) and self.loss(s) >= self.GOAL

    def open(self, s):
        """Rows left to meet: up to HARD_MOST meetings a row while the subject is behind, MOST otherwise."""
        return bool(self.queue[s]) and (self.behind(s) or self.passes[s] < self.rows[s] * self.most)

    def pick(self, eligible=None):
        """(subject, row index), or None when every row has been met as often as it may be."""
        alive = [s for s in self.queue if self.open(s)]
        if not alive:
            return None
        alive = [s for s in alive if s not in self.waiting] or alive  # a subject just skipped waits its turn
        rest = [s for s in alive if s not in self.FIXED]
        s = next((f for f in self.FIXED if f in alive and self.passes[f] < self.SHARE * (self.n + 1)), None)
        if s is None:
            fresh = [x for x in rest if not self.losses[x]]
            if eligible is not None:  # Other rows in this micro-batch have not reported their losses yet.
                fresh = [x for x in fresh if not self.passes[x]] or fresh
            # the first sweep meets the new web subjects first, then the other new hard ones, then the rest
            fresh = [x for x in fresh if x in self.web] or [x for x in fresh if x in self.hard] or fresh
            late = [x for x in rest if self.behind(x)]
            others = [x for x in rest if not self.behind(x)]
            state = self.rng.getstate() if eligible is not None and late and others else None
            if fresh:
                pool = fresh
            elif late and (not others or self.rng.random() < self.FIRST):
                pool = late
            else:
                pool = others or rest or alive
            late_pool = pool is late
            if eligible is not None:
                pool = [x for x in pool if eligible(self.queue[x][-1])]
                if not pool:
                    if state is not None:
                        self.rng.setstate(state)
                    return None
            s = self.rng.choices(pool, weights=[self.odds(x) * (self.WEB_ODDS if late_pool and x in self.web else 1.0) for x in pool])[0]
        elif eligible is not None and not eligible(self.queue[s][-1]):
            return None  # Finish this batch so the next draw can honor the due read/judged share.
        self.passes[s] += 1
        self.n += 1
        return s, self.queue[s].pop()

    def learned(self, s, loss):
        self.waiting.clear()
        if s not in self.first:
            self.first[s] = loss
        self.losses[s].append(float(loss))

    def table(self):
        return [{"subject": s, "rows": self.rows[s], "passes": self.passes[s],
                 "loss_first": round(self.first[s], 4) if s in self.first else None,
                 "loss_last": round(self.loss(s), 4) if self.losses[s] else None}
                | ({"new_hard": True} if s in self.hard else {})
                for s in sorted(self.rows, key=lambda x: -self.passes[x])]


class _RowCheckpoints:
    """Toggle non-reentrant checkpointing on the decoder suffix, without installing input-gradient hooks."""

    def __init__(self, model, first):
        from functools import partial
        from torch.utils.checkpoint import checkpoint as recompute
        from transformers.modeling_layers import GradientCheckpointingLayer

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = tuple(getattr(getattr(base, "model", None), "layers", ()))[first:]
        if not self.layers or not all(isinstance(layer, GradientCheckpointingLayer) for layer in self.layers):
            raise ValueError("per-row checkpointing needs checkpointable decoder layers")
        self.enabled = False
        for layer in self.layers:
            # An adapter can receive gradients even if its inputs came from a fully frozen prefix.
            layer._gradient_checkpointing_func = partial(recompute, use_reentrant=False)

    def set(self, enabled):
        if enabled != self.enabled:
            for layer in self.layers:
                layer.gradient_checkpointing = enabled
            self.enabled = enabled


def _cgroup_limits(root=None, membership=None):
    """Finite v2 limits (memory.max and memory.high; "max" is none) for this process and its ancestors in a standard
    Linux cgroup mount. The paths can be given, so a test needs no real cgroup."""
    root = Path("/sys/fs/cgroup") if root is None else root
    try:
        line = next(x for x in (Path("/proc/self/cgroup") if membership is None else membership).read_text().splitlines() if x.startswith("0::"))
        leaf = root / line[3:].lstrip("/")
        if ".." in leaf.parts or not leaf.is_dir():
            leaf = root  # A namespace may expose the effective cgroup as the mount root.
        limits = []
        for folder in (leaf, *leaf.parents):
            if root != folder and root not in folder.parents:
                break
            for name in ("memory.max", "memory.high"):  # the runner's cap (scripts/capped_run.sh) sets both
                try:
                    limit = (folder / name).read_text().strip()
                    if limit != "max":
                        limits.append((folder / "memory.current", int(limit)))
                except (OSError, ValueError):
                    continue  # A root without a memory controller must not erase a known child limit.
        return limits
    except (OSError, ValueError, StopIteration):
        return []


def _available(ps, limits):
    available = ps.virtual_memory().available
    for current, limit in limits:
        try:
            used = int(current.read_text())
        except (OSError, ValueError):
            return 0  # A previously observed finite limit can no longer be measured.
        try:
            stat = {key: int(value) for key, value in
                    (line.split() for line in current.with_name("memory.stat").read_text().splitlines())}
            # Reading model shards fills the file cache within this same cgroup. Credit only clean,
            # inactive file pages outside mapped, shared or pinned memory. Missing stats give no credit.
            protected = sum(stat[key] for key in ("file_dirty", "file_writeback", "file_mapped", "shmem", "unevictable"))
            reclaimable = max(0, stat["inactive_file"] - protected)
        except (OSError, ValueError, KeyError):
            reclaimable = 0
        available = min(available, max(0, limit - used + reclaimable))
    return available


def pair_pieces(chosen, rejected, orpo):
    """Peak sequence length charged to one training pair."""
    return max(chosen, rejected) if orpo else chosen + rejected


def row_head_bytes(vocab, target, rejected=0, projection=None, *, orpo=False):
    """Peak head workspace, including policy graphs live during a DPO reference pass (ORPO keeps one at a time)."""
    def workspace(n):
        rows = projection(n) if projection else n + 1
        if rows == n + 1:  # stock CE keeps its full projection for backward
            size = rows * vocab * 16
            return size, size
        # One logits tile and its fp32 conversion. Only per-token normalizers
        # survive forward, so the two policy graphs retain no vocabulary tiles.
        return rows * vocab * 8, 0

    right, keep_right = workspace(target)
    if not rejected:
        return right
    wrong, keep_wrong = workspace(rejected)
    return max(right, wrong) if orpo else keep_right + keep_wrong + max(right, wrong)


class MemoryGuard:
    """Whether a row fits in the memory there is now, measured, not guessed. g38 lost 7 of 24 8B slices and g39 5 of
    20 ("the hosted runner lost communication"); every slice that lived stood at 14.1-14.8 of 15.6 GB, its memory
    jumping when a new longest row came, and the length formula (`fits`) missed 1.5-1.9 GB (scratch buffers, the
    optimizer's state and the OS were not in it). So before each pass the row's need, pieces x cost_per_piece, is
    set against what is available: with less than SPARE GB left the row is skipped for this pass (Study.skip; the
    Study picks another). The cost per piece starts at the formula's and is learned from the run: after each pass,
    the fall of the available memory during it (sampled every 0.1 s, against the lower of before and after, so what
    a pass keeps for good, a scratch buffer or the optimizer's state, is not taken for a row's cost) over its
    pieces; the largest seen, never under the formula's. Only rows at least half as long as the longest trained so
    far teach it: a short row's fixed part (the logits, the update) over few pieces would say a piece costs more than
    it does."""
    GB = 2**30

    def __init__(self, floor, spare_gb=None, psutil=None, every=0.1, *, checkpoint_floor=None, limits=()):
        if psutil is None:
            import psutil
        self.ps = psutil
        self.limits = limits
        self.checkpoint_floor = self.checkpoint_cost = checkpoint_floor
        self.checkpoint_longest = 0
        self.row_modes = {"ordinary": 0, "checkpoint": 0}
        self.last_mode = None
        self.mode_changes = 0
        self.floor = self.cost = float(floor)  # bytes a piece
        self.spare = float(os.environ.get("HIVE_MEMORY_SPARE_GB", 1.0) if spare_gb is None else spare_gb) * self.GB
        self.every, self.longest = every, 0
        self.before = self.low = None
        self.min_avail, self.swap_peak = float("inf"), 0
        self.skips = {}
        self.sample()
        self._on = every > 0
        if self._on:
            import threading
            threading.Thread(target=self._watch, daemon=True).start()

    def available(self):
        return _available(self.ps, self.limits)

    def swap(self):
        return self.ps.swap_memory().used

    def sample(self):
        a, w = self.available(), self.swap()
        self.min_avail, self.swap_peak = min(self.min_avail, a), max(self.swap_peak, w)
        if self.low is not None:
            self.low = min(self.low, a)
        return a

    def _watch(self):
        while self._on:
            time.sleep(self.every)
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - a sample missed is no reason to stop
                pass

    def stop(self):
        self._on = False

    def need(self, pieces):
        return pieces * self.cost

    def admit(self, pieces, s=""):
        """True when a row of `pieces` fits now; otherwise the skip is counted under its subject."""
        a = self.sample()
        if a - self.need(pieces) >= self.spare:
            return True
        self.skips[s] = self.skips.get(s, 0) + 1
        return False

    def fitting_mode(self, pieces, head_bytes, room):
        """Probe a candidate without counting a skip or changing the checkpoint mode."""
        # A sampled peak already includes the head. Only the analytic activation floor needs it added.
        ordinary = max(pieces * self.floor + head_bytes, pieces * self.cost)
        recomputed = (max(pieces * self.checkpoint_floor + head_bytes, pieces * self.checkpoint_cost)
                      if self.checkpoint_floor is not None else math.inf)
        # Resist small fluctuations after a tight row. Cap the exit margin on large models and scale it to
        # the ordinary cost for short rows. No margin on first admission, on entry, or if only ordinary fits.
        margin = min(self.GB / 4, ordinary / 4) if self.last_mode == "checkpoint" and recomputed <= room else 0
        return "ordinary" if ordinary + margin <= room else "checkpoint" if recomputed <= room else None

    def row_mode(self, pieces, subject, head_bytes, *, room=None, rows=1):
        """Count the rows in the selected mode. One sample covers all candidate sizes for a batch."""
        mode = self.fitting_mode(pieces, head_bytes, self.sample() - self.spare if room is None else room)
        if mode is not None:
            self.mode_changes += self.last_mode is not None and self.last_mode != mode
            self.last_mode = mode
            self.row_modes[mode] += rows
            return mode
        self.skips[subject] = self.skips.get(subject, 0) + 1
        return None

    def begin(self):
        self.before = self.low = self.sample()

    def end(self, pieces, mode="ordinary"):
        """After the pass (and the trim): learn the cost of a piece from the fall in available memory during it."""
        after = self.sample()
        if self.before is None or pieces <= 0:
            return
        drop = min(self.before, after) - self.low
        if mode == "checkpoint":
            if pieces >= self.checkpoint_longest / 2:
                self.checkpoint_cost = max(self.checkpoint_cost, drop / pieces)
            self.checkpoint_longest = max(self.checkpoint_longest, pieces)
        else:
            if pieces >= self.longest / 2:
                self.cost = max(self.cost, self.floor, drop / pieces)
            self.longest = max(self.longest, pieces)
        self.before = self.low = None

    def record(self):
        import resource
        return {"peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                "min_available_gb": round(self.min_avail / self.GB, 2), "swap_peak_gb": round(self.swap_peak / self.GB, 2),
                "rows_skipped": sum(self.skips.values()), "skipped_by_subject": dict(sorted(self.skips.items(), key=lambda x: -x[1])),
                "cost_per_piece_mb": round(self.cost / 2**20, 3), "formula_per_piece_mb": round(self.floor / 2**20, 3),
                "spare_gb": round(self.spare / self.GB, 2),
                **({"row_modes": dict(self.row_modes), "mode_changes": self.mode_changes,
                    "checkpoint_cost_per_piece_mb": round(self.checkpoint_cost / 2**20, 3),
                    "checkpoint_formula_per_piece_mb": round(self.checkpoint_floor / 2**20, 3)}
                   if self.checkpoint_floor is not None else {})}


def lr_share(p):
    """The learning rate's share at `p` of the training done (of its time, or of its passes): a warm-up over the
    first 5%, then a straight fall to 5% at the end."""
    p = min(max(p, 0.0), 1.0)
    return max(0.01, p / 0.05) if p < 0.05 else max(0.05, 1.05 - p)


def subject(row, new=False):
    """The subject a row is studied under (Study): kind of task, type of row, new to the line or not. Rows of type
    "done" (a right program's output -> DONE) are learned at once in every kind (g35code: 40 such subjects, loss about
    0.001, 102 of 444 passes, most of them the first sweep's), so they are one subject, two with the new ones. The
    cell's own organ is not a split (it doubled the subjects) but a weight on a subject by its share of own rows."""
    if row.get("type") == "done":
        return "done" + " · new" * bool(new)
    return " · ".join([row.get("kind") or "?", row.get("type") or "row"] + ["new"] * bool(new))


# The kinds whose lessons are studied first (Study): those read from the web, and the date sums on what was read.
# Generation 35: walk a site was under a loss of 0.3 in 4 of 23 slices.
WEB = ("walk a site over several pages", "a fact found on the open web", "a fact read and a date sum on it",
       "a fact from a site and a date sum on it")


def hard_kinds():
    """The kinds of the hard exam (training/kinds/, the kinds with HARD), read from their sources: each maker returns
    (field, kind, task, check) with the kind written out, and some makers reach the network before they return."""
    from training import kinds
    return {k for m in kinds.HARD_KINDS for k in kinds.written(m.KIND)}


def new_hard(subjects):
    """(the new hard subjects, those of them of a WEB kind) among subject names (subject()): rows new to the line of
    a hard kind, of any type but "done" (one subject for every kind, learned at once)."""
    hard_set, hard, web = hard_kinds(), set(), set()
    for s in subjects:
        kind, _, rest = s.partition(" · ")
        if rest.endswith(" · new") and kind in hard_set:
            hard.add(s)
            if kind in WEB:
                web.add(s)
    return hard, web


def _orpo_backward(score, ids, k, wid, wk, params, accum):
    """Differentiate the ORPO objective with only one sequence's activations live at a time."""
    import torch
    chosen = score(ids, k) / (k * ids.shape[0])
    # The coefficient of the chosen gradient depends on the rejected score. Keep only its
    # parameter gradients while scoring the other sequence. Ordinary backward also works with
    # reentrant activation checkpointing. Preserve any earlier minibatch accumulation.
    previous = [p.grad for p in params]
    for p in params:
        p.grad = None
    try:
        torch.autograd.backward(chosen)
        chosen_grads = [p.grad for p in params]
    finally:
        for p, grad in zip(params, previous):
            p.grad = grad
    chosen = chosen.detach().requires_grad_()
    rejected = score(wid, wk) / (wk * wid.shape[0])
    logp = -torch.stack((chosen, rejected))
    # expm1 retains precision near probability one. Bound log(0) when CE rounds to zero.
    log_odds = logp - (-torch.expm1(logp)).clamp_min(torch.finfo(logp.dtype).tiny).log()
    loss = chosen - 0.1 * torch.nn.functional.logsigmoid(log_odds[0] - log_odds[1])
    (loss / accum).backward()
    coefficient = chosen.grad.item()
    for p, grad in zip(params, chosen_grads):
        if grad is not None:
            if p.grad is None:
                p.grad = grad.mul_(coefficient)
            else:
                p.grad.add_(grad, alpha=coefficient)
    return chosen.item(), loss.item()


def _sft_batch(study, first, data, limit, guard, head_bytes):
    """Extend the Study's first pick with distinct SFT rows that fit the same padded pass."""
    idx = first[1]
    picks, seen = [first], {idx}
    lengths, targets = [data[idx][0].shape[1]], [data[idx][1]]
    room = guard.sample() - guard.spare if guard else None

    def eligible(idx):
        row = data[idx]
        if idx in seen or len(row) != 2:
            return False
        sizes = lengths + [row[0].shape[1]]
        if 4 * max(sizes) > 5 * min(sizes):  # At most 25% longer than the shortest row.
            return False
        return not guard or guard.fitting_mode(len(sizes) * max(sizes), head_bytes(sizes, targets + [row[1]]), room) is not None

    while len(picks) < limit:
        pick = study.pick(eligible)
        if pick is None:
            break
        picks.append(pick)
        idx = pick[1]
        seen.add(idx)
        lengths.append(data[idx][0].shape[1])
        targets.append(data[idx][1])
    pieces = len(picks) * max(lengths)
    mode = None
    if len(picks) > 1:
        mode = guard.row_mode(pieces, first[0], head_bytes(lengths, targets), room=room, rows=len(picks)) if guard else "ordinary"
    return picks, pieces, mode


def _padded_answers(rows, pad_id):
    """Right-pad canonical CPU sequences and supervise only their real answer tokens."""
    import torch
    ids = torch.nn.utils.rnn.pad_sequence([x[0] for x, _ in rows], batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([x.shape[1] for x, _ in rows])
    mask = torch.arange(ids.shape[1]) < lengths[:, None]
    labels = torch.full_like(ids, -100)
    for i, (x, k) in enumerate(rows):
        end = x.shape[1]
        labels[i, end - k:end] = x[0, -k:]
    return ids, mask, labels


def train(rows_file, name, base="t17", epochs=2.0, rank=16, lr=2e-4, max_len=1152, layers="all", accum=4, seed=0, limit=0, minutes=0, log=print,
          lean="", weight=3, foresee=False, ternary_edit=False, out_dir=None, checkpoint=True, dtype="bf16", pairs=False, beta=0.1, int8_base=False, orpo=None, read=0, judged=0, parent="", parent_cell="",
          trunk="", pull=0.1, gen="", fresh=2, shard="", loss_chunk_tokens=None,
          donor="", donor_pull=None, sibling="", sibling_cis=None):
    """A LoRA weight edit of a ternary member from the hive's rows, on a CPU (or a GPU where there is one, without
    --int8-base: a Colab notebook, training/colab.py); written as a GGUF adapter the
    ternary runtime loads beside the packed base.

    parent: the edit the child starts as. trunk: the body's shared edit (training/merge.py); the child then starts
    as its parent pulled `pull` of the way toward it, so every line of descent stays part of one body.
    donor: an active, release-named adapter file or a directory of candidates. A positive HIVE_DONOR_PULL
    takes that share from the parent only after two flat generations. sibling: an offline adapter or B-only
    snapshot. HIVE_SIBLING_CIS is the coefficient of the squared cross-Gram penalty. Both default to zero.
    parent_cell: the parent's name; what that cell (and the trunk) got wrong in practice, set beside the program
    that was right, is taught to the child as a correction. gen: this generation's name; its rows are new to the
    whole line and are studied as subjects of their own (Study). epochs: the passes over the rows when no `minutes`
    are given; with `minutes` the cell trains until the time is up. weight: how much more the cell's own organ
    weighs in the pick of the next subject. fresh: no longer used (the Study's loss rule replaced the copies).
    shard: "k/n", this runner's slice of the rows (each row in one slice, by its content): a child is grown on n
    runners at once, each from the same start on its own slice, and their edits are averaged into the child
    (training/merge.py join). Generation 28's children got through a third of their rows in their time; n runners
    study n times as many."""
    born = time.time()
    donor_pull = float(os.environ.get("HIVE_DONOR_PULL", 0)) if donor_pull is None else donor_pull
    sibling_cis = float(os.environ.get("HIVE_SIBLING_CIS", 0)) if sibling_cis is None else sibling_cis
    if not (0 <= donor_pull <= 1) or donor_pull and not (pull >= 0 and pull + donor_pull <= 1):
        raise ValueError("birth weights must be nonnegative and sum to one")
    if not math.isfinite(sibling_cis) or sibling_cis < 0:
        raise ValueError("HIVE_SIBLING_CIS must be finite and nonnegative")
    selected = None
    if donor_pull and donor:
        from training.lineage import organ_of
        selected = _birth_donor(donor, lean or organ_of(name), base, gen)
        if selected is None:
            log("no local cross-organ donor qualifies on the last two recorded exams; using the usual birth weights", flush=True)
    batch_size = int(os.environ.get("HIVE_TRAIN_BATCH", "1"))
    if batch_size not in (1, 2, 4):
        raise ValueError("HIVE_TRAIN_BATCH must be 1, 2 or 4")
    # --orpo (a versus trial can pass it) or HIVE_PAIRS_ORPO=1; off by default
    orpo = pairs and (orpo if orpo is not None else os.environ.get("HIVE_PAIRS_ORPO", "0") == "1")
    plateau_mode = os.environ.get("HIVE_TRAIN_PLATEAU", "off")
    if plateau_mode not in ("off", "observe", "stop"):
        raise ValueError(f"Unknown training plateau mode {plateau_mode!r}")
    import torch
    from huggingface_hub import snapshot_download
    from training.lm_head_loss import answer_nll, batch_projection_rows, projection_rows
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if loss_chunk_tokens is None:
        loss_chunk_tokens = int(os.environ.get("HIVE_CHUNK_CE", 0))
    if loss_chunk_tokens < 0:
        raise ValueError("loss chunk size must be nonnegative (0 uses the stock readout)")

    from team import ternary
    torch.set_num_threads(int(os.environ.get("AGENT_TEAM_THREADS", os.cpu_count() or 4)))
    random.seed(seed)
    torch.manual_seed(seed)
    from team import solo
    rows = [json.loads(x) for x in Path(rows_file).read_text().splitlines() if x.strip()]
    for r in rows:  # rows made before the short orders existed carry the long ones: the taught member is trained under the short
        day = re.search(r"Today is (\d{4}-\d\d-\d\d \(\w+\))", r["messages"][0]["content"])
        r["messages"][0] = {"role": "system", "content": solo.SHORT.format(today=day.group(1) if day else time.strftime("%Y-%m-%d (%A)"))}
    if not foresee:
        rows = [r for r in rows if r.get("type") != "foresee"]
    # what the child's own parent got wrong (and the trunk, the body as one), with the program that was right: the
    # child is taught the correction from where its parent stood (the wrong program, what it was shown, the verdict)
    mend = []
    for r in rows:
        wrong = str(r.get("rejected_who") or "").partition("+")[2]
        if r.get("type") == "pair" and parent_cell and wrong and (wrong == parent_cell or re.fullmatch(r"g\d+trunk", wrong)):
            mend.append({k: v for k, v in r.items() if k not in ("rejected", "rejected_who", "shown")} | {
                "type": "mend", "messages": r["messages"] + [{"role": "assistant", "content": r["rejected"]},
                                                             {"role": "user", "content": MEND.format(shown=r.get("shown") or "It ran.")}]})
    if not pairs:
        rows = [r for r in rows if r.get("type") != "pair"]
    rows += mend
    seen, uniq = set(), []
    for r in rows:  # the same exchange from two sittings teaches nothing twice
        k = json.dumps([r["messages"][1:], r["output"], r.get("rejected")])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    # Lessons kept back (kept_back): chosen from all the rows before they are sliced, so every slice keeps back the
    # same ones; no row of their tasks is trained on. Their rows are measured, not their "done" rows (an output ->
    # DONE is learned at once in every kind, about 0.001, and says nothing).
    back = {t: k for k, ts in kept_back(rows).items() for t in ts}
    back_rows = [r for r in rows if r.get("who") == "lesson" and r.get("task") in back and r.get("type") != "done"]
    rows = [r for r in rows if r.get("task") not in back]
    part = ""
    if shard:
        k, n = (int(x) for x in str(shard).split("/"))
        # What is new this generation is studied by every slice; only what the parent already studied is shared out.
        # Generation 35: 3 lessons of a kind fell one to a slice, each slice met its one row 4 times at the most, and
        # the new hard subjects ended at a loss of 0.4 to 1.6. An average of slices keeps what they all learned.
        fresh_rows = [r for r in rows if (gen and r.get("gen") == gen) or r.get("type") == "mend"]
        rows = fresh_rows + slice_rows([r for r in rows if not ((gen and r.get("gen") == gen) or r.get("type") == "mend")], k, n)
        part = f".s{k}"
        log(f"slice {k + 1} of {n}: {len(rows)} rows, {len(fresh_rows)} of them new this generation and studied by every slice "
            f"(the other runners study the rest of the old rows; the slices are averaged into the child)", flush=True)
    src = snapshot_download(UNPACKED[base])
    tok = AutoTokenizer.from_pretrained(src)
    import contextlib
    # bf16 halves the memory and is fast where the CPU has instructions for it (AVX512-BF16); elsewhere 32-bit floats
    # may be the faster way (measured by `speed`)
    half = dtype != "f32"
    if half and int8_base:
        # On a CPU without the 16-bit instructions (about one free runner in four) PyTorch's 16-bit arithmetic is 10 to
        # 16 times slower than its 32-bit, and a cell's first row took hours (three of the first twelve 8B cells were
        # lost that way, with nothing in their logs). There the whole cell is trained in 32-bit: rows have to be
        # about half as long to fit, and it is several times faster than crawling.
        from training import ternary_base
        if not ternary_base.quick16():
            half = False
            log("this CPU has no quick 16-bit arithmetic (no AVX512-BF16, no AMX): the cell is trained in 32-bit numbers", flush=True)
    # a GPU where there is one (a Colab notebook, training/colab.py): the DNA in 16-bit numbers on it, the same LoRA
    # in the same layout as on a CPU, so a cell grown there is a cell like any other
    gpu = torch.cuda.is_available() and not int8_base and os.environ.get("HIVE_GPU", "1") != "0"
    dev = torch.device("cuda" if gpu else "cpu")
    cast = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if half else contextlib.nullcontext
    from transformers import AutoConfig
    nl = AutoConfig.from_pretrained(src).num_hidden_layers
    which = (list(range(nl)) if layers == "all" else list(range(nl // 2, nl)) if layers == "top" else
             list(range(nl - int(layers[4:]), nl)) if layers.startswith("last") else list(range(*map(int, layers.split("-")))))
    row_checkpoint_requested = not checkpoint and not gpu and os.environ.get("HIVE_ROW_CHECKPOINT", "0") == "1"
    limits = _cgroup_limits() if row_checkpoint_requested else ()
    cell = None
    if int8_base:
        # the frozen DNA held as what it is, ternary: one byte a weight, so a 4B or 8B DNA trains on a free runner
        # (training/ternary_base.py); the edit is the same LoRA, saved in PEFT's layout
        if ternary_edit:
            raise ValueError("--ternary is not made for --int8-base")
        from training import ternary_base
        model, cell = ternary_base.load(src, rank, 2 * rank, 0.0, which, torch.bfloat16 if half else torch.float32, log=log,
                                        release_source_cache=bool(limits))
        # how long a row fits beside the DNA on this machine: worked out below, once the optimizer exists
    else:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16 if half else torch.float32).to(dev)
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, layers_to_transform=which, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if ternary_edit:
        # The cell's own edit in three values as well: each of its matrices is held as ordinary numbers while it is
        # trained, and what the net computes with is their ternary form (-s, 0, +s; s the mean size), the way
        # BitNet b1.58 trains. What is saved is the ternary form, so the whole cell is a BitNet.
        from torch.nn.utils import parametrize

        class Three(torch.nn.Module):
            def forward(self, w):
                s = w.detach().abs().mean().clamp(min=1e-8)
                return w + ((w / s).round().clamp(-1, 1) * s - w).detach()
        edits = [m for n, m in model.named_modules() if n.endswith(("lora_A.default", "lora_B.default"))]
        for m in edits:
            parametrize.register_parametrization(m, "weight", Three())
    model.config.use_cache = False
    start = parent if parent and Path(parent).exists() else ""
    if selected or (trunk and Path(trunk).exists()):
        # the body's shared edit pulls every line toward it: the child starts as its parent moved `pull` of the way to
        # the trunk (with no parent: as the trunk), cut exactly to the child's rank (training/merge.py)
        from training import merge
        if selected:
            mixed, kept = merge.birth_mix(start, trunk, selected, pull, donor_pull, rank)
            kept_birth_energy = kept
            log(f"birth donor {selected.name}: parent {1 - (pull + donor_pull):g}, trunk {pull:g}, donor {donor_pull:g}; least_energy_kept={kept:.9f}", flush=True)
        else:
            mixed, kept = merge.merge(([merge.load(start)] if start else []) + [merge.load(trunk)], [1 - pull, pull] if start else [1.0], rank)
        GROW.mkdir(parents=True, exist_ok=True)
        start = str(GROW / f"{base}+{name}.start.safetensors")
        merge.save(mixed, start)
        if not selected:
            log(f"the child starts as its parent pulled {pull:g} of the way to the trunk ({Path(trunk).name}); at least {kept:.3f} of every matrix is kept at rank {rank}", flush=True)
        else:
            del mixed
    if start:
        # the child starts as its parent: every weight of the parent's edit is copied in before training, so what
        # the parent learned (its corrections included) is the child's from the start, and training moves it on
        from safetensors.torch import load_file
        sd = load_file(start)
        got = 0
        if cell:
            for n, m in cell.layers.items():
                a, b = sd.get(f"base_model.model.{n}.lora_A.weight"), sd.get(f"base_model.model.{n}.lora_B.weight")
                if a is not None and b is not None and a.shape == m.lora_A.shape and b.shape == m.lora_B.shape:
                    m.lora_A.data.copy_(a)
                    m.lora_B.data.copy_(b)
                    got += 1
        else:
            for n, p_ in model.named_parameters():
                k = saved_key(n)
                if p_.requires_grad and k in sd and sd[k].shape == p_.shape:
                    p_.data.copy_(sd[k].to(p_.dtype))
                    got += 1
        log(f"the child starts as its parent ({Path(parent).name if parent else Path(trunk).name}): {got} of its matrices inherited", flush=True)
        if not got:
            raise RuntimeError(f"the child was to start from {Path(start).name}, and none of its matrices fit this edit (rank {rank}, layers {which[0]}-{which[-1]})")
        if selected:
            del sd
    # Each layer's inner values are worked out again on the way back instead of being kept: a third slower, and the
    # memory no longer grows with the length of a row (the first run on a 16 GB runner was killed at step 5).
    if checkpoint:
        model.gradient_checkpointing_enable()
    row_checkpoints = None
    if row_checkpoint_requested:
        try:
            row_checkpoints = _RowCheckpoints(model, which[0])
        except ValueError as exc:
            log(f"row checkpointing unavailable: {exc}; using fixed admission", flush=True)
            limits = ()
    params = cell.params if cell else [p for p in model.parameters() if p.requires_grad]
    siblings = _sibling_factors(sibling, model, cell) if sibling and sibling_cis else []
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    # Measured on the 8B (2026-11-15): 8.1 GB held, and every piece of a row takes about 5 MB more while it is trained
    # on (46 values of the hidden size for each trained layer). A row that does not fit sends the machine into its swap.
    # Gradients travel through every layer above the first adapter, including frozen layers above a range.
    suffix_layers = nl - which[0] if batch_size > 1 else len(which)
    per_piece = suffix_layers * model.config.hidden_size * (2 if half else 4) * 46
    checkpoint_floor = None
    eval_max_len = max_len
    if row_checkpoints is not None:
        suffix = len(row_checkpoints.layers)
        hidden_bytes = model.config.hidden_size * (2 if half else 4)
        per_piece = suffix * 46 * hidden_bytes
        # One active full decoder layer during recomputation, plus an input and residual allowance per suffix layer.
        checkpoint_floor = min(per_piece, (46 + 2 * suffix) * hidden_bytes)
    if int8_base:
        # How long a row fits beside the DNA on this machine, from the memory available now that the DNA, the edit and
        # its optimizer exist (it was the machine's total less what was held at load, which left out the OS and the
        # runner, about 0.8 GB). g38: 7 of 24 slices were lost ("the hosted runner lost communication": out of
        # memory); a 16-bit slice keeps 1 GB more spare (PR 78). Only a first cut: the memory guard in the loop
        # (MemoryGuard) measures what a row really takes.
        import psutil
        avail = _available(psutil, limits) / 2**30
        spare = 1.7 + (1.0 if half else 0.0)  # from `available`, which already leaves out the OS (about 0.8 GB): fits as before; the measuring guard is the safety
        fits = int((avail - spare) * 2**30 / per_piece)
        eval_max_len = min(max_len, max(256, fits))
        if eval_max_len < max_len:
            log(f"this machine has {avail:.1f} GB available beside the DNA: fixed cap {eval_max_len} pieces (asked for {max_len})", flush=True)
            if row_checkpoints is None:
                max_len = eval_max_len
            else:
                log(f"training candidates up to {max_len} pieces receive per-row admission; held-back evaluation keeps the fixed cap", flush=True)

    def encode(row, cap):
        p = tok(tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
        t = tok(row["output"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        if len(p) + len(t) > cap and not row.get("rejected"):
            # A row that does not fit loses the middle of an earlier program it shows (its context, not what is taught):
            # g28's rows lost 38 of 767 this way, 23 of the 73 rows of modules, almost all "fix" and "done" rows that
            # carry a whole earlier program before the answer. What is taught (the answer) is never cut.
            for keep in (0.6, 0.35, 0.15):
                short = shorten(row["messages"], keep)
                if short is row["messages"]:
                    break
                p = tok(tok.apply_chat_template(short, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
                if len(p) + len(t) <= cap:
                    break
        if len(p) + len(t) > cap:
            return None
        if row.get("rejected"):
            # DPO retains both graphs. ORPO releases the chosen graph before scoring the rejection.
            w = tok(row["rejected"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
            return None if pair_pieces(len(p) + len(t), len(p) + len(w), orpo) > cap else (torch.tensor([p + t]), len(t), torch.tensor([p + w]), len(w))
        return (torch.tensor([p + t]), len(t))
    # Each row once, sorted into its subject (Study): what is trained on next is picked by where the error is, no
    # longer by copies of a row. The cell's own organ weighs `weight` (3) times in that pick: every cell learns all
    # of the hive's checked work (so any two can check each other), each one its own part best. This generation's
    # rows and its parent's corrected mistakes (new to the whole line) are subjects of their own, so the first sweep
    # meets them, and they keep passes for as long as their loss is high. `fresh` is no longer used.
    # rows_by_kind: what became of this generation's new rows (the Study's `new`), by kind, each row once: given to
    # this slice, left out as too long (encode), never trained on because the memory guard skipped it every time it
    # came up, trained on. What is left (given - too_long - skipped_memory - trained) was not reached before the time
    # was up. g41's read put all three together under "left out" and named T4 for kinds with no row too long.
    data, tags, own_rows, mine, new_rows, long = [], [], {}, 0, 0, 0
    by_kind, new_kind = {}, {}  # {kind: counts}; {index in data: kind} of the new rows
    for row in rows:
        new = (bool(gen) and row.get("gen") == gen) or row.get("type") == "mend"
        if new:
            counts = by_kind.setdefault(row.get("kind") or "?", {"given": 0, "too_long": 0, "skipped_memory": 0, "trained": 0})
            counts["given"] += 1
        d = encode(row, max_len)
        if d is None:
            long += 1
            if new:
                counts["too_long"] += 1
            continue
        own = bool(lean) and ternary.organ(row["task"]) == lean
        if new:
            new_kind[len(data)] = row.get("kind") or "?"
        data.append(d)
        tags.append(subject(row, new))
        own_rows[tags[-1]] = own_rows.get(tags[-1], 0) + own
        mine += own
        new_rows += new
    if limit:
        data, tags = data[:limit], tags[:limit]
    if lean:
        log(f"this cell leans to {lean}: {mine} of its rows are that organ's, weighed {weight} times in the pick", flush=True)
    if gen or mend:
        log(f"{new_rows} rows are new to this line (generation {gen or '-'}'s own, {len(mend)} of them its parent's mistakes corrected), each a subject of its own", flush=True)
    if long:
        log(f"{long} rows are longer than {max_len} pieces and are left out", flush=True)
    judged_used = 0
    if judged:
        # answers people voted on (training/raw.judged): the best-voted beside the worst-voted for the same question,
        # trained as a right-against-wrong pair, the way the hive's own pairs are
        from training import raw
        for row in raw.judged(judged, name + part, lean, log=log):
            d = encode(row, max_len)
            if d is not None:
                data.append(d)
                tags.append("judged")
                judged_used += 1
        log(f"judged: {judged_used} pairs of answers people voted on", flush=True)
    read_used = 0
    if read:
        # raw text and code from the open web (training/raw.py): three times as many drawn as are read, each measured
        # by how well the DNA already predicts it, and the middle band kept: not known yet, but learnable
        from training import raw
        drawn = raw.draw(3 * read, name + part, lean, log=log)
        scored = []
        model.eval()
        read_until = measure_until(time.time(), minutes, 0.12)
        for row in drawn:
            if read_until and time.time() > read_until and len(scored) >= 8:
                log(f"read: the time for measuring raw pieces is up after {len(scored)} of {len(drawn)}", flush=True)
                break
            # every piece of a raw text carries the loss, not only an answer's: each one keeps the scores of the whole
            # vocabulary (152 thousand words on the 8B) through the step back. 512 pieces filled a 16 GB machine
            # (2026-11-15), so a raw piece is cut to READ_LEN.
            ids = tok(row["text"], add_special_tokens=False)["input_ids"][:READ_LEN]
            if len(ids) < 64:
                continue
            x = torch.tensor([ids])
            with torch.no_grad(), cast():
                loss_r = (answer_nll(model, x.to(dev), len(ids) - 1, loss_chunk_tokens) / (len(ids) - 1)).item()
            scored.append((loss_r, (x, len(ids) - 1)))
        model.train()
        chosen = raw.pick(scored, read)
        data += chosen
        tags += ["read"] * len(chosen)
        read_used = len(chosen)
        if scored:
            ls = sorted(l for l, _ in scored)
            log(f"read: {len(drawn)} raw pieces drawn, {len(scored)} measured (the DNA's error from {ls[0]:.2f} to {ls[-1]:.2f}), "
                f"{read_used} kept from the middle band", flush=True)
    random.seed(f"{seed}:{lean}:{name}")  # two children of one organ in a generation see their rows in their own orders
    torch.manual_seed(abs(hash(lean)) % 10**6 + seed)
    log(f"{len(rows)} rows, {len(data)} within {max_len} pieces (mean {sum(d[0].shape[1] for d in data) // max(1, len(data))}); "
        f"LoRA rank {rank} on layers {which[0]}-{which[-1]} of {UNPACKED[base]}", flush=True)
    # What machine this is, and a watch over every row. Three of the first twelve 8B cells never got past their
    # first row (hours, until the job's time was up) and their logs said nothing. Now the log says which CPU this
    # is, a row that takes minutes is reported with the machine's memory, and a row that takes longer than
    # HIVE_ROW_SECONDS (25 minutes) ends the job at once, so the generation goes on with the other children.
    try:
        cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    except OSError:
        cpu = "?"
    if int8_base:
        from training import ternary_base
        log(f"this machine: {cpu}, {torch.get_num_threads()} threads; trained in {'16' if half else '32'}-bit numbers", flush=True)
    elif gpu:
        log(f"this machine: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB), trained on the GPU", flush=True)
    else:
        log(f"this machine: {cpu}, {torch.get_num_threads()} threads", flush=True)
    beat = {"t": time.time(), "what": "its first row", "on": True}

    def watch():
        import faulthandler
        import psutil
        while beat["on"]:
            time.sleep(60)
            idle = time.time() - beat["t"]
            if beat["on"] and idle > 240:
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                log(f"still in {beat['what']} after {idle:.0f} s; memory free {vm.available / 2**30:.1f} GB, swap used {sw.used / 2**30:.1f} GB, load {os.getloadavg()[0]:.1f}", flush=True)
            most = row_limit(time.time(), born, minutes, float(os.environ.get("HIVE_ROW_SECONDS", 1500)))
            if beat["on"] and idle > most:
                log(f"one row has taken more than {most:.0f} s: this machine cannot train this cell. The job ends here so that the generation goes on without it.", flush=True)
                faulthandler.dump_traceback(all_threads=True)
                os._exit(3)
    import threading
    threading.Thread(target=watch, daemon=True).start()
    # The passes are not planned ahead from one timed row any more: the loop runs until the time is up, or until
    # every row has been met as often as the Study lets it (MOST times, HARD_MOST in a new hard subject not learned
    # yet), or (with no time given, as in tests) for len(data) x epochs passes; which row comes next is the Study's
    # pick, and the learning rate follows the share of the time used.
    by = {}
    for j, s in enumerate(tags):
        by.setdefault(s, []).append(j)
    own_share = {s: min(1.0, own_rows[s] / len(idx)) for s, idx in by.items() if own_rows.get(s)}
    hard, web = new_hard(by)
    study = Study(by, own_share, rng=random.Random(f"{seed}:{lean}:{name}{part}"), weight=weight, hard=hard, web=web)
    passes = study.size if minutes else min(int(len(data) * epochs), study.size)
    log(f"{len(data)} rows in {len(study.rows)} subjects; each pass picks a subject by its loss (own organ up to x{weight:g} by its share, read and judged "
        f"{Study.SHARE:.0%} each), a row met at most {study.most} times", flush=True)
    if study.hard:
        log(f"{len(study.hard)} new hard subjects ({len(study.web)} of them web kinds, met first) are studied until their loss is under {Study.GOAL}: "
            f"{Study.FIRST:.0%} of the passes, a row met up to {study.hard_most} times", flush=True)
    # the lessons kept back, measured where the child starts (its parent pulled toward the trunk) and again at the end;
    # the time the first measure took is kept for the second, out of the cell's own time
    kept_items, kept_long = [], 0
    for r in back_rows:
        d = encode(r, eval_max_len)
        if d is None:
            kept_long += 1
        else:
            kept_items.append((back[r["task"]], d[:2]))
    # Fixed checkpointing recomputes the full model. Per-row checkpointing only recomputes the suffix.
    can_cache = which[0] > 0 and not checkpoint and not gpu and os.environ.get("HIVE_PREFIX_CACHE", "1") != "0"
    if batch_size > 1 and getattr(model.config, "attention_dropout", 0):
        can_cache = False  # A frozen prefix with active attention dropout has no reusable boundary.
    prefix = None
    if can_cache and os.environ.get("HIVE_UNLOAD_PREFIX", "0") == "1":
        # The right and rejected sequences of a pair have separate boundaries, also used by the reference policy.
        sequences = [ids for row in data for ids in row[::2]] + [ids for _, (ids, _) in kept_items]
        now = time.time()
        # A partial pre-pass can fall back to the full model while there is still time to train and measure it.
        deadline = now + max(0.0, born + minutes * 60 - now) / 2 if minutes else 0.0
        prefix = _Prefix(model, which[0], log, rows=sequences, cast=cast, beat=beat, until=deadline)
    t_kept = time.time()
    until = precheck_deadline(born, t_kept, minutes, os.environ.get("TRAIN_PROFILE", "auto"))
    cap = measure_until(t_kept, minutes, 0.12)  # the first measure costs the second as much: 12% each, not 35 min each (g59)
    until = min(until, cap) if until and cap else (until or cap)
    kept_before = kept_loss(model, kept_items, dev, cast, beat, until=until,
                            chunk_tokens=loss_chunk_tokens) if kept_items else {}
    # kept_loss stops between rows, so the sampled items are exactly this prefix in the same order.
    kept_items = kept_items[:sum(len(values) for values in kept_before.values())]
    spare = time.time() - t_kept
    if back:
        log(f"lessons kept back: {len(back)}, of {len(set(back.values()))} kinds, never trained on; {len(kept_items)} of their rows measured "
            f"in {spare:.0f} s{f', {kept_long} too long' if kept_long else ''}", flush=True)
    try:
        import ctypes
        trim = ctypes.CDLL("libc.so.6").malloc_trim
    except OSError:
        trim = None
    model.train()
    if can_cache and prefix is None:
        prefix = _Prefix(model, which[0], log, folder=GROW / f"{base}+{name}.prefix" if int8_base else None)
    t0, running, step, hist, paired, i, held = time.time(), 0.0, 0, [], 0, 0, 0
    # the memory guard (MemoryGuard): a row that would leave less than HIVE_MEMORY_SPARE_GB (1) available is skipped
    # for this pass; the cost of a piece is learned from the run
    import gc
    guard = None if gpu else MemoryGuard(per_piece, checkpoint_floor=checkpoint_floor,
                                         limits=limits or (_cgroup_limits() if batch_size > 1 else ()))
    if guard:
        log(f"memory guard: {guard.available() / 2**30:.1f} GB available, a row skipped when it would leave less than {guard.spare / 2**30:.1f} GB "
            f"(a piece taken as {per_piece / 2**20:.2f} MB to start, then as measured)", flush=True)
    in_a_row, skipped_rows, trained_rows = 0, set(), set()
    sft_batches = {}
    budget = max(1.0, minutes * 60 - (t0 - born) - spare) if minutes else 0

    def done_share():
        return max(i / max(1, passes), (time.time() - t0) / budget if minutes else 0.0)

    def update():
        nonlocal running, step, held
        if siblings:
            # Each row's objective was divided by accum, including in a partial final window.
            _add_sibling_gradient(siblings, sibling_cis * held / accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        p = done_share()
        for g in opt.param_groups:
            g["lr"] = lr * lr_share(p)
        opt.step()
        opt.zero_grad()
        if trim:
            trim(0)
        step += 1
        hist.append(round(running / held, 4))
        running, held = 0.0, 0
        if step % 5 == 0:
            import resource
            on_gpu = f", on the GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if gpu else ""
            log(f"step {step} ({p:.0%} done, {sum(1 for q in study.queue.values() if q)} subjects left) loss {hist[-1]:.4f} "
                f"{time.time() - t0:.0f}s, most memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB{on_gpu}"
                + (f", available {guard.available() / 2**30:.1f} GB, swap used {guard.swap() / 2**30:.1f} GB, a piece {guard.cost / 2**20:.2f} MB, "
                   f"{sum(guard.skips.values())} rows skipped" if guard else ""), flush=True)
    plateau = None
    if plateau_mode != "off":
        from training.shard_budget import Plateau
        plateau = Plateau(plateau_mode, len(data))
    stop_for_plateau = False
    def projected(target):
        return projection_rows(model, target, loss_chunk_tokens)

    def batch_head_bytes(lengths, targets):
        return batch_projection_rows(model, lengths, targets, loss_chunk_tokens) * model.config.vocab_size * 16

    while i < passes:
        if minutes and time.time() - born > minutes * 60 - spare:
            # Two trials were lost on 2026-11-14 because their jobs ran into the time limit with nothing saved: the
            # cell stops here and is kept.
            log(f"time is up after {i} passes: the cell is saved as it is", flush=True)
            break
        got = study.pick()
        if got is None:
            log(f"every row has been met as often as it may be ({study.most} times, up to {study.hard_most} for a new hard subject not learned yet), after {i} passes", flush=True)
            break
        s, idx = got
        ids, k = data[idx][:2]
        pair = len(data[idx]) == 4
        pieces = pair_pieces(ids.shape[1], data[idx][2].shape[1], orpo) if pair else ids.shape[1]
        picks = [got]
        batch_limit = min(batch_size, accum - held, passes - i)
        if batch_limit > 1 and not pair:
            picks, pieces, mode = _sft_batch(study, got, data, batch_limit, guard, batch_head_bytes)
        if len(picks) == 1:
            mode = "ordinary"
            if row_checkpoints is not None:
                head = row_head_bytes(model.config.vocab_size, k, data[idx][3] if pair else 0, projected, orpo=orpo)
                mode = guard.row_mode(pieces, s, head)
            elif guard and not guard.admit(pieces, s):
                mode = None
        if mode is None:
            study.skip(s, idx)
            skipped_rows.add(idx)
            in_a_row += 1
            if in_a_row == 1:
                gc.collect()
                if trim:
                    trim(0)
            if in_a_row >= 2 * len(study.queue) + 20:
                log(f"no row left fits in the memory available ({guard.available() / 2**30:.1f} GB) after {in_a_row} tries: the cell is saved as it is", flush=True)
                break
            continue
        in_a_row = 0
        if guard:
            guard.begin()
        if row_checkpoints is not None:
            row_checkpoints.set(mode == "checkpoint")
        row_prefix = prefix if mode == "ordinary" else None
        if len(picks) > 1:
            batch_rows = [data[j] for _, j in picks]
            ids, mask, labels = _padded_answers(batch_rows, tok.pad_token_id or 0)
            ids = ids.to(dev)
            beat["t"], beat["what"] = time.time(), f"passes {i + 1}–{i + len(picks)} ({pieces} padded pieces)"
            # Unloaded weights always require cached boundary states, even when checkpointing the suffix.
            cached = prefix if prefix and (mode == "ordinary" or prefix.unloaded) else None
            with cast(), (cached.use_batch(ids, [x for x, _ in batch_rows]) if cached else contextlib.nullcontext()):
                sums = answer_nll(model, ids, chunk_tokens=loss_chunk_tokens, attention_mask=mask.to(dev), labels=labels.to(dev))
                means = sums / sums.new_tensor([k for _, k in batch_rows])
            (means.sum() / accum).backward()
            losses = means.detach().tolist()
            objectives = losses
            del means, sums
        else:
            ids = ids.to(dev)
            beat["t"], beat["what"] = time.time(), f"pass {i + 1} ({s}, {ids.shape[1]} pieces)"
            def score(x, n):
                with cast(), (row_prefix.use(x) if row_prefix else contextlib.nullcontext()):
                    return answer_nll(model, x, n, loss_chunk_tokens)

            if pair:
                wid, wk = data[idx][2:]
                wid = wid.to(dev)
                paired += 1
            if pair and orpo:
                plain, row_objective = _orpo_backward(score, ids, k, wid, wk, params, accum)
            else:
                right_nll = score(ids, k)
                loss = right_nll / (k * ids.shape[0])
                plain = loss.item()  # The study tracks the error on the chosen answer under either objective.
                if pair:
                    good, bad = -right_nll, -score(wid, wk)
                    with torch.no_grad(), (cell.off() if cell else model.disable_adapter()):
                        ref_good, ref_bad = -score(ids, k), -score(wid, wk)
                    margin = beta * ((good - ref_good) - (bad - ref_bad))
                    loss = loss - torch.nn.functional.logsigmoid(margin)
                    del good, bad, ref_good, ref_bad, margin
                (loss / accum).backward()
                row_objective = loss.item()
                del right_nll, loss
            losses, objectives = [plain], [row_objective]
        for (s, idx), plain, row_objective in zip(picks, losses, objectives, strict=True):
            running += row_objective
            study.learned(s, plain)
            trained_rows.add(idx)
            if plateau:
                plateau.observe(idx, row_objective)
        held += len(picks)
        i += len(picks)
        if batch_size > 1 and not pair:
            sft_batches[len(picks)] = sft_batches.get(len(picks), 0) + 1
        if held == accum:
            update()
            if plateau:
                stop_for_plateau = plateau.after_update(len(trained_rows))
        # what the pass made is let go before the next one (the old logits lived through the next forward), and the
        # memory handed back, so that what is available is true when the next row is weighed
        if pair:
            del wid
        if i // 4 != (i - len(picks)) // 4:
            gc.collect()
        if trim:
            trim(0)
        if guard:
            guard.end(pieces, mode)
        if stop_for_plateau:
            log(f"matched training-row losses plateaued after {step} optimizer updates; saving the cell", flush=True)
            break
    if held:
        update()
    if row_checkpoints is not None:
        row_checkpoints.set(False)
    if guard:
        guard.stop()
        mem = guard.record()
        if row_checkpoints is not None:
            mem.update(training_max_len=max_len, evaluation_max_len=eval_max_len)
        if mem["rows_skipped"]:
            log(f"memory guard: {mem['rows_skipped']} passes skipped a row that would not fit ("
                + ", ".join(f"{k} {v}" for k, v in list(mem["skipped_by_subject"].items())[:8]) + ")", flush=True)
        log(f"memory: most {mem['peak_gb']} GB, least available {mem['min_available_gb']} GB, most swap {mem['swap_peak_gb']} GB, "
            f"a piece {mem['cost_per_piece_mb']} MB (the formula's {mem['formula_per_piece_mb']})", flush=True)
    if minutes and step == 0:
        beat["on"] = False
        if prefix:
            prefix.done()
        raise RuntimeError("the slice completed zero optimizer updates; no trained adapter was produced")
    for j, k in new_kind.items():
        if j in trained_rows:
            by_kind[k]["trained"] += 1
        elif j in skipped_rows:
            by_kind[k]["skipped_memory"] += 1
    subjects = study.table()
    width = max([len(x["subject"]) for x in subjects] + [7])
    log(f"{'subject':<{width}} rows passes  first   last", flush=True)
    for x in subjects:
        f_, l_ = x["loss_first"], x["loss_last"]
        log(f"{x['subject']:<{width}} {x['rows']:>4} {x['passes']:>6} {'-' if f_ is None else f'{f_:.3f}':>6} {'-' if l_ is None else f'{l_:.3f}':>6}", flush=True)
    # the second measure is held to the time the first took (spare) past the budget, and 5 minutes more at most
    kept_after = kept_loss(model, kept_items, dev, cast, beat, until=(born + minutes * 60 + 300) if minutes else 0.0,
                           chunk_tokens=loss_chunk_tokens) if kept_items else {}
    if prefix:
        prefix.done()  # Free the disk space before writing the adapter.
    kept = paired_kept_losses(back, kept_before, kept_after)
    if kept:
        log(f"{'lessons kept back (never trained on), by kind':<{width}} rows before  after", flush=True)
        for k, v in kept.items():
            b_, a_ = v["before"], v["after"]
            log(f"{k:<{width}} {v['rows']:>4} {'-' if b_ is None else f'{b_:.3f}':>6} {'-' if a_ is None else f'{a_:.3f}':>6}", flush=True)
    beat["on"] = False
    work = GROW / f"{base}+{name}"
    work.mkdir(parents=True, exist_ok=True)
    if ternary_edit:
        for m in edits:
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        levels = max(len(torch.unique(m.weight.detach().float())) for m in edits)
        log(f"the cell's edit is ternary: at most {levels} different values in any of its {len(edits)} matrices", flush=True)
    if cell:
        cell.save(work, src)
    else:
        model.save_pretrained(str(work))
    if prefix:
        if prefix.unloaded:
            log(f"frozen prefix read from disk for {prefix.hits} suffix passes", flush=True)
        elif batch_size > 1:
            log(f"prefix cache: {prefix.hits} row hits and {prefix.misses} misses ({prefix.bytes / 2**30:.2f} GB)", flush=True)
        else:
            log(f"frozen layers skipped on {prefix.hits} of {prefix.hits + prefix.misses} passes (their output kept from the first pass of each row, {prefix.bytes / 2**30:.2f} GB{' on disk' if prefix.folder else ''})", flush=True)
    meta = {"member": f"{base}+{name}", "leans": lean, "dtype": dtype, "loss_chunk_tokens": loss_chunk_tokens, "checkpoint": bool(checkpoint), "seconds_a_row": round((time.time() - t0) / max(1, i), 1), "ternary_edit": bool(ternary_edit), "foresee": bool(foresee), "pairs": paired, "int8_base": bool(int8_base), "read": read_used, "judged": judged_used, "parent": Path(parent).name if parent and Path(parent).exists() else "", "parent_cell": parent_cell, "trunk": Path(trunk).name if trunk and Path(trunk).exists() else "", "mended": len(mend), "new_rows": new_rows, "shard": str(shard or ""), "base": UNPACKED[base], "rows": len(rows), "used": len(data), "passes": i, "epochs": round(i / max(1, len(data)), 2), "rank": rank, "lr": lr,
            "layers": [which[0], which[-1]], "steps": step, "seconds": round(time.time() - t0), "loss_first": hist[0] if hist else None,
            "loss_last": round(sum(hist[-5:]) / max(1, len(hist[-5:])), 4) if hist else None, "when": time.strftime("%Y-%m-%d %H:%M"),
            "memory_gb": round(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 2**20, 1),
            "gpu": torch.cuda.get_device_name(0) if gpu else "", "subjects": subjects, "kept_back": kept,
            "rows_by_kind": dict(sorted(by_kind.items()))}
    if selected:
        meta.update(donor=selected.name, donor_pull=donor_pull, least_energy_kept=kept_birth_energy)
    if siblings:
        meta.update(sibling=Path(sibling).name, sibling_cis=sibling_cis, sibling_modules=len(siblings))
    if batch_size > 1:
        meta["train_batch"] = batch_size
        meta["sft_batches_by_size"] = {str(width): count for width, count in sorted(sft_batches.items())}
    if orpo:
        meta["pairs_objective"] = "orpo"
    if shard and os.environ.get("HIVE_SHARD_POLICY", "hash") != "hash":
        meta["shard_policy"] = os.environ["HIVE_SHARD_POLICY"]
    if plateau:
        meta["plateau"] = {**plateau.report, "rows_covered": len(trained_rows)}
    if guard:
        meta["memory"] = mem
    # the adapter as the runtime reads it
    conv = ternary.source() / "convert_lora_to_gguf.py"
    where = Path(out_dir or ternary.ADAPTERS)
    where.mkdir(parents=True, exist_ok=True)
    gguf = where / f"{base}+{name}.gguf"
    try:
        done = subprocess.run([sys.executable, str(conv), "--base", src, "--outfile", str(gguf), "--outtype", "f16", str(work)], capture_output=True, text=True,
                              timeout=gguf_seconds(time.time(), born, minutes, float(os.environ.get("HIVE_GGUF_SECONDS", 900))))
    except subprocess.TimeoutExpired:
        raise RuntimeError("the edit could not be written as a GGUF adapter in time (HIVE_GGUF_SECONDS)") from None
    if done.returncode:
        raise RuntimeError("the edit could not be written as a GGUF adapter: " + done.stderr[-800:])
    meta["file"], meta["megabytes"] = str(gguf), round(gguf.stat().st_size / 2**20, 1)
    (where / f"{base}+{name}.json").write_text(json.dumps(meta, indent=1))
    return meta


def fetch(names=(), log=print):
    """The weight edits the roster names (or those given), downloaded from the release into cells/members/."""
    import urllib.request

    from team import ternary
    from training import release_store
    repo = os.environ.get("GITHUB_REPOSITORY", "Arnav123-s/HiveMIND")
    got = []
    if not names and not os.environ.get("HIVE_MEMBERS"):
        try:
            body = json.loads(ternary.BODY.read_text())
            names = [f"{body['dna']}+{c}" for c in body.get("cells", [])]
        except (OSError, ValueError, KeyError):
            names = []
    for who in names or ternary.roster():
        if "+" not in who or ternary.adapter(who).exists():
            continue
        ternary.ADAPTERS.mkdir(parents=True, exist_ok=True)
        name = f"{who.replace('+', '.')}.gguf"
        if release_store.get(name, ternary.adapter(who), repo, fetch=urllib.request.urlretrieve):
            got.append(who)
        else:
            log(f"{who}: not fetched: {name} is in none of the hive's releases")
    return got


VOCAB_CE_TOKENS = 64  # --vocab-ce: answers longer than this many tokens take PR 167's vocab-outer CE


def speed_flags(argv, environ=None):
    """The CPU speed switches a versus trial can pass, all off unless given: --packed-dna (PR 171's 2-bit packed DNA,
    HIVE_TERNARY_VNNI=1), --train-batch N (PR 181's batched SFT rows, HIVE_TRAIN_BATCH=N, 1, 2 or 4) and --vocab-ce
    (PR 167's vocab-outer CE). The first two are read by the trainer from the environment as it loads, so they are
    set there; --vocab-ce returns the answer-token threshold for train(loss_chunk_tokens), else None."""
    environ = os.environ if environ is None else environ
    if "--packed-dna" in argv:
        environ["HIVE_TERNARY_VNNI"] = "1"
    if "--train-batch" in argv:
        n = argv[argv.index("--train-batch") + 1] if argv.index("--train-batch") + 1 < len(argv) else ""
        if n not in ("1", "2", "4"):
            raise ValueError(f"--train-batch takes 1, 2 or 4, not {n!r}")
        environ["HIVE_TRAIN_BATCH"] = n
    return VOCAB_CE_TOKENS if "--vocab-ce" in argv else None


def main(argv):
    opt = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d  # noqa: E731
    cmd = argv[0] if argv else ""
    if cmd == "dream":
        print(json.dumps(dream(opt("--seed", int(time.time() // 60)), opt("--tasks", 4), opt("--failures", ""), opt("--out", ""), minutes=opt("--minutes", 0), tries=opt("--tries", 2))))
    elif cmd == "practice":
        print(json.dumps(practice(opt("--seed", 10_000_000 + int(time.time() // 60)), opt("--tasks", 16), opt("--level", "mixed"), opt("--out", ""), minutes=opt("--minutes", 0))))
    elif cmd == "gather":
        # python training/grow.py gather --gen g25 --out out/rows.jsonl --old old.jsonl [--old more.jsonl] new1.jsonl new2.jsonl ...
        olds = [argv[i + 1] for i, a in enumerate(argv) if a == "--old" and i + 1 < len(argv)]
        skip = {i + 1 for i, a in enumerate(argv) if a in ("--old", "--gen", "--out")}
        news = [a for i, a in enumerate(argv[1:], 1) if not a.startswith("--") and i not in skip]
        before, added = gather(opt("--out", str(GROW / "rows.jsonl")), opt("--gen", ""), olds, news)
        print(f"rows: {before} from earlier generations, {added} new in generation {opt('--gen', '')}")
    elif cmd == "rows":
        out = opt("--out", str(GROW / "rows.jsonl"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        print(rows_from([a for a in argv[1:] if not a.startswith("--") and a != out], out), "rows added to", out)
    elif cmd == "train":
        vocab_ce = speed_flags(argv)
        print(json.dumps(train(opt("--rows", str(GROW / "rows.jsonl")), opt("--name", "gen"), opt("--base", "t17"), opt("--epochs", 2.0), opt("--rank", 16),
                               opt("--lr", 2e-4), opt("--max-len", 1152), opt("--layers", "all"), limit=opt("--limit", 0), minutes=opt("--minutes", 0), lean=opt("--lean", ""), weight=opt("--weight", 3), foresee="--foresee" in argv, pairs="--pairs" in argv, orpo=True if "--orpo" in argv else None, int8_base="--int8-base" in argv, read=opt("--read", 0), judged=opt("--judged", 0), parent=opt("--parent", ""), parent_cell=opt("--parent-cell", ""), trunk=opt("--trunk", ""), pull=opt("--pull", 0.1),
                               donor=opt("--donor", ""), donor_pull=opt("--donor-pull", 0.0) if "--donor-pull" in argv else None, sibling=opt("--sibling", ""), gen=opt("--gen", ""), fresh=opt("--fresh", 2), shard=opt("--shard", ""), loss_chunk_tokens=opt("--loss-chunk-tokens", 0) if "--loss-chunk-tokens" in argv else vocab_ce, ternary_edit="--ternary" in argv, checkpoint="--no-checkpoint" not in argv, dtype=opt("--dtype", "bf16")), indent=1))
    elif cmd == "speed":
        # how long one row takes on this machine, each way of training: so that the fastest is used and not guessed
        rows, out = opt("--rows", str(GROW / "rows.jsonl")), []
        for dtype, layers, ck in (("bf16", "all", True), ("bf16", "all", False), ("bf16", "last14", False), ("f32", "all", False), ("f32", "last14", False)):
            p = subprocess.run([sys.executable, __file__, "train", "--rows", rows, "--name", "speed", "--limit", "6", "--epochs", "1", "--dtype", dtype,
                                "--layers", layers, "--base", opt("--base", "t17")] + ([] if ck else ["--no-checkpoint"]), capture_output=True, text=True)
            try:
                m = json.loads(p.stdout[p.stdout.rindex("\n{"):])
                out.append({"dtype": dtype, "layers": layers, "checkpoint": ck, "seconds_a_row": m["seconds_a_row"], "memory_gb": m["memory_gb"], "loss_last": m["loss_last"]})
            except (ValueError, KeyError):
                out.append({"dtype": dtype, "layers": layers, "checkpoint": ck, "failed": (p.stderr or p.stdout)[-200:]})
            print(json.dumps(out[-1]), flush=True)
        for f in (ROOT / "cells" / "members").glob("*+speed.*"):
            f.unlink()
    elif cmd == "fetch":
        print(fetch([a for a in argv[1:] if not a.startswith("--")]))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
