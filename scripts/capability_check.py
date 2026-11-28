"""Targeted capability check, before g82: can the g81 body walk a site over several pages?

Background (docs/AGENTS.md, 2026-11-17 05:19): at g38's gate one cell, g38code2, was right on all 5 tasks of "walk a
site over several pages" while the trunk, g38code and the body were each 0 of 5. g38's gate tasks and g38code2 itself
are NOT in the recovery archive (no entry in records/archive-map.tsv.gz), so the check uses the active suite's
equivalent: 5 tasks of the same kind (training/kinds/h_walk.py, books.toscrape.com, checked live) drawn from fixed
seeds, so every arm and every re-run sits the very same 5.

    python3 scripts/capability_check.py show [--set walk|web]    the 5 tasks of a set (walk: h_walk, drawn offline;
                                                                 web: h_web, needs internet even to draw: Wikidata picks)
    python3 scripts/capability_check.py arms                     the arms that would sit them, from cells/body.json
    python3 scripts/capability_check.py sit [--set walk|web] [--arms a,b] [--seconds 900] [--out runs/capcheck/summary-<set>.json]
                                                                 sit them (needs the ternary runtime and the body's
                                                                 cells installed, and internet for the site)

Arms:
    hive      the whole body, as the archived gate seated it (HIVE_MEMBERS from training/lineage.py members,
              AGENT_TEAM_SOLO=2, AGENT_TEAM_SOLO_ALONE=1: grow.yml set both for every job)
    trunk     g81trunk alone (AGENT_TEAM_SOLO=1, AGENT_TEAM_SOLO_ALONE=1)
    <cell>    each code cell of the body alone, the same way (g81code, g81code2, g80code, g80code2, g39code)

Reading the result (written to --out; the verdict line is printed last):
    PASS       the hive arm solves 5 of 5: the g38code2 question is closed.
    ROUTING    some arm, or some member inside the hive arm, solves a task the hive does not: the skill is in the body
               and the routing loses it. Follow-on: debug the seating and agreement, and look for g38code2's lessons.
               Do NOT roll the body back.
    ABSENT     nothing solves them: the skill is not in the body. Follow-on: re-teach the kind (the h_walk lesson
               measures 534 pieces against lessons.LIMIT 508 in tests/test_lessons.py, which may leave it out of
               training). Do NOT roll the body back.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# fixed seeds: the same tasks in every arm and on every re-run
SETS = {"walk": ("h_walk", [f"capcheck-h_walk-{i}" for i in range(5)]),   # Neon: the g38 site-walking question
        "web": ("h_web", [f"capcheck-h_web-{i}" for i in range(5)])}      # the verifier seat's audit: independent web answers (the g34 defect)
SEEDS = SETS["walk"][1]
CODE_CELLS = ("g81code", "g81code2", "g80code", "g80code2", "g39code")


def tasks(which="walk"):
    from training import exam, exam_hard
    maker, seeds = SETS[which]
    return [(s, *exam._made(getattr(exam_hard, maker), s, 0)) for s in seeds]


def dumps(results):
    """{arm: how many answers are dump-shaped}: does the body answer a web task with a page dump, as the 7 quarantined
    g34 rows taught? Measured on what was said, whatever the check decided. exam.sit_one keeps the arm's own answer;
    members' answers are counted only where a record carries one (each_member keeps verdicts, not answers)."""
    from training.kinds._common import _dumped
    out = {}
    for arm, rs in results.items():
        n = 0
        for r in rs:
            n += bool(_dumped(r.get("answer") or "", 12))
            n += sum(bool(_dumped(m.get("answer") or "", 12)) for m in r.get("members") or [] if m.get("answer"))
        out[arm] = n
    return out


def arms(dna=None):
    body = json.loads((ROOT / "cells" / "body.json").read_text())
    dna = dna or body["dna"]
    cells = set(body.get("cells") or [])
    import subprocess
    roster = subprocess.run([sys.executable, str(ROOT / "training/lineage.py"), "members", dna],
                            capture_output=True, text=True, check=True).stdout.strip()  # the string the workflow exported
    out = {"hive": {"HIVE_MEMBERS": roster, "AGENT_TEAM_SOLO": "2",
                    "AGENT_TEAM_SOLO_ALONE": "1"}}
    if body.get("trunk"):
        out["trunk"] = {"HIVE_MEMBERS": f"{dna}+{body['trunk']}", "AGENT_TEAM_SOLO": "1", "AGENT_TEAM_SOLO_ALONE": "1"}
    for c in CODE_CELLS:
        if c in cells:
            out[c] = {"HIVE_MEMBERS": f"{dna}+{c}", "AGENT_TEAM_SOLO": "1", "AGENT_TEAM_SOLO_ALONE": "1"}
    return out


def verdict(results, n=5):
    """results: {arm: [exam.json dicts]}. PASS / ROUTING / ABSENT, with the evidence."""
    hive = results.get("hive") or []
    hive_ok = {r["id"] for r in hive if r.get("ok")}
    solved_elsewhere = {}
    for arm, rs in results.items():
        for r in rs:
            if arm != "hive" and r.get("ok"):
                solved_elsewhere.setdefault(r["id"], set()).add(arm)
            if arm == "hive":
                for m in r.get("members") or []:
                    if m.get("ok"):
                        solved_elsewhere.setdefault(r["id"], set()).add("member:" + str(m.get("who")))
    lost = {t: sorted(w) for t, w in solved_elsewhere.items() if t not in hive_ok}
    if hive and len(hive_ok) == n:
        v = "PASS"
    elif lost:
        v = "ROUTING"
    elif hive:
        v = "ABSENT"
    else:
        v = "NOT RUN"
    return {"verdict": v, "hive": f"{len(hive_ok)} of {n}" if hive else "not sat",
            "by_arm": {a: f"{sum(bool(r.get('ok')) for r in rs)} of {len(rs)}" for a, rs in results.items()},
            "solved_but_not_by_the_hive": lost}


def sit(which=None, seconds=900, out=None, task_set="walk"):
    from training import exam
    drawn, plan = tasks(task_set), arms()
    which = [a for a in (which or plan) if a in plan]
    results = {}
    home = ROOT / "runs" / "capcheck" / "memory-empty"
    home.mkdir(parents=True, exist_ok=True)
    for arm in which:
        env = {k: v for k, v in plan[arm].items()}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update({k: v for k, v in env.items() if v})
        for k, v in env.items():
            if not v:
                os.environ.pop(k, None)
        try:
            results[arm] = [exam.sit_one(f"capcheck/{task_set}/{arm}", f"cap{i}", task, check, home, timeout=seconds, kind=kind)
                            for i, (_seed, _field, kind, task, check) in enumerate(drawn)]
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        print(f"{arm}: {sum(r['ok'] for r in results[arm])} of {len(drawn)}", flush=True)
    rec = {"tasks": [{"seed": s, "task": t} for s, _f, _k, t, _c in drawn], "arms": {a: plan[a] for a in which},
           "set": task_set, "results": results, **verdict(results, len(drawn)), "dump_shaped_answers": dumps(results)}
    out = Path(out or ROOT / "runs" / "capcheck" / f"summary-{task_set}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(f"capability check ({task_set}): {rec['verdict']} (hive {rec['hive']}); dump-shaped answers {rec['dump_shaped_answers']}; written to {out}")
    return 0


def main(argv):
    cmd = argv[0] if argv else ""
    opts = dict(zip(argv[1::2], argv[2::2]))
    if cmd == "show":
        for s, _f, kind, task, _c in tasks(opts.get("--set", "walk")):
            print(f"{s}\t{kind}\t{task}")
        return 0
    if cmd == "arms":
        print(json.dumps(arms(), indent=1))
        return 0
    if cmd == "sit":
        return sit(opts.get("--arms", "").split(",") if opts.get("--arms") else None, int(opts.get("--seconds", 900)), opts.get("--out"),
                   opts.get("--set", "walk"))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
