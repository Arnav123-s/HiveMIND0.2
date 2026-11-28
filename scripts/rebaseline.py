"""Re-baseline the easy floor from a strict re-sit of g81 (Gemini ruling on m_born / m_licence, 2026-11-28).

    python3 scripts/rebaseline.py --results DIR [--results DIR ...] [--bridge manifests/g81.json] [--manifest manifests/g82.json]

The re-sit: g81's own gate seeds (20000811-20000818, `exam.py sit --level mixed --tasks 8`), the exact g81 body, the
gate's inference settings, under eval_v2 (one answer standard for every web kind). Every web kind's check records
both verdicts on the same answer (training/kinds/_common._v2: "[eval_v1: pass; eval_v2: fail; kind: m_born]"), so one
sitting gives both scores. Other kinds score the same under both.

The ruling's rule (delta = g81's recorded loose easy score, 32, minus the strict re-sit's easy score):
    delta 0     the floor stays 30 of 32
    delta 1-2   the floor becomes 30 - delta, and the easy web tasks that pass eval_v1 but fail eval_v2 become explicit
                recovery targets
    delta 3+    STOP: no floor is written, exit 3. A drop that size is a capability finding for the manager.

Also reported, so the delta can be read: the re-sit's own eval_v1 score. 32 - v1 is drift from the new compute and
the live web (not the checker); v1 - v2 is the checker's own effect. The ruling's delta is the sum of the two.

Writes the dual-score bridge entry into the g81 manifest (--bridge) and, unless it stops, the floor into the next
launch's manifest (--manifest).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import preflight  # noqa: E402

RECORDED_LOOSE = (32, 32)  # g81, registry-verified: easy 32/32 (docs/reads/g81.md)
BASE_FLOOR = 30
TAG = re.compile(r"\[eval_v1: (pass|fail); eval_v2: (pass|fail); kind: (\w*)\]")


def score(results):
    """Per easy task: (id, kind, v1, v2). An untagged task is not a web kind: both verdicts are its own."""
    out = []
    for r in results:
        if not str(r["id"]).startswith("e"):
            continue
        m = TAG.search(str(r.get("why", "")))
        if m:
            out.append((str(r["id"]), m.group(3) or "web", m.group(1) == "pass", m.group(2) == "pass"))
        else:
            out.append((str(r["id"]), "not a web kind", bool(r.get("ok")), bool(r.get("ok"))))
    return out


WEEKDAYS = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)


def exposure(results):
    """Easy tasks of the offline kinds (no eval_v1/v2 tag) whose answer could pass a loose check by enumerating:
    more than 8 numbers ('says': any number counts) or two weekday names or more ('has': any substring counts).
    Measured, not scored: the ruling covers the web kinds only (docs/RESUME-G81.md, the easy-kind sweep). Read on the
    first 400 characters exam.json keeps of the answer, so it is a lower bound."""
    from training.kinds._common import _nums
    out = []
    for r in results:
        if not str(r["id"]).startswith("e") or TAG.search(str(r.get("why", ""))) or not r.get("ok"):
            continue
        a = str(r.get("answer") or "")
        n, w = len(_nums(a)), len({m.lower() for m in WEEKDAYS.findall(a)})
        if n > 8 or w >= 2:
            out.append({"id": str(r["id"]), "numbers": n, "weekdays": w})
    return out


def rebaseline(results):
    rows = score(results)
    n = len(rows)
    v1, v2 = sum(r[2] for r in rows), sum(r[3] for r in rows)
    kinds = defaultdict(lambda: {"tasks": 0, "v1": 0, "v2": 0})
    for _id, kind, a, b in rows:
        kinds[kind]["tasks"] += 1
        kinds[kind]["v1"] += a
        kinds[kind]["v2"] += b
    entry = {"easy_v1_loose": {"passed": RECORDED_LOOSE[0], "of": RECORDED_LOOSE[1], "source": "g81 record (docs/reads/g81.md)"},
             "resit": {"tasks": n, "v1": v1, "v2": v2},
             "easy_v2_strict": {"passed": v2, "of": n, "source": "strict re-sit of g81 on its gate seeds"},
             "per_kind": dict(sorted(kinds.items())),
             "drift_new_compute": RECORDED_LOOSE[0] - v1, "checker_effect": v1 - v2,
             "offline_enumeration_exposure": exposure(results)}
    if n != RECORDED_LOOSE[1]:
        entry.update(decision="INVALID", why=f"the re-sit sat {n} easy tasks; the ruling's delta is defined on the full {RECORDED_LOOSE[1]}")
        return entry, None
    delta = RECORDED_LOOSE[0] - v2
    entry["delta"] = delta
    if delta <= 0:
        floor, targets, decision = BASE_FLOOR, [], "floor stays"
    elif delta <= 2:
        floor = BASE_FLOOR - delta
        targets = [i for i, kind, a, b in rows if a and not b]
        decision = "floor lowered by delta; recovery targets tagged"
    else:
        entry.update(decision="STOP", why=f"delta {delta}: a capability finding, back to the manager before any floor is lowered")
        return entry, None
    entry.update(decision=decision, recovery_targets=targets,
                 eval_v2_floor={"min": floor, "of": RECORDED_LOOSE[1], "basis": f"Gemini ruling: 30 - delta ({delta}); strict re-sit {v2}/{n}"})
    return entry, entry["eval_v2_floor"]


def main(argv):
    dirs = [argv[i + 1] for i, a in enumerate(argv) if a == "--results"]
    if not dirs:
        print(__doc__)
        return 2
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    bridge, manifest = Path(opt("--bridge", ROOT / "manifests" / "g81.json")), Path(opt("--manifest", ROOT / "manifests" / "g82.json"))
    entry, floor = rebaseline(preflight.exam_results(dirs))
    b = json.loads(bridge.read_text()) if bridge.exists() else {"generation": "g81"}
    b["eval_bridge"] = entry
    bridge.write_text(json.dumps(b, indent=1) + "\n")
    print(json.dumps(entry, indent=1))
    if floor is None:
        print(f"rebaseline: {entry['decision']}: {entry.get('why', '')}")
        return 3
    m = json.loads(manifest.read_text())
    m["easy_floor"] = floor
    m["recovery_targets"] = entry["recovery_targets"]
    manifest.write_text(json.dumps(m, indent=1) + "\n")
    print(f"rebaseline: {entry['decision']}; eval_v2 floor {floor['min']} of {floor['of']} written to {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
