"""Pre-flight gate: nothing launches a generation (g82 and after) until this passes on its run manifest.

    python3 scripts/preflight.py MANIFEST.json [--results DIR ...] [--out PREFLIGHT.json]

Three checks (council ruling, Neon, 2026-11-28). Any one that fails stops the launch (exit 1):

  1. EASY FLOOR. The easy slice of the body that the generation grows from, re-sat on this compute, must reach
     manifest["easy_floor"]["min"] of ["of"]. The floor is set against g81's verified baseline (easy 32 of 32).
     A breach launches only with a waiver in the manifest: {"check": "easy_floor", "by", "reason", "at"}. The waiver
     is copied into the pre-flight record, so every waived launch stays on file.
  2. UNIQUE SOLVES. When the manifest drops cells (manifest["override"]: a parent override or a rollback), every
     task in the evidence that ONLY a dropped cell solved is flagged. The launch waits until each flagged task id is
     listed in override["unique_solves_acknowledged"]. (The ruling says "flag". Blocking until acknowledged is the
     stricter reading, chosen so a flag cannot go unseen.)
  3. DISSENT. Every entry of manifest["dissent"] with "blocking": true must carry "acknowledged_by" and
     "acknowledged_at" (its delivery was acknowledged) before the launch.

--results: folders holding exam.json files (training/exam.py sit_one writes one per task: {"id", "ok", "members":
[{"who", "ok"}]}). Easy tasks are those whose id starts with "e". With no --results, the manifest's own
"easy_measured": {"passed", "of", "source"} is used.

A manifest (see docs/RESUME-G81.md for a full one):
    {"generation": "g82", "grows_from": "g81", "easy_floor": {"min": 30, "of": 32, "basis": "..."},
     "waivers": [], "override": null, "dissent": []}
"""
import datetime
import json
import sys
from pathlib import Path


def exam_results(dirs):
    out = []
    for d in dirs:
        for f in sorted(Path(d).rglob("exam.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(r, dict) and r.get("id"):
                r["_file"] = str(f)
                out.append(r)
    return out


def easy_score(results):
    easy = [r for r in results if str(r["id"]).startswith("e")]
    return sum(bool(r.get("ok")) for r in easy), len(easy)


def _cell(who):
    """'t80+g81code' -> 'g81code'; a bare name stays as it is."""
    return str(who).split("+", 1)[-1].strip()


def unique_solves(results, dropped):
    """Tasks some dropped cell solved and no kept member did: [(task id, [dropped cells that solved it])]."""
    dropped = {_cell(c) for c in dropped}
    flagged = []
    for r in results:
        right = {_cell(m.get("who", "")) for m in r.get("members") or [] if m.get("ok")}
        if right and right <= dropped:
            flagged.append((str(r["id"]), sorted(right)))
    return sorted(set((t, tuple(c)) for t, c in flagged))


def check(manifest, results=None):
    """(ok, record). record lists every check, what it saw and why it passed or failed."""
    rec = {"generation": manifest.get("generation"), "grows_from": manifest.get("grows_from"),
           "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "checks": [], "waivers_used": []}
    waivers = {w.get("check"): w for w in manifest.get("waivers") or [] if w.get("by") and w.get("reason")}

    # 1. the easy floor
    floor = manifest.get("easy_floor") or {}
    if results:
        passed, of = easy_score(results)
        source = f"{len(results)} exam.json files"
    else:
        m = manifest.get("easy_measured") or {}
        passed, of, source = m.get("passed"), m.get("of"), m.get("source", "manifest")
    if not isinstance(floor.get("min"), int) or not isinstance(floor.get("of"), int):
        c = {"check": "easy_floor", "ok": False, "why": "the manifest sets no easy_floor {min, of}"}
    elif not isinstance(passed, int) or not of:
        c = {"check": "easy_floor", "ok": False, "why": "no easy result measured on this compute (give --results or easy_measured)"}
    else:
        # compare as shares when a re-sit sat a different number of easy tasks than the floor's
        need = floor["min"] / floor["of"]
        hit = passed / of >= need - 1e-9
        c = {"check": "easy_floor", "ok": hit, "measured": f"{passed} of {of}", "floor": f"{floor['min']} of {floor['of']}",
             "source": source, "why": "at or above the floor" if hit else "BELOW the floor"}
        if not hit and "easy_floor" in waivers:
            c.update(ok=True, why="below the floor, launched under a logged waiver", waiver=waivers["easy_floor"])
            rec["waivers_used"].append(waivers["easy_floor"])
    rec["checks"].append(c)

    # 2. unique solves of cells being dropped
    ov = manifest.get("override")
    if ov:
        dropped = ov.get("drop") or []
        evidence = exam_results(ov.get("evidence") or []) + (results or [])
        flagged = unique_solves(evidence, dropped)
        acked = set(map(str, ov.get("unique_solves_acknowledged") or []))
        open_ = [t for t, _ in flagged if t not in acked]
        c = {"check": "unique_solves", "ok": not open_ and bool(evidence), "dropped": dropped,
             "flagged": [{"task": t, "only": list(cs)} for t, cs in flagged], "unacknowledged": open_,
             "why": ("no evidence given for an override: the diff cannot be made" if not evidence else
                     f"{len(open_)} task(s) solved only by a dropped cell, not acknowledged" if open_ else
                     f"{len(flagged)} flagged, all acknowledged" if flagged else "no task solved only by a dropped cell")}
    else:
        c = {"check": "unique_solves", "ok": True, "why": "no override or rollback in this launch"}
    rec["checks"].append(c)

    # 3. blocking dissent, acknowledged
    pending = [d.get("id") or d.get("summary", "?")[:60] for d in manifest.get("dissent") or []
               if d.get("blocking") and not (d.get("acknowledged_by") and d.get("acknowledged_at"))]
    rec["checks"].append({"check": "dissent", "ok": not pending, "unacknowledged": pending,
                          "why": f"{len(pending)} blocking dissent(s) without a delivery acknowledgment" if pending else
                          "every blocking dissent acknowledged (or none)"})
    ok = all(c["ok"] for c in rec["checks"])
    rec["ok"] = ok
    return ok, rec


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    manifest = json.loads(Path(argv[0]).read_text())
    dirs, out, i = [], None, 1
    while i < len(argv):
        if argv[i] == "--results":
            dirs.append(argv[i + 1]); i += 2
        elif argv[i] == "--out":
            out = argv[i + 1]; i += 2
        else:
            raise SystemExit(f"preflight: unknown argument {argv[i]}")
    ok, rec = check(manifest, exam_results(dirs) if dirs else None)
    for c in rec["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['why']}" + (f" ({c['measured']} against {c['floor']})" if "measured" in c else ""))
    print("pre-flight:", "CLEAR TO LAUNCH" if ok else "LAUNCH BLOCKED")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rec, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
