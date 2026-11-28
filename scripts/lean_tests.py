"""The lean test procedure: the full suite minus the reviewed list in tests/LEAN-DESELECT.tsv, nothing else left out.

    python3 scripts/lean_tests.py            run it: exit 0 only when every test outside the list passes
    python3 scripts/lean_tests.py --audit    run ONLY the listed tests and say which now pass (so the list can shrink)

Why a list and not skips in the tests: the old suite asserts on the archived workflows (.github/workflows, removed on
purpose: docs/REPO-POLICY.md) and on data that lives in the Drive archive by design. Those failures say nothing about
the code. Every exclusion is one line with its reason, in one file, under review like any other change; a failure
that is not on the list is a regression, and the run fails.

The list is also checked: an entry that names no test any more is stale, and the run fails until it is removed.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "tests" / "LEAN-DESELECT.tsv"


def entries():
    out = []
    for line in LIST.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("id\t"):
            continue
        tid, cat, why = (line.split("\t") + ["", ""])[:3]
        if cat not in ("WORKFLOW", "DRIVE_DATA", "PREEXISTING"):
            raise SystemExit(f"lean_tests: {tid}: unknown category {cat!r}")
        if not why.strip():
            raise SystemExit(f"lean_tests: {tid}: no reason given")
        out.append((tid, cat, why))
    return out


def stale(items):
    """Listed ids whose file or test function no longer exists."""
    bad = []
    for tid, _c, _w in items:
        path, _, name = tid.partition("::")
        f = ROOT / path
        if not f.is_file():
            bad.append(tid)
        elif name and f"def {name.split('[')[0]}(" not in f.read_text():
            bad.append(tid)
    return bad


def main(argv):
    items = entries()
    bad = stale(items)
    if bad:
        print("lean_tests: stale entries in tests/LEAN-DESELECT.tsv (remove them):\n  " + "\n  ".join(bad))
        return 2
    files = [t for t, _c, _w in items if "::" not in t]
    tests = [t for t, _c, _w in items if "::" in t]
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if "--audit" in argv:
        cmd = base + ["-rA", "--continue-on-collection-errors"] + tests + files
        print(f"auditing {len(items)} listed entries")
        return subprocess.run(cmd, cwd=ROOT).returncode
    cmd = base + ["tests"] + [a for f in files for a in ("--ignore", f)] + [a for t in tests for a in ("--deselect", t)]
    print(f"lean suite: full suite minus {len(files)} files and {len(tests)} tests listed in tests/LEAN-DESELECT.tsv")
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
