"""Audit a preserved rows file for the open-web false positive: answers marked right that were really page dumps.

    python3 scripts/audit_web_rows.py ROWS.jsonl [--out runs/audit/web-rows.tsv] [--quarantine]

The confirmed incident (Codex audit, 2026-11-28): g34, task h10003306_01, the Platte River (Wikidata Q655754). Two
members answered with a dump of Wikidata's page; the old check took ANY number of the text near the value as the
answer, so both were marked DONE, and rows of datasets/g80/rows.jsonl teach that program: the audit counted 7 (lines
1068-1074); R1 found 8 (1068-1075) and a second incident (2706-2711): docs/r1/.

For every row of a web kind, this re-reads the answer the member actually gave: a "done" row carries it as
"OUTPUT:" in its last message. It judges that answer by the fixed check's answer-shape rule
(training/kinds/_common._dumped: no page or JSON dump, few enough numbers). The values themselves were live and are not
re-fetched, so a flag says "the old check could have passed this by luck", not "the value was wrong". A flagged
answer flags every row of the same task, seed and member (its "first", "fix" and "pair" rows teach the same program).

Prints the incidence by kind and writes one TSV line per flagged row: line, kind, type, seed, member, why.
--quarantine also appends the flagged rows' keys to training/data/QUARANTINE.jsonl (once each), so no later
training set takes them. The rows file itself is never changed.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import quarantine  # noqa: E402
from training.kinds._common import _dumped  # noqa: E402

# kind -> the most numbers an honest answer of it holds (the same limits the fixed checks use)
WEB_KINDS = {
    "a fact found on the open web": 6,
    "walk a site over several pages": 8,
    "a fact from a site and a date sum on it": 10,
    "a fact read and a date sum on it": 12,
    "two facts from two sources, combined": 16,
    "operate a website": 12,          # easy kind: its check is a substring test, also open to a dump
    "a fact read from a source": 12,  # easy kind
}


def answer_of(row):
    """The member's printed answer, from a "done" row's last message, or None."""
    ms = row.get("messages") or []
    if row.get("type") != "done" or not ms:
        return None
    text = ms[-1].get("content") or ""
    if not text.startswith("OUTPUT:"):
        return None
    body = text[len("OUTPUT:"):]
    return body.rsplit("\n\n", 1)[0] if "\n\n" in body else body  # after the LAST blank line: solo.MORE, not the answer


def audit(path):
    rows = []
    for i, ln in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append((i, r))
    web = [(i, r) for i, r in rows if r.get("kind") in WEB_KINDS]
    groups = defaultdict(list)
    for i, r in web:
        groups[(r.get("kind"), str(r.get("seed")), str(r.get("task", ""))[:200], r.get("member"))].append((i, r))
    flagged, done_seen, done_bad = [], Counter(), Counter()
    for g, members in groups.items():
        kind = g[0]
        whys = [w for _i, r in members if (a := answer_of(r)) is not None for w in [_dumped(a, WEB_KINDS[kind])]]
        if any(answer_of(r) is not None for _i, r in members):
            done_seen[kind] += 1
        bad = [w for w in whys if w]
        if bad:
            done_bad[kind] += 1
            flagged += [(i, r, bad[0]) for i, r in members]
    return rows, web, flagged, done_seen, done_bad


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    opts = dict(zip(argv[1::2], argv[2::2]))
    out = Path(opts.get("--out", ROOT / "runs" / "audit" / "web-rows.tsv"))
    rows, web, flagged, seen, bad = audit(argv[0])
    print(f"{argv[0]}: {len(rows)} rows, {len(web)} of web kinds")
    print("kind\tanswers (task+member)\tflagged\tshare")
    for k in WEB_KINDS:
        if seen[k] or bad[k]:
            print(f"{k}\t{seen[k]}\t{bad[k]}\t{bad[k] / seen[k]:.1%}" if seen[k] else f"{k}\t0\t{bad[k]}\t-")
    print(f"flagged rows: {len(flagged)} (from {sum(bad.values())} answers of {sum(seen.values())})")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("line\tkind\ttype\tseed\tmember\tsha256\twhy\n")
        for i, r, why in sorted(flagged, key=lambda x: x[0]):
            f.write(f"{i}\t{r.get('kind')}\t{r.get('type')}\t{r.get('seed')}\t{r.get('member')}\t{quarantine.key(r)}\t{why}\n")
    print(f"written to {out}")
    if "--quarantine" in argv:
        have = quarantine.keys()
        new = [(i, r, w) for i, r, w in flagged if quarantine.key(r) not in have]
        with quarantine.FILE.open("a") as f:
            for i, r, why in new:
                f.write(json.dumps({"sha256": quarantine.key(r), "why": why, "source": str(argv[0]), "line": i,
                                    "kind": r.get("kind"), "seed": r.get("seed"), "type": r.get("type")}) + "\n")
                have.add(quarantine.key(r))
        print(f"quarantine: {len(new)} new key(s) added to {quarantine.FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
