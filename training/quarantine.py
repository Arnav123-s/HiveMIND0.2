"""Rows that must never be trained on again, kept as evidence and left out of every training set.

    training/data/QUARANTINE.jsonl   one JSON object a line: {"sha256": <row key>, "why": ..., "source": ..., "line": ...}

A row's key is the sha256 of the same fields training/grow.once compares (type, messages, output, rejected), so the
key does not change when the row is stamped with a generation or copied into another file. scripts/audit_web_rows.py
writes the entries; training/grow.py gather and train drop every quarantined row and say how many. Nothing is deleted
from the archive: the rows stay in datasets/g80/rows.jsonl on Drive as audit evidence.

First entries (Codex audit, 2026-11-28): g34 task h10003306_01 (the Platte River, Wikidata Q655754), two members
answered with a dump of Wikidata's page and were marked right. The audit counted seven rows (lines 1068-1074).
R1's audit (2026-11-28, docs/r1/) found the eighth, line 1075, the second member's pair row, and a second incident
of the same kind (seed 30004256, lines 2706-2711, a page or JSON dump). All 14 keys are in the file.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "training" / "data" / "QUARANTINE.jsonl"


def key(row):
    return hashlib.sha256(json.dumps([row.get("type"), row.get("messages"), row.get("output"), row.get("rejected")],
                                     sort_keys=True).encode()).hexdigest()


def keys(path=None):
    path = Path(path or FILE)  # looked up at the call, not when the module loads
    out = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.add(json.loads(ln)["sha256"])
    return out


def drop(rows, path=None, log=print):
    """ROWS without the quarantined ones (dicts or JSON lines, returned as given)."""
    bad = keys(path)
    if not bad:
        return rows
    kept = [r for r in rows if key(json.loads(r) if isinstance(r, str) else r) not in bad]
    if len(kept) != len(rows):
        log(f"quarantine: {len(rows) - len(kept)} row(s) left out (training/data/QUARANTINE.jsonl)")
    return kept
