"""Rows that must never be trained on again, kept as evidence and left out of every training set.

    training/data/QUARANTINE.jsonl   one JSON object a line: {"sha256": <row key>, "why": ..., "source": ..., "line": ...}

A row's key is the sha256 of the same fields training/grow.once compares (type, messages, output, rejected), so the
key does not change when the row is stamped with a generation or copied into another file. scripts/audit_web_rows.py
writes the entries; training/grow.py gather and train drop every quarantined row and say how many. Nothing is deleted
from the archive: the rows stay in datasets/g80/rows.jsonl on Drive as audit evidence.

First entries (the verifier seat's audit, 2026-11-28): g34 task h10003306_01 (the Platte River, Wikidata Q655754), two members
answered with a dump of Wikidata's page and were marked right. The audit counted seven rows (lines 1068-1074).
R1's audit (2026-11-28, docs/r1/) found the eighth, line 1075, the second member's pair row, and a second incident
of the same kind (seed 30004256, lines 2706-2711, a page or JSON dump). All 14 keys are in the file.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "training" / "data" / "QUARANTINE.jsonl"


def key(row):
    return hashlib.sha256(json.dumps([row.get("type"), row.get("messages"), row.get("output"), row.get("rejected")],
                                     sort_keys=True).encode()).hexdigest()


class QuarantineError(RuntimeError):
    """The quarantine cannot be trusted: training must stop rather than train on rows it may have let through."""


HEX64 = re.compile(r"[0-9a-f]{64}")


def keys(path=None, *, building=False):
    """The set of quarantined row keys. FAILS CLOSED (audit of 2026-11-28): a missing file, an empty one, or any entry
    whose sha256 is not 64 lowercase hex characters raises QuarantineError, so a training set is never built from a
    quarantine that silently filters nothing.

    building=True is only for the tools that WRITE the file (scripts/audit_web_rows.py --quarantine, scripts/r1_run.py):
    there a missing or empty file is the start, not a fault. A malformed entry raises there too."""
    path = Path(path or FILE)  # looked up at the call, not when the module loads
    if not path.exists():
        if building:
            return set()
        raise QuarantineError(f"quarantine file missing: {path} (training stops: it cannot leave out the audited rows)")
    out = set()
    for n, ln in enumerate(path.read_text().splitlines(), 1):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            digest = json.loads(ln)["sha256"]
        except (ValueError, KeyError, TypeError) as e:
            raise QuarantineError(f"{path}:{n}: not a JSON entry with a sha256 ({type(e).__name__})") from None
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise QuarantineError(f"{path}:{n}: sha256 {digest!r} is not 64 lowercase hex characters")
        out.add(digest)
    if not out and not building:
        raise QuarantineError(f"quarantine file has no entries: {path} (training stops: it cannot leave out the audited rows)")
    return out


def drop(rows, path=None, log=print):
    """ROWS without the quarantined ones (dicts or JSON lines, returned as given). Raises QuarantineError when the
    quarantine is missing, empty or malformed (keys)."""
    bad = keys(path)
    kept = [r for r in rows if key(json.loads(r) if isinstance(r, str) else r) not in bad]
    if len(kept) != len(rows):
        log(f"quarantine: {len(rows) - len(kept)} row(s) left out (training/data/QUARANTINE.jsonl)")
    return kept
