import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from training import grow, grow_record as record, lineage


def row(i, kind="common", **fields):
    return json.dumps({"kind": kind, "type": "first", "messages": [{"role": "system", "content": "s"},
                       {"role": "user", "content": str(i)}], "output": str(i), **fields}, separators=(",", ":"))


def write_rows(path, rows):
    path.write_bytes("".join(line + "\n" for line in rows).encode())


def test_strata_and_lossless_local_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_STRATIFIED_POOL", "1")
    monkeypatch.delenv("GITHUB_JOB", raising=False)
    anchors = [row(i, f"rare{i}") for i in range(5)]
    hard = [row(i, hard=True) for i in range(5, 14)]
    fixes = [row(14, type="pair"), row(15, type="fix"), row(16, dream=True),
             row(17, failed_try_1=True, solved_later=True)]
    middle = [row(i) for i in range(18, 28)]
    newest = [row(i) for i in range(28, 36)]
    original = anchors + hard + fixes + middle + newest
    source = tmp_path / "rows.jsonl"
    duplicate = json.dumps(dict(json.loads(anchors[0]), who="another cell"))
    write_rows(source, [*original, duplicate, "invalid json", "null"])
    record.join([source, source], record._row, 20)
    assert source.read_text().splitlines() == anchors + hard[-3:] + fixes + newest
    archive, = (tmp_path / "pool-archive").glob("*.jsonl.gz")
    restored = gzip.decompress(archive.read_bytes())
    assert restored.decode().splitlines() == original
    assert archive.name == f"pool-{hashlib.sha256(restored).hexdigest()}.jsonl.gz"
    # No eligible salience: the reserved places return to recency; the anchor stays.
    ordinary = [row(i) for i in range(20)]
    assert record.union([ordinary], most=5, snapshot=[]) == [ordinary[0], *ordinary[-4:]]
    assert record.union([ordinary], most=0, snapshot=[]) == []
    assert record.union([ordinary], most=21, snapshot=[]) == ordinary
    makers = [row(i, k) for i, k in enumerate(("alone", "h_walk", "plain", "plain", "h_walk", "plain", "plain"))]
    assert record.union([makers], most=5, snapshot=[]) == [makers[0], makers[1], *makers[-3:]]


def test_defaults_and_failures_are_original_fifo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [row(i) for i in range(18)]
    args = [rows[:11], [json.dumps(dict(json.loads(rows[0]), who="later")), "bad", *rows[11:]]]
    for flag in (None, "0"):
        if flag is None:
            monkeypatch.delenv("HIVE_STRATIFIED_POOL", raising=False)
        else:
            monkeypatch.setenv("HIVE_STRATIFIED_POOL", flag)
        assert record.union(args, most=5) == rows[-5:]
        assert record.union(args, most=0) == rows
    monkeypatch.setenv("HIVE_STRATIFIED_POOL", "1")
    failures = [json.dumps(dict(maker="m", rseed=i, program="p")) for i in range(5)]
    assert record.union([failures], record._failure, 2) == failures[-2:]
    assert list(tmp_path.iterdir()) == []


def test_gather_to_record_reconstructs_new_overflow_before_replacing_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_JOB", "gather")
    monkeypatch.setenv("HIVE_STRATIFIED_POOL", "1")
    monkeypatch.setattr(record.union, "__defaults__", (record._row, 20))
    Path("lessons").mkdir()
    note = Path("lessons/lessons.json")
    note.write_text('{"lessons":18}')
    old = [row(i, f"kind{i % 5}", hard=True) for i in range(12)]
    new = [row(i) for i in range(12, 45)]
    write_rows(Path("old"), old)
    write_rows(Path("new"), new)
    grow.gather("got/rows.jsonl", "g81", old=["old"], new=["new"], log=lambda *a: None)
    dest = Path("got/rows.jsonl")
    gathered = dest.read_bytes()
    stamped = [json.dumps(dict(json.loads(line), gen="g81")) for line in new]
    assert all(line in old + stamped for line in gathered.decode().splitlines())
    receipt = json.loads(note.read_text())
    assert receipt["lessons"] == 18
    assert sum(line in stamped for _, line in receipt["pool_v1"]["spill"]) > 0
    monkeypatch.setenv("GITHUB_JOB", "record")
    write_rows(Path("now"), old + [row("concurrent")])
    expected = [*old, row("concurrent"), *stamped]
    uploads = []
    def publish(archive):
        assert dest.read_bytes() == gathered
        assert gzip.decompress(archive.read_bytes()).decode().splitlines() == expected
        uploads.append(archive)
    monkeypatch.setattr(record, "_publish_archive", publish)

    note.write_text(json.dumps(dict(receipt, pool_v1=dict(receipt["pool_v1"], run="122"))))
    with pytest.raises(ValueError, match="gather run"):
        record.join([dest, "now", dest], record._row, 20)
    tampered = json.loads(json.dumps(receipt))
    tampered["pool_v1"]["spill"][0][1] = row("tampered")
    note.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="reconstruct"):
        record.join([dest, "now", dest], record._row, 20)
    assert not uploads and dest.read_bytes() == gathered
    note.write_text(json.dumps(receipt))
    def fail(path):
        raise OSError("archive upload failed")
    monkeypatch.setattr(record, "_publish_archive", fail)
    with pytest.raises(OSError, match="upload failed"):
        record.join([dest, "now", dest], record._row, 20)
    assert dest.read_bytes() == gathered
    monkeypatch.setattr(record, "_publish_archive", publish)
    for _ in range(2):
        dest.write_bytes(gathered)  # a retry redownloads the original job artifact
        record.join([dest, "now", dest], record._row, 20)
        assert dest.read_text().splitlines() == record.union([expected], most=20, snapshot=[])
    assert uploads[0] == uploads[1]


def test_archive_retry_never_clobbers_a_verified_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/hive")
    monkeypatch.setattr(record.release_store, "tags", lambda: ["hive-grown"])
    remote = {}
    monkeypatch.setattr(record.release_store, "_names", lambda *args: set(remote))
    def gh(cmd, verb, tag, *args, check=True):
        if verb == "upload":
            assert "--clobber" not in args
            path = Path(args[-1])
            assert path.name not in remote
            remote[path.name] = path.read_bytes()
        else:
            assert verb == "download"
            Path(args[-1]).write_bytes(remote[args[args.index("--pattern") + 1]])
        return subprocess.CompletedProcess([], 0)
    monkeypatch.setattr(record.release_store, "_gh", gh)
    path = record._archive(tmp_path, [row(1)], publish=True)
    record._publish_archive(path)
    remote[path.name] = gzip.compress(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        record._publish_archive(path)


def sitting(root, arm, oks, cells, *, seed=42, code="0123456", shard=1, prompt="question"):
    folder = root / f"gate-{arm}-{shard}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"e{seed}_{i:02}", "task": f"{prompt} {i}", "kind": "days between two dates" if i else "a sum in words",
             "ok": ok, "members": [{"member": n, "who": who, "ok": grades[i]}
                                  for n, (who, grades) in enumerate(cells.items(), 1)]} for i, ok in enumerate(oks)]
    (folder / "exam.json").write_text(json.dumps({"summary": {"seed": seed, "code": code, "passed": sum(oks),
            "tasks": len(oks), "minutes": 1}, "tasks": rows}))


def test_real_gate_writes_niche_and_lineage_keeps_sole_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "ROOT", tmp_path)
    cells = tmp_path / "cells"
    cells.mkdir()
    body, niche = cells / "body.json", cells / "niche.json"
    report = tmp_path / "generations.jsonl"
    monkeypatch.setattr(record, "RECORD", report)
    monkeypatch.setattr(record, "BODY", body)
    monkeypatch.setattr(lineage, "BODY", body)
    monkeypatch.setattr(lineage, "GENERATIONS", report)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setenv("HIVE_NICHE_LINEAGE", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_JOB", "record")
    before = [f"g{i}code" for i in range(1, 25)]
    baseline = json.dumps({"dna": "t80", "generation": 24, "cells": before, "parents": {"g2code": "g1code"}})
    other = {"t17": {"g9data": {"another kind": 1}}}
    niche.write_text(json.dumps(other))
    got = tmp_path / "got"
    got.mkdir()
    (got / "t80.g25code.gguf").touch()
    (got / "t80.g25code.json").write_text('{"parent_cell":"g24code"}')
    folder = tmp_path / "gate"
    old = [True, False, True]
    sitting(folder, "before", [False] * 3, {"t80+g1code": old})
    sitting(folder, "after", [True] * 3,
            {"t80+g1code": old, "t80+g25code": [False, True, True], "t17+g25code": [True] * 3,
             "t80+g25trunk": [True] * 3})
    sitting(folder, "versus", [True] * 3, {"t80+g25code": [True] * 3})
    args = [folder, got, "g25", "--dna", "t80", "--grow"]
    monkeypatch.setenv("HIVE_WRITE_NICHE", "0")
    body.write_text(baseline)
    record.gate(args)
    assert json.loads(niche.read_text()) == other
    assert "g1code" not in json.loads(body.read_text())["cells"]
    monkeypatch.setenv("HIVE_WRITE_NICHE", "1")
    body.write_text(baseline)
    record.gate(args)
    scores = {"g1code": {"a sum in words": 1., "days between two dates": .5},
              "g25code": {"a sum in words": 0., "days between two dates": 1.}}
    assert json.loads(niche.read_text()) == dict(other, t80=scores)
    assert "g1code" in json.loads(body.read_text())["cells"]
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=tmp_path,
                                  text=True).splitlines() == ["cells/niche.json"]
    saved = niche.read_bytes()
    body.write_text(baseline)
    record.gate(args)
    assert niche.read_bytes() == saved
    sitting(folder, "after", [False] * 3, {"t80+g25code": [True] * 3})
    record.gate([*args, "--trial"])
    assert niche.read_bytes() == saved
    body.write_text(baseline)
    sitting(folder, "before", [True] * 3, {"t80+g1code": old, "t80+g24code": [False, True, False]})
    record.gate(args)
    assert set(json.loads(niche.read_text())["t80"]) == {"g1code", "g24code"}
    assert body.read_text() == baseline
    saved = niche.read_bytes()
    sitting(folder, "before", [True] * 3, {"t80+g1code": old})
    record.gate(args)
    assert niche.read_bytes() == saved  # no pair of retained cells was measured
    # A gate can still grade a body with one real cell even when all training workers were lost.
    body.write_text(json.dumps({"dna": "t80", "generation": 24, "cells": ["g1code"]}))
    (got / "t80.g25code.gguf").unlink()
    record.gate(args)
    assert json.loads(niche.read_text())["t80"] == {"g1code": scores["g1code"]}
    saved = niche.read_bytes()
    with report.open("a") as f:
        f.write(json.dumps({"generation": "g27", "dna": "t80", "kept": False}) + "\n")
    sitting(folder, "before", [True] * 3, {"t80+g1code": [False] * 3})
    record.gate(args)
    assert niche.read_bytes() == saved


def test_only_comparable_individual_receipts_enter_the_panel(tmp_path):
    yes = [True, True]
    sitting(tmp_path, "before", yes, {"t80+g1code": yes, "t80+g2code": [False, True]})
    sitting(tmp_path, "after", yes, {"t80+g1code": yes, "t80+g3words": yes}, prompt="different task")
    roster = dict(before={"g1code", "g2code"}, after={"g1code", "g3words"})
    assert record.niche_scores(tmp_path, "t80", roster) == {}
    sitting(tmp_path, "after", yes, {"t80+g1code": yes, "t80+g3words": yes})
    file = tmp_path / "gate-after-1/exam.json"
    document = json.loads(file.read_text())
    document["tasks"][0]["members"] = {"agreed": True, "agreeing": [1, 2]}
    document["tasks"][1]["members"] = [{"member": 1, "who": "t80+g1code", "ok": True},
                                       {"member": 1, "who": "t80+g3words", "ok": True}]
    file.write_text(json.dumps(document))
    assert set(record.niche_scores(tmp_path, "t80", roster)) == {"g1code", "g2code"}
    assert record.niche_scores(tmp_path, "t17", roster) == {}


def test_extra_seats_do_not_discard_shared_tasks(tmp_path):
    yes = [True] * 3
    sitting(tmp_path, "before", yes, {"t80+g1code": yes, "t80+g2code": [False, True, False]})
    older = tmp_path / "gate-before-1/exam.json"
    old_receipt = json.loads(older.read_text())
    for task in old_receipt["tasks"]:
        for member in task["members"]:
            del member["member"]  # early exam summaries already carried a real per-cell checker result
    older.write_text(json.dumps(old_receipt))
    sitting(tmp_path, "after", yes, {"t80+g3words": yes})
    path = tmp_path / "gate-after-1/exam.json"
    doc = json.loads(path.read_text())
    for task in doc["tasks"][1:]:
        task["members"] = []
    path.write_text(json.dumps(doc))
    scores = record.niche_scores(tmp_path, "t80", {"before": {"g1code", "g2code"}, "after": {"g3words"}})
    assert scores == {"g1code": {"a sum in words": 1., "days between two dates": 1.},
                      "g2code": {"a sum in words": 0., "days between two dates": .5}}


def test_gate_retires_against_the_new_dna_before_writing_the_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_WRITE_NICHE", "1")
    monkeypatch.setenv("HIVE_NICHE_LINEAGE", "1")
    monkeypatch.delenv("GITHUB_JOB", raising=False)
    body, path = tmp_path / "body.json", tmp_path / "niche.json"
    body.write_text(json.dumps({"dna": "t17", "generation": 3, "cells": ["g3code"]}))
    path.write_text(json.dumps({"t17": {"g25code3": {"other": 1}}}))
    monkeypatch.setattr(lineage, "BODY", body)
    monkeypatch.setattr(record, "BODY", body)
    monkeypatch.setattr(record, "RECORD", tmp_path / "records")
    got = tmp_path / "got"
    got.mkdir()
    for i in range(1, 26):
        name = "t80.g25code" + (str(i) if i > 1 else "")
        (got / (name + ".gguf")).touch()
        (got / (name + ".json")).write_text("{}")
    gate = tmp_path / "gate"
    sitting(gate, "after", [True, True], {"t80+g25code": [True, False], "t80+g25code2": [False, True]})
    record.gate([gate, got, "g25", "--dna", "t80", "--grow"])
    new = json.loads(body.read_text())
    assert new["dna"] == "t80" and len(new["cells"]) == lineage.MOST_CELLS
    assert {"g25code", "g25code2"} <= set(new["cells"])
    assert json.loads(path.read_text())["t17"] == {"g25code3": {"other": 1}}
