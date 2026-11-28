"""The resume tools of docs/RESUME-G81.md: the Hugging Face store, the pre-flight gate, the capability check's verdict
and the lean test list. All offline: the store is faked, nothing is downloaded."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training import hf_store, release_store  # noqa: E402
import capability_check  # noqa: E402
import lean_tests  # noqa: E402
import preflight  # noqa: E402


# --- the store -------------------------------------------------------------------------------------------------

def test_a_file_sits_under_its_generation_and_everything_else_is_shared():
    assert hf_store.path_in_store("t80.g81code.safetensors") == "g81/t80.g81code.safetensors"
    assert hf_store.path_in_store("t80.g81trunk.gguf") == "g81/t80.g81trunk.gguf"
    assert hf_store.path_in_store("t80.g44words2.gguf") == "g44/t80.g44words2.gguf"
    assert hf_store.path_in_store("some/dir/t80.g9data.json") == "g9/t80.g9data.json"
    assert hf_store.path_in_store("rows.jsonl") == "shared/rows.jsonl"
    assert hf_store.path_in_store("failures.jsonl") == "shared/failures.jsonl"
    assert hf_store.path_in_store("t80.gguf") == "shared/t80.gguf"  # 'gguf' names no generation


class FakeHub:
    def __init__(self):
        self.files, self.commits = {}, []

    def create_repo(self, **kw):
        assert kw["private"] is True and kw["exist_ok"] is True

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append([op.path_in_repo for op in operations])
        for op in operations:
            self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()

    def download(self, repo_id, filename, repo_type, token, _dir):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        p = Path(_dir) / filename.replace("/", "_")
        p.write_bytes(self.files[filename])
        return str(p)


def test_put_is_one_commit_and_get_reads_it_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_HF_REPO", "someone/hive-cells")
    hub = FakeHub()
    a, b = tmp_path / "t80.g82code.safetensors", tmp_path / "rows.jsonl"
    a.write_bytes(b"weights"); b.write_text("{}\n")
    assert hf_store.put([a, b], api=hub) == 0
    assert hub.commits == [["g82/t80.g82code.safetensors", "shared/rows.jsonl"]]
    dl = lambda **kw: hub.download(_dir=tmp_path, **kw)  # noqa: E731
    out = tmp_path / "got" / "x.safetensors"
    assert hf_store.get("t80.g82code.safetensors", out, download=dl) and out.read_bytes() == b"weights"
    assert not hf_store.get("t80.g82data.safetensors", tmp_path / "none", download=dl)
    assert not (tmp_path / "none").exists()


def test_put_refuses_what_is_not_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_HF_REPO", "someone/hive-cells")
    try:
        hf_store.put([tmp_path / "missing.gguf"], api=FakeHub())
    except SystemExit as e:
        assert "not files" in str(e)
    else:
        raise AssertionError("a missing file was put")


def test_no_repo_named_is_a_clear_stop(monkeypatch):
    monkeypatch.delenv("HIVE_HF_REPO", raising=False)
    try:
        hf_store.repo_id()
    except SystemExit as e:
        assert "HIVE_HF_REPO" in str(e)
    else:
        raise AssertionError("no repo, no stop")


def test_the_release_store_goes_to_the_hub_only_when_asked(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(hf_store, "get", lambda name, out: called.append(("get", name)) or True)
    monkeypatch.setattr(hf_store, "put", lambda files: called.append(("put", list(files))) or 0)
    monkeypatch.setenv("HIVE_STORE", "hf")
    assert release_store.get("t80.g81code.gguf", tmp_path / "x", "any/repo", fetch=None)
    assert release_store.put(["a.gguf"]) == 0
    assert called == [("get", "t80.g81code.gguf"), ("put", ["a.gguf"])]
    monkeypatch.delenv("HIVE_STORE")
    tried = []
    release_store.get("t80.g81code.gguf", tmp_path / "y", "any/repo", fetch=lambda u, o: tried.append(u) or (_ for _ in ()).throw(OSError()))
    assert tried and all("releases/download" in u for u in tried)  # unset: the old releases, exactly as before


# --- the pre-flight gate -----------------------------------------------------------------------------------------

def _manifest(**kw):
    m = {"generation": "g82", "grows_from": "g81", "easy_floor": {"min": 30, "of": 32, "basis": "g81 32/32, minus 2"},
         "waivers": [], "override": None, "dissent": []}
    m.update(kw)
    return m


def _exam(tmp_path, tid, ok, members):
    d = tmp_path / tid
    d.mkdir(parents=True)
    (d / "exam.json").write_text(json.dumps({"id": tid, "ok": ok, "members": [{"who": w, "ok": o} for w, o in members]}))


def _results(tmp_path, easy_right, easy=32):
    for i in range(easy):
        _exam(tmp_path / "r", f"e1_{i:02}", i < easy_right, [])
    return preflight.exam_results([tmp_path / "r"])


def test_the_floor_holds_and_breaks(tmp_path):
    ok, rec = preflight.check(_manifest(), _results(tmp_path, 31))
    assert ok and rec["checks"][0]["measured"] == "31 of 32"
    ok, rec = preflight.check(_manifest(), _results(tmp_path / "b", 29))
    assert not ok and rec["checks"][0]["why"] == "BELOW the floor"


def test_a_breach_launches_only_with_a_logged_waiver(tmp_path):
    res = _results(tmp_path, 28)
    assert not preflight.check(_manifest(), res)[0]
    w = {"check": "easy_floor", "by": "owner", "reason": "runtime rebuilt; re-sit scheduled", "at": "2026-11-28"}
    assert not preflight.check(_manifest(waivers=[{"check": "easy_floor", "by": "owner"}]), res)[0]  # no reason: no waiver
    ok, rec = preflight.check(_manifest(waivers=[w]), res)
    assert ok and rec["waivers_used"] == [w] and rec["checks"][0]["waiver"] == w


def test_no_measurement_is_no_launch():
    ok, rec = preflight.check(_manifest())
    assert not ok and "no easy result" in rec["checks"][0]["why"]
    assert preflight.check(_manifest(easy_measured={"passed": 32, "of": 32, "source": "g81 read"}))[0]


def test_a_task_only_a_dropped_cell_solved_is_flagged_until_acknowledged(tmp_path):
    ev = tmp_path / "ev"
    _exam(ev, "h1_00", False, [("t80+g81code", True), ("t80+g81trunk", False)])   # only the dropped cell: flagged
    _exam(ev, "h1_01", True, [("t80+g81code", True), ("t80+g80code", True)])      # a kept cell also had it: fine
    _exam(ev, "h1_02", False, [("t80+g81data", False)])                           # nobody: nothing to lose
    ov = {"kind": "rollback", "drop": ["g81code"], "evidence": [str(ev)]}
    ok, rec = preflight.check(_manifest(override=ov, easy_measured={"passed": 32, "of": 32}))
    c = rec["checks"][1]
    assert not ok and c["unacknowledged"] == ["h1_00"] and c["flagged"] == [{"task": "h1_00", "only": ["g81code"]}]
    ov["unique_solves_acknowledged"] = ["h1_00"]
    assert preflight.check(_manifest(override=ov, easy_measured={"passed": 32, "of": 32}))[0]


def test_an_override_with_no_evidence_cannot_pass(tmp_path):
    ok, rec = preflight.check(_manifest(override={"drop": ["g81code"], "evidence": []}, easy_measured={"passed": 32, "of": 32}))
    assert not ok and "no evidence" in rec["checks"][1]["why"]


def test_blocking_dissent_needs_a_delivery_acknowledgment():
    d = {"id": "D1", "from": "a council seat", "blocking": True, "summary": "..."}
    m = _manifest(easy_measured={"passed": 32, "of": 32}, dissent=[d, {"id": "D2", "blocking": False}])
    ok, rec = preflight.check(m)
    assert not ok and rec["checks"][2]["unacknowledged"] == ["D1"]
    d.update(acknowledged_by="Muse", acknowledged_at="2026-11-28T10:00Z")
    assert preflight.check(m)[0]


def test_the_runbook_manifest_is_valid_json_and_blocks_until_measured():
    text = (ROOT / "docs" / "RESUME-G81.md").read_text()
    block = text.split("```json\n", 1)[1].split("```", 1)[0]
    m = json.loads(block)
    assert m["generation"] == "g82" and m["easy_floor"]["of"] == 32
    assert not preflight.check(m)[0]  # nothing measured on the new compute yet: no launch


# --- the capability check ----------------------------------------------------------------------------------------

def test_the_five_tasks_are_the_walk_kind_and_the_same_every_time():
    a, b = capability_check.tasks(), capability_check.tasks()
    assert len(a) == 5 and [t[3] for t in a] == [t[3] for t in b]
    assert all(t[2] == "walk a site over several pages" and "books.toscrape.com" in t[3] for t in a)


def test_the_arms_are_the_hive_the_trunk_and_each_code_cell_alone():
    arms = capability_check.arms()
    assert arms["hive"]["AGENT_TEAM_SOLO"] == "2" and arms["hive"]["HIVE_MEMBERS"].startswith("t80+g81trunk,")
    assert arms["trunk"] == {"HIVE_MEMBERS": "t80+g81trunk", "AGENT_TEAM_SOLO": "1", "AGENT_TEAM_SOLO_ALONE": "1"}
    assert {"g81code", "g81code2", "g80code", "g80code2", "g39code"} <= set(arms)


def _r(tid, ok, members=()):
    return {"id": tid, "ok": ok, "members": [{"who": w, "ok": o} for w, o in members]}


def test_the_verdicts():
    five = [f"cap{i}" for i in range(5)]
    assert capability_check.verdict({"hive": [_r(t, True) for t in five]})["verdict"] == "PASS"
    v = capability_check.verdict({"hive": [_r(t, False) for t in five], "g81code": [_r("cap0", True)] + [_r(t, False) for t in five[1:]]})
    assert v["verdict"] == "ROUTING" and v["solved_but_not_by_the_hive"] == {"cap0": ["g81code"]}
    v = capability_check.verdict({"hive": [_r("cap1", False, [("t80+g80code", True)])] + [_r(t, False) for t in five[:1] + five[2:]]})
    assert v["verdict"] == "ROUTING" and "member:t80+g80code" in v["solved_but_not_by_the_hive"]["cap1"]
    assert capability_check.verdict({"hive": [_r(t, False) for t in five], "trunk": [_r(t, False) for t in five]})["verdict"] == "ABSENT"
    assert capability_check.verdict({})["verdict"] == "NOT RUN"


# --- the lean list -----------------------------------------------------------------------------------------------

def test_every_lean_exclusion_has_a_category_a_reason_and_names_a_real_test():
    items = lean_tests.entries()
    assert items and not lean_tests.stale(items)
    assert all(t.startswith("tests/test_") for t, _c, _w in items)
    assert not any("test_resume_tools" in t for t, _c, _w in items)  # this file is never on the list


# --- the open-web checker (Codex audit: g34, h10003306_01, the Platte River, Wikidata Q655754) --------------------

def _web_check(monkeypatch, values=(499.0,)):
    from training.kinds import h_web
    import random
    monkeypatch.setattr(h_web, "_web_values", lambda qid, prop, units: list(values))
    r = random.Random(1)
    r.choice = lambda seq: seq[2]  # the river row of WEB
    return h_web.h_web(r, pick=lambda *a: ("Q655754", "Platte River", "United States"))[3]


WIKIDATA_DUMP = ('{"entities": {"Q655754": {"type": "item", "id": "Q655754", "labels": {"en": {"value": "Platte River"}}, '
                 '"claims": {"P2043": [{"mainsnak": {"datavalue": {"value": {"amount": "+499", "unit": "http://www.wikidata.org/entity/Q828224"}}}}], '
                 '"P625": [{"mainsnak": {"datavalue": {"value": {"latitude": 41.0, "longitude": -96.1}}}}]}, "lastrevid": 2290017715}}}')
HOME_DUMP = "Wikidata Main Page " + " ".join(f"item {i} edits {i * 37} pages 499" for i in range(40))


def test_the_g34_wikidata_dump_no_longer_passes_as_an_answer(monkeypatch):
    check = _web_check(monkeypatch)
    for dump in (WIKIDATA_DUMP, HOME_DUMP):
        ok, why = check(None, dump)
        assert not ok and ("dump" in why or "numbers" in why), why


def test_an_honest_answer_still_passes_and_a_wrong_one_still_fails(monkeypatch):
    check = _web_check(monkeypatch)
    assert check(None, "499")[0]
    assert check(None, "The Platte River is 499 km long (Wikidata Q655754).")[0]
    assert not check(None, "The Platte River is 1,310 km long.")[0]


def test_a_category_walk_needs_the_two_numbers_said_not_a_page_of_prices(monkeypatch):
    from training.kinds import h_walk
    import random
    books = [(f"b{i}", 10.0 + i) for i in range(52)]
    monkeypatch.setattr(h_walk, "_all_books", lambda cat: books)
    r = random.Random(0)
    r.random = lambda: 0.9  # cheapest
    check = h_walk.h_walk(r)[3]
    assert check(None, "books: 52\ncheapest: 10.00")[0]
    page = "\n".join(f"<li>b{i} £{10.0 + i:.2f}</li>" for i in range(52)) + "\n52 results"
    assert not check(None, page)[0]
    assert not check(None, " ".join(f"{10.0 + i:.2f}" for i in range(52)) + " 52")[0]  # every price listed: luck, not an answer


def test_answer_shape_helpers():
    from training.kinds._common import _dumped, _states
    assert _dumped("days: 174") == ""
    assert "dump" in _dumped('<html><body>x</body></html>')
    assert "numbers" in _dumped(" ".join(map(str, range(20))))
    assert _states("count 7, price 3.50", [7, 3.5], 0.01)[0]
    assert not _states("count 7", [7, 3.5], 0.01)[0]


# --- the row audit and the quarantine ----------------------------------------------------------------------------

def _rows_file(tmp_path):
    import audit_web_rows  # noqa: F401
    head = [{"role": "system", "content": "orders"}, {"role": "user", "content": "Find the length of the river Platte River"}]
    rows = []
    for member, answer in ((1, WIKIDATA_DUMP), (2, HOME_DUMP)):  # the g34 incident: two members, both dumps
        base = {"task": "Find the length of the river Platte River", "kind": "a fact found on the open web", "seed": "h10003306_01", "member": member}
        rows += [base | {"type": "first", "messages": head, "output": f"```python\nprint({member})\n```"},
                 base | {"type": "done", "messages": head + [{"role": "assistant", "content": "x"},
                                                             {"role": "user", "content": f"OUTPUT:\n{answer}\n\nMore?"}], "output": "DONE"},
                 base | {"type": "fix", "messages": head + [{"role": "assistant", "content": "y"}], "output": "z"}]
    honest = {"task": "Find the length of the river Rhine", "kind": "a fact found on the open web", "seed": "h1_00", "member": 1}
    rows += [honest | {"type": "done", "messages": head + [{"role": "user", "content": "OUTPUT:\n1233\n\nMore?"}], "output": "DONE"},
             {"task": "sum", "kind": "a sum", "seed": "e1", "type": "done", "messages": head + [{"role": "user", "content": "OUTPUT:\n" + " ".join(map(str, range(50))) + "\n\nMore?"}], "output": "DONE"}]
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f, rows


def test_the_audit_flags_the_whole_incident_and_nothing_honest(tmp_path):
    import audit_web_rows
    f, rows = _rows_file(tmp_path)
    _all, web, flagged, seen, bad = audit_web_rows.audit(f)
    assert len(web) == 7 and seen["a fact found on the open web"] == 3 and bad["a fact found on the open web"] == 2
    assert sorted(i for i, _r, _w in flagged) == [1, 2, 3, 4, 5, 6]  # both members' first, done and fix rows
    # a non-web kind with many numbers is not a web false positive and is not looked at


def test_quarantined_rows_never_reach_a_training_set(tmp_path, monkeypatch):
    import audit_web_rows
    from training import grow, quarantine
    f, rows = _rows_file(tmp_path)
    q = tmp_path / "Q.jsonl"
    monkeypatch.setattr(quarantine, "FILE", q)
    _a, _w, flagged, _s, _b = audit_web_rows.audit(f)
    q.write_text("".join(json.dumps({"sha256": quarantine.key(r)}) + "\n" for _i, r, _why in flagged))
    out = tmp_path / "out.jsonl"
    before, added = grow.gather(str(out), "g82", old=[str(f)], new=[], log=lambda *a, **k: None)
    kept = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(kept) == len(rows) - 6 and all(r.get("seed") != "h10003306_01" for r in kept)
    assert f.read_text().count("h10003306_01") == 6  # the evidence itself is untouched
    stamped = dict(rows[1], gen="g34")  # the key ignores the generation stamp, so a copied row is still caught
    assert quarantine.drop([stamped], path=q, log=lambda *a: None) == []
