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


# --- a lost .safetensors rebuilt from the cell's .gguf (g39code, g44words2: docs/RESUME-G81.md, F5) ---------------

def test_a_cell_survives_the_round_trip_through_its_f16_gguf(tmp_path):
    """The hive's own layout (ternary_base.Cell.save: float32, base_model.model.<module>, the last layers, all seven
    projections), through the converter grow.py uses (--outtype f16), back through training/gguf_to_peft.py."""
    import subprocess
    import numpy as np
    from safetensors.numpy import save_file
    from training import gguf_to_peft, merge
    base, cell = tmp_path / "base", tmp_path / "cell"
    base.mkdir(); cell.mkdir()
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
        "intermediate_size": 192, "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 256, "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False}))
    rng, r = np.random.default_rng(7), 16
    shapes = {"self_attn.q_proj": (64, 64), "self_attn.k_proj": (32, 64), "self_attn.v_proj": (32, 64), "self_attn.o_proj": (64, 64),
              "mlp.gate_proj": (192, 64), "mlp.up_proj": (192, 64), "mlp.down_proj": (64, 192)}
    t = {}
    for layer in (4, 5):
        for m, (o, i) in shapes.items():
            t[f"base_model.model.model.layers.{layer}.{m}.lora_A.weight"] = (rng.standard_normal((r, i)) * 0.02).astype(np.float32)
            t[f"base_model.model.model.layers.{layer}.{m}.lora_B.weight"] = (rng.standard_normal((o, r)) * 0.01).astype(np.float32)
    save_file(t, str(cell / "adapter_model.safetensors"))
    (cell / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": 2 * r,
        "target_modules": [m.split(".")[1] for m in shapes], "layers_to_transform": [4, 5], "base_model_name_or_path": str(base)}))
    conv = ROOT / "training" / "llama_cpp_convert"
    done = subprocess.run([sys.executable, str(conv / "convert_lora_to_gguf.py"), "--base", str(base), "--outfile", str(tmp_path / "c.gguf"),
                           "--outtype", "f16", str(cell)], capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": f"{conv / 'gguf-py'}:{conv}"})
    assert done.returncode == 0, done.stderr[-800:]
    rebuilt, meta = gguf_to_peft.read(tmp_path / "c.gguf")
    assert set(rebuilt) == set(t) and meta["adapter.lora.alpha"] == 32.0
    cfg = gguf_to_peft.config(rebuilt, meta)
    assert cfg["r"] == 16 and cfg["layers_to_transform"] == [4, 5] and len(cfg["target_modules"]) == 7
    save_file(t, str(tmp_path / "real.safetensors"))
    rep = gguf_to_peft.compare(rebuilt, tmp_path / "real.safetensors")
    assert rep["ok"] and rep["max_rel_error_matrix"] < 1e-3 and rep["max_rel_error_edit"] < 1e-3  # float16 rounding, no more
    save_file(rebuilt, str(tmp_path / "rebuilt.safetensors"))
    assert set(merge.load(tmp_path / "rebuilt.safetensors")) == set(merge.load(tmp_path / "real.safetensors"))  # the trunk reads it


# --- eval_v2: one answer standard for every web kind, easy too (Gemini ruling on m_born / m_licence) --------------

def test_the_easy_web_kinds_refuse_a_dump_and_take_an_answer(monkeypatch, tmp_path):
    import random
    from training import hive_bench as B
    from training.kinds import m_born, m_csv, m_licence, m_pypi
    page = ('<html><body><div class="author-details"><span class="author-born-date">March 14, 1879</span> in Ulm '
            + " ".join(f"<a href='/page/{i}'>{i}</a>" for i in range(30)) + "</div></body></html>")
    monkeypatch.setattr(B, "_get", lambda url: page)
    check = m_born.m_born(random.Random(0))[3]
    assert check(None, "Born March 14, 1879.")[0]
    assert not check(None, page)[0]

    monkeypatch.setattr(B, "_hf", lambda model: {"cardData": {"license": "apache-2.0"}})
    check = m_licence.m_licence(random.Random(0))[3]
    assert check(None, "It is published under the Apache 2.0 licence (apache-2.0).")[0]
    card = '{"cardData": {"license": "apache-2.0"}, "siblings": [' + ",".join(f'{{"rfilename": "model-{i}.safetensors"}}' for i in range(12)) + "]}"
    assert not check(None, card)[0]

    class Resp:
        def json(self):
            return {"info": {"version": "2.3.1"}}
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    check = m_pypi.m_pypi(random.Random(0))[3]
    assert check(None, "The current version is 2.3.1.")[0]
    assert not check(None, "versions: " + ", ".join(f"2.{i}.{j}" for i in range(4) for j in range(3)) + ", 2.3.1")[0]

    from bs4 import BeautifulSoup
    listing = BeautifulSoup("".join(f'<article class="product_pod"><p class="price_color">£{10 + i}.50</p></article>' for i in range(5)), "html.parser")
    monkeypatch.setattr(B, "_category", lambda cat: listing)
    check = m_csv.m_csv(random.Random(0))[3]
    good = tmp_path / "good"; (good / "work").mkdir(parents=True)
    (good / "work" / "books.csv").write_text("title,price\n" + "".join(f"b{i},{10 + i}.50\n" for i in range(5)))
    monkeypatch.setattr(B, "files", lambda run, ext: sorted((Path(run) / "work").glob("*" + ext)))
    assert check(good, "")[0]
    bad = tmp_path / "bad"; (bad / "work").mkdir(parents=True)
    (bad / "work" / "books.csv").write_text(str(listing) + "<html>")
    assert not check(bad, "")[0]


# --- eval_v2 re-baseline (Gemini ruling): both verdicts from one sitting, the floor by delta -----------------------

def test_a_web_check_records_both_verdicts_on_the_same_answer(monkeypatch):
    check = _web_check(monkeypatch)
    ok, why = check(None, WIKIDATA_DUMP)
    assert not ok and "[eval_v1: pass; eval_v2: fail; kind: h_web]" in why  # the old check passed the dump
    ok, why = check(None, "499 km")
    assert ok and "[eval_v1: pass; eval_v2: pass; kind: h_web]" in why
    ok, why = check(None, "1310 km")
    assert not ok and "[eval_v1: fail; eval_v2: fail; kind: h_web]" in why


def _resit(tmp_path, strict_failures, loose_failures=0, n=32):
    """n easy tasks: the first `loose_failures` fail both ways, the next `strict_failures` are m_born dumps (v1 pass,
    v2 fail), the rest pass both; plus 4 hard tasks that must not count."""
    d = tmp_path / "resit"
    for i in range(n):
        if i < loose_failures:
            tag, ok = "[eval_v1: fail; eval_v2: fail; kind: m_born]", False
        elif i < loose_failures + strict_failures:
            tag, ok = "[eval_v1: pass; eval_v2: fail; kind: m_born]", False
        else:
            tag, ok = ("[eval_v1: pass; eval_v2: pass; kind: m_born]" if i % 4 == 0 else ""), True
        (d / f"e{i:02}").mkdir(parents=True)
        (d / f"e{i:02}" / "exam.json").write_text(json.dumps({"id": f"e2000081{i % 8 + 1}_{i // 8:02}", "ok": ok, "why": "x " + tag}))
    for i in range(4):
        (d / f"h{i}").mkdir(parents=True)
        (d / f"h{i}" / "exam.json").write_text(json.dumps({"id": f"h20000811_0{i}", "ok": False, "why": ""}))
    return preflight.exam_results([d])


def test_the_floor_follows_the_ruling_by_delta(tmp_path):
    import rebaseline
    e, floor = rebaseline.rebaseline(_resit(tmp_path / "a", 0))
    assert e["delta"] == 0 and floor["min"] == 30 and e["recovery_targets"] == [] and e["decision"] == "floor stays"
    e, floor = rebaseline.rebaseline(_resit(tmp_path / "b", 2))
    assert e["delta"] == 2 and floor["min"] == 28 and len(e["recovery_targets"]) == 2
    assert e["per_kind"]["m_born"]["v1"] - e["per_kind"]["m_born"]["v2"] == 2 and e["checker_effect"] == 2 and e["drift_new_compute"] == 0
    e, floor = rebaseline.rebaseline(_resit(tmp_path / "c", 2, loose_failures=1))
    assert e["decision"] == "STOP" and floor is None and e["drift_new_compute"] == 1 and e["checker_effect"] == 2
    e, floor = rebaseline.rebaseline(_resit(tmp_path / "d", 0, n=31))
    assert e["decision"] == "INVALID" and floor is None


def test_rebaseline_writes_the_bridge_and_the_floor_and_preflight_then_clears_it(tmp_path, monkeypatch):
    import rebaseline
    bridge, manifest = tmp_path / "g81.json", tmp_path / "g82.json"
    bridge.write_text((ROOT / "manifests" / "g81.json").read_text())
    manifest.write_text((ROOT / "manifests" / "g82.json").read_text())
    _resit(tmp_path, 1)
    assert rebaseline.main(["--results", str(tmp_path / "resit"), "--bridge", str(bridge), "--manifest", str(manifest)]) == 0
    b, m = json.loads(bridge.read_text())["eval_bridge"], json.loads(manifest.read_text())
    assert b["easy_v1_loose"]["passed"] == 32 and b["easy_v2_strict"] == {"passed": 31, "of": 32, "source": b["easy_v2_strict"]["source"]}
    assert b["eval_v2_floor"]["min"] == 29 and m["easy_floor"]["min"] == 29 and len(m["recovery_targets"]) == 1
    m["bridge"] = str(bridge)  # an absolute path: ROOT / it is it
    ok, rec = preflight.check(m, preflight.exam_results([tmp_path / "resit"]))
    assert rec["checks"][0]["check"] == "eval_v2_rebaseline" and rec["checks"][0]["ok"]
    assert ok  # 31 of 32 against a floor of 29


def test_the_g82_manifest_blocks_until_the_rebaseline_has_run():
    m = json.loads((ROOT / "manifests" / "g82.json").read_text())
    ok, rec = preflight.check(m, None)
    assert not ok and rec["checks"][0]["check"] == "eval_v2_rebaseline" and not rec["checks"][0]["ok"]


def test_offline_easy_answers_that_enumerate_are_measured_not_scored(tmp_path):
    import rebaseline
    d = tmp_path / "x"
    for tid, ans in (("e1_00", "It is a Tuesday."), ("e1_01", "Monday Tuesday Wednesday"), ("e1_02", " ".join(map(str, range(12)))), ("e1_03", "42")):
        (d / tid).mkdir(parents=True)
        (d / tid / "exam.json").write_text(json.dumps({"id": tid, "ok": True, "why": "", "answer": ans}))
    got = {x["id"] for x in rebaseline.exposure(preflight.exam_results([d]))}
    assert got == {"e1_01", "e1_02"}


def test_the_committed_quarantine_holds_both_r1_incidents():
    from training import quarantine
    rows = [json.loads(x) for x in quarantine.FILE.read_text().splitlines() if x.strip()]
    assert len(rows) == len(quarantine.keys()) == 14
    assert sorted(r["line"] for r in rows) == list(range(1068, 1076)) + list(range(2706, 2712))
    assert {r["seed"] for r in rows} == {10003306, 30004256}


# --- scripts/verify_store.py: every stored file against the extract-time record -----------------------------------

def test_verify_store_proves_each_kind_of_file_and_catches_a_mismatch(tmp_path, monkeypatch):
    import verify_store
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    from training import hf_store
    rec = json.loads((ROOT / "manifests" / "r1-files.json").read_text())
    want = {hf_store.path_in_store(f["store_name"]): f["sha256"] for a in rec["archives"].values() for f in a["files"]}
    json_path = next(p for p in want if p.endswith(".json"))
    body = b'{"a sidecar": 1}'
    monkeypatch.setitem(want, json_path, __import__("hashlib").sha256(body).hexdigest())
    monkeypatch.setattr(verify_store, "expected", lambda: dict(want))
    rebuilt_bytes = b"rebuilt adapter"
    rh = __import__("hashlib").sha256(rebuilt_bytes).hexdigest()

    def entry(path, oid):
        return RepoFile(path=path, size=1, blob_id="0" * 40, oid="0" * 40, lfs={"size": 1, "oid": oid, "pointerSize": 1} if oid else None)
    tree = [RepoFolder(path="g81", oid="0" * 40)] + [entry(p, None if p == json_path else h) for p, h in want.items()]
    tree += [entry(p, rh) for p in verify_store.REBUILT] + [entry(".gitattributes", None)]

    class Api:
        def list_repo_tree(self, *a, **k):
            return tree

    def download(repo_id, filename, repo_type, local_dir):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body if filename == json_path else b"gguf")
        return str(p)
    rebuild = lambda gguf, out: (Path(out).parent.mkdir(parents=True, exist_ok=True), Path(out).write_bytes(rebuilt_bytes))  # noqa: E731
    ok, res = verify_store.verify(Api(), "someone/store", download, rebuild=rebuild)
    assert ok and res[json_path] == ("PROVEN", "downloaded, SHA-256 = extract-time SHA-256")
    assert res["g39/t80.g39code.safetensors"][0] == "PROVEN" and res[".gitattributes"][0] == "HF_OWN"
    assert sum(s == "PROVEN" for s, _ in res.values()) == 86
    bad = next(p for p in want if p.endswith(".gguf") and "g80" in p)
    tree[[getattr(e, "path", "") for e in tree].index(bad)] = entry(bad, "f" * 64)
    ok, res = verify_store.verify(Api(), "someone/store", download, rebuild=rebuild)
    assert not ok and res[bad][0] == "MISMATCH"


def test_a_cell_rebuilt_twice_is_byte_identical(tmp_path):
    """What proves the stored rebuilt adapters: the rebuild is deterministic, so rebuilding from the store's proven
    .gguf must give the stored file's exact bytes."""
    import subprocess
    import numpy as np
    from safetensors.numpy import save_file
    from training import gguf_to_peft
    base, cell = tmp_path / "base", tmp_path / "cell"
    base.mkdir(); cell.mkdir()
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
        "intermediate_size": 192, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 256, "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False}))
    rng = np.random.default_rng(5)
    t = {f"base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": rng.standard_normal((16, 64)).astype(np.float32),
         f"base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": rng.standard_normal((64, 16)).astype(np.float32)}
    save_file(t, str(cell / "adapter_model.safetensors"))
    (cell / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj"], "layers_to_transform": [1], "base_model_name_or_path": str(base)}))
    conv = ROOT / "training" / "llama_cpp_convert"
    subprocess.run([sys.executable, str(conv / "convert_lora_to_gguf.py"), "--base", str(base), "--outfile", str(tmp_path / "c.gguf"),
                    "--outtype", "f16", str(cell)], check=True, capture_output=True, env={**__import__("os").environ, "PYTHONPATH": f"{conv / 'gguf-py'}:{conv}"})
    outs = []
    for i in range(2):
        subprocess.run([sys.executable, str(ROOT / "training" / "gguf_to_peft.py"), str(tmp_path / "c.gguf"), str(tmp_path / f"r{i}" / "a.safetensors")],
                       check=True, capture_output=True)
        outs.append((tmp_path / f"r{i}" / "a.safetensors").read_bytes())
    assert outs[0] == outs[1]
