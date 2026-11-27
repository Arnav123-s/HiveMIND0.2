import ast
import json
from pathlib import Path

import pytest

from team.tools import sandbox
from training import ast_triad as triad
from training import grow, lessons


def test_seeded_grammar_has_distinct_structures_and_bounded_loops():
    structures, nodes, reducers, sources = set(), set(), set(), set()
    for i in range(60):
        level = i % 3 + 1
        candidate = triad.generate(f"grammar:{i}", level)
        assert candidate == triad.generate(f"grammar:{i}", level)
        program, wrong, inputs = candidate
        assert 5 <= len(program.splitlines()) <= 15
        assert len(set(inputs)) == len(inputs) == 9 and program != wrong
        tree = ast.parse(program)
        for node in ast.walk(tree):
            nodes.add(type(node))
            if isinstance(node, ast.Constant):
                node.value = 0  # Count structures independently of the sampled literals.
        structures.add(ast.dump(tree))
        sources.add(program)
        for init in ("s = []", "s = {}", "s = set()"):
            if init in program:
                reducers.add(init)
        if level == 3:
            assert "return f(g(x))" in program and "return g(f(x))" in wrong
    assert {ast.If, ast.While, ast.For, ast.Mod, ast.Sub, ast.BitXor} <= nodes
    assert len(reducers) == 3 and len(sources) == 60 and len(structures) > 50


@pytest.fixture(scope="module")
def checked():
    return triad.verified("tested", 12)


def test_verified_examples_and_preferences_execute_against_their_checks(tmp_path, checked):
    # A fresh member workspace consumes the *emitted* code, including the induction program's unseen checks.
    cases = []
    for item in checked:
        pairs = [r for r in triad.rows(item) if r["type"] == "pair"]
        assert tuple(row["kind"] for row in pairs) == triad.KINDS
        assert not set(item["inputs"][:5]) & set(item["inputs"][5:])
        assert all(row["who"] == "lesson" and row["lesson"] == "ast_triad" for row in pairs)
        cases.append([item, pairs])
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    result = sandbox.run_python(r'''
import contextlib, io, json, re, sys
def fn(code):
    ns = {}
    exec(compile(code, "<check>", "exec"), ns)
    return ns["solve"]
def output(code):
    dest = io.StringIO()
    with contextlib.redirect_stdout(dest):
        exec(code, {})
    return dest.getvalue().rstrip("\n")
for case, rows in json.load(open("cases.json")):
    solve = fn(case["source"])
    for index, row in enumerate(rows):
        gold, bad = [row[key].split("```python\n")[1].rsplit("\n```", 1)[0] for key in ("output", "rejected")]
        if index == 0:
            assert int(output(gold)) == solve(case["x"]) != int(output(bad))
        elif index == 1:
            assert solve(int(output(gold))) == case["y"] != solve(int(output(bad)))
        elif index == 2:
            visible = list(zip(case["inputs"][:5], case["outputs"][:5]))
            hidden = list(zip(case["inputs"][5:], case["outputs"][5:]))
            assert str(visible) in row["task"] and str(hidden) not in row["task"]
            assert all(fn(gold)(x) == fn(bad)(x) == y for x, y in visible)
            assert all(fn(gold)(x) == y for x, y in hidden) and any(fn(bad)(x) != y for x, y in hidden)
        else:
            # Independent before-line observer stores whole snapshots. Derive one diff *after* execution.
            events = []
            def snapshot(frame, event, arg):
                if event == "line" and frame.f_code.co_filename == "<check>":
                    vals = {k:repr(v) for k,v in frame.f_locals.items() if isinstance(v, (int,list,set,dict))}
                    events.append((frame, frame.f_lineno, vals))
                return snapshot
            sys.settrace(snapshot)
            solve(case["x"])
            sys.settrace(None)
            full = []
            for i, (frame, line, vals) in enumerate(events):
                prev = next((v for f, _, v in reversed(events[:i]) if f is frame), {})
                diff = ",".join(f"{k}={vals[k]}" for k in sorted(vals) if vals[k] != prev.get(k))
                full.append([line, diff or "-"])
            assert case["trace"] == full[:40]
            k = int(re.search(r"before-line event (\d+)", row["task"]).group(1))
            line, delta = full[k-1]
            assert f"{line} | {delta}" in row["task"] and len(delta) <= 16
            answer = str(full[k][index-3])
            assert output(gold) == answer != output(bad)
print("all five question types and rejected programs checked")
''', tmp_path)
    assert result["returncode"] == 0, result["stderr"]
    assert result["stdout"].strip() == "all five question types and rejected programs checked"


def test_anti_shortcut_checks_require_hidden_witnesses(checked):
    for case in checked:
        assert 1 < len(set(case["outputs"])) and case["outputs"] != case["inputs"]
        if case["level"] == 3:
            code, mutant, inputs = triad.generate(case["seed"], 3)
            ns, wrong = {}, {}
            exec(code, ns)
            exec(mutant, wrong)
            for shortcut in (ns["f"], ns["g"], wrong["solve"]):
                assert any(shortcut(i) != ns["solve"](i) for i in inputs[5:])
    assert any(c["attempt"] > 1 for c in checked)


def test_trace_copies_mutable_values_and_stops_at_forty():
    ns = {}
    exec(compile("def f(x):\n    a=[]\n    for i in range(60):\n        a.append(i)\n    return len(a)", "<lesson>", "exec"), ns)
    value, trace = triad._observe(ns["f"], 4)
    assert value == 60 and len(trace) == 40
    assert trace[:5] == [(2, "x=4"), (3, "a=[]"), (4, "i=0"), (3, "a=[0]"), (4, "i=1")]


def test_second_sandbox_run_must_match(monkeypatch):
    calls = []
    def altered(code, workdir, **kw):
        calls.append(kw)
        (Path(workdir) / "results.json").write_text(str(len(calls)))
        return {"returncode": 0}
    monkeypatch.setattr(sandbox, "run_python", altered)
    with pytest.raises(ValueError, match="disagreed"):
        triad.verified("j2", 1)
    assert len(calls) == 2 and calls[0] == calls[1]


def test_teach_and_gather_only_emit_opted_in_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(lessons, "SOLVERS", {})
    args = dict(seed="hook", per_kind=1, levels={}, several=0, log=lambda *a, **k: None)
    for flag in (None, "0"):
        if flag is None:
            monkeypatch.delenv("HIVE_AST_TRIAD", raising=False)
        else:
            monkeypatch.setenv("HIVE_AST_TRIAD", flag)
        out = tmp_path / f"rows-{flag}"
        with monkeypatch.context() as m:
            m.setattr(triad, "verified", lambda *a: pytest.fail("off called the verifier"))
            report = lessons.teach(**args, out=out)
        assert out.read_bytes() == b"" and report["lessons"] == report["rows"] == 0
    monkeypatch.setenv("HIVE_AST_TRIAD", "1")
    out = tmp_path / "on"
    report = lessons.teach(**args, out=out)
    assert report["lessons"] == report["made"] == report["right"] == 15 and report["rows"] == 30
    assert report["dropped"] == [] and set(report["by_kind"]) == set(triad.KINDS)
    rows = tmp_path / "train.jsonl"
    assert grow.gather(rows, 83, new=[out, out]) == (0, 30)
    assert all(json.loads(line)["gen"] == 83 for line in rows.read_text().splitlines())
    assert not (set(triad.KINDS) | {"ast_triad"}) & lessons.K.HELD_OUT
    imported = {n.module for n in ast.walk(ast.parse(Path(triad.__file__).read_text())) if isinstance(n, ast.ImportFrom)}
    assert not any(name and name.startswith(("training.kinds", "training.exam")) for name in imported)


def test_teach_reports_failed_verification_and_writes_no_affected_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("HIVE_AST_TRIAD", "1")
    monkeypatch.setattr(lessons, "SOLVERS", {})

    def failed(seed, count):
        raise RuntimeError("sandbox exited")

    monkeypatch.setattr(triad, "verified", failed)
    out = tmp_path / "rows"
    result = lessons.teach(1, levels={}, several=0, out=out, log=lambda *a, **k: None)
    assert out.read_bytes() == b""
    assert result["made"] == 15 and result["right"] == 0
    assert result["failed_by_kind"] == {kind: {"verification failed": 3} for kind in triad.KINDS}
    assert result["dropped"] == ["AST verification: sandbox exited"]
