"""Run 45014809204 (a versus trial, 185 artifacts): the record's download-artifact@v4 pattern saw only the first 100,
downloaded 0 cells and the record failed. The record now pages the run's list itself (scripts/fetch_artifacts.py)."""
import fnmatch
import importlib.util
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_artifacts", ROOT / "scripts" / "fetch_artifacts.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def _record():
    return yaml.safe_load((ROOT / ".github" / "workflows" / "grow.yml").read_text())["jobs"]["record"]["steps"]


def test_the_record_reads_what_it_needs_by_paging_not_by_pattern():
    steps = _record()
    fetch = next(s for s in steps if s.get("name", "").startswith("Every artifact the record reads"))
    for want in ("'cell-*=got'", "'gate-*-[0-9]=gate'", "'held-*-[0-9]=held'", "'talk-smoke-*=smoke'", "'versus-*=vs'", "'trunk=tr'",
                 "'rows-*=prac'", "'practice-work-*=pw'"):
        assert want in fetch["run"]
    assert "fetch_artifacts.py" in fetch["run"] and fetch["env"]["GH_TOKEN"] == "${{ github.token }}"
    patterns = [(s.get("with") or {}).get("pattern") for s in steps if s.get("uses", "").startswith("actions/download-artifact")]
    assert patterns == [None] * len(patterns)  # no pattern download is left to see only the first 100
    assert steps.index(fetch) < next(i for i, s in enumerate(steps) if s.get("name", "").startswith("Each cell named"))


def test_the_listing_goes_through_every_page(monkeypatch):
    pages = {1: [{"id": i, "name": f"part-{i}"} for i in range(100)],
             2: [{"id": 200 + i, "name": f"cell-{i}"} for i in range(85)] + [{"id": 9, "name": "old", "expired": True}]}
    seen = []

    def api(path, binary=False):
        seen.append(path)
        return {"artifacts": pages.get(int(re.search(r"&page=(\d+)", path).group(1)), [])}

    monkeypatch.setattr(fa, "api", api)
    arts = fa.listing("1", "o/r")
    assert len(arts) == 185 and len(seen) == 2 and "per_page=100&page=2" in seen[1]
    got = fa.wanted(arts, [("cell-*", "got"), ("gate-*-[0-9]", "gate")])
    assert len(got) == 85 and {f for _, f in got} == {"got"}
    assert fnmatch.fnmatchcase("gate-after-3", "gate-*-[0-9]")


def test_api_tries_again_after_a_server_error(monkeypatch):
    """g75's record died on one 502 while downloading an artifact; a passing second try now carries on."""
    import subprocess
    calls = []

    def fake_run(cmd, capture_output):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, b"", b"HTTP 502")
        return subprocess.CompletedProcess(cmd, 0, b'{"ok": 1}', b"")

    monkeypatch.setattr(fa.subprocess, "run", fake_run)
    assert fa.api("x", wait=0) == {"ok": 1} and len(calls) == 2


def test_api_gives_up_after_its_tries(monkeypatch):
    import subprocess
    monkeypatch.setattr(fa.subprocess, "run", lambda cmd, capture_output: subprocess.CompletedProcess(cmd, 1, b"", b"x"))
    try:
        fa.api("x", tries=3, wait=0)
    except subprocess.CalledProcessError:
        return
    raise AssertionError("expected CalledProcessError")


def test_a_seat_that_never_downloads_is_warned_and_skipped(monkeypatch, tmp_path, capsys):
    import subprocess
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setattr(fa, "listing", lambda run, repo: [{"id": 1, "name": "gate-after-1"}])

    def fake_api(path, binary=False):
        raise subprocess.CalledProcessError(1, ["gh", "api", path])

    monkeypatch.setattr(fa, "api", fake_api)
    assert fa.main(["9", "gate-*-[0-9]=gate"]) == 0
    assert "::warning::gate-after-1" in capsys.readouterr().out
