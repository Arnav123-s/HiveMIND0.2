"""g78's record (run 45521871368) died on `file_count limited to 1000 assets per release`: hive-grown was full. Cells
now go to the first release with room, a re-run replaces a file where it is, and every read tries the releases in order."""
from pathlib import Path

import pytest
import yaml

from training import release_store as rs

ROOT = Path(__file__).resolve().parents[1]


def _held(first, second=None):
    held = {t: None for t in rs.tags()}
    held["hive-grown"], held["hive-grown-2"] = first, second
    return held


def test_a_full_first_release_sends_new_cells_to_the_next_one():
    full = {f"t80.g{i}.gguf" for i in range(1000)}
    assert rs.place(["got/t80.g78code.gguf", "got/t80.g78code.json"], _held(full)) == {
        "hive-grown-2": ["got/t80.g78code.gguf", "got/t80.g78code.json"]}


def test_a_name_already_held_is_replaced_where_it_is_even_in_a_full_release():
    full = {f"t80.g{i}.gguf" for i in range(999)} | {"t80.g78code.gguf"}
    assert rs.place(["got/t80.g78code.gguf"], _held(full)) == {"hive-grown": ["got/t80.g78code.gguf"]}


def test_the_last_free_place_is_counted_once():
    nearly = {f"t80.g{i}.gguf" for i in range(999)}
    assert rs.place(["a.gguf", "b.gguf"], _held(nearly)) == {"hive-grown": ["a.gguf"], "hive-grown-2": ["b.gguf"]}


def test_all_releases_full_is_said():
    held = {t: {f"{t}.{i}" for i in range(1000)} for t in rs.tags()}
    with pytest.raises(SystemExit, match="full"):
        rs.place(["a.gguf"], held)


def test_a_read_tries_the_releases_in_order(tmp_path):
    tried = []

    def fetch(url, out):
        tried.append(url)
        if "/hive-grown-2/" not in url:
            raise OSError("404")
        Path(out).write_text("w")

    assert rs.get("t80.g78code.gguf", tmp_path / "c", "o/r", fetch=fetch)
    assert tried == ["https://github.com/o/r/releases/download/hive-grown/t80.g78code.gguf",
                     "https://github.com/o/r/releases/download/hive-grown-2/t80.g78code.gguf"]
    assert (tmp_path / "c").read_text() == "w"


def test_a_file_in_no_release_leaves_nothing(tmp_path):
    def fetch(url, out):
        Path(out).write_text("partial")
        raise OSError("404")

    assert not rs.get("x.gguf", tmp_path / "x", "o/r", fetch=fetch)
    assert not (tmp_path / "x").exists()


def test_the_workflow_reads_and_writes_cells_through_the_store():
    text = (ROOT / ".github" / "workflows" / "grow.yml").read_text()
    yaml.safe_load(text)
    assert "releases/download/hive-grown/$DNA." not in text  # no cell read bypasses the store
    assert 'release_store.py get "$DNA.$P.safetensors" parent.safetensors' in text
    assert 'release_store.py get "$DNA.$T.safetensors" trunk.safetensors' in text
    assert "release_store.py put $(ls got/$DNA.${GEN}*.gguf" in text
    assert "gh release upload hive-grown got/rows.jsonl" in text
