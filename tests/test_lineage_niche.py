import json

import pytest

from training import lineage


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_NICHE_LINEAGE", "1")
    monkeypatch.setattr(lineage, "BODY", tmp_path / "body.json")
    monkeypatch.setattr(lineage, "GENERATIONS", tmp_path / "generations.jsonl")
    return tmp_path / "niche.json"


def test_rarity_and_distinct_recorded_children_determine_parent_order(store):
    scores = {"g1code": {"common": 1, "rare": 1}, "g2code": {"common": 1, "rare": 0.6},
              "g3data": {"common": 1}, "g3words": {"common": 1}, "g4sources": {"common": 1}}
    body = {"dna": "t80", "generation": 5, "cells": list(scores), "parents": {"g4code": "g1code"}}
    record = {"generation": "g4", "dna": "t80", "parents": {"g4code": "g1code"}}
    history = [record, dict(record, generation="g3", parents={"g3code": "g1code"}),  # retired child
               dict(record, dna="t17", parents={"g4code2": "g2code"}),
               dict(record, trial=True, parents={"g4code2": "g2code"}),
               dict(record, generation="g6", parents={"g6code": "g2code"})]
    lineage.GENERATIONS.write_text("\n".join(map(json.dumps, history)) + "\n{")
    store.write_text(json.dumps({"t80": scores}))
    # common is fully masked. The rare-task mask is (1 - 1.6/5)^2, and the same child is counted just once.
    mask = (1 - 1.6 / 5) ** 2
    assert lineage._parent_scores("g6", "t80", body, scores) == pytest.approx(
        {"g1code": mask / 3, "g2code": 0.6 * mask, "g3data": 0, "g3words": 0, "g4sources": 0})
    for organs in (None, ["code"]):
        picks = lineage.plan("g6", "t80", body, organs=organs, steps={"g1code": 800})
        assert next(c["parent"] for c in picks if c["organ"] == "code") == "g2code"
    body["parents"].clear()
    lineage.GENERATIONS.unlink()
    assert lineage.plan("g6", "t80", body, organs=["code"])[0]["parent"] == "g1code"
    assert lineage.plan("g6", "t17", body, organs=["code"])[0]["parent"] == ""


def test_rare_task_veteran_survives_a_younger_redundant_child(store, monkeypatch):
    veteran, anchor, clone = "g1code", "g2code", "g80code"
    before = [veteran, anchor] + [f"g{i}words" for i in range(3, 25)]
    scores = {c: {f"task-{c}": 1} for c in before}
    scores[veteran] = {"unseen-grid": 1}
    scores[anchor] = dict.fromkeys("abcd", 1)
    scores[clone] = dict.fromkeys("abc", 1)
    original = json.dumps(scores)
    parents = {anchor: veteran, clone: anchor}
    lineage.BODY.write_text(json.dumps({"dna": "t80", "generation": 79, "cells": before, "parents": parents}))
    store.write_text(json.dumps({"t80": scores}))
    assert lineage.grown_body(before, [clone], parents) == (before, [clone])
    assert lineage.grown_body(before, [clone], parents, scores=scores) == (before, [clone])
    assert json.dumps(scores) == original
    monkeypatch.setenv("HIVE_NICHE_LINEAGE", "0")
    assert lineage.grown_body(before, [clone], parents, scores=scores) == (before[1:] + [clone], [veteran])


@pytest.mark.parametrize(("weaker", "stronger", "retired"), [(0.69, 1, "g1code"),
                                                            (0.70, 1, "g2code"),
                                                            (0.70, 0.70, "g1code")])
def test_overlap_threshold_and_strict_dominance(store, weaker, stronger, retired):
    scores = {"g1code": {"shared": 1}, "g2code": {"thin": weaker},
              "g3code": {"shared": 0.1, "other": 1}, "g4code": {"thin": stronger}}
    # At 70% overlap the weak leaf gives way. An equal peer or a looser overlap leaves the age fallback intact.
    assert lineage.grown_body(list(scores), [], {"g4code": "g3code"}, 3, scores)[1] == [retired]


def test_anchor_stays_even_if_a_different_organ_dominates_it(store):
    scores = {"g1code": {"a": 0.8}, "g2code": {"a": 0.75}, "g4data": {"a": 1}, "g8data": {"a": 0.8}}
    kept, retired = lineage.grown_body(list(scores)[:3], ["g8data"], {"g2code": "g1code"}, 2, scores)
    assert retired == ["g2code", "g8data"]
    assert kept == ["g1code", "g4data"]


def test_sole_solver_is_recomputed_after_each_eviction(store):
    scores = {"g1code": {"rare": 1}, "g2code": {"rare": 1}, "g4code": {"common": 1, "easy": 1},
              "g8code": {"common": 1}}
    kept, retired = lineage.grown_body(list(scores), [], most=2, scores=scores)
    assert retired == ["g1code", "g8code"]
    assert kept == ["g2code", "g4code"]


@pytest.mark.parametrize("parents", [{"g2code": "g1code"}, {"g2code": "g1code", "g3code": "g2code"}])
def test_oldest_carried_ancestor_is_last_when_no_cell_is_dominated(store, parents):
    scores = {"g1code": {"red": 1, "blue": 0.25}, "g2code": {"red": 0.3, "blue": 1},
              "g3code": {"red": 0.7, "blue": 0.7}}
    # g3code anchors the organ. Give up g2code before the oldest ancestor even if both have descendants.
    assert lineage.grown_body(list(scores), [], parents, 2, scores) == (["g1code", "g3code"], ["g2code"])


@pytest.mark.parametrize("contents", [None, "{", "[]", '{"t17":{"g1code":{"x":1}}}',
                                      '{"t80":{"g1code":{"x":1.1},"g2code":{"x":NaN}}}'])
def test_missing_or_unusable_profile_keeps_stock_plan_and_retirement(store, contents, monkeypatch):
    body = {"dna": "t80", "generation": 4, "cells": ["g1code", "g2code"], "parents": {"g2code": "g1code"}}
    lineage.BODY.write_text(json.dumps(body))
    if contents is not None:
        store.write_text(contents)
    def outputs():
        return (lineage.plan("g5", "t80"), lineage.plan("g5", "t80", organs=["code"]),
                lineage.grown_body(body["cells"], ["g5code"], body["parents"], 2))

    enabled = json.dumps(outputs())
    monkeypatch.delenv("HIVE_NICHE_LINEAGE")
    assert json.dumps(outputs()) == enabled


def test_trunk_and_record_agree_on_the_roster_from_the_optional_file(store):
    before = ["g1code", "g2code"] + [f"g{i}words" for i in range(3, 25)]
    body = {"dna": "t80", "generation": 78, "cells": before, "parents": {"g2code": "g1code"}}
    scores = {c: {c: 1} for c in before}
    scores.update(g2code=dict.fromkeys("abcd", 1), g80code=dict.fromkeys("abc", 1))
    store.write_text(json.dumps({"t80": scores}))
    lineage.BODY.write_text(json.dumps(body))
    kids = [c for c in lineage.plan("g80", "t80") if c["name"] == "g80code"]
    assert len(kids) == 1
    parents = dict(body["parents"], g80code=kids[0]["parent"])
    kept, retired = lineage.grown_body(before, ["g80code"], parents)
    assert retired == ["g80code"]
    leaves = lineage.leaves(kept, {c: p for c, p in parents.items() if c in kept})
    assert lineage.trunk_of("g80", "t80", iter(["g80code"])) == {"name": "g80trunk", "of": leaves, "older": leaves}


@pytest.mark.parametrize("flag", ["0", "1"])
def test_unprofiled_one_shot_iterators_keep_the_legacy_results(store, monkeypatch, flag):
    monkeypatch.setenv("HIVE_NICHE_LINEAGE", flag)
    body = {"dna": "t80", "generation": 4, "cells": []}
    assert lineage.grown_body(["g4code"], iter(["g1code", "g2code"]), most=2) == (["g4code", "g2code"], ["g1code"])
    assert lineage.trunk_of("g5", "t80", iter(["g5sources"]), body) == {
        "name": "g5trunk", "of": ["g5sources"], "older": ["g5sources"]}
