"""S1 (hours), part B: the gate's seats are split over jobs (fresh exam / held exam / talk exam) so that none runs 100
minutes (g51: after seat 2 sat the fresh exam 55 + the held exam 39). Same exams; the record's artifact names unchanged."""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "grow.yml").read_text())
GATE = FLOW["jobs"]["gate"]
MATRIX = GATE["strategy"]["matrix"]


def _step(start):
    return next(s for s in GATE["steps"] if s.get("name", "").startswith(start) or (s.get("with") or {}).get("name", "").startswith(start))


def test_the_matrix_has_every_seat_once_in_each_part():
    both = re.findall(r"'(\[.*?\])'", MATRIX["include"])  # a versus run's, any other run's
    vs, inc = json.loads(both[0]), json.loads(both[1])
    assert [i for i in vs if i not in inc] == [{"arm": "versus", "seed": s, "part": "held"} for s in range(1, 9)]  # versus sits no talk
    held = {(i["arm"], i["seed"]) for i in inc if i["part"] == "held"}
    assert held == {(a, s) for a in ("before", "after") for s in range(1, 9)} and len([i for i in inc if i["part"] == "held"]) == 16
    assert [i for i in inc if i["part"] == "talk"] == [{"arm": "after", "seed": 1, "part": "talk"}]
    assert MATRIX["part"] == ["fresh"]


def test_each_exam_is_sat_in_its_own_part_with_the_same_arguments():
    fresh = _step("A fresh exam")
    assert "matrix.part == 'fresh'" in fresh["if"] and "--level mixed --tasks 8" in fresh["run"]
    held = _step("The fixed held-out exam")
    assert "matrix.part == 'held'" in held["if"] and "--seat ${{ matrix.seed }} --seats 8 --per 3 --seconds 900" in held["run"]
    talk = _step("The talk exam")
    assert "matrix.part == 'talk'" in talk["if"] and "--who body --minutes 45 --per 120" in talk["run"]
    smoke = _step("Three plain messages")
    assert "matrix.part == 'fresh'" in smoke["if"] and "matrix.seed == 2" in smoke["if"]


def test_the_uploads_keep_the_names_the_record_reads():
    assert "matrix.part == 'fresh'" in _step("gate-${{ matrix.arm }}-${{ matrix.seed }}")["if"]
    assert "matrix.part == 'held'" in _step("held-${{ matrix.arm }}-${{ matrix.seed }}")["if"]
    assert "matrix.part == 'talk'" in _step("talk-${{ env.GEN }}")["if"]
    work = next(s for s in GATE["steps"] if "gate-work" in str((s.get("with") or {}).get("name", "")))
    # the held and talk jobs' work must not match the record's gate-*-[0-9] nor read_generation's gate-work-*
    assert "format('work-{0}', matrix.part)" in work["with"]["name"]


def test_a_versus_run_trains_both_arms_on_the_runners_and_sits_versus_on_the_held_exam():
    train = "\n".join(s.get("run", "") for s in FLOW["jobs"]["train"]["steps"])
    assert "${{ format('{0}', inputs.pairs) != '0' && 'x' || '' }}${{ inputs.slow }}${{ inputs.versus }}\" ]; then" in train  # no Colab when versus is set
    assert "matrix.arm != 'dna'" in _step("The fixed held-out exam")["if"] and "matrix.arm != 'dna'" in _step("held-${{ matrix.arm }}")["if"]


def test_the_line_trains_on_pairs_unless_pairs_is_0():
    # trial g65 (run 45014809204): with --pairs, held out 3 of 27 against 1 of 29, taught tasks equal (51 of 64 each)
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/grow.yml").read_text()
    # g67 (run 45073898716) trained without --pairs: dispatched with no inputs, `inputs.pairs != '0'` compared an
    # empty input as the number 0 and came out false. format() makes it a string first, as for inputs.loop.
    assert "${{ format('{0}', inputs.pairs) != '0' && '--pairs' || '' }}" in wf
    assert "inputs.pairs != '0'" not in wf and "inputs.pairs && '--pairs'" not in wf
