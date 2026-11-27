"""A versus trial may set `--pull` (how far the child starts toward the trunk): the workflow and the record accept it,
and the trainer reads the first one given, which is the versus flag since $VERSUS comes first."""
import re

from training import grow, grow_record


def _workflow_flags():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / ".github/workflows/grow.yml").read_text()
    return re.search(r'\[\[ "\$VERSUS" =~ \^(.*?)\$ \]\]', text).group(1)


def test_versus_flag_lists_accept_pull_and_reject_unknown_flags():
    for pat in (_workflow_flags(), grow_record.FLAGS):
        for ok in ("--pull 0.1", "--pull 0", "--pairs --pull 0.1", ""):
            assert re.fullmatch(pat, ok) or re.match(pat + "$", ok), (pat, ok)
        assert not (re.fullmatch(pat, "--bogus 1") or re.match("^" + pat.lstrip("^") + "$", "--bogus 1"))


def test_train_gets_the_versus_pull_placed_first(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(grow, "train", lambda *a, **k: seen.update(k) or {})
    grow.main(["train", "--pull", "0.1", "--parent-cell", "g1code", "--trunk", "t.safetensors"])
    assert seen["pull"] == 0.1
    grow.main(["train"])
    assert seen["pull"] == 0.1  # the line's default since trial g77 (run 45460563586): 55 vs 52 of 64 against 0.5, 0 regressions


def test_a_versus_trial_may_pass_orpo_and_the_line_stays_on_dpo(monkeypatch):
    """PR 176's two-pass ORPO is off by default; a versus trial switches it on with --orpo (HIVE_PAIRS_ORPO unset)."""
    for pat in (_workflow_flags(), grow_record.FLAGS):
        assert re.fullmatch(pat, "--orpo") or re.match(pat + "$", "--orpo")
    monkeypatch.delenv("HIVE_PAIRS_ORPO", raising=False)
    seen = {}
    monkeypatch.setattr(grow, "train", lambda *a, **k: seen.update(k) or {})
    grow.main(["train", "--orpo", "--pairs"])
    assert seen["orpo"] is True and seen["pairs"] is True
    grow.main(["train", "--pairs"])
    assert seen["orpo"] is None  # falls back to HIVE_PAIRS_ORPO, off by default
