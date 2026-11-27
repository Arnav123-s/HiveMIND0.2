"""The three CPU speed switches (PRs 167, 171, 181) as versus flags: off unless given, and accepted by the workflow
and the record. A trial passes them through $VERSUS, placed first on the trainer's command line."""
import re
from pathlib import Path

import pytest

from training import grow, grow_record

ROOT = Path(__file__).resolve().parent.parent


def _workflow_flags():
    text = (ROOT / ".github/workflows/grow.yml").read_text()
    return re.search(r'\[\[ "\$VERSUS" =~ \^(.*?)\$ \]\]', text).group(1)


@pytest.mark.parametrize("flags", ["--packed-dna", "--train-batch 4", "--vocab-ce", "--packed-dna --vocab-ce", "--orpo --train-batch 2"])
def test_the_workflow_and_the_record_accept_the_speed_flags(flags):
    for pat in (_workflow_flags(), grow_record.FLAGS):
        assert re.fullmatch(pat, flags) or re.match(pat + "$", flags), (pat, flags)


def test_without_flags_nothing_is_set_and_the_trainer_keeps_its_defaults(monkeypatch):
    env = {}
    assert grow.speed_flags(["train", "--pairs"], env) is None and env == {}
    for name in ("HIVE_TERNARY_VNNI", "HIVE_TRAIN_BATCH"):
        monkeypatch.delenv(name, raising=False)
    seen = {}
    monkeypatch.setattr(grow, "train", lambda *a, **k: seen.update(k) or {})
    grow.main(["train", "--pairs"])
    assert seen["loss_chunk_tokens"] is None
    import os
    assert "HIVE_TERNARY_VNNI" not in os.environ and "HIVE_TRAIN_BATCH" not in os.environ


def test_each_flag_switches_its_own_path(monkeypatch):
    env = {}
    assert grow.speed_flags(["--packed-dna", "--train-batch", "4", "--vocab-ce"], env) == grow.VOCAB_CE_TOKENS
    assert env == {"HIVE_TERNARY_VNNI": "1", "HIVE_TRAIN_BATCH": "4"}
    with pytest.raises(ValueError):
        grow.speed_flags(["--train-batch", "3"], {})
    with pytest.raises(ValueError):
        grow.speed_flags(["--train-batch"], {})
    seen = {}
    monkeypatch.setattr(grow, "train", lambda *a, **k: seen.update(k) or {})
    monkeypatch.setattr(grow.os, "environ", {})
    grow.main(["train", "--vocab-ce"])
    assert seen["loss_chunk_tokens"] == grow.VOCAB_CE_TOKENS
    grow.main(["train", "--vocab-ce", "--loss-chunk-tokens", "128"])
    assert seen["loss_chunk_tokens"] == 128  # an explicit threshold wins
