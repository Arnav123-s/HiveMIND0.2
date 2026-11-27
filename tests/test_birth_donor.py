"""Step-zero cross-organ inheritance and a fixed sibling's penalty in the real trainer."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from training import grow, merge
from test_held_out import _lesson, _train


def history():
    records = []
    for i in (7, 8):
        arms = {}
        for arm, solved in (("before", 3), ("after", 2)):
            arms[arm] = {"by_lean": {
                "code": {"organ": {"code": [solved, 6]}, "kind": {"table": [0, 4]}},
                "data": {"organ": {"code": [2, 3]}, "kind": {"table": [2, 4]}},
                "words": {"organ": {}, "kind": {"table": [1, 4]}}}}
        records.append({"generation": f"g{i}", "dna": "t17", "arms": arms})
    return records


def test_donor_requires_two_consecutive_flat_paired_exams_and_an_active_other_organ():
    cells = ["g8words", "g8code", "g7data", "g8data", "g8trunk", "g9data"]
    records = history()
    assert grow.choose_donor("code", records, reversed(cells)) == "g8data"
    assert grow.choose_donor("code", records, ["g8code", "g8trunk"]) is None
    assert grow.choose_donor("code", records[1:], cells) is None
    assert grow.choose_donor("code", {"only": "one", "snapshot": [1, 3]}, cells) is None
    assert grow.choose_donor("code", [records[0] | {"generation": "g6"}, records[1]], cells) is None
    assert grow.choose_donor("code", [records[0] | {"trial": True}, records[1]], cells) is None
    assert grow.choose_donor("code", [records[0] | {"dna": "t80"}, records[1]], cells) is None
    for score in ([4, 6], [6, 6], [0, 0], [-1, 6], [2]):
        records[0]["arms"]["after"]["by_lean"]["code"]["organ"]["code"] = score
        assert grow.choose_donor("code", records, cells) is None


def test_three_weight_cut_matches_dense_optimum_for_mixed_ranks_and_partial_modules(tmp_path):
    rng = np.random.default_rng(73)
    edits, paths = [], []
    for i, rank in enumerate((2, 4, 3)):
        edit = {"common": (rng.normal(size=(rank, 8)).astype("f"), rng.normal(size=(7, rank)).astype("f"))}
        if i == 2:
            edit["donor_only"] = edit["common"]
        path = tmp_path / f"{i}.safetensors"
        merge.save(edit, path)
        edits.append(edit)
        paths.append(path)
    result, least = merge.birth_mix(*paths, pull=.1, donor_pull=.1, rank=2)
    fractions = []
    for module, (a, b) in result.items():
        dense = sum(w * (e[module][1].astype("d") @ e[module][0].astype("d"))
                    for w, e in zip((.8, .1, .1), edits) if module in e)
        u, s, vh = np.linalg.svd(dense, full_matrices=False)
        fractions.append(sum(s[:2] ** 2) / sum(s ** 2))
        np.testing.assert_allclose(b @ a, (u[:, :2] * s[:2]) @ vh[:2], rtol=1e-5, atol=1e-6)
    assert least == pytest.approx(min(fractions), rel=1e-10)
    with pytest.raises(ValueError, match="weights"):
        merge.birth_mix(*paths, pull=.8, donor_pull=.3, rank=2)
    with pytest.raises(ValueError, match="missing"):
        merge.birth_mix("", paths[1], paths[2], pull=.1, donor_pull=.1, rank=2)
    invalid = tmp_path / "invalid.safetensors"
    save_file({"other.weight": torch.ones(2)}, invalid)
    with pytest.raises(ValueError, match="no LoRA factor pairs"):
        merge.birth_mix(paths[0], paths[1], invalid, pull=.1, donor_pull=.1, rank=2)


def test_sibling_gradient_matches_autograd_with_an_existing_gradient_and_unequal_ranks():
    generator = torch.Generator().manual_seed(3)
    b = torch.randn(19, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    sibling = torch.randn(19, 6, generator=generator, dtype=torch.float64)
    frozen_copy = sibling.clone()
    w = .07 * 2 / 3
    grad, = torch.autograd.grad(w * (b.T @ sibling).square().sum(), b)
    b.grad = torch.ones_like(b)
    grow._add_sibling_gradient([(b, sibling)], w)
    torch.testing.assert_close(b.grad, 1 + grad)
    assert torch.equal(sibling, frozen_copy)
    assert sibling.grad is None


@pytest.mark.parametrize("batch_size", [1, 4])
def test_step_zero_mixture_and_partial_window_penalty_in_real_training(tmp_path, monkeypatch, batch_size):
    rows = [_lesson("code", i, who="member")[0] for i in range(3)]
    # The shared helper uses PEFT on a real local tiny Qwen and stubs just the external GGUF converter.
    workspace = tmp_path / "inputs"
    (workspace / "cells").mkdir(parents=True)
    (workspace / "training/reports").mkdir(parents=True)
    records = history()
    history_file = workspace / "training/reports/generations.jsonl"
    history_file.write_text("".join(json.dumps(x) + "\n" for x in records))
    (workspace / "cells/body.json").write_text(json.dumps({"dna": "t17", "cells": ["g8code", "g8data"]}))
    monkeypatch.setattr(grow, "ROOT", workspace)
    monkeypatch.setenv("HIVE_TRAIN_BATCH", str(batch_size))
    monkeypatch.setenv("HIVE_SIBLING_CIS", "0")
    monkeypatch.setenv("HIVE_DONOR_PULL", "0")
    options = dict(layers="last1", checkpoint=False, limit=3)
    with monkeypatch.context() as patch:
        _train(tmp_path, patch, rows, "", **options)
    stock = tmp_path / "grow/t17+kb/adapter_model.safetensors"
    sd = load_file(stock)
    parent, trunk, donor = [workspace / f"t17.g8{name}.safetensors" for name in ("code", "trunk", "data")]
    for scale, path in ((1, parent), (.5, trunk), (2, donor)):
        save_file({k: val * (scale if k.endswith(".lora_B.weight") else 1.) for k, val in sd.items()}, path)
    assert grow._birth_donor(workspace / "missing.safetensors", "code", "t17", "g9") is None
    sib = workspace / "sibling.safetensors"
    save_file({k: v for k, v in sd.items() if k.endswith(".lora_B.weight")}, sib)
    with monkeypatch.context() as patch:
        off, _ = _train(tmp_path, patch, rows, "", donor=workspace / "absent", sibling=workspace / "absent", **options)
    assert all(torch.equal(sd[k], v) for k, v in load_file(stock).items())
    assert "donor" not in off and "sibling" not in off
    monkeypatch.setenv("HIVE_DONOR_PULL", ".1")
    monkeypatch.setenv("HIVE_SIBLING_CIS", ".3")
    calls = []
    add = grow._add_sibling_gradient
    def spy(pairs, weight):
        calls.append((len(pairs), weight))
        add(pairs, weight)
    with monkeypatch.context() as patch:
        patch.setattr(grow, "_add_sibling_gradient", spy)
        # Three presentations with accum=2 produce one whole and one partial window.
        original_train = grow.train
        def one_epoch(*a, **kw):
            return original_train(*a, **(kw | {"epochs": 1.0}))
        patch.setattr(grow, "train", one_epoch)
        meta, _ = _train(tmp_path, patch, rows, "", lean="code", gen="g9", parent=parent, trunk=trunk,
                         donor=workspace, sibling=sib, **options)
    assert calls == [(7, .3), (7, .15)]
    if batch_size == 4:
        assert meta["sft_batches_by_size"] == {"2": 1, "1": 1}
    assert meta["steps"] == 2 and meta["sibling_modules"] == 7 and meta["donor"] == donor.name
    expected, kept = merge.birth_mix(parent, trunk, donor, .1, .1, 4)
    assert meta["least_energy_kept"] == kept
    merged = merge.load(tmp_path / "grow/t17+kb.start.safetensors")
    for key, (a, b) in expected.items():
        np.testing.assert_array_equal(merged[key][0], a)
        np.testing.assert_array_equal(merged[key][1], b)
    records[-1]["arms"]["after"]["by_lean"]["code"]["organ"]["code"] = [4, 6]
    history_file.write_text("".join(json.dumps(x) + "\n" for x in records))
    assert grow._birth_donor(donor, "code", "t17", "g9") is None
    assert grow._birth_donor(donor, "code", "t17", "g7") is None


def test_cli_forwards_donor_and_sibling(monkeypatch):
    got = {}
    def train(*a, **kw):
        got.update(kw)
        return {}
    monkeypatch.setattr(grow, "train", train)
    grow.main(["train", "--donor", "d", "--donor-pull", ".1", "--sibling", "b"])
    assert (got["donor"], got["donor_pull"], got["sibling"]) == ("d", .1, "b")
