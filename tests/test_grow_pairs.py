import json

from training import grow


def _member(home, n, first, final, answer):
    msgs = [{"role": "system", "content": "orders. Today is 2026-11-14 (Friday)."}, {"role": "user", "content": "TASK: add"},
            {"role": "assistant", "content": f"```python\n{first}\n```"}]
    if final != first:
        msgs += [{"role": "user", "content": "It failed."}, {"role": "assistant", "content": f"```python\n{final}\n```"}]
    (home / f"member{n}.json").write_text(json.dumps({"messages": msgs, "program": final, "answer": answer}))


def test_pairs_set_the_agreed_program_beside_first_tries_that_were_wrong(tmp_path):
    home = tmp_path / "members"
    home.mkdir()
    _member(home, 1, "print('2'+'3')", "print(2+3)", "5")   # wrong first, then right
    _member(home, 2, "print(2+3)", "print(2+3)", "5")       # right at once: no pair from it
    _member(home, 3, "print(6)", "print(6)", "6")           # disagreed: its program is a wrong one
    (home / "members.json").write_text(json.dumps({"agreed": True, "agreeing": [1, 2]}))
    rows = grow.pair_rows(tmp_path, "add")
    assert [r["rejected"] for r in rows] == ["```python\nprint('2'+'3')\n```", "```python\nprint(6)\n```"]
    assert all(r["output"] == "```python\nprint(2+3)\n```" and r["type"] == "pair" for r in rows)
    assert rows[0]["messages"][1]["content"] == "TASK: add" and "2026-11-14" in rows[0]["messages"][0]["content"]


def test_no_pairs_where_nobody_agreed(tmp_path):
    home = tmp_path / "members"
    home.mkdir()
    _member(home, 1, "print(1)", "print(1)", "1")
    (home / "members.json").write_text(json.dumps({"agreed": False, "agreeing": []}))
    assert grow.pair_rows(tmp_path, "add") == []


def test_sequential_pair_admission_and_projected_memory():
    cap = 1152
    assert grow.pair_pieces(600, 552, False) == cap
    assert grow.pair_pieces(601, 552, False) > cap
    assert grow.pair_pieces(1152, 940, True) == cap
    assert grow.pair_pieces(940, 1153, True) > cap
    # The last unchunked answer can project one more row than a longer chunked one.
    project = lambda n: n + 1 if n <= 64 else 64
    assert grow.row_head_bytes(100, 64, 100, project, orpo=True) == 65 * 100 * 16
    # DPO: the unchunked chosen graph keeps its full projection; the vocab-tiled rejected side keeps no tile (PR 167)
    assert grow.row_head_bytes(100, 64, 100, project) == (65 + 65) * 100 * 16


def test_orpo_matches_joint_autograd_with_accumulation_and_checkpointing():
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    chosen_ids = torch.tensor([[1, 2, 3], [3, 3, 1]])
    rejected_ids = torch.tensor([[2, 4, 3, 1], [1, 7, 4, 5]])
    expected, actual = [[torch.nn.Parameter(torch.tensor(v, dtype=torch.double))
                         for v in (0.2, 0.5, -0.3, 1.0)] for _ in range(2)]
    for params in (expected, actual):
        params[0].grad = torch.tensor(0.11, dtype=torch.double)
    events = []

    def score(params, ids, k, observe=False):
        side = 1 if ids is chosen_ids else 2
        def nll(shared, local):
            return F.softplus(shared * ids[:, -k:] + local).sum()
        if observe:
            events.append(("forward", side))
            result = checkpoint(nll, params[0], params[side], use_reentrant=True)
            result.register_hook(lambda _: events.append(("backward", side)))
            return result
        return nll(params[0], params[side])

    # Unequal answer lengths and batch size two exercise mean answer log probability.
    chosen = score(expected, chosen_ids, 1) / 2
    rejected = score(expected, rejected_ids, 3) / 6
    odds = lambda nll: -nll - torch.log1p(-torch.exp(-nll))
    joint = chosen - 0.1 * F.logsigmoid(odds(chosen) - odds(rejected))
    (joint / 4).backward()
    plain, loss = grow._orpo_backward(lambda x, k: score(actual, x, k, True),
                                      chosen_ids, 1, rejected_ids, 3, actual, 4)
    assert abs(plain - chosen.item()) < 1e-12 and abs(loss - joint.item()) < 1e-12
    assert events == [("forward", 1), ("backward", 1), ("forward", 2), ("backward", 2)]
    for a, b in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    assert actual[3].grad is None


def test_orpo_saturated_answer_probabilities_have_finite_gradients():
    import math
    import torch

    ids = torch.ones((1, 2), dtype=torch.long)
    for first, second in ((0., 0.), (0., 1000.), (1000., 0.), (1e-8, 2e-8)):
        params = [torch.nn.Parameter(torch.tensor(value)) for value in (first, second)]
        scores = iter(params)
        _, loss = grow._orpo_backward(lambda x, k: next(scores) * 1, ids, 1, ids, 1, params, 1)
        assert math.isfinite(loss) and all(torch.isfinite(p.grad) for p in params)


def test_train_switches_pair_admission_and_reference_passes(tmp_path, monkeypatch):
    from team import solo
    from test_held_out import _tiny, _train

    dna = tmp_path / "dna"
    dna.mkdir()
    tok = _tiny(dna)
    messages = [{"role": "system", "content": solo.SHORT.format(today="2026-11-25 (Wednesday)")},
                {"role": "user", "content": "TASK: add"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    length = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    short = {"type": "pair", "task": "add", "kind": "math", "messages": messages,
             "output": "1", "rejected": "2"}
    cap = 2 * (length(prompt) + length("1<|im_end|>\n"))
    rows = [short, dict(short, rejected="2 " * 8), dict(short, rejected="2 " * (cap + 1))]
    for flag, used in (("0", 1), ("1", 2)):
        with monkeypatch.context() as patch:
            patch.setenv("HIVE_PAIRS_ORPO", flag)
            patch.setenv("HIVE_GPU", "0")
            meta, passes = _train(tmp_path, patch, rows, "", max_len=cap, pairs=True, checkpoint=False)
        assert meta["used"] == used and meta["pairs"] == meta["passes"] > 0
        assert sum(grad for _, grad in passes) == 2 * meta["passes"]
        assert sum(not grad for _, grad in passes) == (2 if flag == "0" else 0) * meta["passes"]
