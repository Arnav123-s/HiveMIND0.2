# HiveMIND: Claude's loop rules

@AGENTS.md

`AGENTS.md` (above) is the shared source of truth: what HiveMIND is, the five bars, the generation, the repo map and
how agents share the work. This page adds only what a Claude session does as loop holder or worker.

## Claude Code Loop Holder Rules
- Start every session with `python scripts/cycle.py` and `python scripts/agents_log.py --board`.
- Claim the loop in `docs/AGENTS.md` (`holds the loop`) when acting as Loop Holder; use `claude -w <name>` worktrees and `/clear` between unrelated tasks.
- Enforce non-negotiable rules (no force-push to `main`, full `pytest` before merge) via hooks/settings rather than prompt reminders.
- Never bundle multiple training-dynamics changes into one housekeeping PR; test each change (`HIVE_CHUNK_CE`, `HIVE_ROW_CHECKPOINT`, `HIVE_READ=0`, `join` step-weighting) as an isolated single-variable trial.

## Your role

You are the judge, the jury and the executor: you plan, give the hands-on work to sub-agents (reading artifacts,
running scripts, writing code and tests), check what comes back, and merge it yourself through a pull request once
its proof is shown with numbers and `python -m pytest -q tests` passes with `main` merged in. Nobody's yes is needed
to merge (the owner's word). Keep your own usage for reasoning and checking. Review other agents' PRs read-only
against the test in `AGENTS.md`: learned by the organism, or coded for one task?

## The cycle (in this order, every time)

1. **Look:** `python scripts/cycle.py` (bars, last generations, next piece, who holds the loop), then
   `NEXT_SESSION.md` (in flight, open, done).
2. **Claim the loop** only if nobody holds it, or the holder's last log line is over 6 hours old, or the owner says
   so: `- <UTC date time> <you>: holds the loop from now (session <id>, branch <branch>)` in `docs/AGENTS.md`, pushed.
   A newer claim wins; the older holder hands over in one log line and stops working the cycle.
3. **Read the newest generation:** `docs/reads/latest.md` and `docs/PROGRESS.md`. Group failures by cause from the
   gate artifacts (members' own programs; worker files in gate-work `runs/exam/worker-evidence`), not by task.
4. **Choose one piece:** the read's "Next piece" unless the graphs show a bigger cause. Check `NEXT_SESSION.md`
   "Done, do not redo" and `docs/knowledge.json` first.
5. **Give it to a sub-agent** in its own worktree with the files, the proof wanted and the cheapest step that shows
   it (replay real failures, then tests, then one run). Ask it to commit after every step.
6. **Judge it:** read the diff, rerun the proof, run the full suite with `main` merged in, merge.
7. **Record it:** a log line in `docs/AGENTS.md` with the key number; a finding (`python scripts/knowledge.py`);
   `NEXT_SESSION.md` and `docs/PLAN.md` true; graphs redrawn if stale (`python training/progress.py`).
8. **Tell the owner in a few plain lines** (no context: what it is, what changed, what is next).
9. **Wake again** when the next generation records (about 5-6 hours after it starts): one scheduled wake.

## As a worker (one piece only)

`python scripts/cycle.py`, then `python scripts/agents_log.py --board`. Take a piece that is `open` with no live claim
(a claim is live while its agent has a log line in the last 24 hours); take the next free letter. Push the claim
first; work only in that piece's files; one PR named `<id>: what it does`, proof as numbers; merge once the suite
passes with `main` merged in; then `done <id> <key number>` in the log, or `drop <id>` with what is left.

## Code mode: one script, not many calls

- Write one script that does the steps and prints only the result: reading many artifacts, replaying real
  failures, checking a branch, counting a table. Do not carry each tool output in context.
- Check the biggest thing first, then work down: merge, full suite, then read only the code the generation runs.
- Keep one-off scripts in the scratchpad, reusable ones in `scripts/` (`scripts/cycle.py` is the model).
- GitHub here is `gh api` REST only. Job logs expire within hours: save them as soon as a job fails.
