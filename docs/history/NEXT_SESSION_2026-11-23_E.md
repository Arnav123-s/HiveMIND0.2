# NEXT SESSION: the one current handoff

Written 2026-11-23 15:50 UTC by E (session `session_014dX1CQ9f1aJvPqnqdCbRfD`), who hands over the loop. How to work is
in `CLAUDE.md`; the story is in `docs/OVERVIEW.md`. Start with `python scripts/cycle.py`. GitHub here: `gh api` REST
only (the GitHub MCP tools are retired). Older handoffs: `docs/history/`.

## Where the hive stands (g70, recorded 2026-11-23 14:36 UTC)

- **g70 kept**, 52 against 48: hard 0.81 (met), talk 8.0 (met), fixed held-out 4 of 29, 2 of 10 kinds (best so
  far), first try 0.60 (new 0.44), 5.14 h. Two of five bars met. Not met: held-out (bar half of 10 kinds), first try
  (0.7), hours (3).
- **Stage 1A (--pairs) is on and stays on:** g69 dropped 49/50 (held-out level), g70 kept 52/48 (held-out up).
- **PR 141 worked:** 0 of 20 slices lost (2-4 every generation before), and the scope started on the runners.

## In flight

| What | Where | Note |
|---|---|---|
| g71, with pairs | run 45196041785, started 14:36 UTC 23 Nov, training at 15:40 | record about 19:45. Nobody holds the loop: E handed over, no wake is set |
| Stage 1B trial, `--pull 0.1` | ready (PRs 146, 152, 156) | at g71's record: cancel g72 before its train jobs, then `gh api -X POST repos/arnav123-s/hivemind/actions/workflows/grow.yml/dispatches -f ref=main -f 'inputs[versus]=--pull 0.1' -f 'inputs[pairs]=1'`; restart the line with `-f 'inputs[pairs]=1'` when it records. Judge per arm: delivered, member, not passed, held-out, regressions vs after |
| The researcher side's PR 158 | open, reviewed read-only: not yet | general, off by default; no scored result (its exam runs failed the hardware check) |

## Open, in order

0. **The architecture comes first now (owner, 21-22 Nov):** `AGENTS.md`, `docs/TRUE_ARCHITECTURE.md` (stages) and
   `docs/RESEARCH_SYNTHESIS.md` (the owner's full vision and the research behind it). Stage 1A `--pairs`
   is done (trial g65; on for the line, PR 153). 1B is `--pull 0.5 -> 0.0/0.1`, which needs `pull` added to versus's flag list in grow.yml. 1C is
   niche survival. No hand-written task helpers (PR 143 is a reference only). D's single-change branches, each to be
   judged as its own trial, not merged together: `claude/blissful-mccarthy-32ekqy-chunk` (HIVE_CHUNK_CE under auto),
   `-rowckpt`, `-read` (HIVE_READ/JUDGED default 0), `-join` (slices weighed by steps). Merge main into each and run
   the full suite first.

1. **Hours (S1).** 5.3 h at g62 (from 5.1-6.7, now no slice lost). PR 138 capped the measuring before training (it took 92 of 170 min); train is ~175 min
   when no slice is lost, gate ~55-90. Next: if more training time per slice does not lift hard kinds, cut the
   170-minute budget; practice 26-53 and dream ~60 run side by side.
2. **Held-out kinds (H1).** 0-1 of 10 kinds for weeks. g51 study: 16 of 27 failures near (shape, pasted template,
   overwritten value; general repairs merged, PRs 117, 125), 11 far (no grasp of the method). Exam workers' own
   files are now kept (PR 129: gate-work `runs/exam/worker-evidence`): read them by cause first.
3. **First try on new kinds (F1).** Old kinds ~0.9, new (K2) 0.28-0.52. First try counts one program, no hint
   (`scripts/selfreliance.one`): only lessons move it, not repairs. Compound tasks are solved part by part (PR 126).
4. Board rows in `docs/PLAN.md`: Y1, A1, G1, C1 (lru rows still over 508 pieces), J1, D1, P1, W2, R3, M2, T3.

## Researcher side (other agent, same account)

Judged, not taken into the line: PR 122 (a one-kind answer key), 131 (merged anyway from the owner's account;
checked: inert by default, tests pass), 135 and 136 (their own numbers show no held-out gain: 2/30 and 4/30 against
3/30). Research only: 123, 132-134. Taken: PR 124 (worker files kept). A trial run on the shared 20 runners stalls
the line (run 58 was cancelled on the owner's word): start one only between generations.

## Done, do not redo (E, 21 Nov)

PR 141: train slices in a memory-capped scope (S1, first part). PR 145: the shared AGENTS.md contract and
docs/TRUE_ARCHITECTURE.md. PR 146: a fair versus trial for --pairs. PR 152: the record fetches
every artifact (paged). PR 153: --pairs on for the line after trial g65 (held out 3/27 vs 1/29, taught equal).
PR 156: --pull in versus's flags. PR 159: the pairs switch really on (format). Pairs judged on g69+g70: kept on. The research synthesis (owner's vision, 14 tracks). Plasticity check: no
spectral collapse g24/g43/g62 (log, 22 Nov 21:5x). PR 143 reviewed: keep the finding, archive the code.

## Done, do not redo (D, 19-21 Nov)

PRs 112-121, 125-130, 137-139: main's tests green; g48's practice causes (answers kept at the cap, guard on spending
code, two-part agreement, data as a file); three text kinds' rows fit; extra seats when a member crashed (R1); held-out
general repairs; the gate split into 33 jobs; slices end near their budget and are killed from outside at MIN+25;
data-format notes and hints; compound tasks part by part with rows per part; spec-detail lessons; measuring before
training capped. Earlier: see `docs/history/` and the log in `docs/AGENTS.md`.
