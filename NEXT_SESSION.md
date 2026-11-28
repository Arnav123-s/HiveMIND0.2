# NEXT SESSION: the one current handoff

Written 2026-11-26 06:50 UTC by F (session `session_014dX1CPC6pqYD4ypd7HxZko`, branch `claude/cool-hypatia-9cqjec`),
who **holds the loop** since 17:20. How to work: `AGENTS.md` (the contract), `CLAUDE.md` (the cycle),
`docs/TRUE_ARCHITECTURE.md` (the stages), `docs/RESEARCH_SYNTHESIS.md` (the owner's vision and research, Parts 0-VII).
Start with `python scripts/cycle.py`. GitHub here: `gh api` REST only. Repo variables are refused by the proxy (403);
dispatch, cancel and `rerun-failed-jobs` work. Previous handoff: `docs/history/NEXT_SESSION_2026-11-23_E.md`.

## In flight: ORPO trial g84; the speed stack was slower

- **Trial g84 (run 45768806535)**, started 06:31 UTC 26 Nov: versus `--orpo`, pairs on (PR 176's two-pass ORPO vs the
  line's DPO pairs). Judge per arm at its record, then restart the line with `pairs=1`.
- **Speed-stack trial g83 (run 45723265418): not for the line.** versus `--packed-dna --vocab-ce --train-batch 4`:
  median 33.2 s a row vs 13.1 (about 2.5x slower, 40% of the rows in the same 170 min), peak memory 10.4 vs 13.5 GB;
  delivered 56 vs 54 (hard 24 vs 22), 0 regressions, but fixed held-out 1/28 vs 3/29. The flags took (logs: "four
  weights per byte", train_batch 4, loss_chunk_tokens 64). Three switches at once, so the slow one is unknown; likeliest
  the packed DNA. Next: time each switch alone on a runner before any further speed trial (finding `speedstack`).
- **g81 (run 45673775503) kept, 55 vs 54** (hard 23 vs 22, held out 2/28 vs 1/28, talk 8.2, 5.17 h, 0 slices lost):
  the second kept generation in a row at `--pull 0.1` (g80 53 vs 50).
- **Trial g83 (run 45723265418)**, started 23:42 UTC 25 Nov on 4d99dda: versus `--packed-dna --vocab-ce --train-batch 4`,
  pairs on. g82 (run 45721743804) was cancelled in practice, before training. Judge per arm at its record: train
  minutes per slice, peak memory, delivered, member right, not passed, fixed held-out, regressions vs after; check
  the versus cell json (train_batch 4, loss_chunk_tokens 64) and its train log ("four weights per byte").
- **g80 (run 45621202333) kept, 53 vs 50**. `--pull 0.1` read four times: trial g77 +3, g79 -1, g80 +3, g81 +1.
- **Trial order (the owner, 25 Nov 22:00 UTC):**
  1. **Leap 1, at g81's record:** cancel g82 before training, then the full speed stack in one trial (three switches
     at once by the owner's call): `gh api -X POST repos/arnav123-s/hivemind/actions/workflows/grow.yml/dispatches -f ref=main -f 'inputs[versus]=--packed-dna --vocab-ce --train-batch 4' -f 'inputs[pairs]=1'`.
     Read train minutes per slice, peak memory and quality per arm; if quality holds, propose the stack for the line.
  2. **Then** `versus='--orpo'` with `pairs=1`.
  3. Then trials for PR 188 (niche lineage; needs `cells/niche.json` written by the record first) and PR 186 (birth
     donor / sibling repulsion; needs `grow.yml` artifact wiring for `--donor` / `--sibling` first). PR 187 parked.
- **The researcher side is building three core organism PRs in parallel, no file overlap:** `training/lineage.py` niche
  survival / apoptosis; `training/grow.py` + `training/merge.py` PR VII-1 cross-organ borrowing + twin repulsion;
  `training/hive_engram.py` + `team/ternary.py` NP-LoRA one-mind stacking + Self-Canary sleep fold. Review each
  read-only, merge flag-off, and trial one per generation.
- g79 (run 45567825482) dropped 49 vs 50 at 0.1; its record worked (cells in `hive-grown-2`), then the line's guard
  stopped it after 3 recorded drops running; F restarted it.
- **g78 (run 45521871368) is unrecorded, but would have been dropped** (rebuilt from its artifacts: 52 vs 53, hard 20 vs
  21, held out 3/29 vs 4/29), so the body lost nothing. Its record hit GitHub's 1000-file limit on the
  `hive-grown` release. Cells now go to `hive-grown-2` and on when a release is full (`training/release_store.py`);
  every cell read tries them in order. Nothing was deleted. 48 old `t17` files could be deleted by the owner to tidy.
- **Stage 1B trial g77 (run 45460563586) judged: `--pull 0.1` wins.** Delivered 55 vs 52 of 64 (hard 23 vs 20),
  member right 57 vs 53, not passed 9 vs 12, 0 regressions vs after, pulse held-out 4/9 vs 2/9, fixed held-out flat
  (3/26 vs 3/27). +3 is at the seed-noise edge (about 3 of 64), but every count moved one way.
- g75 (run 45387523263) dropped 52 vs 52; its record passed on the 5th re-run (four GitHub 502s). PR 164 retries.

## Where the hive stands (g74, recorded 2026-11-24 11:47 UTC)

| gen | verdict | body (before) | hard | held-out fixed exam | first try | talk | hours |
|---|---|---|---|---|---|---|---|
| g69 | dropped | 49 (50) | 17/32 | 2/29, 1 kind | 0.64 | 8.0 | 6.16 |
| g70 | kept | 52 (48) | 20/32 | 4/29, 2 kinds | 0.60 | 8.0 | 5.14 |
| g71 | dropped | 50 (52) | 18/32 | 4/28, 1 kind | 0.64 | - | 5.45 |
| g72 | kept | 50 (48) | 18/32 | 1/29, 0 kinds | 0.57 | - | 5.38 |
| g73 | kept | 52 (50) | 20/32 | 2/28, 0 kinds | 0.64 | - | 5.23 |
| g74 | dropped | 50 (52) | 18/32 | 2/27, 1 kind | 0.60 | 7.2 | 5.08 |

- All with `--pairs` (on since g69; g67 trained without it by a switch bug, fixed in PR 159). 0 of 20 slices lost
  since PR 141. Bars at g74: hard 0.67 (bar 0.78), held-out 1 of 10 kinds (bar 5), first try 0.60 (0.70), talk 7.2
  (8), 5.08 h (3). None met at g74; hard and talk were met at g70.
- **Honest read on pairs:** trial g65 (3/27 vs 1/29) and g70 (4/29, 2 kinds) looked like a held-out gain; g71-g74
  (4/28, 1/29, 2/28, 2/27) are back in the old 1-4 of 29 band. Taught tasks unchanged. Pairs cost little time. Keep
  them on (no harm seen) but do not count them as the held-out answer. Seeds swing a body by about 3 of 64.

## Open, in order

1. **Judge the ORPO trial g84, restart the line; time each speed switch alone** (above).
2. **Astra's PR 163** (`HIVE_QUORUM_K`): reviewed 25 Nov, **parked, flag off**: 42.9% time saved but wrong deliveries 0 -> 3 (held out 0 -> 2) on 12 matched g73 tasks.
3. **Researcher-side speedup PRs** (reviewed 25 Nov): PR 170 Phase 0 prefix unload merged **off** (`HIVE_UNLOAD_PREFIX=0`;
   bit-identical, 4.63 GiB freed; a real slice needs 25.3 GiB disk, so trial it only after a disk check). PR 167
   vocab-outer CCE parked for a trial after g80 (needs `HIVE_CHUNK_CE` set in a versus run). PR 171 merged off since (see 4). Do not flip a training default until `--pull 0.1` is read at g79-g80.
4. **Queued trials:** see "Trial order" above (Leap 1 speed stack, then ORPO, then PRs 188 and 186). **PR 173** (IX-1 relay to a disputed third seat) **parked, flag off**: tied stock 5/9 vs 5/9
   on g73, 0 new wins, 0 Seat 3/4 rescues.
5. **The researcher side's PR 158**: final, **parked in draft, flag off**: 17 of 19 runtime property-plan calls timed out
   at 25 s -> 0 usable plans; held out 0 of 10; wrong 17->9 only by withholding 16->25; correct 5->4. Property checks
   move offline into `niche_dream` (Research Part IV).
6. **D's four single-change branches**, one trial each, never together: `claude/blissful-mccarthy-32ekqy-chunk`
   (HIVE_CHUNK_CE under auto), `-rowckpt`, `-read` (HIVE_READ/JUDGED default 0), `-join` (slices weighed by steps).
   Merge main into each and run the full suite first.
7. Then the stages in `docs/TRUE_ARCHITECTURE.md` (1C niche survival and the anti-collapse pool; 2 runtime; 3 one mind).
   Held-out (H1) is the bar furthest away: read the gate-work worker files by cause before building.

## Traps seen (do not repeat)

- GitHub expressions compare an empty input as the number 0: use `format('{0}', inputs.x) != '0'` (PR 159).
- `download-artifact` with `pattern` sees only the first 100 artifacts (PR 152); one failed download killed the record
  until PR 164 (3 tries, then a warning and go on).
- A re-run of a failed job runs the run's own commit, not today's `main`: a fix merged after the run does not reach it.
- A trial beside a line generation stalls both (20 shared runners). Cancel a generation only before it trains.
- Helpers in this environment are often refused `git commit`; the loop holder commits their staged work only when the
  owner asked for that change.
- Merge the right branch: PR 155 went in without its change (a helper's worktree branch was not the one merged).

## Done, do not redo (F, 24 Nov)

PR 164 fetch_artifacts retries; PR 165 research Parts IV-VII, new kills, merge geometry in place of KnOTS, PR 158
recorded parked; g75 recovered and recorded.

## Done, do not redo (E, 21-23 Nov)

PR 141 trainer memory cap (0 of 20 slices lost since); PR 145 AGENTS.md contract; PRs 146/152 fair, fully fetched
versus trials; PR 153/159 `--pairs` on for the line; PR 156 `--pull` in versus; PR 157 research synthesis; plasticity
check (no spectral collapse g24/g43/g62); PR 143 reviewed (archive the code). Earlier: `docs/history/`, `docs/AGENTS.md`.
