# Resuming the loop from g81 on the new compute

**Status:** plan plus the first tools, 2026-11-28. Nothing has trained or run on Kaggle or Colab yet. **No long run
starts before the report-back point (step R5).** Owner: the council (manager Muse). Policy: `docs/REPO-POLICY.md`.
GitHub Actions stays off: every step below runs in a notebook or on the owner's machine.

## 1. Where we start

- **Body:** g81. Easy 32/32, hard 23/32, 55/64; talk 8.2/10; no crashes (`docs/reads/g81.md`). DNA `t80`
  (`prism-ml/Ternary-Bonsai-8B`). 24 cells plus `g81trunk` (`cells/body.json`).
- **Weights:** in the private Drive archive, not on GitHub (`BACKUP-INVENTORY.md`). 82 release assets for
  g73/g80/g81 (3.27 GB). Cumulative training rows exist only as of g80 (`datasets/g80/rows.jsonl`, 56.7 MB).
- **Compute:**
  - Kaggle is the main lane: about 30 GPU-h a week, sessions of at most 12 h.
  - Colab Pro is for GPU bursts.
  - Hugging Face Free (`asrinna118`) is the weight store.
  - HF Jobs is not available.
- **Council inputs folded in:**
  - **Neon (Gemini):** a site-walking capability check before g82 (§5, R4), and a pre-flight gate on every launch
    (§6).
  - **Codex audit:** the open-web false positive. 7 rows are quarantined, the checker is fixed, g81 gets a
    behavioural web test, and the incidence is measured (§4, R1 and R4).
- **Assumption on record:** the g38→g39 gate breach (easy 25/32 against C's floor of 28/32) is a **closed
  historical breach**, reconciled retroactively, with no rollback and no blocking authority over g82 (Neon ruling;
  logged in the council record by Muse). The g82 floor below is set fresh against g81. It does not inherit C's 28.

## 2. What the code and the data say (found while planning, not assumed)

| # | Finding | Consequence |
|---|---|---|
| F1 | g81 took 5.17 h of wall time **across about 20 parallel Actions runners**: practice 12 × 43 min, dream 8 × 61, train 20 slices × 174 min, gate 33 jobs over 63 min. That is about **100 runner-hours** (4-vCPU) a generation. | A g81-shaped generation **cannot** run on one 12 h session. It must be split into stage-sessions that each checkpoint out, made smaller, or both. The probe (R2/R3) decides which, with numbers. |
| F2 | The trainer uses the GPU only **without** `--int8-base` (`training/grow.py`, `gpu = cuda and not int8_base`). It then loads the 8B in bf16: about 16 GB of weights before activations. | It probably **does not fit a 16 GB Kaggle T4/P100** and should fit a Colab L4 (24 GB) or A100. Kaggle's GPU quota is not a t80 training lane without a code change (multi-GPU sharding). R3 measures this rather than assume it. |
| F3 | Colab was never actually used: 34 of 34 recorded reads say "not seen" or "not there". | **GPU training time is completely unmeasured.** The only figure is CPU at 9–18 s a row (`training/colab.py`). |
| F4 | Weights went through GitHub Releases (`training/release_store.py`). The new repo has none. | New `training/hf_store.py`, switched on with `HIVE_STORE=hf`: same get/put, one commit per put. |
| F5 | `g39code` and `g44words2` are body cells, but **only their inference `.gguf` survived**: no `.safetensors`. The g82 trunk plan (`lineage.py trunk g82`) averages both. | `merge.py trunk` skips a missing leaf with a warning, so **g82's trunk would quietly lose two of its 18 leaves.** Needs a decision (R5). |
| F6 | g81's own new rows (about 786) and the cumulative `failures.jsonl` are not in the archive. | g82 trains on rows up to g80 plus its own. Dreams start with no old failures. |
| F7 | The ternary runtime builds from source in about 25 min on 4 cores (`team.ternary install`). | Cache the built binary in the HF store, keyed by CPU flags (as the Actions cache was), or lose 25 min every session. |
| F8 | **The open-web defect is still live in the code.** The shared check `_says` took any number of the text as the answer, and `h_walk` (the capability-check kind) used it too. | Fixed on this branch for the 5 hard web kinds **before** any web evaluation, or Neon's check could produce a false PASS (§4). |
| F9 | The `h_walk` lesson measures 534 pieces against `lessons.LIMIT` 508 (test fails on the untouched snapshot as well). g81's read says "walk a site: lessons none made". | A **hypothesis** for why the site-walking skill doesn't reach the body. The count uses a fallback rule without the DNA tokenizer; re-measure on Kaggle with the real tokenizer. |
| F10 | `g38code2` and g38's gate files (run 43950235200) are **not in the archive** (no entry in `records/archive-map.tsv.gz`). | Neon's follow-on "extract g38code2's traces" is not possible from the backup. Its lessons can be **re-made** (`training/kinds/h_walk.py` is deterministic) and its rows may survive inside `datasets/g80/rows.jsonl`; check in R1. |

## 3. The order of work

Each step names where it runs, how long it may take, what it produces, and the condition for moving on.

### R0. Tools (this branch, done)
- **Weight store:** `training/hf_store.py`.
- **Pre-flight gate:** `scripts/preflight.py`.
- **Capability check:** `scripts/capability_check.py`.
- **Stage probe:** `scripts/stage_probe.py`.
- **Lean tests:** `scripts/lean_tests.py`, with the reviewed list in `tests/LEAN-DESELECT.tsv`.
- **Row audit and quarantine:** `scripts/audit_web_rows.py`, `training/quarantine.py`, and the checker fix in
  `training/kinds/`.
- **Tests:** `tests/test_resume_tools.py`.

### R1. Move the body into the store and audit the rows (Colab, CPU runtime, Drive mounted; at most 2 h; no model)
1. Mount Drive. Reassemble and verify archive group `03` with the archive's own tool:
   `python RESTORE.py --manifest RECOVERY-MANIFEST.json --only 03-released`. It checks every part's SHA-256.
   Extract into an empty directory.
2. With `HIVE_STORE=hf`, `HIVE_HF_REPO=asrinna118/hivemind-cells` and `HF_TOKEN` from Colab secrets, run
   `python3 training/hf_store.py put` on `models/g81/*`, `models/g80/*`, `models/g73/*` and the two surviving
   `.gguf` of `g39code`/`g44words2` (archive group `05`, saved as `t80+g39code.gguf`: rename to `t80.g39code.gguf`, the store's form), plus `datasets/g80/rows.jsonl` as `rows.jsonl`. Then run
   `hf_store.py ls` and check the count: 82 release files + 2 + rows.
3. `python3 scripts/audit_web_rows.py rows.jsonl --quarantine` on the g80 rows. **Pass:** lines 1068–1074 are among
   the flagged rows (the confirmed incident). The script prints the **incidence by kind** across every
   positively-labelled web answer, which answers the audit's open question. Commit the filled
   `training/data/QUARANTINE.jsonl` and `runs/audit/web-rows.tsv`.
4. Search the rows for any row of g38 on "walk a site over several pages" (F10) and record what is there.

### R2. Stage probe on Kaggle (CPU session; at most 3 h)
`HIVE_STORE=hf DNA=t80 python3 scripts/stage_probe.py --rows rows.jsonl` covers env, store, install, practice,
gate, train_cpu and upload. It writes `runs/probe/probe-<host>.json` and puts it in the store.
- It measures seconds per practice task, seconds per gate task, CPU seconds per training row, peak memory, setup
  time and store speed.
- It prints an estimate for a g81-shaped generation on this machine.
- Also upload the built runtime (`~/.cache/hive/ternary/ik_llama.cpp/build/bin`) to the store (F7).

### R3. GPU probe (Colab L4 or A100, at most 1 h; and one Kaggle GPU session, at most 30 min)
`python3 scripts/stage_probe.py --stages env,store,train_gpu --rows rows.jsonl`. **Outcome:** GPU seconds per row
on Colab, and on Kaggle whether a T4 fits t80 at all. An out-of-memory error is recorded as a result (F2).

### R4. Early web evaluation of g81 (Kaggle CPU, internet on; at most 4 h). One sitting serves both rulings.
```
python3 scripts/capability_check.py sit --set walk     # Neon: 5 site-walking tasks; hive, trunk, each code cell
python3 scripts/capability_check.py sit --set web      # Codex: 5 independent open-web answers (Wikidata)
```
Both sets run under the **fixed** checker. Each records:
- the verdict: PASS, ROUTING or ABSENT;
- tasks a cell or member solved that the hive did not;
- how many answers were dump-shaped. This is the behavioural test of whether g81 learned the g34 dump.

**Reading the result:**
- **Walk PASS:** the g38code2 question is closed for good.
- **ROUTING or ABSENT:** debug the routing, and re-teach the kind (F9, F10). **The body is not rolled back.**
- **Web:** any dump-shaped answer is evidence that the defect reached g81's behaviour, and goes to the council.

### R5. Report back to Muse. **STOP: nothing long runs before this.**
The report carries:
- the probe numbers;
- the generation shape that fits (§7);
- the R4 verdicts;
- the audit incidence;
- the F5 decision. Choose one:
  1. rebuild the two `.safetensors` from their `.gguf` (feasibility unverified);
  2. let g82's trunk go without them, logged in the manifest;
  3. hold the trunk's makeup to what survives.

### R6. Pre-flight re-sit and launch g82 (after Muse's go-ahead)
- Re-sit g81's easy slice on Kaggle: 4 fixed seeds × 8 tasks, the gate's "before" arm.
- `python3 scripts/preflight.py manifests/g82.json --results runs/exam --out runs/g82/preflight.json` must print
  **CLEAR TO LAUNCH**.
- g82 is a **parity generation**: g81's code plus only the required fixes (store, checker, quarantine), sized from R2
  and R3.

### R7. Fold in the three improvements, one per generation
The repo's own rule is "one change a generation" (`docs/reads/g81.md`, next piece: dose). Each goes through the
same pre-flight.
- **g83:** `--orpo` (ex-PR #176). Already in main as an opt-in training flag (`grow.py train --orpo`); turning it
  on is the whole change.
- **g84:** Stratified Pool (PR #195, branch `organism/niche-and-pool`). #195 and #200 merge cleanly onto main
  together, and their 15 tests pass in the merged tree.
- **g85:** Fresh AST Practice (PR #200, branch `organism/ast-triad`).

The order runs from the smallest change to the largest. Folding all three into one generation is possible if the
council prefers it, but then no single change could be credited or blamed.

## 4. The open-web defect (Codex audit) and what this branch changes

- **Cause:** a check passed if **any** number in the answer lay near the expected value. A Wikidata page holds
  hundreds of numbers, so a dump passed.
- **Fix** (`training/kinds/_common.py`: `_dumped`, `_states`): an answer must look like an answer, with no
  page/HTML/Wikidata JSON markers and no more than N numbers, where N is set per kind with room for honest dates and
  versions. It must also state each expected value.
  - **Applied to the 5 hard web kinds:** `h_web`, `h_walk`, `h_age`, `h_since`, `h_two`.
  - **Tests:** the g34 dump now fails, and honest answers still pass (`tests/test_resume_tools.py`).
- **Not yet changed:** the easy kinds `m_born` ("operate a website", a substring test) and `m_licence`. Changing an
  easy check moves the easy floor's baseline, so that is for the council to decide. The audit still measures both.
- **Quarantine:** `training/quarantine.py` drops quarantined rows in **both** `grow.py gather` and `train`. A row is
  identified by the same key `once()` uses, so a copied or re-stamped row is caught too. The archive is never
  changed: the rows stay on Drive as evidence.
- **Comparability:** the hard web kinds now score under a stricter check than g81's own gate did. Easy kinds are
  unchanged, so the easy floor still compares. The hard web counts of g82 and later are not directly comparable
  with g81's 23/32; R4 gives the strict baseline.

## 5. The capability check (Neon)

The g38 tasks themselves are gone (F10), so the check uses the active suite's equivalent: 5 `h_walk` tasks
(books.toscrape.com, checked live) on fixed seeds `capcheck-h_walk-0..4`.

| Arm | Members | `AGENT_TEAM_SOLO` | `AGENT_TEAM_SOLO_ALONE` |
|---|---|---|---|
| hive | the full roster from `lineage.py members` | 2 | 1 (as the gate seated it) |
| trunk | `g81trunk` alone | 1 | 1 |
| each code cell | `g81code`, `g81code2`, `g80code`, `g80code2`, `g39code`, each alone | 1 | 1 |

Per-member verdicts inside the hive arm also catch "a member solved it and the hive didn't". Whether a single-member
arm behaves exactly as described must be confirmed on the first sitting.

## 6. The pre-flight gate (Neon); a launch is blocked unless all three pass

1. **Easy floor.** The re-sat easy slice must meet `easy_floor`. Proposed: **30 of 32**, which is g81's verified
   32/32 less the 2-point margin the paired gate already allows. The council must confirm the number. A breach
   launches only with a waiver `{check, by, reason, at}`, and the waiver is copied into the pre-flight record.
2. **Unique solves.** On any parent override or rollback, every task that **only** a dropped cell solved is
   flagged. The launch waits until each flagged task is acknowledged in the manifest. This is stricter than "flag",
   so a flag cannot go unseen.
3. **Dissent.** Every blocking dissent needs `acknowledged_by` and `acknowledged_at` before launch.

The g82 manifest (to be saved as `manifests/g82.json`). It blocks until R6 measures the easy slice:

```json
{
  "generation": "g82",
  "grows_from": "g81",
  "dna": "t80",
  "easy_floor": {"min": 30, "of": 32, "basis": "g81 verified easy 32/32 (docs/reads/g81.md), less the paired gate's 2-point margin; proposed, for the council to confirm"},
  "assumptions": ["g38->g39 gate breach (easy 25/32 vs 28/32) closed: reconciled retroactively, no rollback, no blocking authority over g82 (Neon ruling)"],
  "required_changes": ["HIVE_STORE=hf", "open-web checker fix (5 hard web kinds)", "training/data/QUARANTINE.jsonl filled by R1"],
  "trunk_leaves_missing": ["g39code", "g44words2"],
  "waivers": [],
  "override": null,
  "dissent": []
}
```

## 7. How the timing question gets answered

R2 and R3 give four rates on the real machines: seconds per practice task, seconds per gate task, seconds per
training row (CPU and GPU), and setup minutes. With them, `stage_probe.py` works out a g81-shaped generation stage by
stage. The rule for sizing g82: **every session must end, store written, by 10 h of its 12.** That leaves room for
setup (about 30 min with a cached runtime) and the checkpoint upload.

- **If the whole generation fits in 10 h on one machine:** one session per generation.
- **If not**, which F1 expects, split it into stage-sessions that each start from the store and end by writing to it:
  1. practice and dream: one or more Kaggle CPU sessions, a shard each;
  2. train: Colab GPU, or Kaggle CPU slices;
  3. merge and gate: Kaggle CPU, shards of seeds;
  4. record: anywhere, no model.

  Shrink the shape (fewer practice tasks, fewer gate seeds) only as far as the numbers require. A session that dies
  loses one stage, not a generation.
- **Not built yet, on purpose:** the stage runner that replaces `grow.yml`'s DAG. Its shape depends on R2 and R3.

## 8. The lean test procedure

```bash
python3 scripts/lean_tests.py            # must exit 0 before any commit that touches training/ or team/, and before a launch
python3 scripts/lean_tests.py --audit    # runs only the listed exclusions: any that now pass come off the list
```

- **What it runs:** the full suite minus `tests/LEAN-DESELECT.tsv`. Every line there carries a category and a
  reason:
  - **WORKFLOW** (68): asserts on the archived workflows. Rewrite or retire; never restore the workflows.
  - **DRIVE_DATA** (12): needs files that live on Drive.
  - **PREEXISTING** (4): fails on the untouched recovered snapshot. These are real defects to fix, not hide.
- **What counts as a regression:** any failure off the list.
- **Stale entries:** an entry that names a test that no longer exists also fails the run.
- **What it doesn't cover:** whether the hive actually works. That is the job of the exams: the pre-flight re-sit
  (§6), the capability check (§5) and the gate of each generation.
