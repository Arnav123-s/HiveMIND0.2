# Resuming the loop from g81 on the new compute

**Status:** working runbook, accepted by the manager 2026-11-28 (easy floor 30/32 accepted; eval_v2 ruled for every
web kind). Nothing has trained or run on Kaggle or Colab yet. R1 waits for the manager's go-ahead. **No long run
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
| F5 | `g39code` and `g44words2` are body cells, but **only their inference `.gguf` survived**: no `.safetensors`. The g82 trunk plan (`lineage.py trunk g82`) averages both. | `merge.py trunk` skips a missing leaf with a warning, so **g82's trunk would quietly lose two of its 18 leaves.** Recommended: rebuild both from their `.gguf` (`training/gguf_to_peft.py`, round trip tested; §9), validated in R1 on a cell that has both files. |
| F6 | g81's own new rows (about 786) and the cumulative `failures.jsonl` are not in the archive. | g82 trains on rows up to g80 plus its own. Dreams start with no old failures. |
| F7 | The ternary runtime builds from source in about 25 min on 4 cores (`team.ternary install`). | Cache the built binary in the HF store, keyed by CPU flags (as the Actions cache was), or lose 25 min every session. |
| F8 | **The open-web defect is still live in the code.** The shared check `_says` took any number of the text as the answer, and `h_walk` (the capability-check kind) used it too. | Fixed on this branch **before** any web evaluation, or Neon's check could produce a false PASS. Since the eval_v2 ruling: every web kind, the 5 hard and the 4 easy (§4). |
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
- **Lost-cell rebuild:** `training/gguf_to_peft.py` (§9).
- **eval_v2 re-baseline:** `scripts/rebaseline.py`, with `manifests/g81.json` (the bridge) and `manifests/g82.json`.
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
5. **Rebuild the lost `.safetensors` (F5, §9), validated first.**
   - Validate on a cell that has both files:
     `python3 training/gguf_to_peft.py t80.g81code.gguf check/adapter_model.safetensors --compare t80.g81code.safetensors`.
     It must exit 0: same names and shapes, relative error under 2e-3 (f16 rounding is about 5e-4).
   - If it does, rebuild both cells:
     `python3 training/gguf_to_peft.py t80.g39code.gguf t80.g39code.safetensors` (and the same for `g44words2`),
     then `hf_store.py put` both.
   - If it fails, stop and report: g82's trunk then goes without them, by ruling (§9, fallback).

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
- the F5 outcome: R1's validation on `g81code` and the two rebuilt cells, or the fallback if validation failed (§9).

### R6. Strict re-baseline, pre-flight, and launch g82 (after Muse's go-ahead)
- **R6a, the strict re-sit (eval_v2, Gemini ruling).** Re-sit the exact g81 body on **its own gate seeds**,
  `python training/exam.py sit --level mixed --tasks 8 --seed 2000081N --empty-memory` for N = 1..8. That is the
  full 32-task easy slice, plus 32 hard tasks, with the gate's inference settings (`AGENT_TEAM_SOLO=2`,
  `AGENT_TEAM_SOLO_ALONE=1`, `HIVE_TASK_SECONDS=900`). No preserved g81 answers exist to re-score: the archive holds
  none for seeds 20000811–20000818.
- **R6b, the re-baseline:** `python3 scripts/rebaseline.py --results runs/exam`. It writes the dual-score bridge into
  `manifests/g81.json`:
  - `easy_v1_loose` 32/32 (recorded), `easy_v2_strict`, the per-kind breakdown, and the active `eval_v2_floor`;
  - drift (`32 − v1 on the re-sit`) and checker effect (`v1 − v2`), so the delta can be read;
  - offline easy answers that enumerate (§10).

  The floor follows the ruling. **Delta 0:** 30/32. **Delta 1–2:** 30 − delta, with the easy web tasks that
  passed v1 and fail v2 tagged as recovery targets. **Delta 3 or more:** the script stops, writes no floor, and
  exits 3. **Back to Muse before anything else.**
- **R6c, pre-flight:** `python3 scripts/preflight.py manifests/g82.json --results runs/exam --out runs/g82/preflight.json`
  must print **CLEAR TO LAUNCH**. Under eval_v2 it refuses until R6b has written a floor.
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
  - **Applied to every web kind (eval_v2, Gemini ruling 2026-11-28):** the hard `h_web`, `h_walk`, `h_age`,
    `h_since`, `h_two`, and the easy `m_born`, `m_licence`, `m_pypi`, `m_csv`. The ruling named `m_born` and
    `m_licence`; `m_pypi` (the PyPI JSON lists every version) and `m_csv` (the page's HTML saved as `.csv` holds
    every price) were open to the same dump, so they follow the same standard.
  - **Both verdicts on every answer:** each web check tags its reason with
    `[eval_v1: pass|fail; eval_v2: pass|fail; kind: …]`, so one sitting gives both scores (R6b).
  - **Tests:** the g34 dump and dumps for each easy kind now fail, and honest answers still pass
    (`tests/test_resume_tools.py`).
- **Quarantine:** `training/quarantine.py` drops quarantined rows in **both** `grow.py gather` and `train`. A row is
  identified by the same key `once()` uses, so a copied or re-stamped row is caught too. The archive is never
  changed: the rows stay on Drive as evidence.
- **Comparability:** every web kind now scores under a stricter check than g81's own gate did. The easy floor is
  re-based on the strict re-sit (R6b), and the bridge in `manifests/g81.json` keeps both scores side by side. Hard web
  counts from g82 on compare with R6a's strict hard score, not with g81's recorded 23/32.

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

1. **Easy floor.** The re-sat easy slice must meet `easy_floor`. **30 of 32, accepted by the manager 2026-11-28**
   (g81's verified 32/32 less the paired gate's 2-point margin). Under eval_v2 it is final only after the strict
   re-baseline (R6b), which keeps it or lowers it by the ruling's delta rule. A breach
   launches only with a waiver `{check, by, reason, at}`, and the waiver is copied into the pre-flight record.
2. **Unique solves.** On any parent override or rollback, every task that **only** a dropped cell solved is
   flagged. The launch waits until each flagged task is acknowledged in the manifest. This is stricter than "flag",
   so a flag cannot go unseen.
3. **Dissent.** Every blocking dissent needs `acknowledged_by` and `acknowledged_at` before launch.

The g82 manifest is `manifests/g82.json` (copied here). It blocks until R6b has re-based the floor and R6c measures the
easy slice:

```json
{
 "generation": "g82",
 "grows_from": "g81",
 "dna": "t80",
 "easy_floor": {
  "min": 30,
  "of": 32,
  "basis": "accepted by the manager 2026-11-28 (g81 32/32 less the paired gate's 2-point margin); under eval_v2 it is final only once scripts/rebaseline.py has run"
 },
 "assumptions": [
  "g38->g39 gate breach (easy 25/32 vs 28/32) closed: reconciled retroactively, no rollback, no blocking authority over g82 (Neon ruling)"
 ],
 "required_changes": [
  "HIVE_STORE=hf",
  "eval_v2: one answer standard for every web kind (5 hard + 4 easy)",
  "training/data/QUARANTINE.jsonl filled by R1",
  "g39code/g44words2 .safetensors rebuilt from .gguf (if ruled)"
 ],
 "trunk_leaves_missing": [
  "g39code",
  "g44words2"
 ],
 "waivers": [],
 "override": null,
 "dissent": [],
 "eval": "v2",
 "bridge": "manifests/g81.json",
 "recovery_targets": []
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

## 9. The two cells with no `.safetensors` (F5): recommendation

**Rebuild them from their `.gguf`, validated first on a cell that has both files.** Fall back to a logged
16-leaf trunk only if that validation fails.

**Why it is feasible.** The `.gguf` stores the very same LoRA matrices:
- `grow.py` writes it with the runtime's `convert_lora_to_gguf.py --outtype f16`.
- The converter copies `lora_A` and `lora_B` unscaled, keeps `lora_alpha` in the metadata, and permutes nothing for
  the Qwen3 family.

`training/gguf_to_peft.py` reverses it into the hive's own layout: `base_model.model.<module>.lora_{A,B}.weight` in
float32 plus `adapter_config.json`.

**What was tested here** (`tests/test_resume_tools.py::test_a_cell_survives_the_round_trip_through_its_f16_gguf`).
The test builds a synthetic Qwen3-shaped cell in the hive's layout (rank 16, alpha 32, the last layers, all 7
projections) and runs it through the repo's converter at f16 and back:
- every name and shape came back, with the rank, alpha and layers;
- relative error was 4.7e-4 per matrix and 3.0e-4 on the edit B·A, which is float16 rounding and nothing more;
- `merge.load` reads the rebuilt file.

In an 18-leaf trunk built with `merge.merge`, swapping one leaf for its rebuild changed the trunk by **1.5e-4**.
Leaving that leaf out changed it by **0.36**. Those synthetic leaves were independent, so 0.36 overstates the real
effect (real cells are correlated), but the gap is three orders of magnitude.

**Not yet tested:** a real cell. The pair `t80.g81code.gguf` / `.safetensors` is in the archive but over the 10 MB
connector limit, hence the validation in R1 step 5. A real t80 cell matches rank 16 on the last 14 of 36 layers,
float32: 14 × 1.212 M parameters × 4 B = 67.88 MB, the released `.safetensors` exactly.

**Cost of each option in capability terms:**

| Option | Trunk of g82 | Capability cost | Work |
|---|---|---|---|
| **Rebuild (recommended)** | the 18 leaves the lineage plans | about none: float16 rounding of two leaves (trunk change ~1e-4 in the test) | ~10 min in R1, plus validation |
| Go without them, logged | 16 leaves, re-weighted | the trunk loses the code line of g39 and the words line of g44 from its average (1/9 of its weight). The size of the loss is unmeasured, and g82's trunk departs from the lineage's own rule. Both cells still sit in the body as members: their `.gguf` serves inference. | none |
| Hold the trunk to the survivors | same as going without | same, written as a rule, not an exception | none |

The last two differ only in bookkeeping. Either way, no future trunk could contain these two lines again.

## 10. The easy-slice sweep (manager's ask: is the strict standard uniform?)

- **Every easy kind that reads the network is now under eval_v2:** `m_born`, `m_licence`, `m_pypi`, `m_csv`.
- **The offline easy kinds have no page to dump, but several pass on loose matches.** These are not changed, since
  the ruling covers web kinds; they are **measured** instead. R6b lists every passing offline easy answer with more
  than 8 numbers or two weekday names or more (`offline_enumeration_exposure` in the bridge), so the council gets an
  incidence, not a guess.

| Kind | Check | Open to |
|---|---|---|
| `m_code`, `m_fix`, `m_days`, `m_percent`, `m_units` | `says()`: any number within tolerance | an answer that lists many numbers |
| `m_table` | `says()` for mean and total, plus a labelled line | partly (the labelled line is strict) |
| `m_weekday` | `B.has(weekday)`: a substring | **an answer that lists all seven days** |
| `m_know` | the symbol and the atomic number both present | little |
| `m_translate`, `m_summary` | keyword substrings (`m_summary` must also be one sentence) | little to some |
| `m_follow`, `m_steps`, `_programs` | exact form, structured parts, hidden tests | strict |

