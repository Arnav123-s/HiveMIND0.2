# HiveMIND — Shared Agent Map & Anti-Drift Contract

Read this file first. Then read `docs/TRUE_ARCHITECTURE.md` (and `docs/RESEARCH_SYNTHESIS.md` for the owner's full vision and the research behind it), `NEXT_SESSION.md`, and `docs/reads/latest.md`.

## 1. True North Star (Never Drift From This)
HiveMIND is ONE self-improving digital organism on one frozen 1.58-bit ternary DNA (`t80`, `prism-ml/Ternary-Bonsai-8B`) running on free GitHub Actions CPUs (zero GPUs, zero paid APIs):
1. **Immortal Cell-Division Tree (`1 -> 2 -> 4 -> 8...`):** Children divide from their immediate parent in tiny, smooth steps using mistake-correction dreams (`--pairs`), harder dreams, and real-world data. Parents stay immortalized digitally; children may trade old skills for NEW capabilities; only strictly dominated/useless child leaves are pruned.
2. **Zero-Router "One Mind" Resonance:** No brittle keyword traffic-cop router and no 50% global trunk yank (`--pull 0.5`). Use SVD activation resonance (Arrow) so immortal parents and specialized child branches blend continuously into one mind.
3. **Orthogonal Layered Engrams + Sleep Consolidation:** Train child updates in subspaces orthogonal to their parent (O-LoRA) so `DNA + Parent + Child` stack without erasing the past, and fold proven breakthroughs downward during sleep (`training/hive_engram.py fold` + Grassmannian Flag Mean + Core-Space CART / Woodbury Low-Rank RegMean merging; `KnOTS` OOMs 16 GB RAM).
4. **Hard Anti-Drift Rule:** Do NOT turn HiveMIND into a hand-coded Python tool wrapper (`hive_reader.py`, `./hive exposure`, `./hive schedule`). All skill discovery, verification, and learning must be autonomous inside `grow.yml`.

## 2. Five Final Build Bars (`g60` Baseline on `main`)
| Bar | Target | `g60` (`main`) |
| :--- | :--- | :--- |
| Taught hard kinds | `>= 0.78` (`18/23`) | `0.78` (MET) |
| Held-out unseen kinds (`HELD_OUT`, 10 kinds) | `>= 5 / 10` | `0 / 10` (`1/30` fixed exam) |
| First-try practice | `>= 0.70` | `0.53` (`0.92` old / `0.30` new K2) |
| Talk exam | `>= 8.0 / 10` | `8.2 / 10` (MET) |
| Generation runtime | `<= 3.0 h` | `6.3 h` |

## 3. Exact Live Code Map (`main`)
- `.github/workflows/grow.yml`: `:492` `HIVE_CHUNK_CE || '0'`; `:552` `HIVE_ROW_CHECKPOINT` forced `'0'` on `train_minutes == 'auto'`; `:582` `HIVE_READ || '40'`, `HIVE_JUDGED || '10'`; `:763` 33-job `gate` on 20-runner cap.
- `training/grow.py`: `:211` `pair_rows` / `--pairs` DPO built-but-off; `:591` `_Prefix` disk cache; `:991` `MemoryGuard`; `:1156` `train()` default `max_len=1152`, `lr=2e-4`; `:1805` `--pull 0.1` (0.5 until trial g77).
- `training/lineage.py`: Fibonacci division up to `MOST_CELLS=24`, age-based retirement (`grown_body()`).
- `training/merge.py`: `_cut` QR+SVD; `join` (`:164`) currently passes `weights=None` (`:180`).
- `training/hive_engram.py`: built-but-off fast engram (`update` 50% replay + `fold` into trunk; line 17: "Nothing calls this unless asked").
- `team/solo.py` & `team/deliverables.py`: `SHORT` vs `MEMBER` orders, `same_files()` skips `.py`; `inspect()` checks shape only.
- `training/ternary_base.py` & `docs/KERNELS.md`: `ternary_base._C` C dequant; `ik_llama.cpp` hits `27.6 tok/s` cliff at 4 streams vs `76.5` at 3 streams.

## 4. Division of Labor
- **The builder (Loop Holder & `main` Integrator):** Owns `main` health, `grow.yml` switches, loop tracking (`scripts/cycle.py`, `docs/AGENTS.md`, `NEXT_SESSION.md`), read-only review of Astra's PRs, and clean merges.
- **Astra (the researcher — Lead ML Scientist):** Works in isolated `git worktree` branches off `main` building one tight, single-variable PR at a time against `docs/TRUE_ARCHITECTURE.md`. Never create giant `+50k LOC` folders under `training/experiments/`.

## 5. Empirical Kills Ledger (Never Repeat These)
- CPU speculative decoding (PR #122: slower, `23–37s` vs `13–18s`).
- Sequential gate alone (PR #135: saved `0` jobs across `9` generations).
- Naive unaligned TIES/DARE (PR #122: lost to plain sum `24/30`; align first via Grassmannian Flag Mean / Core-Space CART, not `KnOTS`).
- Text-overlap engram routing (`team/engram.py`, PR #109, PR #111: zero gain / high misfire).
- Small-sample unconstrained LoRA (PR #135: 4 updates dropped G55 held-out `3/30 -> 2/30`).
- Straight-through ternary weight edits (`g9`: `13/64` vs `26/64` float LoRA).
- Task-specific answer keys / duration hacks (PR #122, PR #123: `0/8` on blind English orders).
- MeZO / Zeroth-Order SFT (fails 8B code) and PiSSA + LoRA-FA (drops GSM8K).

- *Added Nov 22, 2026 from `docs/RESEARCH_SYNTHESIS.md` (its falsification ledger has the sources):*
- `Bitnet.cpp` (`TL2`) / `T-MAC` LUT kernels: slower than AVX-512 VNNI on these Xeons (`6.20 ns` vs `3.77 ns`).
- `TSV-M` / `Iso-C` / `WUDI` spectral flatteners: collapse heterogeneous 8B merges (`-3.79` to `-8.84`).
- Hypernetworks (`Text-to-LoRA`, `DnD`), per-token routers (`Arrow`/`SpectR`/`LAG`), `AKOrN`: unfeasible on 4 vCPU or break batched CPU decode (route once at prefill instead).
- Gradient-TTT (`70.8 min/task`), `TTT-E2E`, full `4096x4096` FAAST, 24-pass prompt-NLL BMA (`1.87 s/prompt`), `CMA-ES`/`ENMP`, `LISA`, `LESS`: too slow, mutate the frozen DNA, or zero gain.
- 1-bit LoRA (`BIN`/`RTN`), `BitDelta`/`DeltaZip`: `-14.6` to `-58.5 pts`, or `30x–60x` larger dense deltas.
- CPU activation sparsity (`TEAL`, `CATS`, `PowerInfer`, `TurboSparse`) and latent reasoning (`Coconut`, `Quiet-STaR`, `Huginn`): `-6` to `-12 pts`, incompatible with `PQ2_0_R8`, or need full pretraining.
- Mid-only LoRA (`14–29`), `IRM`/`V-REx`, passive paraphrase SFT, partial pass-rate rewards, online GRPO/DAPO, pure SimPO, `GaLore`/`FLoRA`, paid GPT-4/VLM judges, and all non-backprop CPU learners (`Oja`, `DFA`, `EqProp`, `Forward-Forward`, `LOZO`/`Infer-ZO`, `MeZO`): dead ends on 8B CPU code.
- *Added Nov 24, 2026 from `docs/RESEARCH_SYNTHESIS.md` Parts IV–VII:*
- Raw 8B hand-computed `assert f(x)==y` `CodeT` tests (50%–73% wrong) and online 25s property-plan generation on 4 CPUs (PR #158: 17/19 timed out, 0 usable plans).
- Naive per-factor `Muon` on skinny LoRA, `Lion`, `Adafactor`, `Prodigy`.
- Poetic simulations (quantum-phase LoRAs, spin-glass MCMC, 2D Turing/sandpile PDEs, synchronous cross-runner ODEs, random-detector strings, static Wolpert axes, token VCG auctions, global hormonal scalars, full `4096x4096` RG SVD, multi-hop *Physarum* meshes).
- Iterative Grassmannian Karcher/Thomson packing, `KnOTS` (OOMs 16 GB RAM), full Fisher/K-FAC/Sinkhorn OTFusion, strict Sarkar hyperbolic embeddings (`float64` overflow), and per-task `.gguf` rewrites (`10–30 s` server restart).
- 2-wave mid-training cross-runner sync (wastes 25–45m rebuilding 6 GB `_Prefix` cache; CPU slices already run `<180` steps) and intra-runner mid-run plateau injection (shocks AdamW past LMC split point and breaks `merge.py join`).
- *Added Nov 25, 2026 from `docs/RESEARCH_SYNTHESIS.md` Parts VIII–IX:*
- **WARNING: never enable `-fa` / `--flash-attn` on the pinned `ik_llama.cpp` `7aa7e014` (`team/ternary.py:52`): it silently disables ALL LoRAs** (ik_llama.cpp #2846, fixed in #2867). Bump the pin first, and pass `--cache-ram 0` + `cache_prompt=False` on adapter swaps (#2854).
- Tsetlin Machines, Weightless/LUT NNs (`BTHOWeN`/`ULEEN`/`DWN`), flat HDC/VSA, `SLIDE`/`MADDNESS` for Transformer FFNs, unstructured `SET`/`RigL`, MatMul-Free CPU backward, pure linear RNNs (`RWKV-7`/`Mamba-2/3`), `FP4` backprop on Xeon, `NoProp`/`uPC`/Target Prop, and from-scratch `1.3B–8B` CPU pretraining.
- Per-token HTTP logit-delta PoE (`3x–4x` CPU decode slowdown + KV invalidation).
- *Added Nov 25, 2026 from `docs/RESEARCH_SYNTHESIS.md` Part X:*
- Blind static, unprojected `cell + engram` LoRA stacking via `POST /lora-adapters` (LoRI: HumanEval `63.2% -> 22.3%`, GSM-P2 `68% -> 34%`); protect with frozen random `A` (LoRI-D) or soft null-space projection + Pythagorean scaling + sleep reset, and beat an equal-rank single LoRA by `>= +1` HELD_OUT with `0` Hard regressions.
- Dense or never-reset shared engrams.
- Instant all-to-all social sharing / uncontrolled LLM copy decisions (lineage collapse, `30–52%` token inefficiency).

## 6. Verification Standard
Every PR must stay single-variable, pass full `pytest`, and report verbatim stdout/stderr + exit codes with separate counts for raw member wins, final delivered wins, wrong deliveries, `HELD_OUT` wins, and regressions.

## 7. The generation, the repo map, how agents share the work, the rules (from PR 144)

### One generation (`.github/workflows/grow.yml`, it starts the next by itself)

fresh -> practice + dream -> gather + plan -> train (20 slices, 170-minute budget each) -> join -> merge ->
paired gate (before/after on 8 seeds, the fixed held-out exam, the talk exam, 33 jobs) -> record (keep or drop,
`docs/reads/gNN.md`, graphs, next generation). Never start, cancel or re-run a generation while one runs unless you
hold the loop. A trial run shares the same 20 runners and stalls the line: only between generations.

### Repo map

| Path | What it is |
|---|---|
| `team/solo.py` | members, agreement, hints, extra seats, the time cap; `team/by_part.py` compound tasks part by part |
| `team/tools/sandbox.py` | where member programs run; task data saved as files |
| `team/talk.py`, `team/ternary.py`, `team/guard.py` | chat routing; cells and seating; what programs may not do |
| `training/grow.py` | practice, dream, gather, plan, train (MemoryGuard, deadlines), the child's start |
| `training/merge.py` | `join` slices into a child, `mix` with the trunk, `trunk` |
| `training/exam.py`, `training/talk_exam.py` | the gate's exams; `training/kinds/` one file per task kind (`docs/KINDS.md`) |
| `training/lessons.py` | lessons made per kind; `training/reads.py` writes each generation's read |
| `scripts/cycle.py` | one screen: bars, last generations, next piece, who holds the loop |
| `scripts/agents_log.py --board` | every branch's claims; `scripts/capped_run.sh` the trainer's memory cap |
| `docs/PLAN.md` | the board of pieces; `NEXT_SESSION.md` the current handoff; `docs/knowledge.json` findings |
| `docs/reads/latest.md`, `docs/PROGRESS.md` | the newest generation's read and the graphs |
| `docs/DECISIONS.md`, `docs/OVERVIEW.md` | why things are as they are; the whole story |
| `tests/` | `python -m pytest -q tests`, must pass before any merge |

### How agents share the work (git is the only channel)

- **The loop holder (one at a time, the builder):** owns `main` stability, `grow.yml` switches, judges each generation,
  reviews other agents' PRs read-only, merges. Claims with a "holds the loop" line in `docs/AGENTS.md`.
- **Astra (the researcher side) and other workers:** build tight, single-variable experiment PRs in their own branches.
  One variable per PR, its proof in numbers, an off switch, nothing on the generation path by default until the loop
  holder takes it. Large binaries and experiment data stay out of `main`.
- **Claim before work:** a log line `- <UTC time> <letter>: claim <id>. <proof>` in `docs/AGENTS.md`; the earlier
  claim on a piece holds it. Work only in that piece's files; merge `main` in often.
- **Proof before merge:** replay real failures through the change, then the full suite, then one generation. One
  training-dynamics change per generation, so its effect can be read.
- **Shared files** (`grow.yml`, `docs/AGENTS.md`, `docs/PLAN.md`, `NEXT_SESSION.md`, `docs/knowledge.json`): small
  commits, append, keep both sides on a conflict.

### Rules that never change

Ask before spending money, using the owner's accounts beyond GitHub, or sending anything in the owner's name. No AI
model names in commits, PRs or files. Keep the owner's email private. Reach `main` only through a pull request.
Commit and push before you stop. Talk to the owner simply; assume no context.
