# Repository policy (standing)

This rule applies to every branch of HiveMIND0.2. It does not expire.

1. **No GitHub Actions compute.** Nothing in this repository may run on GitHub-hosted or self-hosted Actions
   runners. The original repository was blocked after heavy Actions use.
2. **No scheduled or event-driven training workflows.** That means no `schedule:`, `push:`, `workflow_dispatch:`,
   `workflow_run:` or other triggered jobs that train, evaluate, grow generations or chain one run into the next.
3. **Training and experiments run locally, on Kaggle, or on Hugging Face only.**
4. **`.github/workflows/` stays empty or absent.** The historical workflow and composite-action definitions are
   kept for reference only under `archive/workflows/` and `archive/actions/`. They are inert there: GitHub
   executes only files under `.github/workflows/`. Do not move them back.
5. Model weights, datasets and full Git history stay in the owner's private Drive recovery archive and are not
   committed here (see `BACKUP-INVENTORY.md` on `main`). Keep committed files under 50 MB.

Sections of `README.md`, `AGENTS.md`, `CLAUDE.md` and other docs that describe running the hive from the Actions
tab or on free Actions runners are **historical**. They describe the old repository and are not instructions.
