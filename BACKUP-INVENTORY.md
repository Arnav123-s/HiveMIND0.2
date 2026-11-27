# Backup inventory

Full history, model weights, datasets and experiment records live in the owner's **private Google Drive** folder
"HiveMIND recovery — 2026-11-26". They are **not** on GitHub. This repository carries code and text only.

Folder: <https://drive.google.com/drive/folders/1OeaQS-ORh56bgqgdWQyZEWFnZQad_xBq> (owner only, not shared)

## Folder contents

| Item | What it is |
|---|---|
| `00-READ-ME-RESTORE.txt` | Offline restore instructions |
| `TRANSFER-RECEIPT.json` | Upload and verification receipt (2026-11-26 19:53 UTC) |
| `RECOVERY-MANIFEST.json` | Every archive part with its SHA-256 (manifest SHA-256 `7c29047a…a6f54084`) |
| `CORE-MANIFEST.json`, `RESTORE.py`, `RESTORE-LAYOUT.py` | Reassembly and layout tools (named in the receipt and restore notes) |
| `01-HiveMIND-all.bundle.part001–006` | **Full Git history**: 474,450,768 B, SHA-256 `7f7a0b0859e03945f12b5350161173578bcbdcb405bdd74f28a9105efbc5fc7c` |
| `02-HiveMIND-source.zip.part001–005` | Source snapshot zip (SHA-256 `cc2ae2f0…50c9311`) |
| `03`–`12` `*.tar.zst.part*` | Model assets, data, tests and research done outside Git. Each numbered archive is independent. |
| `HiveMIND-code-docs-and-branches.zip` | 1,496 text files from the saved main, nine branch index entries, seven patches |
| `HiveMIND-project-memory.txt` | Searchable README, architecture, decisions and research notes |
| `project-inventory.zip` | File catalogue, hashes, release receipts, branch tables and archive map |
| `HiveMIND-local-development.zip` | Local development source (no Actions definitions) |
| `HiveMIND-local-code-only.bundle` | 2,400,288 B, SHA-256 `8a675e4588693759fee16d55bf97af2fc1259db9432ef347b3465f936d9b04d3`. **Source of this repo's code branches.** |
| `hivemind-muse-side.bundle` | Muse's lab record, branch `hivemind_muse` at tip `c061695`, 789,389,516 B |

Totals per the receipt: 339 items, 23.2 GB, 329 of 329 archive parts confirmed, and 84 of 84 independent archives
complete.

## Verified results (Stage 1, passed)

- **1,117 commits** and **8,901 Git objects** were restored independently from the full bundle. Last fetched main
  is `47aa83b4f163f27ef81333f9b75a68d7d8667eba`. There are 10 local branches and 14 worktrees.
- **Bundle hashes match.** All 6 Git pieces and all 5 source-zip pieces were downloaded again and matched their
  SHA-256.
- **fsck is clean.** `git fsck --full` reported no errors on either the local or the restored repository. It
  listed only dangling trees.
- The 52,945 inventoried files are all in the archive map, and their SHA-256s matched.

## Release coverage

| Generation | Release assets | Status |
|---|---|---|
| g73 | 27 | 27 verified |
| g80 | 27 + 1 cached rows receipt | 28 verified |
| g81 | 27 | 27 verified |
| **Total** | **82** | **82 / 82** SHA-256 matched to saved release receipts |

**g74 and g82** survive as **experimental checkpoints**, not releases. They are under
`workspace/organism-179aeed/island/geometry/` (`g74-main`, `g74-off`, `g82-period2`, `g82-off`, among others) in
the numbered archives. Use `records/archive-map.tsv.gz` to locate them.

The archive also holds 185 historical inference adapters (generations 28–73).

## Named losses (not recoverable from this backup)

1. **g84 final save step.** No g84 file appears anywhere in the archive map. The final save of g84 was never
   preserved.
2. **Early-generation release weights.** Only g73, g80 and g81 release assets were cached locally. Release
   weights of earlier generations, if they existed only on GitHub Releases, are gone.
3. **PR comment threads and other server-only data.** PR discussions, review threads, Actions logs and artifacts
   lived only on GitHub, and the original repository is blocked (HTTP 403, reason `tos`, 2026-11-26 11:44 UTC).
   Commits and PR heads survive in the bundle. The conversations do not.

## Where things live

| What | Where |
|---|---|
| Full Git history (all 1,117 commits, every branch and PR head) | Drive: `01-HiveMIND-all.bundle.part*` |
| Model weights, datasets and checkpoints | Drive: `03`–`12` archives |
| Muse's lab record | Drive: `hivemind-muse-side.bundle` |
| Lean code and docs (this repository) | GitHub HiveMIND0.2: `main`, `organism/niche-and-pool`, `organism/ast-triad`; `hivemind_muse` is planned |

Restore steps are in `00-READ-ME-RESTORE.txt`. No GitHub Actions are needed to restore anything.
