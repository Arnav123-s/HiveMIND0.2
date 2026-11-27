# Project recovery

All recovered project material is retained in the owner's private Google Drive recovery archive. The original Git history and model artifacts are preserved independently.

- [Recovery archive](https://drive.google.com/drive/folders/1OeaQS-ORh56bgqgdWQyZEWFnZQad_xBq)
- [Full archive manifest and checksums](https://drive.google.com/file/d/1j4_4nDljnc3xWh15wMSs6j3eXbta2IWs/view)
- [Local development source archive](https://drive.google.com/file/d/1KWSmMaeNY7_kYmino54ZYTciHaYA7eFl/view)
- [Recovery verification receipt](https://drive.google.com/file/d/124do4aRrTTV12m7nKnngSVXPUn0DFTO_/view)
- [Searchable project reference](https://drive.google.com/file/d/1pVDNli7NjYKgz39eJQytJ3s9H4oxL5iI/view)

The recovery set includes 329 archive parts, 1,117 Git commits, 8,901 Git objects, nine saved feature branches, 52,945 supplemental project files, and the available model artifacts and experimental records. The local-development archive contains 1,488 tracked text files and excludes all eight GitHub Actions definitions. Recovery archives include instructions for independent offline restoration. These materials reside in the private archive.

## Compute policy

Do not start training, scheduled experiments, or other hosted jobs on GitHub Actions. Use local compute or separately authorized infrastructure for training. Keep the archive private.

The original repository's access restriction is still under GitHub's control. Review the enforcement notice and obtain clarification before republishing the disabled repository's contents.

## Standing repository policy

No GitHub Actions compute. No scheduled or event-driven training workflows. Training runs local, on Kaggle or on
Hugging Face only. `.github/workflows/` stays empty or absent on every branch; historical definitions are inert
under `archive/workflows/`. Full text: [`docs/REPO-POLICY.md`](docs/REPO-POLICY.md). Archive contents and
verified results: [`BACKUP-INVENTORY.md`](BACKUP-INVENTORY.md).
