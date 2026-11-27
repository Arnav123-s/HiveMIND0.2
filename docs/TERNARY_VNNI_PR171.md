# Packed ternary CPU path: the full arc of PR #171

## Summary

PR #171 (`perf/packed-ternary-vnni`) went through two designs:

| Stage | Commit | Outcome |
|---|---|---|
| VNNI integer-matmul design (blocked) | branch tip `797b086c1bd6c7a26f1a4cb7a74f95acce02f32e` | **Blocked.** Parked as a draft and never merged. |
| Pivot: packed 2-bit storage + tile dequantizer | PR head `32fdb9567e70331db3a6286c0a3bbaf065b1dd35` | **Merged** on old main as `bd0246b2b5f7a9e3fa1242fdc0c11acadffd2d21` (2026-11-25). |

The branch tip `797b086` is **not** what shipped. The saved branch inventory still records `797b086` as the tip
of `perf/packed-ternary-vnni`. That is the blocked VNNI version, last moved 2026-11-24 17:55 PST per the saved
reflog. The merged work is the pivot. An earlier version of this file described only the blocked tip and said the
proposed fixes were unimplemented. That was incomplete, and this version corrects it.

Sources for these IDs, all in `project-inventory.zip` in the Drive recovery folder:
- `records/saved-pr-branches.tsv` gives the tip of PR 171's branch.
- `records/merged-pr-history.tsv` gives PR 171's integrated commit and PR head.
- `records/git-objects.txt` shows that all three commits exist in the full Git bundle.
- `records/reflog-hashes.txt` records when the branch tip moved.

## Stage 1: VNNI design (797b086, blocked)

- Frozen ternary weights (-1, 0, +1) were packed four codes per byte, with one BF16 scale per 128-weight group.
  That gives 0.25 + 2/128 = 0.265625 bytes per weight.
- Activations and upstream gradients were quantized dynamically and accumulated with register-tiled AVX-512 VNNI
  integer dot products (VPDPBUSD). Backward used a compact transposed representation.
- The design lived in `training/ternary_base.py` and `training/ternary_vnni.c`.

Reported results from Nov 24 (historical, not re-measured): warm forward + backward 21.96%–35.66% faster, peak
RAM 5.2976 GiB lower (full-size probe RSS 9,017.18 MB down to 3,328.90 MB), and 1,052 tests passed.

**Why it was blocked:**
1. Backward input-gradient max abs error relative to peak was **34.56%**, against **5.67%** for stock BF16. That
   is over the error budget.
2. The vocabulary-sized output projection (the LM head) ran slower.

## Stage 2: pivot (32fdb95, merged as bd0246b)

The pivot kept the packed 2-bit storage, which is where the memory win came from, and dropped the VNNI integer
arithmetic, which is where the accuracy problem came from. Packed weights are dequantized tile by tile into the
same BF16/FP32 PyTorch matmul that int8 storage already used. Numerics therefore match the stock path while the
weights stay at 2 bits each in RAM.

The saved project main (`47aa83b`, recovered here as `58f927f`) carries this design in `training/ternary_base.py`:
- `HIVE_TERNARY_VNNI=1` packs four codes per byte in the decoder, word embeddings and vocabulary head.
- The default is `HIVE_TERNARY_VNNI=0`, which keeps int8 codes plus a BF16 group scale.
- Forward and backward dequantize the weights they need into the stock matmul. `_pack` and `_unpack` do the
  conversion, and the code works through the weights in row tiles.
- `training/ternary_vnni.c` is not in the saved main. The VNNI C kernel exists only at `797b086`.

The Nov 24 20:14 follow-up request, which proposed per-128-element activation scaling and excluding the head
from VNNI, was cancelled by the owner at 20:15. The pivot is a separate design: it removes the VNNI arithmetic
altogether rather than patching it.

## Limitations of this record

- This lean repo holds a squashed code-only snapshot. Neither the ancestry `bd0246b` → `47aa83b` nor the exact
  pivot diff can be checked here. Both are in the full Git bundle (`01-HiveMIND-all.bundle.part*` on Drive,
  SHA-256 `7f7a0b08…5fc7c`).
- No benchmark of the merged pivot was re-run for this record. No pivot speed figures are claimed.
- Never substitute `797b086` for the merged design, or the other way round.
