# HANDOFF: W-A7-RJ5 (2026-11-17)

Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_wa7rj5/`. Report: `WA7_RJ5_READ.md`.

## State
- Prep, labels (1,018 / 1,018), scoring (1,018 rows, 0 errors) are DONE. Scores: `run/scores.jsonl`. Labels: `run/labels/`.
- Analysis was run WITHOUT `--c1-records`, so the formal row is **INVALID** (`run/W_A7_RESULT_asrun_INVALID.md`, `w_a7_result_asrun.json`).
- P5-R Delta -0.01337 pp [-0.09283, +0.12245] (n 1,018, 11 logs): PASS-shaped, not REBUILD. Tc20s -0.727 [-1.882, -0.057] (reported only).
- Controls ii, iii, iv, coverage PASS; (i'') bank-run validity PASS standalone; supplement of (i') labels-only PASS.
- Membership 1,018 / 1,018 verified; fallback token `d06ff3cbd9025da7` is not among the rows.

## Exact next step
Get `labels_rekeyed.jsonl` (sha256 `90318b56cb88e55ad15dccbebf3b0ec843fdd17773f1811437650d88184b5c5e`, 150,603 records) from the pod `/workspace/data/p5c_label_a33/` (pod use is a PI/MM call; the brief forbade it). Then:
```
set TMP=F:\tanitad_scratch\wa7rj5\tmp & set CUDA_VISIBLE_DEVICES=-1
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe w_a7_read.py --run run --stage analyze ^
 --traj-source F:/tanitad_scratch/rj5nt/join_c/rj5_bank_join.jsonl.gz ^
 --expect-join-sha256 4670f88ac409e71a1f7e1bf02328d93229fb3e153ad63f4127f08d53bbfed739 ^
 --bank-run-result E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7/run/w_a7_result.json ^
 --c1-table F:/tanitad_scratch/rj5nt/label_a33/P5C_label_table_p5r-rj5_rank0.json.gz --c1-records <labels_rekeyed.jsonl>
```
It takes seconds. Back up `run/w_a7_result.json` first (it is overwritten).

## Deletions
None. Scratch `F:/tanitad_scratch/wa7rj5/` holds ~70 KB of copied frozen documents (A2.md, A3*.md), `hf_list.py`, and an empty `tmp/`; keep until the result is accepted.
