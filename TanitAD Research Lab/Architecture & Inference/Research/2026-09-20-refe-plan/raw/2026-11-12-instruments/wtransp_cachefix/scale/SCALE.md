# W-TRANSP `--metric-cache-root` at scale: harness-identical on all 12,146 navtest tokens (2026-11-12, MM)

**PASS.** The registered teacher T seam, scored through W-TRANSP's tracked arm with `--metric-cache-root`, matches the banked pinned-harness CSV on **12,146 / 12,146** tokens. **0** tokens are over 1e-6 on any of NC, DAC, DDC, EP, TTC, C and PDMS.
- Largest |d|: EP 2.44e-11, PDMS 1.02e-11; the other five are 0.0.
- 0 error rows; all 3 shards rc 0; 560 s wall on 3 CPU processes.
- **Evidence class:** MEASURED. **Tier:** T1, executed (LQR-tracked by W-TRANSP's own tracker, identical to the harness here).

## What was run
| item | value |
|---|---|
| plans | `refe_p4_teacher_navtest.npz` (12,146 tokens; sha256 `29c8129f…`), handed as `plans8` (`t_plans8.npz` sha256 `776923a8…`) |
| reference | `refe_p4_teacher_navtest.csv`, the banked harness output of 2026-10-05 (sha256 `2c5ccb91…`) |
| scorer | `eval/instruments/transp_score.py` at tip `60b3ef23` (= `a3cc6a77`; the file is unchanged between them), `--split navtest --metric-cache-root F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest --no-fam --shard i/3` |
| driver | `drive_scale.py` (this folder); `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=-1`, navsim-crun venv |
| rows | `rows_scale.tar.xz` (3 shard files; per-token tracked and transparent sub-scores) |

## Why the other two arms of the 60-token check were not repeated at scale
- **The DB route without the flag** is already measured at scale. TF-N1's void read put N1-T, the same T plans, through it: **3,346 / 12,146** tokens over 1e-6 (`PK/raw/2026-11-11-teacher-fidelity/n1/run_void/RESULT_N1_VOID.md`). So at this scale the check can tell the two routes apart.
- **Mutation** (the routing line deleted) went RED at 60 tokens (`../results/check_cli_cachefix.json`). That proves the flag reaches the scorer, and the scale changes nothing about it.

## What it licenses, and what it does not
- With the flag, W-TRANSP's tracked sub-scores on navtest can be quoted as harness-identical. Its families (computed on W-TRANSP's tracked-executed states) then sit on the harness scenery.
- Navtrain has no harness cache, so navtrain W-TRANSP levels stay contrast-only.
- Callers still have to pass the flag; it is off by default. TF-N1 A1 and TF-N2 A1 were frozen without it, and their families stay labelled "not harness-identical".
- The selection arms of a card are unaffected: this is a control on a reference seam, and it uses no K-ledger entry.
