# P2 full-navtest 64-proposal table: plan, budgets, gates (written BEFORE launch; job launched 2026-10-16 00:03 Berlin)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. This package = `<PK>/raw/2026-10-16-p2-full-table/` (code = archive of `origin/agent/arch-inf-20260803` @ `1b09570f`, `*.pt` excluded; E:/D: trees untouched).
Reuses the LANDED `p2_own_table` code (80775e9, unmodified); copied and parametrised: `c3_run_p2full.py` (diff `c3_run_p2full_vs_p2.diff`), `p2full_driver.py` (diff `p2full_driver_vs_p2tab.diff`). Both diffs are large because the dump/score stages became chunked and pipelined; every unchanged line (dump command, fidelity arithmetic, `read_csv`, table layout, the W3 harness call) is the landed one.

## What is built
P2 (`model_final.pt`, sha256 `ea31db22...`, rule `navsim_v1`) -> all 64 proposals + 6 scorer logits per token (dump) -> each proposal column scored by W3's harness UNCHANGED -> `table_p2_full.npz` over **all 12,146 navtest tokens**: `pdms [N,64]`, `sub [N,64,6]`, `valid`, `logits [N,64,6]`, `proposals [N,64,8,3]`, `pick`, `token` (sorted), `log`, `chunk`, `fingerprint`, `pdms_at_pick_banked`.
**Nothing is fitted on it.** Descriptive anatomy only (`anatomy.json`). It is the CONFIRMATION set of the planned DAC-aware selection-rule study: a rule fit here would void that role.

## Gates (all built into the job; the first failure stops it)
| gate | where | rule |
|---|---|---|
| preflight | start | ckpt sha256 == `ea31db22...`; banked csv sha256 == `61073a61...`; `navtest_full_tokens.json` sha256 == `927bbe38...`; chunk plan covers exactly the 12,146 tokens; `wait_idle.py` present |
| **same-pick, PER CHUNK, BEFORE scoring** | after each dump | pick pose vs P2's banked `stage2_p2_full.npz` <= 1e-3 m (expected bit-identical), fingerprints equal, 64 proposals, rule `navsim_v1`, `repair_last_heading` on, ckpt path, seam report rows == asked, misses 0, frame control <= 1 mm OR only on the 4 known logs (36 known over-bar tokens, `2026-10-04-navtest-final/frame_control_navtest_full.json`, max <= 6.19 m). **Fail => `P2FULL_FAILED` rc 2, that chunk and every later chunk are never scored. Report it; do not re-run.** |
| G3 | table | every (token, proposal) cell valid and finite |
| G2 | table | PDMS at the pick == banked csv, <= 1e-9, on bit-identical-pose tokens |
| **same-pick proof, ALL 12,146** | after the table | pose <= 1e-3 m, fingerprints equal, PDMS at pick == banked csv <= 1e-9 (exact), six sub-scores <= 1e-9, token set == the 12,146. Written to `same_pick_proof.json` (also written, as a stub naming the failing gates, if the job stops at rc 2). |
| cross-check | after the proof | the 509 zero-token rows of the full table are bit-identical (pdms, sub, valid, logits, proposals, pick) to the landed 509-token `table.npz`; the landed `measured()` on that subset must give `n_SEL 445 / n_PROP 64` again. Non-fatal (flagged in `P2FULL_DONE`), because a mismatch is a finding about determinism, not an instrument failure. |

## Time budget (the 509-token run's timing scaled, plus two direct measurements)
| item | MEASURED | source | scaled to 12,146 |
|---|---|---|---|
| dump, banked full run | 16,402 s for 12,146 tokens (1.35 s/token) | `data/refe_navtest/seams/stage2_p2_full.report.json` `seconds` | **4.6 h** GPU (+14 model loads ~0.15 h) |
| dump, 509 zero tokens | 838 s (1.65 s/token; sparser tokens per log) | `p2_own_table/STATUS.json` `dump_seconds` | (fit: 1.69 s/log + 1.33 s/token) |
| harness, one full-navtest column | 1,106.6 s, peak RSS 2,258 MB, 1 worker | `refe_stage2_p2_full.counts.json` | 0.091 s/token-column |
| harness, 509-token columns | 60-61 s each, 4 workers, 965 s for 64 columns | `p2_own_table/logs/p10.log`, `STATUS.json` | 0.118 s/token-column |
| **harness, a real 990-token chunk, this session** | **174.8 s**, peak RSS 724.6 MB, loaded machine | `C:/Users/Admin/p2full_work/_cal/cal.log` | 0.164 s/token-column + ~12 s start |

Score work = 12,146 x 64 = 777,344 token-columns + 896 starts. CPU-seconds: **83 ks** (0.091 s rate) to **138 ks** (0.164 s rate). With 6 workers: **3.9 h to 6.4 h**. Dump and score overlap (chunk c scores while c+1 dumps), so wall = max(dump 4.8 h, score 3.9-6.4 h) + ~0.4 h lag for chunk 0 = **~5.2 h to ~7 h after the GPU is acquired** (sequential would be ~9-11 h; the earlier estimate was 10.5 h). Central: ~6 h.
**The GPU start is not mine to set.** The scene-tap probe holds `devbox_gpu.lock` until ~00:30 Berlin (the job waits for the probe process itself to exit, not only for a gap in its lock holds), and `refcv8-g0-step5000` has been queued on the same lock via `gpu_lock.py acquire` since 21:47 (12 h max wait) and polls every 60 s, so it will almost surely be ahead of this job. ETA = (when the lock is mine) + 5-7 h.
The lock is kept from chunk to chunk and handed over at a chunk boundary only if a `gpu_lock.py acquire` waiter exists. Another job's lock is NEVER removed by this job, however old it looks (`gpu_lock.py`'s rule; a dead-looking pid may be an MSYS pid): it is reported in `STATUS.json` and waited on.

## Disk budget (measured per-file/per-token sizes of the 509 run, scaled)
D: and E: are the same exFAT drive: **1 MiB allocation unit, 16.7 GiB free at launch** (Win32_Volume). C: is NTFS, 4 KiB units, 211 GiB free.
| what | where | size | basis |
|---|---|---|---|
| W3 run dirs `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/refe_p2full_main_c*_p*` (hard-coded EXP in W3's `run_v1.py`) | D: | **5.25 GiB** = 896 labels x 6 files x 1 MiB | `du` = 6,144 KiB per label at 509 tokens (file COUNT, not tokens, drives it on exFAT) |
| 14 dump chunks + `table_p2_full.npz` + manifests | D: `TanitAD-artifacts/p2_full_table/` | ~0.25 GiB | proposals.npz 4.03 MB and table.npz 3.79 MB per 509 tokens -> ~96 MB + ~90 MB |
| **D: total** | | **~5.5 GiB of 16.7 free** (leaves ~11 GiB) | guard: the job PAUSES (never writes) if D: free < 3 GiB, status `guard` |
| 896 seams (+14 dump-run seams) | C: `C:/Users/Admin/p2full_work/seams` | 0.34 GB | 164,218 B per 509-token seam = 322.6 B/token/column |
| harness outputs (`score/`), 6 files per label | C: `.../p2full_work/score` | ~1.5 GB | 1.0 MB per 509-token label, scaled by tokens |
| second copy of the table | C: `.../p2full_work/final/` | 0.09 GB | the table must not live on the external drive only |
| **C: total** | | **~2.0 GB** of 211 GiB | |
Sufficient on both drives: no STOP. (My own smoke/calibration run dirs, 129 x 6 MiB = 0.8 GiB on D:, were deleted by exact name after use.)
Why harness outputs are on C:: W3's `cmd_score` ends in `os.path.relpath(csv, PKG)`, which raises across drives; PKG (this archive) is on C:.

## RAM budget
W3's wrapper aborts a harness run (exit 3) once system-available RAM stays below 3,000 MB, and that kills every concurrent run. Available RAM at launch ~8 GB, because the probe holds ~8 GB working set; it should be ~17 GB once the probe exits, and the job does not start before that. One column peaks at ~0.75 GB, the dump at an unmeasured ~4 GB (ESTIMATED). Hence: a start only with >= 5,000 MiB available, starts staggered >= 15 s, 6 workers, and a W3 RAM abort is retried (120 s) without counting as a failure.

## Smoke evidence (this session, before launch)
- Real tools, CPU, 8 P2 zero tokens in 2 chunks of 4, all 64 columns through the unchanged harness: gates PASS, G2 PASS, same-pick proof PASS (pose tolerance loosened to 5 mm for CPU-vs-GPU, smoke only; the main job is pinned to 1 mm in code), `P2FULL_DONE`. `_smoke/`.
- Real harness on a real 990-token chunk fed P2's BANKED poses: PASS, command line accepted (longest chunk override string 19,466 chars of the 32,767 limit), and its 990 x 7 score columns equal the banked full-run rows exactly (max |diff| 0.0). So scoring a chunk alone reproduces scoring inside the full set.
- Tests: `test_p2full.py` (see `test_p2full.log`): synthetic known-answer world + fake runner; every gate and the lock protocol have a deliberate-regression arm that must go RED.
- End-of-job analysis wiring dry-run on the landed 509 table: cross-check PASS, `measured()` reproduces `n_SEL 445 / n_PROP 64`.

## Not verified before launch (honest list)
- The GPU dump of a chunk in THIS code path (the command is the landed one; the CPU path ran; the lock protocol is tested against mocks and read-only against the real lock).
- The dump process's RAM (needed to size the overlap) and the real harness rate under the overlap.
- `wait_idle.py` ran once for real (idle after 47 s) but the full take-lock sequence has not run against a free real lock.
