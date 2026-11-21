# RESULT -- step-5,000 G0 detection residual `eval_box3d_det_ap2_automobile_0_20` (EvalFlyWheel 2026-10-16)

**Status: PARTIAL.** Mechanism localised on banked data (CPU, MEASURED); the registered GPU arms A0/A1/A2/A4 have NOT run -- the shared GPU lock is held since 2026-10-16T15:29:23 by a
FOREIGN job (`refe-stage2-p2s1_full`, never touched). **No R1/R2/R3 verdict exists; G0 is NOT marked PASS.** The queued chain runs the arms at the first free lock and applies the committed rule by itself.
PREREG (written before any arm; Addendum 1 before any CPU arm): `PREREG_DETECTION_RESIDUAL.md`. Artifact: `detection_residual_probe.json` (ckpt md5 a32e3b70f46aabbd3ab55f285d03f82d, perm sha 99adbc7e...).

## 1. How the in-run row vs the dev-box G0 differ in path (file:line, launch tree C:/lgs/Lfb47a58376)
| item | in-run (Thor) | dev-box G0 |
|---|---|---|
| compile | `--trunk-compile` in config.json argv[137]; `refc_v3_train.py:423`, `refc.py:1503`, `timm_trunk.py:693` `torch.compile(self.net, backend=inductor)`; config `trunk_memory_levers.compile = inductor` | `refcv7_loader.py:144-146` DROPS the flag; `g0.json` model.trunk_memory_levers_built has no compile key |
| autocast | `--trunk-bf16` argv[133]: `timm_trunk.py:849-869` `_backbone` = bf16 autocast around the BACKBONE only (chunks of 8 frames, `--trunk-chunk-ckpt 8`), outputs `.float()`; heads / planner fp32 | identical levers (bf16, NHWC, fold-BN 104, dedup, chunk 8) |
| windows | `refc_v3_train.py:9953-9956` randperm seed 12345 -> 128 = 8 x 16, loop `:10853-10878` (`model.eval()`, no_grad, `compute_losses_v3`), packs pooled `:10899-10908` | `refcv7_loader.py:920-925` verbatim; `g0_refcv7.py:1015` same subset (perm sha 99adbc7e...); `run_eval` `:896` ports the loop |
| batch shape | 16 windows = 48 frames per backbone call = six 8-frame chunks | forward micro-batched `2,2,3,3,3,3` (`g0_refcv7.py:940,1074`): 6 / 9-frame calls -> other cuDNN shapes (arm A4 tests this) |
| decode | eval cache codec PNG (lossless; magic 89 50 4E 47 in `*.v2ep.pt` jpeg_buf) + eval-input md5 compare ALL IDENTICAL => bit-exact | same files |
| GT side | 219/219 COUNT terms OK incl. the GT automobile count n = 97 | same |
| hardware | Jetson Thor (compiled Inductor kernels) | RTX 4060 eager; A3 (Inductor) IMPOSSIBLE here: `import triton` fails, no cl/gcc/clang |

## 2. What the banked packs say (MEASURED, `g0.packs.npz` s0 + fp32_s0, `detres/pack_analysis.json`, `detres/event_finder*.json`)
* Pack path reproduces the banked values: s0 0.31453 (97 GT, 1,606 ranked rows, 74 TP), fp32_s0 0.31912 (75 TP). Support: 97 GT automobiles in 0-20 m (265 all bands) on 121 labelled windows, 300 slots/window;
  only **1** automobile slot is above the 0.5 presence gate, so this AP ranks low-confidence (0.1-0.4) detections.
* The deviation lives at the 2 m threshold only: in-run minus replay for this cell = +0.0011 (0.5 m), +0.0015 (1 m), **+0.0238 (2 m)**, +0.0066 (4 m). The replay AP-vs-threshold curve steps **+0.0197 between 2.00 and 2.05 m with the TP count unchanged** (1.95: 0.3102, 2.00: 0.3145, 2.05: 0.3342, 2.10: 0.3345).
* ONE event carries it: window 7, slot 129 (conf 0.361) sits **2.024 m** from its GT (fp32: 2.037 m) -- 2.4 cm outside the threshold; a competing slot (conf 0.253) takes the TP. Counting that slot as TP alone adds **+0.01838 = 77 % of the 0.02382 gap** (fp32 arm: +0.01626). Adding the other +0.05 m flips gives +0.01967 (83 %).
* Near-threshold census: windows with a TP/FP flip within +-0.05 m of 2 m: 5 (up, +0.0197 total) / 8 (down, -0.0042); within +-0.1 m: 14 (up, +0.0200) / 15 (down, -0.0100). 3 GT and 27 detections sit within 0.1 m of the 20 m band edge.
* Numerics scale (s0 vs fp32_s0, 6,973 automobile slots): centre displacement p50 0.030 m, p90 0.083 m, p99 0.237 m, max 1.02 m. So a 2.4 cm shift of one slot is INSIDE the typical bf16/fp32 jitter.
* P4 (SIMULATED, independent per-slot draws from that empirical pool, 200 draws/scale): fraction of draws moving the cell by >= 0.02382 = 5.0 % at x1 (sd 0.0086), 15.5 % at x2 (sd 0.0125), 23 % at x4, 47.5 % at x8 (upward-only 5 / 14.5 / 15 / 8 %).
  Reading: a >= 0.0238 excursion is an ~1-in-20 to 1-in-7 event at bf16-scale jitter -- plausible, not typical. Population caveat: across the 72 DETECTION (n >= 30) AP cells in-run minus replay has mean +0.0014, sd 0.0039 and this cell is the maximum (cells are nested/correlated, so no independent-sample test is claimed).
* Already banked by G0 itself: the SPEC-A6 fp32 arm moved this cell by +0.00459 (19 % of D*), i.e. one numerics realisation does NOT reach the gap; only the event structure above shows how few cm it takes.

## 3. Decision-rule status (committed in the PREREG, unchanged)
R1 NUMERICS_SUFFICIENT (a bar-clean numerics-only GPU arm with |v - v_A0| >= 0.02382) -> `AMENDMENT_D_R8_1_DRAFT.md` via `code/detres_amendment_count.py` (exact count of cells whose tolerance max(0.02, 2/n, 3 phi) changes verdict; NOT written yet, there is no arm data).
R2 NOT_NUMERICS (all |D| < 0.01191) -> localise: L1 GT counts and L2 decode are ALREADY clean (above); remaining suspects = micro-batch / chunk shapes (A4) and Thor's Inductor kernels (not testable here).
R3 PARTIAL otherwise. Current CPU-side lean (HYPOTHESIS, not a verdict): the term is a 2.4 cm single-event flip of a rank statistic; a numerics-floor reading is plausible but unproven.
CPU contingency arms C0/C2 (PREREG Addendum 1) were NOT run: host free commit fell to 0.01 GB while another session's jobs ran and the probe's own watchdog (floor) ended it before harming anything.

## 4. Blockers and what unblocks them
1. GPU lock held by `refe-stage2-p2s1_full` (pid 33488, since 15:29:23; end unknown); the G0-10500 chain waits behind it too. Unblock: lock release. Nothing to do on the MM side: `sh code/run_detres_gpu.sh "A0,A1,A2:1001,A2:1004" acq1` then `... "A2:1002,A2:1003,A4,A2:1005" acq2`
   are armed in the background (sh pid 25416; each waits for commit >= 5 GB, then the lock, <= ~18 min of lock, EXIT trap releases) and `code/detres_autofinalize.sh` then writes `DECISION` into the probe json. Re-launch the same commands if the session that started them is gone; they are resumable.
2. A3 stays impossible without Triton/MSVC; the only faithful test of the compile hypothesis is a Thor replay with `--trunk-compile` on/off after training (PI/MM call).
