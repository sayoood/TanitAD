# BUILD_AMEND7: build report for AMEND_HWMPLAN_7 (RA7, a nonlinear 0-2 s acceleration readout), written 2026-11-18

Spec: `amend7/AMEND_HWMPLAN_7.md`, sha256 `95c97b39f2692aeb9bff67383e8f8ba58771aeee18739d8693130e80297f7be9` (checked before the build and by every chain run). Brief: `BRIEF_BUILD_AMEND7.md`. Tier T0 throughout; CPU only; zero world-model training; no box 2, no pod. Every file named below lives in `F:/TanitAD-scratch/actrsp/lonplan/amend7/build/`; the sha256 of each is in `MANIFEST_sha256.txt`.

## 1 What was built

| file | role |
|---|---|
| `amend7_env.py` | points AMEND_5's modules at the amend7 folders, `ST.SPEC` / `ST.SPEC_SHA` at the card, `ST.RA_SHA` from `RA7_FIT_<k>.json` (else `'0'*64`), and replaces `ra_l1.load_readout` / `ra_l1.plan_arrays` by the `ra7_l1` versions; no RA5 or RA6 fallback |
| `mlp7.py` | pure functions: the net, seeds, selection (2.3), cross-fit, LinearD, planted and shuffled targets, the bootstrap, the gate (section 4), every pure FIT control with its regression hook |
| `pairs7.py` | the pair table (z0, z10, y, clip index in both orders, fold, frame, swap partner, 10 scaled actions, F_true, F_alt, GAIN7 anchors and their +/- delta rollouts); asserts C-PAIRS7 and C-ISO7 on build |
| `fit7.py` | CLI `crossfit`, `final`, `gain`, `gate` |
| `ra7_l1.py` | `MLPReadout`, `load_readout7` (sha256 before parsing), the `plan_arrays` fork |
| `amend7_controls.py` | `fit` (8.1), `run` (8.2), `merge --which fit|run` |
| `amend7_run.py`, `amend7_read.py` | SCORE-stage planner (arms RAL1, RAL1B, NEG, SHUF, ORACLE, OFF; MEAN-L1 reference copied from AMEND_6 after a sha256 check) and readout (`train`, `score`, `outcome7`) |
| `test_amend7.py` | 30 tests, synthetic data only |
| `run_amend7.sh fit|score`, `amend7.cmd`, `launch7.ps1` | the chain |

## 2 Results of the build

* **Tests:** `test_amend7.py` 30 passed (149 s, one pytest at a time, `CUDA_VISIBLE_DEVICES=-1`, `--basetemp` on F:). Every regression arm of the card's 8.1 reads RED in a test (C-PAIRS7 flipped chunk byte, C-ISO7 planted SCORE eid, C-CURVE7 sorted-id fold map, K-control misaligned targets, C-FOLD7, C-SEL7, C-CONST7, C-ZERO7, C-WM7 exchanged rollouts, PLANT P3, SHUF-T identity shuffle); so do C-GRAD7 (detached; reading after 12 actions) and C-PLANT7 (i). Also pinned: LinearD equals `fit_ra6.ridge_fit` within 1e-8; same seed bit-identical, seed sets A and B differ, batch order changes with the seed; selection never reads the outer test fold and a tie goes to the earlier config; `gate7` reads R7-FIT-PASS on a planted nonlinear truth and R7-FIT-FAIL on a linear one; the section 4 and section 7 outcome tables row by row; the `plan_arrays` fork equals `ra_l1.plan_arrays` within 1e-6 for a readout exactly linear in D (modes l1, neg, shuf, oracle, off); a readout with a mismatched sha256 or one changed byte refuses; a chassis with no `RA7_FIT_<k>.json` gets `'0'*64` and refuses.
* **Smoke on the real s0 data (plumbing only):** `pairs7_s0.npz` built in 47 s: 58,934 pairs in 1,000 clips, C-PAIRS7 green (D, y and clip index equal to `fit_ra6.pairs` bit for bit), C-ISO7 green (0 SCORE episodes), the chunk sha256 list and `CHUNKS.json` sha256 `e144d1e3fea1e8b707c57cad89a20a80a3498333dee2709107daaa970432f763`. Table sha256 `8c0f61382d32ef772538974cb30ead87dd93b913e08ae671c47cd4ff244a4a7d`, 5,922 anchors. RA's cross-fit reproduced the 1,000 row of `FIT_CURVE_s0.json` (r2, r, both CIs) to 6.7e-16 (C-CURVE7 bar 1e-9). One MLP fit, one config, 1 epoch, on 50 clips (2,946 rows): shapes and finite outputs only, metrics not read. Time per epoch at 47,000 rows with a synthetic target: 0.75 s (2 threads; the card's basis is 0.55 s).
* **Synthetic plumbing runs in scratch roots (deleted afterwards):** the whole FIT chain on synthetic pair tables with tiny learners (crossfit of all eight arms, final A and B, GAIN7, `fit_controls` on both chassis, merge, gate and both result files); the `run` controls on real TRAIN windows with a synthetic-trained readout (11 of 12 green; the twelfth, C-PLANT7-ii-smoke, reads RED only because that readout is flat: its one-iteration regression arm stays under the bar, which AMEND_6's RA does not; the real RA7 is not flat); the TRAIN-stage planner and `amend7_read.py train` on s0; `amend7_read.py score` on AMEND_6's finished folders as stand-ins (it reproduced AMEND_6's RA-L1 - HOLD +0.4547 and floor 0.0042 at s0, and the new contrasts and `outcome7` ran). No amend7 number was produced by any of these.

## 3 Reachability trace (argv to the knots written to plan_raw)

1. Arm name and argv: `amend7_run.py:11` `SCORE_ARMS` (RAL1 and RAL1B are `--arm RAL1`, RAL1B only with another `--out` folder and readout B); `amend5_stage.py:111` `ST.build` names `--ra-l1 l1 --ra-src l1 --mean-src none` (a default of `off` would disable the pass, so every flag is named and `ra_l1.check_declared` refuses an unnamed one); `amend7_run.py:52` `R.run_arm(...)`.
2. `amend7_env.py:44-45`: `R.load_readout = ra7_l1.load_readout7` and `R.plan_arrays = ra7_l1.plan_arrays7`; `ra_l1.py:398` `run_arm` calls `plan_arrays(...)`, which resolves to the fork at call time; `ra_l1.load_ctx` loads the readout through `load_readout`.
3. `ra7_l1.py:63` `plan_arrays7`; line 105: `ra_fn = lambda Dz: ctx.ra.read(Dz, z0, shuf=...)` (the batch's z0 on every call, RAL1, NEG, SHUF and ORACLE paths: lines 105, 115, 118, 121).
4. `ra7_l1.py:31-39` `MLPReadout.read`: `x = [z, z + Dz]` (z0 broadcast to Dz's shape), standardised, through the 256-256 GELU net.
5. Inside the cost of `ra_l1.ra_pass`: `ra_l1.py:174` `r = ra_fn(roll(st, z0, a_eff, read_after) - z0.unsqueeze(1))`; the gradient reaches the knots through `roll` (the stack's level-1 rollout, read after 10 actions).
6. `ra_l1.py:180` `st._optimize(...)` (the library, `hjepa.py:948`, differentiates with `torch.autograd.grad` at line 978).
7. `ra7_l1.py:125` `out["plan_raw"][:, :KS, 0] = acc_out`: knots 1-10 only; yaw and knots 11-30 stay the base's bytes (C-YAW-ID / C-BAND-ID).
Mutation checks of the trace: C-GRAD7 (gradient exactly 0.0 on knots 11-12 and yaw; RED when RA7 is detached or reads after 12 actions) runs on the toy stack and on 16 real TRAIN windows in the `run` stage; the fork-equivalence test pins hops 3-4.

## 4 Where the build departs from, or interprets, the card (the card was not edited)

1. **Launch scope.** The card says that on R7-FIT-PASS the SCORE read runs at once in the same chain. The MM ordered FIT only and the SCORE launcher built but not run. `run_amend7.sh fit` ends after the gate; `run_amend7.sh score` (refuses unless `G7_FIT.json` reads R7-FIT-PASS) is the card's step 5. Running `fit` then `score` is the card's single chain split in two.
2. **Chassis order.** Serial, as ordered (the card allows side by side at 11 GB; `AMEND7_PARALLEL=1` enables that).
3. **C-WM7's roll check** runs in float64: in float32 a 10-knot and a 12-knot call differ by 1.4e-6 (kernel shapes), above the card's 1e-6; in float64 the difference is exactly 0.0. The float32 value is recorded.
4. **C-PLANT7 (i)** "reaches the target read within 1e-6" is read as final cost (squared read error) at most 1e-6; on the toy the pass reaches 1e-8 (read error about 1e-4) after 90 iterations. The analytic identity (cost 0 at the source's actions, at 1e-6) and "moves off HOLD_c" are checked as well.
5. **Epoch grid.** A weight decay's runs for 2, 6, 12 and 24 epochs are one 24-epoch run with snapshots: the batch order of epoch e depends only on (seed, e), so each snapshot is bit-equal to its own run (a test pins it). Same function, about 45 % less CPU.
6. **RA7-SHUF** permutes D's coordinates by the inverse index of `RandomState(0).permutation(128)`, so for a readout linear in D it equals RA-SHUF's `W[perm]` (the fork-equivalence test covers shuf).
7. **FIT-stage C-PAIRS7** reads the build's own assertion (equal to `fit_ra6.pairs` bit for bit) from `PAIRS7_<k>.json`, re-verifies the table's sha256 on load and every chunk's sha256 at control time; its regression arm is a flipped chunk byte.
8. **MEAN-L1 reference.** AMEND_5's `score()` and `train_stage()` read a MEANL1 folder in the root they are given, so `amend7_run.py` copies AMEND_6's MEAN-L1 folders in after checking each plan file and `windows.npz` against its `SOLVER_RA.json` (the card: references are read from existing files, sha256 checked).
9. **CEIL-ORACLE7** uses AMEND_5's oracle floor (the ORACLE arm has inference seeds 0 and 1; there is no readout-B ORACLE arm).
10. **GAIN7:** for RA6 and RA7-A, s is that readout's own cross-fit s_swap at N = 1,000 (`fit/RA_<k>.npz`, `fit/MLPA_<k>.npz`); RA5 uses Option B's s_a. r_swap is the Pearson r of (R(real) - R(swap)) with (y - y'). The r_wm of RA5 on the 859 new clips uses B 2,000 over those 859 clips; GAIN7's means use B 1,000, as the card says.
11. **RA7_FIT_<k>.json** keys: `sha256` (readout A), `sha256_B`, `readout`, `readout_B`, with the chosen epochs, weight decay and the full selection record.

## 5 Not exercised before the FIT launch

The real-scale MLP selection (the 12-config grid on 47,000 rows; only the 50-clip, one-config, one-epoch fit and the synthetic-target timing ran), `fit7.py gain` and `fit_controls` on real tables (both ran on synthetic tables of the real shapes), the bash chain end to end (syntax checked; each Python step ran separately), the `run` controls with a real RA7 readout, the SCORE-stage arms RAL1B / RAL1 on the 994 real windows, and `amend7_read.py score` on real RA7 plans. Real-scale memory is ESTIMATED at about 1 GB per FIT process.

## 6 Deletions (own temporary data; literal paths)

* `F:/TanitAD-scratch/actrsp/lonplan/amend7/work/smoke1` 81,826,408 bytes
* `F:/TanitAD-scratch/actrsp/lonplan/amend7/work/smoke2` 250,064,051 bytes
* `F:/TanitAD-scratch/actrsp/lonplan/amend7/work/tmp/bt1`, `bt2`, `bt3` (pytest base temps, a few MB together)
Kept: `pairs/pairs7_s0.npz` (134,591,220 bytes) with `PAIRS7_s0.json`; the chain reuses it when its sha256 matches. The chain deletes `pairs/` and `work/` at its own end and logs the sizes in `read.log`.
