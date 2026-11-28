# P1PB_RESULT: P1′b (THEORY_CRAWL_AMEND_2) on the RA-1 holdout tick-0 windows

**VERDICT, as the criterion is written: FAILS.** Both groups clear the floor of 30 at both seeds (CONFIDENT 42, UNSURE 125). Δ is positive at both seeds, and is separated at s1; at s0 the lower end of the 95 % interval is -0.028. The criterion (AMEND_1 §4): "P1′ fails if the difference is not separated at either seed, or is negative."

Written consequence (`THEORY_CRAWL_AMEND_1.md` §4): the under-acceleration is not a mean over unresolved intent, and T1′ is wrong too. Nothing below moves a threshold, a band or a group; the descriptive material in §5 to §7 is labelled and does not change this line.

The probe can test T1′ on this set: its AUC on the holdout windows is 0.748 [0.628, 0.849] in the P1′ band where its label exists (v0 in [3, 6), n 587, 31 episodes) and 0.741 [0.594, 0.864] on the P-MOVE range (n 942, 31 episodes); neither interval covers 0.5.

Evidence class MEASURED (ours, dev-box CPU, zero planner training, no weight edit, nothing staged). Tier: a tick-0 plan-level read of the C0 checkpoints on logged windows (the plan is made from the logged history); not a driving result. Estimator: episode-cluster bootstrap over the holdout clips, B 2,000, seed 0 (the `tp_lib` construction). It answers "another draw of EPISODES" and is blind to training and inference variance (H-ESTIM-SEED-1); the two planner seeds (two training runs of the same recipe) are read separately, and a separated interval from them is necessary, not sufficient, for any lever claim.

## 1. Spec and order of work
* `THEORY_CRAWL_AMEND_2.md` sha256 `0427c32592434e3e...` (frozen, verified); `THEORY_CRAWL_AMEND_1.md` §4 sha256 `9d39ac53fbb55773...` (frozen, verified). The rule is exactly P1′'s; the P1′ code and probe artifact (`p1p_tp_lib.py`, `p1p_probe_refit.py`, `p1p_read.py`, `P1P_PROBE_MODELS.npz`) are loaded unchanged and their sha256 equal the values recorded in `P1P_RESULT.json`.
* Order: (1) the holdout list and admissibility, `P1PB_ADMISSIBILITY.json`, before any score; (2) `SPEC_P1PB.md` sha256 `f2cc7efbc1454c07746b231d7c8cfc7d486d59298900fb47aef9ee80e0aa7ecc` stamped in `SPEC_P1PB.sha256` (2026-11-07T00:07:38Z, 01:07:38 +0100) before any holdout feature, forward or posterior existed; (3) control FW (the driver reproduces the T1-kin dumps' tick-0 rows) before the holdout forward; (4) the holdout tick-0 forward of C0 s0 and s1; (5) the read.
* Rule fixed in the spec before scoring, stated here because it decides which table is the verdict: the primary window set is EVERY window of the 60 clips (stride 1, t in [3, n − 30), the training enumeration), as AMEND_2 says "every window passing the filter is read"; the eval-141 grid construction (t = 3 mod 4) is the declared sensitivity G4. Clips with no row in the dev-box v9 label release (nav input all zero, a state C0 never saw) are outside the primary set and inside S5.
* Run history: `p1pb_admissibility.py` ran once (its first launch failed on shell quoting before any line executed); `p1pb_forward.py control`, `p1pb_forward.py holdout` and `p1pb_read.py` each ran once, and none of those three was edited after its first run; `p1pb_finalize.py` (report text only, no statistic) was re-run after wording edits. Before the spec, three scratch probes (kept in `p1pb/`: `ho_list.py`, `chk_cache.py`, `chk_inputs.py`) read identity and structure only: the holdout list and its shas, manifest records and file sizes of the cache, the number of holdout clips with native tracks and with v9 rows (56 of 60, which is why the spec has a label-coverage rule). No feature, v0, a_human, plan or posterior of a holdout window existed.

## 2. Admissibility of the holdout set (checked before any score; all pass)

| check | what | result |
|---|---|---|
| A1 holdout list | clips 540:600 of the sha12 order of the 4,572 cache_full TRAIN clips (PREREG_V8F_MINI Amendment 4, `--exclude 540:600`); rebuilt from the box 2 manifest copy | sha256 `f6a1dee02af0b96f...` equals the digest every run of the chain recorded (`excluded_sha256`); train list `37f4f9ee468e0872...` equals `train_sha256`; overlap 0 |
| A2 no holdout clip in C0's training clip list | the summaries and argv of M2b, dagoacc and C0 (s0 and s1): C0_s0, C0_s1, dagoacc_s0, dagoacc_s1, M2b_s0, M2b_s1 | all six: n_clips 4,512, `--exclude 540:600`, `excluded_sha256` and `train_sha256` as above; chain M2b → dagoacc → recft_C0, every stage excluded the same 60 |
| A3 no holdout clip among the go/stay probe's training windows | `windows_train.npz` (306,566 windows, 4,512 clips) and the sampled rows | window-list digest `e144adcd9b6273f2...` equals C0's `train_window_list_sha256`; holdout clips among them 0; among the 12,000 sampled P-MOVE rows (2,140 clips) 0 |
| A4 disjoint from eval-141 | `join_eval141.npz` clip list | overlap 0 of 141 |
| A5 frozen features on the dev box | `F:/TanitAD-scratch/speedread/cache/train/clips` (the cache the probe's own V was extracted from) | 60 of 60 files present with md5 equal to the box 2 manifest; same encoder route as the eval-141 cache (float8_e4m3fn, bf16, cuda, local256, grids 4x10 and 8x20); 6,056 cache steps |

Provenance limits, stated: the box 2 manifest is read from a copy (`t2_scenes/raw/box2_cache_full_train_manifest.jsonl`); the dev-box v9 release is f63ece41 (4,369 clips) while the C0 chain trained on 83e30bd3 (4,572 clips), whose rows for the old 4,369 clips are bitwise identical on the non-offset fields (INHERITED from `SPEC_REFCV8_DRAFT.md` line 1542, not re-verified); 4 holdout clips have no f63ece41 rows (270 windows) and are outside the primary set.

## 3. Window set and group counts

| step (primary: every window of the labelled holdout clips, v0 in [3, 8), `plan`) | windows |
|---|---|
| tick-0 rows of the 60 clips (every t in [3, n − 30)) | 4076 |
| in the 56 clips with v9 rows | 3806 |
| and v0 in [3, 8) | 1180 |
| and gt_ok | 1180 |
| and a_human > 0.5 m/s² | 291 |
| and all of a_human, â_plan, ρ, posterior finite | 291 (23 episodes) |

Posterior bins over the 291 windows: below 0.35: 39; UNSURE [0.35, 0.65]: 125; (0.65, 0.8): 85; CONFIDENT ≥ 0.8: 42. The windows, a_human and the posterior do not depend on the planner seed, so both seeds hold the same groups. CONFIDENT spans 11 episodes (largest clip holds 24 % of its windows; median 3 windows per clip), UNSURE 21 episodes (largest clip 12 %). Stride-1 windows overlap within a clip, so the support is the number of episodes, not of windows.

Domain note: the probe was trained on v0 below 6. 106 of the 291 windows have v0 in [6, 8) and were scored by extrapolation (S1 restricts to the in-domain band).

## 4. The statistic, holdout alone (this table is the verdict)

| seed | n CONF / UNSURE (episodes) | median ρ CONFIDENT [95 %] | median ρ UNSURE [95 %] | Δ [95 %] | Δ > 0 and separated |
|---|---|---|---|---|---|
| s0 | 42 / 125 (11 / 21) | 0.740 [0.556, 0.856] | 0.495 [0.372, 0.652] | +0.245 [-0.028, +0.407] | no |
| s1 | 42 / 125 (11 / 21) | 0.702 [0.554, 0.859] | 0.390 [0.299, 0.579] | +0.312 [+0.068, +0.463] | yes |

Median a_human (m/s²) is 1.396 in CONFIDENT and 0.817 in UNSURE; median â_plan is 0.916 / 0.468 at s0 and 0.925 / 0.385 at s1; median v0 (m/s) is 4.33 in CONFIDENT and 5.51 in UNSURE. Means of ρ: CONFIDENT 0.746 / 0.742, UNSURE 0.486 / 0.402 (s0 / s1).

## 5. Reported beside, deciding nothing: the pooled read (eval-141 primary windows ∪ holdout primary windows)

eval-141's 168 windows are SEEN (they produced the NOT EVALUABLE of P1′ and its descriptive Δ). The two clip sets are disjoint (checked); the pooled set holds 459 windows from 68 episodes.

| set | seed | n CONF / UNSURE | Δ [95 %] | Δ > 0 and separated |
|---|---|---|---|---|
| eval-141 alone (P1′ primary, seen) | s0 | 24 / 75 | +0.365 [+0.213, +0.534] | yes |
| eval-141 alone (P1′ primary, seen) | s1 | 24 / 75 | +0.437 [+0.240, +0.585] | yes |
| holdout alone (the verdict) | s0 | 42 / 125 | +0.245 [-0.028, +0.407] | no |
| holdout alone (the verdict) | s1 | 42 / 125 | +0.312 [+0.068, +0.463] | yes |
| pooled | s0 | 66 / 200 | +0.301 [+0.140, +0.434] | yes |
| pooled | s1 | 66 / 200 | +0.329 [+0.169, +0.510] | yes |

The pooled logic returns HOLDS. It is a table of two sets, one of them seen, and it is not promoted over the holdout line. The eval-141 and holdout point estimates agree in sign; the holdout estimates are smaller by 0.120 (s0) and 0.124 (s1).

## 6. Controls

| control | what it checks | result |
|---|---|---|
| C4 unit tests | weighted median = `np.median` of the replicated array; bootstrap counts = tp_lib construction; group edges at 0.8 / 0.35 / 0.65 | pass |
| FW the driver reproduces the T1-kin dumps (run before the holdout forward) | 240 evenly spaced eval-141 tick-0 windows, both seeds, vs the dump rows | pass: v0 and gt max abs 0e+00 / 0e+00; plan max abs 6.3e-05 m (s0), 7.7e-05 m (s1); plan_raw 1.6e-05 / 1.5e-05 m; â_plan 4.8e-06 / 2.4e-06 m/s² (tolerances 1e-6, 1e-6, 5e-3, 5e-3, 2e-3) |
| F1 holdout feature path | store pose grid = native track (checked per clip by the store), n_frames and gt_ok on every enumerated t, finite plans, identical v0/gt between seeds, v0 = kin3[t0, 0] | pass: 4076 windows, v0 vs kin3 max abs 0.0, v9 release md5 f63ece41 |
| F2 features rebuilt twice | 50 random windows, V and E built twice from the cache | pass: V bit-equal True, E max abs 0.0, v0 vs kin3 0.0 |
| Probe artifacts unchanged | sha256 of `P1P_PROBE_MODELS.npz`, `p1p_tp_lib.py`, `p1p_probe_refit.py`, `p1p_read.py` vs `P1P_RESULT.json` and `trigger_probe/tp_lib.py` | pass |
| K1 probe AUC on holdout, P-MOVE range | posterior vs GO, 0.5 ≤ v0 < 6 | 0.741 [0.594, 0.864], n 942 (499 GO, 31 episodes); above chance |
| K2 probe AUC on holdout, P1′ band | posterior vs GO, 3 ≤ v0 < 6 | 0.748 [0.628, 0.849], n 587, 31 episodes; above chance |
| K3 two-second go indicator | posterior vs a_human > 0.5, 3 ≤ v0 < 8 (descriptive) | 0.842 [0.748, 0.919], n 1180 (291 positive, 36 episodes) |
| N1 posterior shuffle | 20 permutations of the posterior, real ρ | pass: separated 1 of 20 at s0, 0 of 20 at s1 (limit 3) |
| N2 identity (ρ ≡ 1, also from `plan := gt`) | exact values | pass (medians 1.0, Δ 0.0, interval [0, 0], not separated) |
| N3 planted (ρ = posterior) | the pipeline sees a planted mechanism | pass (Δ +0.333 [+0.297, +0.389] at s0, separated) |

## 7. Sensitivities (declared in SPEC_P1PB.md §5; none is the verdict, each through the identical verdict logic)

| variant | n CONF / UNSURE | Δ s0 [95 %] | Δ s1 [95 %] | verdict logic returns |
|---|---|---|---|---|
| G4: t = 3 mod 4 (eval-141 grid construction) | 12 / 30 | +0.217 [-0.060, +0.510] | +0.397 [+0.064, +0.671] | NOT EVALUABLE |
| S1: v0 [3,6) | 35 / 72 | +0.102 [-0.151, +0.397] | +0.254 [-0.009, +0.401] | FAILS |
| S2: plan_raw | 42 / 125 | +0.260 [+0.002, +0.417] | +0.330 [+0.083, +0.467] | HOLDS |
| S3: all-train probe posterior | 43 / 126 | +0.207 [-0.056, +0.377] | +0.293 [+0.052, +0.430] | FAILS |
| S4: GO24 posterior | 124 / 74 | +0.253 [-0.007, +0.485] | +0.317 [+0.099, +0.461] | FAILS |
| S5: all 60 clips (zero nav where no v9 rows) | 42 / 127 | +0.232 [-0.036, +0.407] | +0.296 [+0.055, +0.457] | FAILS |

Reading of the table: Δ is positive at both seeds in every variant. Separated at both seeds: S2; at s1 only: G4, S3, S4, S5; at neither: S1. Variants returning NOT EVALUABLE: G4 (G4 keeps 12 CONFIDENT windows against the floor of 30). S2 (`plan_raw`, the network's plan before smoothing and projection) is a different plan field from the one P1′ names and is not promoted. The primary line uses every window, as fixed before scoring.

## 8. What the read says and does not say (descriptive; HYPOTHESIS beyond the verdict line)
* **The written verdict stands: FAILS.** It rests on one interval: at s0 the 95 % interval of Δ is [-0.028, +0.407], 0.028 below zero, from 11 CONFIDENT episodes; at s1 it is [+0.068, +0.463]. The rule requires both seeds; no threshold, band or group is moved to change that.
* **The direction is the predicted one at both seeds.** Δ is +0.245 / +0.312 on the holdout against +0.365 / +0.437 on the seen eval-141 windows. A shrinkage that is larger where the probe is unsure is present in both sets; what the holdout does not give is a separated interval at both seeds with 11 CONFIDENT episodes.
* **A confound the grouping carries (post hoc, not tested):** the CONFIDENT windows have median v0 4.33 m/s against 5.51 and median a_human 1.396 against 0.817 m/s²: go-confident windows are the slower, harder-accelerating ones, so a difference in ρ between the groups is not by itself a signature of intent mixing.
* **Median ρ against the groups' median posterior (post hoc, descriptive):** CONFIDENT ρ 0.740 / 0.702 (s0 / s1) against a median posterior of 0.858; UNSURE ρ 0.495 / 0.390 against 0.525. T1′'s form â ≈ π·a_go + (1 − π)·a_stay gives ρ ≈ π when a_stay ≈ 0. The medians of ρ lie below the posterior by 0.12 / 0.16 in CONFIDENT and 0.03 / 0.14 in UNSURE (s0 / s1): the ordering and the rough size match, the level is lower. This is an observation made after the verdict, with no interval on the comparison, and it does not alter the verdict.
* **What is not available:** a larger unseen set on the dev box. The 4,512 training clips are inadmissible for this test; eval-141 is seen; the 60 holdout clips are all of the unseen clips with cached features here, and they are seen after this read. Another look at the same 60 clips under a changed rule would be a post hoc rescue and is not proposed.
* **Next levers (the Master Mind's to pre-register; none was run; zero training):** (a) a fresh pre-registration whose statistic uses the continuous posterior (ρ against π over all windows, including the neither-bin windows the split discards: 124 of 291 here), with a committed power statement, read on FRESH windows; (b) fresh windows need clips outside the 4,572 encoded by DINOv3: blocked on encode compute (box 2 or a pod) and on the PI's go for it, since the dev box holds no further unseen cached features; (c) P4 on the LoRA arm, the pre-registered next test of the lever that never depended on T1′ (T2 and T3 are unchanged by this verdict).

## 9. Provenance and manifest (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/`; nothing staged, committed or pushed; each exists in this one place only)

| artifact | role | sha256 (first 16) |
|---|---|---|
| `SPEC_P1PB.md` | pre-registered spec | `f2cc7efbc1454c07` |
| `SPEC_P1PB.sha256` | spec hash and time | `f8e37699b6afeccd` |
| `P1PB_ADMISSIBILITY.json` | admissibility checks A1 to A5 | `cc643b11785faf45` |
| `P1PB_HOLDOUT_60.json` | the 60 holdout clips | `64c7b42d80ab1132` |
| `p1pb_admissibility.py` | admissibility script | `e149f746cf7e83d1` |
| `p1pb_forward.py` | tick-0 forward (control and holdout modes) | `b2d2a92bed603c74` |
| `p1pb_read.py` | the read, loads the P1′ code unchanged | `4e17a0d32b8d1316` |
| `p1pb_finalize.py` | this report | `6ede8db9c07336a8` |
| `P1PB_FW_CONTROL.json` | control FW | `666ce16efe3e55b7` |
| `P1PB_F1_CONTROL.json` | control F1 | `0f54c2f7943ca6aa` |
| `P1PB_TICK0_s0.npz` | holdout tick-0 plans, C0 s0 | `ced1c8c5600eec7f` |
| `P1PB_TICK0_s1.npz` | holdout tick-0 plans, C0 s1 | `e185c9bebf90dc0e` |
| `P1PB_PERWINDOW.npz` | per-window posteriors, a_human, a_plan, ρ | `b980d027fcba50af` |
| `P1PB_RESULT.json` | machine-readable result | `b70b1f0ee9a0dbb0` |
| `p1pb_fw_control.log` | log | `c8e5de792af1b051` |
| `p1pb_forward_holdout.log` | log | `fef947867a0819f0` |
| `p1pb_read.log` | log | `bd5ad87a864892c1` |

Inputs read, unmodified: the checkpoints `v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt`; the F: train cache `F:/TanitAD-scratch/speedread/cache/train/{manifest.jsonl,clips/<60 holdout clips>.pt}`; `sfresh_inputs/inputs/native_tracks.pt`; `v9labels/v9_labels_train.npz`; `t2_scenes/raw/{box2_cache_full_train_manifest.jsonl,TRAIN_CLIPS_4512.json}`; the run summaries named in §2; `trigger_probe/raw/windows_train.npz`, `sample_pmove.npz`; the P1′ files `P1P_PROBE_MODELS.npz`, `P1P_PERWINDOW.npz`, `P1P_RESULT.json`; the T1-kin dumps `C:/Users/Admin/rr_scratch/out/C0C_s{0,1}/ticks/` (control FW only); `ri/code/v8f_t1_eval_ri.py`; the dagt1 tree `dagt1_tree_b43c7a9a`.
