# RESULT -- H-V8F-FLOOR-1: does refcv7-50,400 beat the learned kinematic floor P10?

**2026-10-17, stream S1, 0 GPU (CPU only; Thor read-only, one 25 MB + one 0.75 MB manifest pull).** Pre-registration:
`Project Steering/PREREG_HJ_STAGE0.md` (bars and arms unchanged). Staged by the orchestrating session, never committed by S1.

## 0. Verdict

**POS -- the committed bar is met at both inference seeds.** `os - P10`, ADE 0-6 s (S6 grid), paired episode-cluster bootstrap B = 2000:

| inference seed | `os - P10` ADE 0-6 s [95 % CI] | FDE | n windows / episodes | bar (< 0, CI excludes 0) |
|---|---|---|---|---|
| s0 | **-0.4061 [-0.5936, -0.2161] SEP** | -1.1730 [-1.6715, -0.6729] SEP | 3,668 / 139 | MET |
| s1 | **-0.4054 [-0.5911, -0.2231] SEP** | -1.1669 [-1.6516, -0.6876] SEP | 3,668 / 139 | MET |

Source: `raw/floors_S6_s{0,1}.json` -> `paired_x_minus_y["os-P10"]`; verdict computed mechanically in `code/finalize.py` -> `raw/summary.json`.
**Tier: T1 (self-action OPEN loop; the model does not control the vehicle) -- never closed loop, never "driving performance".**
Evidence class: MEASURED (ours, paths above). Mean ADE 0-6 s: `os` 2.5675 m, `P10` 2.9736 m, `damp50` 3.5338 m, `kd_x` 3.6201 m, `ha0_ext` 3.5479 m, `ha0` 4.9926 m.

### But read the margin correctly -- three things the bar does not say

1. **It is a 6 s, LATERAL margin.** At 0-2 s (S2) `P10` beats `os`: ADE `os - P10` = **+0.0276 [+0.0163, +0.0398] SEP** (s0), +0.0282 [+0.0167, +0.0401] SEP (s1); `os` is separated-worse on LON speed (+0.035), LON along (+0.027), LON accel (+0.067) and trajectory-derived TAC-lon (-0.027 correct-rate). At 6 s, `os`'s gain over P10 sits in the lateral family (cross-track -0.598 m, heading -2.06 deg, yaw-rate -0.028 rad/s, TAC-lat +0.090) and **not** in speed (-0.023 [-0.078, +0.037]) or along-track (+0.015 [-0.112, +0.153]). Table 3.
2. **Variance answered.** The CI is the EPISODE variance (models held fixed). Inference seed: the bar replicates at s0 and s1 (banked inference-seed floor 0.0006 m ADE, `E:/refcv7_eval_kit/battery/step50400_a6text/inference_seed_replicate.json`). **P10's own training seed: measured here** -- a seed-1 refit moves mean ADE 6 s by 0.0213 m (2.9736 -> 2.9523; `raw/p10_seed_replicate.json`); `os - P10_seed1` = -0.3849 [-0.5732, -0.1936] (inf. s0) / -0.3841 [-0.5693, -0.2016] (s1), i.e. the margin is **18x the floor's training-seed floor**. **`os`'s own training-seed variance is NOT measured** (one refcv7 training run) -- the one number this read cannot give.
3. **A unit defect inflated every earlier margin against `ha0_ext` / `damp50` (Section 6).**

## 1. Setup (what was run, so it can be re-run)

* **Arms.** `os` = refcv7-50,400 from the banked T1 dumps `E:/refcv7_eval_kit/battery/step50400_a6text/{dump_s0,dump_s1}` (S2: 4 steps x 0.5 s) and `{s6_s0,s6_s1}` (S6: 6 steps x 1.0 s); inference seeds 0 and 1. Grid: eval139 = 139 episodes, S2 = 4,754 windows, S6 = 3,668 windows (the dump's own `ws`; `dump_s0` is bit-identical to `panel_s0` on os/ha0_ext/g, checked on every 10th episode).
* **Floors** (`taniteval/taniteval/floors_kin.py`, 18 known-value tests green; `taniteval/tests/test_floors_kin.py`):
  * `P10` = the MLP of `probe_te.py::fit_mlp_kin` (h 256, GELU, zero-init output, AdamW 1e-3, batch 256, 60 epochs, wd {1e-4, 1e-2}, 4-fold grouped inner CV over wd x epoch), 8 features `(v, a, k, v^2, v*a, v*k, v^2*k, a*k)` of `(v0, a0, k0) = (v[t0], (v[t0]-v[t0-1])/0.1, actions[t0,0])` -- features unchanged; residual on `kd_x`. **`fit_p10` reproduces `fit_mlp_kin` bit-for-bit** (max |pred diff| 0.0 at both horizons, `raw/control_refav1.json`).
  * `damp50 = 0.5 ha0 + 0.5 ha0_ext`. `kd_x` = the damp50 path re-timed by `ha0_ext`'s travelled distance. **WHICH kd_x:** re-timed on the native 0.1 s polyline (60 ticks for S6, 20 for S2) and index-selected -- the refcv analogue of the 30 x 0.2 s `kd_x_6`. Sensitivity (S6): `kd_x30_6s` (the 30 x 0.2 s definition) and `kd_x_coarse` (re-timed on the 6 dump instants) move the mean ADE by -0.018 / +0.005 m and leave `os - kd_x` at -1.03 to -1.06 m (Table 1).
* **P10 fit (primary).** TRAIN windows of the refcv8 train view (`/home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt`, md5 `77dafb50...`, 4,572 clips, read-only scp; eval manifest md5 `34433232...`): **kinematics + futures only, no frame, no label**. Window rule `t0 = 7 + (e mod 5) + 5j`, every `t0` with `t0 + 20` (S2) / `t0 + 60` (S6) inside the clip: S2 **157,239 windows**, S6 **120,663 windows**, 4,572 episodes each; train/eval clip overlap asserted 0. Selection: S2 wd 1e-4, 52 epochs; S6 wd 1e-4, 59 epochs. **The FALLBACK was not needed for the verdict** and is reported as a sensitivity (Section 5).
* **Feature/GT exactness.** From the eval manifest alone, `v0`, `g`, `ha0`, `ha0_ext` reproduce the banked dump arrays (`raw/manifest_vs_dump_exactness.txt`: torch path max diff 0; float64 rebuild < 2.2e-4 m, `raw/floors_*.json` -> `controls`).

## 2. Known-value controls (all PASS)

| control | required | read | artifact |
|---|---|---|---|
| `P10 - P10` ADE (S2/S6, both seeds) | +0.0000 [0, 0] | +0.0000 [+0.0000, +0.0000] | `floors_*` `P10-P10` |
| P10 refit on the **refav1 grid** (600 clips, 5,398 windows = the P10 package's exact count) | 6 s ADE 2.9257 +- 0.02 m | **2.92572 m** (|err| 1.6e-5); 2 s 0.22287 vs banked 0.22287 | `raw/control_refav1.json` |
| TRAIN rebuild validated against the package's banked EVAL arrays (139 common episodes, 2,365 windows) | ws identical, arrays equal | ws identical; max abs diff: kin 1.2e-7, g6 2.4e-5, ha0_ext_6 9.6e-5 | same |
| my `kd_x` vs the original `P.kdx` on the banked arrays | equal | max abs 2.3e-4 (6 s) / 8.6e-6 (2 s), float32 | same |
| `os - ha0_ext` through MY direct bootstrap vs the banked battery numbers | equal | S2 s0 -0.0102 [-0.0260, +0.0067] (RESULT_SECTION identical); S6 s0 -0.9804 [-1.2940, -0.6815] (= BAR-R7-3, identical); the battery's analysis-route intervals (string episode ids) differ only in Monte-Carlo noise: S6 -0.9804 [-1.2901, -0.6778] | `summary.json` -> `controls` |
| derived-dump analysis (`analyze_refcv3` on my dump) reproduces the banked `analysis_*.json` for `os - ha0_ext` | exact | exact (same lo/hi, all four combos) | same |
| mutation (deliberate-regression) of the floors | tests go RED | damp50 weights, a<->k swap, retime scale, feature drop, forward difference at t0: all RED | `raw/mutation_check.txt` |

## 3. Headline paired deltas (ADE / FDE, B = 2000, paired episode-cluster bootstrap, x - y)

Full list incl. FDE and both seeds: `raw/tables.md`, `raw/summary.json`. S6, seed 0 (floors are identical across seeds; `os` differs):

| pair | ADE 0-6 s [CI] | FDE |
|---|---|---|
| `os - P10` | **-0.4061 [-0.5936, -0.2161] SEP** | -1.1730 [-1.6715, -0.6729] SEP |
| `os - damp50` | -0.9664 [-1.3010, -0.6545] SEP | -2.1878 [-3.0267, -1.4181] SEP |
| `os - kd_x` | -1.0526 [-1.3803, -0.7445] SEP | -2.8101 [-3.6546, -2.0039] SEP |
| `os - ha0_ext` | -0.9804 [-1.2940, -0.6815] SEP | -2.8743 [-3.7144, -2.0515] SEP |
| `P10 - ha0_ext` | -0.5743 [-0.7744, -0.3753] SEP | -1.7012 [-2.2490, -1.1399] SEP |
| `P10 - kd_x` | -0.6465 [-0.8674, -0.4463] SEP | -1.6371 [-2.2037, -1.1122] SEP |
| `damp50 - ha0_ext` | -0.0140 [-0.2162, +0.2197] (n.s.) | -0.6865 [-1.1829, -0.1129] SEP |

S2 (0-2 s), seed 0: `os - P10` **+0.0276 [+0.0163, +0.0398] SEP** (FDE +0.0258 [-0.0039, +0.0569]); `os - damp50` -0.1341 [-0.1661, -0.1042] SEP; `os - kd_x` -0.0625 [-0.0930, -0.0351] SEP; `os - ha0_ext` -0.0102 [-0.0260, +0.0067]; `P10 - ha0_ext` -0.0378 [-0.0478, -0.0278] SEP; `damp50 - ha0_ext` **+0.1239 [+0.0985, +0.1506]** (damping is *worse* at 2 s). Seed 1 differs in the 4th decimal (`raw/tables.md`).

### Four families, `os - P10` (battery's own tool `analyze_refcv3`, injected pair; negative = os better for errors, positive = os better for the TAC correct-rates)

| family | metric | S6 s0 | S2 s0 |
|---|---|---|---|
| ADE | ade_m / fde_m | -0.4061 SEP / -1.1730 SEP | +0.0276 SEP / +0.0258 |
| LONGITUDINAL | speed MAE m/s | -0.0231 [-0.0776, +0.0374] | **+0.0351 [+0.0243, +0.0471] SEP** |
| | along MAE m | +0.0148 [-0.1123, +0.1529] | **+0.0265 [+0.0179, +0.0365] SEP** |
| | accel MAE m/s^2 | -0.0405 [-0.0648, -0.0155] SEP | **+0.0674 [+0.0527, +0.0839] SEP** |
| | min headway m / time gap s / min TTC s (n 919 / 872 / 919; S2 n 1257 / 1122 / 1257) | +0.043 / -0.006 / -0.078, all n.s. | -0.080 / -0.013 / -0.085, all n.s. |
| LATERAL | cross-track MAE m | **-0.5983 [-0.7625, -0.4537] SEP** | +0.0043 n.s. |
| | heading MAE deg (n 3579; 89 dropped) | **-2.0597 [-2.6169, -1.5637] SEP** | -0.0514 n.s. |
| | yaw-rate MAE rad/s (masked valid cell) | **-0.0279 [-0.0353, -0.0213] SEP** | -0.0299 [-0.0445, -0.0184] SEP |
| TACTICAL (trajectory-derived) | lat / lon correct-rate | **+0.0902 SEP / +0.0750 SEP** | +0.0217 SEP / **-0.0267 SEP** |
| TACTICAL goal-setting / declared decisions | | UNAVAILABLE for the floors (a kinematic floor has no decision head; `os`'s declared-head quality is in the banked RESULT_SECTION) | same |
| STRATEGIC | | **PAUSED (PI R6 ruling 2026-09-19, PI_DECISION_QUEUE item 25)** -- neither a pass nor omitted | same |

Source: `raw/analysis_S{2,6}_s{0,1}.json` -> `refcv3.families_paired.paired_os_minus_P10`, `refcv3.distance_keeping.paired`. Distance-keeping is scored on the shared instants {1, 2} s only (the banked lead block's common grid) -- a coarse read.

## 4. The "acts" check (does the arm accelerate at all?)

Acceleration = second difference of path-implied speed (`floors_kin.acts_check`; zero tolerance 1e-3 m/s^2), over every window and step; `raw/floors_*.json` -> `acts`.

| arm | S6 mean abs a | S6 zero-accel frac | S2 mean abs a | S2 zero-accel frac |
|---|---|---|---|---|
| `os` (s0 / s1) | 0.3704 / 0.3699 | 0.0233 / 0.0251 | 0.4577 / 0.4571 | 0.0262 / 0.0258 |
| `P10` | 0.3607 | 0.0017 | 0.3959 | 0.0027 |
| `damp50` | 0.2356 | 0.0595 | 0.2383 | 0.0445 |
| `kd_x` | 0.4497 | 0.0559 | 0.4816 | 0.0412 |
| `ha0_ext` | 0.4416 | 0.0559 | 0.4814 | 0.0412 |
| `ha0` (constant velocity) | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| GT | 0.4825 | 0.0238 | 0.4795 | 0.0318 |

`os` is not a constant-velocity plan (zero-accel fraction 2-3 %, = GT's 2.4-3.2 %); the margin over `ha0` is not a straight line beating a noisy control. `P10` never outputs an exact zero acceleration (0.2-0.3 %). `damp50` under-accelerates by construction (it averages with a zero-acceleration path).

## 5. Sensitivities and EXPLORATORY items (never lever claims)

* **FALLBACK P10** (grouped 5-fold CV inside eval139, folds by episode rank mod 5, hyper-parameters on the fit folds only) -- reported beside, **not the verdict**: S6 mean ADE 3.2479 (vs 2.9736 TRAIN-fit; `P10cv - P10` +0.2742 [+0.1684, +0.4005], the fallback is a weaker floor from ~110 episodes instead of 4,572); `os - P10cv` S6 **-0.6804 [-0.9219, -0.4456]** (s0), -0.6796 [-0.9193, -0.4523] (s1); S2 `os - P10cv` +0.0202 [+0.0082, +0.0328] / +0.0208 [+0.0089, +0.0332]. Same signs as the primary at both horizons. `raw/fallback_cv_S{2,6}.json`.
* **Late fusion `alpha*os + (1-alpha)*P10`** -- **EXPLORATORY**, alpha by grouped 5-fold CV (fit on the 4 held-in folds, applied to the held-out one). S6: alpha 0.65-0.70 in every fold; fused - P10 **-0.5318 [-0.6865, -0.3881]**, fused - os **-0.1257 [-0.1847, -0.0699]** (s0; s1 -0.5367 / -0.1313). S2: alpha 0.40 in every fold; fused - P10 -0.0192 [-0.0255, -0.0132], fused - os -0.0468 [-0.0563, -0.0383]. `os` and `P10` carry complementary information at both horizons; this is a hypothesis generator, not a result. `floors_*.json` -> `fusion`.
* **Training-seed replicate of P10** -- Section 0 item 2; `raw/p10_seed_replicate.json`.

## 6. SIDE FINDING (escalate to the Master Mind and the retraction log): the refav1 `ha0_ext` echo control integrates a STEER ANGLE as a curvature

`actions[:, 0]` is a road-wheel STEER ANGLE (`arctan(L*kappa)`, `refav1_loader.py:27`). The P10 package's `extract.py` builds `ha0_ext` / `ha0_ext_6` with `paths_from_controls` at its DEFAULT `action_units="kappa"`, i.e. the steer angle is integrated as if it were 1/m (~2.9x over-rotation). **MEASURED** (`code/refav1_units_check.py` -> `raw/refav1_units_check.json`, 141 episodes / 2,399 windows, the P10 package's own `feat_s4`): rebuilding with `kappa` reproduces the banked `ha0_ext_6` to 9.6e-5 m; rebuilding with the correct `steer` unit differs from it by up to **87 m**.

| 6 s ADE, refav1 grid | banked unit (`kappa`) | correct unit (`steer`) |
|---|---|---|
| `ha0_ext` | 5.356 | **2.838** |
| `damp50` | 3.454 | 2.934 |
| `kd_x` | 3.558 | 2.940 |

* **`damp50 - ha0_ext` @ 6 s: -1.9025 [-2.3605, -1.4971] (banked) -> +0.0956 [-0.0761, +0.2901] (correct unit).** The plan's lever 7 ("the prior swap is the largest measured lever we have not used: `damp50 - ha0_ext` = -1.90 m @ 6 s", `PLAN_V8F...md` Sec. 0 item 7) is a **unit artifact of a defective echo control**, not a lever. refcv7/8 integrate `ha0_ext_pose` in the correct unit already; on the refcv grid `damp50 - ha0_ext` = -0.0140 [-0.2162, +0.2197] (S6, n.s.) and **+0.1239 [+0.0985, +0.1506] SEP worse** at S2 (this read, Table 3).
* The P10 package's headline "2.93 m vs 5.36 m echo" is likewise a margin against the defective control; against the correctly-integrated `ha0_ext` the refav1-grid number is 2.838 m (< P10's 2.926 -- the refav1-grid P10 was fit as a residual on the defective `kd_x`). **This S1 verdict is not affected:** every refcv floor here is in the correct unit and `P10` is refit on correct-unit `kd_x` (S6: `P10 - ha0_ext` = -0.574 [-0.774, -0.375]).
* Root-cause class for `RETRACTION_LOG.md`: *a control inherits the unit of the nearest default* (same class as `C-REFCV3-ARM-SAME-DEFECT`, here in the P10 package and the plan built on it). I did not edit any tracked file.

## 7. Next lever (RULE ZERO -- the turn does not end at a verdict)

* **PREREG POS branch:** the 6 s margin is registered as the programme's first measured "vision helps over a learned kinematic floor" quantity, quoted only with its estimator and tier: `os - P10` ADE 0-6 s -0.406 m [-0.594, -0.216] (s0) / -0.405 m [-0.591, -0.223] (s1), episode-cluster paired bootstrap B = 2000, T1 open loop, eval139 S6 (3,668 windows / 139 episodes), single refcv7 training run, P10 training-seed floor 0.021 m. **At 0-2 s the sign reverses** (P10 better, +0.028 m separated), so the sentence "refcv7 beats the learned kinematic floor" is false without the horizon.
* **Cheapest next experiment that could make the 2 s side work (I did not run it; it needs a trained arm):** swap the REF-C decoder's residual prior from `ha0_ext_pose` to **P10** (trained here on the refcv8 train view, correct units) -- evidence: P10 beats `ha0_ext` separated at both horizons (S2 -0.038, S6 -0.574), `os` loses to P10 at 2 s on speed/along/accel, and the exploratory fusion is better than either (alpha 0.4 at 2 s, 0.7 at 6 s). This is plan rung alpha's `alpha1` arm with **P10 as the prior, not `damp50`** (damp50 is dominated: S2 +0.124 worse than `ha0_ext`).
* **Plan correction needed before rung alpha is costed:** lever 7 (`damp50 - ha0_ext` = -1.90 m) must be re-stated (Section 6).
* **What would make this read stronger and is blocked/unrun:** `os` training-seed variance (one refcv7 run), a refcv8 arm read against the same floors (refcv8 EMA-15k dumps were not read: `C:/Users/Admin/qland/hold_r8batt/` was left untouched per instructions), and event strata (turn onset / stop / lead braking) -- the lateral/6 s localisation suggests turn windows are where the margin lives; not run (not in the PREREG for this hypothesis).

## 8. UNVERIFIED / not measured

* `os`'s training-seed variance (single run). P10's was measured (one replicate seed only).
* TRAIN windows are S1's own construction (all stride-5 windows of the 4,572 clips), not the refcv8 trainer's sampled window list; the trainer's own per-epoch sampling was not read. P10 uses no image and no label, so only the kinematic distribution matters; the S1 P10 was never exposed to any eval139 clip (overlap asserted 0).
* The Thor manifest is the *cache* manifest (poses/actions of the v2 cache), not the raw egomotion; it reproduces the banked eval arrays to float precision, which is the check that matters.
* refcv8 dumps (`qland/hold_r8batt`) not read -> no refcv8-vs-floors statement.
* The battery's analysis route and my direct bootstrap use different episode-id strings and so different (equally valid) seed-0 draws: bounds differ by <= 0.004 m at 6 s; no bar decision is sensitive to it.
* Distance-keeping rows use only the {1, 2} s shared instants.

## 9. Deliverable manifest

| artifact | location |
|---|---|
| module `floors_kin.py` (kin_features, fit_p10 incl. `seed`, damp50, kd_x/retime, acts_check, fusion, grouped folds) | repo:`taniteval/taniteval/floors_kin.py` (NEW, unstaged) |
| tests (18, known-value + mutation-style) | repo:`taniteval/tests/test_floors_kin.py` (NEW, unstaged) |
| this report + `raw/` (floors/analysis/arms/fallback/replicate/controls/summary JSON, tables.md, logs) | repo:`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/` (NEW, unstaged) |
| run code: `s1_lib.py`, `run_control_refav1.py`, `fit_p10_refcv.py`, `run_refcv7_floors.py`, `run_fallback_cv.py`, `p10_seed_replicate.py`, `refav1_units_check.py`, `finalize.py`, `chain.sh`, `mutate_check.py`, `probe_manifest_vs_dump.py` | repo:`.../2026-10-17-p10-floors-refcv/code/` (NEW, unstaged) |
| fitted P10 models (S2/S6, seeds 0/1) | **dev box only:** `C:/Users/Admin/s1_floors_scratch/p10_{S2,S6}{,_seed1}.pt` (regenerable in ~6-14 min by `fit_p10_refcv.py`) |
| Thor manifests (train 4572 / eval 139), derived 4-family dump dirs | **dev box only:** `C:/Users/Admin/s1_floors_scratch/` (`train4572_v2manifest.pt` 24.9 MB, `eval139_v2manifest.pt`, `dumps/`) -- regenerable from Thor / `arms_*.npz` |
| eval-window arms (os, ha, ha0, ha0_ext, damp50, kd_x, P10, sensitivities) | repo:`.../raw/arms_{S2,S6}_s{0,1}.npz` |

Nothing was `git add`ed (session amendment: the orchestrator lands through qland).
