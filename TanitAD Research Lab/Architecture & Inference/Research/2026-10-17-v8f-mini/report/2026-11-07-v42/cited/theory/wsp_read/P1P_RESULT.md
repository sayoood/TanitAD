# P1P_RESULT: P1′ (THEORY_CRAWL_AMEND_1 §4) on the T1-kin eval-141 tick-0 windows

**VERDICT, as the criterion is written: NOT EVALUABLE.** The CONFIDENT group holds 24 windows (floor 30) at both seeds; the UNSURE group holds 75. The criterion names this case and says it is never padded, so P1′ is neither held nor failed, and T1′ is neither supported nor contradicted by this read.

The probe can test T1′ in principle: its AUC on the eval windows is 0.782 [0.703, 0.853] in the P1′ band where its label exists (v0 in [3, 6), n 333, 63 episodes) and 0.750 [0.673, 0.825] on the whole P-MOVE range; neither interval covers 0.5.

Evidence class MEASURED (ours, dev-box CPU, zero planner training, no weight edit). Tier: a tick-0 plan-level read of the T1-kin dumps (the plan is made from the logged history); not a driving result. Estimator: episode-cluster bootstrap over the eval-141 episodes, B 2,000, seed 0 (the tp_lib construction); it answers "another draw of EPISODES" and is blind to training and inference variance (H-ESTIM-SEED-1). The two planner seeds are read separately.

## 1. Spec and order of work
* `SPEC_P1P.md` sha256 `73640c6730d66c84afb071a446e74688840bc2a9d9cd4253e8ae3ac5b4e40a97` (frozen file `SPEC_P1P.sha256`: 2026-11-06T23:16:43Z, i.e. 00:16:44 +0100 on 2026-11-07), written before any eval-141 posterior or ρ existed. The probe refit (train rows only) ran after it; the eval read ran after that.
* `SPEC_P1P_AMEND_1.md` reports one defect in the wording of controls N2 and N3 (verdict-string wording that the group-size gate makes unmeetable); no criterion changes. `THEORY_CRAWL_AMEND_1.md` sha256 `9d39ac53fbb55773...` unchanged.
* Probe: P-MOVE (the only population that overlaps v0 in 3–8; P-STOP is defined on v0 < 0.5), label GO (the registered launch mask, the probe's decision label), V+E linear. Five fold models refit by the committed `tp_lib.py` (byte-identical copy `p1p_tp_lib.py`) on the probe's own 12,000 windows and outer folds; posterior = mean of the five `predict_proba`. Chosen C: 0.1 at all five folds and for the all-train model.
* Run history, for the record: `p1p_probe_refit.py` was started three times (two stray duplicate processes were killed by PID; the outputs are from the third, single run). `p1p_read.py` stopped twice on code errors (a string-array comparison with `equal_nan`; the missing `taniteval` import path) before any feature, posterior or ρ was computed; the third launch is the one reported, and the code fixes touched no statistic.

## 2. Window set and group counts

| step (primary, v0 in [3, 8), `plan`) | windows |
|---|---|
| tick-0 rows of the dump | 2399 |
| v0 in [3, 8) | 586 |
| and gt_ok | 586 |
| and a_human > 0.5 m/s² | 168 |
| and all of a_human, â_plan, ρ, posterior finite | 168 (45 episodes) |

Posterior bins over the 168 windows: below 0.35: 28; UNSURE [0.35, 0.65]: 75; (0.65, 0.8): 41; CONFIDENT ≥ 0.8: 24. The windows, a_human and the posterior do not depend on the planner seed, so both seeds hold the same groups. CONFIDENT spans 16 episodes, UNSURE 36.

Domain note: the probe was trained on v0 below 6. 50 of the 168 filtered windows have v0 in [6, 8) and were scored by extrapolation; they place 2 windows in CONFIDENT and 24 in UNSURE (S1 below restricts to the in-domain band).

## 3. The statistic, shown as DESCRIPTIVE because the groups are below the floor (it is not the verdict)

| seed | n CONF / UNSURE | median ρ CONFIDENT [95 %] | median ρ UNSURE [95 %] | Δ [95 %] | Δ > 0 and separated |
|---|---|---|---|---|---|
| s0 | 24 / 75 | 0.856 [0.813, 1.005] | 0.491 [0.377, 0.640] | 0.365 [0.213, 0.534] | yes |
| s1 | 24 / 75 | 0.960 [0.837, 1.065] | 0.523 [0.381, 0.675] | 0.437 [0.240, 0.585] | yes |

Median a_human (m/s²) is 1.095 in CONFIDENT and 0.872 in UNSURE; median â_plan is 0.877 / 0.525 at s0 and 0.999 / 0.539 at s1. The direction is the one P1′ predicts and the interval excludes 0, but it rests on 24 windows from 16 episodes in one group. The rule treats that as too few to read, and this report does the same: no verdict is drawn from this table.

## 4. Controls

| control | what it checks | result |
|---|---|---|
| C4 unit tests | weighted median = `np.median` of the replicated array; bootstrap counts = tp_lib construction; group edges at 0.8 / 0.35 / 0.65 | pass |
| R1 refit reproduces the banked artifacts (train only) | five fold PCAs vs `pca_V_pmove.npz` S; held-out scores vs `oof_pmove_V_linear.npz` (`V+E`, `GO24__V+E`) | pass: PCA relative difference ≤ 1.0e-06; held-out score difference ≤ 6.0e-07 (tolerance 1e-6; the same fit on the banked PCA scores differs by ≤ 3e-13, so the gap is PCA float noise); per-fold AUC equals the banked value to 4 places |
| R2 all-train model vs the banked eval-141 confirmation | AUC of linear V+E on the 507 windows with 0.5 ≤ v0 < 6 | pass: 0.7475 vs banked 0.7475; C 0.1 vs 0.1 |
| R3 eval feature construction | E, V, GO label, v0 on the 608 windows `eval141_features.npz` covers | pass: E max abs 0.0, V bit-equal True, GO equal True, v0 vs kin3 max abs 0.0 |
| K1 probe AUC on eval, P-MOVE range | posterior vs GO, 0.5 ≤ v0 < 6 | 0.750 [0.673, 0.825], n 507 (266 GO, 67 episodes); above chance |
| K2 probe AUC on eval, P1′ band | posterior vs GO, 3 ≤ v0 < 6 | 0.782 [0.703, 0.853], n 333; above chance |
| K3 two-second go indicator | posterior vs a_human > 0.5, 3 ≤ v0 < 8 (descriptive) | 0.852 [0.808, 0.892], n 586 (168 positive) |
| (K1 on GO24) | GO24 posterior vs GO24, 0.5 ≤ v0 < 6 (descriptive) | 0.876 [0.827, 0.918] |
| N1 posterior shuffle | 20 permutations of the posterior, real ρ | pass: separated 1 of 20 at s0, 0 of 20 at s1 (limit 3) |
| N2 identity (ρ ≡ 1) | exact values | pass on the statistic (medians 1.0, Δ 0.0, interval [0, 0], not separated); as run, the original verdict-string flag reads false (defect, AMEND_1) |
| N3 planted (ρ = posterior) | the pipeline sees a planted mechanism | pass on the statistic (Δ 0.364 [0.325, 0.395], separated); as run, the verdict-string flag reads false (defect, AMEND_1) |

## 5. Sensitivities (declared in SPEC_P1P.md §4; none is the verdict, each through the identical verdict logic)

| variant | n CONF / UNSURE | Δ s0 [95 %] | Δ s1 [95 %] | verdict logic returns |
|---|---|---|---|---|
| S1: v0 [3,6) | 22 / 51 | 0.413 [0.205, 0.570] | 0.446 [0.251, 0.644] | NOT EVALUABLE |
| S2: plan_raw | 24 / 75 | 0.335 [0.212, 0.530] | 0.429 [0.249, 0.569] | NOT EVALUABLE |
| S3: all-train probe posterior | 24 / 76 | 0.393 [0.213, 0.590] | 0.431 [0.257, 0.600] | NOT EVALUABLE |
| S4: GO24 posterior | 49 / 41 | 0.271 [0.144, 0.422] | 0.373 [0.201, 0.550] | HOLDS |

S1, S2 and S3 are also below the floor. S4 (the probe's secondary label GO24, a different probe from the one AMEND_1 names) reaches 49 / 41 windows and the logic returns HOLDS; that is a reading of a different go/stay probe, declared in advance as non-decisive, and it is not promoted here.

## 6. What blocks, and the cheapest next steps (the Master Mind's to pre-register; none was run)
* **The blocker is group size, not direction.** The primary band holds 168 windows with a_human > 0.5 from 45 episodes; the CONFIDENT bin takes 24 of them, 6 short of the floor. The 41 windows in (0.65, 0.8) are the neighbouring bin and are not CONFIDENT under the criterion; no threshold is moved here.
* **Next step 1 (zero compute):** a dated pre-registration, before it is read, that names the evaluable variant it wants: either the GO24 probe as the banked go/stay probe (S4 already shows 49 / 41, Δ 0.271 / 0.373 with lower bounds 0.144 / 0.201), or a lower CONFIDENT threshold with its own committed floor. Either changes AMEND_1 and needs its own sha before any reading is quoted.
* **Next step 2 (CPU, no training):** more windows. The tick-0 plan of C0C needs only a forward pass; a second held-out window set with cached `pool4x10`, logged history and logged future would add windows to both groups. Blocker: no such second set is in the artifacts store that this read has touched (UNVERIFIED that none exists elsewhere); the training windows are not admissible because the planner was trained on them.
* Ticks 1 and 2 of the dump cannot supply windows: their history is the arm's own and no probe feature exists for them.

## 7. Provenance and manifest (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/`, nothing staged, committed or pushed; each exists in this one place only)

| artifact | role | sha256 (first 16) |
|---|---|---|
| `SPEC_P1P.md` | pre-registered spec | `73640c6730d66c84` |
| `SPEC_P1P.sha256` | spec hash and time | `d55feb81cd7d3c8d` |
| `SPEC_P1P_AMEND_1.md` | defect note on N2/N3 wording | `9a136911468956d2` |
| `p1p_tp_lib.py` | byte copy of trigger_probe/tp_lib.py | `44b8797740af5cb0` |
| `p1p_probe_refit.py` | train-only refit and R1 | `897c8363022d9ead` |
| `p1p_read.py` | eval read, controls, sensitivities | `b85a6feaf1a002f3` |
| `p1p_finalize.py` | amended N2/N3 flags and this report | `a22b8998d8c7c511` |
| `P1P_PROBE_MODELS.npz` | refit PCA bases, scalers, coefficients (6 models, GO and GO24) | `1ea2c84cd85d968b` |
| `P1P_REFIT_R1.json` | R1 numbers | `ada550ec795e5d9b` |
| `P1P_PERWINDOW.npz` | per-window posteriors, a_human, a_plan, ρ | `96921647caf7e3f2` |
| `P1P_RESULT.json` | machine-readable result | `eec7dc9b1f6d0b0e` |
| `p1p_probe_refit.log` | log | `18da38bcf6e17205` |
| `p1p_read.log` | log | `b933fa1d01133021` |

Inputs read, unmodified: the two T1-kin dumps `C:/Users/Admin/rr_scratch/out/C0C_s{0,1}/ticks/ticks_C0C_s{0,1}.npz`; `trigger_probe/raw/{windows_train,sample_pmove,oof_pmove_V_linear,pca_V_pmove,eval141_features}.npz`, `results_eval141.json`; `F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch/{V_pmove,V_eval141}.npy`; the eval cache `sfresh_inputs/cache/eval/clips`; `sfresh_inputs/inputs/native_tracks.pt`. `WSP_READ.md` / `WSP_READ.json` are not edited; their P1′ row is the Master Mind's to update from this file.
