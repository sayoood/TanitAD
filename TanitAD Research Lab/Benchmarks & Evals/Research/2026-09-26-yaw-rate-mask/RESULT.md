# D-YAWMASK-1 — the shared paired yaw-rate cell is now scored only on step pairs that have a tangent

**Date:** 2026-09-26, Europe/Berlin. **Written by:** a Benchmarks & Evals subagent acting for the Master Mind.
**Branch:** `agent/arch-inf-20260803`. The work started at tip `1218921`. At hand-off the tip was `048ae3b`; `taniteval/tools/refav1_arm.py`, every consumer below, the battery's `test_yaw_valid.py` and `PREREG_REFCV6.md` are byte-identical between the two (checked by blob).
**Register id:** this closes the OPEN row `D-YAW-UNMASKED` (§6.1). `D-YAWMASK-1` is only this package's working name.
**Compute:** 0 GPU. Everything ran on the dev-box CPU (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS` ≤ 4). Free host RAM read 7.85–11.3 GB at every check.
**Tier:** every re-derived number is **T1 (self-action OPEN loop)** unless it says otherwise. The withheld-bank panel is a rig-scope T1-style read.
**Estimator:** every interval is the paired episode-cluster bootstrap (`taniteval/taniteval/ci.py`, n_boot 2000, seed 0, cluster = clip/episode). It answers *"would another draw of episodes say this?"* It says nothing about training variance or inference variance.
**Evidence class:** every number below is **MEASURED (ours, 2026-09-26)**, with its artifact under `raw/`, unless it is marked INHERITED.

---

## 0. Headline

**Which claims move, and by how much.** Each row gives the landed number, then the number with the fixed cell. Both come from the claim's own instrument, re-run on the claim's own dumps. The reproduction control was exact on every record.

| claim | landed | with the fixed cell | verdict |
|---|---|---|---|
| **M26 / `D-FEASDEC-T1-1`**: feasibility projection, yaw-rate | 0.2176 → 0.0427 rad/s, Δ **−0.1749** [−0.2563, −0.1088] separated, "**−80.4 %**", "from worse than a constant-velocity line to better than it" | 0.0320 → 0.0304 rad/s, Δ **−0.0014** [−0.0029, −0.0004] separated, **≈ −4.3 %**. `base` **already beats** the `ha0` line: **−0.0100** [−0.0168, −0.0037]. n = 4,524 of 4,823 windows / 141 episodes | ⛔ **AFFECTED: the lateral headline is RETRACTED.** The effect is ~1/125 of what was quoted. ADE, LON, heading, cross, TAC and STR do not move (0 of 108 cells). |
| **refcv3 @40,284 LATERAL**: `D-REFCV3-40284c`, MODEL_REGISTRY, the HF card, LEADERBOARD:109 | `os − ha0` yaw **+0.1700** [+0.1006, +0.2534] "**LOST**"; the family verdict is **LOST (2/3)** | **−0.0100** [−0.0168, −0.0037] **WON**; the family verdict is **WON (3/3)**. `os − ha`: +0.1849 → +0.0064 [+0.0023, +0.0107] (still lost, 29× smaller) | ⛔ **AFFECTED: the sign flips.** The "two yaw numbers disagree in sign" row is resolved. Its own HYPOTHESIS ("near-stationary steps") is now MEASURED. |
| **`H-ESTIM-SEED-1` / `D-REPLICATE-FPRATE` / `CLAUDE.md`**: the replicate false-positive rate | **6/42 = 14.3 %**. The VERDICT 2-s view reads **3/14 = 21.4 %**. Three of the six were yaw cells. | **4/42 = 9.5 %**; the VERDICT view reads **2/14 = 14.3 %** | ⛔ **AFFECTED: the number is corrected.** The doctrine stands: a zero-lever replicate still separates on 4 of 42 cells. About 80 sites quote the old number (`raw/propagation_sites.txt`). |
| **`D-RL-VETO-T1-1`** | yaw +0.0191 / +0.0478 against a floor of 0.0287, read as "**WITHIN-NOISE**"; "7 of 11 quotable" | **+0.0251 / +0.0294, floor 0.0043 (5.8×): QUOTABLE**; now "8 of 11" | ⛔ **AFFECTED. The defect HID a real lateral regression.** |
| **`D-REFCV3-40284e`**: stratified, "loses yaw-rate MAE almost everywhere" | `os − ha` separated worse on 6 of 8 strata, up to +0.241 rad/s | separated worse on **4 of 8**, all ≤ **+0.0167**. Turns, manoeuvre and accelerate: not separated | ⚠️ **AFFECTED, weakened.** The ADE-based conclusion of the row stands. |
| All 14 other re-derived records (§4.7) | — | the same separation on every quoted cell. Magnitudes move ≤ ~25 % (for example 19× → 19.7×, 8.3× → 8.4×) | ✅ **UNAFFECTED** |
| `PREREG_REFCV6.md` L1–L4 | — | no code judges L2 today. `PREREG_REFCV6_V2` says this document governs only the design stopped on 2026-09-11. | ⚠️ The amendment is **drafted, not applied** (§4.6) |

**The fix and its test status: ✅ done, and it clears its bar.**
* **The fix.** `_components` now averages yaw-rate per window over `pred.pair_valid & gt.pair_valid`, exactly as heading is averaged. A window with no valid pair gets NaN, which every consumer drops and counts. `_paired_families` now stamps `yaw_rate_cell` on every block. The key name is unchanged; its definition changed.
* **The new test.** `taniteval/tests/test_refav1_components_yaw_mask.py` has 11 tests with literal targets. All 11 pass on the fix. Each regression arm turns it **RED**:
  * the real historical defect: 7 red;
  * the yaw line reverted alone: 6 red;
  * a GT-only mask: 1 red;
  * a pred-only mask: 3 red;
  * a second-step-only mask: 1 red.
* **The related suites** ran in a clean tree. That tree is a `git archive` of the tip plus the two files, and every import was asserted to resolve under it.
  * With the fix: **276 passed / 0 failed / 2 skipped.**
  * On the tip: 265 / 0 / 2.
  * The same two tests skip in both runs, because their input files are outside the scratch tree.
* **Independent cross-check on real data.** This used the battery's step-5000 panel, 4,754 windows. The fixed shared cell equals the battery's own A3 cell, written by a different author, **on every field of all 13 pairs**. The 117 non-yaw cells do not move.
* **Across 21 landed records:** 2,838 of 2,838 cells reproduce exactly with the tip cell, and **no non-yaw cell moves** with the fixed one.

**⚠️ Integration: read §5 before landing.**
* Landing turns the battery's landed `test_yaw_valid.py` **2 of 3 RED, by design.** That test pins the defect in the shared cell. A replacement (4/4 green on the fix, 2/2 red on the tip) is in `code/proposals/`. It is not applied: that package belongs to the EvalFlyWheel.
* **The public HF card for refcv3 carries the flipped verdict.** Republishing is a PI decision.

**Stopping condition (Rule Zero clause 3):**
* The fix clears its bar.
* Every claim whose dumps exist on this box has been re-derived.
* **Blocked, with the blocker named:**
  * the Thor half of `D-REFAV1-LON-T1`: its dumps are on Thor, and this brief forbids Thor access. The dev-box twin moved −0.1009 → −0.1161, with the same sign and still separated.
* **Not re-derived, next lever named:** `H-KINGATE-1` uses a *separate* instrument (`kin_gate_eval.py`), which has its own unmasked heading, yaw and curvature. Same class, different code (§4.8).

---

## 1. The defect and the fix

**Before** (tip blob `c7013107`, `taniteval/tools/refav1_arm.py:2058`):
```python
"LAT_yaw_rate_mae_radps": (Pg["yaw_rate"] - Gg["yaw_rate"]).abs().mean(1).numpy(),
```
`four_families._seq_geometry` publishes `pair_valid = valid[:, 1:] & valid[:, :-1]`, where `valid = ds > MIN_DS_MPS·dt`. On the 0.5 s grid that threshold is 0.25 m. A step shorter than it has no path tangent, so its `atan2` heading is noise. `four_families.lateral` masks yaw-rate with `P["pair_valid"] & G["pair_valid"]` (`four_families.py:786-791`). `_components` masked heading with `valid` and applied **no mask** to yaw-rate. Its docstring nevertheless said *"four_families' OWN geometry … never a re-derivation"*. The geometry was shared; the reduction was re-derived without the mask.

**After** (package file `code/fix/taniteval/tools/refav1_arm.py`, blob `963d98e6`):
```python
both_pair = Pg["pair_valid"] & Gg["pair_valid"]
npv = both_pair.sum(1)
yaw_err = (Pg["yaw_rate"] - Gg["yaw_rate"]).abs()
yaw = torch.where(npv > 0, (yaw_err * both_pair).sum(1) / npv.clamp_min(1), torch.nan).numpy()
```
* It is a per-window mean over the masked pairs, built exactly as the heading cell next to it is built. It is bit-identical to the battery's A3 `yaw_rate_valid` (§2.4).
* `_paired_families` now writes `"yaw_rate_cell": YAW_RATE_CELL` into every block. **The key did not change and its definition did**, so an artifact must say which one it carries. A block without the stamp predates the fix.
* Nothing else in the file changes. `diff` against the tip blob shows 4 hunks:
  * the `_components` docstring;
  * the masked yaw block;
  * the emitted line;
  * the `YAW_RATE_CELL` constant, placed after `_FAMILY_OF`, plus the stamp in `_paired_families`.

**Why the mask matters, measured on the M26 dump** (`raw/m26/m26_yaw_decomposition.json`; 4,823 windows, dt 0.5 s):
* **347** base windows contain a pair with no tangent. They carried **86.9 %** of the unmasked yaw sum.
* **236** base windows read above 1 rad/s unmasked. For proj07 and for `ha0` the count is 15.
* On the fully-valid windows, the old and new cells agree **exactly**: 0.0307 = 0.0307 for base, 0.0294 = 0.0294 for proj07, 0.0414 = 0.0414 for `ha0`. That is the control showing the fix moves only the undefined steps.

## 2. Tests

### 2.1 The new test: literal, analytic targets
`code/fix/taniteval/tests/test_refav1_components_yaw_mask.py` (blob `10d4d683`; repo path `taniteval/tests/`). The synthetic windows are on the 0.5 s grid, with `min_ds` 0.25 m.

| test | window | literal expectation |
|---|---|---|
| GT stopped | pred 0.1 m jitter, GT zeros | NaN (the defect read π rad/s) |
| pred stopped, GT drives | pred jitter, GT 10 m/s straight | NaN. **A GT-only mask reads π** |
| GT jitters, pred drives | pred straight, GT jitter | NaN. **A pred-only mask reads π** |
| exact plan | straight / straight | 0.0 |
| constant turn | 0.1 rad per step | 0.2 rad/s |
| partial window | GT valid on pair 0 only | 0.2 rad/s. The unmasked mean of the same pairs is 2.16106 |
| GT pulling away | GT's first step 0.1 m, then 5 m steps | 0.0. **A second-step-only mask reads π/3** |
| heading unchanged | partial window | 2.8647890° |
| key set | — | the 10 keys, as a literal list |
| paired cell | 3 episodes × (driving, stopped-GT); the two arms differ only in standstill jitter | Δ = 0.0 [0.0, 0.0], not separated, `n_dropped_nonfinite` = 3. The defect read **π/2, separated** |
| stamp | — | `"pred.pair_valid AND gt.pair_valid" in blk["yaw_rate_cell"]` |

### 2.2 Regression arms: the same file, one swap each, in a clean tree
Source: `raw/regression_arms.json`, produced by `code/run_regression_arms.py`. The tree was the tip `1218921` via `git archive` (2,027 files; `taniteval/results` excluded). `tanitad`, `four_families` and `ci` were all asserted to import from the tree.

| arm | what it reintroduces | result |
|---|---|---|
| **FIXED** | — | ✅ **11 passed / 0 failed** |
| M0 tip verbatim | the real historical blob `c7013107` | ⛔ **7 failed** |
| M0 yaw line only | the fixed file with only the emitted line reverted | ⛔ **6 failed** |
| M1 GT-only mask | `both_pair = Gg["pair_valid"]` | ⛔ **1 failed**, the pred-stopped window |
| M2 pred-only mask | `both_pair = Pg["pair_valid"]` | ⛔ **3 failed** |
| M3 second-step mask | `(valid & valid)[:, 1:]` | ⛔ **1 failed**, the pull-away window |

Each mutation anchor was asserted to occur **exactly once** before it was replaced, and each mutant was asserted to differ from the fix. So no arm could pass by failing to commit its defect.

### 2.3 Related suites, fixed vs. tip, in the same clean tree (with `products/` present for the criteria registry)
Source: `raw/related_suites.json`. The 16 files are: `test_refav1_arm`, `test_refcv3_arm`, `test_paired_openloop`, `test_openloop_suite`, `test_refav1_kin_contract`, `test_refav1_openloop_report`, `test_refcv3_ablations`, `test_refcv3_ha0_ext_shared`, `test_refav1_lead_block`, `test_refav1_window_list`, `test_render_refav1_arms`, `test_bench_suite_internal_t1`, `test_four_families_lateral_undefined`, `test_four_families_dt`, `test_ci`, and the new test.
* **FIXED: 276 passed, 0 failed, 2 skipped.** **TIP: 265 passed, 0 failed, 2 skipped** (the tip run omits the new test's 11).
* No test fails only with the fix.
* The 2 skips are identical in both runs and are scope skips:
  * `PREREG_REFCV4B_HIERARCHY_EVAL.md` is not in the tree;
  * the E9 record is under the excluded `taniteval/results`.
* A first pass without `products/` failed 7 tests on a missing `CRITERIA_REGISTRY.json`. Those 7 failed identically at the tip too; that run was discarded and the pass redone.

### 2.4 An independent implementation, on real data
Source: `raw/battery_a3_crosscheck.json`, produced by `code/probes/battery_a3_crosscheck.py`. It used the battery's unmodified `refcv6_panel.cross_paired` against a clean tip tree carrying the fixed file, on the battery's step-5000 seed-0 panel (4,754 windows / 139 episodes).
* **The shared cell equals the battery's A3 `_valid` cell on every field, for all 13 pairs.**
* The A3 cell reproduces the battery's banked A3 numbers on all 13 pairs.
* **All 117 non-yaw cells are identical** to the banked ones.
* `os − refcv4b`: −0.1683 [−0.2405, −0.1009] separated → **−0.0042 [−0.0075, −0.0011]** separated, n 4,487 (267 dropped). This re-measures the battery's F10.
* The battery's per-arm figures (refcv4b 0.2034 unmasked vs. 0.0318 masked) are INHERITED from `battery/raw/a3/yaw_mask_probe_step5000_s0.json` and were not re-run here.

## 3. Every consumer passes the NaN through, and the NaN is dropped and counted

| consumer | path | handling | evidence |
|---|---|---|---|
| `refav1_arm.analyze_refav1` | `_paired_families` (`refav1_arm.py:2086`) | `np.isfinite` on both arms, `n_dropped_nonfinite` | `test_refav1_arm` green; two refav1 records re-derived (§4.7) |
| `refcv3_arm.analyze_refcv3` | `ra._paired_families` (`refcv3_arm.py:3100`) | same | `test_refcv3_arm` green; refcv3 record re-derived, 70/70 cells reproduced |
| `paired_openloop.py` | `_boot_single` / `_boot_paired` (`:607-628`); difference of margins on the windows finite for A, B **and** the floor | same | M26 plus 7 other records through the real CLI; `n_dropped_nonfinite` filled |
| `stratified_openloop.py` | `cell_stat` (`:316`). `control_decomposition` and C5 read `ade_m` only | same | `strat_40284` through the real CLI, 178/178 cells reproduced |
| `refav1_paired_delta.py` (blob-identical to the research copies) | `keep = isfinite & isfinite` | same | 8 records through the real CLI; known-value control PASS |
| `openloop_suite.py` | uses only `ade_m` / `LON_speed_mae_mps` of `_components`; reads `families_paired` blocks | unaffected | `test_openloop_suite` green |
| refcv6 battery `refcv6_panel` | `_paired_families` + A3 | same | §2.4 |

⚠️ One more unmasked yaw read exists in `refav1_arm.py:1087`, the `gt_kappa` hint for the T0 oracle arms: `yaw_rate.mean / speed.mean.clamp_min(0.5)`. It is an oracle *input* recipe (`tools/gt_kappa.py`), not a metric cell, and it is left unchanged here. It is noted for the owner.

## 4. The claims

### 4.1 M26 / `D-FEASDEC-T1-1`: ⛔ AFFECTED. Retract the lateral headline.

| cell (proj07 vs. base; floor `ha0`) | landed (reproduced 0/114 differences) | fixed cell |
|---|---|---|
| `base:os` level | 0.2176 [0.144, 0.308], n 4,823 | **0.0320** [0.028, 0.0363], n 4,545 (278 dropped) |
| `proj07:os` level | 0.0427 [0.0324, 0.0573] | **0.0304** [0.0272, 0.0341], n 4,545 |
| `ha0` level | 0.0476 | 0.0414, n 4,532 |
| base − `ha0` | **+0.1700** [+0.1006, +0.2534], separated, "worse than the CV line" | **−0.0100** [−0.0168, −0.0037], separated. **Base beats the line** |
| proj07 − `ha0` | −0.0049 [−0.012, +0.0017], not separated | −0.0114 [−0.0181, −0.0053], separated |
| **(B−f) − (A−f)** | **−0.1749** [−0.2563, −0.1088], separated | **−0.0014** [−0.0029, −0.0004], separated, n 4,524 / 141 (299 dropped) |
| relative | −80.38 % (full set) | **−4.34 %** (on the cross cell's own kept set: 0.0314 → 0.0300) |

* **proj07e** (the +entry variant) reads the same to 4 dp.
* **projoff** (the disabled lever) reads 0.0 [0, 0] both before and after. That structural zero is the control and it holds.
* **Nothing else in M26 moves:** 0 of 108 non-yaw cells. ADE +0.0012 [0.0002, 0.0024], LON, heading, cross, TAC and STR are all byte-identical. Sources: `raw/m26/m26_rederivation.json` and `raw/m26/paired_*_vs_base.{TIPCELL,FIXED}.{json,md}`, which carry sha12 ids.
* **Mechanism**, from `raw/m26/m26_yaw_decomposition.json`:
  * The projection changed 421 windows. **306 of them have a GT step pair with no tangent.** On those 421 windows the old delta averaged −2.0041 rad/s. Only 131 of them are scorable after the fix, and they read −0.047.
  * The projection's own stop handling (`D-FEASDEC-STOPSTEP-1`: hold the heading through a stop, freeze the yaw) removed exactly the jitter the unmasked cell had scored. The lever "fixed" the metric's defect inside the path. That is what the −80 % measured.
* ⚠️ **ADJACENT, a different instrument (`fan_safety.score_paths`), NOT the `_components` defect.** Source: `raw/m26/m26_envelope_standstill.json`.
  * The banked envelope 0.0865 reproduces exactly.
  * **323 of the 417 violating base windows (77.5 %) are standstill or crawl windows**: a base step or a GT step below `min_ds`, or `v0` < 0.5 m/s.
  * **The moving-path violation rate is 94 / 4,823 = 1.95 %, and the projection takes it to 0.**
  * The structural zero survives. What changes is how "measurably safer" should be scoped.

### 4.2 refcv3 @40,284: LATERAL family (`D-REFCV3-40284c`, `MODEL_REGISTRY.md:2782-2797`, `HF_CARD_tanitad-refc-v3.md:364-373`, `LEADERBOARD.md:109`, `taniteval/results/*refcv3-40284-openloop*`): ⛔ AFFECTED, the sign flips

Record: `taniteval/results/refcv3-40284-openloop.ARM.json`. The families_paired blocks reproduce **70/70** exactly on `C:/Users/Admin/_wp56/dump/refcv3_40284_dump`, which establishes it is the pod dump the record names. Source: `raw/claims/refcv3_40284_arm.result.json`.

| pair (yaw, rad/s) | landed | fixed cell (n) |
|---|---|---|
| `os − ha0` | +0.1700 [+0.1006, +0.2534] LOST | **−0.0100 [−0.0168, −0.0037] WON** (4,524) |
| `os − ha` | +0.1849 [+0.1162, +0.2683] LOST | +0.0064 [+0.0023, +0.0107] LOST (4,535) |
| `os_navzero − ha0` | +0.1957 [+0.1186, +0.2864] | −0.0082 [−0.0154, −0.0011] (4,519) |
| `os − os_navzero` | −0.0257 [−0.0391, −0.0136] | −0.0022 [−0.0038, −0.0009] (4,537) |
| `ha − ha0` | −0.0149 [−0.0201, −0.0106] | −0.0162 [−0.0216, −0.0117] (4,523) |
| `oracle_sel − os` | +0.0025 [−0.0069, +0.0121], not separated | +0.0024 [+0.0010, +0.0043], separated (4,545) |
| `os − os_navshuf` | −0.0021, not separated | −0.0007 [−0.0015, 0.0], not separated |

* ⇒ **The LATERAL verdict vs. `ha0` goes from LOST (2/3 won) to WON (3/3 won).**
* The level (1.794 vs. 2.371 °/s, masked) and the paired cell **now agree in sign**. The registry's and the card's HYPOTHESIS, *"the sign flip is carried by near-stationary steps"*, is now **MEASURED**.
* ⚠️ **The HF card is public** (`HF_CARD_tanitad-refc-v3.md`; the publish receipt says the README was banked on HF). Republishing is a **PI decision**. This package edits nothing there. Whether the live README carries the row is UNVERIFIED (INHERITED from the claims audit).

### 4.3 `H-ESTIM-SEED-1` / `D-REPLICATE-FPRATE` / `CLAUDE.md` (the "separated is necessary, not sufficient" rule): ⛔ AFFECTED, 14.3 % → 9.5 %

* The panel's `paired_vs_A0` was re-derived from the per-arm npz files in `C:/Users/Admin/run_wbank/score_dumps`: **300/300 cells reproduce**. Source: `raw/claims/wbank_panel.result.json`.
* The count was redone with the register-repair walker's rule. Source: `raw/claims/replicate_fp_rate_fixed.json`. Control C2: the banked per-arm counts reproduce the register's "6 / 42" and its "8 / 16 / 18 / 39 / 40".

| arm | separated, banked → fixed cell | yaw cells | VERDICT 2-s 7-row view |
|---|---|---|---|
| **A0b_replicate** (zero levers) | **6/42 → 4/42 (14.3 % → 9.5 %)** | 3 → 1 | **3/14 → 2/14 (21.4 % → 14.3 %)** |
| A1_pred | 40/84 → 41/84 | 2 → 3 | 5 → 6 |
| A2_random | 18/63 → 19/63 | 3 → 4 | 9 → 9 |
| A3_drop25 | 16/42 → 15/42 | 1 → 0 | 7 → 7 |
| A4_none | 8/42 → 8/42 | 2 → 2 | 3 → 3 |
| A5_regress | 39/42 → 39/42 | 4 → 4 | 14 → 14 |

* A0b's yaw cells:
  * w/2 s +0.0759 [+0.0164, +0.1473] separated → +0.0318 [−0.0005, +0.0675], not separated;
  * k/6 s +0.0141 separated → +0.0086, not separated;
  * w/6 s +0.0183 separated → +0.0187 [+0.0019, +0.0368], **still separated**.
* **The rule survives.** Two runs that differ in nothing still read "separated" on 4 of 42 cells, about twice the nominal 5 %. The 55.6 % WP-D replicate rate that `PREREG_REFCV6_V2.ERRATUM-1` cites comes from a masked instrument and is unaffected (INHERITED, audit).
* **The number is what moves.** `CLAUDE.md:183` and about 80 other lines quote 14.3 % or 6/42 (`raw/propagation_sites.txt`, 81 lines).

### 4.4 `D-RL-VETO-T1-1`: ⛔ AFFECTED. The defect hid a real regression.

Sources: `raw/claims/veto_s0.result.json` and `veto_s1.result.json`. Both reproduce 114/114.

| | landed | fixed cell |
|---|---|---|
| s0 Δ (veto − base, difference of margins) | +0.0191 [+0.0060, +0.0316] | **+0.0251 [+0.0205, +0.0305]** |
| s1 Δ | +0.0478 [+0.0355, +0.0612] | **+0.0294 [+0.0248, +0.0345]** |
| two-seed floor \|s1 − s0\| | 0.0287 → "**WITHIN-NOISE**" | **0.0043 → s0 is 5.8× its floor: QUOTABLE** |
| levels (masked) | base 0.2176, s0 0.2367, s1 0.2654 | base 0.0320, s0 0.0572, s1 0.0625 |

⇒ **"7 of 11 non-structural rows are quotable regressions, 2 within-noise, 2 not separated" becomes "8 of 11 quotable, 1 within-noise (cross), 2 not separated."** The veto roughly **doubles** the masked yaw-rate error (0.0320 → 0.0572 / 0.0625). The unmasked cell's standstill noise had inflated the seed floor enough to hide it. This is the defect cutting the other way.

### 4.5 `D-REFCV3-40284e` (stratified, `taniteval/results/refcv3-40284-stratified.json`): ⚠️ AFFECTED, weakened

Reproduced 178/178 on the wp56 dump. The recorded scratch dump is gone; the exact reproduction establishes it was the same one. Source: `raw/claims/strat_40284.result.json`. The table shows `os − ha` yaw by stratum:

| stratum | landed | fixed cell |
|---|---|---|
| lane_keep | +0.2122, separated worse | +0.0083 [+0.0055, +0.0113], separated worse |
| turn_left / turn_right | not separated | not separated |
| brake_stop | +0.0451, separated worse | +0.0167 [+0.0104, +0.0226], separated worse |
| steady | +0.2150, separated worse | +0.0071 [+0.0033, +0.0118], separated worse |
| accelerate | +0.1448, separated worse | −0.0111 [−0.0250, +0.0015], **not separated** |
| straight_const | +0.2410, separated worse | +0.0070 [+0.0042, +0.0101], separated worse |
| manoeuvre | +0.0701, separated worse | +0.0052 [−0.0039, +0.0150], **not separated** |

* ⇒ "Loses yaw-rate MAE almost everywhere" becomes "**loses it by ≤ 0.017 rad/s on 4 of 8 strata (lane-keep, steady, straight-constant, brake-stop); ties on turns, accelerate and manoeuvre.**"
* The row's actual conclusion is untouched. It rests on ADE and the longitudinal class: *"a lateral improvement cannot move the headline; deceleration can move most of it."*
* Against `ha0`, the fixed cells read **better** on turn_left, turn_right, steady, accelerate and manoeuvre.

### 4.6 `PREREG_REFCV6.md` L1–L4 (the LATERAL non-regression clause): would it be judged with the shared cell?

**Findings, from source:**
1. **L2 names no instrument.**
   * `code/verdict_refcv6.py:256-296` reads `panel["families"]["LATERAL"]["yaw_rate_err"]` with fields `{arm, reference, separated, n}`, plus `replicate_floor["yaw_rate_err"]`.
   * **No code in the tree builds that panel.** `git grep yaw_rate_err` returns only the verdict tool, the dropproof test, the prereg and the register.
   * The only **paired** yaw-rate cell in the harness is `_components`' `LAT_yaw_rate_mae_radps`, via `refcv3_arm` `families_paired`, the battery's `cross_paired`, or `paired_openloop`. **So a panel builder written before today would have judged L2 with the defective cell.**
2. **The clause's reference values are a different statistic, in different units.**
   * 1.0534 / 1.4542 are **°/s four_families levels** (masked, a micro mean over steps), re-read from `2026-09-07-refcv5-v2-comparison/raw/refcv5-v2_vs_refcv4b.json` (§ "FACT 10'S REFERENCE COLUMN").
   * The separated test would be in **rad/s**, as a per-window mean. The verdict's relative margin is unit-free only if `arm` and `reference` come from the same cell.
3. **Governance.** `PREREG_REFCV6_V2.md:9-14`:
   * It **does not amend** `PREREG_REFCV6.md`.
   * `PREREG_REFCV6.md` *"governs the design the PI stopped on 2026-09-11 and nothing else."*
   * The live refcv6 (V2) has **no yaw-rate clause**: its lateral bar is masked curvature vs. `ha0_ext` (§9).
   * The battery's bars are ADE-only: *"No bar reads yaw-rate"* (battery SPEC A3).
   * ⇒ **No live verdict depends on L2 today.**

**Amendment, DRAFTED and NOT applied.** It is admissible now because no refcv6 final exists.

> **AMENDMENT A-YAW (drafted 2026-09-26, before any refcv6 final panel exists): L2 names its cell, its mask and its units.**
> Scope: `PREREG_REFCV6.md` §5B clause **L2**, and `verdict_refcv6.py` `REQUIRED_LATERAL["yaw_rate_err"]`. L1, L3 and L4 are unchanged. Heading was always masked, cross-track has no tangent, and masked curvature is four_families' own.
> 1. L2's separated test is the paired cell `LAT_yaw_rate_mae_radps` of `refav1_arm._components` **as fixed on 2026-09-26**: `pred.pair_valid ∧ gt.pair_valid`, a per-window mean, and a window with no valid pair dropped and counted. The estimator is the paired episode-cluster bootstrap, n_boot 2000, seed 0. `arm`, `reference`, `separated` and `replicate_floor["yaw_rate_err"]` **all come from that one cell, in rad/s, on its kept windows.**
> 2. The panel entry must carry the block's `yaw_rate_cell` stamp, which must contain `pred.pair_valid AND gt.pair_valid`, and must carry `n_dropped_nonfinite`. **Without them L2 is `MISSING_DATA`**, never PASS or FAIL. A block built before the fix cannot supply the stamp, so it cannot be scored.
> 3. The reference values 1.0534 / 1.4542 °/s stay as the **direction** anchor only (four_families level). They are never compared numerically with the rad/s paired cell.
> 4. No bar, tolerance or clause moves.
>
> Proposed guard for `verdict_refcv6.py`, in the L1–L4 loop, before `_margin_and_sep`:
> ```python
> if metric == "yaw_rate_err" and "pred.pair_valid AND gt.pair_valid" not in str(m.get("yaw_rate_cell", "")):
>     add(cid, "MISSING_DATA", {"metric": m}, "⛔ yaw_rate_err without the 2026-09-26 mask stamp: "
>         "the pre-fix shared cell scored standstill jitter (D-YAWMASK-1) and is inadmissible")
>     continue
> ```
> with a dropproof mutant: a `yaw_rate_err` entry with no stamp must read `MISSING_DATA`.

⚠️ **If V2 ever adopts the L clauses, it needs the same amendment.** The same holds for any panel builder written for the battery's step-30,000 or final read.

### 4.7 Everything else re-derived: ✅ UNAFFECTED (same separation on every quoted cell)

Every record reproduced **exactly** with the tip cell, and the fix moved **no non-yaw cell**. Per-record artifacts are in `raw/claims/<id>.result.json`, with a summary in `raw/claims/claims_summary.json`.

| claim (file:line on the tip) | record → pair | landed → fixed cell | why unaffected |
|---|---|---|---|
| `G:5533` refav1 step-1000 read | `refav1_t1_step1000.json`, `cl − ha` | −0.0477 [−0.0729, −0.0270] → −0.0486 [−0.0751, −0.0271] (n 137/20) | same sign, still separated |
| `LEADERBOARD:128` refav1 21,109 | `refav1-21109-openloop.json`, `cl − ha` | −0.0429 [−0.0569, −0.0301] → −0.0439 [−0.0584, −0.0301] (n 270/140) | same |
| `G:6387` `D-REFAV1-SURFACE-PAIRED` | `paired_delta_phase1.json`, `ccos_comp − cos` / `− ha0_ext` | +0.0159 [+0.0056, +0.0289] → +0.0159 [+0.0051, +0.0293]; −0.0250 → −0.0260 | "9 of 13" unchanged |
| `G:7147`, `DM5:1002`, `MR:2096` `D-REFAV1-CG-WK15` | `pd_all/pd_kamm/pd_fact`, `wk15 − ccos_argmax` | −0.1571 [−0.2579, −0.0670] (19×) → **−0.1935** [−0.2996, −0.0936] (floor +0.0098 → **19.7×**), n 33/8 | stronger, same verdict |
| `G:7155`, `DM5:1279` `D-REFAV1-CG-KAMM-ARM` | `kamm07 − ccos_argmax` | −0.0676 [−0.1476, −0.0038] (8.3×) → −0.0819 [−0.1735, −0.0050] (**8.4×**) | same |
| `G:7129`, `G:7175` seed floor ("4 of 10" / "4 of 8") | `pd_seedfloor`, `ccos_seed1 − ccos_argmax` | +0.0081, not separated → +0.0098, not separated | counts unchanged |
| `G:7157` / `G:7159` (L3 null / combined) | `l3ladder − ccos_argmax`; `combined − kamm07` | −0.0118 → −0.0143, not separated; −0.0145 → −0.0176, not separated | still not separated |
| `G:7673` `D-REFAV1-LON-D2-T1` | `pd_lonshift`, `lonshift − wk15` (seed `wk151 − wk15`) | +0.0146 (+0.0096) → +0.0211 (+0.0125), not separated | "LATERAL UNTOUCHED" holds |
| `G:7677` `D-REFAV1-LON-P4` | `lonshift − ha0_ext` | −0.1009 [−0.1461, −0.0525] → −0.1161 [−0.1694, −0.0600] | same |
| `G:45` `D-REFAV1-LON-T1`, dev-box half | `pd_devbox_lonshift` | the same cells as above; "yaw straddles" holds | Thor half: see §4.8 |
| `G:6203` `D-RL-FANSAFE-4` | `paired_rl_vs_base.json`, cross | +0.2836 [+0.2231, +0.3516] → +0.3151 [+0.2619, +0.3760]; RL margin over `ha0` +0.4536 → +0.3106, separated | "worse than the CV floor on every metric" holds |
| `G:11917-11944` refcv6-rl-stage-a | `paired_l1-rl-s{0,1}__vs__base.json` | base − s0: −0.0169 → −0.0179 [−0.0228, −0.0136]; base − s1: −0.0084 → −0.0097 [−0.0126, −0.0072]. Levels 0.0358 → 0.0527 become masked **0.0182 → 0.0359**. Two-seed floor ratio **1.5× → 1.7×** | "ALL TEN" holds; "the other eight at 0.6×–1.5×" becomes "0.6×–1.7×" |
| `G:11904` "0 of 10" (nav inert) | `t1_base_four_families.json` (reproduced 40/40 on `t1_base_dump`, which confirms the audit's inferred dump), `os − os_navzero` | +0.0000 [−0.0014, +0.0015] → 0.0000 [−0.0017, +0.0017] | unchanged |
| `G:6335` `D-REFCV4B-EGODROP2` | `EGODROP_9500.json` (2 s; derived dump present), `osw − osk` | +0.0634 [+0.036, +0.0979] → +0.0253 [+0.0192, +0.0313], still separated | no yaw number is quoted. ⚠️ Before *and* after, the paired yaw cell contradicts the row's *"only the longitudinal family moves"* (that claim is read off levels). Flagged for the owner; this defect is not the cause. |

Structural zeros (a lever that leaves the trajectory bit-identical) read exactly 0 before and after. Examples are `ccosh_w000 − ccos_argmax` and `cos_wk − ha0`, per the audit list in the manifest.

### 4.8 Not re-derived

| claim | why | what would unblock it |
|---|---|---|
| `G:45` `D-REFAV1-LON-T1`, the **Thor** half (`pd_thor.json`, "−0.0998 vs `ha0_ext`") | the dumps are on Thor (`/home/nvidia/refav1_lon/out/dump_T_*`, not banked), and this brief forbids Thor access | pull the three `dump_T_*` dirs, or bank them. The dev-box twin moved −0.1009 → −0.1161 (same sign, separated). **UNVERIFIABLE today.** |
| `G:6279` `H-KINGATE-1`, `DM5:787` ("curvature and yaw-rate SIGN-FLIP between draws, none separated") | **a different instrument** (`2026-09-05-kinematic-gate/raw/kin_gate_eval.py:410-449`) with its **own unmasked** `_head`, `_yaw` and `_kappa`. Same class, not `_components` | **the next lever:** re-score its banked fan npz with the `pair_valid` mask. The claim is a refutation of its own lateral hypothesis, so a masked read could revive or bury it. Offered as a separate task. |
| `G:7017` / `DM5:263` (M16 "every family degrades") | no yaw number and no artifact cited (INHERITED, audit) | name the artifact |
| `G:215` E-BEV-AUX-1, `G:11328` / `11330` P1 | their own **masked** instruments (`b9_pair_5b.py:112`, `p1-agent-gate/code/analyze.py:167`) | none needed. `G:11330` *cites* the 6/42 rate (§4.3). |

## 5. Integration consequences: read before landing

1. **The battery's landed `test_yaw_valid.py` goes 2 of 3 RED.** This is by design: it asserts the shared cell still reads π on a stopped window, and π/2 separated when paired. Source: `raw/battery_test_interaction.json`.
   * Measured in a clean tree: 3/3 green on the tip cell, **1 green / 2 red on the fixed cell.**
   * **`code/proposals/battery_test_yaw_valid.PROPOSED.py`** re-implements the historical line locally as its regression arm. It reads **4/4 green on the fix and 2/2 red on the tip**.
   * It is **not applied**: the package is the EvalFlyWheel's, and the brief says not to modify it. **Escalated.**
2. **`recompute_a3.py` would REFUSE any pre-A3 panel once the shared cell moves.** It requires every non-A3 cell, including the shared yaw cell, to be bit-identical. The step-5000 recompute is already banked at tip `51a4443` (`raw/step5000/A3_recompute_record.json`), so no pending tag is exposed. Future tags are computed by `cross_paired`, which already carries A3.
3. **The battery runs from `C:/Users/Admin/ev6`**, a separate clone:
   * `refcv6_loader.REPO` points there, and `post_tag.sh` sets `PYTHONPATH` to it;
   * the loader's comments cite `ev6 == 287d72e`.
   **Landing on the branch changes nothing live until that clone is synced.** After a sync, the shared and A3 cells are identical (§2.4).
4. **The public HF card** (§4.2) and **about 80 lines quoting 6/42 / 14.3 %**, including `CLAUDE.md:183` (§4.3; `raw/propagation_sites.txt`), are for the Master Mind to annotate. This package edits none of them.
5. `paired_openloop` records do not carry the new stamp (it goes on `_paired_families`). After the fix, a non-zero `n_dropped_nonfinite` on the yaw row is the tell on any dump with a stopped window.

## 6. Ready-to-append blocks

### 6.1 `Project Steering/GOALS_AND_CLAIMS.md`

⚠️ **The register already has an OPEN row `D-YAW-UNMASKED`** (tip `048ae3b`, `GOALS_AND_CLAIMS.md:15913`). It says *"the shared fix, and re-reading M26 and PREREG_REFCV6 L1–L4 under it, are with the yaw-rate agent"*. The first row below **closes it** under the same id. This package's working name, `D-YAWMASK-1`, is only an alias, so no new id is created for the fix itself.

```
| ⛔⛔ **D-YAW-UNMASKED — CLOSED 2026-09-26 (alias D-YAWMASK-1)** | **THE SHARED PAIRED YAW-RATE CELL SCORED STANDSTILL JITTER AS YAW-RATE ERROR — FIXED 2026-09-26.** `taniteval/tools/refav1_arm.py::_components` averaged `|yaw_rate_pred − yaw_rate_gt|` over EVERY step pair; `four_families.lateral` masks the same term with `pred.pair_valid & gt.pair_valid` (a step under `min_ds` = 0.25 m on the 0.5 s grid has no tangent). The cell feeds `_paired_families` (refav1_arm, refcv3_arm, the refcv6 battery), `paired_openloop`, `stratified_openloop`, `refav1_paired_delta`. Fixed: a per-window mean over the masked pairs, NaN (dropped and counted) with none — exactly as heading; every `_paired_families` block now stamps `yaw_rate_cell`, because the KEY did not change and its DEFINITION did. | **SUPPORTED (MEASURED 2026-09-26, 0 GPU).** New test 11/11 green on the fix, RED on the historical blob (7), the yaw line alone (6), a GT-only mask (1), a pred-only mask (3), a second-step mask (1); related suites in a clean tip tree 276 passed / 0 failed / 2 skipped (tip 265/0/2, same skips). On the refcv6 battery's step-5000 panel (4,754 windows) the fixed shared cell equals the battery's independently written A3 cell on every field of all 13 pairs, 117 non-yaw cells unchanged. 21 landed records re-derived with the tip cell reproduce 2,838/2,838 cells; the fix moves no non-yaw cell in any of them. | `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-26-yaw-rate-mask/RESULT.md`; `raw/regression_arms.json`, `raw/related_suites.json`, `raw/battery_a3_crosscheck.json`, `raw/claims/` |
| ⛔⛔ **D-FEASDEC-T1-1-YAW** | **M26's LATERAL HEADLINE IS RETRACTED: the feasibility projection's yaw-rate effect is −0.0014 rad/s, not −0.1749.** Re-run of `paired_openloop.py` on the SAME dev-box dumps (reproduction control 0/114 cells), only the yaw cell swapped: `base:os` 0.2176 → **0.0320**, `proj07:os` 0.0427 → **0.0304** rad/s (n 4,545 each); Δ (difference of margins over `ha0`) −0.1749 [−0.2563, −0.1088] → **−0.0014 [−0.0029, −0.0004] separated** (n 4,524 / 141, 299 without a valid pair), **≈ −4.3 %** not −80.4 %. ⛔ *"from worse than a constant-velocity straight line to better than it"* is FALSE: `base` already beats `ha0`, **−0.0100 [−0.0168, −0.0037]**. Mechanism: 347 base windows with a tangent-less pair carried 86.9 % of the unmasked sum; 306 of the 421 windows the projection changed have a GT stop — its own stop handling removed exactly the jitter the cell scored. ADE +0.0012, LON, heading, cross, TAC, STR: **0/108 cells move**. ⚠️ Adjacent (a different instrument): 323/417 (77.5 %) of the base's `envelope` violations are standstill/crawl windows; the MOVING-path rate is **94/4,823 = 1.95 % → 0** — the structural zero stands, its scope is narrower. | **SUPPORTED (MEASURED 2026-09-26, T1)** — retracts the yaw row of `D-FEASDEC-T1-1` and Decisions 2026-09-05 M26 §1 | `…/2026-09-26-yaw-rate-mask/raw/m26/` (`m26_rederivation.json`, `m26_yaw_decomposition.json`, `m26_envelope_standstill.json`, `paired_*_vs_base.{TIPCELL,FIXED}.*`) |
| ⛔ **D-REFCV3-40284c-RESOLVED** | **THE TWO refcv3 YAW-RATE NUMBERS NO LONGER DISAGREE: the paired cell flips to a WIN and refcv3's LATERAL family verdict goes LOST (2/3) → WON (3/3).** `refcv3-40284-openloop.ARM.json` families_paired reproduced 70/70 on the wp56 dump; with the fixed cell `os − ha0` yaw +0.1700 [+0.1006, +0.2534] → **−0.0100 [−0.0168, −0.0037]** (n 4,524); `os − ha` +0.1849 → +0.0064 [+0.0023, +0.0107] (n 4,535); `os_navzero − ha0` +0.1957 → −0.0082. The registry's and HF card's HYPOTHESIS (*"the flip is carried by near-stationary steps"*) is now MEASURED. | **SUPPORTED (MEASURED 2026-09-26, T1).** ⛔ `MODEL_REGISTRY.md` refcv3 LATERAL row, `LEADERBOARD.md:109`, `taniteval/results/*refcv3-40284-openloop*.md` and the **public** `HF_CARD_tanitad-refc-v3.md` carry the old verdict — republishing the card is a PI decision | `…/2026-09-26-yaw-rate-mask/raw/claims/refcv3_40284_arm.result.json` |
| ⛔ **D-REPLICATE-FPRATE-2** | **THE REPLICATE FALSE-POSITIVE RATE IS 4/42 = 9.5 %, NOT 6/42 = 14.3 % — three of the six were the defective yaw cell.** Withheld-bank panel `paired_vs_A0` re-derived from `run_wbank/score_dumps` (reproduction 300/300 cells), yaw cells swapped: A0b_replicate 6/42 → **4/42**; the VERDICT 2-s 7-row view 3/14 → **2/14 = 14.3 %**; the other arms 40/18/16/8/39 → 41/19/15/8/39. The rule of `H-ESTIM-SEED-1` STANDS (two runs differing in nothing still separate on 4 of 42 cells, ~2× nominal); the NUMBER changes at ~80 quoting sites incl. `CLAUDE.md:183`. | **SUPPORTED (MEASURED 2026-09-26)**; C2 control: the banked counts reproduce `D-REPLICATE-FPRATE`'s 6/42 and 8/16/18/39/40 | `…/2026-09-26-yaw-rate-mask/raw/claims/replicate_fp_rate_fixed.json`, `wbank_panel.result.json`, `raw/propagation_sites.txt` |
| ⛔ **D-RL-VETO-T1-1-YAW** | **THE DEFECT HID A REAL LATERAL REGRESSION: the veto doubles the masked yaw-rate error and it is QUOTABLE, not within-noise.** Both seeds re-derived (114/114 each): Δ +0.0191/+0.0478 (floor 0.0287, "WITHIN-NOISE") → **+0.0251 [+0.0205, +0.0305] / +0.0294 [+0.0248, +0.0345], floor 0.0043, 5.8×**; levels base 0.0320 → s0 0.0572 / s1 0.0625. ⇒ "7 of 11 quotable" → **8 of 11 quotable, 1 within-noise (cross), 2 ns**. | **SUPPORTED (MEASURED 2026-09-26, T1)** — amends `D-RL-VETO-T1-1` | `…/raw/claims/veto_s{0,1}.result.json` |
| ⚠️ **D-REFCV3-40284e-YAW** | **"refcv3 loses yaw-rate MAE almost everywhere" is WEAKENED to "by ≤ 0.017 rad/s on 4 of 8 strata".** Stratified re-run reproduced 178/178; `os − ha` yaw, fixed cell: lane_keep +0.0083, brake_stop +0.0167, steady +0.0071, straight_const +0.0070 (all separated worse); accelerate −0.0111 and manoeuvre +0.0052 NOT separated (were +0.1448 / +0.0701 separated worse). The row's ADE/longitudinal conclusion is untouched. | **SUPPORTED (MEASURED 2026-09-26, T1)** | `…/raw/claims/strat_40284.result.json` |
| **D-YAW-UNMASKED-SCOPE** | **14 further landed claims that quote the cell were re-derived and do NOT move** (same separation on every quoted cell; magnitudes ≤ ~25 %): refav1 step-1000 and 21,109 reads; `D-REFAV1-SURFACE-PAIRED`; the p4 cost-geometry rows (`wk15` 19× → 19.7×, `kamm07` 8.3× → 8.4×, seed floor +0.0081 → +0.0098 ns); `D-REFAV1-LON-D2-T1` / `-P4` / `-LON-T1` (dev-box half); `D-RL-FANSAFE-4`; refcv6-rl-stage-a (ALL TEN holds, yaw floor ratio 1.5× → 1.7×); the nav-inert "0 of 10"; EGODROP. NOT re-derived: the Thor half of `D-REFAV1-LON-T1` (Thor-only dumps); `H-KINGATE-1` (its own unmasked `_yaw`/`_head`/`_kappa` in `kin_gate_eval.py` — the next lever). | **SUPPORTED (MEASURED 2026-09-26)** | `…/raw/claims/claims_summary.json` and the per-record `*.result.json` |
```

### 6.2 `Project Steering/RETRACTION_LOG.md`

```
### RETR-2026-09-26-YAWMASK — four landed yaw-rate readings came from a cell that scored standstill jitter

**Retracted / corrected** (every one re-derived on its own dumps; reproduction control exact on each):
1. Decisions 2026-09-05 **M26** / `D-FEASDEC-T1-1`: *"yaw-rate error 0.2176 → 0.0427 rad/s, −80.4 %, separated"* and
   *"yaw-rate goes from worse than a constant-velocity straight line to better than it"* → **0.0320 → 0.0304 rad/s,
   Δ −0.0014 [−0.0029, −0.0004], ≈ −4.3 %; base already beats the line (−0.0100 [−0.0168, −0.0037])**. Retracted.
2. refcv3 @40,284 LATERAL *"LOST (2/3 won), yaw-rate +0.1700 [+0.1006, +0.2534]"* (`D-REFCV3-40284c`, MODEL_REGISTRY,
   HF card, LEADERBOARD) → **−0.0100 [−0.0168, −0.0037] WON; family WON (3/3)**. The sign flips.
3. `H-ESTIM-SEED-1` / `D-REPLICATE-FPRATE` / `CLAUDE.md`: *"6 of 42 = 14.3 %"* (and *"3 of 14 = 21.4 %"*) →
   **4/42 = 9.5 % (2/14 = 14.3 %)**. The rule stands; the number was half defect.
4. `D-RL-VETO-T1-1`: *"`LAT_yaw_rate_mae_radps` … WITHIN-NOISE"* → **QUOTABLE regression (5.8× its floor)** —
   the defect hid a real effect.

**Class: AN ADMISSIBILITY MASK PUBLISHED BY THE PRODUCER AND APPLIED TO ONE SIBLING METRIC BUT NOT THE NEXT** —
`_seq_geometry` returns `valid` and `pair_valid`; `_components` put the first on heading and never put the second on
yaw-rate, under a docstring that said "never a re-derivation" (the GEOMETRY was shared; the REDUCTION was re-derived).
It is C29's consequence (2) in an eval cell — *a mask that no caller uses* — and `D-FEASDEC-STOPSTEP-1`'s
"`atan2(0,0) == 0` is a heading, not an undefined" one level up, in the metric instead of the path.

**Why it survived for weeks, and the part worth keeping:** the contradiction was ON THE RECORD. The registry and the
HF card both printed "TWO YAW-RATE NUMBERS DISAGREE IN SIGN AND BOTH ARE MEASURED", filed it as a *scope trap*, wrote
down the correct HYPOTHESIS (near-stationary steps) — and kept the unmasked number as the family verdict. A
disagreement in SIGN between two instruments on one metric is a DEFECT REPORT, not a scope note; the discriminating
experiment (mask the pairs, re-read) cost minutes and was run only when a battery tripped over it (F10).
⛔ And the feasibility lever "won" −80 % by removing, inside the PATH, the very jitter the METRIC was scoring — a
lever and an instrument defect cancelling, read as skill.

**Durable fix:** `_components` masks yaw-rate exactly as heading, and every `_paired_families` block stamps
`yaw_rate_cell` (a key whose definition changed must name its definition); `taniteval/tests/
test_refav1_components_yaw_mask.py` pins literal targets and goes RED on the historical defect and on three wrong
masks. **Rule:** a metric built from a geometry that publishes a validity mask must use the mask the reference
estimator uses, and must carry a known-value control on a stopped window (undefined → dropped, never scored).
```

## 7. What this does NOT establish

* ⛔ **Nothing about closed-loop driving.** Every number is T1, a self-action open loop.
* ⛔ **Nothing about training or inference variance** beyond what each claim's own replicate already measured. The bootstrap answers the episode question only.
* ⛔ **Nothing about the `envelope` instrument's correctness.** §4.1's standstill share is a scoping observation about `fan_safety.score_paths`. That instrument was not audited.
* ⛔ **No claim about the Thor-only dumps**, and none about `H-KINGATE-1` (a separate instrument).
* The battery's per-arm 0.2034 / 0.0318 are **INHERITED**; its paired cells were **re-measured** (§2.4).
* ⚠️ **The full `stack/` suite was NOT run.** Only the 16 related files were (§2.3). The pre-commit gate should run the full suite on the combination that actually lands.

## 8. Deliverable manifest

Every path below is relative to this package, `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-26-yaw-rate-mask/`, which lives on **D: only**. Nothing has been staged; the Master Mind is the single committer. `LANDING_READY.txt` lists the repo paths.

| artifact | what | lives |
|---|---|---|
| `code/fix/taniteval/tools/refav1_arm.py` | **THE FIX.** Blob `963d98e6`, based on tip blob `c7013107`, which is unchanged at tip `048ae3b` | package (land at `taniteval/tools/refav1_arm.py`) |
| `code/fix/taniteval/tests/test_refav1_components_yaw_mask.py` | **THE TEST**, a new file, blob `10d4d683` | package (land at `taniteval/tests/`) |
| `RESULT.md`, `LANDING_READY.txt` | this report and its landing list | package |
| `code/run_regression_arms.py` | clean tree, regression arms, related suites | package |
| `code/rederive_m26.py`, `code/rederive_claims.py` | the claim re-derivations: tip cell and fixed cell through each claim's own instrument | package |
| `code/probes/{m26_yaw_decomposition,m26_envelope_standstill,battery_a3_crosscheck,replicate_fp_rate_fixed}.py` | probes | package |
| `code/summarize_battery_interaction.py`, `code/propagation_sites.py`, `code/sanitize.py` | helpers (sha12 sanitizer, UUID scan) | package |
| `code/proposals/battery_test_yaw_valid.PROPOSED.py` | the proposed battery test. **NOT APPLIED; owner: EvalFlyWheel** | package |
| `raw/regression_arms.json`, `raw/related_suites.json`, `raw/battery_test_interaction.json`, `raw/battery_a3_crosscheck.json` | test and cross-check artifacts | package |
| `raw/m26/*` (15 files) | M26 re-derivation, decomposition, envelope scope | package (sha12-sanitized; scan reads 0 UUIDs) |
| `raw/claims/*` (23 files) | 21 per-record re-derivations, the summary, the FP-rate recount | package (sanitized) |
| `raw/propagation_sites.txt` | 103 lines on the tip that quote a moved number | package |
| `C:/Users/Admin/ym26/` (`tree`, `tree_m26`, `te_tip`, `te_fix`, `probe_tools`, `battery_code`, `claims`, `m26_out`, `junit`) | scratch trees, unsanitized raw tool outputs, JUnit XML | ⚠️ **dev box only, scratch.** Every reportable artifact is already in the package. Safe to delete. |
