# SIM_DESIGN: EP-ADMISSION R1 re-simulation at C1's geometry (24 logs, 905 tokens x 3 search seeds) (Research Lab agent, 2026-10-19; MM_RULING_SPC.md DQ6)

**Written and hashed BEFORE the first run** (`SIM_DESIGN.sha256` beside it; `epadm_sim_c1.py` refuses to start without a matching sidecar). Evidence class of every rate: **SIMULATED on the real estimator** (not MEASURED on a candidate). The only runs before this hash were the unit tests on planted data and the adapter's build / identity check on carve B (which computes no admission statistic).

## 1. Question and the bars (committed by the Master Mind; copied, not chosen here)
Do the R1 gate's error rates transfer to C1's geometry? From `MM_RULING_EPADM_R1.md` section 2, ratified unchanged by `MM_RULING_SPC.md` DQ6:
1. **A1-r1 false refusal**: Wilson 95 % **upper <= 5.0 %** (rule: FAIL iff on ANY of the 3 seeds D_M lower CI > 0 OR D_M point > +0.05).
2. **A2 (margin 1.05) false refusal**: Wilson 95 % **upper <= 5.0 %** (rule: FAIL iff on ANY seed, for ANY of NC / DAC / TTC, the lower CI of observed / predicted unsafe mass > 1.05).
3. **DAC power at x1.25 (margin 1.05)**: **point >= 70 %** (rule: a planted DAC under-prediction of x1.25 is refused iff on ANY seed DAC's lower CI > 1.05).
**All met => admission is read on C1's bank. Any missed => fallback (b) (an S0-geometry bank of the search), the miss reported; the Master Mind's call.** No margin, bar or rule is tuned: `decide()` in `epadm_sim_c1.py` is a pure function of three rates with these constants (tested, with mutation arms).

## 2. The object being simulated (the adapter, `epadm_searchbank.py`)
One P2 bank (one set per token) and **three candidate banks, one per search seed** (one set per (token, seed)), laid out in the frozen gate's own format: the search output lives in a host slot of the 64-slot fan, carries its own label, and the frozen pick rule reads it (guard G4). The gate's seed quantifier ("any seed") is therefore applied over the 3 search seeds, as R1 applied it over 2 EP seeds. Geometry: **24 logs**; tokens per log are C1's (905 in all, `raw/geometry_c1.json`, derived from token lists only: heldout minus Step A minus carve B; its sorted-token sha256 equals the sealed B-REST pin `3a85cc81...`; per log min 5, median 21.5, max 131; effective cluster count 12.4).

## 3. The base population (the one thing that is not in R1's design)
R1 resampled the REAL 117-log S0 bank. C1's bank does not exist yet (and no B-REST label is read). The base is therefore **carve B** (400 tokens on the SAME 24 logs, already read, the adapter's test fixture), read through the gate's own loaders:
* per-SET vectors of the R1 simulator's components (`token_vectors`; their per-log sums equal `ep_admission_sim.components` bit for bit, tested);
* **replicate base**: for each log l, `n_l` (C1's count) sets are drawn WITH REPLACEMENT from carve B's sets of that log (`default_rng([BASE_SALT, rep + offset])`), the same draw for all 3 seeds (the seeds share the tokens); per-log sums by `np.bincount`. This is "fresh tokens from the same 24 logs at C1's per-log counts" and carries carve B's real label noise, its heterogeneity and its sparse unsafe events; the changed-pick share is carve B's measured one (the search's own behaviour), not a design choice;
* **calibration by construction and plants: the R1 simulator's own `build_condition`** (`a_u, a_p, a_e, k_h` scalars on the replicate's base, so every population ratio is exactly 1 and D_M exactly 0; `plant_ep`: +delta at the changed picks; `plant_a2`: one head under-predicts risk by rho);
* **generator A (as R1's primary)**: `idx = default_rng([OUTER_SALT, rep + offset]).integers(0, 24, 24)` redraws the 24 logs with replacement, shared by the 3 seeds and all heads; then the REAL estimator on the drawn logs: `fastneg_lib.diff_ci` (A1) and `ratio_ci` (A2) with the registered draws `boot_draws(24, 10000, 20260927)` (paired log-cluster bootstrap, B = 10,000, seed 20260927, percentile 2.5 / 97.5). Nested: the rate is the frequency with which the CI rule fires when another draw of tokens AND of logs is taken from a population whose true D_M is 0 and true ratios are 1.
* **NEW replicate seeds**: `BASE_SALT = 20261201`, `OUTER_SALT = 20261202`. R1 used 20261027 / 20261028 / 20261029 and EP-RELIAB's null used 20261019; none is reused (tested). The first run uses `--rep-offset 0` of these new streams.
* The carve-B candidate's own reliability (its raw D_M, calibration scalars, A2 ratios) is **not** an output of this study: the RAW (uncalibrated) condition of R1 is omitted, and the scalars are not written to any result file.

## 4. Conditions and replicate counts (fixed here)
| condition | decides | N |
|---|---|---|
| `CAL` (calibrated) | A1-r1 and A2 (all heads, all 3 seeds) on the same replicates; false refusal at margins 1.00 (the module default), **1.05 (the bar)**, 1.10 | **2,000** |
| `PLANT_A2_DAC_1.25` | DAC power at x1.25 (margin 1.05 is the bar; 1.00 and 1.10 beside it) | **2,000** |
| `PLANT_A2_{NC,TTC}_1.25`, `PLANT_A2_{NC,DAC,TTC}_1.5` | power, descriptive | 1,000 each |
| `PLANT_EP_0.135`, `PLANT_EP_0.05` | A1-r1 power, descriptive; the PC2 teeth | 1,000 each |
| `IDENTITY` (P2 as the candidate) | PC6 | 50 |
N >= 1,000 for every condition as the brief requires; the two bar conditions are 2,000 (Wilson half-width at p = 0.05: 1.0 pp). Wilson 95 %. Run as 2 processes (conditions split by index), each under 2 GB, CPU only.

## 5. Controls (all gating: any failure VOIDS the simulation; tolerances are R1's, copied)
* **PC1 calibrated by construction**: over the `CAL` replicates the mean `M_p`, `M_e` of each seed are within 0.01 of 1, the mean D_M within 0.005 of 0, the mean A2 ratios within 0.05 of 1.
* **PC2 teeth**: `PLANT_EP_0.135` and `PLANT_EP_0.05` must read a mean D_M within 0.005 of the planted delta AND fail the calibrated form of PC1.
* **PC3 the shortcut IS the estimator**: on a SYNTHETIC adapted bank (planted data, built by the adapter), the per-log shortcut (`diff_ci` / `ratio_ci` on bincount vectors) equals the frozen gate's own `analyze_candidate` D_M and A2 intervals to 0.0 for all 3 seeds. No real candidate statistic is computed for this.
* **PC4 the decision code is the module's**: on every replicate the inline restatement of each rule equals the frozen module's verdict (A1: `a1_verdict_r1_seeds`; A2 margin 1.00: `a2_verdict_r1_seeds` / `a2_verdict_r1`; margin 1.05 / 1.10: `a2_head_ok_r1(ci, bar=margin)`), a NaN bound failing closed in both; 0 disagreements.
* **PC5 determinism**: the first 20 replicates of `CAL` and `PLANT_A2_DAC_1.25` recomputed in a separate process are bit-identical.
* **PC6 identity**: P2 as the candidate reads D_M exactly [0, 0, 0] on every replicate and is never refused.

## 6. Flags committed in advance (reporting only; they change nothing)
* **R-PC1BIAS**: if PC1 fails ONLY on the A2 ratio means (|mean - 1| > 0.05) the run is VOID as written; it would be repeated under a new, hashed `SIM_DESIGN` revision that states the bias, never by loosening a tolerance in place.
* **R-NAN**: the number of NaN intervals (a replicate whose base has no unsafe event for a head) is reported per condition; they count as refusals (fail closed), as the module does.
* **R-GEOM**: the pool's changed-pick counts (carve B, per seed) are reported, because A2 power falls with fewer changed picks (R2-CONFIRM T5).

## 7. What these rates are NOT
* They answer "another draw of tokens and logs from a calibrated population", i.e. the gate's false-refusal / power; NOT another training run of P2 (one seed, `H-ESTIM-SEED-1`) and NOT another inference run beyond the 3 search seeds the gate itself quantifies over.
* The base is carve B's token pool; C1's real tokens may differ in their label noise and unsafe-event rate. The simulation transfers the GEOMETRY (24 logs, per-log counts, 3 seeds, the search's changed-pick share) and carve B's noise structure, not C1's data.
* Calibrated means D_M = 0 and observed / predicted = 1 on average at the picks. A candidate calibrated on average but wrong in a log-concentrated way is not simulated; the planted over- / under-prediction is spread over all changed picks (the EASY case for detection).
* P2 is in-sample on S0 and SEEN-SCENE on carve B (the bank inherits it). Tier: SELECTION, label lookup, T0-class; nothing here is a navtest or driving claim.
* The A2 margin 1.05 is NOT in the frozen module (its constant is 1.00): the simulated 1.05 rule is the registration's, applied by `a2_head_ok_r1(ci, bar=1.05)`.
