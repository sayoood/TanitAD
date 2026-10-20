# SIM_DESIGN_S0: EP-ADMISSION R1 re-simulation at the S0 OWN-FAN geometry of a search-output bank (EPADM-SEARCHBANK-S0, Research Lab agent, 2026-10-20)

**Written and hashed BEFORE the first run that reads a rate** (`SIM_DESIGN_S0.sha256` beside it; `epadm_sim_s0.py` refuses to start without a matching sidecar). Evidence class of every rate this design produces: **SIMULATED on the real estimator** (not MEASURED on a candidate). The only runs before this hash are unit tests on PLANTED data and the geometry file build (token counts only). No label or search output is computed anywhere in this design: the two bases are already-banked arrays.

## 1. Question and the bars (committed by the Master Mind; copied, not chosen here)
Do the R1 gate's error rates (`MM_RULING_EPADM_R1.md` section 2, ratified unchanged by `MM_RULING_SPC.md` DQ6) hold for a SEARCH-OUTPUT bank at the S0 own-fan geometry, for two data options:
* **GS41**: the 41 frame-complete S0 logs (2,782 tokens), registered-table departure (`--no-ranked-pin`, own 41-row table);
* **GS117**: all 117 S0 logs (8,922 tokens), the registered ranked table (sha256 `a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1`) unchanged.
(`GS43` = GS41 + the two logs whose frames sit in `navtrain_current_25.tgz`, 2,919 tokens, is NOT simulated: +4.9 % tokens; reported as arithmetic only.)

The three bars, per geometry (rule restated: A1-r1 FAIL iff on ANY of the 3 search seeds D_M lower CI > 0 OR D_M point > +0.05; A2 FAIL iff on ANY seed, for ANY of NC / DAC / TTC, the lower CI of observed / predicted unsafe mass > the margin 1.05; the bootstrap is the registered paired log-cluster bootstrap, B = 10,000, seed 20260927, over the geometry's logs):
1. **A1-r1 false refusal**: Wilson 95 % upper <= 5.0 %.
2. **A2 (margin 1.05) false refusal**: Wilson 95 % upper <= 5.0 %.
3. **DAC power at x1.25 (margin 1.05)**: point >= 70 %.
`decide()` in `epadm_sim_s0.py` is a pure function of these three rates with these constants (tested, with mutation arms). **No margin, bar or rule is tuned.**

## 2. The object being simulated
The adapter's layout (`epadm_searchbank.py`, sha256 `c4d6d76f...`): one P2 bank (one set per token = P2's OWN fan, P2's pick slot) and three candidate banks, one per search seed (20260927 / 28 / 29), the search output laid in a host slot of the 64-slot fan with its own label. Unit = one TOKEN. Sets per bank: GS41 2,782; GS117 8,922. Tokens per log = the registered table's `n_fit_tokens` (`geometry_s0.json`, sha256 `7a2e71e6d8e8c9707faab9bafa57fb6dd54b53c6ce6a471ffbf9e7b75124a7ca`; equal to `s0probe/s0_inputs.json` on all 117 logs). The cluster for the bootstrap is the LOG: 41 or 117 clusters. The S0 STORED-fan bank (15,808 sets: on-policy fans of earlier checkpoints scored by P2) is a DIFFERENT object and is used only as Base X (section 3).

## 3. The two bases (the one thing no S0 search bank gives us yet)
**Base C (PRIMARY: the candidate type is a real search output).** Carve B's adapted bank (`epadm_search/work/carveB_bank2`: 400 tokens, 24 logs, 3 search seeds, real P2HEAD outputs and labels; read through the frozen gate's loaders; bank manifest sha256 `7106ae14...` / `621319ab...` / `6344c5ee...` / `dcc4977f...`). Replicate base: for every S0 log l with n_l tokens, a SOURCE carve-B log c(l) is drawn with probability proportional to its number of carve-B sets, then n_l sets are drawn WITH REPLACEMENT from c(l)'s sets (`default_rng([S0_BASE_SALT, rep + offset])`), the same draw for all 3 seeds (the seeds share the tokens). A simulated log therefore keeps one real log's heterogeneity and the search's real changed-pick share (26.5 % in the pool). Calibration by construction and plants: the R1 simulator's own `build_condition` (so every population ratio is exactly 1 and D_M exactly 0 before the planted shift).
**Base X (CROSS-CHECK: the real S0 label distribution, a synthetic candidate).** The R1 simulator's per-log sufficient statistics on the real S0 stored bank for geometry `G25` (candidate = the EP-led composite rows on a Bernoulli subset of sets, about 25 % changed picks; 2 seeds), read from `stats.npz` of EP-ADMISSION-R2-CONFIRM (sha256 `4c34b63960ee0b2f3b5662e47f1a4a978a263f29c33963593cb9001b99d514a8`), restricted to the geometry's logs (calibration scalars recomputed on the restriction). The per-log sums are of STORED SETS (4,941 / 15,808), not of own-fan tokens (2,782 / 8,922): Base X is therefore OPTIMISTIC in precision by about sqrt(1.77) = 1.33x in standard error. It has 2 seeds, so its "any seed" quantifier is weaker than the 3-seed rule.
Neither base is S0's own-fan search bank, which does not exist. Their disagreement is reported, not averaged.

**Generators.** `A` (PRIMARY, as in R1 and in the C1 re-simulation): `idx = default_rng([S0_OUTER_SALT, rep + offset]).integers(0, N, N)` redraws the geometry's N logs with replacement, shared by all seeds and heads; then the REAL estimator `fastneg_lib.diff_ci` / `ratio_ci` on the drawn logs with `boot_draws(N, 10000, 20260927)`. `T` (CROSS-CHECK, Base C only): the logs are FIXED (`idx = arange(N)`), only the tokens are re-drawn; it answers "this bank, other tokens", free of the duplicate-log artefact of `A` (the C1 run's `A` drew 15.3 distinct logs of 24).
**NEW replicate seeds:** `S0_BASE_SALT = 20261301`, `S0_OUTER_SALT = 20261302`. R1 used 20261027 / 28 / 29, EP-RELIAB's null 20261019, the C1 re-simulation 20261201 / 02; none is reused (tested). The first run uses `--rep-offset 0`.

## 4. Conditions and replicate counts (fixed here)
Per (base, geometry), generator A: `CAL` 2,000 (decides bars 1 and 2); `PLANT_A2_DAC_1.25` 2,000 (decides bar 3; margins 1.00 and 1.10 beside it); `PLANT_A2_{NC,TTC}_1.25` and `PLANT_A2_{NC,DAC,TTC}_1.5` 1,000 each (descriptive); `PLANT_EP_0.135`, `PLANT_EP_0.05` 1,000 each (A1 power, PC2 teeth); `IDENTITY` 50 (PC6). Generator T (Base C): `CAL` and `PLANT_A2_DAC_1.25`, 1,000 each. Wilson 95 %. At most 2 processes, CPU only.

## 5. Controls (all gating unless marked REPORTED; any failure VOIDS that run)
* **PC1a calibrated by construction**: over `CAL` the mean `M_p`, `M_e` of each seed within 0.01 of 1 and mean D_M within 0.005 of 0 (R1's tolerances).
* **PC1b base exactness (replaces the A2-ratio-mean tolerance of C1's PC1, whose failure VOIDED that run, flag R-PC1BIAS)**: on EVERY replicate, seed and head with a non-zero observed unsafe mass in the replicate base, observed / predicted at the base is 1 to within 1e-12 BEFORE the log redraw. **REPORTED, not gating:** the mean A2 ratio after the redraw per head (a ratio of sums over few logs is biased; C1 read DAC 1.09-1.10).
* **PC2 teeth**: `PLANT_EP_0.135` and `PLANT_EP_0.05` read a mean D_M within 0.005 of the planted delta AND fail the calibrated form of PC1a.
* **PC3 the shortcut IS the estimator**: on a SYNTHETIC adapted bank (planted data, built by the adapter) the per-log shortcut equals the frozen gate's `analyze_candidate` D_M and A2 intervals to 0.0 for all 3 seeds (`epadm_sim_c1.py pc3`).
* **PC4 the decision code is the module's**: on every replicate the inline restatement of each rule equals the frozen module's verdict (A1: `a1_verdict_r1_seeds`; A2 margin 1.00: `a2_verdict_r1_seeds`; margins 1.05 / 1.10: `a2_head_ok_r1(ci, bar)`); a NaN bound fails closed in both; 0 disagreements.
* **PC5 determinism**: the first 20 replicates of `CAL` and `PLANT_A2_DAC_1.25` recomputed in a separate process are bit-identical.
* **PC6 identity**: P2 as the candidate reads D_M exactly [0, 0, 0] on every replicate and is never refused.
* **PC7 geometry and clone integrity**: the geometry file's token counts sum to 2,782 / 8,922 and equal the registered table's; in replicate 0 every simulated log's sets all come from ONE source log (tested with a RED input that mixes sources).
* **PC8 the code is the R1 simulator (Base X)**: with R1's own salts and `--rep-offset 1,000,000`, the GS117 / Base X `PLANT_A2_DAC_1.25` replicates 0-19 equal the banked replicates of `a2_power_g25_g05_replicates.npz` of EP-ADMISSION-R2-CONFIRM (`G25|A|PLANT_A2_DAC_1.25`; dm, a2, mp, me) bit for bit.

## 6. The NaN policy for an undefined A2 ratio (stated BEFORE the run; the C1 run left it open)
Calibration by construction sets `k_h = sum OBS_h / sum PRED_h`; when a replicate base has NO observed unsafe pick for head h, `k_h = 0` wipes the prediction and the ratio is 0/0 (NaN), which the frozen module fails closed. A real bank has `PRED_h > 0` (the row's sigma is below 1), so its ratio would be 0 and the head would pass. **Primary convention ("real-sigma")**: for such a replicate-seed-head the prediction is left UNSCALED (`PRED_h` as in the base, divided by rho if h is the planted head), which is what the real gate would see. **Second convention ("fail-closed", the C1 run's)**: such a replicate-seed-head counts as a refusal. Both are computed from the same replicates (the flag `k0[rep, seed, head]` is stored), both are REPORTED, and **the bars are read on the primary convention; a bar met on it and missed on the second is flagged R-NAN.** The count of `k0` replicates per head is reported per condition.

## 7. Status of each bar (committed now, a mechanical rule; the consequence of a status is NOT decided here)
Per geometry and per bar: **MET** = met on Base C (generator A) AND on Base X; **MISSED** = missed on Base C; **BASE-DEPENDENT** = met on Base C and missed on Base X. The geometry is **ALL-MET** only if all three bars are MET; any other combination is reported with the miss (never tuned, never a re-run with another margin). The consequence is the amendment's section on misses (DQ), not this file's.

## 8. Flags committed in advance (reporting only; they change nothing)
* **R-NAN**: a bar's status differs between the two NaN conventions.
* **R-EVENTS**: the distinct unsafe-pick counts in the Base C pool (NC / DAC / TTC per seed) are printed beside every rate: a base with 1 NC event cannot test NC.
* **R-GEN**: generator A and T false-refusal rates (Base C) differ by more than 3 percentage points.
* **R-SEED**: Base C (3 seeds) and Base X (2 seeds) rates for the same geometry differ by more than 5 percentage points.
* **R-DUP**: the mean number of distinct logs under generator A (reported per geometry; it is below N by construction).
* **R-DOSE**: the changed-pick share of the Base C pool per seed (the quantity A2 power depends on, EP-ADMISSION-R2-CONFIRM T5).

## 9. What these rates are NOT
* They answer "another draw of tokens and logs from a calibrated population" only, not another training run of P2 (one seed, `H-ESTIM-SEED-1`) and not another inference run beyond the 3 search seeds the gate itself quantifies over.
* Base C holds ONE NC-unsafe pick, 10-11 DAC-unsafe and 15 TTC-unsafe picks per seed in 400 tokens (printed by R-EVENTS): its A2 rates rest on that handful of events, duplicated by resampling. Base X holds the real S0 event structure but a synthetic candidate. Calibrated means D_M = 0 and observed / predicted = 1 at the picks on average; a candidate calibrated on average but wrong in a log-concentrated way is not simulated, and a planted shortfall is spread over all picks (the EASY case for detection).
* P2 is in-sample on S0 (biased toward OVER for the realised side) and carve B is SEEN-SCENE; the search's real behaviour on S0 tokens (its moved share, its output labels) is UNKNOWN until the bank exists. Tier: SELECTION, label lookup, T0-class; nothing here is a navtest or driving claim. STRATEGIC PAUSED.
