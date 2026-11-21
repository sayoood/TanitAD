# RESULT PRIVGEOM-SAFE: the information the DAC row misses exists upstream (DAC INFO-UPSTREAM, primary and companion); the NC row is INCONCLUSIVE and underpowered

Research Lab agent for the Master Mind, 2026-10-19. Prereg `PREREG_PRIVGEOM_SAFE.md` sha256 `c6d6e1fb228f7e4806d5b0316699f8fef05855684d0c535bd246248217bd7084` (sealed 10:39:42) + Amendment 1 `7382a2d7c345db05afaed9b1e2084975d943c6d605d568ce5d78fc7e1ed306c9` (sealed 10:54:07, before the first label read). Every call below is the registered rule applied to the registered fits; nothing in sections 1-5 was changed after a number was seen.

- Evidence class: **MEASURED**, SELECTION tier, T0-class label lookup, SEEN scenes (`D-L1-G0-1`), P2 IN-SAMPLE (every ARM is out-of-fold). No capability claim. Not a model eval: the four-family rule is not engaged (no lateral, longitudinal, tactical or strategic claim is made); the pick-level PDMS is a label lookup.
- ⛔ **DIAGNOSTIC ONLY. Privileged geometry is never an inference input** (`CLAUDE.md` rule of 2026-08-03). Nothing here is deployable.
- Estimator: paired LOG-cluster bootstrap over the **41 logs**, B = 10,000, seed 20260927. It answers "would another draw of LOGS say this?" and nothing about another training run (`H-ESTIM-SEED-1`). The two training seeds differ only in batch order; they agree to 0.0007 on every Delta_F, which is a floor on training variance, not an estimate of it.
- Data: 4,941 sets (0 lost in extraction), 41 of the 117 f10 logs (those with a local nuPlan DB), 316,224 proposals. The tier labels follow `Project Steering/EVAL_DOCTRINE.md`.

## 1. Headline (per row: final call, companion, the registered reading, the committed next arm)

| row | primary Delta_F = AUC_F(L+G) - AUC_F(L), seed 0 / 1 | final call | companion (§3.4) Delta_F, seed 0 / 1 | companion call | registered reading (§3.4) |
|---|---|---|---|---|---|
| **DAC** | **+0.1610 [+0.1221, +0.1984] / +0.1609 [+0.1220, +0.1984]** | **INFO-UPSTREAM** | DAC-nosign **+0.0789 [+0.0562, +0.0984] / +0.0782 [+0.0561, +0.0972]** | **INFO-UPSTREAM** | primary AND companion INFO-UPSTREAM: *boundary PROXIMITY information beyond the contact event exists upstream*. |
| **NC** | +0.0211 [+0.0112, +0.0346] / +0.0211 [+0.0112, +0.0341] | **INCONCLUSIVE** | NC-floored +0.0103 [+0.0040, +0.0186] / +0.0102 [+0.0040, +0.0183] | INCONCLUSIVE | no INFO-UPSTREAM, so the companion rule does not engage. SH3 is **neither supported nor refuted** by this arm. |

**Committed next arms (prereg §9, unchanged):**
- **DAC -> SH5**: a scorer-branch geometry read from cached `visual_ctx`; a dev-box GPU registration. Escalated to the Master Mind (section 9). ⚠️ Read it beside section 5: the companion's jump is weaker evidence for *proximity* than the registered reading says.
- **NC -> INCONCLUSIVE: "named by the report; the 76 logs without a local DB are the first lever" (§9, §11).** Named here: (1) the 76 logs, blocked on the pod's nuPlan DBs (`fetch_navtrain_dbs.py`, a PI decision, outside this arm); (2) ESTIMATED: widening 41 -> 117 logs shrinks the half-width about 1/sqrt(n) (+0.0117 -> about +0.0069), so with the point at +0.021 the lower CI would sit near +0.014, still under the +0.02 bar and the upper CI (about +0.028) still over +0.01: **widening alone is unlikely to resolve NC unless the point moves**; (3) the design lever of section 5.3 (within-set centring), which narrows the NC interval by about half on these same arrays.

**NC cannot be called NO-JUMP by this instrument anyway:** C-POWER did not fire (section 2), so a NO-JUMP would have been converted to UNDERPOWERED (§6.3). The raw call is INCONCLUSIVE, so the conversion did not arise.

## 2. Controls (every one reported; each was built so it could fail)

| control | result |
|---|---|
| **Gate** (`run/gate_report.json`, token bound to prereg + Amendment 1 + code + subset + join) | **PASS**, 11:37 |
| **C-AGREE** (recomputed NAVSIM verdicts `rl6` clipped to [0,1] = stored `labels6`) | NC **1.000000** and DAC **1.000000** exact on 316,224 proposals each (bar 0.99). Geometry computed on the proposals and scenes the labels belong to. |
| **C-JOIN** | 4,941 / 4,941 sets retained, 0 lost (disclosure threshold 5 %); census `z6` = pickrule `z6` = EP-R1 keys on every retained set. |
| **C-REACH** (G := the row's label through the real fit and score code, real arrays) | F-AUC L+G: NC 1.0000 / 1.0000, DAC 0.999997 / 0.999997; G-only 1.0000 on all four (seeds 0 / 1). Bar 0.99. |
| **C-RED** (branch table + RED mutations), **C-ID-L**, **C-CONV**, mute-G mutation | `branch_table_failures` = []; `test_pg_geom.py` 19 tests OK; `test_pg_lib.py` 9 tests OK (synthetic: L+G 0.8810 vs oracle 0.8842 vs L 0.7737; lr 1e-3 for the G block FAILS C-REACH as designed). Re-run on the STAGED code: section 7. |
| **C-NULL** (5 within-set permutation replicates, both rows, both seeds) | **0 / 5 fire on both rows -> no VOID.** ⚠️ The permuted block's Delta_F is NEGATIVE, not about 0: NC **-0.0149 to -0.0176**, DAC **-0.0609 to -0.0654** (all 20 CIs wholly below 0). Explained in section 5.3 (descriptive): the null is not centred, so it is conservative, and its non-fire is weaker evidence than "no leak" would be. |
| **C-POWER** (weak plant, label + N(0,1)) | **DAC fires:** Delta_F +0.0389 [+0.0222, +0.0563] / +0.0391 [+0.0222, +0.0567]. **NC does NOT fire:** +0.0263 [+0.0117, +0.0425] / +0.0270 [+0.0123, +0.0433] (lower CI < +0.02). NC's F-AUC is already 0.880, so a planted signal of scalar-alone AUC about 0.76 leaves little room (a ceiling effect), and probably the section 5.3 offset eats part of it (HYPOTHESIS, not tested for the plant). |
| **Label-identity rates (§3.4, no gate)** | `P(dac_clear_min<0 given DAC=0)` = **1.0000**, `P(dac_clear_min>=0 given DAC=1)` = **1.0000**, `P(dac_frac_out>0 given DAC=0)` = 1.0000 (n DAC=0 = 53,221). `P(nc_min_dist=0 given NC=0)` = **0.8511**, `P(nc_min_dist>0 given NC=1)` = 0.9563 (n NC=0 = 17,705). |
| Fire list / void flags | NC and DAC: void_null False, void_inputs False. |

## 3. Per row, per seed (F = Q4, within-token AUC; full tables `raw/results/RESULT_tables.md`, `results.json`)

| row | seed | F-AUC L | F-AUC L+G | F-AUC G-only | Delta_F (L+G - L) | Delta Q1-3 | G-only - L (F) | companion Delta_F |
|---|---|---|---|---|---|---|---|---|
| NC | 0 | 0.8798 [0.8454, 0.9106] | 0.9008 [0.8694, 0.9256] | 0.6845 [0.6361, 0.7326] | +0.0211 [+0.0112, +0.0346] | +0.0055 [-0.0065, +0.0177] | -0.1952 [-0.2492, -0.1451] | +0.0103 [+0.0040, +0.0186] |
| NC | 1 | 0.8794 [0.8449, 0.9102] | 0.9004 [0.8686, 0.9256] | 0.6837 [0.6345, 0.7326] | +0.0211 [+0.0112, +0.0341] | +0.0052 [-0.0067, +0.0175] | -0.1957 [-0.2509, -0.1447] | +0.0102 [+0.0040, +0.0183] |
| DAC | 0 | 0.8322 [0.7929, 0.8722] | 0.9932 [0.9876, 0.9966] | 0.9994 [0.9989, 0.9998] | +0.1610 [+0.1221, +0.1984] | +0.0682 [+0.0433, +0.0972] | +0.1672 [+0.1273, +0.2064] | +0.0789 [+0.0562, +0.0984] |
| DAC | 1 | 0.8323 [0.7928, 0.8721] | 0.9932 [0.9876, 0.9966] | 0.9994 [0.9989, 0.9998] | +0.1609 [+0.1220, +0.1984] | +0.0682 [+0.0433, +0.0971] | +0.1671 [+0.1273, +0.2065] | +0.0782 [+0.0561, +0.0972] |

Pair counts (seed-independent): NC F 189,584, Q1-3 123,229; DAC F 332,614, Q1-3 929,740.

Reads of the table (differences of point estimates, no interval):
- **DAC fast-band deficit.** L: F 0.8322 vs Q1-3 0.9274 = -0.095 on this subset (the census had -0.0718). L+G primary: 0.9932 vs 0.9957 = -0.003 (closed). The companion L+G(V): 0.9111 vs 0.9561 = -0.045 (about half).
- **G-only reaches 0.9994 on DAC** and the primary L+G 0.9932. That is the §3.4 sufficiency result: the block contains the verdict's own test.
- **NC: G-only is far BELOW L** (0.684 vs 0.880) and L has no fast-band deficit (F 0.8798 vs Q1-3 0.8795; census +0.0323). The four NC scalars add +0.021 to a head that already ranks well.
- **Spread over seeds** is 0.0000 to 0.0007 on every Delta_F.

## 4. The pick-level read (DESCRIPTIVE, no authority over any branch; label lookup, SEEN, P2 in-sample)

Seed 0 shown (seed 1 within 0.0010 on every rate and 0.08 on PDMS x100; full tables in `RESULT_tables.md`). Rates are per set (4,941 picks); EP = P2's `z6` with the EP column replaced by EP-R1's c* OOF column; picks by the pinned `pickrule_lib.rule_pick`.

| composite | NC-unsafe pick | DAC-unsafe pick | unsafe (EP-R1) | PDMS x100 | dPDMS x100 vs P2 |
|---|---|---|---|---|---|
| P2 | 0.0067 [0.0035, 0.0103] | 0.0358 [0.0209, 0.0512] | 0.0427 [0.0282, 0.0579] | 88.861 | - |
| EP (EP-led) | 0.0111 | 0.0480 | 0.0595 | 87.560 | -1.302 [-1.998, -0.602] |
| EP[L] | 0.0107 | 0.0453 | 0.0563 | 87.800 | -1.061 [-1.791, -0.379] |
| **EP[L+G]** | 0.0083 [0.0036, 0.0134] | **0.0099** [0.0057, 0.0140] | **0.0190** [0.0122, 0.0259] | **91.749** | **+2.888** [+1.243, +4.595] |
| EP[L+G, NC row only] | 0.0097 | 0.0468 | 0.0571 | 87.818 | -1.043 [-1.700, -0.371] |
| EP[L+G, DAC row only] | 0.0107 | 0.0099 | 0.0217 | 91.443 | +2.582 [+0.945, +4.311] |
| EP[G-only] | 0.0101 | 0.0075 | 0.0184 | 91.714 | +2.852 [+1.022, +4.656] |

- EP[L+G] minus EP: unsafe(EP-R1) **-0.0405 [-0.0582, -0.0240]**, dPDMS x100 **+4.189 [+2.526, +5.947]**; minus EP[L]: d_dac0 -0.0354 [-0.0510, -0.0205], d_nc0 -0.0024 [-0.0054, +0.0000].
- **What it shows, no more:** feeding the privileged DAC geometry to the DAC row removes the DAC-unsafe pick excess entirely (0.0099 vs P2 0.0358). It is the §3.4 sufficiency result at pick level (the contact test is inside G), and it matches the prereg forecast. **Feeding the NC geometry to the NC row does NOT recover the NC-unsafe pick rate** (EP[L+G, NC only] 0.0097 vs P2 0.0067; vs P2 d_nc0 +0.0030 [-0.0004, +0.0069]): consistent with NC INCONCLUSIVE and with no NC lever from this geometry.

## 5. DESCRIPTIVE (post-hoc, no authority over any branch): what carries the DAC companion's jump, and three things the analysis exposed

All of section 5 ran AFTER the result, on seed 0, with the registered recipe, folds and bootstrap draws. Scripts: `code/pg_descr_columns.py`, `pg_descr_dac_signleak.py`, `pg_descr_null_centering.py`, `pg_descr_nc_identity.py`. Outputs: `raw/descr/`. They decided nothing and are not in the prereg's decision rule.

### 5.1 The requested table: which DAC-nosign columns carry the jump (F stratum, seed 0)

Reference: L 0.8322; L+G(V), all 10 columns 0.9111, Delta_F **+0.0789 [+0.0562, +0.0984]**. Columns = `abs_clear_min`, then `abs_clear_t00` ... `abs_clear_t40` (the clearance at step t, 0.1 s steps).

| column | std. L+G(V) coefficient u (mean +- sd over 5 folds) | drop-one: Delta_F of the 9-col arm vs L | **drop effect** = AUC(all 10) - AUC(without j) | single-column: Delta_F of L + col j | raw within-set AUC of the column alone (larger = safer) |
|---|---|---|---|---|---|
| `abs_clear_min` | **-1.783 +- 0.163** | +0.0276 [+0.0130, +0.0430] | **+0.0512 [+0.0368, +0.0637]** | +0.0041 [+0.0021, +0.0069] | 0.406 |
| `abs_clear_t00` | -0.008 +- 0.095 | +0.0789 | -0.0000 | -0.0002 | 0.500 |
| `abs_clear_t05` | -0.055 +- 0.040 | +0.0789 | -0.0000 | -0.0001 | 0.501 |
| `abs_clear_t10` | -0.126 +- 0.057 | +0.0789 | -0.0001 | -0.0000 | 0.561 |
| `abs_clear_t15` | -0.077 +- 0.041 | +0.0791 | -0.0003 | +0.0012 | 0.614 |
| `abs_clear_t20` | +0.161 +- 0.060 | +0.0787 | +0.0001 | +0.0050 [+0.0005, +0.0092] | 0.683 |
| `abs_clear_t25` | +0.639 +- 0.040 | +0.0817 | -0.0028 | +0.0133 [+0.0022, +0.0274] | 0.740 |
| `abs_clear_t30` | **+1.116 +- 0.097** | +0.0759 | +0.0029 [-0.0007, +0.0078] | +0.0234 [+0.0098, +0.0385] | 0.768 |
| `abs_clear_t35` | **+1.420 +- 0.144** | +0.0624 [+0.0419, +0.0816] | **+0.0165 [+0.0107, +0.0222]** | +0.0234 [+0.0102, +0.0372] | 0.739 |
| `abs_clear_t40` | +0.521 +- 0.098 | +0.0783 | +0.0006 | -0.0118 [-0.0248, +0.0025] | 0.593 |

**Reading:** the jump is carried by **`abs_clear_min` against the late-horizon clearances `abs_clear_t25` ... `abs_clear_t35`**: opposite-signed weights (-1.78 against +0.64, +1.12, +1.42), the largest drop effects (abs_clear_min +0.051, t35 +0.017), and a **non-additive** structure (abs_clear_min alone adds only +0.004; no single abs_clear_t adds more than +0.023; together +0.079). The early steps (t00-t15) add nothing: t00 is the start clearance, the same for every proposal in a set. Full tables with every CI: `raw/descr/DESCR_columns.md`.

### 5.2 ⚠️ The companion still re-derives the verdict's sign (measured, with the logic)

Logic, not data: `c_min` is the minimum over ALL steps, so `c_t >= c_min` at every key step. If `c_min >= 0` (DAC-safe) then `abs(c_t) >= abs(c_min)` at every key step. **So any key step with `abs(c_t) < abs(c_min)` PROVES `c_min < 0`, i.e. DAC = 0.** The pair (`abs(c_min)`, `abs(c_t)`) carries the sign through its contrast, and a linear fit with opposite-signed weights on exactly these columns (5.1) can use it. Measured on the real arrays (`raw/descr/dac_signleak.json`):
- precision of that rule **1.000** (47,206 proposals fire it; **0** are DAC-safe);
- recall **88.7 %** of all DAC-unsafe proposals (89.4 % in F);
- the raw, no-fit within-set F-AUC of the contrast `min_k abs(c_t) - abs(c_min)` is **0.9530 [0.9460, 0.9594]**, above L (0.832) and above the companion arm (0.911).

⇒ **The registered DAC-nosign companion is not sign-free JOINTLY, only column by column** (§3.4 said "no column is a monotone function of the label", true per column). A companion INFO-UPSTREAM therefore does not isolate boundary proximity from the verdict. This is a design gap in my own prereg, found after the result; the registered call and committed reading stand as written. What a sign-clean read shows, descriptively:
- the 9 key-step `abs(c_t)` without `abs(c_min)` (`drop0`): **+0.0276 [+0.0130, +0.0430]**; centred (5.3): **+0.0352 [+0.0194, +0.0495]**. Both sit at or just under the +0.02 bar (INCONCLUSIVE if the registered rule were applied; it was not registered);
- a single late column, `abs_clear_t30` or `abs_clear_t35`: +0.0234.

⇒ **ESTIMATE of the proximity-specific headroom for a vision-reachable read: about +0.03 F-AUC (+0.028 uncentred, +0.035 centred; linear read of privileged geometry), against the +0.16 of the verdict-bearing block.** That is the number SH5 should be sized against. The companion's +0.079 is the upper bound of "proximity plus sign re-derivation".

### 5.3 The registered C-NULL is not centred (its non-fire is conservative), and why

The permuted block's Delta_F is about -0.017 (NC) and -0.062 (DAC), not 0 (section 2). HYPOTHESIS (stated before the run): within-set permutation keeps each set's G distribution, so the G block learns the BETWEEN-set association and applies it per proposal as within-set noise. Test: C-NULL rep 0, then subtract each set's mean from every column, refit, seed 0 (`pg_descr_null_centering.py`):

| | registered null rep 0 | within-set-centred null rep 0 |
|---|---|---|
| NC | -0.0171 [-0.0267, -0.0095] | **-0.0006 [-0.0015, +0.0001]** |
| DAC | -0.0612 [-0.0772, -0.0417] | **-0.0015 [-0.0021, -0.0009]** |

Supported (MEASURED, one replicate, one seed). Centred real arms, seed 0 (Delta_F vs L, registered arm beside it):

| arm | registered (uncentred) | **within-set centred** |
|---|---|---|
| DAC primary (11 cols) | +0.1610 [+0.1221, +0.1984] | +0.1623 [+0.1231, +0.2008] |
| DAC-nosign (10 cols) | +0.0789 [+0.0562, +0.0984] | +0.0803 [+0.0556, +0.1005] |
| DAC-nosign without `abs_clear_min` (9 cols) | +0.0276 [+0.0130, +0.0430] | +0.0352 [+0.0194, +0.0495] |
| NC primary (4 cols) | +0.0211 [+0.0112, +0.0346] | **+0.0136 [+0.0082, +0.0190]** |
| NC-floored | +0.0103 [+0.0040, +0.0186] | **-0.0000 [-0.0065, +0.0056]** |

- **DAC is invariant to centring** (so the DAC call does not depend on the null's offset).
- **NC moves:** centred, the primary reads +0.014 with an upper CI of +0.019 (under the +0.02 bar) and the sign-clean companion reads 0.000 [-0.007, +0.006] (under the +0.01 bar). The centred primary interval is about half as wide (0.011 vs 0.023). **Descriptively, NC leans to "the contact event is all there is" and away from SH3**; this is one seed, post-hoc, and does not change the registered INCONCLUSIVE.
- Consequence for instrument design: a centred G block and a centred null belong in any PRIVGEOM-type arm; the registered null could not have flagged a positive bias smaller than its own negative offset (about 0.017 NC / 0.062 DAC).

### 5.4 The NC scalar is half a tautology

`nc_min_dist == 0` holds for 85.1 % of NC = 0 proposals (5.5 % lie in (0, 0.5] m, 7.2 % in (0.5, 2] m, 2.3 % beyond; median of the rest 0.73 m) and for 4.4 % of NC = 1 and 88.3 % of NC = 0.5 proposals (`raw/descr/nc_identity.json`). From those rates (ESTIMATED arithmetic) about 30,150 proposals are in contact, of which only about 50 % are NC = 0: contact is **necessary for most NC verdicts but far from sufficient**; the at-fault type decides. So the NC primary is not the DAC-style identity.

## 6. Limits (stated in advance, and the ones the data added)

- **41 of 117 logs (31.3 % of sets)**, a filter on file presence, not a random sample; what it licenses is a statement about these 41 logs. The 76 other logs' DBs live only on the pod (`fetch_navtrain_dbs.py`), a PI decision outside this arm.
- SELECTION tier, SEEN scenes, P2 in-sample (a bias AGAINST every out-of-fold composite). One rig, two refit seeds; intervals answer "another draw of LOGS" only; with 41 logs and 3 logs in one fold they are wider than the census's.
- **Sufficiency (§3.4):** the primary DAC block contains the label's own test (identity rates exactly 1.0000), so the primary INFO-UPSTREAM shows the verdict closes the label, not that a vision-reachable signal exists. The companion separates the two readings **only partly** (5.2). A linear row on standardised scalars cannot reproduce a sign test exactly; the planted controls bound that.
- "Upstream" means upstream of the **256-d token under a linear read** (`D-REFE-HEADROW-PROBE-1`: a 2-layer head is no better). It does not say the information is absent from `visual_ctx` or from the images; that is what SH5 tests.
- The lead is a straight-corridor approximation; agent velocity is a finite difference of an interpolated track; the geometry is on NAVSIM's simulated (LQR + bicycle) states, not the raw plan.
- C-NULL is not centred (5.3); the NC row is below the instrument's resolution (C-POWER).
- Section 5 is post-hoc, seed 0, single replicate where a replicate is involved.

## 7. Package and reproducibility

- Stage: `C:/Users/Admin/par18/privgeom/out/` + `paths.json` (`build_package.py`; stage, never push). The large payloads (scalars 27 MB, OOF logits, join arrays; about 150 MB) are on `D:/Projects/TanitAD-artifacts/privgeom_2026-10-19/` with sha256 in `raw/ARTIFACTS_D.json` (re-hashed after copy).
- **Tests re-run on the STAGED code in a deploy layout** (`code/run_deploy_tests.py`: a copy of the staged `code/` beside the landed `ref/` files, sha256-checked against `code/REFS.json`; record `raw/deploy_tests.json`, which also carries the sha256 of every tested file): `test_pg_geom.py` **19 / 19 OK** (rc 0), `test_pg_lib.py` **9 / 9 OK** (rc 0, 81 s). CPU only.
- Pins unchanged: all seven CODE_FILES re-hash to the gate token's `code_pins` (every post-gate stage checks it). The new `pg_descr_*.py`, `build_package.py` and `run_deploy_tests.py` are NOT pinned and decided nothing.

## 8. Forecast (prereg §10) against the outcome

| prediction (committed before any number) | outcome |
|---|---|
| DAC primary INFO-UPSTREAM about 0.95 | hit |
| DAC-nosign INFO-UPSTREAM 0.45 / INCONCLUSIVE 0.35 / NO-JUMP 0.20 | INFO-UPSTREAM (the 0.45 bin) |
| NC primary INFO-UPSTREAM 0.65 / INCONCLUSIVE 0.25 / NO-JUMP 0.10 | **INCONCLUSIVE (the 0.25 bin)** |
| NC-floored INFO 0.30 / INCONCLUSIVE 0.40 / NO-JUMP 0.30 | INCONCLUSIVE (the 0.40 bin) |
| pick level: R-EP[L+G] cuts the unsafe-pick rate below P2's | hit for DAC (unsafe(EP-R1) 0.0190 vs P2 0.0427); **not for NC-only** (0.0097 vs 0.0067) |

## 9. Escalations and the next lever (Rule Zero: a result is not a stopping point)

1. **Master Mind decision: register SH5** (dev-box GPU). Suggested design points from section 5, all HYPOTHESIS-level: size the expected gain against the sign-clean headroom (about +0.03 F-AUC), not +0.16; make the supervised read the late-horizon UNSIGNED clearance profile (steps 25-35), not `abs(c_min)`; gate SH5 on a sign-clean comparison, because the verdict-bearing pair re-derives the label.
2. **PROPOSED, not registered, CPU-only, about 15 s per fit: PRIVGEOM-CLEAN.** Within-set-centred G and a centred null; DAC columns = the 9 key-step `abs(c_t)` (no `abs(c_min)`); NC columns = floored; same bars, seeds, bootstrap. Outcomes to commit in advance: DAC-clean INFO-UPSTREAM -> SH5 is supported on proximity; INCONCLUSIVE -> SH5 stays registered but unsupported by this arm; NO-JUMP -> proximity headroom is absent and SH5 loses its premise. NC-clean INFO-UPSTREAM -> SH3; NO-JUMP -> NC has no lever in this geometry. ⚠️ Run on these 41 logs it is SEEN twice (I have seen 5.2 and 5.3); **the clean test is the 76 logs**, which is the PI decision on the pod DBs.
3. **PI decision (blocks one item only):** fetch the 76 logs' nuPlan DBs for a widening run. ESTIMATED cost: linear scaling of the 1.78 CPU-h projected for the 41 logs' 4,941 sets gives about 3.9 CPU-h for the 76 logs' 10,867 sets.
4. Free C: before any widening run: it had 0.46 GB free when checked during this run, and `run/join/feat_sub.npy` (162 MB) and `run_smoke/` (16 MB) are re-derivable.

## 10. PROPOSED register text (NOT applied; the registers are not edited by this agent)

**`GOALS_AND_CLAIMS.md`, new row:**

| **D-REFE-PRIVGEOM-SAFE-1** (2026-10-19) | **PRIVGEOM-SAFE reads DAC = INFO-UPSTREAM (primary AND the DAC-nosign companion, both seeds) and NC = INCONCLUSIVE (C-POWER did not fire). The DAC primary is a SUFFICIENCY result by construction (the clearance sign IS the verdict); the companion's jump rides on a column contrast that still re-derives the sign (DESCRIPTIVE).** PREREG sha256 `c6d6e1fb…` + Amendment 1 `7382a2d7…` (sealed before the first label read). 41 of 117 f10 logs (those with a local nuPlan DB), 4,941 sets, 0 lost. Gate PASS; C-AGREE exact on 316,224 proposals; C-NULL 0/5; C-REACH about 1.0. Delta_F = AUC_F(L+G) - AUC_F(L), seeds 0/1: **DAC +0.1610 [+0.1221, +0.1984] / +0.1609; companion +0.0789 [+0.0562, +0.0984] / +0.0782**; **NC +0.0211 [+0.0112, +0.0346] / +0.0211; companion +0.0103 [+0.0040, +0.0186]** (INCONCLUSIVE; C-POWER +0.0263 [+0.0117, +0.0425], no fire). Pick level (descriptive): EP[L+G] DAC-unsafe pick 0.0099 vs P2 0.0358; NC-only 0.0097 vs P2 0.0067. Descriptive findings: (i) the companion's jump is the `abs_clear_min` vs late `abs_clear_t25..t35` contrast (88.7 % of DAC-unsafe proposals have a key step with `abs(c_t) < abs(c_min)`, which proves the sign); the sign-clean 9-column read is +0.0276 [+0.0130, +0.0430]; (ii) the registered null is not centred (-0.017 NC / -0.062 DAC; centred -0.0006 / -0.0015). Committed next arm: SH5 (DAC); NC: the 76 logs. Paired log-cluster bootstrap 10,000 / seed 20260927 (another draw of LOGS only). `…/raw/2026-10-19-privgeom-safe/RESULT.md` | MEASURED (selection tier) |

**`H-REFE-FASTNEG-1` / `D-REFE-FASTNEG-CENSUS-1` follow-up line:** "PRIVGEOM-SAFE (D-REFE-PRIVGEOM-SAFE-1): the DAC fast-band deficit is closed by the verdict-bearing geometry (F-AUC 0.832 -> 0.993) and halved by the sign-free companion (0.911); the sign-clean proximity headroom is about +0.03. NC: INCONCLUSIVE."

**`H-REFE-RL1-NCRES`:** append "PRIVGEOM-SAFE: privileged NC geometry (4 scalars) adds only +0.021 [+0.011, +0.035] to the NC row and does not recover the EP-led pick's NC-unsafe rate (0.0097 vs P2 0.0067). Not supported as a geometry-resolution lever; INCONCLUSIVE at 41 logs."

**`BACKLOG.md`:** (1) SH5 registration (dev-box GPU), sized to the sign-clean headroom; (2) PRIVGEOM-CLEAN pre-registration (CPU, centred G and null); (3) PRIVGEOM widening to the 76 logs, blocked on the pod DBs (PI); (4) instrument: a within-set-centred null becomes the default for any PRIVGEOM-type arm.

**`RETRACTION_LOG.md` (a limitation of a registered control, not a retracted number), root-cause class:** "A control that removes a leak column by column can leave it recoverable JOINTLY: DAC-nosign kept `abs(c_min)` beside `abs(c_t)`, and a min over steps bounds every per-step value, so the pair re-derives the sign. Ask of every companion: could any COMBINATION of the kept columns reproduce the removed signal?"

## Deliverable manifest

| artifact | where |
|---|---|
| this RESULT.md, prereg, Amendment 1, sha256 files, SEALED_AT, code (extract, geom, lib, run, tests, descriptive scripts, package scripts), cost, subset, gate report + token, join report, fit JSONs, results (`results.json`, `RESULT_tables.md`), descriptive outputs, logs, `deploy_tests.json`, `ARTIFACTS_D.json`, `MANIFEST.md` | staged: `C:/Users/Admin/par18/privgeom/out/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-privgeom-safe/` (+ `out/paths.json`); sources in `C:/Users/Admin/par18/privgeom/` |
| 41 per-log scalar stores, OOF logit arrays (primary + descriptive), join arrays | **D: only (+ the working copy on C:)**: `D:/Projects/TanitAD-artifacts/privgeom_2026-10-19/` |
| nothing lives on a pod or in a worktree | - |
