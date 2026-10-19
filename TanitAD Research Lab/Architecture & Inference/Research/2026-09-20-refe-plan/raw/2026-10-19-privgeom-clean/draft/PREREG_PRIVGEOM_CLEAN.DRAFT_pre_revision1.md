# PREREG PRIVGEOM-CLEAN: does privileged boundary PROXIMITY carry DAC information BEYOND the verdict's own sign, on 76 logs PRIVGEOM-SAFE never read? (Research Lab agent for the Master Mind, 2026-10-19)

**STATUS: DRAFT. NOT SEALED.** The Master Mind rules on section 14, then seals (`sha256sum` of this file into `PREREG_PRIVGEOM_CLEAN.sha256` + `SEALED_AT.txt`). **No number on the 76 logs has been computed** (disclosures, section 0). Every stage refuses on a hash mismatch; a change after sealing comes only by a written amendment (`PREREG_PRIVGEOM_CLEAN_AMENDMENT<k>.md`) saved and hashed before the number it governs. The gate (the first stage that opens a label) runs only after the seal.

- Tier: SELECTION, T0-class label lookup (S0 fit sets, 64 proposals each; no closed loop). P2 IN-SAMPLE (its frozen rows are in-sample; every ARM below is out-of-fold). 0 GPU, CPU only. No navtest, no dev table.
- ⛔ **DIAGNOSTIC ONLY. Privileged geometry is NEVER an inference input** (inference is vision-only, `CLAUDE.md` binding rule of 2026-08-03). Nothing here is deployable. No capability claim.
- Source of the question: `raw/2026-10-19-privgeom-safe/MM_RULING_PRIVGEOM.md` (binding) and `RESULT.md` §5; `RETRACTION_LOG.md` **R54** (removing a sign is not removing the information; a companion leaked the verdict through column ORDER). SH5's `PREREG_SH5_DAC_PROX*.md` (Amendments 2 and 3) is the precedent for the call rule and the VOID rule.

## 0. Disclosures (what I had seen, and what I computed, before this draft)

1. I read PRIVGEOM-SAFE's whole package (RESULT incl. §5, MM ruling, prereg + Amendment 1, `DESCR_columns.md`, the code) and R54. I know its descriptive sign-clean readings: 9-column arm **+0.0276 [+0.0130, +0.0430]** uncentred, **+0.0352 [+0.0194, +0.0495]** within-set-centred (seed 0, the 41 seen logs). I know the descriptive column table (the jump rides on `abs_clear_t25..t35`).
2. Banked SEEN data (the 41 logs, labels already opened by that arm) was used for exactly THREE design calculations, none an outcome on the 76: (i) the SE / MDE model (section 8); (ii) the C-POWER plant calibration (section 7); (iii) the admissibility check's rehearsal (section 4). The SEEN arrays were also re-fitted once through the new runner as a rehearsal / identity (section 9). **The 41 SEEN logs are excluded from the read** (forking-path rule of the MM ruling).
3. On the 76: **label-free** coverage only (set / token / stratum / DB-size counts, from the store keys and the census `code`, which is speed-derived and label-free by its own proof). I opened **NO** `labels6` array, **no** stored label of any kind.
4. **G-LIVE on 2 of the 76 logs** (the frozen extraction in memory): the NAVSIM scorer computes the recomputed verdicts internally as a by-product of the geometry call (the geometry is read from its state); on those 2 logs they were dropped in memory without being read, printed, stored or compared. Persisted: the sha256 of the 9 sign-clean columns, finiteness / cap / variation checks, timings. No signed column, no `|c_min|`, no NC column, no verdict was persisted or inspected. (Same disclosure form as PRIVGEOM-SAFE §0.3.)
5. The two G-LIVE logs' block hashes are registered here so the read's full extraction can be checked against them (section 7, C-ID).

## 1. Question

`MM_RULING_PRIVGEOM.md` §2: the registered PRIVGEOM-SAFE DAC companion's +0.079 re-derived the verdict through the order of its unsigned columns; the sign-clean 9-column read is +0.028, post-hoc and descriptive. **Question: on 76 logs this arm did not see, does a block of privileged boundary-PROXIMITY columns, certified by a PLANTED admissibility check not to prove the DAC verdict, still lift the deployed DAC row's fast-stratum within-token AUC (a held-out confirmation of the descriptive +0.03)?** DAC row only. The recipe, estimator, strata and folds are PRIVGEOM-SAFE's, FIXED as written (section 5); the only additions are the sign-clean block, within-set centring (a preprocessing step with no free parameter, section 3), and the admissibility gate (section 4).

**Not asked here:** the NC row (its own prereg; the same extraction carries the NC columns at no extra DB cost, they are not read by any stage of this arm); whether a VISION read can reach the proximity (that is SH5); any pick-level read.

## 2. Data and subset (every fact below was fixed before any label was read; MEASURED, `run/coverage.json`)

- S0's 15,808 fit sets at f = 10 % in 117 logs (census `meta.npz`, sha256 `060af1a3f8f96f525056a37c7f6938050b3a0fbbb08e7b69bacbb85b43031675`; per-log stores `C:/Users/Admin/srttc_s0/work/sets/<log>.npz`; tokens `feat_f10.npy` sha256 `f96ef18616ce37be5149ffc32a54046f6b6be5bf0bc3c8405f19bb20770c1af3`).
- **Resolution:** all **117 / 117** f10 logs now resolve to a local nuPlan DB through the landed `index_dbs()` (`D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits`): the 41 PRIVGEOM-SAFE read (its `run/subset.json`, sha256 `5574b57fae127e141788dcc001399e9e323348a8d4f203dc35c2ca3d0de61a25`) + **76 unseen** (overlap 0). The 76 DBs total **11,324,555,264 B**, equal to the pod-stage manifest.
- **The read set = the 76 unseen logs, ALL of them, whole logs, no cost-based subsetting** (PRIVGEOM-SAFE's 16 CPU-h cap does not bind: projected cost section 12).

| | the 41 SEEN (excluded) | the 76 UNSEEN (read) |
|---|---|---|
| sets / tokens | 4,941 / 2,782 | **10,867 / 6,140** |
| proposals | 316,224 | **695,488** |
| proposals per stratum S / Q1 / Q2 / Q3 / Q4 (census cuts 0.9076 / 1.0453 / 1.2228) | 3,392 / 66,772 / 84,237 / 84,538 / 77,285 | **9,088 / 183,036 / 165,571 / 165,270 / 172,523** |
| F (= Q4) share of proposals | 24.4 % | **24.8 %** |
| logs per census fold 0-4 | 10 / 8 / 8 / 3 / 12 | **14 / 16 / 15 / 20 / 11** |
| sets per census fold 0-4 | 1,030 / 661 / 1,368 / 179 / 1,703 | **2,038 / 3,353 / 2,244 / 2,326 / 906** |
| sets per log min / median / max | 4 / 99 / 443 | **2 / 84 / 1,495** |

⚠️ **One log holds 1,495 sets (13.8 % of the 76's sets, 12.2 % of its F proposals).** It will carry a large share of the F pairs and the log-cluster interval will be wide for it (section 8: widening 41 → 76 logs buys little precision).
- Folds = the census fold of each log (`rank mod 5`), refit **within the 76** (train on the other four folds' logs, score the held-out one: OUT-OF-FOLD logits for every set). L, L+G and G-only are all refit on the same training sets. Nothing trains on the 41 seen logs.
- Execution: the FROZEN `pg_extract.py work` on `run/subset_clean.json`, at most 2 Python processes, psutil available >= 4 GB to start and >= 3 GB while running (scripted), `CUDA_VISIBLE_DEVICES=-1`, DBs read-only.

## 3. The block: sign-clean columns only, and within-set centring

- **The block = the 9 key-step |signed clearance| columns** `abs_clear_t00, t05, ..., t40` = `|g[..., 2:11]|` of PRIVGEOM-SAFE's scalar layout (clearance of the 4-corner ego footprint to the drivable-area union boundary at steps 0, 5, ..., 40 on NAVSIM's simulated states, capped at 10 m; `pg_geom.py`, pinned). It is PRIVGEOM-SAFE's `drop0` arm.
- **EXCLUDED, and unreachable by construction:** `|dac_clear_min|` (column 0; with `|c_t|` it proves `c_min < 0` through `|c_t| < |c_min|`, R54), `dac_frac_out` (column 1; the verdict), every SIGNED clearance, every NC column. `pgc_lib.clean_block` indexes columns 2..10 only; `test_pgc_lib.py` mutates columns 0, 1 and 11-14 and requires the block not to move, with a control that moving a key column DOES move it, and `test_pgc_admit.py` requires a mutated block that reads column 0 to be REFUSED.
- ⛔ **Not the same as admissible:** a block that excludes the sign column can still re-derive the verdict through the ORDER of the columns it keeps. Which is why section 4 TESTS it, by recomputation, on the data.
- **PRIMARY arm (`LGc`) = the block centred within set** (each set's mean over its 64 proposals subtracted from every column, then the registered fold-wise standardisation of `pg_lib.standardise`). Centring removes the BETWEEN-set association and nothing else (PRIVGEOM-SAFE §5.3: the registered permutation null sat at -0.017 NC / -0.062 DAC because the G block learned set-level association and applied it per proposal; centred it reads -0.0006 / -0.0015). It has no free parameter. `abs_clear_t00` is the start clearance, shared by the 64 proposals of a set (measured by G-LIVE on both logs), so it is exactly 0 after centring (the registered standardisation sets a zero-variance column to 0): the primary block is effectively 8 columns, t05..t40. (Measured constant within every set on both G-LIVE logs; the uncentred companion keeps it as a set-level covariate.)
- **COMPANION arm (`LGu`) = the block UNcentred = PRIVGEOM-SAFE's `drop0` recipe exactly** (its +0.0276 on the 41). Same estimator, same rule, reported beside the primary, **no authority over the call**. ⛔ DECISION FOR THE MASTER MIND (section 14, DQ1): the brief says "recipe FIXED as written"; I read centring as preprocessing, not tuning, and made the centred arm primary because (i) the MM ruling centres every future permutation null, which only guards a centred arm, (ii) PRIVGEOM-SAFE RESULT §9.2 proposed it, (iii) it removes a measured negative offset from the quantity being confirmed. On the SEEN logs the two arms read +0.0352 and +0.0276 (a +0.0076 gap), so the choice matters little at the margin but it is yours. One constant, `PRIMARY_ARM` (`pgc_lib.py:21`, default `"LGc"`), selects which arm the call reads (`pgc_run.py` `analyze` and `write_tables` read it; the other arm becomes the companion); flipping it edits a pinned file, so it must be made BEFORE the seal and the pins table regenerated (`pgc_pins.py`). ⚠️ Scope of the flip: the C-NULL replicates and the C-POWER plant stay on the CENTRED block either way (the MM ruling centres every permutation null), so an uncentred primary would NOT be guarded by an uncentred null; if you choose `LGu`, say whether to add uncentred null/plant arms (not wired).
- **G-only (`Gc`)**: the centred block alone, no token; reported, no branch reads it.

## 4. THE PLANTED ADMISSIBILITY CHECK (A-ADMIT; R54: "Every leak-removal step I specify ships with a planted recompute-the-label check in the same prereg")

**What it asks:** *could the DAC verdict be RECOMPUTED from this block?*, answered by a SEARCH with PLANTED controls, **never by inspecting column definitions**. Code `pgc_admit.py` (pinned; its ancestor is SH5's `sh5_admit.py`, whose 792 rules are re-implemented here as families U, P, V, A, plus two combination families).

- **Recomputer families** (all one-sided: a rule that PROVES DAC-UNSAFE): **U** unary `C_a < theta | C_a > theta` (value leaks); **P** ordered pair `C_a + m < C_b`, m in {0, 0.25, 0.5, 1, 2} (order leaks, the R54 case); **V** V-dip `C_m + m < min(C_a, C_b)`; **A** aggregates; **T** a depth-5 histogram tree on [values, all pairwise differences, equally spaced second differences, min / max / mean], every node evaluated as a rule (nonlinear combinations); **LIN** an L2 logistic regression on the same features, flagged iff a prefix of its UNSAFE ranking is a proof (linear combinations). T / LIN fit on a fixed-seed subsample of at most 300,000 proposals; U / P / V / A on all of them.
- **NOT ADMISSIBLE iff** some rule / node / prefix has **precision >= 0.999 at recall >= 0.5 % of the UNSAFE class**, over all proposals. One-sided on purpose: "every key clearance is large, so the proposal is DAC-safe" is what proximity MEANS (a path that never nears the boundary cannot have left it) and is reported, not penalised.
- **Planted controls (run FIRST; the check is trusted only if all read right):**

| control | construction | must be |
|---|---|---|
| POS-REGISTERED | the registered PRIVGEOM-SAFE 10-column DAC-nosign block (`|c_min|` + the 9) | **flagged** (R54's case) |
| POS-EQUAL | the block + the verdict itself as a column | flagged |
| POS-VALUE | the block + `label x 5 + noise(0.5)` | flagged |
| POS-ORDER | the block + a column equal to `C_t30 -+ 0.05` by the verdict: no unary rule separates it, only its ORDER against `C_t30` does | flagged |
| POS-LIN | the block + a column equal to `0.7 C_1 + 0.4 C_4 - 0.9 C_7 +- 0.05` by the verdict: only a COMBINATION proves it | flagged |
| NEG-PERM | every block column permuted within each set | **not flagged** |
| NEG-NOISE | the block + a pure-noise column | not flagged |
| NEG-PROX | the block + `C_t30 + N(0, 1)`: informative about the label, not a function of the verdict (a clean PROXIMITY column) | not flagged |

- **RED ARM: disable the check => the planted leaks get through.** `AdmitConfig(disabled=True)` makes the search report "nothing proven": every POS plant is then ADMITTED, `controls_ok` is False, the gate FAILS. MEASURED on the synthetic suite (`test_pgc_admit.py`, 14 tests) AND on the real SEEN arrays (`raw/admit_seen.json`): disabled => **5 / 5** planted leaks (POS-REGISTERED, POS-EQUAL, POS-VALUE, POS-ORDER, POS-LIN) leak through. **Mutation table (each must break `controls_ok`):** disabled; precision bar 1.01; recall floor 2.0; sense inverted; no families. **All 5 break it** (the inverted-sense mutation breaks all 8 controls). **Family ablation (descriptive, defence in depth; a family alone):** U and A catch POS-EQUAL / POS-VALUE; P catches POS-REGISTERED and POS-ORDER; V catches POS-REGISTERED; T catches every POS plant except POS-LIN; **LIN catches all five and is the ONLY family that catches POS-LIN**; no family flags a NEG plant.
- **Where it runs: at the gate, BEFORE any fit, on the 76's REAL arrays** (it needs the labels, so it is the gate's first act after they open): the planted controls rebuilt from the 76's real block and real labels (POS-REGISTERED from the extraction's real `|c_min|`), the RED arm re-run, then the candidate. `C-ADMIT pass` = controls all read right AND RED arm leaks AND candidate not flagged. **A flagged candidate = NO-CALL(ADMIT): the gate FAILS, no fit starts**, and the report names the proof rule, its precision and recall (a named blocker, not a result).
- **MEASURED rehearsal on the 41 SEEN logs** (316,224 proposals, 53,221 unsafe; `raw/admit_seen.json`): all 5 POS flagged, all 3 NEG not flagged; **the candidate is ADMISSIBLE: best UNSAFE proof by family U 0.843 @ recall 0.154 (`|c_t30| < 0.1 m`), P 0.438 @ 0.017, V 0.884 @ 0.038 (`V(C5 + 0.5 < min(C4, C6))`), A 0.302 @ 0.040, T 0.886 @ 0.328, LIN 0.849 @ 0.192; nothing reaches 0.999** (the same V-dip 0.884 @ 3.8 % as SH5's A-TEST, an independent re-implementation). The registered 10-column block is flagged by P (43 rules at precision >= 0.999; best by recall `C8 + 0 < C0`, i.e. `|c_t35| < |c_min|`, precision 1.000 at recall **0.776**: SH5's 77.6 % again), V (107 rules), T (4) and LIN (1). The candidate DOES prove SAFE for some proposals (5 rules; proximity, reported). Admissibility is a statement about the SIGN path, not about information content: a flexible recomputer reaches F-AUC 0.900 from this block alone (SH5 §3.3). Cost: 131 s on 316,224 proposals; about 5 min on the 76's 695,488.
- ⛔ **This check guards EVERY privileged target** (SH5's 9-output regression head, any later PRIVGEOM-type arm): `pgc_admit.a_test` is the one function; a target block that is not ADMISSIBLE on its own data is not run.

## 5. Arms, recipe, folds, estimator (PRIVGEOM-SAFE §4-§5, FIXED as written; `pg_lib.py` pinned, imported unchanged)

- **L** (the 256-d token, HEADROW-PROBE linear recipe): `z = z_P2 + feat . (w - w_P2) + (b - b_P2)`, initialised at P2's DAC row, BCE-with-logits on the (soft) DAC label `labels6[..., 1]`, Adam lr 1e-3, wd 1e-4, betas (0.9, 0.999), eps 1e-8, 20 epochs, batches of 256 sets, `torch.randperm` from `Generator().manual_seed(seed x 1,000,003 + n_train_sets)`.
- **L+G** (the token + the standardised block): `z = z_L + g_std . u`, `u` zero-initialised, its own Adam group at **lr 1e-2** (the registered G rate; lr 1e-3 cannot move a zero-initialised block, PRIVGEOM-SAFE §4); `g_std` = columns minus the TRAINING sets' mean over proposals, over their std (std < 1e-8 -> 0), applied to the held-out fold with the training statistics.
- **Seeds 0 and 1** (they differ only in batch order). 5 folds = the census fold. **Within-token AUC** = the census definition: unsafe = label 0, safe = label 1 (0.5 excluded), pairs inside ONE set, both members in the stratum, score = the OOF logit, ties 0.5. **F = Q4.** Pair counts reported.
- **Estimator: the paired LOG-cluster bootstrap over the 76 logs, B = 10,000, `numpy.random.default_rng(20260927).integers(0, 76, (10000, 76))`, the SAME draws for every quantity, percentile 2.5 / 97.5, every ratio a ratio of per-log sums, every difference on the same resampled logs.** ⚠️ **Which question the interval answers: "would another draw of LOGS say this?" and nothing about another training run** (`H-ESTIM-SEED-1`); the two seeds are a floor on training variance, not an estimate of it (they agreed to 0.0007 in PRIVGEOM-SAFE). The replicate that reads the rig's own run-to-run noise is C-NULL (section 7): the block scrambled within set, same machinery, different permutation AND training seeds.
- **Decision quantity:** `Delta_F = AUC_F(L+G) - AUC_F(L)`, one interval per seed, primary arm.

## 6. Decision (committed now; one definition `pgc_lib.seed_call` used for the call, the machinery null and the plant)

Per seed: **CONFIRM** iff `Delta_F` lower CI > 0 **AND** point >= **+0.020** (SH5 Amendment 2's CTX-REACHABLE rule: the lower-CI conjunct is significance, the point conjunct is materiality). **NULL** iff `Delta_F` upper CI < **+0.020** (the interval lies wholly below the same materiality bar, so CONFIRM and NULL are mutually exclusive: point <= upper CI). Otherwise **INCONCLUSIVE**. **Row call: BOTH seeds must agree** (CONFIRM only if both CONFIRM; NULL only if both NULL; else INCONCLUSIVE). Descriptive sub-reading, no authority: **SMALL** = both seeds lower CI > 0 and point < +0.02 (a real but sub-bar held-out effect), reported on an INCONCLUSIVE or a NULL.

- ⚠️ I moved the NULL bar from SH5's NOT-REACHABLE `upper CI < +0.01` to `< +0.02`: with SE about 0.0073 an upper CI under +0.01 needs the point under -0.004, so the sealed-SH5-style bar would call NULL only 28 % of the time at a TRUE zero effect (section 8). The question being refuted is "proximity headroom of the CONFIRM size exists", whose complement is "the interval lies below the CONFIRM bar". Flag: MM ruling DQ2.
- **Conversions (they can only REMOVE a call; CONFIRM is never converted by the power control), in this precedence:** **NO-CALL(ADMIT)** (the block is REFUSED: no fit starts) > **VOID-INPUTS** (C-AGREE on the DAC column < 0.99, or C-JOIN fails) > **VOID** (any C-NULL replicate fires, section 7) > **UNDERPOWERED** (a NULL call whose planted effect did not fire CONFIRM on both seeds: an instrument that cannot see a planted effect of the committed size cannot call "no effect").
- **Committed next arm per outcome (Rule Zero: a refutation is a waypoint):**

| final call | what it says | next arm (committed now) | what blocks it, if anything |
|---|---|---|---|
| **CONFIRM** | held-out proximity information beyond the verdict's sign exists upstream of the token under a linear read, at least +0.02 | SH5 stays registered and its +0.03 sizing is re-based on the held-out point (an SH5 amendment, the MM's call, before any SH5 number); the NC-CLEAN prereg on the SAME extraction (no new DB cost); `pgc_admit.a_test` becomes the gate for every privileged target | nothing from this arm; SH5 FULL's ctx build waits on its own C-REACH fix (Amendment 4), independent of this arm |
| **NULL** | the +0.028 did not replicate at a LINEAR read: the held-out upper CI is under +0.02, so SH5's CTX-REACHABLE point bar (+0.02) would need a vision read that BEATS the privileged linear ceiling | SH5's sizing premise is withdrawn and SH5 is deprioritised (not logically closed: a nonlinear read of the same block is not bounded by a linear one). **First, cheap, CPU, 15 s per fit: CLEAN-NL**, the same 76 logs and block through a 2-layer head (its own prereg), which decides whether any nonlinear proximity headroom exists before a GPU-day is spent; then the label-learnability audit of the fast band and the trunk (P6), per PRIVGEOM-SAFE §9. The DAC fast-band deficit is then the verdict's sign (closed only by the +0.16 sign-bearing geometry), not proximity | CLEAN-NL: nothing (CPU). The audit needs a dev-box GPU slot; the trunk is P6-gated |
| **INCONCLUSIVE** | neither bar: the held-out interval straddles the bar | SH5 stays registered but UNSUPPORTED by this arm (the MM's own wording for SAFE); a DESCRIPTIVE inverse-variance combination with the seen 41 is reported (the 41 carry the post-hoc selection; no authority). First lever: PRECISION, which more seeds cannot buy (the interval answers logs). The only routes to more LOGS: an S0 build over more nuPlan logs (compute + corpus; the deferred f = 25 % build adds sets, not logs). Named blocker, not a stop | compute (an S0 build), a PI decision |
| **UNDERPOWERED / VOID / VOID-INPUTS / NO-CALL(ADMIT)** | the instrument or the block failed its own control | named by the report: which control, what it blocks; NO-CALL(ADMIT) names the proof rule and the next block (columns that remove it), re-registered by amendment before any fit | none beyond the MM's amendment |

## 7. Controls (each built so it can fail; every one reported; the gate runs them BEFORE any fit)

- **C-AGREE (provenance of the geometry):** the extraction's recomputed NAVSIM verdicts `rl6`, clipped to [0, 1], equal the stored `labels6` DAC column for >= 99 % of proposals (the NC column is reported, not gated: the NC row is not read here), else VOID-INPUTS. Reported with no gate: `P(dac_clear_min < 0 | DAC = 0)`, `P(dac_frac_out > 0 | DAC = 0)`, `P(dac_clear_min >= 0 | DAC = 1)` (PRIVGEOM-SAFE measured 1.0000 / 1.0000 / 1.0000 on the 41).
- **C-JOIN (label-free):** extraction keys (log, token, step, rank, ckpt_step) match the census bank 1:1 on every retained set; sets lost in extraction are counted and listed (> 5 % disclosed in the headline); **retained logs >= 60 of 76 and every fold >= 1 log**, else VOID-INPUTS.
- **C-ADMIT:** section 4.
- **C-REACH (planted label):** `G := the DAC label` (one column, centred like the block) through the REAL fit and score code on the 76's real arrays: F-AUC >= 0.99 for L+G AND for G-only on BOTH seeds (4 fits). Run on synthetic arrays in the tests AND on the real arrays in the gate. FAIL = gate FAIL.
- **C-POWER (planted effect of the committed size):** `G := label + sigma N(0, 1)` (one column, centred; `default_rng(PLANT_SEED + 1)`, `PLANT_SEED = 20261050`), **sigma = 1.25** (`run/plant_calibration.json`, MEASURED on the SEEN arrays: Delta_F **+0.0276 [+0.0182, +0.0374]** at sigma 1.25 against the +0.030 target; grid 0.75 / 1.0 / 1.25 / 1.5 / 1.75 / 2.0 / 2.5 read +0.0708 / +0.0426 / +0.0276 / +0.0181 / +0.0124 / +0.0086 / +0.0046). Must satisfy the CONFIRM rule on BOTH seeds; otherwise a NULL call converts to UNDERPOWERED. **The plant is built from the 76's labels at the gate, with sigma fixed from the SEEN logs only.**
- **C-NULL (the machinery null / the rig's replicate arm; the MM ruling: "future permutation nulls centre within-set by construction").** The block is permuted across the 64 proposals WITHIN each set (one permutation per set moves the whole 9-vector; token and label untouched), then centred (permutation-invariant, so this IS the within-set-centred null). **5 replicates**, permutation seed `20261019 + 7919 x rep`, replicate `r` read on training seed `r mod 2` (3 on seed 0, 2 on seed 1). ⭐ **VOID rule (SH5 Amendment 3 is the precedent): a replicate FIRES iff it satisfies THE CALL'S OWN RULE (`seed_call == CONFIRM`: lower CI > 0 AND point >= +0.02) on the seed it is read on; the run is VOID iff ANY replicate fires.** A machinery null must be read by the test the claim uses, or it cannot guard that claim (the R52 class: a control blind to the effect size it exists to catch). False-VOID price (ESTIMATED): the within-set-centred null reads Delta_F about -0.0006 / -0.0015 with an interval about 0.001 wide (PRIVGEOM-SAFE §5.3; SH5's noise-trained replicates scattered with sd 0.0006), so a point >= +0.02 is not reachable by noise (sd 0.0006); a machinery that manufactures >= +0.02 would VOID, which is the point.
- **C-ID (identity):** (i) MEASURED pre-seal: the FROZEN extraction on one SEEN log reproduces PRIVGEOM-SAFE's banked `scalars/<log>.npz` `g` and `rl6` **bit-for-bit** (`run/identity/identity_extract.json`); (ii) MEASURED pre-seal: the new runner on the banked SEEN join reproduces the banked `drop0` OOF / Delta_F (section 9); (iii) AT THE READ: the full extraction of the two G-LIVE logs reproduces the block sha256 registered in section 12 (`run/glive/glive.json`).
- **C-RED (branch table):** `pgc_lib.branch_table_failures` (decision functions with their RED mutations: each of 28 mutations must make the table fail; also a property test that CONFIRM and NULL never both hold) and the admissibility mutation table (section 4).
- **C-GRAD (G-LIVE of the fit path):** a one-step probe: every parameter group (token `w`, `b`, block `u`) moves on the real centred block, and `u` is EXACTLY 0.0 on an all-zero block (the control that proves the probe sees a zero).
- **Frozen suites, run unchanged by the gate:** `test_pg_geom.py` (19 tests, 4 RED mutations) and `test_pg_lib.py` (C-ID-L: the L arm equals the landed `fit_rows`; C-REACH; C-CONV; mute-G mutation).

## 8. Power / MDE at 76 logs (design calculation on banked SEEN data ONLY; `run/design.json`, `code/pgc_design.py`; EVIDENCE CLASS: MEASURED inputs, ESTIMATED SE; no label of the 76 read)

- The estimator is a ratio of per-log sums, so its variance is a function of the per-log decomposition `Delta = sum e_l / sum d_l`, `r_l = e_l - Delta d_l`, `SE^2 = sum r_l^2 / D^2` (checked against the registered bootstrap on the SEEN data: **0.00756 vs 0.00765**, ratio 0.988). Model for the 76: `d_l = c x fprop_l^alpha` (alpha 1.353) from the label-free F-proposal counts of the 76; `Var(r_l) = a d_l + b d_l^2` (a = 3.87 set-level, b = 5.6e-4 log-level, NNLS on the 41).
- **SE at 76 logs (ESTIMATED):** pairs scaling **0.00506**, log-count scaling **0.00555**, the model **0.00727** (conservative, sizes the statements), its nested-bootstrap p5 / p50 / p95 **0.0032 / 0.0071 / 0.0133** (**the model is poorly determined**: the 76 contain one log predicted to hold about 20 % of the F pairs, so widening 41 -> 76 logs buys little: SE 0.00765 -> 0.0073).
- **CONFIRM needs the point to clear `max(+0.02, 1.96 SE)`:**

| SE | confirm point threshold | **MDE (80 % power)** | MDE (90 %) | P(CONFIRM \| +0.0276, the SAFE point) | P(CONFIRM \| +0.0352, centred) | P(CONFIRM \| +0.020) | P(false CONFIRM \| 0) | P(NULL \| 0) | P(NULL \| +0.010) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00506 (pairs) | 0.0200 | **+0.0243** | +0.0265 | 0.93 | 0.999 | 0.50 | ~0 | 0.98 | 0.51 |
| **0.00727 (conservative)** | 0.0200 | **+0.0261** | +0.0293 | **0.85** | 0.98 | 0.50 | 0.003 | **0.79** | 0.28 |
| 0.0133 (p95) | 0.0261 | **+0.0373** | +0.0432 | 0.54 | 0.75 | 0.32 | 0.025 | 0.32 | 0.11 |

- ⭐ **READ:** the MDE (+0.026 at the conservative SE) sits **at** the descriptive +0.028 and above a shrunk effect. The arm is powered for the SAFE reading itself (85 %), NOT for half of it: a true +0.020 is a coin flip, a true +0.015 returns CONFIRM only 25 % of the time, and a true zero returns NULL only 79 % of the time (the real-null SE is likely smaller than the real-arm SE, so that 79 % is conservative). **INCONCLUSIVE is the modal outcome if the +0.028 shrank at all** (the reading was post-hoc; the arm was chosen after the leak was found, not by effect size, so winner's curse is expected to be mild but is not zero). The p95 row is the honest tail: if the log-level variance is as the upper end of the model, no outcome except a large effect is decisive.
- Seeds are one draw of logs for this table (the two seeds agree to 0.0007).

## 9. Rehearsal on the SEEN logs (MEASURED pre-seal; the runner and every control exercised on banked data; NOT a result, stamped `rehearsal: true`; `raw/rehearsal/`)

The runner (`pgc_run.py gate -> fit -> analyze`) was pointed at PRIVGEOM-SAFE's banked join of the 41 SEEN logs. **The SEEN logs are excluded from the read; nothing below is evidence about the 76.**
- **Identity to PRIVGEOM-SAFE's banked numbers:** the Delta_F statistics reproduce to **1e-9**: `LGu` (uncentred) seed 0 **+0.027627 [+0.013004, +0.043003]** = the banked `drop0`; `LGc` (centred) seed 0 **+0.035194 [+0.019356, +0.049507]** = the banked centred `dacGVnomin`. The OOF logits agree with the banked arrays to **<= 1.7e-5** (`L`: 2.9e-6, 1.9e-6; `LGu`: 1.7e-5), not bit-for-bit: float round-off from the thread count (OMP 6 here, 8 there). `RowView` over the census tokens equals the banked `feat_sub.npy` rows exactly. `select` reproduces `run/coverage.json` (76 logs, 10,867 sets, 6,140 tokens, same `census_idx`); `join` on the SEEN scalars reproduces PRIVGEOM-SAFE's `arrays.npz` **bit-for-bit** (11 arrays). The FROZEN extraction on one SEEN log (8 sets) reproduces the banked `g` and `rl6` **bit-for-bit** (`raw/identity/identity_extract.json`).
- **Gate PASS on the SEEN arrays** (`raw/rehearsal/gate_report.json`): C-AGREE 1.0000 on 316,224 proposals; C-REACH 1.0000 (L+G and G-only, both seeds); C-ADMIT admissible with every control right and the RED arm leaking; C-ID-EXTRACT n/a (rehearsal); all four suites rc 0 (`test_pg_geom`, `test_pg_lib`, `test_pgc_lib`, `test_pgc_admit`); the branch table has no failures.
- **Analyze on the SEEN arrays:** primary Delta_F **+0.0352 [+0.0194, +0.0495]** (seed 0) / **+0.0352 [+0.0198, +0.0494]** (seed 1); the companion **+0.0276 / +0.0278**; **C-NULL: 5 replicates read -0.0001 to -0.0010 (every interval <= 0.001 wide), 0 fire**; **C-POWER** (sigma 1.25) **+0.0276 [+0.0182, +0.0374] / +0.0276 [+0.0181, +0.0376]**, fires. ⚠️ The rule applied to the SEEN logs reads CONFIRM: that is the reading the rule is sized from, so it shows only that the pipeline returns the committed outcome on a known +0.03 effect and about 0 on a null; it is not a confirmation.
- **Re-rehearsed after the `PRIMARY_ARM` constant was added (2026-10-19, last edit before the draft was closed):** the whole chain (`code/rehearsal_chain.sh`: gate -> fit x 2 shards -> analyze) was re-run from scratch on the final pinned code, the old fits kept aside as `run_rehearsal/fits_pre_primaryarm/`. The 19 re-fitted OOF arrays are **bit-for-bit identical** to the earlier ones (max abs diff 0), the gate passes, the token's code pins equal the current pins, and analyze reads the same tables (`LGc` primary +0.0352 / +0.0352, companion `LGu` +0.0276 / +0.0278, C-NULL 5 of 5 at -0.0001 to -0.0010, none fires, C-POWER +0.0276 on both seeds).
- **Refusal tested for real:** an edit to `pgc_run.py` after the first rehearsal gate made `analyze` REFUSE ("the code / join differ from what the gate token binds"), as the pins are meant to.

## 10. The four families (`TanitAD_BenchmarkCriteria`)

This is a **diagnostic label lookup of a head's within-token ranking**, not a model eval: no lateral / longitudinal / tactical / strategic CLAIM is made and the four-family rule is not engaged for pass / fail. Stated per family so none is silently dropped:

| family | status here | what this read can say | what it cannot say | n |
|---|---|---|---|---|
| **LONGITUDINAL** | NOT MEASURED | nothing: the block has no speed, headway, time-gap or TTC content; the NC columns (lead / TTC geometry) are extracted but NOT read by any stage | any target-speed or distance-keeping claim | 0 proposals scored on NC |
| **LATERAL** | the closest analogue, still not a lateral-skill metric | whether privileged boundary clearance (the lateral-containment quantity) ranks DAC-unsafe against DAC-safe proposals better than P2's DAC head, inside one set, in the fast-speed stratum F | heading, curvature, yaw-rate or cross-track error; any driving skill | F pairs: PRIVGEOM-SAFE had 332,614 on the 41; predicted about 740,000 on the 76 (172,523 F proposals, 10,867 sets, 76 logs) |
| **TACTICAL** | NOT MEASURED | nothing: no manoeuvre decision, goal or anchor selection is scored; no pick-level read is run | any selected-vs-executed manoeuvre claim | 0 |
| **STRATEGIC** | **PAUSED** (PI ruling ITEM 25, 2026-09-19, `PI_DECISION_QUEUE.md`: the strategic layer is switched off temporarily) | reported PAUSED, with this ruling; never a pass, never silently omitted, never extended to the other three | - | - |

Tier stamp on every number: **SELECTION tier, T0-class label lookup, P2 in-sample (arms OOF), privileged geometry = diagnostic only.**

## 11. Forecast (a prediction, not a bar; HYPOTHESIS, committed before any number on the 76)

Prior on the true held-out `Delta_F` (centred arm): centred near **+0.017**, sd 0.010 (the +0.028 / +0.035 descriptive reads, shrunk for selection and for P2 in-sample). With the conservative SE (0.0073) and the rules of section 6 (CONFIRM needs the point >= +0.02; NULL needs the point < +0.0057): **CONFIRM about 0.39, INCONCLUSIVE about 0.40 (of which SMALL about 0.18), NULL about 0.17, UNDERPOWERED / VOID / NO-CALL about 0.04.** C-NULL false fire is about 0; the admissibility check refusing the real block on 76 logs is the main non-statistical risk (the SEEN best proofs are 0.84-0.89, well under 0.999, but 76 new logs can differ). The uncentred companion reads about 0.005-0.008 below the primary.

## 12. Cost (MEASURED on 2 of the 76 + PRIVGEOM-SAFE's realised timings; `raw/cost.json`, `raw/glive/glive.json`)

- **G-LIVE on 2 of the 76** (frozen extraction, one process, CPU only, a LOADED box): `2021.06.09.14.58.55_veh-35_02778_02850` 51 sets / 29 tokens, wall 122 s, **CPU 110 s**; `2021.06.14.11.44.56_veh-35_02696_02932` 93 sets / 51 tokens, wall 172 s, **CPU 160 s**; peak RSS 0.38 / 0.39 GB; 0 sets lost; every sign-clean column finite, in [0, 10], non-negative; `abs_clear_t00` constant within every set, `t05..t40` varying. **Rate: 1.87 CPU-s per set all-in** (2.04 s of component time). PRIVGEOM-SAFE's realised rate on the 41 was 1.13 s per set (1.55 CPU-h for 4,941 sets), its 20-set cost model 3.9 CPU-h for the 76.
- **Projected extraction for the 76 (ESTIMATED, linear in sets):** **3.4 CPU-h (SAFE's realised rate) to 6.2 CPU-h (today's loaded rate); central 5.7 CPU-h**; with 2 workers **1.7 to 3.1 h wall**. The 1,495-set log alone is about 0.8 h at the loaded rate, so the wall is bounded below by that one log. Far under the 16 CPU-h cap.
- **After the extraction (projected from the SEEN rehearsal, scaled by 10,867 / 4,941 = 2.2):** gate about 11 min (the four suites about 3 min, C-ADMIT about 5 min, C-REACH 4 fits about 1.5 min); the 15 read fits about 500 s single-process (about 5 min on 2 shards); analyze under 1 min. **Total about 6.4 CPU-h worst case, about 2-3.5 h wall.**
- **G-LIVE block hashes registered for C-ID (iii)** (sha256 of the OK sets' float64 `|g[..., 2:11]|`; two independent G-LIVE runs reproduced both exactly; the gate's C-ID-EXTRACT recomputes them from the read's extraction and a mismatch FAILS the gate): `2021.06.09.14.58.55_veh-35_02778_02850` `344ac86cb900562538f82d8387a4451d4ccd8809e0ee2739210dd6b4ba7943a9`; `2021.06.14.11.44.56_veh-35_02696_02932` `404d071c4f858ddd7c8e24666d268a0dbb3b61c0ebc34d7586d3bb0494d8c658`.

## 13. Execution plan after the seal (a fresh agent; every long job detached with a HANDOFF.md; poll markers, never turns)

1. `pgc_run.py select` (label-free) -> `run/subset_clean.json`.
2. Two workers: `pg_extract.py work --subset run/subset_clean.json --out run/scalars` (frozen, `launch.ps1`), then `finish_clean.sh` (waits for 76 `.npz`, no locks; then `join` -> `gate` -> `fit` x 2 shards -> `analyze`). Markers `run/FINISH_DONE.txt` / `run/FINISH_FAILED.txt`.
3. If the gate FAILS: read `run/gate_report.json`; do NOT retune anything (the gate's C-ADMIT names a refused block). A C-AGREE failure does not stop the fits: it VOIDs the call.
4. Write `RESULT.md`: the call, the controls first, the per-seed table, the companion, the four-family statement, the next arm from section 6, the proposed register text (the registers are not edited by this agent).
5. Code pins (sha256 at this draft; `pgc_pins.py --check` re-verifies against `code/PINS.json`; the gate token binds the pipeline files and the fit / analyze stages refuse on a mismatch; the FROZEN files carry PRIVGEOM-SAFE's own sealed pins unchanged, so the imported fit, scorer geometry and extraction ARE PRIVGEOM-SAFE's code):

| file | frozen? | sha256 |
|---|---|---|
| `pg_geom.py` | FROZEN (PRIVGEOM-SAFE, unchanged) | `5b78a694f40b6b2b4030fdeb3bf5be2c65b349c8887f9c11c78d0dfd8d9df7de` |
| `pg_extract.py` | FROZEN (PRIVGEOM-SAFE, unchanged) | `64a148906329f8aeba5dc3d87d30aba9f2e91d90389ea77c0169614e1bade23b` |
| `pg_lib.py` | FROZEN (PRIVGEOM-SAFE, unchanged) | `61bfe59692987de6e43cf9d2585c2992192a4b9dca6d513688bffc5c77dd7178` |
| `test_pg_geom.py` | FROZEN (PRIVGEOM-SAFE, unchanged) | `3cf48766adc17b4c1ac07bb1b043e3798ffa2a680226196f124fc69f56406b64` |
| `test_pg_lib.py` | FROZEN (PRIVGEOM-SAFE, unchanged) | `fd1d408aed2d870f86c9a02a556b32ce0f54708042c25823666c425f06b89cfa` |
| `pgc_lib.py` | new | `85561d6b8eb34190ea936236443ef298cfa4206cbba738c44e9beecb611daa9d` |
| `pgc_admit.py` | new | `4a68f58ed1629c831a4263f2bcc30f1bbc730a5cd81ebb6fe50f0f5c99354696` |
| `pgc_run.py` | new | `bc92f313c0b7a091254f46b3944c79710602890bafcb217857e220211c9171ce` |
| `pgc_coverage.py` | new | `06115f13b7c7e4b748225faeff0dbfa118d96df9cd75cd0ceafb39fdfcc6e097` |
| `pgc_design.py` | new | `5411d331d932f4184bd8fb709bb3c12a44ce36334b9156840465233371b74523` |
| `pgc_calibrate_plant.py` | new | `1b7870837420c6132fe6b530a912862f84d33104a78f6cfa232272ad4c783c62` |
| `pgc_glive.py` | new | `4b3ab9409f8c7ff4fa9c23ae3a853203da619927b047a896da774e04c4fc3984` |
| `pgc_admit_seen.py` | new | `070d77428a8e7bb0a7979004f9a1be7ca74674fd50bb00e1f8dcc8f9ebcc15d6` |
| `pgc_cost.py` | new | `69e4d44bf2cd7ecd46174bf868e0587841ac4225f33bda6c48b0c7dcec33d0a1` |
| `pgc_rehearsal_identity.py` | new | `837c18d9dec9da338bff8afbc8b84d32a477d55ad910e445bce5ff1ccc19a12f` |
| `build_package.py` | new | `ec236867698670d7ff8e40dd156c22ff5952b51e92206cc7846eec3975f855a2` |
| `pgc_pins.py` | new | `262d00d75eac9ceb5d2c2c3a9823bb5217569a4b95c1754cd8d9f95ba6ecdb08` |
| `test_pgc_lib.py` | new | `9f701bc784015e5d5c73d5deca59b3290bcff94878505fd4ceefa275d66a2807` |
| `test_pgc_admit.py` | new | `ccc5dfa7f27959b6af1c8b26e8cba8e259615dfbf26b32a2b22611c92783dad3` |
| `finish_clean.sh` | new | `9778a34902e40d75fd59f1c6965d6a08c88361ec1689071b9dfdc3700a5705e1` |
| `launch.ps1` | new | `877622167acf8bce9842f74141e329c5eafad26dd7524010d48deea94d3d4695` |
| `rehearsal_chain.sh` | new | `70a5905aeba43ebabfd7f2cf256d307d9308aba23c8a020d1cf8cd77a421cc42` |
| `ref/fastneg_lib.py` | FROZEN (PRIVGEOM-SAFE, unchanged) | `0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca` |
frozen files unchanged: True

A change to any file after the seal = an Amendment sealed before the stage it governs; the gate FAILS on a stale token.

## 14. Decisions for the Master Mind before sealing

- **DQ1 (primary arm): centred (my default) or uncentred (`drop0` exactly as written).** Section 3; the switch is the constant `PRIMARY_ARM` (`pgc_lib.py:21`), made before the seal, then re-pin and re-run `code/rehearsal_chain.sh` (about 8 min).
- **DQ2 (bars): CONFIRM = lower CI > 0 AND point >= +0.02; NULL = upper CI < +0.02** (not SH5's `< +0.01`). Section 6 and 8. If you want SH5's NOT-REACHABLE bar, NULL fires 28 % of the time at a true zero.
- **DQ3 (scope): DAC only.** NC-CLEAN is a separate prereg on the same extraction.
- **DQ4 (the plant): sigma 1.25 (+0.028 on SEEN).** C-POWER fires iff the CONFIRM rule holds on both seeds.
- Free C: before the read. The two scalar stores and the OOF arrays are about 150 MB; the 256-d tokens are read through a row view of the census `feat_f10.npy` (no 356 MB copy).

## Limits stated in advance

- **76 logs is every log there is at f = 10 %**; a file-staging order chose the 41 / 76 split, not a random draw. The 41 / 76 strata mixes agree (F share 24.4 % / 24.8 %); the rest of the distributions are unknown.
- SELECTION tier, P2 in-sample; one rig, two refit seeds; the interval answers "another draw of LOGS" only.
- `Delta_F` is a LINEAR read of 9 (effectively 8) privileged columns; "upstream" means upstream of the 256-d token under a linear read (`D-REFE-HEADROW-PROBE-1`), not that a vision model can reach it (SH5).
- The clearances are on NAVSIM's simulated (LQR + bicycle) states, not the raw plan.
- The admissibility check is a SEARCH: absence of a proof at precision 0.999 / recall 0.5 % is not a proof of absence of every leak; the planted controls bound what it can see, nothing else does.
- Evidence class for every number this arm produces: MEASURED, SELECTION tier, T0-class label lookup; no capability claim.
