# PREREG L2-FRESH-STEPA, AMENDMENT 2 (to `PREREG_L2_FRESH.md`): the next-arm mapping of O3 ONLY. READ 2 on universe D is WITHDRAWN; O3's next arm is chosen by a design calculation, and it is the UNSEEN-STRATUM DATA JOB

**Status: SEALED** (2026-10-19; the exact time is in `SEALED_AT_AMENDMENT2.txt`).
Sealed by the L2-FRESH-STEPA agent under the Master Mind's ruling `MM_RULING_L2_FRESH_D.md`, carried VERBATIM in section 1.
**No fresh-carve label or score existed when this file was sealed.** The pre-launched B chain was still waiting for the dev-box GPU lock, `B_token_disjoint/` held only `step3.log`, and `SEALED_AT_AMENDMENT2.txt` records that listing and the chain log's last line. READ 1 has not been read. The design calculation below reads ONLY Step A's banked labels, pairs and L2's banked out-of-fold scores (data already read), plus one fixed plan number per log (the B token COUNT, not a B measurement).
Parent, unchanged: `PREREG_L2_FRESH.md` sha256 `{{PARENT_SHA}}` (the file the read's seal guard checks; this amendment does not alter it, its pins, or any code).

## 0. What this amendment changes, and what it does not
* **Changes:** the **O3 row of section 5.4 of the parent, and only that row.** Its "READ 2 on universe D" is WITHDRAWN (section 1) and replaced by the mapping of section 3 and 4.
* **Does NOT change:** any gate; the estimator (paired log-cluster bootstrap, 24 logs, 10,000, seed 20260927); the carve (B, `0629811f...`); the heads; the outcome rule (parent section 5.1, `interval_aware`, no point rule, verbatim); the two-seed combination (most severe of seeds 0 and 1); the stated expectation (O3); the reported-only A3 and log-OOF companion; the **O1 row** (Step B) and the **O2 row** (L2-BRANCH); the pins; the code. The read command in `READY.md` section D is unchanged.
* This amendment is sealed BEFORE READ 1 is read, so that the O3 consequence is not chosen after seeing the result.

## 1. The Master Mind's ruling on READ 2 / universe D, VERBATIM (binding)

````text
{{MM_RULING}}
````

## 2. The design calculation (ESTIMATED: a model fitted on Step A's own banked per-token terms; MEASURED inputs, projected output)
Code `code/design_calc_ball.py` (LF sha256 `{{CALC_CODE_SHA}}`), result `raw/design_calc_ball.json` (sha256 `{{CALC_JSON_SHA}}`), log `logs/design_calc_ball.log`. CPU only; it reads no carve-B label, score or feature, and touches no pinned file.

**2.1 Estimator, stated.** A1 = `stepA.analyse(...)["all"]["A1"]`: the POOLED lon + lat ratio of sums of per-pair concordance over pairs with |true dEP| >= 0.05. The interval is the PAIRED LOG-CLUSTER bootstrap (resample the 24 logs, B = 10,000, seed 20260927, percentile 2.5 / 97.5). **It answers "would another draw of LOGS say this" and nothing else** (not another training seed, `H-ESTIM-SEED-1`).

**2.2 (a) Identity check: PASS.** `stepA.analyse` on the banked out-of-fold scores reproduces the banked L2 A1 EXACTLY (floating-point ==, point, lo, hi, den): seed 0 `{{A1_S0}}` [{{A1_S0_LO}}, {{A1_S0_HI}}] (the stated 0.827 [0.783, 0.873]), seed 1 `{{A1_S1}}` [{{A1_S1_LO}}, {{A1_S1_HI}}], P2 `{{A1_P2}}` [{{A1_P2_LO}}, {{A1_P2_HI}}]; a token-level re-derivation agrees to 1e-12; the RED arm (the EP scores rolled by one token) does NOT reproduce.

**2.3 (b) Between-log versus within-log variance.** Subsampling tokens inside each log at 25 / 50 / 75 / 100 % of Step A's own allocation ({{R_REPS}} draws per level below 100 %, {{N_DRAWS}} draws in all), the bootstrap variance V of the ratio, fitted as `V = a + b / n` with n = tokens per log of the draw (a = BETWEEN-LOG floor, b / n = WITHIN-LOG part). **Fit window n in [{{WIN_LO}}, {{WIN_HI}}] tokens per log; 4 levels; R^2 {{R2_MEANS_S0}} on the 4 level means (seed 0; seed 1: {{R2_MEANS_S1}}; 2 residual df, so weak evidence by itself) and {{R2_ALL_S0}} on all {{N_DRAWS}} single draws (seed 0; a single subsample's bootstrap variance is very noisy; only the mean curve is fitted).**
Seed 0: a = {{A_S0}}, b = {{B_S0}}; seed 1: a = {{A_S1}}, b = {{B_S1}}. **The between-log floor is {{SHARE_STEPA_S0}} of the bootstrap variance at Step A's own allocation and {{SHARE_B_S0}} at the B-ALL allocation (seed 0).** An independent one-way ANOVA moment decomposition of the per-token residuals agrees: sigma_b^2 = {{SB2_S0}}, sigma_w^2 = {{SW2_S0}}, ICC {{ICC_S0}}, and its closed-form floor at B-ALL is {{HW_FLOOR_CF_S0}} (seed 0) against the fit's {{HW_FLOOR_S0}}.
**The binding term is the log-cluster count.** The 24 logs carry an EFFECTIVE cluster count of only {{GEFF_A}} at Step A's per-log allocation and {{GEFF_B}} at B-ALL's (per-log token counts are near-proportional: correlation {{CORR_ALLOC}}, B-ALL / Step A per log {{RATIO_MIN}} to {{RATIO_MAX}}), because the per-log token counts are very unequal (Step A: 3 to 55 per log; B-ALL: 8 to 186).

**2.4 (c) Projection to B-ALL (all 1,305 B tokens, 24 logs; per-log counts from `raw/fresh_universe_probe.json` -> `U0_B/B_per_log`).** Half-width of the A1 interval, as the sealed estimator would report it:

| quantity | seed 0 | seed 1 |
|---|---|---|
| observed at Step A (400 tokens), the MM's 0.045 | {{HW_OBS_S0}} | {{HW_OBS_S1}} |
| **projected at B-ALL, proportional form `a + b/n_B` (the branch input)** | **{{HW_S0}}** | **{{HW_S1}}** |
| projected, allocation-exact regression with B-ALL's real counts | {{HW_EXACT_S0}} | {{HW_EXACT_S1}} |
| the sealed estimator on data simulated from the fitted model (median) | {{HW_SIM_S0}} | {{HW_SIM_S1}} |
| model-free UPPER bracket (Step A's tokens resampled to B-ALL's counts; carries Step A's own within-log noise as signal) | {{HW_UP_S0}} | {{HW_UP_S1}} |
| floor, tokens per log -> infinity (b = 0): the lower bound of ANY extrapolation in n | {{HW_FLOOR_S0}} | {{HW_FLOOR_S1}} |
| at 2x the fitted range (n = {{N2X}}, the admissible extrapolation limit) | {{HW_2X_S0}} | {{HW_2X_S1}} |
| fit-noise interval of the projection (outer bootstrap over the 24 logs, {{N_OUTER}} draws, 2.5 / 50 / 97.5 %) | {{OUT_S0}} | {{OUT_S1}} |
| share of outer draws with projected half-width <= 0.025 | {{SHARE_OUT_S0}} | {{SHARE_OUT_S1}} |
| closed-form floor, normal-theory 95 % band (chi-square, df = G_eff - 1 = {{DF}}) | {{CF_BAND_S0}} | {{CF_BAND_S1}} |

**Flags, stated and not hidden.**
* **EXTRAPOLATION: B-ALL has {{NB}} tokens per log against a fitted maximum of {{WIN_HI}}, i.e. {{XF}}x: BEYOND the 2x rule.** The verdict does not depend on it: the n -> infinity floor ({{HW_FLOOR_S0}} / {{HW_FLOOR_S1}}) is above the threshold, so no value of the extrapolated within-log term can bring the half-width to 0.025 while the floor stands.
* **FIT NOISE: the floor itself is the weak point.** It is estimated from 24 logs; the outer bootstrap's 95 % band for the projection reaches {{OUT_S0_LO}} (seed 0) and {{OUT_S1_LO}} (seed 1), so "<= 0.025" is unlikely ({{SHARE_OUT_S0}} / {{SHARE_OUT_S1}} of outer draws), not excluded by that bootstrap; the normal-theory ANOVA band's lower end is {{CF_LO_S0}} / {{CF_LO_S1}}. Percentile outer bootstraps over 24 clusters are biased low (median {{OUT_MED_S0}} vs point {{HW_S0}}); read the band as a width, not a CI.
* **SCOPE: A1 only.** O1 under `interval_aware` also needs A4's interval to exclude 0.10 and A5's point <= 0.05; the banked OOF reads A4 = {{A4_S0}} and A5 = {{A5_S0}}, nowhere near their gates, so A1's interval is the term that decides O1 versus O3.
* **ASSUMPTIONS (the projection is a model, not a B measurement):** B's per-token A1 terms have Step A's variance structure (mean {{DBAR}} eligible pairs per token, the same residual spread); the A1 point on B is near the banked OOF's (the variance of a ratio shrinks as it approaches 1, grows as it falls, so a lower point widens the interval); one training seed per head.
* **What B-ALL would buy.** It lowers the expected half-width from {{HW_OBS_S0}} (Step A, 400 tokens) to {{HW_S0}} (seed 0), i.e. by {{PCT_GAIN}} % for 3.26x the tokens, and it would clear O1 only with an A1 POINT of about 0.80 + {{HW_S0}} = {{PT_NEEDED}}, against the banked out-of-fold 0.827.

## 3. The branch, applied (the ruling's rule, unchanged)
The ruling: *projected half-width <= 0.025 => O3's next arm is B-ALL; otherwise the Step A gate is structurally unresolvable at the data on this box and the next arm is the unseen-stratum data job.* The larger of the two seeds' projections is compared with 0.025 (the L2 verdict is the most severe of the two seeds; the seeds are near-replicates): **{{HW_MAX}} > 0.025.**
**Selected branch: {{BRANCH}}.**
B-ALL is NOT run. A B-ALL run, or any other re-read on the 24 logs, would need its own sealed amendment.

## 4. O3's next arm after this amendment: the UNSEEN-STRATUM DATA JOB (a plan; nothing here launches, fetches or spends)
**4.1 The stratum (A1.5(ii)), MEASURED** (`raw/scope_unseen_stratum.json`, `code/scope_unseen_stratum.py`, read-only listings and the token table): nuPlan val-split DBs on this box that no navtrain token-table row names. The box holds {{N_TV}} trainval DBs; {{N_TV_IN_NT}} are navtrain logs; **U = {{N_U}} logs**, disjoint from the {{N_NT}} named navtrain logs by construction (so disjoint from M-PDM and M-SCENE BY NAME). The coincidence with M-PDM's size (1,167) is arithmetic ({{N_TV_IN_NT}} - 24 = {{N_TV}} - {{N_NT}}), not a conflation of the sets. The test split (1,349 DBs) shares no name with it.
**4.2 Sizing (ESTIMATED, same model, A1 only, balanced new logs; `design_calc_ball.json` -> `sizing`):** logs needed for an A1 half-width <= 0.025 at the banked variance structure:

{{SIZING_TABLE}}

At 4 tokens per log that is about {{SZ4_LOGS}} logs and {{SZ4_TOK}} tokens, i.e. about {{SZ4_JPG}} jpgs at 4 cameras per token: a SMALL fetch against the {{N_U}} available logs. These are inputs for the data job's own prereg, not a committed design; they assume the new logs have the variance structure of the 24 (an A1 point lower than 0.827 on unseen logs would widen them), and they say nothing about A4.
**4.3 Where the frames would come from (scoping probe at THREE locations, with the repo's own scripts as the guide; no download, no pod):**
* **Location 1, local frame roots: NO frames for U.** Of {{N_ROOTS}} roots listed (`refe_proxy/pixels`, `refe_m5/pixels`, `nuplan-camera/{mini,scenarios,scenarios4,probe}`, `refe_navtrain`, `refe_navtrain10`, `refe_navtest`, `refe_m5r`, `refe_consistent`, `C:/Users/Admin/p2heldout_work/frames_heldout`, `videos`), U appears in exactly one, `nuplan-camera/mini`: ONE log, front camera only. The proxy zips and M5 jpgs are navtrain logs (0 names in U).
* **Location 2, HF `OpenDriveLab/OpenScene` (read through the public dataset pages, nothing saved):** the card is empty apart from `cc-by-nc-sa-4.0` and a total of 3.04 TB. `navsim/` holds only `navtrain_current_{1..32}.tgz` (304.5 GB) and `navtrain_history_{1..32}.tgz` (144.5 GB): NAVTRAIN logs; the one such archive probed on this box (`navtrain_current_25`, 38 logs) has 0 logs in U. `openscene-v1.1/openscene_sensor_trainval_camera_{0..199}.tgz` is 200 gzip-tar shards, 1.242 TB in all (1.77 to 19.3 GB each): gzip tar has no per-frame range access, so a targeted fetch from it is a shard-level download. The shard map `openscene_sensor_trainval_0-199.json` (68 KB) was NOT read: opening it is a file download and needs a yes.
* **Location 3, the nuPlan S3 bucket the repo's own scripts already use (`pull_nuplan_splits.sh` pulled `nuplan-v1.1_val.zip`, the very archive whose DBs are U):** its bucket listing shows `public/nuplan-v1.1/sensor_blobs/val_set/` with 12 camera archives `nuplan-v1.1_val_camera_{0..11}.zip` (866.7 GB in all, 56 to 85 GB each) and `public_set_val_sensor.txt` (8,945 B). Zip central directories are range-readable, and the repo's `fetch_front_camera.py --list / --only-logs` already does this for the mini set (central directories cached in `data/nuplan-camera/_cdcache`, 357 MB for 19 archives), including the extra channels (`nuplan-camera/scenarios4` holds CAM_B0 / CAM_L0 / CAM_R0 zips). **This is the route.** It needs one network step first (the 12 central directories, ESTIMATED 0.2 to 0.4 GB by scaling the mini cache's ~19 MB per archive to the val archives' size, then the chosen jpgs).
* **UNVERIFIED** (named, not assumed): that the val camera archives hold every U log (check against the central directories or `public_set_val_sensor.txt`, 8.9 KB); the byte-run size of the chosen jpgs; the token frame (which scene tokens of a U log are admissible: the navtrain tokens come from the OpenScene metadata, `openscene_metadata_trainval.tgz` 7.05 GB, while a U log's DB alone gives lidar_pc tokens; the frame rule must be fixed and preregistered before any label); exposure of U to training manifests OTHER than M-PDM / M-SCENE (trunk caches, the DriveRL val14 list).
**4.4 Gates the data job must carry (existing, not new):** G0 re-measured on U against EVERY manifest that can have touched a nuPlan val log (the name-level result is 0 for M-PDM and M-SCENE; the rest is UNVERIFIED); G1, G2a, G2c, G3 as in L1; **G2b** has no banked fan for U tokens, so P2's fan comes from the bundle code that `bridge_g2b.py` first reproduces bit for bit on B tokens (built and tested on a fake planner, NOT run; it needs the dev-box GPU after the B chain releases the lock); C-FEAT; the same `interval_aware` rule with clusters = logs.
**4.5 Priority, per the ruling (verbatim in section 1):** set by the Step B diagnostic. A NOT CLEARED Step B makes L2-BRANCH the lever and the data job moot for this head. Approvals the job needs and this amendment does NOT give: the PI's approval of the network fetch (P6 overlap rule), any pod or dev-box GPU time, and disk (D: is nearly full, exFAT, 1 MiB clusters: the fetch must land in per-log zip containers, as `fetch_front_camera.py --container` does).
**4.6 The withdrawn arm.** READ 2 on universe D (99 tokens over 84 logs, 83 of 84 logs inside M-PDM and M-SCENE through other tokens, estimated half-width about 0.054) is withdrawn; `PREREG_L2_FRESH_D_DRAFT.md` stays as a measured scoping record and has no force. The 412 / 396-jpg pod-fetch request is withdrawn.

## 5. What this licenses, and what it does not
It licenses only the O3 mapping of section 3 and 4 after READ 1 reads O3. It licenses no capability claim, no unseen-log claim, no navtest number, and no fetch, spend or launch. **Every number in section 2 is ESTIMATED (a model of the bootstrap variance fitted on Step A), not MEASURED on any fresh token.** The STRATEGIC family stays PAUSED (PI ruling 2026-09-19, ITEM 25).

## 6. The guard, and what it covers
The read's seal guard (`analyse_fresh.py`, parent section 8.2) checks the PARENT's sha256 and the five code pins. **This amendment touches none of them, so the guard needs no amendment list and no sealed code was edited:** `analyse_fresh.py` and `score_fresh.py` contain no next-arm text (the outcome is `stepA.classify`, unchanged), and `analyse_fresh.py --seal-check-only` on the parent still prints `ZZSEAL_ACCEPTED` with identical pins after this file exists (recorded in `SEALED_AT_AMENDMENT2.txt`). The amendment's own checks live in `code/amendment2_check.py` (RED-tested in `code/test_amendment2.py`): its sha256, the SEALED marker, the parent's sha, the ruling verbatim, the calculation's two sha256 pins and its branch word, and that no label or score artifact of the run is older than the seal.

### Pins
* parent `PREREG_L2_FRESH.md` sha256: `{{PARENT_SHA}}`
* the ruling `MM_RULING_L2_FRESH_D.md` (LF) sha256: `{{RULING_SHA}}`
* the design calculation, code `design_calc_ball.py` (LF) sha256: `{{CALC_CODE_SHA}}`
* the design calculation, result `design_calc_ball.json` sha256: `{{CALC_JSON_SHA}}`
* the scoping probe, code `scope_unseen_stratum.py` (LF) sha256: `{{SCOPE_CODE_SHA}}`; result `scope_unseen_stratum.json` sha256: `{{SCOPE_JSON_SHA}}`
