# RESULT A5 — a TIME-LOCALISED nav input for the pick: T2 misses its replication criterion, T3 clears, the control hurts

**Stamp on every number below.**
* **Evidence class:** MEASURED. Source: `raw/a5_nav_tl.json` (every arm, both draws, every control), tables generated from it by
  `code/make_tables_a5.py` → `raw/TABLES_a5_generated.md`. Code: `code/nav_tl_a5.py` (md5 in `LANDING_READY_A5.txt`).
* **Pre-registration:** `SPEC_ADDENDUM_A5.md`, sha256 `5f102584…`, registered 2026-10-04T10:01:55Z; the script refuses to run if the file on disk
  differs. No arm, horizon, reported arm or bar was changed. Judgement calls the SPEC leaves open are listed in §7.
* **Tier:** OPEN-LOOP single-shot planning on logged frames of the held-out eval139 clips (EVAL grid, 8 windows per episode = 1,112 windows,
  139 episodes). Never closed loop (EVAL_DOCTRINE). Not a driving-performance claim.
* **Model:** refcv7-r101-s0, step 50,400, **one training seed**. **Inference replicate:** sampler seed 1 (`eval_s1`) on the same windows.
* **Nav is an ORACLE** (v8 `nav_30s`, provenance ego-future) — an inference-time route-following gain under that caveat, not a training lever.
* **Intervals:** paired episode-cluster bootstrap vs V0 (B = 2000, the SPEC §5 draws). It answers *"another draw of EPISODES"* only; the seed-1
  column answers the INFERENCE question; the TRAINING question is not measured (`H-ESTIM-SEED-1`) — a separated CI is necessary, not sufficient.
* **Order of work:** K0/K1/K2 and the token tally were produced and read BEFORE any arm was scored; the clip-token reproduction controls had to pass
  before any A5 arm was scored (`controls_all_pass: true`).

## 1. Bottom line

1. **Controls: all PASS.** K1 max |diff| **5.0e-5 s** over 2,059 bank windows (tolerance 1e-3); K2 3/3 captured windows (+ 17/17 on a dense supplement).
   The clip-token reproductions of V2, V3, the A2 X1 refit and B1t equal the banked numbers to **0.0** — the instrument is the SPEC §5 instrument.
2. **T2 (REPORTED ARM) FAILED the bar — on criterion 4 (replication) only.** Seed 0: turn ΔADE **−0.264 [−0.568, −0.0005]**, turn dir-correct
   **+0.093 [+0.031, +0.171]**, straight ΔADE **+0.009 [+0.001, +0.020]**, all-window ΔADE **−0.043 [−0.089, −0.005]** — criteria 1–3 hold, but
   criterion 1 clears by **0.45 mm** of CI margin. Seed 1: turn ΔADE −0.204 **[−0.447, +0.028]**, turn dir-correct +0.047 **[+0.000, +0.108]** → does not replicate.
3. **T2c (control) fails as required — and is harmful**: all-window ΔADE **+0.089 [+0.037, +0.146]**, straight +0.106 [+0.051, +0.166].
4. **Reading-rule branch: NONE of the four named branches applies** (T2 neither passes, nor fails on straight damage, nor shows "no turn gain"; T2c does not pass).
   Applied literally the rule says nothing; §5 gives the facts a decision needs.
5. **T3 (secondary arm: soft nav-compliance term ×10 with the time-localised predicate) CLEARS all four criteria** on EVAL, and its sign repeats on a
   second, non-held-out episode sample (post-hoc, §6). T3 is not the reported arm; this is a lead, not a verdict.
6. **Scale:** T2's turn gain is **0.467** of A4's B1t bound (ΔADE_turn −0.264 / −0.566); per-draw ratio bootstrap 95 % [0.001, 0.853].
7. **Attribution warning (post-hoc):** **54 % of T2's turn gain (−0.143 of −0.264) sits on 7 windows where the SHIPPED clip token was FOLLOW** and
   `nav_30s` lists a turn — extra information, not timing. The 45 windows where the clip token was already active contribute −0.121.

## 2. Controls, token tally, window counts (read first)

| control | expected | measured | verdict |
|---|---|---|---|
| **K1** computed `t_now_raw` vs the replay bank's `t_label_s` | ≤ 1e-3 s on every window | **5.0e-5 s** max over 2,059 windows / 12 clips (the bank rounds to 4 dp, so a correct clock reads ≤ 5e-5); W re-derived from the bank (`now_row − t_start_row` = 7 on every row) = **8** | PASS |
| **K2** nav_tl == clip-token side at t_rel ∈ [−0.05, +0.05], `args.time_s` ≤ 6.0 | 100 % | **3/3** captured windows (reel). ⚠ the EVAL grid itself has **0** windows in the band (VACUOUS there), the TRAIN grid 0. Supplementary (one integer t per eval clip at t_rel ≈ 0 on the clip's own clock; NOT a captured window): **17/17** | PASS |
| K0 join: bank `nav` == record `nav_command` side | 100 % | 1,112/1,112 on eval_s0g, eval_s1 and train_s0 | PASS |
| K0 sidecar: the 3 eval clips with no row | = the run config's 3 `tactical_excluded_sids` | equal; fallback (0.0, 0.1) verified from their cached poses (`raw/a5_fallback_clock_check.txt`: 0 / 5 / 0 moving steps < 10 ⇒ `pose_dt` None ⇒ nominal) | PASS |
| reproduction (clip token through THIS code): V2|0.05, V3|10, A2 X1 refit, B1t vs the banked tables | ≤ 2e-4 | **0.0** each; X1 refit α and 5-fold CV ADE identical | PASS |
| recomputed navc term (clip token) == the shipped term | ≤ 1e-9 | 0.0 (eval, seed 1, train) | PASS |
| unit tests `code/test_nav_tl_a5.py` | literal | 17/17 (+ 7/7 of `test_route_metrics.py`); mutation H = 1e9 changes the answer at t_rel = −4 | PASS |

**Token tally (139 eval clips).** Clip token (`nav_command`): 88 follow / 13 left / 38 right. `nav_30s` entry tokens: 85 FOLLOW_ROAD / 52 TURN_R / 21 TURN_L
(123 clips with 1 entry, 13 with 2, 3 with 3). **`nav_command` and `nav_30s.entries[0]` disagree on 9 clips**: 6 × (command FOLLOW vs a TURN_R
entry: 4 × "curve, not a turn", 2 × "contested turn (obstacle pass) — nav does not command a turn (PI 2026-08-29)"), 2 × (command TURN_R vs FOLLOW-only entries),
1 × (TURN_L vs FOLLOW-only). Window reasons (1,112): follow-token 680, turn > H ahead 280, ahead ≤ H 84,
under way 43, all finished 25; t_rel range [−6.12, +9.42] s.

**Window counts (EVAL grid).**

| window class | n | nav_tl L / follow / R | clip token L / follow / R | nav_tl non-follow | clip-token non-follow |
|---|---|---|---|---|---|
| **GT-turn** | 107 | 12 / 55 / 40 | 13 / 57 / 37 | **52** | 50 |
| GT-turn left / right | 40 / 67 | 11 / 28 / 1 · 1 / 27 / 39 | 13 / 24 / 3 · 0 / 33 / 34 | 12 · 40 | 16 · 34 |
| **GT-straight** | 588 | 4 / 572 / 12 | 52 / 395 / 141 | **16** | **193** |
| gentle | 105 | 6 / 83 / 16 | 11 / 57 / 37 | 22 | 48 |
| unclassified | 312 | 5 / 275 / 32 | 28 / 195 / 89 | 37 | 117 |
| all windows | 1,112 | 27 / 985 / 100 | 104 / 704 / 304 | 127 | 408 |

nav_tl vs the geometric GT direction on GT-turn windows: **50 correct side, 55 follow, 2 opposite** (clip token: 47 / 57 / 3). On GT-straight windows the
time-localised nav is active on **16 of 588** where the clip token was active on **193** — that is the straight damage V2 paid, removed by construction.
H = 8 s adds no GT-turn window (52 active, same as H = 6) and 17 straight ones (33 vs 16).

## 3. The bar (EVAL; paired episode-cluster bootstrap vs V0; seed 1 = sampler replicate)

| arm | picks changed | turn ΔADE [CI] | turn dir-correct Δ [CI] | straight ΔADE [CI] | all ΔADE [CI] | seed 1: turn ΔADE [CI] | seed 1: turn dir-correct Δ [CI] | criteria 1/2/3/4 | bar |
|---|---|---|---|---|---|---|---|---|---|
| **T2 (REPORTED)** | 0.037 | −0.264 [−0.568, −0.0005] | +0.093 [+0.031, +0.171] | +0.009 [+0.001, +0.020] | −0.043 [−0.089, −0.005] | −0.204 [−0.447, **+0.028**] | +0.047 [**+0.000**, +0.108] | Y/Y/Y/**N** | **FAILED** |
| T3 (a: term replaced) | 0.020 | −0.159 [−0.312, −0.040] | +0.065 [+0.020, +0.124] | +0.011 [+0.002, +0.025] | −0.014 [−0.040, +0.010] | −0.143 [−0.286, −0.032] | +0.056 [+0.011, +0.113] | Y/Y/Y/Y | **CLEARS** |
| T3b (b: term added; sensitivity) | 0.017 | −0.159 [−0.312, −0.040] | +0.065 [+0.020, +0.124] | +0.011 [+0.002, +0.025] | −0.014 [−0.040, +0.010] | −0.143 [−0.286, −0.032] | +0.056 [+0.011, +0.113] | Y/Y/Y/Y | CLEARS |
| T4 refit linear re-scorer | 0.053 | −0.006 [−0.171, +0.138] | +0.009 [−0.021, +0.045] | −0.034 [−0.076, +0.000] | −0.018 [−0.059, +0.019] | +0.013 [−0.120, +0.162] | +0.000 [−0.026, +0.027] | N/Y/Y/N | FAILED |
| **T2c CONTROL** (derangement) | 0.088 | −0.071 [−0.351, +0.135] | +0.009 [−0.035, +0.062] | **+0.106 [+0.051, +0.166]** | **+0.089 [+0.037, +0.146]** | −0.029 [−0.233, +0.150] | +0.000 [−0.041, +0.050] | N/N/N/N | **FAILED (required)** |
| T2h8 (H = 8, sensitivity) | 0.057 | −0.264 [−0.568, −0.0005] | +0.093 [+0.031, +0.171] | +0.016 [−0.009, +0.043] | −0.039 [−0.082, −0.000] | −0.204 [−0.447, +0.028] | +0.047 [+0.000, +0.108] | Y/Y/Y/N | FAILED |
| B1t (A4 label-side bound) | 0.015 | −0.566 [−0.986, −0.239] | +0.159 [+0.080, +0.258] | 0.000 | −0.076 [−0.132, −0.029] | −0.546 [−0.945, −0.233] | +0.149 [+0.076, +0.245] | – | bound |
| ORACLE-117 (bound) | 0.597 | −1.995 [−2.537, −1.548] | +0.121 [+0.057, +0.202] | −0.925 [−1.102, −0.754] | −1.175 [−1.343, −1.007] | −1.986 | +0.103 | – | bound |

For scale (INHERITED from RESULT.md §2, reproduced exactly by the control): **V2 with the clip token** read turn ΔADE −0.125 [−0.305, +0.055], turn dir-correct
+0.065, **straight +0.326 [+0.197, +0.459]**, all +0.225. Time-localising the same rule removed the straight damage (+0.326 → +0.009) and doubled the turn gain.

T4: the refit selector moves nothing — 5-fold CV ADE 1.4650 (α 0.01) vs A2's clip-token X1 1.4693 vs V0 1.4838; the nav features keep a weight of 0.078
against the sampler-confidence weight of 1.73. The re-scorer cannot use a better nav signal, which is A2's finding again.

## 4. Four metric families (SPEC §2.1 format; EVAL seed 0; per family, never pooled)

| GT-turn (107) | ADE | FDE | LON \|along\| 6 s | LON along 6 s signed | LON speed MAE 0–2 s | LAT \|cross\| 6 s | LAT heading MAE 0–2 s ° | LAT curv MAE | LAT term. heading err ° | TAC dir correct | STR nav compl. (clip token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 3.186 | 10.504 | 7.418 | +3.272 | 0.312 | 6.182 | 4.006 | 0.010 | 22.369 | 0.841 | 0.740 |
| T2 | 2.922 | 9.623 | 6.426 | +2.415 | 0.290 | 5.964 | 3.928 | 0.009 | 20.074 | 0.935 | 0.900 |
| T3 | 3.028 | 9.928 | 6.937 | +2.901 | 0.302 | 5.882 | 3.947 | 0.009 | 20.607 | 0.906 | 0.860 |
| T2c | 3.115 | 10.296 | 7.192 | +2.725 | 0.302 | 6.089 | 3.992 | 0.010 | 22.878 | 0.851 | 0.740 |
| B1t | 2.620 | 8.629 | 6.233 | +2.189 | 0.289 | 5.019 | 3.576 | 0.008 | 17.217 | 1.000 | 0.940 |
| ORACLE | 1.191 | 3.598 | 2.194 | +0.266 | 0.217 | 2.402 | 3.405 | 0.007 | 13.404 | 0.963 | 0.880 |

| GT-straight (588) | ADE | FDE | LON \|along\| 6 s | LON along 6 s signed | LON speed MAE | LAT \|cross\| 6 s | LAT heading MAE ° | LAT curv MAE | LAT term. heading err ° | straight-keeping | STR nav compl. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 1.656 | 5.220 | 4.824 | +0.426 | 0.246 | 1.079 | 0.833 | 0.002 | 2.348 | 0.964 | 0.021 |
| T2 | 1.666 | 5.242 | 4.820 | +0.413 | 0.246 | 1.125 | 0.848 | 0.002 | 2.453 | 0.954 | 0.047 |
| T3 | 1.667 | 5.261 | 4.849 | +0.403 | 0.247 | 1.105 | 0.841 | 0.002 | 2.479 | 0.956 | 0.047 |
| T2c | 1.762 | 5.627 | 4.944 | +0.102 | 0.245 | 1.540 | 0.940 | 0.002 | 3.656 | 0.918 | 0.062 |
| ORACLE | 0.731 | 2.124 | 1.796 | −0.156 | 0.162 | 0.757 | 0.802 | 0.002 | 2.228 | 0.954 | 0.026 |

TACTICAL (`four_families.tactical_from_trajectory`, 0–2 s, 1,112 windows) lateral κ / longitudinal κ: V0 0.8229 / 0.5395, T2 0.8250 / 0.5545, T3 0.8281 / 0.5522,
T2c 0.8239 / 0.5531, ORACLE 0.8512 / 0.8530. The gains are lateral (terminal heading −2.3° on turns, cross-track −0.22 m) and along-track at 6 s (|along| −1.0 m),
not 0–2 s speed. LONGITUDINAL distance keeping: UNAVAILABLE (no lead tracks). STRATEGIC decision: UNAVAILABLE (`--no-strategic`); the nav-compliance column is
measured against the CLIP token as in RESULT.md §2.1. All arms, all classes (turn / straight / gentle / all) are in `raw/a5_nav_tl.json`.

## 5. The reading rule, applied (the rule was coded before any arm was scored)

| A5 branch | applies? |
|---|---|
| T2 passes, T2c fails ⇒ the defect is largely nav TIMING; L2 enters refcv8 AND ships as an inference rule | **No** — T2 did not clear criterion 4 |
| T2 fails on straight damage despite time-localisation | **No** — straight ΔADE +0.009 [+0.001, +0.020] |
| T2 fails on turns (no turn gain) | **Not as worded** — there IS a turn gain on both draws, same sign and size (−0.264 / −0.204; draw difference −0.039 [−0.102, +0.005]); it is not CI-separated on the replicate |
| T2c passes ⇒ the instrument is broken | **No** — T2c fails and hurts |

**No named branch applies; the Master Mind chooses.** The facts that bear on it:
* *For "timing matters":* the true series beats a mismatched series (paired T2 − T2c ADE, post-hoc): all windows **−0.132 [−0.197, −0.072]**, straight −0.097
  [−0.158, −0.041], turn −0.193 [−0.426, +0.016] (seed 1: −0.146 / −0.108 / −0.188). A wrong-in-time nav is worse than none; a right-in-time nav is not worse than none.
* *Against reading T2 as a confirmed gain:* the seed-0 criterion-1 margin is 0.45 mm; T2 changes **14 of 107** GT-turn picks (10 better, 4 worse, in 11 episodes) and the 5
  largest gains outweigh the whole sum (−28.9 vs −28.3); on the TRAIN-DIAG second sample (§6) T2's turn ΔADE is **+0.158 [−0.101, +0.477]**.
* *For the soft form:* T3 changes 7 turn picks (6 better, 1 worse, 6 episodes) but with a tighter CI on both draws, and repeats on TRAIN-DIAG.

## 6. Post-hoc attribution (NOT pre-registered; never a bar result)

* **Both sampler draws pooled** (per-window ΔADE averaged): T2 turn **−0.245 [−0.523, +0.006]**, straight +0.008 [−0.003, +0.019], all −0.041 [−0.087, −0.003];
  T3 turn −0.156 [−0.309, −0.037], straight +0.011, all −0.014 [−0.039, +0.008].
* **Pure timing vs extra information** (GT-turn windows, seed 0; contribution to the 107-window mean):

| part | n | T2 | T3 |
|---|---|---|---|
| clip token active AND nav_tl active (timing only) | 45 | −0.121 | −0.107 |
| clip token FOLLOW but nav_tl active (`nav_30s` lists a turn that `nav_command` suppressed: "curve, not a turn" / "contested turn") | 7 | **−0.143** | −0.052 |
| clip token active but nav_tl follow (time-localisation switches it off) | 5 | 0.000 | 0.000 |
| both follow | 50 | 0.000 | 0.000 |

* **How much of the GT-turn set can nav touch at all?** nav_tl is active on 52 of 107 GT-turn windows. On those 52, T2 reaches −0.544 [−1.119, −0.001] vs the
  B1t bound −0.817 [−1.416, −0.306] (67 %). On the **55 windows where nav is silent** B1t would still give **−0.329 [−1.046, +0.007]**: no nav signal can reach that
  half; it needs a vision-only turn cue (the tactical-label lever, RESULT.md §2.3 R-TAC-LABEL).
* **A second sample for the RULE** (T2 and T3 have no fitted parameter, so the captured TRAIN-DIAG fans — 139 OTHER episodes — can be scored; the checkpoint was TRAINED
  on those clips, so this is NOT held-out and carries no verdict): turn ΔADE **T2 +0.158 [−0.101, +0.477]**, **T3 −0.119 [−0.221, −0.042]** (dir-correct +0.096 [+0.047, +0.154],
  straight +0.001), T2c +0.578 [+0.125, +1.059] (all +0.245 [+0.143, +0.364]). The hard filter's sign flips between samples; the soft term's does not.

## 7. Judgement calls and what I could not do

* **T3, "the predicate recomputed against nav_tl".** The SPEC does not say whether the shipped 1× clip-token term stays in the score. T3 = reading (a): the term built
  from the clip token is removed and replaced by k × gate × 1[dir == nav_tl side]; T3b = reading (b): the shipped term stays and (k−1) × the new term is added (the literal V3
  code with the term swapped). Fixed before scoring; both clear identically on seed 0 and to ±0.001 on seed 1.
* **T2c.** "Each clip's nav_tl series taken from another clip" is implemented as: the donor clip's `nav_30s` entries evaluated at the RECIPIENT window's own t_rel (a seeded
  derangement of the 139 clips; digest in the JSON). 218 of 1,112 windows get a different side from the true series; 22 are active in both.
* **nav_tl reads `nav_30s.entries`, as written**, not `nav_command`. They disagree on 9 clips (§2), which makes part of T2's gain information rather than timing (§6).
* **K2 is thin.** The EVAL grid never lands within ±0.05 s of the anchor (0 windows); the captured evidence is 3 reel windows plus a 17-clip dense supplement that is not a captured window.
* **K1 validates my clock against the replay tool's `t_label_s`**, which came from the trainer's own `_now_s` path; it does not re-validate the sidecar against physical time
  (that is the run's own G3 check, INHERITED).
* **Not done:** no new GPU capture (CPU only; the dev-box GPU and Thor GPU were not touched), no second training seed, no closed-loop evaluation. UNVERIFIED: whether T3's
  soft-term behaviour holds on windows outside this 139-episode grid.
* **Estimators.** Every interval is the paired episode-cluster bootstrap; 38 of the 139 episodes contain a GT-turn window, so the turn-window CIs are set by about 38 clusters.

## 8. Rule Zero — what comes next

* **T2 failed its bar; the next lever is already measured:** T3 (same nav_tl, soft ×10 term) clears the bar and repeats in sign on a second episode sample. It was a pre-registered
  secondary arm, so scoring it is not selection, but the reading rule is silent about it and only T2 was the reported arm.
* **Cheapest experiment that could still settle it:** one pre-registered addendum (A6) naming T3 as the reported arm, T2 as its hard-filter foil and T2c as the control, scored on a
  DENSER capture of the 139 held-out eval episodes (all GT-turn windows ≈ 2,300 ESTIMATED from 107 / 1,112 × 23,772, plus a random straight sample; ~0.39 s per window forward,
  MEASURED in RESULT.md §0 ⇒ ~15–30 min of Thor GPU). ⚠ densifying windows does not add episodes: 38 turn episodes remain, so it narrows within-episode noise only.
  **Blocked on:** a Thor GPU slot and the Master Mind's A6 SPEC (label-side window selection is class-conditional by construction, so only per-class deltas are admissible).
* **Do not wait on it for the recipe question:** T2c hurts and both soft and hard time-localised forms are same-signed on both sampler draws, so L2 (time-localised nav as a
  TRAINING input, with `nav_30s` as the source rather than `nav_command`) stays a refcv8 candidate. The label finding — `nav_command` suppresses a turn on 6 of 139 eval clips (4 "curve, not
  a turn", 2 "contested turn") that `nav_30s` lists, and those clips hold 7 of the 107 GT-turn windows — belongs in the refcv8 data audit.
* **The other half of the turn gap is not a nav problem:** 55 of 107 GT-turn windows have no nav signal in range (§6).

## 9. Register rows proposed (the Master Mind owns `GOALS_AND_CLAIMS.md` / `RETRACTION_LOG.md`)

* `H-A5-NAVTL-T2`: T2 FAILED (criterion 4, 2026-10-04); seed 0 turn ΔADE −0.264 [−0.568, −0.0005], seed 1 [−0.447, +0.028]; a 14-pick effect; MEASURED `raw/a5_nav_tl.json`.
* `H-A5-NAVTL-T3`: T3 cleared its bar on EVAL (secondary arm); needs the A6 denser scoring before it is quotable as a lever; one training seed (`H-ESTIM-SEED-1`).
* `D-A5-NAV-SOURCE`: `nav_command` vs `nav_30s.entries[0]` disagree on 9/139 eval clips (6 suppressed turns: 4 "curve, not a turn", 2 "contested turn"; the suppression is deliberate in the builder, so using `nav_30s` as a source is a labelling choice, not a bug fix).
