# PREREG ZT5 — the tracking-made DAC zeros: mechanism, an admissible execution fix, dev-set build, one-shot navtest confirm

**Written 2026-10-17 ~14:30Z, BEFORE any dev-set number and before any new navtest number below was computed.** Diagnosis +
pre-registration agent for the Master Mind (PI asked for work on the plan-selection lever). The sha256 of this file goes
to `PREREG_ZT5.md.sha256` before the first analysis run. Later changes are appended as `Amendment n` with time and
reason; the body is never edited.

## 0. What was seen before writing (and nothing else)

* ZT3's published package (`C:/Users/Admin/zt3/raw/2026-10-17-zero-tail-mechanism/`): PREREG, RESULT.json incl.
  `report`, per_token_A.jsonl (field names + its published shares), ZT4_BASELINE.json. Its headline: 144 of u400's
  343 DAC-only SEL zeros (42.0 % [33.9, 49.4]) are TRACKER_MADE; shallow (median 0.23 m), late (3.1 s), far (17.9 m).
* Prior programme results read as text: P4 F2 (`eval/RESULT_P4_TEACHER_NAVTEST.md` §5: tracker-compensated plans,
  student +1.07 [+0.68, +1.45] on navtest, NOT PROVEN, PI decision (a) ESCALATED and still open), P5/P5-S
  (compensated drawn paths fail their own interp8 check on ~13 % of rows), A7 confirm (`RESULT_A7_a7confirm_ep015.md`:
  heading vs path tangent median 0.0376 rad at 3.8 s; a last-pose path-tangent variant read +15.31 vs A7's +16.20).
* NAVSIM v1.1 source (vendored `refe/navsim_pdm/`): the LQR tracks a TIME-INDEXED reference (lateral error against the
  reference pose at the current step), its curvature feed-forward comes from the HEADING channel of the poses, the
  bicycle has first-order actuator lags (accel 0.2 s, steering 0.05 s).
* **Plumbing smoke, dev set (declared):** 3 held-out sets (log 2021.06.14.17.26.26_veh-38_05455_05749): the rebuilt
  scenario + `navsim_pdm_targets` reproduce the banked PDM labels of all 64 proposals with max |Δ| = 0.0 on NC, DAC, EP,
  TTC, C, PDMS; ~3 s metric-cache parts + ~2-3 s scoring per set; the nuPlan ego state carries tire steering 0.0 and
  yaw rate 0.0. Dev availability: 3,137 held-out sets / 24 logs banked; DBs on the dev box for **13 logs / 2,466 sets**.
  No dev pick, no dev score, no dev DAC rate, no geometry of any dev or navtest unit was read.

## 1. Status, tier, estimator, admissibility

* **Navtest is the CONFIRM set.** On navtest this package runs ONLY the DAC-level mechanism diagnostic of §3 on ZT3's
  144 tracker-made tokens and their controls (task 1 of the brief). No navtest PDMS of any fix is computed here; no fix
  constant is chosen from navtest; the one-shot confirm (§7) is written, not run.
* **Tier T1** (NAVSIM v1.1 pseudo-simulation of one open-loop plan; LQR + kinematic bicycle). Dev labels are the
  programme's own NAVSIM v1.1 PDM re-implementation on rebuilt navtrain scenarios (`refe/navsim_pdm_targets.py`,
  validated against NAVSIM's navtest cache; re-verified here by gate DG1).
* **Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (`zero_tail.Boot`, vendored
  byte-identical from ZT3). Dev clusters = the 13 dev logs; navtest clusters = navtest logs. **Every interval answers
  "another draw of LOGS"** for fixed scorers — not another training run (scorer seeds are read separately as
  replicates N-s1/N-s2, §2), not another inference run (deterministic argmax + deterministic tracker).
* **Admissibility of every fix:** inputs = the planner's own proposals and selection scores, and the ego status NAVSIM's
  agent input provides (velocity, acceleration; pose = origin). Never the map, the human future, GT labels, the metric
  cache. The re-simulation inside F_b/F_a starts from (origin, v, a, steering 0, yaw rate 0). Gate AG1 checks that this
  equals NAVSIM's initial state (steering and yaw rate are 0 in the cached ego state) on every processed dev set and
  navtest token; if AG1 fails anywhere, the fix uses the admissible state and the mismatch share is reported.
* **Real-driving rule:** a fix is ELIGIBLE only if the drawn plan stays the intended path. F_a (pre-compensation)
  deliberately draws a path off the intended one so that THIS tracker lands on it; it is measured as a CEILING and
  is not eligible (it also sits on the open PI decision (a)).

## 2. Data

**DEV (navtest-disjoint: navtrain held-out logs, nuPlan trainval):** the 2,466 held-out v5 sets on the 13 logs whose DB
is on the dev box (`D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets/onpolicy_r*_hw*.jsonl`, MD5SUMS
in that dir); fan = `model_final` ckpt_step 10075 (u400's generator: u400 = scorer_states_u400 on model_final, so its
fan is THIS fan); banked PDM labels `s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_*.jsonl` (label_version 5,
"executed (A7 repair)"). **Picks:** scorers N-s0 (PRIMARY), N-s1, N-s2 (`s1_read_cache/arms/N-s{0,1,2}/dump/
scores_update400_B.npz` = u400 + 400 SFT-4 updates, three seeds), rule `navsim_v1_lane` (u400's deployed rule,
`rl_val_eval.select`, ties to the lowest index). ⚠ u400's OWN held-out scores are not on the dev box (named limitation:
the dev picks are u400's 400-update siblings; the confirm uses u400 itself).
**Gates:** DG0 keys of sets, labels and the three score dumps join 1:1 on (log, token, step, rank, ckpt 10075);
DG1 the re-scored baseline pick (A7 executed plan) reproduces the banked label (max |Δ| ≤ 1e-5 on NC, DAC, EP, TTC, C,
PDMS) on 100 % of processed sets; DG2 the sim DAC from this package's polygons reproduces the banked DAC on all 64
proposals of every processed set (≥ 99.9 %, else the DAC-level reads are reported as unreliable). Sets whose scenario
cannot be built (5 s future guard) are counted and dropped from every arm.

**NAVTEST (diagnostic only):** ZT3's T_A tracker-made 144 u400 picks (`per_token_A.jsonl` `tracker_made`), poses from
ZT3's `jobs.npz` (A_P); CONTROLS = u400 picks of the C3 random stratum (529 non-zeroed tokens) with DAC = 1 at the
pick, post-stratified to T_A's (turn bin |Δψ_P| <15°, 15–45°, ≥45°) × (v0 bin <5, 5–10, ≥10 m/s) distribution.
Metric caches `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest`. Gate NG1: this package's base sim
DAC equals the C3 table's DAC at the pick on 100 % of processed tokens.

## 3. Mechanism (task 1)

Per unit (an executed 8-pose plan P on its ego state): R = NAVSIM-interpolated raw states (41 × 0.1 s), S = simulated
states. DAC rule = pdm_scorer's (4 corners in ROADBLOCK ∪ INTERSECTION ∪ (DRIVABLE_AREA) ∪ CARPARK_AREA). A unit is
**tracker-made** iff DAC(R) = 1 and DAC(S) = 0.

**Single-factor counterfactual re-simulations (the attribution).** A unit is RESCUED by a CF iff DAC(S_CF) = 1.
| CF | what changes | kind |
|---|---|---|
| **CF-H** | headings := path tangent (= fix F_c, §4) | admissible plan property |
| **CF-S** | initial tire steering := atan(L·κ0), κ0 = 2·y1/(x1²+y1²) of the first knot (0 if \|p1\| < 0.5 m), L = wheelbase, clipped ±π/3; yaw rate := v0·κ0 | diagnostic (NAVSIM starts every ego at steering 0) |
| **CF-V** | initial ego speed := \|p1\|/0.5 s, acceleration 0 | diagnostic (initial speed mismatch) |
| **CF-A** | actuator lags removed (accel and steering time constants 1e-6 s) | diagnostic |
| **CF-D** | (dev only) the model's own 20-pose plan, A7, linearly interpolated to 0.1 s directly instead of 8 knots | diagnostic (0.5 s resampling) |
| **CF-P** | pre-compensation (= F_a) | ceiling |

**Descriptors** (tracker-made vs controls): max over knots of |ψ_k − τ_k| (heading-tangent inconsistency, knots where
the tangent is defined); Δv0 = \|p1\|/0.5 − v0; κ0; δ = max_t ‖S_xy − R_xy‖ (self-deviation); at the step of max
deviation, the along-track and cross-track components of S − R in R's local frame (tangent from R's finite difference),
cross-track signed INSIDE (+) / OUTSIDE (−) of R's local turn (sign of R's heading change over ±1 s; |Δψ| < 2° = straight).

**Units.** NAVTEST: the 144 T_A tracker-made picks (PRIMARY for the mechanism, as named in the brief); controls as §2.
DEV: (i) tracker-made picks of N-s0/N-s1/N-s2 (deduplicated per set); (ii) M-fan: up to 4 tracker-made proposals per set
(banked DAC 0 and DAC(R) = 1), sampled with `numpy.random.default_rng(20261017)` per set in index order, each with ONE
within-set control = the DAC(S) = 1, DAC(R) = 1 proposal nearest in (|Δψ| final, path length) (L2 on (deg/10, m/5)).

**Committed reading (R-MECH).** The DOMINANT cause = the single-factor diagnostic/admissible CF (CF-H, CF-S, CF-V,
CF-A, CF-D) with the largest rescue share on the navtest 144, reported with its CI and its dev replication on (i) and
(ii). If none rescues ≥ 25 %, the reading is "DISTRIBUTED TRACKING LAG" and the descriptors name its form (corner cut =
inside cross-track with along-track lag; overshoot = outside). CF-P's rescue share is the reach ceiling of ANY
execution-side fix.

## 4. Candidate fixes (task 2) — every constant fixed here

* **F_c HEADING-TANGENT** (generalises A7; positions bit-identical). Knots p0 = (0,0) at t = 0, p1..p8 = the executed
  8 poses. k = 1..7: chord c = p_{k+1} − p_{k−1}; if ‖c‖ ≥ 1.0 m, τ_k = atan2(c). k = 8: if ‖p8 − p7‖ ≥ 0.5 m and
  ‖p7 − p6‖ ≥ 0.5 m, θa = atan2(p7 − p6), θb = atan2(p8 − p7), τ_8 = θb + 0.5·wrap(θb − θa). New heading
  ψ'_k = ψ_k + wrap(τ_k − ψ_k) where τ_k is defined, else ψ_k unchanged. No tuning.
* **F_b TRACK-CONSISTENT SELECTION (near ties only).** δ_j = max over 41 steps of ‖S_j,xy − R_j,xy‖ (rear axle,
  time-indexed) from the admissible initial state. Candidates C = {j : s_j ≥ ρ·max s}; j* = argmin_{j∈C} δ_j (ties: higher
  s, then lower index); the pick switches to j* iff δ_pick − δ_j* ≥ 0.10 m. ρ ∈ {0.99, 0.97, 0.95, 0.90}; **ρ\* is chosen
  on DEV (N-s0) as the grid value with the largest lower 95 % bound of ΔPDMS; ties → the larger ρ.** Executed plan = the
  normal A7 plan. **F_bc** = F_b with δ and execution on the F_c-repaired plans (same ρ grid, same choice rule).
* **F_a PRE-COMPENSATION (ceiling, not eligible).** P4's `compensate`: r0 = p; r ← r + (p − track(r)) at the 8 knots
  (xy, wrapped heading), 3 iterations, keep the iterate with the smallest max knot-xy deviation.

**Reach on the mechanism (task 2):** for each fix, the navtest rescue share of the 144 (DAC level, from §3) × ZT3's
tracker-made PDMS-at-stake (1.07 PDMS) = an ESTIMATED upper bound of its navtest gain on that tail (it ignores harm and
EP/C/TTC changes, which only the dev set and the confirm measure).

## 5. Dev measurement (task 3)

One pass per set (2 shards by log, balanced on set count; ≤ 2 processes): scenario → ego (AG1) → drivable polygons →
all 64 proposals: R, S (base) and the F_c versions → DAC(R), DAC(S) (DG2), δ; picks of N-s0..2; F_b / F_bc picks for
the ρ grid; §3 CFs for the dev units; then metric-cache parts and FULL PDM scoring (`navsim_pdm_targets.score_plans`)
of: each baseline pick (DG1), each pick under F_c and F_a, every distinct F_bc pick. F_b needs no scoring (banked labels
of the A7 plans). If the measured pass rate projects > 3 h wall for all 2,466 sets, full scoring runs on a log-stratified
random 50 % of sets (`default_rng(20261017)`), decided from timing only and recorded as an amendment before any dev
score is read.

**Dev statistics per fix (each for N-s0 primary; N-s1, N-s2 replicates):** ΔPDMS (fixed − baseline pick, ×100,
paired log bootstrap); Δ NC, DAC, EP, TTC, C; Δ zero rate; rescued / broken DAC zeros; drawn-plan check: share of sets
whose SUBMITTED plan's own interpolated footprint fails DAC (must not rise for an eligible fix: CI upper ≤ +0.5 pp);
**four families** — LONGITUDINAL: ΔEP, speed MAE of S vs R; LATERAL: δ, mean |cross-track| of S vs R, heading MAE of S vs
R's path tangent; TACTICAL: manoeuvre class (±30° final heading) of the new pick vs the old (F_b/F_bc only; 1.0 by
construction for F_c/F_a); STRATEGIC: PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE item 25).

## 6. Dev decision (committed)

A fix is **ELIGIBLE** iff (1) admissible and real-driving (§1; F_a never), (2) dev ΔPDMS (N-s0) lower 95 % bound > 0
AND point ≥ +0.05 PDMS, (3) the point estimate is > 0 on N-s1 AND N-s2, (4) no NAVSIM sub-score separates adversely
(upper bound < 0), (5) the drawn-plan check holds. **CHOSEN** = the eligible fix with the largest N-s0 point ΔPDMS
(F_bc competes only if F_b and F_c are each eligible). If none is eligible: the turn reports which condition failed per
fix, and names the next lever from the §3 reading (Rule Zero) — no navtest confirm is written for an ineligible fix.

## 7. Navtest confirm (task 4) — written here, run ONLY by the Master Mind, ONE shot

* **F_c chosen:** transform u400's full navtest seam (`D:/Projects/TanitAD/data/refe_navtest/seams/
  stage2_sft4_B_u400_navtest_full.npz`, 12,146 tokens) with `code/zt5_seam_fix.py --fix F_c`, score it with
  `eval/score_navtest_refe.py` on the whole navtest (no `--tokens`; WinError 206), and pair with u400's banked CSV
  (`refe_stage2_sft4_B_u400_navtest_full.csv`) by `code/zt5_confirm_read.py`.
* **F_b / F_bc chosen:** lookup on the C3 table (1,058 tokens, all 64 proposals harness-scored) + δ from the navtest
  metric caches' ego states; stratified estimator (zero census weight 1, random stratum weight 11,617/529).
* **Statistic:** per-token PDMS(fix) − PDMS(u400), mean over navtest, paired log-cluster bootstrap over the navtest logs,
  10,000, seed 20260927. **ADOPT iff lower bound > 0 AND no NAVSIM sub-score (NC, DAC, EP, TTC, C) separates
  adversely; REFUTED iff upper bound < 0; else NOT PROVEN.** ADOPT ⇒ the fix becomes the next planner amendment
  (`REFePlanner.executed` for F_c; selection for F_b) as a deployment CANDIDATE (PI's call); NOT PROVEN / REFUTED ⇒
  not adopted, and the §3 reading names the next lever. Reported, not gating: the change in ZT3's 144 tracker-made
  zeros; the same statistic on `navtest_unseen_by_sft4u400_tokens.json`.

## 8. Mutation controls (each new metric goes RED when broken)

`code/test_zt5.py`: F_c positions bit-identical and the tangent exact on a circular arc (mutant: chord k→k+1 instead
of central → RED); F_c identity on a straight line; δ = 0 for the tracker's own output fed back? (no — δ ≥ 0 and δ of a
plan shifted 500 m vs its own R is unchanged: translation invariance; mutant: absolute vs relative → RED); F_b with
ρ = 1 equals the baseline pick bit-for-bit (mutant: candidate set ignores ρ → RED); bootstrap self-pair = exactly 0;
the CF switch with no factor changed reproduces the base sim bit-for-bit (mutant → RED).

## 9. What is NOT done

No commit, no push, no pod, no GPU, no navtest PDMS of any fix, no A40.

---
## Amendment 1 (2026-10-17 ~14:55Z — written while the navtest and dev jobs run, BEFORE any mechanism, rescue, descriptor, PDMS or DAC outcome of either was read; only gate booleans (AG1, NG1, DG1, DG2) and timings of smokes were seen)

1. **Navtest control filter (implementation bug, caught before any read).** The C3 `tokens.json` labels strata `zero` /
   `random`, not `zero_census`. The running `zt5_nav.py` therefore also enqueued the 113 zero-stratum picks with DAC = 1
   (NC-only zeros) as CTRL units. They are NOT controls under §2. `zt5_analyze.py` keeps only `stratum == "random"`
   CTRL records (the record carries its stratum), which is exactly §2's set; the code is fixed for any rerun
   (`== "random"`). The 113 extra units are counted and reported, never used.
2. **Dev M-fan cap 4 → 2 per set**, each still with one within-set control, decided from the measured smoke rate
   (~4–5 s per set with 2 processes) so the full 2,466 sets fit the §5 3-hour rule without the 50 % fallback.
   Controls get CF-H (the fix-harm read) and the descriptors; the other CFs run on tracker-made units only.
3. **CF-P on navtest** reads `P_drawn` too (the compensated plan's OWN interpolated footprint DAC), as the drawn-path
   evidence for the real-driving rule.

---
## Amendment 2 (2026-10-17 ~15:05Z — AFTER the navtest mechanism read (§3, DAC level), BEFORE any dev outcome was read)

The navtest read named CF-H dominant; a POST-HOC descriptive (`data/posthoc_signed_heading_navtest.json`) shows the
interior knots consistent with the path tangent (median signed error ≈ 0°) and the LAST knot (A7's 3.8 s heading at
t = 4.0 s) under-rotated in ~70 % of turning picks. **Added, REPORTED ONLY, deciding nothing:** the split of F_c into
F_c8 (tangent at knot 8 only) and F_c17 (knots 1–7 only), as DAC rescue on (i) the navtest 144 (labelled POST-HOC) and
(ii) the dev tracker-made picks and M-fan units (registered here, before the dev read). The §6 decision is unchanged:
F_c as defined in §4 is the only heading fix that can be chosen.

---
## Amendment 3 (2026-10-17 ~16:35Z — AFTER the §6 dev decision (NO FIX ELIGIBLE) and the POST-HOC tracker-saved count; BEFORE any number of the arm below exists) — the NEXT ARM (Rule Zero)

**Why.** Dev: F_c rescues tracker-made zeros but breaks more compliant picks than it rescues; the F_a ceiling nets
≈ 0 on DAC (rescued 13 / broken 14). POST-HOC (`data/posthoc_tracker_saved.json`): at the PICK, tracker-SAVED
(own footprint fails, simulated passes) ≈ tracker-MADE (dev N-s0 27 vs 20; navtest ≈ 132 ESTIMATED vs 144). The scorer,
trained on simulated-state labels, already selects for NAVSIM's execution, so faithful tracking has ~0 net reach; the
zero tail is a MARGIN problem (ZT3: near-twin grazes, median depth 0.23 m). The cheapest admissible map-free margin
proxy is the fan itself: the scorer's near-tie band brackets the corridor it believes drivable.

**F_m FAN-CONSENSUS MARGIN SELECTION** (admissible: proposals + scores only; executed plan = the normal A7 plan, so
the drawn plan is a model proposal and the real-driving rule holds). C = {j : s_j ≥ ρ·max s}. If |C| ≥ 3: reference
r_k = score-weighted mean of the candidates' 8 executed knots (xy); d_j = mean over k of ‖p_j,k − r_k‖; j* = argmin_C d_j
(ties: higher s, lower index); switch iff d_pick − d_j* ≥ 0.10 m; else keep the pick. ρ ∈ {0.99, 0.97, 0.95, 0.90},
ρ* chosen on dev N-s0 by the largest lower bound of ΔPDMS (ties → larger ρ). Evaluated as a LOOKUP on the banked dev
labels (no re-simulation: the executed plan is unchanged). **Eligibility and choice: §6 verbatim** (drawn-plan check =
banked-free: the new pick is an A7 proposal). **If eligible**, its navtest confirm is the C3-table lookup of §7
(stratified: zero census weight 1, random stratum 11,617/529; u400's own logits and the harness's PDMS of every
proposal; one shot, run by the Master Mind; ADOPT iff lower bound > 0 and no sub-score adverse; REFUTED iff upper < 0).
**If not eligible**, the execution/selection side is closed for the zero tail at this scorer, and the named next lever
is ZT3's: the boundary representation (P6 trunk DAC gradient → ZT4, blocked on the P6 checkpoint).
