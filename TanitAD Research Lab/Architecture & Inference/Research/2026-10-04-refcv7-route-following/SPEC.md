# SPEC — refcv7-r101-s0: where the route-following chain breaks, the fan, the boxes (pre-registration)

**Written 2026-10-04 before any chain, fan, lever or NMS number was computed.** sha256 + UTC time in
`raw/SPEC_SHA256.txt`. Anything added after that time is logged in RESULT.md §"post-hoc" and is not a
pre-registered result.

**Trigger (PI, verbatim, after the final-checkpoint reel):** *"The boxes are completely messy. The planner is
not following the route as shown in the roundabout example and the turn left example … the fan is sometimes
completely messy."*

**Model.** refcv7-r101-s0 final, Thor `/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt`, step **50,400**, md5
`d5f104ee54ba6b2861e38030b4f6fcf1`; launch tree `/home/nvidia/refcv7_run/fec3a0dccf` (its own
`stack/tanitad/eval/refcv7_loader.py`, STRICT load, step asserted). One training seed.

**Tier / scope stamp (travels with every number).** OPEN-LOOP single-shot planning on logged frames of the
held-out eval139 clips (perception context fixed by the recording; no rollout; never closed loop — EVAL_DOCTRINE).
The nav token fed at inference is the v7.2 `nav_command` **ORACLE (provenance: ego future), one token per CLIP**
(`config.json nav_cmd_derivation`) — a supplied route, optimistic by construction. **Speed ceiling:** on this
launch tree the ceiling filters the DECODER's argmax (`sel_idx_base`) but NOT E9's emitted pick (`sel_idx`,
SPEC_REFCV7 §26.1). Every pick-level finding is reported for the shipped E9 pick AND for the §26.1-fixed rank
(E9 argmax over `reach_keep ∧ ceil_keep`, emulated post-hoc on the same captured scores) so "does it matter" is
measured, not argued.

## 1. Window sets

| set | definition | use |
|---|---|---|
| **EVAL** | the map-box diagnostics' EVAL-DIAG grid: 139 eval episodes × 8 windows at positions `(j+0.5)·n/8` (`2026-10-04-refcv7-map-box-diagnostics/code/run_diag.py`, `--split eval_diag`) = 1,112 windows | every reported result |
| **TRAIN** | the same grid on the diagnostics' 139-episode TRAIN-DIAG subset (`REFCV6_REMAP_OVERRIDES=…/cache/remap_train_diag.json`) | fitting lever parameters and NMS radius/IoU ONLY |
| **REEL** | the 12 reel clips, every window (2,059), from the reel bank `…/video/bank/final/rows.jsonl`, plus a re-forward of those windows with the reel's own seeding (batch 1, `torch.manual_seed(0)` per window) to recover the full 117-candidate fan | links (a)(b)(d) from the bank; (c)(e)(f) from the re-forward; the reel anecdotes |

Control C4: my EVAL window digest (`sha256(json([(sha12, t)…]))`) must equal `raw/run_eval_final.json
window_sha12_t_sha256` of the diagnostics package.

## 2. The forward and the capture (read-only; no model code changed)

Batch 1; `torch.manual_seed(seed)` immediately before each window's `compute_losses_v3` (eval mode, no_grad) —
the reel's own forward. **seed 0 = primary; seed 1 = the INFERENCE replicate on EVAL** (the DDIM sampler is
stochastic: the reel's draw-independence control moved `traj` by 0.58–1.15 m between seeds 0/1, so a third
variance — inference — rides on every plan number; H-ESTIM-SEED-1 / D-REFAV1-SEED-GOAL-MISMATCH).

Captured per window by a forward hook on the model and by call-through wrappers (each wrapper calls the
original and returns its value unchanged):
* `out`: `anchor_traj` [117, 8, 2] (the fan), `sel_score` (decoder rank score, unmasked), `sel_score_v3` (E9
  blended), `reach_keep`, `sel_idx_base` (decoder pick), `sel_idx` (E9 pick), `traj`, `tacv6_lat/lon_logits`,
  `tacv6_goal_logits`, `goal_point_tac`, `sel_tele`;
* the decoder's `_apply_grafts` per surface (`conf`, `refined`, `rank`): base, every term, result;
* `refcv6_selection.nav_compliance_prior` output (the navc predicate) and `navc_gate`;
* the E9 `apply_seam_clamp(surface="goal_sel")` base / graft / result; `goal_gate`;
* the speed-ceiling keep mask, recomputed with the model's own `SpeedCeilingFilter` and fed limit;
* GT: `refb_labels.waypoint_targets` at the 8 slots (5,10,15,20,30,40,50,60 ticks) + validity, `lat_v7`,
  `lon_v7`, nav token; box3d slots (presence logit, box x/y/l/w, yaw, class) and the trainer's own
  `_det_pack_box3d` / `_det_pack_agent` packs.

## 3. Path geometry (one definition for GT, candidates and picks)

Ego frame at NOW (x forward, y LEFT, metres). For a path `P` at the 8 slots, the **terminal heading**
`θ(P) = atan2(Δy, Δx)` of the slot-50 → slot-60 segment (5 → 6 s), `θ = 0` when that segment is < 0.05 m —
exactly `refc_selector_targets.compliance_target`'s predicate (the run's own nav-compliance definition), so
"turning" means the same thing for the label, the candidate and the graft. `dir(P; τ) = L if θ ≥ τ, R if θ ≤ −τ,
else S`.

* **GT-turn window**: slot 60 valid, GT path length to slot 60 ≥ 5 m, `|θ_gt| ≥ 30°`. Direction = sign.
* **GT-straight window**: slot 60 valid, length ≥ 5 m, `|θ_gt| < 10°`.
* **GT-gentle**: 10° ≤ |θ_gt| < 30° — reported, not used by the break rule. Everything else = unclassified (n given).
* **Turn-correct (primary)**: `dir(P; τ_c) == dir(GT)` with `τ_c = 0.18063741505146028 rad` (10.35°, the run's own
  `--nav-compliance-tau-rad`). **Heading-agree (secondary)**: `|wrap(θ_P − θ_gt)| ≤ 15°`.
* ADE = mean L2 over the VALID slots of the 8; FDE = L2 at slot 60 when valid (the reel's `plan_errors`).
* **Oracle candidate** = argmin ADE over the 117 (label-side, never an inference input).

## 4. The chain links (on GT-turn windows; each also reported on GT-straight and GT-gentle)

| link | turn-correct means |
|---|---|
| N  nav input | clip nav token side (left→L, right→R) == GT direction; `follow` = uninformative (counted separately) |
| T  tactical lat | side of `argmax tacv6_lat`: {LANE_CHANGE_L, NUDGE_L, TURN_L}→L, {…_R}→R, {LANE_KEEP, ABORT_LC}→S; == GT direction. Strict variant: argmax == TURN_L/TURN_R. Label control: `lat_v7` side vs geometric GT |
| G  goal tokens | `p(TURN_L)` vs `p(TURN_R)` (sigmoid): side of the larger if it is ≥ 0.5, else S; == GT direction |
| C  fan containment | ∃ candidate among the 117 with `dir(·; τ_c) == dir(GT)`; also restricted to `reach_keep` (the set E9 ranks); oracle best-of-117 ADE and the oracle candidate's turn-correctness |
| K  decoder pick | `sel_idx_base` turn-correct |
| E  emitted pick | `sel_idx` (E9) turn-correct; also the §26.1-fixed rank |
| e  nav-compliance graft | per window: # candidates satisfying the predicate (what a hard filter would keep), whether the oracle candidate satisfies it, `navc_gate`, the term's size vs the score spread (`navc_gate / std(score)`), and the # windows whose decoder argmax changes when the term is removed (counterfactual on the captured score) |
| f  E9 score | rank of the oracle candidate and of the best turn-correct candidate under the E9 blended score (within `reach_keep`), score gap oracle − pick |

Chance control R: the turn-correct rate of a uniformly random `reach_keep` candidate (expectation per window).

**DECISION RULE — where the chain breaks.** On EVAL GT-turn windows, with rates `r_T` (tactical, primary side
coding), `r_C` (containment in `reach_keep`), `r_K`, `r_E`:
* drops `D_gen = r_T − r_C`, `D_core = r_C − r_K`, `D_E9 = r_K − r_E` (a negative drop is a gain, not a break);
* each with a **paired episode-cluster bootstrap** 95 % CI (B = 2000, episodes resampled with replacement,
  the same draws for all links);
* **the chain breaks at the link with the largest drop whose CI excludes 0.** If the runner-up's CI overlaps the
  leader's point estimate, both are named (leader first). If no drop's CI excludes 0: "no single break".
* the same rule is read on seed 1; a break named on seed 0 but not on seed 1 is reported as NOT REPLICATED.
* The REEL set's rates are reported beside EVAL as a second, non-random window set (its clips were chosen by
  GT/labels, not by turns); the decision is read on EVAL only.

## 5. Lever (priority 2): inference-only selection rules, fitted on TRAIN, scored on EVAL

All rules act on the SAME captured fan and scores, so lever vs baseline is paired with zero sampler variance;
the replicate (seed 1) tests that the effect survives another sampler draw.

| id | rule (argmax restricted to `reach_keep`; empty survivor set → unrestricted E9 argmax) | grid (fitted on TRAIN) |
|---|---|---|
| V0 | shipped E9 pick | — |
| V0c | V0 with the §26.1 ceiling mask | — |
| V5 | decoder pick `sel_idx_base` (E9 removed) | — |
| V1a | **tactical-side filter**: if tac side ∈ {L, R} and `p_lat(argmax) ≥ p_min`, keep only `dir(c; τ) == tac side` | τ ∈ {0.05, 0.10, 0.18, 0.30} rad × p_min ∈ {0, 0.5, 0.7, 0.9} |
| V1b | V1a, and when tac side = S with `p ≥ p_min`, keep only `dir(c; τ) == S` | same grid |
| V2 | **nav-side filter** (clip nav token): on left/right clips keep only `dir(c; τ) ==` nav side | τ grid |
| V3 | navc term scaled ×k (soft) | k ∈ {1, 2, 5, 10, 20, 50, 100} |
| V4 | soft tactical bonus: E9 score + β·p_side(c) where p_side(c) = Σ p_lat over the classes of c's side | β ∈ {0.5, 1, 2, 5, 10, 20} |

**Fit:** per rule, the grid point minimising TRAIN mean ADE over ALL classified-or-not windows (so straight
regressions are paid for). **The reported lever** = the TRAIN-best rule among V1a/V1b/V2/V3/V4 (one choice, made
on TRAIN). V0c and V5 are reported as free ablations.

**Bar for "the lever clears" (EVAL, seed 0, paired episode-cluster bootstrap vs V0):**
1. GT-turn windows: ΔADE < 0 with CI excluding 0, AND turn-correct rate up with CI excluding 0;
2. GT-straight windows: ΔADE point ≤ +0.05 m AND CI upper bound ≤ +0.10 m (no hidden regression);
3. all windows: ΔADE ≤ 0 point estimate;
4. replicate: on seed 1 criterion 1 holds with the same sign.
A lever that misses any of 1–4 **FAILED** and is reported so; Rule Zero then names the next lever.

Planner metrics reported for V0, V0c, V5, the chosen lever and the oracle (four families, per family, never
pooled): ADE, FDE; LONGITUDINAL — signed/abs along-track error at 2 s and 6 s, speed MAE 0–2 s
(`four_families._seq_geometry` on the 4 slots at 0.5 s); LATERAL — abs cross-track at 2 s / 6 s, heading MAE and
curvature MAE 0–2 s (same geometry), terminal-heading error at 6 s; TACTICAL — turn-correct rate (GT-turn) and
straight-keeping rate (`dir == S` on GT-straight), plus `four_families.tactical_from_trajectory` on the 0–2 s
slots; STRATEGIC — nav-compliance rate of the plan on left/right clips' GT-turn windows (route following, oracle
nav), strategic DECISION unavailable (`--no-strategic` arm). Distance keeping: UNAVAILABLE in this harness (no
lead tracks) — stated, not dropped.

## 6. Fan incoherence (priority 3)

On EVAL (all, GT-turn, GT-straight): over the top-8 candidates by E9 score within `reach_keep` (incl. the pick):
circular std and range of `θ`; fraction of windows whose top-8 holds both an L and an R candidate (`τ_c`); the
same for the decoder's own score. **Quality correlation:** per window, Spearman ρ between score and −ADE over the
`reach_keep` candidates (E9 score; decoder score), mean over windows with episode-cluster CI. **Control C5:** a
uniformly random score (fixed seed) in place of the model score must give |mean ρ| < 0.02 with a CI containing 0.

## 7. Box duplicates (priority 4)

box3d head primary, agent head secondary, from the trainer's own packs (`_det_pack_*`). Gates: the TRAIN P = R
gate **0.2589** (the run's in-run `eval_box3d_calib_pr_gate` at 50,400, as briefed) and **0.5**.
* **Duplicates per object:** for each VIS-1 positive GT, the number of slots with `p ≥ gate` whose BEV centre is
  within 2 m; reported as mean, fraction ≥ 2, histogram (over GTs with ≥ 1 such slot, and over all positives).
  Fraction of greedy-2 m FPs that lie within 2 m of an already-matched positive ("duplicate FPs").
* **NMS (inference-only):** (i) BEV centre-distance NMS, radius r ∈ {0.5, 1, 1.5, 2, 2.5, 3, 4} m; (ii) BEV
  rotated-box IoU NMS, IoU ∈ {0.0+, 0.05, 0.1, 0.2, 0.3, 0.5}. Score-ordered greedy suppression per window, all
  classes together. **Fit on TRAIN:** r* / IoU* = argmax TRAIN AP@2 m (tie → the milder setting); the NMS'd
  operating gate = the TRAIN P = R gate re-fitted after NMS. **EVAL rows:** no-NMS, centre-NMS(r*), IoU-NMS(IoU*):
  precision, recall, F1, conf_ratio at (a) the re-fitted TRAIN gate and (b) the fixed 0.2589; AP@2 m;
  duplicates per object; paired episode-cluster bootstrap CIs on ΔF1 and ΔAP vs no-NMS.
* Control C8 (analytic IoU): identical boxes → 1, disjoint → 0, two 4×2 axis-aligned boxes shifted 2 m along x
  → 1/3, a box vs its 90°-rotated square twin → 1; the IoU code must read these exactly (1e-9).
* Control C9: my packs vs the diagnostics package's banked EVAL packs — presence logits max |diff| reported
  (batch 1 vs batch 8; perception is draw-independent per the reel record).

## 8. Controls that must read known values (fail loud)

* **C1 identity:** `traj == anchor_traj[sel_idx]` (max |diff| 0); my argmax of captured `sel_score_v3` over
  `reach_keep` == `sel_idx` on 100 % of windows; my argmax of captured `sel_score` over `reach_keep ∧ ceil_keep`
  == `sel_idx_base` on 100 %.
* **C2 predicate identity:** my numpy `dir`/compliance on the captured fan == the model's own
  `nav_compliance_prior` output on 100 % of candidates of informative-nav windows.
* **C3 analytic geometry:** a circular arc turning left by a known angle yields that terminal-segment heading to
  1e-6; straight → 0; mirror → negated (unit tests in `code/`).
* **C4** window digest (§1). **C5** random-score Spearman (§6). **C6 reel identity:** the REEL re-forward's `traj`
  equals the bank's (≤ 1e-3 m, the bank rounds to 3 dp) and `sel_idx` equal on 100 %. **C7** seed-1 replicate (§2,
  §4, §5). **C8/C9** (§7).
* A control that fails stops the dependent result (reported as UNVERIFIED with the failing control).

## 9. Intervals

Episode-cluster bootstrap over the EVAL episodes (B = 2000, percentile 95 %), paired where two arms share
windows. It answers *"would another draw of EPISODES say this?"* only. The seed-1 replicate answers the
INFERENCE question. Training-run variance is NOT measured (one training seed) — no claim here is a training-lever
effect.
