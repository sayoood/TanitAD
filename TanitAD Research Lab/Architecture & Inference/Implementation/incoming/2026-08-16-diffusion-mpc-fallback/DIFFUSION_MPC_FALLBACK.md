# DIFFUSION PROPOSALS · MPC TOP-K REFINEMENT · FALLBACK TRIGGER — the three missing diagram cells, built gated DEFAULT-OFF

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Author:** arch-inf subagent
**Scope:** the three ⬜ NOT BUILT cells on the planner/selection surface of the binding v6 diagram
(`DIAGRAM_CONFORMANCE.md` **F-15** operative-brain diffusion proposals · **§3/D-1** selection-cell
MPC refinement · **F-17** context-brain fallback trigger). Everything is machinery, gated OFF, and
**nothing was trained** — tier stamp: MECHANISM, no capability claim anywhere in this document.

**Raw:** `raw/dmf_measurements.json` (every number below) · `raw/dry_st_all_on/` (the real CLI
dry-run artifact: `dry_run.json`, `config.json`, `stage_gate.json` stamped `_dry_run`).
**Tests:** `stack/tests/test_v6_diffusion_mpc_fallback.py` — **53 passed**; v6 family
(`test_v6_staged` + selector + gstr_port + factored_goal + chain + stage_revalidation) **257
passed**; adjacent files (ladder_edges, runbook, anchor_loss, ckpt_layout, probe_trunk, p8) **147
passed**; full suite from `stack/`: see §7.

---

## 0. THE INVARIANT FIRST — the live resumes are untouched

| geometry | params | keys | required | verdict |
|---|---|---|---|---|
| default `V6Config()` | **87,893,449** | **405** | 87,893,449 / 405 | ✅ unchanged |
| config E (rebuilt from the banked live argv) | **336,542,025** | **573** | 336,542,025 / 573 | ✅ unchanged |

MEASURED (`raw/dmf_measurements.json:identity`). The default build is additionally proved
**byte-identical per tensor** (`torch.equal`, never a container digest — C72) **plus the RNG
stream** against a **CONTENT-anchored pre-change revision** of `v6.py` — the newest git revision
not carrying the `prop_diffusion` marker, never HEAD (**C75**):
`test_default_is_byte_identical_to_the_PRE_CHANGE_architecture`. All three cells are built at the
**very end of `__init__`**, only under their flags, so the `"none"`/default path draws **no RNG**
and creates **no state_dict key** — the same construction discipline as `cand_score`,
the factored-goal levers, and `cond_tac_dyn`.

**Param/key cost of each arm (MEASURED, production geometry):**

| lever | params | new keys | notes |
|---|---|---|---|
| `proposals="diffusion"` | **+437,954** | +8 | the denoiser (`prop_diffusion.*`), planner group |
| `mpc_refine=True` | **0** | **0** | a PROCEDURE — no parameter, no buffer; flippable over any checkpoint |
| `fallback_trigger=True` | **0** | +8 | calibration **buffers** only (`fallback.calib_*`, `w_spread`, `w_rollvar`) — the P7 band ships with the checkpoint, the anchor-table rule |

`STAGE_MAY_INTRODUCE["S-T"]` now admits `prop_diffusion.` and `fallback.` (alongside
`cand_score.`, `cond_tac_dyn.`); the introduction path is EXECUTED in
`test_load_stage_init_introduces_the_new_keys_over_a_bare_ckpt`, including the refusal direction
(the same load at S-S, whose allowance is `()`, raises).

---

## 1. CELL 1 — DIFFUSION PROPOSALS (`DiffusionProposalGenerator`, F-15)

**What the diagram asks:** *diffuse the full 6 s CONTROL sequence — 60 × (a, κ) — with temporally
correlated noise, as the candidate-fan generator.*

**What was built.** `cfg.proposals ∈ {"query" (default), "diffusion"}`. The `"diffusion"` arm is a
truncated **DDIM x0-prediction denoiser over the RAW control sequence** (default 4 steps —
DiffusionDrive's truncated regime; the programme's own precedent is REF-C's
`AnchoredDiffusionDecoder`, but over **controls**, not trajectories):

* **Controls, never waypoints** — H1 measured a per-waypoint offset head amplifying ε by **25×**
  in acceleration at dt 0.1, and the v5f dense fan was 97.6 % infeasible-steps. Every sample here
  passes the same bounding function as the emission (`cfg.emission_squash`, i.e. the
  measured-correct `_squash` on new arms) and integrates through the same `unicycle_rollout` —
  **feasible by construction** (MEASURED on the built fan: max |a| 3.96 < 4.0, max |κ| 0.199 < 0.2).
* **Temporally correlated noise** — a stationary OU/AR(1) draw along the 60-step axis
  (`e_t = ρ·e_{t−1} + √(1−ρ²)·w_t`, unit marginal variance), `ρ = cfg.diffusion_noise_rho` (0.9).
  White noise on a control sequence integrates to near-cancelling jitter; OU noise puts the fan's
  energy at manoeuvre timescales.
* **Zero-init discipline** — the denoiser's output layer is zero-init and the x0-prediction is
  **residual** (`x0_hat = x_k + net(·)`), so at initialisation the fan **is the squashed
  correlated-noise prior**: diverse, feasible, and anything the module later does is LEARNED.
  MEASURED consequence, one window: the query fan at the CV warm start is **degenerate** (endpoint
  spread < 1e−6 — every candidate identical) while the diffusion fan's spread is > 0.1 at init
  (`test_diffusion_fan_is_DIVERSE_at_init_…`).
* **How it trains** — through the SAME plan loss as the query fan (WTA + ε-WTA + `softade`), which
  reads `plan["waypoints"]` generator-agnostically. MEASURED in the dry S-T step: all 8
  `prop_diffusion.*` parameter tensors receive gradient (max |g| 25.7). A dedicated
  denoising-score-matching loss is a possible later trainer-side lever, deliberately NOT built —
  same-objective arms are what keep the comparison attributable to the GENERATOR.
* **The paired reference** — with the arm ON, the query fan is **still emitted** beside it
  (`qfan_a/kappa/waypoints`): the pre-registered comparison lives on the same window in one
  forward, and it keeps `emission.*`/`cand_queries` REACHABLE by the X3 totality probe
  (`test_planner_surface_is_total_with_the_diffusion_arm` — zero invisible planner params).

### 1.1 ⚠️ The binding constraint, discharged: the noise correlation is MEASURED, not asserted

Every forward returns `prop_noise_lag1_autocorr` — the **empirical** lag-1 autocorrelation of the
actual draw (`measured_lag_autocorr`), beside `prop_noise_rho_target`. The instrument itself is
validated in both directions (MEASURED, `raw/dmf_measurements.json:ou_autocorr`):

| ρ target | T | lag-1 MEASURED | Kendall-expected E[r₁] ≈ ρ−(1+3ρ)/T | lag-2 MEASURED (theory ρ²) |
|---|---|---|---|---|
| 0.0 | 60 | **0.0076** | 0.0 | −0.0443 (0.0) |
| 0.5 | 60 | **0.4686** | 0.4583 | 0.1750 (0.25) |
| 0.9 | 60 | **0.8331** | 0.8383 | 0.6786 (0.81) |
| 0.9 | 600 | **0.8952** | 0.8938 | — |

⚠️ **The T=60 estimate sits on the estimator's known small-sample (Kendall) bias, not off the
process** — the T=600 row closes on ρ exactly. The test band is centred on the bias-corrected
expectation on purpose: centring on ρ would fail on the bias, and "fixing" that by widening the
band would blunt the absence check (ρ=0 must read ≈0, and does).

---

## 2. CELL 2 — MPC TOP-K REFINEMENT (`MpcRefiner`) — built per the D-1 re-read, NOT as drawn

**What the diagram says:** *"distilled selector warm-starts, MPC refines the top-K."* **What the
measurements forbid** (DIAGRAM_CONFORMANCE §3, all binding here): the pure roll-consistency argmin
is REFUTED (winner's curse — **+5.9787 [+5.3217, +6.7625] m** worse than the trained selector,
error-rank **RISING** with N 0.241→0.286); W7-PROG requires a **goal-conditioned,
candidate-independent** component in any selection cost; E-S1-0 measured refined-readout
re-scoring **2.8–2.95× worse** purely for scoring off-distribution; SEL-1 itself stands **REFUSED**
pending the E-WC2-SW dump (`assert_selector_admissible` enforces).

**The composed cost, exactly as the constraints order it:**

| term | role | weight | default |
|---|---|---|---|
| `‖endpoint − ĝ‖` (the selector's goal point, candidate-INDEPENDENT) | ⭐ **PRIMARY** | `mpc_w_goal` | 1.0, **refused ≤ 0** |
| kinematic smoothness (normalised |Δa| + |Δκ| — the §1.14 tie-breaker) | regularizer | `mpc_w_kin` | 0.1 |
| imagined-consistency (roll P_O under the candidate's OWN actions, decode, compare) | regularizer, **never primary, never the re-score** | `mpc_w_consist` | **0.0** |

**Mechanism.** The trained selector's scores pick the top-K (the one surviving sense of "distilled
selector warm-starts"); `mpc_steps` iterations of gradient descent on a raw control **delta**,
re-bounded through `_squash` every iterate (**every iterate feasible by construction**); all model
inputs detached at entry, **every output detached at exit** — the refinement trains nothing and
nothing trains through it (MEASURED: no parameter accumulates grad through a refine;
`grad_fn is None` on every output). **The post-refinement re-score is GOAL DISTANCE ONLY** —
`mpc_selected = argmin goal_dist_post` — and the emitted `mpc_rescore` string carries the refusal
provenance so a dump is self-identifying.

**MEASURED in the dry S-T step** (tiny geometry, 2 descent steps): composed cost **30.67 → 18.96**,
goal term **30.12 → 18.43**, consistency regularizer **2.571 → 2.495** — the descent descends, the
primary term leads, and the regularizer is not fighting it.

**⛔ Inert unless a selector is admissible — enforced twice.** `V6Config.__post_init__` refuses
`mpc_refine` without `selector="goal"` (the `"mlp"` capacity control emits no goal point;
descending on its score would be candidate-DEPENDENT — the refuted family), and the trainer
preflight repeats the refusal in the launch surface. Since `assert_selector_admissible`
(`v6_chain.py`) refuses ANY selector launch while SEL-1 stands refused, **the MPC path cannot
reach a launch command today at all** — it exists as tested machinery for the day the E-WC2-SW
dump admits a selector. A CEM inner loop is a possible later arm; plain descent is the
deterministic, testable form built now.

**Roll depth is explicit and cheap:** the consistency roll is `mpc_roll_k` operative steps
(default 0 = off; the config refuses `mpc_w_consist > 0` with no roll). The shared instrument is
`V6Stack.roll_consistency` — the W7 quantity on the v6 stack, with the dual-use-only docstring
(uncertainty signal + regularizer; **never a selector**) and unconditional trunk detachment (the
`zh_op_seam` discipline; MEASURED: gradient reaches the controls, none reaches passed trunk
tensors).

---

## 3. CELL 3 — THE FALLBACK TRIGGER (`FallbackTrigger`, F-17, context brain)

**What the diagram asks:** *fires when imagined consequences disagree beyond the calibrated band
(fan spread + roll-cost variance → calibrated uncertainty, P7 ρ 0.716 measured).*

**Signal** = `w_spread · fan_endpoint_spread + w_rollvar · Var_N(roll_cost)` — exactly the
context-row quantity, and the ONE place the roll-cost survives its refutation (P7 calibration
ρ **0.7164 [0.5847, 0.7696]** on the repaired trunk — INHERITED from the P7 instrument;
`w7_roll_rerank.py` owns it).

**The distinction the brief ordered, kept in CODE, not prose:**

1. **Permutation-invariant by construction** — every statistic is a set statistic over the
   candidate axis (std/var), and **no per-candidate output exists**, so the module cannot reorder
   or choose among candidates even in principle. MEASURED: permuting the fan changes every output
   by exactly **0.0** (`test_fallback_is_permutation_invariant_hence_not_a_selector`, which also
   asserts no `[B, N]`-shaped output leaks).
2. **Monitored, never optimised** — the whole trigger runs under `no_grad` at the call site
   (nothing it emits carries a graph), so no loss can learn to reduce the uncertainty signal and
   blind its own trigger — the P2 "monitored, never optimised" rule applied to uncertainty.

**The calibrated band is LOADED, never invented.** `load_calibration` installs the P7-fit
spread→error mapping (slope/intercept/threshold/weights) as **persistent buffers** and **REFUSES**
any artifact failing P7's pre-registered gate (ρ ≥ 0.3 with CI excluding 0 — `P7_GATE_RHO`,
pinned equal to the instrument's constant by test). Until calibrated, the comparator returns
`fired: None` **with the reason** — it says so instead of inventing a boolean (the
`AnchorGoalHead.table_ready` refusal pattern). MEASURED both ways once calibrated: a tight fan
does not fire, a wide fan fires (`[false,false]` / `[true,true]`).

**The defined fallback action** is the **hold-v0/CV emission**: zero commanded accel, zero
curvature, integrated through the same `unicycle_rollout` from the true `v0` — feasible by
construction, emitted beside the plan on every forward (`fb_controls`/`fb_waypoints`), MEASURED
correct (x advances at exactly `v0·6.0` m, y ≡ 0).

⚠️ **What this cell still needs from the instrument side (stated, not hidden):** the calibration
ARTIFACT does not exist yet — producing `{spearman_rho, rho_ci, slope, intercept, threshold,
w_spread, w_rollvar}` from a real run is the P7 instrument's extension (a spread→error band fit on
`w7_roll_rerank.py`'s existing calibration block), and the trigger's **T1 operating-point eval**
(F-17 item 4) is eval-side work. Both are 0-GPU-to-cheap and are the natural next stream items.
The trainer's `--fallback-calibration <json>` path is already wired and refuses a missing file in
milliseconds (the `--gate-probes` lesson).

---

## 4. X3 — {0, 0, 0} with each ON alone and combined (MEASURED)

| arm | planner→encoder | tactical→below | strategic→below | pass |
|---|---|---|---|---|
| diffusion alone | 0 | 0 | 0 | ✅ |
| MPC alone (consist+roll on) | 0 | 0 | 0 | ✅ |
| fallback alone (roll on) | 0 | 0 | 0 | ✅ |
| **all three combined** | **0** | **0** | **0** | ✅ |

Plus the stronger loss-graph fact: in the full dry S-T step (all three ON, every S-T loss in
force), **zero encoder/readout parameters receive gradient even before `apply_stage_freeze`**
(`encoder_params_reached_by_ST_loss: []`) — every S-T term reads a cut or detached view.

The planner-side declaration grew by exactly the differentiable new surface: `qfan_*` (which is
also what keeps the emission/queries probe-reachable under the diffusion arm). MPC outputs are
detached and the fallback is no-grad — both are graph-free, so declaring them would probe nothing;
their isolation is structural (stated in-code where the declaration is extended).

## 5. EVERY GUARD, AND ITS PROVEN-TO-FAIL TEST

| guard | fires on | proven by |
|---|---|---|
| `proposals` name check | unknown arm | `test_config_refuses_an_unknown_proposals_arm` |
| diffusion knob checks | steps<1 · ρ∉[0,1) · σ≤0 · hidden≤0 | `test_config_refuses_bad_diffusion_knobs` (6 cases) |
| MPC needs the goal selector | `selector∈{none,mlp}` | `test_mpc_refuses_to_exist_without_the_goal_selector` (+ preflight twin) |
| MPC primary-term floor | `mpc_w_goal ≤ 0` | config + module-level (`test_mpc_module_itself_refuses_a_zeroed_primary_term`) |
| consist-without-roll | `mpc_w_consist>0, mpc_roll_k=0` | `test_mpc_config_guards_fire` |
| topk/steps/lr bounds | out-of-range | `test_mpc_config_guards_fire` (7 cases) |
| P7 calibration gate | ρ<0.3 · CI incl. 0 · no CI · thr≤0 · zero weights · missing keys | `test_fallback_calibration_gate_refuses_an_uncalibrated_signal` (6 directions, plus the good path) |
| uncalibrated comparator | no band loaded → `fired: None` + reason | `test_fallback_uncalibrated_refuses_to_fire_and_says_why` |
| S-W refusals (state_dict protection) | `--proposals diffusion` / `--fallback-trigger` in S-W | `test_preflight_refuses_diffusion_and_fallback_in_S_W` |
| orphan/missing calibration file | file w/o trigger · missing path | `test_preflight_refuses_an_orphan_or_missing_calibration` |
| stage introduction | new keys admitted at S-T ONLY; S-S load refused | `test_load_stage_init_introduces_the_new_keys_over_a_bare_ckpt` |
| roll-consistency selection ban | argmin over it forbidden by docstring + the goal-only re-score identity | `test_mpc_rescore_is_goal_distance_only` |

## 6. WHAT IS DELIBERATELY NOT BUILT (so nobody reads absence as oversight)

1. **A denoising-score-matching loss** for the diffusion arm — trainer-side, and it would make the
   arm comparison objective-confounded; the WTA/softade path already trains the generator
   (measured). If added later it is a DECLARED loss arm.
2. **CEM as the MPC inner loop** — descent is the deterministic form; CEM is a later arm on the
   same cost.
3. **The P7 calibration artifact + band fit + T1 operating-point eval** — instrument/eval-side
   (§3 note). The model-side contract (`CALIB_KEYS`) is frozen and tested.
4. **Any training run** — S-T launch decisions ride `v6_chain.py` and the SEL-1/E-WC2-SW
   admission, unchanged by this build.

## 7. TEST EVIDENCE

* `tests/test_v6_diffusion_mpc_fallback.py` — **53 passed** (byte-identity C75 · counts at both
  live geometries · every guard both ways · X3 alone/combined · totality with the diffusion arm ·
  the dry S-T step end-to-end · trainer flags/preflight/introduction).
* v6 family regression: **257 passed** (`test_v6_staged`, `test_v6_selector`,
  `test_v6_gstr_port`, `test_v6_factored_goal`, `test_v6_chain`, `test_v6_stage_revalidation`);
  adjacent: **147 passed** (`test_v6_ladder_edges`, `test_runbook_commands`,
  `test_v6_anchor_loss`, `test_v6_ckpt_layout_compat`, `test_v6_probe_trunk`, `test_p8_v6`).
* Full suite from `stack/`: baseline re-confirmed this session at **3396 passed / 17 skipped /
  2 xfailed** before the change. The first post-change full run came back **3 failed / 3486
  passed** — all three in `tests/test_v6_stage_init_introduction.py`, the EXACT-tuple pins on
  `STAGE_MAY_INTRODUCE["S-T"]` **doing precisely their declared job**: growing the allowance must
  fail there first and be extended consciously. The pins were extended to the new 4-entry tuple
  (with the rationale in their docstring) and the file re-passes. **Final full suite: 3532
  passed / 0 failed / 17 skipped / 2 xfailed (384 s).** ⚠️ The concurrent tree also carries two
  sibling streams' work (S2 loss; X4/P9), so the full-suite total exceeds baseline+mine —
  per-file ownership is what the targeted runs above pin.
* Real CLI dry-run (S-T, all three ON): `raw/dry_st_all_on/` — 2 optimiser steps, finite losses,
  gate stamped `_dry_run` INCONCLUSIVE (correctly un-launchable), and the log shows the selection
  head training on the diffusion fan (`sel_norm_err_rank` 0.75 → 0.0 across the two steps).

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `DiffusionProposalGenerator` + `MpcRefiner` + `FallbackTrigger` + `V6Stack.roll_consistency` + config/wiring/groups | `stack/tanitad/models/v6.py` | repo, staged (⚠️ shared file — the staged blob ALSO carries a concurrent sibling stream's X4 per-layer spectrum work: `layer_spectrum_policy` / `x4_rank_verdict` / `sigreg_trend_verdict` / `LayerSpectrumMonitor`. Disjoint hunks; named here per the commit-hygiene rule rather than discovered later) |
| trainer flags + preflight refusals + `STAGE_MAY_INTRODUCE` additions + calibration load | `stack/scripts/train_v6_staged.py` | repo, staged (⚠️ shared file — a sibling stream owns the S2 loss section (`w_s2_goal`, `s2_goal_loss`); my hunks are `STAGE_MAY_INTRODUCE`, `build_parser` block, `build_stack_from_args` block, `preflight` block — disjoint from `v6_loss_step`/`V6LossWeights`) |
| the test battery (53) | `stack/tests/test_v6_diffusion_mpc_fallback.py` | repo, staged |
| extended allowance pins (the 3 exact-tuple tests that correctly failed) | `stack/tests/test_v6_stage_init_introduction.py` | repo, staged |
| measurements | `…/incoming/2026-08-16-diffusion-mpc-fallback/raw/dmf_measurements.json` | repo, staged |
| CLI dry-run artifact | `…/incoming/2026-08-16-diffusion-mpc-fallback/raw/dry_st_all_on/` | repo, staged |
| this document | `…/incoming/2026-08-16-diffusion-mpc-fallback/DIFFUSION_MPC_FALLBACK.md` | repo, staged |

**Nothing committed, nothing pushed** (AGENT_OPERATING_STANDARD rule 1). Nothing lives on a pod,
a worktree, or Thor; Thor was not touched; no GPU was used.

## Escalations

1. ⭐ **The P7 calibration-artifact producer is the missing half of F-17** — a band fit
   (signal→error slope/intercept/threshold) added to `w7_roll_rerank.py`'s existing calibration
   block, emitting the exact `CALIB_KEYS` JSON. 0 GPU on banked dumps. Until it exists the trigger
   correctly refuses to fire everywhere.
2. ⚠️ **The diffusion arm's pre-registered comparison vs the query fan** should be written before
   any S-T launch that turns it on (endpoints: fan oracle at matched N, selection rank/p10 on each
   fan, LON family per the four-families rule). The machinery emits both fans on the same window
   precisely so this comparison is paired.
3. ⚠️ F-15's own sequencing note stands: proposals and selection are one experiment surface — do
   not fund the diffusion arm's training before the SEL-1/E-WC2-SW admission settles.
