# PRE-REGISTRATION — REF-C v3 4B-DOMINANCE EXPERIMENT (E-V3DOM-1)

**Registered:** 2026-08-18, before any training step of any arm. **Status: REGISTERED, NOT RUN** —
⛔ no launch while Thor trains (~4.7 days at registration). Design: `REFC_V3_DESIGN.md` (same
package). This document commits the arms, the reads, the estimator, the controls, the gates, and
**both outcomes** in advance. Deviations require a written amendment BEFORE unblinding.

---

## 1. Question

Does the **goal-mediated strategic/tactical/operative hierarchy** (the 4B thesis, expressed on the
supervised imitation arm) improve driving-relevant prediction over a config-identical flat arm — at
matched data, matched steps, matched optimizer, and a pinned config delta?

This is the supervised-arm half of the programme's dominance claim, parallel to (never a substitute
for) the self-supervised WM track.

## 2. Arms

| arm | definition | hierarchy levers |
|---|---|---|
| **v3-H** | `refc_v3_hier_config()` | goal cascade ON: strategic goal head + E4 conditioning; PhiTac tactical state; factored heads fed from z_tac; ĝ_tac heads; target_latent port fed; goal selection behind admission (§5) |
| **v3-F** | `refc_v3_flat_config()` | goal cascade OFF. Keeps the INCUMBENT REF-C seams (strategic ctx token, own-maneuver H19 from pooled) exactly as the trained 30 k baselines carry them |

Both arms share (identical by construction, §3): small-class encoder (~48 M), 128 anchors (6 s
rebuild, seed 0), d384/4L decoder, 6 s masked-valid horizon, factored lat×lon action space, v2.1
labels, reach clamp, Adam lr 1e-4 / warmup 2000 / cosine, batch 20, 30 k steps, `--mode diffusion`,
milestones {5 k, 15 k, 20 k, 30 k}.

**Seeds:** 3 per arm (0, 1, 2). Seed-0 pair runs first and is the early paired read at 5 k;
seeds 1–2 follow only if the 5 k pair passes the sanity gates (§6.4) — a stopped ladder is a
reported outcome, not a silent one.

**What the comparison is NOT:** v3-F is not "REF-C base as trained" (different size/horizon).
Cross-links to the trained 30 k arms are REPORTED (same val windows, same instrument) but carry the
size+horizon confound explicitly and never decide the dominance question.

## 3. The pinned delta — C122's lesson as an instrument

*"An ablation's 'everything else is identical' must be DERIVED from the two configs, and pinned."*

* `refc_v3.config_delta(cfg_h, cfg_f)` walks both configs field-by-field and returns the exact
  differing set.
* `test_refc_v3.py::test_dominance_delta_is_pinned` asserts that set **equals** the registered
  lever list of §2 row 1 — nothing more, nothing less. A new field that silently differs FAILS the
  suite.
* Param delta MEASURED at build by `param_breakdown_v3` and pinned in-band (< 6 % of total).
  **MEASURED 2026-08-18 (dev box, `param_breakdown_v3`): v3-H 62,930,419 · v3-F 60,389,402 ·
  delta +2,541,017 = +4.04 %** (PhiTac 1,708,288 · tac_latent_proj 262,656 · gstr_cond 66,816 ·
  heads 9,429 · scorer 1,156 · decoder target-latent FiLM 492,672). Context that bounds the
  capacity confound: 2.42× params bought +0.0013 (NOT separated) on this corpus (registry §4.3) —
  this 1.042× delta sits far inside that null. Any config change that moves the delta re-runs this
  paragraph BEFORE launch.

## 4. Reads — instrument, tier, families, estimator, strata

* **Instrument:** `taniteval.refc_eval` + the four-family driving suite with the val40 lead block
  (`lead_source.py`), canonical val 40 eps / 881 windows, stride 8, nav=follow, 2 denoise steps.
  ⛔ **Instrument prerequisite (0-GPU, before any arm is read):** retire the `driving.py:606-608`
  distance-keeping refusal (*"no lead-agent state exists"* — a STALE BLOCKER: `lead_source.py` and
  the val40 lead block attach row-for-row, C122) by wiring `win["lead"]` into the distance-keeping
  family — the improvement review's L2a, cited as the owner of that fix. Until it lands, the
  LONGITUDINAL family reports per-family-unavailable with reason, and the experiment does NOT
  proceed to unblinding.
* **Tier, stamped from source (C122's collision named):** the metric suite is the
  **open-loop metric tier (`tier0` in block names — the METRIC-SUITE tier, NOT doctrine-T1)**. In
  doctrine terms these are T0-class prediction/attribution reads: **no capability claim
  ("drives") will be made from them.** Both arms are read at the SAME tier by the SAME instrument —
  the comparison is legal; the wording is constrained. AlpaSim T2 is a directional secondary only
  (reconstruction-OOD 3.21×; ordering readable, levels not — registry §4.4).
* **Primary read (decides the experiment):** paired per-window Δ(v3-H − v3-F) on **selected
  ADE@2s** (full-set mean), paired **episode-cluster bootstrap** (`taniteval/ci.py`, B=2000, over
  the 40 val episodes). ⛔ Never `overlapping_holdout_se`; never split-means.
* **The four families — per family, never pooled; ADE alone is INCOMPLETE:**
  * LONGITUDINAL: target-speed accuracy; distance-keeping (headway / time-gap / min-TTC) via the
    lead block;
  * LATERAL: heading error, curvature error, yaw-rate error, cross-track;
  * TACTICAL: factored lat/lon confusion (per axis), selected-vs-executed manoeuvre agreement,
    ĝ_tac endpoint error (σ at 2/4/6 s) vs hindsight, sel_gap and goal-selection deltas;
  * STRATEGIC: g_str bearing accuracy vs the leak-guarded route label; dist_pref reported with its
    declared confound.
  Each family: paired episode-cluster bootstrap + CI on the same windows; a family that cannot be
  computed is reported per family with reason and n (e.g. 6 s slots without GT).
* **Strata (C121):** primary stratum cut = **edge-free LEAD vs NO_LEAD**; the 3-band gap splits are
  reported, never gating. Speed/curvature strata as in registry §4 for continuity.
* **Horizon reporting:** ADE at 2 s (primary, comparable to every trained arm) and at 6 s
  (masked-n stated). A horizon sweep of ADE is one row of four families, never "the result".

## 5. The selection admission gate — both branches registered

The goal-selection gate (`goal_gate`) is admitted into the emitted ranking **only if** the trained
ĝ_tac clears **σ ≤ 0.8 m 1-sigma endpoint error at 2 s** on val (the MEASURED requirement curve:
σ=0.5 better −0.1591 SEPARATED; σ=1.0 worse +0.0943 SEPARATED — `v6.py::GoalDistanceScorer`).

**The measured adverse prior (added at reconciliation with the landed improvement review, BEFORE
any run):** E-WC2 (2026-08-16) REFUSED this admission on a ridge over **frozen REF-C pooled
latents** — σ **4.7104 m [3.8087, 5.6860]**, σ/ADE 9.99 — and measured a **0-parameter
constant-yaw-rate kinematic extrapolation at σ = 1.1888 m**. v3's head differs in the two ways the
refusal names (trained end-to-end; reads window-integrating z_tac, not the last-frame pool), but
the registered expectation is **Branch B**. Two admission controls travel with the σ read:

* **kinematic comparator (mandatory):** the trained head must beat the 0-param constant-yaw-rate
  extrapolation's σ on the same windows, else the selection story is dead regardless of the 0.8 m
  bar — and that is the reported finding;
* **E-WC shape read (the sibling's S-B rule):** the v3 goal score's error-rank-vs-N curve on the
  arm's own fan must show the FALLING shape before any selection delta is read; a rising shape
  refuses the graft, whatever the σ.

* **Branch A (admitted):** selection deltas are read as part of v3-H, with the goal-echo control
  (corpus-marginal ĝ) reported beside every selection number, and the conditional capacity-control
  arm (§6.3) becomes mandatory before any mechanism claim.
* **Branch B (not admitted):** `goal_gate` stays 0 (ranking bit-identical to v3-F); the dominance
  read proceeds on the conditioning edges alone; **the σ result is itself a registered finding**
  (the K7 prior predicts the along-track/speed components are the hard half; bearing/lateral the
  easy half — this prediction is written down HERE, before the run).

## 6. Controls — per arm, not per study (C107)

1. **Positive control (instrument power):** inject the HINDSIGHT goal (GT endpoint) into the
   selection rule on the banked fan of each arm — MUST beat the arm's own selection separated (the
   E-WC machinery showed σ=0 wins; an instrument that cannot see the planted win is UNPOWERED and
   the affected read is refused, never reported as a null).
2. **Trivial-proxy control:** ego-speed as stratifier for any interaction claim (C121 discipline);
   goal-echo (marginal ĝ) for any selection claim.
3. **Negative control:** shuffled-goal selection (permute ĝ across the batch) must NOT beat
   baseline; if it does, the selection read is refused (leak or artefact).
4. **Sanity gates at 5 k (seed-0 pair):** loss curves finite; freeze-history gate PASSES on v3-H
   (else the H arm is flat-in-disguise — experiment VOID, root-caused before any GPU continues);
   intervention audit clean (frames move goals; v0 moves none); seam telemetry not saturated
   (`seam_*_sat_steps` = 0 pattern).

### 6.3 Conditional capacity-control arm (the C6 firewall)

If Branch A fires AND v3-H beats v3-F separated on the primary read, one additional arm is REQUIRED
before any "mechanism" language: **v3-Hmlp** — identical to v3-H with the goal-distance rule
replaced by the information-matched `MLPCandidateScorer` shape (endpoint + goal embedding, ~127×
params, no distance prior). If v3-Hmlp matches v3-H, the win is capacity, not mechanism, and is
reported as such.

## 7. Cost and placement

| item | value | basis |
|---|---|---|
| per 30 k run | ~7–9 h A40 (ESTIMATED) | small 30 k measured 7 h 10 m; v3 adds ~1 ms/tick decoder+hierarchy vs small |
| registered ladder | 2 arms × seed-0 (early read) → +4 runs if gates pass | ≈ 2.0–2.5 A40-days total |
| where | one idle pod A40, sequential; ⛔ never a training pod, ⛔ never Thor mid-run | invariants |

## 8. Data, parity, contamination — stated, with the oracle named

* **Train:** `physicalai-train-e438721ae894` (2,376 eps, skip-hash `f09e44db`), window-set
  bit-identical to the canonical 406,099 (the §3 masked-valid design of `REFC_V3_DESIGN.md` — the
  horizon extension does NOT re-enumerate windows; the test pins the count at smoke scale and the
  preflight pins it at full scale).
* **Eval:** canonical val40 / 881 windows. Alpamayo contamination is real (C112/C113: the aug120
  cohort IS parity-train; 6 of 40 val episodes inside the record set) — **the exclusion oracle is
  `parity.py` §10 (`assert_eval_split_clean`-class calls) and the build gate §10c**; v3 builds no
  new corpus, so §10c is not triggered, and any future v3-related eval on Alpamayo-derived splits
  MUST call §10 first. Nothing in this experiment reads Alpamayo.
* **Lead-rider probes — ANSWERED at reconciliation:** the train-side obstacle join exists —
  **2,308 episodes / 12,122,129 agent boxes** (`…/Implementation/incoming/2026-08-17-train-obstacle-join/
  TRAIN_OBSTACLE_JOIN.md`, via the improvement review §4.2). The v3.1 lead rider (their L2b) is
  UNBLOCKED; it remains OUTSIDE this experiment's arms (one decision surface at a time).
* **Registration note (D-6/D-7 of the design):** the (a,κ) parameterisation and the softade
  objective are PI-decision riders on the SHARED surface. Adopting either before launch is a §10
  amendment that re-runs §3's delta pin and §7's cost row; neither may be adopted between launch
  and unblinding.

## 9. Outcomes — both committed in advance

* **OUTCOME H (dominance supported):** Δ(primary) separated in v3-H's favour AND no family
  separated-worse AND the §6 controls hold → the 4B goal cascade is promoted on the supervised arm;
  the efficiency claim (~65 M ≥ 10 Hz hierarchy ≥ flat) is published with its CI; next step is the
  capacity-control arm (§6.3) if selection contributed, then the v3.1 lead rider.
* **OUTCOME F (dominance not supported):** Δ not separated, or separated against H → **published as
  the finding**: the goal-mediated hierarchy buys nothing measurable on this corpus at this scale on
  the supervised arm. The registered diagnosis order (run, not improvised): (1) freeze-history +
  intervention audits re-checked (was H real?); (2) per-edge ablation ladder (E4, E7, E9 off one at
  a time, seed-0 only); (3) goal-σ vs admission (did selection ever act?); (4) the C123 caution is
  the frame: a capability present in a representation need not be usable by the interface — the
  supervised arm may need a different USE of the goals, not more goals. ⛔ No seed re-rolls beyond
  the registered 3; no post-hoc stratum hunting (C121); a null here does NOT indict the
  self-supervised 4B track (different mechanism, different evidence).
* **OUTCOME V (void):** freeze-history gate fails on v3-H, or the config-delta test fails, or an
  instrument control is unpowered → the experiment does not count either way; fix, re-register the
  amendment, and only then run.

## 10. Amendment log

*(empty at registration — any entry here must predate unblinding of the read it touches)*
