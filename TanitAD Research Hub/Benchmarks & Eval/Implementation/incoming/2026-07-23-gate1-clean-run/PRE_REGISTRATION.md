# Pre-registration — the PROPER clean Gate-1 closed-loop-aware fine-tune

**Status:** pre-registered protocol for the run that becomes executable once §A blockers clear. Both
outcomes committed in advance (CLAUDE.md §Operating standard 5). Not yet runnable — see
`GATE1_CLEAN_RUN_P0_FINDINGS.md` for why the P0 gate holds it.

---

## A. Blocking preconditions (all measured absent today)

1. **≥ ~100 distinct real junction scenes**, scene-disjoint from eval. The 40-ep clean val yields only
   ~11–16 trainable junction episodes (MEASURED, `gate1_junction_inventory.json`) → memorizing scale.
   Source: junction episodes mined from the **train corpus** (parity-firewalled — never re-select the
   canonical `physicalai-train-e438721ae894` episodes into eval) and/or L2D/comma turns.
2. **A low-OOD lane-departure metric.** Add `corridor_departure_rate` to `lowood_closedloop.py`:
   `depart = |dlat| > LANE_HALF (1.75 m)`, aggregated per scene (the on-policy `dlat` is already computed).
   This is the clean lane-keeping analog of off-road. **Collision-avoidance stays out of scope for the
   clean source** (no reactive agents) — route it to AlpaSim with the ~3.2× OOD caveat, or the hybrid (c).
3. **REF-C wired into the low-OOD harness** (~1 h; ckpt on the eval pod). Decode via
   `model(fw_warped, nav_cmd=None, v0, steps=2)["waypoints"][k]`, warp exactly as `lowood_probe`.

## B. Protocol (once A clears)

- **Split:** episode-disjoint. Junction episodes → **~80/20 train/held-out**, plus a 5-fold
  leave-scene-out for a CI on generalization. Held-out episodes NEVER appear in training rollouts or labels.
- **P2 — on-policy rollouts on the LOW-OOD source** (real footage, arc-length re-index + on-policy warp),
  REF-C in the loop, over the TRAIN junction scenes. Generate recovery labels = GT expert path 0.5–2 s
  ahead in rig frame (as the prototype's `ref_lookahead_rig`).
- **P2 label hygiene — CAT-K / RoAD filter (`catk_road_filter_and_dev_regularizer.py`):** drop recovery
  labels whose target leaves the **P1-measured envelope** (|left| > 3.0 m or heading-corr > 12°) or points
  backward (`fwd_end ≤ 0`). *(On the prototype's NuRec labels this dropped 49 % — MEASURED.)* Cite CAT-K
  (Zhang/Karkus et al., CVPR 2025) + DAgger (Ross 2011).
- **P3 — fine-tune** REF-C decoder-only, loss `traj_L1 + cls_CE + λ_dev·‖FT_traj − base_traj‖₁`
  (`λ_dev = 1.0`, base-plan trust region). Eval on **HELD-OUT** scenes: closed-loop ADE@2s,
  `corridor_departure_rate`, and `plan_shift_from_base` (deviation side-effect). Decision-grade CIs via
  `taniteval/ci.py` episode-cluster bootstrap.

## C. Pre-registered outcomes (both committed)

- **PROMOTE:** on **held-out** scenes, `corridor_departure_rate` and/or closed-loop ADE **drop**
  (CI-separated) **AND** `plan_shift_from_base` stays bounded (≤ base + tolerance) → the closed-loop-aware
  lever is a **robust** low-OOD lane-keeping improvement. Report to Sayed; consider promoting to a v4.x
  planner FT. *(Note: this is the lane-keeping claim; the off-road/collision claim still needs AlpaSim.)*
- **BOUND:** held-out metrics do not move (memorization) OR `plan_shift_from_base` blows up (deviation
  side-effect persists) OR the junction inventory is still < ~100 distinct scenes → report the bound and the
  specific data/instrument gap. **Do not** promote; **do not** re-tune schedules to chase it.

## D. What today's measurement already tells us about C

At **n=15** (prototype scale), the naive FT hits the **BOUND** decisively (MEASURED, `gate1_clean_loo.json`,
5 leave-3-out folds): held-out recovery-L1 **5.06 → 5.06 (Δ≈0)** while train → 0.41 (memorization); held-out
plan-shift **7.58 m**. CAT-K + `λ_dev` shrink the plan-shift to **2.88 → 1.49 m (−80 %)** but held-out
recovery stays ~5 → **the binding lever is #A.1 (more distinct scenes)**, not label quality.
So this pre-registration will only reach PROMOTE once §A.1 is satisfied; running it at n≈15 is
pre-committed to BOUND and is not worth training GPU.
