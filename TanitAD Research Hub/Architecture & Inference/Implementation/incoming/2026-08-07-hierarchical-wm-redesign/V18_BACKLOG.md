# v1.8 BACKLOG — the hierarchical WM increment

**2026-08-07.** Executable decomposition of `HIERARCHICAL_WM_REDESIGN.md`. v1.8 :=
v1.7 + eval-doctrine flip + controllability post-training + 6 s horizon + tactical layer
with goal-conditioned operative decode. Strategic layer and sim-loop are v1.9+ unless
gates land early. Items carry: effort (A40-h unless noted), dependencies, and the
pre-registered gate that decides them. Priority order within each epic; epics E1→E5 are
the critical path, E6+ parallel.

---

## E1 — Eval doctrine flip (T1 primary) — 0 GPU
- [ ] **1.1** `Project Steering/EVAL_DOCTRINE.md`: T0 (teacher-forced, WM-diagnostic only)
      / T1 (self-actioned closed loop, PRIMARY) / T2 (re-perception sim, when provisioned).
      Every future registry row stamped with its tier. *(0.5 h doc; gate: none)*
- [ ] **1.2** Promote `closed_loop_dump.py` + `analyze_cl.py` into `taniteval/tools/`
      as `t1_eval.py` (parameterised ckpt/head; same 40-ep grid; four families + S-rate +
      lag + sel_gap hooks). *(3 h dev; gate: reproduces §1.12 numbers byte-close)*
- [ ] **1.3** Registry annotations: stamp §1.10/§1.11 tables as T0, §1.12 as T1; add the
      doctrine note to CLAUDE.md eval rules. *(0.5 h)*
- [ ] **1.4** T1 runs for v1arch baseline + v5f-at-30k when it lands (comparability row).
      *(1 GPU-h)*

## E2 — E-H1: 6 s horizon probe — decides the operative plan length
- [ ] **2.1** Extend readout rollout to k=60 (`rollout_transitions` k param; readout loop
      unchanged — it is per-step). Train v1.7-recipe head at k=60, losses on the full 6 s,
      speed-profile term included. *(2 GPU-h; dep: pod free after Alpamayo batch)*
- [ ] **2.2** Measure: ADE(τ) curve at τ ∈ {1,2,3,4,5,6} s; reliance gate at k=60; T1
      closed-loop at 6 s; temporal stability of the 2 s prefix. *(1 GPU-h)*
      **Gate (pre-registered): usable if ADE(6 s) ≤ 3× ADE(2 s) AND reliance ≥ 0.5.**
      Pass → v1.8 plans 6 s. Fail → horizon extension becomes a trunk-training item
      (v5f-class); v1.8 stays 2 s and the failure curve is banked as the requirement spec.
- [ ] **2.3** PUBLISHED-anchor note in registry (Alpamayo 6.4 s / nuPlan 8 s / Waymo 8 s /
      UniAD 6 s) with the measured curve. *(0.5 h)*

## E3 — Stage-A controllability post-training (the §2.2 losses)
- [ ] **3.1** Implement `L_ctrl`: sampled action perturbations (envelope-bounded; also
      planner-style CV/hold sequences), roll predictor, decode with FROZEN v1.7 readout,
      L1 to `Unicycle(ã, v0)`. Flag-gated in the trainer. *(4 h dev)*
- [ ] **3.2** Implement `L_scene`: latent-subspace stability — ego-subspace from the §1.10
      ridge-probe weight vectors (banked); penalise off-ego drift between perturbed and
      human rolls. *(3 h dev)*
- [ ] **3.3** Pre-registration doc: λ grid {0, 0.1, 0.5}, gates BEFORE launch:
      **T1 ADE improves CI-separated AND S-rate(hold-action) > 0.15 AND T0 ADE degrades
      < CI**; outcome B = controllability needs trunk-scale training (v5f lever), not
      abandonment. *(1 h doc)*
- [ ] **3.4** Run 3×3 h post-training arms (predictor-only first; +encoder λ_enc second).
      *(9–12 GPU-h; dep: E1.2 for gating; pod free)*
- [ ] **3.5** Probe pack rerun on the winner (vision_vs_ego + S-curve + lag). *(1 GPU-h)*

## E4 — Tactical layer (stage 0): the pillar's first brick
- [ ] **4.1** Hindsight goal labeler: `g_tac(t,τ)` = ego-frame pose+heading+speed at
      t+τ, τ∈{2,4,6} s, from poses (labels-may-use-ego). Extend `refb_labels` with the
      3-axis severity manoeuvre split (lat: 5-way incl. sharp; lon: 5-way; lane: 3-way).
      Unit tests. *(4 h dev, 0 GPU)*
- [ ] **4.2** `phi_tac`: temporal pool over z_op(t−3..t) → z_tac ∈ R^512 @1 Hz (TCN, ~2 M).
      `f_tac(z_tac, g_tac)`: 1 Hz latent predictor (~4 M). Goal head: fan of N=8 candidate
      g_tac. *(1 day dev)*
- [ ] **4.3** Tactical selector: ranks the fan by rolling f_tac and scoring against
      hindsight-oracle outcome; RANKING loss (not CE). `sel_gap_tac` = oracle-vs-selected
      goal error — first-class metric. *(0.5 day dev)*
- [ ] **4.4** Train stage-0 on frozen trunk, 600-ep corpus. *(3 GPU-h)*
      **Gates: goal FDE@4 s < CV-extrapolated goal FDE (CI); manoeuvre 3-axis macro-F1 >
      the 5-way head's remapped F1; sel_gap_tac reported (no threshold — baseline row).**
- [ ] **4.5** Alpamayo-teacher distillation arm (OPTIONAL, flag): auxiliary CE from the
      augmentation dataset's meta-actions on overlapping clips; contamination caveat
      carried. Zero-ablation mandatory. *(2 GPU-h; dep: aug batch ≥ 2,000 clips)*

## E5 — Goal-conditioned operative decode (stage 1): the S-curve restoration test
- [ ] **5.1** Readout input surface += g_tac token (projected, layer-normed). Warm start
      from v1.7 (or E2/E3 winner). ⛔ goal is geometric ONLY — the admissibility probe
      ("could this be computed from the situation classifier's output?") documented in-code.
      *(3 h dev)*
- [ ] **5.2** Train two arms at matched params: goal-conditioned vs goal-shuffled control
      (breaks goal-trajectory correspondence; catches capacity-only wins). *(4 GPU-h)*
- [ ] **5.3** **THE test (pre-registered): T1 closed-loop with PREDICTED goals — S-rate
      target > 0.5 (from 0.05) at ADE not CI-worse than v1.7-T1.** Outcome A: hierarchy
      restores lateral skill without GT actions — the pillar's first measured win. Outcome
      B: goal conditioning insufficient at readout scale → the lever moves to trunk
      conditioning (v1.9), and we know it cheaply. *(1 GPU-h)*
- [ ] **5.4** Full T1 four-family + video (GT + v1.8) reel. *(2 GPU-h + render)*

## E6 — Efficiency ladder (the §3.6 claim) — parallel after E4
- [ ] **6.1** Arms H (stage 0+1) vs M (monolithic readout+params-matched) × {150,300,600}
      eps. Pre-registration with both outcomes bound. *(6 GPU-h)*
- [ ] **6.2** Report: T1 ADE + S-rate + goal-quality vs data curve; verdict paragraph
      pre-written both ways. *(0 GPU)*

## E7 — Strategic layer scaffold (design-complete, v1.9 execution)
- [ ] **7.1** Corridor labeler from 25 s hindsight heading (`nav_command`), stratified by
      the aug road classifier's intersection-rich class. *(0 GPU; dep: road classes banked)*
- [ ] **7.2** phi_str/f_str/selector mirroring E4 at 0.2 Hz, d=256. *(v1.9)*
- [ ] **7.3** STRATEGIC family in taniteval populated by goal/corridor quality — removes
      the standing "STRATEGIC UNAVAILABLE" for hierarchy arms. *(v1.9)*

## E8 — Supporting instruments & ops (parallel, 0 GPU unless noted)
- [ ] **8.1** `sel_gap` instrument generalised (per-level oracle-vs-selected) into
      taniteval; applied to v5f logs for the comparability row. *(3 h)*
- [ ] **8.2** Ensemble-disagreement hook in MPC cost (B2 pessimism; 2 extra predictor
      heads trained cheap on frozen encoder). *(4 GPU-h; feeds MPC E1)*
- [ ] **8.3** MPC E0 probe (lead-gap/curvature decodability from z_hat — already specced
      in MPC_WM_DESIGN.md) — unchanged, runs when pod frees. *(1 GPU-h)*
- [ ] **8.4** T2 provisioning decision memo to PI: AlpaSim/NuRec cost + what B1 unlocks
      (the only path to interaction-aware post-training). *(doc)*

## Sequencing & resource summary

```
now ──────────────► pod4 busy (Alpamayo batch ~60 h) ──────────► pod4 free
 E1.1/1.3, E3.3, E4.1, E7.1, E8.1  (0-GPU, this week, during batch)
                                            then: E2 (3h) → E3.4 (12h) → E4.4 (3h)
                                                  → E5.2-5.4 (7h) → E6 (6h) → E1.4/8.2/8.3
```
Total GPU for v1.8: **≈ 35 A40-hours** (fits one pod-week alongside ops). v5f untouched.
Data: existing v2bal + hindsight labels; no new corpus needed until E7/strategic.

**Definition of done for v1.8:** doctrine flipped (E1), horizon decided by measurement
(E2), controllability gates passed or refuted-with-class (E3), tactical layer trained with
its gates (E4), and the S-curve restoration test executed with outcome recorded (E5.3) —
plus registry §1.13 entry, T1 video, and HF push of the winning head.
