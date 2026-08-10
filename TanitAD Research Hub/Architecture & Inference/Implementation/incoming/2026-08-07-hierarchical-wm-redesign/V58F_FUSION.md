# v5.8f — the fusion arm (PI directive 2026-08-09: "putting all our best assets in this frontier resulting model")

**Draft v1, 2026-08-10 (overnight autonomous session).** v5.8f := the v5f trunk lineage +
every measured win of the v1.6/v1.7/v1.8 line + the diffusion-MPC synthesis. Everything
below cites a MEASURED basis or is tagged as the experiment that will decide it.

## 1. What each parent contributes (evidence-based asset inventory)

| asset | from | evidence |
|---|---|---|
| **w120 cylindrical trunk, from-scratch co-trained, spatial imagination** | v5f | oracle_ade 0.1975 @30k; imagination = only occlusion mechanism in-programme |
| **unicycle control parameterisation** | v1.6/v1.7 | jerk 36→1.13; v5f's own accel MAE **8.11 m/s²** vs v1.7's 0.55 is the disease it cures |
| **speed-profile loss** | v1.7 | ADE −16 % CI-separated (T0) |
| **latents-only decode discipline + reliance gate** | v1.6 | reliance 1.18 (v1.7); the shortcut-removal methodology |
| **fan + explicit selection topology** | v5f | fan works (0.20), selector doesn't (gap 0.20) — MPC re-rank is the measured-defect fix |
| **goal-conditioned generation (gated, goal-dropout)** | v5f (cond_vtarget/route) | mechanism exists; E5 upgrades class→geometric g_tac |
| **controllability post-training (L_ctrl/L_scene)** | v1.8 stage-A | §1.10 probes: action under-weighting measured on v1arch trunk; v5f trunk TBD |
| **tactical/strategic layers (temporal hierarchy)** | v1.8 E4/E7 | the pillar; S-curve restoration test is its gate |
| **T1 eval doctrine** | v1.8 E1 | §1.12: open-loop lateral was action echo |
| **6 s horizon** | E-H1 (pending) | Alpamayo 6.4/nuPlan 8 anchors; gate ADE(6s) ≤ 3×ADE(2s) |

## 2. Architecture (composition, not reinvention)

```
frames(8, 176x624cyl) ─► v5f ENCODER ─► z_op ──► φ_tac ─► z_tac (1 Hz) ─► φ_str ─► z_str
                                        │            │ goal fan+sel (rank-loss)     │
                                        │            ▼                              ▼
                             IMAGINATION FIELD    g_tac* (geometric)            g_str*
                                        │            │          ▲ conditions ▼
                                        ▼            ▼
             UNICYCLE-ANCHOR DIFFUSION HEAD (fan of (a,κ) sequences, 6 s, 2-step denoise,
             conditioned: states + per-candidate imagined consequences + g_tac*)
                                        │
             MPC RE-RANK (top-8: per-candidate WM roll + explicit costs) ─► plan
             └── distilled back into the fast selector (L4)
```

Key deltas vs v5f: anchors emit **(accel, curvature) sequences** integrated to waypoints
(feasible by construction — kills the 8.11 m/s² defect structurally); imagination gains a
**per-candidate axis** for the top-8 (the documented no-candidate-axis limitation); goals
become geometric; horizon 6 s pending E-H1; trunk post-trained with stage-A controllability
so T1/closed-loop rolls are trustworthy.

## 3. The wedge experiments (each pre-registered before its run)

| id | question | pod | cost | gate |
|---|---|---|---|---|
| W1 = X0-lite | does kinematic-cost re-rank alone close sel_gap? (no WM roll needed) | 5 | 1 h | ≥30 % gap closed — **RUN 2026-08-09: REFUTED, −16.7 %** (re-rank worsens; registry §1.13) |
| W2 | fan feasibility census (v5f 30k own fan) | 5 | in W1 dump | **RUN 2026-08-09: 97.6 % steps / 100 % candidates infeasible incl. oracle; mean \|a\| 252 m/s² → retrofit urgency MAXIMAL** |
| W2b | (added, exploratory) 3-tap smoother probe | 5 | in W1 dump | sel accel 8.10→3.09, BOTH ADEs improve (jitter = denoise residue) — partial "W4-lite", stack it, not a substitute |
| W3 = stage-A on v5f trunk | is the co-trained trunk more action-controllable than v1arch's? | 5 | 3 h probes | R²(action-response) comparison |
| W4 = unicycle-anchor head retrofit | retrain ONLY the offset head to emit (a,κ) on the frozen 30k trunk | 5 | ~4 h | **RUN 2026-08-10: PASS both gates — accel MAE 0.774 (<1.5), oracle 0.1077 (BEATS 0.1991 by 46 %), violations 0.0. New defect exposed: frozen selector near-uninformed on the new fan (sel ADE 0.79) → W4b selector recalibration + W7 now unblocked. Registry §1.13; HF /w4/** |
| W5 = E-H1-w120 | 6 s horizon on the w120 trunk | 5 | 2 h | ADE(6s) ≤ 3×ADE(2s) |
| W6 = E4+E5 tactical | S-curve restoration via g_tac (v1arch trunk first — cheaper) | 4 | 7 h | T1 S-rate > 0.5 |
| W7 = X0 full | WM-roll MPC re-rank top-8 (needs W3 pass) | 5 | 2 h | ≥50 % sel_gap closed at T1 |

W4 is the **load-bearing novelty** of v5.8f: it converts the fan to controls at head-scale
cost (the anchor head is small) — if it passes, v5.8f needs NO trunk retrain to get a
feasible fan; if it fails, unicycle anchors become a v6 trunk-training lever and v5.8f
ships with projection-to-manifold instead (stated fallback).

**W1/W2 measured update (2026-08-09, registry §1.13):** W1's failure is *structural* — with
97.6 % of fan steps infeasible, any waypoint-space kinematic cost ranks jitter, not manoeuvre
quality. Therefore W7's WM-roll re-rank is gated not only on W3 but on a **kinematically clean
fan** (W4 head, or minimally the 3-tap smoother). W4 training launched the same night
(`train_v58f_unicycle_head.py` on the frozen 30k trunk, pod5; gates in `w4_gate.json`).

## 4. Assembly sequence

W1/W2 (tonight) → W3, W5 (tonight/tomorrow) → W4 (tomorrow) → W6 on pod4 post-Alpamayo →
W7 → **v5.8f assembly** = 30k trunk + stage-A post-train + unicycle head + g_tac
conditioning + MPC re-rank + distilled selector → full T0+T1 four-family eval + video →
registry §1.13 + HF gated push. Ladder cost ≈ 20 GPU-h; assembly ≈ 10 GPU-h.
