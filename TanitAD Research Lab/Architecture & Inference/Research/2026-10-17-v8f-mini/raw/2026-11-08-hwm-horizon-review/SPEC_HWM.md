# SPEC_HWM: the three bands and the hierarchical world-model planner (Master Mind, 2026-11-08)

Requirements: `REQ_HORIZONS.md` (R-OP, R-TAC, R-STR, R-TRAJ, R-ABS, R-VIS, R-DATA, R-RETRAIN, R-PLAN). Evidence: `AUDIT_MODEL.md`, `AUDIT_DATA.md`. Numbers are cited there with their class. This spec fixes the design; each training or read stage still gets its own frozen card and launch check.

## 1. The facts that decide the design
| fact | class | consequence |
|---|---|---|
| The world-model loss is one-step teacher-forced, so trained reach is 0.2 / 0.4 / 0.8 s | MEASURED, `hjepa.py` `_pred_loss` | New step sizes alone cannot meet the bands: a multi-step loss is required |
| Camera video ends at about 20.17 s per clip; egomotion runs a median of 140 s | MEASURED, three probes per layer | Image targets stop at 20 s; geometry targets reach 30 s |
| 4,293 / 4,211 / 3,950 of the 4,512 train clips support 30 s of checkpoints from a key at 0 / 7.8 / 20 s | MEASURED | The 6 to 30 s labels are feasible on most clips, with a validity mask |
| The strategic labels reach no loss; 4 of 8 strategic goal tokens occur | MEASURED | A strategic head and new per-frame labels are new work |
| v7.2 bands are 0-2, 2-6, 8-30 s; v9 tactical labels span 2-8 s | MEASURED | Tactical is re-cut to 2-6 s, strategic to 6-30 s, per the PI's definition |
| S checkpoints at 30 / 50 / 80 m are 3 / 5 / 8 s at the median speed | MEASURED, eval grid | Spatial checkpoints do not cover the band: time-indexed ones are added |
| The top-down planner `plan_topdown` exists and was never run on driving | MEASURED | The baseline reuses it |

## 2. The levels
| | Operative (L1) | Tactical (L2) | Strategic (L3) |
|---|---|---|---|
| Band | 0 to 2 s | up to 6 s | 6 to 30 s |
| World-model step | 0.2 s | 0.6 s (stride 3) | 3.0 s (stride 5), the PI's 2026-08-31 ladder |
| Trained reach | k = 10, 2.0 s | k = 10, 6.0 s | latent targets to k = 6 (18 s), masked to frames that exist |
| 18 to 30 s | n/a | n/a | L3 rolls out to k = 10 under logged macro-actions; supervised by geometry decoded from the rolled state, not by latents |
| Outputs | Trajectory knots 1-10 | Tactical vocabulary (goals, lat / lon actions, constraints) and trajectory knots 11-30 | Time-indexed checkpoints at 6 to 30 s, detected context, derived goals |

* **Multi-step loss:** free-run k-step term per level (`HJEPAStack.rollout`), with k = 1 bit-equal to today's loss. Trained reach is declared per level and read back by the gate.
* **Windows:** a world-model-only window sampler, per-level masks on the steps that exist, and a frozen-adapter pooled path so a 91-frame window fits an 8 GB GPU.
* **History:** multi-rate past frames at each level's own step (K = 3: 0.4 / 1.2 / 6.0 s of history), as an arm, since a long L3 history shortens the observable future.
* **Inverse dynamics stays on at every level.** H-JEPA's collapse argument applies to us.
* **Trajectory (R-TRAJ):** knots 1-10 come from the operative level and knots 11-30 from the tactical level, conditioned on the operative end state, with a continuity term. It is one 6 s trajectory, scored per band and in full.

## 3. The strategic level (R-STR, R-VIS)
* **Checkpoints:** ego route points at 6, 9, 12, 15, 18, 21, 24, 27, 30 s (the L3 step), from egomotion, masked where the log is short. The 30 / 50 / 80 m points stay as today's input for T and O until an arm replaces them.
* **Detected context, vision-only, from the current and past frames:** country (25 classes, class-balanced), period of day, illumination (`dark`, `lit_night`, `twilight`, `day`, `glare`), weather, surface, visibility. Hour and month build and check labels only. They are never an input.
* **Derived goals:** strategic goal tokens with `by_time_s` and `within_m`, a predicted strategic goal latent (section 4), and context-derived driving constraints (speed margin, headway margin).
* **Admissibility:** computed from scene tokens, nav and vision-only context. No situation-classifier output enters. The context stays a separately evaluable head, and every arm using it has a context-swap control.
* **PAUSE:** STRATEGIC stays reported as PAUSED (item 25) until the PI lifts it. Training these heads is the PI's redesign request; reporting them as a pass is not done.

## 4. The baseline planner, HWM-PLAN (R-PLAN)
Top-down at each replan, through `plan_topdown`:
1. **Goal:** a predicted L3 goal latent from vision, nav and context. Trained on the encoded true future at the observable L3 steps, plus decoded-checkpoint consistency to 30 s. The true-future goal is an oracle diagnostic only.
2. **L3** plans macro-actions to the goal and hands subgoals to L2.
3. **L2** plans macro-actions to 6 s, seeded from candidates of the tactical vocabulary, and hands subgoals to L1.
4. **L1** plans primitive actions to 2 s.
5. **Trajectory:** L1's actions for 0 to 2 s, L2's macro-actions decoded for 2 to 6 s, integrated by the kinematic model, with friction and speed bounds.

The planner samples candidates, so every read carries an inference-seed replicate.

## 5. Comparison arms
All read on the same windows: T1-kin, then T2, per band, in the four families, STRATEGIC PAUSED, paired episode-cluster bootstrap, with a replicate arm for any claim that a lever moved a metric.
| arm | what |
|---|---|
| **HWM-PLAN (baseline)** | Section 4 with the predicted goal |
| HWM-ORACLE | Same with the true-future goal, T0 diagnostic ceiling, never a driving number |
| FLAT-WM | Same goal and capacity, planning at L1 only: what the hierarchy adds |
| REACTIVE | DAG-OACC, and its band-split retrain |
| HYBRID | The reactive plan as the seed, refined through the levels |
| REL | The reactive planner reading rollouts as tokens, with the shuffle control |
| P10 | The kinematic floor |

A DOMINANT claim follows the existing rule, and no other comparison changes it.

## 6. Data release (R-DATA)
* `tanitad-context-labels-v1`, **private** on the PI's HF account: parquet labels only, one row per clip and 0.2 s frame (schema in `AUDIT_DATA.md` section 2.2), sha256 verified after a re-download. No raw frames.
* Scope: the 4,512 train clips, the eval clips and a new 1,000-clip context held-out set (`CTX-HO`) outside every chunk touching train. No existing selection changes. Whole-corpus extraction is out of local reach (about 3.7 TB of video).
* Extraction: image statistics on every frame (CPU); a SigLIP2-class model on every 0.2 s frame; Qwen2.5-VL-3B at 1 Hz as reference for weather, surface, visibility, with a probe filling the remaining frames. About 10 to 13 h of dev-box GPU under the shared lock, in 30-minute jobs (ESTIMATED).
* Geometry labels (checkpoints, goal tokens re-cut to the bands, `g_gap_6_8s` folded into strategic) come from egomotion: CPU.
* Validity: a 600-frame blind-then-reveal audit for the PI, the split key is the chunk merged with continuation matches, and hour plus sun band are cross-checks only.

## 7. Order
| stage | work | retrain class | where |
|---|---|---|---|
| 0 | PLANTEST and the per-band reactive baseline (running) | none | dev box CPU |
| 1 | Per-frame labels and HF release | none | dev box |
| 2 | Ladder code: strides, multi-step loss, window sampler, pooled path, gate entries; `PREREG_V8F_LADDER.md` | world-model only | side branch, then box 2 |
| 3 | HWM-PLAN harness on T1-kin; goal head | head only | dev box, box 2 |
| 4 | Band split of O; T re-labelled to 2-6 s | O only, then T + O | box 2 |
| 5 | Strategic heads, context-swap controls | head only | box 2 |
| 6 | Encoder LoRA on the new labels if it reads positive | encoder | pod |
| 7 | All arms of section 5 | reads | dev box, box 2 |

Constraints: `stack/tanitad/**` does not land on `agent/arch-inf-20260803` until RefCV9 signals that its e2 chain restarted; box-2 runs come from a side branch. Every training run passes the RECOVER-NEXT selfcheck with BUILT levers asserted. MRATE, LFS, DIVRS and REL are re-carded or amended on the new ladder (`AUDIT_MODEL.md` section 6.1) before they run.

## 8. Limits stated now
* No real images exist past 20.17 s in this corpus, so nothing about the scene at 20 to 30 s can be checked from images. Successor clips exist for only about 2.7 % of drives. A corpus with longer video is the lever if image supervision at 20 to 30 s is needed.
* All 25 countries drive on the right, so left-hand-traffic adaptation cannot be validated here.
* Two earlier world-model planners failed on this programme. The reactive planner stays the fallback arm, and PLANTEST on the new ladder is a required gate.
