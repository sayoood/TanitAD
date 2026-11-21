# Scene-tap probe: ROW4. The scene tap is ELIMINATED as the K2 lever. Frozen visual features carry the road only to 0.64 IoU held-out and 0.70 in-sample (D2).

**Master Mind, 2026-10-16 ~00:45 Berlin.**
- The probe was built and run by a Sonnet agent on the dev box (RTX 4060), with 0 A40.
- The prereg (`PREREG_SCENE_TAP_PROBE.md` §4, plus A1) was committed before any held-out IoU existed.
- The decision below is the mechanical `scene_tap_probe.decide` (`PROBE_RESULT.json`, `PROBE_DONE`).

## MEASURED
- **Data:** 1,058 navtest tokens, 123 logs, 5 log-disjoint folds (`folds.json` sha256 `20dec343…`), 2,000 updates, 3 seeds.
- **Tier:** T0-like head perception on teacher-forced frames, never driving performance.
- **Estimator:** paired log-cluster bootstrap over 123 logs, 10,000 resamples, seed 20260927. It answers another draw of LOGS, plus the training-seed floor. The fold models are held fixed.

| arm | held-out IoU [0, 40) (mean of 3 seeds) [95 %] | seeds | in-sample | near-boundary SDF MAE |
|---|---|---|---|---|
| **V** (visual tap, `visual_ctx` [7680, 256]) | **0.637 [0.605, 0.670]** | 0.635 / 0.638 / 0.638 (floor 0.0036) | **0.704** | 2.50 m |
| **S** (scene tap, `scene_ctx` [64, 256]) | **0.609 [0.577, 0.644]** | 0.609 / 0.610 / 0.609 (floor 0.0015) | 0.733 | 2.63 m |
| C (shuffle control, 1 seed) | 0.326 [0.302, 0.352] | — | 0.331 | 3.52 m |

- **S − V = −0.027 [−0.035, −0.020].** It is separated and ADVERSE, so the scene tap is worse held-out, even though it fits better in-sample (0.733).
- **V − C = +0.311 [+0.275, +0.350].** The features carry real road information beyond the label marginal, so this is not prior-dominated.
- **Mechanical row:** `ROW4_SCENE_ELIMINATED`. The bar is 0.80; V is short of it by 0.16.
- **In-sample read: D2.** V reaches only 0.704 on its own training folds. Under the committed reading, *"the head cannot fit these frozen features even in-sample, so the tap or the trunk is the lever."*
- **Instrument identity:**
  - Stage 1 recomputed `visual_ctx` in the same forward pass as `scene_ctx`. It is bit-identical to the cache on 1,058 of 1,058 tokens (max |d| 0.0), and the scorer logits are bit-identical on 1,058 of 1,058.
  - `scene_ctx` admissibility (`P2_ADMISSIBILITY.md`): computed from the four camera images plus per-log camera calibration only. It takes no ego state, goal, command or map, which makes it admissible under SPEC 6. The finding is reported here; the decision is the PI's.

**Observation (NOT pre-registered; different training data, so it is not a comparison):**
- S1's heads, trained on navtrain labels, read 0.50 on navtest.
- The same head family trained on navtest's own other folds reads 0.64 here.
- So in-domain labels are worth something. That is a hypothesis for the label-logs lever, not a measured effect.

## RULE ZERO: the next lever and its state
1. **Registered next lever:** an unfrozen-trunk graft (scene tap preferred), which is a **PI decision** (A40 plus an architecture change). It is **named, not taken.**
2. **The cheapest experiment that could still make the FROZEN trunk work, run now:** D2 at 2,000 updates does not separate *"the features cannot express the road"* from *"the head was under-trained"*.
   - `PREREG_SCENE_TAP_PROBE.md` **Amendment A2** (this commit, written before any 10k unit exists) pre-registers arm V at **10,000 updates**: seed 0, the same folds, everything else identical. The four outcomes are committed in A2.
   - It runs on the dev box (~1.6 h of GPU, under the shared lock).
3. **The trunk lever is already scheduled independently:** P6 trains its trunk (queued on the A40 behind R0c).
   - This matches `D-REFE-SELRULE-FIT` (5e0984a): the scorer's DAC errors are confident.
   - Two instruments now point the same way: the frozen representation of drivable area is the limit, not the head's weighting.

## Families
This is a perception probe; the four driving families do not apply (T0-like). LONGITUDINAL / LATERAL / TACTICAL: N/A here. STRATEGIC: PAUSED (PI 2026-09-19).

## Not in git (artifact paths)
- Units: `D:/Projects/TanitAD-artifacts/scene_tap_probe/units/` (157 MB).
- `scene_ctx` cache: `D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_scene_ctx/` (2.1 GB; manifest sha256 `05ca8b27…`).
- Both exist on ONE disk only. Banking them to HF is a backlog item.
