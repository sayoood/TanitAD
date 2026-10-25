# PREREG P4 — GT-MAP ORACLE: is map quality a planning lever for refcv8?

* Registered 2026-10-25 by the Architecture & Inference agent (PI approval of P4: 2026-10-25). Sealed by `PREREG_P4.sha256`
  BEFORE any scored roll. The roll refuses to run (`--require-seal`) and the analysis refuses to read if this file's sha256 differs.
* Evidence classes: every number below is MEASURED (artifact named), INHERITED, ESTIMATED or HYPOTHESIS, and says which.

## 1. Question

How much does refcv8's planner gain when its BEV input carries the **TRUE map** instead of what the model perceives? That is the
headroom perception levers P1–P3 (box matching loss, thin-class map loss, temporal BEV fusion) could at most buy **through the map**.
If ≈ 0, perception ranks below the planner levers L1/L2 for driving; if large, it sets their target.

MEASURED context: the planner READS its BEV (`C:/Users/Admin/refcv8_pf/perc2plan/RESULT.md`: BEV_ZERO − BASE S6 ADE +1.1505 m (s0) /
+1.1282 m (s1), inference-seed floor S6 0.0237 m, S2 0.0018 m); the map head is weak on thin classes
(`C:/Users/Admin/refcv8_pf/perc8/RESULT.md` §3); R8-ORACLE (tactical GT) found NO HEADROOM but never touched the map/BEV.

## 2. How refcv8 forms the planner's BEV (from source; tree = `git archive fb47a583763f`, the launch commit, 7/7 load-bearing files sha256-equal to the battery's launch tree)

* `stack/tanitad/refs/refc_v3.py:2189-2192` — `hi = mh(fmap_s8, grid, valid)`; `pout = br(fmap_s16, bev_hires=hi["map_hires_bev"])`;
  `pout["map_hires_logits"] = hi["map_hires_logits"]`.
* `stack/tanitad/models/map_head_hires.py:849-873` — stride-8 lift @0.25 m → BEV encoder = `map_hires_bev` `[B,64,400,240]`
  (100 m × ±30 m); the 10 cm map = `refine(feats [+ near lift of fmap_s8, 0–20 m])` → `[B,8,1000,600]` logits, a **leaf**.
* `stack/tanitad/models/refcv6_perception_branch.py:638-640, 342-346` — `bev_feats = proj(avg_pool2d(crop_{0–60 m, ±16 m}(map_hires_bev), 2))`
  (1×1 conv + GN, 64→96, `[B,96,120,64]` @0.5 m); `bev_tokens = adaptive_avg_pool(bev_feats → 30×16)` (480 tokens, zero params).
* `stack/tanitad/refs/refc.py:4798, 4826, 5073, 5175` — the planner reads `bev_tokens` (scene hook / tactical decoder) and `bev_feats`
  (decoder BEV coupling, `bev=`).
* `stack/tanitad/refs/refc_v3.py:2620-2630` — the ONLY consumer of the map logits is the refcv8 drivable critic `r8_drv`, which is
  **not built** in refcv8-r101-s0 (MEASURED: perc2plan census `r8_drv_is_none: true`, `C:/Users/Admin/refcv8_pf/perc2plan/run/BASE_s1/meta_s1.json`).

⇒ **No path lets a map raster (or the map logits) enter the planner's tokens.** Map information reaches the planner only as the shared
0.25 m encoder features that the map head also decodes. A zero-training *substitution* is therefore impossible.

## 3. Design — feature-space map injection through the model's OWN frozen map decoder (zero new parameters, zero training)

Per window, after the battery's own forward (= **BASE**, its record and dump, unchanged), a perturbation δ on `map_hires_bev`
**restricted to the planner's crop** (rows 0–60 m, ±16 m: `[64, 240, 128]` @0.25 m) is optimised so that the model's own frozen 10 cm
decoder (`refine` + its near path, fixed `fmap_s8`) reads a **TARGET map** on the crop's target-seen 10 cm cells. The planner then
receives `bev_feats` / `bev_tokens` recomputed from `map_hires_bev + δ` through the run's own pool modules. Everything else in the
forward (trunk, agent slots, box3d — whose output the planner does not consume — nav, ego) is unchanged.

| arm | target map on the crop | role |
|---|---|---|
| **BASE** | — (no δ) | the battery's own roll (reproduction control) |
| **TRUE** | SAM3 GT 10 cm codes of THIS window — read by the trainer's own path (`enable_map_hires` → `_map_fine_item(ep, t+w−1)`, raw frame `ws + n_stack − 1`) from `E:/refcv6_eval_kit/data/sam3_gt_v3_eval` | the ORACLE |
| **OWNMAP** | the model's OWN decoded map (fp32 argmax of `refine(map_hires_bev)`), on the SAME seen-cell mask as TRUE | procedure control: δ-optimisation artefact through the identical procedure |
| **SHUF** | the GT of ANOTHER episode's window (cyclic NEXT clip with a GT file in episode-index order; position-matched window `min(j, L−1)`) | liveness control: must HURT |

**The oracle effect is TRUE − OWNMAP** (same procedure, same mask, same step budget; only the label content differs), so the
procedure's own artefact cancels — the role the brief gave the adapter fed the model's own map.

Committed hyper-parameters (the step budget was fixed on smoke REACH before sealing — §8 discloses every smoke read):
* Adam, **80 steps**, lr = **0.05 × rms(map_hires_bev on the crop)** of the window, δ initialised at 0;
* loss = class-weighted CE (the run's own `_map_hires_class_weight`, sqrt-median-frequency, sha256 d70dec80…), ignore 255, on the
  crop's 600×320 10 cm cells + **0.01 × mean(δ²)/mean(f0_crop²)**; arms are independent problems (batched 3 per optimisation);
* the optimisation steps run through a **local surrogate of the decoder**: `HiresRefine` (+ near path) on the crop + 15 cells of
  context with every GroupNorm's statistics FROZEN at their exact full-extent δ = 0 values (~2.8× fewer 10 cm cells). Its exactness at
  δ = 0 is measured on every window (smoke: max |surrogate − exact| ≤ 8.6e-6, argmax agreement 1.000). **Targets, reach and the
  injected δp never use the surrogate**: the OWN map and every reach number come from the EXACT full head in fp32;
* optimisation steps under bf16 autocast on CUDA; the final decode (reach) in fp32;
* the planner sees **δp = 2×2 mean of δ** (exactly what its pool reads), rounded to fp16; δp is computed ONCE (seed-1 pass) and cached
  (`F:/p4_gtmap/delta_cache`); the seed-0 pass READS it, so **both inference seeds receive the identical injected tensor**.
* Windows without a GT file / without seen cells: TRUE and OWNMAP **are BASE by construction** (δ = 0, counted); likewise SHUF without
  a donor.

**Information-leak statement.** TRUE (and SHUF) feed a privileged label at inference: this is an **oracle BOUND, never a deployable arm**
(tier T0-ORACLE). OWNMAP uses the model's own output plus the GT *seen mask* (which cells are supervised — identical to TRUE's, so the
contrast differs only in label content). No GT trajectory, ego future, or tactical label enters any arm. The goal path is untouched
(the situation classifier's output is not involved).

## 4. Data, model, tier, estimator, n

* Checkpoint: refcv8 EMA-48,852, `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`, md5 `b4364029b68f29a33eeb4aebbbbd002b`; config `E:/refcv8_eval_kit/ckpt/config.json`.
* Roll: the SHIP-R8 battery child `roll_seed_r8_pf.py` (hold_r8pf, per-frame nav feed), `--os-only`, eval139: **4,754 windows / 139
  episodes** (S2 = 4,754, S6 = 3,668, INHERITED from perc2plan RESULT.md), windows `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/windows_s{0,1}.json`
  (md5 a3a78dba…, identical across seeds and to perc2plan's).
* Inference seeds **0 and 1**; every arm re-runs the model from the SAME pre-call RNG state inside `torch.random.fork_rng` (common random
  numbers; the r8oracle mechanism).
* Tier: **T0-ORACLE on the SHIP-R8 T1 self-action open-loop roll** (logged eval frames). A bound, never driving performance.
* Interval: **paired episode-cluster bootstrap** (`taniteval.ci` via `pair_read_r8.paired_cell`), B 2000, seed 0 — answers *another
  draw of EPISODES* only; one training seed (not another training run); the inference question is answered by the two seeds and the
  floors below.
* Families (all reported, never pooled): ADE/FDE; LONGITUDINAL (speed, along-track, accel, target-speed accuracy); LATERAL (heading,
  curvature, yaw-rate, cross-track); TACTICAL (trajectory-derived lat/lon decision correctness); STRATEGIC = NOT APPLICABLE (n 0:
  `--no-strategic`, route checkpoint off). LONGITUDINAL distance-keeping (headway/TTC) = NOT COMPUTED (n 0: no lead-agent instrument in
  `pair_read_r8.METRICS`) — a named gap.
* Floors: **F_S = |mean ADE(BASE s1) − mean ADE(BASE s0)|** (the perc2plan convention; perc2plan read F_S6 0.0237 m) and
  **F'_S = |Δ_{TRUE−OWNMAP}(s1) − Δ_{TRUE−OWNMAP}(s0)|**; the bar uses **max(F, F')**.

## 5. Controls (each must read its known value)

| id | control | known value | if it fails |
|---|---|---|---|
| C1 | BASE (s1) reproduces the banked battery dump `step48852ema_pf/dump_s1` (os + plan8) | max abs ≤ 1e-5 | VOID |
| C2 | captured BASE trajectory == the run's own dump plan8 | exactly 0 | VOID |
| C3 | recompute identity: the δp = 0 recompute of `bev_feats`/`bev_tokens` == the model's own, on EVERY BASE call; no-target windows' arms == BASE | bit-exact | VOID |
| C3c | smoke: `P4_ZERO_DELTA=1` (every arm's δp forced to 0, full arm machinery) → every arm == BASE | bit-exact | VOID |
| C6 | both seeds inject the identical δp per window | exact | VOID |
| C7 | GT future `g` in the run's dump == the banked dump | exactly 0 | VOID |
| C8 | the dataset item's 10 cm GT == a direct store read of (clip, ws) | equal | VOID |
| unit | `test_p4_pool_shortcut.py`: pool(f0+δ) via the shortcut == explicit full-res placement; a one-column misplacement goes RED | PASS (MEASURED 2026-10-25, `code/unit_test.log`) | VOID |
| **R (reach)** | TRUE-arm decoded map vs GT, pooled IoU on the crop's seen cells (seed-1 compute pass): **mIoU(classes 1–7) ≥ 0.80**, **lane / crosswalk / non-drivable-edge IoU ≥ 0.50 in 0–20 m and 20–40 m** (where present), **mIoU gain over the model's own map ≥ +0.20** | all three | INCONCLUSIVE-REACH |
| **L (liveness)** | **SHUF − OWNMAP, S6 ADE, separated WORSE (CI lo > 0) at BOTH seeds** — a wrong map injected the same way must hurt | both seeds | INCONCLUSIVE-INERT |

## 6. Outcomes — committed before data, both directions (primary: S6 ADE, Δ = TRUE − OWNMAP, per seed)

Evaluated in this order (first match wins):
1. **VOID** — any of C1 (s1), C2, C3, C6, C7, C8, applied-once fails.
2. **INCONCLUSIVE-REACH** — R fails.
3. **INCONCLUSIVE-INERT** — L fails.
4. **ADVERSE** — Δ CI lo > 0 at BOTH seeds (the true map makes the planner worse than its own).
5. **HEADROOM** — at BOTH seeds: Δ CI hi < 0 **and** −Δ > max(F_S6, F'_S6). If any family metric (S2 or S6) of TRUE − OWNMAP is
   separated WORSE at both seeds, the verdict is stamped **HEADROOM (with a family cost: …)**.
6. **PARTIAL** — the HEADROOM effect at exactly one seed.
7. **NO-HEADROOM** — otherwise (CI spans 0 or the effect is under the floor at both seeds). The CI's lower end (most negative) at the
   two seeds is then quoted as *the most a perfect map could buy at 95 %*.

Beside (decides nothing): S2 ADE and all family rows for every contrast (TRUE−OWNMAP, SHUF−OWNMAP, OWNMAP−BASE, TRUE−BASE,
SHUF−BASE); Δ as a share of perc2plan's BEV_ZERO effect; reach per class/band incl. the planner-visible (pooled-δ) map.

## 7. Next lever, committed per outcome (Rule Zero)

* **HEADROOM** → perception IS a planning lever and the TRUE−OWNMAP gain is P1–P3's ceiling. Next (same harness, zero training): a
  **per-class oracle** (GT only for {drivable}, {lane, edge}, {crosswalk}; own map elsewhere) to localise which map content carries the
  gain, then the matching lever (P2 thin-class loss if lanes/edges, P3 temporal fusion if far-range) on the v7-tiny ladder.
* **NO-HEADROOM** → the map is not where the driving error is: rank perception-for-map **below L1/L2** and spend there. The agent
  channel is the remaining perception route (perc2plan AGT_ZERO S6 +0.11 m): next = the **GT-agent-token oracle** on this harness.
* **INCONCLUSIVE-INERT** → the planner does not read map semantics through the decoder's subspace; next = the **trained adapter oracle**
  (map raster → BEV-token residual fitted on TRAIN windows, TRUE − OWN through the identical adapter, shuffled-GT arm).
* **INCONCLUSIVE-REACH** → raise the step budget / free δ beyond the crop (one amendment), re-run seed 1.
* **ADVERSE** → the planner is co-adapted to its own map errors: a perception fix must be paired with planner fine-tuning (oracle +
  fine-tune arm).
* **PARTIAL** → a third inference seed (seed 2) on the same cache before any claim.

## 8. Smoke disclosure, amendment policy, compute

* **What the smoke read before sealing** (6 windows = 2 episodes × 3 of eval139, seed 1, CPU fp32; `C:/Users/Admin/refcv8_pf/gtmap_oracle/smoke*/`):
  controls and REACH only — `p4_analyze.py --smoke` returns before any planner surface, so **no planner delta was computed**.
  MEASURED controls: C1 BASE vs the r8oracle CPU plain dump (another environment, same code by sha256) max |os| 7.2e-7, |plan8| 7.6e-6;
  C2 exact; C3 recompute identity 6/6 bit-exact; **C3c zero-δ identity: every arm == BASE bit-exact (max |traj| 0.0, sel equal)**;
  C8 6/6; cache round trip: injected δp checksums identical, trajectories within 2.3e-5 m (CPU float nondeterminism), sel identical.
  REACH (TRUE vs GT, pooled IoU, crop): 30 steps exact-decode optimisation → mIoU 0.786, non-drivable edge 0.477 (FAILS the bar);
  60 steps surrogate (TRUE only) → 0.846, edge 0–20 m 0.500; **80 steps surrogate (TRUE only) → mIoU 0.857, lane 0.857, crosswalk
  0.906, edge 0.646 (0–20 m 0.536, 20–40 m 0.660), drivable 0.988; own map 0.172** ⇒ 80 steps committed. **The reach BAR (§5) was
  written before the first smoke and is unchanged.** Beside: what survives the planner's 2×2 pool (TRUEpool) is mIoU 0.723, edge
  0.365, crosswalk 0.573 — sub-0.5 m map detail is partly invisible to the planner's BEV by construction.
* **Final smoke with the committed hyper-parameters** (all three arms batched, `smoke_final/A`, `P4_SMOKE.json`): C1, C2, C3, C3b,
  C3c (`smoke_final/B`: zero-δ, every arm == BASE bit-exact), C8, applied-once all PASS; refine reproduction (CPU fp32) max diff 0.0;
  surrogate exactness at δ = 0 ≤ 8.6e-6, argmax agreement 1.000. Reach: TRUE vs GT mIoU **0.857** (own map 0.172; gain +0.685),
  lane / crosswalk / edge 0–20 m 0.749 / 0.938 / 0.536, 20–40 m 0.878 / 0.937 / 0.660; SHUF vs its donor GT 0.810 and vs THIS window's
  GT 0.065 (a genuinely wrong map); OWNMAP vs the own map 0.914. δp rms / feature rms (median): TRUE 0.93, OWNMAP 0.84, SHUF 1.01 —
  large edits, which is why OWNMAP (a same-size edit carrying no new content) is the reference for TRUE.
* No further amendment: the hyper-parameters above are final for the scored roll.
* Compute (RTX 4060): a read-mode seed ≈ OWN 1,522–1,614 s + 3 arm forwards × 653–690 s + ~400 s overhead ≈ **3,900 s** (MEASURED
  on this GPU: r8oracle `oracle_s{0,1}.npz` `wall_by_arm_s`, same roll + fork_rng mechanism); the seed-1 inversion = 240 surrogate
  arm-steps/window ≈ 22 GFLOP each ⇒ **0.5–1.1 s/window ≈ 2,400–5,200 s** (ESTIMATED from FLOPs at 5–10 effective TFLOPS) ⇒ total
  **≈ 2.8–3.6 GPU-h** (ESTIMATED). The seed-1 roll projects both seeds' wall after 150 windows and STOPS (`--budget-max-s 16200`,
  exit 7, marker) above 4.5 h, before the budget is spent.
* GPU only after `C:/Users/Admin/refcv8_pf/perc8tol/GPU_DONE` exists and no process command line contains `final_pinned_f`.
