# PRE-REGISTRATION — v4 grounding instrument (GAP 1) + v4 fan-clamp zero-change test (GAP 2)

**Written 2026-07-27 03:06 Europe/Berlin, BEFORE any v4 grounding number or any v4 fan number
existed.** Host: dev box + **pod3** (A40, MEASURED idle 0 MiB / 0 % at 02:5x). ⛔ pod1, pod2 and the
eval pod were not contacted.

At the moment of writing, the following were already MEASURED and are the inputs, not the outcomes:

* `Sayood/flagship-v4-fromscratch/ckpt.pt` pulled to `pod3:/workspace/v4instr/v4fs_ckpt.pt`
  (3,243,109,310 B, 15 s at 216 MB/s), `step = 29999`, top-level keys
  `['controller','goal_head','grounding','head','lam_mult','model','opt','phases','step']`.
* Its `grounding` sub-dict holds **42 tensors** — a complete `HierarchicalGrounding`
  (op/tac/str × invdyn + step), `state_dim = 2048`.
* `model['predictor.act_emb.0.weight'].shape == (768, 3)` ⇒ **v4 has `action_dim` 3, i.e. the `v0`
  channel is present.**
* pod3 carries the **parity** val cache `physicalai-val-0c5f7dac3b11` (600 episodes) and a copy of
  v1 `flagship4b-speedjerk` step 29999 at `/workspace/tmp/idm/ckpt.pt`.

---

## 0. The correction this pre-registration is built on

X4 (`…/2026-07-27-x1-latent-metric-probe/X1_LATENT_METRIC.md` §3.1) concluded *"`train_flagship_v4.py`
/ `flagship_v4.py` contain no reference to `grounding_losses`, `MetricInverseDynamics` or
`StepDisplacementReadout`"* and from that inferred **"v4 does not carry the instrument."**

The **string-level** claim is true. The **inference is not**, and a second probe shows why:
`train_flagship_v4.py:95` calls `flagship_loss(world, grounding, …)`, and
`tanitad/train/flagship_losses.py:359` calls `grounding_losses(...)` and merges its log verbatim
(`**g_log`, line 425). `train_flagship_v4.py:220/291/825` build the heads with `build_grounding(...)`
and `:861` puts `grounding.parameters()` into the optimizer's `head_group`.

⇒ **v4 computes all six `g_*` numbers every training step and trains the grounding heads.** What is
missing is one line of **logging**: `train_flagship_v4.py:159` forwards only `g_op_fwd_ade_m` into the
JSONL row, so the paired real-side partner never reaches disk. This is a *logging* gap, not an
instrument gap, and it is the "absence found at ONE location is not absence" trap (Operating Standard
rule 2). **This pre-registration therefore treats a v4 checkpoint's `grounding` as a genuine v4-trained
instrument**, not as inherited v1 weights — and the **from-scratch** arm makes that airtight, because
its trunk *and* its grounding heads were random-initialised and only ever trained by v4's own loop
(`train_flagship_v4.py:974` — "`build_grounding()` already random-initialize every tensor, so 'from
scratch' is simply NOT calling `_warmstart_trunk`").

---

## 1. GAP 1 — the measurement

Emit, for the v4-from-scratch arm at step 29999, on the **same batch and the same forward pass**, the
two paired quantities v1's trainer logs:

* `g_{lvl}_mid_de_m` — `MetricInverseDynamics` on **REAL** latent pairs `(z_t, z_{t+k})`;
* `g_{lvl}_fwd_ade_m` — `StepDisplacementReadout` on the predictor's **IMAGINED** rollout.

reproducing `metric_dynamics.grounding_losses`' definitions exactly (endpoint DE averaged over the
level's horizons; ADE over waypoints 1…k). Levels `op` / `tac` / `str`,
`LEVEL_CFG = {"op": ((1,2,4),4), "tac": ((8,16),16), "str": ((20,),20)}`.

### 1.1 The confound that must be removed before any comparison

v1's calibration bands (0–1k `2.2486 / 0.9541`, 28–30k `1.0129 / 0.0304`) are **TRAIN-corpus,
train-time** numbers. Comparing a v4 **val** number to them mixes three differences at once. So the
primary read is **not** v4-vs-v1's-log; it is:

> **the SAME script, the SAME val windows, the SAME statistic, run on BOTH v1 and v4.**

v1's log bands are used only as the **reproduction anchor** (does our re-implementation land in v1's
committed range?), exactly as `x1_cache_latents.py`'s fidelity stage does.

### 1.2 ⭐ Both outcomes, committed now

| outcome | what it means | what happens to the registry |
|---|---|---|
| **A — v4's ratio is of the same order as v1's on matched windows** (v4 within ~⅓…3× of v1's matched ratio) | the real-vs-imagined decode gap is a property of the *family*, not of v1's particular run | the F4 row **widens beyond the v1 family** and names v4 with its own measured numbers |
| **B — v4's ratio is materially smaller** (< ⅓ of v1's matched ratio) | the two arms genuinely differ; v1's gap is a v1 fact | **the v1-scoped row stands, and that is the answer, not a failure.** The row gains a measured v4 counter-example |
| **C — v4's ratio is materially larger** (> 3× v1's matched) | the gap is *worse* under joint planner training | row widens, with v4 flagged as the extreme |

### 1.3 ⚠️ The instrument must be validated in BOTH directions before any band is quoted

| # | check | rule (fixed now) | direction |
|---|---|---|---|
| **V1** | v1 fidelity — imagined | v1's `op_fwd` on our val windows ≤ **3×** 0.0304 m | must SUCCEED |
| **V2** | v1 fidelity — real pair | v1's `op_mid` on our val windows ≤ **2×** 1.0129 m | must SUCCEED |
| **V3** | shuffle control | `invdyn['op']` on **mismatched** real pairs > **1.5×** the matched value, on BOTH arms | must FAIL (the deliberately failing input) |
| **V4** | strict load | both `model` and `grounding` load with `strict=True` on both arms | must SUCCEED |
| **V5** | dt self-check | realised displacement ÷ logged speed ≈ 0.1 s | must SUCCEED |

If **V1/V2** fail on v1, the pipeline is wrong and **no v4 number is admissible**. If **V3** passes
(i.e. the shuffle does *not* degrade) the head ignores its input and every band is an artefact.

### 1.4 Estimator

Paired **episode-cluster bootstrap**, `taniteval/ci.py`, **B = 2000**, unit = episode cluster, seed 0.
⛔ **`overlapping_holdout_se` is not used anywhere.** Ratios are reported with the aggregation caveat
(`*_mid_de_m` is an endpoint statistic, `*_fwd_ade_m` an ADE; the generous correction is ×0.5).

---

## 2. ⭐ The `v0`-shortcut mechanism test — the prediction, committed before the measurement

**v4 has `action_dim` 3 (MEASURED above), i.e. `v0` IS injected.** The X1/X4 mechanism says the
imagined decode rides that injected scalar. So:

### 2.1 The surgical within-model ablation (this is X2, which the program still owes)

Re-run the identical forward pass on the identical windows with the `v0` action channel replaced:

* `v0_zero` — channel set to 0;
* `v0_shuffled` — channel taken from a different window in the batch (same marginal, wrong value);
* `v0_half` / `v0_double` — channel × 0.5 / × 2.0 (the scaling response).

### 2.2 ⚠️ Predictions, written before the run

| quantity | prediction | why |
|---|---|---|
| **REAL side `*_mid_de_m` under every `v0` ablation** | **bit-exactly unchanged (0.0000 % on all three levels)** | `invdyn` reads only encoder latents `(z_t, fut_states)`; the encoder never sees `v0`, which enters as an *action* into the predictor. This is a **structural must** — if it moves at all, the ablation is wired wrong and the imagined numbers are inadmissible. It is a stronger control than X4's cross-arm +3.0/+3.8/+3.1 %, which had two different encoders in it |
| **IMAGINED side `*_fwd_ade_m` under `v0_zero`** | **degrades by ≥ 2× at `op`**; I expect the same order as X4's cross-arm ×6.87/×6.27/×5.75, i.e. **×3…×10** | the readout integrates a speed it is handed |
| **direction** | **worse, never better**, at every level | removing information cannot help a metric readout |
| **`v0_shuffled` vs `v0_zero`** | **shuffled ≥ zero** (a *wrong* speed is worse than *no* speed) | zero is at least a consistent prior |
| **`v0_half` / `v0_double`** | monotone in |Δv0|, roughly linear in the induced displacement error | it is being integrated |

### 2.3 ⛔ The falsifiers

* **F-A.** If the REAL side moves under any `v0` ablation ⇒ **the ablation is wired wrong**; stop and
  say so; no imagined number is admissible.
* **F-B.** If the IMAGINED side degrades by **< 1.2×** at `op` under `v0_zero` ⇒ **the `v0`-shortcut
  mechanism is REFUTED on v4** and v4's small imagined decode is something else (manifold geometry,
  as M1 claims). This outcome is written down as a real possibility, not a failure.
* **F-C.** If the IMAGINED side *improves* under any ablation ⇒ the readout is not integrating `v0`
  at all and the whole M2 story needs re-opening on v4.

---

## 3. GAP 2 — v4's fan geometry and the reachability-clamp zero-change test

Dump, from the same v4 checkpoint's `head`, the **per-candidate emitted fan geometry + the ranked
score** on the canonical **881 val windows / 40 episodes**, then run the FIX 2 test:

`reachability_mask(fan, v0, accel_max = 2.5, horizon_s = 2.0)` (`flagship_v15.py`), the head's own
`sel_accel_max` band, **nothing tuned on held-out error**.

### 3.1 ⚠️ The verdict rule, fixed now — and it is NOT relaxable

| quantity | bar to flip `V4Config.sel_reach_clamp` to `True` |
|---|---|
| windows where the **pick moves** | **exactly 0 of 881** |
| **paired Δ ADE** (episode-cluster bootstrap, B = 2000) | **0.0000 [0.0000, 0.0000]** |
| **ADE-oracle survives** | **100 % of windows** |
| windows with an **empty** survivor set | 0.00 % |
| candidates removed | reported, not a bar |

> ⛔ **If the pick moves on even ONE window, the default stays OFF and this report says so.** The test
> is not re-run at a wider `accel_max` to make it pass; a band chosen to make the pick stop moving is
> a tuned selector, not physics. The only admissible band is the head's own `sel_accel_max` = 2.5.

### 3.2 The both-directions validation for GAP 2

**Δ = 0 is only evidence if the instrument CAN move the pick.** So the same code is driven at
`accel_max ∈ {0.5, 0.2, 0.05}` on v4's own fan, and the pick **must** move there and the ADE must get
**strictly worse**. If a tight band ever *helps*, the clamp is a tuned selector and the whole test is
void.

### 3.3 Independent cross-check

`…/2026-07-26-v5-imagination-selection/raw/fan_last_along_v4.pt` ([881,256] fp32, md5 `63dd7b77…`)
holds a v4 fan's last-waypoint along-track coordinate and reproduces a **100.57 m / 181 km/h** maximum.
Our fresh dump must reproduce that surface's maximum to within float tolerance, or the two dumps are
not the same fan and the discrepancy is the finding.

### 3.4 ⚠️ Not assumed from a name

`base_rank` is **NOT a rank** — it is `[as-trained pick] ++ [anchor index order]` and carries **zero**
score information (retraction class **C15**). Nothing here reads it. The ranking used is the head's own
`refined_logits` **after** the factorised grafts, i.e. the tensor `select` actually argmaxes.
The 256 anchors are bitwise identical to real human windows; **the clamp targets the unbounded offset
head, never the vocabulary.**

---

## 4. Priority order (a killed agent still yields value)

1. GAP 1's v4 real-vs-imagined bands (they decide a registry row).
2. The `v0` mechanism test.
3. GAP 2's fan dump.
4. The clamp verdict.

Banked incrementally: every stage writes its own JSON before the next starts.
