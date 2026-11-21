# Post-mortem — why flagship **v3enc** is materially worse than **v1**

**Date:** 2026-07-21 · **Arm:** `flagship4b-v3enc-30k`, stopped by the PI at step 10,800 after the
pre-registered 10 k gate returned `RESTART` · **Audience:** the v4 flagship design.

> **Deliverable of record.** The thing v4 needs from this note is the **DO-NOT-CARRY list (§8)**.
> Everything above it is the evidence chain that earns each item.

**Estimators declared up front (CLAUDE.md rule).**
Held-out driving/ADE numbers are the **episode-cluster bootstrap** over the 40 val episodes
(`taniteval/ci.py`, B=2000), **paired** where two arms share the 881 windows. Training-log numbers are
**per-batch (B=16) values averaged over 2 k-step buckets** — a descriptive summary, *not* an interval,
and noisy row-to-row by ±0.3 on `g_op_fwd_ade_m`. No power-law exponent is quoted anywhere in this
note; the gate's own fit is `R²=0.179`, **UNSUPPORTED**.

**Primary sources actually read** (nothing here is copied from prose):
`Project Steering/Gates/flagship-v3enc-gate-10k-2026-07-21.json` ·
`taniteval/results/{flagship-v3enc-10k,paired_v3enc10k_vs_flagship30k,diagv2_summary}.json` ·
`taniteval/results/driving_{flagship-v3enc-10k,flagship-30k,flagship-v2-6k,flagship-speed,flagship-nospeed}.json` ·
`MODEL_REGISTRY.md` §1.1–1.4 · **four run `config.json` + four `train_log.jsonl` pulled live from
the pods** (v1, v3enc, v2, no-speed — see manifest) · the code paths
`stack/scripts/train_flagship4b.py`, `stack/tanitad/train/{flagship_losses,decorr}.py`,
`stack/tanitad/models/metric_dynamics.py`, `stack/tanitad/config.py`, `stack/scripts/run_gate.py`.

---

## 0. The one-paragraph answer

**Budget-matching does not rescue v3enc.** At the *identical* step 8–10 k the training
forward-consistency ratio is **4.48×** — the same order as the unmatched 10 k-vs-30 k eval ratio of
4.60×; and the most conservative clean-eval framing available (v3enc @10 k vs v1 @**19 k**, i.e. v1
carrying 1.9× the budget) still gives **3.19×**. **The regression is not in the encoder's metric
grounding — it is in the predictor's use of the speed channel.** The encoder's metric-inverse-dynamics
decode error is within **1.10–1.15×** of v1 at matched steps, while the forward-consistency error on the
action-conditioned rollout is **4.2–4.5×** worse. On matched steps v3enc recovers only **~23 %** of the
speed-input benefit that v1 recovers over the no-speed ablation control. The mechanism with the tightest
fit is `v2_ego_dropout=0.25`, which — via `flagship_losses.py:229-231` — **zero-fills the v0 speed action
channel on 25 % of samples**, i.e. tells the model "you are stopped" while showing it a moving car.
v3enc did **not** diverge like v2; it **plateaued**, and the parts of the lever pack aimed at bias and
at the command-echo demonstrably worked.

---

## 1. Fixing the comparison first — how much of the gap is budget?

The headline everyone will repeat is *"v3enc is 4.6× worse than v1"*. That compares 10 k steps to 30 k.
Three framings, worst-to-best for v1:

| framing | v3enc | v1 | ratio | budget parity |
|---|---:|---:|---:|---|
| **Headline** eval `ade_0_2s`, full-set, same 881 windows | 1.9654 | **0.4271** @30 k | **4.60×** | ✗ 3× |
| Nearest **clean eval** of v1 that exists (`flagship-speed`, 19 k relay) | 1.9654 | **0.6152** @19 k | **3.19×** | ✗ 1.9×, still favouring v1 |
| **Strictly matched step**, training `g_op_fwd_ade_m`, 8–10 k bucket mean | 0.4717 | **0.1052** | **4.48×** | ✅ exact |

*(Paired episode-cluster bootstrap on the headline row: Δ **+1.5383 [+1.2697, +1.8159]**, CI-separated
against v3enc. v3enc is also CI-separated **worse than the CV floor**: Δ +1.1277 [+0.8741, +1.4134].)*

**There is no evaluable v1 checkpoint at 10 k.** The registry leaderboard's earliest flagship-v1 eval is
the 19 k relay; `flagship4b-speedjerk-30k/` on pod2 holds one `ckpt.pt` (final) and no milestone
archives. So a like-for-like *eval* at 10 k **cannot be produced** — MEASURED absence, not an oversight.

**What the matched training metric can and cannot support.**
`g_op_fwd_ade_m` is the operative-level forward-metric-consistency ADE from `grounding_losses`
(`metric_dynamics.py:311`) — the grounding readout decoded on the predictor's rollout with *true*
actions, on **train** batches. It is not val ADE, and it is per-batch noisy (raw rows swing 0.30–0.90 at
9–10 k with no trend). Its **horizon is fixed** (`level_cfg["op"] = ((1,2,4), 4)` in *both* runs'
`config.json`), so the rollout-K schedule does not move the measurement — the comparison is
apples-to-apples on that axis. ⚠️ It is **not** apples-to-apples on one axis: v3enc's batches carry the
ego-dropout, so ~25 % of its samples are scored with v0 zeroed (§4.1). That inflates v3enc's number by
an amount we cannot isolate without a re-run.

**Answer: essentially none of the gap survives being explained away by budget.** The strictly matched
read (4.48×) is as large as the unmatched read (4.60×); the artifact-free eval read is 3.19× *with v1
handicapped by 1.9×*. **A ≥3× deficit survives every framing that can be constructed from existing
artifacts.**

### 1.1 The shape of the deficit: plateau, not divergence

v3enc's **own** training curve, 2 k buckets (`g_op_fwd_ade_m`):

| bucket | 0–2k | 2–4k | 4–6k | 6–8k | 8–10k | 10–12k |
|---|---:|---:|---:|---:|---:|---:|
| **v1** | 0.6458 | 0.3389 | 0.2437 | 0.1625 | **0.1052** | 0.1060 |
| **v3enc** | 1.0364 | 0.6374 | 0.4762 | 0.4268 | **0.4717** | 0.4824 |
| **no-speed control** | 1.3152 | 0.9425 | 0.7556 | 0.6686 | **0.5794** | 0.5510 |
| **v2 (killed)** | 1.3111 | 0.9104 | 0.6810 | 0.5828 | — | — |
| ratio v3enc/v1 | 1.61 | 1.88 | 1.95 | 2.63 | **4.48** | 4.23 |

Two reads that the gate's "WIDENING" flag hides:

1. **v3enc flatlined from ~step 4,500** (0.476 → 0.427 → 0.472 → 0.482). The widening ratio is *v1
   continuing to learn*, not v3enc degrading. That is categorically different from v2, whose speed
   rollout diverged 15.1 → 23.9 m/s.
2. **v3enc sits closer to the no-speed ablation control than to v1.** Expressing it as *fraction of the
   speed-channel benefit recovered* — `(nospeed − arm) / nospeed` — removes the level and is the
   cleanest single number in this note:

| bucket | v1 recovers | v3enc recovers | v3enc as % of v1 |
|---|---:|---:|---:|
| 2–4k | 64.0 % | 32.4 % | 50.6 % |
| 4–6k | 67.7 % | 37.0 % | 54.6 % |
| 6–8k | 75.7 % | 36.2 % | 47.8 % |
| **8–10k** | **81.8 %** | **18.6 %** | **22.7 %** |

**v3enc has the speed channel wired in and gets less than a quarter of its value — and the share is
falling, not rising.** MEASURED, matched-step, four arms, one metric.

---

## 2. Where the regression lives — term-level localisation (the key new evidence)

`grounding_losses` contributes two separable terms per level, both logged:

* **(a) `g_*_mid_de_m`** — metric inverse dynamics on **real latent pairs** `(z_t, z_{t+k}) → Δpose`.
  **No action input.** This is the *train-side twin of the gate's encoder speed probe*, and it is the
  term `v2_invdyn_gradscale` scales.
* **(b) `g_*_fwd_ade_m`** — forward metric consistency on the **predictor rollout with true actions**
  (which, under `speed_input`, carry v0).

| metric, 8–10 k bucket | v1 | no-speed | v3enc | v3enc/v1 |
|---|---:|---:|---:|---:|
| `g_op_mid_de_m` (a, encoder) | 1.2542 | 1.3331 | 1.4406 | **1.15×** |
| `g_tac_mid_de_m` (a) | 5.1572 | — | 5.8009 | 1.13× |
| `g_str_mid_de_m` (a, 2 s) | 4.1627 | 4.6480 | 5.7519 | 1.38× |
| `g_op_fwd_ade_m` (b, rollout) | 0.1052 | 0.5794 | 0.4717 | **4.48×** |
| `g_tac_fwd_ade_m` (b) | 0.3624 | — | 1.5152 | 4.18× |
| `g_str_fwd_ade_m` (b) | 0.4752 | — | 1.8492 | 3.89× |
| `inv` (A5 action inverse-dynamics) | 0.2194 | 0.3644 | 0.3784 | **1.72×** |

**The encoder's metric grounding is nearly intact (1.13–1.38×). The action-conditioned rollout is
4×.** Whatever v4 fixes, it is not primarily "the encoder lost speed capacity".

Two corroborating reads from `diagv2_summary.json` (same eval, same windows):

* **Step-1 operative speed R² = 0.9529** (v1 0.9987). At the first predicted step the model still
  decodes speed almost perfectly. The failure is **compounding over the rollout**, not at the readout.
* **The imagined latent barely moves.** Over 20 rollout steps: `znorm` 40.7 → 75.9 (**1.86×**) vs v1's
  40.1 → 147.9 (**3.68×**); `zcos` ends at **0.618** vs v1's 0.397. And the mean rollout speed profile
  is **non-monotone** — 12.91 → 12.14 (step 8) → 13.05 (step 16) → 12.63 — where v1's drifts monotonically
  12.78 → 13.25. v3enc's imagination is **damped and oscillatory**, i.e. an under-committed dynamics
  model, exactly what you get from training a rollout whose conditioning signal is unreliable.

---

## 3. The exhaustive lever diff — from both runs' `config.json`, not from memory

Pulled live: `tanitad-pod2:/workspace/experiments/flagship4b-speedjerk-30k/config.json` and
`tanitad-pod:/workspace/experiments/flagship4b-v3enc-30k/config.json`. **Fourteen differences, not
four.** The brief's list was incomplete in three places (rows 13–15, marked ⭐).

| # | knob | v1 | v2 | **v3enc** | class |
|---|---|---|---|---|---|
| 1 | `v2_ego_to_planners` | *absent* (false) | true | **true** | encoder-grounding |
| 2 | `v2_ego_dropout` | *absent* (0.0) | 0.25 | **0.25** | **encoder-grounding + action channel** |
| 3 | `v2_fa_dropout` | *absent* (0.0) | 0.30 | **0.15** | rollout |
| 4 | `train.rollout_k` | **4** | 12 | **4 <5k → 8 <10k → 12** | rollout |
| 5 | `v2_goal_decode` | *absent* | true | **true** | decode-side |
| 6 | `v2_nav_dropout` | *absent* (0.0) | 0.5 | **0.5** | strategic |
| 7 | `v2_traj_jerk` | *absent* (0.0) | 0.02 | **0.02** | decode-side |
| 8 | `v2_gated_intent` | *absent* | true | **true** | decode-side |
| 9 | `v2_anchor_tactical` | *absent* | true | **true** | decode-side (+9.5 M params) |
| 10 | `v2_route_from_vision` (w 0.3) | *absent* | true | **true** | strategic (2nd fwd pass/step) |
| 11 | `v2_encoder_ego_decorr` | *absent* | true, w 0.05 | **true, w = 0.0 until 10 k** | **INERT in the measured window** |
| 12 | `v2_invdyn_gradscale` | **1.0** (default) | 0.25 | **0.5** | encoder-grounding |
| 13 | `v2_labels` | *absent* (v1 labels) | true | **true** | data-side |
| 14 ⭐ | `aux_accel` head (528,897 params) | **true** | — | **ABSENT** | removed vs v1 |
| 15 ⭐ | `jerk_weight` (v1-style) | **0.02** | — | **ABSENT** | removed vs v1 |
| 16 ⭐ | `needed_fut` (future frames encoded/step) | **10** | 16 | **16** | +33 % encoder forwards/step |
| — | GPU | A40 (pod2) | A40 (pod2) | RTX A6000 (tanitad-pod) | — |

⚠️ **Reconstruction caveat that bounds every attribution below.** Registry §1.2 records that **v1 trained
with a pod-side trainer that was never committed** (`--jerk-weight`/`--aux-accel` are still absent from
the committed arg parser — re-verified). We are diffing v1's *recorded config* against v3enc's *code*.
Any un-recorded difference in v1's trainer body is invisible to this analysis. **UNVERIFIED and
unresolvable without committing the pod2 trainer diff.**

**Cost note (matters for "equal budget").** v3enc encodes **24 frames per sample** (window 8 + 16
futures) against v1's **18**, plus a second strategic forward pass and a rollout up to K=12 vs 4.
Log-derived cost is nonetheless within 5 % (v3enc 10.37 s/step, v1 10.89 s/step over a log containing one
resume). **So equal-step ≈ equal-wallclock here, but equal-step is *not* equal-FLOP — it flatters
v3enc.** ⚠️ v1's `summary.json` records `wallclock_s 191206` for 30 k steps = 6.37 s/step, which
contradicts its own log by 1.7×. Flagged UNRESOLVED; it does not change any conclusion.

---

## 4. Ranked attribution

Ranked by how well each lever explains **the specific signature**: *encoder metric grounding nearly
intact, rollout 4× worse, speed benefit ~23 % recovered, longitudinal bias cured, dynamics damped.*

### 4.1 🥇 `v2_ego_dropout = 0.25` — and specifically its **zero-fill of the v0 action channel**

**Mechanism (code, not inference).** `flagship_losses.py:214-217` builds a `keep_mask` at p=0.25 for the
planner ego vector. Then **`flagship_losses.py:227-231`**:

```python
if getattr(cfg, "speed_input", False):
    v0a = pose_last[:, 3:4] / 10.0
    if keep_mask is not None:
        v0a = v0a * keep_mask.to(v0a.dtype)     # <-- the SAME mask zeroes the SPEED channel
    actions      = cat([actions,      v0a.expand(...)], -1)
    fut_actions  = cat([fut_actions,  v0a.expand(...)], -1)
```

The base action channels are `(steer_rad, accel_mps2)` — **control-space, carrying no absolute speed**.
So on 25 % of samples the model's *only* speed input is set to **0.0**, which after `SPEED_SCALE=10` is a
perfectly in-distribution value meaning **"stationary"**. This is not masking; it is a **confident lie**,
on both `actions` and `fut_actions`, i.e. through the entire K-step rollout. There is no validity flag
and no learned null embedding.

**Why it explains *this* signature and the others don't:**
* It corrupts **term (b)** (rollout, action-conditioned) and leaves **term (a)** (real latent pairs, no
  action) untouched — exactly the 4.5×-vs-1.15× split measured in §2.
* It corrupts the **A5 action-inverse-dynamics target** (`loss_inv = (a_hat − actions[:,-2])²`), whose 3rd
  channel is v0. The encoder is therefore trained to make v0 *unrecoverable* from `(z_{t-1}, z_t)` on a
  quarter of samples. MEASURED: `inv` = **0.3784** for v3enc vs **0.2194** for v1 — and **0.3644 for the
  no-speed control**, i.e. **v3enc's action-inverse-dynamics carries no more speed information than an
  arm with no speed channel at all.**
* A model taught that a channel lies 25 % of the time learns to **discount it**, which is precisely the
  eval signature: speed MAE **1.8075** vs the hold-v0 floor **0.4818** (paired Δ −1.3258 [−1.5887,
  −1.0750], CI-separated **for the floor**), `tracks_speed_better_than_cv = false`, and a damped
  oscillating rollout speed.
* It is quantitatively over-strong: dropping a channel 25 % of the time should cost ~25 % of its value;
  v3enc loses ~77 % of it. Discounting, not sampling noise.

**Evidence against:** none found. **Confound:** it also inflates v3enc's own training metric (§1), so the
matched-step ratio over-states the deficit by an unknown amount — but the *eval* numbers are computed
with dropout off and are equally damning.

### 4.2 🥈 The rollout pack — `rollout_k` 4→8→12 **plus** `v2_fa_dropout = 0.15`

Three things made the rollout objective harder at once, against v1's constant K=4 with true actions:
recursion depth ×2 then ×3, 15 % of samples rolled with a zero-order-hold action substitute
(`flagship_losses.py:261-268`), and 25 % with a zeroed speed (§4.1).

* v3enc's `g_op_fwd_ade_m` **plateaus from ~4.5 k**; the 2 k-bucket ratio steps 1.95 → 2.63 across the
  **K 4→8 boundary at step 5,000**. *(Caveat: inspecting the ±100-smoothed series, v3enc is flat across
  5,000 and the ratio moves because **v1 keeps improving** — so the boundary is suggestive, not
  established. Marked WEAK.)*
* The damped-imagination read (`znorm` 1.86× vs 3.68×, `zcos` 0.618 vs 0.397) is what a hedging objective
  produces: when the conditioning signal is unreliable 40 % of the time (0.25 ∪ 0.15), the minimum-risk
  prediction is *move less*.
* v3enc's rollout is **better than v2's** (K=12 from step 0, fa_dropout 0.30) at every matched bucket, by
  21–30 % on `g_op_fwd_ade_m`. The staging helped; the direction was right and the dose still wrong.

### 4.3 🥉 `v2_invdyn_gradscale = 0.5` (v1: 1.0)

`config.py:224` states the intent plainly: it "**softly decouples the static ego-motion probe from the
encoder trunk**". The gate metric *is* an ego-motion probe of the encoder trunk. Halving the gradient of
the only action-free encoder→metric-ego-motion term is, by construction, adverse to that gate.

**But the measured effect is modest and cannot carry the failure.** Dose-response at matched steps
(`g_*_mid_de_m`, 6–8 k, the last bucket v2 reached):

| gradscale | arm | `g_op_mid_de_m` | `g_str_mid_de_m` |
|---|---|---:|---:|
| 1.0 | v1 | 1.4386 | 4.9707 |
| **0.5** | **v3enc** | **1.4797 (+2.9 %)** | **5.7222 (+15.1 %)** |
| 0.25 | v2 | 1.5964 (+11.0 %) | 6.8357 (+37.5 %) |

Monotone in the lever, in the predicted direction — **so the lever is real** — but a 3–15 % degradation
of encoder metric grounding does not turn a 0.861 probe into 0.393. **Verdict: contributory, not causal.**

### 4.4 `v2_ego_to_planners` (the shortcut the dropout was guarding)

Feeding `[v0, yr0]` directly to the strategic and tactical brains removes their incentive to read speed
from vision on 75 % of steps. Plausible contributor to a *held-out* probe collapse; **no isolating
measurement exists** (it was never run without the dropout, and v1 never had it). **UNVERIFIED.**

### 4.5 Removal of `--aux-accel` and `--jerk-weight` ⭐ *(the brief's list did not contain these)*

v1 carried a 528,897-param auxiliary acceleration head; v3enc does not. On the face of it this is a
removal of longitudinal-dynamics supervision — the obvious suspect. **It is not.** MEASURED from v1's own
log, `aux_accel_r2` across the run:

| step | 0 | 4k | 8k | 12k | 16k | 20k | 24k | 28k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `aux_accel_r2` | 0.003 | 0.031 | **−0.119** | 0.209 | **−0.423** | 0.049 | 0.447 | 0.328 |

**v1's auxiliary acceleration head never learned a stable signal** (it goes negative twice). v1's `jerk`
term ran 3e-4…3.7e-3 at weight 0.02 — a ≤1e-4 contribution to a total loss of ~4. **Verdict: not a cause,
and not worth restoring in v4.**

### 4.6 Decode-side levers (anchored tactical, goal decode, gated intent, traj-jerk) — **exonerated**

The anchored decoder is mechanically healthy: `n_modes` 1 → 13, `conf_norm` 40 → 189, `wta` 0.253 → 0.034
over 10 k steps; `man_acc` matches v1 (0.516–0.559 vs 0.566). And the level ordering is *the same as v1's*:
`ade2s_str` 1.5497 < `ade2s_tac` 1.6809 < `ade2s_op` 1.9654 (v1: 0.269 < 0.260 ≈ 0.427). **Nothing points
at the decode side.** *(Its isolated contribution to ADE is UNMEASURED — no ablation exists.)*

### 4.7 `v2_encoder_ego_decorr` — **inert, therefore unattributable**

CONFIRMED from code: `train_flagship4b.py:92` returns `decorr_w = 0.0` for `step < 10000`, applied at
`:428-429` into `weights.decorr`, and `flagship_losses.py:395` adds `weights.decorr * loss_decorr`. The
penalty was **computed and multiplied by zero** for the whole measured window (the log's `decorr` value
runs 0.077–0.136 the entire time — a monitored quantity with no gradient). **D-031's "decorr strangled
speed capacity" is refuted, and equally: decorr has produced zero evidence in two runs and must not be
carried forward as "known good".**

---

## 5. What v3enc did **BETTER** — do not let the post-mortem erase these

1. **The runaway longitudinal integrator is genuinely cured.** `highspeed_long_overshoot_m` **+2.195**
   (gate ≤ 8.0 ✅) against v2's **+23.7**; and the whole-set signed along-track error is **−0.5162
   [−1.4273, +0.4431]** — **not separated from zero**. v3enc has **no systematic longitudinal bias**.
   ⚠️ The honest completion: `long_abs_2s_m` is **3.2166** (v1 0.8412). **The bias is cured; the variance
   is not.** That is still a categorical improvement over v2, which had both.
2. **Straight-road heading MAE 3.642° vs v1's 7.980°** on the identical 634 straight windows — a real,
   large improvement on v1's single worst heading stratum. ⚠️ Two caveats that must travel with it:
   (i) both arms remain **CI-separated worse than the CV floor** (1.399°); (ii) v3enc is **4.9× worse on
   gentle curves** (10.085° vs 2.060°) and **4.7× worse on sharp** (17.777° vs 3.811°), so the aggregate
   heading MAE is a **tie** (6.706 vs 6.606). v3enc's heading error is *flatter across curvature*, not
   smaller. **Separate action item for v4: v1's 7.98° straight-road heading against a 1.40° floor is a
   v1 defect worth fixing on its own merits.**
3. **v3enc beats the CV floor on sharp-curve heading**: 17.777° vs 28.743°, paired Δ **+10.97
   [+7.18, +15.26]**, CI-separated **for the model**. It is not floor-beaten everywhere.
4. **`v2_nav_dropout` did exactly its job.** `route_acc` falls 0.98–1.00 (v1, no-speed) → 0.70–0.83
   (v2, v3enc) **at identical `nav_valid_frac`**. H26's finding was that v1's route head is a pure
   command-echo; removing the command removes the echo. **This is the lever working, not a label defect
   (§6).**
5. **Staging beat firing everything at once**, measurably: v3enc < v2 at every matched bucket, by 21–30 %
   on `g_op_fwd_ade_m` and 7–13 % on `g_op_mid_de_m`. The *method* is validated even though the arm failed.
6. **A less collapsed latent.** `erank` **27.7** vs v1's **21.2** (+31 %), `dim_std` +2 %. The encoder is
   spreading over more directions — the stated H25 goal. It simply did not convert into driving.

---

## 6. The label confound — checked, and **REFUTED as a cause**, by a stronger route than the gate's

The gate agent argued the broken pre-v2.1 route labels *cannot* remove scalar speed from the encoder,
because a 3-way lateral topology class at loss weight 0.5/0.3 cannot beat inv-dyn at 2.0. **That argument
is sound, and I can now quantify it and add a decisive independent check.**

**Confirmed, with numbers.** Route CE (`weights.route = 0.5`) and the route-from-vision aux
(`weights.route_vis = 0.3`) are **both masked by `nav_valid`**, so their effective mass is
`0.8 × 0.21 ≈ 0.17` against `invdyn = 2.0 applied at 3 levels = 6.0` on **100 %** of samples — a **~35:1**
ratio. And a 3-class *lateral* target carries no gradient direction toward or away from a *scalar speed*.

**The decisive new evidence — `nav_valid_frac` is the same in all four arms:**

| arm | 0–2k | 2–4k | 4–6k | 6–8k | 8–10k |
|---|---:|---:|---:|---:|---:|
| **v1 (deployed, probe 0.861)** | 0.2453 | 0.2500 | 0.2328 | 0.2078 | 0.2531 |
| no-speed control | 0.2453 | 0.2344 | 0.2703 | 0.2422 | 0.2500 |
| v2 | 0.2062 | 0.2266 | 0.2234 | 0.2118 | — |
| **v3enc** | 0.2062 | 0.2266 | 0.2234 | 0.2062 | 0.2281 |

**The deployed v1 — the arm with `encoder_speed_probe_r2 = 0.861` and ADE 0.4271 — trained on the same
~21–25 % route coverage.** The coverage defect is program-wide and pre-dates v3enc. It therefore cannot
be a v3enc-vs-v1 differentiator on *any* metric, let alone the speed probe. **REFUTED.**

**One refinement the gate agent did not make.** `v2_labels` also swaps the **maneuver** label to
`classify_maneuver_v2` (`refb_labels.py:320`), and the maneuver vocabulary *is* longitudinal
(`ACCELERATE`/`BRAKE_STOP`). But the only v2 change is a **curvature gate on the turn test**, and turn
has priority over brake/accel — so v2 labels make turns *rarer* and push **more** windows into the
longitudinal classes. The direction is toward *more* longitudinal supervision, not less. `man_acc` is
unchanged (v3enc 0.516–0.559 vs v1 0.566). **The label change cannot explain the speed failure either.**

**What the labels DO invalidate stands:** every route/strategic reading from this arm, and the gated-intent
path into the wp/goal heads. Unchanged from the registry.

---

## 7. Number-hygiene defects found while doing this — three corrections

1. 🟥 **"v1 reached v3enc's 0.4101 at step 450 (~22.7× faster)" is a noise artifact — do not repeat it.**
   Replicated from v1's raw log: the 3-point rolling median in `run_gate.py:286-300` first crosses at step
   450 because v1's raw `g_op_fwd_ade_m` at steps 300–550 is **0.758, 0.616, 0.404, 0.687, 0.384, 0.816**.
   A 3-point median cannot smooth a series that swings 2× between adjacent rows. v1's *bucket* means reach
   ≈0.41 in the **2 k–4 k** range (0.3389). **The defensible step-efficiency deficit is ~3.5–5×, not ~23×.**
   This does not change the RESTART verdict — that turned on the probe — but the 23× framing would badly
   mis-set v4's expectations. *(Suggest `reference_reached_at` require k consecutive crossings.)*
2. 🟥 **The "in-sample `ego_r2` 0.79–0.85 vs held-out probe 0.393 ⇒ generalisation failure" inference is
   not admissible.** `ego_linear_r2` (`decorr.py:84`) is an **in-batch, in-sample** ridge fit on B=16 with
   D=2048, and its own docstring says only its *trend* is meaningful. Decisive: **v3enc logs
   `ego_r2 = 0.595` at step 0 — on a randomly initialised encoder** — and it swings 0.59–0.94 row to row.
   It is largely a capacity artifact of a 2048-d probe on 16 points. It is also **not comparable** to
   `probe_speed_r2` (episode-held-out over 8 episodes, λ-grid, 881 val windows), and **v1 never logged it**
   (the key only exists when `v2_encoder_ego_decorr` is on), so there is **no control arm**. The
   generalisation story may still be true; it is **UNVERIFIED**, and §2 shows the encoder is not where the
   damage is anyway. *(The one genuinely held-out hint that survives: the probe selected **λ\* = 10.0**, the
   maximum of its grid, for v3enc vs **λ\* = 1.0** for v1 — the z→v0 map is less stable.)*
3. ⚠️ **A falling `route_acc` in a v2-family arm is not evidence of broken labels** — it is
   `v2_nav_dropout` removing the command-echo (§5.4). Reading it as label damage would send v4 chasing the
   wrong defect.

---

## 8. 🟥 DO-NOT-CARRY into v4 — with the reason per item

| # | Do not carry | Reason (evidence) |
|---|---|---|
| **1** | **`v2_ego_dropout` as implemented — zero-fill of the v0 action channel** | `flagship_losses.py:229-231` multiplies v0 by the keep-mask, so 25 % of samples are told "0.0 m/s" — an in-distribution lie — through `actions` **and** `fut_actions`. Measured: `inv` at the no-speed arm's level (0.378 vs 0.364 vs v1's 0.219); only **22.7 %** of v1's speed-channel benefit recovered at 8–10 k; eval speed MAE 1.81 vs hold-v0 0.48, CI-separated for the floor. **If v4 wants an anti-shortcut guard: (i) never touch the operative action channel — drop only the planner ego vector; (ii) use an explicit validity flag or a learned null embedding, never a zero on a channel whose zero is in-distribution; (iii) start at p ≤ 0.1.** |
| **2** | **`rollout_k > 4` before the operative rollout is healthy** | v3enc's `g_op_fwd_ade_m` **plateaus from ~4.5 k** at ~4× v1; v1 got 0.105 at 8–10 k with a constant **K=4**. K also drives `needed_fut` 10→16, +33 % encoder forwards per step — you pay 33 % more FLOPs for the regression. Raise K only after a matched-step check against K=4. |
| **3** | **`v2_fa_dropout` at 0.15+** | Hedging in the rollout loss; with (1) it makes the conditioning unreliable on ~40 % of samples. Damped imagination is measured: `znorm` ×1.86 vs v1's ×3.68, `zcos` 0.618 vs 0.397, non-monotone rollout speed. |
| **4** | **`v2_invdyn_gradscale < 1.0` while `encoder_speed_probe_r2` is a gate** | It is *designed* to decouple the ego-motion probe from the encoder trunk (`config.py:224`) — i.e. designed to lower the number the gate scores. Monotone dose-response measured (1.0 → 0.5 → 0.25 gives +0 % / +15 % / +38 % on `g_str_mid_de_m`). Effect is modest, the conflict of purpose is not. |
| **5** | **`v2_encoder_ego_decorr` as "already validated"** | It has been **inert for its entire measured life** (`decorr_w = 0.0` for step < 10 k). Two runs, zero evidence. If v4 wants it, it needs its own 2-arm ablation, not inheritance. |
| **6** | **More than ~2 encoder-touching levers per arm** | This is the actual repeat root cause: v2 fired 12 at once and died; v3enc softened 4 of them and plateaued. **Neither run can attribute its own failure** — that is a design defect in the *experiment*, not the model. v4 must change ≤2 encoder-touching levers per arm and keep a v1-identical control. |
| **7** | **`--aux-accel` / `--jerk-weight` (restoring v1's)** | v1's `aux_accel_r2` never stabilised (0.003 → −0.119 → −0.423 → 0.447); `jerk` contributed ≤1e-4 of a ~4.0 loss. They are neither the cause of the regression nor worth the params. |
| **8** | **The framing "v1 was ~23× more step-efficient"** | Noise artifact (§7.1). The defensible figure is **~3.5–5×**. |

### ✅ DO carry
* **Staging itself** — v3enc beat v2 at every matched bucket, on both grounding terms.
* **`v2_nav_dropout` (or an equivalent echo-killer)** — measurably removed the command-echo at equal
  nav coverage. It is the only lever in the pack with a clean, isolated, positive read.
* **The anchored tactical decoder** — mechanically healthy (n_modes 1→13, wta 0.253→0.034, `man_acc`
  at v1 level), no evidence against. *(Its ADE contribution remains UNMEASURED.)*
* **`speed_input` at full strength, undropped** — the whole gap in §1.1 is the difference between using
  it and half-using it.

### ⚠️ KEEP AS A GATE, FIX THE ESTIMATOR
`encoder_speed_probe_r2` is the right thing to gate on, but §2 shows it is **not where the damage was**.
v4's card should add a **rollout-side secondary** — e.g. matched-step `g_op_fwd_ade_m` against the
no-speed control (the "fraction of speed benefit recovered" statistic in §1.1), which localises the
failure in one number and is immune to the level.

---

## 9. What would settle the two open questions (cheap, CPU/1-GPU, no new 30 k run)

| # | Question | Experiment | Cost |
|---|---|---|---|
| A | Is §4.1 the cause? | Re-run v3enc's exact config for **2 k steps** with `v2_ego_dropout=0.0` (planner ego still fed). Read `g_op_fwd_ade_m` and `inv` against the archived v3enc log at matched steps. | ~6 GPU-h |
| B | How much of the matched-step ratio is the measurement artifact? | Evaluate `ckpt_step10000.pt` **with the ego-dropout mask forced off vs on** on train batches. No training. | <1 GPU-h |
| C | Is the encoder recoverable? | Fit the held-out episode-disjoint speed probe on `ckpt_step5000.pt` **and** `ckpt_step10000.pt` (both archived on `tanitad-pod`) — does the probe fall over training, or start low? | CPU + 1 encode pass |

⚠️ `tanitad-pod:/workspace/experiments/flagship4b-v3enc-30k/` holds `ckpt.pt` (10,800),
`ckpt_step10000.pt` and `ckpt_step5000.pt`. **10 k was not on D-032's archive list — it is the only 10 k
state that will ever exist.** Do not let that pod be recycled before (B) and (C) run.

---

## 10. Provenance / deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| **This note** | `repo: TanitAD Research Hub/Architecture & Inference/Research/2026-07-21-flagship-v3enc-postmortem.md` (staged) | no |
| v3enc gate + card | `repo: Project Steering/Gates/flagship-v3enc-gate-10k-2026-07-21.json`, `flagship-v3enc.card.json` | no |
| v3enc eval + driving + paired | `repo: taniteval/results/{flagship-v3enc-10k,driving_flagship-v3enc-10k,paired_v3enc10k_vs_flagship30k,diagv2_summary}.json` | no |
| **v1 / v3enc / v2 / no-speed `train_log.jsonl` + `config.json`** | `tanitad-pod2:/workspace/experiments/{flagship4b-speedjerk-30k,flagship4b-phase0-30k,flagship4b-v2-30k}/` · `tanitad-pod:/workspace/experiments/flagship4b-v3enc-30k/` · working copies in the session scratchpad | **⚠️ YES — pod-only.** See escalation below |
| v3enc checkpoints 5 k / 10 k / 10.8 k | `tanitad-pod:/workspace/experiments/flagship4b-v3enc-30k/` | **⚠️ YES — pod-only** |

**No GPU was used.** All work was CPU/offline plus read-only `ssh`/`scp` of small text files; `gpu_lock.sh`
was not needed and pod3's VLM lock was not touched. No file outside the note was modified.

---

## 🔺 ESCALATION — three items that need a decision, not a footnote

1. **Four training logs and two configs that every number in this post-mortem depends on exist ONLY on
   pods.** `flagship4b-speedjerk-30k/train_log.jsonl` is the *deployed model's* only training record and
   pod2 is the pod that has already filled its quota twice. **Ask: commit the four `train_log.jsonl` +
   `config.json` pairs (≈1.5 MB total) into the repo.** I did not stage them myself because adding pod
   artifacts to the tree is a repo-policy call, not mine.
2. **`run_gate.py`'s `reference_reached_at` produces a noise-driven answer** (§7.1) and it has already
   propagated into `MODEL_REGISTRY.md` §1.4 as "step 450". Needs a one-line fix (require k consecutive
   crossings, or use a bucket mean) **and** a registry correction.
3. **Registry §1.4 should be amended** with: the complete 14-item lever diff (§3, incl. the three items
   its list omits), the term-level localisation (§2), the `nav_valid_frac` parity that refutes the label
   confound (§6), and the retraction of the "step 450 / 22.7×" framing.

---

# Appendix B (2026-07-21) — §9 row B executed: the zero-fill priced, and half of §1's matched-step gap retracted

> **VERDICT: PARTIAL — mechanism CONFIRMED, magnitude halved.**
> At fixed weights, feeding `v0 = 0.0` costs the operative rollout **3.58×** (0.2633 → 0.9421 m), and
> the p=0.25 mask accounts for **39.2 %** of the masked metric and **~51 %** of the matched-step gap
> to v1. §8's DO-NOT-CARRY #1 is **earned**. But **~50 % of the matched-step gap survives the mask
> being switched off** (corrected ratio **2.69×**, not 4.48×), and the **held-out eval gap (§1 rows
> 1–2) is untouched** — TanitEval already runs `.eval()`, so it never built the mask. **The zero-fill
> is a real and expensive defect; it is not the whole root cause.** v4 keeps its zero-fill rule and
> must NOT retire DO-NOT-CARRY #2/#3 on the strength of it.

**Estimator.** Per-window values; point estimate = full-set mean over **6,400 train windows drawn
uniformly from the canonical corpus** `physicalai-train-e438721ae894` (2,192 distinct episodes);
intervals = **paired episode-cluster bootstrap, B = 2000** (`taniteval/ci.py`, byte-copy md5
`ef925f06febd20a99f5901491fcf75cb`). ⚠️ Train windows are drawn uniformly over 2,376 episodes, so
most episodes contribute 1–3 windows and the cluster bootstrap here **degenerates gracefully toward
a window bootstrap** — it is *not* the 40-episode val construction and must not be quoted as one.
Reference bucket means are **recomputed from the raw `train_log.jsonl`** (rows deduped on `step`,
keeping the last occurrence — v1 and the no-speed arm replay steps after a resume), which moves them
a hair from §1–§2's prose: v1 `g_op_fwd_ade_m` **0.1062** (not 0.1052), no-speed **0.5740** (not
0.5794), v1 `inv` **0.2149** (not 0.2194). No conclusion in the note turns on that.

## B.1 What was run

`ckpt_step10000.pt` opened **read-only** (`step` field verified = 10000; size and mtime
`2026-07-21 10:13:04.089217322` **unchanged after the run**). **No training, no optimizer step, no
checkpoint written.** Five action-channel conditions on the **same batches, same latents, same
seed** — only the third action channel varies:

| condition | v0 fed to `actions` **and** `fut_actions` |
|---|---|
| **off** | true `v0/10` on every row — the mask forced OFF |
| **on25** | `v0/10 × keep`, `p_drop = 0.25` — the training condition (realised keep = 0.7464) |
| **zero** | `0.0` on every row — the mask forced ON for all rows |
| **perm** | another row's `v0/10` — a **wrong but in-distribution** speed (supplementary run) |
| **x2** | `2 × v0/10` — sensitivity readout |

`on25` is also reported **analytically** as `0.25·zero + 0.75·off`, the exact expectation over mask
draws on these windows with no mask-sampling noise. Realised and analytic agree to 0.4 % (0.4346 vs
0.4330 on `g_op_fwd_ade_m`), so the particular draw does no work.

**Why the pairing is exact, and what it is NOT a test of.**
* The encoder runs **once per batch**; `states`/`fut_states` are shared byte-for-byte by all
  conditions (they cannot depend on actions).
* The flagship 4-brain graph contains **no `Dropout` and no `BatchNorm`** (asserted at runtime), so
  the `eval()` forward is bit-identical to the trainer's `train()` forward.
* The per-window extraction was cross-checked against the real `grounding_losses` — the function
  that wrote the training log — on batch 0, agreeing to the log's 4-decimal rounding for every level
  and every condition.
* **Precision is not a factor.** A full **fp32 replicate** on the identical 6,400 windows reproduces
  every quantity to within **0.5 %** and the artifact share to three decimals (op **0.3919** bf16 vs
  **0.3932** fp32). The supplementary run reproduces the main run's `off`/`zero`/`on25`/`x2` values
  **bit-exactly**.
* ⚠️ **`g_*_mid_de_m` is a null BY CONSTRUCTION, not a finding.** Grounding term (a) reads only
  `(z_t, fut_states)` and **no actions at all** (`metric_dynamics.py:362-372`), so it is exactly
  mask-invariant; the harness measured `max|Δ| = 0.0` at all three levels. The brief's prediction
  "encoder grounding does not move" is *trivially* satisfied and **B supplies no evidence about the
  encoder.** Only §9 row C can.
* ⚠️ The model's A5 output `a_hat` is likewise bit-identical across conditions (`max|Δ| = 0.0`), so
  **100 % of the `inv` movement in B.3 is target corruption**, never a behavioural change. The `inv`
  columns must **not** be read as sensitivity.

**One methodological correction the run forced.** §4.2 says the conditioning is unreliable on ~40 %
of samples (`0.25 ∪ 0.15`). For the **logged grounding metric** only the 0.25 applies:
`flagship_losses.py:349` passes `fut_actions` to `grounding_losses`, **not** `fa_roll`, so
`v2_fa_dropout` never touches `g_*_fwd_ade_m`. The ego-dropout zero-fill is the **only** lever in the
v3enc pack that contaminates the metric §1 row 3 is built on — which is what makes B a clean
isolation.

## B.2 The rollout term (b) — the headline

`v1` / `v3enc` / `no-speed` are raw-log **8–10 k bucket means**; `on`/`off`/`zero` are measured at the
step-10 000 weights (bf16, the trainer's own precision).

| `g_*_fwd_ade_m` | v1 log | v3enc log | **on (meas.)** | **off (meas.)** | zero | artifact share | **recovered fraction of the gap** | logged ratio → **corrected** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **op** | 0.1062 | 0.4699 | **0.4330** | **0.2633** | 0.9421 | **39.2 %** | **50.6 %** | 4.42× → **2.69×** |
| **tac** | 0.3700 | 1.5088 | 1.5166 | 0.9231 | 3.2972 | 39.1 % | 51.8 % | 4.08× → **2.48×** |
| **str** | 0.4798 | 1.8446 | 1.8170 | 1.0556 | 4.1014 | 41.9 % | 56.6 % | 3.84× → **2.23×** |

*artifact share* = `(on − off)/on` — assumption-free, internal to this checkpoint.
*recovered fraction* = the share of the **logged** `v3enc − v1` bucket gap removed by rescaling the
logged bucket by `off/on` (multiplicative transport; the additive transport gives 46.7 / 52.1 /
55.8 %, so the choice does not matter). All paired contrasts are CI-separated — on
`g_op_fwd_ade_m`: `on − off` **+0.1697 [+0.1617, +0.1780]**, `zero − off` **+0.6788 [+0.6469,
+0.7121]**, `x2 − off` **+2.1799 [+2.1158, +2.2398]** (paired episode-cluster bootstrap, 2,192
episodes, B = 2000).

**The measured `on` reproduces the training log.** 0.4330 measured vs **0.4699** logged (8–10 k) and
0.4738 (10–12 k) — an 8 % offset in the expected direction, since the checkpoint is the *end* of the
window the bucket averages. The harness is measuring the quantity the log wrote.

**§1.1's cleanest statistic, corrected.** Fraction of the speed-channel benefit recovered,
`(nospeed − arm)/nospeed`, at 8–10 k on `g_op_fwd_ade_m`:

| | v1 | v3enc **as logged** | v3enc **mask-corrected** |
|---|---:|---:|---:|
| speed benefit recovered | 81.5 % | **18.1 %** | **50.2 %** |
| as % of v1's | 100 % | **22.2 %** | **61.6 %** |

**"v3enc has the speed channel wired in and gets less than a quarter of its value" is retracted.**
The mask-free figure is **~62 % of v1's** — clearly worse than v1, but not the categorical failure
§1.1 reported, and the "share is falling" reading rests on the same contaminated series.

## B.3 `inv` (A5 action inverse-dynamics)

| | v1 log | v3enc log | on (meas.) | **off (meas.)** | zero | no-speed log |
|---|---:|---:|---:|---:|---:|---:|
| `inv` (3-channel mean) | 0.2149 | 0.3759 | 0.3979 | **0.3465** | 0.5520 | 0.3723 ⚠️ |

`inv` moves in the predicted direction but only **28 % of the way**: mask-off takes it 0.3979 →
**0.3465** (paired Δ **−0.0514 [−0.0552, −0.0474]**, CI-separated), i.e. **28.1 %** of the gap to v1's
0.2149 (30.2 % transported onto the logged bucket). **72 % of the `inv` gap is not the mask.**

⚠️ **A comparability defect in §4.1 / §8 row 1, found while doing this.** The no-speed control
`flagship4b-phase0-30k` has **`predictor.action_dim = 2`** — its `inv` averages **(steer, accel)
only**. v1 and v3enc have `action_dim = 3` and average a **third** channel (v0). *(Read from each
run's own `config.json`.)* **"v3enc 0.3784 vs the no-speed control's 0.3644" is a 3-channel mean
against a 2-channel mean — not the same statistic**, so it cannot support *"v3enc's
action-inverse-dynamics carries no more speed information than an arm with no speed channel at
all."* Channel-matched, the conclusion reverses:

| | steer | accel | v0 | **2-channel mean (steer, accel)** |
|---|---:|---:|---:|---:|
| v3enc @10 k, mask off | 0.0041 | 0.5721 | 0.4632 | **0.2881** |
| no-speed control, 8–10 k log | — | — | n/a | **0.3723** |

**v3enc's non-speed inverse dynamics is 23 % BETTER than the no-speed control's.** What lifts its
3-channel mean is the v0 channel alone (0.4632). *(v1's per-channel split is **UNMEASURED** — the log
records only the 3-channel mean and v1 has no 10 k checkpoint to re-measure.)*

The encoder's speed content is nevertheless genuinely weak mask-free: the A5 head's own v0 output
explains **R² = 0.4976** of true v0 on train windows. Not a fitted probe (it is the model's own
prediction), but it *is* train data and has **no v1 control** — quotable only as *"v3enc's A5 head
recovers about half of v0's variance in-sample"*, marked **UNVERIFIED** as a comparative.

## B.4 The sub-claim B refutes — and the sharper one that replaces it

§4.1 argued: *"A model taught that a channel lies 25 % of the time learns to **discount** it, which is
precisely the eval signature."* **Measured: the opposite, and something more interesting.**

Feed a deliberately wrong v0 and ask what fraction of the resulting error the rollout actually
realises. A rollout that tracked the fed speed exactly would incur
`ADE = 0.25 · E|v_fed − v_true|` over op `fwd_k = 4` at 0.1 s/step:

| fed v0 | `E|Δv0|` (m/s) | ADE if it swallowed the lie whole | measured **excess** ADE | **realised fraction** |
|---|---:|---:|---:|---:|
| **zero** (the mask value) | 12.97 | 3.241 | 0.679 | **20.9 %** |
| **perm** (another window's v0) | 10.13 | 2.533 | 1.825 | **72.0 %** |
| **×2** | 12.97 | 3.241 | 2.180 | **67.3 %** |

*(Same normalisation by RMS perturbation: damage per unit RMS = zero **0.421**, perm **1.383**,
`perm/zero` = **3.29×**. Both normalisations agree.)*

**The model is not discounting v0 — it follows a wrong speed almost literally (67–72 %). It discounts
v0 at exactly ONE value: 0.0, where it realises only 21 % and falls back on vision.** In other words,
**the zero-fill taught the model to build an implicit null embedding at the sentinel value** — the
very mechanism v4 wants, built accidentally and built badly, because that sentinel is aliased with a
real driving state:

| | |
|---|---:|
| train windows genuinely stopped (`v0 < 0.5 m/s`) | **6.45 %** |
| windows **presented** as `v0 = 0` under the p=0.25 mask | **29.84 %** |
| of those, genuine | **21.6 %** |
| of those, **a lie** | **78.4 %** |

*(v0 over the same 6,400 windows: mean **12.97 m/s**, std **9.60**, range 0–43.5.)*

**Consequence for the note's causal chain:** §4.1's bridge from the training-time zero-fill to the
*held-out eval* signature (speed MAE 1.81 vs the hold-v0 floor 0.48) went through "the model
discounts v0". That step is false. The eval-side speed failure therefore still needs its own
explanation, and **fixing the zero-fill is not guaranteed to fix it.**

## B.5 What B settles, and what it explicitly cannot

**Settles.** The number here is the **inference-time (measurement) contribution of the mask at fixed
weights** — how much of the *logged* metric is the mask rather than the model: **39.2 %** of the
metric, **~51 %** of the matched-step gap.

**Cannot settle.** B probes a **trained checkpoint**. It cannot separate *"corruption baked into the
weights during training"* from *"corruption applied at eval"*. The surviving **2.2–2.7×** is
consistent with **either** genuine weight damage from the same zero-fill **or** the other levers
(`rollout_k` 4→8→12, `fa_dropout`, `invdyn_gradscale`). **Only §9 row A** — 2 k steps re-run with
`v2_ego_dropout = 0.0` — **can split those**, and B raises rather than lowers its value.

**Out of scope.** §1 rows 1–2 (eval `ade_0_2s` 4.60×, and 3.19× against the 19 k relay) are
**unaffected**: TanitEval loads with `.eval()` (`taniteval/loaders.py:151`), so `model.training` is
False and `flagship_losses.py:215` never builds the mask. Those are already mask-free measurements
and stand exactly as published.

## B.6 Corrections this appendix forces on the body of the note

| § | as published | corrected |
|---|---|---|
| **§0 / §1 row 3** | "strictly matched step … **4.48×**"; "essentially none of the gap survives being explained away by budget" | The matched-step ratio is inflated ~1.6× by the mask. Mask-free: **2.69×** (op), 2.48× (tac), 2.23× (str). A **≥2.2× deficit** survives every framing — not ≥3× on the matched-step read. The *eval* framings (4.60×, 3.19×) are unaffected. |
| **§1** | "inflates v3enc's number by an amount **we cannot isolate** without a re-run" | Isolated: **39.2 %** of the metric, **50.6 %** of the gap. |
| **§1.1** | v3enc recovers **18.6 %** of the speed benefit at 8–10 k = **22.7 %** of v1's | Mask-free: **50.2 %** = **61.6 %** of v1's. |
| **§2 table** | `inv` row "0.2194 / 0.3644 / 0.3784, **1.72×**" | Channel mismatch: the 0.3644 column is a **2-channel** mean (see B.3). Mask-free v3enc `inv` is **0.3465** (1.61× v1); channel-matched, v3enc's steer+accel `inv` **0.2881** *beats* the no-speed control's **0.3723**. |
| **§4.1** | "a model … learns to **discount** it" | **REFUTED.** It follows a wrong speed at 67–72 %; it discounts **only** the value 0.0 (21 %). The mechanism is an accidental null embedding, not discounting. |
| **§4.2** | conditioning unreliable on "~40 % of samples (0.25 ∪ 0.15)" | For the **logged metric**, only the 0.25 applies — `grounding_losses` receives `fut_actions`, not `fa_roll`. |
| **§9 / §10** | `ckpt.pt` described as step 10,800 | **MEASURED: `ckpt.pt` carries `step = 10000`.** `--ckpt-every 1000` put the last save at 10,000; the run reached 10,800 but never checkpointed again, and identical size + mtime confirm `ckpt_step10000.pt` is its archive copy. **The 10 k state is held in duplicate on that pod, not singly** — good news for the D-032 gap, though still one disk. |

## B.7 What this means for v4 (the reason B was run)

1. **KEEP the zero-fill rule** (`V4_FLAGSHIP_DESIGN.md` §5.3 / X15, and P5b's learned null row for
   `flagship_v15.py:348-351` — **confirmed still live**, at `ego_dropout = 0.5`, i.e. **twice** the
   flagship's rate, while the goal and route paths in the *same function* already use learned
   DROPPED embedding rows). Zeroing v0 costs **3.58×** on the operative rollout at fixed weights, and
   78 % of the zeros the model sees are lies. The rule is earned.
2. **The learned-null-row fix is the right shape, and B shows why:** the model *already* builds a
   null behaviour at the sentinel, just implicitly and aliased with genuinely stopped vehicles
   (6.45 % of windows). An explicit validity flag / null row removes the aliasing; a bare zero
   cannot.
3. **Do NOT retire DO-NOT-CARRY #2/#3 on the strength of this.** Half the matched-step gap and *all*
   of the held-out gap survive the mask.
4. **Drop the "the model discounts v0" narrative** from v4's rationale — measurably false, and it
   would mis-aim the design. The failure is not under-use of v0.
5. **§9 row A is now the highest-value follow-up** (~6 GPU-h): the only experiment that can attribute
   the surviving 2.2–2.7×.

## B.8 Appendix manifest

| artifact | where | only copy? |
|---|---|---|
| Results JSON (all conditions, paired CIs, derived) | `repo: taniteval/results/postmortem_b_egodropout_v3enc10k.json` (staged) | no |
| Raw per-precision measurements | `tanitad-pod:/workspace/expb/exp_b_{bf16,fp32,bf16_perm}.json`, `v0_stats.json` — all folded into the results JSON above | no |
| Experiment harness, stage 1 (pod) | `repo: taniteval/postmortem_b_egodropout.py` + `taniteval/postmortem_b_v0_stats.py` (staged) · deployed copies at `tanitad-pod:/workspace/expb/` | no |
| Experiment harness, stage 2 (analysis) | `repo: taniteval/postmortem_b_analyze.py` (staged) — recomputes the reference buckets from the raw logs and writes the results JSON | no |
| no-speed + v2 train logs & configs (rescued from pod2 for the reference buckets) | `repo: taniteval/results/trainlogs/{nospeed-phase0,v2}_{train_log.jsonl,config.json}` (staged) | no — resolves 2 of the 4 pod-only logs in §10's escalation |
| `ckpt_step10000.pt` | `tanitad-pod:/workspace/experiments/flagship4b-v3enc-30k/` — **read-only, size+mtime unchanged** | ⚠️ pod-only (in duplicate with `ckpt.pt`) |

**GPU:** `gpu_lock.sh acquire exp-b` held on pod1 for the whole run and released; pod3's
`vlm-production` lock untouched. Total GPU time **≈ 40 min** across bf16 + fp32 + the supplementary
run (the spec's budget was <1 GPU-h for one run; the fp32 replicate and the `perm` arm are extra).
