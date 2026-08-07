# The OOD-val four-family eval protocol — what v1arch is, and how it is scored

**Status:** live. **Evidence class:** MEASURED (pod4, 2026-08-05) unless a line says otherwise.
**Supersedes nothing.** Adds the corpus + harness that make a v1-shaped checkpoint scorable
against the binding four-family rule.

---

## 0. The one-line summary

`flagship-v1arch-v2bal-30k` is the **v1 ARCHITECTURE trained on the v2bal 9000-clip pool**. Every
`v2_*` lever in its own `config.json` is **false**. That is what makes the PI's stated experiment —
*"I wanted to understand the effect of more and better distributed data"* — cleanly attributable:
**architecture held constant, data varied.** It is not a v2-architecture arm and must never be
quoted as one.

MEASURED, from the run's own `config.json` (`cfg` block), pod4 2026-08-05:

```
speed_input            = true          predictor.action_dim   = 3
v21_route_labels       = false         tactical_policy        = {n_maneuvers 5, cadence 5,
v2_anchor_tactical     = false                                   waypoint_horizons [5,10,15,20]}
v2_ego_dropout         = 0.0           strategic_policy       = {n_commands 4, n_route 3,
v2_ego_to_planners     = false                                   d_ctx 256, cadence 20}
v2_encoder_ego_decorr  = false         h15                    = {enabled true, mask_prob 0.5}
v2_fa_dropout          = 0.0
v2_gated_intent        = false         checkpoint keys        = [grounding, model, opt, step]
v2_goal_decode         = false         step                   = 29999   (clean finish)
v2_invdyn_gradscale    = 1.0
v2_labels              = false
v2_nav_dropout         = 0.0
v2_route_from_vision   = false
v2_traj_jerk           = 0.0
```

---

## 1. ⛔ Why `eval_flagship_v4.py` cannot produce a single binding family here

`stack/scripts/eval_flagship_v4.py` gates its full metric path on

```python
is_v4 = isinstance(ck, dict) and ("head" in ck) and not a.canary_only
```

v1arch's checkpoint keys are `['grounding','model','opt','step']` — **no `head`** — so on this
checkpoint that script can only ever run `MODE_A_canary_only_validation`. It **exits 0**, prints an
ADE, and emits **no per-window `pred`/`gt` at all**. That is a *structural* limit, and it reads
exactly like a completed eval, which is the dangerous part.

⇒ The four-family path is **`taniteval/tools/eval_four_families.py`**:

| pass | gives | why the other one cannot |
|---|---|---|
| `rollout.collect` | LONGITUDINAL + LATERAL (dense 10 Hz path) | a fidelity pass never traverses the decision heads |
| `hierarchy.run` | TACTICAL + STRATEGIC | it is the only pass that runs `strategic_policy` → `tactical_policy` |
| `four_families.all_families` | assembles them, **per-family, never pooled** | — |

---

## 2. The corpus, and why it is not the canonical val

**`physicalai-oodval-6f4b94e4c7ce`** — 290 episodes, PhysicalAI-AV's **own official eval split**
(`reasoning/ood_reasoning.parquet`, 1740 rows, `{train: 1450, val: 290}`).

⚠️ **RETRACTED on 2026-08-05: "PhysicalAI ships no eval split."** I searched 70,775 *filenames*, got
0 hits, and never read the dataset card. Root-cause class: **absence found at ONE location is not
absence** — already in `CLAUDE.md`, and it still caught me. Logged in `RETRACTION_LOG.md`.

**Why this corpus and not the canonical 40-episode val:** v1arch's config records
`"v2_parity": {"parity": false, "checked": false, "clips_present": 9000}` and
`"require_parity": false`. **21 of the 40 canonical val episodes are inside its 9000-clip training
pool.** The 290 official-val clips have **ZERO overlap** with that pool. See
`LEAK_v1arch_val_2026-08-05.md`.

⛔ **Numbers on this corpus are NOT comparable to any canonical-val number** until the other arms are
re-scored on it. The corpus key travels in every output record precisely so that cannot be forgotten.

### 2.1 Format parity with training — the q90 sibling

v1arch trained on `.v2ep.pt`, whose frames are `tvio.encode_jpeg(frame, quality=90)` decoded at load
(`scripts/v2_compressed.py:140`). The rebuilt corpus stores **raw uint8**, so the model would be
scored on pristine pixels it never saw. `physicalai-oodval-6f4b94e4c7ce-q90` applies
`decode(encode(x, quality=90))` per 3-channel sub-frame — the exact training transform.

MEASURED: **max |pixel delta| = 185** (the confound is real in pixels) but

| corpus | `ade@2s` | n_windows |
|---|---|---|
| raw | 0.5727 | 6,382 |
| q90 | **0.5705** | 6,382 |

**Δ = −0.0022 (~0.4 %)** — negligible in the metric. Both arms are scored and reported; **q90 is the
headline** because it is the format-faithful one.

### 2.2 Sampling contract — checked, not assumed

- **three past images:** the episodes are `[T, 9, 256, 256]` — `n_stack=3` × 3 channels. ✅
- **frequency:** `hz=10`, matching training; `dt_s = 0.1` on the dense path. ✅
- **geometry:** `{'size': 256, 'n_stack': 3, 'hz': 10, 'calib': 'ftheta_v2'}`,
  `frame_tag = 256x256f266pin`, first-clip `per_clip: True`. Cache key matched the build params. ✅

---

## 3. The harness corrections, each earned by a measured defect

| correction | the defect it removes |
|---|---|
| `hierarchy.run(..., max_eps=len(eps))` | its default is **40**. On 290 episodes that silently scores 14 % — the same class as the `--episodes` default that gave **0.4892** against the full-corpus **0.5727**, 17 % optimistic. |
| `--out` refuses a directory **before** `loaders.load` | measured: `IsADirectoryError` fired only *after* the whole scoring pass. |
| CI components come from `four_families._seq_geometry` | a second implementation lets the interval and the point estimate drift apart silently. |
| the record NAMES `overlapping_holdout_se` as unused | a bare "CI" invites the reader to assume the banned estimator, which **biases the point estimate**, not only the interval. |
| `corpus` + `corpus_key` + `episodes_scored/available/flag` in the record | a truncated denominator must not survive the log scrolling away. |
| `_vision_only` travels with the JSON | `route_acc_nav` feeds the model a **label-derived** nav command; v1 scored exactly **1.0000** on it — an echo of its own input read as skill. `route_acc_follow` is the deployable read. |

---

## 4. Pod-sync check at FUNCTION granularity — `git log` was not available and would not have sufficed

`/workspace/TanitAD` on pod4 is **not a git checkout**, and four files' whole-file md5s differ from
HEAD. Whole-file hashes cannot say whether the *eval* is affected. Fingerprinting only the functions
the eval calls settles it:

```
classify_maneuver · nav_command · route_target · wrap_to_pi · maneuver_labels    IDENTICAL to HEAD
StrategicPolicy · TacticalPolicy · WorldModel · run_hierarchy                    IDENTICAL to HEAD
```

`taniteval/rollout.py` and `four_families.py` **did** differ and HEAD's were pushed to the pod
(md5-verified in both directions), so the eid fix and the decision-capture seam are live there.

⇒ Generalisation for the runbook: **when a pod tree has no `.git`, hash the FUNCTIONS the job calls,
not the files.** A whole-file diff over-reports (comments, unrelated helpers) and a bare
`import tanitad` under-reports (the `stack_guard` shadowing case).

---

## 5. What each family can and cannot report on this corpus

| family | status | note |
|---|---|---|
| LONGITUDINAL — speed / along-track / accel / ego-progress | **OK** | on the dense 10 Hz path |
| LONGITUDINAL — distance-keeping (headway / time-gap / TTC) | ⛔ **UNAVAILABLE** | our ingest reads 4 of 36 PhysicalAI features and `obstacle.offline` is not among them. **A WORK ITEM, not a pass** — the reader exists (`build_lead_tracks.py`); the corpus build must be extended. |
| LATERAL — heading / curvature / yaw-rate / cross-track | **OK** | |
| TACTICAL — 5-way manoeuvre, per-class + `never_predicted` | **OK** | via `hierarchy.run`; GT from `refb_labels.classify_maneuver` (labels may use ego) |
| STRATEGIC — route/goal setting | **OK, legacy path only** | ⛔ the **option-set** path is impossible here: PhysicalAI ships **no map** — the card says verbatim *"we do not include open maps data"*. So the route label is read off the ego's own future yaw and **cannot tell whether a branch existed**. `GATE_PROTOCOL §0.7`'s `nonav_route_beats_majority` is **VOID BY CONSTRUCTION** and the flag travels in the output. |

---

## 6. Reading rules that travel with any number from this protocol

1. **ADE is one row of four families.** Never present a horizon sweep of ADE as "the result".
2. **Not comparable to canonical-val arms** until those arms are scored on this corpus.
3. **The OOD split is harder by construction** — it is PhysicalAI's own out-of-distribution
   selection, not a random holdout. A worse absolute number here is not evidence of regression.
4. **`route_acc_follow`, never `route_acc_nav`.**
5. **A family reported UNAVAILABLE is a WORK ITEM, not a pass.**
