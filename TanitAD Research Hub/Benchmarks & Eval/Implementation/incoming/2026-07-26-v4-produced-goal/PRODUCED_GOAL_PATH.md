# The flagship-v4 produced-goal path — and what the 30 k gate must be read against

**Date:** 2026-07-26 · **Pod:** `tanitad-eval` (A40, 0 MiB at start; pod1/pod2/pod3 untouched)
**Checkpoint:** `flagship-v4-fromscratch` **@ step 15000** (`/workspace/models/flagship-v4-fromscratch-15k/ckpt_step15000.pt`, already local — nothing transferred from pod2)
**Estimator, everywhere below:** **`paired_episode_cluster_bootstrap`** / `episode_cluster_bootstrap` (`taniteval/ci.py`), B = 2000, 40 val episodes, 881 stride-8 windows, percentile CI α = 0.05. **`overlapping_holdout_se` is not used anywhere in this work.**
**Not done here, on purpose:** the 30 k gate itself (checkpoint does not exist yet; the orchestrator owns it). Nothing was committed — files are written, **not** `git add`ed.

---

## 0. The one-paragraph answer

The v4 gate's primary is a **goal-oracle number**, and on our own arm at 15 k the privilege is worth
**+0.1738 m of ADE@2s [+0.1247, +0.2356], `paired_episode_cluster_bootstrap`, CI-separated, p = 1.0** —
a **+29.8 %** degradation when the goal is taken from the model instead of from the ego's future.
That is large enough to flip two verdicts: the oracle number **passes** the 0.60 m gate primary
(0.5839) and the deployable number **fails** it (0.7577); and **`oracle` beats constant velocity
CI-separated (−0.2538 [−0.4337, −0.1124]) while `produced` does not (−0.0800 [−0.2503, +0.0661])**.
The mechanism is measured, not inferred — the only model-side goal producer on this checkpoint
(`GoalScalarHead`) regresses the curvature the route derives from at **R² = 0.0749**, so the produced
route collapses to `straight` on **90.6 %** of windows. Two findings I did not expect: the gap is
**entirely longitudinal** (along +0.1815 separated, cross +0.0114 **not** separated), i.e. what the
arm actually buys from the oracle is **target speed**, not route; and the produced goal is
**significantly worse than no goal at all** (+0.1012 [+0.0600, +0.1460] vs the learned null rows).
**The 30 k gate should still render on `oracle` (the historical, comparable surface) but must be
reported as `MODE B, goal-oracle inputs, ADE@2s = X`, and read against a produced-goal number that is
now one flag away.**

---

## 1. ⚠️ FIRST: the eval pod was stale — much worse than reported

**MEASURED** (md5 of every `.py`, pod vs repo working tree, CRLF normalised to LF; full lists in
`EVAL_POD_STALENESS.json`). The wheelbase agent's report — *"`rollout.collect` predates the dense-path
upgrade"* — was correct but **understated the blast radius by an order of magnitude**:

| tree | pod path | repo files | stale (md5 ≠) | missing | **wrong** |
|---|---|---:|---:|---:|---:|
| `taniteval` package | `/root/taniteval/taniteval` | 37 | 18 | 5 | **62.2 %** |
| `stack/scripts` (eval subset) | `/root/v4eval/stack/scripts` | 18 | 11 | 4 | **83.3 %** |
| `tanitad` package | `/root/v4eval/stack/tanitad` | 86 | 39 | 6 | **52.3 %** |

The load-bearing items:

- **`taniteval/rollout.py` stale** → no `pred_dense`/`gt_dense`. Confirms the brief: `lateral.block`
  cannot run off it. (`lateral.py` itself was already **current** — the break was upstream.)
- **`taniteval/hierarchy_guard.py` MISSING** → the *current* `rollout.collect` imports it, so the
  sync could not be partial: copying `rollout.py` alone would have produced `ModuleNotFoundError`.
- **`taniteval/corridor.py` MISSING** → `GATE_PROTOCOL.md` §0's **co-primary emitter did not exist on
  the eval pod.** A 30 k gate run there today could not have produced a corridor block at all.
- **`taniteval/blind_baseline.py`, `strategic_probes.py`, `label_overlay.py` MISSING.**
- **`tanitad/data/parity.py` MISSING** → the val-parity guard (`assert_val_cache`) was inert.
- **`stack/scripts/goal_provenance.py` MISSING** → the goal-oracle disclosure banner §0.8 relies on
  had never reached the pod.

### 1.1 A SECOND stale tree the brief did not name — and it is the default one

`/root/TanitAD/stack` is a git clone pinned at **`0f93b98`**, and **every `taniteval` submodule
hard-codes it**: `sys.path.insert(0, "/root/TanitAD/stack")` appears in `bench.py`, `closedloop.py`,
`corridor.py`, `refc_eval.py`, `cam_overlay.py`, `corpus_overlay.py`, `blind_baseline.py`, … It held
**`tanitad/lake/` with 1 of 16 modules** and no `parity.py`.

I found this the way the operating standard says to — *absence found at one location is not absence*:
an import smoke that imported `taniteval.refc_eval` **before** `tanitad.data.parity` failed, because
importing any taniteval submodule silently prepends the stale tree to `sys.path`. Had I probed only
`/root/v4eval/stack` I would have declared the pod clean.

**Synced both trees.** `/root/v4eval/stack` (the v4 eval PYTHONPATH) **and** `/root/TanitAD/stack`
(the default every taniteval submodule injects), plus `/root/taniteval/taniteval`.

### 1.2 What was done, and how it is verified

1. Backup: `/root/_bak_20260726_produced_goal/` (taniteval pkg · v4eval stack · v4eval results ·
   the `/root/TanitAD/stack` `tanitad`+`scripts` trees) — 8.8 MB, nothing destroyed.
2. Payload built locally, **LF-normalised**, md5-verified in flight
   (`721ee506949d9f65ff52c5d98ac0d154` both ends). `code/sync_eval_pod.sh`.
3. `parity_manifest.json` shipped separately (it is a `.json`; a `*.py`-only payload would have left
   the freshly-synced `parity.py` unable to find its manifest — caught before the first run).
4. All `__pycache__` in the synced trees removed.

**Verification — three independent checks, all MEASURED:**

| check | result |
|---|---|
| md5 of every synced file vs the LF-normalised repo | **141 / 141 identical**, 0 missing, 0 extra |
| import smoke, 14 modules incl. `corridor`, `hierarchy_guard`, `parity`, `lateral` | **14 / 14 ok**; `rollout.collect` emits `pred_dense` ✔ ; `dense_speed_profile` present ✔ |
| **MODE A on flagship v1** (`/root/models/flagship-30k/ckpt.pt`) | **`canary_ade@2s = 0.4214799702167511`**, n = 881 |

v1's number is **bit-identical to the pre-sync run** (`0.4214799702167511`, `results/v1-validation.json`,
2026-07-23) — same float64 digits — and sits **−0.0056** from the registry full-set **0.4271** (tolerance
0.05), so `HARNESS_VALIDATED: true`. The sync changed **no number** on a known checkpoint, which is
exactly what a sync should do. Artifact: `/root/v4eval/results/v1-postsync.json`.

`stack/` test suite after all edits: **`pytest -q` → 1020 passed, 3 skipped.**

### 1.3 ⚠️ A residual that is NOT staleness — a live unit bug in the current repo

The brief attributes *"`paired_cross_track` reports `horizon_s=0.4` against a 4-knot surface"* to the
stale pod. **It is not.** `taniteval/lateral.py:529` (current repo, post-sync) stamps

```python
horizon_s=round(step * DT, 2)          # DT = 0.1, unconditionally
```

`paired_cross_track` takes no `dt`. On the **`sparse_4wp`** surface a knot is **0.5 s**, so `step=4`
(truly **2.0 s**) is labelled **0.4 s**. `from_sparse_windows` re-stamps its own horizons for exactly
this reason (`out["horizon_s"] = out["horizon_K"] * 0.5`); `paired_cross_track` was not given the same
treatment. Every lateral horizon in this report is **re-stamped at 0.5 s spacing** in
`code/analyse_gap.py`. **Escalated, not buried** — this is a one-parameter fix (`dt=DT`) in
`taniteval/lateral.py` and it will mislabel the 30 k gate's paired lateral block if left.

---

## 2. The implementation — `--goal-mode {oracle,produced,neutral}`

New module **`stack/scripts/goal_modes.py`**; wired into **`stack/scripts/eval_flagship_v4.py`**.
`goal_provenance.py` is left alone — its own docstring promises it "changes NO number and NO code
path", so the switch does not live there.

```
--goal-mode oracle     # DEFAULT, unchanged, bit-identical to the historical path
--goal-mode produced   # two-pass, the model's own goal — THE DEPLOYABLE PATH
--goal-mode neutral    # the head's learned no-goal-given rows — the control
--goal-fallback        # opt-in; lets `produced` degrade to `neutral` on a ckpt with no goal_head
```

The mode is stamped into **every** output JSON — `goal_provenance.goal_mode`, plus
`goal_mode` / `goal_provenance_short` / `primary_is_deployable_capability` in the gate summary, and
`goal_mode=…` inside `method` on the persisted windows dump. An artifact cannot be read without it.

### 2.1 🔴 CORRECTION: it is THREE oracle fields, not four

`GATE_PROTOCOL.md` §0.8 and `goal_provenance.py` both name **`route`, `route_graded`, `vt_band`,
`vt_speed`**. Reading the function that actually assembles them —
`train_flagship_v4._goal_inputs`, the single call site shared by the trainer and the evaluator:

```python
if cfg.cond_vtarget:
    kw["vt_band"]  = batch.get("vt_band", ...)     # ORACLE — future speed
    kw["vt_speed"] = v0                            # <-- NOT the batch field
```

**`vt_speed` is overwritten with `v0 = pose_last[:, 3]`, the last OBSERVED speed.** The batch's
future-derived `vt_speed` never reaches the head, in training or in eval. The selection penalty it
drives (`FlagshipV15Head.select`: `-|v_term − clamp(vt_speed, v0 ± reach)|`) therefore reduces to a
**hold-v0** term. `vt_speed` is an observation, not a goal oracle.

This does **not** shrink the problem — the three remaining channels are the ones the S3 firewall
measured (`route`/`route_graded` alone lifted a no-image baseline QWK 0.1128 → 0.3381) — but the
docs should say three. Recorded in every `goal_provenance` block as `vt_speed_note`.

### 2.2 What `produced` does

Two passes over **one** encode of the observation window:

- **pass 1** — `goal_head(states[:, -1])` where `states = world.encode_window(frames)`. This is the
  exact call `train_flagship_v4.v4_loss_step` trains on. **Frames only**: no future, no label, no
  batch goal field is read.
- **pass 2** — the four produced scalars `(ttm, curv_3s, curv_5s, tspeed_5s)` are mapped into the
  head's goal channels and `FlagshipV4Head` is run on them.

The `goal_head` is loaded from the checkpoint with `in_dim`/`hidden`/`n_out` read off **its own tensor
shapes**, not assumed (the design allows either `z_strat`=128 or `state_dim`=2048).

### 2.3 The mappings — and where each is an approximation, stated not hidden

| channel | produced from | honest status |
|---|---|---|
| `route_graded` | `tanh(curv_5s_pred / refb_labels.CURV_TURN_PER_M)` | **the label's OWN formula.** `route_graded ≜ tanh(mean_curv / CURV_TURN_PER_M)`, `mean_curv = net_dyaw/arc`; `curv_5s` is that same functional quantity (`v4_labels.mean_curvature`). ⚠️ **horizon differs**: label integrates the adaptive route horizon (≤ 25 s), produced is exactly 5 s |
| `route` (3-class) | threshold `\|graded\| ≥ tanh(1)` ⇔ `\|mean_curv\| ≥ CURV_TURN_PER_M`; sign → left/right | ⚠️ **approximation of a differently-defined label.** The labeler thresholds `peak_kappa` **with a transience gate**; the head produces no `peak_kappa`. Agreement is measured, §4.3 |
| `vt_band` | `lake.vocab.vtarget_band(tspeed_5s_pred)` | ⚠️ **same banding function, different statistic.** Label `vt_band` = `vtarget_v2` = 85th percentile of future speed over **10–20 s** dropping steps braking > 1.5 m/s²; `tspeed_5s` = smoothed speed at exactly **5 s** |
| `vt_speed` | `v0`, unchanged | **not an oracle** — see §2.1 |

### 2.4 A channel with no model-side producer — named, not invented

**`vt_band` has no faithful producer on this checkpoint.** The head regresses `tspeed_5s`, which is a
different statistic over a different horizon from the label. The honest options were: (a) map through
the shared banding function and stamp the mismatch, or (b) feed the learned no-vtarget row. **I chose
(a) and stamped it**, because (b) silently withholds a channel the arm was trained with and would
inflate the measured gap. The alternative — a hold-last-observed fallback, i.e.
`vtarget_band(v0)` — is *available and defensible* and is what I would use if the goal were a
deployment stub rather than a measurement; it is **not** used here because it would conflate "the
model cannot produce a target speed" with "the model's target speed is its current speed".

**`route`'s discrete class has no faithful producer either** — the `peak_kappa`+transience rule is not
in the head. There is **no route classifier anywhere in `FlagshipV4Head`** (the 15 k head has
`route_emb` / `route_graded` — *consumers* — and no `route_logits`). This is the same gap
`eval_flagship_v4.py`'s own `nonav_route_beats_majority: null` note already recorded.

**Refusal, not substitution.** On a checkpoint with **no** `goal_head`, `--goal-mode produced`
**raises and refuses**; only an explicit `--goal-fallback` degrades it, and the result is then stamped
`fallback: "neutral"` and `"this is NOT a produced-goal number"`. Verified in the CPU smoke.

---

## 3. `oracle` is bit-identical — verified, not asserted

Compared against **`/root/v4eval/results/windows_flagship-v4-fromscratch-15k.pt`**, written
**2026-07-25 22:07 Z** by the **pre-sync, pre-`--goal-mode`** `eval_flagship_v4.py`
(`50a114e3…`) on the **stale** stack. So the comparison spans *both* changes at once.

**`ALL_BIT_IDENTICAL: true`** — every persisted tensor, byte for byte
(`results/GOAL_MODE_GAP.json` → `bit_identity_oracle_vs_prepublished_baseline`):

| tensor | bit-identical | max abs diff | md5 (new == baseline) |
|---|---|---:|---|
| `pred` | ✅ | **0.0** | `08cf2973943c…` == `08cf2973943c…` |
| `gt` | ✅ | 0.0 | `ae78378677a9…` == `ae78378677a9…` |
| `cv` | ✅ | 0.0 | `b41f21e17c49…` == `b41f21e17c49…` |
| `speed` | ✅ | 0.0 | `b28b13e129f5…` == `b28b13e129f5…` |
| `head_deg` | ✅ | 0.0 | `b50e0a4c8202…` == `b50e0a4c8202…` |
| `eid` | ✅ identical, n = **881** | — | — |

And the derived scalars reproduce to the last **float64** digit:

| quantity | baseline (2026-07-25) | post-sync + `--goal-mode oracle` |
|---|---|---|
| `ade_0_2s` (self-computed) | `0.5839243086729363` | **`0.5839243086729363`** |
| `dense_headhorizons_ade_2s` | `0.4595726001817441` | **`0.4595726001817441`** |
| `dense_headhorizons_oracle_ade` | `0.2400555380458323` | **`0.2400555380458323`** |
| `dense_headhorizons_sel_gap` | `0.21951706130712986` | **`0.21951706130712986`** |
| `wp4_oracle_ade_0_2s` | `0.2797142271117374` | **`0.2797142271117374`** |
| `wm_canary_ade_2s` | `2.073894500732422` | **`2.073894500732422`** |

*(Structural, not lucky: `--goal-mode oracle` **delegates to `train_flagship_v4._goal_inputs`
verbatim** rather than reimplementing it, so it cannot drift from the historical path. The CPU smoke
asserts key-set and tensor equality between the two — `code/smoke_goalmode.sh`.)*

Two things this proves, and one it does not:

- ✅ the `--goal-mode` edit **did not move the evaluated forward pass** — `pred` is byte-for-byte the
  same tensor;
- ✅ **nor did the full stack sync** — 62 %/83 %/52 % of the eval-relevant Python changed underneath
  and the v4 planner numbers did not move by one float64 digit (nor did v1's, §1.2);
- ⚠️ it does **not** prove the stale stack was harmless in general — it proves the stale stack was
  harmless **for these two code paths**. `corridor.py` and `lateral.block`, which were the actually
  broken surfaces, produced nothing before and produce something now; there is no "before" number
  for them to be identical to.

---

## 4. The gap — oracle vs produced

All three modes: **same checkpoint, same 40 episodes, same 881 windows, same `gt`** (asserted
tensor-equal in `code/analyse_gap.py` before any comparison). Evidence class **MEASURED (ours)**;
artifacts in `results/`.

### 4.1 The levels

`episode_cluster_bootstrap` (`taniteval/ci.py`), B = 2000, 40 episodes, 881 windows.
These reproduce `taniteval.driving`'s own `cluster_bootstrap.model` block exactly — two independent
paths over the same persisted windows.

| goal mode | ADE@2s | 95 % CI | `miss@2m` | deployable? |
|---|---:|---|---:|---|
| **`oracle`** (historical) | **0.5839** | [0.4962, 0.6821] | 0.1691 | ❌ upper bound |
| `neutral` (no goal at all) | 0.6565 | [0.5553, 0.7749] | 0.2054 | control |
| **`produced`** (model's own goal) | **0.7577** | [0.6621, 0.8692] | 0.2293 | ✅ **the real number** |
| constant-velocity baseline | 0.8377 | [0.6352, 1.0899] | — | — |
| *flagship v1 reference* | *0.4271* | *(registry full-set, different arm — a POINT reference, not paired)* | | |

### 4.2 The gap — `paired_episode_cluster_bootstrap`, B = 2000

| pair | Δ ADE@2s (m) | 95 % CI | separated | p(Δ>0) |
|---|---:|---|:-:|---:|
| **`produced` − `oracle`** | **+0.1738** | **[+0.1247, +0.2356]** | ✅ | 1.000 |
| `neutral` − `oracle` | +0.0726 | [+0.0387, +0.1132] | ✅ | 1.000 |
| **`produced` − `neutral`** | **+0.1012** | **[+0.0600, +0.1460]** | ✅ | 1.000 |
| `produced` − `oracle`, **p90 tail** | +0.3572 | [+0.1282, +0.5217] | ✅ | 0.999 |

**⭐ The headline: the goal oracle is worth +0.1738 m (+29.8 %) on our own arm — and the model's own
goal is significantly WORSE THAN NO GOAL AT ALL** (+0.1012 [+0.0600, +0.1460], separated). The
learned "no goal given" row degrades gracefully; a *confidently wrong* route/vtarget does not. So the
+0.1738 decomposes into **+0.0726 "having any goal at all"** and **+0.1012 "the produced goal is
actively harmful"** — and only the first part is what a better goal producer could recover.

### 4.3 ⭐⭐ The decisive read — does it beat constant velocity?

`paired_episode_cluster_bootstrap` of **model − CV** on the same 881 windows
(`results/CV_BASELINE_PAIRED.json`). Negative = the model wins.

| goal mode | Δ vs CV | 95 % CI | **beats CV, CI-separated** |
|---|---:|---|:-:|
| `oracle` | −0.2538 | [−0.4337, −0.1124] | ✅ **yes** |
| `neutral` | −0.1812 | [−0.3432, −0.0515] | ✅ yes |
| **`produced`** | **−0.0800** | **[−0.2503, +0.0661]** | ❌ **NO — CI includes zero** |

**With the goal oracle, flagship-v4 @ 15 k clearly beats constant velocity. With its own goal, it is
statistically indistinguishable from constant velocity.** That is what "a goal-oracle number is not a
deployed capability" means, in one line, on our own arm.

### 4.4 Where the gap lives — it is **entirely longitudinal**

`taniteval.lateral.decompose(mode="ego")`, paired, produced − oracle. **Horizons re-stamped at the
0.5 s knot spacing** (§1.3).

| axis | oracle | produced | paired Δ | 95 % CI | separated |
|---|---:|---:|---:|---|:-:|
| **along** (longitudinal) | 0.4283 | 0.6098 | **+0.1815** | **[+0.1280, +0.2452]** | ✅ |
| **cross** (lateral) | 0.2819 | 0.2932 | +0.0114 | [−0.0012, +0.0242] | ❌ |

Per horizon, longitudinal — the compounding signature, separated at every knot:
**0.5 s +0.0392** [+0.0287, +0.0519] · **1.0 s +0.1188** [+0.0820, +0.1626] ·
**1.5 s +0.2345** [+0.1676, +0.3180] · **2.0 s +0.3335** [+0.2311, +0.4546].
Lateral is **not separated at any of the four** (2.0 s: +0.0297 [−0.0018, +0.0637]).
Longitudinal energy share rises 0.6191 (oracle) → **0.7371** (produced).

**Read this carefully — it is a genuine surprise.** The channel the S3 firewall indicted is
`route`/`route_graded`, which is *lateral* intent; but on this arm the oracle's value is almost
entirely *longitudinal*, i.e. it comes through **`vt_band`**. Two different measurements of two
different things (S3: QWK of a no-image baseline; here: ADE@2s of the real arm), and they do **not**
license each other. What our arm actually buys from the oracle is **target speed**.

### 4.5 Why — the produced goal's own quality, MEASURED

R² of the checkpoint's `GoalScalarHead` against the kinematic labels it was trained on, on the same
881 windows (`results/GOAL_MODE_GAP.json` → `produced_goal_quality`):

| scalar | R² | n valid | note |
|---|---:|---:|---|
| `tspeed_5s` | **0.7145** | 721 | the best-learned scalar |
| `ttm` | 0.3117 | 283 | |
| `curv_3s` | 0.2166 | 584 | |
| **`curv_5s`** | **0.0749** | 625 | **← the scalar `route` / `route_graded` derive from** |

Consequences, measured:

- **`route`: exact agreement with the oracle 0.4994** (n = 881). The produced route predicts
  **`straight` on 90.58 %** of windows (left 5.56 %, right 3.86 %) against an oracle distribution of
  left 24.06 % / straight 44.72 % / right 13.73 % / unknown 17.48 %. At R² 0.075 the curvature
  regression is near-blind, so the derived class **regresses to the majority** — the same
  regress-to-mean failure this program has logged before.
- **`vt_band`: exact 0.1317, within-one-band 0.3417.** Despite `tspeed_5s` being the *best* scalar
  (R² 0.71), the band agreement is poor — because the label is a **different statistic** (85th pctile
  of future speed over 10–20 s) from what the head regresses (smoothed speed at exactly 5 s). §2.3.
  This is the honest limit of the mapping, not a bug in it.

### 4.6 A KILL secondary that the harness calls "NOT REACHABLE" is now reachable

`eval_flagship_v4.py` reports `nonav_route_beats_majority: null` ("no ROUTE classifier exists"). True —
but the **produced** route can be scored against the majority baseline. On the 727 windows where the
*oracle* route is a real judgement (the produced head cannot emit `ROUTE_UNKNOWN`, so those 154
windows are **unscoreable, not wrong**):

- produced accuracy **0.6052** · majority-class accuracy **0.5420** · **beats majority by +6.33 pp.**

It clears the bar, but only just, and via a curvature-regression proxy rather than a route head.
**This supports — with a positive number rather than an absence — `LOOP_STATE`'s standing
instruction that this secondary is VOID BY CONSTRUCTION at the 30 k gate and must be adjudicated
INSTRUMENT-FAIL, not MODEL-FAIL.**

### 4.7 Bonus, verified: `lateral.block` now runs natively on a v4 windows dump

`collect_planner` discarded 16 of the head's 20 dense steps, so `lateral.block` — the 10 Hz
decomposition `GATE_PROTOCOL.md` §0's co-primary is read against — **skipped** on every v4 arm. The
head already computes the dense plan and the dense target already exists, so this was persistence,
not compute. Added **additively**; `pred`/`gt` keep their exact 4-waypoint meaning.

Verified on a fresh `--goal-mode oracle` run (`results/DENSE_PATH_VERIFICATION.json`):

- ✅ **`pred` STILL bit-identical** to the baseline (`08cf2973943c…`) — the addition moved nothing;
- ✅ `pred_dense` / `gt_dense` `(881, 20, 2)`, `dense_steps` 1…20, `dt_s` **0.1**;
- ✅ `pred == pred_dense[:, wp_steps−1]` **exactly**; `gt` vs `gt_dense[:, wp_steps−1]` max |Δ| =
  **8 × 10⁻⁶** (two independent GT code paths agreeing to 8 µm);
- ✅ **`lateral.block` runs**: `skipped: False`, `dt_s 0.1`, `horizon_K 20`, `horizon_s 2.0`,
  `axis_check.verified: true`. Longitudinal share **0.6434**; lateral grows **×15.255** vs
  longitudinal **×10.4462** over 0.5 → 2.0 s (cross faster by ×1.46) — **20 growth points, not 4**,
  and the horizon labels are now correct (the sparse surface reported "0.1 → 0.4 s" for the same span).

A guard refuses to emit dense keys when the head's horizons are not a contiguous 10 Hz `1..K`
(e.g. the tactical instance's `5,10,…,50`) — a non-10 Hz surface labelled as one is precisely the
mismatch this whole section is about.

---

## 5. What the 30 k gate should do

**The recommended reading, in one sentence:** render the gate on **`oracle`** (it is the only surface
comparable to the 10 k/15 k history), word it as §0.8 requires, and **immediately re-render the same
checkpoint with `--goal-mode produced`** — the pair is the verdict, and the deployable member of the
pair is the one that answers "does this arm drive".

**1. Render on `--goal-mode oracle`. Do not switch the gate's primary mid-flight.**
Every v4 MODE-B number in the record (v4.1-10k, v4.2-4000, v4-fromscratch-15k) is an oracle number.
Making `produced` the gate primary at 30 k would compare the 30 k arm against a differently-fed
history — the exact error §0.8 point 4 forbids for REF-C's `follow_constant`. The default is
unchanged for this reason, and `oracle` is now proven bit-identical (§3), so the 30 k number lands on
the same surface as its predecessors.

**2. Word the verdict as §0.8 point 2 requires — the harness now does it for you.**
Admissible: *"MODE B, goal-oracle inputs, ADE@2s = X."* Inadmissible: *"the flagship achieves X."*
Every result JSON now carries `goal_provenance.goal_mode`, `goal_provenance_short` and
`primary_is_deployable_capability` **inside the gate summary**, next to the number they qualify.

**3. Then run `--goal-mode produced` on the SAME 30 k checkpoint. It costs ~5 minutes.**
```bash
python3 eval_flagship_v4.py --ckpt <30k ckpt> --anchors-dense <anchors> \
    --head-config <config.json> --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
    --goal-mode produced --key flagship-v4-30k-produced \
    --out  /root/v4eval/results/flagship-v4-30k-produced.json \
    --results-dir /root/v4eval/results --episodes 40 --stride 8 --batch 16
```
Report **both**, always as a pair. The gap is the deployability discount and it belongs in the
registry row, not in a footnote.

**4. ⚠️ Do NOT transplant +0.1738 to the 30 k number as a correction.**
The gap is a function of `GoalScalarHead` quality, and that head is still training (`curv_5s` R² was
0.075 at 15 k). It could shrink a lot or not at all. **Measure it at 30 k.** Quoting the 15 k gap
against the 30 k number would be an INHERITED number deciding a GPU-day.

**5. Read the 0.60 m primary TWICE — at 15 k the two readings disagree.**

| | ADE@2s | vs 0.60 threshold |
|---|---:|---|
| `oracle` (what the gate will print) | 0.5839 | **PASS** |
| `produced` (what the arm can actually do) | 0.7577 | **FAIL** |

If the 30 k oracle number lands anywhere near 0.60, the deployable number is very likely well over it.
A CONTINUE justified only by the oracle reading should say so explicitly.

**6. The strongest available bar is free: does `produced` beat constant velocity, paired and
separated?** At 15 k it does **not** (−0.0800 [−0.2503, +0.0661]). This is a better question than the
0.60 threshold because CV is on the same windows and needs no threshold chosen in advance.
`code/cv_test.py` (in `results/CV_BASELINE_PAIRED.json`) runs it in seconds off the persisted windows.

**7. Adjudicate `nonav_route_beats_majority` as INSTRUMENT-FAIL, and now cite a number for it.**
§4.6: the produced route beats majority by **+6.33 pp** (0.6052 vs 0.5420, n = 727) — via a
curvature-regression proxy, because **no route classifier exists in `FlagshipV4Head`**. `LOOP_STATE`'s
standing instruction to adjudicate INSTRUMENT-FAIL is correct and now has positive evidence behind it.

**8. Two eval-pod preconditions that would have silently degraded the gate — both now closed, one
still open in the repo.**
- ✅ **`taniteval/corridor.py` was MISSING from the eval pod.** The horizon-honest **co-primary
  emitter did not exist there.** A 30 k gate run this morning could not have produced a corridor
  block at all. Synced.
- ✅ `lateral.block` skipped on every v4 dump. Fixed additively and verified (§4.7) — re-run the eval
  with the current `eval_flagship_v4.py` and the co-primary's lateral panel is available at 10 Hz.
- 🔴 **OPEN: `taniteval/lateral.py:529` `paired_cross_track` has no `dt`** and stamps
  `horizon_s = step × 0.1` unconditionally (§1.3). On a 4-knot surface it under-reports the horizon by
  **5×**. One-parameter fix. **If the 30 k gate quotes a paired lateral horizon off the sparse
  surface, it will be mislabelled.**

**9. What the gap does NOT say.** It is one checkpoint (15 k), one corpus, open-loop, 2 s. It does not
transfer to REF-C (different architecture, different goal channel) and it may not be compared to
REF-C's 0.4728 / 0.4714, which were collected with the route input **never exercised** (§0.8 point 4).
The `neutral` control is what makes the number attributable at all — without it, "+0.1738" could not
be split into "having a goal" (+0.0726) and "having a *bad* goal" (+0.1012).

---

## 6. Deliverable manifest

**Nothing is committed and nothing is `git add`ed** (per the brief). Everything below is in the repo
working tree, so nothing is stranded on a pod.

### Repo — source changes (working tree, unstaged)

| path | what |
|---|---|
| **`stack/scripts/goal_modes.py`** | **NEW.** The `oracle` / `produced` / `neutral` switch, the mappings and their stated approximations, `GoalAgreement` (produced-vs-oracle quality), the provenance block |
| **`stack/scripts/eval_flagship_v4.py`** | `--goal-mode` + `--goal-fallback`; loads `goal_head`; goal resolution in `collect_planner`; provenance stamped in every mode; goal mode in the gate summary and in `method`; **additive `pred_dense`/`gt_dense` emission** |

`pytest -q` on `stack/`: **1020 passed, 3 skipped** (run in `C:\Users\Admin\venvs\tanitad`).

### Repo — deliverables

`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-v4-produced-goal/`

| file | what |
|---|---|
| `PRODUCED_GOAL_PATH.md` | this document |
| `EVAL_POD_STALENESS.json` | the pre-sync md5 audit, full per-file stale/missing lists |
| `results/GOAL_MODE_GAP.json` | **the main result** — bit-identity, levels, paired gaps, lateral/longitudinal, goal quality, route-vs-majority |
| `results/CV_BASELINE_PAIRED.json` | paired model − CV per goal mode |
| `results/DENSE_PATH_VERIFICATION.json` | dense keys + `lateral.block` running + `pred` still bit-identical |
| `results/v1-postsync-harness-validation.json` | MODE A on v1 after the sync (0.42148 vs registry 0.4271) |
| `results/v4-15k-goal-{oracle,produced,neutral}_v4_diagnostics.json` | the three gate summaries |
| `results/driving_v4-15k-goal-{oracle,produced,neutral}.json` | `taniteval.driving` panels (episode-cluster bootstrap) |
| `results/v4-15k-goal-produced.json` | full produced-mode result incl. the `goal_provenance` block |
| `code/goal_modes.py`, `code/eval_flagship_v4.py` | copies of the two source files as run |
| `code/sync_eval_pod.sh`, `code/verify_sync_eval_pod.sh` | the pod sync and its verification |
| `code/smoke_goalmode.sh` | the CPU smoke (oracle == `_goal_inputs`, legal indices, refusal behaviour) |
| `code/run_goalmodes.sh`, `code/analyse_gap.py`, `code/cv_test.py`, `code/verify_dense.sh` | the runs and the analysis |

### Eval pod (`tanitad-eval`) — state left behind

| path | what |
|---|---|
| `/root/taniteval/taniteval/` · `/root/v4eval/stack/` · `/root/TanitAD/stack/` | **synced to the repo working tree**, 141/141 md5-verified (+ `parity_manifest.json`) |
| `/root/_bak_20260726_produced_goal/` | pre-sync backup of all three trees + the prior results (8.8 MB) — **delete only after someone confirms the gate is happy** |
| `/root/v4eval/results_goalmode/` | the four runs' windows dumps + JSONs (the `.pt` files are pod-only; every derived number is in the repo) |
| `/root/v4eval/results/v1-postsync.json` | the post-sync harness validation |
| `/root/{do_sync,manifest,verify_sync,smoke_goalmode,run_goalmodes,verify_dense,peek}.sh`, `/root/{analyse_gap,cv_test}.py` | the scripts as run |

**pod1 / pod2 / pod3 were not touched.** Nothing was read from pod2's rolling checkpoint; the 15 k
checkpoint was already local to the eval pod.

### Escalations (not buried in a doc — these need an owner)

1. 🔴 **`taniteval/lateral.py:529` `paired_cross_track` needs a `dt` parameter.** It mislabels the
   horizon by 5× on the `sparse_4wp` surface. One line. **Before the 30 k gate quotes a paired
   lateral horizon.**
2. 🟠 **`GATE_PROTOCOL.md` §0.8 and `goal_provenance.py::_ORACLE_FIELDS` list `vt_speed` as an oracle
   field. It is not** (§2.1) — `_goal_inputs` overwrites it with the last observed speed. The code now
   discloses the correction in every provenance block; the documents should be corrected by their
   owner.
3. 🟠 **`goal_provenance.py::AFFECTED_PUBLISHED_NUMBERS` can now be quantified for v4 MODE B:** the
   privilege is **+0.1738 m [+0.1247, +0.2356]** at 15 k. `v1.5` / `v1.6` (0.4375) remain
   unquantified — the same `--goal-mode` treatment applied to `eval_flagship_v16.py` would close
   that, and the v1.6 head has the same `route_emb`/`vtarget_emb` conditioning geometry.
4. 🔴 **Why the eval pod drifted 52–83 % undetected — READ, not guessed.** The nightly
   `stack/scripts/pod_git_drift.py` **cannot see this class of problem, for two independent reasons:**
   - **`SEARCH_ROOTS = ["/root", "/workspace"]` with `find … -maxdepth 3`** (line 42 / 117). So
     `/root/taniteval/taniteval/rollout.py` (depth 3) is covered, but
     `/root/v4eval/stack/scripts/eval_flagship_v4.py` (depth 4),
     `/root/v4eval/stack/tanitad/data/parity.py` (depth 5) and the whole of
     `/root/TanitAD/stack/tanitad/**` (depth 5) are **below the horizon**. The two `stack/` trees that
     were 52 % and worse wrong were never scanned.
   - **It asks the wrong direction.** Its two questions are *"is this pod file in git?"* (POD_ONLY) and
     *"does the pod file match the repo?"* (DRIFTED). **A repo file MISSING from the pod is invisible
     to it by construction** — and that is exactly what took out `corridor.py` (the co-primary
     emitter), `hierarchy_guard.py`, `parity.py` and `goal_provenance.py`.

   The tool is doing its job — *"a pod is not storage"* — but the gate needs the **converse** check:
   *is the pod running the repo?* `code/verify_sync_eval_pod.sh` + the md5 manifest in
   `code/sync_eval_pod.sh` are a working two-command version of it. **A gate rendered on a 62 %-stale
   harness is not a gate**, and nothing in the fleet would currently say so.
