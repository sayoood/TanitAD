# REFCV3_ARM — the eval adapter for REF-C v3 (`refcv3`), and the definition of what its arm IS

**Instrument:** `taniteval/tools/refcv3_arm.py` · **owner:** Benchmarks & Evals FlyWheel ·
**written:** 2026-09-03 · **status of every number it can produce today:** UNVERIFIED on a real
checkpoint (validated on a random-init `RefCV3Model` at `refc_v3_smoke_config` over a synthetic
3-episode slice; refcv3 is still training and this box must not contact `tanitad-refcv3`).

> ⛔⛔ **§2 IS THE DELIVERABLE THE MASTER MIND REVIEWS BEFORE ANY NUMBER IS QUOTED.** The
> previous attempt at this adapter was killed mid-task and never wrote it; the draft it left
> behind asserted the opposite of what the source says (it called the arm `cl` and described it
> as a "mirror of `t1_eval.roll_closed` … the loop collapses to the single tick"). There is no
> loop to collapse. Read §2 before reading any table this tool prints.

> ⛔ **PROVENANCE OF THIS FILE.** Every claim below carries `file:line` against the repo at
> `agent/arch-inf-20260803`. Where a prior document's line number disagrees, this file states
> both — `T1_CHECKLIST.md` cites `physicalai.py:620` and `t1_eval.py:753`; MEASURED here they
> are **`physicalai.py:621`** and **`t1_eval.py:760`** (same statements, one/seven lines of
> drift). Nothing else in either citation changed.

---

## 1. The contract this adapter targets

### 1.1 The dump — `t1_eval.py --analyze-only` must stay valid on it

Two files per episode, written by `run_dump()`, so the banked dump can be re-analysed with zero
GPU (`CLAUDE.md`: an analysis-time import failure after a completed rollout has already destroyed
a paid-for 2-arm/40-episode rollout once):

```
<dump>/ep{fi:03d}.npz            the t1_eval contract, UNCHANGED
    g   [N, K, 2]     GT ego-frame waypoints at the grid instants
    os  [N, K, 2]     the arm under test         (T1-candidate, UNRULED — §2.5)
    ha0 [N, K, 2]     constant velocity at the measured v0        (T1)
    ha  [N, K, 2]     hold the last OBSERVED (a, steer)           (T1)
    os_navshuf [N, K, 2]  os with nav_cmd PERMUTED across windows (T1)
    os_navzero [N, K, 2]  os with nav WITHHELD (nav_cmd=None)     (T1)  ⭐ §2.8
    [oracle_sel [N, K, 2]]  the a_star-selected anchor, opt-in    (T0)
    ws  [N]           window origin, in PROVIDER frame index t0
    eid [1]  clip_index [1]  v0 [N]
<dump>/decisions/ep{fi:03d}.npz  the refcv3 sidecar (this tool's own analysis)
<dump>/manifest.json             model provenance, grid, tiers, T1 definition,
                                 action-units, nav policy, label join, episodes
```

⛔ **THE DUMP'S KEY SPACE IS THE ARM SPACE** (`t1_eval.py:170`, `_META_KEYS = ("eid",
"clip_index", "v0")`). Anything else in the npz is enumerated as a prediction arm and
`resolve_tiers` (`t1_eval.py:308`) raises on it if it carries no `T0`/`T1` stamp. `v0` is a
per-window **covariate**, not an arm.

### 1.2 The four binding families

`t1_eval.analyze` (`t1_eval.py:324`) is **imported, never copied**, and it computes
`four_families.all_families` (`four_families.py:1675`) per arm on the same windows. Beside it
this adapter emits its own per-window component table (the `refav1_arm._components` geometry,
`four_families._seq_geometry` at `four_families.py:149` and `maneuver_kinematics` at `:994`), so
that a **paired** interval exists per family:

| family | what is reported | source |
|---|---|---|
| **ADE / FDE** | L2 norm of (pred − GT) over the grid, mean over slots | recomputed from the dumped path — ⛔ **never converted** from refcv3's training-time `traj`, which is a mean **L1 per coordinate** (`refc_v3_train.py:465`) |
| **LONGITUDINAL** | target-speed MAE, along-track MAE, accel MAE **and** distance-keeping (headway / time-gap / min-TTC), per speed band | `four_families` + `lead_metrics.distance_keeping` (`lead_metrics.py:125`) over the **common-grid view** of the banked B1 lead block (§1.4) |
| **LATERAL** | cross-track, heading, **yaw-rate** MAE (and curvature through the same geometry) | `four_families._seq_geometry`; ⛔ every curvature/yaw number is downstream of the unit contract (§1.5) |
| **TACTICAL** | trajectory-derived lat/lon manoeuvre agreement (`refc_tactical.factor_from_kinematics`) **and** the DECLARED heads vs the v7.2 labels, **and** `anchor_acc` (chance **1/128 = 0.0078**, `refc_v3.py:196`) plus the factored-head CE against chance `ln 8 = 2.0794` | this tool's sidecar |
| **STRATEGIC** | the route head vs the v2.1 route label, under **three** nav conditionings (true / shuffled / zero), with the echo index | this tool's sidecar; ⛔ **inadmissible without the `navshuf` control beside it** |

`t1_eval.fam_row` (`t1_eval.py:263`) is the §1.12 reproduction row (`ade_m`, `speed_bias_mps`,
`speed_mae_mps`, `accel_rms_mps2`, `jerk_rms_mps3`, `net_yaw_err_rad`) — a **mean-of-episode-means**
by construction, emitted by `analyze` for continuity only. ⛔ **The headline rows are the
FULL-SET pooled means**, never `fam_row`'s episode-mean path, and never `overlapping_holdout_se`
(which biases the point estimate, not merely the interval).

### 1.3 The strategic block, and why it needs the shuffle

Nav is an **input** to refcv3 (`refc_v3.py:480`, `nav_cmd`) and the route label derives from the
same clip. A route head that reproduces its own input scores well and has learned nothing —
flagship v1's scored **1.0000** on a bijection of the fed nav. The block therefore reports, per
conditioning, the agreement block (`four_families._agreement_block`, `four_families.py:1052`),
the **echo index** (`route_pred == the route the FED token maps to`), the majority-class rate, the
paired true-minus-shuffled accuracy, and the **changed subset** — the windows whose token actually
moved, which is where the control has power at all.

⛔ **THREE conditionings, not two, and they are not interchangeable — see §2.8.** `nav_true`
withholds nothing, `nav_shuffled` withholds the **pairing**, `nav_zero` withholds the **signal**.
BACKLOG R39 binds both controls to every nav-conditioned claim, because nav here is an **oracle**
input that will not exist at deployment.

### 1.4 The lead-block join (LONGITUDINAL, second half)

The banked B1 EVAL block is one row per `(clip_id, RAW 10 Hz frame)` for the 147 v7.2 EVAL clips,
on a **0.2 s / K = 10** horizon grid. refcv3's grid is **not** a subset of it (§2.3), so the join
is by **index-select on BOTH sides onto the instants they share** — for `--grid 2s` that is
`{1.0, 2.0} s` — with **no resampling of the lead track and no interpolation of the path**. The
join key is `(clip_id from the dump manifest, RAW frame)`; the guards are
`refav1_arm.join_lead_block`'s own: the exact-grid assertion, the **label-free speed proof**
(`dump v0 == block speed at the joined row`, tol 1e-3 m/s — both are egomotion interpolated at
the same instant, so a mismatch is a wrong clip or a wrong frame, not noise), and **NO_LABEL is
counted and never scored as free flow**.

⚠️ **A min-over-2-instants is coarser than the block's own 10-step min**: a lead that is closest
*between* 1.0 s and 2.0 s is not seen. The record states the instants it scored, and the rebuild
that removes the limitation is a **WORK ITEM**, not a caveat:
`taniteval/tools/build_lead_block_b1.py --dt 0.5 --k 4`.

### 1.5 The action-unit contract (`--action-units`) — and the defect it removes

⛔ **`actions[:, 0]` IS A ROAD-WHEEL STEER ANGLE, NOT A CURVATURE.**
`stack/tanitad/data/physicalai.py:621` computes `steer = np.arctan(float(wheelbase) * curv)` and
`:632` writes `actions = np.column_stack([steer, accel])`. The `refav1_loader` docstring calling
channel 0 *"the MEASURED true-kappa channel"* (`refav1_loader.py:23`) is **wrong**, and the
rescued refcv3 draft inherited that sentence verbatim into its `recorded_controls` (≈ line 477 of
`…/incoming/2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py`) — register
`C-REFCV3-ARM-SAME-DEFECT`. On refav1's eval slice the identical defect cost **0.716 m** of
curved-window lateral error against a **0.053 m** pose-yaw floor.

The bridge is `stack/tanitad/models/kinematic.py`: `STEER_WHEELBASE_M = 2.9` (`:51`),
`kappa_of_steer` (`:57`), `steer_of_kappa` (`:62`), `as_curvature` (`:74`), `as_command` (`:86`).
`--action-units` names the unit **channel 1 arrives in**:

* `steer` — **the correct reading for a RECORDED v2ep action**, and therefore for `ha`. The
  integrator is fed `κ = tan(steer) / L_enc` first.
* `kappa` — the LEGACY, unconverted reading. Kept as the default **only** so a re-analysis of a
  banked dump is byte-identical; ⛔ **the value is printed in the manifest**, because after the
  fact a repaired and an unrepaired dump are otherwise indistinguishable.

⭐ **This is a call-site decision, not a model property**, because a planner candidate (curvature)
and a recorded action (steer) reach the same integrator in different units. **refcv3 has no
planner candidate at all** — it emits a path directly — so on the `os` arm the conversion is
**structurally absent**, and it applies to `ha` only. `ha0` is exactly zero, which is 0 in either
unit; that is precisely what makes `ha0` **bit-comparable across refav1 and refcv3** and `ha` not.

### 1.6 The estimator

* **Point estimates are FULL-SET pooled means over windows.**
* **Intervals: the episode-cluster bootstrap** — `taniteval.ci.episode_cluster_bootstrap`
  (`ci.py:225`); episodes are resampled with replacement because windows inside one clip are
  strongly dependent and the **episode** is the independent unit.
* **Two arms on the same windows ⇒ the PAIRED form** —
  `taniteval.ci.paired_episode_cluster_bootstrap` (`ci.py:275`). ⛔ Never two single-arm intervals
  combined in quadrature.
* Per-window `eid` is in the dump by construction; without it there is no cluster to resample,
  which is the exact wall the in-training `metrics.jsonl` hits.

**The headline paired blocks are `os − ha0` and `os − ha`**, per family. For the H-vs-F claim the
statistic is the **difference of each arm's margin over the same floor** — `(cl − ha0)_refav1` vs
`(os − ha0)_refcv3` — ⛔ **never `cl` against `os` as levels** (`D-HF-COMPARABILITY`).

### 1.7 Tier stamps

`t1_eval.DEFAULT_TIERS` (`t1_eval.py:145–156`) **already carries `"ha0": "T1"` at `:154`** — I
verified it with two differently-bound probes (a `sed` read through the Bash tool and
`Select-String -LiteralPath` through PowerShell), so **BACKLOG R13 has landed** and this tool did
not need to work around it. It works either way: `--tiers` is passed straight through to
`t1_eval._parse_tiers` and merged over the defaults, and this tool declares its own
`ARM_TIERS` for every arm it writes, so `resolve_tiers` never sees an unstamped key even against
an older `t1_eval.py`. ⛔ `t1_eval.py` was **not** edited.

`os`, `os_navshuf`, `os_navzero`, `ha`, `ha0` are stamped **T1**; `oracle_sel` is stamped **T0**.
⚠️ The `os*` stamps additionally carry `status = "UNRULED"` in the record — see §2.5.

⚠️ **A new arm needs its stamp on the command line too when the dump is re-read through the
untouched `t1_eval.py --analyze-only` CLI.** MEASURED: omitting `os_navzero=T1` there exits **1**
with *"arms ['os_navzero'] carry no T0/T1 tier stamp"* — the guard working exactly as designed.
Use `--tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0`.

### 1.8 REFUSED and ABSENT — the two states, and the reasons they carry

A family or a metric that cannot be computed is **never silently dropped**. Two distinct shapes:

| state | meaning | shape |
|---|---|---|
| **REFUSED / UNAVAILABLE** | the metric is defined for this model but its **inputs** are missing here | `{"status": "REFUSED"/"UNAVAILABLE", "reason": <why>, "n": <count>, "tier": <stamp>, "estimator": "n/a — inputs missing (WORK ITEM, not a pass)"}` |
| **ABSENT** | the arm **does not exist for this model at all** | `{"status": "ABSENT", "arm": "ol", "reason": <the structural reason>, "for_comparison": <what to use instead>}` |

The one **ABSENT** entry this tool always writes is **`ol`**. It is not missing, not skipped and
not a work item: refcv3 **consumes no recorded actions**, so "the recorded future actions
integrated from v0" is not a rollout *of this model* — it is a property of the corpus. The record
says so, names `ha0` as the shared floor instead, and points at `refav1_arm.py` for the arm that
does exist there. ⛔ Never emit an `ol` column for refcv3, empty or otherwise.

---

## 2. ⛔ WHAT refcv3's ARM **IS** — the definition, from source

### 2.1 What the model outputs, per window, on what grid

`RefCV3Model.forward` (`stack/tanitad/refs/refc_v3.py:480`):

```python
def forward(self, frames, nav_cmd=None, v0=None, steps=0, lan=None, nav_known=None) -> dict
```

**There is no action argument.** The whole signature is: pixels, a route token, one scalar speed,
a diffusion-step count, an optional lane-graph feature, an optional nav-known mask.

* The decoder emits a **fan of 128 anchor trajectories** — `AnchorConfig(n_anchors=128,
  pool_size=4096)` (`refc_v3.py:196`) — each an **8-slot path**,
  `TrajectoryConfig(horizons=V3_HORIZONS)` (`:197`) with
  `V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)` (`:106`) in 10 Hz steps, i.e. **0.5, 1.0, 1.5,
  2.0, 3.0, 4.0, 5.0, 6.0 s**. In the model's output that fan is `out["anchor_traj"]` `[B, N, S, 2]`
  (`refc.py:1533`, re-exported at `refc.py:2170`).
* **One** of those 128 is selected, and `out["traj"]` `[B, S, 2]` is it.
* ⇒ **one forward pass produces the entire 6 s path.** There is no per-step decode, no
  autoregression, no state that advances.

### 2.2 What the model consumes at inference, and why none of it is future information

| input | source at eval | admissible? |
|---|---|---|
| `frames` | the observed window only — `[t, t+w)`, `pose_last = poses[t + w - 1]` (`refb_train.py:229`) | ✅ vision, by construction |
| `nav_cmd` | the clip's v7.2 `nav_command` token (`V3Dataset.enable_nav_from_v7`, `refc_v3_train.py:246–306`) | ✅ a goal/route input — PI 2026-08-03. ⚠️ It is an **ORACLE** (provenance `ego-future`) and will not exist at deployment, which is why §2.8's nav-zero arm exists. |
| `v0` | `v0 = pose_last[:, 3]` (`refc_v3_train.py:445`) — the speed **measured at the last observed frame** | ✅ PI ruling 2026-09-02: *"velocity as initial measured state at its cycle time"* |
| `steps` | the decoder's diffusion-step count | ✅ a hyper-parameter |
| `lan` | not fed by this adapter | — |

Nothing after the window origin reaches the forward pass. The **only** place a future frame or a
future pose is read is `compute_losses_v3` (`refc_v3_train.py:428`ff), and only to build
**targets**: `fut_ext = batch["future_poses_ext"]`, `traj_tgt = refb_labels.waypoint_targets(
pose_last, fut_ext, core.trajectory.horizons)` (`:452–453`).

### 2.3 The grid the dump uses — index-select, never interpolation

`t1_eval`'s dump contract is a **uniform** `[N, K, 2]` grid; the model's slots are not uniform
(0.5 s stride to 2 s, then 1 s). The dump is therefore an **INDEX-SELECT of the model's own
slots**:

| `--grid` | dt | K | model slots (10 Hz steps) | slot indices | dropped |
|---|---|---|---|---|---|
| `2s` (default) | 0.5 s | 4 | 5, 10, 15, 20 | 0, 1, 2, 3 | 30, 40, 50, 60 |
| `6s` | 1.0 s | 6 | 10, 20, 30, 40, 50, 60 | 1, 3, 4, 5, 6, 7 | 5, 15 |

⛔ A grid the model cannot serve by index-select is **refused**. There is no resampling.
GT is read at exactly those instants through the trainer's own target function
(`refb_labels.waypoint_targets`), so the GT the arm is scored against is the GT it was trained
against.

`ha` / `ha0` are integrated at the corpus's native **0.1 s** tick and then index-selected onto the
same instants, so the controls and the model share one grid and one GT array.

### 2.4 ⛔ WHY `t1_eval.roll_closed` CANNOT BE PORTED, AND WHY THE ARM IS NOT CALLED `cl`

`t1_eval.roll_closed` (`t1_eval.py:760`) closes the loop like this: at each of K steps the
predictor emits `z_hat`; a `UnicycleStepReadout` **head** decodes `(a_j, yaw_j)`; the path
accumulates from those; and the action fed to the **next** predictor step is
`(steer = atan(L·κ), a_j)`, with perception frozen at t0.

That works for the flagship — **which is also supervised** — because the flagship is *additionally*
an **autoregressive, action-conditioned predictor with a per-step readout**. Being supervised was
never what made `cl` possible; being autoregressive was.

refcv3 has **none** of that machinery: no action input (§2.1), no per-step decode, no state to
advance. **There is no action to feed back, so there is no loop to close.** Porting `roll_closed`
is not a small engineering job — there is no object to port it onto.

⇒ **The arm is named `os` (one-shot). NEVER `cl`.** A shared column name is exactly how two
different procedures end up in one table and get read as levels of one quantity. Consequences the
register `D-HF-COMPARABILITY` binds and this tool implements:

1. **`ha0` is the only bit-comparable arm** between refav1 and refcv3 — identically defined
   (`a = 0, κ = 0` at the measured `v0`), identical windows, identical integrator.
2. **`ol` does not exist for refcv3** (§1.8) — ABSENT with its reason, never a silent drop.
3. **The admissible claim is each arm's margin over the same `ha0` floor, per family, paired** —
   ⛔ never `os` vs `cl` as levels.

### 2.5 ⚠️ The tier is UNRULED, and the tool says so on every number

`EVAL_DOCTRINE.md`'s T1 row reads *"the predictor consumes the decoder/planner's own actions"*.
That sentence does not literally cover a model that consumes **no** actions. Whether the doctrine
admits an action-free model at T1 is an **OPEN PI / Master-Mind ruling** (BACKLOG R30, RESULT §5.4
work item W6) — **not something a FlyWheel may assume in either direction**.

So the instrument is built to be right either way: `os` is stamped `T1` in `ARM_TIERS` (so
`resolve_tiers` accepts the dump and every number is produced), and the record carries
`_tier_ruling = {"arm": "os", "stamped": "T1", "status": "UNRULED", ...}` naming the open decision,
the doctrine sentence it does not literally satisfy, and the recommendation. **Benchmarks'
recommendation, flagged as a recommendation:** admit it — the doctrine's *purpose* is to keep
future information out of inference, and a correctly-gated refcv3 forward pass (§2.2) admits
none — but keep the distinct arm name.

⭐ **The margin framing is what makes this survivable either way.** `os − ha0` is a difference of
two arms measured on the same windows with the same instrument; it stays meaningful whatever tier
label the ruling attaches, and it is the only statistic the H-vs-F comparison may use.

### 2.6 ⛔ THE SELECTION GATE — `sel_score_v3`, and never `a_star`

This is the second way a refcv3 number goes wrong, and it is subtler than the arm name.

**What the deployed arm must use.** On the hierarchical arm (`hier=True`) the model ranks its own
fan and picks:

```python
graft   = self.goal_gate * sc["score"]                 # refc_v3.py:509
blended = apply_seam_clamp(out["sel_score"], graft, ...)# refc_v3.py:512
rank    = blended.masked_fill(~out["reach_keep"], -inf) # refc_v3.py:518-519
idx     = rank.argmax(dim=1)                            # refc_v3.py:520
traj    = fan[arange(b), idx]                           # refc_v3.py:521
out["traj"], out["sel_idx"] = traj, idx                 # refc_v3.py:525
out["sel_score_v3"] = blended                           # refc_v3.py:527
```

On the flat arm (`hier=False`) `forward` delegates straight to the core (`refc_v3.py:484–486`) and
the core does the same thing with its own score: `traj = x[arange(b), idx]` (`refc.py:1531`),
`out = {..., "sel_score": score, "traj": traj, "sel_idx": idx}` (`refc.py:1532–1534`).

⇒ **`out["traj"]` IS the deployed selection on both arms.** This adapter takes `out["traj"]` and
nothing else, and banks `sel_idx` / `sel_score_v3` in the sidecar so the choice is auditable.

**What it must NEVER use.** The training loss does *not* score `out["traj"]`:

```python
dist   = (((traj_tgt[:, None] - anchors[None]) ** 2).sum(-1) * sv[:, None]).sum(-1)
a_star = dist.argmin(dim=1)                             # refc_v3_train.py:460
recon  = out["anchor_traj"][ar, a_star]                 # refc_v3_train.py:463
loss_traj = (((recon - traj_tgt).abs().sum(-1)) * sv).sum() / denom   # :465
```

`a_star` is **the anchor nearest the GROUND TRUTH**. `eval_traj` in `metrics.jsonl` is therefore an
**oracle-anchor-selected** error and a **loose lower bound** on what refcv3 would drive — the
model's own selection agrees with the oracle on ~57 % of windows in the clean era
(`RESULT.md` §1). It is also a **mean L1 per coordinate** (`:464–465`, `denom = sv.sum() * 2`),
not an L2 ADE. ⛔ **Never put it in a column with anyone's ADE, and never convert one into the
other** — this tool recomputes ADE from the dumped path.

The oracle path is available as the **opt-in `oracle_sel` arm, stamped T0**, reported *beside* the
deployed arm as the ceiling it is (how much of `traj` was selection), and ⛔ **never compared to a
T1 number**.

### 2.7 The one-line definition, for the report

> **refcv3's `os` arm is: one forward pass of `RefCV3Model` at the window origin, consuming the
> observed frames, the clip's v7.2 nav token and the measured `v0` at t0 and nothing else,
> emitting the whole 6 s path as the model's OWN `sel_score_v3`-ranked choice among its 128
> anchors; scored on the index-selected grid against the trainer's own waypoint targets. It is a
> one-shot planning-free trajectory prediction — there is no action loop to close, its tier is an
> open ruling, and it may only be compared to refav1 through each arm's margin over the shared
> `ha0` floor. **Because its nav token is an ORACLE that will not exist at deployment, the arm is
> reported beside `os_navzero` — the same forward with nav withheld — and `os_navzero − ha0` is the
> deployment-relevant margin.****

### 2.8 ⛔ THE THREE NAV CONDITIONS — what each one removes, and why two controls

Nav reaches refcv3 at **three** places: the core measurement encoder (`refc.py:2021–2024`) and,
through E13, the **tactical** and **strategic** states (`refc_v3.py:437–441`). The arm therefore
needs two different nulls, and they answer different questions:

| condition | what is withheld | what it answers | arm |
|---|---|---|---|
| `nav_true` | nothing — the clip's real v7.2 token | the deployed reading | `os` |
| `nav_shuffled` | the **pairing** between the window and its token. The nav **marginal is preserved exactly** (it is a permutation), so the model still sees a plausible token everywhere. | *is the model using **this** window's nav?* | `os_navshuf` |
| `nav_zero` | the **signal** | *what does the model do when the oracle nav is **not there*** — i.e. **deployment** | `os_navzero` |

⛔ **They are not interchangeable, and a shuffle cannot stand in for a zero.** MEASURED elsewhere
(`D-REFAV1-TAC-DECODER-PANEL`): refav1's tactical head ranked turns at **AUC 0.873** under true nav
and **collapsed to 0.520 — chance — under nav_zero**, while a **nav-only predictor beat the model
outright (0.684 > 0.650)**. `BACKLOG R39` binds every nav-conditioned claim to carry the nav-zero
arm beside the nav-shuffle one.

**The null, derived from source — and it is NOT `nav_known`.** The obvious candidate is
`RefCV3Model.forward`'s `nav_known` argument. MEASURED: **it is not usable here.**
`RefCConfig.nav_known_channel` defaults to **`False`** (`refc.py:594`), nothing in the v3 path turns
it on (**0** references in `refc_v3.py`, **0** in `refc_v3_train._pin_trainer_cfg`), and
`refc.py:2042–2045` **raises** if `nav_known` is supplied while the gate is off — *"it would be
silently dropped. Turn the gate on or stop passing it."* So passing it would be **refused**, not
principled.

⇒ the null is **`nav_cmd=None`**, which is the model's **own** documented no-nav path and the
published REF-C eval convention (`refc.py:343–345`: *"every published REF-C number decodes with
`nav_cmd=None` -> index 0"*). Nothing is invented. What it removes, per layer:

| layer | under `nav_cmd=None` |
|---|---|
| **tactical** (E13 `PhiTac`) | ⭐ **REMOVED ENTIRELY** — `refc_v3.py:437–441` guards the injection on `nav_cmd is not None`, so `nav_to_tac` is never added |
| **strategic** (E13 `ctx`) | ⭐ **REMOVED ENTIRELY** — same guard; `out["nav_injected"]` reads **False** (`refc_v3.py:471`), and the tool banks it per window |
| **core** (measurement encoder) | ⚠️ **NOT removed — COLLAPSED onto the majority token.** `refc.py:2021–2024` substitutes `one_hot(0)` = `follow`. `NAV_COMMANDS` (`refc.py:136`) has no `unknown` entry and `nav_known_channel` is off, so **"no nav" and "a genuine follow" are byte-identical at the core's input** — the exact defect `nav_known_channel` was written to fix (`refc.py:594–604`: **62.4 %** of `follow` windows are a collapsed UNKNOWN). |

⛔ **Therefore `os_navzero` is a LOWER BOUND on nav dependence**, and the record says so: a model
that read nav only through the core would look *less* nav-dependent here than it is. State that
beside any nav-dependence number taken from this arm.

⭐ **The mechanism is measured, not asserted.** `stack/tests/test_refcv3_arm.py` pins it two ways:
on the **hier** build the E13 edge is **live under the fed nav (`nav_injected_true == 1` on every
window) and dead under the null (`nav_injected_zero == 0`)**, so `os_navzero` differs from `os`
*even on `follow` windows* — which the shuffle provably cannot do; and on a **flat** build, where no
E13 path exists, `nav_cmd=None` is **bit-identical** to a fed `follow` token at matched batch size
(`torch.equal`, max |Δ| **exactly 0.0**) while a fed `left` moves the path by **> 1e-3 m**.

⚠️ **One instrument caveat that follows from this, and it is a trap.** `os_navzero` is produced by a
**separate forward call** (`nav_cmd=None` is a whole-call property), while `os` and `os_navshuf` are
two **rows of one batched call**. Different batch sizes take different GEMM kernel paths, so a
float32 floor sits under any cross-call comparison: **MEASURED 5.96e-07 m** (batch-2 row 0 vs
batch-1, same nav) against **0.0 exactly** at matched batch size, while a real nav difference is
**2.78e-02 – 3.41e-02 m — ~4.7 × 10⁴ times the floor**. ⇒ `trivial_profile.identical_to` uses a
**1e-9 m** threshold and **cannot** resolve a cross-call arm as identical even when it is. The
record names the cross-call arms and carries this measurement; read `identical_to` for
`os`-vs-`os_navshuf` (same call, exact) and **ignore it** for `os`-vs-`os_navzero`.
---

## 3. The real read, AFTER the epoch ends

⛔ **GATE 0 FIRST.** `T1_CHECKLIST.md` in
`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-epoch-conclusions/` is the
runbook: `summary.json` reads `{"done": true, ...}`, **the supervisor has exited**
(`ps -eo args | grep -c 'sup[_]refcv3'` must read **0** — a supervisor whose run never wrote a
done-marker resurrects it and overwrites `ckpt.pt`), and the rolling `ckpt.pt` has been copied to
an **immutable name and md5'd**. Nothing below may be run against the rolling file.

### 3.1 The two-step invocation (the second step is the expensive one)

**Step 1 — price it on two clips, then choose the stride.** The tool prints a `[cost]` line after
its first forward; that number, not an assumption, sets the stride.

```bash
OMP_NUM_THREADS=6 python taniteval/tools/refcv3_arm.py \
  --ckpt   /workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt \
  --episodes  <B1 EVAL v2 cache dir>            \
  --labels    <.../v72/s2_labels_v7.2_eval.jsonl.gz>  \
  --nav-source v72 --grid 2s --action-units steer \
  --episodes-n 2 --with-oracle-sel \
  --dump-dir /workspace/eval/refcv3_t1_probe --out /workspace/eval/refcv3_probe.json \
  --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0
```

**Step 2 — the real read**, with the stride chosen from step 1's `[cost]` line and the banked lead
block attached:

```bash
OMP_NUM_THREADS=6 python taniteval/tools/refcv3_arm.py \
  --ckpt   /workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt \
  --episodes  <B1 EVAL v2 cache dir>            \
  --labels    <.../v72/s2_labels_v7.2_eval.jsonl.gz>  \
  --nav-source v72 --grid 2s --action-units steer \
  --window-stride <from step 1> --with-oracle-sel \
  --lead-block "TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz" \
  --dump-dir /workspace/eval/refcv3_t1_dump --out /workspace/eval/refcv3_t1.json \
  --n-boot 2000 --seed 0 --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0
```

* ⭐ **Both nav controls are on by default** (`os_navshuf` and `os_navzero`); `--no-navshuf` /
  `--no-navzero` turn them off and the tool warns that the record is then not admissible for a
  nav-conditioned claim. The nav-zero arm costs **one extra forward per window** (it cannot share
  the batch — §2.8), so budget for it in step 1's `[cost]` line.
* `--tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0` is **belt and braces** here, not a
  requirement: this tool
  declares its own `ARM_TIERS` (§1.7) and `t1_eval.DEFAULT_TIERS` already carries `"ha0": "T1"`
  (`t1_eval.py:154`, BACKLOG R13 landed). Pass it anyway — it makes the tier stamps visible on the
  command line, and it is what the **untouched** `t1_eval.py --analyze-only` CLI needs if the dump
  is re-read through that path (MEASURED: it does, exit 0, §4).
* ⛔ **`--analyze-only <dump-dir>` before re-running anything.** The rollout is the only expensive
  part; an analysis-time failure is recoverable with zero GPU.
* ⛔ **Never run this on the training pod.** GATE 1.

### 3.2 The split, and the assertion that must precede it

* **147 records in the v7.2 EVAL blob; 141 carry pixels in the B1 epcache** (the other 6 are the
  deployed-val40 clips the parity gate drops, `D-REFCV3-EVAL-LEAK`). **Score the 141** — i.e. point
  `--episodes` at the 141-clip cache, and check the manifest's `corpus.labels.n_joined` /
  `n_missing` afterwards. Labels md5 `aa12c948f062181c3297265b51526ec5`; nav distribution
  `follow 90 / left 13 / right 38, missing 0`.
* ⛔ **Assert train ∩ eval = ∅ by episode id before scoring.** The armed chain once pointed
  `--v2-cache` at all 4,713 clips, which would have trained on 141 of the 147 eval clips. This tool
  does not and cannot check that — it is a property of the *run*, and it is GATE 3.
* ⛔ Do **not** reuse the in-training eval's 160 windows: a `manual_seed(12345)` subset chosen as a
  loss monitor, carrying no `eid`, far too small for an episode-cluster bootstrap.

### 3.3 ⏱ Time — how to get the estimate, and the band I can honestly give

**MEASURED here:** nothing about the real model. This box may not touch the training pod, and the
only checkpoint I could load is a random-init smoke model on CPU.

**ESTIMATED, and stated as such.** The window count is
`n_clips × (T_out − window − max_horizon)` with `T_out = T_raw − (n_stack − 1)`, `window = 8`
(`RefCConfig.window`) and `max_horizon = 20`. At the INHERITED B1 figure of ~199 raw frames per
clip that is ~169 windows/clip, so **~23,800 windows at stride 1 over 141 clips**. The forward is
one batched pass of 2 rows (`os` + `os_navshuf`) through a ~63 M-parameter ResNet trunk per
window. ⇒ **run step 1 and read the `[cost]` line**; if it lands anywhere above ~0.15 s/window a
stride of 5 (~4,760 windows) is the right trade, and the manifest records the stride so the grid is
reproducible. ⛔ Do not quote a wall-clock from this file — quote the `[cost]` line from the run.

### 3.4 What the read must print, in this order

1. the `[model]` / `[grid]` / `[units]` provenance lines,
2. ⭐ the **trivial profile** — per arm `straight_frac`, `const_speed_frac`, `CONSTANT-VELOCITY`,
   `identical_to`,
3. ⭐ the **selection profile** — `n_distinct_selected`, the modal anchor and its share, the
   selection entropy, and how often the deployed selection coincides with the oracle,
4. only then the family rows and the paired margins — **including `os_navzero − ha0`, the
   deployment margin, beside `os − ha0`**.

⛔ **If either profile is degenerate the read is VOID, not negative.** Void means the instrument saw
the baseline (or one constant anchor), not the model.

---

## 4. MEASURED on the synthetic fixture — and what stays UNVERIFIED

**What ran (2026-09-03, dev box, CPU, 0 GPU):** a random-init `RefCV3Model` at
`refc_v3_smoke_config(hier=True)` with the encoder widened to the corpus's 9 channels at 64 px,
over a synthetic 3-clip v2 cache (40 raw frames/clip, JPEG, `n_stack = 3`) with a 3-record v7.2
label blob. **42 windows, 3 episodes, 6 arms** (`os`, `os_navshuf`, `os_navzero`, `ha`, `ha0`, `oracle_sel`),
plus a second **flat** (`hier=False`) build for the nav-mechanism control.
`stack/tests/test_refcv3_arm.py`: **20 passed**.

⛔ **These are INSTRUMENT numbers on a random-init model. They are not evidence about refcv3.**
They are here because a control that reads a **known value** is the only thing that shows the
instrument works:

| what | reads | the known value it had to read |
|---|---|---|
| `ha0` straight / const-speed | **1.0000 / 1.0000** on 42/42 | a zero-control unicycle rollout is a straight line at constant speed, by construction |
| `ha0` chord length per step | `v0 × 0.5 m` to < 1e-3 | the integrator advances on `v0` and never changes it |
| `os` trivial fraction | **0.0000** | a random-init anchor model bends; it is *not* the CV plan |
| `os` identical to `os_navshuf` | **17/42** | exactly the 42 − 25 windows whose nav token the permutation did **not** change |
| `os − ha0` (ADE, paired) | **+7.96 m [6.15, 9.76], separated** | a random-init model must **lose** to the constant-velocity floor |
| `os − os_navshuf` (ADE, paired) | **−0.0002 m [−0.0024, 0.0016], not separated** | untrained weights carry no nav dependence |
| `os − os_navzero` (ADE, paired) | **−0.0001 m [−0.0021, 0.0017], not separated** | same — and the two controls agree, as they must on an untrained model |
| `os_navzero − ha0` (ADE, paired) | **+7.9572 m [6.1513, 9.7646], separated** | the deployment margin must also lose to the floor at random init |
| E13 edge, per window | `nav_injected_true` **1.0** on 42/42; `nav_injected_zero` **0.0** on 42/42 | ⭐ the per-layer claim of §2.8, MEASURED rather than asserted |
| flat build, matched batch | `nav_cmd=None` vs fed `follow`: **exactly 0.0** (`torch.equal`); vs fed `left`: **> 1e-3 m** | with no E13 path the null IS the `follow` token, bit for bit |
| cross-call float32 floor | **5.96e-07 m** (batch-2 row 0 vs batch-1, same nav) vs a real nav difference **2.78e-02 – 3.41e-02 m** = **~4.7 × 10⁴ ×** | the separation that makes a nav-zero reading trustworthy — and the reason `identical_to` at 1e-9 m must not be read across calls |
| `ha − ha0` (ADE, paired) | **+0.133 m [−0.050, 0.233], not separated** | ⭐ **the whole reason `ha0` exists**: holding a noisy observed steer is *not better* than doing nothing, so a win over `ha` alone is not skill |
| `anchor_acc` | **0.0000** (chance 1/20 = 0.05) | untrained logits do not find the GT-nearest anchor |
| selection profile | `n_distinct = 1`, modal #2 at **100 %**, entropy **0.0** | ⛔ and the trivial profile read **0.0000** on the same arm — see below |
| a future-only perturbation | `os`, `ha`, `ha0` move by **exactly 0.0**; `g` moves > 1 m | nothing recorded after t0 reaches the arm, and the perturbation provably landed |
| the untouched `t1_eval.py --analyze-only` CLI | **exit 0** on the same dump | the dump really is the `t1_eval` contract |

⭐ **THE FIXTURE PRODUCED A REAL FINDING ABOUT THE INSTRUMENT.** The random-init model selected
**one anchor on 42/42 windows** — a completely degenerate arm — while its `trivial_frac` read
**0.0000**, because the constant anchor is neither straight nor constant-speed. **The trivial
profile, the instrument this whole package is built around, is BLIND to refcv3's characteristic
degeneracy.** That is why `refcv3_arm.py` adds a second gate, the **selection profile**, printed
beside it and before any family row. Without it a family table computed over "always anchor #2,
refined" would have been read as scene understanding.

### 4.1 UNVERIFIED until a real checkpoint is read

* **Everything about refcv3 itself.** No real checkpoint has been loaded here; every number above is
  a random-init instrument control.
* **`rebuild_config` via `argv`.** The fixture uses the `refcv3_arm_model_cfg` escape hatch. The
  `config.json[argv] -> build_parser + _pin_trainer_cfg` path is written and reviewed but has
  **never been exercised on a real run's `config.json`** — run step 1 of §3.1 first, and treat a
  `SystemExit` there as trainer/checkpoint version skew, not as a model problem.
* **The lead-block join at real scale.** The common-grid index-select and the `(clip_id, RAW frame)`
  key are implemented and reviewed; the **label-free speed proof** is the guard that would catch a
  wrong frame offset (it refuses the episode with `SPEED_MISMATCH` rather than scoring another
  clip's traffic). ⚠️ It has **not** been exercised against the banked block, and a known limitation
  is recorded in the record: `join_lead_block` reaches only EVEN raw frames (it maps `t → 2t`), so
  window origins on an odd raw frame join to `frame − 1`. The count is printed
  (`_odd_raw_frames`) and the fix — a frame-identity mode on `join_lead_block` — is a **WORK ITEM**.
* **The `6s` grid.** Its slot arithmetic is pinned by test; no dump has been produced on it (the
  fixture's clips are too short to carry 60 frames of future).
* **Wall-clock.** §3.3.

---

## 5. What the Master Mind must decide

| # | decision | why it cannot be a FlyWheel's |
|---|---|---|
| **D1** | ⛔ **The tier ruling (BACKLOG R30 / RESULT §5.4 W6): does the doctrine admit as T1 a model that consumes NO actions?** | It is a change to `EVAL_DOCTRINE.md`'s meaning, not a measurement. The instrument stamps `T1` + `status: UNRULED` and carries the margin framing so the numbers are correct either way (§2.5). **Benchmarks recommends: admit it, keep the name `os`.** |
| **D2** | Whether `oracle_sel` (T0) is published beside the deployed arm in the H-vs-F table. | It is the ceiling and it flatters the model; it belongs in the record, and whether it belongs in the *headline* is an editorial call. ⛔ It may never be compared to a T1 number. |
| **D3** | The `--window-stride` for the real read, from step 1's `[cost]` line. | It is a compute-spend decision, and it fixes the grid both models must share. ⚠️ The nav-zero arm adds one forward per window (§2.8); price it in. |
| **D3b** | ⭐ Whether the **deployment margin** (`os_navzero − ha0`) or the **oracle-nav margin** (`os − ha0`) is the headline of the H-vs-F table. | Both are computed and both are admissible. Benchmarks' view, flagged as a view: the oracle-nav margin is the fair like-for-like against refav1 (which is also fed nav), and the deployment margin is the one that answers "does this drive"; quoting only the first overstates the system. |
| **D4** | Whether the **odd-raw-frame** limitation in the lead join (§4.1) is fixed before the read or accepted with its printed count. | It trades a code change against tonight's schedule. |
| **D5** | Whether to rebuild the lead block on refcv3's own grid (`build_lead_block_b1.py --dt 0.5 --k 4`) so the full 2 s horizon is scoreable instead of `{1.0, 2.0} s`. | Same trade; the record states the instants it scored either way. |

⛔ **And the refusal that stands regardless:** publish no H-vs-F number if any of
`D-HF-COMPARABILITY`'s six inadmissibility conditions holds — different grids, refcv3's
oracle-selected `traj` against refav1's `cl`, a cross-tier or `cl`-vs-`os` comparison as levels, a
degenerate trivial **or selection** profile, the unfixed steer defect, or no shared `ha0` floor.
