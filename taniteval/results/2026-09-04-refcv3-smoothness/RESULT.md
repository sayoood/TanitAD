# WP-5 / WP-6 — the refcv3 trajectory is rough, and *why* is not what the frame suggests

**2026-09-04 · Arch+Inference FlyWheel · ZERO GPU, zero checkpoint reads · evidence class MEASURED (ours)**
**Tier:** T1 **open loop** (self-action). refcv3's `forward` takes no action argument, so there is no loop
to close — `D-OPEN-VS-CLOSED`.

> **PI, verbatim:** *"the fan and trajectory hypotheses in the fan are not smooth and differentiable,
> I did not see this in refc-base"* · *"We extra introduced the learnings from the Alpamayo paper about
> output action space to assure feasibility of the trajectory… but the result is worse than the past!!!"*

## The answer in five sentences

1. **Part of what you see is the renderer, and the factor is 7.44×.** The frame draws GT from
   **60 dense 10 Hz poses** and the model from **8 slots joined by straight lines**. The *same
   physically smooth GT path* has a mean vertex kink of **0.3067°** at 10 Hz and **2.2820°** at
   refcv3's 8 slots. refcv3's own kink is **1.3074°** — **4.26× the dense GT polyline beside it,
   but 0.57× the same GT resampled onto the model's own grid.** On a like-for-like grid refcv3's
   path is *less* angular than ground truth.
2. **A real defect sits underneath it, and it is longitudinal + zig-zag, not lateral.** refcv3's
   **jerk is 5.83× GT** and its turn angle **changes sign 12× as often as GT's** — and sign-flip is
   **grid-invariant** (GT reads 0.000 dense, 0.017 at 8 slots), so the renderer cannot produce it.
3. **It IS worse than refc-base, paired, on the same 170 windows: jerk 2.80×, curvature-rate 1.44×,
   zig-zag 1.77×.** The PI's comparison is correct.
4. **The Alpamayo action space was never applied to the output.** `rollout_unicycle` and the A2S
   limits are imported by `refa_v1_plan.py` and by nothing in REF-C. refcv3 emits
   `x = anchors + offset` — **a free per-slot 2-D offset** — and its loss is an **L1 on waypoint
   positions with no jerk, curvature, smoothness or feasibility term at all.** Nothing constrains
   the emitted path to be drivable, and **41 of its 128 anchors break a dry-road Kamm circle**.
5. **The root cause is one line that is missing from the launch script.**
   `refcv3_b1_launch.sh` passes **no `--anchors`**, so refcv3 trained on the *synthetic bootstrap*
   vocabulary; refc-base/XL/small all trained on **data-driven** `refc_anchors_*.pt`. The synthetic
   vocabulary's **oracle-in-vocabulary ADE is 0.9433 m against the data-driven construction's
   0.4369 m (2.16×, worse on 91.7 % of 2,412 held-out windows) — and worse than a single
   constant-velocity straight line (0.6780 m).** A perfect selector over refcv3's fan cannot beat a
   straight line, so the free offset head has to override the vocabulary by **0.49 m** — and that
   offset is exactly the unconstrained mechanism that destroys smoothness. **The two defects are
   the same defect.**

---

## Substrates (all in-repo, all read at zero GPU)

| substrate | path | n |
|---|---|---|
| paired open-loop rollouts, 3 arms | `taniteval/results/2026-09-04-refcv3-closedloop/raw/rollouts_{refcv3,refc-base,flagship-v1}_openloop.json` | 190 windows, **1 clip** (`00040136-…`); refcv3 banks `extra.traj_full_6s` (8 slots), the other two bank `plan` (4 slots) |
| refcv3 @ 40,284 open-loop dump | `taniteval/results/refcv3-40284-openloop-dump.tar.gz` md5 `aff9bad5aff3869c261903d8ff1768e9` | 141 eps / 4,823 windows — **2 s grid only, the 6 s slots are dropped** |
| refc-base decoded fan | `taniteval/results/fan_refc-base-30k.pt` | 881 windows × **128 candidates** + gt + cv + v0 |

**Guard:** a window is admitted only where every segment of every compared path is
`>= MIN_DS_MPS(0.5) * dt` (`tanitad.models.kinematic.MIN_DS_MPS`) — below that the path tangent, and
therefore curvature, carries no information. 130/130 · 170/170 · 4,438/4,823 (92.0 %) · 828/881 (94.0 %).

**Definitions.** `turn` = the exterior angle at an emitted slot — **exactly the kink
`render_refcv3_video.densify()` draws**. `kappa` = turn / mean adjacent segment **arc length**, which
is *dt-invariant* and therefore the only curvature comparable across the 2 s seam. `sign-flip` =
fraction of adjacent interior-vertex pairs whose turn changes sign, deadband 0.05° — **grid-invariant**.

### The controls, and the three probes they refuted

Every panel carries a control that must read a known value exactly.

| control | must read | read |
|---|---|---|
| constant-velocity straight line | turn 0, flip 0, jerk 0 | **exact** |
| constant-curvature arc | flip 0, dκ/ds 0 | **exact** (jerk 0.001) |
| hold-action arm, 4,438 windows | flip 0, jerk 0 | **exact** (dκ/ds 6e-6) |
| numpy `roll64` vs `rollout_unicycle` | 0 m | **0.000e+00** |
| unicycle fitter on a path that *is* a unicycle rollout | 0 m | **1.4e-14 m** |

⛔ **Three first attempts were refuted by these controls and are recorded so nobody re-derives them:**
- the **unicycle fitter v1** read **3.36 m** on a path that *is* a unicycle rollout — the float32
  `rollout_unicycle` was being finite-differenced with a float64 step, so the Jacobian was noise;
- the **sign-flip metric v1** read **0.3678 zig-zag on the banked constant-velocity path** — ±1e-16 rad
  of float noise on exactly-zero turns;
- computing refc-base's **offsets** as `fan − default_anchors` was inadmissible: the anchor-identity
  control read **0.0 % diagonal-nearest against a chance of 0.8 %** — refc-base did not train on that
  vocabulary. That failure is what led to finding #5.

---

## 1 · Is the roughness a rendering artefact? — **partly, and the factor is 7.44×**

`render_refcv3_video.py:1025` draws GT from `_ego_future(poses, t0, 60)` — **60 dense 10 Hz poses**.
`:1027` draws the model from `densify(sel)` — **the 8 emitted slots joined by straight lines**
(`densify`, `:403-417`). The two polylines on the same frame differ in resolution by 7.5×.

| the SAME GT path, drawn two ways (n=130) | mean \|turn\| | p90 | max |
|---|---|---|---|
| **GT @ 10 Hz, 60 pts — as GT is drawn** | **0.3067°** | 0.517 | 0.68 |
| **GT @ refcv3's 8 slots — as the model is drawn** | **2.2820°** | 4.560 | 6.02 |
| refcv3 @ its own 8 slots | 1.3074° | 2.866 | 8.92 |

- **Amplification of a smooth path by the 8-slot drawing: ×7.44.**
- refcv3's kink vs the dense GT polyline beside it: **×4.26** — this is what the eye compares.
- refcv3's kink vs the **same** GT on the **same** grid: **×0.57.**

**The 2 s seam is a second, independent grid artefact.** refcv3's slots are non-uniform (0.5 s ×4 then
1.0 s ×4). For a path whose true curvature is **exactly constant**, the drawn turn angle reads
`11.459 / 11.459 / 11.459 / 17.189 / 22.918 / 22.918 / 22.918°` — a **1.50× step at the 2 s vertex and
2.00× after it**. On refc-base's uniform 4-slot grid the same arc reads **11.459° flat**. Measured on
real GT: turn pre-seam **1.431°** → post-seam **3.137°** (**2.19×**) while its arc-length curvature is
flat (0.00379 → 0.00415, **1.10×**). ⇒ **A perfectly smooth path looks twice as kinked after 2 s on
refcv3's grid and perfectly uniform on refc-base's** — which is precisely the difference the PI
describes, and it is in the grid, not the model.

---

## 2 · What is *not* the renderer — the model defect

`sign-flip` cannot be manufactured by resampling: GT reads **0.000** dense and **0.017** at 8 slots.

### A · 6 s / 8-slot grid, refcv3's real output (n=130)

| path | mean\|turn\| ° | mean\|κ\| 1/m | mean\|dκ/ds\| 1/m² | mean\|jerk\| m/s³ | **sign-flip** |
|---|---|---|---|---|---|
| **refcv3** | 1.307 | **0.00235** | 0.000134 | **1.994** | **0.2016** |
| GT | 2.282 | 0.00397 | 0.000092 | 0.342 | 0.0168 |
| CONTROL constant-velocity | 0.000 | 0.00000 | 0.000000 | 0.000 | 0.0000 |
| CONTROL constant-curvature arc | 1.960 | 0.00343 | 0.000000 | 0.001 | 0.0000 |

**vs GT: jerk ×5.83 · dκ/ds ×1.46 · sign-flip ×12.01 · curvature ×0.59.**

Two findings, and they point in opposite directions:
- **Laterally refcv3 is not "rough" — it is too STRAIGHT.** Mean curvature **0.59× GT**, and past the
  seam it **collapses**: κ pre-seam 0.00364 → post-seam **0.00134** (**0.37×**) while GT's is flat
  (**1.10×**). **After 2 s the model emits a nearly straight line.** That is the frame where "GT turns
  hard left and the whole fan points forward", quantified.
- **Longitudinally it is genuinely rough.** Per-segment implied speed:
  GT `15.78 15.49 15.22 14.97 14.64 14.29 14.00 13.78` (monotone deceleration) vs
  refcv3 `16.10 16.29 16.32 **17.09** 16.48 16.89 16.96 16.84` from a measured `v0 = 15.90` — it
  **accelerates where GT decelerates**, and spikes on the 2.0→3.0 s segment. That irregular spacing of
  the emitted circles is a large part of what reads as "not smooth".

### B · 2 s / 4-slot grid — three arms, PAIRED on the same 170 windows, **uniform grid, no seam**

| path | mean\|turn\| ° | mean\|κ\| | mean\|dκ/ds\| | mean\|jerk\| | **sign-flip** |
|---|---|---|---|---|---|
| **refcv3** | 1.353 | 0.00323 | **0.000197** | **2.396** | **0.1855** |
| **refc-base** | 1.585 | 0.00421 | 0.000137 | 0.856 | 0.1049 |
| flagship-v1 | 1.485 | 0.00663 | 0.001541 | 1.114 | 0.2128 |
| GT | 1.255 | 0.00330 | 0.000092 | 0.397 | 0.0067 |
| CONTROL constant-velocity | 0.000 | 0.00000 | 0.000000 | 0.000 | 0.0000 |
| CONTROL constant-curvature arc | 1.229 | 0.00323 | 0.000000 | 0.000 | 0.0000 |

⇒ **refcv3 vs refc-base, paired: jerk ×2.80, dκ/ds ×1.44, zig-zag ×1.77.** **The PI's comparison is
correct and the regression is real.** It is not the seam (this grid has none) and not the renderer
(these are metric quantities on the emitted waypoints).

### C · the same at large n, on the banked dumps

| arm | n | sign-flip | jerk | dκ/ds | vs its own GT |
|---|---|---|---|---|---|
| **refcv3** @ 40,284, 141 eps | 4,438 | **0.4425** | 1.651 | 0.011291 | jerk **×4.99** · flip **×6.15** · dκ/ds **×11.81** |
| **refc-base** @ 29,999, 40 eps | 828 | 0.2818 | 0.610 | 0.007089 | jerk ×1.96 · flip ×4.49 · dκ/ds ×4.47 |
| refcv3 **ORACLE-selected** anchor | 4,438 | 0.4656 | 2.035 | 0.011061 | — |
| refc-base **WHOLE FAN** (128 cands) | 105,984 | 0.4834 | 11.940 | 0.124876 | — |

⭐ **The oracle-selected candidate is just as rough as the selected one (0.4656 vs 0.4425).**
**Roughness is not a selection failure — every candidate in the emitted fan is rough.**

---

## 3 · The parameterisation and anchor diff

| | **refcv3** | **refc-base** |
|---|---|---|
| slots | **8** at 0.5/1/1.5/2/3/4/5/6 s (`refc_v3.py:120 V3_HORIZONS`) | **4** at 0.5/1/1.5/2 s (`refc.py:304-310`) |
| spacing | **NON-UNIFORM** — 0.5 s ×4 then 1.0 s ×4 | **uniform** 0.5 s |
| representation | free ego-frame waypoints | free ego-frame waypoints |
| emission | `x = anchors + offset` (`refc.py:1400`) | *identical* |
| **anchor vocabulary** | **SYNTHETIC** — `refc.default_anchors` = FPS over 4,096 `synth_anchor_pool` rollouts | **DATA-DRIVEN** — `refc_anchors_base128.pt`, `build_refc_anchors.py --data-root`, FPS over **real ego-frame GT trajectories** (`MODEL_REGISTRY.md` §4.3) |

**Why refcv3 got the synthetic set — three probes, differing in path-binding:**
1. `stack/scripts/refcv3_b1_launch.sh:122-133` and `sup_refcv3.sh:64` pass **no `--anchors`**;
2. `refc_v3_train.py:1544` — `--anchors default=None`, *"the model's synthetic default otherwise"*;
3. repo-wide, **every banked anchor table has horizons `[5,10,15,20]` = 2.0 s** — `refc_anchors_full_REBUILD.pt`,
   `refc_anchors_small64.pt`, `anchors_dev256.pt`, stated independently at `tanitad/data/anchor_goal.py:36`,
   `tanitad/data/g_tac_geom.py:210` and `tanitad/models/v6.py:2736`. **No 6 s vocabulary was ever built.**

### ⭐ The measurement that makes this the headline

**Oracle-in-vocabulary ADE 0–2 s**, split-half, the data-driven arm scored **out of sample**, paired on
the same 2,412 held-out windows of real GT (from the 4,823-window dump):

| vocabulary | mean | p50 | p90 |
|---|---|---|---|
| **SYNTHETIC 128 — refcv3's actual set** | **0.9433** | 0.9275 | 1.2821 |
| **DATA-FPS 128 — refc-base's construction** | **0.4369** | 0.4434 | 0.6547 |
| CONTROL: one constant-velocity straight line | 0.6780 | 0.4385 | 1.6309 |
| CONTROL: zero path | 14.1801 | — | — |

**×2.16, +0.5064 m, synthetic worse on 91.7 % of windows — and the synthetic vocabulary's *oracle*
(0.9433 m) is worse than a single straight line (0.6780 m).** *If the fan does not contain a good
trajectory, no selector can find one* — this is that statement, measured.

**The vocabulary is also not feasible.** `synth_anchor_pool` (`refc.py:176-207`) integrates a
**constant yaw-rate that is independent of v** and clamps v at 0; `rollout_unicycle` — the Alpamayo
action space — uses `yaw_rate = v · κ` and its own docstring says a path that turns at a standstill is
*"a path no vehicle drives"*. Measured on refcv3's 128 anchors: **20.98 % of segment-vertices exceed a
dry-road Kamm circle (0.7 g), 41/128 anchors violate somewhere, peak lateral accel 13.91 m/s² (1.4 g),
and 5 anchors keep turning after stalling.**

---

## 4 · What the Alpamayo learning actually is, and in which unit convention

**Adopted:** (a) the **factored lat × lon tactical head** — `refc_v3.py:412` cites *"Alpamayo 40.62 %
dual-axis"*; (b) the anchor **pool** is a unicycle rollout.

⛔ **Not adopted: the (accel, curvature) action space itself.** `tanitad.models.kinematic`'s
`rollout_unicycle`, `unicycle_controls_from_path`, `A2S_ACCEL_LIMIT`, `A2S_CURVATURE_LIMIT` are
imported by **`refs/refa_v1_plan.py` only**. `refc_v3.py` and `refc_v3_train.py` import **nothing**
from that module.

**The loss confirms it.** `refc_v3_train.py:634`:
`loss = TRAJ·L1(waypoints) + ANCHOR_CLS + LAW + ROUTE + LAT + LON` (+ `GOAL_TAC`/`GOAL_STR`/`SEL_V3`).
**No jerk term, no curvature term, no smoothness term, no feasibility term.** An L1 on absolute
waypoint positions penalises *where* each of the 8 points is and **never how they connect** — a path
may zig-zag freely as long as each waypoint lands near its target.

⇒ **Does it guarantee feasibility? No.** The only kinematics in refcv3 is in the anchor *pool*, and
32 % of those anchors are infeasible; the emitted path is anchor + free offset and is unconstrained.
⇒ **Did it shrink or distort the reachable set relative to refc-base?** Not by tightening — it
*enlarged* the vocabulary's angular span (terminal bearings to 74.5° vs base's 23.6°) while making its
**oracle 2.16× worse**, because the span is synthetic rather than where real trajectories live.

**The unit convention is NOT implicated — a clean negative.** The reachability gate is
**position-only** (`‖wp_last‖ / horizon_s`); `synth_anchor_pool` samples **yaw-rate in rad/s**, never
steer; and refcv3's config carries no `ego_state_inject` and no `goal_kin_extrap`, so the
`L = 2.9` steer constant (`physicalai.py:621`) **never enters its forward path**. It touches only the
`ha` control arm, where `refcv3_arm.py` already applies `κ = tan(steer)/2.9`. The suspected ~2.9×
tighten/loosen cannot be happening — **because there is no feasibility constraint on the output at all.**

---

## 5 · The reachability gate — `reach_keep` is *not* the culprit

Rule (`flagship_v15.reachability_mask`): `‖wp_last‖/T ∈ [max(0, v0 − a·T), v0 + a·T]`, `a = 2.5`.
`T` is **derived** (`refc.py:652`): refcv3 **6.0 s → band ±15.0 m/s**; refc-base **2.0 s → ±5.0 m/s**.

| | refcv3 (anchors, 4,823 windows) | refc-base (raw anchors) | refc-base (**decoded fan**, 881 windows) |
|---|---|---|---|
| **killed** | **37.10 %** (mean **47.5/128**) | 77.28 % | **73.76 %** |
| survivors / window | 80.5 | 29.1 | 33.6 |
| windows with an empty survivor set | **0.00 %** | 0.00 % | 0.00 % |

**refcv3's gate is far LOOSER than refc-base's, because the band half-width is `a·T` and `T` tripled.**
Kill rate by `v0` (m/s): `0-2` 92.0 · `2-5` 77.4 · `5-10` 57.0 · `10-15` 35.3 · `15-20` **16.4** ·
`20-25` 9.9 · `25-40` 32.0 out of 128 — so the frame's `killed by reach_keep: N/128` is a **speed
readout**, and at highway speed it is ~10–20/128.

**Did a left-turning anchor exist and get killed, or never exist?** ⇒ **It existed and it survived.**
Of refcv3's 128 anchors, **61 have terminal bearing > 30°**, of which **40.4 survive per window on
average, and 0.00 % of windows have zero survivors** (>60°: 10 exist, 8.06 survive, 0.00 % empty).
And on refc-base's *decoded* fan the gate costs almost nothing: the best-matching candidate is
**0.22° off GT's bearing**, **0.32°** once restricted to survivors — **a cost of 0.106°**; the selected
anchor survives on **100.00 %** of windows.

⚠️ **UNVERIFIED and it is the one gap:** refcv3's kill rate on its **own decoded fan** is not
measurable from any banked artifact — `refcv3_arm.py` banks only `anchor_traj[a_star]`, never the full
fan or `reach_keep`. The 37.10 % is the **raw-anchor** rate; on refc-base the decode moved the rate by
**−3.52 pp**, so the decoded figure is expected near 33–37 %.

---

## 6 · The ADE decomposition, and why the two defects are one defect

All 4,823 banked windows, 0–2 s:

| arm | ADE (m) |
|---|---|
| **refcv3 emitted (`os`)** | **0.4419** |
| hold-action control (`ha`) | **0.2996** |
| constant-velocity control (`ha0`) | 0.6723 |
| oracle-**selected** anchor, refined (`oracle_sel`) | 0.3668 |
| **ORACLE-IN-RAW-ANCHOR (no offset)** | **0.9368** |

Hold-action beats the model by **0.1423 m**; a perfect chooser buys **0.0751 m** — both reproduce the
figures the brief carries, which cross-validates this pipeline against the banked suite.

⭐ **The free offset head buys +0.4948 m over the best anchor it could possibly have picked.** The
vocabulary is so poor that the offset must *override* it — and the offset is exactly the unconstrained
mechanism that destroys smoothness. **Fix the vocabulary and the offset no longer has to be large;
constrain the offset without fixing the vocabulary and accuracy collapses to 0.94 m.** They must be
fixed together.

## 7 · What a kinematically-constrained head would cost — **nothing**

Best piecewise-constant `(accel, curvature)` unicycle path fitted through the real GT slots.
**16 controls vs 16 free waypoint coordinates — the same parameter count.**

| grid | n | best unicycle-constrained ADE | refcv3's own emitted ADE |
|---|---|---|---|
| 6 s / 8 slots | 130 | **0.0000 m** (p90 0.0000) | 5.0701 m |
| 2 s / 4 slots | 600 | **0.0080 m** (p50 0.0000) | 0.4487 m |

Fitter controls: a true unicycle path fits to **1.4e-14 m**; the numpy integrator matches
`rollout_unicycle` to **0.000e+00 m**.

⇒ **The Alpamayo action space costs ZERO representational capacity on this corpus. Feasibility is
free.** refcv3's paths are rough not because the constraint was too tight — **but because it was never
applied to the output.**

---

# refcv4 — the recommendation

Ordered by measured evidence, each with the cheapest experiment that would falsify it.

### R1 (blocking) — build the 6 s **data-driven** anchor vocabulary and pass `--anchors`
*Evidence:* synthetic oracle **0.9433 m** vs data-FPS **0.4369 m** (×2.16, worse on 91.7 % of 2,412
held-out windows) and **worse than a straight line (0.6780 m)**; no 6 s table exists in the repo; the
launch script omits the flag.
*Action:* `build_refc_anchors.py --data-root <v2 cache> --horizons 5 10 15 20 30 40 50 60 --n-anchors 128`,
then add `--anchors` to `refcv3_b1_launch.sh`. ⚠️ `build_refc_anchors.py` currently takes horizons from
its caller — verify it emits `[128, 8, 2]`, and re-measure the reach survivor counts, which are a
property of *those* anchors (`refc_select.py:210`).
*Falsifier (zero GPU, minutes):* rebuild the table and re-run `run_vocab.py`. **If the data-driven 6 s
oracle is not ≥1.5× better than 0.9433 m, R1 is wrong.**

### R2 (blocking, and it is the PI's actual question) — emit **controls**, not waypoints
*Evidence:* the emitted path is `anchors + free offset` (`refc.py:1400`) with **no kinematic model and
no smoothness term in the loss** (`refc_v3_train.py:634`); a unicycle-constrained path with the **same
parameter count** reproduces real GT to **0.0000 m**.
*Action:* replace the `[S, 2]` offset head with an `[S, 2] = (accel, curvature)` head integrated
through `kinematic.rollout_unicycle` with `A2S_ACCEL_LIMIT`/`A2S_CURVATURE_LIMIT` via `_squash`.
Anchors become control sequences; the reach gate then applies to a provably feasible set. ⚠️ Keep the
`_squash` knee — `clamp` and `tanh` are both dead-gradient traps (`kinematic.py:186-207`).
*Falsifier (dev-box, ~30 min/arm on the `V3_RIG_SIZES["tiny"]` rung under `TanitAD_ValidateAIDesign`):*
train tiny with the control head vs the waypoint head on identical data. **If the control head's ADE is
worse by more than the 0.0080 m representational ceiling, the constraint is not free and R2 is wrong.**

### R3 (cheap, do it anyway) — make the grid **uniform**, or state the seam on the frame
*Evidence:* a path of exactly constant curvature reads `11.46/11.46/11.46/17.19/22.92/22.92/22.92°` on
refcv3's grid (a 1.50×/2.00× step) and `11.46°` flat on refc-base's; GT itself shows **2.19×** across
the seam at constant arc-length curvature.
*Action:* prefer uniform slots — `(5,10,15,20,25,…,60)` at 12 slots, or 8 uniform at 0.75 s. If the
non-uniform grid is kept for label reasons, the **renderer must densify the model's path through the
same interpolation it uses for GT** and print the per-slot dt.
*Falsifier:* re-render one clip with a 60-point model polyline. **If the "kinked" impression survives a
like-for-like draw, R3 is not the visual cause** — the metric numbers say ~7.4× of it is.

### R4 (loss) — add a **curvature-rate** term and a **jerk** term
*Evidence:* refcv3 zig-zags **6.15× GT** and jerks **4.99× GT** on 4,438 windows; the **oracle**
candidate is just as rough (0.4656), so no selector fixes it; the loss has no term for either.
*Action:* under R2 these become penalties on the emitted controls (`|Δκ|`, `|Δa|`) — one line each, and
`kamm_circle_violation` already exists in `kinematic.py` as a third.
*Falsifier:* the same tiny-rung pair with/without the terms. **If sign-flip does not fall below GT's
×2 without an ADE regression, R4 is wrong.**

### R5 (do not touch) — leave `reach_keep` alone
*Evidence:* it kills **37.10 %** at 6 s (vs refc-base's **73.76 %** on the decoded fan), leaves
**0.00 %** of windows empty, and a **>30° turning anchor survives in 100 % of windows**; on refc-base
its cost is **0.106°** of bearing. It is not the reason the fan points forward.
*Action:* **none**, except instrumentation — see R6. If R2 lands, revisit `accel_max` once, because the
band's half-width `a·T = 15 m/s` at 6 s is so wide it is nearly inert.

### R6 (instrumentation, blocking for the next diagnosis) — bank the fan
*Evidence:* refcv3's decoded-fan kill rate and fan span are **UNVERIFIED** because `refcv3_arm.py`
banks only `anchor_traj[a_star]`. `fan_refc-base-30k.pt` shows exactly what is needed.
*Action:* add `--dump-fan` to `refcv3_arm.py` writing `anchor_traj` + `reach_keep` + `sel_score_v3` per
window (4,823 × 128 × 8 × 2 fp16 ≈ **19.8 MB**). `--analyze-only` already exists
(`refcv3_arm.py:1998`, `openloop_suite.py:548`) — ⛔ **check it before re-rolling anything.**

---

## Deliverable manifest

| artifact | lives at | only copy? |
|---|---|---|
| this report | `repo: taniteval/results/2026-09-04-refcv3-smoothness/RESULT.md` | no |
| headline record (machine-readable) | `repo: …/raw/WP56_HEADLINE.json` | no |
| paired 6 s + 2 s tables | `repo: …/raw/FINAL_smoothness.json`, `paired_smoothness.json` | no |
| large-n tables | `repo: …/raw/WIDE_smoothness.json` | no |
| render-amplification record | `repo: …/raw/render_artifact.json` | no |
| the 11 analysis scripts (rerunnable, zero GPU) | `repo: …/raw/scripts/` | no |

**Escalation.** R1 and R2 are **launch-blocking for refcv4** and are architecture + launch-script
changes, not doc changes. R6 blocks the next round of fan diagnosis. No training run was launched.
