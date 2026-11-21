# The closed loop finally has a floor — and on this scene the floor wins

**Instrument:** `stack/experiments/alpasim-gsplat/closedloop_drive.py` (three new trivial
policies) + `openloop_drive.py` (`--floor`) → `cl_metrics.py`, **unchanged scorer** ·
**host:** Jetson Thor (`tanitad-thor-wifi`) · **scene:** `00040136-e651-4abd-991d-0655ccda9430`,
`--condition empty` · **written:** 2026-09-04 · **owner:** Architecture & Inference FlyWheel
**Answers:** *"every closed-loop number TanitAD has published lacks an `ha0`-equivalent bar —
build one, run it, and re-read the existing panels against it."*

---

## ⭐ VERDICT IN SIX LINES

1. **The closed loop now has a trivial floor.** Three arms — `cl_ha0` (constant velocity at
   the measured `v0`), `cl_ha` (hold the action closing at t0), `cl_ha0_ext` (constant
   acceleration **and** constant yaw rate; CTRA) — driven through the **identical**
   render → plan → `wp_to_control` → bicycle → `GroundFollower` → `cl_metrics.py` path as
   every model arm.
2. ⛔ **NEITHER refcv3 NOR refc-base BEATS THE FLOOR IN CLOSED LOOP.** Paired over the same
   windows: `refcv3 − cl_ha0` = **+0.1307 m [−0.6213, +0.8809]**, `refc-base − cl_ha0` =
   **−0.0897 m [−0.4029, +0.1994]**. Both intervals straddle zero. The best closed-loop
   `ade_0_2s` on this scene belongs to a **trivial CTRA extrapolation** (`cl_ha0_ext`
   **2.6552 m**), with refc-base tied at 2.6554 m and refcv3 behind at 2.8755 m — those
   are LEVELS over slightly different window sets; the PAIRED margins in §3 are the
   admissible statement and they say the same thing.
3. ⛔ **THE 2026-08-03 HEADLINE WAS 98.75 % A MODEL LOSING TO A STRAIGHT LINE.** The published
   `flagship-v1 − refc-base = +7.1642 [+5.2654, +8.9661]` decomposes **exactly** on the same
   437 windows into `flagship-v1 − cl_ha0` = **+7.0745 [+5.1151, +9.0440]** (separated, floor
   wins) minus `refc-base − cl_ha0` **−0.0897** (indistinguishable). It was never evidence
   that refc-base drives; it is evidence that flagship-v1 is **7 m worse than doing nothing**.
4. ⭐ **IN OPEN LOOP, ON A CORPUS AND AN INSTRUMENT THAT SHARE NOTHING WITH THE 2026-09-04
   PhysicalAI READ, THE SAME FINDING REPRODUCES.** `refcv3 − cl_ha0_ext` = **+0.3558 m
   [+0.0224, +0.6981]**, separated, **floor wins**; `refcv3 − cl_ha` = +0.3510 [+0.0183,
   +0.6921], also separated. The PhysicalAI panel measured `os − ha` = +0.1423 [+0.1187,
   +0.1658] on 4,823 windows / 141 episodes with `refcv3_arm.py`. Two differently-bound
   probes, same verdict.
5. ⭐ **WHAT THE MODELS DO BUY IS LATERAL, NOT LONGITUDINAL.** Closed loop vs `cl_ha0`,
   refcv3 wins **every** lateral metric (cross-track −1.6906 m, heading −0.0716 rad,
   curvature −0.0026 1/m, yaw-rate −0.0284 rad/s, lateral ADE −0.2188 m) and **no**
   longitudinal metric (all three straddle zero). Against the *strongest* floor
   (`cl_ha0_ext`) even most of the lateral advantage disappears — only heading and curvature
   error still separate.
6. ⛔ **AND THE TACTICAL FAMILY GOES THE OTHER WAY.** `manoeuvre_plan_eq_logged`: the floor's
   plan agrees with the logged manoeuvre **more often** than refcv3's does — closed loop
   −0.1517 [−0.2736, −0.0489] vs `cl_ha0` and −0.2069 [−0.4184, −0.0244] vs `cl_ha0_ext`;
   open loop refcv3 **0.2882** against `cl_ha0_ext`'s **0.5588**.

**Scope, carried into every line above: n = 1 SCENE, 9 rollout starts used as bootstrap
clusters, 435–437 windows.** §7 says exactly what that does and does not license.

---

## 1 · WHAT WAS BUILT — a POLICY, not a second harness

`KinematicFloorPolicy` (`closedloop_drive.py`) satisfies the same `plan(frames, intr, v0,
nav_cmd, nav_known)` contract as `RefCV3Policy`, so it is driven by `run_rollout` through the
same code with the same `window = 8` (hence the same `f0 = start + 9`, the same first-decision
pose, the same 50 ticks). **The frames are still rendered — they are simply never read.**

| arm | control law | what it consumes |
|---|---|---|
| **`cl_ha0`** | `a = 0`, `kappa = 0` — constant velocity, straight, forever | the current speed only |
| **`cl_ha`** | constant `a0` and constant **curvature** `kappa0` (a fixed steering angle) | + the action closing at t0 |
| **`cl_ha0_ext`** | constant `a0` and constant **yaw rate** `omega0` (CTRA) | + the action closing at t0 |

⛔ **NO FUTURE INFORMATION.** `a0` and `omega0` are backward differences of the logged rig
poses at indices `f0-2, f0-1, f0` — every index `<= t0`. This is pinned by
`test_set_t0_reads_no_future_pose`, which **poisons every pose after `t0` with NaN** and
requires the answer to be unchanged; presence of a plausible number is not evidence, only the
poisoning is. *(PI ruling 2026-09-02: velocity at cycle time is a legal initial state.)*

⚠️ **One inherited asymmetry, named rather than hidden.** The harness's own initial speed is
`gt_poses_xyv`'s **forward** difference, which uses pose `f0+1`. That is the harness's
convention, fed identically to every model arm; the floor consumes it through `plan(v0=…)`
unchanged. Changing it for the floor alone would have broken the pairing the floor exists to
support. The floor's *own* measurements (`a0`, `omega0`) use no such value.

⚠️ **THE CONTROLLER IS NOT A PASS-THROUGH, AND THE FLOOR SUFFERS IT TOO.** `wp_to_control`
reads only the 0.5 s waypoint and sets `v_target = x/(L·dt)` — the **average** speed over the
lookahead — while `rollout_unicycle` advances position on the speed at the *start* of each
step. So a constant intended acceleration `a` is executed as
`a·dt·(L−1)/2 / speed_tc = **0.4·a**`. MEASURED, not derived-and-assumed: I first wrote `0.5a`
from the continuous-time integral and `test_the_p_speed_controller_executes_40_percent_…`
caught it. Reported, never corrected — correcting it would make this harness incomparable with
every published TanitAD closed-loop number.

---

## 2 · ⛔ CONTROLS THAT MUST READ KNOWN VALUES — expected beside measured

| # | control | expected | measured | verdict |
|---|---|---|---|---|
| C1 | `cl_ha0` commanded **steer**, every tick, 450/450 | exactly `0.0` | **0.0** | ✅ |
| C2 | `cl_ha0` commanded **accel**, every tick, 450/450 | exactly `0.0` | **0.0** | ✅ |
| C3 | `cl_ha0` `v_target − v`, every tick | exactly `0.0` | **0.0** | ✅ |
| C4 | `cl_ha0` speed constant within each rollout | `True` | **True** (9/9) | ✅ |
| C5 | `cl_ha0` plan lateral offset `y` | exactly `0.0` | **0.0** | ✅ |
| C6 | **DETERMINISM** — `cl_ha0` re-run, identical flags | bit-identical | **450/450 steps**, max Δplan/Δego/Δv/Δsteer/Δaccel = **0.0** | ✅ |
| C7 | **RENDER INDEPENDENCE** — `cl_ha0` under the *morning* render (no cull, no sky) vs the HQ render | bit-identical | **450/450 steps**, all max Δ = **0.0** | ✅ |
| C8 | C6/C7 at the **metric** level (`cl_metrics.py`, 437 windows) | every scored metric `delta = 0.0`, zero-width CI | **13/13 scored, 0 non-zero** on both | ✅ |
| C9 | **STATIONARY START** — `cl_ha0` from `v = 0` | plan all-zero; pose and yaw unchanged after 50 ticks | exact (`test_ha0_from_a_stationary_start_stays_stationary`) | ✅ |
| C10 | **DEGENERACY NESTING** — `cl_ha`/`cl_ha0_ext` on a straight constant-speed log | bit-identical to `cl_ha0` | `np.array_equal` | ✅ |
| C11 | **NO FUTURE POSE** — poses after t0 set to NaN | t0 state and plan unchanged, finite | `t0_clean == t0_poisoned` for all 3 arms | ✅ |
| C12 | **OPEN-LOOP DEGENERATE FIVE** (`cross_track_abs_m`, `cross_track_signed_m`, `dist_to_gt_traj_m`, `executed_speed_err_ms`, `route_corridor_departure_rate`) | exactly `0.0` because the ego IS the logged path | **0.0** on the floor arms *and* on refcv3 | ✅ |
| C13 | `f_eff` / `ckpt` / `canon.reads_pixels` on a floor payload | `null` / `null` / `false` | `null` / `null` / `false` | ✅ |

**C7 is the load-bearing one.** It is the positive proof that the floor reads no pixels, and it
is what licenses using ONE floor run as the bar for panels rendered differently.

⭐ **A fourteenth control fell out of the data.** In OPEN loop `cl_ha0`'s
`abs_target_speed_err_ms` is **exactly 0.0000** — by construction, since `v_target == v ==`
the log's own speed at the scored index. refcv3 reads **0.2028** and refc-base **0.0526** on
the same windows. Both models are *worse at setting the target speed than repeating the speed
they were handed.*

**Unit tests:** `stack/tests/test_closedloop_floor.py`, **29 passed**. Also asserted there:
the closed-form `a=0, kappa=0` branch agrees with the programme's own
`kinematic.rollout_unicycle` to `< 1e-12`; `cl_ha0_ext` really holds `yaw(k) = omega0·k·dt`;
`cl_ha` really holds a constant curvature; the `set_t0` hook is **inert** for model policies
(none of `FlagshipV1Policy` / `RefCPolicy` / `RefCV3Policy` has the attribute), so no
historical rollout can move.

---

## 3 · CLOSED LOOP — the levels, with the floor in the table

`ade_0_2s`, mean over windows, episode-cluster bootstrap over the 9 rollout starts.

| arm | n | `ade_0_2s` | 95 % CI (episode-cluster bootstrap, 9 clusters) |
|---|---|---|---|
| **`cl_ha0_ext`** (floor) | 436 | **2.6552** | [1.6350, 3.9873] |
| `refc-base` | 437 | 2.6554 | [1.7555, 3.5254] |
| **`cl_ha`** (floor) | 436 | 2.6651 | [1.6613, 3.9886] |
| **`cl_ha0`** (floor) | 437 | **2.7452** | [1.9395, 3.5615] |
| `refcv3` @ 40,284 | 435 | 2.8755 | [1.6750, 3.9814] |
| `flagship-v1` | 450 | 9.6955 | [8.2293, 11.3271] |

⚠️ **These unpaired intervals overlap heavily and cannot separate anything** — which is exactly
why the paired table below exists. On 9 clusters the unpaired CI is ~±1.2 m wide while the
paired one is ~±0.75 m, and only the paired estimator holds the windows fixed.

**The paired margins are the admissible statement, not the levels** (different arms truncate
different windows). `delta > 0` on an error metric means **the arm is worse than the floor**.

| contrast | `ade_0_2s` margin | verdict |
|---|---|---|
| `refcv3 − cl_ha0` | **+0.1307 [−0.6213, +0.8809]** | indistinguishable |
| `refcv3 − cl_ha` | +0.2074 [−1.2728, +1.5305] | indistinguishable |
| `refcv3 − cl_ha0_ext` | +0.2173 [−1.2725, +1.5321] | indistinguishable |
| `refc-base − cl_ha0` | **−0.0897 [−0.4029, +0.1994]** | indistinguishable |
| `refc-base − cl_ha` | −0.0091 [−1.3723, +1.1683] | indistinguishable |
| `refc-base − cl_ha0_ext` | +0.0008 [−1.3627, +1.1809] | indistinguishable |
| `flagship-v1 − cl_ha0` | **+7.0745 [+5.1151, +9.0440]** | ⛔ **floor wins** |
| `flagship-v1 − cl_ha0_ext` | +7.1703 [+5.5969, +8.8889] | ⛔ **floor wins** |
| `cl_ha − cl_ha0` | −0.0798 [−1.3222, +1.3652] | indistinguishable |
| `cl_ha0_ext − cl_ha0` | −0.0897 [−1.3329, +1.3530] | indistinguishable |

### 3.1 · THE FOUR FAMILIES, SEPARATELY — `refcv3 − cl_ha0`, closed loop, n = 435

| family | metric | margin vs floor | verdict |
|---|---|---|---|
| **ADE** | `ade_0_2s` | +0.1307 [−0.6213, +0.8809] | indistinguishable |
| | `dist_to_gt_traj_m` | −1.6906 [−2.8907, −0.4545] | arm wins |
| **LONGITUDINAL** | `abs_target_speed_err_ms` | +0.2529 [−0.3100, +0.8428] | indistinguishable |
| | `abs_executed_speed_err_ms` | +0.1885 [−0.2929, +0.6987] | indistinguishable |
| | `along_track_ade_m` | +0.2316 [−0.5502, +1.0440] | indistinguishable |
| | distance-keeping (`real_lead_*`) | **n = 0** | ⚠️ no annotated lead in frame on this scene under `--condition empty`; not computable, not dropped silently |
| **LATERAL** | `lateral_ade_m` | −0.2188 [−0.3170, −0.1239] | arm wins |
| | `heading_err_rad` | −0.0716 [−0.0972, −0.0461] | arm wins |
| | `curvature_err_1pm` | −0.0026 [−0.0035, −0.0016] | arm wins |
| | `yawrate_err_rads` | −0.0284 [−0.0422, −0.0140] | arm wins |
| | `cross_track_abs_m` | −1.6906 [−2.8907, −0.4545] | arm wins |
| **TACTICAL** | `manoeuvre_plan_eq_logged` | **−0.1517 [−0.2736, −0.0489]** | ⛔ **floor wins** |
| | `manoeuvre_head_eq_logged` | **n = 0** | the floor has no manoeuvre head — a trivial control that consulted one would not be trivial |
| **STRATEGIC** | `route_corridor_departure_rate` | −0.1816 [−0.3619, −0.0088] | arm wins |
| | `route_head_eq_logged` | **n = 0** | the floor has no route head (same reason) |

⚠️ `cross_track_abs_m`, `dist_to_gt_traj_m` and `route_corridor_departure_rate` are **one
signal in three costumes** here — the last two are computed from the cross-track deviation
(`cl_metrics.py:386`). Read them as one lateral result, not three.

### 3.2 · Against the STRONGEST floor — `refcv3 − cl_ha0_ext`, closed loop, n = 435

| family | metric | margin | verdict |
|---|---|---|---|
| ADE | `ade_0_2s` | +0.2173 [−1.2725, +1.5321] | indistinguishable |
| LONGITUDINAL | `abs_target_speed_err_ms` | +0.3076 [−0.6706, +1.1392] | indistinguishable |
| LATERAL | `lateral_ade_m` | +0.0567 [−0.0820, +0.2110] | indistinguishable |
| | `heading_err_rad` | −0.0337 [−0.0574, −0.0074] | arm wins |
| | `curvature_err_1pm` | −0.0014 [−0.0021, −0.0008] | arm wins |
| | `yawrate_err_rads` | −0.0040 [−0.0114, +0.0047] | indistinguishable |
| | `cross_track_abs_m` | +0.0499 [−0.4442, +0.4383] | indistinguishable |
| TACTICAL | `manoeuvre_plan_eq_logged` | **−0.2069 [−0.4184, −0.0244]** | ⛔ **floor wins** |
| STRATEGIC | `route_corridor_departure_rate` | +0.0598 [−0.0759, +0.1534] | indistinguishable |

⇒ **Once the bar is a constant-turn-rate extrapolation rather than a straight line, refcv3's
entire measurable closed-loop advantage is two lateral shape terms — and it is behind on
manoeuvre agreement.**

---

## 4 · ⛔ THE RE-READ — both published panels, as margins over `cl_ha0`

### 4.1 · The 2026-08-03 panel (`results/closedloop-hq-render/HQ_flagship_vs_refc_empty.json`)

**Published:** `flagship-v1 − refc-base`, `ade_0_2s` = **+7.1642 [+5.2654, +8.9661]**,
437 paired windows / 9 clusters / 1 scene — quoted since as the programme's headline
closed-loop contrast.

**Re-read on the SAME 437 windows with the SAME estimator:**

| term | margin over `cl_ha0` | verdict |
|---|---|---|
| `flagship-v1 − cl_ha0` | **+7.0745 [+5.1151, +9.0440]** | ⛔ floor wins |
| `refc-base − cl_ha0` | **−0.0897 [−0.4029, +0.1994]** | indistinguishable |
| **sum (identity check)** | **+7.0745 − (−0.0897) = +7.1642** | **exactly the published number** |

⇒ **98.75 % of the published margin is flagship-v1 losing to a constant-velocity line.**
refc-base's own contribution is −0.0897 — 1.25 % of the headline, and not separated from zero.
The panel is a **valid measurement of flagship-v1's failure** and **not** evidence that
refc-base drives. Line one of any future quotation of `+7.1642` must be: *neither arm was
compared to doing nothing, and when it is, one of them is indistinguishable from it.*

⭐ **A free confirmation of the 2026-09-04 patch-neutrality control.** The banked 2026-08-03
rollouts and the v3-tree re-runs, scored against the same floor, give **identical** margins to
four decimals (`+7.0745 [+5.1151, +9.0440]` and `−0.0897 [−0.4029, +0.1994]` from both). The
harness patch really is neutral, now shown through a third path.
⚠️ **What that identity does and does not prove.** `taniteval/ci.py` seeds its resampler
(`seed=0`), so identical intervals follow deterministically from identical inputs — the
identity is evidence about the ROLLOUTS, not an independent statistical agreement. That is the
correct reading and it is the stronger one: the two files agree window-for-window.

### 4.2 · Today's refcv3 panel (`taniteval/results/2026-09-04-refcv3-closedloop/`)

**Published:** `refcv3 − refc-base` = +0.2185 [−0.6198, +1.1586], *"statistically
indistinguishable"*, 435 paired windows.

**Re-read:** `refcv3 − cl_ha0` = +0.1307 [−0.6213, +0.8809]; `refc-base − cl_ha0` = −0.0897
[−0.4029, +0.1994]; the two margins differ by +0.2204, reproducing the published +0.2185 to
within the 435-vs-437 window difference.

⇒ The panel's conclusion survives **and gets stronger and worse at the same time**: the arms
are indistinguishable from each other **and both are indistinguishable from a straight line at
constant speed.** *"refcv3 drives closed-loop"* (verdict line 1 of that panel) is **not
supported by that panel's own instrument once the floor is present**, and should read
*"refcv3 completes a closed-loop rollout; its `ade_0_2s` is not separated from the trivial
control's."*

---

## 5 · OPEN LOOP ON THE SAME SCENE — an independent replication of the PhysicalAI finding

The three floor arms were also run through `openloop_drive.py` (`--floor`, per-tick `set_t0`),
same scene, same render, same 9 disjoint segments, n = 170 windows.

| arm | `ade_0_2s` | 95 % CI | `abs_target_speed_err_ms` | `lateral_ade_m` | `manoeuvre_plan_eq_logged` |
|---|---|---|---|---|---|
| **`cl_ha0_ext`** (floor) | **0.4873** | [0.4207, 0.5490] | 0.1594 | **0.1120** | **0.5588** |
| **`cl_ha`** (floor) | 0.4921 | [0.4264, 0.5529] | 0.1594 | 0.1234 | 0.5588 |
| `refc-base` | 0.6444 | [0.4392, 0.8497] | 0.0526 | 0.1644 | 0.4059 |
| `refcv3` @ 40,284 | 0.8431 | [0.5015, 1.1928] | 0.2028 | 0.1911 | 0.2882 |
| **`cl_ha0`** (floor) | 0.8480 | [0.6055, 1.1058] | **0.0000** | 0.5291 | 0.3824 |
| `flagship-v1` | 7.1479 | [6.0645, 8.3711] | 5.9517 | 0.3077 | 0.4412 |

| contrast | `ade_0_2s` margin | verdict |
|---|---|---|
| `refcv3 − cl_ha0_ext` | **+0.3558 [+0.0224, +0.6981]** | ⛔ **floor wins** |
| `refcv3 − cl_ha` | **+0.3510 [+0.0183, +0.6921]** | ⛔ **floor wins** |
| `refcv3 − cl_ha0` | −0.0049 [−0.2051, +0.1570] | indistinguishable |
| `refc-base − cl_ha0_ext` | +0.1571 [−0.0722, +0.3932] | indistinguishable |
| `refc-base − cl_ha0` | −0.2036 [−0.4556, +0.0045] | indistinguishable |
| `flagship-v1 − cl_ha0_ext` | +6.6606 [+5.5817, +7.9024] | ⛔ floor wins |
| `cl_ha0_ext − cl_ha0` | **−0.3608 [−0.6324, −0.0926]** | the strong floor beats the weak one |

⭐ **WHY THIS MATTERS MORE THAN THE CLOSED-LOOP TABLE.** The 2026-09-04 PhysicalAI open-loop
read found `ha` beating `os` by **+0.1423 [+0.1187, +0.1658]** on 4,823 windows / 141
episodes, using `taniteval/tools/refcv3_arm.py` on the v7.2 episode cache. This run finds the
same sign and a larger magnitude on **NuRec reconstructions**, through a **different
instrument** (`openloop_drive.py` + `cl_metrics.py`), with a **different windowing** and a
**different floor implementation**. That is a second *probe*, not a second sample through one
channel — which is the distinction `CLAUDE.md`'s `ls-tree` rule exists to insist on.

⚠️ And `refcv3 − cl_ha0` is *indistinguishable* while `refcv3 − cl_ha0_ext` is *separated*.
**The choice of floor decides the verdict**, which is precisely why the floor must be named,
its inputs stated, and all three reported rather than one.

---

## 6 · WHAT THE FLOOR MEASURED AT t0 (provenance, per rollout start)

| start | `f0` | `a0` (m/s²) | `omega0` (rad/s) | `kappa0` (1/m) | `v_bwd(t0)` (m/s) |
|---|---|---|---|---|---|
| 0 | 9 | +0.1044 | +0.00209 | +0.000103 | 20.2024 |
| 17 | 26 | +2.0150 | −0.00513 | −0.000259 | 19.8488 |
| 34 | 43 | −1.1451 | −0.00406 | −0.000216 | 18.7543 |
| 51 | 60 | −1.1589 | −0.01654 | −0.000925 | 17.8761 |
| 68 | 77 | −0.3653 | −0.06501 | −0.004145 | 15.6831 |
| 85 | 94 | −2.0501 | −0.08602 | −0.006423 | 13.3938 |
| 102 | 111 | −0.5336 | −0.08986 | −0.007346 | 12.2335 |
| 119 | 128 | +0.9934 | −0.06464 | −0.005294 | 12.2089 |
| 136 | 145 | +0.6168 | −0.05960 | −0.004697 | 12.6901 |

The clip is a decelerating left-ish curve; `|a0|` reaches 2.05 m/s² and `|omega0|` 0.090 rad/s,
so `cl_ha` and `cl_ha0_ext` are **not** degenerate copies of `cl_ha0` on this scene — the
degeneracy is exercised only in the unit test, deliberately.

---

## 7 · ⛔ HONEST SCOPE — this travels with every number above

* **n = 1 SCENE.** Nine rollout starts on one 20 s clip are the bootstrap's resampling unit;
  they are **disjoint segments of one clip, not independent episodes**. This is a
  **provisioning** ceiling, not a compute one: only **2 of 79** NuRec directories carry a real
  `volume.nurec`, and one of the two (`470c4ecc-…`) **is in refcv3's training set** and must
  never be used as a render scene. The whole floor panel cost ~4 minutes of GPU.
* ⛔ **THE LOOP IS CLOSED ON PERCEPTION, NOT ON INTERACTION.** Nothing in the scene responds
  to the ego. There is no map. `route_corridor_departure_rate` is
  `int(abs(cross_track) > CORRIDOR_M)` against the **logged** path (`cl_metrics.py:386`) — it
  is a *deviation-from-the-log* rate, **not an off-road rate**, and must never be quoted as one.
* **WITHIN-SIM RELATIVE.** REF-C's open-loop ADE is 1.5157 on these reconstructions vs 0.4728
  on real footage (3.21× OOD). Orderings survive; absolute rates do not.
* **The floor's plan goes through the same lossy controller as the models** (§1, the 0.4×
  acceleration). It is a floor on *what this harness executes*, not on the underlying
  kinematics.
* **The distance-keeping half of the LONGITUDINAL family is n = 0** on this scene under
  `--condition empty` — reported per family with the reason, never silently dropped. The
  synthetic-lead conditions (`lead25`/`lead15`/`lead8`/`cutin`) exist in the harness and the
  floor now runs in them unchanged; that is a work item, not an excuse (§9).
* **A floor being un-beaten does NOT prove the models learned nothing** — it proves that *on
  this scene, in this harness, with these metrics*, their advantage over a trivial
  extrapolation is not separated. The lateral family is where they do separate, and it is
  real.

---

## 8 · DELIVERABLE MANIFEST

| artifact | where it lives |
|---|---|
| `KinematicFloorPolicy` + `FLOOR_ARMS` + `set_t0` hook + arm dispatch | `repo:stack/experiments/alpasim-gsplat/closedloop_drive.py` |
| `--floor` flag + per-tick `set_t0` + floor-aware `build_policy` | `repo:stack/experiments/alpasim-gsplat/openloop_drive.py` |
| 29 known-value guards | `repo:stack/tests/test_closedloop_floor.py` |
| machine-readable levels + margins + controls | `repo:taniteval/results/2026-09-04-closedloop-floor/FLOOR_SUMMARY.json` |
| 21 paired `cl_metrics.py` outputs + `RAW_CONTROLS.json` | `repo:…/raw/metrics/` |
| closed-loop floor rollouts (3 arms + 2 controls) | `repo:…/raw/rollouts/` |
| open-loop floor rollouts (3 arms) + sweep summary | `repo:…/raw/openloop/` |
| run scripts + the summarizer | `repo:…/raw/scripts/` |
| Thor run logs | `repo:…/raw/logs/` |
| the same trees on the renderer host | `tanitad-thor-wifi:~/cl_out_floor/`, `~/tanitad_cl_v3/stack/experiments/alpasim-gsplat/{closedloop,openloop}_drive.py` (md5-verified against the repo copies), pre-patch backups at `~/tanitad_cl_v3/*.bak-prefloor` |

**Nothing here lives in only one place.** The only Thor-only files are the `video_*_openloop.json`
regroupings, which contain the same steps as the banked scoring files.

**Code currency, md5, repo == Thor** (presence proves transfer, md5 proves bytes, and neither
proves currency — so both were checked after every ship):

| file | md5 | note |
|---|---|---|
| `closedloop_drive.py` | `a0b048f322c2b00c9a0c8acfdd15b0df` | identical on both; **this is the version that produced every closed-loop number here** |
| `openloop_drive.py` | `d1e4d426fe44ae5943e6f9bb2f296ced` | identical on both. ⚠️ **The open-loop sweep was produced by an earlier build** (`5a594c04fde06bdda74d71b0b1fed5ed`); the only delta is that arm/`--floor` validation was **moved to a preflight** so a bad arm spec costs 2 s instead of a 30 s scene load. The arm dict is assembled by the same code from the same flags, and the three refusals were exercised after the move. Stated rather than claimed away. |

Pre-patch backups of both files are at `tanitad-thor-wifi:~/tanitad_cl_v3/*.bak-prefloor`.

---

## 9 · ESCALATION — what needs a decision, not a note

1. ⛔ **Every closed-loop claim in `MODEL_REGISTRY.md` and the paper needs its floor margin
   beside it.** The two panels re-read here are the only closed-loop panels the programme has;
   both change meaning. A registry note has been added; the paper text has not been reviewed.
2. ⛔ **`+7.1642` must not be quoted again without its decomposition.** It is currently the
   programme's most-quoted closed-loop number and 98.75 % of it is a model losing to a
   straight line.
3. **The floor should become mandatory in the panel scripts**, exactly as `ha0` is mandatory in
   `taniteval/tools/openloop_suite.py`. It costs ~4 GPU-minutes on a run that costs hours.
   Concretely: add `cl_ha0` and `cl_ha0_ext` arms to `run_panel_hq.sh` / `run_panel_v3.sh` and
   make `cl_metrics.py` refuse a panel with no floor arm — the same shape of refusal that
   already guards a mistyped `--tracks`.
4. **Run the floor under the synthetic-lead conditions** (`lead25`/`lead15`/`lead8`/`cutin`) so
   the LONGITUDINAL family's distance-keeping half stops reading `n = 0`. The floor needs no
   change for this; it is one more invocation of the existing script.
5. **n = 1 scene is the binding limit on all of it.** Two renderable NuRec volumes exist and
   one is contaminated. Either more scenes are provisioned, or every closed-loop statement this
   programme makes is a single-clip statement and should say so in its first line.

---

## 10 · COMPUTE NOTE

Run on Thor **concurrently with a sibling's `refav1_arm.py` full-141-clip eval** (PID 2874655),
which was checked first and found to be carrying `OMP_NUM_THREADS=6`; this job carried it too
(14 cores, load average 0.54 at launch). MEASURED from its own log: the sibling's per-clip
rate was a steady **75 s** before launch, rose to **87–104 s** while the floor panel and the
bootstrap batches ran, and **recovered to 75–76 s** afterwards (clips 50–54 at 4064/4139/4214/
4289/4364 s). It was never interrupted and it lost no work.
⛔ No process was killed and `pgrep -f` / `pkill -f` were never used. Device memory was never
probed with `mem_get_info` / `free` / `tegrastats` — the floor allocates nothing on the GPU and
the renderer is the harness's own.

⚠️ **The self-match trap fired once during the shutdown check, and it is worth recording.** A
`ps -eo args | grep -c "[c]losedloop_drive\|…"` reported **2 live processes** after every job
had written its DONE marker. The bracket trick protects the *pattern* but not the rest of the
command line — the same one-liner also ran `md5sum …/closedloop_drive.py`, so the literal token
was present and the shell matched itself. Re-probed with a token disjoint from the search
(`awk '/loop_drive|cl_metric/'` and an opaque `ZZ…ZZ` count), the true answer is **0**. Same
family as the monitor-echo trap in `CLAUDE.md`, in a *shutdown-check* costume: it invents
presence rather than absence, which would have made a clean exit look like a leak.
