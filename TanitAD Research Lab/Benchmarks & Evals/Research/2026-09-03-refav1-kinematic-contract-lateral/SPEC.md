# SPEC — refav1 kinematic contract, LATERAL: which single channel defect explains the 0.716 m miss

**Package:** `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refav1-kinematic-contract-lateral/`
**Owner:** Benchmarks & Evals FlyWheel · **PI:** Sayed · **Written:** 2026-09-03, dev box, **0 GPU** (CPU arithmetic on banked trajectories only; Thor never contacted).
**Resolves:** the AMENDMENT to `D-REFAV1-STEP1000-READ` / `D-REFAV1-PAIRED-READ-VOID`
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refav1-step1000-read/RESULT.md` §7, item 3).

## 0. The finding under test

On the banked T1 dumps the OPEN-LOOP arm `ol` — the RECORDED `(a, kappa)` replayed through the
adapter's unicycle from the measured `v0` — misses the human's lateral position by a mean
`|y_ol - y_gt|` of **0.716 m** on the 72 windows where the human's own path curves
(`max |y_gt| >= 0.3 m`), which is **WORSE than a straight line's 0.496 m** there; on the 68
human-straight windows 0.069 m vs 0.035 m. The kinematic contract *"the recorded actions must
reproduce GT"* therefore does **not** hold laterally, and every LATERAL row of every refav1 T1
read is suspended until it is resolved.

## 1. The exact path under test (file:line)

| step | file:line | what it does |
|---|---|---|
| action mint | `stack/tanitad/data/refav1_loader.py:258-264` `_kin_actions` | `a = (v[2(j+1)] - v[2j]) / 0.2` from `poses[:,3]`; `kappa = kap[2j]` where `kap = o["actions"][:,0]` (`:252`) |
| arm | `taniteval/tools/refav1_arm.py:606` | `ol = paths_from_controls(act[0], v0, DT, k)` |
| integrator wrapper | `taniteval/tools/refav1_arm.py:383-397` `paths_from_controls` | slices `c[:, :k]`, `v0` as a 1-vector |
| planner wrapper | `stack/tanitad/refs/refa_v1_plan.py:291-301` `unicycle_paths` | `state0 = (0,0,0,v0)`, returns `[..., :2]` |
| integrator | `stack/tanitad/models/kinematic.py:176-184` `rollout_unicycle` | `x += v cos(yaw) dt; y += v sin(yaw) dt; yaw += v*kappa*dt; v += a*dt` — **position advances on the START-of-step yaw, yaw updates after, v last** |
| GT | `taniteval/tools/refav1_arm.py:400-410` `gt_waypoints` -> `stack/tanitad/models/metric_dynamics.py:155-167` `gt_ego_waypoints` -> `:84-94` `ego_delta` | frames `2t+2 .. 2t+2k` step 2, rotated by `-yaw0` about `poses[2t]` — the **t0 HEADING frame**, y positive LEFT |

`v` and `kap` are read from the v2ep dict directly (`refav1_loader.py:252`): `poses[:,3]` and
`actions[:,0]`. The v2ep grid is **10 Hz** (MEASURED below); the cache grid is every 2nd frame.

## 2. Evidence gathered BEFORE this SPEC was committed (declared, so no reading here is post-hoc)

1. **The four numbers reproduce exactly** off the banked dumps (0 GPU): `ol` 0.7156 curved /
   0.0685 straight; the `y == 0` line 0.4960 / 0.0350; `ha` 0.7625 / 0.1203. n = 140 windows,
   72 curved / 68 straight. (Evidence class MEASURED.)
2. **The v2ep grid is 10 Hz**: median `|dp| / v` over an episode = **0.1006 s**.
3. **A CODE READ of the corpus builder** — `stack/tanitad/data/physicalai.py:620-630` `signals_at`:

   ```python
   curv  = col("curvature")
   steer = np.arctan(float(wheelbase) * curv)     # road-wheel angle proxy [rad]
   accel = col("ax")
   actions = np.column_stack([steer, accel])
   ```

   ⇒ `actions[:,0]` is **STEER = atan(L·kappa)**, *not* curvature. The module docstring says so at
   `physicalai.py:12`. `WHEELBASE = 2.9` (`physicalai.py:64`), `DEFAULT_WHEELBASE_MODE = "const2p9"`
   (`physicalai.py:80`).
4. **The implied gain is measured, not assumed**: least-squares `tan(steer) ~ L * kappa_pose` over the
   20 local episodes (`kappa_pose = dyaw/(v*0.1)` from `poses[:,2]`, `v > 3 m/s`) gives
   **L = 2.59 - 3.11, median 2.855**, at r = 0.81 - 1.0000 — i.e. the stored channel is ~**2.9x**
   the true curvature.

⚠️ This makes **H-C1 the pre-registered favourite** and that is declared here rather than
discovered later. Point 3 is a read of the PRODUCER'S SOURCE, not a fit to the scored slice;
point 4 is a fit to the ACTION-vs-POSE channels, not to the lateral error being scored. **No
hypothesis below has been scored against `g` at the time of writing.**

⛔ **This is why the loader's own check missed it.** `refav1_loader.py:17-24` justifies the channel
order with *"correlates r = 0.995 with pose-derived curvature"* — and that is TRUE. **A correlation
is scale-invariant: it cannot see a 2.9x gain.** The channel-ORDER instrument was correct and the
channel-UNIT question was never asked. (Same family as the CLAUDE.md `df` / `step_s` / cylindrical-FOV
traps: a true measurement quoted outside the question it answers.)

## 3. Pre-registered hypotheses and their COMMITTED expected readings

**Scoring.** Per window, over K = 10 steps at dt = 0.2 s, in the t0 heading frame:
`LAT = mean_k |y_arm[k] - y_gt[k]|`, `LON = mean_k |x_arm[k] - x_gt[k]|` — the SAME estimator as the
amendment, so the numbers are directly comparable. Points are **full-set pooled means** over windows
(never `overlapping_holdout_se`); intervals are the **episode-cluster bootstrap**
(`taniteval/ci.py`, 20 episode clusters, 2000 resamples), paired where two hypotheses are compared
on the same windows. n is printed with every cell.
**Curved** = `max_k |y_gt[k]| >= 0.3 m` (72/140); **straight** = the other 68.

**Verdict rule, committed:** a hypothesis RESOLVES the contract iff, in ONE change,
(i) curved LAT **< 0.496 m** (the straight line, the bar the shipped arm fails), AND
(ii) LON over all 140 windows **<= 0.26 m** (the shipped 0.240 + 0.02 tolerance — a repair must not
break the along-track channel), AND (iii) straight-window LAT does not rise above 0.069 m.
A hypothesis that improves LAT while LON rises above 0.26 is REPORTED, not adopted.

| id | the ONE variable changed | committed expected curved LAT (m) | committed expected LON all (m) | what the reading would mean |
|---|---|---|---|---|
| **REF-ol** | none — the shipped path, replayed by me from the v2ep | **0.716** (must match the banked `ol` to < 1e-3) | **0.240** | harness VALID; anything else and the whole panel is void |
| **REF-line** | `a = 0, kappa = 0` at the measured `v0` | **0.496** | — | the bar (and the `ha0` arm of task 3) |
| **REF-floor** | integrate x,y from the TRUE pose yaw + TRUE pose speed (no action channel at all) | **< 0.10** | < 0.10 | the achievable floor. If this is LARGE the miss is NOT in `kappa` and every hypothesis below is mis-aimed |
| **REF-kapose** | `kappa = dyaw_pose / (v * 0.2)` on the cache grid, `a` as shipped | **< 0.15** | <= 0.24 | what the `(a, kappa)` CONTRACT can do with a perfect kappa — the contract's own ceiling |
| **H-A** sign | `kappa -> -kappa` | **> 1.0** (much worse) | ~0.24 | REFUTED if it improves |
| **H-B1** shift -1 | `kappa = kap[2j-2]` (the "closes at t0" half-step) | **0.70 +- 0.06** | ~0.24 | a timing error cannot cancel a gain |
| **H-B2** shift +1 | `kappa = kap[2j+2]` | **0.73 +- 0.06** | ~0.24 | " |
| **H-B3** mid-frame | `kappa = kap[2j+1]` | **0.71 +- 0.03** | ~0.24 | " |
| **H-C0** per-frame/per-second | `kappa -> kap[2j] / 2` (the cache holds every 2nd frame) | **0.30 - 0.45** | ~0.24 | improves, but by the WRONG factor: distinguishes a 2x grid error from a 2.9x gain |
| **H-C1** (favourite) unit | `kappa = tan(kap[2j]) / 2.9` (`physicalai.WHEELBASE`, the `const2p9` build regime) | **0.10 - 0.25**, and < 0.496 | **<= 0.24** | **THE REPAIR.** The stored channel is a steering angle, not a curvature |
| **H-C2** unit, oracle L | `kappa = tan(kap[2j]) / L_hat_ep`, `L_hat_ep` = the per-episode LS fit | **<= H-C1** | <= 0.24 | CEILING ONLY — `L_hat` is fit on this slice; never a shippable repair |
| **H-D1** order | yaw updated BEFORE the position step | **0.70 +- 0.10** | ~0.24 | an integration order cannot fix a gain |
| **H-D2** midpoint | position advanced on `yaw + 0.5*v*kappa*dt` | **0.71 +- 0.06** | ~0.24 | " |
| **H-E1** frame, y sign | GT `y -> -y` | **> 1.0** | — | confirms y is positive LEFT in both GT and the integrator |
| **H-E2** frame, velocity vs heading | GT re-expressed in the t0 **velocity** frame (`yaw0 = atan2(dy, dx)` of the first pose step) | **0.716 +- 0.04** | 0.240 +- 0.04 | the frame is not the cause |
| **H-F** slip | pose yaw change over the 10 steps vs `sum kappa*v*dt` | shipped: **~2.9x over-rotation**; under H-C1: **< 15 %** residual | — | if the repaired yaw AGREES and y still misses, the miss is position integration, not `kappa` |
| **H-C1+D2** (combination, labelled) | H-C1 and H-D2 together | **<= H-C1** | <= 0.24 | how much the midpoint rule is still worth after the unit repair |

## 4. Controls that must read a KNOWN value (the CLAUDE.md probe rule)

* **REF-ol** must reproduce the banked `ol` to < 1e-3 m. A replay that does not is not a replay.
* **REF-floor** must read ~0 on the STRAIGHT windows (a straight human path is exactly what a
  zero-yaw integration reproduces). If it does not, the geometry/frame is wrong before any
  hypothesis is asked.
* **REF-line** must reproduce 0.4960 / 0.0350 from the dumps.
* Every table cell carries its **n**.

## 5. Families

This is a **kinematic-contract probe on recorded actions**, not a policy read. LONGITUDINAL
(along-track) and LATERAL (cross-track, yaw) are both scored and both binding here. TACTICAL and
STRATEGIC are **REFUSED with a reason** — a replay of the human's own recorded actions declares no
manoeuvre and no route, so those families have no arm to score; they are not silently omitted.

## 6. What this SPEC does NOT do

* It does **not** apply a fix. Thor is training on `refav1_loader.py`; the repair ships as a
  **PROPOSED diff** in RESULT.md and a decision for the Master Mind.
* It does **not** re-read any checkpoint. 0 GPU, banked trajectories + v2ep kinematics only.
* It cannot resolve the **per-clip** wheelbase: the dataset's `calibration/vehicle_dimensions` is not
  on this box (`$TANITAD_PAI_WHEELBASE` unset; only the population summary
  `.../2026-07-26-wheelbase-impact/wheelbase_population.json` is in-repo). Which build regime minted
  this corpus (`const2p9` vs `per_clip_v1`) is therefore **UNVERIFIED** here and is called out in
  RESULT.md as an open question for the Data FlyWheel.

## 7. Task-3 companion: the `ha0` control arm (pre-registered acceptance)

`taniteval/tools/refav1_arm.py` gains ONE new arm, `ha0` — a constant-velocity STRAIGHT line at the
measured `v0` (`a = 0, kappa = 0`), tier **T1**, beside `ha` (hold observed `a, kappa`). It is the
STRONGEST trivial baseline and the arm the echo test must actually be run against. Committed
acceptance, on a random-init RefAV1 over 3 synthetic windows:

1. `ha0` is straight (`max |y| < 1e-6`) and constant-speed (`ptp(step length) < 1e-4`) on **every**
   window;
2. `ha` is NOT straight on a window whose observed `kappa != 0`;
3. the new **trivial-profile instrument** reads `straight_frac = 1.0` and `const_speed_frac = 1.0`
   for `ha0`;
4. two arms that are bit-identical are REPORTED as identical by the instrument's `identical_to`
   field (this is exactly what would have caught the void read the night before).

The instrument prints **BEFORE any family row**, and no existing arm's semantics change.
