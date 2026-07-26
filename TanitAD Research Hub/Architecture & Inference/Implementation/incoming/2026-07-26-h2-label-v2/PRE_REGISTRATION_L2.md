# H2 · L2 — PRE-REGISTRATION of a BEHAVIOURAL sensor-need label

**Written 2026-07-26, BEFORE any `L2` number existed** — before the builder was run, before the DEV
split was scored, before any threshold was chosen. The file timestamp precedes every JSON in this
folder. Nothing here may be edited after the first DEV result; corrections go into
`H2_LABEL_V2_RESULTS.md` as marked amendments.

**Author:** research engineer (H2 label-v2 stream). **CPU only** — no GPU, no training, no model
inference, no pod touched (pod1/pod2 training, pod3 on E1c).

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Why a second label, and what killed the first

`L1_gate` was refuted at the stop gate (INHERITED, `2026-07-25-h2-e0-e1/H2_E0_E1_RESULTS.md`, whose
numbers I re-use but do not re-derive): held-out lift **1.16× [0.9975, 1.3272]**, paired
episode-cluster bootstrap, B = 2000, 2,159 clusters. CI includes 1.

**The diagnosis I am acting on has two parts, and the second one is mine.**

1. *(INHERITED, from the E1 doc)* The label captured **geometric presence**, not decision-relevance.
   Presence is near-ubiquitous; the meaningful event is the ego *having to do something*.

2. *(MEASURED-from-the-E1-artifact, and it is the part nobody has named yet)* **The RESPONSE variable
   was as badly posed as the gate.** `P(response | gate⁻) = 23.15 %` — nearly a quarter of all frames
   satisfy "the ego decelerated ≥ 1 m/s over 4 s". A response that fires on 23 % of ordinary driving
   is not a behaviour, it is a speed fluctuation. **No trigger can show a large lift over a 23 %
   background of routine coasting.** Any redesign that fixes only the trigger repeats the error.

L2 therefore replaces **both halves**: a conflict-based, counterfactual trigger *and* a rare,
attributable behavioural response.

### Three specific mechanisms in `L1_gate` that L2 removes

| # | `L1_gate` | why it destroys signal | `L2` |
|---|---|---|---|
| **M1** | conflict scored against the agent's **REALISED** future track | the agent's realised track already contains **the agent's reaction to the ego**. A conflict that either party resolved leaves no close approach — the label deletes its own positives. (The substrate agent identified this trap for the *ego* and correctly froze the ego's speed; it did **not** apply the same reasoning to the *agent*.) | agent extrapolated at **constant velocity** from `t` — "if nobody reacts" |
| **M2** | ego continued **straight** at constant heading | at a junction the ego's route is a turn; a straight continuation manufactures conflicts with oncoming traffic and misses the actual crossing conflict. Junctions are where the label matters most (E0: junction gate rate 2.66 % vs 1.76 %) | ego follows its **realised PATH** with its speed **frozen** at `v(t)` (path-speed decomposition). Route preserved, speed response removed — the yield is still visible |
| **M3** | trigger = **Euclidean centre distance ≤ d** | 3.0 m between two *centres* is a lane width; the profile is monotone in `d` and crosses 1.0 at ≈3.5–4.0 m precisely because proximity grades into free-flow adjacency | trigger = **required ego deceleration `a_req`**, computed against real oriented footprints. `a_req` *is* the "must act" quantity; distance is only an input to it |

---

## 1. The label — frozen here, character for character

Notation: episode `e`; 10 Hz grid `t`; cameras `X ∈ {cross_left, cross_right}`; agents `a` from
`obstacle.offline`, resampled per-track onto the grid with a **0.5 s max-gap guard** (the documented
trap, `H2_SUBSTRATE §6.6`).

### 1.1 Geometry primitives (all reused, none re-derived)

Projection, frustum membership, the encoder crop and the track resampler are **imported verbatim**
from the validated E0/E1 machinery (`2026-07-25-h2-e0-e1/scripts/_vendored_crux.py` ≡
`scratchpad/crux.py`): `clip_rig`, `project`, `in_frame`, `in_model_crop`, `resample_tracks`. That
code carries **per-clip `(cx, cy)` and per-clip 6-DoF extrinsics on every projection**, which is
mandatory on this two-rig corpus (rig A `cy` ≈ 543 / rig B ≈ 755; a geometric-centre assumption is
~215 px wrong for rig B and would corrupt one rig's labels while looking like noise). It reproduced
the substrate audit's published rates to three digits (`fidelity_check.json`), so it is admissible.

```
CanonicalCrop(t) := the c x c native-pixel box on the clip's own (cx, cy) of camera_front_wide_120fov,
                    c = 2 * r_ftheta(atan(128/266))       -> the encoder's true 51.4 deg field
FrontFull(t)     := the full 120.5 deg front_wide frame
```

### 1.2 Footprints — real boxes, not points

```
ego     : L_e = 4.872 m, W_e = 2.121 m     (MEASURED, calibration/calibration/vehicle_dimensions,
                                            identical on all 100 local rows -> single fleet spec)
agent a : L_a = size_x,  W_a = size_y      (MEASURED, obstacle.offline, per box)
```
Separation between two oriented rectangles is taken along the centre-to-centre line:

```
rho(theta; L, W) = min( (L/2) / |cos theta'| , (W/2) / |sin theta'| )      # exact radial extent
gap(t, h)        = || p_a(t+h) - p_ego(t+h) || - rho_e - rho_a             # <= 0  <=>  contact
```
`theta'` is the bearing of the other object expressed in each box's own frame; agent yaw comes from
its `orientation_*` quaternion, ego yaw from the tangent of its path. **This is a footprint test, not
a centre-distance test — M3 is closed by construction.**

### 1.3 The counterfactual ego — path preserved, speed frozen, braking applied

Let `S(.)` be arc length along the ego's **realised** future path from `t` (extended straight along
the final heading if the realised path is shorter than the horizon needs — which is exactly the case
when the ego stopped). For a candidate constant deceleration `A >= 0`:

```
s_A(h)      = min( v(t)*h - 0.5*A*h^2 ,  v(t)^2 / (2A) )        # arc length travelled, clipped at stop
p_ego(t+h)  = P( S(t) + s_A(h) )                                 # position on the realised path
psi_ego(t+h)= tangent heading at that arc length
```
`A = 0` is the "keeps doing what it is doing" continuation. **The ego's speed response is removed
(so a successful yield still counts) while its route is retained (so junction geometry is right).**

### 1.4 The three interaction measures (agent at constant velocity, h ∈ (0, 4.0 s])

Agent velocity is the central finite difference of its **world-frame** position over ±0.3 s
(≥ 7 consecutive grid samples required, else the agent is dropped from the trigger).

```
MSD(a,t)   = min_h  gap(t,h)  at A = 0                       # minimum separation, mutual CV
TTC(a,t)   = min{ h : gap(t,h) <= 0 } at A = 0, else inf      # time to first contact
a_req(a,t) = min{ A in {0,.5,1,1.5,2,2.5,3,4,5,6,8} : min_h gap(t,h) > 0 }      # 8 = not avoidable by braking
```

> **`a_req` is the label's axis.** It is the deceleration the ego *must* apply, on its own route, to
> keep its footprint clear of an agent that does not react. It is a standard traffic-conflict measure
> (the DRAC family), it is monotone in severity, and it is not a proximity.

### 1.5 The TRIGGER — with the agent-removal counterfactual built in (HOIST, lifted to sensors)

```
a_req_off(X,t) = max over agents a with  proj_front_wide(a,t) not in CanonicalCrop(t)
                                    and  proj_X(a,t) in frame(X)             of a_req(a,t)
a_req_seen(t)  = max over agents a with  proj_front_wide(a,t) in CanonicalCrop(t)   of a_req(a,t)

L2_trigger(X,t) = 1  iff   a_req_off(X,t) >= tau   AND   a_req_seen(t) < tau
```

The second clause **is** the counterfactual-with-agent-removed: *with* the off-front agent the ego
must brake at ≥ τ; **remove every off-front agent and nothing the encoder can see requires it.** The
off-front agent is the *binding* constraint, so a second camera is the only remedy. (`L1_gate`
clause (iv) was a weaker, distance-only form of this.)

### 1.6 The RESPONSE — rare, behavioural, and computed from ego kinematics ALONE

`v(t)` is ego speed at 10 Hz; `alon` is `d|v|/dt` smoothed with a 0.5 s centred moving average
(derived from the 200 Hz `egomotion` speed series, **not** from the `ax` column, whose frame is not
documented in the manifest).

```
R2(t) = 1  iff   v(t) >= 3.0 m/s                                   # actually moving
            AND  mean_{u in [t-0.5s, t]} alon(u) >= -0.5 m/s^2     # NOT already braking: free flow
            AND  min_{h in (0, 4.0 s]} alon(t+h) <= -2.0 m/s^2     # a genuine brake application
```

*"The ego was rolling freely and then had to brake."* This is a **deceleration ONSET**, not a speed
difference — which is the whole point, because `L1`'s `v(t+4) - v(t) <= -1 m/s` fires on 23 % of
frames and cannot separate anything.

> **Stated design degree of freedom, bounded in advance.** If the DEV base rate of `R2` falls outside
> **[1 %, 15 %]**, the two constants (`-0.5`, `-2.0`) may be adjusted **once**, on DEV only, to bring
> it inside that band; the adjustment and the reason are recorded in the results doc. **CONFIRM sees
> exactly one response definition.** No other change to `R2` is permitted.

### 1.7 The full label

```
L2_label(X,t) = L2_trigger(X,t) AND R2(t)     # the high-precision BEHAVIOURAL slice
```
`L2_trigger` is the **training target** (it conditions on no future ego behaviour). `L2_label` is the
evaluation slice and the thing whose positives are behavioural.

### 1.8 Non-circularity — the check that `route_target = _NAV_TO_ROUTE[nav_cmd]` failed

The model receives, at decision time: the **256 px / 51.4° front-camera crop** (3 stacked frames) and
its **own speed** (`action_dim=3`; INHERITED, `MODEL_REGISTRY`, flagship-v1 = `flagship4b-speedjerk-30k`).
`L2_trigger` is a function of (a) 3D agent boxes from `obstacle.offline`, (b) per-camera calibration
and cross-camera frustum membership, (c) the ego's realised future path. **None of the three is a
model input, and no clause is a lookup over one.** The `a_req_seen < tau` clause reads agents that are
*inside* the crop — but from their 3D boxes, which the model does not receive; recovering that term
from pixels is precisely the capability under test, not a leak.

---

## 2. The DEV / CONFIRM split — committed before any L2 number exists

Local chunks with `obstacle.offline` + calibration + `egomotion`: **26** (MEASURED, the intersection
listed in `h2e_probe_split.py` plus the 22-chunk calibration pull).

**Rule, fixed here:**
1. Any chunk used for **threshold selection in any prior H2 analysis** goes to DEV → `{0036, 0170}`
   (the `L1` sweep's two chunks).
2. The remaining 24 are sorted ascending and assigned by index `j`: `j mod 3 == 0` → DEV, else CONFIRM.

```
DEV     (10 chunks) : 0036 0170 0174 0834 0868 0928 1852 1870 2433 2503
CONFIRM (16 chunks) : 0181 0617 0840 0852 0906 0919 0931 1573 1860 1864 1880 1900 2498 2500 2820 2838
```

**Disjoint by chunk, therefore disjoint by clip; asserted in code before any statistic is computed.**
Both sides clear the n ≥ 40 episode-cluster bar by more than an order of magnitude
(≈ 900 / ≈ 1,440 clips ESTIMATED at ~90 clips per chunk).

*Recorded for transparency, because it would otherwise look chosen:* the two chunks that
individually reproduced the old 2.2× headline by chance (E1 §3: 0906 = 2.52×, 0928 = 2.28×) land on
**opposite** sides of this split (0906 → CONFIRM, 0928 → DEV). The rule was written before that was
checked, and it is not being adjusted now.

---

## 3. How τ is chosen — a POWER rule, not an argmax

> **This is the clause that exists because of `L1`.** `L1`'s 3.0 m was *the argmax of a six-point lift
> sweep on 80 clips*. Choosing τ by the argmax of the DEV lift would re-import exactly that error one
> split later.

**τ is selected on DEV by a rule that never reads the lift:**

```
tau* = the SMALLEST tau in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} m/s^2 such that, on DEV,
       the trigger has  n+ >= 200 positive frames  AND  >= 40 trigger-positive episode-clusters
```

Smallest-that-is-powered, not best-performing. If the mechanism is real the lift *increases* with τ,
so this rule deliberately selects the **weakest** admissible operating point — it can only bias the
confirmation **against** L2. If no τ in the grid is powered, the verdict is **UNDERPOWERED** and no
lift is quoted.

**The DEV lift curve is computed and published in full anyway** (§4 rule 3), because a spiky profile
that peaks at the operating point is the signature of noise and a monotone one is the signature of a
mechanism — but it is **descriptive and cannot move τ\***.

---

## 4. CONFIRM — one number, one threshold, one look

At `tau*` **only**, on the 16 CONFIRM chunks, computed once:

```
lift = P(R2 | L2_trigger = 1) / P(R2 | L2_trigger = 0)
```

**Estimator:** **paired episode-cluster bootstrap**, ratio form, `B = 2000`, seed 0, resampling
**clips** with replacement; both arms recomputed inside the same draw so episode difficulty cancels.
Machinery imported from `taniteval/taniteval/ci.py` (`episode_index`, `_draws`) via
`2026-07-25-h2-e0-e1/scripts/h2e_stats.py`. **`overlapping_holdout_se` is used nowhere.**

### Outcomes, committed in advance

| | Condition (all evaluated at `tau*` on CONFIRM) | Consequence |
|---|---|---|
| **GO** | lift CI **excludes 1.0 from above** **AND** point estimate **≥ 1.5×** **AND** n⁺ ≥ 200 frames **AND** ≥ 40 trigger-positive clusters | `L2` is decision-relevant. **Training can start**: build the per-camera Bernoulli head on `L2_trigger`. |
| **BOUND** | CI includes 1.0, **or** point < 1.5×, **or** the power floor is missed | **Report BOUND and STOP. Do not iterate to a third label inside this task.** Two refuted labels is itself the finding: it bounds what the PhysicalAI-AV corpus can support as a sensor-need target. |

**Binding, per the brief:** no re-sweep, no second τ, no alternative response, no rescue. Any
follow-up label is a *new* pre-registration by a *later* task.

### Mandatory reporting, whatever the verdict

1. **The FULL response curve** — lift as a function of τ over the whole grid, on **both** DEV and
   CONFIRM, with CIs and trigger rates. Monotone ⇒ mechanism; spiky-at-τ* ⇒ noise. Descriptive.
2. **Coverage and class balance** — trigger rate, label rate, positive frames, positive episodes,
   per-camera (left/right) split, per-class (`automobile`/`person`/…), junction and lane-change strata.
3. **Speed-matched lift** (Mantel–Haenszel over ego-speed bins). E1 found gate⁺ frames at 5.87 m/s vs
   gate⁻ at 12.31 m/s; the same confound must be excluded here or the result is not readable.
4. **Sensitivities, all descriptive:** pure-CV ego continuation instead of path-preserved (isolates
   M2); agent realised-future instead of agent CV (isolates M1); the residual-only trigger
   (out of the full 120.5° front field, E0's genuine off-front 36.4 %); response constants ±.
5. **Comparison against `L1_gate` on the identical CONFIRM clips** — so the two labels are read on one
   sample, not across two papers.

---

## 5. Discipline binding this run

- Evidence class on **every** number: MEASURED (+path/command) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.
- **CPU only.** No GPU, no training, no model inference, **no pod touched.**
- Read-only access to `C:\Users\Admin\tanitad-data\physicalai\`. **No new download.**
- **Per-clip `(cx, cy)` and per-clip 6-DoF extrinsics mandatory** (two-rig corpus).
- Estimator named on every interval; ≥ 40 episode-clusters; **never `overlapping_holdout_se`**.
- `obstacle.offline` is `scene:obstacles:autolabels:v2` — **machine labels**. Stamp `prov: "autolabel"`,
  never `"human"`. Systematic misses of small/distant agents attenuate any lift and are not excluded here.
- **Parity is untouched** — nothing here re-selects training episodes.
- **No `git add`, no commit, no push.** Files are written into the repo working tree; the orchestrator stages.
