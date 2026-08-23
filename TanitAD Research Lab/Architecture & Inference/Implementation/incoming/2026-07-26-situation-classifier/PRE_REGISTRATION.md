# SITUATION CLASSIFIER — PRE-REGISTRATION

**Written 2026-07-26, local (Europe/Berlin), BEFORE any classifier weight existed and before any
head was trained.** The file timestamp precedes every checkpoint and every JSON in `artifacts/`.
Nothing here may be edited after the first held-out number; corrections go into
`SITUATION_CLASSIFIER.md` as numbered amendments.

**Author:** research engineer (situation-classifier stream).
**PI instruction being executed:** *"build the model classifying the situations as specified to
request the usage of additional cameras as specified by me"* — the three situations, verbatim:
**lane change · roundabout · intersection**.

**Host:** pod2 (A40). pod1 (training), pod3 and the eval pod are **not touched**.
⚠️ pod2 has a **co-tenant** — the `tblind-ladder` stream's `tb_rung1_v4.py` (PID 258915, 1.4 GiB of
46 GiB, started 20:14 UTC). Declared here rather than discovered later; this stream keeps its own
GPU footprint small and never kills a foreign PID.

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

🔒 **PhysicalAI-AV is gated-confidential.** No clip UUID and no raw content appears anywhere in this
folder; clips carry an integer index only. The UUID map lives on the dev box, outside the repo.

---

## 0. What is new here, and why it is not H2 again

H2 asked *"is an agent that needs braking outside the front crop?"* — a **rare, reactive, braking**
event. Its verdict was `UNDERPOWERED` at 306 held-out positives across **35** clusters, and its
`a_pre` channel was *"braking already under way"*, i.e. **no anticipation**.

This stream changes three things, each of which is a measured, not argued, difference:

| | H2 (`L2_trigger`) | this stream |
|---|---|---|
| **what is labelled** | a braking hazard (`a_req`) | the PI's **three situations**: lane change, roundabout, intersection |
| **the substrate** | 26 `obstacle.offline` chunks ∩ the pod2 episode cache = **582 clips** | the ego trajectory **is already inside every cached episode** (`poses = [x,y,yaw,v]` at 10 Hz, `physicalai.py:487-501`) ⇒ **2,976 clips**, 5.1× |
| **the clock** | two clocks, an integer-lag correlation guard, **10.65 % of clips dropped** | **one clock.** The label is built on the episode's own frame index. There is nothing to align. |
| **the target** | `y(t) = 1` if the hazard is present at `t` | ⭐ `y(t) = 1` iff the situation's **ONSET** falls in `(t, t+3 s]`, **and every frame inside an ongoing situation is masked out of scoring**. Anticipation is enforced by the label, not hoped for in the metric. |

---

## 1. The substrate — fixed here, and PROVEN not assumed

| | |
|---|---|
| universe | the **2,976 cached episodes** of the parity corpus `physicalai-train-e438721ae894` (2,376) + `physicalai-val-0c5f7dac3b11` (600). **Parity untouched — nothing here re-selects episodes.** |
| ego trajectory | `ep["poses"]` = `[x, y, yaw, v]`, float32 `[T,4]`, 10 Hz, index-for-index with `ep["frames_u8"]` `[T,9,256,256]`. MEASURED: T ∈ [188, 205], median 199 (≈20 s). |
| join | `episode_id = int.from_bytes(clip_id.encode()[:4])`; replaying the cache's own recipe (`sorted(clip_id)` → `split_clips(val_frac=0.2, seed=0)`) reproduces **2376/2376** stored train ids. MEASURED, `artifacts/join_proof.json`. |
| agent tracks | `obstacle.offline`, `prov: "autolabel"` (`scene:obstacles:autolabels:v2`) — **machine labels**; systematic misses of small/distant agents attenuate everything derived from them. |
| camera geometry | per-clip `(cx, cy)` + per-clip 6-DoF extrinsics for all 7 cameras, via the validated `crux.py` (`clip_rig`, `project`, `in_frame`, `in_model_crop`). **Mandatory on this two-rig corpus** (rig A `cy`≈543 / rig B ≈755). |
| encoder | `flagship4b-speedjerk-30k` @ 29999 — the **deployed v1** (INHERITED, `MODEL_REGISTRY §1.2`; *not* `flagship4b-phase0-30k`, the no-speed ablation control). **FROZEN**, STRICT-loaded via `tanitad.eval.ckpt_compat.build_world_from_ckpt`. |
| classifier input | ⭐ **the front camera only** (the PI's Step-1 scoping) — the 2048-d `SpatialGridReadout` state the world model itself consumes — plus the ego's own **speed**, which v1 receives by design (`action_dim=3`). |

### 1.1 Two coverage claims in the brief, both CHECKED here rather than inherited

MEASURED from `metadata/feature_presence.parquet` (306,152 clips — the tool that owns the fact):

- **`obstacle.offline` coverage = 97.444 %** (298,326 / 306,152) corpus-wide, **96.900 %** on the
  3,000-clip parity selection. ✅ The brief's INHERITED-UNVERIFIED figure is **confirmed**.
- ⛔ **The brief's *"~500 multi-view clips (front + L + R + rear)"* is WRONG.** All seven cameras are
  present on **306,152 / 306,152 = 100.000 %** of clips, and on **3,000 / 3,000** of the parity
  selection. **Multi-view coverage is not a limit on this study at all.** What *is* limited is the
  local calibration pull and the decoded-video cache — geometry, not availability.

### 1.2 Non-circularity — stated before the fact

The model receives at decision time: the **256 px / 51.4° front-camera crop** (3 stacked frames)
and its own **speed**. The labels are functions of `(x, y, yaw)` — the ego's **world-frame pose
track** — and of 3-D agent boxes + 7-camera calibration. ⚠️ **`yaw` and `x, y` are NOT model
inputs.** v1's action vector is `[steer, accel, speed]` with `steer = atan(WHEELBASE·curvature)`;
**`steer` is never given to any arm in this study**, precisely because curvature is what constructs
two of the three labels. The only ego channels any arm receives are listed in §5.2 and each is
strictly causal.

The `taniteval.blind_baseline` firewall is run on the ego context for each situation and its verdict
(`CIRCULAR` / `LEAKY` / `CLEAN`) is published **whatever it says** (§6.3).

---

## 2. THE THREE SITUATIONS — frozen here, character for character

Implemented once in `scripts/sc_situations.py` and imported everywhere; never re-implemented.
Signals at 10 Hz from `poses`: `psi = unwrap(yaw)`; `omega = dpsi/dt` smoothed with a **0.5 s
centred** moving average; `kappa = omega / max(v, 1.5)` smoothed with a **1.0 s centred** moving
average; `alon = dv/dt` (0.5 s centred).

### 2.1 LANE CHANGE — *lateral displacement of one lane width with near-zero net heading change*

Over a window `[i, i+W]`, `W = 4.0 s`:

```
L1  v[i] >= 8.0 m/s  and  min v over the window >= 6.0 m/s
L2  |psi[i+W] - psi[i]| <= 8 deg                        # NET heading change ~ 0 -> not a turn
L3  lat(u) = -sin psi[i] * dx + cos psi[i] * dy ;  2.4 <= |lat[i+W]| <= 5.5 m     # one lane
L4  |lat[i+W]| >= 0.85 * max_u |lat(u)|                 # a NET offset, not an oscillation
L5  both yaw-rate lobes present: min(sum d+, sum d-) >= 1.5 deg   # the S-shape
L6  sign(first lobe) == sign(lat[i+W])                  # you steer TOWARD the target lane first
```
Overlapping windows are merged into events; **onset = the earliest start index**.

### 2.2 ROUNDABOUT — *SUSTAINED curvature of one sign over a long arc at lower speed*

On maximal same-sign runs of `kappa` with `|kappa| >= 0.020 m⁻¹` (R ≤ 50 m), dropouts below the
deadband shorter than 0.3 s not breaking a run:

```
R1  duration >= 3.0 s                                   # the DURATION half of the discriminator
R2  |Delta psi| over the run >= 90 deg
R3  std(kappa)/mean|kappa| <= 0.5                       # the CONSTANCY half
R4  mean speed in [2.0, 14.0] m/s
R5  bracketing counter-deflection: >= 3 deg of OPPOSITE-sign heading change within 3.0 s
    before the run OR within 3.0 s after it            # the entry/exit deflection
```
`ROUND_core` = R1–R4 only, reported as the pre-registered sensitivity.

> **⭐ How R1–R5 were chosen, declared in full.** The constants were selected **on the TRAIN chunks
> only**, by a rule that reads **geometry and country, never any classifier score**:
> **maximise the number of events subject to DEV counter-clockwise purity ≥ 0.90.**
> This is admissible because the corpus contains **ZERO left-hand-traffic clips** (MEASURED: the 25
> countries in the 3,000-clip selection are all right-hand traffic; `United Kingdom`, `Ireland`,
> `Malta`, `Cyprus` contribute 0 clips), so a **true** roundabout label must be ~100 %
> counter-clockwise. Selected: **22 DEV events at 90.9 % ccw**. The alternatives are published in
> `artifacts/round_sweep.json`.
> **The HELD-OUT counter-clockwise purity is a genuine out-of-sample validation of this label and
> is reported whatever it says.** It was not used to pick anything.

### 2.3 INTERSECTION — *cross traffic is the discriminator; a quantised tight turn is the other half*

⚠️ The existing program proxy `JUNCTION_DEG = 10.0` is a **heading-change threshold** which
`corridor.py` explicitly forbids renaming "intersection". Heading change alone **conflates a curve
with a junction**. The label here therefore has two admissible entrances and **the conflation is
measured, not assumed** (§6.2).

```
TURN half  (ego only), on the same same-sign curvature runs:
  T1  |Delta psi| in [45, 135] deg          # a quantised quarter-turn
  T2  duration <= 6.0 s                     # a junction turn is short
  T3  median turn radius <= 25 m            # tight -> a junction, not a road curve
  T4  mean speed >= 1.0 m/s
  T5  NOT overlapping a ROUNDABOUT event    # roundabouts are subtracted by construction

CROSS half (needs obstacle.offline + calibration):
  X1  >= 1 agent with  speed >= 2.0 m/s,  |cos(psi_a - psi_ego)| <= 0.643  (50-130 deg: PERPENDICULAR),
      whose constant-velocity path CROSSES the ego's realised path within 40 m ahead of the ego
      and within |dt| <= 4 s
  Blocks of >= 1.0 s of X1 are cross events.

INTERSECTION event = TURN event  UNION  CROSS event
```

### 2.4 ⭐ THE TARGET IS ANTICIPATION, AND IT IS ENFORCED BY THE LABEL

For each situation `S` with event onsets `{o}` and `LEAD = 3.0 s`:

```
y_S(t)     = 1  iff  there is an onset o with  t < o <= t + 3.0 s
valid_S(t) = 0  for every frame INSIDE an ongoing S event, and for the last 3.0 s of the episode
```

**Masking the in-progress frames is the whole point.** A head is never scored on a frame in which
the manoeuvre is already happening, so it cannot win by *recognising* what it is already doing —
the only way to score is to see the situation **coming**.

### 2.5 🔴 MINIMUM USEFUL LEAD TIME — registered BEFORE it is measured

**1.0 s.** Justification, fixed here: the gate must fire, the extra camera must wake, and one
encoder pass must complete before the situation begins; the program's own stack already consumes
0.3 s in its 3-frame stack, and a camera that is not already streaming needs a wake-up. **A
classifier whose median lead time at the operating point is below 1.0 s FAILS this study regardless
of its AP.** A high-AP zero-lead trigger is exactly H2's `a_pre` failure one level up.

---

## 3. The split — committed before any label was counted on the held-out side

**Chunk-disjoint ⇒ clip-disjoint ⇒ episode-disjoint.** The 197 chunks of the parity selection are
sorted ascending and assigned by index `j`:

```
j mod 3 == 0  -> TRAIN      (66 chunks,  978 episodes)
otherwise     -> HELD-OUT   (131 chunks, 1,998 episodes)
```

The roundabout constants (§2.2) were selected on **TRAIN chunks only**. **The held-out side is read
exactly once, at the end.** Model selection is **5-fold grouped CV inside TRAIN, grouped by chunk**;
the training script computes **no** held-out metric.

**Encoder-seen confound, declared:** the frozen encoder trained on the 2,376 `train`-cache episodes.
A held-out chunk may still contain encoder-seen clips. The **encoder-unseen sensitivity** (clips
from the `val` cache only) is reported in the results **whatever it says**.

---

## 4. ⭐ THE POWER CEILING AND THE CONTROLS — measured, and each proven able to FAIL

*The control re-adjudication of 2026-07-26 found 4 of 12 controls flipped to FAILED, and one
firewall had an MDE of 0.5555 against a maximum possible effect of 0.2500 — 222 %. It did not pass;
it was never run. Every control below therefore carries its MDE against the effect it exists to
catch.*

| # | control | what it catches | how it can FAIL |
|---|---|---|---|
| **C-POS** ⭐ | **oracle probe** on the *privileged construction summary* (the future 3 s net heading change, net lateral offset, curvature integral) — the very quantities that define the labels | instrument insensitivity | it MUST separate from chance. If a probe on the label's own defining quantity cannot separate at this n, **the study is UNPOWERED and neither Outcome A nor B may be reported.** |
| **C-NEG** | **shuffled features** — the frozen states permuted **across clips**, labels untouched | a leak in the pipeline / an optimistic estimator | it must NOT separate. Its upper 95 % ΔAP bound is the study's empirical **MDE**. |
| **C-CHANCE** | a **constant score** | the wrong above-chance test | AP equals the base rate *inside every bootstrap draw*; the above-chance test is the **paired ΔAP against it**, never "does the AP interval clear the full-sample base rate". |
| **C-BLIND** | `taniteval.blind_baseline.blind_conditioning_baseline` on the ego context, **imported, not re-implemented** | the target being a function of the conditioning | verdict `CIRCULAR` ⇒ the situation is inadmissible and is dropped. `LEAKY` ⇒ every number for it is reported as a **skill over the blind baseline**. |
| **C-FID** | **fidelity check** — the pod-side loader must reproduce the dev-box label counts **exactly** (per situation: events, positive frames, positive clips, per side) | a silent substrate mismatch | any mismatch ⇒ BLOCKED. |
| **C-ALIGN** | the obstacle→episode clock map is fitted on **position** and its residual published **in metres** | a mis-mapped cross-traffic label | median residual > 0.50 m on > 10 % of clips ⇒ the CROSS half is dropped and the intersection label falls back to its TURN half, reported as such. |
| **C-POW** | ⭐ **held-out positive CLUSTERS per situation** | reading a null that is only small-n | **< 40 positive clusters ⇒ that situation is `UNDERPOWERED`** and no verdict is issued for it. Measured and written to disk **before** any classifier score is read. |

---

## 5. The model — and the architecture lesson that is designed in, not re-learnt

### 5.1 ⚠️ Vision enters LOW-RANK. This is not a preference, it is H2's measurement.

H2 MEASURED that on a *working* ego head, **adding the 2048-d image features by concatenation
destroyed it — 3.74× → 1.59× base, from separated to not-separated** — a capacity/swamping
signature at n = 836 positives. Therefore, registered here:

- the frozen 2048-d state is projected to **r dims by a PCA basis fit on TRAIN rows only**,
  `r ∈ {16, 64}` selected by the same grouped CV;
- **the raw-2048-d concatenation is run as an explicit ABLATION arm** (`head_img_ego_concat`) so
  the swamping claim is re-tested rather than assumed.

### 5.2 The head and the arms

```
frozen : frames_u8 [T,9,256,256] --/255--> ViTEncoder(d768,depth12,patch16) --> SpatialGridReadout(4x4,128) --> state[2048]
head   : 8 x ( PCA_r(state)  (+) ego[v/10, alon_pre/2, omega_pre/0.5] )
         -> Linear(d) + positional -> 2 x TransformerEncoderLayer(d, 4 heads, pre-norm, GELU, dropout 0.2)
         -> attention pooling over the 8 steps -> MLP -> 3 INDEPENDENT Bernoulli logits
```
**Three independent Bernoullis, never a softmax over mixed axes** (`H2_SUBSTRATE §C.1`; the 5-way
maneuver-softmax defect that mixed the lateral and longitudinal axes). The window is 8 steps
(0.8 s) ending at `t` and is **strictly causal**; `omega_pre` and `alon_pre` are trailing 0.5 s
means. The encoder receives no gradient.

| arm | inputs | role |
|---|---|---|
| **`head_img_ego`** | PCA-r image + ego | ⭐ **PRIMARY** — the deployable configuration |
| `head_img` | PCA-r image only | is there signal in the **image** at all |
| `head_ego` | ego only | ⭐ **BASELINE (d)** — how much is just ego state, learned |
| `head_img_ego_concat` | raw 2048-d + ego | **the swamping ABLATION** |
| `heur_kin` | not learned | ⭐ **BASELINE (e)** — the simplest one-line kinematic rule per situation (§5.3) |
| `random_at_rate` | none | ⭐ **BASELINE (c)** — matched firing rate, 200 seeds |
| `always` / `never` | none | ⭐ **BASELINES (a) / (b)** |
| `oracle` | the label | the ceiling of the efficiency ledger |

### 5.3 The one-line kinematic rules — fixed here so they cannot be chosen after the fact

```
lane change  : score = |trailing 1.0 s lateral drift rate|      (already easing toward a lane edge)
roundabout   : score = -v                                        (slowing down)
intersection : score = -alon_pre                                 (already decelerating)  [= H2 heur_decel]
```
All three are also reported against every situation, so the choice cannot flatter the head.

### 5.4 The operating point — a rule that never reads a held-out metric

`theta*` per situation is fixed on **TRAIN out-of-fold scores** as the quantile that realises the
budget `B* = 0.05` extra camera-activations per frame (H2's budget, kept for comparability).
The **full budget sweep** `B ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.20}` is published. Baselines are
matched to both the pre-registered budget **and** the head's realised held-out firing rate; matching
a rate reads the held-out **score distribution**, never the held-out **targets**.

---

## 6. What is measured, and how

### 6.1 The estimator — named once

**Paired episode-cluster bootstrap**, `B = 2000`, `seed = 0`, resampling **clips** with replacement,
every arm recomputed inside the same draw. Machinery imported from `taniteval/taniteval/ci.py`
(`episode_index`, `_draws`) via the committed `2026-07-26-h2-classifier/scripts/h2c_stats.py` —
**imported, never re-implemented** (two independent re-implementations produced the nulls that were
overturned on 2026-07-26). ⛔ **`overlapping_holdout_se` is used nowhere.** The resampling unit is
the **episode/clip cluster, never the frame**. `separated` ⇔ the 95 % interval excludes 0.

### 6.2 ⭐ The measurement that discharges *"heading change conflates a curve with a junction"*

On the clips that have `obstacle.offline` **and** calibration, compare
`P(perpendicular cross traffic | TURN event)` against
`P(perpendicular cross traffic | matched-Δψ LARGE-radius curve)` (`|Δψ| ∈ [45°,135°]`,
radius > 40 m — i.e. a road curve of the same heading change). Paired episode-cluster bootstrap.
**If the ratio's CI includes 1.0, the TURN half is NOT a junction detector** and the intersection
label is reported as `CROSS`-only. Committed in advance.

### 6.3 The multi-camera-need measurement — H2's machinery, re-used

For each situation, per camera `X ∈ {cross_left, cross_right, rear_left, rear_right, front_tele}`:
the fraction of situation frames carrying an agent that **projects into `X` but NOT into the
canonical front crop** (`crux.in_frame` / `crux.in_model_crop`, per-clip `(cx,cy)` + per-clip
extrinsics). Reported with a **matched non-situation baseline**, because a camera that always sees
something extra proves nothing. The expectation under test, **not assumed**:
lane change → side/rear; roundabout → left; intersection → left + right.

### 6.4 🔴 The efficiency axis — and what would be DISAPPOINTING, stated BEFORE quoting it

**BOOST_PROGRAM §7.3 is binding.** H2 MEASURED that the naive framing is information-free: against
always-on-7, never-escalating saves 85.7 % and a perfect oracle saves 85.6 % — **the entire span is
0.1 pp**. Therefore this study's efficiency axis is **recall at a fixed camera budget** and
**lead time**, and the compute-saving number is reported only *beside* them.

**Committed in advance — what would be disappointing:**

| axis | DISAPPOINTING | ACCEPTABLE | GOOD |
|---|---|---|---|
| recall @ `B* = 0.05` | ≤ 2× the random-at-matched-rate recall | 2–4× random | ≥ 4× random **and** separated |
| median lead time @ `B*` | < 1.0 s (**FAILS regardless of AP**) | 1.0–1.5 s | ≥ 1.5 s |
| ΔAP vs chance | CI includes 0 | separated | separated **and** ≥ 2× base |
| ΔAP vs `head_ego` | CI includes 0 ⇒ **vision buys nothing; say so plainly** | separated | separated at every budget |

---

## 7. OUTCOMES — both committed in advance, evaluated ONCE

Per situation, on the held-out chunks, at the CV-selected configuration and the TRAIN-fixed `theta*`:

| | condition | consequence |
|---|---|---|
| **A — the classifier works** | `head_img_ego` ΔAP-vs-chance CI excludes 0 from above **AND** ΔAP vs `head_ego` CI excludes 0 from above **AND** median lead time ≥ **1.0 s** **AND** C-POS separated **AND** C-NEG not separated | **A front-camera situation classifier for `S` exists and vision contributes over ego state.** Report the operating point, the budget curve and the per-camera need. This unblocks the selective-activation policy. |
| **A− — the classifier works but vision does not** | ΔAP-vs-chance separated, lead time ≥ 1.0 s, but ΔAP vs `head_ego` CI includes 0 | **The situation is predictable, but not *from the camera* beyond what ego state already gives.** Say it plainly — this is H2's finding one level up, and it must not be dressed up. |
| **B — the representation does not expose it** | no image arm separates from chance, **and** C-POS separates | ⛔ **The frozen v1 front-camera state does not expose this situation's semantics.** The representation is the thing to fix; report the low-rank ablation so the next stream knows whether it was capacity or content. |
| **UNPOWERED** | C-POS does not separate, **or** held-out positive clusters < 40 | Neither A nor B may be reported for that situation. State `UNPOWERED`, report the achievable `n`, and name what would fix it. |

**Binding:** no re-sweep of any §2 constant after a held-out number is seen; no post-hoc arm added
to §5.2; no re-reading of the operating point; no alternative target. Any follow-up is a **new**
pre-registration. A `separated: false` is **UNPOWERED, not refuted** — point estimates move a median
75 % on re-powering, so nothing here may be projected forward.

---

## 8. Discipline binding this run

- Evidence class **and tier** on every number; estimator named on every interval.
- **Bi-directional harness validation** (`e1c_selftest` pattern): a **fidelity check** (C-FID) *and*
  a deliberately failing input (C-NEG shuffled, C-CHANCE constant).
- **Parity is sacred** — nothing here re-selects training episodes.
- pod1 / pod3 / eval-pod untouched; pod2 shared with a declared co-tenant; **kill by explicit PID
  only**; `PYTHONPATH=/workspace/TanitAD/stack`; disk judged by a real `dd`, never `df`; every file
  moved anywhere is **md5-verified**.
- 🔒 No clip UUID, no raw content, in any artifact in this folder.
- **`git add` only.** No commit, no push, no branch switch.
