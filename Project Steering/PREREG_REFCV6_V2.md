# PRE-REGISTRATION — refcv6 under SPEC v2: a jointly-trained trunk, two perception heads, and a tactical layer that must learn from the scene

**Date:** 2026-09-17 (Europe/Berlin) · **Author:** Master Mind · **Branch:** `agent/arch-inf-20260803`
**Status:** pre-registration. ⛔ **NO ARM LAUNCHED. 0 GPU-hours spent on any arm at the time of
writing.** Every number below is MEASURED on the dev box, or cited to a banked artifact with its
evidence class.

⛔ **This file does NOT amend `PREREG_REFCV6.md`.** That document is dated 2026-09-10 and states its
own premise: *"refcv6 is a CONFIGURATION, not new model code"*. SPEC v2 is new model code — a
jointly-trained ImageNet trunk, a map head, a 3-D box head, a DETR tactical decoder, frame and ego
history, and a new input geometry. Amending a pre-registration for a superseded design would let its
criteria be read as governing arms they were never written for. **`PREREG_REFCV6.md` governs the
design the PI stopped on 2026-09-11 and nothing else.**

**Design source, and the only authority:** `Project Steering/SPEC_REFCV6_V2.md` including its §10
amendments. This document does not re-litigate the design. It **commits the criteria before any
data exists.**

**Hypothesis ids** (to be registered in `GOALS_AND_CLAIMS.md` in the same turn as this file):
`E-REFCV6V2-TRUNK`, `E-REFCV6V2-PERCEP`, `E-REFCV6V2-TACTICAL`, `E-REFCV6V2-NAV`,
`E-REFCV6V2-DRIVE`.

**Estimator, every interval:** **paired episode-cluster bootstrap**
(`taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`). ⛔ `overlapping_holdout_se` is
never called. ⛔⛔ **A separated interval is NECESSARY, NOT SUFFICIENT**: the bootstrap resamples
episodes with the models fixed, so it answers *"would another draw of episodes say this?"* and
never *"would another training run say this?"*. One-seed separation carries a measured **14.3 %**
floor and every claim below is stamped accordingly.

**Every eval is T1 — SELF-ACTION OPEN LOOP.** A planner feeding its own predictor is still open
loop (binding ruling, 2026-09-xx). Nothing here is a closed-loop claim; that needs AlpaSim or a
vehicle.

---

## 0. What is under test

| id | claim | refuted by |
|---|---|---|
| `E-REFCV6V2-TRUNK` | An ImageNet-initialised ResNet trained per the papers' recipe beats our from-scratch trunk on the SAME planning bar | no separated planning gain, or a random-init twin that matches it |
| `E-REFCV6V2-PERCEP` | The map and 3-D box heads reach the trunk and are LEARNED, not memorised priors | either head at or below its marginal control |
| `E-REFCV6V2-TACTICAL` | The tactical layer emits valid behaviours **learned from the scene**, not copied from nav | non-turn classes at chance once nav is zeroed |
| `E-REFCV6V2-NAV` | The plan FOLLOWS the nav command and OBEYS the max-speed input | T-FLIP < 0.50, or max-speed obedience < 0.99 |
| `E-REFCV6V2-DRIVE` | The whole thing beats the do-nothing controls on all four metric families | any family not separated above its control |

⛔ **`E-REFCV6V2-DRIVE` is the only one that is a driving claim.** The other four can all pass while
the model drives worse than doing nothing — `refcv5-v2` is the worked example: it completed 40,284
steps, passed its own machinery, and **lost to the echo control `ha0_ext`, separated**
(`MODEL_REGISTRY` §4.7). ⛔ **No arm is called a success on TRUNK/PERCEP/TACTICAL/NAV alone.**

## 1. The arms — `one_variable`, and what is held constant

| | **ARM A (primary)** | **ARM B (comparison, run 2)** |
|---|---|---|
| **the one variable** | `resnet101` ImageNet `a1_in1k` | `resnet34.a1_in1k` — DiffusionDrive's own |
| params (2-map extractor, MEASURED) | **42.5 M** | **21.3 M** |
| stride-16 / stride-32 channels | 1024 / 2048 | 256 / 512 |
| everything else | identical | identical |

**Held constant across A and B, by assertion in the launch checker:** input geometry
(**256 x 1024** cylindrical, `f_ref` **488.9239852**, HFOV 120.000 deg), K = 3 shared-weight history
frames, the ego-history encoder, nav one-hot, max-speed one-hot, the 117 v0-rolled anchors,
F1–F9, the DETR tactical decoder, both perception heads and their loss weights, the seed, the
corpus, the step count, the batch size, and the window.

⛔ **The channel counts are read from timm's `feature_info`, never from a literal.** ⛔ Nothing may
hard-code 640 / 160 / 40 / 20 — shapes come from the feature maps (`SPEC` §10.1).

⚠️ **A and B differ in MORE than the backbone name and this is declared, not hidden:** a 4x channel
change at both strides moves the fusion and head input widths with it. **A vs B is therefore a
RECIPE comparison, not a controlled single-variable ablation**, and it is reported as such. The
controlled one-variable arms are the knockouts in §4.

## 2. Splits

| split | what | rule |
|---|---|---|
| **fit** | the v7.2 train corpus | every hyper-parameter, every threshold, every pos_weight fitted here |
| **val** | carved from FIT only | model selection, early stopping |
| **test** | the **139 B1 eval clips** | SCORED, never tuned on |

⭐ **Disjointness is MEASURED, not assumed**: the eval-139 set was verified against the v7.2 train
split two independent ways (`raw/v72_disjointness.json`). ⛔ A threshold chosen on the scored split
invalidates the arm; the launch checker refuses it.

## 3. The pipeline is validated on the 139 eval clips BEFORE any pod hour

The PI's instruction, verbatim: *"You validate the pipeline wit the 139 eval clips."*
⛔ **No GPU arm launches until the whole chain runs end to end on those 139** — trunk -> lift ->
map head and box head -> tactical -> planner. Their SAM3 maps (135 of 139) and both cache
geometries are already on the dev box.

## 4. Controls that must read a KNOWN value — and the knockouts

⛔ **A panel without these is refused at start, not discounted afterwards.**

| control | must read | why it exists |
|---|---|---|
| **constant-only** | the no-information value EXACTLY | a head that beats nothing beat nothing |
| **raw-pixel floor** | the AP/error of raw pixels | a learned representation below it added nothing |
| **marginal / base-rate** | closed form, NOT read off the same run | `random_ap_base_rate` |
| **hold-action `ha0_ext`** | the echo control refcv5-v2 lost to | the driving bar |
| **`n` and `d` printed** | — | `n << d` is underpowered BY CONSTRUCTION, not a negative |

**Registered knockouts (controlled, one variable each):** ImageNet init -> random init, same
architecture · K = 3 -> K = 1 · ego history on -> off · nav zeroed (T-ZERO) · max-speed forced ·
map head off · box head off · coupling (1) off.

⭐ **The ImageNet knockout is the one that makes `E-REFCV6V2-TRUNK` falsifiable at all.** A
random-init twin of the SAME architecture is the only control that separates *"pretraining helped"*
from *"this architecture is better"*. It is not optional.

## 5. Perception gates — and the ceiling that is already MEASURED

| gate | criterion | source |
|---|---|---|
| **P-MAP** | 9-class soft CE on SAM3 maps, **seen cells only**; per-class AP above the per-class base rate, paired | `SPEC` §6 |
| **P-BOX-2D** | AP above `random_ap_base_rate` AND above the raw-pixel floor | `box3d_head` |
| **P-BOX-3D** | z and h L1 in **metres**, reported with their own counts; `n_z` printed on every row | verified end to end 2026-09-17 |

⛔ **The perception head hangs on the STRIDE-16 map, never the stride-32 tokens.** An ORACLE built
from the target — which prices the ADDRESS SPACE, not perception — tops out at **AP 0.3341** on
8 x 20 against **0.4713** on 16 x 40. A head on the 160 stride-32 tokens is capped **below our 0.60
bar before training starts**. At 256 x 1024 the grids become 16 x 64 = 1024 and 8 x 32 = 256.

⛔ **LiDAR is NOT a BEV training target** (PI). SAM3 maps only. It may be quoted as an INDEPENDENT
evaluation reference and never as GT.

⭐ **The 3-D label path is wired and MEASURED** (2026-09-17): the join carries `cz`/`h` for
**905,512 / 905,512 agent-frames (100.00 %)**, bottom faces a median **-0.103 m** off the ego
ground plane, and `box3d_set_loss` reads `n["z"] = 22 > 0` on the probe frame against a no-label
control that reads exactly 0.0. ⛔ **That is a LABEL-PATH fact, not a capability claim.**

## 6. Tactical gates — per class, never pooled

**Architecture:** DETR-style behaviour decoder; queries are the 22 vocabulary behaviours plus 8
lateral and 8 longitudinal action queries; keys/values are the **agent slots and BEV tokens**
(*"the scene embeddings, for the agent and the map"* — PI). Multi-label **BCE**, because several
behaviours are valid at once.

| gate | criterion |
|---|---|
| **T-CLASS** | per-token AP above that token's own base rate, paired. ⛔ **NEVER pooled across tokens** |
| **T-FLOOR** | every token reports its `n`. Tokens under the **n = 200** floor are reported and **excluded from any headline** |
| **T-ZERO** | nav zeroed: **non-turn** behaviours must stay above chance |

⭐ **T-ZERO is the gate that decides whether the tactical layer LEARNED anything.** **42.7 %** of
the TURN label's entropy is already in the nav token, and the PI explicitly allowed nav -> turn
derivation (*"its totally fine to porcess the nav command and generate from it the turing
command"*). So turn classes are **excluded from T-ZERO by design, not by convenience**, and T-ZERO
is scored on the speed behaviours, yielding, gap targets and lane keeping — where *"learned from
the scene"* has to show.

⚠️ **The label supervision changed on 2026-09-16 and the consequence must travel with every
tactical number.** Under the PI's absence-as-negative ruling, 54,253 cells convert ignore ->
negative and the head mask goes **17/22 -> 21/22** trainable. But **9 of 22 tokens now sit ON the
`goal_pos_weight` cap of 50** (3 before); uncapped, `LANE_CHANGE_R` implies **303.8**,
`TRAFFIC_LIGHT_REACT` **253.0**, `OVERTAKE_VEHICLE` **227.6**. ⛔ **For those nine the CAP, not the
data, sets the weight, and that sentence must appear wherever their numbers are quoted.**

⚠️ **Measured upper bounds on exposure** (necessary-condition probes): `GAP_TARGET` <= 0.730,
`REACT_ON_ONCOMING` <= 0.516, `YIELD` <= 0.844, `CORRIDOR_OFFSET` <= 0.911 but **uninformative**;
`EVADE_IN_CORRIDOR` and `OVERTAKE_VEHICLE` declared **VOID** for probe insensitivity rather than
quoted. ⛔ **Traffic lights carry the largest UNRESOLVED exposure**: of 867 clips asked the
grounding question AND showing a visible light, **692 (79.8 % [77.0, 82.4])** carry no
traffic-light token. ⛔ **That is a PRESENCE RATE — an upper bound on exposure — NOT a
false-negative rate**: the token is a REACTION, and a present light with a correctly absent
reaction is legitimate. **The traffic-light carve-out is an OPEN PI DECISION** (queue item, default:
no carve-out, the sidecar stays opt-in behind `--cot-negative-sidecar`).

## 7. Nav and max speed — the two obedience tests

| gate | criterion | today |
|---|---|---|
| **T-FLIP** | junction command flipped: the plan follows the FED command on **>= 0.50** of windows | **0.205** |
| **T-COMPLY** | true-minus-shuffled nav compliance **>= 0.38** | — |
| **T-MAXSPEED** | force 30 km/h on windows whose GT exceeds 40 km/h; planned max stays <= the limit on **>= 99 %**, ADE cost reported | — |

**Max speed is an INPUT of four discrete values {30, 50, 100, 120} km/h** (PI; D1 settled). The
training value is the window **containing** the realised maximum — (0,30] -> 30, (30,50] -> 50,
(50,100] -> 100, (100,120] -> 120, clamping above 120 with a count. ⚠️ **Declared ego-future
derived (~2 bits) and stamped in `config.json`** like every other oracle channel. **T-MAXSPEED is
what makes it defensible**; without it the channel is an unearned oracle.

⛔ **The whole strategic layer is DEACTIVATED** for these arms (PI). No route head. The flags remain
but default OFF and the heads are not built.

## 8. The gradient-conflict criteria — RESTATED, because the obvious statistic cannot see the defect

⛔⛔ **This section exists because the criterion we would have written is provably blind.** It is the
*"a check that shares the defect it checks for"* class (`e4af94f`), caught by the agent that built
the instrument, **before any arm ran**.

`cos(g_traj, g_aux)` is **invariant under positive scaling** of either argument — that is exactly
what a cosine quotients out. `E-DEC-18`'s measured failure mechanism was **magnitude** (the aux
gradient 10–30x the planner's). An **angle** statistic can never see a magnitude.

**MEASURED, and verified independently rather than taken on report:** a 30x aux moved the cosine by
**1.49e-08** (rounding), and a **32x** rescale — a power of two, so binary-exact — left the cosine
**bitwise identical**. The agent measured **3.4e-8** on the real model with the same 32x bitwise
result. On the 40-step worked example the *deliberate* 30x defect reaches median cos **-0.0515**
against a `< -0.05` line with **75 %** of steps negative against a `>= 80 %` line — ⛔ **it would
have been called REFUTED.**

⇒ **The criteria as pre-registered here:**

| id | channel | criterion | what it catches |
|---|---|---|---|
| **C-DIR** | `cos` | median `cos < -0.05` on `>= 80 %` of steps | **DIRECTION ONLY** — stated as such |
| **C-MAG** | `ratio = \|g_aux\|/\|g_traj\|` | `ratio > 5` sustained | **magnitude domination** — separated by **21x** on the worked example, never ambiguous |
| **C-CANCEL** | `proj = <g_traj,g_aux>/\|g_traj\|^2` | `proj < -1` | the aux more than cancels the planner's own descent |

⛔ **Nothing is gated on `cos` alone.** All three channels are logged from the **same two gradients
at no extra backward**, and `ratio`/`proj` read **exactly 30x** under the 30x mutation.

⚠️ **Cost, corrected:** the estimate *"one extra backward ... no extra GPU-day"* is out by **~2x**.
MEASURED over 3 interleaved replicates at 3 trunk sizes: `probe` mode **+71.7 % / +83.8 % /
+104.2 %**, rising with trunk size; `subtract` mode **+26.3 % / +53.1 % / +62.2 %** but its planning
side means `L_total - L_aux`, not `L_traj` — stamped `cd_plan_side` on every row so the two can
never be silently compared. `--conflict-every N` divides the cost, but ⚠️ **a sparse median is a
different statistic** and is reported as one.

⛔ **Every conflict number to date is a smoke model on a synthetic corpus.** The instrument is
proven; the readings are **not a result**. Nothing was measured on GPU.

## 9. The driving gate — `E-REFCV6V2-DRIVE`, and the bar that bit its own baseline

⛔ **All four metric families, never pooled, ADE alone incomplete:**

| family | metric | control it must beat, SEPARATED |
|---|---|---|
| longitudinal | along-track error, speed R^2 | `ha0_ext` |
| lateral | **MASKED-curvature MAE**, with the straight-line floor printed beside it | `ha0_ext` |
| tactical | per-token AP, per class | each token's own base rate |
| strategic | ⛔ **NOT SCORED** — the layer is deactivated by PI instruction | — |

⛔ **The bar:** the arm must **BEAT `ha0_ext` SEPARATED** on longitudinal and lateral. Tying is
failing. ⭐ **This bar is known to bite**: run on `refcv4b` the same harness writes **FAIL on both
bars by itself**, and `refcv4b` only *ties* the do-nothing baselines (`os - ha` -0.0021,
`os - ha0_ext` +0.0101, neither separated).

⚠️ **What a clean margin still will NOT establish:** these arms are **oracle-nav**, so neither
carries a production nav command; and any stochastic sampler makes the result one **inference**
draw as well as one training draw.

## 10. Power — computed, not asserted

Every guard reports `MIN_CLIPS_FOR_POWER`, computed by a **sign test at the LOWEST measured win
rate**, not at the observed one. ⚠️ **The worked lesson:** an orientation guard read 14/20 (AUC
0.6770 vs 0.6056, p = 0.058) and was called a defect. It was **power 0.24 at n = 20** — a
POWER failure, not a lift failure. It now scores every scorable clip with a computed threshold.
⛔ An underpowered panel is reported as underpowered; it is never reported as a negative.

## 11. What this pre-registration does NOT claim, and what it OWES

1. ⛔ **No closed-loop claim.** Everything is T1, self-action open loop.
2. ⛔ **No capability claim from a wired seam.** The 3-D label path being MEASURED end to end says
   nothing about how well the box head predicts height. No training run has happened.
3. ⛔ **No corpus-scale claim.** Both cache geometries exist only for the 139 eval clips; the
   corpus rebuild waits on SAM3 production (~22 Sep, ETA 2026-09-22 18:56, 22.87 % at time of
   writing).
4. ⚠️ **OWED — the vertical field.** At 256 x 1024 the VFOV falls **45.456 -> 29.341 deg** and the
   nearest visible road moves **3.14 m -> 5.02 m** ahead: a **1.88 m blind strip** in front of the
   ego (MEASURED camera height 1.3158 m, mount pitch +0.016 deg; flat-ground first order). The
   408 x 1024 alternative keeps **99.6 %** of today's field and is BUILT. **Open PI decision**;
   default is the PI's own instruction, 256 x 1024. ⛔ **No arm's result may be generalised across
   geometries** — the ablation has not been run.
5. ⚠️ **OWED — A vs B is a recipe comparison, not a controlled ablation** (§1).
6. ⚠️ **OWED — the traffic-light carve-out** is an open PI decision (§6).
7. ⚠️ **OWED — the conflict overhead on GPU** is UNVERIFIED and EXTRAPOLATED above 6.95 M trunk
   params (§8).

## 12. Refusals — the launch checker exits non-zero on any of these

1. arms differing in more than `one_variable` **as parsed namespaces**, not as intent;
2. a missing control, or a control that does not read its known value EXACTLY;
3. an unregistered hypothesis id;
4. any hyper-parameter selected on the scored split;
5. an estimator field naming `overlapping_holdout_se`;
6. a hard-coded 640 / 160 / 40 / 20 anywhere in the shape path;
7. `--conflict-detector on` with no live perception weight (it built the detector, printed "ON",
   stamped `enabled=true` and emitted **zero** rows — the `--w-agent` defect verbatim);
8. a trunk-prefix that selects **zero** parameters (`core.encoder.` on `RefCV3Model` vs `encoder.`
   on `RefCModel` — a hard-coded prefix would have logged `NaN` for a whole run while
   `config.json` said "on");
9. a pooled tactical headline, or a headline quoting a token under the n = 200 floor;
10. a quoted `goal_pos_weight` for one of the nine capped tokens without the cap sentence.

⛔ **Guards are proven by MUTATION, never by inspection.** Each check above has a test that
reintroduces the defect and requires the check to go RED.

---

## ⛔ ERRATUM-1 (2026-09-17) — READ IT BEFORE QUOTING ANY NUMBER ABOVE

`Project Steering/PREREG_REFCV6_V2.ERRATUM-1.md` is **BINDING** and corrects two numbers in
this file, both quoted from a session summary instead of the record:

1. §6's *"42.7 % of the TURN label's entropy is already in the nav token"* is **WITHDRAWN —
   UNSUPPORTED**. The measured fact is stronger: `refcv5-v2`'s turn recall **0.475 -> 0.000**
   with nav removed.
2. §5's oracle ceiling at 16x40 is **0.4762**, not 0.4713 (`E-READOUT-CEILING-1` re-read after
   `R-2026-09-08-wpa-mirror`). 8x20 = 0.3341 was correct.

⭐ **No criterion, bar, arm or refusal changes.** See also
`RETRACTION_LOG.md` -> `R-2026-09-17-prereg-v2-unchecked-numbers`.
