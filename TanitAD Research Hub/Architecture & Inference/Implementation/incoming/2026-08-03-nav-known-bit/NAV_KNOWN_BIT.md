# Why the arms did not turn when told to — a nav MANIPULATION on the junction scene

**2026-08-03 · Architecture & Inference · branch `agent/arch-inf-20260803`**
Scene `7c72937c-c620-4776-9555-d57222c0081f`, both render conditions, both arms.
Every number below is **MEASURED (mine)** unless labelled otherwise; artifact paths in
the manifest at the end.

---

## The one sentence that changes the picture

**A route command in this system is mostly a BRAKE, not a STEER.** Told `LEFT` instead of
`FOLLOW`, flagship-v1 shortens its 2 s plan by **3.19 m [1.31, 5.62]** (separated) and
moves it sideways by **0.21 m [−0.06, +0.47]** (not separated) — a longitudinal response
**9.06×** the lateral one. Inside the turn itself the lateral authority of the command
**vanishes**: flagship `LEFT − RIGHT` lateral separation is **+0.3160 [−0.6129, +0.7828]**
and REF-C's is **−0.0113 [−0.0551, +0.0913]**, neither separated from zero.

The corollary matters more than the finding. **The missed exit is not a refusal to obey.**
In the turn both arms *do* plan left — because the image shows a left turn — and both fall
about **0.75 m short at 2 s** (flagship in-turn GT 4.445 m vs best-over-nav 3.698 m). The
command cannot close that gap because the command has no lateral authority to spend.

---

## 0 · Controls that license everything below

| control | result | why it is here |
|---|---|---|
| **Determinism** | max \|traj_run1 − traj_run2\| = **exactly 0.0** over 12 ticks × 4 navs | every non-zero delta in this document is the nav input, not sampling |
| **Weights identity** | local `flagship4b-speedjerk-30k` md5 **`b5f07d9e…`** == Thor's `flagship-v1-speedjerk/ckpt.pt`; refc-base **`8f10d6f9…`** == Thor's | the sweep ran on the *same weights* that produced the banked rollouts, not a lookalike |
| **Observation held fixed** | one JPEG stack per tick, reused across all 4 nav values and both arms | this is a MANIPULATION. R-2026-08-03-l established a contingency table cannot separate a user of nav from an echo of nav |
| **JPEG noise floor @2 s** | flagship median **0.3377** (p90 0.723, max 1.259) · REF-C median **0.0134** (p90 0.033) | the honest scale bar — see below |

⭐ **An unplanned corroboration.** The "noise floor" row is a result in its own right:
**flagship-v1's 2 s waypoint moves 25× further than REF-C's under a JPEG q90 round-trip of
its own input** — a change no human can see. That is an independent confirmation, by a
completely different manipulation, of the banked finding that flagship v1's *driven* path
moves 9.05 m under a render change while REF-C's moves 0.43 m (21×).

⚠️ **Read the scale bar next to the effect.** The flagship's whole-clip lateral nav
authority (0.4454 m) is **larger than its median JPEG floor (0.338 m) but smaller than its
p90 (0.723 m)**. Stated plainly: *this model's lateral plan responds about as much to
re-encoding its input as a JPEG as it does to being told which way to go.*

---

## 1 · P1 — which mechanism? Per arm, and they are DIFFERENT

The brief named three candidates. They resolve to different answers for the two arms, and
that is the useful part.

### flagship-v1 — (a) in the channel that matters, and (b) is NOT EVEN ASKABLE

| quantity (empty · objects) | value |
|---|---|
| lateral separation `LEFT − RIGHT`, **all 181 ticks** | **+0.4454 [+0.2201, +0.6583]** SEP · +0.5606 [+0.3041, +0.8163] SEP |
| lateral separation `LEFT − RIGHT`, **in the turn (n=50, 3 clusters)** | **+0.3160 [−0.6129, +0.7828]** not sep · +0.3244 [−0.5249, +0.7643] not sep |
| `dlat(LEFT − FOLLOW)` all ticks | +0.2081 [−0.0585, +0.4733] not sep |
| `dlon(LEFT − FOLLOW)` all ticks | **−3.1935 [−5.6160, −1.3105] SEP** |
| `dlon(RIGHT − FOLLOW)` all ticks | **−3.5134 [−5.0648, −1.8715] SEP** |
| mean \|response\| lateral vs longitudinal | 0.4164 m vs 3.7722 m → **9.06×** (objects 10.46×) |
| sign coherence, told LEFT plans more-left than FOLLOW | all ticks **0.6519 [0.4233, 0.8667]** · in the turn **0.3200 [0.0667, 0.7143]** — ⚠️ **both intervals contain 0.5, so neither the coherence nor the flip is established.** See the caveat below |
| trajectory fan | **NONE.** `tactical_policy.anchor_decoder is None`; `wp_heads = {5,10,15,20}`, four unimodal `Linear(d,2)` |
| lateral reach | GT peak **5.778 m**; best over all navs **4.702 m**; **deficit 1.856 m** at the peak. In-turn shortfall `GT − best` **+0.7474 [−0.2063, +1.3188]** (not separated) |
| route head vs nav | **argmax bijection on 181/181 ticks at each of nav ∈ {0,1,2}**; `conditioning_echo_control` → **`ECHO: True`, `DETERMINISTIC_ECHO: True`** |

⇒ **Verdict: (a) in the lateral channel, and (b) is a category error for this arm.**
The nav input is *not* ignored — it has separated, correctly-signed lateral authority over
the whole clip and very large longitudinal authority. But **where the turn happens, the
lateral effect is not separated from zero**, and the arm has **no candidate set at all**,
so "the turn is outside the fan" cannot be asked of it. Its 2 s waypoint is one linear
regression off one summary token.

⛔ **A claim I withdrew from my own draft, before publishing it.** The in-turn point
estimate is 0.32 — the LEFT command appears to make the plan *less* left than FOLLOW on
68 % of in-turn ticks — and that reads as a spectacular sign inversion. It does not
survive its own interval: **0.3200 [0.0667, 0.7143]**, which contains 0.5. The in-turn
stratum spans only **3 of the 9 clusters**, so every in-turn interval is wide by
construction and a non-separation there is **not** evidence of absence either. What is
established in the turn is the weaker and still decisive statement: **no demonstrated
lateral authority** — *not* demonstrated wrong-signed authority. (The whole-clip claim is
carried by the separated mean difference `LEFT − RIGHT` = +0.4454 [+0.2201, +0.6583], not
by the sign rate, whose whole-clip interval [0.4233, 0.8667] also contains 0.5.)

⚠️ **The honest alternative reading, and why it does not rescue the arm.** In the turn the
`FOLLOW` plan is *already* turning left (≈3.8–4.5 m lateral at 2 s) because the image
shows the bend, so the LEFT command may simply be saturated rather than inert. Both
readings agree on the consequence: **the residual ≈0.75 m of turn is exactly what a route
command would have to supply, and it cannot.**

### refc-base — (c), unambiguously

| quantity | value |
|---|---|
| `route_logits` change across nav ∈ {0,1,2} | **exactly 0.0** (`HEAD_IS_NAV_BLIND`, reproduced) |
| `maneuver_logits` change across nav | **exactly 0.0** — the tactical head is nav-blind too |
| `conditioning_echo_control` | **`ECHO: False`, `DETERMINISTIC_ECHO: False`** |
| lateral separation `LEFT − RIGHT`, all ticks | **+0.0484 [+0.0149, +0.0765]** SEP |
| …**in the turn** | **−0.0113 [−0.0551, +0.0913]** not sep |
| mean \|response\| lat vs lon | 0.0845 m vs 0.4336 m → 5.13× |
| **fan** | 128 anchors, 2 s lateral span **[−12.25, +8.91] m** |
| **is the required turn IN the fan?** | **YES.** Required 2 s lateral **+5.778 m**, well inside +8.91 m |
| GT-nearest anchor | **median rank 2**, in the **top-5 on 76.4 %** of ticks… |
| …but **selected** | only **24.8 %** (follow) / 29.2 % (left) / 24.2 % (right) |
| does nav move the selection? | argmax changes on **13.8 %** (L vs F), **6.6 %** (R vs F), **8.3 %** (L vs R) of ticks |

⇒ **Verdict: (c) — the right trajectory is in the fan, near the top, and loses selection;
and the nav command barely moves the ranking.** REF-C is not out of reach and is not
echoing — it is *selecting badly*, and its route and manoeuvre heads are architecturally
incapable of hearing the command at all.

⚠️ **Guard against an over-generalisation this measurement invites.** R-2026-08-03-l
established `HEAD_IS_NAV_BLIND` for REF-C, and it is exactly reproduced here — but that is
a statement about the **route head**, not about the model. **REF-C's TRAJECTORY is not
nav-blind:** nav enters through `meas_in = [v, nav, keep]` → `measurement` → the decoder
condition (`refc.py`, the `meas_in` line), and the lateral separation `LEFT − RIGHT` =
**+0.0484 [+0.0149, +0.0765]** is separated from zero. Writing "REF-C ignores nav" would be
false. The correct statement is: **REF-C's route and manoeuvre HEADS are nav-blind by
architecture; its PLANNER hears nav faintly** (lateral authority 0.048 m at 2 s, an order
of magnitude below the flagship's 0.445 m — though also on a 25× smaller noise floor, so
the two arms' *signal-to-noise* is closer than the raw magnitudes suggest: 3.6× vs 1.3×).

### The lateral nav effect by plan horizon — and why there is no 4 s row

⛔ **The brief asked for 2 s AND 4 s. There is no 4 s waypoint to sweep in either arm.**
`cfg.tactical_policy.waypoint_horizons = (5, 10, 15, 20)` (flagship) and
`cfg.trajectory.horizons = [5, 10, 15, 20]` (REF-C) are 10 Hz steps, so **both plans end at
2.0 s**. A 4 s lateral displacement is not emitted by the architecture. That is itself
relevant to a strategic claim: *the planner's own horizon is shorter than the manoeuvre it
is being asked to commit to.*

Lateral `LEFT − RIGHT` at 2 s ego-frame, paired episode-cluster bootstrap, empty condition:

| horizon | flagship-v1 | refc-base |
|---|---|---|
| 0.5 s | −0.0051 [−0.0156, +0.0050] not sep | +0.0024 [+0.0016, +0.0033] SEP |
| 1.0 s | +0.0160 [−0.0263, +0.0617] not sep | +0.0079 [+0.0055, +0.0109] SEP |
| 1.5 s | +0.1211 [+0.0078, +0.2317] SEP | +0.0253 [+0.0163, +0.0342] SEP |
| **2.0 s** | **+0.4454 [+0.2201, +0.6583] SEP** | **+0.0484 [+0.0149, +0.0765] SEP** |

Both arms' lateral response grows with horizon, as a route command should. REF-C's is
tiny but **separated at every horizon including 0.5 s**; the flagship's is ~9× larger in
metres but **indistinguishable from zero until 1.5 s** — consistent with a command that
arrives as a late, noisy nudge rather than as a steering intent.

---

## 2 · P2 — the `(cmd, known)` seam, wired and gated

`nav_command_v21` maps **both** `ROUTE_STRAIGHT` and `ROUTE_UNKNOWN` onto `NAV_FOLLOW`, so
at the model input *"the road goes straight"* and *"I could not judge the route"* are
byte-identical. `nav_input_v22` returns the pair; it was wired nowhere.

**Now gated in, OFF by default, at an exact and asserted cost.**

| | |
|---|---|
| flag | `RefCConfig.nav_known_channel: bool = False` — same shape as the `ego_valid_channel` (X15) precedent, one channel over |
| where it enters | `measurement` input, beside the nav one-hot: `d_meas_in = 1 + 4 + ego_valid + nav_known` |
| **parameter cost** | **+128 exactly** (`measurement.0.weight` 128×5 → 128×6; zero bias, zero elsewhere). Verified by construction: 104,191,577 → 104,191,705. *(precedents: a rejected lever cost +272,001; the accepted one +897)* |
| `state_dict` | **key set unchanged**; exactly one shape moves |
| fail-loud, both directions | bit supplied to an ungated model → `ValueError`; gate on with `nav_cmd` but no bit → `ValueError` (defaulting to 1.0 would assert a judgement the labeller never made) |
| `nav_cmd=None` (the TanitEval decode condition) | the `follow` fallback **is** the sentinel, so `known` defaults to **0.0** and needs no argument |
| materiality | the plan **moves** when only the bit moves — asserted, so the seam cannot silently become decoration |
| drivers | `closedloop_drive.py` + `openloop_drive.py` compute `nav_known` on **every** tick and record it with a `nav_known_fed` provenance flag; the bit reaches an arm only if `policy.consumes_nav_known` |
| tests | `stack/tests/test_nav_known_channel.py`, **11 passing**; full suites: `stack` → **1910 passed, 12 skipped, 2 xfailed** (238 s), `taniteval` → **903 passed** (108 s) |

⛔ **The default is NOT flipped.** Turning it on changes what every arm is fed and is the
PI's decision.

⛔ **NOT wired: the flagship's `StrategicPolicy`.** It FiLM-conditions on
`nav_emb(nav_cmd)` alone and has no companion-bit seam. `FlagshipV1Policy.consumes_nav_known`
is `False`, so the driver records the bit as computed-but-not-fed rather than dropping it
silently. ESTIMATED cost to add it symmetrically (a `Linear(1, d_cmd)` added to the nav
embedding, mirroring the existing `ego_emb`): **+256 params** — not implemented here.

⛔ **NOT wired: `stack/scripts/refc_train.py`.** Deliberate — it was modified by a
concurrent stream at session start. See the escalation below.

---

## 3 · P3 — the junction scene's FOUR FAMILIES, per family, with the `known` bit visible

Paired **flagship-v1 − refc-base**, `paired_episode_cluster_bootstrap`, **n = 170 windows
over 9 clusters**. ⚠️ The clusters are **disjoint contiguous segments of ONE clip**, not 40
independent episodes — that is the strongest resampling unit this run affords and it is
stated on every interval. ⛔ `overlapping_holdout_se` is not used anywhere.
Positive delta = **flagship worse** on error metrics, **flagship higher** on accuracies.

### LONGITUDINAL
| metric | empty | objects |
|---|---|---|
| `abs_target_speed_err_ms` | **+1.3645 [+0.9765, +1.8099] SEP** | **+1.4013 [+0.9754, +1.9154] SEP** |
| `along_track_ade_m` | **+1.4082 [+0.6637, +2.1969] SEP** | **+1.4790 [+0.8340, +2.2657] SEP** |
| distance-keeping (headway / time-gap / TTC) | ⛔ **NOT COMPUTABLE, n = 0.** `n_actor_observations = 0` — no annotated agent was observed on any window, and this run was scored without `--tracks`, so a scene-with-no-agents and a missing-input are not distinguishable here. Reported as an **unmeasured family**, not as a scene property. Fix: re-run with `--tracks <sequence_tracks.json>` | same, n = 0 |

### LATERAL
| metric | empty | objects |
|---|---|---|
| `heading_err_rad` | −0.0015 [−0.0613, +0.0659] not sep | +0.0048 [−0.0536, +0.0647] not sep |
| `curvature_err_1pm` | −0.0263 [−0.0830, +0.0090] not sep | −0.0210 [−0.0676, +0.0088] not sep |
| `yawrate_err_rads` | **+0.0115 [+0.0044, +0.0194] SEP** | **+0.0126 [+0.0043, +0.0212] SEP** |
| `lateral_ade_m` | **+0.1358 [+0.0847, +0.1944] SEP** | **+0.1631 [+0.1082, +0.2249] SEP** |
| `cross_track_*` | ⛔ degenerate by construction in open loop (the ego **is** the logged path) — 0.0000 for both arms, do not quote | same |

### TACTICAL
| metric | empty | objects |
|---|---|---|
| `manoeuvre_plan_eq_logged` | +0.0941 [−0.0178, +0.2042] not sep | +0.0471 [−0.0588, +0.1385] not sep |
| `manoeuvre_head_eq_logged` | +0.1941 [−0.0158, +0.4667] not sep | +0.1824 [−0.0473, +0.4564] not sep |
| goal/anchor selection | flagship: **no fan exists** (n/a). REF-C: GT-nearest anchor **median rank 2**, top-5 **76.4 %**, **picked 24.8 %** — measured in §1, not available from `cl_metrics` | |
| `manoeuvre_exec_eq_plan` | ⛔ degenerate in open loop (collapses onto `plan_eq_logged`) | |

### STRATEGIC — and this is where the `known` bit lands
| | flagship-v1 | refc-base |
|---|---|---|
| `route_head_eq_logged` @ **known = 1** (n = 155) | **1.0000 [1.0000, 1.0000]** | **0.6452 [0.3694, 0.9154]** (objects 0.7613 [0.5437, 0.9736]) |
| `route_head_eq_logged` @ **known = 0** | ⛔ **n = 0 — no scorable window** | ⛔ **n = 0** |
| echo control, from the **manipulation** | **`ECHO: True` · `DETERMINISTIC_ECHO: True`** (n_usable 181) | `ECHO: False` · `DETERMINISTIC_ECHO: False` |
| paired flagship − refc | **+0.3548 [+0.0846, +0.6306] SEP (empty)** · +0.2387 [+0.0264, +0.4563] SEP (objects) | |

⛔ **The flagship's 1.0000 and the separated +0.3548 are INADMISSIBLE as strategic skill.**
The echo control fires on the manipulation, `cl_metrics` independently stamps
`CIRCULAR_NAV_ECHO`, and the argmax is a bijection of the input on 181/181 ticks. Only
**REF-C's 0.6452** is an admissible strategic number on this scene.

⭐ **What the `known` bit actually revealed here, and it is not a score change.**
`route_scorable_count` = **155** and `nav_known_1_count` = **155**: the known = 0 stratum
has **zero scorable windows**, so the bit cannot move the strategic *number* on this scene.
What it moves is the **input**: **30.61 % of the `follow` tokens the models were fed on
this scene (15 of 49) were the UNKNOWN sentinel**, and no arm could tell. The bit is an
INPUT lever, not a scoring lever, and its value can only be measured by training with it.

---

## 4 · P4 — how much of the corpus's `follow` supervision is a confession, per speed bin

`nav_command_v21_ex` over **12,504 windows / 500 episodes**, stride 8.
⚠️ **PARITY:** the dev-box caches (`physicalai-train-14231cd29c74`,
`physicalai-val-bb543bdf7836`) resolve to `corpus_key_of = None` — they are **NOT** the
canonical `physicalai-train-e438721ae894`. Admissible as evidence about the **labeller** (a
pure function of poses) and about the **mechanism**; **not** cross-arm comparable, and no
percentage here is a parity number. Nothing was re-selected; the probe only reads.

| speed | n | `follow` % of stratum | **SENTINEL % of `follow`** | turn % | tactical LOSSY % *(INHERITED, `route_gate_speed_probe.json`)* |
|---|---|---|---|---|---|
| v<1 | 2078 | 44.8 | **85.29** | 55.2 | 10.82 |
| 1–3 | 1073 | 35.4 | **96.05** ← peak | 64.6 | **38.24** ← peak |
| 3–6 | 2705 | 27.1 | **87.72** | 72.9 | 34.02 |
| 6–10 | 4896 | 39.8 | **65.28** | 60.2 | 9.88 |
| 10–15 | 1674 | 57.0 | **46.65** | 43.0 | 1.81 |
| 15+ | 78 | 62.8 | **34.69** | 37.2 | 0.00 |
| **all** | **12504** | 40.0 | **70.78 %** (3,537 / 4,997) | | |

⇒ **Yes — the nav collapse tracks the same speed axis as the tactical lossy rate.** Both
peak at **1–3 m/s** and both fall monotonically to 15+ m/s (sentinel 96.05 → 34.69 %;
lossy 38.24 → 0 %). One mechanism — a curvature estimator whose variance grows as 1/v on a
corpus whose median 2 s arc at 1–3 m/s is 4.68 m — degrades the tactical and the strategic
label **together**, in the same regime. A fix aimed at the low-speed arc estimate reaches
both levels; two separate fixes are not needed.

⚠️ **The number is not the corpus agent's 62.4 %, and that is not a contradiction.** Theirs
counts `follow` windows in a different window construction; mine is every stride-8 tick
with `nav_command_v21_ex`. Same mechanism, same order, different denominators — do not
quote them as if one refuted the other.

### ⭐ The structural finding that reframes the fix

**`unknown_sentinel` and `not valid` are the SAME BIT.** `refb_labels.py:621` states
*"`valid` … True iff `route` is a real judgement. ROUTE_UNKNOWN [always comes with
valid=False]"*, and MEASURED here they agree to 3 dp in **all six** speed strata over
12,504 windows (`invalid_pct` == `SENTINEL_pct_of_stratum` in every row).

⇒ **The information was never missing. `nav_command_v21` RETURNS it — as the second element
of its tuple — and every consumer throws it away at the model boundary.** In
`closedloop_drive.nav_from_route`, `valid` selects *which nav to emit* and then never
reaches the policy. So this is not a labelling gap requiring new derivation work; it is a
**plumbing** gap, and the cost of closing it is the +128 params in §2.

---

## 5 · What I did NOT do

1. ⛔ **Did not flip `nav_known_channel` on** anywhere. PI decision.
2. ⛔ **Did not wire `refc_train.py`.** It was modified by a concurrent stream at session
   start; the seam is therefore reachable by the model and the drivers but **no arm can yet
   be trained with it**. This is the one remaining integration step — see below.
3. ⛔ **Did not wire the flagship's `StrategicPolicy`** (ESTIMATED +256 params).
4. ⛔ **Did not re-render or re-run anything on Thor.** Thor was mid-`thor_bench_run.py`
   (`refc_train.py --mode diffusion --steps 1400`, PID 18480); I pulled 78 MB of banked
   frames plus the 1.25 GB refc ckpt over the LAN and ran everything on the dev-box RTX
   4060 instead. **No load was added to a running job.**
5. ⛔ **Did not sweep the first 9 ticks** (k = 9…17): `--save-video-frames` starts writing
   at k = 9, so a 10-frame window is only complete from k = 18. **181 of 190 ticks swept**;
   the 9 dropped are listed in every sweep payload.
6. ⛔ **Did not measure the distance-keeping sub-family** — `n_actor_observations = 0` and
   the run had no `--tracks`, so "no agents" and "no input" are not separable. Named as
   unmeasured, not reported as a scene fact.
7. ⛔ **Did not claim the closed-loop missed exit is explained.** These rollouts are
   **open loop** — the ego is on the logged path. What is measured is the *plan* that
   would have driven it, not the driven path.
8. ⚠️ **Did not eliminate the JPEG confound from the ABSOLUTE re-run values** (median 0.338 m
   for the flagship). It cannot affect the manipulation — every nav arm sees the identical
   JPEG — but the re-run plan is not bit-identical to the banked one, and the reproduction
   gap is published rather than argued away.

---

## 6 · Deliverable manifest

| artifact | where it lives | only one copy? |
|---|---|---|
| `NAV_KNOWN_BIT.md` (this) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-nav-known-bit/` | no |
| `nav_sweep_junction.py` — the manipulation | same dir | no |
| `analyze_nav_sweep.py` — mechanism (a)/(b)/(c) separator | same dir | no |
| `strategic_known_rescore.py` — four families + `known` cut + echo control | same dir | no |
| `nav_sentinel_by_speed.py` — P4 probe | same dir | no |
| `navsweep_analysis.json` — 4 arm×condition analyses | same dir | no |
| `four_families_known_{empty,objects}.json` | same dir | no |
| `nav_sentinel_by_speed.json` | same dir | no |
| `head_invariance_controls.json` — echo + nav-blindness controls | same dir | no |
| `nav_effect_by_horizon.json` — lateral/longitudinal nav effect at 0.5/1/1.5/2 s | same dir | no |
| `in_turn_sign_coherence_ci.json` — the clustered interval that made me withdraw the sign-flip claim | same dir | no |
| `raw/sweep_{flagship,refc}_{empty,objects}.json.gz` — **per-tick, per-nav raw sweep** (181 ticks × 4 navs × 4 arms) | same dir, 5.4 MB total | no |
| `raw/ckpt_md5_local_vs_thor.txt` | same dir | no |
| **`RefCConfig.nav_known_channel` seam** | `repo:stack/tanitad/refs/refc.py` | no |
| **driver wiring + `nav_known` recording** | `repo:stack/experiments/alpasim-gsplat/{closedloop_drive.py,openloop_drive.py}` | no |
| **11 regression tests** | `repo:stack/tests/test_nav_known_channel.py` | no |
| rendered junction frames (2 × 190 JPEG) | `thor:/home/nvidia/ol_out_junction/{empty,objects}/frames` + scratchpad copy | ⚠️ **Thor only** in durable terms — reproducible from the scene; deliberately not committed (78 MB of renders) |
| refc-base ckpt copy | `thor:/home/nvidia/models/refc-base/` + scratchpad | ⚠️ Thor + pods; not a new artifact |

All repo artifacts are **staged, not committed** (agents never commit).

## 7 · ⭐ Integration escalation — one item, and it is small

**`stack/scripts/refc_train.py` must learn `--nav-known-channel` and feed
`refb_labels.nav_input_v22`'s second element into the model.** Until it does, the seam is
reachable by the architecture and by the drivers but **no arm can be trained with it**, so
the +128-param lever cannot be measured. I did not touch that file because a concurrent
stream held it modified at session start. It is ~10 lines: a flag onto the config, and the
`known` scalar carried alongside `nav_cmd` in the window batch.

**Decision the PI owns:** whether to run a matched-capacity A/B with
`nav_known_channel=True`. The pre-registration is trivial because the cost is +128 params
and the control is the same corpus, same seed, same steps.
