# RESULT — REF-C v4: E11′ is implemented, and the anti-echo gate FAILS the arm it must fail

*Architecture & Inference FlyWheel · 2026-09-04 · pre-registration
`PREREG_REFC_V4.md` (this directory), hypotheses **H-ECHO-1..7** in
`Project Steering/GOALS_AND_CLAIMS.md`. Sibling literature package:
`../2026-09-03-refc-ego-inputs-and-anti-echo/RESULT.md`.*

⛔ **NO FULL RUN WAS LAUNCHED.** The PI's sequence is refcv3's eval → video →
push first. §7 carries the ready-to-fire launch line; it waits for the go.

⛔⛔ **SCOPE, FIRST, BECAUSE EVERY NUMBER BELOW DEPENDS ON IT.** The panel runs
on the **validation rig rung** — `--size tiny`, **16,989,725 / 17,015,389**
params — over a **NON-PARITY** 24-episode train slice, scored on **800
windows from 40 episode-disjoint val episodes**. `config.json` stamps `size: "tiny"`,
`rig_rung: true`, and the trainer prints the non-parity warning on every launch.
⇒ **Nothing here is a model claim and nothing here may enter
`MODEL_REGISTRY.md`.** It validates the DESIGN (does the wiring fire?) and the
GATE (does it fail what it must fail?). Precedent and its limits: **H-SCALE-2** —
*"screen architecture on tiny arms, but never quote a tiny-arm number as the
programme's capability."*

---

## 1. The verdict in ten lines

1. **E11′ is implemented and it fires.** The measured `t0` ego state — `v0`,
   `a_long`, `yaw_rate`, `curvature`, plus the X15 presence bit — reaches
   `z_tac`, `ctx → g_str` and `ĝ_tac`. Cost **+25,664 params, identical at the
   tiny and the small rung**, of which 25,536 is the ego block and 128 is the
   measurement encoder's widened input. Registered config delta, derived from
   the dataclasses: exactly `{ego_state_inject, echo_base,
   core.ego_valid_channel}`.
2. **Nothing is differentiated.** `a_long` is the corpus's own `ax`; `curvature`
   is the exact inverse of the corpus's own `atan(2.9·κ)` steer encoding; both
   already sit in every window the contract returns, at the LAST OBSERVED frame.
   No new dataset field, no cache rebuild, no finite difference.
3. **The derivation is validated against an independent corpus field.**
   `yaw_rate` vs `d/dt(unwrap(yaw))` — which comes from the orientation
   quaternion — reads **r = +0.9430** over 40 episodes. That is what confirms
   the **sign** and the **wheelbase inversion**; a sign error reads ≈ −0.94 and
   a wheelbase error ≈ 0, and **both look exactly like a working channel**
   without this probe.
4. ⭐ **The echo is worth 0.4449 m at 2 s from zero pixels** — the same league
   as flagship v1's deployed 0.452 m — against `ha0`'s 0.7040 (val epcache, 40
   ep / 7,963 frames, banked). **So the bar is `ha` and `ha0_ext`, never `ha0`.**
5. ⛔⛔ **The gate FAILS the deliberate-regression arm** (H-ECHO-4 — the
   precondition, and it is now settled). Scene degradation **+0.0000**, ego
   degradation **+1.9884** (separated) ⇒ `ECHOING`. **And it fails it again when
   the arm is re-probed with the REAL frames** (scene **−0.0059**, not
   separated), so the verdict is about what the *weights* learned, not about
   what the harness fed them.
6. ⛔⛔ **THE SHARPEST RESULT OF THE PANEL: GATE 1 ALONE WOULD HAVE CERTIFIED THE
   PURE ECHO.** The image-blind arm **beats `ha` by +9.42 % and `ha0_ext` by
   +4.67 % at 6 s, both with separated CIs.** It fails GATE 1 only because the
   pre-registered **relative margin ≥ 0.10** was committed in advance. ⇒ a
   trained ego-only model learns the corpus's *statistical* dynamics better than
   a fixed constant-`a`/constant-κ integration, so **clearing `ha0_ext` is
   NECESSARY AND NOT SUFFICIENT**, and a criterion of "the CI excludes zero"
   would have passed an arm that never saw a pixel.
7. ⛔ **The STRUCTURAL probe alone would also have passed it** — `READS_BOTH` on
   every ego arm including the blind one, because a live encoder moves the
   output in any model. The intervention measures **wiring**; echoing is about
   **use**. GATE 2b separates them; `assert_not_echoing(gate2b=None)` is now
   labelled `STRUCTURAL_ONLY`.
8. ⭐⭐ **THE WITHHOLDING GUARD IS THE ONLY THING THAT PRODUCED A SCENE-READING
   ARM, AND IT COST EXACTLY WHAT THE LITERATURE SAID IT WOULD.** `ego_dropout
   0.5` (arm C) is the **only** arm with verdict `READS_BOTH` — wrong scene
   costs it **+46.61 %** (separated), wrong ego **+12.02 %** (separated) —
   against `0.0` (arm B) reading `ECHOING`. **B → C is a single lever.** And C
   is simultaneously the **worst** arm on 6 s displacement (17.86 vs B 14.96,
   D 13.27). **PlanTF's OLS −1.48 / R-CLS +5.80 signature, reproduced in our
   architecture:** an ADE-scored gate would have chosen B or D — both echoing —
   and rejected the one arm that reads the road.
9. ⚠️ **E14 made echoing WORSE at this scale, not better — reported against my
   own design.** Adding `echo_base` to the same dropout (C → D, one lever) took
   the scene degradation from **+0.4661 (separated)** to **+0.0318 (not
   separated)** and the verdict from `READS_BOTH` to `ECHOING`. Handing the
   model the extrapolation for free let it lean on the base. H-ECHO-3's
   *honesty* clause holds (`echo_ratio` is legible from step 1); its
   *usefulness* clause is **REFUTED at the rig scale**.
10. **Five corrections land with this package** (§8), including one against a
    claim in my own raise message.

---

## 2. The edge-list delta vs refcv3 — what E11 became

| edge | v3 | v4 |
|---|---|---|
| E1–E10, E12 | — | **UNCHANGED**. In particular the visual trunk still never sees `v0` — the site where the only closed-loop harm was ever measured (CARLA-TransFuser Tab. 10, **DS 56.68 → 45.35**) |
| **E13** `nav_cmd → {z_tac, ctx}` | present | **UNCHANGED — and listed here BECAUSE it is broken.** The shipped refcv3's route head is nav-independent at exactly **+0.0000** (true minus shuffled). It beats the majority rate, so it reads something, but nothing from the route token. ⛔ Deliberately out of scope: repairing two conditioning edges in one arm makes the result non-attributable. `nav_injected` and `ego_injected` are logged per step so an inert edge is visible, not inferred |
| **E11** | ⛔ `v0 ↛ {z_tac, g_str, ĝ_tac}` — REFUSED — implemented as the `tactical_speed_input = False  # goal path stays vision-pure (E11)` line in `_v3_core_base`, and declared in the module docstring | ⭐ **E11′ — `ego_state@t0 → {z_tac, ctx→g_str, ĝ_tac}` is a REQUIRED LIVE edge** |
| **new** | — | ⛔ **E11″ — `future_poses`/`future_actions` ↛ ANY goal node.** The refusal is not deleted, it is **moved to the edge the PI's constraint actually names** |
| **new** | — | ⭐ **E14** — `ĝ_tac = ha0_ext(v0, a0, κ0) + Δ`, `Δ` zero-init, base × `keep` |
| **new** | `core.ego_valid_channel = False` | ⭐ **X15** — `True`, with **ONE** `keep` draw shared by the goal path and the measurement encoder |

**Why the refusal had to move rather than disappear.** A provenance audit whose
only refused edge just became a positive edge has no teeth. And the
mechanically-obvious wrong implementation of this feature —
`a0 = (future[:, 0, 3] − v0)/dt` — would look identical in a config diff,
satisfy any *"does the goal use acceleration?"* check, produce a **better**
number, and be a future read. `test_T10_no_goal_node_reads_the_future` pins it
interventionally.

⭐ **E11′ enters through its own embedding, not through `tactical_speed_input`**
(which stays `False`). So the hierarchy's factored `lat_head_tac`/`lon_head_tac`
see the ego while the **core's own pooled aux heads remain an ego-free control
inside the same arm** — and the registered delta stays three keys instead of
silently widening a second head's input.

---

## 3. The ego channels, and the two probes that validate them

| channel | derivation | differentiated? |
|---|---|---|
| `v0` | `pose_last[:, 3]` | no — corpus value |
| `a_long` | `actions[:, -1, 1]` (the corpus's own `ax`) | **no** — `physicalai.py:13` says the finite difference *"differentiates interpolation noise and lags the true signal"* |
| `curvature` | `tan(actions[:, -1, 0]) / 2.9` | no — an exact inverse of the stored encoding |
| `yaw_rate` | `v0 · curvature` | no — algebraic |
| `keep` | the X15 withholding draw | — |

All five are read at `t0 = W−1`, the **last observed frame**.

**Probe 1 (independent field).** `yaw_rate` vs `d/dt(unwrap(yaw))`:
**r = +0.9430** (mean over 40 episodes). **Probe 2 (expected disagreement).**
`a_long` vs `d/dt(v)`: **r = +0.5624** — deliberately *not* near 1, and that is
the evidence that `ax` is not itself a finite difference.

**Channel statistics** (val epcache, 40 ep / 7,963 frames — these set
`EGO_SCALE_*` so the block enters at ~unit magnitude):
`v0` mean 5.2449 / std **3.6714**, 16.501 % of frames below 0.5 m/s ·
`a_long` mean −0.1607 / std **0.9312**, non-zero on **100.0 %** ·
`yaw_rate` std **0.1593**, |max| 0.8695 · `curvature` std **0.0546**,
non-zero on **99.598 %**.

⚠️ **`curvature` is fed ALONGSIDE `yaw_rate` and the dependence is disclosed.**
`yaw_rate = v0·κ` vanishes at standstill and 16.50 % of frames are near
stationary; a stopped car with the wheel turned has a real path geometry and a
zero yaw rate.

⚠️ **A `per_clip_v1` steer regime is REFUSED with a raise**, not silently
mis-inverted — a wrong `L` scales every curvature by `L_true/2.9` and reads
exactly like a working channel.

---

## 4. X15 — the zero-collision, and why the bit is a precondition

MEASURED (n = **781,635 windows / 4,572 clips**,
`../2026-09-03-ego-zero-collision/`): `v0` is exactly 0.0 on **4.4531 %** of
windows; with `ego_dropout 0.5` the input reads zero on **52.227 %** of TRAIN
samples of which only **4.263 %** are genuine (**22.5 : 1**) — against **100 %**
genuine at eval, a **23.5× shift in the token's meaning**. Consequence,
predicted from source and then measured: `brake_stop` is unreachable at
`v0 = 0` — **0 of 34,807** at eval against 16.0 % brake mass at train.

⇒ admitting four channels whose withheld state is indistinguishable from a real
physical state would multiply the defect by four. `RefCV3Model.__init__`
**raises** if `ego_state_inject` is set without `ego_valid_channel`. The
mechanism differs per channel, and each half is MEASURED
(`raw/census_val40.json`):

| channel | exactly 0.0 | why a withheld 0 is still a lie |
|---|---|---|
| `v0` | **4.4531 %** of windows (a hard atom) | indistinguishable from a genuine standstill; **22.5 : 1** withheld-vs-genuine at train |
| `curvature` | **0.402 %** of frames | indistinguishable from a genuinely straight wheel |
| `a_long` | ⚠️ **0.000 % — never exactly zero** (0 of 7,963) | out of distribution as an exact value, yet at the **centre of the density** (mean −0.1607, std 0.9312): it reads as an ordinary cruise. The model cannot tell "no reading" from "not accelerating" |
| `yaw_rate` | follows `v0 · κ` | vanishes at standstill by construction — which is why `curvature` is fed beside it |

⚠️ **CORRECTED IN THIS PACKAGE.** The first draft of the raise message said
*"`a_long = 0` is the MODE of the distribution"*. The census refutes it —
`a_long` is **non-zero on 100.0 %** of the sample. The X15 argument survives and
is sharper stated per channel; what did not survive was an inherited
plausibility standing where a measurement was available.

⭐ **The 4.4531 % is INHERITED for this package and it decides a design
constraint, so it got a second probe** (`raw/zero_probe_val40.json`, **100 val
episodes / 19,900 frames** — a different split and a different unit):
`v0` exactly 0.0 on **11.00 %** of frames · `curvature` on **0.387 %** ·
`a_long` on **0.000 %** · `yaw_rate` on **11.34 %**. ⚠️ The two `v0` rates
(4.4531 % per train *window* at t0, 11.00 % per val *frame*) are **not the same
quantity** and reporting them as agreeing would be the `df`/`step_s` scope
error. **What reproduces is the load-bearing structural fact — the hard atom at
exactly zero: 13× the next bin in the sibling's train sample, 12.63× here.**

⭐ **One draw, one owner.** `keep` is drawn once per sample in
`RefCV3Model.forward` and handed to both consumers. Pre-multiplying `v0`
outside would be *worse than wrong*: `keep` is derived from `v0 is not None`, so
a pre-zeroed `v0` arrives with `keep = 1` and **asserts** "this really is a
stationary car". The live readout `ego_keep_frac` reads exactly the configured
rate in training and exactly 1.0 in `eval()` — the train/eval asymmetry made
visible in the log rather than inferred.

---

## 5. The tiny-rig panel

**Ladder, one lever per rung** (everything else held: corpus, 24 episodes,
window 8, 8-slot horizons, 2,000 steps, batch 12, lr 1e-4, warmup 250, seed 0,
`--arm hier`, `--size tiny`, diffusion mode, `--u8-batches`, anchors, decoder
geometry, nav source, every loss weight):

### Training summary (2,000 steps each, identical everything but the lever)

| arm | lever added | final loss | `goal2s_err_m` | `echo_ratio` | `ego_keep_frac` | `ego_injected` | wall-clock |
|---|---|---|---|---|---|---|---|
| A `v3` | — (incumbent) | 5.232 | 1.496 | n/a | n/a | 0 | 976 s |
| B `v4_noguard` | `ego_state_inject`, dropout **0.0** | 4.496 | 1.048 | n/a | 1.000 | 1 | 885 s |
| C `v4_drop` | + `ego_dropout 0.5` | 5.021 | 1.796 | n/a | 0.417 | 1 | 866 s |
| D `v4_full` | + `echo_base` (E14) | 5.895 | 2.246 | 6.98507 | 0.417 | 1 | 870 s |
| E `regress` ⛔ | + `--ablate-frames` | 12.879 | 5.012 | 6.10496 | 0.417 | 1 | 856 s |

### The controls, on the shared scored windows (endpoint ADE at each goal slot)

| slot | tau | n windows | `constant_only` | `ha0` | `ha` | `ha0_ext` |
|---|---|---|---|---|---|---|
| 0 | 2.0 s | 800 | 6.3963 | 1.9644 | 0.9820 | 1.3053 |
| 1 | 4.0 s | 707 | 13.2199 | 7.3556 | 5.7505 | 5.9623 |
| 2 | 6.0 s | 618 | 20.3106 | 15.2262 | 14.8758 | 14.1341 |

### GATE 1 — paired episode-cluster bootstrap vs each reference

| arm | slot | arm ADE | vs `ha` (rel, sep, pass) | vs `ha0_ext` (rel, sep, pass) | vs `ha0` | n win / n ep |
|---|---|---|---|---|---|---|
| A `v3` | 0 (2.0 s) | 5.6594 | 0.9820 (-4.7634, sep, fail) | 1.3053 (-3.3357, sep, fail) | 1.9644 (-1.8810, sep, fail) | 800 / 40 |
| A `v3` | 1 (4.0 s) | 12.1966 | 5.7505 (-1.1210, sep, fail) | 5.9623 (-1.0456, sep, fail) | 7.3556 (-0.6581, sep, fail) | 707 / 40 |
| A `v3` | 2 (6.0 s) | 19.3784 | 14.8758 (-0.3027, sep, fail) | 14.1341 (-0.3710, sep, fail) | 15.2262 (-0.2727, sep, fail) | 618 / 40 |
| B `v4_noguard` | 0 (2.0 s) | 1.8826 | 0.9820 (-0.9172, sep, fail) | 1.3053 (-0.4422, sep, fail) | 1.9644 (+0.0417, NOT sep, fail) | 800 / 40 |
| B `v4_noguard` | 1 (4.0 s) | 7.4064 | 5.7505 (-0.2880, sep, fail) | 5.9623 (-0.2422, sep, fail) | 7.3556 (-0.0069, NOT sep, fail) | 707 / 40 |
| B `v4_noguard` | 2 (6.0 s) | 14.9551 | 14.8758 (-0.0053, NOT sep, fail) | 14.1341 (-0.0581, NOT sep, fail) | 15.2262 (+0.0178, NOT sep, fail) | 618 / 40 |
| C `v4_drop` | 0 (2.0 s) | 5.0662 | 0.9820 (-4.1592, sep, fail) | 1.3053 (-2.8812, sep, fail) | 1.9644 (-1.5789, sep, fail) | 800 / 40 |
| C `v4_drop` | 1 (4.0 s) | 11.2304 | 5.7505 (-0.9529, sep, fail) | 5.9623 (-0.8836, sep, fail) | 7.3556 (-0.5268, sep, fail) | 707 / 40 |
| C `v4_drop` | 2 (6.0 s) | 17.8568 | 14.8758 (-0.2004, NOT sep, fail) | 14.1341 (-0.2634, sep, fail) | 15.2262 (-0.1728, sep, fail) | 618 / 40 |
| D `v4_full` | 0 (2.0 s) | 2.6640 | 0.9820 (-1.7129, sep, fail) | 1.3053 (-1.0409, sep, fail) | 1.9644 (-0.3561, sep, fail) | 800 / 40 |
| D `v4_full` | 1 (4.0 s) | 6.5699 | 5.7505 (-0.1425, NOT sep, fail) | 5.9623 (-0.1019, NOT sep, fail) | 7.3556 (+0.1068, NOT sep, fail) | 707 / 40 |
| D `v4_full` | 2 (6.0 s) | 13.2746 | 14.8758 (+0.1076, sep, PASS) | 14.1341 (+0.0608, NOT sep, fail) | 15.2262 (+0.1282, sep, PASS) | 618 / 40 |
| E `regress` ⛔ | 0 (2.0 s) | 1.4613 | 0.9820 (-0.4881, sep, fail) | 1.3053 (-0.1195, sep, fail) | 1.9644 (+0.2561, sep, PASS) | 800 / 40 |
| E `regress` ⛔ | 1 (4.0 s) | 5.7228 | 5.7505 (+0.0048, NOT sep, fail) | 5.9623 (+0.0402, NOT sep, fail) | 7.3556 (+0.2220, sep, PASS) | 707 / 40 |
| E `regress` ⛔ | 2 (6.0 s) | 13.4743 | 14.8758 (+0.0942, sep, fail) | 14.1341 (+0.0467, sep, fail) | 15.2262 (+0.1151, sep, PASS) | 618 / 40 |

### GATE 2 (structural) / GATE 2b (functional) / the verdict

| arm | GATE 2 | GATE 2b: wrong SCENE hurts | GATE 2b: wrong EGO hurts | GATE 2b verdict | gate |
|---|---|---|---|---|---|
| A `v3` | UNPOWERED | +0.2810 (sep) | +0.0000 (NOT sep) | IGNORES_EGO | n/a (this arm has no ego input by constructio) |
| B `v4_noguard` | READS_BOTH | +0.1870 (NOT sep) | +1.4388 (sep) | ECHOING | **RAISED** |
| C `v4_drop` | READS_BOTH | +0.4661 (sep) | +0.1202 (sep) | READS_BOTH | **RAISED** |
| D `v4_full` | READS_BOTH | +0.0318 (NOT sep) | +0.9694 (sep) | ECHOING | **RAISED** |
| E `regress` ⛔ | READS_BOTH | +0.0000 (NOT sep) | +1.9884 (sep) | ECHOING | **RAISED** |
| E `regress` ⛔ — **re-probed with the REAL frames** | — | -0.0059 (NOT sep) | +1.8637 (sep) | ECHOING | |

### The BINDING four families (per slot; ADE alone is an incomplete eval)

| arm | slot | ADE arm/`ha0_ext` | LONG speed err arm/`ha0_ext` | LAT heading | LAT curvature | LAT yaw-rate | LAT cross-track |
|---|---|---|---|---|---|---|---|
| A `v3` | 2.0 s | 5.659 / 1.305 | 2.682 / 1.252 | 0.2004 / 0.0908 | 0.0261 / 0.0160 | 0.1002 / 0.0454 | 1.118 / 0.435 |
| A `v3` | 4.0 s | 12.197 / 5.962 | 2.900 / 2.651 | 0.4046 / 0.2789 | 0.0311 / 0.0291 | 0.1012 / 0.0703 | 4.180 / 2.591 |
| A `v3` | 6.0 s | 19.378 / 14.134 | 3.047 / 3.889 | 0.5943 / 0.5064 | 0.0294 / 0.0502 | 0.0991 / 0.0924 | 8.490 / 6.653 |
| B `v4_noguard` | 2.0 s | 1.883 / 1.305 | 1.436 / 1.252 | 0.1205 / 0.0908 | 0.0211 / 0.0160 | 0.0603 / 0.0454 | 0.779 / 0.435 |
| B `v4_noguard` | 4.0 s | 7.406 / 5.962 | 2.383 / 2.651 | 0.3142 / 0.2789 | 0.0279 / 0.0291 | 0.0786 / 0.0703 | 3.665 / 2.591 |
| B `v4_noguard` | 6.0 s | 14.955 / 14.134 | 2.786 / 3.889 | 0.5104 / 0.5064 | 0.0276 / 0.0502 | 0.0851 / 0.0924 | 7.944 / 6.653 |
| C `v4_drop` | 2.0 s | 5.066 / 1.305 | 2.442 / 1.252 | 0.1334 / 0.0908 | 0.0187 / 0.0160 | 0.0667 / 0.0454 | 0.895 / 0.435 |
| C `v4_drop` | 4.0 s | 11.230 / 5.962 | 2.703 / 2.651 | 0.3195 / 0.2789 | 0.0261 / 0.0291 | 0.0799 / 0.0703 | 3.643 / 2.591 |
| C `v4_drop` | 6.0 s | 17.857 / 14.134 | 2.922 / 3.889 | 0.5068 / 0.5064 | 0.0264 / 0.0502 | 0.0845 / 0.0924 | 7.465 / 6.653 |
| D `v4_full` | 2.0 s | 2.664 / 1.305 | 1.378 / 1.252 | 0.0965 / 0.0908 | 0.0175 / 0.0160 | 0.0482 / 0.0454 | 0.556 / 0.435 |
| D `v4_full` | 4.0 s | 6.570 / 5.962 | 2.099 / 2.651 | 0.2971 / 0.2789 | 0.0299 / 0.0291 | 0.0748 / 0.0703 | 2.717 / 2.591 |
| D `v4_full` | 6.0 s | 13.275 / 14.134 | 2.983 / 3.889 | 0.5321 / 0.5064 | 0.0357 / 0.0502 | 0.0976 / 0.0924 | 6.758 / 6.653 |
| E `regress` ⛔ | 2.0 s | 1.461 / 1.305 | 1.123 / 1.252 | 0.0915 / 0.0908 | 0.0179 / 0.0160 | 0.0458 / 0.0454 | 0.430 / 0.435 |
| E `regress` ⛔ | 4.0 s | 5.723 / 5.962 | 2.409 / 2.651 | 0.2799 / 0.2789 | 0.0328 / 0.0291 | 0.0705 / 0.0703 | 2.574 / 2.591 |
| E `regress` ⛔ | 6.0 s | 13.474 / 14.134 | 3.595 / 3.889 | 0.5086 / 0.5064 | 0.0478 / 0.0502 | 0.0928 / 0.0924 | 6.640 / 6.653 |

⚠️ A goal ROW cannot carry every family, and the ones it cannot are reported with their reason and their n rather than dropped. **headway / TTC**: needs a lead agent in frame, and the `obstacle.offline` join is not built for these episodes. **TACTICAL**: needs the head logits, not a goal row — so it is computed in the next table from `lat_logits_tac` / `lon_logits_tac` against the trainer OWN label function (`refc_tactical.window_factored_labels`). **STRATEGIC**: needs the LAN corridor, which is training-only by E12 and is not emitted on these windows — absent, WITH the reason.


### The TACTICAL family — factored decision accuracy vs the MAJORITY-CLASS control

| arm | axis | n | accuracy | majority-class control | beats it? | per-class recall (support) |
|---|---|---|---|---|---|---|
| A `v3` | lat | 800 | 0.6712 | 0.6762 | **no** | c0 0.970 (0.68) · c1 0.100 (0.15) · c2 0.000 (0.17) |
| A `v3` | lon | 800 | 0.4000 | 0.4675 | **no** | c0 0.207 (0.23) · c1 0.492 (0.47) · c2 0.405 (0.30) |
| B `v4_noguard` | lat | 800 | 0.6637 | 0.6762 | **no** | c0 0.937 (0.68) · c1 0.008 (0.15) · c2 0.165 (0.17) |
| B `v4_noguard` | lon | 800 | 0.4525 | 0.4675 | **no** | c0 0.223 (0.23) · c1 0.604 (0.47) · c2 0.393 (0.30) |
| C `v4_drop` | lat | 800 | 0.6500 | 0.6762 | **no** | c0 0.935 (0.68) · c1 0.033 (0.15) · c2 0.072 (0.17) |
| C `v4_drop` | lon | 800 | 0.4062 | 0.4675 | **no** | c0 0.114 (0.23) · c1 0.529 (0.47) · c2 0.438 (0.30) |
| D `v4_full` | lat | 800 | 0.6587 | 0.6762 | **no** | c0 0.930 (0.68) · c1 0.000 (0.15) · c2 0.173 (0.17) |
| D `v4_full` | lon | 800 | 0.4375 | 0.4675 | **no** | c0 0.228 (0.23) · c1 0.556 (0.47) · c2 0.413 (0.30) |
| E `regress` ⛔ | lat | 800 | 0.6900 | 0.6762 | **yes** | c0 0.961 (0.68) · c1 0.008 (0.15) · c2 0.223 (0.17) |
| E `regress` ⛔ | lon | 800 | 0.4563 | 0.4675 | **no** | c0 0.000 (0.23) · c1 0.947 (0.47) · c2 0.045 (0.30) |

⚠️ **STRATEGIC family, absent WITH its reason:** the STRATEGIC family (g_str bearing/distance vs the LAN corridor) is NOT computed here: the LAN label is training-only (E12) and is not emitted by this dataset, so there is no target on these windows. Reported as absent WITH the reason, per the rule -- not silently dropped.

⚠️ **Selected-vs-executed manoeuvre, filed as a work item:** selected-vs-executed manoeuvre agreement would need the label function applied to the ARM'S OWN 20-step future, and the model emits an 8-slot plan plus 3 goal rows, not a 20-step pose track. Computing it would require re-deriving the classifier on a different support -- a new instrument, filed as a work item rather than approximated here.
### S6 — manoeuvre-stratified ADE (the aggregate hides behavioural collapse)

Census over the scored windows: **straight 235** / **non-straight 565** / unlabelled 0.

| arm | subset | slot | n | arm ADE | `ha0_ext` | `ha` | `ha0` |
|---|---|---|---|---|---|---|---|
| A `v3` | straight | 2.0 s | 235 | 6.722 | 0.599 | 0.529 | 0.530 |
| A `v3` | straight | 4.0 s | 198 | 12.648 | 3.183 | 3.392 | 2.270 |
| A `v3` | straight | 6.0 s | 175 | 18.584 | 9.077 | 9.735 | 6.791 |
| A `v3` | non_straight | 2.0 s | 565 | 5.217 | 1.599 | 1.170 | 2.561 |
| A `v3` | non_straight | 4.0 s | 509 | 12.021 | 7.043 | 6.668 | 9.334 |
| A `v3` | non_straight | 6.0 s | 443 | 19.692 | 16.132 | 16.906 | 18.558 |
| B `v4_noguard` | straight | 2.0 s | 235 | 1.546 | 0.599 | 0.529 | 0.530 |
| B `v4_noguard` | straight | 4.0 s | 198 | 5.646 | 3.183 | 3.392 | 2.270 |
| B `v4_noguard` | straight | 6.0 s | 175 | 11.073 | 9.077 | 9.735 | 6.791 |
| B `v4_noguard` | non_straight | 2.0 s | 565 | 2.022 | 1.599 | 1.170 | 2.561 |
| B `v4_noguard` | non_straight | 4.0 s | 509 | 8.091 | 7.043 | 6.668 | 9.334 |
| B `v4_noguard` | non_straight | 6.0 s | 443 | 16.489 | 16.132 | 16.906 | 18.558 |
| C `v4_drop` | straight | 2.0 s | 235 | 5.892 | 0.599 | 0.529 | 0.530 |
| C `v4_drop` | straight | 4.0 s | 198 | 11.415 | 3.183 | 3.392 | 2.270 |
| C `v4_drop` | straight | 6.0 s | 175 | 16.340 | 9.077 | 9.735 | 6.791 |
| C `v4_drop` | non_straight | 2.0 s | 565 | 4.723 | 1.599 | 1.170 | 2.561 |
| C `v4_drop` | non_straight | 4.0 s | 509 | 11.159 | 7.043 | 6.668 | 9.334 |
| C `v4_drop` | non_straight | 6.0 s | 443 | 18.456 | 16.132 | 16.906 | 18.558 |
| D `v4_full` | straight | 2.0 s | 235 | 2.187 | 0.599 | 0.529 | 0.530 |
| D `v4_full` | straight | 4.0 s | 198 | 4.472 | 3.183 | 3.392 | 2.270 |
| D `v4_full` | straight | 6.0 s | 175 | 8.897 | 9.077 | 9.735 | 6.791 |
| D `v4_full` | non_straight | 2.0 s | 565 | 2.862 | 1.599 | 1.170 | 2.561 |
| D `v4_full` | non_straight | 4.0 s | 509 | 7.386 | 7.043 | 6.668 | 9.334 |
| D `v4_full` | non_straight | 6.0 s | 443 | 15.004 | 16.132 | 16.906 | 18.558 |
| E `regress` ⛔ | straight | 2.0 s | 235 | 0.836 | 0.599 | 0.529 | 0.530 |
| E `regress` ⛔ | straight | 4.0 s | 198 | 3.137 | 3.183 | 3.392 | 2.270 |
| E `regress` ⛔ | straight | 6.0 s | 175 | 8.571 | 9.077 | 9.735 | 6.791 |
| E `regress` ⛔ | non_straight | 2.0 s | 565 | 1.722 | 1.599 | 1.170 | 2.561 |
| E `regress` ⛔ | non_straight | 4.0 s | 509 | 6.729 | 7.043 | 6.668 | 9.334 |
| E `regress` ⛔ | non_straight | 6.0 s | 443 | 15.411 | 16.132 | 16.906 | 18.558 |

### The raw-input floor (ridge from downsampled pixels), with n and d

* **A `v3`** — n_fit 400, n_score 400, **d 192**, lambda 1.0 (inner val of FIT), underpowered=False
  * slot 2.0 s (n=400): floor **4.806** vs arm **5.673**
  * slot 4.0 s (n=358): floor **10.259** vs arm **12.033**
  * slot 6.0 s (n=315): floor **16.210** vs arm **19.038**
* **B `v4_noguard`** — n_fit 400, n_score 400, **d 192**, lambda 1.0 (inner val of FIT), underpowered=False
  * slot 2.0 s (n=400): floor **4.806** vs arm **1.876**
  * slot 4.0 s (n=358): floor **10.259** vs arm **7.100**
  * slot 6.0 s (n=315): floor **16.210** vs arm **14.445**
* **C `v4_drop`** — n_fit 400, n_score 400, **d 192**, lambda 1.0 (inner val of FIT), underpowered=False
  * slot 2.0 s (n=400): floor **4.806** vs arm **5.025**
  * slot 4.0 s (n=358): floor **10.259** vs arm **11.094**
  * slot 6.0 s (n=315): floor **16.210** vs arm **17.634**
* **D `v4_full`** — n_fit 400, n_score 400, **d 192**, lambda 1.0 (inner val of FIT), underpowered=False
  * slot 2.0 s (n=400): floor **4.806** vs arm **2.613**
  * slot 4.0 s (n=358): floor **10.259** vs arm **6.287**
  * slot 6.0 s (n=315): floor **16.210** vs arm **12.458**
* **E `regress` ⛔** — n_fit 400, n_score 400, **d 192**, lambda 1.0 (inner val of FIT), underpowered=False
  * slot 2.0 s (n=400): floor **4.806** vs arm **1.400**
  * slot 4.0 s (n=358): floor **10.259** vs arm **5.547**
  * slot 6.0 s (n=315): floor **16.210** vs arm **12.778**


---

## 6. Reading the panel

### 6.1 ⭐⭐ H-ECHO-4 is settled in the direction that matters: the gate fails the arm it must fail

The deliberate-regression arm is v4 trained with the observed window replaced by
a **scalar constant** — every ego channel present, **zero** scene information, an
echo by construction. The gate **raises** on it.

⛔ **And the structural probe alone would have passed it.** `ego_intervention_test`
reports `READS_BOTH` for it: perturbing `frames` moves every goal node, because
the encoder is a live function of its input in **any** model with a
non-degenerate trunk. **The intervention measures WIRING; echoing is about USE**,
and the two come apart exactly here. That is the same family as the finding
`goal_provenance` was built on — *"a detached wire carries the full signal and
zero gradient"* — read in the converse: **a live wire can carry zero useful
signal.**

⚠️ **The obvious objection, answered by measurement.** Scoring the regress arm on
its own constant frames makes `scene degradation == 0` true *by construction* — a
derangement of a constant is a constant. So the arm is probed **a second time
with the REAL frames**; the table above reports both. The real-frame row is the
one that says what the **weights** learned.

### 6.2 ⛔⛔ GATE 1 IS NECESSARY AND NOT SUFFICIENT — the blind arm clears both kinematic controls

At the **6 s primary slot** the deliberate-regression arm — trained on a
**constant image**, i.e. zero scene information — reads:

| reference | ref ADE | arm ADE | relative margin | CI separated? | required margin | passes? |
|---|---|---|---|---|---|---|
| `ha` | 14.8758 | **13.4743** | **+9.42 %** | **yes** | 0.10 | no (margin) |
| `ha0_ext` | 14.1341 | **13.4743** | **+4.67 %** | **yes** | 0.10 | no (margin) |
| `ha0` | 15.2262 | **13.4743** | **+11.51 %** | **yes** | — | — |

⇒ **an arm that never saw a pixel beats every kinematic control on the primary
slot with separated intervals.** It is failed only because the pre-registration
committed a **relative margin of 0.10** in advance. **A criterion of "the CI
excludes zero" — the obvious one — would have certified a pure echo.**

**Why, mechanically:** `ha`, `ha0` and `ha0_ext` are *fixed* integrations
(constant `a`, constant κ). A **trained** model handed the same four scalars
learns the corpus's *conditional* dynamics — how speed and curvature actually
evolve given this state — and that is strictly better than any constant-parameter
extrapolation. ⇒ **the kinematic controls are a floor on the echo, not a ceiling
on it**, and beating them is necessary, never sufficient. This is the panel's
most transferable finding and it applies to every future arm that consumes ego
state.

⚠️ It also **sharpens `CLAUDE.md`'s own framing**: the file says a v4 that merely
beats `ha0` proves nothing. Measured here, a v4 that merely beats **`ha0_ext`**
proves nothing either.

### 6.3 ⭐⭐ The withholding guard is the only thing that produced a scene-reading arm

`B → C` is a **single lever** (`ego_dropout` 0.0 → 0.5). `C → D` is a single
lever (`echo_base` off → on).

| arm | `ego_dropout` | `echo_base` | wrong SCENE hurts | wrong EGO hurts | verdict |
|---|---|---|---|---|---|
| B | **0.0** | off | +0.1870 (**not** sep) | +1.4388 (sep) | ⛔ `ECHOING` |
| **C** | **0.5** | off | **+0.4661 (sep)** | **+0.1202 (sep)** | ⭐ **`READS_BOTH`** |
| D | 0.5 | **on** | +0.0318 (**not** sep) | +0.9694 (sep) | ⛔ `ECHOING` |
| E | 0.5 | on + **no image** | +0.0000 (not sep) | +1.9884 (sep) | ⛔ `ECHOING` |

⭐ **PlanTF's and DRAMA's mechanism reproduces in our architecture** — and so
does its published *cost*: **C is simultaneously the only `READS_BOTH` arm and
the WORST arm on 6 s displacement** (17.86 against B's 14.96 and D's 13.27).
That is PlanTF's **OLS −1.48 while R-CLS +5.80** in our units. ⛔ **An
ADE-scored gate would have selected B or D — both echoing — and rejected the one
arm that reads the road.** The prereg's refusal to score on ADE was not
bureaucratic; it decides this panel.

### 6.4 ⚠️ E14 made echoing WORSE at this scale — reported against my own design

`C → D` adds only `echo_base`, and it moves the scene degradation from
**+0.4661 (separated)** to **+0.0318 (not separated)**, flipping the verdict
`READS_BOTH → ECHOING`. Handing the model the kinematic extrapolation for free
gave it something to lean on.

⇒ **H-ECHO-3's usefulness clause is REFUTED at the rig scale**; its honesty
clause stands (`echo_ratio` is emitted from step 1 and reads 6.99 at the end of
D's run, so the residual is not inert — the model is *adding* to the base rather
than ignoring it, and still not reading the scene). ⚠️ **Not generalised beyond
the rig**: 2,000 steps on 24 episodes with a zero-init residual is exactly the
regime where the free base is most attractive. **The pre-registered follow-up is
therefore `--ego-dropout 0.5` WITHOUT `--echo-base` as the primary arm**, with
`--echo-base` as the second arm rather than the default — a change to the launch
line this package produced, driven by its own measurement.

### 6.5 The controls read the values they must

* **`constant_only`** (predict the corpus mean) is far worse than every arm at
  every slot (6.40 / 13.22 / 20.31 m) — the panel measures something.
* ⭐ **Arm A's ego degradation is EXACTLY `+0.0000`.** The v3 control has no ego
  input, so deranging ego cannot move it — the no-information value, read
  exactly, which is what says the probe measures use and not noise. Its **scene**
  degradation is **+0.2810 (separated)**: A uses the scene because it has nothing
  else. The gate is declared **not applicable** to A rather than "failed" —
  `IGNORES_EGO` is that arm's definition, and remapping the verdict to make it
  pass would have been worse than reporting it.
* **The raw-pixel floor** carries its printed `n` and `d`; `n_pix = 8` puts
  `d = 192` below `n_fit = 400`, so the ridge is **powered** rather than
  underpowered by construction.
* ⚠️ **The TACTICAL family says the same thing a third time, and it is the
  cleanest statement of it.** **No arm beats the majority-class control on
  either axis** (lat: every arm 0.650–0.690 against a control of **0.6762**;
  lon: 0.400–0.456 against **0.4675**). The single nominal "yes" is the
  **image-blind** arm on lat (0.6900), which is +1.4 points and reads as noise —
  and its lon head predicts one class on **94.7 %** of windows, i.e. it *is*
  close to the constant predictor. ⇒ at 2,000 rig steps **no arm has a working
  tactical decision head**, and the blind arm is not distinguishable from the
  sighted ones on decisions either. **This is the reading the majority-class
  control exists to make possible**: without it, "lat accuracy 0.69" looks like
  competence.
* ⚠️ Note also the **lateral class collapse** visible in the per-class column:
  every arm's `c0` (keep) recall is 0.93–0.97 while `c1`/`c2` are 0.000–0.223 —
  the 3-way head has retreated to the majority class. That is the same shape as
  the `brake_stop` unreachability X15 exists to remove, on the lateral axis, and
  it is *not* something ADE can see.

### 6.6 ⚠️ What the rig does NOT settle

**H-ECHO-1 stays OPEN.** 2,000 steps on 24 non-parity episodes is a wiring and
gate screen, an order of magnitude short of the 40,284-step launch config. What
the rig *did* settle is the precondition (**H-ECHO-4**) and the mechanism
ranking (**§6.3**), which is exactly what `TanitAD_ValidateAIDesign` §2 asks a
tiny rig to do before a design earns compute.

### 6.7 The suite, PAIRED — because a clone failure must not read as a regression

`raw/pytest_scoped.txt`. Scope: every test file under `stack/tests` that imports
`refc_v3` / `refc` / `refc_v3_train` / `echo_gate` / `goal_provenance` — 34
files, i.e. everything the change can reach.

| tree | passed | failed | skipped |
|---|---|---|---|
| **WITH the v4 change** | **608** | 5 | 22 |
| **BASELINE** (`git show HEAD:` for the 3 modified files; the 2 new files removed) | 575 | 5 | 22 |

⇒ **+33 passing tests and IDENTICAL failure sets.** The 5 are pre-existing
clone-environment artefacts, reproduced on the baseline: four in
`test_openloop_suite.py` read repo artifacts that live outside `stack/` and are
absent from this off-Drive clone, and `test_refc_select.py`'s trainer test dies
inside `torch.backends.cudnn` under `CUDA_VISIBLE_DEVICES=""`.

⚠️ **One honest note, kept rather than deleted.** An earlier capture recorded a
**6th** failure. It was a **race** — the suite read `refc_v3.py` while a
docstring edit was being written. It passes in isolation and does not appear in
the clean run. *A suite run concurrent with an edit is not evidence.*

---

## 7. The launch line (⛔ DO NOT FIRE)

⭐⭐ **It is the INCUMBENT'S OWN COMMAND with three flags added** — read from
`HF_CARD_tanitad-refc-v3.md`'s *"exact command (read from live `ps` on the
training pod)"*: same rung (`--size base`, not `small`), same corpus
(`--v2-cache`), same geometry, same nav source, same optimiser, same seed, same
steps. Anything else makes refcv4-vs-refcv3 a bundle instead of a lever. ⚠️ An
earlier draft named `--size small --data-root`; that would have been the C6
confound.

**Preflight — VERIFIED GREEN at the real rung, on CPU, 2026-09-04:**
`params total 107,053,435` with `ego_inject 25,536` · freeze-history
`pass: True` · **`E11' OK: ego reaches the goal path, frames still move
g_str/z_tac, keep=0 withholds cleanly`** · one end-to-end loss step with
`echo_ratio 0.0`, `echo_base_absmean 3.111`, `ego_keep_frac 0.5` · `✅ PASS`.

```
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py --preflight \
  --arm hier --size base --image-hw 256 640 --synth-episodes 4 \
  --out /workspace/experiments/refcv4-pf \
  --ego-state-inject --echo-base --ego-dropout 0.5 --device cpu

OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack \
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --out /workspace/experiments/refcv4-b1-v72-40k \
  --ego-state-inject --ego-dropout 0.5
```

⭐⭐ **`--echo-base` HAS BEEN REMOVED FROM THE PRIMARY ARM BY THIS PANEL'S OWN
MEASUREMENT** (§6.4). At the rig, `C → D` — a single lever — moved the scene
degradation from **+0.4661 (separated)** to **+0.0318 (not separated)** and the
verdict from `READS_BOTH` to `ECHOING`. The primary arm is therefore
`--ego-state-inject --ego-dropout 0.5`; **`--echo-base` becomes the SECOND arm**,
run only if the primary clears its gate, so E14 is tested rather than assumed:

```
  ... --ego-state-inject --ego-dropout 0.5 --echo-base \
  --out /workspace/experiments/refcv4-b1-v72-40k-echobase
```

⚠️ The rig is not the full run and the reversal is not generalised — but *"the
tiny rig changed the launch line"* is precisely what
`TanitAD_ValidateAIDesign` §2 exists to produce, and it would have been the
default without it.

Expected parameter count **107,058,565** = the incumbent's pinned 107,032,901 +
**25,664**. ⭐ The v2 cache already carries what E11′ needs —
`v2_dataset.py` stores `.actions` and `_contract.py:132` returns
`ep.actions[t:t+w]`, the same field the raw path uses — so **no cache rebuild**;
and a cache lacking it makes `ego_state_from_batch` **REFUSE** rather than
substitute zeros.

⛔ `--ablate-frames` is the rig's deliberate-regression lever and must never
appear on a real run. ⛔ Not Thor (training refav1), not `tanitad-refcv3` while
its eval holds it.

---

## 8. Corrections logged with this package

1. **The `ha0_ext` census.** The first draft recorded **0.4476 / 4.5912 m**
   (+36.42 % / +14.26 %). Re-running the CURRENT instrument on the SAME 40
   episodes and the SAME window counts (1,041 @ 2 s / 801 @ 6 s) reproduces
   `ha0` exactly (0.70402 / 5.35474) and reads `ha0_ext` **0.44492 / 4.46515**.
   Mechanism: the deceleration **stop clamp** (τ clamped at `−v0/a0`, so the
   vehicle stops rather than reversing) landed after that measurement; it can
   only lower the error and it lowered both cells. **Class: a derived constant
   re-measured after its input changed** — which is why the banked JSON, not the
   docstring, is the source. Docstrings corrected in the same commit.
2. **`refc.py`'s docstring launderered an unswept hyper-parameter.** It listed
   *"the measurement encoder with per-sample ego-dropout"* among the parts kept
   **verbatim** from TCP. The encoder is TCP's (FC-128 × 2, bit-for-bit); the
   dropout is not — *"dropout"* occurs **zero** times in arXiv 2206.08129 (two
   probes: a full-text search, and TCP's own layer table). `ego_dropout = 0.5`
   is a TanitAD invention nobody has swept; the two planners that did sweep it
   landed on **0.5** (DRAMA, NAVSIM PDMS 0.835 → 0.848) and **0.75** (PlanTF, at
   a measured −1.48 OLS).
3. **The trainer was dropping its own anti-echo readout.** `echo_ratio`,
   `echo_base_absmean` and `g_tac_delta_absmean` were computed on every forward
   and never reached `metrics.jsonl`, so the design's promise that *"is it
   echoing?"* is read off the LOG was not kept by the code that writes the log.
   Same class as the `tac_label_v7` drop already recorded in that file.

---

## 9. What this package does NOT settle

1. **H-ECHO-1** (does the ego state help without echoing?) — needs the full
   30k run at `--size small`; the rig is not powered for it.
2. **H-ECHO-2's effect** — the X15 mechanism is measured and the fix is
   implemented, but `brake_stop` reachability at eval is a property of a trained
   full-rung arm.
3. **The `ego_dropout` rate** is not swept. The panel fixes 0.0 vs 0.5; PlanTF's
   0.75 (and the OLS/CLS trade it implies) is a follow-up arm.
4. **The route head is nav-independent at exactly +0.0000** — a second broken
   input beside the missing ego state. v4 does not touch it; naming it here is
   what stops a v4 result being read as *"the conditioning is fine now"*.
5. **The CV-triviality census of the val split** (*what fraction does a
   constant-velocity agent already solve?*) — the sibling package ranks it the
   highest-leverage item (NAVSIM moved the CV agent **PDMS 79 → 22 by filtering
   alone**). §3/§6.1 of the prereg are the two-point version; the distribution
   and the `hard` split are not built.
6. **No T2, and no closed loop.** Per the 2026-09-02 ruling a planner feeding its
   own predictor is still open loop.

---

## 10. Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| pre-registration | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refc-v4-design/PREREG_REFC_V4.md` | no |
| this result | `repo:.../2026-09-03-refc-v4-design/RESULT.md` | no |
| corpus census (the §3 / §1.4 numbers) | `repo:.../raw/census_val40.json` (+ `census.py`) | no |
| the tiny-rig gate report | `repo:.../raw/gate_report.json` | no |
| per-arm training logs + configs | `repo:.../raw/arms/<arm>/{config.json,metrics.jsonl,summary.json}` | no (checkpoints stay local — `C:\Users\Admin\run_refcv4\arms\<arm>\ckpt.pt`, ~68 MB each, **local disk only**) |
| panel runner / gate scorer | `repo:.../raw/panel.sh`, `repo:.../raw/gate_score.py` | no |
| full-suite log | `repo:.../raw/pytest_full.txt` | no |
| E11′ + E14 + X15 | `repo:stack/tanitad/refs/refc_v3.py`, `repo:stack/tanitad/refs/refc.py` | no |
| the anti-echo gate | `repo:stack/tanitad/eval/echo_gate.py` | no |
| flags, stamps, rig rung, `--ablate-frames` | `repo:stack/scripts/refc_v3_train.py` | no |
| tests (30) | `repo:stack/tests/test_refc_v4.py` | no |
| hypotheses H-ECHO-1..7 | `repo:Project Steering/GOALS_AND_CLAIMS.md` | no |

⚠️ **The five tiny-rig checkpoints live on ONE disk** (`C:\Users\Admin\run_refcv4\arms\`).
They are rig artifacts of a NON-PARITY 2,000-step screen and are reproducible
from `raw/panel.sh` in ~80 minutes, so they are deliberately not banked; the
configs and logs that make them reproducible ARE banked.
