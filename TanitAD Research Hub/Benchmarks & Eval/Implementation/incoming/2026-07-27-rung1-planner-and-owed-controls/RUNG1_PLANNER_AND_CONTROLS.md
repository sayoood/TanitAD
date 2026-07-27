# THE RUNG 1 PLANNER ARM, AND THE 30 OWED CONTROL ROWS

**Date:** 2026-07-27 (Europe/Berlin; pods log UTC). **Streams:** blind-imagination `T_blind` ladder ·
BOOST §7.1 control re-adjudication.
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before any number here existed**
(its mtime precedes every file in `raw/` and `perwindow/`). It is **not edited**; every deviation is
an amendment in §A.

**Host:** pod2 only (A40). `MEASURED` idle before launch **0 MiB / 0 %**; Python **3.11.10**; disk
verified with a **real 2 GB `dd` write** (2,097,152,000 bytes in full, 4.0 GB/s) — `df` never
consulted; `/root/tbr1` (the previous agent's 143 MB) checked before adding `/root/tbr1p`. ⛔ **pod1**
(v2corpus training), **pod3** (situation-classifier build) and the **eval pod** (trafficsim +
wheelbase) were **never connected to**. One job, launched and tracked by **explicit PID 266291**,
never `pkill -f`. The val cache was read only.
🔒 PhysicalAI-AV is gated-confidential: **no clip UUID and no raw content** appears in this folder.

**Estimator, everywhere:** **paired episode-cluster bootstrap** — `taniteval/ci.py`, **B = 2000,
seed 0**, unit = **episode cluster**. ⛔ `overlapping_holdout_se` / `_jack` appears **nowhere** in
this folder.
**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS`. **Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

---

# 0. VERDICT

> ## ⛔⛔ **THE PLANNER ARM IS REFUTED, AND HARD. v1's deployed tactical planner + pure-pursuit controller reaches `T_blind` = 10 steps = 1.0 s [1.0, 1.2] — against a pre-registered prediction of 6–12 s, against the `ema0.8` gain-equivalence of 6.4 s, and BELOW the 2.5 s kinematic-inverse baseline it was supposed to beat. At 2 s it is separated-WORSE than that baseline by 1.28 m [1.11, 1.46]. beats-CV 0/185; `T_useful@1m` 1.0 s against the baseline's 1.4 s. This is §7.2's own stated finding condition, and it fires.**
>
> ## 🔴 **AND THE PRE-REGISTERED REASON IS REFUTED TOO — THE "5× LOWER LONGITUDINAL GAIN" PRODUCES NO LOWER COMMAND. Over the first 0.5 s the planner's mean \|accel\| is 2.118 m/s² against the inverse's 2.058 (ratio 1.029) and it sits at the ±3 clamp 49.7 % of the time against 46.4 % (ratio 1.072). Both slightly WORSE. A 5× lower gain only lowers the command if the error it multiplies is unchanged; the inverse multiplies a ONE-TICK speed change, the planner multiplies a 0.5 s TARGET-SPEED MISMATCH.**
>
> ## ⭐⭐ **THE CONTROLLER IS EXONERATED COMPLETELY — THE FAULT IS THE TACTICAL HEAD'S 0.5 s WAYPOINT. Fed the TRUE 0.5 s lookahead, the SAME `wp_to_control` saturates the sweep: `T_blind` 185 (terminus), `de@2s` 0.2985, beats CV 180/185, `T_useful@1m` 3.6 s — better than `hold_last`, better than `gtkin`, better than every zero-training filter Rung 1 found. The controller can drive. The plan it is given cannot.**
>
> ## 🔴 **AND IT IS A DIFFERENT FAILURE MODE, NOT A WORSE VERSION OF THE SAME ONE. The inverse's command is a near-zero-mean oscillation (sign-flip 0.287/tick, bias/amplitude 0.0165). The planner's is NOT oscillating (sign-flip 0.026) and its bias/amplitude is 0.309 — 19×. It converts a self-cancelling oscillation into a SUSTAINED directional command. That is exactly the mechanism Rung 1 measured as making `every` catastrophic, reached by a different route.**
>
> ## 🔴 **CONTROLS: OUTCOME B AGAIN. Of the 30 owed rows, ZERO are clean. Two FAIL outright — `firewall.E`'s blind head is separated ABOVE chance (+0.2821 [+0.0920, +0.4683]) and `head_ego`'s AP is separated above a TRUE random ranking on 24/24 seeds, contradicting the published "NO ARM IS ABOVE CHANCE". One is VOID (`firewall.H`, MDE 222 % of the maximum physically possible effect). One is resolved by a higher-n sibling. The other 26 are UNDER-POWERED or STRUCTURALLY ATTENUATED, with three newly-measured defects that all bias toward the desired verdict.**

| what was asked | the answer, `MEASURED` |
|---|---|
| the R1 PLANNER arm, scored against 6–12 s | ⛔ **10 steps = 1.0 s [1.0, 1.2]. REFUTED — LOW.** Below the prediction, below `ema0.8` (64), below the 25-step baseline |
| the mechanism | 🔴 **the gain argument is refuted.** Amplitude ratio **1.029**, saturation ratio **1.072**. The planner is a SUSTAINED BIAS (bias/amp 0.309) where the inverse is an OSCILLATION (0.0165) |
| whose fault | ⭐ **the tactical head's 0.5 s waypoint.** `planner_gtlook` (same controller, true target) = **185 steps, `de@2s` 0.2985, beats CV 180/185** |
| the 30 owed rows | 🔴 **2 FAIL · 1 VOID · 1 resolved by a sibling · 26 under-powered/attenuated · 0 clean.** Three new instrument defects, all biasing toward the null |
| what has to be withdrawn | **4 withdrawals + 2 amendments** (§6), the largest being `H2_CLASSIFIER.md`'s *"NO ARM IS ABOVE CHANCE"* and the frozen list's own inverted `DEAD-CONTROL` family description |

**Tiers.** The **planner `T_blind` number is DECISION-GRADE**: pre-registered buckets, primary
statistic and estimator fixed in advance, comparator matched in **both** the readout and the action
source, window-set identity gated at 599/596, validated in both directions, and with a falsifier
demonstrated to fire. The **mechanism sub-verdict is CONFIRMED** (measured on the same run's dense
`fed_actions`, both the planner and the inverse re-measured rather than inherited). The **control
verdicts are CONFIRMED** where they rest on a recomputation that reproduces the committed number
first, and **PROVISIONAL** where they rest on a corrected comparator that the source stream has not
yet adopted.

---

# 1. Pre-registration, and what it fixed before any number existed

`PRE_REGISTRATION.md` fixed, before any computation: for **Part A** the 30 frozen rows by index, the
five method rules, the **six-verdict vocabulary** (three of them failing), the per-family plan with
its failing values, and a **both-directions harness self-test per instrument**; for **Part B** the
primary statistic and estimator, the arm list and eligibility, the **[60, 120]-step prediction
buckets**, the failing value, the four blocking gates, the M10 tier split, and the confound.

Two things it declared in advance rather than discovering late:

* ⚠️ **The planner changes TWO things at once** versus the kinematic inverse — the controller gains
  *and* the source of the intent. §7.2's prediction is framed on the gains alone. Both attacks on
  that confound were registered in advance (§B.3) and both were run.
* ⚠️ **No cadence knob was placed on the planner.** Running `wp_to_control` at a reduced re-plan rate
  is the `every` family Rung 1 measured to be destructive; repeating a refuted intervention as a
  "fix" was ruled out before the arm list existed.

---

# 2. ⛔ The gates — nothing below was read until all of them passed

`raw/r1planner_gates.json`.

## 2.1 Window-set identity — BLOCKING, PASSED

| check | result |
|---|---|
| windows | **599 new vs 599 committed (both dumps)** ✅ |
| episode clusters | **596** |
| `eid` ordering identical vs **both** committed dumps | ✅ |
| `t0` ordering identical vs **both** | ✅ |
| anchor `a_imagination__own__roSTR` | max \|Δ\| = **3.05e-05 m** ✅ (tol 1e-4) |
| anchor `a_imagination__hold__roSTR` | max \|Δ\| = **3.05e-05 m** ✅ |
| anchor `b_frozenlast__own__roSTR` | max \|Δ\| = **0.0 m** ✅ **bit-identical** |
| anchor `b_frozenlast__hold__roSTR` | max \|Δ\| = **0.0 m** ✅ **bit-identical** |

**`IDENTITY_PASS = True`.** The two `b`-arm anchors reproduce the ladder's matched dump **exactly**;
the two `a`-arm anchors carry the **3.05e-05 m** float-kernel noise that Rung 0b measured between two
encode passes and that the tolerance exists for.

## 2.2 Fidelity against the ladder's committed numbers — BLOCKING on levels, PASSED

| quantity | committed | recomputed here |
|---|---:|---:|
| `T_blind` own \| `str` | 25 steps | **25** |
| `T_blind` hold \| `str` | 115 steps | **115** |
| `de@2s` own / hold | 1.8165 / 0.6718 | **1.8165 / 0.6718** |
| `ade_0_2s` own / hold | 0.8710 / 0.3351 | **0.8710 / 0.3351** |
| `T_useful@1m` own / hold | 1.4 s / 2.3 s | **1.4 s / 2.3 s** |

`LEVEL_FIDELITY_PASS = True` · `T_BLIND_EXACT_REPRODUCTION = True`. The integer reproduction was
declared **reported, non-blocking** in advance and did not need the latitude.

## 2.3 ⭐ The plumbing self-test — in BOTH directions

A new action source that is silently the old one produces a flat, confident, wrong curve. Two
independent guards, both registered in advance:

| direction | test | result |
|---|---|---|
| **anti-no-op** | every planner arm must differ from **both** `own` and `hold` | ✅ smallest max \|Δ\| across the four planner arms: **272.25 m** vs hold, **454.98 m** vs own. Arms identical to either: **none** |
| **fidelity** | the fed action must BE `closedloop.wp_to_control`'s output | ✅ pinned on a CPU fixture: `test_planner_feeds_exactly_wp_to_control` asserts `torch.equal` against the deployed controller's own return, and `test_planner_speed_bookkeeping_is_closed_loop_rollouts` pins the `v ← clamp_min(v + accel·DT, 0)` integration |
| **inertness** | every pre-planner call site must stay bit-identical | ✅ `test_no_planner_leaves_every_pre_existing_path_bit_identical` |
| **knob is real** | `vsrc` and `look` must move the fed action | ✅ both asserted non-equal |

> ### ⭐ **`wp_to_control` is IMPORTED from `taniteval.closedloop`, never copied. That is the load-bearing property of this rung: the arm cannot drift from the controller `closed_loop_rollout` deploys, because it *is* that function.** ⚠️ The first version of the `vsrc` anti-no-op test PASSED VACUOUSLY — with a constant far-away lookahead target the accel saturates the ±3 clamp for every input speed, so both arms were identical. That is the very saturation mechanism under test, and it would have certified a knob as real on the strength of a clamp. The fixture was changed to a speed-tracking target; recorded as amendment **A3**.

## 2.4 The failing value, probed on real arms

The primary statistic's failing value is **1 step (0.1 s), not 0**, returned when the first evaluable
horizon already fails. Probe: **identical arms → 1 step; swapped arms → 1 step.** ✅ And
`frac_draws_T_blind_at_floor_1step = 0.000` for every arm here — *a result, not a tautology*, since
Rung 0 measured 1.000 on its unmatched contrast.

---

# 3. ⛔ THE PLANNER ARM — the priority deliverable of Part B

`raw/r1planner_arms.json`, `raw/r1planner_verdict.json`. Every row has a comparator matched in the
readout **and** in the action source — the same planner drives the frozen-percept arm.

| arm | eligible | `T_blind` | CI95 | `de@2s` | `de@6s` | `ade_0_2s` | Δ@2s vs own baseline | **beats CV** | `T_useful@1m` |
|---|---|---:|---|---:|---:|---:|---|---|---:|
| `own_kinematic` (baseline) | — | **25** (2.5 s) | [2.5, 3.9] | 1.8165 | — | 0.8710 | — | ⛔ 0/185 | 1.4 s |
| `hold_last` (ceiling) | — | **115** (11.5 s) | [11.5, 17.4] | 0.6718 | — | 0.3351 | — | ✅ 83/185 | 2.3 s |
| ⛔ **`planner`** ⭐ primary | ✅ | **10** (1.0 s) | **[1.0, 1.2]** | **3.0941** | 21.3673 | 1.5377 | **−1.2776 [−1.4603, −1.1059]** ✅ sep, **WORSE** | ⛔ **0/185** | **1.0 s** |
| `planner_vdec` | ✅ | **10** (1.0 s) | [1.0, 1.2] | 3.1904 | 21.7214 | 1.5752 | −1.3739 ✅ sep, worse | ⛔ 0/185 | 1.0 s |
| 🔴 `planner_vupd` | ✅ | **8** (0.8 s) | [0.8, 0.9] | **5.9511** | 35.3025 | 2.8918 | −4.1345 ✅ sep, worse | ⛔ 0/185 | 0.7 s |
| ⭐ `planner_gtlook` | ⛔ **diagnostic, privileged** | **185 ⚠️ saturated** | [18.5, 18.5] | **0.2985** | 2.9843 | **0.1879** | **+1.5180** ✅ sep, BETTER | ✅ **180/185** | **3.6 s** |

**Best ELIGIBLE arm: `planner`, 10 steps.**

## 3.1 The verdict, applied mechanically

| | |
|---|---|
| rule (fixed before any number) | CONFIRMED: `T_blind` ∈ **[60, 120] steps** · EXCEEDED: > 120 · **REFUTED-LOW: < 60** |
| prediction being scored | *"between 6 and 12 s … If it lands BELOW 6.4 s, the planner is adding a failure mode the gain argument does not explain and that is the finding."* (`TBLIND_RUNG1.md` §7.2) |
| baseline, this run | **25 steps** · ceiling **115** · `ema0.8` gain-reference **64** (`INHERITED-MEASURED`) |
| **`T_blind`** | ⛔ **10 steps = 1.0 s [1.0, 1.2]** |
| below the `ema0.8` equivalence (64) | ✅ yes |
| below the `own_kinematic` baseline (25) | ✅ **yes** |
| **BUCKET** | ⛔⛔ **PREDICTION REFUTED — LOW** |

## 3.2 ⚠️ M10 — the tier split is kept, and the capability claim does NOT fire

The `T_blind` number is an extension **against a frozen percept**. The capability claim rests only on
the two comparator-free statistics, which no readout or action-source mismatch can enter:

| | own (baseline) | **`planner`** | `hold_last` |
|---|---|---|---|
| beats constant velocity | ⛔ 0/185 | ⛔ **0/185** | ✅ 83/185 |
| `T_useful@1m` | 1.4 s | ⛔ **1.0 s** | 2.3 s |
| `T_useful@2m` | 2.1 s | ⛔ **1.5 s** | 3.2 s |
| **capability verdict** | — | ⛔ **NOT CONFIRMED — it moves DOWN** | ✅ |

> ### ⛔ **Both comparator-free statistics move in the WRONG direction. Rung 1's capability break — the first in the program — came from a filter on the action tensor, not from the planner. Nothing in this rung is a capability gain, and the `T_blind` drop and the capability drop agree, which is why the negative reading is admissible.**

## 3.3 ⚠️ And under the planner, imagining is WORSE than freezing the world

The matched comparator is the `frozen_last` arm driven by the *same* planner. At 2 s:

| arm | Δ (comparator − imagination) at 2 s | reading |
|---|---|---|
| `planner` | **−0.7439 [−0.8875, −0.5894]** ✅ separated | 🔴 the **frozen-percept** arm is better by 0.74 m |
| `planner_vupd` | −1.6503 [−1.8641, −1.4201] ✅ separated | 🔴 worse still |
| `planner_gtlook` | **+1.6935 [+1.5664, +1.8320]** ✅ separated | ✅ imagination is better by 1.69 m |

⇒ **Under the deployed planner the world model's dynamics are a net LIABILITY past 1.0 s** — you do
better by pretending the world stopped. Fed a correct target, the same loop is strongly better than
freezing. **The dynamics are fine; the intent is not.**

---

# 4. 🔴 MECHANISM — the gain argument is refuted, and the real one is named

`raw/r1planner_mechanism.json`. Re-measured on this run's **dense `fed_actions`** for both the
planner and the inverse — the inverse's Rung-1 signature is **re-derived here, not inherited**.

## 4.1 The amplitude did not move — the pre-registered signature fails

| arm, first **5** fed steps (0.5 s) | mean \|steer\| | **mean \|accel\|** | frac steer at clamp | **frac accel at clamp** |
|---|---:|---:|---:|---:|
| `own_kinematic` | 0.00978 | **2.0582** | 0.0928 | ⛔ **0.4641** |
| ⛔ **`planner`** | 0.01261 | 🔴 **2.1176** | 0.1506 | 🔴 **0.4975** |
| `planner_vdec` | 0.01263 | 2.2041 | 0.1506 | 0.5169 |
| `hold_last` | 0.02901 | ✅ 0.4506 | 0.1185 | ✅ 0.0050 |
| ⭐ `planner_gtlook` | 0.01320 | ✅ **0.5117** | 0.1275 | ✅ **0.0050** |

**amplitude ratio planner/own = 1.029 · saturation ratio = 1.072.** Both slightly *above* 1.
§7.2 predicted *"a 5× lower longitudinal gain, so a materially SMALLER amplitude and saturation"*.

> ### 🔴 **A 5× lower GAIN only lowers the COMMAND if the ERROR it multiplies is unchanged. `wp_to_control` computes `accel = (v_target − v)/0.5` where `v_target = x/(5·0.1)` is the speed the tactical head's own 0.5 s waypoint IMPLIES. The kinematic inverse computes `(v − v_prev)/0.1` — a ONE-TICK speed change. The planner multiplies a 0.5 s TARGET-SPEED MISMATCH, which is larger than a one-tick change by more than the 5× gain ratio. The gain argument silently assumed `v_target ≈ v`.**

`MEASURED`, by inverting `wp_to_control`'s own P-controller on the **unsaturated** fed steps
(`raw/r1planner_mechanism.json → implied_target_speed`):

| window | frac steps **unsaturated** | mean `v` | mean `v_target` | mean \|`v_target` − `v`\| |
|---|---:|---:|---:|---:|
| first 5 (0.5 s) | ⛔ **0.5025** | 9.80 m/s | 9.84 m/s | **0.622 m/s** |
| first 20 (2 s) | 0.5338 | 8.47 | 8.55 | 0.535 |
| all 185 | 0.8454 | 6.88 | 6.92 | 0.256 |

⇒ **On half of the first five ticks the tactical head asks for a speed change so large that the ±3 m/s²
clamp saturates** — i.e. its 0.5 s waypoint is more than 0.75 m away from the speed-matched value.

## 4.2 ⭐ And it is a DIFFERENT failure mode, not a worse version of the same one

| statistic, all 185 fed steps | `own_kinematic` | **`planner`** | ratio |
|---|---:|---:|---:|
| sign-flip rate of the accel command, per tick | **0.2871** | ✅ **0.0260** | 0.09× |
| mean \|accel\| | 1.0893 | **0.8966** | 0.82× |
| frac at the ±3 clamp | 0.2074 | **0.1546** | 0.75× |
| **bias / amplitude** | ⭐ **0.0165** | 🔴 **0.3092** | **18.7×** |

> ### 🔴 **The planner DID reduce amplitude and saturation over the full horizon — and it is far worse anyway, because it is not oscillating. The inverse emits a near-zero-mean oscillation whose errors largely cancel (bias/amplitude 0.017). The planner emits a SUSTAINED, DIRECTIONAL command (bias/amplitude 0.309, sign-flip 0.026). Rung 1's binding lesson was that *"holding a sample of a zero-mean oscillation removes its cancellation"* — which is why `every` was catastrophic. The planner reaches the same destination by a different route: it never had the cancellation to begin with.**
>
> ⚠️ **This also corrects Rung 1's own headline mechanism as a general statement.** Rung 1 concluded *"what binds is the oscillation's AMPLITUDE, not its presence"*. On the planner arm the amplitude is **lower** on every horizon-wide statistic and the outcome is **2.5× worse**. The correct general statement is that what binds is the **accumulated signed command** — amplitude matters only through it, and for a zero-mean oscillation it barely does.

## 4.3 ⭐ The convention control — the controller is exonerated entirely

`planner_gtlook` holds `wp_to_control` fixed and replaces the intent with the **true** 0.5 s
lookahead, transformed into the current imagined ego frame. It is **privileged and may never be
quoted as deployable**.

| arm | `T_blind` | `de@2s` | `ade_0_2s` | beats CV | `T_useful@1m` |
|---|---:|---:|---:|---|---:|
| `planner` | 10 | 3.0941 | 1.5377 | 0/185 | 1.0 s |
| ⭐ **`planner_gtlook`** | **185 ⚠️ saturated** | **0.2985** | **0.1879** | **180/185** | **3.6 s** |
| `gtkin` (`INHERITED`, Rung 1 §4.5) | 185 ⚠️ | 0.4361 | 0.2552 | 179/185 | 3.0 s |
| `hold_last` | 115 | 0.6718 | 0.3351 | 83/185 | 2.3 s |

> ### ⭐⭐ **`wp_to_control` fed a correct 0.5 s target is the best arm the program has ever measured on this surface — better than the kinematic inverse fed TRUE motion (`de@2s` 0.2985 vs 0.4361), better than never acting at all. ⚠️ C14: 185 is the sweep terminus, so it is a LOWER BOUND. The pure-pursuit controller, the ±3/±0.05 clamps and the 0.5 s time constant are NOT the problem. 100 % of the planner arm's deficit is the tactical head's 0.5 s waypoint.**

## 4.4 The speed-channel variant, and the one arm that must be struck

`planner_vupd` feeds `v0 = v_tracked / SPEED_SCALE`, which is what `closedloop.build_action` **actually
deploys**. It is **worse than the primary on every statistic**: `T_blind` 8 vs 10, `de@2s` **5.9511**
vs 3.0941, `ade_0_2s` 2.8918 vs 1.5377.
⚠️ This is a *milder* version of Rung 1's `own_vupd` catastrophe (`de@2s` 23.94) because the
controller's integrated speed is smooth where the decoded speed oscillates — but it is still a
**1.92× degradation**, and it is the deployed convention. Escalation **E-2**.

---

# 5. 🔴 THE 30 OWED CONTROL ROWS

The predecessor froze **75** control-type nulls, re-adjudicated 45, and left **30 owed** because pod2
held the 600-episode val and was training. pod2 is now free. ⛔ **The frozen list was not added to.**

**Headline: 0 of 30 are clean.**

| family | rows | 🔴 FAILS | ⛔ VOID | ✅ resolved | ⚠️ under-powered / attenuated | GPU needed |
|---|---:|---:|---:|---:|---:|---|
| `S1-BLINDvsMAJORITY` | 3 | **1** | **1** | 0 | 1 | none |
| `DEAD-CONTROL` | 3 | **1** (component) | 0 | **1** | 1 | none |
| `CE-CONTROL` | 14 | 0 | 0 | 0 | **14** | ⛔ blocked, costed |
| `CHANCE-BASELINE` | 10 | **1** | 0 | 0 | 9 (incl. 1 duplicate) | none |
| **total** | **30** | **3** | **1** | **1** | **25** | |

⭐ **Every one of the 30 was re-adjudicated with ZERO GPU** — not because they were easy, but because
in three of four families the answer was inside artifacts the program already owned and had never
read past the matching node. The one family that genuinely needs GPU is blocked by a **measured**
memory ceiling, and that is reported as owed, not as checked.

## 5.1 🔴 `S1-BLINDvsMAJORITY` (rows 23–25) — one FAILS, one VOID

`raw/owed_s1_firewall.json` · `scripts/owed_s1_firewall.py`. n is bounded by **20 AlpaSim scenes**
and cannot grow here. What could be done at the same n, and was not, is three things.

**Fidelity first.** `S1_RESULTS.json` reproduces **exactly** from its own driver (`REPRODUCES = True`
on every field of all three variants). Self-test passes in both directions: the interval estimator
returns not-separated on identical inputs and separates on a planted +1 shift; the packaged
`taniteval.blind_baseline` returns `CIRCULAR` on an echo context and not-`CIRCULAR` on noise; the MDE
emitter returns `can_fire = false` on the one case where truth was already known.

### ⭐ (a) The published interval does not interval its own published point estimate

`s1_slice.py:266-281` sets `acc_blind` to the **MAXIMUM** of two blind attacks — *"the reported
`acc_blind` must be the STRONGEST attack found, never the first one tried: a weak blind test fails
unsafe"* — but computes `blind_vs_majority_paired` from `run_firewall`'s `correct_blind`, which is
the **learned** attack only.

| variant | headline `acc_blind` | intervalled `acc_blind` | mismatch |
|---|---:|---:|---:|
| **E** | 0.7692 (deterministic) | 0.7308 (learned) | **+0.0385** |
| **H** | 0.5000 (deterministic) | 0.2500 (learned) | 🔴 **+0.2500** |
| NOGOAL | 0.6333 | 0.6333 | 0.0 |

⇒ variant **H**'s published `blind − majority` of **−0.5000** is the *learned* head's; the headline
head's is **−0.2500**. The interval understates the leak by construction, in the direction the design
note explicitly warns against.

### ⭐ (b) The leak-relevant contrast, never given an interval — and it SEPARATES

The firewall tests `blind` vs **majority**. A circularity firewall's question is `blind` vs
**chance**. `MEASURED`, paired episode-cluster bootstrap over the same 20 scenes:

| variant | `blind_HEADLINE − chance` | separated? | MDE | max possible | can fire? |
|---|---|:--:|---:|---:|:--:|
| 🔴 **E** | **+0.2821 [+0.0920, +0.4683]** | ✅ **YES** | 0.1882 | 0.5128 | ✅ |
| **NOGOAL** | +0.1500 [−0.0269, +0.3177] | ⛔ no | 0.1723 | 0.5167 | ✅ |
| **H** | +0.0000 [−0.3750, +0.3889] | ⛔ no | 0.3820 | 0.5000 | ✅ |

*(the learned-attack version separates too on E: +0.2436 [+0.0714, +0.3933])*

### (c) The power ceilings, on the rows as published

| row | variant | published contrast | **MDE** | **max possible effect** | MDE as % of max | **can this test EVER fire?** |
|---|---|---|---:|---:|---:|---|
| 23 | ⛔ **H** | blind − majority | **0.5556** | **0.2500** | **222.2 %** | ⛔ **NO — PROVEN IMPOSSIBLE** |
| 24 | NOGOAL | blind − majority | 0.2775 | 0.3000 | 92.5 % | ⚠️ only if blind captured > 92 % of headroom |
| 25 | E | blind − majority | 0.2457 | 0.3077 | 79.9 % | ⚠️ only if blind captured > 80 % |

| row | **VERDICT** | why |
|---|---|---|
| **25 · E** | 🔴 **CONTROL FAILS** | the blind head is separated **above chance**. The conditioning carries real target information — which is exactly what a circularity firewall exists to surface. *(also carries the §5.1a mismatch)* |
| **23 · H** | ⛔ **VOID** | MDE 222 % of the largest effect that can physically exist. Confirms the predecessor's proof independently, and adds that H's interval is on a different statistic than its headline |
| **24 · NOGOAL** | ⚠️ **UNDER-POWERED (OWED)** | not separated, at MDE 92.5 % of max. `blind − chance` = **+0.1500**, and the interval misses zero by **0.027** |

⚠️ **Note on what (b) does and does not establish.** A goal channel *should* carry information about
the branch — that is what a goal is for. What it may not do is *determine* it, and it may not be
scored as a raw accuracy. The packaged module's own verdict ladder says
`blind − majority ≥ SKILL_EPS (0.03)` ⇒ **LEAKY** ⇒ *"admissible ONLY if every reported number is a
SKILL OVER THIS BLIND BASELINE, never a raw score."* On E the headline `blind − majority` is
**+0.0769 ≥ 0.03**. **`s1_slice.py` implements only the CIRCULAR branch of that ladder** (`acc_blind
≥ 0.98 · ceiling`) and never the LEAKY one — so its `ADMITTED` means *"not CIRCULAR"*, which is
strictly weaker than the packaged module's `CLEAN`.

⚠️ **R-E is not executable as written, and that is a measurement.** The predecessor prescribed
deleting the re-implementation in favour of the packaged
`taniteval/taniteval/blind_baseline.py`. It is **not a drop-in**: the packaged firewall classifies
over a FIXED class set from a per-item context matrix, while S1 is a **variable-arity option set**.
Run here on a padded fixed-arity encoding it returns `CLEAN` on all three variants with
`blind_skill_over_majority` of −0.038 / 0.000 / −0.100 — i.e. it is a **strictly weaker attack** and
would have under-stated the leak. `MEASURED`, `raw/owed_s1_firewall.json → packaged_instrument`.
**Consolidation requires extending the packaged module to variable arity first.**

## 5.2 ⚠️ `DEAD-CONTROL` (rows 49–51) — and the frozen list has this family INVERTED

`raw/owed_dead_control.json` · `scripts/owed_dead_control.py`. Fidelity: the committed intervals
reproduce **exactly** from `yawext_40ep_perwindow.npz` (`REPRODUCES = True`); self-test passes in both
directions.

> ### ⛔ **THE FROZEN LIST DESCRIBES THIS FAMILY BACKWARDS.** `CONTROL_READJUDICATION.md` §1 records it as *"a perturbation with no physical content has **no** effect ⇒ the envelope is real"* — treating the null as the DESIRED verdict. The source pre-registers the opposite: *"The DESTROYED-OBSERVATION controls. Each must land **FAR above** the 12 deg value or C-ADE has no dynamic range and outcome C is declared"* (`lowood_ci_yawext.py:54-55`; `P1_REVALIDATION.md:77-79`). **These are dynamic-range controls that must FIRE. A null on them is the failing verdict, not the passing one** — so the whole "a null is a leak we lacked power to see" framing does not apply here, and applying it would have mis-read all three rows.

### ⭐ The pre-registered comparison was never run on this axis

The rule says *"far above the **12° value**"*, but every published dead-control interval is paired
against the **Δ = 0 baseline**. `MEASURED` here for the first time, from repo per-window data:

| dead condition | vs `yaw12`, **ADE** | vs `yaw12`, **ALONG** | vs `yaw12`, **CROSS** |
|---|---|---|---|
| `dead_black` | +1.5119 ✅ sep | +1.0009 ✅ sep | +0.6316 ✅ sep |
| `dead_noise` | +0.0942 ✅ sep | +0.0734 ✅ sep | +0.0194 ⛔ |
| `dead_shuffle` | +0.2888 ✅ sep | **+0.0871 ⛔** | +0.0664 ⛔ |
| `yaw90` *(the 4th control)* | +0.1498 ✅ sep | **+0.0644 ⛔** | +0.1155 ✅ sep |

✅ **The gate as pre-registered — on ADE — holds for all four controls.** ⛔ On the **along-track**
decomposition, **three of four** destroyed-observation controls fail to separate from the 12° value.

### The three rows

| row | deployment · node | committed | **MDE vs the gap it must catch** | **VERDICT** |
|---|---|---|---|---|
| **49** | 12ep · `dead_noise.paired_along_2s` | +0.1035 [−0.0117, +0.2334] ⛔ | 0.1226 = **167 %** of +0.0734 | ✅ **RESOLVED BY A HIGHER-n SIBLING** — the same contrast at n=40 is **+0.1100 [+0.0521, +0.1737] SEPARATED**. The n=12 null was a power artefact |
| **50** | 40ep · `dead_shuffle.paired_along_2s` | +0.1237 [−0.0708, +0.3305] ⛔ | 0.2006 = **230 %** of +0.0871 | 🔴 **CONTROL FAILS (component)** — a control that must fire, does not, at an MDE **2.3× too blunt to have seen the gap**. Same class as `firewall.H`'s 222 % |
| **51** | 12ep · `dead_shuffle.paired_along_2s` | +0.1298 [−0.2663, +0.4725] ⛔ | 0.3694 = **424 %** | ⚠️ **OWED — NOT RE-POWERABLE HERE**: 3.3× the clusters did **not** resolve it (row 50) |

⚠️ **Scope, stated so the verdict is not over-read.** The pre-registered gate is on the **ADE**
headline, which separates in every dead cell; `paired_along_2s` is the decomposition the report added
under its own design rule 3, and `yaw_edge_analysis.py` reads **only** `ade__*`. **So row 50's failing
verdict withdraws no published P1 claim.** Resolving it at α = 0.05 needs **≈ 106 episode clusters**
(`MEASURED` from the observed half-width), reachable only on the 600-ep build — a *different, easier*
corpus whose floor would shift, so it would not be quotable against this curve.

⚠️ **Reducer sensitivity, free from the same arrays:** row 50's mean-reduced +0.1237 becomes
**−0.0142** under a median reducer — **a sign flip**. The mean effect is tail-driven. `MEASURED`.

## 5.3 ⚠️ `CE-CONTROL` (rows 52–65) — structurally attenuated, and re-powering is blocked by a MEASURED ceiling

`raw/owed_ce_control.json` · `scripts/owed_ce_control.py`. All 14 nodes resolve and match the frozen
list (`all_nodes_resolve = True`).

### ⭐ The finding is inside `_folds_detail`, which no MDE screen can reach

The rescorer is a 5-fold episode-disjoint cross-fit with early stopping. `_folds_detail` records
`best_step` per arm per fold — and on several folds it is **0**, with
`finetune_helped_inner_val: false`. **On every window of those folds the arm IS the as-trained head,
bit for bit**, so the paired delta there is **exactly zero by construction**. Folds are 8 val
episodes each, so the zeros are whole episode clusters.

| goal mode | `ce` arm zero folds | `regret` arm zero folds | **both zero** |
|---|---|---|---|
| `produced` | 0, 4 → **40.1 % of windows, 16 of 40 clusters** | 0, 1, 4 → 60.1 % | **40.1 %** |
| `oracle` | 4 → **20.1 %, 8 of 40 clusters** | 0, 1, 3, 4 → 80.0 % | **20.1 %** |

> ### 🔴 **The node that names itself `regret_minus_ce_control_ISOLATES_THE_LOSS` is IDENTICALLY ZERO on 40.1 % of its windows in the `produced` arm — because both arms early-stopped at step 0 on those folds. It is not isolating the loss there; it is measuring nothing. And every `ce_control_minus_as_trained` node reports `n_episodes = 40` when only 24 clusters (produced) / 32 (oracle) can carry any signal at all.**

### The attenuation correction — exact arithmetic, not a projection

The mean over all windows equals `(1 − f₀) ×` the mean over the windows where the fine-tune actually
differed, because the rest are identically zero. So the point estimate is recoverable exactly; the
**interval is not**, and stays owed.

| row | node | committed δ | f₀ | **attenuation-corrected δ** | shift | effective clusters |
|---|---|---:|---:|---:|---:|---:|
| **60** | produced · `ce_control − as_trained` · `ade_0_2s` | +0.0668 | 40.1 % | **+0.1115** | **+67 %** | **24** (reported 40) |
| 56 | produced · same · `along` | +0.0567 | 40.1 % | +0.0946 | +67 % | 24 |
| 64 | produced · `regret − ce_control` · `ade_0_2s` | −0.0414 | 40.1 % | −0.0691 | −67 % | 24 |
| 55 | oracle · `ce_control − as_trained` · `ade_0_2s` | +0.0238 | 20.1 % | +0.0298 | +25 % | 32 |

*(the full 14 are in `raw/owed_ce_control.json → rows`)*

**M8 — the MDE against the effect each row exists to catch.** For row 60 the companion contrast it
must be distinguished from is `regret_minus_ce_control` (0.0414): MDE **0.1169 = 282 % of it**.
⇒ **The control cannot tell itself apart from the treatment effect it exists to isolate — 2.8× too
blunt, the same factor as the S3 longitudinal leak that turned out to be real.**

### ⛔ Why re-powering is OWED, with the numbers

| | |
|---|---|
| `MEASURED` GPU cache at 40 val episodes | **4,373,151,744 bytes = 4.07 GiB** for 6,844 windows (`_cache.bytes_on_gpu`) |
| projected at 600 val episodes | **61.1 GiB** |
| A40 VRAM | **46 GiB** ⇒ ⛔ **does not fit** |
| largest episode count that fits with 6 GiB headroom | **≈ 392** |
| host | the rescorer's `_stack_root` is `/root/v4eval/stack` — the **eval pod**, which this task is forbidden to touch |
| corpus | 40 → N val episodes changes the **corpus** as well as n (registry §1.2a: the 600-ep build is a measurably easier deployment, never substituted) |

**Verdict on all 14: ⚠️ UNDER-POWERED (OWED) + STRUCTURALLY ATTENUATED.** Ranked, costed, handed on.

## 5.4 🔴 `CHANCE-BASELINE` (rows 66–75) — one FAILS, and the comparator is not chance

`raw/owed_chance_baseline.json` · `scripts/owed_chance_baseline.py`. Fidelity: all five
`paired_AP_vs_chance` nodes reproduce **exactly** from the score files (`REPRODUCES = True`).
⭐ **Those score files existed only on pod2** and are now rescued into the repo, md5-verified.

### ⛔ (a) The "chance" comparator is 1.726× chance — MEASURED

`h2c_eval.py:138` builds it as `chance = np.zeros_like(y)`, and `h2c_stats.average_precision` ranks
with `np.argsort(-s, kind="mergesort")` — a **stable** sort. On an all-tied score a stable sort
returns **row order**, and the row order (`h2c_eval.py:85`) is *every left-camera row, then every
right-camera row*. So the "constant score" is really the ranker *"fire the left camera everywhere"*.

| quantity | value |
|---|---:|
| AP of the published constant-zero comparator | **0.005269** |
| the corpus base rate | **0.0030527** |
| **inflation** | 🔴 **1.726×** |
| AP of a genuinely random ranking (24 seeds) | **0.003172** [2.5 % 0.003158, 97.5 % 0.003185] |

⇒ `h2c_stats.average_precision`'s docstring and `h2c_eval.py:134-137` assert that *"a constant score
has AP exactly equal to the base rate within each draw"*. **`MEASURED`: FALSE as implemented.** The
comparator is HARDER than chance, so every AP-vs-chance delta is understated and every null is biased
toward *"not separated"* — **for a negative control, biased toward the desired verdict.**

### The rows, as published and against a corrected comparator

| row | node | published δ [CI95] | sep | **δ vs a TRUE random ranking** | seeds separating | **VERDICT** |
|---|---|---|:--:|---:|---:|---|
| 🔴 **69** | `paired_AP_vs_chance.head_ego` | +0.007489 [−0.001133, +0.031695] | ⛔ | **+0.009586** | ✅ **24/24** | 🔴 **CONTROL FAILS** |
| 73 | `paired_AP_vs_chance.head_img_ego` | +0.002679 [−0.002782, +0.026391] | ⛔ | +0.004776 | 21/24 | ⚠️ UNDER-POWERED |
| 75 | `paired_AP_vs_chance.head_img` | +0.000237 [−0.004207, +0.009181] | ⛔ | +0.002335 | 20/24 | ⚠️ UNDER-POWERED |
| 72 | `paired_AP_vs_chance.heur_decel` | +0.002406 [−0.003141, +0.015347] | ⛔ | +0.004503 | 16/24 | ⚠️ UNDER-POWERED |
| 67 | recall `head_ego − random_at_rate` | +0.133987 [−0.011030, +0.324680] | ⛔ | — | 7/24 | ⚠️ UNDER-POWERED |
| 66 | recall `head_img_ego − random_at_rate` | +0.101307 [−0.015760, +0.272730] | ⛔ | — | 1/24 | ⚠️ UNDER-POWERED |
| 70 | recall `head_img − random_at_rate` | +0.029412 [−0.041670, +0.132880] | ⛔ | — | 0/24 | ⚠️ UNDER-POWERED |
| **68** | `verdict.delta_vs_random` | — | — | — | — | ⚠️ **DUPLICATE of row 66** (byte-identical, `identical = True`) |
| 71 | `c12_fix.arms.head_img_ego.paired_AP_vs_chance` | +0.006006 [−0.000403, +0.039468] | ⛔ | +0.000673 | **0/24** | ⚠️ UNDER-POWERED |
| 74 | `c12_fix.arms.head_img.paired_AP_vs_chance` | +0.003103 [−0.002906, +0.042844] | ⛔ | +0.000673 | **0/24** | ⚠️ UNDER-POWERED |

> ### 🔴 **Row 69 flips. `head_ego`'s average precision IS separated above a genuinely random ranking, on 24 of 24 seeds. `H2_CLASSIFIER.md`'s numbered finding "NO ARM IS ABOVE CHANCE" is false for the ego arm — the arm was compared against a ranker 1.726× better than chance.**
>
> ⭐ **And the correction does NOT rescue the image arms — which is what makes it credible rather than convenient.** On the `c12_fix` surface the corrected comparator moves `head_img_ego` from +0.006006 to **+0.000673** and it separates on **0 of 24** seeds. `H2_CLASSIFIER.md`'s *"adding the image features to a WORKING ego head destroys it"* (`:98-102`) — which rests on rows 71/74 being null while `c12_fix head_ego` separates — **stands, and is strengthened**: the ego arm clears a true random ranking on both surfaces and the image arms clear it on neither. **The bias I found runs against the published conclusion in one place and with it in another, and both are reported.**
>
> ⚠️ **Every one of the 10 rows also fails the stream's OWN under-power rule** (`h2c_eval.py:408`,
> `half_width > |delta|`), which the stream applies to only 2 of them. `MEASURED`: `half_width_exceeds_point_estimate = true` on all 10. **None is VOID** — `can_fire` is true everywhere (MDE is 1.7–2.4 % of the maximum attainable effect), which is exactly the sibling evidence in (b) restated arithmetically.

### ⭐ (b) M9 — two siblings already proved the test can fire, and neither is quoted as such

* `c12_fix.json :: arms.head_ego.paired_AP_vs_chance` = **+0.07659 [+0.05055, +0.13529] SEPARATED** —
  same estimator, same 322 clusters, same head family, 1,642 positives instead of 306.
* `h2c_results.json :: paired_AP_vs_chance.heur_speed` = **−0.003282 [−0.008355, −0.000152]
  SEPARATED** — on the *exact* 306-positive target, in the negative direction.

⇒ the rig **can** separate at this n in both directions. So these nulls are genuine
*not-separated*, not structurally dead — the distinction the VOID verdict exists to make, and it is
the reason none of these ten is VOID.

### ⚠️ (c) The random firing baseline is ONE seed, and the 200-seed null is one key away

`h2c_eval.py:250` draws `default_rng(1000)` once, giving `random_at_rate.recall = 0.035948`. The
artifact's own `operating_point.random_seed_spread` records `recall_mean 0.024706`,
`analytic_expectation 0.025`. Seed 1000 landed near the **90th percentile of its own null** — 44 %
above expectation — biasing all three recall deltas low.

`MEASURED` over 24 fresh seeds: my own random recall mean is **0.029820** [2.5 % 0.014951, 97.5 %
0.043873], and every recall delta shifts up by **+0.006128** (e.g. `head_img_ego` +0.101307 →
**+0.107434**). ⇒ **the bias is real, and it is NOT decisive**: 1/24, 7/24 and 0/24 seeds separate.
Reporting it either way is the point — a defect that does not change a verdict is still a defect, and
saying so is what keeps the ones that *do* change a verdict credible.

### ⚠️ (d) n cannot grow here, and the ×3.4 projection does not transfer

`n = 322` counts **admitted labelled clips**, bounded by pod2 episode-cache membership
(`h2c_prep.py:96-102`), not by episodes. The label side is already ~4× larger (2,320 clips in the full
26-chunk table). Removing the limit is a **~52 GB gated camera re-download + re-decode**, `ESTIMATED`
by the source stream. Applying the episode-shrinkage projection to these rows would be the same
category error the harvest flagged for the McNemar row.

---

# 6. ⭐ DOWNSTREAM DEPENDENCY CHAINS — what must be withdrawn, by name

### CHAIN 1 — 🔴 `head_ego` is above chance ⇒ `H2_CLASSIFIER.md`'s "NO ARM IS ABOVE CHANCE" is withdrawn

```
paired_AP_vs_chance.head_ego  "not separated"  (n=322 clips, comparator AP 0.005269)
   └─ licensed:  H2_CLASSIFIER.md 44-62  "NO ARM IS ABOVE CHANCE"  (a numbered finding)
       └─ licensed:  :387-393, :538  "nothing separates from anything, and nothing clears chance"
           └─ FEEDS:  the corpus-expansion escalation (:104-110, :576-588)
```
⇒ **W-1 (WITHDRAW).** The sentence *"no arm separates from CHANCE"* as written. `MEASURED`: against a
genuinely random ranking `head_ego` separates on **24/24** seeds (δ +0.0096). The comparator used was
**1.726× chance** because a stable argsort on an all-tied score ranks by row order.
🟢 **The verdict `UNDERPOWERED` and the corpus-expansion escalation both STAND** — this makes the
case for expansion stronger, not weaker. But the arm ordering changes: the **ego** arm, not the
image arm, is the one with demonstrated signal, which is the same direction `c12_fix` already showed.

### CHAIN 2 — 🔴 `firewall.E` leaks ⇒ the S1 admissibility sentence needs a second correction

```
firewall.E.blind_vs_majority_paired  "not separated"  (n_cl = 20)   [the PUBLISHED test]
   └─ licensed:  S1_T1_SLICE.md:87  "The S1 target is NOT recoverable from the
                 conditioning channels. It is admissible."
       └─ licensed:  GATE_RESULTS.md 0  "blocked only on corpus size"
           └─ FEEDS:  STRATEGIC_TACTICAL_PROBLEM_SPEC -- 7 of 9 decision problems
```
⇒ **W-2 (WITHDRAW, strengthening the predecessor's W-3).** The predecessor withdrew that sentence on
the ground that variant H's test **could not fire**. This adds a second, independent ground:
**variant E's blind head is separated ABOVE chance (+0.2821 [+0.0920, +0.4683])**, and its headline
`blind − majority` of +0.0769 exceeds the packaged firewall's own `SKILL_EPS = 0.03` ⇒ **LEAKY** ⇒
*every S1 number must be reported as a skill over the blind baseline, never as a raw accuracy.*
🔴 **This is now a measured leak, not only a power argument.**
⇒ **W-3 (AMEND `s1_slice.py`).** It implements only the CIRCULAR branch of the packaged verdict
ladder. Its `ADMITTED` means *"not CIRCULAR"* and must not be read as *"CLEAN"*.
⇒ **W-4 (AMEND `S1_RESULTS.json`).** `blind_vs_majority_paired` must be computed on the **headline**
`acc_blind`, not the learned one. As published, variant H's interval is on a statistic 0.25 away from
its own point estimate.

### CHAIN 3 — the frozen list's `DEAD-CONTROL` family description is inverted

```
CONTROL_READJUDICATION.md 1  "DEAD-CONTROL: a perturbation with no physical content
                              has NO effect => the envelope is real"
   └─ implies: a NULL is the desired verdict
       => MEASURED: the source pre-registers the OPPOSITE
          (lowood_ci_yawext.py:54-55; P1_REVALIDATION.md:77-79)
```
⇒ **W-5 (AMEND `CONTROL_READJUDICATION.md` §1).** For this family a null is a **failing** verdict.
🟢 **No P1 claim is withdrawn**: the pre-registered gate is on ADE, which separates in every dead
cell, and the three frozen rows are the along-track decomposition, which no downstream script reads.

### CHAIN 4 — the CE control cannot isolate what it is named for

```
ce_control_minus_as_trained  "not separated"  (n_episodes reported = 40)
   └─ licensed:  "the counterfactual-equal arm is INERT => the measured loss is the selector's"
       => MEASURED: 40.1% of windows (16 of 40 clusters) are EXACTLY ZERO by construction
          (best_step = 0 folds), and MDE is 282% of the companion effect
```
⇒ **No withdrawal — Bar A's own verdict is already `REFUTE`, so nothing was claimed on the strength
of this control holding.** ⚠️ **Correction to the record:** the `ce_control − as_trained` effect on
the deployable surface is **+0.1115 (attenuation-corrected, MEASURED point estimate, interval OWED)**,
not +0.0668, and the node's `n_episodes = 40` overstates the clusters that can carry signal, which is
**24**. Anyone re-litigating Bar A on the published +0.0668 at n = 40 is arguing against the wrong
figure and the wrong n.

### CHAIN 5 — ⭐ the planner arm re-aims two open decisions

```
TBLIND_RUNG1.md 7.2  "PREDICTION: 6-12 s"   ->  MEASURED 1.0 s
   └─ E-1's re-derived R3 bar (>= 11.6 s) is UNAFFECTED -- it came from the filter axis
   └─ but the ladder's R1 row is now ANSWERED, and answered NEGATIVELY
       └─ FEEDS: the three-planner hierarchy direction, which now has a MEASURED target
```
⇒ **No withdrawal — §7.2 was labelled a PREDICTION, not a measurement, and §7.3 said in terms
*"the planner arm was NOT run"*.** That is the pre-registration working. But two things change:
* the **ladder's R1 row is closed**, and the answer is that v1's tactical planner is *worse than
  no planner at all* on this surface;
* **the hierarchy's value proposition moves.** Rung 1 concluded the action loop is *"fully
  recoverable without retraining"*, which moved the hierarchy's value *away* from stabilising the
  loop. `planner_gtlook` now says the controller and the loop are both fine and **the 0.5 s waypoint
  is the entire deficit** — i.e. the hierarchy's value is exactly *what the loop should be aiming
  at*, and that is now a measured claim rather than an inference.

---

# 7. 🔴 ESCALATIONS — in the headline, not written into a README

**E-1. `closedloop.closed_loop_rollout` is the deployed closed-loop harness, and its planner is
measured to be worse than not planning.** `T_blind` 1.0 s vs 2.5 s for a one-line kinematic inverse
and 11.5 s for holding the last action. Every closed-loop number the program has produced through
that harness inherits this. **This is a PI decision, not an engineering one:** either the tactical
head's 0.5 s waypoint is fixed, or the closed-loop harness should be driven by an action source that
is not it. `planner_gtlook` bounds what a fixed waypoint is worth: **`de@2s` 0.2985 against 3.0941 —
a 10.4× reduction, from the same controller.**

**E-2. `closedloop.build_action`'s speed-channel convention costs 1.92× at 2 s.** `planner_vupd`
(`v0 = v_tracked/SPEED_SCALE`, what the harness deploys) gives `de@2s` **5.9511** against the
constant-`v0` primary's 3.0941. Rung 1 struck the decoded-speed version of this lever; the
*controller-integrated* version is milder but still destructive and is currently deployed.

**E-3. 🔴 `H2_CLASSIFIER.md`'s "no arm is above chance" must be corrected before the corpus-expansion
decision is taken on it.** The comparator is 1.726× chance by a stable-sort artefact, and correcting
it flips `head_ego`. The expansion case survives; the arm ordering does not. **Owner: the H2 stream.**
The fix is ~2 lines in `h2c_eval.py` (`chance = rng.random(y.size)` or an explicit random tie-break)
plus a re-run of `paired_AP_vs_chance` — **CPU-only, ~10 minutes**, and the score files it needs are
now in the repo.

**E-4. The §6 emitter the predecessor asked for is still not implemented.**
`taniteval/taniteval/blind_baseline.py` emits neither `mde` nor `max_possible_effect` nor `can_fire`.
⚠️ **And R-E's prescription — delete the copies, use the packaged module — is NOT executable for S1
as things stand**: `MEASURED`, the packaged firewall cannot express a variable-arity option set and
returns a strictly weaker attack on the same data. **Consolidation must extend the packaged module
first.** Owner: TanitEval.

**E-5. Three of the 30 owed rows were never re-powerable and are now costed, not hand-waved.**
`CE-CONTROL` needs ≈ 392 val episodes max on an A40 (61.1 GiB at 600 vs 46 GiB VRAM) on a host this
task may not touch; `CHANCE-BASELINE` needs a ~52 GB gated camera re-download; `S1` needs ~103
AlpaSim scenes. **All three are data/host decisions, not compute ones.**

## 7.1 What this unblocks, per stream

| stream | what it gets |
|---|---|
| 🔴 **the ladder's R1 row** | **CLOSED, negatively.** 10 steps against a 6–12 s prediction, with the mechanism named and the controller exonerated |
| 🔴 **R3 / scheduled sampling** | a **sharper target**: the fix must attack the **tactical head's 0.5 s waypoint**, not the controller (`gtlook` saturates the sweep) and not the sampling rate (`every` is destructive) and not the gain (measured null) |
| ⭐ **the three-planner / hierarchy direction** | its value proposition is now **measured**: the loop and the controller are sound, the *intent* is the whole deficit. `planner_gtlook` is the ceiling a better tactical brain is aiming at — `de@2s` **0.2985** |
| **the closed-loop harness** | E-1 and E-2: the deployed planner and the deployed speed-channel convention both cost, and both are quantified |
| 🔴 **H2 / the corpus-expansion decision** | a **corrected comparator** and a flipped arm (E-3), before the download is booked |
| **the 4-brain S1 gate** | a **measured leak** on variant E, not only a power argument (W-2), and the LEAKY-branch gap in `s1_slice.py` (W-3) |
| **BOOST §7.1** | 30 rows moved from *owed* to *adjudicated*, of which **0 are clean** — and three new instrument-defect classes to screen for |

## 7.2 What is deliberately NOT claimed

* ⛔ **The planner arm is one arm, one checkpoint, one readout, one corpus.** v1 `flagship4b-speedjerk-30k`
  @ 29999, the `str` (k = 20) readout, the 599 episode-initial windows.
* ⛔ **`planner_gtlook` is privileged and saturates at the sweep terminus** — C14: a LOWER BOUND on
  our configuration, not a horizon, and never deployable.
* ⛔ **Not a safety result.** PhysicalAI-AV ships no map, lane graph or agent boxes. Drift only.
* ⛔ **The CE-CONTROL corrected point estimates have NO interval.** The per-window data is on the eval
  pod. They are `MEASURED` arithmetic on a stated construction, and the interval is **owed**.
* ⛔ **The corrected chance comparator is not yet the H2 stream's own.** Rows 69/72/73/75 are
  `PROVISIONAL` until that stream adopts the fix (E-3).
* ⛔ **No row was added to the frozen list**, and the second list (§A.4) is empty.

---

# 8. Limitations, stated plainly

1. ⚠️ **Everything past 0.4 s is extrapolation** — the operative readout was trained at k = 4, the
   `str` readout at k = 20. `INHERITED-MEASURED`: 20 % of steps at 6 s and 52 % at 12 s are outside
   the measured envelope. The planner's 1.0 s verdict sits *inside* that envelope, which is one
   reason it is the most robust number here.
2. ⚠️ **The window set is EPISODE-INITIAL** (596 of 599 windows at `t0 = 0`) and runs ~6–12 % low in
   absolute level (`INHERITED-MEASURED`). All contrasts are paired on identical windows.
3. ⚠️ **The planner arm changes TWO things at once** — controller gains *and* intent source. Declared
   in advance (§B.3). Both attacks were run and both point the same way: `ema0.8` (the gain-matched
   inverse) reaches 64 steps, and `planner_gtlook` (the intent-fixed planner) reaches 185. **The
   intent is the deficit.**
4. ⚠️ **`wp_to_control` uses `WHEELBASE = 2.7` while the corpus mints steer at 2.9** — a documented
   +7.41 % skew whose open-loop cost is `INHERITED-MEASURED` at ΔADE +0.0026 [−0.0006, +0.0062], not
   separated. The planner arm carries it *because the deployed harness does*, and the steering channel
   is not where the deficit is (mean \|steer\| 0.0126 against a corpus max of 0.016).
5. ⚠️ **The CE-CONTROL attenuation is exact for the POINT estimate and unavailable for the interval.**
6. ⚠️ **The corrected chance comparator uses 24 seeds**, not an analytic null. The seed spread on the
   comparator's own AP is [0.003158, 0.003185] — tight — but the paired separation is reported as a
   seed count, never as one interval.
7. ⚠️ **S1's n cannot grow here.** Everything in §5.1 is at 20 scenes and is stated as such.
8. **No safety metric anywhere in this folder.** Drift and classification only.

---

# 9. Amendments to my own pre-registration — recorded here, not by editing it

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **A1** | §A.3's `CONTROL FAILS` is defined as *"re-measured at higher n and separated"*. For the `DEAD-CONTROL` family the control's direction is **inverted** (§5.2), so the failing condition is the **null**, not the separation. | Applied in the **stricter** direction: the failing label is kept and its *condition* is inverted to match the source's own pre-registration, rather than re-reading a null as a pass. The direction was established **from the source code and the source report before any row was scored**. No row's number changed. |
| **A2** | The `c12_fix` arrays are **frame-level**, not `(camera, frame)`; the shared unfold crashed on them. | Detected from the array shape rather than assumed, after a crash. Affects rows 71/74 only, and only the *marshalling*; every estimator is unchanged. |
| **A3** | ⭐ The first `vsrc` anti-no-op test **passed vacuously**: with a constant far-away lookahead target the accel saturates the ±3 clamp for every input speed, so the two arms were bit-identical for the *right* reason and the test certified nothing. | Replaced with a speed-tracking fixture that keeps the controller off the clamp. **BINDING LESSON: an anti-no-op test must be run in a regime where the knob CAN change the output. A clamp turns a real knob into a no-op and a passing test into a tautology — which is the same class as the guards this task exists to audit.** |
| **A4** | §B.2 listed `planner_vupd` as ELIGIBLE. It is, and it is reported — but it is also the convention `closedloop` actually deploys, which the pre-registration described as "reported, not primary". | Kept out of the verdict exactly as registered. It would not have changed the bucket: 8 steps is further below the window than 10. |

---

# 10. 🔴 Retraction-log rows this stream owes — DRAFTED, NOT FILED

⛔ **Escalated, not filed** — `Project Steering/RETRACTION_LOG.md` is a shared append-only steering
document and siblings are staging into it this session. Handed over ready to paste, each with its
root-cause class per operating-standard rule 4.

> ### R-F · class: **A COMPARATOR THAT IS NOT THE BASELINE IT IS NAMED AFTER**
> **Retracted:** `H2_CLASSIFIER.md` §"NO ARM IS ABOVE CHANCE" (and `:387-393`, `:538`), and
> `h2c_stats.average_precision`'s claim that *"a constant score has AP equal to the base rate inside
> every bootstrap draw"*.
> **Correction:** `MEASURED` — the constant-zero comparator scores **AP 0.005269** against a base
> rate of **0.0030527**, i.e. **1.726× chance**, because a STABLE argsort on an all-tied score ranks
> by row order and the row order is all-left-camera-first. Against a genuinely random ranking
> `head_ego` separates on **24/24 seeds** (δ +0.0096).
> **Root cause:** a baseline was defined by its *intent* (`np.zeros_like`) rather than by its
> *behaviour under the metric*, and the metric is a **rank** statistic in which ties are not neutral.
> The file's own docstring documents the tie bias and then argues it is "the safe direction" — which
> is true for arm-vs-arm contrasts and **false for arm-vs-chance**, and nothing re-checked that.
> **Generalisation:** *a chance baseline must be MEASURED against the quantity it claims to equal, in
> the metric that consumes it, before any null is read off it. For a rank metric, a constant score is
> not a random ranking.*

> ### R-G · class: **AN INTERVAL ON A DIFFERENT STATISTIC THAN THE HEADLINE**
> **Retracted:** `S1_RESULTS.json → firewall.{E,H}.blind_vs_majority_paired` as an interval on the
> published `acc_blind`.
> **Correction:** `s1_slice.py:266-272` sets `acc_blind` to the **maximum** of a learned and a
> deterministic blind attack, then `:279-281` intervals `run_firewall`'s **learned** vector. For
> variant H the headline is **0.5000** and the intervalled quantity is **0.2500**. Recomputed on the
> headline statistic: E `+0.0769 [−0.0667, +0.2502]`, H `−0.2500 [−0.5556, +0.0000]`.
> **Root cause:** the strongest-attack rule was applied to the point estimate and not propagated to
> the interval, so the guard reports its **weakest** attack's uncertainty about its **strongest**
> attack's value — always in the direction that makes the firewall look cleaner.
> **Generalisation:** *when a headline is a max over several estimators, the interval must be taken on
> the SAME max, re-derived inside every resample. An interval computed on a different arm than the
> point estimate is not an interval on the published number.*

> ### R-H · class: **A CONTROL WHOSE DIRECTION IS RECORDED BACKWARDS**
> **Corrected (not retracted):** `CONTROL_READJUDICATION.md` §1's `DEAD-CONTROL` family description —
> *"a perturbation with no physical content has no effect ⇒ the envelope is real"*.
> **Correction:** the source pre-registers the **opposite** (`lowood_ci_yawext.py:54-55`,
> `P1_REVALIDATION.md:77-79`): destroyed-observation controls must land **far above** the 12° value or
> outcome C is declared. A null on them is the **failing** verdict.
> **Verdict on the P1 gate UNCHANGED: it passes on ADE** for all four controls.
> **Root cause:** the family was assigned by a **keyword vocabulary sweep** (`dead_`) and its meaning
> was inferred from the family NAME rather than read out of the emitting code. The sweep's own
> containment check could not catch this, because it checks membership, not semantics.
> **Generalisation:** *a control's DIRECTION is a property of the code that emits it, not of its name.
> A classifier that assigns meaning from a keyword must have each family's direction confirmed against
> the emitter before any verdict rule is applied to it.*

> ### R-I · class: **A CROSS-FIT ARM THAT IS THE BASELINE ON PART OF ITS OWN WINDOWS**
> **Corrected (not retracted):** every `bar_a_*` `ce_control_minus_as_trained` and
> `regret_minus_ce_control_ISOLATES_THE_LOSS` node's point estimate and `n_episodes`.
> **Correction:** `MEASURED` from `_folds_detail` — the fine-tune early-stopped at `best_step = 0` on
> whole episode-disjoint folds, so on **40.1 %** of the `produced` windows (16 of 40 clusters) the arm
> IS the as-trained head and the paired delta is **exactly zero by construction**. The headline
> `ce_control − as_trained` `ade_0_2s` is **+0.1115** on the windows that can carry signal, not
> +0.0668, and the effective cluster count is **24**, not 40. The interval is **owed**.
> **Verdict on Bar A UNCHANGED: `REFUTE`.**
> **Root cause:** early stopping was allowed to return the initial state, and the reporting layer
> counted those windows as observations of the contrast. Nothing downstream distinguished "the
> intervention did nothing" from "the intervention was not applied".
> **Generalisation:** *when an arm can early-stop to its own baseline, the report must emit the
> fraction of the evaluation set on which the arm IS the baseline, and the effective cluster count.
> A contrast that is identically zero on 40 % of its windows is not a null — it is 40 % unmeasured.*

> ### R-J · class: **A PREDICTION SCORED AND REFUTED — the mechanism was right about the lever and wrong about the arm**
> **Retracted:** `TBLIND_RUNG1.md` §7.2's prediction that v1's tactical planner lands at 6–12 s of
> deployable `T_blind`, and the reasoning that its 5× lower longitudinal gain would deliver it.
> **Correction:** `MEASURED` — **10 steps = 1.0 s [1.0, 1.2]**, below the 2.5 s kinematic-inverse
> baseline, and the amplitude does not move (ratio 1.029, saturation ratio 1.072).
> 🟢 **This is the pre-registration working, not a failure of it:** §7.2 was labelled a PREDICTION and
> §7.3 stated plainly *"the planner arm was NOT run"*. Nothing was published on it.
> **Root cause:** a gain was compared without the **error it multiplies**. `(v_target − v)/0.5` and
> `(v − v_prev)/0.1` differ by 5× in gain and by more than 5× in the magnitude of their inputs — a
> one-tick speed change versus a 0.5 s target-speed mismatch. The argument implicitly assumed
> `v_target ≈ v`; `MEASURED`, the ±3 m/s² clamp saturates on **49.7 %** of the first five ticks.
> **Generalisation:** *a controller-gain argument is only admissible with the DISTRIBUTION OF ITS
> ERROR TERM attached. Comparing two control laws by their gains alone compares half of each.*
> **Second, and it corrects a live headline:** Rung 1's *"what binds is the oscillation's AMPLITUDE"*
> is **not general**. The planner has lower amplitude on every horizon-wide statistic and is 2.5×
> worse, because it is not oscillating (sign-flip 0.026 vs 0.287, bias/amplitude 0.309 vs 0.017).
> **What binds is the ACCUMULATED SIGNED command; amplitude matters only through it.**

---

# 11. DELIVERABLE MANIFEST

**Everything is in the repo working tree and STAGED (`git add`). Nothing was committed or pushed.**
Path: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-rung1-planner-and-owed-controls/`

| # | artifact | what it is | where it lives | only ONE copy? |
|---|---|---|---|:--:|
| 1 | `PRE_REGISTRATION.md` | written **before** any number existed: the 30 rows by index, the 6-verdict vocabulary, the per-instrument both-directions self-tests, the planner arm list + eligibility, the [60,120] buckets, the failing values, the 4 gates, the M10 split | **repo** | ❌ |
| 2 | `RUNG1_PLANNER_AND_CONTROLS.md` | this document | **repo** | ❌ |
| 3 | `scripts/tb_r1_planner_sweep.py` | the 12-arm pod driver — `bi_run.stage_sweep` reused **verbatim**, arm list replaced | **repo** · `pod2:/root/tbr1p/` | ❌ |
| 4 | `scripts/tb_r1_planner_compact.py` | pod-side compaction + the anti-no-op check at full K | **repo** | ❌ |
| 5 | `scripts/tb_r1_planner_analyze.py` | ⭐ the adjudication — identity gate, fidelity gate, plumbing self-test, failing-value probe, arms, mechanism, verdict | **repo** | ❌ |
| 6 | `scripts/owed_s1_firewall.py` | rows 23–25: reproduction, the headline-vs-interval mismatch, the never-intervalled `blind − chance`, MDE/`can_fire`, the packaged instrument as a second attack | **repo** | ❌ |
| 7 | `scripts/owed_dead_control.py` | rows 49–51: the direction correction, the never-run `vs yaw12` contrast, the 4th control, MDE, reducer sensitivity | **repo** | ❌ |
| 8 | `scripts/owed_ce_control.py` | rows 52–65: the `best_step = 0` structural zeros, the attenuation correction, the MEASURED re-power ceiling | **repo** | ❌ |
| 9 | `scripts/owed_chance_baseline.py` | rows 66–75: the comparator audit, the corrected random ranking over 24 seeds, the seed audit, the duplicate, the sibling `can_fire` demonstrations | **repo** | ❌ |
| 10 | `raw/r1planner_gates.json` | ⛔ identity gate · plumbing self-test · fidelity vs the ladder · failing-value probe | **repo** | ❌ |
| 11 | `raw/r1planner_arms.json` | every arm with its matched comparator, `T_blind`, `de@2s`, `de@6s`, `ade_0_2s`, beats-CV, `T_useful` | **repo** | ❌ |
| 12 | `raw/r1planner_mechanism.json` | ⭐ dense action statistics at 3 windows for 5 arms + the implied-target-speed inversion | **repo** | ❌ |
| 13 | `raw/r1planner_verdict.json` | the mechanical verdict, the M10 tier split, the mechanism sub-verdict | **repo** | ❌ |
| 14 | `raw/owed_s1_firewall.json` · `owed_dead_control.json` · `owed_ce_control.json` · `owed_chance_baseline.json` | **one JSON per number** for all 30 owed rows | **repo** | ❌ |
| 15 | `raw/h2clf_scores/` (4.2 MB) | ⭐ **RESCUED** — `scores_heldout.npz` / `scores_oof_train.npz` for both H2 runs, which existed **only on pod2**. md5-verified identical on both sides. Without them no chance-baseline row is recomputable | **repo** · `pod2:/workspace/h2clf/` | **was ❌ single-disk, now not** |
| 16 | `perwindow/r1planner_compact_K185.pt` (23.5 MB) | ⭐ **dense per-window `de` [599 × 185] for all 12 arms + both floors, plus dense `fed_actions` for the 5 arms that set the mechanism verdict** — any bar, horizon or stratification recomputes with **no GPU** | **repo** · derived from `pod2:/root/tbr1p/` | ❌ |
| 17 | `raw/r1planner_sweep_meta_K185.json` · `r1planner_sweep_run.log` · `r1planner_analyze.log` · `owed_chance_baseline.log` | run manifests and logs incl. every gate | **repo** | ❌ |
| 18 | `taniteval/taniteval/blindimag.py` | **+~120 lines, 0 deletions** — the `planner` action source, `PLANNER_MOD_KEYS`, `plan_fn`/`gt_pos`, the running-pose transform. `wp_to_control` **imported from `taniteval.closedloop`**, never copied. Every pre-existing path bit-identical | **repo (modified)** · `pod2:/root/taniteval/` | ❌ |
| 19 | `taniteval/tests/test_blindimag.py` | **+18 tests, 0 deletions** — the `wp_to_control` identity, the `closed_loop_rollout` speed bookkeeping, anti-no-op against **both** endpoints, the oracle-target consumption, the running-pose equality, and inertness on every pre-planner call site | **repo (modified)** | ❌ |
| 20 | `…/2026-07-26-blind-imagination/scripts/bi_run.py` | **+~35 lines, 0 deletions** — `_plan_fn_for` (v1's deployed plan step) and `KEEP_FED`. One branch, inert for every non-planner arm | **repo (modified)** · `pod2:/root/bi/` | ❌ |

**Living in only ONE place (declared, per rule 2):** the full `perwindow_sweep_K185.pt` (**31.1 MB**,
md5 `9f67b637f6c3655fc25259567ff126d8`) and `perwindow_peek_K185.pt` exist only at
`pod2:/root/tbr1p/perwindow/`. **The 23.5 MB compaction in the repo carries everything every number
in this report needs**; the full dump rebuilds in ~19 min on an idle A40 (deterministic,
`torch.manual_seed(0)`, no sampling). The pulled file was **md5-verified identical to the pod** before
compaction.

**Suites.** The only files touched outside this folder are `taniteval/taniteval/blindimag.py`, its
test, and `bi_run.py` — all **purely additive** (0 deletions). `taniteval`: **488 passed** (449 before this task; +18 mine, the rest from siblings staging
concurrently). `stack`: **1123 passed, 7 skipped**. Nothing under `stack/` was modified by me.

⚠️ **The repo advanced mid-task.** The orchestrator swept part of this folder and the `taniteval`
changes into commits `2753d01…444ab48` while the work was still running. Nothing was lost and nothing
was committed by this agent; the remainder is staged.

**Cost.** pod2, one job, PID **266291**: encode **807.3 s** + rollout of **12 arms × 599 windows ×
185 steps** in **314.5 s** = **18.7 min**. All 30 control rows: **0 GPU-seconds**. `MEASURED`,
`raw/r1planner_sweep_meta_K185.json`.

---

# 12. Reproduction

```
# pod2 (A40, must be idle) — ~19 GPU-min
PYTHONPATH=/root/bi:/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
OMP_NUM_THREADS=8 python3 tb_r1_planner_sweep.py \
    --out /root/tbr1p/perwindow --episodes 600 --kmax 185

# dev box — ZERO GPU throughout
python scripts/tb_r1_planner_compact.py \
    --dump <pulled perwindow_sweep_K185.pt> --out perwindow/r1planner_compact_K185.pt
python scripts/tb_r1_planner_analyze.py --new <pulled dump> \
    --bi      "<hub>/Architecture & Inference/.../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt" \
    --matched "<hub>/Architecture & Inference/.../2026-07-26-tblind-ladder/perwindow/perwindow_matched_K185.pt" \
    --out raw

python scripts/owed_s1_firewall.py     --gates "<hub>/Architecture & Inference/.../2026-07-26-4brain-gates" --out raw
python scripts/owed_dead_control.py    --p1    ../2026-07-26-p1-envelope-revalidation --out raw
python scripts/owed_ce_control.py      --bara  ../2026-07-26-bar-a-selector \
                                       --frozen ../2026-07-26-control-readjudication/raw/frozen_list.json --out raw
python scripts/owed_chance_baseline.py --h2c   "<hub>/Architecture & Inference/.../2026-07-26-h2-classifier" \
                                       --scores raw/h2clf_scores --out raw
```
