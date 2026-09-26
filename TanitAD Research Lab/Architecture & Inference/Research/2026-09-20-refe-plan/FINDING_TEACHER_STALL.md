# A measured failure mode in the released DriveZero teacher — ⛔ LARGELY RETRACTED, see §0

## READ THIS FIRST — what this document actually establishes, after two retractions

This file grew through a refutation and a retraction and is now 13 sections. A reader should not have
to reconstruct the verdict from the argument. **Status of every claim made here, in one table:**

| claim | status | where |
|---|---|---|
| The teacher fails to resume from a stop | ⛔ **RETRACTED** — non-reactive replay artifact; reactive shows 0 % stalled, 97.60 | §0 |
| REFe would inherit that resume defect | ⛔ **WITHDRAWN** with it | §5 marker |
| The teacher is generally slow | ⛔ **RETRACTED** earlier the same day — it out-travels the human in 4 of 8 scenarios | `RESULT.md` |
| The critic is flat below 2 m/s | ⚠️ **measurement stands, interpretation weakened** — lost its tie-breaker | §9 marker |
| More search ⇒ lower closed-loop score | ✅ **holds, and is stronger under the clean protocol** | §12, §13 |
| Selection is biased toward braking | ✅ **CONFIRMED both protocols** — pool flat in N, selection deepens −1.53→−2.37 | §13.4 |
| …and the pool is neutral, so selection is the whole story | ⚠️ **partly lifted** — reactive pool is offset but FLAT in N, so the N-dependence is still selection | §13.4 |
| Which scenario takes the damage | ⛔ **protocol-dependent** — left turn vs multiple-vehicles | §13.1 |
| Our numbers reproduce DriveZero's published TTS gain | ⛔ **they do not** (+0.11→+0.56 published; ours peaks then falls) | §12 |
| Any of this is a verdict on DriveZero | ⛔ **NO** — 8 scenarios, 1 seed, 6 of 9 terms saturated | §12 |

⭐ **The one thing worth carrying out of this document** is not a result about their teacher. It is
that **the background-agent protocol moved a single scenario by 10.13 points while the entire
N = 1 → 64 search sweep moved the suite by 0.16.** The lever nobody was varying dominated the lever
everybody was measuring. ⇒ every per-scenario number in this programme carries its protocol, and the
REFe-vs-teacher comparison must hold it fixed.

⚠️ **Two process lessons, both earned expensively here:**
1. **A named, queued control is a BLOCKER, not a footnote.** §4 named the confound that killed the
   headline and called its control "queued"; eleven sections were then built before it ran. It cost
   one 15-minute job.
2. **Print the denominator and build the discriminating control.** The instrument bug in §7 produced
   a *correct-looking* mean on a wrong n; the §11 probe is true by construction and needed the
   outcome to mean anything; the §12 pool column is the only reason the braking claim is sayable.

---
## 0 · ⛔⛔ RETRACTION (2026-09-20 13:1x) — the stall is a NON-REACTIVE REPLAY ARTIFACT, not a policy defect

**The control this document itself queued in §4 has now run, and it refutes the headline finding.**
Same token, same checkpoint, same 8-scenario selection; the ONLY change is the background-agent
policy (log replay → IDM):

| protocol | distance driven | stalled steps | final speed | score | progress |
|---|---|---|---|---|---|
| non-reactive (log replay) | 39.5 m | **36 %** | 0.03 m/s | 87.46 | 0.6129 |
| **reactive (IDM)** | **59.8 m** | **0 %** | **3.90 m/s** | **97.60** | **0.9256** |

⇒ **Under reactive agents the teacher does not stall at all.** Zero stalled steps, and the scenario
goes from the worst of the eight to unremarkable. The `changing_lane_to_left` stall **cannot be
reported as a failure to resume in the DriveRL policy.**

### What actually happened, stated as narrowly as the evidence allows

In log-replay the background vehicles execute their recorded trajectories **regardless of the ego**.
The teacher was driven into a state — a lead at **0.5 m** — that reactive traffic never creates,
because an IDM agent yields. §2 eliminated *"blocked by a lead"* on the grounds that the gap later
grew to 16.3 m, and that elimination was **correct about the gap and wrong about the cause**: the
trap was the *arrival* at a 0.5 m gap, not the persistence of one.

⚠️ **What survives, narrowly:** once in that state — stopped, clear road, valid goal 30 m ahead — the
policy did not recover, and the critic could not rank an escape (§8). The **state** is real and the
measurements on it are real. But it is a state reachable essentially only under log replay, so its
practical weight for REFe is small, and the §5 claim that REFe would inherit a resume defect is
**withdrawn**.

### ⛔ What this retraction does NOT touch

§10–§12 (the TTS sweep, the braking-biased selection, the monotone score decline) do **not** rest on
the stall. The scenario carrying that decline, `starting_left_turn`, has **0 % stalled steps in both**
protocols. Those sections stand on their own — though they were all measured non-reactive, so a
reactive replication is running now and will be scored here.

### ⚠️ §9 loses its tie-breaker and must be re-read

§9 argued the critic is flat below 2 m/s and explicitly used this scenario to break the tie against
*"low-speed states in this sample mostly have no better action"*. **That tie-breaker is gone.** The
flatness measurement stands as a measurement; its *interpretation* reverts to the confound §9 itself
named. Do not quote §9 as evidence that the critic is blind where a better action existed.

### ⭐⭐ The constructive payoff: the PROTOCOL is a bigger lever than anything we were measuring

`raw/protocol_sensitivity.txt` — same checkpoint, same 8 tokens, no TTS; **only** log-replay vs IDM:

| scenario type | non-reactive | reactive | delta | progress nr → r |
|---|---|---|---|---|
| changing_lane_to_left | 87.46 | 97.60 | **+10.13** | 0.613 → 0.926 |
| following_lane_with_lead | 99.96 | 92.96 | **−7.00** | 1.000 → 0.775 |
| near_multiple_vehicles | 99.92 | 97.27 | −2.65 | 1.000 → 0.913 |
| starting_left_turn | 96.61 | 95.19 | −1.42 | 0.956 → 0.890 |
| starting_protected_noncross_turn | 99.23 | 99.91 | +0.68 | 1.000 → 1.000 |
| accelerating_at_traffic_light_without_lead | 99.74 | 99.99 | +0.25 | 1.000 → 1.000 |
| starting_unprotected_cross_turn | 94.61 | 94.76 | +0.15 | 0.934 → 0.931 |
| stopping_at_stop_sign_with_lead | 100.00 | 100.00 | +0.00 | 1.000 → 1.000 |
| **SUITE** | **97.19** | **97.21** | **+0.02** | |

⭐ **Two facts that together indict every per-scenario claim made from one protocol:**
1. **The suite moves by +0.02 while individual scenarios move by up to ±10 points.** The protocol
   effects very nearly cancel in the aggregate — the mean hides the entire phenomenon, which is the
   same aggregation trap as the earlier "teacher is slow" retraction, one level up.
2. **The largest protocol effect is +10.13 points on a single scenario. The ENTIRE TTS sweep
   (N = 1 → 64) moved the suite by −0.16, and its largest per-scenario effect was −1.67.** ⇒ **the
   background-agent policy is a ~6× larger per-scenario lever than the variable I was sweeping.**

⚠️ **Neither protocol is "the right one".** They stress different things, and the two extremes are
mirror images: log replay traps the ego behind vehicles that ignore it (`changing_lane_to_left`,
+10.13 when fixed), while IDM holds the ego behind a lead that brakes for it
(`following_lane_with_lead`, **−7.00**, progress 1.000 → 0.775). ⇒ **report both, always**, which is
exactly why their headline suite pairs `_nr` and `_r` for every task — a design choice whose reason
is now measured rather than assumed.

⛔ **Binding consequence for DZ-11 and for REFe:** no per-scenario number from this programme is
quotable without its background-agent protocol attached, and any REFe-vs-teacher comparison must hold
the protocol fixed. A student trained on **non-reactive** teacher rollouts is learning from
demonstrations that include traps reactive traffic never produces.
### ⭐ Root-cause class, for the retraction log

**A benchmark-protocol artifact read as a property of the system under test.** The confound was
named in §4 and the control was queued — then eleven sections were written on top of the finding
**before the control ran**. ⇒ *When a document names a confound and queues its control, the control
is a BLOCKER on the headline claim, not a footnote.* Run it before building on the result, not after.
⚠️ This is the `df` / cgroup / `step_s` family again, with the wrong scope being the **evaluation
protocol** rather than units, host or aggregation.

---

# (Original document follows, retained unedited below this line for the record)

# A measured failure mode in the released DriveZero teacher — and the pre-registered test for it

**Date:** 2026-09-20 · **Evidence class: MEASURED (ours)** · **n = 1 scenario — read §4 before quoting**
**Artifacts:** `C:/dzo/m-nr-n/.../changing_lane_to_left/b2a5c363d1dd5abe.msgpack.xz`,
`media/teacher_mini_b2a5c363d1dd5abe.mp4`, probe `code/lane_change_probe.py`.

## 1 · What was observed

⛔ **RETRACTED — see §0.** The scenario does not stall under the reactive protocol (59.8 m, 0 % stalled, 97.60). The numbers below are correct **as non-reactive measurements** and wrong as statements about the policy.

`changing_lane_to_left` (token `b2a5c363d1dd5abe`) is the **worst scenario of the eight** in Stage 0:
score **0.8746** against a suite mean of 0.9719, and its deficit is almost entirely one term —
`ego_progress_along_expert_route` = **0.6129**, where every other scenario reads ≥ 0.934.

The ego stops at **t = 19.2 s** (step 96) after **38.1 m** and travels **1.4 m** in the remaining
**10.6 s**. It ends at **39.5 m** against the human expert's **64.0 m**.

⭐ **Our probe computes the path-length ratio at 0.617 from the raw poses; their metric engine
computes 0.6129 from its own pipeline.** Two independent derivations agreeing is what makes the
number admissible — re-running their own computation would have measured determinism, not truth.

## 2 · Four explanations, each eliminated by measurement

⛔ **RETRACTED — see §0.** A fifth explanation was not on this list and is the true one: **the non-reactive protocol itself.** The "blocked by a lead" row is right about the gap and wrong about the cause — the trap was ARRIVING at 0.5 m, which only log replay creates.

| candidate | what would confirm it | MEASURED | verdict |
|---|---|---|---|
| blocked by a lead | front gap stays small | gap **grows 0.5 → 16.3 m** during the stall | ⛔ eliminated |
| reached its goal | goal distance → 0 | goals locked at **30.0 m / 60.0 m**, lateral ≈ 0, every stalled step | ⛔ eliminated |
| red light | a RED status on the ego's connector | **no traffic lights in the scenario at all** — zero statuses at every iteration | ⛔ eliminated |
| crossing traffic at a junction | several agents, large relative headings | **one** agent, relative heading **−1.6°** (an aligned lead), `in_intersection=False`, `in_lane=True` throughout | ⛔ eliminated |

⇒ **The teacher brakes to a stop essentially touching its lead (0.5 m), and then does not resume
even as that lead pulls 16 m away, with a clear lane, no light, and a valid in-lane goal 30 m ahead.**
The braking looks correct. The **not-resuming** is the anomaly.

⚠️ **Why the goal cannot rescue it, and why that is interesting on its own.** `route_goal_positions`
sets the far goal at `current_progress + max(‖v‖, min_speed_mps) · horizon_s`. With the released
`min_speed_mps = 5.0` and `horizon_s = 12.0`, a **stopped** ego still sees a goal exactly 60 m ahead,
regenerated every step. The goal input is therefore **identical whether the ego is cruising at 5 m/s
or stationary** — it is progress-relative, so it carries no signal that the car has stalled.

## 3 · ⛔ Pre-registration — written BEFORE the sweep results land (2026-09-20 12:2x, sweep running since 12:19:06)

⚠️ **Registered against a phenomenon that turned out to be a protocol artifact (§0).** Retained verbatim because it was written in advance and its scoring in §6/§10 is part of the record.

The DZ-10 value-guided TTS sweep already running is the discriminating experiment, because at
`tts=0` the policy executes the Beta **mode** (deterministic — the logged `raw_action` repeats its
first two components exactly), while TTS **samples N candidates and selects by the critic's value**.

* **If the stall is a mode artifact** — the mode action at that state is "stay stopped" while better
  actions exist under the same policy — then TTS should escape it, and
  **`changing_lane_to_left` should show the LARGEST per-scenario gain of the eight**, rising toward
  the other scenarios' ≥ 0.93 progress.
* **If the stall is a value/critic artifact** — the critic prefers stopping there — then **TTS will
  not help on this token**, because every candidate is ranked by the same critic. Its per-scenario
  delta should sit near zero while other scenarios move.

Both outcomes are committed in advance. **Either result is informative**, and the second is the more
consequential one for REFe: a teacher whose *critic* prefers stopping would propagate that preference
into any REFe student distilled from its rollouts or scored by its value head.

⚠️ **What would NOT count as confirmation:** a gain on the suite mean. The claim is about *this
token*, and the sweep is paired per scenario precisely so the suite mean cannot stand in for it.

⛔ **POWER CAVEAT, registered before the results — this sample can barely move.** MEASURED from the
aggregator: **six of the nine score terms read exactly 1.0000 on all eight scenarios**
(`drivable_area_compliance`, `driving_direction_compliance`, `no_ego_at_fault_collisions`,
`time_to_collision_within_bound`, `ego_is_comfortable`, `ego_is_making_progress`). Only
`ego_progress_along_expert_route` (0.938) and `speed_limit_compliance` (0.965) vary at all.
⇒ **the entire headroom on this sample is ~2.8 points, and it is almost all the one stall.**
Their published TTS gain (+0.11 at N=8 → +0.56 at N=64) was earned on a benchmark where collisions
and drivable-area violations actually occur; **here those terms cannot improve, because they are
already perfect.** A null result on the suite mean is therefore expected and is NOT evidence against
TTS. The only informative quantity is the **paired per-scenario delta on `changing_lane_to_left`**.

⚠️ A detail worth keeping: **`ego_is_making_progress` reads 1.0000 for the stalling scenario too.**
The binary progress gate does not fire on a 39 % progress loss — only the continuous term does. If
REFe ever uses a binary progress check as a filter, it will pass exactly the frames we most want
caught.

## 4 · ⛔ Scope — what this is not

⭐ **This section named the confound that killed the finding — and the control it calls "queued" is the one that refuted it (§0).** Retained unedited as the primary evidence for the process lesson: a named, queued control is a BLOCKER, not a footnote.

* **n = 1 scenario.** One token, one protocol (non-reactive, log-replay background), one inference
  path. This is **an instance, not a characterisation**. It is not "the teacher has a resume defect".

  ⭐ **SHARPENED by a census over all eight scenarios** (`code/stall_census.py`, CPU-only,
  `raw/stall_census_m-nr-n.txt`): **4 of 8 scenarios stall**, but three of those stalls are
  **correct** — `stopping_at_stop_sign_with_lead` stalls 86 % of its steps and scores **1.0000**,
  while `accelerating_at_traffic_light_without_lead` and `following_lane_with_lead` stall only at
  the start (last stalled step 23 and 13) and finish at 13 m/s. ⇒ **the discriminator is not
  "stalls" but "stalls and never resumes with a clear road", and that is 1 of 8.** The census also
  refutes a reading I had written from the suite mean: the teacher travels **further than the human
  expert in 4 of 8** scenarios and ≥ 0.91 in 6, so it is **not** a generally slow policy — the whole
  suite progress deficit is this one token.
* **No interval, and none is earned.** On the programme's own rule a separated CI here would answer
  *"would another draw of scenarios say this?"* — which is not the question. The replicate that
  matters varies the **inference** path, and the TTS arms are exactly that.
* **Non-reactive replay is a real confound in general** — a fallen-behind ego can sit in a stream of
  replayed traffic. ⭐ It is **excluded here specifically**: one aligned lead, not a stream. The
  reactive protocol (`val14_r`, IDM background) on the same token remains the clean cross-check and
  is queued.

## 5 · Why this matters to REFe

⛔⛔ **WITHDRAWN ENTIRELY — see §0.** There is no resume defect to inherit. ⚠️ The *replacement* consequence for REFe is real and is in §0's payoff section: a student trained on **non-reactive** teacher rollouts learns from traps reactive traffic never produces, so Stage 2 must fix its protocol and say which.

REFe's supervision comes from this checkpoint. A teacher that stalls for a third of an episode
produces **stopped-ego targets** for those frames, and a student trained on them inherits the
behaviour without ever seeing why. ⇒ Stage 2's target builder needs a **stall filter** — or, better,
the stalled frames must be *kept* and labelled, so the defect is measurable in the student rather
than silently absorbed. That is a Stage-2 design decision this finding just forced, and it was not
in the plan before today.

## 6 · Pre-registration outcome at N = 8 — HALF confirmed, and the refuted half is the informative one

**MEASURED 2026-09-20 12:30, paired on the 8 identical tokens** (`raw/tts_paired_n8.txt`):

| scenario type | N=1 | N=8 | delta | progress N=1 → N=8 |
|---|---|---|---|---|
| **changing_lane_to_left** | 87.46 | 87.79 | **+0.33** | 0.613 → **0.625** |
| accelerating_at_traffic_light_without_lead | 99.74 | 99.97 | +0.23 | 1.000 → 1.000 |
| starting_left_turn | 96.61 | 96.81 | +0.20 | 0.956 → 0.962 |
| near_multiple_vehicles | 99.92 | 100.00 | +0.07 | 1.000 → 1.000 |
| the other four | – | – | **+0.00** | unchanged |

✅ **Confirmed (direction):** `changing_lane_to_left` shows the **largest** per-scenario gain of the
eight, as registered. All eight deltas are **≥ 0**, which is the behaviour their conservative
switch-margin rule guarantees — a control that must read a known value, and it does.

⛔ **REFUTED (magnitude), and this was committed in advance:** the registration required progress to
rise **toward the other scenarios' ≥ 0.93**. It moved **0.613 → 0.625**, i.e. **+0.012**. The
scenario still scores 87.79 against a suite of ~99. **Value-guided search over 8 candidates touches
the stall and does not escape it.**

⇒ The verdict at N = 8 sits closer to the registration's **second** branch: the stall is not merely a
mode artifact that any sampled alternative fixes. Two live explanations remain, and the sweep
already running discriminates them:
1. **search budget** — 8 candidates is too few; N = 16/32/64 should then escape it;
2. **the critic prefers the stopped state** — every candidate is ranked by the same value head, so
   more candidates will not help, and the delta should plateau.

⭐ **If (2) holds it is the consequential result for REFe**, because REFe's supervision and any
value-guided selection would inherit that preference from this same critic.

⚠️ **A coincidence that must NOT be reported as a reproduction:** our suite moved **+0.11**
(97.19 → 97.30) and their published N = 8 gain is also **+0.11**. Different benchmark, different
tokens, n = 8, and six of nine score terms pinned at 1.0 here. The agreement is **numerology**, not
a replication, and is recorded only so nobody later mistakes it for one.

## 7 · The mechanism behind the small N=8 effect: TTS almost never fires

**MEASURED** from the per-step candidate tables persisted in every sample
(`trajectory.debug_info.model_input.test_time_scaling`), over the whole N = 8 run:

| quantity | value |
|---|---|
| switch rate (steps executing a sampled candidate rather than the Beta mode) | **2.01 %** |
| mean score margin when it did switch | 0.0586 |
| switches below the configured margin | **0** |

The last row is a control that must read a known value: their conservative rule executes a sampled
candidate only when it beats the mode by at least the configured margin, so any switch below it
would be a harness error. It reads exactly **0**.

⇒ **"TTS at N = 8 did not escape the stall" is substantially "TTS barely acted at all".** On 98 % of
steps the executed action was the mode, i.e. identical to the N = 1 arm. This does **not** rescue the
refuted half of §6 — the stall still survived — but it sharpens which of the two remaining
explanations is live: a 2 % firing rate is consistent with **too few candidates clearing the margin**
(branch 1) and gives the larger N values real room to change the answer.

⛔ **An instrument bug found while validating the analyser on already-banked data, recorded because
its value looked CORRECT.** `per_scenario_scores` keyed on every row that was not `final_score`, but
the aggregator parquet holds **three** row kinds: 8 per-scenario rows (`log_name` set), 8 per-**type**
aggregate rows (`log_name` null), and `final_score`. The paired comparison therefore ran on
**n = 16** for an 8-scenario run. On mini the per-type rows *equal* the per-scenario rows (one
scenario per type), so the mean delta came out **+0.10 either way** and nothing looked wrong —
**only the printed `n` exposed it.** On the six-task DZ-11 suite the per-type rows are averages over
many scenarios, and mixing them in would have silently corrupted every paired delta.
⭐ Two lessons, both already in the programme's own rules: **print the denominator**, and **validate
an instrument on banked data before the run it is meant to read** — this cost zero GPU because the
N = 1 and N = 8 arms were already on disk.

## 8 · RESOLVED at N = 16 — the critic is not biased toward stopping, it is BLIND there

⚠️ **Measurements stand; the framing does not (§0).** TTS genuinely never fired across 106 stalled steps, and the candidate/value numbers are real — but they describe a state reachable essentially only under log replay, so this is not a deployable-behaviour claim.

### The stall is bit-for-bit unmoved by test-time search

| arm | path (m) | stalled | final speed | stall onset | **switches during the stall** |
|---|---|---|---|---|---|
| N = 1 | 39.5 | 36 % | 0.03 m/s | step 96 | n/a (no TTS) |
| N = 8 | 40.3 | 35 % | 0.03 m/s | step 97 | **0 / 52** |
| N = 16 | 39.7 | 36 % | 0.04 m/s | step 95 | **0 / 54** |

Not *"search tried and failed"* — **search never fired once across 106 stalled steps in two arms.**
Meanwhile TTS demonstrably works elsewhere in the same run: `starting_left_turn` +0.20 → +0.32 and
`accelerating_at_traffic_light_without_lead` +0.23 → +0.26 as N goes 8 → 16.

### Why it never fires, discriminated by measurement

Two explanations were live: **(a)** the critic prefers the stopped state, or **(b)** the policy's Beta
has collapsed there, so all N samples are near-copies of the mode and none *can* score better. The
persisted candidate tables settle it. At N = 16 on this token:

| | stalled steps (54) | moving steps (78) |
|---|---|---|
| candidate action spread, mean `max abs(a_k − a_mode)` | **3.59** | 4.56 |
| best-minus-mode advantage, mean | +0.00265 | +0.01168 |
| best-minus-mode advantage, **max** | **+0.01538** | **+0.40630** |
| advantage > 0 | 85 % of steps | 96 % of steps |
| switch margin required | **0.03** | 0.03 |

⛔ **(b) is refuted:** the candidates are *diverse* while stalled (spread 3.59 on a jerk axis bounded
at [−8, +5]), essentially as diverse as while moving. The policy is not collapsed.

⭐ **The real mechanism: the VALUE FUNCTION IS FLAT at the stalled state.** Sixteen genuinely
different actions — including hard accelerations — separate by at most **0.0154** of return, while
the same critic at moving states spreads candidates by up to **0.4063**, a **26× larger** range.
The conservative switch rule needs an advantage above **0.03**, so **the best candidate at a stalled
step never reaches even half the bar.** Zero switches is not bad luck; it is arithmetic.

⇒ **The critic is not biased toward stopping. It is nearly INDIFFERENT** — it cannot tell that
driving away is better than sitting still, by a margin that would matter. That is a sharper and worse
finding than the one registered in §3, because indifference cannot be fixed by sampling harder.

### Provenance of the two load-bearing assumptions (verified, not assumed)

* **index 0 is the mode:** `candidate_sources = ('argmax', 'sample_1', …, 'sample_15')`, read from the
  log itself.
* **the rule is strict-greater against 0.03:** `tts.py:181` sets `total_return_switch_margin`,
  `tts.py:387` is `if bool(advantage > self.total_return_switch_margin)`. A logged step confirms the
  behaviour end to end: scores `[1.10926, 1.11231, 1.11160, 1.11983, …]`, `argmax = 3`, advantage
  **0.01057 < 0.03**, `selected_candidate = 0` — the mode kept.

### ⛔ Pre-registration for N = 32 and N = 64, written now, before those runs finish

If the mechanism above is right, **the bottleneck is the value gap, not the candidate pool**, so:
1. **switches during the stall stay 0** at both N = 32 and N = 64;
2. **max best-minus-mode advantage at stalled steps stays below 0.03**;
3. the stall token's score stays ≈ 87.5 while other scenarios keep improving.

⛔ **What would REFUTE it:** any switch during the stall, or a stalled-step advantage above 0.03.
That would mean the candidate pool *was* the binding constraint and the value signal is merely small,
not absent — a materially better situation for REFe.

### Why this is the session's most consequential result for REFe

REFe plans to take training signal from this checkpoint and, per the plan, may use value-guided
selection. **A critic that is flat exactly where the policy is stuck cannot supervise an escape from
being stuck**, and a student distilled from these rollouts sees 36 % of this episode as "stopped is
correct". ⇒ Stage 2's target builder must treat stalled frames explicitly, and any REFe selection
head must be evaluated *at low speed specifically*, where this critic carries almost no signal.

## 9 · GENERALISED from n = 1 token to the whole sample: the critic goes flat below 2 m/s

⚠️ **LOSES ITS TIE-BREAKER (§0).** This section used `changing_lane_to_left` to rule out "low-speed states here mostly have no better action". That argument is gone. The flatness measurement stands; the interpretation reverts to the confound this section itself named.

§8 was one scenario. `code/critic_flatness.py` runs the same measurement over **every** TTS step of
the N = 16 run — **1,193 steps across all 8 scenarios** — and buckets the advantage by ego speed.
`candidate_sources[0] == 'argmax'` was asserted on all 1,193 steps, so the advantage is well defined
(`raw/critic_flatness_m-nr-16.txt`).

| ego speed | steps | mean advantage | max advantage | above margin | switched | action spread |
|---|---|---|---|---|---|---|
| 0.0 – 0.5 m/s | 190 | 0.00156 | 0.02189 | **0.0 %** | **0.0 %** | 4.466 |
| 0.5 – 1.0 m/s | 30 | 0.00333 | 0.01538 | **0.0 %** | **0.0 %** | 3.749 |
| 1.0 – 2.0 m/s | 30 | 0.00367 | 0.01817 | **0.0 %** | **0.0 %** | 2.949 |
| 2.0 – 5.0 m/s | 321 | 0.00551 | 0.40630 | 1.6 % | 1.6 % | 4.403 |
| 5.0 – 10.0 m/s | 320 | 0.00800 | 0.04872 | 3.8 % | 3.8 % | 4.175 |
| 10.0 + m/s | 302 | 0.00899 | 0.13036 | 5.0 % | 5.0 % | 4.225 |

⭐ **Below 2 m/s — 250 steps spread across all eight scenarios — the best of sixteen candidates never
once reaches the 0.03 switch margin (its ceiling is 0.0219), and TTS fires exactly zero times.**
Value-guided test-time search is **structurally inert at low speed on this checkpoint**, and that is
a property of the released artifact, not of one unlucky token.

**Two clean monotone trends, stated as what they are:**
* **mean** advantage rises with speed, 0.00156 → 0.00899, a **5.8×** span;
* **switch rate** rises with speed, 0 / 0 / 0 → 1.6 / 3.8 / 5.0 %.

⚠️ **The MAX advantage is NOT monotone** — the 2–5 m/s bucket holds the largest single value
(0.40630), above both faster buckets. So the honest claim is a **threshold at ~2 m/s**, not a smooth
gradient: below it the ceiling is 0.022, above it single steps reach 0.41. Reporting "advantage rises
with speed" without that caveat would overstate a noisy tail as a trend.

⭐ **The discriminating control holds sample-wide.** Candidate **action spread is flat across every
speed bucket** (4.47 / 3.75 / 2.95 / 4.40 / 4.18 / 4.23). A collapsed policy would show a small
spread exactly where the advantage vanishes. It does not. ⇒ the diversity is there and the **critic**
cannot rank it — the same conclusion as §8, now on 250 low-speed steps instead of 54.

⛔ **Scope.** One run, one protocol (non-reactive), one checkpoint, N = 16. The low-speed buckets are
dominated by two scenarios that legitimately stop (`stopping_at_stop_sign_with_lead`,
`changing_lane_to_left`), so "the critic is flat below 2 m/s" is measured **on the states this sample
visits**, not proven over the state space. It is a strong lead, not a theorem.

⚠️ **And a genuine confound to name rather than bury:** at a state where stopping really is correct
(a stop sign), a flat value gap is the **right** answer — there is nothing better to do. This
measurement cannot by itself separate "the critic is blind at low speed" from "low-speed states in
this sample mostly have no better action". ⭐ The `changing_lane_to_left` case in §8 is what breaks
the tie, because there a better action demonstrably existed: the human expert drove 64.0 m while the
ego managed 39.5 m on a clear road.

## 10 · N = 32 REFUTES my §8 pre-registration — and my registered READING of a refutation was also wrong

### The prediction, and what happened

§8 registered three things. Scoring them honestly (`raw/tts_sweep_N1_8_16_32.txt`):

| registered | outcome at N = 32 | verdict |
|---|---|---|
| switches during the stall stay **0** | **1** switch | ⛔ **REFUTED** |
| stalled-step max advantage stays **below 0.03** | **1.93402** | ⛔ **REFUTED** |
| stall score stays ≈ 87.5 while others keep improving | 87.40, but **two scenarios REGRESSED** | ⚠️ half right |

⛔ **And the part I must not quietly rewrite:** I registered that a refutation *"would mean the
candidate pool was the binding constraint and the value signal is merely small, not absent — a
materially better situation for REFe"*. **That reading is not supported by the data that refuted it.**
The single stall switch rode an advantage of **1.93402** where the step-wise mean is **0.00510**
— roughly **380×** typical — and the scenario got **worse** (87.54 → 87.40). That is not a critic
finding real signal. It is a critic emitting a wild outlier that cleared a 0.03 bar.

### The account that does fit: selection on estimation noise

Across the **same 1,193 states** with the **same critic**, only N changing:

| N | switch rate | mean advantage | max advantage | mean paired score delta |
|---|---|---|---|---|
| 8 | 2.01 % | 0.00510 | 0.31146 | **+0.105** |
| 16 | 2.68 % | 0.00633 | 0.40630 | **+0.091** |
| 32 | **4.78 %** | **0.01022** | **1.93402** | **−0.071** |

⭐ **If the critic's estimates were accurate, drawing more candidates would CONVERGE** on the best
available action and the measured advantage would saturate. Instead the advantage **keeps climbing**
— 2× from N = 8 to N = 32 on identical states — while the closed-loop score **degrades** and
regressions appear for the first time (0, 0, then 2 scenarios). That is the signature of a
**maximisation bias**: the expected maximum of N noisy estimates grows with N even when the true
values are equal, so a larger pool buys more chances to select a candidate whose value was
**over-estimated**, not a better candidate.

⇒ **This unifies with §9 rather than replacing it.** The critic's genuine value gaps (mean 0.003–0.009)
are the **same order as the 0.03 switch margin**, so clearing the bar is dominated by estimation
error. At small N that rarely happens and TTS is nearly inert; at large N it happens often and the
selections are disproportionately errors. Their published monotone gain (+0.11 → +0.56) must come
from a regime where true value gaps are large relative to that noise — which is precisely what our
8-scenario sample is not, with six of nine score terms already pinned at 1.0.

### ⛔ Falsifiable prediction for N = 64, registered before it finishes

1. switch rate **above 4.78 %**;
2. mean advantage **above 0.01022**;
3. suite score **does not recover** to N = 8's 97.30.

**Refuted if** N = 64 recovers above 97.30 *and* the mean advantage saturates near N = 32's value.

### ⚠️ Two different power levels in one section — do not mix them

* The **mechanism** numbers (switch rate, advantage distribution) come from **1,193 steps per arm** and
  are well powered. The monotone rise in switch rate and mean advantage is solid.
* The **outcome** numbers (score deltas) come from **8 scenarios** with six of nine terms saturated.
  **−0.071 is not a demonstration that TTS hurts.** It is one sample, one seed, no replicate. The
  claim this section supports is about *how selection behaves*, not about a benchmark delta.
⛔ Nothing here is comparable to DriveZero Table 3, and the regression must not be reported as
"TTS makes DriveZero worse".

## 11 · Within-step probe of the pool size — and why it is NOT self-evidence

§10 compared different runs, where a larger N also means different sampled actions and different
resulting trajectories. `code/pool_size_bias.py` removes that confound: it takes the **N = 32 run's
own candidate table** and recomputes the advantage using only the first *k* candidates. **Same 1,193
states, same critic, same candidates — only how many you may maximise over changes.**

| pool k | mean adv | median adv | p90 adv | implied switch rate |
|---|---|---|---|---|
| 2 | 0.00223 | **0.00000** | 0.00505 | 1.51 % |
| 4 | 0.00413 | 0.00109 | 0.00921 | 2.43 % |
| 8 | 0.00587 | 0.00237 | 0.01267 | 3.19 % |
| 16 | 0.00720 | 0.00350 | 0.01444 | 4.02 % |
| 32 | 0.01022 | 0.00435 | 0.01694 | **4.78 %** |

⭐ **Control reading a known value:** at k = 32 the implied switch rate is **4.78 %**, which is
*exactly* the switch rate measured independently from the run's own `selected_candidate` field in
§10. The subsampling reproduces their actual computation at k = N, as it must.

⚠️ **The detail worth keeping:** at k = 2 the **median advantage is exactly 0.00000** — on more than
half of all steps a single sampled candidate does not beat the mode at all.

### ⛔ What this probe does NOT prove, stated because the statistic is true by construction

**`max(scores[:k])` is non-decreasing in k by definition.** The advantage *had* to rise. Quoting
"the advantage grows with pool size" as evidence of bias would be a check that shares the defect it
checks for — the programme has logged four of those in one night before, and this would be a fifth.

**What is actually informative is the SHAPE and the OUTCOME, and neither alone suffices:**
1. **No saturation.** If a genuinely better action existed, the max would rise steeply for small k
   and then **flatten**, because you would find that action early and further samples could not beat
   it. Observed p90 keeps climbing with no knee: 1.00 → 1.82 → 2.51 → 2.86 → **3.35**.
2. **Faster than Gaussian.** Against the max-of-k reference `sqrt(2 ln k)` (1.00 / 1.41 / 1.73 /
   2.00 / **2.24**), the observed p90 grows **3.35×**. Heavier-than-normal tails in the critic's
   error — the single **1.934** outlier at k = 32 is one instance of exactly that.
3. **The realised score FELL** (§10: +0.105 → +0.091 → −0.071, with 0, 0, then 2 regressions). This
   is the load-bearing half. A rising predicted advantage together with a falling achieved score is
   what makes it a **bias** rather than a gain.

⇒ The honest statement is the conjunction: **the predicted advantage grows without saturating, faster
than Gaussian, while the achieved outcome degrades.** Any one of those three on its own is weak.

## 12 · Sweep complete — the N = 64 prediction holds, and the causal chain closes with a control

### Registered prediction, scored

| registered for N = 64 | measured | verdict |
|---|---|---|
| switch rate above 4.78 % | **5.20 %** | ✅ |
| mean advantage above 0.01022 | **0.01028** | ✅ *but see below* |
| suite does not recover to 97.30 | **97.03** | ✅ |

⚠️ **Honest reading of the second row:** 0.01028 against 0.01022 is **+0.6 %** — the mean advantage
has **saturated** between N = 32 and N = 64, after a +61 % jump from N = 16 to N = 32. The prediction
passes on its letter and not on its spirit, and I am recording that rather than banking a tick. The
**p90** did keep rising (0.01694 → 0.01865), so the growth persists in the tail only.

### The full sweep

| | N=1 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| suite score | 97.19 | **97.30** | 97.28 | 97.12 | **97.03** |
| mean paired delta | – | +0.105 | +0.091 | −0.071 | **−0.163** |
| switch rate | – | 2.01 % | 2.68 % | 4.78 % | **5.20 %** |
| scenarios worse than N=1 | – | 0 | 0 | 2 | 1 |

⛔ **We do not reproduce their monotone gain** (+0.11 at N = 8 rising to +0.56 at N = 64). Our curve
peaks at N = 8 and declines monotonically thereafter.

### The mechanism, end to end, on the scenario that carries the decline

`starting_left_turn` loses **−1.67** by N = 64. **Every other score term is bit-identical across all
five arms** (speed limit 0.9195, drivable area / collisions / comfort / TTC / direction all 1.0000).
Only progress moves — and the behaviour behind it is unambiguous:

| arm | switches | distance driven | final speed | progress |
|---|---|---|---|---|
| N=1 | 0 | 149.1 m | 10.02 m/s | 0.9560 |
| N=8 | 11 | 149.9 m | 10.00 m/s | 0.9625 |
| N=16 | 15 | 150.5 m | 9.98 m/s | 0.9663 |
| N=32 | 37 | 144.7 m | 9.32 m/s | 0.9297 |
| N=64 | 40 | **140.7 m** | **8.39 m/s** | **0.9025** |

**More search ⇒ more overrides ⇒ the car drives slower and shorter.**

### ⭐ Why the overrides are slower — measured, with the control that makes it interpretable

`code/switch_direction.py` compares the **selected** candidate's `jerk_long` to the **mode's**, and —
the part that matters — the same statistic over the **non-selected candidates at the same steps**. If
the available pool were itself braking-skewed, a braking selection would say nothing about selection.

| run | switches | selected minus mode | **pool** minus mode | selected slower | **pool** slower |
|---|---|---|---|---|---|
| N=8 | 24 | −0.898 | −0.486 | 66.7 % | 58.3 % |
| N=16 | 32 | −0.658 | −0.255 | 56.2 % | 56.2 % |
| N=32 | 57 | −0.798 | **−0.008** | 59.6 % | **38.6 %** |
| N=64 | 62 | **−1.269** | **−0.011** | **74.2 %** | **41.9 %** |

At N = 32 and N = 64 the **pool is essentially unbiased** (mean −0.008 / −0.011 m/s³; fewer than half
its members are slower than the mode), while the **selection averages −0.80 and −1.27 m/s³** and is
slower on 60 % and 74 % of overrides. ⇒ **the critic's estimation error is not symmetric — it
systematically over-values braking**, and maximising over a larger pool harvests that bias.

⚠️ **Where the control does NOT isolate it:** at N = 8 and N = 16 the pool is itself braking-skewed
(−0.486 / −0.255, and at N = 16 the pool and the selection are slower at the *identical* 56.2 %). The
claim is therefore supported **at N = 32 and N = 64 only**. Quoting it for the small-N arms would be
reading a selection effect off a sampler effect.

### The complete chain

1. Genuine value gaps are ~0.005, the switch margin is **0.03** (§9).
2. Clearing the bar is therefore dominated by estimation error (§10–§11).
3. More candidates ⇒ more spurious clears ⇒ more overrides (24 → 62 overall; 0 → 40 on one scenario).
4. **That error over-values braking** (this section), so overrides are systematically slower.
5. The car drives shorter; **progress is the only term with headroom**, so the score falls.

⭐ This also explains §8 without a separate hypothesis: **a critic that over-values braking is exactly
a critic that cannot supervise an escape from a stall.**

### ⛔ What must not be claimed

8 scenarios, one seed, no replicate, non-reactive only, and six of nine score terms saturated. The
**mechanism** numbers rest on 1,193 steps per arm and are well powered; the **score** numbers do not.
⇒ *"Test-time search makes DriveZero worse"* is **NOT** supported. What is supported: **on this
sample we do not reproduce their monotone gain, and the selection statistics carry the signature of
noise-dominated, braking-biased selection.** The clean test is DZ-11 on val14/test14, where the
saturated terms have real headroom.

## 13 · REACTIVE REPLICATION of §10–§12 — the sweep result survives the protocol, with one real caveat

⭐ **This is the control §12 promised and did not have.** The entire non-reactive sweep is re-run with
IDM background agents. N = 64 is still running; N = 1/8/16/32 are complete.

### 1 · The score decline replicates, and is STRONGER under reactive

| mean paired delta vs N=1 | N=8 | N=16 | N=32 |
|---|---|---|---|
| non-reactive | **+0.105** | +0.091 | −0.071 |
| **reactive** | **−0.026** | **−0.086** | **−0.106** |

⇒ **Under reactive agents TTS never helps at any N.** The decline is monotone from the first arm, and
the small N = 8 gain that opened the non-reactive sweep does **not** reproduce. Two scenarios are
worse than N = 1 at *every* reactive arm (non-reactive had none until N = 32).

⚠️ **The damaged scenario is protocol-dependent.** Non-reactive it was `starting_left_turn` (−1.67 by
N = 64); reactive it is **`near_multiple_vehicles`** (−0.76 at N = 32) while `starting_left_turn`
moves only −0.10. ⇒ **the direction of the effect is robust; the identity of the victim is not.** Any
per-scenario attribution must name its protocol.

### 2 · The braking bias replicates at the powered arms

| run | switches | selected − mode | **pool** − mode | selected slower | **pool** slower |
|---|---|---|---|---|---|
| non-reactive N=32 | 57 | −0.798 | **−0.008** | 59.6 % | **38.6 %** |
| reactive N=8 | 14 | −1.532 | −0.328 | 64.3 % | 71.4 % |
| reactive N=16 | 34 | −1.710 | −0.575 | 76.5 % | 64.7 % |
| reactive N=32 | 33 | **−1.968** | −0.492 | **78.8 %** | 66.7 % |

At the powered reactive arms (34 and 33 switches) **both statistics agree in direction**: the selected
override is more braking than the pool by mean (−1.97 vs −0.49, a **4×** gap) and by count (78.8 % vs
66.7 %). The N = 8 arm remains internally contradictory and underpowered at 14 switches, exactly as
flagged in `raw/reactive_sweep_interim.txt` before these arms existed.

### 3 · ⛔ The caveat that changes the effect SIZE, and it is not small

**Under reactive the candidate POOL is itself substantially braking-skewed** (−0.49 mean, 66.7 % of
members slower), whereas under non-reactive at N = 32 the pool was **near-neutral** (−0.008, 38.6 %).
That is what made the non-reactive number so clean: a neutral pool and a braking selection isolate
the selection completely.

⇒ **Under reactive, part of the braking is inherited from the SAMPLER, not created by the SELECTION.**
A selection effect is still present on top (−1.97 against a −0.49 pool), but **the share attributable
to selection alone is smaller than the non-reactive figures implied**, and §12's phrasing —
*"the pool is essentially unbiased … the critic over-values braking"* — is true **only of the
non-reactive arm** and must not be quoted flatly.

⚠️ A plausible and unverified reason for the skew: IDM agents yield and brake, so the ego spends more
time car-following, where the policy's own distribution leans decelerative. **Stated as a hypothesis,
not a finding** — nothing here tests it.

### Verdict

| §12 claim | reactive verdict |
|---|---|
| more search ⇒ lower score | ✅ replicates, **stronger** (never positive at any N) |
| selection is braking-biased | ✅ replicates at the powered arms, both statistics agreeing |
| the pool is neutral, so selection is the whole story | ⛔ **non-reactive only** — reactive pool is −0.49 |
| the effect is carried by `starting_left_turn` | ⛔ **non-reactive only** — reactive it is `near_multiple_vehicles` |

### 4 · N = 64 landed — DZ-10 COMPLETE, and §13.3's caveat is now PARTLY LIFTED

**Full sweep, 10 arms, both protocols, paired on 8 identical tokens throughout**
(`raw/tts_sweep_BOTH_PROTOCOLS.txt`):

| | N=1 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| non-reactive suite | 97.19 | **97.30** | 97.28 | 97.12 | 97.03 |
| non-reactive mean delta | – | +0.105 | +0.091 | −0.071 | **−0.163** |
| **reactive suite** | 97.21 | 97.18 | 97.12 | 97.10 | **96.98** |
| **reactive mean delta** | – | −0.026 | −0.086 | −0.106 | **−0.227** |
| reactive scenarios worse than N=1 | – | 2 | 2 | 2 | **4 of 8** |

⇒ Under the clean protocol the decline is **monotone at every arm**, ends at **−0.227**, and by N = 64
**half the sample is worse than not searching at all**.

### ⭐ The caveat in §13.3 is partly lifted, and I am upgrading the claim as deliberately as I retracted others

§13.3 warned that the reactive pool is braking-skewed, so part of the braking might come from the
**sampler** rather than the **selection**. The N = 64 arm settles it, because the two move differently:

| reactive arm | switches | **selected** − mode | **pool** − mode | selected slower | pool slower |
|---|---|---|---|---|---|
| N=8 | 14 | −1.532 | −0.328 | 64.3 % | 71.4 % |
| N=16 | 34 | −1.710 | −0.575 | 76.5 % | 64.7 % |
| N=32 | 33 | −1.968 | −0.492 | 78.8 % | 66.7 % |
| N=64 | 46 | **−2.368** | −0.476 | **84.8 %** | 69.6 % |

⭐ **The POOL is flat in N** (−0.33, −0.58, −0.49, −0.48; 71.4 → 69.6 %) **while the SELECTION deepens
monotonically** (−1.53 → −1.71 → −1.97 → **−2.37**; 64.3 → 76.5 → 78.8 → **84.8 %**).

⇒ **A sampler effect cannot produce an N-dependence that the sampler itself does not have.** The
deepening is therefore attributable to the **selection**, and the braking bias is confirmed as a
selection phenomenon in **both** protocols. What remains true from §13.3 is only that the reactive
pool has a baseline offset, so the *absolute* selected-minus-mode figure overstates the selection's
own contribution; the *N-dependence* does not.

⚠️ **Scope, unchanged:** 8 scenarios, one seed, six of nine score terms saturated. This is a
well-powered statement about **how the selection rule behaves** (46–62 switches per powered arm,
1,193 steps) and a poorly-powered one about **benchmark scores**. It is still not a verdict on
DriveZero, and the clean score test remains DZ-11.

## 14 · The replicate this programme always lacks — and here it reads EXACTLY ZERO

Obtained free while running Stage 1's control: the same run id, the same flags, **two independent
launches 2.5 hours apart** (12:02:56 and 14:36:47), both parquets preserved side by side
(`raw/stage1_rank0_replicate.txt`).

| scenario type | launch 1 | launch 2 | diff |
|---|---|---|---|
| accelerating_at_traffic_light_without_lead | 99.7388 | 99.7388 | +0.000000 |
| changing_lane_to_left | 87.4620 | 87.4620 | +0.000000 |
| following_lane_with_lead | 99.9604 | 99.9604 | +0.000000 |
| near_multiple_vehicles | 99.9234 | 99.9234 | +0.000000 |
| starting_left_turn | 96.6120 | 96.6120 | +0.000000 |
| starting_protected_noncross_turn | 99.2296 | 99.2296 | +0.000000 |
| starting_unprotected_cross_turn | 94.6138 | 94.6138 | +0.000000 |
| stopping_at_stop_sign_with_lead | 100.0000 | 100.0000 | +0.000000 |

⭐ **The eval is BIT-DETERMINISTIC across launches: max |diff| = 0.000e+00.**

### Why this matters more than it looks

The programme's standing rule is that a separated interval is **necessary and not sufficient**,
because the bootstrap resamples episodes with the models fixed and is blind to run-to-run variance;
the sufficient form needs a **replicate arm**. Here the replicate exists and its floor is **exactly
zero**, so:

1. **Stage 1's rank-1 effect is attributable to the route override**, with no inference-variance
   confound to subtract. That is an unusually clean attribution.
2. ⭐ **It retroactively strengthens §10–§13.** The TTS deltas (+0.105 … −0.227) cannot be
   run-to-run noise, because this rig has none. The sweep's *direction* was never in doubt from
   noise; only its *power on 8 scenarios* was.

⚠️ **What zero does NOT cover, stated so it is not over-claimed:** the no-TTS arms execute the Beta
**mode** and are deterministic by nature; the TTS arms **sample**, and their determinism comes from a
**fixed `tts_seed=42`** shared by every arm. ⇒ the deltas are attributable to **N** rather than to
sampling luck, but **a different `tts_seed` could give different deltas**, and nothing here tests
that. The inference-seed replicate the programme's third variance rule asks for is still **not run**.

⚠️ ⛔ This also does **not** transfer to training. The programme has MEASURED that its own
`ddv2_rl` trainer is *not* reproducible across launches. **Evaluation here is deterministic;
training elsewhere is not.** Do not quote this row outside closed-loop eval of this checkpoint.
