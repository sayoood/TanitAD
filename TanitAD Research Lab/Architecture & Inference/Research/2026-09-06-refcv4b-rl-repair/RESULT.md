# RL reward repair — `progress` is REFUTED as the mechanism, `headway`'s ORDER STATISTIC is not, and an RL arm can no longer run the deliberate regression

**date:** 2026-09-06 · **stream:** RL reward-repair (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **pre-registrations:** `PREREG.md` (commit
`082606e02`) and `PREREG_HEADWAY.md` (commit `02f5dda65`), **both committed BEFORE the code
they test.**

---

## 0. The answer in one paragraph

The Master Mind ruled that `progress` flipping sign with conflict is a **defect in the
reward**, and that the repair must be tested by its own falsifiable consequence: if paying
the constant-velocity path for not slowing down is what makes a RATE and a MEAN disagree,
a corrected term must **shrink that divergence**. ⛔ **It does not.** The repair works on
the term — `progress`'s weighted mean gap stops flipping sign at every conflict rung
(+0.008251 → **+0.000000** at ≤1.5 s) — and the divergence **grows**: `SKEWSPLIT` rises at
all three conflict rungs and the rate at ≤2.0 s rises from 0.5482 to 0.5546. ⇒ **`progress`
is REFUTED as the mechanism**, and `PREREG.md` §3.4 had already named what to do next. So I
ran it. Reading `headway` at a **quantile instead of its single worst step** — it is an
`amin`, a MIN-OVER-N order statistic, the same family of defect as `fan_floor@k` read alone
— **does** shrink the divergence: `Σ DIV` **6 → 3 of 8 rungs** and `SKEWSPLIT` falls at
**all three** conflict rungs, while the term becomes **more** informative, not inert (spread
0.060303 → 0.099617 at ≤2.0 s). ⛔ Its third clause fails: the rate rises again, and a tie
control proves that rise is **real and not a scoring artefact** (tie fraction exactly
0.0000 at that rung). ⇒ ⛔ **P2-PRED FAILS as written on the conjunction, and its two
DIVERGENCE clauses hold** — which is the fourth independent line of evidence that
`G-REWARD`'s rate cannot see what the reward is doing. The next lever is a **magnitude-aware
statistic pre-registered on its own**, and that is a **PI / Master-Mind decision**, not
mine to take. Separately and independently: ⭐⭐ **an RL arm can no longer execute the
deliberate regression under the hypothesis' name** — the control-space sampler is wired, the
launcher REFUSES an undeclared or metre-space hypothesis arm (proved by mutation, 14/14),
and a MEASURED silent-no-op in `control_space.py`'s own `logp` is fixed and guarded. **B6 is
closed on a real 240 × 117 fan.**

⛔ **`G-REWARD` remains FAILED on its committed statistic (the RATE).** Nothing in this
package re-scores it, and every rate below is a **diagnostic of the reward's geometry**.

---

## 1. P1 — the `progress` repair and its pre-registered prediction

### 1.1 What was predicted, before the code changed

`PREREG.md` (commit `082606e02`, landed before `rewards.py` was touched) committed:

* **P1-PRED-A** *(necessary)* — the `progress` per-term weighted mean gap is **not
  positive** at ≤3.0 / ≤2.0 / ≤1.5 s;
* **P1-PRED-B** *(the test)* — the rate/mean **divergence shrinks**: `Σ DIV` below its
  measured baseline of **6 of 8** rungs **and** `SKEWSPLIT = median − mean` falls at ≥2 of
  the three conflict rungs **and** `rate(≤2.0 s)` falls from **0.5482**;
* ⛔ **and, in the same breath**: *"if A holds and B fails, `progress` was NOT the mechanism
  … the next candidate is `headway`, whose `amin` over steps is itself a min-over-N order
  statistic."*

### 1.2 The repair

`progress = min(along, ref_ach) / ref_free`, with
`ref_ach = clamp(min(ref_free, lead_x[-1] − (lead_len + t_star·v0)), min_ref)`, behind
`ctx["progress_lead_cap"]` (`"achievable"` = pre-registered · `"lead"` = the sensitivity
variant capping at the lead bound only · `"off"` = bit-identical legacy).

⛔ **COORDINATE DISCIPLINE KEPT, and TESTED rather than asserted.** Every quantity is a
POSITION query on the lead's recorded track or a scene scalar at t0. No closing rate enters
the RANK channel — `M84` measured that rate a clean null (**+0.0061**) against position's
**+0.4145 [+0.2018, +0.6120]** with a constant control at exactly **+0.000000**. The test
`test_cap_is_a_position_query_and_not_a_rate` perturbs the lead's **intermediate** samples
with its **endpoint held fixed** and requires the term to be **bit-identical**, and it
carries its own **discriminating control**: the same edit *must* move the closing rate and
a per-step reading, or the first assertion would be vacuous. TTC is untouched and remains a
**VETO**.

### 1.3 MEASURED — 6,089 windows / 73 episodes, episode-cluster bootstrap, `n_boot` 4,000

`progress` per-term weighted mean gap (`hold_v0 − human`; **positive = the trivial path is
paid more**):

| rung | cap OFF (baseline) | `achievable` | **`lead`** |
|---|---|---|---|
| all | −0.005600 | +0.002194 | **−0.007737** |
| ≤3.0 s | **+0.003593** | +0.002225 | **−0.001264** [−0.003364, +0.000910] |
| ≤2.5 s | **+0.004053** | +0.000941 | **−0.001618** |
| ≤2.0 s | **+0.006368** | +0.000290 | **+0.000112** [−0.000294, +0.000642] |
| ≤1.5 s | **+0.008251** | 0.000000 | **0.000000** [0.000000, 0.000000] |

⇒ ⭐ **P1-PRED-A HOLDS for `lead`** (not positive, or CI straddling 0, at all three conflict
rungs). It **FAILS for the pre-registered `achievable`** at ≤3.0 s (+0.002225, CI
[+0.001080, +0.003815], separated positive).

The divergence, on the same rows:

| | `Σ DIV` /8 | SKEWSPLIT ≤3.0 s | ≤2.5 s | ≤2.0 s | rate ≤2.0 s |
|---|---|---|---|---|---|
| cap OFF | **6** | 0.009561 | 0.011710 | 0.014943 | 0.5482 |
| `achievable` | 3 | 0.011376 | 0.015228 | 0.021475 | 0.5596 |
| `lead` | 6 | 0.014453 | 0.017456 | 0.021232 | 0.5546 |

⇒ ⛔ **P1-PRED-B FAILS for BOTH variants.** `SKEWSPLIT` **rises** at every conflict rung
under both, and the rate rises. `achievable`'s `Σ DIV` 6 → 3 is the only clause that moves
the right way, and the tie control below shows most of that is a scoring convention.

### 1.4 ⛔ THE VERDICT, reported as written

> **`progress` is REFUTED as the mechanism that generates the rate/mean asymmetry.**
> Removing its payment to the constant-velocity path — completely, to an exact 0.000000 at
> ≤1.5 s — leaves the divergence **larger**, not smaller. The sign flip was a **symptom**,
> not the generator.

⚠️ And a defect **in my own pre-registration**, found by implementing it and disclosed
rather than smoothed over: the pre-registered `min(ref_free, ref_lead)` carries a **second
cap at the free reference** that was not the stated intent (§2 said "inert without a lead").
It is why `achievable` fails P1-PRED-A at ≤3.0 s and why only **1,334 of 3,556** non-binding
windows reproduce cap-OFF exactly, against **3,444 of 3,556** for `lead`. Both columns are
reported; the pre-registered one is the headline.

---

## 2. ⭐⭐ P1 continued — the lever `PREREG.md` named IN ADVANCE, run in the same turn

`PREREG_HEADWAY.md` (`H-RL-HEADWAY-QUANTILE-1`, commit `02f5dda65`, **before `_headway`
changed**) fixed **Q = 0.25** in advance and committed all three clauses of **P2-PRED**.

**The mechanism.** `_headway` reduces the per-step time gaps with **`amin`** — a
MIN-OVER-N order statistic over the horizon's steps. A minimum fires on the single worst
step, so the term is **RARE AND LARGE**: the exact shape that makes a mean and a rate
disagree. ⭐ **MEASURED in a unit test, not argued** — moving three of five lead samples by
up to 5 m leaves `_headway` **bit-identical** while a quantile over the same gaps moves
(`test_headway_is_blind_to_this_perturbation_and_that_is_the_next_candidate`). It is the
same family as `fan_floor@k` read alone, which is why that instrument ships with
`rand_best@k`.

### 2.1 MEASURED — `headway_reduce="q0.25"`, `progress` untouched (ONE variable)

| | `Σ DIV` /8 | SKEWSPLIT ≤3.0 | ≤2.5 | ≤2.0 | rate ≤2.0 | `headway` gap ≤2.0 | `headway` spread ≤2.0 |
|---|---|---|---|---|---|---|---|
| cap OFF | **6** | 0.009561 | 0.011710 | 0.014943 | 0.5482 | −0.015899 | 0.060303 |
| ⭐ **`hq25`** | **3** | **0.006255** | **0.007939** | **0.011650** | 0.6003 | −0.012124 | **0.099617** |

* **P2-PRED clause 1** — `Σ DIV` **6 → 3**: ⭐ **HELD**.
* **P2-PRED clause 2** — `SKEWSPLIT` falls at ≥2 of 3: ⭐ **HELD at ALL THREE**.
* **P2-PRED clause 3** — `rate(≤2.0 s)` falls: ⛔ **FAILED** (0.5482 → **0.6003**).
* **DEGENERACY control** — the spread **ROSE** 0.060303 → 0.099617: the quantile made the
  term **more** informative, so the divergence shrank for the mechanism's reason and not
  because the term went inert. ⭐ That is the control passing in the *informative*
  direction, which is stronger than merely not collapsing.

⇒ ⛔ **P2-PRED is a conjunction and it FAILS as written.** Reported as failed.
⇒ ⭐ **Its two DIVERGENCE clauses HELD where `progress`'s did not.** Converting `headway`
from rare-and-large to frequent-and-moderate does exactly what the mechanism predicted for
the asymmetry — and makes the RATE worse.

### 2.2 ⭐ THE TIE CONTROL — and it does not rescue the rate

`G-REWARD`'s statistic is `hold_v0 ≥ human`, so an **EXACT TIE counts as a hold-v0 win**. A
repair that pins both sides at a shared cap manufactures ties and inflates the rate for a
reason that is a **scoring convention**, not a change in the reward's preference. MEASURED
(`raw/tie_control.json`):

| mode | rung | rate (ties = wins) | tie fraction | rate, ties excluded |
|---|---|---|---|---|
| off | all | 0.4418 | 0.0430 | 0.4167 |
| `achievable` | all | **0.5740** | **0.1573** | **0.4944** |
| `lead` | all | 0.4465 | 0.0517 | 0.4163 |
| `hq25` | all | 0.4649 | 0.0430 | 0.4409 |
| ⭐ `hq25` | **≤2.0 s** | **0.6003** | **0.0000** | **0.6003** |

⇒ ⭐ `achievable`'s all-window rate rise is **substantially a tie artefact** (0.5740 → 0.4944
once ties leave the population). ⛔ **`hq25`'s ≤2.0 s rise is NOT** — the tie fraction is
exactly zero there, so clause 3 fails honestly and is reported as failed.

### 2.3 ⛔ WHERE THIS LEAVES THE QUESTION — the named blocker

`PREREG_HEADWAY.md` §3 committed the alternative in advance: *"if these fail, then neither
ranking term is the generator and the asymmetry is a property of the POPULATION … the honest
next step is a magnitude-aware statistic pre-registered on its own — and I say so rather
than trying a third term."*

Four independent results now say the same thing: 2,400 weightings cannot reach 0.30; the
scene-property ladder makes the rate **rise** with conflict; repairing `progress` to an
exact **0.000000** payment leaves the divergence **larger**; and repairing `headway`'s order
statistic **shrinks the divergence while raising the rate**. ⇒ **The rate is not measuring
what it is being asked to measure**, and the remaining lever is a **statistic**, which is
exactly the object the standing ruling reserves to the PI / Master Mind:

> ⛔ **BLOCKED — PI / Master Mind.** *Should `G-REWARD`'s statistic be a magnitude-aware
> rank test (e.g. a paired signed-rank or a win-magnitude ratio), pre-registered under its
> own hypothesis ID with its own bar, on data not used to discover it?* 0 GPU. I have not
> chosen one, and I have not re-scored the gate.

---

## 3. ⛔⛔ P2 — an RL arm can no longer execute the deliberate regression

### 3.1 The danger, restated from the source

MEASURED: `refcv3_adapter.sample_offsets` scales the emitted **offset waypoints** and has
no control space, so **every** arm in the table explored in **metre space** — which
`E-DDA-3c` §4 pre-registers as the DELIBERATE REGRESSION `reg_metre`, the arm that must
FAIL flyability. Launching `rl` ran the regression and would have tabled it as the
hypothesis.

### 3.2 What is now in place

1. **`PostTrainConfig.sample_space`** — `"control"` | `"metre"`, **validated** and written
   into `config.json` by `to_dict()`, so the record says which space ran.
2. **Every one of the 16 arms DECLARES `sample_space` and `hypothesis_arm`.** The already-
   banked 2026-09-05 arms declare `"metre"` — which is what they RAN, so their reproduction
   is preserved and the record becomes honest. ⛔ **Under `E-DDA-3c`'s arm table the banked
   `rl` result IS `reg_metre` and may never be quoted as `E-DDA-3c`'s `rl`.**
3. **`config.assert_arm_sample_space`** — the refusal lives in the library, beside the field,
   so every launcher inherits it. `mode_arm` calls it **first**, before the checkpoint, the
   corpus or one GPU second.
4. **The control-space path is wired**: the deterministic fan's 2 s prefix → the programme's
   single inverse map `unicycle_controls_from_path` → the two DD-v2 scalars → clamp to the
   envelope → re-roll through `rollout_unicycle`. ⛔ The prefix and not the full fan, because
   the inverse assumes a uniform `dt` and only slots 0–3 of `ARM_HORIZONS` are uniform.
5. ⭐⭐ **A MEASURED SILENT NO-OP IN `control_space.py`, FOUND AND FIXED.** Its shipped
   `logp` was the density of the two scalars — a function of the DRAW alone. MEASURED by
   running it: `requires_grad` **False**, `grad_fn` **None**, and
   `(-(logp·A).sum()).backward()` raises *"element 0 of tensors does not require grad"*.
   ⇒ wiring it in gives a crash, or — summed with anything that does carry grad — a policy
   gradient that is **identically zero**: a no-op arm wearing the hypothesis' name.
   `policy_logp` now computes the score-function density of the **drawn controls** with the
   sample detached and the parameters live (the corrected form `refcv3_adapter` already
   documents), `logp_mode="policy"` is the default, and `assert_carries_policy_gradient`
   **refuses a detached `logp` on every step**.

### 3.3 ⭐ PROVED BY MUTATION, not by inspection — 14/14

A guard that is only read is not a guard; this programme has an AST census that read **0
suspects on both the fixed and the broken trainer**. Every proof below **reintroduces the
defect**:

| mutation | result |
|---|---|
| `rl`, as shipped | `("control", True)` — allowed |
| ⛔ `ARMS["rl"]["sample_space"] = "metre"` (the exact pre-fix state) | **REFUSED**, message names `reg_metre` |
| ⛔ delete the declaration | **REFUSED** — "does not DECLARE" |
| ⛔ unknown value | **REFUSED** |
| ⭐ `reg_metre` (metre, non-hypothesis) | **ALLOWED** — the guard is DISCRIMINATING, not a blanket ban |
| ⛔ `mode_arm` with a mutated table + `LAUNCH_APPROVED=1` | **REFUSED before loading anything** |
| ⛔ `logp_mode="scalar"` | `backward()` raises; the guard **REFUSES** it |
| ⭐ `logp_mode="policy"` | guard passes, `backward()` puts a **non-zero** gradient on the policy |
| ⭐ control-space draw at **25×** the published σ | `envelope_violation` **exactly 0.0** |
| ⭐ metre-space draw on the same fan | **violates** — so "flyable by construction" is discriminating, not a tautology |

### 3.4 ⭐⭐ AND THE WIRED PATH ACTUALLY RUNS — PREFLIGHT ON THE REAL MODEL

⛔ A guard that refuses the wrong path is worth nothing if the right path does not run, so
the mutation proofs are not the last word. `stack/scripts/rl_control_space_preflight.py`
runs the wired sampler on the **real refcv3 checkpoint at step 40,284**, on a **real batch of
2 windows**, and takes a real `backward()`. ⛔ **It is NOT an arm launch**: no optimizer is
constructed, no `step()` is taken, no checkpoint is written, no result is produced.

MEASURED 2026-09-06, dev-box RTX 4060 (`raw/cs_preflight.json`):

| | `rl` (**control** space) | `reg_metre` (**metre** space) |
|---|---|---|
| sampled fan | `[2, 128, 4, 5, 2]` | `[2, 128, 4, 5, 2]` |
| `logp` | `[2, 128, 4]`, `grad_fn` present | `[2, 128, 4]`, `grad_fn` present |
| trainable tensors receiving a **non-zero** gradient | **71 / 71** | 71 / 71 |
| Σ\|grad\| | 4.070923e+05 | 9.546053e+07 |
| ⭐ `envelope_violation` of the **explored** fan | ⭐ **exactly 0.000000** | ⛔ **36.096268** |

⇒ ⭐ **the gradient reaches the decoder** through the control-space `logp` — the silent
no-op is gone — **and every explored candidate is flyable by construction**, while the
pre-registered regression leaves the envelope by 36× on the same model and the same batch.
**"Flyable by construction" is discriminating on the real model, not only on a fixture.**
⭐ All **16 arms** also build and validate their `PostTrainConfig` in the same preflight, so
a config error surfaces at 0 GPU rather than after minutes of paid start-up.

### 3.5 ⇒ CAN AN RL ARM NOW RUN WITHOUT EXECUTING THE DELIBERATE REGRESSION?

> ⭐ **YES** — for the sampler. `rl` and `rl_s1` are declared `sample_space="control"`,
> route through `control_space`, and a mutated or undeclared arm is REFUSED before any
> compute. ⛔ **The launch itself remains gated on `G-REWARD` (§2.3, PI / Master Mind) and
> on a free GPU**, and those are the only two things left in the way.

---

## 4. ⭐ B6 — CLOSED for the instrument, PARTIAL for refcv4b FINAL

`--mode fanbank` did not exist (0 of 26 flags matched `fan`/`bank`/`dump`); it does now, and
it writes exactly the schema `rl_fan_floor.py` consumes (`fan2 [W,K,5,2]`, `eid`, `has_lead`,
`dt_s`, `c_progress c_headway c_collision c_comfort c_feasibility`), verifies the file **by
content** (reads it back; refuses an all-zero bank — the `E-DETECT-1` poisoned-memmap class),
and asserts no `FORBIDDEN_FUTURE_CTX` key reached the reward context.

**MEASURED, dev-box RTX 4060, 30.5 s:** `fan_bank_refcv4b_9500_240w.npz` — **240 windows ×
117 candidates × 5 slots**, 103 episodes, 69 lead windows, `dt` 0.5 s. The 117-wide fan is
refcv4b's v0-conditioned vocabulary, which is how we know the right model loaded.

⭐ **THE PRIMARY READOUT RUNS ON IT, WITH ITS CONTROLS** (`raw/fanfloor_refcv4b_9500_*.json`,
T0 / fan-level):

| metric | refcv4b @ 9,500 | `rand_best@k` (the control) |
|---|---|---|
| `fan_floor@1` | 0.589504 | 0.436155 |
| `fan_floor@8` | 0.564321 | 0.558274 |
| ⭐ `fan_floor@10` | **0.560872** | **0.562476** |
| `fan_floor@32` | 0.524690 | 0.578102 |
| `fan_floor@117` | 0.200698 | 0.589504 |
| `fan_diversity` | 2.565058 m | — |

⭐ **The best-of-N control is live and CROSSES between k = 8 and k = 10** — an independent
replication of the crossing the instrument was built around, on a different model.

⭐⭐ **P3 RE-PROVED BY MUTATION on this new bank**: the deliberate 90 % collapse fires on
**both** halves under the geometry quality — diversity **−2.308552 (rel −0.9000)** and floor
**+13.602342**. ⇒ a **destroyed fan again reports a large POSITIVE floor gain**, replicating
the predecessor's +9.614186 on a different model and a different bank. `floor_verdict` still
**REFUSES** a verdict when diversity is stripped from the readout (mutation-tested this
turn), and it classifies the collapsed pair rather than passing it.

⛔ **WHAT IS NOT DONE:** this is **step 9,500 of 40,284 (23.6 %)**, NOT refcv4b FINAL.
Two probes found **no refcv4b FINAL checkpoint on the dev box**; per
`MODEL_REGISTRY.md` §4.6 it is `ckpt_40284_FINAL.pt` (md5 `99b573e8277d94a5e3bfbf630cb4d751`)
on the **A40 pod**, which is running refcv5's ~44 h training and which I was instructed not
to touch.

> **QUEUED, exact command, ~10 min on the 4060 once the FINAL checkpoint is on the box:**
> ```
> TANITAD_REPO=<clone> PYTHONPATH=<clone>/stack OMP_NUM_THREADS=6 python \
>   stack/scripts/rl_refcv3_min.py --mode fanbank --lead-mode track \
>   --ckpt <ckpt_40284_FINAL.pt> --config <refcv4b config.json> --expect-step 40284 \
>   --episodes <RL-fit v2ep dir> --labels <v7.2 TRAIN labels> \
>   --lead-block <fit lead block> --lru 6 --batch 4 --device cuda --seed 0 \
>   --bank-windows 240 --bank-fan raw/fan_bank_refcv4b_FINAL_240w.npz
> ```
> **It waits on exactly one thing: a copy of `ckpt_40284_FINAL.pt` off the training A40.**
> That is a read-only pull, but it competes for the training pod's disk and network, so it
> needs the Master Mind's word — it is not a decision this stream takes unilaterally.

---

## 5. ⚠️ A BASELINE CORRECTION, and it is not cosmetic

`PREREG.md` §4.1's CHANNEL control required the cap-OFF re-run to reproduce the banked
all-window figures to ≤ 1e-12.

* **RATE: reproduced EXACTLY** — `0.441780259484316`, abs err **0.000e+00**.
* ⛔ **MEAN: did NOT** — `−0.011732071340923` against the banked `−0.011239379601720929`,
  err **4.927e-04**.
* **ARITHMETIC control**: `hold_v0`'s uncapped `progress` reproduces from `v0` alone to
  **2.384e-08** (float32 storage). **frozen** stays far below the human (0.046313).

⭐ **The cause is identified to the last digit rather than waved at.** A per-window
comparison of all five components across all three sides — **18,267 evaluations** — finds
`progress`, `headway`, `feasibility` and `comfort` identical to **exactly 0.0**, and
`collision` differing on **13 of 18,267**, each by a full 1.0, all in the same direction
(the term now FIRES where it did not), inside 2 episodes. **3 of the 13 fall on `hold_v0`
alone**, and `3 / 6089 = 4.927e-04` — **the discrepancy exactly**.

⇒ **`rewards._collision` changed after the 2026-09-05 bank was written.** The banked
**mean** is a stale baseline; the banked **rate** is unaffected. Every comparator in this
package is therefore **this turn's own cap-OFF column**, same code, same day, same rows.
⚠️ Any figure quoting the banked `−0.011239` composed mean gap as a *current* value should
carry this correction.

---

## 6. Suite state, reported as a CONTROLLED comparison

* ⭐ **My tests: 31 new, all green** — `test_rl_progress_leadcap.py` **17/17** and
  `test_rl_sample_space_guard.py` **14/14**.
* ⭐ **The whole RL suite: 254/254 green** (every `stack/tests/test_rl_*.py`), up from the
  predecessor's 106 + 46 — nothing I changed broke a sibling's RL test.
* ⛔ **The full `pytest -q stack/` on this shared branch is NOT green and I do not imply
  otherwise — but I re-measured it rather than inheriting a figure.** MEASURED 2026-09-06,
  off-Drive clone, 784 s: **17 failed · 5,917 passed · 73 skipped · 2 xfailed**, with four
  sibling files excluded because they fail **collection** (`test_closedloop_floor`,
  `test_frame_align`, `test_render_quality_alignment_gate`, `test_xodr_junction_probe` —
  sibling modules absent from the clone's path). ⚠️ Not comparable to the predecessor's
  **171 failed / 6,446 passed / 28 errors**: different day, a branch that has moved, and a
  different exclusion set.
* ⭐ **NONE of the 17 is mine, and that is established POSITIVELY, not by an absence.**
  The failures fall in **11 unrelated files** (`test_cost_chord` ×4, `test_v6_chain` ×2,
  `test_runbook_commands` ×2, `test_ref_offset_repo_wide` ×2, `test_text_encoding_is_explicit`,
  `test_speed_band_derivation_blocker`, `test_secret_scan`, `test_refav1_kin_contract`,
  `test_refav1_arm`, `test_nav_known_channel`, `test_build_parity_guard`) — three siblings'
  live streams. **0 of 17** name a `test_rl_` file, and `control_space`, `progress_lead_cap`,
  `sample_space`, `_reduce_time_gap` and `achievable_along_ref` appear **0 times** anywhere
  in the output.
  ⚠️ **That last "0" is only admissible because it is paired with a positive assertion that
  my tests actually RAN in this invocation** — a zero from a file that was never collected is
  indistinguishable from a genuine absence. `--collect-only` reports
  `test_rl_progress_leadcap.py: 17` and `test_rl_sample_space_guard.py: 14`, and the
  collected total **6,009** equals **17 + 5,917 + 73 + 2 = 6,009** exactly. ⇒ all 31 of my
  tests were collected and passed inside the full run.
* **Clone identity checked by md5** against the repo for all six changed files, so the suite
  ran the code that was committed.

---

## 7. What changed, and where it lives

| artifact | where |
|---|---|
| pre-registration 1 (`H-RL-PROGRESS-LEADCAP-1`) | `…/2026-09-06-refcv4b-rl-repair/PREREG.md` (repo, commit `082606e02`) |
| pre-registration 2 (`H-RL-HEADWAY-QUANTILE-1`) | `…/2026-09-06-refcv4b-rl-repair/PREREG_HEADWAY.md` (repo, commit `02f5dda65`) |
| this report | `…/2026-09-06-refcv4b-rl-repair/RESULT.md` (repo) |
| ⭐ the `progress` lead cap + the `headway` quantile | `stack/tanitad/rl/rewards.py` (repo) |
| ⭐ `sample_space`, its validation and the REFUSAL | `stack/tanitad/rl/config.py` (repo) |
| ⭐ `policy_logp` · `assert_carries_policy_gradient` · `controls_from_path` | `stack/tanitad/rl/control_space.py` (repo) |
| ⭐ arm declarations · the control-space sampling path · `--mode fanbank` | `stack/scripts/rl_refcv3_min.py` (repo) |
| ⭐ the A/B probe (5 columns, one variable each) | `stack/scripts/rl_progress_leadcap_ab.py` (repo) |
| ⭐ tests, 17/17 | `stack/tests/test_rl_progress_leadcap.py` (repo) |
| ⭐ mutation proofs, 14/14 | `stack/tests/test_rl_sample_space_guard.py` (repo) |
| ⭐ the control-space preflight (⛔ not an arm launch) + its artifact | `stack/scripts/rl_control_space_preflight.py`, `…/raw/cs_preflight.json` (repo) |
| the A/B panel + its log + the tie control | `…/raw/leadcap_ab.json`, `leadcap_ab.log`, `tie_control.json` (repo) |
| ⭐ the refcv4b fan bank (B6) + its report | `…/raw/fan_bank_refcv4b_9500_240w.npz` (+`.report.json`) (repo) |
| the primary readout + the complete collapse control | `…/raw/fanfloor_refcv4b_9500_composed.json`, `…_geom.json` (repo) |
| ⚠️ **ONE PLACE ONLY** — the 20 MB per-window rows (5 modes × 3 sides × 5 components × 6,089) | `devbox:C:\Users\Admin\leadcap_ab\leadcap_rows.json` — re-derivable in ~7 min from the banked script and the fit120 corpus, so deliberately not banked |

## 8. Escalations — raised here, not written into a doc for someone to find

1. ⛔⛔ **`G-REWARD`'s STATISTIC.** Four independent results now say the rate cannot see what
   the reward does (§2.3). The remaining lever is a **magnitude-aware statistic with its own
   hypothesis ID and its own bar, pre-registered on data not used to discover it**. ⛔ I have
   not chosen one. **Owner: PI / Master Mind. 0 GPU.**
2. ⛔ **A copy of `ckpt_40284_FINAL.pt` off the training A40**, so B6 closes on refcv4b
   FINAL rather than step 9,500. Read-only pull, but it touches a training pod.
   **Owner: Master Mind.**
3. ⚠️ **`E-DDA-3c` §4's arm table is now implementable but its `ctrl0` / `reg_echo` /
   `ctrl_null` controls exist twice** — legacy metre (banked, reproducible) and new
   `*_cs` control-space. The panel must use the `_cs` ones; a control that does not share
   the arm's ingredients is not a control. **Owner: this stream, at launch.**
4. ⚠️ **`fan_safety.py` still has no `--bank-fan` path.** `rl_refcv3_min.py --mode fanbank`
   now covers the RL rung, but the flag belongs in the eval tool too.
   **Owner: Benchmarks / Eval.**
