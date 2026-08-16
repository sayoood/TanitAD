# V0 ANTI-ECHO — the three controls that make the PI's `v0` admission safe

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Tier:** instrument (no GPU, CPU-only)
**Status:** IMPLEMENTED, WIRED, AND PROVEN TO FIRE. Staged, not pushed.

---

## 0. The ruling, and the hole it opens

⭐ **Sayed, verbatim, 2026-08-16** (`Project Steering/V6F_PLANNER_DESIGN.md` §1.4):

> *"We can use v0 as input since it is measured and is not the future, but we should assure that
> the model/planner later is not cheating by just outputting v0 as longitudinal plan."*

`v0` (ego speed at t0) is **ADMISSIBLE as a planner input**. It is measured present state — an
observation available to any real vehicle at inference — not a future quantity and not derived from
a label. The vision-only rule exists to stop the model reading *the answer*; `v0` is not the answer.

⛔ **But the second half of that sentence is the load-bearing half, and it is an obligation, not a
reassurance.** Admitting `v0` opens a new way to fake competence: a planner can emit **"keep doing
v0"** as its longitudinal plan and score well, because holding the current speed is a strong
baseline on most windows. That is **skill attributed to a copy**.

This programme has been fooled by that exact shape **three times**, each looking like capability
until a control was run:

| failure | what it scored | what it actually was |
|---|---|---|
| **nav-echo** (flagship v1 route head) | `route_acc_nav` = **1.0000** | an exact bijection of the nav it was fed |
| **T1 action echo** | S-curve reproduction **97.9 %** open-loop | **0.0 %** hold-action, ~5 % closed-loop |
| **P1 speed echo** | R² **0.995** | **−0.72** under the v0 shuffle |

⚠️ **AND THE ADMISSION CLOSED NO DEFICIT.** MEASURED (INHERITED from the E-WC2 surface, carried
forward verbatim): even vision **+ v0** sat at σ/ADE **3.527** on the REF-C surface — still worse
than a **0-parameter constant-yaw-rate rule at 1.1888**. Admitting `v0` removed an *argument*, not
a *gap*. Every artifact this work touches now carries that sentence.

---

## 1. What was built

### 1.1 `taniteval/taniteval/v0_antiecho.py` — the three controls

| # | control | what it answers | can it run from a banked dump? |
|---|---|---|---|
| 1 | `holdv0_baseline` | ⭐ **the ADMISSIBILITY test.** Does the arm BEAT a plan that sustains `v0` with zero commanded accel, *separated*, under a **paired episode-cluster bootstrap**? | **yes** |
| 2 | `copy_detector` | **the always-on SCALAR.** Is the emitted longitudinal plan literally the constant-`v0` trajectory? | **yes**, milliseconds |
| 3 | `shuffle_control` | **the FALSIFIER.** Permute `v0` across windows — does the plan follow a value we know to be wrong? | **no** — needs a second forward pass; reports UNAVAILABLE *with reason + n* |

⛔ **None substitutes for another**, and the synthetic panel in §3 shows why: the hard-braking arm
is **CLEAN** on the detector and still **LOSES_TO_HOLDV0**; the "ignores v0 entirely" arm is
**NOT_AN_ECHO** on the shuffle with degradation exactly **0.0**.

### 1.2 Wired so they report **automatically**, not on request

`four_families.longitudinal()` now attaches `anti_echo` to **every** LONGITUDINAL block — no flag,
no opt-in — and `all_families()` promotes one bit to the block's own top level:

```
fam["_longitudinal_claim_admissible"]   # ⛔ the bit a report may not lose
fam["_anti_echo_summary"]               # "holdv0=… · copy_detector=… · shuffle=…"
fam["_anti_echo_rule"]                  # the ruling, travelling with the number
fam["_complete"]                        # now ALSO requires anti_echo["status"] == "OK"
```

`taniteval/tools/eval_four_families.py` **prints** both lines to the console, so an operator sees
the verdict without opening the JSON.

Per the binding four-families rule the controls are **per-family, never pooled**; each carries its
own estimator, CI and `n`; and every branch that genuinely cannot be computed states its **reason
and its n** (clause 5) rather than being dropped.

### 1.3 `v0` resolution — and the refusal that keeps the whole thing falsifiable

`v0` is read, in order, from `win["v0"]` → `win["speed"]` (what `rollout.collect` already
publishes) → `win["lead"]["speeds"]` (the lead block's time-gap denominator **is** the ego speed at
t0, so a caller who supplied a lead block already supplied `v0`).

⛔ **It is NEVER derived from `gt` or `pred`.** The first GT step is a *future* displacement;
imputing `v0` from it would make every control compare the arm against a quantity the arm's own
error moves — **unfalsifiable by construction**, the exact defect class the ruling exists to
prevent. No `v0` ⇒ `UNAVAILABLE` with a reason, `_complete = False`, and an explicit
`⛔_consequence` stating the condition is **UNDISCHARGED**.

---

## 2. Control 3 was **reused, not reinvented** — and the citation was checked

The model-side permutation that produced the P1 row (**R² 0.995 → −0.72**) already exists:

| what | where | verified |
|---|---|---|
| the permutation itself | `stack/scripts/probe_latent_state.py:578` — `collect_grid(..., speed_echo_control=True)` permutes `b["pose_last"][:, 3]` across the batch: *same frames, same recorded actions, WRONG speed scalar* | ✅ |
| the CLI flag | `stack/scripts/probe_latent_state.py:775` — `--speed-echo-control` | ✅ |
| already in production | `stack/ops/pbattery_watcher.py:80` passes it | ✅ |

`SHUFFLE_PRODUCER` in the new module names all three, and
`test_the_existing_machinery_is_actually_there_at_the_cited_lines` **reads the real file** and
asserts the tokens are present — because "absence found at one location is not absence" has a twin:
**a citation nobody checked**.

`shuffle_v0()` here is only the **seeded scorer-side canonical form** of the same construction, so a
caller re-running a planner uses one shuffle rather than a second one that drifts. ⚠️ A stated
difference rather than a blurred one: `probe_latent_state` permutes **within each batch**, which is
strictly weaker (a window can be paired with a `v0` from a neighbouring window of the same episode);
the whole-array form is what this module scores against, and `n_fixed_points` is reported rather
than silently resampled.

---

## 3. ⭐ THE CAN-IT-FIRE PROOF — a guard nobody has watched fail is a hypothesis

Four synthetic arms, **240 windows / 12 episodes**, the human genuinely accelerating or braking on
70 % of them, all scored through `four_families.all_families()` — the same entry point the real
evals use, with `n_boot = 2000`.

MEASURED 2026-08-16, dev box, CPU. Reproduce with `taniteval/tests/test_v0_antiecho.py`.

| | **PURE ECHO**<br>(output ≡ hold-v0) | **HONEST**<br>(real accel structure) | **HARD BRAKE**<br>(−2.0 m/s², wrong) |
|---|---|---|---|
| `speed_mae_mps` (the headline the rule already required) | 0.9153 | 0.2543 | 2.0477 |
| `target_speed_acc within_2.0_mps` | **0.8365** ⚠️ | 1.0000 | 0.5771 |
| **`echo_index`** ⭐ | **1.0000** | 0.1167 | 0.0000 |
| `echo_index_gt` (the corpus's own) | 0.3125 | 0.3125 | 0.3125 |
| **`echo_index_excess`** | **+0.6875** | −0.1958 | −0.3125 |
| `cmd_accel_mae_mps2` (GT 0.9312) | **0.0000** | 1.0042 | 1.9863 |
| `dev_ratio` | **0.0000** | 1.0782 | 2.1795 |
| `dev_r` | **None** (degenerate) | 0.9723 | 0.0186 |
| `echo_frac_where_human_acted` (n = 142) | **1.0000** | 0.0141 | 0.0000 |
| **copy_detector verdict** | **ECHO** | CLEAN | CLEAN |
| hold-v0 Δ `speed_mae` [lo, hi] | **+0.0 [+0.0, +0.0]** | **−0.6610 [−0.7775, −0.5405]** | +1.1325 [+1.0018, +1.2762] |
| separated | **False** | **True** | True |
| **holdv0 verdict** | **NOT_SEPARATED** | **BEATS_HOLDV0** | **LOSES_TO_HOLDV0** |
| ⛔ **`_longitudinal_claim_admissible`** | **False** | **True** | **False** |

*(paired episode-cluster bootstrap, `taniteval/ci.py`, 12 episodes, B = 2000)*

**Shuffle control** (pure echo re-run under permuted `v0` vs an arm whose output is unchanged by it):

| | PURE ECHO | IGNORES-v0 |
|---|---|---|
| `tracks_shuffled_v0` | **0.0000** | 3.7250 |
| RMS to the lie / to the truth | 0.0000 / 5.5174 | 5.7456 / 1.5424 |
| degradation Δ [lo, hi] | +3.6703 [+3.2905, +4.0964] **separated** | **+0.0 [+0.0, +0.0]** n.s. |
| verdict | **ECHO** | NOT_AN_ECHO |

### What the panel establishes

1. ⭐ **The detector FIRES on the cheat.** `echo_index` is exactly **1.0000**, `cmd_accel` exactly
   **0.0**, and on **every one of the 142 windows where the human demonstrably acted**, the echo
   held anyway (`echo_frac_where_human_acted = 1.0`).
2. ⭐ **It does NOT fire on genuine structure.** The honest arm is CLEAN, admissible, and beats
   hold-v0 by **0.661 m/s [0.5405, 0.7775]**, separated.
3. ⛔ **THE ECHO WOULD HAVE LOOKED FINE.** Beside the controls, the pure copy carries a
   `speed_mae` of **0.9153 m/s** and is within 2 m/s of the right speed on **83.65 %** of horizon
   steps. A report quoting the four-families LONGITUDINAL row **without** the controls would have
   published that as target-speed competence. **This is the whole case for the rule.**
4. ⛔ **A pure echo can never be separated from hold-v0** — its Δ is identically zero. So
   "NOT_SEPARATED" is not a near-miss; it is the signature.
5. ⭐ **No control is redundant.** The hard-braking arm is **CLEAN** on the detector (correctly — it
   is not a copy) yet **LOSES_TO_HOLDV0** and is inadmissible. The ignores-`v0` arm is
   **NOT_AN_ECHO** on the shuffle with **zero** degradation.

---

## 4. Two real instrument defects the proof caught

Both were found by *watching the guard fail*, not by reading the code.

### 4.1 ⛔ An exact-zero degeneracy test gave a **PURE ECHO a real number**

`dev_r` — the correlation between the arm's departure from `v0` and the human's — is the one scalar
a pure echo cannot fake: its departure is nil, so the correlation must be **undefined**.

MEASURED: the round trip `hold_v0_path(v0) → _seq_geometry → speed` is **NOT bit-exact in float32**.
The geometry recovers speed from `norm(diff(positions))/dt` rather than reading it back, leaving a
residue of **1.9073e-06 m/s max, 5.6974e-07 m/s RMS** (and 7.6294e-06 m/s² on the accel). With a
`== 0` degeneracy test the detector **correlated that float noise** and reported

```
dev_r = -0.0133      for a PURE ECHO
```

— a *number*, where the honest answer is "undefined". A reader would have taken it as a
genuine-but-weak planner instead of the copy it is. **Fixed** with a *physical* gate
(`DEV_R_MIN_RMS_MPS = 1e-3`, ~1750× the measured residue and ~100× below `ECHO_DEV_MPS`), pinned by
`test_the_dev_r_degeneracy_gate_is_physical_not_an_exact_zero_test`. Same family as every
"a probe that reports the wrong scope is worse than no probe" trap in `CLAUDE.md`.

### 4.2 ⛔ A verdict label that claimed more than it measured

The shuffle's top band was first called **`USES_SPEED`**. MEASURED: an arm that **ignores `v0`
entirely** produces an identical plan under the shuffle, so it lands in that band with degradation
**exactly 0.0 [0.0, 0.0]** and ratio **3.725**. The label asserted the arm *uses* speed; the
control can only ever **refute a copy**. Renamed **`NOT_AN_ECHO`**, and the band table carries a
`_naming` note saying why. The verdict text now reads *"establishes nothing else"*.

⚠️ **Generalise it:** a control's verdict label must name the inference it supports, not the
inference the reader hopes for. `CLEAN` on the copy-detector is documented the same way — it can
refute a copy, never establish skill.

### 4.3 One more design choice worth stating: `r` alone is not the detector

MEASURED: a **pure echo** scores `r_vs_holdv0 = 1.0000`; a planner braking at a violent **−2.0
m/s²** still scores **0.9872**. The entire usable range of `r` between "literal copy" and
"emergency stop" is under **2 %** — any threshold on it is knife-edge. `echo_index` separates the
same two arms by **> 0.5**. `r` is published as a *component*, explicitly labelled necessary and not
sufficient, and pinned by `test_r_alone_has_almost_no_dynamic_range_which_is_why_it_is_never_alone`.

⭐ And `echo_index` is **always published beside the GT's own value** (0.3125 here), because most of
this corpus genuinely is cruising — quoting the arm's number alone would charge the arm for the
corpus. The headline is the **excess**, which is threshold-robust: the same PROPOSED thresholds are
applied to both paths.

---

## 5. The `e_wc2` contradiction, fixed — and its downstream

`stack/scripts/e_wc2_sigma_star.py:188` classed `"v0": "ECHO"` (inadmissible) while
`V6F_PLANNER_DESIGN.md` §1.4 declared it admissible. Both were at HEAD. The PI settled it.

* **`"v0": "ECHO"` → `"v0": "MEASURED_PRESENT"`** — a new admissibility class meaning *measured
  present state: admissible at inference, carrying the anti-echo obligation*. The comment now
  records the **real constraint** (the three controls, with their file paths) instead of a
  superseded prohibition, plus the σ/ADE **3.527 vs 1.1888** context.
* **`"measurement"` stays `ECHO`** — it carries the **nav** signal, which *is* derived from the
  ego's own future. The ruling moved `v0` only. Pinned.
* `MEASURED_PRESENT` **does not gate a verdict** (unlike `any_echo`); it stamps
  `_anti_echo_obligation` at the record's **top level** and inside `features`, and the
  `four_families.LONGITUDINAL` note points at it. A note nested three keys deep is a note a report
  loses.
* `--declare` now accepts the class; its help text says so.

**Downstream propagation** — `stack/scripts/e_ag1_anchor_floor.py` had built an entire arm (**E-AG3**)
whose purpose was to *measure the magnitude of this contradiction*, stamped
`PENDING_PI_ADJUDICATION` and gated behind `--allow-echo-features`. Two of its tests asserted the
superseded state and **failed** the moment `e_wc2` was corrected. Updated: E-AG3 is now
`⭐ ADJUDICATED 2026-08-16`, runs **without** the flag, keys on `any_measured_present`, and its
status string carries the ruling, the three controls and the 3.527/1.1888 context.

---

## 6. Tests

| suite | baseline | now | delta |
|---|---|---|---|
| `taniteval/` | 1058 / 0 | **1090 / 0** | **+32** (all `test_v0_antiecho.py`) |
| `stack/` | 3282 / 0 / 17 skipped / 2 xfailed | see §8 | **+5 new**, **2 rewritten** |

New: `taniteval/tests/test_v0_antiecho.py` (32) · 5 added to
`stack/tests/test_e_wc2_sigma_star.py` · 2 rewritten in `stack/tests/test_e_ag1_anchor_floor.py`.

---

## 7. ⚠️ Limits, stated

1. **`hold_v0_path` is a STRAIGHT line.** It is the LONGITUDINAL floor and nothing else; it must
   never judge a lateral claim. For turns the floor family is `driving.FLOORS` (`cv`/`holdv0`/`ctrv`)
   — the CTRV addition moved **16 of 25** banked arms' headline verdicts. Pinned **bit-identical**
   to `taniteval.driving.hold_v0` at the tier-0 surface so the programme keeps exactly one hold-v0.
2. **The shuffle needs the model.** From a banked dump it is a WORK ITEM with a reason and an `n`,
   naming the producer. It has **not** yet been run against a real checkpoint — ⛔ CPU-only this
   turn; Thor is training on the only GPU.
3. **All six thresholds are PROPOSED**, and every one is applied to the arm *and* the GT alike, so
   the headline (the excess) is threshold-robust. None is a gate.
4. **On a pure-cruising window set the controls cannot discriminate** — hold-v0 *is* the right
   answer there, so an echo is indistinguishable from skill **by construction**. That is a fact
   about driving, not a weakness; it is why `echo_frac_where_human_acted` reports the
   discriminating subset with its own `n` (142 of 240 here).
5. **The panel is SYNTHETIC.** It proves the instrument fires and does not fire; it says nothing
   about any TanitAD checkpoint. Evidence class: **MEASURED (ours)** for the instrument's behaviour,
   **not** for any model claim.

---

## 8. ⛔ ESCALATION — one stale claim I did NOT edit

`stack/tests/test_v6_factored_goal.py:849-851` still reads:

> *"``v0``'s admissibility is an OPEN PI DECISION (V6F_PLANNER_DESIGN §1.4 vs
> e_wc2_sigma_star.py:188)…"*

**It is no longer open.** The test's *assertions* remain correct and pass (the anchor head must not
read `v0`, which is still true and still worth pinning) — only the docstring's premise is stale. I
left it because that file belongs to the live `v6.py` stream and a concurrent edit risks a conflict.
**A stale "open decision" inside a test is exactly how a superseded claim propagates**, so it is
raised here rather than left to an audit. A background task has been filed.

---

## 9. Deliverable manifest

All paths relative to the repo root, all **staged in the working tree, NOT committed, NOT pushed**.

| file | state | what |
|---|---|---|
| `taniteval/taniteval/v0_antiecho.py` | **NEW** | the three controls + `hold_v0_path`, `shuffle_v0`, `resolve_v0`, `anti_echo` |
| `taniteval/tests/test_v0_antiecho.py` | **NEW** | 32 tests — the can-it-fire proof, both defect regressions, the citation check |
| `taniteval/taniteval/four_families.py` | modified | `longitudinal(…, win=…)` attaches `anti_echo`; `all_families` promotes `_longitudinal_claim_admissible`, folds it into `_complete` |
| `taniteval/tools/eval_four_families.py` | modified | `_complete` recompute + prints the verdict to the console |
| `stack/scripts/e_wc2_sigma_star.py` | modified | `v0` → `MEASURED_PRESENT`; `ADMISSIBILITY_CLASSES`; `ANTIECHO_OBLIGATION` stamped at top level |
| `stack/tests/test_e_wc2_sigma_star.py` | modified | +5 tests pinning the corrected classification and the obligation |
| `stack/scripts/e_ag1_anchor_floor.py` | modified | E-AG3 `PENDING_PI_ADJUDICATION` → `ADJUDICATED`; keys on `any_measured_present` |
| `stack/tests/test_e_ag1_anchor_floor.py` | modified | 2 tests rewritten for the resolved ruling |
| `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-16-v0-antiecho/V0_ANTIECHO.md` | **NEW** | this document |

**Nothing is stranded**: no pod, no worktree, no agent context. Every artifact is in the repo.

---

## 10. What this does and does not license

✅ A LONGITUDINAL number may now be emitted from a `v0`-fed arm — **with** its three controls
attached automatically and `_longitudinal_claim_admissible` visible beside it.

⛔ A LONGITUDINAL number emitted while `_longitudinal_claim_admissible` is **False** is a
**fidelity diagnostic, not a longitudinal capability result**, and must not be presented as one.
Not beating hold-v0, separated, is **not a small miss** — it means the longitudinal head has learned
nothing beyond its own input.

⚠️ And the sentence that must survive into every report quoting this work: **admitting `v0` removed
an argument, not a deficit.** Vision + `v0` = σ/ADE **3.527**; a 0-parameter constant-yaw-rate rule
= **1.1888**.
