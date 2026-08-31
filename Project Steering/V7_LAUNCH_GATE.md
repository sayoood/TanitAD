# ⛔ v7 TRAINING IS GATED — the remaining problems, stated so the gate is checkable

**PI DIRECTIVE 2026-08-31, verbatim:** *"we should not start training of v7 until the
remaining problems are solved."* **BINDING.**

⚠️ **Why this document exists.** *"Until the remaining problems are solved"* is
unfalsifiable as written — it will drift, and someone will eventually declare it
satisfied by feel. Below is the list, each item with its **evidence**, its **state**,
and **what would close it**. If an item is not on this list it is not a gate; if it is,
it needs a measurement, not an opinion.

⛔ **And the closure rule, from the record:** on 2026-08-26 this programme declared
action-conditioning *"CLOSED, negatively"* and the PI **withdrew it**: *"That was not
my decision to make — closing a research direction is a programme call."* ⇒ **no item
below may be marked closed by me.** I can report that a criterion is met; the PI closes.

---

## P1 — ⭐⭐ NO v7 ARM HAS EVER BEATEN ITS OWN HOLD-ACTION CONTROL

**This is the actual problem.** P2–P4 are hypotheses about *why* it holds.

* **Evidence:** the programme's first and only T1 read (2026-08-30, 3 arms, 6,924
  windows, episode-cluster bootstrap). **In every distance metric, for every arm, the
  closed-loop arm is worse than its own hold-action control.** All three
  `holdv0=LOSES_TO_HOLDV0`; all three `_longitudinal_claim_admissible=false`; heading
  MAE ~95° (chance).
* ⭐ **The one genuinely positive result there:** `copy_detector = CLEAN`,
  `echo_index 0.0000` on all three. v1.x's failure was *fake skill by echo*; the v7 line
  has traded that for **honest absence of skill**. That is a real structural change.
* ⚠️ **Scope:** those were ~19 M v7-TINY arms on stage S-W with **every planner
  objective at zero** — a world-model trunk plus a readout, never trained to drive.
  *"v7 cannot drive"* is **not** supported by that read.
* **CLOSES WHEN:** a v7 arm beats hold-action at T1 across the four metric families,
  paired bootstrap, with the echo control still clean.

## P2 — THE MODEL DOES NOT USE ITS ACTIONS, AND WE DO NOT KNOW WHY

* **Evidence (converging, three instruments):** action moves the prediction **0.4–0.6 %**
  as much as the scene (MM-E10, C0/C1 both valid) · `ego_state` adds **−0.0006 (t −0.48)**
  over drift in predicting Δz, held-out, constant control exactly 0.0000
  (`latentmotion.json`) · an earlier probe found **−0.0109 (t −0.57)** (`deltaz.py`) ·
  the model's own FiLM gain **converged** rather than straining (MM-E18).
* **Three candidate causes. One under test, two never tested:**

  | cause | state |
  |---|---|
  | (a) the horizon is too short for actions to matter | ⏳ **UNDER TEST — MM-E19**, reads ~19 h |
  | (b) ⭐ **our "action" is realised motion, not a command** — E-DEC-57: the channel is a kinematic restatement of the pose change (r 0.9988) ⇒ **a genuine command channel has never been tested** | ⛔ **NEVER TESTED** |
  | (c) the action representation / latent geometry | ⛔ untested |

* ⛔ **Eliminated, so they are not the answer:** the missing objective (MM-E11: O1 on
  → ratio **fell** 0.40× against a criterion demanding a 10× rise) · zero-init FiLM
  (it trained: ‖W‖ 2.97/3.06/5.44) · a starved conditioning gain (MM-E18: converged).
* ⚠️ **Status is OPEN by PI ruling** — the 2026-08-26 "closed negatively" was withdrawn,
  and the evidence supports only *"not found to work under the conditions tested."*
* **CLOSES WHEN:** (a), (b) and (c) are each measured, or one of them is confirmed and
  the fix raises the action ratio materially.

## P3 — DRIFT: A REAL, SEED-STABLE 3.3× EFFECT WHOSE CAUSE IS UNKNOWN

* ⛔ **DO NOT SAY "initialisation is the lever" — that was RETRACTED** (E-DEC-60/C164).
  `postrain30k` and `postrain30k_seed1` are `--init-from` the **same** distilled
  checkpoint and read **0.669 / 0.679** — the *scratch* band — while distilled
  `splitp30k` reads **0.199**. Two arms share the supposed lever and sit **0.47 apart**;
  the distilled/scratch separation was a **confound**.
* ⭐ **What survives is stronger than what was retracted:** `splitp30k`'s drift **0.199
  against 0.657–0.679** for three other 30k arms is large, real and **seed-stable**
  (~1.5 % run-to-run). **The effect is genuine; the attribution was not.**
* **Also known:** drift is **self-dominated** (MM-E6 — the scene subspace carries ~20×
  less drift than a random subspace of equal rank) and **trivially reducible** (MM-E4 —
  the shuffle control fired), so it is a **symptom**, and **nine objective terms** failed
  to move it.
* **CLOSES WHEN:** the ablation that distinguishes `splitp30k` from `postrain30k`
  **with init held fixed** identifies the cause. That experiment is defined and unrun.

## P4 — THE HORIZON LADDER IS SHORT OF THE LABELS

* **Evidence:** the labels describe events a median **12.5 s** ahead
  (`by_time_s` = a real manoeuvre `t_start_s` on 76.7 % of records); the bands are
  operative 0–2 s, tactical 2–6 s, strategic 8–30 s (§4b, binding). The campaign trained
  `o5_k 8` = **0.8 s**.
* ⏳ **PARTIALLY ADDRESSED:** `k60p30k` runs at 6.0 s — covering operative **and**
  tactical. ⛔ **Strategic (8–30 s) is still unreached**, and per MM-E15 the naive fix
  (`o5_k 80`) is 10× rollout compute on a predictor that collapses past one tick.
* **CLOSES WHEN:** the recipe reaches the tactical band (MM-E19) **and** the strategic
  level has a mechanism — the temporal-abstraction ladder (MM-E16, ratios **1 : ~3 : ~15**
  read off the corpus), not a longer flat rollout.

---

## ✅ Closed 2026-08-31 — no longer gates

| was | closed by |
|---|---|
| the nav data join (v7f could not launch with nav) | `--nav-labels` + `NavEmitter` wiring, count-verified join, 197 tests |
| dead horizons corrupting measurement (h2/h4 zero-gradient noise read as results) | trainer refusal + `trained_horizons` + backprop-pinned tests |
| v7.2 artifacts identified by path, not content (a guard pinned the WRONG copy) | `resolve_v72` keys on md5; labels **and** indexes content-addressed |
| the 6 s cost unknown | MEASURED: 2.90 s/step, 24.2 h, 6.27 GB — no OOM risk at this scale |

## ⏳ The nearest decision point

`k60p30k` reads in ~19 h. **HORIZON-WORKS** closes P2(a) and most of P4.
**HORIZON-INERT** promotes **P2(b) — the untested command channel — to the front**, and
that is the deeper question: *we have never given this model a command, only a
description of what it already did.*
