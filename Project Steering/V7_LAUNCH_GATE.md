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
* ⭐⭐ **P2(b) SCOPED 2026-08-31 — AND IT CANNOT BE TESTED ON OUR TRAINING CORPUS AT ALL.**
  Checked the source rather than inferring from the correlation:

  | corpus | action signal | is it a command? |
  |---|---|---|
  | **PhysicalAI** (`physicalai-train-e438721ae894`, what every v7 arm trains on) | `actions = (steer_road_rad, accel_mps2)`, where **`steer = atan(wheelbase × curvature)`** — read off the `curvature` column | ⛔ **NO.** It is the realised path, restated. The v2ep cache stores only `actions [201,2]` + `poses [201,4]`; there is no command in the bytes |
  | **comma2k19** | `processed_log/CAN/steering_angle` — the **driver's steering-wheel angle from CAN** (`STEER_RATIO = 15.3` wheel→road) | ⭐ **YES** — measured at the wheel, it leads the yaw response and contains corrections the path never shows |

  ⇒ **E-DEC-57 is confirmed at the SOURCE, not merely by an r = 0.9988 correlation:** the
  incumbent channel is documented in `physicalai.py` as a *"road-wheel angle proxy"*
  computed from curvature. **We have never given this model a command because our corpus
  does not contain one.**

* ⛔ **TWO BLOCKERS ON THE ONLY CORPUS THAT COULD TEST IT:**
  1. **GEOMETRY.** comma2k19's entire field is **65.203°**; our frame is **120°**. The
     trainer warns on every run: *"comma2k19 cannot supply it at any resolution — it must
     be letterboxed (explicit unobserved mask), given its own frame, or dropped from the
     mix. That is a PI decision, not a default."* Any comma2k19 arm therefore breaks
     same-data comparability with every v7 arm.
  2. **DATA.** comma2k19 is **not on Thor** — only `extract_comma2k19.py` and its tests.

* ⭐ **THE CHEAPEST DISCRIMINATING TEST NEEDS NO ARM, AND IT TRIAGES THE WHOLE QUESTION:**
  once the data is present, correlate **CAN steering against the curvature-derived proxy
  on the same clips**. If r ≈ 0.999 — the same regime as our action's 0.9988 with realised
  pose change — then even a CAN command is nearly the realised path, and P2(b) is a weak
  lever that does not deserve an arm. If r is materially lower, there **is** independent
  command information and P2(b) becomes the strongest open candidate. ⚠️ Data-only: no
  training, no GPU, and it decides whether to spend either.

* **CLOSES WHEN:** (a), (b) and (c) are each measured, or one of them is confirmed and
  the fix raises the action ratio materially. ⚠️ **(b) additionally needs a PI call on the
  geometry** before any comma2k19 arm — that decision is named above and is not mine.

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
* ⭐⭐ **PROGRESS 2026-08-31 — THE ABLATION IS NOW DESIGNED, AND ONE CANDIDATE HAS A
  MEASURED MECHANISM.** Config-diffed the two arms, separating *absent keys* from
  *value changes* (the `.get()` trap that reported 16 phantom variables on MM-E11):

  ```
  postrain30k 194 args | splitp30k 181 args
  SHARED KEYS WITH DIFFERENT VALUES:  freeze_encoder False->True · o5_k 8->4 · out
  keys only in postrain30k: 13, and all three TERMS are INERT (w_o9_ema/w_o10_psg/w_o11_cf = 0.0)
  init_from: IDENTICAL (both distill_init.pt) · steps/seed/batch/lr/window/stage/w_o5/w_o6/o5_form all matched
  ```

  ⇒ **EXACTLY TWO candidate variables**, and the identical `init_from` confirms
  E-DEC-60's retraction directly: same init, drift 0.669 vs 0.199.

* ⭐⭐ **AND THE FREEZE ACTUALLY HELD — MEASURED on the checkpoints:**

  | arm | encoder tensors changed | ‖Δ‖/‖init‖ |
  |---|---|---|
  | `splitp30k` (drift **0.199**) | **0 of 41** | **0.000000** |
  | `postrain30k` (drift **0.669**) | **41 of 41** | **0.233** |

  `splitp30k`'s encoder is **bit-identical to `distill_init.pt`**; `postrain30k`'s moved
  23 %. ⇒ **HYPOTHESIS (not yet causal): the low drift is what a FROZEN LATENT SPACE
  looks like, not a recipe virtue.** Drift asks how much of Δz is predictable from
  `z_t`; with a frozen encoder the same frames map to the same latents throughout
  training, so the space cannot shift under the predictor.
  ⛔ **If that holds, the "lever" DISSOLVES rather than transfers** — and it would come
  attached to the frozen-encoder ceiling REF-A already hit.
  ⚠️ **A sharper worry it raises about the METRIC:** drift may conflate *"the predictor
  is self-referential"* with *"the encoder moved underneath it."* That is a
  measurement-validity question about drift itself, not about any arm.

* ⛔ **STILL CONFOUNDED:** `o5_k` 8→4 moves with the freeze in this pair. ⚠️ And note its
  direction — the LOW-drift arm used the SHORTER rollout, which cuts **against** the
  hopeful reading of MM-E19's drift read (already flagged low-power).

* **CLOSES WHEN:** one arm resolves it — `postrain30k` + `--freeze-encoder`, **one
  variable**, matched otherwise. Reads ≈0.199 ⇒ freezing explains it and P3's lever
  dissolves. Reads ≈0.669 ⇒ `o5_k` is the cause and the horizon story gains a second
  front. *(Was "defined and unrun"; it is now designed, one-variable, and costed at one
  tiny-rig arm.)*

## P5 — ⭐⭐ THE PREDICTOR ADDS NOTHING OVER THE CURRENT LATENT (L3)

⚠️ **ADDED 2026-08-31 after the PI asked about decodability — my first list omitted it,
and it belongs here.**

* ✅ **DECODABILITY ITSELF IS NOT OPEN. L1 and L2 are MEASURED and PASSED:**

  | level | criterion | result |
  |---|---|---|
  | **L1** no collapse | participation (σ²) val-side **≥ 8.56** (frozen DINOv3) | **3.80/3.62 → 25.58/26.96** ✅ |
  | **L2** the latent CARRIES the environment | beat raw-pixel floor **AND** constant **AND** frozen DINOv3, paired | `splitp30k` `n_agents` **+0.1220 vs DINOv3 +0.0998**; `occ_center` +0.3351; corridor +0.2080 ✅ |

  ⭐ **Our own trained encoder beats the frozen teacher we distil from, on the target
  that matters most (other agents).** ⛔ C156 retracted the opposite claim — a
  recommendation to replace our encoder with DINOv3 that quoted *"behind on 4 of 5"*
  while omitting the one target we win. **The encoder learns the environment.**

* ⛔ **WHAT IS OPEN IS L3 — and the programme's own words call it "the dissociation gate,
  and the actual v7 bar":** does `zhat` beat `z_t` on the same targets, paired,
  |t| ≥ 2.9? ⇒ *does the PREDICTOR add anything over simply looking at now?*

* ⭐⭐⭐ **AND L3 IS NOT A SEPARATE TOPIC TO CHECK LATER — IT IS P2 AND P3 SEEN THROUGH
  THE DECODABILITY LENS.** "The predictor adds nothing over `z_t`" and "Δz is largely
  predictable from `z_t`" (drift) and "the action barely moves the prediction" are three
  instruments on **one** phenomenon: **the predictor is not transporting the scene, it is
  restating it.** That is precisely *"the model did not learn the driving environment and
  extract from it the trajectory."*

* ⭐⭐ **THE OBSERVATION THAT TIES IT TOGETHER, and it is uncomfortable:** `splitp30k` is
  simultaneously **the L2 WINNER** (best representation), **the lowest-drift arm**
  (0.199), **the FROZEN-ENCODER arm** (0 of 41 tensors moved — measured today), **and the
  L3 DELIBERATE-REGRESSION ARM** — *"a known predictor-dead arm (t −3.69 / −5.62 /
  −6.26)"*. ⇒ **the best encoder has the deadest predictor.** Encoder quality and
  predictor liveness appear to TRADE OFF, and the freeze may be buying the first at the
  cost of the second.
  ⚠️ **HYPOTHESIS, not causal** — one arm, and `o5_k` 8→4 is still confounded with the
  freeze (P3). But it makes P3's ablation dual-purpose: the same arm reads drift **and**
  L3, so it can test the trade-off directly.

* **CLOSES WHEN:** an arm clears **L3** — `zhat` beats `z_t` paired at |t| ≥ 2.9 — while
  the deliberate-regression arm (`splitp30k`) still FAILS it. ⛔ Per the validation
  standard: *if the gate does not fail the regression arm, a PASS means nothing.*

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
