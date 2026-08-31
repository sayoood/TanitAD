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
  | ⭐⭐ **(d) TARGET CONSTRUCTION — our target is TEACHER-FORCED, so it already contains the action's effect and an action-invariant solution is admissible** | ⛔ **untested, and the only candidate with a published mechanism, magnitude AND fix** |

* ⭐⭐⭐ **(d) ADDED 2026-08-31, AND I VERIFIED THE SOURCE MYSELF RATHER THAN RELAYING IT.**
  The Lab agent cited `UWM-JEPA` (arXiv **2605.25313**, Radha & Goktas). I fetched the
  paper: it exists, title and authors match, and its abstract states the claim **more
  strongly than the agent did** —

  > *"Action sensitivity itself requires training against counterfactual rather than
  > teacher-forced targets, **a finding that applies beyond the unitary parameterisation**."*

  ⇒ the authors themselves generalise it past their own architecture. And the transfer to
  us is **MEASURED in our source, not assumed**: our `--o5-target` modes (`live`/`ema`/
  `frozen`) **all** encode the real observed future, so we are teacher-forced in exactly
  their sense.

* ⭐⭐ **A SECOND SENTENCE IN THAT ABSTRACT INDEPENDENTLY DESCRIBES OUR P5:**
  > *"both nevertheless tie on a held-out context probe, **locating the separation in the
  > predictor rather than the encoder**."*

  That is our pattern exactly — **L2 passes** (our encoder beats frozen DINOv3) while
  **L3 is open** (the predictor adds nothing over `z_t`). An independent group reports the
  same dissociation and locates it in the same place.

* ⚠️ **NECESSARY, NOT SHOWN SUFFICIENT — and the abstract is the reason to be careful:** a
  *"parameter-matched LSTM-JEPA trained under the same counterfactual-target objective and
  action head collapses to majority-class accuracy (0.53)"*. ⇒ counterfactual targets did
  **not** rescue their vector-latent baseline; latent geometry mattered too. **Switching
  our target construction may be necessary and still not sufficient**, which is a reason to
  test it cheaply before betting a scaled run on it.
  ⛔ And the published fix uses *"simulator-state access during training"* — **we have no
  in-loop simulator**, so the transfer is not free. ⭐ But **O11 is the simulator-free
  version we already have**: it builds negatives from other batch elements' real action
  sequences instead of simulated counterfactuals.

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

* ⭐⭐⭐ **2026-08-31 — THE PREDICTOR HAS ALREADY BECOME ACTION-SENSITIVE ONCE, AND THE ARM
  WAS ABANDONED AT 25 %.** Found by checking whether the programme had already answered
  this before proposing it as new. `O11` is a **counterfactual-action InfoNCE**: roll the
  same states under the true future actions and `n_neg` action sequences from other batch
  elements, and require the true one to match the observed future.
  ⭐ **It carries its own known-value control by construction:** *"an action-independent
  predictor scores EXACTLY ln(1+n_neg) and cannot do better."*

  **`o11p30k`, the ONE arm ever to run it (`w_o11_cf 1.0`; 28 other arms ran 0):**

  ```
  steps 1000-5200   o11_loss 1.3813-1.4462   floor 1.3863   excess ~0   pick_acc ~chance
  step  5400        o11_loss 0.4024          excess 0.9839  pick_acc 0.750   <- BREAKOUT
  steps 6000-7600   pick_acc 1.000 for 12 CONSECUTIVE rows, excess -> 1.386 (max 1.3863)
  ```

  ⇒ **It sat at the provable no-information floor for 5,200 steps, then went to essentially
  perfect action discrimination and STAYED there.** ⭐ The floor is a *mathematical property
  of the instrument*, not a cross-arm comparison, so the within-arm transition survives every
  confound below: **this predictor's output became action-dependent.**
  ⛔ **The run stopped at 7,600 of 30,000 — killed, not crashed** (the log ends cleanly, no
  error lines). `ckpt.pt` is banked and probeable.

* ⛔ **TWO REASONS THIS IS NOT YET A RESULT, both of which I would rather state than have
  someone discover:**
  1. **`o11p30k` IS NOT A ONE-VARIABLE ARM.** Against `postrain30k` it differs on **four**
     shared keys — `w_o11_cf 0→1`, `o11_k 6→4`, `o11_negs 1→3`, and ⛔ **`init_from`
     `distill_init.pt` → `None` (it trained from SCRATCH)**. So *O11 caused the breakout* is
     not established; only *the breakout happened* is.
  2. ⛔ **A NON-DYNAMICAL SHORTCUT IS AVAILABLE.** The negatives are action sequences from
     **other batch elements — i.e. other clips**. Since actions correlate with scene identity
     (clips differ in speed and geometry), *"which action produced this future"* may be
     solvable as *"which action belongs to this scene"*, which requires no dynamics at all.
     Perfect `pick_acc` is consistent with both.

* ⭐ **THE DISCRIMINATING READ IS CHEAP, DEFINED, AND USES A BANKED CHECKPOINT:** run the
  MM-E10 action-divergence probe on `o11p30k/ckpt.pt`, same corpus and n. The three 30k arms
  sit at **0.00408 / 0.00416 / 0.00595**. If o11p30k is **materially above that band**, the
  sensitivity is general and P2 has its first positive lever. If it sits **inside** the band,
  the discrimination lived in the O11 head and never reached the predictor — a shortcut, and
  the arm is a negative rather than an abandoned success. ⚠️ Needs the GPU; queued behind
  `k60p30k`.

* **CLOSES WHEN:** (a), (b) and (c) are each measured, or one of them is confirmed and
  the fix raises the action ratio materially. ⚠️ **(b) additionally needs a PI call on the
  geometry** before any comma2k19 arm — that decision is named above and is not mine. ⚠️ **(b) additionally needs a PI call on the
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

## ⛔ THE GPU QUEUE — ~50 h of committed work on ONE machine, and the ordering is a PI call

Thor is the only compute. Every item below is designed, costed and unblocked *except* by the
queue itself. ⚠️ **I am not silently ordering these** — the sequence decides which gate item
closes first, which is a programme choice.

| # | experiment | cost | closes / serves | dependency |
|---|---|---|---|---|
| **0** | `k60clip05p30k` — **RUNNING** (step 600, gnorm 1.35, 22.6 h left) | 22.6 h | P4 + P2(a) | in flight |
| **1** | ⭐ **actdiv on `o11p30k`'s BANKED ckpt** | **~4 min** | **P2 — decides whether the O11 breakout was real or a scene-matching shortcut** | none; needs only a GPU gap |
| **2** | clip-0.5 control at **k=8** | ~10 h | ⛔ **REQUIRED** before any HORIZON-WORKS verdict — `k60clip05p30k` differs from the incumbent in TWO places now | after 0 |
| **3** | P3 freeze ablation — `postrain30k + --freeze-encoder` | ~8.6 h | P3 **and** P5/L3 (dual-purpose: reads drift *and* whether the freeze buys representation at the predictor's expense) | none |
| **4** | O11 re-run with a matched init | ~8.6 h | P2(d) | ⚠️ **only if #1 says the signal is real** |

⭐⭐ **#1 IS FOUR MINUTES AND IT GATES AN 8.6 h ARM.** Run it in any gap. If `o11p30k`'s
action/scene ratio sits inside the incumbent band (0.00408–0.00595), the O11 discrimination
never left its own head and #4 should not run at all. If it is materially above, P2 has its
first positive lever and #4 becomes the highest-value arm in the queue.

⚠️ **#2 is the one that is easy to skip and must not be.** `k60clip05p30k` now carries
`o5_k 60` **and** `clip 0.5`. Without a k=8/clip-0.5 control, a positive result is
attributable to either — and "we changed the horizon and it worked" would be exactly the
kind of two-variable claim this gate exists to prevent.

### The O11 re-run design (#4), so it is ready if #1 clears it

**Arm:** `postrain30k` + `--w-o11-cf 1.0 --o11-k 4 --o11-negs 3` — O11 **at the settings that
produced the breakout**, with the **distilled init retained**. That isolates the one confound
that most threatens the finding (`o11p30k` trained from scratch, `init_from=None`). ⚠️ The
o11 hyper-parameters are part of *the treatment*, not separate variables; this tests
reproducibility, not tuning.

⛔ **PRIMARY READ IS THE actdiv RATIO, NOT THE o11 METRIC.** `o11_excess` leaving its floor
only proves the O11 head discriminates; the question is whether the **predictor** became
action-sensitive. Incumbent 0.00595.

⭐ **THE ANTI-SHORTCUT CONTROL, and it is the part that makes the arm worth running:** O11's
negatives are action sequences from **other batch elements — other clips**. Since actions
correlate with scene identity, *"which action produced this future"* may be solvable as
*"which action belongs to this scene"*, needing no dynamics. ⇒ **a same-clip-negatives
variant** (negatives from other time windows of the SAME clip) must be run, or the result is
uninterpretable. If discrimination survives same-clip negatives it is dynamical; if it
collapses to the floor it was scene-matching.

⛔ **CORRECTION — I CALLED THIS "a small code change to the negative sampler". IT IS NOT, AND
THE REASON MATTERS.** Read the sampler (`train_v6_staged.py:3253-3265`):

```python
off = 1 + (q % (B - 1))
fa_neg = torch.roll(fa3, shifts=off, dims=0)   # negatives = another BATCH ROW's future actions
```

⭐ The existing code is already careful — a cyclic **roll** is a derangement by construction,
and its comment records why `randperm` would be wrong: it fixes points with probability ~1/B,
and a fixed point silently makes that row's "counterfactual" the TRUE action, pulling the loss
toward the floor and **reading as action-blindness that is not there**. Same defect class the
actdiv probe guards against, already handled here.

⛔ **But a SAME-CLIP negative cannot be obtained by changing that roll.** Batch rows are windows
sampled i.i.d. from **319,002 windows across 2,400 episodes**; at batch 8 the chance any two
rows share a clip is **~1 %**. Same-clip negatives barely exist in a random batch, so the
control needs **grouped batch construction** — a sampler change, not a loss change.

| candidate control | what it tests | real cost |
|---|---|---|
| grouped same-clip negatives | cleanest: same scene, different action | ⛔ batch-construction change **and** a parity question — grouping changes the sampling distribution, so the arm is no longer same-data comparable |
| ⭐ time-shifted negatives from the window's OWN clip | same scene family, different action, no regrouping | needs `future_actions2` **longer than `o5_k`** — at k=60 that is ≥60+shift steps per window. ⚠️ **Measure the dataset's available future-action horizon first**; if it equals `o5_k`, this is not free either |

⚠️ **Neither is small, and pretending otherwise would have sent someone down the wrong one.**
The control is still **REQUIRED** — without it a positive O11 result cannot be distinguished
from scene-matching — but it must be scoped honestly **before** the arm is queued, and the
dataset's future-action horizon is the first thing to measure.

## ⭐⭐⭐ refav1 ALREADY IMPLEMENTS WHAT P4 IS ASKING FOR — and inherits P2(d) unaddressed

**Reviewed 2026-08-31 on a PI question about leveraging DINOv3.** `refav1` is not a plan —
it is **TRAINING-READY**: `stack/tanitad/refs/refa_v1.py` (`RefAV1Config`, `DINOV3_GEOMETRY`),
a trainer, a successor `RefAV1Prime`, a design doc with nine sourced changes, **39/39 tests
green, CPU+GPU smokes measured. Never launched** — v6F owned Thor, then v7 did.

### ⭐⭐ The convergence, and it is independent

**Change 9 is the temporal-abstraction ladder, built in August:**

```
operative  d0.2 s x 30 = 6.0 s      ratios 1 : 3 : 7.5
tactical   d0.6 s x 10 = 6.0 s
strategic  d1.5 s x  4 = 6.0 s
```

MM-E15 derived **1 : ~3 : ~15** off the label bands on 2026-08-31, with no knowledge of this
file. ⇒ **two independent derivations of the same ladder** — one from a PI directive plus the
literature, one from the corpus. ⭐ And it **structurally avoids the defect now killing v7f
arms**: the deepest chain is **30 steps, not 60**, so the full-chain BPTT instability
(gnorm → 2.1e9 at clip 1.0; a persistent spiking regime at clip 0.5) **cannot arise in the
same form**. The Deployment finding — a flat K=300 rollout is undeployable — does not bite it
either.

⇒ **P4's answer may already be implemented and unlaunched.**

### ⛔ But it inherits P2(d), and that is NOT among its nine changes

**Change 4 — *"primary loss = predict future patch features (L2)"*** (DINO-WM's recipe) — is
**teacher-forced**: the target already contains the action's effect, so an action-invariant
solution is admissible. That is P2(d) exactly, and UWM-JEPA states the finding *"applies
beyond the unitary parameterisation."*

⛔ **MEASURED: `refa_v1.py` and `refa_v1_train.py` contain NO counterfactual-action term** —
no InfoNCE over actions, no negatives. ⇒ the defect this campaign spent a week localising is
inherited by refav1's primary objective, **unaddressed**, and would present as REF-A's
original symptom: a model that scores on context and ignores its actions.

⭐ **The fix is small and already written.** O11 is implemented, is the simulator-free form of
the published counterfactual-target fix, and is the one term that ever moved our predictor
off the no-information floor. **Adding it to refav1 before launch** would make it the first
arm carrying **both** the horizon structure and a non-teacher-forced target.

### ⚠️ Two tensions, stated rather than discovered later

1. **Our TRAINED encoder beats frozen DINOv3 on `n_agents`** (+0.1220 vs +0.0998) — the
   target the record calls the one that matters most — while being behind on the other four
   (see P-5). A frozen-DINOv3 arm therefore **starts behind us on agents and ahead
   elsewhere**. ⇒ refav1 is a **different trade, not a strict upgrade**, and its result must
   be read that way.
2. ⭐ **`RefAV1Prime` already exists** with an `ActionStreamPredictor` (`n_act_tokens = 2`,
   deliberately set at **parameter parity** so a win isolates the design choice rather than
   capacity). That is a **P2(c) intervention designed before we had the diagnosis** — the
   action entering as tokens rather than as FiLM conditioning.

⭐ **And the design doc's account of why REF-A failed is a good one:** REF-A had
*"configuration A's consumer with configuration B's encoder class and neither one's
compensating strength."* Frozen encoders succeed either as a **huge frozen VLM** with a wide
interface and supervised head (FROST-Drive: a frozen 14 B **beats the same encoder
fine-tuned**), or as a **moderate frozen encoder with future-feature prediction AND test-time
planning** (DINO-WM, V-JEPA 2-AC). v1 commits to configuration B in full.

## ⛔ refav1's STAGE-1 CACHE — the blocker is the LADDER'S FRAME GRID, not the disk

Went to wire refav1's DataLoader. The trainer never touches an image — it consumes a
stage-1 cache: `<episode>.pt` → fp16 `[T, 640, 1024]`, DINOv3 ViT-L/16 patch tokens,
CLS discarded, 256×640 at 120°, and it **refuses** a cache whose geometry disagrees.

⚠️ **RETRACTED, SAME TURN, BEFORE IT WAS QUOTED.** My first pass priced this against
**B1 (4,713 ep)** and reported "2.9× the free space, unfixable." The design doc specifies
the **2,400-episode parity corpus**, not B1. Corrected below. *Root-cause class: I
substituted the corpus the rest of the campaign uses for the one this design names, and
did the arithmetic before reading which corpus it was for.*

**MEASURED** (`jpeg_len` on 15 sampled episodes per corpus, both 256×640 cylindrical):
`T = 201` frames/episode · parity **2,400** ep · B1 **4,713** ep · Thor **426 GiB free** of 937.

```
per frame  1.250 MiB   (640 tokens x 1024 dims, fp16)

                          PARITY 2400        B1 4713
every frame  (0.1 s)        588.9 GiB       1156.4 GiB
every 2nd    (0.2 s)      ⭐ 295.9 GiB        581.1 GiB
```

### ⭐⭐ THE ACTUAL CONSTRAINT: the ladder's rates must share a frame grid

The three rates are **0.2 / 0.6 / 1.5 s** on a 10 Hz corpus — **2 / 6 / 15 frames**.

```
gcd(2, 6, 15) = 1 frame = 0.1 s
```

⛔ **`str_dt = 1.5 s` is an ODD number of frames.** Strategic targets land at
+1.5 / 3.0 / 4.5 / 6.0 s = frames **15 / 30 / 45 / 60** — two of the four are off any
every-2nd-frame grid. ⇒ **the ladder as designed forces caching EVERY frame**, i.e.
**589 GiB against 426 GiB free — short by 163 GiB.** The disk is not the defect; the
rate choice is what doubles the requirement.

### ⇒ The one-line fix, and it may be a BETTER ladder

Every `str_dt` below keeps the **binding §4b 6.0 s horizon exactly** and keeps
operative:tactical at 1:3, while landing on the even grid so **295.9 GiB suffices**:

| `str_dt` × steps | ratios | note |
|---|---|---|
| 1.2 s × 5 | 1 : 3 : 6 | nearest to the designed 1 : 3 : 7.5 |
| 2.0 s × 3 | 1 : 3 : 10 | |
| **3.0 s × 2** | **1 : 3 : 15** | ⭐ **VERIFIED** — the ratio MM-E15 read off the corpus, carried in `GOALS_AND_CLAIMS` D-HORIZON-LADDER, `PREREG_MM_E16` and the A&I knowledge base. ⚠️ but only 2 strategic steps |

⭐ **Recommendation: `str_dt = 1.2 s × 5`.** It is the smallest departure from the
designed ladder, keeps five strategic steps, and turns an unbuildable cache into one that
fits with **130 GiB headroom**. ⛔ It is still a design change to a PI-directed structure
(change #9), so it is the PI's call, not mine.

### What this does and does not block

* ✅ **refav1 on the PARITY corpus is buildable today** once the rate lands — 296 GiB fits.
* ⛔ **refav1 on B1 is not**: 581 GiB even on the even grid. So refav1 cannot join v7f /
  refcv3 / refd as a same-corpus arm without freeing ~155 GiB or quantising.
* ⚠️ **MEASURED: no DINOv3 cache exists anywhere on Thor** — unstarted work, not a
  half-built asset. Encode cost is one gradient-free pass, resumable.
* ⚠️ **The trainer is still a 178-line scaffold** (`SmokeData`, real DataLoader "wired in
  later"). The model, its nine changes and change #10 are complete and tested; the data
  path is not. That is the second launch blocker and it is code, not a decision.
* ⛔ **Narrowing the interface is NOT on the table** — change #3 forbids it on
  FROST-Drive's measured 8.17 → 7.68 width ablation. It is the defect v1 exists to remove.

### ⭐⭐ AND THE SAME ARITHMETIC EXPOSES A REAL DESIGN QUESTION

`1 : 3 : 15` is not a free parameter I picked to fit the disk — **MM-E15 read it off the
corpus** (D-HORIZON-LADDER, `s2_labels_v7.2_train` md5 `0ff90213…`, 4,572 records). Put
that ratio into refav1's ladder and the strategic rung becomes `str_dt = 3.0 s`. With the
designed `str_steps = 4` that reaches **12.0 s** — and MM-E15's median manoeuvre start is
**12.5 s**, inside the `strategic_s [8, 30]` band.

⛔ **refav1's strategic rung, as designed, reaches 6.0 s — BELOW the strategic band's near
edge of 8 s.** So it is "strategic" by *rate*, not by the band its labels occupy. That is
the same defect MM-E15 found in the v7 arms, reproduced one level up: a level named for a
horizon it does not reach.

⚠️ **This is a PI question, not a fix I can apply**, because it touches the §4b horizon:

> §4b binds the **control output** to 6 s. Does it also bind the **strategic context
> predictor**, whose job is to FiLM the level below it rather than to emit a trajectory?

If it does not, `str_dt = 3.0 s × 4 = 12.0 s` gives a strategic level that (a) reaches the
band its own labels live in, (b) matches the corpus ratio exactly, (c) lands on the even
frame grid, and (d) still emits control only to 6 s. If it does, `1.2 s × 5 = 6.0 s` is the
even-grid option that stays inside the letter of §4b.

⭐ Either way the cache is **295.9 GiB and fits**. The horizon question is orthogonal to
the storage question — it just happened to surface from the same arithmetic.


## ⏳ The nearest decision point

`k60p30k` reads in ~19 h. **HORIZON-WORKS** closes P2(a) and most of P4.
**HORIZON-INERT** promotes **P2(b) — the untested command channel — to the front**, and
that is the deeper question: *we have never given this model a command, only a
description of what it already did.*
