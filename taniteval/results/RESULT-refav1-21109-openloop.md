# refav1 @ step 21,109 — the epoch-end OPEN-LOOP four-family read

**Arm** `refav1-b1-v72-ep3-speed`, step **21,109** (the full epoch) · **checkpoint**
`/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt`, 2,122,997,633 B,
md5 `1189bc020018c2c67ce03d566c390285`, 407 tensors / 182,459,701 params, **strict load**
(`missing_keys: []`, `unexpected_keys: []`) · **corpus** the full **141-clip v7.2 EVAL split**
(`/home/nvidia/data/refav1-fp8-eval`, labels md5 `aa12c948f062181c3297265b51526ec5`, join
141/141 eps, 0 missing) · **n = 282 windows over 141 episode clusters**, `--window-stride 40`,
K = 10 at dt 0.2 s (2.0 s), plan `{samples 300, iters 30, elites 30, seed 0}`,
`--action-units kappa` (the legacy reading — see W4) · **estimator** episode-cluster bootstrap
(`taniteval/ci.py`), `n_boot 2000`, cluster unit = the episode, paired across arms on the same
windows; ⛔ `overlapping_holdout_se` appears nowhere · **compute** rollout on the Jetson Thor
(`OMP_NUM_THREADS=6`, 10,898 s), analysis on the dev box **CPU, 0 GPU** · **tag**
`refav1-21109-openloop` · **owner** Architecture & Inference FlyWheel

> ⛔ **EVERY ARM HERE IS OPEN LOOP** (PI ruling 2026-09-02). The model controls nothing; the ego
> data keeps arriving from the recording. The `T0`/`T1` stamps the record carries are the
> doctrine's *conditioning* labels — teacher-forced vs the arm's own actions — not loop labels.
> The words "closed loop" describe nothing in this document, and the suite's vocabulary guard
> confirms **0 occurrences** in either rendering.

---

## §0 THE HEADLINE — the question this read was armed to ask, answered

⛔ **THE PLANNER STILL DOES NOTHING, AND THE FULL SPLIT MAKES THE STATEMENT SHARPER RATHER THAN
WEAKER.** The previous 20-clip reads found `cl` bit-identical to the constant-velocity floor
`ha0` on **120/120** and **140/140** windows. On 141 clips the identity is **270/282 = 0.9574** —
and **every one of the 12 exceptions is a DIFFERENT injected trivial baseline**, a constant
−1.5 m/s² brake. The honest quantities are the shape ones:

| quantity | value | what it means |
|---|---|---|
| `cl` bit-identical to `ha0` (< 1e-9 m) | **270/282 = 0.9574** | max residual **2.7000 m**, mean 0.1149 m |
| `cl_controls` **exactly zero** | **270/282 = 0.9574** | the plan is literally `_baseline_controls["cv"]` / `["hold_v0"]` |
| ⭐ **κ identically zero** | **282/282 = 1.0000** | **the planner never turns. Not once, on any window.** |
| ⭐ **acceleration constant in time** | **282/282 = 1.0000** | it never modulates within the 2.0 s plan |
| ⭐ **distinct plans emitted, in total** | **2** — `a = 0` on 270, `a = −1.5` on 12 | over 282 windows the entire output of a 2,100-candidate search is two scalars, and **both are candidates the planner INJECTS into its own population** |

⇒ **`cl` is a two-valued function, not a planner.** The 12 exceptions are not evidence of planning;
they are the third injected baseline winning instead of the first. **The plan is an injected trivial
baseline on 282/282 = 100 % of windows.**

⛔ **And it is now measurably WORSE than the floor it collapses onto:** paired
`cl − ha0` ADE = **+0.0158 m [+0.0007, +0.0315], separated** — the 12 brake windows cost more than
they buy. LONGITUDINAL `cl − ha0` speed MAE **+0.0308 [+0.0049, +0.0585] separated**, along-track
**+0.0258 [+0.0061, +0.0467] separated**. Every LATERAL and every trajectory-TACTICAL `cl − ha0`
delta is **exactly 0.0000 with a zero-width interval**, because on those families the two arms are
the same array.

⚠️ **Read the LATERAL table as a control, never as skill.** `cl` and `ha0` are identical on every
LATERAL row (heading MAE 2.4620°, cross-track 0.2257 m) and both "beat" `ha` (4.6732°, 0.3826 m) and
`ol` (4.2655°, 0.3351 m). That is a straight line beating a noisy held curvature — the exact failure
`D-REFAV1-PAIRED-READ-VOID` was opened for. The TACTICAL family says the same thing in the only
statistic that can: `cl`'s lateral-decision **κ = 0.0000** against `ha` 0.4814 and `ol` 0.5971.

**Two independent probes agree**, and the report refuses to render if they do not: the record's own
`trivial_profile` and a raw-array recomputation through a different code path both read
**270/282** identical windows over **282** windows — `cross_check: OK`.

---

## §1 WHY — a property of the COST, not of the weights

Four steps, each measured or read from source. Nothing here depends on what the network learned,
which is why an epoch of training changed the answer by zero.

1. **`cv` and `hold_v0` ARE the zero control sequence.** `MEASURED (ours)`: calling
   `tanitad.refs.refa_v1_plan._baseline_controls(PlanConfig(), v0, "cpu", None)` at
   `v0 ∈ {0.0, 1.9116, 10.2758}` returns `[10, 2]` tensors with `max|·| = 0` and `all_zero = True`
   for both; only `decel_1.5` is non-zero (`a ≡ −1.5`, κ ≡ 0). The function's own docstring says so
   (`refa_v1_plan.py:172-179`). **All three are straight lines at constant acceleration** — exactly
   the two shapes §0 measures, and the reason `n_distinct_plans_emitted = 2`.
2. **Those tensors compete INSIDE the CEM loop, every iteration.** `icem_plan` concatenates
   `base_stack` onto `samples` before every `cost_fn` call (`refa_v1_plan.py:252-254`) and
   `best_cem = samples[idx[0]]` — so the "CEM optimum" can BE an injected baseline row, bit for bit.
3. **The cost is dominated by a penalty minimised at exactly zero.** `_cost_chunk`
   (`refa_v1.py:2175-2177`) adds `W_JERK · jerk² + W_KAPPA · κ²` to the goal term. Every
   coloured-noise sample carries non-zero jerk and non-zero κ; so does every non-trivial goal seed —
   an imagined `TURN_R` carries κ ≠ 0, an imagined `ACCELERATE` carries jerk ≠ 0. Only the injected
   constant-accel straight lines pay **zero** penalty.
4. **The world-model term cannot outvote it, by a margin already banked.**
   `D-REFAV1-COST-SURFACE` (`INHERITED`, quoted at `refa_v1.py:273-281`): the modelled response
   along κ over the whole candidate box is **1.63e-10**, against a float32 `1 − cos` representable
   step of **5.96e-08**; at the shipped `W_KAPPA = 0.05` the penalty over the whole box is ~90×
   larger than one representable step of the term it trades against.

⇒ **`plan()` at the shipped settings is a penalty minimiser whose argmin is a constant-accel
straight line.** A better-trained network moves nothing, because the only quantity the search can
resolve is `W_JERK · jerk² + W_KAPPA · κ²`. ⭐ This read is the epoch-scale confirmation:
`D-REFAV1-PAIRED-READ-VOID`'s evidence was step-1,000-era on 20 clips; the whole epoch on 141 clips
changes the identity fraction from 1.0000 to 0.9574 and the trivial-plan fraction not at all.
**"It is early in training" is refuted as an explanation.**

---

## §2 ⛔ AN INSTRUMENT DEFECT FOUND IN PASSING — `baseline_won_frac` UNDER-REPORTS, AND IT IS A FALSE-POSITIVE GENERATOR

`MEASURED (ours)`, the cross-tab of `plan_source` against the CONTENT of `cl_controls`, all 282
windows:

| `plan_source` | controls exactly zero | controls the `decel_1.5` brake | total |
|---|---|---|---|
| `baseline:hold_v0` | 180 | 0 | 180 |
| **`cem`** | **90** | **5** | **95** |
| `baseline:decel_1.5` | 0 | 7 | 7 |

⇒ **`plan_source` reads `cem` on 95 windows (33.7 %), and on 95/95 of them the returned controls are
bit-exactly an injected baseline** — zero on 90, the −1.5 brake on 5 (verified per window: `a` at
every step exactly −1.5, κ exactly 0, path residual against `ha0` exactly 2.7000 m, which is
`a·dt²·Σ(k−1) = −1.5 × 0.04 × 45`). The record therefore prints `baseline_won_frac = 0.6631` where
the true trivial-plan rate is **1.0000**.

**Why neither existing repair fires: both are gated on the LABEL, not on the content.**
`icem_plan`'s `<=` re-attributes only on an EXACT cost tie (`refa_v1_plan.py:277-281` — and that
comment records the same failure being fixed once already: *"the floor HELD but the provenance
LIED"*), and `plan()`'s `hold_v0`-preference runs only `if res.source.startswith("baseline:")`
(`refa_v1.py:2203`), which a `cem` label skips entirely.

⚠️ **HYPOTHESIS, not measured here:** the baselines are scored **twice** — inside the loop in a
`cost_chunk`-sized batch and again afterwards in a batch of 3 — and two evaluations of the same
tensor at different batch shapes need not return the same float, so the exact tie never forms. The
discriminating test is one GPU-minute (W2) and it decides which repair is correct.

⇒ **`plan_source` is not evidence about what produced a plan. The CONTENT of the controls is.**
A reader who trusts `baseline_won_frac` concludes that the search wins a third of the time. It never
wins.

---

## §3 WHAT THIS DOES *NOT* SAY — and what the epoch actually bought

⛔ **This is not an indictment of the trained model, and reporting it as one would be exactly the
`true-but-wrong-for-the-reader` failure.** It says the `cl` arm cannot see the model. Three
measurements in the same record DO read the weights, and two of them are positive.

**⭐ 3.1 The world model has learned dynamics — it beats the raw-input floor, and the margin GROWS
with horizon.** Teacher-forced (T0), standardised DINOv3 space, n = 282 / 141 clusters:

| horizon | model | persist-last-field (the raw-input floor) | paired `const − model` | separated |
|---|---|---|---|---|
| step 1 (0.2 s) | 0.3052 | **0.2610** | ⛔ **−0.0442** [−0.0553, −0.0339] | **yes — the control WINS** |
| step 5 (1.0 s) | 0.3813 | 0.4942 | +0.1129 [+0.0960, +0.1286] | yes |
| step 10 (2.0 s) | 0.4253 | 0.5959 | +0.1705 [+0.1518, +0.1873] | yes |
| step 20 (4.0 s) | 0.4690 | 0.6877 | +0.2188 [+0.1993, +0.2363] | yes |
| step 30 (6.0 s) | 0.4952 | 0.7406 | **+0.2454** [+0.2259, +0.2625] | yes |
| mean over steps | 0.4387 | 0.6173 | +0.1786 [+0.1613, +0.1940] | yes |

**The constant-only control reads its KNOWN value**: `feat_mse_zero = 1.0294` [1.0217, 1.0373]
against a target variance of ~1 (`tgt_std_mean 0.9539`) — the panel is readable. ⚠️ And the one
honest negative is stated rather than averaged away: **at a single 0.2 s step the model LOSES to
persistence**, separated. It earns its keep from 1.0 s outward.

**⭐ 3.2 The declared LONGITUDINAL tactical head has vision-derived skill that survives nav
withdrawal.** Under `nav_zero` — the deployment condition — it reads accuracy **0.4326**
[0.3546, 0.5177] against a majority-class floor of **0.2979**, κ **0.2378**. The CI excludes the
floor, and this is its BEST conditioning, so it is not a nav echo. (LAT is at its floor: `nav_zero`
0.6738 = floor 0.6738, κ 0.0000.)

**⛔ 3.3 The strategic route head is a pure echo of its own input, and now decisively so at n = 141.**
`nav_true` accuracy **1.0000**, κ 1.0000 — a bijection of the token it is fed. `nav_shuffled`
**0.4184**, κ **−0.1427** (below chance). `nav_zero` **0.6383**, κ **0.0000** — exactly the
majority-class rate. On the 82 windows where the shuffle actually changed the token, the head
follows the **shuffled nav on 1.0000** and the **label on 0.0000**. ⇒ **zero route skill from
vision**; the head reproduces its input. (`paired true − shuffled` = +0.5816 [+0.4965, +0.6596] is
the echo magnitude, not skill.)

**⭐ 3.4 The tactical decoder — the lever `D-REFAV1-COST-SURFACE` identified — HAS moved.** Its
imagined goal over 282 windows: LAT `LANE_KEEP` 244 / **`TURN_R` 31 / `TURN_L` 7** = **13.5 % turns
asked for**; LON spread over **four** classes (`CRUISE` 116, `ADAPT_SPEED_FOR_CURVE` 78,
`ACCELERATE` 55, `BRAKE_TO` 33). The register's incumbent decoded `LANE_KEEP` on **140/140** and
asked for a turn on **0 of 27** turning windows. ⇒ **the decoder now asks for turns; the planner
still never executes one** (κ ≡ 0 on 282/282). The bottleneck has moved downstream of the decoder,
into the cost.

**3.5 `ol`, the kinematic contract**, is the arm that bounds what any planner in these units could
reach: ADE **0.4237** [0.3466, 0.5070] against the floor's 0.5316 — i.e. the recorded actions
integrated by the programme's own unicycle beat constant velocity by ~0.11 m, and `cl − ol` is
**+0.1237 [+0.0687, +0.1770] separated**. ⚠️ Its LATERAL rows carry the known ~2.9× over-rotation
of the legacy `kappa` reading (W4).
---

## §4 THE refav1-vs-refcv3 COMPARISON THE PI ASKED FOR

`D-HF-COMPARABILITY` binds the statistic to **each arm's margin over the same trivial floor, per
family** — `cl − ha0` for refav1 against `os − ha0` for refcv3 — ⛔ **never `cl` against `os` as
levels**. Both reads are on the **same 141-clip v7.2 EVAL split** and `ha0` is bit-identically
defined on both sides (`a = 0`, `κ = 0` at the measured `v0`, the same integrator; and it is exactly
zero in either action unit, so the steer/κ contract cannot touch it).

| | refav1 @ 21,109 (this read) | refcv3 @ 40,284 (`RESULT-refcv3-40284-openloop.md`) |
|---|---|---|
| deployed arm | `cl` | `os` |
| **margin over `ha0`, ADE** | ⛔ **+0.0158 m** [+0.0007, +0.0315] — **WORSE than the floor, separated** | ⭐ **−0.2304 m** [−0.2881, −0.1781] — **better than the floor, separated** |
| margin over `ha` (hold-action), ADE | +0.0083 [−0.0574, +0.0703] not separated | ⛔ +0.1423 [+0.1187, +0.1658] — loses to hold-action |
| arm degenerate? | ⛔ **yes** — constant-velocity on 0.9574, κ ≡ 0 on 1.0000, **2 distinct plans** | no — `trivial_frac 0.0000`, 51 distinct anchors |
| n | 282 windows / 141 clusters | 4,823 windows / 141 clusters |

**The answer, stated plainly.** refav1's planner does not merely fail to beat refcv3 — **it does not
reach its own trivial floor**, and the margin it contributes is a small positive number in the wrong
direction. refcv3 clears the floor by 0.23 m and still loses to the hold-action control. ⇒ on the
open-loop surface **refcv3 is ahead of refav1 by roughly the whole of its own margin**, point
estimate ≈ **0.246 m ADE** in refcv3's favour.

⚠️ **What that number is NOT.** It is a **difference of two independently estimated margins**, not a
paired episode-cluster bootstrap over shared windows: the two dumps sit on different window grids
(refav1 at 0.2 s × 10 from cache index `t`, refcv3 at 0.5 s × 4 from provider index `ws + 2`), so
`taniteval/tools/paired_openloop.py`'s integer-RAW-frame join finds no common instants between a
stride-40 refav1 dump and a stride-5 refcv3 dump. Each margin carries its own CI above; the
DIFFERENCE does not carry one. That is W6.

⚠️ **And the levels are not comparable, which is why only margins are quoted:** the two reads score
different window sets, so even `ha0` itself reads 0.5316 m here and 0.6723 m there. A level-to-level
sentence about `cl` and `os` would be meaningless.

---

## §5 CRITERIA COMPLETENESS — the same gate the refcv3 read passed

Run through `taniteval/tools/openloop_suite.py --arm-json … --dump-dir …` against
`products/P7-TanitEval/CRITERIA_REGISTRY.json` **v2.5.0**:

* **VIOLATIONS (silently absent, required): 0**
* **WORK ITEMS (refused with a reason, or partial): 3** — the two STRATEGIC rows of
  `four_families` (UNAVAILABLE by construction on a trajectory-only dump; the declared route head IS
  reported, from the sidecar — §3.3) and `strategic.echo_test` (the suite reads refcv3's key path;
  refav1 publishes the same echo evidence under `refav1.strategic.conditionings`, so this is a suite
  shape mismatch, not a missing measurement — W7)
* **constant-only control (`const0`): OK** — the harness reads a known value, so the panel is readable
* **loop-vocabulary guard: CLEAN** — 0 occurrences of the forbidden phrase in either rendering
* LONGITUDINAL 4/4 present incl. **distance-keeping** (headway / time-gap / min-TTC, lead block
  joined on 141/141 episodes, coverage `LEAD 90 / NO_LEAD 63 / NOT_STRAIGHT 108 / NO_LABEL 21`,
  speed-check max 1.3e-05 m/s); LATERAL 4/4 present.

---

## §6 WORK ITEMS — escalated, not silently parked

| # | item | why it matters | owner |
|---|---|---|---|
| **W1** | ⛔ **`plan_source` must attribute by CONTENT, not by label** — compare `best_ctrl` bitwise against each `_baseline_controls` entry before naming the source, or score the baselines once and reuse the in-loop costs. | `baseline_won_frac` reads 0.6631 where the truth is 1.0000. A reader who trusts it concludes the search wins a third of the time; it never wins. This is a false-positive generator. | ArchInf — **PI / Master-Mind call**: it moves banked provenance strings in the live planner. |
| **W2** | **Settle §2's batch-shape hypothesis with one GPU-minute**: evaluate the same zero-control tensor inside a `cost_chunk`-sized batch and alone, print both floats. | It is the only part of W1's mechanism that is a hypothesis, and it decides which W1 repair is correct. | ArchInf |
| **W3** | ⭐ **A `--no-plan-arms` mode for `refav1_arm.py`.** | `plan()` is ~40 s/window on Thor and 100 % of the cost; `ha`/`ha0`/`ol` are pure kinematics. It would turn a `--action-units steer` re-read of the LATERAL family from a 3.5 h job into minutes — and `cl`/`ha0` are provably unchanged by the unit flip, because zero is zero in either unit. | Benchmarks & Evals |
| **W4** | **The LATERAL rows here are the LEGACY `kappa` reading.** The corpus stores `arctan(L·κ)` (`REFAV1_ARM.md` §3b), so `ol` and `ha` over-rotate by ~2.9×; `cl` and `ha0` are unaffected (both exactly zero). | Every `ol`/`ha` LATERAL number carries a known inflation, so their margins over `ha0` are conservative. Stamped, not re-decided. | already ruled (`D-STEER-INTERFACE-RESOLVED`) |
| **W5** | ⛔ **A PREFLIGHT IMPORT PROBE in `refav1_arm.py`.** MEASURED today: the Thor rollout completed all 141 episodes (10,898 s, `REFAV1_DUMP_DONE`) and then `analyze()` died on `FileNotFoundError: …/taniteval/tools/eval_four_families.py`, a module absent from the shipped tree. `T1_EXIT` reads like total failure; it was a 100 %-complete run missing its last step. Recovered with `--analyze-only` on the dev-box **CPU, 0 GPU**. | This is the documented class (CLAUDE.md, *"an analysis-time import that fails after the rollout destroys the run's output while the compute is already paid for"*) recurring verbatim. A 2-second startup probe would have caught it. The ship list must also be derived, not hand-written. | Benchmarks & Evals |
| **W6** | **A paired refav1 × refcv3 read on shared windows.** Today's cross-model number is a difference of two independently estimated margins. `paired_openloop.py` needs the two dumps on a common integer RAW-frame grid — i.e. a refcv3 dump at stride 1, or a refav1 dump at a stride whose origins refcv3 covers. | `D-HF-COMPARABILITY` asks for a paired episode-cluster bootstrap; §4's difference has no interval. | Benchmarks & Evals |
| **W7** | **`openloop_suite.py`'s `strategic.echo_test` reads refcv3's key path**, so a refav1 record reports it as a work item although the evidence exists under `refav1.strategic.conditionings`. | A completeness gate that cannot see a present measurement teaches the wrong lesson. | Benchmarks & Evals |
| **W8** | **STRATEGIC stays UNAVAILABLE inside `four_families`** (`_decision_family` needs `route_pred`/`route_gt`, which `t1_eval`'s window whitelist never forwards). | The binding rule calls a missing family a work item, not a pass. The declared head is reported separately (§3.3). | Benchmarks & Evals |

---

## §7 DELIVERABLE MANIFEST

⚠️ Nothing below lives in only one place except where marked.

| artifact | what it is | where |
|---|---|---|
| `taniteval/results/RESULT-refav1-21109-openloop.md` | this document | repo (staged) |
| `taniteval/results/refav1-21109-openloop.json` | the arm record — four families, intervals, paired blocks, decision sidecar analyses, WM diagnostic, manifest | repo (staged) |
| `taniteval/results/refav1-21109-openloop-identity.json` | the INDEPENDENT identity + control-shape probe and its cross-check against the record | repo (staged) |
| `taniteval/results/refav1-21109-openloop-extra.json` | the `plan_source` × controls-content cross-tab, the imagined-goal token histograms, per-episode identity | repo (staged) |
| `taniteval/results/openloop-suite-refav1-21109.{json,md,html}` | the criteria-completeness gate (registry v2.5.0), `const0` control, vocabulary guard | repo (staged) |
| `taniteval/results/refav1-21109-openloop-dump.tar.gz` | the full 141-episode dump + decisions sidecars + manifest — everything above is reproducible from it with `--analyze-only`, 0 GPU | repo (staged) |
| `taniteval/results/refav1-21109-openloop-run.log` | the Thor rollout log incl. the analysis-time failure of W5 | repo (staged) |
| `taniteval/tools/refav1_openloop_report.py` | the report renderer + the independent probe + the refusal | repo (staged) |
| `stack/tests/test_refav1_openloop_report.py` | its pin — 12 tests, incl. two deliberate-regression arms | repo (staged) |
| `taniteval/tools/REFAV1_ARM.md` | §6 resolved paths + measured Thor wall-clock; §7's six UNVERIFIED items closed | repo (staged) |
| the rollout dump | as produced | also `tanitad-thor-wifi:/home/nvidia/refav1_evalrun/full_dump/` |
| the shipped eval tree on Thor | `stack/tanitad` + `taniteval/taniteval` + two tools, 1.9 MB tar | `tanitad-thor-wifi:/home/nvidia/refav1_evalrun/repo/` — **a copy of repo content, nothing original** |

**Boxes used.** Rollout: Jetson Thor (`tanitad-thor-wifi`), sole occupant, 3.03 h. Analysis and every
number in this document: the dev-box CPU, 0 GPU. ⚠️ A dev-box companion run over the banked 20-clip
slice at stride 5 was **stopped after 1 of 20 episodes and yielded**: a sibling's
`refc_v3_train.py --arm hier --size tiny` started on the same 8 GB card 5 minutes after it and the
GPU reached 7,797 / 8,188 MiB across four jobs. Its one completed episode is kept as the **cross-box
confirmation**: 14/14 windows bit-identical to `ha0` with `cl_controls` exactly zero, on a different
GPU and a different CUDA stack (`devbox_partial_identity.json`).

---

# APPENDIX — the instrument's own rendering, unedited

Everything below is the verbatim output of
`taniteval/tools/refav1_openloop_report.py --record … --dump …`. Every level, interval and paired
delta is READ from the record (which computed them with `taniteval/ci.py`'s episode-cluster
bootstrap); the tool recomputes no family metric. The one thing it computes itself is the identity
and control-shape probe in its §1, which exists to be a SECOND code path over the record's own
`trivial_profile` — and which refuses to render if the two disagree.

# refav1 @ step 21,109 — OPEN-LOOP four-family read, full 141-clip v7.2 EVAL split

> **OPEN LOOP** (PI ruling 2026-09-02). The model is not controlling a vehicle; a predictor consuming its own planner's actions is still open loop. The `T0`/`T1` stamps below are the doctrine's CONDITIONING labels, not loop labels.

## 1. Shape before metrics — is the planner arm the trivial floor?

* **n = 282 windows over 141 episode clusters** (the bootstrap's resampling unit is the episode)
* `cl` **bit-identical** to `ha0` (< 1e-09 m) on **270/282 = 0.9574**; bitwise-equal arrays on 270/282; max residual **2.700006485 m**
* the planner's OWN emitted `cl_controls` are **exactly zero** on **270/282** windows (max |control| = 1.500000000)
* ⭐ **the control SHAPE, which is the sharper statement**: κ is *identically zero* on **282/282** windows (the planner never turns), the acceleration is *constant in time* on **282/282**, and both together on **282/282** = 1.0000
* over those windows it emitted **2 distinct plans in total**, as constant accelerations (m/s²) with counts: `{'0': 270, '-1.5': 12}`
* record's own `trivial_profile`: constant-velocity frac **0.9574**, degenerate arms **['cl', 'ha0']**
* the two probes cross-check **OK** (270 vs 270 identical windows)

## 2. The four families, per arm, never pooled

**LONGITUDINAL**

| | `cl` | `ha` | `ha0` | `ol` |
|---|---|---|---|---|
| target-speed MAE (m/s) | 0.5412 | 0.2706 | 0.5104 | 0.0991 |
| speed bias (m/s) | -0.1140 | -0.0512 | -0.0565 | -0.0725 |
| speed RMSE (m/s) | 0.8888 | 0.4662 | 0.8529 | 0.1199 |
| along-track MAE (m) | 0.4241 | 0.2937 | 0.3984 | 0.2018 |
| along-track final bias (m) | -0.1513 | -0.4073 | -0.0364 | -0.4274 |
| accel MAE (m/s^2) | 0.5499 | 0.3423 | 0.5261 | 0.0802 |
| speed MAE, 95 % CI | 0.5412 [0.4741, 0.6105] | 0.2706 [0.2420, 0.3019] | 0.5104 [0.4452, 0.5797] | 0.0991 [0.0905, 0.1079] |
| along MAE, 95 % CI | 0.4241 [0.3741, 0.4744] | 0.2937 [0.2490, 0.3438] | 0.3984 [0.3490, 0.4515] | 0.2018 [0.1673, 0.2427] |
| distance-keeping: headway (m) / time-gap (s) / min-TTC (s) | 26.2149 / 4.6913 / 22.9028 | 26.9053 / 4.7930 / 24.8532 | 26.1645 / 4.6877 / 22.8378 | 27.0347 / 4.8065 / 25.0674 |

**LATERAL**

| | `cl` | `ha` | `ha0` | `ol` |
|---|---|---|---|---|
| heading MAE (deg) | 2.4620 | 4.6732 | 2.4620 | 4.2655 |
| yaw-rate MAE (deg/s) | 2.1993 | 4.6713 | 2.1993 | 4.0860 |
| curvature MAE (1/m) | 0.0072 | 0.0158 | 0.0072 | 0.0143 |
| cross-track MAE (m) | 0.2257 | 0.3826 | 0.2257 | 0.3351 |
| cross-track final MAE (m) | 0.5888 | 1.0471 | 0.5888 | 0.8895 |
| cross MAE, 95 % CI | 0.2257 [0.1772, 0.2788] | 0.3826 [0.3045, 0.4677] | 0.2257 [0.1772, 0.2788] | 0.3351 [0.2632, 0.4133] |
| heading MAE, 95 % CI | 2.6423 [1.6208, 4.1577] | 4.9263 [3.2832, 7.1880] | 2.6423 [1.6208, 4.1577] | 4.5280 [2.9437, 6.6809] |

**TACTICAL — executed (trajectory-derived, refc 3-way labeller)**

| | `cl` | `ha` | `ha0` | `ol` |
|---|---|---|---|---|
| LAT decision acc / kappa | 0.8759 / 0.0000 | 0.8191 / 0.4814 | 0.8759 / 0.0000 | 0.8688 / 0.5971 |
| LON decision acc / kappa | 0.6773 / 0.0546 | 0.8050 / 0.5722 | 0.6809 / 0.0000 | 0.9752 / 0.9487 |
| 5-way collapsed acc / kappa | 0.6277 / 0.0604 | 0.7057 / 0.5206 | 0.6241 / 0.0000 | 0.8617 / 0.7752 |
| tactical goal-point error (m) | 1.3802 | 1.4737 | 1.3448 | 1.0899 |


### 2d. STRATEGIC

* windows 282 · route-labelled **141** · nav-valid frac 1.0000 · excluded (no route label) 141
* ⛔ `nav_true` accuracy is a **nav ECHO index**, not skill — `route_label` and `nav_cmd` derive from the same field. Read `nav_shuffled` / `nav_zero` and the changed subset.

**STRATEGIC — route head vs `route_label`**

| conditioning | n | accuracy | kappa | majority-class floor | acc 95 % CI |
|---|---|---|---|---|---|
| `nav_true` | 141 | 1.0000 | 1.0000 | 0.6383 | 1.0000 [1.0000, 1.0000] |
| `nav_shuffled` | 141 | 0.4184 | -0.1427 | 0.6383 | 0.4184 [0.3404, 0.5035] |
| `nav_zero` | 141 | 0.6383 | 0.0000 | 0.6383 | 0.6383 [0.5603, 0.7163] |

paired `true − shuffled` accuracy: **0.5816** [0.4965, 0.6596], separated **yes**, n_windows 141, n_episodes 141

changed subset (n = 82): follows the LABEL **0.0000**, follows the SHUFFLED NAV **1.0000** — mutually exclusive by construction


### 2e. TACTICAL — declared (selected) half

* vocabulary `v7.0` · windows 282 · the DECLARED (selected) half of the TACTICAL family; the executed half is the trajectory-derived table above, in a DIFFERENT vocabulary — no mapping is invented

**TACTICAL declared — LAT head**

| conditioning | n | accuracy | kappa | majority-class floor | acc 95 % CI |
|---|---|---|---|---|---|
| `lat_nav_true` | 141 | 0.7021 | 0.2886 | 0.6738 *(derived)* | 0.7021 [0.6312, 0.7801] |
| `lat_nav_shuffled` | 141 | 0.6454 | 0.1149 | 0.6738 *(derived)* | 0.6454 [0.5674, 0.7234] |
| `lat_nav_zero` | 141 | 0.6738 | 0.0000 | 0.6738 *(derived)* | 0.6738 [0.5957, 0.7518] |

**TACTICAL declared — LON head**

| conditioning | n | accuracy | kappa | majority-class floor | acc 95 % CI |
|---|---|---|---|---|---|
| `lon_nav_true` | 141 | 0.3901 | 0.2056 | 0.2979 *(derived)* | 0.3901 [0.3121, 0.4752] |
| `lon_nav_shuffled` | 141 | 0.3546 | 0.1613 | 0.2979 *(derived)* | 0.3546 [0.2766, 0.4326] |
| `lon_nav_zero` | 141 | 0.4326 | 0.2378 | 0.2979 *(derived)* | 0.4326 [0.3546, 0.5177] |

paired `true − shuffled` LAT accuracy: **0.0567** [0.0000, 0.1135], separated **no**, n 141

paired `true − shuffled` LON accuracy: **0.0355** [-0.0284, 0.1064], separated **no**, n 141


### 2f. ADE / FDE — one row of four families, ⛔ never "the result"

| | `cl` | `ha` | `ha0` | `ol` |
|---|---|---|---|---|
| ADE (m), 95 % CI | 0.5474 [0.4839, 0.6108] | 0.5391 [0.4544, 0.6301] | 0.5316 [0.4710, 0.5939] | 0.4237 [0.3466, 0.5070] |
| FDE (m), 95 % CI | 1.3802 [1.2132, 1.5442] | 1.4737 [1.2369, 1.7301] | 1.3448 [1.1816, 1.5089] | 1.0899 [0.8781, 1.3256] |

## 3. Paired margins on the same windows

**cl - ha0** — estimator `paired_episode_cluster_bootstrap`, conditioning `T1 minus T1`; both arms OPEN LOOP

| family | metric | delta | 95 % CI | separated |
|---|---|---|---|---|
| ADE | `ade_m` | 0.0158 | [0.0007, 0.0315] | **yes** |
| ADE | `fde_m` | 0.0354 | [-0.0044, 0.0769] | no |
| longitudinal | `LON_speed_mae_mps` | 0.0308 | [0.0049, 0.0585] | **yes** |
| longitudinal | `LON_along_mae_m` | 0.0258 | [0.0061, 0.0467] | **yes** |
| longitudinal | `LON_accel_mae_mps2` | 0.0238 | [-0.0023, 0.0518] | no |
| lateral | `LAT_cross_mae_m` | 0.0000 | [0.0000, 0.0000] | no |
| lateral | `LAT_heading_mae_deg` | 0.0000 | [0.0000, 0.0000] | no |
| lateral | `LAT_yaw_rate_mae_radps` | 0.0000 | [0.0000, 0.0000] | no |
| tactical | `TAC_traj_lat_correct` | 0.0000 | [0.0000, 0.0000] | no |
| tactical | `TAC_traj_lon_correct` | -0.0035 | [-0.0248, 0.0213] | no |

**cl - ha** — estimator `paired_episode_cluster_bootstrap`, conditioning `T1 minus T1`; both arms OPEN LOOP

| family | metric | delta | 95 % CI | separated |
|---|---|---|---|---|
| ADE | `ade_m` | 0.0083 | [-0.0574, 0.0703] | no |
| ADE | `fde_m` | -0.0936 | [-0.2877, 0.0899] | no |
| longitudinal | `LON_speed_mae_mps` | 0.2706 | [0.2140, 0.3312] | **yes** |
| longitudinal | `LON_along_mae_m` | 0.1304 | [0.0789, 0.1802] | **yes** |
| longitudinal | `LON_accel_mae_mps2` | 0.2076 | [0.1503, 0.2674] | **yes** |
| lateral | `LAT_cross_mae_m` | -0.1569 | [-0.2131, -0.1072] | **yes** |
| lateral | `LAT_heading_mae_deg` | -1.6596 | [-2.2826, -1.0858] | **yes** |
| lateral | `LAT_yaw_rate_mae_radps` | -0.0429 | [-0.0569, -0.0301] | **yes** |
| tactical | `TAC_traj_lat_correct` | 0.0567 | [-0.0071, 0.1206] | no |
| tactical | `TAC_traj_lon_correct` | -0.1277 | [-0.1879, -0.0674] | **yes** |

**cl - ol** — estimator `paired_episode_cluster_bootstrap`, conditioning `T1 minus T0`; both arms OPEN LOOP

| family | metric | delta | 95 % CI | separated |
|---|---|---|---|---|
| ADE | `ade_m` | 0.1237 | [0.0687, 0.1770] | **yes** |
| ADE | `fde_m` | 0.2902 | [0.1402, 0.4343] | **yes** |
| longitudinal | `LON_speed_mae_mps` | 0.4421 | [0.3776, 0.5083] | **yes** |
| longitudinal | `LON_along_mae_m` | 0.2223 | [0.1678, 0.2754] | **yes** |
| longitudinal | `LON_accel_mae_mps2` | 0.4697 | [0.4051, 0.5340] | **yes** |
| lateral | `LAT_cross_mae_m` | -0.1094 | [-0.1375, -0.0846] | **yes** |
| lateral | `LAT_heading_mae_deg` | -1.2711 | [-1.5794, -0.9764] | **yes** |
| lateral | `LAT_yaw_rate_mae_radps` | -0.0351 | [-0.0458, -0.0266] | **yes** |
| tactical | `TAC_traj_lat_correct` | 0.0071 | [-0.0532, 0.0674] | no |
| tactical | `TAC_traj_lon_correct` | -0.2979 | [-0.3617, -0.2340] | **yes** |

## 4. What the planner did

| arm | n | plan sources | baseline_won_frac | goal source | selected LON goal | mean cost |
|---|---|---|---|---|---|---|
| `cl` | 282 | {'cem': 0.3369, 'baseline:hold_v0': 0.6383, 'baseline:decel_1.5': 0.0248} | 0.6631 | {'tactical_imagined': 1.0} | {'CRUISE': 0.4113, 'BRAKE_TO': 0.117, 'ADAPT_SPEED_FOR_CURVE': 0.2766, 'ACCELERATE': 0.195} | 1.2e-05 |


## 5. World-model diagnostic (teacher-forced)

* tier **T0** — teacher-forced — prediction quality only, NEVER driving performance
* n 282 · K 30 · space `standardised DINOv3 space (to_enc(op_pred) vs std(future); the trainer's loss_fe` · tgt_std_mean 0.9539

| quantity | mean | 95 % CI |
|---|---|---|
| `feat_mse_model_mean_over_steps` | 0.4387 | [0.4218, 0.4541] |
| `feat_mse_const_mean_over_steps` | 0.6173 | [0.5849, 0.6470] |
| `feat_mse_zero_mean_over_steps` | 1.0294 | [1.0217, 1.0373] |
| `feat_mse_model_step1` | 0.3052 | [0.2942, 0.3156] |
| `feat_mse_const_step1` | 0.2610 | [0.2447, 0.2764] |
| `feat_mse_model_step5` | 0.3813 | [0.3659, 0.3957] |
| `feat_mse_const_step5` | 0.4942 | [0.4637, 0.5228] |
| `feat_mse_model_step10` | 0.4253 | [0.4076, 0.4413] |
| `feat_mse_const_step10` | 0.5959 | [0.5612, 0.6283] |
| `feat_mse_model_step15` | 0.4494 | [0.4314, 0.4661] |
| `feat_mse_const_step15` | 0.6475 | [0.6134, 0.6787] |
| `feat_mse_model_step20` | 0.4690 | [0.4506, 0.4862] |
| `feat_mse_const_step20` | 0.6877 | [0.6511, 0.7216] |
| `feat_mse_model_step30` | 0.4952 | [0.4759, 0.5127] |
| `feat_mse_const_step30` | 0.7406 | [0.7031, 0.7737] |

`paired_const_minus_model` — per horizon (positive = the model beats the control)

| horizon | delta | 95 % CI | separated |
|---|---|---|---|
| `mean_over_steps` | 0.1786 | [0.1613, 0.1940] | **yes** |
| `step1` | -0.0442 | [-0.0553, -0.0339] | **yes** |
| `step5` | 0.1129 | [0.0960, 0.1286] | **yes** |
| `step10` | 0.1705 | [0.1518, 0.1873] | **yes** |
| `step15` | 0.1980 | [0.1803, 0.2144] | **yes** |
| `step20` | 0.2188 | [0.1993, 0.2363] | **yes** |
| `step30` | 0.2454 | [0.2259, 0.2625] | **yes** |

`paired_zero_minus_model_mean_over_steps`: **0.5907** [0.5743, 0.6088], separated yes
