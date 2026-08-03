# STREAM B — the STRATEGIC family, scored on a real sample of the T1 branch scenes

**Task #55.** *"One scene cannot carry a CI."* It now does not have to.

All numbers **MEASURED 2026-08-03 on tanitad-thor** unless labelled otherwise. Every number in
this document cites the JSON that produced it. Run directory: this directory.

---

## 0. Headline

| | before (task #51) | now |
|---|---|---|
| scenes carrying the family | **1** | **77** |
| bootstrap clusters | 1 → `CI_NOT_ADMISSIBLE` on every block | **77 → every interval admissible** |
| scoreable decision events | 2 | **116** |
| poses at which each arm was asked | 450 ticks, 1 scene | **4 745**, under **4 nav commands each** |
| download to get there | 1.8 GB (one usdz) | **451 MB of labels for all 141 T1 scenes + 3.24 GB of camera video** — against ~231 GB of usdz archives never fetched |

And the result that matters is not the accuracy. It is this:

> ⛔ **flagship-v1's strategic route head does not decide the route — it relabels the nav
> command it is handed.** Its argmax changes when *only* the nav input changes and the pixels do
> not, at **100 % of 4 745 poses**. The nav command is `refb_labels.nav_command_v21(gt_future)` —
> an **oracle**. And the decisive number: **`flagship-v1/navORACLE − NAV_ECHO = +0.0000
> [+0.0000, +0.0000]`** — under the oracle nav the flagship's route head is *indistinguishable*
> from a lookup table that never sees an image, on every one of 116 decisions. So the closed-loop
> `route_class_accuracy = 1.0000` previously published for flagship-v1 is the oracle passing
> through the network. **Retracted** (`RETRACTION_LOG.md`).
>
> **REF-C's route head is the opposite and is nav-BLIND by architecture** — its logits are
> bit-identical under all four nav commands — so its answer is genuinely a function of the image.
> At the deployable setting it beats flagship-v1 by **−0.5000 [−0.6053, −0.4017]**, CI-separated,
> and survives the leak-free control at **−0.5254 [−0.6897, −0.3823]**.

Both halves are confirmed **three times over** — source reading, an open-loop manipulation here,
and a closed-loop observation from a sibling stream (§5.3).

---

## 1. What was run, and why it is open-loop

`score_t1_strategic.py` places the ego on its **logged clipgt poses** and asks, at each pose on a
junction approach: *given the real camera observations up to now, which continuation do you
choose?* The route logits are then joined to the map-derived option sets by
`taniteval.strategic_optionset.event_predictions_from_ticks` and scored by `strategic_family`.

**Why not closed-loop.** The previous run drove the policies inside a rendered NuRec scene. That
needs `volume.nurec` (~600 MB of gaussians) per scene plus a render loop, and it answers a
different question, because the rollout drifts off the track the labels are indexed on. Open-loop
on the logged track buys:

* **an exact join** — the tick *is* the labelled pose, so no offset has to be fitted (the
  closed-loop run needed one, and scanned ±8 to find it);
* **real observations** — the recorded 4K `camera_front_wide_120fov.mp4`, not a reconstruction.
  REF-C's open-loop ADE is 1.5157 on these reconstructions vs 0.4728 on real footage
  (**3.21× OOD**, `alpasim-gsplat`), so real footage is *more* in-distribution;
* **cost** — see §2.

It is a **different experiment** from the closed-loop panel and is reported as such. It cannot see
closed-loop instability. It can see whether the strategic head chooses.

**Geometry asserted, never assumed.** Every scene's canonicalization is checked against the
training frame: `ftheta_crop_resize(..., 256, center="principal")` with the scene's own per-clip
f-theta intrinsics, and the achieved `f_eff` must be within 8 px of `F_REF = 266.0` or the scene is
**refused**. MEASURED range over the run: 265.8–266.2. Raster asserted at `(1, 8, 9, 256, 256)`
per pose.

**The pose→frame join is measured, not assumed.** Video is 30 fps, clipgt poses are 10 Hz;
matching by timestamp gives a **stride of exactly 3.0** with a median residual of **6.4 ms**
(`probe_video_pose_join.py`). Every scene's residual travels in the output.

⚠️ **Ops note, paid for once.** The first full pass was lost at ~22/78 scenes when Thor rebooted:
the runner serialised its ticks only at the END, and `/tmp` — where the log lived — is tmpfs and
did not survive. `score_t1_strategic.py` now writes a **per-scene checkpoint** and **skips scenes
already on disk**, so a restart resumes, and it logs to `/home/nvidia` (local nvme). The
programme's standing rule *"on pods LOG TO /tmp, never a network mount"* was written for the
RunPod boxes, where `/tmp` is local and `/workspace` is MooseFS. **On Thor the danger is
inverted** — `/tmp` is the volatile one — so the rule has to be read as *"log to local disk"*, not
as the literal path.

---

## 2. The download cost the brief warned about — and why it was avoidable

> *"each usdz is ~2 GB. Choose a defensible sample size, say why, and report what you actually ran on."*

A USDZ is a zip whose members are all **STORED, never deflated**, so an HTTP `Range` request pulls
one member out of the archive. The option-set label needs `map.xodr` + `pose_record.json` + two
clipgt parquets, and the camera video is a **separate ~40 MB file that is not inside the usdz at
all**. Only `volume.nurec` is genuinely large, and a route-head evaluation on the logged track does
not need it.

| stage | scenes | traffic | archives not downloaded |
|---|---|---|---|
| labels, pass 1 | 141 | **310.3 MB** | 160 264 MB |
| labels, pass 2 (retry of 42 transient pull failures) | 141 | **140.8 MB** | 71 359 MB |
| camera video for the scenes that carry a decision | 78 | **~3.4 GB** | — |

**So the sample size is not a budget compromise: labels were built for ALL 141 T1 scenes**
(`strategic_gt_T1_index.json`), and the ~40 MB video was then downloaded only for the scenes that
turned out to carry a real branch. That is the whole tier, not a sample of it — the 78 is what the
*maps* admit, not what the wallet allowed.

---

## 3. From 141 candidate scenes to 78 scoreable ones — the full attrition

`build_t1_labels.py` → `strategic_gt_T1_index.json`:

| | n | why |
|---|---|---|
| T1 scenes in the survey (\|turn\| ≥ 60°, complete traversal) | **141** | geometry candidates |
| labelled without error | **141** | 0 failures on the retry pass |
| **refused** by the map-vs-realised-heading self-consistency control | **31** | §3.1 |
| admissible | **110** | |
| admissible **and carrying ≥1 scoreable (≥2-option) decision** | **78** | the scored set |
| scoreable decision **events** | **117** | the values |

Events per scene: 1×50, 2×20, 3×6, 4×1, 5×1. Branching factor (max per scene): 2×47, 3×23, 4×5,
5×3. Label class balance over the 117 events: **LEFT 51, STRAIGHT 26, RIGHT 40, UTURN 0.**

⭐ **UTURN count is 0**, so the known deflation — a 3-way deployed head
(`refb.py:68`) cannot emit the map's 4th class — **does not apply to this set at all**.
`n_events_gt_outside_head_vocabulary = 0` on every arm. That was a real risk on the single winner
scene, where three of four options at junction 149 were UTURN.

### 3.1 The 31 refusals are reported, not quietly dropped

The self-consistency control compares the **map-derived** branch class against the ego's
**realised** heading change on that branch — two independent descriptions of one manoeuvre — and
refuses the scene above 35°. Over all 293 untruncated events the error is
**median 4.89°, p75 14.20°, p90 57.34°**, and **13.0 %** exceed the threshold.
Histogram over `[0,5,15,35,60,90,120,150,181]`: `[148, 75, 32, 13, 6, 4, 2, 13]` — i.e. a tight
main mode plus a distinct **~180° tail (13 events)**, the signature of a branch-direction
ambiguity rather than a graded error.

**24 of the 31 refused scenes fail on a SINGLE event**, and inside the refused scenes only
**38 of 78** untruncated events actually fail. So the refusal is scene-level and conservative: it
costs 38 otherwise-scoreable events. Alignment quality is *not* the cause — rms is 0.2176 m max in
the refused scenes vs 0.2447 m max in the admissible ones, i.e. indistinguishable.

⚠️ **This is stated as a limitation, not repaired.** Loosening the control to recover events would
be fitting the gate to the desired sample size, which is the failure this programme has a
retraction log for. The honest statement is: **the family is scored on 78 of 110 admissible
scenes, and a per-event rather than per-scene refusal would be the principled way to recover the
other 38 events — a work item, not a knob to turn now.**

---

## 4. Control 1 — the labels can discriminate (MEASURED, and this time with an interval)

`strategic_family_control.py --labels results/strategic_gt_t1` →
`strategic_family_control_T1.json`. Estimator: `episode_cluster_bootstrap`, **cluster = scene**,
n_boot 2000, **n = 117 events over 78 scenes**.

| arm | route_class_accuracy | 95 % CI |
|---|---|---|
| ORACLE | **1.0000** | [1.0000, 1.0000] |
| UNIFORM_RANDOM over option classes | 0.5470 | [0.4500, 0.6417] |
| **CONSTANT_LEFT** (the best constant) | **0.4359** | [0.3333, 0.5399] |
| CONSTANT_RIGHT | 0.3419 | [0.2459, 0.4445] |
| CONSTANT_STRAIGHT | 0.2222 | [0.1389, 0.3077] |
| NO_HEAD | 0.0000 | [0.0000, 0.0000] |

**ORACLE − BEST_CONSTANT = +0.5641 [0.4601, 0.6667], `separated = true`** (paired
episode-cluster bootstrap). ⇒ **`DISCRIMINATES: true`**, `constant_predictor_does_not_score_well:
true`, `no_head_scores_zero: true`.

Floors: uniform-over-options **0.4425**, uniform-over-option-*classes* **0.5242**, best constant
**0.4359**. The floor that decides is the **best constant fitted on these same events**, which
makes any win conservative.

⭐ **The power problem is solved.** A constant arm's CI is now **~0.103 wide** against **0.516** on
the 8-scene shortlist — better than the ±0.10 the earlier power analysis said a strategic verdict
needs.

---

## 5. Control 2 — the degeneracy the first control structurally cannot see

`discrimination_control` proves the **labels** carry entropy. It says nothing about an **arm**,
because its ORACLE is built by copying `route_gt_class` — a tautology. And `BEST_CONSTANT` cannot
catch the failure that actually happened, because **an echo is not constant and beats every
constant.**

The harness computes the nav command from the ego's own logged future
(`closedloop_drive.py:348 nav_from_route` → `refb_labels.nav_command_v21`) and **feeds it to the
policy**. Both deployed arms consume it. Scoring the route head against the branch the ego took,
while handing the model an oracle summary of that same branch, can be pure pass-through.

So each arm is run under a **full sweep of the nav vocabulary at a FIXED observation** — a
manipulation, not an observational contingency table. An observational nav-vs-head table cannot
identify an echo, because a competent head and an echo agree whenever the nav is correct.

⚠️ **A permutation control was tried first and was silently useless.** 50 of the 78 scoreable
scenes carry exactly ONE decision event, so a within-scene shuffle is the identity. It was replaced
by the sweep, and `navSHUFFLED` is now derived from the sweep by permuting **globally over all 117
events**. Fixed points and same-value landings are counted in `SHUFFLE_CONTROL`.

### 5.1 Why `navFOLLOW` is the headline and not an unfair OOD setting

The obvious objection is that `nav = follow` starves the flagship of an input it was trained with.
It does not. Quoting the corpus's own description at `stack/tanitad/refs/refc.py:66-68`:

> the 4-way `nav_cmd` is `follow` on **~75-79 % of windows** and is a **CONSTANT at eval**
> (`nav_cmd=None` → index 0)

So `follow` is both the **majority training condition** and the **standard evaluation condition**
for this programme. `navFOLLOW` is therefore the in-distribution, deployable setting, and
`navORACLE` — a per-pose command derived from the ego's own future — is the exotic one.

### 5.2 The mechanism, read from source, agreeing with the manipulation

| arm | what the route head reads | source | measured |
|---|---|---|---|
| **flagship-v1** | `route_head(norm(x[:,-1]))` where every causal block is **FiLM-conditioned on `nav_emb(nav_cmd)`** | `models/fourbrain.py:58, 77-86` | argmax moves with nav at **100 %** of poses |
| **REF-C** | `route_head(pooled)` — the **image** pool. The nav one-hot is built at `:1130` and enters only `self.measurement(...)` at `:1137`, **never the route head** | `refs/refc.py:1130, 1137, 1140` | logits **bit-identical** under all 4 navs |

⚠️ And the flagship's auxiliary route CE target (`route_target_v21`) is derived from **the same GT
future** as `nav_command_v21`. The shortcut — copy the FiLM condition — is available by
construction, and the measurement says it was taken. **That is a supervision-design defect, not an
instrument artifact.**

### 5.3 The echo is not an artifact of THIS harness — three independent lines agree

| line of evidence | harness | result |
|---|---|---|
| **manipulation** (this run) | open-loop, real 4K camera, logged track, 78 scenes | flagship argmax moves with nav at 100 % of poses; REF-C at 0 % |
| **observation** (`strategic_conditioning_control.py`, sibling stream, same day) | **closed-loop**, gsplat-rendered, 1 scene, 450 ticks × 2 conditions | flagship head is an **exact bijection** of nav (nav=1→LEFT 369/369, nav=0→STRAIGHT 81/81); REF-C is not a function of nav |
| **source** | none — code reading | `fourbrain.py:77-86` vs `refc.py:1130/1137/1140` |

Different harness, different renderer, different scene count, same verdict on both arms. ⚠️ Note
the two empirical lines are **not redundant**: the observational one saw only navs 0 and 1 (the
values that happened to occur) and so could not, on its own, rule out a competent head that merely
agreed with the command. The sweep can, because it holds the pixels fixed and moves only the input.

This control is now **structural, not a script**: `strategic_optionset.conditioning_echo_control`
plus `strategic_family(..., conditioning_sweeps=...)`, which emits
`STRATEGIC_SKILL_ADMISSIBLE`. With no sweep supplied it returns `None` (**UNTESTED**) rather than a
pass — an untested control is a gap, not a clearance. 7 new tests pin it, including
`test_an_echo_arm_beats_every_constant_yet_is_INADMISSIBLE`.

---

## 6. Leakage control

`leakage_check_t1.py` → `leakage_check_t1.json`. NuRec scenes carry their source
PhysicalAI-AV clip UUID, and both arms were trained on `physicalai-train-e438721ae894` drawn from
that corpus.

| | n |
|---|---|
| scored scenes | 78 |
| in the PhysicalAI-AV **train split** | **39** |
| in the val split | 22 |
| not in the catalogue | 0 |

⚠️ **39/78 is an UPPER BOUND on leakage and is nowhere near the real figure.** The train split
holds **153 625** valid clips; our corpus took **2 400** of them (**1.56 %**). The expected number
of scored scenes actually trained on is therefore **≈ 0.6**.

Two further readings, both stated with their evidence class:

* **MEASURED**: only **3 of the 78** scored scenes appear in the 9 000-clip `r0_selection_v2`
  candidate pool. ⚠️ but whether the `e438721ae894` build drew from *that* pool is **NOT verified**
  — the pool's clip-id digest does not match the manifest's
  `clip_id_sha256_sorted`, so this is suggestive, not decisive, and is not used to license
  anything.
* **DECISIVE and assumption-free**: the family is re-scored on the **39 scenes not in the train
  split at all** (`--leak-free`). If the verdict survives that, leakage is not driving it.

---

## 7. Results

`t1_strategic_families.json` — **116 scoreable events over 77 scenes, 4 745 observed poses.**
Estimator: episode-cluster bootstrap, cluster = scene, n_boot 2000. (One of the 78 scenes,
`bb4394e7`, is dropped: its junction sits too early in the clip for a 10-frame history. Reported,
not silently absorbed.)

Both arms' route-logit keys resolved as expected on **77 of 77** scenes — flagship `s_route_logits`,
REF-C `route_logits`. `f_eff` 265.82–266.18 against `F_REF` 266.0; pose→frame stride exactly 3.0 on
every scene.

### 7.1 The nav sweep — the whole result in one table

Argmax distribution over all **4 745** poses, per nav command, pixels unchanged:

| nav fed to the model | flagship-v1 answers | refc-base answers |
|---|---|---|
| `follow` | **STRAIGHT 4745 / 4745** | LEFT 1842 · STRAIGHT 1272 · RIGHT 1631 |
| `left` | **LEFT 4745 / 4745** | LEFT 1842 · STRAIGHT 1272 · RIGHT 1631 |
| `right` | **RIGHT 4745 / 4745** | LEFT 1842 · STRAIGHT 1272 · RIGHT 1631 |
| `straight` | LEFT 288 · STRAIGHT 44 · **RIGHT 4413** | LEFT 1842 · STRAIGHT 1272 · RIGHT 1631 |

`nav_passthrough_rate`: **flagship-v1 1.0000**, **refc-base 0.0000** (n = 4 745 both).
Logit-variance decomposition: flagship nav-to-image ratio **5.191** (it is *not* blind to pixels —
its logits move 1.86 across poses at fixed nav — the nav term simply outweighs them);
refc-base `HEAD_IS_NAV_BLIND = true`, across-nav std **exactly 0.0**.

### 7.2 The family

| arm / condition | route_class_accuracy | 95 % CI | − BEST_CONSTANT (LEFT @ 0.4397) | `STRATEGIC_SKILL_ADMISSIBLE` |
|---|---|---|---|---|
| flagship-v1 / **navFOLLOW** | **0.1983** | [0.1240, 0.2727] | **−0.2414 [−0.3950, −0.0948]** sep | **False** (echo) |
| flagship-v1 / navORACLE | 0.8707 | [0.8053, 0.9298] | +0.4310 [+0.3190, +0.5462] sep | **False** (echo) |
| flagship-v1 / navSHUFFLED | 0.3879 | [0.3103, 0.4722] | — | False |
| **refc-base / navFOLLOW** | **0.6983** | [0.6179, 0.7810] | **+0.2586 [+0.1238, +0.3966]** sep | **True** |
| refc-base / navORACLE · navSHUFFLED · navLEFT/RIGHT/STRAIGHT | 0.6983 | [0.6179, 0.7810] | identical | True |
| **NAV_ECHO** (no image at all) / navORACLE | **0.8707** | [0.8053, 0.9298] | +0.4310 | False |

⛔ **`flagship-v1/navORACLE − NAV_ECHO = +0.0000 [+0.0000, +0.0000], separated = false`** — on
**every one of the 116 events**. Under the oracle nav the flagship's strategic route head is not
merely *correlated with* a lookup table that never sees an image; it is **indistinguishable from
one**. Its 0.8707 is the oracle's number, and `NAV_ECHO` earns exactly the same 0.8707 with no
model at all.

Supporting paired contrasts (same events, same estimator):

| contrast | delta | 95 % CI | separated |
|---|---|---|---|
| flagship: navORACLE − navSHUFFLED | **+0.4828** | [+0.3879, +0.5715] | yes — the score rides on the oracle |
| flagship: navFOLLOW − navORACLE | **−0.6724** | [−0.7798, −0.5680] | yes — remove the oracle and it collapses |
| refc: navORACLE − navSHUFFLED | **+0.0000** | [0, 0] | no — nav-invariant, as designed |
| refc: navORACLE − NAV_ECHO | −0.1724 | [−0.2520, −0.0991] | yes (REF-C is *worse* than the oracle command — expected; the oracle is an oracle) |
| **flagship − refc @ navFOLLOW** | **−0.5000** | **[−0.6053, −0.4017]** | **yes** |
| flagship − refc @ navORACLE | +0.1724 | [+0.0991, +0.2520] | yes — but this is the oracle beating REF-C, not the flagship |

⇒ **At the deployable setting REF-C's strategic route head beats flagship-v1's by 0.50 accuracy,
CI-separated, on 116 map-derived decisions across 77 scenes.**

### 7.3 Precision beside recall, and both denominators

| arm / condition | LEFT (n_true 51) | STRAIGHT (26) | RIGHT (39) |
|---|---|---|---|
| flagship / navFOLLOW | rec 0.0000 · prec — (n_pred **0**) | rec 0.8846 · prec **0.2091** (n_pred **110**) | rec 0.0000 · prec — (n_pred **0**) |
| refc / navFOLLOW | rec 0.7059 · prec 0.8000 (n_pred 45) | rec 0.5769 · prec 0.6250 (n_pred 24) | rec 0.7692 · prec 0.7317 (n_pred 41) |

⚠️ flagship/navFOLLOW is the textbook recall-without-precision trap the programme has already been
burned by: **STRAIGHT recall 0.8846 looks fine and is worthless** — the arm predicts STRAIGHT on
110 of 116 events, so precision is 0.2091 and it emits **no LEFT and no RIGHT at all**.
`prediction_degenerate = true`.

`n_events_without_a_prediction = 6` for **both** arms — identical, so it is clip geometry (no tick
inside the admissible approach), not an arm gap.

### 7.4 A strategic error the confusion matrix cannot name

`n_predictions_outside_the_option_set` — a manoeuvre the **map does not admit** at that junction:

| arm / condition | off-map predictions (of 116) |
|---|---|
| flagship / navFOLLOW | **34 (29.3 %)** |
| refc / navFOLLOW | 13 (11.2 %) |
| flagship / navORACLE | 3 (2.6 %) |

"Wrong branch" and "no such branch" are different failures, and only an option-set label separates
them.

### 7.5 Accuracy by branching factor

| branching | n | flagship navFOLLOW | flagship navORACLE | refc navFOLLOW |
|---|---|---|---|---|
| 2 | 81 | 0.2469 | 0.8395 | 0.7037 |
| 3 | 26 | 0.1154 | 0.9615 | 0.6538 |
| 4 | 6 | 0.0000 | 0.8333 | 0.6667 |
| 5 | 3 | 0.0000 | 1.0000 | 1.0000 |

REF-C holds ~0.65–0.70 as the branch count rises; the flagship's neutral-nav score *falls to zero*
exactly where the decision gets harder.

### 7.6 `decision_lead_distance_m` — run against two real arms for the first time

Previously **HYPOTHESIS**-class ("specified and computable, but never run against an arm"). It has
now been run, and it is **MEASURED** — with its censoring stated.

| arm / condition | lead (m) | 95 % CI | censored | n |
|---|---|---|---|---|
| flagship / navFOLLOW | 6.75 | [3.85, 9.72] | 23 / 110 | 110 |
| flagship / navORACLE | 15.59 | [12.71, 18.69] | **86 / 110** | 110 |
| refc / navFOLLOW | 7.78 | [5.60, 10.12] | 26 / 110 | 110 |

Available approach: mean **19.80 m**, max **59.68 m**. ⚠️ **Two censors, not one.** The clip's own
approach length is the first (already in the metric); the second is mine — `--max-poses-per-event
60` caps the observed approach at 60 poses (~36 m at 6 m/s). Both arms are capped identically so
the *contrast* is fair, but every absolute value here is a **lower bound**. The flagship's oracle
15.59 m with 86/110 censored is mostly "correct for the whole observable approach" — which, given
§7.1, means *"it was handed the answer early"*, not *"it committed early"*.

### 7.7 Leakage control — the verdict survives

`t1_strategic_families_leakfree.json`, re-scored on the **39 scenes that are not train-split clips
at all** (59 events):

| | full (77 scenes) | leak-free (39 scenes) |
|---|---|---|
| flagship nav_passthrough_rate | 1.0000 | **1.0000** |
| refc nav_passthrough_rate | 0.0000 | **0.0000** |
| flagship / navFOLLOW | 0.1983 | **0.1525** |
| flagship / navORACLE | 0.8707 | **0.9153** |
| flagship navORACLE − NAV_ECHO | +0.0000 [0,0] | **+0.0000 [0,0]** |
| refc / navFOLLOW | 0.6983 | **0.6780** |
| **flagship − refc @ navFOLLOW** | −0.5000 [−0.6053, −0.4017] sep | **−0.5254 [−0.6897, −0.3823] sep** |
| refc − BEST_CONSTANT | +0.2586 [+0.1238, +0.3966] **sep** | +0.2203 [**+0.0000**, +0.4154] **NOT sep** |

⚠️ **One thing does NOT survive, and it is reported rather than buried:** on the leak-free subset
REF-C's margin over the best constant has the **same point estimate but a lower bound of exactly
0.0000**, so it is **no longer CI-separated**. That is a **power loss** (39 clusters vs 77), not a
reversal — but the honest statement is:

> **REF-C beats the best constant predictor on the full 77-scene set (separated). On the 39-scene
> leak-free subset the same margin is NOT separated. The head-to-head against flagship-v1 is
> separated on both.**

### 7.8 Control bookkeeping

* `SHUFFLE_CONTROL`: 110 events permuted globally, **0 fixed points**; 46 landed on the same nav
  *value* (a 4-value vocabulary makes that unavoidable) — which makes `navSHUFFLED` a
  **conservative** control, since it retains some genuine signal.
* `LEAKAGE_CONTROL`: 38 of 77 scored scenes are train-split clips; base rate of selection into our
  2 400-clip build = **0.01562**.
* `NEGATIVE_CONTROL` (§4) re-run inside this artifact: `DISCRIMINATES = true`.

### 7.9 What this does and does not license

| claim | verdict |
|---|---|
| flagship-v1's strategic route head **echoes its conditioning input** | **MEASURED**, 3 independent lines (§5.3) |
| flagship-v1's previously published closed-loop `route_class_accuracy = 1.0000` is that echo | **MEASURED** ⇒ retracted, `RETRACTION_LOG.md` |
| REF-C's route head **reads the scene** and beats a constant on 77 scenes | **MEASURED**, separated; ⚠️ **not** separated on the 39-scene leak-free subset |
| REF-C's route head beats flagship-v1's at the deployable setting | **MEASURED**, separated on both sets |
| this says anything about **closed-loop** behaviour | ⛔ **NO.** Open-loop on the logged track cannot see closed-loop instability. |
| this says anything about ADE or the other three families | ⛔ **NO.** See §8. |
| `decision_lead_distance_m` discriminates hierarchical from flat policies | ⚠️ still **HYPOTHESIS** — it has now been *run* on two arms, but the flagship's lead is inflated by its oracle input, so the metric has not yet been given a fair test of that specific claim. |

---

## 8. Four-families note

⛔ Per the binding rule (Sayed, 2026-08-02), an eval must report **longitudinal / lateral /
tactical / strategic**, and a missing family is a work item. This run is the **STRATEGIC family
instrument itself** and deliberately reports no ADE — the failure mode the rule exists to prevent
is ADE crowding out the other three, and this is its mirror image.

Stated per family, with the reason and the n, as the rule requires:

| family | status here | reason |
|---|---|---|
| **STRATEGIC** | **REPORTED**, n = 117 events / 78 scenes | this document |
| LONGITUDINAL | **not computed**, n = 0 | the arms were queried for route logits only; no waypoints were decoded, so no speed/headway/TTC exists in this artifact. Recoverable by re-running `score_t1_strategic.py` with the tactical head enabled — a work item, not a claim. |
| LATERAL | **not computed**, n = 0 | same reason. |
| TACTICAL | **not computed**, n = 0 | same reason. |

⚠️ These three are also **not defined on this corpus** the way the programme quotes them: the
published longitudinal/lateral/tactical numbers live on the 40-episode / 881-window
`physicalai-val-0c5f7dac3b11` benchmark. Computing them on 20 s NuRec clips would produce numbers
that are not comparable to any registry entry. **The correct fix is to run all four families on the
canonical val set and the strategic family here, then report them side by side with their
different denominators stated — not to manufacture three families on the wrong corpus.**
