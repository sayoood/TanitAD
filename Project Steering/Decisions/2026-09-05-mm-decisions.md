# Master Mind decisions, 2026-09-05

## M1. The pod's stale `train_v6_staged.py` ships at the NEXT v7 launch preflight — never mid-run

From the pod-currency audit (`…/2026-09-04-pod-currency-audit/POD_CURRENCY_AUDIT.md` §2 rank 2,
commit `94ceaf4`): `tanitad-refcv3:/workspace/TanitAD/stack/scripts/train_v6_staged.py` is
**113 KB (~19 %) behind HEAD** and lacks the `--bptt-truncate` wiring. A v7 launch from that box
would run the O5 rollout with the unbounded BPTT chain — the MEASURED `PREREG_MM_E19` failure
(gnorm 2.1e9, run killed at step 9,000) — and would fail *silently*, because the box is
stale-but-internally-consistent. The escalation was correct: this file must not be shipped while
`refcv4b` trains (a supervisor relaunch imports whatever is on disk), and it must not be shipped
in isolation (its compatibility with a future v7 config cannot be verified from here).

⇒ **Decision:** the v7 launch runbook gains a mandatory step: run
`stack/scripts/pod_currency_audit.py` against the target pod, ship `train_v6_staged.py` and
`tanitad/models/metric_dynamics.py` as part of that preflight, verify by md5 at both ends and by a
real `import` on the pod, and grep-verify `--bptt-truncate` is present before any launch. Owner:
whoever launches the next v7 arm. Not before refcv4b finishes.

## M2. `metric_dynamics.py` on the pod stays as it is until M1

Imported but never called by the live trainer; not a pure addition (38 pod-only lines);
default-equivalence only INHERITED from a docstring and a test not run against this pod. Zero
benefit to the live run, non-zero relaunch risk. Ships with M1.

## M3. The ego-dropout burden is NOT a live-run change

`D-REFCV4B-EGODROP2` (commit `ef0d5ec`): real, longitudinal, shrinking on its own. The fix
(`H-EGODROP-PRED`, roll the withheld bank at the model's own predicted speed) is pre-registered
for refcv5 on the v7-tiny ladder with a shuffle-the-prediction control. The live run is read
again on its final checkpoint.

## M4. Never run two `kb_add` writers concurrently

`library.json` was found at 0 bytes after the API-limit deaths — a rewrite interrupted mid-flight.
Restored from HEAD and re-verified (`7d1f48f`). Every literature brief now carries the rule, and a
0-byte `library.json` is to be read as the signature of an interrupted writer, never as an empty
library.

## M5. The three unapplied cost-repair rows are SUPERSEDED, not applied

The units stream (commits `8cb69ac`…`1291bf5`) correctly refused to apply
`…/2026-09-03-cost-repair/PROPOSED_REGISTER_ROWS.md` §2–4 (`H-COST-WEIGHTS-1`,
`D-COST-ARGMIN-MOVES`, `D-COST-SURFACE-REPRODUCED`) and its §5 prediction-retraction: the
register already frames `D-COST-CHORD` as FAILING its own pre-registered criterion (BACKLOG R41,
R55/R56), and those three rows are intermediate findings of the chord experiment. The live
hypothesis is no longer the chord (monotone-equivalent, cannot re-rank) but the centred cosine
(`ccos`, commit `cee5d99`), whose panel — exclusion fraction, L/R share, distinct plans, κ ≡ 0,
weight-neutrality factor, under `cos` / `chord` / `ccos` with constant-only and
deliberate-regression controls — is running now and measures exactly what those rows claimed.

⇒ **Decision:** none of the three is registered as written. `H-COST-WEIGHTS-1` is replaced by the
`ccos` weight-compensated arm; `D-COST-ARGMIN-MOVES` and `D-COST-SURFACE-REPRODUCED` are re-read
against the `ccos` panel when it lands and registered then, in whichever form survives it. The
§5 retraction stays in the package file as the author's own record; nothing is lost.

## M6. `H-EGO-LIT-4` gets an owner: the 5-arm withheld-bank panel runs on the v7-tiny rig now

The literature stream escalated the discriminating experiment for `H-EGODROP-PRED` with no
owner. It is the cheap pre-retrain validation the PI asked for, it needs no pod GPU, and both
outcomes are already committed. Launched 2026-09-05 as its own stream: implement the stamped
`--withheld-bank {fixed,pred,random,none}` trainer flag, run A0 fixed / A1 predicted / A2
random-marginal control / A3 dropout 0.25 / A4 speed-blind vocabulary, score on the `H-ECHO-8`
separation instrument and four families on kept AND withheld rows — never on ADE.

## M7. ⭐ PI RULING — the environment extension ships in TWO releases: v5a pure vision, then v5b LiDAR

**PI, verbatim (2026-09-05):** *"i prefer to do the environment extensions in two versions/steps, let
start by pure vision and then add lidar. So check, what we can do maximally with vision, bev, und was
else?"*

⇒ `REFCV5_DESIGN_PLAN.md` §3/§7 were written on a single-track assumption and must be restructured
into **v5a (camera-only, maximised)** and **v5b (LiDAR BEV)**. The §7 ladder its author is writing now
is still valid as a set of work packages; only the release boundary changes. A dedicated study is
running: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vision-only-maximum/`.

**The inventory that makes v5a large, MEASURED** (`…/Data Engineering/Implementation/incoming/
2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md`, the dataset's own 36-row
`features.csv`: 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar):

* **We read ONE of SEVEN cameras.** `camera_front_wide_120fov` only (`physicalai.py:232`, pinned at
  2 / 5 / 6 features by layer in `test_physicalai_feature_readset.py`). Unread: `front_tele_30fov`,
  `cross_left_120fov`, `cross_right_120fov`, `rear_left_70fov`, `rear_right_70fov`,
  `rear_tele_30fov` — each 1.6–2.5 GB/chunk, 5.1–7.9 TB total. ⭐ **This is the single biggest
  vision-only lever and it needs no new sensor modality.**
* **`camera_intrinsics` already covers all 7 cameras at 100 %** and we already read it; the rig
  transform for every sensor is in `sensor_extrinsics`, also already read. Adding a camera is a
  DECLARATION change in the episode build, not a new calibration problem.
* **`camera_intrinsics.offline` carries `ego_mask_image_png`** (97.44 %, 0.2 MB/chunk, unread) — a
  free ego-occlusion mask.
* ⚠️ **A third option the two-release framing does not cover: RADAR.** 19 radar features in three
  mutually-exclusive rig configurations on **160,761 clips (52.49 %)**; the probe's own words:
  *"the whole radar suite is cheaper than one camera."* Radar gives measured RANGE-RATE directly —
  the LONGITUDINAL family is where 88.7 % of our oracle gap lives. It is neither vision nor LiDAR,
  and it is the cheapest sensor in the dataset. **Flagged to the PI as a possible v5a-plus or v5b
  alternative; no decision taken.**
* LiDAR for comparison: **32,340 MB/chunk, 101.7 TB**, 97.44 % coverage (this census; §3 of the plan
  read 99.6 TB from the HF blob listing — same order, two probes).

**What v5a can recover of DiffusionDrive's three grounded attentions** (to be settled by the study,
stated here as the question): waypoint-indexed sampling in PERSPECTIVE view needs no BEV at all
(`H-DDA-1`, already pre-registered); agent cross-attention is reachable from cameras because
`obstacle.offline` supplies 3D cuboids as **LABELS** on 97.44 % of the corpus and the binding rule is
*labels may use privileged signals, inference is vision-only* — so a monocular 3D detector trained on
them yields agent tokens at inference without LiDAR; a metric BEV is the one that genuinely wants
either surround cameras (a lift) or v5b.

⇒ **Decision:** v5a is maximised and shipped first. v5b is judged on what gap remains AFTER v5a, not
on its own merits.

## M8. `E-DDA-1` runs the PLAN's four-arm form with its three controls — not the study's two-arm form

The reconciliation surfaced a genuine conflict of FACT (not naming) and correctly refused to
adjudicate it: `E-DDA-1` (waypoint-indexed sampling in perspective view) is **4 arms ≈ 2 h with
three named controls** in `REFCV5_DESIGN_PLAN.md` §7, and **2 arms ≈ 1.0 h with none listed** in
`…/2026-09-05-vision-only-maximum/RESULT.md`.

⇒ **Decision: the four-arm form.** The hour saved buys nothing; the control it drops is the
**projection control** — the arm that proves the instrument can SEE a wrong projection. Without
it a null result is uninterpretable (the sampler may be reading the wrong pixels and we could not
tell), and a positive result is unfalsifiable. This programme has paid for that lesson twice this
week: the anti-echo gate is only meaningful because a deliberately image-blind arm FAILS it
(`H-ECHO-4`), and the refcv3 route metric scored 1.0000 while measuring nothing because no
intervention control existed. ⛔ A control that must read a known value is not an optional cost line.

⚠️ Corollary recorded so the cheaper number does not leak into a plan: the reconciled v5a total is
**≈ 29–38 rig-GPU-h (the union of both ladders)**, NOT the study's ≈ 13.4 h. Quote the union.

## M9. Rung A1 is launched now, ahead of refcv4b's finish

`ha0_ext` is absent from the REF-C harness (0 occurrences in `refcv3_arm.py` against 12 in
`refav1_arm.py`, same-breath control 28 `add_argument` calls — the file reads fine), so half the
refcv5 acceptance bar is unreadable; and **8 of the 12** registered hierarchy ablations have no CLI
flag. The eighth missing one is the **frame-blind deliberate regression** (`--ablate-frames`), which
the prereg's own escalation omits and on which the panel's validity depends — a gate never shown to
FAIL an image-blind arm certifies nothing.

Both are zero-GPU harness work and both block the post-training hierarchy panel, which becomes
runnable when refcv4b finishes (~2026-09-06 08:00 UTC). Launched 2026-09-05 as its own stream rather
than scheduled, because the window is ~20 h and the work is free.

## M10. Open, carried, NOT decided

* `D-V5A-CAM3` is a **retraction-log candidate** — a banked DataFlyWheel camera figure is
  ×1.39–1.58 high. Nobody has written it to `RETRACTION_LOG.md`; it needs the same treatment as
  #23.
* The **B1 chunk spread is unbanked** and BOTH ladders rest on it — a one-query readout for the
  DataFlyWheel. If the build pulls whole chunks rather than per-clip ranges, the 429 GB camera
  figure silently becomes terabytes.
* The study's manifest claims **11** `D-V5A-*` rows; the register holds **10**. Left as found —
  ⛔ adding a row to make a manifest agree is how a register stops being evidence.

## M11. ⛔ TWO `ha0_ext` KINEMATICS EXIST AND DISAGREE BY MORE THAN THE MARGIN — the INTEGRATOR is canonical

MEASURED by the Rung A1 stream and re-verified at source: the programme carries two
implementations of the same control — `stack/tanitad/eval/echo_gate.py:171 ha0_ext(v0, a0, k0,
taus_s)` (closed form) and `taniteval/tools/refav1_arm.py:407 hold_ext_controls(...)` (rolled
through the integrator). They differ by **0.540642 m at 2 s** and **1.862923 m at 15 s**.

⚠️ **That is LARGER THAN THE ENTIRE QUANTITY WE ARE MEASURING.** refcv3's gap to `ha` is
**0.1423 m**. Which implementation the harness calls therefore decides the verdict, not merely
its precision.

⇒ **Decision: `refav1_arm.hold_ext_controls` — the INTEGRATOR — is canonical for every ARM.**
Two reasons, the second binding:
1. Every banked refav1 number is already published under it; switching would silently move a
   published result.
2. ⭐ **It is the same kinematic the candidate fan itself is built with.** The anchors roll
   through `rollout_unicycle` (`refc.py::roll_bank`); a closed-form control would compare the
   model's integrated candidates against a differently-derived baseline, and the difference
   between two kinematics would be scored as model skill.

The closed form stays for `echo_gate`'s internal use. ⛔ They may never be mixed inside one
comparison, and no arm may quote `ha0_ext` without naming which derivation produced it.

⚠️ **`D-REFCV5-LADDER-2` NEEDS AMENDING**: it instructs the port to "call the same
`echo_gate.ha0_ext`". Followed literally that would have made the REF-C arm **incomparable with
refav1's** — the exact failure it was written to prevent. The Rung A1 stream followed the
brief's wording (the integrator), pinned the divergence with a test, and escalated rather than
choosing silently. That was right.

## M12. The prereg's `H19-OFF` ablation removes nothing — ERRATUM 1 issued

`PREREG_REFCV4B_HIERARCHY_EVAL.md` §3 registers `H19-OFF` as
`decoder.maneuver_to_anchor = None`. MEASURED at source: `refc.py:1237` initialises that
attribute to `None`, `:1247` assigns a layer **only in the non-factored branch**, and every v3/v4
build forces `factored_maneuver = True`. On refcv4b the attribute **is already `None`**; the live
prior is `lat_to_anchor` + `lon_to_anchor`.

⇒ The registered arm would have run, changed nothing, and been read as *"the H19 seam is
inert"* — which §5 lists as a condition **refuting the hierarchy thesis**. An ablation that
cannot fail, inside a panel whose whole purpose is to be able to fail. Fourth instance of that
class this week.

⇒ **`Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.ERRATUM-1.md`** issued as a SEPARATELY
STAGED document — not edited into the prereg, whose falsifiable object is its staged blob id.
The mechanism is corrected to remove what the build actually carries; `H-H19-1` and its committed
outcomes are unchanged. No result is retracted — the panel has not run.

## M13. Adopt `mktree_commit.py` as the default committer while the mount behaves this way

`mm_commit.py` failed **25 consecutive attempts over ~40 min**, every one dead in `read-tree` (a
full-tree walk across the mount), while `rev-parse` / `status` / `cat-file` returned 0 in the same
seconds and six control reads of `CLAUDE.md` succeeded — so **not a wedge**, and no Drive
restart was warranted. `stack/scripts/mktree_commit.py` rebuilds only the named paths' ancestor
trees (`hash-object` + `ls-tree -z` + `mktree`), keeps the compare-and-swap, and asserts each path
positively before and after. Four commits landed with it first time.

⇒ Use it where `mm_commit.py` stalls in `read-tree`. ⛔ Both still verify by content
afterwards; neither exit code is evidence.

## M14. Carried, not decided

* **E3** — the Lab tree has both `Benchmarks & Eval` (singular, where earlier packages sit) and
  `Benchmarks & Evals` (plural, the PI-binding form). Rung A1 banked at the plural path and
  flagged the conflict rather than guessing. A sweep is owed; nothing is lost meanwhile.
* **E4** — the hierarchy panel must roll **FULL first**, because `gstr_shuffle` needs the bank.
* `test_refav1_kin_contract::test_A6_adding_ha0_moves_no_existing_arm` fails **without** Rung A1
  too (demonstrated on restored pre-patch files) — a work item for the refav1 stream, not a
  regression from this rung.

## M15. ⭐ APPROVED — the lateral goal vocabulary gets THREE sustained curvature magnitudes, and the decision metric is medAE-on-turns, NOT RMSE

**The escalation, and it is well made.** The refav1 goal-margin stream priced the fix before
spending any GPU. Scoring gives each design an ORACLE token chooser, so the residual is the best
*available* sustained curvature against the road — an upper bound on what any head could achieve
with that vocabulary:

| design | RMSE | medAE-on-turns | expressible |
|---|---|---|---|
| L=0, `LANE_KEEP` only | 0.01785 | 0.01942 | 0.0000 |
| L=1, κ=0.02 | 0.01460 | 0.00609 | 1.0000 |
| **L=1, κ=0.08 — SHIPPED** | **0.01015** | 0.01551 | 0.3866 |
| **L=3 at the measured quantiles** | 0.01032 | **0.00421** | **1.0000** |
| CONTINUOUS ceiling | 0 | 0 | 1.0000 |

⇒ **L=3 is APPROVED.** 100 % of real turns become expressible instead of **38.7 %**, and the median
curvature error on a turn falls **3.7×**, at **the same RMSE** (0.01032 vs 0.01015, +1.7 %).

⭐⭐ **THE RULING THAT MATTERS MORE THAN THE APPROVAL: `RMSE` AND `medAE` RANK THESE DESIGNS
DIFFERENTLY, AND RMSE PICKS THE VOCABULARY WE ALREADY KNOW IS BROKEN.** κ=0.08 has the **best**
single-magnitude RMSE and nearly the **worst** medAE on turns — because RMSE is dominated by the
**rare sharp** turn, which 0.08 serves, while the **median** turn is gentle and 0.08 serves it
barely better than doing nothing. A design chosen on RMSE reproduces the shipped vocabulary that
cannot express 61 % of turns.

⇒ **For vocabulary design the decision metric is `medAE-on-turns`.** RMSE may be reported; it may
not decide. ⚠️ **Same family as the two estimator rules in `CLAUDE.md`** — *never quote an interval
without its estimator*, and *`overlapping_holdout_se` biases the point estimate* — with the object
swapped again: here the statistic is computed correctly and **answers a narrower question than the
claim being hung on it**. A summary statistic dominated by the tail cannot adjudicate a median
failure mode.

**Controls, checked before approving:** the CONTINUOUS ceiling reads **exactly 0.0**, the
`LANE_KEEP`-only floor reads **exactly the road RMS curvature to 1e-12**, and the shipped row is
present by assertion. A design table without a known-value row at each end is not admissible, and
this one has both.

## M16. ⛔ THE DECISION-RULE PROGRAMME FOR refav1 IS CLOSED — MEASURED WORSE, TWICE

The A/B landed (**T1**, 14 episodes / 28 windows, `ccos` + Stage-B weights, step 21,109):
`ADE` **1.6098 → 2.5025** (+0.8927), `FDE` +1.8712, cross-track +0.8232, heading
**+9.07°**. ⭐ **Every family degrades EXCEPT longitudinal speed, which is unchanged (−0.005)** —
exactly what a *lateral-only* bias should leave alone, so **the one channel the lever does not
touch is the one channel that does not move.** That is a located effect, not a drift.

The lever did precisely what it was designed to do: constant-velocity share **0.214 → 0.000**,
straight-plan share **0.250 → 0.000**. **It turned more, and turning more made it worse** — which
is what M15's table predicts, because the turns it forces cannot be rendered by a vocabulary with
one sustained curvature at R 12.5 m.

⭐ **And the dense panel removes the motivation entirely.** On the stride-2 grid (**4,786 windows,
17× the original**, forward pass verified: banked `lat_logits` reproduced at max abs diff **0.0**,
argmax 282/282), at |gt_κ| > 4e-2 — **R 25 m, the MEASURED vocabulary crossover, not a threshold
picked for a good number** — the **shipped** head already decodes a curvature-carrying token on
**74.4 %** of windows at a **12.0 %** false-turn rate, **AUC 0.8806**, with shuffled-label AUC flat
at **0.498–0.504**. The measured crossover **0.04101** agrees with the analytic
`GOAL_KAPPA_TURN/2 = 0.040`, and **the agreement TIGHTENS with more data — which is what a real
quantity does and an artefact does not.**

⇒ **GATE 1 IS LARGELY OPEN WHERE IT MATTERS.** The "20.5 % turn recall" that launched the
decision-rule programme is a statistic about **R 1000 m** curves the vocabulary cannot express
anyway. ⛔ `lat_logit_bias` stays in the tree as a **parameterised instrument** (zero-bias is
bit-identical to no-flag, pinned) — it is **not** a fix and must not be shipped as one.

⚠️ **Two things this does NOT settle, and they must travel with any refav1 claim:**
1. **Parity.** Changing `canonical_controls` changes the ACTION SPACE, so **every banked refav1
   number is under the OLD vocabulary.** New arms carry the vocabulary version explicitly, and a
   cross-vocabulary comparison is inadmissible without saying so.
2. ⛔ **Both A/B arms sit 3.6–5.9× BELOW the trivial floors on turning windows.** The vocabulary is
   a **necessary** fix; it is not shown to be sufficient, and the floor gap is a separate open item
   that no vocabulary change is entitled to claim.

## M17. ⭐ RULING — `--agent-queries` becomes **100**, and the criterion is the DROP POLICY, not the distribution

**The escalation.** `--agent-queries 32` was set from val40. MEASURED on the train corpus:

| corpus | mean | p99 | **max** |
|---|---|---|---|
| val40 (the basis for 32) | 3.16 | 19 | **24** |
| **train2400** | **4.39** | **30** | **94** |

At N = 32 on train: **41,362 boxes (2.18 %) dropped across 3,250 frames, nearest sacrificed target
at 13.1 m.** Zero-drop floor **94**; at N = 64 the nearest sacrificed target is 33.9 m.

⇒ **DECISION: `--agent-queries = 100`.** Four reasons, and the second is the binding one.

1. **The stream's own pre-committed rule was "a covering N with zero drop", explicitly chosen over
   "use the p99".** Picking 64 now — after seeing that 94 is expensive — would be **selecting the
   threshold on the data it is scored against**, which is the exact failure this programme keeps
   paying for. A rule committed in advance is not renegotiated because its answer is inconvenient.
2. ⛔ **`match_slots` keeps the NEAREST N, so a DROP IS BY CONSTRUCTION THE CLOSEST THING WE FAILED
   TO SEE.** That inverts the usual reading of a truncation: this is not a long tail of distant
   clutter being trimmed, it is the near field going unmodelled once a scene is dense enough.
   **13.1 m is inside the braking envelope at any urban speed.** And **N = 64 does not fix it
   either** — 33.9 m is still inside a comfortable-braking envelope at 15 m/s (≈ 1 s reaction plus
   v²/2a at −4 m/s² ≈ 43 m). A safety-relevant truncation is not a hyper-parameter to be traded
   against decoder cost.
3. ⚠️ **94 is a MAX OVER A SAMPLE, NOT A BOUND.** 2,308 episodes are not every scene we will ever
   meet, and setting N to the observed maximum guarantees truncation on the first denser frame.
   **100** carries headroom over the sample and is a round, stable number that will not drift each
   time the corpus grows.
4. ⭐ **100 is the ORDINARY setting for this architecture, and 32 was the anomaly.** DETR's own
   convention is ~100 queries against ~7 objects per image; ours is a mean of **4.39**. A decoder
   whose surplus queries emit "no object" is the design working as intended, not waste.

⚠️ **What must accompany it:** the decoder cost at N = 100 is **MEASURED and reported**, never
assumed — step time and peak memory against N = 32, on the box the arm actually runs on. If that
cost turns out to bind, the answer is a **different architecture for the near field**, not a
silently smaller N. ⛔ And the flag's help text and docstring still assert the val40 claim; they are
corrected in the same change, because the number was wrong in the place a reader would check it.

## M18. ⛔ A HASH WITHOUT ITS ARTIFACT SCOPE IS NOT A VERIFICATION — a correct-looking checker refused a good file

MEASURED by the DataFlyWheel stream: the two agent joins' sidecars record `md5` **over different
artifacts** — val40's covers the **decompressed `.jsonl`**, train2400's covers the **compressed
`.xz`**. The refcv5 density script's C5 integrity check, inherited from val40, therefore **REFUSES
the train join**: a checker that is correct in every line, applied to a file it was never scoped
for, rejecting a file that is perfectly good.

⇒ **RULE: a recorded hash carries the ARTIFACT it was taken over — compressed or decompressed, and
the exact filename — or it is not a verification, it is a number.** A consumer that cannot see that
scope will either refuse a good file (here) or, worse, **accept the wrong one and report success**.

⭐ **This is the anchor-units trap in a new costume**, and the third member of the family in two
days: `control_units` on a `.pt` (`M11`'s neighbour, the 396 g / 0.31 g inversion), `corpus.labels`
being a dict where a path was expected (`D-NAVCOMP-SHAPE-1`), and now an md5 whose subject is
unstated. Each time the VALUE was right and its **scope** was missing, and each time the failure
wore the costume of a working check. ⇒ The durable form is the one already adopted for anchors:
**builders write the scope INTO the sidecar**, and a consumer that finds no declared scope
**refuses rather than guessing**.

⚠️ Two more from the same stream, both the same class and both open:
* ⛔ **`--agent-w-project` / `--agent-w-ground` are SILENT NO-OPS** — `model._rig_camera` is set to
  `None` and never assigned, so both loss terms are guarded out **while being stamped into
  `config.json`**. The run record therefore states a training configuration that did not happen.
  A flag that parses and does nothing is worse than a missing one; it must **REFUSE**.
* **`w_agent` / `w_u0` are absent from `config.json` entirely** — a run cannot say what weight its
  detector trained at, which makes any later comparison between arms unfalsifiable.

## M19. ⛔ AMENDS M15 — I APPROVED ON A CEILING AND QUOTED IT AS A PAYOFF. The binding term is HEAD RECALL, not magnitude.

**My error, stated first.** M15's operative sentence reads *"⇒ **L=3 is APPROVED.** 100 % of real
turns become expressible instead of 38.7 %, and the median curvature error on a turn falls
**3.7×**."* The table above it says, correctly, that each row is scored with an **ORACLE token
chooser**. ⛔ **I labelled it a ceiling and then used it as the expected gain in the sentence that
decided.** That is the programme's own *model-free vs model-inclusive* rule — **never compare a
ceiling to an achievement** — broken inside a decision.

**What the realised number actually is.** MEASURED by the refav1 stream (`realised_kappa.py`,
**4,520 windows / 141 episodes, 644 real turns**), composing the vocabulary with the **SHIPPED
head's actual argmax decode** and the real `canonical_controls` profile:

| arm | medAE-on-turns | turns goaled correctly |
|---|---|---|
| ZERO floor (the goal never turns) | 0.01942 | — |
| realised κ_turn = 0.02 | 0.01831 | **0.2811** |
| realised κ_turn = 0.04 (best) | **0.01734** | **0.2811** |
| realised κ_turn = 0.08 — SHIPPED | 0.01775 | **0.2811** |

⇒ Tuning the magnitude **alone** buys **2.3 %** (0.01775 → 0.01734), not 3.7×. Against the
never-turn floor the **entire** shipped lateral goal is worth **8.6 %**, and the best reachable
constant **10.7 %**.

⭐⭐ **THE WHOLE ANSWER IS IN ONE COLUMN, AND IT IS CONSTANT.** *"Turns goaled correctly"* reads
**0.2811 for every κ_turn**, because **κ_turn does not touch the head logits**: the decode is
unchanged, and only **28.1 %** of real turns receive a correctly-signed sustained goal *whatever
magnitude is commanded*. ⇒ **THE BINDING TERM IS THE HEAD'S RECALL AT THE CROSSOVER, NOT THE
MAGNITUDE.** A vocabulary that can express a turn does nothing for a turn the head does not call.

### What this changes in M15

1. ⛔ **The "3.7×" must not be quoted as a payoff anywhere.** It is an oracle ceiling. The realised
   figure for magnitude tuning is **2.3 %**, and it is the one that belongs in a plan.
2. ⚠️ **The APPROVAL OF L=3 IS NOW CONDITIONAL, AND THE CONDITION IS NOT COSMETIC.** The oracle
   table assumes a chooser that picks correctly among **three** tokens. Today's head cannot choose
   among three — it is not trained to, and the realised table shows its *binary* decode already
   fails on 71.9 % of real turns. **Adding tokens a head cannot select between is strictly worse
   than one token it can**, because it splits the same probability mass across more classes.
   ⇒ **L=3 ships ONLY together with a head able to select among the new tokens.** A vocabulary-only
   arm is now a **deliberate-regression control**, not the product.
3. ⚠️ **The two levers fight, and that must be designed for rather than discovered.** Lowering
   κ_turn lowers the crossover, making more road expressible **while asking the head to make a
   harder decision on gentler curves**. Expressibility and decodability move in opposite
   directions; the design point is a trade, not a maximum.

### What this does NOT change

* **M15's ruling on the METRIC stands, and is reinforced.** `medAE-on-turns` still decides and RMSE
  still may not — and note that the realised table is *also* read on medAE, which is what let the
  2.3 % be seen at all.
* **M16 stands.** The decision-rule programme is still closed: the lever made driving worse at
  11.2× the seed floor, independently measured.
* **`PREREG_D-VOCAB-L3_FLOOR_GAP` stands, and its REFUTED branch is now MORE likely** — which is
  the point of having written it before the fix. ⚠️ Its arms must be re-read in light of item 2: a
  vocabulary-only L=3 arm is now expected to be flat or worse, and if it is, **that is the
  pre-registered REFUTED outcome arriving on schedule, not a surprise.**

⭐ **Why the stream deserves credit rather than a correction.** It priced the fix, escalated, got an
approval, and then **kept measuring and refuted its own escalation's headline** — composing the
oracle with the real decode is work nobody asked for. That is the behaviour the programme wants,
and it caught a Master Mind error that would otherwise have shipped as an approved plan.

## M20. ⛔ THE OBJECTIVE I SET WAS ILL-POSED — a raw-logit margin is a property of the weight norm; and the priority reorders to CURVATURE DISCIPLINE first

### 1. My error, with its mechanism

I briefed the goal-margin stream that the binding quantity was *"the margin gap: only 0.297
logits"* and that widening it was the objective. ⛔ **That objective is VOIDED, and not by taste.**
MEASURED: `lat_head` is `LayerNorm → Linear`, so scaling its weights by *c* multiplies
`median_margin_gap_logits` by *c* **exactly** — at c = 3 the decode is **bit-identical 282/282**,
AUC unchanged to **< 1e-12**, and the gap moves **×3.0000**.

⇒ **A raw-logit margin can be made arbitrarily large without changing a single decision.** It is a
property of the weight norm, not of the model's discriminability. An objective defined on it is
optimisable to any value and means nothing.

⭐ **The scale-free quantities are the ones that should have been in the brief**, and the stream
supplied them: **AUC 0.7120** (shuffled 0.5055 ± 0.0336), **Cohen's d 0.5902**, **overlap
coefficient 0.5845** — and the 0.297 gap is **0.329 pooled SD**, which is the honest way to say it.
⚠️ **Class: a quantity quoted without the transformation it is invariant under** — the same family
as the units and estimator traps, with "scope" replaced by "scale". ⇒ **Before setting an objective
on a scalar, ask what transformation leaves the model unchanged but moves the scalar.**

⚠️ **Second, related:** the calibration-vs-separability verdict **flips with the turn definition**,
and the stream's own pre-registration failed to pin it — which it *named* rather than exploited. At
the inherited |κ| > 1e-3 (**R 1000 m**) it reads SEPARABILITY (balanced accuracy 0.6886); at the
curvature the token actually commands (0.08) it reads CALIBRATION (0.8339). **Same data, opposite
diagnosis, and only the threshold moved.**

### 2. ⛔ THE PRIORITY REORDERS: curvature discipline FIRST, vocabulary SECOND

MEASURED on the shipped `cos` arm + argmax (40 windows): executed curvature is non-zero on
**0.9091** of GT-**turn** windows — **and on 0.9000 of GT-STRAIGHT windows**, both **saturating
`kappa_max` = 0.2**, with direction correct only **0.4545**. Four families at **T1**: planner
**ADE 1.8944 [1.0547, 2.9504]** against `ha0_ext` **0.8772** — **2.16× worse than holding the
measured state, and it chose those plans** (`baseline_won_frac` 0.075).

⇒ **refav1 turns. It turns almost everywhere, at the clip bound, in the wrong direction half the
time.** ⭐ **And the uncontrolled variable is `W_KAPPA` — the curvature penalty — which is EXACTLY
ZERO in the `{0.0, 0.0, 64.297}` triple that BOTH of the predecessor's A/B scripts pass.** A
planner with no curvature penalty has no reason not to saturate `kappa_max`, and that is what the
readout shows.

⇒ **This sits UPSTREAM of both M15 and M19.** If the executed curvature saturates at 0.2 regardless
of a goal that commands 0.08, then the goal vocabulary is not what is binding — the **cost** is.
**Ordering, revised: (1) the one-variable `W_KAPPA` arm, both outcomes committed; (2) head recall
at the crossover (`M19`); (3) the vocabulary (`M15`, already conditional).** ⛔ The vocabulary work
does not stop, but **no vocabulary claim may be made until the curvature penalty is controlled** —
a comparison across arms whose curvature penalty differs by construction is confounded between the
vocabulary and the cost.

### 3. ⭐ What went RIGHT, and should be copied

* **The controls beat the stream's own verdict function, twice.** A re-fit was attempted twice and
  failed twice: the zero-parameter soft posterior shrank to zero with its real correlation **on the
  wrong side of its shuffled control**, and the trained ridge beat a constant by +0.019 while losing
  **94.5 % of its R² under nav permutation** — *it reads the route, not the road*. In both cases the
  stream's own verdict function said SUCCEEDS and its controls said FAILS. **The controls won.**
* ⭐ **The PI's vision-only rule is now VERIFIED at source rather than assumed.** `plan()` builds the
  goal input with `ego=None` ⇒ vision + nav only; the situation classifier enters **nowhere**; and
  nav-only AUC **0.5967** against the head's **0.7120** detection and **0.9043** direction ⇒
  detection and direction are **not** nav echoes. That closes the admissibility check the binding
  ruling requires, in the direction we wanted.
* **A shipped bug was found and fixed by a same-breath control** — a bare `%` in argparse help kills
  `--help`, caught only because the control read 0 as well.

### 4. Housekeeping that is not housekeeping

⚠️ **`intent_stride2.npz` (the 4,786 × 256 goal-head input bank) and the P4 dumps live ONLY on the
dev box** at `C:/Users/Admin/refav1_margin/`. That is the stranding failure the operating standard
exists to prevent — banked analysis is re-runnable, the inputs are not. **Ship or re-derivable, in
the next turn that touches this package.**

⚠️ **2 of 41 paths did not blob-verify** because `refa_v1.py` / `refav1_arm.py` carry a sibling's
uncommitted M15 work on top. ⛔ **Not lost, and correctly NOT swept into another agent's commits** —
the right call, and the reason it is visible here rather than discovered in an audit.

## M21. ⛔ CORRECTS M18 — I wrote an INHERITED claim as MEASURED; plus a fourth dead flag, and M17's cost measured

### 1. My error: M18's third bullet is retracted

M18 stated *"`w_agent` / `w_u0` are absent from `config.json` entirely — a run cannot say what
weight its detector trained at."* ⛔ **FALSE.** MEASURED on the escalating run's own banked
artifact: `seams.w_agent = 1.0`, `seams.w_u0 = 0.5`, **present since `3fd2291`**. They were absent
from the **top level**, which is where the escalation looked.

⇒ **I took an agent's escalation and wrote it into a decision as fact without re-verifying it.**
The operating standard is explicit — `MEASURED` · `PUBLISHED` · `INHERITED` · `ESTIMATED` — and
*"a claim that decides a GPU-day must be MEASURED or PUBLISHED, never INHERITED."* An escalation is
INHERITED by definition; promoting one to a ruling without a second read is how a wrong fact
acquires authority. **Second evidence-class failure of mine today**, after quoting an oracle
ceiling as a payoff (`M19` / retraction #29).

⭐ **There WAS a real defect underneath, and it is now fixed properly:** the stamp was a
**hand-written dict**, so any new knob silently failed to be recorded. `agent_knob_dests(parser)`
now reads all **17** `--agent-*` / `--w-*` dests **off argparse**, and `assert_knobs_stamped` runs
immediately before the config write — **a run whose record cannot state its weights refuses to
start.** The test asserts **by VALUE** over the argparse-derived set, so it survives key renames
and cannot be satisfied by a rotted list, with a same-breath control that ≥ 12 knobs were probed.

### 2. ⛔ A FOURTH dead flag, and it is the worst of the four

`--agents off --w-agent 1.0 --agent-join <file>` **PASSES** the existing loss-time guard — which
fires only when `agent_box` is *absent*, and the join puts it there — falls through
`"agent_slots" in out`, computes **nothing**, and stamps **`w_agent: 1.0`**.

⇒ **That configuration manufactures *"the agent head does not help"* from a seam that was never
built.** It is the silent-no-op class with the sign flipped: not a missing capability reported as
present, but **a null result produced by a switch that was off**, carrying a record that says it
was on. ⚠️ No banked run is affected. Now refused at startup.

### 3. `_rig_camera` is WIRED, not removed — and the third state is gone

The projection and ground terms answer the one axis a monocular head cannot see, so deleting the
flags would have removed **capability** to tidy a defect. `--agent-rig-camera {off,nominal,extrinsics}`
now exists and every failure is a **startup refusal, before a model and before the GPU**: a weight
> 0 with the camera off refuses (the deliberate-regression control), a camera set with `--agents off`
refuses (the mirror image), and ⭐ **geometry with no DECLARED `CanonicalFrame` refuses rather than
inventing an `f_ref`** — *a frame is not its pixel count, and this corpus is cylindrical.* Both
weights zero + camera off leaves every banked arm bit-identical. An AST pin forbids
`model._rig_camera = None` ever returning.

⭐ **The self-catch worth copying:** the first proof draft used a random head and read **exactly 0.0
over n = 0** — the no-information value, which would have "passed". The shipped proof requires
**n > 0** and a perfect prediction reading exactly 0.0 *over that n*.

### 4. M17's cost is MEASURED, and it does not bind

`--agent-queries` 32 → 100, shipped path, 256×640 (160 memory tokens), `n_pad 94`, **each arm run
twice**: step time **3.346 → 3.622 s = +8.2 %** (replicate spread 3.8 % / 1.9 %; the arms' medians
**do not overlap**, 3.4096 < 3.5878); parameters **+17,408 = 68 × 256 exactly = +0.077 %**; peak
memory **no detectable difference** — and the analytic activation delta of **≈ 8.1 MiB** against a
~500 MB step is *why* the null, which is the difference between a measured null and an absent
measurement.

⇒ **M17 stands, now with a number instead of an assumption.** ⚠️ **Scope: dev-box CPU** — the GPU
was occupied by another stream throughout, so none was added. The ratio and the
parameter/activation accounting transfer; **the absolute does not**, and a pod-side re-measure is
owed.

### 5. The repo hazard is resolved

Two streams independently flagged it. MEASURED and cleared: `.git/index.lock` had stood since
**15:15** and blocked every index write. ⭐ **The safe test is that on Windows, deleting a file
another process holds open FAILS** — so attempting the removal *is* the check, unlike POSIX where
`unlink` succeeds regardless. It removed cleanly ⇒ nothing held it. Index backed up first
(1,694,353 B) and verified intact after (**9,935 files**, control read OK).

Then **151 staged deletions classified 151/151 PHANTOM** (present in HEAD *and* non-empty on disk,
0 genuine) and cleared index-only in batches of 30, worktree untouched — and the `CLAUDE.md`
landmine is **REPAIRED**: its index entry now matches HEAD, where before it was a blob **5,353 B
shorter** whose commit would have silently reverted the `H-ESTIM-SEED-1` block.

⚠️ **The orphaned `git grep` (PID 24768, started 2026-09-04 07:54, parent gone) SURVIVES a
`Stop-Process`** — almost certainly blocked on G: mount I/O, which is also why it outlived a day of
outages. It does not hold the index lock and is not the cause; it is left alone and recorded.
