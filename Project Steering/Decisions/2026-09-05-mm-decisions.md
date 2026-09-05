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

## M22. ⛔ THE L=3 CHOOSER IS DEFERRED, NOT BUILT — and `run_ab*.sh` passes `W_KAPPA = 0`, which scopes every arm banked through it

### 1. The chooser: DEFERRED

M15 approved a three-level vocabulary. The stream implemented it, validated it **against the
approval rather than assuming it** (the constant reproduces the approved L=3 row at medAE
**0.00420751** / expressible **1.0000**, and the shipped control reproduces **to every printed
digit**), and then escalated correctly: ⛔ **M15 approved a vocabulary and no chooser.** The v7.0
head has ONE `TURN_L`/`TURN_R` slot and `kappa_turn` never touches its logits.

⇒ **Decision: do NOT build a chooser now.** Three reasons, and the second is decisive:

1. **The goal side is now the least likely place to fix refav1.** A perfect goal delivered
   honestly LOSES to `ha0_ext` on turns (1.9486 vs 1.8567) while the shipped arm WINS (0.7868), and
   the `kappa_max` saturation **survives de-confounding** — it is the goal field, not a missing
   seed. Building a chooser now optimises the lever we have the most evidence against.
2. ⛔ **A 3-way chooser makes the binding term WORSE, not better.** `M19` measured *"turns goaled
   correctly" = **0.2811** for every magnitude* — the head's **binary** decode already fails on
   71.9 % of real turns. Splitting the same probability mass across three magnitudes makes that
   decode harder, and with the cost saturating at `kappa_max` regardless **we would have no way to
   read whether it helped.** An experiment that cannot fail informatively is not worth its GPU.
3. The cheapest informative arm needs no chooser at all: **L=3 with an ORACLE κ hint, at T0.** It
   bounds what a perfect chooser would buy, before anyone trains one.

⭐ **The stream's implementation choice is ratified and is the right shape:** `goal_kappa_hint` is
**REQUIRED, never defaulted** — *"a default would hide the tier."* An arm that silently supplies an
oracle hint would be a **T0 ceiling wearing a T1 costume**, which is the exact confusion the tier
stamps exist to prevent. ⇒ L=3 stays in the tree as a **parity-pinned instrument**, not a product;
its zero-flag path is bit-identical and every arm stamps `goal_kappa_vocab`.

### 2. ⛔ `run_ab.sh` / `run_ab_targeted.sh` pass `W_KAPPA = 0` — a scope caveat on banked arms

MEASURED: both drivers pass the weight triple `(0, 0, 64.297)`, i.e. the **curvature penalty is
exactly zero**. A planner with no curvature penalty has no reason not to saturate `kappa_max`, and
the readout agrees — curvature is non-zero on **90.0 % of GT-STRAIGHT windows**, at the clip bound.

⇒ **Every arm banked through those scripts sits on a planner that curves at the clip bound on
straight road.** That is not a retraction: the arms are internally paired and their *contrasts*
remain valid, because the penalty is zero in both. ⛔ **What is inadmissible is reading any of them
as refav1's driving behaviour**, or comparing them to an arm with a non-zero penalty. Any such
number carries `W_KAPPA` or it is not quotable — the same rule as *never quote an interval without
its estimator*, with the object being a cost weight.

⇒ **The one-variable `W_KAPPA` arm is the programme's next deciding measurement** (`M20`), it is
**already queued** behind the running `ccos` arm, and its driver has been polling for it. Both
outcomes are committed by the stream that queued it.

### 3. What is now settled about refav1, and what is not

**Settled (MEASURED):** it turns; it turns almost everywhere, at the clip bound, in the wrong
direction about half the time; the head is not the binding constraint (**74.4 % recall at AUC
0.8806** on expressible turns); the goal is not either (a perfect one is worse); the vocabulary
explanation of the floor gap is **withdrawn at its premise** (the gap is equally present on
straights).

**Not settled:** whether a non-zero curvature penalty changes any of it. ⛔ **Until that arm reads
out, "refav1 does not drive" is a statement about a planner whose curvature penalty was switched
off** — which is worth saying plainly, because it is both the honest scope and the most hopeful
open question the arm has.

## M23. The RL exit is FAILURE as committed — and the product is the thing that fell out of it: a ZERO-TRAINING kinematic gate

### 1. The formal exit stands: FAILURE

SPEC 7's committed SUCCESS text required feasibility to improve **and** ADE not to regress past the
replicate floor. MEASURED at **T1** (4,823 windows / 141 episodes, `void: false`, 0 dropped):
`ade_m` **+0.0362 [+0.0261, +0.0460] separated** ⇒ **the ADE guard fails.** ⛔ The committed rule is
reported as written and the exit is **FAILURE**. *(It is 14× gentler than the composed-reward arm's
+0.5014 m, which is context, not a pass.)*

⭐ **And the replicate did its job again:** it killed **four** metrics a single seed would have
shipped — including `top32_infeasible`, the metric the *previous* package proposed shipping on,
which spans **13.6× across three runs, two of which share a seed**. Only `fan_peak_g_mean` survives:
**−0.0929 / −0.1158** against a **0.0229** seed floor, with `ctrl_null` drifting **+0.1343 the other
way**, and three independent runs all negative.

### 2. ⭐⭐ THE PRODUCT — and it is not the veto

Split by object, because the axes disagree: the **emitted fan** is safer (−0.098 g, three runs,
clears both floors); the **driven path is not** (`sel_peak_g` WORSENED, +0.0155/+0.0080 against a
0.0075 floor).

⇒ **But the driven path IS safer under a top-2 KINEMATIC GATE that requires no training at all:**
`sel_envelope` **0.1062 → 0.0729 (−31 %)**, `sel_peak_g` **−20 %**, and `ade_m` **+0.0037, NOT
separated** — using **only the candidate's own waypoints**: no scene input, no new perception, no
gradient step.

⇒ **Decision: PROMOTE it to a pre-registered arm, do NOT ship it.** It was found **post-hoc**, by a
stream that was not looking for it, and this programme has a standing rule that a separated result
from one seed is necessary and not sufficient. ⛔ Its own SPEC must be banked **before** the
validating arm runs, carrying: both outcomes committed, an **inference-seed replicate**, the
four families, and a **deliberate-regression control** (the gate disabled must reproduce the
ungated numbers bit-identically). ⚠️ A post-hoc finding promoted without that ceremony is exactly
how `top32_infeasible` nearly shipped.

### 3. The reward is NOT disqualified, and the attribution names one term

P1's disqualification branch did **not** fire. MEASURED over **30,720 candidate scores** (240
windows / 121 episodes), per-window Spearman, episode-cluster bootstrap:
**ρ(default reward, envelope) = −0.5367 [−0.5581, −0.5138]**, `kamm_over` −0.5790, `contact`
−0.6326 — **all negative, all separated** ⇒ the reward ranks violating candidates **lower**, which
is the direction we want. Controls read their known values (self **+1.0000**, constant UNDEFINED
240/240, random **−0.0138** = the probe's own bias floor).

⭐ **`progress` is the ONLY positive term** — ρ(progress, `peak_g`) **+0.2900**, `ttc_below`
**+0.4697**, `contact` **+0.3286**. And the arithmetic prices the obvious fix at zero: **`feasibility`
×4 moves ρ(envelope) by 0.001; deleting `progress` moves ρ(`peak_g`) by 0.27.** ⇒ **Turning the
feasibility weight up is a 270× worse lever than touching the term that actually rewards violation**
— which is why `M20`-era instinct ("add a feasibility term") was wrong, and why the arm that had
`feasibility: 0.5` got worse.

### 4. ⛔ A "NULL" ARM WITH A LIVE OPTIMISER IS NOT A NULL — AdamW normalises the gradient away

MEASURED: with the veto now explicit, `ctrl_null` reads `veto_rate_mean` **0.0000 exactly** (it read
0.0897 this morning) — and it **still moved all 71 trainable tensors** (mean |Δθ| 3.1e-05) and
**separated 35 of 57 metrics**, more than the 14/57 that fired the previous VOID gate.

⇒ The mechanism: the only live loss is the trust region at ~**1e-10 m²**, and **AdamW normalises by
the gradient's own scale**, so a vanishing loss still produces unit-scale updates. ⛔ **`ctrl0`
(lr = 0) is the ONLY arm that cannot move**, and it is therefore the only admissible zero-lever
floor on this rig.

⭐ **The general rule, which belongs beside the estimator family:** *a control defined by zeroing a
LOSS is not a null under an adaptive optimiser; only a control that zeroes the UPDATE is.* Same
shape as the three variance questions riding on one interval — the arm is arithmetically what it
claims and answers a different question than the one being asked of it.

### 5. What the veto is actually worth, and the real work item

⛔ **The veto addresses ~2.7 % of the gap.** MEASURED: the frozen anchor vocabulary is drivable at
**0.48 g** while the decode emits **4.11 g** — **8.56×** — so the infeasibility is manufactured
downstream of the vocabulary. ⇒ **The real work item is a FEASIBILITY-AWARE DECODE**, not a better
reward and not a bigger veto. That is now the deploy-side successor to this line.

### 6. Retractions this stream logged against itself: #26, #27, #30

⭐ **#30 is the one worth carrying:** a λ-sweep's identity control **passed while interpolating the
wrong tensor** — *the control checked the arithmetic, not the object.* ⇒ **A control must assert
WHICH object it operated on, not only that the operation was self-consistent.** This programme's
controls have caught many things today; this is the first time a control was itself the defect, and
it generalises to every identity/parity check we run.

## M24. ⭐ THE KINEMATIC GATE IS QUOTABLE AT **T0** — and this is the first Rule Zero turn that behaved exactly as intended

### 1. The result

**The driven path is measurably safer at no measurable ADE cost**, on two **episode-disjoint** draws:

| metric | draw A | draw B | replicate floor | verdict |
|---|---|---|---|---|
| `sel_envelope` | **−0.0300 SEP** | **−0.0200 SEP** | 0.0100 | **QUOTABLE** |
| `sel_peak_g` | **−0.0378** | **−0.0330** | 0.0048 | **QUOTABLE** |
| `ade_m` | +0.0064 | −0.0099 | 0.0163 | not separated, **sign flips** ⇒ no cost |

⭐ **And one four-family movement is separated and is an IMPROVEMENT:** LONGITUDINAL `accel_mae`
**−0.1846 [−0.2840, −0.1014]** and **−0.1402 [−0.2327, −0.0629]** against a 0.0444 floor ⇒
**−19.5 % / −14.3 %**.

⛔ **BUT THIS IS T0, AND T1 WAS NOT RUN** (the 4060 was saturated all session). Per
`EVAL_DOCTRINE`, **T1 is the primary tier for any capability claim** ⇒ **this is not yet a driving
result and must not be quoted as one.** The hook is one site — `taniteval/tools/refcv3_arm.py:1117`,
all four tensors already in `out`, and it **must use `sel_score_v3`** — and it needs a GPU slot.

### 2. ⭐ All three variances were named BEFORE any number existed

This is the `CLAUDE.md` THIRD-VARIANCE block being applied on the day it was written, and applied
better than it was written:

* **V1 (episode draw)** — paired episode-cluster bootstrap **plus an episode-disjoint second draw**.
* **V2 (training run)** — ⭐ **STRUCTURALLY ABSENT**: zero training steps, one frozen checkpoint, no
  seed enters. The hole `H-ESTIM-SEED-1` warns about **cannot exist here**, and saying so is better
  than measuring a floor for it.
* **V3 (inference sampling)** — ⭐ **ASSERTED, NOT ASSUMED**: `refc.py:1720`/`:1501` gate their
  stochastic ops on **config, not `self.training`**, so two forwards on one batch in one process are
  **bitwise identical**. That makes V3 a **structural identity, not an estimate of zero.**

⇒ **The `one_variable` claim is held BY CONSTRUCTION**, not by care: one forward per window banks the
fan, and both rules, both draws, all four families and every bootstrap are computed **offline from
that same fan**. `model` and `gate_k` differ only in *which of 128 already-emitted candidates is
returned*. That is the strongest form of one-variable control this programme has produced.

### 3. ⛔ Its own deliberate-regression control found a confound — and the identity control had never run

The probe ranked by `sel_score`; refcv3 is the **`hier`** arm and `refc.py:1763` argmaxes
**`sel_score_v3`**. They disagreed with the model's own `sel_idx` on **35/400 (8.75 %)** and
**29/400 (7.25 %)** — *the gate's candidate set did not contain the model's own pick on about one
window in eleven.*

⚠️ **And the reason it went unnoticed is the sharpest lesson:** the probe's docstring called `k = 1`
*"a built-in identity control"* while `GATE_KS = (2, 4, 8, …)` — **`k = 1` is not in the tuple, so
the identity control never ran.** A control that is documented but not executed is worse than none,
because it is cited. ⇒ On the corrected ranking `gate1 == model` on **400/400 and 400/400, every
metric exactly 0.0, asserted on the INDEX** (retraction #30's rule: assert *which object*).
The confound had inflated gate2's share from **46.2 % → 50.3 %**.

### 4. The 8.56× question was MINE and it had the wrong denominator

I asked how much of the 8.56× fan-to-vocabulary gap the gate closes. **Answer: 0.00 %, and that is
a STRUCTURAL ZERO** — the 8.56× is a **fan** property and a selection rule changes no waypoint.

⭐ The correct denominator is the one a re-ranking rule actually owns: **D2 = 0.0819 g**, against
which **gate2 closes 46.2 %**. And the driven path is already **24.3× better conditioned than the
fan's average member** and **2.80× below the vocabulary's own 0.4808 g**. ⇒ **A question can carry a
scope error as easily as an answer**, and the right response was to correct the denominator rather
than compute a meaningless ratio.

### 5. What it does NOT claim — all three volunteered

* ⛔ **Its own committed LATERAL hypothesis is NOT supported** — curvature and yaw-rate **sign-flip
  between draws**. The gain comes from the *longitudinal* half of `comfort`, not the curvature half.
* ⚠️ **The real cost is TACTICAL:** manoeuvre agreement **−2.5 pp** consistently on both draws, and
  the gate changes the pick on **~49 %** of windows. ⛔ **No paired CI exists for those rows** — and
  that is *stated as un-intervalled, not hidden*.
* **"No scene input" is sharpened, not just confirmed.** Under a full scene garble the gate's score
  **and its argmax** are bitwise unchanged (0.000e+00) while `headway` moves 0.75, `collision` 1.0,
  `progress` 3.06. But the **candidate set** is `sel_score_v3` masked by `reach_keep` — model
  outputs that are scene-conditioned. ⇒ the admissible wording is **"adds no NEW scene input and no
  new perception"**, and `M23`'s phrasing is corrected to that.
* **STRATEGIC reads `UNAVAILABLE` with `defect: false`** — a corpus property, **not**
  `D-NAVCOMP-SHAPE-1`. The defect classifier added today did its job on its first real use.

### 6. ⭐⭐ Rule Zero worked, and so did the bar it does not lower

The successor arms **ran without being asked**: similarity scaling was found to be the **wrong
operation** (lat_acc scales by *s*), and a control-space-roll stand-in produced *"a spectacular
result that is NOT quoted because its known-answer control failed at 3.17e-2 m."*

⇒ **That is precisely the pair this rule was written to produce**: keep going after a refutation,
**and refuse to quote a spectacular number whose control failed.** It is the clearest evidence so far
that Rule Zero can be followed without loosening anything.

## M25. refcv5 is CODE-COMPLETE and VALIDATED — **NO-GO on four blockers, none of them code** — and a dead flag one level deeper than any so far

### 1. ⛔⛔ `--agent-w-ground` COMPUTES A TAUTOLOGY. The term runs, and is zero by construction.

MEASURED: `ground_range_prior` projects a foot at rig `z = 0` and back-projects onto
`z = ROAD_PLANE_Z_M`, **which is 0.0** — exact inverses, so `r_back == r_pred` **by
construction**. Loss **2.61e-08**, gradient **8.73e-11**.

⇒ An arm would **train, converge, stamp the weight, add exactly zero**, and read afterwards as
*"the ground prior does not help."*

⭐⭐ **THIS IS A NEW AND DEEPER MEMBER OF THE DEAD-FLAG FAMILY, AND IT DEFEATS EVERY GUARD WE
BUILT TODAY.** The four earlier ones were *the term is guarded out* — a flag set while the code
path never executes, catchable by inspecting the guard. Here **the flag is set, the camera IS
built, and the term DOES run.** No flag audit, no config-provenance assertion, no
argparse-derived stamp can see it. ⇒ **Only a GRADIENT PROBE can.** It now refuses at startup,
with a **same-breath positive control** — a flat control stops the run as INCONCLUSIVE, never as
a pass.

⇒ **RULE: a loss term earns its weight by producing a NON-ZERO GRADIENT on real data, asserted
once at startup — not by appearing in the loss sum.** Same family as *"a flag that parses and
does nothing is worse than a missing one"*, moved one level in: **a term that computes and
cancels is worse than one that never runs**, because it survives every structural check.

### 2. The per-clip camera is real, and its guard fired on its first run

`agent_losses(cam=…)` now takes `None` · one `RigCamera` · **a per-ROW sequence**, resolved by a
`RigCameraBank` keyed on `stable_episode_id(clip_id)` from the batch's new `agent_ep`, selected
with the **same mask** as the targets. `config.json` stamps `mount_pose_scope`
(`PER-CLIP` / `SINGLE-CAMERA-WHOLE-CORPUS` / `NONE`), and an uncovered corpus **refuses** unless
accepted by name. Controls: **B identical cameras reproduce the single-camera path to < 1e-12**
(so no banked arm shifts), and a mixed batch lands **strictly between** two mount heights (so
`cams[b]` is genuinely read). Root-cause class **C28** — *a constant where the quantity is
per-clip*.

⭐ **`build_rig_extrinsics_table.py` covered 2,400/2,400 parity clips, and its clip-set assertion
against `parity_manifest.json` FIRED ON THE FIRST RUN**: `r0_selection.parquet` holds **500**
clips, and a table built over those would have supplied cameras **for a different corpus** —
silently, and with every downstream number looking fine. *Parity is sacred, and this is what
enforcing it looks like in code rather than in prose.*

⚠️ **Camera height is MEASURED on parity for the first time: 1.2131–1.6672 m, 554 distinct values
across 2,400 clips** — refuting **both** prior bands (12-clip and 40-clip) **at both ends**. The
stream retracted its own sentence from an hour earlier (*"1.22 m is below every observed
minimum"* — false on parity) and logged the class: **a small-sample extremum quoted as a bound.**

### 3. The standing blocker is REFUTED — absence had been found at one location

*"No parity-corpus episode source on this box"* has gated refcv5 for days. MEASURED: the complete
**85.00 GB** v2 cache is **on HF**, the join is **on this box**, md5 exact. ⭐ And it was priced on
**the consumer's loader**, not on the 349 GB raw epcache — the `DE-C152` rule applied correctly,
which is the difference between an 85 GB move and a "1.38 TB wall".

### 4. Validation: `V-RC5-READY` passes all five gates

SPEC banked **before** any arm ran. **Four deliberate regressions refused before a checkpoint
existed**; the **converse control was NOT refused** (so the guard is not over-broad — the check
that makes a refusal evidence rather than a blanket); and the **replicate arm is byte-identical**.
Preflight `refcv5_preflight.py`: **14 PASS / 1 FAIL / 2 INCONCLUSIVE**, INCONCLUSIVE counted as
failure. Parity digest MATCH · join **2,308/2,400 on the stable id, 0 legacy-only** · per-clip
cameras **2,400/2,400** · **18/18 knobs stamped**.

### 5. ⛔ GO/NO-GO: **NO-GO.** Four blockers, and the split matters

| # | blocker | whose |
|---|---|---|
| 1 | **a GPU.** The only live pod is training refcv4b (**67.4 %**, ~10 h left); five legacy pods refuse connections | ⛔ **PI — spend** |
| 2 | the **85.00 GB** cache moved onto it (~12 min HF→pod) | mechanical, no decision |
| 3 | an **`anchors.pt` with DECLARED units** — without one the run silently falls back to the synthetic default (oracle-in-vocabulary **1.0882 m** vs **0.3796 m**) | work item — build it |
| 4 | **v7.2 supervision covers 190/2,400 = 7.92 % of parity**, against the trainer's own **50 %** floor | ⛔ **PI — ruling, or a label build** |

⭐ **On (4), my recommendation is the parity-scoped label build, not the kin3 fallback.** At 7.92 %
coverage the TACTICAL and STRATEGIC families would be computed on **one twelfth** of the corpus —
and today's most expensive lesson is that a family which is *thin or absent* in the record is worse
than one that fails, because it reads as fine. The STRATEGIC family was silently missing from
**every refcv3 eval for its whole life**. Starting refcv5's first real arm with 8 % tactical and
strategic coverage would manufacture exactly that shape: three families that look healthy and two
that are statistically empty. ⚠️ Note also that the trainer already carries a **50 % floor**, i.e.
someone has already ruled that 8 % is not enough — running under it would be overriding a
committed guard, which needs to be a decision and not a default.

⚠️ **Operational note worth keeping:** a docstring edit was **silently reverted between writing and
committing**, caught only by the end-of-turn marker check whose control read 6. That is the
*"a commit is not a latch"* class, observed live again.

## M26. ⭐⭐⭐ THE RESULT: refcv3's driven path is measurably safer at **T1**, and the whole cost is **1.2 mm**

### 1. The number

**T1**, 4,823 windows / 141 episodes, paired episode-cluster bootstrap, `void: false`:

| | before | after |
|---|---|---|
| envelope-violation rate | 0.0865 | **0.0000** — a *structural zero*, not a small number |
| yaw-rate error | 0.2176 rad/s | **0.0427 rad/s** — **−80.4 %**, separated |
| friction load | — | **−9.6 %** |
| **total cost** | — | **`ade_0_2s` +0.0012 m** |

⇒ **1.2 millimetres.** The RL stage this replaces failed at **+0.0362 m separated** *while making the
fan less safe*: **30× more ADE for a worse outcome.**

⭐ And the LATERAL family carries the headline in the strongest possible form: yaw-rate goes from
**worse than a constant-velocity straight line** to **better than it**.

### 2. Why it works — and the finding that reframes the whole offset-head line

`stack/tanitad/refs/feasible_decode.py` is a **control-space projection that inverts the EXACT
finite-difference map `fan_safety.score_paths` uses**, clamps to the box + Kamm disc, and
re-integrates sequentially. ⇒ **an envelope- or Kamm-violating path becomes UNREPRESENTABLE** — an
identity, and ⭐ **asserted through the CONSUMER rather than our own bookkeeping**, which is the
discipline that has caught three defects today.

Fan: `envelope` **0.8879 → 0.0000**, `kamm_over` **0.8408 → 0.0000**, `peak_g` **4.1789 → 0.5964 g**
⇒ **96.87 % of the gap, 8.69× → 1.24×.** The pre-registered `+entry` variant drives
`fan_infeasible` **0.8916 → 0.0000 exactly** across all **51,200** candidates.

⭐⭐ **THE REFRAME: at MATCHED `fan_peak_g` the projection costs −0.0103 m of oracle-ADE, where an
isotropic shrink costs +0.7680 m. It costs NEGATIVE.** ⇒ **The infeasible component of the offset
head's 9.29 m displacement was WASTE, not a trade against accuracy.** Every previous reading of that
displacement — including my own in `M23` — implicitly assumed it was buying something. It was not.

⇒ This also settles the question **I** framed badly: I asked how much of the **8.56×** a *selection
rule* closes (answer: 0 %, structurally). The right instrument was never a selection rule; it is the
decode, and it closes **96.87 %**.

### 3. What FAILED, reported as written

* ⛔ **`progress` FAILED its committed criterion.** `progress_v2` clears the ranking bar
  (ρ vs `peak_g` **+0.2900 → −0.5897**; `ttc_below` +0.4697 → −0.1820; composed reward
  −0.4603 → −0.6898) and **fails the progress bar at 0.7123 against the committed 0.95**.
  `progress_v3` fails *worse* on both draws (+0.5042 / +0.5135) — because `along` is net +x and the
  projection **straightens** the swinging candidates.
  ⭐ **Then the question DISSOLVES:** on a feasibility-aware decode, ρ vs `envelope` is **UNDEFINED
  on 400/400 windows** — there are no violations left to correlate with. A comfort-grade preference
  survives (ρ = +0.3232) and a reward change is still owed; **the safety-grade defect is gone.**
* ⛔ **`P2-C4` FAILED as committed** — `off_reach` **+0.2311** on the headline arm; only the
  `+entry` variant removes it. Stated, not buried.

### 4. The floor, and why it is stronger than `ctrl0`

⭐ **The lever takes ZERO gradient steps**, so `ctrl0` is not the right floor — the *disabled-lever
arm* is, and it is **bit-identical on 4,823/4,823 windows, max |Δ| = 0.0**, independently confirmed
by the harness's own degeneracy profile. ⚠️ Scope stated by the stream itself: **that retires
training variance for THIS ARM ONLY.** TACTICAL and STRATEGIC read exact zeros **by construction**
(those heads are never touched) — said out loud so nobody reads a structural zero as evidence of
safety.

### 5. Two self-corrections, both caught in-turn by controls

1. Its SPEC claimed *"monotone renormalisation is a no-op"* — **exact for the unclamped ratio and
   FALSE for the clamped component**, because ties move a Spearman.
2. *"The residual is a scorer artifact, not a leak"* — ⭐ **refuted by a second IMPLEMENTATION
   reading the same 0.3396.** It was a **real leak** (a stopped path read as a hard turn), now fixed
   and **0/51,200 exact**. ⚠️ And the fix itself was wrong **twice in the same scope-error family**
   (float64 exactness ≠ the fp32 consumer's reading; an absolute threshold where the quantity is
   relative).

⇒ **A second implementation beat a second look, again.** That is the `ls-tree` lesson in a new
costume: repeating a probe through one channel is one sample.

### 6. Status and what is owed

**Default OFF**, disabled path returns the same object, 15 pins in
`stack/tests/test_feasible_decode.py`, 37/37 paths blob-verified.

⛔ **Owed, and named rather than assumed:** the **in-decoder** arm — the confidence head ranking
*projected* geometry — needs **one ~1.5 h GPU forward**, and the 4060 is saturated by refav1 plus
the sibling panel. **Not-yet-run, not not-possible.** Until it runs, this result is the projection
applied to a fan the model ranked *before* projection.

⇒ **This is the deploy-side answer to `M23`'s "the real work item is a feasibility-aware decode",
and it arrived within the same day.** ⚠️ It is a **decoder option**, not a new model version, so it
takes no `MODEL_REGISTRY` row until an arm ships with it ON.

## M27. refav1 reaches **PARITY** with the trivial floors and **BEATS** them on turns — and the remaining blocker is now named exactly

### 1. The number. `W_KAPPA` is a real lever, and it is large.

**T1**, 40 windows / 8 episodes, ckpt 21,109, every arm carrying its `(metric, W_JERK, W_KAPPA, W_VEND)`:

| arm | ADE | curv MAE | heading MAE | cross MAE | goal FDE |
|---|---|---|---|---|---|
| `ccos` **W_KAPPA 0** (the banked A/B setting) | 1.3272 | 0.055369 | 23.4578 | 0.8784 | 2.9639 |
| **`ccos` W_KAPPA 15.11245** | **0.8934** | **0.030982** | **15.2704** | 0.3670 | **2.1628** |
| `ha0_ext` floor | 0.8772 | 0.077298 | 27.7357 | 0.4018 | 2.2693 |

⇒ **ADE 1.3272 → 0.8934, −33 %**, and the paired gap to `ha0_ext` moves from **+0.4500** to
**+0.0162 [−0.1648, +0.1980]** — *smaller than this rig's own **0.0607** inference-seed floor*.
⭐ **Against the floors that is PARITY, reached while the planner is genuinely acting** — and on the
**GT-turn stratum it BEATS `ha0_ext`, 0.9699 vs 1.1521** (`frac better` 0.53).

**Each row against the inference-seed floor** (so each is a lever effect, not noise): ade −0.4338
(**7.1×**), fde −0.8012 (2.3×), cross −0.5114 (7.2×), heading −3.8969 (3.5×), yaw-rate −0.1571
(**19×**) all better; **speed_mae +0.0764 (20×) and accel_mae +0.0874 (14×) WORSE**. All seven clear
the floor.

### 2. ⭐⭐ The shipped configuration is the do-nothing plan — CONFIRMED TWICE, INDEPENDENTLY

I measured it from the dumps (`|cl − ha0| = 0.000e+00` on 40/40, control `|cl − ha| = 10.99 m`);
the stream measured it from the record (`paired ADE +0.0000 [0, 0]`, control `ccos_argmax` 15.08 m
apart) and added what I could not see: the control tensor is **exactly zero on 40/40**, and
⭐ **it reaches that BY SEARCH** (`plan_source` cem 0.425). ⇒ **The shipped cost's optimum IS the
do-nothing plan.** Not a degenerate fallback — the optimiser looked and chose nothing.

⇒ **This reverses my own framing in `M22`.** I wrote that arms banked at `W_KAPPA = 0` sat on a
planner curving at the clip bound, as though zero were an oversight. MEASURED: **at the shipped
triple the planner does not curve at all**, and `W_KAPPA = 0` is the setting at which it acts.
⚠️ And the collapse is **not attributable to `W_KAPPA` alone** — the shipped triple also drops
`W_VEND` **64.297 → 0.1**, a 643× cut in the goal-endpoint weight. The clean one-variable arm is
`wk15`, and that is the one in the table above.

### 3. What each lever did, and what it did not

* **`W_KAPPA`** — the whole lateral family, plus straight-window damage **5.5× smaller**
  (+1.0675 → +0.1957) with the turn-window win kept. ⛔ Did **not** fix longitudinal (made it
  separably worse) and did **not** move the tactical decision metrics at all (+0.0000).
* **`ccosh` hold branch (L2)** — the gate **fires** (`basecost_cv` exactly-1.0 fraction 0.20 → 0.00),
  but realised curvature is unchanged on matched episodes. ⚠️ **And my brief's number was wrong for
  this grid**: its addressable population is **~20 %** of windows, not the **86.5 %** I quoted — that
  figure came from a different grid. *A count carries its grid, or it is not a count.*
* **Seed pool (L3)** — ⛔ its own pre-registered premise **refuted**: `wk15` realises **15 distinct
  curvatures including R 19–87 m**, so the gate decides whether curvature is *seeded*, not whether
  the search can *modulate* it. `D-REFAV1-DRIVE-GATE` stands (LANE_KEEP still yields exactly 0 on
  18/18).
* **Feasible decode (L4)** — ⭐ **reused the sibling's module rather than rebuilding it**, and it paid
  immediately: refav1 **never leaves the actuator box** (`envelope_rate 0.0000`) but leaves the
  **μ = 0.7 friction circle on 29.6 % of windows at v0 ≥ 2 m/s, 42.1 % at ≥ 5 m/s, peak 3.262 g**
  against ground truth **0.373** — **because `kappa_max` is a CONSTANT.** `PlanConfig.kamm_mu`
  (|κ| ≤ μg/v² on the candidate's own speed) implemented, 15 pinned controls, arm queued.
* **Retrain the goal head (L5)** — **not taken, and the evidence says do not**: the now-complete
  8-episode de-confounded oracle reads `cl − cl_oracleseed` ADE **−1.2181 [−2.3250, −0.3862]**,
  **20× the seed floor**. A perfect goal is worse, and that is now well-powered.

### 4. ⭐ `H-ESTIM-SEED-1` reproduces on the INFERENCE rig

Two arms differing **only** in `--plan-seed` read **`separated` on 4 of 10** paired family metrics.
⇒ the THIRD-VARIANCE block added to `CLAUDE.md` today is not a theoretical worry; it is a **40 %
false-positive rate** on this rig, measured the same day it was written.

### 5. ⛔ The blocker, named exactly — and it is the next work

**refav1 does not beat the floors; it reaches parity and wins on turns.** The gap is the
**LONGITUDINAL family**, which **no lever in this package touches**.

MEASURED mechanism: **29 of 40 windows decode `ADAPT_SPEED_FOR_CURVE`, whose canonical control is
`a == 0`** — so the plan holds **constant acceleration** while `ha0_ext` holds the **measured a₀**.
That is why `speed_mae` sits at 0.79 against the floor's 0.31 while every lateral metric wins.

⇒ **The next arm is the LONGITUDINAL ANALOGUE of this package's levers** — a longitudinal vocabulary
whose canonical control is not identically zero. It is **unblocked, unbuilt, and next**, and it is
the first time refav1's remaining gap has had a single named mechanism rather than a list of
suspects.

⚠️ Three levers (`ccosh`, seed pool, `kamm_mu`) are implemented, pinned, and **OFF by default**;
making any of them the default is a PI/Master Mind decision, as `COST_METRICS` itself requires.

## M28. L2 is a CLEAN NULL with a mechanism — and a process-counting error that halved a queue and inflated my own reports

### 1. `ccosh` (the hold branch): the cost is repaired, and not one plan moves

`ccosh_w000` = `ccosh` + `(0, 0, 64.297)` — **the metric is the only variable** against banked
`ccos_argmax`.

* **Plans bit-identical:** `cl` trajectories max abs diff **0.000000e+00** over 40 windows; the
  paired episode-cluster bootstrap reads **`+0.0000 [+0.0000, +0.0000]` on all ten family metrics**.
* ⭐ **Same-breath control that must read non-zero, and does:** `wk15` against the same baseline is
  **1.508402e+01 m** apart with seven separated deltas. ⇒ **the zero is a measurement, not a dead
  pipe** — which is the whole difference between a null and a broken harness.
* **The cost genuinely changed:** `basecost_cv` frac == 1.0 **0.2500 → 0.0000**, min
  **0.884896 → 3.9105e-08**, median `plan_cost` **3.10838e-05 → 9.26852e-06**.

⭐⭐ **And one number is the mechanism:** the `cem` win fraction rose **0.750 → 0.925** *while the
emitted controls stayed bit-identical*. ⇒ **On the affected windows the CEM's own best sample was
ALREADY the all-zero plan**, so making `cv` cheap only re-labelled which of two *identical* control
sequences won the argmin. A repaired cost cannot move a plan that was already the optimum.

⇒ `ccosh` stays a **pinned, non-default, CORRECT instrument** — it makes the cost defined where it
was not, which will matter in any arm where `cv` is not already the winner.

### 2. ⛔ "A LANE_KEEP decode is not a hold goal" — the 86.5 % figure, corrected twice and now explained

`COST_METRICS` and my brief put the affected population at **86.5 %** of the grid. An earlier pass
corrected it to ~20 %. MEASURED here, with the mechanism: **the goal IS the hold field on 25.0 % of
windows.** LANE_KEEP is decoded on **45.0 %**, but on most of those **the LON token still commands
non-zero acceleration**, so `g ≠ z_ref` and the branch **correctly does not fire**.

⇒ **A decode is not a goal state.** The lateral token being `LANE_KEEP` says nothing about whether
the *goal vector* equals the hold field, because the longitudinal token is still free. ⚠️ Same
family as the units and grid traps: a population counted on the wrong predicate.

### 3. ⛔⛔ A PROCESS COUNT THAT COUNTS PARENTS AND CHILDREN — it halved a queue, and it inflated MY reports

MEASURED: the queue lanes gated on *"live arm processes ≤ 1"*, but **one arm is a parent and its
child — two `python.exe` entries, BOTH carrying the full command line** — so the gate **could never
open while any arm ran**, and each lane's log looked like a perfectly normal wait. Fixed by gating
on ≤ 2 under **new filenames** (⚠️ never edited in place: bash reads a running script lazily by byte
offset).

⛔ **And I made the same error in every fleet report this evening.** I quoted *"8 arms"*, *"11
procs"*, *"20 compute procs"* from a raw process count. Re-measured just now: **10 processes, 2
distinct `--out` targets**, with the tree running **2–4 deep** (`22708 → 13332 → 24904 → 23444`).
⇒ **I over-reported concurrency by 2.5–5×.**

⭐ **The rule:** count the **ARTIFACT**, not the process — distinct `--out` targets, distinct dump
dirs, distinct records. A process count is a claim about the scheduler; an artifact count is a claim
about the work. ⚠️ Third member of this family today, after the waiter that watched the wrong
directory and the `k = 1` identity control that was never in its own tuple — **all three are a
monitor measuring something adjacent to what it claims.**

### 4. Lever scoreboard

| lever | status | changed | did NOT change |
|---|---|---|---|
| **L1 `W_KAPPA`** | **LANDED, large** | ADE −33 %, whole lateral family (all ≥ 2.3× the seed floor), straight-window damage 5.5× smaller | longitudinal (separably worse), tactical decisions (+0.0000) |
| L1b shipped weights | LANDED | everything → bit-identical `ha0`, reached **by search** | did not make it drive |
| **L2 `ccosh`** | **LANDED, NULL** | the cost on 25 % of windows | ⛔ **not one plan**, structurally |
| L3 seed ladder | queued | its own P1 already refuted — the search *can* modulate curvature | — |
| **L4 Kamm cap** | **running** (18:33:59Z) | audit: 29.6–42.1 % of plans outside μ = 0.7, peak **3.262 g** | — |
| L5 retrain head | not taken, correctly last | — | a perfect goal is **worse**, −1.2181 separated |

⇒ **Verdict unchanged: refav1 does not yet beat the trivial floors at T1.** It reaches statistical
*parity* while genuinely acting and **beats them on the GT-turn stratum**. ⭐ **L2's null NARROWS the
search rather than ending it** — the hold branch is not the longitudinal blocker, and the named
mechanism (`ADAPT_SPEED_FOR_CURVE`'s canonical control being `a == 0`) still stands as next.

⚠️ Nothing is blocked on another stream: `wk151`, `l3ladder`, `kamm07` and `combined` carry a
`HANDOFF.md` and a `raw/finalize.sh` sufficient to land them **without re-deriving anything** — the
stranding rule applied prospectively rather than after the fact.

## M29. ⛔ THE RL POST-TRAINING CAMPAIGN FAILED. The win that came out of that stream contains NO RL.

**The PI asked directly: *"Are you saying the RL postrain campaign was successful?"* The answer is
NO, and the question exposes a framing error of mine that needed correcting.**

### What RL actually did — two arms, two committed FAILURES

| arm | committed exit | result |
|---|---|---|
| the **V2-faithful RL stage** | ADE must not regress past the replicate floor | ⛔ **FAILED** — `ade_m` **+0.0362 [+0.0261, +0.0460] separated**, **7 of 11** non-structural T1 rows quotable **regressions**, and the fan came out **LESS safe** |
| the **veto-only** arm | feasibility improves AND ADE holds | ⛔ **FAILED** — the replicate killed **four of five** metrics a single seed would have shipped; only `fan_peak_g_mean` survived, and `sel_peak_g` **WORSENED** |

⛔ **And the two RL-derived levers partially CANCEL:** `base + gate2` beats `veto + gate2` on all
three axes (0.0729 vs 0.0896 · 0.1459 vs 0.1617 · 0.4705 vs 0.4834). ⇒ **ship the gate on the
UNMODIFIED base; do not ship the veto'd checkpoint.**

### What actually produced the win — and it has zero gradient steps

* **`feasible_decode`** — a **control-space projection** that inverts the scorer's own
  finite-difference map. **No training. No reward. No policy.** ⭐ Its floor is a *bit-identical
  disabled-lever arm* rather than `ctrl0` **precisely because nothing trains**.
* **the top-2 kinematic gate** — a **re-ranking rule** over already-emitted candidates. No training,
  no new parameters, no new perception.

⇒ **The 1.2 mm result is a deterministic geometry fix, not an RL outcome.**

### ⚠️ My framing error, stated plainly

I reported M26 under the heading of the stream tasked with *"prove RL efficiency"* and let that
banner carry a non-RL result. The stream itself never claimed otherwise — it reported its RL exit as
**FAILURE** as written. **The conflation was mine, in the summary layer**, and it is the same class
as quoting an oracle ceiling as a payoff (#29): **a true number presented under a heading that
implies the wrong cause.**

### What the RL campaign WAS worth — diagnostic, and it was decisive

⭐ It is the reason we looked at the decode at all:
* ρ(reward, envelope) **−0.5367**, all terms separated ⇒ **the reward is NOT broken** — a
  disqualification branch that did not fire, which stopped us "fixing" the wrong thing;
* **`progress` is the only positive term**, and deleting it moves ρ by **0.27** while `feasibility`
  ×4 moves it by **0.001** ⇒ a 270× lever-ranking that redirected the work;
* the anchor bank is drivable at **0.48 g** while the decode emits **4.11 g** ⇒ **the infeasibility
  is manufactured DOWNSTREAM**, which is the sentence the projection was built from.

⇒ **RULE ZERO in its exact intended form: the campaign was refuted, it did not stop there, and its
successor — which is not RL — is the result.** ⛔ But *"the RL post-training campaign succeeded"* is
**false**, and no report may say it.

## M30. The `W_KAPPA` ladder is CLOSED at an interior optimum — and the rung DEFINITION is what makes it evidence

### 1. The ladder, T1, 40 windows

| arm (metric, `W_KAPPA`) | ADE | curv MAE | heading | TAC lat kappa | turn recall L / R |
|---|---|---|---|---|---|
| `cos` + 0 | 1.8944 | 0.080478 | 29.5086 | **−0.1249** *(below chance)* | 0.0 / 0.25 |
| `ccos` + 0 | 1.3272 | 0.055369 | 23.4578 | 0.3795 | 0.3636 / 0.75 |
| ⭐ **`ccos` + 15.11 (10 %)** | **0.8934** | **0.030982** | **15.2704** | 0.2611 | 0.0 / 0.5 |
| `ccos` + 151.12 (100 %) | 0.9084 | 0.038019 | 15.3785 | **0.0000** | **0.0 / 0.0** |
| `cos` + SHIPPED 0.05 | 0.9251 | 0.040083 | 20.1374 | **0.0000** | 0.0 / 0.0 |
| `ha0_ext` floor | **0.8772** | 0.077298 | 27.7357 | 0.6277 | — |

⇒ **An interior optimum at the 10 % rung, with BOTH neighbours worse.** The ladder is **exhausted as
a lever**.

⭐ **The collapse is monotone and comes from the arm tool's OWN trivial-profile probe** (not a
re-implementation): plans bit-`identical_to ha0` climb **3/40 → 11/40 → 10/40 → 17/40 → 40/40**, and
`CONSTANT-VELOCITY` **0.0750 → 0.2750 → 0.2500 → 0.4250 → 1.0000**.

### 2. ⭐⭐ The rung DEFINITION is the methodological result

**"100 %" was fixed from banked data BEFORE any arm ran** — defined as *the curvature charge equalling
the whole goal decision*. And that is **precisely where both turn recalls hit 0.0 and the arm becomes
the do-nothing plan.** ⇒ The scale **predicted the collapse point**, which is what makes the interior
optimum evidence rather than a lucky pick.

⚠️ **A sweep chosen by eye had no way to know the interesting range was `0 < W_KAPPA < 151`** — the
shipped 0.05 and the "obvious" large values both sit in the dead zone. Same family as `M15`'s ruling
that the *metric* decides: here the **parameterisation** decides, and defining it from the physics
before looking is what kept the sweep honest.

### 3. ⚠️ A CAVEAT ON A CLAIM I HAVE BEEN REPORTING

I have been telling the PI that `wk15` **"beats the floors on turns."** That is true **on ADE over
the GT-turn stratum** (0.9699 vs 1.1521) and it stands. ⛔ **But the turn RECALL is asymmetric and
thin:** **0.5 right (n = 8) vs 0.0 LEFT (n = 11)**, where `W_KAPPA = 0` reads **0.75 / 0.3636**.
⇒ **the penalty costs LEFT turns first.**

Against the measured tactical seed floor (**0.0750 absolute**) an 11-window recall is **far too thin
to call**. ⇒ **`wk15` must not be quoted as a lateral FIX until a wider panel exists** — ADE on a
stratum and per-direction recall are different claims, and I had been letting the first carry the
second.

### 4. Two defects introduced and caught before they cost GPU

* `l3ladder` / `combined` died at startup on an **`UnboundLocalError`**: `inspect` is imported
  *inside* the `--goal-kappa-turn` branch and the new seed-ladder block referenced it from a sibling
  branch. ⭐ **Zero GPU wasted** — the tool raised in `run_dump` **before the rollout**, which is
  exactly what its preflight-import design exists for (the 2026-08-11 lesson, working).
* A **bare `%`** in the `--kamm-mu` help text broke **`--help` itself** — argparse `%`-formats help
  strings, and the file's own convention is `100 %%`. Invisible to every running arm; it surfaces
  only when an operator asks for help.

⇒ **Class logged: a conditional import is not a module import, and a help string is CODE.**

### 5. Standing

⛔ **refav1 does not beat the trivial floors at T1.** It reaches **parity** at the ladder's interior
optimum while genuinely acting, and wins the turn stratum on ADE. **The lateral side is now
exhausted**; the blocker is the **LONGITUDINAL family**, which no arm in this package addresses and
which is the subject of the successor brief. `kamm07` (L4) at 3/8, `l3ladder` running, `combined`
queued, with `HANDOFF.md` + `raw/finalize.sh` sufficient to land them.

## M31. ⭐⭐⭐ CONSTRAINT BEATS PENALTY — measured, and it dissolves the left-turn asymmetry

**`DESIGN_CONSTRAIN_BY_CONSTRUCTION.md` was written as a hypothesis and confirmed within the hour by
an arm that was already running when I wrote it.** The prediction was: *"a penalty strong enough to
prevent bad behaviour also prevents GOOD behaviour; a projection deletes the infeasible turn and
leaves the feasible one untouched."*

### 1. The safety axis — the only lever in the package that moves it

`assert_feasible` at `v0 ≥ 2 m/s`, n = 27; the **ground-truth control reads `envelope 0.0000 /
kamm_over 0.0000`**, so the block is admissible:

| | `ccos_argmax` | **`kamm07`** (constraint) | ground truth |
|---|---|---|---|
| `kamm_over_rate` (μ = 0.7) | 0.2963 | **0.1481** | 0.0000 |
| `peak_g` mean | 0.558 | **0.256** | 0.178 |
| **`peak_g` max** | **3.262** | ⭐ **0.707 — equal to μ to tolerance** | 0.373 |
| `max |κ|` | 0.2000 *(the clip)* | **0.174670** *(below it)* | 0.1701 |

⭐ **`peak_g` max lands exactly on μ.** A constraint with units binds where its physics says it
should — that is the signature of a constraint rather than a tuned penalty.

### 2. ⭐⭐ It is FREE on the longitudinal family, where the penalty is separably worse

| metric | `kamm07 − base` | `wk15 − base` (penalty) |
|---|---|---|
| `LON speed_mae` | **−0.0017 — inside its 0.0038 floor** | **+0.0764 WORSE**, 20× floor, separated |
| `LON accel_mae` | **+0.0013 — inside its 0.0061 floor** | **+0.0874 WORSE**, 14× floor, separated |
| `LAT yaw_rate` | −0.0676 [−0.1476, −0.0038] sep., 8.3× | −0.1571, 19×, sep. |
| `ade_m` | −0.3344 [−0.9536, +0.0625] not sep. | **−0.4338** sep., 7.1× |

### 3. ⭐⭐⭐ IT DISSOLVES THE LEFT/RIGHT ASYMMETRY — the asymmetry belongs to the PENALTY

The PI asked for a wider panel to settle whether `wk15`'s left-turn collapse is real. **A different
arm answered it structurally:**

| | base | **`kamm07`** | `wk15` |
|---|---|---|---|
| TAC lat kappa | 0.3795 | **0.3644** *(inside the floor)* | 0.2611 |
| **`turn_left` recall** | 0.3636 | ⭐ **0.3636 — IDENTICAL** | ⛔ **0.0000** |
| **GT-turn ADE** | 0.9195 | ⭐ **0.9195 — IDENTICAL** | 0.9699 |

⇒ **The constraint removes the curvature the tyre cannot deliver and leaves every turn DECISION
untouched, bit-identically.** The penalty charges *every* curvature and pays for its accuracy by
suppressing turning in general — left first.

⇒ **The left-turn asymmetry is a property of the PENALTY, not of the corpus, the vocabulary, or the
head.** ⚠️ The wider panel still runs — it settles whether the *penalty's* asymmetry is real at n —
but the **fix is already identified and does not depend on that answer.**

### 4. The residual, stated as a scope difference rather than a failure

`kamm_over_rate` is **0.1481**, not 0. ⚠️ **The cap and the checker are not the same object:** the
cap acts on the candidate's controls at plan time using its **own accel-integrated speed**, while
`assert_feasible` re-derives `κ = a_lat / v_mid²` from the **rolled-out path** with `v_mid` floored at
0.5 m/s. ⇒ closing that is a **work item with a named mechanism**, not an unexplained gap — and it is
the same *"price the artifact the CONSUMER opens"* rule that has bitten this programme before.

### 5. What this changes

* ⭐ **`combined` (ladder + cap) is now clearly the right final arm** — their straight-window gains
  differ (+0.1957 vs +0.4304 against `ha0_ext`), so they attack **different halves of the same 21
  windows** and should compose rather than cancel. Queued behind `l3ladder` (7/8).
* ⭐ **The design principle now has a second independent confirmation** — the friction *projection*
  on refcv3 (96.87 % of the gap at 1.2 mm) and the friction *cap* on refav1 (peak_g 3.262 → 0.707,
  free on longitudinal, tactical decisions bit-identical). **Two arms, two codebases, same shape.**
* ⇒ **The collision projection now running is the third instance of the same design**, and its
  prediction is sharper because of this: it should remove colliding candidates **without moving the
  non-colliding ones at all** — which is exactly control #3 of its SPEC.

⚠️ **Standing, unchanged and honestly scoped:** refav1 does not yet beat the floors overall — parity
at the ladder optimum, a win on turns, and the **longitudinal family** still the blocker. ⭐ But that
blocker is now the *only* one, it has a named mechanism (`ADAPT_SPEED_FOR_CURVE`'s canonical control
is `a == 0`), and the lever that fixes it is predicted by this same principle: **representability,
not penalty.**

## M32. ⛔⛔ `v7f` HAS NEVER BEEN TRAINED — and it is not blocked. It is UNATTENDED.

**The PI observed that a whole day of planning omitted the flagship. The audit says the omission is
older and larger than one day: the flagship has never run at all.**

### 1. The finding, on three independent probes each with a same-breath control

* `grep -c -i v7f MODEL_REGISTRY.md` = **0** — controls `flagship` **238**, `refcv3` **31**, so the
  probe reads. ⚠️ **All 30 `v7` hits in the registry are the `v7.2` LABEL VOCABULARY**, not the model.
* `V7_LAUNCH_GATE.md` carries a **binding PI directive of 2026-08-31**: *"we should not start
  training of v7 until the remaining problems are solved."*
* `PREREG_V7F` §9's launch line still holds **unfilled placeholders**.

⇒ **No checkpoint, no run directory, no registry row.** ⭐ Everything the programme has been calling
"the v7 result" is **v7-tiny** — ~19 M-param proxies at 2k–30k steps **with every planner objective
at zero**. ⇒ The flagship section having no entry newer than 2026-08-09 is **not a banking failure.
Nothing ran.**

### 2. ⭐ Three of the four "blockers" are not blockers

| row | verdict |
|---|---|
| **`D-CORPUS-B1`** | ✅ **NOT a blocker.** Built twice, byte-verified, **4,572 + 141 = 4,713 exactly**, and a **completed 40,284-step refcv3 run already trained on it**. `B1_TRAINING_PREP.md` is **stale by ~6 days** and is the sole source of the "waiting on Thor" premise. |
| **`D-EMA-ADOPT`** | ⛔ UNMET — the arm never ran; the instrument is fully built (24 tests). ⚠️ **It is a τ-ramp (EMA decay), not "λ"** — my brief said λ and was wrong. |
| **`D-V7-TRUNK-ANCHOR`** | ✅ engineering done · ⛔ unexercisable, and **worse than recorded**: the seed on the box is **ViT-L/16**, which `build_trunk_anchor` **refuses BY DESIGN** against the ViT-B/16 trunk. |
| **`E-DEC-9b`** | ⛔ OPEN. The switched-off term is **O7 distillation** (`--w-o7-distill 0`, verified at source). ⭐ **But turning it on is measured NEGATIVE in v7f's form** — **−0.2553 co-trained vs +0.3274 pure**, with LayerLock showing both weight schedules collapsing. ⇒ **the remedy is staging ORDER, not a weight sweep.** |

⚠️ **And the compute premise is stale too:** *"Thor is the only compute and it is committed"* —
`refav1-b1-v72-ep3-speed` **completed 2026-09-04**, and **Thor holds the B1 epcache**. ⇒ **v7f is
blocked by neither data nor (probably) GPU.**

### 3. ⭐ The launch line is TWO DEFECTS from clean — and one "blocker" was false

The stream **ran the launch line** rather than reading about it. Exactly two real defects: the LDAD
triple does not exist (confirmed two ways — grep **and** real argparse), and — **on nobody's list,
mine included** — **`--horizons` is never passed**, so it defaults to `(1,2,4)`, **which the trainer
refuses**.

⇒ **One loss term and one flag from argparse-clean.**

⭐ **And it refuted a false blocker that would have cost a day:** `--enc-init-from` **is**
implemented, as a registered alias at `:8759` — the claim came from a **docstring 832 lines
earlier**. *A docstring is not the code it sits above.*

### 4. ⛔ The uncomfortable transfer answer — and the one thing that DOES transfer

**None** of today's four levers transfer to v7f: `feasible_decode`, `kamm_mu`, the top-2 gate and
`W_KAPPA` are **all decoder-side levers over a control vocabulary v7f does not have**
(`train_v6_staged.py` scores 0 for `feasible` / `kamm` / `anchor_meta`).

⭐⭐ **What transfers is the PARAMETERISATION, and it is the same sentence as `M31`.** refcv4b's
`(a_lon, a_lat)` is a **command channel with units**; v7f's `omega_accel_v` **has no units — it is
realised motion played back**. `M31` measured that *"a constraint with units binds exactly where its
physics says it should"* — the Kamm cap landed `peak_g` max on **μ to tolerance** precisely because
its channel has units. **v7f cannot be given such a constraint until its command has units.**

⇒ That turns `V7_LAUNCH_GATE` **P2(b)** from *"never tested"* into a **runnable arm**.

### 5. The single next thing, and the decision it needs

⭐ **Give v7f a real command — `(a_lon, a_lat)` — instead of `omega_accel_v`.** It is the only
untested **P2** cause with an instrument already in hand, it is the **precondition for the
counterfactual-target arm**, and it attacks **P1** — *no v7 arm has ever beaten its own hold-action
control* — rather than the shipping configuration.

⚠️ **The gating experiment is NOT the τ-ramp.** `D-EMA-ADOPT` (08-29) gates how v7f **ships**; the
PI's directive (08-31, **two days later**) gates whether it **trains**. Those are different gates and
the later one governs.

⛔ **PI DECISION REQUIRED, and it is now a real choice rather than a hypothetical:** the 08-31
directive says *do not start v7 training until the remaining problems are solved.* The audit shows
the remaining problems are **two argparse defects, one unit-less command channel, and one staging
question** — not a corpus, not compute. ⇒ **Either the directive is lifted for a scoped
`(a_lon, a_lat)` arm, or it stands and v7f waits.** Both are defensible; what is not defensible is
the status quo, in which nobody is working on the flagship and nobody decided that.

## M33. ⛔ CORRECTION — the τ-ramp arm DID run. `D-EMA-ADOPT`'s condition is MET.

**I told the PI, one message ago, that `D-EMA-ADOPT`'s conditional arm had never run. That is
FALSE.** A fourth probe — which the v7f audit predicted *"can only corroborate"* — refuted it, and I
then settled it **at source** rather than by document triangulation.

### 1. Read from Thor's own run directory, not from a document

`thor:/home/nvidia/v7tiny/emao14_30k_tauramp/` holds `ckpt.pt` (144,111,717 B), `config.json`,
`metrics.json` (415,493 B), `stage_gate.json`, `summary.json`, `train_log.jsonl` (1,609,183 B).
Its `config.json` reads:

| field | value |
|---|---|
| `ema_decay_ramp` | ⭐ **`"cosine"`** |
| `o5_target` | `"ema"` |
| `ema_decay` | 0.996 |
| `steps` | 30000 |
| `summary.done` | **True** |

⇒ **The arm ran with the ramp on, to 30,000 steps, and completed 2026-08-30 ~06:1x UTC.**
⇒ **`D-EMA-ADOPT`'s stated condition is MET, and `D-V7F-TAU-RAMP-UNRUN` is RETRACTED.**

### 2. ⭐⭐ Why the negative was believable — and it is a systematic auditing hazard

`BINDING_TRAINING_IMPROVEMENTS.md` (B3): **`--save-every` OVERWRITES `ckpt.pt`**, and the banked 30k
run dirs *"contain exactly ONE file each."* The run directory lived **only on Thor**; the **config
was never banked.**

⇒ **A `config.json`-based audit SYSTEMATICALLY UNDER-REPORTS WHICH ARMS RAN.** Searching all **95**
banked `config.json` files for `--ema-decay-ramp` returns nothing **even though the run happened** —
the corpus being searched structurally cannot contain the evidence.

⛔ **And the register row was stamped `MEASURED (source + git)` while its evidence column cited only
the INSTRUMENT being built** (`train_v6_staged.py:9300`, the 24 tests). **It cites no probe that
searched for run artifacts.** ⇒ **an unverified negative wearing a MEASURED stamp** — the most
expensive kind, because the stamp is what stops the next reader checking.

⭐ Same family as the day's other absence errors (`ls-tree` truncation, the `.claude/worktrees`
non-hydration, the comma-formatted numbers), with a new and general form worth naming:
**when an artifact class is known to be overwritten or unbanked, its absence is not evidence — and
the audit must say which corpus it searched and whether that corpus could contain the answer.**

### 3. ⛔ What STANDS, because they are different claims

`D-V7F-NEVER-TRAINED` is **unaffected and still MEASURED**: every arm found — `emao14_30k_tauramp`,
`emao14_30k`, `o14fut30k`, the P0 2k bake-off, MM-E19 K60/K8 — is **v7-tiny (~19 M params, planner
objectives at zero)**. ⇒ **The v7f FLAGSHIP has still never been trained.**

⇒ Two claims were bundled and only one was wrong: *"the τ-ramp never ran"* (**false**) and *"the
flagship never trained"* (**true**). ⚠️ I stated them in one breath, which is how the correct one
lent credibility to the incorrect one.

### 4. ⭐ And the plan is UNCHANGED — for a better reason than before

I wrote that *"the gating experiment is NOT the τ-ramp."* That was right, and is now **better
supported**: the τ-ramp is not the gate **because it already ran and cleared**, not because it was
pending. ⇒ **The gate remains P1 — no v7 arm has ever beaten its own hold-action control** — and the
single next thing is still **giving v7f a command channel with units (`a_lon, a_lat`) instead of
`omega_accel_v`.**

### 5. Closed at source, so it cannot recur for this arm

`config.json`, `summary.json` and `stage_gate.json` are **pulled from Thor and banked** into
`…/2026-08-30-t1-first-v7-read/raw/`. ⚠️ **The `MODEL_REGISTRY.md:4156` raws it cites
(`tau_drift.json`, `tau_nrmse.json`, `tau_absorb.json`) still do not exist in the repo** — confirmed
by two independent methods. That is a remaining provenance gap on a **completed** arm, and it is the
same stranding that produced this correction.

## M34. ⭐⭐ THE SEED POOL WAS NEVER THE CONSTRAINT — and the design principle needs a THIRD part

### 1. `l3ladder`: a null so clean it identifies the mechanism

`ccos` + `(0, 0, 64.297)` + **10 extra iteration-0 candidates at R 500/200/100/50/25 m** — the
candidate set is the only variable.

| | `ccos_argmax` | `l3ladder` | |
|---|---|---|---|
| ADE | 1.3272 | **1.3236** | Δ **0.0036 = 17× BELOW** the 0.0607 seed floor |
| EXACTLY-constant series | 0.7750 | **0.7750** | unchanged |
| turn recall L / R | 0.3636 / 0.75 | **0.3636 / 0.75** | identical |

⭐ **The decisive row is the histogram — NOT ONE WINDOW OF 40 REALISES A RUNG:**

```
l3ladder   0.0000 x10   0.0800 x21   then 0.1009 0.1189 0.1253 ... all ABOVE 0.10
wk15       0.0000 x18   0.0800 x9    then 0.0115 0.0120 0.0140 ... 0.0530
```

The rungs sit at **0.002–0.04**. `l3ladder`'s extra mass is **entirely above 0.10**, and its
`0.0000 ×10 / 0.0800 ×21` spine is **bit-for-bit the uncapped arm's**.

### 2. ⭐⭐⭐ The answer is complete because it comes from BOTH directions

* **`wk15` produced curvatures 0.0115–0.0530 — exactly the band the rungs occupy — with NO LADDER AT
  ALL**, purely because the penalty gave the search a *reason to prefer* them.
* **`l3ladder` hands the search those very magnitudes and it never picks one**, because with
  `W_KAPPA = 0` the cost is **indifferent** and the goal-aligned canonical seed wins.

⇒ **`D-REFAV1-DRIVE-GATE`'s gate is NOT a wall around the reachable set. It is an ABSENCE OF
PREFERENCE.** Confirmed by supplying the candidates without the preference (nothing happens) and by
supplying the preference without the candidates (the search synthesises them).

### 3. ⛔ THE DESIGN PRINCIPLE WAS TWO-PART AND IT NEEDS THREE

`DESIGN_CONSTRAIN_BY_CONSTRUCTION.md` said: *make the bad unrepresentable and the good
representable.* **That is incomplete.** The corrected form, with today's evidence attached to each
part:

| part | mechanism | evidence |
|---|---|---|
| **1. make the bad UNREPRESENTABLE** | constraint with units | Kamm cap: `peak_g` max **3.262 → 0.707 = μ**, free on longitudinal, turn decisions **bit-identical**; friction projection: **96.87 %** of the gap at **1.2 mm** |
| **2. make the good REPRESENTABLE** | vocabulary / candidate set | ⛔ **alone it does NOTHING**: L=3 realised **2.3 %** (no chooser); `l3ladder` realised **0 of 40 windows** (no preference) |
| **3. ⭐ GIVE THE SEARCH A REASON TO PREFER IT** | cost / preference | `W_KAPPA` alone produced **exactly the rungs' band without the rungs** |

⭐ **And the sharpest consequence: (3) can SUBSTITUTE for (2) whenever the search can synthesise what
it needs.** A wider vocabulary is only worth its cost when the search **cannot reach** the band — a
test that is cheap to run and that we had never run.

⚠️ **The ladder is also not free**, exactly as its own docstring warned *before* the arm ran (*"with
`W_KAPPA = 0` a wider curvature set can only add ways to be wrong"*): `lane_keep` recall
**0.7143 → 0.5714** (~2× the floor) and goal FDE **2.9639 → 3.3200**.

⇒ **The informative combination is ladder + `W_KAPPA`, never ladder alone — and never ladder + a
CONSTRAINT**, because *a cap can forbid a curvature but cannot make a rung attractive*. That is part
3 of the principle stated as an operational rule. **`wk15_ladder`** (one variable against `wk15`)
started 19:13:04Z; `combined` (ladder + cap) is at 2/8 and is now known to be the **less** informative
pairing.

### 4. Where refav1 stands: three of four levers answered

| lever | verdict |
|---|---|
| `W_KAPPA` | **the ACCURACY lever**, interior optimum, ADE −33 %, costs the longitudinal family and turn symmetry |
| Kamm cap | **the SAFETY lever**, `peak_g` max → μ, **free** on longitudinal, decisions untouched |
| hold branch (`ccosh`) | **NULL** — the CEM's best sample was already the all-zero plan |
| seed pool (`l3ladder`) | **NULL** — an absence of preference, not of candidates |

⇒ **The LONGITUDINAL family remains the blocker**, and the three-part principle now predicts its fix
precisely: `ADAPT_SPEED_FOR_CURVE`'s canonical control is `a == 0`, so part **2** is missing *and*
part **3** has nothing to prefer. ⛔ **Fixing either alone will null**, exactly as the lateral side
just demonstrated twice.

⚠️ **Process note worth keeping:** two commits were **correctly REFUSED** by `mktree_commit.py`'s
compare-and-swap when a sibling moved HEAD mid-build. Reducing the commit to only the **29 changed
paths of 225** shrank the build window and it won immediately. ⇒ **the fix for a lost CAS race is to
commit what CHANGED, not to force.**
