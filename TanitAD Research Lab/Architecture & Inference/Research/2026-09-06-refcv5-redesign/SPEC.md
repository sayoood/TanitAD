# SPEC — `refcv5-cap`: the CAPABILITY arm the PI ordered

**Stream:** Architecture & Inference · **2026-09-06** · branch `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv5-redesign/`
**Supersedes:** the arm `refcv5-ddim-b1-v72-40k` (stopped by the PI; **no result exists**)
**Companion:** `REVIEW.md` in this package — the element-by-element audit the PI asked for.
**Validated with:** `TanitAD_ValidateAIDesign`.

> ⛔⛔ **NO NUMBER IN THIS DOCUMENT IS A refcv5 RESULT.** refcv5 produced none. Every
> quantity is either a **committed bar** (written before the data exists — the only moment
> a pre-registration is worth anything) or a **MEASURED** quantity from another arm,
> carrying its artifact and tier inline.

---

## 0. THE SCHEMA BLOCK

```yaml
hypothesis: H-REFCV5-CAP-1          # NEW — registered in GOALS_AND_CLAIMS.md this turn
                                    # (extends H-REFCV5-DDIM-1, which this arm subsumes)
arm: refcv5-cap-b1-v72-40k
one_variable: CAPABILITY-BUNDLE     # ⛔ DELIBERATELY NOT ONE — see §1. The PI has ruled
                                    # for capability; attribution is bought by CONSTRUCTION
                                    # (§1.2) and discharged by the ladder in §6.
held_constant: [corpus, labels, anchors, seed, steps, batch, lr, warmup, window, horizons]
success: "P1 — `cap - ha0_ext` on ade_m is NEGATIVE, paired episode-cluster CI excludes
          zero, and |delta| > F(ade_m), the INFERENCE-seed floor from the replicate."
failure: "F1 NULL — `cap - refcv4b os` not separated, or |delta| <= F(ade_m).
          F2 REGRESSION — `cap - refcv4b os` separated POSITIVE."
controls: [constant_only, raw_input_floor, deliberate_regression, replicate]
splits: {fit: "B1 train 4,572 clips", val: "carved from FIT only",
         test: "B1 eval 141 clips / 4,823 windows — scored, never tuned on"}
```

---

## 1. ⛔ THE ATTRIBUTION TENSION — resolved by CONSTRUCTION, not by scoping down

**The tension is real and must not be waved away.** `TanitAD_ValidateAIDesign` refuses a
run whose arms differ in more than `one_variable`. This arm deliberately moves several.

### 1.1 Why scoping down is the wrong resolution here

⛔ **The one-lever arm was already tried and the PI stopped it.** `refcv5-ddim-b1-v72-40k`
was refcv4b + `--sampler ddim --w-u0 0.5`, and three MEASURED facts say it could not have
delivered a capability:

1. ⛔ **`D-REFCV5-PLAN-10` — THE DEFICIT IS IN THE FAN, NOT IN SELECTION AND NOT IN
   ROUTING.** Arithmetic on banked numbers (n = 4,823 windows / 141 episodes, **scoped to
   refcv3-40284**, because `oracle_sel` is instrument-invalid on refcv4b —
   `D-REFCV4B-ASTAR-GEOMETRY`): `oracle_sel − os` = **−0.0751 [−0.0884, −0.0618]** and
   `os − os_navzero` = **−0.0239 [−0.0428, −0.0089]** ⇒ **perfect selection PLUS the route
   input is 0.099 m against a 0.1423 m gap to `ha`.** ⇒ *Any* arm built only on selection
   or nav wiring is, **by arithmetic**, incapable of clearing the bar.
2. ⛔ **The DDIM sampler has no measured effect size on driving.** It is not on
   `D-REFCV5-LEVERS` — the programme's own list of validated improvements *ranked by
   measured effect size* — at all.
3. ⛔ **A one-seed separated CI would not have settled it anyway.** `H-ESTIM-SEED-1`:
   **6 of 42 cells = 14.3 %** read "separated" between two arms differing in **nothing**.
4. ⛔⛔ **AND DD'S OWN PUBLISHED ABLATIONS SAY THE SAMPLER IS THE CHEAP HALF.**
   PUBLISHED (2411.15139v3, banked `2411.15139`): removing the **waypoint-indexed spatial
   cross-attention** costs **88.1 → 55.1 PDMS (−33.0)**; dropping from **2 denoising steps
   to 1** costs **88.1 → 87.9 (−0.2)**. ⇒ refcv5 spent ~45 h on the component DD prices at
   **≈0.2** and omitted the one DD prices at **33**. ⚠️ Those are *DD's* magnitudes on
   *NAVSIM with LiDAR* and **do not transfer** to our vision-only surface — **the RANKING
   is what transfers**, and it agrees with our own independent ranking
   (`D-REFCV5-LEVERS`, `D-REFCV5-PLAN-10`).

### 1.2 ⭐ THE RESOLUTION: attribution is bought at CONFIG TIME, not by running less

Three mechanisms, all already in this trainer:

1. ⭐ **Every element stamps its flag AND its effective weight into `config.json`.** The
   preflight row *"every knob reaches `config.json`"* passed at **18 knobs, 0 missing**
   with a control `n_knobs > 12`. A post-hoc ablation is therefore **exact** — it reads
   the stamp, not a memory of the launch line.
2. ⭐ **Every added element is behind a ZERO-INIT GATE**, so the arm is bit-identical to
   refcv4b at step 0. An element that never learns to fire is *visible as a dead gate*,
   which is the M18 defect check, not a confound.
3. ⭐ **The ablation ladder is named in advance (§6), ranked by measured effect size, and
   every rung is an EVAL-TIME switch on ONE checkpoint** ⇒ training variance (V2) does not
   enter, and the rungs cost minutes, not GPU-days.

### 1.3 ⛔ WHAT THE HEADLINE CLAIM IS, STATED PLAINLY AND IN ADVANCE

> **The headline arm is a CAPABILITY claim: "this bundle drives better than the strongest
> model-free control." It is NOT a per-element attribution claim, and no sentence in the
> RESULT may read as one. Per-element attribution comes ONLY from the named ablations in
> §6, and until those run, no element in this arm may be quoted as "worth X".**

⚠️ **`TanitAD_ValidateAIDesign` §2 also says "never validate a design on a full-scale
run".** ⛔ **DEVIATION, DECLARED (as `B4` was for refcv5):** this is a full-scale run. Its
consequence is committed here — **a NULL is a statement about THIS ARM AT THIS BUDGET, not
about any element in it.** ⭐ Mitigation that is *not* a promise: §7.0 GATE 0 runs the
wiring proofs on the tiny/CPU rig **before a GPU second**, and every one is a
**mutation** proof, not an inspection (`D-STRAT-BYPASS-WIRED-1` is the precedent — the
first version of that probe passed while never looking at the seam it was proving).

---

## 2. ⛔ THE FIFTEEN ITEMS — every one IN (flag + weight) or OUT (named blocker)

⛔ **No item is silently absent.** Flags verified **MEASURED** against the trainer's own
argparse at HEAD (83 flags; control: `add_argument` count = 84, non-zero).

### A · FROM THE PAPERS

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **1** | **Truncated/anchored diffusion policy (DD v1)** | ⭐ **IN — CARRIED FORWARD UNCHANGED** | `--sampler ddim --sampler-space control --sampler-infer-t 8 --sampler-steps 2 --sampler-train-t-max 50 --sampler-groups 1 --w-u0 0.5` |

⭐ **Why carried, not dropped:** it was MEASURED **live, not merely stamped** (`u0`
0.27576 / 0.18663 / 0.32519 at steps 50/100/150 — the M18 dead-flag check passed) and it
costs **+5.3 %** wall-clock (4.046 vs refcv4b's 3.844 s/step). It is also **V2's
prerequisite** (`D-REFC-DDAUDIT-6`: *"the prerequisite — a stochastic denoising policy with
a log-probability — is absent"*), so dropping it would block item 3 permanently.
⚠️ **THREE honest qualifications, all carried into the RESULT:** (a) it runs in **CONTROL
space** where DD runs in **waypoint** space (`control_norm [4.0, 3.0]`) — a defensible
deviation, since our vocabulary is kinematic, but it must never be called "DD-faithful"
without it; (b) ⛔ `D-REFCV5-DIFFUSERS-PIN-CANNOT-PASS` — `assert_matches_diffusers` **has
never certified anything**, so the schedule math is an **unverified re-implementation**;
(c) ⛔ **`--w-u0` IS NOT A DD ELEMENT.** MEASURED from `hustvl/DiffusionDrive@9b52ed0`:
DD-v1 has **no ε-prediction and no denoising MSE term at all** — `diff_loss_weight = 20.0`
is **dead code**, the model never emits `diffusion_loss`, and the only trajectory
supervision is matched-anchor L1 + focal scoring. ⇒ our x₀ loss is **BEYOND DD**, and the
RESULT must not present it as a port of one.
⇒ **GATE 0 (§7.0) makes that pin PASS or the arm does not launch.**

⛔⛔ **AND TWO SAMPLER FLAGS ARE INERT — MEASURED THIS TURN, AND THE FIRST IS A DD-FIDELITY
DEFECT, NOT A COSMETIC ONE.**

| flag | defect | consequence |
|---|---|---|
| **`--sampler-train-t-max`** | ⛔ **HAS NO CONSUMER.** Whole-tree byte probe with the readability gap closed to zero: `refc.py` ×1 (**the dataclass field only**), `refc_v3_train.py` ×2 (assignment + argparse), **and nothing reads it.** `_sample` has **no `self.training` branch** — it uses `sampler_infer_t` at train **and** eval | ⛔ **DD's `t ~ U[0, 50)` TRAINING DRAW IS NOT IMPLEMENTED.** We train and infer at the *same* `t = 8`. DD deliberately trains across `t ∈ [0, 49]` and infers from a much cleaner `t = 8`; the denoiser we train has therefore **never seen the noise levels DD's is trained on** |
| **`--sampler-steps`** | ⛔ **SHADOWED** — `k = int(steps) if int(steps) > 0 else int(cfg.sampler_steps)`, and the trainer passes `steps` from `diffusion_steps` | the CLI value is reachable **only when `steps == 0`**. Both are `2` today, so behaviour is unchanged — but the flag is **inert**, and a ladder rung that sweeps it would sweep nothing |

⇒ **ESCALATION E6.** ⛔ **Until E6 lands, `--sampler-train-t-max 50` on the launch line is
DECORATION and the RESULT must say so** — a stamped flag that changes nothing is the M18
defect, and this SPEC will not repeat refcv5's mistake of presenting a stamp as a mechanism.
⚠️ Related DD quirks recorded so nobody "fixes" us into them: DD noises to `t=8` but takes
its first DDIM step **as if at t=10** (an off-by-two), and its own train/infer distributions
are likewise unmatched — *DD's mismatch is deliberate; ours is an unimplemented flag.*

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **2** | ⭐ **Environment / agent conditioning (WP-6, DD #20)** | ⚠️ **IN, CONDITIONAL ON THE JOIN — the launch preflight decides, and stamps which** | `--agents head --agent-join <B1-TRAIN join> --w-agent 1.0 --agent-queries 100 --agent-rig-camera nominal --agent-w-project 0.05` |

⛔⛔ **AND `--agents oracle` IS NOT MERELY INADMISSIBLE — IT IS UNRUNNABLE FROM THIS
TRAINER.** MEASURED this turn by a call-graph trace with the readability gap closed to
zero: `agent_gt` appears in `refc.py` (9), `refc_v3.py` (10), `rl/refc_adapter.py` (7) and
four test files — and **0 times in `refc_v3_train.py`**, read successfully in the same pass
(control: it reports non-zero for `sampler_train_t_max` and `control_norm`). The trainer's
only forward is `model(frames, nav_cmd=…, v0=…, steps=…, lan=…, ego_state=…)` — **no
`agent_gt` under ANY flag combination.** ⇒ the oracle's forward-time `ValueError` fires
regardless of `--agent-join`, and there is **no pin-time guard for oracle at all**, while
the `head` guard's own remedy text *recommends oracle as the escape*. ⛔ **That remedy is
wrong and is ESCALATION E7.** ⇒ **`head` is not merely the right choice; it is the only
runnable one.**

⛔ **`--agents head`, NEVER `--agents oracle`.** MEASURED from the trainer's own help:
`head` = *"the LEARNED monocular 3D head (the deliverable arm; vision-only at inference,
obstacle.offline as TRAIN-TIME labels)"*. `H-V5A-AGT-1` states the oracle arm is
**"deliberately inadmissible as a capability claim (it feeds a label at inference,
violating I3)"** — it is a *ceiling*, run once, behind a flag a deployable config can never
set. ⇒ **A capability arm must use `head`.** This also disposes of the brief's refuted
precondition: `oracle` needs the join too, because *the oracle's tokens ARE the boxes*.

⭐⭐ **AND `head` IS THE DD-FAITHFUL CHOICE, NOT MERELY THE ADMISSIBLE ONE.** PUBLISHED,
read from `hustvl/DiffusionDrive@9b52ed0`: DD's `agents_query` comes from **the model's
OWN auxiliary detector** — 30 queries out of a 3-layer `TransformerDecoder` over BEV+status,
supervised by a **Hungarian-matched 3D box + class loss** — **not** from an external
detector or tracker, and **not** from GT boxes at inference. ⇒ **`--agents head` with
`obstacle.offline` as TRAIN-TIME labels is structurally the same construction.**
⭐ **And this is the element DD's own ablation prices highest**: removing the grounded
spatial attention costs **88.1 → 55.1 PDMS**. ⚠️ Ours grounds on **agents** (DD #20); the
**waypoint-indexed spatial** half (DD #19) still needs a metric frame we do not have, and
stays OUT — see rung **L9**.

⭐ **AND THIS IS THE ANSWER TO LEVER #1, NOT MERELY TO THE PI'S QUESTION.**
`D-REFCV5-LEVERS` rank 1 is **"encode CLOSING RATE"** — a **representation** requirement
(*"no cost re-weighting substitutes"*), because lead **position** is decodable
(**+0.4145 [+0.2018, +0.6120]** paired vs pixels, constant control **exactly +0.000000**)
while closing **rate** is a clean null (**+0.0061**). MEASURED from the trainer's own help,
`--agent-join-no-rates` says the default **computes the per-track rate finite
differences**, because *"`v_rel_x` is what the LONGITUDINAL family (closing speed, TTC) is
built from"*. ⇒ **The agent join is the closing-rate representation.** Rank 1 and the PI's
environment question are **the same work item**.

⛔ **`--agent-w-ground 0.0` — DO NOT ENABLE.** `D-RC5-GROUND-DEAD`: `ground_range_prior` is
a **TAUTOLOGY** (projects at rig `z=0`, back-projects onto `ROAD_PLANE_Z_M = 0.0` — exact
inverses), loss **2.61e-08**, parameter gradient **8.73e-11**. An arm passing it *trains,
converges, stamps the weight, and adds exactly zero.* Corroborated at refcv5's own
preflight: `agent_w_ground` **1.164e-10 DEAD** against `agent_w_project` **1.462e-03 LIVE**
(control: 1 of 2 live, so the probe was proven able to fail).
⚠️ `--agent-w-project` requires `--agent-rig-camera nominal` or the run **REFUSES at
startup** (MEASURED from the flag's own help). `nominal` = boresight-forward, no pitch,
**stamped as such** — an honest scope, not a claimed per-clip calibration.

⛔ **BLOCKER B-JOIN, and it is the ONLY thing gating this item.** The **B1 EVAL** join
exists (139/141 clips, 26,394 rows, 905,512 boxes, md5 `3ddb42ecbd3926066795a94587af2aed`,
`D-B1-AGENT-JOIN-1`). The **B1 TRAIN** join does not. Priced at **≈2.3 GB** by per-clip
HTTP range requests (505 KB/clip) — **not** the 121 GB a full-chunk pull would cost.
⚠️ **Evidence class on that price: ESTIMATED, not MEASURED** — it is `505 KB × 4,572 clips
≈ 2.31 GB`, extrapolated from a per-clip figure, and **no receipt backs it**. It is written
with an "≈" everywhere and must not harden into a MEASURED number without one.
⛔ **The only TRAIN-side join we hold is PARITY-scoped and it is the wrong corpus.**
`train2400_agents.jsonl.xz` — **433,040 records / 2,308 clips**, md5
`24cbdca8c3b23aafc2fb17e6bf99cf76` — is real and md5-verified, but
**|TRAIN2308 ∩ B1| = 193**, i.e. B1 coverage of **199 / 4,719 = 4.22 %** across every join
held. ⚠️ **The "7.92 %" in my first draft was a percentage quoted without its fraction and
is CORRECTED**; the register's `D-B1-AGENT-JOIN-1` gives numerator and denominator and is
the quotable form. ⇒ substituting it would train the seam on **~4 %** of the corpus and
manufacture *"agent tokens do not help"* from a **starved** seam. ⛔ **Do not substitute it.**
**What unblocks it:** the DataFlyWheel build (`stack/scripts/build_obstacle_join.py`),
**subject to an HF quota reading taken BEFORE the pull** (`HF quota is a hard ceiling`).
⭐ **This does not gate the launch — see §7.2's decision rule.**

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **3** | **RL post-training (DD-v2: GRPO, truncated advantage, scale-adaptive noise, PDM selector)** | ⛔ **OUT — REFERENCED, NOT BUILT** | ⛔ gated on the anchored `S6_wmr` bar; `stack/tanitad/rl/` is **sibling-owned** (`advantage.py`, `rewards.py`, `audit.py`). ⭐ This arm's job w.r.t. item 3 is to **produce its prerequisite** — a stochastic denoising policy — which item 1 does. |

### B · VALIDATED refcv4b FINDINGS

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **4** | ⭐ **Geometric goal point (E15)** | ⭐ **IN — the highest-ranked UNBLOCKED lever** | `--goal-point-inject --goal-point-w 1.0 --goal-point-t 4.0 --goal-point-geo-prior` |

⛔ **The weight MUST be passed explicitly**: `GOAL_POINT_WEIGHT_DEFAULT = 0.0` (MEASURED),
and the flag's own help says the pre-registered launch value is **1.0** and that
`--goal-point-inject` with weight 0 is **REFUSED**. ⭐ MEASURED this turn: there are **four**
refusals in `_check_goal_point_args`, not one — weight-0-with-inject; `--goal-point-geo-prior`
without `--goal-point-inject`; `--goal-point-w > 0` without `--goal-point-inject` (*"which
is the `w_agent` defect verbatim"*); and the **leak guard** on `--goal-point-t`, enforced
both in the trainer **and** structurally in `GoalPointConfig.__post_init__`.

⭐⭐ **AND THE ENCODING ALREADY CARRIES THE RANGE — MEASURED FROM SOURCE, WHICH SETTLES THE
BRIEF'S SHARPEST WARNING.** `stack/tanitad/refs/goal_point.py`: `GOAL_POINT_FEATS = 3`, and
the injected vector is **Cartesian**, not polar —
**`[clip(x/40, ±4.0), clip(y/40, ±1.0), valid]`** — where `range_norm_m = 40.0` is a
**FIXED constant**, the docstring being explicit that it is fixed *"so the ego speed never
enters the conditioning"*. ⇒ ⭐ **Range is not a separable component that could be
stripped — it is the magnitude of `x`.** The `+2.3632 m` figure is an **eval-side probe**
(`D-GOALPOINT-VALUE-IS-RANGE`, artifacts `raw/GP_LONG_AXIS_t4.json`,
`raw/GP_SELECTION_CEILING.json`), not a flag, and **no flag can produce a bearing-only arm**
— rung L3 must build it as a deliberate eval-time intervention.
⭐ `GoalPointHead` is a bare `nn.Linear(d_ctx, 2)` from **strategic context (vision only, by
construction)**, and `GoalPointConditioning` is a **zero-init** MLP whose outputs are
multiplied by the validity bit — so a window with no goal contributes **exactly zero**,
not the encoder's bias.
⭐ **MEASURED value, and its mechanism** (`D-GOALPOINT-VALUE-IS-RANGE`, n = 4,286 / 141):
at **t = 4 s** the point recovers **−0.0945 [−0.1164, −0.0757] m = 57.1 % of the
longitudinal selection ceiling** and **−0.1184 m/s** speed MAE, separated; on the lateral
axis it recovers **78.4 %** (−0.0414 of a −0.0528 ceiling). ⛔ **The value is RANGE:** the
same goal with its range stripped is separated **WORSE by +2.3632 [+2.2004, +2.5166] m**,
and on the lateral axis at a fixed arc a point and a bearing **tie exactly** (−0.0414 /
−0.0414) — an *identity*, not a measurement. ⇒ **the range is the whole marginal value,
and it is longitudinal.** The longitudinal ceiling is **3.1×** the lateral one.
⭐ **Why it replaces rather than augments the route token:** `--goal-point-inject` sets
`nav_inject = False` **by pre-registration, not as a side effect**, stamped as
`goal_point.replaces_nav_inject` — because running both would be a two-variable experiment.
This is grounded, not merely tidy: `D-CATEGORICAL-ROUTE-CEILING-IS-ZERO` measured that an
**oracle 3-way route command carries exactly as much lateral-selection information as
assuming the road goes straight** — the strongest deterministic decoder picks *straight*
for **all three classes**. ⛔ **VOCABULARY:** the nav token is a first-class **ROUTE INPUT
simulating the vehicle's nav system** (PI ruling). Never "oracle nav", never "deployment
gap". `--nav-from-v7` is **KEPT** so the loader and labels stay byte-identical.
⛔ **Admissibility, re-asserted:** the goal is **PREDICTED from vision**, its
`inference_inputs` are `pooled` conv features, `contains_situation_classifier_output:
false`, and `future_poses/future_actions → any goal node` is a **refused edge**. `--goal-point-t
4.0` is **strictly beyond the 2 s scored horizon** — the trainer enforces it, because a
goal inside the horizon *is the answer*.

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **5** | ⭐ **Decoder refinement fix (R4b)** | ⚠️ **IN AT EVAL, OUT OF TRAINING — no trainer flag exists** | ⛔ **ESCALATION E1 (§8).** MEASURED: `refine` appears **0** times in the trainer (control: `add_argument` = 84). |

⭐ **This is not a downgrade — it is how R4b was VALIDATED.** `H-R4B-1` measured it as an
**INFERENCE-TIME projection with no retraining**: take `off = emitted − anchor` (the
free-waypoint refinement) and replace it with `anchor + argmin_u ‖u − off‖² + λ‖D²u‖²` — a
**closed-form second-difference ridge on the OFFSET ONLY, anchor untouched**. Not a blend of
two paths, not a clamp. **λ = 3 chosen on a disjoint 71-episode / 2,431-window half** by a
rule stated before scoring, then scored on a held-out 70-episode / 2,392-window split ⇒
**curvature −0.001903 [−0.003359, −0.000587] separated better (0.007200 → 0.005330, −26 %)**,
**ADE −0.000211 [−0.000690, +0.000266] NOT separated — the halving is KEPT**, speed MAE
**−0.008266 separated better**, cost **+0.000763 m cross-track (0.76 mm)**.
⇒ It applies to **this arm's landing dump at zero GPU**, with λ re-chosen on a disjoint
half of *this* arm's episodes. ⛔ λ is **never** selected on the scored split.
⭐ **Its deliberate-regression control fires**: `ROUGHEN_hf_x2` reads **+0.005626 separated
WORSE**, so the instrument is proven able to move in **both** directions — which is what
makes the −26 % admissible rather than a one-sided readout.

⛔⛔ **A PROPAGATED OVERSTATEMENT, CORRECTED HERE AND IN THE REGISTER.** The brief, and
`H-R4B-1`'s own register row, and `…/2026-09-06-goal-point/RESULT.md` §5, and
`PREREG_R4B.md` §0 all say the refinement **"halves ADE"**. **MEASURED: it does not.**
anchor **0.4281** → emitted **0.2965** = **−0.131669 m [−0.145772, −0.118232]**, a
**−30.7 %** reduction. A halving would be 0.214. ⚠️ **The "doubles curvature" half IS
correct** — 0.004019 → 0.008150 = **2.03×**. ⇒ the true trade is *"−30.7 % ADE for 2.03×
curvature"*, and R4b keeps the first while removing a quarter of the second.
⚠️ **CURVATURE CARRIES ITS MASK OR IT IS NOT QUOTABLE:** **0.008097** is the shipped
`ff.lateral` **own-mask** form (13,560 curvature steps); **0.008150** is the
**intersection-mask** form (13,558 steps). Both are real and they are not interchangeable.
`ha0` reads **0.006802** under both.
⚠️ **Do NOT compose it with `--feasible-decode` in the headline.** `D-R4B-FEASIBLE-DISJOINT`:
they are **complementary, not duplicated** — the projection caps curvature **MAGNITUDE**,
R4b targets curvature **ERROR** — but the projection alone costs **+0.001886 m ADE,
separated WORSE**, and composed (`FEASIBLE_ridge_3`, −40 % curvature) still costs
**+0.001058 m**. ⇒ `--feasible-decode` **OFF** in the headline; the composition is **ladder
rung L5**.

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **6** | ⭐ **22-token tactical goal vocabulary** | ⛔ **OUT — ESCALATION E2, a plumbing gap** | MEASURED: `TACTICAL_GOAL` appears **0** times in the trainer; `tac_vocab_version` stamped **`"v7.0"`** while labels are **v7.2**. |

⛔⛔ **CORRECTION TO THE BRIEF — AND TO MY OWN FIRST DRAFT. IT IS *NOT* "A PLUMBING GAP,
NOT AN ARCHITECTURE GAP". FOR REF-C IT IS BOTH.** The brief states *"`goal_head_tac` is
ALREADY 22 wide ⇒ a plumbing gap"*. MEASURED this turn by byte probe with all four files
read successfully: **`goal_head_tac` appears 0 times in `refc_v3_train.py`, `refc_v3.py`,
`refc.py` AND `v7_labels.py`.** It lives in **`stack/tanitad/models/v6.py`** (×25) — **it is
a v6 head.** ⇒ *"already 22 wide"* is true **of the v6 staged line, not of anything in the
REF-C line.* There is **no head in refcv5 to receive these tokens.**
⚠️ Two further corrections in the same family: `TACTICAL_GOAL_TOKENS_V7` is **not defined in
`v7_labels.py`** at all (it is in `stack/tanitad/models/vocab_v7.py`; `v7_labels.py` does
not even import it), and `v7_labels.HEADS` has exactly **four** entries — `tac_lat`,
`tac_lon`, `str_action`, `str_goal` — none of them the goal set.
⚠️ ⛔ **And the goal set is MULTI-LABEL — a *set* per window, not a class** — so it is not a
softmax head like the other four. **Adding it is a design decision, not a one-line dict
edit**, which is precisely why E2 must not be described as plumbing.

⭐ **The rest of the brief's characterisation holds and is confirmed.** `load_v7_labels`
routes `g_tac` to `_goal_audit` → `audit["goal_flags"]`, which keeps **only metadata**
(`disputed`, `time_basis`, `t_nominal_s`, `held`, `grounded`, `grounding`) — the token
survives merely as a dict *key* — and `V7Label` declares the destination in source:
*"audit-only, NEVER a training input (spec §6)"*. The only `g_tac` content reaching a real
field is `tac_anchor`. Proven by **mutation**: a red↔green swap changed **0 / 4,572**
supervised records against a control at **779 / 4,572**. This is what makes traffic lights
(`TL_RED` 376 / `TL_GREEN` 363) and lane changes (`LANE_CHANGE_L` 23 / `LANE_CHANGE_R` 15)
learnable **at all**.
⚠️ **Related, and it explains the `v7.0`-on-v7.2 stamp:** `tac_vocab_version` is a
**hardcoded literal** — `"v7.0" if --v7-labels else "kin3"` — and never reads the blob's
declared version, even though `v7_labels.py` separately validates `EXPECTED_VOCAB = 'v7'`
and refuses a mismatch. The validated value is **never propagated**. **ESCALATION E8.**
⛔ **But it must NOT be wired into this arm even once the diff lands**, for two measured
reasons that are items 12 and 13: **88.13 %** of the usable horizon carries no v7 tactical
GT, and **604 of 779** traffic-light records are `disputed`. **Supervising on that teaches
noise.** ⇒ `v7_labels.py` is **sibling-owned**; E2 files the exact diff, and the arm that
consumes it is **gated on a quality filter**, not on the diff.

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **7** | ⭐ **Selector training (WP-7)** | ⛔ **OUT — double-blocked** | (a) **unbuilt** — no trainer flag; (b) ⛔ **unscoreable** until `D-REFCV4B-ASTAR-GEOMETRY` lands |
| | ⛔ **`--sel-refined`** | ⛔ **NOT PASSED — and cannot be from THIS trainer** | MEASURED by 4 mechanisms: `sel_refined` = **0** in `refc_v3_train.py` (same-breath control `sel_accel_max` = **7**). |

⛔⛔ **BUT "THE FLAG DOES NOT EXIST" IS THE WRONG STATEMENT, AND THE RIGHT ONE IS A DEFECT
IN THIS ARM.** MEASURED this turn: `--sel-refined` **is registered in the legacy trainer**
`stack/scripts/refc_train.py`, and the mechanism **is implemented** in `refc.py`
(`sel_refined: bool = False` in `RefCConfig`, wired through `SelectionConfig(refined=…)`,
armed by the preset `refc_select_config()`). ⇒ **the correct statement is that the refcv5
trainer cannot arm a lever its own core implements**, so every refcv5 run ships with
`SelectionConfig.refined == False`.
⛔⛔ **AND THE SAMPLER MAKES THIS STRICTLY WORSE THAN IT WAS FOR refcv4b.** `refc.py` stamps
`"sampler_ranks_the_fan": bool(self.sel.refined)` — **False**. ⇒ **refcv5 added stochastic
sampling and then ranked the sampled fan with a score that never saw the sample.** The
sampler's entire product is *diversity*, and the selector is structurally blind to it.
`refc.py` itself quantifies the cost of ranking with an unrefined score at **>2× worse than
the fan-best pick on 41.09 % of windows (base)** / **45.4 % (REF-C-XL)**.
⇒ ⭐ **This reframes item 7 from "a lever we are declining" to "a defect the sampler
sharpened", and it is why E5 (deep supervision) is ranked as high as it is.** ⛔ It does
**not** license flipping the flag: `D-REFCV4B-SELREF-1` measured that arming the
*untrained* readout is **separated WORSE**, and a sampled fan does not fix an unsupervised
ranker. **Training it is the work; E5 is the escalation.**

⛔ **The prohibition's reason stands independently of the flag's absence:**
`D-REFCV4B-SELREF-1` — `os` 0.3055 → **0.3314 m**, paired **−0.0259 [−0.0505, −0.0033],
p 0.017**, **51/171 = 29.82 %** of picks flip, fan collapses 18 → 16 anchors, and both
minority recalls fall (`brake_stop` 0.6429 → 0.5714, `accelerate` 0.6667 → 0.5926) while
accuracy *rises* — a **textbook majority-class trap** (`steady` is 76 % of windows).
Mechanism: `seen_in_training: no` ⇒ an **out-of-distribution use of an untrained head**.
⇒ *"refcv5 cannot buy DiffusionDrive's selection by flipping a flag; WP-7 selector TRAINING
is the load-bearing part."*
⛔ **And no selector claim from this arm may rest on `anchor_acc`**: `a_star` is computed
from `decoder.anchors` (the family rolled at the **reference speed**) instead of
`out["anchor_bank"]`, which the trainer itself forbids for a v0-conditioned vocabulary.
Consequence: `oracle_sel` reads **+0.9179 separated WORSE than `os`** — *a "ceiling" 4×
above the arm it bounds is not a ceiling.* **Fix ≈ 2 lines + a T0 re-roll — ESCALATION E3.**

⭐⭐ **AND WP-7 IS BIGGER THAN "TRAIN THE RANKING HEAD" — IT IS DD'S DEEP SUPERVISION.**
PUBLISHED from source: DD applies **`(reg, cls)` at EVERY cascade layer and sums the
losses**, with the cascade **gradient-detached between layers**, and the output is
`argmax(poses_cls[-1])` on `poses_reg[-1]` — *the last layer scores the fan it emits.*
REF-C has **neither half**: CE on the `t=0` confidence only, and gradient flowing through
the whole path. ⇒ **DD elements #8 and #28 are one work item with item 7**, and it is
`ESCALATION E5`. ⚠️ Its classification loss is **sigmoid FOCAL (γ=2.0, α=0.25)**, not the
BCE the paper's Eq. 6 states, and not our softmax CE — the class imbalance that makes
`--sel-refined` a majority-class trap is exactly what focal loss exists to fix, so **the
loss form is part of the escalation, not a detail.**

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **8** | ⭐ **`--no-strategic` bypass** | ⛔ **NOT IN THE FIRST ARM — deliberately** | the flag **EXISTS and is proven** (`D-STRAT-BYPASS-WIRED-1`, commit `bfcdd75`) |

⭐ **Stated plainly, as the brief requires: the first arm does NOT use it.** Not because it
is unavailable, but because `D-STRAT-BYPASS-1` is **already pre-registered as its own
ONE-VARIABLE experiment** (refcv4b's argv byte-for-byte + `--no-strategic`), and folding it
into a capability bundle would destroy **both** claims: the bypass would lose its
attribution, and the capability arm would gain a lever whose sign is unknown. It is
`BLOCKED ON COMPUTE ONLY` and is the **next A40 arm after this one** (§6, rung L7).
⭐ Its wiring is proven by **mutation in both directions**: flag ON, overwriting all 18
strategic tensors (5,710 elements) with `N(0,3)` leaves **0 of 28** downstream surfaces
changed; the same mutation on an OFF build moves **20 of 28** (`z_tac` max|Δ| **156.14**).

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **9** | ⭐ **Distance-keeping cost** | ⚠️ **OUT of this trainer; the REPRESENTATION it needs is IN via item 2** | MEASURED: `dk-w`/`dk_w` appear **0** times in `refc_v3_train.py`. The implementation is **refav1's**, eval-time (`stack/tanitad/refs/refav1_lon_cost.py`, `--dk-w/--dk-tau/--dk-d0/--dk-gap-source/--dk-gap-block` in `refav1_arm.py`) |

⭐ **MEASURED and validated there:** position-based, **fires 21/90**, **re-ranks 21/21**,
broke the constant-velocity degeneracy. ⭐ **The form, read from source this turn** —
`w_dk · mean_k( relu(s*(v_k) − gap_k)² )`, with an **IDM-affine desired gap
`s*(v) = d0 + τ·v` in metres** and a predicted gap that is a **position difference**
(`gap[k] = gap0 + s_lead(t_k) − s_ego(t_k)`). **Every term entering the penalty has units
of metres**, so the penalty is `relu(m − m)²`. Default **off** (`DK_W = 0.0`), with
exact-zero returns when unarmed.
⭐ **The module states the exclusion as a first-class design fact, not an omission:**
*"this term CANNOT price closing, and does not pretend to."*
⚠️ **AND THE SMUGGLING RISK IS REAL AND LOCATED.** A rate term **does** exist elsewhere —
`stack/tanitad/rl/rewards.py` computes `closing = (gap[…, :-1] − gap[…, 1:]) / dt` plus a
TTC veto — but it is a **finite difference of the same position gap, used as a hard veto**,
and it is consumed by `refc_selector_targets.py`, **not by the trainer**. ⇒ **the trainer
carries no rate term today — by ABSENCE, not by design**, which is exactly the condition
under which one gets added by accident. **GATE 0 asserts it stays absent.**
⛔ **It prices a GAP, not CLOSING** — and that is a *finding about the trunk*, not a design
preference: lead **position** is decodable (**+0.4145** vs pixels; constant control
**exactly +0.000000**) while closing **rate** is a **clean null (+0.0061)** on every arm,
and a temporal difference recovers nothing. ⛔ **Nothing in this arm may smuggle a rate
in** — a cost term over a quantity the trunk cannot decode would fit noise.
⭐ **The correct move is therefore item 2, not a cost term**, which is exactly what
`D-REFCV5-LEVERS` rank 1 says: *"A REPRESENTATION requirement — no cost re-weighting
substitutes."* Once the agent join supplies `v_rel_x`, the rate becomes decodable **as an
input**, and a rate-aware cost becomes admissible — **in a LATER arm, pre-registered
separately, and only after a decodability probe says the rate is now readable** (§6, L6).

⛔ **TWO SCOPE FACTS THAT MAKE "PORT THE DK COST" MORE EXPENSIVE THAN IT SOUNDS:**
**(a)** it is a **REF-A planner cost**, hooked into `refa_v1.plan` and armed only from
`taniteval/tools/refav1_arm.py`. **REF-C does not call that planner** ⇒ porting it is real
work, not a flag. **(b)** ⛔ **its gap input is currently the ORACLE B1 label** — so the
`21/90` / `21/21` result is a **CEILING, not a capability**, and must never be quoted as
one. Its A/B on the real checkpoint is **n = 4 windows / 2 episodes** — a plumbing proof.
⭐ The direction check is the solid part: MEASURED on **8,339 lead rows / 147 clips** with a
**within-clip shuffle control** (excess −0.2120 at τ=1.2, −0.2053 at τ=1.5), and the shipped
`DK_TAU_TARGET_S = 1.5` sits in the optimum.

| # | item | disposition | flag + weight / blocker |
|---|---|---|---|
| **10** | **Longitudinal kin3 head** | ⚠️ **NOT DIRECTLY ADDRESSED — stated, not hidden** | no flag disables it |

⭐ **WHAT IT ACTUALLY IS, read from source this turn — and it is not what its name suggests.**
`kin3` is a **3-way categorical manoeuvre head**, `LON_CLASSES = ("brake_stop", "steady",
"accelerate")`, `nn.Linear(aux_hidden, 3)`. ⛔ **Nothing in the kin3 path reads a lead
vehicle or a distance.** There is **no CLI flag for its weight** (`LON_WEIGHT` is an
imported module constant); the only flag that changes its vocabulary is `--v7-labels`.

MEASURED (`D-REFCV5-LEVERS` rank 5): the **eval-set** prior floor is **0.842145 nats**
(marginal [0.16131, 0.684429, 0.154261], counts [778, 3301, 744], n = 4,823 / 141) against
an eval CE of **1.00903** ⇒ **0.1669 nats WORSE than predicting the class marginal.**
(Lateral floor **0.519904** and the lateral head *does* beat it at 0.4558. Uniform is
1.098612, so the head sits only **0.0896** below chance.)

⛔⛔ **BUT THE TWO HALVES OF THAT SUBTRACTION COME FROM DIFFERENT SOURCES AND DIFFERENT
STEPS, AND THAT IS AN EVIDENCE-CLASS VIOLATION.** MEASURED: the **floor 0.842145** is from
the **eval dump at step 40,284**; the **CE 1.00903** is `eval_lon` from the **TRAINER LOG at
step 31,500** (`raw/refcv4b_metrics_31700.jsonl`) ⇒ **INHERITED, not eval-harness output**,
which CLAUDE.md rule 1 forbids for a decision-grade number (*"trainer val watches a curve;
only eval output is quotable"*, and the historic cost was a ~10 % optimistic reading).
⇒ ⛔ **`0.1669` is a MIXED-SOURCE, MIXED-STEP figure and is NOT quotable as it stands.**
⚠️ The same package's `STATUS.md` still carries a superseded **0.1126** (floor 0.8964 from
`ckpt_30000.pt`'s `core.lon_log_prior`) — **three values are circulating for one quantity.**
⭐ **What this SPEC does about it:** the arm's landing eval **re-derives both halves from
its OWN eval dump at its OWN final step**, and the committed prediction in the next
paragraph is stated against **that** number, not against 0.1669. **ESCALATION E10** — the
register row needs the same repair. ⭐ **This arm's hypothesis about it, committed in advance:**
the head is not broken, it is **starved** — it is asked for a longitudinal decision from a
trunk in which closing rate is a clean null. Item 2 supplies exactly that signal. ⇒ **The
committed prediction is that kin3 eval CE falls below 0.8421 in the agent-token arm and
does NOT in the no-join arm.** ⛔ If it fails to move *with* the join, the head is a real
defect and rung **L8** removes it. Either way the number is **reported**.

| **11** | **H19 anchor prior** | ⛔ **OUT — de-prioritised, correctly** | MEASURED 0.0034 m, **NOT separated**, 5.85 % of picks |

### C · BLOCKERS — STATED, NOT HIDDEN

| # | blocker | disposition |
|---|---|---|
| **12** | ⛔ **88.13 % of the usable horizon carries no v7 tactical GT** | **STATED AS A DEPENDENCY.** MEASURED: **one record per clip** (4,572 over 4,572 distinct `clip_id`), `t0_s` a **constant 8.0**, band `[2.0, 6.0]` = **4.0 s wide** ⇒ supervised window `[6.0, 10.0] s`; outside it the label is `IGNORE_ID (-100)`, never clamped to a neutral class. **11.87 %** covered against `available_s` (p50 35.0 s); **3.59 %** against `recording_span_s` (p50 139.7 s). ⛔ **Both denominators are reported** — quoting only 3.59 % overstates it. ⛔ **The remedy is NOT a wider tolerance** (that relabels frames the emitter never examined); it is more anchors per clip — a **Data Engineering** work item. ⭐ **A sibling is establishing writer-limit vs source-limit; item 6 depends on that answer, and this arm does not wait for it.** |
| **13** | ⚠️ **Label quality** | **STATED.** Only **175 of 779** traffic-light records are `grounded`; **604 are `disputed`** (`vlm-cot`). ⛔ Supervising without a quality filter teaches noise ⇒ item 6 stays OUT. ⚠️ **YELLOW n = 22 fails `GOAL_MIN_N_FOR_METRIC = 200`** — **representable, not scoreable**; it is reported with its `n` and never as a rate. |
| **14** | ⛔ **Five zero-weight defaults** | **ALL EXPLICIT ON THE LAUNCH LINE.** MEASURED at HEAD: `U0_WEIGHT_DEFAULT = 0.0`, `AGENT_WEIGHT_DEFAULT = 0.0`, `GOAL_POINT_WEIGHT_DEFAULT = 0.0`, `--agent-w-project` default `0.0`, `--agent-w-ground` default `0.0`. Each is **hard-guarded so the term is SKIPPED ENTIRELY**, not scaled. ⇒ every one is passed by name, and **GATE 0 asserts the effective value in `config.json`** and that its gradient is **non-zero** (the M18 check). ⛔ `--agent-w-ground` is passed as **`0.0` deliberately** and the RESULT says why (item 2). |
| **15** | ⛔ **Corpus: B1, not parity** | **STATED, AND IT TRAVELS WITH EVERY NUMBER.** `v2_parity.parity: false`, `require_parity: false`, 4,572 clips. ⛔ **Inadmissible for any parity claim or cross-arm comparison with the parity arms** — the trainer says so itself at startup. It is the right corpus *here* because refcv4b trained on it, so `cap vs refcv4b` is matched. ⚠️ **Anchor units are OPERATOR-ASSERTED** (`control_units_source: cli-override-legacy-file`) — grounded because the file is byte-identical (`file_sha256 e86cf507d5…`) to the bank refcv4b trained 40,284 steps on. ⭐ **A re-stamped, self-declaring bank exists** (`anchors_117_alat_declared.pt`, tensors proven bit-identical, undeclared original still refused) ⇒ **this arm uses it and drops the CLI override**, so the record reads `control_units_source: file`. |

---

## 3. ⛔ THE BARS — both outcomes committed, verbatim, before any data exists

Every bar is on **`ade_m`**, **paired episode-cluster bootstrap** (`taniteval/ci.py`,
n_boot 2000, seed 0, cluster = episode), on **the same 4,823 windows / 141 episodes**
refcv4b was scored on. ⛔ **Never `overlapping_holdout_se`.**

⭐ **SOURCE, and it is the registry — not this SPEC and not the brief.** `MODEL_REGISTRY.md`
refcv4b row, **LANDING READ — T1, 4,823 windows / 141 episodes, paired episode-cluster
bootstrap (`taniteval/ci.py`), n_boot 2000, seed 0**, artifacts in
`…/2026-09-06-refcv4b-landing/`.

| arm read against | ADE (m) | note |
|---|---|---|
| **`ha0_ext`** — constant `a0` **and** `ω0` at t0 — **the strongest model-free control** | **0.2874** | ⭐ **the bar** |
| `os` (refcv4b) — the incumbent | **0.2975** [0.2706, 0.3295] | ⚠️ **corrected — see §3.4** |
| `ha` — hold-action | **0.2996** | |
| `os_navshuf` — pairing broken, route marginal preserved | **0.3013** | ⚠️ **corrected** |
| `os_navzero` — route input stripped | **0.3928** | ⚠️ **corrected**; reported, never gating |
| `ha0` — constant velocity, the straight-line floor | **0.6723** | |
| `frames_blind` — ⛔ **the deliberate regression** | **1.0491** | must regress, or the panel is VOID |
| `ego_zero` — the PI-binding vision-only read | **1.1310** | reported, never gating |

### 3.1 ⭐ PASS — **P1, the only criterion that decides "progress"**

> **`cap − ha0_ext` on `ade_m` is NEGATIVE, its paired CI excludes zero, and
> `|delta| > F(ade_m)`** — where `F` is the **inference-seed floor** from the replicate.

⭐ **This is the bar refcv4b FAILED:** `os − ha0_ext` = **+0.0101 [−0.0050, +0.0273] —
NOT separated, a TIE on the wrong side of zero.**

### 3.2 ⚠️ PARTIAL — **P2, `LEVER-SUPPORTED / NOT-DRIVING`**

> **`cap − refcv4b os` is NEGATIVE, separated, and `|delta| > F(ade_m)` — while P1 does
> NOT hold.**

⛔ **Beating refcv4b while still only tying `ha0_ext` is NOT DRIVING, and the headline
sentence must say so.** Two arms that both tie a constant-velocity-plus-yaw control are two
arms that have not shown skill; a delta between them is **a delta between two ties**.
refcv4b already cut refcv3's deficit **7×** and still did not cross this bar.

### 3.3 ⛔ FAIL — committed in advance

* **F1 NULL** — `cap − refcv4b os` not separated, **or** `|delta| ≤ F(ade_m)`.
  ⇒ the bundle does not move driving at this scale on this surface.
  ⚠️ Under `H-ESTIM-SEED-1` this refutes **this arm at this budget**, not the elements.
* **F2 REGRESSION** — separated POSITIVE. ⇒ the bundle costs driving quality; the next
  lever is the **ladder (§6)**, to find which element, **not more training**.

⭐ **RULE ZERO: neither F1 nor F2 ends the turn.** On either, the ladder in §6 runs **in
the same session** — every rung is eval-time on the banked checkpoint, minutes not
GPU-days — and the turn ends with *"X failed, Y is the next lever, here is Y's result."*

### 3.4 ⛔ VOID — not "negative", VOID

Any of: **G-ARGV** fails · the **replicate** gate fails in either direction · the
deliberate-regression arm `frames_blind` does **not** regress · the model-free arms are not
bit-identical across arms · `paired(a, a)` is not exactly `0.0000000000` · the window grids
do not match (`ws`/`eid` unequal). **A panel whose instruments have not been shown able to
FAIL certifies nothing.**

⛔⛔ **TWO DUMPS OF THE SAME ARM DISAGREE, AND THAT ARM'S OWN REPRODUCTION CONTROL DECLARES
`FAIL`. THIS IS AN OPEN INSTRUMENT DEFECT, NOT A TRANSCRIPTION ERROR.**

⚠️ **I first wrote that the brief's `os` 0.2965 / `os_navshuf` 0.3006 / `os_navzero`
0.3926 were simply "wrong" because `MODEL_REGISTRY.md` reads 0.2975 / 0.3013 / 0.3928.
That was an absence-found-at-ONE-location error, and it is CORRECTED HERE.** MEASURED:
**both families are real** — two different dumps of refcv4b @40,284 on the same 4,823
windows / 141 episodes.

| dump | `os` | `os_navshuf` | `os_navzero` |
|---|---|---|---|
| **landing** — `…/2026-09-06-refcv4b-landing/raw/refcv4b_t1.json`, what the registry quotes | **0.2975** [0.2706, 0.3295] | 0.3013 | 0.3928 |
| **navpred / navflip** — `…/2026-09-06-refcv4b-navpred/raw/flip_analysis.json`, what the brief quotes | **0.2965** [0.2682, 0.3272] | 0.3006 | 0.3926 |

⛔ **`paired_navpred.json` carries its own `C3_os_reproduction` control, and it reads
`measured 0.2965 / banked 0.2975 / abs_diff 0.001031 / tolerance 0.001` ⇒ verdict `FAIL`.**
The instrument **already knows** the two dumps do not reproduce each other, by **3 % over
its own tolerance**. ⇒ ⛔ **Neither family is quotable as "the" value until that control
passes** — and `os_navpred` **0.3012**, the arm item 4 is about, exists **only** in the
failing dump.
⭐ **No verdict in this SPEC moves.** `ha0_ext` **0.2874**, `ha` **0.2996** and `ha0`
**0.6723** are identical in both; the bar is on `ha0_ext`; and every paired delta is
computed **within** a dump, never across. ⛔ **The analysis script therefore reads every
comparison from ONE dump, re-derives the paired deltas inside it, refuses to mix families,
and asserts against `MODEL_REGISTRY.md`** — never against a prose table, including this one.
⛔ **ESCALATION E9.** Two dumps of one checkpoint on one window grid should agree; that they
do not is either re-roll non-determinism or a grid difference, and either contaminates any
cross-dump read. **A failing control on a banked artifact is resolved, not carried.**

### 3.5 The deployment read — reported always, gating never

`cap os_navzero − ha0_ext` (refcv4b: **+0.1054 [+0.0874, +0.1241] separated WORSE**).
⚠️ **This is a PATHWAY margin, not an information one** — the conditioning vector's
**PRESENCE** carries almost all of it, its **CONTENT** almost none.
⛔ **BUT TWO DECOMPOSITIONS OF THAT ONE QUANTITY ARE CIRCULATING AND THEY DISAGREE:**
`H-NAVPRED-1` reads **95.7 % presence / 4.3 % content**, while `goal_point.py` and
`flip_analysis.json` read **97.7 % / 2.3 %** (`os_navflip − os` = **+0.0022 [−0.0006,
+0.0052] not separated** against `os_navzero − os` ≈ **+0.0961 separated**). ⇒ ⛔ **Pick one
and name its arithmetic before quoting it** — this is the load-bearing argument for
replacing the categorical token with the goal point (item 4), so an unstated denominator
here would be a percentage without its fraction on the SPEC's own central claim.
**ESCALATION E11.** ⭐ This arm *does* move a route mechanism (item 4 swaps the categorical token
for a predicted geometric point), so unlike refcv5 it is **expected to move** — and
`os_navpred` rides the same forward at **zero extra GPU**.

---

## 4. ⛔ THE FOUR FAMILIES — never ADE alone, never pooled

Each family carries its **estimator** (paired episode-cluster bootstrap), its **CI**, its
**n**, and a **T-tier stamp** on every number. **T1** (self-action open loop) is the
primary tier; ⛔ T0 numbers are **WM diagnostics, never driving performance**.

| family | reported | committed expectation |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy, **headway / time-gap / min-TTC** vs the replay, per speed band, with `n` per band | ⭐ **the sharpest prediction of items 2+4.** The goal point's range is longitudinal (57.1 % of the ceiling, −0.1184 m/s speed MAE); agent tokens supply `v_rel_x`. **Committed: speed MAE separated BETTER than refcv4b.** ⚠️ Where no lead is in frame, report **per family with the reason and the n** — never silently dropped. |
| **LATERAL** | ⛔ **curvature MAE with BOTH floors beside it** — the straight-line `ha0` **0.006802** *and* the echo control `ha0_ext` **0.003712** — against refcv4b's `os` **0.008097**; plus heading (1.2950°), yaw-rate (1.7309°/s), cross-track (0.0979 m, best of any arm) | ⛔ **NEVER the `\|dyaw\| > 0.15` gate — the human fails it 3/9.** ⭐ **The defect this family exists to expose:** refcv4b's curvature sits **above the straight-line floor and 2.2× the echo control** ⇒ *it tracks the road WORSE than a plan that never steers.* **A SHAPE defect, not a positional one** — which is exactly why cross-track being best-in-programme must never be reported alone. Committed: R4b's re-score reads curvature **separated BETTER** with ADE **not separated worse**. |
| **TACTICAL** | manoeuvre-decision quality, turn recall **beside** ADE, confusion over classes | ⚠️ **kin3 longitudinal CE reported against its own eval-set prior floor 0.8421 nats** (item 10). ⛔ **`oracle_sel` / `anchor_acc` / `sel_agrees_oracle` are INVALID and are NOT reported** (item 7). ⚠️ YELLOW n=22 < 200: representable, not scoreable. |
| **STRATEGIC** | strategic decision + goal/route setting quality, **with its anti-echo control** | ⛔ **The route head must beat its own input, or the block is VOID** — the flagship v1 route head scored 1.0000 as an exact bijection of the nav it was fed. |

---

## 5. ⛔ THE CONTROLS — each must read a KNOWN value

| control | must read | why |
|---|---|---|
| **`paired(a, a)`** | **exactly `0.0000000000`**, `separated: false` | if the estimator cannot report zero difference for an arm against itself, nothing it reports is admissible |
| **constant-only** — `ha0` (constant velocity), `ha0_ext` (constant `a0`, `ω0`) | **bit-identical across every arm** in the panel | they are model-free; a difference proves the harness moved, not the model |
| **raw-input floor** — pixels, on every decodability probe | the learned representation must **beat** it | `D-REFCV5-LEVERS` #1's controls: lead position **+0.4145** vs pixels, **constant control exactly +0.000000** |
| ⛔ **deliberate regression** — `frames_blind` | **must REGRESS, separated.** refcv4b read **1.0491** against its `os` 0.2975 — a **3.5×** regression, so the gate has a known-good reference and is not merely "some positive number" | without it nothing above is admissible. ⭐ Precedent: `D-STRAT-BYPASS-WIRED-1`'s first probe PASSED while never looking at `z_tac` — *a proof that does not look at the seam it is proving is not a proof.* |
| ⛔ **replicate** — `cap_R1` (same flags, **different inference seed**) | establishes `F(ade_m)` | ⛔ **MANDATORY: this arm SAMPLES at inference.** `H-ESTIM-SEED-1` measured **6/42 = 14.3 %** false "separated" from *training* variance alone; a stochastic planner adds a **third**. ⛔ **No margin is quoted before `F` exists.** |
| **`n` and `d` printed** in every probe table | — | `n ≪ d` is underpowered **by construction**, not a negative |

### 5.1 The gate ladder (`TanitAD_ValidateAIDesign` §3) — in order

* **G-RANK** — participation ratio (**σ², never `effective_rank`**) against a **MATCHED**
  reference: **refcv4b itself**, same corpus, same episode count, same ambient `d`.
  ⛔ **The bare 8.56 floor is RETIRED and is not used in either direction.** Absent a
  matched reference the honest reading is **UNDECIDABLE — neither pass nor fail**.
  ⛔ **Rank is NECESSARY, NOT SUFFICIENT** (C131) — never pass an arm on it.
* **G-DECODE** — ⭐ **the decisive gate for item 2.** The lead-position / closing-rate probe
  must beat **both** the raw-pixel floor **and** the constant control, with λ and PCA basis
  fit on the FIT split only. **Committed:** in the agent-token arm, closing rate moves off
  its **+0.0061** null; in the no-join arm it does not.
  ⚠️ **A negative from a LINEAR probe is not a negative about learnability** — state the
  function class, or use a nonlinear probe with a **time-shuffled** control.
* **G-DRIVE** — §3 and §4.

---

## 6. ⭐ THE ABLATION LADDER — named in advance, RANKED BY MEASURED EFFECT SIZE

⛔ **This is how per-element attribution is discharged.** Every rung **L1–L6** is an
**eval-time switch on ONE checkpoint** ⇒ training variance does not enter, and the correct
claim form is *"this switch changes this metric on this checkpoint"*, **never** *"this
lever is worth X in the next arm"*.

| rung | ablation | ranked by | cost |
|---|---|---|---|
| **L1** | **agent tokens off** (`--agents off` at eval) | rank 1 lever; +0.4145 decodability, the only *representation* item | minutes |
| **L2** | **goal point → categorical route** (`gp_cond` → `base`) | 57.1 % long / 78.4 % lat of the selection ceiling | minutes |
| **L3** | **goal point, RANGE STRIPPED** (bearing only) | ⭐ the **value-sensitivity GATE** — must cost **≥ +2.3632 m**, or the goal is not being read | minutes |
| **L4** | **R4b re-score off** | −26 % curvature at 0.76 mm | 0 GPU |
| **L5** | **R4b ∘ `--feasible-decode`** | −40 % curvature at **+0.001058 m ADE** | 0 GPU |
| **L6** | **closing-rate decodability, with and without the join** | settles whether item 2 fixed lever 1 | minutes |
| **L7** | ⭐ **`--no-strategic`** — **a SEPARATE 44 h arm**, already pre-registered (`D-STRAT-BYPASS-1`) | blocked on compute only | ~44 h A40 |
| **L8** | **kin3 head removed** — only if item 10's committed prediction fails | 0.1669 nats | ~44 h A40 |
| **L9** | ⭐⭐ **DD #19 + #8/#28: waypoint-indexed spatial attention + per-layer deep supervision** | ⛔ **DD's own ablation prices #19 at −33.0 PDMS — the single largest published effect in either paper** | E5 + a metric frame |

⚠️ **L7–L9 are TRAINING arms and are NOT part of this panel's attribution** — they are
named so a reader knows the plan, and each carries the replicate requirement separately.
⛔ **L9 is the honest answer to "what is still missing after this arm."** Grounding the
attention **at the candidate's own waypoints** needs a metric frame REF-C does not have
(no BEV lift, no LiDAR, no map — settled at five probes). The PV analogue is already
pre-registered as `H-DDA-1` on the v7-tiny ladder, and ⭐ **that ladder rung should run
BEFORE any GPU-day is spent on L9** — which is precisely the `TanitAD_ValidateAIDesign`
rung-1 discipline this arm had to declare a deviation from.

---

## 7. ⛔ THE ARMS AND THE EXACT COMMANDS

### 7.0 ⛔ GATE 0 — before a single GPU second

Every one is a **mutation or positive-content assertion**, never an inspection, and each
carries a **same-breath control that must read a known value**.

⛔⛔ **AND EVERY GATE-0 CHECK MUST BE ON THE REAL PATH, NOT ONLY IN `preflight`.** MEASURED
by a sibling stream this same day (`D-EFFW-5`): **`refc_v3_train.py`'s `main` runs
`preflight` ONLY under `--preflight`, and otherwise calls `train(args)` directly** — so a
guard wired into `preflight` alone covers **one launch path of two**. This is why every
existing guard there (`_check_nav_from_v7_args`, `_check_goal_point_args`,
`_read_anchor_artifact`, `_check_anchor_artifact_against_cfg`) is **double-called**.
⇒ ⛔ **A green `--preflight` run is NOT evidence that the real launch is guarded**, and this
SPEC's GATE 0 is run as `--preflight` **and** asserted from the resulting `config.json` of
the real 2-step run (step 7).
⭐ Corroboration for item 14 from the same stream: refc has **exactly five** weight-like
flags — `--w-u0`, `--w-agent`, `--agent-w-project`, `--agent-w-ground`, `--goal-point-w` —
and **all five were already gated**, with `REFC_WEIGHT_GATES` now **exhaustive over the
parser** so the next `--w-*` cannot ship ungated.

1. **G-ARGV** — the launched `argv` and `config.json` `seams` match this SPEC **exactly**;
   every one of the five zero-weight knobs (item 14) is present with a **non-zero effective
   value** where intended, and **`0.0` where deliberate**.
2. ⛔ **M18 DEAD-FLAG CHECK** — each weighted term has a **non-zero parameter gradient** in
   a real backward. Control: at least one term must read LIVE and the known-dead
   `agent_w_ground` must read **DEAD** (it read **1.164e-10** at refcv5's preflight).
3. ⛔ **THE DIFFUSERS PIN MUST PASS** — `refc_sampler.assert_matches_diffusers`.
   `D-REFCV5-DIFFUSERS-PIN-CANNOT-PASS` says it has never certified anything; ⛔ **an
   unverified re-implementation does not launch a 45 h run.** If it cannot pass, the arm
   launches with the sampler and the RESULT stamps the pin as **INCONCLUSIVE**.
4. ⛔ **PREFLIGHT IMPORT PROBE** — `taniteval`, `taniteval.ci`, and every analysis import
   resolve **at startup**. *(An analysis-time import killed a 100 %-complete two-arm
   rollout after the compute was already paid for.)*
5. **STALE-STACK REFUSAL** — the supervisor refuses unless `DDIMSchedule`, `roll_controls`,
   `build_agent_head` **and** the goal-point symbols import from the shipped tree.
   ⛔ **Ship files (md5-verified); never `git fetch` on a pod** — it hangs, and a failed
   fetch followed by a checkout destroys the shipped files.
6. **ANCHOR UNITS** — the re-stamped bank readback reads `control_units=alat
   source=file`, and the **undeclared original is still REFUSED** (the negative control
   proving the guard was not weakened).
7. **A 2-STEP RUN** completes and states every weight; **`--analyze-only` / the dump dir is
   checked BEFORE any re-run.**
8. **`OMP_NUM_THREADS=6`** before any multi-arm panel.

### 7.1 ⭐ THE PRIMARY LAUNCH COMMAND — ready now, blocked on nothing

```bash
OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack \
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 \
  --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 \
  --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --anchors /workspace/experiments/refcv5-cap-b1-v72-40k/anchors_117_alat_declared.pt \
  --n-anchors 117 --anchor-v0-conditioned \
  --sel-accel-max 2.0 --goal-str \
  --ego-state-inject --ego-dropout 0.5 \
  --sampler ddim --sampler-space control \
  --sampler-infer-t 8 --sampler-steps 2 --sampler-groups 1 \
  --w-u0 0.5 \
  --goal-point-inject --goal-point-w 1.0 --goal-point-t 4.0 --goal-point-geo-prior \
  --agents off --w-agent 0.0 --agent-w-project 0.0 --agent-w-ground 0.0 \
  --out /workspace/experiments/refcv5-cap-b1-v72-40k
```

⚠️ **`--anchor-control-units alat` is deliberately ABSENT**: the re-stamped bank declares
its own units, so the record reads `control_units_source: file` instead of
`cli-override-legacy-file`. ⛔ **GATE 0 step 6 asserts this**; if the re-stamped bank is not
in place, add `--anchor-control-units alat` back and the RESULT says the units are
operator-asserted.
⚠️ **The four zero-weight flags are passed EXPLICITLY AT ZERO** so `config.json` records
that they were considered and declined, not forgotten (item 14).
⛔ **`--sampler-train-t-max` is DELIBERATELY OMITTED from this launch line.** It has **no
consumer** (item 1), so passing it would stamp a mechanism that does not exist — the exact
M18 defect this SPEC exists to avoid repeating. ⭐ **Add it back only after E6 lands**, and
say in that arm's record that the train draw became real.
⚠️ **`--sampler-steps 2` is passed although it is currently SHADOWED** — it is what the
model will use anyway (`diffusion_steps = 2`), so the value is honest; the *flag* is inert
until E6, and the RESULT says so rather than implying the CLI chose it.

### 7.2 ⭐ THE ONE-LINE DELTA THAT ADDS WP-6 — and the decision rule

**Decision rule, committed now:** at launch preflight, if a **B1-scoped TRAIN**
`obstacle.offline` join is present and passes `--agent-join-verify auto`, launch **7.2**;
otherwise launch **7.1** and WP-6 becomes the next arm. ⛔ **The parity-scoped join
(7.92 % overlap) is NEVER substituted.**

Replace the `--agents off …` line in 7.1 with:

```bash
  --agents head --agent-join /root/data/b1_train_agents.jsonl.xz \
  --agent-join-verify auto --agent-queries 100 \
  --w-agent 1.0 --agent-rig-camera nominal --agent-w-project 0.05 --agent-w-ground 0.0 \
```

### 7.3 The rolls, in ranked order — a kill at any point still leaves value

1. **GATE 0** (§7.0) · 2. **`cap_R0`** — the four-family read + every model-free control ·
3. ⛔ **`frames_blind`** — without it nothing above is admissible ·
4. ⛔ **`cap_R1`** — the **inference-seed** replicate; `F(ade_m)` exists after this and
**not before**, and ⛔ **no margin is quoted until it does** ·
5. the paired reads: **P1**, **P2**, **L1–L6** · 6. **`os_navzero` + `os_navpred`** (free,
same forward) · 7. **R4b re-score** (0 GPU) · 8. `cap_R2` if GPU remains.

---

## 8. ⛔ ESCALATIONS — the wiring changes I need, with the exact diff, NOT a "please merge"

⛔ **I do not own these files.** Each is escalated to the Master Mind **with the change
stated**, because *an orthogonality instrument sat unmerged for 10 days because the request
lived in a README nobody re-read.*

| id | file (owner) | the change | blocks |
|---|---|---|---|
| **E1** | `stack/tanitad/refs/refc.py` **or** the eval re-scorer | expose R4b's curvature-penalised smoother — `argmin ‖u − off‖² + λ‖D²u‖²`, **λ = 3**, λ selected on a **disjoint** episode half — as a decode-time option with the λ stamped into the dump | item 5 at eval (⭐ **not** the launch) |
| **E2** | `stack/tanitad/data/v7_labels.py` | add `TACTICAL_GOAL_TOKENS_V7` to `v7_labels.HEADS` and route it out of `audit["goal_flags"]` into a supervised batch field; **`goal_head_tac` is already 22 wide**. ⛔ **Must ship WITH a `grounded`-only quality filter** (175/779) — the diff alone is not safe to consume | item 6 |
| **E3** | `taniteval/tools/refcv3_arm.py` | compute `a_star` from `out["anchor_bank"]`, not `decoder.anchors` (**≈2 lines + a T0 re-roll**) | item 7, and **every** selector metric on **every** v0-conditioned arm |
| **E4** | DataFlyWheel · `stack/scripts/build_obstacle_join.py` | build the **B1-scoped TRAIN** `obstacle.offline` join, **≈2.3 GB** via per-clip HTTP range requests (505 KB/clip). ⛔ **Take an HF quota reading BEFORE the pull** | item 2 (§7.2) |
| **E5** | `stack/tanitad/refs/refc.py` (decoder) + `refc_v3_train.py` | ⭐ **DD #8/#28 — deep supervision**: emit `(reg, cls)` at **every** decoder pass, sum the loss over passes, **detach between passes**, and select with `argmax(cls[-1])` on `reg[-1]` so the scored object IS the emitted object. Classification as **sigmoid focal (γ=2.0, α=0.25)**, not softmax CE — the imbalance that makes `--sel-refined` a majority-class trap is what focal exists to fix. ⛔ **Now the highest-value escalation**: refcv5 ranks a **sampled** fan with a score that never saw the sample (`sampler_ranks_the_fan: False`), costing **>2× worse-than-fan-best on 41.09 %** of base-size windows | item 7, rung **L9** |
| **E6** | `stack/tanitad/refs/refc.py` (`_sample`) | ⛔ **`--sampler-train-t-max` HAS NO CONSUMER** — add the `self.training` branch so the train draw is `t ~ U[0, t_max)` as DD does, instead of reusing `sampler_infer_t` at train time; and un-shadow `--sampler-steps` (the forward's `steps` argument currently always wins) | item 1's DD fidelity; **any** future sampler ladder rung |
| **E7** | `stack/scripts/refc_v3_train.py` (`_pin_refcv5_seams`) | ⛔ the `--agents head` guard's remedy text **recommends `--agents oracle`**, which is **unrunnable** — the trainer never passes `agent_gt`. Either wire `agent_gt` for oracle **or** delete the recommendation and refuse `oracle` at pin time. ⚠️ A guard whose remedy does not work sends the next reader down a dead end | item 2's honesty; nothing in this arm |
| **E8** | `stack/scripts/refc_v3_train.py` | `tac_vocab_version` is a **hardcoded literal** (`"v7.0" if --v7-labels else "kin3"`) and never propagates the version `v7_labels.py` already validates from the blob ⇒ a v7.2 corpus is stamped `v7.0` | provenance on **every** v7 arm, incl. refcv5's banked record |
| **E9** | Benchmarks · `…/2026-09-06-refcv4b-navpred/` | ⛔ **`C3_os_reproduction` is a FAILING control on a BANKED artifact** — `measured 0.2965 / banked 0.2975 / abs_diff 0.001031 / tolerance 0.001` ⇒ `FAIL`. Two dumps of one checkpoint on one window grid disagree by 3 % over tolerance. Resolve (re-roll determinism, or a grid difference) rather than carry | **any** cross-dump read; `os_navpred` exists only in the failing dump |
| **E10** | register row `D-REFCV5-LEVERS` #5 | ⛔ **the `0.1669 nats` figure MIXES SOURCES AND STEPS** — floor from the **eval dump @40,284**, CE from the **trainer log @31,500** (INHERITED, forbidden by CLAUDE.md rule 1 for a decision-grade number). A third value (**0.1126**) is still live in the same package's `STATUS.md`. Re-derive both halves from one eval dump at one step | item 10's committed prediction |
| **E11** | register `H-NAVPRED-1` vs `flip_analysis.json` | ⚠️ **two decompositions of one quantity** — **95.7 / 4.3** vs **97.7 / 2.3** presence-vs-content. Name the arithmetic and retire one | item 4's central argument |
| **E12** | ⚠️ propagated in **four** places — `H-R4B-1`, `…/goal-point/RESULT.md` §5, `PREREG_R4B.md` §0, and the redesign brief | ⛔ **"the refinement HALVES ADE" is an overstatement.** MEASURED: anchor **0.4281** → emitted **0.2965** = **−30.7 %**, not −50 %. (The *"doubles curvature"* half is correct: 0.004019 → 0.008150 = **2.03×**.) Correct all four | anything quoting R4b's trade |

---

## 9. ⛔ THE ONE LINE, COMMITTED BEFORE THE DATA EXISTS

> **refcv5 moved ONE mechanism that is on none of the programme's validated-lever lists,
> and by `D-REFCV5-PLAN-10`'s arithmetic could not have cleared its bar. `refcv5-cap`
> carries that mechanism forward and adds the two highest-ranked levers that are actually
> unblocked — the predicted geometric goal point (whose entire marginal value is RANGE, and
> range is LONGITUDINAL) and, the moment its join exists, agent conditioning (which is
> simultaneously the PI's environment question and the closing-rate REPRESENTATION that
> rank 1 demands). It is a CAPABILITY claim; per-element attribution comes from L1–L6, and
> until they run no element in it may be quoted as worth anything. If it clears `ha0_ext`
> it is the first REF-C arm that drives; if it only beats refcv4b it is
> LEVER-SUPPORTED / NOT-DRIVING and the report will say exactly that.**
