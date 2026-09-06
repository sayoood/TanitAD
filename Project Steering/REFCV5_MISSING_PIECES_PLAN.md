# refcv5 — the missing-pieces plan: VALIDATE, then COMPOSE

**Author:** TanitAD Master Mind · **Date:** 2026-09-06 · **Status:** ACTIVE
**Authority:** PI directive, 2026-09-06 — *"develop a plan and implement it and validate it to build
all missing pieces for refcv5. It is ok not to follow the one variable rule, but it is important to
validate the improvements to be sure that they will bring benefit. And please assure that all the
validated measures are included. Please use the whole tactical and strategic vocab."*

---

## 0. The shape this directive implies

Two constraints, and together they settle the structure:

* ⭐ **"OK not to follow the one-variable rule"** ⇒ the launched arm is **multi-lever**. Attribution is
  not the deliverable; capability is.
* ⛔ **"Validate the improvements to be sure they will bring benefit"** ⇒ each lever is **shown to
  help BEFORE it enters the arm.** Post-hoc attribution is not validation.

⇒ **VALIDATE-THEN-COMPOSE, not compose-then-attribute.** A lever earns its place on a cheap rig with
a pre-registered bar; only the survivors are composed. This is stricter than the one-variable rule in
the part that matters (each lever is evidenced) and looser in the part the PI released (the final arm
carries many at once).

⚠️ **Why this is not the ablation-ladder plan it replaces.** A post-hoc ladder tells you what a lever
was worth *after* you have spent the GPU-days. It cannot stop a harmful lever from entering. MEASURED
precedent: `--sel-refined` is **0.0259 m separated WORSE** [n = 171 windows / 20 episodes], flipping
29.82 % of picks and lowering **both** minority recalls — a lever that *looks* like an improvement,
is wired and available, and would have degraded any arm that adopted it unvalidated.

### 0.1 The two strategic instructions, reconciled

The PI has said both *"remove the strategic layer in the next experiments … include a flag to ignore
it"* (2026-09-06) and *"use the whole tactical and strategic vocab"* (2026-09-06, later).

⇒ **These address different objects and are both honoured:**

| object | disposition |
|---|---|
| the strategic **VOCABULARY** (`STRATEGIC_GOAL_TOKENS_V7` 8 + `STRATEGIC_ACTION_TOKENS_V7` 7 = **15 tokens**, coverage 4,572/4,572, **0 hits in the live trainer**) | ⭐ **SUPERVISED.** Heads + losses, extracted from vision + measured ego. |
| the strategic **LAYER's conditioning path** (`ctx`→decoder, E4 FiLM on `z_tac`, route_prior→selection) | ⭐ **SWITCHABLE** via `--no-strategic` (built, mutation-proven both ways, commit `bfcdd758b`). |

⇒ **The vocabulary is trained either way; the ablation answers whether the conditioning helps.**
⛔ If the PI intended the conditioning path restored unconditionally, that is a one-word correction and
this document is wrong in exactly one row — flagged deliberately so it is cheap to fix.

---

## 1. The missing-piece register

Every item is IN the plan. The column that matters is **how it earns entry**.

| # | piece | state today | validation rig | owner |
|---|---|---|---|---|
| **P1** | **Agent / environment conditioning** — DD's waypoint-indexed spatial cross-attention | ⛔ `--agents off`; `AGENT_WEIGHT_DEFAULT = 0.0` skips every term, no graph built | B1 TRAIN join → tiny-rig arm | join stream |
| **P2** | **Geometric goal point** (E15) | ✅ wired + gated; ⛔ `GOAL_POINT_WEIGHT_DEFAULT = 0.0` | banked dumps (done) + tiny rig | goal-point stream |
| **P3** | **22-token TACTICAL vocabulary** | ⛔ not in `v7_labels.HEADS`; `audit`-only | head + loss, tiny rig | label-wiring stream |
| **P4** | **15-token STRATEGIC vocabulary** | ⛔ 0 hits in the live trainer | head + loss, tiny rig | **UNOWNED** |
| **P5** | **Decoder refinement fix (R4b)** | ✅ measured on a held-out split | eval-time, done | goal-point stream |
| **P6** | **Selector training (WP-7) + DD deep supervision** | ⛔ module built (4,530,444 params), **wired to nothing**; trainer absent | blocked by P12 | **UNOWNED** |
| **P7** | **Distance-keeping cost** | ✅ built on refav1; ⛔ not ported to REF-C | banked dumps | **UNOWNED** |
| **P8** | **`--no-strategic`** | ✅ built, mutation-proven both directions | its own arm | done |
| **P9** | **RL post-training (DD-v2)** | ⛔ gated on the anchored `S6_wmr` bar | — | RL stream |
| **P10** | **`g_str` sign fix** | ⛔ lateral negative on 4,823/4,823 windows | sign histogram + recall | steering stream |
| **P11** | **Nav args (`distance_m`, `time_s`)** | ⛔ written on every TURN token, never read by REF-C | loader change | steering stream |
| **P12** | **`a_star` geometry** | ⛔ ceiling reads **below its own floor** | ceiling assertion | a_star stream |
| **P13** | **DD's `t ~ U[0,50)` training draw** | ⛔ **no consumer** — flag stamped, trained at `t = 8` | tiny rig | **UNOWNED** |
| **P14** | **Sampler ranks the fan** | ⛔ ranks with a score computed **before any sample was drawn**; **>2× worse than fan-best on 41.09 %** of windows | banked fan | **UNOWNED** |
| **P15** | **Max-speed constraint** | ⛔ no admissible source yet | source search | speed-limit stream |

---

## 2. The validation contract — what "brings benefit" means

⛔ **Every piece carries all six of these before it may enter the composed arm.** No exceptions, and a
piece that cannot be validated is reported as **UNVALIDATED — carried on the PI's authority**, never
silently included.

1. **A pre-registered bar**, written before the number exists, with **both outcomes committed**.
2. ⛔ **A deliberate-regression arm.** If the gate cannot FAIL a knowingly-broken version, a PASS
   means nothing. *(An AST census once read 0 suspects on BOTH the fixed and the broken trainer.)*
3. ⛔ **Controls that must read a KNOWN value** — a constant-only control at the no-information value,
   a raw-input floor, and `n` and `d` printed. *(Four estimator failures in one afternoon each produced
   a confident, publishable-looking number; every one was caught only by a control.)*
4. ⛔ **Four metric families**, never pooled, never ADE alone. **LATERAL is read on curvature MAE with
   the straight-line floor beside it**, because an arm can win ADE while tracking the road worse than a
   plan that never steers.
5. ⛔ **The variance the interval answers, named.** Episode draw / training run / inference run / rig.
   A one-seed separated CI is **necessary, not sufficient** — measured replicate false-positive rate
   **6/42 = 14.3 %**.
6. ⛔ **A vacuity gate.** A safety zero means nothing if the arm never approaches the limit — and
   MEASURED, one zero-violation result turned out to be **bought by a `turn_left` recall of exactly
   0.0000**. **Report the manoeuvre rate beside every constraint-satisfaction number.**

### 2.1 Where validation runs

⭐ **The v7-tiny ladder is the rig** — v6's real trainer at ~19 M params, parity corpus, **~17 min/arm
on Thor**. ⛔ Never validate a design on a full-scale run. Where a piece can be settled on **banked
dumps at zero GPU**, that is preferred and several already have been.

⚠️ **Tiny-rig results carry the rig's own noise floor.** MEASURED: a **zero-lever replicate** produced
"separated" differences on **6 of 42** family cells. ⇒ **A tiny-rig PASS is entry to the composed arm,
not a published result.**

---

## 3. Phases

### Phase 1 — VALIDATE (parallel, cheap)
Each piece runs its own pre-registered validation. **Only PASSes proceed.** A FAIL is banked with its
next lever named — ⛔ **and a FAIL does not end the piece**: RULE ZERO requires the next lever be run,
not that the piece be dropped.

### Phase 2 — COMPOSE
The validated set is assembled into **`refcv5-cap-b1-v72-40k`** and launched on the A40.
⛔ **Every element's flag and effective weight is stamped into `config.json`** — because an operator
can currently pass a weight that a stage silently zeroes (`V6LossWeights.for_stage`), and five weights
in this trainer default to 0.0 and are hard-guarded so the term never builds a graph.

### Phase 3 — CONFIRM
The ablation ladder runs at eval time on the one checkpoint, to confirm that composition did not
break what was individually validated. ⛔ **This is a check on composition, not the validation itself.**

---

## 4. The gate on Phase 2

⛔ **The arm does not launch until P1 (agent conditioning) is either IN or explicitly declared OUT by
the PI.** It is the piece the PI named, and DD's own ablation prices it at **88.1 → 55.1 PDMS** for
removal, against **88.1 → 87.9** for the denoising-step count that refcv5 spent 45 h on.
⚠️ Those magnitudes are NAVSIM-with-LiDAR and do not transfer; **the ranking does**, and it agrees
with our own measured ranking.

⭐ **P1 does double duty:** the join computes **`v_rel_x`** — relative velocity — which is the
**closing rate** a frozen-trunk probe measured as a **clean null on every arm** (+0.0061; an explicit
temporal difference recovered nothing) and named as a **representation** defect. Agent conditioning is
simultaneously the environment link and that missing representation.

---

## 5. What this plan will not do

* ⛔ **Enable a zero-weight default silently.** Flipping one changes the recipe every banked arm was
  trained under, and those arms are the comparison basis. Each is a pre-registered arm with its cost.
* ⛔ **Wire a label without a quality filter.** Of 779 traffic-light records only **175 are
  `grounded`; 604 are `disputed`**. Supervising on disputed labels teaches noise.
* ⛔ **Feed a label at inference.** Alpamayo GT **supervises heads that extract from vision + measured
  ego**. The exceptions are the **nav command** — *"NOT a training signal; an INPUT simulating the nav
  system of the vehicle"* — and measured `v0`.
* ⛔ **Quote a contaminated metric.** `oracle_sel` / `anchor_acc` / `sel_agrees_oracle` stay INVALID
  until P12 lands; both refcv4b number families stay unquotable until `C3_os_reproduction` passes.

---

## 6. The honest ceiling on all of it

⛔ **88.13 % of the usable horizon carries no v7 tactical GT** — one record per clip, `t0_s = 8.0`,
valid over ±2.0 s. Wiring the tactical and strategic vocabularies onto **12 %** of frames still leaves
the rest unsupervised.

⇒ ⭐ **If that band is a WRITER limit rather than a SOURCE limit, regenerating labels is worth more
than every head in this plan.** That determination is in flight and gates how much P3 and P4 can
deliver. It is stated here rather than discovered later.

---

## 7. STATUS — 2026-09-06 late, appended (the register above is the plan; this is what it has met)

⛔ **Appended, never rewritten.** §1's table states the plan; this states the outcome. Where they
disagree, this section is later and wins — and the disagreements are named, not smoothed.

### 7.1 Resolved

| # | piece | outcome |
|---|---|---|
| **P5** | decoder refinement (R4b) | ⭐ **VALIDATED on a held-out split**, λ chosen on a disjoint half: curvature **−26 % separated**, ADE **not** separated (nothing given back), speed better, cost **0.76 mm**. ⚠️ *"halves ADE"* was an overstatement propagated into four artifacts — it is **−30.7 %**; the 2.03× curvature half is correct. |
| **P8** | `--no-strategic` | ⭐ **BUILT**, zero new parameters. ON changes **0 of 28** downstream surfaces; OFF moves **20 of 28**; OFF is **byte-identical** over 341 tensors. Strict load 0/0 on a banked checkpoint, with `hierarchy=False` (the delete route) still **failing** as the negative control. |
| **P11** | nav args | ⭐ **MEASURED, and it is a loader change.** `NAV_ARG_SLOTS = ("distance_m","time_s")` is written on every TURN token — median **27.3 m / 7.2 s** (L), **36.6 m / 7.4 s** (R). **v6/v7f FEEDS both; refav1 DISCARDS; refcv3/refcv4b NEVER READS.** ⛔ Three traps: `NAV_FOLLOW_ROAD` carries `args: {}` on **2,897/2,897** train, **69.97 %** of train records present `distance_m = 0.0` under the consumer's default, and the field needs an explicit validity channel. |
| **P12** | `a_star` geometry | ⭐ **FIXED — the ceiling is a ceiling again**: best-in-fan **0.1993 m** under `os` **0.2970 m**, where the defect had it 4× above. ⛔ **Broader than first scoped: `a_star` is consumed on EVERY roll, so all past refcv4b `anchor_acc` and `sel_agrees_oracle` are contaminated**, not only `--with-oracle-sel` rolls. ⚠️ And it is an **empirical** ceiling, never a bound — `a_star` minimises SSE while ADE is a different norm. |
| **P15** | max speed | ⛔ **NO ADMISSIBLE SOURCE.** `road_class` explains 57.9 % of speed variance against a 3.1 % shuffle floor **but is defined by a speed threshold** — circular, and the shuffle control is structurally blind to it. The CoT carries **57/4,729** posted limits with a number, **0 grounded**, ego-coupled at median ratio **1.01**. ⛔ **The PI has ruled out re-asking Alpamayo.** Under-driving stays unscoreable; over-driving and agent clearance do not. |

### 7.2 New, and not in §1 because they were found after it

* ⭐⭐ **The effective-weight guard LANDED** — both live trainers now refuse a weight the operator typed that a later layer zeroes, and stamp `default → layer → effective → builds-a-graph` into `config.json`. ⛔ It corrected this plan's own premise: **S-T zeroes 10 terms, not 9** (`w_s1_multi`), and **S-S zeroes 14** — a set never enumerated. **"Explicit" is read from argv against sentinel defaults, never from the value**, because `--w-o5 1.0` *is* the default. ⚠️ And `refc_v3_train.main` runs `preflight` **only under `--preflight`** — a guard there alone covers **one launch path of two**.
* ⛔⛔ **A BLOCKING REGRESSION, introduced this evening.** `refc_v3.py` builds `tac_goal_tok_head` unconditionally under any non-`kin3` vocabulary ⇒ a rebuild carries **11,286 params absent from the recorded `param_breakdown`** ⇒ **refcv4b and all three local refcv3 checkpoints are currently UNROLLABLE.** The head is provably inert (`tac_goal_logits` written once, read nowhere across four files). **Fix in flight; this gates all eval work.**
* ⛔ **`SPEED_BAND` is supervised DEGENERATELY** by the new tactical-goal head: present on **4,572/4,572** and inside `_MEASURED_GEOMETRY_TOKENS`, so every cell gets **weight 1.0 and target 1.0** — a fully-weighted constant-1 BCE target with **zero negatives**. It cannot teach; it can only saturate a logit while contributing loss.
* ⚠️ **The session's load-bearing eval dumps live in a TEMP DIRECTORY** — `refcv4b_t1_dump` and `refcv3_40284_dump` (141 episodes / 4,823 windows each) are in the session scratchpad, not the repo. Banking in flight. *"An artifact on one disk is NOT done"*, and a scratchpad is worse than a disk.
* ⭐ **The two-segment anchor extension is a PORT, not a design.** `refa_v1.py` already defines `GOAL_LANE_CHANGE = (2.0, 1.75)` and emits it as an S-curve. Open parameter is only the split points: **2/3/4 s is worth 0.1645 m against 0.0304 m** for a single 3 s split. ⇒ The two lines have **complementary** defects — refav1's S-curves have zero net heading change; refcv4b has all sustained arcs and no S-curves. **Both shapes are needed.**

### 7.3 ⛔ Corrections to numbers this plan or its briefs carried

* **`os` / `os_navzero`**: the registry's **A40 landing roll (0.2975 / 0.3928) IS quotable** — its `C3_os_reproduction` selftest **PASSES** at abs_diff **2.6e-05**. The **Thor re-roll (0.2965 / 0.3926)** misses by 0.001031, from cross-hardware argmax tie-breaks, with `ha`/`ha0` reproducing at **0.000000**. ⇒ **A number carries its ROLL.** Earlier statements that *both* families were unquotable were too strong.
* **`--sel-refined`**: **n = 171 windows / 20 episodes**, not the 4,823-window grid -- **and on
  refcv4b `ckpt_30000`, NOT the landing `ckpt_40284`.** Its surface's model-free arms read `ha`
  **0.2860** / `ha0` **0.6542** / `ha0_ext` **0.2769** -- DIFFERENT WINDOWS from the landing table's
  0.2996 / 0.6723 / 0.2874. Delta 0.0259 separated worse, 51/171 = **29.82 %** flipped.
  => Admissible as a **DIRECTION** (do not enable the flag); NOT comparable in magnitude to
  any landing number. **Stamp the checkpoint AND the surface, not just the n.**
* **The "refcv4b beats refcv3 by 35.1 %" figure** is the same `ckpt_30000` panel. Only the landing
  read's **0.1444 [0.1647, 0.1227] separated better** at step 40,284 belongs beside the landing ADEs.
* **`g_str` removal vs value ratio**: **41.0×**, not 40.9× — the earlier figure divided unrounded values while quoting rounded ones, and the ratio is normalisation-dependent.
* **The 11th obstacle class** is `train_or_tram_car`, **not** `protruding_object` (always in the canonical ten). ⇒ **No infrastructure clearance is measurable** — `protruding_object` is overhead (bottom face +1.68 m) and the join drops `z`.
* **`traffic_light_visible = False`** elsewhere is **not-probed, not absent** — the question was asked on **998/4,572** clips.

### 7.4 ⛔ Standing PI constraints, binding on every stream

* ⛔ **No label generation.** *"Don't generate labels again."* The K=9 regeneration was stopped and will not restart without authorisation — even though it was priced at **0.5 min for 100 % band coverage**, because the instruction is the instruction.
* ⛔ **No Alpamayo re-ask.**
* ⛔ **The nav command is NOT a training signal** — it is an INPUT simulating the vehicle's nav system.
* ⭐ **Use the whole tactical AND strategic vocabulary** — both are supervised; the strategic *layer's* conditioning stays switchable.
* ⛔ **The arm does not launch until agent conditioning is IN or explicitly declared OUT.**
