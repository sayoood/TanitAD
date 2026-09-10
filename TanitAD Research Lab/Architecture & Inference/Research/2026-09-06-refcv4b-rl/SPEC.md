# SPEC — `E-DDA-3c`: RL post-training on refcv4b, read on the FAN'S FLOOR

**status: PRE-REGISTERED, NOT RUN.** ⛔ **No arm has executed. No result exists.**
Both GPUs are occupied (a sibling's refcv5 ~44 h training on the A40; an eval on Thor)
and this SPEC does not touch either.

**owner:** RL post-training stream (Arch + Inference FlyWheel) · **date:** 2026-09-06
**schema:** `TanitAD_ValidateAIDesign` §1 · **hypothesis:** `H-DDA-5` (OPEN) ·
**vehicle:** `E-DDA-3c`, the arms of `E-DDA-3b` re-based on refcv4b and read on the
primary readout that did not exist when `E-DDA-3b` was written.

⭐ **THIS SPEC EXTENDS, IT DOES NOT DUPLICATE.**
`…/2026-09-05-rl-posttrain-refcv4b-refcv5/SPEC.md` already pre-registered the policy,
the parameter set, the arms and the exit order. **Nothing in that document is
re-litigated here.** This SPEC adds exactly the three things that were named as
blockers there and were still missing on 2026-09-06:

| what `E-DDA-3b` left open | what this SPEC adds |
|---|---|
| **B2** — `fan_floor@k` / fan-collision / diversity **do not exist**, *"a hard blocker on the primary readout [that] must land before any arm"* | ⭐ **the instrument, built, tested 31/31, and run on a real 240 × 128 banked fan** (§2) |
| the PRIMARY endpoint had no committed numeric bar | ⭐ **PASS/FAIL written with both outcomes, on the metric the instrument now emits** (§5) |
| the deliberate-regression arms were pre-registered but never shown to be **failable** | ⭐ **MEASURED: the collapse control fires on both halves** (§3) |

---

## 1. The one variable

**The frozen refcv4b fan → a head-only, two-scalar SCALE policy over it,
post-trained by the DD-v2-shaped estimator.** Everything else — checkpoint, corpus,
window grid, conditioning, selector, eval geometry — is held fixed, and the arms in
§4 exist to prove that it was.

**Base:** refcv4b FINAL. MEASURED (`b01cc14`): refcv4b − refcv3 **−0.1444 m
[−0.1647, −0.1227] separated**; `os` **0.2975**, `ha0_ext` **0.2874**, `ha` **0.2996**,
`ha0` **0.6723**, `os_navzero` **0.3928**.
⛔ `oracle_sel` and `anchor_acc` are **INVALID on refcv4b** and appear nowhere in this
SPEC — not as an endpoint, not as a reward, not as a diagnostic.
⛔ `--sel-refined` is **NOT SET** on any arm: MEASURED **0.0259 m separated WORSE**.

---

## 2. ⭐ THE PRIMARY READOUT, BUILT THIS TURN — and what DD-v2 actually publishes

### 2.1 The published mechanism, from the banked primary

Banked: `2512.07745` — *DiffusionDriveV2: Reinforcement Learning-Constrained Truncated
Diffusion Modeling in End-to-End Autonomous Driving*, sha256
`076ce47e0b0a323c9bb0072432e7f1219ccdcf645b8ff6db172f1bb9efb85023`,
`TanitAD Research Lab/Library/papers/2512.07745_DiffusionDriveV2-…pdf`, tags
`diffusiondrive · refc-lineage · rl-posttrain`. Read together with the pinned code at
`1cd12a1` (`…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/`, sha256s in
`raw/ddv2_fetched_sha256.txt`). Both are `PUBLISHED-PRIMARY` / `PUBLISHED-CODE`.

**Exactly what its reward is computed over** (`PUBLISHED-CODE`; the paper never defines
`R(τ₀)` in 17 pages, a `MEASURED-ABSENCE` the code closes): NAVSIM's **PDM score**, from
`PDMSimulator` + `PDMScorer` run in a 16-process pool over each candidate's lzma-pickled
metric cache — a **4 s LQR-tracked kinematic rollout at 10 Hz against REPLAYED agents**.
`_pairwise_scores` rebuilds it per candidate with the GT as proposal 0:
`final = NC × DAC × (5·EP + 5·TTC + 2·C + 0·DDC) / 12`, `EP` re-normalised as
`progress_i / max(progress_GT, progress_i)`. The `collision` predicate of Eq. 10 is
`NC × DAC` — **half of it is map-dependent**.

**Exactly what its policy gradient flows through:** the DDIM mean. Two forward passes per
batch — a `no_grad` rollout that samples, scores and builds advantages, then `get_rlloss`
re-running the decoder **with gradient on the stored chain**; `per_token_loss =
−exp(logp − logp.detach()) · A`, so the ratio is identically 1 and this is **plain
REINFORCE with one update per rollout — no importance ratio, no clip, no reference policy,
no KL** (grep of the released file finds none). `log_prob` is an isotropic Gaussian summed
over the 8 × 2 coordinates, and the gradient reaches the decoder through `μ` at each of the
**10** rollout steps, γ = 0.8.

⭐⭐ **AND THE FAN IT EXPLORES WITH IS DETERMINISTIC-PLUS-TWO-SCALARS.** `std_dev_t_add =
0.0` in both branches: the additive DDPM noise the paper's "η = 1" describes is generated
and **multiplied by zero**. The only stochasticity is **two multiplicative scalars per
trajectory** (longitudinal, lateral) at a σ floor of **0.04**, likelihood σ **0.10**.
⇒ **A deterministic fan is not a disqualification** — that is the finding that makes this
arm possible on refcv4b at all, and it is why `M52`'s *"refcv3 has no denoiser to
post-train"* does **not** transfer.

**The published effect is on the FAN, not the pick** (Tab. 3, 20 trajectories, before any
selector): PDMS **@1 93.5 → 94.9 (+1.4)**, **@5 84.3 → 91.1**, **@10 75.3 → 84.4
(+9.1)**, **diversity 42.3 → 30.3 (−28 %)**. ⇒ the gain is **6.5× larger at the fan's
FLOOR than at its TOP**, and it is bought partly by narrowing the fan.

### 2.2 The instrument (`B2`, closed)

⛔ `fan_floor@k`, fan-collision-vs-replay and diversity **did not exist**. Re-verified
ABSENT 2026-09-06 by a content-verified scan of **1,180 `.py`/`.md` files under `stack/`
and `taniteval/`, 0 unreadable** — an absence claim about content that was READ.

**Built, landed and tested this turn:**
`stack/tanitad/rl/fan_floor.py` · `stack/scripts/rl_fan_floor.py` ·
`stack/tests/test_rl_fan_floor.py` (**31/31 green**; 106/106 across the RL suite).

⛔⛔ **AND IT CARRIES THE TWO CONTROLS THE PROGRAMME HAS ALREADY BEEN BURNED WITHOUT.**

1. **`fan_floor@k` IS AN ORDER STATISTIC OVER K.** The 2026-09-05 retraction — *"the RL
   arm's objective is the 2.11× selection gap"*, root-cause class **A BEST-OF-N STATISTIC
   READ AS A SKILL GAP** — leaves a binding rule: any min/max-over-N quantity is reported
   beside the **same statistic from a RANDOM selector at the same N**. `summarise_fan`
   therefore computes `rand_best@k` = E[max over a random k-subset] **unconditionally**,
   and the two questions are documented as different and non-interchangeable.
   `assert_equal_k` **refuses** a paired comparison across fans of different K; `fan_quantile`
   is the K-free form for when K genuinely differs.
2. ⛔⛔ **`fan_floor@k` IS MAXIMISED BY MODE COLLAPSE.** `floor_verdict` **refuses to
   return a verdict** unless diversity was measured on the same windows, and emits
   `COLLAPSE-SUSPECT` when the floor rises while diversity falls >30 %. ⭐ DD-v2's own
   published stage sits in that band (−28 %) — which is exactly why it is a **named
   outcome**, not a failure: the trade must be *visible and priced*, never invisible.

---

## 3. ⭐ PROOF THE GATE CAN FAIL A KNOWINGLY-BROKEN POLICY — MEASURED, 0 GPU

⛔ A gate that cannot fail a deliberate regression cannot certify anything. This one was
run, not asserted, on the **real banked 240-window × 128-candidate refcv3 fan**
(`fan_bank_base_240w.npz`, `…/2026-09-05-veto-only-fan-safety/raw/`).

**The deliberate regression:** `collapse_fan(fan, 0.9)` — every candidate shrunk 90 %
toward the fan's own per-window mean path. A destroyed fan by construction.

| half of the control | MEASURED | reads |
|---|---|---|
| diversity | `d_diversity` **−4.089131 m**, rel **−0.9000** | **FIRED** |
| floor | ⭐ `d_fan_floor@128` **+9.614186** | **FIRED** |

⭐⭐ **READ THE SECOND ROW.** A fan that has been destroyed reports a **large POSITIVE
floor gain**. ⇒ **`fan_floor@k` quoted alone would have PASSED this arm with a wide
margin.** That is the measured justification for `floor_verdict`'s refusal, and it is why
§5's PASS text names diversity as a **co-condition and not a footnote**.

⚠️ **A DISCLOSED LIMIT OF THE CONTROL, found by running it.** On a bank whose
per-candidate quality columns are **precomputed**, collapsing the geometry cannot move
them, so the floor half reads **INERT** and only diversity fires. The driver now
**detects this and prints `PARTIAL` with the reason** instead of printing `OK`. The
complete control requires a quality recomputed from the waypoints
(`--quality geom_feasibility`), which is how the row above was produced.

⚠️ **A units bug of mine, caught by its own implausibility before it was quoted.** The
first run of that control used `rewards.kinematics`' default `DT_S = 0.1` on a fan banked
on the **0.5 s grid**, inflating every acceleration **25×** and reporting
`fan_floor@128 = −268.25`. The driver now derives `dt` from the waypoint count and
**states it in the provenance string**. Same family as the programme's `alat`/`kappa`
unit trap: a correct formula under the wrong units reads exactly like an answer.

---

## 4. Arms — one variable, every control reads a KNOWN value

Unchanged from `E-DDA-3b` §3 except the base and the primary readout. Listed so this SPEC
is self-contained; ⛔ **the arm set is not re-opened after seeing any number.**

| arm | reward | veto | lr | steps | what it MUST read |
|---|---|---|---|---|---|
| `base` | — | — | — | — | the banked refcv4b FINAL fan, RL OFF |
| **`rl`** | RANK + BAR | on | 1e-5 | 2,000 | the hypothesis |
| **`rl_s1`** | identical | on | 1e-5 | 2,000 | ⭐ **SEED REPLICATE, budgeted from the start.** `H-ESTIM-SEED-1`: a separated CI at one seed is **necessary, not sufficient** — a zero-lever replicate produced "separated" on **3 of 18** family metrics |
| **`veto_only`** | all 0.0 | **on** | 1e-5 | 2,000 | ⛔ **MANDATORY ATTRIBUTION.** The only refcv3 arm that ever moved the fan the right way (`fan_peak_g_mean` −0.0859 g). Without it a `rl` gain is not attributable to the reward |
| **`ctrl_null`** | all 0.0 | **off** | 1e-5 | 2,000 | advantage identically 0; `veto_rate` **exactly 0.0000** |
| **`dose_null`** | all 0.0 | off | 1e-5 | 2,000 | dose-matched drift control — same steps, same LR, no information |
| **`ctrl0`** | RANK + BAR | on | **0.0** | 200 | frozen weights ⇒ **every readout delta exactly 0** and the state-dict sha256 unchanged, or the panel is **VOID** |
| ⛔ **`reg_echo`** | `gt_similarity` 1.0 | — | 1e-5 | 2,000 | **DELIBERATE REGRESSION**: the ego's GT future inside the advantage. Must collapse the fan; if it does not, the panel is **VOID** |
| ⛔ **`reg_metre`** | RANK + BAR | on | 1e-5 | 2,000 | **DELIBERATE REGRESSION #2**: metre-space noise instead of control-space. Must **FAIL flyability** |
| ⛔ **`reg_collapse`** | — | — | — | — | ⭐ **NEW, and it costs no GPU**: `collapse_fan(base, 0.9)` scored as if it were an arm. **MEASURED to move `fan_floor@128` by +9.61 with diversity −0.90 rel** (§3). It must be classified `COLLAPSE-SUSPECT`, never `FLOOR-GAIN` |

⛔ **`rl` vs `veto_only` is the attribution.** `rl` vs `dose_null` is the drift control.

---

## 5. ⛔ PRE-REGISTERED PASS/FAIL — both outcomes committed BEFORE compute

**PRIMARY (T0, fan-level).** On the same eval windows, same K, paired episode-cluster
bootstrap over episodes (`taniteval/ci.py`), `n_boot` 4,000:

> ⭐ **PASS** requires **ALL FOUR**:
> 1. **`fan_floor@32` rises** vs `base`, CI excluding 0, **same sign at both seeds**;
> 2. **and the same is true at `fan_floor@64`** — the published shape is a FLOOR effect,
>    so a gain that exists only at `@1` is **not** this hypothesis;
> 3. **and `fan_diversity` has not fallen more than 30 %** — otherwise the outcome is
>    **`COLLAPSE-SUSPECT`, reported as such and NOT as a pass**, however large the floor
>    gain (§3: a destroyed fan scores +9.61);
> 4. **and `fan_collision_all` does not rise** (CI not excluding 0 upward).
>
> ⛔ **FAIL** is written now, in the same breath, and will be reported as written:
> * `fan_floor@32` **does not rise with a separated CI** ⇒ **FAIL-NULL**. The DD-v2 lever
>   does not transfer to a deterministic-fan REF-C under a map-less reward, and the named
>   next lever is `E-DDA-2b` (the selector arm), not another RL dose.
> * it rises **only at `@1`** ⇒ **FAIL-SHAPE**: we moved the top, not the floor, which is
>   the half DD-v2 did *not* claim.
> * it rises **and diversity collapses > 30 %** ⇒ **SPLIT / COLLAPSE-SUSPECT**: report the
>   trade with both numbers; ⛔ do **not** call it a pass.
> * the seeds **disagree in sign** ⇒ **FAIL-REPLICATE**, and one seed's separated CI is
>   explicitly not rescued by being separated.
> * `rl` is **indistinguishable from `veto_only`** ⇒ **FAIL-ATTRIB**: the gain is the
>   constraint, not the reward, and the product is the veto (`H-RL-VETO-1`).

**GUARD (T1, self-action OPEN loop — never "closed loop").** The four families
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC, **never pooled**, each with its estimator,
CI and n, on the same windows.

> ⛔ **A safety gain bought with driving quality is a FAILURE, not a split.** Specifically:
> * ⛔ **the arm must beat the trivial controls it is read against** — refcv4b only
>   **TIES** `ha0_ext` (0.2874) and `ha` (0.2996) at `os` 0.2975. **An RL arm that does not
>   beat `ha0_ext` and `ha` has not driven**, and no fan-floor number rescues that.
> * LATERAL is read on **curvature MAE with the straight-line floor beside it** — refcv4b's
>   `os` curvature **0.008097** is *worse* than `ha0`'s **0.006802** while its cross-track
>   is the best of any arm, so cross-track alone is inadmissible for this family.
> * `ha0` **0.6723** and `os_navzero` **0.3928** are reported unchanged as context.

**EXIT ORDER (mechanical, first match wins, never chosen past a fired gate):**
`VOID` (`ctrl0` moved · `reg_echo` did not collapse · `reg_metre` passed flyability ·
`reg_collapse` classified `FLOOR-GAIN`) → **FAIL-GUARD** → **FAIL-REPLICATE** →
**FAIL-ATTRIB** → **FAIL-SHAPE** → **PASS** → **SPLIT** → **FAIL-NULL**.

---

## 6. What the reward is computed from — and its COORDINATE EXPRESSIBILITY

### 6.1 Provenance, term by term

| channel | term | computed FROM | admissible? |
|---|---|---|---|
| **VETO** (pins at −1, outside the ranking) | collision · TTC-imminent · feasibility-envelope · comfort-envelope | candidate geometry + the **lead agent's own recorded track** (`obstacle.offline`, pod-side join) | ✅ a constraint PINS, it does not rank |
| **RANK** | `progress` · `headway` · ⭐ `robust_contact` | candidate geometry + other agents' recorded tracks | ✅ the environment, not the answer |
| **BAR** | ≥ GT mask | the human's **SCORE** under the same rule-based reward — never the human's **SHAPE** | ✅ a bar, not a target |
| **ANCHOR** | `w_anchor` trust region to the frozen policy + λ_IL | the frozen policy | ✅ attributed separately |

⛔ **NOTHING IN THE REWARD IS AN EGO-STATE METRIC OR A SITUATION-CLASSIFIER OUTPUT.**
`fan_floor.ECHO_QUALITY_KEYS` **refuses by name** `ade`, `fde`, `minade`, `minfde`,
`gt_similarity`, `gt_distance`, `path_match`, `speed_match`, `speed_profile`,
`nav_agreement`, `route_agreement`, `ep_ratio`; `rewards.FORBIDDEN_REWARD_INPUTS` covers
the selector-derived ones. The goal path and the situation path stay
**information-disjoint at inference**, and the RL stage adds **no inference-time input at
all** — it changes weights only, so the deployed conditioning is unchanged and the
vision-only rule is untouched.

⚠️ **The reward is scored at TRAIN time on recorded scenes with replayed agents.** Labels
may use ego; inference is vision-only. Privileged tracks are admissible here and are
never available to the deployed model.

### 6.2 ⭐ COORDINATE EXPRESSIBILITY, stated against `M84`

`M84` MEASURED that refcv4b-lineage latents decode the lead's **POSITION** (paired vs
pixels **+0.4145 [+0.2018, +0.6120]**, constant control exactly **+0.000000**) but that
its **CLOSING RATE is a clean null on every arm** (+0.0061; an explicit temporal
difference recovers nothing). ⇒ **a reward can express *"be at distance d"* and CANNOT
express *"stop closing"*.**

| term | the coordinate it is written in | expressible under `M84`? |
|---|---|---|
| `headway` | **lead POSITION** at each step — a distance | ✅ **yes** |
| `collision` (dt = 0) | **lead POSITION**, swept segment-to-segment | ✅ **yes** |
| ⭐ `robust_contact` | **lead POSITION under a TIME SHIFT of the lead's own recorded track** — still a position query, evaluated at several times and averaged | ✅ **yes — and this is the point.** It marginalises over timing error **without ever asking the model for a rate** |
| `progress` | ego's own along-track displacement vs `v0 · horizon` | ✅ yes (ego geometry) |
| ⛔ TTC-as-a-RANKING-term | **a RATE**: range ÷ closing speed | ⛔ **NOT reliably expressible.** Kept as a **VETO** (a threshold on a quantity that must merely be *crossed*), never as a graded ranking term whose ordering depends on a rate the latent does not carry |

⭐⭐ **This is why `robust_contact` is the RANK term and TTC is only a VETO.** A naive
design would have put a graded TTC in the ranking channel, where its ordering would rest
on exactly the coordinate `M84` measured to be absent. `robust_contact` reaches the same
*safety* question through **positions at shifted times**, which the latent does carry.
⛔ Any future term proposed for the RANK channel must be shown to be a **position query**,
or it inherits this defect.

⚠️ And it is where the headroom provably is: contact at `dt = 0` is **closed by
construction** (3.4277 % → **0.000000 %** at **+0.0000 m** ADE), but contact under lead
timing error is **monotone and knee-free** across −1.0…+2.0 s (0.034277 → 0.039811 at
−0.5 s → 0.066243 at −1.0 s). A knee can be widened away with a bigger margin; a smooth
monotone residual cannot. **That is what a learned policy is for.**

---

## 7. ⛔ BLOCKERS — named, with what unblocks each

| # | blocker | state on 2026-09-06 | unblocked by | GPU |
|---|---|---|---|---|
| **B2** | `fan_floor@k` / fan-collision / diversity do not exist | ✅ **CLOSED THIS TURN** — built, 31/31 tests, run on a real 240 × 128 fan, deliberate-regression control fires on both halves | — | 0 |
| **B4** | prefix tables are refcv3 literals | ✅ **CLOSED for refcv4b** (0 ranking surfaces trainable); refcv5 needs a typed ruling on `core.decoder.layers.{0,1}.agent_gate` | a PI/MM ruling **for refcv5 only** | 0 |
| **B1** | ⛔⛔ **`G-REWARD` is UNREACHABLE as written** — min **0.3840** over 2,400 admissible weightings vs a ≤ 0.30 ceiling; its population is ~99 % windows with **no safety information at all**, corroborated from the other side by the GRPO advantage being **identically zero in 92 % of windows** | ⛔ **OPEN — PI / Master Mind.** This is the gate that stops the launch | a ruling on the gate's **POPULATION** (§7.1) | 0 |
| **B5** | both GPUs busy | ⛔ **OPEN** — A40 on refcv5's ~44 h run, Thor on an eval | a free slot; ~30 min/arm on the dev-box 4060 | queue |
| **B6** | no **refcv4b FAN dump** exists | ⛔ **OPEN** — the banked refcv4b artifacts carry the SELECTED paths only (`g`, `os`, `ha`, `ha0`, …), not the 128-candidate fan | one inference pass with the fan banked (§8 step 1) | ~10 min |

### 7.1 ⭐ The `G-REWARD` escalation, restated so it can be decided

⛔ **I am not moving this goalpost and I am not launching past it.** The measured facts:
the ≤ 30 % ceiling is **unreachable by every admissible weighting** (min 0.3840 / 2,400
tried); the population it scores is **98.97 % windows in which no safety term differs at
all**; the same sparsity appears from the other direction as a **92 % all-zero GRPO
advantage**. On the signal-bearing sub-population the repaired reward reads **0.0635
[0.0000, 0.3158]** — but that is **n = 63 windows / 6 EPISODES**, **post-hoc**, and its
upper bound **straddles the ceiling**. ⇒ **necessary, not sufficient**, and it may not
become the gate without its own pre-registration on a lead-dense corpus.

⭐⭐ **AND THE QUESTION HAS CHANGED, because I ran the next lever rather than stopping at
the escalation.** MEASURED 2026-09-06, 0 GPU, on a **pre-registered scene-property ladder**
(`RESULT.md` §5.1; ladder fixed before any rate was computed; every rung reported; three
channel controls exactly **0.000e+00** against the banked panel):

* ⛔ **the "restrict the population" repair is REFUTED, and in the opposite direction** —
  the rate **RISES** with conflict, **0.4418 → 0.5482 at ≤ 2.0 s** (1,576 windows /
  **29 episodes**, 5× the banked signal population), with **every rung's CI entirely above
  the 0.30 ceiling**;
* ⭐⭐ **but at five of six rungs the MEAN GAP is SEPARATED IN THE HUMAN'S FAVOUR**
  (all-windows **−0.011239 [−0.017039, −0.005306]**). **The reward does prefer the human;
  the RATE cannot see it**, because the distribution is skewed — the human wins big on a
  minority of windows and loses small on a majority.

⇒ ⛔⛔ **`G-REWARD`'s defect is its STATISTIC, not its population and not its threshold**,
and that is also why 2,400 weightings could not reach 0.30: no reweighting makes a rate see
a magnitude it is structurally blind to.

**What the PI / Master Mind must decide** (the REPLACED question): *should `G-REWARD`'s
statistic be the rate, the mean gap, or a magnitude-aware rank test — and should `progress`
be re-referenced so it stops paying the trivial path in close following?* (`progress`'
weighted mean gap **flips sign** with conflict: −0.005600 all-windows → **+0.008251** at
≤ 1.5 s.) ⛔ A gate whose passing region cannot be entered cannot gate — but a statistic
adopted **after** seeing which one passes is a moved goalpost, which is why this is
escalated rather than adopted.

---

## 8. The exact launch command

⛔ **Not run. Both GPUs are occupied and this SPEC does not touch either.**
⛔ **Step 0 is `B1`.** Steps 1–4 are otherwise runnable as written.

```bash
# 0. GATE. Do not proceed without the B1 ruling (§7.1).

# 1. Bank refcv4b's FAN (B6). ~10 min on the dev-box 4060 once free.
PYTHONPATH=stack OMP_NUM_THREADS=6 python taniteval/tools/fan_safety.py \
    --dump --arm refcv4b --ckpt <refcv4b FINAL> \
    --bank-fan raw/fan_bank_refcv4b_base.npz

# 2. The PRIMARY readout on the base fan, with its own control. 0 GPU.
PYTHONPATH=stack python stack/scripts/rl_fan_floor.py \
    --bank raw/fan_bank_refcv4b_base.npz --quality composed \
    --ks 1,5,8,10,32,64,128 --self-test \
    --out raw/fanfloor_refcv4b_base.json
PYTHONPATH=stack python stack/scripts/rl_fan_floor.py \
    --bank raw/fan_bank_refcv4b_base.npz --quality geom_feasibility \
    --self-test --out raw/fanfloor_refcv4b_base_control.json   # both halves must FIRE

# 3. The arms (§4). ~30 min each on the 4060; run them one at a time.
#    ⚠️ FLAGS BELOW ARE THE LAUNCHER'S REAL ONES (content-verified 2026-09-06,
#    74,437 bytes read, 26 flags). `--mode` is REQUIRED; the model is rebuilt from
#    --config, so there is NO `--model`; per-arm reward/bar/noise settings live in
#    the script's own ARMS table, not on the command line.
for ARM in rl rl_s1 veto_only ctrl_null dose_null ctrl0 reg_echo reg_metre; do
  PYTHONPATH=stack OMP_NUM_THREADS=6 python stack/scripts/rl_refcv3_min.py \
      --mode arm --arm "$ARM" \
      --ckpt <refcv4b FINAL> --config <refcv4b config.json> \
      --episodes <RL-fit v2ep dir> --labels <v7.2 TRAIN labels> \
      --lead-block <lead block> --lead-mode track \
      --group 4 --noise 0.04 --batch 2 --device cuda \
      --seed $([ "$ARM" = rl_s1 ] && echo 1 || echo 0) \
      --out-dir raw/run/$ARM
done

# 4. The paired readout, per arm, against the base. 0 GPU.
PYTHONPATH=stack python stack/scripts/rl_fan_floor.py \
    --bank raw/fan_bank_refcv4b_rl.npz --base raw/fan_bank_refcv4b_base.npz \
    --quality composed --ks 1,5,8,10,32,64,128 --n-boot 4000 \
    --out raw/fanfloor_paired_rl.json
```

### 8.1 ⛔⛔ WHAT STOPS STEP 3 TODAY — and it is bigger than a missing flag

⚠️ **CORRECTION, made against the source rather than the design.** An earlier draft of
this SPEC said the launcher was *"missing two flags"*. **MEASURED 2026-09-06 by reading the
shipped code:** the gap is a **missing CAPABILITY**.

* ⛔ **`refcv3_adapter.sample_offsets` HAS NO CONTROL SPACE.** It scales the **offset
  waypoints** (`scale = mean.abs() * cfg.noise_scale`), and the strings `control`,
  `rollout_unicycle`, `a_lon`, `alat` appear **ZERO times** in that file (content-verified
  read, 9,078 bytes, with a non-zero control on the same read). ⇒ **the arm that would run
  today is metre-space — i.e. `reg_metre`, the pre-registered deliberate regression** — and
  its result table would have read as the hypothesis. That is the single most dangerous
  thing in this package and it is why the SPEC now says it in the launch section.
* ⛔ `rl_refcv3_min.py` has **no `--bank-fan` / fan-dump path at all** (0 flags matching
  `fan`/`bank`/`dump`), so no arm emits the fan the primary readout consumes.

⭐ **WHAT I BUILT SO THIS IS ONE WIRING LINE RATHER THAN A DESIGN TASK:**
`stack/tanitad/rl/control_space.py` (**NEW**, `stack/tests/test_rl_control_space.py`
**15/15 green**) implements the DD-v2 two-scalar policy **in control space** — scale
`(accel, curvature)`, clamp to the envelope, re-roll through the programme's single
`rollout_unicycle` — with the published σ floors kept distinct (**exploration 0.04**,
**likelihood 0.10**, because they are different numbers in the released code and collapsing
them silently rescales the gradient).

⭐⭐ **AND ITS CENTRAL CLAIM IS TESTED AGAINST ITS OWN COUNTEREXAMPLE:**
`envelope_violation` reads **exactly 0.0** on a control-space sample **even at 25× the
published σ**, while the metre-space arm **does** violate — so *flyable by construction* is
a discriminating property and not a tautology. A property both arms satisfied would prove
nothing.

⛔ **STATED LIMIT, not hidden:** the clamp makes the sampler a non-injective pushforward, so
`logp` is the density of the **two scalars**, not of the emitted trajectory. This is the
**same class of object as the published estimator** — DD-v2 evaluates a 16-coordinate
isotropic Gaussian on samples drawn from a 2-scalar family — and it is a valid
**scale-perturbation** gradient, quoted as one and never as an exact policy gradient.

⇒ **ESCALATION (Training FlyWheel / Master Mind), now concrete:** wire
`control_space.sample_control_space` into the RL stage's sampling path behind a recorded
config field, and add a fan-dump path. ⛔ **This stream does not own `refcv3_adapter.py` or
`rl_refcv3_min.py`, so I did not edit them.** Until that wiring lands, ⛔ **step 3 must not
be run as the `rl` arm** — it would execute `reg_metre` under the hypothesis' name.

---

## 9. Manifest

| artifact | where it lives |
|---|---|
| this pre-registration | `…/2026-09-06-refcv4b-rl/SPEC.md` (repo) |
| the result note | `…/2026-09-06-refcv4b-rl/RESULT.md` (repo) |
| ⭐ the instrument | `stack/tanitad/rl/fan_floor.py` (repo) |
| ⭐ the driver | `stack/scripts/rl_fan_floor.py` (repo) |
| ⭐ its tests, 31/31 | `stack/tests/test_rl_fan_floor.py` (repo) |
| the measured refcv3 baseline + both control panels | `…/2026-09-06-refcv4b-rl/raw/*.json` (repo) |
| banked primary | `2512.07745`, sha256 `076ce47e…`, `TanitAD Research Lab/Library/papers/` |

Nothing lives in only one place. **No arm has run; no result exists.**
