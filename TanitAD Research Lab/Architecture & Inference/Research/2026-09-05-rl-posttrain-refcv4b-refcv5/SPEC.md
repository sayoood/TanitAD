# SPEC — RL post-training for refcv4b and refcv5 (`E-DDA-3b`)

**Stream:** Architecture & Inference · **Date:** 2026-09-05/06 · **Branch:** `agent/arch-inf-20260803`
**Status:** ⛔ **PRE-REGISTERED, NOT LAUNCHED.** Both outcomes committed below before any arm runs.
**GPU spent by this document:** 0.

```yaml
hypothesis: H-DDA-5              # exists: GOALS_AND_CLAIMS.md:2919 (OPEN, carries E-DDA-3b)
vehicle: E-DDA-3b                # the code-faithful scale policy, base = refcv4b FINAL
                                 # (D-REFCV5-PLAN-6, GOALS_AND_CLAIMS.md:3101)
one_variable: policy_update      # frozen fan  ->  head-only scale policy under the reward
held_constant: [base_checkpoint, corpus, split, seed, steps, batch, group_size,
                noise_mode, noise_scale, w_anchor, veto_channels, reward_weights]
success: >
  On the EVAL windows, paired episode-cluster bootstrap (n_boot 2000), the fan's
  ROBUST-CONTACT floor improves -- E[contact] over the timing-shift grid falls with a
  CI EXCLUDING ZERO on the fan and on top-32 -- AND no T1 family regresses past its
  guard (ADE lo <= +0.02 m, oracle-in-fan not worse, R_FAN >= 85 % of base),
  AND the effect REPLICATES on a second training seed with the same sign.
failure: >
  Any of: the robust-contact floor does not move with a separated CI; it moves but a
  T1 family breaches its guard; or it moves on one seed and not the other. Any of
  these CLOSES the composed-reward RL line on refcv4b and moves the lever to the
  vocabulary/sampler (H-DDA-3), exactly as the refcv3 campaign's verdict did.
controls: [zero_reward_null, dose_matched_null, veto_only, deliberate_regression,
           frozen_weight_ctrl0, seed_replicate]
splits: {fit: "120 train-split clips", val: "carved from FIT only",
         test: "141 EVAL clips -- scored, never tuned on"}
```

---

## 1. ⛔ THE BLOCKING PRECONDITION — `G-REWARD` — AND ITS MEASURED STATE

`D-REFCV5-PLAN-6` is explicit and it outranks this SPEC:

> **`G-REWARD` is a PRECONDITION, not a milestone**: hold-v0 must beat the human on
> **≤ 30 %** of lead windows … **A reward that fails G-REWARD may not train anything**

**MEASURED 2026-09-05/06** (`stack/scripts/rl_greward_gate.py`, 0 GPU, 6,089 lead windows /
73 episodes, re-weighting of the banked `humanflag_fit120.json` component table; channel
control reproduces the banked rate to **err = 0.0e+00**):

| spec | hold-v0 ≥ human | median gap | G-REWARD |
|---|---|---|---|
| `default` (the shipped `DEFAULT_WEIGHTS`) | **0.7249** | **+0.0115** | ⛔ FAIL |
| `veto_feas_comfort` (the plan's named repair) | **0.4418** | **−0.0024** | ⛔ FAIL |

⭐⭐ **AMENDED AFTER B1 (same turn, before any arm): `G-REWARD` AS WRITTEN CANNOT BE PASSED.**
MEASURED over **2,400 admissible weightings** of (`progress`, `headway`, `collision`,
`robust_contact`): **minimum reachable rate 0.3840** against a ≤ 0.30 ceiling. The cause is the
gate's POPULATION, not its threshold — the safety terms fire on **1.03 %** (`robust_contact`) and
**0.13 %** (`collision`) of windows, while `progress`/`comfort` fire on **95.70 %**, so the rate is
decided by windows carrying no safety information. Independently corroborated by
`D-RL-COLL-SPARSE-1` (*GRPO advantage identically zero in 92 % of windows*).
On the **signal-bearing** population the repaired reward **with `robust_contact`** reads
**0.0635 [0.0000, 0.3158]** vs the default's **0.8413** — ⚠️ **n = 63 windows / 6 EPISODES,
post-hoc**, so it is NECESSARY, NOT SUFFICIENT and may not become the gate until pre-registered on
its own. Full panel: `RESULT.md` §4.1–§4.3.

⇒ ⛔ **NO ARM IN THIS SPEC MAY LAUNCH UNTIL `G-REWARD` IS RE-SPECIFIED** (a PI / Master-Mind
decision — it is a committed gate, and this SPEC does not get to move its own goalpost). The rate
remains a *gate*, never the objective, and §2.3 says why it must not be gamed.

---

## 2. What is post-trained, and with what

### 2.1 The parameters — head-only, trunk frozen

`select_trainable(freeze_trunk=True, trainable_prefixes=("core.decoder",))`, forbidden
prefixes covering **every ranking surface**, re-derived from `model.named_parameters()` on
the real refcv4b build and pinned by a test (⚠️ the shipped list is nine *refcv3* selector
names; refcv5's agent branch adds surfaces inside `core.decoder` that are not on it —
`S2` in the audit). Registered trainable fraction: **9,206,032 / 107,032,901 = 8.60 %**.
This matches DDv2's own stage I, which trains `_trajectory_head` only and puts every other
module in `.eval()` (`_rl_agent.py:61-67`, PUBLISHED-CODE).

### 2.2 The policy — a two-scalar scale family, and it needs NO sampler

⭐ **The finding that unblocks refcv4b.** DDv2's *released* RL stage does not use the
diffusion chain's randomness at all: the additive DDPM term is **multiplied by zero**
(`_model_rl.py:640,643,666`), so the only stochasticity is **two multiplicative scalars per
trajectory** — a (stretch-along, stretch-lateral) family at a constant **4 %** floor
(`D-DDV2-ORDER-1`; the 2026-09-05 DDv2 analysis §1.2b). ⇒ **A deterministic fan is not a
disqualification**, and `H-RL-COLL-1`'s "refcv3 has no denoiser to post-train" (`M52`) does
**not** transfer to this design: we reproduce the mechanism the code actually ships.

Implementation: `cfg.noise_mode="two_scalar"` (already in `refcv3_adapter.sample_offsets`),
σ floor 0.04, likelihood σ 0.10, G = 4 intra-anchor.
⭐ **Preferred variant for refcv4b/refcv5: the scalars act in CONTROL space** `(a_lon,
a_lat)` — refcv4b's `anchors.pt` carries `controls[117,2]` with `control_units: alat` — and
the sample is re-rolled through `rollout_unicycle`, so **every explored candidate is
flyable by construction** (`D-REFCV5-PLAN-4`). The metre-space form is retained as the
**deliberate-regression arm that must FAIL flyability**.

### 2.3 The reward — and the term the measurement says is missing

⛔ **A `G-REWARD` rate can be gamed by DELETING terms, and the panel proves it**: stripping
comfort *and* headway reads **0.4099**, which is **exactly the `hackable_progress_only`
deliberate-regression arm's rate** (0.4099). ⇒ **the rate is a necessary gate, never the
objective**; a spec that passes it by collapsing onto `progress` has not been repaired.

The composition (DDv2's *structure*, none of its map terms — PhysicalAI has no map, so
`DAC` and lane-keeping are refused as unconstructible):

| channel | term | why |
|---|---|---|
| **VETO** (pinned at −1, outside the ranking) | collision, TTC-imminent, feasibility-envelope, comfort-envelope | a constraint PINS; it does not rank. Moving feasibility/comfort here is `D-REFCV5-PLAN-6`'s named repair |
| **RANK** | `progress` · `headway` · ⭐ **`robust_contact`** | the ordering signal |
| **BAR** | ≥ GT mask (`use_gt_bar=True`) | the human's SCORE, not shape — *a bar, not a target* |
| **ANCHOR** | `w_anchor` trust region to the frozen policy + λ_IL | DDv2 has only the IL term and no KL; we carry both and attribute them |

⭐ **`robust_contact` is the new term, and it is where the headroom provably is.** Contact
at `dt = 0` is closed **by construction** (`contact_projection.py`: 3.4277 % → **0.000000 %**
at **+0.0000 m** ADE) — ⛔ no reward budget may be spent on it. But contact under
**agent-motion prediction error** is not closed and cannot be: MEASURED `fan_contact`
**0.034277** (dt = 0) → **0.039811** (−0.5 s) → **0.066243** (−1.0 s), **monotone over
−1.0…+2.0 s and roughly linear, with NO THRESHOLD to sit safely below**
(`D-RL-TIMING-SURFACE-1`). ⚠️ **Sign as GEOMETRY:** `dt < 0` places the lead **earlier along
its own path**, i.e. **closer to a following ego** — the risk direction.
⇒ A knee can be widened away by a bigger margin; a smooth monotone residual cannot, because
every margin is still on the slope. **That is what a learned policy is for.**

⛔ **Echo terms refused in advance** (the DDv2 analysis §3.3 table): ADE/FDE to the GT,
speed-profile match, "stay near the recorded path", route/nav agreement, and EP-as-a-term.
Each reads the ego's own recorded future and is the imitation loss in a reward costume;
inside a group-relative advantage an imitation term is a **fan-collapse objective**.

---

## 3. Arms — one variable, and every control reads a KNOWN value

| arm | reward | veto | lr | steps | what it must read |
|---|---|---|---|---|---|
| `base` | — | — | — | — | the banked refcv4b FINAL dump, RL OFF |
| **`rl`** | RANK + BAR | on | 1e-5 | 2,000 | the hypothesis |
| **`rl_s1`** | identical | on | 1e-5 | 2,000 | ⭐ **SEED REPLICATE.** `H-ESTIM-SEED-1`: a separated CI at one seed is NECESSARY, NOT SUFFICIENT |
| **`veto_only`** | all 0.0 | **on** | 1e-5 | 2,000 | ⛔ **MANDATORY.** Separates the reward's effect from the mere act of filtering. It is also the only refcv3 arm that moved the fan the right way (`fan_peak_g_mean` −0.0859 g) |
| **`ctrl_null`** | all 0.0 | **off** | 1e-5 | 2,000 | the **zero-reward null**: advantage identically 0, `veto_rate` **exactly 0.0** |
| **`dose_null`** | all 0.0 | off | 1e-5 | 2,000 | **dose-matched**: the optimiser takes the same number of steps at the same LR with no information. The refcv3 campaign's costs were NOT separable from this arm's own drift — budget for it |
| **`ctrl0`** | RANK + BAR | on | **0.0** | 200 | frozen-weight: every readout delta **exactly 0**, state-dict sha256 unchanged, or **VOID** |
| **`reg_echo`** | `gt_similarity` 1.0 | — | 1e-5 | 2,000 | ⛔ **deliberate regression**: the ego GT future inside the advantage. `R_FAN` must collapse ≥ 30 % with separation, **or the whole panel is VOID** |
| **`reg_metre`** | RANK + BAR | on | 1e-5 | 2,000 | ⛔ deliberate regression #2: metre-space noise, **must FAIL flyability** |

⛔ **`rl` vs `veto_only` is the attribution**, and it is the comparison the refcv3 campaign
lacked until its control turned out not to be a null. `rl` vs `dose_null` is the drift control.

---

## 4. Readout — four families, tiers stamped, estimator named

**PRIMARY (T0, the fan):** robust-contact floor over the shift grid, on fan / top-32 /
top-8 / selected, each on its **own population with its own n**; plus `fan_floor@k`,
diversity, Kamm rate.
⚠️ **`fan_floor@k`, fan-collision-vs-replay and diversity DO NOT EXIST as instruments**
(`D-REFCV5-PLAN-9` (3), ESCALATED, 0 GPU). ⛔ **They are a hard blocker on the primary
readout and must land before any arm** — see §6.

**SECONDARY (T1, self-action OPEN loop — never "closed loop"):** the four families
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC, **never pooled**, each with its estimator,
CI and n; the guard that a safety gain was not bought with driving quality.

⛔ **Estimator discipline, all binding:**
* the paired episode-cluster bootstrap answers *"would another draw of EPISODES say this?"*
  — **not** another training run (`H-ESTIM-SEED-1`) nor another inference run. **This SPEC
  names EPISODES as the question its CI answers, and budgets `rl_s1` for the training one.**
* ⛔ **`sep` is unreliable on this rig (`M44`)** — read every result by its **delta and
  interval**, never by the flag.
* ⚠️ **the separation floor is PER METRIC, derived from that metric's own quantum and
  units.** A metric whose base value sits below its own floor is stamped
  **UNDETECTABLE-DOWNWARD**, not reported as null — the refcv3 campaign reported a null on
  `mass_rank_contact` whose base (3.470e-05) was **0.347 ×** its own 1e-4 floor, so only a
  worsening could ever have been detected.

---

## 5. Exit order (applied mechanically, first match wins)

`VOID` (ctrl0 moved · reg_echo's G-FAN did not fire · reg_metre passed flyability) →
`FAIL-COLLAPSE` (R_FAN < 85 %) → `FAIL-GUARD` (T1 guard breached) → `FAIL-REPLICATE`
(seeds disagree in sign) → `FAIL-ATTRIB` (`rl` indistinguishable from `veto_only`) →
`PASS` → `SPLIT` → `NULL`.

⚠️ **The exit is read from the gate order, never chosen past a fired gate** — RETRACTION #25's
class: an outcome selected past its own VOID because the gate's failure looked informative.

---

## 6. ⛔ Blockers — named, with what unblocks each

| # | blocker | unblocked by | GPU |
|---|---|---|---|
| B1 | **`G-REWARD` FAILS** at 0.4418 under the plan's own repair (§1) | a reward term the human beats hold-v0 on — `robust_contact` is the candidate, and it must be **scored on the humanflag corpus before any arm** | 0 |
| B2 | **`fan_floor@k` / fan-collision-vs-replay / diversity do not exist** | implement in `taniteval` (`D-REFCV5-PLAN-9` (3)) | 0 |
| B3 | **no refcv4b checkpoint has been evaluated** — training ETA ~2026-09-06 08:00 UTC | the run finishing | — |
| B4 | forbidden/exclude prefix tables are refcv3 literals (`S2`) | re-derive on the real build + pin | 0 |
| B5 | both GPUs busy (Thor 97–98 %, dev-box 4060 100 %) | a free slot; the arms are ~30 min each on the 4060 | queue |

**B1, B2 and B4 are all zero-GPU and are the critical path — none of them waits on B3.**
