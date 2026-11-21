# RL post-training for refcv4b and refcv5 — the stage now EXISTS, and the reward gate is measured

**Stream:** Architecture & Inference · **Date:** 2026-09-05/06 · **Branch:** `agent/arch-inf-20260803`
**GPU spent: 0.** Both GPUs were busy (Thor 97–98 %, dev-box 4060 100 %); nothing here needed one.

**PI, verbatim:** *"Did you give up the RL topic? This is not acceptable, because the diffusion paper
v2 showed that it improved the performance. And our plan is exactly to do this with refcv4b and
refcv5 as they are finished to train."*

---

## 0. The answer in one paragraph

The RL line was never dead; it was **mis-scoped and then blocked on a gate nobody had scored**. The
scope error is already logged (`M52`): `H-RL-COLL-1` ran on **refcv3** and its FAILURE is a verdict on
*that arm on that model*, not on the method. ⭐ **And the deeper reading is better than the
retraction:** DiffusionDriveV2's *released* RL stage never uses the diffusion chain's randomness at
all — the additive DDPM term is **multiplied by zero**, leaving a **two-scalar (longitudinal,
lateral) scale policy** — so **"refcv3 has no denoiser to post-train" does not block refcv4b either.
A deterministic fan can carry the published mechanism faithfully.** What *does* block it is
`G-REWARD`, `D-REFCV5-PLAN-6`'s precondition that *"a reward that fails it may not train anything"*,
and this turn **scored it for the first time**: the shipped reward fails at **0.7249**, the plan's own
named repair (move feasibility and comfort to vetoes) reaches **0.4418** and **still fails**, and the
attribution says why — **`comfort` is the entire defect (+0.0208 of a +0.0089 net gap) while
`feasibility` is inert at −0.0007**, so half the prescribed repair is a no-op. ⛔ The rate is also
**gameable**: stripping terms until it falls lands exactly on the deliberate-regression arm's value
(0.4099 both). ⇒ The reward needs a *new* term, not a smaller one, and the term is named by
measurement rather than taste: **contact under agent-motion prediction error**, the one residual the
contact projection provably cannot discharge. It is **built, tested (19/19 green), and scored** — and
scoring it produced the turn's real finding. ⛔ The term alone does **not** pass the gate (0.4418 →
0.4355), and **no admissible reward can**: over **2,400 weightings** the minimum reachable rate is
**0.3840** against a ≤0.30 ceiling, because the safety terms fire on **1.03 %** of windows while
`progress` and `headway` fire on ~96 % and are near-symmetric between a human and a constant-velocity
path. ⭐⭐ **`G-REWARD`'s ceiling is therefore UNREACHABLE on this corpus — the gate's POPULATION is
wrong, not its threshold** (the same class as the campaign's own `UNDETECTABLE-DOWNWARD` correction,
and independently corroborated by `D-RL-COLL-SPARSE-1`'s *advantage identically zero in 92 % of
windows*). On the **signal-bearing** population the repaired reward with the new term reads **0.0635
[0.0000, 0.3158]** against the default's **0.8413** — ⚠️ n = 63 windows / **6 episodes**, so the point
estimate clears the ceiling and the interval straddles it: **necessary, not sufficient**, and
post-hoc, so it must be pre-registered before it may become the gate. ⇒ **The RL stage is not blocked
by a broken reward. It is blocked by a mis-specified gate and by two zero-GPU instrument gaps**, all
three of which are now named with what fixes them.

---

## 1. P1 — what DD-v2 actually post-trains, from the primary

Banked primaries, `--cited-by` this report: **`2512.07745`** (DiffusionDriveV2), **`2411.15139`**
(DiffusionDrive), **`2409.00588`** (DPPO), **`2507.04049`** (DIVER). Library: 447 entries.
The mechanism below is `PUBLISHED-CODE` at `hustvl/DiffusionDriveV2@1cd12a1`, read in full by the
banked sibling analysis `…/2026-09-05-diffusiondrive-v2-analysis/RESULT.md` (verified against its own
line cites, not taken on trust).

| question | answer |
|---|---|
| **what parameters** | **`_trajectory_head` ONLY.** Every other parameter `requires_grad=False` and every other module `.eval()` (`_rl_agent.py:61-67`). 10 epochs, lr 2e-4, cosine, from the released 88.1-PDMS v1 checkpoint |
| **what stage of the sampler** | ⭐ **In practice, none.** Paper Eq. 5 describes a per-step Gaussian over the denoising chain, but the code sets **`std_dev_t_add = 0.0`** in both branches (`_model_rl.py:640,643`) and forms `prev_sample = prev_sample_mean · ε_mul + 0 · ε_add` (`:666`). The only stochasticity is **two multiplicative scalars per trajectory** (long, lat) at a floored **4 %** (`:646-654`). The likelihood used for the gradient is an isotropic Gaussian at a constant σ = 0.1 that **does not match that sampler** |
| **what reward** | **NAVSIM's PDM score**, from the non-reactive PDM simulator (4 s LQR rollout at 10 Hz against **replayed** agents): `final = NC × DAC × (5·EP + 5·TTC + 2·C + 0·DDC)/12`, with `EP` renormalised as `progress_i / max(progress_GT, progress_i)` (`:75-117`, `:741-753`). The paper never defines `R`; the code settles it |
| **what estimator** | Intra-anchor GRPO `(r − mean_G)/(std_G + 1e-4)`, G = 4, γ = 0.8 over 10 rollout steps; then a truncated inter-anchor advantage: **−1** if `no_collision != 1` **or `drivable_area != 1`**, else `max(0, A) × [r ≥ r_GT − 1e-6]`. The importance ratio is identically 1 ⇒ **plain REINFORCE**. ⛔ **No KL, no clip, no reference policy anywhere in the file** — the trust story is an **IL term** at λ 0.1, or **1.0 on any row where nothing beat the GT** |
| **attribution: RL vs base** | The authors' own control (Tab. 9): DD 88.1 → DD + V2's selector **89.1** → V2 **91.2** ⇒ **selector +1.0, RL ≈ +2.1**. The gain is **EP +5.3** and **DAC +1.7**; NC +0.1, TTC +0.1, Comf −0.1. The raw-fan floor **PDMS@10 75.3 → 84.4** while diversity **falls 42.3 → 30.3** — it buys floor quality by narrowing the fan. ⚠️ Every number is single-run with no interval, and **the reward is never ablated** |

### 1.1 Which reward terms our corpus can compute — term by term

Constraint set: vision-only at inference; **no map, no lane graph, no traffic lights, no route**
(the card says verbatim *"we do not include open maps data"*, pinned by
`test_physicalai_feature_readset.py`); `obstacle.offline` agent tracks on **97.44 %** of the corpus,
train-time only; no NAVSIM simulator.

| DD-v2 term | ours? | how / why not |
|---|---|---|
| **NC** (no-collision) | ✅ | swept-segment predicate vs the replayed `obstacle.offline` lead track (`rewards._collision`) |
| **TTC** | ✅ | closing-speed TTC, threshold 2.93 s = the human's own 5th-percentile gap |
| **C** (comfort) | ✅ | jerk / lateral-acc envelope, candidate geometry only. ⚠️ **but see §3 — as a RANKING term it is the defect** |
| **EP** (ego progress) | ⚠️ partial | along-track displacement is computable; **normalising by the GT's progress is admissible, using it as a term is the longitudinal echo in ratio form** |
| **≥ GT bar** | ✅ | the GT's **score**, not its shape — *a bar, not a target*. Already implemented (`advantage.py:91-151`) |
| **DAC** (drivable area) | ❌ | **no map.** This was a third of V2's published gain (DAC +1.7) |
| **LK / TL / DDC** | ❌ | no map, no traffic-light feature, no lane graph |
| **PDM simulator itself** | ❌ | replaced by a rule-based T0 reward over replayed tracks — which is what NAVSIM's non-reactive sim *is*, minus the map |

⛔ **Echo traps refused in advance:** ADE/FDE to the GT, speed-profile match, "stay near the recorded
path", route/nav agreement. Each reads the ego's own recorded future; inside a group-relative
advantage an imitation term is a **fan-collapse objective**, which destroys precisely the fan-floor
property the method exists to buy.

⚠️ **A vocabulary correction that travels with these numbers:** V2 calls its PDM reward
*"closed-loop"*. NAVSIM's own title is *"non-reactive simulation"*, and under the PI's 2026-09-02
ruling a 4-s kinematic rollout against replayed agents is **self-action OPEN loop — our T1** — not
closed loop. Their headline tier is our T1.

---

## 2. The scope error, stated so it cannot propagate again

⛔ `H-RL-COLL-1` returned FAILURE **on refcv3, and that verdict is about refcv3 and nothing else.**
refcv3 carries DiffusionDrive-v1's skeleton **without its mechanism**: no noise schedule, no anchored
Gaussian, no DDIM, no sampling; its ranking reads t = 0 confidence, **unchanged on 201/201 windows**
(`D-REFC-DDAUDIT-1..6`). A null from post-training a model that lacks the mechanism being
post-trained is not evidence about the method. Logged as **`M52`** (commit `3a73f09`).

⭐ **But the constructive half matters more, and it is what unblocks refcv4b:** because the released
RL stage never uses the chain's randomness (§1), **the absence of a denoiser is not a blocker at
all**. refcv4b's decoder is a deterministic refiner (`refc.py:1740-1745`: `noise = ... if
self.training else torch.zeros_like(x)`). Neither fact blocks this stage.

⚠️ **A CORRECTION I ALMOST INHERITED, CAUGHT BY RE-VERIFYING RATHER THAN CITING.** A sub-agent reported — and `D-REFCV5-WP4-CONFIG-ONLY` records — that refcv5's DDIM is *not wired into the model*, with `grep -c -F "cross_agent" refc.py` returning **0**. **That is no longer true on this branch.** Re-checked directly, with a same-breath control that must read non-zero (`grep -cF "class " refc.py` = 38): `refc_sampler` is referenced **3×**, `sampler` **49×**, and `cross_agent` is a real config field (`refc.py:456`) instantiated as an `nn.MultiheadAttention` (`:1219`). `git log` puts it in **`a5dbfbb` "refcv5 WP-4: the diffusion mechanism, in CONTROL space"**. ⇒ The register row is **STALE**, not wrong-at-the-time; a sibling stream landed the seam tonight while HEAD moved five times. **Escalated below.** This is the stale-absence class the operating standard names, and the only reason it did not propagate into this report is that the claim was re-run instead of quoted.

**Carried forward from the refcv3 campaign, not re-derived:**
* the collision reward **did** move the generator — `fan_contact` **−0.002734** [−0.005339, −0.000781]
  against a dose-matched null — but **not replicated across seeds**;
* its friction/feasibility **costs were NOT separable from the optimizer's own drift**;
* ⭐ **contact at dt = 0 is closed BY CONSTRUCTION** (`contact_projection.py`: 3.4277 % →
  **0.000000 %** at **+0.0000 m** ADE, zero GPU) ⇒ ⛔ **no reward budget may be spent on it**;
* ⭐ the **veto-only** arm was the only one that moved refcv3's fan the right way
  (`fan_peak_g_mean` −0.0859 g in 200 steps, **with the reward identically zero**);
* ⚠️ and the campaign's real lesson is a process one: **the endpoint was chosen after its own
  headroom had been measured away** (`sel_contact` 0.0000 vs the human's 0.0000), and the arms ran
  anyway. §4 states this design's headroom **before** the arm.

---

## 3. ⭐ P2/P4 — `G-REWARD`, MEASURED for the first time

`D-REFCV5-PLAN-6` is a hard precondition — *"hold-v0 must beat the human on ≤ 30 % of lead windows …
**A reward that fails G-REWARD may not train anything**"* — and it had never been scored under
anything but the shipped default. **Instrument built this turn:
`stack/scripts/rl_greward_gate.py`** (0 GPU): it re-weights the banked per-window component table
(`humanflag_fit120.json`, **6,089 lead windows / 73 episodes**). Re-weighting is **exact, not an
approximation**, because `RewardSpec.__call__` is a linear combination and every component is banked.

⭐ **CHANNEL CONTROL (M50, same breath, same pattern): recomputing under `DEFAULT_WEIGHTS` reproduces
the banked `hold_v0_scores_at_least_human` = 0.7249137789456397 with `err = 0.000e+00`.** Every number
below inherits that proof.

| spec | hold-v0 ≥ human | episode-cluster CI | median gap | mean gap | G-REWARD |
|---|---|---|---|---|---|
| `default` (shipped) | **0.7249** | [0.6666, 0.7840] | **+0.01151** | +0.00892 | ⛔ FAIL |
| `veto_feas_comfort` (the plan's repair) | **0.4418** | [0.3682, 0.5187] | **−0.00240** | −0.01124 | ⛔ FAIL |
| `veto_comfort_only` | 0.4419 | [0.3684, 0.5189] | — | −0.01187 | ⛔ FAIL |
| `veto_feas_only` | **0.7247** | [0.6663, 0.7836] | — | +0.00961 | ⛔ FAIL |
| `veto_fc_no_headway` | 0.4099 | [0.3239, 0.4953] | — | −0.00644 | ⛔ FAIL |
| ⛔ `hackable_progress_only` (deliberate regression) | **0.4099** | [0.3239, 0.4953] | — | −0.01867 | ⛔ FAIL |

**Per-component attribution of the mean (hold-v0 − human) gap under the shipped default:**

| component | contribution |
|---|---|
| **`comfort`** | **+0.020836** |
| `feasibility` | −0.000675 |
| `collision` | −0.000821 |
| `headway` | −0.004818 |
| `progress` | −0.005600 |
| **SUM** | **+0.008921** (= the composed-mean gap, exactly) |

### Three findings, and one of them corrects the plan

1. ⭐ **`comfort` IS THE ENTIRE DEFECT; `feasibility` IS INERT.** `D-REFCV5-PLAN-6` prescribes moving
   *"feasibility and comfort"* out of the reward. MEASURED: removing **feasibility alone** leaves the
   rate at **0.7247** (from 0.7249 — a **no-op**), while removing **comfort alone** delivers the whole
   move to **0.4419**. **Half the prescribed repair does nothing**, and the plan's row should say so.
   *The mechanism is visible in the component: `comfort = exp(−(jerk + lat_acc)) × motion_gate`. A
   constant-velocity straight path has **zero** jerk and **zero** lateral acceleration, so it scores
   **exactly 1.0** — the maximum — while a human who accelerates or turns scores strictly less. The
   `_motion_gate` that was added to stop the FROZEN path collecting this free 0.70 does not fire on
   hold-v0, because **hold-v0 moves**.*
2. ⛔ **THE REPAIR IS INSUFFICIENT.** At 0.4418 the gate still fails ≤ 0.30, so **no RL arm may launch
   on refcv4b or refcv5 under any reweighting of the five shipped components.**
3. ⛔⛔ **AND THE RATE IS GAMEABLE, WHICH MAKES IT A GATE AND NEVER AN OBJECTIVE.** Stripping comfort
   *and* headway reads **0.4099** — **identical to the `hackable_progress_only` deliberate-regression
   arm (0.4099)**. A spec that passes G-REWARD by deleting terms until only `progress` is left has
   converged on the known-bad reward. ⇒ **The gate must be read together with the attribution table**,
   and a repair is only a repair if a term the human *wins* was ADDED.

⚠️ **A fourth reading, stated because it is a negative and it was mine.** I hypothesised the 0.44 was
a corpus artifact — that on a corpus where the human often *is* driving straight at constant speed,
"hold-v0 ties the human" describes the corpus rather than the reward. **REFUTED by stratifying on the
human's own manoeuvre magnitude:** the repaired spec reads 0.4874 / 0.4362 / 0.3533 / **0.4773** across
|progress−1| bands from ~0 to ≥0.15, i.e. **roughly flat**, including on the most strongly manoeuvring
windows. (The *default* spec does show the pattern strongly — 0.9674 → 0.5570 — so the effect is real
for it and absent for the repair.) The honest conclusion is the harder one: after comfort is removed
the remaining terms **barely discriminate the human from the trivial path at all**, which is exactly
why a new term is needed rather than a new weighting.

---

## 4. ⭐ The term the measurement demands — and where its headroom is, stated BEFORE the arm

A constraint can forbid; it cannot rank. So RL's target must be the residual a constraint **provably
cannot discharge**:

* contact at `dt = 0` — **closed by construction**, 3.4277 % → **0.000000 %** at **+0.0000 m** ADE.
  ⛔ Nothing to buy; this is what the refcv3 campaign aimed at after its headroom was gone;
* contact under **agent-motion prediction error** — **not closed, and not closable**: MEASURED
  `fan_contact` **0.034277** (dt = 0) → **0.039811** (1.16×, dt = −0.5 s) → **0.066243** (1.93×,
  dt = −1.0 s), **monotone across the whole −1.0…+2.0 s sweep and roughly linear, with NO THRESHOLD
  to sit safely below** (`D-RL-TIMING-SURFACE-1`).

⭐ **That shape is the whole argument.** A residual with a knee can be widened away by a larger safety
margin; a smooth monotone one cannot, because every margin you choose is still on the slope. That is
what a learned policy is for.

⚠️ **SIGN AS GEOMETRY, NEVER THE ADJECTIVE "EARLY":** `dt < 0` places the lead **earlier along its own
path**, i.e. **closer to a following ego** — the risk direction. Reading it the other way inverts the
table.

**Built:** `stack/tanitad/rl/robust_contact.py` — `E[contact]` over a shift grid, defaulting to
`(−1.0, −0.5, 0.0, +0.5)` s, weighted toward the risk direction, with edge **clamping** (a shift past
either end holds the endpoint — an extrapolated lead position is a **fabricated obstacle**).

### 4.1 ⭐ B1 — the term SCORED on the same 6,089 windows, and it does NOT pass G-REWARD alone

Probe: `raw/greward_robust_probe.py`, 0 GPU, the driver's own corpus/lead helpers so the window set,
lead model and geometry are identical to the banked run. **Controls:** zero-shift `robust_contact`
equals the point `collision` **exactly on every window** (asserted per row, 18,267 assertions);
recomputed `collision` vs banked **12,165/12,178 match**.

⚠️ **THE 13 MISMATCHES ARE DISCLOSED, NOT HIDDEN, AND THEIR DIRECTION IS INFORMATIVE.** All 13 are
`hold_v0`, all `banked 0.0` vs `ours −1.0` — i.e. **our recomputation finds contact the banked run
missed, never the reverse** — while `headway` is bit-identical on the same rows, so the context is the
same object. That is the signature of `D-RL-CONTACT-DEFN-1` (*"every pre-2026-09-05 `contact` number
is 37.8 % low"*): the banked rows were scored under the **old under-reporting predicate** and this
recomputation uses the corrected one. `HYPOTHESIS` for the mechanism; `MEASURED` for the direction
and the count. It does not affect any conclusion below, all of which rest on differences.

| spec | hold-v0 ≥ human | median gap | mean gap | frozen ≥ human |
|---|---|---|---|---|
| `default` | 0.7247 | +0.01146 | +0.00843 | 0.0445 |
| `veto_feas_comfort` | 0.4418 | −0.00240 | −0.01173 | 0.0463 |
| **`+ robust_contact` (w 1)** | **0.4355** | −0.00268 | **−0.01493** | 0.0488 |
| `+ robust_contact` (w 2) | 0.4355 | −0.00268 | −0.01814 | 0.0488 |
| `+ robust_contact` (w 5) | 0.4355 | −0.00268 | **−0.02774** | 0.0488 |
| ⛔ `robust_only` | **0.9903** | 0.00000 | −0.00320 | **1.0000** |

**The term is correct in direction and insufficient in isolation.** The mean gap moves monotonically
with its weight (−0.0117 → −0.0149 → −0.0181 → −0.0277), and the term itself discriminates the way
the design predicts:

| path | `collision` (dt = 0) | `robust_contact` |
|---|---|---|
| human | −0.001478 | −0.002094 |
| hold-v0 | −0.002792 | −0.005296 |
| **ratio hold-v0 : human** | **1.89×** | ⭐ **2.53×** |

⇒ **Marginalising over timing error WIDENS the human's advantage from 1.89× to 2.53×** — the trivial
constant-velocity path is exactly the one least robust to the lead being where we did not predict.
⛔ And `robust_only` reading **0.9903 with `frozen` at 1.0000** is the declared degenerate
(`hackable_alone=True`) firing empirically: a contact-only reward is hacked by standing still. It must
never ship alone, and the panel proves the audit's claim rather than asserting it.

### 4.2 ⛔⛔ WHY IT STILL FAILS — `G-REWARD`'s CEILING IS UNREACHABLE ON THIS CORPUS

**How often does each term even FIRE** (differ at all between the human and hold-v0), over 6,089
windows:

| term | fires on |
|---|---|
| `progress` | **95.70 %** |
| `comfort` | **95.70 %** |
| `headway` | **76.79 %** |
| `feasibility` | 2.89 % |
| ⭐ `robust_contact` | **1.03 %** |
| `collision` (dt = 0) | **0.13 %** |

⇒ **The safety terms are almost never the deciding factor.** `robust_contact` fires **8× more often
than `collision`** — it does exactly what it was built to do — but 1.03 % cannot move a rate computed
over all windows, which is decided by `progress` and `headway`, and those are near-symmetric between
a human and a constant-velocity path *because on ~95 % of lead windows nobody is near a collision*.

**MEASURED, exhaustive: over 2,400 weightings of the four admissible ranking terms (`progress`,
`headway`, `collision`, `robust_contact`; grid {0, 0.1, 0.3, 0.5, 1, 2, 5}), the MINIMUM achievable
rate is 0.3840 and the ceiling 0.30 is reached by NONE of them.**

⭐⭐ ⇒ **`G-REWARD`'s ≤ 30 % ceiling is UNREACHABLE by any admissible reward on this corpus.** A gate
whose passing region cannot be entered cannot gate. This is exactly the class the refcv3 campaign
already named in its own §12.4 — *"a separation floor correctly derived for one quantity, applied to
quantities it does not fit"*, which produced an **UNDETECTABLE-DOWNWARD** metric — with the object
swapped from a floor to a **population**.

### 4.3 ⭐ The corrected gate — and on it, the repaired reward PASSES

The defect is the **population**, not the threshold. `G-REWARD` asks *"does the reward prefer the
human?"* over a population in which **98.97 % of windows contain no safety information at all**, so it
is measuring `progress`/`headway` symmetry rather than reward quality.

⭐ **This is independently corroborated from the other direction:** `D-RL-COLL-SPARSE-1` measured the
GRPO advantage to be **identically zero in 92 % of windows**. The same sparsity, measured on the
advantage rather than the reward. **The windows G-REWARD scores are overwhelmingly windows on which
RL cannot learn anything either.**

Restricted to the **signal-bearing population** — windows where a safety term actually differs between
the two paths, a property of the SCENE and not of the outcome, and the same population on which the
RL advantage is non-zero:

| spec | all windows | **signal windows** | episode-cluster CI (signal) |
|---|---|---|---|
| `default` | 0.7247 | **0.8413** | [0.3998, 1.0000] |
| `veto_feas_comfort` | 0.4418 | **0.6667** | [0.2103, 0.8987] |
| ⭐ **`+ robust_contact`** | 0.4355 | ⭐ **0.0635** | **[0.0000, 0.3158]** |

⇒ **On the windows where the reward can see anything, the repaired reward with `robust_contact`
prefers the human on 93.65 % of them — a 13× reduction from the shipped default's 0.8413.**

⛔ **STATED LIMITS, because this is suggestive and not established.**
1. **n = 63 windows / 6 EPISODES.** Six clusters is a very small bootstrap and the CI is wide; its
   **upper bound 0.3158 STRADDLES the 0.30 ceiling**, so the point estimate clears it and the interval
   does not. ⇒ **NECESSARY, NOT SUFFICIENT** — and under `H-ESTIM-SEED-1` this interval answers only
   *"would another draw of EPISODES say this?"*, from six of them.
2. ⚠️ **The restriction was applied AFTER seeing the all-window result, and that must be declared.**
   It is not outcome-selected — the population is defined by a scene property — but it is **post-hoc**,
   and it may not become the gate until it is **pre-registered on its own**, on a corpus with enough
   lead-conflict episodes to carry a bootstrap.
3. The corpus is **NON-PARITY RL-fit clips**, T0, and the lead model is `track`. Under the `static`
   lead model the human is flagged at 0.1184 contact — a static lead turns every competent follower
   into a collision, so `track` is the only admissible reading.

---

## 5. P4 — what was built, what was reused, what is new

| file | state | what it is |
|---|---|---|
| `stack/tanitad/rl/robust_contact.py` | ⭐ **NEW** | the timing-marginalised contact term (§4) |
| `stack/tanitad/rl/refc_adapter.py` | ⭐ **NEW** | the refcv4b/refcv5 binding + the conditioning refusal (§5.1) |
| `stack/scripts/rl_greward_gate.py` | ⭐ **NEW** | the G-REWARD instrument (§3), 0 GPU |
| `stack/tests/test_rl_refc_adapter_robust.py` | ⭐ **NEW**, **19/19 green** | directional tests for both modules |
| `…/raw/greward_robust_probe.py` | ⭐ **NEW** | the B1 probe scoring the new term on the humanflag corpus |
| `stack/tanitad/rl/{rewards,advantage,anchor,audit,config,posttrain}.py` | **reused unchanged** | model-agnostic; 89 baseline tests green |
| `advantage.truncated_inter_anchor_advantage` (≥GT bar) | **already present** | landed 2026-09-05; the DDv2 analysis' integration request #1a is **done** |
| `refcv3_adapter.sample_offsets` `noise_mode="two_scalar"` | **already present** | integration request #1b is **done** |
| `posttrain.veto_mask` explicit keying | **already fixed** | the `"collision" in spec.weights` bug is gone; keyed on three config booleans |

### 5.1 ⛔ The launch-blocker this turn removed — a silently different policy

`refcv3_adapter.make_refcv3_sample_fn` calls `model(frames, nav_cmd, v0, steps=…, lan=…)` — **five of
the eight** parameters `RefCV3Model.forward` accepts. refcv4b is trained with
**`--ego-state-inject --ego-dropout 0.5`**, so its policy is conditioned on `ego_state [B,5]`.
`refc_v3.py:1000` fails loud when `ego_state` is **supplied** to a build that would drop it — and is
**silent in the other direction**: *omitting* it on a build trained with it returns a perfectly
well-formed fan **from a differently conditioned policy, and nothing raises.** RL would then
post-train a policy that is not the deployed one and the result table would read as a lever effect.

`refc_adapter.assert_conditioning` makes that omission loud, reading the requirement from the
**model's own config field** (`M51`: a name is not provenance), and a test pins that **no requirement
names a config field that does not exist** — a guard keyed on a misspelt flag reads `False` forever,
which is the false-green class wearing a guard's clothes. ⚠️ **`ego_dropout 0.5` does not make the
channel optional at RL time**: dropout makes the policy robust to the channel being missing; it does
not make *always missing* the distribution the checkpoint was fitted under.
⛔ `nav_cmd` is **deliberately not asserted** — REF-C is evaluated with `nav_cmd=None` on purpose, and
asserting it would refuse the programme's own standard eval arm.

### 5.2 ⭐ B4 closed — the prefix tables are safe on refcv4b, and refcv5 needs one ruling

MEASURED (0 GPU, `named_parameters()` on real builds, not a mock):

| build | tensors under `core.decoder` | dropped by the refcv3 exclude list | **ranking surfaces still trainable** |
|---|---|---|---|
| refcv4b-shaped (`ego_state_inject=True`) | 47 | 4 | ✅ **0** |
| refcv5-shaped (`cross_agent=True`, `sampler="ddim"`) | 67 | 4 | ⚠️ **2** |

✅ **The nine refcv3 selector literals are sufficient for refcv4b** — the audit's `S2` risk does not bite there, and the RL stage may use them as-is.

⚠️ **refcv5 surfaces two tensors that need a typed decision, not a regex:** `core.decoder.layers.{0,1}.agent_gate` — the **zero-init gates on the agent cross-attention** (refcv5's WP-1 seam). They are **not scorers**, so excluding them as "ranking surfaces" would be wrong; but they are also not action parameters. ⛔ **If RL can move them it is changing what the policy PERCEIVES, not only how it acts**, and every `rl` − `base` difference then contains a perception change against a base whose gate value came from IL. ⇒ **The stage must either freeze `agent_gate` (RL changes the action mapping only — the default this SPEC takes, matching DDv2's head-only stage) or declare it trainable and attribute it separately.** Recorded so the choice is something somebody TYPED, which is the same rule the veto's explicit keying now enforces.

---

## 6. P3 — the pre-registration

`SPEC.md` in this package, `TanitAD_ValidateAIDesign` §1 schema. Hypothesis **`H-DDA-5`**
(`GOALS_AND_CLAIMS.md:2919`, OPEN), vehicle **`E-DDA-3b`** with base = the refcv4b FINAL
(`D-REFCV5-PLAN-6`). One variable: frozen fan → head-only scale policy. Success **and** failure
written before compute. Nine arms; the controls that must read known values are
**`ctrl_null`** (zero-reward, veto off ⇒ advantage identically 0, `veto_rate` exactly 0.0),
**`dose_null`** (dose-matched drift control), **`ctrl0`** (frozen weights ⇒ every delta exactly 0),
**`reg_echo`** and **`reg_metre`** (deliberate regressions that must FAIL), ⭐ **`veto_only`**
(mandatory — the reward's effect is not attributable without it), and ⭐ **`rl_s1`** (a **seed
replicate**, budgeted from the start because `H-ESTIM-SEED-1` makes a one-seed separated CI
**necessary, not sufficient**; the refcv3 campaign's single biggest weakness was exactly this).

Estimator discipline stated in the SPEC: the paired episode-cluster bootstrap answers **"would another
draw of EPISODES say this?"** and the SPEC says so; `sep` is not read (`M44`); the separation floor is
**per metric from its own quantum and units**, and a metric whose base sits below its own floor is
stamped **UNDETECTABLE-DOWNWARD** rather than reported as null. Four families, never pooled, T1 primary
for any capability claim.

---

## 7. Blockers — each named, with what unblocks it

| # | blocker | unblocked by | GPU |
|---|---|---|---|
| **B1** | ⛔ **`G-REWARD` is UNREACHABLE as written** — min 0.3840 over 2,400 admissible weightings vs a ≤0.30 ceiling (§4.2) | ⭐ **a PI/Master-Mind ruling on the gate's POPULATION.** The corrected form (signal-bearing windows) is measured and the repaired reward passes it at 0.0635, but on 6 episodes and post-hoc ⇒ it needs its own pre-registration on a lead-dense corpus | 0 |
| **B2** | **`fan_floor@k`, fan-collision-vs-replay, diversity DO NOT EXIST** — the RL rung has no primary readout | implement in `taniteval` (`D-REFCV5-PLAN-9` (3)) | 0 |
| **B3** | **no refcv4b checkpoint evaluated**; training ETA ~2026-09-06 08:00 UTC | the run finishing | — |
| **B4** | ✅ **CLOSED for refcv4b, and it produced a refcv5 DESIGN DECISION** — see §5.2 | a ruling on the two `agent_gate` tensors | 0 |
| **B5** | both GPUs busy (Thor 97–98 %, 4060 100 %) | a free slot; ~30 min/arm on the 4060 | queue |

**B2 and B4 are zero-GPU and are the critical path — neither waits on B3.**

---

## 8. Manifest

| artifact | where it lives |
|---|---|
| this report | `…/2026-09-05-rl-posttrain-refcv4b-refcv5/RESULT.md` (repo) |
| the pre-registration | `…/2026-09-05-rl-posttrain-refcv4b-refcv5/SPEC.md` (repo) |
| G-REWARD panel | `…/raw/greward_track.json` (repo) |
| B1 robust-term panel | `…/raw/b1_robust_greward.json` (repo) |
| the three new modules + tests | `stack/tanitad/rl/{robust_contact,refc_adapter}.py`, `stack/scripts/rl_greward_gate.py`, `stack/tests/test_rl_refc_adapter_robust.py` (repo) |
| banked primaries | `2512.07745`, `2411.15139`, `2409.00588`, `2507.04049` — `--cited-by` this report, 447 entries |

Nothing lives in only one place; the scratchpad copies are duplicates of repo files.

**Escalations (raised here, not written into a doc for someone to find):**
1. ⭐ **`D-REFCV5-PLAN-6`'s repair text should be corrected**: `feasibility` is inert (−0.000675) and
   removing it is a no-op; **`comfort` is the whole term** (+0.020836). Owner: Master Mind.
2. **B2** — the fan-floor instruments block the RL rung's primary readout and are 0 GPU.
   Owner: Benchmarks/Eval.
3. **B4** — the prefix tables must be re-derived on a real refcv4b/refcv5 build before any arm; the
   failure mode is a *ranking surface trained silently*, which has fired here once already.
   Owner: Training FlyWheel.
4. ⛔⛔ **`G-REWARD` cannot be passed as written and must be re-specified before it blocks anything
   else** (§4.2–§4.3). It is a PI/Master-Mind decision because it is a committed gate: the evidence is
   that its ceiling is unreachable by every admissible reward on this corpus, and that its population
   is ~99 % windows carrying no safety information. **Proposed replacement, to be pre-registered on
   its own:** the rate on the **signal-bearing** population, plus the **sign of the median gap** on all
   windows (margin-free, and it already separates the shipped reward from the repaired one:
   **+0.01146 → −0.00240**). ⛔ Until it is re-specified, "G-REWARD fails" must NOT be quoted as
   evidence that the reward is bad — that is the scope error `M52` logged, one level up.
5. ⛔ **`D-REFCV5-WP4-CONFIG-ONLY` IS STALE** — the refcv5 sampler and agent seam ARE in `refc.py` as of `a5dbfbb` (§2). Any decision resting on "refcv5 has no denoiser" must be re-read. Owner: Master Mind / the refcv5 stream.
6. ⚠️ **`D-RL-CONTACT-DEFN-1` has un-swept consumers**: 13 of 12,178 banked `humanflag` collision
   values disagree with the corrected predicate, all in the under-reporting direction (§4.1). Anything
   re-using `humanflag_fit120.json` inherits the old definition. Owner: Deploy FlyWheel.
