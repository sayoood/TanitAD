# RESULT — VETO-ONLY FAN SAFETY: the reward is exonerated, the veto is a product, and the rig's "separated" needs two floors

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/RESULT.md`
Architecture & Inference FlyWheel · 2026-09-05 · dev-box RTX 4060 only · ⛔ `tanitad-refcv3` (refcv4b live) and Thor never touched · SPEC banked before any arm ran.
Register (same turn): `D-RL-REWARD-RANK-1`, `D-RL-VETO-EXPLICIT-1`, `D-RL-READOUT-CTXRANK-1`, `H-VETO-FAN-1` (new); `H-RL-VETO-1` updated. Cites `D-RL-FANSAFE-1..4`, `D-RL-REWARD-FLOOR-1/2`, `H-ESTIM-SEED-1`.
Evidence classes: **MEASURED** (ours + artifact path) · **INHERITED** (another WP, not re-run) · **HYPOTHESIS**. Tiers: **T0** = readout on the emitted fan, never a driving claim · **T1** = self-action OPEN loop.

**status (2026-09-05 15:2x local / 13:2x Z): BANKED INCREMENTALLY WHILE THE ARMS RUN.** §1-§3 are complete and their artifacts are in `raw/`. §5 (the veto-only arms + their two floors), §7 (T1, four families) and §8 (the 0-training selection rule) are filled in as each lands; this header names what is missing so a killed session leaves no silent gap. Arms completed at the time of this write: `ctrl_null` (s0), `veto200` (s0).

> **The PI's standing correction governs this package:** *"our goals in TanitAD programme is not to refute hypotheses, its about achieving excellent results and really driving autonomously … we have too many refutes."* Everything below is written to produce a fan that is measurably safer than refcv3's was this morning.

---

## 0. What this package establishes, in eight lines

1. ⭐⭐ **The composed RL reward is NOT disqualified.** Within a window's own 128-candidate fan it ranks envelope-violating candidates **LOWER**: Spearman **ρ = −0.5367 [−0.5581, −0.5138]** over 240 EVAL windows / 121 episodes. The committed disqualification branch does not fire, and the reward line stays open.
2. ⭐ **The attribution names one term.** `progress` is the **only** component positively correlated with violation (ρ(progress, peak_g) **+0.2900**, ttc_below **+0.4697**, contact **+0.3286**, all separated). Quadrupling `feasibility` changes ρ(envelope) by **0.001**; deleting `progress` changes ρ(peak_g) by **0.27**. The lever is the progress term, exactly as the brief warned and against the obvious fix.
3. ⭐⭐ **The rule reward out-ranks refcv3's trained selector on feasibility**: the reward's argmax picks an envelope-violating candidate on **5.4 %** of windows vs the model selector's **10.8 %** and the fan's base rate of **88.8 %**.
4. **P2 delivered:** the veto is now three typed, recorded fields composed by `posttrain.veto_mask()`, with 8 pinning tests; the full RL suite is **172 passed**. A zero-weight arm with the veto off now reads `veto_rate_mean` **EXACTLY 0.0** (it read **0.0897** this morning).
5. ⛔⛔ **But a zero-weight, veto-off arm is STILL not a null — for a new reason, MEASURED here.** `ctrl_null` moved **35 of 57** fan-safety metrics with paired separation and moved all **71** trainable tensors by mean **|Δθ| 3.1e-05**. Its advantage is identically zero, so the only live loss is the **anchor trust region**, whose divergence starts at ~**1e-10 m²** — and **AdamW normalises by the gradient's own scale**, so a numerically negligible gradient still steps at order `lr`.
6. ⇒ **On this rig "separated vs base" is nearly uninformative on its own.** Every lever must clear **two** floors: the **seed-replicate** floor (`H-ESTIM-SEED-1`) and the **zero-information** floor (`ctrl_null`). Both are computed per metric by `raw/analyze_veto.py`.
7. **A defect found while building P1 and fixed in the same turn:** `readout()`'s `R1`/`R2` were a **batch × batch outer product** — every window's fan scored against every window's lead.
8. §5–§7: the veto-only arms, their two floors, and the T1 four families. §8: the 0-training selection rule that makes the *selected* path safer today, priced in ADE.

---

## 1. P1 — does the reward rank violating candidates higher? MEASURED: **NO, decisively**

`stack/scripts/rl_reward_envelope_rank.py`, one forward on the dev-box 4060. **240 EVAL windows / 121 episodes / 30,720 candidate scores.** Per-window Spearman ρ over the 128 EMITTED candidates (`anchor_traj`, 2 s prefix on the 0.5 s grid — the object the RL stage perturbs), aggregated by episode-cluster bootstrap, n_boot 4,000. Flags from `taniteval/tools/fan_safety.score_paths` — **one definition**, the same module the primary endpoint and the paired bootstrap use; no second friction derivation was written.

**A group-relative advantage IS a ranking**, so this is the question that decides admissibility: whatever the reward puts on top is where probability mass goes.

| ρ(DEFAULT reward, ·) | value | 95 % CI | n (windows / episodes) | undefined |
|---|---|---|---|---|
| `envelope` | **−0.5367** | [−0.5581, −0.5138] | 228 / 116 | 12 |
| `kamm_over` | **−0.5790** | [−0.6106, −0.5441] | 240 / 121 | 0 |
| `peak_g` (continuous friction load) | **−0.4603** | [−0.5472, −0.3698] | 240 / 121 | 0 |
| `infeasible` | −0.5297 | [−0.5506, −0.5078] | 228 / 116 | 12 |
| `ttc_below` (2.93 s, lead windows) | −0.3544 | [−0.4317, −0.2679] | 54 / 41 | 11 |
| `contact` (lead windows) | **−0.6326** | [−0.7423, −0.5184] | 19 / 14 | 46 |
| `off_reach` | −0.0756 | [−0.1437, −0.0089] | 181 / 101 | 59 |

⇒ **Every one is negative and separated. The reward prefers feasible candidates.** The SPEC's disqualification branch — *"if the reward ranks violating candidates HIGHER it is disqualified before any training run"* — **does not fire**, and the 2026-09-05 fan-safety regression cannot be laid at the reward's ordering of the envelope.

⚠️ **`undefined` is reported, not averaged away.** A window where all 128 candidates carry the same flag has no rank correlation; 12 windows are like that for `envelope`, 46 of the 65 lead windows for `contact`. Counting them as 0.0 would have manufactured a null.

### 1.1 The controls, each reading its known value

| control | reads | required |
|---|---|---|
| `peak_g` vs **itself** | **+1.0000** [+1.0000, +1.0000] | exactly +1 |
| **constant** score (identically 0) | **UNDEFINED on 240 / 240** | undefined, and counted |
| **random** uniform score | **−0.0138** [−0.0250, −0.0022] | ≈ 0 |

⚠️ The random control's −0.0138 is the probe's **own residual bias** (a fixed per-window seed against 240 correlated windows), and it is stated rather than rounded away: **|ρ| below ~0.02 is not interpretable here.** The effects in the table above are ~**39×** that.

### 1.2 The attribution — and it prices the obvious fix at zero

| component | ρ vs `envelope` | ρ vs `peak_g` | reading |
|---|---|---|---|
| **`progress`** | −0.0078 (ns) | **+0.2900** [+0.1811, +0.3966] | ⛔ **the only positive**: covering more ground in 2 s means accelerating harder |
| `comfort` | −0.5227 | **−0.9695** [−0.9747, −0.9640] | an almost perfect *inverse* ranker of friction load |
| `feasibility` | **−0.5389** | −0.6454 | ranks the envelope, as designed |
| `headway` | −0.2093 | −0.5410 | (lead windows) |
| `collision` | −0.0922 | −0.3588 | fires on 13 of 240 windows; the rest are undefined |

And `progress` is also the term that closes on the lead: ρ(progress, `ttc_below`) **+0.4697** [+0.3772, +0.5482] and ρ(progress, `contact`) **+0.3286** [+0.2732, +0.4039], both separated.

**The weight panel, scored on the same fan at zero extra GPU:**

| weights | ρ(·, envelope) | ρ(·, peak_g) | ρ(·, ttc_below) |
|---|---|---|---|
| `DEFAULT` | −0.5367 | −0.4603 | −0.3544 |
| `feasibility` ×4 (0.5 → 2.0) | **−0.5377** | −0.5386 | −0.3522 |
| **`progress` REMOVED** | −0.5371 | **−0.7306** | **−0.5632** |
| `progress` removed + `feasibility` ×4 | −0.5377 | −0.7232 | −0.5359 |
| `progress` alone | −0.0078 (ns) | **+0.2900** | **+0.4697** |
| `comfort` alone | −0.5227 | **−0.9695** | −0.5643 |

⇒ ⭐ **Quadrupling `feasibility` buys 0.001 of rank correlation. Deleting `progress` buys 0.27.** The brief's instruction — *"do not simply add a feasibility reward term"* — is now MEASURED rather than inferred, and it comes with the replacement: **the progress term is the one to change.**

⚠️ **This is a claim about RANKING, not about outcomes.** It says where the reward points, not what 2,000 gradient steps through a diffusion decoder will do with it. `H-RL-MIN-1`'s measured outcome (the planner ended below the constant-velocity floor) stands unchanged; what this removes is the explanation *"the reward wanted infeasible candidates"*. It did not.

### 1.3 The practical reads — and one of them is a product

| quantity | value | 95 % CI |
|---|---|---|
| fraction of the 128-fan violating the envelope | **0.8877** | [0.8768, 0.8988] |
| mean reward, violating **minus** feasible candidates | **−0.5316** | [−0.5479, −0.5174] |
| **top-1 by REWARD is envelope-violating** | **0.0542** | [0.0254, 0.0889] |
| top-1 by the **MODEL's selector** is envelope-violating | **0.1083** | [0.0669, 0.1583] |
| top-8 by reward violating | 0.1052 | [0.0660, 0.1510] |
| top-8 by model violating | 0.2490 | [0.2123, 0.2887] |

⭐⭐ **The rule reward is a better feasibility ranker than refcv3's trained scorer** — half the violation rate at top-1 and 2.4× better at top-8, with no training at all. §8 turns that into a measured product with its ADE price.

**Artifacts:** `raw/reward_envelope_rank.json` (panel · practical · 240 per-window rows), `raw/reward_envelope_rank.log`, and **`raw/fan_bank_base_240w.npz`** — the 240 × 128 fan with every reward component, every flag, `peak_g`, the selector score and the conf logits, **banked so any future reward design is scorable at 0 GPU**. The reason P1 needed a GPU at all is that nobody had banked the fan; that is now fixed.

---

## 2. P2 — the veto is a typed field, and a zero-weight control reads exactly zero

⛔ **The defect** (`D-RL-FANSAFE-2`, RETRACTION #24): `posttrain.rl_objective` composed the veto as `if "collision" in spec.weights:` — **key membership**, true at weight `0.0` — with the TTC channel reading no config at all, while `advantage.py:146` pins vetoed candidates at `−1.0` **outside** the group-relative centring. A constant reward therefore produced a **veto-only advantage at full strength**, not a zero one.

**The fix**, shipped and pinned:

* `PostTrainConfig` gains `veto_enabled` / `veto_collision` / `veto_ttc` (all default `True`), recorded into every run's `config.json` by `to_dict()` — MEASURED in this run's own record: `{"veto_enabled": false, "veto_collision": true, "veto_ttc": true}`.
* `posttrain.veto_mask(traj, ctx, cfg)` composes the mask from the config and **never** consults `spec.weights`. It returns an all-`False` **tensor** (never `None`) when off, so a `veto_rate` of `0.0` is still logged — *a channel that reports 0.0 is evidence; a channel that reports nothing is not.*
* `reg_echo` is pinned to `veto_collision=False`, which is what the old key-membership code actually gave it, so the banked arm still reproduces bit-for-bit.
* New arms: `veto200` / `veto2k` (zero reward, veto ON — the product) and `ctrl_null` (zero reward, veto OFF — the null).
* `stack/tests/test_rl_veto_explicit.py`, **8 tests**, asserting both directions and carrying a **same-breath control** in each that the constructed scene really does contain the violation being vetoed (an absence proved over an empty population is not a pass). **Full RL suite: 172 passed.**

**MEASURED, the headline pin, on the real rig:** `ctrl_null` (zero weights, veto OFF, 200 steps) reads

| | before P2 (`ctrl_const`) | after P2 (`ctrl_null`) |
|---|---|---|
| `veto_rate_mean` | **0.0897** | **0.0000** (exactly) |

⇒ **G1 PASSES.** The zero-weight control is a zero-*information* control again.

---

## 3. ⛔⛔ AND IT IS STILL NOT A NULL — the second mechanism, MEASURED here

`ctrl_null`'s advantage is **identically zero** (reward 0, veto rate 0), so the policy-gradient term contributes exactly nothing. It nevertheless:

| MEASURED on `ctrl_null` (200 steps, `w_anchor` 1.0, lr 1e-5) | value |
|---|---|
| `veto_rate_mean` | 0.0000 |
| `final_loss` (the anchor trust region alone) | **0.04459** |
| `weights_changed` | **True** |
| trainable tensors that moved | **71 of 71** (478 compared) |
| mean \|Δθ\| over decoder tensors | **3.1e-05** (max **5.7e-04**) against a typical \|θ\| of **1.8e-02** |
| fan-safety metrics separated vs base | **35 of 57** |
| `R_REACH` | −0.0417, separated · `R_ORACLE` +0.0062, separated · `R3` +0.0060 **ns** |
| selector agreement with base | 0.9917 |

**The mechanism, from measured facts rather than a guess.** The only surviving loss is the **anchor trust region**, whose step-0 divergence between the live model and its own frozen deepcopy is ~**9e-11 m²** (INHERITED, `…/refc-rl-readiness/RESULT.md` §3.2). A gradient that small should move nothing — **except that AdamW normalises by the gradient's own second moment**, so `m/(√v + ε)` ≈ `sign(g)` and the update is of order `lr` regardless of how small `g` is. Predicted per-step movement ~1e-5; MEASURED mean \|Δθ\| **3.1e-05** over 200 steps, i.e. steps of order `lr` that largely cancel as the trust region pulls back. A 1e-10-scale gradient producing a 1e-10-scale movement would have read ~1e-10; it read **3.1e-05**, five orders larger.

⇒ ⭐⭐ **On this rig, `ctrl0` (lr = 0) is the ONLY arm that cannot move. Any arm with a live optimizer and `w_anchor > 0` drifts whatever its reward says.** That makes "separated vs base" nearly uninformative by itself: a *zero-information* arm separates on **35 of 57** metrics — **more** than the `ctrl_const` arm whose 14-of-57 fired the previous panel's VOID gate.

⚠️ **Consequence, applied as a STRENGTHENING of the committed rule and never a loosening.** SPEC §7's SUCCESS/FAILURE text is unchanged and is reported as written (`quotable_as_lever`). A **second, stricter** verdict (`quotable_strict`) additionally requires the lever to exceed the `ctrl_null` **zero-information floor**. Both are in `raw/veto_verdict_*.json` for every metric; §5 reports both.

Root-cause class: the same family as `H-ESTIM-SEED-1` — **an estimator answering a narrower question than the claim hung on it** — with a new source of nuisance movement (the optimizer, not the sampler).

---

## 4. Provenance — the exact code every arm ran, and one mid-session drift recorded rather than absorbed

| item | value |
|---|---|
| base checkpoint | `ckpt_step40284_frozen.pt`, md5 `b1ed7075ff730d0993d2eaa3c86f6b56`, step asserted **40284** |
| tree that RAN | `C:\Users\Admin\refcv4b_repo` (the off-Drive clone; G: cannot run the stack) |
| `stack/tanitad/rl/rewards.py` in force | blob **`d9532aebe0b238d0d5f778b7560fbeaf3af78720`**, md5 `ae0ae5f1556ba5f597055206b7ae0790` |
| `posttrain.py` / `config.py` / `rl_refcv3_min.py` / `fan_safety.py` | **bit-identical to repo HEAD** (blob comparison, all 40-char) |
| fit corpus | 120 train-split B1 v7.2 clips, `fit120_lead_block.npz`, **NON-PARITY** (as the base is) |
| readout | the same fixed **120 EVAL windows** (`random.Random(1234)`), 79 episodes, 36 lead windows, for every arm |

⚠️ **A sibling stream changed `_collision` while this panel was running, and the panel was NOT re-synced.** Commit `9765634` (`…/2026-09-05-swept-collision/`) replaced the point-sampled contact test with a **swept-segment** one — *"it tested the waypoints, not the path between them"*. Every arm and both probes here ran the **pre-change, point-sampled** `_collision` (`d9532aeb…`), and the clone was deliberately left frozen for the rest of the panel so `s0` and `s1` remain comparable. ⇒ **What this scopes:** the `collision` reward component and the `contact` flag only. `envelope` / `kamm_over` / `off_reach` / `peak_g` — where every result below lives — do not call `_collision` at all. **What it predicts:** the swept test detects strictly more contacts, so the veto would fire *more* often under it; re-running this panel on the swept version should **increase** the veto's effect, not reverse it. Registered as the first follow-up in §9 rather than asserted here.

---

## 5. P3 — the veto-only arms at 200 steps, both seeds, against BOTH floors

`veto200`: reward weights **all 0.0**, `veto_enabled=True`, `w_anchor` 1.0, lr 1e-5, 200 steps, `two_scalar` noise, G 4, batch 2. Seed 0 in **214.7 s**, seed 1 in **150.8 s**. `veto_rate_mean` **0.0897** (s0) / **0.0980** (s1) — the constraint channel really fired, and s0 reproduces the banked `ctrl_const` 0.0897 to four decimals, which is the point: the accident is now a configuration.

**Guards.** G1 `ctrl_null` `veto_rate_mean` **0.0000** exactly ✅ · G2 every arm's BEFORE readout **bitwise identical**, max abs diff **0.000e+00** over 120 windows × 65 metrics ✅ · G3 both seeds present ✅.

| metric | base | Δ s0 | Δ s1 | seed floor | `ctrl_null` drift | veto − null (s0) | verdict |
|---|---|---|---|---|---|---|---|
| **`fan_peak_g_mean`** (g) | 4.18090 | **−0.09285** | **−0.11577** | 0.02292 | **+0.13431** | **−0.22716** ✱ | ⭐ **IMPROVED** |
| `top32_infeasible` | 0.63385 | −0.00640 | −0.00105 | 0.00534 | +0.00916 | −0.01556 ✱ | null |
| `top8_kamm_over` | 0.14271 | −0.01055 | −0.00264 | 0.00791 | +0.00158 | −0.01213 ✱ | null |
| `top32_envelope` | 0.63073 | −0.00620 | −0.00119 | 0.00501 | +0.00916 | −0.01536 ✱ | null |
| `top32_kamm_over` | 0.49141 | −0.00409 | −0.00283 | 0.00125 | +0.01088 | −0.01497 ✱ | null |
| `sel_infeasible` | 0.13333 | −0.00633 | −0.00633 | 0.00000 | −0.03165 | +0.02532 | null |
| **`sel_peak_g`** (g) | 0.19465 | **+0.01547** | **+0.00802** | 0.00745 | −0.00071 | +0.01619 ✱ | ⛔ **WORSENED** |
| `mass_rank_contact` | 3.47e-05 | 0 | 0 | 0 | 0 | 0 | **UNDETECTABLE-DOWNWARD** |

*(✱ = the paired veto-minus-`ctrl_null` contrast excludes 0. Full 57-metric table in `raw/veto_verdict_veto200.json`.)*

### 5.1 ⭐ What the replicate bought, stated plainly

**`top32_infeasible` is the metric the previous package proposed shipping on.** Seed 0 reads **−0.0064**; seed 1 reads **−0.0011**; the seed-replicate floor is **0.0053**. The effect is **not reproducible across training runs**, and three more metrics (`top8_kamm_over`, `top32_envelope`, `top8_infeasible`) fail the same way. On one seed every one of them would have been reported as a separated improvement — which is `H-ESTIM-SEED-1` doing exactly the job it was registered for, on a rig where a *zero-information* arm separates 35 of 57 metrics.

### 5.2 What survives, and what it costs

**`fan_peak_g_mean` clears all three hurdles**: separated at both seeds, same sign, magnitude 4–5× the seed floor, and the zero-information arm drifts it **the other way** (+0.134) so the drift cannot be the explanation. The direct veto-vs-`ctrl_null` contrast is **−0.227 g**, separated. In level terms the fan's mean peak friction load falls **4.1809 → 4.088 (s0) / 4.065 (s1) g**, a **2.2–2.8 %** reduction, from a constraint channel carrying **no reward at all**.

⛔ **And the cost is real and is reported, not buried.** `sel_peak_g` **WORSENED** and clears the same floors: **+0.0155 / +0.0080 g** against a seed floor of 0.0075, from a base of 0.1947 g — **+4 to +8 % on the path the car actually drives**, while the fan as a whole got blander. Selector agreement with base is **1.000 (s0) / 0.992 (s1)**: the fan moved under a selector that did not change its pick, so the selected candidate itself got slightly more aggressive. ⇒ **At 200 steps the veto buys fan-level feasibility and gives back a little selected-path feasibility.** A safety claim that quoted only `fan_peak_g_mean` would be true and misleading.

⚠️ **Stated limit:** `ctrl_null` has **one** seed, so the zero-information floor is itself unreplicated. It is used as a *direction* and a *magnitude* check, never as an estimate with its own interval.

---

## 9. Escalations, follow-ups and stated limits

**ESCALATIONS — raised here, in the report's own body, not written into a README for somebody to find.**

1. ⭐ **`stack/tanitad/rl/rewards.py` moved under this panel and the panel was deliberately NOT re-synced** (§4). Commit `9765634` replaced the point-sampled `_collision` with a swept-segment test. **Owner: Master Mind / Training FlyWheel** — the veto arms should be re-run on the swept version, where the constraint channel will fire *more*; the prediction is that the veto's effect grows. This is the cheapest next experiment in the line (~5 min per 200-step arm).
2. ⛔ **`readout()`'s `R1`/`R2` were a batch × batch outer product** (RETRACTION #26). Fixed here, but **any other consumer of `reward_ctx()` must be checked for its candidate rank** — the helper is shaped for `[B, N, G, S, 2]` and returns a plausible number for anything else. **Owner: whoever owns `rl_refcv3_min.py` next.**
3. ⛔⛔ **AdamW converts a numerically-zero gradient into a full-size step** (RETRACTION #27). This is not specific to the RL line: **any experiment on this programme that uses a trust-region / regulariser loss and expects "no signal ⇒ no movement" is exposed.** `ctrl0` (lr = 0) is the only construction that cannot move. **Owner: Master Mind** — it belongs in `CLAUDE.md`'s estimator family beside `H-ESTIM-SEED-1`.
4. **`strat.nav_compliance` still crashes** (`TypeError: _path_exists: path should be string`), taking out 3 of 3 STRATEGIC nav-compliance criteria rows on every refcv3 arm. Inherited, unfixed, re-confirmed here. **Owner: Benchmarks/Eval.**

**LIMITS — none silent.**

1. **NON-PARITY** fit corpus (120 train-split B1 v7.2 clips), as the base itself is.
2. The PRIMARY endpoint is a **T0 readout on 120 windows** — the fan the model emits, never a driving claim. T1 is self-action **OPEN loop** (PI ruling 2026-09-02); no closed-loop claim is made anywhere in this package.
3. `kamm_over` / `peak_g` are a **LOWER bound**: the emitted fan is free waypoints, so `flyability.friction_load` — the exact, control-rolled instrument — cannot be applied, and the finite difference under-reports by 1.21–1.85×. **Rank correlations (§1) are invariant to a monotone under-report; levels are not.**
4. **`ctrl_null` has ONE seed**, so the zero-information floor is itself unreplicated. It is used as a direction and a magnitude check, never as an estimate with its own interval.
5. Two seeds **bound** the rig's training noise; they do not make it small. `fan_peak_g_mean`'s seed-replicate floor is 0.0229 g against a lever of 0.093–0.116 g — a factor of 4–5, not a factor of 100.
6. The P1 rank correlations for the `collision` component and the `contact` flag are computed under the **point-sampled** `_collision` (§4). The `envelope` / `kamm_over` / `off_reach` / `peak_g` results do not call it.
7. `mass_rank_contact` is **UNDETECTABLE-DOWNWARD** on this rig: its base value (3.47e-05) sits below its own separation floor, so an improvement could not be reported even if it occurred. Stated rather than reported as a null.
