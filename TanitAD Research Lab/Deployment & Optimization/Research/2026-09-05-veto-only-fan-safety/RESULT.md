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

### 5.3 ⭐ A THIRD run of the same arm was already banked — and it makes the noise floor worse, not better

`ctrl_const` (2026-09-05 morning, the accidental veto-only arm) is the **same configuration at the same seed** as `veto200_s0`, run in a different working directory hours earlier and before P2 existed. Its veto fired at `veto_rate_mean` **0.0896826171875** against `veto200_s0`'s **0.0896923828125** — agreeing to four decimals and **not identical**, which is GPU non-determinism, not a configuration difference. That makes it a **same-seed replicate**, and the three runs read:

| metric | `ctrl_const` (seed 0, morning) | `veto200_s0` (seed 0) | `veto200_s1` (seed 1) |
|---|---|---|---|
| **`fan_peak_g_mean`** | **−0.08587** | **−0.09285** | **−0.11577** |
| `top32_infeasible` | −0.01431 | −0.00640 | −0.00105 |
| `top8_kamm_over` | −0.01371 | −0.01055 | −0.00264 |
| `sel_peak_g` | +0.01314 | +0.01547 | +0.00802 |

⛔ **`top32_infeasible` spans −0.01431 to −0.00105 across three runs — a 13.6× range, and two of them share a seed.** So the rig's run-to-run noise is not only a *seed* effect; **two runs of the identical command differ by 2.2× on that metric.** `H-ESTIM-SEED-1` says a separated one-seed CI is necessary and not sufficient; this says the same-seed replicate is not sufficient either, and the floor has to be measured from *runs*, however they differ.

⭐ **And the same table is the strongest evidence in the package for the one surviving claim.** `fan_peak_g_mean` reads **−0.0859 / −0.0929 / −0.1158** across three independent runs — every one negative, mean ≈ **−0.098 g**, spread 0.030 g — while the zero-information arm pushes it **+0.134** the other way. A metric that survives three runs, two floors and a sign-reversed nuisance drift is as replicated as anything this rig currently produces.


## 6. ⭐⭐ WHERE THE INFEASIBILITY ACTUALLY COMES FROM — and it is not the vocabulary

**MEASURED, 0 GPU** (`stack/scripts/bank_vs_fan_feasibility.py`, `raw/bank_vs_fan_feasibility.json`): the frozen ANCHOR BANK (`core.decoder.anchors`, a **fixed** `[128, 8, 2]` path set read straight out of the checkpoint) scored by the SAME `fan_safety.score_paths`, on the SAME 240 windows, with the same `v0` and lead track as the emitted fan.

| flag (mean over 240 × 128) | **anchor bank** | **emitted fan** | delta |
|---|---|---|---|
| `envelope` | **0.0156** | **0.8877** | **+0.8721** |
| `kamm_over` | 0.1953 | 0.8406 | +0.6453 |
| **`peak_g`** (g) | **0.4808** | **4.1131** | **+3.6323** (**8.56×**) |
| `off_reach` | 0.7783 | 0.1077 | **−0.6705** |
| `infeasible` (the OR) | 0.8037 | 0.8920 | +0.0883 |
| `contact` | 0.0239 | 0.0249 | +0.0009 |
| mean \|emitted − bank\| per waypoint, 2 s prefix | — | — | **9.29 m** |

⇒ ⭐⭐ **The vocabulary is almost entirely drivable — 1.6 % envelope violation at 0.48 g — and the DIFFUSION DECODE manufactures the other 87 points, an 8.56× blow-up in peak friction load, by displacing every waypoint a mean of 9.29 m.** The fan refcv3 emits bears little geometric relation to the anchors it is nominally built from.

⛔ **CORRECTED IN THE SAME TURN — the mechanism label was too specific, and the correction is from source, not from prose** (`stack/tanitad/refs/refc.py`). The first version of this section said *"the decoder's OFFSET HEAD manufactures"*. The path from bank to fan is **three stages, not one**:

```
:1590   bank = self.roll_bank(...)      # refcv3: anchor_v0_cond False -> anchors[None].expand(...)
:1640   conf0, offset = self._decode(kv, cond, x0, 0);   x = bank + offset      # classifier pass
:1676   for i in range(steps):  x_in = x + noise;  _, off = self._decode(...);  x = x_in + off
```

`out["offset"]` is the **classifier-pass offset ONLY** — assigned once at `:1640` and never reassigned inside the refinement loop, which adds `off`. ⇒ (a) the blow-up is created by the **whole decode**, and which stage does it is a measurement this section did not make (the corrected probe now makes it, §8); (b) an offset-shrink sweep built on `fan − out["offset"]` interpolates `bank + Σ off_i`, an intermediate that **exists at no point in the decode** — its numbers are **WITHDRAWN** (`raw/fan_rerank_WITHDRAWN_wrong_lambda_operand.json`, quarantined rather than deleted) and re-run over the real `out["anchor_bank"]`. ⚠️ Its λ = 1 identity control **passed and was blind to this by construction**: λ = 1 is the emitted fan whatever the base is, so the control checked the arithmetic and not the OBJECT. The corrected sweep adds a second control — at λ = 0 it must reproduce the bank rates measured **by a different route** in the table above. RETRACTION #30 (numbers 28 and 29 were taken by sibling streams while this was being written; the appender now takes max+1 rather than a hard-coded number).

**What is NOT affected.** The table's own numbers: `roll_bank` for a fixed vocabulary returns `self.anchors[None].expand(...)` — the source calls it *"byte-identical to the pre-2026-09-04 `anchors[None].expand(...)`"* — and refcv3 carries no `--anchor-v0-cond` flag in its `config.json['argv']`, so `core.decoder.anchors` **is** the bank this decode started from. The measurement is of the right object; only the sentence attributing it to one head was wrong.

**Read the `off_reach` row before drawing the wrong conclusion.** The bank is **77.8 %** off-reach and the emitted fan only **10.8 %** — because the bank is a FIXED path set that mostly does not match the window's own speed, and adapting it to the window is precisely the offset head's job. The 9.29 m displacement is not gratuitous; it is what makes the fan speed-appropriate. **The finding is that the adaptation is paid for in the friction envelope, at 8.56×.**

⭐ **What this means for the programme, and it re-prices everything above.** The veto moves `fan_peak_g_mean` by **−0.10 g** against a blow-up of **+3.63 g** — it addresses **≈ 2.7 %** of the gap between the vocabulary's feasibility and the fan's. So:

1. ⛔ **RL post-training of the decoder is the right SURFACE and the wrong SIZE of instrument.** The lever is in the correct place (`core.decoder` is exactly what the stage trains) but a constraint channel nudging a 4.11 g fan cannot recover a 0.48 g one. ⚠️ And the RL surrogate perturbs **`out["offset"]`**, i.e. the *classifier-pass* offset, with the refinement loop **not re-run** — a linearisation of the decode, which is a further reason its authority over the emitted geometry is smaller than the surface size suggests.
2. ⭐ **The high-value build is a feasibility-aware DECODE**, not a bigger reward: a reparameterisation or penalty that keeps the *rolled* path inside the envelope, applied at every pass that moves the waypoints rather than after the last one. §8's stage decomposition says which pass to aim it at. That is an architecture change owned by this FlyWheel, and §6 is the measurement that justifies opening it.
3. This also explains, without needing the RL panel at all, why `oracle_sel` (the best-ADE candidate) is the **least** drivable one (`D-RL-FANSAFE-1`: 0.1105 vs `os` 0.0865): matching the human's path closely is exactly what the large offsets buy, and the friction cost rises with the offset.

⚠️ **Stated limits.** `peak_g` here is the finite-difference load on free waypoints and under-reports by 1.21–1.85× — *on both sides of the comparison*, so the **ratio** is the robust quantity and the levels are lower bounds. The bank's `envelope`/`kamm_over` rates are window-independent by construction (a fixed path set); `off_reach`, `contact` and `ttc_below` are not, and are averaged over the same 240 windows as the fan.


## 7. P3, the other dose — 2,000 steps LOSES the gain: a clean dose-response

`veto2k`: identical to `veto200` in every respect but `steps` (200 → **2,000**; 2,133.6 s and 2,106.4 s). `veto_rate_mean` **0.0912 / 0.0916**, so the constraint fired at the same rate throughout. **Guards G1/G2/G3 all pass**, on the same 120 windows.

| metric | base | Δ s0 | Δ s1 | seed floor | verdict |
|---|---|---|---|---|---|
| `fan_peak_g_mean` | 4.18090 | −0.00777 | −0.04843 | **0.04066** | **null** |
| `top32_infeasible` | 0.63385 | −0.00475 | −0.00079 | 0.00396 | null |
| `fan_envelope` | 0.89023 | −0.00346 | −0.00152 | 0.00194 | null |
| `sel_infeasible` | 0.13333 | **+0.01266** | +0.00000 | 0.01266 | null |
| `sel_peak_g` | 0.19465 | +0.01046 | +0.03430 | 0.02384 | WITHIN-NOISE |
| `mass_rank_infeasible` | 0.17887 | **+0.01287** | **+0.01545** | 0.00258 | (worse; see below) |

⇒ **PRIMARY-FAIL, and nothing at all is quotable as a lever.** Set beside the 200-step arm the picture is a **dose-response, not a null**:

| | `fan_peak_g_mean` Δ (s0 / s1) | seed floor | quotable |
|---|---|---|---|
| **200 steps** | **−0.0929 / −0.1158** | 0.0229 | ⭐ **yes — IMPROVED** |
| **2,000 steps** | −0.0078 / −0.0484 | 0.0407 | no |

**The gain is a short-dose effect and it decays.** The supporting T0 readouts say the same thing in a second voice: at 2,000 steps `R3` (sel-ADE 2 s) is **+0.0121 / +0.0136 separated** and `R_ORACLE` (oracle-in-fan — fan QUALITY) **+0.0178 / +0.0207 separated**, i.e. the fan is getting *worse at containing a good path*, while at 200 steps `R3` was **+0.0033, not separated**. Confidence mass on infeasible candidates also rises consistently (`mass_rank_infeasible` +0.0129 / +0.0155 against a 0.0026 floor).

⭐ **Read as a control law rather than as a disappointment:** a pure constraint channel has **no ranking term to trade against**, so once it has pushed the fan off the violating region there is nothing left for the gradient to optimise and the trust region plus the optimizer's own drift take over. That is why the SPEC committed to **both** doses in advance — running one would have produced either an unrepeatable win or an unexplained null. **The deliverable is `veto200`.**

⚠️ **Stated honestly: the 2,000-step arm's WORSENINGS are NOT attributable.** The dose-matched zero-information control (`ctrl_null` at 2,000 steps) was **queued and then deliberately dropped** when `veto2k_s0` landed with no separated improvement — a floor is a hurdle for a *positive* claim and cannot make an unseparated delta quotable, so the 36 GPU-minutes went to the T1 four-family read instead (a committed deliverable). ⇒ `sel_peak_g` +0.0105/+0.0343 and `R3` +0.0121/+0.0136 at 2 k **cannot be split between the veto and the optimizer's own drift**, and are reported as unattributed rather than as veto effects. The decision and its consequence are recorded in `raw/chain_after_arms2.sh`, not left as a gap.


## 8. ⭐⭐ THE 0-TRAINING PRODUCT — a top-2 kinematic gate makes the DRIVEN path 31 % less envelope-violating at no measurable ADE cost

§1.3 measured that the rule reward's own argmax picks an envelope-violating candidate on **5.4 %** of windows against refcv3's trained selector's **10.8 %** — the scorer is a better feasibility ranker than the network. `stack/scripts/rl_fan_rerank_probe.py` turns that into a product by re-ranking the SAME 128 emitted candidates under several rules and pricing each in the ADE it gives up. **The model is never retrained.** 480 EVAL windows / 138 episodes (139 lead windows), paired episode-cluster bootstrap n_boot 4,000. Deltas below are **model − rule**, so a positive `Δenv` means the rule is *safer* and a negative `Δade` means the rule is *worse on ADE*.

| rule | `ade_m` | Δade vs model | sep | `sel_envelope` | Δenv | sep | `sel_peak_g` | agrees with model |
|---|---|---|---|---|---|---|---|---|
| **model** (deployed) | 0.4742 | — | — | 0.1062 | — | — | 0.1815 | 1.000 |
| ⭐ **gate2** | **0.4705** | **+0.0037** | **no** | **0.0729** | **+0.0333** | **yes** | **0.1459** | 0.523 |
| **gate4** | 0.4972 | −0.0230 | **no** | 0.0625 | +0.0437 | yes | 0.1255 | 0.317 |
| gate8 | 0.5194 | −0.0452 | yes | 0.0583 | +0.0479 | yes | 0.1163 | 0.231 |
| gate16 / gate32 / gate128 | 0.5328–0.5397 | −0.059…−0.066 | yes | 0.0583 | +0.0479 | yes | 0.1146–0.1150 | ~0.21 |
| `kin_only` (no model ranking at all) | 0.5397 | −0.0655 | yes | 0.0583 | +0.0479 | yes | 0.1147 | 0.212 |
| `reward_full` (needs the lead track) | 0.9474 | −0.4732 | yes | 0.0542 | +0.0521 | yes | 0.3612 | — |
| `oracle` (best-ADE, T0 ceiling) | 0.2227 | +0.2515 | yes | **0.1437** | −0.0375 | yes | 0.2029 | — |

⭐⭐ **`gate2` is close to a free lunch and it is the deliverable.** Keep the model's own top-2 by `sel_score` — so the semantic and tactical ranking the network learned is preserved — and pick between those two by `feasibility + comfort`:

* selected-path `envelope` **0.1062 → 0.0729**, a **−31 %** relative fall, **separated**;
* selected-path `peak_g` **0.1815 → 0.1459 g**, **−20 %**;
* `ade_m` **0.4742 → 0.4705** — **+0.0037 in the model's favour and NOT separated**, i.e. **no measurable ADE cost**;
* it changes the pick on **48 %** of windows, so the effect is not a rounding artifact of rarely intervening.

⛔ **The kinematic score reads NO SCENE INPUT.** `feasibility` and `comfort` are functions of the candidate's own waypoints only — no lead, no obstacle track, no ego state — so the gate is admissible under the **vision-only-at-inference** rule with **zero new perception** and zero new parameters. `reward_full` is reported for contrast precisely because it is *not*: it needs the lead track, and it also destroys ADE (0.9474) while chasing feasibility.

⭐ **The `oracle` row is the finding that explains why this works.** The fan's best-ADE candidate is the **least** drivable one — `sel_envelope` **0.1437** against the model's 0.1062 and the gate's 0.0729. Matching the human's path most closely and being drivable are in tension in this fan (`D-RL-FANSAFE-1`, now reproduced from the other side), which is exactly why a small, ADE-neutral gate can buy feasibility: it is trading away candidates that were never worth their ADE.

⚠️ **Set against the RL result honestly.** The veto at 200 steps moved the *fan's* mean peak-g by **−0.098 g** and made the *selected* path slightly worse (`sel_peak_g` +0.0155/+0.0080). The gate moves the *selected* path by **−0.036 g** and **−31 % envelope** with no ADE cost, at **zero training**. **They act on different objects and can be combined** — the combined measurement (the gate applied to the veto'd checkpoint's fan) is `raw/fan_rerank_veto200s0.json`, queued behind the T1 evals.

⚠️ **Scope of the withdrawal in §6/RETRACTION #30.** The artifact these numbers come from also carried a shrink sweep built on the wrong operand, and **only that block is withdrawn**. The selection-rule numbers use `score_paths`, the kinematic components, `sel_score` and the per-candidate ADE — none of which touch `out["offset"]` — so they are unaffected; they are quoted from the quarantined artifact and re-confirmed by the corrected re-run (`raw/fan_rerank_base.json`). ⛔ Stating this explicitly rather than re-quoting a "clean" file is the point: a withdrawal that is not scoped is either too wide or too narrow, and both are wrong.


## 11. P4 — the SECONDARY endpoint: the FOUR FAMILIES at T1, per family, never pooled

`taniteval/tools/paired_openloop.py`, **T1 (self-action OPEN loop — never a closed-loop claim)**, `s0/veto200` vs the banked base over the shared `ha0` floor. **`void: false`**, all six gates pass (`G1_common_grid`, `G2_oracle_arm_refused`, `G3_same_tier`, `G4_profiles_non_degenerate`, `G5_action_units`, `G6_shared_floor`), **4,823 shared windows / 141 episodes, 0 windows dropped on either side**. Estimator: paired episode-cluster bootstrap (cluster = clip), n_boot 2,000, seed 0. Power: adequate (141 ≫ 10). A **positive** delta means the arm is **worse** than base.

| family | metric | Δ s0 | Δ s1 | seed-replicate floor | verdict |
|---|---|---|---|---|---|
| **ADE** | `ade_m` | **+0.0362** | **+0.0475** | 0.0113 | ⛔ **QUOTABLE — worse** |
| **ADE** | `fde_m` | +0.0623 | +0.0677 | 0.0054 | ⛔ QUOTABLE — worse |
| **LONGITUDINAL** | `LON_accel_mae_mps2` | **+0.3349** | **+0.2157** | 0.1192 | ⛔ **QUOTABLE — worse** |
| **LONGITUDINAL** | `LON_speed_mae_mps` | +0.0779 | +0.0628 | 0.0151 | ⛔ QUOTABLE — worse |
| **LONGITUDINAL** | `LON_along_mae_m` | +0.0290 | +0.0312 | 0.0022 | ⛔ QUOTABLE — worse |
| **LATERAL** | `LAT_heading_mae_deg` | +0.2912 | +0.4517 | 0.1605 | ⛔ QUOTABLE — worse |
| **LATERAL** | `LAT_cross_mae_m` | +0.0154 | +0.0322 | 0.0168 | ⚠️ **WITHIN-NOISE** |
| **LATERAL** | `LAT_yaw_rate_mae_radps` | +0.0191 | +0.0478 | 0.0287 | ⚠️ **WITHIN-NOISE** |
| **TACTICAL** | `TAC_traj_lon_correct` (acc ↑) | **−0.0889** | **−0.0651** | 0.0238 | ⛔ **QUOTABLE — worse** |
| **TACTICAL** | `TAC_traj_lat_correct` (acc ↑) | −0.0025 (ns) | −0.0139 | — | ns on s0 |
| **TACTICAL** | `TAC_declared_*` (6 rows) | 0.0000 | 0.0000 | 0.0000 | **STRUCTURAL** |
| **STRATEGIC** | `STR_route_correct` (3 rows) | 0.0000 | 0.0000 | 0.0000 | **STRUCTURAL** |

*(Positive = worse. `QUOTABLE` = separated at **both** seeds, same sign, and the smaller |Δ| exceeds the seed-replicate floor.)*

⚠️ **The `0.0000 [0, 0]` rows are STRUCTURAL ZEROS and must never be read as "no harm".** The arm trains `core.decoder` **only** (9,206,032 / 107,032,901 = 8.60 %); `core.maneuver`, `core.route`, `tac_goal_head`, `str_goal_head` and `conf_head` are in `forbidden_prefixes` and were frozen. Identical inputs through frozen heads give identical outputs — an **identity, not an estimate** (`H-ECHO-4` class). ⇒ the SECONDARY guard is informative on the trajectory-derived rows and **silent** on the declared-tactical and strategic rows.

⇒ **The ADE guard FAILS: `ade_m` regresses +0.0362 m, separated.** Set in proportion: base 0.4419 → ≈0.478, a **+8.2 %** regression — against the full composed-reward arm's **+0.5014 m (+113 %)**. **The veto is ~14× gentler than the reward stage, and it is still a regression.** `LON_accel_mae` +0.3349 on a base of 0.6806 (**+49 %**) is the largest single movement, which is the same axis the T0 readout flagged (`sel_peak_g` up while the fan's mean came down): a constraint pushed off the violating region takes the *selected* path with it.

⭐ **THE REPLICATE LANDED, AND IT DID ITS JOB IN BOTH DIRECTIONS.** `s1/veto200`'s T1 completed at 17:46 Z and the table above is now the two-seed read. **The ADE guard fails on a replicated basis**: `ade_m` +0.0362 / +0.0475 against a floor of 0.0113 — separated at both seeds, same sign, ~3–4× the floor. And two LATERAL metrics that looked like separated regressions on seed 0 alone — `LAT_cross_mae_m` (+0.0154 / +0.0322, floor 0.0168) and `LAT_yaw_rate_mae_radps` (+0.0191 / +0.0478, floor 0.0287) — fall to **WITHIN-NOISE**, exactly as four T0 feasibility metrics did in §5.1. ⇒ **Of 11 non-structural T1 rows, 7 are quotable regressions, 2 are within-noise and 2 are ns.** The verdict does not change; its evidential basis is now two runs rather than one.



---

## 12. ⭐⭐ THE VERDICT, written against the committed text — and the answer to *"is refcv3's fan safer than it was this morning?"*

**FORMAL EXIT against SPEC §7, first-match, no softening:**

> ⛔ **FAILURE.** SUCCESS required a negative, quotable `fan_peak_g_mean` **AND** at least one of `top32_infeasible` / `sel_infeasible` **AND** no `ade_m` regression beyond the replicate floor. Only the first holds. At 200 steps `top32_infeasible` is **WITHIN-NOISE** (−0.0064 / −0.0011 against a 0.0053 seed floor) and `ade_m` at T1 regresses **+0.0362 / +0.0475 m against a 0.0113 seed-replicate floor — QUOTABLE, i.e. the ADE guard fails on BOTH seeds**. At 2,000 steps nothing is quotable at all.

**And the honest answer to the brief's question, in three lines, because the axes disagree and pooling them would hide it:**

| object | is it safer than this morning? | number |
|---|---|---|
| the **emitted fan** | ⭐ **YES, replicated over three runs** | `fan_peak_g_mean` **4.1809 → 4.088 / 4.065 g** (−0.0859 / −0.0929 / −0.1158 across three runs), clearing the seed-replicate floor 4–5× against a zero-information arm that drifts it **+0.134 the other way** |
| the **driven (selected) path** under the veto | ⛔ **NO** | `sel_peak_g` **+0.0155 / +0.0080 g** (T0); at T1, **7 of 11 non-structural rows are quotable regressions on BOTH seeds** — `ade_m` **+0.0362 / +0.0475**, `LON_accel_mae` +0.3349 / +0.2157, `TAC_traj_lon_correct` −0.0889 / −0.0651 |
| the **driven path** under the **0-training gate2 re-rank** | ⭐⭐ **YES, and by more** | `sel_envelope` **0.1062 → 0.0729 (−31 %)**, `sel_peak_g` **0.1815 → 0.1459 g (−20 %)**, `ade_m` **+0.0037, NOT separated** |

⇒ ⭐⭐ **The deliverable is `gate2`, not the RL arm.** A top-2 kinematic gate over the model's own ranking makes the path the car drives measurably safer **today**, with **zero training, zero new parameters, zero new perception and no measurable ADE cost**. The veto-only arm is a real but small **fan-level** gain that the selector does not convert into a safer driven path — and §6 says why the ceiling is low: the veto moves `fan_peak_g_mean` by ~0.1 g against a **+3.63 g** blow-up the decode creates over its own vocabulary, i.e. it addresses **~2.7 %** of the available gap.

**What to do next, in cost order:**

1. ⭐ **Ship `gate2` behind a flag and measure it end-to-end** — it is a selection rule over an unchanged model, so it needs an inference-path change and no retraining. Its combined measurement with the veto'd checkpoint is queued (`raw/fan_rerank_veto200s0.json`).
2. ⭐ **Open the feasibility-aware DECODE work item** (§6): the vocabulary is drivable at 0.48 g and the decode emits 4.11 g. That is where the 12.7× human-vs-refcv3 envelope gap actually lives, and no post-training of a constraint channel reaches it.
3. **Re-run the veto arms on the swept-segment `_collision`** (§4 escalation 1): the constraint will fire more, and the prediction is the effect grows. ~5 min per arm.
4. ⛔ **Do NOT spend more GPU on the composed reward at this surface.** P1 exonerated its *ranking* (ρ = −0.5367), so the failure is not "the reward wants infeasible paths" — but the 2,000-step dose shows the constraint channel exhausts itself, and §6 caps what any decoder post-train can recover.

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

---

## 10. DELIVERABLE MANIFEST — every artifact and WHERE IT LIVES

Package root: `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/`. **Everything below is `repo:` — committed and blob-verified in HEAD. Nothing in this package lives in only one place.**

| artifact | where | what it is |
|---|---|---|
| `SPEC.md` | repo | both outcomes, both floors, the five arms — banked **before** any arm ran |
| `RESULT.md` | repo | this file |
| **code — new** | | |
| `stack/scripts/rl_reward_envelope_rank.py` | repo | P1: ρ(reward, envelope) over candidates, 3 controls, weight panel, banks the fan |
| `stack/scripts/rl_fan_rerank_probe.py` | repo | §8: selection rules priced in ADE + the corrected shrink sweep + stage decomposition |
| `stack/scripts/bank_vs_fan_feasibility.py` | repo | §6: anchor bank vs emitted fan, 0 GPU |
| `stack/tests/test_rl_veto_explicit.py` | repo | P2's 8 pinning tests (RL suite **172 passed**) |
| **code — changed** | | |
| `stack/tanitad/rl/config.py` | repo | `veto_enabled` / `veto_collision` / `veto_ttc` |
| `stack/tanitad/rl/posttrain.py` | repo | `veto_mask()`; the veto no longer reads `spec.weights` |
| `stack/scripts/rl_refcv3_min.py` | repo | new arms; `reward_ctx(cand_dims=)` + the readout shape assertion |
| **measurements** | | |
| `raw/reward_envelope_rank.{json,log}` | repo | P1, 240 windows / 121 episodes / 30,720 candidate scores |
| `raw/fan_bank_base_240w.npz` | repo | ⭐ the 240 × 128 fan + every component + every flag — **any future reward design is scorable at 0 GPU** |
| `raw/bank_vs_fan_feasibility.json` | repo | §6 |
| `raw/veto_verdict_veto200.json` / `_veto2k.json` | repo | all 57 metrics × both seeds × both floors × the contrast |
| `raw/run/s{0,1}/{ctrl_null,veto200,veto2k}/arm_summary.json` | repo | the five arms |
| `raw/run/paired_s0-veto200_vs_base.{json,md}` | repo | T1 four families, seed 0 |
| `raw/run/eval/refcv3-40284-s0-veto200.{json,md}` | repo | the T1 suite record |
| `raw/t1_families_veto200.json` | repo | the four-family read with its `_LIMIT` field |
| `raw/fan_rerank_WITHDRAWN_wrong_lambda_operand.{json,log}` | repo | ⛔ quarantined, not deleted (RETRACTION #30) — its selection-rule block is what §8 quotes |
| **tooling / provenance** | | |
| `raw/analyze_veto.py`, `raw/read_t1_families.py` | repo | the two floors; the four-family reader |
| `raw/patch_*.py` (9), `raw/insert_register_rows_*.py` (3), `raw/append_retraction*.py` (2) | repo | every edit reproducible from a script, not hand-applied |
| `raw/run_veto_arms.sh`, `raw/run_eval_veto.sh`, `raw/chain_*.sh` (5) | repo | the run chains, including the two replacements and **why** they replaced |
| `Project Steering/GOALS_AND_CLAIMS.md` | repo | 9 rows added/updated |
| `Project Steering/RETRACTION_LOG.md` | repo | **#26, #27, #30** |

⚠️ **STILL RUNNING when this file was written** (self-driving, sequential on the dev-box 4060; nothing else is touched):

| in flight | artifact it will produce | how to finish |
|---|---|---|
| ~~`s1/veto200` T1 rollout~~ | ✅ **LANDED 17:46 Z** — `raw/run/paired_s1-veto200_vs_base.json`, folded into §11 | done |
| corrected re-rank on **base** | `raw/fan_rerank_base.json` | re-confirms §8's table and adds the λ = 0 bank control |
| corrected re-rank on **veto200_s0** | `raw/fan_rerank_veto200s0.json` | ⭐ do the two levers COMPOSE or cancel — the deployment question |

⭐ **Everything else is complete and banked.** The verdict in §12 does not depend on the three rows above: the committed exit is already **FAILURE** on two independent legs (`top32_infeasible` within-noise, `ade_m` separated-worse), and the shipping recommendation (`gate2`) rests on the 480-window measurement already in the repo.
