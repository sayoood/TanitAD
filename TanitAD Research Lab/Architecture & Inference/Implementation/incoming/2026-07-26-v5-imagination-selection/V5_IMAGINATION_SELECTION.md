# V5 — imagination-scored selection over the frozen v4 fan

**Stream:** v5 synthesis (PI directive, 2026-07-26, after Bar A refuted the discriminative selector).
**Host:** `tanitad-eval` (`1e0bac0df88a`). **Corpus:** the canonical 40-episode / 881-window val
deployment. **Surface:** `produced` (deployable) unless a row says `oracle`.

> ⛔ **Everything above the line `--- MEASUREMENTS BEGIN ---` was written and staged BEFORE any
> number in this experiment existed.** BOOST_PROGRAM M2/M3.

---

## 0. PRE-REGISTRATION

### 0.1 The hypothesis, and why Bar A does not already answer it

Bar A established (MEASURED, `…/2026-07-26-bar-a-selector/raw/bar_a_produced.json`) that **re-scoring
the frozen v4 fan with a learned head over the existing latent features cannot reach v1**: fitting
*and* scoring on the **same** 6,844 windows — maximum overfit, zero generalization gap, 796 k free
parameters — bottoms out at **0.4907** (CE) / **0.5224** (regret) against v1's **0.4271**. Out of
fold both arms were *worse than doing nothing* (−4.2 % / −11.0 % of the waste recovered).

Four things Bar A did **not** establish:

1. **that the fan is bad** — it is good: `oracle_in_fan` = **0.2505** on the deployable surface,
   41.3 % better than v1's 0.4271;
2. **that a different selector INPUT fails** — only the existing feature set (`q_final`,
   `state_last`) was tested;
3. **that SIMULATIVE selection fails** — Bar A tested a *discriminative* scorer, which reads
   features and emits a rank;
4. **that HIERARCHICAL selection fails** — a flat 1-of-256 scorer was tested.

⇒ **H-V5: the information a discriminative head could not extract from the features is recoverable
by ROLLING EACH CANDIDATE FORWARD THROUGH THE WORLD MODEL AND LOOKING AT WHAT HAPPENS.**

### 0.2 The instrument — reused, not rebuilt

`tanitad/models/metric_dynamics.py::rollout_decode` advances its window by appending the model's
**own predicted latent** (`win_s = cat([win_s[:, 1:], z_hat])`); **no new frame is ever encoded** and
`k` is free. Verified in the pod's live tree (`/root/v4eval/stack`), which is byte-identical to repo
HEAD for this module modulo line endings (§1.2). It is a per-candidate imagination roll-out as it
stands.

### 0.3 Candidate → action sequence (the one new piece of mechanism)

The v4 fan is 256 **trajectories** (`[B, 256, 20, 2]`, dense 0.1 s ego waypoints), not action
sequences. `rollout_decode` needs actions. The map is the *exact inverse of the corpus's own action
definition* (`tanitad/data/physicalai.py:51,412` — `WHEELBASE = 2.9`,
`steer = arctan(WHEELBASE · curvature)`, `accel` = the dataset's accel), so it introduces no new
convention:

```
dt = 0.1 ;  p = [ (0,0) , w_c[0..19] ]                        # 21 points
d_j   = p_j - p_{j-1}                       v_j = ||d_j|| / dt        (v_0 := observed v0)
psi_j = atan2(d_j.y, d_j.x)  (held when ||d_j|| < 1e-3)      psi_0 = 0
omega_j = wrap(psi_j - psi_{j-1}) / dt      kappa_j = omega_j / max(v_j, 0.5)
steer_j = clip(arctan(2.9 * kappa_j), ±MAX_STEER_RAD)
accel_j = (v_j - v_{j-1}) / dt
3rd channel = v0 / SPEED_SCALE (=10.0), constant — the leakage-safe v1 contract
```

**Rollout convention (recorded so it is falsifiable):** transition 1 is driven by `win_a[:, -1]`, so
the candidate takes control of it by **overwriting the last window action with `a_c[0]`**;
`future_actions = a_c[1:]` then drives transitions 2..20. The candidate therefore controls the whole
2 s roll-out and nothing else changes.

### 0.4 The arms — frozen before measurement. NO TRAINING in the first pass.

| id | rule (argmin/argmax over the 256 frozen candidates) | uses WM? | per-candidate roll? |
|---|---|---|---|
| **A0** | `as_trained` — v4's own selector score. The thing to beat. | – | – |
| **A1** ⭐ | **`imag_consistency`** — argmin_c mean_j ‖τ̂_c[j] − w_c[j]‖. *"Does the world model agree that executing this plan produces this plan?"* **PRIMARY.** | ✅ | ✅ |
| **A2** | `imag_goal_speed` — argmin_c \|v̂_term(τ̂_c) − v_goal\|, `v_goal` = produced `vt_speed` clamped to the same reachable set `select()` uses. | ✅ | ✅ |
| **A3** | `imag_kinematic` — argmin_c mean plausibility (‖Δ²τ̂_c‖, i.e. imagined accel/jerk magnitude). | ✅ | ✅ |
| **A4** | `imag_combo` — weighted sum of A1–A3 + the normalised base score. Weights fitted **(i) in-sample** (comparable to Bar A's in-sample ceiling) and **(ii) 5-fold episode-disjoint out-of-fold** (deployable). | ✅ | ✅ |
| **C1** | **CONTROL, no WM** — A1 with τ̂ replaced by a CTRV/constant-velocity forward model. | ❌ | ✅ |
| **C2** | **CONTROL, one imagination** — argmin_c ‖w_c − τ̂_ref‖, τ̂_ref = the WM roll-out under the **observed** (zero-order-hold) action. No per-candidate rolling. | ✅ | ❌ |
| **R** | `random` — the deliberately failing input. | – | – |
| **O** | `oracle_in_fan` — upper bound. **Never a deployable number.** | – | – |

**C1 and C2 are not decoration.** They are what makes a positive result attributable: C1 asks whether
a WM is needed at all, C2 asks whether *per-candidate* rolling is needed as opposed to one reference
trajectory. A win that C1 or C2 matches is **not** an imagination result.

### 0.5 The scoring world model — the experiment's biggest confound, run as two arms

v4's own `wm_canary_ade_2s` is **1.1409** against a ≤ 0.55 bar; the v1 line is cited at ~0.452
(INHERITED — established here, §2). **Scoring with a bad simulator scores badly.** Two scorer arms:

- **W-v4** — v4's own world model (`/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt`);
- **W-v1** — **v1's world model scoring v4's fan** (`/root/models/flagship-30k/ckpt.pt`). This is the
  PI's direction (1) literally: v1's world model + the REF-C-derived anchored-diffusion fan.
  **Feasibility is to be ESTABLISHED, not assumed** (§4); if infeasible, the reason is a real finding
  about arm compatibility and is reported as such.

### 0.6 Outcomes — committed in advance

| verdict | condition |
|---|---|
| **CONFIRM** | the best pre-registered imagination rule's **in-sample** `ade_0_2s` **< 0.4907** ⇒ imagination-scoring does something discriminative re-scoring of this fan provably cannot. |
| **STRONG** | additionally its **out-of-fold** `ade_0_2s` **< 0.4271** ⇒ the synthesis is real and v5 has a spine. |
| **REFUTE** | fails to beat **0.4907** ⇒ the limitation is the **fan's own action-sequence information**, not the scorer. Say so plainly; do not re-scope. |
| **NOT ATTRIBUTABLE** | (modifier) any of the above where **C1 or C2 matches the winning arm within its paired CI** — reported as a *scoring-rule* result, not an imagination result. |

**Why the 0.4907 comparison is in-sample on both sides:** 0.4907 is itself Bar A's *in-sample*
ceiling. A0/A1/A2/A3/C1/C2 are **parameter-free**, so their single number is simultaneously in-sample
and out-of-fold. Only A4 has free weights, and it is reported both ways.

### 0.7 Estimator

**Paired episode-cluster bootstrap** (`taniteval/ci.py`, B = 2000, resampling unit = **episode
cluster**, 40 clusters / 881 windows), on identical windows. **`overlapping_holdout_se` is never
used, and the `legacy_*` blocks in any JSON are not quoted.** Every number is decomposed into
**along-track (LONGITUDINAL) / cross-track (LATERAL)**, because v4's 15k→30k regression was 100 %
longitudinal and an undecomposed number hides which axis a lever acts on.
⚠️ **A 40-episode "not separated" is UNPOWERED, not refuted** (half-widths shrink ×2.8–3.9 at
n = 600, `MODEL_REGISTRY.md §1.2a`). Power is stated wherever a null is reported.

### 0.8 Self-tests the harness must pass before it may adjudicate (M3)

Both directions, as Bar A did — a harness that has only seen good input has not been tested.

| # | test | pass condition |
|---|---|---|
| S1 | **cache fidelity** | as-trained `ade_0_2s` = **0.8563**, `oracle_in_fan` = **0.2505**, and the pick agrees with the real forward pass on **100 %** of windows |
| S2 | **committed-number reproduction** | the true-action roll-out reproduces `wm_canary_ade_2s` = **1.1409** on these windows |
| S3 | **inverse-map fidelity** | `traj_to_actions(GT waypoints)` recovers the dataset's own `future_actions` (per-channel MAE + correlation reported) |
| S4 | **derivation error budget** | rolling under `traj_to_actions(GT)` lands close to the true-action roll-out; the gap is the derivation's own error and is **reported, not hidden** |
| S5 | **failing input** | a uniform-random pick scores ≈ **15.36** (Bar A's number), and an anti-oracle (argmax fan error) is worse still |

Any S1/S2 failure **aborts before any arm is scored** and this file records `ABORTED`.

### 0.9 What I commit to reading into a REFUTE, in advance

If A1/A4 cannot beat 0.4907, then combined with Bar A the correct reading is: **the 256 fan
candidates do not differ from one another in a way that either their features OR their dynamical
consequences can rank.** `oracle_in_fan` = 0.2505 would then be a statement about the *marginal*
coverage of a 256-anchor vocabulary, not about a recoverable selection surplus — and the v5 lever
must move to the **fan itself** (proposal generation / candidate count / goal-conditioned anchors),
not to any scorer over it. I will write that, not a scoped-down version of the current claim.

### 0.10 Goal-oracle honesty

`route` / `route_graded` / `vt_band` are minted from the ego's own future poses — **three** channels
(`vt_speed` is overwritten with the observed `v0`). Oracle and produced are reported **as a pair**;
**oracle numbers are never deployed capability.**

### 0.11 🔒 Confidentiality

PhysicalAI-AV is gated-confidential. No clip UUIDs and no raw content appear in any artifact here;
episodes are referenced by their val index only.

---
--- MEASUREMENTS BEGIN ---
---

*(Everything below was written after the corresponding measurement.)*

### 0.12 Corrections received mid-stream, and what they did NOT change

Three corrections arrived from the coordinator **after** §0 was staged and **before** any
E-V5-1 number existed. Recorded here because a pre-registration whose amendments are invisible is
not a pre-registration.

| # | correction | effect on this stream |
|---|---|---|
| 1 | ⛔ **`/root` on the eval pod is 99 % full and SILENTLY TRUNCATES** (a 6 GiB `dd` reported success after 3.0 GiB) | **None — no artifact of this stream was ever written to `/root`.** Everything lives under `/workspace/_v5/`. VERIFIED: the harness `md5` is **identical local ↔ pod** (`fe66b95add866416a470a27fa99276fe`), and `ls /root/_v5* /root/v5*` is empty. |
| 2 | The eval pod is **Python 3.12.3**, not 3.11.10 (that is pod2) | Confirmed independently by this stream's own provenance block: `py=3.12.3`, `torch 2.8.0+cu128`, A40. No PEP-701 issue arose. |
| 3 | ⭐ **Bar A follow-up: the in-sample ceiling moves 0.4907 → 0.4138 when the SCORE IS GIVEN GOAL INFORMATION**, while changing the *objective* moved it 0.0317 / 0.0089 and in the wrong direction — information beat objective by **2.4×–8.6×** | **The pre-registered CONFIRM bar STAYS at 0.4907 exactly as registered.** What changes is its *interpretation*, and that change is stated rather than absorbed — see below. |

⚠️ **The interpretation change, stated precisely.** §0.6 registered **0.4907** as *"the in-sample
ceiling of **any** re-scoring of this fan."* With 0.4138 measured, that gloss is **no longer true**:
0.4907 is the ceiling of **feature-only** re-scoring. Beating it therefore licenses the narrower
claim *"imagination-scoring beats what a discriminative head over the existing features can do"* —
which is still exactly the question E-V5-1 was built to answer. **0.4138 is INHERITED-UNVERIFIED
here** (this stream did not re-derive it) and is quoted only as a secondary reference line, never as
a verdict bar. Moving a pre-registered bar after seeing a result is the failure mode §0 exists to
prevent, and it has not been done.

⇒ **The correction's real force is on E-V5-2**, and it was acted on: the hierarchical experiment was
built as a **CONDITIONING × STRUCTURE factorial** rather than a bare "hierarchical vs flat", so that
*what the selector is conditioned on* is a first-class variable held separate from *how the selection
is organised*.

Two further Bar-A findings, both handled by construction rather than by assertion:

- **fp16 is unsafe on this selector** (256 candidates separated by less than fp16's ULP). This
  harness caches `fan` / `tgt` / `states` / `sel_score` in **fp32 throughout**, and self-test S1
  carries a per-window identity check against the real forward pass — `ref_ade4_identity_max_abs`
  measured **0.000000**, i.e. the cached path is not merely close on the mean, it is **bit-identical
  on every window**.
- **Fine-tuning drove the factorised grafts past the head's own `seam_fail = 1.5` guard.** ⇒ **This
  stream's scorer lives entirely OUTSIDE the graft path.** It consumes only `anchor_traj` (the fan)
  and the world model; **no parameter is trained in A0–A3 / C1 / C2**, and E-V5-2 *reads* the graft
  weights without ever writing them. The guard therefore cannot fire here, and its absence from these
  results is a structural fact, not a lucky one.

Finally, from Bar B (HYPOTHESIS-grade, and taken as a caution rather than a premise): WM error
concentrates on **manoeuvre demand**, not representational novelty (`latent_novelty` was the
second-weakest covariate, 1.364×, not separated). The §E-V5-1 stratification is therefore **not**
framed as an OOD analysis.

### 0.13 ⚠️ A correction I owe the coordinator, from the primary artifact

The `0.4907 → 0.4138` claim was relayed as *"when the score is given goal INFORMATION."* I went to
the primary source rather than quoting it, and **0.4138 is the goal-ORACLE surface**
(`…/2026-07-26-bar-a-selector/raw/bar_a_oracle.json → in_sample_ceiling.ce.ade_0_2s_in_sample`),
against 0.4907 on the produced surface. That matters, in two ways:

1. **It is doubly non-deployable** — in-sample *and* oracle-conditioned. The goal channels are minted
   from the ego's own future poses. It says: *hand the selector the ego's own future-derived goal
   AND let it fit the test windows, and you reach 0.4138.* It does **not** say conditioning is a
   deployable 0.077 m lever.
2. **It is not a pure conditioning effect.** On the oracle surface the **fan** is also better —
   `oracle_in_fan` **0.2330 vs 0.2505** — so ≈ **0.0175 m of the 0.0769 m ceiling move (23 %) is the
   fan improving, not the selector being better conditioned.** The selector-attributable part is
   ≈ 0.0594 m.

The direction of the coordinator's point survives — conditioning outweighs objective — and E-V5-2 was
built around it. But "information beat objective by 2.4×–8.6×" should read **"an ORACLE goal beat
objective by ~1.9×–6.7× once the fan effect is removed."** Both numbers are recomputed here from the
two committed Bar-A JSONs, so this is MEASURED, not a re-reading.

---

## 1. HARNESS VALIDATION — both directions, before anything was adjudicated

**MEASURED · CONFIRMED.** Artifact: `raw/v5_v4.json → selftest`. Host `1e0bac0df88a`, Python
**3.12.3**, torch 2.8.0+cu128, A40. Live tree `/root/v4eval/stack`; every imported module stamped by
path **and md5** in the artifact. v4 ckpt md5 `8771c1d9d3da696dcde2a745d628f6a8`, step **29999** —
the same checkpoint Bar A used.

| # | test | result | |
|---|---|---|---|
| **S1** | cache fidelity vs the committed forward pass | `ade_0_2s` **0.8563** vs 0.8563 (Δ **2e-05**) · `oracle_in_fan` **0.2505** vs 0.2505 (Δ **5e-05**) | ✅ |
| **S1b** | **per-window identity** vs the same run's own forward pass | `ref_ade4_identity_max_abs` = **0.000000** — not "close on the mean", **bit-identical on all 881 windows** | ✅ |
| **S2** | reproduce a **committed** number: `wm_canary_ade_2s` | **1.1381** vs committed **1.1409** (Δ **0.0028**) | ✅ |
| **S3** | inverse-map fidelity vs the dataset's own actions | steer MAE **0.0126 rad**, corr **0.843**; accel MAE **0.513**, corr **0.402** | ✅ |
| **S3b** | inverse-map **exact round-trip** (bicycle forward → inverse), off-GPU | steer max abs err **0.00000**, accel max abs err **0.00017**, corr **1.00000** both channels | ✅ |
| **S4** | derivation error budget | roll under TRUE actions **1.1381** vs under DERIVED actions **1.1197** → gap **−0.0184 m** | ✅ |
| **S5** | failing input | uniform-random pick **15.8738** (Bar A: 15.3622) · anti-oracle **45.5488** · as-trained **0.8563** | ✅ |

Three of these deserve emphasis because they pre-empt the obvious objections to everything below:

- **S2 is the "reproduce a committed number before quoting a new one" rule discharged in the
  hardest form available** — not a re-reduction of a saved tensor but a fresh encode → roll →
  ground → SE(2) pass landing on the published 1.1409.
- **S4 says the candidate→action inversion is NOT the bottleneck.** The derived actions produce a
  *marginally better* roll-out than the dataset's own actions (−0.0184 m). So no REFUTE below can be
  blamed on the inversion — the mechanism I added is, if anything, slightly favourable.
- **S5 confirms the harness can render a failing verdict**, at a value independently consistent with
  Bar A's own random-pick number on the same fan.

⚠️ Also recorded: **`/root` on this host is 99 % full and silently truncates.** Nothing in this
stream was written there. The harness md5 is identical local ↔ pod
(`fe66b95add866416a470a27fa99276fe`), and the v1 checkpoint was integrity-checked by **CRC, not
presence** — `zipfile.testzip()` over 2 708 members returns clean, so it is not truncated.

---

## 2. E-V5-1 — the result. **VERDICT: REFUTE.**

`raw/v5_v4.json`. Produced (deployable) surface, 881 windows / **40 episode clusters**.
Estimator: **paired episode-cluster bootstrap**, B = 2000, unit = episode. Tier: **CONFIRMED, and
DECISION-GRADE for the negative decision** (§6).

| arm | `ade_0_2s` | LONGITUDINAL | LATERAL | paired Δ vs as-trained |
|---|---:|---:|---:|---|
| **A0 as-trained** | **0.8563** | 0.5847 | 0.1889 | — |
| A1 `imag_consistency` ⭐PRIMARY | **11.5298** | 9.2737 | 0.5680 | **+10.6735 [+8.5014, +12.9382]** ✅sep |
| A2 `imag_goal_speed` | 10.3863 | 8.5974 | 0.4511 | +9.5300 [+5.3848, +14.3241] ✅sep |
| A3 `imag_kinematic` | 13.1805 | 10.7143 | 0.5906 | +12.3242 [+9.4493, +15.3494] ✅sep |
| **C1** CONTROL, *no WM* | 1.7836 | 1.3355 | 0.2573 | +0.9273 [+0.4888, +1.6253] ✅sep |
| **C2** CONTROL, *one* imagination | 1.0653 | 0.7452 | 0.2282 | +0.2090 [+0.0487, +0.3599] ✅sep |
| **A4 `imag_combo`, OUT-OF-FOLD** | **0.7706** | **0.4949** | 0.2166 | **−0.0857 [−0.1864, +0.0044]** |
| R random | 15.8738 | 13.0732 | 0.6908 | +15.0176 ✅sep |
| O `oracle_in_fan` *(bound, never deployable)* | 0.2505 | 0.1194 | 0.1366 | −0.6057 ✅sep |

**Against the pre-registered bars:** best imagination `ade_0_2s` = **0.7706**, in-sample and
out-of-fold alike, against **CONFIRM < 0.4907** and **STRONG < 0.4271**. It misses the CONFIRM bar by
**1.57×**. ⇒ **REFUTE**, as registered, and I am not re-scoping it.

### 2.1 The attribution is not merely absent — it is INVERTED, and separated

| comparison | paired Δ | reading |
|---|---|---|
| **A1 − C1** (per-candidate WM roll − *no WM at all*) | **+9.7462 [+7.3047, +12.1443]** ✅sep | the world model makes it **9.7 m worse** than an analytic bicycle model |
| **A1 − C2** (per-candidate roll − *one* WM reference roll) | **+10.4645 [+8.2743, +12.7032]** ✅sep | **rolling each candidate is 10.5 m worse than rolling once** |

§0.4 committed C1/C2 as the controls that make a positive result attributable. They did more than
that: they show the per-candidate imagination is **actively harmful**, separated, on 40 clusters.

### 2.2 THE MECHANISM — and it is the finding, not the REFUTE

`raw/v5_posthoc.json`. Per §"verify before alarming", a rule landing near random is as likely to be
a bug as a result, so I measured **what each rule picks** rather than reasoning about it.

| rule | mean 2 s along-track of its pick | bias vs GT | p05 | p95 |
|---|---:|---:|---:|---:|
| ground truth | 25.40 m | — | 1.96 | 59.88 |
| A0 as-trained | 25.63 | **+0.24** | 1.92 | 59.85 |
| **A1 imagination-consistency** | **45.05** | **+19.66** | 2.27 | 90.94 |
| A2 | 18.08 | −7.31 | −13.69 | 82.94 |
| A3 | 36.85 | +11.45 | −10.10 | 95.92 |
| C1 (no WM) | 27.18 | +1.78 | 3.92 | 68.79 |
| **C2 (one WM roll)** | 25.36 | **−0.03** | 0.31 | 65.23 |
| **A4** | 25.37 | **−0.03** | 1.92 | 61.23 |
| R random | 35.15 | +9.75 | 1.45 | 84.44 |

And the fact that explains the whole table — **the fan's own longitudinal envelope**:

> Within a single window the 256 candidates span, at 2 s, from **−15.47 m to +100.57 m** of
> along-track displacement — a **mean per-window span of 108.7 m**, against a ground-truth mean of
> 25.40 m. A 2 s candidate that travels 100 m is a **181 km/h** plan, and it is in the fan.

⇒ **THE MECHANISM: the world model does not VETO an implausible plan — it obediently SIMULATES it.**
Ask it to execute "travel 100 m in 2 s" and the imagined roll-out also travels ≈ 100 m. That
candidate is therefore **maximally self-consistent**, and imagination-*consistency* ranks it first.
A1's +19.66 m along-track bias is not noise; it is the rule working exactly as specified on a
simulator with no plausibility prior.

**This is why A1 is worse than a coin flip's cousin while C2 is not.** C2 asks a different question —
*"which candidate is nearest to what the world model thinks will actually happen?"* — and that
question has a single answer per window rather than 256 self-fulfilling ones. Its longitudinal
calibration is near-perfect (**−0.03 m** bias).

⇒ **Discharging §0.9's advance commitment, verbatim:** the limitation is **the fan's own
action-sequence information** — specifically its longitudinal over-dispersion — **not the scorer.**
`oracle_in_fan` = 0.2505 is a statement about the *marginal* coverage of a 256-anchor vocabulary that
also contains 181 km/h plans, not about a recoverable selection surplus. **The v5 lever must move to
the fan itself** (proposal generation, longitudinal admissibility, goal-conditioned anchors), not to
any scorer over it.

### 2.3 Stratified by canary quality — the "deployable gate" escape hatch is CLOSED

The brief anticipated: *"if it works where the WM is good, that is a deployable gate."* It does not.

| stratum | n (win / eps) | A0 | **A1** | C2 | A4 | paired A1 − A0 |
|---|---|---:|---:|---:|---:|---|
| `canary ≤ 0.55` (**22.7 %**) | 200 / 29 | 0.7085 | **7.3870** | **0.3330** | 0.5017 | +6.6785 [+3.6939, +10.3307] ✅sep |
| `canary > 0.55` | 681 / 40 | 0.8997 | 12.7465 | 1.2804 | 0.8496 | +11.8468 ✅sep |
| q1 best canary (≤ 0.586) | 221 / 31 | 0.7451 | **7.5799** | 0.3530 | 0.5182 | +6.8348 [+4.0900, +10.1071] ✅sep |
| q2 | 220 / 37 | 0.7244 | 10.6710 | 0.7166 | 0.5746 | +9.9467 ✅sep |
| q3 | 220 / 38 | 0.9309 | 12.0814 | 1.1385 | 0.8497 | +11.1505 ✅sep |
| q4 worst canary | 220 / 29 | 1.0253 | 15.8047 | 2.0565 | 1.1410 | +14.7794 ✅sep |

**WM quality does grade the damage monotonically (7.58 → 10.67 → 12.08 → 15.80), and it never
rescues the rule.** Even in the best quartile A1 is separated-worse than doing nothing by +6.83 m.
The confound the brief flagged — *"scoring with a bad simulator scores badly"* — is therefore
**not** the explanation for this REFUTE. §3 tests it a second way, with a different simulator.

⭐ **But the same table contains a genuinely positive, separated result — with an honesty condition
attached that must travel with it.** On the 22.7 % of windows where the world model is good, **C2
is separated-BETTER than the as-trained selector: −0.3754 [−0.5123, −0.2656]**, i.e. **0.7085 →
0.3330, a 53 % reduction in selector error with ZERO training** (A4: −0.2068 [−0.3625, −0.0811],
also separated).

> ⛔ **AND THE GATE IS NOT DEPLOYABLE, WHICH KILLS THE OBVIOUS PRODUCT CLAIM.** `wm_canary_ade_2s` is
> computed **against ground-truth future poses**. It cannot be evaluated at deploy time. So *"use
> imagination-selection where the WM is good"* is an **ORACLE-GATED** claim until an observable proxy
> exists, and the only deploy-time scalar available here does not supply one: `corr(v0, canary_err)`
> = **0.2645** (canary mean by v0 tercile 1.0138 / 1.0239 / 1.3763). I am recording this as a
> **HYPOTHESIS with a named missing instrument**, not a capability. Finding an observable canary
> proxy is a concrete, cheap, unowned piece of work (§7).

### 2.4 The one thing that DID work — and what makes it interesting

**A4, the weighted combination, is the first selection rule in this program to beat the as-trained
selector out-of-fold, and it needed no training at all** — three weights on a 4-value grid, chosen
5-fold episode-disjoint.

- `ade_0_2s` **0.7706** vs 0.8563 — paired **−0.0857 [−0.1864, +0.0044]**, p(Δ>0) = **0.033**, *not
  separated at 95 %*.
- ⚠️ **UNPOWERED, NOT REFUTED.** `MODEL_REGISTRY.md §1.2a`: half-widths shrink **×2.8–3.9** going
  40 → 600 episodes. This interval is ±0.0954 at n = 40; at n = 600 it would be ≈ ±0.025–0.034 and
  would separate. **This is the single cheapest open question in the stream** (§7).
- ⭐ **On the LONGITUDINAL axis it IS separated now: −0.0898 [−0.1708, −0.0160], p = 0.008.** That is
  the axis v4's 15k→30k regression was **100 %** concentrated on, and the axis **Bar A's regret loss
  could not move at all** (+0.0038 [−0.0236, +0.0292], p = 0.63 — flat). A **training-free scoring
  rule moved the axis a trained ranking objective could not.**
- It is paid for laterally: **+0.0277 [+0.0147, +0.0430]**, separated-worse. Net still favourable.
- **The identical weight vector won in all 5 folds** — `A1 0.5 · C1 2.0 · C2 2.0`, with `A2`, `A3`
  and the as-trained prior all at **0**. Not fold noise.

> ⚠️ **And read those weights honestly: A4 is NOT an imagination result.** Its mass sits on the two
> **controls** — the WM-free bicycle model and the single-reference roll. The per-candidate
> imagination term carries the smallest non-zero weight in the set. **A4 is a
> longitudinal-plausibility result that happens to use a world model as a reference trajectory**,
> which is a different and much cheaper mechanism than imagination-scored selection.

---

## 3. **v1's WORLD MODEL SCORING v4's FAN** — the PI's direction (1), executed

`raw/v5_v1.json`. This is the confound the brief named — *"scoring with a bad simulator scores
badly"* — tested by swapping the simulator and changing nothing else. Same fan, same 881 windows,
same rules.

### 3.1 FEASIBILITY VERDICT: **FEASIBLE**, and here is why — established, not assumed

| property | v1 (`flagship-30k`) | v4 (`flagship-v4-fromscratch-30k`) |
|---|---|---|
| predictor window | **8** | **8** |
| predictor `action_dim` | **3** | **3** |
| step | 29999 | 29999 |
| checkpoint integrity | CRC-verified (`testzip` clean, 2 708 members) | md5 `8771c1d9…` |

**The structural reason it works, and it generalises beyond this pair:** the fan is **metric
trajectories in the ego frame**, not latents. A foreign world model therefore only needs to (a)
consume this dataset's frames and (b) emit metres. **`state_dim` need not match**, because the two
models never exchange a latent — they meet at the metric interface. Both arms load through the same
`_eval_cfg()`, so window/action geometry is shared by construction.

⇒ **Cross-arm imagination-scoring is a general capability of this program's architecture, not a
special case.** Any arm with a grounded step-readout can score any other arm's proposals.

### 3.2 ⭐ Establishing the v1 line myself, and correcting the brief's number

The brief cited the v1 world-model line as **~0.452, INHERITED-UNVERIFIED**, and asked me to
establish it. **MEASURED here: 0.4271** (true-action roll-out → grounding → SE(2), 881 windows /
40 clusters).

> **0.452 is the `heldout` split-mean; 0.4271 is the `full_set` mean.** This is exactly the
> estimator hazard `CLAUDE.md` warns about — `overlapping_holdout_se`'s central value is a
> mean-of-split-means, not the full-set mean. `MODEL_REGISTRY.md §1.2` publishes both
> (`0.4522 ± 0.0312` heldout / **`0.4271`** full-set). **My independent re-derivation lands on the
> full-set value to 4 decimal places**, which is a clean confirmation of both the registry row and
> the rule that produced it.
>
> It also confirms something the registry states but is easy to miss: **v1's canary and v1's headline
> `ade_0_2s` are the SAME QUANTITY** (`wm_fidelity_ade_2s` under `pc2`, `actions_source=expert_future`).

**And the simulator gap is large: v1's WM is 2.67× better than v4's on identical windows —
0.4271 vs 1.1381.**

### 3.3 The result — the simulator matters enormously, and it still REFUTES

| arm | v4's WM | **v1's WM** | improvement | paired Δ vs as-trained (v1's WM) |
|---|---:|---:|---:|---|
| A0 as-trained | 0.8563 | 0.8563 | — | — |
| **A1 `imag_consistency`** | 11.5298 | **1.2472** | **9.2×** | +0.3910 [+0.2385, +0.5539] ✅sep worse |
| A2 `imag_goal_speed` | 10.3863 | 5.0746 | 2.0× | +4.2183 ✅sep worse |
| A3 `imag_kinematic` | 13.1805 | 6.2200 | 2.1× | +5.3638 ✅sep worse |
| C1 (no WM) | 1.7836 | 1.7836 | — *(WM-free by definition)* | +0.9273 ✅sep worse |
| **C2 (one WM roll)** | 1.0653 | **0.5645** | **1.9×** | **−0.2918 [−0.4245, −0.1662]** ✅**sep BETTER** |
| **A4 combo, out-of-fold** | 0.7706 | **0.5645** | 1.4× | **−0.2918 [−0.4245, −0.1662]** ✅**sep BETTER** |
| O `oracle_in_fan` (bound) | 0.2505 | 0.2505 | — | −0.6057 ✅sep |

**VERDICT: still REFUTE.** Best imagination arm **0.5645** vs CONFIRM **< 0.4907** — a 1.15× miss.
Much closer than v4's 1.57×, and still on the wrong side.

### 3.4 Four things this arm establishes that the v4 arm could not

**1. The simulator quality confound is REAL and LARGE — and it does not rescue the hypothesis.**
Swapping v4's WM for v1's improved A1 by **9.2×** with zero other changes. Anyone reading the §2
REFUTE as "imagination is worthless" would be over-reading it; the honest statement is that
**imagination-scoring is strongly simulator-limited, and even an un-limited simulator does not clear
the bar.**

**2. ⭐ The world model EARNS ITS PLACE — the attribution flips.**

| comparison | v4's WM | **v1's WM** |
|---|---|---|
| **A1 − C1** (per-candidate roll − *no WM at all*) | +9.7462 ✅sep **worse** | **−0.5364 [−1.2234, −0.0842]** ✅sep **BETTER** |
| **A1 − C2** (per-candidate roll − *one* WM roll) | +10.4645 ✅sep worse | +0.6827 [+0.5017, +0.8799] ✅sep worse |

With a good simulator, **per-candidate imagination beats the world-model-free control, separated.**
That is a genuine imagination result and it is the first one in this stream. **But it remains
separated-worse than rolling ONCE.** ⇒ **The failure is not the world model and not imagination —
it is the PER-CANDIDATE structure.** Asking "does the simulator reproduce candidate c?" is the wrong
question no matter how good the simulator is, because a simulator with no plausibility prior answers
"yes" to implausible candidates (§2.2).

**3. ⭐ A separated improvement over v4's own selector, with ZERO training.** C2/A4 at **0.5645
[0.4855, 0.6528]** vs as-trained **0.8563**, paired **−0.2918 [−0.4245, −0.1662]**, separated,
p(Δ>0) = 0.000. Bar A's *trained* head moved this the wrong way (+0.0254 / +0.0668). **A
parameter-free rule beat a fine-tuned one by 0.32–0.36 m.**

**4. ⛔ AND THE SYNTHESIS IS NET-NEGATIVE, WHICH IS THE MOST IMPORTANT SENTENCE IN THIS DOCUMENT.**

> **v1's world model alone scores 0.4271. v1's world model projected onto v4's 256-anchor fan scores
> 0.5645.** The fan does not improve v1's answer — it **degrades it by 0.1374 m**, which is the fan's
> **quantisation cost**: the price of being forced to answer with one of 256 anchors instead of the
> trajectory the world model actually produced.
>
> Meanwhile `oracle_in_fan` = **0.2505**, so the fan demonstrably **contains** trajectories far better
> than v1's own. **Both facts are true at once, and together they are the whole finding:** the fan
> holds better answers than v1 can produce, and no realisable rule — discriminative (Bar A) or
> simulative (here) — can find them. What a realisable rule *can* do is recover v1's own answer,
> minus a quantisation tax.

⇒ **"Combine v1's WM with REF-C's planner" is not, as a SELECTION problem, a win.** It is a 0.1374 m
loss against simply deploying v1. That is a clean, decision-grade answer to the PI's direction (1) —
and it says the combination must happen somewhere other than in the selector.

### 3.5 Stratified by v1's canary quality — the pattern inverts, informatively

v1's WM is under the 0.55 bar on **74.5 %** of windows (v4: 22.7 %).

| stratum | n | A0 | A1 | **C2 / A4** | `oracle_in_fan` |
|---|---:|---:|---:|---:|---:|
| `canary ≤ 0.55` (74.5 %) | 656 | 0.8298 | 1.2359 | **0.4235** | 0.2142 |
| `canary > 0.55` | 225 | 0.9335 | 1.2804 | 0.9755 | 0.3566 |
| **q1 best canary (≤ 0.218)** | 221 | 0.8164 | 1.3589 | **0.2794** | 0.1794 |
| q2 | 220 | 0.8595 | 1.0839 | 0.4248 | 0.2269 |
| q3 | 220 | 0.7999 | 1.2685 | 0.5702 | 0.2345 |
| q4 worst canary | 220 | 0.9496 | 1.2771 | 0.9849 | 0.3617 |

Two readings, and the second is the one that matters:

- **C2 tracks simulator quality almost perfectly** (0.2794 → 0.4248 → 0.5702 → 0.9849). Of course it
  does — C2 *is* the simulator's own trajectory, quantised to the fan. Its ceiling is the canary.
- **A1 does NOT track it at all** (1.3589 / 1.0839 / 1.2685 / 1.2771 — essentially flat, and *worst*
  in the best-canary quartile). ⇒ **The per-candidate consistency rule is not limited by simulator
  accuracy in any stratum.** It is limited by its own functional form. This is the cleanest evidence
  in the stream that §2.2's mechanism, not simulator noise, is the cause.

---

## 4. E-V5-2 — HIERARCHICAL vs FLAT, as a CONDITIONING × STRUCTURE factorial

`raw/v5_hier.json`. Same frozen fan, same 881 windows. **Selection fidelity = 1.000000 in all three
goal modes** — `F_flat` reproduces the head's own forward-pass pick on every window, so the structure
arms are being compared against the real thing.

### 4.1 The mechanism check first — is the hierarchy even non-vacuous?

The three class→anchor grafts are **zero-init by construction** (ReZero). If training never moved
them, a hierarchical selector built on them would be a statement about noise. **Measured:**

| graft | ‖W‖_F | max abs | shape | still zero-init? |
|---|---:|---:|---|---|
| `lat_to_anchor` | **0.6457** | 0.0966 | [256, 8] | **No** |
| `lon_to_anchor` | **0.6592** | 0.0998 | [256, 7] | **No** |
| `dist_to_anchor` | **0.7260** | 0.1145 | [256, 8] | **No** |

⇒ **The hierarchy is genuinely learned, so everything below is a real result about it, not about an
untrained parameter.**

### 4.2 AXIS 2 — STRUCTURE. Hierarchical selection is separated-WORSE than flat, everywhere.

Paired Δ vs `F_flat` on the **produced (deployable)** surface. Positive = worse.

| structure | `ade_0_2s` | paired Δ vs flat | distinct picks |
|---|---:|---|---:|
| **F_flat** (as-trained, 1-of-256) | **0.8563** | — | 128 |
| F_base_only (grafts + vt-penalty removed) | 0.8781 | **+0.0218 [+0.0009, +0.0491]** ✅sep | 163 |
| H_graft **q=64** | 1.0621 | **+0.2059 [+0.1290, +0.2975]** ✅sep | 89 |
| H_graft **q=32** | 1.2000 | +0.3437 [+0.2284, +0.4855] ✅sep | 61 |
| H_graft **q=16** | 2.6510 | +1.7948 [+1.2345, +2.4488] ✅sep | 34 |
| H_graft **q=8** | 6.6752 | +5.8190 [+4.4381, +7.1956] ✅sep | 18 |
| H_imag q=64 … q=8 | 13.20 … 21.64 | +12.35 … +20.78 ✅sep | 94 … 22 |

**The same result holds in all three goal modes** (neutral: +0.2314 … +5.5816; oracle: +0.2527 …
+6.0572, all separated), so it is not a conditioning artefact.

> ⭐ **THE FINDING, and it is sharper than "hierarchy does not help":** *the identical learned
> class→anchor prior is **+0.0218 m USEFUL as a soft additive bias** and **0.21–5.82 m HARMFUL as a
> hard commitment**.* Removing the grafts from the flat score makes it separated-worse
> (0.8563 → 0.8781); **committing** to the class they favour and searching only inside makes it
> separated-worse by an order of magnitude more. And the damage is **monotone in how tight the
> commitment is** (q = 64 → 32 → 16 → 8 : +0.21 → +0.34 → +1.79 → +5.82).
>
> ⇒ **The flat score's ability to OVERRIDE the tactical class is doing essential work.** The tactical
> heads know something — enough to be worth +0.0218 m as a prior — but they are **not reliable enough
> to be given veto power** over the operative candidate set. "Hierarchical *biasing*" is what the
> head already does, and it is mildly positive; **"hierarchical *selection*" (commit-then-search) is
> REFUTED on this fan.**

⚠️ **And this answers §5.3's D3 in the direction that closes the efficiency argument.** Restricting to
a committed class of q candidates does cut imagination work by 256/q — but the restriction itself
costs 0.21–5.82 m. **Hierarchy does not buy efficiency here; it buys error.**

### 4.3 AXIS 1 — CONDITIONING. ⭐ **The PRODUCED goal is WORSE THAN NO GOAL, separated.**

This is the arm the brief demanded — *"a hierarchical selector with a working goal input has never
been tested against one without"* — and the answer is the opposite of what the framing assumed.

| goal mode | `oracle_in_fan` (the FAN) | `F_flat` (the SELECTION) | paired Δ vs produced |
|---|---:|---:|---|
| **oracle** *(never deployable)* | 0.2330 | **0.6423** | **−0.2140 [−0.2778, −0.1573]** ✅sep |
| **neutral** *(no goal — the `nav_cmd=None` analogue)* | 0.2387 | **0.7620** | **−0.0943 [−0.1302, −0.0589]** ✅sep |
| **produced** *(goal_head — DEPLOYABLE)* | 0.2505 | **0.8563** | — |

*(`0.6423 / 0.8563 / 0.2330 / 0.2505` reproduce BOOST_PROGRAM §3.2's committed values exactly.)*

**Both the fan and the selection get worse as you go from oracle → neutral → produced**, and the
ordering puts the *deployable* conditioning LAST — behind supplying no goal at all.

**The fan-vs-selection decomposition, which is what keeps this honest:**

| contrast | fan effect (`oracle_in_fan` Δ) | total selected Δ | **selection-attributable** | share |
|---|---:|---:|---:|---:|
| neutral − produced | −0.0118 | −0.0943 | **−0.0825** | **87 %** |
| oracle − produced | −0.0175 | −0.2140 | **−0.1965** | **92 %** |

⇒ **Nearly all of the conditioning effect is SELECTION, not proposal quality** — which is the
coordinator's point, confirmed independently and on the deployable surface. But the sign matters:

> ⛔ **v4's goal_head is NET-HARMFUL. Turning the produced goal OFF is a free, separated
> −0.0943 [−0.1302, −0.0589] m improvement on the deployable surface, available today at zero
> training cost.** The conditioning lever the coordinator identified is real — an *oracle* goal is
> worth −0.2140 m — but **the only conditioning signal we can actually deploy is worse than
> nothing.** The gap between them (0.2140 − (−0.0943) → the goal_head sits **0.1197 m** on the wrong
> side of neutral, and **0.3083 m** behind oracle) is the size of the *goal-production* problem, and
> it is a **producer** problem, not a *consumer* problem.

⇒ **This relocates the "information beats objective" finding.** It is true, and it is **not currently
harvestable through this goal head.** The actionable form is: *fix or disable the goal producer* —
and disabling it is measurable, separated, and free.

⚠️ **Oracle honesty, restated because these rows invite exactly the wrong quote:** `route`,
`route_graded` and `vt_band` are minted from the ego's own future poses (`vt_speed` is overwritten
with observed `v0`). **0.6423 is not capability.** It is the bound that says how much a *perfect*
goal producer would be worth.

---

## 5. E-V5-3 — the efficiency half. **The imagination budget has NEGATIVE marginal value.**

`raw/v5_cost_curve.json`. Computed **entirely off-GPU** from the persisted roll-outs: the k-step
imagination is a strict causal **prefix** of the 20-step one (`accumulate_se2` is causal), so no
re-rolling was needed.

### 5.1 §7.3 discharged FIRST — what would have been disappointing, committed before the numbers

| # | disappointing outcome | why it would be information-free |
|---|---|---|
| **D1** | `ade(n, k)` spans **less** than the paired CI half-width of A1-vs-A0 across the whole grid | then rolling 20 steps is indistinguishable from rolling 2, and no efficiency axis exists to design on — H2's "85.7 % vs 85.6 %" failure exactly |
| **D2** | top-2 candidates as good as all 256 | the fan's other 254 are decoration and "efficiency" is trivially maximal |
| **D3** | tactical-commit top-q == base-score top-q | candidate **count** is the lever, not hierarchy (answered in §4) |

**D1 does NOT fire — the axis has real dynamic range.** Span over the grid **12.5553 m** vs paired
CI half-width **2.2184 m**, a **5.7× margin.** So the numbers below carry evidence. They simply point
the opposite way to the hypothesis.

### 5.2 The quality surface — `ade_0_2s` by breadth (n candidates) × depth (k steps)

`n = 1` is the as-trained pick by construction, so every column starts at 0.8563.

> ⛔ **LABEL CORRECTED 2026-07-27 — the breadth axis is NOT "the top-n by the as-trained
> base ranking".** The dumped key `base_rank` in `raw/v5_v4_windows_reduced.pt` is
> **`[the as-trained pick] ++ [anchors 0..255 in INDEX order]`**, verified on **881/881 rows**
> (E-H1 §9.2; re-verified independently here — `raw/eh2_gate.json::G4`, which also finds
> **881/881** rows whose tail is *strictly increasing anchor index*, i.e. carrying no score
> information at all). So the `n` axis varies *"the deployed pick plus the first n−1 anchor
> indices"*, not *"the n best-scoring candidates"*. The `n = 1` column (0.8563) is exact.
>
> **§5.2's and §5.4's CONCLUSIONS SURVIVE** and are not weakened: the finding is that *letting
> the imagination rule consider more candidates makes it worse*, which holds for **any nested
> family** of candidate sets — and index order is still a nested family. *"Breadth costs
> −10.66 m"* and *"at every budget, spend none of the imagination budget"* stand as written.
> Root-cause class: **a tensor's semantics taken from its NAME rather than from its
> construction site** (`code/v5_cost_curve.py`, now corrected in place, with a `nested_order`
> key and a `_base_rank_IS` note added to the dump).

| k \ n | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.856 | 1.499 | 1.970 | 2.629 | 2.731 | 2.799 | 2.926 | 2.890 | 2.906 |
| 2 | 0.856 | 1.346 | 1.846 | 2.190 | 1.888 | 1.639 | 1.590 | 1.477 | **1.423** |
| 4 | 0.856 | 1.190 | 3.417 | 3.557 | 2.787 | 2.297 | 1.778 | 1.440 | 1.325 |
| 8 | 0.856 | 0.976 | 5.471 | 6.206 | 6.059 | 5.150 | 4.051 | 3.242 | 2.818 |
| 10 | 0.856 | 0.950 | 5.517 | 6.615 | 7.005 | 6.724 | 5.552 | 4.674 | 3.820 |
| 14 | 0.856 | 0.909 | 5.996 | 8.151 | 8.897 | 9.542 | 8.700 | 7.690 | 6.726 |
| **20** | 0.856 | **0.870** | 6.947 | 11.151 | 12.286 | 13.412 | 12.742 | 12.541 | **11.530** |

**D2 fires in reverse, and hard.** `ade(n=2, k=20)` = **0.8695**, `ade(n=256, k=20)` = **11.5298** ⇒
breadth *costs* **−10.66 m**. **Every additional candidate you let the imagination rule consider
makes the answer worse**, because every additional candidate is another chance to draw one of the
fan's 181 km/h plans that the world model will faithfully and self-consistently reproduce (§2.2).

### 5.3 The cost axis, in measured units

| quantity | measured | vs a full camera pass |
|---|---:|---:|
| `encode_window` (the full camera pass), batch 1 | **27.955 ms** | 1.00× |
| one predictor imagination step, batch 256 | **25.183 ms** | **0.901×** |
| one predictor imagination step, batch 1024 / 2048 | 95.567 / 188.085 ms | 3.42× / 6.73× |
| **scoring the whole 256-fan, k = 20 deep** | **503.7 ms** | **18.0×** |

A 256-candidate fan fits in **one** batched predictor call per imagination step, so scoring the whole
fan k steps deep costs **k** predictor steps, not 256 k. Even so: **you pay 18 full camera passes to
make the answer 13× worse.**

### 5.4 The informative axis — quality at a fixed imagination budget

Budget = `n_candidates × k` = candidate-steps per decision; spend it on breadth or depth.

| budget (candidate-steps) | 64 | 128 | 256 | 512 | 1024 | 2048 | 5120 |
|---|---|---|---|---|---|---|---|
| best `ade_0_2s` | 0.8563 | 0.8563 | 0.8563 | 0.8563 | 0.8563 | 0.8563 | 0.8563 |
| optimal (n, k) | (1, 1) | (1, 1) | (1, 1) | (1, 1) | (1, 1) | (1, 1) | (1, 1) |

> ⇒ **At EVERY budget from 64 to 5 120 candidate-steps, the optimal way to spend the imagination
> budget is to spend NONE of it.** The optimum never moves off `(n = 1, k = 1)`, which is the
> as-trained pick with the imagination term inert. There is no trade-off to design here — the
> marginal value of imagination-consistency compute is **negative over the entire measured range**.

### 5.5 Where hierarchy *would* buy efficiency, and the honest version of that claim

The brief asked where hierarchy buys efficiency, e.g. "rolling only the tactical class's candidates
deeply". Restricting to a committed class of q candidates cuts imagination work by **256/q** — but
§5.2 shows the quality curve **improves** as n shrinks under this rule, so the "saving" is
confounded with the rule being bad at breadth. **A compute saving that is achieved by doing less of
a harmful thing is not an efficiency result**, and I decline to quote it as one. §4 supplies the
matched-count comparison that would separate the two.

⭐ **The efficiency finding that IS real, and it is the useful one:** the arm that worked (A4, §2.4)
does **not** need per-candidate imagination at all. Its weight mass is on C1 (analytic bicycle model,
**zero** predictor steps) and C2 (**one** reference roll per decision). Measured in the units that do
not depend on batching: **A1 needs 256 rolled candidates per decision, A4/C2 need 1 — a 256×
reduction in imagination work for the arm that is 15× better.**

### 5.6 The same surface under v1's world model — flatter, same conclusion

`raw/v5_cost_curve_v1.json`. A better simulator makes the degradation far gentler — grid span
**1.1587 m** vs v4's **12.5553 m**, a **10.8× reduction in how much damage breadth-and-depth can
do** — and it does not change the decision.

| k \ n | 1 | 2 | 8 | 32 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.856 | 0.937 | 1.593 | 1.836 | 1.866 | 1.921 |
| 2 | 0.856 | 0.995 | 1.548 | 1.154 | 1.135 | **1.101** |
| 8 | 0.856 | 1.032 | 1.864 | 1.438 | 1.269 | 1.174 |
| 20 | 0.856 | 0.993 | 1.357 | 1.266 | 1.242 | 1.247 |

D1 again does not fire (span 1.1587 > half-width), so the axis carries evidence. **The budget
optimum is `(n = 1, k = 1)` at every budget from 64 to 5 120 candidate-steps here too.** Under both
simulators, the measured answer to *"how many candidates, rolled how far, buys how much?"* is
**none, zero, and nothing** — for the *consistency* functional. The compute is better spent on the
single reference roll C2 uses.

---

## 6. VERDICT, TIER, AND WHAT THIS DOES AND DOES NOT LICENSE

### 6.1 The headline

> # E-V5-1: **REFUTE.**
> **Scoring the frozen v4 fan by imagining each candidate's consequences does NOT beat 0.4907.**
> Best imagination arm **0.7706** with v4's world model, **0.5645** with v1's — a **1.57×** and
> **1.15×** miss. Per-candidate imagination is **separated-WORSE than the as-trained selector** under
> both simulators (+0.3910 to +10.6735), and separated-worse than **rolling once** under both.

**Tier: CONFIRMED, and DECISION-GRADE for the negative decision.** Exact, and here is where it stops:

- **DECISION-GRADE** for *"do not build v5 around imagination-scored selection over this fan"* — the
  effect is separated on 40 episode clusters, replicated under two independent world models, holds in
  every canary stratum, has a measured mechanism (§2.2), and the harness reproduced a committed
  number (S2) and detected a deliberately bad input (S5) before adjudicating.
- **CONFIRMED but NOT decision-grade** for the positive results, which are single-run: A4's
  −0.0857 m is **not separated** at n = 40, and C2's −0.2918 m, though separated, has been measured
  once. Both need the 600-episode deployment before they steer GPU-days.

### 6.2 What is settled

1. **The discriminative/simulative distinction is not where the problem lives.** Bar A refuted the
   discriminative scorer; this refutes the simulative one. Together they bracket the scorer family.
2. **The fan's longitudinal envelope is the binding constraint** — a **108.7 m** per-window span with
   candidates up to **181 km/h**, and a world model that **simulates implausible plans rather than
   vetoing them** (§2.2).
3. **Cross-arm imagination-scoring is architecturally FEASIBLE and general** — arms meet at the metric
   trajectory interface, `state_dim` need not match (§3.1).
4. **The PI's direction (1), as a selection problem, is net-negative**: v1's WM alone **0.4271**;
   v1's WM projected onto v4's fan **0.5645**; the fan's quantisation tax is **0.1374 m** (§3.4).
5. **Hierarchical *selection* is refuted; hierarchical *biasing* is what already works** — the same
   learned prior is +0.0218 m useful as a soft bias and 0.21–5.82 m harmful as a hard commitment (§4.2).
6. **v4's produced goal is worse than no goal**, separated, −0.0943 m available for free (§4.3).
7. **The imagination compute budget has negative marginal value over the entire measured range**,
   under both simulators (§5.4, §5.6).

### 6.3 What is NOT settled, and what I refuse to conclude

- **NOT** "world models are useless for planning." With a good simulator, per-candidate imagination
  **beat** the WM-free control, separated (A1 − C1 = −0.5364, §3.4). The world model contributes; the
  **per-candidate consistency functional** is what fails.
- **NOT** "a better scoring rule cannot exist." I tested six rules and one 4^6 weighted grid. I did
  **not** test a *learned* cost over imagined outcomes, or imagination under a **plausibility-
  constrained** action set — and §2.2 says the second of those is the obvious next probe.
- **NOT** "the fan is bad." `oracle_in_fan` = **0.2505** stands, and is 41.3 % better than v1. The
  fan contains the answers; nothing realisable finds them.
- **NOT** a deployable "use it where the WM is good" gate. That stratum result is **oracle-gated**
  (§2.3) — `wm_canary_ade_2s` needs ground-truth future poses and `corr(v0, canary)` is only 0.2645.

### 6.4 Threats to validity I could not remove

| threat | status |
|---|---|
| the candidate to action inversion could be the cause | **Excluded, measured.** Exact round-trip (S3b) and a derivation budget of **−0.0184 m / +0.0025 m** (S4) — the derived actions are *not worse* than the dataset's own. |
| a bad simulator could be the cause | **Excluded by replication** — two world models, 2.67x apart in canary, same verdict (§3.3). |
| my rollout convention (overwriting the last window action) | **Documented, not validated against an alternative.** A convention that left transition 1 uncontrolled was not run. Low risk (it affects 1 of 20 steps) but it is an untested degree of freedom. |
| A4's weight grid (4 values, 6 terms) is coarse | Real. A finer grid could improve A4; it cannot rescue A1–A3, which are parameter-free. |
| n = 40 episodes | **Named everywhere it matters.** A4's null is **UNPOWERED, not refuted.** |

---

## 7. WHAT THIS UNBLOCKS (§7.4, required field) — and three escalations

### 7.1 Streams this unblocks

| stream | what it gets |
|---|---|
| **S-2 (the selector) — CLOSED, with a redirect** | Bar A + E-V5-1 jointly close the scorer family. **The selector is no longer the highest-value engineering task**; the fan's longitudinal admissibility is. Recommend S-2 be re-scoped from *"convert the 41 % proposal advantage via selection"* to *"constrain the proposal distribution."* |
| **v4 restart decision** | Strengthens BOOST_PROGRAM §3.4's *do not spend 59 GPU-hours*. Bar A's arithmetic said the selector cannot pay for a restart; this adds that **no scorer of any family can**, and supplies a **cheaper, better-motivated** intervention (§7.2). |
| **Bar B (`wm_canary`) — now has a named consumer** | §3.5 measures the exact dependence of every rule on simulator quality. C2's ADE tracks the canary 1:1 (0.2794 → 0.9849 across quartiles). **Any Bar-B improvement now converts directly into selection quality through C2**, which was previously an unowned bar with no lever *and* no consumer. |
| **v2corpus arm (pod1)** | Inherits a **training-free, separated −0.2918 m post-processor** (C2) that needs only a grounded step-readout — which that arm has. Applicable the moment it finishes, no retraining. |
| **The 600-episode re-adjudication (BOOST §7.1 harvest)** | Supplies **two ranked entries**: A4's −0.0857 [−0.1864, +0.0044] (|effect| / half-width = 0.90) and the goal-off −0.0943 (already separated at 40; would become decision-grade at 600). |
| **Blind-imagination sweep (pod2)** | Shares this stream's mechanism: a world model with no plausibility prior reproduces whatever it is asked to. §2.2 is a directly reusable diagnostic. |

### 7.2 The cheapest discriminating experiment this points to, pre-registered in outline

§2.2 says the fan spans 108.7 m of longitudinal displacement per window. **Clip the fan to a
kinematically admissible longitudinal band around `v0` before any scoring** — pure post-processing,
no training, minutes of CPU, fully recomputable from `raw/v5_v4_windows_reduced.pt` with **no GPU**.

- **CONFIRM** if flat selection over the clipped fan beats **0.8563** by more than A4's −0.0857;
- **STRONG** if it beats **0.4907** — which would mean the whole Bar-A / E-V5-1 REFUTE was an artefact
  of an unconstrained proposal distribution, not of scoring;
- **REFUTE** if `oracle_in_fan` degrades faster than the selection improves — i.e. the extreme
  candidates are load-bearing for the oracle bound.

⚠️ Both outcomes committed here in advance, before anyone runs it. **I did not run it** — it is
outside this brief's scope and the pre-registration must be visible before the measurement.

### 7.3 THREE ESCALATIONS — these must not sit in a file

1. **ACTION AVAILABLE TODAY: turn v4's produced goal OFF.** −0.0943 [−0.1302, −0.0589] m,
   separated, deployable surface, zero training (§4.3). **This is a decision, not a finding**, and it
   needs an owner. It also means every v4 produced-surface number in the program was measured with a
   net-harmful conditioning channel engaged.
2. **`MODEL_REGISTRY.md` should record that v1's WM line is 0.4271, not ~0.452.** I re-derived it
   independently (§3.2); **0.452 is the deprecated `heldout` split-mean**. The brief I was given
   carried `~0.452` as an unverified premise — exactly the C-II failure BOOST_PROGRAM M2 exists to
   stop. The registry row is already correct; the *briefs* are not inheriting from it.
3. **The 0.4138 relay needs correcting at source** (§0.13): it is the **goal-ORACLE** surface, and
   **23 % of the 0.4907 → 0.4138 move is the fan improving, not the selector being better
   conditioned.** Left uncorrected it will be quoted as a deployable conditioning lever, which §4.3
   shows it is not.

### 7.4 For `RETRACTION_LOG.md` — root-cause classes

- **C-new: "the instrument is faithful" is a FAILURE MODE, not a virtue.** The imagination-consistency
  rule failed *because* the simulator faithfully reproduced implausible action sequences. Any
  self-consistency objective over a generative model with no plausibility prior is
  **self-fulfilling**. Generalises well beyond this stream.
- **C6 (confounded comparison), avoided by construction.** C1 / C2 were registered *before* measurement;
  without them A4's −0.0857 would have been written up as an imagination result, when its weight mass
  is on the two controls (§2.4).
- **C-II (unverified premise in a brief), caught twice in one stream** — `~0.452` (§3.2) and the
  0.4138 relay (§0.13). Both were resolved by going to the primary artifact, and both had been
  travelling in briefs.

---

## 8. DELIVERABLE MANIFEST

**STAGED, NEVER COMMITTED, NEVER PUSHED.** All repo paths are under
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-v5-imagination-selection/`.
Every pulled artifact was **md5-verified pod to repo** (`/root` on the eval pod is 99 % full and
silently truncates, so presence was never treated as completeness).

| artifact | repo | pod | note |
|---|---|---|---|
| `V5_IMAGINATION_SELECTION.md` (this file) | staged | — | pre-registration staged **before** any measurement |
| `code/v5_imagination_select.py` | staged | `/workspace/_v5/` | E-V5-1 harness; md5 verified identical |
| `code/v5_hierarchical_select.py` | staged | `/workspace/_v5/` | E-V5-2 factorial |
| `code/v5_cost_curve.py` | staged | `/workspace/_v5/` | E-V5-3, off-GPU |
| `code/v5_posthoc.py` | staged | `/workspace/_v5/` | mechanism + axis/stratum CIs, off-GPU |
| `raw/v5_v4.json` | staged | yes | E-V5-1, v4's WM · md5 `65bd250a…` |
| `raw/v5_v1.json` | staged | yes | E-V5-1, **v1's WM** · md5 `314b0747…` |
| `raw/v5_hier.json` | staged | yes | E-V5-2 · md5 `85d1e384…` |
| `raw/v5_posthoc.json` | staged | yes | md5 `8368c029…` |
| `raw/v5_cost_curve.json` · `raw/v5_cost_curve_v1.json` | staged | yes | md5 `1a9009a8…` / `266de058…` |
| `raw/v5_v4_windows_reduced.pt` · `raw/v5_v1_windows_reduced.pt` | staged | yes | 14.6 MB each; md5 `e4ebeeb4…` / `37812057…` |
| `raw/v5_hier_windows.pt` | staged | yes | per-window ADE, all 3 goal modes x all structures |
| **full per-window tensors** (`fan` / `imag` / `ctrv`, 114 MB each) | **no** | ⚠️ **`tanitad-eval:/workspace/_v5/v5_{v4,v1}_windows.pt` ONLY** | **Deliberately not staged** (228 MB). **Regenerable** in ~11 GPU-min each by re-running the harness; and **every bar in this document is recomputable from the reduced dumps with NO GPU.** Flagged per operating-standard rule 2. |

**Recomputing any bar with no GPU:** `fan_err4` `[881, 256]` is the 4-waypoint error of *every*
candidate, so any selection rule's `ade_0_2s` is `fan_err4.gather(1, pick).mean()`. `cost_A1_by_k`
supplies the whole E-V5-3 depth axis. `costs` supplies every pre-registered rule.

✅ **This claim was VERIFIED, not asserted.** Re-deriving from `raw/v5_v4_windows_reduced.pt` alone —
on the dev box, no GPU, no pod access — reproduces **all nine arms to 4 decimal places** (0.8563 /
15.8738 / 0.2505 / 11.5298 / 10.3863 / 13.1805 / 1.7836 / 1.0653 / 0.7706) and the E-V5-3 depth axis
at n = 256 (k1 2.906 · k2 1.423 · k8 2.818 · k20 11.530). A reader with this repo and no compute can
check every number in this document.

**Nothing was launched, restarted, committed or pushed. No steering file was edited. No training
run was touched. `pod1` / `pod2` / `pod3` were never contacted.**
