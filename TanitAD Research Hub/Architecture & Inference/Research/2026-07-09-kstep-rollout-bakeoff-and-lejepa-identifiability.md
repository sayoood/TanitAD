# Architecture & Inference — 2026-07-09 — K-step rollout bake-off (first measured arm) + LeJEPA identifiability theory

> **Run:** Wednesday weekly agent. Calendar note: dated by **wall-clock (2026-07-09)** per the
> Data-Eng / my-own-STATE precedent; the hub's other notes are forward-dated to mid/late-July by the
> autonomous loop. Budget used: **4 web searches + 2 fetches, 1 measured experiment (2 trained arms),
> ~1 loop iteration, ≈2.2 h** — under the 4-iter / 4-h / 25-search caps.
> **QUALITY: full** (G-A…G-H, G-AI1, G-AI2 met).

Consumed this week's prior outputs: **Monday (Tools&DevEnv)** — Colab T4 burst compute is now **LIVE**
(33 s cold-to-done, $0, OAuth done by Sayed) and the MetaDrive front-cam arm; **Tuesday (Data Eng)** —
DATASET_LANDSCAPE acquisition queue + PhysicalAI R1 (2000 clips) promoted. Consumed the mid-week stack
change: **the K-step rollout mechanism landed** (`train.rollout_k`, commit c4375f8) — flipping my
backlog's top experiment from *planned* to *runnable*.

---

## 1. Headline

The `kstep_rollout` bake-off lever went **planned → runnable** on 2026-07-09
(`train_worldmodel._rollout_loss` + `future_actions` in the window contract). I executed the **first
measured arm of that lever**: two arms trained at **matched compute** on real comma2k19 camera data,
identical in every config field **except `train.rollout_k`** (verified by the harness' own `lever_diff`),
each scored through the integrated D1–D3 gate runner. Three measured takeaways:

1. **K-step rollout is nearly free.** K=2 costs **+0.5 % wall-clock** (749.4 s vs 745.4 s) and **0 extra
   params** — it reuses the already-encoded latents and adds only K predictor passes. Matched compute is
   real, not nominal.
2. **The backlog's chosen falsifier metric (D2 P1 direction-acc ≥ +0.02) SATURATED at 1.0 on both arms**
   (P1 probe fit R²≈0.9999) → it is **ceiling-limited / indeterminate** at this scale, *not* a true fail.
   The discriminative signal is **`imag_rel`** (latent-prediction error vs persistence).
3. **On `imag_rel`, K=2 shows a large, directional win at the 1-step horizon (2.914 → 1.049, −64 %) but
   NO win at the 4-step horizon (I4 1.451 → 1.645, worse).** This is exactly what a K=2 recursive loss
   should do — it optimizes 2-step consistency, not 4-step. **Design consequence: rollout K must cover the
   decode horizon you care about** (K≈4+ for the D3 2-s claim, matching the 2512.24497 Pareto).

No architecture change is executed (D-004/D-018): **D1 FAILs and D3 is BLOCKED on both arms**, so no gate
licenses a claim at this reduced scale. Decision-grade K-sweep = operative-scale matched arms from the
pod2 step-8k checkpoint (Phase C).

---

## 2. The experiment (G-H measured, backlog P0 #2)

**Design.** OFAT: arm A `rollout_k=1` (single-step baseline) vs arm B `rollout_k=2` (recursive 2-step
rollout loss). `lever_diff(cfgA, cfgB) == ["train.rollout_k"]` asserted in-script before running (reuses
`tanitad.eval.bakeoff.lever_diff` — a hidden confound raises). Same seed (0), same route-split train/val
episodes, same 2000-step budget, same reduced-but-real config.

**Config — reduced-but-REAL probe (honest scope, P8).** `d_model=256 / encoder-depth 6 / predictor-depth
4 / window 8 / horizons (1,2,4) / 128 px / 9-ch real camera / no tactical / no H15`. **11,743,554 params.**
This shrinks the operative stack so two arms fit the Wednesday wall-clock on the local RTX 4060; it is a
**directional probe, not the decision-grade sweep** (which needs operative-scale arms from the trained
pod2 checkpoint). Both arms share the identical reduced config, so the *lever contrast* is valid even
though the *absolute* numbers are not decision-grade.

**Hardware / cost.** RTX 4060, 2 × 745–749 s train + eval, **$0** (local). CNCE input recorded.

### Results

| metric | K=1 | K=2 | Δ (K2−K1) | read |
|---|---|---|---|---|
| train wall-clock | 745.4 s | 749.4 s | +4.0 s (**+0.5 %**) | rollout ≈ free |
| params (measured) | 11,743,554 | 11,743,554 | 0 | OFAT / G-AI2 ✓ |
| final single-window pred loss | 0.239 | 0.199 | −0.040 | rollout also helped 1-window fit |
| final rollout loss | 0.0 (off) | 0.0675 | — | K=2 term active, finite, stable |
| erank (collapse-health) | 39.5 | 40.1 | +0.6 | **no collapse** either arm |
| **D2 imag_rel @1-step (diag.)** | **2.914** | **1.049** | **−1.865 (−64 %)** | **the lever's real effect** |
| D3 I4 @4-step | 1.451 | 1.645 | +0.194 | K=2 does **not** help beyond its K |
| D2 P1 direction-acc | 1.000 | 1.000 | 0.000 | **saturated** — non-discriminative |
| D2 P1 probe fit R² | 0.9999 | 1.0000 | — | probe over-fits at n_val=8 |
| D1 status (ADE@1s vs <1.0 m) | **FAIL** | **FAIL** | — | under-trained at this scale |
| D2 status | PASS | PASS | — | via saturated P1/P4 (lever-blind) |
| D3 status | **BLOCKED** (I4>1) | **BLOCKED** (I4>1) | — | **no claim** (D-004) |

Artifacts: `../Implementation/kstep_bakeoff_probe/{kstep_bakeoff_probe.py, results/2026-07-09-kstep_bakeoff_result.json, results/k{1,2}_{config,metrics}.json}`.

### Why `imag_rel` moves so much at 1-step but not 4-step (metric mechanics, verified)

`imag_rel = ‖ẑ−z_true‖ / ‖z_true−z_prev‖` (beat persistence ⇒ <1). The **1-step** denominator is small
(little changes in 100 ms ⇒ persistence is a *hard* baseline ⇒ imag_rel large), the **4-step** denominator
is larger (⇒ imag_rel smaller). K=2 rollout trains recursive 2-step consistency, which sharply improves the
1-step fed-back prediction (2.91→1.05, from far-worse-than-persistence to **persistence-parity**) but leaves
the un-trained 4-step horizon flat-to-worse (I4 1.45→1.65). Internally consistent; it is *the* argument for
matching K to the target horizon.

### Honest verdict (supersedes the script's mechanical string)

The script prints "K=2 FAILS falsifier (P1 dir-acc +0.000 < +0.02)". That label is an **artifact of a
saturated metric** and is **superseded** here: direction-of-displacement is too coarse a readout once the
calibrated probe is near-perfect. **The measured, discriminative result is the −64 % `imag_rel` drop at
1-step — a directional POSITIVE for K-step rollout (H5) — with the caveat that it needs K to cover the
decode horizon and needs operative-scale + ≥3 seeds (Thursday's rule) to become decision-grade.**

---

## 3. Theory watch (D-013) — two findings, each translated to a lever

### 3a. "When Does LeJEPA Learn a World Model?" (arXiv 2605.26379, LeCun/Klindt lineage) — **strengthens H3**

LeJEPA (alignment + Gaussian regularization = our SIGReg objective) **linearly and *orthogonally*
identifies** the world's latent variables from nonlinear observations in worlds where latents evolve under
**stationary, additive-noise transitions**; among all such worlds the **Gaussian is the *unique* latent
prior** for which the guarantee holds; and **"linear, orthogonal identifiability enables optimal
latent-space planning."** Approximate identifiability degrades gracefully; non-Gaussian latents break it.

**Why this matters for TanitAD (three concrete translations):**
1. **It theoretically grounds the `p0-spectral-sizing` tool.** That tool fits `(z_t,a_t)→z_{t+1}`
   **linearly** and got fit R²=0.99–0.999 on trained latents (and again here: 0.9989 / 0.9999). Under
   LeJEPA, the SIGReg latent *is* linearly identifiable, so the linear transition proxy is not a
   convenience — it is the *correct* estimator, and its effective-rank spectrum is a principled sizing
   signal. → **D-021** ("2048 readout is OVER-PROVISIONED, size to the knee") gains theory backing.
2. **It upgrades the rationale for SIGReg-only anti-collapse (H3).** SIGReg's Epps–Pulley term targets an
   **isotropic Gaussian** embedding — *exactly* the unique-prior condition the identifiability theorem
   requires. Our anti-collapse choice is not just "empirically stable" (LeWM) but sits on the prior for
   which optimal-planning identifiability is *proven*. Evidence row added to H3 (no status change, P8 —
   external theory, not our measurement).
3. **Named experiment (added to backlog):** add an **orthogonality instrument** to `spectral.py` — check
   the trained readout covariance is ~isotropic/diagonal (the theorem's "orthogonal" condition). If the
   latent is *not* orthogonal, the optimal-planning guarantee is only approximate and the imagine-and-
   select scoring (H1/H15) inherits that gap. Cheap, high-leverage, and it makes an otherwise-abstract
   theorem falsifiable on our own checkpoint.

### 3b. ACWM action-conditioning ablation (arXiv 2605.08567) — **refines the `adaln_conditioning` planned lever**

Measured ablation: **cross-attention conditioning beats AdaLN for *high-dimensional* action spaces but
offers *no benefit* for *low-dimensional* actions**; AdaLN (modulate by summed timestep + compressed action
embedding) is the standard low-cost injection. **Our action space is 2-D (steer, accel) — squarely the
low-dim regime.** Consequence for the `adaln_conditioning` planned lever (AdaLN vs our FiLM): the prior
"AdaLN > FiLM" triangulation (2512.24497 / Delta-JEPA / OmniDreams) still points to AdaLN, but the
**expected effect size is bounded** — with 2-D actions there is little headroom for a fancier conditioning
map, and **no reason to reach for cross-attention**. Keep AdaLN as the target mechanism, but **budget the
lever as a small-Δ smoke test first** (backlog P1 #3 already says "promote to a Colab arm only if smoke
shows ≥ +2 % probe fit") — this ablation lowers my prior that it clears that bar. Recorded, no change.

### 3c. Adjacent, logged as Phase-1 watch (no action)

GraphWorld (2606.16274, long-horizon planning graph — near our strategic VQ graph), Latent-CoT E2E driving
(2512.10226), Co-Evolving Latent Action WM (2510.26433). Citation-walk anchors updated:
`github.com/klindtlab/lejepa-identifiability` added to the LeJEPA-theory anchor set.

---

## 4. Actionable recommendations (G-B, G-AI1)

Each names the falsifying gate + the bake-off that isolates it (instrument doctrine, no gate → no change):

- **R1 — carry K-step rollout to the decision-grade sweep, swept over K∈{1,2,4}, at operative scale.**
  Falsifying gate: **D2/D3**. Bake-off: `kstep_rollout` lever (already in `default_levers()`), but run on
  matched-compute *trained* operative arms (pod2 Phase C, from the step-8k ckpt). Nearly-free compute
  (+0.5 %) makes this low-risk. **D-018 Tactic → escalate before it touches the trained config.**
- **R2 — change the K-step falsifier metric from D2 direction-acc to `imag_rel` (per-horizon).** Direction-
  acc saturates once the calibrated probe fits; `imag_rel` is discriminative and cheap. Update the backlog
  falsifier accordingly (done, §5 below). Falsifying gate: D3 (multi-step decode).
- **R3 — match rollout K to the decode horizon.** The 1-step-helps / 4-step-doesn't-help split predicts K
  must cover the target horizon. For the D3 2-s claim (which currently only imagines k∈{1,2,4} = 0.4 s),
  this couples with the separate need to **extend imagination horizons** (open item in
  `evaluate_checkpoint.py`'s honest-horizon note). Falsifying gate: D3.
- **R4 — add an orthogonality instrument to `spectral.py`** (3a.3). Falsifying gate: none directly — it is
  an *instrument* (like I1–I4), so it gates admissibility of the D-021 sizing claim, not an architecture
  change. Ships as an intake with a test.

---

## 5. Backlog upkeep + ledger

- **P0 #2 (K-step rollout bake-off) → DONE this run** (measured, above). Superseded by two follow-ups:
  decision-grade K∈{1,2,4} sweep at operative scale (new P0), and the metric swap to `imag_rel` (applied to
  the K-step falsifier). Orthogonality instrument added (P1). Full re-prioritization in `BACKLOG.md`.
- Ledger: **H3** evidence row (LeJEPA identifiability, theory support, no status change); **H5** evidence
  row (K-step rollout first measured arm — 1-step imag_rel −64 %, horizon-matched, no gate passed).

## 6. Gate/quality self-check

- G-A source/repo refs ✓ · G-B R1–R4 tied to gates/WPs ✓ · G-C KB deltas (newest first) ✓ · G-D ledger
  H3/H5 rows ✓ · G-E the experiment script runs + produced JSON (and the stack suite is 188✓/1s green —
  unchanged BY this run, which adds no stack code; count rose from 181 as other agents' intakes landed
  mid-session) ✓ · G-H measured arms with numbers + falsifier verdict ✓ ·
  G-AI1 every rec names its gate + isolating bake-off ✓ · G-AI2 params/wall-clock measured, never mixed
  with estimates ✓.
