# Our two measures, measured — SigReg SEES collapse at 42 σ while the monitor watching it is nearly blind; O3 and O6 were both BELOW the noise floor until yesterday

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **Base HEAD** `a84a1a0`
**Tier** N/A — these are **loss-mechanics and representation-geometry** results, not capability
claims. No model number is quoted or moved; no T0/T1 number appears here.
**Evidence class** MEASURED (ours) unless stamped otherwise. Box: dev box, **CPU-only**,
torch 2.11.0+cu128.
**Thor untouched.** No ssh, no GPU work, no contact with `~/experiments/v6F-SW-30k`.
**Suite** `3346 passed / 0 failed / 17 skipped / 2 xfailed` — **identical to the brief's baseline**.

---

## 0. Pre-registration — written before the numbers existed

⛔ Stated first, on purpose. Everything from §2 onward was run *after* this section was fixed.

### 0.1 Experiment B — the masking / regularisation gradient probe

| outcome | reading | meaning |
|---|---|---|
| **orthogonal** | `cos(g_term, g_rest) ≈ 0` | the term adds an **independent** signal |
| **redundant** | `cos > 0`, sign-consistent | it **amplifies** what the rest already says |
| **fighting** | `cos < 0`, sign-consistent | it **opposes** the rest of the objective |
| **null** | `‖g_term‖ = 0` on the trunk | the lever never reaches what it claims to shape |

⚠️ **A cosine near zero is NOT automatically "independent."** Two random vectors in *D* dimensions
have `|cos| ≈ 1/√D` by chance. Every cosine below is reported against `chance_level` **and as a
multiple of it**, plus the full pairwise term × term matrix — so *"O3 is orthogonal"* can be checked
against *"everything here is orthogonal,"* which would make the statement about the dimension.

### 0.2 Experiment A — the `w_o6` study

⛔ **NOT a keep/drop study.** The PI corrected the framing mid-turn, **before any Experiment-A
number existed**. SigReg is a **MUST** — it comes from the LeWM/LeJEPA line and the decision is
made. The `w_o6 = 0` arm exists **only as the control that makes the on-arm interpretable**. The
open questions are whether SigReg is used **correctly**, whether it is **forgotten**, and whether
its effectiveness is **validated** against the collapse specific to predictive/JEPA architectures.

| outcome | what we do |
|---|---|
| **1. measurably prevents rank collapse** | validated — report effect size and the retention it buys; that becomes what the S-W gate defends |
| **2. no separated difference** | ⚠️ the finding is about our **CONFIGURATION or our POWER**, not about SigReg. Honest reading: *"we cannot yet demonstrate the effect."* Deliverable = what would make it demonstrable. **Never** recommend removal |
| **3. appears to harm** | a **mis-configuration** finding — deliverable is *which knob*, and escalate |

**Estimator commitments, fixed in advance:** pooled spectrum (ceiling 1535, not 47) ·
leave-one-cluster-out **jackknife**, never the bootstrap · ≥3 seeds or say plainly that I did not ·
a null-detection control for every probe · four metric families per family, never pooled ·
⛔ never `overlapping_holdout_se`.

**Outcome reached: a split.** Question (1) — *does SigReg see and resist collapse at our
configuration* — is **outcome 1, validated** (§4). Question (2) — *does the live run need it* —
is **outcome 2, a power failure**, and it is a GPU item (§3).

---

## 1. The answer first

⭐ **SigReg's statistic separates a 2× rank collapse from healthy by 42 estimator standard
deviations at our exact live geometry — while `effective_rank`, the monitor built to watch for
collapse, moves 0.3 % on the same event and cannot see it.** We have been reading the blind
instrument and ignoring the sensitive one sitting next to it in the same log.

| on the same 48-row batch, same 2× collapse (2048 → 1024 retained dims) | healthy | collapsed | separation |
|---|---:|---:|---|
| `effective_rank` (the O6 **monitor**) | 46.86 | 46.73 | **0.3 %** — buried in its own noise |
| **`o6_sigreg` (the loss term itself)** | **0.402** | **1.228** | **3.05×, and 42.4 σ** |

⚠️ **Provenance of that comparison.** Both `effective_rank` values and both `o6_sigreg` values are
MEASURED here (`raw/sigreg_response.json`), on the same latents. The *"buried in its own noise"*
claim rests on `effective_rank`'s CV **0.1204** at n = 48, which is **INHERITED** from
`SIGREG_GATE_POWER.md` §3.2 and not re-measured by me — but the direction does not depend on it: a
0.3 % move against a CV of even 0.01 would still be invisible, and that document's independently
derived isotropic reading (46.861 ± 0.006) matches my 46.86 to three decimals.

⭐ **`--sigreg-slices 512` is CORRECTLY configured and is doing real work.** It is a pure
**variance** knob (the mean is identical across 8 → 2048 slices) and it cuts the estimator's CV
**7.8×** versus the 8-slice test fixture — a 1/√M law, MEASURED. Crucially the noise is **worst at
the healthy end**, which is where the live run should be sitting: at 8 slices the loss would carry
**36.6 %** noise; at 512 it carries **4.7 %**.

⭐ **O3 (masking) and O6 (SigReg) are the only two S-W terms not aligned with the rest of the
objective — and the only two whose own gradient was AT OR BELOW the pre-fix resample noise floor.**
The two terms the PI asked about are exactly the two that were unmeasurable before `142ce34` landed
yesterday. That is not a coincidence; it is why nobody could answer this before.

| S-W term | `cos(g_term, g_rest)` on trunk | vs chance | sign stable over 5 seeds? | ‖g_term‖ ÷ pre-fix noise floor |
|---|---:|---:|---|---:|
| O1 (control/factual/scene) | +0.322 | 164× | ✅ | 9.34× |
| O2 (near-field) | +0.508 | 258× | ✅ | 4.08× |
| O5 (rollout consistency) | +0.577 | 293× | ✅ | 2.70× |
| **O3 (masked cells)** | **+0.052** | 27× | ❌ **flips** | **1.00×** |
| **O6 (SigReg)** | **−0.012** | 6× | ❌ **flips** | ⛔ **0.69×** |

⛔ **`o6`'s entire trunk gradient (1.151) is smaller than the noise two *identical* arms used to
differ by (1.658).** Any `w_o6` A/B before yesterday was reading resample noise, not the lever.

⛔ **The live-run half of Experiment A could NOT be done on CPU, and I did not fake it.** MEASURED:
on a synthetic build self-distillation collapse **does not develop** — with SigReg fully OFF and the
loss driven down **8.6×**, pooled effective rank moved **446.49 → 453.10** (it went *up*). §3.

---

## 2. Experiment B — the paired single-batch gradient probe

`code/mask_grad_probe.py` → `raw/mask_grad_probe.json`. **2.54 s, 0 GPU**, 5 seeds, stage S-W.

### 2.1 The estimator is EXACT, not statistical

Gradients are linear in the loss, so with `generator` and `sigreg_generator` fixed and everything
else bit-identical:

```
g(lever ON) − g(lever OFF)  ==  the lever term's OWN gradient, exactly
```

No fitting, no resampling, no interval on *this* quantity — it is an identity. The only uncertainty
is **across seeds**, which is where it is reported.

### 2.2 ⛔ A fixture artifact that nearly became a false finding

My first run reported **O3's trunk gradient as exactly 0.0** — which would have meant the masking
objective trains only its own aux head and never touches the representation. **False**, and the
mechanism is a live configuration hazard.

`sample_cell_block_mask` places `n_blocks` blocks of `block_hw` on a `grid × grid` field. At
**grid = 2** with the default **2 × 2** blocks, **the block IS the whole grid** → `mask_rate 1.0` →
`MaskedCellPredictor.forward` (`v6.py:2089`) replaces **every** cell with the learned mask token →
its input no longer depends on `cells` → **the trunk gradient is exactly zero, by construction.**

| geometry | cells | `o3_mask_rate` | ‖g_O3‖ on trunk | reaches trunk? |
|---|---:|---:|---:|---|
| grid 2 × 2 (the `stack/tests` fixture) | 4 | **1.0000** | **0.000** | ⛔ **no** |
| **grid 4 × 4 (LIVE)** | 16 | **0.4375** | **1.663** | ✅ yes |

The live run is `--readout-grid 4 --o3-block-h 2 --o3-block-w 2` (MEASURED from the running
trainer's argv), so **the live configuration is fine**. But the degenerate case is one flag away, it
fails silently, and the loss still falls — the aux head just learns the marginal. `o3_mask_rate` is
now stamped into the probe's record for the same reason `rank_ceiling` is stamped into the spectrum
record: so the misreading is not available.

### 2.3 The controls — all three fire correctly, in both geometries

| control | expectation | MEASURED (grid 4) | ✓ |
|---|---|---:|---|
| **N1** same arm twice, seeded | **exactly 0** | **0.0** | ✅ |
| **N2** ⭐ *null-detection*: a structurally inert lever — `t1_latent` 0→1 in S-W, which `for_stage` forces to zero | **exactly 0** | **0.0** | ✅ |
| **N3** ⭐ *positive*: same arm twice on the **incumbent** global-RNG path | **non-zero** | **1.658** (8.86 % of the trunk gradient) | ✅ |

**N2 is the null-detection control the brief demanded**: a lever that genuinely does nothing, moved
through its full range, reported as no effect. **N3 is its mirror** — the probe is not merely
insensitive — and it *is* the pre-fix noise floor quoted in §1.

### 2.4 The interpretation — and the check that it is not just dimensionality

Pairwise `cos(g_i, g_j)` on the trunk, mean over 5 seeds (chance = `1/√257995` = **0.00197**):

| | o1 | o2 | o3 | o5 | o6 |
|---|---:|---:|---:|---:|---:|
| **o1** | — | +0.321 ✅ | +0.127 ❌ | +0.322 ✅ | −0.011 ❌ |
| **o2** | | — | +0.028 ❌ | **+0.870** ✅ | +0.005 ❌ |
| **o3** | | | — | +0.031 ❌ | +0.042 ❌ |
| **o5** | | | | — | −0.019 ❌ |

✅ = sign-consistent across all 5 seeds; ❌ = sign flips, i.e. **not separated from orthogonal**.

⭐ **This is what makes the headline admissible.** *"Everything is orthogonal at high D"* is
refuted: the three fitting terms sit at **163–293× chance with stable sign**, while O3 and O6 sit at
**27×** and **6×** with **flipping sign**. The contrast is real and it is not the dimension.

**Reading it against §0.1:**

- **O6 (SigReg) — ORTHOGONAL.** Independent of every other term (|mean cos| ≤ 0.042, all
  sign-unstable) ⇒ it **adds an independent signal**; **not** fighting the trunk, **not** redundant
  with it. Its pull is **5.3 %** of the rest of the objective's trunk gradient.
- **O3 (masking) — NOT SEPARATED from orthogonal at 5 seeds.** Mean +0.052 (27× chance) but the sign
  flips (−0.285 … +0.306). Honest statement: *independent, but the sign of its small alignment is
  unresolved at n = 5 seeds* — not *"orthogonal, established."* Pull **7.7 %**.
- **O2 ↔ O5 — REDUNDANT, and this one IS resolved.** +0.870, sign-stable, tight range (0.845–0.904).

### 2.5 Where each measure lands — per module, never pooled

| module group | ‖g_O3‖ | ‖g_O6‖ | ‖g_rest‖ |
|---|---:|---:|---:|
| `encoder` | 1.148 | **1.082** | 16.373 |
| `readout` | 0.123 | **0.121** | 1.924 |
| `predictor_op` | **0.327** | ⛔ **0.000** | 8.774 |
| `masked_cells` (aux) | **0.967** | 0.000 | ⛔ **0.000** |

1. ⭐ **SigReg exerts NO force on the predictor.** It is applied to `z_op_win` — the *encoded*
   window — which never passes through `predictor_op`. O6 shapes **encoder + readout only**. Correct
   and expected, but it **bounds what the measure can do: SigReg cannot prevent a collapse that
   lives in the predictor**, only one in the encoded representation.
2. **`masked_cells` receives gradient from O3 and nothing else.** Turn O3 off and it is a frozen
   random head.

---

## 3. Experiment A, part 1 — the training ablation, and why I stopped

`code/collapse_trajectory.py` → `raw/collapse_trajectory.json`.

An ablation that finds "no separated difference" is informative **only if the thing being defended
against was present**. So before spending the budget I ran the control arm — **`w_o6 = 0`, SigReg
fully OFF, maximum collapse pressure** — and watched pooled effective rank (1536 rows, ceiling
1535, fixed probe set):

| condition | lr | ER start → end (1200 steps) | loss | verdict |
|---|---:|---:|---:|---|
| **SELF** (collapse-capable) | **1e-4** (live) | **446.49 → 453.10** (**+1.5 %**) | 2.06 → 0.16 | ⛔ rank went **UP** |
| **SELF** | 1e-3 (10× live) | 446.49 → **432.72** (−3.1 %) | 0.20 → 0.024 (**8.6×**) | negligible |
| **FIXED** (null targets) | 1e-3 | 446.49 → 443.75 (−0.6 %) | flat ≈ 2.71 | as designed |

⇒ **Collapse does not develop.** The loss falls 8.6× while the rank barely moves. Running `w_o6`
0.1 vs 0.0 there would compare two arms in a world without collapse, and its null would be **a
statement about my fixture**. Per the brief, I stopped rather than reporting it as the answer.

**Why the fixture cannot collapse** — four named reasons, so the next attempt does not repeat it:

1. ⭐ **The inputs are `torch.randn` frames.** The collapse shortcut is *"exploit the structure
   shared across samples"*; random noise has none. Real driving video is almost all shared
   structure — which is exactly where the shortcut lives.
2. **Encoder `d_model 32, depth 1`** vs the live **768 / 12**.
3. **`o5_k = 2` vs the live 60.** The long rollout is where the incentive concentrates: a constant
   latent is trivially predictable over 60 steps.
4. **1 200 steps vs 30 000.**

### 3.1 The target construction is load-bearing — and the obvious fixture gets it wrong

The real trainer builds targets from **the model's own encoder, detached**
(`train_v6_staged.py:1974-1978`):

```python
z_flat = stack.readout(stack.encoder(ff.reshape(fb * fk, *ff.shape[2:])))
z_true = [z_flat[:, j].detach() for j in range(need_k)]
```

Self-distillation: the encoder shapes **both** the prediction and the target, so it can lower the
loss by making its own output trivially predictable. Detaching stops gradient *into* the target; it
does **not** remove the incentive, because the same weights produce the target next step. **This is
the JEPA collapse mode SigReg exists to prevent.**

⛔ **`synthetic_train_batch` (`train_v6_staged.py:1512`) instead uses fixed external `torch.randn`
targets** — collapsing the encoder cannot make a fixed random target easier to hit, so **there is no
collapse incentive at all**. Any ablation built on the stock synthetic batch is structurally
incapable of seeing SigReg work. My harness implements both (`SELF` / `FIXED`); the `FIXED` arm is
the null and the trajectory confirms it stays flat.

---

## 4. Experiment A, part 2 — the mechanism, measured directly

Since collapse will not *develop* on CPU, `code/sigreg_response.py` asks the question **without
waiting for it**: construct a latent collapsed **by construction** at the live geometry
(`d_op = 2048`, `n = 48`), and ask whether SigReg *sees* it and what `--sigreg-slices` buys.
Collapse is a **squeeze**, not a hard truncation — a truncation is the easiest thing to detect and
would flatter the instrument.

### 4.1 ⭐ Does the term SEE collapse at our geometry? — YES, overwhelmingly

n = 48, d_op = 2048, `--sigreg-slices 512`, `--sigreg-free-dims 0` (all live values); estimator sd
from 32 independent direction draws at fixed z:

| retained dims | `effective_rank` | `o6_sigreg` | separation from healthy |
|---:|---:|---:|---:|
| 2048 (healthy) | 46.86 | 0.4023 ± 0.0195 | — |
| **1024 (a 2× collapse)** | **46.73** | **1.2283** | **42.4 σ** |
| 512 | 46.43 | 3.1006 | 138.4 σ |
| 256 | 45.79 | 4.9214 | 231.9 σ |
| 64 | 41.38 | 6.9755 | 337.2 σ |
| 16 | 23.83 | 7.6035 | 369.5 σ |

⭐ **Monotone, 18.9× dynamic range, and the mildest collapse tested is already 42 σ away.** At our
d_op and our batch geometry the SigReg statistic is an extremely sensitive collapse detector.

⭐ **AND IT CORRECTS A STANDING CLAIM.** `SIGREG_GATE_POWER.md` §7.1 warns that *"a regulariser's
loss can fall because it is satisfied **or** because the representation degenerated."* MEASURED
here: **degeneration RAISES the Epps-Pulley statistic, monotonically, at every level.** So the
second branch is **refuted for this statistic at this geometry** — a *falling* `o6` cannot be
collapse. ⚠️ It is a *relative* signal (it needs a within-run baseline), and it was measured on
synthetic latents under a squeeze collapse model, not on the live representation.

### 4.2 ⭐ What `--sigreg-slices 512` actually buys — a VARIANCE knob, correctly set

Fixed z, **32 independent direction draws**, n = 48 (the estimator isolated from the data):

| retained | slices 8 | slices 64 | **512 (LIVE)** | 2048 |
|---|---:|---:|---:|---:|
| **2048 (healthy)** CV | **0.3662** | 0.0854 | **0.0470** | 0.0142 |
| 256 CV | 0.0338 | 0.0140 | 0.0047 | 0.0023 |
| 16 CV | 0.0040 | 0.0018 | 0.0006 | 0.0003 |
| *loss mean* @ healthy | 0.4380 | 0.4124 | 0.4152 | 0.4194 |

Three findings:

1. **It is a pure variance knob, not a bias knob** — the mean is identical across 8 → 2048 at every
   collapse level (0.438 / 0.412 / 0.415 / 0.419).
2. ⭐ **512 cuts the CV 7.8× versus the 8-slice fixture**, following the expected 1/√M law
   (8 → 512 is 64×; √64 = 8; MEASURED 7.8×).
3. ⛔ **The noise is WORST at the healthy end — where the live run should be sitting.** At 8 slices
   the statistic carries **36.6 %** noise there; at 512, **4.7 %**. A low slice count would inject
   that noise straight into the encoder's gradient. ⇒ **`--sigreg-slices 512` is validated as
   correctly configured**, and 2048 would buy a further 3.3× for 4× the compute.

### 4.3 ⛔ SigReg's cost is QUADRATIC in the row count — and that bounds every recommendation

`sigreg.py:118` materialises `diff = proj[None] − proj[:, None]` of shape **[n, n, M]** in fp32:

| n | M | intermediate |
|---:|---:|---:|
| 48 (live) | 2048 | 19 MB |
| 192 | 2048 | 302 MB |
| 1536 (pooled) | 2048 | ⛔ **19.3 GB** |

MEASURED: an n = 1536 sweep reached a 7.3 GB working set, made **no progress in 37 minutes**, and
was killed by explicit PID. It was thrashing, not computing. ⇒ **SigReg acts on the current batch's
48 rows (4 episodes at `--eps-per-batch 4`) and CANNOT simply be evaluated on the pooled set.**
Pooling is available to the **monitor**, not to the loss. Any "just use more rows" proposal for the
loss must price this.

### 4.4 ⚠️ A metric I designed, measured, and am NOT quoting as a finding

I built an `anti_collapse_gain` — does a normalised SigReg gradient step raise effective rank more
than a **matched-norm random step**? It came out **negative nearly everywhere**, which reads like
"SigReg harms." **It does not, and I am recording why rather than reporting the number.**

The decomposition shows the sign is driven **entirely by the control**: `d_er_sigreg` ≈ 0
(+0.0001 … +0.0033) while `d_er_random` grows to **+0.6134**. Effective rank is **maximised by
isotropy**, so an isotropic random perturbation is *by construction* the optimal one-step
rank-raiser and **any** structured direction loses to it. The comparison is stacked, it is close to
tautological, and it cannot distinguish *"SigReg does not push toward rank"* from *"one step of
anything structured loses to isotropic noise."* **A single gradient step is the wrong probe for a
regulariser whose effect is a fixed point over many steps.** The key survives in the artifact as
`anti_collapse_gain_DIAGNOSTIC_do_not_quote_as_harm`, so the negative result stays visible instead
of being quietly dropped.

⚠️ Per §0.2 outcome 3, an apparent harm must be escalated with evidence. **I am explicitly NOT
escalating it**, because the evidence says the metric is ill-posed, not that SigReg harms.

---

## 5. The four metric families

Stated **per family with the reason and the n**, never pooled and never silently dropped.

⛔ **Three of the four are not merely uncomputable on my fixture — they are UNDEFINED AT THIS STAGE
OF THE LADDER.** `V6LossWeights.for_stage("S-W")` sets `t1_latent = s1_latent = lambda_plan =
seam_op = w_select = w_anchor = 0`: **in S-W the planner is ABSENT by construction**, which is what
makes S-W attributable as a pure world stage. A tactical or strategic number from S-W would be a
number from a module that contributed exactly zero loss.

| family | computable? | reason | n |
|---|---|---|---:|
| **LONGITUDINAL** — target speed, headway / time-gap / TTC | ❌ | needs a **lead agent** in frame and real ego trajectories; inputs are `torch.randn`, so there are no agents and no trajectories | **0** |
| **LATERAL** — heading, curvature, yaw-rate, cross-track | ❌ | needs real ego poses; the fixture's `gt_wp` is a zero tensor by construction | **0** |
| **TACTICAL** — manoeuvre decision, goal/anchor selection | ❌ | **planner absent in S-W by design** (`lambda_plan ≡ 0`, `w_select = 0`); no manoeuvre is emitted at all | **0** |
| **STRATEGIC** — strategic decision, route/goal setting | ❌ | `s1_latent = 0` in S-W and `layer_str` is frozen (`STAGE_GROUPS`) | **0** |

⇒ **These become computable at S-T / S-S / S-J on the real corpus, and that is where they must be
reported for any capability claim about O3 or O6.** ⚠️ Neither experiment here is a capability
claim — both are loss-mechanics and representation-geometry results, which is why they carry no tier
stamp. **A missing metric is a work item, not an excuse**: the work item is §7's GPU-side paired
training A/B, which *does* carry all four.

---

## 6. Seeds, estimators, and what is NOT claimed

- **Experiment B: 5 seeds** (model init + batch), paired, estimator **exact by linearity**.
- **§4.1/§4.2: 3 seeds** for the response curve; the slice-variance table is **32 independent
  direction draws at fixed z**, which isolates the knob from the data.
- **§3: 1 seed.** It is a **power precondition check**, not an effect estimate, and I am stating
  that plainly rather than dressing it as one.
- ⛔ **No claim here that SigReg improves any driving metric.** Determinism ≠ significance; a
  same-seed A/B is *attributable*, not *significant*. The capability claim needs §7.1.
- ⛔ **Not once is a falling loss read as a measure working.** §4.1 goes the other way: it measures
  the loss's *response to imposed collapse*, which is the only direction that licenses reading it.

---

## 7. Escalations — decisions, not documentation

1. ⛔ **The live-run validation needs a GPU and the real corpus. It cannot be done on CPU and I did
   not fake it.** Required: real PhysicalAI frames (the collapse shortcut lives in shared visual
   structure), the live encoder width, `o5_k = 60`, and enough steps. Shape: a **paired S-W A/B at
   matched steps**, `w_o6` 0.1 vs 0.0, read on the **pooled** spectrum (`--spectrum-accum 32`,
   ceiling 1535) with the cluster jackknife — **plus all four metric families**. It belongs in the
   queue **behind** the live run; nothing here justifies touching v6F S-W.
2. ⭐ **Use the `o6` loss as the collapse alarm we already have.** MEASURED (§4.1): on the same
   48-row batch, a 2× collapse moves `effective_rank` **0.3 %** (invisible) and `o6_sigreg`
   **3.05×, 42 σ**. The monitor is nearly blind exactly where the loss term is exquisitely
   sensitive. **A baselined `o6` trend is a better S-W collapse guard than the rank gate at n = 48**
   — and it is already in every log at zero cost. This deserves a PI decision.
3. ⚠️ **The live S-W gate will return INCONCLUSIVE on O6 as configured** (`--spectrum-accum` unset ⇒
   ceiling 47 < 1024, clause 1). **INCONCLUSIVE is not a pass.** 0-GPU fix at the next natural
   restart — do **not** kill a job ~6 days from done for a monitor.
4. ⭐ **O2 and O5 are nearly collinear (`cos = +0.870`, sign-stable over 5 seeds).** Two
   independently-weighted measures pulling the trunk the same way. A real finding about the
   objective's design that nobody asked for: *is O2 buying anything over O5 at its current weight?*
   Not investigated here.
5. ⚠️ **A one-flag silent failure: `--readout-grid 2` with 2×2 O3 blocks fully masks the grid and
   O3's trunk gradient becomes exactly zero** while the loss still falls. The live run is not in this
   state. Worth a guard — `o3_mask_rate == 1.0` should warn, in the spirit of `rank_ceiling`.
6. ⚠️ **`stack/tests/test_loss_determinism.py`'s minimal fixture IS that degenerate geometry.**
   Harmless for bit-exactness; misleading if reused for an effect study.
7. ⚠️ **SigReg cannot defend the predictor** (§2.5) and **cannot be pooled** (§4.3). Both bound what
   "SigReg prevents collapse" can mean: it constrains the **encoded representation**, on **48 rows
   from 4 episodes**, per step.

### Retraction-class note

**Class: a control that is optimal by construction, read as a baseline** (§4.4). Comparing a
structured direction against isotropic noise on a rank functional is a rigged race — isotropy
*maximises* the statistic being measured. Same family as the programme's scope-mismatch traps
(`df`, Thor `free`, cgroup `usage_in_bytes`, `effective_rank` against `d`): **the number was
well-formed and answered a different question than the one asked.** Caught by decomposing the metric
into its two components before quoting it.

**Second, smaller class: a test fixture whose geometry silently disables the thing under test**
(§2.2) — `grid 2` fully masks the O3 grid. Generalises: *before reusing a unit-test fixture for an
effect study, check that every term under test is actually live in it.*

---

## 8. Deliverable manifest

| artifact | path (repo-relative) | state |
|---|---|---|
| this document | `TanitAD Research Hub/…/incoming/2026-08-16-o6-ablation/O6_ABLATION_AND_MASK_PROBE.md` | **STAGED** |
| gradient probe | `…/2026-08-16-o6-ablation/code/mask_grad_probe.py` | **STAGED** |
| its artifact | `…/2026-08-16-o6-ablation/raw/mask_grad_probe.json` | **STAGED** |
| collapse-trajectory diagnostic | `…/2026-08-16-o6-ablation/code/collapse_trajectory.py` | **STAGED** |
| its artifact | `…/2026-08-16-o6-ablation/raw/collapse_trajectory.json` | **STAGED** |
| ablation harness (SELF/FIXED, pooled + jackknife) | `…/2026-08-16-o6-ablation/code/w_o6_ablation.py` | **STAGED** |
| anti-collapse response + slice-variance instrument | `…/2026-08-16-o6-ablation/code/sigreg_response.py` | **STAGED** |
| its artifact | `…/2026-08-16-o6-ablation/raw/sigreg_response.json` | **STAGED** |
| suite evidence — **3346 passed / 0 failed / 17 skipped / 2 xfailed** | `…/2026-08-16-o6-ablation/raw/stack_pytest.txt` | **STAGED** |

**Nothing is on a pod or in a worktree. No commit, no push. `stack/` was NOT modified** — every
instrument is additive and lives under `incoming/`. The suite count is **identical to the brief's
baseline** in all four categories, which is the expected result for a turn that added no test and
touched no library code.
