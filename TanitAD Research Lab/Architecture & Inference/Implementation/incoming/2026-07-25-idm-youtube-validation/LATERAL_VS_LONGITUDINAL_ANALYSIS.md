# Lateral vs longitudinal error decomposition — and why "lateral deviation is not considered"

**Date:** 2026-07-25 · **Trigger:** PI question — *"I'm worried about the topic of non-considered lateral
deviation, how can we improve it?"* · **Evidence class: MEASURED**, computed directly from the committed
reconstruction `recon_ep00020.npz` (n=175 windows, held-out clip `ep_00020`, ego frame).

**Axis convention verified before any claim:** axis0 mean |displacement| = 5.22/10.65/16.28/**22.10 m** at
0.5/1/1.5/2 s against a mean speed of 10.15 m/s (⇒ ~20.3 m expected at 2 s) → **axis0 = along-track
(longitudinal), axis1 = cross-track (lateral)**. Confirmed, not assumed.

---

## 1. The decomposition

| horizon | L2 (ADE) | \|lon\| | \|lat\| | lat % of L2 | **longitudinal share of squared error** |
|---|---:|---:|---:|---:|---:|
| 0.5 s | 1.201 | 1.199 | 0.042 | 3.5 % | **99.8 %** |
| 1.0 s | 2.004 | 1.996 | 0.125 | 6.2 % | **99.6 %** |
| 1.5 s | 2.990 | 2.966 | 0.241 | 8.0 % | **99.2 %** |
| 2.0 s | 3.938 | 3.836 | 0.589 | 15.0 % | **97.9 %** |
| **all** | **2.533** | **2.499** | **0.249** | **9.8 %** | **98.6 %** |

### 1.1 The structural problem, in one line
**The ADE this program optimizes and reports is dominated by the longitudinal axis.** The lateral channel
receives a small minority of the squared-error signal, so any loss or metric built on undecomposed L2 is,
numerically, a *mostly-longitudinal* loss.

> ⚠️ **CORRECTED 2026-07-25 (same day, before this doc was used for a decision).** The figure first written
> here — *"98.6 % longitudinal / ~1.4 % lateral"* — is **`ep_00020`-specific and does NOT replicate**. The
> HPP-1 agent measured the energy share across **8 committed arms**: it ranges **0.607–0.976**, and the
> deployed **flagship v1 sits at 0.873** (⇒ lateral ≈ **13 %**, not 1.4 %). The *structural* claim survives —
> longitudinal dominates everywhere measured (61–98 %), so lateral is systematically under-weighted — but
> **"nearly invisible" was an n=1 overstatement and is withdrawn.** The quotable form is: *lateral carries
> ~2–39 % of the squared-error signal depending on the arm, ~13 % on the deployed model.*
> **What DOES replicate is the finding that actually matters — see §3: the compounding law holds 8/8.**

This is *not* a claim that the model's lateral behaviour is bad here (it is decent — see §2). It is a claim
about the **instrument and the objective**: they cannot see, and do not reward, the axis that causes lane
departures.

---

## 2. Two readings, both true — and the second is the dangerous one

**Reading A (reassuring):** mean lateral error is small — 0.249 m overall, median 0.330 m at 2 s — and the
path *shape* is well recovered (yaw-rate R² **0.940**). The IDM is not wandering.

**Reading B (the concern, MEASURED):** the **tail** is already unsafe at a 2 s horizon, and the mean hides it.

| statistic (cross-track \|lat\| @ 2 s) | value |
|---|---:|
| p50 | 0.330 m |
| p75 | 1.028 m |
| p90 | **1.404 m** |
| p95 | **1.489 m** |
| p99 | **1.920 m** |
| max | **2.062 m** |
| fraction > 0.5 m | **40.6 %** |
| fraction > 1.0 m | **26.3 %** |
| fraction > 1.75 m (≈ half lane width) | **2.9 %** |

A 1.4 m cross-track error is a **lane departure** in any real lane. It occurs at p90 — one window in ten —
while the headline ADE (2.53 m) reports a number dominated by a *speed* error that would not, by itself,
leave the lane.

---

## 3. The decisive finding — lateral is the COMPOUNDING axis

| horizon | lon error (growth) | lat error (growth) | lon error as % of distance travelled |
|---|---:|---:|---:|
| 0.5 s | 1.199 (×1.00) | 0.042 (×1.00) | 23.0 % |
| 1.0 s | 1.996 (×1.66) | 0.125 (×3.00) | 18.7 % |
| 1.5 s | 2.966 (×2.47) | 0.241 (×5.77) | 18.2 % |
| 2.0 s | 3.836 (**×3.20**) | 0.589 (**×14.11**) | 17.4 % |

**Longitudinal error is a bounded SCALE error** (~17–23 % of distance travelled, *stable-to-improving* with
horizon — the classic monocular scale ambiguity). **Lateral error COMPOUNDS: ×14.1 over the same window,
4.4× faster than longitudinal.** Its share of L2 rises monotonically 3.5 % → 15.0 % and shows no sign of
saturating.

**Extrapolating the axis that is growing 4.4× faster is exactly what E1a measured on the planner side:** at
K=185 (18.5 s), the failure mode is *corridor departure* — a **lateral** failure — at 59 % overall / 84 % at
junctions, with **peak cross-track error 38.94 m**. Two independent instruments, same conclusion:

> **Longitudinal error is what the 2 s metric shows you. Lateral error is what actually ends the drive.**

---

## 4. Why the program has under-served lateral (root cause, not blame)

1. **The "83 % of 2 s error is longitudinal" finding is correct — and it was over-generalized.** At 2 s the
   error *is* longitudinal-dominated (we measure 98.6 % here), which correctly motivated the longitudinal
   lever. What was never checked is that this ratio is **an artifact of the 2 s horizon**: longitudinal is
   bounded, lateral compounds, so the ratio inverts as the horizon grows.
2. **L2/ADE mixes the axes and is dominated by the larger one.** With ~22 m of forward travel and ~0.85 m of
   lateral travel at 2 s, the lateral signal is buried by construction.
3. **The corpus is 74 % straight / 0 % semantic** — most windows have almost no lateral dynamics to get
   wrong, so aggregate lateral statistics look benign while the tail (turns, junctions) carries the risk.
4. **Same class as the E1a horizon confound and the `band_ade2d` knife-edge lesson**: an aggregate statistic
   concealing a safety-critical stratum. This is a recurring, now three-times-observed pattern.

---

## 5. How to improve it — concrete, ordered

### M1. Decompose EVERY trajectory error into lateral + longitudinal, everywhere *(S, do immediately)*
No ADE is reported without its `(lat, lon)` split. Add to `taniteval` metrics and to every results JSON.
**Rationale:** you cannot manage what you never separate. This alone would have surfaced the issue months
ago. *(Pairs with Wave-1-A, which is already touching the metrics layer.)*

### M2. Make cross-track error a first-class GATE metric — with TAIL statistics *(M, Wave 2)*
Gate on **p90/p95/max XTE and the fraction beyond a lane-relative threshold**, not the mean. The mean is
0.25 m while p90 is 1.40 m — the mean is the *least* informative statistic for a safety axis.
Fold into the Tier-1 #1 gate-primary change (`corridor_departure_rate @ K=max`), which is itself a lateral
metric — the two proposals reinforce each other.

### M3. Re-balance the objective so lateral receives real gradient *(M, needs a training run)*
Today lateral gets ~1.4 % of the squared-error signal. Options, cheapest first:
- **Per-axis normalization** — scale each axis by its own std before the loss, so lateral and longitudinal
  contribute comparably (near-free, no new terms).
- **Explicit lateral term** with its own weight, tuned on the tail rather than the mean.
- **Heading/yaw-consistency auxiliary** — yaw-rate is already the best-predicted channel (R² 0.940); a
  consistency term between predicted yaw and lateral displacement should transfer that strength into
  cross-track accuracy.
⚠️ Pre-register with a falsifier: re-weighting **will** cost some longitudinal ADE. The pre-registered
question is whether XTE-p90 improves more than ADE degrades — and the answer must be judged on the
*decomposed* metrics from M1, never on aggregate ADE (which will simply report the trade as a regression).

### M4. Evaluate at horizons where lateral dominates *(already in flight)*
The 2 s window is where lateral looks smallest. E1a's K=185 is where it is decisive. Every lateral claim
should carry its horizon — the same rule the program already applies to estimators and exponents.

### M5. E1b is already the right lever — confirm this framing *(running now on pod3)*
Failure-gated CL-SFT supervises anchor scores toward the *recovering* anchor: a **lateral** recovery
objective. E2a's decomposition supports it precisely — the lateral offset is **perceivable (oracle R² 0.72,
ceiling ρ 0.91)** and **91 % of the loss is downstream** (the planner ignores what it sees). So the fix is
the objective, exactly as M3 argues for the IDM.

### M6. Fold lateral into the Hierarchy Proof Program *(design note)*
Route-following, junction handling and lane-keeping **are lateral phenomena**. HP-2 (advantage concentrates
at decision points) and HP-3 (route-counterfactual: same scene, different `nav_cmd` ⇒ different trajectory)
are measured in the **cross-track** channel. Fixing lateral instrumentation is therefore a **prerequisite for
proving the hierarchy** — a strategic level that chooses a route can only demonstrate its value on the axis
that expresses route choice. M1/M2 are on the critical path of HPP-2/HPP-3.

---

## 5b. ⚠️ SECOND-CLIP REPLICATION — the growth law holds, and clip `ep_00020` was the FAVOURABLE case

Run immediately after §1–§5 on the other committed reconstruction (`recon_ep00009.npz`, n=175), because an
n=1 tail statistic is inadmissible here (C5). **MEASURED:**

| | `ep_00020` (§1–§3) | `ep_00009` |
|---|---:|---:|
| longitudinal share of squared error | 98.6 % | **84.6 %** |
| lon growth 0.5→2 s | ×3.20 | ×4.43 |
| **lat growth 0.5→2 s** | **×14.11** | **×26.14** |
| **lateral growing faster by** | **4.4×** | **5.9×** |
| \|lat\|@2 s p50 | 0.33 m | **3.43 m** |
| \|lat\|@2 s p90 | 1.40 m | **5.68 m** |
| \|lat\|@2 s max | 2.06 m | **7.54 m** |
| windows > 1.0 m cross-track | 26.3 % | **73.1 %** |

**Two conclusions, and the second is the important one:**

1. **The compounding law REPLICATES and strengthens** — lateral error grows 4.4×/5.9× faster than
   longitudinal on two independent clips. This is now the **durable, replicated** claim of this analysis.
2. **"Reading A (reassuring)" in §2 was itself an n=1 artifact.** On `ep_00009` the *median* cross-track
   error at 2 s is **3.43 m** — several lane widths — and **73 % of windows exceed 1 m**. Yet longitudinal
   still carries **84.6 %** of the squared error, so **the aggregate ADE headline for this clip would still
   read as a longitudinal problem while the vehicle is metres out of its lane.** This is the concealment
   mechanism, demonstrated end-to-end on real data.

⚠️ **This also qualifies the headline IDM result.** The reported `ep_00020` ADE@2s = 2.53 m is a **favourable
clip on the lateral axis**; a lateral-aware reading of the same model on `ep_00009` is far worse. The
reconstruction result stands as reported (it was correctly caveated and its controls are sound), but any
forward-looking claim about IDM *lateral* quality must wait for the 40-episode decomposition below.

---

## 6. Honest bounds on this analysis

- **n = 2 clips** (350 windows). The **growth-rate ordering replicates on both** and matches the independent
  E1a planner-side result at 18.5 s — that is the claim to carry forward. The **absolute tail percentiles
  vary enormously between clips** (p50 0.33 m vs 3.43 m), so no single-clip tail number is quotable; per C5,
  bucket over clips, never a single row.
  **Next check (cheap, 0 GPU, pure post-processing on existing artifacts):** run the same decomposition
  across **all 40 held-out episodes**, stratified by turn/straight, with the episode-cluster bootstrap. This
  converts the finding from "replicated on 2" to decision-grade — and it is the natural first task of M1.
- Channel identities: ch0 = **speed** and ch1 = **yaw-rate** are confirmed (they reproduce the independently
  reported MAE 2.08 m/s and 0.027 rad/s). The remaining two scalar channels were **not** confirmed against
  the head definition and are deliberately **not quoted** here.
- This clip is PhysicalAI (gated) — internal use only, never published.
