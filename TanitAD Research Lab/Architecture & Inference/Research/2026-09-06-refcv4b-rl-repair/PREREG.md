# PRE-REGISTRATION — `H-RL-PROGRESS-LEADCAP-1`: repair `progress`, and test it by its own prediction

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE THE CODE CHANGE.**
**date:** 2026-09-06 · **owner:** RL reward-repair stream (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **GPU: 0.**

⛔ **THIS IS NOT A REPAIR OF `G-REWARD` AND MAY NOT BE READ AS ONE.**
The Master Mind's ruling stands and is restated here so no later reader can soften it:

> **`G-REWARD` FAILED on its committed statistic (the RATE), and is reported as FAILED.**
> The mean gap **may not** be adopted as `G-REWARD`'s statistic; if it is ever to carry a
> claim it needs its own hypothesis ID, its own bar, pre-registered, and scored on data
> not used to discover it.

What this document pre-registers is the **separate, ruled-on work item**: *"`progress`
FLIPPING SIGN with conflict is a DEFECT IN THE REWARD, not a scoring question. Fix it."*
Nothing below re-opens the gate, moves a goalpost, or re-scores `G-REWARD` as a pass.

---

## 1. The mechanism under test, stated so it can be wrong

`rewards._progress` is `along / max(v0 · H, min_ref)` — along-track displacement
normalised by **maintaining the current speed**. Its upper reach is unbounded, so a
candidate that holds `v0` while the human decelerates for a lead scores **strictly
higher**, and it scores higher *exactly where the safety question is live*.

MEASURED (banked, `…/2026-09-06-refcv4b-rl/raw/greward_rate_vs_mean.json`; per-term
weighted mean gap `hold_v0 − human`, positive = the trivial path is paid more):

| rung | all | ≤5.0 s | ≤4.0 s | ≤3.0 s | ≤2.5 s | ≤2.0 s | ≤1.5 s | ≤1.0 s |
|---|---|---|---|---|---|---|---|---|
| `progress` | −0.005600 | +0.000109 | +0.002382 | **+0.003593** | **+0.004053** | **+0.006368** | **+0.008251** | **+0.014398** |
| `headway` | −0.004818 | −0.006209 | −0.007446 | −0.011256 | −0.013454 | −0.015899 | −0.013504 | −0.014625 |
| `collision` | −0.000821 | −0.001077 | −0.001221 | −0.001602 | −0.001859 | −0.003173 | −0.005731 | −0.044944 |

⇒ **`progress` is the only term that changes sign**, and it does so monotonically as the
scene tightens.

**THE CLAIM:** that payment is what generates the rate/mean asymmetry — the human wins
**big on a minority** of windows and loses **small on a majority**, so a RATE cannot see
what a MEAN can.

---

## 2. ⛔ THE REPAIR — one variable, and its coordinate discipline

```
ref_free  = max(v0 * H, min_ref)                        # UNCHANGED
standoff  = lead_len_m + t_star * v0                    # scene scalars at t0 only
ref_lead  = lead_path[..., -1, 0] - standoff            # a POSITION query on the lead
ref_ach   = clamp(min(ref_free, ref_lead), min_ref)     # the achievable bound
progress  = min(along, ref_ach) / ref_free              # CAPPED
```

Behind a recorded context flag `progress_lead_cap` (default **True**), inert when
`lead_path` is absent.

⛔ **COORDINATE DISCIPLINE — the load-bearing constraint, kept.**
Every quantity above is a **POSITION** (`lead_path[..., -1, 0]`, the lead's own recorded
x at the horizon end) or a **scene scalar measured at t0** (`v0`, `lead_len_m`,
`t_star`). ⛔ **No closing rate appears anywhere in the RANK channel.** `M84` measured the
lead's closing rate as a clean null (**+0.0061**) against position's **+0.4145 [+0.2018,
+0.6120]** with a constant control at exactly **+0.000000**; a ranking term whose
*ordering* rested on that rate would inherit a coordinate the latent does not carry. TTC
remains a **VETO only**, untouched by this document.

⭐ **Three properties that make this a repair rather than a re-specification:**

1. **The cap is a property of the SCENE, not of the candidate** — identical for every
   candidate in a window — so it cannot rank on anything the reward may not see.
2. **It is inert without a lead**, so the no-lead sub-population is unchanged *by
   construction*, not by luck.
3. **It removes a payment; it does not add a penalty.** Going too fast is already
   punished by `headway` (−0.015899 at ≤2.0 s) and by the collision/TTC channels. A
   second penalty for the same fact would be double-dipping, and the cap deliberately
   floors at 0, not below.

⛔ **What it is NOT.** DD-v2's `EP = progress_i / max(progress_GT, progress_i)` caps at
the **human's own progress**. That is the human's SHAPE inside a RANK term and is
inadmissible here (`E-DDA-3c` §6.1: the bar may use the human's SCORE, never their
SHAPE). The cap above is derived from **the lead agent's recorded track**, which §6.1
already admits into the RANK channel.

---

## 3. ⛔ THE PREDICTION — both outcomes committed, before the code changes

### 3.1 The two statistics, and the divergence between them

Per rung `r` of the **FIXED** ladder `(∞, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0) s` — the
same ladder, the same rows, the same weights `{progress 0.3, collision 1.0, headway 0.3}`,
the same episode-cluster bootstrap (`n_boot` 4,000):

* `rate(r)` = P(hold_v0 composed ≥ human composed) — **`G-REWARD`'s committed statistic**;
* `mean(r)` = mean(hold_v0 composed − human composed);
* ⭐ **`DIV(r) = 1` iff `rate_ci_lo(r) > 0.30` AND `mean_ci_hi(r) < 0`** — the rate says
  FAIL with its whole interval above the ceiling **while** the mean says the human wins
  with its whole interval below zero. **MEASURED BASELINE: `Σ DIV = 6 of 8 rungs`**
  (all · ≤5.0 · ≤4.0 · ≤3.0 · ≤2.5 · ≤2.0; not ≤1.5, not ≤1.0).
* ⭐ **`SKEWSPLIT(r) = median(gap) − mean(gap)`** — the asymmetry in ONE number, with no
  ceiling in it. Positive means the typical window disagrees with the average window,
  which is precisely "frequent small losses, rare large wins".
  **MEASURED BASELINE** (`hold_v0 − human`, track lead, 6,089 windows / 73 episodes):

  | rung | all | ≤5.0 | ≤4.0 | ≤3.0 | ≤2.5 | ≤2.0 | ≤1.5 | ≤1.0 |
  |---|---|---|---|---|---|---|---|---|
  | mean | −0.011239 | −0.007177 | −0.006285 | −0.009265 | −0.011259 | −0.012704 | −0.010984 | −0.045171 |
  | median | −0.002404 | −0.001666 | −0.000561 | −0.000345 | −0.000293 | +0.000969 | +0.000273 | +0.000004 |
  | **SKEWSPLIT** | **0.008836** | 0.005511 | 0.005724 | 0.008920 | 0.010966 | **0.013674** | 0.011257 | 0.045175 |

  Positive at **8 of 8** rungs and rising with conflict through ≤2.0 s.

### 3.2 ⛔ P1-PRED-A — the term (NECESSARY, not sufficient)

> After the repair, `progress`'s weighted mean gap is **NOT POSITIVE** at the three
> conflict rungs ≤3.0 s / ≤2.0 s / ≤1.5 s: each value ≤ 0, **or** its episode-cluster CI
> straddles 0. The monotone sign flip in §1 is gone.

### 3.3 ⭐⭐ P1-PRED-B — the falsifiable CONSEQUENCE (this is the actual test)

> **The rate/mean divergence SHRINKS on the same rungs.** Both of:
>
> * **B1.** `Σ_r DIV(r)` **falls below 6** — at least one rung stops disagreeing; and
> * **B2.** `SKEWSPLIT(r)` **falls in magnitude at the conflict rungs** ≤3.0 / ≤2.5 /
>   ≤2.0 s — at least 2 of those 3 strictly smaller than the baseline row above.
>
> ⭐ And the direction is committed too: `rate(≤2.0 s)` must **fall** from **0.5482**.

### 3.4 ⛔ THE OTHER OUTCOME, written now and reported as written

* **A holds and B FAILS** ⇒ ⛔ **`progress` WAS NOT THE MECHANISM.** The sign flip was a
  symptom, not the generator, and I say so. **The next candidate is named in advance:
  `headway`** — it carries the LARGEST per-term magnitude at every conflict rung
  (−0.015899 at ≤2.0 s, 2.5× `progress`), and its statistic `tg = amin over steps` is
  itself a **min-over-N order statistic**, the same family of defect as `fan_floor@k`
  read alone. The cheapest next experiment is to re-read `headway` at a low **quantile**
  of the per-step time gap instead of the single worst step, and that experiment is
  0 GPU on these same banked rows plus geometry.
* **A FAILS** ⇒ the implementation does not do what it was built to do. Report the
  repair as FAILED with the measured per-term table; do not tune the cap and re-run.
* ⛔ **Neither outcome re-opens `G-REWARD`.** Whatever happens to the rate here is a
  DIAGNOSTIC of the reward's geometry, never a re-scored gate.

---

## 4. ⛔ CONTROLS THAT MUST READ KNOWN VALUES

The 2026-08-22 rule (four probe failures in one afternoon, each caught only by a control
that had to read a known value) applies in full.

1. ⭐ **CHANNEL — cap OFF must reproduce the banked panel EXACTLY.** Re-running the
   humanflag build with `progress_lead_cap=False` must return the banked all-window rate
   **0.441780259484316** and mean gap **−0.011239379601720929** to ≤ 1e-12. If it does
   not, the regeneration is wrong and every other number here is void.
2. ⭐ **NO-LEAD / NON-BINDING SUB-POPULATION — EXACTLY unchanged.** With the cap ON, every
   window in which `ref_lead ≥ ref_free` must reproduce its cap-OFF `progress` to
   ≤ 1e-12, for **both** sides. Reported as a count and a max abs error.
3. ⛔ **DEGENERACY GUARD — the term must still RANK.** A cap that makes `progress`
   identical for every candidate would move the rate for the wrong reason. Report the
   mean spread `|progress_hold − progress_human|` per rung before and after
   (**baseline: 0.088491 all-windows, 0.060303 at ≤2.0 s**) and the fraction of windows
   where it is exactly 0. ⛔ **A collapse of the spread to ~0 is reported as a DEGENERATE
   REPAIR and NOT as a pass, however the rate moves.**
4. ⭐ **ARITHMETIC — `hold_v0`'s progress is analytically reproducible from `v0` alone**
   with the cap OFF: `min(1, v0·H / max(v0·H, min_ref))`. **MEASURED before the change:
   max abs err 2.384e-08** (float32 storage). Re-run after.
5. **`frozen` must stay far below the human** at every rung, cap on or off
   (baseline all-window **0.045492**).

---

## 5. Estimator, tier, evidence class

* **Estimator:** episode-cluster bootstrap over episodes, `n_boot` 4,000, α 0.05
  (`greward_power_curve.py`'s own function, unchanged).
  ⛔ **It answers ONE question: "would another draw of EPISODES say this?"** — not
  another training run (`H-ESTIM-SEED-1`) and not another inference run. **No arm is
  trained and no planner samples here**, so training and inference variance do not enter;
  this is arithmetic over re-computed geometry on fixed rows.
* **Tier:** **T0** instrument probe, NON-PARITY RL-fit windows. ⛔ Not a driving number.
* **Evidence class:** MEASURED (ours), artifacts under `raw/` in this package.
* ⛔ **Four families:** this document scores a REWARD TERM, not a policy. It is not an
  eval and produces no capability claim; no arm's four-family table is asserted or
  implied by it.

## 6. What is NOT changed by this document

`headway`, `collision`, `feasibility`, `comfort`, `robust_contact`, the TTC veto, the
`≥ GT` bar, `DEFAULT_WEIGHTS`, the arms table, and every banked result. The cap defaults
ON but is **inert without a lead**, and `progress_lead_cap=False` restores the exact
pre-repair term for reproduction of any banked number.
