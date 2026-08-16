# The collapse gate cannot detect collapse — how far short it falls, and the estimator that closes it

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **Compute** CPU-only (Thor is
training the live v6F S-W run; no GPU work, no ssh to Thor).

**The PI asked whether SigReg is doing its job.** The honest answer is at the bottom of this
document and it is not the one anybody wants: **the instrument could not have told us** — and
below is the number proving it could not.

---

## 0. The answer first

⭐ **`effective_rank` ≈ 15 is a CEILING ARTIFACT, not collapse.** MEASURED: pooling the *same
healthy synthetic population* from 48 rows to 1536 rows moves the reading **14.02 → 121.57**.
The representation did not change. Only the estimator's ability to see it did.

⛔ **The gate criterion it feeds — `>= 0.8x effective rank across phases` — fires when nothing
happened between 11 % and 38 % of the time, with power 0.145 against the collapse it exists to
catch.** A guard that goes off on noise, scheduled to decide whether S-W passes.

⛔ **Do not convert that into "SigReg is working".** *Not collapse* and *SigReg prevented
collapse* are different claims. The second needs a λ=0 ablation, which the loss-determinism
fix landed today has only just made runnable. See §7.

---

## 1. What the reading actually measures — the geometry, read off the running job

MEASURED (ours), from the argv of the running trainer captured in
`TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-08-15-v6-thor-resume/code/RESTART_v6F_SW.sh`
(generated from `/proc/25477` while it was live):

| flag | value | consequence |
|---|---|---|
| `--batch` | 8 | |
| `--window` | 6 | 8 × 6 = **48 rows** per spectrum call |
| `--eps-per-batch` | 4 | those 48 rows come from **4 episodes** |
| `--readout-grid 4 --readout-dim 128` | | `d_op` = 4 × 4 × 128 = **2048** |
| `--spectrum-every` | 200 | 38 records by step 7600 ✓ |

The monitor computes `spectrum_report(z_op_win.reshape(-1, d_op))` — the SVD of the **centred**
48 × 2048 matrix (`stack/tanitad/models/v6.py`, emitted at `stack/scripts/train_v6_staged.py`).

⛔ **A centred covariance built from n rows has rank ≤ n−1.** At n = 48 the reading is bounded
by **47**, whatever the representation does. So the banked mean of **15.13 is 32.2 % of the
achievable 47**, not 0.74 % of 2048. Reading it against `d` is wrong by construction, and that
misreading is the original defect: the record did not carry its own ceiling, so nothing stopped
it.

⚠️ **And the 48 rows are not 48 independent draws.** They are 6 consecutive frames per window
(near-duplicates) across 8 windows drawn from 4 episodes. The estimator's variance is set by the
number of independent **clusters**, which is ~4–8, not by the row count.

⭐ **The clustering is deliberate, and its trade was accepted for a different consumer.** Both
sampler paths (`InteractionSampler`, and `make_sampler` on the O4 control arm) group few
episodes × many windows on purpose — `train_v58f_unicycle_head.py:353` states it verbatim: *"The
mild within-batch correlation is an accepted, stated trade"*, taken to cut MooseFS cold payload
loads ~8×. That trade is fine for a **loss**, which only needs an unbiased gradient. It is not
fine for a **rank estimator**, which needs independent directions. ⇒ **The I/O optimisation was
inherited by an instrument that cannot survive it, and nothing in between checked.** That is the
mechanism, not a coincidence.

---

## 2. The power deficit, quantified

MEASURED (ours) — `code/sigreg_gate_power.py`, artifact `raw/sigreg_gate_power.json`,
CPU-only, seeded, torch 2.11.0+cu128. The generative model is stated in the script: a power-law
population spectrum over d = 2048, sampled through the **real nested correlation structure**
(episode factor → window factor → frame), with the sampler's own geometry (4 episodes,
8 windows, 6 frames).

The fast Gram path is verified against `torch.linalg.svdvals` on every shape used, max relative
difference **2.08 × 10⁻⁹** (`meta.fast_path_verification`) — the speedup is checked, not assumed.

### 2.1 ⛔ The estimator COMPRESSES: 7.3× of true collapse becomes 2.06× of reading

n = 48, d = 2048, iid rows (the *most favourable* case — real correlation only makes it worse):

| true effective rank | reading (mean ± sd) | reading / ceiling |
|---:|---:|---:|
| 2048.00 (isotropic) | **46.861** ± 0.006 | 0.997 |
| 1961.40 | 46.724 ± 0.016 | 0.994 |
| 1587.45 | 43.995 ± 0.257 | 0.936 |
| 881.72 | 34.296 ± 0.733 | 0.730 |
| 281.36 | **22.755** ± 0.949 | 0.484 |
| 19.91 | 8.995 ± 0.570 | 0.191 |
| 5.12 | 4.290 ± 0.274 | 0.091 |

⭐ **A 7.3× true collapse (2048 → 281) moves the reading only 2.06× (46.86 → 22.76).** The
reading is still *monotone* in the truth — it is not blind — but the mapping is compressed by
roughly the log of the ratio, which is what an entropy statistic evaluated near a hard ceiling
does. Above true rank ≈ 900 the compression is severe: the entire span 882 → 2048 (2.3× of
truth) occupies **34.3 → 46.9**, a 1.36× band.

⚠️ **And these are iid rows — the favourable case.** With the real nested correlation the same
population reads **14.01 ± 1.67**, i.e. the sd rises from 0.9 on a reading of 22.8 (CV 0.042) to
1.67 on a reading of 14.0 (**CV 0.119, ~2.9× worse**). Compression plus that variance is what
makes the criterion unusable; neither alone would.

### 2.2 The false-positive rate of `>= 0.8x`, bracketed

The criterion compares two readings. Its false-positive rate is P(ratio < 0.8) when **nothing
changed**. Two independent estimates, deliberately chosen to bracket:

| estimate | basis | FP at `0.8×` | threshold that WOULD give 5 % FP |
|---|---|---:|---:|
| **lower bound** | model-based null, calibrated to the banked mean (ρ_ep 0.5, ρ_win 0.4, α 2.0) | **11.00 %** | **0.742** |
| **upper bound** | the run's OWN banked spread, lognormal fit (σ_log 0.5106, CV 0.546) | **37.87 %** | **0.305** |

⇒ **The criterion fires on nothing between 11 % and 38 % of the time.** The nominal rate was
never stated anywhere, which is itself the finding: nobody chose 0.8, so nobody could have
chosen it wrongly — it was inherited.

The lower bound is a *lower* bound because the calibrated simulation's spread (9.18 → 18.23 over
600 draws) is **narrower than the banked 3.37 → 30.06**; no simulated regime in the grid
reproduced a range that wide (`calibration[*].covers_banked_range` is `false` for all 28). The
upper bound is an *upper* bound because it charges the whole banked spread to noise; the brief's
three time points (step 200 → 16.75, 4000 → 12.10, 7600 → 17.59) show no trend, so little of it
is drift, but "little" is not "none".

⚠️ To have a 5 % false-positive rate on the run's actual spread, the threshold would have to be
**0.305×** — the criterion could only fire on a **3.28× drop in the reading**, which §2.1 maps
to essentially total collapse.

### 2.3 The effect size it can actually detect

Calibrated regime, `>= 0.8x` applied to single readings (so FP = 11.00 %):

| true collapse | true rank ratio | reading ratio | **power** |
|---|---:|---:|---:|
| 1.43× (mild) | 0.700 | 0.973 | **0.145** |
| 3.35× | 0.299 | 0.850 | **0.375** |
| 10.5× | 0.095 | 0.627 | 0.933 |
| 13.1× | 0.077 | 0.466 | 0.998 |

⭐ **Against a 1.43× true collapse the criterion has power 0.145 while firing on nothing 11.0 %
of the time — a likelihood ratio of 1.32.** That is not a weak test; it is very nearly no test
at all. 80 % power arrives somewhere between a **3.4× and a 10.5× true collapse**.

⛔ **This is C13's family inverted.** C13/C14 are instruments structurally unable to report the
answer they are cited for — a guard that cannot fail. This is a guard that **fires when nothing
happened**, and it is scheduled to decide whether S-W passes.

### 2.4 Where the noise comes from — and where it does not

The loss-determinism stream landed an opt-in `sigreg_generator` and hypothesised that part of
the `effective_rank` spread was SigReg resampling its slice directions. **That premise is
refuted**, structurally and by measurement.

*Structural:* `spectrum_report` takes one argument — the latent tensor — and returns the SVD of
its centred rows. SigReg's directions live inside `o6_sigreg_loss` / `SigReg._forward_fp32` and
never enter the statistic.

*MEASURED* (`code/sigreg_slice_vs_batch.py`, `raw/sigreg_slice_vs_batch.json`; real `V6Stack`
through the real `encode_window` → readout path, 24 reps of 48 rows):

| contrast | `effective_rank` | `o6` loss |
|---|---:|---:|
| vary ONLY the SigReg generator, batch fixed | sd **0.000000**, range **0.000000** | sd 0.019896, range 0.083679 |
| vary the BATCH, generator fixed | sd 0.045668, range 0.159196 | sd 0.040259, range 0.146155 |
| **share of `effective_rank` variance from slice directions** | **0.0 (exactly)** | — |

⚠️ **The exactly-zero needs its companion row or it should not be believed.** A variance of
0.000000 is indistinguishable from "the knob was never connected". The second column is the
negative control for the negative control: **the same contrast moves the `o6` loss** (sd
0.019896, range 0.083679), so the generator demonstrably does something — and still moves the
spectrum reading by nothing.

⇒ The 3.37 → 30.06 spread is **batch composition** at n = 48 against a ceiling of 47. Fixing the
SigReg generator makes the loss reproducible and does not touch this at all.

---

## 3. The estimator that closes it

`stack/tanitad/models/v6.py` — three additions, all **default-off or purely additive**, because
v6F S-W is training from this file with ~6.8 days to run.

### 3.1 The ceiling is stamped in the record (unconditional)

`spectrum_report` now always emits:

```
"rank_ceiling":       min(n-1, d),
"effective_rank_frac": effective_rank / rank_ceiling,
"rank_admissible":     rank_ceiling >= O6_ADMISSIBLE_CEILING,
"ceiling_note":       "a centred covariance from n=48 rows has rank <= 47;
                       effective_rank is bounded by that, NOT by d=2048"
```

⛔ This is the fix for the original sin. No record can now be quoted without the bound that
constrains it, and `15 of 2048` is unwritable from the artifact.

### 3.2 `SpectrumAccumulator` — pool CONSECUTIVE steps, not consecutive calls

A bounded ring of raw rows. `--spectrum-accum N` (**default 1 = off, the incumbent path**) pools
the N steps immediately preceding each emission.

⚠️ **Consecutive steps, not consecutive spectrum calls.** Pooling 32 *calls* at
`--spectrum-every 200` would span 6 400 steps, over which the representation genuinely moves —
the pooled spectrum would then measure the *union over training* and read high for the wrong
reason. 32 consecutive **steps** span 32 steps ≈ 14 min of Thor wall-clock at 26.35 s/step, where
drift is negligible, and they draw ~32 × 4 = 128 distinct episodes instead of 4.

MEASURED, same population, three pool sizes:

| pool | rows | ceiling | reading | CV | FP @ 0.8× | 5 %-FP threshold | reading when collapsed (keep 16, floor 0.01) | separation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (incumbent) | 48 | 47 | 14.02 | 0.1204 | 0.0767 | 0.768 | 8.52 | 1.65× |
| 8 | 384 | 383 | 53.21 | 0.0466 | 0.000 | 0.919 | 14.82 | 3.59× |
| **32 (recommended)** | 1536 | 1535 | **121.57** | **0.0198** | 0.000 | **0.960** | 19.41 | **6.26×** |

⭐ **14.02 → 121.57 on an UNCHANGED population.** That single row is the whole argument.

**Cost, stated:** ring of 32 × 48 × 2048 float32 on CPU = **12.6 MB**; per pushed step one
detach + a 393 KB D2H copy against a 26.35 s step; one extra spectrum of 1536×2048 every 200 steps
(MEASURED 0.291 s against 5 270 s of training — **0.006 %**; with the interval on,
28.28 s = **0.54 %**, dev-box CPU at 6 threads). Raising `--batch` instead would cost GPU
memory linearly and buy far less, because it does not raise the episode count.

### 3.3 An interval, with the CLUSTER as the resampling unit

`spectrum_report(..., ci_reps=N, block=W, generator=g)` adds an interval whose unit is a **block
of `window` consecutive rows** — the correct unit when rows are consecutive frames. Treating the
6 near-duplicate frames as 6 independent facts is the `overlapping_holdout_se` error class in a
new costume.

⛔ **The interval is a leave-one-CLUSTER-out JACKKNIFE, not a bootstrap — and that is a measured
choice, not a preference.** §4 gives the coverage: the bootstrap variants cover **0.00** at 384
rows, the jackknife **0.867**. The bootstrap bounds are still emitted under
`bootstrap_DIAGNOSTIC_do_not_quote`, carrying the reason and their coverage, so the departure
from the programme's bootstrap doctrine is visible in every record rather than buried here.

⚠️ Neither path draws from the global RNG: the default draws nothing at all and the diagnostic
bootstrap draws only from the generator it is handed — pinned by
`test_default_spectrum_report_consumes_NO_global_rng`, so switching the interval on cannot move
the run's loss (the failure the SigReg determinism fix just closed, on the other side of the
seam).

⚠️ **The interval is the expensive part**, MEASURED on the dev box (6 threads; Thor's CPU may
differ) because the jackknife costs one eigendecomposition per cluster:

| rows | clusters | plain reading | with interval | share of a 200-step interval @ 26.35 s/step |
|---:|---:|---:|---:|---:|
| 48 | 8 | 0.005 s | 0.009 s | 0.0002 % |
| 384 | 64 | 0.037 s | 0.390 s | 0.007 % |
| 1536 | 256 | 0.291 s | **28.28 s** | **0.54 %** |

### 3.4 The four options, and why this combination

| option | verdict | evidence |
|---|---|---|
| **Raise `n` per call** (bigger `--batch`) | ⛔ **rejected** | it costs GPU memory *linearly* on a run that is 6.8 days from done, and it does **not** raise the cluster count — `--eps-per-batch 4` caps the episodes regardless of batch size, so the added rows are more near-duplicates. The variance is set by clusters (§1). Reaching ceiling 1024 this way needs `--batch 171` at `--window 6`. |
| **Accumulate a running d×d covariance** | ⛔ **rejected on a specific ground** | it works for the point estimate (2048² fp32 = 16.8 MB, comparable to the ring's 12.6 MB) but it **destroys the raw rows**, and every candidate interval here resamples *clusters of rows*. A running covariance cannot support a cluster interval at all — it would force back exactly the "point estimate with no interval" the new criterion refuses. |
| **Ring of raw rows over consecutive steps** | ✅ **adopted** | 12.6 MB CPU at capacity 32; keeps the rows, so the cluster interval is available; raises ceiling **47 → 1535** and clusters **~4 → ~128**. |
| **Bootstrap CI on `effective_rank`** | ⚠️ **adopted in intent, replaced in method** | the interval is required, but the *bootstrap* does not cover (§4). Shipped as a leave-one-cluster-out jackknife. |
| **Rank RATIO against a fixed reference** | ✅ **adopted, as clause 2** | with the phase-start pooled reading as reference — plus clause 3, because a ratio alone cannot see a phase that *started* collapsed. |

---

## 4. Does the interval cover? — the bootstrap does NOT, and that changed the design

⛔ **A CI nobody has checked is a decoration.** Three candidate intervals were built from the
same cluster resamples and their coverage MEASURED against the estimand that is actually
identified at finite n — **E[ER̂ | n]**, the finite-n expectation. (The *population* effective
rank is **not** identified at n = 48; an interval claiming to cover it would be a lie, which is
the whole reason the ceiling is stamped in the record.)

60 independent datasets per cell, calibrated regime, nominal **0.95**:

| interval | coverage @ 48 rows | coverage @ 384 rows | mean width @ 384 |
|---|---:|---:|---:|
| percentile cluster bootstrap | 0.250 | **0.000** | 8.108 |
| pivotal (basic) cluster bootstrap | 0.300 | **0.000** | 8.108 |
| **leave-one-cluster-out jackknife** | **0.850** | **0.867** | 7.741 |

**Why both bootstraps fail.** Resampling blocks *with replacement* duplicates them — about
36.8 % of blocks are absent from any given resample and their slots are filled by duplicates.
Duplicated rows are **exactly rank-deficient**, so every replicate has a lower rank than the
original sample. For a rank functional that is a systematic downward bias, not sampling noise.
At pool 32 the point estimate was **127.47** with a percentile interval of **[90.83, 102.86]** —
entirely below the estimate.

⚠️ **The pivotal correction does not rescue it.** It has the identical width and merely reflects
a wrong interval to the other side of the point estimate (coverage 0.300 → 0.000). This is why
it was measured rather than reasoned about: the pivotal bootstrap is the textbook fix for
bootstrap bias and it is *worse than useless* here.

⇒ **Shipped: the leave-one-cluster-out jackknife.** The bootstrap bounds are still emitted, under
the key `bootstrap_DIAGNOSTIC_do_not_quote`, carrying the reason and their measured coverage.

⚠️ **This is a deliberate carve-out from the programme rule that decision-grade intervals are the
paired episode-cluster bootstrap** (`CLAUDE.md`, `taniteval/ci.py`). That rule is right for
mean-like eval metrics; it fails for a **rank** estimand, and the failure is measured above, not
argued. **It does not generalise** — nothing here licenses replacing the bootstrap anywhere else,
and any other estimand needs its own coverage check before it may follow this precedent.

⚠️ **0.85–0.87 against a nominal 0.95 is mildly ANTI-CONSERVATIVE.** The jackknife interval is a
working uncertainty, not an exact 95 % guarantee, and the verdict in §5 is built to fail only
when the interval *excludes* the threshold — which is the conservative direction for a
too-narrow interval to be wrong in. Widening it is a follow-up, not a blocker.

---

## 5. The re-derived gate criterion — PRE-REGISTERED, both outcomes committed

**Replaced:** `"O6_rank_retention": ">= 0.8x effective rank across phases"` — no estimator, no
`n`, no interval, and (§2.2) an 11–38 % false-positive rate with 0.145 power against the collapse
it exists to catch.

**Replacement**, owned by `tanitad.models.v6.o6_rank_verdict` and wired into
`STAGE_GATE_SPEC["S-W"]["criteria"]`:

> **(1) ADMISSIBILITY.** A reading whose `rank_ceiling` < **1024** is **INCONCLUSIVE** — never
> PASS, never FAIL. Every single-batch reading is this case.
> **(2) RETENTION.** FAIL only when the cluster-JACKKNIFE interval on `ER_cur / ER_ref` lies
> **wholly below 0.8×**; PASS only when it lies wholly at or above; otherwise INCONCLUSIVE.
> **(3) FLOOR.** FAIL when the pooled `effective_rank` < **64** regardless of retention.

**Why each number.**

- **1024** — the smallest power of two clearing the absolute floor by 16×, reached by
  `--spectrum-accum 22` or more (22 × 48 − 1 = 1055); the recommended 32 gives 1535.
- **0.8×** is *kept*, deliberately. It was never the problem: at pool 32 its measured
  false-positive rate is **0.000** and the 5 %-FP threshold has risen to **0.960**, so the
  threshold is now far more conservative than the noise requires rather than far more
  aggressive. Changing the number and the estimator at once would make the change
  non-attributable.
- **64** — MEASURED at pool 32 (ceiling 1535): a healthy α = 2 population reads **121.6**, the
  same population collapsed to 16 retained directions reads **19.4**. 64 sits between them with
  ~2× margin on each side, and is 8× the `top_k` the energy share is taken over.

⛔ **The interval must exclude the threshold for the guard to fire.** That is what makes the new
criterion unable to fire on noise, and it is why INCONCLUSIVE had to become a possible answer.

### 5.1 Pre-registration — what each outcome means, decided in advance

| outcome at the S-W gate | reading | what we do |
|---|---|---|
| **PASS** (retention CI wholly ≥ 0.8×, `ER ≥ 64`, ceiling ≥ 1024) | the operative latent kept its rank across the phase | S-W's O6 row is a genuine pass; SigReg is *consistent with* preventing collapse, and the λ=0 ablation (§7) upgrades that to *caused* |
| **FAIL on clause 2** | rank dropped, confidently | S-W does not propagate. Investigate SigReg weight, `--sigreg-free-dims`, and the O5 rollout term |
| **FAIL on clause 3** | rank was already below floor | the phase started collapsed — retention would have been a *pass*, which is exactly why clause 3 exists |
| **INCONCLUSIVE on clause 1** | ceiling < 1024 | the run was not pooled. **Not a pass.** Re-run the monitor with `--spectrum-accum 32`; this is a 0-GPU fix |
| **INCONCLUSIVE on clause 2** | CI straddles 0.8× | pool more steps or take more reference readings. **Not a pass** |

⚠️ **INCONCLUSIVE is not a pass** — the existing `assert_stage_precondition` already refuses to
propagate an inconclusive gate without an explicit `--allow-inconclusive-gate` and a written
reason, so the new verdict lands on machinery that already treats it correctly.

---

## 6. ⭐ The guard has been WATCHED FAIL — and watched not fire

A guard nobody has seen fire is a hypothesis. `stack/tests/test_o6_spectrum_power.py`
(**19 tests, all passing**) makes it an executable fact:

| test | what it proves |
|---|---|
| `test_guard_FIRES_on_a_synthetically_collapsed_representation` | a population squeezed onto 8 of 256 directions ⇒ verdict **FAIL**, for a *named* clause |
| ⭐ `test_guard_FIRES_on_clause_2_alone_with_the_floor_disarmed` | with the floor set to 1.0 so it cannot trip, a 3× squeeze still FAILS **from the interval**: retention 0.341, CI [0.336, 0.345], wholly below 0.8 |
| `test_guard_DOES_NOT_fire_on_a_healthy_representation` | the same population twice ⇒ a real **PASS** (retention 0.999, CI [0.988, 1.010]), not merely the absence of a FAIL |
| `test_guard_fires_on_the_absolute_floor_without_any_reference` | clause 3 catches an already-collapsed phase start, which retention cannot |
| `test_guard_refuses_a_point_ratio_with_no_interval` | a bare ratio ⇒ **INCONCLUSIVE**, not a verdict |
| ⭐ `test_a_true_collapse_THROUGH_the_0_8_threshold_reads_as_no_change` | a population whose **true** rank ratio is *below* 0.8 reads **> 0.9** at n = 48, and **0 of 24** pairings fire — the power deficit, executable |
| ⭐ `test_pooling_separates_what_n48_could_not` | the same collapse becomes fully separable once pooled (bands do not touch) |
| `test_at_n48_the_verdict_is_INCONCLUSIVE_by_construction` | the live run's own reading can neither pass nor fail, and now says so |
| `test_record_stamps_its_own_rank_ceiling` | `15 of 2048` is unwritable from the record |
| `test_default_spectrum_report_is_UNCHANGED_vs_the_pre_change_revision` | **CONTENT-anchored** no-change proof (§8) |
| `test_default_spectrum_report_consumes_NO_global_rng` | the CI cannot perturb the live run's loss |
| `test_interval_is_reported_with_its_kind_and_unit` | the interval names its estimator and its unit, and brackets the point estimate |
| `test_the_bootstrap_is_kept_ONLY_as_a_labelled_diagnostic` | the bootstrap's downward bias is pinned — if it ever stops being biased down, the reason in the record is wrong and must be re-derived |
| `test_the_pooling_window_ends_AT_the_emission` | the pooled block is exactly the `accum` consecutive steps ending at the read — an off-by-one here would change what is measured while the record still looked well-formed |

The collapse fixture is a **squeeze** (`scale[keep:] *= floor`), not a hard truncation — a hard
truncation is the easiest possible thing to detect and would flatter the instrument.

⚠️ The pre-existing `test_o6_spectrum_monitor_detects_collapse` in `tests/test_v6_staged.py`
exercises the estimator at **n = 64, d = 16** — the *well-conditioned* regime, which is exactly
not the regime it runs in. It is left untouched (it is a correct test of a different thing), but
it is why the guard looked proven and was not.

---

## 7. ⚠️ Is SigReg preventing collapse on the live run?

**CANNOT BE DETERMINED AT THIS n.** Stated plainly because manufacturing a verdict here is the
failure this whole document is about.

What the evidence supports, exactly:

1. ⛔ **`o6_sigreg` being flat is not collapse evidence in either direction.** It is the
   regulariser's loss value; a regulariser's loss can fall because it is satisfied *or* because
   the representation degenerated.
2. ⭐ **`effective_rank` ≈ 15 is NOT evidence of collapse.** At n = 48 a *healthy* population
   reads ~14; the same population pooled to 1536 rows reads 121.6. The reading was pinned by the
   sampler, not by the representation.
3. **There is no collapse trend in the banked series** (200 → 16.75, 4000 → 12.10,
   7600 → 17.59) — but at CV ≈ 0.55 a series this noisy could hide a substantial monotone drift,
   so *absence of trend* is weak evidence, not reassurance.
4. ⛔ **"Not collapsed" ≠ "SigReg is working".** Even a fully resolved, healthy pooled spectrum
   would only show the representation is fine — not that **SigReg** is why. That requires a
   `--w-o6 0` ablation against the same seed and schedule. **The `sigreg_generator` fix landed
   today is what makes that ablation clean**, because previously the O6 term was the one
   non-reproducible part of the loss (S-W 3.379698 vs 3.384279, the entire discrepancy in `o6`).

**What would settle it, in priority order and all 0-GPU-blocked-on-nothing except the last:**

1. Re-run the monitor with `--spectrum-accum 32 --spectrum-ci-reps 32` — lifts the ceiling from
   47 to 1535. MEASURED cost 0.54 % of step time (§3.3). ⚠️ **This does NOT require restarting
   the live run**, and should not: the flags take effect at the next natural restart, or the same
   estimator can be run offline over banked activations. Do not kill a job 6.8 days from done for
   a monitor.
2. Pull the 38 raw records off Thor into the repo. They are **not banked anywhere in this
   repo** — I could not find them at any path, and the summary statistics in this document are
   therefore INHERITED, not MEASURED by me. That is a stranding in the sense of rule 3.
3. The `--w-o6 0` ablation, now that `o6` is reproducible.

---

## 8. Default behaviour is unchanged — CONTENT-anchored (C75)

`test_default_spectrum_report_is_UNCHANGED_vs_the_pre_change_revision` walks `v6.py`'s **own
history** for the newest revision that does not yet contain `O6_ADMISSIBLE_CEILING`, imports it
side by side, and asserts every pre-existing key is **bit-equal** on identical input across four
shapes including the live 48 × 2048. The additions are asserted to be exactly
`{rank_ceiling, effective_rank_frac, rank_admissible, ceiling_note}`.

⚠️ **Not `HEAD`.** HEAD moves under a working file; a HEAD comparison then compares the module
with itself and passes by construction. The content anchor is stable no matter how many commits
land, and it is the semantically right reference: it *is* the code the live checkpoint was built
from. (Pattern taken from `stack/tests/test_v6_factored_goal.py`.)

Behavioural surface:

| change | default | effect on the live run |
|---|---|---|
| `rank_ceiling` / `frac` / `admissible` / `note` in the record | always on | **record content only**; no tensor, no loss, no `state_dict` |
| `--spectrum-accum` | **1** | accumulator is `None`; emission path identical |
| `--spectrum-ci-reps` | **0** | no interval, no RNG draw |
| `o6_rank_verdict` in the gate artifact | always on | reports **INCONCLUSIVE** for the current n; `O6_spectrum` was and remains `"reported"`, never `required`, so no gate verdict moves |
| `STAGE_GATE_SPEC["S-W"]["criteria"]` text | — | criteria strings are reported, never evaluated by the trainer |

No model class, config field, parameter or `state_dict` key was touched.

---

## 9. Retraction-class note

**Class: a probe that reports the wrong scope, read as an answer.** Identical in shape to `df`
reporting the 965 TB cluster instead of the pod quota, `free`/`tegrastats` on Thor's unified
memory, and cgroup v1 `usage_in_bytes` counting reclaimable page cache. Here the statistic was
`effective_rank` at n = 48 read against d = 2048 — **a number bounded by 47 quoted as a fraction
of 2048**.

The durable fix is the same one that worked for `step_s`: **the record carries its own
definition**. `rank_ceiling` and `ceiling_note` are emitted unconditionally so the misreading is
not available.

**Second class, distinct and worth logging separately: a threshold nobody chose.** `0.8×` has no
stated nominal false-positive rate anywhere in the programme. It was inherited, and an inherited
threshold cannot be wrong — which is why it survived. ⇒ **Any gate threshold must ship with the
false-positive rate it achieves on the estimator that will actually be used.**

Both are appended to `Project Steering/RETRACTION_LOG.md` as **C76 — a gate threshold nobody
chose, on an estimator nobody sized**, together with three sub-lessons that generalise beyond
this instrument: an accepted batching trade must list the *other* consumers of the same tensor
before they inherit it; every null result carries the positive control proving the manipulation
was live; and an interval must have its coverage measured before it is shipped.

---

## 10. Deliverable manifest

| artifact | path (repo-relative) | state |
|---|---|---|
| this document | `TanitAD Research Hub/…/incoming/2026-08-16-sigreg-gate-power/SIGREG_GATE_POWER.md` | STAGED |
| power simulation | `…/2026-08-16-sigreg-gate-power/code/sigreg_gate_power.py` | STAGED |
| simulation artifact | `…/2026-08-16-sigreg-gate-power/raw/sigreg_gate_power.json` | STAGED |
| slice-vs-batch decomposition | `…/2026-08-16-sigreg-gate-power/code/sigreg_slice_vs_batch.py` | STAGED |
| its artifact | `…/2026-08-16-sigreg-gate-power/raw/sigreg_slice_vs_batch.json` | STAGED |
| estimator (`rank_ceiling`, `SpectrumAccumulator`, `o6_rank_verdict`, cluster jackknife) | `stack/tanitad/models/v6.py` | STAGED |
| trainer wiring + re-derived criterion | `stack/scripts/train_v6_staged.py` | STAGED |
| the guard proved to fire | `stack/tests/test_o6_spectrum_power.py` | STAGED |
| full suite evidence — **3346 passed / 0 failed / 17 skipped / 2 xfailed** | `…/2026-08-16-sigreg-gate-power/raw/stack_pytest.txt` | STAGED |
| **C76** retraction class (appended) | `Project Steering/RETRACTION_LOG.md` | STAGED |

⚠️ The brief's baseline was **3282 passed** at `655ce40`. The final count is **3346** — my 19
tests plus 45 that landed from sibling streams on this branch while I worked (`test_loss_determinism.py`
among them). **0 failed** either way.

**Nothing is on a pod or in a worktree.** No commit, no push.

### Escalations — these need a decision, not a doc

1. ⛔ **The 38 spectrum records are not in this repo.** Every summary statistic about the live
   run in this document is INHERITED. Someone with Thor access should pull
   `/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl` into
   `…/2026-08-16-sigreg-gate-power/raw/` — then §2.2's upper bound becomes a measurement rather
   than a lognormal fit.
2. ⚠️ **The S-W gate is scheduled to be decided by O6 as currently configured.** With
   `--spectrum-accum` unset it will return **INCONCLUSIVE**, which is correct and is *not a
   pass*. Whoever runs that gate should either pool, or record the inconclusive verdict
   deliberately.
3. **The λ=0 SigReg ablation is now runnable** and is the only thing that answers the PI's
   question as asked. It is GPU work and belongs in the queue behind the live run.
4. ⚠️ **`stack/tanitad/models/sigreg.py` and the four `sigreg_generator` hunks in
   `train_v6_staged.py` belong to the loss-determinism stream and were NOT touched.** My edits to
   `train_v6_staged.py` are confined to the spectrum/gate path (`STAGE_GATE_SPEC` criteria text,
   `in_spectrum_window`, the `run_stage_gate` O6 probe, the emission block, two new flags) and do
   not overlap their hunks. Nothing here needs a change inside their seam.
5. ⚠️ **The jackknife carve-out (§4) touches programme doctrine.** `CLAUDE.md` names the paired
   episode-cluster bootstrap as *the* decision-grade interval. This document does not amend that
   rule — it records a measured exception for **rank estimands only**. If anyone wants the
   exception generalised, that is a PI decision and needs its own coverage study.
