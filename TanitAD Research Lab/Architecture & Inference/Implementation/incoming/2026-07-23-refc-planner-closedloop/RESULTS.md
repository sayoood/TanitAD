# RESULTS — D2 REF-C recovery augmentation (MEASURED on eval, `gpu_lock refc-cl-improve`)

**Date:** 2026-07-23 (Berlin). **Host:** `tanitad-eval` (A6000), acquired AFTER `abe82f1f` freed it (GPU
idle, no lock verified). **Every number here is `MEASURED (ours + artifact on the pod / pulled to this
dir)`.** REF-C deployed ckpt read-only throughout; FT wrote a NEW ckpt.

---

## P2a — recovery-response probe (ZERO training): REF-C is covariate-shift-BLIND ✅ GREENLIGHT

`recovery_probe.json` (pulled). REF-C base step 29999, 40 clean-val eps, **n=881 windows**, episode-cluster
bootstrap. `recovery_ratio` = fraction of the GT recovery correction REF-C's 0.5 s plan actually produces
when shown a real window warped to an in-envelope off-path pose (1 = full recovery, 0 = blind).

| perturbation | demand (m) | response (m) | **recovery_ratio [95% CI]** | read |
|---|---:|---:|---|---|
| lateral 0.5 m | 0.500 | 0.0043 | **0.0086 [0.0053, 0.0122]** | blind |
| **lateral 1.0 m** (primary) | 1.000 | 0.0074 | **0.0074 [0.0036, 0.0115]** | **blind** |
| lateral 1.5 m | 1.500 | 0.0099 | **0.0066 [0.0026, 0.0112]** | blind |
| yaw 3° | 0.333 | −0.0251 | −1.255 [−3.35, −0.05] (pooled −0.075) | blind / slightly anti |
| yaw 5° | 0.555 | −0.0442 | −1.579 [−4.44, −0.06] (pooled −0.080) | blind / slightly anti |
| lateral 1.0 m + yaw 3° | 1.333 | −0.0162 | −0.011 [−0.015, −0.006] | blind |

**Verdict (pre-registered): BLIND → GREENLIGHT P2b.** Shown a view offset by 1 m, REF-C moves its 0.5 s plan
by **7 mm** — it corrects **<1 %** of the offset (ratio 0.0074, CI upper 0.0115 ≪ the 0.35 BLIND bar). Yaw
response is negligible and slightly wrong-signed. This is the **direct mechanism** behind the on-policy plan
drift Gate-1/G1prep measured (`plan_xte 0.57 → 12.98 m`): REF-C, off its path, does not see/correct the
drift, so the plan compounds. The covariate-shift blindness the augmentation targets is **MEASURED, not
hypothesised.**

*(Independently valuable: this is the first direct measurement that REF-C's anchored-diffusion planner is
covariate-shift-blind to lateral offset — a mechanism result, separate from whether the FT fixes it.)*

---

## P2b — decoder-only recovery-augmentation FT + held-out corridor_departure

**FT (MEASURED, `ft.log`):** base REF-C step 29999, decoder-only **8,634,505 trainable** (frozen 90 M
encoder — the WM cannot be degraded), FT eps 0:28 (28 eps / 617 windows), envelope lat≤1.75 m / yaw≤5° /
clean 0.30 / λ_dev 0.5, 1500 steps @ lr 1e-4, ~0.83 s/step. **Training is healthy** — the recovery objective
is being learned (traj-L1 **0.64→0.44→0.41→0.29** by step 350; anchor-CE 2.27→1.26; gnorm settled ~20 after
a transient step-150 spike). The decoder is installing the recovery response P2a showed it lacked.

**FT terminal:** `RECOVERY_AUG_FT_DONE` (step 1500, ckpt `/workspace/refc-recovery-ft/ckpt.pt`);
anchor-CE 1.93→0.99 (the decoder learned to select recovery anchors). **Eval terminal:** `EVAL_CHAIN_DONE`
+ `CORRIDOR_SPLIT_DONE`, `corridor_split_results.json` (pulled). Lock **released cleanly, GPU FREE** (verified
via markers + `gpu_lock status`, not inferred from idle — RETRACTION_LOG C2).

### Held-out result (episode-disjoint 28:40, **12 eps / 264 windows**, PAIRED episode-cluster bootstrap)

Base REF-C vs recovery-FT on the **same** held-out windows. Positive `dCDR` = FT departs the corridor LESS
(better); positive `dADE` = FT closer to GT (better).

| stratum | dCDR@1.75 m [CI] | dADE@2s [CI] | dPEAK_xte [CI] |
|---|---|---|---|
| **overall** (264w) | **+0.0089 [+0.0008, +0.0197] ✓SEP** | **−0.288 [−0.428, −0.149] ✓SEP (worse)** | +0.0005 [−0.18, +0.23] n.s. |
| **junction** (69w, ≥10°) | **+0.0333 [+0.0091, +0.0576] ✓SEP** | −0.164 [−0.28, −0.05] ✓SEP (worse) | +0.497 [−0.01, +0.94] n.s. |
| longitudinal (102w) | +0.0005 [−0.003, +0.005] n.s. | −0.347 [−0.56, −0.04] ✓SEP (worse) | −0.131 [−0.39, +0.08] n.s. |

Absolute (overall): corridor_departure_rate@1.75 m **base 0.0174 → FT 0.0085** (halved); by-threshold
base {1.0:0.052, 1.75:0.017, 2.5:0.0036} → FT {1.0:0.043, 1.75:0.0085, 2.5:0.0013} (monotone, biggest
relative cut at the widest threshold); window-departure-rate **0.129 → 0.057**; closed_ade2s **0.587 → 0.875**;
peak_xte 0.549 → 0.548 (unchanged).

### Pre-registered verdict: **WIN on the primary metric — with a MEASURED, separated ADE cost (report both)**

**The lever WORKS for its target.** The pre-registered WIN condition is met exactly: overall held-out
corridor_departure @1.75 m is **CI-separated lower** (+0.0089 [+0.0008, +0.0197]) and the peak_xte guard
**holds** (dPEAK n.s.). The effect is **strongest at junctions** (+0.0333, 3.3 pp, separated) — precisely
where heading covariate-shift dominates, a strong mechanistic corroboration — and it **generalizes to
held-out episodes**, which is the whole novel claim: Gate-1's real-junction recovery FT gave held-out Δ≈0
(memorised at n≈15); this **synthetic in-envelope recovery, generated from every held-in window,
generalizes** (`MEASURED`). It halves the window-departure rate (0.129→0.057).

**The honest cost (elevated, not buried).** Closed-loop **ADE got WORSE, CI-separated, in every stratum**
(overall −0.288 m, 0.587→0.875). The recovery FT bought fewer lane departures at the price of average
tracking accuracy — the **exact drivability-vs-accuracy trade the research predicted** (P0 §3.4 / DESIGN §6;
Nachkov 2409.07965: closed-loop training trades minADE for off-road/drivability). Peak_xte is unchanged, so
the *worst* excursions are not cut — the *frequency* of departure is. Mechanism: a decoder-only 1500-step FT
at lr 1e-4 (39 epochs on 28 eps; `dev`/gnorm rose late in training) **over-corrects on average** — it steers
toward lane-centre more aggressively, reducing departures but adding oscillation/longitudinal error.

**Net, decision-grade:** in-envelope geometric recovery augmentation is a **real, data-efficient closed-loop
lever** that reduces REF-C's held-out corridor departures (esp. at junctions) where the memorising Gate-1 FT
could not — but the naive first cut trades ADE, so it is **not yet a free improvement**. The pre-named next
step (DESIGN §6.3 / PRE_REG HURT-branch fix, here indicated by the ADE cost rather than peak_xte): a
**gentler FT** — larger `λ_dev` and/or smaller `lat_max`/`clean_frac`, fewer steps, lower lr — to keep the
departure win while recovering ADE; then an **AlpaSim confirmation** (this is a low-OOD lane-keeping
mechanism result, **not** a safety rate). ⚠️ If closed_ade had been pre-registered as a co-guard (it was
listed as *context*, not a guard), this would read PARTIAL; I report both numbers so the reader is not
misled by the primary-metric WIN.

### Honest bounds
- **Lane-keeping, not safety** (map/agent-free source; a real off-road/collision rate needs AlpaSim, ~3.2×
  OOD — the very thing this instrument escapes). n=12 held-out eps (wide bands); ground-plane-lateral warp is
  optimistic (yaw exact). Within-instrument RELATIVE. Single seed, one FT config.
- **The ADE regression is the load-bearing caveat**: the lever is validated as a *departure-reduction
  mechanism that generalizes*, NOT as a drop-in REF-C upgrade. Do not deploy this FT ckpt.

---

## P2c — GENTLE-FT SWEEP: is the departure↓/ADE↑ trade escapable for decoder-only? → NO (Pareto frontier)

Pre-registered `GENTLE_SWEEP_PREREG.md` (both outcomes committed). 3 gentler configs, decoder-only (frozen
encoder), same held-out 28:40 eval, paired episode-cluster bootstrap. `dCDR>0` = FT fewer departures;
`dADE>0` = FT better ADE; **S** = CI excludes 0. NET WIN needed **dep held** (dCDR ≥ +0.005 & S) **AND**
**ADE recovered** (dADE CI∋0). Raw: `corridor_g{1,2,3}.json`, `gentle_sweep.log`.

| cfg (steps / lat / λ_dev / lr) | ftADE (base 0.587) | CDR base→ft | overall dCDR [CI] | overall dADE [CI] | pre-reg |
|---|---|---|---|---|---|
| **naive** (1500 / 1.75 / 0.5 / 1e-4) | 0.875 | 0.0174→0.0085 | **+0.0089 [+0.0008,+0.0197] S** | **−0.288 [−0.428,−0.149] S** | dep held, ADE worse |
| **g1** (500 / 1.75 / 1.0 / 1e-4) | 0.910 | 0.0174→0.0155 | +0.0019 [−0.012,+0.016] n.s. | −0.323 [−0.562,−0.109] S | dep LOST + ADE worse → **dominated** |
| **g2** (700 / 1.0 / 1.0 / 5e-5) | 0.713 | 0.0174→0.0117 | +0.0057 [−0.003,+0.017] n.s. | −0.125 [−0.236,−0.028] S | dep n.s., ADE recovering |
| **g3** (400 / 0.75 / 2.0 / 5e-5) | 0.686 | 0.0174→0.0206 | −0.0032 [−0.007,−0.000] **S (departs MORE)** | −0.099 [−0.177,−0.029] S | too gentle: dep reverses |

### Pre-registered verdict: **THE TRADE IS FUNDAMENTAL for decoder-only recovery-FT.**
No config nets a win: **dADE is CI-separated-worse in ALL FOUR** (ADE never returns to within noise of base),
and as the FT is gentled enough to shrink the ADE cost (g2 −0.125 → g3 −0.099), the departure reduction
**vanishes (g2 n.s.) then reverses (g3 departs more, S)**. `dCDR` and `dADE` move together along the
gentleness knob — the **Pareto frontier** is `naive (+0.0089, −0.288) → g2 (+0.0057, −0.125) → g3 (−0.0032,
−0.099)` (g1 is dominated). My pre-run hypothesis "fewer steps alone fixes it" is **refuted** — g1 (500
steps) lost the departure win AND kept the full ADE cost.

### Where the cost lives (MEASURED, directs the next mechanism)
The ADE cost is **concentrated OFF-junction**: at junctions g1/g2 *improve* ADE (junction dADE **+0.029 /
+0.133**) while overall dADE is negative → the damage is on **longitudinal/straight** windows (naive
longitudinal dADE was −0.347, the worst stratum). Mechanism (`HYPOTHESIS`, grounded in the P2a probe): the
**frozen encoder barely encodes the lateral offset** (probe recovery_ratio ~0), so the decoder cannot cleanly
separate "recover" from "continue" — to cut departures it must amplify its response to a faint signal, which
makes it **globally more reactive** and degrades straight-line/longitudinal tracking.

### Next mechanism (the lever is NOT retired — it needs one more ingredient)
1. ⭐ **Add a return-to-GT-speed / progress term** to the recovery objective (decoder-only, WM-safe, cheap) —
   the ADE damage is MEASURED to be longitudinal/straight, exactly what a speed/progress-preservation term
   protects; **g2 is the base to build on** (recovers most ADE, improves junction ADE +0.133, trends
   departure-positive). This is the cheapest discriminating next step and is data-directed.
2. **Encoder in the loop** (light-FT the frozen encoder) — the probe says the *features* don't encode the
   offset; unfreezing lets off-path features separate so the decoder needn't over-generalize. Higher ceiling,
   but touches the WM (the v4 hazard) → gate on the plan-free canary. Only if #1 is insufficient.

**Bottom line for P2c:** decoder-only recovery-FT is Pareto-bound; the pre-named next step was the
return-to-GT-speed term on g2 (tested next in P2d).

---

## P2d — g2 + RETURN-TO-GT-SPEED term: does the progress term escape the coupling? → NO (decoder-only EXHAUSTED)

Pre-registered `SPEED_TERM_PREREG.md`. Added `--lambda-prog` (extra L1 on the forward/x recovery component)
to **g2** (700 / lat 1.0 / yaw 3° / clean 0.5 / λ_dev 1.0 / lr 5e-5), decoder-only. Raw: `corridor_g2s1.json`,
`corridor_g2s2.json`, `speed_term_sweep.log`. NET WIN needed dep held (dCDR ≥ +0.005 & S) **AND** ADE
recovered (dADE CI∋0) **AND** peak guard.

| cfg | λ_prog | ftADE (base 0.587) | overall dCDR [CI] | overall dADE [CI] | junction dADE | NET WIN? |
|---|---|---|---|---|---|---|
| g2 | 0.0 | 0.713 | +0.0057 [−0.003,+0.017] n.s. | −0.125 [−0.236,−0.028] S | **+0.133** (good) | no |
| **g2s1** | 1.0 | **1.007** | −0.0064 [−0.019,+0.001] n.s. (departs MORE) | −0.420 [−0.680,−0.208] S | −0.131 S | no |
| **g2s2** | 2.5 | 0.695 | +0.0009 [−0.001,+0.003] n.s. (≈0) | −0.108 [−0.189,−0.042] S | −0.002 | no |

### Direction-2 PROMOTABILITY VERDICT: **decoder-only recovery-FT is EXHAUSTED — NOT promotable.**
The progress term **does not escape the coupling** (pre-registered branch 2). dADE stays **CI-separated-worse
in both** speed configs (ADE never returns to within noise of base 0.587), and the term is self-defeating:
at **λ_prog 1.0 it destabilizes** (ftADE 1.007 — worse than g2 AND base, departs more, and it *destroys* g2's
junction-ADE gain: +0.133 → −0.131 S), and at **λ_prog 2.5 it protects ADE only by pulling the decoder back
toward base's non-recovering plan** (ftADE 0.695 but dCDR ≈ 0 — the departure win is gone). **No λ_prog holds
departures AND recovers ADE.** Together with the gentle sweep (naive/g1/g2/g3 all Pareto-bound), the entire
decoder-only lever space — fewer steps, gentler envelope, stronger anchor, progress protection — fails to net
a win.

**Why (MEASURED-grounded):** the ADE cost is **not a forward-progress/speed error** a longitudinal term can
fix — it is the decoder's **global over-reactivity**, forced because the **frozen encoder does not encode the
lateral offset** (P2a recovery_ratio ~0). A progress term either fights for decoder capacity (λ_prog 1.0 →
worse) or, when strong, restores base behaviour and loses recovery (λ_prog 2.5). The decoder cannot decouple
"reduce lateral departure" from "stay accurate" **because the frozen features do not carry the offset
information** needed to do one without the other. **The encoder is the bottleneck, confirmed twice** (probe +
sweep).

### The sole remaining lever (what a promoter runs next)
**Encoder-in-the-loop light-FT**, plan-free-**canary-gated**. Unfreeze the last k encoder blocks (or a
low-lr full-encoder FT) so the off-path *features* separate — then the decoder can cut departures without
global over-reactivity. **Must** gate on the operative-rollout canary (roll the WM under TRUE actions →
ADE@2s; v1 baseline 0.452) so the world model cannot be silently degraded (the v4/v4.1/v4.2 hazard). Recipe
in `INTAKE.md`. This is a *training* escalation beyond decoder-only and belongs to a sanctioned arm, not this
additive-research stream.

**FINAL Direction-2 verdict:** in-envelope geometric recovery augmentation is a **validated, data-efficient,
generalizing** closed-loop lane-departure mechanism (halves held-out departures, strongest at junctions,
beats Gate-1's memorization wall) — the open quadrant is real. **But decoder-only over a frozen encoder it is
Pareto-bound and NOT promotable**; the measured, twice-confirmed bottleneck is the frozen encoder's failure to
encode lateral offset, so the only promotable path is the **encoder-in-the-loop (canary-gated) light-FT**.
Everything here is low-OOD lane-keeping, not a safety rate — a promotable arm still needs an AlpaSim
confirmation.
