# ⛔ THE F-18 SLOT PROBE FAILS ITS OWN POSITIVE CONTROL — **D1 IS WITHDRAWN**

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` · **Agent:** probe-positive-control
**Cites, and does not touch:** `…/incoming/2026-08-17-slot-probe-parity/SLOT_PROBE_PARITY.md`,
`…/incoming/2026-08-16-slot-probe-run/SLOT_PROBE_RUN.md`,
`…/incoming/2026-08-16-agent-slot-decoder/AGENT_SLOT_DECODER.md`,
`…/incoming/2026-08-16-o6-ablation/O6_ABLATION_AND_MASK_PROBE.md`.
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent readout is a world-model diagnostic and is
**never** driving performance.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**The positive control FAILS, so the pre-registered outcome is the loud one: `D1` — *"the encoder
does not carry agent geometry"* — is WITHDRAWN pending apparatus repair.** Handing the *identical*
probe (`sp2_probe.py`, md5 **`aabbee36fce5f164d47a555fad369cbd`**, byte-identical to the parity
run's), on the *identical* window pool (2 721 GT-lead windows, **70 episode clusters**), split,
seeds and estimator — the synthetic arms score 2 714–2 720 of them, the difference being their own
counted abstentions — a memory tensor that is **a direct encoding of the frame's OWN GT BOXES**
yields
**10.175 m [9.182, 11.168]** against a constant's **5.133 m** — **K1 Δ +5.042 [+4.080, +6.065],
separated, FAILED**. ⭐ **It fails at ALL THREE seeds** — K1 **+5.042 / +4.946 / +1.946**, every one
positive and separated — so the failure is not fit noise, even though its *magnitude* is (§2.5).
⇒ the apparatus is **no better on a representation that literally contains the answer** than on the
trained v6 latent (5.442–7.169 m) or on random vectors (5.949 m), and on two of three seeds it is
**decisively worse than both**.
⭐ **And the failure is localised, not mysterious.** A plain ridge regression on a memory that puts
the GT lead at a **fixed address** recovers it at **1.016 m, r = +0.979, K1 −4.116 PASS** — while
**the slot probe on that IDENTICAL tensor scores 6.319 m and still loses to the constant
(+1.178 [+0.313, +2.131], separated): 6.2× worse than a linear map on the same numbers.** So the
information is present and even linearly available; what cannot extract it is the **74-slot decoder
+ `pred_lead` readout rule**. ⛔ **At the 74-query operating point, even with that perfect
representation AND the rule repaired, the apparatus's ceiling is a TIE with a constant (+0.522, not
separated) — it never produces a K1 PASS.** ⛔ **And that rule is provably not the function its
own docstring claims:** the GT side selects **the NEAREST in-corridor agent within 30 m**, the
prediction side selects **the MOST CONFIDENT in-corridor slot with no range cap at all**
(`sp2_probe.py:99` vs `:141`). Making the two symmetric recovers **3.2 m of the 5.0 m** the oracle
loses — and still does not let it beat a constant. ⇒ **Every F-18 conclusion — 2026-08-16's D1,
2026-08-17's "no better than noise", and the `tokens` reading that localised the loss to the
ENCODER — rests on an instrument that has now been shown unable to pass a control it should pass
trivially. None of them may be quoted as evidence about the world model until the apparatus is
repaired and re-passes this control.** ⚠️ This does **not** show the v6 latent *does* carry agents;
it shows **we do not know**, which is a different and much weaker state than the one the programme
has been reasoning from.

⭐⭐ **AND THE RUN DOES NOT END ON A NEGATIVE — THE REPAIR IS A ONE-FLAG CHANGE (§2.7).** The same
probe, the same oracle cache, `--n-queries 16` instead of 74: **error 10.175 → 2.982 m, median
0.816 m, K1 −2.186 [−3.165, −1.192] separated — the FIRST K1 PASS ANYWHERE IN F-18**, with the
constant on the retained windows unchanged (5.167 vs 5.133), so it is not an abstention artefact.
⇒ **The 74-query operating point, not the readout task, is what breaks the instrument** — and 74 was
chosen by fitting the in-grid *agent-count* p99, which is correct for SET PREDICTION and
catastrophic for LEAD SELECTION, because it puts ~13 slots in the 3.5 m corridor for the rule to
argmax over (§2.6-i). ⭐⭐ **AND THE REPAIR PASSES ITS OWN NEGATIVE CONTROL (§2.7-i): at the SAME 16
queries the window-matched RANDOM-LATENT NULL fails K1 by +4.808 [+4.112, +5.482], separated** — a
**~7 m separation between answer and noise**, and **the first configuration of this probe that both
passes on signal and fails on noise.** ⭐⭐ **AND THE REAL ARM AT THAT REPAIRED POINT IS THE FIRST
F-18 READING FROM A WORKING INSTRUMENT (§2.7-ii): `v6F@11250` scores 8.331 m, K1 +3.217
[+2.310, +4.246] — still FAILS — with BOTH anti-echo controls unseparated, like noise and unlike the
oracle.** ⇒ **D1 points the same way it always did — but on evidence that can now bear it.**
⚠️ **It is NOT restored here:** one seed against a measured 3.096 m seed spread, arm-vs-arm ordering
that is marginal rather than paired (window sets 2 408 / 2 577 / 2 665), and an early-read at 37.5 %.
⭐ **The path is now short and specified: {oracle, latent, null} @ 16 queries × ≥3 seeds, re-run at
30 k — ~9 fits, no trunk compute.**

⭐ **Two further results stand on their own and do NOT depend on the broken headline**, because they
are computed from the banked heads' own outputs and from a second, independent instrument:

1. **TASK 2 — the failure is NOT a resolution limit, and it is UNIFORM.** The GT lead is a *large*
   object: median apparent width **37.8 px ≈ 2.4 ViT patches**, only **4.34 %** of scored windows
   below one patch and **0.29 %** below half a patch. The probe is *worst* where the agent is
   **nearest and biggest** (K1 **+5.670** at `cx` < 10 m vs **−5.262** at 20–30 m). ⭐ **The
   mechanism, MEASURED:** every arm emits a **near-constant** gap — @11250 predicts **20.7 ± 1.6 /
   21.0 ± 2.2 / 20.6 ± 1.6 m** across three range strata whose true means are **7.8 / 14.5 /
   24.0 m** — and the **random-latent null does exactly the same** (18.0 ± 2.8 everywhere). The
   stratum profile is the interaction of a constant emission with the label distribution, not a
   property of the latent.
2. **TASK 3 — ⛔ the brief's premise is wrong for O4 and the correction matters.**
   `build_o4_weights` (`stack/scripts/train_v6_staged.py:745`) returns **per-window SAMPLING
   weights** for `InteractionSampler` (`:2470`). **It is not a loss term, has no gradient, and
   "its cosine" does not exist.** The pairwise O-term cosines are **cited from the precedent**
   (5 seeds, live grid, exact-linearity, controls at exactly 0): **O2 ↔ O5 = +0.870 — nearly
   collinear**, while **O2 ↔ O3 = +0.028 with an unstable sign**, and O6 orthogonal to everything
   (|cos| ≤ 0.042 vs chance 0.002). ⛔ **The O-vs-AGENT-READOUT cosines did NOT land** and — because
   their reference direction is the failing head's own loss — **should be re-scoped before they are
   run at all** (§5.3).

---

## 0. ⛔ THE STAMPS THAT BIND EVERY NUMBER BELOW

1. **`v6F-SW-30k@11250` / `@9000`** — the checkpoint is part of the arm. The synthetic arms are
   built **from the @11250 cache**, so they inherit its window set exactly and carry its stamp.
2. ⚠️ **EVERY v6 POINT HERE IS AN EARLY-READ at 37.5 % of training.** 30 000 remains the primary
   read. Nothing here executes or blocks the pre-registered D1 DROP — it **removes the evidence
   the DROP would have been decided on.**
3. **The probe is byte-identical to the parity run's.** `md5(sp2_probe.py)` =
   `aabbee36fce5f164d47a555fad369cbd` at all three of: the parity run's `code/`, its scratch copy,
   and mine. ⛔ **No second fit procedure was written** — that is the whole point of a control.
   Equivalence is therefore established by **byte identity**, which is stronger than the parity
   run's own re-run regression and costs no GPU.
4. **Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, clustered on the 70
   eval episodes.** ⛔ `overlapping_holdout_se` is never imported.
5. ⚠️ **LEAD-ENRICHED, NOT PARITY**, inherited unchanged from the parity run (§2.3 there):
   `parity: False` is in every cache meta and remains so in mine.
6. **GPU: dev-box RTX 4060 only. ⛔ Thor was never used for compute** — no checkpoint was pulled,
   no snapshot was made, nothing was run there. Only banked local artifacts were read.

---

## 1. ⛔ WHY THIS RUN EXISTED — the gap, restated in one line

`SLOT_PROBE_PARITY.md` ran five controls (C-CONST, C-SHUF, C-EPMEAN, C-SHUF-XEP, matched
random-latent). **Every one is NEGATIVE**: they establish that the probe is not cheating. **Not one
establishes that the probe can SUCCEED.** So *"the v6 latent does not carry agents"* was
confounded with *"this probe/label/window/fit pipeline cannot read agents from anything"* — and the
`tokens` arm, reading the same pixels SAM3 detects agents in, failed identically, which is exactly
what an apparatus ceiling looks like.

**Committed in advance, in the brief, before any fit:**

| outcome | consequence |
|---|---|
| positive control PASSES, v6 arms FAIL | D1 **hardens** into a statement about our encoder |
| positive control **FAILS** | ⛔ **D1 WITHDRAWN pending apparatus repair; the defect is in the probe, not the world model** |

**The measured outcome is the second one.**

---

## 2. TASK 1 — THE POSITIVE CONTROL

### 2.1 The construction, declared before any fit (`code/pc1_oracle_cache.py`)

It is `pA_null_matched.py`'s move **with signal instead of noise**: take the REAL @11250 cache and
replace **only** the `cells` tensor, keeping every window, target, class, rate, `clip_id`,
`episode_uid` and the declared split — so the paired bootstrap's clusters are literally the same
objects.

`agents` is `[A, 6] = (cx, cy, yaw, l, w, occ)`, already restricted by `sp1` to the in-grid box.

* **`ORC-CELLS`** — sort the frame's agents by `cx` ascending, keep the first 16 (the real memory
  length), cell *k* = `P @ f_k` with
  `f = [cx/60, cy/16, sin yaw, cos yaw, l/10, w/5, 1.0]` and `P` a fixed `[128, 7]` N(0,1) matrix
  (seed 20260817 — a random projection, so no memory dimension is a bare copy of a label field);
  empty cells are zero; the tensor is rescaled to the **real** cells' global std (**0.030898**),
  then Gaussian noise at **0.10 ×** that std is added.
  *This is the SCENE, honestly encoded: the head must still learn the corridor rule, the
  projection, the set structure and the selection.*
* **`ORC-DIRECT`** — identical, except **the GT lead itself (chosen by `gt_lead_gap`'s own rule) is
  placed in CELL 0**. ⚠️ This deliberately leaks the corridor rule into the feature — **that is
  what an upper bound is for**, and it is stated rather than hidden. A probe that fails here is
  broken, full stop.

| MEASURED (`raw/pc1_meta_orc010.json`, `raw/pc1_meta_orcdir.json`) | ORC-CELLS | ORC-DIRECT |
|---|---|---|
| rows / cells / dims | 5 617 · [16, 128] | 5 617 · [16, 128] |
| windows truncated at 16 cells | 1 075 (19.1 %) | 1 075 (19.1 %) |
| **GT lead survives the truncation** | **98.22 %** | **100 %** (by construction) |
| signal rescale factor | 0.024420 | 0.024423 |

### 2.2 ⛔ THE RESULT — the apparatus fails on both

Paired episode-cluster bootstrap, n_boot 2000, **identical 70 clusters**. Positive Δ = the arm is
**WORSE** than the control. `C-CONST` **5.133 m** · `C-EPMEAN` **3.122 m**, identical to the parity
run's because they are label-side on the same windows.

| arm | `lead_gap_abs_err_m` | **K1** Δ vs C-CONST | **K5** Δ vs C-EPMEAN | **K2** Δ vs C-SHUF | n win / clust |
|---|---|---|---|---|---|
| ⭐ **`ORC-CELLS` seed 0** | **10.175** [9.182, 11.168] | **+5.042** [+4.080, +6.065] sep ⛔ | +7.053 sep ⛔ | **−0.751** sep ✅ | 2 719 / 70 |
| ⭐ **`ORC-CELLS` seed 1** | **10.090** [8.700, 11.634] | **+4.946** [+3.493, +6.583] sep ⛔ | +6.985 sep ⛔ | **−0.732** sep ✅ | 2 701 / 70 |
| ⭐ **`ORC-CELLS` seed 2** | **7.099** [6.295, 7.933] | **+1.946** [+1.028, +2.847] sep ⛔ | +3.963 sep ⛔ | **−0.917** sep ✅ | 2 694 / 70 |
| ⭐⭐ **`ORC-DIRECT` seed 0** (§2.4) | **6.319** [5.515, 7.173] | **+1.178** [+0.313, +2.131] sep ⛔ | +3.200 sep ⛔ | **−1.012** sep ✅ | 2 714 / 70 |
| `v6F-SW-30k@9000` *(parity run, cited)* | 5.442 | +0.309 sep ⛔ | +2.320 sep ⛔ | +0.007 ns | 2 721 / 70 |
| `v6F-SW-30k@11250` *(parity run, cited)* | 7.169 | +2.036 sep ⛔ | +4.047 sep ⛔ | +0.105 sep ⛔ | 2 721 / 70 |
| RANDOM-LATENT NULL *(parity run, cited)* | 5.949 | +0.816 sep ⛔ | +2.827 sep ⛔ | −0.046 ns | 2 721 / 70 |

⛔ **THE ORACLE IS THE WORST ARM IN THE TABLE.** A memory built from the answer scores 10.2 m; the
random-latent null scores 5.9 m. **Any ordering derived from this instrument is uninterpretable**,
because the instrument ranks *perfect information* below *noise*.

⭐ **AND THE PROBE IS NOT MERELY "NOT LEARNING" — IT IS LEARNING AND THEN LOSING IT AT THE READOUT.**
Three independent facts, all from the oracle arm's own banked JSON:

| fact | ORC-CELLS | what it rules out |
|---|---|---|
| training loss `loss_centre` 23.42 → **1.12** over 3 000 steps | ✅ | "the head cannot fit" |
| **K2 SEPARATES IN THE CORRECT DIRECTION** — the arm is **0.75 m BETTER** than its own C-SHUF twin, separated | ✅ **first time anywhere in F-18** | "the head ignores its input" |
| `_diag_oracle_slot_abs_err_m` **median 0.713 m** (best in-corridor slot vs the GT lead) | ✅ | "the geometry never reaches the slots" |

⚠️ **The third row must be read against the parity run's own §5.2 warning** — at 74 queries a random
latent earns an oracle median of **0.790 m**, so 0.713 is *at* that floor and is **not** admissible
on its own. It is listed because it agrees with the two rows above it, not because it carries the
argument. **The argument is carried by K2 and by §2.3.**

⭐ **AND I TRIED TO RESCUE THAT DIAGNOSTIC AND FAILED — which extends the parity run's warning.**
The obvious repair is to **gate it on presence**: a head that merely scatters slots cannot buy
confidence. MEASURED (`raw/pc5c_gated_oracle_diag.json`), median best-in-corridor-slot error vs the
GT lead:

| gate | `ORC-CELLS` | `v6F@11250` | `v6F@9000` | **RANDOM-LATENT NULL** ⇐ the floor |
|---|---|---|---|---|
| none | 0.713 | 0.809 | 0.765 | **0.916** |
| presence ≥ 0.5 | 1.202 | 0.809 | 0.769 | **0.924** |
| presence ≥ 0.7 | 2.430 *(n 1 148)* | 1.042 *(n 1 916)* | 0.957 *(n 1 851)* | **2.038** *(n 1 465)* |

⛔ **Every arm sits at or worse than the noise floor at every gate — including the oracle.** With
~13 in-corridor emissions per window (§2.6-i), *some* slot lands near the GT by geometry alone, and
confidence does not separate them. ⇒ **The oracle-slot diagnostic is not merely "at chance at 74
queries" (parity §5.2) — it is uninformative in every gated form I could construct, and it should be
DROPPED rather than annotated.** Escalation 4 of the parity run is upgraded accordingly.

### 2.3 ⭐ THE DISAMBIGUATION THAT SETTLES IT — a ridge on the same tensors

Two readings survived §2.2: **(a)** my oracle encoding does not actually contain the answer, so the
control is the broken thing; **(b)** it does, and the slot apparatus cannot extract it. A ridge
regression settles it in seconds — a deliberately dull second instrument, flattened memory →
lead gap, alpha chosen on an **episode-disjoint inner split of the PROBE-TRAIN clips only**, scored
on the same eval windows with the same estimator (`code/pc6_linear_readout.py`).

⛔ **THIS IS A DIFFERENT INSTRUMENT AND IS NEVER COMPARABLE TO AN F-18 SLOT NUMBER.** It is run for
exactly one job — to say whether the information is *present* — and it carries its own floor because
it is run on the null too.

| memory | ridge err (m) | vs C-CONST 5.133 | **K1** | corr(pred, GT) | n |
|---|---|---|---|---|---|
| ⭐ **`ORC-DIRECT`** (lead at a fixed address) | **1.016** [0.989, 1.044] | −4.116 | **[−4.726, −3.603] PASS ✅** | **+0.979** | 2 721 / 70 |
| `ORC-CELLS` (lead at a *variable* address) | 5.562 [5.067, 6.099] | +0.429 | [−0.074, +0.895] fail | +0.177 | 2 721 / 70 |
| `v6F-SW-30k@11250` | 6.713 [6.010, 7.482] | +1.580 | [+0.765, +2.455] fail | **+0.159** | 2 721 / 70 |
| `v6F-SW-30k@9000` | 6.944 [6.194, 7.816] | +1.811 | [+1.044, +2.692] fail | **+0.108** | 2 721 / 70 |
| RANDOM-LATENT NULL | 8.534 [8.148, 8.936] | +3.401 | [+3.014, +3.766] fail | **−0.018** | 2 721 / 70 |

⭐ **Reading (b) is the answer.** `ORC-DIRECT` is recovered at **1.016 m with r = 0.979** — the
encoding pipeline is sound and the lead gap is *linearly* available from it. The **same** slot
probe on the **same** tensor is the arm reported in §2.4.

⚠️ **`ORC-CELLS` failing the ridge is EXPECTED and is not evidence against the encoding.** The lead
gap is `min{cx : |cy| ≤ 1.75}` — a **selection over a set** whose answer sits in a *different cell
in every frame*. No linear map can compute a conditional min. I designed the first disambiguation
badly, noticed, and built `ORC-DIRECT` — where the answer is at a fixed address — precisely to
remove that objection. **Stated here rather than quietly dropped.**

⭐ **AND THE RIDGE PRODUCES A GENUINELY NEW FACT ABOUT THE v6 LATENT, which no F-18 arm could give.**
Under a linear readout the real v6 arms are **1.6–1.8 m better than the random-latent null on the
identical windows**, with a **positive** correlation to the true gap (**+0.159 / +0.108**) where the
null's is **−0.018**. ⇒ ⚠️ **The parity run's sharpest sentence — *"the trained v6 latent serves this
readout no better than random vectors"* — is TRUE OF THE SLOT PROBE AND FALSE OF A LINEAR READOUT.**
The latent carries *some* linearly-decodable lead information; it is simply nowhere near enough to
beat a constant (both fail K1, and both fail K5 by a wide margin). ⚠️ These ridge numbers are a
**single fit each, no seed replicate** (a ridge is a closed-form solve, so there is no optimiser
seed — but the alpha selection has one; it landed on 10–100 in every arm).

### 2.4 ⭐⭐ `ORC-DIRECT` THROUGH THE SLOT PROBE — the sharpest form of the finding

The answer is at a **fixed, known address** (cell 0), the lead survives truncation in **100 %** of
windows, and a ridge on that very tensor recovers it at **1.016 m, r = +0.979**. The identical slot
probe, on the identical windows:

| `GT-ORACLE-DIRECT` | **seed 0** | **seed 1** |
|---|---|---|
| `lead_gap_abs_err_m` | **6.319** [5.515, 7.173] · median 4.142 | **8.585** [7.539, 9.611] |
| **K1** vs C-CONST (5.133) | **+1.178** [+0.313, +2.131] **sep ⛔ FAILS** | **+3.449** [+2.396, +4.488] **sep ⛔ FAILS** |
| **K5** vs C-EPMEAN (3.122) | +3.200 [+2.352, +4.087] sep ⛔ | +5.470 sep ⛔ |
| **K2** vs C-SHUF | **−1.012** [−1.326, −0.701] sep ✅ | **−0.801** sep ✅ |
| **vs C-SHUF-XEP** | **−2.076** [−3.054, −1.094] sep ✅ | — |
| n windows / clusters | 2 714 / 70 | 2 716 / 70 |
| verdict | `DROP/RE-SCOPE` — `KEEP: false` | `DROP/RE-SCOPE` |

⚠️ **Two seeds, so NO RANGE IS QUOTED** — §2.5's own rule. What both seeds agree on is the only
thing claimed: **K1 fails, positive and separated, on a memory a ridge reads at 1.016 m.**

⛔ **THE ONE-LINE STATEMENT OF THE WHOLE RUN:**

> **On a memory tensor from which a ridge regression recovers the lead gap at 1.016 m with
> r = 0.979, the F-18 slot probe scores 6.319 m and LOSES TO A CONSTANT, separated.**
> **It is 6.2× worse than a linear map on the same numbers.**

⭐ **And the two anti-echo controls now BOTH separate in the correct direction** (K2 −1.01, XEP
−2.08). Per the parity run's own §5.4 decomposition, that is the regime *"window-to-window
information is in use — the only regime in which an agent claim is available at all."* **The head is
reading its input at window granularity and still cannot beat a constant.** ⇒ The deficit is not
information, not fitting, and not episode-recognition: it is the reduction of 74 slots to one
number.

**Under the symmetric readout rule (§2.6) it improves but still does not pass:**

| rule | err (m) | **K1** Δ vs C-CONST | corr(pred, GT) |
|---|---|---|---|
| R0 incumbent | 6.311 | +1.178 [+0.314, +2.130] sep ⛔ | +0.309 |
| R1 nearest, `cx ≤ 30`, no gate | 5.764 | +0.625 [−0.217, +1.442] **not sep** | +0.455 |
| **R2 nearest, `cx ≤ 30`, presence ≥ 0.5** | **5.538** | **+0.522** [−0.341, +1.350] **not sep** | **+0.465** |

⇒ ⛔ **THE CEILING AT 74 QUERIES, MEASURED: with a perfect representation AND a repaired rule, the
best this instrument achieves is a TIE with a constant** (not separated, still positive), at
r = 0.465 — **it never produces a K1 PASS at this operating point.** Any "K1 fails" verdict it
emits there is therefore **uninformative about the representation**, which is precisely why D1 is
withdrawn. ⭐ **§2.7 shows the ceiling is a property of the 74 queries, not of the task: at 16
queries the incumbent rule passes K1 by 2.19 m.**

### 2.5 ⛔ SEED SPREAD — **AND I DREW A CLAIM FROM n=2 THAT n=3 REFUTED. READ THIS.**

The brief requires ≥3 seeds and a between-condition vs between-seed comparison. The parity run
MEASURED **1.826 m of K1 spread** across three seeds on one frozen cache — *larger* than the
1.727 m spanned by five checkpoints.

⛔ **I wrote, from seeds 0 and 1 alone, that "the oracle arm is 19× more reproducible than the real
arm" and reasoned that a converged fit implies real signal. SEED 2 REFUTES IT** — it lands at
**7.099 m / K1 +1.946**, and the true 3-seed range is **3.096 m of K1, LARGER than the parity run's
1.826 m, not 19× smaller.**

| condition, one frozen cache | seeds | err (m) | **K1** | **K1 range** | all fail K1? |
|---|---|---|---|---|---|
| **`ORC-CELLS`** (this run) | 0, 1, 2 | 10.175 · 10.090 · **7.099** | +5.042 · +4.946 · **+1.946** | **3.096** | **YES — all 3** |
| `v6F@11250` *(parity run, cited)* | 0, 1, 2 | 7.169 · 6.026 · 5.343 | +2.036 · +0.894 · +0.210 | **1.826** | YES — all 3 |
| `ORC-DIRECT` (this run) | 0 | 6.319 | +1.178 | ⚠️ **n=1** | YES |

⇒ ⚠️ **THE CORRECTED READING, and it is the opposite of what I first wrote:** the fit is **MORE**
seed-dependent on the oracle than on the real latent. A perfect representation does **not** make
this probe converge — which is one more piece of evidence that the instrument, not the input, is
what is unstable. ⭐ **THE HEADLINE IS UNAFFECTED AND THAT IS WHY IT SURVIVES: K1 FAILS AT ALL THREE
SEEDS, ALWAYS POSITIVE, ALWAYS SEPARATED** (+5.042, +4.946, +1.946) — exactly the structure the
parity run found for the real arms, and exactly why "the oracle fails" is robust while "how badly"
is not.

⚠️ **ROOT-CAUSE CLASS, logged against myself:** *a spread estimated from two points.* The parity run
escalated **"a single probe fit is not a measurement"** as its highest-value finding; I re-committed
a weaker form of it with n=2 and was caught by the third fit. **A range needs ≥3 points before it is
a range at all.**

⚠️ Seed 2 first **crashed with `CUDA error: an illegal memory access`** — collateral from my killing
a stranded 12 GB process on the shared GPU, **not** a code fault. It was relaunched cleanly; both
the crash and the relaunch are in `raw/chain_a.log` / `raw/chain_a3.log`.

### 2.6 ⛔ THE INSTRUMENT DEFECT THE CONTROL EXPOSED — the readout rule is not symmetric

`sp2_probe.py`'s own section header reads *"the lead readout — §4.1, applied IDENTICALLY to
prediction and to GT"*. **It is not.**

| side | predicate | selection |
|---|---|---|
| **GT** (`gt_lead_gap`, `sp2_probe.py:99`) | `cx > 0` ∧ `|cy| ≤ 1.75` ∧ **`cx ≤ 30`** | **min `cx`** — the NEAREST |
| **PRED** (`pred_lead`, `sp2_probe.py:141`) | `cx > 0` ∧ `|cy| ≤ 1.75` — **no range cap** | **argmax presence** — the MOST CONFIDENT |

`SlotDecodeRanges` decodes to **60 m**, so a head that correctly and confidently detects a car at
45 m is scored against a GT lead at 8 m. "Nearest" and "most confident" are different functions.

`code/pc5_readout_rule.py` re-reads the **banked heads' own slot outputs** — no refit, no change to
`sp2_probe.py` — and applies alternative rules on the same windows with the same estimator. **R0
reproduces every banked headline to ±0.008 m**, which is the gate.

| rule | ORC-CELLS | @11250 | @9000 | RANDOM-LATENT NULL |
|---|---|---|---|---|
| **R0** incumbent (argmax presence, no cap) | **10.183** (r +0.143) | 7.169 (r −0.008) | 5.442 (r −0.034) | 5.949 (r +0.027) |
| **R1** nearest, `cx ≤ 30`, no gate | 7.601 (r +0.268) | 7.366 | 7.562 | 10.967 |
| **R2** nearest, `cx ≤ 30`, presence ≥ 0.5 | **6.959** (r +0.334) | 7.366 | 7.534 | 9.864 |
| **R2** nearest, `cx ≤ 30`, presence ≥ 0.7 | 6.968 *(n 1 091)* | 6.329 *(n 1 915)* | 6.694 *(n 1 851)* | 5.958 *(n 1 465)* |
| C-CONST on the same windows | 5.133 | 5.133 | 5.133 | 5.133 |

⇒ **Three things, and the third is the one that matters:**
1. ✅ **The asymmetry is real and expensive on a good representation** — making the rule symmetric
   recovers **3.2 m of the oracle's 5.0 m K1 deficit** (10.18 → 6.96) and lifts its correlation with
   the truth from **+0.14 to +0.33**.
2. ✅ **It is a fix, not a cheat** — the same rule makes the **random-latent null WORSE**
   (5.95 → 9.86), which is what a rule that rewards real geometry must do.
3. ⛔ **It does not rescue the oracle.** Under the symmetric rule the oracle is still **+1.90
   [+0.98, +2.78] worse than a constant**, separated. **So the readout rule is *a* defect, not
   *the* defect, and repairing it alone will not make D1 quotable.**

### 2.6-i ⭐ HOW MUCH CLUTTER THE SELECTION FACES — and why five negative controls could not see it

MEASURED over all 3 023 eval windows from the banked heads' own slot outputs
(`raw/pc5b_selection_diag.json`):

| arm | in-corridor slots per window (mean / median) | geometric emission rate | **chosen slot beyond 30 m** | beyond 40 m |
|---|---|---|---|---|
| **`ORC-CELLS`** | **19.58 / 13** | 0.9997 | ⛔ **15.98 %** | 8.80 % |
| `v6F@11250` | 12.93 / 13 | 1.0000 | 0.30 % | 0.23 % |
| RANDOM-LATENT NULL | 12.31 / 12 | 1.0000 | **0.00 %** | 0.00 % |

⇒ **Two facts, and the second is the reason this run had to exist.**
1. The incumbent rule selects among **~13 in-corridor candidates on essentially every window**
   (74 queries; the corridor is 3.5 m of a 32 m-wide grid). It is not picking "the" lead — it is
   picking one of thirteen.
2. ⭐ **THE MISSING 30 m CAP CORRUPTS ~1 IN 6 ORACLE WINDOWS AND *ZERO* NULL WINDOWS.** A head with
   real, varied geometry confidently emits agents out to the 60 m decode limit and is then scored
   against a GT that stops at 30 m; a degenerate head that emits ~20 m on every frame never trips
   the defect at all. ⛔ **The bug is INVISIBLE on a broken arm and only bites on a working one.**
   That is precisely why C-CONST, C-SHUF, C-EPMEAN, C-SHUF-XEP and the random-latent null — five
   controls, two studies — could not have caught it, and why a **positive** control is not optional.

### 2.6a ⭐ WHAT THE TWO RULES ACTUALLY DO — the emitted distributions, and they name the mechanism

| arm | **R0** pred (m) | **R2** pred (m) | GT (m) |
|---|---|---|---|
| **`ORC-CELLS`** | 15.34 ± **12.91** · r **+0.143** | 9.34 ± 5.25 · r **+0.334** | 15.53 ± 6.20 |
| `v6F@11250` | 20.85 ± **1.96** · r −0.008 | 8.69 ± 0.71 · r +0.052 | ″ |
| `v6F@9000` | 16.75 ± **1.05** · r −0.034 | 8.47 ± 0.76 · r −0.016 | ″ |
| RANDOM-LATENT NULL | 18.05 ± 2.76 · r +0.027 | 6.12 ± 3.59 · r −0.001 | ″ |

⭐ **THE ORACLE HEAD IS NOT EMITTING A CONSTANT — IT IS EMITTING THE WRONG SLOT.** Its R0 spread is
**12.91 m**, twice the GT's own 6.20 m, and its mean (15.34) sits almost exactly on the GT mean
(15.53). It is *choosing*, and choosing badly. **R0 over-predicts** (argmax presence lands on a
confidently-detected FAR agent, up to the 60 m decode limit); **R1/R2 under-predict** (nearest lands
on a spurious NEAR false positive: 9.34 m against 15.53 m). ⇒ ⛔ **Neither "most confident" nor
"nearest" identifies the lead among the head's own ~8 in-corridor emissions.** The apparatus lacks a
lead-identifying criterion at all — the geometry is present (§2.3), the head decodes it (K2, §2.2),
and the reduction from 74 slots to one number destroys it. That is a **specifiable, fixable
defect**, and it is what escalation 5 and §2.7 test.

⚠️ **And the real arms are the opposite shape:** spread **1.05–1.96 m** against the GT's 6.20 m, with
correlation ≈ 0 under every rule. **They emit one number; the oracle emits the wrong one of many.**
Those are different failures, and only the second is an apparatus failure — which is why the oracle
had to be run before either could be named.

### 2.6b ⛔ THE ROOT CAUSE, STATED AT DESIGN LEVEL — the objective and the functional disagree

The head is trained with `slot_set_loss`, a **DETR set loss under Hungarian matching**. Its
`presence` is calibrated to *"is this slot matched to SOME ground-truth agent"* — it is **not**, and
was never asked to be, *"is this slot the LEAD"*. The evaluation functional then asks for **one
specific element of that set**, chosen by a rule the training objective never optimised.

⇒ **Among the head's true positives, `argmax presence` is close to arbitrary with respect to the
question being scored.** With 74 queries the corridor (3.5 m wide out of a 32 m grid) collects ~8
emissions per frame, so the selection is being made over roughly eight candidates on nearly every
window. **A perfect set-predictor can therefore score arbitrarily badly on this metric, which is
exactly what §2.2 measures.** This is the same family as the programme's `--v2` conflation failure
and the C6 confound: **a quantity is being read off an object that was optimised for a different
question.** ⭐ **It also means the fix is not "train longer" or "a bigger head": the readout needs a
lead-identifying criterion** — an explicit lead query, presence-weighted nearest, de-duplication, or
a far smaller query set — **and the positive control is the cheap test of any candidate.**

⚠️ **The real arms are unaffected in direction under every rule** (K1 fails everywhere, `r` stays in
[−0.03, +0.08]) — they have no geometry to lose to a bad rule. That is consistent with D1 being
true; **it is not evidence for it**, because the instrument that would have to show it has just
failed its own control.

---

### 2.7 ⭐⭐ THE REPAIR — **the positive control PASSES at `n_slot_queries` = 16**

§2.6-i said the selection faces ~13 in-corridor candidates per window at 74 queries. The cheapest
possible repair is therefore to stop emitting so many. `code/chain_c.sh` re-runs the **same probe**
on the **same `ORC-CELLS` cache** with the head's only change being `--n-queries 16`
(3 207 445 params — still inside the prereg §6 2–4 M band):

| `ORC-CELLS`, seed 0 | **n_queries 74** | **n_queries 16** |
|---|---|---|
| `lead_gap_abs_err_m` | 10.175 [9.182, 11.168] | ⭐ **2.982** [2.337, 3.709] |
| **median** abs err | **7.431 m** | ⭐ **0.816 m** |
| **K1** vs C-CONST | **+5.042** [+4.080, +6.065] sep ⛔ | ⭐ **−2.186** [−3.165, −1.192] sep ✅ **PASSES** |
| **K5** vs C-EPMEAN | +7.053 sep ⛔ | **−0.156** [−0.877, +0.643] **not sep** — ties the episode-identity oracle |
| **K2** vs C-SHUF | −0.751 sep ✅ | ⭐ **−3.111** [−3.670, −2.590] sep ✅ |
| vs C-SHUF-XEP | — | ⭐ **−6.076** [−7.371, −4.817] sep ✅ |
| C-CONST on the scored set | 5.133 | **5.167** ⇐ *the control that matters* |
| C-EPMEAN on the scored set | 3.122 | **3.137** |
| scored windows / clusters | 2 719 / 70 | 2 408 / 70 |

⇒ ⭐⭐ **THE APPARATUS IS REPAIRABLE, AND THE REPAIR IS A ONE-FLAG CHANGE.** At 16 queries the probe
recovers the lead gap from the oracle memory to a **median of 0.816 m** and **beats the constant by
2.19 m, separated** — the first K1 PASS anywhere in F-18. **The 74-query operating point, not the
readout task, is what breaks it.**

⚠️ **THE ABSTENTION CHECK, because a smaller head emits less and could be dropping the hard
windows.** The `cells` arm abstains on **108 of 2 721** windows (4.0 %); the paired set falls to
2 408 (11.5 % dropped) once every arm must emit. ⭐ **But the constant's own error on the retained
subset is 5.167 m against 5.133 m on the full set, and C-EPMEAN's is 3.137 vs 3.122** — i.e. **the
surviving windows are not easier**, and the K1 pass is not an artefact of dropping the hard ones.
⚠️ It is still **not a paired comparison with the 74-query arm**, because the window sets differ;
both n are printed and neither is quoted as the other's.

⚠️⚠️ **ONE SEED. Given §2.5 — 3.096 m of K1 spread across three seeds at 74 queries — a single fit
is not a measurement, and this one must be replicated at ≥3 seeds before "16 queries fixes it" is
quoted as a repair.** What it establishes now is weaker and still decisive: **an operating point
exists at which this apparatus CAN pass its positive control**, so the 74-query failure is a
property of the configuration, not of the question.

⛔ **AND THE REPAIR DOES NOT GET A FREE RIDE — IT HAS ITS OWN NEGATIVE CONTROLS.** A geometry change
that makes the ORACLE pass K1 has fixed nothing if it also makes **noise** pass; that would be this
package's own error class, one level up. So `n_queries 16` is re-run on the **window-matched
random-latent null** and on the **real `v6F@11250` arm**
(`raw/results_nullmatched_nq16.json`, `results_s11250_nq16.json`, `code/chain_c.sh`). ⭐ **The
decision rule, committed before those fits returned:**

| null @ 16 | real arm @ 16 | reading |
|---|---|---|
| **fails K1** | fails K1 | ✅ the repair is real; **D1 must be RE-MEASURED at the repaired operating point** before it is either restored or abandoned |
| **fails K1** | **passes K1** | ⭐ the v6 latent *does* carry readable agent geometry and F-18's negative was an instrument artefact end to end |
| **PASSES K1** | either | ⛔ the "repair" is a **scoring artefact** of the smaller query set, not a repair — report it as such and do not adopt it |

### 2.7-i ⭐ THE NULL CONTROL RETURNED: **THE REPAIR IS REAL, NOT A SCORING ARTEFACT**

`RANDOM-LATENT-NULL-MATCHED@11250`, same cache-replacement construction as the parity run's, same
`n_queries 16`, same seed, same split (`raw/results_nullmatched_nq16.json`):

| @ `n_queries` 16 | **`ORC-CELLS`** (the answer) | **RANDOM-LATENT NULL** (noise) |
|---|---|---|
| `lead_gap_abs_err_m` | ⭐ **2.982** [2.337, 3.709] · median **0.816** | **9.943** [9.335, 10.572] · median 7.730 |
| **K1** vs C-CONST | ⭐ **−2.186** [−3.165, −1.192] sep ✅ **PASS** | ⛔ **+4.808** [+4.112, +5.482] sep — **FAILS, badly** |
| **K5** vs C-EPMEAN | −0.156 [−0.877, +0.643] not sep | +6.822 sep ⛔ |
| **K2** vs C-SHUF | **−3.111** sep ✅ | +0.054 [−0.090, +0.211] **not sep** — noise does not read its input, correctly |
| vs C-SHUF-XEP | **−6.076** sep ✅ | +0.291 [−0.108, +0.702] not sep |
| C-CONST / C-EPMEAN on its scored set | 5.167 / 3.137 | 5.135 / 3.121 |
| windows / clusters | 2 408 / 70 | 2 665 / 70 |

⇒ ⭐⭐ **THE REPAIR SEPARATES SIGNAL FROM NOISE BY ~7 m OF K1 AT THE SAME OPERATING POINT**
(−2.186 vs +4.808). ⛔ **The third row of the pre-registered table is REFUTED: `n_queries 16` is not
a scoring artefact.** It is the first configuration of this instrument that both **passes on the
answer** and **fails on noise** — which is the definition of a working measurement, and the
property the 74-query configuration never had.

### 2.7-ii ⭐⭐ AND THE REAL ARM RETURNED — the **first F-18 reading from a working instrument**

`v6F-SW-30k@11250`, `n_queries 16`, seed 0, same split, same estimator
(`raw/results_s11250_nq16.json`). All three arms at the **repaired** operating point:

| @ `n_queries` 16, seed 0 | err (m) | **K1** vs its own C-CONST | **K2** vs C-SHUF | vs C-SHUF-XEP | n win / clust |
|---|---|---|---|---|---|
| ⭐ **`ORC-CELLS`** (the answer) | **2.982** [2.337, 3.709] | **−2.186** [−3.165, −1.192] **PASS ✅** | **−3.111** sep ✅ | **−6.076** sep ✅ | 2 408 / 70 |
| **`v6F-SW-30k@11250`** (the latent) | **8.331** [7.318, 9.474] | **+3.217** [+2.310, +4.246] **FAILS ⛔** | +0.205 [−0.070, +0.503] **ns** | +1.225 [−0.009, +2.636] **ns** | 2 577 / 70 |
| **RANDOM-LATENT NULL** (noise) | **9.943** [9.335, 10.572] | **+4.808** [+4.112, +5.482] **FAILS ⛔** | +0.054 **ns** | +0.291 **ns** | 2 665 / 70 |

⇒ **This is the FIRST row of the pre-registered decision table: the repair is real and the real arm
still fails.** ⛔ **And it is the first evidence about the world model that this programme has ever
had from an instrument that PASSED its positive control.** At an operating point where the probe
demonstrably recovers agent geometry when it is present, **the v6 latent still loses to a constant
by 3.2 m, and both of its anti-echo controls are UNSEPARATED — exactly like noise, and unlike the
oracle, whose K2 and XEP separate by 3.1 and 6.1 m.** ⇒ **D1 points the same way it always did.**

⚠️⚠️ **BUT IT IS NOT RESTORED, AND I WILL NOT RESTORE IT HERE. Four limits, each sufficient on its
own:**
1. **ONE SEED.** §2.5 measured **3.096 m of K1 spread across three seeds** on one frozen cache. The
   real arm's +3.217 could move by more than the 1.6 m that separates it from the null.
2. **THE WINDOW SETS DIFFER** (2 408 / 2 577 / 2 665) because each arm abstains differently. Each
   **K1 is paired against that arm's own C-CONST on its own windows and is valid**; the **arm-vs-arm
   ordering above is MARGINAL, not paired**, and is not quoted as a paired result.
3. ⚠️ **EARLY-READ at 37.5 %** of training. 30 000 is the primary read.
4. **The apparatus repair itself is one seed** and has not been through the ≥3-seed replication
   §2.7 requires before it is adopted.

⇒ ⭐ **THE ACTIONABLE STATE: D1 is WITHDRAWN as filed, and the path to restoring it properly is now
short and specified** — replicate {oracle, latent, null} @ 16 queries at ≥3 seeds, re-run at 30 k,
and report K1 with the seed spread beside it. **That is ~9 fits and no trunk compute.**

### 2.7a ⛔ AND `K3` IS WORSE THAN VACUOUS — it PREFERS NOISE TO THE BEST ARM IN THIS RUN

The parity run escalated K3 as *vacuous* (τ\*-gated recall pinned at ≈0.50; a head trained on pure
noise scores **0.5002** and "passes `K3 ≥ 0.50`"). The positive control sharpens that:

| arm | K3 recall | K3 verdict |
|---|---|---|
| random-latent null *(parity run, cited)* | **0.5002** | ✅ **"passes"** |
| ⭐ `ORC-CELLS` @ 16 queries — **median error 0.816 m, K1 PASS** | **0.4432** | ⛔ **"fails"** |

⇒ ⛔ **K3 ranks a head trained on noise ABOVE the only arm in this programme that has ever passed
K1.** It is not merely a criterion that cannot fail — at this operating point it is **anti-correlated
with quality**. ⛔ **K3 must be removed from the KEEP gate, not re-thresholded.**

⚠️ **A reporting defect noticed in passing:** the field `K4_median_err_lt_0.9769m` does **not** report
the median test — `sp2_probe.py:696` computes `k4p = k1p and k2p and k3p and med < 0.9769`. The
nq-16 arm's median is **0.816 m**, which *does* clear the threshold, yet the field reads `false`
because the vacuous K3 dragged the conjunction down. **A field whose name describes one test and
whose value is a four-way conjunction will be misread.**

---

## 3. TASK 2 — IT IS NOT A RESOLUTION LIMIT, AND THE FAILURE IS UNIFORM

⚠️ **THE BRIEF'S PREMISE NEEDED CORRECTING FIRST.** `lead130_agents.jsonl` records are
`{clip_id, frame_idx, t_s, agents:[{cx, cy, yaw, l, w, occ, track_id, cls}]}` — **EGO-FRAME 3-D
GEOMETRY ONLY. There is NO image-space box in this join**, so "the fused records give image-space
extents" does not hold and no pixel extent can be *read off*. It must be **DERIVED**, and is
therefore **ESTIMATED**:

> apparent width (px) ≈ `f_ref · w_obj / cx`, `f_ref = 305.5774907364391` px (the cache meta's own
> value for the 256×640 120° cylindrical field), patch = **16 px** (`token_grid` [16, 40]).

⚠️ **HEIGHT IS ABSENT FROM THE JOIN, so pixel AREA is not derivable and is not reported.** Width is,
and width is the axis the "less than one patch" worry is about.

### 3.1 The GT lead is a LARGE object — MEASURED over the 2 721 scored windows

| | mean | median | p10 | p90 | < 1 patch | < ½ patch |
|---|---|---|---|---|---|---|
| `cx` (m) | 15.53 | 15.05 | 8.15 | 24.33 | — | — |
| **apparent width (px)** | 51.03 | **37.77** | 22.11 | 72.93 | **4.34 %** | **0.29 %** |

Class mix: automobile 2 267 · person 228 · bus 82 · rider 55 · heavy_truck 43 · trailer 43 ·
protruding_object 3.

⇒ ⛔ **THE RESOLUTION HYPOTHESIS IS REFUTED FOR THIS READOUT.** The metric asks for the *nearest*
in-corridor agent *within 30 m* — by construction a near, large object: **median 2.4 patches wide**,
and fewer than 1 in 20 windows below a single patch. The corpus-wide "median mask 52 px²" statistic
is about *all* agents; **it does not describe the ones this metric is scored on.**

### 3.2 K1 per stratum — worst where the agent is NEAREST and BIGGEST

`v6F-SW-30k@11250`, seed 0, banked head re-evaluated (`code/pc3_stratify.py`; the rebuilt headline
is asserted against `results_s11250.json` — **2 721 / 70 / 7.1688 / 5.1329, exact** — before any
stratum is printed).

| stratum | n win | n clust | arm (m) | C-CONST (m) | **K1** Δ | sep |
|---|---|---|---|---|---|---|
| `cx` < 10 m | 558 | 40 | **12.927** | 7.256 | **+5.670** [+5.117, +6.258] | ⛔ |
| 10 ≤ `cx` < 20 m | 1 478 | 64 | 6.569 | 2.523 | **+4.046** [+3.559, +4.522] | ⛔ |
| 20 ≤ `cx` ≤ 30 m | 685 | 50 | 3.773 | 9.035 | **−5.262** [−5.594, −4.919] | ✅* |
| px width < 8 (< ½ patch) | **8** | **4** | 6.493 | 13.126 | −6.633 | ⚠️ **n=4 clusters — NO CLAIM** |
| 8 ≤ px < 16 (½–1 patch) | 110 | 12 | 3.658 | 4.359 | −0.702 [−2.584, +0.525] | ns |
| **px ≥ 16 (≥ 1 patch)** | **2 603** | **70** | 7.319 | 5.141 | **+2.178** [+1.363, +3.009] | ⛔ |

\* ⚠️ **The 20–30 m "win" is not perception.** Against `C-EPMEAN` the same stratum reads
**−0.512 [−1.780, +0.608], NOT separated** — the arm merely happens to emit a constant closer to
that stratum's mean than the global constant is.

### 3.3 ⭐ THE MECHANISM — every arm emits a NEAR-CONSTANT, and so does noise

| arm | pred at `cx`<10 | at 10–20 | at 20–30 | GT means |
|---|---|---|---|---|
| `v6F@11250` | 20.68 ± 1.57 | 21.04 ± 2.23 | 20.57 ± 1.55 | **7.76 / 14.52 / 24.05** |
| `v6F@9000` | 16.78 ± 0.99 | 16.77 ± 1.13 | 16.71 ± 0.92 | ″ |
| **RANDOM-LATENT NULL** | 17.93 ± 2.82 | 18.06 ± 2.74 | 18.12 ± 2.76 | ″ |

⇒ ⛔ **THE STRATUM PROFILE IS NOT A PROPERTY OF THE LATENT.** Each arm emits one number regardless
of the scene; the per-stratum K1 is then arithmetic on where that number falls relative to the
stratum's true mean. **The random-latent null reproduces the whole profile**, which is the control
that makes this attributable. **Failure uniform ⇒ not a resolution limit, per the brief's own
decision rule.** ⚠️ And it is uniform *because the head is constant*, which — given §2 — is at
least as much a fact about the apparatus as about the latent.

---

## 4. THE FOUR METRIC FAMILIES

⛔ ADE is not reported: this is a perception readout, not a trajectory eval. Per family, with the
reason where it does not apply.

| family | served? | how / why not |
|---|---|---|
| **LONGITUDINAL** | ✅ **directly — it is the primary** | **headway** = the lead slot's `cx`, with its estimator and CI, in §2.2 / §2.6 / §3.2, per stratum and per rule. **Time-gap:** GT mean **19.256 s** over 2 719 windows. **TTC** under the 0.5 m/s physical closing floor: `ORC-CELLS` **9.101 s [7.573, 10.905], n = 544** — i.e. the oracle arm's TTC is as unusable as the real arms' (13.9–15.2 s) and no better than the null's 9.46 s. ⚠️ `taniteval.lead_metrics.distance_keeping` is **NOT** attached: it consumes a predicted ego **path** `(W,K,2)` and this is a single-frame readout that produces no path — the T1 integration prereg §4.6 defers. |
| **LATERAL** | ❌ not served | the head emits **agent** geometry, not ego path. Heading / curvature / yaw-rate / cross-track are ego quantities this readout never produces. ⭐ **However `cy` IS emitted**, and the corridor test `|cy| ≤ 1.75 m` is a lateral predicate the rule analysis in §2.6 exercises directly — the emission rate is ~1.0 at 74 queries, which is itself a lateral finding (the head puts *something* in the corridor on essentially every frame). |
| **TACTICAL** | ⚠️ **enabling condition only — NOT COMPUTED** | slot-referent agreement needs a trained goal head with the categorical `agent_slot` arg on. S-T has not run, so `GAP_TARGET` / `YIELD_AT` / `WAIT_FOR_ONCOMING` / `EVADE_IN_CORRIDOR` still index an empty set. Unchanged from the parity run. |
| **STRATEGIC** | ❌ not served, **and provably not servable from this label** | `obstacle.offline` has no map, lane graph, junction, traffic-light or route feature — 10 classes, all dynamic agents. Nothing strategic is derivable here at all. |

---

## 5. TASK 3 — THE LEVERS' DIRECTION

### 5.1 ⛔ THE BRIEF'S PREMISE IS WRONG FOR O4 — and this is the most useful part of Task 3

The brief asks for "each term's gradient … O2, O3 and O4". **O4 is not a term.**

* `o2_near_field_loss` — `stack/scripts/train_v6_staged.py:608`, weight `o2_nearfield` ✅ a loss term
* `o3_masked_cell_loss` — `:645`, weight `o3_masked` ✅ a loss term
* ⛔ `build_o4_weights` — **`:745`**, *"O4 — per-window **sampling weights** from ACTIONS ONLY"*,
  consumed at **`:2470`** by `InteractionSampler`. **There is no `o4_*` field in `V6LossWeights`.**

⇒ **O4 changes WHICH windows are drawn, not the loss on a drawn window. It has no gradient, and
"the cosine of O4's gradient with the agent-readout gradient" is not a quantity that exists.**
Reporting one would have been a units/scope error of the `df`-on-a-pod family: a number computed in
the wrong space and read as an answer. The honest analogue — implemented in `code/pc4_grad_cosine.py`
`--o4` — is (i) the correlation between a window's O4 saliency weight and its GT lead geometry, and
(ii) the cosine between the **agent-readout gradient on an O4-weighted draw** and **on a uniform
draw**. That is a statement about the data distribution, and it must never be tabulated beside a
loss-term cosine as though they were the same quantity.

### 5.2 The pairwise O-term cosines — **CITED, not re-derived**

The brief asks for these and says the precedent should be reused rather than reinvented. It is:
`…/incoming/2026-08-16-o6-ablation/raw/mask_grad_probe.json`, geometry `live_grid4`
(`readout_grid` 4, `o3_mask_rate` 0.4375 — the live run's), **5 seed pairs**, exact-linearity
estimator, controls **N1 = 0.0** and **N2 = 0.0** exactly. Mean over seeds [min, max], on the trunk
support, **chance = 1/√D = 0.001969** at D = 257 995:

| pair | mean cos | range | × chance |
|---|---|---|---|
| **O2 ↔ O5** | **+0.8700** | [+0.845, +0.904] | ~442× ⇒ **NEARLY COLLINEAR** |
| O1 ↔ O2 | +0.3211 | [+0.115, +0.446] | ~163× |
| O1 ↔ O3 | +0.1267 | [−0.208, +0.364] | ~64×, sign unstable |
| **O2 ↔ O3** | **+0.0277** | [−0.129, +0.228] | ~14×, **sign unstable** |
| O3 ↔ O5 | +0.0308 | [−0.160, +0.264] | ~16×, sign unstable |
| **O6 ↔ {O1, O2, O3, O5}** | \|mean\| ≤ **0.0424** | — | orthogonal at this D |

⚠️ **SCOPE, and it is not small:** this is the precedent's **SYNTHETIC CPU build**, not the live
v6F@11250 checkpoint (its own `meta.build` says so). It is quoted for the **relative geometry of the
O-terms**, which is what the brief asks for, and **not** as a statement about the live trunk.
⇒ **The one durable reading: O2 and O3 are nearly orthogonal to each other with an unstable sign, so
they are not redundant levers — while O2 and O5 are nearly the same direction (+0.870), so tuning
them independently is close to tuning one thing twice.**

### 5.3 O2 / O3 vs the AGENT-READOUT direction — **NOT LANDED**

⛔ **I am reporting this as NOT MEASURED rather than presenting a partial number.** The instrument
is written, follows the precedent (`…/2026-08-16-o6-ablation/code/mask_grad_probe.py`) exactly —
exact linearity `g(all) − g(term=0)`, cosines against `chance = 1/√D`, the full pairwise matrix, and
controls N0/N1/N2 that must return **exactly 0** — and the v6F@11250 trunk **loads and runs**
(336.54 M params, X3 isolation pass). It did not complete inside this run's compute window: the
dev-box 4060 was carrying the positive-control fits, and the trunk process reached **~12 GB working
set** and had to be killed by explicit PID to protect them. **What is banked is the instrument and
the scope declaration, not a result.**

⚠️ **Two scope limits are already fixed and will travel with the eventual number:**
1. `o5_k` must be reduced from the live run's **60** (and `o1_k` from **10**) — a 60-step rollout
   plus 60 future-frame encodes does not fit the box. Any cosine is therefore at a **different
   operating point** from the live objective.
2. A gradient cosine is a **local, single-checkpoint, single-head** quantity. It is **ESTIMATED**,
   never MEASURED-as-causal, and the reference direction is *the head that fails K1* — the readout
   we have, not a good one.

⭐ **AND §2 CHANGES WHAT THE ANSWER WOULD MEAN.** The reference direction is
`slot_set_loss(head(cells(z_op)), ·)` — **the objective of the very apparatus this run has just
shown cannot pass its own positive control.** A cosine against it would inherit that defect. ⇒
**Task 3 should be re-scoped to point at a readout that PASSES the control before it is run at
all.** Running it first would have produced a number, and the number would have been
unattributable. ⭐ **And §2.7 makes that re-scoping concrete rather than aspirational: a readout
that passes K1 now exists** — the same `AgentSlotDecoder` at **`n_queries` 16** — so
`code/pc4_grad_cosine.py` should be pointed at *that* head (`--head <out_orc010_nq16>/head_cells.pt
--n-queries 16`) once its ≥3-seed replication is in. **That is a one-flag change to the instrument
already banked here.**

---

## 6. ⛔ VERDICT AND ESCALATIONS — read this before any F-18 number is quoted again

1. ⛔⭐ **`D1` IS WITHDRAWN — the negative is not admissible as evidence about the world model.**
   Both 2026-08-16's D1 and 2026-08-17's *"no better than noise"* / *"the loss is at the ENCODER,
   not the readout"* rest on an apparatus that ranks **an explicit encoding of the GT boxes
   (10.18 m) BELOW random vectors (5.95 m)**. ⚠️ **This is not a claim that the latent DOES carry
   agents** — §2.3's ridge says it carries a little, and far too little to beat a constant. It is
   the claim that **we do not currently know**, and the pre-registered **D1 DROP must not be
   executed on this evidence at 30 k or at any step.** *(Owner: whoever holds
   `AGENT_SLOT_DECODER.md` §1.4 / §4.4 — I have NOT edited that document; it is another stream's.)*
   ⭐ **ACTION I TOOK, so the integrator does not have to discover it:** the withdrawal is appended
   to **`Project Steering/RETRACTION_LOG.md`** with its **root-cause class** — *"AN INSTRUMENT
   VALIDATED ONLY BY NEGATIVE CONTROLS"* — and the standing rule it earns. `CLAUDE.md` names that
   log as the programme's learning mechanism and requires the class, not just the correction; a
   withdrawal that lived only in this package would not be read before the next probe is designed.
   The file is **staged, not committed** (append-only edit; nothing else in it touched).
2. ⛔ **THE READOUT RULE IS ASYMMETRIC AND MUST BE FIXED — AND FIXING IT IS NOT ENOUGH (§2.6).**
   `gt_lead_gap` selects the nearest in-corridor agent **within 30 m**; `pred_lead` selects the
   **most confident** in-corridor slot with **no range cap**, while the decode runs to 60 m
   (`sp2_probe.py:99` vs `:141`). The fix is worth **3.2 m** on `ORC-CELLS`, lifts correlation
   +0.14 → +0.33, and makes the random-latent null **worse** (5.95 → 9.86), so it is a genuine
   repair. ⛔ **But at 74 queries it is not sufficient: with the fix AND a perfect representation the
   ceiling is a TIE with a constant** (`ORC-DIRECT` R2: +0.522 [−0.341, +1.350], not separated,
   r 0.465). ⇒ **Shipping the rule fix alone, at the incumbent operating point, would produce a
   differently broken instrument.** ⭐ **The query count is the bigger lever (escalation 5): at 16
   queries the INCUMBENT rule already passes K1.** The two are complementary — fewer queries reduce
   the clutter the rule must choose over, the symmetric rule stops the choice being made outside the
   GT's own range. ⭐ **The positive control must be re-run after every apparatus change and must
   PASS before any arm is scored.**
3. ⛔⭐ **A POSITIVE CONTROL MUST BE A STANDING REQUIREMENT OF THIS INSTRUMENT, NOT A ONE-OFF.**
   Five negative controls were run across two studies and none of them could have caught this. The
   cost is **one cache rewrite (seconds) plus one fit (~20 min)** — `code/pc1_oracle_cache.py` +
   `code/chain_a.sh` do it end to end. **This generalises past F-18:** any frozen-latent probe in
   this programme is exposed to the same class, and the same two-line move (replace the memory with
   an encoding of the label; keep everything else) applies to all of them.
4. ⚠️ **THE `tokens` READING IS THE MOST DAMAGED CONCLUSION.** *"640 raw patch tokens × 768 dims —
   240× the `cells` surface — does not rescue it, so the loss is at the ENCODER, not the readout"*
   was the parity run's strongest architectural claim. An apparatus that cannot read the answer off
   an oracle **cannot distinguish `cells` from `tokens` either**, and the `tokens` arm's headline
   (K1 +0.208, *not* separated) is inside the band the oracle and the null both occupy.
5. ⭐⭐ **THE REPAIR IS IDENTIFIED AND IT IS ONE FLAG: `--n-queries 16` (§2.7).** MEASURED: at 74
   queries the emission rate is ~1.0 and **~13 slots land in the 3.5 m corridor per window**, and
   the rule argmaxes presence over that clutter (§2.6-i). At **16** queries the *same probe* on the
   *same* oracle cache scores **2.982 m, median 0.816 m, K1 −2.186 [−3.165, −1.192] separated — the
   FIRST K1 PASS ANYWHERE IN F-18** — with the constant on the retained windows unchanged (5.167 vs
   5.133), so it is not an abstention artefact. ⚠️ **ONE SEED; replicate at ≥3 before calling 16 "the
   fix" (§2.5).** ⇒ **Recommended repair order, cheapest first:** (a) `n_slot_queries` sweep at ≥3
   seeds, (b) the symmetric readout rule (§2.6), (c) a lead-identifying criterion (§2.6b) — **and
   the positive control re-run after each.** ⚠️ **The programme's `n_slot_queries` fitting rule needs
   revisiting too:** 74 came from the in-grid *agent-count* p99, which is right for SET PREDICTION
   and wrong for the LEAD functional the metric actually scores. `code/chain_c.sh` also queues
   `ORC-DIRECT`@8 and `ORC-CELLS`@8; the chain is idempotent, so re-running it completes the sweep
   without redoing what is banked.
6. ⚠️ **TASK 3 IS NOT LANDED AND SHOULD BE RE-SCOPED BEFORE IT IS RUN (§5.2).** Its reference
   direction is the failing head's own loss. Point it at a readout that passes the control first.
7. ⚠️ **O4 CANNOT BE MEASURED THE WAY THE BRIEF ASKS (§5.1)** — it is a sampler
   (`train_v6_staged.py:745`, `:2470`), not a loss term. Any future request for "O4's gradient"
   should be redirected to the sampler analogue.
8. ⛔ **`K3` IS ANTI-CORRELATED WITH QUALITY AND MUST BE REMOVED FROM THE KEEP GATE, NOT
   RE-THRESHOLDED (§2.7a).** The parity run escalated it as *vacuous* (a noise head scores 0.5002
   and "passes `K3 ≥ 0.50`"). The positive control shows worse: the **nq-16 arm — median error
   0.816 m, K1 PASS — scores 0.4432 and "fails"**. ⇒ **K3 ranks a noise head above the only arm that
   has ever passed K1.** ⚠️ Related reporting defect: `K4_median_err_lt_0.9769m` does not report the
   median test — it is a four-way conjunction (`sp2_probe.py:696`), so the nq-16 arm's 0.816 m median
   reads `false` because K3 dragged it down. A field whose name describes one test and whose value is
   a conjunction will be misread.
9. ⚠️ **STILL OPEN FROM THE PARITY RUN, UNTOUCHED HERE:** the oracle-slot diagnostic is at chance at
   74 queries **and stays at chance under every presence gate I tried (§2.2) — it should be dropped,
   not annotated**; `C-V5F` has never
   been run; a *positive* result would need the val split, whose `obstacle.offline` join still does
   not exist; and `sp2` still banks no per-window error arrays, which is why §2.2's oracle-vs-null
   comparison is **overlapping marginal intervals, not a paired test**. ⭐ Banking those arrays costs
   nothing and would have made this run's central comparison paired.
10. ⚠️ **A REAL IMAGE-ENCODER POSITIVE CONTROL (SAM3 / DINOv2) WAS SCOPED AND NOT RUN.** SAM3 is
   local (`C:/Users/Admin/.cache/huggingface/hub/models--facebook--sam3`), the isolated venv
   `C:/Users/Admin/venvs/sam3run` has the package, and dense features are reachable in one call
   (`sam3/model/sam3_image_processor.py:59`, `state["backbone_out"] = model.backbone.forward_image`).
   ⛔ **It was deprioritised once the synthetic control FAILED**, because a realistic representation
   cannot be informative about an apparatus that an unrealistic and sufficient one already breaks.
   It becomes the right next control **after** the apparatus passes §2.

---

## 7. Test state — my repository delta is ZERO

⛔ **This run changed NO repository code.** Every script lives under
`…/2026-08-17-probe-positive-control/code/`; nothing under `stack/` or `taniteval/` was edited, and
`sp2_probe.py` was copied **byte-identically** (md5 `aabbee36fce5f164d47a555fad369cbd`) rather than
modified — the alternative readout rules in §2.6 are applied in a *separate* file to the *banked
heads' outputs*, never by editing the probe.

MEASURED with the named interpreter
(`PYTHONUTF8=1 OMP_NUM_THREADS=6 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`),
both suites, **run while the probe fits were live on the same box**:

| suite | result | baseline in the brief | |
|---|---|---|---|
| `stack` | **3782 passed · 0 failed · 7 skipped · 2 xfailed** (530.81 s) | 3782 / 0 / 7 / 2 | ✅ exact |
| `taniteval` | **1092 passed · 0 failed** (143.43 s) | 1092 / 0 | ✅ exact |

Artifacts: `raw/suite_stack.txt`, `raw/suite_taniteval.txt` (each ends with its `rc=0`).

### 7.1 ⚠️ ONE OPS TRAP HIT HERE, WORTH THE PROGRAMME'S TIME

⛔ **APPENDING TO A `bash` SCRIPT THAT IS STILL EXECUTING MAKES IT RUN THE APPENDED LINES.** `bash`
reads a script **incrementally, by byte offset**, so adding the two negative-control steps to
`chain_c.sh` while `chain_c.sh` was mid-run caused the running shell to pick them up — and the same
arm was then launched **twice concurrently**, by the old chain and by the new one, both writing the
same log (via `>`, so they truncated each other) and the same output directory. It looked like a fit
that had restarted itself.

⇒ **Never append to a live chain script. Write a NEW script.** The idempotence guard
(`[ -f raw/results_$LBL.json ] && skip`) does not help, because it is evaluated *before* either
instance finishes. Detection came from `Get-CimInstance Win32_Process` showing two PID pairs for one
arm, not from any log — the same shape as the programme's other "looks like it is running" failures.
⚠️ **No result is affected**: the duplicate ran the same seed on the same cache and the surplus
process was killed by explicit PID (never `pkill -f`, per `CLAUDE.md`).

---

## 8. Deliverable manifest

| artifact | where | only one place? |
|---|---|---|
| **this report** | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-probe-positive-control/PROBE_POSITIVE_CONTROL.md` | staged |
| ⭐ oracle-cache builder (both variants) | `repo:…/code/pc1_oracle_cache.py` | staged |
| ⭐ readout-rule analysis (the §2.6 defect) | `repo:…/code/pc5_readout_rule.py` | staged |
| ⭐ ridge disambiguation (the §2.3 second instrument) | `repo:…/code/pc6_linear_readout.py` | staged |
| stratification (Task 2) | `repo:…/code/pc3_stratify.py` | staged |
| gradient-cosine instrument (Task 3, **written, not landed**) | `repo:…/code/pc4_grad_cosine.py` | staged |
| chains + summariser | `repo:…/code/chain_a.sh`, `code/chain_b.sh`, `code/pc7_summarise.py` | staged |
| ⭐ **positive-control results — `ORC-CELLS` 3 seeds** | `repo:…/raw/results_orc010_seed{0,1,2}.json` | staged |
| ⭐⭐ **`ORC-DIRECT`** (the trivially-sufficient bound) | `repo:…/raw/results_orcdir_seed0.json` (+ `seed1`/`seed2` if the chain completed) | staged |
| ⭐⭐ **the REPAIR — `ORC-CELLS` @ `n_queries` 16, the first K1 PASS in F-18** | `repo:…/raw/results_orc010_nq16.json` | staged |
| ⭐⭐ **the repair's own NEGATIVE CONTROLS @ 16 queries** — the null and the real v6 arm | `repo:…/raw/results_nullmatched_nq16.json`, `results_s11250_nq16.json` | staged |
| n_queries sweep, remaining points | `raw/results_orcdir_nq8.json`, `results_orc010_nq8.json` — **NOT produced (stopped to free the GPU); `code/chain_c.sh` is idempotent and re-runs exactly these** | not produced |
| slot-selection clutter diagnostic (§2.6-i) | `repo:…/raw/pc5b_selection_diag.json` | staged |
| presence-gated oracle diagnostic (§2.2) | `repo:…/raw/pc5c_gated_oracle_diag.json` | staged |
| ⛔ **the D1 withdrawal, with its ROOT-CAUSE CLASS** | `repo:Project Steering/RETRACTION_LOG.md` (**append-only edit; staged, not committed**) | staged |
| oracle-cache construction metas (coverage, truncation, scaling) | `repo:…/raw/pc1_meta_orc010.json`, `pc1_meta_orcdir.json` | staged |
| ridge readouts (oracle · both v6 arms · null) | `repo:…/raw/pc6_ridge_*.json` | staged |
| readout-rule sweeps (oracle · both v6 arms · null) | `repo:…/raw/pc5_rules_*.json` | staged |
| stratification (both v6 arms · null), incl. the reproduction gate | `repo:…/raw/pc3_strata_*.json` | staged |
| roll-up + rendered tables | `repo:…/raw/SUMMARY.json`, `raw/RENDER_TABLES.md` | staged |
| chain logs (incl. the seed-2 CUDA crash) | `repo:…/raw/chain_a.log`, `raw/chain_a2.log`, `raw/chain_b.log` | staged |
| suites | `repo:…/raw/suite_stack.txt`, `raw/suite_taniteval.txt` | staged |
| ⚠️ oracle latent caches (2 × 32 MB) | `<scratch>/pc/cache_orc010/`, `cache_orcdir/` | **scratch only** — regenerable in seconds from the parity run's `cache_s11250` via `code/pc1_oracle_cache.py` |
| ⚠️ **points I STOPPED to free the shared GPU for the controls** — `ORC-DIRECT` seed 2, `ORC-DIRECT`@8, `ORC-CELLS`@8 | not produced; their partial logs are in `<scratch>/pc/log_*.txt` and the kills are visible as `rc=127` in `raw/chain_a2.log` / `raw/chain_c.log` | **re-runnable**: `code/chain_a.sh` / `code/chain_c.sh` are idempotent and will produce exactly the missing points |
| ⚠️ trained probe heads | `<scratch>/pc/out_*/head_cells.pt` | **scratch only** — the probe is the disposable part |
| ⚠️ the parity run's caches and heads I consumed | `<scratch>/sp2/cache_*/`, `out_*/` | **scratch only**, inherited; regenerable per that run's manifest |
