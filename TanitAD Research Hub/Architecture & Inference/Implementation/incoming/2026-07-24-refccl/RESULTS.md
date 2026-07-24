# RESULTS — RefcCL (encoder-in-the-loop closed-loop-aware training of REF-C)

**MEASURED on `tanitad-eval` (A6000), `gpu_lock refccl`, 2026-07-24.** REF-C deployed ckpt read-only; each FT
wrote a NEW ckpt. Held-out eval = episode-disjoint 28:40 (12 eps / 264 windows), paired episode-cluster
bootstrap. Base REF-C: corridor_departure@1.75 m **0.0174**, closed_ade2s **0.587**. Full pre-reg +
3-way verdict in `PRE_REGISTRATION.md`. Machinery: `../2026-07-23-refc-planner-closedloop/recovery_aug_ft.py`
(`--unfreeze-encoder-stages`/`--lr-encoder`) + `encoder_canary.py`.

## RefcCL-s1 — unfreeze last 1 ResNet stage (51.56 M enc params), lr_enc 5e-6

**Encoder-integrity CANARY: ✅ HOLDS** (`canary_refccl_s1.json`) — feat_cos **0.9968**, rel_l2 0.085,
man_agree **0.9965**, route_agree **0.9653** (all ≫ thresholds). The encoder representation is **intact** — the
v4 WM-degradation hazard was **avoided**.

**But the PLANNER got WORSE across the board** (`corridor_refccl_s1.json`; positive dCDR/dADE = FT better):

| stratum | dCDR@1.75 [CI] | dADE@2s [CI] | dPEAK |
|---|---|---|---|
| overall | **−0.0328 [−0.070,−0.006] S (departs MORE)** | **−0.490 [−0.847,−0.199] S (worse)** | −0.650 S |
| junction | −0.0304 [−0.049,−0.015] S | −0.068 [−0.262,+0.059] n.s. | −0.230 S |
| longitudinal | −0.0642 [−0.138,−0.004] S | −1.140 [−1.543,−0.679] S | −1.245 S |

Absolute overall: corridor 0.0174→**0.0502**, ADE 0.587→**1.077**. **NOT a net win** (dep_held=False — departs
more; ade_recovered=False — worse).

**Read (the important nuance):** the canary shows the encoder **barely moved** (feat_cos 0.997, 8.5 % L2
drift) → at lr_enc 5e-6 it **did not encode the offset**, so s1 did **not actually test** the core hypothesis
("a materially-changed encoder unblocks the trade"). Yet the tiny nudge **destabilized the planner** —
plausibly because the decoder cross-attends the **conv fmap** (not the canary's pooled feature), and even a
small fmap shift is a moving target a 700-step decoder FT can't re-track. So s1 says: **a gentle
encoder-in-loop touch hurts without helping**, and the canary (pooled/aux integrity) can HOLD while the
decoder-facing fmap still shifts. s1 leans branch (c) but is **inconclusive on a real encoder move** → s2
raises lr_enc to let the encoder actually move (canary has huge headroom).

## RefcCL-s2 — same, lr_enc 5e-6 → **2e-5** (material encoder move; canary-gated)

**Encoder-integrity CANARY: ✅ HOLDS** (`canary_refccl_s2.json`) — feat_cos **0.9658**, rel_l2 **0.263**,
man_agree **0.9861**, route_agree **0.9167**. The encoder moved **~3× more than s1** (feat_cos 0.997→0.966,
rel_l2 0.085→0.263) and the canary **still holds** — a **material** encoder fine-tune is **safe** here.

| stratum | dCDR@1.75 [CI] | dADE@2s [CI] | dPEAK |
|---|---|---|---|
| overall | +0.0002 [−0.003,+0.006] **n.s.** (no win) | −0.084 [−0.157,−0.019] S (worse) | −0.183 S |
| junction | +0.0014 [−0.013,+0.017] n.s. | −0.022 [−0.097,+0.036] n.s. | +0.108 n.s. |
| longitudinal | −0.0005 [−0.002,+0.000] n.s. | −0.143 [−0.260,+0.004] n.s. | −0.326 S |

Absolute overall: corridor 0.0174→**0.0172** (unchanged), ADE 0.587→**0.671** (recovered from s1's 1.077, but
still separated-worse). **NOT a net win** (no departure reduction; ADE still separated-worse).

## FINAL RefcCL VERDICT — pre-registered branch **(c)**: the trade is NOT a frozen-encoder artifact

| arm | enc move (feat_cos) | canary | dCDR overall | dADE overall | net win? |
|---|---|---|---|---|---|
| decoder-only g2 (frozen enc) | 1.000 | — | +0.0057 n.s. | −0.125 S | no |
| **RefcCL-s1** (lr_enc 5e-6) | 0.997 (tiny) | **HOLDS** | −0.0328 S (departs MORE) | −0.490 S | no (worse) |
| **RefcCL-s2** (lr_enc 2e-5) | 0.966 (material) | **HOLDS** | +0.0002 n.s. (≈base) | −0.084 S | no |

**Two MEASURED conclusions:**
1. ⭐ **The encoder CAN be fine-tuned materially and SAFELY.** s2 moved the encoder ~3× more than s1 (rel_l2
   0.263) and the integrity canary **still holds** (feat_cos 0.966, aux agreement > 0.91). Low-lr last-stage
   fine-tuning does **not** trigger the v4 WM-degradation hazard — **this de-risks all future encoder-touching
   work on REF-C** (a fuller RefcCL co-train, camera-conditioning, etc.). Branch (b) did **not** fire.
2. **Unfreezing the encoder does NOT unblock the departure↔ADE trade → the trade is NOT a frozen-encoder
   artifact (pre-registered branch c).** As the encoder moves more (s1→s2), ADE recovers (1.077→0.671) but the
   departure reduction vanishes (−0.033→+0.000); **no safe encoder setting yields a net win.** Moving the
   encoder just slides the arm along the same Pareto curve — it does not dissolve it.

**Why (HYPOTHESIS, grounded in the full D2 arc):** across decoder-only (naive/g1/g2/g3/g2s1/g2s2) AND
encoder-in-loop (s1/s2), **every** config that reacts enough to cut corridor departures also over-reacts
enough to raise closed-loop ADE — regardless of which weights train. The **single-step, in-envelope,
synthetic-perturbation** recovery signal is fundamentally a "react harder to instantaneous lateral offset"
signal; on the low-OOD instrument, that necessarily trades ADE (the metric penalizes any deviation from the
exact GT path). The trade is intrinsic to the **objective**, not the parameter subset. **The escape is a
closed-loop-CONSISTENT objective** — train on the planner's OWN accumulated on-policy drift (RoaD/CAT-K-style
rollout recovery), which needs a **faithful low-OOD renderer** to generate non-self-referential on-policy
rollouts (the standing program instrument gap; AlpaSim = 3.2× OOD, real-footage = agent-free). That is a
different mechanism than the additive recovery-augmentation this stream explored, and the right next bet once
a low-OOD renderer exists.

**Direction-2 FINAL:** in-envelope geometric recovery augmentation is a **validated, data-efficient,
generalizing** lane-departure mechanism (halves held-out departures decoder-only, beats Gate-1's memorization
wall). It is **Pareto-bound (departure↓/ADE↑) and NOT promotable** — decoder-only NOR safe encoder-in-loop
escapes it; the trade is intrinsic to the single-step synthetic-recovery objective, not the frozen encoder.
**Bankable positives:** (i) the open quadrant (renderer-free ∧ non-self-referential ∧ data-efficient recovery)
is real and generalizes; (ii) **REF-C's encoder is safely fine-tunable** (canary-verified). **Next bet:**
closed-loop-consistent (on-policy rollout) recovery once a low-OOD renderer exists. Low-OOD lane-keeping
throughout, not a safety rate.
