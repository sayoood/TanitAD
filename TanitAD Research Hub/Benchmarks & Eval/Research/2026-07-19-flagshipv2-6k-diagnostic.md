# flagship-v2 @6k — why 6.18 m ADE@2s, and continue-vs-restart (mechanism evidence)

**Date:** 2026-07-19 · **Discipline:** Benchmarks & Eval · **Compute:** eval pod (A40) only, pod2 read-only · **Status:** decision-grade · nothing committed to stack

Sayed's question: v2@6k evals **6.18 m ADE@2s** (~7-14× behind the v1 lineage). Is this
**A (benign — decorr/gradscale removed the kinematic speed shortcut, v2 must relearn
speed from vision, will converge)** or **B (malign — a lever is broken/mis-scaled, will
not converge, restart)**? Continue to 30k or restart with an optimized lever setup?

## TL;DR verdict

**Mechanism is A (confirmed).** The 6.18 m is ~80 % longitudinal, concentrated entirely
in the high-speed stratum, produced by an **accelerating over-prediction in the operative
latent rollout**; the encoder's linear speed content has collapsed by design
(probe R² 0.30 vs v1 0.86). Lateral/steering is **not** broken (sharp-curve stratum tracks
CV; the *straight* stratum is the worst = pure speed error). No single lever is
catastrophically broken (anchored decoder converging, no NaN, no gnorm spike) — so it is
**not** pure B.

**But A does not rescue "continue to 30k".** The decisive rate evidence says the gap is
**not closing** — the same-step v2/v1 ratio on the forward-consistency metric is flat-to-
widening (~2.5→4.3×), v2's learning-curve exponent is **−0.50 vs v1's −0.84**, and a
power-law projection puts **v2@30k ≈ 0.27 m** forward-consistency (0.4 s) vs v1's actual
**0.030 m** — i.e. ~9× worse at the same budget, still far above CV. The simultaneous-
levers recipe made the optimization too hard; grinding to 30k burns ~4 days to land short
of where v1 already sits.

**Recommendation: RESTART with staged/softened encoder-grounding levers** (decorr after
10k warmup @0.02; rollout-k 4→8 ramp; invdyn_gradscale 0.25→0.5; fa_dropout 0.3→0.15
early). Keep the decode-side levers (anchored tactical, gated intent, labels-v2, jerk,
ego→planners, route-vis) from step 0 — they are healthy. **10k gate (below) is the
trigger**; the v2@10k rolling ckpt lands in ~9 h and settles this cheaply.

---

## 0. First: the train-telemetry vs eval "paradox" is a horizon artefact (not the bug)

The worry that "training says `g_op_fwd_ade_m`=0.50 m but eval says 6.18 m" is **not** a
contradiction. In `level_cfg`, the operative level is `op: [[1,2,4], 4]` → the training
`g_op_fwd_ade_m` is a **4-step / 0.4 s** forward-consistency ADE. Eval **ADE@2s** rolls the
same op readout **20 steps**. Different horizons. My offline diagnostic decodes the op
readout to 20 steps and reproduces the harness exactly (`ade2s_op` = 5.94 = harness
`full_set` 5.94), so the harness number is faithful — the gap is a real 20-step rollout
failure, not a metric bug.

---

## 1. The eval anchors (harness, physicalai-val, op path, 20-step)

| model | step | ADE@2s (heldout) | de@0.5s | note |
|---|---|---|---|---|
| **flagship-v2** | **6000** | **6.179** | **1.239** | 4-lever-stack v2 |
| flagship-30k (v1 final) | 29999 | 0.452 | 0.076 | v1 recipe |
| flagship-speed (19k relay) | 19000 | 0.628 | 0.116 | v1 recipe, mid |
| flagship-nospeed | 22000 | 2.918 | 1.049 | no-speed ablation |
| **CV baseline** | — | **0.838** | 0.129 | constant-velocity |

v2@6k is **7.4× CV** and even its **0.5 s** error (1.24 m) already exceeds CV's full 2 s
error — the divergence starts immediately, not just at the 2 s tail.

---

## 2. DIAGNOSTIC 1 — speed content: encoder lost it (by design), op path hasn't compensated

Offline per-episode-held-out ridge probes on the same 40 val eps (`diagv2_summary.json`):

| quantity | v2@6k | v1@30k | 19k-relay | reading |
|---|---|---|---|---|
| **encoder z_t → v0 probe R²** | **0.300** (λ=10) | 0.861 | 0.858 | encoder speed content **collapsed** |
| **op step-1 decoded speed R²** | **0.723** | 0.9987 | 0.9963 | even 1-step decode weakly tracks v |
| op step-1 speed err (m/s) | +2.39 | +0.06 | +0.13 | over-predicts from the first step |

The encoder half of the story is exactly what **decorr was designed to do** — strip linear
ego/speed from the latent (H25). The problem is the **operative path has not compensated**:
rolled 20 steps, v2's predicted per-step speed **dips then ramps monotonically** 12.9 → 23.9
m/s while GT is flat ~12.7 m/s:

```
rollout step:     1     3     5     10    15    20     (GT flat ≈12.7 m/s)
v2 speed_pred: 15.1  12.9  13.9  17.6  21.1  23.9    → err +2.4 … +11.1 m/s
v1 speed_pred: 12.8  12.7  12.7  12.9  13.1  13.1    → err  +0.06 … +0.41 m/s
```

The op readout inflates displacement as the latent rolls. v1@30k tolerates *more* latent
drift (‖z‖ grows 40→148 over 20 steps vs v2's 40→57) yet still decodes correct
displacement — so v1's readout is **robust to drift**; v2's readout is **not yet grounded**.
This is the A mechanism made concrete: speed was removed from the encoder and the
op-rollout has not relearned to hold it.

---

## 3. DIAGNOSTIC 3 — decompose the 6.18 m: ~80 % longitudinal, all at high speed

Error in the GT track frame, op path, by waypoint (0.5/1/1.5/2 s):

| component | 0.5 s | 1 s | 1.5 s | 2 s |
|---|---|---|---|---|
| \|longitudinal\| | 1.06 | 2.98 | 6.51 | **11.52** |
| \|lateral\| | 0.14 | 0.57 | 1.48 | 3.06 |
| signed long (mean) | +0.40 | +1.95 | +5.15 | **+9.74** (overshoot) |

At 2 s the along-track term is **79 %** of the L1 budget and **positive** = the model drives
too far. Speed strata + curvature strata make it unambiguous:

| stratum | v2 ADE@2s | v2 signed long@2s | CV ADE@2s |
|---|---|---|---|
| **high speed** | **11.40** | **+23.7 m** | 0.65 |
| med speed | 3.42 | +4.9 | 0.93 |
| low speed | 2.99 | +0.53 | 0.93 |
| **straight** (no lateral demand) | **6.68** | — | 0.44 |
| gentle curve | 4.62 | — | 1.36 |
| sharp curve | 3.43 | — | 2.38 |

The **straight** stratum (pure longitudinal — no turning required) is the *worst* absolute,
and **sharp curves** are the *closest* to CV. If steering/lateral were structurally broken
(Hypothesis B), sharp curves would be the worst. They are the best. **Lateral is clean;
the failure is high-speed longitudinal over-prediction.** → A, not B.

---

## 4. DIAGNOSTIC 2 (decisive) — the rate is NOT closing the gap

Same-step `g_op_fwd_ade_m` (0.4 s forward-consistency), both `train_log.jsonl`, 1k buckets:

| step | v1 | v2 | v2/v1 |
|---|---|---|---|
| 0–1k | 0.954 | 1.443 | 1.51 |
| 1–2k | 0.484 | 1.222 | 2.53 |
| 2–3k | 0.421 | 0.984 | 2.34 |
| 3–4k | 0.267 | 0.862 | 3.23 |
| 4–5k | 0.288 | 0.743 | 2.58 |
| 5–6k | 0.201 | 0.627 | 3.12 |
| 6–7k | 0.140 | 0.604 | 4.33 |
| 7–7.5k | 0.186 | 0.609 | 3.27 |

- The ratio is **flat-to-widening** — the gap is **not** closing. (Parallel-at-best, mildly
  diverging.)
- v1 reached v2's *current* 7.5k value (0.61 m) at **step ~250** → v2 is ~**30× slower** to
  the same forward-consistency on this metric.
- **Power-law fit (1.5k–7.5k), extrapolated:**

| metric | v2 exp | v2 @10k | v2 @30k | v1 actual @30k |
|---|---|---|---|---|
| g_op_fwd (0.4 s) | −0.50 | 0.474 | **0.273** | **0.030** |
| g_str_fwd | −0.46 | 1.906 | **1.154** | **0.161** |

v1's exponent is **−0.84** (steeper). Even a generous power-law puts **v2@30k ≈ 9× worse
than v1@30k** on forward-consistency; mapping to the 20-step eval, v2@30k projects to
**~1.5–2.5 m ADE@2s — still above CV 0.84 and 3–5× v1's 0.45.** Continuing to 30k does not
reach parity within budget.

---

## 5. DIAGNOSTIC 4 — per-lever telemetry: no single break, but a collective drag

| lever | signal (0k→7.5k) | verdict |
|---|---|---|
| **anchored tactical** | cls 3.04→1.42 ↓, wta 0.069→0.039 ↓, n_modes ~10–11 stable, conf_norm 78→214 ↑ | **converging, healthy** — multimodal, not collapsed |
| **encoder-ego decorr** | decorr loss flat ~0.08 (not ↓), in-sample ego_r2 **rising** 0.41→0.79 | penalty **not winning**; weight 0.05 too weak to reduce ego decodability, but **strangles speed capacity** (offline probe 0.30) — a **drag** |
| **rollout-k 12** (v1=4) | `roll` starts 14.3 (6× v1's 2.36), converges to 0.16 | harder early objective; op latent is what over-inflates — **contributes to slow start**, not broken |
| **invdyn_gradscale 0.25** | (with decorr) double-decouples ego/metric grounding from encoder | compounds the speed-content collapse — **drag** |
| fa_dropout 0.3 | forces imagination while speed is being relearned | two hard problems at once — **early drag** |
| gnorm / NaN | none observed; only a transient bf16 `linalg.solve` dtype crash at launch (already fixed, run resumed clean) | **no instability** |

No NaN, no gnorm spike, anchored decoder trains cleanly → **rules out pure B**. The four
levers that touch the **encoder's speed/metric grounding** (decorr, invdyn_gradscale,
fa_dropout, rollout-k) **jointly** strip and starve the very signal (longitudinal speed)
the eval punishes — that is why A is real *and* why convergence is too slow.

---

## 6. Verdict

- **Hypothesis A — CONFIRMED as the mechanism.** 80 % longitudinal, high-speed-only,
  accelerating op-rollout overshoot, encoder speed content collapsed, lateral clean,
  straight-road worst. This is speed-relearning, not structural breakage.
- **A is benign in mechanism but not in schedule.** The rate is flat-to-widening, the
  learning-curve exponent is shallower (−0.50 vs −0.84), projection lands v2@30k short of
  both v1 and CV. The simultaneous-levers stack made the optimization landscape too hard.
- **Not B** (no broken lever), so a restart does **not** mean abandoning the levers — it
  means **staging** the four that compete for encoder speed capacity so speed locks in
  first.

**→ RESTART with staged levers.** Do not grind v2 to 30k.

---

## 7. The 10k gate (trigger, lands in ~9 h)

Run the harness op-path on the v2@10k rolling ckpt (`ade@2s` heldout) + re-run
`diag_v2mech.py flagship-v2-10k` for the mechanism probes. **Continue only if ALL hold;
any failure triggers the staged restart:**

| gate | continue-to-30k threshold | restart trigger | why this number |
|---|---|---|---|
| **primary — eval ADE@2s (op)** | **≤ 2.5 m** | > 2.5 m | ≈3× CV; the *floor* from which a −0.84-style tail could still reach parity by 30k. Projection says v2 will be ~4–5 m → expected to **fail** |
| **mechanism — encoder speed probe R²** | **≥ 0.55** | < 0.55 (still ~0.30) | must be recovering toward v1's 0.86; still-flat = decorr strangling speed |
| **high-speed long overshoot @2s** | **≤ 8 m** | > 8 m (now +23.7) | the actual failure mode must be visibly collapsing |

Expectation on the current trajectory: v2@10k **fails the primary gate (~4–5 m)** → restart.
The gate exists to make that call on evidence, not projection.

---

## 8. The staged/optimized restart setup (each change justified from §2–5)

Keep the same 12-lever intent; change only the **schedule** of the encoder-grounding four:

1. **decorr: OFF until 10k warmup, then weight 0.05 → 0.02.** Evidence: offline probe 0.30
   (encoder lost speed) while decorr loss is flat and ego_r2 *rose* — the penalty isn't even
   reducing ego decodability, it is pure capacity-drag during the critical early speed-
   learning window. Let speed lock in, then decorrelate gently.
2. **rollout-k: ramp 4 → 8 → 12** (not 12 from step 0). Evidence: `roll` started 6× v1's and
   the op latent is exactly what over-inflates on long rollout; a shorter early rollout
   stabilizes the speed decode before deepening.
3. **invdyn_gradscale: 0.25 → 0.5.** Evidence: v1's step-1 speed R² 0.9987 came from full
   metric grounding into the encoder; 0.25 (with decorr) double-decouples it and the speed
   content cratered. 0.5 restores enough grounding gradient that displacement tracks speed.
4. **fa_dropout: 0.3 → 0.15 early, restore to 0.3 after 10k.** Evidence: forcing imagination
   while relearning speed is two hard problems at once; the op path over-predicts partly
   because it cannot lean on given controls early.
5. **Keep from step 0 (all healthy):** anchored tactical (cls/wta converging, n_modes
   stable), gated intent, labels-v2, jerk, ego→planners, route-from-vision. These are not
   the drag.

Net intent: **stage the ENCODER-grounding levers, keep the DECODE-side levers.** This
preserves every architectural gain while letting the longitudinal speed signal — the entire
measured failure — establish before it is decorrelated away.

---

## Provenance / reproducibility

- Eval-pod artefacts: `/root/diag_v2mech.py`, `/root/taniteval/results/diagv2_summary.json`,
  `diagv2_flagship-v2-6k.pt`, `diagv2_flagship-30k.pt`, `diagv2_flagship-speed.pt`;
  harness result `flagship-v2-6k.json`; run under
  `TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2`.
- pod2 (read-only, gentle scp): `flagship4b-v2-30k/{train_log.jsonl,config.json}` (to step
  7550), `flagship4b-speedjerk-30k/train_log.jsonl` (v1, to 30k).
- Local analysis: `scratchpad/v2diag/{compare_logs.py,trend.py}`.
- Stack refs: `stack/tanitad/train/decorr.py` (ego decorr + probe),
  `stack/tanitad/models/fourbrain.py` (AnchoredTacticalDecoder),
  `stack/tanitad/models/metric_dynamics.py::{rollout_decode,grounding_losses}`,
  `stack/tanitad/train/flagship_losses.py` (lever wiring, `invdyn_gradscale`).
- Compute: A40 eval pod, ~3 model diagnostic passes (~40 s harness + ~3 min/model diag),
  wall ~25 min; cost $0 (standing pod). pod2 untouched except two small text scp reads.
