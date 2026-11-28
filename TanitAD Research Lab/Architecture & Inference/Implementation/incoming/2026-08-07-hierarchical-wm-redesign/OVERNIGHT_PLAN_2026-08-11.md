# OVERNIGHT PLAN 2026-08-11 22:00Z → 2026-08-12 ~08:00Z
# "Finish v5.8f, design v6 ready-to-train, validate the VLM pilot" — PI mandate

**Mandate (PI, verbatim intent):** loop autonomously, no idle, use both pods optimally to
produce world-class results; finish ALL v5.8f improvements/analysis/implementation/docs
INCLUDING pseudo-closed-loop (T1) validation; finish the v6 architecture design + trainer
+ documentation + a paper-quality architecture diagram; finish the VLM/algorithmic
complete run after the pilot validates. **Ultimate goal: best possible v5.8f results, and
READY TO START V6 TRAINING TOMORROW.**

## 0. Resource map (the rule: neither pod idles a single minute)

| pod | role tonight | why |
|---|---|---|
| **pod5** | v5.8f completion track: p8c(BEV) → H-COTRAIN curve → **E1.4 T1 rows (pseudo-closed-loop)** → four-families rescore → W5 6 s baseline | holds the v5f/stage-A ckpts + train corpus; the release-row numbers must come from ONE pod for parity |
| **pod4** | VLM track: p8c(BEV twin) → **W7-FULL** → PH0 smoke ×3 arms → PH0 mini-pilot(8 clips)+videos → 50-clip pilot | the VLM pod (3 arms prefetched, 69 GB cache); W7-FULL rides here because pod5 owns the release-row queue |
| **dev-box** | 0-GPU, continuous: v6 architecture diagram (SVG), v6 trainer skeleton + design doc, registry/paper banking of every verdict as it lands | never blocked by GPU |

**Contention rule:** one trainer per pod at a time (enforced tonight after the duplicate
p8c race was killed at 21:47Z). Chains wait on EXIT markers, never on wall-clock.

## 1. Timeline (each item: trigger → action → gate → banking)

| # | ETA | pod | item | gate / decision |
|---|---|---|---|---|
| A1 | 22:15Z | 4 | **p8c4 BEV gate** (attempt-2: pos_weight 79.7 auto, dice, τ* sweep) | IoU retention ≥0.8× at k=10; occluded <2× visible. PASS → I1c reel frames pulled; FAIL-with-nonzero-IoU → report the τ* curve, no third attempt tonight |
| A2 | 22:45Z | 4 | **W7-FULL** (topk 256, selector-free) — *the* selection verdict, zero stale parts | ≤0.4505 → v5.8f selection story CLOSES; else read shortlist-oracle vs pick and record the roll-cost ceiling honestly |
| A3 | 23:15Z | 4 | **PH0 video-template smoke ×3 arms** | any arm that cannot ingest video FAILS PH0 outright (prereg item 6) |
| A4 | 00:30Z | 4 | **PH0 mini-pilot: 8 clips, 4 engines + overlay/BEV/text VIDEOS** | schema-valid rows + one video per clip → PI review deliverable |
| A5 | 03:00Z | 4 | **PH0 full 50-clip pilot** (if mini validates) | G1 sign-OCR ≥0.9, G2 schema ≥0.9, G3 consistency baseline |
| B1 | 22:30Z | 5 | **p8c2 BEV twin gate** (same seed → cross-pod reproducibility check) | agreement with A1 within noise; disagreement = instrument alarm |
| B2 | 23:30Z | 5 | **H-COTRAIN milestone curve** (5k/10k/15k/20k + banked 30k) + SIGReg spectrum | scene-vs-ego decodability curve across the λ_plan ramp; PR retention ≥0.8× ⇒ SIGReg validated |
| B3 | 01:00Z | 5 | **E1.4 T1 rows for v5.8f** — PSEUDO-CLOSED-LOOP, the PRIMARY tier | the release row's headline; T1 ADE + four families, action-closed |
| B4 | 03:00Z | 5 | **Four-families rescore + episode-cluster CIs** on the banked windows | completes §1.14 → RELEASE ROW |
| B5 | 05:00Z | 5 | **W5 / E-H1: 6 s baseline for v5.8f** (now REQUIRED per the 6 s spec) | ADE(6s) ≤ 3×ADE(2s) — the incumbent v6 must beat |
| C1 | 22:00Z→ | dev | **v6 architecture diagram** (paper-quality SVG: 4 layers, per-layer predictors/encoders, goal-down/latent-up wiring, 6 s trajectory band, interpretation heads) | committed + rendered |
| C2 | 23:00Z→ | dev | **v6 trainer** (`train_v6_staged.py` skeleton: S-W/S-T/S-S stages, per-stage gates, gradient-isolation matrix, label-free levers O1–O6 wired) + design doc | `pytest -q` green |
| C3 | continuous | dev | bank EVERY verdict (registry §1.13c/§1.14, fusion doc, battery doc, paper round) + HF for heavy artifacts | no artifact on one disk |
| C4 | 07:00Z | dev | **morning report + v6 GO package** for the PI | one document, all verdicts, GO/NO-GO inputs |

## 2. Success criteria (what "world-class" means tonight, measurably)

1. **v5.8f release row exists** in `MODEL_REGISTRY §1.14`: selected/oracle ADE at 0–2 s
   AND 0–6 s, four families, T1 (pseudo-closed-loop) rows, episode-cluster CIs, artifacts
   on HF — i.e. every claim tier-stamped and reproducible.
2. **The selection question is answered** (W7-FULL): either a PASS that closes the arc, or
   a measured ceiling with the mechanism named — both are results, neither is a stall.
3. **The WM-physics battery is complete enough to publish**: P8 decoded-BEV (env
   readout) + P1/P2/P3/P6/P7 + I4a load-bearing imagination → the interpretability story.
4. **v6 is READY TO TRAIN**: architecture frozen + diagram + trainer skeleton with tests +
   staged protocol + measure catalog + vocabulary v0.2 — the PI can say GO at breakfast.
5. **The VLM pipeline is validated end-to-end** on real clips with the overlay videos as
   evidence, and the full run either started or blocked only on the PI's spend decision.

## 3. Standing discipline (unchanged, restated because it is 04:00 and nobody is watching)

- Failed gates are RESULTS; instrument-vs-model is diagnosed before any verdict is quoted.
- Every number: tier stamp (T0/T1) + estimator (episode-cluster bootstrap) + artifact path.
- Registry is the only quotable source; docs cite it, never each other.
- Never idle: if a headline item is gated, drop to the next unblocked item IN THE SAME
  turn (the pull-list is §1's table plus `BACKLOG.md`).
- Re-arm the wake ONLY after something was executed.
