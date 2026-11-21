# Data Engineering — Standing Goals (D-029)

1–3 measurable objectives with a target number + deadline. Each run advances one with a measured step;
a goal with no movement for two runs is escalated in STATE, not silently carried.

## G1 — Unblock the license-clean OWNED real-urban tier (OWN_DATASET_PLAN §7)
- **Target:** ≥2 owned real-urban corpora (ZOD CC-BY-SA + PandaSet CC-BY) ingesting to the contract with
  `drop_in=True` (achieved f_eff = 266 ± 5%) and episode-contract PASS **on real bytes**.
- **Blocking dependency (surfaced 2026-07-15):** D-016 R1 **pad/letterbox crop + undistort** in
  `stack/tanitad/data/calib.py` — square-crop is height-bound on 16:9 frames (any `fx>1122 px` on a 1080-tall
  frame), so PandaSet lands f_eff=467 and ZOD's fisheye + PandaSet's k1=−0.589 distortion need handling. Filed
  via the PandaSet intake; needs MVP integration.
- **Deadline:** 2026-08-15 (Phase-0 data close).
- **Status (2026-07-15):** PandaSet loader shipped + **fails loud** on the blocker (16✓); the blocker is now
  quantified and promoted from "R1 nicety" to blocking. Next: land the pad-crop → flip PandaSet drop-in →
  fetch+verify one real ZOD drive + one PandaSet sequence. **Movement: yes.**
- **Status (2026-07-17):** blocking dependency **RESOLVED for the pinhole family** — `pinhole_rectify` built +
  validated (intake `2026-07-17-d016-r1-pinhole-rectify/`, 9✓): PandaSet **467→266.0 exact drop-in** (synthetic
  bytes; 37.7% masked periphery measured), comma reference untouched, fisheye family (ZOD) covered by existing
  `ftheta_*`. Every owned real-urban source now has a proven rectify path. **Remaining to close G1:** MVP
  integration + `drop_in=True` + contract-PASS on **real** PandaSet & ZOD bytes (2 corpora). **Movement: yes
  (geometry blocker cleared; real-bytes verification is the last mile).**

## G2 — Close the H7 data-efficiency loop (pose-less → trainable)
- **Target:** a working inverse-dynamics (IDM) head that pseudo-labels a pose-less corpus (WorldModel-Synth
  264k-clip long-tail, or YouTube dashcam), measured by **action-agreement r ≥ 0.6 vs real CAN** on a held-out
  labelled set (comma2k19/ZOD).
- **Deadline:** Phase-1 kickoff (needs the 30k-class encoder; ~2026-09).
- **Status (2026-07-15):** WorldModel-Synth **confirmed pose-less** → this is now its ONLY action-path; the
  usable-now semantic-label index (captions+metadata) is the interim value. Literature recipe identified
  (frozen-encoder IDM+WM on unlabelled video). **Movement: dependency clarified; head not yet built.**

## G3 — Data-side attack on the top program risk (single-camera driving-capability gap, D1 sub-metre)
- **Target:** deliver the owned real-urban mix (ZOD+PandaSet, ~35% of the mix per plan §6.2) and measure whether
  real EU/urban diversity moves the flagship **D1 straight-stratum ADE** vs the comma+physicalai baseline.
- **Deadline:** 2026-09-01 (after G1 unblocks the corpora).
- **Status (2026-07-15):** gated on G1. No movement this run (first run this goal is recorded) — carried, not
  escalated (fresh).
- **Status (2026-07-18): MOVEMENT — the baseline this goal measures against is now quantified.** Curve-rebalance
  measured the comma+PhysicalAI baseline on 630 real eps (intake `2026-07-18-curve-rebalance/`, 12✓): comma
  **83.1%** straight, PhysicalAI **56.0%**, natural pool **63.9%** (D1 eval strata). This establishes the
  straight-fraction denominator the ZOD/PandaSet urban add must move, AND localizes the pathology to comma/highway
  (urban already ~56%). Two quantified levers (source-mix + turn-weight β) shipped. **Remaining for G3:** land the
  ZOD/PandaSet real bytes (gated on G1 access), re-measure per-urban-corpus contribution, and — via an escalated
  proposal — see whether the urban mix moves the flagship D1 straight-stratum ADE. **Movement: yes (baseline +
  recipe measured; corpus delivery still gated on G1 access).**
