# PRE-REGISTRATION — RefcCL: encoder-in-the-loop closed-loop-aware training of REF-C

**Arm name: RefcCL** (Sayed-greenlit 2026-07-24). Stream D2's banked "sole promotable path." **Committed
before the run.** Direction-2 phase-1/2 MEASURED that in-envelope geometric recovery augmentation is a real,
data-efficient, generalizing lane-departure lever, but **decoder-only over a frozen encoder it is
Pareto-bound** (departure↓ costs ADE↑), because the **frozen encoder does not encode the lateral offset**
(P2a recovery_ratio ~0; twice-confirmed by the gentle + speed-term sweeps). RefcCL unfreezes the encoder so
the off-path *features* can separate — the test is whether that unblocks the trade **without wrecking the
representation** (the v4 WM-degradation hazard).

## Arm (built + smoked): `recovery_aug_ft.py --unfreeze-encoder-stages k --lr-encoder 5e-6`
Start from the **g2** recovery-aug settings (the least-ADE-cost Pareto point): steps 700, lat_max 1.0, yaw 3°,
clean 0.5, λ_dev 1.0, **lr_head 5e-5**, λ_prog 0. Change vs g2 = **unfreeze the last-k ResNet encoder stages**
at **lr_encoder 5e-6** (10× below head — the warm-trunk lesson from v4.1); **BN stays frozen** (eval-mode
running stats — the safe FT). Decoder-only path is byte-identical when k=0.
- **refccl_s1** = unfreeze **last 1** stage (deepest; ~the lightest touch). Run FIRST.
- **refccl_s2** = unfreeze **last 2** stages (more capacity) — run only if s1's canary HOLDS (else more blocks
  = more hazard; back off instead).

## Eval (each config): held-out corridor + ADE + the CANARY
- **Planner:** `eval_corridor_split.py` on held-out 28:40 (paired episode-cluster bootstrap) — corridor_
  departure_rate@1.75 m + closed_ade2s + peak_xte, vs base REF-C.
- **Encoder-integrity CANARY (plan-free, label-free):** `encoder_canary.py` on held-out 28:40 — `feat_cos`
  (readout alignment vs the frozen base encoder), `rel_l2`, `man_agree` / `route_agree` (does the encoder
  still support the aux tasks the WM depends on). **Gate:** HOLDS = feat_cos ≥ 0.90 AND man_agree ≥ 0.80 AND
  route_agree ≥ 0.80; DEGRADED = feat_cos < 0.85 OR man_agree < 0.70 OR route_agree < 0.70.

## The THREE committed verdicts (primary = OVERALL held-out; base ADE 0.587)

- **(a) ✅ NET WIN + canary HOLDS → RefcCL is PROMOTABLE.** ALL of:
  1. **departure held**: overall dCDR(base−ft) **≥ +0.005 & CI∌0**,
  2. **ADE recovered**: overall dADE(base−ft) **CI∋0** (within noise of base 0.587),
  3. peak_xte guard holds (dPEAK not separated < 0), **and**
  4. **canary HOLDS** (encoder representation intact).
  → the frozen-encoder bottleneck **was** the cause and touching the encoder unblocks it **safely**. Name the
  config **RefcCL-s{1,2}**, mark PROMOTABLE, flag a MODEL_REGISTRY entry, and recommend the AlpaSim
  confirmation (still low-OOD lane-keeping, not a safety rate).

- **(b) ⚠️ CANARY DEGRADES → the encoder-unfreeze hit the v4 hazard.** Any canary threshold tripped (regardless
  of corridor/ADE — a departure win bought by wrecking the WM is not a win). → report the degradation and
  **back off**: fewer stages (k→k−1), lower lr_encoder (→1e-6), or add a frozen-teacher feature-distillation
  pin. This is the pre-registered "unfreeze is unsafe at this setting" branch.

- **(c) ⚠️ NO NET WIN even with the encoder unfrozen (canary HOLDS).** Canary intact but corridor/ADE still
  Pareto-bound (dCDR CI∋0, or dADE separated-worse). → the departure↔ADE trade is **not** merely a frozen-
  encoder artifact; the lever needs a different mechanism (closed-loop-consistency training on real on-policy
  rollouts once a low-OOD renderer exists, or a longer/full-encoder co-train). Report the residual + the
  encoder-drift (canary numbers) as the diagnostic.

## Cost / safety
Decoder + last-k stage light-FT; ~700-step FT (~12 min, the encoder-in-loop adds a backward through the
unfrozen stage) + eval (~6 min) + canary (~1 min) per config. `gpu_lock acquire refccl` tied to the sweep
PID, **released on completion**. ⚠️ **light-FT — finish well before the from-scratch 10k gate (~16h) needs
eval**; checkpoint + release if that approaches. Do NOT touch pod2 (from-scratch) / pod3 (IDM) / pod1
(better-planner). REF-C deployed ckpt read-only; each FT writes a NEW dir. Low-OOD lane-keeping, not safety.
