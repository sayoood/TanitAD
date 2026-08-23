# PRE-REGISTRATION — g2 + return-to-GT-speed term: the promotability verdict for Direction 2

**Stream D2** (`a1f26c92`). **Committed before the run.** The gentle sweep (`GENTLE_SWEEP_PREREG.md`) found
decoder-only recovery-FT is Pareto-bound (departure↓ costs ADE↑), and **localized the cost**: it lands on
**longitudinal/straight** windows while **junction ADE improves** (g2 junction dADE **+0.133**). Named next
step: add a **return-to-GT-speed / progress-preservation** term = extra L1 on the FORWARD (x) recovery
component (`--lambda-prog`; protects longitudinal without resisting the lateral recovery). Built on **g2**
(700 steps / lat_max 1.0 / yaw 3° / clean 0.5 / λ_dev 1.0 / lr 5e-5), decoder-only, frozen encoder (WM-safe).

## Arms (one thing added vs g2 = the forward-progress term)

| cfg | λ_prog | else |
|---|---|---|
| **g2** (done) | 0.0 | ftADE 0.713, dCDR +0.0057 n.s., dADE −0.125 S (worse) |
| **g2s1** | **1.0** | = g2 |
| **g2s2** | **2.5** | = g2 |

## Committed verdicts (primary = OVERALL held-out; paired episode-cluster bootstrap; base ADE 0.587)

- **✅ NET WIN → the lever is PROMOTABLE (name the config).** A config with ALL:
  1. **departure held**: overall dCDR(base−ft) **≥ +0.005 and CI excludes 0** (~≥ base/2 real reduction),
  2. **ADE recovered**: overall dADE(base−ft) **CI INCLUDES 0** (no separated regression vs base 0.587),
  3. peak_xte guard holds (dPEAK not separated < 0).
  → decoder-only recovery-FT + the progress term is a **net closed-loop improvement**; promote the config,
  mark promotable, recommend the AlpaSim confirmation (still a low-OOD lane-keeping result, not a safety rate).

- **⚠️ COUPLING SURVIVES → decoder-only is EXHAUSTED.** Neither g2s1 nor g2s2 satisfies all three (departure
  reduction lost, or ADE regression still separated). → conclude the departure↔ADE coupling is intrinsic to
  a **frozen-encoder** recovery-FT even with progress protection (the P2a mechanism: the frozen encoder does
  not encode the lateral offset, so the decoder cannot separate recover-vs-continue without global
  over-reactivity — a progress term cannot manufacture information the features lack). **The only remaining
  lever is the encoder-in-the-loop light-FT**, gated on the plan-free operative-rollout **canary** (so the
  WM cannot be silently degraded — the v4 hazard). This closes Direction 2's decoder-only phase with a
  measured reason.

## Cost / safety
Decoder-only, frozen encoder. 2 configs × (~700-step FT ≈ 10 min + held-out eval ≈ 6 min) ≈ 32 min.
`gpu_lock refc-cl-improve` tied to the sweep PID, **released on completion** (D1 queued behind). REF-C
deployed ckpt read-only; each FT writes a NEW dir. Bank each as it lands. Low-OOD LANE-KEEPING, not safety.
