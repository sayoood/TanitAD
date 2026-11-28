# PREREG_SSGUARD_S1 -- does GUARD-ALL replicate at inference seed 1? (Master Mind, 2026-10-26 ~03:10 CET)

Sealed BEFORE the seed-1 SELFSHARP roll exists. One GPU roll (os-only, 4,754 windows, the same EMA-48,852 checkpoint, seed 1).

## 0. What is known
* `PREREG_SSGUARD.md` (2b9d1827) read **PASS (GUARD-ALL)** at inference seed 0: MD -0.0798 [-0.1317, -0.0357], FD +0.0108
  [-0.0165, +0.0358], no S6 row separated worse, ADE -0.1090 [-0.2105, -0.0153], FDE -0.3410 [-0.6152, -0.0932]. Its committed next
  step is this replicate: iCEM-style sampling makes the same checkpoint give a different answer per inference seed, so a one-seed
  PASS is necessary, not sufficient.
* The labels are seed-invariant: z_tac is bit-identical across inference seeds (L1 K6) and `windows_s1.json` == `windows_s0.json`,
  so `labels_selfsharp_s0.npz` is the seed-1 force too.

## 1. Arm, population, estimator
GUARD-ALL exactly as `PREREG_SSGUARD.md` section 1, built from the seed-1 SELFSHARP roll and L1's seed-1 os-only bank (BASE-os s1).
S6 compliant at seed 1, stage on; paired episode-cluster bootstrap over EPISODES B 2000 seed 0; one training seed; T1 open loop.

## 2. Controls
* **K-NONE:** BASE-os s1 MD / FD reproduce L2's seed-1 rates (0.47105 / 0.13739) to 5 dp.
* **K-ID / K-SSID:** the identity reconstructions as `PREREG_SSGUARD.md`.
* **K-HOOK:** the roll's counters show all 4,754 windows forced, none unarmed, batch 1.
* (K-IND is a property of the roll code and was measured at seed 0; it is not re-measured.)

## 3. Bars and outcomes
* **REPLICATED** = P1 + P2 + G-strict on GUARD-ALL vs BASE-os at seed 1 -> GUARD-ALL is a two-inference-seed result. Next: register
  it in GOALS_AND_CLAIMS, put the factored decode (native path x self-sharpened progress) to the PI as a refcv8 STAGE candidate
  (SHIP-R8 battery) and as refcv9's decode design (lon class drives the speed profile only).
* **NOT REPLICATED** (any bar fails) -> name the bar; the seed-0 PASS is reported as one-seed only and the cross-seed spread is
  read against the inference floor (|MD s0 - s1| of BASE-os = 0.0016); next lever: the temperature roll, pre-registered first.
