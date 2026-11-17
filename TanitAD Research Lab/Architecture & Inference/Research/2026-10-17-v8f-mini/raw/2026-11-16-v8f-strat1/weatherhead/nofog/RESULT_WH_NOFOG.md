# RESULT: weather head with fog dropped (AMEND_WH_NOFOG, read 2026-11-17)

**Outcome (the card's rules, unchanged): weather = WH-DETECTS + WH-INPUT-FEATURES on clear / rain / snow.** The other variables and driving read as in the main read: country POOL-GAIN; surface, illumination and period_of_day POOL-REPLACES; driving NO-COST-BY-IDENTITY.

**Evidence class and tier.** MEASURED, T0, zero training. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. Amendment `AMEND_WH_NOFOG.md` (sha256 43899fd2) was written before any number.

- **Windows:** 68 fog-labelled windows in 8 episodes are unscored. That leaves 2,331 windows over 138 episodes.
- **Instrument:** every required check passes. K0 runs on the unmasked world.

| | H_s0 | H_s1 |
|---|---|---|
| accuracy (clear / rain / snow) | 0.9511 | 0.9575 |
| constant (clear) | 0.8825 | 0.8825 |
| accuracy - constant | +0.0686 [+0.0189, +0.1229] | +0.0751 [+0.0265, +0.1285] |
| balanced accuracy | 0.870 | 0.855 |
| recall clear / rain / snow | 0.965 / 0.706 / 0.939 | 0.976 / 0.661 / 0.927 |

**Controls and pairs.**

- The re-initialised head K4p does not detect.
- H - RA (trained vs random adapter):
  - s0: -0.0069 [-0.0171, +0.0004], not separated;
  - s1: +0.0107 [+0.0004, +0.0239], separated.

  WH-ADAPTER-ADDS needs both seeds, so the reading stays WH-INPUT-FEATURES.
- H - shipped:
  - s0: +0.0283 [+0.0073, +0.0539];
  - s1: +0.0099 [-0.0051, +0.0257].
- Beside, no bar: H - FEAT -0.0133 (s0, separated) and +0.0017 (s1); H - probe, with the probe unmasked, +0.009 at both seeds, not separated.

**What it means.** Weather without fog is detected at both seeds and ships as a clear / rain / snow output.

- Rain recall (0.66-0.71) is the weakest class.
- Rain's own label is borderline (8/11 in Q3). The PI's second rain draw decides whether that recall is a head limit or a label limit.
