# REFe final model — the full NAVSIM navtest (12,146 tokens)

**MEASURED 2026-10-04.**
- **Checkpoint:** `model_final.pt`, md5 `b54773f8c302b2ec8ff4c5971d1e1370`: run `vitl16_navtrain10_grow_tau0.3`, 10,075 steps, 27 epochs.
- **Settings:** pick rule navsim_v1; Amendment 7's last-pose heading repair ON; Amendment 8's goal sanitisation OFF.
- **Tier:** NAVSIM v1.1 PDMS, a non-reactive pseudo-simulation of a 4 s open-loop plan against logged agents.
- **Interval:** log-cluster bootstrap over 136 logs (10,000 resamples, seed 20260927). It answers *another draw of logs* only: one training run, deterministic inference.

## Headline

| | PDMS | NC | DAC | EP | TTC | C | DDC |
|---|---|---|---|---|---|---|---|
| **REFe final, all 12,146** | **83.24 [81.77, 84.63]** | 97.64 | 93.72 | 75.53 | 92.99 | 99.32 | 96.93 |
| HUMAN (log replay) | 94.55 | 100.00 | 100.00 | 86.96 | 100.00 | 99.90 | |
| STOP | 61.82 | 97.40 | 96.53 | 31.00 | 96.40 | 69.44 | |
| refcv4b A1 (earlier programme model) | 58.97 | 91.35 | 73.46 | 52.37 | 82.54 | 98.02 | |
| constant velocity | 20.65 | 68.02 | 57.84 | 19.44 | 50.03 | 100.00 | |

Paired comparisons, same tokens (`parse_navtest6.py`, W3's banked floors):

| vs | ΔPDMS [95 % CI] | wins / ties / losses |
|---|---|---|
| STOP | +21.42 [+19.76, +22.85] | 9,843 / 1,094 / 1,209 |
| constant velocity | +62.58 [+60.73, +64.48] | 10,898 / 899 / 349 |
| refcv4b A1 | +24.26 [+22.04, +26.40] | 7,099 / 2,163 / 2,884 |
| HUMAN | −11.32 [−12.51, −10.22] | 2,771 / 2,481 / 6,894 |

The 200-token subset point banked on 2026-10-01 (84.18) lies inside this interval. Scoring the same 200 tokens from this
seam reproduces 84.1843 exactly.

## Four families (families6, all 12,146 tokens; constant velocity in brackets)

| family | result |
|---|---|
| **Longitudinal** | speed MAE 0.93 m/s [0.88, 1.00] (CV 1.28); speed bias −0.30 m/s (slower than the human); within 0.5 / 1 / 2 m/s: 49.0 / 67.8 / 86.7 %; along-track MAE 1.34 m; final along-track bias −1.16 m (falls short) |
| **Lateral** | heading MAE 2.97° [2.73, 3.20] (CV 8.48); yaw-rate MAE 3.77 °/s (CV 4.54); curvature MAE 0.028 1/m (CV 0.020, better); cross-track MAE 0.44 m (CV 1.08); final cross-track MAE 1.05 m |
| **Tactical** | lateral decision accuracy 0.909, κ 0.834; longitudinal decision accuracy 0.638, κ 0.453; 5-way manoeuvre accuracy 0.760, κ 0.689 |
| **Strategic** | UNAVAILABLE in NAVSIM by design (the benchmark scores no strategic decision; n = 12,146) |

## Where the missing points are

| slice | tokens | PDMS |
|---|---|---|
| on-route | 11,671 | 84.37 [83.05, 85.62] |
| **off-route** (the scenario route skips the ego's roadblock) | 475 | **55.45 [45.97, 65.30]**, costing the full mean 1.13 points |
| command LEFT / STRAIGHT / RIGHT | 2,501 / 8,070 / 1,575 | 81.83 / 84.35 / 79.76 |

- **983 picks (8.1 %) score exactly 0**: DAC 721, NC 220, both 42. 169 of them are off-route.
  - If they scored like the rest, the mean would gain about 6.7 points. This is the largest single lever.
  - The scorer's own predicted PDMS on those picks averages 50.0, against 70.0 on the others.
  - Control: the re-implemented aggregation reproduces the planner's pick on 12,146 / 12,146 tokens.
- **ELIMINATED (exploratory, CPU, exact): a confidence-gated fallback.**
  - The rule: when the maximum predicted PDMS is below τ, execute the STOP plan.
  - τ = 20 was chosen on the 1,123 selection tokens and frozen.
  - On the 11,023 held-out tokens it reads +0.07 [−0.11, +0.32] (fires on 1.2 %).
  - A constant-velocity fallback never helps.
  - ⇒ the lever is a better choice **among** the 64 hypotheses: a scorer fine-tune on on-policy hard negatives, whose data are banked.
- **The off-route goal:** `raw/2026-10-01-goal-trigger/` (why it happens, NAVSIM's own route correction, Amendment 9 pre-registration).

## Declared departure — the frame/time control

The seam's own control failed its 1 mm bar as written: max 6.18 m, median 9e-16 m.
- `eval_checkpoint.py` therefore stopped at SEAM_FAILED on 2026-10-02 and did not score the seam.
- The per-token census (`raw/2026-10-04-navtest-final/frame_control_navtest_full.json`) localises the failure to **36 tokens in 4 logs**. On each, the error is exactly 0 up to a fixed absolute timestamp, then takes a constant value (5.6151 / 5.1237 / 2.4174 / 1.9889 / 1.1356 m) that moves one pose earlier per later token.
- That pattern is a nuPlan-DB vs OpenScene pose disagreement at those instants. It is not a frame-convention or a token-pairing error.
- The unchanged seam was then scored. Without the 36 tokens: **83.24 [81.77, 84.64]** on 12,110. They score 81.13 themselves.

## Artifacts

- `raw/2026-10-04-navtest-final/`: the readout (`navtest_final_readout.py/.json`), the per-token CSV, the seam report, the floors (`3_parse.json`), the families (`4_families.json`), the frame-control census, and the drive-repointed runners.
- `eval/raw/points/navtest_final.json`: the point record.
- The seam npz and the 64-proposal dump: `<drive>/Projects/TanitAD/data/refe_navtest/{seams,proptable/navtest_final}/`. The external drive now mounts as E:.
