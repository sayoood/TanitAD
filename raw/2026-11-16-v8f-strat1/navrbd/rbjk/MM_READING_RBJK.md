# MM reading: RB-JOINT-K (2026-11-18): RBJK-CARRIES

**Committed reading.** The frozen card is PREREG_V8F_NAVRBD.md (sha256 881d7f3d, section 7, row RBJK). It was frozen before this check ran. This check reads **RBJK-CARRIES**: the 3 N1D arms are asked for as carded, once the build and gates pass.

**Evidence and scope.**
- Evidence class: MEASURED (ours). CPU only; zero model training.
- Tier: **T0 decodability, not driving performance.**
- Estimator: paired episode-cluster bootstrap (B 2,000, seed 0). It answers "another draw of EPISODES" only.

## Numbers (`RESULT_RBJK.json`)
| | value |
|---|---|
| target | 3-class kinematic TAC-LAT (`read_otband.labels`) of the GT 6 s track, the D1-HO label |
| population | 9,236 windows / 395 train episodes (RB-JOINT's 12,201 strided windows minus 2,965 without a complete 6 s future) |
| S (state only) | 0.4075, equal to the majority rate (lane_keep 0.4075) |
| ED (state + exit number + countdown distance) | 0.6406 |
| **ED - S** | **+0.2331 [+0.1906, +0.2748]**, separated, above 0.0383 |
| SHUF_ED - S (train labels permuted) | -0.0642 [-0.0896, -0.0409]: below 0.0383, control PASS |
| reference: RB-JOINT on 5-way v9 `lat_cls` | ED - S +0.0581 [+0.0287, +0.0857], n 12,201 / 426 |

**Label proof.** The GT track builder `D.gt_future(pose_grid)[t]` reproduces the STRAT1 dump's `g` exactly (max abs diff 0.0). Its kinematic label agrees with the dump's label on 2,399 of 2,399 eval-141 windows (`PROOF_RBJK.json`). The train cache shares that builder. The proof was not repeated on train clips.

## What it means, and what it does not
- On the label the arm is scored on, the route text's exit number and distance carry much of the roundabout turn decision. They carry more of it than they carry of the 5-way `lat_cls`.
- This retires the label-definition risk that the card named at freeze.
- **+0.233 is a gain over a state-only tree that sits at the majority class. It is not a gain over a trained vision arm.** N1R already reads the scene: it scores 0.5905 / 0.6328 on D1-HO. The headroom the channel can add on top of N1R is therefore unknown until N1D trains.
- The GAIN bar stays 0.0424 at both seeds.
