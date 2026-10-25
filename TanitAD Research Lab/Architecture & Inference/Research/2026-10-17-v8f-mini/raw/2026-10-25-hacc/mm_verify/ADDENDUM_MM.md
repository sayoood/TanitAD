

## MM independent re-derivation (2026-10-25, Master Mind; descriptive; it changes no row of the read)

The verdict above came from a build agent. Before it was quoted anywhere, the Master Mind re-derived the headline with code that shares nothing with the agent's fit or bootstrap (`raw/2026-10-25-hacc/mm_verify/verify_mm.{py,log,json}`):
* its own normal equations;
* a **different solver**: L-BFGS-B on ½wᵀAw − bᵀw, bounds [0, 1];
* its own D2 with the origin prepended;
* its own λ0, ADE and float32 accel;
* its own episode-cluster bootstrap (numpy, B = 2,000, rng 20261025).

It reuses only HSPEC's `load_world` (the eval arrays HBLEND and HSPEC already read) and the banked training fit windows.

| check | s0 | s1 |
|---|---|---|
| λ0 at HBLEND's w, train windows (mine / banked) | 2499.588 / 2499.588 | 2308.211 / 2308.211 |
| w at λ\* re-fitted from the TRAIN windows alone, different solver: max \|w_mine − w_banked\| | 1.5e-5 | 7.2e-6 |
| my ADE6 / accel vs the eval's `arm_pw` (P10), max abs | 0.0 / 4.8e-7 | 0.0 / 4.8e-7 |
| HACC − P10 ADE2 | −0.0106 [−0.0151, −0.0062] * | −0.0083 [−0.0132, −0.0032] * |
| HACC − P10 ADE6 | −0.5034 [−0.6082, −0.4030] * | −0.4708 [−0.5704, −0.3777] * |
| HACC − P10 accel MAE (m/s²) | −0.0215 [−0.0354, −0.0085] * | −0.0402 [−0.0518, −0.0291] * |
| KEEP: HACC − H ADE6, upper bound (bar ≤ +0.06037) | +0.0151 [−0.0029, **+0.0331**] | +0.0313 [+0.0031, **+0.0581**] |

* **The λ rule was re-applied by hand** to the agent's train grid (`fit_M2b_s*.json`). The smallest grid λ meeting all four train conditions is m = 4 at s0 (m = 0 and 1 fail curvature) and m = 16 at s1 (m = 0, 1 and 4 fail curvature). This matches λ\* as banked.
* **Every number agrees with the read to the bootstrap draw.**

**What the result is, and what it is not:**
* **The claim.** At both seeds, v8F's own output (M2b), blended with HACC's 30 × 2 train-fitted weights, is separated-better than its kinematic prior P10-yaw-4512 on every row of HSPEC's dominance row except distance keeping. That covers ADE2, ADE6, accel, curvature, speed, along, cross-track, heading, yaw rate, both plan-derived decision agreements and CONSISTENCY. Distance keeping (headway, time gap) is not separated, and LEAD-VIOL is separated-better at s0 and not separated at s1. Nothing is separated-worse.
* ⚠ **The tier is T1-class open loop:** the model's own held outputs, with no recorded future action and no re-perception. It is **not** a closed-loop driving result. The committed next tier is the T1-kin read, which needs a PI go.
* ⚠ **The baseline is the programme's kinematic floor, not the state of the art.**
* ⚠ **The KEEP margin at s1 is thin:** the upper bound +0.0581 (mine) / +0.0586 (the read) against +0.0604. HACC is separated-worse than H on ADE6 at s1, by +0.031 m. It passes a pre-registered F/2 bar, and that is how it is reported.
* ⚠ **One estimator question is answered:** another draw of episodes. Training-seed variance enters through the two seeds, both of which pass. Inference is deterministic.
