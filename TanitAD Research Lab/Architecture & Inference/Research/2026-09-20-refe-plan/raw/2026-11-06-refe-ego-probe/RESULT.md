# REFe ego-reliance probe on real nuPlan frames: RESULT (2026-11-06, Master Mind; probe built by a Sonnet agent)

**Tier T0** (open loop, one step, logged ego history). This is an input-reliance diagnostic, NOT a driving-performance claim.
- **Model:** u400 (`model_sft4_B_u400.pt`, sha256 `8b4dba48...`, the checkpoint TanitScenarioFlyWheel deploys).
- **Data:** 1,058 C3 navtest tokens over 123 logs. CRUISE subset: 154 tokens over 51 logs (v0 > 5 m/s, |ax0| < 0.3, |GT v4 − v0| < 1 m/s).
- **Estimator:** paired log-cluster bootstrap, B = 2000. It answers "another draw of logs" only. This is one checkpoint and a deterministic planner (C1), so no claim about training or inference variance is made.
- Full tables: `full/analysis/READOUT.md`.

## Controls (all PASS)
| control | result |
|---|---|
| C1 determinism | A0r replicate bit-identical on 265/265 tokens |
| C2 identity | A0 logits bit-identical to `raw/2026-10-11-c3-selection-headroom/table.npz` on 1,058/1,058 |
| C3 arm reached input | every arm reached the model input, 1,058/1,058 each |
| C4 GREY trunk cache | 42/42 fresh rechecks bit-identical |
| C5 dead slots | real `yaw_rate` and `steer` are non-zero on **0.0 %** of tokens, the dead-slot defect seen in the input itself (`D-REFE-DEADSLOTS-1`) |

## Committed verdict: **UNDECIDED** (as written in `PREREG.md`; reported, not re-cut)
- R2 needs b_A0(CRUISE) ≤ −0.5 with CI-hi < 0. R1 and R3 need |b_A0| < 0.5.
- MEASURED: b_A0(CRUISE) = **−0.535 [−1.391, +0.221] m/s**. That is past the 0.5 bar, but its CI includes 0, so no committed reading holds.

## What the numbers say (MEASURED, descriptive beyond the committed rule; CRUISE, Δ vs A0 at 4 s)
| arm | Δv@4s (m/s) | share of CRUISE planning a >15 % slow-down | plan v4 / v0 |
|---|---|---|---|
| A0 (real) | — | 0.409 [0.255, 0.600] (GT: 0.013) | 0.934 [0.803, 1.054] |
| AX0 (ax = ay = 0) | −0.039 [−0.117, +0.036] | 0.429 | 0.928 |
| AXP / AXM (ax ±0.5) | +0.522 / −0.581 | 0.273 / 0.506 | 1.004 / 0.856 |
| DEAD (yaw 0.3, steer 0.2) | +0.011 [−0.009, +0.031] | 0.409 | 0.935 |
| **GREY** (all cams 128) | **−2.273 [−3.254, −1.235]** | **0.987** | **0.645** |
| **SWAP** (another log's images) | **−1.884 [−2.599, −1.039]** | 0.675 | 0.693 |

1. **Vision is REFe's dominant speed input on real data.** GREY and SWAP move v@4s by about 2 m/s and the lateral offset by 0.8–1.0 m; ax = 0 moves v@4s by 0.04 m/s.
   - The sim's statement "REFe steers its speed mostly by its own speed and acceleration, not by what it sees" does **not** hold on in-distribution nuPlan frames.
   - The sim session had already corrected its grey-image observation to a 20–30 % effect.
2. **ax has a real but bounded gain:** +1.10 [+0.92, +1.30] m/s of planned v@4s per 1 m/s² of ax (CRUISE). In closed loop that feeds back, but per replanning cycle the loop is contractive. On its own it does not explain a crawl.
3. **The dead slots barely matter at the plan level:** 3 cm ADE, Δv@4s +0.011 m/s. They are still a wiring defect to remove, but they are not the crawl.
4. **An in-distribution slow tendency exists, unseparated:**
   - 41 % of CRUISE tokens plan a >15 % slow-down, where the logged drive does so on 1.3 %.
   - On the non-zeroed stratum (n = 63), b_A0 = −0.746 [−1.468, −0.064] m/s, separated. This is descriptive and post hoc, so it is not promoted.
5. **Reading for the sim (HYPOTHESIS, to test there):** REFe's sim plans keep v4/v0 = 0.50–0.78. That sits nearer the GREY regime (0.645) than real-data A0 (0.934), so the leading suspect is image informativeness. Candidates are the reprojected Cosmos frames and the junction and light ambiguity.
   - The discriminating sim test: on a straight, empty road, REFe on sim frames gives v4/v0 near 0.93 if the images are informative, and near 0.65 if they read as no information.

## Next levers (Rule Zero)
- **Sim side (TanitScenarioFlyWheel, in progress):** a straight-road clip probed with frames vs grey at a = 0, read against the two anchors above. The feedback-cut closed loop (ax forced to 0) is a PI decision on that session's Thor.
- **Policy side:** the 41 % planned-slow-down share on CRUISE is a selection-or-generation question. Selection-vs-generation (READOUT) shows that for GREY and SWAP the pick slows 2× more than the mean proposal, so the scorer amplifies the slow-down. A pick-rule probe on navtest PDMS (which proposals, and which progress weight) is the cheapest next experiment; it is not run here.

Files: `PREREG.md` (with Addendum A1), `refe_ego_probe.py`, `analyze_ego_probe.py`, `make_subset.py`, `launch_cmd.txt`, `tokens_*.json`, `HANDOFF.md`, `smoke/` (first smoke, a float32-vs-float64 check bug kept as evidence), `smoke2/`, `full/` (run log, `results.npz`, `analysis/READOUT.md`, `readout.json`).
