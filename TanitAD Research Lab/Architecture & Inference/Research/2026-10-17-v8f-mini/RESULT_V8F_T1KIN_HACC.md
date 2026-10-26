# RESULT_V8F_T1KIN_HACC: HACC's margin over the P10 floor SURVIVES self-action, with the declared speed bucket. Outcome: T1 BEATS THE FLOOR (not dominance: lateral curvature, at low speed)

`PREREG_V8F_T1KIN_HACC.md` (`323624f2`), with Amendments 1 (`15abe8d9`), 2 (`f712c0a0`) and 3 (`cf46caf3`). Read 2026-10-26 by `raw/2026-10-26-t1kin-hacc/t1k_read.py`, applied verbatim. Claim row `H-V8F-T1KIN-HACC-1`.

* **Tier T1 (T1-kin).** Self-action OPEN loop: the ego state the model reads is rebuilt from its own executed motion at every replan, and the camera frames stay logged. ⛔ This is not closed loop, and it does not say "the model drives".
* **Settings:** m = 10 (replan every 2 s), `--nav deadreckon --scene logged --limit logged --schedule periodic`.
* **Data:** eval-141, 2,399 windows, 141 episodes. Dev-box CPU.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000. It answers "another draw of EPISODES"; the two training seeds are the replicate. F = 0.1207 m, 2F = 0.2415 m.
* **Artifacts:**
  * The runs: `raw/2026-10-26-t1kin-hacc/run/` holds 12 `t1v8f_*.json`, the chain and its logs, and per-run dumps (float32, `run/dumps/`, with `dumps_manifest.json` carrying the float64 content sha256).
  * The float64 originals: `E:/Projects/TanitAD-artifacts/v8f-mini-hacc/t1kin_hacc_dumps_f64_2026-10-26.tar` (1,692 files, 74,577,920 B, sha256 `34fed9c1…f9e8d1`, `tar -d` verified).
  * The reads: `read2/` (decisive), `read3/` (+ brackets) and `read4/` (+ m = 5). Their decision rows are identical.

## The read (HACC − P10, both in their own loop, m = 10; lower is better)

| row (family) | s0 | s1 |
|---|---|---|
| **ADE6, PRIMARY (m)** | **−0.8291 [−0.9711, −0.6894]** ✅ | **−0.7945 [−0.9379, −0.6575]** ✅ |
| ADE2 (m) | −0.0112 [−0.0158, −0.0061] better | −0.0094 [−0.0146, −0.0033] better |
| speed MAE (m/s), LONG | −0.3868 [−0.4817, −0.2955] better | −0.4306 [−0.5166, −0.3459] better |
| along-track MAE (m), LONG | −0.5359 [−0.6785, −0.4059] better | −0.5811 [−0.7039, −0.4668] better |
| accel MAE (m/s²), LONG | +0.0451 [−0.1205, +0.2108] n.s. | −0.2288 [−0.4024, −0.0699] better |
| heading MAE (deg), LAT | −0.4317 [−1.1139, +0.2473] n.s. | −0.7696 [−1.3714, −0.1844] better |
| **yaw-rate MAE (deg/s), LAT** | **+4.2048 [+2.4880, +5.9866] WORSE** | +1.5304 [−0.0706, +3.1005] n.s. |
| **curvature MAE (1/m), LAT** | **+0.0400 [+0.0262, +0.0550] WORSE** | **+0.0155 [+0.0077, +0.0242] WORSE** |
| cross-track MAE (m), LAT | −0.5721 [−0.6914, −0.4608] better | −0.4325 [−0.5659, −0.3092] better |
| TACTICAL / STRATEGIC | TACTICAL from the executed track (2026-10-26, `RESULT_V8F_T1_FAMILIES`, POST-DATA Amendment 1): NO WORSE; TAC-LAT +0.136, TAC-LON +0.131 accuracy, better. STRATEGIC and the model's own T code: NOT COMPUTED: the T1 dumps carry no per-tick T code or S checkpoints (work item) | TAC-LAT +0.092, TAC-LON +0.132, better; the rest same |
| distance keeping | NO WORSE (2026-10-26, `RESULT_V8F_T1_FAMILIES`, POST-DATA Amendment 1; lead block joined by the reader, since `v8f_t1_eval` passes no lead track and `distance_keeping` is UNAVAILABLE in the tool): DK-TG -0.101 s, DK-HW -0.32 m, better; DK-VIOL n.s. | DK-TG -0.073 s better; DK-HW, DK-VIOL n.s. |

Mean ADE6 at m = 10: HACC 2.614 / 2.649 m against P10 3.443 m. Masked rows: heading n = 2,373; yaw rate and curvature n = 2,366.

**Outcome (the table, verbatim): T1 BEATS THE FLOOR.** PRIMARY holds at both seeds. DOMINANCE fails because curvature is separated-worse at both seeds and yaw rate at s0, all in the LATERAL family.

## Controls (all read before any outcome row)

| control | s0 | s1 |
|---|---|---|
| C-1(i), m = 30 is open loop: max \|cl − ol\| ≤ 1e-9 m (Amendment 3) | 1.1e-13 | 1.4e-13 |
| C-1(ii), the floor equals HBLEND's banked P10 plans, ≤ 1e-4 m (Amendment 2) | 1.5e-5 | 1.5e-5 |
| C-2, the floor's m = 30 ADE6 = 2.48699 ± 1e-5 m | 2.486992 | 2.486992 |
| C-3, the post-map is in the loop: max \|HACC − B\| executed at m = 10 > 1e-2 m | 80.4 | 89.3 |
| C-4, `test_v8f_t1.py` on the run's tip | 23 passed | |
| R-1, the reader's rows equal `four_families`' values (7 rows × 2 arms, to its 4-dp rounding) | PASS | PASS |
| R-2, the reader's paired ADE6 equals `analyze()`'s | PASS | PASS |
| R-3, known-value synthetic tracks | PASS | |

⚠ **The first read was REFUSED on C-1(i).** As pre-registered, that clause said "exactly", and the measured difference was 1.1e-13 m: float round-off. Amendment 3 restored the instrument's own 1e-9 m (`R74`). The refused read's JSON was banked unopened (sha256 `77b11bfa…`). The re-read equals it bit for bit on every row, differing only in C-1's verdict and the outcome (`read2/integrity.txt`).

**Independent MM re-derivation** (own numpy ADE6 and own bootstrap, `mm_verify/verify_t1k_mm.json`): s0 −0.8291 [−0.9659, −0.6835], s1 −0.7945 [−0.9385, −0.6568]. The points agree exactly, the intervals to the draw.

## Beside (decides nothing)

* **m = 30 (one shot, T1 = open loop at tick-0 ages):** HACC − P10 ADE6 −0.5292 [−0.6394, −0.4242] / −0.5288 [−0.6360, −0.4238]; accel −0.0247 / −0.0398 m/s². No envelope clips.
* **The margin grows in the loop:** −0.53 m at m = 30, −0.83 / −0.79 m at m = 10. The floor degrades in its own loop (2.487 to 3.443 m) much more than HACC does.
* **Plain M2b (B) at m = 10:**
  * B − P10 ADE6: −0.4101 [−0.6654, −0.1419] / −0.6524 [−0.8544, −0.4455].
  * B − P10 accel: **+1.2232 [+0.8151, +1.5813] / +1.0772 [+0.6769, +1.4614] m/s², separated-worse.**
  * HACC − B ADE6: −0.4190 [−0.6800, −0.2041] / −0.1421 [−0.3121, +0.0012].
  * ⇒ **The post-map is what keeps O's loop smooth.** Without it, O is 1.1–1.2 m/s² rougher than the floor at T1.
* **Envelope** (the tool's 60 m/s bound, counted per 10 Hz step): 172 / 163 HACC windows and 178 floor windows at m = 10.
  * POST HOC: excluding the windows whose 0.2 s executed knots reach the bound (19 per seed; the 10 Hz count is not recoverable from the dumps), HACC − P10 ADE6 is −0.8405 / −0.8000 m.
* **Age effect** (Amendment 2, beside): |`ol` − banked open-loop HACC plan| median 0.23 / 0.34 m, max 31.7 / 38.1 m. 12.1 % of windows are below 1e-4 m. The cause is the sample-and-hold ages, MEASURED in `ages_probe{,2}.log`.
* **m = 5 (a replan every 1 s; the stress read, `read4/T1K_READ.json`):**
  * HACC − P10 ADE6: −0.1493 [−0.3183, +0.0310] (s0, not separated) / **−0.4663 [−0.7294, −0.1955]** (s1, separated).
  * Accel: −0.883 / −0.877 m/s², separated-better.
  * Both loops run away at this period. The floor reaches 5.41 m, and HACC 5.26 / 4.94 m. T1 − open ADE is +3.30 / +2.98 m, and the envelope clips 516 / 566 HACC windows and 648 floor windows.
  * ⇒ At a 1 s replan period neither is usable. HACC still degrades less than the floor, but the decisive period is the pre-registered m = 10.
  * read4's decision rows are identical to read2's.
* **Brackets:** the next section.

## Brackets: how much the two open channels carry (beside, MEASURED, `read3/T1K_READ.json`; the decision rows are identical to read2)

| HACC − P10 ADE6 at T1-kin m = 10 | s0 | s1 | bracket − default HACC |
|---|---|---|---|
| default (camera logged, speed bucket logged) | −0.8291 | −0.7945 | — |
| `--scene frozen` (every tick reads the t0 frames: the camera channel closed, doctrine-literal T1) | **−0.7869 [−0.9246, −0.6563]** | **−0.7055 [−0.8351, −0.5833]** | +0.042 / +0.089 m |
| `--limit off` (the speed bucket removed) | **+0.2378 [−0.0455, +0.5271] n.s.** | **+0.0776 [−0.1729, +0.3308] n.s.** | **+1.067 / +0.872 m** |

* **The camera channel flatters HACC only a little.** With the cameras frozen at t0, the margin over the floor is still −0.79 / −0.71 m, separated.
* ⚠⚠ **The speed bucket carries the margin.** Without it, HACC is no better than the floor at T1 at either seed. HACC's ADE6 rises by about 1 m, its accel MAE becomes separated-worse than the floor (+0.67 / +0.33 m/s²), and its envelope-clipped windows roughly double (344 / 245, against 172 / 163).
  * The bucket is a PI-declared INPUT (`PI-V8F-2026-10-17`: *"measured ego dynamics, max target speed and nav"*).
  * But on PhysicalAI it is built from the max of the ego's realised future speed (`D-SPEEDBAND-EGO-ECHO-1`, INHERITED, not re-verified here). ⇒ **It is optimistic by construction on this corpus.**
  * A deployable bucket comes from a speed limit (map or sign), which PhysicalAI does not carry.
  * ⚠ "Off" is also out of the training distribution (the model was trained with the bucket present), so the +1 m mixes the information the bucket carries with the model's response to an absent input. The two are not separated here.
* **This is the same finding as SPEEDCAL's, from the other side.** Speed is the largest lever, and in this read the model gets it from the declared bucket, not from what it sees.

## Where the lateral excess lives (DESCRIPTIVE diagnosis, decides nothing)

1. **Not the replan seams** (`curv_profile.log`): the step-pairs straddling the junctions carry the same excess as the rest (+0.040 vs +0.038 / +0.015 vs +0.015 1/m).
   - In plan 0 (the logged state, the first 2 s), HACC is BETTER than the floor, by about 0.02 1/m.
   - The excess begins with the first plan issued from the model's OWN executed state.
2. **Not the camera viewpoint** (`viewpoint_probe.log`): the excess does not grow with the executed-vs-logged pose offset at the replan tick (Spearman +0.04 / +0.06). It is largest in the smallest-offset quintile.
3. **It is low speed** (`speed_probe.log`), measured on pairs 10–29, the plans issued from the model's own state:

| GT speed quintile (m/s) | curvature excess HACC − P10 (1/m), s0 / s1 |
|---|---|
| 0.1 – 5.2 | **+0.399 / +0.249** |
| 5.2 – 8.6 | +0.063 / +0.013 |
| 8.6 – 11.8 | +0.005 / +0.004 |
| 11.8 – 17.0 | +0.002 / +0.000 |
| 17.0 – 36.5 | +0.000 / +0.000 |

   - Near standstill a small lateral wiggle is a large curvature. HACC's executed track picks it up from O's residual once it plans from its own slow state.
   - ⚠ HACC's objective (positions plus a second-difference term, both in metres) is nearly blind to it, because a heading wobble at walking pace moves positions by millimetres.

## What the result is, and what it is not

* **It is v8F's first T1-tier result over the kinematic floor.** With HACC's zero-training post-map, the model living on its own executed motion for 6 s beats the floor driven the same way by 0.83 / 0.79 m of 6 s ADE, at both seeds. It is separated-better on ADE2, speed, along-track and cross-track at both seeds, and on accel and heading at s1. Nothing longitudinal is worse. With the cameras frozen at t0 the margin is still 0.79 / 0.71 m.
* ⚠⚠ **It rests on the declared speed-bucket input.** With `--limit off`, the margin is not separated at either seed (brackets section). On this corpus the bucket comes from the ego's realised future speed, so the claim is conditional on that declared, optimistic input.
* ⚠ **Not dominance.** Lateral curvature is worse at both seeds and yaw rate at s0, concentrated below 5 m/s.
* ⚠ **Not closed loop.** The cameras are the human's (T2 is not provisioned). The floor reads no camera and no speed bucket, so those two open channels flatter the model; the brackets measure how much. The baseline is the programme's own floor, not the state of the art.

## Next (committed by the PREREG's outcome row)

LATERAL is the next lever's target, under its own PREREG.
* The committed first lever is HACC's λ rule re-fitted on T1 executed tracks of TRAINING windows. HBLEND's 141 training clips are cached on the dev box, so it runs on the dev-box CPU.
* The diagnosis adds a second arm aimed at the measured cause, the low-speed lateral residual, pre-registered beside the committed one before any number (`PREREG_V8F_HACC_LAT.md`).
* **Beside the committed next, for the PI:** the speed-bucket dependency. Is a claim on the declared bucket admissible as v8F's T1 result, or must it be re-read with a non-oracle speed input? Filed as `D-V8F-SPEEDBUCKET-T1` in `PI_DECISION_QUEUE.md`.
