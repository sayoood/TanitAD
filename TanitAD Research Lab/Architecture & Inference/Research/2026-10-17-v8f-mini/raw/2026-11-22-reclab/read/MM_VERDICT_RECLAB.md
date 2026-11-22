# MM verdict: RECLAB (the DAgger swap's label starts at the own speed, RL_s0 / RL_s1): RECLAB-PARTIAL, MECH-NO (MM, 2026-11-22)

## Card, arms and method
* **Card:** `PREREG_V8F_RECLAB.md` (sha256 154d7ad9), frozen 2026-11-22T03:53:32Z before any RL step, pod census or number. PI go "Run both", 2026-11-22 about 07:12Z.
* **Arms:** RL_s0 and RL_s1 = RH's argv plus exactly `--dagger-label speed_v1` (and `--out`), on the RH banks, init UE_s0.A / UE_s1.A, code cf27dbb7. Comparators: the banked RH_s0/s1 and B0_s0/s1 (97fe0a56; the gold test makes `human` at cf27dbb7 equal 97fe0a56 bit for bit).
* **Evidence class:** MEASURED (ours). The pod trained both arms; the dev box ran 12 rollouts and the reads with zero training.
* **Tier:** P1 and ATTR at T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record). G-T0 and MECH at T0 (teacher-forced on logged history: a world-model diagnostic, not driving performance).
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0 (`taniteval/ci.py`). It answers "another draw of EPISODES"; the two seeds answer "another TRAINING run"; inference is deterministic. Floor F per cell = the larger seed spread of the two arms (ROLLHIST card section 3).
* **Files** (this folder): `RESULT_ROLLHIST_RL.json` (sha256 5913724116a0b5e2), `GT0_ROLLHIST_RL.json` (4927bb51d4a6eb1f), `READ_STRAT_ROLLHIST.json` (56dc1f6a6c894e5e, strategic controls all pass), `hw/` and `q3/` (headway and Q3 readers), `MECH_RECLAB.json` (119a22ba2e42fee6, from `mech_rl.py` 2fc2ed2bba243e62), `pod_rl/` (each arm's `selfcheck.json`, `summary.json`, `post.json`, `train_log.jsonl`, pulled from the pod).
* The read tool is the ROLLHIST reader with RL as the treated arm, so it prints ROLLHIST names (`ROLLHIST-PARTIAL`) and the line "the card names no lever after it". Both are the ROLLHIST card's wording. The RECLAB name and the next lever below follow `PREREG_V8F_RECLAB.md` section 5, read by hand from the same numbers. MECH is not in the read tool; `mech_rl.py` computes it with the G-T0 instrument and estimator.

## The arms (pod)
* Lock held 08:58:55Z to about 09:40Z (about 0.7 pod-h with both selfchecks). Training: RL_s0 567.2 s, RL_s1 564.4 s for 3,000 steps (RH 569 s). G-SUITE at cf27dbb7 ran before, standalone on the pod CPU (07:46Z to 08:24Z, no GPU). Cost against the card's estimate of about 0.9 pod-h: inside it.
* Gate record `rl-chain-gate-record-v1` pass at both seeds; post check PASS; zero-grad gate PASS at both seeds.

## Census and lever-acts check (card sections 4 and 3, from each arm's `--selfcheck-only` on the full bank)
| | RL_s0 | RL_s1 |
|---|---|---|
| ok records (eligible, equal before and after) | 1,490,036 | 1,478,090 |
| relabelled | 805,009 (**0.540**) | 856,048 (**0.579**) |
| identity (label equals the human's) | 645,242 | 586,314 |
| refused (path end + checker) | 17,161 + 22,624 | 14,338 + 21,390 |
| relabelled share, tick 1 / tick 2 | 0.386 / 0.735 | 0.412 / 0.795 |
| abs(v_own - u_h), p50 / p95 (m/s) | 0.77 / 6.16 | 0.88 / 6.11 |
| label shift at 6 s, p50 / p95 (m) | 0.034 / 6.13 | 0.058 / 6.08 |
| selfcheck probe: label rows that differ from the human's | 20 | 31 |
| train log: swapped rows that took the speed_v1 label, of all swapped rows (3,000 steps) | 15,038 of 15,411 (at least 220 per logged interval) | 15,048 of 15,380 (at least 214) |

* The census clears the 0.5 bar at both seeds, so the arms are not BLOCKED. The lever acts at both seeds. Under `human` on the new code the build census reads exactly 0 attached labels (`FACTS_RECLAB_3.md`, gold test).
* The train-log count includes identity labels, which equal the human's; the probe count is the one that differs.

## The decisive reads
**G-T0** (RL - B0, T0 ADE6; bar: upper bound at most +0.0705 m, F_T0 INHERITED from one seed pair):
* s0: +0.0479 m [+0.0090, +0.0897], **FAILS**.
* s1: +0.0061 m [-0.0312, +0.0470], **HOLDS**.

**P1** (RL - B0 under the form of record `R0C10XHb20`, both seeds):
| family | cell | RL - B0 [interval] | per seed s0 / s1 | F | verdict |
|---|---|---|---|---|---|
| LONGITUDINAL | speed_mae | -0.254 m/s [-0.381, -0.140] | -0.190 / -0.319 | 0.126 | BETTER |
| | along_mae | -0.215 m [-0.390, -0.056] | -0.122 / -0.308 | 0.122 | BETTER |
| | headway U (unsafe rate) | +0.0222 [+0.0070, +0.0394] | +0.0160 / +0.0285 | 0.0231 | NOT-SEPARATED (straddles F; by sign WORSE) |
| LATERAL | heading_mae | -0.929 deg [-1.145, -0.750] | -0.862 / -0.996 | 0.699 | BETTER |
| | yaw_rate_mae | -5.31 deg/s [-6.95, -3.91] | -4.63 / -6.00 | 3.06 | BETTER |
| TACTICAL | ade_tac | -0.368 m [-0.629, -0.129] | -0.223 / -0.513 | 0.164 | BETTER |
| STRATEGIC | position at tick 2, mark 30 m | -1.80 m [-2.26, -1.39] | -2.00 / -1.61 | 0.71 | BETTER |
| | mark 50 m | -3.50 m [-4.40, -2.71] | -4.05 / -2.96 | 1.77 | BETTER |
| | mark 80 m | -5.29 m [-6.76, -3.92] | -5.95 / -4.64 | 2.48 | BETTER |

* **Families:** LATERAL, TACTICAL and STRATEGIC BETTER; LONGITUDINAL MIXED (speed and along better, the unsafe rate up); no family WORSE.
* Beside, no bar: cross-track -0.040 m [-0.067, -0.017]; curvature -0.093 1/m [-0.123, -0.070]; tactical error rates lon -0.074, lat -0.048, joint -0.087 (all intervals below 0).
* Headway reader (card ADD25 engine, its own spread rule, decides nothing here):
  * U +0.0222 reads WORSE against its spread 0.0178; C (contact rate) +0.0171 [+0.0044, +0.0315] reads inside its spread.
  * F (falls behind by more than 10 m) -0.0639 [-0.1012, -0.0363] reads BETTER.
  * Time-gap error over the scored steps -0.71 s [-1.02, -0.45].
  * So the arm keeps up better and comes closer, at a higher unsafe rate.
* STRATEGIC decision between routes: NOT COMPUTED (n = 0, no instrument; as row 120). STRATEGIC is reported in full on the position cells.

**MECH** (RL - RH, T0 ADE6, paired; MECH-YES needs the interval below 0 at both seeds):
* s0: -0.0024 m [-0.0273, +0.0247].
* s1: -0.0156 m [-0.0376, +0.0058].
* **MECH-NO.**

## Label of record, by the card's precedence (BLOCKED, then HARMS, then SEED-DISAGREE, then the table order)
* **BLOCKED:** no. The census is at least 0.5 at both seeds, every gate check passed, and the lever acts.
* **HARMS:** no. G-T0 holds at s1, and no family is WORSE at either seed.
* **SEED-DISAGREE:** no. No family reads BETTER at one seed and WORSE at the other (the read lists no seed-disagree cell). The one straddling cell, headway U, is in LONGITUDINAL, which reads MIXED either way, and with straddles read by sign the name does not change.
* **REPAIRS:** no. G-T0 fails at s0.
* **PARTIAL:** yes. G-T0 holds at one seed only.
* ⇒ **RECLAB-PARTIAL, with MECH-NO.** It is reported as written.

## What it says
* **H-V8F-RECLAB-1 is not supported at this dose.**
  * MECH reads NO: the speed-consistent label does not lower the T0 cost against RH at either seed. The point values are -0.002 m and -0.016 m, inside intervals that cross 0.
  * The one-seed hold of G-T0 is RH's own pattern (RH - B0: s0 +0.0503 fails, s1 +0.0217 holds). The label did not move it.
  * The card's reading column for PARTIAL says "part of the cost is the label". MECH, the card's committed mechanism read, does not support that. On MECH-NO the card itself routes the line to the HARMS / MECH-NO lever: "the label is not the cause". That is the reading taken here.
* **The label also changes nothing measurable at T1.** ATTR, RL - RH under `R0C10XHb20`, reads all four families NOT-SEPARATED.
  * LONGITUDINAL: speed +0.010 m/s [-0.001, +0.022]; along +0.010 m [-0.012, +0.032]; U +0.004 [-0.007, +0.016].
  * LATERAL: heading -0.119 deg [-0.224, -0.034], which straddles F 0.51.
  * TACTICAL: ade_tac -0.005 m [-0.036, +0.027].
  * STRATEGIC: -0.19, -0.27 and -0.27 m at the 30, 50 and 80 m marks, all inside F.
  * Beside, cross-track -0.018 m [-0.029, -0.008] separates. One beside cell of 0.018 m is not read as a lever effect: on the N1 line a same-seed replicate separated on 6 of 24 cells at about 0.002 m (N1DREP, `H-ESTIM-SEED-1`).
* **The card's named risk R1 (the label moves the target) is not seen:** the longitudinal cells of RL - RH are not separated. **R2 (the conflict is elsewhere: yaw rate or acceleration of the own history) is consistent** with MECH-NO, and so is a cost that has nothing to do with the label.
* **The own-action gains of rolled history hold under either label.** RL - B0 matches RH - B0 in pattern (LATERAL, TACTICAL and STRATEGIC better, LONGITUDINAL mixed). The T0 cost at s0 is also the same size.
* P2, P3 and P4 of the ROLLHIST card fail on RL, as on RH. The reader's line: "speed label repairs NEITHER P2 nor P3".
* Limits:
  * Two training seeds per arm. A separated interval is necessary, not sufficient.
  * F_T0 is INHERITED from one seed pair.
  * T1-kin only, with no re-perception.
  * By the eval doctrine T0 is a world-model diagnostic. The card made G-T0 binding, so the label stands, and this reading sits beside it.

## Next lever (card section 5, HARMS row, MECH-NO branch, committed before the data)
* **Keep the logged samples whole and ADD the own states:** own-state samples are appended to the batch instead of being swapped in. This separates interference (the own states teach something that conflicts with logged history) from replacement (the logged sample that the swap removes is what T0 misses).
* It is a new arm, so it needs a pre-registration frozen before any number, a zero-training build (a trainer flag whose default reproduces cf27dbb7 bit for bit, its G-DVB entry, reachability, mutants and a census), and a PI go with the cost.
* The MM starts the card and the build now. The run waits for the go.
