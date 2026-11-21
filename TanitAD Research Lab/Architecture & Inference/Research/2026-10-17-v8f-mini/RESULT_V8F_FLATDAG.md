# RESULT_V8F_FLATDAG: does the hierarchy beat flat once BOTH get own-state supervision? Outcome **NOT ESTABLISHED**

Claim row `H-V8F-FLATDAG-1`. Pre-registration `Project Steering/PREREG_V8F_FLATDAG.md` (promoted at `9b91b654` before any FLAT-DAG bank, run
or row; raw sha256 `8713d0a3…`). Read 2026-10-27 20:28 by the pre-registered reader (`raw/2026-10-27-flatdag/t1read/flatdag_read.py`,
sha256 `f978c4e3…` of the file that ran), all **91 reader controls PASS**. Output `raw/2026-10-27-flatdag/read/FLATDAG_READ.json`
(sha256 `7274b53209944305b998b40df3707d9d433cfb2d105d12e58342bff681e1c841`).

* **Tier.** T1-kin: self-action OPEN loop, m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms. Never "closed loop";
  the lead follows its logged future. R-T0 is at T0 (box 2's standard eval).
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired. It
  answers *"would another draw of EPISODES say this?"*. No replicate arm was trained, so a separated row is a result about one training run
  per seed (`H-ESTIM-SEED-1`); the planner is deterministic, so there is no inference-seed term.
* **Arms (box 2, all four through their launch gates, token commit `3101dbf1`).** FLAT-DAG = FLAT_s (`8496934a…` / `6fdfc9b0…`) fine-tuned
  3,000 steps with DAG's recipe at `--dagger-p 0.1` on FLAT's own bank, `--o-interface cut` kept; FLAT-DAG-0 = the same argv at
  `--dagger-p 0`. Banks: s0 `1fa89521…`, s1 `f5247555…`, 613,132 records over 4,512 clips each, COVERAGE 0.751 / 0.751. Swap share reached in
  training 0.1011 (s0) and 0.1005 (s1), 0 for both FLAT-DAG-0. Checkpoints FLAT-DAG `629819f7…` / `df1fe1c4…`, FLAT-DAG-0 `5dc2bc3f…` /
  `668fd099…`. DAG's T1-kin rollouts are the banked ones (`raw/2026-10-26-dagger-t1`).
* **Queue order, disclosed.** PREREG §5 put FLAT-DAG third, after DAG-OACC. DAG-OACC's first attempt stopped at its gate asserter before
  anything trained, so FLAT-DAG ran second (box 2, 15:37–20:17). No row, bar or arm depends on the order.

## 1. The outcome, as the table decides it

| | s0 | s1 |
|---|---|---|
| PRIMARY H-T1 (ADE6 DAG − FLAT-DAG ≤ −0.2415 m, separated) | **MET: −0.3451 [−0.4421, −0.2506]** | not met: −0.0544 [−0.1605, +0.0547] |
| FLAT BETTER (ADE6 separated-worse) | no | no |
| family rows separated-worse, DAG − FLAT-DAG | **DK-TG +0.0734 [+0.013, +0.150], DK-VIOL +0.0370 [+0.015, +0.062]** | **accel MAE +0.8252 [+0.521, +1.165]** |
| STRATEGIC, paired | not computed: no paired DAG − FLAT-DAG instrument (never counts as passing) | not computed |

**NOT ESTABLISHED** (REFUSED, BLOCKED, FLAT BETTER and HIERARCHY BEATS FLAT do not apply). The PRIMARY fails at s1, and family rows are
separated-worse at both seeds. Committed next, as written: *"B-2's margin was DAgger's, not the hierarchy's. The interface stays the
defect; next is MIXO's committed T-side arm, under its own PREREG"* (§4 resolves which arm that is).

Controls (C-F1 to C-F4, 91 checks): both banks pass K-B0…K-B5 with headers naming that seed's FLAT checkpoint. The pair rule holds (argv
identical but `--out`, `--dagger-p` and the bank flags). **The cut survives the fine-tune**: the checkpoint config reads `o_interface: cut`
and oracle − own is exactly 0 [0, 0] on 2,399 windows at both seeds (max |ΔADE6| 0.0). P10's floor reads exactly 0.0. The reader
reproduces the banked B-2 table (19 rows, max |diff| 0.0; DAG − FLAT −0.5957 / −0.3505).

## 2. The four families (T1-kin; point [95 % CI]; B / W = separated-better / -worse)

| family | row | DAG − FLAT-DAG s0 | s1 | FLAT-DAG − FLAT-DAG-0 s0 | s1 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 (m) | **−0.3451 [−0.442, −0.251] B** | −0.0544 [−0.161, +0.055] – | −0.1522 [−0.218, −0.088] B | −0.1905 [−0.277, −0.120] B |
| LONGITUDINAL | speed MAE | −0.3241 [−0.443, −0.213] B | +0.0451 [−0.066, +0.151] – | +0.0737 W | +0.0600 – |
| | along-track MAE | −0.1621 [−0.258, −0.066] B | +0.0107 [−0.094, +0.126] – | −0.0427 – | −0.0893 B |
| | accel MAE | −0.1192 [−0.454, +0.213] – | **+0.8252 [+0.521, +1.165] W** | +0.3842 W | +0.2798 W |
| | DK-VIOL | **+0.0370 [+0.015, +0.062] W** | −0.0084 [−0.033, +0.013] – | +0.0168 – | +0.0319 W |
| | DK-HW | +0.0320 [−0.265, +0.265] – | +0.0028 [−0.218, +0.223] – | +0.1371 – | −0.1032 – |
| | DK-TG | **+0.0734 [+0.013, +0.150] W** | +0.0005 [−0.032, +0.031] – | −0.0074 – | −0.0103 – |
| LATERAL | heading MAE | −3.1694 [−4.257, −2.257] B | −1.3925 [−2.036, −0.869] B | −1.4594 B | −1.8141 B |
| | yaw-rate MAE | −11.6272 [−16.844, −7.298] B | −0.6107 [−3.019, +2.104] – | +0.4058 – | −2.0253 B |
| | curvature MAE | −0.0461 [−0.070, −0.025] B | −0.0037 [−0.019, +0.012] – | −0.0088 – | −0.0393 B |
| | cross-track MAE | −0.2989 [−0.372, −0.231] B | −0.0791 [−0.147, −0.015] B | −0.1897 B | −0.1998 B |
| | L1 lateral-accel MAE | −2.6848 [−3.923, −1.625] B | −0.1380 [−0.682, +0.419] – | +0.5423 W | +0.0526 – |
| | L2 curvature MAE (GT ≥ 5.2) | −0.0393 [−0.070, −0.015] B | −0.0068 [−0.025, +0.013] – | +0.0040 – | −0.0110 – |
| TACTICAL | TAC-LAT | +0.0600 [+0.033, +0.088] B | +0.0167 [−0.005, +0.040] – | +0.1263 B | +0.0863 B |
| | TAC-LON | +0.0554 [+0.025, +0.088] B | +0.0088 [−0.030, +0.047] – | +0.0092 – | +0.0042 – |
| | lat head F1 (H-T1-HEADS) | −0.0304 [−0.092, +0.028] – | +0.0070 [−0.041, +0.062] – | +0.1104 B | +0.0957 B |
| | lon head F1 (H-T1-HEADS) | +0.0107 [−0.020, +0.041] – | −0.0078 [−0.046, +0.033] – | +0.0340 B | +0.0691 B |
| STRATEGIC | S-POS@30 / @50 / @80 vs CTRV (m), per arm | DAG −1.72 / −4.48 / −9.69 B; FLAT-DAG −1.65 / −4.67 / −9.83 B | DAG −1.87 / −4.90 / −10.33 B; FLAT-DAG −1.63 / −4.62 / −10.29 B | per arm, not a contrast: FLAT-DAG-0 −1.04 / −3.58 / −7.87 B | per arm: FLAT-DAG-0 +0.21 – / −1.11 B / −5.03 B |
| crawl | M1 crawl share | −0.0115 [−0.022, −0.001] B | −0.0133 [−0.023, −0.004] B | −0.0082 B | −0.0058 B |

ADE6 arm means (m, 2,399 windows / 141 episodes): DAG 2.2777 / 2.4565, FLAT-DAG 2.6228 / 2.5109, FLAT-DAG-0 2.7750 / 2.7014. D-FLAT's
guards held: R-T0 FLAT-DAG − FLAT-DAG-0 +0.0254 [−0.0039, +0.0575] / +0.0082 [−0.0205, +0.0357] m, point ≤ +0.0604 at both seeds. The CIs of
the D-FLAT columns other than ADE6 and the head rows are in `read/FLATDAG_READ.json` (`beside.per_seed.*.D-FLAT`). STRATEGIC is reported,
not paused, for v8F.

## 3. What the numbers say (beside the outcome; none of this moves it)

1. **Most of B-2's margin was DAgger's.** DAG − FLAT was −0.596 / −0.351 m (B-2). Against a flat model given the same DAgger recipe it is
   −0.345 / −0.054 m, so the hierarchy keeps 58 % of its margin at s0 and 15 % at s1. The s0 contrast still clears 2F; the s1 one is not
   separated.
2. **DAgger helps flat too, with the same guards held.** FLAT-DAG − FLAT-DAG-0 ADE6 −0.152 / −0.191 m, lat head F1 +0.110 / +0.096, lon head
   F1 +0.034 / +0.069, all separated; R-T0 held. Own-state supervision is a recipe win on both architectures, not a hierarchy property.
3. **The swap's acceleration cost is the recipe's, on both architectures.** FLAT-DAG − FLAT-DAG-0 accel MAE +0.384 / +0.280 (W at both
   seeds); DAG2 − DAG2-0 was +0.58 / +0.60. The s1 row that blocks the claim, DAG − FLAT-DAG accel +0.825, sits on DAG's worse seed (DAG − P10
   accel +0.81 / +1.44, `RESULT_V8F_DAGGER2` §2). DAG-OACC, now training on box 2, is the arm aimed at this row.
4. **Where the hierarchy still leads at both seeds, the lead is in O's use of the code, not in T's decisions.** Heading MAE (−3.17 / −1.39),
   cross-track MAE (−0.30 / −0.08) and M1 crawl share (−0.012 / −0.013) are separated-better at both seeds. T's own heads do not differ
   (lat F1 −0.030 / +0.007, lon F1 +0.011 / −0.008, none separated). FLAT's O does not read T's code, so the most direct reading
   (interpretation, no bar) is that the interface contributes lateral precision. At s0 it comes with closer following (DK-TG +0.073,
   DK-VIOL +0.037, W).
5. **The contrast moves by 0.29 m between seeds**, more than 2F. DAG's mean worsens by 0.18 m from s0 to s1 while FLAT-DAG's improves by
   0.11 m. With one training run per seed the interval cannot say which seed is typical. That is the training-run question
   (`H-ESTIM-SEED-1`), which the episode bootstrap does not answer.

## 4. Next (Rule Zero)

* **The committed next, resolved from the register.** MIXO's T-side line ran KINDX (speed is the carrier at both seeds: gain_speed
  +0.716 / +0.668 m, T0) → SPEEDAVG **MIXED** (the seeds' speed errors correlate 0.71–0.85: systematic, not variance) → SPEEDCAL
  **INFORMATION-LIMITED** (an affine map moves `v_a_ms` RMSE 1.92 → 1.33 m/s and ADE6 not at all; the GT speed cells would cut ADE6 by
  0.72 / 0.67 m, T0). SPEEDCAL's committed next is perception: `PREREG_V8F_LORA.md`, projected at 470–494 h per seed on box 2, so its venue
  rule names the A40.
* **Blocker, named.** The allocation is box 2 and the dev-box CPU (PI: *"No, keep everything on box 2 and the dev-box CPU for now"*). The
  A40 form of the perception arm waits on `D-V8F-SPEED-PERCEPTION`.
* **What runs instead.** (a) A box-2-feasible form of the T-side speed lever, inside LORA's own venue rule (≤ 36 h per seed), drafted under
  its own PREREG. (b) DAG-OACC on box 2 (since 20:21, both gates PASS), aimed at the accel row of §3.3. (c) DAG-ORC (the crawl rows), built
  and gated, waiting for the PI's go.
* **Instrument gaps this read exposed.** A paired DAG − FLAT-DAG S-POS row (STRATEGIC is per arm only), and one same-argv replicate per
  architecture, so that a 0.29 m seed swing can be read against a measured run-to-run floor.

## 5. Provenance

Box 2: chain `chain_v8f_flatdag.sh` at tip `3101dbf1`, 15:37–20:17 (`box2/logs/chain_v8f_flatdag.out`, marker `ZZV8FMINI-DONE`). Each run
took about 37.6 min for 3,000 steps (`summary.json` `seconds` 2,251–2,263). Dev-box CPU: chain `t1read/flatdag_t1_chain.sh` ran 4 / 4
T1-kin rollouts (258–261 s each) and the reader, 20:13–20:28 (`t1read/logs/chain.out`). Before launch the self-test, mutation proof and a
dry run on the real config ran (`t1read/selftest_output.txt`, `mutation_output.txt`, `dry_run_real_config.txt`). The code is
banked byte-identical to the files that ran (sha256 in `t1read/MANIFEST.txt`). Per-episode dumps and tick arrays
(about 72 MB of ticks plus the dumps), the four checkpoints and the bank shards stay on E: (`v8f-flatdag-t1/`, `v8f-flatdag/`), single
copy, not in the repo.
