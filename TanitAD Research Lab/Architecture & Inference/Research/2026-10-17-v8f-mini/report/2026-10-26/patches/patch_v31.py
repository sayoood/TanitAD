"""Report v31: FLAT-DAG (7.22, NOT ESTABLISHED) and DAG-OACC (7.23, LONGITUDINAL FIXED); abstract, 8.1, 8.2, 8.3, 8.4 rows 1u-1x.
Builds the v31 .md from the v30 .md, and the .html from page_template_v30.html (the template is unchanged in v31: no new video).
python patch_v31.py <tip> [report_dir_in [report_dir_out]]
  <tip>            repository tip the results run through (first 8 hex digits are written to the status line)
  report_dir_in    folder holding the v30 V8F_SCIENTIFIC_REPORT.md and page_template_v30.html (default: the folder above this script)
  report_dir_out   folder to write V8F_SCIENTIFIC_REPORT.md and v8f_report.html into (default: report_dir_in)
Every number below is copied from RESULT_V8F_FLATDAG.md or RESULT_V8F_DAG_OACC.md (tables are copied line for line)."""
import os
import re
import sys

TP = sys.argv[1][:8]
HERE = os.path.dirname(os.path.abspath(__file__))
RIN = (sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(HERE)).rstrip("/\\") + "/"
ROUT = (sys.argv[3] if len(sys.argv) > 3 else RIN).rstrip("/\\") + "/"
p = RIN + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()
assert "\r" not in s
assert s.count("**Status:** research report, v30 of 2026-10-27") == 1

# ---- tables, copied line for line from the two RESULT files -------------------------------------------------------------------
FLAT_OUTCOME = """| | s0 | s1 |
|---|---|---|
| PRIMARY H-T1 (ADE6 DAG − FLAT-DAG ≤ −0.2415 m, separated) | **MET: −0.3451 [−0.4421, −0.2506]** | not met: −0.0544 [−0.1605, +0.0547] |
| FLAT BETTER (ADE6 separated-worse) | no | no |
| family rows separated-worse, DAG − FLAT-DAG | **DK-TG +0.0734 [+0.013, +0.150], DK-VIOL +0.0370 [+0.015, +0.062]** | **accel MAE +0.8252 [+0.521, +1.165]** |
| STRATEGIC, paired | not computed: no paired DAG − FLAT-DAG instrument (never counts as passing) | not computed |"""
FLAT_FAMILIES = """| family | row | DAG − FLAT-DAG s0 | s1 | FLAT-DAG − FLAT-DAG-0 s0 | s1 |
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
| crawl | M1 crawl share | −0.0115 [−0.022, −0.001] B | −0.0133 [−0.023, −0.004] B | −0.0082 B | −0.0058 B |"""
OACC_OUTCOME = """| | s0 | s1 |
|---|---|---|
| PRIMARY-ACC (accel MAE, DAG-OACC − DAG, < 0 and separated) | **PASS: −1.6243 [−1.9766, −1.3236]** (means 1.2727 vs 2.8970 m/s²) | **PASS: −2.0450 [−2.5628, −1.6227]** (1.4885 vs 3.5335) |
| KEEP (ADE6, DAG-OACC − DAG, upper bound ≤ +0.0604 m) | PASS: −0.0356 [−0.1258, +0.0423] | PASS: **−0.1363 [−0.2057, −0.0731]**, separated-better |
| LON-CLEAR (no longitudinal row of DAG-OACC − P10 separated-worse) | clear | clear |
| Read A D1 (ADE6 vs P10 separated-better, ≤ −2F) | yes: −1.2012 | yes: −1.1231 |
| Read A D2 (no row separated-worse vs P10) | **fails**: yaw-rate, curvature, L2 | **fails**: yaw-rate, curvature, L2 |
| Read A D3 (every family has a separated-better row) | yes | yes |
| crawl rows (A1: DOMINANT also needs them to pass) | **fail**: M1 +0.0568 [+0.038, +0.080] | **fail**: M1 +0.0604 [+0.039, +0.086] |
| R-T0 (beside: T0 ADE6, DAG-OACC − DAG) | −0.0100 [−0.0329, +0.0130] | −0.0214 [−0.0536, +0.0077] |"""
OACC_FAMILIES = """| family | row | DAG-OACC − P10 s0 | s1 | DAG-OACC − DAG s0 | s1 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 (m) | −1.2012 [−1.395, −1.019] B | −1.1231 [−1.330, −0.916] B | −0.0356 [−0.126, +0.042] – | −0.1363 [−0.206, −0.073] B |
| LONGITUDINAL | speed MAE | −0.7761 [−0.981, −0.600] B | −0.7030 [−0.909, −0.523] B | −0.1996 [−0.281, −0.124] B | −0.4010 [−0.530, −0.295] B |
| | along-track MAE | −0.6809 [−0.844, −0.521] B | −0.5876 [−0.770, −0.412] B | −0.0527 [−0.153, +0.030] – | −0.1304 [−0.254, −0.044] B |
| | accel MAE | −0.8172 [−1.177, −0.500] B | −0.6014 [−0.906, −0.335] B | **−1.6243 [−1.977, −1.324] B** | **−2.0450 [−2.563, −1.623] B** |
| | DK-VIOL | +0.0017 [−0.023, +0.027] – | −0.0034 [−0.024, +0.018] – | −0.0336 [−0.059, −0.012] B | −0.0084 [−0.030, +0.009] – |
| | DK-HW | −0.3318 [−0.694, −0.024] B | −0.4466 [−0.820, −0.169] B | −0.2475 [−0.458, −0.041] B | −0.2080 [−0.439, +0.016] – |
| | DK-TG | −0.0568 [−0.123, +0.016] – | −0.0724 [−0.138, −0.009] B | −0.0725 [−0.145, −0.012] B | +0.0210 [−0.020, +0.086] – |
| LATERAL | heading MAE | −2.5920 [−3.239, −1.975] B | −2.0539 [−2.709, −1.371] B | **+0.7674 [+0.348, +1.265] W** | −0.4548 [−1.093, +0.101] – |
| | yaw-rate MAE | **+3.4726 [+0.660, +6.331] W** | **+8.6844 [+5.169, +12.574] W** | **+3.5126 [+1.429, +5.878] W** | +0.0292 [−1.762, +1.879] – |
| | curvature MAE | **+0.0854 [+0.058, +0.117] W** | **+0.1352 [+0.089, +0.192] W** | **+0.0241 [+0.011, +0.039] W** | +0.0218 [−0.003, +0.053] – |
| | cross-track MAE | −0.9620 [−1.108, −0.832] B | −0.9742 [−1.115, −0.841] B | +0.0173 [−0.015, +0.051] – | −0.0473 [−0.095, +0.020] – |
| | L1 lateral-accel MAE | −2.4618 [−3.810, −1.292] B | −1.8473 [−2.962, −0.850] B | −0.0296 [−0.308, +0.258] – | −0.7120 [−1.204, −0.234] B |
| | L2 curvature MAE (GT ≥ 5.2) | **+0.0443 [+0.023, +0.074] W** | **+0.0529 [+0.028, +0.085] W** | +0.0158 [−0.001, +0.041] – | −0.0075 [−0.030, +0.014] – |
| TACTICAL | TAC-LAT | +0.1905 [+0.151, +0.232] B | +0.1972 [+0.160, +0.236] B | −0.0021 [−0.019, +0.015] – | +0.0121 [−0.010, +0.033] – |
| | TAC-LON | +0.1396 [+0.077, +0.200] B | +0.1371 [+0.075, +0.200] B | +0.0108 [−0.017, +0.039] – | +0.0229 [−0.004, +0.051] – |
| STRATEGIC | S-POS@30 / @50 / @80 vs CTRV (m), per arm | DAG-OACC −1.74 / −4.53 / −9.56 B (DAG −1.72 / −4.48 / −9.69) | DAG-OACC −1.83 / −4.61 / −9.92 B (DAG −1.87 / −4.90 / −10.33) | not computed: no paired S-POS instrument | |
| crawl | M1 crawl share | **+0.0568 [+0.038, +0.080] W** | **+0.0604 [+0.039, +0.086] W** | **+0.0102 [+0.003, +0.017] W** | −0.0011 [−0.007, +0.005] – |"""

# ---- abstract -----------------------------------------------------------------------------------------------------------------
OLD_ABS = "while it falls by 0.10 m at seed 1 (§7.21)."
NEW_ABS = (OLD_ABS + "\n"
    "* **FLAT-DAG, the attribution read, does not establish the hierarchy's lead (§7.22).** The flat control is given the same DAgger"
    " recipe and read at T1-kin (self-action open loop; paired episode-cluster bootstrap, B = 2,000, 141 episodes). DAG's ADE6 margin over it falls from 0.596 / 0.351 m, against FLAT (B-2, §7.20),"
    " to 0.345 / 0.054 m. The PRIMARY bar (≤ −0.2415 m, separated, at both seeds) is met at seed 0 and missed at seed 1, and family rows are"
    " separated-worse at both seeds (DK-TG and DK-VIOL at seed 0, accel MAE at seed 1). The outcome is NOT ESTABLISHED. Own-state"
    " supervision helps the flat model too (FLAT-DAG − FLAT-DAG-0 ADE6 −0.152 / −0.191 m, separated), so most of B-2's margin was DAgger's.\n"
    "* **DAG-OACC** (OACC's acceleration term inside DAG's fine-tune recipe, `--w-acc 49.6969`) **reads LONGITUDINAL FIXED and is the adopted"
    " base (§7.23).** Against DAG at T1-kin (same tier and estimator), accel MAE falls by 56 % / 58 % (2.90 → 1.27 and 3.53 → 1.49 m/s²; −1.6243 [−1.9766, −1.3236]"
    " and −2.0450 [−2.5628, −1.6227], separated), and ADE6 is kept (−0.0356 [−0.1258, +0.0423] and −0.1363 [−0.2057, −0.0731] m, against a"
    " bar of +0.0604 m on the upper end). Every longitudinal row is clean against P10 at both seeds, and ADE6 against P10 is −1.2012 /"
    " −1.1231 m. It is not DOMINANT: yaw-rate, curvature, L2 and the launch-crawl share M1 stay separated-worse against P10 at both seeds.\n"
    "* **What is next.** The open gap is the crawl set. DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`) is aimed"
    " at it, and SPEEDREAD (`PREREG_V8F_SPEEDREAD`) asks whether the speed information O needs is present in T's own input. Both are"
    " pre-registered and not yet read, and no number for either is reported here.")

# ---- section 7.22 and 7.23 ----------------------------------------------------------------------------------------------------
SEC = """### 7.22 FLAT-DAG: the flat control given the same DAgger recipe

* **Setup** (`PREREG_V8F_FLATDAG`, promoted at `9b91b654` before any FLAT-DAG bank, run or row). The question: does the hierarchy beat flat once
  BOTH get own-state supervision? Everything is read at T1-kin: m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms,
  self-action OPEN loop; the lead follows its logged future. R-T0 is at T0 (box 2's standard eval).
  * **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows, per seed. It answers *"would another draw of
    EPISODES say this?"*. No replicate arm was trained, so a separated row is a result about one training run per seed (`H-ESTIM-SEED-1`).
    The planner is deterministic, so there is no inference-seed term.
  * **Arms** (box 2, all four through their launch gates, token commit `3101dbf1`). FLAT-DAG = FLAT_s fine-tuned 3,000 steps with DAG's recipe
    at `--dagger-p 0.1` on FLAT's own bank, `--o-interface cut` kept. FLAT-DAG-0 = the same argv at `--dagger-p 0`. Each bank holds 613,132
    records over 4,512 clips, COVERAGE 0.751 / 0.751. The swap share reached in training is 0.1011 (s0) and 0.1005 (s1), and 0 for both
    FLAT-DAG-0. DAG's T1-kin rollouts are the banked ones (`raw/2026-10-26-dagger-t1`).
  * **Queue order, disclosed.** FLAT-DAG ran second, after DAG-OACC's first attempt stopped at its gate asserter before anything trained.
    No row, bar or arm depends on the order.
* **Outcome [M]** (`RESULT_V8F_FLATDAG.md`; reader `raw/2026-10-27-flatdag/t1read/flatdag_read.py`; all 91 reader controls PASS; output
  `raw/2026-10-27-flatdag/read/FLATDAG_READ.json`, sha256 `7274b532…`):

""" + FLAT_OUTCOME + """

  The outcome is **NOT ESTABLISHED** (REFUSED, BLOCKED, FLAT BETTER and HIERARCHY BEATS FLAT do not apply). The PRIMARY fails at s1, and
  family rows are separated-worse at both seeds. Committed next, as written: *"B-2's margin was DAgger's, not the hierarchy's. The interface
  stays the defect; next is MIXO's committed T-side arm, under its own PREREG"*.
* **Controls** (C-F1 to C-F4, 91 checks, all PASS):
  * both banks pass K-B0…K-B5 with headers naming that seed's FLAT checkpoint, and the pair rule holds (argv identical but `--out`,
    `--dagger-p` and the bank flags);
  * **the cut survives the fine-tune:** the checkpoint config reads `o_interface: cut`, and oracle − own is exactly 0 [0, 0] on 2,399 windows
    at both seeds (max |ΔADE6| 0.0);
  * P10's floor reads exactly 0.0, and the reader reproduces the banked B-2 table (19 rows, max |diff| 0.0; DAG − FLAT −0.5957 / −0.3505).
* **The four families [M]** (T1-kin; point [95 % CI]; B / W = separated-better / -worse):

""" + FLAT_FAMILIES + """

  ADE6 arm means (m, 2,399 windows / 141 episodes): DAG 2.2777 / 2.4565, FLAT-DAG 2.6228 / 2.5109, FLAT-DAG-0 2.7750 / 2.7014. D-FLAT's guards
  held: R-T0 FLAT-DAG − FLAT-DAG-0 +0.0254 [−0.0039, +0.0575] / +0.0082 [−0.0205, +0.0357] m, point ≤ +0.0604 at both seeds. The CIs of the
  D-FLAT columns other than ADE6 and the head rows are in `read/FLATDAG_READ.json` (`beside.per_seed.*.D-FLAT`). STRATEGIC is reported, not
  paused, for v8F.
* **What the numbers say** (beside the outcome; none of this moves it):
  1. **Most of B-2's margin was DAgger's.** DAG − FLAT was −0.596 / −0.351 m (B-2). Against a flat model given the same DAgger recipe it is
     −0.345 / −0.054 m, so the hierarchy keeps 58 % of its margin at s0 and 15 % at s1. The s0 contrast still clears 2F; the s1 one is not
     separated.
  2. **DAgger helps flat too, with the same guards held.** FLAT-DAG − FLAT-DAG-0 ADE6 −0.152 / −0.191 m, lat head F1 +0.110 / +0.096, lon
     head F1 +0.034 / +0.069, all separated; R-T0 held. Own-state supervision is a recipe win on both architectures, not a hierarchy
     property.
  3. **The swap's acceleration cost is the recipe's, on both architectures.** FLAT-DAG − FLAT-DAG-0 accel MAE +0.384 / +0.280 (W at both
     seeds); DAG2 − DAG2-0 (DAgger iteration 2 against its control) was +0.58 / +0.60 (**[I]**, quoted from `RESULT_V8F_FLATDAG.md`). The s1 row that blocks the claim, DAG − FLAT-DAG accel +0.825, sits on DAG's worse seed (DAG −
     P10 accel +0.81 / +1.44, `RESULT_V8F_DAGGER2` §2). DAG-OACC is the arm aimed at this row (§7.23).
  4. **Where the hierarchy still leads at both seeds, the lead is in O's use of the code, not in T's decisions.** Heading MAE (−3.17 / −1.39),
     cross-track MAE (−0.30 / −0.08) and M1 crawl share (−0.012 / −0.013) are separated-better at both seeds. T's own heads do not differ (lat
     F1 −0.030 / +0.007, lon F1 +0.011 / −0.008, none separated). FLAT's O does not read T's code, so the most direct reading (interpretation,
     no bar) is that the interface contributes lateral precision. At s0 it comes with closer following (DK-TG +0.073, DK-VIOL +0.037, W).
  5. **The contrast moves by 0.29 m between seeds**, more than 2F. DAG's mean worsens by 0.18 m from s0 to s1 while FLAT-DAG's improves by
     0.11 m. With one training run per seed the interval cannot say which seed is typical. That is the training-run question
     (`H-ESTIM-SEED-1`), which the episode bootstrap does not answer.
* **Next (Rule Zero).**
  * The committed next resolves, from the register, to the T-side speed line. KINDX: speed is the carrier at both seeds (gain_speed +0.716 /
    +0.668 m, T0). SPEEDAVG: **MIXED**, the seeds' speed errors correlate 0.71–0.85, so the error is systematic, not variance. SPEEDCAL:
    **INFORMATION-LIMITED**, an affine map moves `v_a_ms` RMSE 1.92 → 1.33 m/s and ADE6 not at all, while the GT speed cells would cut ADE6 by
    0.72 / 0.67 m (T0). SPEEDCAL's committed next is perception (`PREREG_V8F_LORA.md`, projected at 470–494 h per seed on box 2), so its venue
    rule names the A40.
  * **Blocker, named.** The allocation is box 2 and the dev-box CPU. The A40 form of the perception arm waits on `D-V8F-SPEED-PERCEPTION`.
  * **What runs instead.** (a) A box-2-feasible form of the T-side speed lever, inside LORA's own venue rule, under its own PREREG:
    `PREREG_V8F_SPEEDREAD` commissions the cheapest experiment first (is the speed information in T's own input?), and it is pre-registered
    and not yet read. (b) DAG-OACC on box 2 (§7.23). (c) DAG-ORC, built and gated at the time of this read and waiting for the PI's go; it has
    since been re-based on DAG-OACC (§7.23).
  * **Instrument gaps this read exposed:** a paired DAG − FLAT-DAG S-POS row (STRATEGIC is per arm only), and one same-argv replicate per
    architecture, so that a 0.29 m seed swing can be read against a measured run-to-run floor.
* **Provenance.** Box 2: chain `chain_v8f_flatdag.sh` at tip `3101dbf1`, marker `ZZV8FMINI-DONE`; each run took about 37.6 min for 3,000 steps.
  Dev-box CPU: 4 / 4 T1-kin rollouts (258–261 s each) and the reader. Before launch the self-test, mutation proof and a dry run on the real
  config ran. Per-episode dumps, tick arrays, the four checkpoints and the bank shards stay on E: (`v8f-flatdag-t1/`, `v8f-flatdag/`), single
  copy, not in the repo.

### 7.23 DAG-OACC: the acceleration term inside DAgger's recipe fixes the longitudinal family and keeps position

* **Setup** (`PREREG_V8F_DAG_OACC` with **Amendment 1**, `A1-2026-10-27`, landed `db6a1a43` before any DAG-OACC T1-kin rollout or value).
  * **Arms** (box 2, r2 chain at tip `aeb54468`, both gates PASS). DAG-OACC_s = DAG_s's argv verbatim plus `--w-acc 49.6969`. W was computed
    once, by the `waccoacc` stage on DAG_s0, and C-O3 matched the offline recomputation. The swap share reached 0.1013 (s0) and 0.1008 (s1).
    Control = DAG_s, banked (a one-flag pair, C-O1). The first attempt stopped at its gate asserter before anything trained.
  * **Tier and estimator:** T1-kin (self-action OPEN loop; m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms; the lead
    follows its logged future). R-T0 is at T0. Paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows, per seed. The control
    is a different training run (DAG), and the training-run question is carried by the two seeds and F = 0.1207 m (`H-ESTIM-SEED-1`). The
    planner is deterministic, so there is no inference-seed term.
* **Outcome [M]** (`RESULT_V8F_DAG_OACC.md`; reader `raw/2026-10-27-dag-oacc/t1read/dagoacc_read.py`, coding the amendment; enumeration 0 unplaced
  cells; all 60 reader controls PASS; output `raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json`, sha256 `9e6bd4f6…`):

""" + OACC_OUTCOME + """

  The outcome is **LONGITUDINAL FIXED** (PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not DOMINANT, because D2 and the crawl rows fail at
  both seeds). The unamended table gives the same word (`outcome_unamended`, descriptive), so the amendment moved no outcome here. Committed
  next, as amended: *the term is adopted and DAG-OACC becomes the base; next is the lever of the row still open*. The open rows are crawl rows,
  so the lever is **DAG-ORC re-based on DAG-OACC**, whose Amendment 1 is landed at `6cd741b6`, before any DAG-ORC run. Under Read A's own table
  (`PREREG_V8F_DAG_REREAD` §5) the same rows read MARGIN, NOT DOMINANT.
* **The four families [M]** (T1-kin; point [95 % CI]; B / W = separated-better / -worse):

""" + OACC_FAMILIES + """

  STRATEGIC is reported, not paused, for v8F. The T0 four-family leaves (box 2's standard eval, DAG-OACC against DAG) are in `DAGOACC_READ.json`
  (`beside.per_seed.*.T0.four_families_T0`). They are descriptive only: no paired T0 four-family instrument exists. K4 is NOT RUN, because the
  PREREG scheduled no logged-bucket rollout or fresh-age T0 eval.
* **What the numbers say** (beside the outcome; none of this moves it):
  1. **The acceleration error is fixed, and position comes along.** Accel MAE falls by 56 % / 58 % (2.90 → 1.27 and 3.53 → 1.49 m/s²). Against
     the kinematic floor P10 it flips from separated-worse for DAG (+0.81 / +1.44, `RESULT_V8F_DAGGER2` §2) to separated-better (−0.82 /
     −0.60). ADE6 is kept at s0 and improves at s1 (−0.136, separated). The effect is 2.6–3.2 times the cross-seed spread of DAG's own accel row
     (0.63 m/s²). That comparison is context, with no bar.
  2. **Every longitudinal row is clean against P10 at both seeds.** Speed, along-track and accel MAE and headway are separated-better; the time
     gap is better or not separated; DK-VIOL is not separated. Against DAG, speed MAE is separated-better at both seeds (−0.20 / −0.40), so the
     term fixed target speed as well as smoothness.
  3. **What stays open is the crawl set, and at s0 the term costs some lateral.** Against P10, yaw-rate, curvature, L2 and M1 are separated-worse
     at both seeds. Against DAG, s0 reads heading +0.77, yaw-rate +3.51, curvature +0.024 and M1 +0.010, all separated-worse; s1 reads none of
     them worse. The acceleration term did not repair the crawl, so NEXT-CRAWL's COMPOSE condition (DAG-OACC − DAG separated-better on the crawl
     rows) is not met.
  4. **The hierarchy's decisions are untouched.** TAC-LAT and TAC-LON against DAG are not separated at either seed. The lever acts on O's
     execution, as designed.
* **Next (Rule Zero).**
  * **Adopted.** DAG-OACC is the base.
  * **The crawl rows go to DAG-ORC, re-based** (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`). Form: COMPOSE. A = DAG-OACC's argv plus O-RECOVER's
    perturbation, behind a new default-off composition flag. CTRL = DAG-OACC, banked. REP is new. Launch blocker: the composition flag and its
    reachability proofs (in build at the read). It is pre-registered and not yet read.
  * **DAgger iteration 2**, the other lever v30 named for the crawl rows, read HARMS (`RESULT_V8F_DAGGER2`, cited by `PREREG_V8F_DAG_ORC`; **[I]**,
    not re-read here).
  * **Evidence videos.** The PREREG re-renders them only on DOMINANT, so they are not re-rendered on this read.
* **Provenance.** Box 2: chain `chain_v8f_dagoacc_r2.sh` ran 20:21–21:51, marker `ZZV8FMINI-DONE`; each run took 41 and 42 min with its T0 eval. The
  r2 asserter is banked at `raw/2026-10-27-oaccflat-queue/r2/`. Dev-box CPU: 2 / 2 T1-kin rollouts and the reader; DAG's and P10's runs are the
  banked ones. Reader checks before launch: self-test 287 / 287; mutation proof 127 / 127 killable mutants killed, with 2 real-run-only branches
  UNVERIFIED; real-config dry run `ZZDAGOACCT1-DRY-OK`. Per-episode dumps, tick arrays and checkpoints stay on E: (`v8f-dagoacc-t1/`,
  `v8f-dagoacc/`), single copy, not in the repo.

"""
ANCHOR_8 = "\n---\n\n## 8. Discussion"

# ---- 8.1 ----------------------------------------------------------------------------------------------------------------------
OLD_81 = "It is not dominant: curvature, L2 and the launch crawl remain worse than the floor."
NEW_81 = (OLD_81 + "\n10. **Given the same DAgger recipe, the hierarchy's lead over flat is not established** (§7.22; T1-kin, paired episode-cluster bootstrap). Own-state supervision helps the flat"
          " control as well (ADE6 −0.152 / −0.191 m, separated). DAG's margin over it is −0.3451 m at seed 0 (past the bar) and −0.0544 m at seed 1"
          " (not separated), with family rows separated-worse at both seeds, so item 9's comparison with FLAT is confounded by DAgger. The hierarchy"
          " keeps 58 % of its B-2 margin at seed 0 and 15 % at seed 1.\n"
          "11. **The acceleration term fixes the longitudinal family on DAG's recipe, and position comes along** (§7.23; T1-kin, same estimator). DAG-OACC is the adopted"
          " base: accel MAE falls by 56 % / 58 % against DAG (separated), ADE6 is kept, and no longitudinal row is separated-worse than P10 at either"
          " seed. The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10); the arms aimed at it are pre-registered and not yet read.")

# ---- 8.2 ----------------------------------------------------------------------------------------------------------------------
OLD_82 = "M2b − FLAT −0.036 / −0.003 m ADE6, n.s. (§7.15)."
NEW_82 = (OLD_82 + "\n* That the **hierarchy, rather than the DAgger recipe, earns DAG's margin over flat**. Given the same recipe, DAG − FLAT-DAG ADE6 is"
          " −0.3451 [−0.4421, −0.2506] m at seed 0 (past the bar) and −0.0544 [−0.1605, +0.0547] m at seed 1 (not separated), and family rows are"
          " separated-worse at both seeds. The outcome is NOT ESTABLISHED (§7.22; T1-kin), and one training run per seed cannot say which seed is typical.")

# ---- 8.3 ----------------------------------------------------------------------------------------------------------------------
OLD_83 = ("**Update (v30).** On the model's own motion (T1-kin), DAG with HACC's post-map is two row-families away from dominance over the"
          " floor:")
NEW_83 = ("**Update (v31).** On the model's own motion (T1-kin), the adopted base is now DAG-OACC, and its open rows are lateral and crawl rows:\n"
          "* every longitudinal row is clean against the floor P10 at both seeds, both tactical rows are separated-better, S's checkpoints beat a"
          " kinematic arc at 30 / 50 / 80 m, and ADE6 is kept (§7.23);\n"
          "* yaw-rate, curvature, L2 and the launch crawl M1 remain separated-worse at both seeds, so the arm is not DOMINANT;\n"
          "* the arms aimed at that crawl set are DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`) and SPEEDREAD"
          " (`PREREG_V8F_SPEEDREAD`), both pre-registered and not yet read;\n"
          "* DAgger iteration 2, the other lever v30 named, read HARMS (`RESULT_V8F_DAGGER2`, cited by `PREREG_V8F_DAG_ORC`; **[I]**).\n\n"
          "Attribution to the hierarchy is not established: against a flat model given the same DAgger recipe, DAG's ADE6 margin clears its bar at"
          " seed 0 and not at seed 1 (§7.22).\n\n" + OLD_83)
OLD_83B = ("The levers aimed at those rows are DAgger iteration 2 and DAG-OACC, both on box 2. Attribution to the hierarchy waits on FLAT-DAG. ")
NEW_83B = ("\nAt v30, the levers aimed at those rows were DAgger iteration 2 and DAG-OACC, both on box 2, and attribution to the hierarchy waited on"
           " FLAT-DAG; v31 reads DAG-OACC and FLAT-DAG (above). ")

# ---- 8.4 rows -----------------------------------------------------------------------------------------------------------------
OLD_1T = " Next: DAG-OACC, DAgger iteration 2 and FLAT-DAG on box 2 (PI go) |\n| 2 | **LoRA on DINOv3**"
NEW_1T = (" Next: DAG-OACC, DAgger iteration 2 and FLAT-DAG on box 2 (PI go); DAG-OACC and FLAT-DAG are read in rows 1u and 1v |\n"
          "| 1u | **FLAT-DAG**: FLAT given DAG's recipe (3,000 steps, `--dagger-p 0.1` on FLAT's own bank, `--o-interface cut` kept), against FLAT-DAG-0"
          " (`--dagger-p 0`) and DAG | PRIMARY H-T1: ADE6 DAG − FLAT-DAG $\\le-2F=-0.2415$ m, separated, at both seeds. HIERARCHY BEATS FLAT also needs"
          " no family row separated-worse; the order is REFUSED, BLOCKED, FLAT BETTER, HIERARCHY BEATS FLAT, NOT ESTABLISHED | **NOT ESTABLISHED**"
          " (§7.22; `RESULT_V8F_FLATDAG.md`, 91 / 91 controls PASS, T1-kin): −0.3451 [−0.4421, −0.2506] m (s0, met) / −0.0544 [−0.1605, +0.0547] m"
          " (s1, not met); separated-worse family rows: DK-TG +0.0734 and DK-VIOL +0.0370 at s0, accel MAE +0.8252 at s1. FLAT-DAG − FLAT-DAG-0 ADE6"
          " −0.152 / −0.191 m. Next: SPEEDREAD (row 1x) |\n"
          "| 1v | **DAG-OACC**: DAG's fine-tune argv verbatim plus `--w-acc 49.6969` (OACC's acceleration term), against DAG (banked) | PRIMARY-ACC accel"
          " MAE DAG-OACC − DAG < 0, separated; KEEP ADE6 upper bound ≤ +0.0604 m (+F/2); LON-CLEAR no longitudinal row of DAG-OACC − P10"
          " separated-worse; all at both seeds. DOMINANT also needs Read A's D1, D2 and D3 and the crawl rows (Amendment 1, `A1-2026-10-27`) |"
          " **LONGITUDINAL FIXED** (§7.23; `RESULT_V8F_DAG_OACC.md`, 60 / 60 controls PASS, T1-kin): accel MAE −1.6243 [−1.9766, −1.3236] / −2.0450"
          " [−2.5628, −1.6227]; KEEP PASS (−0.0356 / −0.1363 m); LON-CLEAR clear at both seeds; not DOMINANT (D2 and the crawl rows fail at both"
          " seeds). Adopted as the base. Next: DAG-ORC re-based on DAG-OACC (row 1w) |\n"
          "| 1w | **DAG-ORC, re-based on DAG-OACC** (COMPOSE: DAG-OACC's argv plus O-RECOVER's perturbation behind a new default-off composition flag;"
          " CTRL = DAG-OACC, banked; REP new) | the PREREG's outcome table (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`), aimed at the crawl rows:"
          " yaw-rate, curvature, L2 and M1 | **pre-registered, not yet read**. Launch blocker: the composition flag and its reachability proofs |\n"
          "| 1x | **SPEEDREAD**: is the per-window speed information O needs present in T's own input? A read-out of the 5 GT speed cells, computed from"
          " T's input and fitted on training clips only, replaces T's speed cells in the code O reads; no training of the driving model; dev-box"
          " CPU | the PREREG's decision table: INFO IN INPUT / PARTIAL / NOT IN INPUT / MIXED (`PREREG_V8F_SPEEDREAD`, claim row"
          " `H-V8F-SPEEDREAD-1`) | **pre-registered, not yet read** (Amendment 1, `A1-2026-10-27`, before any SPEEDREAD number). Commissioned"
          " by FLAT-DAG's committed next (`RESULT_V8F_FLATDAG.md` §4 (a)); it runs first because it is the cheapest experiment that separates"
          " the speed being in T's input from its not being there |\n"
          "| 2 | **LoRA on DINOv3**")

E = [(OLD_ABS, NEW_ABS), (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + "\n" + ANCHOR_8), (OLD_81, NEW_81), (OLD_82, NEW_82),
     (OLD_83, NEW_83), (OLD_83B, NEW_83B), (OLD_1T, NEW_1T)]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
lowered = [b.lower() for _, b in E]
assert all("closed loop" not in b and "closed-loop" not in b and "drives" not in b for b in lowered)
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert all(re.search(r"\b" + w + r"\b", b) is None for b in lowered), w
assert s.count("**Status:** research report, v30 of 2026-10-27") == 1
s = re.sub(r"\*\*Status:\*\* research report, v30 of 2026-10-27, results through repository tip `([0-9a-f]{8})` \(branch",
           lambda m: f"**Status:** research report, v31 of 2026-10-27, results through repository tip `{TP}` (v30: `{m.group(1)}`; branch", s, count=1)
assert "research report, v31 of 2026-10-27" in s
assert "\r" not in s and "</script" not in s.lower()
open(ROUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
t = open(RIN + "page_template_v30.html", encoding="utf-8").read()
assert t.count("%%MD%%") == 1
open(ROUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(t.replace("%%MD%%", s))
print("report v31 built", len(s))
