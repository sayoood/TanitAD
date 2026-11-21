# RESULT_V8F_SPEEDSUB_T1: does a causal ego read-out in O's five speed cells, at every replan tick, move DAG-OACC at T1-kin? Outcome **NULL** (label NULL-ONE: one hit of four runs, at the slow-T draw REP s0)

Claim row `H-V8F-SPEEDSUB-T1-1`; the floor rule is its beside hypothesis `H-V8F-SPEEDSUB-T1-2`. Pre-registration `Project Steering/PREREG_V8F_SPEEDSUB_T1.md`, sha256 `69187967…` at tip `266b470e`: promoted at `842ec2ac` (09:00:08), before any T1-kin substitution rollout and before its read-out was fitted; **Amendment 1** (`A1-2026-10-28`: guard G-VIOL and four settled readings), landed at `4cec09e7` (10:17:23), before the smoke (10:21:44) and before any rollout of this experiment. The instrument was built against that text (`raw/2026-10-28-speedsub-t1-build/`) and the read-out `mlp_ego_c` fitted once (`readout_c.pt`, sha256 `799f65ed…`). Chain run 2026-10-28 10:21:41 to 11:09:29 (47 min 48 s) on the dev box (CPU, `CUDA_VISIBLE_DEVICES=-1`, OMP / MKL 8): smoke, twelve decisive rollouts, the reader, and four beside rollouts. The pre-registered reader `speedsub_t1_read.py` (sha256 `9d3c2792…`) ran after the last decisive rollout (stage `read` done 10:59:49; the reader's own wall 454.4 s), printed `ZZSPEEDSUB-T1-READ-DONE NULL NULL-ONE`, and **all 158 control flags PASS**. Output `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json` (sha256 `b3b72325…`), committed at `266b470e` (11:12:21), beside it `SPEEDSUB_T1_BESIDE.json` (sha256 `cce47a69…`). **The outcome word is not copied:** section 1 recomputes the hits, the four guards, the PREREG section 6 table and the section 6.2 floor rule from the per-run cells, in code written from the PREREG text, and `gen_result.py` stops on any mismatch with the reader. Every number below is MEASURED (ours; artifact path in `raw/2026-10-28-speedsub-t1-result/audit/NUMBERS_AUDIT.md`) unless marked INHERITED (another document, not re-measured here), DERIVED (arithmetic shown in the audit), ESTIMATED or PUBLISHED.

**Summary** (each value is in the section named; evidence class and tier as stated there).

* **Outcome NULL, label NULL-ONE (section 1).** A hit needs `egoc − identity` ADE6 separated-better and at or below −0.06035 m; PASS needs it at all four runs; the read has 1 hit of 4 runs (REP s0, −0.1949 [−0.2619, −0.1313] m). No control fails and no guard fires.
* **The healthy draws do not move (section 5).** CTRL s0 −0.0233 [−0.0629, +0.0156], CTRL s1 −0.0076 [−0.0500, +0.0349], REP s1 −0.0333 [−0.0832, +0.0131] m: all intervals include 0; M1 crawl share moves at no run.
* **The hit is a repair of a slow-T draw (section 4).** At REP s0 8 rows separate better and `VIOL-ANY` separates worse (+0.0421); floor rule NOT COLLAPSE (the ADE6 gap to CTRL s0 0.4025 → 0.2309 m).
* **The logged-lane read-out is flat too (section 6, beside).** +0.0008 and −0.0185 m at CTRL s0 and s1, not separated.
* **Next (section 7).** Committed: PREREG section 8 (a). Master Mind's ranking (ESTIMATED / INFERENCE): (a) stays registered and is not run now; the largest measured crawl lever is DAG-ORC's perturbation, and its registered route (LAUNCH-INTENT, box 2) waits on the PI's go.

* **Tier.** READ.json `tier`: `T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan). Never 'closed loop', never T2.` So every decisive row is **T1-kin, a self-action open loop** (the executed plan feeds the next tick; the lead follows its logged future and the cameras are logged), never closed loop and never T2. The `gt`-style ceiling of SPEEDSUB (T0) has no counterpart here: no row of this read is T0. The one T0 item in this document is the INHERITED speed-cell diagnostic of REP s0 (section 4).
* **Estimator.** READ.json `estimator`: `taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per run, paired. It answers 'would another draw of EPISODES say this?'. The training-run question is answered by the four runs (two seeds x two replicates); the planner is deterministic, so there is no inference-seed term.` It answers *"would another draw of EPISODES say this?"* (141 episodes / 2,399 windows per run, paired against the same run's `identity` rollout, or against the P10 floor). The four runs (CTRL s0 / s1 = DAG-OACC at trainer seeds 0 / 1; REP s0 / s1 = DAG-OACC's argv at trainer seeds 10 / 11) are four training runs of one recipe and carry *"would another TRAINING RUN say this?"* at n = 4 (`H-ESTIM-SEED-1`). The read-out is fitted **once**: *"would another READ-OUT FIT say this?"* is not answered. The planner is deterministic, so there is no inference-seed term.
* **Base and arms.** X = DAG-OACC, four checkpoints, sha256 prefixes CTRL s0 `67265ec1…`, CTRL s1 `90b01474…`, REP s0 `b03fd6ef…`, REP s1 `77853e56…` (pins.json `ckpts`; K-SHA true at all 12 rollouts). No model is trained. Per run: `identity` (X's own cells through the hook; the paired base), **`egoc`** (the arm: `mlp_ego_c(ego block at the tick)`, standardised by X's `cons_mu` / `cons_sd`, into code indices [46, 56, 57, 58, 59] at every replan tick, `--limit causal`) and `egoshufc` (the same cells permuted across the 2,399 windows by `numpy.random.default_rng(20261029).permutation(2399)`, sha256 `64785f8a…`; the deliberate-regression arm). Beside, CTRL only: `identity_l` and `egol` in the `--limit logged` lane with the banked `mlp_ego` (`a205609a…`).
* **The primary lane is not the open channel.** `--limit causal` feeds the bucket of the **past** 20 rows of measured ego speed (`causal_bucket`). SPEEDSUB's PASS was read with the **logged** `limit` lane, the bucket of the ego's own realised **future** max speed (`D-SPEEDBAND-EGO-ECHO-1`, a declared open channel). The logged lane appears only beside the table (section 6) and decides nothing.
* **Coverage.** The four families are in section 3, per family and per run, never pooled. Named gaps, per family and with the reason: S-POS across arms (STRATEGIC), the anti-echo block (LONGITUDINAL) and goal-head F1 at T1-kin (TACTICAL). The outcome above uses none of them.

## 1. The outcome, as the pre-registered table decides it

**Hit rule (PREREG section 6):** a hit at run X is `egoc − identity` ADE6 **separated-better (CI upper end < 0) and a point at or below −0.06035 m**. PASS needs a hit at all four runs.

| ADE6, `egoc − identity` (m; point [95 % CI]) | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| point [CI] | −0.0233 [−0.0629, +0.0156] | −0.0076 [−0.0500, +0.0349] | **−0.1949 [−0.2619, −0.1313]** | −0.0333 [−0.0832, +0.0131] |
| separated-better (CI upper end < 0) | no | no | yes | no |
| point ≤ −0.06035 | no | no | yes | no |
| **hit** | no | no | **yes** | no |
| point as a multiple of the bar (DERIVED) | −0.39 × | −0.13 × | −3.23 × | −0.55 × |
| mean ADE6, `identity` → `egoc` (m) | 2.2421 → 2.2189 | 2.3201 → 2.3125 | 2.6446 → 2.4497 | 2.3317 → 2.2984 |
| hits | 1 of 4 | | | |

**The table's walk (REFUSED first, then HARMS, then PASS, PARTIAL, NULL):**

| step (PREREG section 6, with Amendment 1) | reading | fires |
|---|---|---|
| REFUSED: a control fails, or a run is missing | 158 of 158 control flags true; 4 of 4 runs present | no |
| HARMS: a guard fires (G-ADE, G-WORSE, G-NEW or G-VIOL) | none of the four fires (guard table below) | no |
| PASS: no guard; a hit at all four runs | 1 of 4 | no |
| PARTIAL: no guard; hits at both CTRL seeds | CTRL s0 no, CTRL s1 no | no |
| **NULL**: anything else | hits at 1 of 4 runs, not at both CTRL seeds: label **NULL-ONE** (the label describes, the table decides) | **yes** |

**Committed next, verbatim from READ.json:** *"next: section 8 (a), the history-aware causal read-out"* (section 7 gives it, its cost, and the Master Mind's ranking beside it).

### The guards, recomputed from the cells

| guard | rule (PREREG) | reading, recomputed | fires |
|---|---|---|---|
| G-ADE | `egoc − identity` ADE6 not separated-worse at any run | separated-worse at 0 of 4 runs | no |
| G-WORSE | no Read-A row (19 rows, ADE6 included) has `egoc − identity` separated-worse at ≥ 3 of the 4 runs | the most any row reaches is 1 of 4 runs; three rows are separated-worse at one run each, all at CTRL s1: accel MAE +0.1598 [+0.0743, +0.2611] W, yaw-rate MAE +1.3443 [+0.2401, +2.5747] W, L1 lateral-accel MAE +0.3627 [+0.1230, +0.6572] W | no |
| G-NEW | no row with `identity − P10` not separated-worse becomes separated-worse for `egoc − P10` at both CTRL seeds (Amendment 1 item 3: read per seed, must hold at both) | rows with `egoc − P10` separated-worse at both CTRL seeds: curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share; each is **also separated-worse under `identity − P10` at both seeds** (the crawl gap to P10 is there before the substitution), so none is new. The reader's coded rule (identity not separated-worse at ≥ 1 CTRL seed) and the Amendment 1 reading (not separated-worse at each) both give the empty set | no |
| G-VIOL (Amendment 1) | `VIOL-ANY` not separated-worse at ≥ 3 of the 4 runs | separated-worse at 1 of 4 runs (REP s0: +0.0421 [+0.0246, +0.0612] W) | no |

**Sensitivity (DERIVED, same code).** Without G-VIOL the word is the same (NULL, NULL-ONE); under the Amendment 1 reading of G-NEW the word is the same. The two halves of the hit rule name the same single run: separated-better alone gives REP s0 only, and the bar alone (no interval) gives REP s0 only (no other run's point reaches −0.06035). So neither the interval rule nor the bar decides the word.

### How to read this NULL (reported as written; none of it changes the word)

* **NULL is "the bar was not cleared at all four runs", not "the effect is zero".** At CTRL s0 and REP s1 the interval's lower end (−0.0629, −0.0832) lies beyond −0.06035: the data at those runs do not exclude an ADE6 cut as large as the bar. At CTRL s1 the lower end (−0.0500) lies inside the bar, so a bar-sized cut is excluded there at the 95 % level (the episode question). All three points are small (the multiples of the bar are in the table above) and their intervals include 0.
* **The one hit is the slow-T draw.** REP s0's T predicts all five speed cells 0.73 to 1.17 m/s slow at T0 (INHERITED, section 4). PREREG section 10 limit 4 reads this in advance: a hit there "partly measures repair of that draw", which is why both CTRL seeds were required as well. Neither CTRL seed hit.
* **The regression arm reads its known value.** `egoshufc − identity` ADE6 is separated-worse at all four runs (K-REG, section 2): the instrument sees a harm of about 4 m when O is fed shuffled cells, so the small `egoc` movements at the healthy draws are not a blind instrument.

### Controls (each must read its known value before any row is read; any failure is REFUSED)

| control | must read | reading (MEASURED, all 4 runs) | catches |
|---|---|---|---|
| K-ID (4) | `identity` per-window ADE6 equals the banked own rollout within 1e-4 m | max absolute difference in ADE6 0.0 m, and over every `cl` coordinate 0.0 m (bit-equal) | hook drift, a thread-count or tool-tree change |
| K-ZERO (4) | the reader's `identity − identity` reads exactly 0.0 on every computed row | delta = lo = hi = 0.0 on every computed row of the 4 runs (asserted here from `tables.identity-identity`) | a reader that manufactures differences |
| K-NONSPEED (1; 12 values) | at every (window, tick) the 55 non-speed code dimensions O reads equal T's own | max absolute difference 0.0 at all 12 rollouts | a wider substitution than declared |
| K-REC (4) | the cells O reads in `egoc` equal `mlp_ego_c` on the recorded ego block, standardised, ≤ 1e-5 | max 1.9073e-06 (O-read side 1.9073e-06); 7,197 rows per run | a substitution that is not the read-out |
| K-REACH (4) and K-REACH-PLAN (4) | `egoc` moves the cells away from T's own on ≥ 99 % of (window, tick); `egoshufc` differs from `egoc` on ≥ 95 %; the `egoc` executed plan differs from `identity` | moved fraction 1.0, 1.0, 1.0 at all 4 runs | a substitution that never reaches O |
| K-SHUF (4) | the permutation sha is the registered one; fixed points ≤ max(3, 1 %) | recorded = recomputed = registered `64785f8a…`; 0 fixed points of 2,399 | an unregistered shuffle |
| K-FLOOR (12) and K-WIN (12) | the P10 floor arm, windows and GT equal the banked rollout bit for bit | max absolute difference 0.0 at all 12 rollouts; 2,399 windows / 141 episodes | a patch reaching the floor |
| K-SHA (12) and K-ALIGN (12) | checkpoints, `mlp_ego_c`, tool, wrapper, instrument equal their pins; the tick rows align with the tool's | true at all 12 rollouts; 7,197 tick rows each (2,399 windows × 3 ticks) | a swapped file, a misaligned record |
| K1, K2, K3, K5, K6, K7, GRID (84) | the T1 heads pipeline's own per-rollout controls (inherited from the DAG-ORC reader's layer; their definitions are in that reader and are not restated here). K7: the rebuilt executed track equals the dumped `cl` | all true at all 12 rollouts; K7 max absolute difference in `cl` 0.0, pose deviation ≤ 3.02e-05 m (float32 class) | pipeline drift |
| **K-REG** (1) | `egoshufc − identity` ADE6 separated-worse at all four runs | CTRL s0 +4.2928 [+3.4386, +5.1945]; CTRL s1 +3.9857 [+3.2389, +4.7695]; REP s0 +4.4851 [+3.5747, +5.4322]; REP s1 +4.1886 [+3.3996, +5.0549] | an instrument that cannot see a harm |

**Controls before any rollout (the fit; INHERITED from the build, MEASURED there in `raw/2026-10-28-speedsub-t1-build/evidence/fit_c.json`, sha256 `9aaa5edd…`):** C-FIT PASS at 9.412e-06 m/s against the 1e-5 tolerance (**a thin margin, disclosed**); C-CAUSAL 0 mismatches against the tool's own `build_item --limit causal` at tick 0 on 2,399 windows and at ticks 1 and 2 on 600 each; C-COVER 306,566 of 306,566 TRAIN windows carry a finite past (causal-bucket agreement with the logged bucket 76.7 %); C-DISJ eval ∩ fit clips = ∅ (141 eval clips, 4,512 fit clips); C-PARITY 4,512 clips / 306,566 windows; C-INSTR PASS. The smoke (`--max-windows 200`, CTRL s0, `identity` and `egoc`, 600 tick rows) passed K-NONSPEED, K-REC (2.384e-07), K-REACH, K-SHA and K-ALIGN at 10:21:44–10:23:53; the CTRL s0 preflight passed before CTRL s1 was rolled (10:31:00).

## 2. The rows: `egoc − identity` per run, on every row (T1-kin; point [95 % CI]; B / W = separated-better / separated-worse, – = not separated)

Each cell is the paired episode-cluster bootstrap of the arm minus the same run's `identity` rollout on the same windows (B = 2,000, seed 0). Direction: an error row is better when lower; TAC-LAT and TAC-LON (agreement) are better when higher. `n` is windows / episodes (DK rows are the LEAD windows: n differs by row and run; lateral rows exclude tangent-less steps). Rows are grouped by the family they belong to; the family tables and the gaps are in section 3.

| row | family | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|---|
| ADE6 | PRIMARY | −0.0233 [−0.0629, +0.0156] – | −0.0076 [−0.0500, +0.0349] – | −0.1949 [−0.2619, −0.1313] B | −0.0333 [−0.0832, +0.0131] – |
| speed MAE | LONGITUDINAL | −0.0072 [−0.0371, +0.0209] – | +0.0179 [−0.0143, +0.0516] – | −0.1061 [−0.1495, −0.0664] B | −0.0193 [−0.0597, +0.0202] – |
| along-track MAE | LONGITUDINAL | −0.0281 [−0.0715, +0.0146] – | −0.0119 [−0.0616, +0.0365] – | −0.2064 [−0.2785, −0.1393] B | −0.0367 [−0.0886, +0.0129] – |
| accel MAE | LONGITUDINAL | +0.0387 [−0.0176, +0.0916] – | +0.1598 [+0.0743, +0.2611] W | −0.0732 [−0.1463, −0.0057] B | −0.0233 [−0.1388, +0.0937] – |
| DK-VIOL | LONGITUDINAL | +0.0000 [−0.0081, +0.0076] – | −0.0034 [−0.0144, +0.0065] – | +0.0101 [+0.0000, +0.0215] – † | −0.0034 [−0.0124, +0.0043] – |
| DK-HW | LONGITUDINAL | +0.0036 [−0.1355, +0.1564] – | +0.0091 [−0.1350, +0.1376] – | −0.0621 [−0.2061, +0.0685] – | −0.0047 [−0.1682, +0.1280] – |
| DK-TG | LONGITUDINAL | +0.0394 [−0.0233, +0.1431] – | −0.0149 [−0.0366, +0.0062] – | −0.0133 [−0.0391, +0.0102] – | +0.0016 [−0.0342, +0.0345] – |
| heading MAE | LATERAL | +0.1714 [−0.1698, +0.4806] – | +0.0260 [−0.2861, +0.2939] – | −0.5464 [−1.0054, −0.1855] B | +0.0769 [−0.1957, +0.3433] – |
| yaw-rate MAE | LATERAL | −0.4443 [−2.4850, +1.1922] – | +1.3443 [+0.2401, +2.5747] W | −3.7761 [−5.9452, −1.9696] B | +0.0811 [−0.8026, +0.9589] – |
| curvature MAE | LATERAL | −0.0071 [−0.0220, +0.0056] – | −0.0076 [−0.0337, +0.0110] – | −0.0463 [−0.0747, −0.0225] B | +0.0045 [−0.0072, +0.0173] – |
| cross-track MAE | LATERAL | +0.0057 [−0.0111, +0.0224] – | +0.0034 [−0.0166, +0.0222] – | −0.0056 [−0.0229, +0.0124] – | −0.0064 [−0.0252, +0.0109] – |
| L1 lateral-accel MAE | LATERAL | +0.0585 [−0.1062, +0.2145] – | +0.3627 [+0.1230, +0.6572] W | −0.1063 [−0.2761, +0.0587] – | −0.0644 [−0.3605, +0.1731] – |
| L2 curvature MAE (GT>=5.2) | LATERAL | −0.0004 [−0.0175, +0.0145] – | +0.0104 [−0.0023, +0.0240] – | −0.0347 [−0.0674, −0.0105] B | −0.0082 [−0.0218, +0.0022] – |
| TAC-LAT | TACTICAL | +0.0000 [−0.0092, +0.0092] – | +0.0054 [−0.0104, +0.0200] – | +0.0050 [−0.0071, +0.0187] – | +0.0063 [−0.0033, +0.0163] – |
| TAC-LON | TACTICAL | +0.0121 [−0.0092, +0.0359] – | −0.0008 [−0.0325, +0.0321] – | +0.0158 [−0.0050, +0.0367] – | +0.0046 [−0.0179, +0.0280] – |
| S-POS@30 | STRATEGIC | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| S-POS@50 | STRATEGIC | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| S-POS@80 | STRATEGIC | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| M1 crawl share | crawl | −0.0007 [−0.0059, +0.0045] – | +0.0021 [−0.0021, +0.0063] – | −0.0052 [−0.0115, +0.0007] – | −0.0003 [−0.0049, +0.0042] – |
| VIOL-ANY | RULES | +0.0121 [−0.0050, +0.0308] – | +0.0033 [−0.0184, +0.0267] – | +0.0421 [+0.0246, +0.0612] W | −0.0158 [−0.0371, +0.0058] – |
| VIOL-ANY-OL | RULES | +0.0033 [−0.0050, +0.0137] – | −0.0017 [−0.0104, +0.0075] – | +0.0138 [+0.0062, +0.0208] W | −0.0063 [−0.0150, +0.0017] – |

Units are those of the DAG-ORC reader's rows (`RESULT_V8F_DAG_ORC.md` section 2); shares and agreement rows are fractions. `n` per cell (windows / episodes) for this table:

| row | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| ADE6 | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| speed MAE | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| along-track MAE | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| accel MAE | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| DK-VIOL | 595/70 | 595/70 | 595/70 | 595/70 |
| DK-HW | 713/72 | 709/72 | 707/72 | 712/72 |
| DK-TG | 654/72 | 653/72 | 651/72 | 654/72 |
| heading MAE | 2373/141 | 2373/141 | 2373/141 | 2373/141 |
| yaw-rate MAE | 2366/141 | 2366/141 | 2367/141 | 2367/141 |
| curvature MAE | 2366/141 | 2366/141 | 2367/141 | 2367/141 |
| cross-track MAE | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| L1 lateral-accel MAE | 2366/141 | 2366/141 | 2367/141 | 2367/141 |
| L2 curvature MAE (GT>=5.2) | 2102/135 | 2106/135 | 2105/135 | 2105/135 |
| TAC-LAT | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| TAC-LON | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| S-POS@30 | - | - | - | - |
| S-POS@50 | - | - | - | - |
| S-POS@80 | - | - | - | - |
| M1 crawl share | 2102/135 | 2106/135 | 2105/135 | 2105/135 |
| VIOL-ANY | 2399/141 | 2399/141 | 2399/141 | 2399/141 |
| VIOL-ANY-OL | 2399/141 | 2399/141 | 2399/141 | 2399/141 |

† A rounded bound of the cell is exactly 0.0000. The flag (B / W / –) is the estimator's own reading of the unrounded bound, not of the rounded one: egoc-identity REP s0 DK-VIOL. Cells where the flag disagrees with the rounded interval: none.

**Census of the table above (DERIVED; counted by `gen_result.py` over the 18 computed rows per run):**

| run | rows separated-better | rows separated-worse | rows not separated |
|---|---|---|---|
| CTRL s0 | 0 | 0 | 18 |
| CTRL s1 | 0 | 3 | 15 |
| REP s0 | 8 | 2 | 8 |
| REP s1 | 0 | 0 | 18 |

### The deliberate-regression arm, `egoshufc − identity` (T1-kin; every row; K-REG reads the ADE6 row)

| row | family | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|---|
| ADE6 | PRIMARY | +4.2928 [+3.4386, +5.1945] W | +3.9857 [+3.2389, +4.7695] W | +4.4851 [+3.5747, +5.4322] W | +4.1886 [+3.3996, +5.0549] W |
| speed MAE | LONGITUDINAL | +2.0712 [+1.6872, +2.4721] W | +2.0311 [+1.6652, +2.4187] W | +2.0827 [+1.6733, +2.5123] W | +2.0741 [+1.6866, +2.4824] W |
| along-track MAE | LONGITUDINAL | +4.3269 [+3.4381, +5.2631] W | +4.0125 [+3.2265, +4.8387] W | +4.4729 [+3.5454, +5.4541] W | +4.2073 [+3.3868, +5.0914] W |
| accel MAE | LONGITUDINAL | +1.2346 [+1.0975, +1.3730] W | +1.1159 [+0.9518, +1.2837] W | +1.1507 [+0.9888, +1.3067] W | +1.0862 [+0.8763, +1.3167] W |
| DK-VIOL | LONGITUDINAL | −0.0034 [−0.0243, +0.0167] – | −0.0101 [−0.0227, +0.0017] – | −0.0168 [−0.0345, −0.0033] B | −0.0168 [−0.0351, −0.0017] B |
| DK-HW | LONGITUDINAL | +0.4766 [+0.2099, +0.7401] W | +0.5800 [+0.3296, +0.8660] W | +0.3465 [+0.1010, +0.6448] W | +0.4370 [+0.1721, +0.7214] W |
| DK-TG | LONGITUDINAL | +0.0864 [+0.0337, +0.1493] W | +0.0763 [+0.0381, +0.1196] W | +0.0123 [−0.0346, +0.0597] – | +0.0382 [−0.0085, +0.0863] – |
| heading MAE | LATERAL | +3.5257 [+2.8667, +4.2556] W | +3.3821 [+2.6405, +4.1342] W | +3.5458 [+2.7642, +4.4563] W | +3.3012 [+2.5314, +4.1188] W |
| yaw-rate MAE | LATERAL | +18.1799 [+15.4638, +21.1508] W | +15.0359 [+12.3596, +18.0159] W | +13.3399 [+10.5811, +16.1737] W | +14.8258 [+11.8695, +17.9586] W |
| curvature MAE | LATERAL | +0.1001 [+0.0794, +0.1221] W | +0.0593 [+0.0302, +0.0871] W | +0.0575 [+0.0270, +0.0892] W | +0.0810 [+0.0532, +0.1102] W |
| cross-track MAE | LATERAL | +0.2572 [+0.2052, +0.3118] W | +0.2722 [+0.2106, +0.3320] W | +0.3350 [+0.2792, +0.3987] W | +0.2862 [+0.2303, +0.3428] W |
| L1 lateral-accel MAE | LATERAL | +2.2118 [+1.8114, +2.6552] W | +1.7868 [+1.3834, +2.2128] W | +1.9137 [+1.4959, +2.3396] W | +1.7055 [+1.0593, +2.3708] W |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0597 [+0.0335, +0.0800] W | +0.0552 [+0.0386, +0.0710] W | +0.0304 [−0.0022, +0.0575] – | +0.0658 [+0.0403, +0.0898] W |
| TAC-LAT | TACTICAL | −0.0825 [−0.1059, −0.0583] W | −0.1230 [−0.1484, −0.0971] W | −0.0984 [−0.1197, −0.0780] W | −0.1088 [−0.1318, −0.0846] W |
| TAC-LON | TACTICAL | −0.1263 [−0.1659, −0.0859] W | −0.1288 [−0.1701, −0.0863] W | −0.0479 [−0.0763, −0.0221] W | −0.1371 [−0.1856, −0.0926] W |
| S-POS@30 | STRATEGIC | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| S-POS@50 | STRATEGIC | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| S-POS@80 | STRATEGIC | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| M1 crawl share | crawl | +0.0620 [+0.0513, +0.0732] W | +0.0691 [+0.0573, +0.0811] W | +0.0617 [+0.0494, +0.0738] W | +0.0743 [+0.0628, +0.0865] W |
| VIOL-ANY | RULES | −0.0429 [−0.0813, −0.0067] B | −0.0542 [−0.0942, −0.0167] B | +0.0029 [−0.0363, +0.0371] – | −0.1000 [−0.1454, −0.0592] B |
| VIOL-ANY-OL | RULES | −0.0454 [−0.0708, −0.0217] B | −0.0409 [−0.0655, −0.0183] B | −0.0113 [−0.0329, +0.0088] – | −0.0517 [−0.0763, −0.0284] B |

The shuffled cells hurt: ADE6 is separated-worse at all four runs (K-REG), and so are the speed, along-track, accel, heading, yaw-rate, curvature, cross-track and L1 rows at every run (rows separated-worse per run, DERIVED: CTRL s0 15, CTRL s1 15, REP s0 13, REP s1 14; separated-better: CTRL s0 2, CTRL s1 2, REP s0 1, REP s1 3). The only cells that read **separated-better** under the shuffle are rule-violation shares: VIOL-ANY at CTRL s0, CTRL s1, REP s1; VIOL-ANY-OL at CTRL s0, CTRL s1, REP s1; DK-VIOL at REP s0, REP s1. A lower violation share under a shuffle that costs about 4 m of ADE6 is not an improvement of the plan; the cells are listed so that no separated-better cell is omitted.

## 3. The four families (beside the decision; ADE is one row of four; per family, never pooled)

Per family and per run, two things: the census of the section 2 cells for that family, and the arm's standing against the **P10 floor** (`egoc − P10`, T1-kin; same estimator and windows). STRATEGIC is reported, not paused, for v8F.

### LONGITUDINAL

| census of `egoc − identity` | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| rows separated-better / separated-worse / not separated | 0 / 0 / 6 | 0 / 1 / 5 | 3 / 0 / 3 | 0 / 0 / 6 |

`egoc − P10` (LONGITUDINAL):

| row | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| speed MAE | −0.7833 [−0.9856, −0.6121] B | −0.6850 [−0.8833, −0.5135] B | −0.5449 [−0.7564, −0.3517] B | −0.6785 [−0.8810, −0.5045] B |
| along-track MAE | −0.7091 [−0.8648, −0.5573] B | −0.5994 [−0.7792, −0.4222] B | −0.4277 [−0.6426, −0.2186] B | −0.6126 [−0.7887, −0.4456] B |
| accel MAE | −0.7785 [−1.1532, −0.4568] B | −0.4416 [−0.7129, −0.1980] B | −0.3301 [−0.6892, −0.0099] B | −0.4242 [−0.7481, −0.1227] B |
| DK-VIOL | +0.0017 [−0.0228, +0.0257] – | −0.0067 [−0.0286, +0.0136] – | +0.0101 [−0.0174, +0.0373] – | −0.0067 [−0.0308, +0.0161] – |
| DK-HW | −0.2641 [−0.6818, +0.1348] – | −0.4115 [−0.7749, −0.0871] B | −0.1728 [−0.5489, +0.1538] – | −0.3812 [−0.7886, −0.0297] B |
| DK-TG | −0.0184 [−0.1328, +0.1534] – | −0.0874 [−0.1500, −0.0307] B | −0.0468 [−0.1105, +0.0092] – | −0.0973 [−0.1557, −0.0451] B |

### LATERAL

| census of `egoc − identity` | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| rows separated-better / separated-worse / not separated | 0 / 0 / 6 | 0 / 2 / 4 | 4 / 0 / 2 | 0 / 0 / 6 |

`egoc − P10` (LATERAL):

| row | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| heading MAE | −2.4206 [−3.1566, −1.7506] B | −2.0279 [−2.7477, −1.3526] B | −0.6310 [−1.3850, +0.1513] – | −1.7095 [−2.4988, −0.9451] B |
| yaw-rate MAE | +3.0286 [−0.2135, +6.2640] – | +10.0287 [+6.4331, +13.9961] W | +15.5496 [+11.4219, +20.1882] W | +9.6555 [+6.0362, +13.5666] W |
| curvature MAE | +0.0783 [+0.0549, +0.1043] W | +0.1275 [+0.0871, +0.1752] W | +0.1716 [+0.1294, +0.2209] W | +0.1388 [+0.0900, +0.1964] W |
| cross-track MAE | −0.9563 [−1.1041, −0.8301] B | −0.9708 [−1.1133, −0.8394] B | −0.9841 [−1.1328, −0.8500] B | −0.9582 [−1.1082, −0.8190] B |
| L1 lateral-accel MAE | −2.4032 [−3.8112, −1.2015] B | −1.4846 [−2.4589, −0.5725] B | −1.3428 [−2.5587, −0.2180] B | −1.5896 [−2.7105, −0.6001] B |
| L2 curvature MAE (GT>=5.2) | +0.0425 [+0.0244, +0.0643] W | +0.0633 [+0.0365, +0.0971] W | +0.0967 [+0.0614, +0.1422] W | +0.0507 [+0.0296, +0.0755] W |

### TACTICAL

| census of `egoc − identity` | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| rows separated-better / separated-worse / not separated | 0 / 0 / 2 | 0 / 0 / 2 | 0 / 0 / 2 | 0 / 0 / 2 |

`egoc − P10` (TACTICAL):

| row | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|
| TAC-LAT | +0.1905 [+0.1513, +0.2318] B | +0.2026 [+0.1668, +0.2416] B | +0.1534 [+0.1122, +0.1971] B | +0.1772 [+0.1380, +0.2198] B |
| TAC-LON | +0.1517 [+0.0898, +0.2127] B | +0.1363 [+0.0713, +0.1987] B | +0.0604 [−0.0100, +0.1250] – | +0.1613 [+0.0980, +0.2209] B |

### STRATEGIC

S-POS@30 / 50 / 80 are the strategic head's position error **within a run** (S − CTRV, ticks ≥ 1; READ.json `four_families.STRATEGIC.note`, `n` = *"ticks>=1, unpaired vs CTRV"*, interval machinery INHERITED from the DAG reader; direction `lower` = better, as the reader codes it). The reader's two columns below are the same quantity in two rollouts. **The paired `egoc − identity` S-POS row is NOT COMPUTED** (PREREG section 7): the reader scores S against CTRV inside each rollout and does not pair the two rollouts' S errors window by window, so a difference between the two columns is not a tested contrast. STRATEGIC therefore carries no across-arm statement in this read; it is reported, with this reason, and not passed.

| row (S − CTRV, within run) | mode | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|---|
| S-POS@30 | `identity` | −1.7428 [−2.2737, −1.2861] B | −1.8255 [−2.3433, −1.3880] B | −1.4837 [−1.9587, −1.0595] B | −1.7135 [−2.1974, −1.3049] B |
| S-POS@30 | `egoc` | −1.6754 [−2.1616, −1.2452] B | −1.8415 [−2.3319, −1.4113] B | −1.4009 [−1.8202, −1.0285] B | −1.6566 [−2.1108, −1.2707] B |
| S-POS@30 | `egoc − identity` | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| S-POS@50 | `identity` | −4.5331 [−5.6695, −3.5422] B | −4.6092 [−5.6305, −3.6576] B | −4.1419 [−5.1720, −3.1830] B | −4.8481 [−5.9214, −3.8649] B |
| S-POS@50 | `egoc` | −4.4812 [−5.5861, −3.4515] B | −4.6270 [−5.6365, −3.6953] B | −4.1583 [−5.2174, −3.2088] B | −4.6535 [−5.7059, −3.6993] B |
| S-POS@50 | `egoc − identity` | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |
| S-POS@80 | `identity` | −9.5624 [−11.6843, −7.6124] B | −9.9249 [−11.9750, −7.9686] B | −9.3703 [−11.3441, −7.5010] B | −10.4419 [−12.5718, −8.3877] B |
| S-POS@80 | `egoc` | −9.6310 [−11.7915, −7.6545] B | −9.8671 [−11.8118, −7.9544] B | −9.5742 [−11.6457, −7.6143] B | −10.1261 [−12.2716, −8.1095] B |
| S-POS@80 | `egoc − identity` | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED |

### Gaps, per family: what is not computed, with the reason

| family | row | status | reason |
|---|---|---|---|
| LONGITUDINAL | anti-echo block | **NOT BUILT** | the DAG-ORC reader carries none (READ.json note; PREREG section 7 allows *"NOT BUILT with the reason"*). A work item. |
| LONGITUDINAL | DK-VIOL / DK-HW / DK-TG | computed | on the LEAD windows only (n per cell in section 2); DK is computed in this read, unlike SPEEDSUB's T1-class path |
| TACTICAL | goal-head F1 at T1-kin | **NOT COMPUTABLE here** | the paired tool reads eval dumps, not tick dumps (READ.json note); n not available. TAC-LAT / TAC-LON are the 3-way agreement of the executed track with the human's. A work item. |
| STRATEGIC | S-POS paired across arms | **NOT COMPUTED** | the reader does not pair S errors across rollouts (above) |
| all | stop-window and event strata | not computed | not registered for this read |

### The rule-violation rows (post-Amendment 1; RULES, beside the families)

`VIOL-ANY` is the per-window any-rule violation of the executed `cl` path (rules SPEED / STOP / TURN / LEAD, the T0 eval's own `violations`, against the GT constraints of `eval_context`); `VIOL-ANY-OL` is the tick-0 plan, comparable with SPEEDSUB's T0 finding, and decides nothing.

| row | contrast | CTRL s0 | CTRL s1 | REP s0 | REP s1 |
|---|---|---|---|---|---|
| VIOL-ANY | `egoc − identity` | +0.0121 [−0.0050, +0.0308] – | +0.0033 [−0.0184, +0.0267] – | +0.0421 [+0.0246, +0.0612] W | −0.0158 [−0.0371, +0.0058] – |
| VIOL-ANY-OL | `egoc − identity` | +0.0033 [−0.0050, +0.0137] – | −0.0017 [−0.0104, +0.0075] – | +0.0138 [+0.0062, +0.0208] W | −0.0063 [−0.0150, +0.0017] – |
| VIOL-ANY | `egoc − P10` | −0.4260 [−0.4698, −0.3837] B | −0.4206 [−0.4652, −0.3768] B | −0.4694 [−0.5150, −0.4252] B | −0.3947 [−0.4369, −0.3544] B |
| VIOL-ANY | `identity − P10` | −0.4381 [−0.4821, −0.3953] B | −0.4239 [−0.4706, −0.3792] B | −0.5115 [−0.5588, −0.4673] B | −0.3789 [−0.4212, −0.3371] B |

Mean violation share of the executed path (`identity` → `egoc`): CTRL s0 0.3618 → 0.3739; CTRL s1 0.3760 → 0.3793; REP s0 0.2885 → 0.3306; REP s1 0.4210 → 0.4052. The P10 floor sits at 0.7999 on this measure (Amendment 1's known-value check), so every DAG-OACC run is far below it with or without the substitution.

## 4. REP s0, the only hit: the repair of a slow-T draw

At REP s0 `egoc` cuts ADE6 by **−0.1949 [−0.2619, −0.1313] m**, 3.23 times the bar. Of the 18 computed rows (T1-kin, `egoc − identity`), 8 are separated-better and 2 separated-worse (section 2):

| row | REP s0, `egoc − identity` |
|---|---|
| ADE6 | −0.1949 [−0.2619, −0.1313] B |
| speed MAE | −0.1061 [−0.1495, −0.0664] B |
| along-track MAE | −0.2064 [−0.2785, −0.1393] B |
| accel MAE | −0.0732 [−0.1463, −0.0057] B |
| heading MAE | −0.5464 [−1.0054, −0.1855] B |
| yaw-rate MAE | −3.7761 [−5.9452, −1.9696] B |
| curvature MAE | −0.0463 [−0.0747, −0.0225] B |
| L2 curvature MAE (GT>=5.2) | −0.0347 [−0.0674, −0.0105] B |
| VIOL-ANY | +0.0421 [+0.0246, +0.0612] W |
| VIOL-ANY-OL | +0.0138 [+0.0062, +0.0208] W |

The separated-better rows are the three longitudinal ones (speed, along-track, accel), ADE6, and four lateral rows (heading, yaw-rate, curvature, L2). **The cost is rule violation:** the share of windows on which the executed path breaks a SPEED / STOP / TURN / LEAD rule rises from 0.2885 to 0.3306 (`VIOL-ANY` +0.0421). G-VIOL needs 3 of the 4 runs and fires at 1, so it is a disclosed cost beside the NULL, not a guard firing. M1 crawl share does not separate (−0.0052 [−0.0115, +0.0007] –): the repair is of the speed cells' draw, not of the crawl as such.

### Why REP s0 (INHERITED, T0 diagnostic; `raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md`, sha256 `211601de…`)

REP s0's T predicts every speed cell slow against CTRL s0 on the same 2,399 windows at T0, separated (REP s0 − CTRL s0, m/s, mean [95 % CI]): `lon_v_target_ms` −1.1097 [−1.2814, −0.9345]; `v_a_ms` −0.7315 [−0.8635, −0.5901]; `v_end_ms` −1.0056 [−1.1436, −0.8686]; `v_lo_ms` −1.1700 [−1.3210, −1.0175]; `v_hi_ms` −0.9913 [−1.1383, −0.8487]. So the five cells read 0.73 to 1.17 m/s slow. The plan inherits it: speed bias −0.544 against −0.176 m/s, final along-track bias −3.08 against −0.75 m (T0). REP s1 and A sit with CTRL. The diagnosis is the Master Mind's, written before this read; this read is its T1-kin test. The substitution helps at the draw whose cells are most wrong and at no other run, which is what the diagnosis predicts; it is **one** slow-T draw (n = 1), and the floor rule below shows the cells carry part of REP s0's T1-kin gap, not all of it.

### The floor rule `H-V8F-SPEEDSUB-T1-2` (beside; it decides which floors later arms use)

`g_id(r)` = REP s0^identity − CTRL s0^identity and `g_c(r)` = REP s0^egoc − CTRL s0^egoc, both paired on the same windows (READ.json `floor_rule_H_V8F_SPEEDSUB_T1_2`). **COLLAPSE** needs `g_id(r) > 0` and `g_c(r) ≤ 0.5 · g_id(r)` on **all three** rows; otherwise **NOT COLLAPSE** and the floors stay DAG-ORC's.

| s0 (decisive) row | g_id | g_c | g_c / g_id | closed by `egoc` (DERIVED) | threshold 0.5 · g_id | collapses |
|---|---|---|---|---|---|---|
| ADE6 | +0.4025 | +0.2309 | 0.574 | 42.6 % | +0.2013 | no |
| yaw-rate MAE | +15.8553 | +12.5214 | 0.790 | 21.0 % | +7.9276 | no |
| curvature MAE | +0.1324 | +0.0932 | 0.704 | 29.6 % | +0.0662 | no |

**s0 verdict: NOT COLLAPSE** (READ.json: *"the floors stay as DAG-ORC's"*). The REP s0 − CTRL s0 gap on ADE6 falls from 0.4025 to 0.2309 m (a drop of 42.6 %), yaw-rate MAE and curvature MAE fall by less, and no row reaches half. The arithmetic ties the pieces together (DERIVED, asserted): g_c = g_id + d(REP s0) − d(CTRL s0) = 0.4025 + (−0.1949) − (−0.0233) = 0.2309 (reader: 0.2309), so the part of the gap the substitution removes is REP s0's own repair net of the small CTRL s0 movement.

| s1 (reported) row | g_id | g_c | collapses | note |
|---|---|---|---|---|
| ADE6 | +0.0116 | −0.0141 | yes | g_id above 0 and g_c below half |
| yaw-rate MAE | +0.8903 | −0.3732 | yes | g_id above 0 and g_c below half |
| curvature MAE | −0.0008 | +0.0112 | no | g_id is not above 0: there is no gap to collapse |

**s1 verdict: NOT COLLAPSE.** REP s1 − CTRL s1 is about 0 on identity (the gap rows are +0.0116 / +0.8903 / −0.0008), so the rule cannot collapse what is not there; curvature MAE has `g_id` below 0 and fails the first clause.

The means in the cells reproduce the ADE6 points to 4 dp; for yaw-rate and curvature the difference of means over the two runs' valid windows agrees with the reader's paired point within 1.5 % (the windows differ by one tangent-less step at s0). The verdict per row is the same either way (asserted).

## 5. The healthy draws: CTRL s0, CTRL s1 and REP s1 show no separated ADE6 change

| run | ADE6 `egoc − identity` | multiple of the bar | rows separated-better / separated-worse | note |
|---|---|---|---|---|
| CTRL s0 | −0.0233 [−0.0629, +0.0156] | −0.39 × | 0 / 0 | no row of 18 separates |
| CTRL s1 | −0.0076 [−0.0500, +0.0349] | −0.13 × | 0 / 3 | accel MAE, yaw-rate MAE and L1 lateral-accel MAE separated-worse (one run each: G-WORSE needs 3 runs per row) |
| REP s1 | −0.0333 [−0.0832, +0.0131] | −0.55 × | 0 / 0 | no row of 18 separates |

* **CTRL s0 and REP s1 move nowhere:** 0 of 18 computed rows separate in either direction (0 + 0 and 0 + 0). Their ADE6 points are −0.0233 and −0.0333 m, with intervals that include 0.
* **CTRL s1 is the one healthy draw that moves, and it moves the wrong way on three rows:** accel MAE +0.1598 [+0.0743, +0.2611] W; yaw-rate MAE +1.3443 [+0.2401, +2.5747] W; L1 lateral-accel MAE +0.3627 [+0.1230, +0.6572] W. ADE6 is −0.0076 m, not separated, 0.13 of the bar. Three separated-worse rows at one run do not fire G-WORSE (3 runs per row), but they are the first place a harm would show: the logged-lane read-out raises the same three rows at CTRL s1 (section 6), which is INFERENCE that CTRL s1's O reacts to substituted speed cells on those rows, not a measured mechanism.
* **M1 crawl share moves at no run.** `egoc − identity`: CTRL s0 −0.0007 [−0.0059, +0.0045] –, CTRL s1 +0.0021 [−0.0021, +0.0063] –, REP s0 −0.0052 [−0.0115, +0.0007] –, REP s1 −0.0003 [−0.0049, +0.0042] –. The crawl is not a speed-cell effect that this read-out reaches.

### How much of the crawl gap to P10 `egoc` closes (DERIVED, in DAG-ORC's form: (identity−P10 − egoc−P10) / identity−P10; negative = the gap widens)

The P10 floor is the crawl reference. DAG-ORC's perturbation arm A closed 27.6 % to 79.9 % of the DAG-OACC crawl gap (INHERITED, `RESULT_V8F_DAG_ORC.md` section 4 (b)); the same arithmetic for `egoc`:

| row | CTRL s0 `egoc` | CTRL s0 DAG-ORC A | CTRL s1 `egoc` | CTRL s1 DAG-ORC A | REP s0 `egoc` | REP s1 `egoc` |
|---|---|---|---|---|---|---|
| yaw-rate MAE | 12.8 % | 79.9 % | −15.5 % | 66.9 % | 19.5 % | −0.8 % |
| curvature MAE | 8.3 % | 52.1 % | 5.7 % | 52.0 % | 21.3 % | −3.4 % |
| L2 curvature MAE (GT>=5.2) | 4.1 % | 56.4 % | −19.7 % | 58.8 % | 26.4 % | 13.9 % |
| M1 crawl share | 1.9 % | 46.0 % | −3.5 % | 27.6 % | 5.4 % | 0.5 % |

At the healthy CTRL seeds `egoc` closes at most 12.8 % of the gap on any crawl row (and widens it on 3 of 8 cells), against 27.6 % at the least for DAG-ORC's perturbation. At REP s0, the slow-T draw, it closes 19.5 % to 26.4 % on yaw-rate, curvature and L2 and 5.4 % on M1. The two tables are not one experiment (different arms, different n by one to three windows per cell); the comparison is of sizes, not a contrast.

## 6. Beside, deciding nothing: the logged-lane bridge

Lane: `--limit logged (declared OPEN channel: the logged speed bucket is the bucket of the ego's own realised future max speed, D-SPEEDBAND-EGO-ECHO-1)`. Role: beside; decides nothing; never a capability row. CTRL only: `egol − identity_l`, the banked `mlp_ego` (sha256 `a205609a…`, fitted on the logged bucket) in the lane where T itself reads the logged bucket. Tier: T1-kin. Same estimator.

| row | CTRL s0 `egol − identity_l` | CTRL s1 `egol − identity_l` | CTRL s0 `egoc − identity` (primary lane) | CTRL s1 `egoc − identity` (primary lane) |
|---|---|---|---|---|
| ADE6 | +0.0008 [−0.0374, +0.0375] – | −0.0185 [−0.0637, +0.0260] – | −0.0233 [−0.0629, +0.0156] – | −0.0076 [−0.0500, +0.0349] – |
| speed MAE | +0.0059 [−0.0226, +0.0333] – | −0.0010 [−0.0331, +0.0305] – | −0.0072 [−0.0371, +0.0209] – | +0.0179 [−0.0143, +0.0516] – |
| along-track MAE | +0.0015 [−0.0377, +0.0399] – | −0.0247 [−0.0753, +0.0239] – | −0.0281 [−0.0715, +0.0146] – | −0.0119 [−0.0616, +0.0365] – |
| accel MAE | +0.0312 [−0.0256, +0.0881] – | +0.0980 [+0.0154, +0.1916] W | +0.0387 [−0.0176, +0.0916] – | +0.1598 [+0.0743, +0.2611] W |
| heading MAE | +0.2038 [−0.1045, +0.4759] – | +0.2147 [−0.0596, +0.4457] – | +0.1714 [−0.1698, +0.4806] – | +0.0260 [−0.2861, +0.2939] – |
| yaw-rate MAE | +0.6521 [−1.0941, +2.2367] – | +2.0000 [+1.0303, +3.1066] W | −0.4443 [−2.4850, +1.1922] – | +1.3443 [+0.2401, +2.5747] W |
| curvature MAE | +0.0035 [−0.0097, +0.0152] – | +0.0019 [−0.0201, +0.0188] – | −0.0071 [−0.0220, +0.0056] – | −0.0076 [−0.0337, +0.0110] – |
| L1 lateral-accel MAE | +0.1149 [−0.0623, +0.3086] – | +0.3345 [+0.1144, +0.5729] W | +0.0585 [−0.1062, +0.2145] – | +0.3627 [+0.1230, +0.6572] W |
| L2 curvature MAE (GT>=5.2) | +0.0142 [−0.0045, +0.0319] – | +0.0146 [+0.0038, +0.0277] W | −0.0004 [−0.0175, +0.0145] – | +0.0104 [−0.0023, +0.0240] – |
| M1 crawl share | +0.0068 [+0.0004, +0.0136] W | +0.0065 [+0.0022, +0.0109] W | −0.0007 [−0.0059, +0.0045] – | +0.0021 [−0.0021, +0.0063] – |
| TAC-LAT | +0.0071 [−0.0042, +0.0204] – | −0.0017 [−0.0171, +0.0129] – | +0.0000 [−0.0092, +0.0092] – | +0.0054 [−0.0104, +0.0200] – |
| TAC-LON | +0.0204 [−0.0017, +0.0475] – | +0.0263 [−0.0079, +0.0667] – | +0.0121 [−0.0092, +0.0359] – | −0.0008 [−0.0325, +0.0321] – |

**The read-out with the most future information available moves ADE6 by +0.0008 and −0.0185 m at CTRL s0 and s1: not separated** (intervals −0.0374 to +0.0375 and −0.0637 to +0.0260). Its separated rows are all separated-worse: M1 crawl share at CTRL s0, and accel MAE, yaw-rate MAE, L1 lateral-accel MAE, L2 curvature MAE and M1 at CTRL s1 (the same three rows that `egoc` raises at CTRL s1, plus L2 and M1).

What the bridge says and does not say:

* **It is the read-out SPEEDSUB passed with.** SPEEDSUB (M2b-mixo, T1-class open loop, the same `mlp_ego`, the logged lane) cut ADE6 by −0.1290 m [−0.2107, −0.0548] at s0 and −0.0686 m [−0.1151, −0.0232] at s1 (INHERITED, `RESULT_V8F_SPEEDSUB.md`, PREREG section 0). On DAG-OACC at T1-kin the same read-out in the same lane does not move ADE6.
* **The change has four ingredients at once:** the base (M2b-mixo against DAG-OACC, whose O has seen its own rollouts), the tier (T1-class against T1-kin), the rollout machinery, and the checkpoint seeds. No arm here isolates one. The bridge does not say why SPEEDSUB's PASS did not carry; it says that on the adopted base the open-channel read-out is not a lever either.
* **The open channel (`D-SPEEDBAND-EGO-ECHO-1`).** The logged bucket is the bucket of the ego's own realised future max speed (R² 0.9702 from bin plus `v0`; INHERITED, `Project Steering/GOALS_AND_CLAIMS.md`), admissible only as a declared input and never a training signal. **INFERENCE:** because the best-informed read-out is also flat at CTRL, the causal read-out's flatness is not explained by the future information it lacks.

TRAIN hold-out RMSE of the two read-outs (mean over the five cells, SD units, lower is better; fold 0, 846 clips, 57,479 hold-out rows; decides nothing): the logged-bucket read-out 0.2475, the causal `mlp_ego_c` 0.2793; the logged bucket is the better predictor of T's speed cells on TRAIN, as it should be for an oracle input.

## 7. Next (Rule Zero): a NULL leaves a committed next, a ranking, and named blockers

**Committed next, verbatim from READ.json:** *"next: section 8 (a), the history-aware causal read-out"*. PREREG section 8 (a): the tick-frame design discards the four-frame history that `frame_ok` only masks; the same MLP is fitted on the four history frames of `p10f` plus the causal bucket and nav (an extract over `train_arrays`' window index; ESTIMATED under 1 CPU-h), **under its own amendment with the same table**. Its stated reason: on the pure-dynamics fit the per-tick information is what loses to `T_own`.

**Cost of (a), so the PI can overrule the ranking below (ESTIMATED from measured rollout times):** the extract and fit under 1 CPU-h (PREREG), then twelve rollouts at the measured 138 to 141 s each (27 min 52 s for the twelve decisive rollouts of this read, MEASURED in `chain.log`) plus the reader (454.4 s): at most about 1.6 h of dev-box CPU in all. No GPU and no spend; the PREREG names no PI decision for it. It is blocked on nothing external; it needs only its amendment text.

### Master Mind's ranking (beside the table)

**Label: ESTIMATED / INFERENCE ranking, not a measured outcome.** The table's committed next above is unchanged; this ranks what to do first.

1. **(a) stays registered and is not run now, because arithmetic predicts NULL.** The read-out with the most future information available, the logged oracle bucket (TRAIN hold-out 0.2475 SD units, against 0.2793 for `mlp_ego_c`), did not move healthy CTRL at T1-kin (ADE6 +0.0008 and −0.0185 m, not separated; section 6). A history-aware causal read-out carries less future information than that oracle. **So the speed-cell channel is not what limits healthy DAG-OACC draws.** Its measured use is repairing slow-T training draws (REP s0, section 4).
2. **The largest measured crawl lever is DAG-ORC's perturbation** (`RESULT_V8F_DAG_ORC.md` section 4 (a)): arm A is separated-better on every crawl row at both seeds, yaw-rate MAE −2.7748 and −5.8026, curvature MAE −0.0447 and −0.0700, L2 −0.0254 and −0.0311, M1 −0.0262 and −0.0166 (s0 and s1), closing 27.6 % to 79.9 % of the crawl gap (section 5), while `egoc` at the same CTRL seeds moves M1 by −0.0007 and +0.0021, not separated. **The registered crawl route is LAUNCH-INTENT on box 2, blocked on the PI's go.** INHERITED: `RESULT_V8F_DAG_ORC.md` section 6 registers NEXT-CRAWL → LAUNCH-INTENT as the sequel; the blocker (the PI's go, and box 2 for the training runs) is the Master Mind's statement for this ranking and is UNVERIFIED here.
3. **(a) runs if a later arm shows healthy-draw speed cells binding.** This document's reading of "binding", for the Master Mind to confirm or restate: an arm whose healthy draws (not a slow-T draw) move ADE6 or a crawl row under a speed-cell substitution, separated and at a size above the rig's replicate floor. Until then it is a registered, costed, unrun item.

**Not in the ranking, for the Master Mind to place (unblocked, 0 GPU):** PREREG section 8 (b), the per-cell attribution at T1-kin (each of the five cells substituted alone at CTRL s0 / s1: 10 rollouts, ESTIMATED about 1 h, about 23 min 10 s at the measured mean of 139.3 s per rollout). The PREREG routes (b) from HARMS, not from NULL; it is the instrument that would name the cell behind the three CTRL s1 rows in section 5.

### What is blocked, and on what (Rule Zero: every remaining lever)

| lever | status | blocker, and what unblocks it |
|---|---|---|
| (a) the history-aware causal read-out | registered, costed, **not run** by the ranking | nothing external; its own amendment. Deprioritised by arithmetic (item 1), not blocked. |
| (b) per-cell attribution at T1-kin | unblocked, not routed from NULL | the Master Mind's call to place it. |
| LAUNCH-INTENT (the registered crawl route, NEXT-CRAWL) | registered | **the PI's go** and box 2 for the training runs (the Master Mind's statement, UNVERIFIED here; the route itself is INHERITED from `RESULT_V8F_DAG_ORC.md` section 6). |
| the dominance read (Read A on the substituted CTRL, PREREG section 6.1) | **not reachable** | it is read on PASS only; the outcome is NULL. |
| adoption of `mlp_ego_c` as a `forward_infer` speed sub-head | **not adopted** | the table adopts on PASS only. |

## 8. What this read cannot claim, disclosures, and the run record

### What this read cannot claim

* **It is not closed loop** (T1-kin is a self-action open loop) and it is not a capability claim for the deployed car: the read-out is not built as a sub-head, and no component here is deployed.
* **One read-out fit.** *"Would another fit say this?"* is not answered (PREREG section 10 limit 1).
* **Ticks 1 and 2 read the model's own executed motion**, outside the fit's distribution by construction (limit 2). The ticks-1-and-2 inputs may be why the substitution does less at T1-kin than the per-tick TRAIN accuracy suggests; this read does not separate that from the other explanations.
* **`egoshufc` is a global shuffle** (limit 3), a control with a committed reading, not an estimate of the read-out's information.
* **REP s0 is a slow-T draw** (limit 4): its hit partly measures repair of that draw. **The four runs are two seeds × two replicates of one recipe, not four recipes** (limit 5).
* **NULL is not zero** (section 1): at CTRL s0 and REP s1 the intervals do not exclude a bar-sized cut; the episode-cluster bootstrap cannot say more.
* **Not computed or not built:** S-POS across arms, the anti-echo block, goal-head F1 at T1-kin (section 3). They are work items.
* **No replicate of the read-out and no inference-seed arm.** The planner is deterministic; the rig's run-to-run floor for a lever effect is the REP pair, not an interval.

### Disclosures

1. **Amendment 1 precedes every rollout.** It landed at `4cec09e7` (10:17:23), the first rollout (the smoke) started at 10:21:44, and the chain ran from the tree at that commit (`landed_4cec09e7` in `pins.json`). It added G-VIOL and settled four readings (the tick frame is the window's NOW row; G-NEW per seed; the smoke's control set; C-FIT's float32 tolerance). **No guard fires (G-NEW under either reading),** so the word NULL does not depend on the amendment (section 1 sensitivity).
2. **G-NEW's two readings.** The reader's coded rule fires at least as often as the Amendment 1 reading (set inclusion on this read's data, asserted); the reader's docstring and the build notes (`NOTES.md` section 7 item 3) word the coded rule differently from each other. Both rules give the empty set here, so the difference does not matter for this result.
3. **C-FIT's margin is thin:** 9.412e-06 m/s against the 1e-5 tolerance (94.1 % of it; a float32 summation-order effect, reproducible on the rerun, disclosed in Amendment 1 item 5). The control passed as written.
4. **The fit was reused, not re-run, in the chain.** The chain's fit stage found `fit_c.json` FITTED and reused it (`chain.log`: *"fit: reusing F:/TanitAD-scratch/speedsub_t1/fit (fit_c.json FITTED)"*). The fit was made once at build time, before Amendment 1 and before any rollout; `readout_c.pt` sha256 `799f65ed…` and `fit_c.json` sha256 `9aaa5edd…` were pinned in `pins.json` before the first rollout. There is no selection among read-outs, and C-DISJ (eval ∩ fit clips = ∅) holds, so nothing was tuned on eval-141.
5. **Threads and cost.** OMP / MKL 8 (the default of `run_speedsub_t1.sh`, `${OMP_NUM_THREADS:-8}`; the detached launcher `run_speedsub_t1.cmd` sets no override, and the effective value is not echoed in the log), `CUDA_VISIBLE_DEVICES=-1`, one rollout at a time. Measured rollout times are in the table below: 138 to 141 s per decisive rollout, against the PREREG's ESTIMATE of about 305 s (MEASURED for DAG-OACC in FINDING_3, INHERITED). The cause of the difference is not examined; K-ID reads a max absolute difference of 0.0 against the banked rollouts, so the speed difference changes no number.
6. **Prior.** The PREREG's ESTIMATED prior was that PASS is not the expected outcome (the causal fit loses the oracle part of `limit`). The outcome agrees with the prior; that is not evidence for the prior beyond this one read.
7. **What the re-derivation is and is not.** `gen_result.py` recomputes the decision from the reader's per-run cells (4 dp, with the estimator's separation flags) and checks the JSON against itself and against the reader's flags: it tests the decision logic, the guards, the floor rule and the consistency of `SPEEDSUB_T1_READ.json`. It does **not** recompute the cells from the per-window dumps or re-run the bootstrap (the dumps are a single copy on F:); a defect inside the estimator or the heavy layer would not show here. The reader's heavy layer was exercised on the banked DAG-OACC dumps before the read (38 of 38 rows equal, max absolute difference 0.0; INHERITED, `NOTES.md` section 4).
8. **Single copies.** The per-episode dumps, tick arrays and run trees are on `F:/TanitAD-scratch/speedsub_t1/out/` only (single copy); the reader's JSON, the manifest, the pins and the logs are in the repo. If F: is lost, the tables here stay checkable against READ.json but cannot be recomputed from the dumps.

### Run record (`chain.log`; every stage ran once, in order; no stop, no refusal, no re-run)

| rollout | start | done | wall (s) |
|---|---|---|---|
| SST1_SMOKE_CTRL_s0_identity (smoke) | 10:21:44 | 10:23:23 | 99 |
| SST1_SMOKE_CTRL_s0_egoc (smoke) | 10:23:24 | 10:23:51 | 27 |
| SST1_CTRL_s0_identity (decisive) | 10:23:54 | 10:26:12 | 138 |
| SST1_CTRL_s0_egoc (decisive) | 10:26:14 | 10:28:33 | 139 |
| SST1_CTRL_s0_egoshufc (decisive) | 10:28:35 | 10:30:55 | 140 |
| SST1_CTRL_s1_identity (decisive) | 10:31:01 | 10:33:22 | 141 |
| SST1_CTRL_s1_egoc (decisive) | 10:33:24 | 10:35:43 | 139 |
| SST1_CTRL_s1_egoshufc (decisive) | 10:35:46 | 10:38:05 | 139 |
| SST1_REP_s0_identity (decisive) | 10:38:07 | 10:40:26 | 139 |
| SST1_REP_s0_egoc (decisive) | 10:40:27 | 10:42:47 | 140 |
| SST1_REP_s0_egoshufc (decisive) | 10:42:49 | 10:45:09 | 140 |
| SST1_REP_s1_identity (decisive) | 10:45:11 | 10:47:31 | 139 |
| SST1_REP_s1_egoc (decisive) | 10:47:33 | 10:49:51 | 138 |
| SST1_REP_s1_egoshufc (decisive) | 10:49:53 | 10:52:13 | 140 |
| SST1_CTRL_s0_identity_l (beside (logged lane)) | 10:59:51 | 11:02:10 | 139 |
| SST1_CTRL_s0_egol (beside (logged lane)) | 11:02:12 | 11:04:33 | 141 |
| SST1_CTRL_s1_identity_l (beside (logged lane)) | 11:04:36 | 11:06:55 | 139 |
| SST1_CTRL_s1_egol (beside (logged lane)) | 11:06:57 | 11:09:16 | 139 |

Stage ends (`ZZSST1-STAGE-DONE`): fit 10:21:41, pin 10:21:42, static 10:21:42, smoke 10:23:53, ctrl0 10:30:55, preflight 10:31:00, ctrl1 10:38:05, rep0 10:45:10, rep1 10:52:13, read 10:59:49, beside 11:09:29. The manifest was written at 10:52:13 and the reader ran from there to 10:59:49 (stage wall 7 min 36 s; the reader's own wall 454.4 s).

## 9. Provenance (artifact path, sha256)

Repo paths are under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/` unless they start with `Project Steering/`. **Status** says where the bytes are: *in the repo at the tip* (checked against the live tip by the hold builder), *staged by this hold*, or *not in the repo*.

| role | repo path | sha256 | status |
|---|---|---|---|
| the read (decision of record) | `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json` | `b3b72325e5b8943a90c8b96ecd7c484e1b1be53ab814e4e0aa525c9b530adae0` | in the repo at the tip |
| the beside logged-lane read | `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_BESIDE.json` | `cce47a696766d150a6f6c6d44677bb043ea17cbea895e7571fc009b627de6c2c` | in the repo at the tip |
| pins (checkpoints, read-out, tool, wrapper, instrument, reader) | `raw/2026-10-28-speedsub-t1-read/pins.json` | `59ea9a855630a7ce9bf29188be487777788fd1155bfc1cf972e21711e70dd479` | in the repo at the tip |
| runs manifest | `raw/2026-10-28-speedsub-t1-read/runs_manifest.json` | `9f03cb3b21e46383f1996aabae12f865c601d3ac18e6eafd532439f60c30fb94` | in the repo at the tip |
| chain log | `raw/2026-10-28-speedsub-t1-read/chain.log` | `a8a8f8b11554e3d4efd1fdee521861478e4d93c16b2a8d177ba074af9380831f` | in the repo at the tip |
| chain console output | `raw/2026-10-28-speedsub-t1-read/chain_speedsub_t1.out` | `7a7fe043ca09f5d8a18d0dcbfc1d3b96a9a72268f3639f11368b914afce6e0ac` | in the repo at the tip |
| CTRL s0 preflight record | `raw/2026-10-28-speedsub-t1-read/preflight_CTRL_s0.json` | `116c9e5509b55a12ea855f419ef1ab25e8bdd41b17291df85691f308d6a7dd7d` | in the repo at the tip |
| smoke record | `raw/2026-10-28-speedsub-t1-read/smoke_check_CTRL_s0.json` | `d2b936e1ebe03b60bad2afa48d973940df7355517744e94c5f09a64cfb5a36aa` | in the repo at the tip |
| the pre-registered reader | `raw/2026-10-28-speedsub-t1-build/read/speedsub_t1_read.py` | `9d3c27926ec31768684dd52dc3bae586aa52984eefe8eb2763fb69234af0b98f` | in the repo at the tip |
| the fit and its controls | `raw/2026-10-28-speedsub-t1-build/evidence/fit_c.json` | `9aaa5edde167b767d1bbf586f2e2909f1f21b40fc17e53047123101d60fd499a` | in the repo at the tip |
| build notes | `raw/2026-10-28-speedsub-t1-build/NOTES.md` | `e90ebbaf381edf8f7c5112f017cab1ed9381b01bc627dd8312bc51475fe6cb5b` | in the repo at the tip |
| chain script | `raw/2026-10-28-speedsub-t1-build/run_speedsub_t1.sh` | `026849728f9e790f9b86f2f2b3062cf444cf659442afef9b1ca007d7adafaca3` | in the repo at the tip |
| PREREG with Amendment 1 | `Project Steering/PREREG_V8F_SPEEDSUB_T1.md` | `69187967e82bf64dac85bf9ae81810b6f8aac7161cd2558d719d45cdf61fd861` | in the repo at the tip |
| REP s0 speed-cell diagnostic (INHERITED) | `raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md` | `211601de6592fee86b1e4f4ab883d16acc12bdba47028b7bf008ac7eb49a1d64` | in the repo at the tip |
| DAG-ORC RESULT (INHERITED numbers, section 4) | `RESULT_V8F_DAG_ORC.md` | `5ce374218288bf2a02edfd7965239e84c21dc8159ba23ac5e7e3aa8206059a6c` | in the repo at the tip |
| SPEEDSUB RESULT (INHERITED numbers) | `RESULT_V8F_SPEEDSUB.md` | `4bdc87a27a13bb34ea161cf6ea05e43edca2b4960fc7c2fe9e645a09c8965f7c` | in the repo at the tip |
| this document's generator, coverage check, audit, register-row text, mutation check and its output, builder | `raw/2026-10-28-speedsub-t1-result/audit/` and `raw/2026-10-28-speedsub-t1-result/landing/` (hashes in `raw/2026-10-28-speedsub-t1-result/PACKAGE_SHA256SUMS.txt`) | see that file | staged by this hold |

**Not in the repo (single copy on F: scratch):** `F:/TanitAD-scratch/speedsub_t1/out/` (the 18 rollouts' dumps, tick arrays and records, `runs_manifest.json` lists every path), `F:/TanitAD-scratch/speedsub_t1/fit/readout_c.pt` (sha256 `799f65ed1c597063bf98c6dd53138f63d219a8ecb90d5b9fcae2564d459e3fcb`), the four DAG-OACC / DAG-ORC checkpoints (sha256 in `pins.json`: CTRL s0 `67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f`, CTRL s1 `90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333`, REP s0 `b03fd6efebec285f747673fa90d093cb9202a649d560ba8f5c016475f1fb4bca`, REP s1 `77853e564f52ff029cfc241042b733d75dba8191a99a7be0fc59778d511aeca2`), and the banked `mlp_ego` (`a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d`).

The in-repo instrument tests are INHERITED from the build (`NOTES.md`): the wrapper 33 tests and 29 of 29 mutants, the pin step 52 tests and 25 of 25 mutants, the reader 73 tests and 110 of 110 mutants. They were not re-run for this document.
