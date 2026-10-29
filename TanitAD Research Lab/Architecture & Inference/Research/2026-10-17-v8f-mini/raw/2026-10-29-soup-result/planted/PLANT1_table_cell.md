# RESULT_V8F_SOUP: a uniform same-init weight average ("soup") of two DAG-OACC-recipe O fine-tunes, read at T1-kin. Outcome **HARMS** (not adopted)

Claim row `H-V8F-SOUP-1`. Pre-registration `Project Steering/PREREG_V8F_SOUP.md`, promoted at `e03b348` (2026-10-29 02:00:06; sha256 `830d9c91…`, status line and section 0 item 5 changed at promotion, nothing else). Read 2026-10-29 by the pre-registered reader `soup_read.py` (sha256 `de39aefe…`; it imports the LAUNCH-INTENT reader, sha256 `fb3d2d7d…`, unchanged): all **368 reader controls PASS** (368 of 368), reader wall 634.9 s, rc 0. Output `raw/2026-10-29-soup-read/out/read/SOUP_READ.json` (sha256 `dca10c07…`), banked at `0c7b598` (2026-10-29 03:10:30).

* **The word, as registered.** **HARMS.** The uniform two-draw soup is not adopted. The guard set H fails at s0, in both pairs: R-ADE fails in the main pair (ADE6, in m, SOUP_0 − DAG-OACC_0: +0.0770 [+0.0213, +0.1348] W) and in the replicate pair (ADE6, in m, SOUP'_0 − A_0: +0.0959 [+0.0321, +0.1624] W), and R-TLAT fails in the main pair (reader boolean). At s1 every guard holds. The registered rule is that H must hold at both seeds, so the s1 pass does not change the word (PREREG section 10). Nothing below softens that.
* **No crawl row moved by its bar, and no crawl row reaches P10.** R-MOVE is false in all four pairs and R-CRAWL is false in all four pairs. Of the 16 crawl cells X − Y, 0 are separated-better and 9 are separated-worse (8 of them at s0, where the soup carries part of the slow member's crawl). Read A reads **MARGIN, NOT DOMINANT** for SOUP, SOUP' and SOUP4 at both seeds: D1 (ADE6 better than P10 by at least 2F) and D3 (each family has a separated-better row) hold, D2 fails on the same four crawl rows.
* **Tier.** T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan). Never 'closed loop'. The R-T0 points and the slow-draw cells are at T0 (teacher-forced own-code, eval-141, ages cycle, dev-box CPU on both arms of a pair). Never 'driving performance'.
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, same windows and matched cells (JSON `estimator`). Intervals are 95 % (INHERITED: the convention of `RESULT_V8F_LAUNCH_INTENT.md`; the JSON stores the bounds only). It answers *"would another draw of EPISODES say this?"* and nothing else. *"Would another TRAINING RUN say this?"* is answered only by the replicate pair at each seed and the second seed (`H-ESTIM-SEED-1`). The planner is deterministic and a soup is a fixed set of weights, so no inference-seed question arises.
* **Arms** (zero training; every member banked from the LAUNCH-INTENT and DAG-OACC runs). SOUP_s = DAG-OACC_s + REP_s (donor DAG-OACC_s); SOUP'_s = A_s + A_rep,s (donor A_s; both members carry `--launch-intent-w` 1.58926); SOUP4_s = all four, beside. Weights are averaged in float64 and rounded once to float32; the soups share the init M2b_s with their members. Control of the main pair: DAG-OACC_s. Control of the replicate pair: A_s (PREREG section 0 item 1).
* **Coverage.** The four metric families at T1-kin are in section 4, per seed, per pair, never pooled. **The T0 four families are not in this read** (section 4.7). The R-TLAT / R-TLON macro-F1 values, the T1-kin goal-head F1 and head accuracy, and the anti-echo block are not computed by the rig (sections 2.3 and 4.7). They are work items, and the outcome above does not use them.

**Summary** (each value is in the section named; evidence class MEASURED (ours) unless stated).

* **HARMS (section 1).** Controls pass. At s0: R-ADE fails in both pairs, R-TLAT fails in the main pair, and the adoption row R-KEEP fails in both pairs (point +0.0770 and +0.0959 against +0.0604). At s1: every guard, R-KEEP and R-NOBASE hold. R-MOVE and R-CRAWL are false everywhere.
* **The soup gain is real and there is no barrier (section 3).** SG on ADE6 (the soup minus the mean of its members' errors, m): SOUP -0.1242 [-0.1517, -0.0977] B (s0), -0.0163 [-0.0379, +0.0050] – (s1); SOUP' -0.1416 [-0.1693, -0.1152] B (s0), -0.0488 [-0.0678, -0.0301] B (s1); SOUP4 -0.1301 [-0.1568, -0.1050] B (s0), -0.0216 [-0.0392, -0.0044] B (s1). Five of six cells are separated-better; SOUP at s1 is not separated. Routing: **FLAT**, so the committed next is SOUP-K.
* **Mechanism (section 6, reading, decides nothing).** At s0 the slow draw REP_0 is +0.4025 m worse than DAG-OACC_0 on ADE6. The soup recovers most of it (SOUP_0 − REP_0 -0.3255 [-0.4077, -0.2478] B) and still lands worse than the good draw (SOUP_0 − DAG-OACC_0 +0.0770 [+0.0213, +0.1348] W).
* **Next (section 7).** Committed, as registered: SOUP-K (FLAT route; 4 box-2 runs; its own PI go). Rule Zero: the crawl diagnostics say the crawl of DAG-OACC s0 sits at the outcome-blind floor (on the point), S_v is the frozen prior's, the M2b checkpoints measured are FIT-LIMITED on TRAIN, the images add little go/stay information beyond ego (none on green-light or lead-departure clips), and the trunk's go/stay edge is the declared speed-ceiling and nav oracles; averaging can shrink the draw-specific part of the crawl and cannot move the part every member shares, so the lever the evidence points to is perception. A finer-token DINOv3 probe is PROPOSED, not registered.

## 1. The outcome, as the table decides it

Evidence class MEASURED (ours). Tier and estimator as in the header: T1-kin unless a cell says T0; paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Source: `pairs.<seed>.<main|rep>` and `OUTCOME` in the JSON. X is the soup of the pair (SOUP in the main pair, SOUP' in the replicate pair); Y is the pair's control (DAG-OACC, A). The R-TLAT and R-TLON cells are the reader's guard booleans; the values behind them are not in the JSON (section 2.3).

| predicate (PREREG section 10) | s0 main (SOUP − DAG-OACC) | s0 replicate (SOUP' − A) | **s0, both pairs** | s1 main (SOUP − DAG-OACC) | s1 replicate (SOUP' − A) | **s1, both pairs** |
|---|---|---|---|---|---|---|
| controls (reader, 368) | PASS | PASS | **PASS** | PASS | PASS | **PASS** |
| R-TLAT (T head lat macro-F1, ticks >= 1, X − Y not separated-worse) | FAILS | held | **FAILS** | held | held | **held** |
| R-TLON (T head lon macro-F1, ticks >= 1, X − Y not separated-worse) | held | held | **held** | held | held | **held** |
| R-ADE (T1-kin ADE6, X − Y, not separated-worse) | FAILS: +0.0770 W | FAILS: +0.0959 W | **FAILS** | held: -0.0105 – | held: -0.0842 B | **held** |
| R-T0 (T0 own-code ADE6, X − Y, point <= +0.0604 m) | held: +0.0552 | held: +0.0585 | **held** | held: -0.0527 | held: -0.0443 | **held** |
| R-NEW (no row of D fires) | held: 0 of 15 rows | held: 0 of 15 rows | **held** | held: 0 of 15 rows | held: 0 of 15 rows | **held** |
| R-MOVE (every crawl row clears its bar) | false: 0 of 4 rows clear | false: 0 of 4 rows clear | **false** | false: 0 of 4 rows clear | false: 0 of 4 rows clear | **false** |
| R-CRAWL (every crawl row of X − P10 not separated-worse) | false: 0 of 4 rows | false: 0 of 4 rows | **false** | false: 0 of 4 rows | false: 0 of 4 rows | **false** |
| R-KEEP (T1-kin ADE6, X − Y, point <= +0.0604 m) | false: +0.0770 | false: +0.0959 | **false** | true: -0.0105 | true: -0.0842 | **true** |
| R-NOBASE | true: 0 of 15 rows violate | true: 0 of 15 rows violate | **true** | true: 0 of 15 rows violate | true: 0 of 15 rows violate | **true** |

Per-seed booleans (`OUTCOME.per_seed`; PREREG section 10). Each holds at a seed only when it holds in both pairs of that seed.

| seed | H (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW, both pairs) | MV (R-MOVE, both pairs) | AD (R-KEEP and R-NOBASE, both pairs) | DM (Read A D1, D2, D3 for SOUP and SOUP') |
|---|---|---|---|---|
| s0 | false | false | false | false |
| s1 | true | false | true | false |

**HARMS.** The table's condition, verbatim from PREREG section 10: *"controls pass; H fails at ≥ 1 seed"*. H fails at s0 (R-ADE in both pairs, R-TLAT in the main pair), so the word is HARMS. REFUSED is decided first and HARMS second; the controls pass. Committed next, verbatim from the JSON: *"the uniform two-draw soup is not adopted. NEXT-SOUP (§11)"*, with `next_soup` = SOUP-K and `route` = FLAT.

⚠ **Reported as written, beside the word.** At s1 alone every guard and both adoption rows hold (H and AD true), and the lever still moves no crawl row (MV false) and does not reach P10 on the crawl rows (DM false). The word is not taken seed by seed: PREREG section 10 requires H at both seeds. The adoption row R-KEEP also fails at s0 in both pairs (section 2.2), so the soup would not have replaced DAG-OACC at s0 even without the guards.

The word is the reader's output (`matching_rows_in_table_order`: one row, HARMS). `check_result.py` recomputes R-ADE, R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-T0 and R-KEEP per pair from the JSON's cells, floors and points, derives H, MV, AD, DM and the word with the PREREG section 10 precedence, and finds every flag equal to the reader's. R-TLAT and R-TLON are taken from the reader (their statistic is not in the JSON).

## 2. The decisive rows

Tier T1-kin unless a heading says T0. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Evidence class MEASURED (ours): every cell is a copy of `tables.<contrast>.<seed>.<row>` in the JSON. Cells read point [95 % CI] and a flag: B = separated-better, W = separated-worse, – = not separated. Bold = separated. P10 is the kinematic floor. The `n` of a cell is windows / episodes.

### 2.1 The floors and the bar (`floors`)

`bar_r = 2 × max(F_r, RF0_r)`, reused unchanged from `PREREG_V8F_LAUNCH_INTENT` section 6 (PREREG_V8F_SOUP section 7), one value per row, applied at both seeds and in both pairs, inclusive. F is 0.1207 m on ADE6 and 0 on every other row (PREREG). RF0 is the frozen same-seed cross-build floor (`P0_RF0.json`, sha256 `1608d70b…`). RF is the larger of two REP − DAG-OACC draws, reported beside every row; it is not a bar term. Here RF is also the main soup's own member spread.

| row | better | F | RF0 | **bar** = 2 × max(F, RF0) | RF (beside, not a bar term) | RF / bar |
|---|---|---|---|---|---|---|
| ADE6 | lower | 0.1207 | 0.0138 | **0.2414** | 0.4025 | 1.67 |
| speed MAE | lower | 0.0000 | 0.0796 | **0.1592** | 0.3373 | 2.12 |
| along-track MAE | lower | 0.0000 | 0.0201 | **0.0402** | 0.4596 | 11.43 |
| accel MAE | lower | 0.0000 | 0.1323 | **0.2646** | 0.5603 | 2.12 |
| DK-VIOL | lower | 0.0000 | 0.0050 | **0.0100** | 0.0017 | 0.17 |
| DK-HW | lower | 0.0000 | 0.1308 | **0.2616** | 0.2540 | 0.97 |
| DK-TG | lower | 0.0000 | 0.0041 | **0.0082** | 0.1042 | 12.71 |
| heading MAE | lower | 0.0000 | 0.3801 | **0.7602** | 2.5074 | 3.30 |
| yaw-rate MAE | lower | 0.0000 | 0.5535 | **1.1070** | 15.8553 | 14.32 |
| curvature MAE | lower | 0.0000 | 0.0263 | **0.0526** | 0.1324 | 2.52 |
| cross-track MAE | lower | 0.0000 | 0.0320 | **0.0640** | 0.0224 | 0.35 |
| L1 lateral-accel MAE | lower | 0.0000 | 0.3985 | **0.7970** | 1.2242 | 1.54 |
| L2 curvature MAE (GT>=5.2) | lower | 0.0000 | 0.0068 | **0.0136** | 0.0870 | 6.40 |
| TAC-LAT | higher | 0.0000 | 0.0183 | **0.0366** | 0.0421 | 1.15 |
| TAC-LON | higher | 0.0000 | 0.0125 | **0.0250** | 0.0950 | 3.80 |
| S-POS@30 | lower | 0.0000 | 0.1877 | **0.3754** | 0.2592 | 0.69 |
| S-POS@50 | lower | 0.0000 | 0.4449 | **0.8898** | 0.3912 | 0.44 |
| S-POS@80 | lower | 0.0000 | 1.0082 | **2.0164** | 0.5170 | 0.26 |
| M1 crawl share | lower | 0.0000 | 0.0048 | **0.0096** | 0.0388 | 4.04 |

### 2.2 ADE6: the guard R-ADE and the adoption row R-KEEP (T1-kin)

| seed | pair | row | X − Y | mean X | mean Y | X − P10 | Y − P10 | n (windows / episodes) |
|---|---|---|---|---|---|---|---|---|
| s0 | main (SOUP − DAG-OACC) | ADE6 | **+0.0770 [+0.0231, +0.1348] W** | 2.3191 | 2.2421 | **-1.1242 [-1.3415, -0.9136] B** | **-1.2012 [-1.3945, -1.0186] B** | 2399/141 |
| s0 | replicate (SOUP' − A) | ADE6 | **+0.0959 [+0.0321, +0.1624] W** | 2.3086 | 2.2128 | **-1.1346 [-1.3472, -0.9266] B** | **-1.2305 [-1.4302, -1.0448] B** | 2399/141 |
| s1 | main (SOUP − DAG-OACC) | ADE6 | -0.0105 [-0.0477, +0.0263] – | 2.3096 | 2.3201 | **-1.1337 [-1.3387, -0.9306] B** | **-1.1231 [-1.3301, -0.9159] B** | 2399/141 |
| s1 | replicate (SOUP' − A) | ADE6 | **-0.0842 [-0.1298, -0.0400] B** | 2.2842 | 2.3684 | **-1.1591 [-1.3614, -0.9603] B** | **-1.0749 [-1.2743, -0.8735] B** | 2399/141 |

R-ADE needs X − Y not separated-worse. It fails in both pairs at s0 and holds in both pairs at s1; at s1 the replicate pair is separated-better (SOUP'_1 − A_1). R-KEEP reads the same contrast's point against +0.0604 m and fails at s0 in both pairs. R-ADE and R-KEEP read one contrast, as the PREREG discloses.

### 2.3 R-TLAT, R-TLON, R-T0 and R-KEEP

R-T0 is at tier T0 (a point, no interval in the JSON; dev-box CPU on both arms of the pair, K-T0DEV). R-KEEP is T1-kin. The TAC-LAT and TAC-LON cells are the rig's **plan-derived agreements** (JSON `four_families` note), shown beside the guard booleans; they are not the statistic the guards test.

| seed | pair | TAC-LAT X − Y (plan-derived agreement) | R-TLAT (reader) | TAC-LON X − Y (plan-derived agreement) | R-TLON (reader) | T0 own-code ADE6, X − Y (point) | R-T0 (<= +0.0604) | T1-kin ADE6, X − Y (point) | R-KEEP (<= +0.0604) |
|---|---|---|---|---|---|---|---|---|---|
| s0 | main (SOUP − DAG-OACC) | **-0.0125 [-0.0250, -0.0004] W** | FAILS | -0.0288 [-0.0588, +0.0017] – | held | +0.0552 | held | +0.0770 | false |
| s0 | replicate (SOUP' − A) | -0.0104 [-0.0221, +0.0017] – | held | **-0.0475 [-0.0775, -0.0179] W** | held | +0.0585 | held | +0.0959 | false |
| s1 | main (SOUP − DAG-OACC) | +0.0058 [-0.0088, +0.0225] – | held | +0.0058 [-0.0150, +0.0283] – | held | -0.0527 | held | -0.0105 | true |
| s1 | replicate (SOUP' − A) | +0.0000 [-0.0175, +0.0179] – | held | +0.0021 [-0.0167, +0.0233] – | held | -0.0443 | held | -0.0842 | true |

R-TLAT and R-TLON are booleans in the JSON; the T-head macro-F1 values behind them are not carried (UNVERIFIED as numbers). On TAC-LAT the plan-derived cell and the guard agree in all four pairs: in the s0 main pair both fail (TAC-LAT, SOUP_0 vs DAG-OACC_0: -0.0125 [-0.0250, -0.0004] W). On TAC-LON they do not: in the s0 replicate pair the cell is separated-worse (-0.0475 [-0.0775, -0.0179] W) while the reader's R-TLON holds. The two statistics differ, so a TAC cell is never read as the guard. The word does not depend on it: H already fails at s0 through R-ADE in both pairs. The register row `H-V8F-SOUP-1` labels the -0.0125 cell "TAC-LAT macro-F1"; in the JSON it is the TAC-LAT row (disclosure 6).

### 2.4 R-MOVE: the crawl rows, X − Y against the bar (T1-kin)

R-MOVE needs, in each pair, every crawl row separated-better with an improvement of at least the bar. Improvement is −delta (all four rows are lower-is-better). The crawl set is yaw-rate MAE, curvature MAE, L2 curvature MAE (GT >= 5.2) and M1 crawl share (`pairs.*.crawl_set`).

| seed | pair | crawl row | X − Y | improvement | bar | improvement / bar | clears (R-MOVE row) |
|---|---|---|---|---|---|---|---|
| s0 | main (SOUP − DAG-OACC) | yaw-rate MAE | **+4.8651 [+3.1469, +6.8062] W** | -4.8651 | 1.1070 | -439.5 % | false |
| s0 | main (SOUP − DAG-OACC) | curvature MAE | **+0.0417 [+0.0217, +0.0641] W** | -0.0417 | 0.0526 | -79.3 % | false |
| s0 | main (SOUP − DAG-OACC) | L2 curvature MAE (GT>=5.2) | **+0.0418 [+0.0172, +0.0713] W** | -0.0418 | 0.0136 | -307.4 % | false |
| s0 | main (SOUP − DAG-OACC) | M1 crawl share | **+0.0174 [+0.0109, +0.0249] W** | -0.0174 | 0.0096 | -181.2 % | false |
| s0 | replicate (SOUP' − A) | yaw-rate MAE | **+4.8035 [+2.7980, +6.9995] W** | -4.8035 | 1.1070 | -433.9 % | false |
| s0 | replicate (SOUP' − A) | curvature MAE | **+0.0290 [+0.0081, +0.0499] W** | -0.0290 | 0.0526 | -55.1 % | false |
| s0 | replicate (SOUP' − A) | L2 curvature MAE (GT>=5.2) | **+0.0306 [+0.0126, +0.0532] W** | -0.0306 | 0.0136 | -225.0 % | false |
| s0 | replicate (SOUP' − A) | M1 crawl share | **+0.0171 [+0.0103, +0.0245] W** | -0.0171 | 0.0096 | -178.1 % | false |
| s1 | main (SOUP − DAG-OACC) | yaw-rate MAE | +0.0805 [-1.4927, +1.7032] – | -0.0805 | 1.1070 | -7.3 % | false |
| s1 | main (SOUP − DAG-OACC) | curvature MAE | -0.0115 [-0.0320, +0.0069] – | +0.0115 | 0.0526 | 21.9 % | false |
| s1 | main (SOUP − DAG-OACC) | L2 curvature MAE (GT>=5.2) | +0.0070 [-0.0097, +0.0261] – | -0.0070 | 0.0136 | -51.5 % | false |
| s1 | main (SOUP − DAG-OACC) | M1 crawl share | +0.0017 [-0.0024, +0.0058] – | -0.0017 | 0.0096 | -17.7 % | false |
| s1 | replicate (SOUP' − A) | yaw-rate MAE | -0.2182 [-1.8500, +1.3532] – | +0.2182 | 1.1070 | 19.7 % | false |
| s1 | replicate (SOUP' − A) | curvature MAE | **+0.0345 [+0.0130, +0.0616] W** | -0.0345 | 0.0526 | -65.6 % | false |
| s1 | replicate (SOUP' − A) | L2 curvature MAE (GT>=5.2) | +0.0054 [-0.0067, +0.0209] – | -0.0054 | 0.0136 | -39.7 % | false |
| s1 | replicate (SOUP' − A) | M1 crawl share | +0.0018 [-0.0031, +0.0066] – | -0.0018 | 0.0096 | -18.8 % | false |

0 of the 16 cells are separated-better and 0 clear their bar. No pair clears all four rows. 9 cells are separated-worse (8 at s0, 1 at s1): at s0 the soup is worse than its control on every crawl row in both pairs, the soup carrying part of the slow member's crawl (section 6, Fact 2b).

### 2.5 R-CRAWL: the crawl rows, X − P10 (T1-kin)

The share closed is (Y − P10 − (X − P10)) / (Y − P10) from the point deltas; a negative share means the gap to P10 widens. MEASURED (arithmetic on the JSON).

| seed | pair | crawl row | X − P10 | Y − P10 | X − P10 not separated-worse (R-CRAWL row) | share of the Y − P10 gap closed |
|---|---|---|---|---|---|---|
| s0 | main (SOUP − DAG-OACC) | yaw-rate MAE | **+8.3376 [+4.8170, +12.1092] W** | **+3.4726 [+0.6601, +6.3308] W** | false | -140.1 % |
| s0 | main (SOUP − DAG-OACC) | curvature MAE | **+0.1273 [+0.0878, +0.1730] W** | **+0.0854 [+0.0580, +0.1175] W** | false | -49.1 % |
| s0 | main (SOUP − DAG-OACC) | L2 curvature MAE (GT>=5.2) | **+0.0857 [+0.0458, +0.1362] W** | **+0.0443 [+0.0232, +0.0744] W** | false | -93.5 % |
| s0 | main (SOUP − DAG-OACC) | M1 crawl share | **+0.0735 [+0.0507, +0.0992] W** | **+0.0568 [+0.0378, +0.0798] W** | false | -29.4 % |
| s0 | replicate (SOUP' − A) | yaw-rate MAE | **+7.5465 [+3.7165, +11.5338] W** | **+2.7420 [+0.3388, +5.2598] W** | false | -175.2 % |
| s0 | replicate (SOUP' − A) | curvature MAE | **+0.1138 [+0.0802, +0.1506] W** | **+0.0847 [+0.0549, +0.1200] W** | false | -34.4 % |
| s0 | replicate (SOUP' − A) | L2 curvature MAE (GT>=5.2) | **+0.0681 [+0.0370, +0.1088] W** | **+0.0375 [+0.0201, +0.0588] W** | false | -81.6 % |
| s0 | replicate (SOUP' − A) | M1 crawl share | **+0.0735 [+0.0500, +0.0996] W** | **+0.0564 [+0.0374, +0.0788] W** | false | -30.3 % |
| s1 | main (SOUP − DAG-OACC) | yaw-rate MAE | **+8.8067 [+5.1990, +12.9014] W** | **+8.6844 [+5.1690, +12.5740] W** | false | -1.4 % |
| s1 | main (SOUP − DAG-OACC) | curvature MAE | **+0.1236 [+0.0808, +0.1721] W** | **+0.1352 [+0.0889, +0.1915] W** | false | 8.6 % |
| s1 | main (SOUP − DAG-OACC) | L2 curvature MAE (GT>=5.2) | **+0.0590 [+0.0332, +0.0919] W** | **+0.0529 [+0.0281, +0.0847] W** | false | -11.5 % |
| s1 | main (SOUP − DAG-OACC) | M1 crawl share | **+0.0616 [+0.0404, +0.0866] W** | **+0.0604 [+0.0391, +0.0864] W** | false | -2.0 % |
| s1 | replicate (SOUP' − A) | yaw-rate MAE | **+8.7270 [+5.2159, +12.6264] W** | **+8.9439 [+5.3604, +12.9899] W** | false | 2.4 % |
| s1 | replicate (SOUP' − A) | curvature MAE | **+0.1310 [+0.0859, +0.1853] W** | **+0.0964 [+0.0650, +0.1321] W** | false | -35.9 % |
| s1 | replicate (SOUP' − A) | L2 curvature MAE (GT>=5.2) | **+0.0514 [+0.0288, +0.0798] W** | **+0.0462 [+0.0260, +0.0716] W** | false | -11.3 % |
| s1 | replicate (SOUP' − A) | M1 crawl share | **+0.0633 [+0.0424, +0.0877] W** | **+0.0620 [+0.0415, +0.0870] W** | false | -2.1 % |

All 16 X − P10 crawl cells are separated-worse. So are the Y − P10 cells in every pair: that is how the crawl set was fixed.

### 2.6 R-NEW and R-NOBASE (T1-kin; domain D is every row not separated-worse for Y − P10)

R-NEW fires on a row of D when (1) X − P10 is separated-worse, (2) Y − P10 is not (the row is in D), and (3) X − Y is separated-worse with a worsening of at least the bar. R-NOBASE reads (1) and (3) over D. Worsening is delta on lower-is-better rows and −delta on higher-is-better rows.

| seed | pair | control Y | rows in D | X − Y separated-worse (rows of D) | of those, worsening >= bar | of those, X − P10 separated-worse (R-NEW fires) |
|---|---|---|---|---|---|---|
| s0 | main (SOUP − DAG-OACC) | DAG-OACC − P10 not separated-worse on D | 15 | ADE6, speed MAE, along-track MAE, accel MAE, DK-TG, heading MAE, L1 lateral-accel MAE, TAC-LAT, S-POS@30 | along-track MAE, DK-TG, heading MAE | none |
| s0 | replicate (SOUP' − A) | A − P10 not separated-worse on D | 15 | ADE6, speed MAE, along-track MAE, accel MAE, heading MAE, L1 lateral-accel MAE, TAC-LON | along-track MAE, heading MAE, TAC-LON | none |
| s1 | main (SOUP − DAG-OACC) | DAG-OACC − P10 not separated-worse on D | 15 | speed MAE, accel MAE, L1 lateral-accel MAE | accel MAE | none |
| s1 | replicate (SOUP' − A) | A − P10 not separated-worse on D | 15 | none | none | none |

**R-NEW and R-NOBASE hold at every seed and pair because condition (1) fails, not because condition (3) does.** Rows of D that meet condition (3): s0 main, along-track MAE, DK-TG, heading MAE; s0 replicate, along-track MAE, heading MAE, TAC-LON; s1 main, accel MAE; s1 replicate, none. None of them has X − P10 separated-worse. The registered predicate is applied as written.

### 2.7 Read A: X − P10 (`PREREG_V8F_DAG_REREAD` section 2 and 5; T1-kin)

D1: ADE6 separated-better with point <= −2F = −0.2415 m (PREREG_V8F_SOUP section 7). D2: no row of the 19 separated-worse. D3: each family has a separated-better row. The label is the reader's `read_a.<soup>.outcome`; its rule is `PREREG_V8F_DAG_REREAD` section 5 (*MARGIN, NOT DOMINANT*: D1 at both seeds, D2 fails at >= 1 seed; if the failing rows are crawl rows, that PREREG names DAgger iteration 2, which this read does not route).

| soup | seed | ADE6 X − P10 | D1 | D2 (rows separated-worse) | D3 | Read A label (soup level) |
|---|---|---|---|---|---|---|
| SOUP | s0 | **-1.1242 [-1.3415, -0.9136] B** | true | false (4 rows: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share) | true | MARGIN, NOT DOMINANT |
| SOUP | s1 | **-1.1337 [-1.3387, -0.9306] B** | true | false (4 rows: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share) | true | MARGIN, NOT DOMINANT |
| SOUP' | s0 | **-1.1346 [-1.3472, -0.9266] B** | true | false (4 rows: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share) | true | MARGIN, NOT DOMINANT |
| SOUP' | s1 | **-1.1591 [-1.3614, -0.9603] B** | true | false (4 rows: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share) | true | MARGIN, NOT DOMINANT |
| SOUP4 (beside) | s0 | **-1.1266 [-1.3391, -0.9168] B** | true | false (4 rows: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share) | true | MARGIN, NOT DOMINANT |
| SOUP4 (beside) | s1 | **-1.1354 [-1.3389, -0.9358] B** | true | false (4 rows: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share) | true | MARGIN, NOT DOMINANT |

D3 families that hold, from `D3_families` (all five keys true in every row of the table above): LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC and PRIMARY (ADE6). Read A is the same state for all three soups and both seeds, and it is the state of the control: DAG-OACC − P10 and A − P10 are separated-worse on the same four crawl rows at both seeds (section 2.5, Y − P10 column).

## 3. The soup gain SG and the routing variable (T1-kin)

SG on ADE6 (PREREG section 7): per window, sg_w = ADE6 of the soup − the mean of its members' ADE6, causal lane; paired episode-cluster bootstrap, B = 2,000. SG < 0 separated: the weight average beats the average of its members' errors. SG > 0 separated: a **barrier**. Source: `sg.<soup>:<seed>` (`SOUPR` is SOUP'). The arithmetic columns recompute the mean difference from the table means (`mean_arm`, `mean_ref`, full precision in the JSON) and agree with SG to the rounding shown (the dropped-window count is 0 in every cell).

| soup | seed | members | SG (JSON) | barrier | soup mean ADE6 | member means (in the order of the members) | mean of members | soup − mean of members (arithmetic) | n windows |
|---|---|---|---|---|---|---|---|---|---|
| SOUP (decisive) | s0 | DAG-OACC, REP | **-0.1242 [-0.1517, -0.0977] B** | false | 2.3191 | 2.2421, 2.6446 | 2.4434 | -0.1242 | 2399 |
| SOUP (decisive) | s1 | DAG-OACC, REP | -0.0163 [-0.0379, +0.0050] – | false | 2.3096 | 2.3201, 2.3317 | 2.3259 | -0.0163 | 2399 |
| SOUP' (decisive) | s0 | A, A_rep | **-0.1416 [-0.1693, -0.1152] B** | false | 2.3086 | 2.2128, 2.6877 | 2.4502 | -0.1416 | 2399 |
| SOUP' (decisive) | s1 | A, A_rep | **-0.0488 [-0.0678, -0.0301] B** | false | 2.2842 | 2.3684, 2.2976 | 2.3330 | -0.0488 | 2399 |
| SOUP4 (beside) | s0 | DAG-OACC, REP, A, A_rep | **-0.1301 [-0.1568, -0.1050] B** | false | 2.3167 | 2.2421, 2.6446, 2.2128, 2.6877 | 2.4468 | -0.1301 | 2399 |
| SOUP4 (beside) | s1 | DAG-OACC, REP, A, A_rep | **-0.0216 [-0.0392, -0.0044] B** | false | 2.3078 | 2.3201, 2.3317, 2.3684, 2.2976 | 2.3295 | -0.0216 | 2399 |

**Routing (PREREG section 11).** BARRIER = SG separated-worse (> 0) in at least one of the four decisive soup-seed cells (SOUP and SOUP' at both seeds). Here no cell is a barrier (`barrier` false in all six, `separated-worse` in none), so the route is **FLAT** (JSON `route`). FLAT means the weights average at least as well as the metrics; the harm or the shortfall is the members' draws. Committed next: **SOUP-K** (section 7.1). ENS2, the barrier branch, is not triggered.

## 4. The four families (T1-kin; per family, per seed, per pair, never pooled)

Tier: T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan). Never 'closed loop'. Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, per pair. Evidence class MEASURED (ours); source `four_families.<pair>:<seed>.<family>.rows`, which equals the `tables` cells cell for cell (asserted by `check_result.py`). **STRATEGIC is not paused for v8F (PI ruling 2026-10-17); the JSON carries `paused: false` and the ruling string in every STRATEGIC block.** Each table shows X − Y, the one-lever contrast of the pair, then X − P10 in section 4.6.

### 4.1 Separated cells per family, per seed, per pair (B / W / not separated; X − Y)

| family (rows) | s0 main: B / W / not separated | s0 replicate: B / W / not separated | s1 main: B / W / not separated | s1 replicate: B / W / not separated |
|---|---|---|---|---|
| LONGITUDINAL (6) | 0 / 4 / 2 | 0 / 3 / 3 | 0 / 2 / 4 | 4 / 0 / 2 |
| LATERAL (6) | 1 / 5 / 0 | 0 / 5 / 1 | 0 / 1 / 5 | 2 / 1 / 3 |
| TACTICAL (2) | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 0 / 2 | 0 / 0 / 2 |
| STRATEGIC (3) | 0 / 1 / 2 | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 0 / 3 |
| crawl row outside the family blocks (M1) (1) | 0 / 1 / 0 | 0 / 1 / 0 | 0 / 0 / 1 | 0 / 0 / 1 |

### 4.2 LONGITUDINAL (speed, along-track, accel, distance keeping)

| row | better | s0 main (SOUP − DAG-OACC) | s0 replicate (SOUP' − A) | s1 main (SOUP − DAG-OACC) | s1 replicate (SOUP' − A) | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| speed MAE | lower | **+0.0780 [+0.0458, +0.1096] W** | **+0.0933 [+0.0548, +0.1289] W** | **+0.0432 [+0.0065, +0.0838] W** | **-0.1580 [-0.2508, -0.0800] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| along-track MAE | lower | **+0.1154 [+0.0620, +0.1734] W** | **+0.1244 [+0.0620, +0.1910] W** | -0.0171 [-0.0525, +0.0178] – | **-0.0410 [-0.0808, -0.0002] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| accel MAE | lower | **+0.1601 [+0.1002, +0.2202] W** | **+0.2113 [+0.1376, +0.2824] W** | **+0.2944 [+0.1409, +0.4724] W** | **-0.5807 [-0.9460, -0.2843] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| DK-VIOL | lower | +0.0017 [-0.0125, +0.0148] – | -0.0067 [-0.0239, +0.0102] – | +0.0000 [-0.0083, +0.0083] – | +0.0000 [-0.0096, +0.0088] – | 595/70 ; 595/70 ; 595/70 ; 595/70 |
| DK-HW | lower | +0.0705 [-0.0784, +0.2290] – | +0.0282 [-0.0779, +0.1477] – | -0.0156 [-0.1176, +0.0985] – | **-0.1749 [-0.3244, -0.0532] B** | 712/72 ; 714/72 ; 710/73 ; 710/73 |
| DK-TG | lower | **+0.0534 [+0.0035, +0.1247] W** | +0.0243 [-0.0076, +0.0584] – | -0.0190 [-0.0857, +0.0330] – | -0.0235 [-0.0525, +0.0002] – | 653/72 ; 654/72 ; 653/72 ; 653/72 |

Not built for this family: the anti-echo block (JSON note: *"the anti-echo block is NOT BUILT (the DAG-ORC rig carries none)."*). M-LAUNCH is a separate artifact and is not part of this read.

### 4.3 LATERAL (read first: the crawl rows are LATERAL plus M1)

| row | better | s0 main (SOUP − DAG-OACC) | s0 replicate (SOUP' − A) | s1 main (SOUP − DAG-OACC) | s1 replicate (SOUP' − A) | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| heading MAE | lower | **+0.7747 [+0.4088, +1.1698] W** | **+0.8238 [+0.4677, +1.2019] W** | -0.2560 [-0.6931, +0.1543] – | -0.0932 [-0.2936, +0.0903] – | 2373/141 ; 2373/141 ; 2373/141 ; 2373/141 |
| yaw-rate MAE | lower | **+4.8651 [+3.1469, +6.8062] W** | **+4.8035 [+2.7980, +6.9995] W** | +0.0805 [-1.4927, +1.7032] – | -0.2182 [-1.8500, +1.3532] – | 2366/141 ; 2366/141 ; 2365/141 ; 2367/141 |
| curvature MAE | lower | **+0.0417 [+0.0217, +0.0641] W** | **+0.0290 [+0.0081, +0.0499] W** | -0.0115 [-0.0320, +0.0069] – | **+0.0345 [+0.0130, +0.0616] W** | 2366/141 ; 2366/141 ; 2365/141 ; 2367/141 |
| cross-track MAE | lower | **-0.0333 [-0.0594, -0.0067] B** | -0.0158 [-0.0438, +0.0121] – | +0.0137 [-0.0167, +0.0426] – | **-0.0701 [-0.0996, -0.0442] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| L1 lateral-accel MAE | lower | **+0.4114 [+0.2641, +0.5765] W** | **+0.4133 [+0.2476, +0.5860] W** | **+0.4395 [+0.0446, +0.8770] W** | **-1.4128 [-2.6088, -0.4338] B** | 2366/141 ; 2366/141 ; 2365/141 ; 2367/141 |
| L2 curvature MAE (GT>=5.2) | lower | **+0.0418 [+0.0172, +0.0713] W** | **+0.0306 [+0.0126, +0.0532] W** | +0.0070 [-0.0097, +0.0261] – | +0.0054 [-0.0067, +0.0209] – | 2103/135 ; 2106/135 ; 2105/135 ; 2105/135 |

The M1 crawl share is a crawl row outside the LATERAL block of the JSON; its X − Y cells:

| row | better | s0 main (SOUP − DAG-OACC) | s0 replicate (SOUP' − A) | s1 main (SOUP − DAG-OACC) | s1 replicate (SOUP' − A) | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| M1 crawl share | lower | **+0.0174 [+0.0109, +0.0249] W** | **+0.0171 [+0.0103, +0.0245] W** | +0.0017 [-0.0024, +0.0058] – | +0.0018 [-0.0031, +0.0066] – | 2103/135 ; 2106/135 ; 2105/135 ; 2105/135 |

### 4.4 TACTICAL

| row | better | s0 main (SOUP − DAG-OACC) | s0 replicate (SOUP' − A) | s1 main (SOUP − DAG-OACC) | s1 replicate (SOUP' − A) | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| TAC-LAT | higher | **-0.0125 [-0.0250, -0.0004] W** | -0.0104 [-0.0221, +0.0017] – | +0.0058 [-0.0088, +0.0225] – | +0.0000 [-0.0175, +0.0179] – | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| TAC-LON | higher | -0.0288 [-0.0588, +0.0017] – | **-0.0475 [-0.0775, -0.0179] W** | +0.0058 [-0.0150, +0.0283] – | +0.0021 [-0.0167, +0.0233] – | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |

These are the plan-derived agreements. The JSON note: *"T1-kin goal-head F1 and head accuracy are NOT COMPUTABLE here (the paired tool reads eval dumps, not tick dumps); TAC-LAT / TAC-LON are the plan-derived agreements; R-TLAT / R-TLON are the guards."* The guard booleans are in section 2.3.

### 4.5 STRATEGIC (not paused for v8F)

| row | better | s0 main (SOUP − DAG-OACC) | s0 replicate (SOUP' − A) | s1 main (SOUP − DAG-OACC) | s1 replicate (SOUP' − A) | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| S-POS@30 | lower | **+0.1560 [+0.0280, +0.2845] W** | +0.1351 [-0.0350, +0.3378] – | -0.1162 [-0.2768, +0.0311] – | -0.1468 [-0.4234, +0.0863] – | 2321/141 ; 2321/141 ; 2321/141 ; 2321/141 |
| S-POS@50 | lower | +0.0677 [-0.1959, +0.3082] – | +0.0155 [-0.2782, +0.3015] – | **-0.4784 [-0.8116, -0.1886] B** | -0.0759 [-0.4705, +0.2615] – | 2313/141 ; 2313/141 ; 2313/141 ; 2313/141 |
| S-POS@80 | lower | -0.2972 [-0.7534, +0.1140] – | -0.1196 [-0.6610, +0.3655] – | **-0.6932 [-1.2906, -0.1623] B** | -0.1443 [-0.8481, +0.4621] – | 2290/141 ; 2290/141 ; 2290/141 ; 2290/141 |

S-POS@30 / @50 / @80 (lower is better). The X − Y cell is the difference of the two runs' (S − CTRV, ticks >= 1) contrasts on the same windows (JSON `kind`). Goal and route-setting quality (the strategic decision itself) is not a row of this read, and whether another tool computes it for these arms is UNVERIFIED here.

### 4.6 Against P10: X − P10 and Y − P10, every row (T1-kin)

X − P10 for the decisive soups (SOUP in the main pair, SOUP' in the replicate pair). The STRATEGIC X − P10 cells are unpaired against CTRV (JSON `n`: *ticks>=1, unpaired vs CTRV*); every other cell is paired. Lower is better except TAC-LAT and TAC-LON.

| row | better | SOUP − P10, s0 | SOUP' − P10, s0 | SOUP − P10, s1 | SOUP' − P10, s1 | n, s0 SOUP / s0 SOUP' / s1 SOUP / s1 SOUP' |
|---|---|---|---|---|---|---|
| ADE6 | lower | **-1.1242 [-1.3415, -0.9136] B** | **-1.1346 [-1.3472, -0.9266] B** | **-1.1337 [-1.3387, -0.9306] B** | **-1.1591 [-1.3614, -0.9603] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| speed MAE | lower | **-0.6981 [-0.9162, -0.5103] B** | **-0.7029 [-0.9219, -0.5151] B** | **-0.6598 [-0.8724, -0.4800] B** | **-0.6813 [-0.8880, -0.5092] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| along-track MAE | lower | **-0.5656 [-0.7527, -0.3884] B** | **-0.5750 [-0.7611, -0.3988] B** | **-0.6046 [-0.7871, -0.4233] B** | **-0.6229 [-0.8014, -0.4417] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| accel MAE | lower | **-0.6571 [-1.0175, -0.3428] B** | **-0.6627 [-1.0461, -0.3307] B** | -0.3070 [-0.6459, +0.0186] – | **-0.4430 [-0.7666, -0.1380] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| DK-VIOL | lower | +0.0034 [-0.0197, +0.0263] – | +0.0000 [-0.0225, +0.0216] – | -0.0034 [-0.0226, +0.0162] – | -0.0067 [-0.0303, +0.0160] – | 595/70 ; 595/70 ; 595/70 ; 595/70 |
| DK-HW | lower | -0.2220 [-0.6084, +0.1263] – | -0.2034 [-0.5971, +0.1360] – | **-0.4586 [-0.8480, -0.1419] B** | **-0.5446 [-0.9204, -0.2276] B** | 712/72 ; 713/72 ; 712/72 ; 709/72 |
| DK-TG | lower | -0.0231 [-0.1047, +0.0820] – | -0.0331 [-0.1078, +0.0444] – | **-0.0912 [-0.1503, -0.0423] B** | **-0.1053 [-0.1675, -0.0505] B** | 651/72 ; 652/72 ; 653/72 ; 652/72 |
| heading MAE | lower | **-1.8173 [-2.5117, -1.1031] B** | **-2.1893 [-2.9517, -1.4496] B** | **-2.3099 [-3.0755, -1.5946] B** | **-2.0433 [-2.7943, -1.3272] B** | 2373/141 ; 2373/141 ; 2373/141 ; 2373/141 |
| yaw-rate MAE | lower | **+8.3376 [+4.8170, +12.1092] W** | **+7.5465 [+3.7165, +11.5338] W** | **+8.8067 [+5.1990, +12.9014] W** | **+8.7270 [+5.2159, +12.6264] W** | 2366/141 ; 2366/141 ; 2365/141 ; 2366/141 |
| curvature MAE | lower | **+0.1273 [+0.0878, +0.1730] W** | **+0.1138 [+0.0802, +0.1506] W** | **+0.1236 [+0.0808, +0.1721] W** | **+0.1310 [+0.0859, +0.1853] W** | 2366/141 ; 2366/141 ; 2365/141 ; 2366/141 |
| cross-track MAE | lower | **-0.9952 [-1.1413, -0.8629] B** | **-0.9969 [-1.1414, -0.8658] B** | **-0.9605 [-1.1087, -0.8253] B** | **-0.9723 [-1.1189, -0.8404] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| L1 lateral-accel MAE | lower | **-2.0493 [-3.3682, -0.9100] B** | **-2.0912 [-3.5278, -0.8665] B** | **-1.3530 [-2.4506, -0.3671] B** | **-1.8163 [-3.0492, -0.7231] B** | 2366/141 ; 2366/141 ; 2365/141 ; 2366/141 |
| L2 curvature MAE (GT>=5.2) | lower | **+0.0857 [+0.0458, +0.1362] W** | **+0.0681 [+0.0370, +0.1088] W** | **+0.0590 [+0.0332, +0.0919] W** | **+0.0514 [+0.0288, +0.0798] W** | 2104/135 ; 2106/135 ; 2105/135 ; 2105/135 |
| TAC-LAT | higher | **+0.1780 [+0.1363, +0.2198] B** | **+0.1863 [+0.1471, +0.2288] B** | **+0.2030 [+0.1659, +0.2419] B** | **+0.1972 [+0.1568, +0.2364] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| TAC-LON | higher | **+0.1109 [+0.0434, +0.1777] B** | **+0.1071 [+0.0391, +0.1731] B** | **+0.1430 [+0.0830, +0.2028] B** | **+0.1405 [+0.0792, +0.2027] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| S-POS@30 | lower | **-1.5868 [-2.0893, -1.1647] B** | **-1.7342 [-2.2501, -1.2806] B** | **-1.9417 [-2.4501, -1.5150] B** | **-1.7908 [-2.2301, -1.4026] B** | unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV |
| S-POS@50 | lower | **-4.4654 [-5.5847, -3.4515] B** | **-4.7471 [-5.9188, -3.7021] B** | **-5.0875 [-6.2087, -4.0736] B** | **-4.8386 [-5.8850, -3.9111] B** | unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV |
| S-POS@80 | lower | **-9.8597 [-12.0514, -7.8539] B** | **-10.0826 [-12.1916, -8.0721] B** | **-10.6181 [-12.7240, -8.5841] B** | **-10.4044 [-12.5794, -8.4010] B** | unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV |
| M1 crawl share | lower | **+0.0735 [+0.0507, +0.0992] W** | **+0.0735 [+0.0500, +0.0996] W** | **+0.0616 [+0.0404, +0.0866] W** | **+0.0633 [+0.0424, +0.0877] W** | 2104/135 ; 2106/135 ; 2105/135 ; 2105/135 |

Y − P10 for the controls (DAG-OACC, A):

| row | better | DAG-OACC − P10, s0 | A − P10, s0 | DAG-OACC − P10, s1 | A − P10, s1 | n, s0 DAG-OACC / s0 A / s1 DAG-OACC / s1 A |
|---|---|---|---|---|---|---|
| ADE6 | lower | **-1.2012 [-1.3945, -1.0186] B** | **-1.2305 [-1.4302, -1.0448] B** | **-1.1231 [-1.3301, -0.9159] B** | **-1.0749 [-1.2743, -0.8735] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| speed MAE | lower | **-0.7761 [-0.9806, -0.5996] B** | **-0.7962 [-1.0078, -0.6137] B** | **-0.7030 [-0.9087, -0.5230] B** | **-0.5233 [-0.6966, -0.3707] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| along-track MAE | lower | **-0.6809 [-0.8443, -0.5214] B** | **-0.6994 [-0.8611, -0.5457] B** | **-0.5876 [-0.7702, -0.4119] B** | **-0.5819 [-0.7575, -0.4109] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| accel MAE | lower | **-0.8172 [-1.1770, -0.4998] B** | **-0.8740 [-1.2470, -0.5533] B** | **-0.6014 [-0.9063, -0.3345] B** | +0.1377 [-0.2067, +0.5134] – | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| DK-VIOL | lower | +0.0017 [-0.0231, +0.0272] – | +0.0067 [-0.0164, +0.0314] – | -0.0034 [-0.0243, +0.0181] – | -0.0067 [-0.0302, +0.0162] – | 595/70 ; 595/70 ; 595/70 ; 595/70 |
| DK-HW | lower | **-0.3318 [-0.6942, -0.0242] B** | -0.2519 [-0.6506, +0.0829] – | **-0.4466 [-0.8200, -0.1686] B** | **-0.4007 [-0.7827, -0.0973] B** | 710/72 ; 712/72 ; 710/72 ; 710/72 |
| DK-TG | lower | -0.0568 [-0.1232, +0.0163] – | -0.0537 [-0.1320, +0.0207] – | **-0.0724 [-0.1384, -0.0085] B** | **-0.0817 [-0.1396, -0.0318] B** | 652/72 ; 653/72 ; 652/72 ; 653/72 |
| heading MAE | lower | **-2.5920 [-3.2391, -1.9751] B** | **-3.0131 [-3.7375, -2.3511] B** | **-2.0539 [-2.7092, -1.3705] B** | **-1.9501 [-2.6594, -1.2479] B** | 2373/141 ; 2373/141 ; 2373/141 ; 2373/141 |
| yaw-rate MAE | lower | **+3.4726 [+0.6601, +6.3308] W** | **+2.7420 [+0.3388, +5.2598] W** | **+8.6844 [+5.1690, +12.5740] W** | **+8.9439 [+5.3604, +12.9899] W** | 2365/141 ; 2365/141 ; 2366/141 ; 2366/141 |
| curvature MAE | lower | **+0.0854 [+0.0580, +0.1175] W** | **+0.0847 [+0.0549, +0.1200] W** | **+0.1352 [+0.0889, +0.1915] W** | **+0.0964 [+0.0650, +0.1321] W** | 2365/141 ; 2365/141 ; 2366/141 ; 2366/141 |
| cross-track MAE | lower | **-0.9620 [-1.1077, -0.8319] B** | **-0.9811 [-1.1264, -0.8486] B** | **-0.9742 [-1.1152, -0.8405] B** | **-0.9021 [-1.0493, -0.7638] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| L1 lateral-accel MAE | lower | **-2.4618 [-3.8097, -1.2924] B** | **-2.5056 [-3.8661, -1.3292] B** | **-1.8473 [-2.9617, -0.8501] B** | -0.4029 [-1.3012, +0.4388] – | 2365/141 ; 2365/141 ; 2366/141 ; 2366/141 |
| L2 curvature MAE (GT>=5.2) | lower | **+0.0443 [+0.0232, +0.0744] W** | **+0.0375 [+0.0201, +0.0588] W** | **+0.0529 [+0.0281, +0.0847] W** | **+0.0462 [+0.0260, +0.0716] W** | 2105/135 ; 2106/135 ; 2106/135 ; 2106/135 |
| TAC-LAT | higher | **+0.1905 [+0.1508, +0.2319] B** | **+0.1967 [+0.1580, +0.2378] B** | **+0.1972 [+0.1600, +0.2364] B** | **+0.1972 [+0.1593, +0.2363] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| TAC-LON | higher | **+0.1396 [+0.0771, +0.1997] B** | **+0.1546 [+0.0912, +0.2194] B** | **+0.1371 [+0.0751, +0.2001] B** | **+0.1384 [+0.0746, +0.2006] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| S-POS@30 | lower | **-1.7428 [-2.2737, -1.2861] B** | **-1.8693 [-2.4537, -1.3756] B** | **-1.8255 [-2.3433, -1.3880] B** | **-1.6440 [-2.0440, -1.2864] B** | unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV |
| S-POS@50 | lower | **-4.5331 [-5.6695, -3.5422] B** | **-4.7626 [-5.9590, -3.7194] B** | **-4.6092 [-5.6305, -3.6576] B** | **-4.7627 [-5.7624, -3.8251] B** | unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV |
| S-POS@80 | lower | **-9.5624 [-11.6843, -7.6124] B** | **-9.9630 [-12.0882, -7.9003] B** | **-9.9249 [-11.9750, -7.9686] B** | **-10.2601 [-12.4419, -8.1915] B** | unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV ; unpaired vs CTRV |
| M1 crawl share | lower | **+0.0568 [+0.0378, +0.0798] W** | **+0.0564 [+0.0374, +0.0788] W** | **+0.0604 [+0.0391, +0.0864] W** | **+0.0620 [+0.0415, +0.0870] W** | 2105/135 ; 2106/135 ; 2106/135 ; 2106/135 |

### 4.7 Coverage per family, and what is not computed

`strategic_family` in the JSON reads *"NOT PAUSED for v8F (PI ruling 2026-10-17)"*. Per family, whether a row exists in this read, with the reason and the n where it does not:

| family | rows in this read (T1-kin, X − Y and X − P10) | n (windows / episodes) | not in this read, with the reason |
|---|---|---|---|
| LONGITUDINAL | speed MAE, along-track MAE, accel MAE, DK-VIOL, DK-HW, DK-TG | 2399/141 (speed, along-track, accel); 595/70 (DK-VIOL); 709 to 714 / 72 or 73 (DK-HW); 651 to 654 / 72 (DK-TG) | the anti-echo block: NOT BUILT (the DAG-ORC rig carries none) |
| LATERAL | heading MAE, yaw-rate MAE, curvature MAE, cross-track MAE, L1 lateral-accel MAE, L2 curvature MAE (GT>=5.2); M1 crawl share beside | 2373/141 (heading); 2365 to 2367 / 141 (yaw-rate, curvature, L1); 2399/141 (cross-track); 2103 to 2106 / 135 (L2, M1) | none registered for T1-kin |
| TACTICAL | TAC-LAT, TAC-LON (plan-derived agreements); R-TLAT and R-TLON as reader booleans | 2399/141 | goal-head F1 and head accuracy: NOT COMPUTABLE here (the paired tool reads eval dumps, not tick dumps); the T-head macro-F1 values behind R-TLAT / R-TLON are not carried |
| STRATEGIC | S-POS@30, S-POS@50, S-POS@80 against CTRV within a run (not paused) | 2321/141 (@30); 2313/141 (@50); 2290/141 (@80); X − P10 unpaired | strategic decision and goal / route-setting quality: no row in the JSON; UNVERIFIED whether another tool computes it for these arms |
| all four, at T0 | none: the T0 four families are not in the read | not applicable | PREREG section 8 puts them on the dev-box standard eval (`--families`); the reader carries only the R-T0 points (section 2.3) and the slow-draw cells (section 5.3). Nothing is claimed for T0 LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC |

## 5. Beside, deciding nothing

### 5.1 The members: the slow draw, and the soups against the other contrasts (ADE6 and the crawl rows, T1-kin)

`REP_CTRL` is REP − DAG-OACC (the registered member spread RF). `SOUP_REP` is SOUP − REP, `SOUPR_Arep` is SOUP' − A_rep, `SOUPR_CTRL` is SOUP' − DAG-OACC, `SOUP4_CTRL` is SOUP4 − DAG-OACC.

| contrast | row | s0 | s1 |
|---|---|---|---|
| REP − DAG-OACC | ADE6 | **+0.4025 [+0.2761, +0.5343] W** | +0.0116 [-0.0531, +0.0765] – |
| REP − DAG-OACC | yaw-rate MAE | **+15.8553 [+11.9088, +20.2953] W** | +0.8903 [-0.5695, +2.5371] – |
| REP − DAG-OACC | curvature MAE | **+0.1324 [+0.0904, +0.1795] W** | -0.0008 [-0.0335, +0.0292] – |
| REP − DAG-OACC | L2 curvature MAE (GT>=5.2) | **+0.0870 [+0.0503, +0.1344] W** | +0.0060 [-0.0092, +0.0216] – |
| REP − DAG-OACC | M1 crawl share | **+0.0388 [+0.0270, +0.0519] W** | +0.0048 [-0.0019, +0.0117] – |
| SOUP − REP | ADE6 | **-0.3255 [-0.4077, -0.2478] B** | -0.0221 [-0.0635, +0.0192] – |
| SOUP − REP | yaw-rate MAE | **-10.9849 [-14.1520, -8.2237] B** | -0.8140 [-1.8212, +0.1082] – |
| SOUP − REP | curvature MAE | **-0.0906 [-0.1214, -0.0632] B** | -0.0107 [-0.0348, +0.0091] – |
| SOUP − REP | L2 curvature MAE (GT>=5.2) | **-0.0432 [-0.0705, -0.0195] B** | +0.0001 [-0.0155, +0.0189] – |
| SOUP − REP | M1 crawl share | **-0.0214 [-0.0292, -0.0139] B** | -0.0036 [-0.0088, +0.0012] – |
| SOUP' − A_rep | ADE6 | **-0.3790 [-0.4645, -0.3006] B** | -0.0134 [-0.0567, +0.0333] – |
| SOUP' − A_rep | yaw-rate MAE | **-12.3493 [-15.6355, -9.2541] B** | -0.6297 [-1.7819, +0.4908] – |
| SOUP' − A_rep | curvature MAE | **-0.0968 [-0.1269, -0.0690] B** | -0.0135 [-0.0429, +0.0102] – |
| SOUP' − A_rep | L2 curvature MAE (GT>=5.2) | **-0.0438 [-0.0620, -0.0281] B** | **-0.0176 [-0.0383, -0.0006] B** |
| SOUP' − A_rep | M1 crawl share | **-0.0264 [-0.0348, -0.0177] B** | -0.0034 [-0.0085, +0.0015] – |
| SOUP' − DAG-OACC | ADE6 | **+0.0665 [+0.0076, +0.1272] W** | -0.0359 [-0.0792, +0.0049] – |
| SOUP' − DAG-OACC | yaw-rate MAE | **+4.0731 [+2.1073, +6.0517] W** | +0.0426 [-1.3181, +1.4513] – |
| SOUP' − DAG-OACC | curvature MAE | **+0.0283 [+0.0130, +0.0445] W** | -0.0042 [-0.0277, +0.0238] – |
| SOUP' − DAG-OACC | L2 curvature MAE (GT>=5.2) | **+0.0225 [+0.0074, +0.0421] W** | -0.0003 [-0.0145, +0.0141] – |
| SOUP' − DAG-OACC | M1 crawl share | **+0.0163 [+0.0094, +0.0241] W** | +0.0034 [-0.0021, +0.0087] – |
| SOUP4 − DAG-OACC | ADE6 | **+0.0746 [+0.0174, +0.1347] W** | -0.0123 [-0.0519, +0.0256] – |
| SOUP4 − DAG-OACC | yaw-rate MAE | **+4.4301 [+2.6098, +6.3854] W** | +0.3224 [-0.9141, +1.7029] – |
| SOUP4 − DAG-OACC | curvature MAE | **+0.0332 [+0.0162, +0.0510] W** | -0.0104 [-0.0289, +0.0068] – |
| SOUP4 − DAG-OACC | L2 curvature MAE (GT>=5.2) | **+0.0306 [+0.0132, +0.0529] W** | +0.0027 [-0.0123, +0.0168] – |
| SOUP4 − DAG-OACC | M1 crawl share | **+0.0171 [+0.0101, +0.0249] W** | +0.0028 [-0.0015, +0.0070] – |

### 5.2 SOUP4 on every row (beside; its members overlap both pairs)

| row | better | SOUP4 − DAG-OACC, s0 | SOUP4 − DAG-OACC, s1 | SOUP4 − P10, s0 | SOUP4 − P10, s1 |
|---|---|---|---|---|---|
| ADE6 | lower | **+0.0746 [+0.0174, +0.1347] W** | -0.0123 [-0.0519, +0.0256] – | **-1.1266 [-1.3391, -0.9168] B** | **-1.1354 [-1.3389, -0.9358] B** |
| speed MAE | lower | **+0.0750 [+0.0426, +0.1066] W** | **+0.0458 [+0.0084, +0.0859] W** | **-0.7011 [-0.9167, -0.5157] B** | **-0.6571 [-0.8622, -0.4844] B** |
| along-track MAE | lower | **+0.1119 [+0.0559, +0.1714] W** | -0.0193 [-0.0532, +0.0142] – | **-0.5690 [-0.7566, -0.3923] B** | **-0.6068 [-0.7902, -0.4294] B** |
| accel MAE | lower | **+0.1734 [+0.1095, +0.2396] W** | **+0.2622 [+0.1264, +0.4088] W** | **-0.6438 [-1.0165, -0.3225] B** | **-0.3392 [-0.6469, -0.0390] B** |
| DK-VIOL | lower | +0.0034 [-0.0090, +0.0155] – | -0.0017 [-0.0108, +0.0070] – | +0.0050 [-0.0183, +0.0288] – | -0.0050 [-0.0268, +0.0165] – |
| DK-HW | lower | +0.0670 [-0.0850, +0.2270] – | -0.1000 [-0.2464, +0.0299] – | -0.1829 [-0.5940, +0.1868] – | **-0.4840 [-0.8665, -0.1720] B** |
| DK-TG | lower | **+0.0364 [+0.0014, +0.0783] W** | -0.0268 [-0.0886, +0.0151] – | -0.0261 [-0.1127, +0.0784] – | **-0.0992 [-0.1590, -0.0483] B** |
| heading MAE | lower | **+0.6231 [+0.2494, +0.9974] W** | -0.2093 [-0.6022, +0.1663] – | **-1.9689 [-2.7140, -1.2138] B** | **-2.2632 [-2.9888, -1.5923] B** |
| yaw-rate MAE | lower | **+4.4301 [+2.6098, +6.3854] W** | +0.3224 [-0.9141, +1.7029] – | **+7.9030 [+4.0760, +11.9063] W** | **+9.0068 [+5.4835, +13.0824] W** |
| curvature MAE | lower | **+0.0332 [+0.0162, +0.0510] W** | -0.0104 [-0.0289, +0.0068] – | **+0.1187 [+0.0833, +0.1595] W** | **+0.1248 [+0.0832, +0.1723] W** |
| cross-track MAE | lower | **-0.0310 [-0.0576, -0.0031] B** | +0.0132 [-0.0156, +0.0417] – | **-0.9930 [-1.1389, -0.8612] B** | **-0.9609 [-1.1072, -0.8309] B** |
| L1 lateral-accel MAE | lower | **+0.3668 [+0.2041, +0.5314] W** | +0.2196 [-0.0526, +0.5088] – | **-2.0940 [-3.4661, -0.8994] B** | **-1.6277 [-2.7762, -0.5952] B** |
| L2 curvature MAE (GT>=5.2) | lower | **+0.0306 [+0.0132, +0.0529] W** | +0.0027 [-0.0123, +0.0168] – | **+0.0756 [+0.0400, +0.1241] W** | **+0.0535 [+0.0322, +0.0787] W** |
| TAC-LAT | higher | -0.0042 [-0.0179, +0.0092] – | +0.0033 [-0.0121, +0.0208] – | **+0.1863 [+0.1451, +0.2289] B** | **+0.2005 [+0.1633, +0.2405] B** |
| TAC-LON | higher | -0.0250 [-0.0529, +0.0029] – | +0.0033 [-0.0158, +0.0242] – | **+0.1146 [+0.0475, +0.1794] B** | **+0.1405 [+0.0817, +0.2014] B** |
| S-POS@30 | lower | +0.1258 [-0.0315, +0.2777] – | -0.0967 [-0.3008, +0.1182] – | **-1.6170 [-2.1034, -1.1866] B** | **-1.9222 [-2.4083, -1.5095] B** |
| S-POS@50 | lower | -0.0418 [-0.3448, +0.2079] – | **-0.4128 [-0.7510, -0.1044] B** | **-4.5750 [-5.7105, -3.5567] B** | **-5.0220 [-6.0803, -4.0406] B** |
| S-POS@80 | lower | **-0.4763 [-0.9683, -0.0546] B** | **-0.6826 [-1.3173, -0.0948] B** | **-10.0387 [-12.2035, -7.9959] B** | **-10.6075 [-12.7584, -8.5777] B** |
| M1 crawl share | lower | **+0.0171 [+0.0101, +0.0249] W** | +0.0028 [-0.0015, +0.0070] – | **+0.0743 [+0.0509, +0.0998] W** | **+0.0623 [+0.0416, +0.0870] W** |

### 5.3 The slow-draw cells (T0, dev-box standard eval; speed cells, m/s; per-window paired difference, episode-cluster bootstrap)

`beside.slow_draw`: the soup's speed cells against its pair's control, the method of `rep_s0_speedcells.py`. The reader's `threshold_m_s` is 0.5 and its `label` is null in all four entries. The labelling rule lives in the reader and is not re-read here (UNVERIFIED); no label is claimed. Point [95 % CI], bold when the interval excludes 0.

| contrast (T0) | lon_v_target_ms | v_a_ms | v_end_ms | v_hi_ms | v_lo_ms | reader label | threshold (m/s) |
|---|---|---|---|---|---|---|---|
| SOUP_0 − DAG-OACC_0 | **-0.6143 [-0.7063, -0.5211]** | **-0.3573 [-0.4248, -0.2875]** | **-0.4959 [-0.5693, -0.4233]** | **-0.5088 [-0.5907, -0.4312]** | **-0.6422 [-0.7301, -0.5534]** | null | 0.5 |
| SOUP_1 − DAG-OACC_1 | +0.0067 [-0.0674, +0.0770] | **+0.0898 [+0.0301, +0.1540]** | -0.0638 [-0.1325, +0.0070] | **-0.0999 [-0.1761, -0.0200]** | +0.0441 [-0.0298, +0.1171] | null | 0.5 |
| SOUP'_0 − A_0 | **-0.6880 [-0.7841, -0.5974]** | **-0.4763 [-0.5535, -0.4035]** | **-0.5138 [-0.5896, -0.4360]** | **-0.5747 [-0.6695, -0.4822]** | **-0.6426 [-0.7350, -0.5550]** | null | 0.5 |
| SOUP'_1 − A_1 | +0.0162 [-0.0575, +0.0925] | **+0.0840 [+0.0220, +0.1481]** | -0.0603 [-0.1340, +0.0126] | **-0.1401 [-0.2210, -0.0564]** | +0.0128 [-0.0677, +0.0941] | null | 0.5 |

At s0 the soup's speed cells are negative with intervals below 0 in both pairs (the soup is slower than its control); at s1 every cell is within 0.15 m/s of 0. This is a T0 reading on one control per pair and decides nothing.

### 5.4 Reader self-checks the JSON carries (controls, by family)

368 controls in 43 families, all true (`controls_PASS`). Among them K-RC (the reader reproduces the banked LAUNCH-INTENT tables A − CTRL, Arep − REP, CTRL − P10, REP − P10, REP − CTRL, A − P10, Arep − P10 and RF: 8 controls), K-BASIN and K-MID and K-SOUP-REC (6 each, one per soup), K-XINIT (2, the cross-init soups fail as required), K-ZERO-W (1), K-ZERO (10, (X, X) reads exactly +0 [0, 0]), K-T0DEV-BESIDE (4) and the C3 / C4 window controls (24).

| control family | entries | all true |
|---|---|---|
| ADE-LINK | 4 | PASS |
| C3-FLOOR | 12 | PASS |
| C4-WINDOWS | 12 | PASS |
| DUMP-WINDOWS | 14 | PASS |
| GRID | 14 | PASS |
| GRID-COUNTS | 14 | PASS |
| GRID-logged | 10 | PASS |
| K-BASIN | 6 | PASS |
| K-GATE | 8 | PASS |
| K-INIT | 8 | PASS |
| K-MASK | 4 | PASS |
| K-MID | 6 | PASS |
| K-P0 | 1 | PASS |
| K-PAIR | 6 | PASS |
| K-RC | 8 | PASS |
| K-REGISTERED | 8 | PASS |
| K-REP-LICENCE | 2 | PASS |
| K-SHARE | 8 | PASS |
| K-SOUP-REC | 6 | PASS |
| K-T0DEV-BESIDE | 4 | PASS |
| K-TREE | 6 | PASS |
| K-XINIT | 2 | PASS |
| K-ZERO | 10 | PASS |
| K-ZERO-W | 1 | PASS |
| K1 | 14 | PASS |
| K1-LI-ROWS | 1 | PASS |
| K2 | 14 | PASS |
| K2-logged | 10 | PASS |
| K3 | 14 | PASS |
| K3-logged | 10 | PASS |
| K4 | 10 | PASS |
| K5 | 14 | PASS |
| K6 | 14 | PASS |
| K7 | 14 | PASS |
| R2-ADE6 | 14 | PASS |
| STATIC | 1 | PASS |
| SUMMARY | 8 | PASS |
| T0-CKPT | 14 | PASS |
| T0-WINDOWS | 4 | PASS |
| T0DEV-CKPT | 4 | PASS |
| T0F-CKPT | 10 | PASS |
| T1-CONFIG | 14 | PASS |
| T1-CONFIG-logged | 10 | PASS |

## 6. Reading (decides nothing)

**Label: MEASURED facts from the tables above, arithmetic on them, and one mechanism statement (HYPOTHESIS). The outcome is HARMS, as registered. Nothing in this section changes a flag, a bar or the word.**

**Fact 1. The weight average beats the mean of its members' errors, and no cell is a barrier.** SG on ADE6 is negative at all six soup-seed cells and separated in five (section 3). SOUP at s1 reads -0.0163 [-0.0379, +0.0050], not separated.

**Fact 2. At s0 the soup recovers most of the slow draw and still lands worse than the good one.**

| seed 0, ADE6 (m, T1-kin) | main pair (DAG-OACC_0, REP_0) | replicate pair (A_0, A_rep,0) |
|---|---|---|
| slow member − good member (the member spread) | REP_0 − DAG-OACC_0 +0.4025 [+0.2761, +0.5343] W | A_rep,0 − A_0 +0.4749 (arithmetic: the A − soup delta minus the A_rep − soup delta) |
| soup − slow member | SOUP_0 − REP_0 -0.3255 [-0.4077, -0.2478] B | SOUP'_0 − A_rep,0 -0.3790 [-0.4645, -0.3006] B |
| soup − good member (the control) | SOUP_0 − DAG-OACC_0 +0.0770 [+0.0213, +0.1348] W | SOUP'_0 − A_0 +0.0959 [+0.0321, +0.1624] W |
| share of the member spread recovered (−delta vs slow member / spread) | 80.9 % | 79.8 % |
| position of the soup between the good and the slow member (delta vs good member / spread) | 19.1 % | 20.2 % |

MEASURED (arithmetic on the banked point deltas; the member spread of the replicate pair is the difference of two banked deltas). A linear interpolation of the members' errors puts a two-member soup at 50 % of the way; the PREREG's first-order prediction for SOUP_0 − DAG-OACC_0 was about +0.20 m (ESTIMATED, section 0 item 3: half of +0.4025). The measured point is +0.0770, so the average does better than linear interpolation of the errors, and it is still separated-worse than the good draw: at s0 the lever replaces the good draw with a worse one. A soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base (PREREG section 12).

**Fact 2b. The same holds on the four crawl rows at s0: the soup recovers part of the slow member's crawl and stays worse than the good member on every one.**

| pair (seed 0, T1-kin) | crawl row | member spread (slow − good) | soup − slow member | share of the spread recovered | soup − good member (the control) | position between good and slow member |
|---|---|---|---|---|---|---|
| main (DAG-OACC_0, REP_0) | yaw-rate MAE | +15.8500 | **-10.9849 [-14.1520, -8.2237] B** | 69.3 % | **+4.8651 [+3.1469, +6.8062] W** | 30.7 % |
| main (DAG-OACC_0, REP_0) | curvature MAE | +0.1323 | **-0.0906 [-0.1214, -0.0632] B** | 68.5 % | **+0.0417 [+0.0217, +0.0641] W** | 31.5 % |
| main (DAG-OACC_0, REP_0) | L2 curvature MAE (GT>=5.2) | +0.0850 | **-0.0432 [-0.0705, -0.0195] B** | 50.8 % | **+0.0418 [+0.0172, +0.0713] W** | 49.2 % |
| main (DAG-OACC_0, REP_0) | M1 crawl share | +0.0388 | **-0.0214 [-0.0292, -0.0139] B** | 55.2 % | **+0.0174 [+0.0109, +0.0249] W** | 44.8 % |
| replicate (A_0, A_rep,0) | yaw-rate MAE | +17.1528 | **-12.3493 [-15.6355, -9.2541] B** | 72.0 % | **+4.8035 [+2.7980, +6.9995] W** | 28.0 % |
| replicate (A_0, A_rep,0) | curvature MAE | +0.1258 | **-0.0968 [-0.1269, -0.0690] B** | 76.9 % | **+0.0290 [+0.0081, +0.0499] W** | 23.1 % |
| replicate (A_0, A_rep,0) | L2 curvature MAE (GT>=5.2) | +0.0744 | **-0.0438 [-0.0620, -0.0281] B** | 58.9 % | **+0.0306 [+0.0126, +0.0532] W** | 41.1 % |
| replicate (A_0, A_rep,0) | M1 crawl share | +0.0435 | **-0.0264 [-0.0348, -0.0177] B** | 60.7 % | **+0.0171 [+0.0103, +0.0245] W** | 39.3 % |

MEASURED (arithmetic on the banked point deltas). The spread is the difference of the two banked deltas, (soup − good member) − (soup − slow member), so the two shares add to 100 %. The banked `REP_CTRL` delta (the registered RF) is the same quantity read on its own window set: the two agree on ADE6 and M1 to the 4th decimal and differ on the masked rows by up to 3 windows' worth (the n column of the tables), which moves a share by about one percentage point (ESTIMATED). The soup lies between its members on every crawl row, at 23.1 to 49.2 % of the way from the good to the slow member against 50 % for a linear interpolation of the errors, and is separated-worse than the good member on all four rows in both pairs.

**Fact 3. At s1 the members agree, and the soup sits with them.** REP_1 − DAG-OACC_1 is +0.0116 [-0.0531, +0.0765] –, not separated; SOUP_1 − DAG-OACC_1 is -0.0105 [-0.0477, +0.0263] –, not separated. In the replicate pair A_rep,1 is 0.0708 m better than A_1 by the arithmetic of the two soup deltas, and SOUP'_1 − A_1 is -0.0842 [-0.1298, -0.0400] B, separated-better, while SOUP'_1 − A_rep,1 is -0.0134 [-0.0567, +0.0333] –, not separated: the soup sits near the better member, a little beyond it (SG -0.0488). On the four crawl rows at s1, REP_1 − DAG-OACC_1 is not separated on any (section 5.1), so there is no draw-specific crawl for the average to remove.

**Fact 4. What both draws share, the soup keeps.** No crawl row clears its bar in either pair at either seed (R-MOVE false, section 2.4), and every crawl row of every soup stays separated-worse against P10 (R-CRAWL false, section 2.5). Both controls, DAG-OACC and A, are separated-worse than P10 on the four crawl rows at both seeds (section 2.5, Y − P10 column): the good draw sits above the floor too. Read A is the control's state: D2 fails on the same four rows.

**Reading (HYPOTHESIS).** The crawl rows carry two parts. One is draw-specific: at s0 the slow draws REP_0 and A_rep,0 are separated-worse than DAG-OACC_0 and A_0 on all four crawl rows, and averaging recovers 50.8 to 76.9 % of that excess (Fact 2b). The PREREG's hypothesis (section 1), *"part of the crawl-row error of one draw is draw-specific, and an average of two draws keeps the shared part and halves the draw-specific part,"* is supported in direction at s0. The other part is shared by both draws: both controls are separated-worse than P10 on those rows at both seeds, and on every crawl row at s0 the soup landed between its members (Fact 2b), so it landed above the floor as well. The registered bar is read against the control, which at s0 is the good draw; on the four crawl rows the average with a worse draw did not beat it, and at s1, where the draws agree, there is nothing to average away. So under this design the soup could not have cleared R-MOVE at either seed, which is what the read shows.

## 7. Next (Rule Zero)

### 7.1 Committed next, as registered: SOUP-K

PREREG section 11, FLAT row: **SOUP-K**, K = 4 pure-recipe draws per seed (DAG-OACC_s's argv at trainer seeds 20 + s and 30 + s added to DAG-OACC_s and REP_s; 4 new box-2 runs, about 2.8 h ESTIMATED from the LAUNCH-INTENT runs; its own PI go), read under the same table with SOUP-K_s in place of SOUP_s. **Status: not launched; blocked on the PI go (box-2 time).** The route was decided by the JSON (`route` = FLAT), not by this document.

### 7.2 Rule Zero: what the crawl diagnostics say, and the lever they point to

A refutation is a waypoint. The soup's bar was not cleared, so the question is what the cheapest next lever is and whether the evidence already ranks it. Evidence class as each row states it (MEASURED, PARTIAL or NOT EVALUABLE), quoted from `Project Steering/GOALS_AND_CLAIMS.md` at the tip `0c7b598`; the reading that joins them is a HYPOTHESIS and decides nothing.

| register row | what it measured (quoted) | reading (HYPOTHESIS): what it implies for averaging draws |
|---|---|---|
| `H-CRAWL-01` | On the 327 GT-launch windows (49 episodes) an outcome-blind predictor built on TRAIN reads a gap of the same order as the model's: floor median -1.034 [-1.298, -0.774], mean -0.800 [-1.071, -0.529] m/s; DAG-OACC s0 gap -0.875 [-1.143, -0.601]; D = +0.159 [-0.136, +0.468] (median), -0.075 [-0.377, +0.232] (mean), at the floor on the point; REP s0 gap -1.431, D -0.397 [-0.777, +0.010] (median), -0.631 [-1.008, -0.225] (mean). On GT-stay windows (32 windows / 10 episodes) the models move +1.272 [+1.005, +1.624] and +1.215 [+1.037, +1.366] m/s. | the crawl of DAG-OACC s0 sits at the outcome-blind floor on the point only (outcome (b); CI half-width 0.30): the smaller crawl is what a predictor blind to the launch trigger shows. A different training draw moves the model around that floor (REP s0 crawls deeper than it: D -0.631, mean table, separated); a weight average of draws does not move the floor. |
| `H-CRAWL-02` | S_v >= 0.6 on the stopped-start windows (61 windows / 15 episodes): DAG-OACC +0.736 [+0.593, +0.829], REP +0.628 [+0.501, +0.726]. The frozen P10 prior alone gives +0.686 [+0.557, +0.768], identical in both checkpoints; the learned residual adds +0.050 [+0.015, +0.090] (DAG-OACC) and -0.059 [-0.121, +0.012] (REP). The tracking prediction for the run difference is refuted by sign. | S_v is the frozen prior's. The prior is the same in every member, so an average of members leaves it where it is. |
| `H-CRAWL-03` | NOT EVALUABLE: no intermediate checkpoint exists for either run (the trainer keeps one `ckpt_last.pt` per run). It unblocks with a trainer option that keeps the last k checkpoints, at about zero box-2 cost in the next fine-tune. | whether the 0.556 m/s run-to-run launch difference is between-run or within-run noise stays open; SOUP-K gives more draws, not intermediate checkpoints. |
| `H-ST-REGRET-PRE` | Crawl states are 5.752 % [5.439, 6.062] (s0) and 2.608 % [2.443, 2.782] (s1) of usable DAgger-bank records; the 50 : 50 deficit-weighted multiplier is 2.68 [2.60, 2.77] (s0) and 3.08 [2.99, 3.18] (s1), below the 5x branch. MIDDLE BAND: weighting buys about 3x. | a data-weighting lever buys about 3x on a rare state; it is not a draw-variance lever. |
| `H-ST-RECOVERYLABEL-1` | Kept fraction at tick 2: 53.85 % to 59.51 % (s0), 54.69 % to 61.95 % (s1); the 75 to 81 % prediction is NOT met (V2: 58.20 % / 60.07 %). | the recovery-label route for the off-route states is weaker than predicted. |
| `H-DD-FITGAP-M2B` | On M2b s0 / s1 (2 of the card's 6 checkpoints) the launch-speed deficit on the model's own TRAIN GT-launch windows is as large as on eval-141: r = 1.229 [0.976, 1.680] (s0), 1.612 [1.023, 4.416] (s1), both FIT-LIMITED. PARTIAL: DAG-OACC s0 / s1 and REP s0 / s1 are not measured. | the model crawls on its own training windows as much as on eval, so the crawl is not a data-coverage gap; by the card's reading no data-amount or data-selection lever can clear a crawl bar. Scope: the two M2b checkpoints, not the soup's members. |
| `H-CRAWL-06-PROBE` | On TRAIN, P-MOVE: AUC E 0.731, V+E 0.784, Z+E 0.880; dAUC(V+E - E) +0.053 [0.041, 0.066] by the frozen rule, but with a nonlinear ego baseline +0.016 [0.007, 0.025]. P-STOP as frozen is NOT EVALUABLE (control C2 missed its committed bound by 7e-5). Image gain over E on P-STOP: +0.099 on red-light clips, -0.058 on green-light clips, -0.019 where the lead vehicle departs. | the current-frame image features add little go/stay information over ego, and none on green-light or lead-departure clips. |
| `H-CRAWL-06-PROBE2` | Nav is built from turns of the whole recording on the ego's own future path; the larger ego-future channel is the speed-ceiling input (declared oracle, the PI's 2026-09-10 ruling): L+E - E +0.072 [0.057, 0.087] (P-STOP) / +0.053 [0.044, 0.062] (P-MOVE). With nav and the speed ceiling both scrambled, Z_nl+E - V+E +0.006 [-0.016, 0.027] on P-STOP. NO-MOTION: V4+E - V+E -0.005 [-0.011, +0.001] / -0.000 [-0.003, +0.002]. | the trunk's go/stay edge is the declared speed-ceiling and nav oracles; the 4 frames add nothing. The trigger is weakly present in what v8F sees, and absent for green lights and departing leads. |

**Reading (HYPOTHESIS; decides nothing).** The soup separates two kinds of error. The draw-specific part (REP_0's excess over DAG-OACC_0 on ADE6 and on the crawl rows) is partly recovered by averaging (section 6, Facts 2 and 2b). The part that remains once the draws agree is shared by every member, and the diagnostics place it upstream of anything a weight average acts on: the crawl sits at the outcome-blind floor, S_v is carried by the frozen prior that every member shares, it is fit-limited on the M2b checkpoints that were measured, and the information that would separate go from stay (the trigger) is weakly present in the current-frame features while the trunk's go/stay edge is the declared oracle channels. So averaging, and by the same arithmetic more draws of the same recipe (SOUP-K), can shrink the draw-specific part further and cannot supply a trigger the features barely contain. **The lever the evidence points to is perception.**

**Proposed next experiment, PROPOSED and not registered: a finer-token DINOv3 probe** of the go/stay trigger. The question it would ask is whether DINOv3 tokens finer than the pooled `pool4x10` features of `H-CRAWL-06-PROBE` carry the go/stay trigger beyond ego. Which clips, which probe family and which outcomes is not decided here. It has no card, no committed outcomes and no cost estimate; none is invented. It needs its own PREREG with both outcomes committed before any data (operating standard, rule 5).

### 7.3 What is blocked, and on what (every remaining lever)

| lever | state | what would unblock it |
|---|---|---|
| SOUP-K (committed next) | not launched | the PI go for 4 box-2 runs (about 2.8 h, ESTIMATED in the PREREG) |
| ENS2 (output ensemble; the BARRIER branch) | not triggered: no cell is a barrier | a BARRIER cell in a later soup read; it would need its own PREREG with its own K-ZERO |
| LOOP-STATE (zero training; PREREG_V8F_LAUNCH_INTENT section 11) and an architecture change giving O a launch-intent input | named in PREREG_V8F_SOUP section 11 as the blockers if SOUP-K and ENS2 are exhausted; their status is not re-read here (UNVERIFIED) | the LOOP-STATE read, or a PI decision on the architecture change |
| `H-DD-FITGAP` on the soup's members | PARTIAL: 2 of 6 checkpoints measured (register row) | 3 of DAG-OACC s0 / s1 and REP s0 / s1 (about 0.17 h each on a 10 % clip subset, ESTIMATED in the register) |
| `H-CRAWL-03` intermediate checkpoints | NOT EVALUABLE | a trainer option that keeps the last k checkpoints, with its G-DVB entry |
| finer-token DINOv3 probe (perception) | PROPOSED, not registered | a PREREG with committed outcomes; the cost is not estimated here |

## 8. What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged. T0 is teacher-forced and is not driving performance.
* **A separated interval is necessary, not sufficient** (`H-ESTIM-SEED-1`). The bootstrap resamples EPISODES with the models held fixed: it answers *"would another draw of EPISODES say this?"* and is silent on *"would another training run say this?"*. A soup is two training draws inside one number. Here the two pairs of a seed agree on R-ADE (it fails in both at s0 and holds in both at s1) and the two seeds disagree; the seed-level difference is the slow draw (REP_0, A_rep,0) at s0. With two seeds the read cannot say how often a seed like s0 occurs.
* **A soup effect is the effect of averaging these two draws.** N = 2 per decisive soup; nothing here says a soup of K > 2 behaves the same. SOUP4 is beside and its members overlap both pairs.
* **The control is asymmetric by design** (PREREG section 12). Each soup is read against its trainer-seed-s member. A soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base.
* **The replicate soup carries the launch-intent lever** on both members; its result is the soup lever on the A recipe and attributes nothing to the launch weight.
* **R-T0 is a dev-box CPU T0 reading** on both arms of the pair; it is not box 2's standard eval.
* **SG is a mean-of-errors statistic.** SG < 0 says the average beats the mean of its members' errors. It does not say the soup beats the better member (Fact 2).
* **The words 'perception is the lever' are a reading of diagnostics** (section 7.2), each measured on other arms and checkpoints (`H-DD-FITGAP-M2B`: the two M2b checkpoints; `H-CRAWL-06-PROBE2`: one checkpoint, DAG-OACC s0). They are not a result of this read.
* ⛔ No bar, row, member set, donor rule, K-BASIN factor, routing variable or outcome changed after any soup row was seen.

## 9. Disclosures

1. **The reader's provenance label.** The JSON's `provenance.prereg` reads *"PREREG_V8F_SOUP.md (DRAFT, 2026-10-28)"*; the sha256 it checked is the PROMOTED file's (`830d9c91…`). The promotion (2026-10-29) changed the status line and section 0 item 5 only, and the reader's PREREG-TEXT check passed on it (INHERITED from the brief). The JSON carries no control named PREREG-TEXT; its static controls are one boolean, `controls_PASS.STATIC`, true, and whether PREREG-TEXT sits inside it is UNVERIFIED here.
2. **The launcher was edited and restored.** `soup_chain.cmd` had its two PREREG lines set (lines 10 and 11). A Git-bash `sed` converted it to LF and the Master Mind restored it from the as-shipped CRLF copy before launch (INHERITED from the brief). MEASURED here from the two banked copies (both under `raw/2026-10-29-soup-read/as_launched/`, both CRLF): the launched file differs from the shipped one in exactly those 2 lines.
3. **Disk.** C: had 307 to 886 MB free at hand-back (INHERITED from the brief); 46 old scratchpad folders (about 32 GB) were MOVED to F: before launch (INHERITED from the brief), and the chain's 2 GB gate was not lowered (the gate line below reads 2048 MB). MEASURED in the chain log: the gate line reads *"C: gate: 34097 MB free >= 2048 MB"* at 02:19:45.
4. **A deletion.** The SOUP finishing agent deleted its own test directories in the session scratchpad (INHERITED from the brief). The rule is to move, not delete.
5. **R-T0 used the dev-box T0 of both arms** (PREREG section 0 item 4, K-T0DEV): the soup and the pair's control were both evaluated on the dev-box CPU. Box 2's standard T0 of the members is not the comparison.
6. **A register label.** The register row `H-V8F-SOUP-1` calls the -0.0125 cell "TAC-LAT macro-F1". In the JSON that cell is the TAC-LAT row, a plan-derived agreement (`four_families` note); the guard R-TLAT is a boolean whose macro-F1 value is not carried. Section 2.3 shows the two agree in direction on TAC-LAT in all four pairs and differ on TAC-LON in the s0 replicate pair.
7. **Single copies.** The per-window dumps and tick arrays are under `F:/TanitAD-scratch/soup_t1_out/` (`NOT_LANDED.txt` lists them with sha256). The repo holds the rollout logs, the argv, the done records, the chain log and the JSON.
8. **What `check_result.py` is and is not.** It re-finds every table cell, every quoted string and every prose number in the banked JSON, logs, PREREG and register rows, recomputes the decision from the cells, and checks the document for the banned words and LF. It does **not** recompute the cells from the per-window dumps or re-run the bootstrap. A defect inside the estimator or the reader's heavy layer would not show there.
9. **Instrument status (INHERITED).** The reader's tests and mutation pass are as the register row records (127 tests; 286 of 288 mutants killed, 2 named equivalent); the PREREG's package status records that the heavy layer had not run on real dumps before launch precondition 4, and K-RC (8 controls, all true here; PREREG section 9 defines it as 0 differences from the banked LAUNCH-INTENT tables) is that check.

## 10. Run record

Evidence class MEASURED (ours): the dev-box chain log as banked (`raw/2026-10-29-soup-read/logs/chain.out`). One torch process at a time, dev-box CPU.

The chain started 02:19:45 and its last step ended 02:56:08. Twenty-eight steps ran, each at attempt 1: for each of the six soups a causal rollout, a logged rollout, a fresh-T0 eval and a standard-T0 eval, then the standard T0 of DAG-OACC_0, DAG-OACC_1, A_0 and A_1. The reader started 02:56:28 and the chain closed at 03:07:03 with *"ZZSOUP-DONE 28/28 steps, reader rc 0"*. The 28 steps sum to 2074 s of step wall (derived by `check_result.py` from the log).

| soup | causal rollout (s) | logged rollout (s) | fresh-T0 eval (s) | standard-T0 eval (s) |
|---|---|---|---|---|
| SOUP_0 | 214 | 139 | 15 | 17 |
| SOUP_1 | 138 | 137 | 15 | 18 |
| SOUP'_0 | 185 | 167 | 15 | 18 |
| SOUP'_1 | 138 | 137 | 15 | 18 |
| SOUP4_0 | 138 | 138 | 15 | 18 |
| SOUP4_1 | 137 | 138 | 15 | 18 |

Controls' standard T0 on the dev box (s): DAG-OACC_0 18, DAG-OACC_1 18, A_0 17, A_1 18.

## 11. Provenance (artifact path, sha256)

Evidence class MEASURED (ours). Every sha256 below is of the bytes at the tip `0c7b598edd16a5cbb422ec050a67760d4ba5b11e`, re-measured for this document. Paths are under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/` unless they start with `Project Steering/`.

| role | repo path | sha256 |
|---|---|---|
| read of record (reader output) | `2026-10-29-soup-read/out/read/SOUP_READ.json` | `dca10c07f5fabda9ddc9ea2725f425f53f1462bec8488d58ebc92ecd8d33de6a` |
| read of record, console | `2026-10-29-soup-read/logs/soup_read.out` | `9dabc3d61ee63a6afc2ea474dd692dbb5f597e78415e27bcf92db60780a4d7a9` |
| dev-box chain log | `2026-10-29-soup-read/logs/chain.out` | `b7953fcf98b4166ec6aae9f9386ef0091ffa6ce56b518219ce575cad8c80bc93` |
| launcher as launched (CRLF; two PREREG lines set) | `2026-10-29-soup-read/as_launched/soup_chain.cmd` | `5e4afa1f281239c1e286dc27b21b864c613490f41b12614b6c794fb87450edf5` |
| launcher as shipped (CRLF) | `2026-10-29-soup-read/as_launched/soup_chain.cmd.as_shipped` | `e7a8803e7a52d18a17b50a5590071d552b5cd210cd3c2329540edae952dd77bb` |
| dry run on the promoted PREREG | `2026-10-29-soup-read/as_launched/dry_run_promoted.out` | `76bd8200a8286b87cdb28729b18db79b6b60b036e05770f3a389bcc070e30738` |
| files kept on the dev box (listing) | `2026-10-29-soup-read/NOT_LANDED.txt` | `9ab14a71660a2f6d0cc451c33b8f8b7cbe6e948f3438110ec717172c3fb68ade` |
| reader of record `soup_read.py` | `2026-10-29-soup/read/soup_read.py` | `de39aefe8d5770889309e874e33eceb89fbbb2046b766a049de26ec5a6321c67` |
| LAUNCH-INTENT reader, imported unchanged | `2026-10-29-soup/read/ref/launch_intent_read.py` | `fb3d2d7d945b54209b62d5f906d4dcea30925a24f0b807459b8cbf3d7f0747a9` |
| chain pins | `2026-10-29-soup/PINS.txt` | `fc1e1ccd7e76cc16ca67b8af7fd2759c57c0ed3b5fb7f855ee5cad668dc4a0e1` |
| rendered plan (28 steps) | `2026-10-29-soup/argv/PLAN.txt` | `d7fef7b9423a520bf5f53418a841aad770ed2da59c66addc5e9c166a31072cd6` |
| soup spec | `2026-10-29-soup/soups.json` | `d8935b41d039f69d3db2b9440ff46754f8e26164029df26ec27a8e31da185d97` |
| weight-space readings (K-BASIN, K-XINIT) | `2026-10-29-soup/evidence/basin.json` | `4417351bf9e26bb78fedd1a2a436e5a56c2b90b06bf68672fc16c6980ebb55d2` |
| PREREG as promoted (and at the tip) | `Project Steering/PREREG_V8F_SOUP.md` | `830d9c9143850628188f7d8489566e92f2a9cbdf050689331f1eaa7d3c78993c` |

The dev-box manifest `runs_soup.json` (not staged; F: only) is pinned by the JSON as `provenance.manifest_sha256` `35833493…`. The reader's pins in the JSON: `reader_sha256` `de39aefe…`, `li_reader_sha256` `fb3d2d7d…`, `prereg_sha256` `830d9c91…`, `wall_s` 634.9. The soup checkpoints (E: only, 29.0 MB each, `opt` None): SOUP_s0 `238aab03…` (the chain log prints `238aab039dac` for M2b_s0_soup), SOUP_s1 `81924b1a…`, SOUP'_s0 `3db5166a…`, SOUP'_s1 `e0d1b6ec…`, SOUP4_s0 `006b2a0a…`, SOUP4_s1 `8b07ab1a…` (PREREG, weight-space readings).

