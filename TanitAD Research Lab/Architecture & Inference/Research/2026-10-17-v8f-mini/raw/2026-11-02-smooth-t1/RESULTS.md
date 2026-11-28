# RESULTS: H-SMOOTH-T1-1, the SG(7, 2) residual smoother read inside the T1-kin rollout (2026-11-02)

Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class of every number: **MEASURED (ours; artifacts in `MANIFEST.txt`)** unless a line says ESTIMATED. Estimator on every interval: **paired episode-cluster bootstrap, B = 2,000, seed 0** (`taniteval.ci` draw stream via the landed `gapdecomp.decompose`); it answers *would another draw of EPISODES say this?* and nothing about training or inference variance (each arm is one training run per seed; the planner is deterministic, so there is no inference seed). Card: `PREREG_FROZEN.md`, sha256 `4030d22f...`, frozen 2026-11-02T14:40:42+0100 before any smoothed number existed. Zero training, CPU only.

## 0. Headline

**The outcome is REFUSED by the card's own rule: controls K1, K2 and the floor clause of K5 fail.** They fail on one cause that is not the smoother: the banked rollouts (Y) ran at torch's default **16 threads**, the SMOOTH-T1 rollouts at **1 thread** (the lab's CPU cap), and float32 reduction order differs by thread count. The diagnostics below show the patched tool in `--smooth off` is **bit-identical** to the unpatched tip tool at the same thread count, and that the four row tables move by at most 4.7e-4 (no flag changes sign or separation, same outcome words) between 1 and 2 threads. I did not relax the card's 1e-4 bounds; doing so after the data is not a reader fix.

**Descriptive reading by the committed rule (NOT quotable while the controls fail):** headline arm **DAG-OACC: NO EFFECT**; **SOUP: MOVED**. Neither is DOMINANT (D2 fails on the crawl rows at both seeds), neither HARMS (0 separated-worse rows of 19 in X - Y at every arm and seed; ADE6 X - Y is separated-better, -0.054 to -0.069 m).

What would make it quotable (named, per Rule Zero): (a) the lab lifts the 4-thread cap for one set of **16-thread** reruns (K1 `--smooth off`, then the four sg72 runs; about 3 min each; `k1_matched16_chain.sh` is ready for K1 and was NOT run), or (b) the PI / Master Mind amends the K1 / K2 / K5-floor tolerances to a thread-noise-aware bound, dated after this data and stated as such.

## 1. Controls (known answers, read before any smoothed row)

| control | the card | measured | result |
|---|---|---|---|
| **K1** patched tool, `--smooth off`, DAG-OACC s0, full grid, **threads = 1** | dump and ticks equal the banked ones bit for bit, else max abs deviation <= 1e-4 m (thread count stated) | not bit-identical. Dump `cl` 6.32e-3 m, `fl` 3.76e-3 m, `ol` (tick-0 plan) 1.9e-5 m; ticks: 15 float fields differ; of 7,197 ticks, 2,225 differ by more than 1e-4 m, 42 by more than 1e-2 m and one by more than 0.1 m (max 0.274 m in `plan` / `tau` / `P`, a tick at v0 = 2.0 m/s whose plan reaches 485 m) | **FAIL as written** |
| **K2** (X_off, Y) reads 0 on all 19 rows | row abs(delta) <= 1e-4 | max abs(delta) **2.6e-4** on yaw-rate MAE (deg/s; its interval [-6.7e-4, -1.3e-5] excludes 0, against a Y - P10 gap of +3.47); the other 18 rows <= 1.6e-5 (DK-VIOL, TAC-LAT, TAC-LON, M1 crawl share exactly 0). Per-window max (the stricter first draft): 0.44 | **FAIL as written** (one row) |
| **K3** Y - P10 reproduces the banked `SOUP_READ.json` rows within 6e-5 | | 76 of 76 rows (delta, lo, hi, separated) | PASS |
| **K4** an arm against itself reads exactly 0, [0, 0] | | 76 of 76 | PASS |
| **K5** smoothed ticks: tick-0 `plan_raw` = Y's tick-0 `plan` (<= 1e-4); `plan` = `P + SG(plan_raw - P)` offline (<= 1e-4); floor `fl` = Y's | | tick-0: <= 2.3e-5 m at all four runs; recompute: exactly 0.0 at all four (after the float32 cast, READER_FIXES.md fix 4); S changed the plan by up to 0.90-0.95 m (S is active); `fl`: **max abs deviation 3.76e-3 m, not equal** | plan clauses PASS, **floor clause FAIL** |
| **K6** pytest | planted noisy plan loses high-frequency curvature energy; quadratic and zero residual unchanged; `off` returns the plan object; deleting the smoother call goes RED | `test_v8f_t1_smooth.py` 10 passed (operator, C-OFF against the tip blob, REACH, 2 mutations that must go RED and do); see section 6 | PASS |
| **K7** tick-0 `P` = `p10_yaw4512/p10_eval_pred6.npy` (descriptive) | | max abs 1.5e-5 m | PASS |

**K1 diagnostic (supplementary, not in the card; `raw/K1_DIAG.json`, `code/k1_diag_compare.py`).** Four DAG-OACC s0 rollouts, banked arguments, compared pairwise by max abs deviation (dump / ticks):

| pair | dump | ticks | bit-identical |
|---|---|---|---|
| patched `off`, 1 thread vs **unpatched tip tool** (sha256 `61e140ed...`, the banked tree's tool), 1 thread | 0 | 0 | **yes** |
| patched `off`, 1 thread vs patched `off`, 3 threads | 8.4e-3 m | 0.254 m | no |
| banked (16 threads) vs patched `off`, 1 thread | 6.3e-3 m | 0.274 m | no |
| banked (16 threads) vs patched `off`, 3 threads | 6.5e-3 m | 0.203 m | no |

So (i) the patch does not change `off` (identical bytes out at equal thread count), and (ii) thread count alone moves the rollout by about the same amount as the whole K1 deviation. The deviation is small on average (`cl` mean 1.8e-5 m) and concentrated in a few ticks (counts in the K1 row above); I did not trace the amplification of those ticks to a mechanism.

**Thread-noise replicate (supplementary; `raw/THREADREP.json`, Table D below).** The four sg72 rollouts were run again at 2 threads and read with the same machinery. The largest change of any row delta (X - P10 or X - Y, 19 rows, both arms, both seeds) is **4.7e-4** (SOUP s0, yaw-rate MAE); **no row changes its separated flag or its sign**; the outcome words are identical (DAG-OACC NO EFFECT, SOUP MOVED). This bounds the 1-vs-2-thread effect on the rows, not the 1-vs-16-thread effect on X; the 16-thread effect on the 19 rows is bounded only through K2 (X_off - Y, at most 2.6e-4).

## 2. The outcome by the committed rule (descriptive; see the headline)

The rule: **DOMINANT** iff at BOTH seeds X - P10 has ADE6 separated-better with point <= -0.2415 m (D1), no row of 19 separated-worse (D2), and each of the four families has a separated-better row (D3; STRATEGIC is not paused for v8F). **HARMS** iff at EITHER seed ADE6 X - Y point > +0.0604 m or a non-crawl row (the 15 rows other than yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share) reads X - Y separated-worse. **MOVED** iff not DOMINANT and at BOTH seeds all four crawl rows read X - Y separated-better with ADE6 X - Y <= +0.0604 m. **NO EFFECT** otherwise. Precedence DOMINANT, HARMS, MOVED, NO EFFECT. Multiplicity: HARMS tests 15 rows x 2 seeds = 30 tests per arm; the count of separated-worse rows of 19 is in Table A (0 at all four arm-seeds, so HARMS did not fire before any multiplicity adjustment).

Reading Table A (generated below):
* **D1 passes everywhere** (ADE6 X - P10 between -1.19 and -1.26 m, separated-better) and **D3 passes everywhere** (all four families, STRATEGIC included, hold a separated-better row; Table E).
* **D2 is what blocks DOMINANT**, at both seeds of both arms, always on the same three crawl rows (curvature MAE, L2 curvature MAE, M1 crawl share), plus yaw-rate MAE at seed 1 of both arms.
* **DAG-OACC fails MOVED at seed 1**: curvature MAE, L2 curvature MAE and M1 crawl share are not separated in X - Y there (only yaw-rate MAE moves). Hence NO EFFECT, although seed 0 meets the MOVED condition. **SOUP meets MOVED at both seeds**, two of its eight crawl-row X - Y intervals ending at -0.0002 (curvature MAE and M1 crawl share at seed 1) and a third at -0.0008 (L2 curvature MAE at seed 1), so MOVED for SOUP is a boundary result (it survives the 2-thread replicate, Table D). DAG-OACC s0's L2 curvature MAE interval also ends at -0.0004.
* Statement per arm, as the card asks: headline arm DAG-OACC **NO EFFECT**; SOUP stated separately **MOVED**. Both are descriptive until the controls pass.

## 3. The four crawl rows, ADE6, and the four families

Table B gives X - P10, X - Y and Y - P10 for ADE6 and the four crawl rows at both seeds of both arms; Table C gives all 19 rows (all four families) for each arm and seed; Table E counts rows by flag per family. The reading, from those tables:
* **S moves the crawl rows in the right direction and does not close them.** Share of the Y - P10 gap removed (ESTIMATED, point-estimate ratio, Table B): yaw-rate MAE 0.52 to 1.40, curvature MAE 0.10 to 0.32, L2 curvature MAE 0.05 to 0.26, M1 crawl share 0.06 to 0.14. On yaw-rate MAE X - P10 is not separated at seed 0 of both arms (separated-worse at seed 1); curvature MAE, L2 curvature MAE and M1 crawl share stay separated-worse than P10 in all four arm-seeds.
* **S costs nothing on ADE6** and slightly improves it (X - Y -0.054 / -0.067 m for DAG-OACC, -0.069 / -0.056 m for SOUP, all separated-better).
* **No family is harmed**: no row of 19 is separated-worse in X - Y at any arm-seed (Table A, Table E).
* **X - Y, per family (Table E):** no separated-worse row in any family at any arm-seed; separated-better rows: LONGITUDINAL 2 to 3 of 6, LATERAL 4 to 6 of 6, TACTICAL 0 to 2 of 2, STRATEGIC 3 of 3 (the TACTICAL range runs from 2 of 2 at DAG-OACC s0 to 0 of 2 at SOUP s1). Against P10 (X - P10) every family holds a separated-better row at every arm-seed, and the only separated-worse rows are the crawl rows (D2 above).

## 4. Beside the rule (descriptive; decides nothing): curvature jitter in the 3-8 m/s band of M (413 windows)

Source `raw/SMOOTH_T1.json` `descriptive`. Mean |kappa| (1/m) is on the steps where the path and the GT are pair-valid (the mask that produced the banked 0.1845 / 0.2621, which Y reproduces for DAG-OACC: **0.1845 / 0.2621**). Roughness = mean squared first difference of kappa along the path (1/m^2 per step^2).

| run | executed path, Y: mean abs kappa / roughness | executed path, X (with S) | issued plan before S -> after S: roughness | P10 / GT mean abs kappa |
|---|---|---|---|---|
| DAG-OACC s0 | 0.1845 / 1.103 | 0.1658 / 1.381 | 0.0992 -> 0.0798 | 0.0623 / 0.0135 |
| DAG-OACC s1 | 0.2621 / 1.788 | 0.2444 / 2.750 | 0.1381 -> 0.0926 | 0.0623 / 0.0135 |
| SOUP s0 | 0.2912 / 2.329 | 0.2415 / 2.165 | 0.1423 -> 0.1181 | 0.0623 / 0.0135 |
| SOUP s1 | 0.2366 / 1.662 | 0.2079 / 1.688 | 0.1549 -> 0.1300 | 0.0623 / 0.0135 |

**What it shows.** S does what it is built to do on the issued plan (roughness down 16 to 33 percent) and lowers the executed path's mean |kappa| by 7 to 17 percent, but the executed path stays at 2.7 to 3.9 times P10's mean |kappa| and its roughness did **not** fall (up at both DAG-OACC seeds, down 7 percent or flat for SOUP). Smoothing a plan does not smooth the path the executor commits.

**Post-hoc diagnostic, not in the card (`code/seam_diag.py`, `raw/SEAM_DIAG.json`).** The executor commits the first 10 knots of each replan, so the path is three 10-step blocks joined at steps 10 and 20, and S smooths within a plan only. Mean |kappa| next to the seams (steps 9-11, 19-21) against all other steps: Y 0.2202 vs 0.1449, X 0.1664 vs 0.1238 (DAG-OACC s0); P10 0.0792 vs 0.0368, GT 0.0136 vs 0.0132. The seam bump (seam minus other steps) is real in Y (0.075 at DAG-OACC s0) and in P10 (0.042). **S lowers it** (X 0.043 at DAG-OACC s0, equal to P10's; Y 0.072 -> X 0.047 at s1; SOUP 0.057 -> 0.048 and 0.080 -> 0.061; `SEAM_DIAG.json`). What remains is an **excess over P10 that is nearly uniform across steps** (X minus P10 at the seam steps against the other steps: 0.087 vs 0.087 at DAG-OACC s0, and within 3 to 15 percent of each other in the other three runs); it covers 23 of the 29 curvature steps, and an SG(7, 2) smoother, which removes high-frequency residual within a plan, does not act on it. HYPOTHESIS (not tested here): the uniform excess is a per-knot curvature overshoot of the decoded plan itself, which a curvature-bounded decode, not a smoother, would address.

## 5. Prediction against result

Card prediction (ESTIMATED from the T0 effect, written before the read): DOMINANT <= 5 percent, MOVED about 30 percent, NO EFFECT about 55 percent, HARMS about 10 percent; lateral rows move separated-better but do not close; the M1 crawl share may not move.
* Headline arm: NO EFFECT, the modal prediction (descriptive). SOUP landed on MOVED, the 30 percent branch. No HARMS, no DOMINANT.
* "Lateral rows move but do not close": **held** (yaw-rate MAE X - Y separated-better at 4 of 4 arm-seeds; curvature MAE, L2 curvature MAE and M1 crawl share still separated-worse than P10 at 4 of 4).
* "The M1 crawl share may not move": **too pessimistic** by a little: it moved separated-better at 3 of 4 arm-seeds (not DAG-OACC s1), by 0.004 to 0.008, which is 7 to 14 percent of its gap. The step rule is the least reachable row.

## 6. Tests (K6) and an integration finding

Run against the patched tool (sha256 `caa71055...`) in a raw-blob export of the tip `51fd683c...` plus the patched file (`F:/TanitAD-scratch/smooth_t1/pt/`; logs in `tests/`), 2 threads, CPU:

| file | patched tool | note |
|---|---|---|
| `test_v8f_t1_smooth.py` (new) | **10 passed** | C-OFF run against the tip blob `3a516754` (env `V8F_T1_TIP_TOOL`) |
| `test_v8f_t1.py` | **23 passed** | |
| `test_v8f_t1_causal.py` | **12 passed** | |
| `test_v8f_t1_heads.py` | **14 passed** | its C-OFF test pins a different tip blob (`635487e5`, sha256 `baf16101...`, the tool before `--dump-heads`); it is run with that file in `V8F_T1_TIP_TOOL` |
| `test_v8f_t1_speedsub.py` | **5 passed, 28 failed or errored** | every one traces to the first setup error `REFUSED: the T1 tool ... has sha256 caa71055f50f..., the pin is 61e140edb9da...` (the rest are pytest fixture-cache knock-ons). **Unpatched tip tool in the same tree: 33 passed** (`tests/pytest_speedsub_unpatched_baseline.log`) |

**Integration finding:** `stack/scripts/v8f_t1_speedsub.py:44` pins the T1 tool by sha256 (`TOOL_SHA256 = "61e140ed..."`), so overwriting `stack/scripts/v8f_t1_eval.py` with the patched file makes that wrapper refuse it. Many banked artifacts (`PINS.txt`, launch-gate tokens under `.../raw/2026-10-27-*`) also name `61e140ed...`; those are historical records and must not be rewritten. Two ways to land, the Master Mind's call: (1) overwrite the tool and move the speedsub pin to `caa71055...` in the same landing (the behaviour in `off` is byte-identical, see K1 diagnostic), or (2) keep the tip tool untouched and land the patch as a separate tool file with its own test import.

## 7. The next arm (Rule Zero)

The card commits, for NO EFFECT (DAG-OACC) and for MOVED (SOUP) alike, the same next zero-training arm: **a curvature-bounded decode** (`stack/tanitad/refs/feasible_decode.py` envelope, KAPPA_MAX 0.2) as a flag of the same tool, composed with S for SOUP (MOVED names the rows still separated-worse vs P10: curvature MAE, L2 curvature MAE and M1 crawl share at both seeds, yaw-rate MAE at seed 1). The seam diagnostic (section 4) supports it: the remaining excess is in-block, per-knot curvature. I did **not** run it: it needs its own pre-registration frozen before data, and any new arm compared against the banked Y at fewer than 16 threads inherits the K1 / K2 failure above. **Named blockers:** (1) the thread-count decision of section 0 (a / b); (2) the landing decision of section 6; (3) the pre-registration of the decode arm (design: arms D = decode alone, SD = S + decode, both against Y and against P10, same 19 rows and the same outcome words).

## 8. Files, and what was changed

`READER_FIXES.md` lists the six reader changes (none touches an outcome function, row, bar or estimator; chronology stated there). Package files: `code/` (patched tool `v8f_t1_eval.py`, extractor `extract_smooth.py`, reader `smooth_t1_read.py`, table generator `make_results_tables.py`, and the supplementary `k1_diag_compare.py`, `threadrep_read.py`, `seam_diag.py`, `bank_rollouts.py`); chains `smooth_t1_chain.*` (the card's five rollouts), `k1_diag_chain.*` and `smooth_t1_threadrep_chain.*` (supplementary, run), `k1_matched16_chain.sh` (READY, NOT RUN: needs 16 threads); `raw/` (`SMOOTH_T1.json`, `THREADREP.json`, `K1_DIAG.json`, `SEAM_DIAG.json`, `RESULTS_TABLES.md`, the extracted per-window arrays in `arrays/` and `arrays_t2/`, and `rollouts/` = the banked rollout outputs: the card's five runs with dump, ticks and JSON, and the replicate and diagnostic runs with dump and JSON); `logs/`, `tests/`, `MANIFEST.txt`. Everything is a SINGLE COPY on E: (and regenerable scratch on F:); nothing is in git; the Master Mind lands it. Caveats: Y is the banked arrays of the H-GAPDECOMP-1 package (not re-rolled); precedence HARMS before MOVED is the card's own tie-break (no overlap occurred); this read says nothing about training variance (one run per seed) or inference variance (deterministic planner).

# Tables (generated from `raw/SMOOTH_T1.json` and `raw/THREADREP.json` by `code/make_results_tables.py`)

<!-- generated by code/make_results_tables.py from raw/SMOOTH_T1.json: no hand-typed number -->

Cell = `delta [95% interval] flag`; flag **B** = separated on the better side of the row's direction, **W** = separated on the worse side, `.` = not separated. Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0 (answers another draw of EPISODES only). Tier T1-kin. Units as in `dagland/gapdecomp/RESULTS.md`. n = windows / episodes of the row's pool. On S-POS@k the P10 side is the CTRV of the run's own ticks (a within-run contrast), so X - P10 is not (X - Y) + (Y - P10) on those three rows.

### Table A. The outcome rule, condition by condition (descriptive: the controls refuse the outcome, see section 1)

| arm | seed | D1 ADE6 X - P10 | D2 no separated-worse row (X - P10) | separated-worse rows (X - P10) | D3 each family has a separated-better row | ADE6 X - Y | all four crawl rows X - Y separated-better | non-crawl rows X - Y separated-worse | separated-worse rows of 19 (X - Y) | MOVED cond. | HARMS cond. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DAG-OACC | s0 | PASS (-1.2556 m) | FAIL | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE | PASS (LON:Y, LAT:Y, TAC:Y, STR:Y) | -0.0544 m | yes | none | 0 | met | not met |
| DAG-OACC | s1 | PASS (-1.1903 m) | FAIL | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE | PASS (LON:Y, LAT:Y, TAC:Y, STR:Y) | -0.0671 m | no | none | 0 | not met | not met |
| SOUP | s0 | PASS (-1.1928 m) | FAIL | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE | PASS (LON:Y, LAT:Y, TAC:Y, STR:Y) | -0.0687 m | yes | none | 0 | met | not met |
| SOUP | s1 | PASS (-1.1895 m) | FAIL | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE | PASS (LON:Y, LAT:Y, TAC:Y, STR:Y) | -0.0558 m | yes | none | 0 | met | not met |

| arm | DOMINANT | HARMS (either seed) | MOVED (both seeds) | outcome by the committed rule |
|---|---|---|---|---|
| DAG-OACC | no | no | no | **NO EFFECT** |
| SOUP | no | no | yes | **MOVED** |

### Table B. ADE6 and the four crawl rows, both arms, both seeds

`share removed` = -(X - Y) / (Y - P10) at the point estimates: the share of the Y - P10 gap that S removed (ESTIMATED arithmetic on two point estimates, no interval).

| arm | seed | row | X - P10 | X - Y | Y - P10 | share removed |
|---|---|---|---|---|---|---|
| DAG-OACC | s0 | ADE6 | -1.2556 [-1.4503, -1.0678] B | -0.0544 [-0.0859, -0.0230] B | -1.2012 [-1.3945, -1.0186] B | n/a |
| DAG-OACC | s0 | yaw-rate MAE | -1.3765 [-3.9127, +1.0256] . | -4.8537 [-6.7267, -3.1576] B | +3.4726 [+0.6601, +6.3308] W | 1.40 |
| DAG-OACC | s0 | curvature MAE | +0.0578 [+0.0364, +0.0821] W | -0.0276 [-0.0418, -0.0148] B | +0.0854 [+0.0580, +0.1175] W | 0.32 |
| DAG-OACC | s0 | L2 curvature MAE (GT>=5.2) | +0.0325 [+0.0135, +0.0590] W | -0.0104 [-0.0205, -0.0004] B | +0.0443 [+0.0232, +0.0744] W | 0.23 |
| DAG-OACC | s0 | M1 crawl share | +0.0483 [+0.0302, +0.0698] W | -0.0080 [-0.0135, -0.0031] B | +0.0568 [+0.0378, +0.0798] W | 0.14 |
| DAG-OACC | s1 | ADE6 | -1.1903 [-1.3986, -0.9782] B | -0.0671 [-0.1014, -0.0358] B | -1.1231 [-1.3301, -0.9159] B | n/a |
| DAG-OACC | s1 | yaw-rate MAE | +4.1333 [+1.1648, +7.3581] W | -4.5561 [-6.2809, -2.9151] B | +8.6844 [+5.1690, +12.5740] W | 0.52 |
| DAG-OACC | s1 | curvature MAE | +0.1210 [+0.0757, +0.1727] W | -0.0141 [-0.0397, +0.0097] . | +0.1352 [+0.0889, +0.1915] W | 0.10 |
| DAG-OACC | s1 | L2 curvature MAE (GT>=5.2) | +0.0505 [+0.0269, +0.0791] W | -0.0024 [-0.0195, +0.0123] . | +0.0529 [+0.0281, +0.0847] W | 0.05 |
| DAG-OACC | s1 | M1 crawl share | +0.0565 [+0.0347, +0.0829] W | -0.0039 [-0.0088, +0.0011] . | +0.0604 [+0.0391, +0.0864] W | 0.06 |
| SOUP | s0 | ADE6 | -1.1928 [-1.4078, -0.9880] B | -0.0687 [-0.1017, -0.0360] B | -1.1242 [-1.3415, -0.9136] B | n/a |
| SOUP | s0 | yaw-rate MAE | +1.9311 [-1.0565, +5.0084] . | -6.4174 [-8.3852, -4.7241] B | +8.3376 [+4.8170, +12.1092] W | 0.77 |
| SOUP | s0 | curvature MAE | +0.0919 [+0.0584, +0.1301] W | -0.0354 [-0.0538, -0.0194] B | +0.1273 [+0.0878, +0.1730] W | 0.28 |
| SOUP | s0 | L2 curvature MAE (GT>=5.2) | +0.0628 [+0.0279, +0.1076] W | -0.0219 [-0.0372, -0.0071] B | +0.0857 [+0.0458, +0.1362] W | 0.26 |
| SOUP | s0 | M1 crawl share | +0.0666 [+0.0428, +0.0927] W | -0.0073 [-0.0127, -0.0021] B | +0.0735 [+0.0507, +0.0992] W | 0.10 |
| SOUP | s1 | ADE6 | -1.1895 [-1.3938, -0.9847] B | -0.0558 [-0.0890, -0.0229] B | -1.1337 [-1.3387, -0.9306] B | n/a |
| SOUP | s1 | yaw-rate MAE | +3.0978 [+0.2214, +6.2180] W | -5.6927 [-7.6797, -3.9568] B | +8.8067 [+5.1990, +12.9014] W | 0.65 |
| SOUP | s1 | curvature MAE | +0.1068 [+0.0667, +0.1533] W | -0.0175 [-0.0356, -0.0002] B | +0.1236 [+0.0808, +0.1721] W | 0.14 |
| SOUP | s1 | L2 curvature MAE (GT>=5.2) | +0.0454 [+0.0224, +0.0739] W | -0.0138 [-0.0318, -0.0008] B | +0.0590 [+0.0332, +0.0919] W | 0.23 |
| SOUP | s1 | M1 crawl share | +0.0576 [+0.0364, +0.0823] W | -0.0044 [-0.0089, -0.0002] B | +0.0616 [+0.0404, +0.0866] W | 0.07 |

### Table C. DAG-OACC s0: all 19 rows, X (arm + S) against P10 and against Y (the same checkpoint, no S)

| family | row | n | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 | 2399/141 | -1.2556 [-1.4503, -1.0678] B | -0.0544 [-0.0859, -0.0230] B | -1.2012 [-1.3945, -1.0186] B |
| LONGITUDINAL | speed MAE | 2399/141 | -0.8330 [-1.0378, -0.6583] B | -0.0569 [-0.0962, -0.0192] B | -0.7761 [-0.9806, -0.5996] B |
| LONGITUDINAL | along-track MAE | 2399/141 | -0.7192 [-0.8806, -0.5627] B | -0.0383 [-0.0684, -0.0104] B | -0.6809 [-0.8443, -0.5214] B |
| LONGITUDINAL | accel MAE | 2399/141 | -0.9389 [-1.3075, -0.6219] B | -0.1217 [-0.2283, -0.0258] B | -0.8172 [-1.1770, -0.4998] B |
| LONGITUDINAL | DK-VIOL | 595/70 | +0.0168 [-0.0157, +0.0625] . | +0.0151 [-0.0035, +0.0418] . | +0.0017 [-0.0231, +0.0272] . |
| LONGITUDINAL | DK-HW | 707/72 | -0.2604 [-0.6426, +0.0675] . | +0.0634 [-0.0657, +0.1912] . | -0.3318 [-0.6942, -0.0242] B |
| LONGITUDINAL | DK-TG | 650/72 | -0.0588 [-0.1226, -0.0011] B | +0.0167 [-0.0134, +0.0543] . | -0.0568 [-0.1232, +0.0163] . |
| LATERAL | heading MAE | 2372/141 | -3.4518 [-4.2137, -2.7528] B | -0.8616 [-1.2326, -0.5485] B | -2.5920 [-3.2391, -1.9751] B |
| LATERAL | yaw-rate MAE | 2364/141 | -1.3765 [-3.9127, +1.0256] . | -4.8537 [-6.7267, -3.1576] B | +3.4726 [+0.6601, +6.3308] W |
| LATERAL | curvature MAE | 2364/141 | +0.0578 [+0.0364, +0.0821] W | -0.0276 [-0.0418, -0.0148] B | +0.0854 [+0.0580, +0.1175] W |
| LATERAL | cross-track MAE | 2399/141 | -1.0009 [-1.1468, -0.8671] B | -0.0390 [-0.0614, -0.0171] B | -0.9620 [-1.1077, -0.8319] B |
| LATERAL | L1 lateral-accel MAE | 2364/141 | -2.7324 [-4.0990, -1.5824] B | -0.2697 [-0.5718, +0.0551] . | -2.4618 [-3.8097, -1.2924] B |
| LATERAL | L2 curvature MAE (GT>=5.2) | 2104/135 | +0.0325 [+0.0135, +0.0590] W | -0.0104 [-0.0205, -0.0004] B | +0.0443 [+0.0232, +0.0744] W |
| TACTICAL | TAC-LAT | 2399/141 | +0.2038 [+0.1639, +0.2450] B | +0.0133 [+0.0025, +0.0242] B | +0.1905 [+0.1508, +0.2319] B |
| TACTICAL | TAC-LON | 2399/141 | +0.1751 [+0.1138, +0.2389] B | +0.0354 [+0.0138, +0.0588] B | +0.1396 [+0.0771, +0.1997] B |
| STRATEGIC | S-POS@30 | 2321/141 | -1.6443 [-2.1654, -1.1864] B | -0.2018 [-0.3060, -0.1101] B | -1.7428 [-2.2737, -1.2861] B |
| STRATEGIC | S-POS@50 | 2313/141 | -4.5744 [-5.8090, -3.5098] B | -0.3381 [-0.5177, -0.1810] B | -4.5331 [-5.6695, -3.5422] B |
| STRATEGIC | S-POS@80 | 2290/141 | -9.8284 [-12.1173, -7.7692] B | -0.5404 [-0.8388, -0.2789] B | -9.5624 [-11.6843, -7.6124] B |
| crawl | M1 crawl share | 2104/135 | +0.0483 [+0.0302, +0.0698] W | -0.0080 [-0.0135, -0.0031] B | +0.0568 [+0.0378, +0.0798] W |

### Table C. DAG-OACC s1: all 19 rows, X (arm + S) against P10 and against Y (the same checkpoint, no S)

| family | row | n | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 | 2399/141 | -1.1903 [-1.3986, -0.9782] B | -0.0671 [-0.1014, -0.0358] B | -1.1231 [-1.3301, -0.9159] B |
| LONGITUDINAL | speed MAE | 2399/141 | -0.8141 [-1.0377, -0.6191] B | -0.1112 [-0.1631, -0.0631] B | -0.7030 [-0.9087, -0.5230] B |
| LONGITUDINAL | along-track MAE | 2399/141 | -0.6302 [-0.8196, -0.4510] B | -0.0426 [-0.0738, -0.0117] B | -0.5876 [-0.7702, -0.4119] B |
| LONGITUDINAL | accel MAE | 2399/141 | -0.9720 [-1.3432, -0.6573] B | -0.3706 [-0.5472, -0.2191] B | -0.6014 [-0.9063, -0.3345] B |
| LONGITUDINAL | DK-VIOL | 595/70 | +0.0000 [-0.0196, +0.0212] . | +0.0034 [-0.0090, +0.0142] . | -0.0034 [-0.0243, +0.0181] . |
| LONGITUDINAL | DK-HW | 707/72 | -0.5171 [-0.9057, -0.2118] B | -0.0296 [-0.1164, +0.0521] . | -0.4466 [-0.8200, -0.1686] B |
| LONGITUDINAL | DK-TG | 653/72 | -0.0869 [-0.1542, -0.0278] B | -0.0117 [-0.0436, +0.0218] . | -0.0724 [-0.1384, -0.0085] B |
| LATERAL | heading MAE | 2373/141 | -2.8957 [-3.6062, -2.2013] B | -0.8418 [-1.1816, -0.5690] B | -2.0539 [-2.7092, -1.3705] B |
| LATERAL | yaw-rate MAE | 2365/141 | +4.1333 [+1.1648, +7.3581] W | -4.5561 [-6.2809, -2.9151] B | +8.6844 [+5.1690, +12.5740] W |
| LATERAL | curvature MAE | 2365/141 | +0.1210 [+0.0757, +0.1727] W | -0.0141 [-0.0397, +0.0097] . | +0.1352 [+0.0889, +0.1915] W |
| LATERAL | cross-track MAE | 2399/141 | -1.0217 [-1.1686, -0.8803] B | -0.0476 [-0.0729, -0.0230] B | -0.9742 [-1.1152, -0.8405] B |
| LATERAL | L1 lateral-accel MAE | 2365/141 | -2.6083 [-3.8412, -1.5152] B | -0.7603 [-1.1823, -0.3600] B | -1.8473 [-2.9617, -0.8501] B |
| LATERAL | L2 curvature MAE (GT>=5.2) | 2106/135 | +0.0505 [+0.0269, +0.0791] W | -0.0024 [-0.0195, +0.0123] . | +0.0529 [+0.0281, +0.0847] W |
| TACTICAL | TAC-LAT | 2399/141 | +0.2126 [+0.1709, +0.2540] B | +0.0154 [+0.0021, +0.0321] B | +0.1972 [+0.1600, +0.2364] B |
| TACTICAL | TAC-LON | 2399/141 | +0.1638 [+0.1021, +0.2225] B | +0.0267 [-0.0008, +0.0533] . | +0.1371 [+0.0751, +0.2001] B |
| STRATEGIC | S-POS@30 | 2321/141 | -1.7415 [-2.2209, -1.3273] B | -0.3100 [-0.5077, -0.1403] B | -1.8255 [-2.3433, -1.3880] B |
| STRATEGIC | S-POS@50 | 2313/141 | -4.7599 [-5.8591, -3.7483] B | -0.5164 [-0.8574, -0.2229] B | -4.6092 [-5.6305, -3.6576] B |
| STRATEGIC | S-POS@80 | 2290/141 | -10.2999 [-12.3713, -8.3307] B | -0.8490 [-1.4105, -0.3742] B | -9.9249 [-11.9750, -7.9686] B |
| crawl | M1 crawl share | 2106/135 | +0.0565 [+0.0347, +0.0829] W | -0.0039 [-0.0088, +0.0011] . | +0.0604 [+0.0391, +0.0864] W |

### Table C. SOUP s0: all 19 rows, X (arm + S) against P10 and against Y (the same checkpoint, no S)

| family | row | n | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 | 2399/141 | -1.1928 [-1.4078, -0.9880] B | -0.0687 [-0.1017, -0.0360] B | -1.1242 [-1.3415, -0.9136] B |
| LONGITUDINAL | speed MAE | 2399/141 | -0.7817 [-0.9919, -0.5956] B | -0.0836 [-0.1229, -0.0456] B | -0.6981 [-0.9162, -0.5103] B |
| LONGITUDINAL | along-track MAE | 2399/141 | -0.6291 [-0.8126, -0.4582] B | -0.0635 [-0.0919, -0.0358] B | -0.5656 [-0.7527, -0.3884] B |
| LONGITUDINAL | accel MAE | 2399/141 | -0.8791 [-1.2581, -0.5583] B | -0.2220 [-0.3324, -0.1237] B | -0.6571 [-1.0175, -0.3428] B |
| LONGITUDINAL | DK-VIOL | 595/70 | +0.0067 [-0.0196, +0.0346] . | +0.0034 [-0.0065, +0.0145] . | +0.0034 [-0.0197, +0.0263] . |
| LONGITUDINAL | DK-HW | 707/72 | -0.2040 [-0.5885, +0.1367] . | -0.0365 [-0.1833, +0.0805] . | -0.2220 [-0.6084, +0.1263] . |
| LONGITUDINAL | DK-TG | 650/72 | -0.0560 [-0.1173, -0.0018] B | -0.0200 [-0.0623, +0.0255] . | -0.0231 [-0.1047, +0.0820] . |
| LATERAL | heading MAE | 2372/141 | -2.9613 [-3.7496, -2.1786] B | -1.1463 [-1.6053, -0.7476] B | -1.8173 [-2.5117, -1.1031] B |
| LATERAL | yaw-rate MAE | 2364/141 | +1.9311 [-1.0565, +5.0084] . | -6.4174 [-8.3852, -4.7241] B | +8.3376 [+4.8170, +12.1092] W |
| LATERAL | curvature MAE | 2364/141 | +0.0919 [+0.0584, +0.1301] W | -0.0354 [-0.0538, -0.0194] B | +0.1273 [+0.0878, +0.1730] W |
| LATERAL | cross-track MAE | 2399/141 | -1.0265 [-1.1729, -0.8943] B | -0.0313 [-0.0574, -0.0069] B | -0.9952 [-1.1413, -0.8629] B |
| LATERAL | L1 lateral-accel MAE | 2364/141 | -2.6418 [-4.0007, -1.5149] B | -0.5906 [-0.8591, -0.3451] B | -2.0493 [-3.3682, -0.9100] B |
| LATERAL | L2 curvature MAE (GT>=5.2) | 2105/135 | +0.0628 [+0.0279, +0.1076] W | -0.0219 [-0.0372, -0.0071] B | +0.0857 [+0.0458, +0.1362] W |
| TACTICAL | TAC-LAT | 2399/141 | +0.1847 [+0.1417, +0.2274] B | +0.0067 [-0.0050, +0.0175] . | +0.1780 [+0.1363, +0.2198] B |
| TACTICAL | TAC-LON | 2399/141 | +0.1467 [+0.0792, +0.2138] B | +0.0358 [+0.0175, +0.0558] B | +0.1109 [+0.0434, +0.1777] B |
| STRATEGIC | S-POS@30 | 2321/141 | -1.6062 [-2.1526, -1.1337] B | -0.3367 [-0.5276, -0.1760] B | -1.5868 [-2.0893, -1.1647] B |
| STRATEGIC | S-POS@50 | 2313/141 | -4.5825 [-5.8449, -3.4774] B | -0.5547 [-0.8785, -0.2737] B | -4.4654 [-5.5847, -3.4515] B |
| STRATEGIC | S-POS@80 | 2290/141 | -10.1917 [-12.6077, -8.0749] B | -0.8017 [-1.2640, -0.3920] B | -9.8597 [-12.0514, -7.8539] B |
| crawl | M1 crawl share | 2105/135 | +0.0666 [+0.0428, +0.0927] W | -0.0073 [-0.0127, -0.0021] B | +0.0735 [+0.0507, +0.0992] W |

### Table C. SOUP s1: all 19 rows, X (arm + S) against P10 and against Y (the same checkpoint, no S)

| family | row | n | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 | 2399/141 | -1.1895 [-1.3938, -0.9847] B | -0.0558 [-0.0890, -0.0229] B | -1.1337 [-1.3387, -0.9306] B |
| LONGITUDINAL | speed MAE | 2399/141 | -0.7737 [-0.9864, -0.5882] B | -0.1139 [-0.1677, -0.0627] B | -0.6598 [-0.8724, -0.4800] B |
| LONGITUDINAL | along-track MAE | 2399/141 | -0.6260 [-0.8049, -0.4497] B | -0.0214 [-0.0494, +0.0085] . | -0.6046 [-0.7871, -0.4233] B |
| LONGITUDINAL | accel MAE | 2399/141 | -0.7964 [-1.1397, -0.4913] B | -0.4894 [-0.6984, -0.3000] B | -0.3070 [-0.6459, +0.0186] . |
| LONGITUDINAL | DK-VIOL | 595/70 | -0.0067 [-0.0265, +0.0122] . | -0.0034 [-0.0120, +0.0039] . | -0.0034 [-0.0226, +0.0162] . |
| LONGITUDINAL | DK-HW | 710/72 | -0.5481 [-0.9300, -0.2218] B | -0.0792 [-0.2162, +0.0356] . | -0.4586 [-0.8480, -0.1419] B |
| LONGITUDINAL | DK-TG | 652/72 | -0.0863 [-0.1467, -0.0356] B | +0.0050 [-0.0248, +0.0433] . | -0.0912 [-0.1503, -0.0423] B |
| LATERAL | heading MAE | 2373/141 | -2.9475 [-3.6319, -2.3097] B | -0.6376 [-0.9459, -0.3870] B | -2.3099 [-3.0755, -1.5946] B |
| LATERAL | yaw-rate MAE | 2365/141 | +3.0978 [+0.2214, +6.2180] W | -5.6927 [-7.6797, -3.9568] B | +8.8067 [+5.1990, +12.9014] W |
| LATERAL | curvature MAE | 2365/141 | +0.1068 [+0.0667, +0.1533] W | -0.0175 [-0.0356, -0.0002] B | +0.1236 [+0.0808, +0.1721] W |
| LATERAL | cross-track MAE | 2399/141 | -1.0189 [-1.1607, -0.8829] B | -0.0584 [-0.0843, -0.0336] B | -0.9605 [-1.1087, -0.8253] B |
| LATERAL | L1 lateral-accel MAE | 2365/141 | -2.5743 [-3.8473, -1.4601] B | -1.1617 [-1.6670, -0.7008] B | -1.3530 [-2.4506, -0.3671] B |
| LATERAL | L2 curvature MAE (GT>=5.2) | 2106/135 | +0.0454 [+0.0224, +0.0739] W | -0.0138 [-0.0318, -0.0008] B | +0.0590 [+0.0332, +0.0919] W |
| TACTICAL | TAC-LAT | 2399/141 | +0.2138 [+0.1747, +0.2548] B | +0.0108 [-0.0008, +0.0229] . | +0.2030 [+0.1659, +0.2419] B |
| TACTICAL | TAC-LON | 2399/141 | +0.1584 [+0.0975, +0.2172] B | +0.0154 [-0.0092, +0.0392] . | +0.1430 [+0.0830, +0.2028] B |
| STRATEGIC | S-POS@30 | 2321/141 | -1.7972 [-2.2610, -1.3948] B | -0.2302 [-0.3819, -0.1000] B | -1.9417 [-2.4501, -1.5150] B |
| STRATEGIC | S-POS@50 | 2313/141 | -5.0253 [-6.1550, -3.9714] B | -0.3732 [-0.6209, -0.1651] B | -5.0875 [-6.2087, -4.0736] B |
| STRATEGIC | S-POS@80 | 2290/141 | -10.7619 [-12.9488, -8.6791] B | -0.5872 [-0.9837, -0.2610] B | -10.6181 [-12.7240, -8.5841] B |
| crawl | M1 crawl share | 2106/135 | +0.0576 [+0.0364, +0.0823] W | -0.0044 [-0.0089, -0.0002] B | +0.0616 [+0.0404, +0.0866] W |

### Table E. Rows by flag within each family (B / W / not separated), X - P10 and X - Y

| arm | seed | family (rows) | X - P10: B / W / . | X - Y: B / W / . |
|---|---|---|---|---|
| DAG-OACC | s0 | PRIMARY (1) | 1 / 0 / 0 | 1 / 0 / 0 |
| DAG-OACC | s0 | LONGITUDINAL (6) | 4 / 0 / 2 | 3 / 0 / 3 |
| DAG-OACC | s0 | LATERAL (6) | 3 / 2 / 1 | 5 / 0 / 1 |
| DAG-OACC | s0 | TACTICAL (2) | 2 / 0 / 0 | 2 / 0 / 0 |
| DAG-OACC | s0 | STRATEGIC (3) | 3 / 0 / 0 | 3 / 0 / 0 |
| DAG-OACC | s0 | crawl (1) | 0 / 1 / 0 | 1 / 0 / 0 |
| DAG-OACC | s1 | PRIMARY (1) | 1 / 0 / 0 | 1 / 0 / 0 |
| DAG-OACC | s1 | LONGITUDINAL (6) | 5 / 0 / 1 | 3 / 0 / 3 |
| DAG-OACC | s1 | LATERAL (6) | 3 / 3 / 0 | 4 / 0 / 2 |
| DAG-OACC | s1 | TACTICAL (2) | 2 / 0 / 0 | 1 / 0 / 1 |
| DAG-OACC | s1 | STRATEGIC (3) | 3 / 0 / 0 | 3 / 0 / 0 |
| DAG-OACC | s1 | crawl (1) | 0 / 1 / 0 | 0 / 0 / 1 |
| SOUP | s0 | PRIMARY (1) | 1 / 0 / 0 | 1 / 0 / 0 |
| SOUP | s0 | LONGITUDINAL (6) | 4 / 0 / 2 | 3 / 0 / 3 |
| SOUP | s0 | LATERAL (6) | 3 / 2 / 1 | 6 / 0 / 0 |
| SOUP | s0 | TACTICAL (2) | 2 / 0 / 0 | 1 / 0 / 1 |
| SOUP | s0 | STRATEGIC (3) | 3 / 0 / 0 | 3 / 0 / 0 |
| SOUP | s0 | crawl (1) | 0 / 1 / 0 | 1 / 0 / 0 |
| SOUP | s1 | PRIMARY (1) | 1 / 0 / 0 | 1 / 0 / 0 |
| SOUP | s1 | LONGITUDINAL (6) | 5 / 0 / 1 | 2 / 0 / 4 |
| SOUP | s1 | LATERAL (6) | 3 / 3 / 0 | 6 / 0 / 0 |
| SOUP | s1 | TACTICAL (2) | 2 / 0 / 0 | 0 / 0 / 2 |
| SOUP | s1 | STRATEGIC (3) | 3 / 0 / 0 | 3 / 0 / 0 |
| SOUP | s1 | crawl (1) | 0 / 1 / 0 | 1 / 0 / 0 |

### Table D. Thread-noise replicate: the largest change of any row delta between the card's rollouts (1 thread) and the same four rollouts at 2 threads

| arm / seed | max abs change, X - P10 rows | rows whose flag or sign changed | max abs change, X - Y rows | rows whose flag or sign changed |
|---|---|---|---|---|
| DAG-OACC s0 | 4.98e-05 | 0 | 5.20e-05 | 0 |
| DAG-OACC s1 | 5.91e-05 | 0 | 5.91e-05 | 0 |
| SOUP s0 | 4.66e-04 | 0 | 4.66e-04 | 0 |
| SOUP s1 | 3.58e-05 | 0 | 1.87e-05 | 0 |

| arm | outcome, card rollouts (1 thread) | outcome, replicate (2 threads) |
|---|---|---|
| DAG-OACC | NO EFFECT | NO EFFECT |
| SOUP | MOVED | MOVED |
