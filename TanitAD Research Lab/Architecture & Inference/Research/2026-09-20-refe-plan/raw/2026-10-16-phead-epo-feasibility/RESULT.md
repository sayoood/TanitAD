# PHEAD: a head-only EPO keeps 4-16 % of the full arm's J gain at matched lr, and pays TTC for it -- row R3 (CANNOT CARRY)

Research Lab agent (Sonnet), 2026-10-16. CPU only (0 A40, 0 GPU). Pre-registration: `PREREG_PHEAD.md`, sha256 `1e8b741ed13b792e1ce0ce83640512595ac5224ff116f3e50419ca36f2395b96` (re-verified unchanged at the end; written before any restricted-arm number except the one disclosure in section 8).
Source question: `eval/PREREG_R0C.md` section 4, row O3, next lever; `raw/2026-10-16-r0c-read/RESULT.md`. Evidence class of every sim number: **ESTIMATED** (real scorer weights / labels / trajectories, substituted visual-context cache V = 0.997, CPU fp32; offline selection read, not T1, not a driving claim).

## 1. Headline and outcome row

- **Row R3, CANNOT CARRY**, on the committed rule (decided on f_min = min(f_train, f_val), closed loop, regime P, K = 50, lam 1, matched lr 1e-4).
- No candidate reaches 0.2: **PH_A (EP row, 257 params) 0.037, PH_B (six PDM rows, 1,542) 0.107, PH_C (whole `score_head`, 1,799) 0.164**. PH_C is the best candidate; its two pools read 0.164 (train) and 0.211 (val), so it sits on the 0.2 edge and its val interval is wide ([0.09, 0.56]). The row is not moved (prereg s.8). Three data orders read PH_C f_min 0.164 / 0.156 / 0.164, so the data order does not move it.
- **The TTC proxy does not rescue the restriction**: per unit of J gained the head-only arms LOSE TTC (P2, x100 per unit J: PH_C -31, PH_B -40, PH_A -94) while the full arm does not (+4 at K = 50). Raising the head's lr x4 / x16 lifts f_min to 0.37 / 0.41 and takes the pooled TTC proxy to -0.43 / -0.75 (g5 bound -0.5).
- **Where the gain lives**: head + all four scoring-decoder blocks (LAD_2) carries f_min 0.90; head + the last block only (LAD_1) 0.22 with TTC proxy -0.46 pooled. LAD_2 is 98 % of the parameters, so it is not a restriction.
- **Recommendation (a PI decision, not made here): do NOT spend an A40 arm on head-only EPO.** Next lever in section 7.

## 2. "Progress heads" and "frozen trunk", from source

- The scorer is `score_q_mlp` (trajectory encoder, `model.py:560-562`) -> `score_dec` (4 CrossBlocks, `model.py:551-552`) -> `score_head = Linear(256, 7)` (`model.py:563`; `scorer_finetune.py:75`). The backbone, the proposals and the visual context are frozen / detached in EVERY arm (`scorer_finetune.py:5-6`, `:621-626`).
- There is **no separate progress module**. EP is **row 2 of `score_head`** (`train.py:502`, rows nc, dac, ep, ttc, c, ddc, then the lane row 6; `scorer_finetune.py:80-85`). The question is ambiguous, so the three candidates were each run as an arm (table in `PREREG_PHEAD.md` s.2, parameter counts in `groups.json`).
- "Frozen trunk" = `score_q_mlp` + `score_dec` get no gradient and are not in AdamW or the clip (kind R); the restricted optimiser hands only the group's entries to AdamW and `clip_grad_norm_` (`scorer_finetune.py:1360-1362`, `:1391-1395`), so weight decay never touches a frozen entry. Arm A stays the full BCE-only control.
- R0's EPO displacement sits 99.1 / 99.3 % in `score_dec` and 0.02 % in `score_head` at u250 / u500 (`disp_real.json`, MEASURED); that is why the head is a different regime.

## 3. Identity controls (all passed before any restricted number was reported, except as disclosed in section 8)

| control | result | tolerance |
|---|---|---|
| (a1) FULL through `Restricted` vs `adam_sim.run`, bank stream 0, P, lam 0 / .25 / 1, K = 50 | max abs diff **0.0**; `S_C80`, `S_all`, `cos_C80`, `U`, `cos_full` equal `adam_sim.json` with rel diff **0.0** | 1e-7 / 1e-6 |
| (a2) J(theta_0) on the 256 sets of `groups[:32]` vs `late_probe.json` base.mean_J 0.8391432501 | 0.8391432439, abs diff **6.2e-9** | 1e-5 |
| (a3) closed loop FULL lam 1, K = 25 vs banked `closed_P_1.0.pt` D[25] | rel L2 **1.9e-4** (max abs 5.5e-5; thread count differs) | 1e-3 |
| (b) EMPTY group, open loop P and T (K = 50) and closed loop K = 3 | `theta - theta_0` max **0.0**, Delta J **0.0** (`==`), all pools. Sensitivity: a mask-the-gradient AdamW moves frozen entries by 7.7e-5 in 50 steps (the bug this check catches) | exact |
| (c) mutation: delete the restriction line (`PHEAD_MUTATE_NO_RESTRICT=1`) | outside-group displacement 0.0 unmutated vs **5.1e-3 mutated (RED)**; open-loop PH_C Delta J 0.0042 -> 0.0362 (train), 0.0042 -> 0.0541 (val). Closed loop K = 3: 3.0e-4 (RED) | RED required |
| (d) TTC proxy reads the known R0 trade (banked scorer states, 364 sets) | P2 TTC B - A x100: **-1.53 / -1.25 / -1.07** at u250 / u500 / u1000 (measured RL-val -1.23 / -0.63 / -1.00); C0 floor +0.0016 / +0.0019 / +0.0001. P1 pick: -2.2 / -2.5 / -2.5. **Proxy validated.** | >= 2 of 3 negative, C0 < 0.5 |
| (e) not in the prereg: the cached-feature head path vs the real pipeline | PH_C lam 0 / 1 and PH_B lam 1: max abs Delta J per set **<= 6.6e-7**, identical argmax picks, equal displacement norms (`cached_vs_real.json`) | informational |

## 4. Results: the closed-loop arms, regime P, K = 50, lam 1 (MEASURED in the sim, class ESTIMATED; estimator: paired log-cluster bootstrap, 10,000, seed 20260927, over the pool's logs; train 302 sets / 30 logs, in-sample; val 62 sets / 13 logs, out-of-sample)

Full arm (banked `closed_P_{0.0,1.0}.pt`): Delta J_FULL **train +0.0407 [0.0258, 0.0557]**, **val +0.0333 [0.0064, 0.0558]**. Context: R0's measured J(B) - J(A) at u500 is +0.0315 (`late_probe.json`, 256 train sets), the same scale.

| arm (kind R unless noted) | params | f train [CI] | f val [CI] | **f_min** | TTC proxy B - ctrl x100: P1 pick / P2 policy (pooled) | row |
|---|---|---|---|---|---|---|
| PH_A_EPROW | 257 | 0.042 [0.025, 0.071] | 0.037 [0.006, 0.165] | **0.037** | -0.82 / -0.15 | R3 |
| PH_B_SIX | 1,542 | 0.107 [0.064, 0.149] | 0.139 [0.059, 0.385] | **0.107** | 0.00 / -0.17 | R3 |
| **PH_C_HEAD** | 1,799 | 0.164 [0.112, 0.216] | 0.211 [0.092, 0.562] | **0.164** | -0.27 / -0.21 (TTC condition holds) | **R3 (f < 0.2)** |
| PH_C kind E (EPO masked to the head, BCE moves all) | 1,799 EPO | 0.051 [-0.040, 0.146] | 0.273 [0.096, 0.743] | 0.051 | 0.00 / -0.15 | R3 |
| *LAD_1 rung (head + last block)* | 1,055,239 | 0.421 [0.308, 0.522] | 0.222 [-0.518, 0.541] | 0.222 | -0.27 / -0.46 (val P2 -1.15) | TTC worse |
| *LAD_2 rung (head + 4 blocks)* | 4,215,559 | 0.900 [0.841, 0.966] | 1.024 [0.890, 1.455] | 0.900 | +0.82 / -0.17 | R1-like, not a restriction |
| PH_C lr x4 | 1,799 | 0.373 | 0.418 | 0.373 | -0.55 / -0.43 | TTC worse |
| PH_C lr x16 | 1,799 | 0.482 | 0.406 | 0.406 | -1.37 / -0.75 | TTC worse |
| PH_A lr x4 / x16 | 257 | 0.066 / 0.098 | 0.050 / 0.063 | 0.050 / 0.063 | -0.55 / -0.25, -0.55 / -0.32 | R3 |
| FULL reference | 4,296,967 | 1 | 1 | 1 | +0.27 / +0.16 (NC P2 +0.32) | |

- **Data-order sensitivity (closed loop, cached path, three bank streams, f against the stream-0 full denominator)**: PH_C Delta J train / val 0.0067 / 0.0070, 0.0063 / 0.0068, 0.0067 / 0.0070 (f_min 0.164 / 0.156 / 0.164); PH_B f_min 0.107 / 0.099 / 0.106. The interval above answers "another draw of evaluation sets"; this answers "another data order". Neither answers "another training run".
- **Open-loop replay (adam_sim style, frozen gradient, K = 50, three streams)**: f_min PH_C 0.078 / 0.101 / 0.105, PH_B 0.054 / 0.072 / 0.073, PH_A 0.021 / 0.020 / 0.028, LAD_1 0.111 / 0.170 / 0.178, LAD_2 1.31 / 1.15 / 1.26. Same bands (R3 for every candidate); lower than closed loop because the frozen-gradient replay cannot let the head's gradient adapt.
- **Regime T, K = 250, open loop**: reported but NOT interpreted. The frozen-gradient replay is outside its validity there: the full arm's KL(base || pi) is 16.0 and LAD_2's val Delta J is negative (f -0.53). A closed-loop T arm for the full model costs about 70 min of CPU per run and was not run.
- **Reference against the full control A (g5 as registered)**: f_A on val is 1.1-1.2 for every head arm, but that is the full BCE-only control LOSING 0.041 J on val in 50 updates (J 0.8126 at theta_0, 0.7715 after BCE-only), not the head carrying the gain. Against A the head arms' TTC proxy is **-1.6 to -1.9** (P2, x100): a restricted B compared with the full A breaches g5 by construction, because A's BCE trunk training raises TTC and a frozen-trunk B cannot. **Any restricted A40 arm would need a matched restricted control A'.**

## 5. Safety terms: what the sim computes and what it cannot (per term)

- **Computed (P1 pick, P2 policy expectation, both pools, x100, B - same-restriction control)**: TTC, NC, DAC in the table above and `summary.json`; pick EP / C / PDMS in `extra_proxies_K50.json`. NC stays within -0.05 for the three head candidates (P2 pooled -0.03 to -0.04); DAC rises (+0.46 PH_C P2). The prereg's TTC condition was applied to TTC; NC would not have flipped any row.
- **P1 resolution**: 1/62 = 1.6 points on val, 0.33 on train, so P1 cannot certify a -0.5 bar by itself; P2 is the continuous reading.
- **Cannot be computed**: PDMS on the 3,000-set RL-val and the full-navtest endpoint (the cache covers 62 RL-val sets); longitudinal speed and headway (n = 0, no instrument in a scorer-only sim); lateral and tactical (n = 0, the scorer selects among fixed proposals); **STRATEGIC: PAUSED** (PI ruling 2026-09-19, item 25), not a pass.
- **What the TTC proxy cannot say**: the sim's FULL arm shows no TTC loss at K = 50 (+0.16), while R0 loses -1.0 by u1000 and the proxy validated on the banked u250-u1000 states reads -1.1 to -1.5. K = 50 is 5 % of R0's updates, so the sim cannot test whether LAD_2 / the full arm eventually pays TTC. It can say that the head-only arms pay TTC EARLY and at a worse rate per unit J (HYPOTHESIS for the mechanism: a head-level move shifts the EP logit for every proposal, including the fast ones TTC punishes, whereas decoder-level moves can be proposal-specific).

## 6. The committed rows, applied

- R1 needs f_min >= 0.5 with the TTC condition: no candidate (best 0.164). R2 needs 0.2 <= f_min < 0.5: no candidate. **R3 (f_min < 0.2) holds for PH_A, PH_B and PH_C.** The best candidate PH_C meets the TTC condition at matched lr (per-pool P1 -0.33 / 0.00, P2 -0.23 / -0.09, all >= -0.5); the row is R3 on f alone.
- The prereg's R3 sub-levers, read in order: **(i)** an lr x {4, 16} arm reaching f_min >= 0.5 with the TTC condition: **not met** (PH_C 0.37 / 0.41 with TTC worse; PH_A 0.05 / 0.06). **(ii)** LAD_1 / LAD_2 reaching it: **LAD_2 met (0.90, TTC condition holds), LAD_1 not met (0.22, TTC condition fails)**. So the gain needs the scoring decoder's first blocks, and step size is not the limit (the lr arms saturate near 0.4-0.5 while their TTC cost keeps growing).

## 7. Next lever (Rule Zero) and what blocks it

1. **Recommended: no A40 arm on head-only EPO** (PI decision). It keeps <= 16 % of the J gain at matched lr and 40 % at a TTC cost that crosses g5 at x16.
2. **The gain lives in `score_dec`, and the TTC trade has to be broken another way**: prereg R3 (ii) -> RL-S1 O2 (ii), a scorer whose safety outputs EPO cannot move (a separate safety head; an architecture change; PI decision). In the sim the head-only restriction is the wrong cut, because it removes the proposal-specific capacity and keeps the coarse EP logit.
3. **Cheapest CPU experiment that could still discriminate, not run (needs about 70 min of CPU per run, over this session's budget)**: closed-loop FULL and LAD_2 in regime T to K = 250 (and a middle rung, head + blocks 2-3), reading the TTC proxy (P2) against the banked-state reference of control (d). It would show whether the proxy goes negative for the decoder-released arms at R0's horizon, which is what any safety-head design must beat. The harness runs it as is: `python phead_closed.py --group LAD_2 --kind R --lam 1.0 --regime T --K 250 --snaps 50 100 250 --out-dir closed_out`, plus the lam 0 pair and the FULL pair.
4. **Blocked, named**: the A40 arm (PI decision; also needs the `scorer_finetune.py` parameter-group change, a matched control A', a launch-gate PASS).

## 8. Disclosures and deviations (nothing here changes a row)

- **Seen before all controls closed**: control (c)'s open-loop output printed PH_C's open-loop Delta J (0.0042 / 0.0042) while (a3) and (d) were still running. The closed-loop numbers were read only after (a1)-(d) had passed.
- **PH_A and PH_B decision numbers come from the cached-feature path** (`phead_closed_cached.py`, the decoder features are theta-independent when the trunk is frozen), because three simultaneous real-pipeline head runs exhausted RAM (free physical memory 1.3 GB, one cached run died with `ArrayMemoryError`; the duplicate real-pipeline PH_A / PH_B runs were killed by PID and their partial outputs deleted). The path was validated against the real pipeline on PH_C lam 0 / 1 and PH_B lam 1 (control e). **PH_C (the deciding candidate) and both rungs and the E arm ran through the real pipeline**; PH_A lam 0 / 1 and PH_B lam 0, and every lr x4 / x16 arm, ran through the cached path only.
- The queue was reorganised twice (xargs dispatchers stopped by explicit PID); the three scripts `run_closed_queue*.sh` are kept as the record. The open-loop scope was cut to lam in {0, 1}, P at K = 50 on three streams and T at K = 250 on stream 0 (the prereg listed lam 0.25 and K = 100 too: **not run**).
- **Kind E ran for PH_C only** (as preregistered); its low f (0.051 / 0.273) is vs the full BCE-only control, whose val J collapses (see section 4).
- Row logic as written: the TTC condition compares P1 and P2 on each pool; NC is reported, not part of the row. A pool-level P1 on 62 sets is coarse (section 5).
- The prereg's "R1-like" label for LAD_2 is a reading only; LAD_2 is a rung, not a candidate, so no outcome row is drawn from it.
- 16:02-17:37 local, 24 logical cores; no GPU touched (the lock `devbox_gpu.lock` was never opened).

## 9. Manifest (nothing staged: the tree is a plain archive of b339e90, not a git repository; the Master Mind lands the package from this path)

Tree PK = `C:/Users/Admin/AppData/Local/Temp/phead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`. Package = `PK/raw/2026-10-16-phead-epo-feasibility/` (716 KB of files + 6 MB of result pickles).

| artifact | where | note |
|---|---|---|
| `PREREG_PHEAD.md`, `.sha256` | package | prereg, hash verified |
| `RESULT.md`, `summary.json`, `streams.json`, `extra_proxies_K50.json`, `proxy_validity.json`, `cached_vs_real.json`, `groups.json`, `probe0_fullarm_baseline.json` | package | results |
| `controls_open.json`, `controls_closed.json` | package | identity controls |
| `closed_out/*.pkl` (real pipeline), `closed_cached/*.pkl` (cached path, streams 0-2, lr arms), `open_out/*.pkl`, `full_ref.pkl` | package | per-set arrays, the inputs of every number |
| `phead_common.py`, `phead_open.py`, `phead_closed.py`, `phead_closed_cached.py`, `phead_controls.py`, `phead_checks_closed.py`, `phead_proxy_validity.py`, `phead_full_ref.py`, `phead_summarize.py`, `phead_streams.py`, `probe0_fullarm_baseline.py`, `run_*.sh` | package | the instrument; reuses `raw/2026-10-15-rls1-adam-dose/` unchanged |
| `logs/` | package | every run log |
| gradient banks `bank_s{0,1,2}.pt` (3.7 GB), pool pickles `phead/pool_*.pkl` | **scratchpad only** (`.../scratchpad/bank`, `.../scratchpad/phead`) | regenerable with `grad_bank.py` / `phead_common.get_pool`; the banks are from the H-ADAM package |

## 9. Addendum (Master Mind, 2026-10-16 ~20:00Z): §7 item 3 RUN. The sim does NOT reproduce R0's TTC loss even at K = 250
- **Run:** closed loop, regime T, stream 0, K = 250, snaps 50 / 100 / 250, groups FULL and LAD_2, each at lam 1 (B) and lam 0 (its own control), on the real pipeline (`closed_out/closed_s0_T_*k250T.pkl`, about 55–93 min CPU each, `logs/k250T_queue.txt`).
  - Read by `phead_k250T_read.py`, which uses `phead_summarize.prox`'s estimator (pooled train+val, log-cluster bootstrap) → `k250T_read.json`.
  - The middle rung (head + blocks 2–3) was not run.
- **MEASURED, P2 TTC (policy expectation), B − control ×100, pooled [95 % CI]:**

| group | K = 50 | K = 100 | K = 250 |
|---|---|---|---|
| FULL | −0.59 [−1.13, −0.10] | +0.12 [−0.57, +0.91] | **+0.23 [−0.50, +1.04]** |
| LAD_2 | −0.38 [−0.79, +0.00] | −0.13 [−0.74, +0.54] | −0.03 [−0.98, +0.95] |

- P1 (pick) TTC, P2 NC and P2 DAC are in the JSON. At K = 250, P2 NC is +0.10 for both groups, and P2 DAC is +0.45 (FULL) / +0.67 (LAD_2).
- **Reading.** The validated proxy (§3 (d)) reads R0's banked states at −1.53 / −1.25 / −1.07 (u250 / u500 / u1000), and RL-val measured −1.23 / −0.63 / −1.00.
  - The sim's FULL arm at the same update count (K = 250) reads **+0.23**, its CI clearly above the −1.5 the real states show.
  - ⇒ **The frozen-feature sim does not carry the mechanism that costs R0 its TTC.** It can rank J-gain restrictions (§4) but cannot test a TTC remedy.
- Any safety-head test therefore needs the real pipeline, i.e. an A40 arm. Stage 1 of that experiment was told the same day.
- Corroboration from outside RL (`eval/RESULT_P2REP.md`, 2026-10-16): P2's supervised seed replicate shows TTC −0.49 [−0.97, −0.07] with EP +1.80. The trade is not specific to EPO.
