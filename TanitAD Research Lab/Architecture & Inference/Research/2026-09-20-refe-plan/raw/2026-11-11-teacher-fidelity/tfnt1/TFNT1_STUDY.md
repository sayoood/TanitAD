# TF-NT1 study: N7-T3 on the navtrain labels of the REFe student (design, feasibility, cost)

REFe agent for the MM, 2026-11-15. Study only; nothing was run on the full job, no git write, no navtest token. Code read from repo mirror commit `f6cb5c80`
(`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`, `TF` = `PK/raw/2026-11-11-teacher-fidelity`).
Evidence classes: M = MEASURED (artifact named), P = PUBLISHED, I = INHERITED (not re-verified), E = ESTIMATED, H = HYPOTHESIS.
Line cites are lines of the files in the repo at `f6cb5c80` (n7 chain copy `TF/n7/chain/`, A1 builder `TF/n4/a1/`, N4 repair `TF/n4/chain/n4_repair.py`).

## 0. Headline
- N7-T3 is not a model; it is a **label-editing cascade** on the 20-pose teacher label of tokens whose tracked builder DAC is 0 (navtest: 20 of 12,146 = 0.165 %, 18 fixed). On navtrain the label is the **P5-R label of the RJ5 roll**; the cascade edits ~0.5 % of rows. M `TF/n7/run/RESULT_N7.md`.
- It **can** run on navtrain labels with the existing code: `n7_nt_build.py derive --route navtrain` already ran on the 17 navtrain residual tokens (M `TF/n7/chain/smoke/SMOKE_N7.md`, mean 102 s/residual token with every lever run twice).
- Population = the **rank-0 rows of the P7 bank (92,896)**, not 150,603: RJ5 exists only for rank 0 (`TF/rj5_navtrain/RJ5_NAVTRAIN_PLAN.md` s1, `PREREG_P5C.md` s4.3 rank0 policy). Rank-1 stays identity.
- Cost is small (**~16-31 CPU-h** on the cheap cache route, E) but it needs one thing the dev box lacks: the **navtrain scene** per token. Cheap route = pod D9 parts cache; dev-box route needs a re-fetch of 877 DBs (~145 GB inflated, M).
- The cascade's student-level value is **small** (it touches ~0.5 % of rows, ~0.1 PDMS at teacher level on navtest). It is a label-completeness result, not a P5-C lever on its own (s5, s6).

## 1. Stage map (Q1)
"builder" = the TanitAD tracked scorer on the scene (`n4_geom.Ctx.score`, NAVSIM PDM components with the LQR tracker). It is **not** the pinned harness. On navtest the harness was run on the final plans; on navtrain **no harness exists** (no navtrain NAVSIM metric cache, s2), so NT1 numbers are builder-tier, like `REPAIR_NT.md` (M `TF/rj5_navtrain/repair_nt/REPAIR_NT.md`).

| # | stage | computes | per-token inputs | code path | scorer |
|---|---|---|---|---|---|
| S0 | **RJ5 roll** | the DriveRL teacher rolled with positive-jerk limit 5.0 (registered 1.0), 4 s, 20 poses at 5 Hz; row = traj, traj_world, goal, origin_world, knobs, bank_dmax | nuPlan DB scenario (log+token), nuPlan maps, DriveRL checkpoint | `TF/rj5_navtrain/rj5_roll.py:27-34` (ARM_JERK / overrides), `:99` main; chain `rj5_chain.py:80` (fetch batch, roll, delete DBs) | none (rollout only) |
| S1 | **Tc20s then P5-R (lambda 3.0)** | tracker-aware label R: cubic Tc20s iteration `iterate20` then the regularised inversion; R is the 20x3 label the student imitates | the RJ5 `traj` (t20), **DB initial ego** (never the cache ego: A1 rule), tracker | `TF/n4/a1/n4_geom.py:84-90` (`Ctx.labels`), `PK/raw/2026-11-11-p5c-readiness/a1/p5c_dual_labels.py:88-112,113` (`_p5r_row`, main; DBs via `P5C_DBROOT`), `rj5_label.py:51-79` (merge, key) | tracker in loop, no scene |
| S2 | **Builder scan** | tracked builder score (DAC, NC, DDC, TTC, C, PDMS) of every label; DAC = 0 set is the residual R0 | scene parts (traj, ego, obs, centerline, lanes, drivable map) = `PT.metric_cache_parts(scenario)` or D9 cache, the label | `n4_geom.py:45-67` (`Ctx`, `from_parts`), `:92-95` (`score`); `n4_nt_build.py:207-222` (`ctx_stream`: cache route = `from_parts`, navtrain route = DB rebuild), `n7_nt_build.py:431-500` (derive loop: `ref = cx.score([plan5])`, `resid = ref["dac"]==0`) | builder, **metric-cache parts or DB rebuild** |
| S3 | **REPAIR-D** | shift the TARGET t20 along the scorer's return vector (margin 0.05, <= 5 iterations), re-run S1 on the shifted target; first certified candidate | scene parts, t20, DB ego | `TF/n4/chain/n4_repair.py:113` (`repair_target`), cert `:25-27` (DAC=1, NC/DDC/TTC/C not worse) | builder |
| S4 | **L1** | two-sided lateral repair of the target (LP over the corridor), same certificate | same + drivable corridor | `TF/n5_l1/n4_repair_l1.py` (`repair_l1`), `TF/n5_l1/L1_DEV.md` | builder |
| S5 | **Wider generator, tiers T0-T3** | T0 = N6 (S3/S4 iterates, cap 5); T1 = cap 20, margins 0.05/0.10/0.20; T2 = time-stretch s 0.95-0.80 of label/target; T3 = lateral offsets of the label dy +-0.05..1.0 m, ramp 0.2/1/2 s (plus stretch 0.9/0.8 + offsets). T4 (s 0.7-0.5) is **excluded** (secondary arm only) | scene parts, t20, label, ref score | `TF/n7/chain/n7_lib.py:54-118` (`time_stretch`, `lateral_offset`, `run_lever`), `n7_select.py:28-61` (`describe`, `cascade`), `n7_nt_build.py:54-81,236-277` (card tables), `:279-332` (`gen_levers`, `token_row`) | builder |
| S6 | **Tolerance certificate + selection** | first tier with a certified candidate; max builder PDMS; ties earliest. Certificate: DAC = 1, NC and DDC not worse, builder PDMS strictly better, **TTC and C free** | the N5-A-equivalent builder score (`ref`) | `n7_nt_build.py:106-155` (`card_cascade`, `make_selector`), `:236-262` (certificate readback), `n7_lib.py:18-35` (`cert_clauses`) | builder |
| S7 | harness read (navtest only) | PDMS 96.499 / DAC 99.9835 on 12,110 tokens | metric cache, LQR harness | `n7_nt_chain.py` (score_T3_T) | **harness, navtest only: not part of NT1** |

N5-A on navtest = N2-R (RJ5 + P5-R) + REPAIR-D + L1 (`TF/n5_l1/a1/run/RESULT_N5_A1.md`). So the N7-T3 driver on navtrain is S0 to S6 in that order. Navtest cache route inputs: harness metric cache + DB ego (`db_ego_fn`, `n4_nt_build.py:192-204`), the DB only for label-building tokens. N7 navtest build used no DB scoring.

## 2. Coverage on navtrain (Q2)
**Populations**
- (a) P7 bank: 150,603 tuples = **92,896 rank-0** + 57,707 rank-1 twins (augmentation, `PREREG_P5C.md` s4.3; I from `RJ5_NAVTRAIN_PLAN.md` s1). The driver reaches rank 0 only; rank-1 rows keep the identity label.
- (b) RJ5 bulk rows (M `F:/tanitad_scratch/rj5nt/run_*/rows/rj5_*.jsonl`, arm RJ5, distinct log/token, counted 2026-11-15):

| set | logs | RJ5 rows | state |
|---|---|---|---|
| local roll | 198 | 16,133 | done, **P5-R labelled** (`label_local`, 0 fallbacks), REPAIR-D run (`REPAIR_NT.md`) |
| chain A | 439 | 24,191 | rolled (`ZZRJ5_CHAIN_DONE`), DBs deleted, **no labels, nothing scored** |
| chain B | 246 | 38,623 | rolled (done), same |
| chain C | 192 | ~13,949 (E: 92,896 - 78,947) | NOT rolled: sequenced behind W-A7 (`seq_wa7_c.log`: "waiting for W-A7: free virt 3992 MB < 5120 MB", not started) |

Merge target: 92,896 bank keys (M `rj5_merge_report.json`, local-roll hold). Rows total ~250 MB (M `du`), cheap to ship.
- Exposure to disclose: 189 local RJ5 rows sit on bank rows with `teacher_device: None` (M `RJ5_LOCAL_ROLL.md`); R0 identity read only on `cpu` rows.

**What is missing per stage**
- S0: chain C (192 logs, ~13.9k rows, ~11 CPU-h at 2.9 CPU-s/row, M: chain A roll 37,568 s x 2 procs / 25,815 rows).
- S1: labels exist only for the 16,133 local rows. **76,763 rows (A + B + C) have no P5-R label.** They need the DB initial ego.
- S2-S6: nothing exists for the 76,763; for the 198 local logs S2-S3 were done (builder DAC = 0: **90 = 0.56 %**, REPAIR-D repaired 73, **17 residual on 7 logs**, M `REPAIR_NT.md`) and the cascade was run on the 17 (13 certified in T0-T3, 16 with T4, M `TF/n7/gen/DEV_N7.md`, `gen/raw/sel17_*.json`). So **17.4 % of the population needs no further compute**, only application. Caveat: those 17 set the cascade (in-sample).

**Scene source (two locations probed before calling it absent)**
1. Dev box: **no navtrain NAVSIM metric cache.** MEASURED probes now: `E:/Projects` and `F:/tanitad_scratch`, `F:/Archive`, `F:/q` for `*metric_cache*` give only `F:/Archive/devbox-C/navsim/exp/metric_cache_navhard_two_stage` and `_warmup_two_stage`; W-A7's own probe (`E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7/INPUTS.md`, M) adds `metric_cache_navtest`, `_smoke20`, no navtrain. W-A7 rebuilds scenery from DBs via `navsim_pdm_targets.metric_cache_parts`, not from a cache.
2. Pod: the **D9 parts cache** (`refe/build_metric_cache.py`, `<log>/<token>.pkl.xz`, parts = `(traj, ego, obs, centerline, lanes, dam)`). Its docstring says the parts are exactly `PT.metric_cache_parts(scenario)` and its `verify` asserts score(cached) == score(fresh) bit-exactly. That is the same object the builder uses on the navtrain route (`n4_geom.py:53`) and the cache route (`from_parts`). The online labeller reads it for every P7/P5-C training token (`train.py:1230-1231`, `--metric-cache`), so it covers the rank-0 bank (I; coverage and path on the pod are **not verified here**; control C-CACHECOV).
- ⇒ the cheap route is **`n7_nt_build.py derive --route cache --metric-cache-root <D9>`** with `Ctx.from_parts(parts, label_ego_fn=db_ego_fn)` (`n4_nt_build.py:207-222`). The pod holds the DBs (`DBROOT=/workspace/data/navtrain_dbs`, `p5c_pod_label.sh`, I) for the label ego. Whether the D9 `ego` equals the DB label ego (it is `scenario.initial_ego_state` on the DB route, so H: yes) is control C-DERIV.

**DB re-fetch volume if the dev box does it (M, chain logs)**
- Chain A: 439 logs, 56.4 GB inflated, fetch wall summed 29,855 s (**includes inter-batch waits**, so an upper bound, mean 136 s per 2-log batch).
- Chain B: 246 logs, 76.9 GB, 55,886 s summed (upper bound; waited on the F: floor).
- Chain C: 192 logs, ~12 GB (E: 145.5 GB total from `RJ5_BULK_FETCH.md` 135.55 GiB, minus A and B).
- Total ~145 GB; at A's rate ~9 s per log-GB it is ~9-24 h wall, serial with the DB scan. Not recommended; F: has 12 GB free and E: 5.8 GB (M `df`, now).

## 3. Cost (Q3), per token, MEASURED unless marked
| stage | per token | source |
|---|---|---|
| S0 roll | 2.9 CPU-s/row (A: 37,568 s x 2 / 25,815; B 2.88) | chain logs `run_fetch_A/B/chain.log` |
| S1 P5-R label | 0.52 s/row (P5-R EVAL, 1 proc, `RESULT_P5R.md`); **0.70 CPU-s/row** on the dev box (16,133 rows in 94 min x 2 shards, `label_local/label.log`) | M |
| S2 scan, cache route | mean 0.25 s/token all-in for the whole N7 derive (12,146 tokens, 0.843 CPU-h incl. 20 cascades x 2 runs, `n7_build_report.json timing`); pure scoring is the remainder, ~0.08 (E, subtraction) | M / E |
| S2 scan, DB route | ~1.1-1.4 s/token (REPAIR_NT: 16,133 tokens in 13:38-16:42 x 2 shards, minus ~90 x 3 arms x 17 s repairs) | M-derived |
| S3 REPAIR-D | 12-21 s per DAC-0 token (M `CHAIN_N4.md` s7 smoke: 12.5-21.2 s) | M |
| S4/S5/S6 cascade | mean **102 s, max 159 s** per residual token, every lever run twice + live N6 capture (M `SMOKE_N7.md`); once only ~51 s (E) | M |

**Totals for the 92,896 rank-0 rows (E, from the rates above)**
| population / step | CPU-h |
|---|---|
| S0 chain C | ~11 |
| S1 labels, 76,763 rows | 11-15 |
| S2 scan, all rows, cache route | 2-6.5 (DB route: 28-35) |
| S3 REPAIR-D on ~520 DAC-0 (0.56 % M; 890 if navtest's 0.96 %) | 1.7-5.2 |
| S4-S6 cascade on ~100-170 residual (E) | 1.4-4.8 |
| **NT1 without S0 (pod cache route)** | **~16-31** |
| dev-box-only equivalent (DB route, 2 procs) | ~55-65 CPU-h = ~28-32 h wall + the ~145 GB fetch |

**Placement**
- **Dev box (i9-12900F):** free virtual memory is 2,944 MB now (M, 2026-11-15); the N7 chain gated each worker at > 10 GB and the N7 families stage hit `MemoryError` under load (M `RESULT_N7.md` Disclosures). It can take only S0 chain C (already sequenced) and the tiny steps (table gates, re-key, build). It cannot carry S1-S6 for 77k rows alongside W-A7.
- **Box 2:** reserved for flagship v8, not used (PI 2026-11-13).
- **Pod:** P7 trains until ~06:15Z 2026-11-16, no load before. After P7, the P5-C label job (`p5c_pod_label.sh`, 16 shards, ~20 min on 64 of 128 cores, `RESULT_P5R.md` cost line) owns the CPUs; then C1/C0 train with 13 online-label workers + loaders. **NT1 fits only in the window between P7's last step and P5-C's launch**, at <= 32 cores, ~0.5-1 h wall (E). It competes with the P5-C label job for cores (run after it, same window) and for the DB tree. It must not run during C1/C0 training. If the window is not free, the next best is a 16-core slot after C1/C0 end.
- Ship to the pod: RJ5 rows (~250 MB), cascade code (`n7_*`, `n4_*`, `n5_*`, small), no DBs (pod has them, I).

## 4. Cheapest-first staging (Q4)
1. **Reuse first (0 compute):** 198 local logs, 16,133 rows: labels, DAC-0 list (90), REPAIR-D results (73) and the cascade picks on the 17 exist (`rj5nt/label_local`, HF `evidence/2026-11-13-rj5-navtrain-labels-local`, `derive_rows.tar.xz`, `sel17_*`). Recompute on the pod only as the identity control (C-LOCAL).
2. **S1 labels for A+B+C rows (11-15 CPU-h)**; these are needed by P5-C-RJ5 anyway, so this is shared cost, not NT1's.
3. **S2 scan on the cache route** (2-6.5 CPU-h): DAC (and NC, DDC) first. Only DAC-0 tokens (~0.56 %) go on. No generator on the other 99.4 %.
4. **S3 REPAIR-D, S4 L1, then S5-S6 cascade on what remains.** The cascade is first-certified-tier, so most tokens end at T0/T1 and the expensive T3 grid (126 candidates) runs only when T1/T2 fail.
5. Residual size (E, label **ESTIMATED**):
   - zero-DAC after P5-R: 0.56 % (M, 198 logs; navtest N2-R 0.96 %) = ~520 rows (range 520-890);
   - after REPAIR-D (73 of 90 = 81 %, M): ~0.105 % = ~98 rows (range 98-170);
   - after the cascade (13 of 17 = 76 % on the in-sample dev set; navtest 18 of 20 = 90 %, navtest-seen): ~**0.025 %, ~23 rows** (range 10-40). Navtest end state was 2 of 12,110 = 0.017 % (M).
   - The local 198 logs are not a random sample (dev-box logs); every rate above carries that.
6. A DB-free pre-screen on the map only (RJ5 `traj_world` + nuPlan maps, no DB) could shortlist logs, but untracked DAC misses tracker-induced exits, so it is a **filter that can only add false negatives**; not recommended as the gate.

## 5. How the output reaches the student (Q5)
- **Format:** `refe/traj_label_table.py build --labels <jsonl> --bank-dir <P7 bank> --out T.json.gz --rank-policy rank0 --meta-json <meta>` (flag and meta per A2, `PREREG_P5C_A2.md` s44-45; the hold copy of the module is the pre-A2 one). One [20,3] label per row, key `log|token|rank|sha256(traj)`.
- ⚠️ **Key mismatch (found while reading, M from `traj_label_table.py:104-116` and `rj5_label.py:51-79`):** `rj5_label.py` keys labels by the sha of the **RJ5** traj. `Table.lookup` hashes the **bank row's own** (registered-jerk) traj and raises `TableRefused` when the triple exists under another sha. The ~85 % of rows RJ5 moved (M, 13,718 of 16,133) would be refused. Fix: re-key to the bank traj sha with the RJ5 sha kept in meta, or ship an RJ5 bank (new `targets_sha256`, new pins). MM decision (O3).
- **Two ways in:**
  1. **P5-C successor fine-tune (recommended):** the table's edited labels join the RJ5+P5-R table; arms as `PREREG_P5C.md` s3 with `--traj-label-table`; rank-1 rows identity.
  2. A full retrain on the bank + bulk: only after (1) reads, per `PREREG_P5C.md` s12.
- **Tracker-aware check before any student sees an edited label (per edited row, deterministic, no statistics needed because the set is ~100-600 rows):**
  - G-T1 (certificate): builder tracked DAC = 1, NC and DDC not worse than the unedited P5-R label.
  - G-T2 (re-simulation): the tracker's 8-knot re-simulation of the edited label is bit-identical on re-run (P5-R S-gate: max dev 0.0, `RESULT_P5R.md`).
  - G-T3 (TB-6 per row): max |acc| and max |heading rate| of the edited label within 2x of its raw `traj` (the TB-6 bar applied row-wise; the table-level TB-6 is unmoved by <1 % of rows).
  - G-T4 (edit size): lateral displacement of the edited label from the unedited label <= 1.0 m, with T2/T3 rows listed by tier. Rows that fail G-T2..G-T4 revert to the unedited label (`source: p5r_unrepaired`), counted.
  - G-T5 (transparent twin): the same row scored on the **transparent** (untracked) path (`Ctx.transparent`, `n4_geom.py:97-99`); report tracked and transparent PDMS per tier and the tracker dependence TD. T3 edits move the label itself (not the target), so their transparent path is shifted by up to the offset: they are expected **tracker-mediated**; say so.
- **PI conditions (kept verbatim in the card):** every number printed tracked and transparent; wording "survives the benchmark's tracker"; labels may use privileged data (map, DB ego, agents), the student's inference stays vision-only; strategic family PAUSED, not dropped.

## 6. Honest value of the edit (what it can and cannot show)
- The edited rows are ~0.56 % of rank-0 rows, ~0.34 % of the 150,603 tuples. The cascade's own teacher-level gain on navtest was +0.109 PDMS over N6-T and +0.132 over N5-A (M `RESULT_N7.md`); at the student level, E: <= 0.1 PDMS. The P5-C floor is **B = 0.7031** (`PREREG_P5C.md` s6.1). A student arm that differs from the RJ5+P5-R arm **only** by these rows cannot be separated by that rig (H, by arithmetic on E). So NT1 should be a **label-side** deliverable with deterministic bars (card), and its table rides inside the P5-C-RJ5 successor; it must not buy a separate 28 L40-h arm.
- The larger student lever is the tracker-aware label set itself (the label's teacher-level gain is +7.64 PDMS, `PREREG_P5C.md` s9, M), and RJ5 on top of it; not the 0.5 % cascade edit.
- Caveat that travels: the builder tier uses the nuPlan-DB rebuild; on navtest the DB rebuild and the harness cache agreed on ego, centerline and drivable map and differed 1-15 cm on object positions (`n4_geom.py` docstring, M). NC can differ at the margin; the harness is not available on navtrain, so NC is reported as builder-tier.

## 7. Smoke (Q7): SKIPPED
- `F:/tanitad_scratch/rj5nt/seq_wa7_c.log` last line: "waiting for W-A7: free virt 3992 MB < 5120 MB"; no "W-A7 chain returned". W-A7 has not started.
- Free virtual memory read now: **2,944 MB** (< 5 GB).
- Per-token times above are therefore from the banked N7 / REPAIR-NT / RJ5 logs, not from a new run.

## 8. Open decisions for the MM
1. **O1 Placement:** pod window after P7 and the P5-C label job, <= 32 cores (recommended), vs dev-box-only (~30 h + 145 GB fetch, not recommended).
2. **O2 Cache route validity:** approve a 200-token C-DERIV on the pod (label from D9-parts ego == banked P5-R labels, bar 0.0) before using the cache route; plus D9 key coverage of 92,896 rank-0 keys.
3. **O3 Key convention:** re-key RJ5 labels to the bank traj sha (recommended) or build an RJ5 bank.
4. **O4 Chain C:** wait for W-A7 and roll 192 logs (~11 CPU-h) or build NT1 on 78,947 rows with `require_coverage` ~0.85 and the rest identity (a narrower claim).
5. **O5 Student read:** ride the P5-C-RJ5 successor, no separate arm (recommended; power arithmetic above).
6. **O6 T4:** excluded here (PI's open T4 question does not gate this); N7-T's T4 token is a secondary arm only.
7. **O7 Ledger:** label-side, deterministic, no navtest: no K entry proposed; the student read carries its own.
