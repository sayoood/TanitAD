# RJ5 label path for the REFe student: join, DB-free feasibility, cost (REFe agent, 2026-11-15)

Feasibility and tooling only. No git write, no pod, no box 2, no navtest data. Code read from mirror branch `agent/arch-inf-20260803` (`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`, `TF` = `PK/raw/2026-11-11-teacher-fidelity`).
Evidence classes: M = MEASURED (artifact named), I = INHERITED (not re-verified), E = ESTIMATED, H = HYPOTHESIS. Hold = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rj5path/`.

## Q1. The join (M)
Tool `rj5_bank_join.py` (stdlib, deterministic, 8 tests green: `test_rj5_bank_join.py`). Key = `log|token|0|sha256(bank traj, float64 LE)`, byte-identical to `traj_label_table.traj_sha256` (test against an independent reimplementation). Manifest: `rj5_bank_join_manifest.json` (hold).

| item | value |
|---|---|
| bank rank-0 keys (slim copy) | 92,896 (0 duplicate keys) |
| RJ5 mapped | **78,947 = 84.98 %**: local 16,133 + chain A 24,191 + chain B 38,623 |
| missing (no RJ5 traj) | **13,949 = 15.02 %** = exactly chain C's expected ~13.9k |
| duplicates / conflicts / unmapped RJ5 rows / error rows / torn lines | 0 / 0 / 0 / 0 / 0 (a source that wins a key is the first listed: local, A, B) |
| RJ5 rows that moved > 1e-6 m vs bank | 73,494 of 78,947 = 93.1 % (local alone: 85 %, `RJ5_LOCAL_ROLL.md`; the extra is consistent with the provenance term below) |
| knob check (RJ5 rows jerk 5.0, R0 rows 1.0, from `knobs`) | 0 mismatches |
| joined file | `F:/tanitad_scratch/rj5path_tmp/join_out/rj5_bank_join.jsonl.gz`, 58,741,722 B, sha256 `426e2abc4c404c848dca1a3e9301d95196f65ec14f1815f6b374426cc5aa5170`. One line per bank key: bank_key, both traj shas, status, source, dmax/dxy vs bank, `rj5_traj`. KEPT for the MM (not on E:). |

Slim bank = full dev-box bank copy at key level: per-log key digests over the 1,075 P7 logs are identical (`bank_keys_per_log_digest_slim.tsv`, ALL digest `697b451c...4102d`; M, `rj5_bank_keys_digest.py`). **Pod bank identity is UNVERIFIED** (plan s1: the dev copy's file sha256 `5471b25a` is not the pod's `daa23b96`). Run `rj5_bank_keys_digest.py` on the pod's `targets_rank0.jsonl` and diff against the .tsv before any table build: a mismatching log means every key in it would be REFUSED by `Table.lookup`.

### Control finding that matters (M): the R0 identity control FAILS on the bulk chains
R0 re-roll (registered jerk) vs bank, 5,183 control rows (`rj5_roll.py:199` `bank_dmax`; bar 1e-6 m, plan s4):
- 2,730 exact (key and sha equal bank), **2,452 over the bar, max 0.853 m** (not the 6.4 mm of the local roll).
- Local (198 logs): 1,766/1,766 `cpu` rows exact; 27 `None` rows max 6.4 mm (as `RJ5_LOCAL_ROLL.md` says).
- **Chains A and B: `cpu` rows 861/1,520 (A) and 1,085/1,390 (B) are over the bar.** At log level (`control_by_log_bulk.json`) **28 of 36 control logs differ on EVERY control row**, 8 are exact. So the difference is per log, not per-row noise.
- Cause UNVERIFIED (H: the publicly fetched DBs or maps differ from the ones the pod bank was built from for those logs; or the bank copy differs from the pod's for them; a dev-box BLAS effect is excluded by magnitude, 0.85 m vs the 1e-14..4e-4 class in memory). The pod bank digest above separates the two readings.
- Consequence: for rows in those logs, "RJ5 minus bank" = lever + provenance. The median RJ5 movement is 0.97 m (local), so the provenance term is not negligible on the worst logs. Tables, A3 and any card must carry the per-log control status (A3 TB-8).

## Q2. Can Tc20s -> P5-R be computed for an RJ5 traj WITHOUT the nuPlan DB?
**Dev box: CAN only for the 198 logs whose DBs are on disk (16,133 rows, already labelled), CANNOT for the other 76,763 RJ5 rows (A, B, C) without re-fetching ~145 GB of DBs (the DBs were deleted after rolling). Pod: CAN (it holds the DBs, INHERITED). Without any DB: the PINNED labeller CANNOT; a new, unpinned reconstruction is possible but not equal to the A1-ruled label.**

Per-row inputs read by the pinned path (file = hold copies of the repo files; line numbers as in those files):
1. `traj` (the RJ5 traj goes here): `p5c_dual_labels.py:42-49` (`load_rows`), `:180-182`.
2. **Initial ego state from the DB scenario**: `p5c_dual_labels.py:166` (`NS.build_scenarios_for_log(db, tokens)`, needs the DB file), `:179` `get_ego_state_at_iteration(0)`; a log with no DB writes an error record (`:161`, "no local DB") = raw fallback. Used by Tc20s `P.iterate20` (`:182`; `p5_target_labels.py:89-120`) through `p4_tracker_comp.tracked8` (`p4_tracker_comp.py:44-51`) -> `navsim_dac.simulated_states` (`navsim_dac.py:129-161`; `pdm_simulator.py:44,52`). The tracker reads: rear-axle world pose x,y,heading, time_point, vehicle parameters, rear-axle velocity, acceleration, tire steering angle, **tire steering rate, angular velocity, angular acceleration** (`pdm_array_representation.py:63-81`).
3. P5-R: `_p5r_row` (`p5c_dual_labels.py:96-110`): the solve uses only `traj`, the Tc20s warm start and the tracker closure `S.real(ego)` (`p5r_run.py:143-144`). The **map** enters only through `ND.drivable_polygons(sc.map_api, ego)` (`p5c_dual_labels.py:199`, `navsim_dac.py:95-100`) and `S.score` (`:105`), i.e. the `p5r_sim8` / `E_maxxy` diagnostics, not the label (`p5r_labels.solve` takes no polygons). It is still called unconditionally inside the same `try`, so a map failure turns the row into `p5r_error` (raw fallback).

What the rows carry (M, bank row keys and `build_teacher_rollouts.py:338-399`):
| input | bank row | RJ5 row |
|---|---|---|
| world pose of the rear axle | `origin_world` (x, y, yaw) | same |
| rear-axle velocity x,y; acceleration x,y; angular velocity; tire steering angle; speed | `ego` = [vx, vy, ax, ay, yaw_rate, steer, speed] (l.372-377) | not in the RJ5 row (the join fetches it from the bank row of the same key) |
| tire steering rate, angular acceleration, time_point, vehicle parameters | **not carried** | not carried |
| map name / map_api (drivable polygons) | **not carried** (needs the DB's log table, or the maps root plus the location) | not carried |

Where else: the DB (`scenario.initial_ego_state`, map name) on the pod (`/workspace/data/navtrain_dbs`, all 1,192 DBs, 164 GB; I from `PK/raw/2026-11-11-p5c-readiness/labels/HANDOFF.md`) or the public nuPlan S3 (`code/fetch_navtrain_dbs.py`). Pod bank-build provenance (I): `pod_dataprep.sh` stage `rank0`, CPU, built from DB scenarios (`RJ5_NAVTRAIN_PLAN.md` s1).
**Why a bank-row reconstruction is not the pinned label (M):** TF-N4's A1 rule. Labels from the metric-cache ego, which is also an exact serialised ego but not the DB object, differ from the DB-ego labels on 40/40 tokens, max 6.36 mm (`TF/n4/a1/PREREG_TF_N4_A1.md:17`, `A1_FIX.md:6`). A reconstruction from the 7 `ego` numbers (two tracker-state fields missing) would be a third ego variant; its size is UNMEASURED (E: mm class). Not recommended: it would break TB-0 (bit identity pod vs dev box) and the A1 rule.

## Q3. Cost measurement: SKIPPED
`F:/tanitad_scratch/rj5nt/seq2.log`: "23:21:33Z free virt 11200 MB >= 5120 MB: go W-A7", no "returned" line (last line of the log). W-A7 is running by the log; I did not start a labelling process. Cost for the pod route (all M/I, from banked numbers, not new): label 0.70 CPU-s/row on the dev box (`label_local/label.log`, M via `TFNT1_STUDY.md` s3), 0.52 s/row P5-R eval (`RESULT_P5R.md`); 76,763 unlabelled rows = ~11-15 CPU-h (E); pod label job ~20 min on 64 cores for the bank (I).

## Open decisions for the MM
1. Pod bank key digest vs the dev copy (needed before any table; 1 min of CPU on the pod after P7).
2. Cause of the bulk control failure (28/36 logs): the pod digest settles bank-vs-copy; if the banks agree, the public DBs differ from the pod's for those logs and RJ5 rows there are lever + provenance.
3. Chain C (13,949 rows) before C1, or the A3 fallback rule (see `PREREG_P5C_A3_DRAFT.md`).
4. Label placement: pod (CAN) after P7 and the P5-C label job window.

## MM addendum 2026-11-15 ~00:10Z: what the bulk R0 control failure is, and what it is not

All probes ran on the pod at nice 19 / ionice idle with a 4 GB address-space limit, while P7 trained. Pod cgroup: 143 GB cap, 15.8 GB anon. Scripts and outputs are in `mm_provenance_2026-11-15/`.

### Open decision 1 is SETTLED: pod bank == dev copy (M)

`rj5_bank_keys_digest.py` ran on the pod's P7 bank (`refe_navtrain/train_grow_cmd_3c_p7/targets_rank0.jsonl`, file sha256 `a5690f81`, = the reserve filter's `sha256_out`). It gives ALL `697b451c...4102d` over 92,896 keys. The per-log TSV is byte-identical to `bank_keys_per_log_digest_slim.tsv` (`cmp` rc 0).

The dev copy's file sha `5471b25a` differs from the pod's only in serialisation (the slim copy carries 6 fields). The pod's `daa23b96` is the reserve filter's INPUT (`sha256_in`, 103,037 keys), not the P7 bank. So the plan's "UNVERIFIED" pairing compared the wrong two files.

The earlier banks `train_grow`, `train_grow_cmd` and `train_grow_cmd_3c` carry identical rank-0 keys (ALL `6d3e0035`, 103,037). There was ONE rank-0 build, by parallel CPU shards, Sep 23 22:29Z to Sep 24 05:01Z (`assembler.log`).

### Hypotheses for the 28/36 bulk-log control failure

| hypothesis | probe | verdict |
|---|---|---|
| bank copy differs from the pod bank | digest above | REFUTED |
| DB bytes differ (public vs pod) | sha256 of 1 diverged + 1 exact log, pod vs public | REFUTED (f239625d / 3486b1e3 equal) |
| map version differs | sha256 of `us-nv-las-vegas-strip/9.15.1915/map.gpkg` + `nuplan-maps-v1.0.json`, pod vs dev | REFUTED (f1509eee / b9c0a623 equal) |
| several bank build epochs | 4 pod banks, same keys | REFUTED |
| thread count (local `--procs 1` vs bulk `--procs 2`) | `rj5_chain.py:109-110` pins OMP/MKL/OPENBLAS = 1 for every roll process | REFUTED |
| `REFE_CD_CACHE` (bulk only) | used only by `fetch_front_camera.py:50` (zip central directory for the DB fetch) | not causal (the DB bytes are equal) |
| yaw-only vs full 3D quaternions in the DB | `quat_probe.py`: 3 diverged + 3 exact logs all store 100 % full 3D quaternions | REFUTED |
| a distinct producer machine for those logs | bank `dt_ms` per log (`dt_probe.py`): both classes ~20 ms median | not supported |

### What it IS (M, `ctl_cmp_pod2.py` over all 5,183 control rows vs the FULL pod bank rows)

- The only input field that differs is the **yaw of `origin_world`, by exactly 1 ULP** (2.22e-16 / 4.44e-16). x, y and goal are equal, apart from goal float32 rounding ≤ 1.5e-5.
- It is **per log**:
  - in the 28 diverged logs, 721/2,452 rows (~30 %) carry the 1-ULP yaw difference, and **100 %** of their rows have a different traj (1,731 of them with an equal origin: history and agent poses go through the same yaw computation);
  - in the 8 exact logs, **0** rows differ in any input and 0 in traj.
- ⇒ **The R0-vs-bank difference is a ULP-seeded numeric difference in the ego-pose yaw computation, amplified by the teacher's closed-loop roll.** The amplification is median 1.7 mm (A) / 7.7 mm (B), p99 0.16 / 0.27 m, max 0.80 m.
- It is **not** a different DB, map, bank or label definition. Why one log gets the ULP split and another does not is still OPEN (H: libm / SIMD path differences on specific input ranges). It does not change the decision below.

### Consequence for A3 / C1

- The RJ5 rows on those logs are valid RJ5 rolls of the same checkpoint, knobs and inputs to 1 ULP. The ULP difference is a measured **teacher reproducibility floor**, not a provenance bias. For scale, that floor (p50 mm, p99 ≤ 0.27 m) against the RJ5 lever (median 0.97 m, `RJ5_LOCAL_ROLL.md`) is about 1 % at the median.
- A3 must disclose it and carry the per-log control status (TB-8). It must NOT require bit identity between the RJ5 traj and a pod re-roll. Bit identity (TB-0) stays on the labelling step (Tc20s -> P5-R) given a fixed RJ5 traj.
- Open decision 4 (label placement: pod) is unchanged: the pod holds all 1,192 DBs, so the pinned DB-ego labeller runs there.
