# PREREG_P5C — Amendment A3 (DRAFT): C1 trains on the optimized privileged driver's labels (RJ5 traj → Tc20s → P5-R)

**Status: DRAFT by a REFe agent, 2026-11-15 ~00:45Z. The Master Mind freezes it (sha256 file beside it) BEFORE any C1 label exists.**
- At drafting time: no RJ5-based P5-R label on the pod, no C1 table, no P5-C training step, no P5-C read, no W-A7 number of any kind (the bank-traj W-A7 run is in progress on the dev box, `p5c_launchkit/w_a7/run`, and I did not read its scores).
- `PREREG_P5C.md`, `_A1.md` and `_A2.md` are not edited. A3 sits beside them. Everything A3 does not change stands.
- Evidence tags: **[M path]** MEASURED by us (artifact named) · **[I]** INHERITED, not re-verified · **[E]** ESTIMATED · **[H]** hypothesis.
- Keys: `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan` read at mirror tip `15f0909b`; `SG` = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107`; `HOLD` = `SG/p5c_a3`.

## 0 · The ruling
PI 2026-11-14 (GAC `D-REFE-PI-2026-11-15-1`) [I, the brief]:
- **C1** trains on the optimized privileged driver's labels: the RJ5 teacher trajectory (`positive_jerk_limit_when_nonnegative_acc` 5.0, same `checkpoint_2400`) → Tc20s → P5-R (λ\* 3.0). That is N2-R's label construction (teacher navtest 95.61 against 89.81 for the bank's raw traj) [I].
- **C0 is unchanged**: the bank's raw `traj`, no table. **C0 runs first.** Same approved spend.
- This is a pre-data change of C1's label. No C1 label or table exists, and no P5-C number exists.

## 1 · The C1 label (definition, committed)
For every P7 bank row `r` (150,603 tuples; `targets_sha256 92af8d66…`, rank-0 file sha256 `a5690f81…`, rank-1 `8b6ccd1b…`, as the A2 runner pins):

| row | label input traj `u(r)` | C1 table label | `label_source` / `fallback_reason` |
|---|---|---|---|
| rank 0, an RJ5 roll exists in the pinned join | the RJ5 traj of the same (log, token) | P5-R(λ\*=3.0) warm-started from Tc20s(`u`), both from the **pinned dual labeller run unchanged** with `u` as its `traj` | `rj5` / null |
| rank 0, join status `missing` | the bank traj | P5-R of the bank traj, the same pipeline (A2's label for that row) | `bank_fallback` / `missing` |
| rank 0, the P5-R solve raises (`p5r_error`) | (either) | **raw fallback: label := the bank traj** (C0's target; the gates' own rule) | its source / `raw_fallback` in the table |
| rank 1 | the bank traj (labelled, so the unchanged gates see every row) | **identity: label := the bank traj** (`raw_rank1`): rank policy **rank0, forced** (§3.3) | `bank_fallback` / `rank1` |

- **The key is unchanged:** `log|token|rank|sha256(the BANK row's own traj)`, exactly what `refe/traj_label_table.py` (A2 pin `d65e01c4`) looks up. The trainer, the table format, the reader's three outcomes and its rank-policy enforcement are unchanged.
- **The label VALUE is computed from `u(r)`.** Mechanism (`HOLD/a3/p5c_rj5_path.py`):
  - `feed` writes a feed bank: the same rows, with `traj := u(r)`.
  - The pinned labeller runs on the feed bank (`P5C_BANKDIR=<feed>/bank`).
  - `rekey` rewrites each record's `traj_sha256` from the sha of `u` to the bank row's sha. It REFUSES unless the record was computed from exactly the fed `u`.
  - The pinned gates and the pinned table builder then run on the REAL bank.
- **Per-row provenance** is carried in the rekeyed records and in a sidecar `C1_provenance.jsonl.gz`, one line per table key: `label_source`, `fallback_reason`, `label_traj_sha256` (the RJ5 traj sha for `rj5` rows), `rj5_source`, `roll_platform` (`devbox` | `pod`, declared per source, never guessed), and the R0 fields of §4.
  - The sidecar's sha256 goes into the gates' meta under `a3` (`provenance` subcommand). Through `--meta-json`, the table's `labeller_meta_sha256` therefore binds it. `traj_label_table.load_labeller_meta` accepts the block [M `test_provenance_sidecar_and_sha_bound_meta`].
- **No pinned label-math file changes.** These are byte-unchanged:
  - `p5c_dual_labels.py 50b9f0db`, `p5r_labels.py 16d2aaf0`, `p5r_run.py d0a4084b`, `p5_target_labels.py 294644df`, `lambda_star.json 83988dda…`;
  - `p5c_table_gates.py 5c2f49cf` (imported as a library by the tests and by `tb0r`);
  - `traj_label_table.py d65e01c4`.
- **This route is already proven on real data** [M `HOLD/logs/REAL_REKEY_LOCAL.log`].
  - The pinned labeller's own 16,133 RJ5 records (the MM's 2026-11-13 dev-box local run, blob `50b9f0db` on an RJ5 feed bank) rekey with 0 refusals.
  - The pinned gates then join them onto the real bank rows with **0 sha mismatches**.
  - Without the rekey, the gates drop **13,724 of 16,133** rows on their own sha check (85.1 %: the rows whose RJ5 traj moved).

## 2 · RJ5 traj sources (committed; shas)
| source | rows | roll platform | file | sha256 |
|---|---|---|---|---|
| join of local + A + B | 78,947 / 92,896 rank-0 keys = **84.98 %** | `devbox` (all three) | `F:/tanitad_scratch/rj5path_tmp/join_out/rj5_bank_join.jsonl.gz` | `426e2abc4c404c848dca1a3e9301d95196f65ec14f1815f6b374426cc5aa5170` [M; my `feed` recomputes it and refuses any other] |
| its manifest (771 row files, each sha256) | | | `SG/rj5path/rj5_bank_join_manifest.json` | `93c87ac3992aaa62344455d16eb96c4a692170e3df447333be2bebe7e74706e8` |
| join tool | | | `SG/rj5path/rj5_bank_join.py` (blob `4713739f`; tests `test_rj5_bank_join.py` blob `4e630817`) | `66744c866c64cbb19d30289aaa3d9832db0da09773243866ed087d8ddb7523bc` |
| chain C (to roll) | 13,949 rank-0 keys over 192 logs | `devbox` (the MM's queued `c_after_wa7.sh`) or `pod` (§6) | rows dir `run_fetch_C/rows` (dev) or the pod's | logs list `logs_fetch_C.txt` sha256 `07d3ea80226ad54dc441e7922d53f80289c49a9795eecc695d5f72e29c1a1db1` |
| roller | | | tip `PK/raw/2026-11-11-teacher-fidelity/rj5_navtrain/rj5_roll.py` (blob `7c2fe686`), `rj5_chain.py` (`e6064314`), TF-N2 patch `PK/raw/2026-11-11-teacher-fidelity/n2/reroll/build_teacher_rollouts.py` (`791fa8b1`) | `e43733e2…`, `a2702a01…`, `4c3d5869…` |

- **The C1 build uses the join RE-RUN with `--rows C=<chain C rows>`.**
  - Its output sha256 and manifest sha256 are written into `LABEL_SELECTION.md` and passed as `--expect-join-sha256` BEFORE `feed` runs.
  - A join that reports any conflict or knob mismatch is not used (`rj5_bank_join.py` exits non-zero).
- ⚠️ **Banking before the window.** `rj5_bank_join.py`, its tests, the join file and `rj5_labels_local.jsonl.gz` (`F:/tanitad_scratch/rj5nt/label_local/`, sha256 `23f59e53…`, the TB-0r reference) exist only on E:/F: today.
  - They are A3 inputs and must be banked (repo `raw/` or private HF, verified by sha256) before the pod window.

## 3 · Coverage rule and rank policy (committed before any label)
1. **Launch bar.** C1 launches only if the RJ5 coverage of P7 rank-0 keys is **≥ 99.0 %**, i.e. `bank_fallback/missing` ≤ 1.0 % of 92,896, at most 928 rows.
   - Same bar as TB-4 / S4's 1.0 % fallback.
   - `feed --min-rj5-coverage 0.99` refuses below it with exit 3 and writes no feed bank. Today's join reads 0.849843 and refuses [M `HOLD/logs/REAL_FEED_FULL_diag.log`].
2. **A missing row gets the declared fallback, never a silent fill.**
   - A row with status `missing` (no roll, or a roll error) gets `bank_fallback`.
   - A bank key ABSENT from the join is a REFUSAL (a stale join).
   - Every fallback row is listed in the sidecar and counted in the meta.
3. **Rank-1 rows: rank policy `rank0`, forced.**
   - No RJ5 roll exists for a rank-1 goal twin. `rj5_roll.py` refuses a patched route-lane rank, and the join is rank-0 only.
   - Giving rank-1 rows A2's bank-traj P5-R label would put a second, different label into C1 (a mixture of two levers). Identity keeps C1's lever to "RJ5 → Tc20s → P5-R on rank-0 rows".
   - The gates' §4.3 / A1 rank decision is still computed and reported. It decides the table only through **STOP** (RK0 fires ⇒ no launch). `all` and `rank0` both build `merged_labels_rank0only.jsonl`.
   - Reach: under scene-epoch rotation the lever covers ≈ 69 % of draws [E, card §4.3].
   - What ALL-RANKS would cost: rank-1 RJ5 rolls need new roller code (route-lane rank) plus ≈ 57,707 × 2.9 dev-proc-s ≈ 46 dev-proc-h [E from §6's measured rate]. Not proposed.
4. **Raw fallbacks** (P5-R solve errors), per rank ≤ 1.0 %: TB-4 / S4 unchanged.

## 4 · Gate set on the RJ5-based table (A2's gates, thresholds unchanged unless shown not to apply)
| id | A3 form | applies as written? |
|---|---|---|
| **TB-0** | A2's R-B (R-TOL-1/2/3) + R-A Tc20s warm start, on the **control pass**: the pinned labeller run with the override OFF on the 2,400 P5-R EVAL rows (bank trajs), thresholds unchanged | **Not on the RJ5 table.** The EVAL labels are built from bank trajs and an RJ5 label differs by construction: 85.1 % of the local RJ5 rows moved [M], median ≈ 0.97 m [I `RJ5_LOCAL_ROLL.md`], against R-TOL-2's median bar of 1e-3. So it runs where it can apply: on a labeller-reproduction control. |
| **TB-0r** (new) | the pod's RJ5-based labels against the dev box's on the same RJ5 inputs: the 16,133 local rows of `rj5_labels_local.jsonl.gz`, same pinned labeller. Tc20s ≤ 1e-9 (R-A); P5-R R-TOL-1/2/3 (R-B), per rank; coverage n == 16,133; every dev row's traj sha == the fed traj's (`p5c_rj5_path.py tb0r`, through the pinned gate functions) | new: TB-0's question asked of the actual C1 inputs. RED arms: Tc20s +2e-9, P5-R +0.01 m, coverage n+1, dev row off the fed traj. Each FAILS [M tests]. |
| **TB-1** | A2's TB-1 (Tc20s 1,018 file + reproduction of all 2,400 EVAL labels) on the control pass | same reason as TB-0 |
| TB-2 | S3 ≤ 1e-6 on every `p5r` row of the RJ5 table | yes |
| TB-3 | identity control on the bank | yes |
| TB-4 | 150,603 / 150,603 keys; raw fallback ≤ 1.0 % per rank; **plus §3.1** (bank_fallback ≤ 1.0 % of rank-0 keys) | yes, + the coverage bar |
| TB-5 | finite, \|Δyaw\| < π against **the bank traj**, `sim_hz` 10 | yes (the literal "vs `traj`" = the row the label replaces). `feed` additionally refuses an RJ5 traj with \|Δyaw\| ≥ π before any label. |
| TB-6 | per rank, ≤ 2× **against the bank traj** (the literal reading: the target C0 trains on). **Rank-0 FAIL ⇒ STOP.** Rank 1 is reported; it decides nothing under forced rank0 (A1 R-D). | yes |
| TB-6r (reported) | the same ratio against the RJ5 traj (the P5-R distortion alone) | context only |
| TB-7 | `n_differs` per rank against the bank traj (feeds G-LIVE (a)) | yes |
| **RK0** | r_0 = sim8-fail(**bank traj**, C0's target) − sim8-fail(RJ5-based P5-R label), same fixed-rule 300 rank-0 rows, same estimator (paired log-cluster bootstrap, 10,000 draws, seed 20260927). Input = `repair-combine` of the pinned labeller's `repair` on the real bank and on the feed bank (the sample rule picks the same rows on both: all logs have DBs on the pod). r_0' with T = the RJ5 traj is reported. | yes; the upper bound < 0 ⇒ no C1 launch |
| **TB-8** (disclosed, never gated) | per row: `r0_log_status` ∈ {exact, diverged, uncontrolled}, `r0_row_dxy_m`, `r0_log_max_dxy_m`, `rj5_source`, `roll_platform`, recomputed against the bank by `feed` (never read from the rows' own `bank_dmax`) | today, over local+A+B, see below |
| **TB-9** (new, integrity) | `feed` and `rekey` exit 0 (every refusal rule of the module); the feed-map sha is pinned into `rekey`; the rekey counts equal the feed counts; the sidecar keys equal the table keys | |

TB-8 today [M `HOLD/logs/REAL_feed_manifest_full.json`]:
- 52 controlled logs: **28 diverged, 24 exact**.
- Rank-0 rows by their log's status: 2,567 diverged, 2,616 exact, **87,713 uncontrolled**.
- Per the MM addendum this is a measured reproducibility floor, not a provenance bias. It is disclosed with every C1 number and never required to be zero.

**Both outcome branches, every check:**
- Any of TB-0, TB-0r, TB-1, TB-2 to TB-5, TB-9 FAILS ⇒ `ZZP5CL_FAIL` (a pipeline fault). No C1 launch. Diagnose, fix, rebuild. C0 proceeds: it needs no table.
- TB-6 rank 0 FAILS, or RK0 fires ⇒ **STOP**: the RJ5-based label is not learnable or is harmful at bank scale. The next lever, committed now: **P5-T** (PREREG_P5R §10) on RJ5 inputs.
  - C1 is not swapped to A2's bank-traj label without a new PI ruling. That would be a moved goalpost.
- §3.1 coverage below 99.0 % ⇒ no C1 launch. The next lever is to roll the missing logs (§6) and rebuild. C0 proceeds.
- All PASS ⇒ the MM writes `LABEL_SELECTION.md` (label `p5r-rj5`, the two `gates.json`, `tb0r.json`, the join sha) and fills `P5C_LABEL_TABLE_SHA256`.
- TB-0r FAILS while TB-0 passes ⇒ the nondeterminism depends on the input. It is reported as such: a finding, not a pass.

## 5 · W-A7 on RJ5-based labels
- **The 1,018 W-A7 rows are P7 bank keys with RJ5 trajs** [M `HOLD/logs` via `wa7_membership`]:
  - 1,018 / 1,018 rank-0 keys over 11 logs, all 11 in the bank;
  - 1,018 / 1,018 carry an RJ5 traj (sources: local 490, B 393, A 135);
  - each W-A7 traj's sha == its bank row's sha.
  - Rolling cost: **0**. The dev box holds these 11 logs' DBs (W-A7 is labelling them now).
- **Committed: W-A7 is re-read on the RJ5-based labels (W-A7-RJ5), with A2 §6's definition, estimator and bar unchanged.**
  - The P5-R Δ decides. Tc20s is reported.
  - The re-read uses W-A7's own instrument on a 1,018-row feed (`traj := RJ5`).
  - Cost ≈ 1–1.5 h dev-box CPU at 1 thread [E, W-A7 HANDOFF's timings].
  - It is a **C1** launch blocker, not a C0 one, and it runs during C0.
- Control (i) cannot apply on RJ5 inputs: no banked TF-1 per-row PDMS exists for RJ5 labels.
  - It is replaced by **(i′)**: the W-A7-RJ5 no-A7 labels reproduce the C1 table's labels on the same 1,018 rows within R-TOL-1/2/3.
  - The instrument itself is validated by the bank-traj W-A7 run's control (i), which must be VALID. Its Δ is reported as context.
  - Controls (ii) to (iv) are unchanged.
- Outcome rows, unchanged from A2 §6:
  - INVALID ⇒ fix and re-run;
  - **W-A7-PASS** ⇒ the blocker is cleared;
  - **W-A7-REBUILD** (CI upper bound < 0 AND point ≤ −0.10 PDMS) ⇒ the RJ5-based labels are rebuilt with A7 inside the solver (a further amendment, new pins) and W-A7-RJ5 is re-read.

## 6 · Run order (committed), and why it beats "C0 → window → C1"
Facts [M unless tagged]:
- **Pod CPU quota: cgroup `cpu.max` 1530000/100000 = 15.3 CPUs, while `nproc` = 128.** Memory cap 143.0 GB; 15.9 GB anon with P7 training (probe 2026-11-15 00:16Z).
- **Dev-box RJ5 roll rate:** roll phase 1.44–1.46 wall-s/row at 2 procs; 1.78–1.86 including the DB fetch (chains A and B, 66,204 rows; per-row roller `sec` mean 1.86–1.93 s). That is ≈ 2.9 proc-s per row, not the ~13 s/row behind the brief's "~50 CPU-h, 40 min on 80 procs" [I], which also assumed cores the cgroup does not grant.
- Chain C on the dev box ≈ 13,949 × 1.86 s ≈ **7.2 h wall** [E from the measured rate].
- Chain C on the pod ≈ 11.8 dev-proc-h ÷ an EPYC-7452 per-core factor of 0.55–0.8 ÷ 14 procs ≈ **1.1–2 h** [E].
- P7: step 7,013 of 9,075 (train.log tail, 00:2xZ). 2,062 × 51.2 s [I] ≈ 29 h ⇒ ends ≈ 2026-11-16 05:40–06:40Z [E].
- Label pass on the pod: 150,603 rows × 0.60 dev-CPU-s [M, A1 Point 6] = 25.1 dev-CPU-h ⇒ **≈ 2–3 h wall on 15 shards** [E].
  - A1 Point 6's "≈ 0.5 h on 64 cores" assumed 64 cores; the pod grants 15.3.

**Order:**
1. **Now:** dev-box chain C (already queued by the MM, `c_after_wa7.sh`). It finishes ≈ 7–8 h after its start, i.e. well before P7 ends. Re-run the join with C and pin its sha. One roll platform (`devbox`) for every RJ5 row.
2. **P7 ends → pod window (GPU idle ≈ 2.5–3.5 h [E]):** `feed` → control pass (2,400 EVAL rows, override off) and RJ5 pass (all 150,603 rows, ≤ 15 shards) → `rekey` → both `repair` runs + `repair-combine` → gates (control pass: TB-0/TB-1; main: TB-2..TB-7 + RK0) + `tb0r` → `provenance` → table build `--rank-policy rank0 --require-coverage 1.0 --meta-json <a3 meta>` → verify.
3. **C0 launches** (it needs no table; a label failure does not hold it).
4. **During C0 (~28 h):** W-A7-RJ5 on the dev box; P7's read; any label fix.
5. **C1 launches right after C0** (no second window).

**Why not C0 → window → C1:**
- The GPU idles the same ≈ 3 h either way.
- With the window first, a gate failure is found and fixed during C0's 28 h at no GPU cost. Found after C0, it idles the GPU while it is fixed.

**Fallback, if chain C is not complete on the dev box when P7 ends:**
- roll the missing logs on the pod inside the same window (+1.1–2 h [E]; §7), with the pod R0 control;
- or, if the MM prefers, start C0 and open the window between C0 and C1.

## 7 · Chain C on the pod: feasibility (filesystem probes only, nice 19 / ionice idle; nothing launched)
**CAN, after P7 ends, with three ship items** [M pod probes 2026-11-15 00:16–00:40Z]:
- `/workspace/venv-teacher`: Python 3.11; torch 2.7.1+cu128, numpy 1.26.4, nuplan-devkit 1.2.2 (editable), driverl (editable), shapely 2.1.2, scipy 1.17.1. These are the dev box's `driverl-eval` versions. No threadpoolctl (as A1 R-C).
- `checkpoint_2400.pt` sha256 `d7b1fdd0…` and `driverl_teacher.yaml` `0521575f…` both equal `pod_teacher_env.sh`'s pins. DriveZero at `2495954`.
- DBs: 1,192 flat `<log>.db` files under `/workspace/data/navtrain_dbs`, including **192 / 192 of chain C's logs**. Maps are under `/workspace/data/nuplan-maps/nuplan-maps-v1.0`.
- `refe/build_targets.py` (`b231d1e1`) and `refe/navtrain_scenarios.py` (`848fe684`) equal the tip's in both `refe-plan-p7` and `refe-plan-vdac`.
- **`refe/build_teacher_rollouts.py` there is the UNPATCHED `049b0290` (no `TEACHER_OVERRIDES`), so `rj5_roll.py` would refuse.**

**Ship list** (to a new dir, e.g. `/workspace/data/rj5c_ship/`; nothing under `refe-plan-p7`):

| file | blob | sha256 |
|---|---|---|
| `rj5_roll.py` | `7c2fe686` | `e43733e2af4e34e10b5df3b4071eb0b74c03402ecf8de6e6b76dc83ea9d69ffe` |
| `rj5_chain.py` | `e6064314` | `a2702a01c4ca887900e1321348c0f1fc25c69e5e001bd4d52f183c10ff12a801` |
| `patch/build_teacher_rollouts.py`, first on `PYTHONPATH` (empty overrides ⇒ bit-identical to `049b0290`'s build) | `791fa8b1` | `4c3d58697be8569a0bc361b109c425542d6495d0cb2d7adae77ca8da08aed251` |
| `logs_fetch_C.txt` | `3a6beb44` | `07d3ea80…` |

- **Bank:** the pod's own `train_grow_cmd_3c_p7/targets_rank0.jsonl` (`a5690f81…`). `rj5_roll.load_bank` reads only the log, token, rank and traj of rank-0 rows.
- **Env:** `source /workspace/teacher_env.sh`, then:
  - `DRIVERL_EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 REFE_SIM_HZ=10`;
  - `unset DRIVERL_EVAL_ROUTE_LANE_RANK`;
  - `PYTHONPATH=<ship>/patch:<pkg>/refe:<pkg>/code:<pkg>/eval:$DZ_ROOT/nuplan-devkit` (Linux `:`), with `<pkg>` a copy of `refe-plan-vdac`.
- **Command:** `rj5_chain.py --held-db-dir /workspace/data/navtrain_dbs --db-dir <empty scratch dir> --procs 14 --batch-logs 48 --arms RJ5 --control-every 20`, **without `--fetch`**.
  - Its delete loop removes only files it fetched (`rj5_chain.py`, the `fetched` list), so no pod DB can be deleted. The chain's launcher must assert that `--fetch` is absent.
- **Memory:** 15 roller processes at 1–2 GB each [E] ≪ the 143 GB cap, after P7 only. Not while any trainer runs.
- **Pod R0 control** (required iff any RJ5 row is rolled on the pod; optional otherwise): R0 re-rolls of the 36 bulk control logs.
  - **[H] prediction:** pod R0 == bank bit-for-bit on the logs where the dev box split by 1 ULP. Same Linux / glibc numerics class as a Linux bank builder; the Windows build is the odd one out.
  - **Outcome A, all exact:** the floor is dev-box-specific. Pod-rolled rows carry no floor; TB-8 marks them `pod`.
  - **Outcome B, some logs diverge:** the bank was built on a platform whose numerics differ from this pod's too. The bank-builder CPU of 2026-09-23/24 is [U]: it was another pod. The floor applies to both platforms and is disclosed per log.
  - **Either way:** rows whose bank `teacher_device` is `None` (built on CUDA: 104 A rows, 376 B rows, 27 local [I `control_breakdown.json`]) can never reproduce on a CPU. They are split out, never counted as a platform effect.
  - Neither branch blocks C1. That is the MM addendum's ruling: a floor, not a bias.

## 8 · Defect found in the A2 kit (pre-data; fix in the A3 chain)
- **`refe/p5c_pod.sh` (A2 pin `cf823843`, lines 137–140) defaults `P5C_NSHARD` to `nproc` and caps it by `nproc`.**
  - On this pod `nproc` = 128 while the cgroup grants 15.3 CPUs [M]. The default would start 128 label shards on 15.3 CPUs. Each shard loads nuPlan and the maps, about 0.8 GB per labeller-class process [M: RSS of the P7 label workers].
  - The result is a thrash and roughly 100 GB of RSS against the 143 GB cap.
  - `code/pod_dataprep.sh:30-41` documents exactly this trap ("CORES FROM THE CGROUP QUOTA, NOT nproc").
- **Fix (A3):** read `cpu.max` (v2) or `cpu.cfs_quota_us / cpu.cfs_period_us` (v1) as `pod_dataprep.sh` does, and cap the shard count at floor(quota). Add a test that fails if `nproc` is used.
- **Until the fix lands:** pass `P5C_NSHARD=15` explicitly.

## 9 · Pins
**New (HOLD → repo `PK/raw/2026-11-11-p5c-readiness/a3/`, paths ≤ 200 chars; `git hash-object --no-filters`; 0 CR bytes):**

| file | blob | sha256 |
|---|---|---|
| `a3/p5c_rj5_path.py` | `409f448152b4d51ee4ffc571c9bdf499e0ecc93d` | `951d3e1d5027bedc933b2bd766c5df2d98373b3829209ffd71f87441944ae309` |
| `a3/test_p5c_rj5_path.py` | `c3cc16a361d96d36762032ddc24fe45c7c60d215` | `43f28014a1d1dca7397812b922ada941e7c209103ce862170e9c1f35c9bed63f` |
| `a3/run_mut.py` (mutation harness) | `3dd4c96c0786b4cb0545cfa0bfdfb93a3c175ad4` | `cadc14909489853acb55d1c4a3b16f308cf6bd9d2e6c7e7ec94752cb4206dc68` |

- **Tests:** 23 / 23 pass. **8 / 8 mutation arms RED:**
  - wrong-key check removed;
  - override never fed;
  - rekey sha check removed;
  - rekey keeps the fed sha;
  - missing record filled silently;
  - absent key falls back;
  - repair raw taken from the feed;
  - platform guessed.
  - Source: `HOLD/logs/TEST_RUN_A3.log`, run serially with TMP on the scratch and `CUDA_VISIBLE_DEVICES=-1`.
  - The end-to-end tests import the pinned `traj_label_table.py d65e01c4` and `p5c_table_gates.py 5c2f49cf` unchanged. The labeller is a stub with the pinned labeller's record format.
- **Unchanged and re-asserted:** every A1 Point 2 blob and A2 §3 blob except the two below.
- **MOVE (A3 edits, not yet written; pins filled AT FREEZE, blank = not yet known):**
  - `a1/p5c_pod_label.sh`: a two-pass runner (control pass + RJ5 pass), `feed` / `rekey` / `repair-combine` / `tb0r` / `provenance`, forced `rank0`, classify from `gates.json` per gate rather than from the ZZTB line, new pins (join, feed tool). A2 `d25e1a92` → `____`.
  - `refe/p5c_pod.sh`: cgroup shard cap (§8); passes the join sha and `P5C_LABEL=p5r-rj5`. A2 `cf823843` → `____`.
  - The W-A7-RJ5 driver (or a `--traj-source` input to `w_a7_read.py`): `____`.
  - Each of them gets a deliberate-regression arm.
  - ⚠️ A `launch_gate.py` pin edit moves 6 profile fingerprints in 3 tests: run all ~69 launch_gate-reading tests before landing [I MEMORY].

## 10 · What A3 does not change
- Bar B = 0.7031.
- The recipe (§3 of the card, with A2 §4's deviations).
- The aux head is OFF.
- C0, and its argv.
- The C1/C0 argv diff test (`--traj-label-table`, `--out`).
- K accounting.
- The PI's two conditions: tracked AND transparent; the wording "survive the benchmark's tracker".
- W-TRANSP and W-FAM as read blockers.
- The closure-land rule.
- The launch-gate PASS token.
- C0r still needs the PI.
- The claim is scoped **"RJ5 → Tc20s → P5-R on rank-0 rows"**, with TB-8 disclosed.
