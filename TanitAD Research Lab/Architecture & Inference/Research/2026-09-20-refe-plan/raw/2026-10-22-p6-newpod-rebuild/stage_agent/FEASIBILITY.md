# FEASIBILITY: can the 3-C bank (`train_grow_cmd_3c`) be rebuilt BYTE-IDENTICALLY? (2026-10-21, p6-stage agent)

Read-only study of the repo tip `f2f604f` (mirror, branch `agent/arch-inf-20260803`) plus read-only HF listings. Nothing built, nothing trained.
Evidence classes: MEASURED (artifact named) / INHERITED (a document, not re-verified) / ESTIMATED / UNVERIFIED.
`R` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## VERDICT

**No structural obstacle. The rebuild is deterministic BY CONSTRUCTION, and its acceptance test is three recorded sha256s. Byte-identity cannot be PROVEN in advance, because one input is hardware: the CPU teacher network (float32) now runs on a different machine, and no pod CPU model is recorded anywhere (`R/eval/P6_ON_11VCPU.md` l.62, MEASURED absence). Proceed; the cheap early discriminator is the 74-twin control already banked in the repo (section 5).**
It is therefore NOT a "cannot be byte-identical" case, so the brief's STOP rule does not fire. It is a "byte-identical if the arithmetic reproduces, and we will know by sha" case.

## 1. What the bank is (MEASURED, `R/raw/2026-10-18-p6-gate-diagnosis/raw/gate_a5a4ee8/merge_report_3c.json`, `R/raw/2026-10-17-twin-bank-3c/PREREG_3C.addendum6.md`)

| file | lines | sha256 (full) |
|---|---|---|
| `train_grow_cmd/targets_rank0.jsonl` (input) = `train_grow_cmd_3c/targets_rank0.jsonl` (copied) | 103,037 | `daa23b9625…` full: `daa23b962e9513673347694e097fb714a53b646c4d402802d6d7b96354c0bc5e` |
| `train_grow_cmd/targets_rank1.jsonl` (input) | 86,851 (44,409 lane-rank twins + 42,442 other) | `2f3e461391d7bbc6c92a4153ef1837943c458d9635aad13fdb27b74a46035898` |
| `train_grow_cmd_3c/targets_rank1.jsonl` (OUTPUT) | 64,263 (= 86,851 - 22,588 dropped; 21,821 twins kept: K1 18,489 / K2 2,781 / K3 551) | `2a63acc2f63404818939b21d45cfa27efb5412cbefdf38e14aeb8673598cf409` |
| trainer's `identity.targets_sha256` (from the run's `config.json`) | n_tuples 167,300 | `8775e73ee948588c6db6b18e3f5a65f6686b313405d6b28e439f85114d5f87a4` |

A kept twin row is the input row with `traj`, `traj_world`, `goal` taken from the NEW 10 s CPU-teacher rollout (route held at step 0, goal horizon 6 s), plus `aug.{divergence_m, route_hold, goal_horizon_s, commit_f10, reroll}`. Every other row passes through byte-identical, in input order; a dropped twin is absent (`reroll_bank_3c.py` header; `controls.json` I3 with four mutation arms).
So the OUTPUT depends on every re-rolled float. That is the whole risk.

## 2. Every input, where it comes from, and whether it is deterministic

| # | input | source | token? | deterministic? | check after rebuild |
|---|---|---|---|---|---|
| 1 | `train_grow` bank (P2's bank) | HF dataset `Sayood/tanitad-refe-navtrain-banks`, `refe_train_grow.tar.zst` (1,043,549,966 B; LFS sha256 `a8e77141b5d8f921c886aeca3d34f4fb6d50a7ae1522f2bfe6f79f7d1947a58a`, MEASURED from the HF tree API) | **YES, private** | content-addressed | the tar's sha256; then #2's output shas |
| 2 | `train_grow_cmd` (commands applied) | chain `commands` stage: `p6_prep.py command-table` + `annotate_commands.py` (pure JSON rewrite, `sorted(glob)`, no RNG/time: MEASURED by reading the file) | needs #1 | yes | rank0 sha `daa23b96…`, rank1 sha `2f3e4613…` above. Cross-check: the old run's `commands.json` is on HF `Sayood/tanitad-refe-p6-paper-faithful/logs/commands.json` (7,521,701 B) |
| 3 | NAVSIM metadata (command table source) | public OpenScene tgz, pinned to revision `a76f840b`, 7,052,210,642 B, sha256 `193138ac68b2…` (verified by the chain) | no | yes | the chain refuses a different byte count / sha |
| 4 | nuPlan DBs (1,192) | `code/fetch_navtrain_dbs.py`, range-fetch from the public nuPlan S3 zips (163.8 GiB on disk) | no | yes (static public data) | `have_dbs` count 1192 |
| 5 | nuPlan maps v1.0 | public S3 zip, size pinned 971,557,640 | no | yes | `pod_teacher_env.sh` size + 4 city dirs |
| 6 | teacher env | `code/pod_teacher_env.sh`: DriveZero `2495954`, `checkpoint_2400.pt` sha `d7b1fdd0…` and config sha `0521575f…` asserted, 113 pinned deps (`code/requirements_teacher_lock.txt`), torch `2.7.1` from the cu128 index, NAVSIM `3e8291b` | no | pinned, but the **CPU arithmetic is the machine's** | see section 3 |
| 7 | D9 logged parts `metric_cache` (103,037) | chain `cache` stage (`build_metric_cache.py` is blob `695c5dd91f` at `c1dc5d3`, `bcc04b5` AND the tip: identical) | no | yes (PDM geometry; the chain's own D9 exactness check) | chain `ZZD9_PASS` |
| 8 | twin parts `metric_cache_twins_3c` (44,409 = K1 32,929 / K2 8,090 / K3 3,390) | `R/raw/2026-10-17-twin-bank-3c/code/build_twin_parts_3c.sh`, run with the **frozen D7 tree** (see 9) | no | a pure function of (log, token, K) (script header); NOT banked, 74/44,409 exist in the repo only as the mini control | `check_twin_parts_distinct.py` (`ZZD7_DISTINCT_OK`, differ >= 0.9) |
| 9 | frozen D7 tree `$PKG` (`/workspace/ops/d7fix/pkg`) | = commit **`c1dc5d3`** `refe-plan` (`code/ refe/ splits/ SHIP_MANIFEST.json`) + the 6 files of `R/raw/2026-10-16-p6-d7-pdm-route-rank/code/refe/`, shas in its `FIXED_FILES.sha256` | no | pinned | `sha256sum -c FIXED_FILES.sha256` |
| 10 | frozen 3-C code (5 files) | `R/raw/2026-10-17-twin-bank-3c/code/` + `MANIFEST.json` (shas pinned into every shard's `shard_meta.json`; a changed file makes `shard_meta` REFUSE) | no | pinned | `MANIFEST.json` |
| 11 | the 3-C PREREG | sha256 `e8365f34fd5f11a23a1b5e0fceda14dc14426277ae09db3e5335526c1bd6cde3` (`merge` checks it) | no | pinned | merge |
| 12 | shard plan | `run_full.sh`: **720 shards, `--chunk 32`**, `xargs -P NPROC`; one shard = one `i/N` slice of the bank (deterministic) | no | yes. Keep 720 / 32 (the planner is stateful across a chunk: `BR._reset_planner_state`), concurrency may change | merge refuses unless every lane twin has exactly one audit record |

⚠️ The unbanked, un-re-creatable inputs: **none.** Everything that is not in the repo is either on HF (#1) or public (#3-#6) or derived (#2, #7, #8).
⚠️ **Teacher rollouts and merged twin parts are NOT banked** (the brief is right): the per-twin `audit_all.jsonl` and all 720 shards lived only on the old pod's volume. Searched at 3 locations (HF dataset listing, all `Sayood/*` model repos by name, the dev box `C:/Users/Admin/par18`, `qland`, `tanitad-push`, `D:/Projects`): no copy.

## 3. Sources of non-determinism, and what I found

* **RNG / sampling / time in the pipeline: none found.** `reroll_bank_3c.py`, `build_teacher_rollouts.py::make_row`, `annotate_commands.py`: no `random`, no `shuffle` on the 3-C path (the only seeded shuffle, `build_teacher_rollouts.py:515`, is in the old per-log harvest, not in `reroll_bank_3c.py run`), `time.time()` only feeds `rollout_s` (an audit field, not a row field). Shard order, chunking and merge order are functions of the bank.
* **Hardware float arithmetic: UNVERIFIABLE a priori.** The teacher is a neural network on the CPU (`pod_dataprep.sh`: "THE TEACHER RUNS ON THE CPU", "outputs differ from CUDA by float noise only, 0-7 cm"; the 3-C run forces `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`). torch/oneDNN pick kernels by CPU ISA, so a different CPU family can change the last bits; the rows store floats, so one flipped bit changes the file's sha. The old pod's CPU model: **recorded nowhere** (MEASURED absence, `P6_ON_11VCPU.md` l.62). Same-pod reproducibility IS measured (controls I1/I2: 74/74 twins, per-twin `f`, `d_twin`, commit set and the 23 kept rows bit-equal to the earlier probe); cross-pod is not.
* **Commit boundary:** `commit = round(f10, 3) >= 0.75`. `f10_round_vs_raw_disagree` is 4 / 1 / 0 (K1/K2/K3) in the merge report, so a handful of twins sit near the boundary; a one-ulp change could flip one and change the line count, not just a digit.
* **Concurrency:** the planner caches (`map_feature_cache`, `query_cache`, `map_step_speedups`) are per-process and reset per twin/chunk; the old run used `xargs -P 12`, shards resumable. Not expected to matter, but keep chunk 32 and shard count 720 to leave no order-dependence untested.
* **Known tolerance, not a flip:** 2/74 `origin_world` rows differ in the last bits between a rebuilt `make_row` and the input row; the code tolerates 1e-6 and writes the INPUT's `origin_world`, so it does not reach the output.

## 4. Cost (ESTIMATED from MEASURED rates; the new pod's CPU quota is UNVERIFIED until probed)

* Re-roll CPU time: `rollout_s` sum **356,604 s = 99 CPU-h** over 44,409 twins (merge report). Old pod at an 11.05-CPU quota: ~11.2 h. At 24 cores: ~4.1 h; at 96 real cores: ~1.1 h.
* Twin parts: K2/K3 measured 11,480 twins in 3,548 s at 12 workers (3.2 twins/s): all 44,409 ~ 3.8 h at 12 workers, ~1 h at 48.
* D9 logged parts: ~24 CPU-h (the chain's own estimate).
* DBs 164 GB + pixels ~84 GB + P6 ~34-60 GB + twin parts 18 GB + logged parts: **the volume quota is UNVERIFIED** (an empty MooseFS volume; `df` is blind to it, TRAPS).

## 5. The cheap discriminating experiment, banked in the repo (not yet run)

`R/raw/2026-10-17-twin-bank-3c/results/controls/pod/{bank_i3,held,merged}/` = the 74-twin mini bank, its 74 audit lines, its 23 kept rows and the merged bank (sha256 `fd4ff8ea389f2135358c024cd2281ba467ac789e0ce9f28839983ffbf463c164`, bank_i3 `94e0280d…`). Rebuild its 148 parts (74 twin + 74 logged; the parts need only the ~10 logs those twins sit in) with the frozen D7 tree and re-roll the 74 twins with the frozen 3-C code, then compare `audit.jsonl` (`f_at_s`, `d_twin_at_s`, `commit`) and `rows.jsonl` byte-for-byte with the banked ones.
* **bit-equal** => the CPU arithmetic reproduces across the two pods (n = 74 twins, 23 kept rows, ~2,800 floats): launch the 99-CPU-h job with confidence.
* **any difference** => STOP before the 99 CPU-h; hand back to the MM with the differing fields. Cheapest alternatives then (each a PI/MM decision, none built by me): (a) restart P6 from step 0 on a freshly built bank (the run's resume identity is content-based, `train.py:993-996`, and will refuse a bank whose `targets_sha256` differs from `8775e73e…`); (b) a pre-registered relaxed identity if the diff is a handful of rows; (c) resume on the pre-3C bank is refused by the same identity check.
It needs the DBs of ~10 logs, the maps and the teacher env only: **no HF token.**

## 6. The exact rebuild order (for the HANDOFF)

1. env (teacher venv + maps) -> `weights` -> DBs -> navlogs -> (pixels, last): all token-free, via the chain's own scripts (`pod_teacher_env.sh`, `pod_train.sh STAGES=weights`, `pod_dataprep.sh STAGES=dbs|pix`).
2. **74-twin control (section 5) -> go / stop.**
3. TOKEN: chain with `P6_STOP_AFTER=cache`, **without `P6_CBANK`** (so `commands` may build `train_grow_cmd`: the chain refuses `commands` when `P6_CBANK` is set), `P6_D7_FALLBACK=1`. Verify rank0/rank1 shas (section 1).
4. Frozen D7 tree + 3-C code on the pod; `build_twin_parts_3c.sh`; `run_full.sh launch` (720 shards); `merge`.
5. Verify `2a63acc2…`, `daa23b96…`; then the chain WITH `P6_CBANK`; verify `targets_sha256 8775e73e…` (trainer preflight) and ARGS sha `570619873b1ebaa0` (pin `P6_LABEL_WORKERS=6`: the ARGS string carries `--label-workers`, derived from the core count, 6 on the old pod; MEASURED `chain.log` 2026-10-18T09:43 and 2026-10-20T05:44).

## 7. The new pod, MEASURED 2026-10-21 (changes the cost line of section 4)

* **CPU quota is LIMITING: cgroup v1 `cfs_quota 765000 / period 100000` = 7.65 CPUs** (`nproc` says 96). Cgroup RAM limit 49,999,998,976 B (~50 GB; `free` shows 503 GB). The brief's "quota not limiting" does not hold. CPU model Intel Xeon Gold 6342 @ 2.8 GHz (Ice Lake; avx2, fma, avx512f, avx512_vnni; NO avx512_bf16 / amx). The OLD pod's CPU model is unrecorded, so same-microarchitecture cannot be asserted either way.
* Volume: a 60 GiB sequential write (1-GiB `dd` pieces, fsync) succeeded at ~125 MB/s; the quota beyond 60 GiB is UNVERIFIED (`df` is blind: 756T/466T/290T cluster numbers). The chain's own `dd` test (`P6_DD_GB=16`) is the later gate.
* Therefore (ESTIMATED from the section-4 CPU-hours at 7 parallel workers): D9 logged parts ~3.4 h, twin parts (44,409) ~5.4 h, **re-roll ~14 h**, i.e. **about 23 h of CPU-bound work before the 3-C bank exists**, with the A40 idle throughout, then the chain's lanes / lab / mem / egr / gate (~1-2 h) and the resume. The 74-twin control (section 5) costs ~15-25 min and decides whether those 23 h are worth spending.
