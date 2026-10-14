# P6 on the 11-vCPU pod: feasibility arithmetic, a CPU probe, and a plan that stops the pod idling

PI, 2026-10-14, verbatim: *"Find a solution with our 11 cpu pod."*  Then: *"The pod is idling, this is not acceptable."*
Author: Research Lab subagent for the Master Mind.  **Master Mind, 2026-10-14:** the three patches of section 11 are APPLIED (`refe/p6_pod.sh`, `refe/test_p6_chain.py`, `refe/build_metric_cache.py`, whose 7-line hunk is only the `REFE_PAUSE_FILE` pause point the overlap guard needs). Re-run on the dev box: `test_p6_chain.py` ZZP6CHAIN_PASS 341, `test_p6_overlap.py` 51, `test_p6_cpu_probe.py` 111, `test_flock_rw.py` 59, `test_launch_gate.py --static` 552. The chunk-2 and `StateIndex.size()` levers (section 7) are NOT code here: they wait for the early G-LIVE-LAB reading. The disk (section 5) is the open blocker, a PI decision.
No gate value was changed or proposed for change (section 8 puts the one real decision to the PI, with arithmetic).

Evidence classes on every number: **MEASURED** (ours, artifact path) / **PUBLISHED** / **INHERITED** (another agent or document, not re-verified) / **ESTIMATED** / **UNVERIFIED**.
All dev-box seconds are on an i9-12900F thread; a pod vCPU is slower by a ratio `r` that **nobody has measured yet** (section 3.3).  Everything below is written so that `r` is the one unknown.

---

## 1. The answer

1. **CPU.** P6 as written clears its gate on 11 vCPU **iff the pod's core is no more than about 1.6 to 1.8 times slower than the dev-box thread** (`r` below). Concretely the pod must achieve, with 7 label workers:
   * **per-set wall `c` <= 1.04 s** (6.735 sets/s, the gate's pool-alone bar) and **<= 0.937 s for a 4-set micro-batch** (the 5 % stall bar), and the need alone (4.49 sets/s) allows 1.56 s;
   * in dev units: 0.585 s per set (pool alone) to 0.65 s (beside a trainer) today. Break-even `r` = **1.77 to 1.78** (gate throughput, quiet box), **1.60 to 1.67** (beside a trainer stand-in), **1.66** (5 % stall, one process per set), **1.80** (5 % stall, sets split in two). Below `r` = 1.3 it clears with no stall (0.04 %); at 1.9 it stalls 7 to 10 %; at 2.1 it stalls 18 to 20 % (MEASURED, section 3.4).
   * **The binding constraint is per-set LATENCY, not throughput.** More vCPU do not shorten one set; the levers that do are splitting a set into two jobs and the free hot-path fix (section 7).
2. **The 11-vCPU question gets its first answer in the first half hour, not after D9** (section 9): an early G-LIVE-LAB reading, beside the P2 seed replicate's trainer, written to `lab_early.json`. `refe/p6_cpu_probe.sh` is **no longer a separate window**; it stays as the diagnostic for the MARGINAL/SHORT branch.
3. **The pod stops idling** (section 9): P6's CPU-only stages (D6, D9, D7, G-LIVE-LAB: verified GPU-free, stage by stage) run **beside** P2-s1's training, niced, capped to the cores the trainer leaves, guarded by the trainer's own step time, and the chain **waits** before its first GPU stage (G-LIVE-MEM). Patch `raw/2026-10-14-p6-11vcpu/PATCH_p6_pod.diff`, not applied; `test_p6_chain.py` 341 PASS on the patched copy (baseline 204).
4. **Disk is the harder blocker.** P6 adds **44 to 85 GB** (34 to 60 GB with `P6_FREE_NAVLOGS=1`) to a pod that has about **35 to 43 GB** free after S1 and the replicate (ESTIMATED). It needs **>= 49 to 57 GB cleared first in the high case (24 to 32 GB with the metadata hunk; central case 21 to 29 / 11 to 19)**: S1 / RL-S1 checkpoints after their HF egress is verified (about 24 GB) plus the metadata hunk (8 to 25 GB) cover 32 to 49 GB, so P2-only trees must go too, section 5. The chain's own 96 GB write test (72 GB with the hunk) will refuse until that is done, which is the point.
5. **RAM is not a blocker** (peak anonymous memory about 15 to 18 GB of 71 GB, section 4); the trainer host RSS of P6 itself is UNVERIFIED.
6. **Time.** Labeller keeps up: 6.8 days. Throughput-limited at `R` sets/s: `6.8 x 4.49 / R` days (4.0 -> 7.6, 3.0 -> 10.2, 2.0 -> 15.3). Stall `s`: `6.8 / (1 - s)` (5 % -> 7.2, 10 % -> 7.6, 20 % -> 8.5), section 6.

**Next unblocked actions, in order:** (a) clear the disk (Master Mind, 0 GPU); (b) apply the three patches + ship `p6_overlap.py` / `p6_cpu_probe.py` and re-pin `SHIP_MANIFEST.json` (Master Mind); (c) start P6 once P2-s1 logs `ZZP2REP_TRAIN_START`; read `ZZP6_EARLY_LAB` in `chain.log` 15 to 30 minutes later; (d) decide by section 7 / 8.

---

## 2. Inputs and method

| fact | value | class and source |
|---|---|---|
| pod CPU quota | cgroup v1 CFS 1105000/100000 = **11.05 vCPU** (`nproc` says 96) | MEASURED by the Master Mind 2026-10-11, `eval/P2R_POD_RUNBOOK.md` section 6 |
| pod memory limit / volume | 71 GB; `/workspace` 373 GB NFSv4 volume (`dd` 8 GB at 1.2 to 1.3 GB/s); a 500 GB container disk that a restart wipes | same |
| chain workers | `LW = CORES - 4 = 7`, `CW = CORES = 11` (`refe/p6_pod.sh:124-125`); it logs a WARNING below 12 | read from source |
| gate bars | pool-alone throughput >= 1.5 x 4.49 = **6.735 sets/s** at the launch worker count; GPU stall <= **5 %** of a window (max over windows after the first, in the SMOKE run) | `refe/gate_checks.py:1151,1154-1180` |
| step | 57.0 s per 256-sample step = 64 micro-batches of 4 -> **P = 0.8906 s**; 10,075 steps = 6.65 days | SPEC_P6 0.2, section 3 |
| loop | `F(k)` -> `submit(k)` -> `collect(k-1)` (= the GPU stall) -> `B(k-1)`: the labels of k-1 get one period P; <= 2 micro-batches x 4 sets = **8 sets in flight**, so workers beyond 8 add nothing | read from `train.py` / `diag_p6_labeller_identity.py:57-66`; forward share 1/3 ESTIMATED |
| P2-s1 | `--workers 2`, OMP 4, `--log-every 1`, about 60 s per step, 1,209 steps, about 20 to 21 h | `refe/p2rep_pod.sh:46`, `eval/P2REP_POD_RUNBOOK.md:75` |
| GPU idle seen | about 11.5 h overnight (RL-S1 CPU-only labelling) + about 28 min between consecutive arms | INHERITED (Master Mind brief 2026-10-14) |

**Method.** The real `OnlineLabeller` (spawn `ProcessPoolExecutor`, D9-format cached parts) on **11 logical CPUs of the i9-12900F** (threads 0-7 and 12-14: 5.5 P-cores' SMT pairs, so one vCPU = one hyperthread as on a cloud host), alone or beside a trainer stand-in (1 spinning process = the trainer main thread, 2 loader stand-ins at 0.25 core each; `bench_loader.json`: 0.056 CPU-s per 4-camera sample). Modes: `thru` (flat out), `latency` (idle pool, one micro-batch at a time), `paced` (the trainer's clock: one micro-batch per period, stall = time the "GPU" waits in `collect`). Parts are 240 of M-LAB's navtest parts (`raw/2026-10-11-m-lab/plan/cached_r1.json`), not navtrain. `REFE_SIM_HZ` is irrelevant to D9 and the labeller (10 vs 20 Hz: 0.696 vs 0.695 CPU-s per set, `simhz/*.json`). Scripts: `raw/2026-10-14-p6-11vcpu/{emu_driver,emu_burner,make_emu_cache}.py`; results `emu/*.json`, `emu.log`, `emu2.log`, `emu3.log`.

**Caveats, all of them.** The dev box is shared: every arm records the background load inside its CPU set (`bg_cores_in_affinity`, 0.4 to 2.8 cores); the same arm repeated under other users' load moved **11.96 -> 8.09 sets/s (-32 %)** (`thru_w7_omp4` vs `rep2_thru_w7_omp4`), so every dev number is good to about +-25 % and every break-even `r` to about +-0.3. One run per arm. A spinner is not a trainer. Hyperthread pairing is assumed, not known, for the pod.

---

## 3. CPU

### 3.1 What binds
At the need the pool does 4.49 sets/s x 0.47 to 0.51 CPU-s x `r` plus the trainer (1.0 main + 0.24 `r` loaders + about 0.5 to 1.0 other): about **5.7 to 6.2 cores of 11.05** at `r` = 1.7. The quota is half used on average; what fails first is **one set taking longer than the period** (L > P = 0.891 s), then SMT contention among 7 workers + trainer near saturation. `nice`, caps and OMP do not change L.

### 3.2 Largest per-worker wall `c` that passes (formula: `R = min(W,8) / c`)
| label workers | passes the need (4.49) | passes the gate (6.735) | latency (stall <= 5 %) |
|---|---|---|---|
| 7 (the chain's default) | c <= **1.56 s** | c <= **1.04 s** | micro-batch service <= **0.937 s** |
| 8 | <= 1.78 s | <= 1.19 s | same |
| 9 or more | <= 1.78 s (8 in flight) | <= 1.19 s | same |

Dev `c` = `W / R`: **0.585 s** (W7 alone, `thru_w7_omp4` 11.96 sets/s), **0.62 to 0.65 s** (beside the stand-in, 10.78 / 11.23), 0.79 to 0.87 s (neighbour load bg about 2, `rep2_*`). CPU-s per set 0.47 to 0.51 (the gap is SMT dilation, 1.24x).

### 3.3 `r`, the unknown
No pod CPU model is recorded anywhere (the INHERITED relabel rate even says "CPU count not recorded"). The only pod rate we have is the P2 relabel: **212 labels/min on 11 workers** with the pixel extraction running beside it (`eval/P2R_POD_RUNBOOK.md` section 6) = 3.11 s per set per worker, against 0.804 CPU-s for the same build+label path on a quiet pinned P-core (`raw/2026-10-11-m-lab/all_arms_table.txt`, `build_r1p_n1`): **r <= 3.9, an upper bound only** (the pod was also extracting and reading NFS). The break-even is 1.6 to 1.8, so this neither passes nor fails P6. `p6_cpu_probe.sh` and the early lab read it directly.

### 3.4 Dev-box results (MEASURED, `raw/2026-10-14-p6-11vcpu/emu/`)
Isolated pool (the gate's own condition), sets/s, OMP 4 / OMP 1:

| W | 7 | 8 | 9 | 10 |
|---|---|---|---|---|
| alone, quiet | 11.96 / 11.90 | 12.38 / 11.20 | 12.10 / 11.09 | 12.39 / 11.00 |
| beside the trainer stand-in | 10.78 / 11.23 | 10.37 / 9.28 | 11.14 / - | 8.88 / - |

Throughput is **flat from W7 to W10** (the box saturates at 5.5 physical cores; 8 sets in flight) and **OMP 4 vs 1 is a no-op** (differences of -5 % to +12 %, both signs, inside the repeat noise): the chain's `OMP_NUM_THREADS=4` oversubscribes nothing measurable, **no OMP hunk is warranted**. W8 and W10 beside the trainer are worse than W7 (9.3 to 10.4, 8.9): do not raise `P6_LABEL_WORKERS` for speed.

Paced at the trainer's clock scaled to a pod `r` times slower (W7, trainer stand-in, 400 sets), **stall fraction**:

| r | 1.0 | 1.3 | 1.6 | 1.9 | 2.1 |
|---|---|---|---|---|---|
| one job per set | 2.5 % (bg 2.5) | **0.04 %** | **3.6 %** | 10.2 % | 19.5 % |
| set split in two | - | 0.02 % | 0.4 % | 7.4 % | 17.8 % |
| W8, one job / two | - | - | - | - | 28.4 % / 20.1 % (bg 1.6) |

Interpolated 5 % crossing: **r = 1.66** (one job per set), **r = 1.80** (two jobs). Idle-pool service latency of a 4-set micro-batch at W7: p50 0.593 s, p95 0.70 s (bg 2.8): `0.937 / 0.593 = 1.58`, `0.937 / 0.70 = 1.34`. Gate throughput break-even: `R_dev / 6.735` = 1.78 (quiet), 1.60 to 1.67 (beside), 1.20 to 1.32 (neighbour load). **All of these agree on 1.3 to 1.8.**

### 3.5 Oversubscription: what runs where
Trainer main 1 core + `OMP=4` torch threads (mostly idle) + 2 loader workers + 7 labeller workers (each with about 11 idle threads, `threads_worker_max`) + the labeller parent + the watcher. CFS throttling (`nr_throttled`) **cannot be seen on the dev box**: UNVERIFIED. The probe reads `cpu.stat` per pool run and reports `throttled`; the overlap guard makes throttling visible through the trainer's step time. `cv2` sizes its thread pool from the 96 visible cores; `bench_loader.json` shows no gain from it (0.0557 vs 0.0568 CPU-s per sample), so it is not a lever.

### 3.6 D9, the largest new CPU job
Per token on dev: **0.425 CPU-s** for the parts (`simhz/build_r1_hz10.json`, pinned quiet P-core, MEASURED); M-LAB estimates 0.85 CPU-s per scene all-in (ESTIMATED). Tokens: 103,037 plain, up to 189,888 with every twin (exact count: the probe's bank scan). Wall = `tokens x c x r / (workers x 0.9)` (0.9 pool efficiency ASSUMED):

| tokens, c | r = 1.3 | r = 1.7 | r = 2.1 |
|---|---|---|---|
| 103,037 x 0.43, 7 workers (overlap) / 11 (alone) | 2.5 / 1.6 h | 3.3 / 2.1 h | 4.1 / 2.6 h |
| 103,037 x 0.85 | 5.0 / 3.2 h | 6.6 / 4.2 h | 8.1 / 5.2 h |
| 189,888 x 0.43 | 4.7 / 3.0 h | 6.1 / 3.9 h | 7.6 / 4.8 h |
| 189,888 x 0.85 | 9.3 / 5.9 h | 12.1 / 7.7 h | 14.9 / 9.5 h |

ESTIMATED throughout. Every cell fits inside P2-s1's 20 h except the corner (all twins, 0.85, r = 2.1, 7 workers: 14.9 h plus D6 and the lab). D9 is resumable (existing parts are skipped; writes are `tmp` + `os.replace`).

---

## 4. RAM against the 71 GB cgroup

| consumer | GB | class |
|---|---|---|
| a trainer beside (RL arm) | about 4.5 RSS + about 10 page cache | MEASURED by the Master Mind (brief); P2-s1 itself UNVERIFIED |
| D9 / D7 workers | 7 x 0.47 = 3.3 (peak 0.474 GB per builder) | MEASURED dev, `simhz/*.json` `mem_end` |
| G-LIVE-LAB / early lab pool | 7 x 0.28 + parent = about 2.3 (`rss_tree_sum` 1.8 to 1.9) | MEASURED dev, `emu/*.json` |
| D6 annotate (bank rows) | <= 8 (banks about 7.8 GiB at full scale) | UNVERIFIED whether it streams |
| page cache from reading 176 GB of DBs and writing 27 to 50 GB of parts | grows to the limit, **reclaimable**; cgroup-v1 `usage_in_bytes` counts it (`TRAPS.md`): do not read it as pressure | INHERITED |

Peak anonymous memory beside P2-s1: about 4.5 + 3.3 + 2.3 = **10 GB** (18 GB if the annotate step holds the whole bank). **Not a blocker.** The patch gives D9, D7, annotate and the lab `oom_score_adj` 1000, so a memory surprise kills P6's child, never the trainer. UNVERIFIED: P6's own host RSS at steady state (bank + 2 loader workers + 7 labellers + CUDA context); read `python refe/p6_cpu_probe.py cg` during the first P6 hour.

---

## 5. Disk (the real blocker)

**Already on the pod, reused in place** (`have_bank || fetch_bank`, `have_dbs`, `have_pix`; weights are hashed and kept): DBs 163.82 GiB (175.9 GB, MEASURED 2026-09-23, `code/pod_dataprep.sh:10`), pixels about 84 GB (MEASURED on shard 25, `:14`), banks, venv, backbones. P2's peak is about **310 GB** (ESTIMATED, `P2R_POD_RUNBOOK.md`) of 373 GB: 63 GB free at P2's peak, **about 39 GB after S1's 24 GB of checkpoints, about 32 to 35 GB once P2-s1 has written its run dir** (ESTIMATED; GB/GiB not reconciled).

**What P6 writes or fetches** (GB; only the first row crosses the network, everything else is local):

| stage | what | low | high | class |
|---|---|---|---|---|
| D6 | NAVSIM metadata `.tgz` transit (then deleted) | 7.05 | 7.05 | MEASURED HEAD (`p6_pod.sh:37`) |
| D6 | extracted pickles under `navsim_logs/` | 8 | 25 | ESTIMATED: navtest is 1.1 GB for 147 logs (MEASURED here) x 1,192 to 1,310 logs; 25 = a 3.5x gzip expansion |
| D6 | `train_grow_cmd` (the bank + commands) | 0.6 | 1 | ESTIMATED: 3.36 kB/row (pilot bank, MEASURED) x 189,888 |
| D9 | parts, plain tokens | 27.0 | 27.0 | ESTIMATED: 103,037 x 262 kB (navtest mean, M-LAB) |
| D7 | lane-rank twins | 0 | 22.8 | upper bound = every twin tuple (86,851) as a lane twin; the probe's scan gives the exact count |
| 8-12 | `mem_run`, `lg`, `gate_bank`, `egr` scratch | 1 | 2 | ESTIMATED |
| 13 | run dir: `ckpt_last` 0.23 x 2 + 25 snapshots + `model_final` 1.29 | 4 | 7.5 | `SPEC_P6` 223-224 |
| total persistent | | **44** (34 with `P6_FREE_NAVLOGS=1`) | **85** (60) | ESTIMATED |

Beside P2-s1 add its own growth, about 7 GB. The `dd` write test in the patch is therefore **96 GB by default, 72 GB with `P6_FREE_NAVLOGS=1`** (the high case + that + a margin); the probe's exact bank scan lets the Master Mind set `P6_DD_GB` tighter. The old 8 GB proved nothing about P6.

**Deficit and what to clear first** (in this order; each only after its result is verified on HF, never on the pod's word):
1. **S1 and RL-S1 checkpoints, about 24 GB** (egress verified by the remote listing, then delete locally).
2. **`P6_FREE_NAVLOGS=1`** (patch hunk): deletes the metadata tgz tree and `navsim_logs/trainval` right after `commands.json` exists (frees 8 to 25 GB; a rebuild of the `commands` stage re-fetches it, 7 GB).
3. P2-only trees P6 never reads: the on-policy sets / relabel output under `refe_navtrain/onpolicy` and P2's run directories (size UNVERIFIED: the probe's `du` lines in `disk.txt` give it exactly).
4. **Not deletable:** the DBs (D9, D7 and the lab instrument read them; `inputs` re-downloads them on every chain start if fewer than `P6_EXPECT_DBS` are present, `p6_pod.sh:186-190`), pixels, `train_grow`, backbones, maps, `teacher_env`.

Need (P6 high case + P2-s1's growth = 92 GB, or 67 GB with the hunk) against 35 to 43 GB free: **deficit 49 to 57 GB, or 24 to 32 GB with the hunk** (central case: need 64 / 54 GB, deficit 21 to 29 / 11 to 19). Items 1 + 2 clear 32 to 49 GB, so item 3 is needed in the high case (ESTIMATED). **What P6 still fetches given the pod:** only the 7.05 GB metadata archive (about 1 min at the INHERITED 118 MB/s). If P2's DBs or pixels were deleted by a cleanup, the `inputs` stage re-fetches 97.57 GiB / 304.5 GB: do not.

---

## 6. Time

| situation | P6 wall |
|---|---|
| labeller keeps up (stall about 0) | **6.8 days** (6.65 x 1.02 checkpoint overhead) |
| throughput-limited at `R` sets/s | `6.8 x 4.49 / R`: R=4.0 -> 7.6 d; 3.5 -> 8.7; 3.0 -> 10.2; 2.5 -> 12.2; 2.0 -> 15.3 |
| latency-limited, stall `s` | `6.8 / (1 - s)`: 3.6 % -> 7.05 d; 5 % -> 7.16; 7.4 % -> 7.34; 10 % -> 7.56; 17.8 % -> 8.27; 19.5 % -> 8.45; 28 % -> 9.44 |
| prep before step 0, overlapped | D6 + D9 + D7 + lab hide inside P2-s1's 20 h (section 3.6); the GPU stages after the wait (mem, egr, launch gate, preflight) remain about 30 min (the measured gap between arms is 28 min, INHERITED) |
| prep before step 0, serial (today) | + 2 to 16 h of pod idling (D6 + D9 + D7 + lab, section 3.6) |

---

## 7. Ranked levers

| # | lever | effect | cost / class |
|---|---|---|---|
| 1 | **Read `r` now**: early lab (in the patch) or `p6_cpu_probe.sh` | turns the one unknown into a number before any hour is spent | free; section 9 |
| 2 | **Disk**: S1/RL-S1 egress + delete, `P6_FREE_NAVLOGS=1` | 34 to 49 GB | 0 GPU; the PI's pod, Master Mind executes |
| 3 | **Split each 64-proposal set into 2 jobs** in `OnlineLabeller.submit` (labels bit-identical: max abs dPDMS 0.0, `raw/2026-10-11-m-lab/chunk_probe.json`) | latency halves; break-even r 1.66 -> **1.80** (+8 %); CPU +8 % (isolated 10.78 -> 9.90 sets/s) | about 30 lines + test + gate re-pin; Master Mind decision; MEASURED emulation |
| 4 | **Free hot-path fix**: cache `StateIndex.size()` (it rebuilds `dir()` 160,080 times per 60 sets) | **-8.5 % CPU per set, bit-identical** (0.2718 -> 0.2487, 6 x 40 sets, `prof_patch_ab.json`); about +9 % on every break-even | vendored `navsim_pdm` edit + gate re-pin; Master Mind decision; MEASURED |
| 5 | 3 + 4 together | break-even about **1.96** (product, not measured jointly) | |
| 6 | more vCPU | fixes utilisation (SMT), **not latency**; needed together with 3 + 4 if `r` >= 1.9 | the PI provisions and pays |
| 7 | accept a stall above 5 % | time `6.8/(1-s)` | a gate value: section 8 |
| 8 | `P6_LABEL_WORKERS`, `OMP_NUM_THREADS`, `REFE_SIM_HZ`, cv2 threads | **no effect** (MEASURED dev) | do not spend time here |
| 9 | D9 on another box | none: it already hides inside P2-s1; the dev box has no navtrain cache (M-LAB) and a 1.2 MB/s uplink (INHERITED) | rejected |

Follow-up, not built: pre-run the CPU-only checks of the launch gate (G-HYG, G-DVB, G-SUITE, G-CKPT, if they need no GPU: UNVERIFIED) during the overlap, to shorten the 28 min of gating after the wait.

---

## 8. For the PI (decisions, with arithmetic; no gate value is changed here)

* **If the early lab or the probe lands at `r` <= 1.6: nothing to decide.** P6 runs as written.
* **If `r` is 1.7 to 1.9:** the gate's 5 % stall bar fails by a few percent (3.6 % at 1.6, 7 to 10 % at 1.9). Apply levers 3 and 4 (Master Mind), re-read.
* **If `r` >= 1.9** even with levers 3 and 4: three honest options, none of them mine to choose:
  (a) provision more vCPU **and** apply 3 + 4 (more vCPU alone does not shorten a set);
  (b) accept a stall bar of `X %`: the cost is `6.8/(1-X)` days, +0.7 d at 10 %, +1.7 d at 20 % (section 6); the gate's `LAB_STALL_MAX` is the PI's;
  (c) accept a throughput margin below 1.5x: the margin exists because co-locating the trainer costs the pool 6 to 10 % (11.96 -> 10.78 to 11.23 sets/s), and the bar is measured pool-alone, so lowering it to 1.0x (4.49) leaves no room for that; I do not recommend it and have not touched it.
* Under no outcome is P6 a refutation: it is either FIT, or MARGINAL with named levers, or SHORT with each blocker named (compute: the PI; code: Master Mind).

---

## 9. Overlap: P6's CPU prep during P2-s1's GPU training

### 9.1 Which stages are CPU-only (verified, not assumed)
| stage of `p6_pod.sh` | GPU? | evidence |
|---|---|---|
| pod checks, `dd` | no | |
| env (`pod_teacher_env.sh`) | **has a CUDA smoke test** (`:66-72`) but is skipped when `teacher_env.sh` exists (P2-s1's pod has it) | read |
| watcher, weights (`curl` + `sha256sum`), recover (HF) | no | `pod_train.sh:84-101`; `p2r_sync.py` |
| inputs (`have_* \|\| fetch_*`) | no | |
| D6 `commands` (`p6_prep.py`, `annotate_commands.py`) | no | no `torch`/`cuda` in either |
| D9 cache, D7 lanes (`build_metric_cache.py`) | no | no `torch`/`cuda` in it, `navsim_pdm_targets.py` or `online_labeller.py`; NAVSIM may import torch, which creates no CUDA context |
| G-LIVE-LAB (`diag_p6_labeller_identity.py`) | no | same |
| **9 G-LIVE-MEM (`train.py $MEM_ARGS`)** | **YES: the first GPU stage** | confirmed |
| 10 G-EGR (`hf_egress.py`, imports torch at `:250`), 11 launch gate, 12 preflight, train | yes (after 9) | |

### 9.2 Design (all in the patch, tested)
* **Admission.** `P2`, `RL-S1`, `S1` locks must be free (probes `( flock -n 9 ) 9<> file`, `test_flock_rw.py` passes). The replicate's lock `refe_p2rep/chain.lock` may be held only if its `chain.log` carries **`ZZP2REP_TRAIN_START`** (its own gates measure this pod: P6 does not start beside them). `P6_OVERLAP=0` restores the plain refusal.
* **Cores.** Pool size for D9 and D7 = `floor(11 - burn - 1)` where `burn` is the **cgroup's measured CPU rate over 60 s** (`p6_overlap.py overlap-cap`; the cgroup is the pod, so the burn is the trainer's). Expected **burn 3.0, workers 7** (1.0 main + about 0.4 loaders + 1.5 other, ESTIMATED); the chain logs the measured value, falls back to `floor(11/2) = 5` with "UNVERIFIED" if the counter is unreadable, `P6_OVERLAP_WORKERS` fixes it. Everything `nice -n 19`; the lab instrument keeps its 7 workers (the gate requires the launch count) and runs about a minute.
* **Memory.** `oom_score_adj` 1000 on D9, D7, annotate and the lab pool; peak about 10 to 18 GB (section 4).
* **GPU.** `CUDA_VISIBLE_DEVICES=""` for every CPU-only stage, restored right before the wait ends (tested: the first GPU stage sees the original value again).
* **NFS I/O.** The 96 GB write test is done in 1 GiB pieces with a 1 s pause; D6/D9 are throttled by the guard below. Bandwidth is no issue (the trainer reads about 2 MB/s of JPEGs against 1.2 GB/s); queueing at the server is what the guard watches.
* **The guard (the "measure before and after, back off above 5 %" rule).** Baseline = median of the trainer's last 5 per-step times from its `metrics.jsonl` before P6 puts any load on the pod; every `P6_GUARD_POLL_S` (300 s) the same median is compared; **two consecutive reads above +5 %** PAUSE the running stage (its workers sleep per token while `overlap.pause` exists; `build_metric_cache.py` patch) for `P6_GUARD_PAUSE_S` (600 s, doubling to 1 h while it keeps happening). No readable baseline = **REFUSED** (the replicate cannot be protected), unless the Master Mind sets `P6_OVERLAP_UNGUARDED=1`.
* **The wait** before stage 9: no chain lock held (P2, RL-S1, S1, P2-s1), `ZZP2REP_DONE` in `refe_p2rep/chain.log` (when a replicate ever ran; gone without it = REFUSED, `P6_P2REP_ABANDONED=1` only on the Master Mind's word), and `nvidia-smi` empty on 3 consecutive polls of 10 s (an `nvidia-smi` that cannot answer counts as busy; `P6_WAIT_MAX_S` bounds it). Median idle added: about 30 s.

### 9.3 The early G-LIVE-LAB, and how to read it
Right after D6 the chain builds D9 parts for the **first 3 logs** (about 250 parts, minutes; the full build skips them), runs the labeller instrument on them at the launch worker count **beside P2-s1's trainer**, and logs `ZZP6_EARLY_LAB ZZLAB_BAR ok|below|bad thr=... bar=6.735`. It writes `lab_early.json`, **never** `lab.json`; only an identity failure refuses (a defect at any speed).

* It is **pessimistic for the gate's quantity** (the gate measures the pool alone) and **close to production co-location** (P6's own trainer burns about what P2-s1's does). Dev: beside-trainer / alone = 0.90 to 0.94.
* **Read it so:** `thr >= 6.735` -> the gate's throughput half passes with margin, utilisation at the need <= 67 %, expected stall <= about 4 % (dev sweep at r = 1.6). `6.2 <= thr < 6.735` -> alone it likely clears; the chain re-measures after the wait automatically. `4.49 <= thr < 6.2` -> MARGINAL: section 7. `thr < 4.49` -> SHORT.
* Mapping `thr` (pod, beside) -> dev-equivalent `r = 11.0 / thr`: 8.5 -> 1.3 (stall 0.04 %); 6.9 -> 1.6 (3.6 %); 5.8 -> 1.9 (10 %); 5.2 -> 2.1 (20 %); 4.49 -> 2.45 (SHORT). ESTIMATED from section 3.4.
* The full `lab` stage (after D9 and D7) is **kept** if its beside-the-replicate reading already clears the bar, **re-measured after the wait** if not (the gate then sees the pool-alone number).
* **It answers throughput and identity, not the stall.** The stall is decided at the launch gate's smoke run; the sweep above is the only predictor until then.

### 9.4 Is `refe/p6_cpu_probe.sh` still needed?
**Not as a separate window.** If the early lab reads >= 6.735, run nothing else. Keep the probe for the MARGINAL / SHORT branch (it alone gives the worker sweep 7 to 10, the OMP A/B, the paced stall, the exact D9 CPU-s per token, the exact twin count and a real `dd` of P6's need) in the gap between P2-s1 and P6 or any other quiet window; it refuses beside a trainer by design (a loaded cgroup voids every number).

### 9.5 Starting it
Prerequisites: disk cleared (section 5); the three patches applied and tested; `p6_overlap.py`, `p6_cpu_probe.py` shipped and `SHIP_MANIFEST.json` re-pinned. Then, once P2-s1's `chain.log` has `ZZP2REP_TRAIN_START`:
`P6_FREE_NAVLOGS=1 nohup setsid bash /workspace/refe-plan-p6/refe/p6_pod.sh > /workspace/data/refe_p6/stdout.log 2>&1 < /dev/null &` (with the chain's usual environment). Watch for, in `chain.log`: `ZZP6_OVERLAP on`, `ZZP6_OVERLAP_BASELINE`, `ZZOVERLAP_CAP`, `ZZP6_EARLY_LAB`, `ZZP6_OVERLAP_SLOW` / `ZZP6_OVERLAP_BACKOFF` (if the guard acts), `ZZP6_WAITING`, `ZZP6_WAIT_OK`. P2-s1's step time is the thing to watch: if the guard ever fires, report it.

---

## 10. The probe (`refe/p6_cpu_probe.sh`), pre-registered rules

Run: `P6CPU_PKG=/workspace/refe-plan-p6 nohup setsid bash /workspace/refe-plan-p6/refe/p6_cpu_probe.sh > /workspace/data/refe_p6cpu/stdout.log 2>&1 < /dev/null &`; read `probe.log` (`ZZP6CPU_*`) and `verdict.json`. **Runtime <= 60 min** (`P6CPU_BUDGET_MIN`; each stage skips itself, logged, when the budget is spent). CPU only; no GPU, no token, no network; never `pkill`; scratch removed.
It **refuses** (`ZZP6CPU_REFUSED`) if any chain lock is held (rls1, s1aux, refe_p2rep, refe_p2, refe_p6; the `9<>` form), the GPU is busy or cannot be asked, another probe runs, or the cgroup already burns more than `P6CPU_MAX_BG` (1.0) cores.
Stages: environment (cgroup cores, never `nproc`); one pass over the bank (the exact D9 workload and a whole-log sample); D9 on the sample into a scratch dir (CPU-s per token from the cgroup counter, extrapolated to the full bank); for W in 7 8 9 10 the gate's own instrument (identity must PASS) + the same pool instrumented + service latency + the trainer's clock; OMP 4 vs 1; chunk-2; `du` of what exists and a chunked `dd` of P6's need.
**Verdict (one line, rules fixed before the run):**
* `VOID` no measurement, or any identity / stale-refusal failure;
* `FIT` the default worker count (cores - 4) reaches >= 6.735 sets/s **and** its paced stall <= 5 %;
* `MARGINAL` not FIT, but some measured worker count reaches >= 4.49 sets/s;
* `SHORT` no worker count reaches 4.49.
`REFUSED` is separate (it did not run). Tests: `refe/test_p6_cpu_probe.py` `ZZTEST_PASS 111` (static properties + 25 mutations, helper units, the real script in a sandbox: FIT / MARGINAL / SHORT / VOID, every refusal).

---

## 11. The patches (not applied) and tests

| file | content |
|---|---|
| `raw/2026-10-14-p6-11vcpu/PATCH_p6_pod.diff` | (a) stage-0 write test sized for P6 (96 / 72 GB); (b) opt-in `P6_FREE_NAVLOGS=1`; (c) lock guard over P2 / RL-S1 / S1 with `9<>`; (d) overlap admission, CUDA hiding, baseline, caps, `nice 19`, `oom_score_adj`, the guard (`guard_watch`); (e) the early lab; (f) the wait before G-LIVE-MEM, the lab re-measure rule |
| `raw/2026-10-14-p6-11vcpu/PATCH_build_metric_cache.diff` | `_pause_point()`: the builder sleeps per token while `REFE_PAUSE_FILE` exists (no effect when unset) |
| `raw/2026-10-14-p6-11vcpu/PATCH_test_p6_chain.diff` | static properties + 25 mutations for the new behaviour; sandbox: every refusal, the overlap run with capped / niced / CUDA-hidden children, the guard actually pausing a stage, the wait end to end (lock released, GPU busy, then idle), the lab re-measure both ways, the chunked `dd`, the builder pause; 5 behavioural mutants |

Base files (the working tree of 2026-10-14): `refe/p6_pod.sh` sha256 `6130f11b6e4a9030...` = the `SHIP_MANIFEST.json` pin, md5 `b24920c69d78`; `build_metric_cache.py` md5 `57abd7825a5e`; `test_p6_chain.py` md5 `3127201c26cf`. Apply with `patch -p1` from `refe/` (not `git apply` under `core.autocrlf=true`, which rewrites the shell script to CRLF; the test insists on LF). Results on the patched copy: `test_p6_chain.py` **ZZP6CHAIN_PASS 341** (baseline 204; `raw/2026-10-14-p6-11vcpu/p6chain_baseline_PASS204.log`, `p6chain_patched_PASS341.log`), `test_flock_rw.py` ZZTEST_PASS 59, `test_p6_overlap.py` ZZTEST_PASS 51, `test_p6_cpu_probe.py` ZZTEST_PASS 111. A mutant without the metadata hunk goes RED (`p6chain_mutant_without_hunk_f_RED.log`). No CPU, OMP or worker-count change is in the patch: the measurements do not support one.
**Integration needs (escalated, not left in this document):** ship `refe/p6_overlap.py` and `refe/p6_cpu_probe.py` (the patched chain REQUIRES both) and re-pin `SHIP_MANIFEST.json` (which pins `refe/p6_pod.sh` today); chunk-2 and the `StateIndex.size()` cache are Master Mind decisions with a launch-gate re-pin.

---

## 12. Evidence ledger and what is NOT known

MEASURED here: emulation tables (section 3.4), hot-path A/B, loader cost, sim-Hz irrelevance, labeller RSS, builder RSS, navtest metadata size. MEASURED elsewhere: pod quota/RAM/volume (Master Mind), relabel 212/min, DB and pixel volumes, 262 kB per scene (navtest), chunk probe.
**UNVERIFIED, each with the thing that would settle it:** the pod core speed `r` and its SMT layout (early lab / probe `lscpu`); CFS throttling under the real mix (`cpu.stat` in the probe); P2-s1's CPU burn (the chain's census); P6's host RSS; the extracted metadata size (the chain logs it: `du` of `navsim_logs`); the exact twin count and so the D9 size (the probe's bank scan); whether `annotate_commands.py` streams; whether the launch gate's CPU-only checks can run early; every disk figure labelled ESTIMATED.
Raw data is under `raw/2026-10-14-p6-11vcpu/`. The emulation cache (240 parts) lives only in the session scratchpad; `make_emu_cache.py` rebuilds it from M-LAB's plan.
