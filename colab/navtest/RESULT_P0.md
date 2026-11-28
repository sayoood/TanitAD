# NAVSIM-on-Colab P0: snapshot 016 on sub200, gate by gate (2026-09-27)

**Scope.** Phase P0 of the PI-approved plan (`colab/NAVTEST_ON_COLAB_PLAN.md`, "Yes, start now", 2026-09-27):
reproduce snapshot 016's sub200 eval on Colab and test it against the plan's pre-registered parity gates (§f).
The thresholds below are the plan's, written before any VM ran. None was changed after the data came in.

**Evidence.** Raw evidence is in `colab/raw/2026-09-27-navtest-p0/`. The gate evaluator is
`colab/navtest/compare_p0.py`, and its output is `gates_p0.json`. Every number is MEASURED unless marked otherwise.
The references are the dev box's own ep016 artifacts in `D:/Projects/TanitAD/data/refe_navtest/`: the seam, the E-6
dump and table, and the per-token CSV. The point is **PDMS 76.5243**.

## Answer first

1. **The GPU stage passes all three of its gates on an L4.** This is the stage worth moving: it blocks the dev-box GPU.
   - **G-S PASS:** 0 pick flips, executed poses within **15 µm**, aggregate within 1.8e-6.
   - **G-P PASS:** PDMS **76.5243**, Δ = 0.0000, and every discrete sub-score is equal on every token.
   - **G-REP PASS:** 3 shards merge **bitwise** equal to 1 process.
   - **G-ENV PASS** on the L4.
2. **The CPU harness on Linux FAILS its exact gates, at the 1e-13 to 1e-9 level.**
   - **G-H FAILED** its bar of max |Δ| = 0.0:
     - on the CPU VM, **6.90e-13** on 2 of 1,400 cells (one token's EP and score);
     - on the L4 VM, **3.07e-13** on a *different* token.
   - **G-H′ (sub200 form) FAILED** its "CSV equal" half. The VM-built cache moves EP by up to 1.09e-9 on 120 tokens.
   - **G-E6 FAILED** its 0.0 bar: **37 of 12,800** table cells differ, all in EP, by at most **1.23e-11** in PDMS.
   - In every one of these failures, all discrete terms were equal, and the 4-dp PDMS and the floor means were
     identical.
3. **Diagnosis of the harness failure (MEASURED): the platform's floating-point libraries differ.**
   - This is not a different function.
   - 12 of 15 elementary operations hash differently between the dev box's Windows build and the VM's Linux build of
     the same pins: libm sin, cos, exp, log and arctan2; OpenBLAS dot and solve; and GEOS `project`.
   - The same VM reproduces itself bitwise, and two Linux VMs disagree with each other.
   - ⇒ **No environment setting can make a Linux VM bit-exact with the dev box.** libm and GEOS are fixed by the
     platform build. The one pinnable part, the BLAS kernel, was not the only difference, and pinning it was not run.
4. **The next lever was run, and it works: the hybrid route.**
   - Score the **Colab seam** with the **dev box's own harness**.
   - Result: PDMS **76.5243**, and every discrete term equal on all 200 tokens.
   - This keeps every CPU-stage number bit-identical to the banked history. It needs no new gate: the seam is qualified
     by G-S/G-P/G-REP, and the harness is the unchanged dev-box function.
5. **Cost: 0.41 CU of the 6 CU cap** (7 %). Session B (L4) cost **0.33 CU** and session A (CPU) **0.074 CU**. Both
   VMs were checked stopped on the server, which read `assignments: []` and rate 0.
6. **Decisions for the PI and orchestrator** (the end of this file):
   - P1 route: hybrid, or all-Colab with a re-registered harness tolerance.
   - Whether to adopt that tolerance.

## The gates (section f, pre-registered)

| gate | bar (as written before the run) | measured | verdict | evidence |
|---|---|---|---|---|
| **G-ENV** (L4) | `ZZCONVERTER_EXACTZZ` with the dev box's numbers; 4/4 JPEG hashes; 322,070,854 / 18,991,426 params; CUDA available; `ZZPULL_OK` | selftest exact (2.499e-02 / 1.875e-02 / 9.998e-03 m, 8.9e-16 rad, the dev box's values); JPEG **4/4**; params **322,070,854 / 18,991,426**, rule `navsim_v1`, `REPAIR_LAST_HEADING=True`; real CUDA conv2d on **NVIDIA L4** (sm 8.9), torch 2.7.1+cu128, cuDNN 90701, driver 580.82.07; relay pull **3/3 md5 OK** | **PASS** | `navtest-p0-l4/results/out/status.json` (`genv`, `relay`), `logs/genv_*.log` |
| **G-H** harness, exact | the dev box's ep016 seam + the dev box's cache, scored on Linux: every cell equal (**max \|Δ\| = 0.0**), 200/200 valid, C1–C3 pass, PDMS 76.5243 | CPU VM: max \|Δ\| **6.90e-13** (token `024d89a3e1e752dc`, EP and score; 1,398/1,400 cells textually equal). L4 VM: **3.07e-13** (token `03b33d7830ba522b`). Both: PDMS **76.5243**, 200/200 valid, C1 0.0, C2/C3 pass, C8 (navsim from `3e8291b`, nuplan from `ce3c323`) pass | **FAILED** (bar 0.0) | `navtest-p0-cpu/results/out/gh/`, `navtest-p0-l4/results/out/gh/` |
| **G-H′** VM-built cache (sub200 form) | CSV equal to G-H's; floors equal W3's banked per-token CSVs (means to 4 dp) | cache rebuilt for 200 scenes. Seam CSV vs G-H: max \|Δ\| **1.09e-9**, 120 tokens, **EP only**. Floors vs W3's per-token rows: CV 1.50e-10 (49 tokens), STOP 5.91e-09 (171), HUMAN 1.09e-09 (134), **EP only**. sub200 means equal to 4 dp: REFe 76.5243, CV 21.8222, STOP 62.5812, HUMAN 94.1195 | **FAILED** the exact "CSV equal" half; floors half **PASS** at 4 dp on sub200 | `navtest-p0-cpu/results/out/gh2/` |
| **G-E6** E-6 table, exact | the dev box's dump → 64 single-proposal runs; `pdms` and `sub` equal `table.npz` (**max \|Δ\| = 0.0**) | 64/64 runs PASS, every token valid (the VM's G3). **37 of 12,800 cells** differ, over 22 tokens, **EP only**: sub max 2.96e-11, PDMS max **1.23e-11**. NC/DAC/TTC/C/DDC equal in every cell. The VM's own G1 (dump vs landed seam) reads 0.0, and its G2 (table at the pick vs the landed CSV) reads 2.9e-13. Oracle **96.2258**, random 60.245431 and at-pick **76.5243** are identical to the dev box's, and the per-token best proposal is equal on 200/200 | **FAILED** (bar 0.0) | `navtest-p0-cpu/results/out/ge6/`, `ge6_residual_breakdown.json` |
| **G-S** seam, tolerance | 200/200 rows; frame control ≤ 1 mm; **no flip** on the 160 tokens with dev-box margin ≥ 1e-3 (flips below it counted); same-pick \|Δ pose\| ≤ 1e-3; \|Δ aggregate\| ≤ 1e-3 | 200/200 rows, token order equal, frame control **1.7e-14 m**; **0 flips** in total (including the 40 tokens under 1e-3); executed poses **1.53e-5 m / 1.67e-6 rad**; aggregate over all 64 proposals **1.85e-6**; all-proposal poses 1.91e-5 (reported); \|Δ logit\| **1.55e-5** | **PASS** | `navtest-p0-l4/results/out/seam/` |
| **G-P** point, tolerance | all picks equal ⇒ \|PDMS − 76.5243\| ≤ **0.01** and every discrete sub-score equal per token | PDMS **76.5243** (Δ **0.0000**); NC/DAC/TTC/C/DDC equal on all 200 tokens; max per-token \|Δ score\| 1.51e-7 | **PASS** | `navtest-p0-l4/results/out/gp/` |
| **G-REP** | seam in 1 process vs K shards on the same VM: bitwise | K=3 (31/31/31 logs, 71/68/61 rows): poses, proposals, logits and picks **bitwise equal**. Re-checked on the dev box from the downloaded npz | **PASS** | `navtest-p0-l4/results/out/rep/` |

**How the G-S bars were read.** The pose bar applies to the **executed** rows: the plan measured 0.53 mm on them, so
1 mm is about 2× headroom. The aggregate bar applies to **all 64 proposals**: the plan measured 1.1e-4 there, so 1e-3 is
about 9× headroom. Both readings are the plan's own, so this is not a choice made after the data. The measured values
sit 65× and 540× inside the bars. A GPU-to-GPU move of the same TF32 path (RTX 4060 → L4, both Ada) moved the model by
far less than the CPU-vs-GPU scale the plan had priced.

**Which variance this answers** (CLAUDE.md's three questions): none of them. The question here is *"is this the same
compute path?"* REFe's selection is an argmax, so the numbers carry no training, episode or inference-sampling variance.

## G-H failed: diagnosis, and the levers that were run

**What failed.** One cell out of 1,400 differed on each Linux VM, and it was a different token on each. The residual
sits only in EP, the one continuous term. EP is `agent progress / PDM-Closed progress`, where the PDM-Closed reference is
re-simulated at scoring time: an LQR tracker, a kinematic bicycle model, and progress measured along the centerline with
shapely.

**Lever 1: is the VM deterministic?** Yes. The same dev-box seam and cache were scored again on the same CPU VM
(`refe_ghx_default`), and the result was **bitwise identical to that VM's G-H run**. It differed from the dev box in the
same 2 cells. Evidence: `navtest-p0-cpu/results/out/ghx/`.

**Lever 2: is it the platform's float libraries?** Yes, MEASURED. `ufunc_probe.py` hashes 15 operations on fixed inputs.
It was run in the dev box's `navsim-crun` venv and in the VM's rebuilt venv: numpy 1.23.4 and shapely 2.0.7 / GEOS
3.11.4 on both.

- **12 of 15 differ:** `sin`, `cos`, `tan`, `arctan2`, `exp`, `log`, `hypot`, `power`, `arcsin` (libm: MSVC UCRT vs
  glibc), `dot` and `solve` (OpenBLAS kernels: i9-12900F vs AMD EPYC 7B12), and `shapely.project` (GEOS built by MSVC
  vs GCC).
- **3 are equal:** `sqrt` (IEEE-exact), `cumsum`, `matmul`.
- Evidence: `ufunc_probe_compare.json` and `ufunc_probe_{devbox,vm_cpu}.txt`.

**Lever 3: pin the kernels.** This was not run: a quoting defect in `gh_lever.py`'s second arm (`env: 'AVX512CD': No
such file or directory`) left `refe_ghx_avx2` with no output. It is also moot. Neither CPU has AVX-512 (the VM is an AMD EPYC 7B12 with AVX2 and FMA;
the dev box is an i9-12900F), and libm and GEOS cannot be pinned by an environment variable at all.

⇒ **The Linux harness is the same function evaluated with different elementary-function implementations.** A bitwise
bar against the dev box cannot be met on Linux, and the L4 result shows it cannot be met even between two Linux hosts.
The 1e-13 to 1e-9 residuals never changed a discrete term, a 4-dp PDMS or a floor mean.

**Lever 4, and it works: the hybrid route.** The Colab L4 seam was scored by the **dev box's own Windows harness**
(`score_navtest_refe.py`, unchanged, label `refe_colabp0_l4seam_ep016`).

- Result: status PASS, PDMS **76.5243**, and **every discrete term equal on all 200 tokens**.
- Max per-token \|Δ\| is 3.63e-7 (118 tokens, EP). That comes from the seam's 15 µm pose differences, not from the
  harness.
- ⇒ **Run the seam on Colab and keep the harness and E-6 on the dev box.** This frees the dev-box GPU, which was the
  reason for the move. It keeps every CPU-stage number bit-identical to 016 and to every earlier point, and it needs no
  gate beyond G-S/G-P/G-REP, which passed.
- Evidence: `devbox_harness_on_vm_seam/`; the `hybrid_devbox_harness_on_vm_seam` block in `gates_p0.json`.

**A tolerance for an all-Colab harness would be a new pre-registration. It is proposed here and NOT adopted:**

> *G-H/G-E6 (Linux), proposed: every discrete term bitwise equal per token; \|Δ EP\| ≤ 1e-8 per token; PDMS equal at
> 4 dp.*

The proposal is set about 1.7× above the largest residual seen, 5.9e-9 (the STOP floor on the VM cache). That is thin
headroom, and it is drawn from only one snapshot's residuals. On the measured data it would pass G-H on both VMs, G-E6
(1.2e-11) and the G-H′ sub200 form. Adopting it is the PI's call. It was not applied to any verdict above.

## Resume-from-seam: dev-box steps 3–4 on a Colab seam

`colab/navtest/resume_from_seam.py` is a new wrapper. `eval_checkpoint.py` was not edited. The wrapper runs the same
`parse_navtest6.py` and `families6.py` commands in the same venv, and writes a point JSON stamped with `device`,
`vm_gpu` and `parity_gate`.

It was run on the L4 seam and the L4 harness CSV (`resume_from_seam/sub200_ep016_colabL4.json`) and took 10.2 s.

- **Families:** all 1,038 numeric leaves are **identical** to the dev-box point `points/sub200_ep016.json`.
- **Floors:** 10 of 260 shared leaves differ, by at most 2.4e-9. These are the unrounded PDMS point values, 0.76524278
  on both.

## Compute units

| job | runtime | assigned | rate (CU/h) | CU | evidence |
|---|---|---|---|---|---|
| session A: setup + G-H + G-H′ + G-E6 (+ the G-H levers) | CPU (2 vCPU AMD EPYC 7B12, 12 GB) | ~3,310 s (16:53:50Z–17:49:05Z) | 0.08 | **0.074** | `navtest-p0-cpu/session_record.json`, `navtest-p0-cpu/manual_stop.txt` |
| session B: setup + G-ENV + G-H + seam + G-P + G-REP | L4 (12 vCPU, 52 GB, L4 23 GB) | ~778 s (17:01:27Z–17:14:25Z) | 1.54 (read as 1.62 = L4 + session A's 0.08) | **0.33** | `navtest-p0-l4/session_record.json`, `navtest-p0-l4/manual_stop.txt` |
| **total** | | | | **0.41** of the 6 CU cap | `colab/USAGE_LOG.md` (one row per job) |

The meter read a balance of 200 before P0 and 199.63 after both sessions stopped. The balance lags and rounds, so the
per-job rate × time above is the evidence.

⚠️ **Session B's own stop did not run.**

- The cause: `p0_session.ps1`'s `finally` deleted its temp directory first, and that directory held the empty stdin file
  that every `Start-Process` in the script redirects from. So `stop` and the server readout never started.
- The record then read "not still assigned", because the failed readout had been replaced by a placeholder list.
- The server-side check showed the L4 still assigned. It was stopped by hand about 33 s after the runner ended, and the
  server then showed only session A.
- Both defects are fixed in `p0_session.ps1`: the stdin file is kept until after the server check, and an unreadable
  server state is now UNKNOWN, never False.
- ⚠️ **The fix has not yet run on a live VM (UNVERIFIED).** Session A was already running the old script, so a watcher
  stopped it 11 s after its driver ended and read the server: `assignments: []`, rate 0
  (`navtest-p0-cpu/manual_stop.txt`).
- ✅ **The token refresh is verified live.** Session A ran for 55 min. `refresh_session.py` fired at minute 40
  (`ZZREFRESH_OK navtest-p0-cpu 3600`), and none of the 52 polls failed.

## Measured wall-clock per stage

| stage | CPU VM (2 vCPU) | L4 VM (12 vCPU) |
|---|---|---|
| `colab new` → READY | ~15 s | 14.6 s |
| relay signed-link pull (md5-verified) | 54.5 MB in 3.1 s | 297 MB in 18.3 s: **10–19 MB/s per file, single stream** (plan item 5: GPU VM ≈ 1.8× the CPU VM's 10.4 MB/s) |
| nuPlan test DBs: 93 members range-fetched from the 96 GB public zip, inflated, CRC-verified | — | **5.81 GB in 64.8 s = 90 MB/s** with 12 streams (the plan's floor was 21.3 MB/s on 4 streams); 93/93 CRC equal to the dev box's DBs |
| nuPlan maps: 8 members, CRC-verified | 19.3 s | 32.5 s (971 MB, 30 MB/s) |
| navsim_logs tgz (sha256-verified) + extraction | 32.4 s | 32.5 s |
| DINOv3 ViT-L trunk (sha256-verified) | — | 31.1 s (1.21 GB) |
| harness venv (py3.9.25, 197 pins) | 31.1 s | 25.8 s |
| seam venv (py3.11.13, torch 2.7.1+cu128 **with** its nvidia wheels, then 115 pins `--no-deps`) | — | 56.7 s |
| **setup total** (parallel) | ~65 s | **~87 s** |
| G-ENV | — | 11 s |
| harness, sub200, 1 run (G-H / G-P) | 61.5 s | 56.3 s / 48.9 s |
| **seam, 200 tokens, 1 process** | — | **251.9 s (239.9 s in the loop, about 1.2 s/token)** |
| seam, K=3 shards on the one L4 | — | 232.5 s (per shard 199–219 s): **GPU-bound, so sharding one L4 buys 8 %** |
| metric-cache rebuild, 200 scenes | 327.8 s (1.64 s/scene incl. startup) | — |
| E-6, 64 harness runs (2 workers) | **2,596 s** (per run 77.8–88.6 s, median 80.8; the dev box: 34–96 s, median 49.5) | — |
| runner, launch → end | ~54 min (setup 65 s + G-H 62 s + G-H′ 551 s + E-6 2,596 s) | 688 s |

**A reading that changes P1.** fp32 ViT-L on an L4 runs at about **1.1–1.2 s/token**, not the 0.78 s ESTIMATED in the
plan. It is GPU-bound: three concurrent shards finished in almost the same wall time as one process. Sharding pays only
on a bigger GPU (G4) or across GPUs.

## Traps found in P0 (each MEASURED)

1. **The `finally` deleted what its own `stop` needed** (above). A stop's exit code was never the evidence; the
   server-side check caught it.
2. **With the WindowsPath shim, W3's observation hook writes token `''` into `*_rows.jsonl`.** The hook derives the
   token from `Path(str(PureWindowsPath))`. Scores and the CSV are unaffected. Compare hook rows **by order**, or build
   the cache on the VM.
3. **An L4 seam is GPU-bound, so shards do not help on one GPU** (above).
4. **The Linux harness differs from the dev box at the 1e-13 level, and between two Linux hosts too**, by different
   tokens (above).

## What P1 (the first automated per-snapshot eval on Colab) still needs

1. **A route decision (PI or orchestrator):**
   - **(a) Hybrid (recommended):** the seam on Colab, and the harness, E-6, parse, families and readouts on the dev box,
     unchanged. It is qualified today by G-S/G-P/G-REP plus the hybrid readout.
   - **(b) All-Colab:** needs the tolerance above re-registered for G-H/G-E6.
2. **A per-snapshot driver.** It pushes `snap_epochNNN.pt` to the relay (77 s), mints links, runs
   `vm_p0.py --stages setup_seam,genv,seam` on an L4, and downloads the seam, report and dump (≈ 1.7 MB). Then, on the
   dev box, it places them where the pipeline reads them: `data/refe_navtest/seams/refe_<name>.npz` and
   `proptable/<name>/proposals.npz`. After that, `score_navtest_refe.py`, `proposal_table.py --reuse-dump`,
   `selection_readout.py` and the rest run as today.
   - This is a change to the live eval flow (`eval_snapshot.sh` / `eval_checkpoint.py`). ⛔ It was **not** made here.
     It is the integration item for the orchestrator.
   - `resume_from_seam.py` covers steps 3–4 with a device stamp.
3. **Cost of route (a):** about 1.5 min of setup (no harness venv or navsim_logs needed) plus 4.2 min of seam plus the
   download, so about **6–7 min of L4 per snapshot, about 0.17 CU** (ESTIMATED from the measured stages above). The plan
   estimated 0.70 CU; its E-6 and harness parts stay on the dev box's CPU.
4. **First-run checks P1 must do**, because P0 did not exercise them live:
   - the fixed `finally` stop (UNVERIFIED);
   - `setup_seam` without `setup_harness`: the code path exists but was only run together with it.
   - The token refresh WAS exercised live (session A, 55 min).
5. **Still open for the final (P2):**
   - the full-set half of G-H′;
   - G-S/G-P on a G4;
   - the per-shard-parallel frame build;
   - `model_final` through the relay.
   - The DB range-fetch rate measured here (90 MB/s) makes the full test-DB set (7.68 GB) about 1.5 min.
   - ⚠️ **The measured L4 seam rate changes the final's L4 option.** At about 1.2 s/token (GPU-bound), 12,146 tokens
     take about **4.0 h of seam alone** (ESTIMATED from the MEASURED rate), not the plan's 158 min. That strengthens the
     plan's G4 recommendation for the final.

## Decisions escalated

1. **P1 route: hybrid (a) or all-Colab (b).** Recommended: (a).
2. **Adopt the proposed Linux-harness tolerance, or not.** It is needed only for (b), and it is a new pre-registration.
3. **Integration owner** for the per-snapshot driver and the `eval_snapshot.sh` hand-off (item 2 above).

## Deliverable manifest

| artifact | where | note |
|---|---|---|
| VM runner, launcher and poller: `vm_p0.py`, `launch_p0.py`, `poll_p0.py` | repo: `colab/navtest/` | |
| Linux harness launcher and wrapper shim: `linux_run_v1.py`, `navsim_v1_linux.py` | repo: `colab/navtest/` | promoted from the plan's prototype; adds `cache`, floor arms and `--paths` |
| Linux E-6 driver `e6_linux.py`; sharded seam and merge `seam_shards.py` | repo: `colab/navtest/` | |
| public-data range fetcher `zip_member_fetch.py`; `maps_manifest.py` | repo: `colab/navtest/` | |
| relay bundle builder `build_sub200_bundle.py`; code packer `pack_code.py`; job merge `merge_jobs.py` | repo: `colab/navtest/` | |
| session driver `p0_session.ps1` (stop bug fixed); `refresh_session.py` | repo: `colab/navtest/` | the fix is UNVERIFIED live |
| gate evaluator `compare_p0.py`; `resume_from_seam.py`; `ufunc_probe.py`; `gh_lever.py` | repo: `colab/navtest/` | |
| this file | repo: `colab/navtest/RESULT_P0.md` | |
| raw evidence (manifests, session records, results trees + tarballs, gates, probes, resume output) | repo: `colab/raw/2026-09-27-navtest-p0/` | |
| relay bundle `p0_harness_sub200.tar` (54.5 MB), `p0_frames_sub200.tar` (166.2 MB) | HF private `Sayood/tanitad-colab-relay:refe/navtest_sub200/` (+ MANIFEST) | the local copies are in the session scratchpad only; the relay is the durable copy |
| VMs | none | both stopped and checked on the server |
