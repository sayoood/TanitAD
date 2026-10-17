# P6's first real launch gate: two refusals, neither a P6 config defect (R42, R43, R44)

**Master Mind, 2026-10-16.** P6 (`p6_paper_faithful`, argv sha `7c2b7b97d1bbad2d`, ship c1dc5d3) REFUSED at 18:50:29Z.
- **G-LIVE-LAB** failed with "GPU stall 33.1 % of a window > 5 %".
- **G-CKPT** failed with "setup failed: U rc 1, H1 rc 1", both CUDA OOM.
- Every other check PASSED.

Evidence class: MEASURED on the pod unless marked otherwise. The A40 was idle throughout, so every probe ran alone on the card.

## Evidence index
| file | what |
|---|---|
| `refe_p6/launch_gate.log`, `refe_p6/lg/*.json` | the refusal: evidence of the two failing checks, G-LIVE-MEM, the FAIL token |
| `refe_p6/lg/work/smoke_run/metrics.jsonl` | the accum-2 smoke's labeller windows: stall 0.367 (start-up) / **0.331** (the one judged), window 3.24 s |
| `refe_p6/mem_metrics.jsonl`, `refe_p6/mem.log` | the launch-argv run (accum 50, 200 sets/window): stall 0.1414 / **0.0500021** / 0.0278 / 0.0480; peak `cuda_max_mem_gb` **41.29 GiB** |
| `refe_p6/lg/work/ckpt/U.log`, `H1.log` | gate run: OOM, 37.72 GiB allocated + **5.79 GiB reserved-unallocated**, gate process 474 MiB |
| `refe_p6_probe/run_es_noenv_fail/` | probe 1 (19:01Z): invalid, missing `REFE_BACKBONE_ROOT` (kept as a record) |
| `refe_p6_probe/run_es/` | probe 2 (19:08Z, `expandable_segments`, old order): U OOM at **43.60 GiB allocated**, fragmentation 55 MiB, gate process 394 MiB |
| `refe_p6_probe/r43_es/` | probe 3 (19:23Z, R42+R43 code): "CUDA initialised in this process: False"; U still OOM **alone**, at 43.70 GiB allocated; H1 fit |
| `refe_p6_probe/r44_es/` | probe 4 (19:38Z, R42+R43+R44): **G-CKPT PASS** (643/643 tensors, optimiser, data position, loss trajectory identical, batch_accum [2, 4]); **G-LIVE PASS** |
| `p6_ckpt_probe*.sh` | the probe drivers (`--only G-LIVE,G-CKPT`, scratch out-dir and key: they cannot mint a token the chain accepts) |
| `patch_r42.py`, `patch_r43.py`, `patch_r44.py` | the exact-anchor patches, applied identically to the ship tree (c1dc5d3) and the tip (deltas compared: identical) |
| `sweep_new/`, `sweep_tipnew/` | full-form runs of every test naming an edited file (R39–R41 rule), on the patched ship tree and the patched tip |

## Status
- P6 now waits for the **twin re-roll** (PI 2026-10-16 ~20:00Z, option 3: *"faithful for the paper"*).
- The re-measure of G-LIVE-LAB at the launch accum 64 happens in P6's chain (the mem stage) once the re-rolled bank exists. Both outcomes are committed in R42 below.


### R42 (2026-10-16) - "G-LIVE-LAB's 5 % stall bar can be judged on the gate's smoke run" (gate_checks.judge_lab, built and fixture-tested before P6 shipped at c1dc5d3)
- **What was true.** P6's first real launch gate REFUSED at 18:50Z.
  - G-LIVE-LAB failed on "GPU stall 33.1 % of a window > 5 %".
  - The smoke that reading came from runs at `--accum 2` (2 micro-batches per optimizer step) and logs 2 windows. The judge drops the first (pool start-up), so the bar was decided by ONE 3.24 s window: 1.07 s of stall over 2 micro-batches.
  - `train.py` (D4, the deferred backward) **drains each accumulation window's LAST micro-batch before the optimizer step**. One label latency per window is therefore unhidden by construction, and the stall fraction scales like 1/accum.
  - At accum 2 that is ~1/3 of the window on ANY pod. At the launch's accum 64 it is ~1/64.
- **MEASURED on the same pod, same argv sha 7c2b7b97d1bbad2d.** The G-LIVE-MEM run (the launch argv at `--accum 50`, 4 windows of 200 sets) read **14.1 % (start-up) / 5.0002 % / 2.8 % / 4.8 %** (`refe_p6/mem_metrics.jsonl`).
  - The smoke's drain-dominated window read 33.1 %.
  - So **the bar as built could never pass**. The fixture it was tested with (`test_p6_levers.py`: smoke windows 0.30 / 0.01) is physically impossible for an accum-2 smoke.
- **Fixed** (both outcomes committed before the re-run):
  - The L1 stall bar (SPEC_P6 3.2, `LAB_STALL_MAX` 5 %, unchanged; it is the PI's) is judged on the G-LIVE-MEM run.
  - That run now uses the LAUNCH `--accum` (`p6_prep.mem_argv`: 4 × 64 = 256 micro-batches; it was `--accum 50`).
  - It requires: the run's `bank` accum equals the launch argv's `--accum`; at least 3 labeller windows; lag 0; and the max over windows after the first ≤ 5 %.
  - The smoke still proves the labeller RAN (event present, lag 0, coverage). Its stall is reported, not judged.
  - The pre-R42 smoke-only call now FAILS ("no steady-state stall run"); it can never pass silently.
  - Regression arms in `test_p6_levers.chk_gate_judges`: the measured 33.1 % smoke passes; a 20 % window fails; the measured 5.00021 % window fails (it IS > 5 %); a start-up-only 90 % first window passes; accum 50 vs launch 64 fails; a missing accum fails; too few windows fail; lag fails; no stall run fails; `check_live_lab` end to end PASS/FAIL ×3.
- **Committed before the re-run.**
  - The re-measured launch-accum run **≤ 5 % ⇒ G-LIVE-LAB passes, and P6 launches** if every other check passes.
  - **> 5 % ⇒ G-LIVE-LAB fails honestly**, and the next lever is P6_ON_11VCPU lever 3: split each 64-proposal set into 2 labeller jobs (MEASURED bit-identical, latency halves). That lever is mine, not a bar change.

**Class:** NEW, sibling of the "built, tested, unreachable" family. Call it *built, tested, UNSATISFIABLE*.
- The judge was correct on its fixtures, but the run it judged on cannot produce the quantity the bar is about.
- It was found only by the first real gate. The fixtures encoded what the author EXPECTED the smoke to read, not what an accum-2 smoke CAN read.
- ⇒ **For every gate bar, derive the bar's quantity from the code path of the run it is judged on**, here the drain-per-window identity, before trusting a fixture. A fixture value the real run cannot produce is a test of nothing.
- ⇒ Prefer judging a rate on the run that has the launch's denominators (accum, batch, workers). A smoke shrinks exactly those.

### R43 (2026-10-16) - "freeing the smoke before G-CKPT gives its two subprocess trainings the card" (launch_gate.run_checks, the 2026-10-12 P2 fix)
- **What was true.** The same REFUSAL failed G-CKPT too: U and H1 (the uninterrupted and the halt-resume trainings, each a SUBPROCESS) both died CUDA OOM.
  - The 2026-10-12 fix had freed the smoke, and the gate process logged 0.14 GiB reserved before G-CKPT.
  - **But a freed model leaves its CUDA CONTEXT behind.** The OOM message names "Process 397154 has 474.00 MiB memory in use", which is the gate process. No free can release that.
- **Two readings on the idle A40, same argv (sha 7c2b7b97d1bbad2d):**
  - (1) The gate run itself (18:15Z), default allocator: 37.72 GiB allocated + **5.79 GiB reserved-but-unallocated**, OOM.
  - (2) A probe (`--only G-LIVE,G-CKPT`, scratch key and out-dir, 19:08Z) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: fragmentation gone (55 MiB). OOM anyway, at **43.60 GiB ALLOCATED** with the gate process holding 394 MiB.
  - P6's compile-free smoke genuinely needs ~43.6+ GiB of the A40's 44.43. G-LIVE runs the same smoke IN-process (one context) and passes; a subprocess pays a second context and cannot fit.
- **Fixed.**
  - `check_ckpt` runs FIRST in `run_checks`, before the build and the smoke, i.e. before this process opens a CUDA context. Its evidence is emitted in its usual place.
  - It reads only `s_argv` and the profile, never the build or the smoke capture, so no input changes.
  - `test_gate_gpu_probes.order_ok` now pins "check_ckpt is called before the first instrumented_run, and emitted". Mutations: moved after the smoke → RED; not emitted → RED.
  - Separately, `p6_pod.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the gate, every instrument and the launch. The launch argv peaks at **41.29 GiB allocated** (G-LIVE-MEM), and reading (1) showed 5.79 GiB of fragmentation, so fragmentation alone could end a 7-day run.

**Class:** the 2026-10-12 fix's *mechanism was right and its model of the resource was incomplete* ("free the tensors" ≠ "free the device").
- A fix verified only by "the reserved counter reads ~0" verified the counter, not the card.
- ⇒ **When a check spawns a process that needs the device, measure the device from the CHILD's side (the OOM message names every holder), not the parent's allocator.**

### R44 (2026-10-16) - "with the gate process off the card, G-CKPT's batch-4 round trip fits" (R43's fix, before its own probe)
- **What was true.** R43's probe (19:23Z, `--only G-LIVE,G-CKPT`, expandable_segments) logged "CUDA initialised in this process: False". The gate held nothing.
  - **U still died OOM ALONE on the card:** 43.70 GiB allocated, 266 MiB reserved-unallocated, 44.34 GiB in use of 44.43.
  - H1, the identical config, survived the same step 0.
- **Why.** `_train_sub` runs every G-CKPT training under DET_BOOT (deterministic algorithms, which the bitwise round trip needs; MEASURED 2026-10-12). The deterministic kernels cost memory the in-process smoke (G-LIVE, non-deterministic) does not pay.
  - P6's eager micro-batch of 4 sits at the card's edge under them: one run fits and one does not.
  - The launch itself (compiled, non-deterministic) peaks at 41.29 GiB and is not affected.
- **Fixed.** The `refe_p6` profile carries `ckpt_micro_batch: 2`. `check_ckpt` runs U/H1/H2 at micro-batch 2 with the accumulation doubled (effective batch 8 kept) and records `batch_accum` in its evidence.
  - Rationale (the gate's own, for `--compile`): the resume contract (model, optimiser, data position, loss trajectory bit-identical) is independent of the micro-batch.
  - Other profiles are unchanged.
  - `test_gate_gpu_probes`: P6 key present and P2's absent; the built argv is scaled; the default is unchanged; mutation "key ignored" → RED.

**Class:** R43's class again, one level down. The probe that FIXED the parent's share of the card also MEASURED the child's own need, and the child's need was the binding term.
- ⇒ A resource fix is verified by the run it was for completing, not by the resource it freed.
