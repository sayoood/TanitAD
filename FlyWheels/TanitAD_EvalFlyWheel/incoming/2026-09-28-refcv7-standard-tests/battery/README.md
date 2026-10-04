# refcv7 four-family milestone battery — how it runs

* **What is tested and how it is judged:** `SPEC.md` (pre-registration + amendments A1–A4; the sha256 of every version is in `raw/SPEC_SHA256_*.txt`).
* **What has been measured so far:** `RESULT.md`.
* **The files to land:** `LANDING_READY.txt`. The Master Mind is the single committer; nothing here is `git add`ed.

## Armed milestone
* `code/chain_step5000.sh wait` runs `code/waiter_milestone.sh 5000`, detached. It polls Thor read-only every 5 min (every 1 min once the checkpoint file exists). It proceeds when **Thor's `ckpt_5000.pt` is complete**: size and mtime stable for ≥ 20 s, and the step-5,000 eval row is present (the trainer saves before writing that row).
* It pulls with `code/pull_ckpt.py`: Thor md5 before == after == local, and the `step` key is read locally. A file whose step is not 5,000 is REFUSED. If there is no milestone file, it falls back to the rolling `ckpt.pt` while the run's last logged step is in [5,000, 5,500).
* It then calls `code/chain_milestone.sh 5000`. The same two scripts serve 15k / 20k / 30k: pass the step number.
* Logs are in `D:/refcv7_eval_kit/chain/`. Artifacts are written to `D:/refcv7_eval_kit/battery/step<N>/` (the WORK dir, with raw clip ids) and banked sanitized to `raw/step<N>/` (sha12 only).
* **By hand:** `sh code/chain_step5000.sh wait`, or `sh code/chain_step5000.sh run` once `D:/refcv7_eval_kit/ckpt/ckpt_5000.pt` and `D:/refcv7_eval_kit/chain/pull_5000_OK.json` exist.

## Scheduling rules on the shared dev-box GPU
These are agreed with the Master Mind and the Research Lab on 2026-09-28. They are rules of operation, not SPEC changes.
1. **One GPU job at a time, through the lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`.**
   * The lock is created exclusively (noclobber) as JSON `{job, pid, win_pid_of_locker, host, acquired}`. Fields written by other sessions are tolerated.
   * It is removed only by its owner: same job AND pid (`code/gpu_lock.py`).
   * Every GPU stage also requires `nvidia-smi` to show no other python compute app. No process is ever killed or signalled.
2. **The chain-active marker `C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active`** (JSON `{chain, pid, started}`).
   * It is created when the milestone chain first holds the lock, and removed on every exit path (EXIT trap), only if its content is still this chain's (`code/chain_marker.py`).
   * While it exists, long foreign jobs must not acquire the lock. REFe snapshot seams (≤ 20 min) may.
3. **The evening yield.** After every arm/stage that finishes after 20:00 Berlin, the chain releases the lock, sleeps ≥ 90 s and re-acquires it (`code/gpu_yield.py`).
   * The yield runs between: G0 and the rolls; the seed-0 and seed-1 rolls; the perception passes; and every refcv6@38k baseline stage (G0, A1, A2, roll seed 0, roll seed 1).
   * Each gap, and who took the lock in it, is logged to `D:/refcv7_eval_kit/chain/gpu_yields.jsonl`.
4. **Collision watch.** While the chain runs, `code/gpu_watch.py` logs every python GPU app that is not one of this battery's stages to `gpu_watch_<N>.jsonl`. The count is stamped in the chain log (`ZZCOLLISIONROWS`).
5. **Resumable, one retry per stage.**
   * A G0 artifact for the same checkpoint md5 is reused.
   * A seed dump with a manifest and a roll record is not re-rolled.
   * Each perception pass and the refcv6 baseline stages are skipped when their artifact exists.
   * A collision costs one arm or seed, not the run.

## Order inside the milestone chain
1. G0: 8 seeds, M1/M2/M4, wrapper clause. The chain STOPS unless G0-A2 = PASS; G0 as registered is always reported first.
2. T1 rolls: every arm at inference seed 0, `os` only at seed 1 (SPEC A3). Then panels, the four families and BAR-R7-1/3. BAR-R7-2 reads **PENDING** until the refcv6@38k rolls exist.
3. VOID gates 5/6, then the refcv6@38k perception dump if it is not already banked, then the refcv7 perception pass (map 10 cm and box, three arms) and BAR-M7-1..4 / BAR-B7-1/2. An interim RESULT is banked here.
4. **Last:** the refcv6@38k S2 rolls (stage-wise, reused by every later milestone per SPEC A4, with `BASELINE_STAMP.json` re-verified before reuse), then BAR-R7-2, then the final RESULT. The RESULT section is appended to `RESULT.md` and a new heading to `LANDING_READY.txt`.

## 2026-10-04 — after the step-5,000 G0 failure (SPEC AMENDMENT A5)
* **G0 now runs 24 inference seeds** (`run_battery_r7.py --g0-seeds`, default 0..23; `chain_milestone.sh` passes them explicitly). `g0.json` carries `verdict_as_registered` (seeds 0..7), `verdict_A2` (seeds 0..7) and `verdict` = **G0-A5, the gate** (all 24). `T_995` has no default any more: a seed count without a registered t-quantile REFUSES.
* **Memory, not numbers:** G0's 8 collated batches go to disk once (`<out dir>/g0_batch_cache_<md5[:8]>/`, removed after G0) and are read back through a file-backed mmap (`g0_refcv7.MmapBatches`); `chain_milestone.sh` first waits for ≥ 6 GB **free commit** (`code/wait_commit.py`), THEN takes the GPU lock — never the other way round, so the chain never holds the card while another stream's process holds the memory.
* **Diagnosis tools:** `code/g0_diag_r7.py` (one lever per arm on G0's own 128 windows), `code/run_g0_diag.sh` (commit wait → lock → diag), `code/g0_rejudge_a5.py` (zero-GPU POST-HOC A5 re-judge of an 8-seed G0 extended by the diag's seeds; refuses unless the diag's `s0` reproduces G0's seed 0).
* **Pull records for checkpoints already on the dev box:** `code/pull_record_local.py` (Thor md5 before/after via read-only `md5sum`, local md5, step key, tensor equality full vs model-only, Thor `metrics.jsonl` md5 == the local copy) → `D:/refcv7_eval_kit/chain/pull_<N>_OK.json` in `pull_ckpt.py`'s shape. Written for 50,400 (final rolling → model-only) and 30,000 (milestone file).
* **The unattended sequence:** `nohup sh code/orchestrate_final.sh > /d/refcv7_eval_kit/chain/orchestrate_final.out 2>&1 &` — diag (step 5,000) → post-hoc A5 re-judge + sanitized bank → `chain_milestone.sh 50400` → `chain_milestone.sh 30000`. Markers `ZZORCH…` in `D:/refcv7_eval_kit/chain/orchestrate_final.log`.
