# Incident 2026-10-17 — the dev-box RTX 4060 was used for ~4 minutes by a test of `run_encode.sh`

**What happened.** To test the control flow of `code/run_encode.sh` (lock → idle gate → stage dispatch → trap release) I ran it with a temporary lock file and `export CUDA_VISIBLE_DEVICES=""`, expecting the encoder to stop at `ZZNO-CUDA`. An EMPTY `CUDA_VISIBLE_DEVICES` does not hide the GPU on this machine (re-measured afterwards: `""` → `torch.cuda.is_available() == True`; `-1` → `False`). The script's `--device cuda` stage therefore ran for real.

**Window.** Started 15:32:54, encode loop 15:33:55 – 15:37:50, killed by `timeout 300` (rc 124). No `pull_encode_pooled` process survives (checked with `Win32_Process`). `nvidia-smi` afterwards: 1,592 MiB used, no process of mine.

**Footprint.**
* GPU: one bf16 ViT-L process, ~4 min, on a card that held ~5.0 GB for another session. The real lock was NOT used by me; at 15:24 it had been taken by `refcv8-g0-step20000` (pid 137308). I ran beside it without holding it. No OOM was observed, but I did not check that job's step time.
* Thor (read-only): 42 train v2ep (416×1024, ~64–88 MB each; 38 encoded + 4 prefetched, ≈ 3.5 GB) scp'd, against an allowance of 2 clips. Total pulled this session: 42 + 2 (earlier smoke, legitimate) + 1 eval (V3b). The 4 un-encoded prefetches were deleted.
* Disk: 38 clips (0.8 GB) at `C:\Users\Admin\refav1_hj_cache\test_runscript\train` (manifest + files, bf16/cuda/thor416-resize, valid by the same checks). Not adopted into `train\`.

**What was NOT affected.** All earlier "CPU" runs created every tensor and model with `device="cpu"` explicitly and never initialised CUDA, so the guard's failure did not matter for them. The real `devbox_gpu.lock` was only read (`status`), never written or renamed.

**Accidental measurement (do not quote as a rate):** 38 clips, median 4.86 s/clip (mean 4.95, first clip 15.5 s) with 3 prefetch workers on a GPU shared with another job. Consistent with the banked 4.72 s/clip; it is contended and unplanned.

**Fixes.** `validate_cpu.py` sets `CUDA_VISIBLE_DEVICES=-1` and asserts `not torch.cuda.is_available()`; the retest of `run_encode.sh` used `-1` and stopped at `ZZNO-CUDA` (rc 10, temp lock released) in 12 s with no GPU and no Thor pull.
