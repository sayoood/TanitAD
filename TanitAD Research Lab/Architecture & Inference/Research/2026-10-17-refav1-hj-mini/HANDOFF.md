# HANDOFF — RA-1a pooled frozen-DINOv3 cache (stream S4, prep) — 2026-10-17

**State: code ready + CPU-validated. Nothing is encoded for the real cache. The 4060 was NOT supposed to be touched; one slip, see §6.**
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/` (`code/`, `raw/`). Cache lands in `C:\Users\Admin\refav1_hj_cache\{train,eval}\` (one `.pt` per clip + `manifest.jsonl`).

## 0. Decide first: the train frames' source is gone
Thor's native 256×640 B1 v2ep dir (`physicalai-b1-w120-256x640cyl`) and the refav1 train fp8 cache were deleted (`Project Steering/Ops/disk_cleanup_2026-10-04/INVENTORY_THOR.md` C01; `ls` on Thor 2026-10-17: absent). The banked `pull_encode.py` therefore cannot run.
`--source auto` falls back (banner `ZZSOURCE-FALLBACK-416ZZ`) to Thor's 416×1024 B1 v2ep (4,713; poses/actions bit-identical to the 256 ones), resampled ×0.625, rows 2:258.
MEASURED (`raw/validation.json` V3b; **1 eval clip, 6 frames; kernel chosen on that clip**): pooled-4×10 vs the refav1 eval cache cos 0.9990 / rel-L2 4.6 % (native route 0.9995 / 3.0 %). Full field 0.9966 vs 0.9994.
Exact alternative: rebuild 256×640 v2ep from HF `Sayood/tanitad-v7-training-corpus` mp4s (3.4–4.6 h, INVENTORY C01), then `--source local --src-dir DIR --src-geom 256`.
Train(416-route) vs eval(native) differ by that residue. Stage D (`-MatchedEval`, +11 min) encodes eval by the same route for a matched A/B read.

## 1. Launch (4060 must be free: `python code/gpu_lock.py status` + `nvidia-smi`)
```
cd "<package>\code"
powershell -NoProfile -File .\launch_detached.ps1 -DryRun            # 30 s preflight, no GPU, no lock
powershell -NoProfile -File .\launch_detached.ps1 -Stages all [-MatchedEval]
```
Dry-run must show: `train==labels=True`, `n_disqualifying: 0` (train), `[source] kind=…`. Eval gate info: 11 of 141 eval clips lie in `physicalai-train-e438721ae894` (irrelevant to refav1's own split; matters when comparing against parity-corpus models: 130 clean).
The run waits for the shared lock `…\refcv7\devbox_gpu.lock` (never deletes foreign locks) and `memory.used ≤ 2500 MiB` twice, then:

| stage | what | time (4.72 s/clip, banked) | disk |
|---|---|---|---|
| A | 600 train, 4×10 + 8×20, then `verify --redo 5` on cuda | 47 min | 12.5 GB |
| B | 141 eval (local native v2ep, no Thor), 4×10 + 8×20 | 11 min | 2.9 GB |
| C | all 4,572 train, 4×10 only (A's 600 skipped) | 5.2 h | 17.3 GB |
| D | optional matched eval, 4×10 | 11 min | 0.6 GB |

A+B+C = 32.8 GB (C: free 61 GB). 8×20 for all 4,572 = 95.6 GB: does not fit, so 8×20 exists for A and B only.

## 2. Progress
`Get-Content C:\Users\Admin\refav1_hj_cache\logs\run_encode.out -Tail 5`; per-stage logs `logs\A_train600.log` …; `(Get-Content …\train\manifest.jsonl).Count` (600, then 4,572); rate line `[pull_encode_pooled] i/N … s/clip`.

## 3. Integrity (run automatically after each stage; by hand)
`python code/verify_cache.py --split train --n 600 --grids 4x10,8x20 --sample 100 --redo 5 --device cuda` → `ZZVERIFY-OK`.
C1 md5/size of every file · C2 `weights_only` load, shapes, finite, content_md5 · C3 disjoint from eval-141 · C4 coverage == pick · **C5 bit-exact GPU re-encode of 5 clips.**
If C5 fails the GPU path is non-deterministic: stop, do not build arms on it.

## 4. Failure tokens (same command resumes; done clips are md5-verified and skipped)
`ZZNO-CUDA` · `ZZLOCK-TIMEOUT` (8 h) · `ZZSOURCE-MISSING` · `ZZSOURCE-BROKEN` rc4 (5 consecutive bad clips) · `ZZDISK-LOW` rc7 (<8 GB free) · `ZZSKIP` rc3 (auto-rerun ×3) · `ZZFP8-FALLBACK-FP16` (2× disk) · `ZZSPLIT-REFUSED`.

## 5. NOT verified
Everything on CUDA by design: bf16 GPU rate (the 4.72 s is the banked number), GPU C5, fp8 saturation off the validated clip (pooled |x| max 3.5 vs 448 limit), stage D end-to-end (dry-run only), the 416 route beyond 1 clip. Colab not prepared (source needs the Thor LAN).

## 6. Incident (disclosed)
While testing `run_encode.sh` flow I exported `CUDA_VISIBLE_DEVICES=""`, which does NOT hide the GPU on this box. For ~4 min (15:33–15:37) `--device cuda` ran on the shared 4060 beside the `refcv8-g0-step20000` job, encoding 38 train clips (median 4.86 s/clip, contended) and pulling 42 v2ep from Thor (≈ 3.5 GB, over the 2-clip allowance). Killed by timeout; nothing running. Details `raw/INCIDENT_gpu_touch.md`. The 38 clips are in `refav1_hj_cache\test_runscript\train` (valid, not adopted). Fixed: guards now use `-1` and assert `not cuda.is_available()`.

## 7. Orchestrator decision (2026-10-17 ~16:00)
* **Not on the dev-box 4060, and no bulk pull from Thor.** An attempt to launch stages A+B+D here was refused by the session's permission guard, as interference with running workloads: the 4060 lock is held by another session's `p2full-main-c06`, and refcv8 is training on Thor. That is correct: the PI designated the second computer ("box 2", 8 GB GPU) for experiments while Thor runs refcv8.
* **Source: the EXACT route, from HF mp4s.** Rebuild the 256×640 v2ep for stage A's 600 train clips from `Sayood/tanitad-v7-training-corpus` on the dev-box CPU: no Thor, no GPU. Then encode on box 2 with `--source local --src-geom 256`.
  * This removes the 416-route residue: train and eval go through the same native route, so stage D is not needed.
  * The eval-141 native v2ep are already local (`C:/Users/Admin/tanitad-data/refav1-eval141/eps`).
* **Stage C (all 4,572) is deferred** until H-DATA-SCALE-1 (25/50/100 % of the 600) says data is the lever, or until refcv8 frees Thor.
* The 38 test clips in `test_runscript\train` (416 route, contended GPU) are **not adopted**.
