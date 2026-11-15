# HANDOFF Q5BUILD (2026-11-15). State: items 1 and 2 built; the pod build is NOT launched because C6a read differently from the spec.

## Stopped on a spec contradiction (needs an MM / PI call)
Amendment 7 C6a: pod-built 416 eval v2ep vs `E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl`, decoded uint8 max |diff| = 0.
MEASURED on 3 eval clips (`smoke_c6a.json`, per-frame sha256 of decoded frames): all 3 clips differ in every checked frame. On clip 01be5919 (frames 0, 2, 100, 200): max |diff| = 1 count,
~0.9-1.2 % of values differ by exactly 1, mean 0.01. poses / actions / n_frames (201) / `frame` record / f_ref are bitwise equal. Same mp4 sha256 (pinned revision a0cf20df), same unedited
`build_v2ep_wide.build_one`. The residual is float rounding in `cylindrical_rectify` between the dev box and the pod CPU/torch build (HYPOTHESIS, not probed).
Options for the MM: (a) re-read C6a as max |diff| <= 1 count with the differing fraction reported (amend the frozen text), (b) rebuild eval-141 on the dev box (45 s/clip) and use pod tokens only for train,
(c) accept as is. Train and eval must share one frame route, so (b) mixes routes. Nothing was launched before this call.
Also: C6b MAE on the 3 clips = 0.931 counts (limit 1.0, reference 0.59); full-set value comes from the ledger aggregate (`controls_c6.json`).

## Passed
* Tests: `test_q5.py` 19 passed (C1, C2 code path, C7 incl. pinhole mutation, C8 code path, C10, writer, encoding= guard).
* Pod smoke C8: sibling at 256 bitwise equal to `encode_episode` on 3 eval clips (`smoke_c8.json`; sparse-vs-full rel-L2 0.0154-0.0173 printed beside).
* Pod smoke C6c: poses / actions / n_frames bitwise vs the 256 v2ep: true on 3 clips.
* S1 sample reproduced on the pod from `/root/runs/bprime/join_train.npz`: sha256 843e4017 (pool 286,035 / 13,116 / 272,919 matches banked). S2: sha 32ad96be. S1: 4,265 clips; S2: 4,256; clip overlap 4,063; window overlap 1,640; union 4,458 clips.
  The pod job builds the union (S1 and S2 share the build; +193 clips over S1 alone).
* Eval windows: 2,365 / 131 positives / 139 episodes (`eval_windows.json`, from STOPDEC BAL-NULL_s0 `has` windows).

## Artifacts (all under `F:/TanitAD-scratch/q5_416/build/`; pod copies in `/root/q5_416/code/`)
q5_now_encode.py, q5_frames_pod.py, q5_sample.py, q5_read.py, test_q5.py, eval_windows.json, smoke_*.json.
Pod also holds `/root/q5_416/dry_20261111/sample.json`, `dry_20261115/sample.json`, `eval_windows.json`, `s0p/` (unedited copies). Not staged in git (no commit by agents).

## Launch (only after the C6a call)
`ssh` via `source C:/Users/Admin/lorapod/podenv.sh; pssh`. Env: `. /root/ops/lorapod2_env.sh; env_lora`.
`cd /root/q5_416; nohup setsid $PY code/q5_frames_pod.py run --samples S1,S2 > run.out 2>&1 < /dev/null &`
Log `/root/q5_416/q5build.log` (ends `ZZ Q5BUILD DONE ZZ` or `ZZ Q5BUILD FAIL ZZ <why>`), ledger `/root/q5_416/ledger.jsonl`, rows `/root/q5_416/rows/<clip>.npz`, marker `q5build.marker`.
Resumable (clip done iff ledger OK + rows file). Eval 416 v2ep kept under `v2ep416/eval` (11.7 GB, for C6a); train 416 v2ep deleted after each clip encodes; disk floor 25 GB checked per chunk.
Expected ~5.5 h. Afterwards: `q5_frames_pod.py hashes --out ...` (C6a), `q5_frames_pod.py gather` (S0-P-format vision dirs under `/root/q5_416/out/{S1,S2,eval}_{416,256p}`).

## Next steps
1. MM decides C6a. 2. Launch. 3. After box 2's ACTSENS chain (~01:30 11-16): C0 (S0-P extractors on box-2 caches, S1 and S2 seed; C4 bitwise).
4. Pull `out/*`, `q5_now_encode.write_s0p_dir(<C0 dir>, out, v40, v160, mean)` per sample (eval: `write_s0p_npz`; the eval npz is re-created by `s0p_extract_eval.py`), assert y/clip/ws/eid equal the pod's.
5. `s0p_fit.py prep` + `fit` per dir (unchanged), then `q5_read.py <cfg.json>` (controls first; C1/C2/C3/C5 inputs come from those fits).
Not built: C5 / C3 fit-side drivers (they need the fits), `q5_read` beside arms (M, P-B, E-limit, V416tok, HEAD-UNDERUSES).

## Deletions / leftovers
Delete was refused by the permission layer, so these remain: pod `/root/q5_416/smoke/` (~0.3 GB incl. smoke rows and 3 eval 416 v2ep), local `F:/TanitAD-scratch/q5_416/build/c6a_probe/` (83 MB pod copy of one v2ep), `F:/TanitAD-scratch/q5_416/tmp_pytest`.
