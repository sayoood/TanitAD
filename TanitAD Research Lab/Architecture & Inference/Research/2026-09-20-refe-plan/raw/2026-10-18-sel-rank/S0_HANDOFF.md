# SR-TTC S0 hand-off (2026-10-18, written 11:50)

## STATE: BUILT AND SMOKE-TESTED, **NOT RUNNING**
The detached relaunch (after the P6 scope change) was **denied by the permission classifier** ("Interfere With Workloads").
No driver, no post-watcher; nothing banked; no GPU lock held. A first launch (11:37) was killed by me on purpose to apply the scope change and has banked nothing.
**To start:** a human or the coordinator approves the launch, then pastes the body of `code/launch_s0.ps1` twice (no script execution on this box, never `-ExecutionPolicy Bypass`):
main: `.\launch_s0.ps1 -MinFreeRamMb 4500`, then post-watcher: `.\launch_s0.ps1 -Post`. The same command RESUMES (banked units are skipped).
Option that touches no pod: add `--no-stage` to the driver = phase L only (816 tokens, 38 logs; ends `S0_FAILED rc5` by design because coverage is incomplete).

## Poll (zero tokens)
`C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I C:/Users/Admin/srttc_s0/code/status_s0.py`
Markers in `C:/Users/Admin/srttc_s0/run/`: `S0_DONE` | `S0_FAILED` (rc 1 preflight/dump, 2 gate, 3 staging, 4 assembly, 5 coverage, 6 C: < 5 GiB, 9 driver died), then `FINALIZED.txt`.
Kill = explicit PID, tree with `taskkill /T /PID <pid>`; never `pkill -f`. Remove only a lock YOU own (`devbox_gpu.lock`).

## P6 overlap rule (coordinator, 2026-10-18), implemented in `srttc_s0_stage.py` + `podcopy.py`
1. pod reader runs as `nice -n 19 ionice -c3`; 2. one stream at a time; 3. no pod-side python/hashing: tar, tail, stat, ls only; 4. before EVERY ~120 MB batch the tail of P6's chain log (`/workspace/data/refe_p6/chain.log`) is read: a `P6_GUARD` line => batch skipped, retry in 10 min (unreadable log => retry in 1 min).
Never touch P6 processes/locks; never stage onto the pod. Staging on `D:/Projects/TanitAD-artifacts/srttc_s0_stage/{dbs,frames/<batch>}`; keep >= 20 GiB free on D: (producer waits at 22), >= 5 GiB on C:; staged copies are deleted after each unit is banked.
**Integrity changed:** no remote sha256 (pod CPU). Now size-equals-pod-stat + local sha256 in the manifest + every file opened by its consumer (sqlite / cv2). Unit-tested: P6_GUARD skip logic, staging smoke 4/4 tokens, cleanup exact.

## Phases / ETA (ESTIMATED, not measured end-to-end)
Phase L (local DB+frames): ~25 min. Phase S (stage from pod, dump, bank, delete): ~3.6-4.5 h wall, plus waiting for `P6_GUARD` lines. Dump MEASURED ~1.6 s/token; staging smoke MEASURED 339 MB in 116-143 s; DB staging ~13.4 GB total (A2.3). Peak host RSS of the dump so far MEASURED 1.5 GiB (dgdump; main dump not yet measured; `rss_watch` is still sampling, exits itself after 24 h).

## For R1
Bank (single copy, C: only): `C:/Users/Admin/srttc_s0/work/bank/<unit>.sets.npz` (+ `.fan.npz`, `.report.json`); a unit = <= 500 tokens, <= 12 logs, content-addressed.
`sets.npz`: `log, token, step, rank, ckpt_step`, `feat [S,64,256] fp16` (input of `score_head`), `logits [S,64,6]`, `top8 [S,8]` (frozen rule: float64 sigmoid, p0*p1*(5p2+5p3+2p4)/12, stable argsort desc), `labels6 [S,64,6]` (order NC, DAC, EP, **TTC = column 3**, C, DDC; clipped [0,1]), `label_order`, controls, `rule`, `ckpt`.
Folds: rank mod 5 from `run/plan/ranked_fit_logs.tsv`; key-order prefixes per A2.1. `S0_SUMMARY.json` = counts only; `logs_without_any_token` is the A2.1 item 7 report (NOT auto-replaced; > 5 % => stop).
Read nothing beyond integrity checks until the gate phase.

## Findings to carry
- **Fit sets are not P2's own fan** (older checkpoints, 1,281 (step, rank, ckpt_step) combos) and P2 trained with `--scorer-sees-trunk`: fit-vs-gate/confirm feature domain shift is inherent to the registered design (MEASURED controls: stored-set logits differ from the fan's for 37/37 smoke sets). Flag, do not fix.
- `wait_idle` counts a running pytest as a busy GPU; the driver overrides it (pytest excluded).
- `srttc_s0_podsets.py` ran BEFORE P6 with the old guard; its `wait_idle` calls `pc.guard()`, which the rewritten `podcopy.py` no longer has. Rerun only with `--skip-fetch`.
- Single copies: the bank (C:), `work/sets/*.npz` (C:), pod fetch `work/sets_pod/` (~1 GB, deletable once S0 is banked).
- STRATEGIC family: PAUSED (PI, 2026-09-19); this stage reports no gate result.
