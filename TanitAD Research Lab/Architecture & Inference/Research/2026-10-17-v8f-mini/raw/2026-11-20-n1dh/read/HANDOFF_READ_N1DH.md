# HANDOFF_READ_N1DH (card f1df05bc + Amendment 1, code f2e91e33), 2026-11-21, read-builder agent
State: all four parts BUILT; the real read NOT run (no N1DH dump or checkpoint exists). Nothing staged, committed or pushed. No GPU. Tree `tree/` = `git archive f2e91e33 stack taniteval` + hb wrapper (74fd8445): 3,439 of 3,439 blobs equal (`tree_blobcheck.txt`).
Replay (N1DH slots = N1DC dumps): `REPLAY_N1DC.json` GREEN, 216 rows (both executors) equal to RESULT_N1DC_D1HO.json (delta exact, lo/hi 4 dp, bar 1e-12), K-RE pass; `REPLAY_N1DC_MUTANT.json` RED (one arm's TAC-LAT shifted a window: 7 rows differ).

## Parts
1. `n1dh_ho_read.py` (+ `read_lib.py`, `replay_n1dc.py`): N1DC reader, arm renamed. Deciding cell D1-HO TAC-LAT N1DH - N1D per seed, bar max(0.0383, F_HO, |N1D draw|, |N1DH draw|); beside, no bar: N1DH - N1R, N1DH - N1DC, 26 R1 cells, Z-USE (`_zuse` = posterior zeroed), zero twin, T0 plan. Card outcome names; guard part prints PENDING-GUARD; POST-WEAK needs `--zpost`.
2. `n1dh_zpost.py` + `n1dh_t0route_lib.py` (Z-POST). The dumps carry NO route_logits (`v8f_t1_eval.py:277` HEAD_KEYS; model emits it at `v8f_mini.py:1198`). The tool runs the pinned evaluator in process with `--replan-steps 30` (one tick) and records the route head at tick 0. Controls K-ALIGN (plans = dump `ol`), K-OL-REAL, K-PERM, K-CONST. Smoked on a SYNTHETIC route-head checkpoint (eval-141 inputs): machinery only.
3. `n1dh_klim_run.py`: runs `n1dh_klim.py` on a real probe batch (the evaluator's own tick-0 batch, `--limit logged`): logged vs causal (one-row) vs off, plus the evaluator's own 4 s causal rule. Controls K-SELF, K-ZERO, K-PERM, K-ALIGN.
4. `eval141/`: chain `run_n1dh141.sh/.cmd` (steps resumable, commit gate 5.5 GB, CPU 2 threads; four families, hb0.2, N1DH - N1D, F2, STRATEGIC + S1-slot view, distance-keeping, per-SEED guard with the N1DC cost cells read first and the HYPOTHESIS printed beside; K-HB0-141 re-run in this tree). `eval141/dry_check.py` = PASS (`DRY_CHECK.json`).

## Commands (MM)
Pull N1DH + N1DC dumps (pod = tar + pscp only; log the removals):
`source C:/Users/Admin/lorapod/podenv.sh; pssh 'mkdir -p /root/strat1/n1dh_readtmp && cd /root/strat1/n1dh/out && tar cf /root/strat1/n1dh_readtmp/n1dh.tar --exclude="ticks_*" --exclude="log_*" --exclude="DONE_*" --exclude="hb_*" --exclude="t1v8f_*" hb0.2 hbnone'`, then pscp it into `in/n1dh/`, `tar --force-local -xf`; same for `/root/strat1/n1dc/out` into `in/n1dc/`; remove both pod tars.
Read: `cd read_n1dh; C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -u n1dh_ho_read.py --out out_real [--zpost RESULT_N1DH_ZPOST.json]` (comparators from `F:/TanitAD-scratch/n1dc/read/in`, flag `--cmp-in`). First read cost ~18 min (new files; Defender).
Z-POST: `python -I n1dh_zpost.py --tree TREE --arm N1DH_s0 CK0 --arm N1DH_s1 CK1 --profile ho --out RESULT_N1DH_ZPOST.json [--hb-dir DIR --real-dump N1DH_s0 DIR]`.
K-LIM: `python -I n1dh_klim_run.py --tree TREE --ckpt CK --profile e141 --max-windows 480 --out KLIM_s0.json`.
Eval-141 chain: copy `ckpts/N1DH_s{0,1}/ckpt_last.pt` + `ckpts/SHA256_PODS.txt` (pod `sha256sum`), then WMI-launch `eval141/run_n1dh141.cmd` (~60 min).

## Left
* Z-POST `--profile ho` needs D1-HO inputs (held-out clip caches, ~10 GB) that are NOT on the dev box: run it on the pod CPU after the chain (pass `--hb-dir /root/strat1/n1dh/hb`, pod defaults are the /root layout) or pull the caches (MM call). K-LIM runs on the dev box (eval-141).
* The final outcome label needs reader + eval-141 guard + Z-POST joined; not auto-combined.
* Blockers: N1DH checkpoints and dumps (pod chain after ROLLHIST).
