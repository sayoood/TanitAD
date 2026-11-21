# HANDOFF_RBDREAD (2026-11-18, read-builder agent)

State: `navrbd_read.py` BUILT; tests and mutation arms green (counts in the report); smoke on stand-ins done; the REAL read is not run (no N1D dump exists).
Nothing staged, committed or pushed; the pod was not touched.

## Dumps the MM must produce (all CPU, 2 threads, one environment)
1. **Held-out, pod** (staged by b4, `/root/strat1/navrbd/dump_rbd_all.sh`, waits for the chain DONE line): `dump_rbd.sh ARM follow 0` for N1D_s0, N1D_s1, N1Dx_s0 into `/root/strat1/navrbd/out/t1/`.
2. **Held-out zero variant, pod** (Z-USE; two runs, about 0.35 h each, E): `bash /root/strat1/navrbd/dump_rbd.sh N1D_s0 zero 0` and `... N1D_s1 zero 0` into `out/t1_zero/` (tag `NAVRBD-HO-N1D_s0_zero`).
3. Pull every `dump_NAVRBD-HO-*` directory (follow, zero and N1Dx) into `F:/TanitAD-scratch/strat1/navrbd/read/dumps/` (flat). N1R pair is read from `step0/t1/` (no copy).
4. **eval-141, dev box** (guard, Z-EVAL, K-RE): pull `runs/N1D_s{0,1}/train/ckpt_last.pt` from the pod to `read/ckpts/N1D_s{0,1}/ckpt_last.pt` (sha256 vs the pod), export the build commit to a tree and run:
   `set NAVRBD_TREE=<tree>` then `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u navrbd_dumps.py NAVRBD-N1D_s0 NAVRBD-N1D_s1 NAVRBD-N1D_s0Z NAVRBD-N1D_s1Z` (from `read/`; WMI launch as BRIEF_COMMON; Git `usr\bin\bash.exe` in the .cmd).
   It wraps the unedited otband `run_dumps.py`, swaps in `v8f_t1_eval_rbd.py`, passes `--nav-rb on --nav-rb-feat exitdist --nav-rb-npz b2/tables/navtext_nav_rbd.npz (sha 298049df) --nav-rb-variant follow|zero`, grid and join from `strat1/read/inputs`. Output `read/dumps141/`. ESTIMATED 4 dumps x 5-7 min (STRAT1 rate). UNVERIFIED: it has not run against a real `v8f_t1_eval_rbd.py` (the tree was not exported when this was built); the pure parts are tested. If the pod holds the eval-141 inputs, the same `dump_rbd.sh` pattern with `--eval-grid/--eval-join` of eval-141 and tag `NAVRBD-N1D_s0` works too; the dir name must be `dump_NAVRBD-<ARM>_m10_causal_heads`.
5. STRATEGIC: run the NAVRB2 view / `navrb2_strategic.py` machinery on N1D (as `launch_navrb2_read.sh` steps 3-4) and pass its `route_cost` file with `--contrast`; without it RBD-GAIN prints as RBD-GAIN-PENDING-STRATEGIC and the status is NOT-YET. Fog stays unscored.

## The read
`cd F:/TanitAD-scratch/strat1/navrbd/read` then
`C:/Users/Admin/venvs/tanitad/Scripts/python.exe navrbd_read.py --out out/RESULT_NAVRBD.json [--contrast J] [--ho-zero-hist-mask M.npy] [--hmax 16]`
(env `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2`, commit headroom 5.5 GB first). Exit 1 on a control problem. Result `out/RESULT_NAVRBD.json` + `.md`.
Runtime MEASURED on stand-ins: full read incl. eval-141 170 s, held-out only 9-90 s; with real dumps ESTIMATED 3-5 min. Run tests first: `pytest -q test_navrbd_read.py` (whole file), mutations `python mutate_navrbd.py`.
Held-out Z-EVAL needs a bool mask of all-zero-channel windows; without it Z-EVAL runs on eval-141 only and the held-out one reads NOT-YET.

## Reading choices (printed in the result, written before any number)
Bar = max(0.0383, F_HO) with F_HO unrounded (0.042364); GAIN is delta >= bar with lower bound above 0; Z-USE separated = held-out TAC-LAT real - zero separated; floor-high = F_HO above 0.0383 and above the signed delta at every seed; an N1Dx gain stands every N1D label down.
