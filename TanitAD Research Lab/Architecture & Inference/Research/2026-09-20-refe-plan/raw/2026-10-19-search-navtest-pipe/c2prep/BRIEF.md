# BRIEF: C2 seal prerequisites -- the carve-B fixture on the FINAL navpipe code, then the full mutation re-run (Master Mind, 2026-10-20)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- Work dir `C:/Users/Admin/par18/c2prep/`; job work dir `C:/Users/Admin/c2prep_run/` (C: has ~60 GB; D: is nearly full, do not use it for scratch). CPU only. Never commit or push.
- No navtest label, no B-REST or C1/dose output is read. Carve B only.

## Why
`BACKLOG SEARCH-NAVTEST-PIPE`: before `seal_c2_amendment.py`, the carve-B fixture must read `ZZNAVPIPE_FIXTURE_IDENTICAL` (reproduce Step B's P2HEAD read) on the FINAL code (tip, after `43c015a` c2pin). The last fixture (old code) STOPPED on `RAM_LOW 1.6 GiB < 3.0` after 3 of 5 chunks (`raw/fixture_result.json`); it was never a defect verdict. A full mutation re-run (all guards, ~125) belongs with the seal (c2pin ran only 12).

## Start
Pin the tip once. Export, keeping the repo layout, `.../2026-09-20-refe-plan/raw/2026-10-19-search-navtest-pipe/`, `.../raw/2026-10-19-search-p2-confirm/`, `.../raw/2026-10-19-l2-stepB/` (INCLUDING its npz shards and `g0_traingrow_scan2.json` if it exists anywhere on the tip; c2pin skipped them for disk), `.../Research/2026-10-18-ep-rl-lit/l1_stepA/`. Read the package `HANDOFF.md` (how `navpipe_chain.py --mode fixture` is run, its markers) and `c2pin/HANDOFF.md`.

## Do
1. ONE detached script (WMI `cmd /c ... < NUL` pattern, see `C:/Users/Admin/par18/spc/` launch logs for the exact form) that runs SEQUENTIALLY, never in parallel (RAM): (a) the fixture chain, all 5 chunks, `--threads 4` (Step B's setting), work dir under `C:/Users/Admin/c2prep_run/`, the RAM guard left at 3.0 GiB; (b) after the fixture ends (either marker), `mutation_check.py` over ALL guards on scratch copies -> `raw/mutations_c2prep.json`. Each step writes its own log and a done marker.
2. Before launching, check free RAM; if < 10 GiB free, write that in the HANDOFF and launch anyway only if the chain's own RAM guard is intact (it stops cleanly; that is acceptable).
3. Write `HANDOFF.md` (<= 600 tokens): the PIDs, logs, markers, and the exact finisher steps: read the verdict; IDENTICAL => stage `raw/fixture_result_c2prep.json` + logs + `raw/mutations_c2prep.json` into `out/<repo path>` with `out/paths.json` (`[repo path, abs src, "raw"]`); DIFFERS => name the differing array from the result json, do not fix code. Then END your turn: do not wait on the job.

## Hand-back (<= 120 words)
PIDs, log paths, RAM at launch, and the handoff path.
