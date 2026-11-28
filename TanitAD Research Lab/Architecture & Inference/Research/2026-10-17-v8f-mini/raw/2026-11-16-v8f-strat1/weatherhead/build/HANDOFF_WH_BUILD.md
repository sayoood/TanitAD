# HANDOFF (WH form (a) build), written for the MM / a fresh agent. State at the end of the build, 2026-11-16 late evening.

Built and tested, NOT run, NOT launched, NOT committed. Everything is in `F:/TanitAD-scratch/strat1/whead/`; read `BUILD_REPORT_WH.md` (hops, mutation table, deviations) and `MANIFEST_WH.md` (sha256 of every deliverable, diff `diff_a4357e79.patch`).

## Exact next steps (the MM)
1. Integrate: commit the 5 tree files (3 edited, 2 new tests) and copy `build/wh_*.py`, `build/test_wh_study.py`, `build/make_manifest.py` into the repo's study-script folder. Land after the STRAT1 seeds / LON-PLAN readout restarts are done (a land that touches `stack/tanitad` between a record bind and the chain restart invalidates the bound record: memory `land-closure-change`).
2. G-SUITE full: run the full suite on a clean export (pod / Thor env as for STRAT1); `wh_check.py suite --base-tree <a4357e79 export> --out-dir <runs>/gate --full --tests <all>`. The subset result (238 tests new, 211 base, 0 regressions) is already in `run/gate/gsuite_result.json`.
3. Per nav family once: `wh_check.py geval text --runs R` and `geval v9` (about 6 min each, CPU, commit-gated).
4. Per arm: `wh_check.py mint ARM --runs R [--family v9 for C0A_s0 / C0A_s1]` (13 arms; each about 1 min). A token binds tree + build sha256: re-mint after any edit.
5. Launch: `powershell -File launch_wh_wmi.ps1` (job_wh.cmd -> launch_wh.sh -> `wh_chain.py all`). The chain refuses without all 13 tokens, takes `devbox_gpu.lock` for every GPU fit (waits while the GPU reads >= 3,500 MiB used, never forces it), commit-gates the CPU stages (>= 5.5 GB free), and ends with `ZZ WH CHAIN DONE ZZ` or `ZZ WH CHAIN FAIL ZZ <why>`.
6. After P1 and P2 (`state/fit_H_s1.done`) the reader has the detection read at both seeds; after P4 the wording control; after P6 the floor. `run/read/WH_READ.json` and `ctx_output_map.json` are written by `wh_read.py` from the card's rules.

## State of the dev box at the end of the build
* GPU lock held by another job (`refe-stage2-p7_full_rep`) and 5.9 GB of 8.2 GB GPU memory used by others: no GPU use by this build; all smokes ran on CPU.
* `P-CACHE` md5 count against the pod train manifest: not computed (a file the MM pulls: pass `--pod-train-manifest`).
* The train cache has no `meta.json` (encoder provenance reads "not recorded", as the card allows).

## Deletions by this build (literal paths; sizes in the report's section 7)
Logged in `BUILD_REPORT_WH.md` section 7 (about 3.2 GB of own temp data: two pytest basetemps, the base export, TMP, SMOKE dumps, demo tokens).
