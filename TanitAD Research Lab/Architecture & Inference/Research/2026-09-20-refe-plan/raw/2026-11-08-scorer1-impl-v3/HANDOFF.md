# HANDOFF SCORER1-IMPL-3 (2026-11-08 ~16:45) -- written by the agent that built v3; a fresh agent finishes it
Nothing committed, staged, landed; nothing under `C:/Users/Admin/tanitad-push` or on any pod was changed (read-only git commands on the mirror only). v2/ untouched. v2's handoff is kept as `HANDOFF_V2.md` (steps A to F there; A to C/D restated in LAND_PLAN.md).

## Done (MEASURED)
1. `v3/` = copy of v2 + `files/test_rl_s1b.py`, `test_rl_epo.py`, `test_hn_screen_wiring.py` (triage patches applied to the tip blobs, LF; changed lines == patch lines). blobs.tsv +3 rows, MANIFEST v3 section, `base/` +3 tip blobs, `scorer1_impl_v3_tests.patch`.
2. Tip moved: `0fd9a3dc` -> `72151b85` (suite tree) -> `9f7df1aa` (now). **All 9 modified-file bases unchanged** (`bash v3/tools/check_bases.sh` -> ZZBASES_OK); the newer commits touch only `raw/` and GOALS_AND_CLAIMS; `refe/ code/ splits/` diff is empty.
3. Profile fingerprints, base vs v3: all 16 existing profiles unchanged (refe_p6 `80fa4d7e`, refe_p7 `94cca167`), refe_scorer1 new `88d02fb7` (`v3/fp_*.json`).
4. P7 ship list: tip package `ZZP7SHIP_OK files=296 scripts=106`; tip+v3 `ZZP7SHIP_BAD differs=9 stale_scripts=8` (6 from v2, 3 from v3). The pod's P7 copy is independent. Details in `LAND_PLAN.md`.
5. `LAND_PLAN.md` written. 6. carve_feed base control reproduced: `F:/scorer1_v3/res/base_carve_feed.log` (rc 1, same assertion as v2's final run).

## RUNNING: full suite on the final bytes
Tree `F:/scorer1_v3/t` (tip 72151b85 refe/eval/splits + v3 files; `eval/harness` removed so the environment equals the base's, whose 7 env failures need code/ and raw/). Runner `v3/tools/run_suite_v3.py`, launched detached 15:58 via `launch_detached_v3.py`: 3 workers, 2400 s per test, CUDA hidden, TMP on F:; 60 refe `test_*.py` + 4 eval tests that read launch_gate/scorer_finetune.
* **Marker: `F:/scorer1_v3/res/ALLDONE`** (last thing written). Progress `F:/scorer1_v3/res/results.txt` (`grp/test_x.py rc=N secs=S last-line`), logs `F:/scorer1_v3/res/logs/<grp>.test_<x>.log`. Expect 1.5 to 3 h. Wait with ONE Monitor until-loop (`until [ -f /f/scorer1_v3/res/ALLDONE ]; do sleep 60; done`), not turns.
* Do NOT kill the runner (python processes whose command line contains `run_suite_v3`). If dead without ALLDONE, rerun only the tests missing from results.txt.

## Exact next step
1. After ALLDONE: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/scorer1_impl_20261108/v3/tools/compare_v3.py` (writes `F:/scorer1_v3/res/COMPARE.txt`). Acceptable: all PASS except the 7 (ego_yaw, sft4r_chain, s1aux_chain, p2r_chain, rls1_chain, p7_ship_list, carve_feed), each `SAME-AS-BASE`.
2. Any other failure: rerun that test alone on the base tree `F:/scorer1_v2_base/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe` (env `CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<that dir as a Windows path> SCORER1_TMP=F:/scorer1_v3/res/tmp`, venv python; see `triage/logs/run1.sh`) and classify with evidence. rc 124 = rerun once alone. aux_g12_bf16 and gate_gpu_probes are CUDA-by-design: with CUDA hidden report them as pass/skip/fail as seen. A failure not on the base is a regression: report it, do not fix production code, do not change a pin.
3. Copy `results.txt`, `COMPARE.txt` and tails of failing logs to `v3/suite/`; put the verdict line (N tests, N pass, the 7) at the top of LAND_PLAN.md and in MANIFEST.
4. Cleanup, by literal path with PowerShell `Remove-Item -LiteralPath` (bash `rm -rf` is refused here), log path and size in MANIFEST: all of `F:/scorer1_v3` (t, res/tmp, ship, ship2, ship3, patchwork, helper .py) AFTER step 3. Not yours: `F:/scorer1_v2_refe`, `F:/scorer1_v2_base`, `F:/scorer1_triage`.
5. Hand back <= 300 words + manifest.
