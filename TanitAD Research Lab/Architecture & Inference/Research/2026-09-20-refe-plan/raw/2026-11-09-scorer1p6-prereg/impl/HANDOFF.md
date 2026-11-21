# HANDOFF: SCORER1P6 trainer + launch-gate overlay (2026-11-09). CPU only; nothing committed, nothing on a pod.

## What it is
An overlay on the LANDED v3 tip (`5cebd7b8`; v3 blobs verified equal to the tip). `files/` = the 11 files that differ from v3 (list in `CHANGED.txt`; full diff `p6_overlay_vs_v3.patch`). Land them into `refe/` (all LF). Re-run `check_bases.sh`-style blob check on the then-current tip first: land only if the tip blobs of the MOD files still equal the v3 blobs.

## State of verification (MEASURED on this dev box, CPU, venv tanitad, CUDA hidden)
PASS: `test_pair_loss_p6` (new), `test_scorer1p6_gate` (new; 17 existing fingerprints unmoved, refe_scorer1p6 = ea5ea864...), `test_scorer1p6_preflight` (new; 6-output stub-trunk preflight, OFF path bit-identical to pre-SCORER1 trainer, 3 mutants RED, calibration, training loop), `test_pair_loss`, `test_scorer1_gate`, `test_scorer1_mutation`, `test_scorer1_layer`.
**Full launch_gate-reading suite: RUNNING detached, 1 worker, started ~13:05.** Results: `F:/TanitAD_p6impl/res/results.txt` (one line per test, `ALLDONE` at the end), logs `F:/TanitAD_p6impl/res/logs/`. Known environmental failures that also fail on the unchanged tip (v3 LAND_PLAN list): carve_feed, p2r_chain, p7_ship_list, rls1_chain, s1aux_chain, sft4r_chain, ego_yaw, aux_navtest_forward, hn_screen_read, r0e_diagnostics, safeobj_outcome, aux_label_store, launch_gate (stale D: path), p6_planner_command, rl_fan_dump, aux_drivable_head, rls1_queue/hnscreen_chain under load. Any OTHER failure: rerun the same test on the unchanged tree `F:/TanitAD_p6impl/work/base/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe` (env: `CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<that dir> SCORER1_TMP=F:/TanitAD_p6impl/tmp/a/b`, `C:/Users/Admin/venvs/tanitad/Scripts/python.exe test_x.py`); identical failure = environmental, else regression of this overlay. Tests that read `scorer_finetune.py` text most likely to move: test_dump_scores, test_rl_s1b, test_rl_epo, test_hn_screen_wiring, test_launch_gate (static count N).

## Exact next step
Wait for `F:/TanitAD_p6impl/res/ALLDONE`, then `python tools/compare` is not provided: diff results.txt against the list above by hand, rerun unknown failures on base (above). Then land.

## Pins and decisions for the Master Mind
1. `launch_gate.SCORER1P6_DATA_PIN = None`: the gate (and the chain) REFUSE until filled with `{"train_sets": N, "heldout_sets": H}` from `labels_manifest.json` `data_expect`. Filling it moves only refe_scorer1p6's fingerprint.
2. Pod paths pinned from the coordinator's message: ckpt `/workspace/data/refe_runs/p6_paper_faithful/model_final.pt` (sha256 968a469d... bound), targets `train_grow_cmd_3c_p7`, calib `train_grow/calib_table.json`, labels `/workspace/data/p6base_labels` (`train_sets train_pdm heldout_sets heldout_pdm logs/heldout_logs.txt`; layout read from p6base/labels chain, UNVERIFIED on the pod). Chain package dir `/workspace/refe-plan-scorer1p6`, data `refe_scorer1p6` (navtest_ids and the frozen card `PREREG_SCORER1P6.md` go there; the chain checks the card sha256 adb5aafd...). HF repo `Sayood/tanitad-refe-scorer1p6` must exist (G-EGR).
3. "one G-LIVE/preflight per the card" read as: each arm has ONE trainer preflight (a token needs its own argv's data event), S1E's is followed by the 3 mutants. If you meant a single preflight for all arms, the chain needs a different gate design.
4. Chain waits for no chain lock and an idle GPU, NOT for ZZP7_DONE (P7 paused per card section 7). Python comes from `teacher_env.sh` (`DRIVERL_EVAL_PYTHON`); the coordinator named `/workspace/venv-teacher/bin/python`: make sure teacher_env.sh points there.
5. S1 (no EP, conditional O3) is in the profile and pod script but not in the chain.
6. `rl_fan_dump.py:561-563` does not touch these files; it is fixed for P6 by `p6base/labels/rl_fan_dump_nav_mode.patch` (the label job's wrapper already avoids it).
7. The hardcoded card path in test_scorer1p6_gate T7 checks the card sha only if the E: artifact exists.

## Two v3 bugs found by running the 6-row preflight (fixed in this overlay)
- `REFE_PAIR_MUTATE_DROP_LOSS` returned a graph-less zero; G16's control `backward()` then raised, so the mutant CRASHED instead of printing ZZSFT_PREFLIGHT_FAIL (the v3 chain's `grep` would have refused).
- `--pair-calibrate` was refused (rc 4, "no PREFLIGHT_PASS.json") because calibration runs before any preflight; now exempt (it trains nothing and is forbidden in a launch argv).

## Not verified
Real ViT-L trunk, real bank, bf16 on the L40; the pod preflight, calibration and mutants; the chain end to end (bash -n + argv diff only); any pod path other than as stated; `p7_ship_list.json` goes stale again (re-derive only when a refe_p7 package ships).

## Scratch (mine; delete when the suite verdict is read, by literal path)
`F:/TanitAD_p6impl/work` (55 MB), `F:/TanitAD_p6impl/tmp` (<1 MB), `F:/TanitAD_p6impl/res` (suite logs, size at the end), helper scripts `F:/TanitAD_p6impl/{env.sh,launch_suite.py,patch_lg.py,patch_chain2.py}`. Nothing deleted so far.
