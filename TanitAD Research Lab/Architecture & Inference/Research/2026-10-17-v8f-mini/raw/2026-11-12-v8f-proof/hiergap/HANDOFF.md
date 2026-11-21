# HANDOFF hiergap (build agent, 2026-11-13; dev box CPU only; nothing staged, committed or pushed)
Folder: `F:/TanitAD-scratch/v8proof/hiergap/`. Rules first: `GAPS_DEFS.md` (A1..A8, B1..B6, controls g1..g4), written before any metric was computed. Code: `hierrep_gaps.py` (imports `hierrep_read` from `pod/stageB`, `HIERREP_STAGEB` env overrides; taniteval from `TANITEVAL_ROOT`, default the pinned dagt1 tree). Tests: `test_hierrep_gaps.py` (6 synthetic tests, all pass, 3 s).

## What the lead pulls from the pod, per arm (24 arms: HIER, FLAT, DAG, FLATDAG x seeds 0..5)
Under the Stage B T1 root `<T1>/<ARM>_s<S>/`:
* `dump_<ARM>_s<S>_m10_causal_heads/ep*.npz` (141 files, about 6 MB per arm; older-format measure 6.1 MB)
* `ticks/ticks_<ARM>_s<S>_m10_causal_heads.npz` (about 18.4 MB per arm; part B needs it; exists when `HR_TICKS=1`, the chain default)
Total about 24.5 MB per arm, about 590 MB for 24 arms. Same folder layout as `hierrep_read.py` reads, so the same pulled `t1/` root serves both. Optional: `<runs>/<ARM>_s<S>/summary.json` for the completeness check (`--runs-root`).
Dev-box inputs already built, nothing to pull: `tables/lead_table.npz` (0.27 MB) and `tables/label_table.npz` (10 KB) from the lead file `v9labels/inputs/lead_eval139.npz` and `join_eval141.npz`.

## Command lines (dev box, Git bash, `CUDA_VISIBLE_DEVICES=-1`, venv python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`)
* Rebuild tables (only if the sources change): `python hierrep_gaps.py tables --lead-npz E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_eval139.npz --join-npz "<join_eval141.npz>" --out-dir F:/TanitAD-scratch/v8proof/hiergap/tables`
* One arm: `python hierrep_gaps.py arm --dump <T1>/HIER_s0/dump_HIER_s0_m10_causal_heads --ticks <T1>/HIER_s0/ticks/ticks_HIER_s0_m10_causal_heads.npz --tables-dir tables --out out/HIER_s0`
* The read (both stages, B 2,000): `python hierrep_gaps.py read --t1-root <T1> --tables-dir F:/TanitAD-scratch/v8proof/hiergap/tables --out-dir <OUT> --pairs HIER:FLAT,DAG:FLATDAG --seeds 0,1,2,3,4,5 [--runs-root <runs>]`
  Outputs `RESULT_HIERGAP.json`, `VERDICT_HIERGAP.md`. Exit 0 valid, 2 a control failed, 3 refused (completeness). Measured on older dumps: 99 s for 3 seeds and one pair at B 2,000; expect about 6 to 10 min for 6 seeds and two pairs (ESTIMATED).
* Smoke (older tags, not HIERREP arms): `python hierrep_gaps.py read --t1-root F:/TanitAD-scratch/v8proof/wsp/t1 --tables-dir tables --out-dir smoke_read_full --pairs HIER:FLAT --seeds 0,1,2 --map smoke_map.json --no-card-counts`

## Named gaps that remain
1. The lead does not react to the rolled ego (logged scene); only the along-path offset enters the gap; lateral departure is counted (offset above 2.0 m), not scored.
2. Lead file carries no track id or class; knots without a lead have no gap (74 of 141 eval clips carry a lead frame).
3. Part B: the label is multi-hot, so no argmax-selection confusion exists (token-wise rows: STOP_POINT decision, lon STOP class, goal macro-F1); ticks 1 and 2 are scored against the logged frame label (tick 0 printed beside the pooled row).
4. g1 (shuffled-arm labels) is NOT-INFORMATIVE below 4 seeds; with 6 seeds it decides. The smoke had 3 seeds.
5. STRATEGIC: PAUSED (PI item 25).
6. Not wired into the chain: `chain_hierrep.sh` does not call this tool; the lead runs the read after the pull. The read does not enter the card's labels.
