# B4: HC40 per-proposal labels for C1's proposals (plan; the script `hc40_label_run.py` is a skeleton and has NOT been run)

Card: `PREREG_VDAC_TRK.md` (sha256 099a429f...), sections 4, 5, 9 B4. Purpose: the calibration of the margin m* (section 5) needs, for each of the 40 HC40 navtrain logs, C1's 64 proposals scored by NAVSIM's harness with TRACKED execution, so that the VT rule's PDMS gain vs C1's pick can be computed on navtrain without ever reading navtest.

## 1. How `tr` is produced for navtest (the thing B4 must match), quoted

Producer of `full_C1_s*of2.jsonl`: `raw/2026-11-20-p5c-next/p5c_prop_score.py` (`ZZPROP_SHARD_DONE C1 0/2 6201 4393.5 s` in `full_C1_s0of2.log`). Verbatim:

- docstring: "per-proposal NAVSIM v1.1 sub-scores for all 64 proposals of a P5-C arm, tracked AND transparent, ONE scorer call per token." and "tracked[k] the 8-pose proposal k handed to NAVSIM's LQR + bicycle (the harness's input form; proposal[pick] == seam poses)".
- core: `parts = TS.load_cache_parts(CACHE, lg, t)` (navtest metric cache); `tracked = [np.asarray(P[i, k], np.float64) for k in range(64)]` with `P = z["proposals"]` of shape `[N, 64, 8, 3]`; `res, _ = TS.score_arms(parts, tracked, ex)`; `"tr": pack(res[:64])` (nc, dac, ddc, ttc, c, ep, pdms, each a list of 64). `TS` = `eval/instruments/transp_score.py`; EP is recomputed pairwise (raw_j / max(raw_PDMS-Closed, raw_j)), so no proposal's EP depends on another's.
- the LQR tracker is inside `PDMSimulator.simulate_proposals` (the "tracked" frame); the proposals are the NAVSIM-grid 8-pose plans = `props8`.

The HC40 equivalent is `sh_label.py` (`raw/2026-11-28-safehead-data-x150/code/sh_label.py`, byte-identical to the 2026-11-27 copy; verified by `diff`). Verbatim docstring: "TRACKED NAVSIM v1.1 PDM labels for all 64 proposals of every dumped token (navsim venv; NAVSIM's own simulator + scorer). Same call the programme's labeller makes (onpolicy_relabel_pdm.label_log): scenario at the token with history 40 / future 100 rows (navtrain_scenarios.build_scenarios_for_log) -> navsim_pdm_targets.metric_cache_parts -> score_plans(parts, plans8) where plans8 are the EXECUTED plans on NAVSIM's [64, 8, 3] grid = the dump's `props8` (A7 last-heading repair, to_navsim) -- the very poses the harness scores for a pick. LQR tracking is inside PDMSimulator.simulate_proposals".

Both paths feed the same `[64, 8, 3]` poses to the same simulator and scorer; they differ only in where the metric-cache parts come from (a prebuilt navtest cache vs a scenario built from the nuPlan DB at the token). The parity of the labeller with the harness was measured by the data agent's `sh_ctrl.py label` (CTRL-LABEL: tracked labels of the PICKED proposal vs the harness csv `refe_stage2_p5c_c1.csv`, per-component max abs diff, on navtest tokens whose pixels are local). That result is INHERITED here (read it in `raw/2026-11-27-safehead-data/ctrl/` before the first HC40 run); B4 adds the checks in section 4.

## 2. Input contract (what B2's `vdt_dump.py` must emit for HC40, in addition to the `vdt_read.py` keys)

`hc40_dump.npz`: `token [N]`, `props8 [N,64,8,3] float32` (= `sh_dump.py` `props8` = `to_navsim(planner.executed(traj, j)).astype(float32)`, the seam's `--dump-proposals` conversion, refe_navtest_seam.py:296-298), `log [N]` (or a `token -> log` json). Proposal axis j is the same axis as `s_plan`'s, so `tr_*[:, j]` aligns with `dk_*[:, j]` and `agg32(s_plan)[:, j]` in `vdt_read.py calibrate`.

Frames are NOT needed by the labeller (it reads the nuPlan DB and the maps only). The frames are B2's input for the C1 forward and the head's field.

## 3. Procedure (what `hc40_label_run.py` does)

1. Read the HC40 list (40 logs, sha256 recorded beside the card by the build) and the dump; REFUSE if a dumped log is not in the list or the list is not 40 logs.
2. Write `<work>/dump/<log>.dump.npz` (`token`, `props8`) per log.
3. For each log, run `PY_N sh_label.py --dump ... --db <nuPlan DB of the log> --out <work>/lab/<log>.lab.npz --workers 6` in the navsim venv, with `sh_env.nenv()`'s environment (CPU only: `CUDA_VISIBLE_DEVICES=-1`, `REFE_SIM_HZ` unset = the 20 Hz default, one thread per BLAS). Resumable (an existing `.lab.npz` is skipped). One log at a time, as `sh_run.py` does (RAM).
4. Report `<work>/label_report.json`: asked vs labelled per log (failures stay in `<out>.report.json`, never silently dropped), finite and in [0, 1].
5. `--det-check`: label the first log twice, require byte-identical output (`save_npz_det` writes fixed zip stamps), the same CTRL-DET as the SAFE-HEAD cache.

Output = the `--lab-glob` of `vdt_read.py calibrate` (`<log>.lab.npz`: `token`, `tr_nc, tr_dac, tr_ddc, tr_ttc, tr_ep, tr_c, tr_pdms`, each `[n, 64]`).

## 4. Controls and gates before the labels are used (each can FAIL)

| id | check | where |
|---|---|---|
| CTRL-HC40-SET | the labelled logs are a subset of the 40-log list, the list is 40 logs, its sha256 equals the one recorded beside the card | `vdt_read.py calibrate --hc40-list` |
| CTRL-HC40-DISJOINT | no labelled log is in navtest's 136 logs (`l2next_read.py prep` writes `navtest_logs.txt`) | `vdt_read.py calibrate --navtest-logs` |
| CTRL-SEL-HC40 | `agg32(s_plan).argmax == k_ship` on every labelled HC40 token (C1's pick is what the rule falls back to) | `vdt_read.py calibrate` |
| CTRL-DET-LAB | two label passes on one log are byte-identical | `hc40_label_run.py --det-check` |
| CTRL-LABEL (inherited, to re-confirm) | the labeler's tracked label of the picked proposal equals the harness csv on a few navtest tokens | `sh_ctrl.py label` (data agent) |
| completeness | `labelled / asked` per log, and the total; tokens failing the 5 s future guard are reported, not imputed | `label_report.json` |

A calibration run on a token subset that lost tokens to the labeler's guard is a calibration on the labelled subset; the count is in `cal.json` (`n_hc40_tokens`) and the report must quote it.

## 5. Cost and risk (ESTIMATED)

- Labeler throughput on navtest was 6,201 tokens in 4,393.5 s on one shard (0.71 s/token, 64 proposals); the HC40 token count is not known until B2 selects tokens; at that rate every 1,000 tokens is about 0.2 h of CPU, serial (ARITHMETIC on the navtest shard rate, not measured on navtrain).
- RAM: `sh_run.py` gates a label job on available RAM >= 4,500 MB and free commit >= 8 GB; this skeleton does not implement the gate (run it under the same chain or add the gate).
- Window mismatch risk: `sh_label.py`'s scenario builder skips tokens whose row index leaves no room for the 40-row history / 100-row future (5 s guard). The calibration must report that count. If it is large the HC40 set needs the same guard applied when B2 selects tokens (`sh_lib.guard_tokens`).

## 6. Not decided here

The HC40 list itself and its sha256 (the card assigns them to the build before any head trains; the rule stated in the card is "the first 40 by sha256(log name)" among the `label_store` logs, which is B1/B2's artifact). `hc40_label_run.py` only consumes it.
