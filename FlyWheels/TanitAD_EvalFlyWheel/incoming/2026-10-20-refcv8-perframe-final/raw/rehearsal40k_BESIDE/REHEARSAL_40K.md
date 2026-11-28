# 40k REHEARSAL, BESIDE, not binding: SHIP-R8 per-frame CPU lane + verdict on `step40000ema_pf`

*40k REHEARSAL, BESIDE, not binding; clip ids sha12 only (this page carries none).* Tier: T1 self-action open loop on logged eval139 frames, not driving performance. Estimator: paired episode-cluster bootstrap, B 2000, which answers "another draw of EPISODES" only. Run 2026-10-20 13:04-13:17 +0200, CPU only, `OMP_NUM_THREADS=4`. Free commit memory stayed at 27.5-27.6 GB before every stage (gate needs 5 GB).

## Lane: all 7 stages OK on the first try (MEASURED, `timings.tsv`, `rehearsal.log`)

| stage | result | wall s | artifact (under `pfpkg/raw/`) | own controls |
|---|---|---|---|---|
| install | OK, `rs_table_loads_and_matches: true`, installed `2aacc9ff32d3...` | not timed | `../PF_LANE_INSTALL.json` | |
| clamp | OK | 64.5 | `clamp_posthoc/trees/step40000ema_pf.json` | K1_all_exact, ALL_PASS |
| compliant | OK | 103.7 | `compliant_reread/trees/step40000ema_pf.json` | C1, C1v, C2, C3, C4, ALL_PASS |
| tf_forward | OK | 231.1 | `retime_trainfit/FORWARD_TRAINFIT_step40000ema_pf.json` | ALL_PASS, const ALL_PASS |
| d8 | OK | 119.1 | `retime_deploy/FORWARD_DEPLOY8_step40000ema_pf.json` (+2 npz) | ALL_PASS, identity vs TRAINFIT pass |
| df | OK | 193.9 | `retime_deploy/FORWARD_DEPLOY_step40000ema_pf.json` | ALL_PASS |
| tac | OK | 24.3 | `retime_deploy/TAC_step40000ema_pf.json` | ALL_PASS, clause5 PASS |
| parity | OK | 2.6 | `retime_deploy/PRIOR_PARITY.json` | known values + source trace pass |

**The 7 stages took 739.2 s (12.3 min) in total.** For comparison, feed parity on the same 40k tree took about 7 s and 6 s per seed (`battery40k_pf/b40pf.log` marker deltas), and the verdict took 0.2 s.

**One deviation from the chain (rehearsal only):** clamp, compliant and tac REQUIRE `<archive>/<tree>/PAIRED_READ_<tree>.json` (`clamp_posthoc_r8.py:386`, `compliant_reread_r8.py:456-457`, `tac_retime_r8.py:190`), and the brief forbids putting a `PAIRED_READ_*` file in the real archive. So the tree was NOT copied there. Instead `rehearsal_lane.py` runs `pf_lane.main()` unchanged, with CP/CR/TAC/PP `.ARCHIVE` pointed at the original read-only root `C:/Users/Admin/refcv8_pf/battery40k_pf`. Tonight this does not apply, because the chain's GPU stage writes `PAIRED_READ` into `$WORK` and asserts it is there before the lane starts.

## Verdict: `overall` **PASS** (ORIENTATION ONLY; `ship_r8/SHIP_R8_step40000ema_pf.json`, run with `--archive` = the original root)

| clause | status | seed 0 | seed 1 |
|---|---|---|---|
| 1 registration + TRAINFIT/DEPLOY8 controls | PASS | the registered, frozen, provenance, gate and FORWARD_TRAINFIT hashes all equal `2aacc9ff32d3` | |
| 1b feed parity | PASS | 4754/4754 windows, 0 only-pf, 0 only-ref, 1157 token != clip, 2802 args known, 8/8 checks true | same numbers |
| 2 E1 S2 ADE − ha0_ext_pose | PASS | −0.0390 [−0.0507, −0.0278] n 4159/138 | −0.0377 [−0.0491, −0.0264] |
| 3 E3 S6 ADE − ha0_ext_pose | PASS | −1.0046 [−1.3211, −0.6989] n 3251/138 | −0.9993 [−1.3097, −0.6967] |
| 4 FZR-NR ADE/LON/LAT | PASS | the three families PASS, FZR-LAT HOLDS, no row separated worse | |
| 5 TAC vs refcv7-50,400 | PASS | S2 lat +0.0048 [0.0005, 0.0093], S2 lon +0.0245 [0.0116, 0.0382], S6 lat +0.0123 [0.0009, 0.0235], S6 lon +0.0111 [−0.0118, 0.0327] | S2 lat +0.0043 [0.0005, 0.0083], S2 lon +0.0156 [0.0029, 0.0287], S6 lat +0.0102 [−0.0010, 0.0209], S6 lon +0.0046 [−0.0181, 0.0261] |
| 6 strategic | N/A | NOT APPLICABLE, n = 0 | |
| 7 prior parity | PASS | 4754 windows, per-slot max 0.0 m | 4754 windows, 0.0 m |

**Negative control:** the same verdict against the real archive (tree absent) reads **INCOMPLETE** (clauses 1 and 1b INCOMPLETE). It never reads PASS.

**Precedence:** the gate time is the mtime of `REGISTERED_PF_TABLE.sha256`, 10:19:21.78. Commit `26adacf` landed 10:18:57. The rolls started 10:20:38 (`b40pf.log`), and the manifests were written 12:16:03 (s0) and 12:47:59 (s1). **The check PASSES**, and it also holds under the stricter roll-start reading, by 77 s. **Where it is ambiguous:** (a) a gate built after the fact, using the chain's literal `time.time()`, would read FAIL. (b) The verdict compares against the manifest mtime, which is written at the END of a roll. So the check proves "manifest written after the gate", not "roll started after the gate". Tonight's chain writes the gate before the GPU stage, so this is structural for 48,852.

## For tonight's 48,852 run

* No defect, failure or retry in the lane. The CPU lane needs about 12.5 min from bank to verdict on a quiet box. A concurrent CPU video render would slow it (UNVERIFIED). Peak lane RSS was not measured (UNVERIFIED).
* The clamp stage writes about 35 MB of scratch to `R8_CLAMP_SCRATCH`. The default is this session's scratchpad; it uses `makedirs`, so it is safe even if that directory is removed.
* Latent, not triggered tonight: `PRIOR_PARITY.json` is the only lane artifact without a tree tag, and `lane()` skips a stage whose artifact exists. Clause 7 never checks which tree the panels cover. The live `pfpkg` does not exist yet (MEASURED), so it starts clean. The optional hardening is `PROPOSED_PATCH_clause7_tree_check.diff`. It was tested on a mutated file: base reads PASS, patched reads INCOMPLETE; on the real file both read PASS.
