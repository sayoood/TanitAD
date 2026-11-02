# HANDOFF: RECOVER-FT trainer patch (state at the end of the build agent's run 2026-11-02)

**Card:** `PREREG_RECOVER_FT_FROZEN.md` sha256 `57caba3b...` + `AMENDMENT_1.md` sha256 `05cff3f8...` (`FROZEN_SHA.txt`). Registered values untouched.
**Integration is the Master Mind's (escalated): land `stack_patch/stack/*` into `stack/`.** Nothing here is in git. Launching is not part of this package.

## LANDING (4 files; base = mirror tip `408cd81f`)
| repo path (under `stack/`) | what | base check |
|---|---|---|
| `tanitad/data/v8f_recover.py` | NEW: `RecoverDevTable` (+`quantiles`, `band_coverage`), `rejoin_target` (AMENDMENT_1), `RecoverPerturb` | new file |
| `scripts/v8f_mini_train.py` | EDIT (+286 / -2 lines; `v8f_mini_train.408cd81f_to_patch.diff`) | tip blob sha256 must be `0d708b3cb841bc1f1832b09fe61e3d8421a4d3f59769f4435c0b22bb77e268ce`; if the tip moved, apply the diff by hand, do not overwrite |
| `tests/test_v8f_recover.py` | NEW (16 tests: algebra, AMENDMENT_1 known answers, table, library mutation) | new file |
| `tests/test_v8f_recover_wiring.py` | NEW (trainer wiring; imports the `pack` fixture of `test_v8f_dagger_train.py` and `_woken` of `test_v8f_ego_perturb.py`) | new file |

The land is the Master Mind's (`land.py` replaces registers wholesale: pass `base=<blob>`). Box 2's checkout `aeb54468` is 368 trainer lines behind the tip: it must be moved to the LANDED commit before any launch.

## WHAT THE TRAINER PATCH DOES (hop by hop; each hop has a control that reads exactly 0 or goes RED)
argv `--recover-{p,dev-bank,dev-bank-sha256,dev-law,target,rejoin-steps,seed}` -> `make_recover` (bank load + sha, K-B3, table from THIS trainer's store; K-B2 deliberately NOT read) -> `RecoverPerturb` -> `declared_vs_built` (`bad_recover`, each flag `rec:<name>` in `FLAG_CHECKS`, built values from `built_levers()`, bank sha = the LOADED manifest's, disk re-hash, exclusivity with `--ego-perturb-p`, `--dagger-p`, `--ego-perturb-with-dagger`, any recover flag at p = 0 refused) -> `recover_probe` (selfcheck.json `recover_probe`) -> `b = rcv.apply(b, step)` in the loop (guard `ZZRECOVER-NOT-APPLIEDZZ`) -> `summary.json` `recover_{p,n,eligible,rows,changed,share}` (KR1; exactly 0 at p = 0) + `levers.recover` + `levers.recover_table`; checkpoint `recover`/`recover_totals` only when p > 0. The loop variable is `rcv` (the log record in `main` is a dict named `rec`).

## ACCEPTANCE (MEASURED here; logs in `logs/`)
* AMENDMENT_1: zero deviation gives `target == gt`, max abs EXACTLY 0.0, curved path, K in {1, 10}, modes rejoin and logged; the old formula reads 0.3032 m at K = 10 (`logs/amendment1_old_formula_red.txt`).
* p = 0 is bit-identical to the TIP trainer: 276/276 model tensors, optimiser moments, per-step log equal; summary differs only by the six added `recover_*` keys (`logs/crosstree_p0_vs_tip_compare.txt`).
* Reach: probe passes (inputs-only and target-only loss/gradient differences non-zero, d total/d p10f > 0, p = 0 control and prior/WM controls EXACTLY 0.0); mutations RED: library apply line deleted, `_rebuilt` returning `{}`, one tensor left unperturbed (gt / p10f / kin3 / base), non-registered target, rc not re-expressed, non-identity p = 0, loop apply line deleted.
* Table build matches an independent per-record recomputation of the card's five numbers to 1e-9 on the generator-written synthetic bank.
* The six argv lines of the card parse with the patched parser (`ARGV_ARMS.txt`, asserted equal to the card's base string and to DAG-OACC's own `argv.json` minus `--dagger-*`).
* Gate: `launch_gate.py` `_v8f_profile` is the refcv8 / PLAN_V8F profile (no mention of `v8f_mini` in `launch_gate*.py`); the v8f_mini gate is the in-trainer `--selfcheck-only` (G-DVB `FLAG_CHECKS` + `bad_*` + G-LIVE + probes), so the entries live in the trainer, not in `launch_gate.py`. Deliberate-regression: a missing `FLAG_CHECKS` entry per flag is refused (7 tests).

## WHAT THE LAUNCH AGENT MUST STILL DO ON BOX 2 (not done; no real TRAIN store exists on the dev box)
1. Check out the landed commit; TRAIN bank is `D:\tanitad\runs\v8f_mini_full\dagger_bank_dag2\s{s}` (sha in the argv lines).
2. Run each R arm's `--selfcheck-only` first. It builds the table from the real TRAIN store: **time it** (`ClipStore.get` loads each of the ~4,512 clip files once; UNVERIFIED cost) and read `selfcheck.json` `info.recover.table`, `table_quantiles`, `table_band_3_8`, `kb3` (K-B3 window-list sha must equal `e144adcd...`, MEASURED equal to DAG-OACC's at both seeds and to bank2's).
3. The gate token binds commit + argv sha256 + data sha256s (CLAUDE.md); never hand-written.
4. Register the table build summary (KR3) and the probe record with the arm.

## REGRESSION (RUNNING at hand-off, 2026-11-02 ~20:17; NOT YET A RESULT: UNVERIFIED until the verdict file exists)
Clean tip export `F:/TanitAD-scratch/recover_ft/tree_clean` (raw LF, trainer sha256 `0d708b3c...`) and `tree_patched` (= tip + `stack_patch`, `cmp`-identical to the package). Launch history: a first Bash-launched attempt ran ~50 min (A1 25 % done, identical dot pattern on tip and patched incl. the one pre-existing `F`) and was stopped by me to re-split the work; two WMI-detached relaunches were killed within ~1 min by an UNIDENTIFIED actor (log shows `^C`); the current run is `F:/TanitAD-scratch/recover_ft/regress_all.sh` as a Bash background job of the build session (it may die with the session: if `logs/REGRESSION_VERDICT.txt` is absent and the group logs lack `exit N`, re-run `sh regress_all.sh` from `F:/TanitAD-scratch/recover_ft`, scripts are in that directory). Six pytest groups on `tree_patched` (logs `F:/TanitAD-scratch/recover_ft/logs/patched_{A1a,A1b,G1,G2,G3,R}.log`; they cover the v8f suites that import the trainer: mini, ego_perturb, ego_dagger_compose, launch_intent, pgt, init_ckpt_o_interface, tac, orc_chain, cz, dagger_train, oacc, window_start, hist, grid, plus the two recover files). A poller (`regress_verdict.py`) waits for all six `exit N` markers, re-runs every FAILED / ERROR node on the clean TIP tree and writes **`logs/REGRESSION_VERDICT.txt`**: `0 regressions` or the named nodes (a node failing on the tip too is pre-existing). If the file is absent, read the six logs. NOT covered (too slow on this shared box, ~2,000 tests): `test_v8f_{flat,mixo,dscale,lora*,kindx_s1,kscore,speed*,dagger}.py`; the Master Mind's own landing suite run covers them. A first single-process full run was stopped at 2 % after 12 min (slow bash-driven `test_v8f_beta*`), so scope was cut, not hidden. Observed at the tip, same position on both trees: one pre-existing failure inside `test_v8f_mini.py` (identify it from the verdict file).
Passed already (before the final edit of the data library only added `quantiles` / `band_coverage`): `test_v8f_recover_wiring.py` 52 passed (`logs/pytest_v8f_recover_wiring_run1.log`); `test_v8f_recover.py` 16 passed. Both are in group `patched_R` again on the final files.
Tools used (copied to `tools/`; the live copies run from `F:/TanitAD-scratch/recover_ft`): regress_all.sh, regress2.sh, regress_verdict.py, sync.sh, pt.sh, crosstree_run.py, crosstree_cmp.py, table_report_test.py, gen_manifest.py.
Re-run command (any group): `CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q -rfE --basetemp=F:/TanitAD-scratch/recover_ft/bt_x -p no:cacheprovider <files>` from the tree root.

## THE TABLE (`RecoverDevTable.from_dagger_bank`)
* SYNTHETIC bank in the real layout (the repo's own generator, 383 ok records, random-weight toy rollouts; `logs/synthetic_table_report.json`): dv signed q05 / q50 / q95 = -7.45 / -0.23 / +4.78 m/s, |dv| q95 7.57; dy q05 / q50 / q95 = -0.44 / +0.29 / +1.42 m, |dy| q95 1.45; dpsi q05 / q50 / q95 = -0.24 / +0.03 / +0.41 rad; share of records with logged v in [3, 8] = 0.580 (222 of 383). Toy world: these numbers say the code runs, not what the real law is.
* REAL bank2 s0, OWN side only (`logs/real_bank2_s0_layout_and_ownside.txt`; 613,132 records, 433,301 ok, 4,512 clips, hmax 16, own pose rows finite at j-1 and j for all ok records, j in [13, 90]): own v q05 / q50 / q95 = 1.66 / 9.10 / 26.8 m/s, share of own v in [3, 8] = 0.316. The bank's stored `e_lat` (m) and `dpsi` (DEGREES, unsigned, at the matched route row) are proxies only and are NOT the card's signed same-clock-row dy / dpsi.

## UNVERIFIED
* The real table's dv / dy / dpsi quantiles and band coverage (needs the box-2 store): only the OWN side of the real bank was read here (`logs/real_bank2_s0_layout_and_ownside.txt`).
* Per-step cost of `apply` on box 2 (card: <= 2 %).
* The loop's behaviour on CUDA (`to_dev` of the nested `lab_s`): same path as the DAgger swap; tested on CPU only.
