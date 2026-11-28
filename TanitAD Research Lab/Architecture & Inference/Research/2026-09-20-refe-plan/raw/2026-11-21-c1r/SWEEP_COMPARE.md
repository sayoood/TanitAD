# C1r land: full REFe sweep vs the landed baseline (MM, 2026-11-21)

Tree swept: `C:/Users/Admin/swp/q` (the baseline tree plus the C1r profile change: `refe/launch_gate.py`, `refe/gate_checks.py`, nine `refe/test_*.py`; the land's `refe` files are byte-checked against this tree by `build.py`).
Runner: `mm_sweep4.sh` (serial, shared pytest mutex, `commit_guard --min 6`, per-file TMP/TEMP/TMPDIR + basetemp on F:, deleted after each file), list `mm_sweep_list.txt` (143 files).
Baseline: `mm_sweep_baseline_c98e6114_results.tsv` (the same list on the same tree before the C1r change, 2026-11-20 15:09-19:07 local).
C1r: `mm_sweep_c1r_results.tsv` (2026-11-21, ended 06:33Z). The dev box carried other sessions' CPU jobs throughout (rollout reads, v8f T1 evals, camera evals, a smoothgap pytest), and the sweep ran about four times slower than the baseline did.

## Result: 127 of 143 files have the same exit code; 16 differ

### 5 files fail in the baseline and pass on the C1r tree
| file | baseline | C1r | why |
|---|---|---|---|
| `refe/test_div_gate.py` | pytest rc 1 (1 error) | script rc 0 | the profile-set pin: the baseline's registered profiles were not the set the test lists; the C1r change updates the list (37 -> 38 profiles) |
| `refe/test_dose_gate.py` | pytest rc 1 (1 error) | script rc 0 | the same profile-set pin |
| `refe/test_field_p7_profile.py` | script --fast rc 1 | rc 0 | the same profile-set pin (file is in the land) |
| `refe/test_p7_integration.py` | script --fast rc 1 (G-RESERVE FAIL: reserve audit on the local bank / metric cache) | rc 0 | data-state dependent (not in the land, not touched by the change) |
| `stack/tests/test_r9gate3_study_e3.py` | pytest rc 1 (46 passed, 3 errors: `shutil.Error` in a copytree of `taniteval/results/bench/...`) | rc 0 (49 passed) | environment (not in the land, not touched by the change) |

### 11 files hit the 900 s per-file timeout (rc 124) on the C1r tree and passed in the baseline
`test_hn_ctx_anchor`, `test_hnscreen_chain`, `test_hnscreen_ctx_chain`, `test_p2rep_chain`, `test_p5c_chain`, `test_p6_chain`, `test_p6_levers` (`--fast`), `test_p7_chain`, `test_rl_r0c`, `test_rl_safeobj` (all baseline rc 0), `test_rls1_queue` (baseline rc 1).
A timeout is not a pass. These were re-run alone with a 3,600 s limit (`mm_sweep5.sh`, `TO=3600`); results in `mm_sweep_c1r_to3600_results.tsv` and below.

### Re-run of the 11 at 3,600 s (2026-11-21, ended 09:24Z): 11 of 11 rc 0
Each pass count equals the baseline's: `test_hn_ctx_anchor` 72, `test_hnscreen_chain` 31, `test_hnscreen_ctx_chain` 22, `test_p2rep_chain` 353, `test_p5c_chain` 126,
`test_p6_chain` 431, `test_p6_levers --fast` 169, `test_p7_chain` 110, `test_rl_r0c` 26, `test_rl_safeobj` 61. `test_rls1_queue` passes 254 here; in the baseline it failed one
timing assertion (`kill_marker_after_th...`), so it is a timing-dependent test, not a change made by C1r.

## Verdict
**0 regressions.** Every file that passed in the baseline passes on the C1r tree (11 of them only after the timeout was lifted from 900 s to 3,600 s on a loaded box),
and 6 baseline failures pass (3 are the profile-set pins this change updates; 3 are environment or timing dependent and outside the change).
Still failing in both, unchanged by the change: the files with the same non-zero rc in both result files (e.g. `refe/test_launch_gate.py` as a script reads a D:-drive
data path that this box does not have; `stack/tests/test_launch_gate.py` passes 150 under pytest).
The 11 files failing in BOTH sweeps with the same rc: `test_carve_feed`, `refe/test_launch_gate` (script), `test_s1z_preflight`, `test_s1zsm_preflight`, `test_vdac_stage0_forward`, `test_r9_e1_m3n20`, `test_r9_e1_t1v_gate`, `test_ra1_hjepa`, `test_secret_scan`, `test_v8f_lora`, `eval/field/test_field_read`.
