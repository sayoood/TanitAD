# PREREG_P5C Amendment A3.2 (DRAFT, not frozen): TB-0's R-A is judged on rank-0 rows. MM agent, 2026-11-16

Evidence class: MEASURED (artifacts in this hold). Status: DRAFT. Nothing is committed or landed.

## The ruling (PI, 2026-11-16, verbatim)
*"Follow your recommendatiobs"*

This selects option 1 of `TB0_RA_FINDINGS.md` (repo `147ba68a`): **R-A is judged on rank-0 rows only. Rank-1 rows' R-A is computed and DISCLOSED (max, count over tol, n), never gated.**
Reason: under A3 section 3.3's forced `rank0` policy, C1 uses only rank-0 Tc20s/P5-R labels; rank-1 rows stay raw.

## Disclosure: this ruling is POST-DATA
The ruling was made after run 1 and run 2 measured R-A = 1.4901395672950457e-09 against tol 1e-9 (2 of 2,400 rows, both rank 1; rank-0 max 9.36e-10). It is not a pre-registered scope. It is listed as such in every report that quotes TB-0 PASS under A3.2.

## The change (and only this)
1. `p5c_table_gates.py`: new CLI flag `--tc20s-ranks {all,0}`, default `all`.
   - `all`: every gate result is byte-identical to the A1 module (test: bit-identical on 3 fixtures in p5r mode, 1 in Tc20s mode).
   - `0`: `gate_tb0` judges max abs diff, sim8 identity and missing rows on rank-0 common rows only. The detail carries `ranks_judged`, `rank0_max_abs_diff`, `rank0_n`, `disclosed_rank1_max_abs_diff`, `disclosed_rank1_n_over_tol`, `disclosed_rank1_n`, and rank-1 missing / sha / sim8 counts.
   - `gates.json` carries top-level `tc20s_ranks`; `table_meta_fields.json` carries `tc20s_ranks_judged_R_A`.
   - Unchanged: R-B (P5-R EVAL per-rank criteria), TB-1 (its Tc20s check stays on all ranks; it is rankless in the control file), TB-2..TB-7, the `ok` logic of R-B.
2. `p5c_pod_label.sh`: `--tc20s-ranks 0` on the CONTROL-pass gates call only; `PIN_GATES` updated. Nothing else.

## Pins (git hash-object --no-filters; 0 CR bytes in each)
| file | blob | sha256 | was |
|---|---|---|---|
| `a1/p5c_table_gates.py` | `1711a72f1b56f7df1d1e1f1194bbedcb7521bcb9` | `0c567514a0bb97708cf3d41bbaff5727c59c7fe380353cc9ccefa43654d00d35` | `5c2f49cf` / `d600a830...a633` |
| `a1/p5c_pod_label.sh` | `8cfe7a72a296988a54bd3e13590c4787388b67da` | `2fb22f24ee72b42649f41692d47884ecbf3e123e55ac82f3ce70ff905c7d502c` | `727e72d4` |
| `a1/test_p5c_table_gates.py` | `1695fef2b4b61f755756f976625283311bdd7353` | `9bda9e9fcd8f12be4f166bec0ade41e04621e9192cab11062058e40d37bdd41f` | `82ce4a58` |
| `a1/test_p5c_pod_label_a31.py` | `6830e33c7b316199f3639da019424e80eb69d0a6` | `46a67a4af2df9b40bf9527954f513b84198d24a2cacc0ccc3fca95f98e203246` | `78dcf8ee` |

`PIN_GATES` in the runner equals the new gates sha256 (asserted by `test_a32_runner_text_pins`).

## Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`)
- Gate suite (`test_p5c_table_gates.py`, staged beside the p5c/a1 fixtures and the a1 runner it parses): **199/199 PASS** (baseline A1: 163/163, all 163 unchanged and green; +36 A3.2 checks). Log: `proof/TEST_RUN_gates_suite_A32.log`.
  - (a) default `all` fails a rank-1-only excess; (b) `0` passes it and discloses the exact max; (c) `0` still fails a rank-0 excess; (d) `0` fails a missing rank-0 row (a missing rank-1 row is disclosed); sim8 both ways; TB-1 not rescoped; R-B detail identical under both scopes.
  - 10 mutants all RED: flag ignored; ranks inverted; max disclosure dropped; over-tol count dropped; disclosure keys dropped; `gate_tb0` default 0; `run_all_p5r` default 0; CLI default 0; TB-1 also rescoped; scope not passed through.
- Runner suite (`test_p5c_pod_label_a31.py`): **49/49 PASS** (42 old, all green, no old test body edited; the Box helper gained an optional `dev_shift`; +7 A3.2). The old suite needed no pin edit: it derives the gates pin from the gates file it is given.
  - New: rank-1-only dev-box shift 1.5e-9 completes with TB-0@control PASS and the disclosed max; rank-0 shift 1.5e-9 stops at `control_gates`; the control gates.json records scope `0`, the main one `all`; runner text carries the flag exactly once, on the control call.
  - Mutants RED: runner without the flag; flag moved to the main call; flag also on the main call.

## Real-data proof (pod, read-only; run 1's control outputs; new dir `/workspace/data/p5c_a32_check/`)
Inputs: `/workspace/data/p5c_label/control/{bank,out}` (15 shards) and `/workspace/data/p5c_ship/*`, the arguments of the runner's control call. The new gates file was shipped to the new dir (sha256 verified `0c567514...`); nothing was written into `p5c_label*` or `p5c_ship`.
| scope | TB-0 | R-A value | detail |
|---|---|---|---|
| `0` | **PASS** | 9.357083996519577e-10 (n 1,200) | rank-0 max 9.357083996519577e-10; disclosed rank-1 max **1.4901395672950457e-09**, 2 of 1,200 over tol; TB-1 PASS; ZZTB_ALL_PASS |
| `all` | FAIL | **1.4901395672950457e-09** (n 2,400) | **TB-0 and every other gate result byte-identical to run 1's `gates.json`** (checked with `json.dumps(sort_keys)`); failing `['TB-0']` |

Under `0`, the R-B part and all gates other than TB-0 are byte-identical to run 1. Banked copies: `proof/r0`, `proof/rall` (gates.json, stdout, meta; sha256 verified after pull), `proof/podrun.sh`, `proof/podchk.py`.

## Not changed
A3, A3.1, R-B, TB-1, any other gate or rule; C0; the main-gates call (scope `all`).
