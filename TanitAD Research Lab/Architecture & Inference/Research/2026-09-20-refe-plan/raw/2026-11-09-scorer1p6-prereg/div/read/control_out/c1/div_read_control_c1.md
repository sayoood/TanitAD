# DIV read (CONTROL c1)

**Status: READ**

**CONTROL RUN: not a DIV result. It reads only an arm already read; the ledger is never written.**

- Card: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_DIV_P6.md` sha256 `e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac` (frozen `e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac`)
- Tier: T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop)
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs').
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check). Training variance is not resampled: ensembles of fixed members are read as fixed scorers. Inference is deterministic.
- Members (8): C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM

## Gates

- G0-PICK: **PASS** n_equal_to_banked_pick=12146, n=12146
- G0-P6LOGITS: **PASS** P6 dir logits bit-identical to the table's logits=True
- G0-TABLE: **PASS** n_tokens_absent_in_p6_csv=0, n_pdms_equal_to_p6_csv=12146
- G0-E2: **PASS** n_e2=2000, n_missing_in_table=0, n_missing_in_world=0, n_pdms_bit_equal=2000, n_sub_bit_equal=2000, max_abs_pdms_diff=0.0, formula_max_abs_diff=0.0, all_finite_valid=True, pick_is_argmax_of_p6_logits=2000, RED_shifted_pick_n_differ=1402
- G0-E2-vs-TABLE: **PASS** E2 table pdms bit-identical to the full table on the E2 tokens=True
- G0-E3[C0]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1E]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r2]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1E1]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1Z1]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1Z]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1ZSM]: **PASS** n=12146, n_tokens_not_bit_identical=0

## Control result

```
{
 "control": "c1",
 "arm": "PESS-8 rebuilt from the eight members",
 "banked_path": "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/pess/read/real/pess_read.json",
 "banked_json_sha256": "4feb7f0a9c48348378fde92f39257a2887bc873fc339e92b6a958db037e15a72",
 "banked_sha256": "8912f1d04ab3e22571397d53580ddafebc3beb78e1e27d34fe21db25e5bf7ae5",
 "rebuilt_sha256": "8912f1d04ab3e22571397d53580ddafebc3beb78e1e27d34fe21db25e5bf7ae5",
 "sha256_equal": true,
 "banked_PDMS_x100": 91.47901718634304,
 "literal_PDMS_x100": 91.47901718634304,
 "rebuilt_PDMS_x100": 91.47901718634304,
 "PDMS_equal_exactly_banked": true,
 "PDMS_equal_exactly_literal": true,
 "members": [
  "C0",
  "C0r",
  "S1E",
  "C0r2",
  "S1E1",
  "S1Z1",
  "S1Z",
  "S1ZSM"
 ],
 "arms_scored": [
  "PESS-8"
 ],
 "PASS": true
}
```
