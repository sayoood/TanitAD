# changed_from_table.py: scorer-only arms read on E3 by table lookup (control, 2026-11-10)

**What.** A scorer-only arm keeps P6's 64 proposals bit for bit; `read_scorer1p6` gate G0-E3 checks this. The harness score of its pick is therefore the truth table's column for that proposal. `changed_from_table.py` writes the reader's `e3_changed_scores.csv` from the table (`raw/2026-11-09-p6-navtest-read/selection/table_p6final_c.npz`, sha256 3bda2080..., 12,146 tokens x 64 proposals). It covers every token where the arm's pick under the navsim_v1 rule (torch float32, `read_scorer1p6.pick_logits`) differs from P6's.

**Control 1, row by row (MEASURED).** Every harness-scored row of Batch 1 is reproduced exactly, in the valid flag, all six sub-scores and the score, compared by float equality:

| arm | rows matched |
|---|---|
| C0 | 5,011 of 5,011 |
| C0r | 4,867 of 4,867 |
| S1E | 4,616 of 4,616 |

The lookup emits more rows (5,998, 5,852 and 5,517). The extra ones are changed picks on E2 tokens, which the reader already takes from the E2 table.

**Control 2, end to end (MEASURED).** `read_scorer1p6.py read` was run with the lookup csvs in place of the harness csvs, with all other inputs equal. Its `scorer1p6_read.json` equals the Batch 1 verdict (`raw/2026-11-09-scorer1p6-prereg/result_e3/verdict/scorer1p6_read.json`) in every field, 0 differences. Only `seconds`, `code_sha256` and `inputs` were excluded from the comparison. Outcome O5 in both.

**Use.** For S1Z, the replicates, ensembles and selector reads on E3, no harness run is needed for the changed picks.

```
python changed_from_table.py --table table_p6final_c.npz --arm-logits <arm>/e3_logits.npz --out <arm>/e3_changed_scores.csv
```

The reader's G0-E3 gate still has to pass; it proves the arm's proposals equal P6's. An arm that changes the proposals, for example a trunk retrain or P7, cannot use this route.
