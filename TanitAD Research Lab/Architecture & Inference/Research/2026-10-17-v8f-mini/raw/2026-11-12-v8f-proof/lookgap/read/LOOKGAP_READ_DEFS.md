# LOOKGAP read kit: what is coded (card PREREG_V8F_PROOF_ADD108, tier T0)

## Control clauses that could fail by construction

1. m2, lists: a record must carry the pilot lists' sha256 and also equal `RUNROOT.json` `train_list_sha256` / `eval_list_sha256`; `RUNROOT.json` `pilot_train_list_sha256` / `pilot_eval_list_sha256` must equal the lists now. If `RUNROOT.json` keeps its own copy of a list with other bytes than the pilot's, or lacks the `pilot_*` keys, m2 fails with no fault of the fits.
2. m2, token folder: `os.path.realpath(<run root>/tok/B)` must equal the real path of `tok/P`. It holds only if `tok/B` is a link or junction to arm P on the pod.
3. m3 depends on the subject in one case (card section 5): a gap head of arm P no better than the permuted fit.
4. m4: the label fields of every dump (15 files) must agree with D0 on the base rows within 1e-5 (relative and absolute). A dump written with another float type for `lab_gap` beyond that tolerance fails m4 with the count in the controls file.
5. m4 and m5 take 3,841 rows for both the row set and the base rows (m5 states "3841" for the base rows). If the base rows differ from the row set, m5 fails on the count even when S4 and S8_out agree.
6. m1 requires at least one clause in `ENC_M1.json`; a file with `pass_m1` true and no clauses fails.
7. m6 includes the lead's output of this kit (`kit_read/lead_pytest.out`), which exists on the pod only after the push.

## Rows and E

Each dump is laid on the join rows with `read_lookat.on_rows`. Base rows: `lab_present` of D0 true, `lab_gap` of D0 finite, finite `gap` in D0 to D4. Row set: base rows that also have a finite gap in P0, P1, P2, B0, B1. Both counts are printed; if the row set is not 3,841 the base rows lacking each of those five fits are printed. `n_label_fields_differ` counts base rows where another dump (covering the row) has another `lab_present` or `lab_gap` than D0; the count over all rows is stored beside it.

E(x) is the mean absolute difference of the gap of x and the label gap (D0) on the row set. R = mean E(P0..P2) / mean E(D0..D2).

## Interval

Per clip sums of absolute error on the row set; numerator = mean of the three P sums, denominator = mean of the D0, D1, D2 sums. `idx = RandomState(108).randint(0, n_clips, size=(2000, n_clips))`; per draw the ratio of summed numerator to summed denominator; two-sided 95 % percentile (2.5, 97.5). With fewer than half of the draws finite (1,000 of 2,000) there is no interval and no name.

## Names

* `LOOKGAP-T0-CUTS`: upper end below 1 and each of E(P0), E(P1), E(P2) below the smallest of E(D0..D4).
* `LOOKGAP-T0-CUTS-NOT-EACH-FIT`: upper end below 1 and at least one P fit not below that smallest error (equal counts as not below).
* `LOOKGAP-T0-WORSE`: lower end above 1.
* `LOOKGAP-T0-NOT-SEPARATED`: every other case with an interval.

The second derivation works from the per-clip sums only, with plain loops and hand-made percentiles; the two must agree on the name and on both ends (1e-9). The names stage stops with rc 4 when they disagree and ends with rc 5 and no name when there is no interval.

## Controls (printed only as `m<i> PASS|FAIL`; every value goes to `LOOKGAP_CONTROLS.json`)

m0 directions (`CHECK_DIRS` all_pass; a file for each of the 500 train and 141 evaluation clips with the frame count of its arm P token file read from the `.npy` header; keys exactly `has_lead`, `ctr_az`, `ctr_el`; `has_lead` true everywhere). m1 `ENC_M1.json`. m2 four fit records and `RUNROOT.json`. m3 ratio E(PERM) / E(P0), lower end above 1, same bootstrap. m4 E(D0) 3.960, E(D1) 3.913, E(D2) 4.112, E(B0) 2.137, E(B1) 1.910 within 1e-3, 3,841 rows, no differing label field. m5 S4 0.8748 and S8_out 0.5844 within 1e-4 from the direction files against the label box files on the base rows, using `read_lookat._stats`. m6 the last non-empty line of each of the four `lead_pytest.out` files holds "passed" and neither "failed" nor "error".

With all seven passing the values are under `payload`; otherwise under `withheld`, and the names stage refuses.

## Printed sections of the names stage

Tier and families (not read, n = 0 each, with the reason); row counts; E of every fit; R, interval and name; share of the bound's gain; correlation with the label gap; by label gap bin (P0, D0, B0: n, E, signed error gap minus label, share more than 5 m short); by absolute label azimuth bin (n, E); the cross read X0, X1 (n and E on the row set rows where finite, beside E of B0, B1 on the same rows and on the row set) with `pass` and `max_abs_diff_m` of `XDUMP_IDENT.json` stated as a fact; m3's ratio and interval; S4, S8, S8_out of the directions on the base rows. On the pod the stage A score lines are appended to the names log under their heading.

## Files

`read_lookgap.py` (reader, `--selftest`), `podread_lookgap.sh`, `pull_lookgap.sh`, `push_lookgap_read.sh`, `test_lookgap_read.py`. The reader imports `read_lookat.py` (sha256 recorded in the outputs, `--lookat-reader`) unchanged.
