# LOOKGAP fit kit (card PREREG_V8F_PROOF_ADD108, stage C): what is coded

## Clauses and steps that could fail by construction (read first)

1. Control m2, "each of the three fit records names ... head H5 on arm P": the fit tool of card ADD99 is handed arm P under the arm name `B` (the run root has `tok/B` as a link to the arm P folder),
   so every FIT_RECORD.json names arm `B`, never `P`. A reader that tests the arm name for `P` fails by construction. The record that ties `B` to arm P is `run/RUNROOT.json`
   (`tok_b_realpath`), which this queue writes; m2 has to read the arm name through it.
2. Control m2, "the settings of card ADD99": the tool writes its own card name and hyper-parameters into the record, so this clause reads what the tool wrote; the queue cannot change it.
3. Control m3 (permuted fit worse than the fit of seed 0): the card states its limit itself. Nothing in this kit changes it. Note the permuted fit's dump is made with the present dump of seed 0.
4. Fit seed 2 has no present dump of its own: `present_tpl/reader_s2_score.npz` is a link to the dump of seed 0 (as card ADD104 did). The dump of seed 2 therefore has the rows of seed 0's present dump.
5. Stop rule of the queue: the sum of the `.SECS` of the finished jobs is checked before each start. The three fits run side by side, so their seconds are added although the wall time is
   shorter; the sum can pass `LG_STOP_S` (5400) before the wall time does. Jobs that are running count only after they finish.
6. The identity cross dump (the pilot head of seed 0 on the pilot's tokens against the pilot's own dump) can read `pass false`: it is logged and the queue goes on, by the card's own order.
   Its fate then belongs to the reader, not to the queue.
7. A failing slot rule is a wait, not a stop; the wait ends with a stop after 21,600 s.

## pod_lookgap_fit_queue.sh

Gate: exit 4 (REFUSED) unless `DONE_ENC.txt` exists, `STOP_ENC.txt` does not, and `ENC_M1.json` has `"pass_m1": true` (read with python json). `STOP_FIT.txt` without `RESUME=1`, or `DONE_FIT.txt`: exit 3.
Then `ulimit -n 16384`, a lock (`fit_queue.lock`), and the run root `lookgap/run`, made once and never overwritten: the pilot's two lists copied with a sha256 compare, `tok/B` linked to `lookgap/tok/P`,
three present links, and `RUNROOT.json` with the keys of the brief. Jobs: four fits (at most three at a time; slot rule = fewer than three GPU compute processes and more than 28 GB free by cgroup),
then four dumps and three cross dumps one at a time. A fit whose record exists, and a dump or cross dump whose meta file exists, is counted done and not run again; an output file without its meta file is a stop.
`DRY=1` prints every command and creates nothing. Nothing is deleted, moved or ended.

## cross_dump.py

Imports `crop_fit` and `crop_dump` unchanged from `--kit-fit` (first on `sys.path`, with a check that both modules come from that folder). Reads `head.pt` (refuses unless the head is H5), the token files of
`--tok-dir` through the tool's own `TokenStore` (its shape and type check stays in force), `head_outputs` per clip and `dump_from_present`; writes the dump through a `.part` file, then the meta file.
`--compare`/`--ident-out` write the identity record (`max_abs_diff_m` over rows with both gaps finite, `nan_pattern_equal`, `valid_equal`, `n_rows`, `tol_m`, `pass`) and print `ZZ CROSS-IDENT ...`; exit 0.
Refusals (existing output, existing meta, existing ident file, missing input, head not H5) exit 2. `--selftest` runs a tiny CPU world and plants a changed token file.

## Tests and push

`test_lookgap_fit.py` covers the dry queue, the compare logic, the selftest, the refusals, the two pilot tool hashes and the wording scan. `push_lookgap_fit.sh` copies three files to `kit_fit` on the pod (noclobber, sha256 compare); the lead runs it.
