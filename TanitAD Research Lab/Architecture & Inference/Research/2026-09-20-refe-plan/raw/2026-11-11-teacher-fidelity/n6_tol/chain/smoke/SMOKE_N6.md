# SMOKE_N6: navtrain only, through the real build CLI (no navtest N6 build or score)

Inputs (`mk_smoke_inputs.py`): the 7 real navtrain DAC-0 instances of the TF-N5 smoke (DEV30; tokens 634f2012, 7a2ceb0a, 85772c23, cee7cb35, f2c6dba4, a7b14cd0, d7587000). Stand-ins: the RJ5-style target `t20` rows are the DEV30 rolls; the "N2-R" and "N5-A" plans are the base label R of each t20 (so all 7 are residual); the "harness CSV" is the TF-N5 smoke's builder base rows (builder DAC 0, PDMS 0). Route `navtrain` = nuPlan-DB rebuild with the builder scorer; the human future comes from the same DB. Commands: `n6_nt_build.py reach --real`, `derive --route navtrain ... --resid-tokens`, `merge ... --expect-residual 7`, `unch`.

| run | derive wall (7 tokens) | verdict | RED controls | N6-H changed (teacher_tol / human) | N6-T | N6-X |
|---|---|---|---|---|---|---|
| clean | 72 s | **PASS** (14 of 14 controls) | none | 7 (7 / 0) | 7 | 0 |
| `--mutate teacher_off` (the human path on real data) | 15 s | FAIL by design | C-REPLAY, C-DET (the repairs were not run) | 7 (0 / **7**) | 0 | 0 |
| `--mutate human_first` | 62 s | FAIL | **C-SELECT, C-ORDER** | 7 (all human) | 7 (human, T mutated too) | 0 |
| `--mutate dac_inverted` (certificate inverted on every arm) | 88 s | FAIL | **C-SELECT, C-CERT-B6, C-X0** | 0 | 0 | 7 |

Sources: `n6_build_report_<run>.json`, `derive_<run>.jsonl`, `unch_clean.json`, `reach_real.json`.

## What the clean run shows (MEASURED, navtrain)
- `reach --real` (two failing navtrain fixtures `easy`, `multi_iter`): iterate 0 == the base label, the capture is bit-deterministic, N6-H picks a DAC-1 teacher candidate (`easy`: REPAIR-D k=1; `multi_iter`: L1 k=3), N6-X picks none, the human candidate builds and reads DAC = 1.
- Per token the teacher candidates number 4 to 10 after de-duplication (3 of them DAC = 1 in each case); the card selector takes REPAIR-D k=1 on 4 tokens, k=2 on 1, k=4 on 1, L1 k=2 on 1. These navtrain instances are easy (both repairs return a plan, the full certificate is met), so **the tolerance is not exercised on real data here**; its mechanism is exercised through the REAL repair loops on fake scorers in `test_n6_chain.py::TestCapture` (a DAC-1 candidate with TTC and C worse is refused by both banked repairs and taken by the card selector, refused by the strict mutation).
- Human path on real data (`teacher_off`): the human candidate has DAC 1, NC 1, DDC 1, TTC 1, C 1 on 7 of 7, builder PDMS 0.982, 0.986, 0.995, 0.861, 0.924, 0.867, 0.931. Under the card N6-H takes it only where no teacher candidate passes; here that was forced by switching the teacher candidates off. N6-T takes none.
- Human plan format: `ND.to_navsim(plan20)` against the logged 0.5 s poses: max 1.14 cm / 1.4e-3 rad (`hum_interp.json`).
- Cost: derive 7.6 s per residual token mean (both repairs twice, human, three selectors, re-scores), navtrain route. On navtest the residual is 20 tokens; the all-token scoring of the N5-A plan is the bulk (TF-N5's 12,146-token derive took 0.2 CPU-h of scoring; ESTIMATED 0.7 to 1.0 h wall with context loading).
