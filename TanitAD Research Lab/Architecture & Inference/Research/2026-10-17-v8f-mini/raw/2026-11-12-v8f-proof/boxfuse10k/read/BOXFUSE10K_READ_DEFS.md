# BOXFUSE10K reader: definitions (card PREREG_V8F_PROOF_ADD131)

Kit: `make_boxfuse_reader.py` writes `read_boxfuse10k.py` and `podread_boxfuse10k.sh` from `read_gapmargin10k.py` (sha256 prefix 0aab17c67eb60756) and `podread_gapmargin10k.sh`
(b44d8181d3894ae9) of card ADD117 by asserted substitutions (CR removed before hashing). Test: `test_boxfuse_read.py`. Pull: `pull_boxfuse10k.sh`.

Tier of every number: T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py`, heading bound 0.2 1/m); not driving performance.

## Runs (names without the prefix I10)

| setting | new | banked |
|---|---|---|
| K0 | `JEc` = RQ50KF5Hb20gJEc, `JA90` = RQ50KF5Hb20gJA90 | `GEc` = RQ50KF5Hb20gGEc, `A90` = RQ50KF5Hb20gA90, `RLc` = RLcKF5Hb20, `RR` = RQ50KF5Hb20 |
| CONV | `JEc` = RQ50Hb20gJEc | `GEc` = RQ50Hb20gGEc, `RLc` = RLcHb20, `RR` = RQ50Hb20 |

Gap file of a new run: `JEc` reads VGAPJE, `JA90` reads VGAPJE90.

## Pairs (first minus second; two-level bootstrap over model seeds and evaluation episodes, 2,000 draws)

Named: `K0-JEc` = JEc - GEc at K0, `CONV-JEc` = JEc - GEc at CONV, `K0-JA90` = JA90 - A90 at K0. Printed with no name: each new arm minus RLc of its setting; GEc minus RLc.

## Names (card section 4)

Per pair P and measure M in (UNSAFE, CONTACT, FALLING-BEHIND): `BOXFUSE10K-<P>-LOWERS-<M>` (interval below 0), `-RAISES-<M>` (above 0), `-<M>-NOT-SEPARATED` (else; an interval that reaches 0 is not separated).
Per pair: `-JOINED-GAP-HELPS` (lowers one or more, raises none), `-COSTS` (raises one or more, lowers none), `-TRADES` (lowers one and raises another), `-NOT-SEPARATED`. 39 names; the test reaches every one.
The T0 names of the card (`BOXFUSE10K-T0-...`) are not made by this reader.

## Controls (printed as `ctl j<k> PASS|FAIL <reason>`; no level, no difference, no name)

j0 to j4 are read from `BOXFUSE10K_GAPFILES_RECORD.json` (`--gap-record`): key `pass_j<k>` is JSON true. j0 also needs `card` = PREREG_V8F_PROOF_ADD131. j1 also needs `files.VGAPJE.sha256` and
`files.VGAPJE90.sha256` to equal the sha256 of the files given by `--gap-file VGAPJE=<path>` and `--gap-file VGAPJE90=<path>`.
j5 wrapper hash (28bc31f21ecb4ccb, pod check record, run texts) and the first line of each new run against the banked GEc run of its setting (JA90 also against GEc). j6 identity of the banked
levels within 1e-4 (failure text holds no value). j7 tests: outputs, and their times against the queue start and the controls start. j8 engine controls of each grid read and of the strategic grid.
j9 counts and rule logs: done marker, 141 dump files per new run, rule log names the sha256 of its own gap file and the closing switch, gap the rule received equals the file's gap within 1e-3 m
(at steps with no output it equals the reader's own logged gap of the RR run at the same model seed).

The names stage refuses unless the controls file has all ten controls passing and every input file hashes as the controls stage hashed it (`inputs.files`). Files that exist are not overwritten:
`BOXFUSE10K_CONTROLS.json`, `BOXFUSE10K_RESULT.json`, `BOXFUSE10K_VERDICT.md`, the names log.

This card has no map of arms to controls.
