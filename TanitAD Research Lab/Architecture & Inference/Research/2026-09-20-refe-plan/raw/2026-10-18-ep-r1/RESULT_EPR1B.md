# EP-R1b RESULT: the dev applier, the navtest bank scoping, and the shared navtest-confirm applier

Date 2026-10-19. Agent: Architecture & Inference (EP-R1b), for the Master Mind. Brief: `C:/Users/Admin/par18/epr1b.md`. Landed base: `54f0258` (the landed `RESULT.md` of EP-R1 is frozen; this is a NEW file).
All code is new, CPU only, staged not committed. Evidence classes: MEASURED (artifact path), INHERITED, ESTIMATED, HYPOTHESIS.

## 1. Deliverables, in the brief's priority order

| # | deliverable | files (all under `code/` unless noted) | tests | status |
|---|---|---|---|---|
| 1 | **EP-R1 dev gate** (PREREG_EPR1 s7, s8; MM_DECISIONS_EPR1 D1-D3) | `epr1_dev.py` (847 lines) | `test_epr1_dev.py`: **88 OK** | built; input phase run on the REAL dev inputs |
| 2 | **navtest feature-bank scoping** | `../NAVTEST_BANK_SCOPING.md`, `make_navtest_bank_plan.py` | `test_make_navtest_bank_plan.py`: **18 OK** | no bank exists; the forward is named, planned and costed |
| 3 | **shared navtest-confirm applier** | `refe_head_confirm.py` (760 lines) | `test_refe_head_confirm.py`: **108 OK** | built; input phase run on the REAL navtest table |

MEASURED: 214 tests, 0 failures, run from a directory that emulates landing (the landed `epr1.py` a751c63e... and `zero_tail.py` 149c3e97... beside the new files), `-W error::ResourceWarning`, CPU,
`raw/test_epr1_dev.log`, `raw/test_refe_head_confirm.log`, `raw/test_make_navtest_bank_plan.log`. Code hashes: `code/CODE_SHA256_EPR1B.json`.
Mutation arms: **14** marked lines in `epr1_dev.py`, **22** in `refe_head_confirm.py`, **4** in `make_navtest_bank_plan.py`; every marker is edited at exactly one place by a test and the check must go RED (a coverage test fails if a marker has no mutant test).

## 2. (1) The dev gate: what it does and what was and was not shown

* Reads P2's held-out table (**1,705 tokens / 24 logs, one row per token**; the dev set is per-token, so the brief's "same folds" do not apply here: the rows were fitted on the S0 bank, the dev table is only read), replaces ONLY column 2 (EP), applies the planner rule, and reads PDMS by lookup (the fan is frozen).
* **Bar** (frozen text): gain >= +0.30 PDMS x100 on BOTH seeds AND C0, C1, C2, C4. Verdict PASS / FAIL / VOID. 24 logs allow no lower-bound rule: the interval is reported, never a bar. Estimator: paired log-cluster bootstrap, B = 10,000, seed 20260927, identical to `zero_tail.Boot` (a test compares the numbers).
* **Protocol order (each refusal fires before any outcome is opened):** registered c* sha256, schema/stage/seeds/pins/code pin, E1 (c* at the grid edge refused without an MM waiver), C2 files, dev-input sha256 pins, then the no-outcome controls (integrity, C0, C1, C4) GATE the read; only then `READ_STARTED` is written exclusively (a crashed first attempt still consumed the read) and `pdms` / `sub` / `valid` are opened.
* **JOINT** (EP-R1 row + SR-TTC's c* TTC row of the same seed) is emitted, labelled SECONDARY, never moves the verdict.
* **Known-value identity (required by the brief):** with the EP row equal to P2's, the dev pick equals P2's pick on every token and the gain is exactly 0.0 (`Identity` tests, plus a hand-built world with a hand-computed gain).
* **MEASURED on the REAL dev inputs, outcomes not opened** (`raw/epr1_dev_inputs_check.json`): the three dev files hash to their landed sha256; P2's head reproduces the stored logits from the cached features to **1.45e-3** (tolerance 2e-3; the next-token negative control reads 15.2); **C0: logits bit-identical, picks equal P2's banked pick on 1,705 / 1,705**; **C4: dump proposals equal the table's, max abs 0.0 m, 1,705 bit-identical** (negative control 0.82 m reads FAIL).
* **NOT shown:** no row has ever been applied (none exists until S0 finishes and EP-R1 is fitted); the dev READ has not been spent; C2 and the outcome-bearing controls were exercised on synthetic worlds only.

## 3. (2) Scoping: the bank does not exist (full note: `NAVTEST_BANK_SCOPING.md`)

MEASURED, two names x six paths (`find` + header listings): **no navtest `score_head` input-feature bank exists** (only the 4-token DG-DUMP, `srttc_s0/work/dgdump/dgdump_feat.npz`; the 14 landed full-table dumps have no feature array).
**The forward:** `srttc_s0_dump.py` fan path, no `--lean`, no `--sets-dir`, `--feat-out`; checkpoint `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` (sha256 `ea31db22...`); **GPU**; **5.43 dev-box GPU-h** (MEASURED rate of the landed forward, 19,531 s over 12,146 tokens; INHERITED for this job); **~0.7 GB** disk (398 MB features + ~0.3 GB fan dump; D: has 120.8 GB free).
`make_navtest_bank_plan.py` turns the landed 14 chunks into `--plan` files and the exact commands (CPU, tested; on the real files: 14 chunks partition the 12,146 tokens once, every chunk sha256 and token_log verified, all 136 nuPlan DBs exist, `raw/refe_head_confirm_inputs_check.json`).

## 4. (3) The shared applier

* **Two stages, one ledger.** `picks` opens no outcome array (a test runs it on a table that has none and a spy proves no outcome member is touched), writes W3-format seams (keys, dtypes and shapes identical to the real banked P2 seam, header-compared; only the `arm` string differs), the picks and the controls, and is repeatable. `read` exclusively creates the look ledger BEFORE opening `pdms` / `sub`, then does the lookups, the registered bar clauses, the PREREG s8 triggers and the multiplicity disclosure; a second `read` is refused; a read that crashes after the ledger still consumed the look; a RED control leaves it unspent.
* **Refuses to start** unless BOTH dev-gate result files exist or a formal-abandonment file (arm, `abandoned: true`, `register_row`, `reason`, `decided_by`) exists for ONE arm; refuses for both abandoned; refuses an arm with both a result and an abandonment. Refusals precede the table and the bank: with no dev result the table and bank paths need not exist. EP-R1 is read iff EP-R1 alone passed; SR-TTC iff its own gate passed; the JOINT is EP-R1's registered SECONDARY (EP-R1 passed AND SR-TTC dev-read AND the two rows fitted on identical logs, else dropped with the reason in the report). No arm passed: look unspent.
* **Binding:** registered c* result sha256 == the file == the one each dev result recorded; every row file == the row sha256 the dev gate applied; the recorded P2-row sha256 == THIS checkpoint's; table sha256 pinned; `epr1.py` pinned; bank: manifest sha256 per chunk, every token once, log names equal to the table's.
* **Controls (all must be GREEN before any seam or the ledger is written):** integrity (P2's head reproduces the stored logits from the features, <= 2e-3, negative control), C0 (identity: P2's own rows give P2's logits bit for bit and P2's pick), C1 (reach: every reading changes its named columns and ONLY those, plus a positive control with the negated P2 row), C4 (dump proposals == the table's, max abs 0.0, pick equal, logits <= 1e-4), the banked-P2-seam control (poses and fingerprints equal).
* **Identity arm (required):** rows equal to P2's give P2's pick on all tokens for P2, EP-R1, SR-TTC and the JOINT; and C1 reads RED for such a no-op row (a pipeline that cannot tell "applied" from "not applied" is the defect class named in CLAUDE.md).
* **MEASURED on the REAL navtest table, outcomes not opened** (`raw/refe_head_confirm_inputs_check.json`): sha256 `33441082...` loads at the pinned 12,146 / 136 through this module's guard; the rule on the stored logits reproduces the stored pick on **12,146 / 12,146**; the banked P2 seam equals the table's executed pose of P2's pick with **max abs 0.0 m**, fingerprints equal.
* **Not shown:** no real row, no real bank, no harness cross-check (`score_navtest_refe.py` was not run: it reads the navtest outcomes; `ready_for_score.json` carries a WARNING to run it only after `read`).

## 5. Defects found while building (root-cause class, not just the fix)

1. **A guard written to keep one artifact out refused the module's own outputs.** The fit-bank token `srttc_s0` matched as a substring, and the module names its readings `SRTTC_s0`: **30 of 106 tests errored on the first run** (`seam_SRTTC_s0.npz` refused). Fixed by matching the S0 run dir as a whole path segment. Class: *a refusal list tested only against the things it must refuse*; the test that found it wrote real-named files.
2. **Reusing the dev guard would have made the module unreachable on the real table:** `epr1_dev.FORBIDDEN_TOKENS` contains `navtest` and `table_p2_full`, which are the real paths. The confirm applier has its own guard; a mutant that restores the dev tokens makes the real names raise (test). Class: built, tested, unreachable from its caller.
3. **A whole-table float64 pick needs ~1 GB of temporaries** on a box whose RAM is scarce; the pick is now chunked (found while sizing the real check).
4. **A synthetic world's TTC component was adverse by chance** (wholly below 0), which correctly blocked the primary; the test world got a `clean_subs` variant for the both-seeds tests and the default world was kept as an end-to-end test of the adverse-sub-score clause.

## 6. Proposed register rows (NOT written; the Master Mind lands them)

| id | row |
|---|---|
| `D-REFE-NAVBANK-1` | No navtest P2 `score_head` input-feature bank exists (MEASURED, 2 names x 6 paths, 2026-10-19). The shared look needs a 5.43 GPU-h forward (`srttc_s0_dump.py` fan path, `--feat-out`, not `--lean`) and ~0.7 GB; path proven on 4 tokens (DG-DUMP). OPEN: schedule behind S0. |
| `C-EPR1B-DEV-APPLIER` | `epr1_dev.py` built, 88 tests / 14 mutation arms; input phase MEASURED on the real dev inputs (featcheck 1.45e-3, C0 1705/1705, C4 0.0 m). The dev read is UNSPENT. |
| `C-EPR1B-CONFIRM-APPLIER` | `refe_head_confirm.py` built, 108 tests / 22 mutation arms; input phase MEASURED on the real navtest table (rule 12,146/12,146, banked seam 0.0 m). The navtest look is UNSPENT. |
| `T-GUARD-SELF-COLLISION` | New trap/retraction class: a refusal token matching the module's own output names (section 5, item 1). |

## 7. Integration asks and decisions for the Master Mind (in order)

1. **Land the staged files** at `REFE/raw/2026-10-18-ep-r1/` (new files only; `paths.json` lists them).
2. **Schedule the navtest bank forward** (5.43 GPU-h behind S0, `gpu_lock.py acquire`): now, or only after a dev pass. It is outcome-free and spends no look; the registered "S3 only on a dev pass" is a cost rule, and one bank serves all three readings. Cheap pre-check first: the 4-token DG-DUMP on one of the 4 logs with the 36 known frame-control tokens (if the new dump differs from the table there, C4 is RED and the applier refuses; no waiver is registered).
3. **SR-TTC has no dev applier**; `selrank_dev.py` is a descriptive attribution, not a gate. `refe_head_confirm.read_dev_result` states the contract it needs from SR-TTC's dev result (`dev_gate.verdict` in PASS/FAIL/VOID, `inputs.rows_sha256` per seed, `inputs.r1_result_sha256`). Either SR-TTC's dev applier writes that (a `--col 3` variant of `epr1_dev.py` against SR-TTC's own bar is the small step), or tell me the format and I write the adapter.
4. **Confirm the JOINT rule:** I implemented "JOINT iff EP-R1 passed its dev gate AND SR-TTC is dev-read (not abandoned) AND identical fit logs". The alternative (JOINT also when only SR-TTC passes) is a one-line change at the `JOINTELIG` marker.
5. **C2 strictness:** dev C2 requires the shuffled-label rows' point gain <= 0 per seed with no tolerance (none is registered). A navtest C2 reading and `families6` on the new seams are NOT built (families6 pending; STRATEGIC is PAUSED, PI item 25; distance-keeping UNAVAILABLE) -- the report says so per family.
6. **Abandonment file format** is defined in `refe_head_confirm.read_abandon`; the register row it names is the Master Mind's.

## 8. Deliverable manifest

Repo path root `REFE` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`; everything staged under `C:/Users/Admin/par18/epr1b/out/` mirroring it (`out/paths.json` lists every row). Nothing is committed or pushed.
`REFE/raw/2026-10-18-ep-r1/`: `NAVTEST_BANK_SCOPING.md`, `RESULT_EPR1B.md`; `code/`: `epr1_dev.py`, `test_epr1_dev.py`, `epr1_dev_inputs_check.py`, `refe_head_confirm.py`, `test_refe_head_confirm.py`, `refe_head_confirm_inputs_check.py`, `make_navtest_bank_plan.py`, `test_make_navtest_bank_plan.py`, `CODE_SHA256_EPR1B.json`; `raw/`: `test_epr1_dev.log`, `test_refe_head_confirm.log`, `test_make_navtest_bank_plan.log`, `epr1_dev_inputs_check.json`, `refe_head_confirm_inputs_check.json`.
