# HANDOFF: P5-C read instrument B10 + B11 + B12 (built 2026-11-13, CPU only, no P5-C number)
State: built, tested (174 passed), dry-run on 200 tokens (rehearsal, `--rehearsal` kept on). Nothing staged in git, nothing pushed, no ref touched. Next: the MM lands `files/` and resolves Q-7..Q-11.
Source: mirror tip `origin/agent/arch-inf-20260803` (built on b852a319; the six files I modified are byte-identical at the current tip 6983de9a). Evidence class of every number: MEASURED (artifacts in `files/raw/2026-11-13-p5c-read/b10b12/dryrun/`), a REHEARSAL on stand-in arms (C1=P6, C0=u400, base=deployed); it says nothing about P5-C.

## Integration (the MM). Paths relative to PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`
NEW (7): `eval/p5c/p5c_determinism.py`, `p5c_seal.py`, `p5c_tactical.py`, `test_p5c_determinism.py`, `test_p5c_seal.py`, `test_p5c_tactical.py`, `test_p5c_b10b12_wiring.py`.
MODIFIED, whole-file replacement of the tip blobs (6; diff in `MODIFIED_EXISTING_FILES.diff`, 283 lines): `p5c_read.py`, `p5c_render.py`, `p5c_read_prep.py`, `p5c_families.py`, `test_p5c_read.py`, `test_p5c_families.py`. Warning: the lander rewrites CRLF sources in place; the files are LF, land copies.
Dry-run evidence (new files): `raw/2026-11-13-p5c-read/b10b12/dryrun/*` (23 files incl. `rows_pass1_pass2_pass3.tar.xz`). Max repo path 95 chars. `SHA256.txt` lists every file.

## What was built
| id | file | what |
|---|---|---|
| B10 | `p5c_determinism.py` | `select` (k = ceil(0.10 n), seed 20261112: 12,146 -> 1,215), `plans` (restrict plans npz to the subset = the second pass's input), `compare` (per arm x scorer x metric, tolerance 0.0 = bit-identical). Result JSON: `pass`, `control: "B10_determinism"`. Refuses a vacuous PASS: same row files, pass 2 a copy of pass 1 (every `t_s` equal), tokens outside the subset, error rows, an edited subset file. |
| B11 | `p5c_seal.py` | `seal` (sha256 COMPUTED from the files; self-hash; written once), `verify` (self-hash, both arms, 64-hex, C1 != C0 != base, card sha, `--recheck-files`, rows not older than the seal), `validate-ledger`, `entries` (card 11 template, n = next free), `check`, `merge` (never in place). Never writes the real ledger. |
| B12 | `p5c_tactical.py` | wraps `command_adherence.run_read` (accuracy, kappa, 3x3 confusion, paired deltas C1-C0, C1-base, C0-base, 64-proposal floor if `--dump C1=`) and `selection_quality.read_table` (if `--selq ARM=table.npz`). Goal-setting = NOT COMPUTABLE (reason, n); STRATEGIC = PAUSED; no pooling (`validate_block`). |
| wiring | `p5c_read.py` | `--tactical`, `--sealed` (+`--card`, `--recheck-files`), `--require-kinds`; control `kind` recorded; the read verifies SEALED itself (a tampered or late seal = INVALID); a real read refuses without `--sealed` and defaults to requiring `B10_determinism` + `B11_sealed` (rehearsal: none); an absent tactical block is written as NOT COMPUTED, never dropped. |
| wiring | `p5c_render.py` | prints the TACTICAL block (every number labelled "no tracker in this number") and a controls / seal section; refuses a read with no tactical block. |
| fixes | `p5c_read_prep.py`, `p5c_families.py`, `p5c_read.py` | identity JSON now carries `control: "B6_identity"`; B9 refuses a RAISED families frame; G2 over 0 comparable components = NOT-EVALUATED (Q-10). |

## Tests (MEASURED, `logs/TEST_RUN_FINAL.log`: fresh tip export + hold overlay, `pytest eval/p5c -q`)
**174 passed, 0 failed, 0 skipped** (20.3 s). Before: 98 passed. New: determinism 23, seal 20, tactical 15, wiring 16; +1 in `test_p5c_read.py`, +1 in `test_p5c_families.py`; one existing G2 test fixture was edited (its tracked frame was empty, which Q-10 now rejects).
RED arms (each must fail): perturbed rerun 1e-12 / 1e-9 / 1e-3 on either scorer; sub-score-only change; copy of pass 1; shared row files; missing / extra / error / NaN rows; edited subset file; tampered SEALED (any field; sha; edit-and-rehash caught by `--recheck-files`); same sha for both arms; C1 == base; wrong card sha; rows older than the seal; ledger duplicate / gap / reorder / backwards time; wrong n / wrong sha / missing arm entry / replayed entries / C0r without approval; in-place merge; pooled score; absent family; STRATEGIC not PAUSED; mismatched seam token sets (INVALID, no numbers); registered read on a subset (G_COUNTS INVALID).
Controls reading known values: identical rerun = 0 tokens over, max diff 0.0; the real ledger validates at K = 50; its next entries are n = 51, 52.
**Reach (proved from the caller, not just the library):** (1) a positive verdict is INCOMPLETE without the B10 result and SEPARATED-SINGLE with it; a failed B10 gives INVALID; an unnamed `{"pass": true}` does not stand in for B10; (2) a tampered seal makes the read INVALID; (3) the tactical block is in the read JSON and the report, and the renderer refuses a read without one; (4) MUTATION tests: the consuming line is deleted in a copy of `p5c_read.py` (tactical line, required-kinds line, sealed line) and the same assertions go RED on the mutant.
Tests need `TANITAD_NAVSIM_ROOT` and `TANITAD_REPO` for the 3 integration tests of `test_p5c_tactical.py` (they skip without them; with an older repo tree they still pass). Run with `TANITAD_REPO` = a tip export.

## Dry run (200 tokens, seed 20261113; 88 logs; stand-in arms; scoring on 1 process)
* B11: SEALED.json from the banked seam files as stand-ins for `model_final.pt` (C1 sha `df93aa24..`, C0 `efbaa194..`), sealed BEFORE any scoring; verify PASS with file recheck. Entries against the real ledger: `dryrun/ledger/REHEARSAL_entries_NOT_FOR_APPEND.json` (n 51, 52), check PASS, merged copy valid at K = 52, re-merging the same entries REFUSED. **Do not append the rehearsal entries.**
* B10: subset 20 of 200 (`subset.json`). Control, identical rerun through the real scorer: **PASS, 0 / 20 tokens over, max |diff| 0.0** (`b10_determinism.json`). RED, C1 plan shifted +1 mm in x through the same scorer: **FAIL, 9 / 20 tokens over, max 1.87e-4** (`b10_RED_perturbed.json`).
* Identity (B6): 200 / 200, 0 over, max 1.9e-12, all three arms.
* Read with `--require-kinds B10_determinism B11_sealed` and `--tactical`: `REHEARSAL:NULL` with B10 PASS; **`REHEARSAL:INVALID` with the RED B10 result** (the verdict moves on real data); B10 absent: the controls block lists `missing_kinds: [B10_determinism]` (verdict stays NULL; INCOMPLETE only applies to a positive verdict, shown by the unit test). G2: 28 components compared, `along_mae_m` flagged adverse for P6 vs u400 on this subset (a property of the stand-ins).
* B12 on 200 tokens: command adherence COMPUTED, all gates ok (incl. bit-equal to `proxy_eval.boot`, kappa equal to the programme's `_kappa_k`); turn-active accuracy C1 0.938 [0.875, 0.985] (n 64), C0 0.906, base 0.906; C1-C0 +0.031 [0.000, +0.079], not separated. Random-proposal floor NOT COMPUTED (the banked seams have no `--dump-proposals` dump; n 200). Selection quality NOT COMPUTED (needs 64 harness columns per arm, ~4 to 6.4 h CPU; n 200). Goal-setting NOT COMPUTABLE; STRATEGIC PAUSED.

## Card questions (listed, not decided; also in the read JSON `card_questions`)
* **Q-7** B10: the card says per-token PDMS bit-identical. Implemented stricter: EVERY sub-score identical too. The subset RNG is not given (n and seed are): `default_rng(seed).choice(n, k, replace=False)` over the sorted tokens, k = ceil(0.10 n).
* **Q-8** B11: the card gives the ledger entry shape, not the seal's schema. SEALED.json = both sha256s, card sha256, base sha256, UTC time, self-hash. A real read now needs it and checks that score rows are not older than the seal (file mtime; a copy that sets mtime into the past would trip it: re-score or touch).
* **Q-9** B12: "computable as in P6's read". Selection quality needs the 64-column scoring and is reported NOT COMPUTED with reason and n unless `--selq` tables are supplied; the adherence floor needs the C1 `--dump-proposals` dump. The tactical family has no verdict authority (command_adherence.AUTHORITY). Default required control kinds for a real read are B10 + B11 only: W-TRANSP (ii)-(iv) JSONs carry no kind name yet, so they cannot be required by name (they still count via `--controls`).
* **Q-10** (found in B7/B9): if `families_run` raises inside a frame, `fam_ci` writes EMPTY families and G2 read PASS over 0 components. Fixed conservatively: B9 refuses a RAISED frame; G2 over 0 components = NOT-EVALUATED. The card is silent. Root cause in my first dry run: `--repo` pointed at an older `taniteval` (the E: tree lags the tip) that rejects `per_window`. **The real read must run families with a repo tree taken from the tip (`p5c_families.py --repo <tip export with taniteval/ and stack/tanitad/>`).**
* **Q-11** ledger path: card 11 names `SG/p6base/selectors/ledger_K.json`, the mirror holds it under PK `raw/2026-11-09-scorer1p6-prereg/selectors/`. The tool takes `--ledger`; K at read time decides n (K was 50 at this tip). Card item 13 (is P7's own read entered in that ledger?) is still open and is the MM's.

## Exact commands for the MM (landed tree; `PK=<repo>/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`)
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim TANITAD_REPO=<tip export with taniteval/ and stack/tanitad/>
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; NPY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe; E=$PK/eval/p5c
$PY -m pytest $PK/eval/p5c -q                                   # expect 174 passed
# B11, BEFORE any pass
$PY $E/p5c_seal.py seal --arm C1=<C1 model_final.pt> --arm C0=<C0 model_final.pt> --card $PK/raw/2026-11-11-p5c-readiness/PREREG_P5C.md --base-sha <P5C_BASE_SHA256> --out $W/SEALED.json
$PY $E/p5c_seal.py verify --sealed $W/SEALED.json --card <card> --recheck-files --out $W/sealed_verify.json
# B5/B6 as in raw/2026-11-13-p5c-read/HANDOFF.md (convert with dumps, score, identity), then B10:
$PY $E/p5c_determinism.py select --plans-dir $W/plans --out $W/b10/subset.json
$PY $E/p5c_determinism.py plans  --plans-dir $W/plans --subset $W/b10/subset.json --out-dir $W/b10/plans2
$PY $E/p5c_read_prep.py score --plans-dir $W/b10/plans2 --out-dir $W/score_b10 --procs 2 --python $NPY
$PY $E/p5c_determinism.py compare --rows1 "$W/score/rows_s*.jsonl" --rows2 "$W/score_b10/rows_s*.jsonl" --subset $W/b10/subset.json --arms C1 C0 base --human human --out $W/b10_determinism.json
# B12 (no --subset-ok on the full navtest)
$PY $E/p5c_tactical.py --seam C1=<C1 seam> C0=<C0 seam> base=<P7 seam> --dump C1=<C1 dump> [--selq C1=table.npz ...] --out $W/tactical.json
# families with the TIP repo, read, render
$PY $E/p5c_families.py --rows "$W/score/rows_s*.jsonl" --arms C1 C0 base --pairs C1:C0 C0:base C1:base --out-dir $W/fam --repo $TANITAD_REPO
$PY $E/p5c_read.py --rows "$W/score/rows_s*.jsonl" --c1 C1 --c0 C0 --base base --human human --fam-ci $W/fam/famci_C1_vs_C0.json --tactical $W/tactical.json --sealed $W/SEALED.json --card <card> --recheck-files --controls $W/identity_C1.json $W/identity_C0.json $W/identity_base.json $W/b10_determinism.json <W-TRANSP ii-iv JSONs> --out $W/read.json
$PY $E/p5c_render.py --read $W/read.json --out $W/report.md
# K ledger at read time (the MM appends; this writes only entries + a merged copy)
$PY $E/p5c_seal.py entries --sealed $W/SEALED.json --ledger <live ledger_K.json> --out $W/entries.json
$PY $E/p5c_seal.py check   --ledger <live ledger_K.json> --entries $W/entries.json --sealed $W/SEALED.json --out $W/ledger_check.json
$PY $E/p5c_seal.py merge   --ledger <live ledger_K.json> --entries $W/entries.json --out $W/ledger_K_merged.json    # the MM then copies it over the ledger
```
Cost (ESTIMATED): the B10 second pass is 10 % of the B5 scoring, ~8 min on 2 processes (measured 3.3 min per 500 tokens x 4 arms); B12 adherence on the full navtest is seconds to a minute (10 s on 200 tokens with the 10,000-draw bootstraps).

## Not done / blockers
* B12 selection quality and the adherence random-proposal floor are not computed in the dry run: no dump or 64-column table exists for the stand-in arms. Blocker for the REAL read: C1 / C0 / P7 `--dump-proposals` dumps, and (if the rows are wanted) ~4 to 6.4 h CPU per arm for the 64 columns.
* The 12,146-token B10 pass is not run (no real arms; the dev box is shared).

## Deletions (CLAUDE.md auto-delete rule; everything is re-creatable from git `origin/agent/arch-inf-20260803` and the hold is the only copy of the new work)
* `C:/Users/Admin/p5cb/repo/taniteval/results` (288 MB, a partial `git archive` export that was trimmed to stay under the 300 MB scratch cap): deleted during the job.
* Final deletion of the whole scratch `C:/Users/Admin/p5cb/` (size and result logged below by the last step).
- deleted C:/Users/Admin/p5cb (160641 KB measured by du, includes the tip export, run rows, plans, scores, pytest basetemps, repo export), 2026-11-13, by literal path; rc 0
