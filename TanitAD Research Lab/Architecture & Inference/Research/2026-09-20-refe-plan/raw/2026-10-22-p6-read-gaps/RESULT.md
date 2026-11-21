# P6-READ-GAPS: RESULT (2026-10-22, dev box; package `C:/Users/Admin/par18/p6gaps/`)

`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Base: mirror tip `6723462398a4bab14df34b4dd4d5944d6f848429` (read with `git -C C:/Users/Admin/tanitad-push show/archive`).
Nothing was committed, pushed or staged; no git working tree and no pod was touched; GPU: none (every run below is CPU). No P6 checkpoint scored or planned on a navtest token: the P6 file was used only on **navtrain** S0 plan tokens (G4 e2e) and the harness on **banked u400 / P2** navtest data (known-value controls).
Evidence classes: **MEASURED** = produced here with the artifact named; **PUBLISHED-OURS** = a banked number of ours in the repo, re-read here; **INHERITED** = another doc, not re-verified; **UNVERIFIED** = stated, not checked.

## HEADLINE (for the Master Mind)

| gap | state |
|---|---|
| **G1** C2 comparator `p2` in `stage3_full.py` | **DONE.** Control reproduces the published C2 - C1 = +0.57 [+0.02, +1.18] **bit-exact** (+0.5714677935040782 [0.017916605286551753, 1.1755185398706811]); a wrong pin and a swapped arm/base both turn it RED. |
| **G2a** `--dump-proposals` through `stage2_sft.py` | **DONE** (wired, command proven unchanged without the flag, exactly +1 flag with it). |
| **G2b** selection quality (random / actual / best-of-64) | **INSTRUMENT DONE and validated on P2's banked 64-proposal navtest table** (anatomy reproduced to 1e-9). The 64 harness columns on **P6's** dump are a read-day CPU job (about 4-6.4 h, runbook step 5): not run, by the navtest rule. |
| **G2c** command adherence | **STOPPED, as the brief orders: PREREG_P6 names the row but does not define it** (questions Q1-Q5 below). Not implemented, not guessed. Report it `NOT COMPUTED`, never a pass, never omitted. |
| **G3** bank the harness + hash guard | **DONE, and it found a bigger problem than the one asked** (see F1): at the tip the scoring step would have died **after the 4.8 h seam**. |
| **G4** `commands.json` for the S0 dump / candidate bank | **DONE**: both paths tested; refusal kept; **MEASURED end-to-end on P6's epoch-6 file** (CPU, 3 navtrain tokens). |
| tests | `python eval/test_p6_read_gaps.py` -> **`ZZP6GAPS PASS 31/31`** (`raw/test_p6_read_gaps.out`); every check carries a RED arm. `eval/selftest_stage3_full.py` -> `ZZSELFTEST_STAGE3 PASS 11/11` (`raw/selftest_stage3_full.out`). |

### Escalations (decisions or integration the MM must make; none is mine)
1. **Land the files** (list in the manifest). `srttc_s0_dump.py` is a **CRLF** file whose pin is a raw sha256: land it as a blob, with no CRLF filter.
2. **F1 / NT-FIX1 caller (decision to ratify).** The tip's `eval/score_navtest_refe.py` is the post-NT-FIX1 caller; it needs W3's `run_v1_fix1.py` (absent from the scratchpad) and byte-pins `run_v1.py` (the repointed scratchpad copy fails the pin). The banked harness + `RUN_V1_FIX1_ALLOW_DRIFT=1` fixes it, and I **measured that both callers give bit-identical per-token scores** (below). Whether the read runs under the pre- or post-NT-FIX1 caller is the MM's to ratify; I changed no scoring value.
3. **Command adherence (G2c)**: answer Q1-Q5 below, then the instrument is a half-day.
4. **Dry-run pin correction.** The dry run's "families md5 `a6a381b4`" for P2 is `p2_full/deployed_families.json`; P2's own families file is `d80a2599...` (CRLF) / `2bedac30...` (LF). The `p2` baseline carries both forms. Fix the dry-run text.
5. **`navsim_v1_win.py` line 49** (the one byte-different harness file): see G3.

---

## G1. C2 comparator (P2, 88.82, `pdm_route`, A7 ON, `navsim_v1`)

**Fix.** `eval/stage3_full.py`:
* `BASELINES["p2"]` (line 107): `pdms_x100` 88.8184 (PUBLISHED-OURS, `raw/2026-10-16-p2-full-table/anatomy.json`); csv md5 pin `bed5687fb75e884d45cf991c2ff391c9` (LF) / `1dc7d7a7e0ed0ec8950311471a914758` (CRLF checkout); seam md5 `49a1f6ecf28c5ebdaac4c68c2997c5f0`; families md5 `d80a2599f637710566408c49f786619d` (CRLF) / `2bedac30...` (LF). Candidate paths cover the dev-box `DATA`/S2 files and the repo copies (`P2_ARM`, line 91, = `raw/p2_readout/p2_full/arm`).
* `_md5_set` (line 151) lets a pin be a string or a tuple (a `core.autocrlf=true` checkout changes the md5 of csv/json); `first_matching` (line 158) and `pick_baseline`'s `pin_ok` (line 336) use it.
* The R2b status text for `--also p2` (line 504) is C2-specific: the P6 rule is applied to this comparator SEPARATELY; GO / NO-GO words are P2's amendment words.
* `STAGE2_SFT_MD5` (line 70) follows the new `stage2_sft.py` (`3a30ab4c5a122664c534f378144cb4a5`; tip `831a8d76...`); `FAMILIES6_CANDIDATES` (lines 71-75) now lists the banked `eval/harness/ev6e/families6.py` first (F2).

**Control (MEASURED, `raw/g1_probe.py`, test `G1 C2 comparator ...`).** From the banked per-token tables of P2 and u400 (12,146 tokens, 136 logs), `stage3_full.read_one` gives C2 - C1 = **+0.5714677935040782, 95 % CI [0.017916605286551753, 1.1755185398706811]**; published (PUBLISHED-OURS, `RESULT_P2_READOUT`) +0.57 [+0.02, +1.18]. Estimator: paired log-cluster bootstrap, B 10,000, seed 20260927 (answers *another draw of logs* only; NOT another training run, NOT another inference run).
**RED arms (all turn the control RED):** a wrong pin on the p2 entry (u400's csv md5) makes the pin gate fail; arm and base swapped does not reproduce the published value; a md5 outside the tuple is refused.

## G2. Selection quality and command adherence

### G2a `--dump-proposals` (`eval/stage2_sft.py`)
New flag (line 111); the seam command gains exactly `--dump-proposals <seams/stage2_<name>.proposals.npz>` (line 143) only when it is set; the final marker appends `proposal_dump=ok|MISSING` (line 232) and `res["proposal_dump"]` records path/bytes (lines 132, 231). The file holds all 64 proposals + the scorer's logits from the **same forward pass** as the seam (REPORT ONLY; about 95 MB on the full navtest per the `stage2_sft.py` docstring: ESTIMATED, confirm on the read).
**Test (MEASURED):** the seam command with the flag absent equals the OLD command string exactly; with it, exactly +1 flag. RED mutants: flag dropped / scratchpad path restored / pin guard removed / drift env dropped, each fails the wiring check.

### G2b selection quality (`eval/selection_quality.py`, new)
Definition used (and the one that reproduces the program's banked anatomy): **actual** = the PDMS of the proposal the planner picks; **random** = per token, the mean over the 64 proposals' PDMS; **best-of-64** = per token, the max. Each level and each paired difference (actual - random, best - actual) gets the paired log-cluster bootstrap (`proxy_eval.boot`, B 10,000, seed 20260927). The table needs the PDMS of **each of the 64 proposals as a single-proposal seam scored by the banked harness** (`seams` -> `score` -> `table` -> `read`; resumable thread pool; CPU).
Gates inside `assemble_table`: **G2 gate** = PDMS at the pick equals the arm's own csv bit for bit (a column mix-up, shuffled token order or a wrong dump cannot pass); **G3 gate** = all cells valid and finite. `read_table` flags the table **INVALID** if random >= actual (an instrument failure, C3's rule, never a result).
**Known-value control (MEASURED, test `G2 selection quality: P2's banked ...`):** on P2's banked 64-proposal navtest table (`C:/Users/Admin/p2full_work/final/table_p2_full.npz`, 12,146 tokens) the instrument reproduces `anatomy.json` to 1e-9: pick **88.8184088897293** [87.86107970695247, 89.77011829946386], best-of-64 **98.23109369448527** [97.81901516296715, 98.61925747503028], random **73.00314921968497** [71.48988586296944, 74.5661639319074]; zero-PDMS pick 509 tokens, oracle strictly better on 8,292. (These are P2 numbers, reproduced; they are NOT P6's.)
**RED arms (all fail the control):** shuffled pick; a duplicated proposal column; a token-shuffled table. Plus a synthetic world with analytic answers (actual = pick PDMS, random = mean over 64, best = max) and the full 64-column chain with a fake runner (resumable; a failed column and a swapped column are each caught).
**`level` (added after the first draft, found while writing the runbook):** PREREG's goal-bar test (94.55) needs P6's own level CI. The instrument would otherwise have had to take it from the 64-column table (4-6.4 h); `selection_quality.py level --arm-csv <csv>` (`level_from_csv`) computes it from the arm's own harness csv in seconds with the same registered estimator. **Control (MEASURED):** on P2's csv it reproduces 88.8184088897293 [87.86107970695247, 89.77011829946386] to 1e-9 (12,146 tokens, 136 logs); RED: another arm's csv (u400), a csv token with no log (an error, never a silent drop) and an empty csv each fail.
**Cost on the read day (ESTIMATED from P2's 64-column run, `raw/2026-10-16-p2-full-table`):** about 777k token-columns, 3.9-6.4 h CPU, 0.34 GB seams + about 0.4 GB harness dirs on this exFAT. **NOT run on P6**: it needs P6's dump, which needs the navtest read.

### G2c command adherence: STOP (question for the MM)
PREREG_P6 line 125, verbatim: *"**TACTICAL** | lateral, longitudinal and 5-way decision accuracy/κ; **command adherence** of the selected proposal; **selection quality** as random / actual / best-of-64 on the same tokens (a random control is mandatory)."* That is the whole definition: a name.
**Searched** the repo at the tip under several names (`command adherence`, `command_adherence`, `commanded_route`, `driving_command`). The only code hits that say "command adherence" are **stubs** (`hn_screen_families.py:19,103`, `hn_screen_read.py:513`, `s1_read.py:808`: "NOT COMPUTABLE / UNAVAILABLE"). The nearest real instruments:
* TanitEval V2 N2 `commanded_route_vs_trajectory` (`taniteval/taniteval/hierarchy.py`): agreement and Cohen's kappa of the commanded route against the executed trajectory's manoeuvre class, with a turn-active subset (and a `<5 deg` threshold).
* `taniteval/taniteval/four_families.py:1330` `tactical_from_trajectory`, lateral decision, which uses the canonical labeller `refc_tactical.factor_from_kinematics`.
**The choices that change the number (PREREG does not make them):**
* **Q1.** How is the selected proposal turned into a left / straight / right class: the canonical labeller (no new thresholds) / a net-heading threshold / a +-2 m end-offset?
* **Q2.** Which statistic: accuracy, kappa, or both, on all 12,146 tokens or on the turn-active subset? (Navtest commands, MEASURED from the export: LEFT 2,501 / STRAIGHT 8,070 / RIGHT 1,575 / UNKNOWN 0. Straight dominates, so accuracy on all tokens is mostly a straight-rate.)
* **Q3.** The selected proposal as it leaves the planner (raw proposal) or the executed plan after the A7 transform?
* **Q4.** A paired CI against C1 / C2, or a level only?
* **Q5.** A random-proposal floor from the dump (the 64 proposals' mean adherence), as selection quality has?
**My recommendation (NOT implemented):** the canonical labeller on the executed plan; accuracy and kappa on all tokens and on the turn-active subset; paired CI vs C1 and C2; with the random-proposal floor from the dump. Everything needed (the dump, the labeller, the command table) is in place; this is a decision, not a build blocker.

## G3. Bank the harness; pin; guard

**Banked** under `PK/eval/harness/`: `score_e.py`, `ev6e/families6.py`, `w3e/e1/navsim_win.py`, `w3e/code/*.py` (17 files) + `HARNESS_PINS.json` (21 .py; per file `sha256_lf`, bytes, scratchpad sha256, origin). `eval/harness_pins.py` (`verify`, `require`, `build`, CLI `--verify|--build`) hashes **LF-normalised** bytes (a CRLF checkout still verifies; tested), and refuses a changed file, a missing file, an **unpinned extra .py**, a missing pins file, and an **unknown scoring caller**. `stage2_sft.py` imports the repo copy: `HARNESS = PKG/"eval"/"harness"` (line 46), `SP`, `EV6E`; `hp = HP.require(HARNESS, caller=PKG/"eval"/"score_navtest_refe.py")` (line 116) is the **first action in `main()`**, before the lock or the seam.

**Byte-identity to the scratchpad the SFT-4 / C2 results used (MEASURED, `raw/prov_check.py`, `raw/harness_provenance.jsonl`, test `G3 byte identity ...`):** 19 of 21 files are **byte-identical** to the scratchpad files; 1 (`navsim_v1_win.py`) differs by **one line**; 1 (`w3e/code/run_v1_fix1.py`) is absent from the scratchpad and is the W3 repo original from the tip. Every scratchpad `w3e/code` file also equals its W3 repo original after the `D:/` -> `E:/` sed (the scratchpad = the repo originals, repointed).
* **The one differing line:** `w3e/code/navsim_v1_win.py:49`, `E1_WRAPPER`. The scratchpad had a hard-coded absolute path **into a previous session's `%TEMP%` scratchpad**; the banked copy resolves `../e1/navsim_win.py` relative to the file (`os.path.normpath(...)`). `HARNESS_PINS.json` records this as `scratchpad_plus_one_line`. Without it the banked harness would have depended on the scratchpad it replaces.

**RED (shown once, as asked):** one changed byte in `run_v1.py` -> the guard refuses and names the file (test `G3 RED: one changed byte ...`); also a missing file, an extra unpinned .py, a missing pins file, an unknown caller.

### F1. FINDING beyond the brief: at the tip the scoring step would have died after the 4.8 h seam
`eval/score_navtest_refe.py` at the tip is the **NT-FIX1** variant: it loads W3's `run_v1_fix1.py`, which byte-pins `run_v1.py` to sha256 `125670597d99...` and applies fixes A (cross-drive relpath), B (stale csv quarantine), C (no summary on count mismatch). The scratchpad `w3e/code` has **no `run_v1_fix1.py`** and its `run_v1.py` is the sed-repointed copy (fails the pin). So once the working tree matches the tip, stage 2 would run the 4.8 h seam and **then** fail at the scoring step.
**Fix:** bank `run_v1_fix1.py` (the W3 original) and have `stage2_sft.py` set `RUN_V1_FIX1_ALLOW_DRIFT=1` for the scoring call (line 179) -- the documented switch for the repointed copy.
**Real-harness controls (MEASURED, CPU only, 24 banked u400 navtest tokens, `raw/harness_ctl/`, `raw/harness_ctl_cmp.py`):**
| arm | what | result |
|---|---|---|
| A | banked harness, **pre-NT-FIX1** caller | PASS; 192 per-token cells **bit-equal** to u400's banked csv; csv sha256 `a71dd17cca45c7109c89d61c8e5b0788a2d58dab6b46e28ef680a960fe341f27` |
| B | banked harness, **NT-FIX1** caller + drift env | PASS; **same csv sha256**, 192/192 bit-equal; `run_v1_matches_pinned: false` is recorded in the compare json (the drift is declared, not hidden) |
| RED1 | the **scratchpad** `w3e` under the NT-FIX1 caller | exits: `run_v1_fix1.py is missing` (`red1_scratchpad_under_tip_caller.log`) |
| RED2 | the banked `w3e` **without** the drift env | raises `Fix1DriftError` (`red2_banked_no_drift_env.log`) |
So the two callers give identical scores on this evidence; **no registered setting changed** (the decision is which caller the read runs under, escalation 2).

### F2. `stage3_full.py` resolved a DIFFERENT `families6.py` than stage 2 (found by running the selfcheck, fixed)
`FAMILIES6_CANDIDATES` tried the W3 original under `D:/Projects/TanitAD/FlyWheels/...` before the banked copy, so on this box stage 3's families fallback and its selfcheck ran a different file from the one stage 2 runs (md5 `964c4ae0...` vs the banked `e275637c...`; they differ only in the `D:/` -> `E:/` adapter path, MEASURED, `raw/f6probe/`). **Fix** (`stage3_full.py:71-75`): env override, then the banked pinned copy, then the W3 original; the previous-session `%TEMP%` scratchpad is no longer a candidate. **Test + RED:** `G3 stage3_full resolves families6 to the banked, pinned copy FIRST`; the reversed (old) order fails it.

## G4. `commands.json` as the command source for command-mode checkpoints

**Fix.**
* `raw/2026-10-18-sel-rank/code/srttc_s0_dump.py` (CRLF; raw sha256 `28e5e56bf2f2b15825a74b81549ea2f078bd7e86bd95c24047d0f1bbfea4441f`; the landed one is `1b41a9fbc21426f949dd4be2362dda467a7d04c630790df9a703cb2b184d2ac3`):
  `load_command_table(path, sha256)` (line 51) refuses a sha mismatch, an empty table or a row that is not `"<log>/<token>" -> one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN]`; `command_vec` (line 66) **raises** for a missing token (no default command ever); `--commands` / `--commands-sha256` (lines 88-89; env `SRTTC_COMMANDS_JSON` / `_SHA256`, because the landed `build_candidate_bank` driver builds the dump argv and cannot forward a flag). The per-token hop (line 162) calls the **seam's own** `set_driving_command` BEFORE `planner.infer`. **The refusal is kept** for a command-mode checkpoint with no source (lines 110-111, rc 2). A goal-mode checkpoint (P2) ignores a supplied source.
* `raw/2026-10-19-p6-read-prep/code/build_candidate_bank.py`: `prepare --commands/--commands-sha256`; `ckpt_nav_mode` (line 125, reads `meta.nav_mode`, default goal); `check_command_source` (line 133: the table must cover **every** plan token and be one-hot); `apply_command_source` (line 150: command-mode + source -> exports the env and requires the G4 dump pin; command-mode with no source -> `BuildError`; goal-mode -> ignores the source and pops stale env); `S0_PINS` accepts the landed OR the G4 dump (lines 53-57); `cmd_dump` calls it before the GPU lock (line 342). `smoke` stays P2-only by design (it compares against S0's banked P2 logits).
**Tests (MEASURED; the file totals 31/31):** the real P6 `commands.json` covers **all 8,922 S0 plan tokens** (103,037 rows; `raw/s0_plan_vs_p6_commands_coverage.json`); RED: a missing token, a non-one-hot row, a wrong sha each refused; both decisions (`main()` with faked heavy imports): no source -> rc 2, with source -> passes, goal-mode -> untouched; RED mutants (default-command on a miss; refusal dropped; `apply_command_source` exporting a default) each fail; a static check that the hop sits before `planner.infer`.
**End-to-end on the REAL P6 epoch-6 checkpoint (MEASURED, CPU, navtrain S0 plan tokens, `raw/g4_e2e.py`, `raw/g4_e2e/`, `raw/g4_e2e.out`):** three tokens, one per command (`000c188876bf5dba` LEFT, `02ebd46f5cd6566d` STRAIGHT, `0f2b9eb422e956f1` RIGHT).
* **A, no `--commands`:** `REFUSED`, **rc 2** (62 s): the kept refusal works on the real file.
* **B, P6 table:** `ZZS0DUMP_OK 3 0`, rc 0 (206 s); report `commands.set == 3`, `nav_mode command`, `misses 0`, table sha256 `8b516ee3f6811853c367e561db8ddf21ea5df9a79a828e960cc4754d1ece5367`, 103,037 rows.
* **C, LEFT <-> RIGHT swapped table (the wiring control):** `ZZS0DUMP_OK 3 0`, rc 0 (192 s), `commands.set == 3`, table sha256 `e219ab4c...`. Against B: the **STRAIGHT** token's 64x6 scorer logits are **bit-identical** (max |diff| 0.0, pick 1 -> 1); the **LEFT** token's logits move by **9.99** and its pick changes **46 -> 9**; the **RIGHT** token's move by **10.24**, pick **41 -> 22**. So the command read from the table does reach the model, token by token, and only where it should (`raw/g4_e2e/g4_e2e_result.json`, `verdict_ok: true`).
  Honest note: `raw/g4_e2e.py` crashed in its own analysis block AFTER all three model runs had finished (the `--lean` dump stores no `proposals` rows, only logits + pick); `raw/g4_e2e_analyze.py` reads the same on-disk logs/npz and writes the result json; the model was not re-run. These three tokens are NAVTRAIN S0 plan tokens, a wiring proof on the real file, not a score.

## Reproduce
`python raw/make_sandbox.py` (tip `eval/` + `refe/` with `code/` on top; scratch, deleted after), then in `sandbox/<PK>`: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/test_p6_read_gaps.py` (the bare `python` has no numpy). Unified diff of the four modified files vs the tip: `code/P6_READ_GAPS.diff`. TIP copies of the four modified files: `raw/*.TIP.py`.
`stage3_full.py --selfcheck` in the sandbox: **`R2_PRECONDITION_MET=True`, `checks_ok=True`, every regression arm failed as required** (`raw/selfcheck_sandbox.out`, `raw/selfcheck_sandbox.json`; MEASURED, CPU, about 5 min): it includes `families6_reproduces_banked_u400` = ok, 0 differences in the families block, using the banked `families6.py` (md5 `e275637c...`), and the `stage2_sft.py` md5 pin `3a30ab4c...`. The FIRST selfcheck run (`raw/selfcheck_sandbox_run1_D_families6_first.out`), with the old candidate order (the W3 original under `D:/Projects/TanitAD/FlyWheels/...` first), read `families6_reproduces_banked_u400: ok false, no output` and `R2_PRECONDITION_MET=False` (it ran concurrently with CPU jobs; the cause was NOT root-caused: the same D: file run directly later returns rc 0, and its families equal the banked copy's modulo the `D:/` vs `E:/` adapter-path string, `raw/f6probe/`). Whatever the cause, the order was wrong for G3 (stage 2 uses the banked copy, stage 3 should too): see F2.

## What is NOT done, and why
* Selection quality **on P6** and command adherence **on P6**: they need the navtest read (one look, `model_final.pt`). Selection quality is runnable by `RUNBOOK_P6_READ.md` step 5; adherence is blocked on Q1-Q5.
* The 4.8 h seam under the real harness with the banked harness path was not run end-to-end (it takes the GPU and a navtest P6 pass). What was run is the harness scoring half on 24 banked u400 tokens (F1 table) and the seam command shape (tested).
* `PKG` in `stage2_sft.py` stays the hard-coded E: tree (unchanged). The estimator answers *another draw of logs* only; none of these numbers speaks to training-seed or inference-seed variance (H-ESTIM-SEED-1).

## DELIVERABLE MANIFEST (nothing staged, committed or pushed; **every item below exists in ONE place only: `C:/Users/Admin/par18/p6gaps/` on the dev box. Single-copy until the MM lands it.**)
Location key: **repo:`<path>`** = the path it must land at (relative to the repo root, `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`); the bytes today are at `C:/Users/Admin/par18/p6gaps/code/<same path>`. "pod" = not used. "worktree" = not used. Hashes of every file: `MANIFEST.sha256`.

| # | artifact | lands at (repo) | kind | where it lives NOW |
|---|---|---|---|---|
| 1 | `stage2_sft.py` (G2a, G3) | repo:`PK/eval/stage2_sft.py` | MODIFIED (md5 `3a30ab4c...`) | **dev box `p6gaps/code/` only: single copy** |
| 2 | `stage3_full.py` (G1, F2) | repo:`PK/eval/stage3_full.py` | MODIFIED | **single copy** |
| 3 | `harness_pins.py` (G3 guard) | repo:`PK/eval/harness_pins.py` | NEW | **single copy** |
| 4 | `selection_quality.py` (G2b + `level`) | repo:`PK/eval/selection_quality.py` | NEW | **single copy** |
| 5 | `test_p6_read_gaps.py` (31 checks, each with a RED arm) | repo:`PK/eval/test_p6_read_gaps.py` | NEW | **single copy** |
| 6 | banked harness: 21 .py + `HARNESS_PINS.json` | repo:`PK/eval/harness/` (`score_e.py`, `ev6e/families6.py`, `w3e/e1/navsim_win.py`, `w3e/code/*.py` x17) | NEW (20 byte-identical to the scratchpad or the W3 original, 1 with one line changed) | **single copy** (the scratchpad originals sit in a previous session's `%TEMP%`, which is why this was banked) |
| 7 | `srttc_s0_dump.py` (G4; **CRLF, land as a blob**) | repo:`PK/raw/2026-10-18-sel-rank/code/srttc_s0_dump.py` | MODIFIED (raw sha256 `28e5e56b...`) | **single copy** |
| 8 | `build_candidate_bank.py` (G4) | repo:`PK/raw/2026-10-19-p6-read-prep/code/build_candidate_bank.py` | MODIFIED | **single copy** |
| 9 | `P6_READ_GAPS.diff` (unified diff of rows 1, 2, 7, 8 vs the tip) | package only | derived | single copy |
| 10 | `RUNBOOK_P6_READ.md` (read-day commands, repo paths) | repo:`PK/eval/RUNBOOK_P6_READ.md` (suggested) | NEW | **single copy** |
| 11 | `RESULT.md`, `HANDOFF.md`, `MANIFEST.sha256` | package root (RESULT.md suggested at repo:`PK/raw/2026-10-22-p6-read-gaps/RESULT.md`) | NEW | single copy |
| 12 | `raw/`: test + selfcheck + selftest outputs, harness controls (`harness_ctl/`), G4 e2e (`g4_e2e/`, its npz/log/result), probes, patch scripts, TIP copies of the 4 modified files, `harness_provenance.jsonl` | suggested repo:`PK/raw/2026-10-22-p6-read-gaps/raw/` | evidence | **single copy** (the e2e npz/logs are ~MB, no secrets; the HF token was never printed) |

Not in the package (scratch, may be deleted): `pkx/` (tip export), `sandbox/` (overlay used by the tests; `raw/make_sandbox.py` rebuilds it and DELETES the old one). Deleted by me as stale scratch: `pkx2/`, `pkx3/`, `sandbox_extra/`.
The harness controls (F1) wrote scratch score/seam labels `refe_p6gaps_ctl*` under the data dir; I deleted them by exact name and re-checked that none remains (`find ... -iname '*p6gaps*'` empty, 2026-10-22). Nothing else was written to `E:/Projects/TanitAD` or the data dir; nothing to the pod or git.
Integration escalation: **none of the rows is landed.** The MM lands rows 1-8 and 10 (a new file or a CRLF blob each; no merge conflict is expected: base = tip `6723462`).

