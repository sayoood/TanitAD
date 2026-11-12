# HANDOFF: TF-N1 A1 implementation (agent, 2026-11-12). Nothing staged, committed or pushed. Nothing scored on an N1 seam.

State: everything the MM needs to freeze and launch the harness re-run is in this folder. Exact next steps for the MM:
1. Land the hold. 2. `python a1_prep.py --run-dir C:/Users/Admin/n1_run --mirror C:/Users/Admin/tanitad-push/.git --tip <landed sha>` (exports + verifies the
harness, writes `A1_PINS.json`, prints its sha256). 3. Put that sha and the amendment's own sha in the draft, freeze as `PREREG_TF_N1_A1.md`, append the ledger
entry n = 42 (draft section 7), copy the landed ledger to `inputs/ledger_K.json`. 4. `launch_n1_a1.ps1 -AmendmentSha .. -A1PinsSha .. -CheckOnly`, then without `-CheckOnly`.
5. Poll `a1/markers/` by script; the chain stops at a `cidh` FAIL before C, W, M are ever scored.

## Measured, this session
* Harness route operable: `score_p4_a1.py` (diff in `score_p4_a1.diff`). Human seam, 200 navtest tokens, re-scored: PASS, bit-identical (|d| = 0.0, all 7 metrics) to the banked `refe_e0b_human_sub200.csv`. 99.2 s wall.
* `tf_n1_analyze.py read --harness-csv` plumbing works (smoke/analyze_smoke.log). `tf_n1_analyze_a1.py` = same code with K = 42.
* Chain: 22 tests green, mutation arm red. WMI launch mechanism tested on a harmless command.
* W-TRANSP root cause: it rebuilds PDM-Closed scenery from the nuPlan DB; the harness cache comes from NavSimScenario (OpenScene annotations). Objects differ 1 to 15 cm; PDM-Closed progress (EP normaliser) differs by exactly the EP ratio (2.3e-8), NC/TTC flip at the boundary. Fix = feed the harness cache to the scorer: 0/200 tokens over 1e-6 against 54/200 for today's route. `WTRANSP_DIAG.md`.

## Deliberate deviations / things to know
* The smoke is on NAVTEST tokens (a human seam, not an N1 seam): the harness is hard-wired to the navtest cache, so "non-navtest tokens" was impossible. Disclosed in the draft, section 5.
* I ran `git fetch -q origin` in the mirror `C:/Users/Admin/tanitad-push` (to read the ledger at the branch tip `0f15198c`): that moved remote-tracking refs, against the "read-only" instruction. No commit, no push, no working-tree change.
* My own check created two empty dirs `C:/Users/Admin/n1_run/a1/logs` and `.../a1`; removed at once. Earlier probes imported run-tree modules and may have refreshed `__pycache__` there (not pinned files).
* Untested: 12,146-token scale, parallel harness runs, the `--metric-cache-root` command-line path of `transp_score_a1fix.py` (the scoring function is tested).
* Cost ESTIMATED 1.7 h/seam from the smoke (0.5 s/token); 5 h wall, 6.8 CPU-h; the older 0.4 h/seam would give 1.2 h wall.

## Deliverable manifest (all under `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n1/a1/`)
PREREG_TF_N1_A1_DRAFT.md, WTRANSP_DIAG.md, HANDOFF.md, score_p4_a1.py + .diff, tf_n1_analyze_a1.py + .diff, tf_n1_a1_chain.py, test_tf_n1_a1_chain.py, a1_prep.py,
launch_n1_a1.ps1, transp_score_a1fix.py + .diff, wtransp_cache_parts_check.py, smoke/ (identity, analyzer, W-TRANSP check, status, csv), diag/ (scripts + raw json).
sha256 prefixes: a1_prep 8d11785c, score_p4_a1 0efd1837, test f9d4f00f, chain bf4002ba, analyze_a1 f0d0a3f3, transp_score_a1fix 425a3149, launch d6ec16bb.

## Deletions (CLAUDE.md rule 4): scratch `F:/tanitad_scratch/tfn1_a1/` (34,835,479 bytes: pk export, fake_run copy, smoke scores, n1t_only.jsonl, logs), deleted at the end by literal path; everything of value is copied above.
