# HANDOFF: A3.2 (TB-0 R-A on rank-0 rows), MM agent 2026-11-16

State: DONE in the hold, STAGED NOTHING in the repo, nothing committed or pushed.

## What exists (hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a32/`)
- `a1/p5c_table_gates.py` (new, sha256 `0c567514...0d35`), `a1/p5c_pod_label.sh` (control call only gets `--tc20s-ranks 0`, PIN_GATES updated), `a1/test_p5c_table_gates.py`, `a1/test_p5c_pod_label_a31.py`.
- `PREREG_P5C_A3_2_DRAFT.md` (pins, tests, proof, post-data disclosure), `proof/` (pod gates.json / stdout / meta for both scopes, test logs, pod scripts).
- Results: gate suite 199/199 (163 old unchanged), runner suite 49/49 (42 old unchanged), 13 mutants RED. Pod: scope 0 gives TB-0 PASS, rank-0 max 9.357083996519577e-10, disclosed rank-1 max 1.4901395672950457e-09 (2 over tol); scope all reproduces run 1's gates.json byte for byte (FAIL 1.4901395672950457e-09).

## Next steps for the MM (integration is yours)
1. Review, then freeze the amendment text: sha256 the PREREG draft, add `PREREG_P5C_A3_2.sha256`.
2. Land the two source files + two tests into the repo (the runner and gates live at `PK/raw/2026-11-11-p5c-readiness/a1/` and `a31/a1/`; the a31 test finds the gates via `PKRAW/a1`). Use `land.py` with base blobs: gates `5c2f49cf`, runner `727e72d4`, tests `82ce4a58` / `78dcf8ee`. Land copies (the lander rewrites CRLF sources). The gate suite needs the p5c/a1 fixtures and `p5c_dual_labels.py` beside it, and the OLD a1 runner (it parses its text); I staged them in scratch.
3. Ship the new gates + runner to `/workspace/data/p5c_ship/` (the runner pins `PIN_GATES`; the shipped gates are still the old sha, so the runner would refuse until shipped). Re-run the control pass or just the gates call; C1 is then unblocked by R-A.
4. Quote TB-0 as "PASS under A3.2 (R-A on rank-0; rank-1 max 1.49e-9 disclosed)", never as the pre-A3.2 gate.

## Notes
- The test file writes under `P5C_TMP` (default `F:/tanitad_scratch/p5c_gates_agent/test`); I set it to my scratch. The fixture's rank-1 twins SHARE nested lists with rank 0 (`dict(d, rank=1)`): new tests unshare them via `fresh()`; old tests are unaffected.
- Pod: `/workspace/data/p5c_a32_check/` holds the shipped gates, `chk.py`, and r0/rall gates.json, stdout, meta (104 KB). I deleted the four `merged_labels_*` files there (14 MB, re-creatable by rerunning; log below).

## Deletions (all mine, literal paths)
- pod `/workspace/data/p5c_a32_check/{r0,rall}/merged_labels_all.jsonl` 3,568,662 B each; `merged_labels_rank0only.jsonl` 3,578,274 B each.
- `F:/tanitad_scratch/p5c_gates_agent/test/a32_cli` (3.8 MB, my test's output dir).
- `F:/tanitad_scratch/p5c_a32_tmp/` (36 MB) at the end of the job.
