# L2-NEXT launch-gate regression: verdict (MM, 2026-11-27)

**Verdict: 0 regressions attributable to the L2-NEXT patch** (with CODE AMENDMENT 1). Every failing test either fails on the base tree too, or failed for a sandbox defect that was then fixed and re-run.

**What ran** (`cmp_chain.sh`, serial, CPU, commit gate >= 6,000 MB): the 58 launch-gate tests on the PATCHED tree (`e1eb6bf3` + the L2-NEXT files), then the BASE tree (`e1eb6bf3`) only for the tests that failed on patched. A test that passes on patched cannot be a regression. Deviation from the SAFE-HEAD regression, disclosed: patched first, and base only for its failures.

**Raw outcome** (`COMPARE.tsv`): "regressions 2 new_failures 0".
1. `test_aux_s1_wiring.py` (patched rc 1, base PASS 20). **Sandbox defect, not the patch.** The agent's sandbox export put CRLF into 8 non-.py files in BOTH sandboxes (fixtures/*.txt, *.json, LICENSE, NOTICE.md, README_DIAGNOSTICS.md), so the test's pinned-blob check (`baseline_fixture_is_the_pinned_blob`) failed.
   - The MM restored all 8 from the `e1eb6bf3` blobs (`git hash-object --no-filters`, 16/16 OK) at ~18:45Z.
   - `rerun_fx.sh`: patched **PASS 20**, base **PASS 20** (`fx_patched.tsv`, `fx_base.tsv`).
2. `test_hnscreen_chain.py` (patched FAIL 1, base PASS 31). Same cause: the patched run predates the restore, the base run follows it. Re-run on patched after the restore: **PASS 31** (`fix2.tsv`).

**The new test** `test_safehead_next.py` (full mode) FAILED twice on deliberate-regression arms that were no-ops for the new families: shmlp `sh_ckpt_bias_dropped` (G-CKPT) and ephead `sh_live_term_dropped` (G-LIVE). Fixed by `../CODE_AMENDMENT_1.md`. Re-run: **PASS 17**; `test_safehead.py` **PASS 14** (`fix2.tsv`).

**Fail on base too (pre-existing, backlog; not this patch):** `test_carve_feed`, `test_launch_gate`, `test_p2r_chain`, `test_p7_ship_list`, `test_rls1_chain`, `test_s1aux_chain`, `test_s1z_preflight`, `test_s1zsm_preflight`, `test_sft4r_chain`, `test_vdac_stage0_forward`. In the SAFE-HEAD regression this morning, `test_carve_feed`, `test_launch_gate` and `test_p2r_chain` failed on base too. ⚠ Several of these run in a sandbox without `.git`, so they may be ENVIRONMENT failures of the sandbox rather than code failures; each is a backlog item to establish, not assumed.

**Evidence:** `COMPARE.tsv`, `patched.tsv`, `base_failed.tsv`, `fx_*.tsv`, `fix1.tsv` (partly killed: the dev box's commit fell to 415 MB beside TRK-SCORE's seam), `fix2.tsv`; scripts `cmp_chain.sh`, `rerun_fx.sh`, `l2next_retest.sh`.
