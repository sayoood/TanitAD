# MANIFEST SCORER1-IMPL-2 (all under E:/Projects/TanitAD-artifacts/scorer1_impl_20261108/v3/; nothing staged in a repo, nothing committed, nothing on the pod; the Master Mind lands)
Base: tip 0fd9a3dc66a55ed5fbabb96514855fa4c87ab19f (agent/arch-inf-20260803). `blobs.tsv` = repo path, base blob sha, file, sha256[:16]. `scorer1_impl_v2.patch` = unified diff (LF) of the 6 modified repo files vs `base/` (the tip blobs). All files are LF.
Repo dir for all `refe/` files: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`.
| artifact | where | status |
|---|---|---|
| scorer_finetune.py (MOD) | v3/files | repo:refe/ target; ONLY here until landed. PairLoss, `--pair-*`, `--navtest-logs/-tokens` + `navtest` data block, G16 (moved before G14 so the P6-HN-S reversal text stays contiguous), `cap["pair_in"]` only when `pairer` exists |
| gate_checks.py (MOD) | v3/files | G16 judge, pair/navtest DVB rows (navtest flags kind `consumed`) |
| launch_gate.py (MOD) | v3/files | profile `refe_scorer1` (`recipe_rules` None: refe_sft4r's key must stay unique, test_sft4r_gate), `scorer1_argv`, `scorer1_refusals`, `sha256_of_file`, `sft4r_refusals` reads `prof["recipe"]` |
| hnscreen_layer.py (MOD, 3 lines) | v3/files | `peel` first peels the SCORER1 layer (outermost) |
| scorer1_layer.py (NEW, generated) | v3/files; generator v3/tools/gen_scorer1_layer.py | 11 reversal entries; peel(trainer) == tip blob 51a28163 proven by generator and test |
| test_launch_gate.py (MOD), test_dump_scores.py (MOD) | v3/files | G16 exempt for r0 / unknown-gate fixture; the parser-dest pin names the six SCORER1 dests |
| test_pair_loss.py, test_scorer1_gate.py, test_scorer1_mutation.py, test_scorer1_layer.py (NEW) | v3/files | all ZZTEST_PASS on the final tree |
| scorer1_pod.sh, scorer1_chain.sh (NEW) | v3/pod | repo:refe/ target; NOT on the pod; bash -n clean; argv diffed by test_scorer1_gate T10 |
| stage0_gap.py (NEW) | v3/stage0 | tools target; `--selftest` ZZTEST_PASS |
| navtest_logs.json, navtest_tokens.txt.gz | v3/data/navtest_ids | byte copies of raw/2026-10-22-p7-session-reserve/raw/src (sha256 in the profile); go to /workspace/data/refe_scorer1/navtest_ids/ |
| SPEC_GAPS_V2.md, HANDOFF.md, MANIFEST.md | v3/ | |
Each repo file above exists in exactly ONE place (this dir) until the Master Mind lands it.

## Suite (every test that reads launch_gate or pins scorer_finetune.py: 43 files), on a tip export under F:/scorer1_v2_refe, CPU only, differential against F:/scorer1_v2_base (the same tree with the unchanged tip files)
Regressions my first edits caused and that the suite caught, all FIXED: (1) `cap["pair_in"]=(.., ego[:,6], ..)` ran with the lever OFF -> IndexError on fixtures with < 7 ego slots (test_aux_s1_wiring OFF-path bit-identity, test_carve_feed_gate); (2) the profile copied `recipe_rules` (test_sft4r_gate); (3) the G16 block sat between G15 and the `# G5` comment and broke the P6-HN-S reversal text (rl_safeobj, rl_r0c, rl_pickcon "text occurs 0 times"); (4) the older pins of the trainer (test_rl_s1b P_pinned_main.train_losses) needed a reversal layer -> scorer1_layer.py; (5) test_dump_scores pins the parser's dests.
* PASS on the final tree: aux_g12_bf16, aux_s1_wiring (OFF path bit-identical to the pre-edit main, 19 RED mutations), carve_feed_gate, gate_gpu_probes (CUDA check SKIPPED), gate_suite_prerun (144), hn_ctx_anchor (72), hnscreen_ctx_chain (22), sft4r_gate (153), dump_scores (16), pair_loss, scorer1_gate, scorer1_mutation, scorer1_layer.
* PASS on the tree one iteration earlier (before the layer / the G16 move; their texts are not touched by those two edits, but they are IN THE RERUN below): hn_content_fp (81), hnscreen_chain (31), launch_gate --static (1169), s1c (220), rls1e_chain (936), rls1_queue (254), p7_head_wiring (182), p7_integration (225), session_reserve (43), suite_nesting (128).
* FAIL IDENTICALLY ON THE UNCHANGED TIP (environmental: this export lacks eval/harness, raw/ and code/; not caused by this work): ego_yaw, sft4r_chain, s1aux_chain, p2r_chain, rls1_chain, p7_ship_list ("1 failing, 24 passing" on both), carve_feed ("C_1_the_draw_IS_SO_CARVE..." on both).
* NOT YET RE-VERIFIED ON THE FINAL TREE (the rerun was still running at hand-off; see HANDOFF "Suite rerun"): rl_s1b, rl_safeobj, rl_r0c, rl_pickcon (these pass on the tip; they need scorer1_layer to pass on the new trainer), hn_screen_wiring, rl_epo, rl_terms_label, r0e_b9b10, label_dir_refusal, p2rep_chain, p6_chain, p6_levers, p7_chain, launch_gate (static) and the rest of the "earlier iteration" list. A first parallel run of those hit `No space left on device` on C: (a 6.5 GB runaway REPL task output, deleted) and file collisions between 3 parallel runners; they are being rerun with 2 runners and TMP on F:.
* Pin-edit hazard: no pin, sha or fingerprint of another profile was changed; the 6 profile-fingerprint tests are in the list above.

## Deletions (CLAUDE.md rule 4)
* `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/c216c849-5fe7-4d43-a3be-32429241f811/tasks/bfbjum908.output`, 6,539,591,390 B: this session's own runaway background task output (a bare Python REPL loop writing a traceback; C: was at 0.06 GB free). Nothing needed it. C: free after: 6.6 GB.
* NOT deleted, still needed by the running suite rerun: `F:/scorer1_v2_refe` (tip export + results), `F:/scorer1_v2_base` (the unchanged-tip comparison tree), `F:/scorer1_v2_export` (an abandoned partial export, safe to delete). All re-creatable from the tip with `scratchpad/export.py`-style `git cat-file` exports; delete them by literal path when the rerun is read.
* NOT deleted, refused by the environment's safety check on `rm -rf`: `F:/TanitAD_scorer1_impl_20261108` (v1's work tree, 12 MB, re-creatable; the files in it are banked: p1_diag.py and u400_reference.py identical in the selection/behaviour packages, the v1 edits in `../files` and `../scorer1_impl.patch`). The user should delete it.

## v3 delta (2026-11-08, SCORER1-IMPL-3; v2/ is untouched)
v3 = v2 + three TEST-ONLY patched files (triage `fix_*.patch`, applied to the tip blobs, LF, diff == patch line for line). Tip at build: `72151b85` (suite tree); bases re-verified at `9f7df1aa` (`tools/check_bases.sh` -> ZZBASES_OK). No pin, sha or expected value changed anywhere.
| artifact | where | status |
|---|---|---|
| test_rl_s1b.py (MOD, base f9b8d4e3 -> 18be735c) | v3/files | peel to the pre-SCORER1 trainer before the R0 segment pin |
| test_rl_epo.py (MOD, base a697103d -> 6fbe830f) | v3/files | `pair_w=0.0` in `a_ns`; the two AST hops skip the `pair_refusals` line / the `--pair-calibrate` loop |
| test_hn_screen_wiring.py (MOD, base 55171c24 -> 93774176) | v3/files | PIN group loads `scorer1_layer.peel(SF)` |
| base/ (3 new tip blobs), blobs.tsv (+3 rows), scorer1_impl_v3_tests.patch (the 3 patches concatenated) | v3/ | scorer1_impl_v2.patch still describes the other 6 modified files |
| fp_base.json, fp_v3.json | v3/ | sha256(repr(sorted(PROFILES[n].items()))) for every refe profile, base vs v3 |
| tools/run_suite_v3.py, launch_detached_v3.py, compare_v3.py, check_bases.sh | v3/tools | runner (3 workers, 2400 s), launcher, base comparison, base-blob check |
| LAND_PLAN.md, HANDOFF.md | v3/ | |
Suite tree (re-creatable, NOT banked): `F:/scorer1_v3/t` (tip 72151b85 `refe/ eval/ splits/ SHIP_MANIFEST.json` + v3 files, `eval/harness` removed to match the base's environment); results `F:/scorer1_v3/res/` (results.txt, logs/, COMPARE.txt). Ship-list probe trees `F:/scorer1_v3/ship3/{tip,v3}`.
Deletions (CLAUDE.md rule 4): none yet; see HANDOFF for the list owed. Environment refuses `rm -rf` on these F: trees; delete with a literal-path PowerShell `Remove-Item -LiteralPath`.
