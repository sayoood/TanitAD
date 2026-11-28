# Integration triage: which stranded-work INTEGRATION items are real defects (2026-10-18)

Tip analysed for A and B: `363a24ff` (`363a24ffd753985efd91c857d9453d8d88990cbe`); section C re-checked at the END of the run against `88fed187` (`88fed187037d4421214af6944200960ffd8f8ad5`). No file under `stack/`, `taniteval/`, `tools/`, `colab/` or `data/` changed between the two tips (MEASURED: `git diff --name-only`), so the A and B verdicts stand at the later tip. Mirror `C:/Users/Admin/tanitad-push`, branch `agent/arch-inf-20260803`. E: files were read only. Evidence class of every count here: **MEASURED** (scripts in `scripts/`, artifacts in `raw/`).

## Headline

* **MISSING = 5 of the audit's 29 package-local code variants (A), and all 5 are ONE defect: the v7F R1-R6 change set never reached `stack/`.** The other 24: APPLIED 3, SUPERSEDED 15, OBSOLETE (variant held by decision) 6.
* **None of the MISSING items touches a LIVE chain.** P6 runs `refe/` (self-contained: no `from tanitad`/`from taniteval` import anywhere under `2026-09-20-refe-plan/refe`, MEASURED by `git grep` at the tip; it only references `launch_gate`, and the one launch_gate variant in the 29 is SUPERSEDED). The refcv7/refcv8 stack trainers (`refc_v3_train.py`, `refc.py`, `refc_v3.py`, `refcv8_train.py`, `refcv8_conditioning.py`, `launch_gate.py`): every variant targeting them is APPLIED, SUPERSEDED or HELD, none MISSING. The MISSING files are `train_v6_staged.py`, `models/v6.py`, `declared_vs_built_v6.py`: no `stack/ops/runs.d` argv exists for v6/v7F, `launch_gate.py` has **0** occurrences of `v7f` (no gate profile, so under the PI's launch-gate rule a v7F run cannot start), `refc_v3_train.py` mentions `train_v6_staged` only in a comment, and the registers' last v7F row is at line 15812 of 16817 (a 2026-09-20/21 row). **Dormant, not live.**
* **Beyond the 29 (extension, MEASURED from every `package path -> repo path` line in the 74 LANDING_READY manifests): the manifests declare 39 mappings the tip does not hold** (21 target files differ, 18 are NEW files absent), in five families. 13 are the held refcv7 R6 / pre-HQS rows and 10 are the v7F merge (both already counted in A). **The other 16 are new findings:** refav1 R1/R3/R5/R6 (11), refav1 R1-VMAX (3), the v7F launch-gate profile (2). Two of those touch LIVE files: `refa_v1_plan.py` and `taniteval/tools/refav1_arm.py` (both edited on the tip 2026-10-18, HJ-UNITS-2) lack the R1-VMAX cap (flag-off, bit-identical when off per the package; no live run passes it). A clean 3-way patch is supplied for these two.
* **Candidate edits (B, 47): MISSING 23 / SUPERSEDED 18 / REGISTER (report only) 5 / machine-local 1.** The one that matters for running code: **the leaderboard builder fix is on E: and not on the tip** (`taniteval/taniteval/leaderboard/{build,render,sources,__main__}.py`, `leaderboard_sources.json`, `tests/test_leaderboard.py`): the tip's builder has no TOMBSTONE handling, no hand-edit guard and no section fingerprint, while the tip carries 10+ `TOMBSTONE.json` withdrawn runs and `Benchmarks & Eval/LEADERBOARD.md` has 0 `SECTION-SHA256` fingerprints. A rebuild from the tip's code would re-publish withdrawn runs and could silently undo a retraction (E: RESULT section 25). ⚠️ **Coupling:** the chunked landing is bringing `test_leaderboard_hand_edit_guard.py` / `test_leaderboard_yawmask_input.py` / `test_leaderboard_navtest_single_stage_ours.py` (they were pending at chunk 2); they pin the E: behaviour and will be RED against the tip's builder unless the leaderboard patch lands with them. I did not run them (UNVERIFIED).
* **Section C: the audit's 1,073 'absent from the tip' manifest entries are mostly NOT absent.** After cleaning the audit parser's artefacts (md5/comment text glued onto the path): 524 + 7 were never missing; 469 were landed by chunks 1-3 of the running landing; 29 are content-moved; **10 still pending (chunk 4)**; 20 optional 5-20 MB; 10 held for scrub; 4 absent everywhere (2 reel .mp4 = BULK media, 2 prose lines).

## A. The 29 S4 package-local code variants

Method: for each variant, (1) blob id vs the tip target and vs EVERY historical blob of the target (`git log --raw -m`; none of the 29 is byte-equal to the tip or to any historical version); (2) the closest historical blob by line-set distance is taken as the variant's base, `git diff --histogram -U0` base -> variant is cut into hunks, and each hunk's added block is searched in the tip text (APPLIED), its removed block likewise (MISSING = tip still has the old block and lacks the new; DEL-MISSING = the variant DELETES lines the tip still has, i.e. the variant is the older stage); (3) the package's own docs and the E: mtime vs the superseding commit decide SUPERSEDED. Hunk tables: `raw/A_hunk_analysis.json`. 'hunks' below = counts by status. READ THE STATUS AGAINST THE DIRECTION: in the `fix_P7` and box-head `fix` rows a MISSING / DEL-MISSING hunk means the TIP holds the later (L1/L2/HQS) form of a line that the OLDER variant lacks or holds in its earlier form; only the v7F rows are variants that are NEWER than the tip file.

| # | variant (package / dir) | target | verdict | hunks | evidence |
|---:|---|---|---|---|---|
| 0 | 2026-10-04-x10-pose-timing / fix | `scripts/refc_v3_train.py` | **APPLIED** | APPLIED 7 | 7/7 hunks of diff(c39798d -> variant) are on the tip (X10 --pose-sync-sidecar wiring); landed by the WP-B landings L1/L2 (badb582, a5c2a9b). |
| 1 | 2026-10-04-x10-pose-timing / fix | `tests/test_declared_vs_built.py` | **SUPERSEDED** | CHANGED 1 | only change is the REGISTRY pin 256 -> 257 (+X10 comment); the tip's pin has moved on (a5c2a9b: 264+) and carries the X10 comment lines. |
| 2 | 2026-09-27-refcv7-box-head / fix | `scripts/refc_v3_train.py` | **SUPERSEDED** | DEL-MISSING 2, MISSING 1 | pre-HQS 'MAIN' box-head variant: it LACKS the A14 HQS lines the tip carries (--slot-query-select, its G-DVB entry, REGISTRY pin 221); the tip took the ANCHORED/HQS landing (commit 28d8365 'refcv7 box head lands (A9-A14.1)'). Variant mtime 2026-09-27 12:50 < 15:26. |
| 3 | 2026-09-27-refcv7-box-head / fix | `tanitad/models/refcv6_perception_branch.py` | **APPLIED** | APPLIED 9 | 9/9 hunks of diff(cef9709 -> variant) are on the tip (A14.1 box-head perception branch), landed by 28d8365. |
| 4 | 2026-09-27-refcv7-box-head / fix | `tanitad/train/declared_vs_built.py` | **SUPERSEDED** | DEL-MISSING 1 | pre-HQS 'MAIN' box-head variant: it LACKS the A14 HQS lines the tip carries (--slot-query-select, its G-DVB entry, REGISTRY pin 221); the tip took the ANCHORED/HQS landing (commit 28d8365 'refcv7 box head lands (A9-A14.1)'). Variant mtime 2026-09-27 12:50 < 15:26. |
| 5 | 2026-09-27-refcv7-box-head / fix | `tests/test_declared_vs_built.py` | **SUPERSEDED** | CHANGED 1 | pre-HQS 'MAIN' box-head variant: it LACKS the A14 HQS lines the tip carries (--slot-query-select, its G-DVB entry, REGISTRY pin 221); the tip took the ANCHORED/HQS landing (commit 28d8365 'refcv7 box head lands (A9-A14.1)'). Variant mtime 2026-09-27 12:50 < 15:26. |
| 6 | 2026-09-27-refcv7-box-head / fix | `tests/test_occ_knob_is_stamped.py` | **SUPERSEDED** | DEL-MISSING 1 | pre-HQS 'MAIN' box-head variant: it LACKS the A14 HQS lines the tip carries (--slot-query-select, its G-DVB entry, REGISTRY pin 221); the tip took the ANCHORED/HQS landing (commit 28d8365 'refcv7 box head lands (A9-A14.1)'). Variant mtime 2026-09-27 12:50 < 15:26. |
| 7 | 2026-09-27-refcv7-box-head / fix_r6 | `scripts/refc_v3_train.py` | **OBSOLETE** | MISSING 4 | variant HELD BY DECISION: +R6 (A10.1) denoising-query groups behind --slot-dn-groups (default 0). The tip's refcv7-r101-s0.argv.json `why` says "+R6 ... is NOT adopted"; BUILD.md (box-head) says R6 was "deprioritised by A14.1"; LANDING_READY_R6.txt says "Use ONLY if a SPEC amendment makes +R6 the launch configuration". No live chain passes the flag. |
| 8 | 2026-09-27-refcv7-box-head / fix_r6 | `tanitad/models/refcv6_perception_branch.py` | **OBSOLETE** | APPLIED 6, ADD-MISSING 6 | 6 hunks APPLIED (the A14.1 branch, on the tip); the other 6 hunks are the R6-only DN-embedder lines - HELD BY DECISION, see the fix_r6 rows. |
| 9 | 2026-09-27-refcv7-box-head / fix_r6 | `tanitad/train/declared_vs_built.py` | **OBSOLETE** | MISSING 4 | variant HELD BY DECISION: +R6 (A10.1) denoising-query groups behind --slot-dn-groups (default 0). The tip's refcv7-r101-s0.argv.json `why` says "+R6 ... is NOT adopted"; BUILD.md (box-head) says R6 was "deprioritised by A14.1"; LANDING_READY_R6.txt says "Use ONLY if a SPEC amendment makes +R6 the launch configuration". No live chain passes the flag. |
| 10 | 2026-09-27-refcv7-box-head / fix_r6 | `tests/test_cls_weight_stamp.py` | **OBSOLETE** | MISSING 1 | variant HELD BY DECISION: +R6 (A10.1) denoising-query groups behind --slot-dn-groups (default 0). The tip's refcv7-r101-s0.argv.json `why` says "+R6 ... is NOT adopted"; BUILD.md (box-head) says R6 was "deprioritised by A14.1"; LANDING_READY_R6.txt says "Use ONLY if a SPEC amendment makes +R6 the launch configuration". No live chain passes the flag. |
| 11 | 2026-09-27-refcv7-box-head / fix_r6 | `tests/test_declared_vs_built.py` | **OBSOLETE** | MISSING 1 | variant HELD BY DECISION: +R6 (A10.1) denoising-query groups behind --slot-dn-groups (default 0). The tip's refcv7-r101-s0.argv.json `why` says "+R6 ... is NOT adopted"; BUILD.md (box-head) says R6 was "deprioritised by A14.1"; LANDING_READY_R6.txt says "Use ONLY if a SPEC amendment makes +R6 the launch configuration". No live chain passes the flag. |
| 12 | 2026-09-27-refcv7-box-head / fix_r6 | `tests/test_occ_knob_is_stamped.py` | **OBSOLETE** | MISSING 1 | variant HELD BY DECISION: +R6 (A10.1) denoising-query groups behind --slot-dn-groups (default 0). The tip's refcv7-r101-s0.argv.json `why` says "+R6 ... is NOT adopted"; BUILD.md (box-head) says R6 was "deprioritised by A14.1"; LANDING_READY_R6.txt says "Use ONLY if a SPEC amendment makes +R6 the launch configuration". No live chain passes the flag. |
| 13 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `ops/runs.d/refcv8-wpb-smoke.argv.json` | **SUPERSEDED** | CHANGED 1, DEL-APPLIED 2, DEL-MISSING 1, BOTH-PRESENT 1 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 14 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `scripts/launch_gate.py` | **SUPERSEDED** | MISSING 2, CHANGED 1, DEL-MISSING 2, DEL-APPLIED 2 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 15 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `scripts/refc_v3_train.py` | **SUPERSEDED** | CHANGED 2, DEL-MISSING 17, DEL-APPLIED 1, MISSING 3 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 16 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tanitad/eval/refcv7_loader.py` | **SUPERSEDED** | DEL-MISSING 2 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 17 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tanitad/eval/speed_leak.py` | **SUPERSEDED** | DEL-MISSING 2 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 18 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tanitad/refs/refc.py` | **SUPERSEDED** | DEL-APPLIED 1, MISSING 2 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 19 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tanitad/refs/refc_v3.py` | **SUPERSEDED** | ADD-MISSING 1 | the X3 past-only ceiling lines are on the tip at refc_v3.py:1930-1943 in a later form (condition widened to 'also a speed-only arm (V0)', enc8 branch added). |
| 20 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tanitad/refs/refcv8_conditioning.py` | **APPLIED** | APPLIED 1 | 1/1 hunk of diff(c39798d -> variant) is on the tip. |
| 21 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tanitad/train/refcv8_train.py` | **SUPERSEDED** | CHANGED 2, MISSING 5, DEL-APPLIED 3, DEL-MISSING 2 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 22 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tests/test_declared_vs_built.py` | **SUPERSEDED** | CHANGED 1 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 23 | 2026-10-04-refcv8-tactical-conditi / fix_P7 | `tests/test_refcv8_speed_input.py` | **SUPERSEDED** | DEL-MISSING 1 | fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form. |
| 24 | 2026-09-27-v7f-refav1-stat / v7f_r1r4 | `scripts/train_v6_staged.py` | **MISSING** | ADD-MISSING 14, MISSING 3 | v7F R1/R3/R4 stream copy; SUPERSEDED BY the unlanded v7f_merge change set (itself NOT in stack/). The tip's train_v6_staged.py/v6.py are byte-unchanged since 2026-09-06 (3886f7b / 982476c): every hunk is absent from the tip. |
| 25 | 2026-09-27-v7f-refav1-stat / v7f_r1r4 | `tanitad/models/v6.py` | **MISSING** | ADD-MISSING 16, MISSING 3 | v7F R1/R3/R4 stream copy; SUPERSEDED BY the unlanded v7f_merge change set (itself NOT in stack/). The tip's train_v6_staged.py/v6.py are byte-unchanged since 2026-09-06 (3886f7b / 982476c): every hunk is absent from the tip. |
| 26 | 2026-09-27-v7f-refav1-stat / v7f_r1r4 | `tanitad/train/declared_vs_built_v6.py` | **MISSING** | (new file) | v7F R1/R3/R4 stream copy; SUPERSEDED BY the unlanded v7f_merge change set (itself NOT in stack/). The tip's train_v6_staged.py/v6.py are byte-unchanged since 2026-09-06 (3886f7b / 982476c): every hunk is absent from the tip. |
| 27 | 2026-09-27-v7f-refav1-stat / v7f_r3 | `scripts/train_v6_staged.py` | **MISSING** | ADD-MISSING 23, MISSING 7 | v7F R1/R3/R4 stream copy; SUPERSEDED BY the unlanded v7f_merge change set (itself NOT in stack/). The tip's train_v6_staged.py/v6.py are byte-unchanged since 2026-09-06 (3886f7b / 982476c): every hunk is absent from the tip. |
| 28 | 2026-09-27-v7f-refav1-stat / v7f_r3 | `tanitad/train/declared_vs_built_v6.py` | **MISSING** | (new file) | v7F R1/R3/R4 stream copy; SUPERSEDED BY the unlanded v7f_merge change set (itself NOT in stack/). The tip's train_v6_staged.py/v6.py are byte-unchanged since 2026-09-06 (3886f7b / 982476c): every hunk is absent from the tip. |

### The 5 MISSING items (v7F) - what they fix, what runs them, patch

* #24 `stack/scripts/train_v6_staged.py`: R1 (--max-speed-input-v6 / --plan-vmax-cap) + R4 (--tac-op-cond: the tactical BEHAVIOURS a_lat x a_lon reach the operative plan; MEASURED swap moves plan by exactly 0.0) - package v7f_r1r4/RESULT.md
* #25 `stack/tanitad/models/v6.py`: same stream: v6.py model side of R1/R4
* #26 `stack/tanitad/train/declared_vs_built_v6.py`: same stream: NEW declared-vs-built registry for the v6 trainer (new file; incompatible reader signature with the R3 file of the same name)
* #27 `stack/scripts/train_v6_staged.py`: R3 (PI binding): all tactical labels reach the tactical loss - at the tip NO tactical label family reaches ANY v7F loss, traffic-light colours included (D-TLIGHT-1 for v7F) - package v7f_r3/RESULT.md
* #28 `stack/tanitad/train/declared_vs_built_v6.py`: same stream: NEW declared-vs-built registry (v7f_merge holds the UNION of the two)

* **Superseded BY an unlanded set:** `v7f_merge` (package `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_merge/`) is the COMPLETE v7F set (R1+R3+R4+R2 nav fixes+R6 `--strategic-off`) and its RESULT says it supersedes every `code/fix/` file of `v7f_r1r4/` and `v7f_r3/`. It is not on the tip's `stack/` either: tip `train_v6_staged.py` last changed 3886f7b (2026-09-06), `v6.py` 982476c (2026-09-06). The tip blob equals the merge's declared base, so the merge applies cleanly.
* **What the merge fixes (INHERITED from the package RESULT, not re-verified by me):** a real `--nav-cond` v7F run could not START at the tip (three defects: `synthetic_batch` carries no nav keys, `nav_token` overwritten by `None` at step 1, the NavConditioner's 12 parameters belong to no `_GROUP_PREFIXES` group); no tactical label family reaches any v7F loss (traffic-light colours included, `D-TLIGHT-1` for v7F); the tactical behaviours do not reach the operative plan (R4); no hard speed cap (R1). The package reports 178 tests passed on the merge and 2,655 passed / 8 failed / 24 errors on the 107-file v6 suite with the failure set identical to its baseline (INHERITED).
* **Live use:** none (see headline). No `runs.d` argv, no gate profile.
* **Proposed patch (NOT applied):** `patches/v7f_merge_R1R6.patch` = tip -> `v7f_merge` for `train_v6_staged.py`, `models/v6.py`, `models/plan_speed_cap.py` (new), `train/declared_vs_built_v6.py` (new) and six test files; `git apply --check` passes on the tip's exact bytes and the result is byte-identical to the package files (`patches/v7f_merge_R1R6.json`); all `.py` files compile. **Tests were not run** (needs the full tree + torch). Tests that name the targets (MEASURED, `raw/tests_naming_the_targets.json`): 62 files name `train_v6_staged`, 63 import `tanitad.models.v6`, 0 name `declared_vs_built_v6`. ⚠️ `models/v6.py` is imported by data modules shared with other programmes (`bev_raster.py`, `tactical_goals.py`, `g_tac_geom.py`, `anchor_goal.py`, `refav1_loader.py`...): run the full suite before landing; the package's 'bit-identical with every new flag off' claim is INHERITED.

## A+. Extension: every declared `package path -> repo path` mapping not applied (beyond the 29)

Source: all 74 `LANDING_READY*.txt` in the E: tree, 5 families below. Mapping statuses (MEASURED, `raw/X_manifest_mappings.json`): {'NOT-A-MAPPING': 33, 'APPLIED-THEN-SUPERSEDED': 66, 'APPLIED-EXACT': 61, 'DIFFERS-NOT-IN-HISTORY': 21, 'NEW-NOT-ON-TIP': 18}. 'tip == declared base' means the tip still holds the exact bytes the variant was built from, i.e. the variant was NEVER applied.

| family | repo targets not applied | tip == declared base | live? | disposition |
|---|---|---|---|---|
| refcv7 box-head R6 (held) | 13 (2 new files) | 1 | not adopted | OBSOLETE: held by decision (argv `why`, BUILD.md, LANDING_READY_R6). |
| refav1 R1-VMAX | 3 (1 new files) | 0 | `refa_v1_plan.py` and `refav1_arm.py` are LIVE files; the cap is flag-off and no live run passes it | MISSING; 3-way merge CLEAN onto the tip (included in the refav1 patch). PI requirement R1 of 2026-09-27. |
| refav1 R1/R3/R5/R6 (refav1_r1r6) | 11 (8 new files) | 2 | refa_v1_train.py last changed 2026-09-03: dormant trainer; `refa_v1.py` is live (HJ-UNITS-2, 10-18) | MISSING (feature set, flags default OFF, 'bit-identical when off' INHERITED). `patches/refav1_R1R6_plus_R1VMAX.patch` is PARTIAL: `stack/tanitad/refs/refa_v1.py` has 1 merge conflict against the tip (HJ-UNITS-2) and is OMITTED; do not apply the patch without resolving it. |
| v7F launch-gate profile (v7f_gate) | 2 (1 new files) | 0 | dormant; `launch_gate.py` itself is live (edited 3766dcd, 2026-10-17) | MISSING; `patches/v7f_gate_profile.patch` 3-way merges CLEAN onto the tip's launch_gate.py (base = declared). Needs v7f_merge first (its checks call `declared_vs_built_v6.check_all`). |
| v7F trainer (v7f_merge) | 10 (6 new files) | 4 | dormant (see headline) | = section A MISSING; `patches/v7f_merge_R1R6.patch` |

(Per-file list: `triage.json` -> `X_unapplied_families`.)

## B. The 47 S2b candidate edits

Method: E blob vs every historical blob of the path on the tip (none matches), E-only lines (E lines absent from the tip file), the closest-history-base hunk analysis (`raw/B_candidate_edits.json`), and a single-pass `git grep -F -f` of every E-only line of 28+ chars against the WHOLE tip (`raw/B_E_only_lines_found_on_tip.json`: does the content live under another path?). Registers: report only, lead-ID check (`raw/REGISTER_lead_ids.json`); E-only lines of the two registers flagged for public IP/port are NOT exported. mtime is used only as a secondary hint.

| # | path | verdict | E-only / tip-only | evidence |
|---:|---|---|---|---|
| 0 | `.claude/settings.local.json` | **N/A** | 3 / 0 | MACHINE-LOCAL permission file (3 lines); never landed. |
| 1 | `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/COMMS.md` | **SUPERSEDED** | 0 / 24 | E is a strict line-subset of the tip (tip only adds 24 lines). |
| 2 | `EF/2026-09-19-navhard-download/raw/navhard_reference_chain.log` | **SUPERSEDED** | 4 / 0 | E is an older raw-output draft (E mtime precedes the tip's landing commit); the tip holds the later value. |
| 3 | `EF/2026-09-19-navsim-refcv4b-bridge/raw/K7_packer_vs_trainer.json` | **SUPERSEDED** | 6 / 6 | E is an older raw-output draft (E mtime precedes the tip's landing commit); the tip holds the later value. |
| 4 | `warmup-reference-epdms/raw/navhard/mirror/data_mirror_verify_navhard_logs.json` | **SUPERSEDED** | 1 / 1 | E is an older raw-output draft (E mtime precedes the tip's landing commit); the tip holds the later value. |
| 5 | `EF/2026-09-20-navhard-stage1-extract/raw/receipts/shard_6.json` | **SUPERSEDED** | 3 / 3 | E is an older raw-output draft (E mtime precedes the tip's landing commit); the tip holds the later value. |
| 6 | `EF/2026-09-23-refcv6-standard-tests/battery/SPEC.md` | **MISSING** | 33 / 0 | doc: AMENDMENT A6 (pre-registered 2026-09-26 ~14:40, 'written BEFORE any A6 number existed') - tip's last change is 14:17; 29 of 29 long E-only lines absent from the whole tip. Pre-registration chronology risk: land from E with its timestamp. |
| 7 | `EF/2026-09-26-navtest-single-stage/RESULT.md` | **MISSING** | 5 / 0 | doc: 'E1 RESOLVED 2026-09-26' paragraph (+5) absent from the tip. |
| 8 | `EF/2026-09-26-suite-runnability-audit/LANDING_READY.txt` | **MISSING** | 43 / 0 | manifest additions (+43) describing the leaderboard fix; entries partly pending in the chunked landing. |
| 9 | `EF/2026-09-26-suite-runnability-audit/RESULT.md` | **MISSING** | 98 / 0 | doc: RESULT section 25 'the leaderboard READ runs it never SHOWED, would have re-published four withdrawn runs, and could silently undo a retraction' (+98) - the write-up of the leaderboard code fix below. |
| 10 | `EF/2026-09-28-refcv7-standard-tests/battery/LANDING_READY.txt` | **MISSING** | 18 / 0 | manifest additions (+18). |
| 11 | `/2026-09-28-refcv7-standard-tests/navsim/raw/controls/KD_exact_dedup_cuda.json` | **MISSING** | 2 / 2 | raw JSON: 2 value lines differ (max_abs_traj_diff_m, ckpt path); E newer by 7 days. UNVERIFIED which run is canonical. |
| 12 | `Project Steering/BACKLOG.md` | **REGISTER** | 2 / 199 | report only. 2 E-only lines (row E10); lead ID is on the tip. |
| 13 | `Project Steering/GOALS_AND_CLAIMS.md` | **REGISTER** | 23 / 4530 | report only. 23 E-only lines; all 15 lead-IDs found are on the tip (rows exist; E carries in-row wording). |
| 14 | `Project Steering/MODEL_REGISTRY.md` | **REGISTER** | 10 / 477 | report only. 10 E-only lines (flagged public_ip/port: not exported); lead IDs on the tip. |
| 15 | `Project Steering/PI_DECISION_QUEUE.md` | **REGISTER** | 0 / 579 | report only. E is a strict subset of the tip (0 E-only lines). |
| 16 | `Project Steering/RETRACTION_LOG.md` | **REGISTER** | 3 / 1248 | report only. 3 E-only lines (flagged: not exported); lead ID on the tip. |
| 17 | `Project Steering/SPEC_REFCV8_DRAFT.md` | **SUPERSEDED** | 196 / 763 | E is the 2026-10-04 draft; the tip rewrote the file through d014ff2 (2026-10-12); 763 tip-only lines. |
| 18 | `Project Steering/TANITAD_PROGRAMME.md` | **MISSING** | 9 / 0 | constitution file, 9 lines: Colab Pro capacity policy (PI 2026-09-27). Tip unchanged since 2026-08-23. Master-Mind-owned - land only by the Master Mind. |
| 19 | `AI/R/2026-09-19-a7-imagenet-knockout/RESULT.md` | **MISSING** | 9 / 3 | doc: status banner 'PAUSED - NO DATA YET' (+9). |
| 20 | `AI/R/2026-09-19-s1-collision-gate/RESULT.md` | **MISSING** | 8 / 3 | doc: status banner 'DONE - verdict landed 2026-09-20' (+8). |
| 21 | `AI/R/2026-09-19-s1-collision-gate/raw/mutation_proof_p_runner.json` | **MISSING** | 43 / 27 | raw: a re-run of the P-runner mutation proof written 1h40 after the landing (n_passed 32/25/32/32/30 recorded). |
| 22 | `AI/R/2026-09-20-refe-plan/eval/PREREG_AUX_DRIVABLE_S1.md` | **SUPERSEDED** | 1 / 143 | E (2026-10-15 06:29) predates the tip's rewrite 421ffc3 (10-15 18:12). |
| 23 | `AI/R/2026-09-20-refe-plan/eval/S1_POD_RUNBOOK.md` | **SUPERSEDED** | 1 / 1 | 1-line difference; tip d257647 is later. |
| 24 | `AI/R/2026-09-20-refe-plan/refe/scorer_finetune.py` | **SUPERSEDED** | 11 / 363 | refe/ P6-lineage code: E (2026-10-15 06:xx) predates the tip's later commits (c2d4042 10-18 09:11 / c968f99 / e2d4422); tip-only lines 363/10/7. LIVE file family, but the tip is the newer. |
| 25 | `AI/R/2026-09-20-refe-plan/refe/test_aux_g12_bf16.py` | **SUPERSEDED** | 4 / 10 | refe/ P6-lineage code: E (2026-10-15 06:xx) predates the tip's later commits (c2d4042 10-18 09:11 / c968f99 / e2d4422); tip-only lines 363/10/7. LIVE file family, but the tip is the newer. |
| 26 | `AI/R/2026-09-20-refe-plan/refe/test_aux_s1_wiring.py` | **SUPERSEDED** | 2 / 7 | refe/ P6-lineage code: E (2026-10-15 06:xx) predates the tip's later commits (c2d4042 10-18 09:11 / c968f99 / e2d4422); tip-only lines 363/10/7. LIVE file family, but the tip is the newer. |
| 27 | `AI/R/2026-10-04-refcv8-perception-architecture/PERCEPTION_DESIGN.md` | **MISSING** | 23 / 0 | doc: appended execution log / design table written after the tip's landing 49a0655 / 2081f5d (E-only 23 / 137 / 29 lines; <=1 found elsewhere on tip). |
| 28 | `AI/R/2026-10-04-refcv8-perception-architecture/RESULT.md` | **MISSING** | 143 / 0 | doc: appended execution log / design table written after the tip's landing 49a0655 / 2081f5d (E-only 23 / 137 / 29 lines; <=1 found elsewhere on tip). |
| 29 | `AI/R/2026-10-04-refcv8-perception-architecture/code/pgrad.py` | **MISSING** | 42 / 4 | package-local probe pgrad.py: TF32 OFF (the registered linearity control FAILED with TF32 on: 6.2e-4 vs 1e-4) + the --batch-start/--batch-count PARTS; tip's instrument lacks both. No live chain. |
| 30 | `AI/R/2026-10-04-refcv8-r1-head-only/DESIGN.md` | **MISSING** | 29 / 6 | doc: appended execution log / design table written after the tip's landing 49a0655 / 2081f5d (E-only 23 / 137 / 29 lines; <=1 found elsewhere on tip). |
| 31 | `AI/R/2026-10-04-refcv8-tactical-conditioning/LANDING_READY_WPB.txt` | **MISSING** | 120 / 19 | manifest (E 125 entries, +120 lines vs tip); 66 of its listed paths are parse artifacts of the audit (section C). |
| 32 | `AI/R/2026-10-04-refcv8-tactical-conditioning/raw/d0_dose_response.json` | **MISSING** | 1 / 1 | raw: 1 line (wall_s 50.4). |
| 33 | `TanitAD Research Lab/Library/LIBRARY.md` | **SUPERSEDED** | 114 / 272 | registers: E has NO entry key the tip lacks (E 624 keys, tip 639); the 31 changed entries differ only in cited_by/tags/note wording. Report only. |
| 34 | `TanitAD Research Lab/Library/library.json` | **SUPERSEDED** | 68 / 535 | same as LIBRARY.md - generated from library.json by --reindex; never hand-edit. |
| 35 | `colab/MCP_SETUP.md` | **MISSING** | 5 / 0 | doc: 'APPLIED 2026-09-27 on the PI's instruction (register mcp server)' note (+5). Tip unchanged since 2026-08-16. |
| 36 | `data/nuplan_cam_calib.json` | **SUPERSEDED** | 65 / 4 | E (2026-09-21 00:10) predates the tip's 24d5ce2 (09-26). |
| 37 | `stack/tests/test_map_head_hires.py` | **SUPERSEDED** | 9 / 365 | E (2026-09-27 02:29) predates the tip's later commits (cef9709 05:06 / 1b8170e 12:42); the tip is the later map_hires version. |
| 38 | `stack/tests/test_semantic_map_gt_fine.py` | **SUPERSEDED** | 2 / 3 | E (2026-09-27 02:29) predates the tip's later commits (cef9709 05:06 / 1b8170e 12:42); the tip is the later map_hires version. |
| 39 | `taniteval/taniteval/leaderboard/__main__.py` | **MISSING** | 3 / 1 | LEADERBOARD CODE: adds --discard-hand-edits. Patch: patches/leaderboard_hand_edit_guard_and_withdrawn_runs.patch. |
| 40 | `taniteval/taniteval/leaderboard/build.py` | **MISSING** | 53 / 5 | LEADERBOARD CODE: TOMBSTONE (withdrawn-run) handling, hand-edit/fingerprint guard (section_state, dropped_lines), discard_hand_edits. |
| 41 | `taniteval/taniteval/leaderboard/leaderboard_sources.json` | **MISSING** | 1 / 1 | LEADERBOARD CODE: registers the refcv3-40284 yawmask ARM json as a source (file itself is in the pending landing set). |
| 42 | `taniteval/taniteval/leaderboard/render.py` | **MISSING** | 108 / 14 | LEADERBOARD CODE: section fingerprint (FP_PREFIX, section_fingerprint), withdrawn-run table, nuScenes runs shown. |
| 43 | `taniteval/taniteval/leaderboard/sources.py` | **MISSING** | 32 / 4 | LEADERBOARD CODE: load_bench_runs reads the TOMBSTONE marker; legacy-supersession map excludes withdrawn runs. |
| 44 | `taniteval/taniteval/map_hires_metrics.py` | **SUPERSEDED** | 12 / 36 | E (2026-09-27 02:29) predates the tip's later commits (cef9709 05:06 / 1b8170e 12:42); the tip is the later map_hires version. |
| 45 | `taniteval/tests/test_leaderboard.py` | **MISSING** | 6 / 6 | LEADERBOARD TEST: run_build(..., discard_hand_edits=True) in 6 call sites. |
| 46 | `taniteval/tests/test_map_hires_metrics.py` | **SUPERSEDED** | 2 / 46 | E (2026-09-27 02:29) predates the tip's later commits (cef9709 05:06 / 1b8170e 12:42); the tip is the later map_hires version. |

Patch for the leaderboard code (the only MISSING group that is running code): `patches/leaderboard_hand_edit_guard_and_withdrawn_runs.patch` (tip -> E: for 6 files; apply-check passes; byte-identical). Tests naming it: 2 (`taniteval/tests/test_leaderboard.py` + the fixtures). A run of `test_leaderboard.py` on a partial export failed identically before the patch on missing input files (UNVERIFIED either way; needs the full tree).

## C. LANDING_READY entries absent from the tip (audit section 7), after the chunked landing

Re-checked at `88fed187` (chunks 1-3 of 4 landed; chunk 4 pending at the time of writing). The audit's `not_on_tip_paths` (1,073 entries over 40 manifests) were re-parsed: its parser glued md5 / comment text onto many paths. Classes per entry (MEASURED, `raw/C_landing_ready_absent.json`):

| class | n |
|---|---:|
| PARSE-ARTIFACT:FILE-ON-TIP | 524 |
| LANDED-SINCE-AUDIT | 469 |
| MOVED | 29 |
| OPTIONAL-P3 | 20 |
| PENDING-LANDING | 10 |
| HOLD-SCRUB | 10 |
| PARSE-ARTIFACT:DIR-ON-TIP | 7 |
| ABSENT-EVERYWHERE | 4 |

Per manifest (only manifests with a non-artifact, non-landed remainder; the two big ones first):

| manifest | at audit | landed since | parse artifact (never missing) | remaining (class: n) |
|---|---:|---:|---:|---|
| `2026-10-04-navsim-thor-backend/LANDING_READY_NAVSIM_THOR.txt` | 317 | 300 | 0 | OPTIONAL-P3: 12, MOVED: 5 |
| `tecture & Inference/Research/2026-09-28-refcv7-restart-options/LANDING_READY.txt` | 215 | 0 | 215 | - |
| `2026-09-28-refcv7-standard-tests/navsim/LANDING_READY_NAVSIM50K_LEGAL.txt` | 173 | 155 | 0 | MOVED: 10, OPTIONAL-P3: 8 |
| `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/LANDING_READY.txt` | 45 | 12 | 26 | PENDING-LANDING: 1, MOVED: 6 |
| `rch Lab/Architecture & Inference/Research/2026-09-20-refe-plan/LANDING_READY.txt` | 23 | 0 | 13 | HOLD-SCRUB: 10 |
| ` Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/LANDING_READY_D4.txt` | 15 | 0 | 14 | PENDING-LANDING: 1 |
| `rchitecture & Inference/Research/2026-09-26-refcv7-launch-gate/LANDING_READY.txt` | 10 | 0 | 9 | MOVED: 1 |
| `Media/refcv7/LANDING_READY.txt` | 9 | 0 | 6 | ABSENT-EVERYWHERE: 3 |
| `2026-09-26-suite-runnability-audit/LANDING_READY.txt` | 5 | 0 | 0 | PENDING-LANDING: 5 |
| `2026-09-23-refcv6-standard-tests/battery/LANDING_READY.txt` | 3 | 0 | 0 | MOVED: 3 |
| `arch/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/LANDING_READY_P4.txt` | 3 | 0 | 0 | PENDING-LANDING: 3 |
| `/Architecture & Inference/Research/2026-09-26-map-signal-audit/LANDING_READY.txt` | 2 | 0 | 0 | MOVED: 2 |
| ` & Inference/Research/2026-10-04-refcv7-ddv2-rl-posttrain/LANDING_READY_WPRL.txt` | 2 | 0 | 1 | ABSENT-EVERYWHERE: 1 |
| `2026-09-23-refcv6-standard-tests/navsim/LANDING_READY.txt` | 1 | 0 | 0 | MOVED: 1 |
| `ab/Architecture & Inference/Research/2026-09-26-box-head-audit/LANDING_READY.txt` | 1 | 0 | 0 | MOVED: 1 |

* NAVSIM50K_LEGAL (173 at audit) and NAVSIM_THOR (317) are the md5-format manifests and were NOT parse artifacts: they were absent for real and are being landed by the chunks (see 'landed since').
* 'MOVED' = the file's CONTENT is on the tip (or in history) under another path (audit classes S4m/S4h); 'OPTIONAL-P3' = 5-20 MB raw data in `paths_P3_optional.json`; 'HOLD-SCRUB' = redacted copy listed in `paths_scrubbed.json`.
* ABSENT-EVERYWHERE (4): the reel + preview `.mp4` (155.9 MB and 13.0 MB; BULK media, not on E:) and two prose lines.

## Patches (all against the tip, NONE applied; `patches/*.json` carries the per-file tip blob ids)

| patch | files | `git apply --check` on tip bytes | byte-identical to source | notes |
|---|---:|---|---|---|
| `patches/v7f_merge_R1R6.patch` | 10 | True | True | complete v7F set; 4 existing targets + 6 new |
| `patches/refav1_R1R6_plus_R1VMAX.patch` | 14 | True | True | PARTIAL: `refa_v1.py` omitted (1 conflict). R1-VMAX files merged 3-way from the declared bases |
| `patches/v7f_gate_profile.patch` | 2 | True | True | adds the `v7f` launch-gate profile + test; depends on v7f_merge |
| `patches/leaderboard_hand_edit_guard_and_withdrawn_runs.patch` | 6 | True | True | tip -> E: copy of 6 files |

## Caveats, stated plainly

* No test was run on any patch (the suite needs the full tree and torch); the claims 'bit-identical with flags off' and the package test counts are **INHERITED**.
* A 'hunk' classification is textual. SUPERSEDED rests on (a) the variant lacking lines the tip has, (b) E: mtimes preceding the superseding landing commit, (c) the package's own statements. Where only (b) supported it I say so; the A rows with 'CHANGED' hunks were each read by hand.
* The audit's `S2b` verdicts were taken as given for set membership; B re-derives the content verdict. mtimes on a live working tree are a hint only.
* The brief's 'P6 runs refe/ at bcc04b5' is INHERITED; I verified only that `refe/` at the tip is import-independent of the 29 targets.
* The register E-only lines were NOT exported (two are flagged for public IP/port); all lead IDs found are already on the tip.
