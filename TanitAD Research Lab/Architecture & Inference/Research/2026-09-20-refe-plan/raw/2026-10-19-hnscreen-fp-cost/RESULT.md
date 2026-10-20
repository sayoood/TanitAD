# RESULT -- HNSCREEN-FP-COST: option (b') for the HN screen's `--ctx-cache` content check

Agent: Sonnet subagent (hnfp3). Base: mirror tip `5ad79d09eec9dc806bace47642f559f83ce5f656` (`agent/arch-inf-20260803`, `TIP_SHA.txt`). Implements `MM_RULING_FPCOST.md` (copied beside this file).
Nothing was committed, pushed, `git add`-ed, or written to a register. CPU only (`CUDA_VISIBLE_DEVICES=-1`). No pod, no GPU lock.

## HEADLINE

**INTEGRATION NEEDED.** (b') is implemented, tested at three levels (gate, chain, reader) and measured: a 3-seed screen reads the cache in full **4 times instead of 20** (MEASURED, section 2). Content mode (`--ctx-mode content`) reproduces the tip
**bit for bit** (R7, MEASURED: 83 of 83 observables identical, and a control that must fail does fail). The files below are staged in `out/` at their real repo paths; **`launch_gate.py`, `hnscreen_devbox.py` and `gate_token.py` are hot files**
(other agents edit them): land by `base=<blob>` against the live tip, never wholesale.

* ⚠️ **The ruling's "17" is an undercount: it is 20** (MEASURED). The saving is 20 -> 4 (5x), not 17 -> 4. Source of the 17: hnfp2's COST.md counts `run` as 2 reads (it is `run_checks` 2 + `finalize` 2 = 4) and one verify per seed (the chain verifies twice per seed: at the mint and at the start of the pass).
* **Finish pass (2026-10-20; `FINISH_NOTES.md`): everything that was pending at hand-off has now run.** Gate matrix **ZZTEST_PASS 72** (38 arms + 34 RED mutants, run3, on the final tree); chain matrix **19** (9 + 10, run2); reader **36**; hnfp2's content-fp matrix **81** (62 + 19, run2) = the tip's 81;
  R7 differential 83/83 + control PASS. Sibling suites (section 6): 30 invocations, **29 rc 0 and 1 rc 1** (`test_rls1_queue.py`, which FAILS ON BOTH: a race in the test itself, pre-existing on the tip); NEW FAILURES none, LOWER COUNTS none. Two failures were looked at: (i) `test_hn_content_fp.py` was NEW as a suite failure (base rc 0) and was a stale
  mutant TABLE, not a production defect: six of its mutant snippets stopped being unique after HNFP3's edits (`_content_listing` itself is unchanged; HNFP3's `_stat_listing` copies its symlink statement and `fp_key` gained a branch); the table is re-anchored (test only), 81/81 again. (ii) `test_rls1_queue.py:184` is the test's own flock race (FINISH_NOTES 1).
* **VERDICT: ready for the Master Mind to land (no new failure, no lower count). UNVERIFIED:** (1) `test_aux_s1_wiring.py` (20), `test_gate_suite_prerun.py` (144) and `test_launch_gate.py` full form (1213) pass on MINE but their BASE run was skipped for lack of disk on C: (and I was told not to run them): NO base comparison exists for them;
  (2) the symlink arms patch `os.path.islink` because symlinks cannot exist on exFAT / D: (the code path is tested, a real symlinked directory is not); (3) the race diagnosis of `test_rls1_queue.py` rests on 3 probe runs (1 of 2 undelayed runs collided) and the file was not run on Linux; (4) two sibling files print no count token (`test_aux_g12_bf16.py`, `test_p6_levers.py`): "SAME" there means rc 0 on both and nothing finer;
  (5) section 7.6 (`gate.token_check_names` on the run box) is still open.

## 1. What (b') is, as built

| | |
|---|---|
| content reads | unchanged at the seal (`ctx_cache.finalize` -> `seal`), the fill (`hn_fingerprints.py`), the chain START (new `launch_gate.py anchor`) and the chain END (new `launch_gate.py end-verify`) |
| intermediate calls | `run` (run_checks + finalize), the mint `verify`, the start-of-pass `verify`, `refusals`: **0 bytes read**; compare `(relpath, size, mtime_ns)` of EVERY file (incl. `content_manifest.json`) against the anchor's signed stat listing. A size or mtime difference, a missing file, an extra file, a symlinked directory: REFUSAL |
| the anchor | HMAC-signed (gate key) `ctx_chain/anchor.json`: chain id, normalised cache dir, `minted_ns`, the registered pin, the stat listing. Minted only if ONE full content fingerprint equals the registered pin and the cache did not change while it was being read. Another cache dir, another chain, older than the chain start, edited, signed by another key: refused |
| the record of the mode | the token's `--ctx-cache` fingerprint: `kind: dir-anchor`, `mode: anchor-bound`, `anchor_sha256`; its key equals the registered `dirc:` key when clean (`dir-anchor-INVALID:...` otherwise); the verify record carries `info.ctx_cache`. **Content mode adds nothing** (no new field, same text) |
| END verify | a second full content fingerprint **and** a stat compare (bracketed by two stat listings); writes a signed `ctx_chain/end_verify.json`, `END_VERIFY_OK` or `VOID`. **VOID is permanent** (`ctx_chain/VOID.json`): a relaunch cannot launder a restored cache |
| the reader | `eval/hn_screen_read.py --chain-root <state dir> [--key-file]` (required): BEFORE any score file is opened, every pass named in `--scores` needs a signed PASS token whose `--ctx-cache` is content-bound, or anchor-bound with a signed `END_VERIFY_OK` bound to the SAME anchor, later than it, covering all those seeds. Else rc 6 `ZZHNSCREEN_REFUSED`, nothing read. A check that cannot run is a refusal |
| modes | `--ctx-mode content` is the gate default. The chain's default is `anchor-bound` (`gate.ctx_mode`); `content` forces today's chain. An anchor named under content mode is refused (never silently ignored); anchor-bound exists only for the `hn_rules` profile and needs `--anchor --chain-id --chain-start-ns` together |
| scope | only the HN screen's ctx cache. Every other data-manifest check of `launch_gate` is untouched |

Edit size vs the tip (`out/raw/code_vs_tip.diff`): `gate_token.py` +243/-10, `launch_gate.py` +120/-9, `hnscreen_devbox.py` +88/-8, `eval/hn_screen_read.py` +18/-1; tests: new `refe/test_hn_ctx_anchor.py` (960 lines), new `refe/test_hnscreen_ctx_chain.py` (415),
`eval/test_hn_screen_read.py` +144/-4, `refe/test_hnscreen_chain.py` +2/-1 (`"ctx_mode": "content"` in its gate config, so the pre-existing chain test still tests today's chain).

## 1b. The call map (`CALLMAP.md`, MEASURED by reading every hop at the tip, `file:line`)

Per seed there are **6** full reads: `run_checks` (data manifest 1 + `hnscreen_refusals` 1), `finalize` (1 + 1), the mint verify (1), the start-of-pass verify (1). Plus the seal and the fill. Probed and NOT re-verifiers (two paths each): the trainer
(`CtxCache(..., verify=False)`), G15 (a sample), G-DVB (`manifest.json` only), `Marks.is_done` for S2 (`manifest.json` only), the S4 tools (never open the cache), `ctx_cache.py check` (manual CLI).

## 2. Reads per screen (MEASURED: `logs/reads_per_screen.json`, script `scripts/measure_reads.py`, log `logs/measure_reads_FINAL.log`)

Instrument: `gate_token._sha256_and_size` counted (one file fully hashed = one read; one full pass = 10 reads on this 10-file synthetic cache built by `ctx_cache.finalize`); the REAL gate as a subprocess per call.
**Estimator question: a deterministic count of file reads on one synthetic cache. Not a timing, no sampling variance.**

| screen | today (content mode = the control) | (b') anchor-bound |
|---|---|---|
| 1 seed | 80 reads = 8 passes | 40 = **4 passes** |
| 2 seeds | 140 = 14 passes | 40 = **4 passes** |
| **3 seeds** | **200 = 20 passes** (3 x run 40 + 6 x verify 10 + seal 10 + fill 10) | **40 = 4 passes** (anchor 10, end-verify 10, seal 10, fill 10; the 11 intermediate gate calls read **0**) |

(b') is constant in the number of seeds. ESTIMATED time on the run box (ruling: 44 MB/s cold on D:, INHERITED from COST.md; cache ~69.6 GiB, INHERITED from PREREG_P6HN_SCREEN.md s3): 4 passes = 4 x 27 min = **~1.8 h**; today's real count 20 = **~9 h**
(the ruling's 17 = 7.7 h undercounted this).

## 3. R1-R7: the named arms and what they read (results: logs named in section 4)

| R | arm (gate level, `refe/test_hn_ctx_anchor.py`; chain level, `refe/test_hnscreen_ctx_chain.py`, prefix C_; reader, `eval/test_hn_screen_read.py`, prefix G_) |
|---|---|
| R1 | `R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate_refuses` (+ 2 controls: untouched chain ends OK; a restored file ends OK); chain: `C_R1_...voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it`; reader: `G_R1_a_VOID_chain_is_refused_before_any_score_is_read` |
| R2 | `R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end`; chain: `C_R2_...the_trainer_is_not_started` |
| R3 | `R3_an_mtime_ns_change_with_identical_bytes_refuses_and_restoring_it_matches_again`, `R3e_...voids_at_the_end_though_the_bytes_are_equal` |
| R4 | extra file, missing file, renamed file, symlinked directory (4 arms) |
| R5 | 9 gate arms (end verify skipped / deleted / VOID / foreign anchor / missing seed / two anchors / pass without token / listing-only token / edited token); chain: `C_R5_the_chains_own_read_command_is_refused_by_the_landed_reader_once_the_end_record_is_deleted`; reader: `G_R5_...` + 7 more `G_` arms |
| R6 | 7 arms: another cache dir, minted before the chain, another chain, edited / foreign-key anchor, an unregistered or tampered cache cannot be anchored, a self-consistent RESEALED cache is refused by the pin alone, a cache edited while being anchored |
| R7 | 6 gate arms (identity of the data manifest; default is content; anchor under content refused; anchor needs all three flags and only the HN profile; refusal text unchanged; cost 4 reads) + chain `C_R7_forcing_content_mode_is_todays_chain_...` + the **literal differential against the tip** (section 5) |
| W | 3 wiring arms (the "built, tested, unreachable from its caller" class of CLAUDE.md): spies prove `cmd_run`, `cmd_verify`, `cmd_refusals` and `hnscreen_refusals` hand THIS call's anchor down, and that content mode hands none |

Chain level, MEASURED once on the final tree (`logs/test_hnscreen_ctx_chain_arms_FINAL.log`, run before the disk filled): **9 of 9 chain arms green** (`ZZTEST_PASS 9`), with the real gate / real HMAC tokens / real sealed cache, stub trainer.
Gate level: **38 of 38 arms green** (`logs/test_hn_ctx_anchor_matrix_run1.log`: "38 arms green on the real tree"). Reader: `ZZTEST_PASS 36` (`logs/test_hn_screen_read_FINAL.log`: the unmutated suite incl. 8 new gate arms + 35 RED mutations, 5 of them new).

## 4. Mutation matrix (delete the single enforcing line, require RED)

<!--MUTANTS-->
Mutants listed: 68 (68 RED, 0 not RED).  The reader's table lists only the 5 NEW mutants; the 30 older ones of `test_hn_screen_read.py` are unchanged and still RED (see its log).
* gate_token.py / launch_gate.py: `test_hn_ctx_anchor_matrix_run3.log` -> ZZTEST_PASS 72  (38 arms on the real tree + 34 RED mutations)
* hnscreen_devbox.py (the chain): `test_hnscreen_ctx_chain_matrix_run2.log` -> ZZTEST_PASS 19  (9 chain arms + 10 RED mutations of hnscreen_devbox.py)
* hn_screen_read.py (the reader): `test_hn_screen_read_run1.log` -> ZZTEST_PASS 36  (1 unmutated suite (incl. items 5, 6, CLI, paired families) + 35 RED mutations)
* gate_token.py / launch_gate.py (hnfp2's test_hn_content_fp.py matrix, table re-anchored on HNFP3's code): `test_hn_content_fp_run2.log` -> ZZTEST_PASS 81  (62 arms on the real tree + 19 RED mutations)
<!--/MUTANTS-->

Full table: `MUTANTS.md` (68 mutants listed, 68 RED, generated from the final logs by `scripts/mutant_table.py`). Gate level: run1 (`test_hn_ctx_anchor_matrix_run1.log`) had 32 of 34 RED + 2 PROBLEM lines, both **mutant-table defects, not code defects** (the "final STAT comparison deleted" mutant expected R2 RED, but a size change also fails the content comparison;
the `fp_key`-only mutant was EQUIVALENT: replaced by a two-layer mutant); run2 died at mutant 8 on a full disk (`WinError 112`; its 7 mutants were RED); `launch_gate.py` was last edited AFTER run1, so **run3 re-ran the whole matrix on the final tree: ZZTEST_PASS 72 (38 arms + 34 RED mutants).**
Chain level: run1 `ZZTEST_FAIL 1`: the "start verify without the anchor flags" mutant WAS red (the read-count arm), but its list also demanded R2, which cannot tell the modes apart (a content verify refuses a size change too): an over-specified expectation, R2 removed from it with the reason in a comment; **run2: ZZTEST_PASS 19 (9 arms + 10 RED mutants).**
hnfp2's own matrix (`test_hn_content_fp.py`, 19 mutants): 6 snippets stopped applying (FINISH_NOTES 2) and the harness reported them as "stayed GREEN"; re-anchored, **ZZTEST_PASS 81 (62 arms + 19 RED mutants)**, equal to the tip's 81.

## 5. R7, the literal form: content mode is the tip, bit for bit (MEASURED: `scripts/r7_identity_vs_tip.py`, `logs/r7_identity_vs_tip.json`)

The tip's `refe/` (base) and mine run the same worker over 9 cache states (clean; same-size-same-mtime flip; size change; mtime only; extra file; missing file; no manifest; manifest.json flip; a valid RESEALED cache that is not the registered one) and
record `content_fingerprint`, `fp_key`, `data_manifest`, `profile_refusals`, `launch_gate verify` (rc, stdout, `--json` record), `launch_gate refusals`, the minted token's binding, and **the number of full-file reads of each call**.
Normalised: wall-clock fields, hmac, `evidence_sha256` (each evidence file embeds its write time), `gate_sha256` (the hash of the changed source itself). One shared key for both sides.
**Result: 83 of 83 observables identical** (`ZZR7_IDENTICAL`), incl. mine with an explicit `--ctx-mode content` against the tip without the flag. Per-call reads on the clean cache (identical on both sides): content_fingerprint 10, data_manifest 10, verify 10, refusals 10, mint 20.
**Control (a differential that cannot fail proves nothing; `scripts/r7_control.py`, `logs/r7_control.json`):** an unmutated copy of mine must come out IDENTICAL, and two one-line mutants must come out DIFFERENT: (A) one word of a content-mode refusal message, (B) an extra full read
per content fingerprint. B went RED (50 DIFF rows); A first stayed identical because the 8 original scenarios never reached that message branch - **the control found a hole in the differential**, fixed by adding the 9th scenario (resealed cache); A then differs (3 DIFF rows, standalone run). The control script's final run is gated and may be pending (HANDOFF).

## 6. Sibling suites: every test file that names an edited file, base vs mine, full form

<!--SIBLINGS-->
| file | base rc | mine rc | base count | mine count | skips b/m | verdict |
|---|---|---|---|---|---|---|
| refe/test_launch_gate.py:--static,--no-banked | 0 | 0 | 1167 | 1168 | 0/0 | MORE (new checks) |
| eval/test_hn_screen_read.py | 0 | 0 | 31 | 36 | 0/0 | MORE (new checks) |
| refe/test_hn_content_fp.py | 0 | 0 | 81 | 81 | 0/0 | SAME |
| refe/test_hn_ctx_anchor.py:--no-mutants | - | 0 | - | 38 | -/0 | NEW FILE (mine only) |
| refe/test_hnscreen_ctx_chain.py:--arms-only | - | 0 | - | 9 | -/0 | NEW FILE (mine only) |
| refe/test_suite_nesting.py | 0 | 0 | 111 | 112 | 0/0 | MORE (new checks) |
| refe/test_ctx_cache.py | 0 | 0 | 15 | 15 | 0/0 | SAME |
| refe/test_hn_screen_wiring.py | 0 | 0 | 120 | 120 | 0/0 | SAME |
| refe/test_sft4r_gate.py | 0 | 0 | 153 | 153 | 0/0 | SAME |
| refe/test_carve_feed_gate.py | 0 | 0 | 10 | 10 | 0/0 | SAME |
| refe/test_gate_gpu_probes.py | 0 | 0 | 14 | 14 | 0/0 | SAME |
| refe/test_aux_g12_bf16.py | 0 | 0 | - | - | 1/1 | SAME |
| refe/test_p2r_chain.py | 0 | 0 | 139 | 139 | 0/0 | SAME |
| refe/test_rl_s1b.py | 0 | 0 | 18 | 18 | 0/0 | SAME |
| refe/test_rl_r0c.py | 0 | 0 | 26 | 26 | 0/0 | SAME |
| refe/test_rl_pickcon.py | 0 | 0 | 20 | 20 | 0/0 | SAME |
| refe/test_rls1_chain.py | 0 | 0 | 522 | 522 | 0/0 | SAME |
| refe/test_s1aux_chain.py | 0 | 0 | 256 | 256 | 1/1 | SAME |
| refe/test_sft4r_chain.py | 0 | 0 | 31 | 31 | 0/0 | SAME |
| refe/test_rls1e_chain.py | 0 | 0 | 936 | 936 | 0/0 | SAME |
| refe/test_hnscreen_chain.py | 0 | 0 | 16 | 16 | 0/0 | SAME |
| refe/test_s1c.py | 0 | 0 | 220 | 220 | 2/2 | SAME |
| refe/test_rl_safeobj.py | 0 | 0 | 61 | 61 | 0/0 | SAME |
| refe/test_rls1_queue.py | 1 | 1 | - | - | 0/0 | FAILS ON BOTH (pre-existing in this layout/environment) |
| refe/test_p2rep_chain.py | 0 | 0 | 353 | 353 | 0/0 | SAME |
| refe/test_p6_chain.py | 0 | 0 | 351 | 351 | 0/0 | SAME |
| refe/test_p6_levers.py | 0 | 0 | - | - | 0/0 | SAME |
| refe/test_aux_s1_wiring.py | - | 0 | - | 20 | -/0 | MINE PASS; NO BASE COMPARISON (base run skipped: no disk) |
| refe/test_gate_suite_prerun.py | - | 0 | - | 144 | -/0 | MINE PASS; NO BASE COMPARISON (base run skipped: no disk) |
| refe/test_launch_gate.py | - | 0 | - | 1213 | -/0 | MINE PASS; NO BASE COMPARISON (base run skipped: no disk) |

NEW FAILURES: none; LOWER COUNTS: none; PENDING: none; NO BASE COMPARISON (base skipped, no disk): ['refe/test_aux_s1_wiring.py', 'refe/test_gate_suite_prerun.py', 'refe/test_launch_gate.py']
<!--/SIBLINGS-->

Lists: `scripts/list_base.txt` (the tip's 29 test invocations) and `scripts/list_mine.txt` (+ the two new tests); chosen by `grep -l 'gate_token|launch_gate|hnscreen_devbox|hn_screen_read'` over the tip's `refe/` and `eval/` (25 files) + `test_ctx_cache`, `test_hn_screen_wiring`, `test_sft4r_gate`;
`test_launch_gate.py` in both its static and its full form. MEASURED (`suite_mine.json`, `suite_base.json`, `suite_extra_mine.json`): `test_launch_gate.py --static --no-banked` rc 0 **1168 vs the base's 1167**: the +1 is the new `HNSCREEN_SUITE` entry, CONFIRMED by the base run (`test_suite_nesting.py` 111 -> 112 is the same entry); `eval/test_hn_screen_read.py` 31 -> 36 (+5 new mutants).
`test_hn_content_fp.py`: the main `suite_mine.json` entry (rc 1) is SUPERSEDED by `suite_extra_mine.json` (rc 0, 81 = base) after the table fix. **FAILS ON BOTH:** `test_rls1_queue.py` (FINISH_NOTES 1). **No base comparison (base run skipped them for lack of disk; MINE only):** `test_aux_s1_wiring.py` rc 0 ZZTEST_PASS 20, `test_gate_suite_prerun.py` rc 0 ZZTEST_PASS 144, `test_launch_gate.py` full form rc 0 ZZLG_FULL_PASS 1213.

## 7. Decisions the Master Mind should know about (each is a judgement call, flagged)

1. **G-SUITE runs the 38 arms, not the 34-mutant matrix**: `HNSCREEN_SUITE` carries `{"script": "test_hn_ctx_anchor.py", "token": "ZZTEST_PASS", "args": ["--no-mutants"]}` (a tree copy per mutant is minutes on the pod). Declared, not skipped; the matrix is run at landing and in the sibling suite. Revert to the bare entry if you want the matrix in G-SUITE.
2. **The chain test is a SIBLING-suite test, not a G-SUITE one**: the pod-side suite must not depend on `eval/`, and the chain test needs the landed GPU lock under `eval/`.
3. **The reader's argv gains `--chain-root` (and `--key-file`) in BOTH modes**: a content-bound chain also needs the tokens present. Cost: any caller of `hn_screen_read.py` outside the chain must now pass `--chain-root`. Searched: the chain is the only caller in `refe/ eval/ code/`.
4. **END verify requires content AND stat equality** (a restored file with a changed mtime also VOIDs): conservative, as the ruling's R3 is.
5. **The chain default is anchor-bound; the gate default is content.** A caller that does not name a mode gets today's behaviour; only the chain asks for (b').
6. **UNVERIFIED, pre-existing, not mine, and it may bite the screen regardless of (b')**: the chain's code default `gate.token_check_names = {"G14": "G14", "G15": "G15"}` (`hnscreen_devbox.py:79`) is looked up in the PASS token's `checks`, but a real `refe_hnscreen` token's checks are
   `G-HYG, G-DVB, G-PREFLIGHT, G-SUITE` (+ `G-CLOCK` as n/a) - MEASURED in my chain test, which therefore maps both to `G-PREFLIGHT`. With the code default `token_checks()` returns False and `assemble_gates` would read G14 = G15 = False (-> SCREEN-INVALID). The production run-box config
   may override it; I could not see it. **Check before launching the screen.**
7. **Run-box prerequisite (S2 note):** the anchor is minted only from a cache whose `content_manifest.json` seal exists and equals the registered pin: the seal (`ctx_cache.finalize`) and the fill (`hn_fingerprints.py` -> `fingerprints_expected`) must both have run, as for hnfp2's content mode.
8. **mtime is part of the contract.** Copying the cache to another volume (or any tool that rewrites mtimes) invalidates an anchor made before the copy: re-anchor (R3, conservative by design). exFAT stores mtime at 10 ms resolution; the anchor records what the OS returns, so a stable volume is stable.

## 8. Evidence classes

* MEASURED (ours, artifact paths in section 9): the call-map line numbers; the read counts (20 -> 4); R7 83/83; the 9 chain arms; the 38 gate arms; the reader suite 36; the control B; the 34 gate-level RED mutants (run3, final tree), the 10 chain mutants (run2), hnfp2's 19 re-anchored mutants (run2), the sibling-suite table, the race probe of `test_rls1_queue.py`.
* INHERITED (not re-verified): the cache size ~69.6 GiB; 44 MB/s cold D: read speed; the 27 min per full read; hnfp2's landed code at the tip (its own tests were re-run in the sibling suite).
* ESTIMATED: the 1.8 h / 9 h times (arithmetic on the INHERITED rate).
* HYPOTHESIS: none asserted.

## 9. Proposed register rows (NOT written; the MM owns the registers)

* `GOALS_AND_CLAIMS.md`: **HNSCREEN-FP-COST** -> *closed pending landing*: option (b') built and tested; 3-seed screen 20 -> 4 full reads (MEASURED); content mode bit-identical to the tip (MEASURED 83/83 + control); residual (same-size, same-mtime edit mid-chain) is detected at the END verify and VOIDs the chain (R1, MEASURED).
* `RETRACTION_LOG.md` (class: *an inherited count not re-derived from source*): the ruling's and hnfp2's "17 reads per screen" is **20** (MEASURED, `logs/reads_per_screen.json`).
* `BACKLOG.md`: (1) **check `gate.token_check_names` in the production chain config** (section 7.6, UNVERIFIED); (2) land the package (hot files); (3) before the screen: run the full chain on the run box once with `gate.ctx_mode` unset and read `ctx_chain/end_verify.json`; (4) C: free space is a programme-level hazard (0.1-0.7 GB free on 2026-10-19 23:20-23:35; a 5.1 GB runaway REPL output
  `tasks/bpg6tw7gz.output` from 12:29-14:29 is in this session's task folder - not mine, not touched).

## 10. Deliverable manifest (staging = `out/` + `out/paths.json`; every item below is on ONE disk until the MM lands it)

Stage dir `C:/Users/Admin/par18/hnfp3/out/` (built by `scripts/make_pkg.py`); the work trees are `C:/Users/Admin/par18/hnfp3/work/` (mine) and `.../base/` (the tip's bytes). Repo targets: code at its real path under
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/{refe,eval}/`; evidence under `.../raw/2026-10-19-hnscreen-fp-cost/`.

| item | repo path | where it lives now |
|---|---|---|
| `gate_token.py`, `launch_gate.py`, `hnscreen_devbox.py` (hot files: land by `base=<blob>`) | `.../refe/` | `hnfp3/out/refe/` |
| `hn_screen_read.py`, `test_hn_screen_read.py` | `.../eval/` | `hnfp3/out/eval/` |
| `test_hn_ctx_anchor.py` (new), `test_hnscreen_ctx_chain.py` (new), `test_hnscreen_chain.py` (+2 lines) | `.../refe/` | `hnfp3/out/refe/` |
| `test_hn_content_fp.py` (hnfp2's test; MUTATIONS table re-anchored, test only: land it with the code or the sibling suite goes red) | `.../refe/` | `hnfp3/out/refe/` |
| FINISH_NOTES.md, scripts/rls1_queue_race_probe.py, scripts/patch_fp_table.py, scripts/run_finish_matrices.sh, logs/test_hn_content_fp_FINAL.log (+ the new matrix logs) | `.../raw/2026-10-19-hnscreen-fp-cost/` | `hnfp3/out/raw/` |
| `code_vs_tip.diff`, `TIP_SHA.txt` | `.../raw/2026-10-19-hnscreen-fp-cost/` | `hnfp3/out/raw/` |
| RESULT.md, CALLMAP.md, MM_RULING_FPCOST.md, MUTANTS.md, HANDOFF.md, scripts/, logs/ | `.../raw/2026-10-19-hnscreen-fp-cost/` | `hnfp3/out/raw/` |

## 11. Finish pass (2026-10-20): what changed after the hand-off, and what it did not touch

* Production code (`gate_token.py`, `launch_gate.py`, `hnscreen_devbox.py`, `hn_screen_read.py`): **unchanged** since the hand-off (file mtimes 22:40-22:59 on 2026-10-19, before every final run; sha256 printed by `scripts/make_pkg.py`: gate_token d8184aa0, launch_gate 3eb504c4, hnscreen_devbox bcf47c6b, hn_screen_read ea30ea7f; identical before and after this pass).
* Changed, TEST TABLES ONLY: `refe/test_hn_content_fp.py` (six mutant snippets re-anchored) and `refe/test_hnscreen_ctx_chain.py` (one mutant's expected-arm list). No arm was added or removed; the 38 gate arms, 9 chain arms and 62 content-fp arms are untouched.
* Found while following HANDOFF step 2 (not in the brief): the chain matrix was `ZZTEST_FAIL 1` and the gate matrix run2 had died on a full disk while still writing a `.DONE`: both re-run green (section 4). A `.DONE` marker is not evidence of success (it was written after a crash).
* `hnfp3/tmp/` (6.7 GB, not mine) was left alone for the PI; my scratch `hnfp3/tmp_fin/` was removed at the end.
