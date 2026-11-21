# P6 navtest READ instrument on the 2026-11-06 drive layout: path roots only, scoring identical

Agent: Sonnet, 0 GPU, nothing committed / staged / pushed, no git index touched. Evidence class of every number: MEASURED (artifact path given). Layout: E: exFAT (repo + artifacts), F: NTFS (navsim archive `F:/Archive/devbox-C/navsim`), D: absent.

## What changed (12 repo files, `MANIFEST.tsv`; copies under `files/<repo path>`)
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

| file (under PK) | change |
|---|---|
| `eval/stage2_sft.py`, `eval/stage3_full.py`, `eval/selection_quality.py` | `EXPORT` = `_navsim_root() + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"` (resolver inlined) |
| `eval/command_adherence.py` | `EXPORT` default via the resolver; `P2_TABLE` default via the artifacts resolver (env `REFE_NAVTEST_EXPORT` / `REFE_P2_TABLE` still win, lazily) |
| `eval/harness/score_e.py` | `m.W3_EXPORT` via the resolver (was `.replace("D:/","E:/")`) |
| `eval/harness/w3e/code/run_v1.py` | `V11_TREE`, `EXP`, `DATA`, `ENV["NUPLAN_MAPS_ROOT"]` via the resolver |
| `eval/harness/w3e/code/navsim_v1_win.py` | `V11_TREE`, `NUPLAN_SOURCE`, `ENV_DEFAULTS` (3 roots) via the resolver |
| `eval/harness/w3e/code/write_cache_metadata.py` | `EXP` via the resolver |
| `eval/harness/HARNESS_PINS.json` | 4 pins recomputed (below) + a `layout_fix_2026_11_06` note per changed file |
| `eval/test_p6_read_gaps.py` | G3 RED arm repaired (decoy); G4 inputs default to the artifacts root; scratchpad-identity test aware of the moved pins; +3 layout checks |
| NEW `eval/rewrite_cache_metadata_root.py` | the CSV re-root tool (also the recipe for the next archive move) |
| `raw/2026-10-22-p6-read-gaps/RUNBOOK_P6_READ.md` | "Drive layout 2026-11-06" section (not a rewrite) |
| (archive, not in git) `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest/metadata/metric_cache_navtest_metadata_node_0.csv` | 12,146 rows re-rooted `D:\` -> `F:\` |

Resolver (identical text in every file; only the name `_root` differs in command_adherence): navsim root = `$TANITAD_NAVSIM_ROOT` if set (a non-existent override REFUSES, it is never skipped), else the FIRST EXISTING of `E:/`, `F:/`, `D:/` + `Archive/devbox-C/navsim`; none => `SystemExit("REFUSING: navsim archive root not found (tried: ...)")` at import. Artifacts root = `$TANITAD_ARTIFACTS_ROOT`, else first existing of `E:/`, `D:/` + `Projects/TanitAD-artifacts` (E: first, unlike the brief's D:/E: order, because E: holds today's artifacts and a stale D: must never shadow it; one tuple to flip).
Untouched on purpose: `score_navtest_refe.py` (so the caller pin is unchanged: variant `NT_FIX1` sha256_lf `94f7b024...`), `run_v1_fix1.py`, `families6.py`, `e1/navsim_win.py`, `stage2_sft.py:138` env map, `--db-dir` (repo data, E:), every estimator / seam / scoring line.

## Pins that moved (computed by `harness_pins.build()` / `md5_file`, never typed)
| pin | old | new |
|---|---|---|
| `score_e.py` sha256_lf | 770a744fc456fa11... | 0b5d7bd24a3f4f50c5fdf823595e73ec92e5d017504adc89f8db80ef21cec754 |
| `w3e/code/run_v1.py` | 00ea0009ca858d09... | 20506fea85463c96c942c4512f791e3666ee1af3554322f37a9b5e53abf260d7 |
| `w3e/code/navsim_v1_win.py` | d02821711e831587... | 873e558bd59b3b118269f4baf6230c29dc580930ca96512626a80c66fad2d081 |
| `w3e/code/write_cache_metadata.py` | b9eca5cb7a4ce01e... | 53bb04cd1c540e3244f7af181985db7a5b8f804302f6704248d8854920b2b880 |
| `HARNESS_PINS.json` `pins_file_sha256` (the value `result.json` records) | 5f590032d075a75e... | 041f9d105eddff19de29b3e479c73ccf6dc0bb58c6d6d6d7a4bbf7d487f8b077 |
| `stage3_full.STAGE2_SFT_MD5` | 3a30ab4c5a122664c534f378144cb4a5 | 74c37dd2d7dba9b03106cde5372ac0e1 |
| metadata CSV sha256 / md5 | ffa54e8ba333bbba... / 9dcae706aa811c02fd1774fe27eec1ee | ed760dd34f144ee762e081002fadbdb2e0a90f70419ed8bc53bc601b44ec1a1c / 160ddd3c8f6be9a1efc3f4009bd34a3c |
Unchanged: caller variants, the other 17 harness pins, every baseline pin in `stage3_full.BASELINES`, `run_v1_fix1.ORIGINAL_SHA256` (the banked run_v1 was already drifted, `RUN_V1_FIX1_ALLOW_DRIFT=1` covers it).

## Acceptance (all `python -X utf8 -I`, CUDA hidden, NO env overrides for archive / artifacts / commands / P2 table; logs `logs/after_*`)
| bar | reading |
|---|---|
| `harness_pins.py --verify` | ok true, n_files 21, caller NT_FIX1 (`logs/after_harness_pins_verify.json`) |
| `test_p6_read_gaps.py` | **ZZP6GAPS PASS 35/35** (was 27/33). G1 C2-C1 = +0.5715 [+0.0179, +1.1755] and G2 level 88.8184 [87.8611, 89.7701] pass their bit / 1e-9 assertions. 35 = the 33 minus the old "epoch-6 ckpt SKIPPED" pseudo-entry (that check now RUNS on `E:/.../p6dry/hf/run/ckpt_last.pt` and passes) plus 3 new layout checks; the G3 scratchpad-identity entry is still a stated SKIP (scratchpad gone) |
| `selftest_stage3_full.py` | ZZSELFTEST_STAGE3 PASS 11/11 |
| `stage3_full.py --selfcheck` | checks_ok=True regressions_failed_as_required=True **R2_PRECONDITION_MET=True** |
| `test_command_adherence.py` | ZZADH PASS 24/24 |
| G3 RED arm repaired | a temp decoy `families6.py` placed FIRST must fail `f6_first`; passes without D: (the old reversed-order arm still runs when a non-banked candidate exists) |
| own RED arms | resolver mutants (never refuses / D: before E: / env override ignored) each fail the semantics check; every eval module imported with `TANITAD_NAVSIM_ROOT=Z:/does/not/exist` REFUSES; a stage2 mutant with the old literal EXPORT fails the wiring check |
| harness chain, no archive override | `score_e.py` with `TANITAD_NAVSIM_ROOT=Z:/does/not/exist` exits 1 `REFUSING: ...` (`harness_control/red_bad_root.log`) |

### Harness identity control (the proof that scoring is unchanged), `harness_control/`
P2's BANKED seam (`stage2_p2_full.npz`, md5 49a1f6ec..., never a P6 checkpoint) re-scored through the REAL scoring call of `stage2_sft.py` (`score_e.py`, cwd = banked harness, `RUN_V1_FIX1_ALLOW_DRIFT=1`) on today's layout: navsim tree, maps, openscene, exp and the re-rooted metadata CSV all on F: (`score/refe_layoutfix_ctl/*_manifest.json` `imported_from` / `env`). Compared per token with P2's banked csv (`data/refe_navtest/score/refe_stage2_p2_full/...csv`), 8 columns (valid, 6 sub-scores, score), exact float equality:
* control A: 24 tokens, 3 logs: harness status PASS, 24/24 valid, C1 max |delta| 0.0, **192/192 cells equal**, mean PDMS x100 84.5553 = banked (`compare.log`).
* control B: 401 tokens, 17 logs: PASS, 401/401, **3,208/3,208 cells equal**, mean PDMS x100 89.1411 = banked (`compareB.log`, `statusB.txt`).
* RED: the ORIGINAL csv has **0 / 12,146** existing paths on this layout, the re-rooted one **12,146 / 12,146** (`red_original_csv_dangles.log`).
Scope of the claim: 401 of 12,146 tokens (136 logs sampled at 17), P2's seam, T1 PDMS; the full 12,146-token re-score was not run (CPU hours); the 21-file pins + 35 tests cover the rest. Estimator unchanged (paired log-cluster bootstrap, 10,000, seed 20260927, answers "another draw of LOGS" only).

## Disclosure paragraph for PREREG_P6 (paste)
> **Layout amendment 2026-11-06 (path-only).** The navsim archive moved E: -> F: and D: disappeared; the read instrument's hard-coded roots were replaced by one inlined resolver (`$TANITAD_NAVSIM_ROOT`, else first existing of E:/F:/D: `Archive/devbox-C/navsim`, refusing if none) in `stage2_sft.py`, `stage3_full.py`, `selection_quality.py`, `command_adherence.py` and four pinned harness files; the metric-cache metadata CSV was re-rooted `D:\` -> `F:\` byte-for-byte on the prefix (12,146 rows, order and header unchanged). No endpoint, comparator, estimator, rule, family, seam argument or scoring call changed. Proven by: `harness_pins.py --verify` ok (21 files, 4 pins moved and listed), `test_p6_read_gaps.py` 35/35 (G1 +0.5715 [+0.0179, +1.1755] and G2 88.8184 [87.8611, 89.7701] to the bit), `selftest_stage3_full` 11/11, `--selfcheck` R2_PRECONDITION_MET=True, `test_command_adherence` 24/24, and the real scoring call re-scoring P2's banked seam on 401 navtest tokens (17 logs) with 3,208/3,208 cells bit-equal to P2's banked csv. Evidence: `E:/Projects/TanitAD-artifacts/p6read_layout_fix/RESULT.md`.

## Not done / for the Master Mind
1. **Not covered:** step 2's smoke code (`raw/2026-10-22-p6-read-dryrun/code/p6_navtrain_plan.py`, `p6dry_common.py`, and `build_candidate_bank.py` / `srttc_*` G4 code) still names `D:/Projects/TanitAD...` for repo data and artifacts (not archive). Workaround without edits: `subst D: E:\`. Unblock = one more path-only pass (not run: scope and context budget).
2. Off-path W3 scripts keep `E:/Archive`: after_a8_chain, after_bridge_chain, analyze_navtest, artifacts_navtest, build_navtest_frames, decompose_gate, export_navtest_inputs, smooth_seam, turn_following (none is imported by the read; `smooth_seam.py` was in the brief but has no caller on the read path). They fail loudly if run.
3. `harness_pins.py --build` DROPS the top-level keys `what / scratchpad / provenance / caller_note` (latent defect; `t_g3_byte_identical_to_scratchpad` would KeyError after a CLI rebuild). I re-pinned through `build()` and merged into the old dict (`repin.py` in the scratchpad: see `logs/repin.log`); `harness_pins.py` itself is unchanged.
4. `HARNESS_PINS.json` is CRLF in the tip blob and in my copy: land it with `hash-object --no-filters` (the manifest's `new_git_blob_id` is that).
5. Side effects of `--selfcheck` (it banks by design): `raw/p2_r2_instrument/{selfcheck.json, selfpair_u400/r2_result.json, swapped_deployed_vs_u400/r2_result.json}` were rewritten in the E: tree (they now record the new stage2 md5); copies in `selfcheck_side_effects/`. Not in the manifest: land or `git checkout` them as you prefer. `__pycache__` dirs appeared (untracked).
6. The harness writes its devkit run dir into the archive (`F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/<label>`); my controls left `refe_layoutfix_ctl` (40 KB) and `refe_layoutfix_ctlB` (256 KB) there (never deleted by me). F: has ~16 GiB free; the read's 64 selq columns add ~64 small dirs.
7. The metadata backup lives in `metric_cache_navtest/metadata_bak_dpaths_20261106/` (and `orig_before/`), NOT beside the CSV as briefed: the devkit loader (`[f for f in metadata_dir.iterdir() if ".csv" in str(f)][0]`) opens the first file whose name CONTAINS ".csv", so `*.csv.bak_dpaths_*` next to it is a trap.
8. If the archive moves again: `python eval/rewrite_cache_metadata_root.py --dry-run`, then without the flag.
