# HANDOFF_V44_BUILD (third agent, 2026-11-07 ~16:10 dev-box clock). BUILD EXIT 0. Nothing staged, committed, pushed or branch-switched. CPU only.

Folder `E:/Projects/TanitAD-artifacts/dagland/report_v44/`. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`. Built on the live tip `0423bb9de2d0461bb1decceedee61979908cf3f8` (checked equal to `git ls-remote` at the end). Base: landed v43 (`R/report/2026-11-07-v43/`, md sha256 `ad2df9ba...`).

## Build and result
`bash build_v44.sh 0423bb9de2d0461bb1decceedee61979908cf3f8` (about 1.5 to 3 min, deterministic; pass the FULL 40-hex tip). Exit 0.

| stage | result |
|---|---|
| patch (`patch_v44.py`) | builds. All anchors unique (C43S twice, as in v43), 5 PI quotations verbatim in a tip source, 42 eight-hex tokens each found in a tip record or a commit of the mirror (one, `2ecd212f`, only in the E: file `HANDOFF_IMAG1.md`, reported), 13 register row ids found in the register, banned-word / closed-loop / dash / `drives` scans 0, math fences intact, 0 placeholders |
| audit (`audit_v44.py`) | 209 added lines (176 audited, 33 exact source table lines), **0 unmatched**, **0 unresolved scope misses** (3,449 source files). New second layer: a token of a line that names C0C-RE, TSWAP, LATSEM2 or the imagination plan must occur in that read's own folders; 7 tokens accepted with a written reason in `SCOPE_OK` |
| checks (`checks_v44.py`) | **25 PASS, 0 FAIL** (base sha, html = template + md, status line, live tip, only 8 carried lines changed, scans, no pending wording, STRATEGIC PAUSED, tiers, plan says no result, outcome words, section order, rows 1bm to 1bu, mutation arm) and it runs `check_numbers_v44.py` |
| numbers (`check_numbers_v44.py`, writes `NUMBERS_V44.md`) | **503 checks, 0 FAIL** (321 of the first agent plus 182 new: each new headline number is found in its tip source file AND in the built report; every place the report prints an interval prints it with the same point). Mutation arm: two numbers changed in a copy make it FAIL |
| offline render | md and html: 296 tables, 64 h3, 649 math spans, **0 ragged tables**, 0 stray placeholders, 0 unfilled braces (v43: 281 tables) |

Outputs: `out/V8F_SCIENTIFIC_REPORT.md` (6,928 lines, 1,194,089 bytes, sha256 `a62b4327bb80149162d7ac462e00552f12e0a67903ceec3399c615d0693d829d`), `out/v8f_report.html` (1,205,509 bytes, `b73dca695471697bf62729121d170cec6a324f479f1b98a4325581e858c8b6fe`), `out/DIRECTIVES.txt`. `MANIFEST.txt` lists every file with bytes and sha256.

## Hold-ready layout (Master Mind; `land_hold.sh` was NOT run)
Land as `R/report/2026-11-07-v44/` (`R` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/`), files directly in the folder, as v43 did:
* from `out/`: `V8F_SCIENTIFIC_REPORT.md`, `v8f_report.html`, `DIRECTIVES.txt`;
* from this folder: `patch_v44.py`, `audit_v44.py`, `checks_v44.py`, `check_numbers_v44.py`, `build_v44.sh`, `render_check_v44.js`, `edits_v44.md`, `sec_747_world_model.md`, `sec_748_renc_town.md`, `NUMBERS_V44.md`, `v44_NUMBERS_AUDIT.md`, `v44_checks.txt`, `v44_render_check.txt`, `MANIFEST.txt`, `HANDOFF_V44.md`, `HANDOFF_V44_BUILD.md`.
* Never land `scratch_orig/` (the first agents' parts as received; the ONLY copy of those originals, so kept), `out/` as a folder name, or anything `*.before_*`. A landed copy is a record, not a runnable tree (the scripts read `out/` and the section files beside them).
* Also due (Master Mind): the claim-register rows for the report itself, if the lander requires them.

## What this build changed in the parts (all recorded in `edits_v44.md` NOTES and `scratch_orig/` for the as-received texts)
1. **C0C-RE read (`6b1127f4`)** replaces the PENDING text: 7.48.2 now reads RE-HARMS from `RESULT_T2_RENC.md`, `.json`, `exclusions5.out` (clauses table, pinned-read result, beside rows, what is not read), and V44S, ABS1, NXT, R83, ROW 1bo, I22, C46C follow. `G_ID_C0CRE.json` is now on the tip (`id_cell/`), the "not on the tip" mentions are removed.
2. **TSWAP (`fafbb4aa`)**: new 7.48.4 (T-NOT, controls, M1 and T share, ADE6 beside, the separated-worse rows named as the card requires); TOWN's "Next" paragraph rewritten; row 1bq; 7.48.5 establishes and 7.48.6 provenance renumbered.
3. **LATSEM2 run 1 INVALID (`59bc4e26`) and LATSEM2B (`d2771e01`)**: new 7.47.8; row 1bs now says run 1 INVALID and LATSEM2B all controls pass (not "a read is running"); 7.47.7 now covers MRATE only. MRATE and LATSEM2 cards are stated as on the tip (`c85eed44`, `e8789391`).
4. **Imagination plan (`912e82e1`) and 50 banked papers (`0423bb9d`)**: new 7.47.9, labelled a plan; row 1bu; appendix rows; library index 743 to 793 entries, 5320.5 to 5755.9 MB (from the LIBRARY.md headers).
5. Renumbering: 7.47.10 (establishes), 7.47.11 (provenance), 7.48.5, 7.48.6; cross-references in S81 updated.

## Flags for the Master Mind (decisions first)
1. **IMAG1 / VOE1 state differs from "pre-registered and running".** The register row `H-V8F-IMAG-1` says "pre-registered, running" and the text quotes that. Dev-box files (NOT on the tip) show: IMAG1 launched 15:20:39 (`theory/imag1/HANDOFF_IMAG1.md`); **VOE1 ran once and is INVALID by its own controls 1 and 2 as committed, and a new card `PREREG_V8F_VOE1B.md` was written** (`theory/voe1/HANDOFF_VOE1.md`, `RESULT_VOE1.md`). 7.47.9 states this in one bullet labelled [I, UNVERIFIED by the build]; no result is claimed. Delete that bullet if you do not want dev-box state in the paper, or land VOE1 / VOE1B first and rebuild.
2. **Register imprecision (D-T2-C0CRE-1 and the card log)**: they say the new separated-worse row at s1 is heading MAE with "interval -5.5 to +6.2 on the pair". Those bounds are the s0 pair row; the s1 pair row is +2.8549 [-1.8623, +8.6949]. The report uses the result file. The report also says (labelled [E]) that the RE-HARMS label at s1 rests on that single row, which is separated-worse against P10 for C0C-RE and not for C0C while the pair row is not separated. Consider correcting the register row.
3. **Audit limits, stated:** the first layer ("first source containing the token") is weak by design; the scoped layer and the 503 number checks are the gate for decisive numbers. TAC-LAT's sign convention (a score where lower is worse) is read from the result file's own status labels, not from a definition. `T share` and the seed gaps are quoted as printed.
4. **UNVERIFIED / not re-run:** the C0C-RE card-log facts "tests 17 of 17", "self-test 26 of 26, 9 of 9 mutations red" ([I]); the DAgger card byte count 5,415 (found in the card's line by the audit); the E: dev-box facts in flag 1.
5. **Incident, repaired:** I first ran the copied `build_v43.sh` before it was renamed, which rebuilt `report_v43/` at the new tip. I restored it by rebuilding at the original tip `ba1cce15...`; the six outputs (`out/` md, html, DIRECTIVES, `v43_NUMBERS_AUDIT.md`, `v43_checks.txt`, `v43_render_check.txt`) equal the original MANIFEST sha256 values bit for bit. Nothing of v43 landed in the repo was touched.
6. E: was at 0 bytes free during the build (other sessions); two build attempts failed with ENOSPC and were re-run. The final files are complete (checks pass, MANIFEST written).

## Deliverable manifest (artifact | where | copies)
* v44 report md, html, DIRECTIVES | `E:/Projects/TanitAD-artifacts/dagland/report_v44/out/` | **single copies on E:** (rebuildable from the tip and the section files; not yet in the repo)
* build scripts, section texts, edits, NUMBERS, audit, checks, render check, MANIFEST, handoffs | `report_v44/` | **single copies on E:**
* `scratch_orig/` (as-received parts of agents 1 and 2) | `report_v44/scratch_orig/` | **single copy**, never land
* nothing is on a pod, in a worktree, staged or pushed

## Deleted (own temp data, literal paths, sizes)
Logged in the final hand-back and below if added after this file was written.
