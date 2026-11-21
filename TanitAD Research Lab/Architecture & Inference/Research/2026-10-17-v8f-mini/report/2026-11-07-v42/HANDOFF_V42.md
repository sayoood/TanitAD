# HANDOFF v42 (build agent 2, DONE; nothing committed, pushed or staged)

Brief: `BUILD_BRIEF_V42.md`. Folder `E:/Projects/TanitAD-artifacts/dagland/report_v42/`. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.

## Build and result (MEASURED, this folder)
* Command: `bash build_v42.sh 1887e9b8a5f9c5a17bab6b3de94959f8c86c2c28` (about 35 s, CPU, git reads only). Exit 0: patch OK, audit 0 unmatched, **all 33 checks PASS**, offline render 0 ragged tables (270 tables, 60 h3, 470 math slots; v41 had 259 / 57 / 324).
* Tip: `git ls-remote` read `1887e9b8…` at build (it moved from `b17b5492` during the build: a yaw-rate fine-tune pre-registration under `2026-09-20-refe-plan` and one register row; nothing under the v8F folder changed; `b17b5492` is its ancestor). Re-check the tip before landing; if it moved again the build re-runs on it unchanged.
* Base: the landed v41 (md sha256 `c73d0ef6…`, html `70b92206…`; files sit directly in `R/report/2026-11-06-v41/`).
* Output: `out/V8F_SCIENTIFIC_REPORT.md` sha256 `ada2083063ed8bebcac82ee64444d8760b2766c8bda643c51a8d1af96c637aea` (1,017,985 bytes); `out/v8f_report.html` sha256 `0bf58200cade629f71c58ebba976c70f16c119a35d74a0c558447c8a8dc04508` (1,029,405 bytes). `MANIFEST.txt` holds every file's sha256.

## What changed (details and every edit in `edits_v42.md`)
* New: 7.42 (T2), 7.43 (LoRA pod), 7.44 (theory and weight-space reading); abstract (two bullets, next-bullet sentence), status line, section 3 line (Library key `takehi2026weightspace`), 8.1 item 32, 8.3 Update (v42), 8.4 rows 1bd to 1bf, section 9 item 20, section 10 block (table, one row per new raw folder).
* Carried v41 lines edited in place (12 lines; the checks pin exactly these): status, evidence classes, abstract contribution paragraph, abstract next bullet, section 6 tiers, the 7.15 T1-kin paragraph, 7.41.6 (b) (pointer appended), the 8.2 driving-claim bullet, 8.3 (iv) with the closing sentence of its paragraph, 8.4 row 6, and two section-9 bullets "No T2 number exists" (items 18, 19). The two-word T2 wording count of the report fell from 15 to 4 (the 4 left sit in the two quoted v34 review paragraphs of section 3); no banned word was added, and the per-word counts equal v41's.
* Section files inserted as written, except one fix in `sec_744_theory.md` (`$\Delta g = $` rendered as literal text; now `$\Delta g$ =`) and the late results below.

## Late results folded in (the brief asked for it)
* **P1': NOT EVALUABLE as pre-registered** (CONFIDENT 24 windows, floor 30; UNSURE 75; 168 windows, 45 episodes), from `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/P1P_RESULT.md` (written 00:35, E: only). Descriptive difference of median rho +0.365 / +0.437, not a verdict; the GO24 sensitivity returns HOLDS and is NOT promoted. 7.44 intro, 7.44.2, 7.44.7 row, 7.44.8 and every summary block say so; a check ties each number to that file.
* WSP-GAIN: the fit finished (`raw/FIT_RESULT.json`, `EDIT_PARAMS_s{0,1}.npz`); no PIN entry, no `READ1.json`, no edited model scored: stays "pre-registered, not yet read" (7.44.6, row 1bf).
* LoRA pod: batch-64 timing still not measured (`PRELIM_SUMMARY.json` is on the pod only; `HANDOFF_LORAPOD3.md` has no number): stays "not yet measured". **Read re-pin** (`E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/HANDOFF_Q.md`, written 00:41, E: only): the dominance read script now takes `--arm NAME`; fourth pin `7fd65ffd…`; identical to the read of record on the real pulls; 80 of 80 mutations RED; the suite run and the card's log entry were in flight. Stated as written in 7.42.6 item 3 and 7.42.7; a check ties it to that file. The read of record still cites the third pin `d6664718…`.

## Land (Master Mind)
Land as `R/report/2026-11-07-v42/` (deliverables directly in the folder, as v41 did): `V8F_SCIENTIFIC_REPORT.md` and `v8f_report.html` and `DIRECTIVES.txt` from `out/`; `patch_v42.py`, `audit_v42.py`, `checks_v42.py`, `build_v42.sh`, `render_check_v42.js`, `edits_v42.md`, `sec_742_t2.md`, `sec_743_lorapod.md`, `sec_744_theory.md`, `v42_NUMBERS_AUDIT.md`, `v42_checks.txt`, `v42_render_check.txt`, `MANIFEST.txt`, `HANDOFF_V42.md`. Commit message: `msg_v42.txt` (ends `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` as the brief asked; the session's own trailer line says Sonnet 5.5, so choose). Landing with the lander (`land.sh` / `scoped_commit.py`), then verify the push by the remote ref.
* E: only, cited by the report as "not yet landed": land these raw artifacts too (suggested `raw/2026-11-07-lora-pod-handoffs/`: `lora_pod/HANDOFF_LORAPOD2.md`, `HANDOFF_LORAPOD3.md`, `AMENDMENT_V8F_LORA_4_LOG.md`; `raw/2026-11-07-theory-p1p/`: `wsp_read/P1P_*`, `SPEC_P1P*`, `HANDOFF_P1P.md`; `raw/2026-11-07-theory-wspedit/`: `wsp_edit/` code and handoff). They exist in one place only.
* The checks read the P1' numbers from the E: file; the numbers audit reads E: files for 2 tokens (6.0e-07, 2.4e-6). Run both on the dev box.

## Next (v43, when the reads land)
7.43.2 and 7.43.6 (batch-64 timing, B' progress), 7.44.6 and row 1bf (WSP-GAIN READ1 and the T1-kin arms), 7.44.2 (a dated P1' variant pre-registration: the GO24 probe or a lower CONFIDENT floor, or more held-out windows), then the LoRA arm's T2 dominance read.

## Deliverable manifest
| artifact | where |
|---|---|
| out/V8F_SCIENTIFIC_REPORT.md, out/v8f_report.html, out/DIRECTIVES.txt | `E:/Projects/TanitAD-artifacts/dagland/report_v42/out/` |
| patch_v42.py, audit_v42.py, checks_v42.py, build_v42.sh, render_check_v42.js, edits_v42.md | `.../report_v42/` |
| sec_742_t2.md, sec_743_lorapod.md, sec_744_theory.md (one fix, see above) | `.../report_v42/` |
| v42_NUMBERS_AUDIT.md, v42_checks.txt, v42_render_check.txt, MANIFEST.txt, msg_v42.txt, HANDOFF_V42.md | `.../report_v42/` |
| `_v41copy_*` (7 files, untouched originals), BUILD_BRIEF_V42.md | `.../report_v42/` (not landed) |
All on the dev box E: only; nothing staged, committed or pushed.
