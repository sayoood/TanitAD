## REVISION 1 NOTE (dated 2026-10-20; MM ruling `MM_RULING_C2FINAL.md` point 1, sha256 `70a08e0912165bc031f499742212bb23daf216eb95a9033d3dbe8207215eb17b`)
**Code revision only. The terms text of this amendment is byte-identical to the sealed `AMENDMENT_DOSE.md` (sha256 `d55ebac686cb4d8ec979872c9ae8111f8f6e14ea31641310d387898d89910679`, kept unedited beside this file).** Exactly three things differ from it: the Status line, this note block, and the pins block (the shas of the four changed code files are re-pinned, `cost_dose.py` is unchanged; `ruling_c2final_sha256` and `amendment_original_sha256` are added). `dose_chain.py::terms_text` removes those three and `check_amendment` refuses unless the rest is equal; `seal_dose.py` asserts the same at the seal; `test_dose.py::test_revision1_terms_text_is_the_originals_byte_for_byte` lists the sections. (This note is not the body's own `## Revision 1 (MM ruling, pre-seal)` section, which folds in `MM_RULING_DOSE.md` and is part of the terms text.)

**The ruling, point 1 (verbatim):** *"'Moved share' has ONE meaning everywhere: `inert.frac_moved`. That is the share of pooled (token, seed) outputs whose ADE vs the pick exceeds 0.5 m, the quantity whose shortfall IS O4_INERT."* It holds for "moved share" in `MM_RULING_DOSE.md` DQ4's table (section 5b of this text), in `MM_RULING_C2_LICENCE.md` section 2 and in every other place this text says "moved share".

**The defect, found before any read.** The sealed `dose_chain.py::licensed()` read `read["moved_share"]`, the mean of the search's `moved` flag, which is any non-zero move. That is a different, looser quantity: a read with `moved_share >= 5 %` and `inert.frac_moved < 5 %` was licensed. No C1 result, no dose result and no navtest label existed when it was found.

**What this revision changes (code only):**
- `licensed()` reads `inert.frac_moved` through one helper, `frac_moved_of()`; a missing or NaN value is NOT a licence.
- `analyse_dose` passes the dose read's own `inert` block (the one `spc_analyse.analyse_c1` returns, computed by `tts_cem.inert_verdict`, the function C1's own INERT uses; nothing is re-computed here).
- `check_c1_done` passes C1's `inert` block from `RESULT_SPC.json` and REFUSES (`C1_NOT_DONE`, naming `inert.frac_moved`) if it is absent.
- `c2_config` records the quantity it read in `moved_term`. `moved_share` stays in every output, DESCRIPTIVE ONLY, and decides nothing.
- `check_amendment` checks THIS file (`AMENDMENT_DOSE_R1.md` + `.sha256`) in place of the original, and additionally the original's own sha256 and the terms-text identity.

**Unchanged:** every term of the amendment, the DQ4 table, the outcome rules, the gates, the pinned configs, `MM_RULING_DOSE.md`, C1's sealed files, the cost estimate, and the strategic pause. The sealed C1 files are imported, never edited.
