# BRIEF: SEARCH-DOSE Revision 1, "moved share" = `inert.frac_moved` (Master Mind, 2026-10-20 ~05:00)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/doser1/`. The package is `C:/Users/Admin/par18/doseamend/` (sealed `AMENDMENT_DOSE.md`, sha256 `d55ebac6…`; landed at tip `51078a6`).
- CPU only. C: has about 0.3 GB free: small files only, TEMP / TMP in a new dir under your work dir, delete only what you create. If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push. No B-REST data, no navtest data.
- **Do NOT touch `C:/Users/Admin/par18/c2final/` or the navpipe code.** Another job is mutating it right now.

## The ruling (binding): `C:/Users/Admin/par18/c2final/MM_RULING_C2FINAL.md` point 1
- `dose_chain.py`'s `licensed()` (`:124-132`) reads `read["moved_share"]`, the mean of the search's `moved` flag.
- The ruled meaning of "moved share" is `inert.frac_moved`: ADE vs the pick > 0.5 m over the pooled outputs, which is what C1's INERT uses (`spc/code/spc_analyse.py:49-51, :94-96`).

## Do
1. **Code:**
   - `licensed()` reads `read["inert"]["frac_moved"]`; a missing or NaN value is NOT licensed.
   - Every caller passes the read's `inert` block: C1's read from `RESULT_SPC.json` (`:280-295`; refuse by name if `inert.frac_moved` is absent) and the dose's own read (`:185-193`).
   - Confirm from source that the dose result carries its own `inert` block, computed exactly as C1's (`file:line`). If it does not, add it by calling C1's own function, never a re-implementation.
   - `moved_share` stays in the output, descriptive only.
2. **Tests:**
   - `moved_share >= 5 %` with `frac_moved < 5 %` must read NOT licensed, for both C1 and the dose read.
   - the reverse case;
   - a missing `inert`;
   - NaN.
   - Update every fixture that builds a read. Mutation-check the changed lines.
   - Report: `test_dose.py` count, the spc suite (115, C1's files untouched), and the mutation result.
3. **Revision 1 seal:**
   - The amendment's terms text must stay BYTE-IDENTICAL, apart from a dated `REVISION 1` note (MM_RULING_C2FINAL.md sha256 `70a08e09…`) and the re-pinned code shas.
   - If `seal_dose.py` cannot re-seal an already-sealed amendment, add the smallest Revision path: a new `AMENDMENT_DOSE_R1.md` + `.sha256`, which `dose_chain.py` checks in place of the original. Add a test proving the terms text equals the original's byte for byte, with the sections listed.
   - Do NOT run the real seal; the MM does.
   - Write the exact command into the HANDOFF.
4. `doser1/HANDOFF.md` (1k tokens or fewer) and `doser1/out/paths.json` (`[repo path, abs src, "raw"]`, repo dir `.../2026-10-19-search-p2-confirm/dose/`).

## Hand-back (200 words or fewer)
The `file:line` of each change, the counts, the seal command, and the manifest.
