# MM ruling on the four points the C2-final hand-back raised (Master Mind, 2026-10-20 ~04:55; C1 has produced no output; the dose has not run; no navtest label has been read)

**1. "Moved share" has ONE meaning everywhere: `inert.frac_moved`.** That is the share of pooled (token, seed) outputs whose ADE vs the pick exceeds 0.5 m, the quantity whose shortfall IS O4_INERT.
- This was already ruled for the C2 licence (`MM_RULING_C2AMEND.md` DQ2). It holds equally for "moved share" in `MM_RULING_DOSE.md` DQ4's table and in `MM_RULING_C2_LICENCE.md` §2.
- The sealed dose code's `licensed()` reads `moved_share`, the mean of the search's `moved` flag: any non-zero move. That is a DEFECT against the ruling, found before any read.
- **Fix: Revision 1 of the SEARCH-DOSE amendment, code only.**
  - `licensed()` and its callers read `inert.frac_moved` from each read's own `inert` block.
  - The terms text stays byte-identical, plus a dated REVISION note naming this ruling.
  - A test with `moved_share >= 5 %` and `frac_moved < 5 %` must read NOT licensed, and the mutation check must cover it.
  - Re-seal, then re-run the sealed dry run.
- `moved_share` stays reported, descriptive only.

**2. C2 waits for the dose read. RATIFIED as the table requires.**
- Every row of the DQ4 table depends on the dose's licence, so the config is undefined until the dose reads.
- The dose is CPU only, about 3.4 h ESTIMATED after `ZZSPC_DONE`; the delay is accepted.
- If the dose cannot run (C1 never reaches `ZZSPC_DONE`), there is no C1 licence either, so no C2.

**3. Test against the REAL dose file.** `dose/code/dose_chain.py` is on the tip since `51078a6`. Keep the double, and ADD a test that imports the real file by path under the sha pin and checks that it agrees with the double on the grid. The pin follows Revision 1.

**4. An infrastructure stop does not consume the look. RATIFIED, under three conditions:**
- the resume carries the IDENTICAL identity (config, every code pin, the C1 and dose result shas, the token list);
- no analysis output of the stopped run is read before completion;
- the marker records every stop and resume.
- A resume with any identity change refuses.
