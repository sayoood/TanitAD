# BRIEF: bind C2 to the REAL dose code at Revision 1 (Master Mind, 2026-10-20 ~06:10)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/c2pin/`.
- CPU only. C: has about 0.35 GB free: TEMP / TMP and basetemp under your work dir, delete only what you create. If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push. No navtest labels, no B-REST data.

## Start
Tip `fa7d856` (mirror `C:/Users/Admin/tanitad-push`; run `git fetch`, then pin the tip once). Export, keeping the repo layout:
- `.../2026-09-20-refe-plan/raw/2026-10-19-search-navtest-pipe/`
- `.../raw/2026-10-19-search-p2-confirm/` (it holds `c2/` and `dose/`, at Revision 1: `dose/AMENDMENT_DOSE_R1.md`, `dose/code/dose_chain.py` sha256 `b41ccab5…`)
- `.../raw/2026-10-19-l2-stepB/`
- `.../Research/2026-10-18-ep-rl-lit/l1_stepA/`

Then read `c2/MM_RULING_C2FINAL.md` point 3 (binding) and `c2/AMENDMENT_C2_DRAFT.md` (FINAL DRAFT).

## Do
1. **The C2 code's dose import:**
   - Point the default dose path at the repo-relative `dose/code/dose_chain.py`.
   - Make the `@@SEAL@@` pin for it the Revision 1 file. Leave the placeholder for the seal tool, but state the expected value `b41ccab5…` in the draft's pin notes.
   - C2 must also check that the dose package it imports is sealed at R1: `AMENDMENT_DOSE_R1.md` + `.sha256` present and matching. Refuse otherwise.
2. **The test against the REAL file:** import the real `dose_chain.py` by path, and assert that its `c2_config` agrees with the test double on the whole grid the double was validated on.
   - Also add the `moved_share` vs `frac_moved` disagreement case: a read with `moved_share >= 5 %` and `frac_moved < 5 %` must NOT be licensed through C2's path.
   - RED arms: a dose package without the R1 files refuses; an edited `dose_chain.py` byte refuses.
3. Run the FULL navpipe suite (baseline 357) and a mutation check on the new guard lines only (`mutation_check.py --only <names>`). Report both counts.
4. Manifest `c2pin/out/paths.json`: `[repo path, abs src, "raw"]`, with `"base=<the tip's blob>"` on every edited existing file. Put a short `RESULT.md` under `.../search-navtest-pipe/c2pin/`. Write `HANDOFF.md` (800 tokens or fewer).

## Hand-back (150 words or fewer)
The `file:line` of each change, the counts, and the manifest.
