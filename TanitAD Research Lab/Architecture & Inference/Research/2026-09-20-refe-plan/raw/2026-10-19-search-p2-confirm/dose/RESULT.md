# RESULT: SEARCH-DOSE amendment, FINAL DRAFT after MM_RULING_DOSE.md (2026-10-20). No B-REST token touched; no outcome exists.
- Built per the ruling: G-SUBID and G-SELFREP (bit-for-bit replicates, RED arms, in the mutation check), `c2_config` (the committed DQ4 table, `ZZDOSE_C2CONFIG`), the ruling pinned in the seal. Amendment folded: section 5b = the ruling's table verbatim, section 10 RULED.
- Counts: test_dose.py 112 passed; spc suite 115 passed (C1 files untouched, manifest 0 mismatches); mutation check 65/65 caught, control passes.
- Proposed register rows (the Master Mind lands them; this agent edits no register):
  - `H-SEARCH-DOSE-1` (HYPOTHESIS, not read): the O4 config on the SAME B-REST bundle clears C1's outcome rules; amendment FINAL DRAFT, unsealed; about 3.4 h CPU, ESTIMATED.
  - `D-SEARCH-DOSE-REACH-1` (MEASURED, doseamend/code/test_dose.py): spc_chain.py hard-wires C8 at :45/:159/:278/:293/:325; the O4 search runs through dose_chain.py with M = 256 observed in the search loop.
  - `D-SEARCH-C2CONFIG-1` (RULED, MM_RULING_DOSE.md DQ4): H-SEARCH-P2-1 gets ONE navtest look with ONE config by a mechanical table of C1's and the dose's reads.
  - Blocker row: D: had 0 bytes free; C1 (12 GiB gate) and the dose cannot write.
