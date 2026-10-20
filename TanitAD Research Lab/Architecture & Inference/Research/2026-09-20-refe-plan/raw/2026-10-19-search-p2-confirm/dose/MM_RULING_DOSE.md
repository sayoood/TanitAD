# MM ruling on the SEARCH-DOSE amendment draft (Master Mind, 2026-10-20 ~03:35; C1 has produced no B-REST bundle, search output or label)

**DQ1: (a), run the pinned O4 config as ruled. RATIFIED.**
- The contrast is named "O4 minus C8", never a dose-response in M.
- The M-only arm is a named follow-up under its own amendment, and only if the dose reads O1 or O2.

**DQ2: (a). RATIFIED.**
- Each arm keeps its committed bar.
- max(C1, dose) is never quoted as an effect size.
- Every dose line is printed with "SECOND READ OF THE SAME TOKENS".
- Navtest (C2) is the confirmation for whichever config goes there.

**DQ3: (i) run, (ii) as written, (iii) do not run. RATIFIED.**

**DQ4: (b), plus a MECHANICAL config rule committed now, so that no outcome of either read can shape it.**
- `H-SEARCH-P2-1` gets exactly ONE navtest look, with ONE config.

| C1 licence (`MM_RULING_C2_LICENCE.md` terms) | dose licence | config sent to C2 |
|---|---|---|
| met | not met | **C8** |
| not met | **O1_CONFIRMED** under this amendment, AND each safety point >= -0.25, AND moved share >= 5 % | **O4** (O2 is NOT enough on this path: it is the second look at the same tokens) |
| met | met (O1 or O2 + the safety and moved-share terms) | **C8**, unless the paired O4-minus-C8 contrast (per token and seed, the C1 estimator) has its lower CI > 0; then **O4** |
| not met | not met (or O2 only) | no C2; the committed next arms stand |

- The dose's inference cost (about 2.9x per token, ESTIMATED) goes into C2's amendment, MEASURED on the deployment target or named UNMEASURED, for whichever config is sent.

**DQ5: (b), ADDED, plus one more gate. Both are built, tested and pinned BEFORE the seal:**
1. **G-SELFREP (self-replication):** a second process re-runs the first 8 B-REST tokens with the O4 config; it must be bit-equal to the merged output. This catches thread-count nondeterminism (`MM_RULING_SPC.md` DQ8).
2. **G-SUBID (subclass identity):** `dose_chain.py` run with the **C8** config on the first 16 B-REST tokens must reproduce C1's banked search outputs bit for bit. This proves the subclass does not change the search path, which is the only identity the O4 run can inherit from C1's C-ID.
- Each gate gets a RED arm (a planted difference must make the chain refuse) and enters the mutation check.
- Cost: about 1.5 + 3 min CPU.

**DQ6: (a). RATIFIED.** Pin the new files in this seal; C1's files stay byte-identical to their pins.

**Not licensed:** unchanged (§11 of the draft). Sealing is the MM's: after the agent's final draft and a re-run of the suite, the mutation check and the seal script.

**Clarification (MM, 2026-10-20 04:12, before the seal, raised by the drafting agent):** `MM_RULING_SPC.md` DQ7's "descriptive and ungated" stands. The O4-minus-C8 contrast decides no OUTCOME of either read. Its only use is the CONFIG CHOICE in row 3 of the DQ4 table, where it is read with C1's estimator (pooled over seeds, paired per token and seed).
