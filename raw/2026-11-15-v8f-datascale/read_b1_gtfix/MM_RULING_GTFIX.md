# MM ruling on the GT-block fix (2026-11-17 ~22:45Z)

1. **The fix is accepted as an instrument fix.** The refusal was not a label difference. Each checkpoint writes the GT constraint block in its own SD units (`(cons - cons_mu) / cons_sd`, with the stats from the arm's own training labels). In physical units U and B agree to 2.5e-5, and which slots are present is identical (MEASURED, `probe_gt.json`).
   - Converting into the reference arm's units, with a refusal beyond 1e-5 SD or on any change in which slots are present, keeps every earlier result bit-identical.
   - The fix comes with 4 tests and 3 mutations that read RED.
2. **Units of record: the reference arm's (B / M2b) units decide.** They are the units of every banked mechanism number in this study. B_rep shares them, so the replicate floor is read in the same units.
   - The physical-unit value is printed beside as a sensitivity row. At seed 0, r is −3.14 % (B units), −3.12 % (U units) and −1.98 % (physical); the label is the same under all three.
   - **If any later label differs between the unit choices, it is reported as UNIT-SENSITIVE with all three values and is not quoted as a single label.**
3. **The reader step ran at F: 23 GB, below the card's 25 GB bar.** Accepted for this one step only: the reader pulls nothing and writes under 1 MB, and the bar exists for the pulls and dumps. The bar stands for every step that pulls or dumps, including the block-2 read.
4. **Seed-0 reading (no label; the block-1 read is the seed-0 family and mechanism read, MM ruling headline 3):**
   - Precondition, R vs U: REACHED, r +34.1 %, separated.
   - U vs B: not T-DATA-LIMITED, r −3.14 %, not separated.
   - The deciding label waits for block 2 (seed 1).
