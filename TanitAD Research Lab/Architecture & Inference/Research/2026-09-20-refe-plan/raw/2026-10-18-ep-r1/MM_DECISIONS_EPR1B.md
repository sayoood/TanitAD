# EP-R1b: Master Mind decisions (2026-10-19, before any dev or navtest outcome exists)

These answer `RESULT_EPR1B.md` section 7. No dev read and no navtest look has been spent. Both ledgers are unwritten.

**D4. Navtest feature bank (5.43 dev-box GPU-h): BUILD IT BEFORE THE DEV READ.**
- The bank is outcome-free: no look is spent, and no label or PDMS is opened. "S3 only on a dev pass" was a COST rule.
- The cost argument no longer holds. After S0, the dev-box GPU's queue holds only the items 3+4 GPU tests (minutes) and L1 Step A (0.25-0.4 GPU-h). After that it idles, and the PI declined extra compute, so idle dev-box time is the free resource.
- One bank serves all three readings (SR-TTC alone, EP-R1 alone, the joint) and any later head-row arm, including L2.
- **Order on the dev-box GPU, each under the lock:** S0 → items 3+4 GPU tests → L1 Step A → **the 4-token DG-DUMP pre-check on one of the 4 logs with the 36 frame-control tokens** → the full bank.
- If the pre-check's dump differs from the banked table there, C4 is RED and the full bank does NOT run. Fix that first. No waiver is registered and none is granted here.

**D5. SR-TTC dev applier: BUILD ONE, against SR-TTC's OWN registered bar.**
- `selrank_dev.py` is a descriptive attribution, not a gate. The gate must implement `PREREG_SELRANK_ARM.md` and its amendments, NOT EP-R1's bar.
- It must write the contract `refe_head_confirm.read_dev_result` reads: `dev_gate.verdict` in PASS/FAIL/VOID, `inputs.rows_sha256` per seed, `inputs.r1_result_sha256`.
- A `--col 3` variant of `epr1_dev.py` is acceptable only if every bar term in it is traced to the SR-TTC registration, with file:line.

**D6. The JOINT rule, as implemented: CONFIRMED.**
- JOINT iff EP-R1 passed its dev gate AND SR-TTC was dev-read (not abandoned) AND the fit logs are identical. This matches PREREG_EPR1 section 8: EP-R1 is on the navtest look only if it passed its dev gate, and the joint is SECONDARY to it.
- The alternative (JOINT when only SR-TTC passes) is REJECTED. It would read a secondary whose own primary is off the look.

**D7. Dev C2 strictness:** as registered, with no tolerance (none was registered, and none is added after the fact).

**D8. Before the look opens, two pieces must be BUILT (launch blockers for the read, not for the bank):**
- the navtest C2 reading;
- `families6` on the new seams, with all four metric families: LONGITUDINAL, LATERAL, TACTICAL, and STRATEGIC printed as PAUSED (PI item 25). A family that cannot be computed (e.g. distance-keeping) is reported with its reason and its n.
- An ADE-only or PDMS-only confirm report is incomplete under the binding four-family rule.
