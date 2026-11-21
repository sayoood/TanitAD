# MM ruling on the C2 (navtest) amendment draft (Master Mind, 2026-10-20 ~04:00; C1 has produced no B-REST output; no navtest label has been read)

This ruling fixes the amendment's TERMS now, before C1 reads. The later seal only adds code pins: the terms text in the seal must be byte-identical to what this ruling fixes.

**DQ1: (a), RATIFIED.**
- C1's result is bound by `--c1-result-sha256` typed on the command line, plus an optional pin.
- When C1 reads, the MM writes `RESULT_SPC.json`'s sha256 into `H-SEARCH-P2-1` in the same turn. That register entry is the independent record the typed value is checked against.

**DQ2: (a) `inert.frac_moved`, RATIFIED.** This is the meaning of "not INERT" in `MM_RULING_C2_LICENCE.md`: ADE vs the pick > 0.5 m on ≥ 5 % of the pooled (token, seed) outputs.

**DQ3: (a), RATIFIED.**
- At seal, the cost status is UNMEASURED, with the §7 protocol committed.
- Any C2 PASS is worded as "a selection-tier gain on navtest; deployability UNMEASURED" until the Thor number exists.
- Writing `thor_search_latency.py` is CPU work and starts now. Running it needs a Thor slot.

**DQ4: (a), RATIFIED.**
- Report ms per token (p50 / p95 / p99) and the ratio to P2's pick-only forward, with no pass/fail.
- An absolute latency budget is an open architecture decision, logged as one. Until it exists, no claim of deployability.

**DQ5: §11's order, RATIFIED, with one constraint.** A safety-family miss is printed IN THE HEADLINE LINE whenever present, whatever the outcome; it is never only in a table.

**DQ6: (b), re-score every P2 pick through the identical route. NOT the recommendation.**
- **Why:** C2 is a one-time look, and its endpoint is a PAIRED delta. Both arms must be scored by the same code on the same harness by construction, not by a 336-token sample identity.
- 18 minutes of harness time is cheap against a biased primary endpoint.
- Keep the planted identity against the banked CSV as a CROSS-CHECK. Any mismatch is reported and stops the chain until diagnosed.
- Build and test this before the seal.

**DQ7: (b), code-checked, built BEFORE the seal. NOT deferred.**
- The fixture chain writes a fixture record: verdict, the code sha256s it ran, its token list sha.
- C2 refuses unless that record reads IDENTICAL, and its code shas equal the sealed pins.
- An operator-checked precondition is exactly the class the launch-gate rule exists to remove.

**DQ8: (a), RATIFIED, with (b) as the fallback.**
- Seal when the landed code is final AND the fixture record reads IDENTICAL; before C1 reads if the disk allows.
- If C1 reads first, seal anyway and record the order. The terms are already fixed by this ruling's sha.
- Re-pin policy: a "Revision n" may re-pin code only, with the terms text byte-identical, as C1 did.

**DQ9: YES.** Build `seal_c2_amendment.py`, modelled on `seal_spc.py`. It is pinned, tested, and has a RED arm: a pin left as a placeholder must refuse.

**DQ10: (a), the dev-box GPU, RATIFIED.** If the bridge fails, diagnose before changing venue. Moving to a pod is a new MM decision, never an automatic fallback.

**DQ11: ON, RATIFIED.** Only `token`, `proposals`, `logits` and `pick` are read; the spy test that proves no label array is touched stays pinned.

**ADDITION (from `MM_RULING_DOSE.md` DQ4): the config sent to C2 is chosen mechanically.**
- C2 must accept exactly ONE of two pinned configs: C8 (`stepB_config.json`) or O4 (`stepB_config_O4.json`, `1ebd7196…`).
- The choice comes from `c2_config(c1_read, dose_read)` in the dose package.
- When O4 is selected, the dose result is bound like C1's: a typed sha256, plus the check that it is the dose read on B-REST.
- C2 refuses if the config it is given is not the one `c2_config` returns.
- Exactly one C2 run per `H-SEARCH-P2-1`. A second navtest run of any config is refused by a marker.
- **Cost:** with O4 selected, the GPU stage is unchanged (the forward is the same), but the CPU search and the Thor latency scale (about 2.9x per token, ESTIMATED). The amendment states both.

**Not licensed:** no navtest look before the seal, the fixture record and C1's licence. No change to the DQ3 bar numbers.
