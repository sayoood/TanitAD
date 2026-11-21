# E0i card -- eval139 DEV / HOLD split (REFCV9_DESIGN §4.1, PI decision D6 of 2026-11-02)

**Question.** Which eval139 episodes may E1 / E2 / D1 read (DEV) and which are reserved for the sealed bars (HOLD)?
**Frozen rule.** Order the 139 episodes by `sha256(clip_id)` hex ascending (the `l1c_select.py` convention); rank even = DEV, rank odd = HOLD. Frozen before any E1 read.
**Instrument.** `e0i_split.py` (inputs: the kit's `s2_windows.json`, `s6_windows.json`; clip ids are read in process, the output carries sha12 only), `test_e0i.py`.
**Known-answer test.** `sha256("a"|"b"|"c")` start `ca97…`, `3e23…`, `2e7d…` (literals); ascending order is c, b, a, so DEV = {c, a}, HOLD = {b}. 5 tests PASS.
**Controls on the real data (all PASS at build).** 70 + 69 = 139 episodes; DEV and HOLD disjoint; S2 and S6 window counts sum to the grid totals (4,754 / 3,668); the clip set equals the v9 label release's `clip__sha12` set; all 139 sha12 prefixes are unique.

**Result (MEASURED, `e0/DEV_HOLD_SPLIT.json`, file sha256 `ca5af0203320aba4f89e09be81a6bf079d9d4d77708ccb8a3ca2a0ea57e8db0a`, `lists_digest` `417f55c9eb5958ae…`).**

| half | episodes | S2 windows | S6 windows |
|---|---|---|---|
| DEV (even rank) | 70 | 2,397 | 1,848 |
| HOLD (odd rank) | 69 | 2,357 | 1,820 |

**Limits.** The split is by episode, so the ~√2 wider CIs of D6 apply. It does not balance scenario mix (no stratification was asked for); a per-half S6c / sidewalk-window count is the first thing E1 should print.
**Evidence class.** MEASURED (ours). Tier: n/a (a protocol artifact).
