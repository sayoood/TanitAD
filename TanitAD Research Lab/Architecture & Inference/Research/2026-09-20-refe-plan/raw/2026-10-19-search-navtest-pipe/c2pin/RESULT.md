# c2pin: C2 bound to the REAL dose code at Revision 1 (Master Mind brief 2026-10-20 ~06:10; tip `fa7d856`; CPU only; nothing committed, pushed, or git-added)

**Counts:** full navpipe suite **380 passed** (baseline 357; `raw/pytest_c2pin_final.log`). Mutation, new/changed guards plus four older guards whose tests were rewritten: **12 of 12 caught**, control `CONTROL_GREEN` (`raw/mutation_check_c2pin.log`, `raw/mutations_c2pin.json`). The earlier 114-guard run is NOT re-run in full.

**Integration needed (headline):** the Master Mind lands the 7 edited files in `out/paths.json` BEFORE sealing. The draft's terms text changed (table row, section 6.3 bullet, section 10 row, pin notes, section 13), so the seal tool must run on the new draft; any earlier seal dry run is stale. `dose_chain_sha256` stays the seal placeholder; its expected value is `b41ccab5a24fa67c4d07c3aa4dce1b60175c97ad9787f513e91491193dccc217`.

## Finding (MEASURED here, mutation `G-C2CONFIG-INERT`)
`select_config` handed the dose chain's `c2_config` only `outcome`, `safety_points`, `moved_share`. At Revision 1 `licensed()` reads `inert.frac_moved` from each read's own `inert` block, so with that call every read has frac_moved NaN, nothing is licensed, and every C2 attempt would have refused at `C2_CONFIG_NONE`. The test double (pre-R1 semantics) could not show this. Fixed: the call now passes `inert` too, exactly the four keys `dose_chain.check_c1_done` returns.

## Changes (`file:line` in the staged files)
- default dose path / R1 names: 46
- check_dose_chain_pin: 282
- check_dose_package_r1: 292
- select_config inert keys: 405
- check_c2_seal R1 call: 609
- seal tool R1: 62
- mutation overrides: 47
- double + planted R1 package: 186
- real-file tests: 260
- seal-tool tests: 218
- draft pin notes: 226
1. Default dose path: it was already repo-relative (`raw/2026-10-19-search-p2-confirm/dose/code/dose_chain.py`); the stale "UNVERIFIED, not on the tip" comment is replaced. `C2_DOSE_CHAIN` moves the path only.
2. `check_dose_package_r1` (before the import): `AMENDMENT_DOSE_R1.md` + `.sha256` present and equal, `Status: **SEALED`, and its PINS block pins `code:dose_chain.py` to the file being imported. Refusals `DOSE_R1_SEAL_MISSING` / `_SHA_MISMATCH` / `_NOT_SEALED` / `_NO_PINS` / `_CODE_PIN_MISMATCH`. The seal tool runs the same check before filling `dose_chain_sha256`. The selection record banks `dose_r1_amendment_sha256`.
3. Tests: the double now has Revision 1 semantics and lives in a planted R1 package; a pre-R1 double is kept only to reach `G-LICENCE-ROW-AGREES` and `G-CONFIG-NONE`, which a real R1 file can no longer trigger. The real-file test lost its skip and its dev-box fallback; it pins `b41ccab5...` explicitly.
   - Grid: the old 63 x 63 x 5 grid (moved value in both fields) plus a disagreement grid (moved_share vs frac_moved), double == real on `config`, `row` and `licensed`.
   - `moved_share >= 5 %` with `frac_moved < 5 %` is refused through C2's path against the real file, on C1 (`LICENCE_INERT`), on the dose-only path (`LICENCE_INERT`) and on the dose read of row 3 (`CONFIG_NOT_C2_CONFIG`).
   - RED arms: 12 R1-package arms (no R1 files, no sidecar, no text, edited text, wrong or empty sidecar, DRAFT status, no PINS block, no or wrong code pin, edited `dose_chain.py` byte), a byte-exact and a CRLF control, the `check_c2_seal` call path (edited byte refuses with the pin filled from the edited file, and with the real pin), and two seal-tool arms.

## Not done / caveats
- Evidence class: all counts MEASURED here on planted worlds plus the real dose package from tip `fa7d856`; no C1 or dose result exists, no navtest label was read.
- `l1_stepA/raw/g0_traingrow_scan2.json` (6.6 MB) and the 70 `.npz` banked shards of `l2-stepB` were not exported (disk); the suite does not read them.
- A full 125-guard mutation re-run was not made; the four older guards chosen are those whose covering tests I rewrote.
