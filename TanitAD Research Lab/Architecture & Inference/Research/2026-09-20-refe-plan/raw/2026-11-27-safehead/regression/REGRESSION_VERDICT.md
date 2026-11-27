# L2 SAFE-HEAD launch-gate regression: base vs patched (MM, 2026-11-27)

The comparison covers the 56 REFe tests that touch `launch_gate`.
- **BASE** = tip `1df04b58`, PK/refe unpatched. Its files are blob-identical to tip `3f689216` for every path that changes; checked with `git hash-object --no-filters`.
- **PATCHED** = BASE plus the 5 SAFE-HEAD files, `launch_gate.py.patch` and `test_profile_pins.patch`.
- Tests ran SERIALLY, CPU only, with TMP on F: and each test gated on commit free ≥ 8 GB (`cmp_chain.sh`).

## As run: `regressions 2` (`COMPARE.tsv`)

| test | base | patched | cause |
|---|---|---|---|
| `test_rep_gate.py` | PASS | FAIL (`AssertionError: 43`) | **a pin-patch arithmetic slip.** The patch added the 3 new profile names to the subtracted set, which leaves the unlisted count at 19, but it also changed the expected count from 19 to 22. `len(PROFILES)` 40 → 43 is correct. |
| `test_scorerft_profile.py` | PASS (526) | FAIL | **a census pin.** `chk_existing_unchanged` requires the profile set to equal exactly the known profiles. The 3 new SAFE-HEAD profiles were not added to it. No existing fingerprint moved. |

Neither failure is a behaviour change. Both are census pins that must move when profiles are added. The other differences:
- `test_hn_content_fp.py`: FAIL on base, PASS on patched. Recorded, not claimed as a fix.
- `test_safehead.py` is new and passes 14/14.
- `test_vdac_stage0_forward.py` fails on both, as in the backlog.

## Fix and re-run (MM, 2026-11-27 ~17:10Z)

1. `test_rep_gate.py`: `== 22` → `== 19`. The three new names are listed, so the unlisted count stays 19, which is what the base asserted.
2. `test_scorerft_profile.py`: the allowed profile set gains `SH_NAMES = {refe_safehead_red, refe_safehead_s0, refe_safehead_s1}`. Every `FP_PINS` fingerprint check is unchanged.

Re-run serially on the PATCHED sandbox, gated on commit free:

| test | result |
|---|---|
| `test_rep_gate.py` | rc 0, ZZTEST_PASS (23 s) |
| `test_scorerft_profile.py` | rc 0, ZZSCORERFTPROF_PASS 526 (135 s) |
| `test_safehead.py` | rc 0, ZZSAFEHEAD_PASS 14 (146 s) |

**Amended verdict: 0 regressions.**

The five SAFE-HEAD files landed under `PK/refe/` have the same sha256 as the copies hashed by the frozen card's `CODE.sha256` (`PREREG_SAFEHEAD.md` sha256 `3e2c69b4…`). The two test edits touch test files only. No profile fingerprint and no `launch_gate.py` pin moves.
