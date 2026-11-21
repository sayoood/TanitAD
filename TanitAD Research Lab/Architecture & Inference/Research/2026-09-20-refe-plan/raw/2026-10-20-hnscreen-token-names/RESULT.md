# HNSCREEN-TOKEN-NAMES: the screen chain reads G14 / G15 from checks a real token carries

Base: tip `065744da119ac9b448c25ae92fc5e6dad8ae7624` (`agent/arch-inf-20260803`, HNFP3 landed). CPU only, nothing committed or pushed. Evidence class: MEASURED (ours, the logs in `logs/`), all on the repo exported at the tip.

## The defect, re-verified at the tip (line numbers moved with HNFP3)

* At the tip, `refe/hnscreen_devbox.py:79` (a literal in `DEFAULTS`): `gate.token_check_names` = `{"G14": "G14", "G15": "G15"}`. `token_checks()` (`:731-735`) returns False on a missing key; `assemble_gates` (`:743-744`, fallbacks `.get("G14", "G14")`) reads G14 / G15 through it.
* The real `refe_hnscreen` profile (`refe/launch_gate.py:370`, `checks` inherited from `refe_rls1` at `:316`) has `checks = (G-HYG, G-DVB, G-PREFLIGHT, G-SUITE)` and `na_ok = (G-CLOCK,)`. `launch_gate.finalize` writes exactly those five keys into the token (the loop at `:1217`). G14 / G15 are judged INSIDE G-PREFLIGHT (`launch_gate.py:1134-1146`: `judge_g14` at `:1139`, `judge_g15` at `:1141`), required by the argv (`gate_checks.py:1043-1046`: G14 iff `--hn-bank`, G15 iff `--ctx-cache`).
* MEASURED probe (one real `gate_token.mint`, the profile's check list, every check PASS, seeds 0-1, `assemble_gates`): **tip source -> G14 = False, G15 = False** (every other gate True); **fixed source -> G14 = True, G15 = True**. The tip's chain tests passed only because their STUB token wrote checks named G14 / G15.

## The change

| file | where | what |
|---|---|---|
| `refe/hnscreen_devbox.py` | `:74-79` | new module constant `TOKEN_CHECK_NAMES = {"G14": "G-PREFLIGHT", "G15": "G-PREFLIGHT"}` with the reason |
| `refe/hnscreen_devbox.py` | `:86` (the fixed file; `:79` at the tip) | `DEFAULTS["gate"]["token_check_names"] = dict(TOKEN_CHECK_NAMES)`. The config override is kept: `_merge` still merges it key by key |
| `refe/hnscreen_devbox.py` | `:750-751` | the `names.get("G14", ...)` fallbacks read `TOKEN_CHECK_NAMES[...]`, not a second copy of the old literal |
| `refe/test_hnscreen_chain.py` | `:44-47`, `:160`, `:274` | the stub gate now writes the checks of the REAL profile: `launch_gate` is imported (not copied), `REAL_TOKEN_CHECKS = PROFILES["refe_hnscreen"]["checks"] + na_ok`, passed to the stub by `STUB_TOKEN_CHECKS` |
| `refe/test_hnscreen_chain.py` | `:702-704` | NEW mutant: the pre-fix default pair -> `A_stage_order_and_markers` RED |
| `refe/test_hnscreen_ctx_chain.py` | `:196` | the explicit `gate.token_check_names` override is removed: that suite mints through the real `finalize`, so it now runs on the DEFAULT with a REAL mint |
| `refe/test_hnscreen_ctx_chain.py` | `:352-355` | NEW mutant: the pre-fix default pair -> the end-to-end arm RED |
| `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py` | new | the invariant test, the real-token test, the RED arms, 7 mutants |

## The new test (`ZZTEST_PASS 23` = 16 checks + 7 RED mutations; 1.2 s)

* **A, the invariant.** Every VALUE of the chain's effective `gate.token_check_names` (a `Cfg` with its defaults merged) is in `launch_gate.PROFILES[cfg.gate.profile]["checks"]`, with the REAL `launch_gate` imported. The keys are exactly G14 / G15. An override wins and merges key by key; a bad override (`G-NOPE`, an unknown profile, a stray key) is caught by the same function; the override reaches `assemble_gates`.
* **B, the two RED arms of the invariant.** (1) The old default pair fails it (2 problems, G14 and G15). (2) A profile whose checks lack `G-PREFLIGHT` fails it (2 problems), while the real profile gives `[]`.
* **C, not only membership.** `G-HYG` is also a member of the profile's checks. The test reads the AST of the REAL `launch_gate.py` and requires the named check to be the one whose evidence function calls both `judge_g14` and `judge_g15` (`{"G-PREFLIGHT"}`), and `scorer_required_gates` to oblige G14 / G15 for `--hn-bank` / `--ctx-cache` (and not without them).
* **D, the real-token test.** `gate_token.mint` with the real profile's check list, every check PASS -> `assemble_gates` reads all ten gates True (G14 = G15 = True). With G-PREFLIGHT FAIL / MISSING / REJECTED / INCOMPLETE or absent -> G14 = G15 = False and the other gates untouched; one bad pass or one missing token is enough (`all`); G15 still needs its own evidence file. (The odd tokens use `verdict_override="PASS"`, because the real gate names a token with a non-PASS check `FAIL_*`, which the chain never looks at; the arm tests the chain's own status read.)
* **E.** The chain test's stub token carries exactly the checks of a real token (`{G-HYG, G-DVB, G-PREFLIGHT, G-SUITE, G-CLOCK}`), not G14 / G15.
* **Mutants (each edits `hnscreen_devbox.py`, snippet exactly once, every expected check goes RED):** the pre-fix pair (the defect itself); a half fix (G15 only); both gates read from `G-HYG` (membership passes, only C and D catch it); the override ignored; `!= "FAIL"` instead of `== "PASS"`; G15 ignoring its evidence file; `any` instead of `all`.
* **The new test is RED on the tip's source.** `HNTOK_PKG=<tip export>`: the unmutated suite FAILS 10 of 16 checks, including `D_a_real_token_reads_G14_and_G15_true`.

## Before / after (MEASURED, `logs/base_*` and `logs/after_*`; `before` = the tip export, `after` = the fixed tree; same interpreter, `CUDA_VISIBLE_DEVICES=-1`)

| suite | before | after | note |
|---|---|---|---|
| `refe/test_hnscreen_chain.py` | `ZZTEST_PASS 16` (12 checks + 15 RED mutations) | `ZZTEST_PASS 17` (12 + 16) | +1 mutant |
| `refe/test_hnscreen_ctx_chain.py` | `ZZTEST_PASS 19` (9 arms + 10 mutations) | `ZZTEST_PASS 20` (9 + 11) | +1 mutant; runs on the DEFAULT now |
| `eval/test_hn_screen_read.py` | `ZZTEST_PASS 36` | `ZZTEST_PASS 36` | unchanged code |
| `refe/test_hn_ctx_anchor.py --no-mutants` | `ZZTEST_PASS 38` | `ZZTEST_PASS 38` | unchanged code; the 72-count full matrix was not re-run |
| `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py` | RED on the tip source (10 of 16 failing) | `ZZTEST_PASS 23` | new |

`test_launch_gate.py` was not run: `launch_gate.py` is untouched.

## Observations (not changed, for the MM)

1. **Proposed hardening, ~5 lines, beyond the brief:** `stage_T` could refuse right after the mint if `token["checks"][name]` is absent for a name the chain will read. That would fail an operator's wrong `gate.token_check_names` override at the first token (before any GPU spend) instead of at SCREEN-INVALID. The invariant test pins the default and a config; it cannot see a run box's own config file.
2. `Cfg._merge` aliases the module-level `DEFAULTS` for any key the config does not override (`out = dict(base)` is shallow), and `Cfg.__init__` then writes `trainer.cmd` / `gate.cmd` into it. Harmless for one `Cfg` per process; it is why this test does not mutate a `Cfg`'s dicts.
3. The chain tests for the stub-based suite still use `ctx_mode: content`; the anchored suite is the one that exercises the default with a real mint.

## Proposed register rows (the MM lands them; nothing was edited)

* `BACKLOG.md` `HNSCREEN-TOKEN-NAMES`: DONE 2026-10-20 (staged by `par18/hntok`): default `{"G14": "G-PREFLIGHT", "G15": "G-PREFLIGHT"}`; `raw/2026-10-20-hnscreen-token-names/RESULT.md`.
* `GOALS_AND_CLAIMS.md`: a launch blocker of the P6-HN screen is closed; the test that would have caught it is `test_hnscreen_token_names.py` (real `launch_gate.PROFILES` + real `gate_token.mint`, 23 / 23). Class: built, tested, unreachable from its caller (a stub that agreed with the default instead of with the real producer).

## Integration owed

The Master Mind lands the 14 repo files in `par18/hntok/out/paths.json`: three EDITS (each with `base=<tip blob sha>`: `hnscreen_devbox.py` `2bd171c1`, `test_hnscreen_chain.py` `20cec616`, `test_hnscreen_ctx_chain.py` `29cc4472`), the new test, this file and 9 logs. The three edits were made on the tip's exact blobs (verified by `git hash-object` before editing), so a base-guarded land is a clean replace. Run-box prerequisite unchanged: the first full chain on the run box with `gate.ctx_mode` unset, reading `ctx_chain/end_verify.json`.
