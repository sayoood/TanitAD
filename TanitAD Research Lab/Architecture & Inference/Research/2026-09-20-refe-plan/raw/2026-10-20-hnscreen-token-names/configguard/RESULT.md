# HNSCREEN-TOKEN-NAMES-STATIC: the chain REFUSES at config load, before S0, when a `token_check_names` value is not a check of the configured profile

Base: tip `2f1969a` (`agent/arch-inf-20260803`; mirror `C:/Users/Admin/tanitad-push`, fetched and pinned once). CPU only, nothing committed or pushed. Evidence class: MEASURED (ours, `configguard/logs/`), all on a repo export at the tip
(the 3 edited files were hash-identical to the tip blobs before editing: `hnscreen_devbox.py` yes, `test_hnscreen_chain.py` yes, `test_hnscreen_ctx_chain.py` yes).

## The change (`refe/hnscreen_devbox.py`, fixed-file line numbers)

| where | what |
|---|---|
| `:949-973` `Chain.token_names_config_problems()` | NEW, pure. Imports the REAL `launch_gate` (from this file's directory, `HERE` put on `sys.path` if absent; never a copy of the list). Problems, each `TOKEN-NAMES-CONFIG: ...`: `launch_gate` unimportable (fail closed); `gate.profile` not a str or not in `launch_gate.PROFILES`; `gate.token_check_names` not a JSON object; and per entry of `{**TOKEN_CHECK_NAMES, **override}` (a stray key counts, as in the mint guard) a VALUE that is not a str in `PROFILES[profile]["checks"]`. |
| `:981-983` in `Chain.preflight()` | `bad = self.token_names_config_problems()`; if any: `ChainStop(RC_CONFIG, ...)` = rc 4, log `ZZHNCHAIN_CONFIG rc 4: TOKEN-NAMES-CONFIG: ...`, status CONFIG. It sits after the missing-parameter and `ctx_mode` checks and BEFORE `os.makedirs(root)`, before `config.sha256` / `config.frozen.json` are written and before any stage. `run()` calls `preflight()` before the stage loop, so S0 never starts. |
| docstring `:8-10` | says so |

Why `preflight` and not `Cfg.load`: `Cfg` is pure data (no import, no side effect), `preflight` is the one place that already refuses a bad config with rc 4 and that precedes the freeze, so a refused config freezes nothing and the corrected config relaunches into the SAME state dir (tested).

## What it protects, and what it does not

* The mint guard (the previous follow-up) could not fire before S2 (the GPU trunk cache, about 19,000 tokens x 1.35-1.65 s) and T0's preflight, because the token does not exist earlier. This check needs no token, so a wrong run-box override now costs nothing: no S0, no S1, no S2.
* **It judges MEMBERSHIP only.** An override naming `G-HYG` passes it, by design: `G-HYG` is a real check of `refe_hnscreen`. That is acceptable here because (a) the mint guard still stops the chain at the first token unless that check is PASS in it, and (b) which check really judges G14 / G15 is already pinned from the AST of `launch_gate.py` and with a REAL minted token by `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py` (check C / D). Tightening to "must equal G-PREFLIGHT" would be a different, narrower rule and would forbid a legitimate future profile.
* `plan` does not call it (it builds a `Chain` but never `preflight`); `run` is the only path that starts stages. Not changed (outside the brief).
* Imports the `launch_gate.py` next to the chain, not `<pkg>/refe/launch_gate.py` of the config: in the stub-gate chain test `pkg` is a fake directory, and on the run box both are the same checkout. If a config pointed `gate.cmd` at a different launch_gate, the profile table judged here would be this checkout's (UNVERIFIED for that case; the default `gate.cmd` is `<pkg>/refe/launch_gate.py` and `pkg` defaults to this file's package).

## Tests (MEASURED, `configguard/logs/`): before = the tip export, after = the fixed tree; same interpreter, `CUDA_VISIBLE_DEVICES=-1`

| suite | before | after |
|---|---|---|
| `refe/test_hnscreen_chain.py` | `ZZTEST_PASS 21` (13 checks + 20 mutations) | `ZZTEST_PASS 31` (14 checks + 30 mutations) |
| `refe/test_hnscreen_ctx_chain.py` | `ZZTEST_PASS 22` (None checks + 21 mutations) | `ZZTEST_PASS 22` (None checks + 21 mutations) |
| `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py` | `ZZTEST_PASS 23` (16 checks + 22 mutations) | `ZZTEST_PASS 23` (16 checks + 22 mutations) |

None dropped. The token-names file is unchanged (23 both ways; its 7 mutants still hit their snippets exactly once).

* **Check M** (`test_hnscreen_chain.py`, `M_config_load_refuses_a_token_check_name_the_profile_does_not_mint_before_s0`): the default passes (rc 0 to S0, config frozen, `[]` problems); control: `G14 -> G-HYG` (and `G15 -> G-DVB`) pass membership; refusals, each asserting rc 4, `stage_order_run == []`, an EMPTY stub journal (not even `s0_stage`), no `stage/` marker, no `config.sha256` / `config.frozen.json` / `outcome.json`, no lock, `STATUS.json` state CONFIG rc 4 reason `TOKEN-NAMES-CONFIG: ...`, log `ZZHNCHAIN_CONFIG rc 4: TOKEN-NAMES-CONFIG:`: an override naming `G14`; the old pair; one wrong value (the good default is not blamed); a stray key; a JSON null; a non-string value; a non-object; an unknown profile; a non-string profile; the default names under `refe_p2` (a real profile without `G-PREFLIGHT`); the same-profile control (`G-LIVE` passes under `refe_p2`, is refused under `refe_hnscreen`); a refusal leaves the state dir reusable; an unimportable `launch_gate` fails closed; a fresh `python -I` interpreter finds the real `launch_gate` itself.
* **Re-targeted, not weakened:** the mint-guard cases of check L (old pair, wrong value, stray key, null) and the ctx-chain arm C10 now reach the mint only with the config-load check switched off (`no_cfg_check`, instance-level), because the new first line refuses those names before S0. Their assertions are unchanged; they keep proving the second line works on its own.
* **Mutants** (10, each edits `hnscreen_devbox.py`, snippet exactly once, check M must go RED): call deleted; no membership judged; unknown profile accepted; non-object accepted; only the G14 / G15 keys judged; the `refe_hnscreen` checks used whatever the profile says; fail OPEN on an import error; the `HERE` `sys.path` line deleted; the check moved AFTER the freeze; the refusal exits rc 0. The unmutated suite and every mutant outcome are in `logs/after_test_hnscreen_chain.log`.
* **RED on the tip's source:** the new check M run against the tip's unmodified `hnscreen_devbox.py`: RED (see logs/new_check_M_on_tip_source.log).

## Proposed register rows (the Master Mind lands them; nothing was edited)

* `BACKLOG.md` `HNSCREEN-TOKEN-NAMES` follow-up 2: DONE 2026-10-20 (staged by `par18/hntok3`): `hnscreen_devbox.py:949-983`; the chain refuses at config load, before S0 and before the freeze, rc 4 `TOKEN-NAMES-CONFIG`, when a `token_check_names` value is not in the real `PROFILES[profile]["checks"]` or the profile is unknown; check M, 10 mutants; counts 21 -> 31, 22 -> 22, 23 -> 23. Remaining limit: membership only (a `G-HYG` override passes it; the mint guard and the token-names test cover the rest).

## Integration owed

Three EDITS, each with `base=<tip blob>`: `hnscreen_devbox.py` `fa4d0a9aabb569733ef371d67e73ab364181ab4b`, `test_hnscreen_chain.py` `d2a8869193021e5209c3c6c76908cfffc235de98`, `test_hnscreen_ctx_chain.py` `6978c4201e97794374c0d5975b5b0b6afd87d2f8`; plus this file, `edits.diff` and the logs. Re-run (CPU, about 10 min): `bash C:/Users/Admin/par18/hntok3/run_suite.sh fix rerun refe/test_hnscreen_chain.py refe/test_hnscreen_ctx_chain.py raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py`.
