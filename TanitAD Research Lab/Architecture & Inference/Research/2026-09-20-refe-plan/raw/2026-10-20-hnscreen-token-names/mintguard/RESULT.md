# HNSCREEN-TOKEN-NAMES follow-up: the chain REFUSES at the mint when a configured token check name is not in the token

Base: tip `08ff54ed71f08bc86c99378d650bea758cf06460` (`agent/arch-inf-20260803`). CPU only, nothing committed or pushed. Evidence class: MEASURED (ours, `mintguard/logs/`), all on a repo export at the tip.

## The change (`refe/hnscreen_devbox.py`, fixed-file line numbers)

| where | what |
|---|---|
| `:564-585` `token_name_problems(seed, tok)` | NEW, pure. `wanted` (`:568`) = every VALUE of the effective `gate.token_check_names` (the module defaults under the config, so a stray key counts). Per name: not a key of the token's `checks` -> `SCREEN-INVALID: token check '<name>' absent from the token of pass s<seed> (its checks: [...])`; status (`:583`) not `PASS` -> `SCREEN-INVALID: token check '<name>' status is '<st>', not PASS`. An unreadable token = every name absent. |
| `:593-596` in `verify_pass_token` | `bad = self.token_name_problems(seed, tok)`; if any: `ChainStop(RC_INVALID, ..., "INVALID")` = rc 2, `ZZHNCHAIN_INVALID rc 2: SCREEN-INVALID: token check '<name>' absent ...`, `outcome.json` row 0. It sits after the "no token" refusal and BEFORE the `launch_gate verify` subprocess. |
| docstring `:14-15` | the T<s> stage line says so |

**Where the chain first holds the token** (the hop that was wrong): `stage_T` `:736` `tok = self.token_file(seed)` (a glob for `lg/PASS_*.json`), then `:739` `self.verify_pass_token(seed, where="mint")` -> `:590`. At the tip the same hops were `:706`, `:709`, `:564`. Before this change nothing in the chain READ the token's `checks` until `token_checks` (tip `:740`) inside `assemble_gates`, called from `stage_S4`: after every pass.

`verify_pass_token` is also what `stage_P` calls (`:750`, `where="start"`), so the same guard runs again immediately before each trainer start, which covers a state dir whose T stage ran under older code.

## What it protects, and what it cannot

* Stage order is `S0 S1 S2 ANCHOR T0 T1 T2 P0 P1 P2 ENDV S4` (`stage_list`). The guard fires at T<s> (the first seed whose token lacks a name): no `launch_gate verify`, no later seed's mint, no pass `P<s>` (the multi-hour trainers), no ENDV, no S4. MEASURED below.
* **It cannot precede S2 (the cache) or T0's own `--preflight`**: the token does not exist until the gate has run, and the gate needs the cache and `preflight.jsonl`. A static check in `preflight()` of the configured names against `launch_gate.PROFILES[profile]["checks"]` would catch a wrong override before S0; not done (outside the brief; it couples the chain to `launch_gate`'s import, and the stub-gate chain test uses a stub gate). Proposed below.
* The guard compares against the TOKEN, not against a profile: a name the token carries with PASS is accepted (e.g. `G14 -> G-DVB` runs on; whether that is the right judge is `test_hnscreen_token_names.py` check C's job, not this guard's).

## Tests (MEASURED, `logs/`): before = the tip export, after = the fixed tree; same interpreter, `CUDA_VISIBLE_DEVICES=-1`

| suite | before | after |
|---|---|---|
| `refe/test_hnscreen_chain.py` | `ZZTEST_PASS 17` (12 checks + 16 mutations) | `ZZTEST_PASS 21` (13 + 20): +check L, +4 mutants |
| `refe/test_hnscreen_ctx_chain.py` | `ZZTEST_PASS 20` (9 arms + 11 mutations) | `ZZTEST_PASS 22` (10 + 12): +1 arm, +1 mutant |
| `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py` | `ZZTEST_PASS 23` | `ZZTEST_PASS 23` (unchanged file; the 7 mutants still hit their snippets exactly once) |

None dropped. `eval/test_hn_screen_read.py` and `refe/test_hn_ctx_anchor.py` were not re-run: neither touches `hnscreen_devbox.py` (only comments in `test_hn_content_fp.py` mention `verify_pass_token`).

* **Check L** (`test_hnscreen_chain.py`, stub gate whose token carries the REAL profile's check names): the default passes (rc 0, all 12 stages); control: an override naming another check the token DOES carry (`G14 -> G-DVB`) passes. Six cases stop at the mint: the old `{G14: G14, G15: G15}` pair; one wrong value (`G15 -> G-NOPE`); a stray key (`G99 -> G-NOPE`); a null (`G14 -> None`); `G-PREFLIGHT` status FAIL; and **only seed 1's token** missing `G-PREFLIGHT` (stops at T1, T0 DONE, no P0). Each asserts rc 2; `stage_order_run` ends at the failing T stage; no DONE marker for it; no `trainer:train`, `i4/i5/d1`, `s4_*`, `families`, `read`, `next_*` call in the journal; the count of `launch_gate verify` calls (0, or 1 for the seed-1 case) and of `--preflight` calls (1, or 2) proves no later mint; the GPU lock is gone; `STATUS.json` state INVALID with a reason starting `SCREEN-INVALID: token check '`; `outcome.json` row 0; `chain.log` has `ZZHNCHAIN_INVALID rc 2: SCREEN-INVALID: token check '` and no `STAGE_DONE` for the failing stage.
* **Arm C_a_wrong_token_name_override_stops_at_the_mint_with_the_real_token_...** (`test_hnscreen_ctx_chain.py`, REAL `finalize` mint): the old pair and `G15 -> G-NOPE` stop at T0: the gate ran `anchor` and `run` only (no `verify`, no `end-verify`), `trainer:pre:1` never ran, no trainer, no later stage. The default passing on the real mint is the existing `C_the_anchored_chain_end_to_end...`.
* **Mutants** (each edits `hnscreen_devbox.py` source, snippet exactly once, expected check RED): the guard call deleted (chain L and ctx arm); the status test removed; only the G14 / G15 names checked (the stray key slips through); the guard always reads seed 0's token (seed 1's missing check is not seen).
* **RED on the tip's source**: the new check L run against the tip's unmodified `hnscreen_devbox.py` FAILS (`logs/new_check_L_on_tip_source.log`).

## Proposed register rows (the Master Mind lands them; nothing was edited)

* `BACKLOG.md` `HNSCREEN-TOKEN-NAMES` follow-up: DONE 2026-10-20 (staged by `par18/hntok2`): `hnscreen_devbox.py:564-596`; a wrong `token_check_names` override stops at the mint, rc 2 `SCREEN-INVALID: token check '<name>' absent`; tests `L_...`, the ctx arm, 5 mutants; counts 17 -> 21, 20 -> 22, 23 -> 23.
* Optional new item, 0 GPU, small: `HNSCREEN-TOKEN-NAMES-STATIC`: in `Chain.preflight()`, check the configured names against `launch_gate.PROFILES[gate.profile]["checks"]` before S0, so even S2's spend is protected. Needs its own RED arm and a decision on the stub-gate chain test (profile name there).

## Integration owed

Three EDITS, each with `base=<tip blob>`: `hnscreen_devbox.py` `47eda339e9986ba80b389845e6995b108aa6994b`, `test_hnscreen_chain.py` `6015a0eef0bbaabf4581fa4d0db8dc68388d22c0`, `test_hnscreen_ctx_chain.py` `4136ba8713b5de2e19a9f3b4e164e7c945c39f9d` (the exports were hash-identical to the tip before editing, so a base-guarded land is a clean replace), plus this file and the logs. Run-box prerequisite unchanged: the first full chain with `gate.ctx_mode` unset, reading `ctx_chain/end_verify.json`.

Re-run (CPU, about 7 min): `bash C:/Users/Admin/par18/hntok2/run_suite.sh tree rerun refe/test_hnscreen_chain.py refe/test_hnscreen_ctx_chain.py raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py`.
