# P6-HN-S implementation, checklist item 8: the dev-box chain (`refe/hnscreen_devbox.py` + `refe/test_hnscreen_chain.py`)

Staged only (no commit, no push, no existing file edited). Tip read: `bcb9a1f57f130f96510512506b6cedda9af22ecd`. Pod never touched.

## Headline
The chain exists and is tested with stubbed trainer / gate / tools and the REAL landed GPU lock on a temporary lock path.
`ZZTEST_PASS 16` = 1 unmutated suite of 12 checks (0 failing) + 15 mutations, every one RED on its expected check (log: `impl8/test_hnscreen_chain.log`).
**It cannot run the real screen yet** (integration owed, escalated to the Master Mind, not left in a doc):
- items 3 and 4 must land first (`scorer_finetune.py --hn-bank/--ctx-cache`, profile `refe_hnscreen`);
- 9 required stage commands (plus 2 optional) have no CLI yet and are NAMED PARAMETERS in the chain config (AMBIGUITIES 5). The chain refuses (rc 4) before running anything while a required one is unset; without `post_pass` the I4/I5/D1 gates have no evidence and the screen ends SCREEN-INVALID before the read (fail closed).

## What the chain does (module docstring has the detail)
Stage order S0, S1 (build, SY, I1, I2), S2 (cache rounds, I6, G15), T0-T2 (preflight, `launch_gate.py run`, `verify`), P0-P2 (verify, trainer, post-pass I4/I5/D1), S4 (gates file, offsets M3, `hn_screen_families.py --per-set-out`, `hn_screen_read.py --families-per-set`), VERDICT (rule E adds seeds 3 and 4 once), NEXT.
- **Token:** `launch_gate.py verify` on the exact argv file, inside the lease, immediately before the trainer starts; the trainer gets the list read back from that file. rc 0 AND `ZZGATE-MATCHZZ` or rc 5 and no trainer.
- **Lock:** `eval/aux_navtest_forward.py gpu_lock` (what `ctx_cache.landed_gpu_lock` wraps), imported, not copied. Held around preflights, I6/G15, verify+trainer+post-pass, offsets. NOT held around the cache driver (it takes the lock per unit itself; a nested hold would deadlock).
- **Never removes another job's lock:** any lock file or the `chain_active` marker = wait (30 s polls, `max_wait_s` 6 h), then rc 75 + HANDOFF, resumable. Release is the landed pid-matched one.
- **Resume:** DONE marker = fingerprint (config, stage, upstream marker digests) + sha256 of every output; skipped only if both match. An interrupted trainer pass is archived (`scores.attempt<ts>`) and retried; the lock is released on every failure.
- **HANDOFF per stage** (`handoffs/HANDOFF_<stage>.md` + `HANDOFF.md`, < 6000 chars): done/pending, artifacts + sha256, NEXT, resume command, STOP-file hint. No key path, pod address or HF token in any file (test B).
- Stops: gate failure (row 0, rc 2) is checked before SY (row 1, rc 3); I6/G15 failure trains nothing; any failed gate stops before the read.

## Tests (`PYTHONIOENCODING=utf-8 python refe/test_hnscreen_chain.py`, venv `tanitad`, CPU, ~7 min)
Checks A order, B handoffs + no key leak, C resume after a crash, D resume by content, E no trainer without token (5 variants), F lock discipline, G foreign lock (live holder, dead-holder 2 h old, chain_active, busy GPU) + a CONTROL, H families/read argv + rule E, I stop points, J levers + named parameters, K config frozen / single instance / low disk / CLI.
The 15 mutations: token check deleted; start-of-pass check deleted; lock acquisition deleted; foreign lock removed while waiting; chain_active ignored; DONE check skipped; resume trusts presence; `--families-per-set` dropped; lock held around the cache driver; SY stop deleted; I6/G15 stop deleted; lever check skipped; rule E twice; gate stop before the read deleted; stale scores not archived.
The shared `devbox_gpu.lock` / `devbox_gpu.chain_active` were stat'ed before and after the whole suite: unchanged. Free RAM was 6.3 GB at launch (waited for > 6 GB), C: 47 GB.

## AMBIGUITIES (each implemented as an explicit named parameter or documented choice)
1. **Which `take_lock`.** `git grep "def take_lock"` finds 8 copies. Used ANF `gpu_lock` (item 2's `ctx_cache` and its HANDOFF name it). The newest copy (`c3_run_p2full.take_lock`, used by the SR-TTC S0 driver) is bound to a one-off script's ctx object.
   **Hazard found, for the Master Mind (not edited):** ANF `acquire_gpu_lock` removes a lock whose holder pid it cannot see once the file is older than 600 s, and the NavSim/battery `gpu_lock.py` writes MSYS pids Windows cannot see (the live `refcv8-battery` lock is of that kind). Called directly it would remove a live job's lock. The chain calls it only when the file was absent a moment before and caps `wait_idle` at 300 s, so that branch is unreachable; test G has a CONTROL showing the landed helper alone removes such a lock. Recommend making that branch report-only.
2. **Cache driver.** `ctx_cache.py` is a library with no CLI. `commands.s2_cache` must print `ZZCTX_PARTIAL` (budget `{budget_s}` = 1800 s reached) or `ZZCTX_DONE` and take the lock per unit through `ctx_cache.landed_gpu_lock`.
3. **`--preflight` is exempt from the token check:** it produces the evidence the token is minted from. Every non-preflight trainer start is gated.
4. **Argv order and `--arm-name`.** Item 4's profile decides the token order. The chain takes `trainer.argv_template` from config and enforces `--seed`, `--out`, `--dump-scores`, `--hn-bank`, `--ctx-cache` exactly once with this pass's values, no `--epo`/`--aux-drv`, `--lam 0`. `gate.arm_name_fmt` (e.g. `S{seed}`) is optional and unset by default; the CLI's `--arm-name` choices must be extended by item 4.
5. **Commands with no CLI yet (config):** `s0_stage`, `s1_build`, `s1_i1`, `s1_i2`, `s1_i3` (optional), `s2_cache`, `s2_i6`, `s2_g15`, `post_pass` (I4/I5/D1), `s4_offsets`, `s4_prep`. Gate commands write `gates/<ID>[.s<seed>].json` with `{"ok": true|false}`; the previous evidence is renamed away first.
6. **Gates file:** I1 I2 I3 I6 G15 from evidence files; I4 I5 D1 per pass; G14 and G15 also from the token's `checks.<name>.status == PASS` (names configurable); `launch_gate` = every pass was token-verified at its start.
7. **NEXT** runs the commands configured for the verdict row and writes `NEXT_LEVER.md`; the prereg's lever experiments (S-DAC, pair-contrast, ...) are not automated here.
8. Score files are `scores/scores_update1000_{A,B}.npz` (checked against `ScoreDump.write`, `scorer_finetune.py:170-176`); `s4.base_scores` is config.
9. The cache is bound on resume by its `manifest.json` bytes only; `CtxCache.verify()` (a re-hash of ~70 GiB on exFAT) is not run by the chain.
10. Rule E is detected as `outcome.row is None` in `hn_screen_read`'s result.

## Deliverable manifest (all in ONE place until landed: `C:/Users/Admin/hn8/out/`)
- `refe/hnscreen_devbox.py` (sha256 `94544eac968e170d...`, 943 lines)
- `refe/test_hnscreen_chain.py` (sha256 `6ee8cca2d7c020eb...`, 734 lines)
- `raw/2026-10-18-p6hn-prereg/impl8/test_hnscreen_chain.log`
- `raw/2026-10-18-p6hn-prereg/impl8/RESULT.md` (this file), all under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`
- `C:/Users/Admin/hn8/out/paths.json` lists them. LF endings.
