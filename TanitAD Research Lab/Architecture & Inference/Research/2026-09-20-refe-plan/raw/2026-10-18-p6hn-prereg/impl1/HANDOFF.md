# HANDOFF: P6-HN-S implementation, items 3, 4, 8 (after B8's carve-feed lands)

State: items 1, 2, 5, 6, 7 done and tested (`RESULT.md`). Next agent (Sonnet) owns items 3, 4, 8 only. Stage into OUT, never commit.

## Item 3: `refe/scorer_finetune.py` (+ `refe/test_hn_screen_wiring.py`)
- Import `hn_bank` (term, `load_bank`, `HNBank`) and `ctx_cache` (`CtxCache`). Flags `--hn-bank <path>` (arm B only), `--ctx-cache <dir>`; each defaults to None and must be NAMED in the arm's argv.
- Arm B in `fwd()`: keep the existing fan call (`armB(cxy, sctx)`) untouched; add `HB.hn_term(armB, cxy, hn_xy, hn_tg, hn_m, sctx)` to `losses6`'s B side only (`lB += out.loss`); batch fields from `bank.batch(keys, ckpt_steps)`.
  The lane output gets nothing (the term slices `[..., :6]`). `--hn-bank` with `--epo`, `--lam != 0`, `--aux-drv`, or a bank of 0 hard-negative rows is a refusal.
- `--ctx-cache`: serve `cache.get_ctx(keys)` as `sctx` (bf16 -> the dtype the live path gives) for base / A / B; arm A through `model.score_trajectories`, the same function forward uses. Verify `CtxCache(dir, expect_trunk_fingerprint=ckpt_fp)`.
- Data event: `hn` block = `bank.hn_block("B", subtypes)`; `ctx_cache` block = `{manifest_sha256, trunk_fingerprint, n_scenes, dtype: "bf16"}`.
- Wiring proof (CLAUDE.md "unreachable from its caller"): hop trace argparse -> `a.hn_bank` -> bank loader -> batch fields -> `hn_term` -> `losses6` -> `L_B` -> `.backward()` with file:line each; `hn_bank.probe_fields` gives the section-4.3 gradient table
  (non-zero on B's scorer rows 0-5, EXACT 0.0 on the lane row and on arm A / base); B == A bit for bit after 2 steps with the bank absent and empty; I3 (`main()` of the pre-edit trainer vs the new one, both flags absent, bit-identical);
  the 7 mutations of prereg 4.3. D1: sum of `hn_m` that entered `L_B` == sum of the served keys' bank rows (`HNBank.n_rows` / `batch`).

## Item 4: `refe/gate_checks.py`, `refe/launch_gate.py` (+ `refe/test_launch_gate.py` additions)
- `DVB_KINDS_SCORER += {hn_bank: data, ctx_cache: data}`, `SCORER_HN_KEYS`, `SCORER_CTX_KEYS`, `judge_g14` (re-judge from `hn_bank.probe_fields` raw fields; use the NEUTRALISED-self-attention duplicate identity or the per-layer oracle,
  NOT the literal draft wording: see RESULT.md finding 2), `judge_g15` (`ctx_cache.cache_vs_live` == 0.0 on >= 64 + >= 64 scenes; a row-shuffled cache must read > 10x; manifest trunk fingerprint == the checkpoint's), `SCORER_GATES += G14, G15`,
  profile `refe_hnscreen` (= `refe_rls1` + the allowed differences of prereg 4.2); the 12 regression arms of prereg 4.4; `refe_rls1*` arms unchanged.

## Item 8: `refe/hnscreen_devbox.py` (+ `refe/test_hnscreen_chain.py`)
- Stage order S0..S4, a token check before every trainer start, DONE markers, a HANDOFF per stage, the lock (`ctx_cache.landed_gpu_lock`), never removing another job's lock. Deleting the token check or the lock acquisition must turn the test RED.
- S2 = `ctx_cache.build_units` in per-log units, `budget_s` <= 1800 s, the lock released between units (resume by content).

## How to run what exists
`PYTHONIOENCODING=utf-8 python refe/test_hn_bank.py`; `... refe/test_ctx_cache.py`; `TANITEVAL_DIR=<taniteval dir> python eval/test_hn_screen_read.py` (venv `tanitad`, CPU, ~1 min each).
