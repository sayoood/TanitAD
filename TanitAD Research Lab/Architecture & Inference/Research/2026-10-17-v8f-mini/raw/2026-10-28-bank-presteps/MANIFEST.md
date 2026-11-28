# bank_presteps: deliverable manifest (Research Lab agent, 2026-10-28; nothing staged, committed or pushed; the Master Mind lands it)

Root: `E:/Projects/TanitAD-artifacts/dagland/bank_presteps/`   (`SHA256SUMS.txt` pins every file below marked *)

**Integration request (headline, not a README footnote): land `recovery/` (labeller prototype, NOT stack/), the three result files and the frozen-card files into the repo; register the three claim rows in `GOALS_AND_CLAIMS.md`.** No trainer flag, no stack edit, no box 2 was used or proposed here.

| deliverable | files |
|---|---|
| frozen before data | `FROZEN_CARDS.md`* (sha `3202a62f...5fd5b`), `FROZEN_CARDS.sha256`, `FROZEN_AMENDMENT_1.md`* (`4676bdb8...93c7`, labeller construction + FITGAP statistic; before real data), `FROZEN_AMENDMENT_2.md`* (`17cce25f...9f47`, POST-HOC V2; after V1's smoke) |
| 1 H-ST-REGRET pre-step | `regret_prestep.json`*, `regret_prestep.md`* |
| 2 H-ST-RECOVERYLABEL | `recovery/labeller.py`* (V1, registered), `recovery/labeller_v2.py`* (V2, post-hoc), `recovery/test_labeller.py`* (21 tests), `recovery/test_real_panel.py`* (3 tests on real data), `recovery/recovery_run.py`*, `recovery/recovery_run_v2.py`*, `recovery/RECOVERY_RESULT.md`*, `recovery/recovery_result.json`*, `recovery/crosscheck_feasible_decode.py`, `recovery/refusal_breakdown.py`, `recovery/make_summary.py` |
| 3 H-DD-FITGAP (M2b, PARTIAL) | `fitgap_m2b.json`*, `fitgap_m2b.md`* |
| inputs built (reproducible) | `raw/bank_table_s{0,1}.npz`* (613,132 records each, validated against `bank.json`), `raw/logged_cache.npz`* (tracks sha256 1db431f1... = bank header), `raw/real_panel.npz`*, `raw/*.json`, `raw/chain_*.log`, `raw/pytest_counts.txt` |
| code | `scripts/` (`bankio.py`, `build_tables.py`, `build_logged_cache.py`, `common.py`, `stats.py` + `test_stats.py` (3 tests: bootstrap equals `taniteval/ci.py`), `regret_prestep.py`, `fitgap_m2b.py`, `run_chain.py` RAM-gated runner) |
| tip sources read | `src_tip/` = copies `git show 5cc7d7b1:<path>` (bank builder, mini model/data/trainer, feasible_decode, floors_kin, ci, p10_prior ...) |

Reproduce: `scripts/run_chain.py A "build_tables.py s0 s1" "build_logged_cache.py" "regret_prestep.py"`; `scripts/fitgap_m2b.py`; `recovery/recovery_run.py s0|s1`, `recovery/recovery_run_v2.py s0|s1`, `recovery/make_summary.py` (about 5 minutes in total, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`, `python -I`).

Housekeeping, disclosed: (1) I removed my own scratch log `raw_build_tables.log` (a RAM-gate refusal message) and `__pycache__` directories inside this package with `rm`; nothing else was deleted. (2) `raw/eval_dumps/` (282 per-episode files extracted from `v8f-mini-hacc/t1kin_hacc_dumps_f64_2026-10-26.tar` with `tar --force-local`) shows ~282 MB allocated on exFAT for ~8 MB of data; re-extractable, delete at will (PI's call). (3) `raw/real_panel_v2_unused.npz` is an unused by-product of the V2 runner. (4) Free RAM fell below 3 GB twice during the session; every step that loads a shard started only when `require_ram(3.0)` / the chain's gate passed.
