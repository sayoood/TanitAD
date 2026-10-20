# refcv8 tactical controls (2026-10-20)

`code/tac_controls_r8.py <battery tree> --out <json>` reads `dump_s<seed>/decisions/ep###.npz` and reports, per tactical head
(lat, lon) and inference seed: accuracy and macro-F1, **head − constant majority class**, **head(nav_true) − head(nav_zero)**,
**head(nav_true) − head(nav_shuffled)**, each with a paired episode-cluster bootstrap (B 2000), plus an echo index.

Why: a tactical head accuracy quoted bare is not admissible. The eval139 lateral label is class 0 on ~90 % of windows, so a
constant guess already scores ~0.90 (GAC R8-NAV-FEED-40K, TAC_ECHO_CHECK.json).

| read | file | note |
|---|---|---|
| EMA-40k, clip-fed battery (`step40000ema_unc`) | `raw/TAC_CONTROLS_step40000ema_unc.json` | stamped: clip-level nav fed (R57) |
| EMA-40k pf smoke (6 windows) | `raw/TAC_CONTROLS_smoke40kema_pf_SMOKE_ONLY.json` | plumbing only, no claim |

For the 48,852 read: `python code/tac_controls_r8.py C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf --out <pkg>/raw/TAC_CONTROLS_step48852ema_pf.json`
(env `PYTHONPATH=<repo>/taniteval`). The nav-zero / nav-shuffled head passes are dumped at seed 0 only. Seed 1 reports them as not run.

Tests: `tests/test_tac_controls_r8.py`, 4 passed, including a deliberate-regression arm (`TAC_CONTROLS_MUTATION=constant_from_pred`)
that must go red.
