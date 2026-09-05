# Withheld-bank panel on the v7-tiny rig (H-EGO-LIT-4) — RESULT

status: IN PROGRESS — done: `--withheld-bank {fixed,pred,random,none}` implemented in `refc.py::roll_bank` / `refc_v3.py` (hook emits the detached `g_tac` 2 s speed) / `refc_v3_train.py` (flag, warm-up, random pool, stamps, `withheld_speed_mae` log); `stack/tests/test_withheld_bank.py` 12/12 pass; paired scoped suite WITH 569 passed / 17 failed vs BASE 557 / 17 — IDENTICAL failure sets (clone-environment), zero regressions; SPEC.md written; `raw/panel.sh` + `raw/warmup_from_log.py` banked / next: trainer smoke on the epcache, then A0..A5 on the RTX 4060 once the sibling refav1 eval releases it, then `raw/panel_score.py`.

Owner: Architecture & Inference FlyWheel agent (Claude Fable 5.1), started 2026-09-05.
Pre-registration: `H-EGO-LIT-4` in `Project Steering/GOALS_AND_CLAIMS.md` (both outcomes committed there before this panel ran).
Compute: dev-box RTX 4060 only. The live run `refcv4b-b1-v72-40k` on pod `tanitad-refcv3` is untouched; Thor is not used.

## Arms (planned)

| arm | one variable vs A0 | status |
|---|---|---|
| A0 | `--withheld-bank fixed` (today's 10 m/s roll) | not started |
| A1 | `--withheld-bank pred` (withheld rows rolled at the model's own DETACHED predicted 2 s speed, `g_tac`) | not started |
| A2 | `--withheld-bank random` (random-marginal CONTROL — right distribution, no information) | not started |
| A3 | `--ego-dropout 0.25` (bank fixed) | not started |
| A4 | `--withheld-bank none` (speed-blind vocabulary, the field's default) | not started |

## Numbers

None yet. Every number in this file will carry its evidence class, its T-tier, its arm and its artifact path under `raw/<arm>/`.
