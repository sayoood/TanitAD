# PLAN — E-DATA-HORIZON-1

`2026-09-02 · Data Engineering · compute budget: 0 GPU, ~2 min CPU`

## Priority order (a killed run still yields value at every step)

| # | step | yields on its own | state |
|---|---|---|---|
| 1 | Read the windowing contract from source (`_contract.py:120`, `:135`) and the `max_horizon` derivation (`train_v6_staged.py:5053`, `:5076`) | answers the gate's question **structurally** — the horizon EQUALS `max_horizon` | ✅ done |
| 2 | Read the four banked arm `config.json` files for derived vs args `max_horizon` | exposes the args-invisible fourth variable (F3) | ✅ done |
| 3 | Measure **T** off banked `*.v2ep.pt` bytes | refutes the docstring's 120 (F1) and re-derives reachability (F2) | ✅ done |
| 4 | Measure the action autocorrelation r(lag) | turns "is the control free?" into "is the free shift a real negative?" (F4) | ✅ done |
| 5 | Cross-check the train build's T against a published window count | separates MEASURED from corroborated (F1) | ✅ done |
| 6 | Bank raw + RESULT, append the KB line, propose backlog rows | the deliverable | ✅ done |

## Compute

* **0 GPU.** The dev-box RTX 4060 was checked free of Python compute before and
  after; nothing was scheduled on it. Thor and the A40 pod were **not touched**.
* CPU only: 24 episode files read for `actions` + `jpeg_len`; the PNG buffer is
  never decoded.

## Owners / integration

* **Owner:** Research Lab (daily pass 2026-09-02).
* **Escalated to the Master Mind** — see `COMMS.md`. Three of the four findings
  change documents the Lab does not own (`V7_LAUNCH_GATE.md`, MM-E19's scope
  sentence, a trainer docstring), so they are escalated as decisions rather than
  written into a doc and left to be re-read.

## Reproduce

```bash
python code/measure_horizon_budget.py --out raw/horizon_budget.json
```

Defaults point at the banked local paths recorded inside `raw/horizon_budget.json`
(`episodes.dir`, `arm_configs.*.config_md5`). The script asserts on episode
CONTENT before use, so a poisoned or truncated bank fails loudly instead of
producing a plausible number.
