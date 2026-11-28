# INTAKE — H15 logging fidelity (accumulation-window meter)

- **Discipline:** Architecture & Inference (Wednesday agent)
- **Date:** 2026-07-15
- **Slug:** `2026-07-15-h15-logging-fidelity`
- **Status:** PENDING orchestrator triage

> ✅ **RE-CONFIRMED STILL TRUE 2026-08-16 — never integrated, and the defect line is STILL THERE
> verbatim.** Three probes:
> 1. `grep -rn "H15Meter" --include="*.py"` over the whole repo (excluding this package and stale
>    worktrees) → **zero hits**. The module lives only at
>    `…/incoming/2026-07-15-h15-logging-fidelity/h15_meter.py`; it never reached
>    `stack/tanitad/train/h15_meter.py`.
> 2. `grep -rn "h15_fire_frac\|h15_fired"` over `stack/` + `taniteval/` → **zero hits**. The two
>    additive log fields were never emitted by anything.
> 3. **The exact line this package replaces is unchanged on the tip:**
>    `stack/scripts/train_flagship4b.py:658` still reads `log["h15"] = float(loss_h15.item())`,
>    inside the `for _micro in range(accum)` loop at `:655-656`. So the measured **46.3 % of log rows
>    falsely reading `h15=0.0`** is still the behaviour of that trainer today.
>
> ⚠️ **BUT the blast radius has shrunk, and that changes the priority, not the verdict.**
> `train_flagship4b.py` is the **v1-era** trainer and is the ONLY file in `stack/` that emits
> `log["h15"]` at all (probe 2 above, repo-wide). The successor line moved to a different imagination
> surface — `train_flagship_v16.py` gates on `cfg.cond_imagination` (`:347`, `:589`) and
> `train_flagship_v4.py` on `_imagination_inputs` (`:113`, `:235`) — neither of which carries the
> stochastic `mask_prob` H15 loss this meter measures. ⇒ Integrating this now repairs **historical /
> v1-line** logging fidelity; it does **not** instrument the current v6 line. Whoever triages should
> decide on that basis rather than on the original "the live flagship run" framing (that run is long
> finished). Swept by the 2026-08-16 stale-blocker sweep.

## What
A tiny pure-Python `H15Meter` (`h15_meter.py`) that accumulates the stochastic H15 imagination loss
over one optimizer step's accumulation window and emits three log fields —
`h15` (window mean, incl. masked-out zeros), `h15_fired` (conditional magnitude), `h15_fire_frac`
(fraction of micros that fired). Replaces the single last-micro `log["h15"] = ...` line in the
flagship trainer.

## Why (measured — see research note 2026-07-15)
The 2026-07-14 program report (§8) raised a WATCH: the flagship4b log shows `h15=0.0` — "is the
imagination edge dark?" A GPU diagnostic on the exact code path
(`train_flagship4b.h15_loss` + `flagship4b_smoke_config`) showed the edge is **healthy**:
- imagination module **built** (22.06 M params, `h15.enabled=True`),
- gradient **reaches** the imagination field (L1 44.6) **and** the encoder (L1 36.7),
- fire rate **0.4525** ≈ `mask_prob` 0.5, mean loss when fired **0.611**.

The `h15=0.0` is a **logging artifact**: `log["h15"]` records the LAST accumulation micro-batch, which
reads 0.0 whenever that micro's `mask_prob` gate didn't fire. Measured: **46.3% of all log rows falsely
read `h15=0.0` while imagination actually trained that step**; the true all-masked (idle) rate is only
6.3% (theory (1−0.5)⁴=0.0625 ✓). The meter makes the edge's status faithful, and `h15_fire_frac→0`
becomes the *real* dark-edge alarm.

## Evidence / tests
- `pytest tests/` → **6 passed (0.12 s)**, standalone (no torch, no stack import).
- Pins: the artifact case (last micro 0.0 but window trained → `h15` > 0), the true-idle case
  (all masked → 0.0, no div-by-zero), all-fired, and `h15_fire_frac ≈ mask_prob` over 500 windows.
- Full stack suite unaffected (343✓/2s, this pkg does not touch `stack/`).

## Proposed target location
- `stack/tanitad/train/h15_meter.py` (the module), **and** a 3-line wiring change in
  `stack/scripts/train_flagship4b.py`:
  ```python
  h15m = H15Meter()                                  # before the micro-loop
  for _micro in range(accum):
      ...
      loss_h15 = h15_loss(model, frames, fut, cfg, device)
      total = total + cfg.h15.weight * loss_h15
      (total / accum).backward()
      h15m.update(float(loss_h15.item()))            # was: log["h15"] = float(loss_h15.item())
  ...
  log.update(h15m.log())                             # after the micro-loop (3 fields)
  ```

## Risk / rollback
- **Risk: minimal.** Log-only change; no effect on the loss, gradients, or checkpoints. The extra two
  log keys (`h15_fired`, `h15_fire_frac`) are additive — any downstream parser keys on `h15` still works
  (its meaning changes from last-micro-sample to window-mean, which is strictly more faithful).
- **Rollback:** revert the 3 trainer lines; delete the module. No data/state migration.
- The change is safe to apply to the **live flagship run** at its next resume (it only alters logging).

## Verdict (orchestrator writes here)
<!-- integrate / integrate-with-changes / defer / reject-with-reason -->
