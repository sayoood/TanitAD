"""Accumulation-window meter for the stochastic H15 imagination loss.

WHY (measured, 2026-07-15 diagnostic): the flagship4b trainer applies the H15
imagination NLL on a random subset of micro-batches (``h15_loss`` gates on
``torch.rand() < cfg.h15.mask_prob``, default 0.5), then logs
``log["h15"] = float(loss_h15.item())`` INSIDE the accumulation micro-loop — so
the logged value is the LAST micro-batch's sample. With ``mask_prob=0.5`` and
``accum=4`` the last micro reads exactly ``0.0`` on ~50% of optimizer steps, and
measurement showed **46.3% of all log rows read h15=0.0 while imagination
actually trained that step** (>=1 micro fired). That false ``h15=0.0`` is what
triggered the "is the imagination edge dark?" WATCH in the 2026-07-14 program
report §8. The edge is NOT dark (imagination module built = 22 M params, gradient
reaches it AND the encoder, fire rate 0.45 ≈ mask_prob) — the LOG was lying.

This meter aggregates the H15 loss over the whole accumulation window so the log
reports whether the edge TRAINED, not a single last-micro sample. Pure-Python, no
torch dependency, so it is trivially testable and import-cheap.

Trainer wiring (replaces the one ``log["h15"] = ...`` line in
``scripts/train_flagship4b.py``'s micro-loop):

    h15m = H15Meter()
    for _micro in range(accum):
        ...
        loss_h15 = h15_loss(model, frames, fut, cfg, device)
        total = total + cfg.h15.weight * loss_h15
        (total / accum).backward()
        h15m.update(float(loss_h15.item()))
    ...
    log.update(h15m.log())     # {h15, h15_fired, h15_fire_frac}
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class H15Meter:
    """Accumulate the per-micro H15 loss over one optimizer step.

    ``update(v)`` is called once per micro-batch with the scalar H15 loss (0.0
    when the ``mask_prob`` gate did not fire). ``log()`` returns three fields that
    make the imagination edge's status unambiguous in the training log:

      ``h15``            mean over the WHOLE window (including the masked-out
                         zeros) — a smoothed magnitude comparable across steps
                         (≈ fire_frac * mean_when_fired). This is the field that
                         replaces the old last-micro ``h15`` — it is > 0 whenever
                         ANY micro fired, so it can never falsely read 0.0 while
                         the edge is training.
      ``h15_fired``      mean over ONLY the micros that fired (the conditional
                         magnitude; 0.0 if the whole window was masked).
      ``h15_fire_frac``  fraction of micros that fired — should track
                         ``cfg.h15.mask_prob`` (a drift here is a real signal:
                         the imagination module fell to None, or mask_prob was
                         mis-set).
    """

    total: float = 0.0
    n: int = 0
    fired_sum: float = 0.0
    fired: int = 0
    _log_round: int = field(default=6, repr=False)

    def update(self, value: float) -> "H15Meter":
        v = float(value)
        self.total += v
        self.n += 1
        if v != 0.0:
            self.fired_sum += v
            self.fired += 1
        return self

    def log(self) -> dict:
        n = max(1, self.n)
        r = self._log_round
        return {
            "h15": round(self.total / n, r),
            "h15_fired": round(self.fired_sum / self.fired, r) if self.fired else 0.0,
            "h15_fire_frac": round(self.fired / n, 4),
        }
