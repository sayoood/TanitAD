# ⛔ The worst-12 / best-12 reels are mostly a SPEED split — added 2026-08-05, after the README

Found while cutting a still for this folder: a "best" episode frame read **`v0 0.0 m/s`**. That is
not a coincidence.

MEASURED over the same 6,382 windows the reels are drawn from:

| set | n windows | mean `v0` | stopped (< 0.5 m/s) | mean ADE |
|---|---|---|---|---|
| **best 12** | 266 | **2.34 m/s** | **32.7 %** | 0.173 m |
| whole corpus | 6,382 | 7.54 m/s | 11.8 % | 0.575 m |
| **worst 12** | 264 | **19.19 m/s** | **0.0 %** | 1.454 m |

**Pearson r(`v0`, per-window ADE) = 0.6408.**

ADE is a displacement over a **fixed 2 s horizon**, so it scales with distance travelled: ~38 m at
19 m/s against ~4 m at 2 m/s. An identical *relative* error is then ~9× larger in metres.

⇒ **A "worst episodes" list selected on ADE is close to a "fastest episodes" list**, and a third of
the best-12 windows are the vehicle standing still, where the prediction is near-trivial.

⇒ **Watching the two ranked reels side by side and concluding *"it falls apart here and nails it
there"* is reading the speed distribution, not the model.** The `spread` reel is the one to judge
behaviour from. The ranked reels are useful for seeing what a large error *looks like*, not for
attributing it.

## What this does NOT touch

Nothing in the four-family result is invalidated. Those numbers are **corpus-wide**, and the
headline LONGITUDINAL finding is a **rate over all windows** — the prediction is ahead of the human
at 2 s on **71.95 %** of windows and faster than the human on **75.51 %** — which a speed correlation
cannot manufacture. What this constrains is **per-episode ADE comparisons and ranked highlight
reels**.

⚠️ Same defect class as quoting a pooled distance-keeping number: a statistic averaged over regimes
that do not resemble each other. **WORK ITEM: a speed-matched episode ranking.** Not done.
