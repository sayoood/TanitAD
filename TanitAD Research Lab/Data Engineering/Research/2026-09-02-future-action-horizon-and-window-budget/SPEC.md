# E-DATA-HORIZON-1 — the future-action horizon, the window budget, and the minimum decorrelating shift

`Research Lab · Data Engineering · daily pass 2026-09-02 · 0 GPU`

## Why this exists

`Project Steering/V7_LAUNCH_GATE.md` gates a queued **8.6 h** arm (queue item #4,
the O11 re-run) on a control it calls **REQUIRED**, then names one measurement as
the thing that must happen first:

> *"time-shifted negatives from the window's OWN clip … needs `future_actions2`
> **longer than `o5_k`** — at k=60 that is ≥60+shift steps per window.
> ⚠️ **Measure the dataset's available future-action horizon first**; if it
> equals `o5_k`, this is not free either"*

and, closing that section:

> *"the control … must be scoped honestly **before** the arm is queued, and the
> dataset's future-action horizon is the first thing to measure."*

This package is that measurement. Data-only: no training, no GPU, banked
artifacts only.

## Hypotheses, with both outcomes committed in advance

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-DH-1** | The available future-action horizon **equals `max_horizon`**, and `max_horizon` is DERIVED from `o5_k` rather than set independently. | If it equals `o5_k` at the O11 re-run's settings, the time-shifted control is **not free** and the grouped-batch alternative must be costed. | If there is spare horizon, the control is **free in storage terms** and the remaining question is scientific, not budgetary. |
| **H-DH-2** | A same-clip time-shifted negative is a genuine counterfactual **only past some shift s\***, set by the action's own autocorrelation. | If s\* fits inside the free budget, the control is free outright. | ⛔ If s\* exceeds the free budget, the control is **affordable but scientifically inadequate** at the incumbent windowing — and reporting only affordability would ship a broken control. |
| **H-DH-3** | The corpus's per-episode frame count **T** is what the trainer's reachability docstring assumes (**120**). | The published limit table (`K ≥ 6` yields zero windows; only `K = 4` of the strategic band is reachable) stands. | ⛔ The table is wrong and the strategic band's reachability must be restated. |
| **H-DH-4** | Raising `max_horizon` changes only the window COUNT, not which windows exist. | Parity reasoning (PI decision D4) is untouched. | ⚠️ If window START positions are truncated, an arm that changes `o5_k` also changes the temporal slice it trains on — an uncontrolled difference. |

## Success criteria, committed in advance

1. **T is MEASURED from episode bytes**, not inferred from a cache NAME, and the
   train build's T is corroborated by an **independent** published figure — or
   the finding is scoped to the val build and says so.
2. The free shift budget and the required shift `s*` are **both numbers**, and
   the verdict names which one binds.
3. Every arm-level claim is read from a **banked `config.json`**, never from prose.
4. ⛔ **A "the control is free" verdict is inadmissible unless the autocorrelation
   is also reported** — affordability alone answers the wrong question.

## Method

* **Episodes:** the 24 banked `*.v2ep.pt` of the `physicalai-val-w120-256x640cyl`
  build. Only `actions` and `jpeg_len` are read; the PNG buffer is never decoded,
  so the pass is seconds on CPU.
  ⛔ Content assertions before use (all-zero / non-finite / shape mismatch) — a
  poisoned bank must fail loudly, per the memmap-of-zeros class.
* **Windowing:** `t_max = T − window − max_horizon` and
  `future_actions = ep.actions[t+w : t+w+max_horizon]`, read from
  `stack/tanitad/data/_contract.py:120` and `:135`.
* **Derivation of `max_horizon`:** `stack/scripts/train_v6_staged.py:5053`
  (`need_k = max(o1_k, o5_k)`) and `:5076`.
* **Autocorrelation:** pooled Pearson r between `a[t]` and `a[t+lag]` per action
  channel; pairs are formed WITHIN an episode and only then concatenated, so no
  pair ever spans two episodes.

## ⚠️ Scope, stated up front

The episodes measured are the **VAL** build of this cache family. The **TRAIN**
build (`physicalai-train-e438721ae894-w120-256x640cyl`) lives on Thor, which this
pass is forbidden to touch. `RESULT.md` corroborates the train build's T from a
figure `V7_LAUNCH_GATE.md` itself publishes, and reports that as a **separate
evidence class** rather than folding it into the measurement.

**Tier:** N/A. This is a corpus/windowing measurement, not a model read, so no
T0/T1 stamp applies and the four-metric-families rule — which binds *evals* —
does not bind here. Stated explicitly so the omission is not read as one.
