# PREREG_V8F_LISTEN2: does the route information travel DOWN the hierarchy? Three follow-ups to the who-listens-to-what grid (written 2026-10-25, before any LISTEN2 data)

**Status: pre-registered.** Zero training; the LISTEN harness (`raw/2026-10-25-listen/listen.py`), banked M2b s0 / s1, dev-box CPU, under `PI-V8F-2026-10-25b` / `-25d`. These are the three cheapest next experiments `RESULT_V8F_LISTEN.md` named.

## Why (MEASURED; `raw/2026-10-25-listen/READ.json`, T1-class open loop, DIAGNOSTIC, paired episode-cluster bootstrap)

* T-lateral's largest dependence is **S's checkpoints**: its F1 drops 0.391 / 0.406. Its dependence on the scene is not separated.
* Nav reaches the plan through S and T: nav at S only +0.73 / +0.82 m, at T only +0.48 / +0.52 m, at O only +0.015 / +0.011 m (ADE6).
* **Open:** is T-lateral reading S's checkpoints for the ROUTE that S derived from nav (a hierarchy), or for geometry that does not depend on nav (a parallel path)? The grid's donor swap replaces the whole checkpoint, so it cannot tell.
* O's single-input swaps hurt more than swapping all of O's inputs together (n > 1), which the grid could not explain.

## E1 (PRIMARY): route through the chain

A donor nav is injected **at S only**: S's checkpoints change, and T and O read them as usual, with their own nav untouched. Read T-lateral's macro-F1 drop. This is the only path by which nav-at-S reaches T-lateral.

| read | definition | bar |
|---|---|---|
| **E1** | T-lateral F1(clean) − F1(nav donor at S only) | **> 0 and separated, at BOTH seeds**, and above the donor-seed floor (donor seeds 0 and 1 on s0) |
| beside | the same for T-longitudinal; O's ADE6 (already +0.73 / +0.82 m); the fraction E1 / T-lateral × S checkpoints | descriptive |

| outcome | condition | meaning |
|---|---|---|
| **ROUTE FLOWS DOWN** | E1 at both seeds | T's lateral decision follows the route S derived from nav: nav → S → T is a working chain |
| **PARALLEL PATH** | E1 not separated at either seed | T-lateral reads S's checkpoints for something nav does not change; nav reaches T only directly. The hierarchy claim is narrowed to "S → O" and "nav → T" |
| **SEEDS DISAGREE** | anything else | reported; seeds before windows |

## E2: is the grid's ordering an artefact of donor swapping?

Each input is replaced by the **training-set mean** of that input (mean over HBLEND's 9,582 training windows, per seed), not by a donor. This is a second intervention family, with no cross-input conflict.

| read | definition | bar |
|---|---|---|
| **E2** | per row, the input with the largest effect under mean replacement equals the grid's largest-n input (S = nav, T-lat = S checkpoints, T-lon = ego, O = ego) | at both seeds: **ROBUST** if all 4 rows agree, **PARTLY ROBUST** if 2–3 do, **NOT ROBUST** if fewer |

## E3: O's coherence

For O, inject ego, S's checkpoints and T's code **together** from the same donor (the prior keeps its own input), against each single swap.

| read | definition | bar |
|---|---|---|
| **E3** | ADE6(joint triple) − max over the three single swaps | **< 0 and separated at both seeds** ⇒ O reacts to the *contradiction* between its inputs more than to a coherent foreign state (`INCOHERENCE`); else `NOT SHOWN` |

## Controls

* The identity donor gives exactly 0.0.
* The clean pass reproduces LISTEN's clean levels to ≤ 1e-4.
* Mean replacement with the window's own value gives exactly 0.0.
* E1's injection leaves T's and O's own nav tokens bit-identical to clean (asserted per window).

⛔ No donor seed, mean set, metric or bar changes after any LISTEN2 number is seen. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes; tier T1-class open loop, DIAGNOSTIC.
