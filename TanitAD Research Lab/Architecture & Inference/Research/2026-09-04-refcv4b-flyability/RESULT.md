# The refcv4b fan carries ~16 % undrivable candidates, and its supervision targets are clean

`MEASURED 2026-09-04 by the Master Mind. Instrument: stack/tanitad/instruments/flyability.py
(committed a3946e5, 14 tests green). Windows: the 4,823-window / 141-episode B1-v7.2 EVAL
bank at TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4b-turn-coverage/raw/per_window.npz.
⛔ MODEL-FREE throughout — no checkpoint, no forward pass. These are properties of the
candidate vocabulary, never a driving result.`

## Why the old gate could not see this

The vocabulary gate measured friction load by differencing the anchor's eight SLOT
waypoints twice. The slots sit **0.5–1.0 s apart** (`config.json` horizons
`[5,10,15,20,30,40,50,60]` at `dt = 0.1`), so the second difference smooths away the peak
it exists to find — **an under-report of 1.21–1.85×**. Under-reporting is the dangerous
direction: a gate that under-reads friction **passes** a fan no car can drive, which is
exactly what refcv3 trained 40,284 steps on.

No finite difference is needed. `kappa` is derived once from v0 and held **constant**
while `v(t)` moves under `a_lon`, so `a_lat(t) = v(t)² · kappa` and the load is available
exactly from the rolled state.

## The defect, and why every prior gate missed it

Load is **worst at LOW speed** and falls monotonically above 4 m/s. Prior gates sampled
v0 ∈ {10, 18, 27, 36} and never looked below 10.

| v0 (m/s) | 0.5 | 2.0 | **4.0** | 8.0 | 10.0 | 18.0 | 27.0 | 36.0 |
|---|---|---|---|---|---|---|---|---|
| peak (g) | 3.85 | 4.52 | **5.51** | 3.05 | 2.28 | 1.21 | 0.87 | 0.73 |
| anchors over μ=0.7 | 22 | 30 | **30** | 24 | 18 | 10 | 4 | 2 |

**Mechanism.** At v0 = 4 m/s a 3 m/s² lateral request needs `kappa = 3/16 = 0.1875`, which
the 0.12 cap binds. That anchor then accelerates at +2.9167 m/s² for 6 s to 21.2 m/s and
carries the **capped curvature** to 54.0 m/s² = **5.51 g**. ⭐ The cap bounds CURVATURE,
which is not the same thing as bounding LOAD.

⚠️ **Cross-check that makes this trustworthy:** at v0 = 10 m/s this instrument reproduces
a sibling agent's independent measurement **exactly** — 18/117 over μ=0.7, peak 2.283 g.
It agrees where the two overlap and extends to speeds nobody measured. Pinned as a test.

## How much it actually costs — the check that bounds it

I first judged low-speed windows rare. **That was wrong, and the corpus says so:**

| | v0 < 2 | v0 < 4 | v0 < 6 | v0 < 8 | v0 < 10 |
|---|---|---|---|---|---|
| windows | 465 | 744 | 1,234 | 1,771 | **2,376** |
| share | 9.64 % | 15.43 % | 25.59 % | 36.72 % | **49.26 %** |

Median v0 is 10.090 m/s, so **half the corpus sits in the band where the load is worst**.
On average **16.27 %** of the 117-anchor fan is undrivable at the window's own v0
(median 15.38 %, max 29.06 %).

⭐ **But the supervision targets are clean, and that is the number that decides it.** The
oracle-best anchor — the target the anchor classifier is trained toward — exceeds μ = 0.7
on **15 of 4,823 windows (0.31 %)**, μ = 1.0 on 3 (0.06 %), and **never** exceeds 1.5 g.
Median load of the selected anchor is **0.059 g**, p95 **0.292 g**, max **1.198 g**. The
15 exceedances are near-stationary (v0 median 3.17 m/s, max 9.27) and involve 8 distinct
anchors.

## Verdict

**CANDIDATE-POOL WASTE, NOT A SUPERVISION DEFECT. Not a restart reason.** ~16 % of the fan
is dead weight the model must learn to ignore, and nothing prunes it — `--sel-accel-max 2.0`
is MEASURED inert (0.00 % killed, 117.0 survivors/window). The cost is capacity, not a
corrupted target.

**The fix belongs in the next build:** clamp `kappa` against the **realised** speed along
the path — require `max_t v(t)² · kappa` under a design load — rather than against v0
alone. That is a one-line change to the builder and it reclaims ~16 % of the vocabulary.

## The reasoning arc, kept deliberately

hypothesis → the instrument disagreed with my own test assertion (5.51 g, not 3.0) → the
corpus refuted "low speed is rare" (49.26 %) → the supervision-target check bounded it
(0.31 %). ⚠️ Two of my three intermediate readings pointed the wrong way. **The assertion
that failed was mine, not the instrument's**, and the number that settled it was a control
asking whether the thing being trained toward was affected at all — which is the same
discipline that catches probe failures elsewhere in this programme.
