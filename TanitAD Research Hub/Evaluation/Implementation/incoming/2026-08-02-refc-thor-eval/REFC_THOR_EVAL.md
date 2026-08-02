# REF-C evaluated on the Jetson Thor — first four-family panel, and two blocking caveats

**PI directive:** *"do the eval of refc on thor instead of recollecting on a pod."* Done — the pods
are no longer needed for REF-C evaluation. **But the run also produced two problems that must be
resolved before any number here is quoted.**

---

## ⛔ CAVEAT 1 — `route_input_exercised` came back **None**, so the C6 confound is NOT yet resolved

The whole purpose of this run was to re-collect REF-C **with the route input exercised**, because
`refc_eval.py` states that *"every REF-C number published before 2026-07-26 (base 0.4728, XL
0.4714)"* was collected with the decoder seeing one constant command.

`nav_mode="produced"` was passed. The result carries **`route_input_exercised: None`** and
**`nav_note: None`** — the collector did not report the provenance fields at all. ⇒ **We do not
know whether the route head was actually driven**, and an unproven "exercised" is worth no more
than the confounded number it was meant to replace.

⛔ **Therefore: the numbers below do NOT supersede 0.4728 / 0.4714.** They are a plumbing result.
Next step is to read `refc_eval.collect`'s return contract and find why the provenance keys are
absent — possibly a version skew between the checkpoint's head and the collector's expectation.

## ⛔ CAVEAT 2 — REF-C-XL failed to run, and the failure is informative

```
RuntimeError: shape '[8, 512, 8, 8]' is invalid for input of size 491520
```

491520 = 8 × 512 × 8 × **15**. The decoder expects an **8×8** conv map; it received 8×15. ⇒ **the
XL checkpoint was trained at a different input geometry than the 176×624 sub-frame this val cache
delivers.** REF-C-base survived the same input, so this is XL-specific.

⚠️ This is a **geometry-provenance gap in the registry**, not a bug in the eval: no REF-C entry
records the frame it was trained at. Resolving it needs the XL run's own `config.json` geometry
block — which is what `loaders.py` uses for the model shape but not for the *input* raster.

---

## What DID work (the plumbing result)

| | |
|---|---|
| clips | **16** (self-healing quarantine active; relay still filling toward 40) |
| REF-C-base | **104.19 M params**, state_dict **missing=0, unexpected=0** (STRICT-equivalent) |
| windows | **353** |

**LONGITUDINAL** — speed MAE **3.0609 m/s**, speed bias **−1.1498 m/s** (runs SLOW), along-track
MAE 0.5543 m, along bias −0.2021, final along bias −0.5131 m.

**LATERAL** — heading MAE **3.4297°**, yaw-rate MAE **22.6241 °/s**, curvature MAE **0.034382 1/m**,
cross-track MAE **0.6013 m**, cross bias **+0.5459 m**.

**TACTICAL / STRATEGIC** — `UNAVAILABLE` (needs a hierarchy-traversing pass; a WORK ITEM).

⛔ **n = 353 windows over 16 clips, no bootstrap CI, and the route provenance is unverified.** Under
the binding rule this is **not a result**. It is proof that REF-C can be scored on Thor.

⭐ One observation worth carrying, marked HYPOTHESIS: REF-C-base's speed bias is **−1.1498 m/s**,
the *same sign and similar magnitude* as v2corpus's **−1.260 m/s**, while v1 runs **+1.465 m/s**
fast. If that survives a proper n, "which arms run slow vs fast" may be an architectural family
trait rather than per-arm noise — and it would be a longitudinal story ADE cannot tell.

## Infrastructure proven along the way

1. **Thor is a complete REF-C evaluation node** — checkpoints from HF, val from the surviving pod,
   `RefCModel` built through `loaders.py`'s own preset+config path so the state_dict loads clean.
2. ⭐ **Self-healing val loading**: clips are **load-verified** and corrupt ones quarantined at
   eval time, because a background relay is still delivering and `tar`-over-SSH truncates silently
   (exit 0). **File count is not integrity — a transferred file must be LOADED to be verified.**
   Three separate truncations were caught this way.
3. **Interface shim**: `refc_eval.collect` reads `ep.feats`; the v2 providers expose the identical
   raster as `ep.frames`. Aliased — no reformat, no re-encode, same eval path.

## Next, in order

| # | step | blocker |
|---|---|---|
| 1 | **Fix the route-provenance reporting** and re-run | ⛔ without it the C6 confound stands |
| 2 | **Record training geometry per REF-C arm in the registry**, then re-run XL | XL cannot load this raster |
| 3 | Complete the 40-clip relay → paired episode-cluster bootstrap | n=353/16 is not decision-grade |
| 4 | Hierarchy-traversing pass so TACTICAL/STRATEGIC populate | both currently UNAVAILABLE |

## Evidence class

| claim | class |
|---|---|
| every LON/LAT number above | **MEASURED (ours)** — `refc_thor_eval.json`, n=353 windows / 16 clips, ⛔ **no CI** |
| `route_input_exercised: None` | **MEASURED** — the field is absent from the collector's return |
| XL shape failure at 8×15 vs 8×8 | **MEASURED** — the raised error |
| "0.4728 / 0.4714 were C6-confounded" | **MEASURED** — `refc_eval.py` says so about those numbers |
| "slow-bias may be an architectural family trait" | ⚠️ **HYPOTHESIS** — n far too small |
