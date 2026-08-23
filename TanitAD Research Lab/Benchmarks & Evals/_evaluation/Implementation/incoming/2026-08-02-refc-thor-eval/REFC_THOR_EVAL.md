# ⛔ RETRACTED — the REF-C Thor numbers were scored at the wrong raster

> **RETRACTION 2026-08-02, same day as publication.** Every LONGITUDINAL and LATERAL number in the
> original version of this file is **INADMISSIBLE** and must not be quoted, compared to the
> registry's 0.4728 / 0.4714, or carried into the paper. The eval fed REF-C a camera raster it was
> never trained on, and the arm accepted it **without error**.
>
> Found by adversarial verification of this very report (stream `hierarchy-tac-str`, R4), not by
> the run that produced it. The original text is preserved below the line for provenance.

---

## What actually happened

REF-C base and XL are trained and canonically evaluated at **256 px square** → `grid_shape (8,8)` =
**64 tokens** (`stack/tanitad/refs/refc.py:214-218`; `MODEL_REGISTRY.md` §4.1 records it verbatim:
*"encoder 9-ch 256 px … → 8×8×F map"*, and §4.3 pins the canonical cache
`/root/valdata/physicalai-val-0c5f7dac3b11`, nav=follow, n=881/40).

`thor_refc_eval.py` fed `v2_subframe="176x624"` — a **6×20 = 120**-token map. The two arms then
diverged in the most dangerous possible way:

| arm | `graft_imagination` | outcome |
|---|---|---|
| **XL** | `True` (`refc.py:349`) | **Failed LOUDLY.** `imagination.py:100-102` reshapes to the declared grid → `RuntimeError: shape '[8,512,8,8]' is invalid for input of size 491520` (= 8 × 512 × **120**, with 512 = `ImaginationConfig.d`, `refc.py:347`). |
| **base** | `False` (`refc.py:292`) | **Failed SILENTLY.** No grid check exists on that path, and `feat_proj(fmap.flatten(2).transpose(1,2))` (`refc.py:577`) accepts **any** token count. It cross-attended 120 tokens it had never seen — and returned numbers. |

⇒ The two "results" were the **same** defect. I read the loud one as an XL-specific geometry gap
and the silent one as a valid measurement. It was the opposite: **XL's crash was the honest
signal, and base's number was the corrupted one.**

**The corroborating evidence was in the artifact and I did not act on it.** `speed_mae_mps 3.0609`
and `yaw_rate_mae_degps 22.6241` are implausible for an arm whose registry ADE@2s is **0.4728**. I
published them with a caveat about *n* and route provenance while the actual defect was sitting in
the magnitudes.

### Three claims in the original that are also refuted

1. **"a geometry-provenance gap in the registry … no REF-C entry records the frame it was trained
   at"** — **FALSE at the second probe.** `MODEL_REGISTRY.md` §4.1 records the 256 px / 8×8 map
   verbatim, and §4.3 pins base's canonical eval cache. A textbook *"absence found at ONE location
   is not absence"* violation, in the rule's own house.
2. **"the collector did not report the provenance fields at all"** — **FALSE.** `refc_eval.py:177-190`
   always stamps `nav_provenance`, including `route_input_exercised`. The bug was in **my caller**:
   `win.get("route_input_exercised")` read the **top level**, where the key has never existed, so
   `dict.get()` silently returned `None`. The `nav_note: null` was the tell — that key has never
   existed at top level under any mode, and a computed boolean can be `True` or `False`, never
   `None`.
3. **"may need only a re-read, not a re-collect"** — **FALSE.** `thor_refc_eval.py` never persists
   `win`; only cherry-picked keys are dumped. The provenance died with the process.

## Fixes applied (this commit)

| file | fix |
|---|---|
| `thor_refc_eval.py` | **Geometry gate** — asserts the fed raster against the arm's own `grid_shape` before scoring and raises on mismatch. The check whose absence caused this. |
| `thor_refc_eval.py` | Reads provenance with `win["nav_provenance"][k]`, **never `.get()`** — a missing required stamp now raises instead of becoming `None`. |
| `stack/scripts/v5_guard.py` | Excludes route class 3 (`ROUTE_UNKNOWN`) from scoring and reports the excluded count (see the separate retraction). |

## Transferable rules earned here

1. ⛔ **A model that accepts any input shape is not validating anything — the EVAL must.** Assert
   the fed geometry against the checkpoint's declared shape before every scoring run. Two arms
   sharing a defect where one crashes and one doesn't is the signature: **the crash is the
   instrument working.**
2. ⛔ **Read a required stamp with `[]`, never `.get()`.** A `.get()` default converts a schema
   mismatch into a plausible `None`. **A `None` from a boolean-valued field is a read-path bug
   until proven otherwise** — never evidence about the thing being measured.
3. ⚠️ **When a metric's magnitude is implausible for the arm, that is the finding.** I caveated *n*
   and provenance while `speed_mae` 3.06 m/s sat in the table contradicting a 0.4728 ADE.
4. ⚠️ `route_input_exercised` conflates **exercised** with **varied**: it is
   `nav_mode != "follow_constant" and len(hist) > 1`, so a route head that collapses to one class
   reports `False` even when the input genuinely was produced. **Read `fed_command_hist`, not the
   flag.**

## Status of the underlying questions — all still OPEN

| question | status |
|---|---|
| Is the C6 route confound resolved for REF-C? | ⛔ **NO.** Never was — the flag was a null from a bad read path. |
| Does REF-C-XL have a config bug? | ⛔ **Not established by this run.** The crash was the eval's raster, and says nothing about XL's training geometry. |
| REF-C four-family numbers | ⛔ **None exist.** Retracted; needs a re-run at 256 px with the geometry gate active. |

⭐ **What DOES survive** (infrastructure, not results): Thor can build `RefCModel` through
`loaders.py`'s own preset+config path with a clean state_dict; the `frames`→`feats` interface shim
works; and the **self-healing load-verified quarantine** caught three silent `tar`-over-SSH
truncations — *file count is not integrity*. The path is proven; the numbers are not.

---
---

# ORIGINAL VERSION — RETAINED FOR PROVENANCE, DO NOT QUOTE

# REF-C evaluated on the Jetson Thor — first four-family panel, and two blocking caveats

**PI directive:** *"do the eval of refc on thor instead of recollecting on a pod."* Done — the pods
are no longer needed for REF-C evaluation. **But the run also produced two problems that must be
resolved before any number here is quoted.**

## ⛔ CAVEAT 1 — `route_input_exercised` came back **None**, so the C6 confound is NOT yet resolved

`nav_mode="produced"` was passed. The result carries **`route_input_exercised: None`** and
**`nav_note: None`** — the collector did not report the provenance fields at all.
*(⛔ REFUTED above: the collector always reported them; the caller read the wrong key path.)*

## ⛔ CAVEAT 2 — REF-C-XL failed to run, and the failure is informative

```
RuntimeError: shape '[8, 512, 8, 8]' is invalid for input of size 491520
```
*(⛔ REFUTED above: this is the eval's raster, not XL's training geometry.)*

## What DID work (the plumbing result)

| | |
|---|---|
| clips | **16** (self-healing quarantine active; relay still filling toward 40) |
| REF-C-base | **104.19 M params**, state_dict **missing=0, unexpected=0** (STRICT-equivalent) |
| windows | **353** |

**LONGITUDINAL** — speed MAE **3.0609 m/s**, speed bias **−1.1498 m/s**, along-track MAE 0.5543 m,
along bias −0.2021, final along bias −0.5131 m. ⛔ **RETRACTED — wrong raster.**

**LATERAL** — heading MAE **3.4297°**, yaw-rate MAE **22.6241 °/s**, curvature MAE **0.034382 1/m**,
cross-track MAE **0.6013 m**, cross bias **+0.5459 m**. ⛔ **RETRACTED — wrong raster.**

**TACTICAL / STRATEGIC** — `UNAVAILABLE`.
*(⚠️ And the proposed fix — passing `hier=taniteval.hierarchy.run(...)` — would NOT have populated
them: `hierarchy.py:431-433` returns `{"skipped": …}` unless the model has `tactical_policy` AND
`strategic_policy` brains, which `RefCModel` does not have. REF-C's `self.strategic` is a **ctx
encoder**, not a policy brain.)*

⭐ The HYPOTHESIS that REF-C-base's slow bias matches v2corpus's is **withdrawn** — it rested on a
retracted number.
