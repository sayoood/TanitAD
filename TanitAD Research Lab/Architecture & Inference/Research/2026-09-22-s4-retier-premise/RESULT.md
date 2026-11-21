
<!-- S4-RETIER-PREMISE-VERIFIED-ON-LIVE-DATA-2026-09-22 -->

### ⭐ 2026-09-22 — the §4 re-tiering pass is DE-RISKED before it runs: its premise verified on live data, and the PI's (c) reporting quantity MEASURED

MEASURED by me, read-only, CPU only, **no npz loads and negligible Thor CPU while production runs**
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-s4-retier-premise/`).

The PI ruled §4 on 2026-09-20 — *"a) for the term, (c) for the reporting"* — and the prereg defers
application to **ONE pass after production completes** (ETA today 17:43), so every clip is judged by
one rule. This verifies that pass's premise **now**, so it is an executed decision at DONE rather
than an improvisation.

**Everything needed was already banked — no re-inference, and no `.npz` opened.**
`publish/SEMANTIC_MAPS_MANIFEST.json` carries `clips_flagged` with **100** entries, of which **77**
carry `path_classes`. The flag histogram matches the live status exactly: **77** *"near path
unlabelled (seen, no class), no path cell on a non-drivable class"* + **23** *"path untestable
(parked / stopped ego)"*. ⚠️ That mattered operationally: loading 77 `.npz` through `path_classes()`
would have put real CPU on Thor at 95 % of a production run, for a number already on disk.

**Codes read from source, never assumed:** `ROAD = (1, 2, 3, 4, 6)`, `NON_DRIVABLE = (5, 7)`
(`corpus_publisher.py:47`), `255` = unseen and excluded by `path_classes` (`:83`), and **`code == 0`
is "seen, no class"** (`sam3map_render_v5b.py:240`). ⇒ under (a), compliant = `{0} ∪ ROAD`.

**THREE CONTROLS, EACH READING AN EXACT KNOWN VALUE:**

| control | must read | measured |
|---|---|---|
| clips with any NON_DRIVABLE path cell | **0** ⇒ compliant share is exactly **1.0** | **0** ✅ |
| clips with a code outside `{0} ∪ ROAD ∪ NON_DRIVABLE` | **0** — the partition is complete | **0** ✅ |
| minimum unlabelled share | **> 0.10**, since failing the 0.9 threshold with `nd = 0` REQUIRES it | **0.1001** ✅ |

⇒ the prereg's *"their compliant share is 1.0 by construction — they pass the 0.9 threshold
outright, not marginally"* is **CONFIRMED on live data**, not inherited. The third control was not
planned: the distribution's lower bound is pinned by the very threshold that created the flag, and
it lands 0.0001 above it.

**⭐ THE PI'S (c) QUANTITY, MEASURED.** The ruling requires the withheld / unlabelled share be
published **beside every DAC / EP / PDMS number**. Across the 77 clips:

| min | median | mean | max | clips > 50 % |
|---|---|---|---|---|
| **0.1001** | **0.1372** | **0.1706** | **0.4603** | **0** |

So the worst clip has **46 %** of its seen path cells unlabelled — compliant under (a), and under
the old rule 54 % drove it below 0.9 and flagged it. That is the mechanism, in one number.

**Tracking:** the prereg projected **78** such clips at completion from `s4_price.json`; the live
count is **77 at 95.19 % done**. The projection is holding.

⚠️ **Scope, unchanged and restated:** this does **not** recover the 40 `GIVEN_UP` clips (different
reasons, not published) nor the 23 *"path untestable (parked / stopped ego)"* clips — those carry no
`path_classes` because `real` is null, and they are the zero-motion family, a separate question.
And re-tiering changes a clip's **tier**, not its existence: flagged clips are already published in
`semantic_maps/gt_flagged/`.

⛔ **NOTHING WAS APPLIED.** Production is live (4,492/4,719), the supervisor holds its lock, and the
prereg's whole point is that the rule change is a numerator change applied ONCE, afterwards. This
firing establishes the premise and the reporting number; the pass itself waits for DONE.
