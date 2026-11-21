<title>Section 4 ruling, priced</title>

# The PI's §4 unlabelled-surface ruling is worth **54 clips now, ~78 at completion** — and it is the same question the DAC sweep could not answer

`Data Engineering · 2026-09-20 · Master Mind · MEASURED, read from corpus_status.py's own output — NO extra load added to Thor, which is running SAM3 inference`
`raw/s4_price.json. Source: corpus_status.py at 2026-09-20 11:14 CEST, corpus 3,269/4,719 (69.27 %).`

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | ⭐ **54 of the 87 export-check failures (62 %)** carry the flag *"near path unlabelled (seen, no class), no path cell on a non-drivable class"*. They hang on §4 and on nothing else. | MEASURED |
| **2** | ⭐ **The second half of that flag is what makes it decisive:** *no path cell on a non-drivable class* — nothing the map positively calls non-drivable lies under the ego path. The ONLY thing failing these clips is surface that is **seen but carries no class**. If §4 rules seen-but-unlabelled = compliant, all 54 pass cleanly. | MEASURED |
| **3** | **~78 clips at corpus completion**, scaling by `total/done`. ⚠️ ESTIMATE, not a measurement — it assumes the rate is stationary. | ESTIMATED |
| **4** | ⛔ **It is the same question the DAC adjudication could not answer.** The sweep's one `cannot-tell` (S53) was a snow-covered car park: trafficable, but the rubric names only pavement/verge/island/kerb as non-roadway and is silent on unlabelled or off-carriageway surface. One ruling settles the export gate **and** the adjudication rubric. | analysis |

## 1 · The numbers

| | n |
|---|---|
| export-check failures | **87** |
| published in the **flagged** tier | 63 |
| — *"near path unlabelled (seen, no class), no path cell on a non-drivable class"* | **54** |
| — *"path untestable (parked / stopped ego)"* | 9 |
| GIVEN_UP, **not published** | 22 |

## 2 · ⚠️ What the ruling does NOT buy

* **It does not recover the 22 GIVEN_UP clips** — they are not published at all and carry different reasons.
* **It does not touch the 9 "parked / stopped ego" clips.** Those are the zero-motion family, the same one that produced four near-stationary sheets in the DAC packs where the footprint sits under the bonnet. A different question, and it deserves its own ruling.
* ⚠️ **Flagged clips ARE published**, in a flagged tier. The ruling changes their **tier**, not their existence. The cost of *not* ruling is that ~78 clips sit outside the clean tier, not that they are lost.

## 3 · Why this is worth the PI's attention now rather than later

The ruling is currently listed as PI-blocked with no cost attached, which makes it easy to defer indefinitely. It has a price, it is accruing at roughly the corpus rate, and **the same ruling unblocks two independent things** — the SAM3 export gate and the DAC rubric's fourth cell (`off-carriageway-trafficable`, proposed and not applied).

⛔ Nothing here recommends which way to rule. Deciding that unlabelled surface is compliant makes a permissive map permissive by fiat; deciding it is non-compliant fails clips on absence of evidence. Both are defensible and neither is mine to pick.
