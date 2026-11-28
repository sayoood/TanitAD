<title>DAC sweep — Master Mind labels</title>

# 57 on-surface, 1 cannot-tell, 0 over-boundary — the sweep finds no false pass, and says so as a bound rather than a proof

`Architecture & Inference · 2026-09-20 · Master Mind, adjudicating the DataFlyWheel's blind sweep pack (5fbac40) · MEASURED (a human read), 0 GPU`
`Labels: LABELS_SWEEP_MM.csv (58 rows). Pack: …/media_sweep/ + raw/sweep_pack.json. The key was NOT consulted.`

## 0 · The result

| label | n |
|---|---|
| **on-surface** | **57** |
| **cannot-tell** | **1** (S53) |
| over-boundary | **0** |
| *of which flagged* **BORDERLINE** | *8* |

⛔ **The key was not consulted.** It is held off-repo by the DataFlyWheel and its sha256 was published inside the pack before any label existed. ⚠️ As with the first pack the blinding is **procedural, not cryptographic**, and this sentence is the record that I could have derived it and did not.

## 1 · The one cannot-tell, and why it is not a complaint about the camera

**S53** is a **stationary manoeuvre in a snow-covered car park**, beside two parked cars. The footprint is entirely below the frame, so **no overlay appears in panel A or B at all**, and the roadway/parking boundary is invisible under snow. Both halves of my tie-break fail together: the surface beneath the path is not shown in any panel, *and* the surface class is itself ambiguous — the rubric names *pavement, verge, traffic island, kerb* as non-roadway, and says nothing about a car park.

⭐ **That is a gap in the RUBRIC, not in the images**, and it is worth fixing before a third pack: a car park is trafficable but is not carriageway, and whether the drivable-surface map is meant to include it changes the label. ⚠️ It is also a fair warning about the corpus: if stationary car-park manoeuvres are common, a DAC term will be scoring them.

## 2 · ⚠️ The same caveat as the first pack, restated because it still governs

My tie-break — *judge whenever the surface is visible in any panel; reserve cannot-tell for when it is visible in none* — was restated in the CSV header before labelling and **leans toward on-surface**. ⇒ **one cannot-tell in 58 is a property of that rule, not evidence that the camera suffices.** A stricter adjudicator would return several, most obviously on the eight **BORDERLINE** rows: S10, S18, S28, S31, S36, S51, S56 and the near-stationary sheets.

⭐ Five of the eight borderlines are **tight turns whose inside corner is a kerb or island**, and in every one the declared **no-depth-test** artifact is doing real work — the far part of the arc draws *over* the island because it lies beyond it. Those five are where a second adjudicator would most usefully disagree with me.

## 3 · What this establishes, in the DataFlyWheel's own pre-registered words

`k = 0` was committed **before** any label existed, and its reading with it:

> the sweep establishes an **upper bound of 5.3 % of clips** on departures where no rule fires. ⛔ It does **not** establish that none exist.

I am not widening that. With 55 clusters and zero observed, the bound is what the arithmetic gives and the rest is unobserved. ⇒ **All three candidates remain unfalsified on misses**, and the `aaf0879` ruling stands unchanged: they differ only in false-alarm rate, no candidate is adopted, and **P2 is provisionally usable as a low-noise penalty and not as a correctness criterion.**

⚠️ **And the combined picture across both packs is the thing to carry:** **118 windows** — 60 disagreement-enriched, 58 swept by clip — produced **zero over-boundary labels**. Every firing of every candidate seen so far is a false alarm, and no rule has yet been caught *missing* anything. ⛔ That is not evidence the rules are safe; it is evidence this corpus cannot test them for missing, which is exactly what `D-DAC-NO-POSITIVES` says.

## 4 · Scoring instructions carried forward

1. Score against the key **per cell**, never folding cannot-tell into agreement or disagreement.
2. Report **n = 55 clusters**, not 58 sheets — enforced in the scorer, not left to memory.
3. Report the **3 mixed-in candidates separately** from the 55; they are a different population and folding them in would undo the clip-first draw.
4. Carry §2 verbatim. A low cannot-tell count from one adjudicator with a stated lenient tie-break is **one reading**, not a rate.
