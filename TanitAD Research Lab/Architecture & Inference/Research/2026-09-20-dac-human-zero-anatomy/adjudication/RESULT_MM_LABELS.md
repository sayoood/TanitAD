<title>DAC blind adjudication — Master Mind labels</title>

# 60 of 60 read as ON-SURFACE — and the honest reading of that number is partly about my own tie-break

`Architecture & Inference · 2026-09-20 · Master Mind, adjudicating the DataFlyWheel's blinded pack (9e29a43) · MEASURED (a human read), 0 GPU`
`Labels: LABELS_MASTERMIND.csv (60 rows). Pack: …/2026-09-20-dac-human-zero-anatomy/media_adjudication/ + raw/adjudication_pack.json.`

## 0 · The result

| label | n |
|---|---|
| **on-surface** | **60** |
| over-boundary | **0** |
| cannot-tell | **0** |
| *of which flagged* **BORDERLINE** *in the note* | *10* |

⛔ **The key was not consulted.** It is held by the DataFlyWheel outside the repo; its sha256 is published inside the pack. ⚠️ **But the blinding here is PROCEDURAL, not cryptographic**: the pack carries the seed (`20260920`) and the builder, and the landed `dac_windows.jsonl` carries every candidate's verdict, so the key is reconstructible by anyone holding this commit — the pack says so itself in `key_is_reproducible`. **My labels are blind by discipline, and this sentence is the record that I could have derived them and did not.**

## 1 · ⚠️ Read this before the number — the tie-break, and the bias it carries

Fixed **before** the first hard case and applied uniformly to all 60, recorded in the CSV header:

> Label on-surface / over-boundary whenever the surface beneath the car's path is visible in **any** panel; reserve **cannot-tell** for when it is not visible in any panel (occluded, unlit, or entirely out of frame).

Two consequences I am not going to bury:

1. **That rule leans toward on-surface.** A stricter adjudicator — one who required the surface under the *specific* corner to be unambiguous — would have returned several **cannot-tell**s, most obviously on the night turns (W18, W19, W28) and the near-stationary windows whose footprint sits under the bonnet (W05, W16, W46, W54).
2. ⛔ **Zero cannot-tell is a property of my rule, not a property of the images.** It should NOT be read as "the camera always shows enough". The PI's 12-window spot-check is the control for exactly this, and **the 10 rows flagged `BORDERLINE` are where it should be spent** — W18, W19, W25, W28, W43, W44, W46, W50, W51, W60.

## 2 · ⭐ What I actually saw, and why it matches the defect already on the record

The sampled paths are **dominated by crossings of PAINTED road surface** — zebra crossings (W01, W17, W29), hatched and gore markings (W58, W59), stop lines and junction paint (W39, W44, W53), tram-rail junctions (W12, W44), bus and cycle lanes (W21, W60). The rubric the pack ships — and the one the PI will label against — says in its own words that paint, markings and crosswalks painted **on** the roadway are on-surface.

⇒ That is precisely the mechanism `D-DAC-HUMAN-ZERO-1` localised: **a one-channel read-out in which painted surface reads as non-drivable.** This adjudication does not prove that mechanism — it was not designed to — but the windows the stratified draw surfaced are exactly the ones where it would bite.

⚠️ **Arithmetic on the pack's own published metadata, not on the key:** the declared pools are A 20 · B 20 · C 10 · D 10, so **50 of the 60 have at least one rule firing** and **10 have all three firing**. My read disagrees with every one of them. ⛔ I am not converting that into a false-positive rate here: that requires the key, and the key is the DataFlyWheel's to release after these labels land.

## 3 · ⛔ A defect in the instrument, found while using it

**The projected overlay has NO DEPTH TEST.** Ground points that lie *beyond* a raised foreground object are drawn *on top of* it, so a raised snow bank, kerb or parked car in the near field makes an on-road path look as though it runs over the obstacle. Caught on **W25**, where the path's right edge appears to lie on a snow mound and panel D shows the car on the cleared carriageway.

⭐ **Why this matters for the adjudication's validity, in both directions:** the artifact biases a human **toward over-boundary**. I still returned **zero** over-boundary, so the artifact cannot explain the result — it makes the zero *stronger*. But any future pack should either depth-test the overlay or say in the sheet header that it does not, because a rendering artifact that pushes labels one way is a systematic error in the ground truth we are about to build.

## 4 · What the DataFlyWheel should do next

1. Release the key; its sha256 is already published in the pack, so it can be proven to be the one the sheets were built from.
2. Score my 60 against it **per stratum**, and report **cannot-tell as its own cell** — never folded into agreement or disagreement.
3. Carry §1's caveat into the write-up verbatim. A 60/60 result from one adjudicator with a stated lenient tie-break is **one reading**, not a rate — and the PI's 12 spot-checks, spent on the BORDERLINE rows, are what turn it into evidence about the rules rather than about me.
