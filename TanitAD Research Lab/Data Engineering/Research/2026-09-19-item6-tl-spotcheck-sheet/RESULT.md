# Item 6 readiness: a 50-frame traffic-light colour spot-check sheet (~15 minutes)

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind · **0 GPU**
⛔ **Banked, NOT sent to the PI.** The Master Mind offers it.

## What it is

`sheet.html` — open it in a browser from this folder. 50 rows, one per clip. Each row has one
image, the label beside it, Alpamayo's own reason sentence, a verdict picker
(✓ lamp shows this colour / ✗ different colour / no relevant lamp / can't tell) and a notes field.
Verdicts persist in the browser; **"Show results as CSV"** prints `n,sha12,label,verdict,notes`
to copy back. `sheet_template.csv` is the same list for pen-and-paper use.

**The label** is the v8.1 traffic-light tactical goal — an **Alpamayo-derived teacher signal**
from the VLM's reasoning text, **not ground truth** (the v8 manifest's own wording rule). The
sheet says so in its first paragraph.

## How it was built — `code/build_sheet.py`, `code/render_sheet_html.py`

| choice | value | why |
|---|---|---|
| sample | **RED 14 · GREEN 14 · YELLOW 12 · colourless 10** | stratified by label colour, fixed seed `20260919`; YELLOW (25) and colourless (18) are rare, so they are over-sampled on purpose |
| pool | RED 384 · GREEN 378 · YELLOW 25 · colourless 18 | the v8.1 blobs, md5-verified before use |
| records with two colours | **0** | none to exclude |
| frame | **the last frame Alpamayo saw** — 5.1 s into the clip | `ALPAMAYO_T0_S` (`stack/tanitad/data/alpamayo_records.py:149`); the label's own `t_nominal_s` is a band-midpoint CONVENTION (`time_basis: untimed`), not evidence of when the light was seen |
| clock | camera and egomotion share one | the first camera frame lands +0.063 s after the first egomotion sample |
| image | full frame (context) **+** a native-resolution band on the clip's OWN horizon | band = `cy − 0.30 H … cy + 0.05 H`, central half; yellow box marks it on the full frame |
| names | **sha12 only** | no clip id in any file name, row or field |

⛔ **The first build got the zoom band wrong, and looking at one image caught it.** It placed the
band by frame fraction (5–50 % of height), which on a **rig-B** clip (horizon ~200 px lower) showed
only night sky while the red lamp sat below it. The band now centres on each clip's own `cy` from
`front_wide_cy.parquet` — the datacard's own rule, *"crop around the CLIP'S cy, never the frame
centre"*. The sample splits **rig B 26 / rig A 24**.

## Checked in a browser, not only generated

Served locally and exercised: 50 rows render with their images; choosing a verdict updates
"1 of 50 judged"; the verdict survives a reload; the CSV quotes a note containing a comma and
quotes correctly. The two temporary servers were stopped afterwards.

## How to read the result, stated before anyone looks

The sheet measures **per-instance label correctness on 50 frames**, which the v8 manifest recorded
as "not established and now never will be" — the PI's "no VLM" ruling closed the *model* check,
and this is a *human* one. With 50 frames it bounds a gross error rate, not a precise one: e.g. 0
wrong in 50 puts a one-sided 95 % upper bound of about 6 % on the error rate for that set.
"Can't tell" and "no relevant lamp" rows are **excluded from the error rate and reported
separately** — a lamp the camera cannot show is a visibility fact, not a label error.

## Files

`sheet.html` · `sheet_template.csv` · `media/NN_<sha12>.jpg` (50, 5.3 MB) · `raw/sample.json`
(per row: sha12, label, grounded, disputed, rig, cy, frame index and time, reason sentence) ·
`raw/build_stdout.txt` · `code/`.
