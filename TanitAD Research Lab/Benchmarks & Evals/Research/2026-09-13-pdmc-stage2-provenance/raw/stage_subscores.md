# PDM-Closed navhard two-stage — stage-wise subscores from both primaries

Retrieval: WebFetch of arXiv HTML renderings on 2026-09-13 (model-summarised extraction), THEN cross-checked locally with PyMuPDF on the banked PDFs.
⭐ **The local cross-check changed the conclusion:** the banked `2506.04218` PDF is **v3 (20 Mar 2026)** and its Table 2 (*"navhard leaderboard. Snapshot from 03/2026"*) carries PDM-C **S1 94.4 / 98.8 / 100 / 99.5 / 100 / 93.5 / 99.3 / 87.7 / 36.0** and **S2 90.5 / 90.6 / 95.4 / 98.4 / 100 / 86.6 / 74.2 / 91.9 / 29.7, EPDMS 56.6** — i.e. the right-hand columns below are **v3 = TOAD**. The left-hand columns are **v2 (27 Aug 2025)**, read via HTML only; in the v3 PDF text the strings `51.3`, `88.1`, `83.1`, `25.4`, `73.7` occur **0 times** while `56.6`/`90.5`/`86.6`/`29.7`/`74.2` occur (same-breath control: `94.4` occurs 3×).

| subscore | 2506.04218 v2 S1 | 2606.07170 v1 S1 | Δ S1 | 2506.04218 v2 S2 | 2606.07170 v1 S2 | Δ S2 |
|---|---|---|---|---|---|---|
| NC | 94.4 | 94.4 | 0 | 88.1 | 90.5 | +2.4 |
| DAC | 98.8 | 98.8 | 0 | 90.6 | 90.6 | 0 |
| DDC | 100 | 100 | 0 | 96.3 | 95.4 | −0.9 |
| TLC | 99.5 | 99.5 | 0 | 98.5 | 98.4 | −0.1 |
| EP | 100 | 100 | 0 | 100 | 100 | 0 |
| TTC | 93.5 | 93.5 | 0 | 83.1 | 86.6 | +3.5 |
| LK | 99.3 | 99.3 | 0 | 73.7 | 74.2 | +0.5 |
| HC | 87.7 | 87.7 | 0 | 91.5 | 91.9 | +0.4 |
| EC | 36.0 | 36.0 | 0 | 25.4 | 29.7 | +4.3 |
| **combined EPDMS** | | | | **51.3** | **56.6** | **+5.3** |

Aggregation (`2506.04218`): "s_combined = s1·s2"; `navsim/docs/metrics.md`: Stage-2 follow-up scenes weighted by a Gaussian kernel on distance to the planner's Stage-1 end position, then multiplied with Stage 1.
⚠️ Combined scores are means of per-scenario products and **cannot be recomputed from these mean subscores** — the table localises the difference; it does not reproduce either total.

DrivoR (base, no TOAD) in `2606.07170` Table 6 for reference: S1 NC 99.1 DAC 98.2 DDC 99.3 TLC 99.8 EP 75.4 TTC 98.7 LK 94.9 HC 97.6 EC 70.2; S2 NC 92.3 DAC 91.6 DDC 97.3 TLC 99.1 EP 75.7 TTC 90.6 LK 56.1 HC 98.4 EC 44.7; EPDMS 54.6.

## v3 Table 2 in full — the 03/2026 substrate, verified locally (PyMuPDF, banked PDF sha256 a8431697…)

Leaderboard source named in v3: `https://huggingface.co/spaces/AGC2025/e2e-driving-navhard` ("Snapshot from 03/2026").

| method | EPDMS (03/2026 snapshot) |
|---|---|
| CV [8] | 11.4 |
| Ego MLP [8] | 14.1 |
| LTF [11] | 25.1 |
| NavFormer | 34.1 |
| LTFv6 [51] | 31.9 |
| RAP [52] | 39.6 |
| ZTRS [53] | 48.1 |
| GuideFlow [54] | 51.5 |
| SimScale [55] | 53.2 |
| DrivoR [56] | 54.5 |
| PDM-C [12] | 56.6 |

⭐ **Every row above is on ONE substrate** — `E-BE-S1S2-1` is already half-served for these eleven entries.
⚠️ **DrivoR reads 54.5 here and 54.6 in TOAD Table 6** (several cells differ by exactly 0.1 — S1 TLC 99.7/99.8, TTC 98.6/98.7, LK 94.8/94.9, HC 97.5/97.6; S2 LK 56.0/56.1 — others equal; a rounding or re-run difference) — a 0.1-point re-run spread on a learned planner. Any wedge argument inside ±0.1 is inside that spread.
