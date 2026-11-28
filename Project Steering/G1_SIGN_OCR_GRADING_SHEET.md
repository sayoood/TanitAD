# G1 — sign-OCR grading sheet (PI action)

**What this is.** The G1 gate (sign-OCR precision) is the one PH0 validation step only the
PI can perform: judging whether the VLM's transcribed sign text matches what the sign
actually says. These are the **31 non-empty OCR texts** from the 50-clip pilot — the
pre-registered sample. The overlay reels already delivered (VLM + full-pipeline videos)
show these clips with the signs highlighted.

**What I need from you — one line back:** the row numbers that are WRONG (e.g. "wrong: 3,
17, 25"), or "all correct". From that I compute precision, and the gate opens or closes:
until then `route_to` goals and `signs[].text` stay extraction-only and are excluded from
PH1 supervision (already enforced in the fused records as `pending_g1_gate`).

| # | clip | kind | OCR text | light state |
|---|---|---|---|---|
| 1 | `d600f5a0-07d9` | speed | **20** | none |
| 2 | `1ad72863-7c25` | speed | **100** | none |
| 3 | `6a7fec0e-3488` | other | **P** | none |
| 4 | `85231b09-0044` | nav | **The Sea** | none |
| 5 | `09759d8c-b66c` | speed | **80** | none |
| 6 | `6cac490c-f9e1` | speed | **1.5** | none |
| 7 | `5c0660df-f0d0` | speed | **10** | none |
| 8 | `030011f7-dfb9` | speed | **30** | none |
| 9 | `f3412785-ffce` | speed | **10** | none |
| 10 | `0ee893cc-ba18` | other | **35** | none |
| 11 | `5f86e111-b219` | speed | **100** | none |
| 12 | `5786a9a5-c5cd` | other | **Kreuzberg** | none |
| 13 | `dafbe237-92c8` | speed | **40** | none |
| 14 | `7ded8ea1-a4d8` | speed | **50** | none |
| 15 | `b683866a-b3e0` | speed | **10** | none |
| 16 | `8f1baf35-90f6` | speed | **30** | none |
| 17 | `01acc9de-8bb9` | speed | **10** | none |
| 18 | `105d818e-527b` | other | **P** | none |
| 19 | `fe2f90e4-5b96` | speed | **20** | none |
| 20 | `769e919b-71bb` | other | **Stadion** | none |
| 21 | `dd8f9293-0df2` | nav | **A12** | none |
| 22 | `15958430-1119` | other | **ΦΑΡΜΑΚΕΙΟ** | none |
| 23 | `b704bd89-f5b9` | speed | **40** | none |
| 24 | `d435c12d-ea9d` | other | **P** | none |
| 25 | `d7da7ade-a66a` | speed | **20** | none |
| 26 | `c5cb15d3-97d0` | other | **SLOW DOWN** | none |
| 27 | `d5172ad5-60cc` | nav | **N204** | none |
| 28 | `b99058dc-6802` | other | **P** | none |
| 29 | `fef1ca56-9637` | other | **APOTHEKE** | none |
| 30 | `fef1ca56-9637` | other | **A** | none |
| 31 | `dd08983e-17e9` | other | **11** | none |

n = 31. Kind distribution and goal backing were validated separately
(PH0_PIPELINE_VALIDATION.md §1); this sheet is ONLY about whether the text matches the sign.
