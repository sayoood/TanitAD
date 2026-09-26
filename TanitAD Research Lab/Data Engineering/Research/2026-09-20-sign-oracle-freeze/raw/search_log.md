# search log — E-DE-SIGN-4 / sign-oracle freeze (2026-09-20)

`Every probe, including the empty ones. An unrecorded empty search is a false-absence generator (CLAUDE.md rule 2).`

| # | probe | hits | note |
|---|---|---|---|
| 1 | repo: the landed `2026-09-18-cot-sign-values/raw/cot_sign_values.json` | 43 valued clips, 42 Vienna | the freeze's sole input; its own sha256 `99662d0b…` is pinned inside the frozen file |
| 2 | repo: prior `sign_value_oracle*` anywhere under `TanitAD Research Lab/` | **0** | ⭐ per **LI19-1** (V-1 extended to the Lab's own packages) — this is the first freeze of this set, and the grep is named rather than the "first" claim being asserted |
| 3 | web: *"traffic sign recognition speed limit reader benchmark validation set size accuracy acceptance threshold 2026"* | 9 | ⛔ **EMPTY for the question asked.** GTSRB-family results (95.85–99.20 % on 39,209 images, 43 classes) are **classification on cropped crops**, not value-reading in the wild, and **no standardised acceptance threshold** for a deployed speed-limit reader surfaced. ⇒ our 0.80 bar has **no published anchor**; it is a programme convention, and should be described as one |
| 4 | web: *"arxiv 2026 vision language model speed limit sign reading autonomous driving evaluation"* | 9 | ⭐⭐ **HIT: `2606.08860`** (Martinez-Sanchez et al., submitted 2026-06-07) — the only located operating point for the E-DE-SIGN-1 task. Banked |
| 5 | primary re-read of `2606.08860` from the banked PDF (local `pypdf`, **not** the fetch summariser, per **FS19-9**) | 7 pages, 38,194 chars | ✅ the summariser's two numbers **verified verbatim**; and the PDF carries three details the summariser dropped: **39 signs**, only **two distinct values (25 / 55 mph)**, and the recall attribution to *"nighttime rain or heavy fog"*. 21/39 = 0.5385 and 21/22 = 0.9545 both reconstruct exactly ⇒ the reported figures are internally consistent |
| 6 | within `2606.08860`: any **abstention / no-read scoring convention** for value accuracy | — | ⛔ **EMPTY.** The paper reports precision and recall separately and never defines a single combined "value accuracy", which is exactly the convention LR15-5's gate also leaves unstated (finding F4). Named so it is not re-searched |

## Standing entries this pass did NOT re-run (per LAB_BACKLOG row 39)

- **OSM is CLOSED-BY-COORDINATES** for our corpus (`egomotion` carries no lat/lon) — not re-probed.
- The ego-future supplier (nav-echo, R² 0.9702) is **REFUTED**; not re-probed by standing instruction.
