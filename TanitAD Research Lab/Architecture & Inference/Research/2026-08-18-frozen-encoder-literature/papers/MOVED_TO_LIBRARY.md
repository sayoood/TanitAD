# The PDFs moved to the central Library

All 11 primary sources cited by `FROZEN_ENCODER_LITERATURE.md` now live in
**`TanitAD Research Hub/Library/`** with sha256 records and a generated index.

Why central and not per-package: two streams citing one paper kept two copies, or none. The Library
is the single evidence store; research packages cite **library keys**.

| key | paper |
|---|---|
| `2601.03460` | FROST-Drive |
| `2411.04983` | DINO-WM |
| `2605.10564` | DeepSight |
| `2406.08481` | LAW |
| `2509.11417` | Preserving Pretrained Representations (dual encoder) |
| `2303.18240` | VC-1 / Artificial Visual Cortex |
| `2310.02219` | Large-scale study of pretrained visual representations |
| `2502.00622` | GPC |
| `2602.04880` | Probing predicts control |
| `2506.09985` | V-JEPA 2 |
| `2008.06389` | iCEM |

Add more with `python tools/kb_add.py <arxiv-id> --tag <topic> --cited-by <report>`.
