# p0-d8-preview — SC-05 degraded-visibility familiarity signal (step 6500, 4060)

**Date:** 2026-07-08 (loop iteration). **Hardware:** local RTX 4060, fp32, strict numerics,
~4 min wall-clock, $0. **Checkpoint:** p0-sB01 step 6500/30000 (~22%). **Script:**
`stack/scripts/d8_preview.py` (3 tests in `stack/tests/test_d8_preview.py`).

## Question
Does the free A9 familiarity signal — 1-step relative imagination error
e = |ẑ₁−z₁| / |z₁−z₀| — already separate in-domain driving (comma val, route-held-out)
from degraded-visibility synthetic driving (Cosmos foggy/rainy/snowy/night)?

## Result: NO at this training stage — and the direction is inverted (P8, first-class negative)

| group | n | median e | mean e |
|---|---|---|---|
| comma_val (in-domain) | 400 | **9.73** | 20.46 |
| cosmos_clear | 100 | 5.27 | 7.21 |
| cosmos_degraded | 200 | 6.00 | 9.45 |

AUROC (degraded should score HIGHER): comma-vs-degraded **0.34** (inverted),
comma-vs-clear 0.30, clear-vs-degraded (weather axis, domain-matched) **0.54** (~chance).

## Why (mechanisms, both checkable)
1. **Step-size normalization artifact.** The denominator |z₁−z₀| tracks scene change rate.
   Comma highway changes slowly (A8≈0.053) → tiny denominator → inflated ratio; cosmos
   changes ~2× faster (A8 0.109–0.14). The ratio measures scene dynamics as much as
   familiarity at this stage. (Comma median 9.7 is consistent with the gate report's
   imag-rel 9.7 — measurement agrees across harnesses.)
2. **Training-mix composition.** The model trains 40/60 comma/PhysicalAI-URBAN; cosmos urban
   content is plausibly more "familiar" than the comma highway domain itself. The missing
   control is a physicalai-val group.

## Follow-ups (queued in Opponent Analyzer BACKLOG P0.2 → revised)
- **Matched-pairs weather test:** cosmos renders the SAME base clip under several weathers —
  pair by base clip id to isolate the weather axis with the domain held fixed.
- **Score redesign:** absolute error with per-corpus z-normalization, and latent Mahalanobis
  drift (the D8 gate design) instead of the raw step-ratio.
- **Add physicalai-val group** as the in-domain urban control.
- **Re-run at 15k/30k:** familiarity contrast should sharpen as prediction error tightens
  onto the training distribution.

## Claim discipline
No claim about H11/D8 is made or retired: gate D8 runs on the trained checkpoint against
real OOD probes (nuScenes, never trained) with the redesigned score. This preview's value is
(a) the harness exists and runs end-to-end on real bytes, (b) the naive score is now known
to be confounded — found at 22% training instead of at gate time.
