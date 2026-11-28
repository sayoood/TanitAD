# AMENDMENT 2 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): Master Mind rulings on the BEHAV-F1 escalations

**Date:** 2026-11-07 ~23:00Z.
**Ruled before** any model rollout exists. No threshold, window or bar changes.

1. **d1 binds to the NAVSIM v1.1 comfort code that our PDMS harness uses.** That is the vendored copy taken from the refe-plan tree, `code/navsim_pdm/pdm_comfort_metrics.py`, sha256 `f8e64a9b...`. The F: archive's `pdm_comfort_metrics.py` belongs to NAVSIM **2.0.0** (`F:/Archive/devbox-C/navsim/devkit/setup.py:15`), and 8 function bodies differ. It is not used for d1.
   - **Why:** REFe's quoted open-loop numbers are NAVSIM v1.1 PDMS. The comfort rule has to mean the same thing in both tiers.
   - The sim's own EPDMS (v2) remains a separate, reported guard.
2. **The order of the collision source governs, and the one-liner in AMENDMENT_1 A-10 is narrowed to match it.** NAVSIM checks STOPPED_EGO first, so a stopped ego (speed <= 5e-02) touched by any object, static ones included, is NOT at fault for f1. f2 (any overlap) still reports it.
   - **Why:** AMENDMENT_1 asked for the source's logic. Where its prose summary and the source disagree, the source wins.
3. **New schema keys the sim must emit:** `agent_cls` and `ego_rear_axle_to_center`. The default of 1.461 m holds only until the sim supplies its own vehicle value. `is_agent_behind` is evaluated at the rear axle.
