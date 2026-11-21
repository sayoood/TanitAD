# INTAKE — 2026-07-23 Gate-1 clean closed-loop-aware fine-tune (P0 gate + P1 fixes)

**Agent:** gate1-clean · **Host:** `tanitad-eval` (A40, FREE; `gpu_lock=gate1-clean`, released) ·
**Date:** 2026-07-23 (Berlin).

## One-paragraph result
Gating condition (a) VERIFIED green from the terminal marker + raw JSON (prototype `a7c1eb9c`: junction
off-road 11→7, at-fault 5→1, pass 3→8 — reproduces exactly, MEASURED). But the **clean** promotable run is
**HELD on two independent measured bounds**: (1) the low-OOD real-footage source is a map-free/agent-free
drift loop that **cannot emit off-road/collision/pass** (code-level + 2 docs) — "low-OOD" and the
prototype's win-metric are mutually exclusive with the instruments that exist; (2) the real 40-ep val holds
only **~13–22 distinct junction episodes**, and a leave-3-out (5 folds) on the prototype's 15 scenes MEASURES
**memorization** (held-out recovery-L1 5.06→5.06, Δ≈0, while train→0.41) with a **7.58 m held-out plan-shift**.
The two P1 fixes (CAT-K target filtering — drops 49% catastrophic labels; base-plan λ_dev regularizer) are
built + measured: they cut the plan-shift **7.58→2.88→1.49 m (−80%)** but do NOT cure memorization (held-out
recovery stays ~5; it is data-quantity-bound). **Training GPU
was NOT spent on the full run** (the mission's P0 rule). Recommendation: HOLD; unblock via more distinct
real junction scenes + a low-OOD lane-departure metric.

## Files
- `GATE1_CLEAN_RUN_P0_FINDINGS.md` — main report (verification, metric-mismatch, inventory, LOO, fixes, bounds).
- `PRE_REGISTRATION.md` — the proper future protocol + both fixes + both committed outcomes.
- `gate1_junction_inventory.py` / `.json` — real-footage junction inventory (MEASURED).
- `gate1_clean_loo.py` / `.json` — leave-3-out memorization + P1-fix measurement (MEASURED).
- `catk_road_filter_and_dev_regularizer.py` — the two fixes as a cited drop-in patch.
- `gate1_reeval_scores.json` — prototype base-vs-ft800 (condition-(a) verification), copied from pod.
- `proto_gate1_finetune.py` / `proto_gate1_extract.py` — prototype scripts (provenance).

## Evidence class
All headline numbers MEASURED (ours + pod path). Prototype base-vs-ft800 MEASURED (pod raw JSON). The
"RoAD" citation is flagged unverified (CAT-K is the load-bearing one). No INHERITED claim decides the GPU
hold — the memorization bound is MEASURED here, not quoted from the brief.

## Staging
`git add`-ed, NOT committed / NOT pushed. Index carries other agents' concurrent work → commit with an
explicit pathspec (CLAUDE.md §Git hygiene). No `stack/` code touched; `pytest` unaffected.
