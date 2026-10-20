# X7 LEAD-VIOL (distance keeping) on refcv8 EMA-48,852 and EMA-40k, per-frame feed, vs refcv7-50,400

Bar: SPEC_REFCV8_DRAFT §8.7 (c), re-read from the SPEC by the Master Mind: on S6, at both inference seeds, refcv8 − refcv7
`share_of_at_risk` < 0 separated (paired) AND `lead_progress_ratio_clipped` ≥ 0.90. Tier T1 open loop on logged frames.
Instrument: the registered `taniteval/taniteval/lead_metrics.py` (byte-identical to repo 75ce238), driver `code/x7_r8.py`
adapted call for call from `2026-10-11-r7base-fourfamily/code/r7base_stageB.py` + `r7base_stageD_leadprog.py`; lead block
md5 `c346dff3b3d356f55ef2786f327c6c91`. `REF_INPUTS.sha256` pins the repo extracts the agent used (they live in the repo at
their own paths; not duplicated here). Report: `X7_48852.md`; all numbers + provenance: `X7_48852.json`. No clip ids.
