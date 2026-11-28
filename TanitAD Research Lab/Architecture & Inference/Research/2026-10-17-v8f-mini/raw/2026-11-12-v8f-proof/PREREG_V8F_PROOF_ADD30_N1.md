# PREREG_V8F_PROOF_ADD30_N1: note to card ADD30 (MARGIN), written before the first rollout of the card (its pod queue is still waiting for the queue of card ADD29)

One change, to the tolerance of control g3 only.

Control g3 of the card reads: at the first tick, the gap the rule used in a margin run equals the gap of the head-fed run of card ADD26 minus `M`, not below 0, within 1e-6 m.

The tool (`distrule/v8f_t1_eval_rule_margin.py`, sha256 194a5c4e7f3f3781e3daeac5a7ecf4ff9e18c039c0f0a6e7a6638796f290acae, on the pod and named in the queue log before this note) subtracts the margin in single precision, the precision of the head output. The difference of two single-precision numbers, rounded to single precision, can differ from the exact difference by half a unit of the last place: up to 1.9e-6 m for a gap between 16 and 32 m and 3.8e-6 m between 32 and 64 m. The bar of 1e-6 m would then fail on arithmetic, not on the lever.

**g3 as it is read:** the same comparison, within 1e-5 m. The tool is not changed. Nothing else of the card changes; no rollout of the card and no calibration number was affected by this note.
