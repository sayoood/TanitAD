# MM addendum 1 to MM_READING_RBSTACK.md (2026-11-19 ~03:30Z): the N1D8 card does not hold as written

The PI approved N1D8 and then UM_s1 + UE_s1 (verbatim "Both, N1D8 first", ~03:00Z). Building N1D8 showed that two statements in the card are wrong. This addendum is logged as R-RBSTACK-N1D8-1 in `Project Steering/RETRACTION_LOG.md`. The STACK-WEAK reading itself is unchanged.

## 1. "No new model code" is false (from source at 3e7efd41; not executed)
* The N1R / N1D arms are 3,000-step fine-tunes that start from M2b_s0 / M2b_s1 (`navrbd_chain_fix.sh`: `init=$INP/M2b_s0/ckpt_last.pt`).
* `n_tok` is one of the architecture fields a fine-tune start must match (`INIT_CKPT_MATCH`, `stack/scripts/v8f_mini_train.py:657`). `load_init_ckpt` raises on any mismatch.
* The scene positional table is shaped per token: `self.pos = nn.Parameter(torch.randn(cfg.n_tok, d) * 0.02)` (`stack/tanitad/models/v8f_mini.py:695`). It is 40 rows at pool4x10 and 160 at pool8x20.
* So a `--grid pool8x20` argv on an M2b start is refused before training.
* The only 8x20 base ever trained is H-V8F-GRID-1's arm G at seed 0 (19,164 steps from scratch). Its seed 1 was not trained.
* N1D8 at two seeds therefore needs one of two things:
  * a new init lever that maps the 40-row table onto 160 rows (new code, with its own selfcheck entry); or
  * a G_s1 base (19,164 steps; that card ESTIMATED 6 to 8 h per seed).
  Neither is in the approved cost.

## 2. The 8x20 premise was carded without the lever's own prior results
* RB-JOINT measured RAW8 - RAW4 at +0.0071 [-0.0046, +0.0191], not separated (`rbjoint/MM_READING_RBJOINT.md`, "not GRID"). That is on the 5-way label, with the same probe and folds.
* H-V8F-GRID-1 read FAIL-NO-MECHANISM for the 8x20 grid on driving (ADE6 -0.076, not separated).
* N1D already reads the 4x10 grid. RBSTACK's vision probe was the 8x20 probe, so its addition over the text tree (+0.023 / +0.028) does not show that 8x20 adds over the 4x10 input N1D already has.

## 3. What this leaves (for the PI)
* **The pod is not left idle.** UM_s1 + UE_s1 is the approved second item. It is being built now (build and dry run only) and runs first.
* **The card's other vision lever, N1DW:** N1D plus an RB window weight in the training loss, at s0 / s1.
  * It works on inputs N1D already has (the 4x10 grid plus the exit + distance channel), which RBSTACK shows are complementary at T0.
  * It needs a new trainer flag with a declared-vs-built selfcheck entry, a reach test and a mutant.
  * Training cost is about N1D's (MEASURED 672 s per 3,000 steps). With G-SUITE and D1-HO dumps, that is about 1 to 1.5 LoRA pod-h for both seeds (ESTIMATED).
  * The dose (the weight) is the PI's.
* **N1D8 with an interpolated position-table init** stays possible. It would test a premise that RB-JOINT already found flat.
