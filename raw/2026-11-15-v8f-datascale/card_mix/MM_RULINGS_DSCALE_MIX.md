# MM rulings on CARD_DSCALE_MIX.md, section 0 (2026-11-18 ~07:50Z, before any UM_s0 code, run or number)

The card is a pre-registration. Nothing has been implemented, trained or read for UM_s0. These rulings settle its section 0, and the card is FROZEN by the sha256 below. A later change goes through an amendment.

1. **Wording: accepted.**
   * The 10k clips carry live lateral and longitudinal class losses, the 7 geometry goal bits and the lat / lon constraint cells. Only the 15 VLM goal bits and the lane-change cells are masked on them.
   * So the arm tests the corpus rows' share of ALL of T's tactical supervision, as section 1 words it, not only the VLM label share.
2. **Code: on its own branch, never the tip, until UM_s0 is read.**
   * The per-source T-block weight (section 3.3) lands as a child commit of `cdbff971` on a new branch. It is UM_s0's code of record only.
   * UM_s0 is read on the unchanged `cdbff971` read tree; the checkpoint's cfg is UE_s0's.
   * Keeping it off `agent/arch-inf-20260803` means no other session's bound record sees a `stack/tanitad` change, so no closure question arises.
   * Identity 1 (bit-for-bit unchanged when the batch key is absent) is a required test, with its mutation.
3. **Pod time is the PI's.** About 4.4 pod-h GPU-holding (E from UE_s0's MEASURED 13,803.9 s loop), plus a pod-environment G-SUITE on CPU. The PI is asked with these numbers. The implementation, its tests and the G-SUITE are MM work and proceed now; none of them uses the GPU.
4. **Power: UM_s0 first.**
   * The replicates (UM_s1, UE_s1, BL_s1, UM_rep) are named in section 5.6 and each comes as a short amendment with its own go.
   * The card already says DILUTION-PARTIAL is a likely outcome under either hypothesis, and its committed next step is the replicate pair.
   * **Recorded before any UM number:** the "visits" reading. TAC-LAT tracks corpus visits in all four block-3 arms (U < B ≈ UE < BL).
     * That reading predicts UM ≈ UE, i.e. NOT-DILUTION.
     * The card's own prediction is a gain of +0.04 to +0.10.
     * Both are written down; the label is the card's rule.
5. **Pod state: MEASURED present (2026-11-18 ~07:45Z, read-only).**
   * `/root/data/cache_union_dscale/train/clips` holds 14,512 entries.
   * `/root/dscale10k/merged/v9_labels_union.npz` md5 is `9881b170a889f743c5b0894f9ad13f1d`, the card's value.
   * `join_union.npz`, `lead_union.npz`, `speed_union.jsonl` and `native_tracks_union.pt` are present.
   * The parent card's section 7 deletion of the union view waits until every arm that reads it (UM_s0, and X_s0's C-JOIN-EQ check) has finished.
6. **Floor: accepted as written.** It is the 19,164-step B-set floor applied at 61,568 steps, as block 3 applied it, and the result says so. UM_rep would measure a union floor at 61,568 steps; it is not launched by this card.
7. **G-DVB without a `launch_gate.py` v8f-mini profile: accepted.**
   * The entry lives in the trainer's `FLAG_CHECKS` and the study checker, as for blocks 1 to 3.
   * `launch_gate.py` is not edited (binding); the section 7 row text stays unapplied, as a note.

**Freeze:** `CARD_DSCALE_MIX.md` sha256 is recorded in `CARD_DSCALE_MIX.md.sha256` beside it, written with these rulings.
