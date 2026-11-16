# QUESTIONS_DATASCALE_TRAIN (2026-11-16): the PI's decisions for the v8f-mini 10k data-scale training card

Card: `CARD_DATASCALE_TRAIN_DRAFT.md` (same folder; DRAFT, not frozen, nothing run).
* Only decisions that are the PI's are listed here: spend, launch, label coverage, rebase and storage.
* The design rulings (baseline 4,512 clips, `--exclude` by id, the gate route, base commit `cdbff971`) are the MM's and sit in card section 0.

Evidence classes: MEASURED (M), INHERITED (I), ESTIMATED (E), UNVERIFIED (U).

## Q1. Launch and pod time: go for the card's blocks on the LoRA pod?

**What it buys.** The arm trains the recipe of record (M2b base plus the OBD-CTL fine-tune) on 4,512 + 10,000 clips at matched steps, against the 4,512-clip control retrained on the same pod.
* A replicate sets the training-run floor.
* A deliberate-regression arm proves the new clips' labels reach the model.
* A second seed checks the sign.
* Block 3 reads `H-DATA-SCALE-1`, open since 2026-10-25: more unique data against more epochs.

| block | arms | pod GPU-h (E, at 0.21 s/step) | worst case (E, union reads 2.5x slower) |
|---|---|---|---|
| 0 pilot + gate | timing, G-LIVE, bank timing | ≈ 0.5 | ≈ 0.8 |
| 1 core | B_s0, U_s0, B_rep, R_s0 | ≈ 5.0 + 3 bank builds (U) | ≈ 8.5 + banks |
| 2 seed 1 | B_s1, U_s1 | ≈ 2.6 + 2 bank builds (U) | ≈ 4.5 + banks |
| 3 data vs epochs | UE_s0, BL_s0 | ≈ 7.2 | ≈ 12.6 |

* The rate is ESTIMATED from the MM's MEASURED 0.19 s/step on the STRAT1 pilots, times box 2's from-scratch/fine-tune ratio of 1.1.
* The GPU is the RTX 4000 Ada (M, MM `nvidia-smi` 2026-11-16).
* A pilot measures the real rate. If block 1 projects above 10 pod-h, the chain stops and comes back.
* No box 2, no Thor, no dev-box GPU. Reads run on the dev-box CPU.

**Recommended answer: GO for blocks 0, 1 and 2, then block 3, in that priority order, after STRAT1's 7-arm chain on the same pod.** One job at a time, under the 10 h per-block projection stop.
* Block 3 is cheap and answers a register row that has been open for three weeks.
* Retraining the 4,512-clip control on the pod (rather than reusing box 2's banked M2b) is part of this spend. A cross-venue control would put GPU and nondeterminism differences inside the data effect.

## Q2. Label coverage: train now with the VLM goal bits masked on the new clips, or run a VLM-CoT pass on the 10k first?

**The fact.**
* The 10,000 new clips have 0 VLM-CoT tactical goal labels: traffic-light colour, stop sign and pedestrian, provenance `vlm-cot` (I, GEO10K register row and `LABEL_INPUTS.md`).
* In the union these bits train only on the 4,512 corpus clips, which are 31 % of the windows. They are masked (weight 0), never filled as "absent". A gate check refuses the launch if a single 10k row carries such a bit with weight 1.
* All geometry labels (nav input, route checkpoints, lateral and longitudinal classes, constraint cells) exist for the new clips. They pass bitwise controls on 20 corpus clips (I, GEO10K).

**The alternative.** A VLM-CoT pass on the 10k needs:
* the frames downloaded again (mp4 about 12.8 MB per clip, about 128 GB transient, E from `COST_DATASCALE.md`);
* a VLM GPU pass whose cost is not measured (U).

**Recommended answer: train now with the bits masked.**
* The card reads the goal F1 split by VLM-class and geometry-class bits, so any dilution shows up.
* The VLM-CoT pass is already the committed next lever on three outcomes: DATA-NEUTRAL, DATA-HELPS-PARTIAL without a tactical gain, and DATA-HURTS.
* Optional, at no GPU cost: a costing of the VLM pass (frames, GPU hours, venue) while block 1 trains, so the next arm is ready when the verdict lands.

## Q3. Rebase: if the card reads DATA-HELPS, does the union become the base of record for the descendant arms?

STRAT1, ACTRSP and the planned LoRA arm all start from M2b, which trained on 4,512 clips.

**Recommended answer: yes, conditional.**
* The rebase applies only on DATA-HELPS at both seeds with the reach control passed.
* The descendants are then re-run on the union base under their own cards. Their existing reads keep their own base.
* There is no rebase on DATA-HELPS-PARTIAL. There, only the family that gained is carried forward, as a named lever.

## Q4. Next rung, conditional on DATA-HELPS: an encode go and its storage

**What it is.** About +14,500 clips from the same frozen order, positions 10,000 to 24,499 (seed 20261115, train split only, as ruled), for a 2x-union rung (about 29k clips). That is the largest step the 2x extrapolation rule admits.
* Encode: about 10.7 pod-h pipelined (E, linear from the pilot's 7.4 h per 10k, `COST_DATASCALE.md`).
* Kept cache: about 303 GB in full format (E: 20.88 MB per clip, M pilot).
* The pod has 146 GB free (M, MM 2026-11-16).

**Recommended answer: a conditional GO on the encode.**
* Storage on the pod: a **pool4x10-only training copy**, about 4.1 MB per clip, about 60 GB for the new clips (E). The v8f-mini trainer reads only that grid.
* The full-format cache stays banked on the private HF dataset as the one complete copy, verified by sha256.
* The training copy must pass a bitwise control: the trainer's own clip loader returns identical arrays.
* With this copy, no volume resize is needed. Choose the resize instead if pool8x20 must stay on the pod for other studies.
