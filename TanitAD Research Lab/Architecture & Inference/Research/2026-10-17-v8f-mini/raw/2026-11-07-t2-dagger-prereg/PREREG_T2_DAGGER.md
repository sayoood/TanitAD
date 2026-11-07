# PREREG_T2_DAGGER: post-training on rendered own states (DAgger at T2), cross-fitted by fold (Master Mind, 2026-11-07 07:40 dev-box clock, from `date`; written before any RENC encoder, M-RENC number or T2 bank exists)

## Why this card exists
* `PREREG_T2_RENC.md` (frozen prefix `425d3496`) section 6 item 3 commits this lever if RENC reads RE-NULL or RE-HARMS: *"post-training C0 on rendered on-policy states of one fold (DAgger at T2), read on the other fold, under its own card."* The RENC result does not exist yet. This card is written first, so its rules cannot be shaped by that result.
* Every post-training stage of C0 so far trained on the model's own states at T1-kin: own ego motion, but the LOGGED camera frame (DAG-OACC, `H-V8F-DAG-OACC-1`; RECOVER-FT, `H-RECOVER-FT-1`). At T2 the planner sees a render at the pose it actually reached. The BLIND read (`PREREG_T1_BLIND.md`, VISION-USED) and GAP48 (render-real cosine 0.8401) say those tokens differ from the logged ones. No stage has ever trained on them.
* The one lever here: the own-state samples carry the tokens of the render at the executed pose, instead of the logged frame's tokens.

## The bank
* **Generator:** the base arm X's stage-4/5 rollouts on the 48 scenes that survive R-SCN, both seeds, frozen. One pass, with no regeneration. X = C0C, or C0C-RE if RENC reads RE-MOVES or RE-DOMINANT. X is named in the log before the bank is built.
* **States:** every tick of every start in `records.jsonl` (executed pose, executed ego history). Each is re-rendered by the same renderer at the executed pose, with agents replayed at log time, and encoded by X's in-loop encoder.
* **Identity, binding (G-BANK):** for every start, the planner of X run on the bank's tokens and the recorded ego history reproduces the recorded plan of each tick, to within the stage-4 tolerance of record. If G-BANK fails, nothing below runs, and the outcome is BANK-BROKEN.
* **Labels:** RECOVER-FT's target law, unchanged: the logged future re-expressed in the executed frame, with its 10-knot rejoin. Labels may use ego and logged futures. Inference is vision plus measured ego history only.
* **Folds:** `PREREG_T2_RALIGN.md` section 2. The first hex digit of sha256(scene id) is even → A, odd → B. Bank_A holds fold-A scenes only, Bank_B fold-B only.

## The arms (per seed s, initialised from C0_s, which is X's planner weights before the composite)
* **DT2_A_s / DT2_B_s:** RECOVER-FT's argv, 3,000 steps, p = 0.3. The deviated samples are drawn from Bank_A or Bank_B instead of the T1-kin own-state bank. Every other flag is C0's.
* **DT2-0_s (control):** the same argv and 3,000 steps at p = 0. It carries the extra training steps without the bank.
* **Read arm DT2C:** composite(DT2), the composite recipe of C0C (smoother and clamp unchanged). A fold-A scene is read with DT2_B and a fold-B scene with DT2_A. The trainer records every scene id its bank opened, and the read refuses a scene that is in its model's bank.
* **Control arm DT2-0C:** composite(DT2-0), read on all 48 scenes.
* No training run starts without a launch-gate PASS that covers the bank flag.

## The read (stage 5, box 2, 48 scenes, 93 starts, both seeds)
* **Dominance:** pinned read pin 4 (`7fd65ffd`), `--arm DT2C --arm-stage 5`, bars, rows and outcome order unchanged. If only seed 0 exists: pin 5 (`56a91667`), outcome SEED0-ONLY, never DOMINANT-T2.
* **Mechanism row:** DT2C − DT2-0C, per seed, paired scene-cluster bootstrap B 2,000, seed 0 (`read_t2_pair.py`), on M1 crawl share, ADE6 and along-track MAE. The seed floor is |DT2-0C_s1 − DT2-0C_s0| per row.
* All four families. STRATEGIC is PAUSED (PI item 25).
* **Beside, deciding nothing:** DT2-0C − X (does more training alone move T2?). A T1-kin read of DT2 on eval-141 tests whether real-frame driving is kept: |ADE6 Δ| against F = 0.12074 m. DT2C-BLIND − DT2C is the token-use check.

## Outcomes, committed now
1. **DT2-DOMINANT:** the pinned read gives DOMINANT-T2 at both seeds. Reported as written, with renders beside P10's. It is not counted as the scale signal until a replicate reads DOMINANT too: the folds swapped by the parity of the second hex digit, the banks rebuilt and the fine-tunes retrained.
2. **DT2-MOVES:** DT2C − DT2-0C is separated-better on M1 crawl share at both seeds and outside the row's seed floor; ADE6 is not separated-worse at either seed; and no new separated-worse row appears against P10. The recipe then joins the programme as the T2 post-training stage for any vision arm, the LoRA arm L included.
3. **DT2-NULL or DT2-HARMS:** anything else. The encoder (RENC) and the planner's own-state data (this card) are then both read. The next lever is more T2 scenes, since 24 per fold bound both. The 14 scenes R-SCN excludes cannot supply them. Its blocker is a corpus: the PI's call.
4. **BANK-BROKEN:** G-BANK fails. The renderer is not deterministic at the executed pose, so no own-state bank is admissible. The finding is reported with the per-tick differences.

## Scope and evidence
* Tier T2: camera rendered at the executed pose, agents replayed at log time, playback executor. Never pooled with T1-kin.
* Training variance enters through four fine-tunes and is read against the control's seed floor. Inference is deterministic.
* Nothing here is a driving claim beyond the arms named. No DOMINANT exists today.

## Log (dated, append-only after the freeze)
* 2026-11-07 07:41:11 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 5415 bytes above this log, sha256 `9234478c42f107a0528a4bcef99c98fcb7ffc1c6a59b9e3bda39eda883de0aeb`, before any RENC encoder, M-RENC number or T2 bank exists (RENC training on box 2 since 07:29:20). Open for the log, before any bank is built: the G-BANK tolerance (the stage-4 tolerance of record, quoted from its file); the trainer path for the bank flag and its launch-gate entry; the measured wall time.
