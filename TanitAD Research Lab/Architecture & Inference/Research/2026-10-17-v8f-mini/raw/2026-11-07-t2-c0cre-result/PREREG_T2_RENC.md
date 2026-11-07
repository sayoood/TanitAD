# PREREG_T2_RENC: adapting the encoder to NuRec renders while keeping its real-frame tokens, cross-fitted by scene (Master Mind, 2026-11-07 06:34 dev-box clock, from `date`; written before any training image is rendered and before any adapted encoder exists)

## 1. Why
* `PREREG_T2_RALIGN.md` (frozen `a6c84433`) read **ALIGN-FAILS-MECH**. A cross-fitted per-channel affine map lifts the held-out render-to-real cosine from 0.840 to 0.894, below the 0.90 bar, and reaches only 0.902 even on the scenes it was fitted on. By its section 6.4, the next lever is encoder adaptation.
* GAP48 shows the gap is structural, not a pull toward the mean: renders sit at 0.710 from the BLIND mean, against 0.743 for their real frames. A map on the tokens cannot reach what the encoder never extracted. A change inside the encoder can.
* Real-frame tokens are where the planner's skill lives: on real frames, removing them costs ADE6 +0.83 m (`D-V8F-T1-BLIND-1`). So the adaptation must leave them unchanged. If it does, the adapted encoder is a render-robust encoder the driving model could use, not only a fix to the T2 camera.

## 2. Data (fixed now)
* The 48 read scenes, with the folds of `PREREG_T2_RALIGN.md` section 2: the first hex digit of sha256(scene id), even goes to A and odd to B.
* On box 2, every 2nd tick of each scene: the frame rendered at the logged pose (`objects`), and the real frame at that tick. Both are kept as the encoder's own 256×640 uint8 cylinders.
* The targets are the frozen encoder's pooled tokens of the real frames, P(E(real)), fp16 [40, 1024], computed once.
* The held-out reads use GAP48's ticks (every 10th) of the other fold's scenes, which the trainer never sees.

## 3. The adaptation (one recipe, fixed now)
* A LoRA on the frozen DINOv3-L of the in-loop encoder, on q and v, rank 32, alpha 32: the same scope as the LoRA arm L (Amendment 4 of the LoRA card).
* **Loss:** mean squared error between P(E′(render)) and P(E(real)), plus λ_keep = 1 times the mean squared error between P(E′(real)) and P(E(real)). P is the encoder's own fixed 4×10 pool, the tokens the planner reads.
* AdamW, learning rate 1e-4, weight decay 0, batch 8, bf16 with fp32 master weights, exactly 10 epochs over the fold's pairs. There is no early stopping and no selection on any held-out number. Seed 0.
* **Two encoders:** E′_A, trained on fold A only, and E′_B, trained on fold B only. A scene is always read through the encoder trained on the other fold.

## 4. Gates (each with a deliberate-regression arm that must fail it)
* **G-ID0:** with the LoRA's B matrices at zero, E′ reproduces the frozen encoder's pooled tokens bit for bit on 100 frames. Regression arm: a perturbed B must fail it.
* **M-RENC (mechanism, held-out fold, GAP48 ticks).** The median flattened cosine of (P(E′(render)), P(E(real))) rises above 0.840 by at least 0.05 and reaches at least 0.90, the same bar as M-ALIGN. Regression arm: E′ scored on its own training fold is flagged by provenance.
* **G-KEEP (held-out fold, real frames).** The median cosine of (P(E′(real)), P(E(real))) is at least 0.99.
* **M-RENC and G-KEEP must both pass** before any T2 stage runs. Otherwise the outcome is **RE-FAILS-MECH**.

## 5. The driving read (stage 5 on box 2, the 48 scenes, both seeds)
* **Arm C0C-RE:** the composite C0C, whose in-loop encoder is E′ of the scene's other fold. Planner weights, ego inputs and executor are unchanged.
* Read by the pinned dominance read (pin 4, `--arm C0C-RE --arm-stage 5`) and by the pair read C0C-RE − C0C, both with the paired scene-cluster bootstrap, B 2,000, seed 0. All four families; STRATEGIC PAUSED (PI item 25).
* **Beside, deciding nothing:**
  * C0C-RE-BLIND − C0C-RE, the T2 token-use check of the LoRA card's open item 5(b);
  * a T1-kin read of C0 on eval-141 re-encoded through E′, which tests whether real-frame driving is kept (|ADE6 Δ| against F = 0.12074 m).

## 6. Outcomes, committed now
1. **RE-DOMINANT:** the pinned read gives DOMINANT-T2 at both seeds. It is reported as written, with renders beside P10's. It is not counted as the scale signal until a replicate reads DOMINANT too: the folds swapped by the parity of the second hex digit, and the encoders retrained.
2. **RE-MOVES:** C0C-RE − C0C is separated-better on ADE6 at both seeds, M1 is not separated-worse at either seed, and no new separated-worse row appears against P10.
   * The recipe joins the programme as a render-robustness step for any vision arm, the LoRA arm L included (trained on top of L's merged encoder, same folds).
   * If the T1-kin read beside also holds |ADE6 Δ| < F at both seeds, it is reported as a candidate encoder change for the driving model, under its own card.
3. **RE-NULL or RE-HARMS:** anything else. The next lever is the planner rather than the encoder: post-training C0 on rendered on-policy states of one fold (DAgger at T2), read on the other fold, under its own card.
4. **RE-FAILS-MECH** (section 4): the same next lever as 3, with no T2 stage spent.

## 7. Scope and evidence
* Tier T2, with the arm's encoder change declared in every row. The estimator answers another draw of scenes. One adaptation seed: the replicate in 6.1 answers the training-run question for a DOMINANT. Inference is deterministic.
* Admissibility at inference: vision only. E′ is an image encoder. It never sees ego state, labels or the scene it is read on. Real frames enter training only as targets, and only from the training fold.

## 8. Log (dated, append-only after the freeze)
* 2026-11-07 06:34 (dev-box clock, from `date`), Master Mind. **Freeze taken again before any data, wording only.** One word in section 1 was changed to "fix", for the house vocabulary, minutes after writing. No image, encoder or number existed. The first freeze was sha256 `15be11b5...`. The frozen prefix is now the first 5544 bytes, sha256 `425d3496eb52e30025c11c2609409ea1478aa2fe70ba6609c9f9477c128fd722`.
* 2026-11-07 06:54 (dev-box clock, from `date`), Master Mind. **Implementation choices the card leaves open, fixed before any training image or encoder exists** (`t2v8f/renc/`, `HANDOFF_RENC.md`).
  * Gradients pass through the encoder's fp8 pooling by straight-through.
  * Batch 8 pairs, accumulated over micro-batches of 4. The learning rate is constant.
  * The merged E' is built with the stage-5 fold path (`LM.fold_lora`).
  * The trainer records every scene it opens, and the gate refuses an encoder whose record includes a held-out scene.
  * G-KEEP adds a shifted-pairing regression arm (real frames paired with the wrong tick) that must fail it.
  * For C0C-RE the stage-5 blob needs C0C's planner heads merged with E's LoRA. That is a packaging step, not a change of recipe.
* 2026-11-07 07:38 (dev-box clock, from `date`), Master Mind. **Box-2 numbers, MEASURED.** Data job: launched 07:01:06, 48 of 48 scenes, 0 FAIL rows, pulled 07:29 (`renc/logs/chain_data.out`). Training launched 07:29:20 (`D:\t2v8f\logs\renc_train_20261107_072913.log`): fold A logs 1.61 s/step at step 50 and 1.54 at step 100 (loss 0.01046 and 0.00809), against the handoff's estimate of 3 to 5 s. GPU 97 %, 3,528 MiB in use at 07:35 (one nvidia-smi read, not a peak). At about 1.57 s/step, 2 folds of 2,990 steps take about 2.6 h, so training should end near 10:10, then the gate encode. The dev-box gate runs as `python -E -s renc_gate.py`: the `-I` form in the handoff drops the script folder from the import path and fails on `import renc_lib`.
* 2026-11-07 07:57 (dev-box clock, from `date`), Master Mind. **C0C-RE packaging, fixed before any C0C-RE cell exists** (`t2v8f/renc/c0cre/`, `HANDOFF_C0CRE.md`).
  * The 06:54 line said the stage-5 blob needs C0C's heads merged with E′'s LoRA. The package instead keeps C0's own planner checkpoints (`C0_s0.pt` / `C0_s1.pt`, sha256 `bc3f926b...` / `43a30398...`, as C0C-RA) and ships the two RENC blobs unchanged, and each cell folds the blob of its scene's other fold. The weights reaching the model are the same, so the recipe is unchanged and the planner and the encoder stay separately hashed. Adopted.
  * **Identity gate G-ID-C0CRE, binding before C0C-RE launches:** the package built with B = 0 blobs (`C0C-RE_ID`, 2 scenes, tar `c5eca630...`, MANIFEST5 `3230269a...`) must reproduce C0C's stage-4 records exactly on box 2 (`c0cre_idmap.py`, adapted from RALIGN's G-IDMAP). Every tick must carry the `renc` stamp with the encoder fold equal to the other fold of its scene. Tests: 17 passed (`logs/pytest_c0cre.run4.log`), including an identity side built from real stage-4 records and five arms that must fail it (status, a missing stamp, a wrong encoder fold, one value moved by 1e-9, one record dropped).
  * A real fold found a builder defect: both B = 0 blobs fold to the frozen encoder, so their merged digests are equal, and a check copied from the LoRA arm's packaging refused that as "the same LoRA twice". For IDENTITY only, the check now requires the two digests to be EQUAL; a GATED package still refuses them. The stub pin of the tests now folds a zero-B blob to the frozen digest, as the real fold does.
  * C0C-RE-BLIND renders every frame: the banked dispatch skips rendering only for the literal arm name BLIND.
  * Box-2 order, run by scripts: RENC train and gate, then the WSP-LORA base-stats job (about 15 to 25 min), then the C0C-RE_ID identity cell (about 10 to 15 min, ESTIMATED), then C0C-RE if M-RENC-PASS, G-KEEP and G-ID-C0CRE all hold (about 1.6 to 1.9 h, ESTIMATED), then C0C-RE-BLIND.
* 2026-11-07 12:24 (dev-box clock, from `date`), Master Mind. **Result: M-RENC-PASS, gate valid.** Pull `C:/Users/Admin/t2v8f_pulls/renc_20261107_121702/M_RENC.json`, read by `renc_gate.py` (log `renc/logs/renc_gate_dev.out`).
  * M-RENC: the median render-vs-real cosine with the OTHER fold's encoder is 0.9427 (scene-cluster bootstrap B 2,000, seed 0: [0.9390, 0.9466]), against 0.8401 before. The rise is +0.1026, and all 48 scenes improve; 960 held-out pairs.
  * G-KEEP: 0.9967 [0.9965, 0.9968] on real frames.
  * G-ID0: the adapter at B = 0 is bit-equal on 100 of 100 frames, merged 100 of 100, and the perturbed B on 0 of 100.
  * Regression arms behave as committed: the same-fold read is flagged on 48 of 48 scenes (a naive 0.9697 would have passed unguarded); the pairing shifted by one fails (0.9254); the perturbed B is flagged. No leak.
  * Training (box 2): 10 epochs × 300 steps per fold, 1.35 / 1.37 s per step, 4,088 / 4,121 s; 3,145,728 trainable parameters; max |B| 0.114 / 0.103.
  * Process: the train,gate chain on the dev box died silently after launching training (07:29). Training finished at 09:48:42. The gate mode was resumed by `renc/renc_resume_gate.sh` at 12:17 (gate encode on box 2, then train's small files pulled by hand; sizes match box 2). The C0C-RE (RE) and C0C-RE-BLIND packages were built from this pull at 12:22; they run after the WSP-LORA job and the C0C-RE_ID identity cell (`c0cre/c0cre_after_id.sh`, detached).
* 2026-11-07 13:04:23 (dev-box clock, from `date`), Master Mind. **Readings of section 6 fixed before the C0C-RE pull exists** (drive running on box 2 since 12:44:32; no C0C-RE record has been read). Built into `stage5/C0C-RE/c0cre_outcome.py` (self-test 26 / 26, 9 / 9 mutations red, rehearsed on the banked BPC pull bit-equal):
  * 6.2's "M1 is not separated-worse" is the M1 crawl-share row of the pair C0C-RE − C0C, the pair the clause is about. M1 against P10 is shown beside and decides nothing.
  * 6.2's "no new separated-worse row against P10" means a row separated-worse for C0C-RE against P10 that is not separated-worse for C0C against P10 at the same seed (C0C from stage 4, cross-checked row by row against the pair's own C0C rows).
  * 6.3's label: RE-HARMS when any 6.2 clause fails on its worse side, else RE-NULL. The card gives both the same next lever, so the label changes nothing that follows.
  * If stage 5 excludes a scene that stage 4 kept, C0C's rows against P10 are re-read on the common scenes by the same pinned reader before 6.2 is applied. Any other failed cross-check makes the read INPUT-INVALID.
* 2026-11-07 13:09:20 (dev-box clock, from `date`), Master Mind. Step count of record: 3,000 steps per fold (10 epochs × 300, `TRAIN_RESULT_A.json` / `TRAIN_RESULT_B.json` in the pull). The "2,990 steps" in the 07:38 line was a pre-run planning figure for the time estimate. Also on record: fold-held-out scene `fa035232` reads 0.8924, below 0.90. The M-RENC bar is on the median (0.9427), and that scene improved too.
* 2026-11-07 14:21:54 (dev-box clock, from `date`), Master Mind. **C0C-RE result: RE-HARMS (card 6.3), as committed.** All gates pass (pinned read G1 to G4; pair G1 to G4 and ATTRIB), 48 scenes, 93 starts, paired scene-cluster bootstrap B 2,000. Pinned read against P10: NOT-DOMINANT at both seeds. RE-MOVES clauses: ADE6 C0C-RE − C0C −0.1956 [−0.4391, +0.0895] / −0.1002 [−0.3435, +0.1502], not separated (fails); M1 −0.0523 [−0.0852, −0.0249] separated-better / −0.0142 [−0.0450, +0.0112] (holds); a new separated-worse row against P10 at s1, heading MAE (fails). Beside: speed MAE −0.1234 / −0.1282 separated-better at both seeds; DK-HW +0.547 separated-worse at s0. Next, as committed: DAgger at T2 (`PREREG_T2_DAGGER.md`, `9234478c`). Files: `t2v8f/stage5/C0C-RE/RESULT_T2_RENC.{md,json}`; the BLIND pair is read beside when its pull lands.
