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
