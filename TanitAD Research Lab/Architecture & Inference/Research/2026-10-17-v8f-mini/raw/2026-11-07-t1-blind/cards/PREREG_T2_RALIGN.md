# PREREG_T2_RALIGN: a render-to-real token map at T2, cross-fitted by scene (Master Mind, 2026-11-07 05:18 dev-box clock, from `date`; written before any GAP48 token exists and before any adapted rollout)

## 1. Why
* `theory/PREREG_T1_BLIND.md` (frozen `9619414a`) read **VISION-USED** at T1-kin, so by its section 5 the T2 reading is **RENDER-GAP-BINDS**:
  * on real frames, removing C0's scene tokens costs ADE6 +0.83 / +0.82 m;
  * on renders, removing them does not move M1;
  * the stage-2 means show BLIND at or below C0 on ADE6 (7.729 against 7.980 on launch starts, 6.794 against 7.336 on moving starts).
* The card commits a render-robustness lever aimed at T2. This one is the cheapest in that family: no training. It is a fixed map applied to the render tokens before the policy sees them.
* **What it is and is not:** it corrects the T2 instrument's camera. It is not a driving lever, because a real car sees real frames. Every number it produces is stamped **T2-RA** and is never pooled with T2 or T1-kin numbers.

## 2. Data and folds (fixed now)
* **Pairs:** GAP48 (`t2v8f/gap48/`). On each of the 48 read scenes, at every 10th tick, it holds the tokens of the frame rendered at the logged pose (`objects`) and of the real frame at that tick, through the in-loop encoder (fp16 [40, 1024]).
* **Folds:** scene s goes to fold A if the first hex digit of sha256(scene id) is even, and to fold B otherwise. The map applied to a scene's renders is always the one fitted on the **other** fold. No scene is read under a map that saw it.

## 3. The map (one family, fixed now)
* **Per position p and channel c, an affine map** x̂[p,c] = a[p,c] · r[p,c] + b[p,c], fitted by ordinary least squares on the fold's pairs (r = render token, x = real token). That is 40 × 1024 × 2 parameters per fold.
* If a cell has fewer than 200 pairs, the map is not fitted and the outcome is NOT EVALUABLE.
* At T2 the map sits between the in-loop encoder and the policy (an `AlignPolicy` wrapper beside `t2v8f_kit.BlindPolicy`). Ego inputs and the executor are unchanged. P10 does not read the image (P10/RR equals P10/T2 in `stage2/SCORE2.md`), so the map cannot change it, and a gate checks that.

## 4. Gates
* **M-ALIGN (mechanism, on the held-out fold of every scene).** The median flattened cosine of (x̂, x) rises above that of (r, x) by at least 0.05, and reaches at least 0.90 (the real-frame 1 s level in `gapscale/GAPSCALE.json`). If not: **ALIGN-FAILS-MECH**. No T2 stage runs, and the next lever is encoder adaptation (below).
* **G-IDMAP:** with a := 1 and b := 0, the wrapped composite reproduces C0C's stage-4 cells bit for bit on 2 scenes.
* **G-P10:** P10 under the wrapper equals P10 exactly.
* Each gate has a deliberate-regression arm that must fail it: a map fitted on the SAME fold for M-ALIGN, a perturbed identity for G-IDMAP.

## 5. The driving read (stage 5 on box 2, the 48 scenes, both seeds)
* **Arm C0C-RA:** C0C with the cross-fitted map.
* Read by the pinned dominance read (pin 4, `--arm C0C-RA --arm-stage 5`), plus the pair read C0C-RA − C0C (`read_t2_pair.py`, after its identity). Both use the paired scene-cluster bootstrap, B 2,000, seed 0. All four families; STRATEGIC PAUSED (PI item 25).
* **Beside, deciding nothing:** C0C-RA-BLIND − C0C-RA, the T2 token-use check of `DRAFT_PREREG_T2_DOM_LORA.md` open item 5(b), run only if box time allows.

## 6. Outcomes, committed now
1. **RA-DOMINANT:** the pinned read gives DOMINANT-T2 at both seeds.
   * It is reported as written, as a T2-RA result with renders beside P10's. It is not counted as the scale signal until a replicate reads DOMINANT too: a second map fitted with the folds swapped by the parity of the second hex digit.
2. **RA-MOVES:** C0C-RA − C0C is separated-better on ADE6 at both seeds, M1 is not separated-worse at either seed, and no new separated-worse row appears against P10.
   * The map joins the T2 instrument for every vision arm, LC included, declared as T2-RA. The LoRA arm's T2 read is run with it and without it, and both are reported.
3. **RA-NULL or RA-HARMS:** anything else.
   * The next lever is encoder adaptation to renders: a LoRA on the encoder with a render-to-real token-matching loss on the GAP48 pairs, under its own card, on the fold split above.
   * The T2 instrument stays as it is.
4. **ALIGN-FAILS-MECH** (section 4): the same next lever as 3, with no T2 stage spent.

## 7. Scope and evidence
* Tier T2-RA: the T2 tier, with this map in front of the policy. The estimator answers another draw of scenes. No training enters. The map is deterministic given its fold, and the two seeds of C0 are read side by side.
* Admissibility: the map is a fixed function of vision tokens. It never sees ego state, labels or the scene it is applied to. The real frames enter only offline, from the other fold.

## 8. Log (dated, append-only after the freeze)
