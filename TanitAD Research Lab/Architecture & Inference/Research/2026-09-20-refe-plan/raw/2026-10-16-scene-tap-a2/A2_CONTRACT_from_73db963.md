### A2 (2026-10-16 ~00:45 Berlin, Master Mind; written AFTER the 2,000-update result, BEFORE any 10,000-update unit exists): does the head under-fit, or do the frozen features fall short?

**Why.**
- The 2,000-update grid read `ROW4_SCENE_ELIMINATED`, and its in-sample read was **D2**: V in-sample 0.704, held-out 0.637.
- D2 at a single update budget cannot tell two causes apart:
  - *(a)* the frozen visual features cannot express the road at 0.80;
  - *(b)* the head was under-trained (2,000 updates × batch 8 is about 19 passes over ~850 train tokens, cosine to 0).
- The registered next lever, an unfrozen-trunk graft, is a PI decision and costs A40 time. ⇒ Rule Zero: run the cheapest discriminator first.
- **Nothing in sections 2–5 changes.** ROW4 stands as read. A2 adds one arm and its own committed outcomes.

**Arm U10k.**
- The same as arm V, seed 0, with the same folds (`folds.json` sha256 `20dec343…`), tokens, labels, head and optimizer.
- The ONE change is **`n_updates` = 10,000**. The learning-rate schedule is the code's own function of `n_updates`, unchanged.
- Units go to a SEPARATE work dir. The config sha256 differs, so the code refuses to mix them with the 2,000-update units.
- Only arm V, seed 0, is run (5 units). No S, no C, no other seeds.

**Reads (same estimator: paired log-cluster bootstrap over the 123 logs, 10,000 resamples, seed 20260927).**
- **H:** U10k's pooled held-out IoU [0, 40).
- **I:** U10k's in-sample IoU [0, 40) (train folds).
- **Δ = H(U10k) − H(V, seed 0, 2,000 updates).** Paired on the same tokens; "separated" means beyond V's seed floor, 0.0036.

**Committed outcomes (exhaustive, applied in this order, thresholds compared on unrounded values).**

| row | condition | reading | next lever |
|---|---|---|---|
| **U1** | H ≥ 0.80 | the 2,000-update budget was the limit; frozen visual features carry the road | S1c: S1's visual head trained ≥ 5× longer (cost it on the dev box first, then the A40) |
| **U2** | H < 0.80, I ≥ 0.80, Δ separated > 0 | longer training helps and the head can fit the features; generalisation is the limit | S1c with more updates AND more label logs, each as its own registered pair |
| **U3** | H < 0.80, I < 0.80 | the head cannot fit frozen visual features even at 5× updates ⇒ **the frozen trunk is the limit** | the trunk: the graft (PI decision) and P6's trained trunk |
| **U4** | H < 0.80, I ≥ 0.80, Δ not separated or ≤ 0 | it memorises without generalising | more label logs is the only frozen-trunk lever; the trunk otherwise |

**Reported, never gating:**
- the near-boundary and overall SDF MAE;
- the per-fold IoU;
- the train-loss curve's last 10 % (did it plateau?);
- the wall time.
