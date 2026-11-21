# PREREG_PRH_SEED: are v8f-mini's trained adapters "platonic" across seeds, and do they sit on the good-view (slow) subspace? (frozen 2026-11-11 ~11:45, before any data is read)

Source of the hypothesis: Balestriero & LeCun, *Proving and Improving the Platonic Representation Hypothesis with JEPAs*, arXiv:2610.39302 (2026-10-29), PUBLISHED.
* Prop. 1: a JEPA with an identity predictor and whitened (second-order) anti-collapse learns the top-k eigenvectors of the graph its positive pairs draw; it is unique up to rotation when the spectrum has a gap.
* Thm. 1 and 2: runs with the same positives agree up to one rotation; with different positives, their misalignment is bounded by the kernel mismatch over the eigengap.
* Fig. 6: alignment lives in the leading spectral directions.
* The paper does NOT cover action-conditioned predictors (its §2.1 says so). This card tests its predictions on our model as an instrument and does not assume they transfer.

## Why this is testable cheaply on v8f-mini
The shared adapter is **affine** in the layer-normalised frozen DINOv3 features (`stack/tanitad/models/v8f_mini.py:1119-1125`: `adapter(f) + pos`, and the docstring of `pooled_adapt_mean`: `Linear(LayerNorm(f)) + pos`). The token-mean the world model reads (`zmean`, 256-d; `WMAux.z1_from_mean`, `v8f_mini.py:895`) is therefore an affine map of the token-mean of the normalised features, `nm_t` (1024-d).
⇒ Two adapters agree up to rotation (after whitening) exactly when their row spaces agree in the feature metric. Measuring that is CCA between the two 256-d representations. No training is needed, and every representation below is computed from the same frozen features.

## Representations (all on the eval-141 clips, frames with `ok`)
| tag | what |
|---|---|
| TR0, TR1 | `zmean` of the trained adapters of `M2b_s0` and `M2b_s1` (`E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s{0,1}/ckpt_last.pt`), computed through the model's own code (read tree `F:/TanitAD-scratch/otband_read/tree`) from the eval cache `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache` |
| RI0, RI1 | the same adapter module, freshly initialised by the model's own constructor under torch seeds 100 and 101 (untrained; the random-init baseline) |
| SLOW | the closed-form linear "good-view JEPA" of Prop. 1 with temporal positives: the top-256 canonical directions between `nm_t` and `nm_{t+5}` (lag 1.0 s, same clip), ridge 1e-3 × mean eigenvalue, fit on the fit half |
| PCA | the top-256 principal directions of `nm`, fit on the fit half |

**Split (fixed rule):** eval clips sorted by clip id; even rank = fit half, odd rank = score half. All whitening, CCA and Procrustes fits use the fit half only. Every number is scored on the score half.

## Metric
`M(A,B)` is the mean of the 256 canonical correlations between A and B. The CCA directions are fit on the fit half, and the correlations along them are evaluated on the score half (out of sample).

Beside rows, which have no bar:
* the R² after the best orthogonal Procrustes rotation of whitened B onto whitened A;
* mutual 10-NN over score frames subsampled to one per second (chance 10/N);
* the per-direction cross-seed correlation after Procrustes, ordered by SLOW's canonical correlations (Fig. 6 analogue).

**Estimator:** episode-cluster bootstrap over the score clips (B = 2,000, 95 %, `taniteval.ci`-style percentile; resample clips, recompute the score-half correlations with the fit-half directions held fixed). It answers "another draw of clips". It does not answer "another training run": the only training-run evidence here is s0 against s1, which is the question itself.

## Predictions (both outcomes committed)
* **P1 (seed platonicity):** `M(TR0,TR1) − M(RI0,RI1) > 0`, separated.
* **P2 (good-view subspace):** for EACH trained seed s ∈ {0,1}, both
  * `M(TRs,SLOW) − M(RIs,SLOW) > 0`, separated, and
  * `M(TRs,SLOW) − M(TRs,PCA) > 0`, separated.

  P2 holds only if all four differences are separated and positive.

| outcome | condition | committed next step |
|---|---|---|
| **O1 PLATONIC-SLOW** | P1 and P2 | Training drives the adapter toward the slow (temporal good-view) subspace. Next: the closed-form SLOW adapter as a seed-free candidate. Pre-register a ladder WM arm on it against the M2b adapter (box-2 slot, PI's call), and in the same turn run the zero-training per-direction predictability read of the WM latent. |
| **O2 PLATONIC-OTHER** | P1, not P2 | Seeds converge on a shared subspace that is not the slow one, plausibly task-driven by the planner losses. Next (zero training, same turn): the overlap of TR with the linear-regression subspace of the 60-d tactical code. SLOW then becomes a WM-only candidate, not a replacement. |
| **O3 NOT-PLATONIC** | not P1 | The trained adapter is no more seed-stable than a random one, so it adds training variance without converging. Next: the deterministic SLOW adapter as a variance-removal lever for the WM path, with the ladder arm pre-registered (box-2 slot, PI's call). |

## Controls (each must read its known value, or the read is unusable and reported as an instrument failure)
* **C-SELF:** `M(TR0,TR0) = 1` within 1e-6.
* **C-ROT:** a random orthogonal rotation applied to TR1 leaves `M(TR0,TR1)` unchanged within 1e-6, since CCA is rotation invariant.
* **C-SHUF:** TR1 frames permuted across score clips (pairing broken) must give `M < M(RI0,RI1)` and a mutual 10-NN within 2× chance.
* **C-ZC:** the locally computed TR0/TR1 `zmean` for 3 eval clips equals box-2's cached `D:\tanitad\zcache\M2b_s{0,1}\eval\<clip>.npz` `zmean` to max abs diff ≤ 1e-4. This is a read of 3 small files over sftp; no process runs on box 2.
* **Deliberate regression (D-MIX):** replace TR1 by `0.5·TR0 + 0.5·(an independent random affine map of nm)`. `M(TR0, D-MIX)` must differ from `M(TR0,TR1)`, and the harness must report the change. A harness that cannot see it is unusable.

## Scope, tier and families
* This is a **representation probe, not a driving eval**: no tier T0/T1 applies, and the four metric families do not apply. It decides only which adapter lever earns a box-2 slot.
* No result here is quoted as driving performance.
* CPU only, zero training, no box-2 compute.
* Parity untouched: the eval-141 clips are read and never re-selected.
