# RESULT_PRH_SEED (2026-11-11 10:50:40)

**Outcome: O2 PLATONIC-OTHER**  (P1=True, P2=False; controls all pass: False)

**A CONTROL FAILED: the read is UNUSABLE (instrument failure); do not interpret the outcome.**

Evidence class: MEASURED, ours; artifact `RESULT_PRH_SEED.json` beside this file. Representation probe, no driving tier.

## M (mean canonical correlation, score half, 95 % episode-cluster bootstrap CI, B=2000)

| M | point | CI95 |
|---|---|---|
| M(TR0,TR1) | 0.6773 | [0.6718, 0.6809] |
| M(RI0,RI1) | 0.6344 | [0.6274, 0.6398] |
| M(TR0,SLOW) | 0.5756 | [0.5683, 0.5807] |
| M(TR1,SLOW) | 0.5743 | [0.5665, 0.5797] |
| M(RI0,SLOW) | 0.5617 | [0.5527, 0.5682] |
| M(RI1,SLOW) | 0.5617 | [0.5527, 0.5682] |
| M(TR0,PCA) | 0.7563 | [0.7520, 0.7594] |
| M(TR1,PCA) | 0.7595 | [0.7551, 0.7626] |
| M(TR0,DMIX) | 0.6796 | [0.6733, 0.6840] |
| M(TR0,RANDAFF)[diagnostic, no bar] | 0.6464 | [0.6405, 0.6510] |
| M(TR0,TR0) | 1.0000 | [1.0000, 1.0000] |

## Paired differences (prediction rows)

| diff | point | CI95 | separated, positive |
|---|---|---|---|
| P1: M(TR0,TR1)-M(RI0,RI1) | 0.0429 | [0.0360, 0.0489] | True |
| P2a_s0: M(TR0,SLOW)-M(RI0,SLOW) | 0.0139 | [0.0050, 0.0226] | True |
| P2b_s0: M(TR0,SLOW)-M(TR0,PCA) | -0.1807 | [-0.1882, -0.1742] | False |
| P2a_s1: M(TR1,SLOW)-M(RI1,SLOW) | 0.0126 | [0.0044, 0.0210] | True |
| P2b_s1: M(TR1,SLOW)-M(TR1,PCA) | -0.1852 | [-0.1926, -0.1784] | False |

## Controls

```json
{
 "C-SELF": {
  "M": 1.0,
  "pass": true
 },
 "C-ROT": {
  "M_rotated": 0.6772927674456499,
  "M_plain": 0.6772927674456111,
  "absdiff": 3.885780586188048e-14,
  "pass": true
 },
 "C-SHUF": {
  "M": -0.0009355371446288246,
  "ci95": [
   -0.002472641770318832,
   0.000589611882731792
  ],
  "M(RI0,RI1)": 0.6344308037331724,
  "mutual_10NN": 0.006898907103825138,
  "chance": 0.006830601092896175,
  "pass": true
 },
 "D-MIX": {
  "M(TR0,DMIX)": 0.6795979124040865,
  "M(TR0,TR1)": 0.6772927674456111,
  "delta": 0.002305144958475447,
  "ci95": [
   -0.004424217756568228,
   0.009166690734529241
  ],
  "reported_change": true,
  "pass": false
 },
 "C-AFFINE(extra)": {
  "max_abs_pooled_adapt_mean_nm_vs_zmean_tr0": 5.7220458984375e-06,
  "note": "nm-affine identity of the card; no bar in the card"
 },
 "C-ZC": {
  "s0": [
   {
    "clip": "01be5919-6e61-45ba-a193-5f447efb4b51",
    "T": 101,
    "max_abs_diff": 9.5367431640625e-07,
    "ok_equal": true
   },
   {
    "clip": "060531e1-9935-482e-8c25-2abf37bdc41b",
    "T": 101,
    "max_abs_diff": 1.9073486328125e-06,
    "ok_equal": true
   },
   {
    "clip": "0fa2b231-435e-4700-9138-30f333dc5d6e",
    "T": 101,
    "max_abs_diff": 9.5367431640625e-07,
    "ok_equal": true
   }
  ],
  "s1": [
   {
    "clip": "01be5919-6e61-45ba-a193-5f447efb4b51",
    "T": 101,
    "max_abs_diff": 2.384185791015625e-07,
    "ok_equal": true
   },
   {
    "clip": "060531e1-9935-482e-8c25-2abf37bdc41b",
    "T": 101,
    "max_abs_diff": 2.384185791015625e-07,
    "ok_equal": true
   },
   {
    "clip": "0fa2b231-435e-4700-9138-30f333dc5d6e",
    "T": 101,
    "max_abs_diff": 1.7881393432617188e-07,
    "ok_equal": true
   }
  ],
  "pass": true
 }
}
```

## Beside rows (no bar)

```json
{
 "TR0_vs_TR1": {
  "procrustes_R2": 0.2399802101950801,
  "mutual_10NN": 0.8890710382513662,
  "chance_10NN": 0.006830601092896175,
  "N_knn": 1464,
  "profile_along_SLOW": "see JSON (256 values)"
 },
 "RI0_vs_RI1": {
  "procrustes_R2": 0.07297122976013115,
  "mutual_10NN": 0.8732923497267759,
  "chance_10NN": 0.006830601092896175,
  "N_knn": 1464,
  "profile_along_SLOW": "see JSON (256 values)"
 }
}
```

Profile TR0_vs_TR1: mean over SLOW directions 1-16 0.930, 17-64 0.736, 65-256 0.605.
Profile RI0_vs_RI1: mean over SLOW directions 1-16 0.898, 17-64 0.702, 65-256 0.546.

Ridge 1e-3 sensitivity (point): {"M(TR0,TR1)": 0.672846648529103, "M(RI0,RI1)": 0.6355165935616262}

n: {"clips": 141, "fit_clips": 71, "score_clips": 70, "fit_frames": 7173, "score_frames": 7064, "slow_pairs": 6818, "B": 2000}

## Choices the card left open

* **cca_ridge_M**: 1e-6 x mean eigenvalue of each side (numerical only; float64). Sensitivity row with 1e-3 reported beside.
* **ridge_beside**: 1e-3 x mean eigenvalue for the whitening used by Procrustes, mutual 10-NN, per-direction profile.
* **slow_ridge**: 1e-3 x mean eigenvalue (card), applied to each side (x_t and x_t+5) of the temporal CCA; SLOW rep = (nm - mean) @ top-256 x-side canonical directions.
* **slow_pairs**: pairs (t, t+5) in the same clip, both frames ok, fit-half clips only.
* **corr_definition**: per-direction Pearson correlation on the score half (score-half means and variances), directions fixed from the fit half.
* **ok**: ok[t] = isfinite(p10f[t]).all() | t<1 from ClipStore.get_kin + kin_features (as theory/ladder/zcache.py clip_inputs).
* **separated**: 95 % percentile CI of the paired difference (same resample counts) has lower bound > 0.
* **bootstrap**: B=2000, rng seed 0, clips of the score half resampled with replacement, fit-half directions fixed; one set of counts shared by every M, so differences are paired.
* **split**: eval clip ids sorted; even rank fit, odd rank score.
* **random_init**: V8FMini built from the checkpoint cfg under torch.manual_seed(100/101), adapter and pos at init, no weights loaded.
* **pca_slow_rep**: reps are the projections of nm onto the fit-half directions (centred by the fit-half mean).
* **knn**: mutual 10-NN: cosine in each rep's fit-half-whitened centred space, score frames with t%5==0 and ok (1 per second), overlap/10 averaged; chance 10/N.
* **profile**: for SLOW direction k: least-squares readout of SLOW_k from whitened A (fit half); apply it to A and to Procrustes-rotated B on the score half; Pearson correlation of the two readouts. Ordered by SLOW's fit-half canonical correlation.
* **D-MIX**: random affine map of nm = nm @ G + b, G ~ N(0,1) [1024,256] seed 202, scaled so its total fit-half variance equals TR0's; D-MIX = 0.5*TR0 + 0.5*that. Pass if the paired CI of M(TR0,DMIX) - M(TR0,TR1) is separated from 0 (either sign).
* **C-SHUF**: TR1 frames permuted (rng seed 5) across clips within the fit half and within the score half (pairing broken in both), CCA refit on the permuted fit half. Pass if M < M(RI0,RI1) (point) and mutual 10-NN <= 2 x chance.
* **C-ROT**: random orthogonal Q (seed 7, QR of a Gaussian) applied to TR1.
