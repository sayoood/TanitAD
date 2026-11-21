# RESULT_WEATHERPROBE: the zero-training probe on S1's frozen adapter, weather

Tier T0 (representation read at the NOW frame). Evidence class MEASURED (ours). Pre-registration: `PREREG_WEATHERPROBE.md` (sha256 0582c78384d35be54af8ed3397f8916cfa16da9e337ffe6676e5af8e74f1ae8e), written before any fit; one amendment before any fit (the eval features are the dev-box cache of the read, the pod's re-encoded eval cache is scored beside). Numbers: `weatherprobe.json` (sha256 4fec02b8...), logits `wprobe_out.npz`, fit code `wprobe_fit.py` (330b8b72...), scorer `wprobe_score.py`.

## Label: DETECTS(weather) by the probe, at BOTH S1 seeds

Estimator: paired episode-cluster bootstrap over the 141 eval episodes, B 2,000, seed 0 (`taniteval.ci`, read tree copy). It answers "another draw of EPISODES" only; the two S1 seeds (two separate trainings and two adapters) are the replicate pair, one probe fit each (probe seed 0).

| arm (eval-141, 2,399 windows) | accuracy | constant | delta | 95 % CI | separated | balanced acc |
|---|---|---|---|---|---|---|
| S1_s0 probe | 0.9358 | 0.8574 | +0.0784 | [+0.0225, +0.1305] | yes | 0.8050 |
| S1_s1 probe | 0.9387 | 0.8574 | +0.0813 | [+0.0283, +0.1326] | yes | 0.7802 |
| S1_s0 R1 (train labels shuffled between clips) | 0.8574 | 0.8574 | +0.0000 | [0, 0] | no | 0.2500 |
| S1_s1 R1 | 0.8574 | 0.8574 | +0.0000 | [0, 0] | no | 0.2500 |
| beside: shipped head (STRAT1_CTX.json), S1_s0 | 0.9037 | 0.8574 | +0.0463 | [-0.0108, +0.1014] | no | |
| beside: shipped head, S1_s1 | 0.9291 | 0.8574 | +0.0717 | [+0.0254, +0.1204] | yes | |
| beside: S1_s0 probe on pod-encoded eval features | 0.9362 | | +0.0788 | [+0.0229, +0.1314] | yes | |
| beside: S1_s1 probe on pod-encoded eval features | 0.9383 | | +0.0809 | [+0.0283, +0.1318] | yes | |
| beside R2: probe on a RANDOM adapter (seed 0, E 8) | 0.9287 | 0.8574 | +0.0713 | [+0.0171, +0.1247] | yes | 0.7947 |

Bar of the card and of the pre-registration (accuracy minus constant > 0 separated at both seeds, R1 NOT DETECTS at each): met. The probe lifts the shipped head by +0.032 (s0) and +0.010 (s1) accuracy points on the same windows; the s0 gain is the one that moves the seed from not separated to separated.

## Controls
* Disjointness (must read 0): table train vs eval clips 0; train cache files vs eval clips 0; train cache files vs eval grid order 0. 36,051 train windows from 4,572 train clips (8 NOW frames per clip, weather valid and flag 0), 0 clips skipped; class counts clear/fog/rain/snow 30,988 / 831 / 1,647 / 2,585.
* Instrument identity: the stored ctx_head applied through the script's own adapter code reproduces the T0 weather argmax on 100.00 % of 2,399 windows at both seeds (max |logit diff| 1.9e-05).
* Constant predictor reads exactly the constant (delta 0). R1 reads NOT DETECTS (it collapses to the majority class, balanced accuracy 0.25), so a detection here is not a pipeline artefact of the fit or the scorer.
* Hyper-parameter (epochs) chosen on a train-only 80/20 clip split: held-out accuracy minus constant, s0 E2/E4/E8 = +0.0913/+0.0931/+0.0946; s1 +0.0912/+0.0955/+0.0980; E = 8 chosen at both (the top of the pre-registered grid, still rising; a longer fit was not tried).

## What it does and does not say
* The weather information is present in S1's frozen adapter tokens at both seeds, and a learned attention pool over the 160 tokens reads it where the token-mean head did not (s0) or only just did (s1). The shipped head's NOT-DETECTED reflects its pooling, not an absent signal.
* ⚠️ R2 (random adapter) also detects, +0.0713. So the signal is largely carried by the cached DINO-class features and survives any random projection; the trained adapter adds little over a random one on this probe (s0 +0.0784 vs R2 +0.0713, one fit each; the difference is within what one fit cannot resolve). The probe shows the cached 4x10 tokens hold weather; it does not show S1's training put it there.
* The balanced accuracy (0.78 to 0.81) shows the rare classes (fog, rain, snow) are read, not just the majority.
* One probe fit per adapter (probe seed 0); the replicate is the other S1 seed. No inference-seed question arises (deterministic probe).

## Next lever (the card's)
DETECTS, so: **the next arm is that head**, an attention-pooling weather head on the adapter tokens (S1 plus this head), as a new card with a PI go for training; its bar should include an R2-style random-adapter or feature-only control so the head's gain is read against what the cache features carry alone. The 8x20 grid / 416x1024 encode lever is NOT triggered.

## Deliverable manifest
All in `F:/TanitAD-scratch/strat1/weatherprobe/` (single copy, not staged in git; the MM lands): PREREG_WEATHERPROBE.md (+ .sha256), RESULT_WEATHERPROBE.md, weatherprobe.json, wprobe_out.npz, wprobe_meta.json, wprobe_log.txt, wprobe_fit.py, wprobe_score.py.
Temporary data deleted: pod `/root/wprobe/` whole (550 MB: two checkpoint copies, eval feature tensor, inputs, logs, copies of the outputs above), local `eval_local_feats_u8.pt` (393 MB, re-creatable from the eval cache).
