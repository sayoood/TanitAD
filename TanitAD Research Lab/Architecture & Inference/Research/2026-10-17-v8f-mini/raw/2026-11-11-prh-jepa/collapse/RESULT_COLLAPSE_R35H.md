# COLLAPSE MONITOR on the banked ladder world models (W3 of PRH_JEPA_FOR_V8F.md)

Tier T0 latent diagnostic, never a driving number. Evidence class MEASURED. Data: `RESULT_COLLAPSE_R35H.json` (checkpoint sha256 and frames sha256 per arm).

## Reading rule (fixed BEFORE any number existed; kept as is once the numbers exist)
Statistic: PR/dim, with PR = (tr C)^2 / tr(C^2) of the covariance of the pooled latents of the 141 eval clips.
* PR/dim < 0.02: collapsed
* 0.02 <= PR/dim < 0.10: partial
* PR/dim >= 0.10: healthy

(The header first said "dim = 256"; the measured latent width is 128. The rule is on the ratio and did not change.)

## R35H (the arm of record), per level, per seed. dim = 128, n = pooled frames of 141 clips
| level (step) | s0 PR | s0 PR/dim | s0 dims 90 / 99 % | s1 PR | s1 PR/dim | s1 dims 90 / 99 % | dead dims | reading |
|---|---|---|---|---|---|---|---|---|
| L1 (0.2 s) | 10.21 | 0.080 | 23 / 62 | 9.42 | 0.074 | 22 / 62 | 0 / 0 | partial |
| L2 (0.6 s) | 10.31 | 0.081 | 22 / 63 | 9.79 | 0.077 | 21 / 61 | 0 / 0 | partial |
| L3 (3.0 s) | 7.93 | 0.062 | 17 / 58 | 8.20 | 0.064 | 17 / 57 | 0 / 0 | partial |

All six cells are partial by the rule; none is collapsed (nearest to the 0.02 line: 0.062). No dead dimension anywhere: the variance is concentrated in about 20 directions (90 %) while 57 to 63 of 128 dims are needed for 99 %.

## Other banked ladder arms, same rule (PR/dim, L1 / L2 / L3)
| arm | s0 | s1 | strides |
|---|---|---|---|
| R35H | 0.080 / 0.081 / 0.062 | 0.074 / 0.077 / 0.064 | 0.2 / 0.6 / 3.0 s |
| R35 | 0.126 / 0.087 / 0.063 | 0.125 / 0.097 / 0.066 | 0.2 / 0.6 / 3.0 s |
| S35 | 0.131 / 0.108 / 0.088 | 0.129 / 0.118 / 0.088 | 0.2 / 0.6 / 3.0 s |
| C0 | 0.128 / 0.119 / 0.093 | 0.125 / 0.120 / 0.099 | 0.2 / 0.4 / 0.8 s |
| R35b (s0) | 0.126 / 0.087 / 0.063 | | identical to R35_s0 to the printed digits (R35B_CKPT_IDENTITY.json is the owner of that fact) |

Plain reading: no ladder arm is collapsed. R35H sits at partial on all three levels, with L1 about 40 % lower in PR than R35, S35 or C0 (10 against 16); the 3.0 s level is partial for every arm. Whether the R35H L1 shortfall is a training effect or a seed-shared property is not decided here (two seeds agree: 10.2 and 9.4). It is a latent-geometry diagnostic only.

## PR against step
No intermediate checkpoint is banked for any ladder arm (every pull holds `ckpt_last.pt` only), so the paper's abrupt-collapse signature (PR against step) cannot be read. The only per-step signal is the trainer log (384 rows per arm): `wm/sigreg/L1` of R35H_s0 runs 27.8 (step 50), 5.9 (2,450), 8.2 (7,250), 5.7 (9,650), 3.2 (16,850); R35H_s1 27.1, 4.7, 4.2, 5.9, 5.3. That is a SIGReg proxy from the trainer log, INHERITED-class for the question, not an effective rank, and shows no abrupt jump in the sampled rows (8 of 384 per arm looked at).

## Not verified
Basis (`--wm-basis slow|pca`) arms: the code path through `slowwm_hook` is written, not run on a real basis arm (none banked in the ladder). Covariance pools all frames of the 141 clips, not only `ok` frames.
