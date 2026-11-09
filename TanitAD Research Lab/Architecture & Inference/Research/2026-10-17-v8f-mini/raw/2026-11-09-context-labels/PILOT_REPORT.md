# PILOT_REPORT: per-frame strategic labels, first 200 clips (Master Mind, 2026-11-08)

All numbers are MEASURED from `F:/strat_out/pilot/PILOT_STATS.json` unless marked. 200 clips (187 train, 13 eval), 25 countries, 20,197 frames at 0.2 s. The sample is stratified over country and clock night/day, so the shares below are not corpus shares. These are label-pipeline facts, not model results. No label reaches any inference path.

## State
* Merged with no schema errors. 66 columns, 34 kB per clip, about 0.15 GB of parquet for all 4,659 clips (ESTIMATED).
* 3 of the 200 clips decode 100 frames, the rest 101: one row per frame that exists.
* Illumination comes from SigLIP. Weather, surface, visibility and view quality come from Qwen at 1 Hz (20.8 % of frames) and a probe on the SigLIP embeddings for the other 79.2 %.

## Label distributions (share of frames, after the pipeline)
| field | shares |
|---|---|
| illumination | day 37.9, lit night 28.1, dark 22.2, glare 9.1, twilight 2.7 |
| weather | clear 88.8, snow 8.2, rain 1.5, fog 0.8, unknown 0.7 |
| surface | dry 82.0, snow or ice 8.8, wet 8.4, unknown 0.8 |
| visibility | reduced 66.8, good 33.2 |
| view quality | wiper 88.0, clean 5.7, glare 4.3, unknown 1.7, tunnel 0.3 |

## Do the two models agree (argmax at the Qwen frames, 4,197 frames)
| field | agreement | reading |
|---|---|---|
| weather | 85.7 % | usable pair |
| surface | 44.4 % | SigLIP calls 71.5 % of Qwen's dry frames wet (2,164 of 3,027): SigLIP zero-shot is biased on this field |
| view quality | 10.0 % | neither is credible: Qwen says wiper on 3,241 of 4,197 frames, SigLIP says glare |
| visibility | not comparable | Qwen only; no SigLIP prompts exist. 66.8 % reduced looks high (HYPOTHESIS) |

## Illumination against the clock (a weak cross-check, never a label)
* Clock night frames: 6,968. The image reads day or glare on 3.8 % of them.
* Agreement with the estimated sun band: 93.2 %. Spearman of sky luminance with estimated sun elevation: 0.667.
* The hand-written brightness rule agrees with SigLIP on only 38.6 % of frames. It is an uncalibrated first guess and is not used for the label.

## Temporal stability (share of consecutive frame pairs that change label)
| field | all frames (0.2 s) | Qwen frames only (1 s) |
|---|---|---|
| weather | 6.2 % | 11.8 % |
| surface | 7.2 % | 13.3 % |
| visibility | 3.9 % | 5.4 % |
| view quality | 8.1 % | 16.8 % |
| illumination | 4.7 % | 7.2 % |

Weather and surface should barely change inside 20 s. These rates say the per-frame labels need temporal smoothing or a per-clip decision.

## Cost (ESTIMATED from pilot rates, linear in clips, ignoring lock waits)
| option | GPU time for 4,659 clips |
|---|---|
| image statistics + SigLIP only | 2.8 h |
| + Qwen at 1 Hz on every clip | 26.0 h (14.7 h of work, the rest model loads in 30-minute lock-held jobs) |

SigLIP 129 frames/s. Qwen 7.4 prompts/s, 11.3 s per clip. Model load 244 s (SigLIP) and 653 s mean, 1,292 s maximum (Qwen), the latter under memory pressure.

## Open until the PI's blind audit
`F:/strat_out/pilot/PILOT_AUDIT.html`, 200 frames, stays local because it holds camera frames. Nothing scales until the PI has compared it. Proposals to decide with that result:
1. Drop view quality, which is not on the PI's list and neither model reads credibly, or re-prompt it before any use.
2. Temporal smoothing or a per-clip decision for weather and surface.
3. Qwen on a subset of clips with the probe filling the rest, if the audit shows the probe is good enough.

## The PI's blind audit (2026-11-08, 60 frames: 40 random, 20 chosen rare; answers `pilot_audit_answers_PI_60.json`, exported before the reveal; frame 39 has no visibility answer)
All 60 frames are Qwen frames. Agreement on the 40 random frames, Wilson 95 % interval, n small, each frame from a different clip. MEASURED.

| label | source | agreement with the PI |
|---|---|---|
| weather | Qwen, SigLIP (each) | 35/40 = 87.5 % [74, 95] |
| surface | Qwen | 34/40 = 85.0 % [71, 93] |
| surface | SigLIP zero-shot | 12/40 = 30.0 % [18, 45] |
| visibility | Qwen | 20/40 = 50.0 % [35, 65] |
| illumination | SigLIP | 29/39 = 74.4 % [59, 85]; 30/39 = 76.9 % with glare folded into day |
| illumination | brightness rule | 19/39 = 48.7 % [34, 64] |
| rough time of day | clock-based sun band | 29/39 = 74.4 % [59, 85] |

Over all 60: the PI called 4 frames glare and the label matched none. Of the frames the PI called good visibility, 18 of 35 were labelled reduced.

**Cheap probe (logistic regression on the stored SigLIP embeddings, Qwen labels from the other pilot clips, out of fold, `score_pilot_audit.py`).** On the 37 random frames from train clips:

| label | probe vs PI | Qwen vs PI, same frames |
|---|---|---|
| weather | 33/37 = 89.2 % [75, 96] | 32/37 = 86.5 % |
| surface | 34/37 = 91.9 % [79, 97] | 32/37 = 86.5 % |
| visibility | 20/37 = 54.1 % [38, 69] | 19/37 = 51.4 % |

Out-of-fold probe against Qwen over 3,924 frames at 1 Hz: weather 84.8 %, surface 83.3 %, visibility 90.7 %. On this sample the probe is not worse than Qwen itself, trained on only 187 clips' labels. The intervals are wide and overlap, so this does not show it is better.

**Reading.** Weather and surface are usable, and a probe can fill frames without Qwen on every clip. Illumination comes from SigLIP, glare is unreliable. Visibility is not usable as it stands. View quality was not audited (both models not credible).

**Option for the full run (ESTIMATED from pilot rates, ignoring lock waits):** SigLIP on all clips 2.8 h plus Qwen on 1,000 clips about 5 h (10 lock-held jobs), about 8 h in total, against 26.0 h with Qwen on every clip.
