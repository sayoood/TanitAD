# H-ST-REGRET zero-training pre-step (deliverable 1)

Frozen cards: `FROZEN_CARDS.md` sha256 `3202a62fbf7699005c60f7a6ae37fb544e4bf0a0f0564b7a9115ee7bf865fd5b`; amendment 1 sha256 `4676bdb8d27c1095c942782889ab9a04932e5aa1cfe1fcda650285ca365b93c7`.
Evidence class MEASURED (ours; artifact `regret_prestep.json`, inputs `raw/bank_table_s{0,1}.npz` built from `v8f-dagger/bank_s{0,1}` and `sfresh_inputs/inputs/native_tracks.pt` sha256 1db431f1... = the bank header's). Tier T1-kin rollout states on TRAIN (the DAgger bank), not eval, not driving performance.
Estimator: clip-cluster bootstrap (4,512 clips, B = 2000, seed 0, 95 % percentile, construction of `taniteval/ci.py`). It answers 'another draw of CLIPS', not 'another training run' (`H-ESTIM-SEED-1`).

## Verdict (frozen R3): **MIDDLE BAND (report the multiplier; card neither dies nor is confirmed)**  -- shares / multipliers fall between the card's branches or the seeds split; CI straddles threshold: False

| seed | scope | usable records | crawl records | crawl share | 95 % CI |
|---|---|---|---|---|---|
| s0 | pooled | 407450 | 23435 | 5.752 % | [5.439, 6.062] % |
| s0 | tick1 | 242357 | 7627 | 3.147 % | [2.920, 3.396] % |
| s0 | tick2 | 165093 | 15808 | 9.575 % | [9.034, 10.162] % |
| s1 | pooled | 414765 | 10818 | 2.608 % | [2.443, 2.782] % |
| s1 | tick1 | 247102 | 2315 | 0.937 % | [0.844, 1.033] % |
| s1 | tick2 | 167663 | 8503 | 5.071 % | [4.732, 5.416] % |

Crawl state = own speed < 0.5 x logged speed at the matched row r' AND logged speed >= 3 m/s; usable = reason ok. Decision statistic = pooled row.

| seed | sampler-effective share | M_def (50:50 deficit-weighted) | M_max ceiling | own-state draws (dagger_n) | expected crawl in draw (record share) | (sampler-effective) |
|---|---|---|---|---|---|---|
| s0 | 4.905 % [4.631, 5.183] | 2.68 [2.60, 2.77] | 9.2 | 14688 | 845 [799, 890] | 721 [680, 761] |
| s1 | 2.072 % [1.936, 2.210] | 3.08 [2.99, 3.18] | 19.7 | 14839 | 387 [363, 413] | 308 [287, 328] |

Beside (not decisive): s0: crawl share over ALL records 6.13 %, over the 95930 off_lat/off_head records 8.79 %; s1: crawl share over ALL records 2.76 %, over the 83973 off_lat/off_head records 3.95 %

The card's own prediction (INHERITED, card text): share < 3 % and ~450-600 crawl samples in ~15-19k draws.
