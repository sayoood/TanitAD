INSTRUMENT-PASS

# MAPEDGE Q1 (mechanical; PREREG_MAPEDGE sha256 31a53010eed1)

tier: perception instrument (map head vs SAM3 GT raster), eval139 per-frame windows; not a planning or driving metric
estimator: pooled window mean; episode-cluster bootstrap over episodes, B 2000, seed 0 -- 'another draw of EPISODES' only; one training seed; the map head is deterministic given the frames

| X | recall @0.3 m | precision @0.3 m | rec 0-20 | rec 20-40 | rec 40-60 | share on GT nocls | share on GT unseen |
|---|---|---|---|---|---|---|---|
| OWN-CODE5 | 0.0019 [0.0010, 0.0030] (n_win 4538, n_ep 136) | 0.7862 [0.6321, 0.8806] (n_win 142, n_ep 24) | 0.006 | 0.000 | 0.000 | 0.097 | 0.000 |
| OWN-BND | 0.2907 [0.2671, 0.3147] (n_win 4538, n_ep 136) | 0.2095 [0.1877, 0.2312] (n_win 4686, n_ep 137) | 0.412 | 0.277 | 0.195 | 0.138 | 0.028 |
| DER-BND | 0.0954 [0.0880, 0.1034] (n_win 4538, n_ep 136) | 0.0783 [0.0703, 0.0869] (n_win 4686, n_ep 137) | 0.140 | 0.079 | 0.068 | 0.205 | 0.033 |
| GT-BND | 0.9641 [0.9564, 0.9717] (n_win 4538, n_ep 136) | 0.8548 [0.8406, 0.8675] (n_win 4567, n_ep 136) | 0.965 | 0.966 | 0.970 | 0.000 | 0.000 |

| bar | pass | paired cell |
|---|---|---|
| B1_rec_OWNBND_minus_OWNCODE5 | True | 0.2888 [0.2659, 0.3121] (n_win 4538, n_ep 136) |
| B2_rec_OWNBND_minus_DERBND | True | 0.1953 [0.1728, 0.2185] (n_win 4538, n_ep 136) |
| B3_prec_OWNBND_minus_DERBND | True | 0.1312 [0.1122, 0.1510] (n_win 4686, n_ep 137) |

controls failed: none
GT-BND = GT drivable cells next to a JUDGED non-drivable cell (reference ceiling, not an arm).

NEXT: Q2: the OWN-BND-T12 selection arm through MAPSEL's scorer (sealed separately before it runs).

ZZMEVERDICTENDZZ