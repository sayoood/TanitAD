| read-out (TRAIN fold-0 holdout, n=56860, n=57231, n=56403, n=57231, n=57231) | dim | lon_v_target_ms | v_a_ms | v_end_ms | v_lo_ms | v_hi_ms | mean (SD units) | mean skill |
|---|---|---|---|---|---|---|---|---|
| mlp_ego (fit.json, banked) |  | 0.3290 | 0.1036 | 0.3291 | 0.2313 | 0.2444 | **0.24751** | 0.9350 |
| lin_ego (fit.json, banked) |  | 0.3707 | 0.1082 | 0.3753 | 0.2778 | 0.2716 | **0.28071** | 0.9158 |
| lin (scene+ego ridge, fit.json, SPEEDREAD primary) |  | 0.3572 | 0.1054 | 0.3651 | 0.2710 | 0.2631 | **0.27236** | 0.9208 |
| mlp (scene+ego MLP, fit.json) |  | 0.3717 | 0.1560 | 0.3749 | 0.2803 | 0.2769 | **0.29196** | 0.9125 |
| MLP REFIT full = mlp_ego (control) | 22 | 0.3290 | 0.1036 | 0.3291 | 0.2313 | 0.2444 | **0.24751** | 0.9350 |
| MLP (a) p10f + frame_ok | 12 | 0.4131 | 0.1112 | 0.4260 | 0.2953 | 0.2969 | **0.30849** | 0.8969 |
| MLP (b) p10f + limit + limit_ok + frame_ok (no nav) | 17 | 0.3518 | 0.1054 | 0.3557 | 0.2546 | 0.2482 | **0.26313** | 0.9259 |
| MLP (c, extra) p10f + nav + frame_ok (no limit) | 17 | 0.3720 | 0.1080 | 0.3813 | 0.2538 | 0.2864 | **0.28032** | 0.9156 |
| RIDGE REFIT full = mlp_ego (control) | 22 | 0.3707 | 0.1082 | 0.3753 | 0.2778 | 0.2716 | **0.28071** | 0.9158 |
| RIDGE (a) p10f + frame_ok | 12 | 0.4357 | 0.1144 | 0.4472 | 0.3083 | 0.3162 | **0.32437** | 0.8858 |
| RIDGE (b) p10f + limit + limit_ok + frame_ok (no nav) | 17 | 0.3724 | 0.1084 | 0.3782 | 0.2824 | 0.2722 | **0.28273** | 0.9146 |
| RIDGE (c, extra) p10f + nav + frame_ok (no limit) | 17 | 0.4291 | 0.1140 | 0.4375 | 0.3008 | 0.3135 | **0.31899** | 0.8897 |
| persistence: cell := measured speed v (no fit) |  | 0.5098 | 0.1885 | 0.5181 | 0.3914 | 0.4371 | **0.40898** | 0.8267 |
| CONST (train mean) |  | 1.0231 | 1.0302 | 1.0197 | 1.0268 | 1.0261 | **1.02519** | 0.0000 |

RMSE in m/s on the same rows, and skill = 1 - MSE / MSE(train-mean CONST) per cell:

| read-out | lon_v_target_ms RMSE m/s | v_a_ms RMSE m/s | v_end_ms RMSE m/s | v_lo_ms RMSE m/s | v_hi_ms RMSE m/s | lon_v_target_ms skill | v_a_ms skill | v_end_ms skill | v_lo_ms skill | v_hi_ms skill |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_ego (fit.json, banked) | 2.566 | 0.799 | 2.548 | 1.838 | 1.806 | 0.8966 | 0.9899 | 0.8958 | 0.9492 | 0.9433 |
| lin_ego (fit.json, banked) | 2.891 | 0.834 | 2.905 | 2.207 | 2.008 | 0.8688 | 0.9890 | 0.8646 | 0.9268 | 0.9299 |
| lin (scene+ego ridge, fit.json, SPEEDREAD primary) | 2.786 | 0.812 | 2.827 | 2.154 | 1.944 | 0.8781 | 0.9895 | 0.8718 | 0.9303 | 0.9343 |
| mlp (scene+ego MLP, fit.json) | 2.899 | 1.203 | 2.902 | 2.227 | 2.046 | 0.8680 | 0.9771 | 0.8648 | 0.9255 | 0.9272 |
| MLP REFIT full = mlp_ego (control) | 2.566 | 0.799 | 2.548 | 1.838 | 1.806 | 0.8966 | 0.9899 | 0.8958 | 0.9492 | 0.9433 |
| MLP (a) p10f + frame_ok | 3.222 | 0.857 | 3.298 | 2.346 | 2.194 | 0.8370 | 0.9883 | 0.8255 | 0.9173 | 0.9163 |
| MLP (b) p10f + limit + limit_ok + frame_ok (no nav) | 2.744 | 0.812 | 2.754 | 2.023 | 1.834 | 0.8818 | 0.9895 | 0.8783 | 0.9385 | 0.9415 |
| MLP (c, extra) p10f + nav + frame_ok (no limit) | 2.901 | 0.833 | 2.952 | 2.017 | 2.117 | 0.8678 | 0.9890 | 0.8602 | 0.9389 | 0.9221 |
| RIDGE REFIT full = mlp_ego (control) | 2.891 | 0.834 | 2.905 | 2.207 | 2.008 | 0.8688 | 0.9890 | 0.8646 | 0.9268 | 0.9299 |
| RIDGE (a) p10f + frame_ok | 3.398 | 0.882 | 3.462 | 2.449 | 2.337 | 0.8186 | 0.9877 | 0.8077 | 0.9099 | 0.9050 |
| RIDGE (b) p10f + limit + limit_ok + frame_ok (no nav) | 2.905 | 0.836 | 2.928 | 2.244 | 2.012 | 0.8675 | 0.9889 | 0.8624 | 0.9244 | 0.9296 |
| RIDGE (c, extra) p10f + nav + frame_ok (no limit) | 3.347 | 0.878 | 3.387 | 2.390 | 2.317 | 0.8241 | 0.9878 | 0.8159 | 0.9142 | 0.9066 |
| persistence: cell := measured speed v (no fit) | 3.976 | 1.452 | 4.011 | 3.110 | 3.230 | 0.7517 | 0.9665 | 0.7419 | 0.8547 | 0.8185 |
| CONST (train mean) | 7.979 | 7.940 | 7.894 | 8.159 | 7.583 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |

MLP seed floor (same design, MLP init/order seeds 0, 1, 2; mean over cells, SD units):

| design | seed 0 | seed 1 | seed 2 | range of the three means | best epochs |
|---|---|---|---|---|---|
| REFIT full = mlp_ego (control) | 0.24751 | 0.24719 | 0.24706 | 0.00045 | [8, 10, 17] |
| (a) p10f + frame_ok | 0.30849 | 0.30822 | 0.30828 | 0.00027 | [3, 5, 8] |
| (b) p10f + limit + limit_ok + frame_ok (no nav) | 0.26313 | 0.26387 | 0.26446 | 0.00133 | [13, 10, 4] |
| (c, extra) p10f + nav + frame_ok (no limit) | 0.28032 | 0.27945 | 0.28019 | 0.00087 | [14, 19, 17] |

T_own (eval-141, m/s, from READ.json; a different population) beside mlp_ego (eval-141) and the ESTIMATED eval-141 RMSE of the refits (mlp_ego eval RMSE x fold-0 ratio):

seed 0: T_own [2.894, 1.919, 2.795, 2.257, 2.093] | mlp_ego [2.54, 0.752, 2.557, 1.861, 1.714] (below T_own in 5/5)
  - (b) p10f + limit + limit_ok + frame_ok (no nav): ESTIMATED [2.716, 0.765, 2.763, 2.048, 1.741] (below T_own in 5/5 cells: [True, True, True, True, True])
  - (c, extra) p10f + nav + frame_ok (no limit): ESTIMATED [2.872, 0.784, 2.962, 2.042, 2.009] (below T_own in 4/5 cells: [True, True, False, True, True])
  - (a) p10f + frame_ok: ESTIMATED [3.189, 0.806, 3.309, 2.375, 2.083] (below T_own in 2/5 cells: [False, True, False, False, True])
seed 1: T_own [2.747, 1.302, 2.692, 2.143, 1.802] | mlp_ego [2.54, 0.752, 2.557, 1.861, 1.714] (below T_own in 5/5)
  - (b) p10f + limit + limit_ok + frame_ok (no nav): ESTIMATED [2.716, 0.765, 2.763, 2.048, 1.741] (below T_own in 4/5 cells: [True, True, False, True, True])
  - (c, extra) p10f + nav + frame_ok (no limit): ESTIMATED [2.872, 0.784, 2.962, 2.042, 2.009] (below T_own in 2/5 cells: [False, True, False, True, False])
  - (a) p10f + frame_ok: ESTIMATED [3.189, 0.806, 3.309, 2.375, 2.083] (below T_own in 1/5 cells: [False, True, False, False, False])
