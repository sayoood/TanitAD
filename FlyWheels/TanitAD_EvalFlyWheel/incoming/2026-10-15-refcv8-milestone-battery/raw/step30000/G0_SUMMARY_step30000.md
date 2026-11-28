## refcv8 G0 -- step30000  (EARLY READ at step 30000 of 48,852; never a bar verdict)

* **Gate verdict: FAIL** under amendment `A7` (registration: {"registered": true, "file_mtime": "2026-10-15T03:05:12", "g0_started": "2026-10-18T18:30:47", "text": "2286d5f42d0858dd98550de79403c3b398114954d06bf3904f076daf80890fe7  SPEC.md after AMENDMENT A7 (20); n_seeds 24; finished True.
* Verdicts by rule: {"as_registered": "FAIL", "A2": "FAIL", "A5": "FAIL", "A6": "FAIL", "A7": "FAIL"}.
* Mutations (M1 must be detected, else G0 is VOID): {"m1": {"detected": true, "n_terms_out": 8, "how": null}, "m2": {"detected": true, "n_terms_out": 65, "how": null}, "m4": {"detected": true, "n_terms_out": 2, "how": null}, "m5": {"detected": true, "n_terms_out": 56, "how": "DETECTED"}}; M1 detected = **True**.
* Checkpoint md5 `70f9330cd3ae7985a0174bfb24741155`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 30000; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 30000}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: 1136 {"SMOOTH": 192, "DETECTION": 100, "EXCLUDED": 11, "MATCHED": 29, "UNDEFINED": 94, "DERIVED": 2, "DETECTION_LOWSUPPORT": 238, "COUNT": 219, "STOCHASTIC": 10, "MAP10_COUNTS": 160, "MAP10_IOU": 80, "THRESHOLD_TARGET": 1}; not-OK by class: {"DETECTION_LOWSUPPORT": 238}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|
| `eval_loss` | 47.49049 | 46.98081375 | 0.08055305979196412 | STOCHASTIC | OK |
| `eval_traj` | 0.53001 | 0.5360504166666666 | 0.006242353568944172 | STOCHASTIC | OK |
| `eval_tacv6_goal_conf_bce` | 0.12473 | 0.12476 | 0.0 | THRESHOLD_TARGET | OK |
| `eval_r8` | 0.75084 | 0.7276941666666666 | 0.01932195775404955 | STOCHASTIC | OK |
| `eval_r8_total` | 0.75084 | 0.7276941666666666 | 0.01932195775404955 | STOCHASTIC | OK |
| `eval_r8_cons` | 0.02729 | 0.02731 | 0.0 | SMOOTH | OK |
| `eval_r8_alloc_l1` | 0.55867 | 0.5348462500000001 | 0.019224630897683677 | STOCHASTIC | OK |
| `eval_r8_sat` | 0.28204 | 0.2829095833333333 | 0.0028259195951001533 | STOCHASTIC | OK |
| `eval_r8_subscore` | 0.31396 | 0.31511458333333336 | 0.0011531920884022833 | STOCHASTIC | OK |
| `eval_r8_v9_cons` | 0.11248 | 0.11268 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_speed` | 0.11565 | 0.11585 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lat_c` | 0.12794 | 0.12805 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lon_c` | 0.19408 | 0.19442 | 0.0 | SMOOTH | OK |

First reasons: ["wrapper clause FAIL: ['merge defect: agent_yaw wrapper 1.638e-05 > bar 1.000e-05', 'merge defect: cascade wrapper 1.701e-01 > bar 9.767e-02', 'merge defect: goal_score_absmean wrapper 1.572e-03 > bar 1.195e-03', 'merge defect: traj wrapper 2.351e-01 > bar 2.108e-01']"]

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
