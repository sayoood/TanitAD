## refcv8 G0 -- step15000  (EARLY READ at step 15000 of 48,852; never a bar verdict)

* **Gate verdict: FAIL** under amendment `A7` (registration: {"registered": true, "file_mtime": "2026-10-15T03:05:12", "g0_started": "2026-10-17T04:21:54", "text": "2286d5f42d0858dd98550de79403c3b398114954d06bf3904f076daf80890fe7  SPEC.md after AMENDMENT A7 (20); n_seeds 24; finished True.
* Verdicts by rule: {"as_registered": "FAIL", "A2": "FAIL", "A5": "FAIL", "A6": "FAIL", "A7": "FAIL"}.
* Mutations (M1 must be detected, else G0 is VOID): {"m1": {"detected": true, "n_terms_out": 17, "how": null}, "m2": {"detected": true, "n_terms_out": 62, "how": null}, "m4": {"detected": true, "n_terms_out": 2, "how": null}, "m5": {"detected": true, "n_terms_out": 47, "how": "DETECTED"}}; M1 detected = **True**.
* Checkpoint md5 `e447dbb5dd213d9c1d93bc85e2ff9898`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 15000; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 15000}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: 1136 {"SMOOTH": 192, "DETECTION": 100, "EXCLUDED": 11, "MATCHED": 29, "UNDEFINED": 94, "DERIVED": 2, "DETECTION_LOWSUPPORT": 238, "COUNT": 219, "STOCHASTIC": 10, "MAP10_COUNTS": 160, "MAP10_IOU": 80, "THRESHOLD_TARGET": 1}; not-OK by class: {"DETECTION_LOWSUPPORT": 238}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|
| `eval_loss` | 50.67281 | 50.697727500000006 | 0.0767002243775418 | STOCHASTIC | OK |
| `eval_traj` | 0.57254 | 0.5619779166666666 | 0.009233228263541787 | STOCHASTIC | OK |
| `eval_tacv6_goal_conf_bce` | 0.11511 | 0.11529 | 0.0 | THRESHOLD_TARGET | OK |
| `eval_r8` | 0.77158 | 0.8062654166666667 | 0.02092994712786829 | STOCHASTIC | OK |
| `eval_r8_total` | 0.77158 | 0.8062654166666667 | 0.02092994712786829 | STOCHASTIC | OK |
| `eval_r8_cons` | 0.03135 | 0.03136 | 0.0 | SMOOTH | OK |
| `eval_r8_alloc_l1` | 0.57813 | 0.61355375 | 0.0209496408213477 | STOCHASTIC | OK |
| `eval_r8_sat` | 0.31202 | 0.30850958333333334 | 0.0027224149751831485 | STOCHASTIC | OK |
| `eval_r8_subscore` | 0.30969 | 0.30890833333333334 | 0.0009738880690379676 | STOCHASTIC | OK |
| `eval_r8_v9_cons` | 0.11675 | 0.11678 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_speed` | 0.11861 | 0.11825000000000001 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lat_c` | 0.13697 | 0.13702 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lon_c` | 0.19973 | 0.19992 | 0.0 | SMOOTH | OK |

First reasons: ["wrapper clause FAIL: ['merge defect: cascade wrapper 1.854e-01 > bar 1.522e-02', 'merge defect: law wrapper 1.167e-03 > bar 2.013e-04', 'merge defect: traj wrapper 1.732e-01 > bar 1.314e-01']"]

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
