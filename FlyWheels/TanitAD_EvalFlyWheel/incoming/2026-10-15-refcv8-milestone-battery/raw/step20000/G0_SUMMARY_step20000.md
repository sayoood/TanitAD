## refcv8 G0 -- step20000  (EARLY READ at step 20000 of 48,852; never a bar verdict)

* **Gate verdict: FAIL** under amendment `A7` (registration: {"registered": true, "file_mtime": "2026-10-15T03:05:12", "g0_started": "2026-10-17T17:48:52", "text": "2286d5f42d0858dd98550de79403c3b398114954d06bf3904f076daf80890fe7  SPEC.md after AMENDMENT A7 (20); n_seeds 24; finished True.
* Verdicts by rule: {"as_registered": "FAIL", "A2": "FAIL", "A5": "FAIL", "A6": "FAIL", "A7": "FAIL"}.
* Mutations (M1 must be detected, else G0 is VOID): {"m1": {"detected": true, "n_terms_out": 6, "how": null}, "m2": {"detected": true, "n_terms_out": 59, "how": null}, "m4": {"detected": true, "n_terms_out": 2, "how": null}, "m5": {"detected": true, "n_terms_out": 50, "how": "DETECTED"}}; M1 detected = **True**.
* Checkpoint md5 `eee65ba210933fc0462084e8b75517b6`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 20000; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 20000}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: 1136 {"SMOOTH": 192, "DETECTION": 100, "EXCLUDED": 11, "MATCHED": 29, "UNDEFINED": 94, "DERIVED": 2, "DETECTION_LOWSUPPORT": 238, "COUNT": 219, "STOCHASTIC": 10, "MAP10_COUNTS": 160, "MAP10_IOU": 80, "THRESHOLD_TARGET": 1}; not-OK by class: {"DETECTION_LOWSUPPORT": 238}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|
| `eval_loss` | 48.97479 | 49.031738749999995 | 0.08313893261138752 | STOCHASTIC | OK |
| `eval_traj` | 0.56549 | 0.5595004166666667 | 0.007007959126820622 | STOCHASTIC | OK |
| `eval_tacv6_goal_conf_bce` | 0.12637 | 0.12449 | 0.0 | THRESHOLD_TARGET | OK |
| `eval_r8` | 0.74949 | 0.7455562499999999 | 0.02568454582081877 | STOCHASTIC | OK |
| `eval_r8_total` | 0.74949 | 0.7455562499999999 | 0.02568454582081877 | STOCHASTIC | OK |
| `eval_r8_cons` | 0.02404 | 0.024039999999999995 | 0.0 | SMOOTH | OK |
| `eval_r8_alloc_l1` | 0.55923 | 0.5544320833333333 | 0.025730489101143388 | STOCHASTIC | OK |
| `eval_r8_sat` | 0.30655 | 0.30716499999999997 | 0.0037314049108473414 | STOCHASTIC | OK |
| `eval_r8_subscore` | 0.30555 | 0.30716166666666667 | 0.0009052535876598442 | STOCHASTIC | OK |
| `eval_r8_v9_cons` | 0.11256 | 0.11251 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_speed` | 0.11225 | 0.11218 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lat_c` | 0.12817 | 0.12813 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lon_c` | 0.19836 | 0.19844000000000003 | 0.0 | SMOOTH | OK |

First reasons: ["wrapper clause FAIL: ['merge defect: cascade wrapper 1.019e-01 > bar 5.377e-02', 'merge defect: lat_tac wrapper 1.065e-05 > bar 1.000e-05', 'merge defect: loss wrapper 1.756e-02 > bar 1.442e-03', 'merge defect: traj wrapper 1.404e-01 > bar 1.397e-02']"]

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
