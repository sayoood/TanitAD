## refcv8 G0 -- step5000  (EARLY READ at step 5000 of 48,852; never a bar verdict)

* **Gate verdict: FAIL** under amendment `A7` (registration: {"registered": true, "file_mtime": "2026-10-15T03:05:12", "g0_started": "2026-10-16T00:53:23", "text": "2286d5f42d0858dd98550de79403c3b398114954d06bf3904f076daf80890fe7  SPEC.md after AMENDMENT A7 (20); n_seeds 24; finished True.
* Verdicts by rule: {"as_registered": "FAIL", "A2": "FAIL", "A5": "FAIL", "A6": "FAIL", "A7": "FAIL"}.
* Mutations (M1 must be detected, else G0 is VOID): {"m1": {"detected": true, "n_terms_out": 12, "how": null}, "m2": {"detected": true, "n_terms_out": 58, "how": null}, "m4": {"detected": true, "n_terms_out": 1, "how": null}, "m5": {"detected": true, "n_terms_out": 37, "how": "DETECTED"}}; M1 detected = **True**.
* Checkpoint md5 `a32e3b70f46aabbd3ab55f285d03f82d`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 5000; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 5000}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: 1136 {"SMOOTH": 192, "DETECTION": 100, "EXCLUDED": 11, "MATCHED": 29, "UNDEFINED": 94, "DERIVED": 2, "DETECTION_LOWSUPPORT": 238, "COUNT": 219, "STOCHASTIC": 10, "MAP10_COUNTS": 160, "MAP10_IOU": 80, "THRESHOLD_TARGET": 1}; not-OK by class: {"DETECTION_LOWSUPPORT": 238, "DETECTION": 1}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|
| `eval_loss` | 53.86168 | 53.75169833333334 | 0.08148930074698155 | STOCHASTIC | OK |
| `eval_traj` | 0.58254 | 0.5874791666666667 | 0.0073473354629958855 | STOCHASTIC | OK |
| `eval_tacv6_goal_conf_bce` | 0.1348 | 0.13562 | 0.0 | THRESHOLD_TARGET | OK |
| `eval_r8` | 0.92021 | 0.8985866666666666 | 0.032980759300810084 | STOCHASTIC | OK |
| `eval_r8_total` | 0.92021 | 0.8985866666666666 | 0.032980759300810084 | STOCHASTIC | OK |
| `eval_r8_cons` | 0.03883 | 0.03868 | 0.0 | SMOOTH | OK |
| `eval_r8_alloc_l1` | 0.71353 | 0.6928066666666667 | 0.032990252534935836 | STOCHASTIC | OK |
| `eval_r8_sat` | 0.31368 | 0.3135029166666667 | 0.0034402078794199637 | STOCHASTIC | OK |
| `eval_r8_subscore` | 0.33296 | 0.3312720833333333 | 0.0007692457014994436 | STOCHASTIC | OK |
| `eval_r8_v9_cons` | 0.13778 | 0.13722 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_speed` | 0.14291 | 0.1419 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lat_c` | 0.13669 | 0.13676 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lon_c` | 0.2293 | 0.22868999999999998 | 0.0 | SMOOTH | OK |

First reasons: ["eval_box3d_det_ap2_automobile_0_20 [DETECTION] in-run 0.33835 vs seeds mean 0.31453 (sd 0)", "wrapper clause FAIL: ['merge defect: cascade wrapper 1.235e-01 > bar 8.638e-02', 'merge defect: r8_subscore wrapper 1.642e-02 > bar 1.379e-02']"]

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
