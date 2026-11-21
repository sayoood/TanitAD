## refcv8 G0 -- step10500  (EARLY READ at step 10500 of 48,852; never a bar verdict)

* **Gate verdict: FAIL** under amendment `A7` (registration: {"registered": true, "file_mtime": "2026-10-15T03:05:12", "g0_started": "2026-10-17T03:23:41", "text": "2286d5f42d0858dd98550de79403c3b398114954d06bf3904f076daf80890fe7  SPEC.md after AMENDMENT A7 (20); n_seeds 24; finished True.
* Verdicts by rule: {"as_registered": "FAIL", "A2": "FAIL", "A5": "FAIL", "A6": "FAIL", "A7": "FAIL"}.
* Mutations (M1 must be detected, else G0 is VOID): {"m1": {"detected": true, "n_terms_out": 8, "how": null}, "m2": {"detected": true, "n_terms_out": 66, "how": null}, "m4": {"detected": true, "n_terms_out": 3, "how": null}, "m5": {"detected": true, "n_terms_out": 27, "how": "DETECTED"}}; M1 detected = **True**.
* Checkpoint md5 `a08a62319b1aa12ecf1b2aee35cd01f2`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 10500; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 10500}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: 1136 {"SMOOTH": 192, "DETECTION": 100, "EXCLUDED": 11, "MATCHED": 29, "UNDEFINED": 94, "DERIVED": 2, "DETECTION_LOWSUPPORT": 238, "COUNT": 219, "STOCHASTIC": 10, "MAP10_COUNTS": 160, "MAP10_IOU": 80, "THRESHOLD_TARGET": 1}; not-OK by class: {"DETECTION_LOWSUPPORT": 238}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|
| `eval_loss` | 51.56521 | 51.43092958333333 | 0.10401581181971543 | STOCHASTIC | OK |
| `eval_traj` | 0.61754 | 0.6178508333333333 | 0.009114097020152546 | STOCHASTIC | OK |
| `eval_tacv6_goal_conf_bce` | 0.12761 | 0.12778 | 0.0 | THRESHOLD_TARGET | OK |
| `eval_r8` | 0.77458 | 0.8083866666666667 | 0.029380450120366806 | STOCHASTIC | OK |
| `eval_r8_total` | 0.77458 | 0.8083866666666667 | 0.029380450120366806 | STOCHASTIC | OK |
| `eval_r8_cons` | 0.02835 | 0.028259999999999997 | 0.0 | SMOOTH | OK |
| `eval_r8_alloc_l1` | 0.57236 | 0.60565125 | 0.029209828201941828 | STOCHASTIC | OK |
| `eval_r8_sat` | 0.27791 | 0.2793316666666667 | 0.003925266722221528 | STOCHASTIC | OK |
| `eval_r8_subscore` | 0.33439 | 0.33513875 | 0.0009666675724633389 | STOCHASTIC | OK |
| `eval_r8_v9_cons` | 0.11633 | 0.11642999999999999 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_speed` | 0.11116 | 0.11108999999999998 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lat_c` | 0.13564 | 0.13569 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lon_c` | 0.20424 | 0.20448999999999998 | 0.0 | SMOOTH | OK |

First reasons: ["wrapper clause FAIL: ['merge defect: cascade wrapper 9.127e-02 > bar 7.138e-02']"]

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
