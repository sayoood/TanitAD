## refcv8 G0 -- step48852  (EARLY READ at step 48852 of 48,852; never a bar verdict)

* **Gate verdict: FAIL** under amendment `A7` (registration: {"registered": true, "file_mtime": "2026-10-15T03:05:12", "g0_started": "2026-10-21T04:54:03", "text": "2286d5f42d0858dd98550de79403c3b398114954d06bf3904f076daf80890fe7  SPEC.md after AMENDMENT A7 (20); n_seeds 24; finished True.
* Verdicts by rule: {"as_registered": "FAIL", "A2": "FAIL", "A5": "FAIL", "A6": "FAIL", "A7": "FAIL"}.
* Mutations (M1 must be detected, else G0 is VOID): {"m1": {"detected": true, "n_terms_out": 10, "how": null}, "m2": {"detected": true, "n_terms_out": 67, "how": null}, "m4": {"detected": true, "n_terms_out": 2, "how": null}, "m5": {"detected": true, "n_terms_out": 54, "how": "DETECTED"}}; M1 detected = **True**.
* Checkpoint md5 `ba49eab9411e21b1376ae52dde802036`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 48852; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 48852}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: 1136 {"SMOOTH": 192, "DETECTION": 100, "EXCLUDED": 11, "MATCHED": 29, "UNDEFINED": 94, "DERIVED": 2, "DETECTION_LOWSUPPORT": 238, "COUNT": 219, "STOCHASTIC": 10, "MAP10_COUNTS": 160, "MAP10_IOU": 80, "THRESHOLD_TARGET": 1}; not-OK by class: {"DETECTION_LOWSUPPORT": 238}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|
| `eval_loss` | 46.10201 | 46.025648749999995 | 0.10397164342983138 | STOCHASTIC | OK |
| `eval_traj` | 0.5623 | 0.5550308333333334 | 0.006936818539451301 | STOCHASTIC | OK |
| `eval_tacv6_goal_conf_bce` | 0.11326 | 0.11269999999999998 | 0.0 | THRESHOLD_TARGET | OK |
| `eval_r8` | 0.75554 | 0.7355825 | 0.02589255697148777 | STOCHASTIC | OK |
| `eval_r8_total` | 0.75554 | 0.7355825 | 0.02589255697148777 | STOCHASTIC | OK |
| `eval_r8_cons` | 0.02426 | 0.02433 | 0.0 | SMOOTH | OK |
| `eval_r8_alloc_l1` | 0.57217 | 0.5515375 | 0.025955068575956582 | STOCHASTIC | OK |
| `eval_r8_sat` | 0.24502 | 0.24929458333333332 | 0.0029480781937105864 | STOCHASTIC | OK |
| `eval_r8_subscore` | 0.30428 | 0.3047604166666667 | 0.0008280646853890128 | STOCHASTIC | OK |
| `eval_r8_v9_cons` | 0.11032 | 0.11041 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_speed` | 0.10898 | 0.10914 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lat_c` | 0.11927 | 0.11939 | 0.0 | SMOOTH | OK |
| `eval_r8_v9_mae_lon_c` | 0.19524 | 0.19505000000000003 | 0.0 | SMOOTH | OK |

First reasons: ["wrapper clause FAIL: ['merge defect: cascade wrapper 1.064e-01 > bar 2.913e-02', 'merge defect: law wrapper 3.559e-03 > bar 2.039e-03', 'merge defect: loss wrapper 8.019e-03 > bar 8.252e-04', 'merge defect: r8_sat wrapper 4.558e-02 > bar 3.662e-02', 'merge defect: traj wrapper 1.448e-01 > bar 2.777e-02']"]

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
