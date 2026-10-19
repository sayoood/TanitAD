## refcv8 G0 -- step40000  (EARLY READ at step 40000 of 48,852; never a bar verdict)

* **Gate verdict: None** under amendment `None` (registration: null); n_seeds 0; finished False (PARTIAL artifact only).
* Verdicts by rule: {"as_registered": null, "A2": null, "A5": null, "A6": null, "A7": null}.
* Mutations (M1 must be detected, else G0 is VOID): {}; M1 detected = **None**.
* Checkpoint md5 `156d2632b166046cd7cf4b86fc962425`; config md5 `4ce2c77230dc4e802e78b39ce05af916`; step 40000; strict load: {"strict": true, "missing": 0, "unexpected": 0, "n_keys": 1165, "step": 40000}; param_breakdown equal: True.
* Eval set: 139 episodes, 23772 windows; in-run subset perm sha256 `99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975`.
* Pose-sync patch (r8_repoint): {"sidecar": "E:/refcv8_eval_kit/data/refcv8_pose_sync_sidecar.jsonl", "sidecar_rows": 4711, "sidecar_meta": {"builder": "scripts/build_pose_sync_sidecar.py", "clock_sidecar": "refcv8_clip_clock_sidecar.jsonl", "delta_ms": {"max": 35.426, "mean": 16.500600099077147, "min": 0.0, "p95": 31.388}, "inputs_md5": {"eval139_v2manifest.pt": "3443323229279a18253691ff36ebffd3", "train4572_v2manifest.pt": "77.
* Terms judged: None null; not-OK by class: {}.

| term | in-run | seeds mean | sd | class | verdict |
|---|---|---|---|---|---|

First reasons: []

Evidence class: MEASURED (E:/refcv8_eval_kit/battery/<tag>/g0.json; tier = REPRODUCTION of the run's own in-run eval row, T1 self-action OPEN LOOP on logged eval139 frames). Estimator: 24-seed inference-draw prediction interval (STOCHASTIC class) / registered tolerances; answers 'does the loader reproduce the run', not 'is the model good'.
