# Pod probes for chain C feasibility (2026-11-15 00:16-00:40Z; filesystem only, `nice -n 19 ionice -c3`, no Python, nothing launched)
Access through `par18/pd.sh` with the IP mask filter. No address, port, id or token was printed or written. Log names were `tr . _`-transformed before the mask so that it could not mangle them.

| probe | result (MEASURED) |
|---|---|
| `nproc` | 128 |
| `/sys/fs/cgroup/cpu.max` | `1530000 100000` = **15.3 CPUs** (cpu.stat: nr_throttled 650,893 of 7,375,544 periods) |
| `memory.max` / anon | 142,999,998,464 B / 15,937,957,888 B (P7 training) |
| CPU | AMD EPYC 7452 32-Core |
| `/workspace/venv-teacher` | python3.11; torch-2.7.1+cu128, numpy-1.26.4, nuplan_devkit-1.2.2 (editable), driverl-0.1.0 (editable), navsim-1.1.0 (editable), shapely-2.1.2, scipy-1.17.1, pandas-2.3.3, sqlalchemy-2.0.54, geopandas-1.1.4, pyproj-3.7.2, hydra 1.3.2, omegaconf 2.3.0; no threadpoolctl. The dev box's `driverl-eval` venv has the same versions of torch, numpy, nuplan_devkit, shapely, scipy, pandas, sqlalchemy, geopandas and pyproj. |
| teacher checkpoint / config | `checkpoint_2400.pt` sha256 `d7b1fdd06e1aadf4dc429b014807534f3b2028a4137f47f8fdb2a0620e3cb27b` and `driverl_teacher.yaml` `0521575f25e0c24327646a07f07f815c80dafead2dcfc98b486480eae1d15e2a`: both equal `pod_teacher_env.sh`'s pins. DriveZero HEAD `24959547edac`. |
| `teacher_env.sh` | DZ_ROOT, NUPLAN_MAPS_ROOT `/workspace/data/nuplan-maps/nuplan-maps-v1.0`, DRIVERL_EVAL_* (DEVICE=cuda: override to cpu for rolls), `unset DRIVERL_EVAL_ROUTE_LANE_RANK` |
| DBs | `/workspace/data/navtrain_dbs`: 1,192 flat `<log>.db` (real files, e.g. 145,805,312 B); **192 / 192 chain-C logs present** (local diff of the listing against `logs_fetch_C.txt`, sha256 `07d3ea80…`) |
| maps | `nuplan-maps-v1.0.json` + 4 locations |
| `refe-plan-p7` code (sha256) | `refe/build_teacher_rollouts.py` `7da350d9…` = tip blob `049b0290` (**no TEACHER_OVERRIDES**: 0 grep hits); `refe/build_targets.py` `70f443f1…` = `b231d1e1`; `refe/navtrain_scenarios.py` `c1c073b2…` = `848fe684`; `splits/navtrain.yaml` `5f3c406a…` (= tip `splits/navtrain.yaml`; the roll path does not read it) |
| `refe-plan-vdac` (the label runner's SRC_TREE) | the same three sha256s as `refe-plan-p7` |
| processes | the P7 trainer python at 6.9 GB RSS, 2 pt_data_workers at ~3.2 GB, ≥ 4 python workers at ~0.83 GB (label-worker class) |
| P7 progress | `refe_p7/.../train.log` tail: step 7,013, 364,802 s elapsed, lr 2.44e-05 |
