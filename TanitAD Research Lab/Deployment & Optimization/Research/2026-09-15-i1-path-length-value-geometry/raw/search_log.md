# Search / probe log — 2026-09-15 I-1 path-length value geometry

| # | probe | result |
|---|---|---|
| P1 | `tasklist /FI "IMAGENAME eq python.exe"` + `nvidia-smi` (dev-box contention; DP13-1's named blocker) | **no python.exe**; 4060 9 % util / 1,196 MiB. Control in the same breath: `nvidia-smi --query-compute-apps` listed 25 processes, so the query itself works. Blocker cleared |
| P2 | pose/speed source for the 139 bank clips: `bev_gt/*.bevgt.npz` keys | `ego_v_ms`, `ego_yaw_rate_rps`, `raw_frame`, `label_valid` on the raw v2ep grid (meta `time_grid`). No new extraction needed |
| P3 | bank clips with a bev_gt file | **134 / 139**. Second probe: `bev_gt_extra/` holds **0** of the 139 (183 files, different clips). The 5 misses are listed in `vgeo3_pathlen.json` → `excluded` |
| P4 | Library check (V-1) for a within-episode path-length value-geometry control | no banked primary. The design derives from DP13-1 and our own 09-13 package. No web search: this package is a measurement |
