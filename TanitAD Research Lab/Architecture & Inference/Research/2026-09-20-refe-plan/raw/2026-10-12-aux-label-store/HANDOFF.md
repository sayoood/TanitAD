# HANDOFF: aux drivable label store, full build running (written 2026-10-12 ~21:20 local, dev box)

State: S0 code, tests and the pre-build gate are DONE and staged (see README.md). The FULL BUILD is running detached on the dev box. Nothing else is pending
except verifying the build when it ends. No pod, no GPU, no ssh was used or is needed.

## The build

* Launched ~21:16 local (pid 39184 at launch), 4 workers (below-normal priority), resumable by key.
* Log (stdout): `D:/Projects/TanitAD-artifacts/aux_drivable/build.log`   stderr: `.../build.err` (should stay empty)
* Store: `D:/Projects/TanitAD-artifacts/aux_drivable/label_store/` (`shards/<log>.aux1`, `manifest.json`, `run_meta_all.json`, `missing_tokens.txt`, `failures_all.jsonl`)
* Progress at 96 s: 25/1192 logs, 15,476 keys (largest logs first, so the log-count ETA of 4,497 s is pessimistic; by keys ~161 keys/s => ~11 min total).
* **DONE marker = the LAST line of build.log:** `ZZAUXSTORE_DONE <stored_keys_in_store> <failed> new=<n> skipped=<n>`
  Expected: `ZZAUXSTORE_DONE 103139 0 new=103139 skipped=0`. A trailing ` ABORTED` means free RAM fell under 2.5 GB (the neighbour's job): just rerun, it resumes.
  Check: `tail -2 D:/Projects/TanitAD-artifacts/aux_drivable/build.log`
* Exact command (PowerShell launched it; cwd = the refe-plan package dir `PKG`; driverl-eval venv; env `PYTHONIOENCODING=utf-8`, `TMP`/`TEMP` = `D:/Projects/TanitAD-artifacts/aux_drivable/tmp`):
  ```
  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -u refe/aux_label_store.py build \
    --bank D:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl \
    --db-root D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits --maps-root D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0 \
    --yaml splits/navtrain.yaml \
    --navtest D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json \
    --logmap D:/Projects/TanitAD-artifacts/aux_drivable/log_map.json --out D:/Projects/TanitAD-artifacts/aux_drivable/label_store --workers 4
  ```
  Before any multi-worker rerun check free RAM >= 6 GB (the builder refuses below that; the other workstream holds ~11 GB).

## Verify after the marker (in this order; each is cheap)

1. **Count**: `manifest.json` `counts.stored_keys` == the marker's first number == 103,139 (103,037 bank + 102 DB-only). `counts.logs` == 1192.
   `coverage_vs_navtrain_yaml`: `missing` should be 149 (tokens with neither a bank row nor a local DB; listed in `missing_tokens.txt`), `stored_not_in_yaml` 0.
   `failed` (marker field 2) and `failures_all.jsonl` should be 0 / empty. Anything else: read the errors before using the store.
2. **Leak**: `manifest.json` `leakage` == `{token_intersection: 0, log_intersection: 0}`.
3. **Origin cross-check** (DB `ego_pose` vs bank `origin_world`): `runs[0].origin_xcheck` should read max_m 0.0, max_rad <= 1e-15, n ~ 3.4e4.
4. **QC on all keys**: `qc_rank0_target_points_inside_label.mean` and `share_zero` (share of keys with NO rank-0 target point inside the label; a wrong map would show as 0).
   The smoke build of 12 Las-Vegas logs read mean 1.0, share_zero 0.0.
5. **sha256 of every file + the fidelity re-check on random stored keys** (about 5 min, driverl-eval venv):
   ```
   cd PKG/raw/2026-10-12-aux-label-store
   PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe s0_gate.py --mode post --n 64 --out s0_gate_post.json
   ```
   It re-hashes all 1,192 shards against the manifest, rejects extra files, and checks 64 random stored keys in logs WITH a local DB (stored label vs a freshly built path A, cell for cell,
   plus the 13-shift fidelity vs `navsim_dac`) and 64 in logs whose map was INFERRED (stored vs recomputed, fidelity vs the inferred map's exact polygons). Last line `ZZGATE_PASS post`.
   It also prints the QC split by map source (db vs inferred): the two should look alike; a collapse in `inferred` means a wrong map.
6. Copy `manifest.json`'s `store_sha256` into `eval/PREREG_AUX_DRIVABLE_DRAFT.md` line 144 and stage `s0_gate_post.json`. ESCALATE: that line says "expect 103,288"; the store holds 103,139 (100.000 % of the 103,037-row train bank, 99.856 % of the yaml). The 149 missing tokens need the nuPlan DBs of their logs (public nuPlan v1.1 trainval; a download needs the PI's permission) or the pod bank rows.

## Facts the next stage needs

* Key = `(log_name, token)`; read with `aux_label_store.read_label(store, log, token) -> int8 [160, 128]` (ix along x forward, iy along y left; +inside, -outside; unit 0.1 m; clip +-12.7 m).
  mask = `sdf > 0`. The 2 m head targets are 4 x 4 block statistics, built in the loader (SPEC 3.1). One shard per log: load a shard once per log, not per token.
* Log -> map table: `raw/2026-10-12-aux-label-store/log_map.json` (463 from the DB, 729 inferred, 0 ambiguous; 462 / 462 inferred == DB).
* Trainer wiring (SPEC section 4: `train.py`, `model.py`, `IDENTITY_ARGS`, G-DVB) is NOT done and was not in scope; those files are owned by other streams.
* If someone later supplies the missing 149 tokens' source (the nuPlan DBs for their logs, or bank rows), rerun the same command: it resumes by key and only writes the new keys.
* `run_meta_all.json` records `code_sha256` of the CRLF working-copy bytes of `refe/aux_label_store.py` at launch; git will store it LF-normalised, so the two hashes differ by design (same code).
