# RJ5 navtrain re-roll on the dev box: 198 local logs DONE (16,133 rows, 0 errors); identity control FAILS its bar as written, explained by bank provenance

2026-11-13, Master Mind. MEASURED. Sources (this folder): `rj5_merge_report.json`, `chain.log`. Rows and merge are banked privately on HF `Sayood/tanitad-refe-scorer1p6`, `evidence/2026-11-13-rj5-navtrain-local`.

## What ran
- `rj5_chain.py` (plan `../RJ5_NAVTRAIN_PLAN.md`) on the 198 nuPlan logs held on the dev box.
- RJ5 = the registered teacher with positive jerk limit 5.0; rank 0 only. CPU, single-thread rollers, 17 batches.
- Start 01:19 local at procs 1. It died at 01:00Z when TF-N4 A1 took the RAM (221 rows of batch 0 written). It was resumed 08:27Z, batch 0 at procs 1, then procs 2. DONE 11:30Z.
- **Rows:** 17 / 17 batches, 17,926 ok rows (16,133 RJ5 + 1,793 R0 control), **0 error rows**.
- **Merge** (`rj5_merge.py --min-coverage 0`):
  - 92,896 bank keys; 16,133 RJ5 (coverage 0.174); 76,763 fall back to the bank's own trajectory.
  - 0 duplicates, 0 conflicts. Output sha256 `3ca35564…`.
  - The other 877 logs need the public nuPlan download (PI decision pending).

## Identity control (R0 = registered teacher re-rolled, against the bank): FAIL as written
- **Bar (plan):** any non-zero is reported, and the bar is ≤ 1e-6 m.
- **Measured:** 1,766 of 1,793 control rows exactly 0.0; **27 non-zero, max 6.36 mm** (p99 0.20 mm). **The bar is not met.**
- **Diagnosis (MEASURED):** all 27 non-zero rows are bank rows with `teacher_device: None`, and all are in one log (`2021.09.14.18.43.41_veh-45_00965_01195`). All 1,766 rows with `teacher_device: cpu` reproduce **exactly (0.0)**.
  - So the dev-box roller is bit-exact against the CPU-rolled bank.
  - The mismatch is the provenance of the bank's older rows, the 3,412 `None` rows of P7 that the plan already flagged.
- **Exposure:** 189 of the 16,133 local RJ5 rows (1.2 %, in 9 logs) sit on `None` rows. Their RJ5-versus-bank movement includes a provenance difference of up to about 6 mm.
- **Next:** any card that uses these rows names the 189 `None` rows and either excludes them or reports them separately. The R0 identity bar is read only on `teacher_device: cpu` rows, stated before any RJ5 label number is seen.

## RJ5 movement (descriptive)
- 13,718 of 16,133 rows (85 %) move more than 1e-6 m from the registered-teacher bank.
- Displacement: p50 0.97 m, p90 4.02 m, p99 6.33 m, max 11.53 m.
- The biggest provenance difference is 6.4 mm, about 150 times smaller than the median RJ5 movement.

## Next (running)
- `label_local.sh` labels the 16,133 rows with the unchanged dual labeller (Tc20s, then P5-R lambda 3.0) once the TF-N5 A1 chain ends, which has RAM priority.
- Uses: a navtrain development set for the next DAC lever (REPAIR-D residuals) and a slice of the RJ5 student-label table. The full table needs the 877 remaining logs.
