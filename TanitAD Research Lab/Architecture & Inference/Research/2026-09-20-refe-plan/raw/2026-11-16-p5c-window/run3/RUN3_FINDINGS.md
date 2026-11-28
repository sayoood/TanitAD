# P5-C label run 3 (A3.2): control PASS, main gates FAULT (TB-5, TB-0r). MM, 2026-11-16

Evidence class: **MEASURED**. Artifacts are in this directory.
- **Run:** `/workspace/data/p5c_label_a32`, started 09:44Z, ended 11:49Z with `ZZP5CL_FAIL gates`.
- **Code:** runner `8cfe7a72`, gates `1711a72f` (A3.2); join `4670f88a`; A3.1 `cda1567b`.

## Result
- **Control pass (A3.2):** TB-0 PASS (R-A on rank 0), TB-1 PASS.
- **Main gates:** TB-2/3/4/7/9 PASS; TB-6 PASS (rank 0 and rank 1); RK0 no-stop. **TB-5 FAIL and TB-0r FAIL** ⇒ class `fault`. Under A3 §4 that means no C1 launch; fix and rebuild.

## TB-0r: the cross-platform floor again, now on rank-0 rows
- **Scope:** pod RJ5-pass labels against the dev-box RJ5 labels on the same RJ5 inputs; 16,133 local rows, all rank 0.
- **R-B (the P5-R labels C1 trains on): passes every criterion.** sim8 identical 16,133/16,133; row-max median 4.6e-6, p99 1.8e-3; TB-6 ratios within 8.4e-7.
- **R-A (the Tc20s warm start, tol 1e-9): FAIL**, max **2.548e-9**.

| rows over | 0 | 1e-12 | 1e-10 | 5e-10 | **1e-9** | 2e-9 |
|---|---|---|---|---|---|---|
| count | 14,758 | 598 | 210 | 95 | **45 (0.28 %)** | 6 |

- Median 1.9e-15; p99 1.2e-10; p99.9 1.3e-9.
- This is the same Linux-vs-Windows numerics floor as TB-0 (`tb0/TB0_RA_FINDINGS.md`). **A3.2's rank-0 scope does not cover it**, because these are rank-0 rows.

## TB-5: one genuine P5-R failure on a near-standstill RJ5 trajectory
- **Count:** 150,602 of 150,603 rows pass. The one failure: `yaw_branch_ge_pi` 1, S2 false 1.
- **The row:** rank 0, token `d06ff3cb…`, RJ5 traj sha `926ffc66…`. The trajectory creeps: 0.966 m in total, 10 of 20 steps under 5 cm, traj yaw 0.001 → 0.210.
- **The defect:** the P5-R label's tail heading flips to 3.788 rad, so |label yaw − traj yaw| is 3.578 at step 19. The Tc20s warm start of the same row is sane (max 0.550).
  - p5r_accepted 9, calls 11, S1 true, S5 true, E_maxxy 0.096 m.
- ⇒ **S2 caught a real P5-R defect** on 1 of 92,896 rank-0 rows. It is not a gate artifact.

## Open: PI ruling needed (post-data; any choice is disclosed as such)
- **R-A (TB-0 and TB-0r):** the measured cross-platform Tc20s floor reaches 2.55e-9. The labels C1 trains on pass R-B on all 18,533 compared rows.
- **TB-5:** decide what replaces the one flipped-heading label: its Tc20s(RJ5) label, the raw bank label, or exclusion of the row.
