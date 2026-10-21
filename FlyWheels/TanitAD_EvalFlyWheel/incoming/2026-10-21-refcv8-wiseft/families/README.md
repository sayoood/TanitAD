# The four metric families on refcv8's NavSim reads (2026-10-21 15:25 Berlin)

**Why this exists.** Every NavSim read of refcv8 so far (the registered navtest read of EMA-48,852 and the R8-WISEFT stage-1
sweep) reported **PDMS only**. The binding rule (Sayed 2026-08-02) says an eval that reports one score is incomplete. This bundle
adds the families **beside** those reads. ⛔ Post-hoc, never a bar: no registered verdict is changed or re-decided by anything here.

**Tier.** NavSim v1 open loop: one model query per scene, plan scored against the logged human future at dt 0.5 s (8 poses, 4 s).
The families' `T1` stamp is the harness's word for "the model's own plan", not a closed loop. One training seed throughout.

## Instruments (nothing re-derived)

* `code/paired_families_navsim.py` -- **paired** B − A per component on identical tokens. Components are
  `taniteval/tools/refav1_arm.py::_components` (four_families' own `_seq_geometry` + `maneuver_kinematics` + the programme's
  `factor_from_kinematics` trajectory labeller). The interval is `refav1_arm._paired_families` →
  `taniteval.ci.paired_episode_cluster_bootstrap` (B 2,000), with the **OpenScene log** as the cluster unit (the NavSim adapter's
  `scenes_to_win(..., log_names=...)`). Added here: **curvature MAE** and a **pair-masked yaw-rate MAE**, both on four_families'
  own geometry and its own mask (`both_pair`); a window with no valid step pair (a stopped scene) is NaN and dropped for both
  arms (767 of 12,146 on navtest; the count is in each JSON).
* Controls, all three pass on all 17 reads: **P1** A − A is exactly 0 on every component; **P2** the human future fed as the
  prediction scores exactly 0 on every error and exactly 1 on both agreements (literal); **P3** both arms carry the same GT
  bit-exact.
* `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code/families7.py` (unchanged, on the tip) --
  the full per-arm family block (`navtest/families_navtest_<arm>.json`) for the six banked EMA-48,852 navtest seams.
* `code/table_families.py` -- builds `FAMILIES_TABLE.md`/`.json` from the paired files.

⛔ **The unmasked yaw-rate component is inadmissible on NavSim.** It averages heading differences over crawling steps where the
path tangent is noise: its mean is 0.48 rad/s against 0.083 masked on the same navtest arm, and it reads **separated at the
inference-seed floor** (−0.0069* navtest, +0.0192* navtrain), where the masked form reads −0.0000 [−0.0007, +0.0007]. It also
**inverts a reading**: unmasked says EMA-15,000 is worse than 48,852 on yaw rate (+0.18*), masked says better (−0.008*). It is
printed only so the defect stays visible; read the pair-masked column.

## navtest (12,146 tokens, 136 logs) -- `navtest/`

| B − A | ADE m | speed MAE | cross m | heading ° | yaw-rate (masked) | curvature 1/m | TAC lat | TAC lon |
|---|---|---|---|---|---|---|---|---|
| EMA-48,852 s1 − s0 (inference-seed floor) | −0.0047 | −0.0044 | +0.0020 | +0.0145 | −0.0000 | +0.0000 | −0.0007 | +0.0039 |
| EMA-48,852 − PRIOR_ha0p | **+0.8471*** | **+0.5925*** | **+0.2179*** | **+1.1461*** | **+0.0126*** | **+0.0103*** | **−0.0957*** | **−0.1150*** |

refcv8 as trained is **worse than its own causal prior on every family on navtest**, each separated and each far above the
inference-seed floor -- the family form of N1/N4. Distance-keeping and STRATEGIC are UNAVAILABLE on this bridge (no lead track;
NavSim scores no strategic decision), n = 12,146, reasons in each JSON.

## navtrain sample (2,089 tokens, 30 logs) -- `navtrain/` (the R8-WISEFT stage-1 sweep + R8-CKPT-TRAJ)

Full table: `navtrain/FAMILIES_TABLE.md`. The readings, with `*` = separated:

* **The WiSE gain is a four-family gain, not a PDMS artefact.** theta_0.5 − EMA-48,852 at seeds 0 / 1: ADE −0.712* / −0.720*,
  speed MAE −0.407* / −0.416*, cross-track −0.233* / −0.221*, heading −2.14* / −2.07°, masked yaw rate −0.029* / −0.027*,
  curvature −0.011* / −0.010*, tactical lateral agreement +15.1* / +15.4 pp, longitudinal +7.1* / +7.8 pp. Every lambda improves
  every family; lambda 0.25 has the best geometry, 0.5 the best longitudinal tactical agreement (PDMS chose 0.5).
* **theta_0.5 against PRIOR splits by family.** LATERAL better: heading −0.50°*, cross-track −0.043*, masked yaw rate −0.005*,
  curvature +0.0009 (not separated). LONGITUDINAL worse: speed MAE +0.096*, along-track +0.132*, tactical longitudinal agreement
  −7.1 pp*; ADE +0.090*. ⇒ **the remaining deficit of the best blend is longitudinal.**
* **Where training lost it (R8-CKPT-TRAJ):** EMA-5,000 is better than 48,852 on every family (ADE −0.770*, TAC lat +16.4 pp*);
  by 15,000 most of the geometry gain is gone (ADE −0.283*) and tactical lateral agreement is no longer separated; 30,000 and
  40,000 are indistinguishable from 48,852 on ADE. Same timing as the PDMS trajectory (74.61 → 71.12 → 63.73).

## Files

`code/` the three tools; `navtest/` six families7 blocks + two paired reads + table; `navtrain/` fifteen paired reads + table.
Seams: navtest at Thor `/home/nvidia/r8ns/navtest_read/full/bridge_navtest/seam_*.npz`, navtrain at
`/home/nvidia/r8ns/navtrain_sel/{ema48852,ema40000,wiseft/*}/bridge/seam_*.npz` (md5 verified on pull; each JSON names its seam).
`code/fam_wise050_waiter.sh` runs the same reads on theta_0.5's navtest seams when the stage-2 roll ends (beside the registered
PDMS rule).
