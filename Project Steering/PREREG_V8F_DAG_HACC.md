# PREREG_V8F_DAG_HACC: does HACC's train-fitted post-map, re-fitted on DAG, remove DAG's longitudinal rows against P10 at T1-kin?

Written 2026-10-27 by the Master Mind. It was written after `RESULT_V8F_DAG_REREAD` (Read A: MARGIN, NOT DOMINANT) and before any DAG-HACC
fit, bake, rollout or row. Claim row `H-V8F-DAG-HACC-1`. Zero training. Dev-box CPU only (`CUDA_VISIBLE_DEVICES=-1`).

**Why.** Read A's failing set on plain DAG (`RESULT_V8F_DAG_REREAD.md` §1):
* accel MAE: +0.807 / +1.444;
* DK-VIOL (s0): +0.0353;
* curvature: +0.0612 / +0.1134;
* L2: +0.0288 / +0.0604;
* M1: +0.0470 / +0.0614;
* yaw-rate (s1): +8.66.

Read A's committed next sends the LONGITUDINAL rows to *"the failing family's lever, under its own PREREG"*. The programme's measured lever
on accel against P10 is HACC:
* PLAIN M2b is worse than P10 in accel at T1-kin; with HACC applied it reads +0.017 n.s. / −0.202 better (`RESULT_V8F_T1KIN_CAUSALBUCKET.md`);
* HACC's T0 fit also cut curvature (HACC − H −0.0131 / −0.0164, `RESULT_V8F_HACC.md`).
DAG has not had it.

## 1. The arm

* **DAG-HACC_s** = DAG_s (`v8f-dagger/M2b_s{s}_dag/ckpt_last.pt`) plus HACC's post-map, with w ∈ [0, 1]^(30 × 2).
* **The fit.** w and λ are fitted by `PREREG_V8F_HACC.md` §§ 1–2 **verbatim**, with only the plans changed (DAG's in place of M2b's):
  * the same fit set W_fit (HBLEND's 141 training clips / 9,582 windows), the same objective, grid, λ rule, bounds and solver;
  * the same declared interpretations I1–I3;
  * DAG's own T0 plans on W_fit.
* **The bake.** `raw/2026-10-26-hacc-bake/bake_hacc.py`, unchanged.
* ⛔ No eval window enters any fit or selection.
* ⛔ If DAG's T0 plans on W_fit cannot be produced on the dev-box CPU (the fit-set cache is absent, or the run would take > 3 h), this
  PREREG is **BLOCKED**, with what is missing named. It never falls back to another fit set.

## 2. The read

* **Read A of `PREREG_V8F_DAG_REREAD.md` §2, verbatim** (rows, D1 / D2 / D3, crawl rows, readers, estimator), with arm = DAG-HACC, at DAG's
  decisive T1-kin settings: m = 10, causal bucket, dead-reckoned nav, logged scene. That is 2 rollouts of about 5 min each, with the same tool
  tree and flags; only `--ckpt`, `--tag` and the output paths change. STRICT (frozen scene) is beside.
* **KEEP (new guard):** DAG-HACC − DAG at T1-kin, ADE6. The upper bound must be ≤ +F/2 = **+0.0604 m** at both seeds. Smoothness may not be
  bought with position.
* **Beside:**
  * DAG-HACC − DAG on every Read-A row, so each row's move is attributed;
  * the four families at T0;
  * S-POS against CTRV within run, if the rollouts carry S checkpoints.

## 3. Controls (each must pass, or the read is REFUSED)

* **C-H1, recipe identity:** the fit code, run on M2b's banked fit inputs, reproduces HACC's banked w for M2b_s0 / s1, max |Δ| ≤ 1e-12, with
  the same λ*.
* **C-H2, train-only:** the λ-selection path receives training arrays alone. HACC's own test that passes an eval array must raise.
* **C-H3, bake identity:** a bake with w ≡ 1 reproduces plain DAG's plans exactly (max |Δ| = 0.0 m) on 24 eval windows.
* **C-H4, bake scope:** the DAG-HACC checkpoint differs from DAG's only in the post-map buffers. Every other tensor is bit-identical.
* **C-H5:** Read A's C1–C4, re-run on the new arm.

## 4. Outcomes, committed now

| outcome | condition | committed next |
|---|---|---|
| **DOMINANT** | D1, D2 and D3 at both seeds; crawl rows pass; KEEP holds | the dominance claim at T1-kin over the kinematic floor, on all four families, at both seeds, is registered for DAG-HACC. Signal the PI. Evidence videos are re-rendered on DAG-HACC. FLAT-DAG stays the attribution read |
| **DOMINANT, CRAWL OPEN** | as DOMINANT, but M1 separated-worse at ≥ 1 seed | as DOMINANT, and DAgger iteration 2 for the crawl (box 2, PI go), evaluated under the same post-map |
| **LONGITUDINAL FIXED** | KEEP holds; no LONGITUDINAL row is separated-worse at either seed; some LATERAL / crawl row still is | the post-map is the longitudinal lever and DAG-HACC is the base's deployed form. The remaining rows go to DAgger iteration 2, read under the post-map |
| **NOT FIXED** | KEEP holds; a LONGITUDINAL row is still separated-worse at ≥ 1 seed | the post-map is not enough on DAG's own motion. Next: OACC's acceleration-matching term added to DAG's fine-tune recipe (box 2, PI go), under its own PREREG |
| **COSTS POSITION** | KEEP fails at ≥ 1 seed | not adopted; DAG stays plain. Next: as NOT FIXED |
| **BLOCKED** | §1's fit inputs cannot be produced on the dev-box CPU | the missing input is named; it needs a box-2 eval of DAG on W_fit (PI go) |
| **REFUSED** | a control fails | fix and re-run |

The table is complete:
* REFUSED and BLOCKED are decided first;
* then KEEP: if it fails, the outcome is COSTS POSITION;
* with KEEP held, DOMINANT is tested, then LONGITUDINAL FIXED, and NOT FIXED takes the rest.

## 5. Reporting

* Per seed: delta [lo, hi], separated, arm means and n, with the tier stamp.
* STRATEGIC is reported (not paused for v8F).
* A row that cannot be computed is reported with its reason and n.
* Package: `raw/2026-10-27-dag-hacc/`. Result: `RESULT_V8F_DAG_HACC.md`. The register row is updated in the same landing.
