# PREREG_V8F_DAG_HACC: result (HACC re-fitted on DAG, zero training; T1-kin read on the dev-box CPU; 2026-10-27)

**Outcome, as committed: COSTS POSITION.**
* KEEP fails at seed 0. DAG-HACC − DAG ADE6 at T1-kin is **+0.1264 [+0.0522, +0.2004] m** (2.404 against 2.278), against a bar of upper bound ≤
  +0.0604.
* At seed 1 KEEP holds: −0.1047 [−0.1834, −0.0334] m.
* The PREREG tests KEEP before anything else, so the outcome is COSTS POSITION whatever the other rows read.
* **Committed next, verbatim:** *"not adopted; DAG stays plain. Next: as NOT FIXED"*, that is, *"OACC's acceleration-matching term added to DAG's
  fine-tune recipe (box 2, PI go), under its own PREREG"*.
* **All registered controls PASS** (`raw/CONTROLS.json`: `all_registered_controls_pass: true`):
  * **C-H1:** the fit code reproduces HACC's banked w for M2b, max |dw| = 0.0, with the same λ*.
  * **C-H2:** fitting is train-only.
  * **C-H3:** a bake with w ≡ 1 is the identity, max |Δ| = 0.0 m.
  * **C-H4:** the bake adds only `hacc_w`; every other tensor is bit-identical.
  * **C1–C4:** the readers reproduce their banked numbers with a difference of 0.0, the identities read 0, the floor is 0.0 m from the banked
    floor, and the windows match.
* **Evidence class:** MEASURED (ours). Every number comes from `raw/2026-10-27-dag-hacc/READ_DAG_HACC.md` and its `raw/` JSON.
* **Tier:** T1-kin, a self-action OPEN loop (⛔ not closed loop), at m = 10 with the causal bucket.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows.
* **Pre-registration:** `Project Steering/PREREG_V8F_DAG_HACC.md` @ `2a416ca7`, before any fit.

## What the post-map did (DAG-HACC − P10, against plain DAG − P10 from `RESULT_V8F_DAG_REREAD`)

| row | s0 DAG-HACC | s0 plain DAG | s1 DAG-HACC | s1 plain DAG |
|---|---|---|---|---|
| **ADE6** (m) | **−1.039** (8.6 F) | −1.166 | **−1.092** (9.0 F) | −0.987 |
| accel MAE | **−0.644 better** | +0.807 WORSE | **−0.847 better** | +1.444 WORSE |
| DK-VIOL | +0.027 n.s. | +0.035 WORSE | +0.003 n.s. | +0.005 n.s. |
| DK-HW / DK-TG | n.s. / n.s. | n.s. / n.s. | **−0.317 / −0.114 better** | n.s. / −0.093 better |
| yaw-rate MAE | −1.14 n.s. | −0.04 n.s. | +1.22 n.s. | +8.66 WORSE |
| curvature MAE | **+0.0174 WORSE** | +0.0612 WORSE | **+0.0471 WORSE** | +0.1134 WORSE |
| L2 (GT ≥ 5.2 m/s) | **+0.0112 WORSE** | +0.0288 WORSE | **+0.0280 WORSE** | +0.0604 WORSE |
| M1 crawl share | **+0.0242 WORSE** | +0.0470 WORSE | **+0.0390 WORSE** | +0.0614 WORSE |
| TAC-LAT / TAC-LON | **+0.202 / +0.165** | +0.193 / +0.129 | **+0.165 / +0.179** | +0.185 / +0.114 |
| S-POS@80 against CTRV (m) | **−10.77** | −9.69 | **−12.07** | −10.33 |

All other rows (speed, along-track, heading, cross-track, L1, S-POS@30 / 50) are separated-better at both seeds.
* **D1 and D3 hold at both seeds.** D2 fails at both, on curvature and L2 only.
* **No LONGITUDINAL row is separated-worse at either seed.**
* **The fit** used DAG's own T0 plans on HBLEND's W_fit (141 TRAIN clips, 9,582 windows): λ* 35309 / 33857 (m* 16 / 16).

## Reading (POST-DATA; it changes no outcome)

* **The post-map does what the read asked of it on the longitudinal family.** It also halves the curvature, L2 and M1 excess, and removes
  s1's yaw-rate row. What remains against P10 is the crawl and curvature set, which is DAgger iteration 2's target.
* **KEEP's two seeds disagree in sign:** +0.126 m and −0.105 m, about ±1 F. That is the order of run-to-run noise on this rig (F = 0.1207 m).
  A single seed failing by 0.066 m over its bar is a failure as registered. It is also the cheapest thing to re-test once more seeds exist.
* **The same trade appeared before:** OACC (trained) bought accel and failed the same KEEP bar on M2b (`RESULT_V8F_OACC.md`). Two levers on
  two routes now say that, on this rig, smoothing the longitudinal profile costs about 0.1 m of 6 s position at one seed.

## Next (Rule Zero)

* **Committed:** OACC's acceleration term added to DAG's fine-tune recipe (DAG-OACC), on box 2, under its own PREREG. It needs the PI's go.
* **Unchanged:** DAgger iteration 2 for the crawl rows, and FLAT-DAG for attribution, both on box 2 with the PI's go.
* **Not done:**
  * The HBLEND fit cache (2.8 GB) is held only in a Master Mind scratchpad on C:. It is derived from the TRAIN cache and reproducible, and it
    is not banked: E: has 6.3 GB free.
  * The four families at T0 for DAG-HACC (beside) are in the package, where computed.
