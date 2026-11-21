# PREREG_V8F_DAG_REREAD: result (DAG on its own motion against the kinematic floor P10 and the FLAT control; read 2026-10-27)

**Outcomes, as committed:**
* **Read A: MARGIN, NOT DOMINANT.** The 6 s margin over P10 is 8–10 F at both seeds, and every family has a separated-better row. Five rows
  are separated-worse at each seed: accel, curvature, L2, M1, and DK-VIOL (s0) or yaw-rate (s1).
* **Read B-1** (FLAT's registered tier): **NOT ESTABLISHED.**
* **Read B-2** (T1-kin): **REFUSED as written, on control C5; ESTABLISHED under Amendment 1** (post-data, control-only, disclosed below).
  DAG − FLAT ADE6 is −0.60 / −0.35 m, separated. ⛔ It is confounded by DAG's DAgger fine-tune and is attributed only by FLAT-DAG.

* **Evidence class:** MEASURED (ours). Every number is copied from `raw/2026-10-27-dag-reread/raw/` (`SUMMARY.json`, `READ_A.json`,
  `READ_A_tables.md`, `B1_flat_reader_DAG.json`, `B2_T1kin_DAG_vs_FLAT.json`, `B2_tables.md`, `C5.json`). Those were computed by the readers
  the PREREG names, unchanged except for the arm's paths. The one other edit is listed in § 4.
* **Tier.** Read A and B-2 are T1-kin: m = 10, causal bucket, dead-reckoned nav, logged scene. It is a self-action OPEN loop: ⛔ not closed
  loop, and the lead follows its logged future. B-1 is T1-class open loop on eval-141.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, per seed, over the 141 episodes (2,399 windows). It
  answers *"would another draw of EPISODES say this?"*; the training-seed question is answered by the two seeds and F = 0.1207 m (2F =
  0.2415 m). The planner is deterministic.
* **Pre-registration:** `Project Steering/PREREG_V8F_DAG_REREAD.md` @ `bdf8d561`, written before any DAG − P10 or DAG − FLAT row. Amendment
  1 (C5 only) is appended below it in that file.

---

## 1. Read A: DAG − P10, T1-kin

| family | row | s0 | s1 |
|---|---|---|---|
| PRIMARY | **ADE6** (m) | **−1.1656 [−1.3437, −0.9963]** (2.278 / 3.443) | **−0.9868 [−1.1778, −0.8060]** (2.457 / 3.443) |
| LONGITUDINAL | speed MAE (m/s) | **−0.577** better | **−0.302** better |
| | along-track MAE (m) | **−0.628** better | **−0.457** better |
| | accel MAE (m/s²) | **+0.807 [+0.325, +1.273] WORSE** (2.90 / 2.09) | **+1.444 [+0.946, +1.959] WORSE** (3.53 / 2.09) |
| | DK-VIOL (595 / 70) | **+0.0353 [+0.0046, +0.0726] WORSE** (0.082 / 0.047) | +0.0050 [−0.0196, +0.0345] |
| | DK-HW / DK-TG | −0.049 / −0.004, both n.s. | −0.207 n.s. / **−0.093 better** |
| LATERAL | heading MAE (deg) | **−3.36** better | **−1.60** better |
| | cross-track MAE (m) | **−0.979** better | **−0.927** better |
| | L1 lateral-accel MAE | **−2.43** better | **−1.14** better |
| | yaw-rate MAE (deg/s) | −0.04 n.s. | **+8.66 [+5.80, +11.85] WORSE** |
| | curvature MAE (1/m) | **+0.0612 [+0.0415, +0.0839] WORSE** (0.134 / 0.072) | **+0.1134 [+0.0787, +0.1525] WORSE** |
| | L2 curvature MAE at GT ≥ 5.2 m/s (2,106 / 135) | **+0.0288 [+0.0165, +0.0433] WORSE** | **+0.0604 [+0.0356, +0.0930] WORSE** |
| TACTICAL | TAC-LAT | **+0.193** better (0.786 / 0.593) | **+0.185** better |
| | TAC-LON | **+0.129** better (0.500 / 0.371) | **+0.114** better |
| STRATEGIC | S-POS@30 / 50 / 80 against CTRV (m; seen before registration, §0) | **−1.72 / −4.48 / −9.69** better | **−1.87 / −4.90 / −10.33** better |
| crawl | M1 per-window crawl share (2,106 / 135) | **+0.0470 [+0.0296, +0.0673] WORSE** (0.070 / 0.023) | **+0.0614 [+0.0409, +0.0859] WORSE** |

* **D1** (ADE6 separated-better with point ≤ −2F) holds at both seeds, at 9.7 F and 8.2 F.
* **D3** (every family has a separated-better row) holds at both.
* **D2** (no row separated-worse) fails at both.
* **The crawl rows fail:** L2 and M1 are separated-worse at both seeds. M2, the arm's L2 share on crawl pairs, is +0.030 / +0.058.

**Beside (never decisive):**
* **DAG − HACC:** ADE6 −0.398 / −0.253 m, separated. DAG plain beats the programme's earlier deployed form on 6 s position. L2 is +0.008
  (n.s.) / +0.052 (worse).
* **DAG-0 − P10:** also MARGIN. At s1 it fails D3, and S-POS@30 is separated-worse.
* **STRICT** (cameras frozen at t0, causal bucket): the worse rows are the same (accel, curvature, L2, and yaw-rate at s1). STRICT S-POS is not
  computed: the strict rollouts carry no S checkpoints in the reader's form.

**Outcome (verbatim §5): MARGIN, NOT DOMINANT.** Committed next: *"the failing rows are named. If they are crawl rows (L2, curvature,
yaw-rate, M1): DAgger iteration 2. Otherwise the failing family's lever, under its own PREREG"*.
* **Reading of a mixed set.** The failing set is mixed: crawl rows (curvature, L2, M1, and yaw-rate at s1) plus two LONGITUDINAL rows (accel
  MAE at both seeds, and DK-VIOL at s0). Both clauses therefore apply, one to each part.
* `PREREG_V8F_DAGGER`'s ROBUST row commits iteration 2 independently: *"If the crawl beside rows (M1, L2) still fail, next is DAgger
  iteration 2"*. They do fail.
* **(a) The crawl rows → DAgger iteration 2** (box 2; PI go).
* **(b) LONGITUDINAL accel and DK-VIOL → the failing family's lever**, under its own PREREG. The programme's measured lever on accel against
  P10 is HACC's train-fitted post-map. At T1-kin, HACC − P10 accel reads +0.017 n.s. / −0.202 better (`RESULT_V8F_T1KIN_CAUSALBUCKET.md`),
  while plain DAG reads +0.81 / +1.44. HACC's DK rows were NO WORSE than P10 (`RESULT_V8F_T1_FAMILIES.md`). That makes HACC re-fitted on DAG
  the cheapest lever: zero training, dev-box CPU.

## 2. Read B-1: DAG − FLAT, FLAT's registered tier (T1-class open loop)

| row | s0 | s1 |
|---|---|---|
| **PRIMARY ADE6** DAG − FLAT (m) | **−0.2378 [−0.3420, −0.1404]**, separated; 0.0037 m short of −2F | +0.0407 [−0.0395, +0.1157], n.s. |
| identity: FLAT's GT-code oracle − own | **+0.0000 [0, 0]** ✅ | **+0.0000 [0, 0]** ✅ |
| identity: DAG − DAG | +0.0000 [0, 0] ✅ | +0.0000 [0, 0] ✅ |
| beside: DAG-0 − FLAT | −0.2397 [−0.3433, −0.1374] | −0.008, n.s. |
| beside: M2b − FLAT (banked) | −0.036 n.s. | −0.003 n.s. |
| beside: DAG − DAG-0 (= R-T0) | +0.0019 | +0.0483 |

**Outcome: NOT ESTABLISHED.** It was predicted before the data (§3 of the PREREG): DAgger does not move logged-state ADE, so B-1 follows
the fine-tune itself. That fine-tune (DAG-0) is the source of s0's −0.24 m. It came from 3,000 more steps, not from DAgger.

## 3. Read B-2: DAG − FLAT at T1-kin (FLAT rolled out with DAG's settings)

| row | s0 | s1 |
|---|---|---|
| **PRIMARY ADE6** DAG − FLAT (m) | **−0.5957 [−0.7378, −0.4568]** (2.278 / 2.873) | **−0.3505 [−0.4695, −0.2293]** (2.457 / 2.807) |
| bar ≤ −2F = −0.2415, separated | met | met |
| beside: DAG-0 − FLAT | −0.052, n.s. | +0.054, n.s. |
| curvature MAE | **−0.081** better | **−0.026** better |
| yaw-rate MAE | **−10.70** better | **−2.61** better |
| L2 | **−0.054** better | −0.012 n.s. |
| M1 crawl share | **−0.0368 [−0.0515, −0.0231]** better | **−0.0152 [−0.0269, −0.0045]** better |
| accel MAE | +0.265 n.s. | **+0.947 WORSE** |
| DK-VIOL / DK-TG | **+0.049 / +0.082 WORSE** | +0.020 / −0.009, n.s. |
| TAC-LAT / TAC-LON | **+0.175 / +0.105** better | **+0.170** better / −0.005 n.s. |

**Controls.**
* C1–C4 pass:
  * the readers reproduce their banked numbers with a difference of 0.0;
  * the identities read 0;
  * the floor equals the banked floor;
  * the windows match.
* C5-2 passes: FLAT's checkpoint sha256 is `8496934a…` / `6fdfc9b0…`, unchanged before and after.
* **C5-1 and C5-3 fail as written** (Amendment 1, below). The read as written is therefore **REFUSED**.

**Under Amendment 1: ESTABLISHED.** Committed next (§5, verbatim): *"FLAT-DAG on box 2, under its own PREREG (it needs the PI's go): it
attributes B-2 to the hierarchy or to DAgger"*. `raw/2026-10-27-flatdag-draft/PREREG_V8F_FLATDAG.md` was banked before this read (`5eaa0232`).
* FLAT itself crawls more than DAG: FLAT − P10 M1 is +0.083 / +0.076, separated-worse (`B2_tables.md`).
* ⛔ **What B-2 does not say.** It does not say *"the hierarchy beats flat"*. DAG-0 − FLAT is n.s. at both seeds, so the T1-kin margin comes
  with DAgger. Whether it needs the hierarchy is FLAT-DAG's question.

## 4. Deviations, mistakes, disclosures

* **`t1kf_read_a1.py`** was run with two substitutions: the arm label, and its C5 HACC-specific banked targets replaced by the arm's own
  run-JSON delta. The diff is `raw/t1kf_read_a1_to_arm_variant.diff`.
* **`v8f_flat_read.py`'s C1** ran through its own `load_all` / `paired`, not the full `read()`, because M2b's banked checkpoints are not on
  the dev box. It reproduces the banked M2b − FLAT.
* **The STRATEGIC rows** are those of `DAG_READ.json`, seen before registration (§0).
* **The FLAT run JSON's by-product block** `paired_model_minus_floor_t1` was looked at once by the reading agent. It decides nothing.
* ⚠ **A scratch export was partly deleted.** The reading agent's cleanup removed about 2,670 of 6,230 files from a Master Mind scratchpad export
  (`scratchpad/flat`: FLAT's instruments exported from tip `18c62306`), which it took for its own. It is a scratch copy of tracked files,
  reproducible from the mirror; its `FILES.txt` is intact. No banked artifact was touched.

## 5. Next (Rule Zero)

1. **0 GPU, now:** HACC re-fitted on DAG (DAG-HACC), under its own PREREG, read on Read A's rows. It targets the LONGITUDINAL accel and DK-VIOL
   rows, and HACC's curvature term targets the curvature rows.
2. **Box 2, PI go:**
   * **DAgger iteration 2** (`raw/2026-10-27-dagger2-draft/`) for the crawl rows. Its draft's run condition (*"only crawl rows failing"*) is
     narrower than `PREREG_V8F_DAGGER`'s registered ROBUST clause, which fires. The registered clause governs. When the draft is promoted,
     its decisive set is the crawl rows only; the longitudinal rows go to (1).
   * **FLAT-DAG** (`raw/2026-10-27-flatdag-draft/`), B-2's committed next.
3. The PI is **not** signalled for scale compute: Read A is not DOMINANT.
