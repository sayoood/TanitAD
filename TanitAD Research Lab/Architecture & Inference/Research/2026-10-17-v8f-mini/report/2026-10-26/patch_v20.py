"""Report v20: the T1-kin read of HACC (T1 BEATS THE FLOOR; not dominance; the margin rests on the declared speed bucket), section 7.15,
rows 1m / 1n / 6, sections 8.1-8.3, and the T1-kin wording (a self-action OPEN loop, never "closed loop").  python patch_v20.py <result commit> <m5 line>"""
import sys

RC, M5 = sys.argv[1][:8], sys.argv[2]
R = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/report/"
p = R + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()
E = []
A = ("We claim neither state of the art nor closed-loop driving. The dominance above is over the programme's own kinematic floor, in open loop;"
     " the closed-loop read is the next tier.")
E.append((A, "We claim neither state of the art nor closed-loop driving. The dominance above is over the programme's own kinematic floor, in open"
          " loop. In a self-action open loop it becomes a margin, not dominance (next paragraph); T2, the closed-loop tier, is not provisioned.\n\n"
          "**The margin survives the model's own motion, conditionally [M].** In T1-kin, the ego state the model reads is rebuilt from its own"
          " executed motion at every 2 s replan, while the cameras stay logged. There, HACC beats the floor driven the same way by **0.83 / 0.79 m**"
          " of 6 s ADE at both seeds, and the margin is 0.79 / 0.71 m with the cameras frozen at the first frame. It is not dominance: curvature is"
          " worse at both seeds, concentrated below 5 m/s. And it rests on one declared input: with the max-target-speed bucket removed, the margin"
          " is gone (+0.24 / +0.08 m, not separated). On this corpus that bucket is built from the ego's realised future speed (§7.15)."))
E.append(("| 1m | **HACC**:", "| 1m | **HACC**:"))  # anchor check only
A = "the videos are re-rendered on HACC (v3). Next: the T1-kin closed-loop read (PI go) |\n"
E.append((A, f"the videos are re-rendered on HACC (v3). At T1-kin: **T1 BEATS THE FLOOR** (row 6, §7.15) |\n"
          "| 1n | **HACC-LAT** (zero training): arm A, HACC's weights and λ rule re-fitted on T1-kin ticks of TRAINING windows; arm B, the cross axis"
          " falls back to P10 below a train-chosen speed gate | T1 DOMINANCE over P10 at both seeds, and ADE6 within F/2 of HACC | **pre-registered**"
          f" (`PREREG_V8F_HACC_LAT`, `{RC}`) before any number of either arm; being built, dev-box CPU |\n"))
A = ("| 6 | **T1-kin / T2** closed-loop read | `D-V8F-T1KIN-1` built; the T1-kin floor runs away in its own loop at short replan periods (P10: 3.44 m at"
     " m = 10, 23.3 m at m = 2) **[I]** | needs the pooled eval cache and a PI go; T2 is not provisioned |")
E.append((A, "| 6 | **T1-kin** (self-action open loop) read of HACC; T2 (closed loop) | `PREREG_V8F_T1KIN_HACC`: PRIMARY HACC − P10 ADE6 ≤ −2F, separated,"
          f" both seeds; DOMINANCE adds no four-family row worse | **T1 BEATS THE FLOOR** (`RESULT_V8F_T1KIN_HACC.md`, `{RC}`, T1): ADE6 −0.829 /"
          " −0.795 m, separated; curvature and s0 yaw rate separated-worse (low speed). The margin rests on the declared speed bucket"
          " (`D-V8F-SPEEDBUCKET-T1`). T2 is not provisioned |"))
A = "5. The variance structure of this rig is quantified, and it dictates experimental design (§5.3): replicates and seeds, not windows.\n"
E.append((A, A + "6. **The margin over the floor survives the model's own motion** (T1-kin, §7.15): 0.83 / 0.79 m of 6 s ADE at both seeds, and 0.79 /"
          " 0.71 m with the cameras frozen at the first frame. It is conditional on the declared speed bucket, and it is not dominance.\n"))
A = ("* Any **driving** claim: all numbers are open loop (T0/T1), no T2 closed-loop run exists, `D-V8F-T1KIN-1` was not run on cameras, and the 2 s"
     " and acceleration families are *worse* than the floor.")
E.append((A, "* Any **driving** claim: all numbers are open loop (T0/T1), and no T2 closed-loop run exists. The T1-kin read (§7.15) keeps the human's"
          " cameras, and its margin disappears without the declared speed bucket, which on this corpus is derived from the ego's future speed."
          " Without the post-map, the raw arms are worse than the floor at 2 s and in acceleration."))
A = "(iv) there is no closed-loop evidence. What the report"
E.append((A, "(iv) there is no closed-loop evidence: in the self-action open loop (T1-kin), the model keeps a 0.8 m margin but loses dominance on low-speed"
          " curvature, and the margin rests on a speed input this corpus builds from the future (§7.15). What the report"))
A = "\n---\n\n## 8. Discussion"
E.append((A, "\n### 7.15 T1-kin: HACC on its own motion (self-action open loop)\n\n"
          "`PREREG_V8F_T1KIN_HACC` (with three amendments, each made before the decision rows were read; R74) asks whether HACC's open-loop"
          " margin survives when the model drives on its own outputs. In **T1-kin**, every replan (m = 10 steps, 2 s) rebuilds the ego state the"
          " model reads from its own executed motion: speed, acceleration, yaw rate and the dead-reckoned nav. The camera frames stay the human's"
          " (T2, re-perception, is not provisioned), so ⛔ this is an open loop, not closed-loop driving. The P10 floor is driven the same way, in"
          " its own loop. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows; the two training seeds are the"
          " replicate.\n\n"
          "| HACC − P10, T1-kin m = 10 (lower is better) | s0 | s1 |\n|---|---|---|\n"
          "| **ADE6 (m), PRIMARY** | **−0.829 [−0.971, −0.689]** | **−0.795 [−0.938, −0.658]** |\n"
          "| ADE2 (m) | −0.011, better | −0.009, better |\n"
          "| speed / along-track MAE (LONG) | −0.39 m/s, −0.54 m, better | −0.43 m/s, −0.58 m, better |\n"
          "| accel MAE (m/s², LONG) | +0.045, n.s. | −0.229, better |\n"
          "| heading / cross-track MAE (LAT) | −0.43 deg n.s., −0.57 m better | −0.77 deg, −0.43 m, better |\n"
          "| **curvature MAE (1/m, LAT)** | **+0.040, worse** | **+0.016, worse** |\n"
          "| **yaw-rate MAE (deg/s, LAT)** | **+4.20, worse** | +1.53, n.s. |\n\n"
          "**Outcome: T1 BEATS THE FLOOR** (the pre-registered table, verbatim). PRIMARY holds at both seeds; DOMINANCE fails on the lateral rows."
          " The tactical and strategic families are not computed at T1 (the dumps carry no per-tick T code; a work item). Every control passed,"
          " and the read was re-derived independently to the draw.\n\n"
          "* **The loop widens the margin.** One shot (m = 30) gives −0.53 m. At m = 10 the floor degrades from 2.49 to 3.44 m in its own loop,"
          f" while HACC degrades from 1.96 to 2.61 / 2.65 m. {M5}\n"
          "* **The post-map is what keeps the loop smooth.** Plain M2b at m = 10 also beats the floor on ADE6 (−0.41 / −0.65 m) but is"
          " **+1.22 / +1.08 m/s² worse in acceleration**. HACC removes that roughness.\n"
          "* **Brackets** (which open channel carries the margin): with the cameras frozen at the first frame, the margin is still −0.79 / −0.71 m,"
          " so the logged cameras flatter little. ⚠ **With the speed bucket removed, it is gone: +0.24 / +0.08 m, not separated**, and HACC's"
          " accel turns worse than the floor's. The bucket is a PI-declared input (max target speed), but on PhysicalAI it is the max of the ego's"
          " realised future speed. So the claim is conditional on a declared, optimistic input, and it is filed for the PI as"
          " `D-V8F-SPEEDBUCKET-T1`. This is SPEEDCAL's finding from the other side: speed is the largest lever, and here the model takes it from"
          " the bucket, not from what it sees. ⚠ \"Off\" is also outside the training distribution, so the 1 m mixes the bucket's information with"
          " the model's response to an absent input.\n"
          "* **Where the lateral excess lives** (descriptive). It is not the replan seams and not the camera viewpoint (it does not grow with the"
          " pose offset). It is speed: below 5.2 m/s the curvature excess is +0.40 / +0.25 1/m, and above 11.8 m/s it is about zero. Near"
          " standstill a millimetre wiggle is a large curvature, and HACC's metre-scale objective cannot see it. In the first 2 s (logged state)"
          " HACC is better than the floor; the excess starts with the first plan issued from its own slow state.\n"
          "* **Next (committed):** `PREREG_V8F_HACC_LAT` (row 1n). It has two zero-training arms fitted on T1-kin ticks of training windows: HACC"
          " re-fitted on those ticks, and a speed gate that hands the cross axis back to the prior at low speed. Its committed bar is T1"
          " dominance at both seeds.\n"
          "\n---\n\n## 8. Discussion"))
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
t = open(R + "page_template.html", encoding="utf-8").read()
assert "</script" not in s.lower()
open(R + "v8f_report.html", "w", encoding="utf-8").write(t.replace("%%MD%%", s))
print("report v20 built", len(s))
