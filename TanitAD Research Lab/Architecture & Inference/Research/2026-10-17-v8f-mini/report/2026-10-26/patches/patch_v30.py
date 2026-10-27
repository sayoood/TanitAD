"""Report v30: the DAG families, the dominance / FLAT re-read (7.20), DAG-HACC (7.21), evidence video v5; abstract, 8.1, 8.3, 8.4 rows 1s / 1t.
python patch_v30.py <tip>"""
import re
import sys

TP = sys.argv[1][:8]
R = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/report/"
p = R + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()

OLD_ABS = "its dominance over P10 and its FLAT repeat are pre-registered and running (`PREREG_V8F_DAG_REREAD`)."
NEW_ABS = (OLD_ABS + "\n"
    "* **On its own motion, DAG beats the kinematic floor by 1.17 / 0.99 m of 6 s ADE (8–10 F), and is NOT dominant (§7.20).** Every family"
    " has a separated-better row, but accel, curvature, L2 and the launch crawl M1 are separated-worse against P10. DAG's tactical"
    " agreement is +0.19 / +0.19 over P10, and its S checkpoints halve their error against DAG-0. The cost DAgger adds is distance keeping:"
    " the lead-violation share doubles against its control (T1-kin, both seeds).\n"
    "* **Against its capacity-matched FLAT control** at T1-kin, DAG is better by 0.60 / 0.35 m. That is a pass under a disclosed post-data"
    " control amendment, and the margin rides on DAgger (DAG-0 − FLAT is not separated), so FLAT-DAG is the attribution read (§7.20).\n"
    "* **HACC re-fitted on DAG (zero training)** clears every longitudinal row against P10, leaving curvature, L2 and M1. It reads COSTS"
    " POSITION, because 6 s ADE rises by 0.13 m at seed 0, while it falls by 0.10 m at seed 1 (§7.21).")

SEC = """### 7.20 DAG against the kinematic floor and the flat control, on its own motion

* **Setup** (`PREREG_V8F_DAG_REREAD`, `bdf8d561`, written before any of its rows). Everything is read at T1-kin: m = 10, causal bucket,
  dead-reckoned nav, logged scene, self-action open loop. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes.
  * **Read A, dominance:** ADE6 separated-better by ≥ 2F (D1), no row of the four families separated-worse (D2), and a separated-better row
    in every family (D3), all at both seeds.
  * **Read B:** DAG − FLAT, at FLAT's own open-loop tier (B-1) and at T1-kin (B-2).
* **Read A [M]** (`RESULT_V8F_DAG_REREAD.md`). DAG − P10:

| row | seed 0 | seed 1 |
|---|---|---|
| ADE6 (m) | **−1.166 [−1.344, −0.996]** | **−0.987 [−1.178, −0.806]** |
| speed / along-track / heading / cross-track MAE | all separated-better | all separated-better |
| TAC-LAT / TAC-LON | **+0.193 / +0.129** | **+0.185 / +0.114** |
| accel MAE (m/s²) | **+0.807 WORSE** | **+1.444 WORSE** |
| curvature MAE / L2 (1/m) | **+0.061 / +0.029 WORSE** | **+0.113 / +0.060 WORSE** |
| M1 launch-crawl share | **+0.047 WORSE** | **+0.061 WORSE** |
| DK-VIOL / yaw-rate | **+0.035 WORSE** / n.s. | n.s. / **+8.66 WORSE** |

  The outcome is **MARGIN, NOT DOMINANT**. The failing set is mixed:
  * the crawl rows go to DAgger iteration 2, as committed by both PREREGs;
  * the longitudinal rows go to the longitudinal family's lever (§7.21).
* **The four families, DAG − DAG-0** (`RESULT_V8F_DAGGER.md` §3; 14 controls PASS):
  * at T1-kin, speed, along-track, heading, cross-track, plan-derived decisions and turn F1 are separated-better at both seeds;
  * S's checkpoint error is about halved: −2.5 / −3.4 m at 30 m, and −7.6 / −10.4 m at 80 m, paired;
  * **distance keeping is separated-worse at both seeds.** The lead-violation share behind a lead goes 0.082 against 0.040, and 0.052
    against 0.022. The min-TTC error rises by 0.87 / 0.60 s.
* **Read B [M]:**
  * **B-1, open loop:** NOT ESTABLISHED (−0.238, separated / +0.041 m). The s0 gain is the fine-tune's, not DAgger's: DAG-0 − FLAT is −0.240.
  * **B-2, T1-kin:** DAG − FLAT is **−0.596 [−0.738, −0.457] / −0.351 [−0.470, −0.229] m**, and also better on curvature, yaw-rate, L2 and
    M1.
    * Its control C5 failed as written: two output-path flags differ, and an inspection key could not show the cut. The read as written is
      therefore REFUSED. It passes under Amendment 1, which is post-data, control-only and disclosed: the cut is shown by the checkpoint
      config and by FLAT's exact-zero oracle identity.
    * DAG-0 − FLAT is not separated at either seed, so the margin comes with DAgger. Whether it needs the hierarchy is FLAT-DAG's question.

### 7.21 HACC on DAG: the post-map clears the longitudinal rows and costs position at one seed

* **Setup** (`PREREG_V8F_DAG_HACC`, `2a416ca7`, before any fit). HACC's rule is re-fitted verbatim on DAG's own T0 plans over HBLEND's TRAIN
  fit set (141 clips, 9,582 windows), giving λ* 35309 / 33857, and baked as 30 × 2 weights. It is read on Read A's rows, with a KEEP guard:
  DAG-HACC − DAG ADE6, upper bound ≤ +F/2.
* **Controls:**
  * the recipe reproduces HACC's banked weights exactly;
  * a bake with w ≡ 1 is the identity (0.0 m);
  * the bake adds only `hacc_w`.
* **Result [M]** (`RESULT_V8F_DAG_HACC.md`):
  * Against P10, accel reads **−0.644 / −0.847** (plain DAG: +0.807 / +1.444). No longitudinal row is separated-worse at either seed, and
    s1's yaw-rate row is gone.
  * The curvature / L2 / M1 excess roughly halves (curvature +0.017 / +0.047, L2 +0.011 / +0.028, M1 +0.024 / +0.039), but stays
    separated-worse.
  * ADE6 against P10 is −1.039 / −1.092 m.
  * **KEEP fails at seed 0:** +0.126 [+0.052, +0.200] m. It holds at seed 1: −0.105 m.
* **Outcome, as committed: COSTS POSITION.** The post-map is not adopted. Its committed next is the trained form of the same lever: OACC's
  acceleration term in DAG's fine-tune (box 2). The trade has now appeared on two routes (OACC on M2b, §7.16; HACC on DAG). On this rig,
  smoothing the longitudinal profile costs about 0.1 m of 6 s position at one seed. The two seeds disagree in sign, at about ±1 F.

"""
ANCHOR_8 = "\n---\n\n## 8. Discussion"
OLD_81 = "cuts the executed 6 s ADE by 0.54 / 0.40 m against its own fine-tune control."
NEW_81 = (OLD_81 + "\n9. **On its own motion the model beats the kinematic floor by 8–10 F of 6 s ADE** (§7.20). It does so with better tactical"
          " agreement and halved strategic checkpoint error, and with better speed, heading and cross-track. It also beats its flat control"
          " at T1-kin with DAgger. It is not dominant: curvature, L2 and the launch crawl remain worse than the floor.")
OLD_83 = "The PI's bar is **met on one axis only**."
NEW_83 = ("**Update (v30).** On the model's own motion (T1-kin), DAG with HACC's post-map is two row-families away from dominance over the"
          " floor:\n"
          "* every longitudinal, tactical and strategic row is better or not separated;\n"
          "* curvature / L2 and the launch crawl M1 remain separated-worse;\n"
          "* its position guard fails at one seed (§7.21).\n"
          "The levers aimed at those rows are DAgger iteration 2 and DAG-OACC, both on box 2. Attribution to the hierarchy waits on FLAT-DAG. "
          + OLD_83)
OLD_1S = "**pre-registered** (`PREREG_V8F_DAG_REREAD`, `bdf8d561`, before any of its rows); running on the dev-box CPU |"
NEW_1S = ("**A: MARGIN, NOT DOMINANT; B-1: NOT ESTABLISHED; B-2: ESTABLISHED under a post-data control amendment** (§7.20;"
          " `RESULT_V8F_DAG_REREAD.md`) |\n"
          "| 1t | **DAG-HACC** (zero training): HACC's rule re-fitted on DAG, read on Read A with KEEP | DOMINANT / LONGITUDINAL FIXED / NOT FIXED /"
          " COSTS POSITION (KEEP: ADE6 upper bound ≤ +F/2 vs DAG) | **COSTS POSITION** (§7.21; `RESULT_V8F_DAG_HACC.md`): every"
          " longitudinal row cleared, KEEP +0.126 m at s0. Next: DAG-OACC, DAgger iteration 2 and FLAT-DAG on box 2 (PI go) |")
OLD_V = "The second video (`v8f_evidence_v4.mp4`, 381 s, below;"
NEW_V = ("The second video (`v8f_evidence_v5.mp4`, 536 s, below; v5 adds chapter 8, **DAgger: trained on its own states**: DAG against DAG-0 on"
         " their own motion, with S's checkpoints, T's decisions against the label and O's executed path at every replan tick, on seven"
         " rule-selected windows with the worst kept; earlier:")
E = [(OLD_ABS, NEW_ABS), (ANCHOR_8, "\n\n" + SEC.rstrip("\n") + "\n" + ANCHOR_8), (OLD_81, NEW_81), (OLD_83, NEW_83), (OLD_1S, NEW_1S),
     (OLD_V, NEW_V)]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
assert all("closed loop" not in b.lower() and "drives" not in b.lower() for _, b in E)
n0 = s.count("**Status:** research report, v29 of 2026-10-27")
assert n0 == 1
s = re.sub(r"\*\*Status:\*\* research report, v29 of 2026-10-27, results through repository tip `[0-9a-f]{8}`",
           f"**Status:** research report, v30 of 2026-10-27, results through repository tip `{TP}`", s, count=1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
t = open(R + "page_template.html", encoding="utf-8").read()
OT = "src=\"v8f_evidence_v4.mp4\"></video>' +\n      '<figcaption><b>Video 2 (v4, 381 s)</b>Eight chapters; v4 adds chapter 7,"
NT = ("src=\"v8f_evidence_v5.mp4\"></video>' +\n      '<figcaption><b>Video 2 (v5, 536 s)</b>Nine chapters; v5 adds chapter 8, <b>DAgger: trained on"
      " its own states</b> (DAG vs DAG-0 on their own motion; S checkpoints, T decisions against the label and O executed path at every replan"
      " tick; seven rule-selected windows, the worst kept; T1-kin, self-action OPEN loop). v4 added chapter 7,")
assert t.count(OT) == 1
t = t.replace(OT, NT)
open(R + "page_template.html", "w", encoding="utf-8", newline="\n").write(t)
assert "</script" not in s.lower()
open(R + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(t.replace("%%MD%%", s))
print("report v30 built", len(s))
