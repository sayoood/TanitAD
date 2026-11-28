"""Report v22: the strict T1-kin read (STRICT MARGIN), HACC-LAT (NOT FIXED), the lateral-row noise floor and LATCOND (CONDITIONED LATERAL
DEFICIT); abstract, 7.15, rows 1n / 6, the status line.  python patch_v22.py <tip>"""
import sys

TP = sys.argv[1][:8]
R = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/report/"
p = R + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()
E = [
    ("It is not dominance: curvature is"
     " worse at both seeds, concentrated below 5 m/s.",
     "It is not dominance: curvature is"
     " worse at both seeds. About a third of that excess is millimetre noise near standstill, and the rest is real at speed (§7.15)."),
    ("and costs 0.06 m against the oracle one (§7.15).",
     "and costs 0.06 m against the oracle one. With both open channels closed at once (the first frame held for 6 s, plus the past-only"
     " bucket), the margin is still **0.73 / 0.64 m** (§7.15)."),
    ("* **Next (committed):** `PREREG_V8F_HACC_LAT` (row 1n). It has two zero-training arms fitted on T1-kin ticks of training windows: HACC"
     " re-fitted on those ticks, and a speed gate that hands the cross axis back to the prior at low speed. Its committed bar is T1"
     " dominance at both seeds.\n",
     "* ⭐ **Strict [M]** (`PREREG_V8F_T1KIN_STRICT`, before data): the cameras frozen at the first frame AND the causal speed bucket.\n"
     "  * **Outcome STRICT MARGIN:** HACC − P10 ADE6 **−0.728 [−0.868, −0.594] / −0.641 [−0.770, −0.519] m**, separated at both seeds.\n"
     "  * The model then reads one frame, its own motion, a speed bucket from the past and the dead-reckoned nav, and still beats the floor"
     " by 0.73 / 0.64 m. Closing the camera channel on top of the causal bucket costs +0.04 / +0.09 m.\n"
     "* **The lateral deficit, examined in three pre-registered steps [M].**\n"
     "  1. **HACC-LAT** (`PREREG_V8F_HACC_LAT`, row 1n): two zero-training arms fitted on T1-kin ticks of training windows. **NOT FIXED.**"
     " HACC's weights re-fitted on the ticks are far worse; a speed gate on the cross axis keeps the margin but not the curvature.\n"
     "  2. **The instrument** (no model arm): 5–10 mm of position noise on the GT, scored against itself, gives 0.021–0.038 1/m of"
     " curvature MAE and 2.4–4.7 deg/s of yaw-rate MAE. That is the size of the deficit, and almost all of it sits below 5 m/s. Curvature"
     " divides by speed, so the canonical curvature and yaw-rate rows are millimetre-noise-dominated near standstill.\n"
     "  3. **Well-conditioned rows** (`PREREG_V8F_T1_LATCOND`, rows fixed from the instrument alone, before any arm was read on them).\n"
     "     * Lateral-acceleration MAE is **not** worse (−0.11 / −0.12 m/s²).\n"
     "     * Curvature MAE at GT speed ≥ 5.2 m/s **is** worse: +0.021 [+0.010, +0.035] / +0.008 [+0.003, +0.015] 1/m. Outcome **CONDITIONED"
     " LATERAL DEFICIT**.\n"
     "     * ⇒ About a third of the excess was the instrument's. The rest is a real curvature-profile mismatch at speed, while heading,"
     " cross-track and lateral acceleration are better or equal.\n"
     "     * From the same state, HACC's single plan is no worse than the floor's, so the excess arises in the loop. Its root cause is being"
     " diagnosed (zero GPU) before the committed box-2 lateral lever is specified.\n"),
    ("| **pre-registered**"
     " (`PREREG_V8F_HACC_LAT`, `96604184`) before any number of either arm; being built, dev-box CPU |",
     "| **NOT FIXED** (`RESULT_V8F_HACC_LAT.md`, T1): arm A −0.22 /"
     " +0.05 m ADE6 against P10, far worse than HACC; arm B −0.88 / −0.76 m, but curvature is still worse. Followed by the noise floor and"
     " `PREREG_V8F_T1_LATCOND` (CONDITIONED LATERAL DEFICIT, §7.15) |"),
    ("−0.734 m, separated. T2 is not provisioned |",
     "−0.734 m, separated. With the cameras frozen as well (`PREREG_V8F_T1KIN_STRICT`): **STRICT MARGIN**, −0.728 / −0.641 m. T2 is not"
     " provisioned |"),
    ("**Status:** research report, v21 of 2026-10-26, results through repository tip `76c7d98e`",
     f"**Status:** research report, v22 of 2026-10-26, results through repository tip `{TP}`"),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
t = open(R + "page_template.html", encoding="utf-8").read()
assert "</script" not in s.lower()
open(R + "v8f_report.html", "w", encoding="utf-8").write(t.replace("%%MD%%", s))
print("report v22 built", len(s))
