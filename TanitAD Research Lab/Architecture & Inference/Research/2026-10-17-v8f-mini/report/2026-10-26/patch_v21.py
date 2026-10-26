"""Report v21: the causal speed bucket (PREREG_V8F_T1KIN_CAUSALBUCKET, CAUSAL KEEPS THE MARGIN) lifts v20's main caveat; abstract, 7.15,
row 6, 8.1-8.3 and the status line.  python patch_v21.py <result commit>"""
import sys

RC = sys.argv[1][:8]
R = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/report/"
p = R + "V8F_SCIENTIFIC_REPORT.md"
s = open(p, encoding="utf-8").read()
E = [
    ("**The margin survives the model's own motion, conditionally [M].**", "**The margin survives the model's own motion [M].**"),
    (" And it rests on one declared input: with the max-target-speed bucket removed, the margin"
     " is gone (+0.24 / +0.08 m, not separated). On this corpus that bucket is built from the ego's realised future speed (§7.15).",
     " And it does not need the future-derived speed bucket the corpus supplies. Rebuilt from the last 4 s of measured speed alone, the bucket"
     " keeps a margin of **0.77 / 0.73 m** at both seeds, and costs 0.06 m against the oracle one (§7.15)."),
    (" The margin rests on the declared speed bucket (`D-V8F-SPEEDBUCKET-T1`). T2 is not provisioned |",
     f" With the speed bucket built from the past 4 s only (`PREREG_V8F_T1KIN_CAUSALBUCKET`, `{RC}`): **CAUSAL KEEPS THE MARGIN**, −0.768 /"
     " −0.734 m, separated. T2 is not provisioned |"),
    ("0.71 m with the cameras frozen at the first frame. It is conditional on the declared speed bucket, and it is not dominance.\n",
     "0.71 m with the cameras frozen at the first frame. It holds with a speed bucket built from the past alone (0.77 / 0.73 m), and it is not"
     " dominance.\n"),
    ("The T1-kin read (§7.15) keeps the human's"
     " cameras, and its margin disappears without the declared speed bucket, which on this corpus is derived from the ego's future speed.",
     "The T1-kin read (§7.15) keeps the human's"
     " cameras. Its speed input can be built from the past alone without losing the margin, but frozen cameras together with that causal"
     " bucket are not measured."),
    ("curvature, and the margin rests on a speed input this corpus builds from the future (§7.15).",
     "curvature; the margin holds with a speed input built from the past alone (§7.15)."),
    ("* **Where the lateral excess lives** (descriptive).",
     f"* ⭐ **The causal bucket [M]** (`PREREG_V8F_T1KIN_CAUSALBUCKET`, landed before data; `RESULT_V8F_T1KIN_CAUSALBUCKET.md`, `{RC}`).\n"
     "  * The bucket the model reads at each replan is the sidecar's own containing-window rule applied to the max MEASURED ego speed of the"
     " last 4 s (its own executed speed after t0). No future row enters it.\n"
     "  * **Outcome CAUSAL KEEPS THE MARGIN:** HACC − P10 ADE6 **−0.768 [−0.918, −0.626] / −0.734 [−0.878, −0.598] m**, separated at both"
     " seeds. Against the oracle bucket it costs +0.061 / +0.061 m, and the two buckets agree on 79 % of ticks.\n"
     "  * ⇒ **The `--limit off` collapse was the model's response to an ABSENT input (it never saw one in training), not the oracle's"
     " information.** v8F's T1 result stands on an input built from the past alone.\n"
     "  * The lateral rows are unchanged, which is HACC-LAT's target.\n"
     "  * Every control passed: logged and off byte-identical, 4 / 4 leak mutants red, the tick-0 bucket equal to a hand computation, and"
     " the floor exactly unchanged.\n"
     "* **Where the lateral excess lives** (descriptive)."),
    ("**Status:** research report, v20 of 2026-10-26, results through repository tip `96604184`",
     f"**Status:** research report, v21 of 2026-10-26, results through repository tip `{RC}`"),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
t = open(R + "page_template.html", encoding="utf-8").read()
assert "</script" not in s.lower()
open(R + "v8f_report.html", "w", encoding="utf-8").write(t.replace("%%MD%%", s))
print("report v21 built", len(s))
