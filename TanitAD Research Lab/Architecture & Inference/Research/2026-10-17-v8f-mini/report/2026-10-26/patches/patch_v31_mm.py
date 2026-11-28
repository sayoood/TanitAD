"""Report v31, Master Mind review pass (after patch_v31.py). The DAGGER2 claims the agent tagged [I] were re-read from RESULT_V8F_DAGGER2.md
(HARMS, section 1; accel MAE DAG2 - DAG2-0 +0.5846 / +0.6034, section 2), so they cite that file directly; one sentence named SPEEDREAD as aimed
at the crawl set, which it is not (it asks whether T's input carries the speed information O needs). The .html is rebuilt from the template.
python patch_v31_mm.py <report dir> <page_template_v30.html>"""
import sys

D, TPL = sys.argv[1], sys.argv[2]
P = D + "/V8F_SCIENTIFIC_REPORT.md"
md = open(P, encoding="utf-8").read()
EDITS = [
    ("DAG2 − DAG2-0 (DAgger iteration 2 against its control) was +0.58 / +0.60 (**[I]**, quoted from `RESULT_V8F_FLATDAG.md`).",
     "DAG2 − DAG2-0 (DAgger iteration 2 against its control) was +0.58 / +0.60, separated at both seeds (`RESULT_V8F_DAGGER2.md` §2)."),
    ("read HARMS (`RESULT_V8F_DAGGER2`, cited by `PREREG_V8F_DAG_ORC`; **[I]**,\n    not re-read here).",
     "read HARMS (`RESULT_V8F_DAGGER2.md` §1: the crawl rows stay separated-worse against P10 at\n    both seeds, and yaw-rate MAE turns newly separated-worse at s0), so it was not adopted."),
    ("* the arms aimed at that crawl set are DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`) and SPEEDREAD"
     " (`PREREG_V8F_SPEEDREAD`), both pre-registered and not yet read;",
     "* the arm aimed at that crawl set is DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`); SPEEDREAD"
     " (`PREREG_V8F_SPEEDREAD`) asks whether the speed information O needs is in T's own input; both are pre-registered and not yet read;"),
    ("* DAgger iteration 2, the other lever v30 named, read HARMS (`RESULT_V8F_DAGGER2`, cited by `PREREG_V8F_DAG_ORC`; **[I]**).",
     "* DAgger iteration 2, the other lever v30 named, read HARMS (`RESULT_V8F_DAGGER2.md` §1), so it was not adopted."),
]
for old, new in EDITS:
    assert md.count(old) == 1, old[:80]
    md = md.replace(old, new)
assert "**[I]**, quoted from" not in md
open(P, "w", encoding="utf-8", newline="\n").write(md)
tpl = open(TPL, encoding="utf-8").read()
assert tpl.count("%%MD%%") == 1
open(D + "/v8f_report.html", "w", encoding="utf-8", newline="\n").write(tpl.replace("%%MD%%", md))
print("v31 MM pass applied:", len(EDITS), "edits")
