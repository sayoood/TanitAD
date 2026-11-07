"""_mk_reader: derive wspd_read.py from the banked wspedit_read.py by exact-string replacements (each must match once). The instrument (the LF / A-series reader, loaded by path) is untouched."""
import sys
sys.dont_write_bytecode = True
src = open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit/wspedit_read.py", encoding="utf-8", newline="").read()
out = src
def rep(a, b, n=1):
    global out
    assert out.count(a) == n, (out.count(a), a)
    out = out.replace(a, b)
rep('"""wspedit_read: READ 2 of PREREG_WSP_GAIN.md section 3', '"""wspd_read (E-GAIN-D copy of the banked wspedit_read.py): READ 2 of PREREG_WSP_GAIN_D.md section 3')
rep('Arms: RIWSP{05,10}_s{0,1} (the edited C0).', 'Arms: RIWSPD_s{0,1} (C0 edited toward the data\'s own gains; ONE dose, alpha 1).')
rep('Outcome rules = the card\'s section 4 table made operational in the card\'s section 6 PIN entry (written before any edited-model score):', 'Outcome rules = the card\'s section 4 table (the WSP-GAIN table, unchanged) made operational in the D card\'s section 6 PIN entry (written before any eval-141 window goes through an E-GAIN-D checkpoint):')
rep('usage: wspedit_read.py --out-root', 'usage: wspd_read.py --out-root')
rep('DOSES = {"05": 0.5, "10": 1.0}', 'DOSES = {"D": 1.0}')
rep('res = {"card": "PREREG_WSP_GAIN.md",', 'res = {"card": "PREREG_WSP_GAIN_D.md",')
rep('"disclosure": "Two doses (alpha 0.5, 1.0) and two seeds were read; the card\'s prediction names alpha 1.0; a counted outcome names its dose and says two were read."', '"disclosure": "ONE dose (alpha 1, the data-matched point, fixed by the card before any score) and two seeds were read; no dose was tuned on eval-141."')
rep('jobs = [(f"RIWSP{d}_s{s}", s) for d in DOSES for s in SEEDS]', 'jobs = [(f"RIWSP{d}_s{s}", s) for d in DOSES for s in SEEDS]                                       # RIWSPD_s0, RIWSPD_s1')
rep('f"{a.raw}/READ2.json"', 'f"{a.raw}/READ2_D.json"', 2)
rep('res["outcome_alpha_1p0"] = res["doses"]["10"]["label"]\n    res["outcome_alpha_0p5"] = res["doses"]["05"]["label"]', 'res["outcome_D"] = res["doses"]["D"]["label"]')
rep('f"{a.raw}/RESULTS_READ2.md"', 'f"{a.raw}/RESULTS_READ2_D.md"')
rep('md = ["# READ 2 (WSP-GAIN edit, T1-kin reindexed, eval-141)"', 'md = ["# READ 2 (E-GAIN-D, T1-kin reindexed, eval-141)"')
rep('print("OUTCOME alpha 1.0:", res["outcome_alpha_1p0"], "| alpha 0.5:", res["outcome_alpha_0p5"])', 'print("OUTCOME E-GAIN-D:", res["outcome_D"])')
open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit_d/wspd_read.py", "w", encoding="utf-8", newline="\n").write(out)
print("wrote wspd_read.py", len(out))
