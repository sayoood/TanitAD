"""mutate_gen_check.py -- proves gen_result.py stops on a mismatch: each mutant injects one wrong value after the sources are loaded (or, for the early ones, before the
separation-flag scan) and gen_result.py must exit non-zero with an AssertionError and write nothing (the RESULT's mtime and sha256 are compared before and after).
python -I -X utf8 mutate_gen_check.py"""
import hashlib
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = "E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_result/"
MARK = "# ---------------------------------------------------------------------------------------------------------------- the decision, recomputed (PREREG sections 5 and 6, Amendment 1)"
MARK_EARLY = "EDGE = []        # (contrast, run, label) where a rounded bound is exactly 0.0000"
src = open(HERE + "gen_result.py", encoding="utf-8").read()
assert src.count(MARK) == 1 and src.count(MARK_EARLY) == 1
FLOORK = '"s0 (decisive, beside)"'
MUT = {
    "M1 CTRL s0 ADE6 made a hit (point -0.1, hi -0.01, flagged separated)": 'TAB["egoc-identity"]["CTRL_s0"]["ADE6"].update(delta=-0.1, lo=-0.2, hi=-0.01, separated=True)',
    "M2 reader says PASS": 'READ["OUTCOME"] = "PASS"',
    "M3 reader hit flag for REP s0 false": 'READ["hits"]["REP_s0"] = False',
    "M4 control K-ID:REP_s0 false": 'READ["controls"]["K-ID:REP_s0"] = False',
    "M5 CTRL s1 ADE6 separated-worse (G-ADE would fire)": 'TAB["egoc-identity"]["CTRL_s1"]["ADE6"].update(delta=0.05, lo=0.01, hi=0.09, separated=True)',
    "M6 reader floor-rule s0 verdict COLLAPSE": 'READ["floor_rule_H_V8F_SPEEDSUB_T1_2"][' + FLOORK + ']["verdict"] = "COLLAPSE"',
    "M7 egoshufc ADE6 at CTRL s0 not separated-worse (K-REG)": 'TAB["egoshufc-identity"]["CTRL_s0"]["ADE6"].update(lo=-1.0, separated=False)',
    "M8 VIOL-ANY separated-worse at CTRL s0 and CTRL s1 as well (G-VIOL would fire)": 'TAB["egoc-identity"]["CTRL_s0"]["VIOL-ANY"].update(lo=0.001, separated=True); TAB["egoc-identity"]["CTRL_s1"]["VIOL-ANY"].update(lo=0.001, separated=True)',
    "M9 reader bar 0.0604 instead of 0.06035": 'READ["bar"] = 0.0604',
    "M10 PREREG text lacks the G-VIOL sentence": 'PRE = PRE.replace("G-VIOL", "G-XXXX")',
    "M11 beside ADE6 at CTRL s0 flagged separated": 'BES["egol_minus_identity_l"]["CTRL_s0"]["ADE6"]["separated"] = True',
    "M12 REP s0 ADE6 flag false while its CI excludes 0 (early)": ('TAB["egoc-identity"]["REP_s0"]["ADE6"]["separated"] = False', "early"),
    "M13 accel MAE separated-worse at CTRL s0 and REP s1 as well (G-WORSE would fire)": 'TAB["egoc-identity"]["CTRL_s0"]["accel MAE"].update(delta=0.1, lo=0.05, hi=0.15, separated=True); TAB["egoc-identity"]["REP_s1"]["accel MAE"].update(delta=0.1, lo=0.05, hi=0.15, separated=True)',
    "M14 reader floor g_c ADE6 at s0 changed to 0.1": 'READ["floor_rule_H_V8F_SPEEDSUB_T1_2"][' + FLOORK + ']["g_c"]["ADE6"] = 0.1',
    "M15 chain.log without the read stage line": 'CHAIN = CHAIN.replace("ZZSST1-STAGE-DONE read", "ZZSST1-STAGE-DONE rEad")',
    "M16 reader G-NEW rows non-empty": 'READ["guards"]["G-NEW"]["rows"] = ["M1 crawl share"]',
}
outs = [HERE + "RESULT_V8F_SPEEDSUB_T1.md", HERE + "REGISTER_ROW.md", HERE + "NUMBERS_AUDIT.md"]
h0 = [hashlib.sha256(open(o, "rb").read()).hexdigest() for o in outs]
m0 = [os.path.getmtime(o) for o in outs]
tmp = HERE + "_mut_tmp.py"
ok = 0
try:
    for name, line in MUT.items():
        where = MARK
        if isinstance(line, tuple):
            line, where = line[0], MARK_EARLY
        open(tmp, "w", encoding="utf-8", newline="\n").write(src.replace(where, line + "\n" + where))
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", tmp], capture_output=True, cwd=HERE)
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        last = err[-1] if err else ""
        killed = r.returncode != 0 and "AssertionError" in last
        same = [hashlib.sha256(open(o, "rb").read()).hexdigest() for o in outs] == h0 and [os.path.getmtime(o) for o in outs] == m0
        print(("KILLED " if killed and same else "SURVIVED ") + name + " | rc " + str(r.returncode) + " | " + last[:110].replace("\n", " ") + " | outputs untouched: " + str(same))
        ok += int(killed and same)
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
print(f"{ok} of {len(MUT)} mutants killed")
assert ok == len(MUT)
