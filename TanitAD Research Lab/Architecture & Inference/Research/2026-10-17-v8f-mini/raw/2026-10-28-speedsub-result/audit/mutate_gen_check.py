"""mutate_gen_check.py -- proves gen_result.py stops on a mismatch: each mutant injects one wrong value after the sources are loaded and before the decision is recomputed,
and gen_result.py must exit non-zero with an AssertionError and write nothing (the RESULT's mtime and sha256 are compared before and after).
python -I -X utf8 mutate_gen_check.py"""
import hashlib
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = "E:/Projects/TanitAD-artifacts/dagland/speedsub_result/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MARK = "# ---------------------------------------------------------------------------------------------------------------- the decision, recomputed (PREREG section 5)"
src = open(HERE + "gen_result.py", encoding="utf-8").read()
assert src.count(MARK) == 1
MARK_EARLY = 'for s in SEEDS:\n    for m in MODES:\n        assert EJ[(s, m)]["speedread"]["x_sha256"] == XPIN[s][0]'
assert src.count(MARK_EARLY) == 1
MUT = {
    "M1 ego s1 point moved to -0.05 (inside the bar)": 'SUB["rows"]["1"]["ego"]["delta"] = -0.05',
    "M2 control S7 flag false at s0": 'SUB["controls"]["0"]["S7_ego_path_eq_identity_pred"] = False',
    "M3 egoshuf s0 lower end below 0 (regression arm not separated-worse)": 'SUB["rows"]["0"]["egoshuf"]["lo"] = -1.0',
    "M4 ego s1 upper end above 0 (not separated-better)": 'SUB["rows"]["1"]["ego"]["hi"] = 0.01',
    "M5 reader says NULL": 'SUB["outcome"] = "NULL"',
    "M6 re-derivation differs from the reader on ego s0 lo": 'RD["seeds"]["0"]["rows"]["ego"]["lo"] = -0.2',
    "M7 family control LL-2 false at s1": 'LL["1"]["controls"]["LL-2_ade6_equals_READ_json"]["ok"] = False',
    "M8 selection-rule flag S13 false": 'SUB["controls"]["S13_selection_rule_picks_mlp_ego"] = False',
    "M9 a dump's x_sha256 differs from the pin (injected before the pin check)": ('EJ[("0", "ego")]["speedread"]["x_sha256"] = "0" * 64', "early"),
    "M10 F_pass_m differs from F/2": 'SUB["F_pass_m"] = 0.0604',
    "M11 capture differs from ego / gt": 'SUB["rows"]["0"]["capture"] = 0.5',
    "M12 the longlat ADE6 row differs from the reader row": 'LL["0"]["contrasts"]["ego-own"]["ade6_m"]["hi"] = -0.05',
}
rs = HERE + "RESULT_V8F_SPEEDSUB.md"
h0, m0 = hashlib.sha256(open(rs, "rb").read()).hexdigest(), os.path.getmtime(rs)
tmp = HERE + "_mut_tmp.py"
ok = 0
for name, line in MUT.items():
    where = MARK
    if isinstance(line, tuple):
        line, where = line[0], MARK_EARLY
    open(tmp, "w", encoding="utf-8", newline="\n").write(src.replace(where, line + "\n" + where))
    r = subprocess.run([PY, "-I", "-X", "utf8", tmp], capture_output=True, cwd=HERE)
    err = r.stderr.decode("utf-8", "replace").strip().splitlines()
    last = err[-1] if err else ""
    killed = r.returncode != 0 and "AssertionError" in last
    same = hashlib.sha256(open(rs, "rb").read()).hexdigest() == h0 and os.path.getmtime(rs) == m0
    print(("KILLED " if killed and same else "SURVIVED ") + name + " | rc " + str(r.returncode) + " | " + last[:110] + " | outputs untouched: " + str(same))
    ok += int(killed and same)
os.remove(tmp)
print(f"{ok} of {len(MUT)} mutants killed")
assert ok == len(MUT)
