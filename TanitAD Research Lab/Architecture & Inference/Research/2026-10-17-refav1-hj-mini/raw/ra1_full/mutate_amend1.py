"""Delete one guard at a time from ra1_full_amend1.py; each mutation must turn test_ra1_full_amend1.py RED. Source restored in `finally`."""
import json, os, subprocess, sys
T, P = sys.argv[1], "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/ra1_full_amend1.py"
TEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_ra1_full_amend1.py"]
src = open(os.path.join(T, P), encoding="utf-8").read()
MUT = {
    "replicate ignores separation": ('rep_ok = bool(d141["delta"] > 0 and d141["separated"])', 'rep_ok = bool(d141["delta"] > 0)'),
    "A2 read on A1's floor only": ("beats_2x_floor=abs(dk[\"delta\"]) > 2 * max(F.values()))", "beats_2x_floor=abs(dk[\"delta\"]) > 2 * F[\"A1\"])"),
    "tie goes to the better arm": ("k_arm = next(arm for arm in ORDER if hold[arm] <= m + TIE_M)", "k_arm = min(hold, key=hold.get)"),
    "unevaluable SENS read as FAIL": ("        if not isinstance(per.get(s), dict):\n            raise", "        if False:\n            raise"),
    "missing run tolerated": ("    if miss:\n        raise", "    if False:\n        raise"),
    "K for A2 read when not picked": ("read=k_arm == \"A2\", PASS=a2_pass if k_arm == \"A2\" else None", "read=True, PASS=a2_pass"),
    "IDM ignores separation": ("idm_pass = all(r[\"below\"] and r[\"separated\"] and r[\"beats_2x_floor\"] for r in idm_rows)",
                               "idm_pass = all(r[\"below\"] and r[\"beats_2x_floor\"] for r in idm_rows)"),
    "IDM floor uses A1 only": ("two_F_idm = 2 * max(F_idm.values())", "two_F_idm = 2 * F_idm[\"A1\"]"),
    "IDM reads one seed": ("idm_pass = all(r[\"below\"] and r[\"separated\"] and r[\"beats_2x_floor\"] for r in idm_rows)",
                           "idm_pass = all(r[\"below\"] and r[\"separated\"] and r[\"beats_2x_floor\"] for r in idm_rows[:1])"),
    "IDM pairs A1 with R0": ("d = rep.paired_runs(runs[(\"A1\", s)], runs[(\"A0\", s)], \"eval141\"",
                             "d = rep.paired_runs(runs[(\"A1\", s)], runs[(\"R0\", s)], \"eval141\""),
}
def run():
    r = subprocess.run([sys.executable] + TEST, cwd=T, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (r.stdout or "").strip().splitlines()
    return r.returncode, (lines[-1] if lines else "")
res = {}
try:
    for name, (old, new) in MUT.items():
        assert src.count(old) == 1, name
        with open(os.path.join(T, P), "w", encoding="utf-8", newline="\n") as f:
            f.write(src.replace(old, new))
        rc, tail = run()
        res[name] = dict(rc=rc, RED=rc != 0, summary=tail)
finally:
    with open(os.path.join(T, P), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
rc, tail = run()
res["_restored"] = dict(rc=rc, summary=tail)
print(json.dumps(res, indent=1))
assert all(v["RED"] for k, v in res.items() if not k.startswith("_")) and res["_restored"]["rc"] == 0
