"""MM 2026-11-17: post-sweep fixes for the field-arm gate land (all are tests that enumerate/pin the profile set, plus one mutation-target collision).
Applies the SAME byte edits to (a) the scratch export used for re-running and (b) the hold's land copies (exported raw from the tip for the 4 tests).
Each edit asserts its old text occurs exactly once."""
import os, subprocess, sys

PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
EXP = "F:/tanitad_scratch/fieldgate/exp/" + PKR
LAND = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/gate/land/" + PKR
M = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]

EDITS = {
    "launch_gate.py": [
        ('    out, name, arm = [], prof["name"], prof.get("p7field_arm")\n    if arm not in FIELD_P7_ARM_NAME or FIELD_P7_ARM_NAME[arm] != name:\n        return [f"{name}: p7field_rules without a known arm ({arm!r})"]\n    if not _p5c_hex64(prof.get("init_from_sha256")):\n',
         '    out, name, arm = [], prof["name"], prof.get("p7field_arm")\n    if arm not in FIELD_P7_ARM_NAME or FIELD_P7_ARM_NAME[arm] != name:\n        return [f"{name}: p7field_rules without a known arm ({arm!r})"]\n    if not _p5c_hex64(prof.get("init_from_sha256")):    # p7field: the P7 graft base pin (same check as P5-C; text kept distinct for the P5-C mutation harness)\n'),
        ('    ow = (GT.flag_values(argv, "--out") or [None])[0]\n    if ow and (Path(str(ow)) / "ckpt_last.pt").exists():\n        out.append(f"{name}: --out {ow} already holds ckpt_last.pt: a field arm starts FRESH',
         '    ow = (GT.flag_values(argv, "--out") or [None])[0]\n    if ow and (Path(str(ow)) / "ckpt_last.pt").exists():    # p7field: a fresh --out (same check as P5-C; text kept distinct for the P5-C mutation harness)\n        out.append(f"{name}: --out {ow} already holds ckpt_last.pt: a field arm starts FRESH'),
    ],
    "test_div_gate.py": [
        ('{"refe_div_p6", "refe_vdac_p6", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"} and len(L.PROFILES) == 29          # + refe_vdac_p6 (PREREG_VDAC_P6)',
         '{"refe_div_p6", "refe_vdac_p6", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r", "refe_field_p7", "refe_field_p7_ctrl"} and len(L.PROFILES) == 31          # + refe_vdac_p6 (PREREG_VDAC_P6); + the P7 field arms F/C (PI 2026-11-17)'),
    ],
    "test_dose_gate.py": [
        ('{"refe_dose_s1z_p6", "refe_div_p6", "refe_vdac_p6", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"} and len(L.PROFILES) == 29          # + refe_div_p6 (PREREG_DIV_P6)',
         '{"refe_dose_s1z_p6", "refe_div_p6", "refe_vdac_p6", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r", "refe_field_p7", "refe_field_p7_ctrl"} and len(L.PROFILES) == 31          # + refe_div_p6 (PREREG_DIV_P6); + the P7 field arms F/C (PI 2026-11-17)'),
    ],
    "test_rep_gate.py": [
        ('assert len(L.PROFILES) == 29 and len(set(L.PROFILES) - {"refe_p5c", "refe_p5c_c0", "refe_p5c_c0r", "refe_rep_p6",',
         'assert len(L.PROFILES) == 31 and len(set(L.PROFILES) - {"refe_field_p7", "refe_field_p7_ctrl", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r", "refe_rep_p6",'),
    ],
    "test_p7_integration.py": [
        ('for nm, prof in L.PROFILES.items() if nm != "refe_p7" and not nm.startswith("refe_p5c")),',
         'for nm, prof in L.PROFILES.items() if nm != "refe_p7" and not nm.startswith(("refe_p5c", "refe_field_p7"))),   # P5-C and the field arms are P7 grafts: they carry the table'),
    ],
}

for fn, edits in EDITS.items():
    lp = os.path.join(LAND, fn)
    if not os.path.exists(lp):                                   # raw tip blob (no autocrlf) into the land set
        b = subprocess.run(["git", "-C", M, "cat-file", "blob", f"{TIP}:{PKR}/{fn}"], capture_output=True, check=True).stdout
        open(lp, "wb").write(b)
    for root in (EXP, LAND):
        p = os.path.join(root, fn)
        s = open(p, "rb").read().decode("utf-8")
        for old, new in edits:
            if s.count(new) == 1 and s.count(old) == 0:
                continue                                         # already applied
            assert s.count(old) == 1, (p, s.count(old), old[:80])
            s = s.replace(old, new)
        open(p, "wb").write(s.encode("utf-8"))
    a, b = (open(os.path.join(r, fn), "rb").read() for r in (EXP, LAND))
    print(fn, "export==land" if a == b else "EXPORT!=LAND (expected only if the export predates the tip)")
print("ZZFGPATCH_OK")
