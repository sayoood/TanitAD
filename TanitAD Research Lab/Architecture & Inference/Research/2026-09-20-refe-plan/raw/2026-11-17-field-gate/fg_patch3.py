"""MM 2026-11-18: the after4 sweep found three more profile-SET pins (same class as test_div/dose/rep_gate in fg_patch.py): each enumerates the
profiles added since its pin and now also sees the two P7 field arms.  Same byte edits to the scratch export and the land copies (raw tip blobs).
Each edit asserts its old text occurs exactly once."""
import os, subprocess, sys

PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
EXP = "F:/tanitad_scratch/fieldgate/exp/" + PKR
LAND = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/gate/land/" + PKR
M = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
F2 = ', "refe_field_p7", "refe_field_p7_ctrl"'

EDITS = {
    "test_s1zsm_gate.py": [
        ('"refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"}, set(L.PROFILES) - set(TIP_FP)',
         '"refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"' + F2 + '}, set(L.PROFILES) - set(TIP_FP)'),
    ],
    "test_scorer1p6_gate.py": [
        ('"refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"}, set(L.PROFILES) - set(PINNED_FP)',
         '"refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"' + F2 + '}, set(L.PROFILES) - set(PINNED_FP)'),
    ],
    "test_vdac_gate.py": [
        ('{"refe_vdac_p6", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"} and len(L.PROFILES) == 29',
         '{"refe_vdac_p6", "refe_p5c", "refe_p5c_c0", "refe_p5c_c0r"' + F2 + '} and len(L.PROFILES) == 31'),
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
                continue
            assert s.count(old) == 1, (p, s.count(old), old[:80])
            s = s.replace(old, new)
        open(p, "wb").write(s.encode("utf-8"))
    a, b = (open(os.path.join(r, fn), "rb").read() for r in (EXP, LAND))
    print(fn, "export==land" if a == b else "EXPORT!=LAND")
print("ZZFGPATCH3_OK")
