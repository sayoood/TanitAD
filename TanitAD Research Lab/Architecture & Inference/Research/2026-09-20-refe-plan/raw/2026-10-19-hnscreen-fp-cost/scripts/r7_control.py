"""CONTROL for the R7 differential: it must be able to fail.  Two one-line mutants of MY gate (copy of refe/ + eval/rls1b_arms.json under tmp/r7_ctl_*): (A) one word of a content-mode refusal message; (B) content_check run twice
(an extra full read per content fingerprint).  Each is run through r7_identity_vs_tip.py; the differential must exit 1 and print `DIFF` rows.  usage: python r7_control.py <out.json>"""
import json
import os
import shutil
import subprocess
import sys

RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
WORK = f"C:/Users/Admin/par18/hnfp3/work/{RP}"
T = "C:/Users/Admin/par18/hnfp3/tmp"
MUT = {
    "0_unmutated_copy": (None, None, None),                      # control of the control: an UNMUTATED copy must come out identical (rc 0, no DIFF)
    "A_refusal_text": ("refe/launch_gate.py", "is not the registered one (RL-S1 C1 / C0's token fingerprint)", "is not the REGISTERED one (RL-S1 C1 / C0's token fingerprint)"),
    "B_extra_full_read": ("refe/gate_token.py", "        r = content_check(p, _pool)\n    finally:", "        r = content_check(p, _pool)\n        r = content_check(p, _pool)\n    finally:"),
}
out = {}
for name, (rel, old, new) in MUT.items():
    d = f"{T}/r7_ctl_{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(f"{d}/eval")
    shutil.copytree(f"{WORK}/refe", f"{d}/refe", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(f"{WORK}/eval/rls1b_arms.json", f"{d}/eval/rls1b_arms.json")
    if rel:
        p = f"{d}/{rel}"
        s = open(p, encoding="utf-8", newline="").read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    r = subprocess.run([sys.executable, "C:/Users/Admin/par18/hnfp3/scripts/r7_identity_vs_tip.py", f"{d}/res.json"], capture_output=True, text=True, timeout=1500, encoding="utf-8", errors="replace",
                       env=dict(os.environ, R7_MINE=f"{d}/refe", R7_TMP=f"{d}/run", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    ndiff = sum(1 for ln in r.stdout.splitlines() if ln.startswith("DIFF "))
    out[name] = {"rc": r.returncode, "n_DIFF_rows": ndiff, "went_RED": (r.returncode == 0 and ndiff == 0) if rel is None else (r.returncode == 1 and ndiff > 0), "first_diffs": [ln[:160] for ln in r.stdout.splitlines() if ln.startswith("DIFF ")][:4], "tail": ([] if r.returncode in (0, 1) else [ln[:300] for ln in (r.stdout + r.stderr).splitlines()[-12:]])}
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("(went_RED for the unmutated copy means: stayed IDENTICAL, as it must)")
print("ZZR7_CONTROL_PASS" if all(v["went_RED"] for v in out.values()) else "ZZR7_CONTROL_FAIL")
