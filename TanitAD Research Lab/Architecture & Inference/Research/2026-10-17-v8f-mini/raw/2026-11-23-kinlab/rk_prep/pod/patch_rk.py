"""patch_rk.py -- build dscale10k_check_rk.py from dscale10k_check_ra.py (sha256 ba5dab8e..., the RA checker, which already holds RH, B0, RHp3, RHK, RL and RA) by anchored edits. Each anchor must match exactly the stated count, else refuse.
The ONLY change: two new arms RK_s0 / RK_s1 (kind "RK", KINLAB, PREREG_V8F_KINLAB): the RH bank under bank/, RH's dose 0.1, the RH declarations, plus EXACTLY the two tokens `--dagger-label kin_v1` (the RL arm's label token with another value), and the own --out.
RH, B0, RHp3, RHK, RL and RA behave as before. A second changed flag, another bank, `speed_v1` / `human` under the RK name is refused. Usage: patch_rk.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("ba5dab8ea3a6cce7"), "source is not the RA checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


sub('"RA_s0": ("RA", 0), "RA_s1": ("RA", 1)}', '"RA_s0": ("RA", 0), "RA_s1": ("RA", 1), "RK_s0": ("RK", 0), "RK_s1": ("RK", 1)}')
sub('RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA")',
    'RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA", "RK")\n'
    '#: ROLLHIST/KINLAB arm RK (PREREG_V8F_KINLAB): RH + exactly `--dagger-label kin_v1` (two tokens), the RH bank under bank/, RH\'s dose 0.1.\n'
    'RK_P = "0.1"\nRK_LABEL = ["kin_v1"]')
sub('({"--dagger-label"} if kind == "RL" else set())', '({"--dagger-label"} if kind in ("RL", "RK") else set())')
sub('        elif kind == "RL":                                                                        # RL:',
    '        elif kind == "RK":                                                                        # RK: RH\'s dose, the RH bank, the kinematic label lever and nothing else\n'
    '            if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RK_P]:\n'
    '                bad.append(f"RK --dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')}; the only dose for this arm is [{RK_P!r}]")\n'
    '            bk = _norm_path((a.get("--dagger-bank") or [""])[0]).rstrip("/")\n'
    '            if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n'
    '                bad.append(f"RK --dagger-bank {bk!r} is not a .../bank/s{seed} path: the arm trains on RH\'s bank")\n'
    '            if a.get("--dagger-label") != RK_LABEL:\n'
    '                bad.append(f"RK --dagger-label reads {a.get(\'--dagger-label\')}, the arm\'s only label is {RK_LABEL}")\n'
    '        elif kind == "RL":                                                                        # RL:')
sub('            if kind == "RL":\n                li = ', '            if kind in ("RL", "RK"):\n                li = ')
sub('"""[ROLLHIST RA build:', '"""[ROLLHIST/KINLAB RK build: dscale10k_check_rk.py = dscale10k_check_ra.py ba5dab8e + the RK kind (patch_rk.py); PREREG_V8F_KINLAB]\n[ROLLHIST RA build:')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
