"""patch_ki.py -- build dscale10k_check_ki.py from dscale10k_check_rk.py (sha256 49a5a116..., the revision-2 RK checker, which already holds RH, B0, RHp3, RHK, RL, RA and RK) by anchored edits. Each anchor must match exactly the stated count, else refuse.
The ONLY change: two new arms KI_s0 / KI_s1 (kind "KI", K-INPUT, PREREG_V8F_KINPUT): the RH bank under bank/, RH's dose 0.1, the RH declarations, plus EXACTLY the two tokens `--dagger-base logged` (appended), and the own --out.
RH, B0, RHp3, RHK, RL, RA and RK behave as before. A second changed flag (a label, a mode, a cache, a kin-tb), another bank, `own` under the KI name is refused. The base pair is dropped from the KI argv before the stock pair check against B0.
Usage: patch_ki.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("49a5a1162980505b"), "source is not the revision-2 RK checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


sub('"RK_s0": ("RK", 0), "RK_s1": ("RK", 1)}', '"RK_s0": ("RK", 0), "RK_s1": ("RK", 1), "KI_s0": ("KI", 0), "KI_s1": ("KI", 1)}')
sub('RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA", "RK")',
    'RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA", "RK", "KI")\n'
    '#: K-INPUT arm KI (PREREG_V8F_KINPUT): RH + exactly `--dagger-base logged` (two tokens, appended), the RH bank under bank/, RH\'s dose 0.1, human label (no label flag), no cache.\n'
    'KI_P = "0.1"\nKI_BASE_FLAG = "--dagger-base"\nKI_BASE = ["logged"]')
sub('({"--dagger-mode"} if kind == "RA" else set()), set(),', '({"--dagger-mode"} if kind == "RA" else set()) | ({KI_BASE_FLAG} if kind == "KI" else set()), set(),')
sub('        elif kind == "RL":                                                                        # RL:',
    '        elif kind == "KI":                                                                        # KI: RH\'s dose, the RH bank, the logged-base lever and nothing else (no label, mode, cache or kin-tb flag: not admitted above)\n'
    '            if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [KI_P]:\n'
    '                bad.append(f"KI --dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')}; the only dose for this arm is [{KI_P!r}]")\n'
    '            bk = _norm_path((a.get("--dagger-bank") or [""])[0]).rstrip("/")\n'
    '            if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n'
    '                bad.append(f"KI --dagger-bank {bk!r} is not a .../bank/s{seed} path: the arm trains on RH\'s bank")\n'
    '            if a.get(KI_BASE_FLAG) != KI_BASE:\n'
    '                bad.append(f"KI {KI_BASE_FLAG} reads {a.get(KI_BASE_FLAG)}, the arm\'s only base is {KI_BASE}")\n'
    '        elif kind == "RL":                                                                        # RL:')
sub('            if kind in ("RL", "RK"):\n                li = ',
    '            if kind == "KI":                                                                      # the base pair is the arm\'s own, as the label pair of RL / RK: dropped before the stock pair check against B0\n'
    '                bi = [str(x) for x in rh_t].index(KI_BASE_FLAG) if KI_BASE_FLAG in [str(x) for x in rh_t] else -1\n'
    '                if bi >= 0:\n'
    '                    rh_t = [str(x) for x in rh_t][:bi] + [str(x) for x in rh_t][bi + 2:]\n'
    '            if kind in ("RL", "RK"):\n                li = ')
sub('"""[REVISION 2: dscale10k_check_rk.py', '"""[K-INPUT build: dscale10k_check_ki.py = dscale10k_check_rk.py 49a5a116 (revision 2) + the KI kind (patch_ki.py); PREREG_V8F_KINPUT]\n[REVISION 2: dscale10k_check_rk.py')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
