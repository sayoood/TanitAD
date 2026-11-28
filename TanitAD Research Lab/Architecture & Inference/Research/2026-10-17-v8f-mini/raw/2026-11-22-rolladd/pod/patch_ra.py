"""patch_ra.py -- build dscale10k_check_ra.py from dscale10k_check_rl.py (sha256 6f7e07f5..., the RL checker, which already holds RH, RHp3, RHK and RL) by anchored edits. Each anchor must match exactly the stated count, else refuse.
The ONLY change: two new arms RA_s0 / RA_s1 (kind "RA"): the RH arm (SAME bank under bank/, SAME dose 0.1, SAME declarations) plus EXACTLY the two tokens `--dagger-mode add`, and the own --out.
RH, B0, RHp3, RHK and RL behave as before. A second changed flag, another bank, or `swap` under the RA name is refused. Usage: patch_ra.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("6f7e07f59caf5c9e"), "source is not the RL checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


sub('"RL_s0": ("RL", 0), "RL_s1": ("RL", 1)}', '"RL_s0": ("RL", 0), "RL_s1": ("RL", 1), "RA_s0": ("RA", 0), "RA_s1": ("RA", 1)}')
sub('RH_LIKE = ("RH", "RHp3", "RHK", "RL")',
    'RH_LIKE = ("RH", "RHp3", "RHK", "RL", "RA")\n'
    '#: ROLLHIST arm RA (PREREG_V8F_ROLLADD): RH + exactly `--dagger-mode add` (two tokens), the RH bank under bank/, RH\'s dose 0.1.\n'
    'RA_P = "0.1"\nRA_MODE = ["add"]')
sub('({"--dagger-label"} if kind == "RL" else set())', '({"--dagger-label"} if kind == "RL" else set()) | ({"--dagger-mode"} if kind == "RA" else set())')
sub('        elif kind == "RL":                                                                        # RL:',
    '        elif kind == "RA":                                                                        # RA: RH\'s dose, the RH bank, the add-mode lever and nothing else\n'
    '            if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RA_P]:\n'
    '                bad.append(f"RA --dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')}; the only dose for this arm is [{RA_P!r}]")\n'
    '            bk = _norm_path((a.get("--dagger-bank") or [""])[0]).rstrip("/")\n'
    '            if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n'
    '                bad.append(f"RA --dagger-bank {bk!r} is not a .../bank/s{seed} path: the arm trains on RH\'s bank")\n'
    '            if a.get("--dagger-mode") != RA_MODE:\n'
    '                bad.append(f"RA --dagger-mode reads {a.get(\'--dagger-mode\')}, the arm\'s only mode is {RA_MODE}")\n'
    '        elif kind == "RL":                                                                        # RL:')
sub('            if kind == "RL":\n                li = ',
    '            if kind == "RA":\n'
    '                mi = [str(x) for x in rh_t].index("--dagger-mode") if "--dagger-mode" in [str(x) for x in rh_t] else -1\n'
    '                if mi >= 0:\n'
    '                    rh_t = [str(x) for x in rh_t][:mi] + [str(x) for x in rh_t][mi + 2:]\n'
    '            if kind == "RL":\n                li = ')
sub('"""[ROLLHIST RL build:', '"""[ROLLHIST RA build: dscale10k_check_ra.py = dscale10k_check_rl.py 6f7e07f5 + the RA kind (patch_ra.py); PREREG_V8F_ROLLADD]\n[ROLLHIST RL build:')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
