"""patch_rl.py -- build dscale10k_check_rl.py from dscale10k_check_rhk.py (sha256 1bd28255..., the RHK checker, which already holds RH, RHp3 and RHK) by anchored edits. Each anchor must match exactly the stated count, else refuse.
The ONLY change: two new arms RL_s0 / RL_s1 (kind "RL"): the RH arm (SAME bank under bank/, SAME dose 0.1, SAME declarations) plus EXACTLY the two tokens `--dagger-label speed_v1`, and the own --out.
RH, B0, RHp3 and RHK behave as before. A second changed flag, another bank, or `human` under the RL name is refused. Usage: patch_rl.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest().startswith("1bd2825538461697"), "source is not the RHK checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


# 1. the arms and the kind
sub('"RHK_s0": ("RHK", 0), "RHK_s1": ("RHK", 1)}', '"RHK_s0": ("RHK", 0), "RHK_s1": ("RHK", 1), "RL_s0": ("RL", 0), "RL_s1": ("RL", 1)}')
sub('RH_LIKE = ("RH", "RHp3", "RHK")',
    'RH_LIKE = ("RH", "RHp3", "RHK", "RL")\n'
    '#: ROLLHIST arm RL (the label lever, Amendment 9 DRAFT): RH + exactly `--dagger-label speed_v1` (two tokens), the RH bank under bank/, RH\'s dose 0.1.\n'
    'RL_P = "0.1"\nRL_LABEL = ["speed_v1"]')
# 2. rh_argv_problems: the allowed additions / own flags
sub('        exp_added, exp_removed, own = set(RH_DECL), set(), {"--out", "--init-ckpt"} | dag_bank | ({"--dagger-p"} if kind == "RHp3" else set())\n',
    '        exp_added, exp_removed, own = set(RH_DECL) | ({"--dagger-label"} if kind == "RL" else set()), set(), {"--out", "--init-ckpt"} | dag_bank | ({"--dagger-p"} if kind == "RHp3" else set())\n')
# 3. the RL branch of the dose / bank / label rule (before the RHp3 fall-through)
sub('        elif a.get("--dagger-p") != [RHP3_P]:                                                    # RHp3:',
    '        elif kind == "RL":                                                                        # RL: RH\'s dose, the RH bank, the label lever and nothing else\n'
    '            if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RL_P]:\n'
    '                bad.append(f"RL --dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')}; the only dose for this arm is [{RL_P!r}]")\n'
    '            bk = _norm_path((a.get("--dagger-bank") or [""])[0]).rstrip("/")\n'
    '            if not bk.endswith(f"/bank/s{seed}") or "/bank_k/" in bk:\n'
    '                bad.append(f"RL --dagger-bank {bk!r} is not a .../bank/s{seed} path: the arm trains on RH\'s bank")\n'
    '            if a.get("--dagger-label") != RL_LABEL:\n'
    '                bad.append(f"RL --dagger-label reads {a.get(\'--dagger-label\')}, the arm\'s only label is {RL_LABEL}")\n'
    '        elif a.get("--dagger-p") != [RHP3_P]:                                                    # RHp3:')
# 4. the stock pair check pins RH's argv exactly: it never sees the label flag (the added tokens are checked above)
sub('            rh_t, b0_t = (toks, pair) if kind in RH_LIKE else (pair, toks)\n',
    '            rh_t, b0_t = (toks, pair) if kind in RH_LIKE else (pair, toks)\n'
    '            if kind == "RL":\n'
    '                li = [str(x) for x in rh_t].index("--dagger-label") if "--dagger-label" in [str(x) for x in rh_t] else -1\n'
    '                if li >= 0:\n'
    '                    rh_t = [str(x) for x in rh_t][:li] + [str(x) for x in rh_t][li + 2:]\n')
# 5. the docstring note
sub('"""[ROLLHIST RHK build:', '"""[ROLLHIST RL build: dscale10k_check_rl.py = dscale10k_check_rhk.py 1bd28255 + the RL kind (patch_rl.py); PREREG_V8F_ROLLHIST Amendment 9 DRAFT]\n[ROLLHIST RHK build:')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
