"""patch_rhp3.py -- build dscale10k_check_rhp3.py from dscale10k_check_rh.py (sha256 69ff1a0f..., Amendments 5 and 6) by anchored edits. Each anchor must match exactly the stated count, else refuse.
The ONLY change: two new arms RHp3_s0 / RHp3_s1 (kind "RHp3"), which are RH arms whose --dagger-p is exactly 0.3 (the card's PARTIAL row, p = 0.3 on the SAME bank) and which differ from the
template in --dagger-p as well. RH and B0 behave as before (the 40 existing tests must still pass). Usage: patch_rhp3.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest() == "69ff1a0f962123fd74c66cf6a1f3cb0e016799146ca75665c58fcb3e71e8cfc6", "source is not the Amendment-5 checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


# 1. the arms
sub('RH_ARMS = {"RH_s0": ("RH", 0), "RH_s1": ("RH", 1), "B0_s0": ("B0", 0), "B0_s1": ("B0", 1)}',
    'RH_ARMS = {"RH_s0": ("RH", 0), "RH_s1": ("RH", 1), "B0_s0": ("B0", 0), "B0_s1": ("B0", 1), "RHp3_s0": ("RHp3", 0), "RHp3_s1": ("RHp3", 1)}\n'
    '#: ROLLHIST next arm (card 3.4 PARTIAL row, Amendment 7): RHp3 = the RH arm with --dagger-p EXACTLY 0.3 on the SAME bank. RH_LIKE = the kinds that carry a bank and the two declarations.\n'
    'RH_LIKE = ("RH", "RHp3")\n'
    'RHP3_P = "0.3"')
# 2. rh_argv_problems: kind tests and the dagger-p rule
sub('    if kind == "RH":\n        exp_added, exp_removed, own = set(RH_DECL), set(), {"--out", "--init-ckpt"} | dag_bank\n',
    '    if kind in RH_LIKE:\n        exp_added, exp_removed, own = set(RH_DECL), set(), {"--out", "--init-ckpt"} | dag_bank | ({"--dagger-p"} if kind == "RHp3" else set())\n')
sub('    if kind == "RH":\n        for f in ("--dagger-bank", "--dagger-bank-sha256"):',
    '    if kind in RH_LIKE:\n        for f in ("--dagger-bank", "--dagger-bank-sha256"):')
sub('        if a.get("--dagger-p") != t.get("--dagger-p") or not (a.get("--dagger-p") and float(a["--dagger-p"][0]) > 0):\n'
    '            bad.append(f"--dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')} (> 0 required)")\n',
    '        if kind == "RH":\n'
    '            if a.get("--dagger-p") != t.get("--dagger-p") or not (a.get("--dagger-p") and float(a["--dagger-p"][0]) > 0):\n'
    '                bad.append(f"--dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')} (> 0 required)")\n'
    '        elif a.get("--dagger-p") != [RHP3_P]:                                                    # RHp3: the dose is EXACTLY 0.3, a literal, never "the template\'s" and never "any p > 0"\n'
    '            bad.append(f"RHp3 --dagger-p reads {a.get(\'--dagger-p\')}, the card\'s only dose for this arm is [{RHP3_P!r}]")\n')
# 3. required_checks
sub('(REQUIRED_RH_RH if RH_ARMS[arm][0] == "RH" else REQUIRED_RH_B0)', '(REQUIRED_RH_RH if RH_ARMS[arm][0] in RH_LIKE else REQUIRED_RH_B0)')
# 4. the lever read of the selfcheck
sub('    if kind == "RH":\n        if token not in ro:', '    if kind in RH_LIKE:\n        if token not in ro:')
# 5. mint
sub('        if kind == "RH" and not rtok:', '        if kind in RH_LIKE and not rtok:')
sub('p_tok = rtok if kind == "RH" else', 'p_tok = rtok if kind in RH_LIKE else')
sub('rh_t, b0_t = (toks, pair) if kind == "RH" else (pair, toks)', 'rh_t, b0_t = (toks, pair) if kind in RH_LIKE else (pair, toks)')
sub('(rh_levers_problems(sc, "RH", rtok) if kind == "RH" else [])', '(rh_levers_problems(sc, "RH", rtok) if kind in RH_LIKE else [])')
sub('        if kind == "RH":\n            put("C-RH-BANKGEN"', '        if kind in RH_LIKE:\n            put("C-RH-BANKGEN"')
# 6. run
sub('if RH_ARMS[a.arm][0] == "RH" and g("readout_token") != rec:', 'if RH_ARMS[a.arm][0] in RH_LIKE and g("readout_token") != rec:')
# 7. the docstring note
sub('"""PREREG_V8F_DSCALE10K study checker', '"""[ROLLHIST P03 build: dscale10k_check_rhp3.py = dscale10k_check_rh.py 69ff1a0f + the RHp3 kind (patch_rhp3.py); PREREG_V8F_ROLLHIST Amendment 7 DRAFT]\nPREREG_V8F_DSCALE10K study checker')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
