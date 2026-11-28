"""patch_rhk.py -- build dscale10k_check_rhk.py from dscale10k_check_rhp3.py (sha256 dcee7cc1..., Amendment 7) by anchored edits; each anchor must match exactly the stated count, else refuse.
The ONLY change: two new arms RHK_s0 / RHK_s1 (kind "RHK", PREREG_V8F_ROLLHIST Amendment 8 DRAFT): an RH-like arm at EXACTLY --dagger-p 0.1 (RH's dose, the template's), whose bank is generated under readout K
(token causal_g0x = --readout causal_g --acc-gain 0 --path-exec, heading bound 0.2) and lives at <root>/bank_k/s<seed>. RH, RHp3 and B0 behave as before (every old test must still pass).
Usage: patch_rhk.py SRC DST"""
import hashlib
import sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
assert hashlib.sha256(s.encode("utf-8")).hexdigest() == "dcee7cc1bf74848594d1e37e4b4bf87b91d8f0cb2588ae8c5c047ccfb86fdb9e", "source is not the Amendment-7 checker"


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old), n)
    s = s.replace(old, new)


# 1. the arms, the kind, the constants
sub('"RHp3_s0": ("RHp3", 0), "RHp3_s1": ("RHp3", 1)}\n', '"RHp3_s0": ("RHp3", 0), "RHp3_s1": ("RHp3", 1), "RHK_s0": ("RHK", 0), "RHK_s1": ("RHK", 1)}\n')
sub('RH_LIKE = ("RH", "RHp3")\nRHP3_P = "0.3"\n',
    'RH_LIKE = ("RH", "RHp3", "RHK")\nRHP3_P = "0.3"\n'
    '#: ROLLHIST arm RHK (Amendment 8 DRAFT): RH\'s dose and flags, a bank generated under readout K (the eval name R0KHb20: --readout causal_g --acc-gain 0 --path-exec, kappa 0.2), kept at <root>/bank_k/s<seed>.\n'
    'RHK_P = "0.1"\nRHK_TOKENS = ("causal_g0x",)\nRHK_BANK_DIR = "bank_k"\n')
# 2. rh_argv_problems: the token set, the dose, the bank path
sub('        if tk not in RH_TOKENS:\n            bad.append(f"--dagger-bank-readout {tk!r} is not one of {RH_TOKENS}")\n',
    '        if tk not in (RHK_TOKENS if kind == "RHK" else RH_TOKENS):\n            bad.append(f"--dagger-bank-readout {tk!r} is not one of {RHK_TOKENS if kind == \'RHK\' else RH_TOKENS}")\n')
sub('        elif a.get("--dagger-p") != [RHP3_P]:',
    '        elif kind == "RHK":                                                                      # RHK: RH\'s dose, the template\'s, AND the literal 0.1 (never "any p > 0")\n'
    '            if a.get("--dagger-p") != t.get("--dagger-p") or a.get("--dagger-p") != [RHK_P]:\n'
    '                bad.append(f"RHK --dagger-p reads {a.get(\'--dagger-p\')}, the template\'s {t.get(\'--dagger-p\')}; the only dose for this arm is [{RHK_P!r}]")\n'
    '            bk = _norm_path((a.get("--dagger-bank") or [""])[0]).rstrip("/")\n'
    '            if not bk.endswith(f"/{RHK_BANK_DIR}/s{seed}"):\n'
    '                bad.append(f"RHK --dagger-bank {bk!r} is not a .../{RHK_BANK_DIR}/s{seed} path: the arm trains on the readout-K bank, never on the RH bank (bank/)")\n'
    '        elif a.get("--dagger-p") != [RHP3_P]:')
# 3. the stock pair check hard-codes causal_g1x: for the RHK kind (and only it) the declared K token is swapped for the stock spelling in a COPY of the RH argv, so every other stock check runs unchanged
sub('def rh_pair_problems(rh_toks, b0_toks, token: str, bank_sha: str | None, pair_check_path: str) -> list:',
    'def rh_pair_problems(rh_toks, b0_toks, token: str, bank_sha: str | None, pair_check_path: str, swap_k: bool = False) -> list:')
sub('    if token != RH_PAIR_CHECK_HARDCODED:\n        return [f"readout {token!r}: stack/scripts',
    '    if swap_k:                                                                      # RHK: the pin of the K token is THIS checker\'s (C-ARGV, C-RH-BANKGEN); the stock file never sees a different token and is not patched\n'
    '        if token not in RHK_TOKENS:\n'
    '            return [f"RHK readout token {token!r} is not one of {RHK_TOKENS}"]\n'
    '        rh_toks = [str(t) for t in rh_toks]\n'
    '        if "--dagger-bank-readout" not in rh_toks or rh_toks[rh_toks.index("--dagger-bank-readout") + 1] != token:\n'
    '            return [f"the RHK argv does not declare --dagger-bank-readout {token!r}"]\n'
    '        rh_toks[rh_toks.index("--dagger-bank-readout") + 1] = RH_PAIR_CHECK_HARDCODED\n'
    '        token = RH_PAIR_CHECK_HARDCODED\n'
    '    if token != RH_PAIR_CHECK_HARDCODED:\n        return [f"readout {token!r}: stack/scripts')
sub('rh_pair_problems(rh_t, b0_t, p_tok, bsha, pc)', 'rh_pair_problems(rh_t, b0_t, p_tok, bsha, pc, kind == "RHK")')
# 4. docstring note
sub('"""[ROLLHIST P03 build:', '"""[ROLLHIST RHK build: dscale10k_check_rhk.py = dscale10k_check_rhp3.py dcee7cc1 + the RHK kind (patch_rhk.py); PREREG_V8F_ROLLHIST Amendment 8 DRAFT]\n[ROLLHIST P03 build:')
open(DST, "w", encoding="utf-8", newline="").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
