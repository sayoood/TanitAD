"""patch_read_k.py -- anchored edits to the dev-box read files in F:/TanitAD-scratch/rollhist/read so that RH_AKIND=RHK reads the arm RHK (RHK - B0, beside RHK - RH), exactly as RH_AKIND=RHp3 reads RHp3.
Each anchor must match its stated count, else the script refuses and writes nothing for that file. The default read (RH_AKIND unset) and the RHp3 read are unchanged (dry_check.py proves both against the pre-edit copies).
Backups (*.py.bk9) are made by the caller before this runs. usage: python -I patch_read_k.py READ_DIR"""
import os
import sys

D = sys.argv[1]


def edit(fname, subs):
    p = os.path.join(D, fname)
    s = open(p, encoding="utf-8").read()
    for old, new, n in subs:
        assert s.count(old) == n, (fname, old[:80], s.count(old), n)
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
    print("patched", fname)


edit("rh_common.py", [
    ('assert A_KIND in ("RH", "RHp3"), A_KIND', 'assert A_KIND in ("RH", "RHp3", "RHK"), A_KIND          # RHK (Amendment 8 DRAFT): the arm trained on a bank generated under readout K; the comparator is still the EXISTING B0, RH dumps are reused beside', 1),
    ('KINDS = ("RH", "B0") if A_KIND == "RH" else ("RHp3", "B0")', 'KINDS = ("RH", "B0") if A_KIND == "RH" else (A_KIND, "B0")', 1),
    ('ROLL_KINDS = ("RH", "B0") if A_KIND == "RH" else ("RHp3",)', 'ROLL_KINDS = ("RH", "B0") if A_KIND == "RH" else (A_KIND,)', 1),
    ('if A_KIND == "RHp3":                                                           # RHp3 minus RH',
     'if A_KIND in ("RHp3", "RHK"):                                                  # (RHK: the same beside pairs, RHK minus RH, the readout-K bank against the C* bank at the same dose 0.1)  RHp3 minus RH', 1),
    ('PAIRS["BESIDE_RHp3_minus_RH_" + _f] = (_r("RHp3", _f), _r("RH", _f))', 'PAIRS["BESIDE_" + A_KIND + "_minus_RH_" + _f] = (_r(A_KIND, _f), _r("RH", _f))', 1),
])
edit("rh_roll.py", [
    ('KIND RH|B0|UEA|RHp3,', 'KIND RH|B0|UEA|RHp3|RHK,', 1),
    ('choices=("RH", "B0", "UEA", "RHp3")', 'choices=("RH", "B0", "UEA", "RHp3", "RHK")', 1),
])
edit("rh_read.py", [
    ('res["next_lever"] = (NEXT_P03 if C.A_KIND == "RHp3" else NEXT).get(base)',
     'res["next_lever"] = (NEXT_P03 if C.A_KIND == "RHp3" else NEXT_K if C.A_KIND == "RHK" else NEXT).get(base)', 1),
    ('\n\n# ---------------------------------------------------------------- synthetic controls (K-SELF, K-PLANT, mutation)',
     '''
# RHK (Amendment 8 DRAFT section 5): the same names for RHK (RH's dose p = 0.1, a bank generated under readout K, token causal_g0x).  The rows are the MM's pre-registration, committed before any RHK number; a lever the card does not name
# is never decided here.
NEXT_K = dict(NEXT)
NEXT_K["ROLLHIST-REPAIRS-K-UNNEEDED"] = "to the coordinator and the PI: the integrated candidate with the causal handover and without readout K at inference; register the base as UE.A + ROLLHIST on the readout-K bank"
NEXT_K["ROLLHIST-REPAIRS"] = "the candidate keeps K beside, trained on the K bank; DAgger iteration 2 from RHK's own states (~2.3 pod-h ESTIMATED) only if P4's gap is above floor on two families"
NEXT_K["ROLLHIST-PARTIAL"] = "both banks (C*, K) are spent at p = 0.1 and the card names no lever after them: the MM pre-registers one before any run (candidates, post hoc until frozen: the recovery label H-ST-RECOVERYLABEL-1; a 9,000-step stage B on the K bank at p = 0.1 ~0.8 pod-h ESTIMATED; DAgger iteration 2 from RHK's own states ~2.3 pod-h)"
NEXT_K["ROLLHIST-NO-EFFECT"] = "a 9,000-step stage B on the K bank at p = 0.1 (~0.8 pod-h ESTIMATED from 0.282 s/step); p = 0.3 on the K bank only after that, because RHp3 failed G-T0 at both seeds"
NEXT_K["ROLLHIST-HARMS"] = "own rolled history harms under BOTH bank readouts (C*, K): the recovery label for off-route states (H-ST-RECOVERYLABEL-1); the beside UE.B localisation stays NOT-COMPUTABLE until STAGEB runs"


# ---------------------------------------------------------------- synthetic controls (K-SELF, K-PLANT, mutation)''', 1),
    ('\n\ndef main(', '''

def k_answer(res, rh, rhp3=None):
    """ONE line: does the readout-K bank repair P2 / P3 and hold G-T0?  RHK's P2 / P3 (pooled over the read's seeds) and G-T0 against RH's (C* bank, p = 0.1, the PARTIAL read; both False, G-T0 failed at seed 0), read only, beside."""
    P = res.get("P") or {}
    p2, p3 = (P.get("P2") or {}).get("pass"), (P.get("P3") or {}).get("pass")
    g = P.get("G_T0") or {}
    gword = "G-T0 HOLDS" if g.get("holds") else ("G-T0 FAILS at all seeds" if g.get("fails_at_all_seeds") else "G-T0 FAILS at one seed" if g.get("fails_at_exactly_one_seed") else f"G-T0 reads {g.get('holds')!r}")
    rp = (rh or {}).get("P") or {}
    r2, r3 = (rp.get("P2") or {}).get("pass"), (rp.get("P3") or {}).get("pass")
    rg = ((rp.get("G_T0") or {}).get("per_seed"))
    word = "REPAIRS both P2 and P3" if (p2 and p3) else "repairs P2 but NOT P3" if p2 else "repairs P3 but NOT P2" if p3 else "repairs NEITHER P2 nor P3"
    return (f"K bank {word} and {gword}: RHK P2 {p2}, P3 {p3} (outcome {res.get('outcome')}); RH on the C* bank (the PARTIAL read): P2 {r2}, P3 {r3}, G-T0 per seed {rg}"
            + ("" if rh else " [the RH result file was not found]"))


def main(''', 1),
    ('''    if C.A_KIND == "RHp3":
        res["p03_answer"] = p03_answer(res, J(a.rh_result))
        print("P03:", res["p03_answer"])
''', '''    if C.A_KIND == "RHp3":
        res["p03_answer"] = p03_answer(res, J(a.rh_result))
        print("P03:", res["p03_answer"])
    if C.A_KIND == "RHK":
        res["k_answer"] = k_answer(res, J(a.rh_result))
        print("K:", res["k_answer"])
''', 1),
])
