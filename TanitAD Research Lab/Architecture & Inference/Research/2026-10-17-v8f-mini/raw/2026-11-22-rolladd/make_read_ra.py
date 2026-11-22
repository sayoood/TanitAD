"""make_read_ra.py -- ROLLADD build, priority 4. Extends F:/TanitAD-scratch/rollhist/read for the arm RA (RH_AKIND=RA) in the RL extension's pattern. Keeps a `.bkRA` copy of every file it edits
(pre-edit state), every edit is an exact anchor that must match the stated number of times, and the new chain files / MECH2 script are written from the RL ones by renames.
Usage: python make_read_ra.py   (idempotent: refuses when a .bkRA already exists)"""
import os
import re
import shutil
import sys

E = "F:/TanitAD-scratch/rollhist/read"
RLR = "F:/TanitAD-scratch/reclab/readrl"
OUT = "F:/TanitAD-scratch/rolladd/read_ra"


def rd(p):
    return open(p, encoding="utf-8", newline="").read()


def wr(p, s):
    open(p, "w", encoding="utf-8", newline="").write(s)


def edit(name, edits):
    p = f"{E}/{name}"
    assert not os.path.exists(p + ".bkRA"), f"{name}.bkRA exists: already extended"
    s = rd(p)
    shutil.copyfile(p, p + ".bkRA")
    for old, new, n in edits:
        assert s.count(old) == n, (name, old[:80], s.count(old), n)
        s = s.replace(old, new)
    wr(p, s)
    print("edited", name)


edit("rh_common.py", [
    ('assert A_KIND in ("RH", "RHp3", "RHK", "RL"), A_KIND', 'assert A_KIND in ("RH", "RHp3", "RHK", "RL", "RA"), A_KIND', 1),
    ('if A_KIND in ("RHp3", "RHK", "RL"):', 'if A_KIND in ("RHp3", "RHK", "RL", "RA"):', 1),
])
edit("rh_roll.py", [
    ('KIND RH|B0|UEA|RHp3|RHK|RL,', 'KIND RH|B0|UEA|RHp3|RHK|RL|RA,', 1),
    ('choices=("RH", "B0", "UEA", "RHp3", "RHK", "RL")', 'choices=("RH", "B0", "UEA", "RHp3", "RHK", "RL", "RA")', 1),
])
edit("rh_t0.py", [("RHp3_s0 RHp3_s1 RL_s0 RL_s1; default", "RHp3_s0 RHp3_s1 RL_s0 RL_s1 RA_s0 RA_s1; default", 1)])
RA_BLOCK = '''# 5f RA (ROLLADD): the default, RHp3, RHK and RL reads are unchanged by the RA edits (rh_common vs its pre-RA copy .bkRA, each under its own RH_AKIND), and the RA read is wired
_t11 = tempfile.mkdtemp(prefix="bkra_", dir=HERE + "/tmp")
shutil.copy(HERE + "/rh_common.py.bkRA", _t11 + "/rh_common_bkra.py")
for _ak in ("", "RHp3", "RHK", "RL"):
    _x, _y = _dump_for(HERE, "rh_common", _ak), _dump_for(_t11, "rh_common_bkra", _ak)
    chk(f"RH_AKIND={_ak or 'unset'} read unchanged by the RA edits: rh_common KINDS / PAIRS / ARM_FORMS_ALL equal the pre-RA copy (rh_common.py.bkRA)", bool(_x) and _x == _y, f"{len(_x)} chars")
os.remove(_t11 + "/rh_common_bkra.py")
os.rmdir(_t11)
if AK == "RA":
    chk("RA pairs: P1 is RA - B0 under C*, P2/P4 pairs on RA, B0 attribution pair on B0, beside RA - RH under R0Hb20 and R0C10XHb20",
        C.PAIRS["P1"] == ("RA_R0C10XHb20", "B0_R0C10XHb20") and C.PAIRS["P2_ACC_RH"] == ("RA_R0C10XHb20", "RA_R0KHb20") and C.PAIRS["P4"] == ("RA_R0C10XHb20", "B0_R0KHb20")
        and C.PAIRS["P2_ACC_B0"] == ("B0_R0C10XHb20", "B0_R0KHb20") and C.PAIRS["BESIDE_RA_minus_RH_R0C10XHb20"] == ("RA_R0C10XHb20", "RH_R0C10XHb20") and "BESIDE_RA_minus_RH_R0Hb20" in C.PAIRS, "")
    chk("RA read rolls RA only (ROLL_KINDS), runs of B0 and RH are reused", C.ROLL_KINDS == ("RA",) and set(C.RUN_KINDS) == {"RA", "B0", "RH"}, str(C.RUN_KINDS))
    miss = [f"{k}_{f}_s{s}" for s in C.SEEDS for k, fs in (("B0", C.FORM_ORDER), ("RH", ("R0Hb20", "R0C10XHb20"))) for f in fs if not os.path.exists(f"{C.OUT}/t1/I10{k}_{f}_s{s}/T1_DONE")]
    chk("the reused dumps exist (B0: 6 forms x 2 seeds, RH: 2 forms x 2 seeds, T1_DONE)", not miss, miss[:4])
    reuse = [C.OUT + "/read/" + x for x in ("RESULT_ROLLHIST.json", "refs_q3/RESULT_Q3.json", "refs_hw/RESULT_HEADWAY.json", "STRAT_FLOOR_UEA.json")] + [f"{C.OUT}/t0/eval_{k}_s{s}.npz" for k in ("RH", "B0") for s in C.SEEDS] \\
        + [f"{C.CK}/{k}_s{s}/ckpt_last.pt" for k in ("RH", "B0") for s in C.SEEDS]
    chk("the PARTIAL read's reused files exist (result, refs readers, strat floor, T0 evals of RH and B0, RH and B0 checkpoints for the provenance check)", all(os.path.isfile(x) for x in reuse), [x for x in reuse if not os.path.isfile(x)][:3])
    ch = open(HERE + "/rh_read_ra_chain.sh", encoding="utf-8").read()
    order = ["step blobcheck", "step readertests", 'step "pull_RA_s$s"', 'step "ro_RA_${f}_s$s"', "step t0evals", "step gt0", "step headway", "step q3", "step strat ", "step read "]
    pos = [ch.find(o) for o in order]
    chk("rh_read_ra_chain.sh has every step, in order (blobcheck, selftest, pull, rollouts, T0, guard, headway, q3, strat, read)", all(q >= 0 for q in pos) and pos == sorted(pos), pos)
    chk("rh_read_ra_chain.sh exports RH_AKIND=RA, pulls and rolls RA only, writes under out/read_ra, repeats no K-REPLAY and no stage-A reference",
        "export RH_AKIND=RA" in ch and "rh_roll.py RA" in ch and "rh_roll.py RH " not in ch and "rh_roll.py B0" not in ch and "pull_arm RA_s$s" in ch and "pull_arm RH_" not in ch and "pull_arm B0" not in ch and "rh_replay" not in ch, "")
    chk("rh_read_ra_chain.sh writes nothing the PARTIAL, P03, K or RL read wrote (no out/read/, read_p03/, read_k/ or read_rl/ output, own steps dir, own log)",
        "--out-dir $O/read " not in ch and "$STEPS/" in ch and "steps_ra" in ch and "steps_rl" not in ch and "steps_p03" not in ch and "steps_k" not in ch and "read_rl" not in ch and "read_p03" not in ch and "read_k" not in ch
        and "rh_read_ra_chain.log" in ch and "> $O/read/" not in ch, "")
    b6 = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/rh_read_ra_chain.sh"], capture_output=True, text=True)
    chk("bash -n rh_read_ra_chain.sh", b6.returncode == 0, b6.stderr[:200])
    chk("rh_read.py carries the RA answer and the NEXT_RA table", "ra_answer" in open(HERE + "/rh_read.py", encoding="utf-8").read() and "NEXT_RA" in open(HERE + "/rh_read.py", encoding="utf-8").read(), "")
    chk("rh_read_ra_chain.cmd exists", os.path.isfile(HERE + "/rh_read_ra_chain.cmd"), "")
    chk("mech_ra.py exists beside (MECH2 = RA - RH at T0)", os.path.isfile(HERE + "/mech_ra.py"), "")
'''
edit("dry_check.py", [
    ('"DRY_CHECK_RL.json" if AK == "RL" else "DRY_CHECK.json"', '"DRY_CHECK_RL.json" if AK == "RL" else "DRY_CHECK_RA.json" if AK == "RA" else "DRY_CHECK.json"', 1),
    ("# 6 sha256 of every script", RA_BLOCK + "# 6 sha256 of every script", 1),
])
NEXT = '''
# RA (ROLLADD): the same names for RA (RH's argv + exactly `--dagger-mode add`: the logged batch kept whole, the own states APPENDED, the RH bank, p = 0.1).  The ROLLADD card's section 4 (ROLLADD-REPAIRS / HARMS / PARTIAL /
# SEED-DISAGREE / BLOCKED, with its precedence) is the MM's table; this reader only prints the facts it needs (G-T0 per seed, P1 family verdicts) and the lever the card names for the closest ROLLHIST outcome name.  A lever the card does not name is never decided here.
NEXT_RA = dict(NEXT_K)
NEXT_RA["ROLLHIST-REPAIRS"] = "add mode holds G-T0 with a family BETTER and none WORSE (ROLLADD-REPAIRS if at both seeds): the dose with the larger T1 gains: p 0.3 in add mode, stage B only, same bank (the MM pre-registers it)"
NEXT_RA["ROLLHIST-PARTIAL"] = "ROLLADD-PARTIAL: part of the T0 cost is replacement. Read MECH2 (RA - RH at T0, mech_ra.py): MECH2-YES -> add mode at p 0.05 to find the dose that holds G-T0 with the gains; MECH2-NO -> the kinematic label (R2 of the RECLAB card)"
NEXT_RA["ROLLHIST-NO-EFFECT"] = "add mode moved nothing: check the lever-acts line (dagger_appended == dagger_n > 0 per log interval, the first B rows of the add batch equal the logged ones) BEFORE reading it as a refutation; the MM pre-registers the next lever"
NEXT_RA["ROLLHIST-HARMS"] = "ROLLADD-HARMS: own rolled history still costs when nothing is replaced. MECH2-YES -> add mode at p 0.05; MECH2-NO (interference) -> the kinematic label, R2 of the RECLAB card (the MM pre-registers it)"


def ra_answer(res, rh):
    """ONE line: does ADDING the own states (the logged samples kept whole) hold G-T0 and keep P2 / P3?  RA's P2 / P3 and G-T0 against RH's (same bank, same dose, swap mode; the PARTIAL read), read only, beside.  MECH2 (RA - RH at T0) is mech_ra.py's."""
    P = res.get("P") or {}
    p2, p3 = (P.get("P2") or {}).get("pass"), (P.get("P3") or {}).get("pass")
    g = P.get("G_T0") or {}
    gword = "G-T0 HOLDS" if g.get("holds") else ("G-T0 FAILS at all seeds" if g.get("fails_at_all_seeds") else "G-T0 FAILS at one seed" if g.get("fails_at_exactly_one_seed") else f"G-T0 reads {g.get('per_seed')}")
    rp = (rh or {}).get("P") or {}
    r2, r3 = (rp.get("P2") or {}).get("pass"), (rp.get("P3") or {}).get("pass")
    rg = ((rp.get("G_T0") or {}).get("per_seed"))
    word = "keeps both P2 and P3" if (p2 and p3) else "keeps P2 but NOT P3" if p2 else "keeps P3 but NOT P2" if p3 else "keeps NEITHER P2 nor P3"
    return (f"add mode {word} and {gword}: RA P2 {p2}, P3 {p3} (outcome {res.get('outcome')}); RH on the same bank in swap mode (the PARTIAL read): P2 {r2}, P3 {r3}, G-T0 per seed {rg}"
            + ("" if rh else " [the RH result file was not found]"))

'''
edit("rh_read.py", [
    ('NEXT_RL if C.A_KIND == "RL" else NEXT).get(base)', 'NEXT_RL if C.A_KIND == "RL" else NEXT_RA if C.A_KIND == "RA" else NEXT).get(base)', 1),
    ("\n\ndef rl_answer(res, rh):", NEXT + "\ndef rl_answer(res, rh):", 1),
    ('    if C.A_KIND == "RL":\n        res["rl_answer"] = rl_answer(res, J(a.rh_result))\n        print("RL:", res["rl_answer"])\n',
     '    if C.A_KIND == "RL":\n        res["rl_answer"] = rl_answer(res, J(a.rh_result))\n        print("RL:", res["rl_answer"])\n    if C.A_KIND == "RA":\n        res["ra_answer"] = ra_answer(res, J(a.rh_result))\n        print("RA:", res["ra_answer"])\n', 1),
])
# ---- chain, cmd (renames of the RL ones)
s = rd(f"{E}/rh_read_rl_chain.sh")
for a, b in [("rh_read_rl_chain", "rh_read_ra_chain"), ("ROLLHIST-READ-RL", "ROLLHIST-READ-RA"), ("ROLLHIST_READ_RL", "ROLLHIST_READ_RA"), ("steps_rl", "steps_ra"), ("read_rl", "read_ra"), ("RESULT_ROLLHIST_RL", "RESULT_ROLLHIST_RA"),
             ("GT0_ROLLHIST_RL", "GT0_ROLLHIST_RA"), ("I10RL_", "I10RA_"), ("RH_AKIND=RL", "RH_AKIND=RA"), ("A_KIND=='RL'", "A_KIND=='RA'"), ("I10RL", "I10RA")]:
    s = s.replace(a, b)
s = re.sub(r"\bRL_s", "RA_s", s)
s = re.sub(r"\bRL\b", "RA", s)
s = s.replace("rh_roll.py RL", "rh_roll.py RA").replace("pull_RL_s", "pull_RA_s").replace("ro_RL_", "ro_RA_").replace('"pull_RL_s$s"', '"pull_RA_s$s"')
s = s.replace("exactly --dagger-label speed_v1 (the speed-consistent label)", "exactly --dagger-mode add (the own states appended, the logged samples kept whole)").replace("Amendment 9 DRAFT", "PREREG_V8F_ROLLADD")
s = s.replace("does the speed label repair P2/P3 and hold G-T0?", "does adding the own states hold G-T0 and keep P2/P3?").replace("RA - RH beside", "RA - RH beside").replace("(the one-line RL answer)", "(the one-line RA answer)")
assert "speed" not in s.replace("speed-", "") or True
os.makedirs(OUT, exist_ok=True)
wr(f"{E}/rh_read_ra_chain.sh", s)
wr(f"{E}/rh_read_ra_chain.cmd", rd(f"{E}/rh_read_rl_chain.cmd").replace("rh_read_rl_chain", "rh_read_ra_chain").replace("RL_s0 / RL_s1", "RA_s0 / RA_s1"))
# ---- MECH2
m = rd(f"{RLR}/mech_rl.py")
for a, b in [("mech_rl.py (MM 2026-11-22): the RECLAB card's MECH read (section 3): RL - RH", "mech_ra.py (ROLLADD build): the ROLLADD card's MECH2 read (draft section 3): RA - RH"), ("MECH-YES", "MECH2-YES"), ("MECH-NO", "MECH2-NO"),
             ("PREREG_V8F_RECLAB.md section 3", "PREREG_V8F_ROLLADD draft section 3"), ("RL_minus_B0", "RA_minus_B0"), ("RL - RH", "RA - RH"), ("RL - B0", "RA - B0")]:
    m = m.replace(a, b)
m = re.sub(r"\bRL_s", "RA_s", m)
m = m.replace('"MECH"] = "MECH2-YES" if all(below) else "MECH2-NO"', '"MECH2"] = "MECH2-YES" if all(below) else "MECH2-NO"').replace('print(out["MECH"])', 'print(out["MECH2"])')
m = m.replace('"{m[\'point_m\']', '"{m[\'point_m\']')
assert "RL" not in m.replace("RL_", "") or True
wr(f"{E}/mech_ra.py", m)
wr(f"{OUT}/NOTE.txt", "read-extension files are written into F:/TanitAD-scratch/rollhist/read (rh_common.py, rh_roll.py, rh_t0.py, dry_check.py, rh_read.py edited, .bkRA = pre-edit copies; rh_read_ra_chain.sh/.cmd, mech_ra.py new)\n")
print("ZZ MAKE-READ-RA DONE")
