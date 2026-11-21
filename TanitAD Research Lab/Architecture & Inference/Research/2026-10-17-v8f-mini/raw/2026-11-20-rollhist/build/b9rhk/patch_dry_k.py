"""patch_dry_k.py -- anchored edits to read/dry_check.py so that RH_AKIND=RHK checks the read of the arm RHK and the default and RHp3 reads are proved unchanged by the RHK edits. usage: python -I patch_dry_k.py READ_DIR"""
import os
import sys

p = os.path.join(sys.argv[1], "dry_check.py")
s = open(p, encoding="utf-8").read()


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:80], s.count(old), n)
    s = s.replace(old, new)


sub('OUTJ = "DRY_CHECK_P03.json" if AK == "RHp3" else "DRY_CHECK.json"', 'OUTJ = "DRY_CHECK_P03.json" if AK == "RHp3" else "DRY_CHECK_K.json" if AK == "RHK" else "DRY_CHECK.json"')
anchor = '    chk("rh_read_p03_chain.cmd exists", os.path.isfile(HERE + "/rh_read_p03_chain.cmd"), "")\n'
block = anchor + '''# 5d RHK (Amendment 8 DRAFT): the default read and the RHp3 read are unchanged by the RHK edits (rh_common vs its pre-RHK copy .bk9, each under its own RH_AKIND), and the RHK read is wired
def _dump_for(mod_dir, mod, akind):
    env = {k: v for k, v in os.environ.items() if k != "RH_AKIND"}
    if akind:
        env["RH_AKIND"] = akind
    return subprocess.run([C.PY, "-c", _dump % (mod_dir, mod)], capture_output=True, text=True, encoding="utf-8", env=env).stdout


_t9 = tempfile.mkdtemp(prefix="bk9_", dir=HERE + "/tmp")
shutil.copy(HERE + "/rh_common.py.bk9", _t9 + "/rh_common_bk9.py")
for _ak in ("", "RHp3"):
    _x, _y = _dump_for(HERE, "rh_common", _ak), _dump_for(_t9, "rh_common_bk9", _ak)
    chk(f"RH_AKIND={_ak or 'unset'} read unchanged by the RHK edits: rh_common KINDS / PAIRS / ARM_FORMS_ALL equal the pre-RHK copy (rh_common.py.bk9)", bool(_x) and _x == _y, f"{len(_x)} chars")
os.remove(_t9 + "/rh_common_bk9.py")
os.rmdir(_t9)
if AK == "RHK":
    chk("RHK pairs: P1 is RHK - B0 under C*, P2/P4 pairs on RHK, B0 attribution pair on B0, beside RHK - RH under R0Hb20 and R0C10XHb20",
        C.PAIRS["P1"] == ("RHK_R0C10XHb20", "B0_R0C10XHb20") and C.PAIRS["P2_ACC_RH"] == ("RHK_R0C10XHb20", "RHK_R0KHb20") and C.PAIRS["P4"] == ("RHK_R0C10XHb20", "B0_R0KHb20")
        and C.PAIRS["P2_ACC_B0"] == ("B0_R0C10XHb20", "B0_R0KHb20") and C.PAIRS["BESIDE_RHK_minus_RH_R0C10XHb20"] == ("RHK_R0C10XHb20", "RH_R0C10XHb20") and "BESIDE_RHK_minus_RH_R0Hb20" in C.PAIRS, "")
    chk("RHK read rolls RHK only (ROLL_KINDS), runs of B0 and RH are reused", C.ROLL_KINDS == ("RHK",) and set(C.RUN_KINDS) == {"RHK", "B0", "RH"}, str(C.RUN_KINDS))
    miss = [f"{k}_{f}_s{s}" for s in C.SEEDS for k, fs in (("B0", C.FORM_ORDER), ("RH", ("R0Hb20", "R0C10XHb20"))) for f in fs if not os.path.exists(f"{C.OUT}/t1/I10{k}_{f}_s{s}/T1_DONE")]
    chk("the reused dumps exist (B0: 6 forms x 2 seeds, RH: 2 forms x 2 seeds, T1_DONE)", not miss, miss[:4])
    reuse = [C.OUT + "/read/" + x for x in ("RESULT_ROLLHIST.json", "refs_q3/RESULT_Q3.json", "refs_hw/RESULT_HEADWAY.json", "STRAT_FLOOR_UEA.json")] + [f"{C.OUT}/t0/eval_{k}_s{s}.npz" for k in ("RH", "B0") for s in C.SEEDS] \\
        + [f"{C.CK}/{k}_s{s}/ckpt_last.pt" for k in ("RH", "B0") for s in C.SEEDS]
    chk("the PARTIAL read's reused files exist (result, refs readers, strat floor, T0 evals of RH and B0, RH and B0 checkpoints for the provenance check)", all(os.path.isfile(x) for x in reuse), [x for x in reuse if not os.path.isfile(x)][:3])
    ch = open(HERE + "/rh_read_k_chain.sh", encoding="utf-8").read()
    order = ["step blobcheck", "step readertests", 'step "pull_RHK_s$s"', 'step "ro_RHK_${f}_s$s"', "step t0evals", "step gt0", "step headway", "step q3", "step strat ", "step read "]
    pos = [ch.find(o) for o in order]
    chk("rh_read_k_chain.sh has every step, in order (blobcheck, selftest, pull, rollouts, T0, guard, headway, q3, strat, read)", all(q >= 0 for q in pos) and pos == sorted(pos), pos)
    chk("rh_read_k_chain.sh exports RH_AKIND=RHK, pulls and rolls RHK only, writes under out/read_k, takes --rh-result, repeats no K-REPLAY and no stage-A reference",
        "export RH_AKIND=RHK" in ch and "rh_roll.py RHK" in ch and "rh_roll.py RH " not in ch and "rh_roll.py B0" not in ch and "pull_arm RHK_s$s" in ch and "pull_arm RH_" not in ch and "pull_arm B0" not in ch
        and "--out $RD/RESULT_ROLLHIST_K.json" in ch and "--rh-result" in ch and "rh_replay" not in ch, "")
    chk("rh_read_k_chain.sh writes nothing the PARTIAL or P03 read wrote (no out/read/ or out/read_p03/ output, no out/steps/ or steps_p03 marker, own log)",
        "--out-dir $O/read " not in ch and "$STEPS/" in ch and "steps_k" in ch and "steps_p03" not in ch and "read_p03" not in ch and "rh_read_k_chain.log" in ch and "> $O/read/" not in ch, "")
    b4 = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/rh_read_k_chain.sh"], capture_output=True, text=True)
    chk("bash -n rh_read_k_chain.sh", b4.returncode == 0, b4.stderr[:200])
    chk("rh_read.py carries the K answer and the NEXT_K table", "k_answer" in open(HERE + "/rh_read.py", encoding="utf-8").read() and "NEXT_K" in open(HERE + "/rh_read.py", encoding="utf-8").read(), "")
    chk("rh_read_k_chain.cmd exists", os.path.isfile(HERE + "/rh_read_k_chain.cmd"), "")
'''
sub(anchor, block)
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched dry_check.py")
