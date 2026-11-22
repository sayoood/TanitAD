"""dry_check.py -- resolves every step of rh_read_chain.sh without a rollout: the 24 rollout argvs rendered with UE_s0.A standing in for each arm (forms, flags, tier-of-record executor, wrapper, paths),
inputs exist, T0 argvs, reader argument acceptance and imports, `bash -n`, the reader's selftest, the tree blobcheck, sha256 of every script.  Runs nothing heavier than imports.  Writes DRY_CHECK.json.
Usage: python dry_check.py   (exit 1 if any check fails)"""
import ast
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import rh_common as C  # noqa: E402

res, bad = [], []
AK = C.A_KIND                                                      # P03: RH_AKIND=RHp3 checks the read of the next arm (rh_read_p03_chain.sh); default RH = the finished PARTIAL read, unchanged
OUTJ = "DRY_CHECK_P03.json" if AK == "RHp3" else "DRY_CHECK_K.json" if AK == "RHK" else "DRY_CHECK_RL.json" if AK == "RL" else "DRY_CHECK.json"


def chk(name, ok, detail=""):
    res.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(("ok   " if ok else "FAIL "), name, str(detail)[:200], flush=True)
    if not ok:
        bad.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# 1 tree
r = subprocess.run([C.PY, HERE + "/blobcheck.py", C.TREE, "43319227"], capture_output=True, text=True, encoding="utf-8")
chk("tree blobcheck 3407 of 3407, 0 bad, 0 extra", r.stdout.startswith("checked 3407 of 3407 bad 0 extra []"), r.stdout.splitlines()[0] if r.stdout else r.stderr[-200:])
chk("hb executor sha256 74fd8445", sha(C.TREE + "/stack/scripts/v8f_t1_eval_hb.py").startswith("74fd8445"))
chk("pinned tool v8f_t1_eval.py sha256 61e140ed (cdbff971 lineage)", sha(C.TREE + "/stack/scripts/v8f_t1_eval.py").startswith("61e140ed"))
chk("study wrapper rzgfh sha256 373ffe72f2655c35 (ADD68)", sha(C.CODE + "/integ10k/v8f_t1_eval_reader_rule_rzgfh.py").startswith("373ffe72f2655c35"))
chk("readout wrapper rzg sha256 93879db8c0233867 (ADD63)", sha(C.CODE + "/integ10k/v8f_t1_eval_reader_rule_rzg.py").startswith("93879db8c0233867"))
# 2 inputs
for k, v in C.INPUTS.items():
    chk(f"input {k} exists", os.path.exists(v), v)
for f in ("dist_rule.py", "v8f_t1_eval_rule.py", "v8f_t1_eval_rule_mix.py", "qcal_map.py"):
    chk(f"distrule file {f}", os.path.isfile(C.CODE + "/distrule/" + f))
chk("readout wrapper adapter rzgi present", os.path.isfile(C.CODE + "/integ10k/" + C.WRAPPER + ".py"), C.WRAPPER)
chk("UE_s0.A checkpoint sha256 2fd4b747", os.path.isfile(C.ckpt_of("UEA", 0)) and sha(C.ckpt_of("UEA", 0)).startswith("2fd4b747"))
# 3 the 24 argvs
seen = set()
for k in C.ROLL_KINDS:
    for s in C.SEEDS:
        for f in C.FORM_ORDER:
            out = C.folder(k, f, s)
            a = C.argv_record(k, f, s, ckpt=C.ckpt_of("UEA", 0))
            j = " ".join(a)
            ok = (a[1].endswith("/stack/scripts/v8f_t1_eval_hb.py") and a[a.index("--kappa-max") + 1] == "0.2" and a[a.index("--inner") + 1] == C.WRAPPER and a[a.index("--form") + 1] == "R0"
                  and a[a.index("run") - 0] == "run" and f"--out-dir {out}" in j and f"--tag {C.tag(k, f, s)}" in j and "--device cpu" in j and "--dump-ticks" in j and "--limit causal" in j)
            fl = C.FORMS[f]
            ok &= all(x in a for x in fl) and (("--readout-log" in a) and a[a.index("--readout") + 1] == fl[1])
            if "--acc-gain" in fl:
                ok &= a[a.index("--acc-gain") + 1] == fl[fl.index("--acc-gain") + 1]
            ok &= ("--path-exec" in a) == ("--path-exec" in fl)
            seen.add(out)
            if not ok:
                chk(f"argv {k} {f} s{s}", False, j[:300])
NR = len(C.ROLL_KINDS) * len(C.SEEDS) * len(C.FORM_ORDER)
chk(f"{NR} rollout argvs render (kind x seed x form), {NR} distinct output folders, tier-of-record executor, six forms", len(seen) == NR and not any(not r["ok"] for r in res if r["check"].startswith("argv ")), f"{len(seen)} folders")
rs = subprocess.run([C.PY, HERE + "/rh_roll.py", C.ROLL_KINDS[0], "R0C10XHb20", "1", "--dry", "--ckpt", C.ckpt_of("UEA", 0)], capture_output=True, text=True, encoding="utf-8")
chk("rh_roll.py --dry renders one argv", rs.returncode == 0 and "v8f_t1_eval_hb.py" in rs.stdout, rs.stdout[:120])
# 4 T0
sys.path.insert(0, HERE)
import rh_t0 as T0  # noqa: E402
a0 = T0.argv_t0(f"{C.ROLL_KINDS[0]}_s0", ckpt=C.ckpt_of("UEA", 0))
miss = [x for x in a0 if x.startswith(("F:/", "E:/")) and not x.endswith("ckpt_last.pt") and not os.path.exists(x) and x != T0.T0DIR]
chk("T0 (G-T0) argv inputs exist", not miss and a0[1].endswith("v8f_mini_eval.py") and "--project" in a0 and "--families" in a0, miss)
# 5 reader tools
for f in ("code/q3/q3_read_r3.py", "code/q3/q3_read.py", "code/headway/headway_read_r2.py", "code/integ10k/strat_roll_read.py", "rh_read.py", "rh_strat.py", "rh_t0.py", "rh_roll.py", "rh_replay.py", "c_exec.py", "rh_dryrun.py", "rh_t0_dry.py"):
    try:
        ast.parse(open(HERE + "/" + f, encoding="utf-8").read())
        chk(f"parses {f}", True)
    except SyntaxError as e:
        chk(f"parses {f}", False, e)
for v in (C.W + "/../../../v8proof/pod/stageB/hierrep_read.py", "F:/TanitAD-scratch/v8proof/pod/stageB/hierrep_read.py", "F:/TanitAD-scratch/v8proof/res/cache/frames.npz", "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"):
    if "../" not in v:
        chk(f"reader dependency {v}", os.path.exists(v))
q3a = subprocess.run([C.PY, HERE + "/code/q3/q3_read_r3.py", "read", "--help"], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, HIERREP_STAGEB="F:/TanitAD-scratch/v8proof/pod/stageB", TANITEVAL_ROOT=C.TREE + "/taniteval"), cwd=HERE + "/code/q3")
chk("q3_read_r3.py imports and takes --t1-root --pairs --seeds --spread-arm --same-sign", q3a.returncode == 0 and "--same-sign" in q3a.stdout, q3a.stderr[-200:])
hwa = subprocess.run([C.PY, HERE + "/code/headway/headway_read_r2.py", "read", "--help"], capture_output=True, text=True, encoding="utf-8",
                     env=dict(os.environ, HEADWAY_Q3DIR=HERE + "/code/q3", HIERREP_STAGEB="F:/TanitAD-scratch/v8proof/pod/stageB", TANITEVAL_ROOT=C.TREE + "/taniteval"), cwd=HERE + "/code/headway")
chk("headway_read_r2.py imports and takes --t1-root --pairs --seeds", hwa.returncode == 0 and "--t1-root" in hwa.stdout, hwa.stderr[-200:])
sr = subprocess.run([C.PY, "-c", "import sys; sys.path.insert(0, r'%s'); import rh_strat as R; r=R.S.run_s4(); print(bool(r['pass']))" % HERE], capture_output=True, text=True, encoding="utf-8")
chk("rh_strat imports the frozen ADD66 reader and its s4 synthetic control passes", sr.stdout.strip().endswith("True"), sr.stdout + sr.stderr[-200:])
st = subprocess.run([C.PY, HERE + "/rh_read.py", "--selftest"], capture_output=True, text=True, encoding="utf-8")
chk("rh_read.py selftest (K-SELF, K-PLANT, K-FLOOR, K-SEEDS, K-HARMS, mutation other-seed arm)", st.returncode == 0 and "all hold: True" in st.stdout, st.stdout.strip().splitlines()[-1] if st.stdout else st.stderr[-200:])
sf = subprocess.run([C.PY, HERE + "/rh_strat_floor.py", "--help"], capture_output=True, text=True, encoding="utf-8")
chk("rh_strat_floor.py imports the frozen ADD66 reader and takes --s0 --s1 --out", sf.returncode == 0 and "--s0" in sf.stdout, sf.stderr[-200:])
chk("rh_read_chain.sh has the strat_floor step before read, and rh_read.py takes --strat-floor", open(HERE + "/rh_read_chain.sh", encoding="utf-8").read().index("step strat_floor") < open(HERE + "/rh_read_chain.sh", encoding="utf-8").read().index('step read "') and "--strat-floor" in open(HERE + "/rh_read.py", encoding="utf-8").read())
pr = subprocess.run([C.PY, "-c", "import sys; sys.path.insert(0, r'%s'); import rh_common as C; print(','.join('I10%%s:I10%%s'%%(a,b) for a,b in C.PAIRS.values()).count(':'))" % HERE], capture_output=True, text=True, encoding="utf-8")
chk("pair list renders", pr.stdout.strip() == str(len(C.PAIRS)), pr.stdout)
b = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/rh_read_chain.sh"], capture_output=True, text=True)
chk("bash -n rh_read_chain.sh", b.returncode == 0, b.stderr[:200])
# 5c P03: the default read is unchanged (rh_common vs its .bp03 copy with RH_AKIND unset), and the RHp3 read is wired
_dump = "import json,sys; sys.path.insert(0, r'%s'); import %s as C; print(json.dumps({'K': C.KINDS, 'P': C.PAIRS, 'A': C.ARM_FORMS_ALL, 'F': C.FORM_ORDER}, sort_keys=True))"
_e = {k: v for k, v in os.environ.items() if k != "RH_AKIND"}
import shutil, tempfile  # noqa: E401,E402
_t = tempfile.mkdtemp(prefix="bp03_", dir=HERE + "/tmp")
shutil.copy(HERE + "/rh_common.py.bp03", _t + "/rh_common_bp03.py")
_a = subprocess.run([C.PY, "-c", _dump % (HERE, "rh_common")], capture_output=True, text=True, encoding="utf-8", env=_e).stdout
_b = subprocess.run([C.PY, "-c", _dump % (_t, "rh_common_bp03")], capture_output=True, text=True, encoding="utf-8", env=_e).stdout
chk("default read unchanged: rh_common KINDS / PAIRS / ARM_FORMS_ALL equal the pre-P03 copy (rh_common.py.bp03) with RH_AKIND unset", bool(_a) and _a == _b, f"{len(_a)} chars")
os.remove(_t + "/rh_common_bp03.py")
os.rmdir(_t)
if AK == "RHp3":
    chk("RHp3 pairs: P1 is RHp3 - B0 under C*, P2/P4 pairs on RHp3, B0 attribution pair on B0, beside RHp3 - RH under R0Hb20 and R0C10XHb20",
        C.PAIRS["P1"] == ("RHp3_R0C10XHb20", "B0_R0C10XHb20") and C.PAIRS["P2_ACC_RH"] == ("RHp3_R0C10XHb20", "RHp3_R0KHb20") and C.PAIRS["P4"] == ("RHp3_R0C10XHb20", "B0_R0KHb20")
        and C.PAIRS["P2_ACC_B0"] == ("B0_R0C10XHb20", "B0_R0KHb20") and C.PAIRS["BESIDE_RHp3_minus_RH_R0C10XHb20"] == ("RHp3_R0C10XHb20", "RH_R0C10XHb20") and "BESIDE_RHp3_minus_RH_R0Hb20" in C.PAIRS, "")
    chk("RHp3 read rolls RHp3 only (ROLL_KINDS), runs of B0 and RH are reused", C.ROLL_KINDS == ("RHp3",) and set(C.RUN_KINDS) == {"RHp3", "B0", "RH"}, str(C.RUN_KINDS))
    miss = [f"{k}_{f}_s{s}" for s in C.SEEDS for k, fs in (("B0", C.FORM_ORDER), ("RH", ("R0Hb20", "R0C10XHb20"))) for f in fs if not os.path.exists(f"{C.OUT}/t1/I10{k}_{f}_s{s}/T1_DONE")]
    chk("the reused dumps exist (B0: 6 forms x 2 seeds, RH: 2 forms x 2 seeds, T1_DONE)", not miss, miss[:4])
    reuse = [C.OUT + "/read/" + x for x in ("RESULT_ROLLHIST.json", "refs_q3/RESULT_Q3.json", "refs_hw/RESULT_HEADWAY.json", "STRAT_FLOOR_UEA.json")] + [f"{C.OUT}/t0/eval_{k}_s{s}.npz" for k in ("RH", "B0") for s in C.SEEDS]         + [f"{C.CK}/{k}_s{s}/ckpt_last.pt" for k in ("RH", "B0") for s in C.SEEDS]
    chk("the PARTIAL read's reused files exist (result, refs readers, strat floor, T0 evals of RH and B0, RH and B0 checkpoints for the provenance check)", all(os.path.isfile(x) for x in reuse), [x for x in reuse if not os.path.isfile(x)][:3])
    ch = open(HERE + "/rh_read_p03_chain.sh", encoding="utf-8").read()
    order = ["step blobcheck", "step readertests", 'step "pull_RHp3_s$s"', 'step "ro_RHp3_${f}_s$s"', "step t0evals", "step gt0", "step headway", "step q3", "step strat ", "step read "]
    pos = [ch.find(o) for o in order]
    chk("rh_read_p03_chain.sh has every step, in order (blobcheck, selftest, pull, rollouts, T0, guard, headway, q3, strat, read)", all(p >= 0 for p in pos) and pos == sorted(pos), pos)
    chk("rh_read_p03_chain.sh exports RH_AKIND=RHp3, pulls and rolls RHp3 only, writes under out/read_p03, takes --rh-result, repeats no K-REPLAY and no stage-A reference",
        "export RH_AKIND=RHp3" in ch and "rh_roll.py RHp3" in ch and "rh_roll.py RH " not in ch and "rh_roll.py B0" not in ch and "pull_arm RHp3_s$s" in ch and "pull_arm RH_" not in ch and "pull_arm B0" not in ch
        and "--out $RD/RESULT_ROLLHIST_P03.json" in ch and "--rh-result" in ch and "rh_replay" not in ch, "")
    chk("rh_read_p03_chain.sh writes nothing the PARTIAL read wrote (no out/read/ output, no out/steps/ marker, own log)", "--out-dir $O/read" not in ch and "$STEPS/" in ch and "steps_p03" in ch and "rh_read_p03_chain.log" in ch and "> $O/read/" not in ch, "")
    b3 = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/rh_read_p03_chain.sh"], capture_output=True, text=True)
    chk("bash -n rh_read_p03_chain.sh", b3.returncode == 0, b3.stderr[:200])
    chk("rh_read.py carries the P03 answer and the NEXT_P03 table", "p03_answer" in open(HERE + "/rh_read.py", encoding="utf-8").read() and "NEXT_P03" in open(HERE + "/rh_read.py", encoding="utf-8").read(), "")
    chk("rh_read_p03_chain.cmd exists", os.path.isfile(HERE + "/rh_read_p03_chain.cmd"), "")
# 5d RHK (Amendment 8 DRAFT): the default read and the RHp3 read are unchanged by the RHK edits (rh_common vs its pre-RHK copy .bk9, each under its own RH_AKIND), and the RHK read is wired
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
    reuse = [C.OUT + "/read/" + x for x in ("RESULT_ROLLHIST.json", "refs_q3/RESULT_Q3.json", "refs_hw/RESULT_HEADWAY.json", "STRAT_FLOOR_UEA.json")] + [f"{C.OUT}/t0/eval_{k}_s{s}.npz" for k in ("RH", "B0") for s in C.SEEDS] \
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
# 5e RL (Amendment 9 DRAFT): the default, RHp3 and RHK reads are unchanged by the RL edits (rh_common vs its pre-RL copy .bkRL, each under its own RH_AKIND), and the RL read is wired
_t10 = tempfile.mkdtemp(prefix="bkrl_", dir=HERE + "/tmp")
shutil.copy(HERE + "/rh_common.py.bkRL", _t10 + "/rh_common_bkrl.py")
for _ak in ("", "RHp3", "RHK"):
    _x, _y = _dump_for(HERE, "rh_common", _ak), _dump_for(_t10, "rh_common_bkrl", _ak)
    chk(f"RH_AKIND={_ak or 'unset'} read unchanged by the RL edits: rh_common KINDS / PAIRS / ARM_FORMS_ALL equal the pre-RL copy (rh_common.py.bkRL)", bool(_x) and _x == _y, f"{len(_x)} chars")
os.remove(_t10 + "/rh_common_bkrl.py")
os.rmdir(_t10)
if AK == "RL":
    chk("RL pairs: P1 is RL - B0 under C*, P2/P4 pairs on RL, B0 attribution pair on B0, beside RL - RH under R0Hb20 and R0C10XHb20",
        C.PAIRS["P1"] == ("RL_R0C10XHb20", "B0_R0C10XHb20") and C.PAIRS["P2_ACC_RH"] == ("RL_R0C10XHb20", "RL_R0KHb20") and C.PAIRS["P4"] == ("RL_R0C10XHb20", "B0_R0KHb20")
        and C.PAIRS["P2_ACC_B0"] == ("B0_R0C10XHb20", "B0_R0KHb20") and C.PAIRS["BESIDE_RL_minus_RH_R0C10XHb20"] == ("RL_R0C10XHb20", "RH_R0C10XHb20") and "BESIDE_RL_minus_RH_R0Hb20" in C.PAIRS, "")
    chk("RL read rolls RL only (ROLL_KINDS), runs of B0 and RH are reused", C.ROLL_KINDS == ("RL",) and set(C.RUN_KINDS) == {"RL", "B0", "RH"}, str(C.RUN_KINDS))
    miss = [f"{k}_{f}_s{s}" for s in C.SEEDS for k, fs in (("B0", C.FORM_ORDER), ("RH", ("R0Hb20", "R0C10XHb20"))) for f in fs if not os.path.exists(f"{C.OUT}/t1/I10{k}_{f}_s{s}/T1_DONE")]
    chk("the reused dumps exist (B0: 6 forms x 2 seeds, RH: 2 forms x 2 seeds, T1_DONE)", not miss, miss[:4])
    reuse = [C.OUT + "/read/" + x for x in ("RESULT_ROLLHIST.json", "refs_q3/RESULT_Q3.json", "refs_hw/RESULT_HEADWAY.json", "STRAT_FLOOR_UEA.json")] + [f"{C.OUT}/t0/eval_{k}_s{s}.npz" for k in ("RH", "B0") for s in C.SEEDS]         + [f"{C.CK}/{k}_s{s}/ckpt_last.pt" for k in ("RH", "B0") for s in C.SEEDS]
    chk("the PARTIAL read's reused files exist (result, refs readers, strat floor, T0 evals of RH and B0, RH and B0 checkpoints for the provenance check)", all(os.path.isfile(x) for x in reuse), [x for x in reuse if not os.path.isfile(x)][:3])
    ch = open(HERE + "/rh_read_rl_chain.sh", encoding="utf-8").read()
    order = ["step blobcheck", "step readertests", 'step "pull_RL_s$s"', 'step "ro_RL_${f}_s$s"', "step t0evals", "step gt0", "step headway", "step q3", "step strat ", "step read "]
    pos = [ch.find(o) for o in order]
    chk("rh_read_rl_chain.sh has every step, in order (blobcheck, selftest, pull, rollouts, T0, guard, headway, q3, strat, read)", all(q >= 0 for q in pos) and pos == sorted(pos), pos)
    chk("rh_read_rl_chain.sh exports RH_AKIND=RL, pulls and rolls RL only, writes under out/read_rl, takes --rh-result, repeats no K-REPLAY and no stage-A reference",
        "export RH_AKIND=RL" in ch and "rh_roll.py RL" in ch and "rh_roll.py RH " not in ch and "rh_roll.py B0" not in ch and "pull_arm RL_s$s" in ch and "pull_arm RH_" not in ch and "pull_arm B0" not in ch
        and "--out $RD/RESULT_ROLLHIST_RL.json" in ch and "--rh-result" in ch and "rh_replay" not in ch, "")
    chk("rh_read_rl_chain.sh writes nothing the PARTIAL, P03 or K read wrote (no out/read/, out/read_p03/ or out/read_k/ output, own steps dir, own log)",
        "--out-dir $O/read " not in ch and "$STEPS/" in ch and "steps_rl" in ch and "steps_p03" not in ch and "steps_k" not in ch and "read_p03" not in ch and "read_k" not in ch and "rh_read_rl_chain.log" in ch and "> $O/read/" not in ch, "")
    b5 = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/rh_read_rl_chain.sh"], capture_output=True, text=True)
    chk("bash -n rh_read_rl_chain.sh", b5.returncode == 0, b5.stderr[:200])
    chk("rh_read.py carries the RL answer and the NEXT_RL table", "rl_answer" in open(HERE + "/rh_read.py", encoding="utf-8").read() and "NEXT_RL" in open(HERE + "/rh_read.py", encoding="utf-8").read(), "")
    chk("rh_read_rl_chain.cmd exists", os.path.isfile(HERE + "/rh_read_rl_chain.cmd"), "")
# 6 sha256 of every script
files = sorted(f for f in glob.glob(HERE + "/*") if os.path.isfile(f) and f.endswith((".py", ".sh", ".cmd", ".md", ".json", ".txt")) and not f.endswith(("MANIFEST_SHA256.txt",)))
files += sorted(glob.glob(HERE + "/code/*/*.py") + glob.glob(HERE + "/code/*/*.md"))
man = {os.path.relpath(f, HERE).replace("\\", "/"): sha(f) for f in files}
json.dump(man, open(HERE + "/tmp/dry_sha.json", "w"), indent=0)
chk("sha256 of every script recorded", len(man) > 20, f"{len(man)} files")
json.dump({"ok": not bad, "failed": bad, "checks": res, "treated_arm": AK}, open(HERE + "/" + OUTJ, "w", encoding="utf-8"), indent=1)
print("ZZ DRY-CHECK " + ("PASS" if not bad else "FAIL " + ",".join(bad)) + " ZZ")
sys.exit(1 if bad else 0)
