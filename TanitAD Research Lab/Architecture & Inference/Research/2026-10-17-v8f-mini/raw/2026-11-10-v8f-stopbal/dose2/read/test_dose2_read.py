"""Tests of the second-dose read additions (dose2_read.py, null_banks.py). CPU, numpy only, no torch, no network, no box.
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_dose2_read.py <scratch dir under F:/TanitAD-scratch/dose2_build/>   -> prints READ2 ok=<n> bad=<m>
Planted fake banks are built under the scratch dir: one per new tag (BAL-G2 s0 / s1, BAL-L2 s0 / s1) and the three NULL arms. Every refusal has a planted case."""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dose2_read as D  # noqa: E402  (imports the first dose's read package by path and patches its tables)
import null_banks as NB  # noqa: E402

SR, RD = D.SR, D.RD
N_OK, N_BAD = 0, 0


def check(name, cond):
    global N_OK, N_BAD
    if cond:
        N_OK += 1
    else:
        N_BAD += 1
        print(f"READ2-FAIL {name}")


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def plant(pulls, tag, gw, lw, commit=None, declared="PASS", ck=None, man_sha=None):
    """one fake bank: ckpt, BANK_MANIFEST.json, RUN.done, selfcheck.json, BANK.done"""
    d = os.path.join(pulls, "stopbal_" + tag)
    os.makedirs(d, exist_ok=True)
    ck = ck if ck is not None else ("ckpt of " + tag).encode()
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(ck)
    s = man_sha or sha_b(ck)
    json.dump({"tag": tag, "commit": commit or RD.COMMIT, "files": {"ckpt_last.pt": {"bytes": len(ck), "sha256": s}}}, open(os.path.join(d, "BANK_MANIFEST.json"), "w"))
    json.dump({"tag": tag, "evidence": {"ckpt_sha256": s}}, open(os.path.join(d, "RUN.done"), "w"))
    json.dump({"declared_vs_built": declared, "levers": {"goal_pos_weight_stop": gw, "lon_class_weight_stop": lw},
               "live": {"terms": {"goal": 0.1, "lon": 0.2}, "dead_params": [], "groups_grad": {"T": 1.0}}}, open(os.path.join(d, "selfcheck.json"), "w"))
    open(os.path.join(d, "BANK.done"), "w").write("verified\n")
    return d


def main(tmp):
    os.makedirs(tmp, exist_ok=True)
    pulls = os.path.join(tmp, "pulls")
    RD.PULL = pulls
    want = {"BAL-G2_s0": (20.81, 1.0), "BAL-G2_s1": (20.81, 1.0), "BAL-L2_s0": (1.0, 22.08), "BAL-L2_s1": (1.0, 22.08)}
    # ---- the tables
    check("CARD_WEIGHTS carries the dose-2 expected levers (G2 20.81 / 1.0, L2 1.0 / 22.08) and keeps the first dose's", SR.CARD_WEIGHTS["BAL-G2"] == (20.81, 1.0) and SR.CARD_WEIGHTS["BAL-L2"] == (1.0, 22.08)
          and SR.CARD_WEIGHTS["BAL-G"] == (4.5616, 1.0) and SR.CARD_WEIGHTS["BAL-NULL"] == (1.0, 1.0))
    check("the lever arms of this read are BAL-G2 and BAL-L2 with the first dose's decision / score keys", SR.LEVERS == {"BAL-G2": {"decision": "goal_stop", "score": "goal_score"}, "BAL-L2": {"decision": "lon_stop", "score": "lon_score"}})
    check("run_dumps knows the four new tags (and still the first dose's seven)", all(t in RD.ARMS for t in D.NEW_TAGS) and len(RD.ARMS) == 11 and RD.HERE == D.HERE)
    check("the code of record of the read is 715ef8fc", RD.COMMIT == "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf")
    check("kind_of reads BAL-G2_s1 as BAL-G2", SR.kind_of("BAL-G2_s1") == "BAL-G2" and SR.kind_of("BAL-NULL_rep") == "BAL-NULL")
    # ---- control 1 on selfchecks (the lever read-back), every new tag
    for t, (gw, lw) in want.items():
        sc = {"declared_vs_built": "PASS", "levers": {"goal_pos_weight_stop": gw, "lon_class_weight_stop": lw}}
        check(f"{t}: a selfcheck reading its own levers passes control 1", SR.check_levers(SR.kind_of(t), sc) == [])
        for name, (g2, l2) in (("first dose's weights", (4.5616 if "G2" in t else 1.0, 4.6991 if "L2" in t else 1.0)), ("both 1.0", (1.0, 1.0)), ("off by 0.01", (gw + (0.01 if gw != 1.0 else 0), lw + (0.01 if lw != 1.0 else 0))), ("swapped", (lw, gw))):
            bad = SR.check_levers(SR.kind_of(t), {"declared_vs_built": "PASS", "levers": {"goal_pos_weight_stop": g2, "lon_class_weight_stop": l2}})
            check(f"{t}: planted '{name}' is refused by control 1", bad != [] if (g2, l2) != (gw, lw) else bad == [])
        check(f"{t}: declared_vs_built FAIL is refused", SR.check_levers(SR.kind_of(t), dict(sc, declared_vs_built="FAIL")) != [])
    # ---- run_dumps refusal() on planted banks, every new tag
    for t, (gw, lw) in want.items():
        plant(pulls, t, gw, lw)
        why, sh = RD.refusal(t)
        check(f"{t}: a planted whole bank is accepted by the dump refusal check", why is None and sh is not None)
    for t, (gw, lw) in want.items():
        d = plant(os.path.join(tmp, "bad"), t, gw, lw)
        RD.PULL = os.path.join(tmp, "bad")
        check(f"{t}: accepted in a second fake pulls dir (control)", RD.refusal(t)[0] is None)
        plant(os.path.join(tmp, "bad"), t, 1.0, 1.0)
        w1 = RD.refusal(t)[0]
        check(f"{t}: RED own lever reads 1.0 (never built): refused, control 1 named", w1 is not None and "control 1" in w1)
        plant(os.path.join(tmp, "bad"), t, 4.5616 if "G2" in t else 1.0, 4.6991 if "L2" in t else 1.0)
        check(f"{t}: RED the first dose's weight is refused", RD.refusal(t)[0] is not None)
        plant(os.path.join(tmp, "bad"), t, gw, lw, man_sha="0" * 64)
        check(f"{t}: RED manifest sha256 differs from the checkpoint: refused", "sha256 differs" in (RD.refusal(t)[0] or ""))
        plant(os.path.join(tmp, "bad"), t, gw, lw, commit="a" * 40)
        check(f"{t}: RED trained at another commit: refused", "commit" in (RD.refusal(t)[0] or ""))
        plant(os.path.join(tmp, "bad"), t, gw, lw, declared="FAIL")
        check(f"{t}: RED G-DVB not PASS: refused", RD.refusal(t)[0] is not None)
        os.remove(os.path.join(d, "ckpt_last.pt"))
        check(f"{t}: RED checkpoint missing: refused", "checkpoint missing" in (RD.refusal(t)[0] or ""))
        plant(os.path.join(tmp, "bad"), t, gw, lw)
        os.remove(os.path.join(d, "BANK_MANIFEST.json"))
        check(f"{t}: RED manifest missing: refused", "BANK_MANIFEST.json missing" in (RD.refusal(t)[0] or ""))
    RD.PULL = pulls
    check("dumps refuses a NULL tag and a first-dose tag (the CLI names them)", all(_exits(lambda t=t: D.cmd_dumps([t])) for t in ("BAL-NULL_s0", "BAL-G_s0", "BAL-L_s1")))
    # ---- the banked NULL arms: reference and floor, checkpoint sha256 re-asserted
    npulls = os.path.join(tmp, "npulls")
    for t in D.NULL_TAGS:
        plant(npulls, t, 1.0, 1.0)
    check("planted whole NULL banks (s0, s1, rep) pass", NB.problems(npulls) == [])
    shas = NB.checkpoint_shas(npulls)
    check("their checkpoint sha256 are the files' own", all(shas[t] == sha_b(("ckpt of " + t).encode()) for t in D.NULL_TAGS))
    for t in D.NULL_TAGS:
        dd = os.path.join(npulls, "stopbal_" + t)
        keep = open(os.path.join(dd, "ckpt_last.pt"), "rb").read()
        open(os.path.join(dd, "ckpt_last.pt"), "wb").write(keep + b"x")
        check(f"RED {t}: a changed checkpoint byte is refused (manifest AND RUN.done sha256)", len([x for x in NB.problems(npulls) if x.startswith(t)]) == 2)
        open(os.path.join(dd, "ckpt_last.pt"), "wb").write(keep)
        for f in ("BANK.done", "BANK_MANIFEST.json", "RUN.done", "selfcheck.json", "ckpt_last.pt"):
            b = open(os.path.join(dd, f), "rb").read()
            os.remove(os.path.join(dd, f))
            check(f"RED {t}: missing {f} is refused and the problem names the file", any(f in x and x.startswith(t) for x in NB.problems(npulls)))
            open(os.path.join(dd, f), "wb").write(b)
        sc = json.load(open(os.path.join(dd, "selfcheck.json")))
        sc["levers"]["goal_pos_weight_stop"] = 4.5616
        json.dump(sc, open(os.path.join(dd, "selfcheck.json"), "w"))
        check(f"RED {t}: a lever other than 1.0 in a NULL selfcheck is refused", any("goal_pos_weight_stop" in x for x in NB.problems(npulls)))
        sc["levers"]["goal_pos_weight_stop"] = 1.0
        json.dump(sc, open(os.path.join(dd, "selfcheck.json"), "w"))
        check(f"{t}: restored, whole again", NB.problems(npulls) == [])
    check("RED a missing NULL bank folder is refused and named", any("BAL-NULL_rep" in x and "missing" in x for x in NB.problems(os.path.join(tmp, "no_such_pulls"))) and len(NB.problems(os.path.join(tmp, "no_such_pulls"))) == 3)
    check("RED the read refuses to start when a NULL bank is missing (exit 2, before stopbal_read runs)", _exits(lambda: D.cmd_read(["--out", os.path.join(tmp, "r.json"), "--pull", os.path.join(tmp, "no_such_pulls")]), code=2))
    check("no read report was written by the refused read", not os.path.exists(os.path.join(tmp, "r.json")))
    check("main() refuses an unknown subcommand", _exits(lambda: D.main(["frobnicate"])))
    if os.path.isdir("C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0"):
        pr, rs = D.refs("C:/Users/Admin/t2v8f_pulls")
        check("the REAL banked NULL arms are whole: manifest, RUN.done, NULL lever read-back 1.0", pr == [] and set(rs) == set(D.NULL_TAGS))
        for t in D.NULL_TAGS:
            m = json.load(open(f"C:/Users/Admin/t2v8f_pulls/stopbal_{t}/BANK_MANIFEST.json"))
            check(f"{t}: the real checkpoint sha256 equals its manifest", rs.get(t) == m["files"]["ckpt_last.pt"]["sha256"])
            check(f"{t}: the real selfcheck passes control 1 as a NULL arm", SR.check_levers("BAL-NULL", json.load(open(f"C:/Users/Admin/t2v8f_pulls/stopbal_{t}/selfcheck.json"))) == [])
    # ---- --levers selection
    check("--levers default keeps both lever arms", list(D.select_levers([])) == ["BAL-G2", "BAL-L2"])
    check("--levers BAL-G2 keeps only it", list(D.select_levers(["BAL-G2"])) == ["BAL-G2"] and list(SR.LEVERS) == ["BAL-G2"])
    check("--levers BAL-L2 keeps only it", list(D.select_levers(["BAL-L2"])) == ["BAL-L2"])
    check("--levers with a first-dose or unknown name is refused", _exits(lambda: D.select_levers(["BAL-G"])) and _exits(lambda: D.select_levers(["BAL-G2", "X"])))
    D.select_levers([])
    # ---- default map
    m = SR.default_map("DUMPS", "SDEC1", "PULLS")
    check("the default map has the three NULL arms and the four new arms and nothing else", set(m) == set(D.NULL_TAGS) | set(D.NEW_TAGS))
    check("NULL arms read the first dose's STOPDEC folder, the new arms the dose-2 folder", all(m[t]["stopdec"].startswith("SDEC1/") for t in D.NULL_TAGS) and all(m[t]["stopdec"].replace("\\", "/").startswith(D.DOSE2_STOPDEC.replace("\\", "/")) for t in D.NEW_TAGS))
    check("every arm's dump is in the shared dump folder and its bank under the pulls dir", all(m[t]["dump"] == f"DUMPS/dump_{t}_m10_causal_heads" and m[t]["bank"] == f"PULLS/stopbal_{t}" for t in m))
    # ---- the job files
    jd = open(os.path.join(HERE, "job_dumps_dose2.cmd"), encoding="utf-8").read()
    line = [x for x in jd.splitlines() if "dose2_read.py" in x][0]
    check("job_dumps_dose2.cmd dumps exactly the four new tags, nothing else", re.findall(r"BAL-[A-Za-z0-9]+_[A-Za-z0-9]+", line) == list(D.NEW_TAGS) and "dumps" in line and "-I" in line)
    check("the job forces CPU and writes its own done file (not the first dose's)", "CUDA_VISIBLE_DEVICES=-1" in jd and "DUMPS_RUN2.done" in jd and "DUMPS_RUN.done" not in jd.replace("DUMPS_RUN2.done", ""))
    sj = open(os.path.join(HERE, "job_stopdec_dose2.cmd"), encoding="utf-8").read()
    check("job_stopdec_dose2.cmd names the four new tags, --split-json and the dose-2 outdir", all(t in sj for t in D.NEW_TAGS) and "--split-json" in sj and "dose2\\read\\out\\stopdec" in sj and "BAL-NULL" not in sj.split("rem", 2)[-1].split("\n", 1)[1].replace("NULL arms'", ""))
    # ---- the first dose's read files are untouched on disk (the patches live in memory only)
    check("the first dose's stopbal_read.py still carries its own LEVERS / CARD_WEIGHTS lines", '"BAL-G": {"decision"' in open(os.path.join(D.FIRST_READ, "stopbal_read.py"), encoding="utf-8").read() and '"BAL-G2"' not in open(os.path.join(D.FIRST_READ, "stopbal_read.py"), encoding="utf-8").read())
    print(f"READ2 ok={N_OK} bad={N_BAD}")
    return 0 if N_BAD == 0 else 1


def _exits(fn, code=None):
    try:
        fn()
    except SystemExit as e:
        return code is None or e.code == code
    return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
