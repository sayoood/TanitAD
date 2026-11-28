#!/usr/bin/env python3
"""dagorct1_selftest.py -- proves dagorct1_lib.py on (a) the REAL r1 box-2 log (A s0 FAILED K-REACH), (b) synthetic r2-shaped logs with one deliberate regression per arm, (c) synthetic pull directories with one
deliberate regression per identity clause, (d) the K-BUILD record against git (positive) and with the evidence broken (negative), (d2) the r2 TWO-STEP K-BUILD mode (--fix-pkg) against the REAL fix package at 8c24823f, with one deliberate regression per evidence clause, and the chain's wiring of its options, (e) the manifest against the READER's own slot_needs and verify_manifest on a
synthetic world (positive and negatives), (f) the rollout-code gate.  Every NEGATIVE arm must read RED/refused.   python -I dagorct1_selftest.py [--quick]   -> exit 0 iff every check is as expected.
CPU only, no torch.  Scratch files go to a temp dir on C: (NTFS), never to E:."""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
for _k in ("DORT1_COMPOSE_PKG", "DORT1_REFUSE_COMMITS"):      # read by the lib at import time; the checks below assume the defaults
    os.environ.pop(_k, None)
import dagorct1_lib as L  # noqa: E402

AR = "E:/Projects/TanitAD-artifacts"
MIRROR = "C:/Users/Admin/tanitad-push"
TREE = f"{AR}/dagt1_tree_b43c7a9a"
RD = f"{AR}/dagorc_rebase/read"
SNAP = f"{HERE}/selftest/box2_log_snapshot_20261028_0213_Kreach_FAIL.out"
ART = f"{AR}/v8f-dag-orc-oacc"
R = {"n": 0, "bad": []}


def check(name, cond, detail=""):
    R["n"] += 1
    if not cond:
        R["bad"].append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------- (a) constants are COPIES of banked values
def t_constants():
    src = open(f"{RD}/dag_orc_read.py", encoding="utf-8").read()

    def grab(name):
        m = re.search(name + r" = \{0: \"([0-9a-f]{64})\", 1: \"([0-9a-f]{64})\"\}", src)
        return {0: m.group(1), 1: m.group(2)} if m else None
    check("const M2B_SHA == the reader's M2B_CKPT_SHA", grab("M2B_CKPT_SHA") == L.M2B_SHA)
    check("const OACC_CKPT_SHA == the reader's", grab("OACC_CKPT_SHA") == L.OACC_CKPT_SHA)
    check("const BANK1_SHA == the reader's OACC_BANK_SHA", grab("OACC_BANK_SHA") == L.BANK1_SHA)
    check("const W_ACC / REP_SEED / STEPS == the reader's", f'W_ACC = "{L.W_ACC}"' in src and "REP_SEED = {0: 10, 1: 11}" in src and L.REP_SEED == {0: 10, 1: 11} and "STEPS, LR, WARMUP, PGT = 3000" in src)
    check("const BASE_COMMIT / BUILD_FILES == the reader's", f'BUILD_BASE_COMMIT = "{L.BASE_COMMIT[:8]}"' in src and all(f'"{f}"' in src for f in L.BUILD_FILES))
    for s in (0, 1):
        sums = open(f"{AR}/v8f-dagoacc/M2b_s{s}_dagoacc/SHA256SUMS.txt").read()
        check(f"DAG-OACC s{s} ckpt in its SHA256SUMS.txt == OACC_CKPT_SHA", f"{L.OACC_CKPT_SHA[s]} *ckpt_last.pt" in sums)
        am = json.load(open(f"{AR}/v8f-dagoacc/M2b_s{s}_dagoacc/argv.json"))["argv"]
        check(f"DAG-OACC s{s} argv names the bank sha / w_acc / init the lib pins", L.BANK1_SHA[s] in am and str(L.W_ACC) in am)


# ------------------------------------------------------------------------------------------------------------------------------- (b) the box-2 log parser
def mk_log(runs, extra_after=None, art="/e/Projects/TanitAD-artifacts/v8f-dag-orc-oacc"):
    """runs: list of (arm, seed, rec, reach) with rec/reach in 'PASS' | 'FAIL' | None (line absent).  Lines are the exact shapes the r1 chain printed (taken from the real snapshot)."""
    out = ["10-28 02:00 box 2 code = tip abc"]
    for (arm, s, rec, reach, *opt) in runs:
        g = L.run_name(arm, s)
        pth = opt[0] if opt else f"{art}/{g}"
        out.append(f"10-28 02:09 {g} pulled to {pth} (checkpoint sha256 {sha(g.encode())})")
        if rec == "PASS":
            out.append(f"DAG-ORC (DAG-OACC base) run record: PASS | {arm} s{s} | trainer seed {s} | argv sha16 b2e18d0d8af33e42 | steps 3000 | ego_perturb_p 0.3 | w_acc 49.6969")
        elif rec == "FAIL":
            out.append(f"DAG-ORC (DAG-OACC base) run record: ['argv --steps 2999'] | {arm} s{s} | trainer seed {s} | argv sha16 b2e | steps 2999 | ego_perturb_p 0.3 | w_acc 49.6969")
        if reach == "PASS":
            out.append(f"DAG-ORC (DAG-OACC base) K-REACH: PASS | {arm} | swap 14688/145037 = 0.1013 (must lie in 0.1 +- 0.02) | ego 23000/76898 = 0.2991 (A: 0.3 +- 0.02; REP: exactly 0) | overlap 0 (A: exactly 0) | flag True")
        elif reach == "FAIL":
            out.append(f"DAG-ORC (DAG-OACC base) K-REACH: FAIL | {arm} | swap 14688/145037 = 0.1013 (must lie in 0.1 +- 0.02) | ego 19161/76898 = 0.2492 | overlap 0 | flag True | A: perturbed fraction outside")
            out.append(f"ZZDAGOACCORC-CONTROL-FAIL {g}: the swap and/or the perturbation did not reach training as registered")
            out.append(f"ZZV8FMINI-STOP reached-control fail {g}")
    return "\n".join(out + (extra_after or [])) + "\n"


def st(text, arm, seed, art=ART):
    P = L.parse_boxlog(text, art)
    return L.run_state(P, arm, seed)[0], P


def t_boxlog():
    real = open(SNAP, encoding="utf-8").read()
    P = L.parse_boxlog(real, ART)
    s0 = L.run_state(P, "dagoaccorc", 0)
    check("REAL r1 log: A s0 reads FAIL (K-REACH FAIL + CONTROL-FAIL + STOP)", s0[0] == "FAIL" and "K-REACH" in s0[1], s0)
    check("REAL r1 log: the other three runs read NOT-PULLED", all(L.run_state(P, a, s)[0] == "NOT-PULLED" for (a, s) in L.RUNS[1:]))
    check("REAL r1 log: meta stops=1 pending=3 pulled=1 done=0", L.log_meta(P) == {"stops": 1, "pending": 3, "pulled": 1, "done": 0}, L.log_meta(P))
    check("REAL r1 log: the run-record line of the FAILED run still parsed as PASS (so the verdict comes from K-REACH, not a missing line)", P["sec"]["M2b_s0_dagoaccorc"]["rec"] is True)
    full = [("dagoaccorc", 0, "PASS", "PASS"), ("dagoaccorcrep", 0, "PASS", "PASS"), ("dagoaccorc", 1, "PASS", "PASS"), ("dagoaccorcrep", 1, "PASS", "PASS")]
    P = L.parse_boxlog(mk_log(full), ART)
    check("synthetic r2 log, 4 x PASS: every run PASS with its pull-line sha", all(L.run_state(P, a, s) == ("PASS", sha(L.run_name(a, s).encode())) for (a, s) in L.RUNS))
    check("... meta: pending 0, done 0 before a DONE line", L.log_meta(P)["pending"] == 0 and L.log_meta(P)["done"] == 0)
    P = L.parse_boxlog(mk_log(full, ["ZZV8FMINI-DONE 10-28 05:00 tip abc: ok"]), ART)
    check("... a DONE line AFTER the last pull -> done=1", L.log_meta(P)["done"] == 1)
    P = L.parse_boxlog("ZZV8FMINI-DONE 10-28 01:00 stale\n" + mk_log(full), ART)
    check("NEG: a DONE line BEFORE the pulls (a stale DONE) -> done=0", L.log_meta(P)["done"] == 0)
    P = L.parse_boxlog(mk_log(full[:3]) + "ZZV8FMINI-DONE x\n", ART)
    check("NEG: DONE with only 3 of 4 pulled -> done=0, pending=1", L.log_meta(P)["done"] == 0 and L.log_meta(P)["pending"] == 1)
    cases = [
        ("NEG: run record not PASS -> FAIL", [("dagoaccorc", 0, "FAIL", "PASS")], "dagoaccorc", 0, "FAIL"),
        ("NEG: K-REACH FAIL (+ CONTROL-FAIL + STOP) -> FAIL", [("dagoaccorcrep", 0, "PASS", "FAIL")], "dagoaccorcrep", 0, "FAIL"),
        ("NEG: pulled, controls not printed yet -> AWAITING-CONTROLS (wait, not a pass)", [("dagoaccorc", 0, None, None)], "dagoaccorc", 0, "AWAITING-CONTROLS"),
        ("NEG: pulled + run record PASS but no K-REACH line -> AWAITING-CONTROLS", [("dagoaccorc", 0, "PASS", None)], "dagoaccorc", 0, "AWAITING-CONTROLS"),
        ("NEG: pulled + K-REACH PASS but no run record -> AWAITING-CONTROLS", [("dagoaccorc", 0, None, "PASS")], "dagoaccorc", 0, "AWAITING-CONTROLS"),
        ("NEG: pulled to ANOTHER directory than this chain reads -> FAIL", [("dagoaccorc", 0, "PASS", "PASS", "/e/elsewhere/M2b_s0_dagoaccorc")], "dagoaccorc", 0, "FAIL"),
        ("NEG: pulled to a *_r1_kreach_stop directory -> FAIL", [("dagoaccorc", 0, "PASS", "PASS", "/e/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc_r1_kreach_stop")], "dagoaccorc", 0, "FAIL"),
    ]
    for name, runs, arm, s, want in cases:
        got = st(mk_log(runs), arm, s)[0]
        check(name, got == want, got)
    got = st(mk_log([("dagoaccorc", 0, None, None)], ["ZZV8FMINI-STOP something broke"]), "dagoaccorc", 0)[0]
    check("NEG: pulled, then a box-2 STOP before both controls printed -> FAIL", got == "FAIL", got)
    # arm / seed confusion
    txt = mk_log([("dagoaccorc", 0, None, None)]).rstrip("\n") + "\n" + "DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorcrep s0 | trainer seed 10 |\n"
    a_, P_ = st(txt, "dagoaccorc", 0)
    check("NEG: REP's run-record line inside A's section -> A is NOT PASS (a section never borrows another run's line) and REP is FAIL (a record with no pull)",
          a_ == "AWAITING-CONTROLS" and L.run_state(P_, "dagoaccorcrep", 0)[0] == "FAIL", (a_, L.run_state(P_, "dagoaccorcrep", 0)))
    txt = mk_log([("dagoaccorc", 0, "PASS", None)]).rstrip("\n") + "\nDAG-ORC (DAG-OACC base) K-REACH: PASS | dagoaccorcrep | swap 1/10 = 0.1 | ego 0/76898 = 0.0 (A: 0.3 +- 0.02; REP: exactly 0)\n"
    a_, P_ = st(txt, "dagoaccorc", 0)
    check("NEG: REP's K-REACH line inside A's section -> A is FAIL (a foreign control line in the section is a defect, and it cannot satisfy A)", a_ == "FAIL", a_)
    # a re-pull restarts the section: old FAIL, new PASS -> PASS
    old = mk_log([("dagoaccorc", 0, "PASS", "FAIL")])
    new = mk_log([("dagoaccorc", 0, "PASS", "PASS")]).split("\n", 1)[1]
    got = st(old + new, "dagoaccorc", 0)
    check("a RE-PULL restarts the section: r1 FAIL followed by an r2 PASS reads PASS (the LAST pull wins)", got[0] == "PASS", got)
    check("... and the old STOP line is BEFORE the new section, so it does not poison it", L.log_meta(L.parse_boxlog(old + new, ART))["stops"] == 1)
    # CRLF
    check("CRLF line endings parse", st(mk_log(full).replace("\n", "\r\n"), "dagoaccorc", 1)[0] == "PASS")
    # robustness: garbage / empty
    check("an empty log: every run NOT-PULLED, pending 4", L.log_meta(L.parse_boxlog("", ART)) == {"stops": 0, "pending": 4, "pulled": 0, "done": 0})
    check("the arm-name prefix trap: 'dagoaccorc' never matches a 'dagoaccorcrep' line", st(mk_log([("dagoaccorcrep", 0, "PASS", "PASS")]), "dagoaccorc", 0)[0] == "NOT-PULLED")


# ------------------------------------------------------------------------------------------------------------------------------- (c) pull directories
def mk_pull(base, arm, seed, commit="1" * 40, mut=None):
    g = L.run_name(arm, seed)
    tg = L.eval_tag(arm, seed)
    d = os.path.join(base, g)
    os.makedirs(d, exist_ok=True)
    sd = seed if arm == "dagoaccorc" else L.REP_SEED[seed]
    argv = ["--arm", "M2b", "--seed", str(sd), "--out", f"D:\\tanitad\\runs\\v8f_mini_full\\{g}", "--init-ckpt", f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\ckpt_last.pt", "--steps", "3000", "--ego-perturb-p",
            "0.3" if arm == "dagoaccorc" else "0", "--dagger-p", "0.1", "--dagger-bank-sha256", L.BANK1_SHA[seed], "--dagger-seed", str(seed), "--w-acc", "49.6969"]
    if arm == "dagoaccorc":
        argv += ["--ego-perturb-seed", str(seed), "--ego-perturb-with-dagger"]
    summ = {"done": True, "steps": 3000, "seed": sd, "levers": {"init_ckpt": {"sha256": L.M2B_SHA[seed]}}}
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "run": g, "binding": {"commit": commit}}
    files = {"ckpt_last.pt": f"ckpt-{g}-{time.time()}".encode(), f"eval_{tg}.json": json.dumps({"ages": "cycle"}).encode(), f"eval_{tg}.npz": b"npz", "selfcheck.json": b"{}", "train_log.jsonl": b"{}\n"}
    if mut:
        mut(argv, summ, tok, files)
    files["argv.json"] = json.dumps({"argv": argv, "sha": "x"}).encode()
    files["summary.json"] = json.dumps(summ).encode()
    files["launch_gate_token.json"] = json.dumps(tok).encode()
    for n, b in files.items():
        open(os.path.join(d, n), "wb").write(b)
    return d, tg


def write_sums(d, tg, drop=None, corrupt=None):
    names = L.sums_names(tg)
    lines = []
    for n in names:
        if n == drop:
            continue
        h = sha(open(os.path.join(d, n), "rb").read())
        lines.append(f"{('0' * 64) if n == corrupt else h} *{n}")
    open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write("\n".join(lines) + "\n")


def t_pull(tmp):
    def run(arm, seed, mut=None, ident=True, expect=None, min_age=0.0, commit="1" * 40, sums=None):
        d, tg = mk_pull(tmp + "/pulls_" + str(time.time_ns()), arm, seed, commit, mut)
        write_sums(d, tg, **(sums or {}))
        ck = sha(open(os.path.join(d, "ckpt_last.pt"), "rb").read())
        return L.pullcheck(d, tg, (arm, seed) if ident else None, expect_sha=ck if expect is True else expect, min_age=min_age), ck

    for arm in ("dagoaccorc", "dagoaccorcrep"):
        (rc, kind, msg), ck = run(arm, 0, expect=True)
        check(f"pullcheck {arm} s0 valid -> COMPLETE, rc 0, returns the ckpt sha", rc == 0 and msg.startswith(ck), (rc, kind, msg[:120]))
    (rc, kind, msg), ck = run("dagoaccorc", 1)
    check("pullcheck A s1 valid -> COMPLETE", rc == 0)

    def m(fn):
        return fn
    wrong = {
        "A: --seed wrong": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--seed") + 1, "7")),
        "A: summary seed wrong": ("dagoaccorc", 0, lambda a, s, t, f: s.__setitem__("seed", 9)),
        "A: --steps 2999": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--steps") + 1, "2999")),
        "A: --ego-perturb-p 0.1": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--ego-perturb-p") + 1, "0.1")),
        "A: composition flag absent": ("dagoaccorc", 0, lambda a, s, t, f: a.remove("--ego-perturb-with-dagger")),
        "A: --ego-perturb-seed wrong": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--ego-perturb-seed") + 1, "5")),
        "A: --w-acc wrong": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--w-acc") + 1, "48")),
        "A: bank sha wrong": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--dagger-bank-sha256") + 1, "0" * 64)),
        "A: --dagger-p 0": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--dagger-p") + 1, "0")),
        "A: init-ckpt of the other seed": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--init-ckpt") + 1, "D:\\x\\M2b_s1\\ckpt_last.pt")),
        "A: summary does not record M2b_s0 as the init": ("dagoaccorc", 0, lambda a, s, t, f: s["levers"]["init_ckpt"].__setitem__("sha256", "f" * 64)),
        "A: --out names another run": ("dagoaccorc", 0, lambda a, s, t, f: a.__setitem__(a.index("--out") + 1, "D:\\x\\M2b_s0_dagoaccorcrep")),
        "A: token names another run": ("dagoaccorc", 0, lambda a, s, t, f: t.__setitem__("run", "M2b_s0_dagoaccorcrep")),
        "A: token verdict FAIL": ("dagoaccorc", 0, lambda a, s, t, f: t.__setitem__("verdict", "FAIL")),
        "A: token binds no commit": ("dagoaccorc", 0, lambda a, s, t, f: t["binding"].__setitem__("commit", "abc")),
        "REP: perturbation on (--ego-perturb-p 0.3)": ("dagoaccorcrep", 0, lambda a, s, t, f: a.__setitem__(a.index("--ego-perturb-p") + 1, "0.3")),
        "REP: composition flag named": ("dagoaccorcrep", 0, lambda a, s, t, f: a.append("--ego-perturb-with-dagger")),
        "REP: --seed 0 (A's seed, not 10)": ("dagoaccorcrep", 0, lambda a, s, t, f: a.__setitem__(a.index("--seed") + 1, "0")),
        "REP s1: --seed 10 (s0's replicate seed, not 11)": ("dagoaccorcrep", 1, lambda a, s, t, f: a.__setitem__(a.index("--seed") + 1, "10")),
    }
    for name, (arm, seed, fn) in wrong.items():
        (rc, kind, msg), _ = run(arm, seed, fn)
        check(f"NEG pullcheck {name} -> WRONG (rc 3)", rc == 3 and kind == "WRONG", (rc, kind, msg[:140]))
    (rc, kind, msg), ck = run("dagoaccorc", 0, commit=L.R1_COMMIT)
    check("NEG pullcheck: token binds the REFUSED r1 commit e7ea20d1 -> WRONG", rc == 3 and "REFUSED" in msg, (rc, msg[:100]))
    os.environ["DORT1_REFUSE_COMMITS"] = "2" * 40
    (rc, kind, msg), ck = run("dagoaccorc", 0, commit="2" * 40)
    check("NEG pullcheck: a commit named in DORT1_REFUSE_COMMITS -> WRONG", rc == 3, (rc, msg[:100]))
    del os.environ["DORT1_REFUSE_COMMITS"]
    (rc, kind, msg), ck = run("dagoaccorc", 0, expect="0" * 64)
    check("NEG pullcheck: the box-2 log's checkpoint sha differs from the pulled file's -> WRONG", rc == 3 and "box-2 log" in msg, (rc, msg[:120]))
    # incomplete
    (rc, kind, msg), _ = run("dagoaccorc", 0, sums={"corrupt": "ckpt_last.pt"})
    check("NEG pullcheck: SHA256SUMS.txt lists a wrong hash for the checkpoint -> INCOMPLETE (a pull still filling / corrupt)", rc == 1, (rc, msg[:100]))
    (rc, kind, msg), _ = run("dagoaccorc", 0, sums={"drop": "train_log.jsonl"})
    check("NEG pullcheck: SHA256SUMS.txt lists 7 of the 8 names -> INCOMPLETE", rc == 1, (rc, msg[:100]))
    (rc, kind, msg), _ = run("dagoaccorc", 0, min_age=3600.0)
    check("NEG pullcheck: SHA256SUMS.txt younger than min-age -> INCOMPLETE", rc == 1, (rc, msg[:100]))
    d, tg = mk_pull(tmp + "/inc", "dagoaccorc", 0)
    check("NEG pullcheck: no SHA256SUMS.txt -> INCOMPLETE", L.pullcheck(d, tg, ("dagoaccorc", 0), min_age=0)[0] == 1)
    write_sums(d, tg)
    os.remove(os.path.join(d, "eval_" + tg + ".npz"))
    check("NEG pullcheck: a listed file missing -> INCOMPLETE", L.pullcheck(d, tg, ("dagoaccorc", 0), min_age=0)[0] == 1)
    d, tg = mk_pull(tmp + "/ref", "dagoaccorc", 0)
    write_sums(d, tg)
    rd = d + "_r1_kreach_stop"
    shutil.copytree(d, rd)
    check("NEG pullcheck: a directory named *_r1_kreach_stop is REFUSED even when it is otherwise COMPLETE", L.pullcheck(rd, tg, None, min_age=0)[0] == 3)
    check("... while the same content under its proper name is COMPLETE (the suffix is the only difference)", L.pullcheck(d, tg, None, min_age=0)[0] == 0)
    bad = L.identity_problems(d, tg, "dagoaccorc", 0, L.OACC_CKPT_SHA[0], json.load(open(d + "/summary.json")), json.load(open(d + "/argv.json")), json.load(open(d + "/launch_gate_token.json")))
    check("NEG identity: a checkpoint equal to DAG-OACC_s0's (nothing trained) is named", any("nothing was trained" in b for b in bad))
    # the real DAG-OACC pull is COMPLETE as a stand-in but is NOT an A run
    rc, kind, msg = L.pullcheck(f"{AR}/v8f-dagoacc/M2b_s0_dagoacc", "M2b-dagoacc_s0", None)
    check("stand-in: DAG-OACC s0's real pull is COMPLETE by content (no identity asked)", rc == 0 and msg.startswith(L.OACC_CKPT_SHA[0]))
    rc, kind, msg = L.pullcheck(f"{AR}/v8f-dagoacc/M2b_s0_dagoacc", "M2b-dagoacc_s0", ("dagoaccorc", 0))
    check("NEG: ... and the same pull is WRONG when asked to be A s0", rc == 3)


# ------------------------------------------------------------------------------------------------------------------------------- (d) K-BUILD
def t_kbuild():
    kb = L.kbuild("unused", MIRROR, L.R1_COMMIT)
    ch = kb["changed"]
    check("K-BUILD (r1 build, REAL git objects): exactly the trainer and the data lib changed; the model and the dagger lib are byte-identical", sorted({h["file"] for h in ch}) == ["stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"] and len(kb["identical_files"]) == 2, (sorted({h["file"] for h in ch}), kb["identical_files"]))
    check("K-BUILD: every hunk default_off, behind the flag, bit_identity_test PASS", ch and all(h["default_off"] is True and h["behind_flag"] == L.COMPOSE_FLAG and h["bit_identity_test"] == "PASS" for h in ch))
    check("K-BUILD: the base blobs are the files DAG-OACC trained (trainer sha256 30819a6f...)", kb["files"]["stack/scripts/v8f_mini_train.py"]["sha256_base"].startswith("30819a6fb947b2f4"))
    spec = importlib.util.spec_from_file_location("dag_orc_read", f"{RD}/dag_orc_read.py")
    R_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R_)
    ok, det = R_.k_build(kb)
    check("K-BUILD record is accepted by the READER's own k_build()", ok is True, det)
    real_git = L._git

    def broken(tag, fn):
        def g(gd, *a, text=False):
            out = real_git(gd, *a, text=text)
            return fn(a, out) if text else out
        L._git = g
        try:
            kb2 = L.kbuild("unused", MIRROR, L.R1_COMMIT)
        finally:
            L._git = real_git
        ok2, det2 = R_.k_build(kb2)
        check(f"NEG K-BUILD ({tag}) -> the reader's k_build() REFUSES", ok2 is False, det2)
    broken("the suite shows 25 passed, not 26", lambda a, o: o.replace("26 passed", "25 passed") if any("RESULTS.md" in x for x in a) else o)
    broken("mutant M13 not killed", lambda a, o: o.replace("M13_off_path_changed: RED (killed)", "M13_off_path_changed: SURVIVED") if any("RESULTS.md" in x for x in a) else o)
    broken("the package's tested train.py sha256 is not the launch blob", lambda a, o: o.replace("6085d4b5c54d2c7e", "0000000000000000") if any("RESULTS.md" in x for x in a) else o)
    broken("the golden's unmodified file is not the base blob", lambda a, o: o.replace("30819a6fb947b2f4", "1111111111111111") if any("RESULTS.md" in x for x in a) else o)
    broken("a mutant survived: 18 of 19 killed", lambda a, o: o.replace("SUMMARY: 19/19 mutants killed; survivors: none", "SUMMARY: 18/19 mutants killed; survivors: M7") if any("RESULTS.md" in x for x in a) else o)
    broken("compose_run2.log reports a failure", lambda a, o: (o + "\n1 failed, 26 passed in 3.2s\n") if any("compose_run2.log" in x for x in a) else o)

    def accepted(tag, fn):      # POSITIVE: the evidence of a re-run for the r2 fix may add tests / mutants; the bar is 'at least the r1 package, no failure, no survivor'
        def g(gd, *a, text=False):
            out = real_git(gd, *a, text=text)
            return fn(a, out) if text else out
        L._git = g
        try:
            kb2 = L.kbuild("unused", MIRROR, L.R1_COMMIT)
        finally:
            L._git = real_git
        ok2, det2 = R_.k_build(kb2)
        check(f"K-BUILD ({tag}) -> still accepted", ok2 is True, det2)
    accepted("a re-run with 28 tests passed and 21 of 21 mutants killed",
             lambda a, o: o.replace("26 passed", "28 passed").replace("19/19 mutants killed", "21/21 mutants killed") if any(("RESULTS.md" in x) or ("compose_run2.log" in x) for x in a) else o)
    kb3 = L.kbuild("unused", MIRROR, L.BASE_COMMIT)
    ok3, _ = R_.k_build(kb3)
    check("K-BUILD with launch == base: nothing changed, record accepted (positive control for the 'identical' branch)", ok3 is True and kb3["changed"] == [])
    try:
        L.kbuild("unused", MIRROR, "0" * 40)
        check("NEG K-BUILD: a launch commit that is not in the mirror raises", False)
    except RuntimeError:
        check("NEG K-BUILD: a launch commit that is not in the mirror raises (the chain STOPs, it does not write a record)", True)
    return R_


# ------------------------------------------------------------------------------------------------------------------------------- (d2) K-BUILD, r2 TWO-STEP mode (--fix-pkg): the fix package re-ran the flag-off test on the launch bytes
FIX_LAUNCH = "8c24823f2b1bd24332148ebe434c870bcf3662f8"      # the r2 launch commit (A2): train.py + data.py changed vs the base, behind --ego-perturb-with-dagger
FIXPKG = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-dag-orc-a2/fix"
FIX_RES, FIX_LOG = f"{FIX_LAUNCH}:{FIXPKG}/RESULTS.md", f"{FIX_LAUNCH}:{FIXPKG}/logs/suite_test_v8f_ego_dagger_compose.log"
R1_RES = f"{L.R1_COMMIT}:{L.COMPOSE_PKG}/RESULTS.md"
GOLD_LAUNCH, GOLD_R1 = f"{FIX_LAUNCH}:{L.GOLDEN_FILE}", f"{L.R1_COMMIT}:{L.GOLDEN_FILE}"
TRAIN_LAUNCH = f"{FIX_LAUNCH}:stack/scripts/v8f_mini_train.py"


def raw_git(*a):
    """an INDEPENDENT read (the git CLI itself, not the lib's wrapper)."""
    import subprocess
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", *a], capture_output=True)
    assert r.returncode == 0, r.stderr[:200]
    return r.stdout


def patched(patches):
    """context manager: `git cat-file blob <spec>` bytes pass through fn for each spec in `patches` = {spec: fn or (fn, nth)}; nth = the first call index (0-based) the patch applies from."""
    import contextlib
    real = L._git

    @contextlib.contextmanager
    def cm():
        calls: dict = {}

        def g(gd, *a, text=False):
            out = real(gd, *a, text=text)
            if not text and a[:2] == ("cat-file", "blob") and a[2] in patches:
                fn, nth = patches[a[2]] if isinstance(patches[a[2]], tuple) else (patches[a[2]], 0)
                k = calls.get(a[2], 0)
                calls[a[2]] = k + 1
                if k >= nth:
                    return fn(out.decode("utf-8", "replace")).encode("utf-8")
            return out
        L._git = g
        try:
            yield
        finally:
            L._git = real
    return cm()


def t_kbuild_fix(R_):
    import contextlib
    import io
    kb = L.kbuild("unused", MIRROR, FIX_LAUNCH, None, FIX_LAUNCH, FIXPKG)
    ch, ev = kb["changed"], kb["evidence"]
    check("R2 K-BUILD (REAL git objects, launch 8c24823f, --fix-pkg): the trainer and the data lib changed, the model and the dagger lib are byte-identical to the base",
          sorted({h["file"] for h in ch}) == ["stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"] and kb["identical_files"] == ["stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_dagger.py"], (sorted({h["file"] for h in ch}), kb["identical_files"]))
    check("R2 K-BUILD: EVERY changed hunk is VERIFIED (bit_identity_test PASS, default_off, behind the flag) and there are hunks", len(ch) > 0 and all(h["default_off"] is True and h["behind_flag"] == L.COMPOSE_FLAG and h["bit_identity_test"] == "PASS" for h in ch), [h["bit_identity_test"] for h in ch][:2])
    ok, det = R_.k_build(kb)
    check("R2 K-BUILD record is accepted by the READER's own k_build()", ok is True, det)
    check("R2 K-BUILD: base blobs are the files DAG-OACC trained (trainer 30819a6f..., data 0e4a8cea...); launch blobs are the fix's (1c888280..., c117e1cf...)",
          kb["files"]["stack/scripts/v8f_mini_train.py"]["sha256_base"].startswith("30819a6fb947b2f4") and kb["files"]["stack/tanitad/data/v8f_mini_data.py"]["sha256_base"].startswith("0e4a8cea43853242")
          and kb["files"]["stack/scripts/v8f_mini_train.py"]["sha256_launch"].startswith("1c88828094d40a2b") and kb["files"]["stack/tanitad/data/v8f_mini_data.py"]["sha256_launch"].startswith("c117e1cf5be08b97"))
    # every evidence path + sha256 is recorded, and each equals an INDEPENDENT read of the same git object
    efs = ev["evidence_files"]
    want = {(L.R1_COMMIT, f"{L.COMPOSE_PKG}/RESULTS.md"), (FIX_LAUNCH, f"{FIXPKG}/RESULTS.md"), (FIX_LAUNCH, f"{FIXPKG}/{L.FIX_SUITE_LOG}")}
    check("R2 K-BUILD evidence: exactly the three package files are recorded (r1 RESULTS.md @ e7ea20d1; fix RESULTS.md and suite log @ 8c24823f), each with its sha256",
          {(e["commit"], e["path"]) for e in efs} == want and all(re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) for e in efs), efs)
    check("R2 K-BUILD evidence: each recorded sha256 == an independent `git cat-file blob | sha256` of that object",
          all(e["sha256"] == sha(raw_git("cat-file", "blob", f"{e['commit']}:{e['path']}")) and e["bytes"] == len(raw_git("cat-file", "blob", f"{e['commit']}:{e['path']}")) for e in efs))
    g = ev["golden"]
    check("R2 K-BUILD evidence: the golden blob is the SAME at the launch commit and at the r1 evidence commit (independent read), and is absent at the base",
          g["sha256_launch"] == g["sha256_r1_evidence_commit"] == sha(raw_git("cat-file", "blob", GOLD_LAUNCH)) == sha(raw_git("cat-file", "blob", GOLD_R1)) and g["sha256_base"] is None and g["r1_evidence_commit"] == L.R1_COMMIT)
    listed = ev["tested_files_listed_in_fix_results"]
    check("R2 K-BUILD evidence: the fix package's three TESTED-file sha256 (two-space form) == the launch blobs (independent read), the base-block lines '(e7ea20d1)' are not read as a listing",
          sorted(listed) == sorted(["stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", L.R1_COMPOSE_TEST]) and all(v == sha(raw_git("cat-file", "blob", f"{FIX_LAUNCH}:{k}")) for k, v in listed.items()), listed)
    check("R2 K-BUILD evidence: step 1 names the golden writer's bytes (30819a6f..., 0e4a8cea...) == the BASE blobs; suite 36 passed (table and log); 27/27 mutants killed, 11 controls rc 0",
          ev["step1_golden_writer"]["unmodified_tip_files"] == {"v8f_mini_train.py": "30819a6fb947b2f4", "v8f_mini_data.py": "0e4a8cea43853242"} and "36 passed in the fix package's table, 36 in its log" in ev["test"] and "27/27 mutants killed; 11 unmutated controls" in ev["mutant"], (ev["test"], ev["mutant"]))
    # the CLI path the chain uses
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = L.main(["kbuild", "--art", "unused", "--git-dir", MIRROR, "--launch", FIX_LAUNCH, "--evidence-commit", FIX_LAUNCH, "--fix-pkg", FIXPKG])
    check("R2 CLI: `kbuild --launch 8c24823f --evidence-commit 8c24823f --fix-pkg <pkg>` exits 0 and prints the same record the function returns", rc == 0 and json.loads(buf.getvalue()) == json.loads(json.dumps(kb)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc2 = L.main(["kbuild", "--art", "unused", "--git-dir", MIRROR, "--launch", FIX_LAUNCH, "--fix-pkg", FIXPKG + "/nonexistent"])
    check("NEG R2 CLI: a fix package that is not in the mirror -> rc 3 (K-BUILD NOT BUILT), no record printed", rc2 == 3 and buf.getvalue().strip() == "", (rc2, buf.getvalue()[:100]))
    # r1 mode is UNCHANGED: the same record as the pre-r2 lib, byte for byte, for three launch commits (and on the r2 launch it is UNVERIFIED: the r1 package's tested sha256 are not the fix's bytes)
    spec = importlib.util.spec_from_file_location("dagorct1_lib_pre_r2", f"{HERE}/dagorct1_lib_pre_r2.py")
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    for lc, tag in ((L.R1_COMMIT, "r1 commit e7ea20d1"), (L.BASE_COMMIT, "base aeb54468"), (FIX_LAUNCH, "r2 launch 8c24823f")):
        new_r = L.kbuild("unused", MIRROR, lc)
        old_r = OLD.kbuild("unused", MIRROR, lc)
        check(f"r1 mode UNCHANGED: kbuild ({tag}) without --fix-pkg == the pre-r2 lib's record, byte for byte", json.dumps(new_r, sort_keys=True) == json.dumps(old_r, sort_keys=True))
    r1_on_r2 = L.kbuild("unused", MIRROR, FIX_LAUNCH)
    okr, _ = R_.k_build(r1_on_r2)
    check("r1 mode on the r2 launch (no --fix-pkg) is UNVERIFIED and the reader REFUSES it (why the r2 mode exists: the r1 package tested the e7ea20d1 bytes)",
          okr is False and r1_on_r2["changed"] and all(h["bit_identity_test"].startswith("UNVERIFIED") and "launch_bytes:" in h["bit_identity_test"] for h in r1_on_r2["changed"]))
    # unit: the listing parser
    h64 = "ab" * 32
    k64 = "cd" * 32
    t1 = f"{h64}  stack/a.py\n{k64} *stack/b.py\r\n{h64}  stack/c.py  (e7ea20d1)\n{h64} stack/d.py\n{h64}  other/e.py\n"
    got, conf = L.listed_shas(t1)
    check("listed_shas: '<sha>  path' (two spaces) and '<sha> *path' both parse (CRLF too); a line with trailing text, a one-space line and a non-stack/ path are NOT listings",
          got == {"stack/a.py": h64, "stack/b.py": k64} and conf == {}, (got, conf))
    got2, conf2 = L.listed_shas(f"{h64}  stack/a.py\n{k64}  stack/a.py\n{h64}  stack/z.py\n{h64} *stack/z.py\n")
    check("listed_shas: one path listed with two DIFFERENT sha256 is a conflict (not silently the first); the same sha listed twice is not", got2 == {"stack/z.py": h64} and conf2 == {"stack/a.py": sorted([h64, k64])}, (got2, conf2))

    def neg(tag, patches, want_key, must=None, **kw):
        with patched(patches):
            kb2 = L.kbuild("unused", MIRROR, FIX_LAUNCH, None, FIX_LAUNCH, FIXPKG, **kw)
        ok2, det2 = R_.k_build(kb2)
        txt = " ".join(h["bit_identity_test"] for h in kb2["changed"])
        keys = set(re.findall(r"(?:UNVERIFIED: |; )((?:fix|r1|golden|flag)[^ ]*?): ", kb2["changed"][0]["bit_identity_test"])) if kb2["changed"] else set()
        single = bool(keys) and all(k.startswith(want_key.rstrip(":")) for k in keys)      # the deliberate regression trips ITS clause only, not an incidental one
        check(f"NEG R2 K-BUILD ({tag}) -> every hunk UNVERIFIED ({want_key}, that clause only) and the reader's k_build() REFUSES",
              ok2 is False and kb2["changed"] and single and (must is None or must in txt) and all(h["bit_identity_test"].startswith("UNVERIFIED") and want_key in h["bit_identity_test"] for h in kb2["changed"]), (txt[:200], sorted(keys), det2 if ok2 else ""))
    tr, dt, tt = "1c88828094d40a2b5e3b5dce45b9fc110286c0b01db3a224faa205138e5c27bc", "c117e1cf5be08b9739f02e2730ce555be5a707a3183ded6481256e419f751752", "faad3be21e1bd4d673352c6fe793d271385839cfa2612c1b1a95e71754a34999"
    neg("the fix package's tested trainer sha256 is not the launch blob", {FIX_RES: lambda t: t.replace(tr, "0" * 64)}, "fix:launch_bytes:stack/scripts/v8f_mini_train.py")
    neg("the fix package's tested data-lib sha256 is not the launch blob", {FIX_RES: lambda t: t.replace(dt, "0" * 64)}, "fix:launch_bytes:stack/tanitad/data/v8f_mini_data.py")
    neg("the fix package's tested COMPOSE TEST file sha256 is not the launch blob", {FIX_RES: lambda t: t.replace(tt, "0" * 64)}, "fix:launch_bytes:stack/tests/test_v8f_ego_dagger_compose.py")
    neg("the trainer is not listed at all (absent != equal)", {FIX_RES: lambda t: "\n".join(x for x in t.split("\n") if tr not in x)}, "fix:launch_bytes:stack/scripts/v8f_mini_train.py")
    neg("the trainer is listed twice with different sha256 (ambiguous)", {FIX_RES: lambda t: t.replace(f"{tr}  stack/scripts/v8f_mini_train.py\n", f"{tr}  stack/scripts/v8f_mini_train.py\n{'0' * 64}  stack/scripts/v8f_mini_train.py\n")}, "fix:launch_bytes:stack/scripts/v8f_mini_train.py", must="ambiguous")
    neg("the trainer's line carries trailing text (so it is not a listing; the base block's '(e7ea20d1)' lines are the only ones left)", {FIX_RES: lambda t: t.replace(f"{tr}  stack/scripts/v8f_mini_train.py\n", f"{tr}  stack/scripts/v8f_mini_train.py  (note)\n")}, "fix:launch_bytes:stack/scripts/v8f_mini_train.py")
    neg("the golden blob CHANGED at the launch commit", {GOLD_LAUNCH: lambda t: t + " "}, "golden:")
    neg("the golden blob differs at the r1 evidence commit", {GOLD_R1: lambda t: t + " "}, "golden:")
    neg("the fix suite log reports a failure", {FIX_LOG: lambda t: t + "\r\n1 failed, 36 passed in 3.0s\r\n"}, "fix:suite_log")
    neg("the fix suite log does not END in a passed line (truncated to its progress line)", {FIX_LOG: lambda t: t.split("\n")[0]}, "fix:suite_log")
    neg("the fix suite log is empty", {FIX_LOG: lambda t: ""}, "fix:suite_log")
    neg("the fix suite log's progress line carries an F while the summary still says 36 passed", {FIX_LOG: lambda t: "F" + t[1:]}, "fix:suite_log")
    neg("the fix suite log counts 25 passed (< 26) and disagrees with the table", {FIX_LOG: lambda t: t.replace("36 passed", "25 passed")}, "fix:suite_log")
    neg("the fix RESULTS.md suite row shows 25 passed (< 26)", {FIX_RES: lambda t: t.replace("| test_v8f_ego_dagger_compose | 36 passed", "| test_v8f_ego_dagger_compose | 25 passed")}, "fix:suite")
    neg("the fix RESULTS.md suite row shows a failure", {FIX_RES: lambda t: t.replace("| test_v8f_ego_dagger_compose | 36 passed in 316.41s (0:05:16) | - |", "| test_v8f_ego_dagger_compose | 36 passed in 316.41s (0:05:16) | 1 |")}, "fix:suite")
    neg("mutant M13 (the flag-off path) not killed", {FIX_RES: lambda t: t.replace("M13_off_path_changed: RED (killed)", "M13_off_path_changed: SURVIVED")}, "fix:mutation")
    neg("mutant M13 is ABSENT from the proof while the SUMMARY is consistent (26/26): only the M13 clause can see it", {FIX_RES: lambda t: "\n".join(x for x in t.split("\n") if not x.startswith("M13_off_path_changed")).replace("SUMMARY: 27/27", "SUMMARY: 26/26")}, "fix:mutation", must="M13_off_path_changed is not shown killed")
    neg("the fix suite log still ENDS in '36 passed' but the line says '1 error' (only the failure / error clause can see it)", {FIX_LOG: lambda t: t.replace("36 passed in 316.41s (0:05:16)", "36 passed, 1 error in 316.41s (0:05:16)")}, "fix:suite_log", must="reports a failure / error")
    neg("a mutant SURVIVED while the SUMMARY still says 27/27 (the per-mutant lines are cross-checked)", {FIX_RES: lambda t: t.replace("M05_loop_never_composes: RED (killed)", "M05_loop_never_composes: SURVIVED")}, "fix:mutation")
    neg("the SUMMARY says 18/18 (< 19 mutants)", {FIX_RES: lambda t: t.replace("SUMMARY: 27/27 mutants killed", "SUMMARY: 18/18 mutants killed")}, "fix:mutation")
    neg("the SUMMARY says 26/27 killed, survivors named", {FIX_RES: lambda t: t.replace("SUMMARY: 27/27 mutants killed; survivors: none", "SUMMARY: 26/27 mutants killed; survivors: M05")}, "fix:mutation")
    neg("an unmutated CONTROL is rc=1 (the 'kill' could be a broken rig)", {FIX_RES: lambda t: t.replace("-k 'flag_off_reproduces': rc=0", "-k 'flag_off_reproduces': rc=1")}, "fix:mutation")
    neg("the r1 package's golden-writing trainer is not the base blob", {R1_RES: lambda t: t.replace("30819a6fb947b2f4", "1111111111111111")}, "r1:golden_writer:stack/scripts/v8f_mini_train.py")
    neg("the r1 package names only 8 hex of the golden-writing data lib (< 16)", {R1_RES: lambda t: t.replace("0e4a8cea43853242", "0e4a8cea")}, "r1:golden_writer:stack/tanitad/data/v8f_mini_data.py")
    neg("the r1 package has no 'Unmodified tip files' section", {R1_RES: lambda t: t.replace("Unmodified tip files", "Some files")}, "r1:golden_writer")
    neg("the launch trainer's flag is not default-off (the check is kept in the r2 mode)", {TRAIN_LAUNCH: (lambda t: t.replace("default=False", "default=True"), 1)}, "flag:")
    for tag, kw, exc in (("the fix package path is not in the mirror", {"fix_pkg": FIXPKG + "/nope"}, RuntimeError), ("the r1 evidence commit is not in the mirror", {"r1_evidence_commit": "0" * 40}, RuntimeError)):
        try:
            L.kbuild("unused", MIRROR, FIX_LAUNCH, None, FIX_LAUNCH, **{"fix_pkg": FIXPKG, **kw})
            check(f"NEG R2 K-BUILD: {tag} raises", False)
        except exc:
            check(f"NEG R2 K-BUILD: {tag} -> raises (NOT BUILT: the chain prints it early and the manifest step refuses; no record is written)", True)
    kb4 = L.kbuild("unused", MIRROR, L.BASE_COMMIT, None, None, FIXPKG + "/nope")
    ok4, _ = R_.k_build(kb4)
    check("R2 mode with launch == base: nothing changed, nothing to attest, the package is not even read; record accepted", ok4 is True and kb4["changed"] == [] and kb4["evidence"] == {})


def t_chain_wiring():
    """the chain script passes the K-BUILD evidence options to EVERY lib call that builds a K-BUILD (the early status, the dry-run manifest, the real manifest), and the .cmd documents the r2 environment"""
    sh = open(f"{HERE}/dagorct1_chain.sh", encoding="utf-8").read()
    calls = [l for l in sh.split("\n") if re.search(r'"\$LIB" (kbuild|manifest) ', l) and "--git-dir" in l]
    check("chain.sh: exactly three lib calls build a K-BUILD from git (early status, dry-run manifest, real manifest)", len(calls) == 3, len(calls))
    check("chain.sh: each of them carries the KBOPTS array (--compose-pkg / --evidence-commit / --fix-pkg / --r1-evidence-commit from DORT1_*)", all('"${KBOPTS[@]}"' in l for l in calls), [l[:80] for l in calls if '"${KBOPTS[@]}"' not in l])
    kb = next((l for l in sh.split("\n") if l.startswith("KBOPTS=")), "")
    check("chain.sh: KBOPTS maps DORT1_FIX_PKG -> --fix-pkg, DORT1_COMPOSE_PKG -> --compose-pkg, DORT1_EVIDENCE_COMMIT -> --evidence-commit, DORT1_R1_EVIDENCE_COMMIT -> --r1-evidence-commit",
          all(x in kb for x in ("DORT1_FIX_PKG:+--fix-pkg", "DORT1_COMPOSE_PKG:+--compose-pkg", "DORT1_EVIDENCE_COMMIT:+--evidence-commit", "DORT1_R1_EVIDENCE_COMMIT:+--r1-evidence-commit")), kb[:200])
    cmd = open(f"{HERE}/dagorct1_chain.cmd", encoding="utf-8").read()
    want = [f'rem   set "DORT1_FIX_PKG={FIXPKG}"', "rem   set DORT1_EXPECT_COMMIT=" + FIX_LAUNCH, "rem   set DORT1_EVIDENCE_COMMIT=" + FIX_LAUNCH, "rem   set DORT1_OUT=F:/TanitAD-scratch/dagorc_t1_out"]
    check("chain.cmd: the r2 environment (DORT1_FIX_PKG quoted because the path holds '&', DORT1_EXPECT_COMMIT, DORT1_EVIDENCE_COMMIT, DORT1_OUT) is documented as `rem` lines for the Master Mind, and NOT set",
          all(w in cmd for w in want) and not re.search(r"^\s*set ", cmd, re.M), [w for w in want if w not in cmd])


# ------------------------------------------------------------------------------------------------------------------------------- (e) manifest vs the reader's slot_needs; verify on a synthetic world
def t_manifest(tmp, R_):
    out = tmp + "/out"
    art = tmp + "/art"
    m = L.build_manifest(art.replace("\\", "/"), out.replace("\\", "/"), f"{AR}/v8f-dagoacc-t1/runs.json", None)
    gaps = []
    for arm in ("A", "REP", "CTRL"):
        for s in ("0", "1"):
            for what, pth in R_.slot_needs(arm, m["arms"][arm][s]):
                if not pth:
                    gaps.append(f"{arm} s{s}: {what}")
    check("manifest: every path the READER's slot_needs() asks of A / REP / CTRL is named (no None)", not gaps, gaps[:5])
    check("manifest: arms are exactly A, REP, CTRL (B absent: not queued)", sorted(m["arms"]) == ["A", "CTRL", "REP"])
    # synthetic world
    commit = "3" * 40
    ckshas = {}
    for arm, s in L.RUNS:
        d, tg = mk_pull(art, arm, s, commit)
        write_sums(d, tg)
        ck = sha(open(os.path.join(d, "ckpt_last.pt"), "rb").read())
        ckshas[(arm, s)] = ck
        ckpath = f"D:\\tanitad\\runs\\v8f_mini_full\\{L.run_name(arm, s)}\\ckpt_last.pt"
        tj = json.load(open(d + "/eval_" + tg + ".json"))
        tj["ckpt"] = ckpath
        open(d + "/eval_" + tg + ".json", "w").write(json.dumps(tj))
        for lim in ("causal", "logged"):
            t = L.t1_entry(out, arm, s, lim)
            os.makedirs(os.path.dirname(t["json"]), exist_ok=True)
            os.makedirs(os.path.dirname(t["ticks"]), exist_ok=True)
            os.makedirs(t["dump_dir"], exist_ok=True)
            os.makedirs(os.path.dirname(t["argv"]), exist_ok=True)
            json.dump({"t1v8f": {"ckpt": f"{art}/{L.run_name(arm, s)}/ckpt_last.pt", "limit": lim, "max_windows": None, "n_windows": 2399}}, open(t["json"], "w"))
            open(t["ticks"], "wb").write(b"t")
            open(t["argv"], "w").write("x\n")
            for i in range(141):
                open(os.path.join(t["dump_dir"], f"ep{i:03d}.npz"), "wb").write(b"e")
            json.dump({"ckpt_sha256": ck, "smoke": False}, open(t["done"], "w"))
        os.makedirs(out + "/t0fresh", exist_ok=True)
        json.dump({"ages": "fresh", "ckpt": f"{art}/{L.run_name(arm, s)}/ckpt_last.pt"}, open(f"{out}/t0fresh/eval_{L.t0f_tag(arm, s)}.json", "w"))
        open(f"{out}/t0fresh/eval_{L.t0f_tag(arm, s)}.npz", "wb").write(b"n")
        json.dump({"ckpt_sha256": ck}, open(f"{out}/done_t0f_{L.t0f_tag(arm, s)}.txt", "w"))
    m = L.build_manifest(art, out, f"{AR}/v8f-dagoacc-t1/runs.json", {"launch_commit": commit})
    bad = L.verify_manifest(m, 141)
    check("verify_manifest on a complete synthetic world -> no problem", bad == [], bad[:4])

    def neg(name, fn, want):
        m2 = json.loads(json.dumps(m))
        undo = fn(m2)
        b2 = L.verify_manifest(m2, 141)
        if callable(undo):
            undo()
        check(f"NEG verify_manifest: {name} -> refused", any(want in x for x in b2), b2[:3])
    t = L.t1_entry(out, "dagoaccorc", 0, "causal")

    def smoke_json():
        j = json.load(open(t["json"]))
        j["t1v8f"]["max_windows"] = 8
        json.dump(j, open(t["json"], "w"))
        return lambda: (j["t1v8f"].__setitem__("max_windows", None), json.dump(j, open(t["json"], "w")))
    neg("A s0's causal rollout json is a --max-windows smoke (max_windows field)", lambda m_: smoke_json(), "max_windows")

    def real_smoke_json():      # what the REAL smoke wrote (smoke/out, 2026-10-28): the harness records n_windows 8 and max_windows None, so n_windows is the effective guard
        j = json.load(open(t["json"]))
        j["t1v8f"]["n_windows"] = 8
        json.dump(j, open(t["json"], "w"))
        return lambda: (j["t1v8f"].__setitem__("n_windows", 2399), json.dump(j, open(t["json"], "w")))
    neg("A s0's causal rollout json has n_windows 8 (the shape the real --max-windows 8 smoke wrote: max_windows None, n_windows 8)", lambda m_: real_smoke_json(), "n_windows")

    def smoke_done():
        json.dump({"ckpt_sha256": ckshas[("dagoaccorc", 0)], "smoke": True}, open(t["done"], "w"))
        return lambda: json.dump({"ckpt_sha256": ckshas[("dagoaccorc", 0)], "smoke": False}, open(t["done"], "w"))
    neg("A s0's done record says smoke=true", lambda m_: smoke_done(), "done record")

    def few_dumps():
        fs = sorted(glob.glob(os.path.join(t["dump_dir"], "ep*.npz")))
        os.rename(fs[0], fs[0] + ".moved")
        return lambda: os.rename(fs[0] + ".moved", fs[0])
    neg("A s0's causal dump dir holds 140 of 141 episodes", lambda m_: few_dumps(), "ep*.npz")

    def wrong_done():
        json.dump({"ckpt_sha256": "0" * 64, "smoke": False}, open(t["done"], "w"))
        return lambda: json.dump({"ckpt_sha256": ckshas[("dagoaccorc", 0)], "smoke": False}, open(t["done"], "w"))
    neg("A s0's done record names another checkpoint", lambda m_: wrong_done(), "done record")
    neg("a train_dir that is a *_r1_kreach_stop directory", lambda m_: m_["arms"]["A"]["0"].__setitem__("train_dir", m_["arms"]["A"]["0"]["train_dir"] + "_r1_kreach_stop"), "REFUSED r1")
    neg("the CTRL slot points at a dir whose checkpoint is not DAG-OACC's", lambda m_: m_["arms"]["CTRL"]["0"].__setitem__("train_dir", art + "/" + L.run_name("dagoaccorc", 0)), "not DAG-OACC_s0")
    neg("an A entry missing", lambda m_: m_["arms"]["A"].pop("1"), "no entry")
    # different commits
    d = f"{art}/{L.run_name('dagoaccorcrep', 1)}"
    tk = json.load(open(d + "/launch_gate_token.json"))
    tk["binding"]["commit"] = "4" * 40
    json.dump(tk, open(d + "/launch_gate_token.json", "w"))
    check("NEG verify_manifest: the four runs bind two different launch commits -> refused", any("different commits" in x for x in L.verify_manifest(m, 141)))
    tk["binding"]["commit"] = L.R1_COMMIT
    json.dump(tk, open(d + "/launch_gate_token.json", "w"))
    check("NEG verify_manifest: a run whose token binds the REFUSED r1 commit -> refused", any("REFUSED commit" in x for x in L.verify_manifest(m, 141)))


# ------------------------------------------------------------------------------------------------------------------------------- (f) rollout-code gate
def t_rollout():
    GATE = ["stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py"]
    r = L.rolloutcode(TREE, MIRROR, L.R1_COMMIT)
    check("rolloutcode (r1 build): the closure holds the harness's v8f_mini and v8f_mini_data and 100+ files", all(g in r["files"] for g in GATE) and len(r["files"]) > 100, len(r["files"]))
    check("rolloutcode (r1 build): v8f_mini.py IDENTICAL, v8f_mini_data.py DIFFERS only in EgoPerturb, which the harness never reaches", r["files"][GATE[0]] == "IDENTICAL" and r["differs"][GATE[1]]["changed"] == ["EgoPerturb"] and r["differs"][GATE[1]]["touched"] == [])
    check("rolloutcode (r1 build): the trainer script is NOT in the closure (the harness never imports it)", "stack/scripts/v8f_mini_train.py" not in r["files"])
    check("rolloutcode (r1 build): the gate on the two gated files passes", L.gate_failures(r, GATE) == [])
    check("NEG rolloutcode: gating a file the harness never loads is a mis-specified gate -> failure", L.gate_failures(r, ["stack/scripts/v8f_mini_train.py"]) == ["stack/scripts/v8f_mini_train.py"])
    check("NEG rolloutcode: gating refc.py (which differs and whose changed name is reached by the static closure) -> failure", L.gate_failures(r, ["stack/tanitad/refs/refc.py"]) == ["stack/tanitad/refs/refc.py"])
    try:
        L.rolloutcode(TREE, MIRROR, "0" * 40)
        check("NEG rolloutcode: a commit not in the mirror raises (never 'nothing differs')", False)
    except RuntimeError:
        check("NEG rolloutcode: a commit not in the mirror raises (never 'every file absent, nothing differs')", True)
    # a mutated tree: make the model file differ and require the gate to fire (deliberate regression on the instrument)
    tmp = tempfile.mkdtemp(prefix="ro_", dir=os.environ.get("DORT1_SELFTEST_TMP"))
    try:
        for rel in ("stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py"):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
        # use the REAL closure of a tiny synthetic tree: harness imports D.foo; launch differs in foo
        os.makedirs(tmp + "/stack/scripts", exist_ok=True)
        os.makedirs(tmp + "/stack/tanitad/data", exist_ok=True)
        os.makedirs(tmp + "/taniteval/tools", exist_ok=True)
        open(tmp + "/stack/tanitad/data/v8f_mini_data.py", "w").write("def foo():\n    return 1\n\ndef bar():\n    return foo()\n\nclass EgoPerturb:\n    pass\n")
        open(tmp + "/stack/tanitad/models/v8f_mini.py", "w").write("X = 1\n") if os.path.isdir(tmp + "/stack/tanitad/models") else None
        open(tmp + "/stack/scripts/v8f_t1_eval.py", "w").write("from tanitad.data import v8f_mini_data as D\nD.bar()\n")
        open(tmp + "/stack/scripts/v8f_mini_eval.py", "w").write("x = 1\n")
        open(tmp + "/taniteval/tools/t1_eval.py", "w").write("y = 2\n")
        sub = L._top("def foo():\n    return 2\n\ndef bar():\n    return foo()\n\nclass EgoPerturb:\n    pass\n")
        base = L._top(open(tmp + "/stack/tanitad/data/v8f_mini_data.py").read())
        changed = sorted(k for k in set(sub) | set(base) if sub.get(k) != base.get(k))
        check("AST diff: changing foo() is seen as a change of foo only (layout / comments ignored)", changed == ["foo"], changed)
        refs = L._refs_in(open(tmp + "/stack/tanitad/data/v8f_mini_data.py").read())
        check("call graph: bar -> foo is followed (the harness reaches bar, hence foo)", "foo" in refs["bar"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    t0 = time.time()
    base = os.environ.get("DORT1_SELFTEST_TMP") or tempfile.gettempdir()
    tmp = tempfile.mkdtemp(prefix="dagorct1_", dir=base).replace("\\", "/")
    print(f"scratch: {tmp}")
    try:
        t_constants()
        t_boxlog()
        t_pull(tmp)
        R_ = t_kbuild()
        t_kbuild_fix(R_)
        t_chain_wiring()
        t_manifest(tmp, R_)
        t_rollout()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"SELFTEST: {R['n'] - len(R['bad'])}/{R['n']} checks as expected ({time.time() - t0:.0f} s)" + ("" if not R["bad"] else "; FAILED: " + "; ".join(R["bad"])))
    print("ZZDAGORCT1-SELFTEST-OK" if not R["bad"] else "ZZDAGORCT1-SELFTEST-FAIL")
    return 0 if not R["bad"] else 1


if __name__ == "__main__":
    sys.exit(main())
