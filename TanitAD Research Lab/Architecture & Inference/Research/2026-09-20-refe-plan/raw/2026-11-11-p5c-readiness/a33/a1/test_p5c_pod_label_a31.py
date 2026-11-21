"""Tests for a1/p5c_pod_label.sh, the A3.1 two-pass runner (PREREG_P5C_A3 section 6 step 2).  CPU only, no pod, no nuPlan, no GPU.
Run (dev box, navsim-crun venv: numpy + pytest; Git bash on PATH):
    CUDA_VISIBLE_DEVICES=-1 TMP=<scratch> python -m pytest -q test_p5c_pod_label_a31.py -p no:cacheprovider

The REAL runner runs under bash in a fake /workspace.  Its absolute paths and the pins of the files that are stubbed here are rewritten in a COPY of its text; the pins of the three files that are
REAL here (traj_label_table.py A2 d65e01c4, p5c_table_gates.py A1 5c2f49cf, p5c_rj5_path.py A3 409f4481) are left as written, so a wrong pin refuses.  The pinned labeller is a STUB with its record
format and CLI (the real one needs nuPlan): the tests prove the PATH (feed -> control pass -> rekey -> gates -> classify -> provenance -> table -> TB-9), not the label math.
Every refusal A3 names has a deliberate-regression arm: the runner text is MUTATED (the refusal removed) and the same scenario must go the other way."""
import glob
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = os.path.abspath(os.path.join(HERE, ".."))
SG = os.path.abspath(os.path.join(HOLD, ".."))
RUNNER = os.path.join(HERE, "p5c_pod_label.sh")


def _first(*cands):
    return next((c for c in cands if os.path.exists(c)), cands[0])


# repo layout (this file at PK/raw/2026-11-11-p5c-readiness/a31/) first, then the dev-box hold layout; env overrides win
PKRAW = os.path.abspath(os.path.join(HERE, ".."))
RJ5PATH = os.environ.get("P5C_A31_RJ5PATH") or _first(os.path.join(PKRAW, "a3", "p5c_rj5_path.py"), os.path.join(SG, "p5c_a3", "a3", "p5c_rj5_path.py"))
GATES = os.environ.get("P5C_A31_GATES") or _first(os.path.join(PKRAW, "a1", "p5c_table_gates.py"), os.path.join(SG, "p5c", "a1", "p5c_table_gates.py"))
TABLE_MOD = os.environ.get("P5C_A31_TABLE_MOD") or _first(os.path.join(PKRAW, "..", "..", "refe", "traj_label_table.py"),
                                                          os.path.join(SG, "p5c_launchkit", "files", "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-09-20-refe-plan", "refe", "traj_label_table.py"))
BASH = shutil.which("bash")
if BASH and os.path.basename(os.path.dirname(BASH)).lower() == "bin":
    real = os.path.join(os.path.dirname(os.path.dirname(BASH)), "usr", "bin", "bash.exe")
    BASH = real if os.path.exists(real) else BASH


def posix(p):
    return str(p).replace("\\", "/")


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def traj_sha(t):
    h = hashlib.sha256()
    for p in t:
        for v in p:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


# ------------------------------------------------------------------ fixture data
LOGS = {"logA": ["a1", "a2", "a3", "a4"], "logB": ["b1", "b2", "b3"], "logC": ["c1", "c2"]}
TWINS = [("logA", "a1"), ("logB", "b2"), ("logC", "c1")]
SRC = {"logA": "A", "logB": "B", "logC": "C"}
EVAL_ROWS = [("logA", "a1", 0), ("logA", "a2", 0), ("logA", "a3", 0), ("logB", "b1", 0), ("logB", "b2", 0), ("logC", "c1", 0), ("logA", "a1", 1), ("logB", "b2", 1)]


def mk_traj(seed, n=20):
    """a gently curving traj: its own acceleration / heading rate are non-zero, so TB-6's ratios (label over the BANK traj) are finite"""
    return [[0.6 * (i + 1) + 0.05 * math.sin(0.35 * i + seed), 0.04 * seed * (i + 1) * 0.1 + 0.03 * math.cos(0.3 * i + seed), 0.02 * math.sin(0.2 * i + seed)] for i in range(n)]


def moved(t, d=0.02):
    return [[p[0] + d * math.sin(0.3 * i + 1.0), p[1] + 0.5 * d * math.cos(0.2 * i), p[2] + 0.01 * d] for i, p in enumerate(t)]


def stub_label(t, big=0.0):
    """the stub labeller's p5r label of a traj; `big` adds a growing acceleration distortion (TB-6 rank-0 FAIL when large)"""
    return [[q[0] + 0.02 * math.cos(0.4 * i) + big * (i % 2) * 5.0, q[1] - 0.01 * math.sin(0.5 * i), q[2] + 0.002] for i, q in enumerate(t)]


def bank_rows():
    r0, r1, s = [], [], 0
    for lg, toks in LOGS.items():
        for t in toks:
            s += 1
            r0.append({"log_name": lg, "token": t, "rank": 0, "sim_hz": 10, "step": 0, "traj": mk_traj(s)})
    for lg, t in TWINS:
        s += 1
        r1.append({"log_name": lg, "token": t, "rank": 1, "sim_hz": 10, "step": 0, "traj": mk_traj(s)})
    return r0, r1


def jwrite(path, objs, gz=False):
    data = "".join(json.dumps(o) + "\n" for o in objs).encode("utf-8")
    if gz:
        with gzip.GzipFile(path, "wb", mtime=0) as g:
            g.write(data)
    else:
        open(path, "wb").write(data)


STUB_DUAL = r'''
import argparse, hashlib, json, os, struct, sys
def sha(t):
    h = hashlib.sha256()
    for p in t:
        for v in p:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()
def rows(d):
    out = []
    for rk in (0, 1):
        for ln in open(os.path.join(d, "targets_rank%d.jsonl" % rk), encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln); out.append((r["log_name"], r["token"], int(r["rank"]), r["traj"]))
    return out
def lab(t, big):
    return [[q[0] + 0.02 * __import__("math").cos(0.4 * i) + big * (i % 2) * 5.0, q[1] - 0.01 * __import__("math").sin(0.5 * i), q[2] + 0.002] for i, q in enumerate(t)]
ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--out"); ap.add_argument("--rank", type=int, default=0); ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
bank = os.environ["P5C_BANKDIR"]
big = float(os.environ.get("STUB_BIG", "0")) if "/feed/" in (bank.replace("\\", "/") + "/") else 0.0        # the distortion applies to the RJ5 (feed) pass only; the control pass reads the bank
real = None
if os.environ.get("STUB_FROM_BANK"):
    real = {(l, t, k): tr for l, t, k, tr in rows(os.environ["STUB_REAL_BANK"])}
R = rows(bank)
if a.mode == "label":
    os.makedirs(a.out, exist_ok=True)
    by = {}
    for lg, tok, rk, tr in R:
        by.setdefault(lg, []).append((tok, rk, tr))
    mine = [lg for i, lg in enumerate(sorted(by)) if i % a.nshard == a.shard]
    tag = "s%dof%d" % (a.shard, a.nshard)
    with open(os.path.join(a.out, "labels_%s.jsonl" % tag), "a", encoding="utf-8", newline="\n") as f, open(os.path.join(a.out, "done_%s.txt" % tag), "a") as dn:
        for lg in mine:
            for tok, rk, tr in by[lg]:
                if real is not None:
                    tr = real[(lg, tok, rk)]
                p = lab(tr, big)
                tc = [[(x + y) / 2 for x, y in zip(q, l)] for q, l in zip(tr, p)]
                f.write(json.dumps({"log": lg, "token": tok, "rank": rk, "traj_sha256": sha(tr), "tc20s": tc, "knot_maxdev": 0.0, "knot_checks": 1, "p5r": p, "p5r_sim8": False,
                                    "p5r_S3_dev": 0.0, "p5r_S1_finite": True, "p5r_S2_yaw_branch": not (rk == 0 and tok in os.environ.get("STUB_S2FAIL", "").split(",") and "/feed/" in (bank + "/").replace(chr(92), "/")), "p5r_S5_lambda_reached": True}) + "\n")
            dn.write(lg + "\n")
    json.dump({"shard": a.shard, "nshard": a.nshard}, open(os.path.join(a.out, "DONE_%s.json" % tag), "w"))
    print("ZZLABEL_DONE", json.dumps({"shard": a.shard}))
    sys.exit(0)
sel = sorted([(hashlib.sha256(("%s|%s" % (lg, tok)).encode()).hexdigest(), lg, tok) for lg, tok, rk, tr in R if rk == a.rank])[: a.n]
rk0 = bool(os.environ.get("STUB_RK0"))
out = [{"log": lg, "token": tok, "raw_sim8": (False if rk0 else i % 2 == 0), "tc20s_sim8": False, "p5r_sim8": bool(rk0), "p5r_sim8_scored_alone": bool(rk0)} for i, (h, lg, tok) in enumerate(sel)]
json.dump({"rank": a.rank, "rule": "stub", "n_requested": a.n, "n": len(out), "n_errors": 0, "rows": out}, open(a.out, "w"))
print("ZZREPAIR", len(out))
'''


class Box:
    """a fake /workspace for the real runner; `opts` craft one scenario"""

    def __init__(self, tmp, join_missing=(), eval_shift=0.0, dev_rj5_shift=0.0, same=(), mutate=None, nshard=2, quota="400000 100000", r0=True, big=0.0, dev_shift=None):
        self.tmp = str(tmp)
        self.big = big
        self.ws = os.path.join(self.tmp, "ws")
        self.ship = os.path.join(self.ws, "data", "p5c_ship")
        self.work = os.path.join(self.ws, "data", "p5c_label")
        self.bank = os.path.join(self.ws, "data", "refe_navtrain", "train_grow_cmd_3c_p7")
        self.cg = os.path.join(self.tmp, "cg")
        for d in (self.ship + "/p5r_eval", self.ship + "/dev_labels", self.bank, self.cg, self.ws + "/data/navtrain_dbs", self.ws + "/data/nuplan-maps/nuplan-maps-v1.0",
                  self.ws + "/refe-plan-vdac/refe", self.ws + "/refe-plan-vdac/code", self.ws + "/venv-teacher/bin"):
            os.makedirs(d, exist_ok=True)
        open(os.path.join(self.cg, "cpu.max"), "w", newline="\n").write(quota + "\n")
        for f in ("navsim_dac.py", "navtrain_scenarios.py"):
            open(os.path.join(self.ws, "refe-plan-vdac", "refe", f), "w").write("# stub\n")
        r0r, r1r = bank_rows()
        self.bank0 = {(r["log_name"], r["token"]): r["traj"] for r in r0r}
        self.bank1 = {(r["log_name"], r["token"]): r["traj"] for r in r1r}
        jwrite(self.bank + "/targets_rank0.jsonl", r0r)
        jwrite(self.bank + "/targets_rank1.jsonl", r1r)
        # the stub labeller + the files the runner pins
        open(self.ship + "/p5c_dual_labels.py", "w", newline="\n").write(STUB_DUAL)
        for f in ("p5c_labels.py", "p5r_labels.py", "p5r_run.py", "p5_target_labels.py", "p4_tracker_comp.py"):
            open(os.path.join(self.ship, f), "w", newline="\n").write("# stub %s\n" % f)
        shutil.copy(RJ5PATH, self.ship + "/p5c_rj5_path.py")
        shutil.copy(GATES, self.ship + "/p5c_table_gates.py")
        shutil.copy(TABLE_MOD, self.ship + "/traj_label_table.py")
        json.dump({"status": "OK", "lambda_star": 3.0}, open(self.ship + "/lambda_star.json", "w"))
        json.dump({"per_rank": {"0": {"TB6_p5r": {"R_acc": 1.0, "R_hr": 1.0}}, "1": {"TB6_p5r": {"R_acc": 1.0, "R_hr": 1.0}}}}, open(self.ship + "/p5r_eval/verdict.json", "w"))
        json.dump({"n": 1018, "bit_identical": 1018, "max_abs_diff": 0.0, "n_errors": 0}, open(self.ship + "/TB1_control_1018.json", "w"))
        # EVAL control: the P5-R labels (and the dev-box Tc20s) of the EVAL rows' BANK trajs (override OFF)
        ev = {0: [], 1: []}
        dev = []
        for lg, tok, rk in EVAL_ROWS:
            tr = (self.bank0 if rk == 0 else self.bank1)[(lg, tok)]
            lab = stub_label(tr)
            lab = [[p[0] + eval_shift, p[1], p[2]] for p in lab]
            ev[rk].append({"log": lg, "token": tok, "rank": rk, "lam": {"3.0": {"label": lab, "sim8": False}}, "raw": {}})
            tc = [[(x + y) / 2 for x, y in zip(q, l)] for q, l in zip(tr, stub_label(tr))]
            if dev_shift and rk in dev_shift:                    # A3.2: the dev-box Tc20s build of this EVAL row sits `dev_shift[rank]` off the pod's (cross-platform floor, one coordinate)
                tc[0][0] += dev_shift[rk]
            dev.append({"log": lg, "token": tok, "rank": rk, "traj_sha256": traj_sha(tr), "tc20s": tc, "knot_maxdev": 0.0, "knot_checks": 1})
        jwrite(self.ship + "/p5r_eval/eval_r0.jsonl", ev[0])
        jwrite(self.ship + "/p5r_eval/eval_r1.jsonl", ev[1])
        jwrite(self.ship + "/dev_labels/labels_s0of2.jsonl", dev)
        jwrite(self.ship + "/dev_labels/labels_s1of2.jsonl", [])
        # the RJ5 join: every rank-0 key (status rj5, or missing for `join_missing`)
        self.n_dev = 0
        dev_rj5 = []
        for (lg, tok), tr in sorted(self.bank0.items()):
            sha = traj_sha(tr)
            o = {"log_name": lg, "token": tok, "rank": 0, "bank_key": "%s|%s|0|%s" % (lg, tok, sha), "bank_traj_sha256": sha}
            if (lg, tok) in join_missing:
                o.update(status="missing", source=None)
            else:
                rt = tr if (lg, tok) in same else moved(tr)
                o.update(status="rj5", source=SRC[lg], rj5_traj_sha256=traj_sha(rt), rj5_traj=rt)
                p = stub_label(rt, big)
                p = [[q[0] + dev_rj5_shift, q[1], q[2]] for q in p]
                dev_rj5.append({"log": lg, "token": tok, "rank": 0, "traj_sha256": traj_sha(rt), "tc20s": [[(x + y) / 2 for x, y in zip(q, l)] for q, l in zip(rt, stub_label(rt, big))], "knot_maxdev": 0.0, "knot_checks": 1,
                                "p5r": p, "p5r_sim8": False, "p5r_S3_dev": 0.0, "p5r_S1_finite": True, "p5r_S2_yaw_branch": True, "p5r_S5_lambda_reached": True})
        self.n_dev = len(dev_rj5)
        jwrite(self.ship + "/rj5_bank_join.jsonl.gz", [json.loads(x) for x in self._join_lines(join_missing, same)], gz=True)
        jwrite(self.ship + "/rj5_labels_local.jsonl.gz", dev_rj5, gz=True)
        if r0:
            os.makedirs(self.tmp + "/r0", exist_ok=True)
            jwrite(self.tmp + "/r0/rj5_ctl.jsonl", [{"log_name": "logA", "token": "a1", "rank": 0, "arm": "R0", "traj": self.bank0[("logA", "a1")]},
                                                    {"log_name": "logB", "token": "b1", "rank": 0, "arm": "R0", "traj": self.bank0[("logB", "b1")]}])
        # the python wrapper the runner calls as $PY: the import probe answers, everything else is the venv's python
        wrapper = '#!/bin/bash\ncase "$*" in *"import sys, nuplan"*) echo "ZZPROBE thread_pin=stub"; exit 0 ;; esac\nexec "%s" "$@"\n' % posix(sys.executable)
        open(self.ws + "/venv-teacher/bin/python", "w", newline="\n").write(wrapper)
        # the runner text with this sandbox's paths and the stubbed files' pins
        t = open(RUNNER, encoding="utf-8").read()
        for old, new in (("SRC_TREE=/workspace/refe-plan-vdac", "SRC_TREE=%s/refe-plan-vdac" % posix(self.ws)), ("BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7", "BANK=%s" % posix(self.bank)),
                         ("DBROOT=/workspace/data/navtrain_dbs", "DBROOT=%s/data/navtrain_dbs" % posix(self.ws)), ("MAPS=/workspace/data/nuplan-maps/nuplan-maps-v1.0", "MAPS=%s/data/nuplan-maps/nuplan-maps-v1.0" % posix(self.ws)),
                         ("PY=/workspace/venv-teacher/bin/python", "PY=%s/venv-teacher/bin/python" % posix(self.ws)), ("EXPECT_ROWS=150603", "EXPECT_ROWS=12"), ("EVAL_N=2400", "EVAL_N=%d" % len(EVAL_ROWS)),
                         ("EXPECT_DEV_RJ5=16133", "EXPECT_DEV_RJ5=%d" % self.n_dev)):
            assert t.count(old) == 1, old
            t = t.replace(old, new)
        pins = {"PIN_BANK_R0": self.bank + "/targets_rank0.jsonl", "PIN_BANK_R1": self.bank + "/targets_rank1.jsonl", "PIN_P5C_LABELS": self.ship + "/p5c_labels.py", "PIN_DUAL": self.ship + "/p5c_dual_labels.py",
                "PIN_P5R_LABELS": self.ship + "/p5r_labels.py", "PIN_P5R_RUN": self.ship + "/p5r_run.py", "PIN_LAMBDA": self.ship + "/lambda_star.json", "PIN_EVAL_R0": self.ship + "/p5r_eval/eval_r0.jsonl",
                "PIN_EVAL_R1": self.ship + "/p5r_eval/eval_r1.jsonl", "PIN_VERDICT": self.ship + "/p5r_eval/verdict.json", "PIN_P5_TARGET": self.ship + "/p5_target_labels.py", "PIN_P4_TRACKER": self.ship + "/p4_tracker_comp.py",
                "PIN_DEV_S0": self.ship + "/dev_labels/labels_s0of2.jsonl", "PIN_DEV_S1": self.ship + "/dev_labels/labels_s1of2.jsonl", "PIN_DEV_RJ5": self.ship + "/rj5_labels_local.jsonl.gz"}
        for k, p in pins.items():
            t, n = re.subn(r"^%s=[0-9a-f]{64}" % k, "%s=%s" % (k, sha256_file(p)), t, flags=re.M)
            assert n == 1, k
        if mutate:
            for old, new in mutate:
                assert t.count(old) == 1, ("mutation target", t.count(old), old[:80])
                t = t.replace(old, new)
        self.runner = os.path.join(self.tmp, "runner.sh")
        open(self.runner, "w", encoding="utf-8", newline="\n").write(t)
        self.nshard = nshard
        self.join_sha = sha256_file(self.ship + "/rj5_bank_join.jsonl.gz")

    def _join_lines(self, join_missing, same):
        out = []
        for (lg, tok), tr in sorted(self.bank0.items()):
            sha = traj_sha(tr)
            o = {"log_name": lg, "token": tok, "rank": 0, "bank_key": "%s|%s|0|%s" % (lg, tok, sha), "bank_traj_sha256": sha}
            if (lg, tok) in join_missing:
                o.update(status="missing", source=None)
            else:
                rt = tr if (lg, tok) in same else moved(tr)
                o.update(status="rj5", source=SRC[lg], rj5_traj_sha256=traj_sha(rt), rj5_traj=rt)
            out.append(json.dumps(o))
        return out

    def env(self, **kw):
        e = {k: v for k, v in os.environ.items() if not k.upper().startswith(("P5CL_", "P5C_", "STUB_"))}
        if self.big:
            e["STUB_BIG"] = str(self.big)
        e.update({"P5CL_LABEL": "p5r-rj5", "P5CL_NSHARD": str(self.nshard), "P5CL_SHIP": posix(self.ship), "P5CL_WORK": posix(self.work), "P5CL_JOIN_SHA256": self.join_sha, "P5CL_A31_SHA256": "ab" * 32,
                  "P5CL_PLATFORMS": "A=devbox B=devbox C=devbox local=devbox", "P5CL_R0_ROWS": "A=%s/r0" % posix(self.tmp), "P5CL_CGROUP_ROOT": posix(self.cg),
                  "STUB_REAL_BANK": posix(self.bank)})
        e.update(kw)
        return {k: v for k, v in e.items() if v is not None}

    def run(self, timeout=600, **kw):
        r = subprocess.run([BASH, posix(self.runner)], capture_output=True, text=True, env=self.env(**kw), timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr


def lines(out, prefix):
    return [ln for ln in out.splitlines() if ln.startswith(prefix)]


def verdict(out):
    for pfx in ("ZZP5CL_DONE", "ZZP5CL_STOP", "ZZP5CL_FAIL"):
        if lines(out, pfx):
            return lines(out, pfx)[0]
    return "NO-VERDICT: " + out[-400:]


@pytest.fixture
def tmpd(tmp_path):
    return tmp_path


# ------------------------------------------------------------------ static
def test_runner_is_lf_bash_n_and_never_pgrep():
    t = open(RUNNER, encoding="utf-8", newline="").read()
    assert "\r" not in t
    assert subprocess.run([BASH, "-n", posix(RUNNER)], capture_output=True).returncode == 0
    code = "\n".join(ln for ln in t.splitlines() if not ln.strip().startswith("#"))
    assert not re.search(r"pgrep|pkill|\bnproc\b\s*\)|\$\(\s*nproc|refe-plan-p7/", code.replace("*refe-plan-p7*", ""))
    assert "ZZTB_ALL_PASS" not in code and "ZZTB_FAIL" not in code.replace("grep -- '^ZZTB_'", "")     # the class is never read from the ZZTB line


# ------------------------------------------------------------------ end to end: the happy path
def test_happy_two_pass_path_builds_the_table_and_every_gate_is_read_per_gate(tmpd):
    b = Box(tmpd)
    rc, out = b.run()
    assert rc == 0 and verdict(out) == "ZZP5CL_DONE", out[-1500:]
    assert lines(out, "ZZP5CL_CLASS rj5=pass")
    g = {ln.split()[2] + "@" + ln.split()[1]: ln.split()[3] for ln in lines(out, "ZZP5CL_GATE")}
    assert g["TB-0@control"] == "PASS" and g["TB-1@control"] == "PASS"
    assert all(g[f"{x}@main"] == "PASS" for x in ("TB-2", "TB-3", "TB-4", "TB-5", "TB-7", "TB-6", "TB-0r", "TB-9"))
    assert g["TB-0@main"] == "NOT_RUN" and g["TB-1@main"] == "NOT_RUN"           # not applicable on the RJ5 table; the ZZTB line of that gate set is therefore a FAIL line, and the run still passes
    assert "ZZTB_FAIL" in out and "ZZP5CL_GATES main" in out
    # the control pass ran on the EVAL rows only, BEFORE the RJ5 pass; the control bank holds exactly the EVAL rows with their bank trajs
    cb = [json.loads(x) for f in sorted(glob.glob(b.work + "/control/bank/targets_rank*.jsonl")) for x in open(f, encoding="utf-8") if x.strip()]
    assert sorted((r["log_name"], r["token"], r["rank"]) for r in cb) == sorted(EVAL_ROWS)
    assert all(r["traj"] == (b.bank0 if r["rank"] == 0 else b.bank1)[(r["log_name"], r["token"])] for r in cb)
    assert out.index("P5CL control pass") < out.index("ZZP5CL_GATE control TB-0") < out.index("P5CL rj5 pass")
    # the table: the pinned reader verifies it; rank policy rank0; the provenance sidecar is bound by sha through the meta
    tl = [ln for ln in out.splitlines() if ln.startswith("ZZP5CL_TABLE p5r-rj5 rank_policy=rank0 ")]
    assert len(tl) == 1
    table = re.search(r"table=(\S+)", tl[0]).group(1)
    sha = re.search(r"sha256=([0-9a-f]{64})", tl[0]).group(1)
    assert sha256_file(table) == sha
    d = json.load(gzip.open(table, "rt", encoding="utf-8"))
    assert d["meta"]["rank_policy"] == "rank0" and d["meta"]["label_name"] == "P5R" and len(d["table"]) == 12
    lm = d["meta"]["labeller_meta"]
    assert lm["rank_policy_gates_decision"] in ("all", "rank0") and lm["a3"]["provenance_sidecar_sha256"] == sha256_file(b.work + "/C1_provenance.jsonl.gz")
    side = [json.loads(x) for x in gzip.open(b.work + "/C1_provenance.jsonl.gz", "rt", encoding="utf-8")]
    assert {s["table_key"] for s in side} == set(d["table"]) and {s["label_source"] for s in side} == {"rj5", "bank_fallback"}
    # rank-1 rows are the identity (label := the bank traj); rank-0 rows carry the RJ5-based label
    for k, lab in d["table"].items():
        lg, tok, rk, _ = k.split("|")
        if rk == "1":
            assert lab == b.bank1[(lg, tok)]
    assert open(b.work + "/P5C_LABEL_TABLE_SHA256_p5r-rj5.txt").read().startswith(sha)


# ------------------------------------------------------------------ end to end: each refusal, and its deliberate-regression arm
def scenario_coverage_stop(b):
    rc, out = b.run()
    return rc == 3 and verdict(out).startswith("ZZP5CL_STOP coverage") and not os.path.exists(b.work + "/control") and not os.path.exists(b.work + "/out")


def scenario_wrong_join_sha(b):
    rc, out = b.run(P5CL_JOIN_SHA256="0" * 64)
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL pin_join")


def scenario_control_gate_fails(b):                    # the control labels differ from the dev EVAL labels: TB-0 FAIL on the control pass: no RJ5 pass is run
    rc, out = b.run()
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL control_gates") and not os.path.exists(b.work + "/out")


def scenario_labeller_saw_bank(b):
    rc, out = b.run(STUB_FROM_BANK="1")
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL rekey")


def scenario_tb6_rank0_stop(b):
    rc, out = b.run()
    return rc == 3 and verdict(out).startswith("ZZP5CL_STOP gates") and not glob.glob(b.work + "/*.json.gz")


def scenario_rk0_stop(b):
    rc, out = b.run(STUB_RK0="1")
    return rc == 3 and verdict(out).startswith("ZZP5CL_STOP gates") and "RK0" in out and not glob.glob(b.work + "/*.json.gz")


def scenario_tb0r_fault(b):
    rc, out = b.run()
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL gates") and "TB-0r" in out and not glob.glob(b.work + "/*.json.gz")


def scenario_over_quota(b):
    rc, out = b.run(P5CL_NSHARD="5")
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL nshard_over_quota")


def scenario_happy(b):
    rc, out = b.run()
    return rc == 0 and verdict(out) == "ZZP5CL_DONE"


def scenario_a31_unset(b):
    rc, out = b.run(P5CL_A31_SHA256="")
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL a31_sha_unset")


CASES = {
    "coverage_stop": (dict(join_missing={("logB", "b3")}), scenario_coverage_stop),
    "wrong_join_sha": (dict(), scenario_wrong_join_sha),
    "control_gate_fails": (dict(eval_shift=0.01), scenario_control_gate_fails),
    "labeller_saw_bank": (dict(), scenario_labeller_saw_bank),
    "tb6_rank0_stop": (dict(big=2.0), scenario_tb6_rank0_stop),
    "rk0_stop": (dict(), scenario_rk0_stop),
    "tb0r_fault": (dict(dev_rj5_shift=0.01), scenario_tb0r_fault),
    "over_quota": (dict(), scenario_over_quota),
    "a31_unset": (dict(), scenario_a31_unset),
}


@pytest.mark.parametrize("name", list(CASES))
def test_refusal_arm(tmpd, name):
    opts, scen = CASES[name]
    assert scen(Box(tmpd, **opts)) is True


def test_no_quota_file_is_a_fault_never_nproc(tmpd):
    b = Box(tmpd)
    os.remove(b.cg + "/cpu.max")
    rc, out = b.run()
    assert rc == 1 and verdict(out).startswith("ZZP5CL_FAIL cpu_quota")
    open(b.cg + "/cpu.max", "w", newline="\n").write("max 100000\n")
    rc, out = b.run()
    assert rc == 1 and verdict(out).startswith("ZZP5CL_FAIL cpu_quota")


def test_label_other_than_p5r_rj5_is_refused(tmpd):
    rc, out = Box(tmpd).run(P5CL_LABEL="p5r")
    assert rc == 1 and verdict(out).startswith("ZZP5CL_FAIL bad_label")


# the deliberate-regression arms: ONE refusal removed from the runner text; the same scenario must then go the other way (scenario returns False)
MUTANTS = [
    ("coverage_bar_lowered", 'MIN_COV=0.99 ', 'MIN_COV=0.50 ', "coverage_stop"),
    ("join_pin_removed", 'pin "$JOIN" "$JOINSHA" pin_join\n', '', "wrong_join_sha"),
    ("control_gates_ignored", '"$PY" "$WORK/classify.py" control "$GATES/control/gates.json" || fail control_gates', '"$PY" "$WORK/classify.py" control "$GATES/control/gates.json" || true #', "control_gate_fails"),
    ("control_pass_on_feed_bank", 'run_shards "$CTL/bank" "$CTL/out" "$EVAL_N" control', 'run_shards "$FEED/bank" "$CTL/out" "$EVAL_N" control', "happy"),
    ("rekey_check_removed", '[ "$RC" = 0 ] && grep -q "^ZZRJ5_REKEY_OK" "$WORK/rekey.log" || fail rekey', 'true || fail rekey', "labeller_saw_bank"),
    ("class_read_from_zztb_line", '  0) ;;\n  3) stop gates', '  0) grep -q "^ZZTB_ALL_PASS" "$GATES/main/gates_stdout.txt" || fail gates_zztb "the ZZTB line" ;;\n  3) stop gates', "happy"),
    ("tb6_rank0_not_a_stop", '    if not r0ok:\n        stop.append("TB-6 rank 0")', '    if False:\n        stop.append("TB-6 rank 0")', "tb6_rank0_stop"),   # + the policy STOP below: the gates' own rule also stops on TB-6 rank 0 (a double guard)
    ("rk0_not_a_stop", '    if dec == "STOP":\n        stop.append("RK0 / rank policy STOP")', '    if False:\n        stop.append("RK0 / rank policy STOP")', "rk0_stop"),
    ("tb0r_not_a_fault", '    if tb0r.get("status") != "PASS":\n        fault.append("TB-0r")', '    if False:\n        fault.append("TB-0r")', "tb0r_fault"),
    ("quota_is_nproc", 'QUOTA="$(cpu_quota)" || fail cpu_quota', 'QUOTA="$(nproc)" || fail cpu_quota', "over_quota"),
    ("a31_check_removed", 'is_sha256 "$A31SHA" || fail a31_sha_unset', 'true || fail a31_sha_unset', "a31_unset"),
]


@pytest.mark.parametrize("name,old,new,case", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_goes_the_other_way(tmpd, name, old, new, case):
    opts, scen = CASES.get(case, (dict(), scenario_happy))
    if case == "happy":
        opts, scen = dict(), scenario_happy
    muts = [(old, new)]
    if name == "tb6_rank0_not_a_stop":                      # both guards removed: TB-6 rank 0 itself AND the policy STOP it also triggers
        muts.append(('    if dec == "STOP":\n        stop.append("RK0 / rank policy STOP")', '    if False:\n        stop.append("RK0 / rank policy STOP")'))
        muts.append(('    elif dec not in ("all", "rank0"):\n        fault.append("RK0 policy %r" % (dec,))', '    elif False:\n        fault.append("RK0 policy %r" % (dec,))'))
    b = Box(tmpd, mutate=muts, **opts)
    assert scen(b) is False, "MUTANT SURVIVED: " + name


# ------------------------------------------------------------------ the classifier alone (TB-9, per-gate verdicts, force-rank0, sidecar vs table), from the runner's own text
def classify_text():
    t = open(RUNNER, encoding="utf-8").read()
    m = re.search(r"# --- CLASSIFY-BEGIN\n(.*?)# --- CLASSIFY-END", t, re.S)
    return "# --- CLASSIFY-BEGIN\n" + m.group(1)


def gates_json(path, **over):
    g = {f"TB-{i}": {"status": "PASS", "detail": {}} for i in range(8)}
    g["TB-0"]["status"] = g["TB-1"]["status"] = "NOT_RUN"
    g["TB-6"]["detail"] = {"rank0": {"pass": True}, "rank1": {"pass": False}}
    g["TB-6"]["status"] = "FAIL"
    pol = {"decision": "rank0"}
    for k, v in over.items():
        if k == "decision":
            pol["decision"] = v
        elif k == "r0pass":
            g["TB-6"]["detail"]["rank0"]["pass"] = v
        else:
            g[k]["status"] = v
    json.dump({"gates": g, "rank_policy": pol}, open(path, "w"))


def manifests(tmp, feed_counts=None, rekey_counts=None, n=12, fb=0, n0=9):
    fc = feed_counts or {"rj5|None|rank0": 9, "bank_fallback|rank1|rank1": 3}
    fc = dict(fc)
    if fb:
        fc["bank_fallback|missing|rank0"] = fb
    rc = rekey_counts or {k + "|p5r": v for k, v in fc.items()}
    json.dump({"counts": fc, "n_bank_rows": n, "n_bank_rank0_keys": n0, "feed_map_sha256": "f" * 64}, open(tmp / "feed_manifest.json", "w"))
    json.dump({"counts": rc, "n_records": n, "feed_map_sha256": "f" * 64}, open(tmp / "rekey_manifest.json", "w"))


def run_classify(tmp, *args):
    p = tmp / "classify.py"
    p.write_text(classify_text(), encoding="utf-8")
    r = subprocess.run([sys.executable, str(p), *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout + r.stderr


def classify_main(tmp, tb0r="PASS", **over):
    gates_json(tmp / "gates.json", **over)
    json.dump({"status": tb0r}, open(tmp / "tb0r.json", "w"))
    return run_classify(tmp, "main", tmp / "gates.json", tmp / "tb0r.json", tmp / "feed_manifest.json", tmp / "rekey_manifest.json", 12, tmp / "class.json")


@pytest.mark.parametrize("over,tb0r,rc", [({}, "PASS", 0), ({"decision": "all"}, "PASS", 0), ({"TB-5": "FAIL"}, "PASS", 1), ({"TB-2": "FAIL"}, "PASS", 1), ({"TB-4": "FAIL"}, "PASS", 1), ({"TB-3": "FAIL"}, "PASS", 1),
                                         ({"TB-7": "FAIL"}, "PASS", 1), ({}, "FAIL", 1), ({"r0pass": False}, "PASS", 3), ({"decision": "STOP"}, "PASS", 3), ({"decision": "UNDECIDED"}, "PASS", 1),
                                         ({"r0pass": False, "TB-5": "FAIL"}, "PASS", 1)])
def test_classifier_per_gate_verdicts(tmp_path, over, tb0r, rc):
    manifests(tmp_path)
    got, out = classify_main(tmp_path, tb0r, **over)
    assert got == rc, (over, tb0r, out)
    assert ("ZZP5CL_CLASS rj5=" + {0: "pass", 3: "stop", 1: "fault"}[rc]) in out


def test_classifier_rank1_fail_alone_decides_nothing_and_tb0_tb1_are_not_judged_on_main(tmp_path):
    manifests(tmp_path)
    got, out = classify_main(tmp_path)
    assert got == 0 and "ZZP5CL_GATE main TB-0 NOT_RUN" in out and "ZZP5CL_GATE main TB-6 PASS rank0=True rank1=False" in out


def test_classifier_tb9_integrity(tmp_path):
    manifests(tmp_path, rekey_counts={"rj5|None|rank0|p5r": 8, "rj5|None|rank0|p5r_error": 0, "bank_fallback|rank1|rank1|p5r": 3})          # one rank-0 record lost between feed and rekey
    assert classify_main(tmp_path)[0] == 1
    manifests(tmp_path, n=11)
    assert classify_main(tmp_path)[0] == 1
    manifests(tmp_path, fb=1, n0=9, feed_counts={"rj5|None|rank0": 8, "bank_fallback|rank1|rank1": 3})                                              # 1 of 9 rank-0 keys is a fallback: over the 1.0 % bar
    got, out = classify_main(tmp_path)
    assert got == 1 and "TB-9 FAIL" in out
    manifests(tmp_path, fb=9, n0=900, feed_counts={"rj5|None|rank0": 891, "bank_fallback|rank1|rank1": 3})                                           # 9 of 900 = exactly 1.0 %: allowed
    assert classify_main(tmp_path)[0] == 0


def test_classifier_control_mode(tmp_path):
    for tb0, tb1, rc in (("PASS", "PASS", 0), ("FAIL", "PASS", 1), ("PASS", "FAIL", 1), ("NOT_RUN", "PASS", 1)):
        json.dump({"gates": {"TB-0": {"status": tb0}, "TB-1": {"status": tb1}, "TB-4": {"status": "FAIL"}}}, open(tmp_path / "g.json", "w"))
        assert run_classify(tmp_path, "control", tmp_path / "g.json")[0] == rc                   # the control pass's other gates (TB-2..7) are not applicable and are never read


def test_classifier_force_rank0_records_the_gates_decision(tmp_path):
    for dec in ("all", "rank0"):
        json.dump({"label_name": "P5R", "rank_policy": dec, "n_rows": 12}, open(tmp_path / "m.json", "w"))
        assert run_classify(tmp_path, "force-rank0", tmp_path / "m.json", tmp_path / "o.json")[0] == 0
        m = json.load(open(tmp_path / "o.json"))
        assert m["rank_policy"] == "rank0" and m["rank_policy_gates_decision"] == dec
    json.dump({"rank_policy": "STOP"}, open(tmp_path / "m.json", "w"))
    assert run_classify(tmp_path, "force-rank0", tmp_path / "m.json", tmp_path / "o.json")[0] == 1


def test_classifier_sidecar_must_equal_table_keys(tmp_path):
    keys = [f"k{i}" for i in range(4)]
    jwrite(str(tmp_path / "side.jsonl.gz"), [{"table_key": k} for k in keys], gz=True)
    with gzip.GzipFile(str(tmp_path / "t.json.gz"), "wb", mtime=0) as g:
        g.write(json.dumps({"meta": {}, "table": {k: [] for k in keys}}).encode())
    assert run_classify(tmp_path, "sidecar", tmp_path / "side.jsonl.gz", tmp_path / "t.json.gz", 4)[0] == 0
    jwrite(str(tmp_path / "side2.jsonl.gz"), [{"table_key": k} for k in keys[:3] + ["other"]], gz=True)
    assert run_classify(tmp_path, "sidecar", tmp_path / "side2.jsonl.gz", tmp_path / "t.json.gz", 4)[0] == 1
    assert run_classify(tmp_path, "sidecar", tmp_path / "side.jsonl.gz", tmp_path / "t.json.gz", 5)[0] == 1


def test_classifier_unit_mutants(tmp_path):
    base = classify_text()
    for old, new, over, tb0r, want_rc in (('for gid in ("TB-2", "TB-3", "TB-4", "TB-5", "TB-7"):', 'for gid in ("TB-2", "TB-3", "TB-4", "TB-7"):', {"TB-5": "FAIL"}, "PASS", 1),
                                          ('    if t9:\n        fault.append("TB-9")', '    if False:\n        fault.append("TB-9")', {}, "PASS", 1),
                                          ('    verdict = "fault" if fault else ("stop" if stop else "pass")', '    verdict = "stop" if stop else ("fault" if fault else "pass")', {"r0pass": False, "TB-5": "FAIL"}, "PASS", 1)):
        assert base.count(old) == 1
        manifests(tmp_path, rekey_counts={"rj5|None|rank0|p5r": 8, "bank_fallback|rank1|rank1|p5r": 3}) if "TB-9" in new or "TB-9" in old else manifests(tmp_path)
        gates_json(tmp_path / "gates.json", **over)
        json.dump({"status": tb0r}, open(tmp_path / "tb0r.json", "w"))
        p = tmp_path / "classify_mut.py"
        p.write_text(base.replace(old, new), encoding="utf-8")
        r = subprocess.run([sys.executable, str(p), "main", str(tmp_path / "gates.json"), str(tmp_path / "tb0r.json"), str(tmp_path / "feed_manifest.json"), str(tmp_path / "rekey_manifest.json"), "12", str(tmp_path / "c.json")],
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode != want_rc, ("MUTANT SURVIVED", old[:60])


# ------------------------------------------------------------------ A3.2: R-A judged on rank-0 rows on the CONTROL pass only (PI ruling 2026-11-16)
def gj(b, where):
    return json.load(open(os.path.join(b.work, "gates", where, "gates.json"), encoding="utf-8"))


# ------------------------------------------------------------------ A3.3 (PI rulings 2026-11-16).  This section SUPERSEDES the A3.2 section that stood here: A3.2 judged R-A on rank-0 rows only (--tc20s-ranks 0) at 1e-9;
# A3.3 judges TB-0's R-A on ALL ranks at 1e-8 (TB-1 stays 1e-9), takes TB-0r to 1e-8, adds the one-row TB-5 label fallback (step 7b) and the re-use of a finished run's shards.
FB_B1 = "b1"                                                    # the fixture row that stands for the ruled row
FB_C2 = "c2"                                                    # a second RJ5 rank-0 row that is NOT ruled


def b1_rj5_sha():
    r0, _ = bank_rows()
    tr = {r["token"]: r["traj"] for r in r0}["b1"]
    return traj_sha(moved(tr))


def fb_mut():
    """the runner's pinned ruled-row constants pointed at the fixture's b1 row (FB_TOKEN / FB_RJ5SHA)"""
    return [("FB_TOKEN=d06ff3cbd9025da7", "FB_TOKEN=b1"), ("FB_RJ5SHA=926ffc6698300e477c7d38e78dd13c0277ffdfb4d59bb91c233f475cd3ff5f91", "FB_RJ5SHA=" + b1_rj5_sha())]


def reship_dev_rj5(b, off, keep_rows=None):
    """the dev-box RJ5 labels: the first record's Tc20s sits `off` away from the pod's (cross-platform floor on RJ5 inputs); the runner's PIN_DEV_RJ5 follows"""
    p = b.ship + "/rj5_labels_local.jsonl.gz"
    recs = [json.loads(x) for x in gzip.open(p, "rt", encoding="utf-8") if x.strip()]
    recs[0]["tc20s"][3][0] += off
    jwrite(p, recs, gz=True)
    t = open(b.runner, encoding="utf-8", newline="").read()
    t, n = re.subn(r"^PIN_DEV_RJ5=[0-9a-f]{64}", "PIN_DEV_RJ5=" + sha256_file(p), t, flags=re.M)
    assert n == 1
    open(b.runner, "w", encoding="utf-8", newline="\n").write(t)


def set_tb1(b, mad):
    json.dump({"n": 1018, "bit_identical": 1000, "max_abs_diff": mad, "n_errors": 0}, open(b.ship + "/TB1_control_1018.json", "w"))


def table_of(out):
    tl = [ln for ln in out.splitlines() if ln.startswith("ZZP5CL_TABLE p5r-rj5 rank_policy=rank0 ")]
    return json.load(gzip.open(re.search(r"table=(\S+)", tl[0]).group(1), "rt", encoding="utf-8"))


def shard_records(b):
    out = {}
    for f in sorted(glob.glob(b.work + "/out/labels_s*of%d.jsonl" % b.nshard)):
        for ln in open(f, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                out[(d["log"], d["token"], d["rank"])] = d
    return out


def scenario_ra_rank1_only(b):
    """both rank-1 EVAL rows' dev-box Tc20s are 1.5e-9 off: A3.3 judges ALL ranks at 1e-8, so the control PASSES; the control scope is 'all' and the exact max (1.5e-9, n 8) is the gate's value"""
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return False
    c = gj(b, "control")
    d = c["gates"]["TB-0"]["detail"]["tc20s_warm_start_vs_dev_build"]
    return c["gates"]["TB-0"]["status"] == "PASS" and c["tc20s_ranks"] == "all" and d["n_common_rows"] == 8 and 1.4e-9 < d["value"] < 1.6e-9 and d["tol"] == 1e-8 and "ranks_judged" not in d


def scenario_ra_rank0_fails(b):
    """a rank-0 EVAL row's dev-box Tc20s is 2e-8 off (over 1e-8): TB-0 FAIL on the control pass, the run stops there"""
    rc, out = b.run()
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL control_gates") and gj(b, "control")["gates"]["TB-0"]["status"] == "FAIL"


def scenario_ra_rank1_over_fails(b):
    """a rank-1 EVAL row's dev-box Tc20s is 2e-8 off: A3.3 judges rank 1 too (A3.2's rank-0 scope is gone), so TB-0 FAIL on the control pass"""
    rc, out = b.run()
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL control_gates") and gj(b, "control")["gates"]["TB-0"]["status"] == "FAIL"


def scenario_ra_5e9_passes(b):
    """both ranks 5e-9 off (over the old 1e-9, under 1e-8): the control PASSES at 1e-8, TB-1 (its own file, 0.0) passes, the run completes"""
    rc, out = b.run()
    return rc == 0 and verdict(out) == "ZZP5CL_DONE" and gj(b, "control")["gates"]["TB-0"]["status"] == "PASS"


def scenario_tb1_stays_1e9(b):
    """the TB-1 control file says max_abs_diff 3e-9: TB-1 stays at 1e-9, so the control FAILS even though TB-0's tolerance is 1e-8"""
    set_tb1(b, 3e-9)
    rc, out = b.run()
    c = gj(b, "control")
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL control_gates") and c["gates"]["TB-1"]["status"] == "FAIL" and c["gates"]["TB-0"]["status"] == "PASS"


def scenario_tb0r_ra_5e9(b):
    """TB-0r: one dev-box RJ5 Tc20s is 5e-9 off (over 1e-9, under 1e-8): TB-0r PASSES at 1e-8 and the run completes"""
    reship_dev_rj5(b, 5e-9)
    rc, out = b.run()
    t = json.load(open(b.work + "/tb0r.json", encoding="utf-8"))
    return rc == 0 and verdict(out) == "ZZP5CL_DONE" and t["status"] == "PASS" and 4e-9 < t["tc20s"]["value"] < 6e-9 and "1e-08" in t["bar"]


def scenario_tb0r_ra_2e8_fails(b):
    """TB-0r: 2e-8 off (over 1e-8): TB-0r FAILS, class fault"""
    reship_dev_rj5(b, 2e-8)
    rc, out = b.run()
    return rc == 1 and lines(out, "ZZP5CL_CLASS rj5=fault") != [] and json.load(open(b.work + "/tb0r.json", encoding="utf-8"))["status"] == "FAIL"


def scenario_ranks_all_both(b):
    """the happy path: the control gates and the main gates are both run with scope 'all'; the control meta records TB-0's tolerance 1e-8 and TB-1's stays the --tc20s-tol 1e-9"""
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return False
    cm = json.load(open(b.work + "/gates/control/table_meta_fields.json", encoding="utf-8"))
    return (gj(b, "control")["tc20s_ranks"] == "all" and gj(b, "main")["tc20s_ranks"] == "all" and cm["p5r"]["tb0_tol_R_A"] == 1e-8 and cm["p5r"]["tc20s_tol_R_A"] == 1e-9
            and "tb0_tol_R_A" not in json.load(open(b.work + "/gates/main/table_meta_fields.json", encoding="utf-8"))["p5r"])


def scenario_no_fallback_when_nothing_fails(b):
    """no row fails S2: n_fallback 0, no fallback key in the sidecar, the meta or the table meta"""
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return False
    man = json.load(open(b.work + "/rekeyed/labels_rekeyed_fb.jsonl.manifest.json", encoding="utf-8"))
    side = [json.loads(x) for x in gzip.open(b.work + "/C1_provenance.jsonl.gz", "rt", encoding="utf-8")]
    d = table_of(out)
    return (man["n_fallback"] == 0 and all("p5r_label_replaced_by" not in s for s in side) and "label_fallback_rows" not in d["meta"]["labeller_meta"]["a3"]
            and "p5r_label_fallback" not in d["meta"]["labeller_meta"] and open(b.work + "/rekeyed/labels_rekeyed_fb.jsonl", "rb").read() == open(b.work + "/rekeyed/labels_rekeyed.jsonl", "rb").read())


def scenario_fallback_ruled_row(b):
    """the ruled row's P5-R label fails S2: it takes its own Tc20s label (marked, counted, in the sidecar and the meta), TB-5 PASSES, the run completes, every other row is untouched"""
    rc, out = b.run(STUB_S2FAIL=FB_B1)
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE" and lines(out, "ZZP5CL_FALLBACK")):
        return False
    man = json.load(open(b.work + "/rekeyed/labels_rekeyed_fb.jsonl.manifest.json", encoding="utf-8"))
    gm = gj(b, "main")["gates"]["TB-5"]
    side = {s["table_key"]: s for s in (json.loads(x) for x in gzip.open(b.work + "/C1_provenance.jsonl.gz", "rt", encoding="utf-8"))}
    d = table_of(out)
    a3 = d["meta"]["labeller_meta"]["a3"]
    recs = shard_records(b)
    key = [k for k in d["table"] if k.startswith("logB|b1|0|")][0]
    tk = man["fallback_rows"][0]["table_key"]
    tc = recs[("logB", "b1", 0)]["tc20s"]
    others_p5r = all(d["table"][k] == recs[(k.split("|")[0], k.split("|")[1], 0)]["p5r"] for k in d["table"] if k.split("|")[2] == "0" and k.split("|")[1] != "b1")
    return (man["n_fallback"] == 1 and man["n_unlisted_s2_candidates"] == 0 and gm["status"] == "PASS" and gm["detail"]["label_fallback"]["n"] == 1 and tk == key
            and side[tk]["fallback_reason"] == "tb5_s2_tc20s" and side[tk]["p5r_label_replaced_by"] == "tc20s"
            and a3["label_fallback_rows"] == {"n": 1, "reasons": ["tb5_s2_tc20s"], "table_keys": [tk]} and d["meta"]["labeller_meta"]["p5r_label_fallback"]["n"] == 1
            and d["table"][key] == tc and tc != recs[("logB", "b1", 0)]["p5r"]
            and sum(1 for s in side.values() if s.get("p5r_label_replaced_by")) == 1 and others_p5r)


def scenario_fallback_unlisted_row_fails(b):
    """b1 is the ruled row, c2 ALSO fails S2: c2 is not ruled, so TB-5 FAILS (class fault, ZZP5CL_FAIL gates) and no table is built"""
    rc, out = b.run(STUB_S2FAIL=FB_B1 + "," + FB_C2)
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL gates") and lines(out, "ZZP5CL_GATE main TB-5 FAIL") != [] and not lines(out, "ZZP5CL_TABLE")


def scenario_pinned_identity_not_matched(b):
    """the runner's own pinned identity (d06ff3cb...) matches no fixture row: a row failing S2 gets NO fallback and TB-5 FAILS"""
    rc, out = b.run(STUB_S2FAIL=FB_B1)
    man = json.load(open(b.work + "/rekeyed/labels_rekeyed_fb.jsonl.manifest.json", encoding="utf-8")) if os.path.exists(b.work + "/rekeyed/labels_rekeyed_fb.jsonl.manifest.json") else {}
    return rc == 1 and verdict(out).startswith("ZZP5CL_FAIL gates") and man.get("n_fallback") == 0 and man.get("n_unlisted_s2_candidates") == 1


CASES["ra_rank1_only"] = (dict(dev_shift={1: 1.5e-9}), scenario_ra_rank1_only)
CASES["ra_rank0_fails"] = (dict(dev_shift={0: 2e-8}), scenario_ra_rank0_fails)
CASES["ra_rank1_over_fails"] = (dict(dev_shift={1: 2e-8}), scenario_ra_rank1_over_fails)
CASES["ra_5e9_passes"] = (dict(dev_shift={0: 5e-9, 1: 5e-9}), scenario_ra_5e9_passes)
CASES["tb1_stays_1e9"] = (dict(), scenario_tb1_stays_1e9)
CASES["tb0r_ra_5e9"] = (dict(), scenario_tb0r_ra_5e9)
CASES["tb0r_ra_2e8_fails"] = (dict(), scenario_tb0r_ra_2e8_fails)
CASES["ranks_all_both"] = (dict(), scenario_ranks_all_both)
CASES["no_fallback_when_nothing_fails"] = (dict(), scenario_no_fallback_when_nothing_fails)
CASES["fallback_ruled_row"] = (dict(mutate=fb_mut()), scenario_fallback_ruled_row)
CASES["fallback_unlisted_row_fails"] = (dict(mutate=fb_mut()), scenario_fallback_unlisted_row_fails)
CASES["pinned_identity_not_matched"] = (dict(), scenario_pinned_identity_not_matched)
A33_CASES = ["ra_rank1_only", "ra_rank0_fails", "ra_rank1_over_fails", "ra_5e9_passes", "tb1_stays_1e9", "tb0r_ra_5e9", "tb0r_ra_2e8_fails", "ranks_all_both", "no_fallback_when_nothing_fails",
             "fallback_ruled_row", "fallback_unlisted_row_fails", "pinned_identity_not_matched"]


@pytest.mark.parametrize("name", A33_CASES)
def test_a33_scenarios_hold(tmpd, name):
    opts, scen = CASES[name]
    assert scen(Box(tmpd, **opts)) is True


def test_a33_runner_text_pins():
    t = open(RUNNER, encoding="utf-8", newline="").read()
    calls = [c for c in re.split(r"\n(?=\"\$PY\" )", t) if c.startswith('"$PY" "$GATES_PY"')]
    assert len(calls) == 2
    ctl = [c for c in calls if '--out-dir "$GATES/control"' in c]
    main = [c for c in calls if '--out-dir "$GATES/main"' in c]
    assert len(ctl) == 1 and len(main) == 1
    assert ctl[0].count('--tc20s-tol "$TB1_TOL"') == 1 and ctl[0].count('--tb0-tol "$TB0_TOL"') == 1 and ctl[0].count("--tc20s-ranks all") == 1 and "--tc20s-ranks 0" not in t
    assert "--tb0-tol" not in main[0] and "--tc20s-ranks" not in main[0] and main[0].count('--tc20s-tol "$TB1_TOL"') == 1 and main[0].count('--tb5-fallback-allow "$FB_TOKEN:$FB_RJ5SHA"') == 1
    assert re.search(r"^TB0_TOL=1e-8\b", t, re.M) and re.search(r"^TB1_TOL=1e-9\b", t, re.M)
    assert re.search(r"^FB_TOKEN=d06ff3cbd9025da7$", t, re.M) and re.search(r"^FB_RJ5SHA=926ffc6698300e477c7d38e78dd13c0277ffdfb4d59bb91c233f475cd3ff5f91$", t, re.M)
    assert t.count('tb0r --gates-module') == 1 and t.count('--tc20s-tol "$TB0_TOL" --out "$WORK/tb0r.json"') == 1
    assert t.count(' fallback --rekeyed ') == 1 and t.count('--allow-row "$FB_TOKEN:$FB_RJ5SHA"') == 1
    assert re.search(r"^PIN_GATES=%s\b" % sha256_file(GATES), t, re.M) and re.search(r"^PIN_RJ5PATH=%s\b" % sha256_file(RJ5PATH), t, re.M)


A33_MUTANTS = [
    ("a33_tb0_tolerance_ignored (TB0_TOL back to 1e-9)", 'TB0_TOL=1e-8 ', 'TB0_TOL=1e-9 ', "ra_5e9_passes"),
    ("a33_tolerance_applied_to_tb1", '--tc20s-tol "$TB1_TOL" --tb0-tol "$TB0_TOL"', '--tc20s-tol "$TB0_TOL" --tb0-tol "$TB0_TOL"', "tb1_stays_1e9"),
    ("a33_rank_scope_back_to_0", '--tb0-tol "$TB0_TOL" --tc20s-ranks all', '--tb0-tol "$TB0_TOL" --tc20s-ranks 0', "ra_rank1_over_fails"),
    ("a33_tb0r_tolerance_not_passed", '--tc20s-tol "$TB0_TOL" --out "$WORK/tb0r.json"', '--out "$WORK/tb0r.json"', "tb0r_ra_5e9"),
    ("a33_fallback_step_skipped", '"$PY" "$RJ5" fallback --rekeyed "$REKEYED_RAW" --out "$REKEYED" --allow-row "$FB_TOKEN:$FB_RJ5SHA" > "$WORK/fallback.log" 2>&1',
     'cp "$REKEYED_RAW" "$REKEYED"; cp "$REKEYED_RAW.manifest.json" "$REKEYED.manifest.json"; echo ZZRJ5_FALLBACK_OK > "$WORK/fallback.log" 2>&1', "fallback_ruled_row"),
    ("a33_gates_allow_list_dropped", ' --tb5-fallback-allow "$FB_TOKEN:$FB_RJ5SHA"', '', "fallback_ruled_row"),
    ("a33_fallback_not_recorded_in_provenance", '--rekey-manifest "$REKEYED.manifest.json"', '--rekey-manifest "$REKEYED_RAW.manifest.json"', "fallback_ruled_row"),
    ("a33_fallback_applied_to_more_rows (tool allow-list widened to every S2 failure by the runner)", '--allow-row "$FB_TOKEN:$FB_RJ5SHA"', '--allow-row "$FB_TOKEN:$FB_RJ5SHA" --allow-row "c2:$FB_RJ5SHA"', "fallback_unlisted_row_fails"),
]


@pytest.mark.parametrize("name,old,new,case", A33_MUTANTS, ids=[m[0] for m in A33_MUTANTS])
def test_a33_mutant_goes_the_other_way(tmpd, name, old, new, case):
    opts, scen = CASES[case]
    muts = list(opts.get("mutate", [])) + [(old, new)]
    opts = {k: v for k, v in opts.items() if k != "mutate"}
    b = Box(tmpd, mutate=muts, **opts)
    assert scen(b) is False, "MUTANT SURVIVED: " + name


# ---- the classifier's own cap on the label fallback (a second guard behind TB-5)
def _fb_manifest(tmp, n_fallback, unlisted=0):
    m = json.load(open(tmp / "rekey_manifest.json"))
    m.update({"n_fallback": n_fallback, "n_unlisted_s2_candidates": unlisted, "fallback_reason": "tb5_s2_tc20s"})
    json.dump(m, open(tmp / "rekey_manifest.json", "w"))


@pytest.mark.parametrize("n,unl,rc", [(0, 0, 0), (1, 0, 0), (2, 0, 1), (1, 1, 1), (0, 3, 1)])
def test_a33_classifier_caps_the_label_fallback(tmp_path, n, unl, rc):
    manifests(tmp_path)
    _fb_manifest(tmp_path, n, unl)
    got, out = classify_main(tmp_path)
    assert got == rc, out
    assert "ZZP5CL_GATE main LABEL-FALLBACK" in out


def test_a33_classifier_cap_mutant(tmp_path):
    base = classify_text()
    old = '    if nfb > 1 or rm.get("n_unlisted_s2_candidates", 0):\n        fault.append('
    assert base.count(old) == 1
    manifests(tmp_path)
    _fb_manifest(tmp_path, 2)
    gates_json(tmp_path / "gates.json")
    json.dump({"status": "PASS"}, open(tmp_path / "tb0r.json", "w"))
    p = tmp_path / "classify_mut.py"
    p.write_text(base.replace(old, '    if False:\n        fault.append('), encoding="utf-8")
    r = subprocess.run([sys.executable, str(p), "main", str(tmp_path / "gates.json"), str(tmp_path / "tb0r.json"), str(tmp_path / "feed_manifest.json"), str(tmp_path / "rekey_manifest.json"), "12", str(tmp_path / "c.json")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, "MUTANT SURVIVED"          # without the guard a 2-row fallback sails through the classifier


# ---- REUSE of a finished run's shards (P5CL_REUSE_SHARDS): only the post-shard stages run
def make_reuse(b, tag="r1"):
    work2 = b.work + "_" + tag
    shutil.copytree(b.work, work2)
    man = os.path.join(b.tmp, "shards_%s.sha256" % tag)
    ln = []
    for sub in ("control/out", "out"):
        for f in sorted(glob.glob("%s/%s/labels_s*of%d.jsonl" % (work2, sub, b.nshard))):
            ln.append("%s  %s/%s" % (sha256_file(f), sub, os.path.basename(f)))
    open(man, "w", newline="\n").write("\n".join(ln) + "\n")
    env = dict(P5CL_WORK=posix(work2), P5CL_REUSE_SHARDS=posix(man), P5CL_REUSE_SHARDS_SHA256=sha256_file(man), P5CL_REUSE_FEED_MAP_SHA256=sha256_file(work2 + "/feed/feed_map.jsonl.gz"))
    return work2, man, env


def scenario_reuse_happy(b):
    rc, out = b.run(STUB_S2FAIL=FB_B1)
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    t1 = table_of(out)
    work2, man, env = make_reuse(b)
    before = {f: sha256_file(f) for f in glob.glob(work2 + "/out/labels_s*") + glob.glob(work2 + "/control/out/labels_s*")}
    rc2, out2 = b.run(STUB_S2FAIL=FB_B1, **env)
    after = {f: sha256_file(f) for f in before}
    if not (rc2 == 0 and verdict(out2) == "ZZP5CL_DONE"):
        return False
    t2 = table_of(out2)
    return (before == after and len(before) == 2 * b.nshard and lines(out2, "ZZP5CL_REUSED control") != [] and lines(out2, "ZZP5CL_REUSED rj5") != [] and "launched" not in out2
            and t1["table"] == t2["table"] and lines(out2, "ZZP5CL_FALLBACK") != [] and lines(out2, "ZZP5CL_REUSE ") != [])


def scenario_reuse_tampered_shard(b):
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    work2, man, env = make_reuse(b)
    f = sorted(glob.glob(work2 + "/out/labels_s0of%d.jsonl" % b.nshard))[0]
    open(f, "a", encoding="utf-8", newline="\n").write("\n")                    # one byte more than the finished run's shard
    rc2, out2 = b.run(**env)
    return rc2 == 1 and verdict(out2).startswith("ZZP5CL_FAIL reuse_sha_mismatch_rj5")


def scenario_reuse_wrong_manifest_pin(b):
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    work2, man, env = make_reuse(b)
    env["P5CL_REUSE_SHARDS_SHA256"] = "0" * 64
    rc2, out2 = b.run(**env)
    return rc2 == 1 and verdict(out2).startswith("ZZP5CL_FAIL pin_reuse_manifest")


def scenario_reuse_wrong_feed_map(b):
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    work2, man, env = make_reuse(b)
    env["P5CL_REUSE_FEED_MAP_SHA256"] = "1" * 64
    rc2, out2 = b.run(**env)
    return rc2 == 1 and verdict(out2).startswith("ZZP5CL_FAIL reuse_feed_map")


def scenario_reuse_in_place(b):
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    work2, man, env = make_reuse(b, "x")
    inplace = os.path.join(os.path.dirname(b.work), "p5c_label_a32")
    shutil.copytree(work2, inplace)
    env["P5CL_WORK"] = posix(inplace)
    rc2, out2 = b.run(**env)
    return rc2 == 1 and verdict(out2).startswith("ZZP5CL_FAIL reuse_in_place")


def scenario_reuse_missing_shard(b):
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    work2, man, env = make_reuse(b)
    os.remove(sorted(glob.glob(work2 + "/control/out/labels_s*of%d.jsonl" % b.nshard))[0])
    rc2, out2 = b.run(**env)
    return rc2 == 1 and verdict(out2).startswith("ZZP5CL_FAIL reuse_missing_control")


def scenario_reuse_off_is_unchanged(b):
    """no P5CL_REUSE_SHARDS: a second run over the same dir does NOT skip the labelling (the old behaviour: the stub appends, the row count is off, the run refuses)"""
    rc, out = b.run()
    if not (rc == 0 and verdict(out) == "ZZP5CL_DONE"):
        return None
    rc2, out2 = b.run()
    return "launched" in out2 and "ZZP5CL_REUSED" not in out2


REUSE_CASES = {"reuse_happy": scenario_reuse_happy, "reuse_tampered_shard": scenario_reuse_tampered_shard, "reuse_wrong_manifest_pin": scenario_reuse_wrong_manifest_pin,
               "reuse_wrong_feed_map": scenario_reuse_wrong_feed_map, "reuse_in_place": scenario_reuse_in_place, "reuse_missing_shard": scenario_reuse_missing_shard,
               "reuse_off_is_unchanged": scenario_reuse_off_is_unchanged}


@pytest.mark.parametrize("name", list(REUSE_CASES))
def test_a33_reuse_scenarios_hold(tmpd, name):
    opts = dict(mutate=fb_mut()) if name == "reuse_happy" else {}
    assert REUSE_CASES[name](Box(tmpd, **opts)) is True


REUSE_MUTANTS = [
    ("a33_reuse_sha_not_checked", '[ "$(sha_of "$f")" = "$want_sha" ] || fail "reuse_sha_mismatch_$tag" "$f differs from the finished run\'s shard"', 'true', "reuse_tampered_shard"),
    ("a33_reuse_manifest_pin_dropped", 'pin "$REUSE" "$REUSE_MANSHA" pin_reuse_manifest', 'true', "reuse_wrong_manifest_pin"),
    ("a33_reuse_feed_map_not_checked", '[ -z "$REUSE" ] || [ "$MAPSHA" = "$REUSE_MAPSHA" ] || fail reuse_feed_map', 'true #', "reuse_wrong_feed_map"),
    ("a33_reuse_in_place_allowed", 'case "$WORK" in *p5c_label_a32|*p5c_label_a32/) fail reuse_in_place', 'case "$WORK" in *NEVER_MATCHES_X) fail reuse_in_place', "reuse_in_place"),
    ("a33_reuse_shard_presence_not_checked", '[ -f "$f" ] || fail "reuse_missing_$tag" "$f"', 'true', "reuse_missing_shard"),
]


@pytest.mark.parametrize("name,old,new,case", REUSE_MUTANTS, ids=[m[0] for m in REUSE_MUTANTS])
def test_a33_reuse_mutant_goes_the_other_way(tmpd, name, old, new, case):
    b = Box(tmpd, mutate=[(old, new)])
    got = REUSE_CASES[case](b)
    assert got is False, "MUTANT SURVIVED: %s (scenario returned %r)" % (name, got)
