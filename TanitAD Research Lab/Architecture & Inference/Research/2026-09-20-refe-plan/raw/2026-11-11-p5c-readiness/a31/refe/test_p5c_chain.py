#!/usr/bin/env python3
"""Tests for the P5-C pod chain refe/p5c_pod.sh (PREREG_P5C 10.1 / 10.4; RUNBOOK_P5C section 8).  Stub-sandbox style of test_p6_chain.py / test_p7_chain.py: the REAL chain runs under bash in a fake /workspace whose
python is a recording stub (no pod, no GPU, no network, no NAVSIM stack).

  1. happy path: the stage ORDER (precheck, label, table verify, MM pin check, lab, mem, egr, then per arm gate mint + verify, preflight, train, post-training G-ADAM, egress), C1 then C0, C0r only with --with-c0r,
     the argv of each call, markers keyed and RESUMABLE (a re-run executes nothing but the precheck), the dry run (prints every command, runs none, touches nothing).
  2. refusals, each with its own arm: unfilled MM pins (NEVER filled by the chain; a FAIL marker is written and the next run refuses until --clear-fail), pins != measured, an existing ckpt_last.pt, an existing FAIL marker,
     P5CL_NSHARD > floor(the cgroup CPU quota) (default = the quota; A3.1: NEVER nproc, a stub `nproc` printing 128 sits first on PATH), P5C_LABEL / label workers unset, a token that does not verify (the trainer is never started), a failed post-training gate (no egress, C0 not started), an unverified egress,
     a killed previous attempt, a table that does not cover every row.
  3. static properties (no token or pin is ever written by the chain) and behavioural MUTANTS of the chain text: each must make its refusal arm go the other way.
  4. A3.1 (PREREG_P5C_A3 sections 8, 9): the cgroup shard cap (v2 cpu.max, v1 cfs, no quota, 'max'), and P5C_LABEL=p5r-rj5 refusing without the A3.1 sha256 / the join sha256 / a match, each with a deliberate-regression arm.
Run: python refe/test_p5c_chain.py  -> ZZP5CCHAIN_PASS <n>   (needs bash on PATH; Git Bash on the dev box).  Never prints ZZTEST_SKIP.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE / "p5c_pod.sh"
NPASS = 0
FAILS: list = []


def ok(cond, name=""):
    global NPASS
    if cond:
        NPASS += 1
    else:
        FAILS.append(name)
        print("  FAIL:", name, flush=True)


def posix(p) -> str:
    return Path(p).as_posix()


def bash_exe() -> str:
    b = shutil.which("bash")
    p = Path(b)
    if p.parent.name.lower() == "bin" and p.parent.parent.name.lower() == "git":
        real = p.parent.parent / "usr" / "bin" / "bash.exe"      # the Git launcher prepends its own /usr/bin and shadows the sandbox's stubs
        if real.exists():
            return str(real)
    return b


NEED = ["refe/train.py", "refe/launch_gate.py", "refe/gate_token.py", "refe/gate_checks.py", "refe/gate_fixture.py", "refe/hf_egress.py", "refe/traj_label_table.py", "refe/p5c_gate.py", "refe/p5c_postgate.py",
        "refe/p6_prep.py", "refe/test_p5c_live.py", "refe/test_p5c_postgate.py", "refe/test_p5c_profile.py", "refe/test_traj_label_table.py", "SHIP_MANIFEST.json"]

FAKEPY = r'''
import json, os, sys, hashlib
a = sys.argv[1:]
calls = os.environ["STUB_CALLS"]
def rec(s):
    open(calls, "a", encoding="utf-8").write(s + "\n")
def opt(flag):
    return a[a.index(flag) + 1] if flag in a else None
E = os.environ.get
if not a:
    sys.exit(0)
if a[0] == "-":                                  # a heredoc script: identify it by its text
    src = sys.stdin.read()
    if "tree_manifest" in src:
        rec("PY tree-check"); print("ZZP5C_TREE_OK 15 files commit deadbeef"); sys.exit(1 if E("STUB_TREE_BAD") else 0)
    if "p5c_argv" in src:
        arm = E("P5C_ARM"); t = E("P5C_TABLE_PATH")
        rec("PY argv-build " + arm)
        argv = ["--backbone", "vitl16", "--init-from", E("P5C_BASE_PATH")] + (["--traj-label-table", t] if arm == "C1" and t else []) + ["--seed", "1" if arm == "C0r" else "0", "--out", E("P5C_DATA") + "/refe_runs/p5c_" + arm.lower(), "--resume"]
        print(" ".join(argv)); sys.exit(0)
    if "P5C_BASE_SHA256" in src:
        rec("PY pins-check " + " ".join(a[2:]))
        m = E("STUB_PINS", "ok")
        if m == "unfilled":
            print("UNFILLED P5C_BASE_SHA256 P5C_LABEL_TABLE_SHA256 P5C_RANK_POLICY"); sys.exit(3)
        if m == "mismatch":
            print("MISMATCH P5C_BASE_SHA256: pinned 0000, measured " + a[2][:4]); sys.exit(4)
        print("ZZP5C_PINS_OK"); sys.exit(0)
    rec("PY heredoc-unknown"); sys.exit(0)
if a[0] == "-c":                                  # the chain's own assertion one-liners are EXECUTED (so a mutant that weakens one is seen)
    rec("PY -c " + ("table-counts" if "d['rows']" in a[1] else "summary-check" if "d['done']" in a[1] else "other"))
    sys.argv = ["-c"] + a[2:]
    try:
        exec(compile(a[1], "<chain -c>", "exec"), {})
    except (AssertionError, SystemExit, KeyError, OSError) as e:
        print("ASSERT", type(e).__name__, str(e)[:200]); sys.exit(1)
    sys.exit(0)
script = os.path.basename(a[0]); rest = a[1:]
def flag(f):
    return rest[rest.index(f) + 1] if f in rest else None
if script == "launch_gate.py":
    sub, prof = rest[0], flag("--profile")
    if sub == "run":
        rec("LG run %s" % prof); od = flag("--out-dir"); os.makedirs(od, exist_ok=True)
        open(os.path.join(od, "PASS_stub.json"), "w").write(json.dumps({"profile": prof, "args": flag("--args")}))
        print("  G-LIVE PASS"); sys.exit(0)
    if sub == "verify":
        cf = calls + ".verifies"; n = (int(open(cf).read()) if os.path.exists(cf) else 0) + 1; open(cf, "w").write(str(n))
        rec("LG verify %s" % prof)
        bad = E("STUB_VERIFY_FAIL")
        if E("STUB_VERIFY_FROM") and n >= int(E("STUB_VERIFY_FROM")):
            sys.exit(1)
        sys.exit(1 if bad and (bad == "all" or bad == prof) else 0)
if script == "train.py":
    out = flag("--out"); pre = "--preflight" in rest
    rec("TRAIN %s%s out=%s" % ("preflight " if pre else "", "c1" if "--traj-label-table" in rest else "other", os.path.basename(out or "")))
    if pre:
        print("PREFLIGHT_OK"); sys.exit(0)
    if "mem_run" in (out or ""):
        os.makedirs(out, exist_ok=True); open(os.path.join(out, "metrics.jsonl"), "w").write("{}\n"); open(os.path.join(out, "ckpt_last.pt"), "w").write("x"); print("TRAIN_DONE"); sys.exit(0)
    if E("STUB_TRAIN_CRASH") and os.path.basename(out) == E("STUB_TRAIN_CRASH"):
        os.makedirs(out, exist_ok=True); open(os.path.join(out, "ckpt_last.pt"), "w").write("dead"); sys.exit(7)
    os.makedirs(out, exist_ok=True)
    for f in ("model_final.pt", "ckpt_last.pt", "config.json", "metrics.jsonl"):
        open(os.path.join(out, f), "w").write("x")
    open(os.path.join(out, "summary.json"), "w").write(json.dumps({"done": True, "steps": 1999 if E("STUB_SUMMARY_BAD") else 2000}))
    print("TRAIN_DONE"); sys.exit(0)
if script == "p5c_postgate.py":
    rec("POSTGATE %s other=%s nfrom=%s" % (os.path.basename(a[1]), os.path.basename(flag("--other") or "-"), "yes" if flag("--n-params-from") else "no"))
    if E("STUB_POSTGATE_FAIL") and os.path.basename(a[1]) == E("STUB_POSTGATE_FAIL"):
        print("ZZP5C_POSTGATE_FAIL 3 registered parameters have NO Adam state"); sys.exit(1)
    print("ZZP5C_POSTGATE_OK 18580000 trainable parameters"); sys.exit(0)
if script == "hf_egress.py":
    sub = rest[0]; rec("EGRESS %s %s" % (sub, flag("--prefix") or ""))
    if sub == "gate":
        od = flag("--out"); os.makedirs(od, exist_ok=True); open(os.path.join(od, "g_egr_result.json"), "w").write("{}"); sys.exit(0)
    if not E("STUB_PUSH_NO_OK"):
        open(flag("--chain-log"), "a").write("2026-01-01T00:00:00 ZZEGRESS_OK %s 12345 bytes\n" % flag("--tag"))
    sys.exit(0)
if script == "p6_prep.py":
    sub = rest[0]; rec("PREP " + sub)
    if sub == "sample-tokens":
        print("ZZTOKENS tok1,tok2")
    elif sub == "mem-argv":
        print("ZZMEM_ARGV --out " + flag("--out") + " --steps 4")
    elif sub == "egr-ok":
        print("ZZEGR_OK")
    sys.exit(0)
if script == "diag_p6_labeller_identity.py":
    rec("LAB diag workers=" + flag("--workers")); open(flag("--out"), "w").write("{}"); print("ZZLAB_IDENTITY_PASS"); sys.exit(0)
if script == "traj_label_table.py":
    rec("TABLE " + rest[0]); print(json.dumps({"rows": 150603, "covered": 150000 if E("STUB_COUNTS_BAD") else 150603, "refused": None, "rank_policy": "all"})); sys.exit(0)
rec("PY unknown " + " ".join(a)); sys.exit(0)
'''

LABEL_SH = '''#!/bin/bash
lab="${P5CL_LABEL:-p5r}"
echo "LABEL nshard=$P5CL_NSHARD" >> "$STUB_CALLS"
echo "LABELENV label=$lab join=${P5CL_JOIN_SHA256:-} a31=${P5CL_A31_SHA256:-} plat=${P5CL_PLATFORMS:-} r0=${P5CL_R0_ROWS:-}" >> "$STUB_CALLS"
mkdir -p "$P5CL_WORK"; echo "table-bytes" > "$P5CL_WORK/P5C_label_table_${lab}_all.json.gz"
sha=$(sha256sum "$P5CL_WORK/P5C_label_table_${lab}_all.json.gz" | cut -c1-64)
[ -n "${STUB_LABEL_STOP:-}" ] && { echo "ZZP5CL_STOP both_labels_fail"; exit 3; }
echo "ZZP5CL_LABEL $lab"
echo "ZZP5CL_TABLE $lab rank_policy=all sha256=$sha table=$P5CL_WORK/P5C_label_table_${lab}_all.json.gz"
echo "ZZP5CL_DONE"
'''


class Sandbox:
    def __init__(self, root, script_text=None):
        self.root = Path(root)
        self.ws = self.root / "ws"
        self.data = self.ws / "data"
        self.pkg = self.ws / "refe-plan-p5c"
        self.stub = self.root / "stub"
        self.calls = self.root / "calls.log"
        self.calls.write_text("", encoding="utf-8")
        (self.pkg / "refe").mkdir(parents=True)
        self.stub.mkdir()
        self.script = self.pkg / "refe" / "p5c_pod.sh"
        self.script.write_text(script_text if script_text is not None else CHAIN.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        for f in NEED:
            (self.pkg / f).write_text("x", encoding="utf-8")
        (self.stub / "fakepy.py").write_text(FAKEPY, encoding="utf-8")
        (self.ws / "venv" / "bin").mkdir(parents=True)
        (self.ws / "venv" / "bin" / "python").write_text(f'#!/bin/bash\nexec "{posix(sys.executable)}" "{posix(self.stub / "fakepy.py")}" "$@"\n', encoding="utf-8", newline="\n")
        (self.stub / "flock").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8", newline="\n")
        (self.stub / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8", newline="\n")
        (self.stub / "nproc").write_text("#!/bin/bash\necho 128\n", encoding="utf-8", newline="\n")          # the pod's nproc: 128 while the cgroup grants 15.3 CPUs; any use of it breaks a test
        self.cg = self.root / "cg"
        self.cg.mkdir()
        (self.cg / "cpu.max").write_text("1530000 100000\n", encoding="utf-8", newline="\n")
        ship = self.data / "p5c_ship"
        ship.mkdir(parents=True)
        (ship / "p5c_pod_label.sh").write_text(LABEL_SH, encoding="utf-8", newline="\n")
        (ship / "traj_label_table.py").write_text("x", encoding="utf-8")
        for d in ("refe_runs/p7_reserve_head", "refe_navtrain/train_grow_cmd_3c_p7", "navtrain_pixels", "refe_navtrain/metric_cache"):
            (self.data / d).mkdir(parents=True, exist_ok=True)
        (self.data / "refe_runs/p7_reserve_head/model_final.pt").write_text("base-model-bytes", encoding="utf-8")
        (self.data / "refe_runs/p7_reserve_head/config.json").write_text(json.dumps({"params": {"trainable": 18580000}}), encoding="utf-8")
        (self.data / "refe_navtrain/ego_yaw_navtrain.json.gz").write_text("y", encoding="utf-8")
        self.hf = self.root / "hf_token.txt"
        self.hf.write_text("hf_" + "a" * 20, encoding="utf-8")
        subprocess.run([bash_exe(), "-c", "; ".join(f"chmod +x '{posix(p)}'" for p in (self.stub / "flock", self.stub / "nvidia-smi", self.stub / "nproc", self.ws / "venv" / "bin" / "python", ship / "p5c_pod_label.sh"))], check=True)
        self.root_dir = self.data / "refe_p5c"

    def env(self, **kw):
        e = {k: v for k, v in os.environ.items() if not k.upper().startswith(("P5C_", "P5CL_", "STUB_", "TANITAD_GATE"))}
        e.update({"P5C_WS": posix(self.ws), "P5C_LABEL": "p5r", "P5C_LABEL_WORKERS": "13", "P5C_HF_TOKEN_FILE": posix(self.hf), "P5C_CGROUP_ROOT": posix(self.cg), "P5C_NVSMI": posix(self.stub / "nvidia-smi"),
                  "P5C_POLL_S": "0", "STUB_CALLS": posix(self.calls), "PATH": posix(self.stub) + os.pathsep + e.get("PATH", ""), "TANITAD_GATE_KEY_FILE": posix(self.root / "key")})
        e.update(kw)
        return e

    def run(self, *args, timeout=240, **kw):
        r = subprocess.run([bash_exe(), posix(self.script), *args], capture_output=True, text=True, env=self.env(**kw), timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr

    def calls_list(self):
        return [ln for ln in self.calls.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def clear_calls(self):
        self.calls.write_text("", encoding="utf-8")

    def chain_log(self):
        p = self.root_dir / "chain.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def stage(self, n):
        return self.root_dir / "stage" / n


def seq(calls, *needles):
    """True iff the needles occur in `calls` in this order (each at a later index than the previous)."""
    i = -1
    for n in needles:
        j = next((k for k in range(i + 1, len(calls)) if n in calls[k]), None)
        if j is None:
            return False
        i = j
    return True


def count(calls, needle):
    return sum(1 for c in calls if needle in c)


def with_box(fn):
    td = Path(tempfile.mkdtemp(prefix="p5cch_"))
    try:
        return fn(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ 1. the happy path
def t_happy():
    def f(td):
        sb = Sandbox(td)
        rc, out = sb.run()
        c = sb.calls_list()
        ok(rc == 0 and "ZZP5C_CHAIN_DONE arms [C1 C0]" in out, ("happy rc", rc, out[-800:]))
        ok(seq(c, "PY tree-check", "LABEL nshard=15", "TABLE verify", "PY -c table-counts", "PY pins-check", "PREP sample-tokens", "LAB diag workers=13", "PREP mem-argv", "TRAIN other out=mem_run", "EGRESS gate", "PREP egr-ok",
               "LG run refe_p5c", "LG verify refe_p5c", "TRAIN preflight c1", "TRAIN c1 out=p5c_c1", "POSTGATE p5c_c1 other=- nfrom=yes", "EGRESS push p5c/c1",
               "LG run refe_p5c_c0", "LG verify refe_p5c_c0", "TRAIN preflight other", "TRAIN other out=p5c_c0", "POSTGATE p5c_c0 other=p5c_c1 nfrom=yes", "EGRESS push p5c/c0"), ("order", c))
        ok(count(c, "LG run refe_p5c_c0r") == 0 and count(c, "p5c_c0r") == 0, "no C0r without --with-c0r")
        ok(count(c, "PY pins-check") == 1 and "PY pins-check" in " ".join(c) and any(re.search(r"pins-check [0-9a-f]{64} [0-9a-f]{64} all", x) for x in c), "the pin check receives the MEASURED base sha, table sha and policy")
        ok(count(c, "TRAIN c1 out=p5c_c1") == 1 and count(c, "TRAIN other out=p5c_c0") == 1, "each arm trains exactly once (the C1 call carries the table flag, C0's does not)")
        # the token is verified BEFORE the trainer (twice: gate stage, preflight, train): at least one verify precedes each launch
        ok(seq(c, "LG run refe_p5c", "LG verify refe_p5c", "TRAIN preflight c1", "LG verify refe_p5c", "TRAIN c1 out=p5c_c1"), "a token verify precedes the preflight AND the trainer start")
        names = sorted(p.name for p in (sb.root_dir / "stage").glob("*.ok"))
        want = sorted(f"{x}.ok" for x in ("inputs", "label", "table", "pins", "lab", "mem", "egr", "gate_C1", "preflight_C1", "train_C1", "postgate_C1", "egress_C1", "trainstart_C1", "gate_C0", "preflight_C0", "train_C0",
                                          "postgate_C0", "egress_C0", "trainstart_C0"))
        ok(names == want, (names, want))
        ok(all(len((sb.root_dir / "stage" / n).read_text(encoding="utf-8").splitlines()) == 2 for n in names), "a marker is two lines: date + key")
        ok((sb.root_dir / "model_final_sha256_C1.txt").exists() and (sb.root_dir / "model_final_sha256_C0.txt").exists(), "both arms' model_final.pt hashed for the ONE sealed manifest")
        # the C1 argv carries the table flag with the SELECTED table, C0's does not
        log = sb.chain_log()
        ok("ZZP5C_START" in log and "ZZP5C_ARM_DONE C1" in log and "ZZP5C_ARM_DONE C0" in log and "REFUSED" not in log, "a success path never prints the bare word REFUSED")
        # RESUMABLE: a re-run does nothing but the precheck
        sb.clear_calls()
        rc, out = sb.run()
        c2 = sb.calls_list()
        ok(rc == 0 and [x for x in c2 if not x.startswith("PY tree-check")] == [], ("rerun executes nothing", c2))
    with_box(f)


def t_c0r():
    def f(td):
        sb = Sandbox(td)
        rc, out = sb.run("--with-c0r")
        c = sb.calls_list()
        ok(rc == 0 and "arms [C1 C0 C0r]" in out, out[-600:])
        ok(seq(c, "TRAIN other out=p5c_c0", "POSTGATE p5c_c0 ", "LG run refe_p5c_c0r", "LG verify refe_p5c_c0r", "TRAIN other out=p5c_c0r", "POSTGATE p5c_c0r other=- ", "EGRESS push p5c/c0r"), c)
    with_box(f)


def t_dry():
    def f(td):
        sb = Sandbox(td)
        rc, out = sb.run("--dry-run", "--with-c0r")
        ok(rc == 0, out[-500:])
        ok(sb.calls_list() == [], "the dry run runs NO tool")
        ok(not sb.root_dir.exists(), "the dry run touches nothing on the volume")
        want = ["p5c_pod_label.sh", "traj_label_table.py verify", "launch_gate.py run --profile refe_p5c ", "launch_gate.py run --profile refe_p5c_c0 ", "launch_gate.py run --profile refe_p5c_c0r ", "train.py", "--preflight", "p5c_postgate.py",
                "hf_egress.py push", "hf_egress.py gate", "diag_p6_labeller_identity.py", "mem-argv", "REFUSE until launch_gate.P5C_BASE_SHA256"]
        ok(all(w in out for w in want), [w for w in want if w not in out])
        ok(out.index("p5c_pod_label.sh") < out.index("traj_label_table.py verify") < out.index("REFUSE until") < out.index("launch_gate.py run --profile refe_p5c ") < out.index("--preflight") < out.index("p5c_postgate.py") < out.index("hf_egress.py push"))
        ok(len(re.findall(r"^DRY: ", out, re.M)) > 25, "every command is printed")
    with_box(f)


# ------------------------------------------------------------------------------------------------ 2. refusals
def refusal(sb, needle, *args, **env):
    rc, out = sb.run(*args, **env)
    return rc, out, sb.chain_log()


def t_refusals():
    # (a) unfilled MM pins: REFUSED at the pin stage, FAIL marker, nothing after it runs; a re-run refuses on the marker; --clear-fail then a filled pin proceeds
    def a(td):
        sb = Sandbox(td)
        rc, out = sb.run(STUB_PINS="unfilled")
        c, log = sb.calls_list(), sb.chain_log()
        ok(rc == 1 and "REFUSED: stage=pins" in log and "never fills a pin" in log, (rc, log[-500:]))
        ok(count(c, "LG ") == 0 and count(c, "TRAIN") == 0 and count(c, "LAB diag") == 0, "nothing after the pins ran")
        ok(sb.stage("pins.FAIL").exists() and sb.stage("label.ok").exists() and sb.stage("table.ok").exists() and not sb.stage("pins.ok").exists(), "label + table done, pins FAIL")
        sb.clear_calls()
        rc, out = sb.run(STUB_PINS="ok")
        ok(rc == 1 and "FAIL marker exists" in sb.chain_log() and sb.calls_list() == [], "a FAIL marker makes the next run refuse before any tool runs")
        rc, out = sb.run("--clear-fail", "pins")
        ok(rc == 0 and not sb.stage("pins.FAIL").exists() and "removed by --clear-fail" in sb.chain_log())
        sb.clear_calls()
        rc, out = sb.run(STUB_PINS="ok")
        c = sb.calls_list()
        ok(rc == 0 and count(c, "LABEL") == 0 and count(c, "PY pins-check") == 1 and count(c, "TRAIN c1 out=p5c_c1") == 1, ("resumed after the pins were filled: label/table are NOT redone", c))
        ok(count(c, "LG run") == 2)
    with_box(a)

    def b(td):                                           # (b) pins filled but not the measured values
        sb = Sandbox(td)
        rc, out = sb.run(STUB_PINS="mismatch")
        ok(rc == 1 and "REFUSED: stage=pins" in sb.chain_log() and count(sb.calls_list(), "LG ") == 0)
    with_box(b)

    def c_(td):                                          # (c) an old ckpt_last.pt, (d) a FAIL marker, (e) the table does not cover every row, (f) a label STOP
        sb = Sandbox(td)
        (sb.data / "refe_runs" / "p5c_c1").mkdir(parents=True)
        (sb.data / "refe_runs" / "p5c_c1" / "ckpt_last.pt").write_text("old", encoding="utf-8")
        rc, out = sb.run()
        ok(rc == 1 and "ckpt_last.pt exists" in sb.chain_log() and sb.calls_list() == [], "an existing ckpt_last.pt refuses before ANY tool runs")
    with_box(c_)

    def d(td):
        sb = Sandbox(td)
        (sb.root_dir / "stage").mkdir(parents=True)
        (sb.root_dir / "stage" / "train_C1.FAIL").write_text("t\nsomething failed\n", encoding="utf-8")
        rc, out = sb.run()
        ok(rc == 1 and "a FAIL marker exists" in sb.chain_log() and sb.calls_list() == [])
    with_box(d)

    def e(td):
        sb = Sandbox(td)
        rc, out = sb.run(STUB_COUNTS_BAD="1")
        ok(rc == 1 and "REFUSED: stage=table" in sb.chain_log() and count(sb.calls_list(), "PY pins-check") == 0, "a table that does not cover every row stops before the pins")
        sb2 = Sandbox(td / "x2") if (td / "x2").mkdir() is None else None
        rc, out = sb2.run(STUB_LABEL_STOP="1")
        ok(rc == 1 and "REFUSED: stage=label" in sb2.chain_log() and count(sb2.calls_list(), "TABLE") == 0, "a label STOP is the label's verdict, the chain refuses")
    with_box(e)

    def g(td):                                           # (g) shards vs CPUs, label, workers
        sb = Sandbox(td)
        rc, out = sb.run(P5C_NSHARD="16")
        ok(rc == 1 and "exceeds this pod's 15 CPUs" in sb.chain_log() and sb.calls_list() == [], "P5C_NSHARD above floor(the cgroup quota 15.3) refuses (nproc says 128)")
        sb.clear_calls()
        rc, out = sb.run(P5C_NSHARD="12")
        ok(count(sb.calls_list(), "LABEL nshard=12") == 1, "an explicit NSHARD <= the quota is used")
    with_box(g)

    def g2(td):
        sb = Sandbox(td)
        (sb.cg / "cpu.max").write_text("400000 100000\n", encoding="utf-8", newline="\n")
        rc, out = sb.run()
        ok(rc == 0 and count(sb.calls_list(), "LABEL nshard=4") == 1, "the default NSHARD is floor(the cgroup quota) (read at run time), not nproc (128) and not the runner's 16")
    with_box(g2)

    def h(td):
        sb = Sandbox(td)
        env_cases = [({"P5C_LABEL": ""}, "never defaulted"), ({"P5C_LABEL": "p5t"}, "never defaulted"), ({"P5C_LABEL_WORKERS": ""}, "P5C_LABEL_WORKERS"), ({"P5C_LABEL_WORKERS": "17"}, "outside 1..16"),
                     ({"P5C_HF_TOKEN_FILE": "/nonexistent/tok"}, "missing, unreadable")]
        for env, needle in env_cases:
            rc, out = sb.run(**env)
            ok(rc == 1 and needle in sb.chain_log() and sb.calls_list() == [], (env, sb.chain_log()[-300:]))
        rc, out = sb.run(STUB_TREE_BAD="1")
        ok(rc == 1 and "not its SHIP_MANIFEST.json" in sb.chain_log() and count(sb.calls_list(), "LABEL") == 0, "a tree that is not its manifest refuses before the label stage")
    with_box(h)

    def i(td):                                           # (i) the token does not verify for C1 -> the gate stage refuses and the trainer is NEVER started
        sb = Sandbox(td)
        rc, out = sb.run(STUB_VERIFY_FAIL="all")
        c = sb.calls_list()
        ok(rc == 1 and "REFUSED: stage=gate_C1" in sb.chain_log() and count(c, "TRAIN preflight") == 0 and count(c, "TRAIN c1") == 0 and sb.stage("gate_C1.FAIL").exists(), (rc, sb.chain_log()[-400:]))
        sb2 = Sandbox(td / "y") if (td / "y").mkdir() is None else None
        rc, out = sb2.run(STUB_VERIFY_FAIL="refe_p5c_c0")
        c = sb2.calls_list()
        ok(rc == 1 and "REFUSED: stage=gate_C0" in sb2.chain_log() and count(c, "TRAIN c1 out=p5c_c1") == 1 and count(c, "TRAIN other out=p5c_c0") == 0, "C1 trained; C0's token failed: C0 is never started")
    with_box(i)

    def j(td):                                           # (j) the post-training gate fails for C1: no egress for it, C0 never starts
        sb = Sandbox(td)
        rc, out = sb.run(STUB_POSTGATE_FAIL="p5c_c1")
        c = sb.calls_list()
        ok(rc == 1 and "REFUSED: stage=postgate_C1" in sb.chain_log() and "NO Adam state" in sb.chain_log() or "G-ADAM" in sb.chain_log(), sb.chain_log()[-500:])
        ok(count(c, "EGRESS push") == 0 and count(c, "LG run refe_p5c_c0") == 0 and sb.stage("postgate_C1.FAIL").exists() and not sb.stage("egress_C1.ok").exists())
    with_box(j)

    def k(td):                                           # (k) an egress that did not verify against the remote listing, a training that exited non-zero, a summary that is not done at 2000 steps
        sb = Sandbox(td)
        rc, out = sb.run(STUB_PUSH_NO_OK="1")
        ok(rc == 1 and "did not verify against the remote listing" in sb.chain_log() or "G-EGR instrument failed" in sb.chain_log() or "egress" in sb.chain_log(), sb.chain_log()[-300:])
        sb = Sandbox(td / "z1") if (td / "z1").mkdir() is None else None
        rc, out = sb.run(STUB_TRAIN_CRASH="p5c_c1")
        ok(rc == 1 and "REFUSED: stage=train_C1" in sb.chain_log() and count(sb.calls_list(), "POSTGATE") == 0, "a crashed trainer: no post-training gate, no egress")
        # the killed attempt: trainstart without train -> the next run (after --clear-fail) refuses until the run was ARCHIVED
        ok(sb.stage("trainstart_C1.ok").exists() and not sb.stage("train_C1.ok").exists())
        rc, out = sb.run("--clear-fail", "train_C1")
        ok(rc == 0 and sb.stage("trainstart_C1.ok").exists(), "--clear-fail keeps trainstart while the dead run's ckpt_last.pt is still there")
        sb.clear_calls()
        rc, out = sb.run()
        ok(rc == 1 and "ckpt_last.pt exists" in sb.chain_log() and count(sb.calls_list(), "TRAIN") == 0, "the dead run's ckpt_last.pt refuses a fresh start")
        shutil.move(str(sb.data / "refe_runs" / "p5c_c1"), str(sb.data / "refe_runs" / "p5c_c1.killed_1"))
        rc, out = sb.run("--clear-fail", "train_C1")
        ok(rc == 0 and not sb.stage("trainstart_C1.ok").exists(), "after the archive, --clear-fail releases the trainstart marker")
        rc, out = sb.run()
        ok(rc == 0 and "ZZP5C_CHAIN_DONE" in out, "and the arm restarts at step 0 (gate stage skipped: its token still verifies)")
    with_box(k)


# ------------------------------------------------------------------------------------------------ 3. static properties and mutants
def code_of(text):
    return "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("#"))


def check_static(text) -> list:
    c = code_of(text)
    bad = []
    if re.search(r">\s*\"?\$?[A-Za-z_/{}\"$]*PASS_|touch [^\n]*PASS_|echo [^\n]*> [^\n]*PASS_", c):
        bad.append("TOKEN_WRITTEN")
    if re.search(r"sed -i[^\n]*launch_gate|>\s*[^\n]*launch_gate\.py|P5C_BASE_SHA256\s*=", c.replace('P5C_BASE_SHA256": (L.', "")):
        bad.append("PIN_FILLED")
    if "pkill" in c or "pgrep" in c:
        bad.append("PKILL")
    if re.search(r"(\$\(|`|\|\||&&|;|^)\s*nproc\b|P5C_NPROC", c, re.M):
        bad.append("NPROC_USED")
    if not re.search(r'^token_ok\(\) \{', c, re.M) or c.count("token_ok") < 4:
        bad.append("TOKEN_UNUSED")
    return bad


def t_static():
    text = CHAIN.read_text(encoding="utf-8")
    ok(check_static(text) == [], check_static(text))
    ok(subprocess.run([bash_exe(), "-n", posix(CHAIN)], capture_output=True).returncode == 0, "bash -n")
    ok("\r" not in text, "LF only")
    ok("P5C_BASE_SHA256" in text and "never fills a pin" in text and "never defaulted" in text)
    # mutations of the static checker: it must see each
    ok("TOKEN_WRITTEN" in check_static(text + '\necho x > "$ROOT/lg_C1/PASS_hand.json"\n'))
    ok("PIN_FILLED" in check_static(text + "\nsed -i 's/P5C_BASE_SHA256 = None/x/' \"$RUN/launch_gate.py\"\n"))
    ok("PKILL" in check_static(text + "\npkill -f train.py\n"))
    ok("NPROC_USED" in check_static(text + '\nncpu="$(nproc)"\n'), "the static checker sees an nproc call")
    ok("NPROC_USED" in check_static(text + '\nx=$(${P5C_NPROC:-nproc})\n'), "and the old P5C_NPROC hook")
    ok(code_of(text).count("token_ok") >= 4, "token_ok guards the gate stage, the preflight and the trainer start")


def mutant(old, new, scenario, expect_diff):
    """run `scenario(sandbox_with_mutated_chain)` on the chain with ONE change; `expect_diff` says what the mutant must break."""
    text = CHAIN.read_text(encoding="utf-8")
    ok(text.count(old) == 1, f"mutation target not found exactly once: {old[:70]!r} ({text.count(old)})")
    res = []

    def f(td):
        sb = Sandbox(td, text.replace(old, new))
        res.append(scenario(sb))
    with_box(f)
    ok(res and res[0] is False, f"MUTANT SURVIVED: {expect_diff}")
    print(f"  mutant RED as required: {expect_diff}", flush=True)


def t_mutants():
    def pins_refuse(sb):                  # True = the chain behaved CORRECTLY (refused)
        rc, _ = sb.run(STUB_PINS="unfilled")
        return rc == 1 and count(sb.calls_list(), "LG run") == 0

    def ckpt_refuse(sb):
        (sb.data / "refe_runs" / "p5c_c0").mkdir(parents=True)
        (sb.data / "refe_runs" / "p5c_c0" / "ckpt_last.pt").write_text("o", encoding="utf-8")
        rc, _ = sb.run()
        return rc == 1 and sb.calls_list() == []

    def nshard_refuse(sb):
        rc, _ = sb.run(P5C_NSHARD="16")
        return rc == 1 and sb.calls_list() == []

    def token_refuse(sb):
        rc, _ = sb.run(STUB_VERIFY_FROM="3")                     # the gate stage and the preflight verify; the trainer start's own check is the one that fails
        return rc == 1 and count(sb.calls_list(), "TRAIN c1 out=p5c_c1") == 0

    def token_refuse_pre(sb):
        rc, _ = sb.run(STUB_VERIFY_FROM="2")
        return rc == 1 and count(sb.calls_list(), "TRAIN preflight") == 0

    def postgate_refuse(sb):
        rc, _ = sb.run(STUB_POSTGATE_FAIL="p5c_c1")
        return rc == 1 and count(sb.calls_list(), "EGRESS push") == 0

    def other_arg(sb):
        rc, _ = sb.run()
        return any("POSTGATE p5c_c0 other=p5c_c1" in x for x in sb.calls_list())

    def order(sb):
        rc, _ = sb.run()
        c = sb.calls_list()
        return rc == 0 and seq(c, "TRAIN c1 out=p5c_c1", "TRAIN other out=p5c_c0")

    def label_refuse(sb):
        rc, _ = sb.run(P5C_LABEL="")
        return rc == 1 and sb.calls_list() == []

    def counts_refuse(sb):
        rc, _ = sb.run(STUB_COUNTS_BAD="1")
        return rc == 1 and count(sb.calls_list(), "PY pins-check") == 0

    def egress_refuse(sb):
        rc, _ = sb.run(STUB_PUSH_NO_OK="1")
        return rc == 1 and not sb.stage("egress_C1.ok").exists()

    def failmarker_refuse(sb):
        (sb.root_dir / "stage").mkdir(parents=True)
        (sb.root_dir / "stage" / "train_C1.FAIL").write_text("t\nx\n", encoding="utf-8")
        rc, _ = sb.run()
        return rc == 1 and sb.calls_list() == []
    mutant('do_stage pins "$TREE-$BASE_SHA-$TSHA" st_pins\n', "", pins_refuse, "the pin check is removed")
    mutant('    [ -f "$S/train_$c.ok" ] || [ ! -e "$(arm_out $c)/ckpt_last.pt" ] || refuse', '    true || refuse', ckpt_refuse, "the ckpt_last.pt refusal is removed")
    mutant('{ [ "$NSHARD" -ge 1 ] && [ "$NSHARD" -le "$ncpu" ]; } || refuse', '{ [ "$NSHARD" -ge 1 ]; } || refuse', nshard_refuse, "NSHARD is no longer capped by nproc")
    mutant('  [ "$DRY" = 1 ] || token_ok "$a" || refuse "the launch-gate token does not verify for $a: train.py is NOT started', '  [ "$DRY" = 1 ] || true || refuse "the launch-gate token does not verify for $a: train.py is NOT started', token_refuse, "the trainer starts without a verified token")
    mutant('  [ "$DRY" = 1 ] || token_ok "$a" || refuse "no launch-gate token verifies for $a: its preflight', '  [ "$DRY" = 1 ] || true || refuse "no launch-gate token verifies for $a: its preflight', token_refuse_pre, "the preflight starts without a verified token")
    mutant('grep -q "^ZZP5C_POSTGATE_OK" "$ROOT/postgate_$a.log" || refuse', 'grep -q "^ZZP5C_POSTGATE" "$ROOT/postgate_$a.log" || refuse', postgate_refuse, "the post-training gate's FAIL line passes")
    mutant("  [ \"$a\" = C0 ] && other=\"--other $(arm_out C1)\"\n", "", other_arg, "C0's gate no longer compares against C1")
    mutant('ARMS="C1 C0"; [ "$WITH_C0R" = 1 ] && ARMS="C1 C0 C0r"', 'ARMS="C0 C1"; [ "$WITH_C0R" = 1 ] && ARMS="C0 C1 C0r"', order, "C0 before C1")
    mutant('case "${P5C_LABEL:-}" in tc20s|p5r|p5r-rj5) ;; *) refuse', 'case "${P5C_LABEL:-}" in *) ;; *) refuse', label_refuse, "the label is defaulted / unchecked")
    mutant('assert d[\'rows\']==$EXPECT_ROWS and d[\'covered\']==$EXPECT_ROWS and not d[\'refused\']', "assert True or d['rows']==$EXPECT_ROWS", counts_refuse, "the table's row coverage is no longer asserted")
    mutant('tail -n +$((n0 + 1)) "$L" | grep -q " ZZEGRESS_OK arm_p5c_$la " || refuse', 'true || refuse', egress_refuse, "an unverified egress passes")
    mutant('  [ -z "$fm" ] || refuse "a FAIL marker exists: $fm', '  true || refuse "a FAIL marker exists: $fm', failmarker_refuse, "the precheck's any-FAIL-marker refusal is removed")


# ------------------------------------------------------------------------------------------------ 4. A3.1: the cgroup shard cap and the p5r-rj5 refusals
A31_TEXT = "PREREG_P5C_A3_1 frozen text (stub)\n"


def sha_text(t) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def rj5_env(sb, a31_text=A31_TEXT, join_bytes=b"rj5-join-bytes"):
    """ship the A3.1 file, the join and the path tool into the sandbox's ship dir; -> the env of a correct p5r-rj5 run"""
    ship = sb.data / "p5c_ship"
    (ship / "PREREG_P5C_A3_1.md").write_text(a31_text, encoding="utf-8", newline="\n")
    (ship / "rj5_bank_join.jsonl.gz").write_bytes(join_bytes)
    (ship / "p5c_rj5_path.py").write_text("x", encoding="utf-8")
    return {"P5C_LABEL": "p5r-rj5", "P5C_A31_SHA256": sha_text(a31_text), "P5C_JOIN_SHA256": hashlib.sha256(join_bytes).hexdigest(),
            "P5C_RJ5_PLATFORMS": "local=devbox A=devbox B=devbox C=devbox", "P5C_RJ5_R0_ROWS": "local=/r0/local A=/r0/A"}


def sub_box(td, name):
    d = td / name
    d.mkdir()
    return Sandbox(d)


def t_a31():
    def quota(td):                                       # (a) the shard cap reads the cgroup, v2 and v1, and refuses without one; nproc (a stub printing 128) is never used
        sb = Sandbox(td)
        rc, out = sb.run()
        ok(rc == 0 and count(sb.calls_list(), "LABEL nshard=15") == 1, ("default NSHARD = floor(15.3) = 15, not nproc 128", rc, out[-300:]))
        sb2 = sub_box(td, "v1")
        (sb2.cg / "cpu.max").unlink()
        (sb2.cg / "cpu").mkdir()
        (sb2.cg / "cpu" / "cpu.cfs_quota_us").write_text("1105000\n", encoding="utf-8", newline="\n")
        (sb2.cg / "cpu" / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8", newline="\n")
        rc, out = sb2.run()
        ok(rc == 0 and count(sb2.calls_list(), "LABEL nshard=11") == 1, ("cgroup v1: 1105000/100000 -> 11", rc, out[-300:]))
        sb2.clear_calls()
        rc, out = sb2.run(P5C_NSHARD="12")
        ok(rc == 1 and "exceeds this pod's 11 CPUs" in sb2.chain_log() and sb2.calls_list() == [], "v1: 12 shards on 11 CPUs refuse")
        sb3 = sub_box(td, "none")
        (sb3.cg / "cpu.max").unlink()
        rc, out = sb3.run()
        ok(rc == 1 and "no readable cgroup CPU quota" in sb3.chain_log() and sb3.calls_list() == [], "no cgroup quota file: refuse, never fall back to nproc")
        sb4 = sub_box(td, "max")
        (sb4.cg / "cpu.max").write_text("max 100000\n", encoding="utf-8", newline="\n")
        rc, out = sb4.run()
        ok(rc == 1 and "no readable cgroup CPU quota" in sb4.chain_log() and sb4.calls_list() == [], "cpu.max 'max' (no quota): refuse")
        sb5 = sub_box(td, "sub1")
        (sb5.cg / "cpu.max").write_text("50000 100000\n", encoding="utf-8", newline="\n")
        rc, out = sb5.run()
        ok(rc == 0 and count(sb5.calls_list(), "LABEL nshard=1") == 1, "a quota under one CPU still gives one shard")
    with_box(quota)

    def happy(td):                                       # (b) p5r-rj5 with the A3.1 sha, the join sha and the declarations: the runner gets them all
        sb = Sandbox(td)
        env = rj5_env(sb)
        rc, out = sb.run(**env)
        c = sb.calls_list()
        ok(rc == 0 and "ZZP5C_CHAIN_DONE" in out, ("p5r-rj5 happy", rc, sb.chain_log()[-400:]))
        want = "LABELENV label=p5r-rj5 join=%s a31=%s plat=local=devbox A=devbox B=devbox C=devbox r0=local=/r0/local A=/r0/A" % (env["P5C_JOIN_SHA256"], env["P5C_A31_SHA256"])
        ok(want in c, ("the runner receives the label, the join sha, the A3.1 sha and the declarations", c[:6]))
        ok("ZZP5C_TABLE p5r-rj5 " in sb.chain_log(), "the chain verifies the table of the p5r-rj5 label")
        ok(sb.stage("label.ok").read_text(encoding="utf-8").splitlines()[1].count("-") >= 5, "the label stage key binds the join and A3.1 shas")
        sb.clear_calls()
        rc, out = sb.run(**env)
        ok(rc == 0 and count(sb.calls_list(), "LABEL") == 0, "resumable: a re-run re-labels nothing")
    with_box(happy)

    def a2_unaffected(td):
        sb = Sandbox(td)
        rc, out = sb.run()
        ok(rc == 0 and count(sb.calls_list(), "LABELENV label=p5r join= a31= ") == 1, "the A2 label p5r needs none of the A3.1 inputs")
    with_box(a2_unaffected)

    def refuse_cases(td):                                # (c) every p5r-rj5 refusal, before any tool runs
        cases = [("no A3.1 sha", {"P5C_A31_SHA256": ""}, "needs the A3.1 sha256"),
                 ("A3.1 sha not hex", {"P5C_A31_SHA256": "Z" * 64}, "needs the A3.1 sha256"),
                 ("A3.1 sha short", {"P5C_A31_SHA256": "ab" * 31}, "needs the A3.1 sha256"),
                 ("A3.1 sha is another file's", {"P5C_A31_SHA256": sha_text("another text")}, "does not match the sha256"),
                 ("no join sha", {"P5C_JOIN_SHA256": ""}, "needs the RJ5 join sha256"),
                 ("join sha is another file's", {"P5C_JOIN_SHA256": "0" * 64}, "P5C_JOIN_SHA256 does not match"),
                 ("platforms undeclared", {"P5C_RJ5_PLATFORMS": ""}, "P5C_RJ5_PLATFORMS is not set"),
                 ("no R0 rows", {"P5C_RJ5_R0_ROWS": ""}, "P5C_RJ5_R0_ROWS is not set")]
        for i, (name, over, needle) in enumerate(cases):
            sb = sub_box(td, f"c{i}")
            env = rj5_env(sb)
            env.update(over)
            rc, out = sb.run(**env)
            ok(rc == 1 and needle in sb.chain_log() and sb.calls_list() == [], (name, rc, sb.chain_log()[-300:]))
        sb = sub_box(td, "nofile")
        env = rj5_env(sb)
        (sb.data / "p5c_ship" / "PREREG_P5C_A3_1.md").unlink()
        rc, out = sb.run(**env)
        ok(rc == 1 and "is not shipped" in sb.chain_log() and sb.calls_list() == [], "the frozen A3.1 file is not shipped")
        sb = sub_box(td, "nojoin")
        env = rj5_env(sb)
        (sb.data / "p5c_ship" / "rj5_bank_join.jsonl.gz").unlink()
        rc, out = sb.run(**env)
        ok(rc == 1 and "is not shipped" in sb.chain_log() and sb.calls_list() == [], "the RJ5 join is not shipped")
        sb = sub_box(td, "stage")
        env = rj5_env(sb)
        env["P5C_A31_SHA256"] = ""
        sb.run(**env)
        ok(not (sb.root_dir / "stage" / "label.ok").exists(), "a refusal writes no label marker")
    with_box(refuse_cases)


def t_a31_mutants():
    def rj5_scenario(over, needle):
        def f(sb):                                       # True = the chain behaved CORRECTLY (refused, for this reason)
            env = rj5_env(sb)
            env.update(over)
            rc, _ = sb.run(**env)
            return rc == 1 and needle in sb.chain_log() and count(sb.calls_list(), "LABEL") == 0
        return f

    def nshard16(sb):
        rc, _ = sb.run(P5C_NSHARD="16")
        return rc == 1 and sb.calls_list() == []

    def nshard_default(sb):
        rc, _ = sb.run()
        return rc == 0 and count(sb.calls_list(), "LABEL nshard=15") == 1

    def no_quota(sb):
        (sb.cg / "cpu.max").unlink()
        rc, _ = sb.run()
        return rc == 1 and sb.calls_list() == []
    mutant('ncpu="$(cpu_quota)" || refuse "no readable', 'ncpu="$(nproc)" || refuse "no readable', nshard16, "NSHARD is capped by nproc (128) instead of the cgroup quota: 16 > 15.3 passes")
    mutant('ncpu="$(cpu_quota)" || refuse "no readable', 'ncpu="$(nproc)" || refuse "no readable', nshard_default, "the default NSHARD is nproc (128), not floor(15.3) = 15")
    mutant('ncpu="$(cpu_quota)" || refuse "no readable', 'ncpu="$(cpu_quota || nproc)" || refuse "no readable', no_quota, "a missing cgroup quota falls back to nproc")
    mutant('  if [ "$P5C_LABEL" = p5r-rj5 ]; then\n', '  if false; then\n', rj5_scenario({"P5C_A31_SHA256": ""}, "needs the A3.1 sha256"), "P5C_LABEL=p5r-rj5 builds a C1 label with no A3.1 sha256 at all")
    mutant('    is_sha256 "${P5C_A31_SHA256:-}" || refuse', '    true || refuse', rj5_scenario({"P5C_A31_SHA256": ""}, "needs the A3.1 sha256"), "an unset A3.1 sha256 is no longer named")
    mutant('    [ "$(sha_of "$A31FILE")" = "$P5C_A31_SHA256" ] || refuse', '    true || refuse', rj5_scenario({"P5C_A31_SHA256": sha_text("another text")}, "does not match the sha256"), "an A3.1 sha256 that does not match the shipped A3.1 file is accepted")
    mutant('    [ "$(sha_of "$JOINF")" = "$P5C_JOIN_SHA256" ] || refuse', '    true || refuse', rj5_scenario({"P5C_JOIN_SHA256": "0" * 64}, "P5C_JOIN_SHA256 does not match"), "a join sha256 that does not match the shipped join is accepted")
    mutant('    is_sha256 "${P5C_JOIN_SHA256:-}" || refuse', '    true || refuse', rj5_scenario({"P5C_JOIN_SHA256": ""}, "needs the RJ5 join sha256"), "an unset join sha256 is no longer named")
    mutant('    [ -n "${P5C_RJ5_PLATFORMS:-}" ] || refuse', '    true || refuse', rj5_scenario({"P5C_RJ5_PLATFORMS": ""}, "P5C_RJ5_PLATFORMS is not set"), "the roll platforms are no longer required to be declared")


def main() -> int:
    for name, fn in (("happy", t_happy), ("c0r", t_c0r), ("dry", t_dry), ("refusals", t_refusals), ("static", t_static), ("mutants", t_mutants), ("a31", t_a31), ("a31_mutants", t_a31_mutants)):
        fn()
        print(f"  {name}: done ({NPASS} ok, {len(FAILS)} failed)", flush=True)
    if FAILS:
        print("FAILED:", FAILS)
        return 1
    print(f"ZZP5CCHAIN_PASS {NPASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
