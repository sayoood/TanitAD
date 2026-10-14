#!/usr/bin/env python3
"""Tests for the P2 training-seed replicate chain (P2-s1): refe/p2rep_pod.sh, refe/p2rep_checks.py and the launch-gate profile `refe_p2rep` (eval/P2REP_POD_RUNBOOK.md).  No pod, no GPU, no network.

  1. THE ARM.  The chain's ARGS, the launch gate's pin (launch_gate.P2REP_ARGV) and refe/p2r_pod.sh's ARGS (P2's, UNTOUCHED), all expanded the way the pod expands them, differ by EXACTLY two things:
     `--out` (the replicate's own run dir) and `--seed 1` added.  Nothing else moved.  The pin is DERIVED from P2's argv by code that makes those two changes and nothing else.
  2. REACHABILITY (CLAUDE.md "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER").  `--seed 1` is traced from the chain's argv to the parser (train.py:1920), `torch.manual_seed(a.seed)` (train.py:1125, BEFORE the
     model is built) and the epoch samplers (train.py:1262/1264, 1500/1501), by an AST pass over the REAL train.py, a DYNAMIC probe (run() with the model build stubbed records the seed it was given) and
     MUTATION: each hop edited in a copy of the source turns the pass RED, and the chain's argv without `--seed 1` is refused by the REAL gate.
  3. THE GATE.  `refe_p2` binds neither --seed nor --out (shown below: it is CLEAN on an argv with P2's own --out and no --seed).  `refe_p2rep` PINS the whole argv; every mutation of it (a seed dropped or changed,
     P2's own --out, any lever moved, a flag added / dropped / doubled, a forbidden flag) is REFUSED by the REAL profile_refusals, run_checks marks every check FAIL and finalize mints NO token.
  4. STATIC properties of the chain (`check_chain`), each broken in a copy by a MUTATION that must turn it RED: argv, repo, run dir, profile, HF_TOKEN hygiene, fd 200, stage order, token before every launch, WAIT
     before anything, no write outside the arm's own dirs, no dropped P2 stage, flock on a read-write fd, markers.
  5. BEHAVIOUR of functions extracted verbatim from the chain under bash: upstream / wait_clear (S1's lock held => WAITS; S1 free and no ZZS1AUX_DONE => REFUSES; ABANDONED => proceeds; the GPU), token_ok against
     tokens minted by the gate's own mint() for refe_p2rep.
  6. p2rep_checks.py in both directions (every shared input wrong in turn; P2's markers; the waiver; READ-ONLY: the whole fixture tree is byte- and mtime-identical afterwards).
  7. The REAL chain executed in a sandbox (a temp WORK; stub trainer / gate / egress; the REAL helper, the REAL pdm_label_complete.py): every refusal path, the WAIT, the happy path in order, no trainer start
     without a verified token, every HF call on THIS arm's repo and none on P2's, HF_TOKEN only in the egress children, and the shared inputs untouched (snapshot before == after).
  8. No pod endpoint (IPv4 literal, ssh target, port, token) in the chain, the helper, this test or the runbook.

Run: python refe/test_p2rep_chain.py   -> ZZTEST_PASS <n>   (needs bash on PATH; Git Bash on the dev box).  This test is part of the profile's G-SUITE ON THE POD, where a skip fails the suite: it prints no ZZTEST_SKIP;
the runbook checks run only where eval/ is shipped (the dev box).
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

CHAIN = HERE / "p2rep_pod.sh"
P2CHAIN = HERE / "p2r_pod.sh"
HELPER = HERE / "p2rep_checks.py"
RUNBOOK = PKGROOT / "eval" / "P2REP_POD_RUNBOOK.md"
BASH = shutil.which("bash")
PY = sys.executable
P2_REPO = "Sayood/tanitad-refe-p2-joint-pdm"
S1_REPO = "Sayood/tanitad-refe-p2-joint-pdm-s1"
N_OK = 0


def ok(cond, msg=""):
    global N_OK
    assert cond, msg
    N_OK += 1


def posix(p):
    return str(p).replace("\\", "/")


def write_sh(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def bash_run(script, env=None, timeout=300):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=e, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def fn_body(text, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    return m.group(0) if m else ""


def one_line_fn(text, name):
    m = re.search(rf"^{name}\(\) \{{.*\}}$", text, re.M)
    return m.group(0) if m else ""


def const(text, name):
    m = re.search(rf"^{name}=(\S+)", text, re.M)
    return m.group(1).strip('"') if m else None


# ============================================================================================ the arm: three copies of one argv
ARGS_RX = re.compile(r'^ARGS="(.*)"$', re.M)


def expand_args(text):
    """`ARGS` as the pod's bash expands it: $RUN resolved from the chain's own RUN= line (DATA=/workspace/data)."""
    run = const(text, "RUN")
    assert run, "no RUN= in the chain"
    run = run.replace("$DATA", "/workspace/data")
    return ARGS_RX.search(text).group(1).replace("$RUN", run).split()


def pairs(toks):
    return [(f, tuple(v)) for f, v in GT.flag_pairs(toks)]


def arm_checks():
    p2 = expand_args(P2CHAIN.read_text(encoding="utf-8"))
    rep = expand_args(CHAIN.read_text(encoding="utf-8"))
    ok(p2 == L.P2_ARGV.split(), "launch_gate.P2_ARGV is not refe/p2r_pod.sh's ARGS as it expands")
    ok(rep == L.P2REP_ARGV and rep == L.p2rep_argv(), "the chain's ARGS is not the gate's pinned replicate argv")
    gp, gr = pairs(p2), pairs(rep)
    added = [g for g in gr if g not in gp]
    removed = [g for g in gp if g not in gr]
    ok(added == [("--seed", ("1",)), ("--out", ("/workspace/data/refe_runs/p2_joint_pdm_s1",))], added)
    ok(removed == [("--out", ("/workspace/data/refe_runs/p2_joint_pdm",))], removed)
    ok([g for g in gp if g in gr] == [g for g in gr if g in gp], "a shared lever changed ORDER or value")
    ok(len(gr) == len(gp) + 1 and dict(gr)["--seed"] == ("1",) and gr.count(("--seed", ("1",))) == 1)
    ok(L.P2REP_RUN_DIR != L.P2_RUN_DIR and L.P2REP_RUN_DIR.endswith("_s1") and L.P2REP_SEED == 1)
    # the derivation itself makes ONLY those two changes, for any P2 argv
    toks = "--a 1 --b x --out /r --resume".split()
    ok(L.p2rep_argv(" ".join(toks)) == ["--a", "1", "--b", "x", "--seed", "1", "--out", L.P2REP_RUN_DIR, "--resume"])
    # the three resolved copies of every data/run path live under the same /workspace/data tree
    ok(const(CHAIN.read_text(encoding="utf-8"), "RUN") == "$DATA/refe_runs/p2_joint_pdm_s1" and const(P2CHAIN.read_text(encoding="utf-8"), "RUN") == "$DATA/refe_runs/p2_joint_pdm")
    ok(P2CHAIN.read_text(encoding="utf-8").count("--seed") == 0, "p2r_pod.sh must stay UNTOUCHED: it names no --seed (P2 ran the parser default)")
    # the P2 tools this chain reuses on SHARED data write only what the chain hands them: pdm_label_complete.py its --json, p2r_gate_prep.py its --out-sets / --out-pdm
    pc = (HERE / "pdm_label_complete.py").read_text(encoding="utf-8")
    ok(sum(1 for ln in pc.splitlines() if "open(" in ln and re.search(r'["\']w["\']', ln)) == 1 and 'open(a.json, "w"' in pc and "shutil" not in pc and "os.remove" not in pc and "os.replace" not in pc, "pdm_label_complete.py must only write --json")
    gp = (HERE / "p2r_gate_prep.py").read_text(encoding="utf-8")
    ok(sum(1 for ln in gp.splitlines() if "open(" in ln and re.search(r'["\']w["\']', ln)) == 2 and "a.out_sets" in gp and "a.out_pdm" in gp and "shutil" not in gp and "os.remove" not in gp, "p2r_gate_prep.py must only write its --out-sets / --out-pdm")
    return rep


# ============================================================================================ 2. reachability of --seed 1
def seed_wiring_problems(src):
    """AST pass over train.py's source: the hops from `a.seed` to the RNG and to the epoch order."""
    tree = ast.parse(src)
    probs = []
    run_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run_fn is None:
        return ["NO_RUN_FUNCTION"]
    seeds = [c for c in ast.walk(run_fn) if isinstance(c, ast.Call) and ast.unparse(c.func) == "torch.manual_seed"]
    good = [c for c in seeds if len(c.args) == 1 and ast.unparse(c.args[0]) == "a.seed"]
    if not good:
        probs.append("MANUAL_SEED_IS_NOT_A_SEED")
    builds = [n.lineno for n in ast.walk(run_fn) if isinstance(n, ast.Assign) and ast.unparse(n.value).startswith("REFe(")]
    if not builds:
        probs.append("NO_MODEL_BUILD_FOUND")
    elif good and min(c.lineno for c in good) > min(builds):
        probs.append("SEED_AFTER_MODEL_BUILD")
    smp = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and ast.unparse(c.func) in ("SceneEpochSampler", "EpochSampler")]
    if len(smp) < 4:
        probs.append(f"SAMPLER_CALLS_{len(smp)}")
    for c in smp:
        if len(c.args) < 2 or ast.unparse(c.args[1]) != "a.seed":
            probs.append(f"SAMPLER_NOT_SEEDED_BY_a.seed:line{c.lineno}")
    parser_ok = False
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "ap.add_argument" and c.args and ast.unparse(c.args[0]) == "'--seed'":
            kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
            parser_ok = kw.get("type") == "int" and kw.get("default") == "0"
    if not parser_ok:
        probs.append("PARSER_SEED_FLAG")
    return probs


def seed_probe(argv):
    """DYNAMIC: parse `argv` with the REAL parser, run the REAL train.run() with the model build replaced by a sentinel, and return the seed torch.manual_seed was given."""
    import torch
    import train as T
    rec = {}
    real = torch.manual_seed

    class Stop(Exception):
        pass

    def fake_seed(s):
        rec["seed"] = s
        return real(s)

    def boom(cfg):
        raise Stop()
    old = T.REFe
    torch.manual_seed, T.REFe = fake_seed, boom
    try:
        a = T.build_parser().parse_args(argv)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                T.run(a)
            except Stop:
                pass
    finally:
        torch.manual_seed, T.REFe = real, old
    return rec.get("seed")


def reachability(rep_argv, tmp):
    src = (HERE / "train.py").read_text(encoding="utf-8")
    ok(seed_wiring_problems(src) == [], seed_wiring_problems(src))
    import train as T
    # hop 1: the parser
    ok(T.build_parser().parse_args(rep_argv).seed == 1 and T.build_parser().parse_args(L.P2_ARGV.split()).seed == 0)
    # hop 2: the RNG -- the model build receives the seed (DYNAMIC); the argv WITHOUT --seed 1 gives 0 (the control that must read the old value)
    tdir = str(tmp / "seedprobe")
    argv1 = [t for t in rep_argv]
    argv1[argv1.index("--out") + 1] = tdir
    argv0 = [t for t in argv1]
    i = argv0.index("--seed")
    del argv0[i:i + 2]
    ok(seed_probe(argv1) == 1, "train.run() did not hand torch.manual_seed the argv's --seed 1")
    ok(seed_probe(argv0) == 0, "the control (no --seed) must read 0")
    # hop 3: the epoch order differs by seed and repeats for one seed
    groups = [[i, i + 100] for i in range(50)]
    o0, o1 = T.SceneEpochSampler(groups, 0).order(), T.SceneEpochSampler(groups, 1).order()
    ok(o0 != o1 and o1 == T.SceneEpochSampler(groups, 1).order() and len(o1) == len(o0) == 50, "the sampler order does not depend on the seed")
    ok(T.EpochSampler(50, 0).order() != T.EpochSampler(50, 1).order())
    # hop 4: the launch gate's G-DVB compares the argv's seed with the BUILT one (torch.initial_seed())
    gc = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    ok('add("seed", a.seed, b.seed, "torch.initial_seed()")' in gc and "cap.seed = torch.initial_seed()" in gc)
    # MUTATION: each hop broken in a copy of the source turns the pass RED
    muts = {"manual_seed ignores the flag": src.replace("torch.manual_seed(a.seed)", "torch.manual_seed(0)", 1),
            "seed set AFTER the model is built": src.replace("    torch.manual_seed(a.seed)\n    model = REFe(cfg)\n", "    model = REFe(cfg)\n    torch.manual_seed(a.seed)\n", 1),
            "the train sampler ignores the seed": src.replace("SceneEpochSampler(ds.scene_groups, a.seed,", "SceneEpochSampler(ds.scene_groups, 0,", 1),
            "the eval sampler ignores the seed": src.replace("EpochSampler(len(d), a.seed,", "EpochSampler(len(d), 0,", 1),
            "the flag default moved": src.replace('ap.add_argument("--seed", type=int, default=0)', 'ap.add_argument("--seed", type=int, default=7)', 1)}
    for name, m in muts.items():
        ok(m != src, name)
        ok(seed_wiring_problems(m), f"the wiring pass stayed GREEN on: {name}")
    # the chain's argv without --seed 1 is refused by the REAL gate (the chain-level mutation of the same hop)
    ok(L.profile_refusals(L.PROFILES["refe_p2rep"], argv0) != [])


# ============================================================================================ 3. the gate
def gate_checks_(rep_argv, tmp):
    P = L.PROFILES["refe_p2rep"]
    ok(L.profile_refusals(P, rep_argv) == [], L.profile_refusals(P, rep_argv))
    ok(L.profile_refusals(P, L.P2REP_ARGV) == [])
    # the profile is refe_p2's, with exactly four keys moved
    p2 = L.PROFILES["refe_p2"]
    moved = sorted(k for k in set(P) | set(p2) if P.get(k) != p2.get(k))
    ok(moved == ["name", "pinned_argv", "required_flags", "suite"], moved)
    ok(P["checks"] == p2["checks"] and "G-P2GATE" in P["checks"] and P["data_flags"] == p2["data_flags"] and P["forbidden_flags"] == p2["forbidden_flags"] and P["mintable"] and P["require_commit"] and not P["rehearsal"])
    ok(P["required_flags"] == {**p2["required_flags"], "--seed": "1"} and tuple(P["pinned_argv"]) == tuple(L.P2REP_ARGV))
    ok(p2.get("pinned_argv") is None and "--seed" not in p2["required_flags"], "refe_p2 must stay as it was")
    for e in P["suite"]:                                           # a renamed PASS token would make G-SUITE fail on the pod, not here
        ok((HERE / e["script"]).is_file() and e["token"] in (HERE / e["script"]).read_text(encoding="utf-8"), e)
    ok(any(e["script"] == "test_p2rep_chain.py" for e in P["suite"]) and any(e["script"] == "test_p2r_chain.py" for e in P["suite"]))
    # WHY a profile of its own: refe_p2 mints for the replicate's argv with NO --seed 1 and with P2's OWN --out (it binds neither)
    no_seed = [t for t in rep_argv]
    i = no_seed.index("--seed")
    del no_seed[i:i + 2]
    own_out = list(rep_argv)
    own_out[own_out.index("--out") + 1] = L.P2_RUN_DIR
    seed7 = list(rep_argv)
    seed7[seed7.index("--seed") + 1] = "7"
    ok(L.profile_refusals(p2, no_seed) == [] and L.profile_refusals(p2, own_out) == [] and L.profile_refusals(p2, seed7) == [], "refe_p2 was expected to be blind to --seed / --out")

    def swap(flag, new):
        a = list(rep_argv)
        a[a.index(flag) + 1] = new
        return a

    bad = {"no --seed": (no_seed, "--seed"), "seed 0": (swap("--seed", "0"), "--seed"), "seed 2": (swap("--seed", "2"), "--seed"),
           "P2's own --out": (own_out, "--out"), "another --out": (swap("--out", "/workspace/data/refe_runs/elsewhere"), "--out"),
           "lr moved": (swap("--lr", "6e-5"), "--lr"), "epochs moved": (swap("--epochs", "4"), "--epochs"), "accum moved": (swap("--accum", "32"), "--accum"),
           "batch moved": (swap("--batch", "8"), "--batch"), "init moved": (swap("--init-from", "/workspace/data/x.pt"), "--init-from"),
           "pdm labels moved": (swap("--pdm-labels", "/workspace/data/refe_sft2/other"), "--pdm-labels"), "workers moved": (swap("--workers", "3"), "--workers"),
           "ckpt cadence moved": (swap("--ckpt-every-min", "30"), "--ckpt-every-min"), "log-every moved": (swap("--log-every", "25"), "--log-every"),
           "tf32 dropped": ([t for t in rep_argv if t != "--tf32"], "--tf32"), "compile dropped": ([t for t in rep_argv if t != "--compile"], "--compile"),
           "resume dropped": ([t for t in rep_argv if t != "--resume"], "--resume"), "a lever added": (rep_argv + ["--weight-decay", "0.1"], "--weight-decay"),
           "seed doubled": (rep_argv + ["--seed", "1"], "--seed"), "a flag added with no value": (rep_argv + ["--grow"], "--grow"),
           "a forbidden flag": (rep_argv + ["--synthetic"], "FORBIDS"), "preflight in a launch": (rep_argv + ["--preflight"], "FORBIDS"),
           "stray leading token": (["x"] + rep_argv, "belongs to no flag")}
    for name, (argv, frag) in bad.items():
        rs = L.profile_refusals(P, argv)
        ok(rs and any(frag in r for r in rs), (name, rs))
    # order does not matter to argparse, so it does not matter to the pin
    shuffled = list(reversed(pairs(rep_argv)))
    flat = [t for f, v in shuffled for t in (f, *v)]
    ok(L.profile_refusals(P, flat) == [], L.profile_refusals(P, flat))
    # the CLI: CLEAN exits 0, a mutated argv exits 5 with the refusal text, and `--profile refe_p2rep` is a known choice
    r1 = subprocess.run([PY, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_p2rep", "--args", " ".join(rep_argv)], capture_output=True, text=True, encoding="utf-8")
    ok(r1.returncode == 0 and "ZZGATE-CLEANZZ" in r1.stdout, r1.stdout + r1.stderr)
    r2 = subprocess.run([PY, str(HERE / "launch_gate.py"), "refusals", "--profile", "refe_p2rep", "--args", " ".join(no_seed)], capture_output=True, text=True, encoding="utf-8")
    ok(r2.returncode == 5 and "ZZGATE-REFUSEDZZ" in r2.stdout and "--seed" in r2.stdout, r2.stdout)
    # the generic rule does nothing to a profile that does not pin
    ok(L.pinned_argv_refusals({"name": "x"}, ["--a"]) == [] and L.profile_refusals(L.PROFILES["refe_p2"], rep_argv) == [])
    # the whole gate, end to end on the red arm: every check FAILs and NO token is minted
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        key = GT.load_key(td / "k" / "gate.key", forbid_under=[PKGROOT])
        for name in ("no --seed", "P2's own --out", "lr moved"):
            out = td / name.replace(" ", "_").replace("'", "")
            st = L.run_checks(P, bad[name][0], PKGROOT, out)
            ok(set(st.values()) == {"FAIL"} and set(st) == set(P["checks"]) | set(P.get("na_ok", ())), st)
            v, pth, t = L.finalize(P, bad[name][0], PKGROOT, out, key)
            ok(v == "FAIL" and not list(out.glob("PASS_*")) and any(bad[name][1] in r for r in t["reasons"]), t["reasons"][:3])
            good, why, _ = GT.verify_token(pth, bad[name][0], PKGROOT, key=key, profile="refe_p2rep")
            ok(not good and any("not PASS" in r for r in why))
    return bad


# ============================================================================================ 4. static properties of the chain
def logical_commands(text):
    """Backslash-joined lines, heredoc bodies dropped, comment lines dropped, an open `( ... )` subshell merged into one command."""
    out, cur, heredoc = [], "", None
    joined, buf = [], ""
    for ln in text.split("\n"):
        if ln.endswith("\\"):
            buf += ln[:-1] + " "
            continue
        joined.append(buf + ln)
        buf = ""
    depth = 0
    for ln in joined:
        if heredoc:
            if ln == heredoc:
                heredoc = None
            continue
        m = re.search(r"<<'?(\w+)'?\s*$", ln)
        if m:
            heredoc = m.group(1)
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cur = (cur + " " + s).strip() if cur else s
        depth += s.count("(") - s.count(")")
        if depth <= 0:
            out.append(cur)
            cur, depth = "", 0
    if cur:
        out.append(cur)
    return out


def code_of(text):
    """The text with comment lines and trailing ` # ...` comments removed (a `#` inside ${x#y} has no space before it)."""
    out = []
    for ln in text.split("\n"):
        if ln.lstrip().startswith("#"):
            continue
        out.append(re.sub(r"\s+#\s.*$", "", ln))
    return "\n".join(out)


ALLOWED_TARGET = re.compile(r'^\$\{?(?:R|S|L|LG|RUN|WSTOP)\}?(?:/|$)')
REDIR_RX = re.compile(r'(?<![<>&\w])(?:\d+)?>>?(?![&>])\s*("[^"]*"|[^\s;|&)]+)')
FLAG_OUT_RX = re.compile(r'(--(?:out|out-sets|out-pdm|out-dir|json|outbox|work|gate-bank|stop-file|chain-log)) +("[^"]*"|[^\s;|&)]+)')
WRITE_CMD_RX = re.compile(r'(?:^|[;&|({]\s*|\bthen\s+|\bdo\s+)(touch|rm|mkdir|cp|mv|tee|dd)\b([^;|&)]*)')
FORBIDDEN_CMD_RX = re.compile(r'(?:^|[;&|({]\s*|\bexec\s+|\benv\s+(?:-u\s+\w+\s+)?)(tar|zstd|curl|wget|rsync|unzip|mv|ln|chmod|chown|truncate|apt-get|uv|pip|git|huggingface-cli)\s')


def write_targets(code):
    """Every path the chain writes: redirections, dd of=, touch / rm / mkdir / cp / mv / tee operands, and the output flags of the python children."""
    out = []
    for c in logical_commands(code):
        for m in REDIR_RX.finditer(c):
            out.append(m.group(1))
        for m in FLAG_OUT_RX.finditer(c):
            if m.group(1) == "--out" and "pdm_label_complete.py" in c:          # that tool's --out is the labels dir it READS; its only write is --json (asserted on its source in arm_checks)
                continue
            out.append(m.group(2))
        for m in re.finditer(r'\bof=("[^"]*"|\S+)', c):
            out.append(m.group(1))
        mc = re.match(r'cp\s+.*?(?:\s+\d*>\S+)*\s+("[^"]*"|[^\s"]+)(?:\s+\d*>\S+)*\s*$', c)
        if mc:
            out.append(mc.group(1))
        for m in WRITE_CMD_RX.finditer(c):
            if m.group(1) in ("dd", "cp"):
                continue
            for t in re.findall(r'"[^"]*"|[^\s"]+', m.group(2)):
                if not t.startswith("-") and not re.match(r"^\d*[<>]", t):
                    out.append(t)
    return out


def allowed_target(t):
    t = t.strip('"')
    return t in ("/dev/null",) or bool(ALLOWED_TARGET.match(t)) or t.startswith("&") or t.isdigit()


ORDER = ['[ -n "${HF_TOKEN:-}" ] || refuse', "p2rep_checks.py tree", "p2r_sync.py watch", "\nwait_clear\n", "p2rep_checks.py inputs", "pdm_label_complete.py --log", "dd if=/dev/zero",
         "p2r_sync.py recover", "p2r_gate_prep.py --sets", "p2_gate.py --init-from", "launch_gate.py run --profile refe_p2rep", "train.py $ARGS --preflight", "train.py $ARGS >>", "p2r_sync.py final"]
CHILD_RX = re.compile(r'"\$PY" +(?:[\w./-]+\.py\b|-c\b|")|\bsleep +"?\$?\w+"? +200|\bdd +if=|^sleep ')
DROPPED = ("p2r_sync.py restore", "p2r_sync.py pdm-push", "p2r_carve.py --sets", "STAGES=dbs", "STAGES=pix", "STAGES=weights", "onpolicy_relabel_pdm.py --sets", 'bash "$PKG/code/pod_teacher_env.sh"',
           "apt-get", "uv pip")


def check_chain(text):
    v = []
    code = code_of(text)
    cmds = logical_commands(text)
    if "\r" in text:
        v.append("CRLF")
    # ---- the arm
    if len(re.findall(r"^\s*ARGS=", text, re.M)) != 1:
        v.append("ARGS_ASSIGNED_NOT_ONCE")
    try:
        if expand_args(text) != L.P2REP_ARGV:
            v.append("ARGS_NOT_THE_PINNED_REPLICATE_ARGV")
    except Exception:                                                    # noqa: BLE001
        v.append("ARGS_UNPARSEABLE")
    if const(text, "RUN") != "$DATA/refe_runs/p2_joint_pdm_s1":
        v.append("RUN_DIR")
    if const(text, "OUT_REPO") != S1_REPO:
        v.append("OUT_REPO")
    if re.search(r"tanitad-refe-p2-joint-pdm(?!-s1)", code):
        v.append("P2_REPO_NAMED_IN_CODE")
    for c in cmds:
        m = re.search(r'"\$PY" +p2r_sync\.py +([\w-]+)', c)
        if m:
            if m.group(1) not in ("watch", "tick", "recover", "final"):
                v.append("SYNC_SUBCOMMAND_NOT_ALLOWED:" + m.group(1))
            if '--repo "$OUT_REPO"' not in c:
                v.append("SYNC_WITHOUT_REPO:" + c[:50])
    if len(re.findall(r'p2r_sync\.py +(?:watch|tick|recover|final)', code)) != 4:
        v.append("SYNC_CALL_COUNT")
    # ---- the gate
    if len(re.findall(r"--profile refe_p2rep\b", code)) != 2 or re.search(r"--profile refe_p2(?!rep)\b", code):
        v.append("PROFILE")
    if not re.search(r'token_ok\(\) \{\n.*launch_gate\.py" verify --profile refe_p2rep --token "\$tok" --args "\$ARGS" --pkg "\$PKG"', text, re.S):
        v.append("TOKEN_OK_NOT_VERIFYING")
    seg = text[text.find("STAGE=train"):text.find('train.py $ARGS --preflight')]
    if "wait_clear" not in seg or "token_ok ||" not in seg:
        v.append("NO_WAIT_OR_TOKEN_BEFORE_PREFLIGHT")
    loop = text[text.find("for att in"):text.find('train.py $ARGS >>')]
    if "wait_clear" not in loop or "token_ok ||" not in loop:
        v.append("NO_WAIT_OR_TOKEN_BEFORE_LAUNCH")
    for c in cmds:
        if re.search(r'"\$PY" +train\.py', c) and not re.search(r'"\$PY" train\.py \$ARGS(?: --preflight)?(?:\s+>>?\s|\s+2>)', c):
            v.append("TRAINER_ARGV_NOT_JUST_ARGS:" + c[:60])
    # ---- HF_TOKEN hygiene
    for ln in code.split("\n"):
        if "HF_TOKEN" in ln and not ("env -u HF_TOKEN" in ln or ln.strip() == '[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"'
                                     or 'if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ] && [ -n "${HF_TOKEN:-}" ]' in ln):
            v.append("HF_TOKEN_USE:" + ln.strip()[:60])
    for c in cmds:
        if re.search(r'"\$PY" +(?:[\w./-]+\.py\b|"\$PKG)', c) and "p2r_sync.py" not in c and "env -u HF_TOKEN" not in c:
            v.append("HF_TOKEN_LEAKS_TO_A_CHILD:" + c[:60])
    if re.search(r"^\s*set\s+-[a-wyz]*x|\bbash\s+-x\b", text, re.M):
        v.append("XTRACE")
    # ---- fd 200, flock
    for c in cmds:
        if CHILD_RX.search(c) and "200>&-" not in c:
            v.append("FD200:" + c[:70])
    if re.search(r"flock[^\n]*\b9<(?!>)", code):
        v.append("FLOCK_READONLY_FD")
    up = fn_body(text, "upstream")
    for needle in ('"$P2/chain.lock"', '"$RLS1/chain.lock"', '"$S1/chain.lock"', "ZZS1AUX_DONE", "P2REP_S1_ABANDONED", "9<>"):
        if needle not in up:
            v.append("UPSTREAM_LACKS:" + needle)
    wc = fn_body(text, "wait_clear")
    if "P2REP_IDLE_POLLS:-3" not in wc or "gpu_busy" not in wc or 'refuse "$why"' not in wc:
        v.append("WAIT_CLEAR")
    if 'nv="${P2REP_NVSMI:-nvidia-smi}"' not in fn_body(text, "gpu_busy") or "|| return 0" not in fn_body(text, "gpu_busy"):
        v.append("GPU_BUSY_NOT_FAIL_CLOSED")
    # ---- order
    pos = [text.find(k) for k in ORDER]
    for k, p in zip(ORDER, pos):
        if p < 0:
            v.append("MISSING:" + k)
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        v.append("ORDER:" + ",".join(ORDER[i] for i in range(1, len(ORDER)) if pos[i] < pos[i - 1]))
    if 'gate_pass "$R/gate/gate.json"' not in text or re.search(r"grep[^\n]*verdict", code):
        v.append("GATE_JSON_NOT_PARSED")
    if '"^PREFLIGHT_OK"' not in text:
        v.append("NO_PREFLIGHT_CHECK")
    # ---- what it must NOT do
    for needle in DROPPED:
        if needle in code:
            v.append("DROPPED_STAGE_PRESENT:" + needle)
    for c in cmds:
        if FORBIDDEN_CMD_RX.search(c):
            v.append("FORBIDDEN_COMMAND:" + c[:60])
    for t in write_targets(code):
        if not allowed_target(t):
            v.append("WRITES_OUTSIDE_THE_ARMS_DIRS:" + t)
    if not re.search(r'of="\$R/\.dd_test"', code):
        v.append("DD_TARGET")
    if "pdm_label_complete.py --log" not in code or '--expect "$EXPECT_SERVED"' not in code or const(text, "EXPECT_SERVED") != "174912":
        v.append("PDM_AUDIT_OR_EXPECT")
    # ---- markers
    for mk in ("ZZP2REP_START", "ZZP2REP_WAITING", "ZZP2REP_REFUSED stage=", "ZZP2REP_PRECHECK_OK", "ZZP2REP_WAIT_OK", "ZZP2REP_INPUTS_OK", "ZZP2REP_GATE_OK", "ZZP2REP_LAUNCH_GATE_OK",
               "ZZP2REP_TRAIN_START", "ZZP2REP_TRAIN_DONE", "ZZP2REP_FINAL_OK", "ZZP2REP_DONE", "ZZP2REP_FINAL_FAILED", "ZZP2REP_FAILED"):
        if mk not in text:
            v.append("MARKER:" + mk)
    if re.search(r"ZZP2R_DONE|ZZP2_DONE", code.replace("ZZP2R_FINAL_OK", "")):
        v.append("P2_MARKER_WRITTEN")
    return v


def mutations(text):
    out = {}

    def sub(name, a, b, tag, count=1):
        assert a in text, (name, a)
        out[name] = (text.replace(a, b, count), tag)
    sub("seed dropped from ARGS", " --seed 1 --out $RUN", " --out $RUN", "ARGS_NOT_THE_PINNED")
    sub("seed 2", " --seed 1 --out $RUN", " --seed 2 --out $RUN", "ARGS_NOT_THE_PINNED")
    sub("P2's out dir in ARGS", " --seed 1 --out $RUN --resume", " --seed 1 --out /workspace/data/refe_runs/p2_joint_pdm --resume", "ARGS_NOT_THE_PINNED")
    sub("lr moved in ARGS", "--lr 5e-5", "--lr 6e-5", "ARGS_NOT_THE_PINNED")
    sub("RUN is P2's", "RUN=$DATA/refe_runs/p2_joint_pdm_s1 ", "RUN=$DATA/refe_runs/p2_joint_pdm ", "RUN_DIR")
    sub("OUT_REPO is P2's", "OUT_REPO=Sayood/tanitad-refe-p2-joint-pdm-s1 ", "OUT_REPO=Sayood/tanitad-refe-p2-joint-pdm ", "OUT_REPO")
    sub("final push without --repo", 'p2r_sync.py final --repo "$OUT_REPO" ', "p2r_sync.py final ", "SYNC_WITHOUT_REPO")
    sub("watcher without --repo", 'p2r_sync.py watch --repo "$OUT_REPO" ', "p2r_sync.py watch ", "SYNC_WITHOUT_REPO")
    sub("exit push without --repo", 'p2r_sync.py tick --repo "$OUT_REPO" ', "p2r_sync.py tick ", "SYNC_WITHOUT_REPO")
    sub("recover without --repo", 'p2r_sync.py recover --repo "$OUT_REPO" ', "p2r_sync.py recover ", "SYNC_WITHOUT_REPO")
    sub("recover into the shared work dir", '--work "$R/recover_work"', '--work "$DATA"', "WRITES_OUTSIDE")
    sub("restore (HF download) is back", '"$PY" p2r_sync.py recover', '"$PY" p2r_sync.py restore', "SYNC_SUBCOMMAND_NOT_ALLOWED")
    sub("profile is refe_p2", "launch_gate.py run --profile refe_p2rep", "launch_gate.py run --profile refe_p2", "PROFILE")
    sub("token_ok verifies another profile", "verify --profile refe_p2rep", "verify --profile refe_p2", "PROFILE")
    sub("token_ok forgets the argv", '--token "$tok" --args "$ARGS" --pkg "$PKG"', '--token "$tok" --pkg "$PKG"', "TOKEN_OK_NOT_VERIFYING")
    sub("no token check before the preflight", 'wait_clear\ntoken_ok || refuse "no launch-gate token verifies for this argv: the preflight and the trainer do not start"\n', "wait_clear\n", "NO_WAIT_OR_TOKEN_BEFORE_PREFLIGHT")
    sub("no token check before a launch", '  token_ok || refuse "the launch-gate token does not verify for this argv (attempt $att): train.py is NOT started"\n', "", "NO_WAIT_OR_TOKEN_BEFORE_LAUNCH")
    sub("no GPU check before a launch", "  wait_clear\n  token_ok || refuse \"the launch-gate token does not verify", "  token_ok || refuse \"the launch-gate token does not verify", "NO_WAIT_OR_TOKEN_BEFORE_LAUNCH")
    sub("the trainer gets an extra flag", '"$PY" train.py $ARGS >> "$R/train.log"', '"$PY" train.py $ARGS --seed 2 >> "$R/train.log"', "TRAINER_ARGV_NOT_JUST_ARGS")
    sub("a child keeps the token", 'env -u HF_TOKEN "$PY" pdm_label_complete.py', '"$PY" pdm_label_complete.py', "HF_TOKEN_LEAKS_TO_A_CHILD")
    sub("the trainer keeps the token", 'env -u HF_TOKEN "$PY" train.py $ARGS >>', '"$PY" train.py $ARGS >>', "HF_TOKEN_LEAKS_TO_A_CHILD")
    sub("the token is echoed", 'log "ZZP2REP_WAIT_OK (upstream over, GPU idle)"', 'log "ZZP2REP_WAIT_OK $HF_TOKEN"', "HF_TOKEN_USE")
    sub("xtrace", "set -u\nWORK=", "set -ux\nWORK=", "XTRACE")
    sub("trainer without fd 200 closed", '"$PY" train.py $ARGS >> "$R/train.log" 2>&1 200>&-', '"$PY" train.py $ARGS >> "$R/train.log" 2>&1', "FD200")
    sub("gate child without fd 200 closed", '--out "$R/gate" > "$R/gate.log" 2>&1 200>&-', '--out "$R/gate" > "$R/gate.log" 2>&1', "FD200")
    sub("lock probe on a read-only fd", '( flock -n 9 ) 9<> "$S1/chain.lock"', '( flock -n 9 ) 9< "$S1/chain.lock"', "FLOCK_READONLY_FD")
    sub("S1's lock is not probed", '  if [ -e "$S1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$S1/chain.lock"; then echo "the S1 chain still holds $S1/chain.lock"; return 2; fi\n', "", "UPSTREAM_LACKS")
    sub("RL-S1's lock is not probed", '  if [ -e "$RLS1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$RLS1/chain.lock"; then echo "the RL-S1 chain still holds $RLS1/chain.lock"; return 2; fi\n', "", "UPSTREAM_LACKS")
    sub("S1's DONE is not required", "ZZS1AUX_DONE", "ZZS1AUX_OTHER", "UPSTREAM_LACKS", count=-1)
    sub("S1 abandonment escape removed", "P2REP_S1_ABANDONED", "P2REP_S1_GONE", "UPSTREAM_LACKS", count=-1)
    sub("one idle poll is enough", "P2REP_IDLE_POLLS:-3", "P2REP_IDLE_POLLS:-1", "WAIT_CLEAR")
    sub("a GPU that cannot be asked is idle", '|| return 0\n  [ -n "$(echo "$apps"', '|| return 1\n  [ -n "$(echo "$apps"', "GPU_BUSY_NOT_FAIL_CLOSED")
    sub("the wait stage does not wait", "STAGE=wait\nwait_clear\n", "STAGE=wait\n", "ORDER")
    sub("the verification is gone", "p2rep_checks.py inputs", "p2rep_checks.py nothing", "MISSING", count=-1)
    sub("the PDM audit is gone", "pdm_label_complete.py --log", "pdm_label_complete.py --nolog", "PDM_AUDIT_OR_EXPECT")
    sub("the served count is not pinned", "EXPECT_SERVED=174912", "EXPECT_SERVED=177836", "PDM_AUDIT_OR_EXPECT")
    sub("the gate JSON is grepped", 'gate_pass "$R/gate/gate.json" 2>> $L', 'grep -q verdict "$R/gate/gate.json" 2>> $L', "GATE_JSON_NOT_PARSED")
    sub("the carve is back (it MOVES shared files)", '"$PY" p2r_gate_prep.py', '"$PY" p2r_carve.py --sets "$SETS" --stash x --heldout-logs y\n  "$PY" p2r_gate_prep.py', "DROPPED_STAGE_PRESENT")
    out["a download is back"] = (text.replace('dd if=/dev/zero', 'curl -o "$R/x" http://h\ndd if=/dev/zero', 1), "FORBIDDEN_COMMAND")
    out["a write into P2's dir"] = (text.replace('log "dd write test', 'touch "$P2/x"\nlog "dd write test', 1), "WRITES_OUTSIDE")
    out["a redirect into the PDM log"] = (text.replace('log "dd write test', 'echo x >> "$PDMLOG"\nlog "dd write test', 1), "WRITES_OUTSIDE")
    out["a --json into the shared tree"] = (text.replace('--json "$R/pdm_complete.json"', '--json "$P2/pdm_complete.json"', 1), "WRITES_OUTSIDE")
    out["a --out-sets into the shared bank"] = (text.replace('--out-sets "$R/gate_sets"', '--out-sets "$SETS/gate"', 1), "WRITES_OUTSIDE")
    out["an rm of the bank"] = (text.replace('log "dd write test', 'rm -rf "$SETS"\nlog "dd write test', 1), "WRITES_OUTSIDE")
    out["a dd into the data root"] = (text.replace('of="$R/.dd_test"', 'of="$DATA/.dd_test"', 1), "WRITES_OUTSIDE")
    out["a mkdir in the shared tree"] = (text.replace('log "dd write test', 'mkdir -p "$DATA/refe_sft2/x"\nlog "dd write test', 1), "WRITES_OUTSIDE")
    out["a P2 marker is written"] = (text.replace('log "ZZP2REP_DONE"', 'log "ZZP2R_DONE"', 1), "P2_MARKER_WRITTEN")
    out["a marker lost"] = (text.replace("ZZP2REP_FINAL_FAILED", "ZZP2REP_FINAL_LOST"), "MARKER")
    out["CRLF"] = (text.replace("\n", "\r\n"), "CRLF")
    return out


def static_checks(text):
    v = check_chain(text)
    ok(v == [], v)
    ms = mutations(text)
    for name, (mut, tag) in ms.items():
        ok(mut != text, name)
        got = check_chain(mut)
        ok(got and any(tag in g for g in got), f"check_chain stayed GREEN (or named another cause) on: {name}: want {tag}, got {got[:3]}")
    return len(ms)


# ============================================================================================ 5. behaviour of functions extracted from the chain
FLOCK_SHIM = '#!/bin/bash\nif [ "$2" = 200 ]; then exit "${FAKE_FLOCK200_RC:-0}"; fi\nf="$(readlink /proc/self/fd/9 2>/dev/null)"\nfor h in ${FAKE_LOCK_HELD//,/ }; do case "$f" in *"/$h/"*) exit 1;; esac; done\nexit 0\n'
NV_SHIM = '#!/bin/bash\nn=$(cat "$NVCOUNT" 2>/dev/null || echo 0)\necho $((n + 1)) > "$NVCOUNT"\nif [ "$n" -lt "${NV_BUSY_CALLS:-0}" ]; then echo 4242; fi\nexit "${NV_RC:-0}"\n'


def behaviour(text, tmp, rep_argv):
    tmp = Path(tmp)
    bindir = tmp / "bin"
    bindir.mkdir()
    write_sh(bindir / "flock", FLOCK_SHIM)
    write_sh(bindir / "nv.sh", NV_SHIM)
    data = tmp / "data"
    p2, rl, s1 = data / "refe_p2", data / "rls1", data / "s1aux"
    for d in (p2, rl, s1):
        d.mkdir(parents=True)
    logf = tmp / "wc.log"
    harness = tmp / "wc.sh"
    write_sh(harness, f'P2="{posix(p2)}"; RLS1="{posix(rl)}"; S1="{posix(s1)}"; L="{posix(logf)}"\n' + one_line_fn(text, "log") + "\nSTAGE=wait\nrefuse() { log \"ZZP2REP_REFUSED stage=$STAGE: $*\"; exit 1; }\n"
              + fn_body(text, "gpu_busy") + fn_body(text, "upstream") + fn_body(text, "wait_clear") + "wait_clear\necho CLEAR\n")

    def wc(**env):
        logf.write_text("")
        (tmp / "nvc").write_text("0")
        e = {"PATH": posix(bindir) + os.pathsep + os.environ["PATH"], "P2REP_NVSMI": posix(bindir / "nv.sh"), "NVCOUNT": posix(tmp / "nvc"), "P2REP_POLL_S": "0", "P2REP_IDLE_POLLS": "3", **env}
        rc, out, err = bash_run(f'bash "{posix(harness)}"', e, timeout=60)
        return rc, out, logf.read_text()
    (s1 / "chain.log").write_text("2026-10-14T00:00:00 ZZS1AUX_DONE arms: N-s0\n")
    rc, out, lg = wc(); ok(rc == 0 and out == "CLEAR" and "WAITING" not in lg, (rc, out, lg))                          # nothing upstream, the GPU idle: 3 polls, no wait
    rc, out, lg = wc(NV_BUSY_CALLS="2"); ok(rc == 0 and out == "CLEAR" and lg.count("ZZP2REP_WAITING") == 2 and "GPU" in lg, (rc, out, lg))      # busy twice, then idle: WAITS, then proceeds
    ok(int((tmp / "nvc").read_text()) == 5, (tmp / "nvc").read_text())                                                    # 2 busy polls + 3 consecutive idle polls
    rc, out, lg = wc(NV_BUSY_CALLS="1000", P2REP_POLL_S="1", P2REP_WAIT_MAX_S="2"); ok(rc == 1 and "ZZP2REP_REFUSED stage=wait: waited" in lg, (rc, out, lg))   # a GPU that never frees: bounded, named
    rc, out, lg = wc(NV_RC="9", P2REP_POLL_S="1", P2REP_WAIT_MAX_S="1"); ok(rc == 1 and "waited" in lg and "GPU" in lg, lg)           # what cannot be proven idle is busy
    for name, d in (("P2", p2), ("RL-S1", rl), ("S1", s1)):
        (d / "chain.lock").write_text("")
        held = {"P2": "refe_p2", "RL-S1": "rls1", "S1": "s1aux"}[name]
        rc, out, lg = wc(FAKE_LOCK_HELD=held, P2REP_POLL_S="1", P2REP_WAIT_MAX_S="2")                                      # THAT chain holds its lock: the chain WAITS (bounded here), naming it
        ok(rc == 1 and f"the {name} chain still holds" in lg and "ZZP2REP_WAITING" in lg and "waited" in lg, (name, lg))
        rc, out, lg = wc(FAKE_LOCK_HELD="some_other_dir"); ok(rc == 0 and "WAITING" not in lg, (name, lg))               # a lock file nobody holds is not a wait
        (d / "chain.lock").unlink()
    (s1 / "chain.lock").write_text("")
    rc, out, lg = wc(FAKE_LOCK_HELD="s1aux,rls1", P2REP_POLL_S="1", P2REP_WAIT_MAX_S="1"); ok(rc == 1 and "waited" in lg, lg)  # S1 running (lock held): waits even though its log has DONE from an earlier run
    (s1 / "chain.lock").unlink()
    (s1 / "chain.log").write_text("2026-10-14T00:00:00 ZZS1AUX_ARM_DONE N-s0\n")                                           # S1's lock is free and it never logged DONE: killed, or never started
    rc, out, lg = wc(); ok(rc == 1 and "ZZP2REP_REFUSED stage=wait: S1 is neither running nor finished" in lg and "WAITING" not in lg, lg)      # REFUSED, never an endless wait
    rc, out, lg = wc(P2REP_S1_ABANDONED="1"); ok(rc == 0 and out == "CLEAR", lg)                                           # ... unless the Master Mind said so
    rc, out, lg = wc(P2REP_S1_ABANDONED="0"); ok(rc == 1, lg)                                                              # only "1" counts
    (s1 / "chain.log").unlink()
    rc, out, lg = wc(); ok(rc == 1 and "neither running nor finished" in lg, lg)                                           # no S1 log at all
    (s1 / "chain.log").write_text("2026-10-14T00:00:00 ZZS1AUX_DONE arms: N-s0\n")
    # ---- token_ok, extracted verbatim, against tokens minted by the gate's own mint() for refe_p2rep
    h3 = tmp / "h3.sh"
    write_sh(h3, fn_body(text, "token_ok"))
    key = GT.load_key(tmp / "k" / "gate.key", forbid_under=[PKGROOT])
    lgdir, rdir = tmp / "lg", tmp / "r"
    rdir.mkdir()
    prof = L.PROFILES["refe_p2rep"]

    def mint(argv, profile="refe_p2rep", checks=None):
        shutil.rmtree(lgdir, ignore_errors=True)
        return GT.mint(profile=profile, commit="a" * 40, tree_man=GT.tree_manifest(PKGROOT), argv=argv, trainer="refe/train.py", data=GT.data_manifest(argv, prof["data_flags"]),
                       checks=checks or {"G-HYG": {"status": "PASS"}}, reasons=[], key=key, out_dir=lgdir)

    def tok(args, keyfile=None):
        env = {"PY": posix(PY), "PKG": posix(PKGROOT), "LG": posix(lgdir), "R": posix(rdir), "ARGS": args, "PYTHONIOENCODING": "utf-8", "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*", "TANITAD_GATE_KEY_FILE": posix(keyfile or (tmp / "k" / "gate.key"))}
        return bash_run(f'bash -c \'. "{posix(h3)}"; token_ok\'', env)[0]
    good = " ".join(rep_argv)
    v, pth, _ = mint(rep_argv)
    ok(v == "PASS" and tok(good) == 0, "a token minted for the replicate argv must verify for it")
    ok(tok(good.replace("--seed 1", "--seed 2")) != 0 and tok(good.replace(" --seed 1", "")) != 0, "a token must not verify for another seed / no seed")
    ok(tok(good.replace("p2_joint_pdm_s1", "p2_joint_pdm")) != 0, "a token must not verify for P2's own --out")
    ok(tok(good.replace("--lr 5e-5", "--lr 6e-5")) != 0)
    mint(rep_argv, profile="refe_p2"); ok(tok(good) != 0, "a token minted for profile refe_p2 must not verify as refe_p2rep")
    mint(L.P2_ARGV.split(), profile="refe_p2rep"); ok(tok(good) != 0, "a token minted over P2's argv must not verify for the replicate's")
    mint(rep_argv, checks={"G-HYG": {"status": "FAIL"}}); ok(tok(good) != 0, "a FAIL token must not verify")
    mint(rep_argv); (lgdir / "PASS_aaaaaaaaaaaa.json").write_text((lgdir / "PASS_aaaaaaaaaaaa.json").read_text().replace('"G-HYG"', '"G-HYX"'), encoding="utf-8")
    ok(tok(good) != 0, "an edited token must not verify")
    mint(rep_argv); ok(tok(good, keyfile=tmp / "k2" / "gate.key") != 0, "a token signed by another key must not verify")
    shutil.rmtree(lgdir); lgdir.mkdir(); ok(tok(good) != 0, "no token => not ok")


# ============================================================================================ 6/7. the pod in miniature
INIT_BYTES = b"init-checkpoint-" * 64
WEIGHTS_BYTES = b"dinov3-vitl16-weights-" * 32
SET_ROWS = [("L1", 0), ("L1", 1), ("L2", 0), ("L2", 1), ("L3", 0), ("L3", 1)]       # 6 served training sets (the sandbox's 174,912)


def set_row(log, i, ck=7):
    return json.dumps({"kind": "onpolicy_set", "log_name": log, "token": f"t{i}", "step": i, "rank": 0, "ckpt_step": ck, "label_version": 3}) + "\n"


STUB_COMMON = r'''
import json, os, sys
a = sys.argv[1:]
def val(f): return a[a.index(f) + 1] if f in a else None
def rec(**kw):
    kw["hf_token"] = "HF_TOKEN" in os.environ
    open(os.environ["STUB_LOG"], "a").write(json.dumps(kw) + "\n")
'''
STUB_TRAIN = STUB_COMMON + r'''
import hashlib
out, pre = val("--out"), "--preflight" in a
rec(tool="train", preflight=pre, argv=a, out=out, seed=val("--seed"))
if pre:
    if os.environ.get("FAKE_PREFLIGHT_FAIL") == "1": print("preflight said no"); sys.exit(3)
    print("PREFLIGHT_OK"); sys.exit(0)
st = os.path.join(os.environ["STUB_STATE"], "train_attempts")
n = int(open(st).read()) if os.path.exists(st) else 0
open(st, "w").write(str(n + 1))
if n < int(os.environ.get("FAKE_TRAIN_FAIL_FIRST", "0")): sys.exit(1)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "ckpt_last.pt"), "wb").write(b"ckpt")
open(os.path.join(out, "model_final.pt"), "wb").write(b"final-model-bytes")
json.dump({"done": True, "model_final_sha256": hashlib.sha256(b"final-model-bytes").hexdigest()}, open(os.path.join(out, "summary.json"), "w"))
'''
STUB_GATE = STUB_COMMON + r'''
import hashlib
cmd = a[0]
rec(tool="launch_gate", cmd=cmd, profile=val("--profile"), args=val("--args"))
if cmd == "run":
    if os.environ.get("FAKE_GATE_REFUSE") == "1": print("ZZGATE-REFUSEDZZ"); sys.exit(1)
    if not os.path.isdir(val("--gate-sets") or "") or not os.path.isfile(val("--p2-gate-json") or ""): print("no gate inputs"); sys.exit(1)
    os.makedirs(val("--out-dir"), exist_ok=True)
    json.dump({"kind": "launch-gate-token", "verdict": "PASS", "argv": val("--args"), "profile": val("--profile")}, open(os.path.join(val("--out-dir"), "PASS_cccccccccccc.json"), "w"))
    print("ZZGATE-PASS-ccccccccccccZZ"); sys.exit(0)
tok = json.load(open(val("--token")))
good = tok.get("argv") == val("--args") and val("--profile") == "refe_p2rep" and os.environ.get("FAKE_VERIFY_FAIL") != "1"
if val("--json"): json.dump({"verdict": "MATCH" if good else "REFUSED"}, open(val("--json"), "w"))
print("ZZGATE-MATCHZZ" if good else "ZZGATE-REFUSEDZZ"); sys.exit(0 if good else 5)
'''
STUB_P2GATE = STUB_COMMON + r'''
rec(tool="p2_gate", init=val("--init-from"), onpolicy=val("--onpolicy"), pdm=val("--pdm-labels"), out=val("--out"))
os.makedirs(val("--out"), exist_ok=True)
v = "FAIL" if os.environ.get("FAKE_P2GATE_FAIL") == "1" else "PASS"
json.dump({"verdict": v, "gates": {"G1_warm": {"ok": True}, "G2_pair": {"ok": True}, "G3_pdm": {"ok": True}, "G4_smoke": {"ok": True}}}, open(os.path.join(val("--out"), "gate.json"), "w"))
'''
STUB_GATEPREP = STUB_COMMON + r'''
rec(tool="gate_prep", sets=val("--sets"), pdm=val("--pdm"), out_sets=val("--out-sets"), out_pdm=val("--out-pdm"))
for d in (val("--out-sets"), val("--out-pdm")): os.makedirs(d, exist_ok=True)
open(os.path.join(val("--out-sets"), "onpolicy_gate.jsonl"), "w").write("{}\n")
'''
STUB_SYNC = STUB_COMMON + r'''
import hashlib, time
cmd = a[0]
rec(tool="p2r_sync", cmd=cmd, repo=val("--repo"), run=val("--run"), work=val("--work"), argv=a)
if cmd == "watch":
    t0 = time.time()
    while not os.path.exists(val("--stop-file")) and time.time() - t0 < 120: time.sleep(0.05)
    sys.exit(0)
if cmd == "tick":
    print("ZZSYNC_TICK_OK stub"); sys.exit(0)
if cmd == "recover":
    if os.environ.get("FAKE_RECOVER_FAIL") == "1": print("ZZRECOVER_FAIL stub"); sys.exit(1)
    print("ZZRECOVER_OK labels=no ckpt=no"); sys.exit(0)
if cmd == "final":
    run = val("--run")
    if os.environ.get("FAKE_FINAL_FAIL") == "1": print("ZZP2R_FINAL_FAIL stub"); sys.exit(1)
    summ = json.load(open(os.path.join(run, "summary.json")))
    line = "ZZP2R_FINAL_OK model_final sha256 %s remote=sha256-verified" % summ["model_final_sha256"]
    open(val("--chain-log"), "a").write("2026 " + line + "\n")
    print(line); sys.exit(0)
sys.exit(2)
'''
SHIM = r'''
import os, subprocess, sys
a = sys.argv[1:]
STUBS = {"train.py": "train", "launch_gate.py": "gate", "p2_gate.py": "p2gate", "p2r_gate_prep.py": "gateprep", "p2r_sync.py": "sync"}
name = os.path.basename(a[0]) if a else ""
if name in STUBS:
    sys.exit(subprocess.call([sys.executable, os.path.join(os.environ["STUBDIR"], STUBS[name] + ".py")] + a[1:]))
sys.exit(subprocess.call([sys.executable] + a))
'''


class Pod:
    """A temp WORK with the pod's inputs in miniature (what P2 built), the REAL helper / pdm_label_complete / p2r_sync PINS patched to the fixtures, stubs for everything heavy, and the REAL chain text."""

    SHARED = ("refe_sft2", "refe_navtrain", "refe_p2", "refe_runs/vitl16_navtrain10_grow_tau0.3", "navtrain_pixels", "navtrain_dbs", "backbones", "rls1", "s1aux")

    def __init__(self, tmp, text):
        self.tmp = Path(tmp)
        W = self.W = self.tmp / "work"
        shutil.rmtree(W, ignore_errors=True)
        self.D = W / "data"
        self.R = self.D / "refe_p2rep"
        self.P2 = self.D / "refe_p2"
        self.RUN = self.D / "refe_runs" / "p2_joint_pdm_s1"
        self.pk = W / "refe-plan-p2rep"
        self.stubdir = self.tmp / "stubs"
        self.stub_log = self.tmp / "stub.jsonl"
        self.state = self.tmp / "state"
        for d in (self.stubdir, self.state, self.tmp / "bin", self.pk / "refe", self.pk / "code", self.pk / "splits", self.D / "backbones"):
            d.mkdir(parents=True, exist_ok=True)
        self.build_inputs()
        # ---- the chain text: WORK, the registered served count, and the waits shortened
        t = text
        for a, b in (("WORK=/workspace;", f"WORK={posix(W)};"), ("EXPECT_SERVED=174912", f"EXPECT_SERVED={len(SET_ROWS)}"), ("sleep 5 200>&-", "sleep 0.2 200>&-"), ("sleep 60 200>&-", "sleep 0 200>&-")):
            assert a in t, a
            t = t.replace(a, b)
        self.text = t
        self.script = self.tmp / "p2rep_sandbox.sh"
        write_sh(self.script, t)
        # ---- the package: every file the chain requires (empty), the REAL helpers, then the manifest over the whole tree
        files = re.search(r'for f in (.*?); do\n  \[ -f "\$PKG/\$f" \]', t, re.S).group(1).replace("\\\n", " ").split()
        self.shiplist = files
        for f in files:
            (self.pk / f).parent.mkdir(parents=True, exist_ok=True)
            if not (self.pk / f).exists():
                (self.pk / f).write_text("", encoding="utf-8")
        for real in ("gate_token.py", "hf_egress.py", "p2r_carve.py", "select_bank_subset.py", "onpolicy_relabel_pdm.py", "pdm_label_complete.py"):
            shutil.copy(HERE / real, self.pk / "refe" / real)
        h = HELPER.read_text(encoding="utf-8")
        h2 = h.replace("EXPECT_SERVED = 174912", f"EXPECT_SERVED = {len(SET_ROWS)}")
        assert h2 != h
        (self.pk / "refe" / "p2rep_checks.py").write_text(h2, encoding="utf-8", newline="\n")
        s = (HERE / "p2r_sync.py").read_text(encoding="utf-8")
        s2 = re.sub(r'"model_final": \("[0-9a-f]{32}", \d+, \[".pt"\]\)', f'"model_final": ("{hashlib.md5(INIT_BYTES).hexdigest()}", {len(INIT_BYTES)}, [".pt"])', s, count=1)
        assert s2 != s
        (self.pk / "refe" / "p2r_sync.py").write_text(s2, encoding="utf-8", newline="\n")
        (self.pk / "splits" / "sft_heldout_logs.txt").write_text("H1\nH2\n", encoding="utf-8")
        (self.pk / "code" / "pod_train.sh").write_text("declare -A W_SHA=([vitl16]=%s [vits16]=%s)\n" % (hashlib.sha256(WEIGHTS_BYTES).hexdigest(), "0" * 64), encoding="utf-8", newline="\n")
        self.manifest()
        # ---- the pod: a python that routes the heavy scripts to stubs, flock / nvidia-smi shims
        for name, src in (("train", STUB_TRAIN), ("gate", STUB_GATE), ("p2gate", STUB_P2GATE), ("gateprep", STUB_GATEPREP), ("sync", STUB_SYNC)):
            (self.stubdir / f"{name}.py").write_text(src, encoding="utf-8", newline="\n")
        (self.tmp / "shim.py").write_text(SHIM, encoding="utf-8", newline="\n")
        fakepy = self.tmp / "bin" / "fakepy.sh"
        write_sh(fakepy, f'#!/bin/bash\nexec "{posix(PY)}" "{posix(self.tmp / "shim.py")}" "$@"\n')
        (W / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{posix(fakepy)}"\n')
        write_sh(self.tmp / "bin" / "flock", FLOCK_SHIM)
        write_sh(self.tmp / "bin" / "nvidia-smi", NV_SHIM)
        self.base_env = {"PATH": posix(self.tmp / "bin") + os.pathsep + os.environ["PATH"], "HF_TOKEN": "hf_notarealtoken", "P2REP_DD_MB": "2", "P2REP_POLL_S": "0", "P2REP_SYNC_MIN": "1",
                         "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*", "STUB_LOG": posix(self.stub_log), "STUB_STATE": posix(self.state), "STUBDIR": posix(self.stubdir), "NVCOUNT": posix(self.tmp / "nvc"), "PYTHONIOENCODING": "utf-8"}

    def build_inputs(self):
        D = self.D
        w = lambda p, b: (p.parent.mkdir(parents=True, exist_ok=True), p.write_bytes(b) if isinstance(b, bytes) else p.write_text(b, encoding="utf-8", newline="\n"))   # noqa: E731
        w(D / "refe_runs" / "vitl16_navtrain10_grow_tau0.3" / "model_final.pt", INIT_BYTES)
        w(D / "backbones" / "dinov3-vitl16" / "model.safetensors", WEIGHTS_BYTES)
        tg = D / "refe_navtrain" / "train_grow"
        w(tg / "calib_table.json", "{}")
        w(tg / "scorer_targets_r0.jsonl", "{}\n")
        w(tg / "targets_rank0.jsonl", "{}\n")
        w(D / "refe_navtrain" / "onpolicy" / "sets" / "onpolicy_r0.jsonl", "".join(set_row(lg, i) for lg, i in SET_ROWS))
        w(D / "refe_navtrain" / "onpolicy" / "sets_full" / "onpolicy_r0.jsonl", "".join(set_row(lg, i) for lg, i in SET_ROWS) + set_row("H1", 9))
        w(D / "refe_sft2" / "train_pdm" / "pdm_onpolicy_r0.jsonl", "".join(json.dumps({"key": [lg, f"t{i}", i, 0], "ckpt_step": 7}) + "\n" for lg, i in SET_ROWS))
        w(D / "refe_sft2" / "pdm_train.log", f"served sets: {len(SET_ROWS)}\nZZRELABEL_PDM_DONE {len(SET_ROWS)} 0 0\n")
        for n in range(1, 33):
            w(D / "navtrain_pixels" / f".done_{n}", "")
            (D / "navtrain_pixels" / f"navtrain_current_{n}").mkdir(parents=True, exist_ok=True)
        w(self.P2 / "weights.log", "=== weights ===\nZZOK weights ZZ\n")
        w(self.P2 / "carve.log", f"ZZCARVE_OK {len(SET_ROWS)} 2\n")
        w(self.P2 / "pix.log", "ZZOK pix shard 32 ZZ\nZZOK pix 32/32 shards ZZ\n")
        w(self.P2 / "chain.log", "2026-10-13T20:00:00 ZZP2R_DONE\n")
        w(D / "rls1" / "chain.log", "2026-10-14T01:00:00 ZZRLS1_DONE arms: R0\n")
        w(D / "s1aux" / "chain.log", "2026-10-14T02:00:00 ZZS1AUX_DONE arms: N-s0\n")
        (D / "navtrain_dbs").mkdir(parents=True, exist_ok=True)
        w(D / "navtrain_dbs" / "a.db", b"")

    def manifest(self):
        GT.write_json(self.pk / "SHIP_MANIFEST.json", GT.ship_manifest(self.pk))

    def snapshot(self):
        out = {}
        for rel in self.SHARED:
            base = self.D / rel
            for dp, dn, fn in os.walk(base):
                for f in fn:
                    p = Path(dp) / f
                    st = p.stat()
                    out[p.relative_to(self.D).as_posix()] = (st.st_size, st.st_mtime_ns)
                for d in dn:
                    out[(Path(dp) / d).relative_to(self.D).as_posix() + "/"] = ("dir",)
        for p in (self.W / "teacher_env.sh",):
            st = p.stat()
            out[p.name] = (st.st_size, st.st_mtime_ns)
        return out

    def reset(self):
        for d in (self.R, self.RUN, self.state):
            shutil.rmtree(d, ignore_errors=True)
        self.state.mkdir(parents=True, exist_ok=True)
        if self.stub_log.exists():
            self.stub_log.unlink()

    def run(self, env_extra=None, drop=(), fresh=False, timeout=600):
        if fresh:
            self.reset()
        (self.tmp / "nvc").write_text("0")
        e = {k: v for k, v in os.environ.items() if k not in drop}
        e.update({k: v for k, v in {**self.base_env, **(env_extra or {})}.items() if k not in drop})
        r = subprocess.run([BASH, posix(self.script)], capture_output=True, text=True, env=e, timeout=timeout)
        return r.returncode, r.stdout + r.stderr, self.log()

    def inputs(self, *extra):
        r = subprocess.run([PY, posix(self.pk / "refe" / "p2rep_checks.py"), "inputs", "--data", posix(self.D), "--pkg", posix(self.pk), "--p2-dir", posix(self.P2),
                            "--json", posix(self.tmp / "inputs.json"), *extra], capture_output=True, text=True, encoding="utf-8")
        return r.returncode, (r.stdout + r.stderr).strip()

    def log(self):
        p = self.R / "chain.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def calls(self, tool=None, cmd=None):
        if not self.stub_log.exists():
            return []
        rows = [json.loads(ln) for ln in self.stub_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [r for r in rows if (tool is None or r["tool"] == tool) and (cmd is None or r.get("cmd") == cmd)]

    def marks(self):
        d = self.R / "stage"
        return sorted(p.stem for p in d.glob("*.ok")) if d.exists() else []


@contextlib.contextmanager
def tweak(path, new=None, delete=False):
    """Change one fixture file for the duration of the block; restore its bytes AND mtime after."""
    path = Path(path)
    old = path.read_bytes() if path.exists() else None
    st = path.stat() if path.exists() else None
    try:
        if delete:
            path.unlink()
        else:
            path.write_bytes(new if isinstance(new, bytes) else new.encode("utf-8"))
        yield
    finally:
        if old is not None:
            path.write_bytes(old)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def helper_checks(pod):
    D, P2 = pod.D, pod.P2
    snap0 = pod.snapshot()
    rc, out = pod.inputs()
    ok(rc == 0 and out.startswith("ZZP2REP_INPUTS_OK") and f"served={len(SET_ROWS)}" in out and "heldout_rows=0" in out and "pix=32/32" in out, out)
    ok(pod.snapshot() == snap0, "p2rep_checks.py inputs WROTE into the shared tree")                                  # READ-ONLY, byte and mtime
    det = json.loads((pod.tmp / "inputs.json").read_text())
    ok(det["verdict"] == "OK" and det["served"] == len(SET_ROWS) and det["heldout_rows"] == 0 and det["pdm_files"] == 1 and det["navtrain_dbs"] == 1 and det["init_md5"] == hashlib.md5(INIT_BYTES).hexdigest())

    def refused(frag, *extra):
        rc, out = pod.inputs(*extra)
        ok(rc == 1 and out.startswith("ZZP2REP_INPUTS_REFUSED") and frag in out, (frag, out))
    init = D / "refe_runs" / "vitl16_navtrain10_grow_tau0.3" / "model_final.pt"
    with tweak(init, INIT_BYTES[:-1] + b"X"): refused("init checkpoint md5")
    with tweak(init, INIT_BYTES + b"X"): refused("bytes")
    with tweak(init, delete=True): refused("init checkpoint is missing")
    wf = D / "backbones" / "dinov3-vitl16" / "model.safetensors"
    with tweak(wf, WEIGHTS_BYTES[:-1] + b"X"): refused("weights sha256")
    with tweak(wf, delete=True): refused("weights are missing")
    with tweak(pod.pk / "code" / "pod_train.sh", "no pin here\n"): refused("no W_SHA[vitl16] pin")
    with tweak(D / "refe_navtrain" / "train_grow" / "calib_table.json", delete=True): refused("train_grow is incomplete")
    sets = D / "refe_navtrain" / "onpolicy" / "sets" / "onpolicy_r0.jsonl"
    base_rows = sets.read_text(encoding="utf-8")
    with tweak(sets, base_rows + set_row("L4", 0)): refused(f"serves {len(SET_ROWS) + 1} sets")                       # the carve undone in the other direction: an extra set
    with tweak(sets, "".join(set_row(lg, i) for lg, i in SET_ROWS[:-1])): refused(f"serves {len(SET_ROWS) - 1} sets")
    with tweak(sets, base_rows + set_row("H2", 5)): refused("SFT held-out logs")                                       # a held-out log's row survived (or came back)
    with tweak(sets, base_rows + set_row("H1", 5, ck=1)): refused("SFT held-out logs")                                 # ... even a superseded one that serves nothing
    with tweak(sets, delete=True): refused("no onpolicy_*.jsonl")
    with tweak(D / "refe_sft2" / "train_pdm" / "pdm_onpolicy_r0.jsonl", delete=True): refused("no pdm_*.jsonl")
    with tweak(D / "navtrain_pixels" / ".done_7", delete=True): refused(".done_1..32 present for 31")
    shutil.rmtree(D / "navtrain_pixels" / "navtrain_current_9")
    refused("fewer than 32 navtrain_current_*")
    (D / "navtrain_pixels" / "navtrain_current_9").mkdir()
    # P2's own markers: each absent => refused; the waiver skips ONLY them
    for f, frag in (("weights.log", "weights_log"), ("chain.log", "p2r_done"), ("carve.log", "carve_ok"), ("pix.log", "pix_ok")):
        with tweak(P2 / f, delete=True):
            refused(frag)
            rcw, outw = pod.inputs("--waive-p2-logs")
            ok(rcw == 0 and outw.startswith("ZZP2REP_INPUTS_OK"), outw)
    with tweak(P2 / "carve.log", "ZZCARVE_OK 999 2\n"): refused("carve_ok")
    with tweak(P2 / "pix.log", "ZZOK pix 31/32 shards ZZ\n"): refused("pix_ok")
    with tweak(P2 / "chain.log", "ZZP2R_FINAL_FAILED\n"): refused("p2r_done")
    with tweak(init, delete=True):                                                                                  # the waiver does NOT excuse a wrong input
        rcw, outw = pod.inputs("--waive-p2-logs")
        ok(rcw == 1 and "init checkpoint" in outw)
    # the tree check, both directions
    r = subprocess.run([PY, posix(pod.pk / "refe" / "p2rep_checks.py"), "tree", "--pkg", posix(pod.pk)], capture_output=True, text=True, encoding="utf-8")
    ok(r.returncode == 0 and r.stdout.startswith("ZZP2REP_TREE_OK"), r.stdout + r.stderr)
    f = pod.pk / "refe" / "train.py"
    with tweak(f, "# edited after the manifest\n"):
        r = subprocess.run([PY, posix(pod.pk / "refe" / "p2rep_checks.py"), "tree", "--pkg", posix(pod.pk)], capture_output=True, text=True, encoding="utf-8")
        ok(r.returncode == 1 and "ZZP2REP_TREE_REFUSED" in r.stdout and "train.py" in r.stdout, r.stdout)
    with tweak(pod.pk / "SHIP_MANIFEST.json", delete=True):
        r = subprocess.run([PY, posix(pod.pk / "refe" / "p2rep_checks.py"), "tree", "--pkg", posix(pod.pk)], capture_output=True, text=True, encoding="utf-8")
        ok(r.returncode == 1 and "missing" in r.stdout)
    ok(pod.snapshot() == snap0, "the helper checks left the shared tree changed")
    # the helper has no write path but --json (static)
    hsrc = HELPER.read_text(encoding="utf-8")
    ok(not re.search(r"""open\([^)]*["'][wa+]b?["']""", hsrc) and "shutil." not in hsrc and "os.remove" not in hsrc and "unlink" not in hsrc and "os.rename" not in hsrc, "p2rep_checks.py must only read")
    ok(len(re.findall(r"write_text", hsrc)) == 1)


def sandbox_runs(pod):
    def refused(out, lg, stage, frag=""):
        ok(f"ZZP2REP_REFUSED stage={stage}:" in lg and frag in lg, (stage, frag, lg[-700:], out[-300:]))

    def no_heavy_calls(tools=("train", "launch_gate", "p2_gate", "gate_prep")):
        c = [r["tool"] for r in pod.calls() if r["tool"] in tools]
        ok(not c, c)
    snap0 = pod.snapshot()
    # ---- precheck refusals (seconds, before any wait)
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_FLOCK200_RC": "1"}); ok(rc == 1 and "ZZP2REP_REFUSED another p2rep_pod.sh holds" in out, out)      # a second chain
    rc, out, lg = pod.run(fresh=True, drop=("HF_TOKEN",)); ok(rc == 1, out); refused(out, lg, "precheck", "HF_TOKEN is not set")
    gone = pod.pk / "refe" / "test_p2rep_chain.py"
    gone.rename(gone.with_suffix(".bak"))
    rc, out, lg = pod.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "test_p2rep_chain.py missing")
    gone.with_suffix(".bak").rename(gone)
    te = pod.W / "teacher_env.sh"
    te.rename(te.with_suffix(".bak"))
    rc, out, lg = pod.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "teacher_env.sh is missing")
    te.with_suffix(".bak").rename(te)
    ft = pod.pk / "refe" / "gate_checks.py"
    ft.write_text("# edited after the manifest\n")
    rc, out, lg = pod.run(fresh=True); ok(rc == 1, out); refused(out, lg, "precheck", "not its SHIP_MANIFEST.json")
    ft.write_text("")
    no_heavy_calls()
    # ---- the WAIT: nothing past `wait` runs while S1 (or RL-S1, or P2) holds its lock, or the GPU is busy
    (pod.D / "s1aux" / "chain.lock").write_text("")
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_LOCK_HELD": "s1aux", "P2REP_POLL_S": "1", "P2REP_WAIT_MAX_S": "2"})
    ok(rc == 1 and "ZZP2REP_WAITING the S1 chain still holds" in lg, lg); refused(out, lg, "wait", "waited")
    ok("ZZP2REP_WAIT_OK" not in lg and "ZZP2REP_INPUTS_OK" not in lg and not pod.marks(), (lg, pod.marks()))
    no_heavy_calls()
    (pod.D / "rls1" / "chain.lock").write_text("")
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_LOCK_HELD": "rls1", "P2REP_POLL_S": "1", "P2REP_WAIT_MAX_S": "2"}); ok(rc == 1 and "the RL-S1 chain still holds" in lg, lg)
    (pod.D / "rls1" / "chain.lock").unlink()
    (pod.D / "s1aux" / "chain.lock").unlink()
    s1log = pod.D / "s1aux" / "chain.log"
    with tweak(s1log, "2026-10-14T02:00:00 ZZS1AUX_ARM_DONE N-s0\n"):                                                  # S1 free, never DONE: REFUSED (not an endless wait)
        rc, out, lg = pod.run(fresh=True); ok(rc == 1 and "WAITING" not in lg, lg); refused(out, lg, "wait", "S1 is neither running nor finished")
        rc, out, lg = pod.run(fresh=True, env_extra={"P2REP_S1_ABANDONED": "1", "P2REP_STOP_AFTER": "wait"}); ok(rc == 0 and "ZZP2REP_WAIT_OK" in lg, lg)    # ... unless the Master Mind said so
    rc, out, lg = pod.run(fresh=True, env_extra={"NV_BUSY_CALLS": "2", "P2REP_STOP_AFTER": "wait"})                      # a busy GPU: waits, then proceeds
    ok(rc == 0 and lg.count("ZZP2REP_WAITING a process is on the GPU") == 2 and "ZZP2REP_WAIT_OK" in lg, lg)
    rc, out, lg = pod.run(fresh=True, env_extra={"NV_BUSY_CALLS": "1000", "P2REP_POLL_S": "1", "P2REP_WAIT_MAX_S": "2"}); ok(rc == 1, out); refused(out, lg, "wait", "a process is on the GPU")
    # ---- the verification refuses every wrong shared input, runs NO heavy tool, and writes NOTHING into the shared tree
    for what, ctx, frag in (("a corrupted init checkpoint", tweak(pod.D / "refe_runs" / "vitl16_navtrain10_grow_tau0.3" / "model_final.pt", INIT_BYTES[:-1] + b"X"), "init checkpoint"),
                            ("a carve that was undone", tweak(pod.D / "refe_navtrain" / "onpolicy" / "sets" / "onpolicy_r0.jsonl", "".join(set_row(lg_, i) for lg_, i in SET_ROWS) + set_row("H1", 9)), "held-out"),
                            ("a missing pixel shard", tweak(pod.D / "navtrain_pixels" / ".done_3", delete=True), "pixel shards are incomplete"),
                            ("P2's own marker gone", tweak(pod.P2 / "carve.log", delete=True), "carve_ok")):
        with ctx:
            rc, out, lg = pod.run(fresh=True); ok(rc == 1, (what, out)); refused(out, lg, "verify", frag)
        no_heavy_calls()
    with tweak(pod.D / "refe_sft2" / "train_pdm" / "pdm_onpolicy_r0.jsonl", "".join(json.dumps({"key": [lg_, f"t{i}", i, 0], "ckpt_step": 7}) + "\n" for lg_, i in SET_ROWS[:-1])):
        rc, out, lg = pod.run(fresh=True); ok(rc == 1, out); refused(out, lg, "verify", "PDM labels are not complete")                  # P2's OWN completeness audit, on the shared labels
    with tweak(pod.D / "refe_sft2" / "pdm_train.log", delete=True):
        rc, out, lg = pod.run(fresh=True); ok(rc == 1, out); refused(out, lg, "verify", "pdm_train.log is missing")
    no_heavy_calls()
    ok(pod.snapshot() == snap0, "a refused verification left the shared tree changed")
    with tweak(pod.P2 / "chain.log", delete=True):
        rc, out, lg = pod.run(fresh=True, env_extra={"P2REP_STOP_AFTER": "verify"}); ok(rc == 1, out); refused(out, lg, "verify", "p2r_done")
        rc, out, lg = pod.run(fresh=True, env_extra={"P2REP_P2_PROVENANCE_WAIVED": "1", "P2REP_STOP_AFTER": "verify"}); ok(rc == 0 and "ZZP2REP_INPUTS_OK" in lg, lg)
    # ---- dry runs end where they are told to
    for stage, last in (("precheck", "ZZP2REP_PRECHECK_OK"), ("wait", "ZZP2REP_WAIT_OK"), ("verify", "ZZP2REP_INPUTS_OK"), ("recover", "ZZP2REP_RECOVER"), ("gate", "ZZP2REP_GATE_OK"), ("launch_gate", "ZZP2REP_LAUNCH_GATE_OK")):
        rc, out, lg = pod.run(fresh=True, env_extra={"P2REP_STOP_AFTER": stage})
        ok(rc == 0 and last in lg and f"stopping after {stage}" in lg, (stage, lg))
        ok(not any(r["tool"] == "train" for r in pod.calls()), stage)
    # ---- the stages that follow the verification: gate / launch gate refusals start NO trainer
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_P2GATE_FAIL": "1"}); ok(rc == 1, out); refused(out, lg, "gate", "the P2 gate did not PASS"); no_heavy_calls(("train", "launch_gate"))
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_GATE_REFUSE": "1"}); ok(rc == 1, out); refused(out, lg, "launch_gate", "did not mint a PASS token"); no_heavy_calls(("train",))
    ok("gate" in pod.marks() and "launch_gate" not in pod.marks(), pod.marks())
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_RECOVER_FAIL": "1"}); ok(rc == 1, out); refused(out, lg, "recover", "recover failed"); no_heavy_calls(("train", "launch_gate"))
    # a token that does not verify: NO preflight, NO trainer
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_VERIFY_FAIL": "1"}); ok(rc == 1, out); refused(out, lg, "launch_gate", "did not mint a PASS token that verifies"); no_heavy_calls(("train",))
    rc, out, lg = pod.run(fresh=True)
    ok(rc == 0 and "ZZP2REP_DONE" in lg, (out[-500:], lg))                                                              # a clean full run to leave a good token behind, then:
    pod.stub_log.unlink()
    ok("launch_gate" in pod.marks())
    for tk in (pod.R / "lg").glob("PASS_*.json"):
        tk.unlink()                                                                                                     # the token is gone and the stage believes it is done: token_ok must stop the chain
    rc, out, lg = pod.run(); ok(rc == 1, out); refused(out, lg, "launch_gate", "no longer verifies"); no_heavy_calls(("train",))
    # a preflight that fails: no trainer
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_PREFLIGHT_FAIL": "1"}); ok(rc == 1, out); refused(out, lg, "train", "PREFLIGHT_OK")
    ok([r["preflight"] for r in pod.calls("train")] == [True], pod.calls("train"))
    # ---- the happy path, in order
    snap_before = pod.snapshot()
    rc, out, lg = pod.run(fresh=True)
    ok(rc == 0, (out[-800:], lg[-800:]))
    order = ["ZZP2REP_START", "ZZP2REP_PRECHECK_OK", "ZZP2REP_WAIT_OK", "ZZP2REP_INPUTS_OK", "dd write test", "ZZP2REP_RECOVER", "ZZP2REP_GATE_OK", "ZZP2REP_LAUNCH_GATE_OK",
             "ZZP2REP_TRAIN_START", "attempt 1 exited rc 0", "ZZP2REP_TRAIN_DONE", "ZZP2REP_FINAL_OK", "ZZP2REP_DONE"]
    pos = [lg.find(m) for m in order]
    ok(all(p >= 0 for p in pos) and pos == sorted(pos), list(zip(order, pos)))
    ok(f"ZZP2REP_FINAL_OK model_final sha256 {hashlib.sha256(b'final-model-bytes').hexdigest()} remote=sha256-verified" in lg, lg[-400:])
    ok("REFUSED" not in lg and set(pod.marks()) == {"gate", "launch_gate"}, (pod.marks(), lg))
    tr = pod.calls("train")
    ok([r["preflight"] for r in tr] == [True, False], tr)                                                              # a preflight, then the run
    want_args = re.search(r'^ARGS="(.*)"$', pod.text, re.M).group(1).replace("$RUN", posix(pod.RUN)).split()
    for r in tr:
        base = [t for t in r["argv"] if t != "--preflight"]
        ok(base == want_args, (base, want_args))                                                                        # the trainer gets EXACTLY the pinned ARGS
        ok(r["seed"] == "1" and r["out"] == posix(pod.RUN) and "--seed" in r["argv"] and not r["hf_token"], r)          # --seed 1 reaches the trainer; the trainer never sees HF_TOKEN
    gt = pod.calls("launch_gate")
    ok([r["cmd"] for r in gt][:1] == ["run"] and all(r["profile"] == "refe_p2rep" for r in gt) and all(not r["hf_token"] for r in gt), gt)
    ok(sum(1 for r in gt if r["cmd"] == "verify") >= 3, "the token must be verified before the preflight AND before each launch")
    # every HF call is on THIS arm's repo; none on P2's; the token reaches only the egress children
    sy = pod.calls("p2r_sync")
    ok({r["cmd"] for r in sy} == {"watch", "recover", "final", "tick"}, sy)
    ok(all(r["repo"] == S1_REPO for r in sy) and not any(r["repo"] in (None, P2_REPO) for r in sy), [r["repo"] for r in sy])
    ok(all(r["hf_token"] for r in sy) and not any(r["hf_token"] for r in pod.calls() if r["tool"] != "p2r_sync"), "HF_TOKEN must reach only p2r_sync")
    rec = [r for r in sy if r["cmd"] == "recover"]
    ok(len(rec) == 1 and rec[0]["work"] == posix(pod.R / "recover_work") and rec[0]["run"] == posix(pod.RUN), rec)       # a scratch work dir: recover cannot reach refe_sft2/
    fin = [r for r in sy if r["cmd"] == "final"]
    ok(len(fin) == 1 and fin[0]["run"] == posix(pod.RUN), fin)
    ok(all(r["run"] == posix(pod.RUN) for r in sy if r["cmd"] in ("watch", "final", "tick", "recover")))
    gp = pod.calls("gate_prep")[0]
    ok(gp["out_sets"].startswith(posix(pod.R)) and gp["out_pdm"].startswith(posix(pod.R)), gp)                           # the gate's small banks live in the arm's dir
    ok(pod.calls("p2_gate")[0]["out"] == posix(pod.R / "gate"), pod.calls("p2_gate"))
    # NOTHING shared was written: P2's dir, the bank, the labels, the pixels, the weights, the init checkpoint, the DBs, RL-S1's and S1's dirs
    ok(pod.snapshot() == snap_before == snap0, [k for k in set(pod.snapshot()) ^ set(snap0)][:5])
    ok((pod.RUN / "model_final.pt").is_file() and not (pod.D / "refe_runs" / "p2_joint_pdm").exists(), "P2's run dir must not exist / change")
    ok((pod.R / "launch_gate_PASS_token.json").is_file())
    # ---- a re-run after success: finished stages are skipped, the gate is NOT run again, the final push is redone
    pod.stub_log.unlink()
    rc, out, lg = pod.run(); ok(rc == 0 and lg.count("ZZP2REP_DONE") == 2, (out[-400:], lg[-400:]))
    ok(not [r for r in pod.calls("launch_gate") if r["cmd"] == "run"] and not pod.calls("p2_gate") and not pod.calls("gate_prep"), "a re-run must skip the finished gates")
    ok([r["preflight"] for r in pod.calls("train")] == [True], "a finished run is not trained again")
    # ---- a trainer that dies is retried (token + GPU check each time); one that never finishes ends ZZP2REP_FAILED and never pushes a final
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_TRAIN_FAIL_FIRST": "2"}); ok(rc == 0 and "attempt 3 exited rc 0" in lg and "ZZP2REP_DONE" in lg, lg)
    ok(len([r for r in pod.calls("launch_gate") if r["cmd"] == "verify"]) >= 5, "each relaunch re-verifies the token")
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_TRAIN_FAIL_FIRST": "99"}); ok(rc == 1 and "ZZP2REP_FAILED" in lg and "attempt 6 exited" in lg and "ZZP2REP_FINAL_OK" not in lg, lg)
    ok(not [r for r in pod.calls("p2r_sync") if r["cmd"] == "final"])
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_FINAL_FAIL": "1"}); ok(rc == 1 and "ZZP2REP_FINAL_FAILED" in lg and "ZZP2REP_DONE" not in lg, lg)
    # the exit trap pushes a last tick for a REFUSED line too (to THIS arm's repo)
    rc, out, lg = pod.run(fresh=True, env_extra={"FAKE_GATE_REFUSE": "1"})
    ok([r["repo"] for r in pod.calls("p2r_sync") if r["cmd"] == "tick"] == [S1_REPO], pod.calls("p2r_sync"))


# ============================================================================================ 8. hygiene
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSH = re.compile("|".join([r"\bssh\s+(?:-\w+\s+)*\S+" + "@", r"\bssh\s+-p\b", "ro" + "ot@", "--po" + r"rt\b", r"\bs" + r"cp\s"]), re.I)


def hygiene_hits(text):
    out = []
    for ln in text.splitlines():
        if IPV4.search(ln.replace("0.0.0.0", "")) and not re.search(r"\b\d+\.\d+\.\d+\.\d+\s*(?:s|ms|GB|MB)\b", ln):
            out.append(("ipv4", ln.strip()[:80]))
        if SSH.search(ln):
            out.append(("ssh", ln.strip()[:80]))
        if re.search(r"hf_[A-Za-z0-9]{20,}", ln):
            out.append(("token", ln.strip()[:80]))
    return out


def hygiene():
    for p in (CHAIN, HELPER, Path(__file__)):
        ok(hygiene_hits(p.read_text(encoding="utf-8")) == [], (p.name, hygiene_hits(p.read_text(encoding="utf-8"))))
    ok(hygiene_hits("ss" + "h ro" + "ot@" + "10." + "1.2.3 -p 2222") and hygiene_hits("host " + "10." + "1.2.3") and hygiene_hits("hf_" + "a" * 30), "the detector must flag what it guards")
    if RUNBOOK.is_file():                                      # shipped to the pod? no: eval/ is the dev box's
        txt = RUNBOOK.read_text(encoding="utf-8")
        ok(hygiene_hits(txt) == [], hygiene_hits(txt))
        ok("p2rep_pod.sh" in txt and "ZZP2REP_DONE" in txt and "refe_p2rep" in txt and "--seed 1" in txt and S1_REPO in txt and "-c core.autocrlf=false" in txt, "the runbook lost a load-bearing fact")


def main() -> int:
    assert BASH, "bash is required"
    text = CHAIN.read_text(encoding="utf-8")
    rep_argv = arm_checks()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        reachability(rep_argv, td)
        bad = gate_checks_(rep_argv, td)
        n_mut = static_checks(text)
        (td / "b").mkdir()
        behaviour(text, td / "b", rep_argv)
        pod = Pod(td / "pod", text)
        helper_checks(pod)
        sandbox_runs(pod)
    hygiene()
    print(f"      ({len(bad)} gate mutations, {n_mut} chain mutations)")
    print(f"ZZTEST_PASS {N_OK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
