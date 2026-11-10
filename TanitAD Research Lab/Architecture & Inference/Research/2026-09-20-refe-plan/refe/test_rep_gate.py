#!/usr/bin/env python3
"""Seed-replicate launch-gate tests (CPU, no data).  Prints ZZTEST_PASS.

  R1 ARGV DIFF: C0r2 vs C0, S1E1 vs S1E, S1Z1 vs S1Z differ in the VALUE of --seed and of --out and in nothing else (lambda included: the parent's measured weight is carried over)
  R2 the legal argv of each replicate passes its profile; the parent's own argv, a replicate with the parent's seed, any other seed, a lane flag, a missing lambda, a calibrate flag: refused
  R3 FINGERPRINTS: refe_s1z_p6 (5c3bff457aa3bf1e) and refe_scorer1p6 (e74d64617f514d35) unmoved, the old profile key sets unchanged, the two new profiles stable (no function / address in them)
  R4 refusals: an unknown arm (rep_argv, the pod script, the chain); a replicate whose seed equals its parent's (the table, rep_argv, the gate, the pod script) -- each refused
  R5 the scripts: bash -n on every script; rep_pod.sh MODE=argv equals launch_gate.rep_argv AND the parent's pod script with --seed/--out swapped; the chain reuses the existing wait functions, has no
     calibrate stage, no human gate, no lane, pauses nothing; the pause script resumes P7 on every path after the stop
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json  # noqa: E402
import torch  # noqa: E402,F401
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

PC, PZ = L.PROFILES["refe_rep_p6"], L.PROFILES["refe_rep_s1z_p6"]
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256}
for _root in (L.SCORER1P6_ROOT, L.S1Z_ROOT):
    for _k, _f in (("--navtest-logs", _root + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", _root + "/navtest_ids/navtest_tokens.txt.gz")):
        SHA[_f] = L.SCORER1_NAVTEST_SHA256[_k]


def fake_sha(path):
    if path in SHA:
        return SHA[path]
    raise FileNotFoundError(path)


L.sha256_of_file = fake_sha
PROF_OF = {"C0r2": PC, "S1E1": PC, "S1Z1": PZ}
WP, WZ = 0.37, 0.41                                       # stand-ins for the parents' measured lambdas
W_OF = {"C0": None, "C0r2": None, "S1E": WP, "S1E1": WP, "S1Z": WZ, "S1Z1": WZ}
PARENT_ARGV = {"C0": lambda: L.scorer1p6_argv("C0"), "S1E": lambda: L.scorer1p6_argv("S1E", WP), "S1Z": lambda: L.s1z_argv("S1Z", twin_w=WZ)}


def refusals(argv, prof):
    return L.profile_refusals(prof, argv)


def has(rs, needle):
    return any(needle in r for r in rs)


def swap(argv, flag, *vals):
    out, i, done = [], 0, False
    while i < len(argv):
        if argv[i] == flag:
            out += [flag] + list(vals)
            done = True
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out if done else out + [flag] + list(vals)


def drop(argv, flag):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def pairs(toks):
    return sorted((f, tuple(v)) for f, v in GT.flag_pairs(list(toks)))


def differing_flags(a, b):
    da, db = {}, {}
    for f, v in GT.flag_pairs(list(a)):
        da.setdefault(f, []).append(tuple(v))
    for f, v in GT.flag_pairs(list(b)):
        db.setdefault(f, []).append(tuple(v))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


# ---- R1 the argv diff
PARENT_OF = {"C0r2": "C0", "S1E1": "S1E", "S1Z1": "S1Z"}
for arm, parent in PARENT_OF.items():
    ra, pa = L.rep_argv(arm, W_OF[arm]), PARENT_ARGV[parent]()
    assert len(ra) == len(pa), (arm, len(ra), len(pa))
    assert differing_flags(ra, pa) == {"--seed", "--out"}, (arm, differing_flags(ra, pa))
    d = [(i, x, y) for i, (x, y) in enumerate(zip(ra, pa)) if x != y]
    assert len(d) == 2 and {pa[i - 1] for i, _, _ in d} == {"--seed", "--out"}, (arm, d)               # exactly two tokens differ, by position: the VALUES of --seed and --out
    assert ra[ra.index("--seed") + 1] == str(L.REP_ARMS[arm]["seed"]) and pa[pa.index("--seed") + 1] == "0"
    assert ra[ra.index("--out") + 1] == L.REP_ROOT + "/" + arm.lower() and L.REP_ROOT != L.SCORER1P6_ROOT != L.S1Z_ROOT
    assert ra[:ra.index("--out")].count("--seed") == 1
    assert "--max-updates" in ra and ra[ra.index("--max-updates") + 1] == "400"                        # the same 400 updates as the parent
    assert ra[ra.index("--navtest-logs") + 1] == pa[pa.index("--navtest-logs") + 1] and ra[ra.index("--ckpt") + 1] == L.SCORER1P6_CKPT, "navtest lists and checkpoint are the parent's"
    # without the lambda: still the same two-value diff
    pin = L.scorer1p6_argv(parent) if parent in ("C0", "S1E") else L.s1z_argv(parent)               # the pin without lambda
    assert differing_flags(L.rep_argv(arm), pin) == {"--seed", "--out"}
print("R1 ok (the three replicates differ from their parents in --seed and --out only)")

# ---- R2 legal / refused argvs
for arm in L.REP_ARMS:
    A, P = L.rep_argv(arm, W_OF[arm]), PROF_OF[arm]
    assert refusals(A, P) == [], (arm, refusals(A, P))
    assert any(a in P["arm_seed"] for a in [arm]) and P["arm_seed"][arm] == L.REP_ARMS[arm]["seed"]
    par = PARENT_ARGV[PARENT_OF[arm]]()
    assert refusals(par, P), (arm, "the parent's own argv (parent out dir) must be refused by the replicate profile")
    same = swap(A, "--seed", "0")
    rs = refusals(same, P)
    assert has(rs, "--seed") and has(rs, "parent"), (arm, rs)                                      # the parent's seed on the replicate's out dir: refused, naming the parent
    assert has(refusals(swap(A, "--seed", "7"), P), "requires --seed"), arm
    assert has(refusals(A + ["--lane-head"], P), "--lane-head") and has(refusals(A + ["--pair-calibrate"], P), "pair-calibrate")
    assert refusals(swap(A, "--out", L.REP_ROOT + "/other"), P), arm
    SHA[L.SCORER1P6_CKPT] = "0" * 64
    assert has(refusals(A, P), "registered P6 checkpoint"), arm
    SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
    assert refusals(A, P) == []
assert has(refusals(drop(L.rep_argv("S1E1", WP), "--pair-w"), PC), "--pair-w") and has(refusals(drop(L.rep_argv("S1Z1", WZ), "--twin-w"), PZ), "--twin-w")
assert has(refusals(L.rep_argv("C0r2") + ["--pair-w", "0.3"], PC), "control"), "C0r2 is a control: no --pair-* flag"
assert refusals(drop(L.rep_argv("S1Z1", WZ), "--twin-topk"), PZ) and refusals(swap(L.rep_argv("S1Z1", WZ), "--twin-heads", "NC,DAC"), PZ)
assert refusals(swap(L.rep_argv("S1E1", WP), "--pair-heads", "NC,DAC,TTC"), PC), "S1E1 keeps S1E's heads"
assert refusals(L.rep_argv("C0r2", None) + ["--twin-w", "0.1"], PC), "a stray lever on the control is refused by the pin"
# the replicate arms are not arms of the parents' profiles (their out dirs are the parents' only)
assert refusals(L.rep_argv("S1E1", WP), L.PROFILES["refe_scorer1p6"]) and refusals(L.rep_argv("S1Z1", WZ), L.PROFILES["refe_s1z_p6"])
print("R2 ok")


# ---- R3 fingerprints
def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


assert fp("refe_s1z_p6").startswith("5c3bff457aa3bf1e"), fp("refe_s1z_p6")[:16]
assert fp("refe_scorer1p6").startswith("e74d64617f514d35"), fp("refe_scorer1p6")[:16]
for n in ("refe_rep_p6", "refe_rep_s1z_p6"):
    r = repr(sorted(L.PROFILES[n].items()))
    assert fp(n) == fp(n) and "function" not in r and "0x" not in r, n
assert fp("refe_rep_p6") != fp("refe_rep_s1z_p6") != fp("refe_s1z_p6") and fp("refe_rep_p6") != fp("refe_scorer1p6")
assert len(L.PROFILES) == 21 and len(set(L.PROFILES) - {"refe_rep_p6", "refe_rep_s1z_p6"}) == 19, len(L.PROFILES)
assert PC["suite"][-1] == {"script": "test_rep_gate.py", "token": "ZZTEST_PASS"} and PZ["suite"][-1] == PC["suite"][-1] and PC["suite"][:-1] == L.SCORER1P6_SUITE and PZ["suite"][:-1] == L.S1Z_SUITE
for k in ("data_pin", "data_expect", "ckpt_sha256", "navtest_sha256", "required_flags", "recipe", "forbidden_flags"):
    assert PC[k] == L.PROFILES["refe_scorer1p6"][k] and PZ[k] == L.PROFILES["refe_s1z_p6"][k], k      # data pin, checkpoint, navtest lists = the parents'
print("R3 ok (refe_s1z_p6 = %s..., refe_scorer1p6 = %s..., rep_p6 = %s..., rep_s1z_p6 = %s...)" % (fp("refe_s1z_p6")[:16], fp("refe_scorer1p6")[:16], fp("refe_rep_p6")[:16], fp("refe_rep_s1z_p6")[:16]))

# ---- R4 refusals
for bad in ("C0", "C0r", "S1E", "S1Z", "S1", "C0r3", "", "c0r2"):
    try:
        L.rep_argv(bad)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown arm %r was accepted" % bad)
assert L.rep_arm_refusals({k: dict(v) for k, v in L.REP_ARMS.items()}) == []
for arm in L.REP_ARMS:
    t = {arm: dict(L.REP_ARMS[arm], seed=0)}
    assert has(L.rep_arm_refusals(t), "equals its parent") and has(L.rep_arm_refusals(t), "floor of 0.0"), arm
assert has(L.rep_arm_refusals({"X": {"parent": "C0", "family": "p6", "seed": 1}, "C0r2": {"parent": "nope", "family": "p6", "seed": 2}}), "not an arm")
assert has(L.rep_arm_refusals({"C0r": {"parent": "C0", "family": "p6", "seed": 5}}), "registered parent arm")
assert has(L.rep_arm_refusals({"Q": {"parent": "C0", "family": "p6", "seed": "1"}}), "not an int")
saved = dict(L.REP_ARMS["S1E1"])
L.REP_ARMS["S1E1"]["seed"] = 0
try:
    L.rep_argv("S1E1", WP)
except ValueError as e:
    assert "equals its parent" in str(e)
else:
    raise AssertionError("rep_argv built a replicate with its parent's seed")
finally:
    L.REP_ARMS["S1E1"].update(saved)
assert L.rep_argv("S1E1", WP)[L.rep_argv("S1E1", WP).index("--seed") + 1] == "1"
print("R4 ok")

# ---- R5 the scripts
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod, chain, pause = (os.path.join(HERE, n) for n in ("rep_pod.sh", "rep_chain.sh", "rep_pause_p7.sh"))
chain_z, chain_old = os.path.join(HERE, "s1z_chain.sh"), os.path.join(HERE, "scorer1p6_chain.sh")
for f in (pod, chain, pause, chain_z, chain_old):
    assert os.path.exists(f), f
    assert b"\r" not in open(f, "rb").read(), "LF only: " + f
    assert os.path.basename(f).split(".")[0].lower() not in ("aux", "con", "prn", "nul"), f
ctext = open(chain, "rb").read().decode("utf-8")
joined = "\n".join(ln for ln in ctext.split("\n") if not ln.lstrip().startswith("#"))
assert "STAGE0_READING" not in joined and "P6_SPEED_READING" not in joined and "lane" not in joined.lower(), "no human gate, no lane"
assert "refe-plan-p6" not in joined and "refe-plan-p7" not in joined and "refe-plan-scorer1p6" not in joined and "refe-plan-s1z" not in joined, "own package dir only"
assert "$DATA/refe_rep" in joined and 'ROOT=$DATA/refe_rep' in joined and "--profile $PROFILE" in joined and "PROFILE=refe_rep_p6" in joined and "PROFILE=refe_rep_s1z_p6" in joined
assert "--pair-calibrate" not in joined and "MODE=calibrate" not in joined and "calibrate_s1" in joined, "no calibrate stage; only the parent's lambda is READ"
assert '--prefix "rep/$LA"' in joined and "ZZREP_CHAIN_DONE" in joined and "adb5aafd4fb8abdfd1a53d9cc5ff865a89bd20f9008c701cf5d308dac595abed" in ctext and "8c7b6b697b94ccf1a4781a727ed28c10ed8b11878c33eeb2d016a507b36efa69" in ctext
assert not re.search(r"\bkill -|SIGSTOP|SIGCONT|pkill|taskkill|ZZP7|P7_PAUSE|pause_p7|resume_p7|lane_pause|lane_restart|refe_p7/", joined), "the chain neither pauses nor resumes P7 (it only READS the P7 chain lock)"
assert "--max-updates" not in joined and "--seed" not in joined, "the chain builds no argv of its own: rep_pod.sh does"
assert L.S1Z_CARD_SHA256 in ctext and L.SCORER1P6_CARD_SHA256 in ctext and L.SCORER1P6_CKPT_SHA256 in ctext
for g in ("nobalance epbalance droploss", "noradius notopk droploss", "GATEK=G16", "GATEK=G18"):
    assert g in joined, g
assert "gate_is $OUT/mutate_$MUT.log $GATEK ok false" in joined


def fn_body(text, name):
    m1 = re.search(r"^%s\(\) \{.*$" % re.escape(name), text, re.M)
    if m1 and "; }" in m1.group(0):
        return m1.group(0)
    m = re.search(r"^%s\(\) \{.*?^\}" % re.escape(name), text, re.S | re.M)
    assert m, name
    return m.group(0)


zt, old_text = open(chain_z, "rb").read().decode("utf-8"), open(chain_old, "rb").read().decode("utf-8")
norm = lambda t: t.replace("S1Z_", "REP_")          # noqa: E731
for name in ("gpu_busy", "lock_held", "wait_clear"):
    assert norm(fn_body(zt, name)) == fn_body(ctext, name), name                           # the wait for no chain lock and an idle GPU is the existing function, byte for byte
assert norm(fn_body(zt, "others_hold")).replace("refe_p7 refe_scorer1p6; do", "refe_p7 refe_scorer1p6 refe_s1z; do") == fn_body(ctext, "others_hold"), "others_hold differs by exactly the one added lock"
for name in ("gate_is", "stage_done", "stage_mark", "tree_sha"):
    assert fn_body(zt, name).replace("S1Z_", "REP_") == fn_body(ctext, name) or name == "gate_is" and fn_body(old_text, name) == fn_body(ctext, name), name
ptext = open(pause, "rb").read().decode("utf-8")
assert "trap 'resume_p7' EXIT" in ptext and "HOLD_P7" in ptext and "rep_chain.sh" in ptext and 'for A in $ARMS' in ptext and "P7_RESUMED" in ptext and "STOPPED=1" in ptext
assert ptext.index("STOPPED=1") < ptext.index("kill -TERM $CHAIN") < ptext.index("for A in $ARMS") < ptext.rindex("resume_p7"), "P7 is marked stopped before it is killed; the arms run between; the resume closes the script"
assert ptext.count("exit 1") >= 3 and "PAUSE_STOPPED_EARLY" in ptext

if bash:
    for sh in (pod, chain, pause):
        r = subprocess.run([bash, "-n", sh], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (sh, r.stderr[:300])

    def run_pod(arm, ws, script=pod, mode="argv", extra=None):
        env = dict(os.environ, ARM=arm, MODE=mode, REP_WS=ws)
        env.update(extra or {})
        return subprocess.run([bash, script], env=env, capture_output=True, encoding="utf-8")

    # no calibrate file: the pin without lambda, exactly rep_argv(arm).  The workspace is an EMPTY temp dir, never /workspace: on the pod /workspace holds the
    # parents' real calibrate files, so this case read a lambda there and failed G-SUITE (MEASURED 2026-11-10 04:58Z, S1E1 --pair-w 0.10810585693683655).
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td0:
        ws0 = td0.replace(chr(92), "/")
        for arm in L.REP_ARMS:
            r = run_pod(arm, ws0)
            assert r.returncode == 0, (arm, r.stderr[:300])
            out = [t.replace(ws0, "/workspace") for t in r.stdout.strip().split()]
            assert pairs(out) == pairs(L.rep_argv(arm)), (arm, set(pairs(out)) ^ set(pairs(L.rep_argv(arm))))
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
        wsx = td.replace(chr(92), "/")
        for rel, key, val in (("data/refe_scorer1p6/calibrate_s1e", "pair_w", WP), ("data/refe_s1z/calibrate_s1z", "twin_w", WZ)):
            os.makedirs(os.path.join(td, rel))
            json.dump({key: val}, open(os.path.join(td, rel, "pair_weight.json"), "w"))
        for arm, parent in PARENT_OF.items():
            r = run_pod(arm, wsx)
            assert r.returncode == 0, (arm, r.stderr[:300])
            out = [t.replace(wsx, "/workspace") for t in r.stdout.strip().split()]
            assert pairs(out) == pairs(L.rep_argv(arm, W_OF[arm])), (arm, set(pairs(out)) ^ set(pairs(L.rep_argv(arm, W_OF[arm]))))
            ppod = os.path.join(HERE, "s1z_pod.sh" if parent == "S1Z" else "scorer1p6_pod.sh")
            pr = subprocess.run([bash, ppod], env=dict(os.environ, ARM=parent, MODE="argv", SCORER1P6_WS=wsx, S1Z_WS=wsx), capture_output=True, encoding="utf-8")
            assert pr.returncode == 0, pr.stderr[:300]
            pout = [t.replace(wsx, "/workspace") for t in pr.stdout.strip().split()]
            assert differing_flags(out, pout) == {"--seed", "--out"}, (arm, differing_flags(out, pout))      # the pod scripts' own output: the seed and the out dir only
            if W_OF[arm] is not None:
                assert out[out.index("--pair-w" if parent == "S1E" else "--twin-w") + 1] == str(W_OF[arm])
        # the pair / twin replicate of an UNCALIBRATED parent is refused (rc 3) in every mode but argv
        with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td2:
            r = run_pod("S1E1", td2.replace(chr(92), "/"), mode="preflight")
            assert r.returncode == 3 and "pair-w" in r.stderr, (r.returncode, r.stderr[:200])
            r = run_pod("S1Z1", td2.replace(chr(92), "/"), mode="run")
            assert r.returncode == 3 and "twin-w" in r.stderr, (r.returncode, r.stderr[:200])
        # RED arm: a copy of the pod script whose S1E1 seed equals S1E's is refused
        os.makedirs(os.path.join(td, "red"))
        for n in ("rep_pod.sh", "scorer1p6_pod.sh", "s1z_pod.sh"):
            shutil.copyfile(os.path.join(HERE, n), os.path.join(td, "red", n))
        txt = open(os.path.join(td, "red", "rep_pod.sh"), encoding="utf-8", newline="").read()
        assert "S1E1) PARENT=S1E; SEED=1;" in txt
        open(os.path.join(td, "red", "rep_pod.sh"), "w", encoding="utf-8", newline="").write(txt.replace("S1E1) PARENT=S1E; SEED=1;", "S1E1) PARENT=S1E; SEED=0;"))
        r = run_pod("S1E1", wsx, script=os.path.join(td, "red", "rep_pod.sh"))
        assert r.returncode == 2 and "equals its parent" in r.stderr, (r.returncode, r.stderr[:200])
        r = run_pod("S1E1", wsx, script=os.path.join(td, "red", "rep_pod.sh"), mode="run")
        assert r.returncode == 2
    for bad in ("C0", "C0r", "S1E", "S1Z", "", "c0r2"):
        r = run_pod(bad, "/workspace")
        assert r.returncode == 2 and "unknown ARM" in r.stderr, (bad, r.returncode, r.stderr[:200])
        r = subprocess.run([bash, chain] + ([bad] if bad else []), env=dict(os.environ, REP_WS="/nonexistent_ws"), capture_output=True, encoding="utf-8")
        assert r.returncode == 2 and "REFUSED" in r.stdout and "unknown arm" in r.stdout, (bad, r.returncode, r.stdout[:200])
    # a mutation the arm's parent does not have is refused (a calibrated sandbox, so the lambda check passes first)
    td_ok = tempfile.mkdtemp(dir=os.environ.get("SCORER1_TMP") or None).replace(chr(92), "/")
    for rel, key, val in (("data/refe_scorer1p6/calibrate_s1e", "pair_w", WP), ("data/refe_s1z/calibrate_s1z", "twin_w", WZ)):
        os.makedirs(os.path.join(td_ok, rel))
        json.dump({key: val}, open(os.path.join(td_ok, rel, "pair_weight.json"), "w"))
    for arm_, mode_ in (("C0r2", "mutate_droploss"), ("S1E1", "mutate_noradius"), ("S1Z1", "mutate_nobalance")):
        r = run_pod(arm_, "/workspace", mode=mode_, extra={"REP_WS": td_ok})
        assert r.returncode == 2 and "has no mutation" in r.stderr, (arm_, mode_, r.returncode, r.stderr[:200])
    print("R5 ok")
else:
    print("R5 SKIPPED (no bash)")
print("ZZTEST_PASS")
