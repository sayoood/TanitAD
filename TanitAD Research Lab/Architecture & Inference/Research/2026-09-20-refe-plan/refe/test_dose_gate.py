#!/usr/bin/env python3
"""Dose launch-gate tests (CPU, no data).  Prints ZZTEST_PASS.  PREREG_DOSE_P6 + AMENDMENT A1 (PREREG_DOSE_P6_A1.md, sha256 1aa0e702...): S1ZD = S1Z with exactly THREE values moved, --max-updates 400 -> 1600, --out, and --epochs 2.0 (right after the --max-updates value).  Tests the SHIPPED A1 state.

  D1 ARGV DIFF: S1ZD vs S1Z differ in the VALUE of --max-updates and of --out and by the added `--epochs 2.0` pair and in nothing else (by position; lambda_z carried over; with and without lambda), in launch_gate.dose_argv
     AND in the pod script's own output vs s1z_pod.sh's.  s1z_pod.sh says '--max-updates 400' twice (a comment on line 4 and the run argv COMMON): the dose touches the COMMON one only.
  D2 legal / refused argvs.  THE CARD'S ORIGINAL TWO-VALUE ARGV (no --epochs) IS REFUSED (reach rule: 60,000 sets / 8 / 8 x epochs 1.0 = 938 updates < 1600, and the missing required --epochs 2.0); the A1 argv passes the real profile.  Wrong update counts, a missing /
     non-positive lambda, --pair-calibrate (a re-measured lambda), the parent's own argv, a wrong seed, a lane flag, a wrong checkpoint: refused.
  D3 FINGERPRINTS: the 23 existing profiles unmoved, the new profile stable, 24 profiles.
  D4 table refusals; D5 the scripts (bash -n, pod script vs launch_gate, no calibrate mode, S1Z's lambda only, chain structure, pause script) and MUTATIONS (400 instead of 1600) that go RED.
"""
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402,F401
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

PD = L.PROFILES["refe_dose_s1z_p6"]
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256}
for _root in (L.SCORER1P6_ROOT, L.S1Z_ROOT, L.DOSE_ROOT):
    for _k, _f in (("--navtest-logs", _root + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", _root + "/navtest_ids/navtest_tokens.txt.gz")):
        SHA[_f] = L.SCORER1_NAVTEST_SHA256[_k]


def fake_sha(path):
    if path in SHA:
        return SHA[path]
    raise FileNotFoundError(path)


L.sha256_of_file = fake_sha
WZ = 0.41                                                 # stand-in for S1Z's measured lambda_z


def refusals(argv, prof=PD):
    return L.profile_refusals(prof, argv)


def has(rs, needle):
    return any(needle in r for r in rs)


def swap(argv, flag, *vals):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            out += [flag] + list(vals)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


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


def strip_epochs(argv):
    """argv without the `--epochs <v>` pair (the card's original two-value form)."""
    return drop(list(argv), "--epochs")


def two_token_diff(mod, arm="S1ZD", parent="S1Z", w=WZ):
    """A1 check (name kept): True iff mod.dose_argv(arm, w) differs from mod.s1z_argv(parent, twin_w=w) in exactly THREE flags: the VALUES of --max-updates (1600) and --out, and the added `--epochs 2.0`
    pair right after the --max-updates value; with that pair removed exactly two tokens differ.  (The check the mutations below must turn RED.)"""
    da, pa = mod.dose_argv(arm, w), mod.s1z_argv(parent, twin_w=w)
    if differing_flags(da, pa) != {"--max-updates", "--out", "--epochs"} or len(da) != len(pa) + 2:
        return False
    i = da.index("--max-updates") if "--max-updates" in da else -1
    if i < 0 or da[i + 2:i + 4] != ["--epochs", "2.0"] or da.count("--epochs") != 1 or "--epochs" in pa:
        return False
    core = strip_epochs(da)
    if len(core) != len(pa):
        return False
    d = [(j, x, y) for j, (x, y) in enumerate(zip(core, pa)) if x != y]
    return len(d) == 2 and {pa[j - 1] for j, _, _ in d} == {"--max-updates", "--out"} and da[i + 1] == "1600"


# ---- D1 the argv diff (A1: three flags)
assert two_token_diff(L), "dose_argv differs from s1z_argv in other than the three A1 values"
DA, PA = L.dose_argv("S1ZD", WZ), L.s1z_argv("S1Z", twin_w=WZ)
assert DA[DA.index("--max-updates") + 1] == "1600" and PA[PA.index("--max-updates") + 1] == "400" and DA[DA.index("--out") + 1] == L.DOSE_ROOT + "/s1zd" and PA[PA.index("--out") + 1] == L.S1Z_ROOT + "/s1z"
assert DA[DA.index("--seed") + 1] == "0" == PA[PA.index("--seed") + 1], "seed 0 (S1Z's)"
assert DA[DA.index("--twin-w") + 1] == str(WZ), "lambda_z is carried over from the parent"
assert DA[DA.index("--ckpt") + 1] == L.SCORER1P6_CKPT and DA[DA.index("--navtest-logs") + 1] == PA[PA.index("--navtest-logs") + 1] and DA.count("--max-updates") == 1 == DA.count("--out")
assert differing_flags(L.dose_argv("S1ZD"), L.s1z_argv("S1Z")) == {"--max-updates", "--out", "--epochs"}, "without lambda: the same three-flag diff"
assert L.DOSE_EPOCHS == 2.0 and DA[DA.index("--max-updates") + 2:DA.index("--max-updates") + 4] == ["--epochs", "2.0"] and DA.count("--epochs") == 1 and "--epochs" not in PA
assert L.DOSE_ROOT not in (L.S1Z_ROOT, L.SCORER1P6_ROOT, L.REP_ROOT) and L.DOSE_PKG not in (L.S1Z_PKG, L.S1ZSM_PKG, L.REP_PKG)
print("D1 ok (dose_argv, A1: exactly three flags differ from S1Z's: --max-updates 1600, --out, --epochs 2.0 right after the --max-updates value)")

# ---- D2 legal / refused, against the REAL shipped profile
REACH = L.dose_reach_refusals(1600, 1.0)
assert REACH and has(REACH, "cannot reach --max-updates 1600") and has(REACH, "update 938") and has(REACH, "A1"), REACH
assert L.dose_reach_total(60000, 1.0, 1600) == 938 and L.dose_reach_total(60000, 2.0, 1600) == 1600 and L.dose_reach_total(60000, 1.7, 1600) == 1594 and L.dose_reach_total(60000, 1.71, 1600) == 1600
assert L.dose_reach_total(60000, 1.0, 400) == 400, "the parent's own 400 updates are reached"
assert L.dose_reach_refusals(1600, 2.0) == [] and L.dose_reach_refusals(1600, 1.7) and L.dose_reach_refusals(1600, 1.0, train_sets=200000) == []
assert PD["recipe"]["epochs"] == 2.0 and PD["required_flags"]["--epochs"] == "2.0" and PD["recipe"]["max_updates"] == 1600
assert refusals(DA) == [], refusals(DA)                                   # THE A1 ARGV IS LEGAL ON THE REAL PROFILE
# THE CARD'S ORIGINAL TWO-VALUE ARGV (no --epochs) IS REFUSED: which refusals fire (printed; both asserted)
CARD = strip_epochs(DA)
assert differing_flags(CARD, PA) == {"--max-updates", "--out"}, "CARD is S1Z's argv with the two card values"
RC = refusals(CARD)
print("   card-original argv refusals:", RC)
assert RC and has(RC, "--epochs") and has(RC, "cannot reach --max-updates 1600"), RC                  # (1) the required --epochs 2.0 is absent, (2) the reach rule: parser default 1.0 ends at 938
assert has(RC, "update 938")
for mu in ("400", "800", "1599", "3200"):
    assert has(refusals(swap(DA, "--max-updates", mu)), "--max-updates"), mu                # a wrong update count is refused (the pin and the required flag)
for ep in ("1.0", "1", "1.5", "3.0"):
    assert has(refusals(swap(DA, "--epochs", ep)), "--epochs"), ep                          # a wrong epochs is refused (required flag); 1.0 / 1.5 also fail the reach rule
assert has(refusals(swap(DA, "--epochs", "1.5")), "cannot reach")
assert has(refusals(drop(DA, "--twin-w")), "--twin-w"), "a missing lambda is refused"
assert has(refusals(swap(DA, "--twin-w", "0")), "--twin-w") and has(refusals(swap(DA, "--twin-w", "nan")), "--twin-w") and has(refusals(swap(DA, "--twin-w", "-0.1")), "--twin-w")
assert has(refusals(DA + ["--pair-calibrate"]), "pair-calibrate"), "a re-measured lambda (the calibration mode) is refused"
assert has(refusals(swap(DA, "--seed", "1")), "requires --seed") and has(refusals(DA + ["--lane-head"]), "--lane-head")
assert refusals(swap(DA, "--out", L.DOSE_ROOT + "/other")) and refusals(swap(DA, "--out", L.S1Z_ROOT + "/s1z"))
assert has(refusals(drop(DA, "--twin-topk")), "--twin-topk") and refusals(swap(DA, "--twin-heads", "NC,DAC"))
SHA[L.SCORER1P6_CKPT] = "0" * 64
assert has(refusals(DA), "registered P6 checkpoint")
SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
assert refusals(DA) == []
SZ = L.s1z_argv("S1Z", twin_w=WZ)
assert refusals(SZ) and refusals(SZ, L.PROFILES["refe_s1z_p6"]) == [], "S1Z's own argv is S1Z's, not the dose"
assert refusals(DA, L.PROFILES["refe_s1z_p6"]) and refusals(CARD, L.PROFILES["refe_s1z_p6"]), "the dose argvs are not arms of refe_s1z_p6"
assert has(refusals(swap(DA, "--max-updates", "400")), "--max-updates 1600"), "the dose profile refuses the parent's 400"
print("D2 ok (the A1 argv is legal on the real profile; the card's original two-value argv is refused by the missing required --epochs AND the reach rule; every wrong case reads as required)")

# ---- D3 fingerprints
def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


TIP_FP = {"refe_p2": "627341286aea3506", "refe_p6": "80fa4d7ec757ad46", "refe_field": "5ddc49053a99fa78", "refe_field_ctrl": "0b6281b1bfa996ed", "refe_rehearsal": "821cd1ef3bf55d4e",
          "refe_rls1": "bce7e25ee8718c08", "refe_aux_s1": "f95f48727b62ccc6", "refe_p7": "94cca167ce121a94", "refe_p2rep": "44ca107f53aceee8", "refe_sft4r": "8972e3c348206b94",
          "refe_scorer1": "88d02fb7b34d4fd8", "refe_scorer1p6": "e74d64617f514d35", "refe_s1z_p6": "5c3bff457aa3bf1e", "refe_rep_p6": "dd348f44c9638f1a", "refe_rep_s1z_p6": "2356afd4efeeb52a",
          "refe_rls1b": "5d6bfbc10152ae96", "refe_rls1c": "26214cf954099d51", "refe_rls1d": "fcc2585f2a5d9b94", "refe_rls1e": "54d96289b19c3cd6", "refe_hnscreen": "d30126bbb5a32957",
          "refe_aux_s1c": "dcdf2dbd2a674f1c", "refe_s1zsm_p6": "86380fbb786daa75", "refe_rep_s1zsm_p6": "27fe236a2fd68ca4"}          # MEASURED at the tip e5437f6c (the 23 profiles that exist before the dose)
moved = [n for n, want in TIP_FP.items() if not fp(n).startswith(want)]
assert not moved, f"existing profile fingerprints moved: {moved}"
assert set(L.PROFILES) - set(TIP_FP) == {"refe_dose_s1z_p6", "refe_div_p6"} and len(L.PROFILES) == 25          # + refe_div_p6 (PREREG_DIV_P6)
r = repr(sorted(PD.items()))
assert fp("refe_dose_s1z_p6") == fp("refe_dose_s1z_p6") and "function" not in r and "0x" not in r, "the new digest must be stable"
assert fp("refe_dose_s1z_p6") != fp("refe_s1z_p6")
base = L.PROFILES["refe_s1z_p6"]
assert PD is not base and PD["recipe"] is not base["recipe"] and PD["required_flags"] is not base["required_flags"] and PD["arm_seed"] is not base["arm_seed"]
for k in ("data_pin", "data_expect", "ckpt_sha256", "navtest_sha256", "forbidden_flags", "twin_spec"):
    assert PD[k] == base[k], k
assert PD["recipe"]["max_updates"] == 1600 and PD["recipe"]["epochs"] == 2.0 and base["recipe"]["max_updates"] == 400 and PD["data_expect_updates"] == 1600 and base.get("data_expect_updates") == 400
assert PD["required_flags"]["--max-updates"] == "1600" and base["required_flags"]["--max-updates"] == "400" and PD["arm_seed"] == {"S1ZD": 0}
assert {k: v for k, v in PD["recipe"].items() if k not in ("max_updates", "epochs")} == {k: v for k, v in base["recipe"].items() if k not in ("max_updates", "epochs")}, "every other recipe value is S1Z's"
assert PD["required_flags"] == {**base["required_flags"], "--max-updates": "1600", "--epochs": "2.0"}, "required flags: S1Z's plus the A1 values"
assert PD["suite"][:-1] == L.S1Z_SUITE and PD["suite"][-1] == {"script": "test_dose_gate.py", "token": "ZZTEST_PASS"} and L.S1Z_SUITE == base["suite"]
print("D3 ok (the 23 existing profile digests unmoved; NEW refe_dose_s1z_p6 = %s...)" % fp("refe_dose_s1z_p6")[:16])

# ---- D4 table refusals
assert L.dose_arm_refusals({k: dict(v) for k, v in L.DOSE_ARMS.items()}) == []
for bad in ("S1Z", "S1Z1", "S1ZD1", "C0", "", "s1zd", "S1ZSM1"):
    try:
        L.dose_argv(bad)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown arm %r was accepted" % bad)
for mu in (400, 399, 0, "1600", True, None):
    assert has(L.dose_arm_refusals({"S1ZD": {"parent": "S1Z", "family": "s1z", "max_updates": mu}}), "not an int above"), mu
assert has(L.dose_arm_refusals({"S1ZD": {"parent": "C0", "family": "s1z", "max_updates": 1600}}), "not the S1Z arm")
for nm in ("S1Z", "S1Z1", "S1ZSM1"):
    assert has(L.dose_arm_refusals({nm: {"parent": "S1Z", "family": "s1z", "max_updates": 1600}}), "registered arm"), nm
saved = dict(L.DOSE_ARMS["S1ZD"])
L.DOSE_ARMS["S1ZD"]["max_updates"] = 400
try:
    L.dose_argv("S1ZD", WZ)
except ValueError as e:
    assert "not an int above the parent's 400" in str(e)
else:
    raise AssertionError("dose_argv built a dose with its parent's 400 updates")
finally:
    L.DOSE_ARMS["S1ZD"].update(saved)
assert L.dose_argv("S1ZD", WZ)[L.dose_argv("S1ZD", WZ).index("--max-updates") + 1] == "1600"
print("D4 ok")

# ---- D5 the scripts and the mutations
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod, chain, pause = (os.path.join(HERE, n) for n in ("dose_pod.sh", "dose_chain.sh", "dose_pause_p7.sh"))
chain_rep, spod = os.path.join(HERE, "rep_chain.sh"), os.path.join(HERE, "s1z_pod.sh")
for f in (pod, chain, pause, chain_rep, spod):
    assert os.path.exists(f), f
    assert b"\r" not in open(f, "rb").read(), "LF only: " + f
    assert os.path.basename(f).split(".")[0].lower() not in ("aux", "con", "prn", "nul"), f
stext = open(spod, encoding="utf-8").read()
occ = [ln for ln in stext.split("\n") if "--max-updates 400" in ln]
assert len(occ) == 2 and sum(ln.lstrip().startswith("#") for ln in occ) == 1 and sum(ln.startswith("COMMON=") for ln in occ) == 1, occ          # one comment (line 4), one run argv (COMMON)
ptxt = open(pod, encoding="utf-8").read()
assert re.search(r'^DOSE_UPDATES=1600$', ptxt, re.M) and re.search(r'^DOSE_EPOCHS="2.0"$', ptxt, re.M) and L.DOSE_ARMS["S1ZD"]["max_updates"] == 1600 and L.DOSE_EPOCHS == 2.0 and str(L.DOSE_EPOCHS) == "2.0", "pod script and launch_gate agree on the dose (A1: DOSE_EPOCHS 2.0 in both)"
ctext = open(chain, "rb").read().decode("utf-8")
joined = "\n".join(ln for ln in ctext.split("\n") if not ln.lstrip().startswith("#"))
assert "PROFILE=refe_dose_s1z_p6" in joined and "--profile $PROFILE" in joined and "ROOT=$DATA/refe_dose" in joined and 'PKG=${DOSE_PKG:-$WS/refe-plan-dose}' in joined
assert L.S1Z_CARD_SHA256 in ctext and "037c127235b51c374780f31ad2e93855ac7c141a4051f89498f40f1beff52ebb" in ctext and L.SCORER1P6_CKPT_SHA256 in ctext
assert 'MUTS="noradius notopk droploss"; GATEK=G18' in joined and "gate_is $OUT/mutate_$MUT.log $GATEK ok false" in joined and 'updates") != 1600' in joined and "ZZDOSE_CHAIN_DONE" in joined
assert '--prefix "dose/$LA"' in joined and "arm_dose_$LA" in joined and "refe-plan-rep" not in joined and "refe-plan-s1z" not in joined and "lane" not in joined.lower()
assert "--max-updates" not in joined and "--seed" not in joined and "MODE=calibrate" not in joined and "--pair-calibrate" not in joined, "the chain builds no argv and has no calibrate stage"
assert not re.search(r"\bkill -|SIGSTOP|SIGCONT|pkill|taskkill|ZZP7|P7_PAUSE|pause_p7|resume_p7|lane_pause|lane_restart|refe_p7/", joined), "the chain neither pauses nor resumes P7"
assert "calibrate_s1z/pair_weight.json" in joined and 'd["micro_batches"]==50' in joined and "not re-measured" in joined


def fn_body(text, name):
    m1 = re.search(r"^%s\(\) \{.*$" % re.escape(name), text, re.M)
    if m1 and "; }" in m1.group(0):
        return m1.group(0)
    m = re.search(r"^%s\(\) \{.*?^\}" % re.escape(name), text, re.S | re.M)
    assert m, name
    return m.group(0)


rt = open(chain_rep, "rb").read().decode("utf-8")
nr = lambda t: t.replace("REP_", "DOSE_")          # noqa: E731
for name in ("gpu_busy", "lock_held", "wait_clear", "gate_is", "stage_done", "stage_mark", "tree_sha", "refuse"):
    assert nr(fn_body(rt, name)) == fn_body(ctext, name), name                       # the wait / marker functions are rep_chain's, byte for byte (REP_ -> DOSE_ only)
assert nr(fn_body(rt, "others_hold")).replace("refe_s1zsm; do", "refe_s1zsm refe_rep; do") == fn_body(ctext, "others_hold"), "others_hold differs by exactly the one added lock (refe_rep)"
ps = open(pause, "rb").read().decode("utf-8")
assert "trap 'resume_p7' EXIT" in ps and "HOLD_P7" in ps and "dose_chain.sh" in ps and "STOPPED=1" in ps and "P7_RESUMED" in ps and "timeout 25200" in ps and "refe_dose" in ps and "S1ZD) ;;" in ps
assert ps.index("STOPPED=1") < ps.index("kill -TERM $CHAIN") < ps.index("for A in $ARMS") < ps.rindex("resume_p7") and ps.count("exit 1") >= 3 and "PAUSE_STOPPED_EARLY" in ps

if bash:
    for sh in (pod, chain, pause):
        r = subprocess.run([bash, "-n", sh], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (sh, r.stderr[:300])

    def run_pod(arm, ws, script=pod, mode="argv", extra=None):
        env = dict(os.environ, ARM=arm, MODE=mode, DOSE_WS=ws, S1Z_WS=ws)
        env.update(extra or {})
        return subprocess.run([bash, script], env=env, capture_output=True, encoding="utf-8")

    tmp = os.environ.get("SCORER1_TMP") or None
    with tempfile.TemporaryDirectory(dir=tmp) as td0:
        ws0 = td0.replace(chr(92), "/")
        r = run_pod("S1ZD", ws0)                                  # S1Z not calibrated: the pin without lambda
        assert r.returncode == 0, r.stderr[:300]
        out = [t.replace(ws0, "/workspace") for t in r.stdout.strip().split()]
        assert pairs(out) == pairs(L.dose_argv("S1ZD")), set(pairs(out)) ^ set(pairs(L.dose_argv("S1ZD")))
        for mode in ("preflight", "run", "mutate_noradius", "mutate_notopk", "mutate_droploss"):
            r = run_pod("S1ZD", ws0, mode=mode)
            assert r.returncode == 3 and "twin-w" in r.stderr and "never re-measures" in r.stderr, (mode, r.returncode, r.stderr[:200])         # an uncalibrated S1Z: no dose
    with tempfile.TemporaryDirectory(dir=tmp) as td:
        wsx = td.replace(chr(92), "/")
        os.makedirs(os.path.join(td, "data/refe_s1z/calibrate_s1z"))
        json.dump({"twin_w": WZ}, open(os.path.join(td, "data/refe_s1z/calibrate_s1z/pair_weight.json"), "w"))
        # a pair_weight.json in the dose's OWN dirs (a re-measured lambda) is never read: the argv carries S1Z's value
        os.makedirs(os.path.join(td, "data/refe_dose/calibrate_s1zd"))
        json.dump({"twin_w": 9.99}, open(os.path.join(td, "data/refe_dose/calibrate_s1zd/pair_weight.json"), "w"))
        r = run_pod("S1ZD", wsx)
        assert r.returncode == 0, r.stderr[:300]
        out = [t.replace(wsx, "/workspace") for t in r.stdout.strip().split()]
        assert pairs(out) == pairs(L.dose_argv("S1ZD", WZ)) and "9.99" not in r.stdout, set(pairs(out)) ^ set(pairs(L.dose_argv("S1ZD", WZ)))
        pr = subprocess.run([bash, spod], env=dict(os.environ, ARM="S1Z", MODE="argv", S1Z_WS=wsx), capture_output=True, encoding="utf-8")
        assert pr.returncode == 0, pr.stderr[:300]
        pout = [t.replace(wsx, "/workspace") for t in pr.stdout.strip().split()]
        assert differing_flags(out, pout) == {"--max-updates", "--out", "--epochs"} and len(out) == len(pout) + 2, differing_flags(out, pout)           # the pod scripts' own output: the three A1 values only
        assert out[out.index("--max-updates") + 2:out.index("--max-updates") + 4] == ["--epochs", "2.0"] and out.count("--epochs") == 1
        d = [(i, x, y) for i, (x, y) in enumerate(zip(strip_epochs(out), pout)) if x != y]
        assert len(d) == 2 and out[out.index("--max-updates") + 1] == "1600" and pout[pout.index("--max-updates") + 1] == "400" and out[out.index("--twin-w") + 1] == pout[pout.index("--twin-w") + 1]
        assert " ".join(out).count("--max-updates") == 1 and "--max-updates 400" not in " ".join(out)
        # no calibrate mode (a dose never re-measures), a mutation S1Z does not have, an unknown arm
        r = run_pod("S1ZD", wsx, mode="calibrate")
        assert r.returncode == 2 and "no calibrate mode" in r.stderr, (r.returncode, r.stderr[:200])
        for m in ("mutate_nobalance", "mutate_noband", "mutate_epbalance"):
            r = run_pod("S1ZD", wsx, mode=m)
            assert r.returncode == 2 and "has no mutation" in r.stderr, (m, r.returncode, r.stderr[:200])
        for bad in ("S1Z", "S1Z1", "S1ZSM1", "", "s1zd", "C0"):
            r = run_pod(bad, "/workspace")
            assert r.returncode == 2 and "unknown ARM" in r.stderr, (bad, r.returncode, r.stderr[:200])
            r = subprocess.run([bash, chain] + ([bad] if bad else []), env=dict(os.environ, DOSE_WS="/nonexistent_ws"), capture_output=True, encoding="utf-8")
            assert r.returncode == 2 and "REFUSED" in r.stdout and "unknown arm" in r.stdout, (bad, r.returncode, r.stdout[:200])
        # ---- MUTATIONS: each must turn a check RED
        os.makedirs(os.path.join(td, "red"))
        for n in ("dose_pod.sh", "s1z_pod.sh"):
            shutil.copyfile(os.path.join(HERE, n), os.path.join(td, "red", n))
        rp = os.path.join(td, "red", "dose_pod.sh")
        txt = open(rp, encoding="utf-8", newline="").read()
        for new in ("DOSE_UPDATES=400", "DOSE_UPDATES=300"):
            open(rp, "w", encoding="utf-8", newline="").write(txt.replace("DOSE_UPDATES=1600", new))
            r = run_pod("S1ZD", wsx, script=rp)
            assert r.returncode == 2 and "not above its parent" in r.stderr, (new, r.returncode, r.stderr[:200])             # RED: the pod script passing the parent's 400 (or less) is refused
            r = run_pod("S1ZD", wsx, script=rp, mode="run")
            assert r.returncode == 2
        open(rp, "w", encoding="utf-8", newline="").write(txt.replace("DOSE_UPDATES=1600", "DOSE_UPDATES=3200"))
        r = run_pod("S1ZD", wsx, script=rp)
        assert r.returncode == 0 and r.stdout.split()[r.stdout.split().index("--max-updates") + 1] == "3200" and pairs([t.replace(wsx, "/workspace") for t in r.stdout.split()]) != pairs(L.dose_argv("S1ZD", WZ)), "a changed dose differs from launch_gate's argv"
        # MUTATION: the pod script's DOSE_EPOCHS dropped (the card's original argv): its output loses --epochs, the A1 diff goes RED, the agreement regex goes RED, and the real profile refuses that argv
        mtxt = txt.replace('DOSE_EPOCHS="2.0"', 'DOSE_EPOCHS=""')
        assert mtxt != txt
        open(rp, "w", encoding="utf-8", newline="").write(mtxt)
        r = run_pod("S1ZD", wsx, script=rp)
        assert r.returncode == 0 and "--epochs" not in r.stdout, r.stderr[:200]
        mo = [x.replace(wsx, "/workspace") for x in r.stdout.split()]
        assert differing_flags(mo, pout) == {"--max-updates", "--out"} != {"--max-updates", "--out", "--epochs"}, "MUTATION: pod DOSE_EPOCHS dropped must turn the pod-output A1 diff RED"
        assert not re.search(r'^DOSE_EPOCHS="2.0"$', mtxt, re.M), "MUTATION: the pod/launch_gate agreement check goes RED"
        assert has(refusals(mo), "cannot reach") and has(refusals(mo), "--epochs"), "the argv of that mutated pod script is refused by the real profile"
        # a parent whose --max-updates is not 400 is refused
        sp = os.path.join(td, "red", "s1z_pod.sh")
        st = open(sp, encoding="utf-8", newline="").read()
        assert "--max-updates 400 --eval-every" in st
        open(sp, "w", encoding="utf-8", newline="").write(st.replace("--max-updates 400 --eval-every", "--max-updates 500 --eval-every"))
        open(rp, "w", encoding="utf-8", newline="").write(txt)
        r = run_pod("S1ZD", wsx, script=rp)
        assert r.returncode == 2 and "the dose card was written against" in r.stderr, (r.returncode, r.stderr[:200])
    # launch_gate mutations (a mutated COPY of the module, loaded under another name; the real module is untouched)
    src = open(os.path.join(HERE, "launch_gate.py"), encoding="utf-8", newline="").read()

    def load_mut(a, b):
        assert src.count(a) == 1, a
        with tempfile.TemporaryDirectory(dir=tmp) as tdm:
            p = os.path.join(tdm, "launch_gate_mut.py")
            open(p, "w", encoding="utf-8", newline="").write(src.replace(a, b))
            spec = importlib.util.spec_from_file_location("launch_gate_mut", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m

    mut400 = load_mut('"S1ZD": {"parent": "S1Z", "family": "s1z", "max_updates": 1600}}', '"S1ZD": {"parent": "S1Z", "family": "s1z", "max_updates": 400}}')       # MUTATION: 400 instead of 1600
    try:
        mut400.dose_argv("S1ZD", WZ)
    except ValueError as e:
        assert "not an int above the parent's 400" in str(e)
    else:
        raise AssertionError("MUTATION 400 instead of 1600 was not refused")
    mut_nop = load_mut('    toks[i + 1] = str(sp["max_updates"])\n', '')                       # the --max-updates value is never swapped: the argv IS S1Z's
    assert not two_token_diff(mut_nop), "MUTATION: dose_argv leaving --max-updates at 400 must turn the argv-diff check RED"
    mut_out = load_mut('    toks[toks.index("--out") + 1] = DOSE_ROOT + "/" + arm.lower()\n    return toks\n', '    return toks\n')
    assert not two_token_diff(mut_out), "MUTATION: dose_argv leaving --out at S1Z's must turn the argv-diff check RED"
    mut_reach = load_mut('    if got != max_updates:\n        return [f"the dose cannot reach', '    if False:\n        return [f"the dose cannot reach')
    assert mut_reach.dose_reach_refusals(1600, 1.0) == [] and L.dose_reach_refusals(1600, 1.0), "MUTATION: a reach rule that never refuses is DETECTED (938 updates would pass)"
    mut_ep = load_mut("\nDOSE_EPOCHS = 2.0 ", "\nDOSE_EPOCHS = None ")                  # MUTATION: DOSE_EPOCHS back to None (the card's original two-value argv)
    mut_ep.sha256_of_file = fake_sha
    mut_ep.HERE = L.HERE                                                                  # the mutant copy lives in a temp dir; its sibling sources are the real ones
    assert mut_ep.DOSE_EPOCHS is None and not two_token_diff(mut_ep), "MUTATION: DOSE_EPOCHS None must turn the A1 argv-diff check RED"
    assert "--epochs" not in mut_ep.PROFILES["refe_dose_s1z_p6"]["required_flags"] and mut_ep.PROFILES["refe_dose_s1z_p6"]["recipe"].get("epochs") != 2.0
    assert has(mut_ep.profile_refusals(mut_ep.PROFILES["refe_dose_s1z_p6"], mut_ep.dose_argv("S1ZD", WZ)), "cannot reach"), "with DOSE_EPOCHS None the card's argv is refused by the reach rule"
    assert fp("refe_dose_s1z_p6") != hashlib.sha256(repr(sorted(mut_ep.PROFILES["refe_dose_s1z_p6"].items())).encode()).hexdigest(), "the dose digest moves with DOSE_EPOCHS"
    mut_req = load_mut('"--max-updates": str(DOSE_ARMS["S1ZD"]["max_updates"])', '"--max-updates": "400"')
    assert mut_req.PROFILES["refe_dose_s1z_p6"]["required_flags"]["--max-updates"] == "400" and PD["required_flags"]["--max-updates"] == "1600"
    print("D5 ok (bash -n x3; dose_pod.sh vs launch_gate vs s1z_pod.sh, both DOSE_EPOCHS 2.0; no calibrate mode; S1Z's lambda only; mutations RED incl. DOSE_EPOCHS None / pod DOSE_EPOCHS dropped)")
else:
    print("D5 SKIPPED (no bash)")
print("ZZTEST_PASS")
