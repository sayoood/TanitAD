#!/usr/bin/env python3
"""lit1_selftest_sh.py -- the SHELL LAYER of lit1_selftest.py: lit1_chain.sh / wake_lit1.sh run for real (through lit1_sealed.run) in throw-away worlds under F:/TanitAD-scratch/launch_intent_t1/w_<name>.
Each scenario is a function ``fn(w) -> [failure, ...]`` (empty = pass).  A MUTANT is a textual edit of the world's own copy of the package (w.H) followed by the SAME scenario: it is killed iff the scenario then fails.
The stub tool writes the files the real tool writes (see lit1_fix.STUB_TOOL); nothing here is a rollout; no process other than bash, python and git (read-only) is started."""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import lit1_fix as X  # noqa: E402
import lit1_lib as L  # noqa: E402
import lit1_sealed as S  # noqa: E402

Chk = X.Chk
A0, A1, R0, R1 = ("launchintent", 0), ("launchintent", 1), ("launchintentrep", 0), ("launchintentrep", 1)
SC: dict = {}      # scenario name -> fn(w)


def scn(name):
    def deco(fn):
        SC[name] = fn
        return fn
    return deco


def Ct(r):
    return L.t1_tag(r[0], r[1], "causal")


def Gt(r):
    return L.t1_tag(r[0], r[1], "logged")


def Zt(r):
    return L.t0f_tag(r[0], r[1])


def go(w, args="", timeout=420, drop=(), world=True, **env):
    e = w.env(LIT1_TEST_RAM_MB="65536", LIT1_TEST_DISK_GB="1000")
    e.update(env)
    if not world:
        e.pop("LIT1_WORLD", None)
        e.pop("LIT1_TEST_RAM_MB", None)
        e.pop("LIT1_TEST_DISK_GB", None)
        e.setdefault("LIT1_EXPECT_PREREG_SHA", "none")      # a real-mode scenario runs with the world's STUB reader / PREREG: the landed-file pins are switched off unless the scenario sets them
        e.setdefault("LIT1_EXPECT_READER_SHA", "none")
    return S.run(w.chain_cmd(args), env=e, timeout=timeout, drop=drop)


def marks(w):
    t = w.chain_out()
    return {"stage": re.findall(r"^ZZLIT1-STAGE-DONE (\d)/4 (\S+) \(([^)]*)\)", t, re.M), "stop": re.findall(r"^ZZLIT1-STOP (.*)$", t, re.M), "done": re.findall(r"^ZZLIT1-DONE (.*)$", t, re.M)}


def starts(w, skip=0):
    return [ln.split()[1] for ln in w.calls()[skip:] if ln.startswith("START ")]


def stop_text(w, out):
    m = marks(w)
    return (m["stop"][-1] if m["stop"] else "") or out


def unfinished(c, w, out, rc, why):
    """Common assertions of a scenario that must end in ONE ZZLIT1-STOP carrying ``why``, with no reader call and no ZZLIT1-DONE."""
    m = marks(w)
    c.eq(rc, 1, "chain rc")
    c.eq(len(m["stop"]), 1, "exactly one ZZLIT1-STOP marker")
    c.eq(m["done"], [], "no ZZLIT1-DONE")
    c.has(" ".join(m["stop"]) + out, why, "the STOP names its cause")
    c.eq(w.reader_calls(), [], "the reader was not called")
    c.no(out, "ZZHARNESS-REFUSED", "the sealed guard did not refuse")


def sequential(w, c):
    open_tag = None
    for ln in w.calls():
        p = ln.split()
        if p and p[0] == "START":
            if open_tag is not None:
                c.fails.append(f"{p[1]} started while {open_tag} was still running (two torch processes at once)")
            open_tag = p[1]
        elif p and p[0] == "END":
            open_tag = None


def flip(path, key):
    t = open(path, encoding="utf-8").read()
    m = re.search(r'("%s": ")([0-9a-f])' % key, t)
    assert m, (path, key)
    t = t[:m.start(2)] + ("0" if m.group(2) != "0" else "1") + t[m.end(2):]
    open(path, "w", encoding="utf-8", newline="\n").write(t)


# ================================================================================================================================ S0: the sealed harness itself
@scn("s0_harness")
def s0_harness(w):
    c = Chk()
    wd = f"{L.SCRATCH}/w_s0_harness"
    os.makedirs(wd, exist_ok=True)
    for i, (name, env, arm_drop, mut_drop) in enumerate(S.arms()):
        wit = f"{wd}/witness_{i}.txt"
        if os.path.exists(wit):
            os.remove(wit)
        cmd = f'echo ran > "{S.px(wit)}"; command -v ssh'
        rc, out = S.run(cmd, env=env, drop=arm_drop)
        c.eq(rc, 97, f"[{name}] the guard must REFUSE")
        c.has(out, "ZZHARNESS-REFUSED", f"[{name}] the refusal text")
        c.ok(not os.path.exists(wit), f"[{name}] NOTHING ran under a refusing guard")
        rc2, out2 = S.run(cmd, env=env, drop=mut_drop)
        c.eq(rc2, 0, f"[{name}] with the check removed the defect passes through (the check is load-bearing): {out2[:200]}")
        c.ok(os.path.exists(wit), f"[{name}] the mutant harness ran the step")
    rc, out = S.run('echo "$HOME"; command -v ssh; command -v scp', env={})
    c.eq(rc, 0, "a clean sealed run")
    c.has(out, S.px(S.NOHOME), "HOME is the empty scratch HOME")
    c.has(out, S.px(S.GSTUB) + "/ssh", "ssh is the world stub")
    c.has(out, S.px(S.GSTUB) + "/scp", "scp is the world stub")
    return c.fails


# ================================================================================================================================ S1: the happy path (+ the manifest verification battery)
def tamper(path, fn):
    old = open(path, "rb").read()
    open(path, "wb").write(fn(old))
    return lambda: open(path, "wb").write(old)


def verify_cli(w, manifest=None, digest=None, n=141):
    r = subprocess.run([L.PY, "-I", "-B", f"{w.H}/lit1_tree.py", "verify", "--manifest", manifest or f"{w.out}/runs.json", "--n-dump-min", str(n), "--tool-digest", digest or w.digest],
                       capture_output=True, stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    return r.returncode, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")


def verify_battery(w, only=None):
    """The manifest builder's verifier on a finished world: the clean manifest passes; each tamper below must make it REFUSE (rc 3).  ``only`` limits it to the named cases (a mutant run)."""
    c = Chk()
    import json
    O, A = w.out, w.art
    rc, txt = verify_cli(w)
    c.eq(rc, 0, f"verify: the clean manifest ({txt[:200]})")
    cases = []

    def case(name, setup):
        if only is None or name in only:
            cases.append((name, setup))

    def dumps(tag):
        return sorted(glob.glob(f"{O}/dump_{tag}/ep*.npz"))

    def removed(path):
        old = open(path, "rb").read()
        os.remove(path)
        return lambda: open(path, "wb").write(old)

    def flipped(path, key):
        old = open(path, "rb").read()
        flip(path, key)
        return lambda: open(path, "wb").write(old)
    case("dump", lambda: removed(dumps(Ct(A0))[-1]))
    case("done-after", lambda: flipped(f"{O}/done_{Gt(A1)}.txt", "ckpt_sha256_after"))
    case("done-ck", lambda: flipped(f"{O}/done_{Gt(A1)}.txt", "ckpt_sha256"))
    case("done-before", lambda: flipped(f"{O}/done_{Ct(R0)}.txt", "ckpt_sha256_before"))
    case("done-tool", lambda: flipped(f"{O}/done_{Ct(R1)}.txt", "tool_digest"))
    case("done-smoke", lambda: tamper(f"{O}/done_{Ct(A0)}.txt", lambda b: b.replace(b'"smoke": false', b'"smoke": true')))
    case("t0f-tool", lambda: flipped(f"{O}/done_t0f_{Zt(A1)}.txt", "tool_digest"))
    case("t0f-ck", lambda: flipped(f"{O}/done_t0f_{Zt(A0)}.txt", "ckpt_sha256"))
    case("t0f-smoke", lambda: tamper(f"{O}/done_t0f_{Zt(R0)}.txt", lambda b: b.replace(b'"smoke": false', b'"smoke": true')))
    case("t0f-ages", lambda: tamper(f"{O}/t0fresh/eval_{Zt(A0)}.json", lambda b: b.replace(b'"fresh"', b'"cycle"')))
    case("t0f-ckpt", lambda: tamper(f"{O}/t0fresh/eval_{Zt(A1)}.json", lambda b: b.replace(b"M2b_s1_launchintent/", b"M2b_s0_launchintent/")))
    case("t0-ages", lambda: tamper(f"{A}/M2b_s0_launchintent/eval_M2b-launchintent_s0.json", lambda b: b.replace(b'"cycle"', b'"fresh"')))
    case("t0-ckpt", lambda: tamper(f"{A}/M2b_s1_launchintent/eval_M2b-launchintent_s1.json", lambda b: b.replace(b"M2b_s1_launchintent/", b"M2b_s0_launchintent/")))
    case("limit", lambda: tamper(f"{O}/t1v8f_{Ct(R0)}.json", lambda b: b.replace(b'"limit": "causal"', b'"limit": "logged"')))
    case("max-windows", lambda: tamper(f"{O}/t1v8f_{Gt(R1)}.json", lambda b: b.replace(b'"max_windows": null', b'"max_windows": 5')))
    case("n-windows", lambda: tamper(f"{O}/t1v8f_{Ct(R1)}.json", lambda b: b.replace(b'"n_windows": 2399', b'"n_windows": 100')))
    case("t1-ckpt", lambda: tamper(f"{O}/t1v8f_{Gt(A0)}.json", lambda b: b.replace(b"M2b_s0_launchintent/", b"M2b_s1_launchintent/")))
    case("missing-ticks", lambda: removed(f"{O}/ticks/ticks_{Gt(R0)}.npz"))
    case("missing-t0f-done", lambda: removed(f"{O}/done_t0f_{Zt(R1)}.txt"))
    case("missing-selfcheck", lambda: removed(f"{A}/M2b_s0_launchintentrep/selfcheck.json"))
    case("empty-argv", lambda: tamper(f"{A}/M2b_s1_launchintent/argv.json", lambda b: b""))

    def dup():
        a0, a1 = f"{A}/M2b_s0_launchintent/ckpt_last.pt", f"{A}/M2b_s1_launchintent/ckpt_last.pt"
        old_a0 = open(a0, "rb").read()
        old_sha, new_sha = L.sha256_file(a0), L.sha256_file(a1)
        recs = [f"{O}/done_{Ct(A0)}.txt", f"{O}/done_{Gt(A0)}.txt", f"{O}/done_t0f_{Zt(A0)}.txt"]
        olds = {r: open(r, "rb").read() for r in recs}
        open(a0, "wb").write(open(a1, "rb").read())
        for r in recs:
            open(r, "wb").write(olds[r].replace(old_sha.encode(), new_sha.encode()))

        def undo():
            open(a0, "wb").write(old_a0)
            for r in recs:
                open(r, "wb").write(olds[r])
        return undo
    case("dup-ckpt", dup)
    for name, setup in cases:
        undo = setup()
        rc, txt = verify_cli(w)
        if undo:
            undo()
        c.eq(rc, 3, f"verify tamper [{name}] must be REFUSED: {txt[:160]}")
    if only is None or "digest-param" in only:
        rc, txt = verify_cli(w, digest="0" * 64)
        c.eq(rc, 3, "verify with another tool digest must be REFUSED")
    if only is None or "ctrl-swap" in only:
        m = json.load(open(f"{O}/runs.json"))
        m["arms"]["CTRL"]["0"]["train_dir"] = m["arms"]["A"]["0"]["train_dir"]
        mp = f"{w.root}/runs_ctrl_swapped.json"
        json.dump(m, open(mp, "w"))
        rc, txt = verify_cli(w, manifest=mp)
        c.eq(rc, 3, "verify with CTRL pointing at a trained arm must be REFUSED")
    rc, txt = verify_cli(w)
    c.eq(rc, 0, f"verify: the manifest is clean again after every tamper was undone ({txt[:200]})")
    return c.fails


@scn("s1_happy")
def s1_happy(w, battery=True):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, LIT1_EXPECT_COMMIT=L.LAUNCH_COMMIT)
    m = marks(w)
    c.eq(rc, 0, f"chain rc ({out[-300:]})")
    c.no(out, "ZZHARNESS-REFUSED", "sealed guard")
    c.eq([x[0] for x in m["stage"]], ["1", "2", "3", "4"], "STAGE-DONE numbering")
    c.eq([x[1] for x in m["stage"]], [L.run_name(*r) for r in (A0, A1, R0, R1)], "STAGE-DONE run order (the box-2 log's pull order)")
    c.eq([x[2] for x in m["stage"]], ["A_0", "A_1", "A_rep,0", "A_rep,1"], "STAGE-DONE labels")
    c.eq(m["stop"], [], "no STOP")
    c.eq(len(m["done"]), 1, "one ZZLIT1-DONE")
    c.ok(m["done"] and m["done"][0].startswith("12/12 steps, reader rc 0 (outcome row written)"), f"DONE text: {m['done']}")
    want = []
    for r in (A0, A1, R0, R1):
        want += [Ct(r), Gt(r), Zt(r)]
    c.eq(starts(w), want, "the 12 steps, in order: causal, logged, fresh T0 of each run")
    sequential(w, c)
    done_files = sorted(os.listdir(w.out))
    c.eq(len([f for f in done_files if f.startswith("done_")]), 12, "12 done records")
    for r in (A0, A1, R0, R1):
        ck = f"{w.art}/{L.run_name(*r)}/ckpt_last.pt"
        c.eq(L.sha256_file(ck), w.sha[r], f"{L.LABEL[r]}: the checkpoint is byte-identical after the chain")
        for tag in (Ct(r), Gt(r)):
            rec = open(f"{w.out}/done_{tag}.txt", encoding="utf-8").read()
            c.ok(f'"ckpt_sha256": "{w.sha[r]}"' in rec and f'"ckpt_sha256_before": "{w.sha[r]}"' in rec and f'"ckpt_sha256_after": "{w.sha[r]}"' in rec and '"smoke": false' in rec, f"done record of {tag}")
    argv = {ln.split()[1]: ln for ln in w.calls() if ln.startswith("ARGV ")}
    for flag in ("--limit causal", "--replan-steps 10", "--nav deadreckon", "--scene logged", "--schedule periodic", "--dump-ticks", "--dump-heads", "--device cpu", f"--ckpt {w.art}/M2b_s0_launchintent/ckpt_last.pt"):
        c.has(argv.get(Ct(A0), ""), flag, "causal rollout argv")
    c.has(argv.get(Gt(A0), ""), "--limit logged", "logged rollout argv")
    c.has(argv.get(Zt(A0), ""), "--ages fresh", "fresh T0 argv")
    c.has(argv.get(Zt(R1), ""), f"--ckpt {w.art}/M2b_s1_launchintentrep/ckpt_last.pt", "A_rep,1's checkpoint")
    rcalls = w.reader_calls()
    c.eq(len(rcalls), 1, "the reader was called exactly once")
    c.ok(rcalls and "n_done=12" in rcalls[0] and "arms=A,Arep,CTRL,REP" in rcalls[0], f"the reader saw 12 done records and the four slots (only after every step): {rcalls}")
    c.ok(rcalls and f"--prereg-sha {w.psha}" in rcalls[0] and "--manifest" in rcalls[0], "the reader got the token-bound PREREG sha256")
    c.ok(os.path.isfile(f"{w.out}/runs.json") and os.path.isfile(f"{w.out}/read/LAUNCH_INTENT_READ.json"), "manifest and reader output exist")
    c.eq(open(f"{w.out}/LAUNCH_COMMIT.txt").read().strip(), L.LAUNCH_COMMIT, "the launch commit the tokens bind is recorded")
    sh_calls = f"{S.GSTUB}/calls.log"
    c.ok(not os.path.exists(sh_calls) or os.path.getsize(sh_calls) == 0, "no remote-shell stub was ever called")
    if battery:
        w.digest = None
        import lit1_tree as T
        w.digest = T.tree_digest(f"{w.root}/tree_stub")
        c.fails += verify_battery(w)
    return c.fails


# ================================================================================================================================ S2: pull order and unfinished pulls
@scn("s2_order")
def s2_order(w):
    c = Chk()
    order = [R0, A1, A0, R1]
    w.all_pulls()
    w.log(specs=[(a, s, {}) for a, s in order])
    rc, out = go(w)
    m = marks(w)
    c.eq(rc, 0, "chain rc")
    c.eq([x[1] for x in m["stage"]], [L.run_name(*r) for r in order], "runs are processed in the order their pulls landed, not the box-2 queue order")
    c.eq(starts(w)[:3], [Ct(R0), Gt(R0), Zt(R0)], "the first pulled run is rolled out first")
    return c.fails


@scn("s2b_unfinished_pull")
def s2b(w):
    c = Chk()
    w.pull(*A0, sums=False)
    for r in (A1, R0, R1):
        w.pull(*r)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "not complete yet")
    c.eq(starts(w), [], "a later complete pull is NOT started ahead of an earlier unfinished one (pull order)")
    return c.fails


# ================================================================================================================================ S3: failures, resume, tampered done records
@scn("s3_resume")
def s3(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("failafter:" + Gt(A1))
    rc, out = go(w)
    m = marks(w)
    c.eq(rc, 1, "run 1 stops")
    c.has(" ".join(m["stop"]), "failed (rc 1", "run 1's STOP names the failing rollout")
    c.eq([x[1] for x in m["stage"]], [L.run_name(*A0)], "only A_0 finished")
    c.eq(starts(w), [Ct(A0), Gt(A0), Zt(A0), Ct(A1), Gt(A1)], "run 1's steps")
    c.ok(not os.path.exists(f"{w.out}/done_{Gt(A1)}.txt"), "a failed rollout leaves NO done record")
    n1 = len(w.calls())
    w.ctl()
    rc, out = go(w)
    m = marks(w)
    c.eq(rc, 0, f"run 2 (RESUME = the same command) finishes ({out[-200:]})")
    c.eq(starts(w, n1), [Gt(A1), Zt(A1), Ct(R0), Gt(R0), Zt(R0), Ct(R1), Gt(R1), Zt(R1)], "run 2 does only the 8 unfinished steps")
    c.has(w.chain_out(), f"skip {Ct(A0)} (complete, same checkpoint", "a finished step is skipped by its done record")
    c.eq([x[0] for x in m["stage"]], ["1", "2", "3", "4"], "all four STAGE-DONE after the resume")
    c.ok(m["done"] and m["done"][0].startswith("12/12 steps"), "DONE after the resume")
    inc = glob.glob(f"{w.out}/_incomplete_*/t1v8f_{Gt(A1)}.json")
    c.ok(len(inc) == 1, f"the failed attempt's output was MOVED ASIDE, not deleted: {inc}")
    return c.fails


def _tamper_round(w, c, specs, label):
    """specs = [(tag, kind, arg)].  Tamper, re-run the SAME command, and require exactly those steps to run again."""
    O = w.out
    exp = []
    for tag, kind, arg in specs:
        t0 = tag.startswith("M2b-")
        done = f"{O}/done_t0f_{tag}.txt" if t0 else f"{O}/done_{tag}.txt"
        if kind == "flip":
            flip(done, arg)
        elif kind == "del-done":
            os.remove(done)
        elif kind == "del-json":
            os.remove(f"{O}/t1v8f_{tag}.json")
        elif kind == "del-ticks":
            os.remove(f"{O}/ticks/ticks_{tag}.npz")
        elif kind == "del-argv":
            os.remove(f"{O}/argv/{tag}.argv")
        elif kind == "del-dumps":
            for f in sorted(glob.glob(f"{O}/dump_{tag}/ep*.npz"))[:5]:
                os.remove(f)
        elif kind == "del-t0json":
            os.remove(f"{O}/t0fresh/eval_{tag}.json")
        elif kind == "del-t0npz":
            os.remove(f"{O}/t0fresh/eval_{tag}.npz")
        exp.append(tag)
    n0 = len(w.calls())
    rc, out = go(w)
    got = starts(w, n0)
    c.eq(rc, 0, f"{label}: the chain finishes ({out[-200:]})")
    c.eq(sorted(got), sorted(exp), f"{label}: exactly the tampered steps run again (every other step is skipped by its done record)")
    return rc


@scn("s3_tamper")
def s3_tamper(w, rounds=(1, 2)):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w)
    c.eq(rc, 0, f"first pass ({out[-200:]})")
    if rc != 0:
        return c.fails
    if 1 in rounds:
        _tamper_round(w, c, [(Ct(A0), "flip", "ckpt_sha256"), (Ct(A1), "flip", "ckpt_sha256_before"), (Ct(R0), "flip", "ckpt_sha256_after"), (Ct(R1), "flip", "tool_digest"),
                             (Gt(A0), "flip", "cmd_sha256"), (Gt(A1), "del-dumps", None), (Gt(R0), "del-ticks", None), (Gt(R1), "del-argv", None),
                             (Zt(A0), "flip", "ckpt_sha256"), (Zt(A1), "flip", "ckpt_sha256_before"), (Zt(R0), "flip", "ckpt_sha256_after"), (Zt(R1), "flip", "tool_digest")], "round 1")
        c.ok(len(glob.glob(f"{w.out}/_incomplete_*/done_{Ct(A0)}.txt")) >= 1, "round 1: the invalidated done record was MOVED ASIDE (nothing deleted)")
        if c.fails:
            return c.fails
    if 2 in rounds:
        _tamper_round(w, c, [(Ct(A0), "del-json", None), (Ct(A1), "del-done", None), (Ct(R0), "del-argv", None), (Ct(R1), "del-ticks", None), (Gt(A0), "del-json", None),
                             (Zt(A0), "flip", "cmd_sha256"), (Zt(A1), "del-t0json", None), (Zt(R0), "del-t0npz", None), (Zt(R1), "del-done", None)], "round 2")
        c.ok(len(glob.glob(f"{w.out}/_incomplete_*/dump_{Gt(A0)}")) >= 1, "round 2: the earlier attempt's dump directory was MOVED ASIDE (nothing deleted)")
    return c.fails


SC["s1_happy_nb"] = lambda w: s1_happy(w, battery=False)
SC["s3_tamper_r1"] = lambda w: s3_tamper(w, rounds=(1,))
SC["s3_tamper_r2"] = lambda w: s3_tamper(w, rounds=(2,))


@scn("s3c_changed_checkpoint")
def s3c(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w)
    c.eq(rc, 0, "first pass")
    old = w.sha[A0]
    w.pull(*A0, mut=lambda P: P.__setitem__("ckpt", X.det_bytes("a-different-checkpoint", 9000)))
    c.ok(w.sha[A0] != old, "the fixture really changed A_0's checkpoint")
    w.log()
    n0 = len(w.calls())
    rc, out = go(w)
    c.eq(rc, 0, f"second pass ({out[-200:]})")
    c.eq(sorted(starts(w, n0)), sorted([Ct(A0), Gt(A0), Zt(A0)]), "a NEW checkpoint under the same name invalidates A_0's done records (and only A_0's)")
    return c.fails


@scn("s3d_changed_command")
def s3d(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w)
    c.eq(rc, 0, "first pass")
    art2 = w.art + "2"
    shutil.copytree(w.art, art2)
    w.art = art2
    w.log()
    n0 = len(w.calls())
    rc, out = go(w, LIT1_ART=art2)
    c.eq(rc, 0, f"second pass ({out[-200:]})")
    c.eq(len(starts(w, n0)), 12, "the same bytes under another PATH change the command: every step runs again (the done record binds the command sha256)")
    return c.fails


@scn("s4_ckpt_mutated")
def s4(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("mutate_ckpt:" + Ct(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "changed during " + Ct(A0))
    c.ok(not os.path.exists(f"{w.out}/done_{Ct(A0)}.txt"), "no done record for a rollout whose checkpoint changed under it")
    c.eq(starts(w), [Ct(A0)], "stopped at once")
    return c.fails


@scn("s4c_ckpt_mutated_t0")
def s4c(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("mutate_ckpt:" + Zt(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "changed during " + Zt(A0))
    c.ok(not os.path.exists(f"{w.out}/done_t0f_{Zt(A0)}.txt"), "no done record")
    return c.fails


@scn("s4b_tree_mutated")
def s4b(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("mutate_tree:" + Ct(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "the tool tree changed (after " + Ct(A0))
    c.eq(starts(w), [Ct(A0)], "stopped at once")
    return c.fails


@scn("s4d_tree_mutated_t0")
def s4d(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("mutate_tree:" + Zt(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "the tool tree changed (after " + Zt(A0))
    return c.fails


@scn("s3f_t0_fail")
def s3f(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("failafter:" + Zt(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "fresh-T0 eval " + Zt(A0) + " failed (rc 1")
    c.ok(not os.path.exists(f"{w.out}/done_t0f_{Zt(A0)}.txt"), "no done record for a failed fresh-T0 eval")
    return c.fails


@scn("s3g_few_dumps")
def s3g(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("fewdumps:" + Ct(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "dumps 5/141")
    return c.fails


@scn("s3h_manifest_refused")
def s3h(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("badlimit:" + Ct(R1))
    rc, out = go(w)
    m = marks(w)
    c.eq(rc, 1, "chain rc")
    c.has(" ".join(m["stop"]), "manifest:", "the manifest verifier's refusal stops the chain")
    c.eq(m["done"], [], "no ZZLIT1-DONE")
    c.eq(w.reader_calls(), [], "the reader is not run on a manifest that did not verify")
    c.ok(not os.path.exists(f"{w.out}/runs.json"), "no manifest is written")
    return c.fails


# ================================================================================================================================ S5-S9: the run record, the controls, the log
def _lever0(P):
    P["summary"]["levers"]["launch_intent_w"] = 0.0
    P["selfcheck"]["levers"]["launch_intent_w"] = 0.0


@scn("s5_wrong_lever_third")
def s5(w):
    c = Chk()
    w.pull(*A0)
    w.pull(*A1)
    w.pull(*R0, mut=_lever0)
    w.pull(*R1)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "is WRONG")
    c.has(" ".join(marks(w)["stop"]), "[sum-lever]", "the STOP names the clause")
    c.eq([x[1] for x in marks(w)["stage"]], [L.run_name(*A0), L.run_name(*A1)], "the two good runs were processed first")
    c.eq(len(starts(w)), 6, "no rollout of the run whose record lacks the lever")
    return c.fails


@scn("s6_kshare_fail_line")
def s6(w):
    c = Chk()
    w.all_pulls()
    w.log(specs=[(A0[0], A0[1], {"ksh": "FAIL"})] + [(a, s, {}) for a, s in (A1, R0, R1)])
    rc, out = go(w)
    unfinished(c, w, out, rc, "FAILED a control")
    c.eq(starts(w), [], "a run that failed its controls is never read")
    return c.fails


@scn("s6b_log_lies")
def s6b(w):
    c = Chk()
    w.pull(*A0, mut=lambda P: (P["summary"].update(launch_n=57000, launch_eligible=191311, launch_rows=191311, launch_weight_share=L.share_of(57000, 191311, L.LI_W))))
    for r in (A1, R0, R1):
        w.pull(*r)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "is WRONG")
    c.has(" ".join(marks(w)["stop"]), "[sum-share]", "the chain re-computed K-SHARE from the pulled summary though the log said PASS")
    c.eq(starts(w), [], "no rollout")
    return c.fails


@scn("s7_missing_share")
def s7(w):
    c = Chk()
    w.pull(*A0, mut=lambda P: P["summary"].pop("launch_weight_share"))
    for r in (A1, R0, R1):
        w.pull(*r)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "is WRONG")
    c.has(" ".join(marks(w)["stop"]), "[sum-keys]", "a run built before the lever (no launch_weight_share) is refused")
    c.eq(starts(w), [], "no rollout")
    return c.fails


@scn("s8_log_sha")
def s8(w):
    c = Chk()
    w.all_pulls()
    w.sha[A0] = "0" * 64
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "box-2 log printed")
    c.eq(starts(w), [], "no rollout")
    return c.fails


@scn("s8b_file_changed_after_sums")
def s8b(w):
    c = Chk()
    d = w.pull(*A0)
    open(d + "/ckpt_last.pt", "ab").write(b"x")
    for r in (A1, R0, R1):
        w.pull(*r)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "not complete yet")
    c.eq(starts(w), [], "no rollout")
    return c.fails


@scn("s9_control_fail")
def s9(w):
    c = Chk()
    w.all_pulls()
    w.log(extra_after=[f"ZZLAUNCHINTENT-CONTROL-FAIL {L.run_name(*A0)} K-SHARE: share 0.5"], done=False)
    rc, out = go(w)
    unfinished(c, w, out, rc, "FAILED a control")
    c.eq(starts(w), [], "no rollout")
    return c.fails


@scn("s9b_control_fail_no_pull")
def s9b(w):
    c = Chk()
    for r in (A0, A1, R0):
        w.pull(*r)
    w.log(specs=[(a, s, {}) for a, s in (A0, A1, R0)], extra_after=[f"ZZLAUNCHINTENT-CONTROL-FAIL {L.run_name(*R1)} run record: [argv-lever]"], done=False)
    rc, out = go(w)
    unfinished(c, w, out, rc, "FAILED a control")
    c.eq([x[0] for x in marks(w)["stage"]], ["1", "2", "3"], "the three runs that passed were processed first")
    return c.fails


# ================================================================================================================================ S10: guards
@scn("s10a_ram_low")
def s10a(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, LIT1_TEST_RAM_MB="100")
    unfinished(c, w, out, rc, "a guard stayed LOW")
    c.has(out, "RAM available 100 MB < 4096 MB", "the RAM message")
    c.eq(starts(w), [], "no rollout under low RAM")
    return c.fails


@scn("s10b_disk_low")
def s10b(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, LIT1_TEST_DISK_GB="1")
    unfinished(c, w, out, rc, "a guard stayed LOW")
    c.has(out, "free 1.0 GB < 5.0 GB", "the disk message")
    c.eq(starts(w), [], "no rollout on a nearly full output drive")
    return c.fails


@scn("s10c_ram_drops_before_logged")
def s10c(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("lowram_after:" + Ct(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "a guard stayed LOW")
    c.eq(starts(w), [Ct(A0)], "the guard runs before EVERY rollout: the logged lane does not start once RAM dropped")
    return c.fails


@scn("s10d_ram_drops_before_t0")
def s10d(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("lowram_after:" + Gt(A0))
    rc, out = go(w)
    unfinished(c, w, out, rc, "a guard stayed LOW")
    c.eq(starts(w), [Ct(A0), Gt(A0)], "the guard runs before the fresh T0 eval too")
    return c.fails


@scn("s10e_c_low")
def s10e(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, LIT1_C_MIN_MB="99999999")
    unfinished(c, w, out, rc, "C: has")
    c.eq(starts(w), [], "no rollout")
    return c.fails


# ================================================================================================================================ S11: one instance
@scn("s11_lock")
def s11(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("sleep:" + Ct(A0))
    e = w.env(LIT1_TEST_RAM_MB="65536", LIT1_TEST_DISK_GB="1000")
    p = S.popen(w.chain_cmd(), env=e)
    t0 = time.time()
    while time.time() - t0 < 90 and not starts(w):
        time.sleep(0.5)
    c.ok(starts(w), "the first instance started a rollout")
    rc2, out2 = go(w, timeout=120)
    c.eq(rc2, 1, "the second instance refuses")
    c.has(out2, "another instance of this chain is running", "the reason")
    c.eq(len(starts(w)), 1, "the second instance started nothing while the first rollout ran")
    try:
        o1, _ = p.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
        o1, _ = p.communicate()
        c.fails.append("the first instance did not finish")
    c.eq(p.returncode, 0, "the first instance finishes")
    c.eq(len(marks(w)["done"]), 1, "the first instance's DONE")
    sequential(w, c)
    return c.fails


@scn("s11b_stale_lock")
def s11b(w):
    c = Chk()
    os.makedirs(f"{w.logs}/chain.lock", exist_ok=True)
    open(f"{w.logs}/chain.lock/winpid", "w").write("2147480000\n")
    rc, out = go(w)
    c.no(out, "another instance", "a lock whose owner is gone is taken over")
    c.has(out, "=== lit1_chain start", "the chain started")
    return c.fails


# ================================================================================================================================ S12: the end of the chain
def _reader_rc(w, rc, why):
    c = Chk()
    open(f"{w.root}/reader_rc.txt", "w").write(str(rc))
    w.all_pulls()
    w.log()
    r, out = go(w)
    m = marks(w)
    c.eq(r, 0, "chain rc")
    c.ok(m["done"] and m["done"][0].startswith(f"12/12 steps, reader rc {rc} ({why})"), f"the reader's exit code and its meaning are in the DONE line: {m['done']}")
    c.eq(len(w.reader_calls()), 1, "the reader ran once")
    return c.fails


SC["s12a_reader_rc2"] = lambda w: _reader_rc(w, 2, "REFUSED, nothing written")
SC["s12a_reader_rc4"] = lambda w: _reader_rc(w, 4, "BLOCKED, a required slot incomplete")


@scn("s12b_no_box2_done")
def s12b(w):
    c = Chk()
    w.all_pulls()
    w.log(done=False)
    rc, out = go(w)
    unfinished(c, w, out, rc, "no ZZV8FMINI-DONE")
    c.eq(len(starts(w)), 12, "all twelve steps ran before the chain waited for the DONE marker")
    return c.fails


@scn("s12c_done_before_last_pull")
def s12c(w):
    c = Chk()
    w.all_pulls()
    lines = []
    for i, r in enumerate((A0, A1, R0, R1)):
        lines += X.log_run(w.art, r[0], r[1], w.sha[r])
        if i == 2:
            lines.append(X.done_line())
    X.write_log(w.b2, lines)
    rc, out = go(w)
    unfinished(c, w, out, rc, "no ZZV8FMINI-DONE")
    return c.fails


@scn("s12d_box2_stop_midrun")
def s12d(w):
    c = Chk()
    w.all_pulls()
    w.log(specs=[(a, s, {}) for a, s in (A0, A1)], done=False)
    w.ctl("b2stop:" + Zt(A1))
    rc, out = go(w)
    unfinished(c, w, out, rc, "wrote a ZZV8FMINI-STOP after this chain started")
    c.eq([x[0] for x in marks(w)["stage"]], ["1", "2"], "the two pulled runs were processed first")
    return c.fails


@scn("s12f_verify_pulls")
def s12f(w):
    c = Chk()
    w.all_pulls()
    w.log()
    w.ctl("mutate_other:" + Zt(R1))
    rc, out = go(w, LIT1_STUB_OTHER=f"{w.art}/{L.run_name(*A0)}/ckpt_last.pt")
    unfinished(c, w, out, rc, "is no longer COMPLETE")
    c.eq(len(starts(w)), 12, "all twelve steps ran (the change is only visible to the final re-verification)")
    return c.fails


# ================================================================================================================================ S13: tokens
@scn("s13a_reader_sha")
def s13a(w):
    c = Chk()
    w.all_pulls()
    w.log()
    open(f"{w.root}/reader_stub.py", "a").write("# edited after the tokens were bound\n")
    rc, out = go(w)
    unfinished(c, w, out, rc, "the tokens bind reader sha256")
    c.eq(starts(w), [], "no rollout")
    return c.fails


@scn("s13b_prereg_sha")
def s13b(w):
    c = Chk()
    w.all_pulls()
    w.log()
    open(f"{w.root}/PREREG.md", "a").write("edited after the tokens were bound\n")
    rc, out = go(w)
    unfinished(c, w, out, rc, "the tokens bind PREREG sha256")
    c.eq(starts(w), [], "no rollout")
    return c.fails


def _disagree(w, **kw):
    c = Chk()
    for r in (A0, R0, R1):
        w.pull(*r)
    w.pull(*A1, **kw)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "do not bind ONE commit")
    c.eq(starts(w), [], "no rollout")
    return c.fails


SC["s13c_commit"] = lambda w: _disagree(w, commit="d" * 40)
SC["s13c_reader"] = lambda w: _disagree(w, rsha="12" * 32)
SC["s13c_prereg"] = lambda w: _disagree(w, psha="34" * 32)


@scn("s13d_expect_commit")
def s13d(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, LIT1_EXPECT_COMMIT="e" * 40)
    unfinished(c, w, out, rc, "LIT1_EXPECT_COMMIT")
    c.eq(starts(w), [], "no rollout")
    return c.fails


def _pin(w, **env):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, **env)
    unfinished(c, w, out, rc, "(LIT1_")
    c.eq(starts(w), [], "no rollout")
    return c, out


@scn("s13f_prereg_pin")
def s13f(w):
    c, out = _pin(w, LIT1_EXPECT_PREREG_SHA="469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a")
    c.has(out, "the landed PREREG_V8F_LAUNCH_INTENT is 469bac1f", "a PREREG that is not the landed one is refused before anything is read")
    return c.fails


@scn("s13g_reader_pin")
def s13g(w):
    c, out = _pin(w, LIT1_EXPECT_READER_SHA="5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35")
    c.has(out, "the landed reader is 5e950dcf", "a reader that is not the landed one is refused before anything is read")
    return c.fails


@scn("s13h_pins_hold")
def s13h(w):
    c = Chk()
    w.all_pulls()
    w.log()
    rc, out = go(w, LIT1_EXPECT_PREREG_SHA=w.psha, LIT1_EXPECT_READER_SHA=w.rsha)
    c.eq(rc, 0, f"pins that equal the files let the chain run ({out[-200:]})")
    c.eq(len(marks(w)["done"]), 1, "DONE")
    return c.fails


@scn("s14f_landed_pins_real")
def s14f(w):
    """Real mode (no world), the world's STUB files with the REAL pins: both refusals, before PINS.txt is even read."""
    c = Chk()
    for what, env, why in (("PREREG", {"LIT1_EXPECT_PREREG_SHA": "469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a"}, "the landed PREREG_V8F_LAUNCH_INTENT is 469bac1f"),
                           ("reader", {"LIT1_EXPECT_READER_SHA": "5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35"}, "the landed reader is 5e950dcf")):
        rc, out = go(w, world=False, LIT1_TREE_LI="F:/TanitAD-scratch/launch_intent_t1/tree_li", timeout=400, **env)
        c.eq(rc, 1, f"[{what}] rc")
        c.has(out, why, f"[{what}] the stub file is not the landed one")
        c.no(out, "PINS.txt", f"[{what}] refused before the pins file is read")
    return c.fails


@scn("s13e_cross_resume")
def s13e(w):
    c = Chk()
    w.all_pulls()
    w.log(specs=[(a, s, {}) for a, s in (A0, A1)], done=False)
    rc, out = go(w)
    c.eq(rc, 1, "run 1 times out waiting for the other pulls")
    c.eq(len(starts(w)), 6, "run 1 processed the two pulled runs")
    c.eq(open(f"{w.out}/LAUNCH_COMMIT.txt").read().strip(), L.LAUNCH_COMMIT, "the commit is recorded")
    for r in (A0, A1, R0, R1):
        w.pull(*r, commit="d" * 40)
    w.log()
    rc, out = go(w)
    unfinished(c, w, out, rc, "one launch commit is required")
    return c.fails


# ================================================================================================================================ S14: real-mode refusals (no LIT1_WORLD)
@scn("s14a_sibling_logs")
def s14a(w):
    c = Chk()
    for name in ("chain_v8f_dagoaccorc.out", "chain_v8f_dagoaccorc_r2.out", "chain_v8f_dagoacc.out", "chain_v8f_dag2.out"):
        p = f"{w.root}/{name}"
        open(p, "w").write("")
        rc, out = go(w, world=False, LIT1_B2_OUT=p, timeout=300)
        c.eq(rc, 1, f"{name}: rc")
        c.has(out, "SIBLING chain's log", f"{name}: a predecessor's log is refused in a real run")
        c.no(out, "lit1_chain start", f"{name}: nothing started")
    return c.fails


@scn("s14b_world_bad")
def s14b(w):
    c = Chk()
    rc, out = go(w, LIT1_WORLD=f"{w.root}/no_such_world", timeout=300)
    c.eq(rc, 1, "rc")
    c.has(out, "ZZLIT1-STOP LIT1_WORLD=", "a LIT1_WORLD that is not a directory inside the scratch root is refused")
    c.no(out, "lit1_chain start", "nothing started")
    return c.fails


@scn("s14c_md5")
def s14c(w):
    c = Chk()
    fake = f"{w.root}/inputs_fake"
    os.makedirs(fake, exist_ok=True)
    rc, out = go(w, world=False, LIT1_INPUTS=fake, LIT1_TREE_LI="F:/TanitAD-scratch/launch_intent_t1/tree_li", timeout=300)
    m = marks(w)
    c.eq(rc, 1, f"rc ({out[-300:]})")
    c.has(" ".join(m["stop"]) + out, "md5 of", "an input whose md5 is not the registered one stops the chain")
    c.has(" ".join(m["stop"]) + out, "native_tracks.pt", "the first input that differs")
    c.eq(starts(w), [], "no rollout")
    return c.fails


def _real(w, **kw):
    return go(w, world=False, LIT1_TREE_LI="F:/TanitAD-scratch/launch_intent_t1/tree_li", timeout=400, **kw)


def _write_pins(w):
    return go(w, "--write-pins", world=False, LIT1_TREE_LI="F:/TanitAD-scratch/launch_intent_t1/tree_li", timeout=400)


PIN_TAMPERS = {"basefile": "basefile=", "ctrl": "ctrl=", "rep": "rep=", "compose": "compose=", "tmpl": "tmpl=", "lib": "lib=lit1_tree.py", "litree": "litree=", "basedigest": "basedigest="}


def _flip_pin(path, prefix):
    lines = open(path, encoding="utf-8").read().split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(prefix):
            m = re.search(r"([0-9a-f]{64})\s*$", ln) or re.search(r"basedigest=([0-9a-f]{64})", ln)
            assert m, ln
            h = m.group(1)
            lines[i] = ln.replace(h, ("0" if h[0] != "0" else "1") + h[1:], 1)
            break
    else:
        raise AssertionError(prefix)
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines))


def s14d(w, variants=("control",) + tuple(PIN_TAMPERS)):
    c = Chk()
    # a REAL-mode scenario: no pulls and no box-2 log exist, so a start that gets past a refusal can only WAIT (and time out), never roll anything out
    rc, out = _write_pins(w)
    c.eq(rc, 0, f"--write-pins ({out[-300:]})")
    pins = f"{w.H}/PINS.txt"
    c.ok(os.path.isfile(pins), "--write-pins wrote PINS.txt")
    if not os.path.isfile(pins):
        return c.fails
    good = open(pins, "rb").read()
    for v in variants:
        open(pins, "wb").write(good)
        if v == "control":
            shipped = f"{HERE}/PINS.txt"
            c.ok(os.path.isfile(shipped) and open(shipped, "rb").read() == good, "the shipped PINS.txt is CURRENT (equal to a fresh --write-pins on the shipped code and the real pinned inputs)")
            rc, out = _real(w)
            c.has(out, "PINS.txt verified", "[control] every pin recomputes")
            c.has(out, "template proof 1:", "[control] the template proofs ran in a real start")
            c.has(out, "template proofs 2-4:", "[control] proofs 2-4 ran")
            c.no(out, "PIN:", "[control] no pin differs")
            c.no(out, "TEMPLATE PROOF FAILED", "[control] the templates reproduce the commands that ran")
        else:
            _flip_pin(pins, PIN_TAMPERS[v])
            rc, out = _real(w)
            c.eq(rc, 1, f"[{v}] rc")
            c.has(out, "PIN:", f"[{v}] the differing pin is named")
            c.has(out, "PINS.txt: 1 difference(s)", f"[{v}] the chain stops on ONE differing pin")
            c.no(out, "template proof 1:", f"[{v}] nothing past the pin check ran")
    open(pins, "wb").write(good)
    return c.fails


SC["s14d_pins"] = s14d
for _v in PIN_TAMPERS:
    SC[f"s14d_pin_{_v}"] = (lambda w, _v=_v: s14d(w, variants=(_v,)))


def s14e(w, which):
    c = Chk()
    f = {"causal": "rollout_causal.tmpl", "logged": "rollout_logged.tmpl", "t0fresh": "t0fresh.tmpl"}[which]
    open(f"{w.H}/argv/{f}", "a", newline="\n").write("--zzz-drift\n")
    rc, out = go(w, "--write-pins", world=False, LIT1_TREE_LI="F:/TanitAD-scratch/launch_intent_t1/tree_li", timeout=400)
    c.eq(rc, 0, "--write-pins on the drifted templates")
    rc, out = _real(w)
    c.eq(rc, 1, f"[{which}] rc")
    c.has(out, "TEMPLATE PROOF FAILED", f"[{which}] a template that no longer reproduces the command that ran stops the chain")
    c.has(out, {"causal": "rollout_causal.tmpl rendered for DAG-OACC s0", "logged": "logged template differs", "t0fresh": "t0fresh template differs"}[which], f"[{which}] the proof that caught it")
    return c.fails


for _k in ("causal", "logged", "t0fresh"):
    SC[f"s14e_tmpl_{_k}"] = (lambda w, _k=_k: s14e(w, _k))


# ================================================================================================================================ S15: the waker
@scn("s15_waker")
def s15(w):
    c = Chk()
    os.makedirs(w.logs, exist_ok=True)
    out_p = f"{w.logs}/chain.out"

    def wake(args="", **env):
        e = {"LIT1_LOG": w.logs, "LIT1_WAKE_POLL_S": "1", "LIT1_WAKE_MAX_POLLS": "2", "LIT1_B2_OUT": w.b2, "LIT1_ART": w.art}
        e.update(env)
        return S.run(f'bash "{w.H}/wake_lit1.sh" {args} < /dev/null', env=e, timeout=120)

    def put(text):
        open(out_p, "w", newline="\n").write(text)
        for f in ("chain.winpid",):
            if os.path.exists(f"{w.logs}/{f}"):
                os.remove(f"{w.logs}/{f}")
    put("10-28 12:00 start\nZZLIT1-STAGE-DONE 4/4 x (A_0): causal\nZZLIT1-DONE 12/12 steps, reader rc 0 (outcome row written): x\n")
    rc, o = wake()
    c.eq(rc, 0, "a DONE marker -> exit 0")
    c.has(o, "CHAIN DONE", "the DONE text")
    put("10-28 12:00 start\nZZLIT1-STOP a refusal\n")
    rc, o = wake()
    c.eq(rc, 4, "a STOP marker -> exit 4")
    c.has(o, "CHAIN STOP", "the STOP text")
    put("ZZLIT1-DONE 12/12 steps\nZZLIT1-STOP late refusal\n")
    rc, o = wake()
    c.eq(rc, 4, "STOP wins over DONE")
    put("10-28 12:00 ZZLIT1-DONE 12/12 steps, a prose line that merely CONTAINS the marker\n10-28 12:01 STOP: ZZLIT1-STOP in prose\n")
    rc, o = wake()
    c.eq(rc, 5, "markers are ANCHORED at line start: prose containing them is not a marker (and the waker times out)")
    c.has(o, "WAKER TIMEOUT", "the timeout text")
    put("10-28 12:00 running\n")
    open(f"{w.logs}/chain.winpid", "w").write("2147480000\n")
    rc, o = wake()
    c.eq(rc, 6, "the chain's process is gone and no marker was written -> exit 6")
    c.has(o, "CHAIN DEAD", "the dead-chain text")
    put("10-28 12:00 running\n")
    rc, o = wake()
    c.eq(rc, 5, "nothing and no pid file -> timeout 5")
    w.all_pulls()
    w.log()
    rc, o = wake("--status")
    c.eq(rc, 1, "--status of a chain with no marker and no pid file is 'still running' (1)")
    c.has(o, "CHAIN RUNNING", "the status text")
    c.has(o, "M2b_s0_launchintent (A_0): PASS", "--status prints the box-2 log's per-run state")
    return c.fails


# ================================================================================================================================ S16: the real dry run (reads the real pinned inputs; starts and writes nothing)
@scn("s16_dry_run")
def s16(w):
    c = Chk()
    rc, out = S.run(f'bash "{HERE}/lit1_chain.sh" --dry-run < /dev/null', env={}, timeout=600)
    c.eq(rc, 0, f"dry-run rc ({out[-300:]})")
    c.has(out, "ZZLIT1-DRY-OK", "the dry run's end marker")
    c.eq(re.findall(r"^ZZLIT1-STOP.*$", out, re.M), [], "no STOP marker (the marker legend line merely names it)")
    c.eq(len(re.findall(r"parse: PARSE-OK", out)), 12, "every one of the 12 rollout argv parses with the pinned tool's own argparse")
    c.no(out, "PARSE-FAIL", "no argv failed to parse")
    c.has(out, "PINS.txt verified", "pins")
    c.has(out, "launch_intent_read.py: present, sha256 5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35", "the LANDED reader (sha256 5e950dcf...) is the one the dry run reads")
    c.has(out, "landed/PREREG_V8F_LAUNCH_INTENT.md", "the LANDED PREREG copy is the default PREREG")
    c.has(out, "sha256 469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a", "the LANDED PREREG pin (469bac1f...)")
    c.has(out, "reader --static-check: LAUNCH-INTENT-READ-STATIC-OK", "the reader's own static check passes on the landed PREREG with its pin")
    c.has(out, "template proofs 2-4", "template proofs")
    c.has(out, "cfg reach of the PINNED tree: CFG-REJECTED", "the measured reason for the one-file overlay tree")
    c.has(out, "cfg reach of the LAUNCH-INTENT tree: CFG-ACCEPTED", "the overlay tree accepts the launch cfg")
    w.dry = out
    return c.fails


# ================================================================================================================================ mutants of the shipped shell code (id, file, old, new, scenario)
Q = '"'
B = "\\"      # one backslash
MUT_SH = [
    ("C-guard-ram", "lit1_chain.sh", '--ram-min-mb "$RAMMIN" --disk "$OUTDRIVE" --disk-min-gb "$FMIN" 2>&1', '--ram-min-mb 0 --disk "$OUTDRIVE" --disk-min-gb "$FMIN" 2>&1', "s10a_ram_low"),
    ("C-guard-disk", "lit1_chain.sh", '--ram-min-mb "$RAMMIN" --disk "$OUTDRIVE" --disk-min-gb "$FMIN" 2>&1', '--ram-min-mb "$RAMMIN" --disk "$OUTDRIVE" --disk-min-gb 0 2>&1', "s10b_disk_low"),
    ("C-guard-t1", "lit1_chain.sh", '  guards_wait "$tag"\n  nostop; pin_check "before $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"\n  mkdir -p',
     '  nostop; pin_check "before $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"\n  mkdir -p', "s10c_ram_drops_before_logged"),
    ("C-guard-t0", "lit1_chain.sh", '  guards_wait "$tag"\n  nostop; pin_check "before $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"\n  t0=$(date +%s)',
     '  nostop; pin_check "before $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"\n  t0=$(date +%s)', "s10d_ram_drops_before_t0"),
    ("C-pin-check", "lit1_chain.sh", 'pin_check() { [ "$(tree_digest)" = "$TOOLD" ] || stop "the tool tree changed ($1)"; }', "pin_check() { true; }", "s4b_tree_mutated"),
    ("C-pin-after-t1", "lit1_chain.sh", '  pin_check "after $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ]',
     '  : "after $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ]', "s4b_tree_mutated"),
    ("C-pin-after-t0", "lit1_chain.sh", '  pin_check "after $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ]',
     '  : "after $tag"\n  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ]', "s4d_tree_mutated_t0"),
    ("C-ckpt-after-t1", "lit1_chain.sh", '[ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ]', 'true\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ]', "s4_ckpt_mutated"),
    ("C-ckpt-after-t0", "lit1_chain.sh", '[ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ]', 'true\n  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ]', "s4c_ckpt_mutated_t0"),
    ("C-rc-t1", "lit1_chain.sh", '{ [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ]', '{ [ -s "$OUTu/t1v8f_$tag.json" ]', "s3_resume"),
    ("C-rc-t0", "lit1_chain.sh", '{ [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ]', '{ [ -s "$OUTu/t0fresh/eval_$tag.json" ]', "s3f_t0_fail"),
    ("C-dumps-post", "lit1_chain.sh", '[ -s "$OUTu/ticks/ticks_$tag.npz" ] && [ "$(n_dumps "$tag")" -ge "$NDUMP" ]; } || stop "rollout', '[ -s "$OUTu/ticks/ticks_$tag.npz" ]; } || stop "rollout', "s3g_few_dumps"),
    ("D-t1-json", "lit1_chain.sh", '[ -s "$OUTu/t1v8f_$1.json" ] && ', "", "s3_tamper_r2"),
    ("D-t1-ticks", "lit1_chain.sh", '[ -s "$OUTu/ticks/ticks_$1.npz" ] && ', "", "s3_tamper_r2"),
    ("D-t1-argv", "lit1_chain.sh", '[ -s "$OUTu/argv/$1.argv" ] || return 1', "true || return 1", "s3_tamper_r2"),
    ("D-t1-dumps", "lit1_chain.sh", '  [ "$(n_dumps "$1")" -ge "$NDUMP" ] || return 1', "  true", "s3_tamper_r1"),
    ("D-t1-ck", "lit1_chain.sh", 'grep -q "' + B + '"ckpt_sha256' + B + '": ' + B + '"$2' + B + '"" "$OUTu/done_$1.txt" && ', "", "s3_tamper_r1"),
    ("D-t1-before", "lit1_chain.sh", 'grep -q "' + B + '"ckpt_sha256_before' + B + '": ' + B + '"$2' + B + '"" "$OUTu/done_$1.txt" && ', "", "s3_tamper_r1"),
    ("D-t1-after", "lit1_chain.sh", 'grep -q "' + B + '"ckpt_sha256_after' + B + '": ' + B + '"$2' + B + '"" "$OUTu/done_$1.txt" && ', "", "s3_tamper_r1"),
    ("D-t1-tool", "lit1_chain.sh", 'grep -q "' + B + '"tool_digest' + B + '": ' + B + '"$TOOLD' + B + '"" "$OUTu/done_$1.txt" && ', "", "s3_tamper_r1"),
    ("D-t1-cmd", "lit1_chain.sh", ' && grep -q "' + B + '"cmd_sha256' + B + '": ' + B + '"$3' + B + '"" "$OUTu/done_$1.txt"', "", "s3_tamper_r1"),
    ("D-t0-json", "lit1_chain.sh", 'if [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ]', 'if [ -s "$OUTu/t0fresh/eval_$tag.npz" ]', "s3_tamper_r2"),
    ("D-t0-npz", "lit1_chain.sh", '[ -s "$OUTu/t0fresh/eval_$tag.npz" ] && [ -s "$OUTu/done_t0f_$tag.txt" ]', '[ -s "$OUTu/done_t0f_$tag.txt" ]', "s3_tamper_r2"),
    ("D-t0-ck", "lit1_chain.sh", 'grep -q "' + B + '"ckpt_sha256' + B + '": ' + B + '"$sha' + B + '"" "$OUTu/done_t0f_$tag.txt" && grep -q "' + B + '"ckpt_sha256_before', 'grep -q "' + B + '"ckpt_sha256_before', "s3_tamper_r1"),
    ("D-t0-before", "lit1_chain.sh", '&& grep -q "' + B + '"ckpt_sha256_before' + B + '": ' + B + '"$sha' + B + '"" "$OUTu/done_t0f_$tag.txt" ' + B, B, "s3_tamper_r1"),
    ("D-t0-after", "lit1_chain.sh", 'grep -q "' + B + '"ckpt_sha256_after' + B + '": ' + B + '"$sha' + B + '"" "$OUTu/done_t0f_$tag.txt" && ', "", "s3_tamper_r1"),
    ("D-t0-tool", "lit1_chain.sh", 'grep -q "' + B + '"tool_digest' + B + '": ' + B + '"$TOOLD' + B + '"" "$OUTu/done_t0f_$tag.txt" && grep -q "' + B + '"cmd_sha256', 'grep -q "' + B + '"cmd_sha256', "s3_tamper_r1"),
    ("D-t0-cmd", "lit1_chain.sh", ' && grep -q "' + B + '"cmd_sha256' + B + '": ' + B + '"$cs' + B + '"" "$OUTu/done_t0f_$tag.txt"; then', "; then", "s3_tamper_r2"),
    ("D-aside-mv", "lit1_chain.sh", '[ -e "$f" ] && mv "$f" "$w/"', '[ -e "$f" ] && rm -rf "$f"', "s3_resume"),
    ("E-b2done", "lit1_chain.sh", 'say "== all $n steps done; waiting for the box-2 DONE marker"\nb2_done_wait\nverify_pulls', 'say "== all $n steps done; waiting for the box-2 DONE marker"\nverify_pulls', "s12b_no_box2_done"),
    ("E-verify-pulls", "lit1_chain.sh", "b2_done_wait\nverify_pulls\n", "b2_done_wait\n", "s12f_verify_pulls"),
    ("E-b2stop", "lit1_chain.sh", '[ "$(b2_stops)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP after', 'false && stop "the box-2 chain wrote a ZZV8FMINI-STOP after', "s12d_box2_stop_midrun"),
    ("E-manifest-stop", "lit1_chain.sh", '|| stop "manifest: $(tr', '|| : "manifest: $(tr', "s3h_manifest_refused"),
    ("E-reader-rc", "lit1_chain.sh", '< /dev/null; RRC=$?', '< /dev/null; RRC=0', "s12a_reader_rc2"),
    ("E-msg-rc2", "lit1_chain.sh", "2) echo 'REFUSED, nothing written'", "2) echo 'REFUSED'", "s12a_reader_rc2"),
    ("E-msg-rc4", "lit1_chain.sh", "4) echo 'BLOCKED, a required slot incomplete'", "4) echo 'BLOCKED'", "s12a_reader_rc4"),
    ("E-stage-marker", "lit1_chain.sh", 'echo "ZZLIT1-STAGE-DONE $K/', 'echo "ZZLIT1-STAGE $K/', "s1_happy"),
    ("G-wrong", "lit1_chain.sh", '[ $prc -eq 3 ] && stop "the pulled $g is WRONG', 'false && stop "the pulled $g is WRONG', "s5_wrong_lever_third"),
    ("G-next-fail", "lit1_chain.sh", '''[ $rc -eq 3 ] && stop "the box-2 chain's record FAILED a control''', '''false && stop "the box-2 chain's record FAILED a control''', "s6_kshare_fail_line"),
    ("G-expect-sha", "lit1_chain.sh", '--expect-sha "$lsha" --min-age "$MINAGE"', '--min-age "$MINAGE"', "s8_log_sha"),
    ("G-commits", "lit1_chain.sh", '|| stop "$g: the pulled launch-gate tokens do not bind ONE commit', '|| : "$g: the pulled launch-gate tokens do not bind ONE commit', "s13c_commit"),
    ("G-reader-sha", "lit1_chain.sh", '[ "$r" = "$RSHA" ] || stop', "true || stop", "s13a_reader_sha"),
    ("G-prereg-sha", "lit1_chain.sh", '[ "$p" = "$PSHA" ] || stop', "true || stop", "s13b_prereg_sha"),
    ("G-expect-commit", "lit1_chain.sh", '[ -z "${LIT1_EXPECT_COMMIT:-}" ] || [ "$c" = "$LIT1_EXPECT_COMMIT" ] || stop', "true || stop", "s13d_expect_commit"),
    ("G-launch-seen", "lit1_chain.sh", '[ "$c" = "$LAUNCH_SEEN" ] || stop', "true || stop", "s13e_cross_resume"),
    ("H-sib-1", "lit1_chain.sh", "chain_v8f_dagoaccorc.out|", "", "s14a_sibling_logs"),
    ("H-sib-2", "lit1_chain.sh", "chain_v8f_dagoaccorc_r2.out|", "", "s14a_sibling_logs"),
    ("H-sib-3", "lit1_chain.sh", "chain_v8f_dagoacc.out|", "", "s14a_sibling_logs"),
    ("H-sib-4", "lit1_chain.sh", "|chain_v8f_dag2.out)", ")", "s14a_sibling_logs"),
    ("H-world", "lit1_chain.sh", '''|| { echo "ZZLIT1-STOP LIT1_WORLD=$WORLD is not a directory inside F:/TanitAD-scratch/launch_intent_t1: refused, nothing was run"; exit 1; }''', "|| true", "s14b_world_bad"),
    ("H-md5", "lit1_chain.sh", '= "$2" ] || stop "md5 of $1 is not $2"', '= "$2" ] || true', "s14c_md5"),
    ("H-pins-stop", "lit1_chain.sh", '[ $bad -eq 0 ] || stop "PINS.txt: $bad difference(s) (listed above): nothing was started"', "[ $bad -eq 0 ] || true", "s14d_pin_lib"),
    ("H-pin-basefile", "lit1_chain.sh", '''basefile=*) got=$(fsha "$BASE/${k#basefile=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: ${k#basefile=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "basefile=*) ;;", "s14d_pin_basefile"),
    ("H-pin-ctrl", "lit1_chain.sh", '''ctrl=*) got=$(fsha "$AR/${k#ctrl=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: banked CTRL file ${k#ctrl=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "ctrl=*) ;;", "s14d_pin_ctrl"),
    ("H-pin-rep", "lit1_chain.sh", '''rep=*) got=$(fsha "${k#rep=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: banked REP file ${k#rep=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "rep=*) ;;", "s14d_pin_rep"),
    ("H-pin-compose", "lit1_chain.sh", '''compose=*) got=$(fsha "${k#compose=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: the DAG-ORC K-BUILD record is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "compose=*) ;;", "s14d_pin_compose"),
    ("H-pin-tmpl", "lit1_chain.sh", '''tmpl=*) got=$(fsha "$H/argv/${k#tmpl=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: argv template ${k#tmpl=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "tmpl=*) ;;", "s14d_pin_tmpl"),
    ("H-pin-lib", "lit1_chain.sh", '''lib=*) got=$(fsha "$H/${k#lib=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: ${k#lib=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "lib=*) ;;", "s14d_pin_lib"),
    ("H-pin-litree", "lit1_chain.sh", '''litree=*) [ "$a" = "$TOOLD" ] || { say "PIN: LAUNCH-INTENT tree digest ${TOOLD:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;''', "litree=*) ;;", "s14d_pin_litree"),
    ("H-tmpl-want", "lit1_chain.sh", '[ "$want" = "$got" ] || stop', "true || stop", "s14e_tmpl_causal"),
    ("H-tmpl-logged", "lit1_chain.sh", '''stop "TEMPLATE PROOF FAILED: logged template differs from the command that RAN for REP s0"''', "true", "s14e_tmpl_logged"),
    ("H-tmpl-t0", "lit1_chain.sh", '''stop "TEMPLATE PROOF FAILED: t0fresh template differs from the command that RAN for REP s0"''', "true", "s14e_tmpl_t0fresh"),
    ("H-lock", "lit1_chain.sh", 'if ! mkdir "$LOGu/chain.lock" 2>/dev/null; then', "if false; then", "s11_lock"),
    ("H-c-gate", "lit1_chain.sh", '[ "$CF0" -ge "$CMIN" ] || stop', "true || stop", "s10e_c_low"),
    ("G-pin-prereg", "lit1_chain.sh", '[ -z "$EXP_P" ] || [ "$PSHA" = "$EXP_P" ] || stop', "true || stop", "s13f_prereg_pin"),
    ("G-pin-reader", "lit1_chain.sh", '[ -z "$EXP_R" ] || [ "$RSHA" = "$EXP_R" ] || stop', "true || stop", "s13g_reader_pin"),
    ("G-pin-prereg-real", "lit1_chain.sh", '[ -z "$EXP_P" ] || [ "$PSHA" = "$EXP_P" ] || stop', "true || stop", "s14f_landed_pins_real"),
    ("G-pin-none", "lit1_chain.sh", '[ "$EXP_P" != none ] || EXP_P=""; [ "$EXP_R" != none ] || EXP_R=""', "true", "s14d_pin_lib"),
    ("W-anchor-stop", "wake_lit1.sh", "if grep -aq '^ZZLIT1-STOP' \"$O\"", "if grep -aq 'ZZLIT1-STOP' \"$O\"", "s15_waker"),
    ("W-anchor-done", "wake_lit1.sh", "if grep -aq '^ZZLIT1-DONE' \"$O\"", "if grep -aq 'ZZLIT1-DONE' \"$O\"", "s15_waker"),
    ("W-stop-first", "wake_lit1.sh", "return 4; fi", "return 3; fi", "s15_waker"),
    ("W-dead-pid", "wake_lit1.sh", '&& ! tasklist', '&& tasklist', "s15_waker"),
    ("W-timeout-rc", "wake_lit1.sh", "exit 5", "exit 7", "s15_waker"),
    ("W-done-rc", "wake_lit1.sh", "return 0; fi\n  local pid", "return 9; fi\n  local pid", "s15_waker"),
]
# verify_manifest mutants: killed by the manifest verification battery on a finished world (lit1_tree.py edited in the battery's own copy)
MUT_VERIFY = [
    ("V-dumps", "lit1_tree.py", "if nd < n_dump_min:", "if False:", "dump"),
    ("V-done-after", "lit1_tree.py", ' or dr.get("ckpt_sha256_after") != sha', "", "done-after"),
    ("V-done-ck", "lit1_tree.py", 'if dr.get("ckpt_sha256") != sha or dr.get("ckpt_sha256_before") != sha', 'if dr.get("ckpt_sha256_before") != sha', "done-ck"),
    ("V-done-before", "lit1_tree.py", ' or dr.get("ckpt_sha256_before") != sha', "", "done-before"),
    ("V-done-smoke", "lit1_tree.py", ' or dr.get("smoke") is not False:\n                    bad.append', ':\n                    bad.append', "done-smoke"),
    ("V-done-tool", "lit1_tree.py", 'if tool_digest and dr.get("tool_digest") != tool_digest:', "if False:", "done-tool"),
    ("V-t0f-tool", "lit1_tree.py", 'if dr.get("ckpt_sha256") != sha or (tool_digest and dr.get("tool_digest") != tool_digest) or dr.get("smoke") is not False:', 'if dr.get("ckpt_sha256") != sha or dr.get("smoke") is not False:', "t0f-tool"),
    ("V-t0f-ck", "lit1_tree.py", 'if dr.get("ckpt_sha256") != sha or (tool_digest', 'if (tool_digest', "t0f-ck"),
    ("V-t0f-smoke", "lit1_tree.py", ' or dr.get("smoke") is not False:\n                bad.append(f"{lab}: the fresh', ':\n                bad.append(f"{lab}: the fresh', "t0f-smoke"),
    ("V-ages", "lit1_tree.py", 'if jj.get("ages") != want or L.tail2(str(jj.get("ckpt", ""))) != e["ckpt_tail"]:', 'if L.tail2(str(jj.get("ckpt", ""))) != e["ckpt_tail"]:', "t0f-ages"),
    ("V-t0-ckpt", "lit1_tree.py", 'if jj.get("ages") != want or L.tail2(str(jj.get("ckpt", ""))) != e["ckpt_tail"]:', 'if jj.get("ages") != want:', "t0f-ckpt"),
    ("V-limit", "lit1_tree.py", 'or j.get("limit") != want_limit or', "or", "limit"),
    ("V-max-windows", "lit1_tree.py", 'if j.get("max_windows") is not None or', "if", "max-windows"),
    ("V-n-windows", "lit1_tree.py", ' or j.get("n_windows") != L.N_WINDOWS:', ":", "n-windows"),
    ("V-t1-ckpt", "lit1_tree.py", 'if L.tail2(j.get("ckpt", "")) != e["ckpt_tail"]:', "if False:", "t1-ckpt"),
    ("V-ctrl", "lit1_tree.py", "if not os.path.isfile(ck) or L.sha256_file(ck) != want[seed]:", "if not os.path.isfile(ck):", "ctrl-swap"),
    ("V-dup", "lit1_tree.py", "if sha in seen:", "if False:", "dup-ckpt"),
    ("V-empty", "lit1_tree.py", "or os.path.getsize(p) == 0:", ":", "empty-argv"),
    ("V-need-files", "lit1_tree.py", 'for f in ("argv.json", "summary.json", "selfcheck.json"):\n            need(f"{d}/{f}", f"{lab} {f}")', "pass", "missing-selfcheck"),
    ("V-ticks", "lit1_tree.py", 'for k in ("ticks", "json", "done", "argv"):', 'for k in ("json", "done", "argv"):', "missing-ticks"),
    ("V-t0f-done-need", "lit1_tree.py", 'need(dn, f"{lab} fresh-T0 done record")', "pass", "missing-t0f-done"),
    ("V-digest-param", "lit1_tree.py", "bad = verify_manifest(L.jload(a.manifest), a.n_dump_min, a.tool_digest)", "bad = verify_manifest(L.jload(a.manifest), a.n_dump_min, None)", "digest-param"),
]
