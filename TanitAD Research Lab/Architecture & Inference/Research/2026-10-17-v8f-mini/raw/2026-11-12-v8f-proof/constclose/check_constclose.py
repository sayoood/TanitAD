"""check_constclose.py -- pod check of card ADD80 (CONSTCLOSE10K), control k1: the constant reaches the rule. One small rollout of form RRe (wrapper v8f_t1_eval_reader_rule_rzgfhqe.py, hash a747abeb51ef041f)
with the constant file const_C20_H1.npz (2.0 m/s, history one frame) on the first --n-clips clips of the pinned grid (all their windows), fresh (no pre-roll), model seed 0, readout K (--readout causal_g
--acc-gain 0 --path-exec --kappa-max 0.2), --rule-every 5, q50 map, reader seed 0. The command is the one of check_estclose.py (run_cmd). Checks on the two logs of the run (replan log rule_log_<name>.npz and
mid-look log rule_log_mid_<name>.npz; both hold ci, t0, j, k, present, c_src, c_est, est_ok, v_l, v_L):
    K1a  the closing column c_src is 2.0 on every row that is present (lead read present) and est_ok, in both logs; and on the present rows without est_ok it is 0.0
    K1b  c_est is 2.0 on every est_ok row of both logs (replan and second look) and NaN on the others
    K1c  no look at clip frame index j >= 4 is marked short (est_ok False): a full history needs the frame j-4..j, so from the fifth frame of a clip on the count of short looks is zero; looks at j < 4 are counted
    K1d  v_L = max(0, v_l + c_src) on the present rows
    L1   the run line "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>": the counts equal those recomputed from the two logs (preroll 0 0) and the sha256 equals that of the file
    F0   the constant file loaded by closest.py returns 2.0 for random inputs of shape [1, 3] and [n, 1, 3] and needs 5 frames (control k0 on the file the rollout used)
    W0   the wrapper has the hash of the card
  Nothing here removes, moves or stops anything; the rollout writes only inside --workdir (refused if it exists, except with --analyze-only, which only reads the run folder already there).
Usage on the pod:
  python check_constclose.py --new WRAPPER.py --tree TREE --ckpt CKPT --tracks native_tracks.pt --eval-cache DIR --eval-join J.npz --grid PINNED_GRID.json --reader READER.pt --qmap MAP.json
         --workdir NEWDIR --distrule-dir DIR --closest-dir DIR --const-file const_C20_H1.npz [--python PY] [--device cuda] [--n-clips 3] [--dry] [--analyze-only]
  python check_constclose.py --selftest
Last line: ZZ CHECK-CONSTCLOSE PASS <counts>  or  ZZ CHECK-CONSTCLOSE FAIL <names>"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALUE = 2.0
WRAP_SHA = "a747abeb51ef041f"
FULL_FROM = 4                    # H 1 + reader history 5 - 1 = 5 frames: j - 4 .. j, so j >= 4 has a full history
LINE = re.compile(r"^est looks scored full (\d+) short (\d+) preroll full (\d+) short (\d+) file ([0-9a-f]{64})$")
RUNNAME = "k1_c20"


# ============================================================================================ pure checks (numpy only)
def _b(z, k):
    return np.asarray(z[k]).astype(bool)


def check_a(z, value=VALUE):
    """K1a on one log -> (ok, detail)"""
    pr, eo, cs = _b(z, "present"), _b(z, "est_ok"), np.asarray(z["c_src"], np.float64)
    full, short = pr & eo, pr & ~eo
    ok = bool(full.any() and np.all(cs[full] == value) and np.all(cs[short] == 0.0))
    return ok, {"present_full": int(full.sum()), "present_short": int(short.sum()), "closing_not_c": int((cs[full] != value).sum()), "short_not_zero": int((cs[short] != 0.0).sum())}


def check_b(z, value=VALUE):
    eo, ce = _b(z, "est_ok"), np.asarray(z["c_est"], np.float64)
    ok = bool(eo.any() and np.all(ce[eo] == value) and np.all(np.isnan(ce[~eo])))
    return ok, {"est_ok_rows": int(eo.sum()), "c_est_not_c": int((ce[eo] != value).sum()), "not_nan_where_short": int((~np.isnan(ce[~eo])).sum())}


def check_c(z):
    eo, j = _b(z, "est_ok"), np.asarray(z["j"]).astype(np.int64)
    late_short = (~eo) & (j >= FULL_FROM)
    return bool(len(j) and not late_short.any() and (eo & (j >= FULL_FROM)).any()), {"short_at_frame_4_or_later": int(late_short.sum()), "short_before_frame_4": int(((~eo) & (j < FULL_FROM)).sum()), "rows": int(len(j))}


def check_d(z):
    pr = _b(z, "present")
    ok = bool(pr.any() and np.allclose(np.asarray(z["v_L"], np.float64)[pr], np.maximum(0.0, np.asarray(z["v_l"], np.float64)[pr] + np.asarray(z["c_src"], np.float64)[pr])))
    return ok, {"present_rows": int(pr.sum())}


def check_line(lines, RL, ML, sha, preroll=(0, 0)):
    """the run line -> (ok, detail); counts recomputed from the est_ok columns of the two logs"""
    full = int(_b(RL, "est_ok").sum() + _b(ML, "est_ok").sum())
    short = int((~_b(RL, "est_ok")).sum() + (~_b(ML, "est_ok")).sum())
    got = [x for x in lines if x.startswith("est looks")]
    m = LINE.match(got[0]) if len(got) == 1 else None
    ok = bool(m and int(m.group(1)) == full and int(m.group(2)) == short and (int(m.group(3)), int(m.group(4))) == tuple(preroll) and m.group(5) == sha)
    return ok, {"lines": len(got), "full_recomputed": full, "short_recomputed": short, "line": got[0] if got else None, "sha256_expected": sha}


def run_checks(RL, ML, lines, sha):
    """-> list of (name, ok, detail) for K1a, K1b, K1c, K1d (both logs each) and L1"""
    out = []
    for nm, fn in (("K1a closing column is 2.0 at every present look with est_ok", check_a), ("K1b c_est is 2.0 on every est_ok row", check_b), ("K1c no short look at frame index 4 or later", check_c),
                   ("K1d v_L = max(0, v_l + closing)", check_d)):
        for tag, z in (("replan log", RL), ("mid-look log", ML)):
            ok, det = fn(z)
            out.append(("%s (%s)" % (nm, tag), ok, json.dumps(det)))
    ok, det = check_line(lines, RL, ML, sha)
    out.append(("L1 run line counts equal the counts recomputed from the logs, file sha256 equals the constant file", ok, json.dumps(det)))
    return out


# ============================================================================================ pod part
def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--new", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqe.py"))
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    for k in ("tree", "ckpt", "tracks", "eval-cache", "eval-join", "grid", "reader", "qmap", "workdir", "const-file"):
        ap.add_argument("--" + k)
    ap.add_argument("--distrule-dir", default="")
    ap.add_argument("--closest-dir", default="")
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    A = ap.parse_args(argv)
    if A.selftest:
        return selftest()
    failed, counts = [], {}

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
        if not cond:
            failed.append(name.split(" ")[0])
    for k in ("tree", "ckpt", "tracks", "eval_cache", "eval_join", "grid", "reader", "qmap", "workdir", "const_file"):
        if not getattr(A, k):
            print("ZZ CHECK-CONSTCLOSE STOP missing --" + k.replace("_", "-"))
            return 3
    W, M = A.workdir, A.replan_steps
    closest_dir = A.closest_dir or os.path.dirname(os.path.abspath(A.new))
    distrule = A.distrule_dir or os.path.dirname(os.path.abspath(A.new))
    wsha = hashlib.sha256(open(A.new, "rb").read()).hexdigest()[:16]
    print("wrapper", A.new, wsha, flush=True)
    check("W0 the wrapper has the hash of the card (%s)" % WRAP_SHA, wsha == WRAP_SHA, wsha)
    if os.path.exists(W) and not A.analyze_only:
        print("ZZ CHECK-CONSTCLOSE STOP OUTPUT-EXISTS " + W)
        return 3
    sys.path.insert(0, closest_dir)
    import closest as CL                                                                      # noqa: E402
    rng = np.random.default_rng(0)
    try:
        E = CL.ClosingEstimator.load(A.const_file)
        vals = [E.apply(rng.normal(0, 9.0, (1, 3)))[0], *E.apply(rng.normal(0, 9.0, (16, 1, 3)))]
        check("F0 the constant file returns %.1f for inputs [1, 3] and [n, 1, 3], H %d, needs %d frames" % (VALUE, E.H, E.need_frames()), E.H == 1 and E.need_frames() == 5 and all(v == VALUE for v in vals))
    except Exception as ex:                                                                   # noqa: BLE001
        check("F0 the constant file loads", False, "%s: %s" % (type(ex).__name__, ex))
    sha = sha_file(A.const_file)
    if not A.dry and not os.path.exists(W):
        os.makedirs(W)
    sub = os.path.join(W, "grid_subset.json")
    if not A.dry and not os.path.exists(sub):
        g = json.load(open(A.grid, encoding="utf-8"))
        order = list(g["order"])[:A.n_clips]
        g2 = dict(g)
        g2["order"], g2["windows"] = order, {c: g["windows"][c] for c in order}
        g2["n_windows"] = int(sum(len(g["windows"][c]) for c in order))
        g2["constclose_subset"] = {"n_clips": len(order), "source": os.path.basename(A.grid)}
        with open(sub, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(g2, fh)
    out = os.path.join(W, RUNNAME)
    tag = RUNNAME + "_m%d_causal_heads" % M
    cmd = [A.python, A.new, "--tree", A.tree, "--form", "RRe", "--reader", A.reader, "--qmap", A.qmap, "--closing-est", A.const_file,
           "--rule-log", os.path.join(out, "ticks", "rule_log_%s.npz" % RUNNAME), "--rule-every", "5", "--readout", "causal_g", "--readout-log", os.path.join(out, "ticks", "readout_log_%s.json" % RUNNAME),
           "--acc-gain", "0", "--path-exec", "--kappa-max", "0.2", "--preroll-steps", "0",
           "run", "--ckpt", A.ckpt, "--tag", tag, "--tracks", A.tracks, "--eval-cache", A.eval_cache, "--eval-grid", sub, "--eval-join", A.eval_join, "--out-dir", out, "--replan-steps", str(M),
           "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze", "--device", A.device]
    if shutil.which("timeout"):
        cmd = ["timeout", "1800"] + cmd
    env = dict(os.environ, DISTRULE_DIR=distrule, CLOSEST_DIR=closest_dir, PYTHONPATH=os.path.join(A.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    if A.dry:
        print("CMD: " + " ".join(cmd))
        print("ZZ CHECK-CONSTCLOSE dry run: nothing executed")
        return 0
    if os.path.exists(out):
        if not A.analyze_only:
            raise SystemExit("OUTPUT-EXISTS " + out)
    else:
        os.makedirs(os.path.join(out, "ticks"))
        with open(os.path.join(out, "ticks", "run_%s.txt" % RUNNAME), "w", encoding="utf-8") as fh:
            fh.write("arm %s form RRe closing-est %s\n" % (RUNNAME, A.const_file))
        with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print("RUN %s rc=%d" % (RUNNAME, rc), flush=True)
        if rc != 0:
            print("ZZ CHECK-CONSTCLOSE FAIL run-rc-%d (see %s)" % (rc, os.path.join(out, "t1.log")))
            return 1
    tk = os.path.join(out, "ticks")

    def load(stem):
        p = os.path.join(tk, stem % RUNNAME)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    try:
        RL, ML = load("rule_log_%s.npz"), load("rule_log_mid_%s.npz")
        lines = open(os.path.join(tk, "run_%s.txt" % RUNNAME), encoding="utf-8").read().splitlines()
    except Exception as ex:                                                                   # noqa: BLE001
        print("ZZ CHECK-CONSTCLOSE FAIL logs-unreadable %s: %s" % (type(ex).__name__, ex))
        return 1
    for name, ok, det in run_checks(RL, ML, lines, sha):
        check(name, ok, det)
    counts.update(replan_rows=int(len(RL["ci"])), mid_rows=int(len(ML["ci"])), const_file_sha256=sha[:16])
    print("counts " + json.dumps(counts, sort_keys=True))
    print("ZZ CHECK-CONSTCLOSE " + ("PASS checks-failed 0 replan-rows %d mid-rows %d" % (counts["replan_rows"], counts["mid_rows"]) if not failed else "FAIL " + "; ".join(failed)))
    return 0 if not failed else 1


# ============================================================================================ selftest (synthetic logs, planted failures)
def synth(n=60, value=VALUE, seed=0):
    rng = np.random.default_rng(seed)
    j = np.tile(np.arange(0, 12), n // 12 + 1)[:n]
    eo = j >= FULL_FROM
    pres = rng.random(n) > 0.2
    pres[(eo) & (np.arange(n) % 7 == 0)] = True
    v_l = rng.uniform(0, 15, n)
    cs = np.where(eo, value, 0.0)
    return {"ci": np.repeat(np.arange(n // 12 + 1), 12)[:n], "t0": np.zeros(n, np.int64), "j": j, "k": np.zeros(n, np.int64), "present": pres, "est_ok": eo.copy(),
            "c_est": np.where(eo, value, np.nan), "c_src": cs, "v_l": v_l, "v_L": np.maximum(0.0, v_l + cs)}


def selftest():
    res = []

    def chk(name, ok):
        res.append(bool(ok))
        print("   %-110s %s" % (name, bool(ok)))
    sha = "ab" * 32
    RL, ML = synth(seed=1), synth(seed=2)

    def line(RL_, ML_, d=(0, 0, 0, 0), s=sha):
        full = int(RL_["est_ok"].sum() + ML_["est_ok"].sum())
        short = int((~RL_["est_ok"]).sum() + (~ML_["est_ok"]).sum())
        return ["arm x", "est looks scored full %d short %d preroll full %d short %d file %s" % (full + d[0], short + d[1], d[2], d[3], s), "rc 0 finished"]

    def all_ok(RL_, ML_, lines):
        r = run_checks(RL_, ML_, lines, sha)
        return all(x[1] for x in r), [x[0][:4] + x[0][-12:] for x in r if not x[1]]
    ok0, bad0 = all_ok(RL, ML, line(RL, ML))
    chk("clean synthetic logs: every check passes (%d checks)" % len(run_checks(RL, ML, line(RL, ML), sha)), ok0)
    # planted failures
    r1 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r1["present"] & r1["est_ok"])[0][3])
    r1["c_src"][i] = 1.9
    r1["v_L"][i] = max(0.0, r1["v_l"][i] + 1.9)
    ok1, bad1 = all_ok(r1, ML, line(r1, ML))
    chk("planted 1.9 in the closing column of the replan log: K1a fails (and only the closing check %s)" % bad1, not ok1 and bad1 == ["K1a (replan log)"])
    m1 = {k: v.copy() for k, v in ML.items()}
    i = int(np.nonzero(m1["j"] == 7)[0][0])
    m1["est_ok"][i] = False
    m1["c_est"][i] = np.nan
    m1["c_src"][i] = 0.0
    m1["v_L"][i] = max(0.0, m1["v_l"][i])
    ok2, bad2 = all_ok(RL, m1, line(RL, m1))
    chk("planted short look at frame 7 in the mid-look log: K1c fails %s" % bad2, not ok2 and any(x.startswith("K1c") for x in bad2))
    ok3, bad3 = all_ok(RL, ML, line(RL, ML, d=(1, 0, 0, 0)))
    chk("planted wrong count in the run line (full + 1): L1 fails %s" % bad3, not ok3 and any(x.startswith("L1") for x in bad3))
    ok3b, _ = all_ok(RL, ML, line(RL, ML, s="cd" * 32))
    ok3c, _ = all_ok(RL, ML, line(RL, ML, d=(0, 0, 5, 0)))
    ok3d, _ = all_ok(RL, ML, ["arm x"])
    chk("run line: another file hash, a non-zero pre-roll count and a missing line each fail", not ok3b and not ok3c and not ok3d)
    r4 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r4["est_ok"])[0][2])
    r4["c_est"][i] = 2.0001
    ok4, bad4 = all_ok(r4, ML, line(r4, ML))
    chk("planted c_est 2.0001 on one est_ok row: K1b fails %s" % bad4, not ok4 and any(x.startswith("K1b") for x in bad4))
    r5 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r5["est_ok"] == False)[0][0])                                           # noqa: E712
    r5["c_est"][i] = 0.0
    ok5, bad5 = all_ok(r5, ML, line(r5, ML))
    chk("c_est not NaN on a short row: K1b fails", not ok5 and any(x.startswith("K1b") for x in bad5))
    r6 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r6["present"] & ~r6["est_ok"])[0][0])
    r6["c_src"][i] = 2.0
    r6["v_L"][i] = r6["v_l"][i] + 2.0
    ok6, bad6 = all_ok(r6, ML, line(r6, ML))
    chk("a closing of 2.0 on a present look without a full history: K1a fails", not ok6 and any(x.startswith("K1a") for x in bad6))
    r7 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r7["present"])[0][0])
    r7["v_L"][i] += 0.5
    ok7, bad7 = all_ok(r7, ML, line(r7, ML))
    chk("v_L that is not max(0, v_l + closing): K1d fails", not ok7 and any(x.startswith("K1d") for x in bad7))
    e = {k: v[:0] for k, v in RL.items()}
    chk("an empty log fails (no row is not a pass)", not all_ok(e, ML, line(e, ML))[0])
    chk("the constants of the card: closing 2.0 m/s, a full history from frame index 4 on", VALUE == 2.0 and FULL_FROM == 4)
    m9 = {k: v.copy() for k, v in ML.items()}
    i = int(np.nonzero(m9["j"] == 4)[0][0])
    m9["est_ok"][i] = False
    m9["c_est"][i] = np.nan
    m9["c_src"][i] = 0.0
    m9["v_L"][i] = max(0.0, m9["v_l"][i])
    chk("a short look at frame 4 fails K1c, short looks at frames 0 to 3 are allowed", any(x.startswith("K1c") for x in all_ok(RL, m9, line(RL, m9))[1]) and all_ok(RL, ML, line(RL, ML))[0])
    ok8, bad8 = all_ok(RL, ML, line(RL, ML))
    chk("short looks before frame 4 are counted and allowed (the clean logs hold %d of them)" % int((~RL["est_ok"]).sum()), ok8 and (~RL["est_ok"]).sum() > 0)
    src = open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    bad = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop"]
    chk("text of the check holds none of the banned words", not [x for x in bad if x in src])
    print("ZZ SELFTEST-CHECK-CONSTCLOSE ok: %s (%d of %d)" % (all(res), sum(res), len(res)))
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
