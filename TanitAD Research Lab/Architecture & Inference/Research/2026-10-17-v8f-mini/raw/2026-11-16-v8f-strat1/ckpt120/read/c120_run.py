"""c120_run.py -- the CKPT120 dump and T0 passes (CPU, 2 threads, zero training).  The far-output wrappers of the code of record (213250ce) are the evaluators:
  stack/scripts/v8f_t1_eval_far.py  (T1-kin dump: s_far, nav_s_in in the ticks file)      stack/scripts/v8f_mini_eval_far.py  (T0: ckpt_far / ckpt_far_now in eval_<tag>_far.npz)
Argv templates are the banked STRAT1 argv files (same inputs, same flags); only the script, ckpt, tag, out dir and the CKPT120 flags change.  Usage:
  python c120_run.py dumps  ARM [ARM ...]     T1 causal m10 dump (the 24 driving cells)               -> t1/dump_CK120-<arm>_m10_causal_heads
  python c120_run.py k9b    ARM [ARM ...]     T1 --limit logged --batch 64 tick dump (K9b)            -> t1/k9b/ticks_CK120K9B-<arm>.npz
  python c120_run.py t0     [--arms A ..] [--variants real donor zero]                               -> out/t0/eval_CK120-<arm>_<variant>[_far].npz
  python c120_run.py --smoke-arm S1_s0 ...  (see test_c120_read.py: a flags-off pass of a banked comparison checkpoint on a subset grid; a code check, never a result)
Every arm is REFUSED (nothing launched) unless its bank manifest, commit, checkpoint sha256 and reach verdict (PASS) hold (c120_lib.bank_refusal / reach_verdict), and the tree pins hold.  A pass whose output exists is skipped."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c120_lib as C  # noqa: E402

SL = C.SL
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MIN_COMMIT_GB = 5.5
VARIANT_FLAGS = {"real": [], "donor": ["--nav-shuffle"], "zero": ["--nav-withheld"]}


def _tokens(path):
    return [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip() and not ln.startswith("STRAT1 extra")]


def set_flag(a, flag, val):
    a[a.index(flag) + 1] = val


def drop_flag(a, flag, has_val=False):
    if flag in a:
        i = a.index(flag)
        del a[i:i + (2 if has_val else 1)]


def env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
             PYTHONPATH=";".join(os.path.normpath(p) for p in (C.TREE + "/stack", C.TREE + "/taniteval")))
    return e


def nav_flags(arm):
    """the nav arguments of an arm: every CKPT120 arm is text-nav (the STRAT1 S1 recipe); H2 / R2 add the strategic token"""
    out = ["--nav-text-npz", SL.pinned("navtext_nav.npz"), "--nav-text-sha256", SL.PINS["navtext_nav.npz"], "--far-out", "on"]
    if arm in C.NAV_S_ARMS:
        out += ["--nav-strat", "on", "--nav-strat-npz", C.pinned("navtext_nav_s.npz"), "--nav-strat-sha256", C.PINS["navtext_nav_s.npz"]]
    return out


def t1_argv(arm, kind, grid=None, max_windows=0, ckpt=None, far=True):
    """kind: 'causal' (the read's dump) or 'k9b' (--limit logged --batch 64).  Template = the banked S1_s0 T1 argv."""
    a = _tokens(f"{SL.HERE}/t1/argv_STRAT1-S1_s0_m10_causal_heads.txt")
    od = C.T1 if kind == "causal" else C.T1 + "/k9b"
    tag = f"CK120-{arm}_m10_causal_heads" if kind == "causal" else f"CK120K9B-{arm}"
    a[1] = os.path.normpath(C.TREE + "/stack/scripts/v8f_t1_eval_far.py")
    set_flag(a, "--ckpt", ckpt or f"{C.CKD}/{arm}/ckpt_last.pt")
    set_flag(a, "--tag", tag)
    set_flag(a, "--out-dir", od)
    set_flag(a, "--dump-ticks", od + "/ticks")
    if grid:
        set_flag(a, "--eval-grid", grid)
    if kind == "k9b":
        set_flag(a, "--limit", "logged")
        a += ["--batch", "64"]
    a += (nav_flags(arm) if far else ["--nav-text-npz", SL.pinned("navtext_nav.npz"), "--nav-text-sha256", SL.PINS["navtext_nav.npz"]]) + ["--dump-nav"]
    if max_windows:
        a += ["--max-windows", str(max_windows)]
    return a, od, tag


def t0_argv(arm, variant, out, grid=None, ckpt=None, far=True):
    a = _tokens(f"{SL.HERE}/out/t0/argv_STRAT1-S1_s0_real.txt")
    a[1] = os.path.normpath(C.TREE + "/stack/scripts/v8f_mini_eval_far.py")
    set_flag(a, "--ckpt", ckpt or f"{C.CKD}/{arm}/ckpt_last.pt")
    set_flag(a, "--tag", f"CK120-{arm}_{variant}")
    set_flag(a, "--out-dir", out)
    if grid:
        set_flag(a, "--eval-grid", grid)
    for f in ("--nav-shuffle", "--nav-withheld"):
        drop_flag(a, f)
    for f in ("--nav-text-npz", "--nav-text-sha256"):
        drop_flag(a, f, True)
    a += (nav_flags(arm) if far else ["--nav-text-npz", SL.pinned("navtext_nav.npz"), "--nav-text-sha256", SL.PINS["navtext_nav.npz"]]) + VARIANT_FLAGS[variant]
    return a


def commit_free_gb():
    import ctypes

    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M()
    m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def wait_commit():
    for _ in range(300):
        if commit_free_gb() >= MIN_COMMIT_GB:
            return
        time.sleep(60)
    raise SystemExit(f"REFUSED: commit headroom stayed under {MIN_COMMIT_GB} GB for 5 h")


def arm_problem(arm):
    if arm not in C.NEW_ARMS:
        return f"unknown arm {arm}"
    why = C.bank_refusal(arm)
    if why:
        return why
    v, p = C.reach_verdict(arm)
    if v != "PASS":
        return f"REFUSED: K-REACH-e verdict for {arm} is {v} ({p}); nothing is read from an arm whose new heads are not proven trained"
    return None


def _run(argv, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as lf:
        return subprocess.run(argv, env=env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode


def cmd_dumps(arms, kind):
    rc = 0
    for arm in arms:
        why = arm_problem(arm)
        if why:
            print("REFUSED", arm, why, flush=True)
            rc = 1
            continue
        a, od, tag = t1_argv(arm, kind)
        done = f"{od}/done_{tag}.txt"
        if os.path.exists(done):
            print("skip (done)", done, flush=True)
            continue
        wait_commit()
        os.makedirs(od + "/ticks", exist_ok=True)
        open(f"{od}/argv_{tag}.txt", "w", encoding="utf-8").write("\n".join(a) + "\n")
        t0 = time.time()
        r = _run(a, f"{od}/log_{tag}.txt")
        hdr, tk = f"{od}/t1v8f_{tag}.json", f"{od}/ticks/ticks_{tag}.npz"
        if kind == "causal":
            dd = f"{od}/dump_{tag}"
            n = len([f for f in os.listdir(dd) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(dd) else 0
            ok = r == 0 and os.path.exists(hdr) and n == 141 and os.path.exists(tk)
        else:
            ok = r == 0 and os.path.exists(tk)
        if ok:
            json.dump({"tag": tag, "arm": arm, "kind": kind, "ckpt_sha256": C.sha256_file(f"{C.CKD}/{arm}/ckpt_last.pt"), "commit": C.COMMIT, "wall_s": round(time.time() - t0, 1),
                       "written_by": "c120_run.py"}, open(done, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), arm, kind, "rc", r, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    return rc


def cmd_t0(arms, variants):
    out = C.OUT + "/t0"
    os.makedirs(out, exist_ok=True)
    rc = 0
    for arm in arms:
        why = arm_problem(arm)
        if why:
            print("REFUSED", arm, why, flush=True)
            rc = 1
            continue
        for v in variants:
            npz = C.new_t0(arm, v)
            if os.path.exists(npz) and os.path.exists(C.new_t0_far(arm, v)):
                print("skip (exists)", npz, flush=True)
                continue
            wait_commit()
            a = t0_argv(arm, v, out)
            open(f"{out}/argv_CK120-{arm}_{v}.txt", "w", encoding="utf-8").write("\n".join(a) + "\n")
            t0 = time.time()
            r = _run(a, f"{out}/log_CK120-{arm}_{v}.txt")
            ok = r == 0 and os.path.exists(npz) and os.path.exists(C.new_t0_far(arm, v))
            print(time.strftime("%H:%M:%S"), arm, v, "rc", r, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
            rc |= (not ok)
    return rc


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if not a or a[0] not in ("dumps", "k9b", "t0"):
        raise SystemExit(__doc__)
    tp = C.tree_problem()
    if tp:
        raise SystemExit("REFUSED: " + tp)
    if a[0] in ("dumps", "k9b"):
        sys.exit(cmd_dumps(a[1:], "causal" if a[0] == "dumps" else "k9b"))
    arms = list(C.NEW_ARMS)
    variants = ["real", "donor", "zero"]
    rest = a[1:]
    if "--arms" in rest:
        i = rest.index("--arms")
        j = next((k for k in range(i + 1, len(rest)) if rest[k].startswith("--")), len(rest))
        arms = rest[i + 1:j]
    if "--variants" in rest:
        i = rest.index("--variants")
        j = next((k for k in range(i + 1, len(rest)) if rest[k].startswith("--")), len(rest))
        variants = rest[i + 1:j]
    sys.exit(cmd_t0(arms, variants))


if __name__ == "__main__":
    main()
