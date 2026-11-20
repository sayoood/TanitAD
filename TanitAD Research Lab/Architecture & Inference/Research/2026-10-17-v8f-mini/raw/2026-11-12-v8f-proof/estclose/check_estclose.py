"""check_estclose.py -- pod check of the wrapper v8f_t1_eval_reader_rule_rzgfhqe.py (card ADD79, control e1: the estimate reaches the rule).
Part A (always; numpy and the standard library only): static facts of the wrapper text, the new text has none of the banned words. --pure-only stops here.
Part B (on the pod; needs the tool tree, the checkpoint, the data and the banked runs): four rollouts of the NEW wrapper on a subset of the pinned grid (the first --n-clips clips with all their
  windows), fresh (no pre-roll), model seed 0, readout K (--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2), --rule-every 5, q50 map, reader seed 0:
    e_rr    form RR (the reference of the same wrapper, same grid, same batches)
    e_zero  form RRe, estimator file of zero weights (planted by closest.save_planted)
    e_rc    form RRe, estimator file that returns the reader's closing output of the last frame (planted)
    e_c2    form RRe, a planted constant estimator of +2 m/s
  plus two refusals (RRe without --closing-est; RR with --closing-est). Checks:
    Z1  e_zero equals e_rr: dump arrays, tick file, rule log, rule-log side file, mid-look log, mid-look side file, first-plan file; every common array, largest difference 0; c_src is 0 on every row
        of both logs; the new keys c_est (NaN where no full history) and est_ok are present row for row in the rule log and in the mid-look log
    Z2  e_zero equals the BANKED RR run (--banked-rr) on the rows of the subset clips (replan and mid-look rule logs keyed by (ci, t0, j, k[, q]); first plan keyed by (ci, t0)): numeric columns
        within --tol-banked (default 1e-3), boolean and integer columns exactly; the counts are reported; the subset has other batches than the banked run, so the bar is a tolerance
    R1  e_rc against the BANKED RRh run (--banked-rrh): on every present row of the subset that has a full history (est_ok) c_src equals the banked c_src within --tol-reader (default 1e-4 m/s: the
        estimate reads float32 per-frame reads from the cache, the banked read is one batch); rows without a full history are counted and have c_src exactly 0 in e_rc
    R2  e_rc: c_est equals the c_read column of the same run on the est_ok rows (tick log side file and mid-look log), within --tol-reader
    C1  e_c2: c_src is 2.0 on every present row with est_ok and 0.0 on every present row without; v_L = max(0, v_l + c_src); c_est is 2.0 on every est_ok row (replan and mid-look logs)
    L1  the run text of e_zero / e_rc / e_c2 has the line "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>": the counts equal those recomputed from the logs (preroll 0 0),
        the sha256 is that of the estimator file; e_rr has no such line
    F1  the two refusals: non-zero return code, text with REFUSED and --closing-est
  Nothing here removes, moves or stops anything; part B writes only inside --workdir (refused if it exists, except with --analyze-only, which only reads the run folders already there).
Usage on the pod:
  python check_estclose.py --new NEW.py --tree TREE --ckpt CKPT --tracks native_tracks.pt --eval-cache DIR --eval-join J.npz --grid PINNED_GRID.json --reader READER.pt --qmap MAP.json
         --workdir NEWDIR --distrule-dir DIR --closest-dir DIR --banked-rr T1RUNDIR --banked-rrh T1RUNDIR [--python PY] [--device cuda] [--n-clips 3] [--dry] [--analyze-only]
Last line: ZZ CHECK-ESTCLOSE PASS <counts>  or  ZZ CHECK-ESTCLOSE FAIL <names>   (--pure-only: ZZ CHECK-ESTCLOSE-PURE PASS or FAIL <names>)"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []
COUNTS = {}


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
    if not cond:
        FAILED.append(name.split(" ")[0])


ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--new", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqe.py"))
ap.add_argument("--pure-only", action="store_true")
ap.add_argument("--analyze-only", action="store_true")
ap.add_argument("--dry", action="store_true")
for k in ("tree", "ckpt", "tracks", "eval-cache", "eval-join", "grid", "reader", "qmap", "workdir", "banked-rr", "banked-rrh"):
    ap.add_argument("--" + k)
ap.add_argument("--distrule-dir", default="")
ap.add_argument("--closest-dir", default="")
ap.add_argument("--replan-steps", type=int, default=10)
ap.add_argument("--n-clips", type=int, default=3)
ap.add_argument("--device", default="cuda")
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--tol-reader", type=float, default=1e-4)
ap.add_argument("--tol-banked", type=float, default=1e-3)
ARGS = ap.parse_args()

SRC = open(ARGS.new, encoding="utf-8").read()
print("wrapper", ARGS.new, hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16], flush=True)

# ==================================================================================================================== part A
texts = [
    ("A1 the flag --closing-est exists and is refused outside form RRe", 'ap.add_argument("--closing-est", default="")' in SRC and "belongs to form RRe only" in SRC and "needs --closing-est" in SRC),
    ("A2 form RRe is a form, takes the q map, the reader and the later looks", '"RRd", "RRe")' in SRC and 'f in ("RR", "RRc", "RRh", "RRd", "RRe")' in SRC and '"RRh", "RRe"):' in SRC),
    ("A3 the closing of RRe is the head value of the mix (the estimate)", '"RRd": "head", "RRe": "head"}.get(f, "zero")' in SRC),
    ("A4 the rule log carries c_est and est_ok, the run line is written after the pre-roll files", 'dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta' in SRC and "        write_est_line()\n" in SRC),
    ("A5 the estimate has the fallback 0.0 without a full history", "return np.where(est_ok, c_est, 0.0)" in SRC and "if e - need + 1 >= 0 and e < len(s.d[\"feats\"]):" in SRC),
]
for n_, c_ in texts:
    check(n_, bool(c_))
BAD = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
head_txt = SRC[:SRC.index("Text of the parent (rzgfhqc) follows.")].lower() if "Text of the parent (rzgfhqc) follows." in SRC else ""
own = open(os.path.abspath(__file__), encoding="utf-8").read().lower()
check("A6 the new header text and this check have none of the banned words", bool(head_txt) and not [w for w in BAD if w in head_txt or w in own], str([w for w in BAD if w in head_txt or w in own]))
if ARGS.pure_only:
    print("ZZ CHECK-ESTCLOSE-PURE " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
    sys.exit(0 if not FAILED else 1)

# ==================================================================================================================== part B (pod)
for k in ("tree", "ckpt", "tracks", "eval_cache", "eval_join", "grid", "reader", "qmap", "workdir", "banked_rr", "banked_rrh"):
    if not getattr(ARGS, k):
        print("ZZ CHECK-ESTCLOSE STOP missing --" + k.replace("_", "-"))
        sys.exit(3)
W = ARGS.workdir
M = ARGS.replan_steps
DISTRULE = ARGS.distrule_dir or os.path.dirname(os.path.abspath(ARGS.new))
CLOSEST = ARGS.closest_dir or os.path.dirname(os.path.abspath(ARGS.new))
TIMEOUT = shutil.which("timeout")
if os.path.exists(W) and not ARGS.analyze_only:
    print("ZZ CHECK-ESTCLOSE STOP OUTPUT-EXISTS " + W)
    sys.exit(3)
if not ARGS.dry and not os.path.exists(W):
    os.makedirs(W)
sys.path.insert(0, CLOSEST)
import closest as CL  # noqa: E402

EST = os.path.join(W, "est")
PLANTED = {"zero": os.path.join(EST, "planted_zero.npz"), "rc": os.path.join(EST, "planted_reader_closing.npz"), "c2": os.path.join(EST, "planted_const_2.npz")}
if not ARGS.dry and not os.path.isdir(EST):
    os.makedirs(EST)
    CL.save_planted(PLANTED["zero"], "zero", 10)
    CL.save_planted(PLANTED["rc"], "reader_closing", 10)
    CL.save_planted(PLANTED["c2"], "const", 10, 2.0)

GRID_SUB = os.path.join(W, "grid_subset.json")
if not ARGS.dry and not os.path.exists(GRID_SUB):
    g = json.load(open(ARGS.grid, encoding="utf-8"))
    order = list(g["order"])[:ARGS.n_clips]
    g2 = dict(g)
    g2["order"], g2["windows"] = order, {c: g["windows"][c] for c in order}
    g2["n_windows"] = int(sum(len(g["windows"][c]) for c in order))
    g2["estclose_subset"] = {"n_clips": len(order), "source": os.path.basename(ARGS.grid)}
    with open(GRID_SUB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(g2, fh)

RUNS = [("e_rr", "RR", None), ("e_zero", "RRe", "zero"), ("e_rc", "RRe", "rc"), ("e_c2", "RRe", "c2")]


def tag_of(name):
    return name + "_m%d_causal_heads" % M


def run_cmd(name, form, est, out=None, extra=()):
    out = out or os.path.join(W, name)
    est_args = ["--closing-est", PLANTED[est]] if est else []
    cmd = [ARGS.python, ARGS.new, "--tree", ARGS.tree, "--form", form, "--reader", ARGS.reader, "--qmap", ARGS.qmap] + est_args + list(extra) + [
        "--rule-log", os.path.join(out, "ticks", "rule_log_%s.npz" % name), "--rule-every", "5",
        "--readout", "causal_g", "--readout-log", os.path.join(out, "ticks", "readout_log_%s.json" % name), "--acc-gain", "0", "--path-exec", "--kappa-max", "0.2",
        "--preroll-steps", "0",
        "run", "--ckpt", ARGS.ckpt, "--tag", tag_of(name), "--tracks", ARGS.tracks, "--eval-cache", ARGS.eval_cache, "--eval-grid", GRID_SUB, "--eval-join", ARGS.eval_join,
        "--out-dir", out, "--replan-steps", str(M), "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal",
        "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze", "--device", ARGS.device]
    if TIMEOUT:
        cmd = [TIMEOUT, "1800"] + cmd
    return out, cmd


def env_run():
    return dict(os.environ, DISTRULE_DIR=DISTRULE, CLOSEST_DIR=CLOSEST, PYTHONPATH=os.path.join(ARGS.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")


def run_one(name, form, est):
    out, cmd = run_cmd(name, form, est)
    if ARGS.dry:
        print("CMD: DISTRULE_DIR=%s CLOSEST_DIR=%s PYTHONPATH=%s %s" % (DISTRULE, CLOSEST, os.path.join(ARGS.tree, "stack"), " ".join(cmd)))
        return 0
    if os.path.exists(out):
        if ARGS.analyze_only:
            return 0
        raise SystemExit("OUTPUT-EXISTS " + out)
    os.makedirs(os.path.join(out, "ticks"))
    with open(os.path.join(out, "ticks", "run_%s.txt" % name), "w", encoding="utf-8") as fh:
        fh.write("arm %s form %s closing-est %s\n" % (name, form, PLANTED[est] if est else "none"))
    with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, env=env_run(), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("RUN %s form=%s est=%s rc=%d" % (name, form, est, rc), flush=True)
    return rc


for name, form, est in RUNS:
    rc = run_one(name, form, est)
    if rc != 0:
        print("ZZ CHECK-ESTCLOSE FAIL run-%s-rc-%d (see %s)" % (name, rc, os.path.join(W, name, "t1.log")))
        sys.exit(1)


def refusal(label, form, est_flag):
    out = os.path.join(W, "refusal_" + label)
    _, cmd = run_cmd("refusal_" + label, form, None, out=out, extra=(["--closing-est", PLANTED["zero"]] if est_flag else []))
    if TIMEOUT:
        cmd[1] = "600"
    path = os.path.join(W, "refusal_%s.log" % label)
    if ARGS.dry:
        print("CMD: " + " ".join(cmd))
        return None, ""
    if os.path.exists(path) and ARGS.analyze_only:
        return None, open(path, encoding="utf-8").read()
    cp = subprocess.run(cmd, env=env_run(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = cp.stdout.decode("utf-8", "replace")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return cp.returncode, text


ref1 = refusal("noflag", "RRe", False)
ref2 = refusal("otherform", "RR", True)
if ARGS.dry:
    print("ZZ CHECK-ESTCLOSE dry run: nothing executed beyond part A")
    sys.exit(0)


# ---------------------------------------------------------------- helpers
def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ticks_dir(name):
    return os.path.join(W, name, "ticks")


def tfile(name, stem):
    return os.path.join(ticks_dir(name), stem % name)


def banked_file(folder, stem):
    nm = os.path.basename(os.path.normpath(folder))
    return os.path.join(folder, "ticks", stem % nm)


def dumps(name):
    d = os.path.join(W, name, "dump_" + tag_of(name))
    return {os.path.basename(f): load_npz(f) for f in sorted(glob.glob(os.path.join(d, "ep*.npz")))}


def maxdiff_dicts(a, b, skip=("tag", "form", "mix", "mix_counts", "closing_est", "reader", "qmap"), only_common=True):
    """-> (largest absolute difference over the common numeric arrays, problems, keys only b has). Arrays that name the run or the form (strings) are skipped."""
    a = {k: v for k, v in a.items() if k not in skip}
    b = {k: v for k, v in b.items() if k not in skip}
    extra = sorted(set(b) - set(a))
    b = {k: v for k, v in b.items() if k in a}
    worst, prob = 0.0, []
    if sorted(a) != sorted(b):
        prob.append("keys differ %s" % (sorted(set(a) ^ set(b))[:4],))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            prob.append("shape of %s" % k)
            continue
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            if not np.array_equal(np.isnan(xf), np.isnan(yf)):
                prob.append("nan pattern of %s" % k)
                continue
            if xf.size and np.isfinite(xf).any():
                worst = max(worst, float(np.nanmax(np.abs(xf - yf))))
        elif x.dtype.kind in "US" and not np.array_equal(x, y):
            continue
        elif not np.array_equal(x, y):
            prob.append("content of %s" % k)
    return worst, prob, extra


def keys_of(L, with_q):
    cols = ["ci", "t0", "j", "k"] + (["q"] if with_q else [])
    return list(zip(*[np.asarray(L[c]).astype(np.int64).tolist() for c in cols]))


def compare_banked(A, B, with_q, tol):
    """rows of A (subset run) against the rows of B (banked) with the same key (ci, t0, j, k[, q]) -> dict of counts. Numeric columns within tol, boolean and integer columns exactly."""
    ka, kb = keys_of(A, with_q), keys_of(B, with_q)
    ib = {k: i for i, k in enumerate(kb)}
    ia = [i for i, k in enumerate(ka) if k in ib]
    jb = [ib[ka[i]] for i in ia]
    cols = [c for c in A if c in B and A[c].ndim == 1 and A[c].shape == (len(ka),) and B[c].shape == (len(kb),) and A[c].dtype.kind in "fiub" and B[c].dtype.kind in "fiub"
            and c not in ("c_est", "est_ok")]
    bad_rows, worst, bad_cols = set(), 0.0, {}
    for c in cols:
        x, y = A[c][ia], B[c][jb]
        if x.dtype.kind in "iub" or y.dtype.kind in "iub":
            ne = x.astype(np.int64) != y.astype(np.int64)
        else:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            with np.errstate(invalid="ignore"):
                ne = ~((np.isnan(xf) & np.isnan(yf)) | (np.abs(xf - yf) <= tol))
            fin = np.isfinite(xf) & np.isfinite(yf)
            if fin.any():
                worst = max(worst, float(np.max(np.abs(xf[fin] - yf[fin]))))
        if ne.any():
            bad_cols[c] = int(ne.sum())
            bad_rows.update(np.nonzero(ne)[0].tolist())
    return {"rows_subset": len(ka), "rows_common": len(ia), "rows_not_in_banked": len(ka) - len(ia), "columns": len(cols), "rows_mismatch": len(bad_rows), "largest_numeric_diff": worst, "bad_columns": bad_cols}


def run_text(name):
    return open(os.path.join(ticks_dir(name), "run_%s.txt" % name), encoding="utf-8").read().splitlines()


LINE = re.compile(r"^est looks scored full (\d+) short (\d+) preroll full (\d+) short (\d+) file ([0-9a-f]{64})$")
STEMS = [("ticks_%s_m%d_causal_heads.npz" % ("%s", M), "tick file"), ("rule_log_%s.npz", "rule log"), ("rule_log_%s_rd.npz", "rule log side file"),
         ("rule_log_mid_%s.npz", "mid-look log"), ("rule_log_mid_%s_rd.npz", "mid-look side file"), ("p1_first_%s.npz", "first-plan file")]

# ---------------------------------------------------------------- Z1: e_zero equals e_rr
worst, prob, files, newk = 0.0, [], 0, {}
DR, DZ = dumps("e_rr"), dumps("e_zero")
if sorted(DR) != sorted(DZ) or not DR:
    prob.append("dump file lists differ or empty")
for f in sorted(set(DR) & set(DZ)):
    w_, p_, _ = maxdiff_dicts(DR[f], DZ[f])
    worst, prob, files = max(worst, w_), prob + p_, files + 1
for stem, label in STEMS:
    pa, pb = tfile("e_rr", stem), tfile("e_zero", stem)
    ea, eb = os.path.isfile(pa), os.path.isfile(pb)
    if ea != eb:
        prob.append("%s present only in %s" % (label, "e_rr" if ea else "e_zero"))
        continue
    if not ea:
        continue
    w_, p_, ex = maxdiff_dicts(load_npz(pa), load_npz(pb))
    worst, prob, files = max(worst, w_), prob + ["%s: %s" % (label, x) for x in p_], files + 1
    if ex:
        newk[label] = ex
RL_Z, ML_Z = load_npz(tfile("e_zero", "rule_log_%s.npz")), load_npz(tfile("e_zero", "rule_log_mid_%s.npz"))
RL_R, ML_R = load_npz(tfile("e_rr", "rule_log_%s.npz")), load_npz(tfile("e_rr", "rule_log_mid_%s.npz"))
c_zero = bool(len(RL_Z["c_src"]) and len(ML_Z["c_src"]) and np.all(RL_Z["c_src"] == 0.0) and np.all(ML_Z["c_src"] == 0.0) and np.all(RL_R["c_src"] == 0.0) and np.all(ML_R["c_src"] == 0.0))
aligned = all(len(RL_Z[k]) == len(RL_Z["ci"]) for k in ("c_est", "est_ok")) and all(len(ML_Z[k]) == len(ML_Z["ci"]) for k in ("c_est", "est_ok"))
n_tick, n_mid = len(RL_Z["ci"]), len(ML_Z["ci"])
nan_pat = bool(np.array_equal(np.isnan(RL_Z["c_est"]), ~RL_Z["est_ok"].astype(bool)) and np.array_equal(np.isnan(ML_Z["c_est"]), ~ML_Z["est_ok"].astype(bool)))
COUNTS.update(zero_tick_rows=n_tick, zero_mid_rows=n_mid, zero_files=files)
check("Z1 e_zero equals e_rr: dump, tick, rule-log, side, mid-look and first-plan arrays, largest difference %.3g (%d files); c_src is 0 on all %d + %d rows; c_est / est_ok row for row (NaN exactly where est_ok is False); new keys %s"
      % (worst, files, n_tick, n_mid, json.dumps(newk)), worst == 0.0 and not prob and files > 0 and c_zero and aligned and nan_pat and "rule log" in newk and "mid-look log" in newk, str(prob[:3]))

# ---------------------------------------------------------------- Z2: e_zero against the banked RR run
BRL, BML, BP1 = banked_file(ARGS.banked_rr, "rule_log_%s.npz"), banked_file(ARGS.banked_rr, "rule_log_mid_%s.npz"), banked_file(ARGS.banked_rr, "p1_first_%s.npz")
if all(os.path.isfile(p) for p in (BRL, BML, BP1)):
    cr = compare_banked(RL_Z, load_npz(BRL), False, ARGS.tol_banked)
    cm = compare_banked(ML_Z, load_npz(BML), True, ARGS.tol_banked)
    P1Z, P1B = load_npz(tfile("e_zero", "p1_first_%s.npz")), load_npz(BP1)
    kz = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(P1Z["ci"], P1Z["t0"]))}
    kb = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(P1B["ci"], P1B["t0"]))}
    common = [k for k in kz if k in kb]
    dplan = float(max((np.abs(P1Z["plan"][kz[k]].astype(np.float64) - P1B["plan"][kb[k]].astype(np.float64)).max() for k in common), default=np.nan))
    COUNTS.update(z2_replan_common=cr["rows_common"], z2_mid_common=cm["rows_common"], z2_plan_common=len(common))
    check("Z2 e_zero against the banked RR run %s on the subset rows: replan log %s; mid-look log %s; first plans %d common, largest difference %.3g (bar %.0e)"
          % (os.path.basename(os.path.normpath(ARGS.banked_rr)), json.dumps(cr), json.dumps(cm), len(common), dplan, ARGS.tol_banked),
          cr["rows_common"] == cr["rows_subset"] and cm["rows_common"] == cm["rows_subset"] and cr["rows_mismatch"] == 0 and cm["rows_mismatch"] == 0 and len(common) == len(kz) and dplan <= ARGS.tol_banked)
else:
    check("Z2 banked RR files exist", False, str([p for p in (BRL, BML, BP1) if not os.path.isfile(p)]))

# ---------------------------------------------------------------- R1, R2: e_rc
RL_C, ML_C = load_npz(tfile("e_rc", "rule_log_%s.npz")), load_npz(tfile("e_rc", "rule_log_mid_%s.npz"))
HRL, HML = banked_file(ARGS.banked_rrh, "rule_log_%s.npz"), banked_file(ARGS.banked_rrh, "rule_log_mid_%s.npz")
if all(os.path.isfile(p) for p in (HRL, HML)):
    out_r = {}
    ok_all = True
    for label, A, B, with_q in (("replan", RL_C, load_npz(HRL), False), ("mid", ML_C, load_npz(HML), True)):
        kb = {k: i for i, k in enumerate(keys_of(B, with_q))}
        ka = keys_of(A, with_q)
        pres, okk = A["present"].astype(bool), A["est_ok"].astype(bool)
        rows = [i for i in range(len(ka)) if pres[i] and okk[i] and ka[i] in kb and bool(B["present"][kb[ka[i]]])]
        d = np.array([abs(float(A["c_src"][i]) - float(B["c_src"][kb[ka[i]]])) for i in rows])
        short = [i for i in range(len(ka)) if pres[i] and not okk[i]]
        short_zero = bool(all(A["c_src"][i] == 0.0 for i in short))
        nz = int(sum(1 for i in rows if A["c_src"][i] != 0.0))
        out_r[label] = {"present_full_compared": len(rows), "largest_diff": float(d.max()) if len(d) else float("nan"), "not_zero": nz, "present_short_history": len(short), "short_c_src_zero": short_zero,
                        "present_full_not_in_banked": int((pres & okk).sum()) - len(rows)}
        ok_all = ok_all and len(rows) > 0 and bool(len(d) and d.max() <= ARGS.tol_reader) and nz >= 1 and short_zero
    COUNTS.update(r1=out_r)
    check("R1 e_rc against the banked RRh run %s on present rows with a full history: %s (bar %.0e m/s)" % (os.path.basename(os.path.normpath(ARGS.banked_rrh)), json.dumps(out_r), ARGS.tol_reader), ok_all)
else:
    check("R1 banked RRh files exist", False, str([p for p in (HRL, HML) if not os.path.isfile(p)]))
SIDE_C = load_npz(tfile("e_rc", "rule_log_%s_rd.npz"))
okt, okm = RL_C["est_ok"].astype(bool), ML_C["est_ok"].astype(bool)
d_t = float(np.abs(RL_C["c_est"][okt] - SIDE_C["c_read"][okt].astype(np.float64)).max()) if okt.any() and len(SIDE_C["c_read"]) == len(okt) else float("nan")
d_m = float(np.abs(ML_C["c_est"][okm] - ML_C["c_read"][okm]).max()) if okm.any() else float("nan")
check("R2 e_rc: c_est equals the c_read of the same run on the est_ok rows: tick log %d rows largest difference %.3g, mid-look log %d rows %.3g (bar %.0e)" % (int(okt.sum()), d_t, int(okm.sum()), d_m, ARGS.tol_reader),
      okt.any() and okm.any() and d_t <= ARGS.tol_reader and d_m <= ARGS.tol_reader)

# ---------------------------------------------------------------- C1: e_c2
RL_2, ML_2 = load_npz(tfile("e_c2", "rule_log_%s.npz")), load_npz(tfile("e_c2", "rule_log_mid_%s.npz"))
res2 = {}
ok2 = True
for label, L in (("replan", RL_2), ("mid", ML_2)):
    pres, okk = L["present"].astype(bool), L["est_ok"].astype(bool)
    a2 = bool(np.all(L["c_src"][pres & okk] == 2.0))
    a0 = bool(np.all(L["c_src"][pres & ~okk] == 0.0))
    avl = bool(np.allclose(L["v_L"][pres], np.maximum(0.0, L["v_l"][pres] + L["c_src"][pres])))
    ae = bool(np.all(L["c_est"][okk] == 2.0) and np.isnan(L["c_est"][~okk]).all())
    res2[label] = {"present_full": int((pres & okk).sum()), "present_short": int((pres & ~okk).sum()), "c_src_2": a2, "c_src_0": a0, "v_L": avl, "c_est": ae}
    ok2 = ok2 and a2 and a0 and avl and ae and (pres & okk).any()
check("C1 e_c2: %s" % json.dumps(res2), ok2)

# ---------------------------------------------------------------- L1: run line
L1 = {}
ok_l = True
for name, est in (("e_zero", "zero"), ("e_rc", "rc"), ("e_c2", "c2")):
    RLg, MLg = load_npz(tfile(name, "rule_log_%s.npz")), load_npz(tfile(name, "rule_log_mid_%s.npz"))
    full = int(RLg["est_ok"].astype(bool).sum() + MLg["est_ok"].astype(bool).sum())
    short = int((~RLg["est_ok"].astype(bool)).sum() + (~MLg["est_ok"].astype(bool)).sum())
    lines = [x for x in run_text(name) if x.startswith("est looks")]
    m_ = LINE.match(lines[0]) if len(lines) == 1 else None
    sha = hashlib.sha256(open(PLANTED[est], "rb").read()).hexdigest()
    good = bool(m_ and int(m_.group(1)) == full and int(m_.group(2)) == short and int(m_.group(3)) == 0 and int(m_.group(4)) == 0 and m_.group(5) == sha)
    L1[name] = {"lines": len(lines), "full": full, "short": short, "line_ok": good}
    ok_l = ok_l and good
no_line = not [x for x in run_text("e_rr") if x.startswith("est looks")]
L1["e_rr_has_no_line"] = no_line
check("L1 run line counts equal the counts recomputed from the logs, file sha256 equals the estimator file: %s" % json.dumps(L1), ok_l and no_line)

# ---------------------------------------------------------------- F1: refusals
r1, t1 = ref1
r2, t2 = ref2
check("F1 refusals: RRe without --closing-est (rc %s) and RR with --closing-est (rc %s) stop with REFUSED naming --closing-est" % (r1, r2),
      (r1 is None or r1 != 0) and (r2 is None or r2 != 0) and "REFUSED" in t1 and "--closing-est" in t1 and "REFUSED" in t2 and "--closing-est" in t2, (t1.strip()[-120:] + " | " + t2.strip()[-120:]))

print("counts " + json.dumps(COUNTS, sort_keys=True))
print("ZZ CHECK-ESTCLOSE " + ("PASS checks-failed 0 " + "tick-rows %d mid-rows %d" % (n_tick, n_mid) if not FAILED else "FAIL " + "; ".join(FAILED)))
sys.exit(0 if not FAILED else 1)
