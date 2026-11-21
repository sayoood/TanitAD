"""check_midlook_closing.py -- checks of v8f_t1_eval_reader_rule_rzgfhqc.py (study "midlook-closing": the later looks of the rule for the forms RRc and RRh).
Part A (always; numpy and the standard library only): the helper _reader_closing_at of the new wrapper is executed here with stubs against known values (window j + q, NaN rows, scaling
  by the stub), static facts of the text (refusal widened, RRd named, reader branch for RRc and RRh, RR closing stays zero, c_read in MID_FIELDS); each with a planted wrong arm.
Part B (on the pod, needs the tool tree, the checkpoints and the data; skipped with --pure-only): runs the parent wrapper and the new wrapper on a tiny restricted grid
  (3 clips, 27 windows; model seed 0; --preroll-steps 30 --readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2) and compares.
  A  --rule-every 10 (the rule at the tick only): forms RR, RRc, RRh: every dump array, every tick-file array (but the run name, array tag) and every rule-log array of the new wrapper
     equals the parent's (largest difference 0).
  B  --rule-every 5: forms RR and RLc: dump, tick, rule-log and mid-look-log arrays equal the parent's (largest difference 0); keys only the new wrapper has are listed, not compared.
  C  reach, --rule-every 5, on the later-look rows (mid-look log) with a lead present:
       RRc  c_src equals the label closing at the clip and frame j + q (the label source recomputed here from the join; 1e-9), where the label has none: 0; and is not 0 on one row at least
       RRh  c_src equals the reader's closing output recomputed here from the cache tokens at j + q (cache file of the clip, float16 tokens, the five frames up to j + q, one row at a time;
            tolerance --tol-reader, default 1e-6), equals the logged c_read exactly, and is not 0 on one row at least
       RR   control: c_src is exactly 0 on every row of the later-look log
       c_read logged by the three reader forms equals the recomputed reader closing on the present rows (tolerance --tol-reader)
       planted: the RR mid-look log read as if it were RRh is caught.
     Not covered: the token cache of the pod is read through the cache file of the clip (the path of check_prev_read.py), not through the window state of the rollout; the equality of those two
     is the wrapper's own premise (the same arrays feed the tick-level read).
  D  --rule-every 5 with form RRd is REFUSED (non-zero return code, text naming RRd).
Nothing here removes, moves or stops anything; part B writes only inside --workdir (refused if it exists, except with --analyze-only, which only reads the run folders already there).
Usage on the pod (part B):
  python check_midlook_closing.py --parent PARENT.py --new NEW.py --tree TREE --ckpt CKPT --tracks native_tracks.pt --eval-cache DIR --eval-join J.npz --grid GRID_P30.json
         --reader READER.pt --qmap MAP.json --workdir NEWDIR --distrule-dir DIR [--python PY] [--device cuda] [--dry] [--analyze-only]
Usage anywhere: python check_midlook_closing.py --pure-only [--new NEW.py]
Last line: ZZ CHECK-MIDLOOK-CLOSING PASS  or  ZZ CHECK-MIDLOOK-CLOSING FAIL <names>   (--pure-only: ZZ CHECK-MIDLOOK-CLOSING-PURE PASS or FAIL <names>)"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
    if not cond:
        FAILED.append(name)


def planted(name, wrong_passes, detail=""):
    print(("PASS " if not wrong_passes else "FAIL ") + "planted wrong arm caught: " + name + (" -- " + detail if detail else ""), flush=True)
    if wrong_passes:
        FAILED.append("planted " + name)


ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--new", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqc.py"))
ap.add_argument("--parent", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhq.py"))
ap.add_argument("--pure-only", action="store_true")
ap.add_argument("--analyze-only", action="store_true")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--tree")
ap.add_argument("--ckpt")
ap.add_argument("--tracks")
ap.add_argument("--eval-cache")
ap.add_argument("--eval-join")
ap.add_argument("--grid")
ap.add_argument("--reader")
ap.add_argument("--qmap")
ap.add_argument("--workdir")
ap.add_argument("--distrule-dir", default="")
ap.add_argument("--replan-steps", type=int, default=10)
ap.add_argument("--preroll-steps", type=int, default=30)
ap.add_argument("--device", default="cuda")
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--tol-reader", type=float, default=1e-6)
ARGS = ap.parse_args()

SRC = open(ARGS.new, encoding="utf-8").read()
print("wrapper", ARGS.new, hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16], flush=True)


def between(a, b):
    i, j = SRC.index(a), SRC.index(b)
    assert i < j
    return SRC[i:j]


# ==================================================================================================================== part A
HELPER = between("def _reader_closing_at", "def mid_looks")


class _Rd:
    hist = 5


class _St:
    def __init__(self, j, feats):
        self.j, self.d = j, {"feats": feats}


def stub_closing(rd, tokens):                                  # a known value per window: ten times the sum of the tokens
    return (np.asarray(tokens, np.float64).sum(axis=(1, 2, 3)) * 10.0).astype(np.float32)


NS = {"np": np, "reader_closing": stub_closing}
exec(HELPER, NS)
rng = np.random.default_rng(5)
FE = [rng.normal(0, 1, (40, 2, 3)).astype(np.float16) for _ in range(3)]
STS = [_St(10, FE[0]), _St(2, FE[1]), _St(34, FE[2]), _St(10, FE[0])]
Q = 5


def want_row(s, q, shift=0):
    e = s.j + q + shift
    if e - 4 < 0 or e >= len(s.d["feats"]):
        return np.nan
    return float(stub_closing(None, np.asarray(s.d["feats"][e - 4:e + 1], np.float32)[None])[0])


got = NS["_reader_closing_at"](STS, _Rd(), Q)
want = np.array([want_row(s, Q) for s in STS])
check("A1 helper: closing read on the window ending at frame j + q (rows with e-4 < 0 or e past the clip are NaN), largest difference %.3g" % float(np.nanmax(np.abs(got - want))),
      np.array_equal(np.isnan(got), np.isnan(want)) and float(np.nanmax(np.abs(got - want))) == 0.0)
check("A1b helper: the row at j = 34, q = 5 (frame 39, the last frame of the clip) is finite; j = 2, q = 5 (frame 7) is finite; the clip end rule: j = 38, q = 5 is NaN",
      np.isfinite(got[2]) and np.isfinite(got[1]) and np.isnan(NS["_reader_closing_at"]([_St(38, FE[2])], _Rd(), Q))[0] and np.isnan(NS["_reader_closing_at"]([_St(-2, FE[2])], _Rd(), Q))[0])
wrong_window = np.array([want_row(s, Q, shift=-1) for s in STS])            # the window one frame early (the tick-level frame of the gap read at j + q - 1)
planted("helper read at frame j + q - 1", bool(np.array_equal(np.isnan(got), np.isnan(wrong_window)) and np.nanmax(np.abs(got - wrong_window)) == 0.0))
wrong_scale = want * 10.0
planted("helper value scaled by ten", bool(np.nanmax(np.abs(got - wrong_scale)) == 0.0))
check("A1c helper: identical windows give identical values (rows 0 and 3), a row is independent of the others",
      got[0] == got[3] and NS["_reader_closing_at"]([STS[2]], _Rd(), Q)[0] == got[2])

checks_text = [
    ("A2 the refusal names RRd and admits RRc and RRh", 'if f == "RRd":' in SRC and 'not available for the rule form RRd' in SRC and 'if f not in ("RLb", "RLc", "RR", "RRc", "RRh"):' in SRC),
    ("A3 the reader branch of mid_looks takes RR, RRc, RRh", 'if form in ("RR", "RRc", "RRh"):' in SRC and 'if form == "RR":\n            p, g, okr = _reader_at' not in SRC),
    ("A4 closing of the later look is the reader read in RRh only (zeros otherwise)", 'hg, hc = used.astype(np.float32).astype(np.float64), (c_read if form == "RRh" else np.zeros(B, np.float64))' in SRC),
    ("A5 c_read is a key of the mid-look log (MID_FIELDS) and of the row", '"nchg", "c_read")' in SRC and "c_src=c_src, c_read=c_read," in SRC),
    ("A6 the label forms log c_read as NaN", "            c_read = np.full(B, np.nan)\n" in SRC),
]
for n_, c_ in checks_text:
    check(n_, bool(c_))
planted("a text without the RRd refusal", 'if f == "RRd":' in SRC.replace('if f == "RRd":', "if f == 'x':"))
BAD = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop", "driving " + "performance")
head = SRC[:SRC.index("Text of the parent (rzgfhq) follows.")].lower() + HELPER.lower()
own = open(os.path.abspath(__file__), encoding="utf-8").read().lower()
check("A7 the new text (header, helper) and this check have none of the banned words", not [w for w in BAD if w in head or w in own], str([w for w in BAD if w in head or w in own]))

if ARGS.pure_only:
    print("ZZ CHECK-MIDLOOK-CLOSING-PURE " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
    sys.exit(0 if not FAILED else 1)

# ==================================================================================================================== part B (pod)
for k in ("tree", "ckpt", "tracks", "eval_cache", "eval_join", "grid", "reader", "qmap", "workdir"):
    if not getattr(ARGS, k):
        print("ZZ CHECK-MIDLOOK-CLOSING STOP missing --" + k.replace("_", "-"))
        sys.exit(3)
W = ARGS.workdir
M = ARGS.replan_steps
P = ARGS.preroll_steps
DISTRULE = ARGS.distrule_dir or os.path.dirname(os.path.abspath(ARGS.new))
TIMEOUT = shutil.which("timeout")
if os.path.exists(W) and not ARGS.analyze_only:
    print("ZZ CHECK-MIDLOOK-CLOSING STOP OUTPUT-EXISTS " + W)
    sys.exit(3)
if not ARGS.dry and not os.path.exists(W):
    os.makedirs(W)

# name, wrapper, form, rule-every
RUNS = [
    ("p_RR10", ARGS.parent, "RR", 10), ("n_RR10", ARGS.new, "RR", 10),
    ("p_RRc10", ARGS.parent, "RRc", 10), ("n_RRc10", ARGS.new, "RRc", 10),
    ("p_RRh10", ARGS.parent, "RRh", 10), ("n_RRh10", ARGS.new, "RRh", 10),
    ("p_RR5", ARGS.parent, "RR", 5), ("n_RR5", ARGS.new, "RR", 5),
    ("p_RLc5", ARGS.parent, "RLc", 5), ("n_RLc5", ARGS.new, "RLc", 5),
    ("n_RRc5", ARGS.new, "RRc", 5), ("n_RRh5", ARGS.new, "RRh", 5),
]


def tag_of(name):
    return name + "_m%d_causal_heads" % M


def run_cmd(name, wrapper, form, every, out=None, extra_form=()):
    out = out or os.path.join(W, name)
    reader = ["--reader", ARGS.reader, "--qmap", ARGS.qmap] if form in ("RR", "RRc", "RRh", "RRd") else []
    cmd = [ARGS.python, wrapper, "--tree", ARGS.tree, "--form", form] + reader + list(extra_form) + [
        "--rule-log", os.path.join(out, "ticks", "rule_log_%s.npz" % name), "--rule-every", str(every),
        "--readout", "causal_g", "--readout-log", os.path.join(out, "ticks", "readout_log_%s.json" % name), "--acc-gain", "0", "--path-exec", "--kappa-max", "0.2",
        "--preroll-steps", str(P),
        "run", "--ckpt", ARGS.ckpt, "--tag", tag_of(name), "--tracks", ARGS.tracks, "--eval-cache", ARGS.eval_cache, "--eval-grid", ARGS.grid, "--eval-join", ARGS.eval_join,
        "--out-dir", out, "--replan-steps", str(M), "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal",
        "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze", "--device", ARGS.device]
    if TIMEOUT:
        cmd = [TIMEOUT, "1800"] + cmd
    return out, cmd


def env_run():
    return dict(os.environ, DISTRULE_DIR=DISTRULE, PYTHONPATH=os.path.join(ARGS.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")


def run_one(name, wrapper, form, every):
    out, cmd = run_cmd(name, wrapper, form, every)
    if ARGS.dry:
        print("CMD: DISTRULE_DIR=%s PYTHONPATH=%s %s" % (DISTRULE, os.path.join(ARGS.tree, "stack"), " ".join(cmd)))
        return 0
    if os.path.exists(out):
        if ARGS.analyze_only:
            return 0
        raise SystemExit("OUTPUT-EXISTS " + out)
    os.makedirs(os.path.join(out, "ticks"))
    with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, env=env_run(), stdout=fh, stderr=subprocess.STDOUT).returncode
    print("RUN %s wrapper=%s form=%s every=%d rc=%d" % (name, os.path.basename(wrapper), form, every, rc), flush=True)
    return rc


for name, wrapper, form, every in RUNS:
    rc = run_one(name, wrapper, form, every)
    if rc != 0:
        print("ZZ CHECK-MIDLOOK-CLOSING FAIL run-%s-rc-%d (see %s)" % (name, rc, os.path.join(W, name, "t1.log")))
        sys.exit(1)

# D: the refusal run (no output folder is needed, only the return code and the text)
REF_OUT = os.path.join(W, "refusal")
_, ref_cmd = run_cmd("refusal", ARGS.new, "RRd", 5, out=REF_OUT, extra_form=("--slope", "1.0", "--k-diff", "2"))
if TIMEOUT:
    ref_cmd[1] = "600"
if ARGS.dry:
    print("CMD: " + " ".join(ref_cmd))
    print("ZZ CHECK-MIDLOOK-CLOSING dry run: nothing executed beyond part A")
    sys.exit(0)
if os.path.exists(os.path.join(W, "refusal.log")) and ARGS.analyze_only:
    ref_text, ref_rc = open(os.path.join(W, "refusal.log"), encoding="utf-8").read(), None
else:
    cp = subprocess.run(ref_cmd, env=env_run(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ref_text, ref_rc = cp.stdout.decode("utf-8", "replace"), cp.returncode
    with open(os.path.join(W, "refusal.log"), "w", encoding="utf-8") as fh:
        fh.write(ref_text)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ticks_path(name):
    return os.path.join(W, name, "ticks", "ticks_%s.npz" % tag_of(name))


def dumps(name):
    d = os.path.join(W, name, "dump_" + tag_of(name))
    return {os.path.basename(f): load_npz(f) for f in sorted(glob.glob(os.path.join(d, "ep*.npz")))}


def tfile(name, stem):
    return os.path.join(W, name, "ticks", stem % name)


SKIP = ("tag",)                                                 # the array holding the run name differs between two runs by construction


def maxdiff_dicts(a, b, skip=SKIP, ignore_extra_in_b=()):
    """-> (largest absolute difference over the common numeric arrays, problems, keys only b has)"""
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
        elif not np.array_equal(x, y):
            prob.append("content of %s" % k)
    return worst, prob, extra


def compare_runs(pn, nn, with_mid):
    """dump arrays, tick file, rule log (and for with_mid: the mid-look log, its reader side file and the first plan file) of run pn (parent) against nn (new)."""
    worst, prob, files = 0.0, [], 0
    DP, DN = dumps(pn), dumps(nn)
    if sorted(DP) != sorted(DN) or not DP:
        prob.append("dump file lists differ or empty")
    for f in sorted(set(DP) & set(DN)):
        w_, p_, _ = maxdiff_dicts(DP[f], DN[f])
        worst, prob, files = max(worst, w_), prob + p_, files + 1
    stems = [("ticks_%s_m%d_causal_heads.npz" % ("%s", M), "tick file"), ("rule_log_%s.npz", "rule log"), ("rule_log_%s_rd.npz", "rule log side file")]
    if with_mid:
        stems += [("rule_log_mid_%s.npz", "mid-look log"), ("rule_log_mid_%s_rd.npz", "mid-look reader side file"), ("p1_first_%s.npz", "first-plan file")]
    new_keys = {}
    for stem, label in stems:
        pp, pn_ = tfile(pn, stem), tfile(nn, stem)
        ep, en = os.path.isfile(pp), os.path.isfile(pn_)
        if ep != en:
            prob.append("%s present only in %s" % (label, "parent" if ep else "new"))
            continue
        if not ep:
            continue
        w_, p_, ex = maxdiff_dicts(load_npz(pp), load_npz(pn_))
        worst, prob, files = max(worst, w_), prob + ["%s: %s" % (label, x) for x in p_], files + 1
        if ex:
            new_keys[label] = ex
    return worst, prob, files, new_keys


# ---------------------------------------------------------------- A: rule every 10
for form in ("RR", "RRc", "RRh"):
    w_, p_, nf, nk = compare_runs("p_%s10" % form, "n_%s10" % form, False)
    check("A %s --rule-every 10: dump, tick-file and rule-log arrays of the new wrapper equal the parent's, largest difference %.3g (%d files)%s" % (form, w_, nf, (" new keys " + json.dumps(nk)) if nk else ""),
          w_ == 0.0 and not p_ and nf > 0, str(p_[:3]))
# ---------------------------------------------------------------- B: rule every 5, forms RR and RLc
for form in ("RR", "RLc"):
    w_, p_, nf, nk = compare_runs("p_%s5" % form, "n_%s5" % form, True)
    check("B %s --rule-every 5: dump, tick, rule-log and mid-look-log arrays equal the parent's, largest difference %.3g (%d files); new keys of the new wrapper (not compared): %s" % (form, w_, nf, json.dumps(nk)),
          w_ == 0.0 and not p_ and nf > 0 and "mid-look log" in nk and "c_read" in nk.get("mid-look log", []), str(p_[:3]))

# ---------------------------------------------------------------- C: reach
sys.path.insert(0, DISTRULE)
sys.path.insert(0, os.path.join(ARGS.tree, "stack"))
import torch  # noqa: E402
import tok_reader as TR  # noqa: E402
import v8f_t1_eval_rule as WR  # noqa: E402

T_, _ = WR.load_pinned(ARGS.tree)
LABEL = WR.LabelSource(T_.D.load_join(ARGS.eval_join))
RD = TR.TokReader.from_ckpt(ARGS.reader)
JC = [str(c) for c in np.load(ARGS.eval_join, allow_pickle=True)["clip"]]
TOK = {}


def clip_tokens(c):
    if c not in TOK:
        d = torch.load(os.path.join(ARGS.eval_cache, "eval", "clips", JC[c] + ".pt"), map_location="cpu")
        TOK[c] = d["pool4x10"].to(torch.float16).float().numpy()
    return TOK[c]


def closing_ref(c, e):
    """The reader's closing output (m/s) at clip c, frame e, written out here (the three-output head of tok_reader, third output times 10 m/s): the five frames e-4 .. e of the cache file,
    one row at a time. NaN where the history does not exist."""
    x = clip_tokens(int(c))
    if e - (RD.hist - 1) < 0 or e >= len(x):
        return np.nan
    E = RD.frame_embed(np.asarray(x[e - RD.hist + 1:e + 1], np.float32)[None])
    z = E.reshape(1, -1)
    h = TR._gelu(z @ RD.w["head.mlp.0.weight"].T + RD.w["head.mlp.0.bias"])
    o = h @ RD.w["head.mlp.2.weight"].T + RD.w["head.mlp.2.bias"]
    return float((o[0, 2].astype(np.float64) * 10.0))


def mid(name):
    return load_npz(tfile(name, "rule_log_mid_%s.npz"))


def present_rows(L):
    return np.asarray(L["present"]).astype(bool)


def feed_test(L, expect):
    """-> (ok, largest |c_src - expect| over present rows, n present rows, n present rows with c_src != 0). expect: [n] float64 per row (NaN allowed off the present rows)."""
    pr = present_rows(L)
    cs = np.asarray(L["c_src"], np.float64)
    if not pr.any():
        return False, float("nan"), 0, 0
    d = np.abs(cs[pr] - expect[pr])
    return bool(np.all(np.isfinite(d))), float(np.max(d)), int(pr.sum()), int((cs[pr] != 0.0).sum())


def rrh_expect(L):
    return np.array([closing_ref(c, int(j)) for c, j in zip(np.asarray(L["ci"], np.int64), np.asarray(L["j"], np.int64))], np.float64)


def rrc_expect(L):
    out = np.full(len(L["ci"]), np.nan)
    for i, (c, j) in enumerate(zip(np.asarray(L["ci"], np.int64), np.asarray(L["j"], np.int64))):
        lc = float(LABEL.read(int(c), int(j))[2])
        out[i] = lc if np.isfinite(lc) else 0.0
    return out


L_RRC, L_RRH, L_RR = mid("n_RRc5"), mid("n_RRh5"), mid("n_RR5")
ok_, dm, npres, nnz = feed_test(L_RRC, rrc_expect(L_RRC))
check("C1 RRc later looks: c_src equals the label closing at (clip, frame j + q) on %d present rows (largest difference %.3g, bar 1e-9), and is not 0 on %d of them" % (npres, dm, nnz), ok_ and dm <= 1e-9 and nnz >= 1)
EXH = rrh_expect(L_RRH)
ok_, dm, npres, nnz = feed_test(L_RRH, EXH)
pr = present_rows(L_RRH)
dlog = float(np.max(np.abs(np.asarray(L_RRH["c_src"], np.float64)[pr] - np.asarray(L_RRH["c_read"], np.float64)[pr]))) if pr.any() else float("nan")
check("C2 RRh later looks: c_src equals the reader closing recomputed from the cache tokens at j + q on %d present rows (largest difference %.3g, bar %.0e), and is not 0 on %d of them; c_src equals the logged c_read, largest difference %.3g (bar 0)"
      % (npres, dm, ARGS.tol_reader, nnz, dlog), ok_ and dm <= ARGS.tol_reader and nnz >= 1 and dlog == 0.0)
cs_rr = np.asarray(L_RR["c_src"], np.float64)
check("C3 RR later looks (control): c_src is exactly 0 on every row (%d rows, largest |c_src| %.3g)" % (len(cs_rr), float(np.max(np.abs(cs_rr))) if len(cs_rr) else float("nan")),
      len(cs_rr) > 0 and bool(np.all(cs_rr == 0.0)))
cr = {}
for nm, L in (("RR", L_RR), ("RRc", L_RRC), ("RRh", L_RRH)):
    pr = present_rows(L)
    ex = rrh_expect(L)
    cr[nm] = (float(np.max(np.abs(np.asarray(L["c_read"], np.float64)[pr] - ex[pr]))) if pr.any() else float("nan"), int(pr.sum()), int((np.asarray(L["c_read"], np.float64)[pr] != 0.0).sum()))
check("C4 c_read of the later-look log equals the reader closing recomputed from the cache tokens on the present rows: %s (largest difference, present rows, rows not 0; bar %.0e)" % (json.dumps(cr), ARGS.tol_reader),
      all(np.isfinite(v[0]) and v[0] <= ARGS.tol_reader and v[2] >= 1 for v in cr.values()))
ok_p, dm_p, _, nnz_p = feed_test(L_RR, rrh_expect(L_RR))
planted("the RR mid-look log (closing 0) read as if it were RRh", bool(ok_p and dm_p <= ARGS.tol_reader), "largest difference %.3g" % dm_p)
L_NEW_RLC = mid("n_RLc5")
check("C5 label form RLc: c_read of the later-look log is NaN on every row (the label forms read no token)", len(L_NEW_RLC["ci"]) > 0 and bool(np.all(np.isnan(np.asarray(L_NEW_RLC["c_read"], np.float64)))))

# ---------------------------------------------------------------- D: refusal
check("D --rule-every 5 with form RRd is refused: non-zero return code %s, the text names RRd" % ref_rc,
      (ref_rc is None or ref_rc != 0) and "REFUSED" in ref_text and "RRd" in ref_text and "--rule-every" in ref_text, ref_text.strip()[-200:])

print("ZZ CHECK-MIDLOOK-CLOSING " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
sys.exit(0 if not FAILED else 1)
