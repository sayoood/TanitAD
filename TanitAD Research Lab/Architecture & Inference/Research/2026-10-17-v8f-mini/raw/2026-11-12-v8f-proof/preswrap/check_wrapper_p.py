"""check_wrapper_p.py -- check of the wrapper v8f_t1_eval_reader_rule_rzgfhqetjvp.py (study "preswrap"), after visrule/check_wrapper_v.py.
Part S (static, standard library and numpy): the source has the asserted hash (LF text, prefix 28bc31f21ecb4ccb); the generator rebuilds the variant text byte for byte from the source; the variant
  compiles; the record file (when present) names the same hashes and says pass; static facts of the text (flags, refusals, p2 before the presence tests on all three paths, the logs); no banned words.
Part T (selftest): the helper functions of the generator run on their own on tiny synthetic arrays: both modes, the fallback where the file has no value, the counts, every refusal of the loader.
Part I (identity, needs numpy and the distrule folder, env DISTRULE_DIR): the source and the variant are imported and head_values_reader is run on a synthetic world:
  I1 flags absent: every output and log column is the source's, no p_used column; I2 a file without any usable value: the source's values in both modes (p_used = the reader's p);
  I3 mode head, I4 mode or: the returned p is p2, the column p is the reader's own, p_used is p2, the gap is the source's, the counts add up;
  I5 with --vis-gap-file and --vis-closing: the variant on the real p equals the source run on a world whose p column was replaced by p2 (gap and closing input equal; this reads the earlier frames of the run too);
  I6 the new flags are refused outside form RR and one without the other.
Last line: ZZ CHECK-WRAPPER-P PASS or ZZ CHECK-WRAPPER-P FAIL ...
Usage: python check_wrapper_p.py [--src SRC.py] [--new NEW.py] [--record REC.json] [--static-only | --identity-only | --selftest]"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DISTRULE_DIR", os.path.normpath(os.path.join(HERE, "..", "distrule")))
import make_wrapper_p as MWP  # noqa: E402

BAD = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
TMPBASE = "F:/TanitAD-scratch/v8proof/preswrap_tmp"


# ============================================================================================ part S
def _before(src, a, b):
    i, j = src.find(a), src.find(b)
    return i >= 0 and j >= 0 and i < j


def static_facts(src):
    return [("S1 flag --vis-pres-file exists, default empty", 'ap.add_argument("--vis-pres-file", default="")' in src),
            ("S2 flag --vis-pres-mode exists with the choices head and or", 'ap.add_argument("--vis-pres-mode", choices=("head", "or"), default="")' in src),
            ("S3 both belong to form RR only (refusal text)", "--vis-pres-file and --vis-pres-mode belong to form RR only" in src and '(mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RR"' in src),
            ("S4 file and mode come together (two refusals)", "--vis-pres-file needs --vis-pres-mode head or or" in src and "--vis-pres-mode needs --vis-pres-file" in src),
            ("S5 the state is None without the file", 'if mine.vis_pres_file else None' in src and 'STATE["pres"] = load_pres(' in src),
            ("S6 tick level: p2 replaces p before feed() and the logged column is the reader's own", _before(src, "p = pres_tick(p)", 'present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])')
             and '(("p", p_rd), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c))' in src),
            ("S7 later look: p2 replaces p before feed(), vis_look and the presence test", _before(src, "p = pres_look(st, q, p)", "_, used = feed(form, p, g, okr") and _before(src, "p = pres_look(st, q, p)", "hp = (p >= W.HEAD_P_MIN) & okr")
             and '(("p", p_rd), ("g_read", g), ("g_used", used), ("ok", okr))' in src),
            ("S8 closing run: the earlier frames are replaced too, not counted", "pe = pres_look(st, int(e_off) - d, pe, False)" in src),
            ("S9 the states are bound for the file as for the gap file", 'if STATE.get("vis") is not None or STATE.get("pres") is not None:\n            STATE["vis_states"] = self.states' in src),
            ("S10 p_used only with the file, tick and later look", 'STATE["cols"].setdefault("p_used", []).append(np.asarray(p))' in src and 'M["rd"].setdefault("p_used", []).append(np.asarray(p))' in src),
            ("S11 run record fields only with the file", "**pres_meta(mine.vis_pres_file, mine.vis_pres_mode," in src and '"vis_pres_file_sha256": sha, "vis_pres_mode": mode, "vis_pres_counts"' in src),
            ("S12 the run-line fields are printed at load", 'print("pres run-line fields:" + pres_fields(' in src),
            ("S13 none of the banned words in the text", not [w for w in BAD if w in src])]


def static_check(src_path, new_path, record_path=None):
    """-> list of (name, bool, detail)"""
    res = []
    raw = open(src_path, "rb").read()
    res.append(("T1 source: LF text with the asserted sha256 prefix", b"\r" not in raw and hashlib.sha256(raw).hexdigest()[:16] == MWP.PARENT_SHA16, hashlib.sha256(raw).hexdigest()[:16]))
    new_raw = open(new_path, "rb").read()
    try:
        t, anchors, _ = MWP.build(raw.decode("utf-8"))
        same = t.encode("utf-8") == new_raw
        res.append(("T2 the generator rebuilds the variant byte for byte (%d edits)" % len(anchors), same, ""))
    except AssertionError as e:
        res.append(("T2 the generator rebuilds the variant byte for byte", False, str(e)[:120]))
    try:
        compile(new_raw.decode("utf-8"), new_path, "exec")
        res.append(("T3 the variant compiles", True, ""))
    except SyntaxError as e:
        res.append(("T3 the variant compiles", False, str(e)))
    res.append(("T4 the variant holds no carriage return", b"\r" not in new_raw, ""))
    for n, ok in static_facts(new_raw.decode("utf-8")):
        res.append((n, bool(ok), ""))
    if record_path and os.path.isfile(record_path):
        rec = json.load(open(record_path, encoding="utf-8"))
        okr = rec.get("source_sha256", "")[:16] == MWP.PARENT_SHA16 and rec.get("variant_sha256") == hashlib.sha256(new_raw).hexdigest() and rec.get("pass") is True
        res.append(("T5 the record names the source and the variant hashes and says pass", bool(okr), ""))
    return res


# ============================================================================================ part T: the helpers on their own
def _world():
    clips = np.array(["a", "b"])
    off, n = np.array([0, 4], np.int64), np.array([4, 3], np.int64)
    p = np.array([0.1, 0.9, np.nan, 0.5, 0.7, 0.3, np.nan])
    return {"clip": clips, "off": off, "n": n}, p


def _write(path, join, pv, **over):
    d = dict(p=pv, has_out=np.isfinite(pv), clips=join["clip"], off=join["off"], n=join["n"], seed=np.int64(0))
    d.update(over)
    d = {k: v for k, v in d.items() if v is not None}
    with open(path, "wb") as fh:
        np.savez(fh, **d)
    return path


def selftest():
    ok, bad = True, []

    def chk(name, cond):
        nonlocal ok
        if not cond:
            ok = False
            bad.append(name)

    ns = MWP.helper_ns()
    join, pf = _world()
    base = tempfile.mkdtemp(prefix="pres_selftest_", dir=TMPBASE) if os.path.isdir(TMPBASE) else tempfile.mkdtemp(prefix="pres_selftest_")
    good = _write(os.path.join(base, "good.npz"), join, pf)
    ci = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, -1, 2])
    fr = np.array([0, 1, 2, 3, 0, 1, 2, 4, 3, 0, 0])
    p = np.array([0.6, 0.4, 0.8, 0.5, 0.2, 0.9, 0.1, 0.7, 0.7, 0.4, 0.4])
    exp_head = np.array([0.1, 0.9, 0.8, 0.5, 0.7, 0.3, 0.1, 0.7, 0.7, 0.4, 0.4])
    exp_or = np.array([0.6, 0.9, 0.8, 0.5, 0.7, 0.9, 0.1, 0.7, 0.7, 0.4, 0.4])
    for mode, exp, raised, lowered in (("head", exp_head, 2, 2), ("or", exp_or, 2, 0)):
        S = ns["load_pres"](good, mode, join)
        cnt = {"rows": 0, "with_file": 0, "no_file": 0, "raised": 0, "lowered": 0}
        p2 = ns["pres_replace"](S, ci, fr, p, cnt)
        chk("replace " + mode, np.array_equal(p2, exp))
        chk("counts " + mode, cnt == {"rows": 11, "with_file": 5, "no_file": 6, "raised": raised, "lowered": lowered})
    chk("no state: a copy of p", ns["pres_replace"](None, ci, fr, p) is not p and np.array_equal(ns["pres_replace"](None, ci, fr, p), p))
    S = ns["load_pres"](good, "or", join)
    chk("or: p not a number takes the file value", ns["pres_replace"](S, [0], [0], [np.nan])[0] == 0.1)
    chk("or: p not a number and no file value stays not a number", np.isnan(ns["pres_replace"](S, [0], [2], [np.nan])[0]))
    chk("exact 0.5 stays present", ns["pres_replace"](ns["load_pres"](good, "head", join), [0], [3], [0.2])[0] >= 0.5)
    # the meta and the line only with the flag
    chk("meta empty without the flag", ns["pres_meta"]("", "", "") == {})
    m = ns["pres_meta"]("f.npz", "head", "ab")
    chk("meta keys", sorted(m) == ["vis_pres_counts", "vis_pres_file", "vis_pres_file_sha256", "vis_pres_mode"])
    chk("fields text", ns["pres_fields"]("f.npz", "ab", "or") == " vis-pres-file f.npz vis pres sha256 ab vis-pres-mode or")

    def refused(label, fn):
        try:
            fn()
        except SystemExit as e:
            chk("refusal " + label + " has the text", "REFUSED" in str(e))
            return
        chk("refusal " + label, False)

    refused("mode", lambda: ns["load_pres"](good, "avg", join))
    refused("empty mode", lambda: ns["load_pres"](good, "", join))
    refused("missing file", lambda: ns["load_pres"](os.path.join(base, "none.npz"), "head", join))
    refused("empty path", lambda: ns["load_pres"]("", "head", join))
    refused("no join", lambda: ns["load_pres"](good, "head", None))
    for k in ("p", "has_out", "clips", "off", "n"):
        refused("array " + k + " missing", lambda k=k: ns["load_pres"](_write(os.path.join(base, "no_%s.npz" % k), join, pf, **{k: None}), "head", join))
    refused("length of has_out", lambda: ns["load_pres"](_write(os.path.join(base, "len.npz"), join, pf, has_out=np.ones(6, bool)), "head", join))
    refused("p two-dimensional", lambda: ns["load_pres"](_write(os.path.join(base, "dim.npz"), join, pf, p=pf.reshape(7, 1), has_out=np.isfinite(pf).reshape(7, 1)), "head", join))
    refused("off differs", lambda: ns["load_pres"](_write(os.path.join(base, "off.npz"), join, pf, off=np.array([0, 5])), "head", join))
    refused("n differs", lambda: ns["load_pres"](_write(os.path.join(base, "n.npz"), join, pf, n=np.array([4, 2])), "head", join))
    refused("clip names differ", lambda: ns["load_pres"](_write(os.path.join(base, "clips.npz"), join, pf, clips=np.array(["a", "c"])), "head", join))
    refused("rows differ", lambda: ns["load_pres"](_write(os.path.join(base, "rows.npz"), join, pf[:6], has_out=np.isfinite(pf[:6])), "head", join))
    refused("p above 1", lambda: ns["load_pres"](_write(os.path.join(base, "hi.npz"), join, np.where(np.arange(7) == 1, 1.2, pf)), "head", join))
    refused("p below 0", lambda: ns["load_pres"](_write(os.path.join(base, "lo.npz"), join, np.where(np.arange(7) == 1, -0.1, pf)), "head", join))
    refused("output marked where p is not finite", lambda: ns["load_pres"](_write(os.path.join(base, "has.npz"), join, pf, has_out=np.ones(7, bool)), "head", join))
    print("ZZ CHECK-WRAPPER-P-SELFTEST " + ("PASS" if ok else "FAIL " + "; ".join(bad)))
    return ok, bad


# ============================================================================================ part I: identity on synthetic inputs
class FakeReader:
    """tokens [B, hist, F]: feature 0 of the last frame is the gap read, feature 1 the presence probability."""
    hist = 5

    def read(self, toks):
        t = np.asarray(toks, np.float64)
        return t[:, -1, 1].astype(np.float32), t[:, -1, 0].astype(np.float32)


class FakeState:
    def __init__(self, ci, j, feats, p10f):
        self.ci, self.j = ci, j
        self.d = {"feats": feats, "p10f": p10f}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def setup_module(m):
    m.reader_closing = lambda rd, tokens: np.zeros(len(tokens), np.float32)
    m.STATE["reader"] = FakeReader()
    m.STATE["xs"], m.STATE["ys"] = np.array([0.0, 100.0]), np.array([0.0, 90.0])
    m.STATE["form"] = "RR"
    m.STATE["cols"] = {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}
    m.STATE["vis"] = None
    m.STATE["vis_counts"] = {k: 0 for k in m.STATE["vis_counts"]}
    if "pres" in m.STATE:
        m.STATE["pres"] = None
        m.STATE["pres_counts"] = {k: 0 for k in m.STATE["pres_counts"]}


def synth_world(seed=3, n_clips=3, n=40):
    rng = np.random.RandomState(seed)
    clips = []
    for c in range(n_clips):
        g = np.clip(30.0 - 0.2 * np.arange(n) * (c + 1) + rng.randn(n) * 0.3, 0.5, 70.0)
        g[12] += 6.0
        p = np.where(rng.rand(n) < 0.3, 0.2, 0.9)
        clips.append((np.stack([g, p], 1).astype(np.float32), np.zeros((n, 3))))
    return clips


def batch_of(clips, frames):
    sts, feats = [], []
    for c, (f, p10) in enumerate(clips):
        for j in frames:
            sts.append(FakeState(c, j, f, p10))
            feats.append(f[j - 4:j + 1])
    return sts, np.stack(feats)


def same_out(a, b):
    return all(np.array_equal(np.asarray(x, np.float64), np.asarray(y, np.float64), equal_nan=True) for x, y in zip(a, b))


def identity_checks(src_path, new_path):
    """-> list of (name, bool, detail)"""
    res = []
    old, new = load_module(src_path, "wp_old"), load_module(new_path, "wp_new")
    n = 40
    clips = synth_world()
    nc = len(clips)
    frames = list(range(4, n))
    join = {"clip": np.array(["c%d" % i for i in range(nc)]), "off": np.arange(nc, dtype=np.int64) * n, "n": np.full(nc, n, np.int64)}
    tmp = tempfile.mkdtemp(prefix="pres_identity_", dir=TMPBASE) if os.path.isdir(TMPBASE) else tempfile.mkdtemp(prefix="pres_identity_")
    rng = np.random.RandomState(11)
    pfile = rng.uniform(0.3, 1.0, nc * n)
    pfile[rng.rand(nc * n) < 0.3] = np.nan
    fin = np.isfinite(pfile)
    pf_path = _write(os.path.join(tmp, "pres_%d.npz" % os.getpid()), join, pfile)
    pn_path = _write(os.path.join(tmp, "presnone_%d.npz" % os.getpid()), join, np.full(nc * n, np.nan))

    for m in (old, new):
        setup_module(m)
    sts_o, feats = batch_of(clips, frames)
    sts_n, _ = batch_of(clips, frames)
    r_old = old.head_values_reader(None, feats, None, None)
    r_new = new.head_values_reader(None, feats, None, None)
    same = same_out(r_old, r_new) and all(np.array_equal(old.STATE["cols"][k][0], new.STATE["cols"][k][0], equal_nan=True) for k in old.STATE["cols"])
    res.append(("I1 flags absent: outputs and log columns are the source's, no p_used column", bool(same) and "p_used" not in new.STATE["cols"] and r_new[0].dtype == r_old[0].dtype, ""))

    ci = np.array([s.ci for s in sts_n])
    jj = np.array([s.j for s in sts_n])
    row = ci * n + jj
    p_read = feats[:, -1, 1].astype(np.float64)
    for mode in ("head", "or"):
        setup_module(new)
        new.STATE["pres"] = new.load_pres(pn_path, mode, join)
        new.STATE["vis_states"] = sts_n
        r0 = new.head_values_reader(None, feats, None, None)
        okc = same_out(r_old[1:], r0[1:]) and np.array_equal(r0[0], r_old[0].astype(np.float64)) and np.array_equal(new.STATE["cols"]["p_used"][0], p_read) and np.array_equal(new.STATE["cols"]["p"][0], old.STATE["cols"]["p"][0])
        cnt = new.STATE["pres_counts"]
        res.append(("I2 mode %s, a file without any usable value: the source's values, p_used = the reader's p, every row counted as without value" % mode,
                    bool(okc) and cnt["no_file"] == len(sts_n) and cnt["with_file"] == 0 and cnt["raised"] == 0 and cnt["lowered"] == 0, str(cnt)))
    for tag, mode in (("I3", "head"), ("I4", "or")):
        setup_module(new)
        new.STATE["pres"] = new.load_pres(pf_path, mode, join)
        new.STATE["vis_states"] = sts_n
        r = new.head_values_reader(None, feats, None, None)
        pfr = pfile[row]
        exp = np.where(fin[row], pfr if mode == "head" else np.fmax(p_read, pfr), p_read)
        cnt = new.STATE["pres_counts"]
        er = int(((p_read < 0.5) & (exp >= 0.5)).sum())
        el = int(((p_read >= 0.5) & (exp < 0.5)).sum())
        good = (np.array_equal(r[0], exp) and np.array_equal(new.STATE["cols"]["p"][0], old.STATE["cols"]["p"][0]) and np.array_equal(new.STATE["cols"]["p_used"][0], exp)
                and same_out(r_old[1:], r[1:]) and cnt == {"rows": len(sts_n), "with_file": int(fin[row].sum()), "no_file": int((~fin[row]).sum()), "raised": er, "lowered": el} and er > 0)
        res.append(("%s mode %s: returned p is p2, column p is the reader's own, p_used is p2, gap and history flag are the source's, counts add up (raised %d lowered %d)" % (tag, mode, er, el), bool(good), str(cnt)))

    # I5: with the gap file and the closing input, the variant on the real p equals the source on a world whose p column was replaced by p2
    gap = np.concatenate([c[0][:, 0].astype(np.float64) + 1.0 for c in clips])        # smooth, so that runs of at least 3 frames exist for the closing slope
    has = rng.rand(nc * n) < 0.9
    gf = os.path.join(tmp, "gap_%d.npz" % os.getpid())
    np.savez(gf, clip_index=np.repeat(np.arange(nc), n), frame=np.tile(np.arange(n), nc), gap=np.where(has, gap, np.nan), has_out=has)
    for mode in ("head", "or"):
        clips2 = []
        for c, (f, p10) in enumerate(clips):
            pr = f[:, 1].astype(np.float64)
            pc = pfile[c * n:(c + 1) * n]
            p2 = np.where(np.isfinite(pc), pc if mode == "head" else np.fmax(pr, pc), pr)
            f2 = f.copy()
            f2[:, 1] = p2.astype(np.float32)
            clips2.append((f2, p10))
        sts2, feats2 = batch_of(clips2, frames)
        setup_module(old)
        old.STATE["vis"] = old.load_vis(gf, True)
        old.STATE["vis_states"] = sts2
        r_ref = old.head_values_reader(None, feats2, None, None)
        setup_module(new)
        new.STATE["vis"] = new.load_vis(gf, True)
        new.STATE["pres"] = new.load_pres(pf_path, mode, join)
        new.STATE["vis_states"] = sts_n
        r_p = new.head_values_reader(None, feats, None, None)
        pres_ref = r_ref[0].astype(np.float64)
        near = np.allclose(r_p[0], pres_ref, atol=1e-6, rtol=0) and np.array_equal(r_p[1], r_ref[1]) and np.array_equal(r_p[2], r_ref[2]) and np.array_equal(r_p[3], r_ref[3])
        res.append(("I5 mode %s with --vis-gap-file and --vis-closing: gap and closing input equal the source's on the world with p2 as its p column (%d non-zero closing rows)" % (mode, int((r_p[2] != 0).sum())),
                    bool(near) and bool((r_p[2] != 0).any()), ""))
    # I6: refusals of the arguments (they come before the pinned tool is loaded)
    bad = []
    for label, args, text in (("form RRc with the file", ["--form", "RRc", "--vis-pres-file", "x.npz", "--vis-pres-mode", "head"], "belong to form RR only"),
                              ("form RR0 with the mode only", ["--form", "RR0", "--vis-pres-mode", "or"], "belong to form RR only"),
                              ("file without mode", ["--form", "RR", "--vis-pres-file", "x.npz"], "needs --vis-pres-mode"),
                              ("mode without file", ["--form", "RR", "--vis-pres-mode", "head"], "needs --vis-pres-file")):
        try:
            new._main_inner(["--tree", "x"] + args)
            bad.append(label + ": not refused")
        except SystemExit as e:
            if "REFUSED" not in str(e) or text not in str(e):
                bad.append(label + ": wrong text")
        except Exception as e:                               # the refusal did not come first
            bad.append(label + ": " + type(e).__name__)
    res.append(("I6 the new flags are refused outside form RR and one without the other", not bad, str(bad)))
    return res


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--src", default=MWP.SRC)
    ap.add_argument("--new", default=MWP.OUT)
    ap.add_argument("--record", default=MWP.RECORD)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    os.makedirs(TMPBASE, exist_ok=True)
    rows = []
    if A.selftest:
        ok, bad = selftest()
        print("ZZ CHECK-WRAPPER-P " + ("PASS" if ok else "FAIL " + "; ".join(bad)))
        return 0 if ok else 1
    if not A.identity_only:
        rows += static_check(A.src, A.new, A.record)
    if not (A.static_only or A.identity_only):
        ok, bad = selftest()
        rows.append(("T-selftest the helpers on tiny arrays", ok, "; ".join(bad)))
    if not A.static_only:
        rows += identity_checks(A.src, A.new)
    failed = [n for n, ok, _ in rows if not ok]
    for n, ok, d in rows:
        print(("PASS " if ok else "FAIL ") + n + ((" -- " + d) if d and not ok else ""))
    print("ZZ CHECK-WRAPPER-P " + ("PASS" if not failed else "FAIL " + "; ".join(x.split(" ")[0] for x in failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
