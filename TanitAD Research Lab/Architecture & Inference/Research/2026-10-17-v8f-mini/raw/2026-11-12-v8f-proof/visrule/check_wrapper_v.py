"""check_wrapper_v.py -- check of the wrapper v8f_t1_eval_reader_rule_rzgfhqetjv.py (card ADD102, control v1 in part), after gaprule/check_wrapper_j.py.
Part A (no GPU, no tool): static facts of the text; import of the old and the new wrapper; the identity check: with the new arguments off the changed functions return what the old ones return on
  synthetic inputs (head_values_reader, with and without a gap table whose every row has no output); the gap of the file on the frames where the reader marks a lead and the file has an output; the
  closing input equals the closing of LabelSource on a synthetic join (the run rule); the refusals. Needs numpy and the folder of the distrule modules (env DISTRULE_DIR) only.
Part B (pod): small rollouts of form RR (model seed 0, reader seed 0, readout K, --rule-every 5, pre-roll of 30 steps (6 s), the first --n-clips clips of the pre-roll grid) with
    jv_old    the source wrapper (rzgfhqetj, hash cbf7063d867c9f4e)
    jv_new    the variant with the new arguments off
    jv_nogap  the variant with --vis-gap-file of a file whose every row has no output (the rule must receive the reader's own gap: equal to jv_old)
    jv_vis    the variant with --vis-gap-file GAPFILE --vis-closing                 [only with --vgap]
 and two refusals (a missing gap file; --vis-closing with form RLc). Checks: V2A jv_new equals jv_old (every numeric array of every npz file, largest difference 0); V2B jv_nogap equals jv_old the same way;
 V2C on jv_vis, in the rule log and the later-look log, every row with a lead and a file output has the gap of the file within 1e-3 m, and every row with a lead and no file output has the gap of jv_old at
 the same (clip, frame); V2D on jv_vis the closing input is not zero on some row and is zero on every row without a lead; V2E the refusals have a non-zero return code and the text REFUSED.
Writes <workdir>/CHECK_WRAPPER_V_DONE.txt (last line ZZ CHECK-WRAPPER-V PASS or FAIL ...) and CHECK_WRAPPER_V.hashes. No removal, no move, no kill.
Usage: python check_wrapper_v.py --old OLD.py --new NEW.py --tree TREE --ckpt CKPT --reader RDR --qmap Q.json --tracks T.pt --eval-cache DIR --eval-join J.npz --grid GRID_P30.json --workdir NEWDIR
          [--vgap GAPFILE.npz] [--distrule-dir DIR] [--closest-dir DIR] [--python PY] [--device cuda] [--n-clips 3] [--dry]  |  --identity-only --old OLD.py --new NEW.py  |  --pure-only --new NEW.py  |  --selftest"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

BAD = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
SKIP_KEYS = ("tag", "rule_join", "rule_join_sha256", "vis_gap_file", "vis_gap_file_sha256", "vis_closing", "vis_counts")


# ============================================================================================ comparison of two runs (as check_wrapper_j.py)
def maxdiff(a, b, skip=SKIP_KEYS):
    a = {k: v for k, v in a.items() if k not in skip}
    b = {k: v for k, v in b.items() if k not in skip}
    worst, prob = 0.0, []
    if sorted(a) != sorted(b):
        prob.append("keys differ %s" % (sorted(set(a) ^ set(b))[:4],))
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            prob.append("shape of %s" % k)
        elif x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            if not np.array_equal(np.isnan(xf), np.isnan(yf)):
                prob.append("nan pattern of %s" % k)
            elif xf.size and np.isfinite(xf).any():
                worst = max(worst, float(np.nanmax(np.abs(xf - yf))))
        elif x.dtype.kind not in "US" and not np.array_equal(x, y):
            prob.append("content of %s" % k)
    return worst, prob


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def compare_runs(dir_a, name_a, dir_b, name_b):
    files, worst, prob = 0, 0.0, []
    for pa in sorted(glob.glob(os.path.join(dir_a, "**", "*.npz"), recursive=True)):
        rel = os.path.relpath(pa, dir_a)
        pb = os.path.join(dir_b, rel.replace(name_a, name_b))
        if not os.path.isfile(pb):
            prob.append("missing %s" % rel)
            continue
        d, p = maxdiff(load_npz(pa), load_npz(pb))
        files += 1
        worst = max(worst, d)
        prob += ["%s: %s" % (rel, x) for x in p]
    return {"files": files, "largest_diff": worst, "problems": prob}


def static_facts(src):
    return [("A1 flag --vis-gap-file exists", 'ap.add_argument("--vis-gap-file", default="")' in src),
            ("A2 switch --vis-closing exists and is off by default", 'ap.add_argument("--vis-closing", action="store_true")' in src),
            ("A3 both belong to form RR only (refusal text)", "--vis-gap-file and --vis-closing belong to form RR only" in src and '(mine.vis_gap_file or mine.vis_closing) and mine.form != "RR"' in src),
            ("A4 the state is None when both are off", 'STATE["vis"] = load_vis(mine.vis_gap_file, mine.vis_closing) if (mine.vis_gap_file or mine.vis_closing) else None' in src),
            ("A5 the tick-level gap is replaced after feed() and before the log columns", src.index("used, vis_c = vis_tick(p, g, used, ok)") < src.index('for k, v in (("p", p), ("g_read", g), ("g_used", used)')),
            ("A6 the later look replaces the gap after feed()", "used, vis_hc = vis_look(st, rd, q, p, g, used, okr)" in src),
            ("A7 the closing source is the head slot only with --vis-closing", 'if mine.vis_closing:' in src and 'mix = (mix[0], mix[1], "head")' in src),
            ("A8 the namespace of the pinned tool keeps --eval-join (no assignment to a.eval_join)", "a.eval_join =" not in src and "a.eval_join=" not in src),
            ("A9 the run record has the four new keys", all(k in src for k in ('"vis_gap_file": mine.vis_gap_file', '"vis_gap_file_sha256"', '"vis_closing": int(bool(mine.vis_closing))', '"vis_counts"'))),
            ("A10 the label source is not read by the variant's helpers", "label" not in src[src.index("VIS_STRIDE = 1 << 20"):src.index("def build_rule_label(")].replace("LabelSource", "").replace("labels", "")),
            ("A11 none of the banned words in the text", not [w for w in BAD if w in src])]


# ============================================================================================ part A: identity on synthetic inputs
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
    m.reader_closing = lambda rd, tokens: np.zeros(len(tokens), np.float32)           # the reader's own closing output is a stub: it is logged, never fed in form RR
    m.STATE["reader"] = FakeReader()
    m.STATE["xs"], m.STATE["ys"] = np.array([0.0, 100.0]), np.array([0.0, 90.0])      # a map with a visible minimum: min(g, Q(g)) = 0.9 g
    m.STATE["form"] = "RR"
    m.STATE["cols"] = {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}
    if "vis_counts" in m.STATE:
        m.STATE["vis_counts"] = {k: 0 for k in m.STATE["vis_counts"]}


def synth_world(seed=3, n_clips=3, n=40):
    """-> states per (clip, frame >= 4), features [n, 2] per clip: gap, probability."""
    rng = np.random.RandomState(seed)
    clips = []
    for c in range(n_clips):
        g = np.clip(30.0 - 0.2 * np.arange(n) * (c + 1) + rng.randn(n) * 0.3, 0.5, 70.0)
        g[12] += 6.0                                  # a step above 3 m
        g[18:24] = [59.0, 60.5, 62.0, 61.0, 59.5, 58.0]   # a stretch partly outside 0 to 60 m, neighbours within 3 m of each other: the range test decides, not the step test
        p = np.where(rng.rand(n) < 0.15, 0.2, 0.9)    # frames the reader does not mark
        clips.append((np.stack([g, p], 1).astype(np.float32), np.zeros((n, 3))))
    return clips


def batch_of(m, clips, frames):
    sts, feats = [], []
    for c, (f, p10) in enumerate(clips):
        for j in frames:
            sts.append(FakeState(c, j, f, p10))
            feats.append(f[j - 4:j + 1])
    return sts, np.stack(feats)


def identity_checks(old_path, new_path):
    """-> list of (name, bool, detail)"""
    res = []
    old, new = load_module(old_path, "wv_old"), load_module(new_path, "wv_new")
    clips = synth_world()
    frames = list(range(4, 40))
    for m in (old, new):
        setup_module(m)
    sts_o, feats = batch_of(old, clips, frames)
    sts_n, _ = batch_of(new, clips, frames)
    r_old = old.head_values_reader(None, feats, None, None)
    r_new = new.head_values_reader(None, feats, None, None)
    same = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(r_old, r_new)) and all(np.array_equal(old.STATE["cols"][k][0], new.STATE["cols"][k][0], equal_nan=True) for k in old.STATE["cols"])
    res.append(("I1 head_values_reader with the new arguments off returns and logs what the old one does", bool(same), ""))
    # a gap table whose every row has no output: the rule receives the reader's own gap
    ci = np.repeat(np.arange(len(clips)), 40)
    fr = np.tile(np.arange(40), len(clips))
    tmp = tempfile.mkdtemp(prefix="vis_identity_")         # synthetic gap files of the identity part (small; left in place)
    pn = os.path.join(tmp, "noout_%d.npz" % os.getpid())
    np.savez(pn, clip_index=ci, frame=fr, gap=np.full(len(ci), np.nan), has_out=np.zeros(len(ci), bool))
    setup_module(new)
    new.STATE["vis"] = new.load_vis(pn, False)
    new.STATE["vis_states"] = sts_n
    r_no = new.head_values_reader(None, feats, None, None)
    res.append(("I2 a gap table with no output on any row: the rule receives the old gap and the closing input is the old zero", all(np.array_equal(a, b, equal_nan=True) for a, b in zip(r_old, r_no)) and not np.any(r_no[2]), ""))
    # a table with outputs: the file's gap on the marked frames with an output, the reader's own elsewhere
    rng = np.random.RandomState(5)
    gap = rng.uniform(2.0, 59.0, len(ci))
    has = rng.rand(len(ci)) < 0.7
    pg = os.path.join(tmp, "without_%d.npz" % os.getpid())
    np.savez(pg, clip_index=ci, frame=fr, gap=np.where(has, gap, np.nan), has_out=has)
    setup_module(new)
    new.STATE["vis"] = new.load_vis(pg, False)
    new.STATE["vis_states"] = sts_n
    r_g = new.head_values_reader(None, feats, None, None)
    p_all = feats[:, -1, 1]
    marked = p_all >= 0.5
    exp = r_old[1].astype(np.float64).copy()
    for i, s in enumerate(sts_n):
        k = s.ci * 40 + s.j
        if marked[i] and has[k]:
            exp[i] = gap[k]
    res.append(("I3 on a frame the reader marks and the file has an output for, the rule receives the file's gap; elsewhere the reader's own", bool(np.allclose(r_g[1], exp.astype(np.float32), atol=0, rtol=0)) and not np.any(r_g[2]), ""))
    res.append(("I4 the reader's presence output and the full-history flag are those of the old wrapper", bool(np.array_equal(r_g[0], r_old[0]) and np.array_equal(r_g[3], r_old[3])), ""))
    # the closing input equals the closing of LabelSource on a synthetic join (the run rule), for a file that carries the gap of the join on every frame
    n = 40
    off = np.arange(len(clips)) * n
    lg = np.concatenate([c[0][:, 0] for c in clips]).astype(np.float32)
    lv = np.ones(len(lg), np.int64)
    for c in range(len(clips)):
        lv[off[c]:off[c] + 4] = 0                     # the reader has no output before frame 4: the label marks no lead there either
    pl = np.where(np.concatenate([c[0][:, 1] for c in clips]) >= 0.5, 1, 0)
    lv = lv * pl                                      # the label marks a lead exactly where the reader does
    J = {"off": off, "n": np.full(len(clips), n), "lead_valid": lv, "lead_gap": lg}
    LS = old.W.LabelSource(J)
    pc = os.path.join(tmp, "close_%d.npz" % os.getpid())
    np.savez(pc, clip_index=ci, frame=fr, gap=lg.astype(np.float64), has_out=np.ones(len(ci), bool))
    setup_module(new)
    new.STATE["vis"] = new.load_vis(pc, True)
    new.STATE["vis_states"] = sts_n
    r_c = new.head_values_reader(None, feats, None, None)
    bad = []
    nz = 0
    for i, s in enumerate(sts_n):
        pres, g, c = LS.read(s.ci, s.j)
        if r_c[2][i] != 0:
            nz += 1
        if pres and abs(float(r_c[2][i]) - c) > 1e-5:
            bad.append((s.ci, s.j, float(r_c[2][i]), c))
        if not pres and r_c[2][i] != 0 and marked[i] and not (0.0 <= lg[off[s.ci] + s.j] <= 60.0):
            bad.append((s.ci, s.j, "closing on a frame outside 0 to 60 m", float(r_c[2][i])))
    res.append(("I5 the closing input equals the closing of LabelSource on the same gaps (%d non-zero rows)" % nz, not bad and nz > 0, str(bad[:3])))
    # counts
    cnt = new.STATE["vis_counts"]
    res.append(("I6 the counters are consistent", cnt["present"] == int(marked.sum()) and cnt["from_head"] == cnt["present"] and cnt["closing_rows"] == len(sts_n), str(cnt)))
    # refusals of the loader
    bad_files = {"missing": os.path.join(tmp, "no_such.npz")}
    pdup = os.path.join(tmp, "dup_%d.npz" % os.getpid())
    np.savez(pdup, clip_index=np.array([0, 0]), frame=np.array([1, 1]), gap=np.array([1.0, 2.0]), has_out=np.array([True, True]))
    pnan = os.path.join(tmp, "nan_%d.npz" % os.getpid())
    np.savez(pnan, clip_index=np.array([0]), frame=np.array([1]), gap=np.array([np.nan]), has_out=np.array([True]))
    pkey = os.path.join(tmp, "key_%d.npz" % os.getpid())
    np.savez(pkey, clip_index=np.array([0]), frame=np.array([1]), gap=np.array([1.0]))
    refused = []
    for nm, p in (("missing", bad_files["missing"]), ("duplicate", pdup), ("nan", pnan), ("key", pkey)):
        try:
            new.load_vis(p, False)
            refused.append(nm + ":not-refused")
        except SystemExit as e:
            if "REFUSED" not in str(e):
                refused.append(nm + ":no-text")
    res.append(("I7 a missing, duplicated, non-finite or incomplete gap file is refused", not refused, str(refused)))
    return res


# ============================================================================================ part B helpers
def key_gap(log, skip_absent=True):
    """{(ci, j): g_src} of the rows with a lead of a rule log or later-look log (npz dict)."""
    out = {}
    pres = np.asarray(log["present"], bool)
    for c, j, g in zip(np.asarray(log["ci"])[pres], np.asarray(log["j"])[pres], np.asarray(log["g_src"], np.float64)[pres]):
        out[(int(c), int(j))] = float(g)
    return out


def vis_log_checks(new_dir, new_name, old_dir, old_name, gapfile):
    """V2C and V2D from the logs of jv_vis against the gap file and jv_old. -> (ok_c, ok_d, detail)"""
    z = load_npz(gapfile)
    tab = {(int(c), int(f)): (float(g), bool(h)) for c, f, g, h in zip(z["clip_index"], z["frame"], z["gap"], z["has_out"])}
    paths = [os.path.join(new_dir, "ticks", "rule_log_%s.npz" % new_name), os.path.join(new_dir, "ticks", "rule_log_mid_%s.npz" % new_name)]
    olds = [os.path.join(old_dir, "ticks", "rule_log_%s.npz" % old_name), os.path.join(old_dir, "ticks", "rule_log_mid_%s.npz" % old_name)]
    n_head = n_noout = n_bad = n_cmp = n_cmp_bad = 0
    nz_close, close_wo_lead = 0, 0
    for pn, po in zip(paths, olds):
        if not os.path.isfile(pn):
            return False, False, "missing " + pn
        L = load_npz(pn)
        O = key_gap(load_npz(po)) if os.path.isfile(po) else {}
        pres = np.asarray(L["present"], bool)
        c_src = np.asarray(L["c_src"], np.float64)
        close_wo_lead += int((c_src[~pres] != 0).sum())
        nz_close += int((c_src[pres] != 0).sum())
        for c, j, g, pr in zip(np.asarray(L["ci"]), np.asarray(L["j"]), np.asarray(L["g_src"], np.float64), pres):
            if not pr:
                continue
            tg, th = tab.get((int(c), int(j)), (float("nan"), False))
            if th:
                n_head += 1
                n_bad += int(abs(g - tg) > 1e-3)
            else:
                n_noout += 1
                if (int(c), int(j)) in O:
                    n_cmp += 1
                    n_cmp_bad += int(abs(g - O[(int(c), int(j))]) > 1e-6)
    ok_c = n_head > 0 and n_bad == 0 and n_cmp_bad == 0
    ok_d = nz_close > 0 and close_wo_lead == 0
    return ok_c, ok_d, "lead rows with a file output %d (off by more than 1e-3 m: %d), without output %d (compared with jv_old %d, differing %d), non-zero closing on lead rows %d, on rows without a lead %d" % (
        n_head, n_bad, n_noout, n_cmp, n_cmp_bad, nz_close, close_wo_lead)


def selftest():
    ok = True
    a = {"x": np.arange(5.0), "vis_counts": np.asarray("a")}
    b = {"x": np.arange(5.0), "vis_counts": np.asarray("")}
    ok &= maxdiff(a, b) == (0.0, [])
    d, p = maxdiff(a, {"x": np.arange(5.0) + 1e-3})
    ok &= d > 0 or bool(p)
    ok &= bool(maxdiff(a, {"x": np.arange(5.0), "y": np.zeros(2)})[1])
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_selftest_check_%d" % os.getpid())
    os.makedirs(os.path.join(tmp, "n", "ticks"))
    os.makedirs(os.path.join(tmp, "o", "ticks"))
    gf = os.path.join(tmp, "g.npz")
    np.savez(gf, clip_index=np.array([0, 0, 0]), frame=np.array([4, 5, 6]), gap=np.array([10.0, 11.0, np.nan]), has_out=np.array([True, True, False]))
    old = dict(ci=np.array([0, 0, 0]), j=np.array([4, 5, 6]), present=np.array([True, True, True]), g_src=np.array([9.0, 9.5, 8.0]), c_src=np.zeros(3))
    good = dict(ci=np.array([0, 0, 0]), j=np.array([4, 5, 6]), present=np.array([True, True, True]), g_src=np.array([10.0, 11.0, 8.0]), c_src=np.array([-1.0, -1.0, 0.0]))
    bad = dict(good, g_src=np.array([10.0, 11.2, 8.0]))
    wrong_old = dict(good, g_src=np.array([10.0, 11.0, 7.0]))
    np.savez(os.path.join(tmp, "o", "ticks", "rule_log_o.npz"), **old)
    np.savez(os.path.join(tmp, "n", "ticks", "rule_log_n.npz"), **good)
    for nm in ("o", "n"):
        np.savez(os.path.join(tmp, nm, "ticks", "rule_log_mid_%s.npz" % nm), ci=np.array([0]), j=np.array([4]), present=np.array([False]), g_src=np.array([np.nan]), c_src=np.zeros(1))
    r = vis_log_checks(os.path.join(tmp, "n"), "n", os.path.join(tmp, "o"), "o", gf)
    ok &= r[0] and r[1]
    np.savez(os.path.join(tmp, "n", "ticks", "rule_log_n.npz"), **bad)
    ok &= not vis_log_checks(os.path.join(tmp, "n"), "n", os.path.join(tmp, "o"), "o", gf)[0]
    np.savez(os.path.join(tmp, "n", "ticks", "rule_log_n.npz"), **wrong_old)
    ok &= not vis_log_checks(os.path.join(tmp, "n"), "n", os.path.join(tmp, "o"), "o", gf)[0]
    np.savez(os.path.join(tmp, "n", "ticks", "rule_log_n.npz"), **dict(good, c_src=np.zeros(3)))
    ok &= not vis_log_checks(os.path.join(tmp, "n"), "n", os.path.join(tmp, "o"), "o", gf)[1]
    print("ZZ CHECK-WRAPPER-V-SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    for k in ("old", "new", "tree", "ckpt", "reader", "qmap", "tracks", "eval-cache", "eval-join", "grid", "workdir", "vgap", "distrule-dir", "closest-dir"):
        ap.add_argument("--" + k)
    ap.add_argument("--pure-only", action="store_true")
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--n-clips", type=int, default=3)
    A = ap.parse_args()
    if A.selftest:
        sys.exit(selftest())
    FAILED = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
        if not cond:
            FAILED.append(name.split(" ")[0])
    src = open(A.new, encoding="utf-8").read()
    sha16 = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    print("wrapper", A.new, sha16, flush=True)
    for nm, c in static_facts(src):
        check(nm, c)
    if A.pure_only:
        print("ZZ CHECK-WRAPPER-V-PURE " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
        sys.exit(0 if not FAILED else 1)
    if not A.old:
        sys.exit("ZZ CHECK-WRAPPER-V STOP missing --old")
    if A.distrule_dir:
        os.environ["DISTRULE_DIR"] = A.distrule_dir
    if A.closest_dir:
        os.environ["CLOSEST_DIR"] = A.closest_dir
    for nm, c, d in identity_checks(A.old, A.new):
        check(nm, c, d)
    if A.identity_only:
        print("ZZ CHECK-WRAPPER-V-IDENTITY " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
        sys.exit(0 if not FAILED else 1)
    for k in ("tree", "ckpt", "reader", "qmap", "tracks", "eval_cache", "eval_join", "grid", "workdir"):
        if not getattr(A, k):
            sys.exit("ZZ CHECK-WRAPPER-V STOP missing --" + k.replace("_", "-"))
    W = A.workdir
    if os.path.exists(W):
        sys.exit("ZZ CHECK-WRAPPER-V STOP OUTPUT-EXISTS " + W)
    here = os.path.dirname(os.path.abspath(A.new))
    dist, clos = A.distrule_dir or here, A.closest_dir or here
    tmo = shutil.which("timeout")
    grid_sub = os.path.join(W, "grid_subset.json")
    noout = os.path.join(W, "noout_gapfile.npz")
    if not A.dry:
        os.makedirs(W)
        g = json.load(open(A.grid, encoding="utf-8"))
        order = list(g["order"])[:A.n_clips]
        g2 = dict(g, order=order, windows={c: g["windows"][c] for c in order}, n_windows=int(sum(len(g["windows"][c]) for c in order)))
        json.dump(g2, open(grid_sub, "w", encoding="utf-8", newline="\n"))
        J = load_npz(A.eval_join)
        n = np.asarray(J["n"], np.int64)
        ci = np.repeat(np.arange(len(n)), n)
        fr = np.concatenate([np.arange(k) for k in n])
        np.savez(noout, clip_index=ci, frame=fr, gap=np.full(len(ci), np.nan), has_out=np.zeros(len(ci), bool))

    def cmd_of(wrapper, name, extra=(), form="RR"):
        out = os.path.join(W, name)
        c = [A.python, wrapper, "--tree", A.tree, "--form", form] + (["--reader", A.reader, "--qmap", A.qmap] if form == "RR" else []) + list(extra) + [
            "--rule-log", os.path.join(out, "ticks", "rule_log_%s.npz" % name), "--rule-every", "5", "--readout", "causal_g", "--readout-log", os.path.join(out, "ticks", "readout_log_%s.json" % name),
            "--acc-gain", "0", "--path-exec", "--kappa-max", "0.2", "--preroll-steps", "30",
            "run", "--ckpt", A.ckpt, "--tag", name + "_m10_causal_heads", "--tracks", A.tracks, "--eval-cache", A.eval_cache, "--eval-grid", grid_sub, "--eval-join", A.eval_join, "--out-dir", out,
            "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze",
            "--device", A.device]
        return out, ([tmo, "1800"] + c if tmo else c)
    env = dict(os.environ, DISTRULE_DIR=dist, CLOSEST_DIR=clos, PYTHONPATH=os.path.join(A.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    runs = [("jv_old", A.old, ()), ("jv_new", A.new, ()), ("jv_nogap", A.new, ("--vis-gap-file", noout))] + ([("jv_vis", A.new, ("--vis-gap-file", A.vgap, "--vis-closing"))] if A.vgap else [])
    for name, wrapper, extra in runs:
        out, c = cmd_of(wrapper, name, extra)
        if A.dry:
            print("CMD: " + " ".join(c))
            continue
        os.makedirs(os.path.join(out, "ticks"))
        with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
            rc = subprocess.run(c, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print("RUN %s rc=%d" % (name, rc), flush=True)
        if rc != 0:
            print("ZZ CHECK-WRAPPER-V FAIL run-%s-rc-%d" % (name, rc))
            sys.exit(1)
    refs = [("jv_ref1", cmd_of(A.new, "jv_ref1", ("--vis-gap-file", os.path.join(W, "no_such_gapfile.npz")))),
            ("jv_ref2", cmd_of(A.new, "jv_ref2", ("--vis-closing",), form="RLc"))]
    if A.dry:
        for _, (_, c) in refs:
            print("CMD: " + " ".join(c))
        print("ZZ CHECK-WRAPPER-V dry run: nothing executed")
        sys.exit(0)
    refused = {}
    for nm, (_, c) in refs:
        cp = subprocess.run(c, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        txt = cp.stdout.decode("utf-8", "replace")
        open(os.path.join(W, "refusal_%s.log" % nm), "w", encoding="utf-8").write(txt)
        refused[nm] = cp.returncode != 0 and "REFUSED" in txt
    d = compare_runs(os.path.join(W, "jv_new"), "jv_new", os.path.join(W, "jv_old"), "jv_old")
    check("V2A jv_new equals jv_old: largest difference %.3g over %d files" % (d["largest_diff"], d["files"]), d["files"] > 0 and d["largest_diff"] == 0.0 and not d["problems"], "; ".join(d["problems"][:3]))
    d = compare_runs(os.path.join(W, "jv_nogap"), "jv_nogap", os.path.join(W, "jv_old"), "jv_old")
    check("V2B jv_nogap equals jv_old: largest difference %.3g over %d files" % (d["largest_diff"], d["files"]), d["files"] > 0 and d["largest_diff"] == 0.0 and not d["problems"], "; ".join(d["problems"][:3]))
    if A.vgap:
        c_ok, d_ok, det = vis_log_checks(os.path.join(W, "jv_vis"), "jv_vis", os.path.join(W, "jv_old"), "jv_old", A.vgap)
        check("V2C the gap the rule received equals the file's gap on lead rows with an output, the old gap on lead rows without", c_ok, det)
        check("V2D the closing input is not zero on some lead row and is zero on every row without a lead", d_ok, det)
    check("V2E a missing gap file is refused, and --vis-closing with form RLc is refused", all(refused.values()) and len(refused) == 2, str(refused))
    open(os.path.join(W, "CHECK_WRAPPER_V.hashes"), "w", encoding="utf-8").write("%s %s\n" % (sha16, os.path.basename(A.new)))
    last = "ZZ CHECK-WRAPPER-V PASS" if not FAILED else "ZZ CHECK-WRAPPER-V FAIL " + "; ".join(FAILED)
    open(os.path.join(W, "CHECK_WRAPPER_V_DONE.txt"), "w", encoding="utf-8").write(last + "\n")
    print(last)
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
