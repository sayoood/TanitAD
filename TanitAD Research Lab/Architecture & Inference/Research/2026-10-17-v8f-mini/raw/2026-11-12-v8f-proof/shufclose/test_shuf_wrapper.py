"""test_shuf_wrapper.py -- CPU tests of the generated wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (card ADD83, form RRe with --closing-table) and of its generator.
The wrapper under test is the file named by the environment variable SHUFCLOSE_WRAPPER (default: the generated file beside this test); the mutation runner points it at mutated copies.
Same rig as estclose/test_estclose_wrapper.py: a real tok_reader.TokReader with small random weights, random float16 clips, the rule (dist_rule.distance_rule) replaced by a recorder.
Run: pycap.sh -m pytest test_shuf_wrapper.py -q --basetemp=<new folder>."""
import difflib
import hashlib
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.normpath(os.path.join(HERE, ".."))
os.environ["DISTRULE_DIR"] = os.path.join(V, "distrule")
os.environ["CLOSEST_DIR"] = os.path.join(V, "closeread")
for _p in (os.path.join(V, "integ10k"), os.path.join(V, "closeread"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
WRAPPER = os.environ.get("SHUFCLOSE_WRAPPER", os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqet.py"))
PARENT = os.path.join(V, "estclose", "v8f_t1_eval_reader_rule_rzgfhqe.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


wr = load(WRAPPER, "wr_shuf")
pm = load(PARENT, "wr_parent_e")
import closest as CL  # noqa: E402
import make_shuf_wrapper as G  # noqa: E402
import tok_reader as TR  # noqa: E402


def make_reader(seed=0, n_tok=3, hist=5, W=2, hid=8, emb=6, mlp=7):
    r = np.random.default_rng(seed)

    def f(*s, sc=0.3):
        return r.normal(0, sc, s).astype(np.float32)
    w = {"tok.0.weight": f(hid, 1024, sc=0.05), "tok.0.bias": f(hid), "tok.2.weight": f(W, hid), "tok.2.bias": f(W), "head.frame.weight": f(emb, n_tok * W), "head.frame.bias": f(emb),
         "head.mlp.0.weight": f(mlp, hist * emb), "head.mlp.0.bias": f(mlp), "head.mlp.2.weight": f(3, mlp), "head.mlp.2.bias": f(3)}
    w["head.mlp.2.bias"][0] = 6.0
    return TR.TokReader(w, hist=hist, n_tok=n_tok)


RD = make_reader()
_rng = np.random.default_rng(11)
LENS = {0: 60, 1: 45}
FEATS = {ci: _rng.normal(0, 1, (n, 3, 1024)).astype(np.float16) for ci, n in LENS.items()}
GT = np.stack([np.arange(1, 31, dtype=np.float64) * 1.0, np.zeros(30)], 1)
XS, YS = np.array([1.0, 100.0]), np.array([1.0, 100.0])


class St:
    def __init__(self, ci, j, t0=None):
        n = LENS[ci]
        self.ci, self.j = ci, j
        self.t0 = j if t0 is None else t0
        self.d = {"feats": FEATS[ci], "gt": {self.t0: GT}, "pose": np.zeros((n, 3)), "p10f": np.zeros((n, 3))}
        self.track = np.zeros((2 * n, 4))
        self.logged_v = np.full(n, 5.0)


STATES = [St(0, 20), St(0, 8), St(0, 7), St(0, 5), St(0, 40), St(0, 50), St(1, 39), St(1, 40), St(1, 13), St(1, 12)]


class LabelStub:
    @staticmethod
    def read(ci, j):
        return True, 20.0 + 0.1 * j, 0.5 * (ci + 1)


class TStub:
    @staticmethod
    def plan_to_world(P, pose):
        return np.asarray(P, np.float64)

    @staticmethod
    def envelope(p):
        return p, 0


CAPT = []


def stub_rule(plan, v0, present, g_src, g0, v_L, m=None, **kw):
    B = len(v0)
    CAPT.append({"present": np.array(present, copy=True), "g_src": np.array(g_src, copy=True), "v_L": np.array(v_L, copy=True)})
    return types.SimpleNamespace(active=np.array(present, copy=True), n_sat=np.zeros(B, np.int64), n_sat_exec=np.zeros(B, np.int64), n_lift=np.zeros(B, np.int64),
                                 changed=np.zeros(B, bool), plan=plan)


def table_from(fn, path, clips=(0, 1)):
    """a table file with one row per (clip, frame) of the stub clips; fn(ci, frame) -> value."""
    ci = np.concatenate([np.full(LENS[c], c, np.int64) for c in clips])
    fr = np.concatenate([np.arange(LENS[c], dtype=np.int64) for c in clips])
    est = np.array([fn(int(a), int(b)) for a, b in zip(ci, fr)], np.float64)
    np.savez(path, clip_index=ci, frame=fr, est=est)
    return path


def window_read(ci, f):
    toks = np.asarray(FEATS[ci][f - 4:f + 1], np.float32)[None]
    p, g = RD.read(toks)
    c = wr.reader_closing(RD, toks)
    return float(g[0]), float(p[0]), float(c[0])


def expected_estimate(est, ci, e):
    H = est.H
    if e - est.need_frames() + 1 < 0 or e >= LENS[ci]:
        return np.nan
    X = np.array([window_read(ci, f) for f in range(e - H + 1, e + 1)], np.float64)
    return float(est.apply(X[None])[0])


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("shuf")
    out = {}
    r = np.random.default_rng(3)
    p = str(d / "lin10.npz")
    CL.save_linear(p, 10, r.normal(0, 0.5, 30), r.uniform(0.5, 2.0, 30), r.normal(0, 0.1, 30), 0.3, seed=0)
    out["lin10"] = p
    est = CL.ClosingEstimator.load(p)
    cache = {}

    def ident(ci, f):
        if (ci, f) not in cache:
            cache[(ci, f)] = expected_estimate(est, ci, f)
        return cache[(ci, f)]
    out["ident"] = table_from(ident, str(d / "ident.npz"))
    out["zero"] = table_from(lambda c, f: 0.0, str(d / "zero.npz"))
    out["const2"] = table_from(lambda c, f: 2.0, str(d / "const2.npz"))
    out["ramp"] = table_from(lambda c, f: 100.0 * c + 0.5 * f - 3.0, str(d / "ramp.npz"))
    out["ramp_hole"] = table_from(lambda c, f: (np.nan if f % 7 == 0 else 100.0 * c + 0.5 * f - 3.0), str(d / "ramp_hole.npz"))
    out["lin10_est"] = p
    return out


@pytest.fixture(autouse=True)
def _patch_rule(monkeypatch):
    for m in (wr, pm):
        monkeypatch.setattr(m.R, "distance_rule", stub_rule)
    CAPT.clear()
    yield
    for m in (wr, pm):
        m.STATE["runner"] = None
        m.STATE["est"] = None


def setup(mod, form, est_path=None, table=None, every=5):
    mod.STATE.update(form=form, reader=RD, xs=XS, ys=YS, every=every, m=10, runner=None, est_states=None, T=TStub)
    mod.STATE["mid"] = {"rows": {f: [] for f in mod.MID_FIELDS}, "rd": {k: [] for k in ("p", "g_read", "g_used", "ok")}, "p1": {"ci": [], "t0": [], "plan": []}, "env": {"plans": 0, "hits": 0}}
    mod.STATE["cols"] = {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}
    lab = form in ("RLb", "RLc")
    mod.X.MIX.update(presence="label" if lab else "head", gap="label" if lab else "head", closing={"RRc": "label", "RLc": "label", "RRh": "head", "RRe": "head"}.get(form, "zero"))
    mod.X.STATE["label"] = LabelStub()
    mod.W.HEAD_P_MIN = 0.5
    if form == "RRe":
        if table is not None:
            mod.load_table(table, "")
        else:
            mod.load_estimator(est_path, RD, "")


def run_mid(mod, form, est_path=None, table=None, states=STATES):
    setup(mod, form, est_path, table)
    pl = types.SimpleNamespace(states=states, m=10, kw={})
    plan = np.zeros((len(states), 10, 2), np.float32)
    plan[:, :, 0] = np.arange(1, 11)[None] * 1.0
    mod.mid_looks(pl, plan)
    M = mod.STATE["mid"]
    rows = {f: np.concatenate(v) for f, v in M["rows"].items() if v}
    return rows, {k: np.concatenate(v) for k, v in M["rd"].items() if v}


def tick_batch(states):
    return np.stack([np.asarray(s.d["feats"][s.j - 4:s.j + 1], np.float32) for s in states])


def run_tick(mod, form, est_path=None, table=None, states=STATES):
    setup(mod, form, est_path, table)
    mod.STATE["est_states"] = states if form == "RRe" else None
    return mod.head_values_reader(None, tick_batch(states))


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def ramp(c, f):
    return 100.0 * c + 0.5 * f - 3.0


# ------------------------------------------------------------------ the lookup
def test_table_value_at_clip_index_and_frame_of_the_look(files):
    setup(wr, "RRe", table=files["ramp"])
    for q in (0, 5):
        c, ok = wr._est_at(STATES, RD, q)
        want = np.array([ramp(s.ci, s.j + q) if s.j + q < LENS[s.ci] else np.nan for s in STATES])
        same(ok, np.isfinite(want))
        same(c[ok], want[ok])
        assert np.isnan(c[~ok]).all()
    assert wr.STATE["est"]["embed_calls"] == 0 and not wr.STATE["est"]["reads"]          # no reader features are read for a table


def test_clip_index_is_the_key_not_the_position(files):
    setup(wr, "RRe", table=files["ramp"])
    c, ok = wr._est_at([St(1, 10), St(0, 10)], RD, 0)
    assert ok.all() and c[0] == ramp(1, 10) and c[1] == ramp(0, 10) and c[0] != c[1]


def test_missing_rows_and_nan_values_are_short_and_feed_zero(files):
    setup(wr, "RRe", table=files["ramp_hole"])
    sts = [St(0, 14), St(0, 15), St(1, 21), St(1, 22), St(1, LENS[1] - 1), St(0, 2)]
    c, ok = wr._est_at(sts, RD, 0)
    assert list(ok) == [False, True, False, True, True, True]
    assert np.isnan(c[0]) and np.isnan(c[2])
    fed = wr.est_fed(c, ok)
    assert fed[0] == 0.0 and fed[2] == 0.0 and np.isfinite(fed).all()
    c2, ok2 = wr._est_at([St(1, LENS[1] - 1)], RD, 1)                  # one frame past the clip: no row
    assert not ok2[0] and np.isnan(c2[0])
    c3, ok3 = wr._est_at([St(0, 4)], RD, -10)                          # a negative frame: no row, no wrap to another clip
    assert not ok3[0] and np.isnan(c3[0])


def test_clip_not_in_the_table_has_no_row(files, tmp_path):
    p = table_from(ramp, str(tmp_path / "only0.npz"), clips=(0,))
    setup(wr, "RRe", table=p)
    c, ok = wr._est_at([St(0, 10), St(1, 10)], RD, 0)
    assert list(ok) == [True, False] and np.isnan(c[1])


def test_infinite_value_is_short(files, tmp_path):
    p = str(tmp_path / "inf.npz")
    np.savez(p, clip_index=np.array([0, 0], np.int64), frame=np.array([3, 4], np.int64), est=np.array([np.inf, 1.5]))
    setup(wr, "RRe", table=p)
    c, ok = wr._est_at([St(0, 3), St(0, 4)], RD, 0)
    assert list(ok) == [False, True] and np.isnan(c[0]) and c[1] == 1.5


def test_row_order_of_the_table_does_not_matter(files, tmp_path):
    with np.load(files["ramp"]) as z:
        ci, fr, v = z["clip_index"], z["frame"], z["est"]
    o = np.random.default_rng(1).permutation(len(ci))
    p = str(tmp_path / "perm.npz")
    np.savez(p, clip_index=ci[o], frame=fr[o], est=v[o])
    setup(wr, "RRe", table=files["ramp"])
    a = wr._est_at(STATES, RD, 5)
    setup(wr, "RRe", table=p)
    b = wr._est_at(STATES, RD, 5)
    same(a[0], b[0])
    same(a[1], b[1])


# ------------------------------------------------------------------ the table of the estimate reproduces form RRe with the estimator
def test_identity_table_gives_the_rows_of_the_estimator_at_the_tick_look(files):
    pe, ge, ce, oke = run_tick(wr, "RRe", est_path=files["lin10_est"])
    ke = {k: [np.asarray(x) for x in v] for k, v in wr.STATE["cols"].items()}
    pt, gt, ct, okt = run_tick(wr, "RRe", table=files["ident"])
    kt = {k: [np.asarray(x) for x in v] for k, v in wr.STATE["cols"].items()}
    same(pe, pt)
    same(ge, gt)
    same(oke, okt)
    np.testing.assert_allclose(ce, ct, atol=1e-4, rtol=1e-4)
    same(ke["est_ok"][0], kt["est_ok"][0])
    np.testing.assert_allclose(ke["c_est"][0], kt["c_est"][0], atol=1e-4, rtol=1e-4, equal_nan=True)
    assert kt["est_ok"][0].any() and (~kt["est_ok"][0]).any()


def test_identity_table_gives_the_rows_of_the_estimator_at_the_later_looks(files):
    re_, _ = run_mid(wr, "RRe", est_path=files["lin10_est"])
    rt, _ = run_mid(wr, "RRe", table=files["ident"])
    assert sorted(re_) == sorted(rt)
    for k in re_:
        if re_[k].dtype.kind == "f":
            np.testing.assert_allclose(re_[k], rt[k], atol=1e-4, rtol=1e-4, equal_nan=True)
        else:
            same(re_[k], rt[k])
    assert rt["est_ok"].any() and (~rt["est_ok"]).any()


def test_zero_table_is_form_RR_with_largest_difference_0(files):
    ra, da = run_mid(wr, "RR")
    rb, db = run_mid(wr, "RRe", table=files["zero"])
    assert set(rb) == set(ra) | {"c_est", "est_ok"}
    for k in ra:
        same(ra[k], rb[k])
    for k in da:
        same(da[k], db[k])
    assert rb["present"].any() and (rb["c_src"] == 0.0).all()
    pr, gr, cr, okr = run_tick(wr, "RR")
    pz, gz, cz, okz = run_tick(wr, "RRe", table=files["zero"])
    for x, y in zip((pr, gr, cr, okr), (pz, gz, cz, okz)):
        same(x, y)


def test_const2_table_shows_2_on_every_look_with_a_row(files):
    r, _ = run_mid(wr, "RRe", table=files["const2"])
    full, pres = r["est_ok"], r["present"].astype(bool)
    assert (pres & full).any() and (r["c_src"][pres & full] == 2.0).all() and (r["c_est"][full] == 2.0).all()
    np.testing.assert_allclose(CAPT[-1]["v_L"][pres & full], r["v_l"][pres & full] + 2.0)
    p, g, c, ok = run_tick(wr, "RRe", table=files["const2"])
    assert (c[ok] == 2.0).all() and ok.any()


def test_mid_look_takes_the_table_at_frame_j_plus_q(files):
    r, _ = run_mid(wr, "RRe", table=files["ramp"])
    full = r["est_ok"]
    want = np.array([ramp(int(c), int(j)) for c, j in zip(r["ci"], r["j"])])
    same(r["c_est"][full], want[full])
    assert np.array_equal(r["j"], np.array([s.j + 5 for s in STATES]))
    assert (~full).any()


def test_tick_look_feeds_the_table_and_logs_it(files):
    p, g, c, ok = run_tick(wr, "RRe", table=files["ramp"])
    want = np.array([ramp(s.ci, s.j) for s in STATES])
    same(c, want)
    same(wr.STATE["cols"]["c_est"][0], want)
    same(wr.STATE["cols"]["est_ok"][0], np.ones(len(STATES), bool))
    log = types.SimpleNamespace(arrays=lambda: {"ci": np.zeros(len(STATES))})
    arr = wr.est_tick_arrays(log)
    assert len(arr["c_est"]) == len(arr["est_ok"]) == len(STATES)


# ------------------------------------------------------------------ the parent's behaviour is unchanged
def test_form_RRe_with_an_estimator_file_is_the_parent(files):
    ra, da = run_mid(pm, "RRe", est_path=files["lin10_est"])
    rb, db = run_mid(wr, "RRe", est_path=files["lin10_est"])
    assert sorted(ra) == sorted(rb)
    for k in ra:
        same(ra[k], rb[k])
    for x, y in zip(run_tick(pm, "RRe", est_path=files["lin10_est"]), run_tick(wr, "RRe", est_path=files["lin10_est"])):
        same(x, y)


def test_other_forms_are_the_parent(files):
    for form in ("RR", "RRc", "RRh", "RLc", "RLb"):
        ra, da = run_mid(pm, form)
        rb, db = run_mid(wr, form)
        assert sorted(ra) == sorted(rb), form
        for k in ra:
            same(ra[k], rb[k])
        assert "c_est" not in rb
    for form in ("RR", "RRc", "RRh"):
        for x, y in zip(run_tick(pm, form), run_tick(wr, form)):
            same(x, y)


# ------------------------------------------------------------------ counts, run line, refusals
def test_run_line_carries_the_sha256_of_the_table(files, tmp_path):
    setup(wr, "RRe", table=files["zero"])
    T_, F_ = np.array([True, True, False]), np.array([False])
    wr.STATE["cols"]["est_ok"] = [T_, T_]
    wr.STATE["mid"]["rows"]["est_ok"] = [F_, np.array([True, False])]
    post = {"cols": {"est_ok": [np.array([True, False, False])]}, "mid": {"rows": {"est_ok": [np.array([False])]}}}
    wr.STATE["runner"] = types.SimpleNamespace(posts=[("model", post)])
    sha = hashlib.sha256(open(files["zero"], "rb").read()).hexdigest()
    assert wr.STATE["est"]["sha256"] == sha
    wr.STATE["est"]["rule_log"] = str(tmp_path / "rule_log_RUNX_s0.npz")
    assert wr.est_line() == "est looks scored full 5 short 4 preroll full 1 short 3 file " + sha
    wr.write_est_line()
    assert open(tmp_path / "run_RUNX_s0.txt", encoding="utf-8").read().splitlines() == [wr.est_line()]


def _refused(argv):
    with pytest.raises(SystemExit) as e:
        wr.main(argv)
    return str(e.value)


def test_flag_refusals(files):
    t, e = files["zero"], files["lin10_est"]
    assert "--closing-table belongs to form RRe only" in _refused(["--tree", "x", "--form", "RR", "--closing-table", t, "run"])
    for form in ("RRh", "RLc", "R0", "RRc"):
        assert "belongs to form RRe only" in _refused(["--tree", "x", "--form", form, "--closing-table", t, "run"])
    assert "--closing-est and --closing-table together" in _refused(["--tree", "x", "--form", "RRe", "--closing-est", e, "--closing-table", t, "run"])
    assert "--closing-est and --closing-table together" in _refused(["--tree", "x", "--form", "RR", "--closing-est", e, "--closing-table", t, "run"])
    m = _refused(["--tree", "x", "--form", "RRe", "run"])
    assert "needs --closing-est" in m and "--closing-table" in m
    assert "belongs to form RRe only" in _refused(["--tree", "x", "--form", "RR", "--closing-est", e, "run"])


def test_table_file_refusals(tmp_path):
    def bad(**arrs):
        p = str(tmp_path / ("bad%d.npz" % len(os.listdir(tmp_path))))
        np.savez(p, **arrs)
        setup(wr, "RR")
        wr.STATE["cols"]["c_est"] = []
        wr.STATE["mid"]["rows"]["c_est"] = []
        with pytest.raises(SystemExit) as e:
            wr.load_table(p, "")
        return str(e.value)
    ci, fr, v = np.array([0, 0], np.int64), np.array([1, 2], np.int64), np.array([0.5, 0.6])
    assert "lacks the array est" in bad(clip_index=ci, frame=fr)
    assert "lacks the array frame" in bad(clip_index=ci, est=v)
    assert "same clip index and frame" in bad(clip_index=ci, frame=np.array([1, 1], np.int64), est=v)
    assert "negative" in bad(clip_index=ci, frame=np.array([-1, 2], np.int64), est=v)
    assert "negative" in bad(clip_index=np.array([-1, 0], np.int64), frame=fr, est=v)
    assert "negative" in bad(clip_index=ci, frame=np.array([1, 1 << 20], np.int64), est=v)
    assert "one length" in bad(clip_index=ci, frame=fr, est=v[:1])
    setup(wr, "RR")
    with pytest.raises(SystemExit) as e:
        wr.load_table(str(tmp_path / "no_such.npz"), "")
    assert "not found" in str(e.value)


def test_form_wiring_text():
    src = open(WRAPPER, encoding="utf-8").read()
    assert 'ap.add_argument("--closing-table", default="")' in src
    assert 'if f == "RRe":\n            if mine.closing_table:\n                load_table(mine.closing_table, mine.rule_log)\n            else:\n                load_estimator(' in src
    assert '    if S.get("table") is not None:' in src and "return _table_at(states, e_off)" in src
    assert 'dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta' in src
    assert "    finally:\n        finish_preroll()\n        write_est_line()\n" in src


# ------------------------------------------------------------------ the generator
def test_generator_changes_only_the_asserted_places():
    raw = open(PARENT, "rb").read()
    assert hashlib.sha256(raw).hexdigest()[:16] == G.PARENT_SHA16
    t, anchors, plines, ext = G.build(raw.decode("utf-8"))
    assert len(anchors) == 6 == len(ext)
    pl, nl = raw.decode("utf-8").split("\n"), t.split("\n")
    sm = difflib.SequenceMatcher(None, pl, nl, autojunk=False)
    changed = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed.update(range(i1 + 1, i2 + 1))
    allowed = set()
    for s, n in ext.values():
        allowed.update(range(s, s + n + 1))
    assert changed <= allowed, sorted(changed - allowed)
    if "SHUFCLOSE_WRAPPER" not in os.environ:
        assert t == open(WRAPPER, encoding="utf-8").read()
    with pytest.raises(AssertionError):
        G.once("a b a", "a", "c")


def test_none_of_the_banned_words_in_the_new_text_or_the_tools():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    src = open(WRAPPER, encoding="utf-8").read()
    new_text = src[:src.index("Text of the parent (rzgfhqe) follows.")].lower() + G.TABLE_HELPERS.lower()
    for f in os.listdir(HERE):
        p = os.path.join(HERE, f)
        if os.path.isfile(p) and f.endswith((".py", ".sh", ".md")) and not f.startswith("tmp_") and f != "v8f_t1_eval_reader_rule_rzgfhqet.py":
            new_text += open(p, encoding="utf-8").read().lower()
    assert not [w for w in bad if w in new_text], [w for w in bad if w in new_text]


def test_a_look_between_two_rows_without_its_own_row_is_short(tmp_path):
    ci = np.zeros(6, np.int64)
    fr = np.array([5, 6, 7, 9, 10, 11], np.int64)
    p = str(tmp_path / "gap.npz")
    np.savez(p, clip_index=ci, frame=fr, est=fr.astype(np.float64) * 0.1)
    setup(wr, "RRe", table=p)
    c, ok = wr._est_at([St(0, 7), St(0, 8), St(0, 9), St(0, 4), St(0, 12)], RD, 0)
    assert list(ok) == [True, False, True, False, False]
    assert c[0] == 7 * 0.1 and c[2] == 9 * 0.1 and np.isnan(c[[1, 3, 4]]).all()
