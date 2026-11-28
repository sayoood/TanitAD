"""test_estclose_wrapper.py -- CPU tests of the generated wrapper v8f_t1_eval_reader_rule_rzgfhqe.py (card ADD79, form RRe) and of its generator.
The wrapper under test is the file named by the environment variable ESTCLOSE_WRAPPER (default: the generated file beside this test); the mutation runner points it at mutated copies.
The reader is a real tok_reader.TokReader with small random weights (numpy), the clips are random float16 tokens; the pinned tool is not needed: the rule (dist_rule.distance_rule) is
replaced by a recorder, everything else of mid_looks / head_values_reader runs as written. Run: pycap.sh -m pytest test_estclose_wrapper.py -q --basetemp=<new folder>."""
import difflib
import hashlib
import importlib.util
import json
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
WRAPPER = os.environ.get("ESTCLOSE_WRAPPER", os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqe.py"))
PARENT = os.path.join(V, "integ10k", "v8f_t1_eval_reader_rule_rzgfhqc.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


wr = load(WRAPPER, "wr_new")
pm = load(PARENT, "wr_parent")
import closest as CL  # noqa: E402
import make_estclose_wrapper as G  # noqa: E402
import tok_reader as TR  # noqa: E402


# ------------------------------------------------------------------ fixtures: a small real reader, two clips, state stubs
def make_reader(seed=0, n_tok=3, hist=5, W=2, hid=8, emb=6, mlp=7):
    r = np.random.default_rng(seed)

    def f(*s, sc=0.3):
        return r.normal(0, sc, s).astype(np.float32)
    w = {"tok.0.weight": f(hid, 1024, sc=0.05), "tok.0.bias": f(hid), "tok.2.weight": f(W, hid), "tok.2.bias": f(W), "head.frame.weight": f(emb, n_tok * W), "head.frame.bias": f(emb),
         "head.mlp.0.weight": f(mlp, hist * emb), "head.mlp.0.bias": f(mlp), "head.mlp.2.weight": f(3, mlp), "head.mlp.2.bias": f(3)}
    w["head.mlp.2.bias"][0] = 6.0                      # presence probability close to 1: every window with a history is a present row
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


@pytest.fixture(scope="module")
def est_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("est")
    out = {}
    for kind in ("zero", "const", "reader_closing"):
        p = str(d / (kind + ".npz"))
        CL.save_planted(p, kind, 10, 2.0)
        out[kind] = p
    r = np.random.default_rng(3)
    for H in (10, 25):
        p = str(d / ("lin%d.npz" % H))
        CL.save_linear(p, H, r.normal(0, 0.5, 3 * H), r.uniform(0.5, 2.0, 3 * H), r.normal(0, 0.1, 3 * H), 0.3, seed=0)
        out["lin%d" % H] = p
    nets = [(r.normal(0, 0.3, (64, 30)), r.normal(0, 0.1, 64), r.normal(0, 0.3, (64, 64)), r.normal(0, 0.1, 64), r.normal(0, 0.3, (1, 64)), r.normal(0, 0.1, 1)) for _ in range(3)]
    p = str(d / "net10.npz")
    CL.save_nets(p, 10, r.normal(0, 0.5, 30), r.uniform(0.5, 2.0, 30), nets, seed=0)
    out["net10"] = p
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


def setup(mod, form, est_path=None, every=5):
    mod.STATE.update(form=form, reader=RD, xs=XS, ys=YS, every=every, m=10, runner=None, est_states=None, T=TStub)
    mod.STATE["mid"] = {"rows": {f: [] for f in mod.MID_FIELDS}, "rd": {k: [] for k in ("p", "g_read", "g_used", "ok")}, "p1": {"ci": [], "t0": [], "plan": []}, "env": {"plans": 0, "hits": 0}}
    mod.STATE["cols"] = {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}
    lab = form in ("RLb", "RLc")
    mod.X.MIX.update(presence="label" if lab else "head", gap="label" if lab else "head", closing={"RRc": "label", "RLc": "label", "RRh": "head", "RRe": "head"}.get(form, "zero"))
    mod.X.STATE["label"] = LabelStub()
    mod.W.HEAD_P_MIN = 0.5
    if form == "RRe":
        mod.load_estimator(est_path, RD, "")


def run_mid(mod, form, est_path=None, states=STATES):
    setup(mod, form, est_path)
    pl = types.SimpleNamespace(states=states, m=10, kw={})
    plan = np.zeros((len(states), 10, 2), np.float32)
    plan[:, :, 0] = np.arange(1, 11)[None] * 1.0
    mod.mid_looks(pl, plan)
    M = mod.STATE["mid"]
    rows = {f: np.concatenate(v) for f, v in M["rows"].items() if v}
    return rows, {k: np.concatenate(v) for k, v in M["rd"].items() if v}


def tick_batch(states):
    return np.stack([np.asarray(s.d["feats"][s.j - 4:s.j + 1], np.float32) for s in states])


def run_tick(mod, form, est_path=None, states=STATES):
    setup(mod, form, est_path)
    mod.STATE["est_states"] = states if form == "RRe" else None
    return mod.head_values_reader(None, tick_batch(states))


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


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------------ the estimate itself
@pytest.mark.parametrize("H", [10, 25])
def test_estimate_is_the_estimator_on_the_reads_of_the_last_H_frames(est_files, H):
    est = CL.ClosingEstimator.load(est_files["lin%d" % H])
    setup(wr, "RRe", est_files["lin%d" % H])
    for q in (0, 5):
        c, ok = wr._est_at(STATES, RD, q)
        want = np.array([expected_estimate(est, s.ci, s.j + q) for s in STATES])
        same(np.isnan(c), np.isnan(want))
        same(ok, np.isfinite(want))
        np.testing.assert_allclose(c[ok], want[ok], atol=1e-4, rtol=1e-4)
        assert ok.sum() >= 3 and (~ok).sum() >= 2, (q, ok)


@pytest.mark.parametrize("H", [10, 25])
def test_history_boundary_and_clip_end(est_files, H):
    need = H + 4
    setup(wr, "RRe", est_files["lin%d" % H])
    sts = [St(0, need - 1), St(0, need - 2), St(1, LENS[1] - 1), St(1, LENS[1] - 1 - 3)]
    c, ok = wr._est_at(sts, RD, 0)
    assert list(ok) == [True, False, True, True], ok
    assert np.isnan(c[1]) and np.isfinite(c[[0, 2, 3]]).all()
    c2, ok2 = wr._est_at([St(1, LENS[1] - 1)], RD, 1)               # the look frame is past the clip: no estimate
    assert not ok2[0] and np.isnan(c2[0])
    c3, ok3 = wr._est_at([St(1, LENS[1] - 1)], RD, 0)
    assert ok3[0] and np.isfinite(c3[0])


def test_planted_zero_const_and_reader_closing_at_the_estimate(est_files):
    for kind, want in (("zero", 0.0), ("const", 2.0)):
        setup(wr, "RRe", est_files[kind])
        for q in (0, 5):
            c, ok = wr._est_at(STATES, RD, q)
            assert ok.any() and (~ok).any()
            assert np.all(c[ok] == want), (kind, c)
            fed = wr.est_fed(c, ok)
            assert np.all(fed[ok] == want) and np.all(fed[~ok] == 0.0) and np.isfinite(fed).all()
    setup(wr, "RRe", est_files["reader_closing"])
    for q in (0, 5):
        c, ok = wr._est_at(STATES, RD, q)
        rr = wr._reader_closing_at(STATES, RD, q)                      # the value of form RRh at that look
        np.testing.assert_allclose(c[ok], rr[ok], atol=1e-4)
        assert np.abs(c[ok]).max() > 1e-3
        assert np.isnan(c[~ok]).all()


def test_cache_returns_the_same_reads_as_no_cache(est_files):
    setup(wr, "RRe", est_files["net10"])
    S = wr.STATE["est"]
    seq = []
    for off in (0, 1, 2, 5):                                           # overlapping histories at successive looks
        sts = [St(0, 20 + off), St(0, 33 + off), St(1, 20 + off), St(1, 31 + off)]
        seq.append((sts, 0))
    orig, cnt = RD.frame_embed, [0]

    def counted(t):
        cnt[0] += len(t)
        return orig(t)
    RD.frame_embed = counted
    try:
        S["cache"] = True
        cached = [wr._est_at(s, RD, q) for s, q in seq]
        calls_cached = cnt[0]
        cnt[0] = 0
        S["cache"] = False
        plain = [wr._est_at(s, RD, q) for s, q in seq]
        calls_plain = cnt[0]
    finally:
        RD.frame_embed = orig
    for (a, oa), (b, ob) in zip(cached, plain):
        same(oa, ob)
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
    assert calls_cached < calls_plain, (calls_cached, calls_plain)
    frames = {(s.ci, f) for sts, _ in seq for s in sts for f in range(s.j - 13, s.j + 1)}          # H = 10: frames j - 13 .. j
    assert calls_cached == len(frames), (calls_cached, len(frames))                              # every frame of the clip is embedded once
    S["cache"] = True
    wr._est_at(seq[0][0], RD, 0)
    n_before = len(S["reads"])
    RD.frame_embed = lambda t: (_ for _ in ()).throw(AssertionError("a cached look embeds nothing"))
    try:
        wr._est_at(seq[0][0], RD, 0)
    finally:
        RD.frame_embed = orig
    assert len(S["reads"]) == n_before


def test_two_clips_do_not_share_reads(est_files):
    setup(wr, "RRe", est_files["lin10"])
    est = CL.ClosingEstimator.load(est_files["lin10"])
    c, ok = wr._est_at([St(0, 30), St(1, 30), St(0, 31)], RD, 0)
    want = [expected_estimate(est, ci, e) for ci, e in ((0, 30), (1, 30), (0, 31))]
    np.testing.assert_allclose(c, want, atol=1e-4, rtol=1e-4)
    assert abs(c[0] - c[1]) > 1e-4


def test_gap_presence_closing_columns_in_the_order_gap_presence_closing(est_files):
    setup(wr, "RRe", est_files["lin10"])
    wr._est_at([St(0, 30)], RD, 0)
    S = wr.STATE["est"]
    g, p, c = window_read(0, 30)
    got = S["reads"][(0, 30)]
    np.testing.assert_allclose(got, (g, p, c), atol=1e-4)
    assert p > 0.9 and abs(g) > 1e-3                                  # a swap of the first two columns would show


# ------------------------------------------------------------------ the tick-level look
def test_tick_look_feeds_the_estimate_and_logs_it(est_files):
    est = CL.ClosingEstimator.load(est_files["lin10"])
    p, g, c, ok = run_tick(wr, "RRe", est_files["lin10"])
    want = np.array([expected_estimate(est, s.ci, s.j) for s in STATES])
    fin = np.isfinite(want)
    np.testing.assert_allclose(c[fin], want[fin], atol=1e-4, rtol=1e-4)
    assert np.all(c[~fin] == 0.0)
    ce, oe = wr.STATE["cols"]["c_est"][0], wr.STATE["cols"]["est_ok"][0]
    same(oe, fin)
    np.testing.assert_allclose(ce[fin], want[fin], atol=1e-4, rtol=1e-4)
    assert np.isnan(ce[~fin]).all()
    log = types.SimpleNamespace(arrays=lambda: {"ci": np.zeros(len(STATES))})
    arr = wr.est_tick_arrays(log)
    assert len(arr["c_est"]) == len(arr["est_ok"]) == len(STATES)
    with pytest.raises(RuntimeError):
        wr.est_tick_arrays(types.SimpleNamespace(arrays=lambda: {"ci": np.zeros(len(STATES) + 1)}))


def test_tick_look_planted_zero_is_form_RR_and_reader_closing_is_form_RRh(est_files):
    pr, gr, cr, okr = run_tick(wr, "RR")
    pz, gz, cz, okz = run_tick(wr, "RRe", est_files["zero"])
    same(pr, pz)
    same(gr, gz)
    same(okr, okz)
    assert np.all(cz == 0.0) and np.all(cr == 0.0)
    ph, gh, ch, okh = run_tick(wr, "RRh")
    pe, ge, ce, oke = run_tick(wr, "RRe", est_files["reader_closing"])
    full = wr.STATE["cols"]["est_ok"][0]
    np.testing.assert_allclose(ce[full], np.asarray(ch, np.float64)[full], atol=1e-4)
    assert np.all(ce[~full] == 0.0) and np.abs(ch[~full]).max() > 0
    pc, gc, cc, okc = run_tick(wr, "RRe", est_files["const"])
    assert np.all(cc[full] == 2.0) and np.all(cc[~full] == 0.0)


def test_other_forms_tick_look_is_the_parent(est_files):
    for form in ("RR", "RRc", "RRh"):
        a = run_tick(pm, form)
        ka = {k: [np.asarray(x) for x in v] for k, v in pm.STATE["cols"].items()}
        b = run_tick(wr, form)
        kb = {k: [np.asarray(x) for x in v] for k, v in wr.STATE["cols"].items()}
        for x, y in zip(a, b):
            same(x, y)
        assert sorted(ka) == sorted(kb) == sorted(("p", "g_read", "g_used", "ok", "c_read"))
        for k in ka:
            for x, y in zip(ka[k], kb[k]):
                same(x, y)
    for form in ("RR", "RRc", "RRh", "RRd"):
        pp = np.array([0.9, 0.2, 0.7])
        gg = np.array([10.0, 20.0, 30.0])
        for x, y in zip(pm.feed(form, pp, gg, np.ones(3, bool), XS, YS), wr.feed(form, pp, gg, np.ones(3, bool), XS, YS)):
            same(x, y)
    for x, y in zip(pm.feed("RR", pp, gg, np.ones(3, bool), XS, YS), wr.feed("RRe", pp, gg, np.ones(3, bool), XS, YS)):
        same(x, y)


# ------------------------------------------------------------------ the later looks
def test_mid_look_rows_of_other_forms_are_the_parent(est_files):
    for form in ("RR", "RRc", "RRh", "RLc", "RLb"):
        ra, da = run_mid(pm, form)
        rb, db = run_mid(wr, form)
        assert sorted(ra) == sorted(rb), form
        for k in ra:
            same(ra[k], rb[k])
        assert sorted(da) == sorted(db)
        for k in da:
            same(da[k], db[k])
        assert "c_est" not in rb and "est_ok" not in rb


def test_mid_look_planted_zero_gives_the_rows_of_form_RR(est_files):
    ra, _ = run_mid(wr, "RR")
    rb, _ = run_mid(wr, "RRe", est_files["zero"])
    assert set(rb) == set(ra) | {"c_est", "est_ok"}
    for k in ra:
        same(ra[k], rb[k])
    assert rb["present"].any() and (rb["c_src"] == 0.0).all()


def test_mid_look_planted_reader_closing_gives_the_rows_of_form_RRh_on_full_history_and_zero_elsewhere(est_files):
    rh, _ = run_mid(wr, "RRh")
    re_, _ = run_mid(wr, "RRe", est_files["reader_closing"])
    full = re_["est_ok"]
    assert full.any() and (~full).any()
    pres = re_["present"].astype(bool)
    assert (pres & full).any()
    np.testing.assert_allclose(re_["c_src"][pres & full], rh["c_src"][pres & full], atol=1e-4)
    assert np.abs(re_["c_src"][pres & full]).max() > 1e-3
    assert (re_["c_src"][pres & ~full] == 0.0).all()
    np.testing.assert_allclose(re_["c_est"][full], rh["c_read"][full], atol=1e-4)
    assert np.isnan(re_["c_est"][~full]).all()


def test_mid_look_const_estimate_is_fed_and_logged(est_files):
    r, _ = run_mid(wr, "RRe", est_files["const"])
    full = r["est_ok"]
    pres = r["present"].astype(bool)
    assert (pres & full).any() and (r["c_src"][pres & full] == 2.0).all() and (r["c_est"][full] == 2.0).all()
    assert (r["c_src"][pres & ~full] == 0.0).all()
    cap = CAPT[-1]
    vl = r["v_l"]
    np.testing.assert_allclose(cap["v_L"][pres & full], vl[pres & full] + 2.0)


def test_mid_look_estimate_is_taken_at_frame_j_plus_q(est_files):
    est = CL.ClosingEstimator.load(est_files["lin10"])
    r, _ = run_mid(wr, "RRe", est_files["lin10"])
    want = np.array([expected_estimate(est, int(c), int(j)) for c, j in zip(r["ci"], r["j"])])
    fin = np.isfinite(want)
    same(r["est_ok"], fin)
    np.testing.assert_allclose(r["c_est"][fin], want[fin], atol=1e-4, rtol=1e-4)
    pres = r["present"].astype(bool)
    np.testing.assert_allclose(r["c_src"][pres & fin], want[pres & fin], atol=1e-4, rtol=1e-4)
    assert np.array_equal(r["j"], np.array([s.j + 5 for s in STATES]))


# ------------------------------------------------------------------ counts, run line, pre-roll file, refusals
def test_count_line_and_run_file(est_files, tmp_path):
    setup(wr, "RRe", est_files["zero"])
    T_, F_ = np.array([True, True, False]), np.array([False])
    wr.STATE["cols"]["est_ok"] = [T_, T_]
    wr.STATE["mid"]["rows"]["est_ok"] = [F_, np.array([True, False])]
    post = {"cols": {"est_ok": [np.array([True, False, False])]}, "mid": {"rows": {"est_ok": [np.array([False])]}}}
    skip = {"cols": {"est_ok": [np.array([True] * 7)]}, "mid": {"rows": {"est_ok": [np.array([True] * 7)]}}}
    wr.STATE["runner"] = types.SimpleNamespace(posts=[("model", post), ("floor", skip)])
    sha = hashlib.sha256(open(est_files["zero"], "rb").read()).hexdigest()
    assert wr.STATE["est"]["sha256"] == sha
    wr.STATE["est"]["rule_log"] = str(tmp_path / "rule_log_RUNX_s0.npz")
    assert wr.est_line() == "est looks scored full 5 short 4 preroll full 1 short 3 file " + sha
    wr.write_est_line()
    wr.write_est_line()
    lines = open(tmp_path / "run_RUNX_s0.txt", encoding="utf-8").read().splitlines()
    assert lines == [wr.est_line()] * 2
    setup(wr, "RR")
    wr.STATE["est"] = None
    assert wr.est_line() is None
    wr.write_est_line()


def test_finish_preroll_pre_rows_carry_their_own_estimate(est_files, tmp_path):
    setup(wr, "RRe", est_files["zero"])
    LF = wr.W.LOG_FIELDS
    rl = str(tmp_path / "rule_log_PRE_s0.npz")
    np.savez(rl, **{k: np.zeros(5) for k in LF}, c_est=np.arange(5, dtype=np.float64), est_ok=np.ones(5, bool), mode=np.asarray("mix"))
    post = {"rule_rows": {k: [np.zeros(3)] for k in LF}, "cols": {"c_est": [np.array([1.0, 2.0, np.nan])], "est_ok": [np.array([True, True, False])]},
            "mid": {"rows": {"ci": [np.zeros(2)], "c_est": [np.array([1.0, 2.0])], "est_ok": [np.array([True, True])]}}}
    runner = types.SimpleNamespace(posts=[("model", post)], post_seg=[0], P=30, pre_exec="plan", offset_arrays=lambda: {"a": np.zeros(1)}, summary=lambda: {})
    wr.STATE.update(runner=runner, pre_orig=(types.SimpleNamespace(make_states=None, rollout=None), None, None), pre_shim=(types.SimpleNamespace(distance_rule=None), None),
                    pre_mine=types.SimpleNamespace(rule_log=rl, readout_log="", form="RRe", readout=None, kappa_max=None, rule_every=5),
                    pre_a=types.SimpleNamespace(replan_steps=10, nav="x", tag="t", dump_ticks=str(tmp_path), out_dir=str(tmp_path)), rulelog=object())
    wr.finish_preroll()
    with np.load(str(tmp_path / "rule_log_pre_PRE_s0.npz"), allow_pickle=False) as z:
        assert len(z["ci"]) == 3 and len(z["c_est"]) == 3 and len(z["est_ok"]) == 3
        assert z["est_ok"].tolist() == [True, True, False] and z["c_est"][:2].tolist() == [1.0, 2.0] and np.isnan(z["c_est"][2])


def _refused(argv):
    with pytest.raises(SystemExit) as e:
        wr.main(argv)
    return str(e.value)


def test_closing_est_flag_is_refused_outside_form_RRe_and_required_inside(est_files):
    assert "needs --closing-est" in _refused(["--tree", "x", "--form", "RRe", "run"])
    for form in ("RR", "RRh", "RLc", "R0"):
        assert "belongs to form RRe only" in _refused(["--tree", "x", "--form", form, "--closing-est", est_files["zero"], "run"])
    assert "not found" in _read_refusal_missing_file()


def _read_refusal_missing_file():
    with pytest.raises(SystemExit) as e:
        wr.load_estimator("no_such_file.npz", RD, "")
    return str(e.value)


def test_form_wiring_text(est_files):
    src = open(WRAPPER, encoding="utf-8").read()
    assert '"RRd": "head", "RRe": "head"}.get(f, "zero")' in src
    assert 'if f not in ("RLb", "RLc", "RR", "RRc", "RRh", "RRe"):' in src
    assert 'if f == "RRe":\n            load_estimator(mine.closing_est, STATE["reader"], mine.rule_log)' in src
    assert 'dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta' in src
    assert "FORMS = " in src and '"RRd", "RRe")' in src


# ------------------------------------------------------------------ the generator
def test_generator_changes_only_the_asserted_places():
    raw = open(PARENT, "rb").read()
    assert hashlib.sha256(raw).hexdigest()[:16] == G.PARENT_SHA16
    t, anchors, plines, ext = G.build(raw.decode("utf-8"))
    assert len(anchors) == 21 == len(ext)
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
    if "ESTCLOSE_WRAPPER" not in os.environ:
        assert t == open(WRAPPER, encoding="utf-8").read()
    with pytest.raises(AssertionError):
        G.once("a b a", "a", "c")                                       # an old text that occurs twice is refused


def test_none_of_the_banned_words_in_the_new_text_or_the_tools():
    bad = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
    src = open(WRAPPER, encoding="utf-8").read()
    new_text = src[:src.index("Text of the parent (rzgfhqc) follows.")].lower() + G.HELPERS.lower()
    for f in ("make_estclose_wrapper.py", "test_estclose_wrapper.py", "run_mutations_ec.py", "check_estclose.py", "pod_check_estclose.sh", "make_estclose_queue.py", "pod_estclose10k_queue.sh",
              "ESTCLOSE_DESIGN.md"):
        p = os.path.join(HERE, f)
        if os.path.isfile(p):
            new_text += open(p, encoding="utf-8").read().lower()
    assert not [w for w in bad if w in new_text], [w for w in bad if w in new_text]


# ------------------------------------------------------------------ the planner, the later-look file, the end of main
def test_planner_hands_its_states_to_the_estimate(est_files, monkeypatch):
    setup(wr, "RRe", est_files["zero"])
    pl = object.__new__(wr.ReaderPlanner)
    pl.states, pl.m = STATES, 10
    seen = []
    monkeypatch.setattr(wr.X.MixPlanner, "__call__", lambda self, batch: (seen.append(wr.STATE["est_states"]) or "plan"))
    wr.STATE.update(est_states=None, every=10, preroll=0)
    assert pl(None) == "plan" and seen[0] is STATES
    setup(wr, "RR")
    wr.STATE.update(est_states=None, every=10, preroll=0)
    assert pl(None) == "plan" and seen[1] is None


def test_save_mid_writes_the_estimate_keys_for_RRe_only(est_files, tmp_path):
    run_mid(wr, "RRe", est_files["const"])
    wr.save_mid(str(tmp_path / "rule_log_SV_s0.npz"), {"form": "RRe"}, 10)
    with np.load(str(tmp_path / "rule_log_mid_SV_s0.npz"), allow_pickle=False) as z:
        n = len(z["ci"])
        assert n == len(STATES) and len(z["c_est"]) == n and len(z["est_ok"]) == n and z["est_ok"].any()
        assert z["c_est"][z["est_ok"]].tolist() == [2.0] * int(z["est_ok"].sum())
    run_mid(wr, "RR")
    wr.save_mid(str(tmp_path / "rule_log_SW_s0.npz"), {"form": "RR"}, 10)
    with np.load(str(tmp_path / "rule_log_mid_SW_s0.npz"), allow_pickle=False) as z:
        assert "c_est" not in z.files and "est_ok" not in z.files and "c_read" in z.files


def test_main_writes_the_run_line_after_the_pre_roll_files():
    src = open(WRAPPER, encoding="utf-8").read()
    assert "    finally:\n        finish_preroll()\n        write_est_line()\n" in src
