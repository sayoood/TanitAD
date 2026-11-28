"""test_imag_plan_q.py -- note PREREG_V8F_PROOF_ADD40_N7: source cand-cvq (cand-cv with the gap min(g_head, q_head)). Synthetic world and tiny checkpoint of the DISTRULE tests (CPU, no real data).
Runs ON THE POD with the plain-python runner (mutlists/common/minipytest.py); the dev box has no torch.  Env DISTRULE_TREE = a tree with the pinned tool (default /workspace/v8proof/tree_leadsup),
IPQ_LIBS = extra library folders (os.pathsep list).  No number here is a result about a real arm."""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("DISTRULE_TREE", "/workspace/v8proof/tree_leadsup")
HERE = os.path.dirname(os.path.abspath(__file__))
_libs = [x for x in os.environ.get("IPQ_LIBS", "").split(os.pathsep) if x] + [os.path.join(HERE, "..", "distrule"), "/workspace/v8proof/distrule", "/workspace/v8proof/probe", "/workspace/v8proof/mutlists/distrule/tree_pristine/src"]   # the last holds test_dist_rule.py (fixtures)
for _p in reversed(_libs):
    if os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import v8f_t1_eval_imagplan_q as WQ  # noqa: E402
import v8f_t1_eval_rule as W  # noqa: E402
import qhead_model as QM  # noqa: E402
from test_dist_rule import T, world, tiny, _lead_join, _env, _ds, stub_planner, arrays_equal  # noqa: E402,F401  (fixtures and helpers of the DISTRULE tests, read only)

assert os.path.abspath(WQ.__file__).startswith(os.path.abspath(HERE)), (WQ.__file__, HERE)
TICKS = tuple(range(3, 40, 4))


class Cap:
    """A base planner that stores the batch features of every call and returns the stub plan."""
    def __init__(self):
        self.feats = []

    def __call__(self, batch):
        self.feats.append(batch["feats"].detach().clone())
        return stub_planner(batch)


def run_rollout(T, world, join_path, planner, cfg, ticks=TICKS, m=10):
    store, tracks, J, wins = _env(T, world["root"], join_path, ticks)
    proxy, ds = _ds(T, J, cfg)
    states = W.attach_logged(T.make_states(store, tracks, J, wins, "deadreckon"), store, J)
    WQ.attach_logged_actions(T, states, store, tracks, J, False)
    planner.bind(states)
    _, exe = T.rollout(planner, states, ds, proxy, m=m, schedule="periodic", device="cpu")
    return np.asarray(exe)


def make_ip(T, source, model, base=stub_planner, **kw):
    return WQ.ImagPlanner(base, model, source, 10, T.fk, T, cset=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0), fallback="zero", family="decel", **kw)


def random_head(model, feats, seed=5):
    hist = int(model.wm.lead_head.hist)
    with torch.no_grad():
        d = int(model.wm.z1(model.adapter(feats[:, -1:].float()) + model.pos[None, None]).shape[-1])
    torch.manual_seed(seed)
    return QM.QHead(d, hist).eval()


def probe_feats(T, world, p, cfg, model):
    cap = Cap()
    ip = make_ip(T, "cand-cv", model, base=cap)
    run_rollout(T, world, p, ip, cfg, ticks=(3, 7))
    return cap.feats[0]


class fake_heads:
    """Replace W.head_values by fixed values: row r reads g = 12, 3 or 70 m by r % 3, presence 1, closing speed 0, all frames real."""
    def __enter__(self):
        self.old = W.head_values

        def fv(model, feats, frame_ok=None, hist=None):
            B = feats.shape[0]
            g = np.array([12.0, 3.0, 70.0], np.float32)[np.arange(B) % 3]
            return np.ones(B, np.float32), g, np.zeros(B, np.float32), np.ones(B, bool)
        W.head_values = fv
        return self

    def __exit__(self, *a):
        W.head_values = self.old


def test_the_min_of_a_hand_computed_case_and_the_gate_stays_on_the_head_gap(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_q1.npz", const_gap=20.0)
    # the function: head gaps 12, 3, 70 with q = 5 give 5, 3, 5
    assert QM.used_gap(np.array([12.0, 3.0, 70.0], np.float32), np.full(3, 5.0, np.float32)).tolist() == [5.0, 3.0, 5.0]
    with fake_heads():
        cv = make_ip(T, "cand-cv", model)
        run_rollout(T, world, p, cv, cfg)
        cq = make_ip(T, "cand-cvq", model, qhead=object(), q_const=5.0)
        run_rollout(T, world, p, cq, cfg)
    a, b = cv.ilog.arrays(), cq.ilog.arrays()
    # the tool's own rows: g_head 12 -> g_used 5; g_head 3 -> g_used 3; g_head 70 -> g_used 5 (never above the head gap)
    for gh, gu in ((12.0, 5.0), (3.0, 3.0), (70.0, 5.0)):
        sel = b["g_head"] == gh
        assert sel.any() and (b["g_used"][sel] == gu).all() and (b["q_head"][sel] == 5.0).all(), gh
    assert (b["g_src"] == b["g_used"]).all()
    # cand-cv logs no quantile and uses the head gap
    assert np.isnan(a["q_head"]).all() and (a["g_used"] == a["g_head"]).all()
    # the gate (gap at most 60 m) is on the HEAD gap: the active set is that of cand-cv, so no row of a 70 m head gap is active although min(70, 5) = 5 is below 60 (control q5)
    assert arrays_equal(a["active"], b["active"]) and not b["active"][b["g_head"] == 70.0].any() and b["active"].sum() > 5


def test_a_quantile_constant_of_1e9_is_the_identity_of_cand_cv_and_a_constant_of_0_makes_the_gap_zero(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_q2.npz", const_gap=20.0)
    head = random_head(model, probe_feats(T, world, p, cfg, model))
    cv = make_ip(T, "cand-cv", model)
    e_cv = run_rollout(T, world, p, cv, cfg)
    c1 = make_ip(T, "cand-cvq", model, qhead=head, q_const=1e9)
    e_1 = run_rollout(T, world, p, c1, cfg)
    a, b = cv.ilog.arrays(), c1.ilog.arrays()
    assert arrays_equal(e_cv, e_1), 'executed paths differ'
    for k in ("ci", "t0", "j", "active", "g_src", "c_src", "chosen_c", "chosen_idx", "any_safe", "safe", "gap_h", "v_h"):
        assert arrays_equal(a[k], b[k]), k
    assert (b["q_head"] == 1e9).all() and arrays_equal(b["g_used"], b["g_head"]), 'q1 columns'
    c0 = make_ip(T, "cand-cvq", model, qhead=head, q_const=0.0)
    e_0 = run_rollout(T, world, p, c0, cfg)
    z = c0.ilog.arrays()
    assert arrays_equal(z["active"], a["active"]), 'q2 active set'
    assert (z["g_used"][z["active"]] <= 0.0).all(), 'q2 g_used'
    assert a["active"].sum() > 5, ('active ticks', int(a["active"].sum()))
    assert not arrays_equal(e_0, e_cv), 'q2 executed paths equal'
    assert z["chosen_c"][z["active"]].mean() > a["chosen_c"][a["active"]].mean(), 'q2: a zero gap must ask for more deceleration on average'


def test_the_head_is_read_at_the_tick_frame_and_scaled_to_metres(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_q3.npz", const_gap=20.0)
    head = random_head(model, probe_feats(T, world, p, cfg, model))
    cap = Cap()
    ip = make_ip(T, "cand-cvq", model, base=cap, qhead=head)
    run_rollout(T, world, p, ip, cfg)
    log = ip.ilog.arrays()
    hist = int(model.wm.lead_head.hist)
    ten = head.tensors()
    exp, exp_first = [], []
    with torch.no_grad():
        for f in cap.feats:
            for sl, out in ((slice(-hist, None), exp), (slice(0, hist), exp_first)):
                z = model.wm.z1(model.adapter(f[:, sl].float()) + model.pos[None, None]).float().numpy()
                out.append(QM.functional_forward(ten, z[:, -hist:]).reshape(-1) * QM.GAP_SCALE_M)
    exp, exp_first = np.concatenate(exp), np.concatenate(exp_first)
    assert len(exp) == len(log["q_head"]) and np.isfinite(exp).all()
    assert np.abs(log["q_head"] - exp).max() < 1e-3, np.abs(log["q_head"] - exp).max()
    # the check can tell the tick frame from another frame window (when the batch holds more frames than the head reads)
    f0 = cap.feats[0]
    assert f0.shape[1] > hist and np.abs(exp - exp_first).max() > 1e-3, (tuple(f0.shape), np.abs(exp - exp_first).max())
    assert np.allclose(log["g_used"], np.minimum(log["g_head"], log["q_head"]), atol=0.0) or np.allclose(log["g_used"], np.minimum(log["g_head"], log["q_head"]), atol=1e-6)


def test_refusals_without_a_head_file_for_another_source_for_a_head_that_is_not_the_banked_or_not_in_band(T, world, tiny, tmp_path):
    model = tiny["model"]
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cvq", "", env={})
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cvq", "", env={"IP_QHEAD_CONST": "1e9"})
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cv", "", env={"IP_QHEAD_FILE": "x.npz"})
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("off", "", env={"IP_QHEAD_CONST": "0"})
    assert WQ.load_qhead_env("cand-cv", "", env={}) == (None, None, {})
    with pytest.raises(SystemExit):
        make_ip(T, "cand-cvq", model, qhead=None)
    with pytest.raises(SystemExit):
        make_ip(T, "cand-cv", model, qhead=object())
    torch.manual_seed(1)
    import pathlib
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(dir=str(tmp_path)))                # a fresh folder: a record of an earlier run must not be there
    f = str(d / "A_s0.qhead.npz")
    QM.save_head(f, QM.QHead(8, 5).eval())
    sha = QM.sha256_file(f)
    ck = str(d / "ck.pt")
    open(ck, "wb").write(b"checkpoint-bytes")
    ck_sha = QM.sha256_file(ck)

    def rec(**kw):
        r = {"head_file_sha256": {"x": sha}, "h5_check_coverage_in_0.15_0.25": True, "states_record": {"ckpt_sha256": ck_sha}}
        r.update(kw)
        json.dump(r, open(f[:-4] + ".json", "w"))
    with pytest.raises(SystemExit):                              # no record yet
        WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": f})
    rec()
    h, c, meta = WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": f, "IP_QHEAD_CONST": "1e9"})
    assert h.hist == 5 and c == 1e9 and meta["qhead_sha256"] == sha
    h, c, meta = WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": f})
    assert c is None
    rec(**{"h5_check_coverage_in_0.15_0.25": False})
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": f})
    rec(head_file_sha256={"x": "0" * 64})
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": f})
    rec(states_record={"ckpt_sha256": "1" * 64})
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": f})
    rec()
    e = d / "eval_heads"
    e.mkdir(exist_ok=True)
    fe = str(e / "A_s0.qhead.npz")
    QM.save_head(fe, QM.QHead(8, 5).eval())
    json.dump({"head_file_sha256": {"x": QM.sha256_file(fe)}, "h5_check_coverage_in_0.15_0.25": True, "states_record": {"ckpt_sha256": ck_sha}}, open(fe[:-4] + ".json", "w"))
    with pytest.raises(SystemExit):
        WQ.load_qhead_env("cand-cvq", ck, env={"IP_QHEAD_FILE": fe})


def test_other_sources_do_not_see_the_head(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_q4.npz", const_gap=20.0)
    with fake_heads():
        cv = make_ip(T, "cand-cv", model)
        run_rollout(T, world, p, cv, cfg)
    a = cv.ilog.arrays()
    assert cv.q_on is False and cv.cand_source == "cand-cv" and np.isnan(a["q_head"]).all() and arrays_equal(a["g_used"], a["g_head"]) and arrays_equal(a["g_src"], a["g_head"])
    with fake_heads():
        cq = make_ip(T, "cand-cvq", model, qhead=object(), q_const=1.0)
    assert cq.q_on is True and cq.cand_source == "cand-cv"
