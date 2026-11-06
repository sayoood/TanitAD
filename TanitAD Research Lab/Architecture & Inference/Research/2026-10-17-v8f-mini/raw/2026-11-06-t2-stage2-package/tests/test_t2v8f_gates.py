"""Gates G1-G4 for the v8F-mini T2 policy (E1) and the in-loop encoder (E2). CPU only. Every `check_*` takes `mut=None`;
`code/run_mutations.py` calls each with a deliberate regression and requires an AssertionError (RED). Run: `python -I -m pytest tests -q`
from the t2v8f dir with CUDA_VISIBLE_DEVICES=-1 (pytest --basetemp parent must exist)."""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import t2a_v8f_policy as P  # noqa: E402

TREE = P.bootstrap()
ART = "E:/Projects/TanitAD-artifacts"
CKPT = ART + "/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"
CACHE = ART + "/sfresh_inputs/cache"
TRACKS = ART + "/sfresh_inputs/inputs/native_tracks.pt"
DATA = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TOOL = ART + "/dagland/motion_c/lf6_build/tool/v8f_t1_eval_ri_lf6.py"
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

_C = {}


def world():
    if _C:
        return _C
    spec = importlib.util.spec_from_file_location("t1tool", TOOL)
    ri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ri)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(DATA + "/join_eval141.npz")
    store = D.ClipStore(CACHE, "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(DATA + "/eval_grid_windows.json", J)
    proxy = ri._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    _C.update(ri=ri, model=model, cfg=cfg, tracks=tracks, J=J, store=store, wins=wins, proxy=proxy, ds=ds)
    return _C


def pick(n):
    w = world()["wins"]
    return [w[i] for i in np.linspace(40, len(w) - 40, n).round().astype(int)]


def banked_tokens(clip):
    return torch.load(f"{CACHE}/eval/clips/{clip}.pt", map_location="cpu", weights_only=True)["pool4x10"]


def make_policy(win, mut=None, arm="model", schedule="periodic"):
    W = world()
    ci, t0 = win[0], win[1]
    clip = str(W["J"]["clip"][ci])
    o, n = int(W["J"]["off"][ci]), int(W["J"]["n"][ci])
    pol = P.V8FMiniT2Policy(CKPT, tree=TREE, arm=arm, schedule=schedule, limit_mode="off" if mut == "limit_off" else "causal")
    tr = W["tracks"][clip][:2 * t0 + 1]
    if mut == "stale_history":
        tr = tr[:-2]                                           # NOW one cache step too old
    nav = W["J"]["nav"][o:o + n]
    pol.seed_track(tr, nav_rows=nav)
    pol.set_nav(row=nav[t0])
    tok = banked_tokens(clip)
    for s in range(t0 + 1):
        pol.push_tokens(s, tok[s - 1] if (mut == "feat_off_by_one" and s == t0) else tok[s])
    if mut == "no_yaw_rate":
        orig = pol._steps
        def steps():
            st = orig()
            st["p10f"] = st["p10f"].copy()
            st["p10f"][:, 2] = 0.0
            return st
        pol._steps = steps
    return pol, clip, tok


def tool_tick0(win, ticks=1, m=30, scene="logged"):
    """The T1-kin tool's own items and plans at ticks 0..ticks-1 (the reindexed tool, floor flags off), with the executed track per tick."""
    W = world()
    ri = W["ri"]
    got = []
    def hook(n, batch, plan, states):
        st = states[0]
        got.append({"batch": {k: v.clone() for k, v in batch.items()}, "plan": plan[0].copy(), "track": st.track[:2 * st.j + 1].copy(), "j": st.j})
    st = ri.make_states(W["store"], W["tracks"], W["J"], [win], "deadreckon")
    ri.rollout(ri.model_planner(W["model"], "cpu"), st, W["ds"], W["proxy"], m=m, schedule="periodic", scene=scene, limit="causal", device="cpu", hook=hook)
    return got[:ticks]


def eq_batch(a, b, what=""):
    assert set(a) == set(b), f"{what}: key sets differ {sorted(set(a) ^ set(b))}"
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, f"{what}: {k} dtype/shape {a[k].dtype} {tuple(a[k].shape)} vs {b[k].dtype} {tuple(b[k].shape)}"
        assert torch.equal(a[k], b[k]), f"{what}: key {k!r} differs (max |d| {float((a[k].double() - b[k].double()).abs().max()):.3g})"


# ------------------------------------------------------------------------------------------------ G1
def check_g1_tick0(mut=None, n=6):
    """The policy fed the BANKED tokens gives the tool's tick-0 batch (every MODEL key, bit for bit) and plan (bit for bit, float32)."""
    W = world()
    for win in pick(n):
        ref = tool_tick0(win)[0]
        pol, _, _ = make_policy(win, mut)
        batch = pol.build_batch(0, 0)
        eq_batch(ref["batch"], batch, f"G1 window {win[:2]}")
        plan, _ = pol.forward_batch(batch)
        assert np.array_equal(plan.astype(np.float32), ref["plan"]), f"G1 plan differs, max |d| {np.abs(plan - ref['plan']).max():.3g} m"
    return n


def check_g1_ticks(mut=None, n=3):
    """Ticks 1 and 2 (m = 10): items rebuilt from the tool's EXECUTED track -- yaw-rate history, dead-reckoned nav, causal bucket, ages."""
    for win in pick(n):
        ref = tool_tick0(win, ticks=3, m=10)
        pol, clip, tok = make_policy(win, mut)
        for n_, r in enumerate(ref):
            if n_:
                for ri_ in range(len(pol.track), len(r["track"])):         # the harness calls set_history on EVERY raw tick: one new row each
                    pol.set_history(r["track"][:ri_ + 1][-16:], n_new=1)
                if mut == "no_deadreckon":
                    pol.nav_log = None
                    pol._nav_rows = lambda st, j, idc, _p=pol: np.tile(_p.nav0, (len(idc), 1))
                for s in range(pol.t0_step + 1 + 10 * (n_ - 1), r["j"] + 1):
                    pol.push_tokens(s, tok[s])
            a_t, a_s = P.ages_for(r["j"] - pol.t0_step, "periodic")
            batch = pol.build_batch(a_t, a_s)
            eq_batch(r["batch"], batch, f"G1 tick {n_} window {win[:2]}")
            plan, _ = pol.forward_batch(batch)
            assert np.array_equal(plan.astype(np.float32), r["plan"]), f"G1 tick {n_} plan differs, max |d| {np.abs(plan - r['plan']).max():.3g} m"
    return n


def check_g1_hacc(mut=None):
    """The HACC post-map acts inside forward_infer for both: same model, same table -> same plan; and it moves the plan (not a no-op)."""
    W = world()
    win = pick(1)[0]
    pol, _, _ = make_policy(win)
    pol2, _, _ = make_policy(win)
    w = np.full((30, 2), 0.5)
    pol2.model.set_hacc_weights(w)
    b = pol2.build_batch(0, 0)
    pl_h, ex = pol2.forward_batch(b)
    pl_0, _ = pol.forward_batch(pol.build_batch(0, 0))
    ref_model = W["model"]
    ref_model.set_hacc_weights(w)
    try:
        with torch.no_grad():
            ref = ref_model.forward_infer(b)["traj30"].float().numpy()[0]
    finally:
        ref_model.clear_hacc_weights()
    assert ex["hacc_active"] and np.array_equal(pl_h.astype(np.float32), ref), "HACC plan differs from the model's own post-map"
    assert float(np.abs(pl_h - pl_0).max()) > 1e-3 if mut != "hacc_noop" else False, "HACC post-map did not move the plan"


# ------------------------------------------------------------------------------------------------ G3
def check_g3_zero(mut=None):
    """arm='p10' returns EXACTLY the prior the tool's floor planner computes (max |d| == 0.0), and the model arm is not that prior."""
    W = world()
    for win in pick(4):
        pol, _, _ = make_policy(win, arm="p10")
        b = pol.build_batch(0, 0)
        if mut == "returns_model":
            pol.arm = "model"
        plan, ex = pol.forward_batch(b)
        ref = W["ri"].floor_planner(W["model"], "cpu")(b)[0]
        if mut == "prior_eps":
            plan = plan + 1e-6
        assert float(np.abs(plan.astype(np.float32) - ref).max()) == 0.0, f"G3: the P10 arm is not the prior (max |d| {np.abs(plan - ref).max():.3g})"
        pm, _ = make_policy(win)[0].forward_batch(b)
        assert float(np.abs(pm - ref).max()) > 1e-3, "G3: the model arm equals the prior; the control would be vacuous"


# ------------------------------------------------------------------------------------------------ G4
def check_g4_keys(mut=None):
    """A peeking variant is caught: (a) an extra batch key, (b) a forward that READS a key outside INFERENCE_INPUT_KEYS. The honest run passes."""
    win = pick(1)[0]
    pol, _, _ = make_policy(win)
    if mut == "guard_off":
        pol._guard = lambda *a, **k: None
    b = pol.build_batch(0, 0)
    pol.forward_batch(b)                                          # honest: must pass
    caught = 0
    bad = dict(b)
    bad["gt"] = torch.zeros(1, 30, 2)
    try:
        pol.forward_batch(bad)
    except P.KeyGuardError:
        caught += 1
    orig = pol.model.forward_infer
    def peek(batch):
        g = M._KeyGuard(batch)
        _ = g["gt"]
        out = orig({k: v for k, v in batch.items() if k != "gt"})
        out["_keys_read"] = set(out["_keys_read"]) | g.read
        return out
    pol.model.forward_infer = peek
    try:
        pol.forward_batch(bad)
    except P.KeyGuardError:
        caught += 1
    assert caught == 2, f"G4: the peeking variants were caught {caught}/2 times"


def test_g1_tick0():
    assert check_g1_tick0() == 6


def test_g1_ticks():
    assert check_g1_ticks() == 3


def test_g1_hacc():
    check_g1_hacc()


def test_g3_zero():
    check_g3_zero()


def test_g4_keys():
    check_g4_keys()


def test_g2_encoder():
    import t2a_g2
    r = t2a_g2.run_g2()
    assert r["pass"], r
