"""Unit tests for ENS-SCORE (PREREG_ENSSCORE.md section 6, "pre-run unit tests"). Run in the SEAM's venv (driverl-eval: torch + the DriveRL
nuplan fork), CPU only, under pytest or directly:

    ENS_PKG=<PK of the tip export> CUDA_VISIBLE_DEVICES=-1 python test_ens_score.py

aggregate-on-probabilities identity, self-ensemble identity, the member term reaches the pick (the MUTATION target), RED, completeness and
shapes, member-config refusal, the patched infer end to end (the seam's caller path: captured inputs incl. kwargs, the member's own forward,
submitted = the primary's planned proposal), the member's own ego-yaw shim, and the controls' independent numpy recompute."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("ENS_PKG", "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, HERE)
import ens_lib as EL  # noqa: E402

TINY = dict(backbone="vits16", width=32, depth=1, heads=2, img_h=32, img_w=32, n_cameras=1, cameras=("CAM_F0",), dec_depth=1,
            n_proposals=64, horizon_steps=20, reg_heads=2, n_registers=2, lora_rank=2)
TOK = "00000000000000aa"


def _tiny(seed=3):
    import torch
    from model import REFe, REFeConfig
    torch.manual_seed(seed)
    m = REFe(REFeConfig(**TINY)).eval()
    with torch.no_grad():                       # move every tensor off its init so the heads are not degenerate
        g = torch.Generator().manual_seed(seed + 2)
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    cap = {}
    m.score_dec[0].register_forward_pre_hook(lambda mod, args: cap.__setitem__("sctx", args[1]))
    return m, cap


def _inputs(m, seed=9):
    import torch
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, 1, 3, 32, 32, generator=g)
    ego = torch.randn(1, m.cfg.ego_dim, generator=g)
    ego[:, 4] = 0.0                              # the yaw slot arrives empty (the shim fills it)
    if getattr(m.cfg, "nav_mode", "goal") == "command":
        goal = torch.zeros(1, m.cfg.n_commands)
        goal[0, seed % m.cfg.n_commands] = 1.0
    else:
        goal = torch.randn(1, 2 * m.cfg.n_goal_points, generator=g)
    return img, ego, goal


def _stub_planner(model):
    import planner as PL
    st = types.SimpleNamespace(model=model, cfg=model.cfg, rule="navsim_v1", V1_W=PL.REFePlanner.V1_W, PDM_W=PL.REFePlanner.PDM_W, select="best",
                               repair_last_heading=True, per_sample_calib=False, trained=True)
    st.aggregate = types.MethodType(PL.REFePlanner.aggregate, st)
    st.executed = types.MethodType(PL.REFePlanner.executed, st)
    st._pick = types.MethodType(PL.REFePlanner._pick, st)
    return st


def _seam():
    sys.path.insert(0, os.path.join(PKG, "eval"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("seam_t", os.path.join(PKG, "eval", "refe_navtest_seam.py"))
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


def _ctrl():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ens_ctrl_t", os.path.join(HERE, "ens_ctrl.py"))
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    return c


def _np_ens_pick(s_a, s_b, margin=False):
    """The independent numpy recompute of the pre-registered rule; with margin=True also the top-2 gap (near-ties are not compared)."""
    c = _ctrl()
    sa, sb = s_a.float().numpy(), s_b.float().numpy()
    ag = c.agg_np((c.sig(sa) + c.sig(sb)) / np.float32(2.0))
    k = int(ag.argmax())
    if not margin:
        return k
    top = np.sort(ag.astype(np.float64))[-2:]
    return k, float(top[1] - top[0])


# ------------------------------------------------------------------ the combination
def test_aggregate_p_is_the_planners_aggregate_bitwise():
    import torch
    pl = _stub_planner(_tiny()[0])
    g = torch.Generator().manual_seed(1)
    for scale in (0.5, 3.0, 20.0, 120.0):                     # incl. saturating logits (sigmoid -> 0 / subnormal / 1)
        s = scale * torch.randn(64, 6, generator=g)
        a1 = pl.aggregate(s.unsqueeze(0))[0]
        a2 = EL.aggregate_p(pl, s.sigmoid())
        assert torch.equal(a1, a2), float((a1 - a2).abs().max())
        assert EL.pick_p(pl, s.sigmoid()) == EL.pick_logits(pl, s) == pl._pick(s.unsqueeze(0))


def test_self_ensemble_is_bitwise_the_member():
    import torch
    pl = _stub_planner(_tiny()[0])
    g = torch.Generator().manual_seed(2)
    for scale in (1.0, 40.0, 150.0):
        p = (scale * torch.randn(64, 6, generator=g)).sigmoid()
        assert torch.equal(EL.ens_probs(p, p), p)
        assert torch.equal(EL.aggregate_p(pl, EL.ens_probs(p, p)), EL.aggregate_p(pl, p))


def test_member_term_reaches_the_pick():
    """The MUTATION target: with the member term removed from ens_probs, k_ens == the primary-alone pick on every draw and != the numpy
    recompute of (pA + pB) / 2, so this test must go RED."""
    import torch
    ma, ca = _tiny(3)
    mb, cb = _tiny(11)
    pa, pb = _stub_planner(ma), _stub_planner(mb)
    differs, n, checked = 0, 16, 0
    for i in range(n):
        x = _inputs(ma, seed=100 + i)
        with torch.no_grad():
            ta, sca = ma(*x)
            tb, scb = mb(*x)
        rec = EL.ens_step(pa, pb, ta[0], sca[0], pa._pick(sca), ca["sctx"], tb[0], scb[0], pb._pick(scb), cb["sctx"], _seam().to_navsim)
        k_np, gap = _np_ens_pick(torch.from_numpy(rec["s_a"]), torch.from_numpy(rec["s_b"]), margin=True)
        if gap > 1e-5:
            checked += 1
            assert rec["k_ens"] == k_np, (i, rec["k_ens"], k_np, gap)
        assert rec["k_id"] == rec["k_ship"] and rec["k_b_own_id"] == rec["k_b_own"], i
        assert rec["id_maxdiff_a"] == 0.0 and rec["id_maxdiff_b"] == 0.0, (rec["id_maxdiff_a"], rec["id_maxdiff_b"])
        assert not np.array_equal(rec["s_a"], rec["s_b"])
        differs += int(rec["k_ens"] != rec["k_a_only"])
    assert checked >= n // 2, f"only {checked}/{n} draws had a top-2 gap > 1e-5"
    assert differs >= 1, f"the ensemble never moved the pick away from the primary alone in {n} draws"


def test_red_misalignment_changes_the_pick():
    import torch
    ma, ca = _tiny(3)
    mb, cb = _tiny(11)
    pa, pb = _stub_planner(ma), _stub_planner(mb)
    moved = 0
    for i in range(16):
        x = _inputs(ma, seed=200 + i)
        with torch.no_grad():
            ta, sca = ma(*x)
            tb, scb = mb(*x)
        rec = EL.ens_step(pa, pb, ta[0], sca[0], pa._pick(sca), ca["sctx"], tb[0], scb[0], pb._pick(scb), cb["sctx"], _seam().to_navsim)
        moved += int(rec["k_red"] != rec["k_ens"])
    assert moved >= 1


def test_completeness_shapes_and_member_config_refused():
    import torch
    for bad in (torch.zeros(63, 20, 3), torch.zeros(64, 19, 3), torch.zeros(1, 64, 20, 3)):
        try:
            EL.check_set(bad)
        except ValueError as e:
            assert "COMPLETE" in str(e)
        else:
            raise AssertionError(f"set {tuple(bad.shape)} was not refused")
    try:
        EL.check_scores(torch.zeros(64, 7), "x")
    except ValueError:
        pass
    else:
        raise AssertionError("a 7-output scorer was not refused")
    pa, pb = _stub_planner(_tiny(3)[0]), _stub_planner(_tiny(11)[0])
    EL.check_members(pa, pb)                                   # same recipe, different weights: admitted
    import dataclasses
    pb.cfg = dataclasses.replace(pb.cfg, dec_depth=2)
    for mut in ("cfg", "rule"):
        q = _stub_planner(_tiny(11)[0])
        if mut == "cfg":
            q.cfg = pb.cfg
        else:
            q.rule = "v2_shape"
        try:
            EL.check_members(pa, q)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a member differing in {mut} was not refused")


# ------------------------------------------------------------------ the seam's caller path
def _fake_planner_class(seeds: dict):
    import torch
    import planner as PL

    class FakePlanner:
        V1_W, PDM_W = PL.REFePlanner.V1_W, PL.REFePlanner.PDM_W
        aggregate = PL.REFePlanner.aggregate
        executed = PL.REFePlanner.executed
        _pick = PL.REFePlanner._pick

        def __init__(self, checkpoint=None, device="cpu", rule="navsim_v1", select="best"):
            self.model = _tiny(seeds[checkpoint])[0]
            self.cfg, self.device, self.rule, self.select = self.model.cfg, device, rule, select
            self.repair_last_heading, self.per_sample_calib, self.trained = True, False, True
            self._scenario = types.SimpleNamespace(_initial_lidar_token=TOK)
            self.seen = []

        def infer(self, ego, img):
            x = img
            with torch.no_grad():
                t, score = self.model(x[0], x[1], x[2], calib=None)      # the calib KWARG, as REFePlanner.infer passes it
            v, sc = t[0], score[0]
            self.seen.append((v, sc))
            return v, sc, self._pick(score)

    return FakePlanner


def _run_patched(yaw: bool):
    import torch
    import ens_select as ES
    import ego_yaw as EYR
    FP = _fake_planner_class({"A": 3, "B": 11})
    fake_pl = types.SimpleNamespace(REFePlanner=FP)
    tdir = tempfile.mkdtemp(prefix="ens_t_", dir=os.environ.get("TMP"))
    tpath = os.path.join(tdir, "yaw.bin")
    with open(tpath, "wb") as f:
        f.write(b"fake yaw table for the unit test")
    tsha = hashlib.sha256(open(tpath, "rb").read()).hexdigest()
    table = {TOK: 0.125}
    shim_b_box = {}

    def from_ck(ck, tp, allow_incoming=False, meta=None):
        if not yaw:
            return None
        s = EYR.EgoYawShim(table, tsha, 3.0, allow_incoming=allow_incoming)
        shim_b_box["s"] = s
        return s
    ey = types.SimpleNamespace(EgoYawShim=types.SimpleNamespace(from_checkpoint=from_ck), _scenario_token=EYR._scenario_token)
    a = types.SimpleNamespace(member_ckpt="B", yaw_table=tpath)
    st = ES.State(a)
    ES.install(a, st, fake_pl, EL, ey, _seam().to_navsim)
    p = FP(checkpoint="A", device="cpu", rule="navsim_v1", select="best")
    if yaw:
        sa = EYR.EgoYawShim(table, tsha, 2.0)
        sa.install(p.model, token_source=lambda: TOK)
        p.ego_yaw_shim = sa
    x = _inputs(p.model, seed=7)
    out_traj, out_score, k = p.infer(None, x)
    os.remove(tpath)
    os.rmdir(tdir)
    return p, x, out_traj, out_score, k, st.recs[TOK], shim_b_box


def test_patched_infer_end_to_end_no_yaw():
    import torch
    import planner as PL
    p, x, out_traj, out_score, k, rec, _ = _run_patched(yaw=False)
    assert out_traj is p.seen[0][0] and out_score is p.seen[0][1]          # the seam gets the primary's own set and own logits
    assert k == rec["k_ens"] and rec["k_id"] == rec["k_ship"] and rec["k_b_own_id"] == rec["k_b_own"]
    assert torch.equal(p.executed(out_traj, k), PL.repair_last_heading(p.seen[0][0][k]))   # the PLANNED proposal of the PRIMARY
    b = p._ens_b
    with torch.no_grad():
        tb, sb = b.model(x[0], x[1], x[2], calib=None)                      # the member's direct forward on the same inputs
    assert np.array_equal(rec["s_b_fwd"], sb[0].numpy()), "the member did not receive the primary's exact inputs"
    assert np.array_equal(rec["props8_b"][rec["k_b_own"]], rec["ship8_b"])
    assert rec["k_ens"] == _np_ens_pick(torch.from_numpy(rec["s_a"]), torch.from_numpy(rec["s_b"]))


def test_member_gets_its_own_yaw_shim():
    import torch
    p, x, out_traj, out_score, k, rec, box = _run_patched(yaw=True)
    b = p._ens_b
    assert getattr(b, "ego_yaw_shim", None) is box["s"]
    ego_b = x[1].clone()
    ego_b[:, 4] = 3.0 * 0.125                                               # the MEMBER's scale x the token's table value
    box["s"].remove()
    with torch.no_grad():
        tb, sb = b.model(x[0], ego_b, x[2], calib=None)
        tb0, sb0 = b.model(x[0], x[1], x[2], calib=None)                    # no yaw: must differ (the shim mattered)
    assert np.array_equal(rec["s_b_fwd"], sb[0].numpy()), "the member's forward did not apply its own yaw"
    assert not np.array_equal(rec["s_b_fwd"], sb0[0].numpy())
    ego_a = x[1].clone()
    ego_a[:, 4] = 2.0 * 0.125
    p.ego_yaw_shim.remove()
    with torch.no_grad():
        ta, sa = p.model(x[0], ego_a, x[2], calib=None)
    assert torch.equal(out_score, sa[0]), "the primary's forward did not apply its own yaw"


def test_ctrl_numpy_recompute_matches_torch():
    import torch
    pl = _stub_planner(_tiny()[0])
    c = _ctrl()
    g = torch.Generator().manual_seed(4)
    for scale in (1.0, 8.0, 60.0):
        sa, sb = scale * torch.randn(64, 6, generator=g), scale * torch.randn(64, 6, generator=g)
        at = EL.aggregate_p(pl, EL.ens_probs(sa.sigmoid(), sb.sigmoid())).numpy()
        an = c.agg_np((c.sig(sa.numpy()) + c.sig(sb.numpy())) / np.float32(2.0))
        assert np.abs(at.astype(np.float64) - an).max() <= 1e-6, np.abs(at - an).max()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print(f"ZZENS_TESTS {'PASS' if fails == 0 else 'FAIL'} failed={fails}", flush=True)
    sys.exit(1 if fails else 0)
