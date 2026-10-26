"""O-RECOVER (PREREG_V8F_HACC_MC section 7): an ego-state perturbation of the kinematic inputs, default OFF.  `--ego-perturb-p` / `-vscale LO HI` / `-vmax` / `-adrop` / `-yawrate-std` / `-seed`, and the fine-tune's start state `--init-ckpt`.

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a lever whose library function is correct and whose CALLER never reaches it. So the lever is proven hop by hop FROM ARGV,
and every proof has a deliberate-regression arm (`raw/2026-10-26-orecover/mutate_orecover.py` deletes the single line and shows the tests RED):

  argv --ego-perturb-* -> make_ego_perturb (the perturber) -> declared_vs_built (G-DVB: FLAG_CHECKS entry + bad_ego_perturb reads the BUILT perturber back) -> ego_perturb_probe (G-LIVE on a PERTURBED batch; the
  perturbed tensors reach the loss gradient; controls read EXACTLY 0.0) -> `b = ego.apply(b, step)` in the training loop -> V8FMini.loss -> .backward()

  (a) defaults    the default run reproduces the saved pre-change trainer + data module (`raw/2026-10-26-orecover/pre_orecover_*.py.ref`): the loss trace and the final weights, bit for bit
  (b) the batch   p = 1: every ego tensor changes where the LOGGED v0 < vmax and none where v0 >= vmax; targets and labels are the same objects; the rebuild equals an independent recomputation
  (c) the loss    the perturbed input reaches the loss gradient (non-zero difference), p = 0 reads exactly 0.0, the frozen prior and the WM auxiliary read exactly 0.0
  (d) G-DVB       each flag is read back from the BUILT perturber: a mismatch is refused, for each flag; a flag that builds nothing, a no-op law and a wrong kin_mode are refused
  (e) the caller  a real `TR.main` run applies the perturbation to the batch the loss receives; the control counts (perturbed / eligible) are exactly 0 at p = 0 and equal at p = 1
  (f)-(h) --init-ckpt  the first optimiser step sees exactly the checkpoint's parameters with an empty optimiser state at step 0; a config mismatch is refused; resume wins; the planned fine-tune argv is accepted
Everything that needs the saved pre-change modules is SKIPPED, not failing, in a checkout that has only `stack/`."""
from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
RAWO = PKG / "raw" / "2026-10-26-orecover"
REF_TRAIN = RAWO / "pre_orecover_v8f_mini_train.py.ref"
REF_DATA = RAWO / "pre_orecover_v8f_mini_data.py.ref"
needs_ref = pytest.mark.skipif(not (REF_TRAIN.exists() and REF_DATA.exists()), reason="raw/2026-10-26-orecover (the saved pre-change trainer + data module) is not in this checkout")
HERE_T = Path(__file__).resolve().parent
DATA_P = REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py"
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_ego", HERE_T / "test_v8f_mini.py")                 # the synthetic world, the batch builder and the tiny model the other v8F-mini tests use
world, TINY = _T.world, _T.TINY


def make_batch(world, n=8, seed=0, hmax=10, a_t=1, a_s=3):
    """A collated batch of the synthetic world in the YAW representation the lever is defined on (kin3 = (v, a, kappa), act = (a, omega), the P10 features of that kin3): the trainer's own ClipStore."""
    root = world["root"]
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(Path(root) / "native_tracks.pt"))
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], a_t, a_s) for i in idx]), J, ds


def tiny_model(J, **kw):
    cfg = M.V8FMiniConfig(**{**TINY, "kin_mode": "yaw", **kw})
    mu, sd = M.cons_stats(J)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd), cfg
COMMON = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
FLAGS = {"ego_perturb_p": ["--ego-perturb-p", "0.3"], "ego_perturb_vscale": ["--ego-perturb-vscale", "0.6", "0.9"], "ego_perturb_vmax": ["--ego-perturb-vmax", "5.0"],
         "ego_perturb_adrop": ["--ego-perturb-adrop", "1.0"], "ego_perturb_yawrate_std": ["--ego-perturb-yawrate-std", "2.0"], "ego_perturb_seed": ["--ego-perturb-seed", "7"]}


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def _J(world):
    return D.load_join(Path(world["root"]) / "join_train.npz")


def _clone(x):
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x.clone() if torch.is_tensor(x) else copy.deepcopy(x)


def _same(x, y):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if torch.is_tensor(x):
        return x.shape == y.shape and x.dtype == y.dtype and bool(((x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)).all())
    return x == y


def _changed(x, y):
    """[B] bool: the row differs (NaN == NaN)."""
    same = ((x == y) | (torch.isnan(x) & torch.isnan(y))) if x.is_floating_point() else (x == y)
    return (~same).reshape(x.shape[0], -1).any(1)


@pytest.fixture(scope="module")
def batch(world):
    b, _, _ = make_batch(world, n=32, seed=0)                                    # hmax 10, a_t = 1, a_s = 3
    v0 = b["kin3"][:, 0]
    assert int((v0 < 8.0).sum()) > 4 and int((v0 >= 8.0).sum()) > 4, "the fixture batch must hold windows on both sides of the default vmax"
    return b


# ------------------------------------------------------------------------------------------------ the partition
def test_every_inference_key_is_rebuilt_kept_or_a_declared_record():
    """No ego tensor the model may read is left unperturbed by omission: a key added to INFERENCE_INPUT_KEYS must be put in one of the three sets (and so thought about)."""
    R, K, A = set(D.EGO_PERTURB_REBUILT), set(D.EGO_PERTURB_KEPT), set(D.EGO_PERTURB_RECORD)
    assert not (R & K) and not (R & A) and not (K & A), "the three sets are disjoint"
    assert set(M.INFERENCE_INPUT_KEYS) <= R | K | A, sorted(set(M.INFERENCE_INPUT_KEYS) - (R | K | A))
    assert R == {"p10f", "base", "kin3", "p10_t", "pose", "pose_t", "pose_s", "nav"} and A == {"act"}                # literals authored here
    assert K == {"feats", "frame_ok", "limit", "limit_ok", "a_t", "a_s", "pose_now"}, "nav moved from KEPT to REBUILT (its progress channel carries the travelled distance)"


def test_forward_infer_reads_only_rebuilt_or_kept_keys(world, batch):
    J = _J(world)
    model, _ = tiny_model(J, wm="off")
    model.eval()
    with torch.no_grad():
        keys = set(model.forward_infer(batch)["_keys_read"])
    assert keys <= set(D.EGO_PERTURB_REBUILT) | set(D.EGO_PERTURB_KEPT), sorted(keys)
    assert "act" not in keys and "pose" not in keys, "act is the WM's transition record; pose (the history) is not read"
    assert {"p10f", "base", "pose_t", "pose_s"} <= keys, "the ego tensors the model DOES read"


# ------------------------------------------------------------------------------------------------ (b) the batch
def _np_kappa(om, v):
    return np.clip(om / np.maximum(v, 2.0), -0.3, 0.3)                           # literals authored here: POSE_KAPPA_V_FLOOR 2.0, POSE_KAPPA_CAP 0.3


def test_the_kappa_constants_are_the_ones_the_expectations_assume():
    from tanitad.models import kinematic_prior as KP
    assert (KP.POSE_KAPPA_V_FLOOR, KP.POSE_KAPPA_CAP) == (2.0, 0.3)


@pytest.mark.parametrize("std", [0.0, 3.0])
def test_p1_changes_every_ego_input_below_vmax_and_none_above_and_the_targets_are_byte_identical(batch, std):
    pre = _clone(batch)
    ego = D.EgoPerturb(p=1.0, yawrate_std_deg=std, seed=3)
    out = ego.apply(batch, 11)
    assert _same(batch, pre), "apply does not modify its input"
    v0 = batch["kin3"][:, 0]
    sel = ego.last["sel"]
    assert bool((sel == (v0 < 8.0)).all()), "p = 1: the selected rows are exactly the eligible ones (logged v0 < vmax)"
    assert ego.last["n_selected"] == ego.last["n_eligible"] == int((v0 < 8.0).sum()) and 0 < ego.last["n_selected"] < 32
    for k in D.EGO_PERTURB_REBUILT:
        ch = _changed(out[k], batch[k])
        assert not bool(ch[~sel].any()), f"{k}: a row with logged v0 >= vmax changed"
    for k in ("p10f", "base", "kin3", "p10_t"):
        assert bool(_changed(out[k], batch[k])[sel].all()), f"{k}: every perturbed row must differ"
    for k in batch:                                                              # everything else: the SAME object (targets, labels, scene, route, ages, the anchor pose, the WM record)
        if k not in D.EGO_PERTURB_REBUILT:
            assert out[k] is batch[k], k
    for k in ("gt", "gt_ok", "s_h", "lead_gap_h", "lab", "lab_t", "lab_s", "limit", "limit_ok", "feats", "frame_ok", "act", "pose_now", "a_t", "a_s"):
        assert _same(out[k], pre[k]), f"{k} must be byte-identical"
    assert torch.equal(out["nav"][..., :3], pre["nav"][..., :3]) and torch.equal(out["nav"][..., 4], pre["nav"][..., 4]), "nav: the token and the validity channels are never touched"
    assert torch.equal(out["nav"][:, -1], pre["nav"][:, -1]), "nav: the NOW frame is the anchor"


@pytest.mark.parametrize("std", [0.0, 3.0])
def test_the_rebuild_equals_an_independent_recomputation(batch, std):
    ego = D.EgoPerturb(p=1.0, yawrate_std_deg=std, seed=5)
    out = ego.apply(batch, 2)
    L = ego.last
    F = batch["p10f"].shape[1]
    base_fn = D.kdx_base
    for b in torch.nonzero(L["sel"]).flatten().tolist():
        s, d, n = float(L["scale"][b]), float(L["drop"][b]), float(L["yaw_offset"][b])
        t = int(batch["t"][b])
        for f in range(F):
            valid = (t - (F - 1) + f) >= 1 and bool(batch["frame_ok"][b, f])
            v, a, om = float(batch["p10f"][b, f, 0]), float(batch["p10f"][b, f, 1]), float(batch["act"][b, f, 1])
            if not valid:
                assert torch.equal(out["p10f"][b, f], batch["p10f"][b, f]), "a padded / NaN-row frame is left as it was"
                continue
            v2, a2 = s * v, a - d
            k2 = float(_np_kappa(s * om + n, v2))
            want = np.array([v2, a2, k2, v2 * v2, v2 * a2, v2 * k2, v2 * v2 * k2, a2 * k2], np.float64)
            np.testing.assert_allclose(out["p10f"][b, f].double().numpy(), want, rtol=2e-6, atol=2e-6)
        # the NOW row: kin3, p10_t and the kd_x base are the same state
        now = out["p10f"][b, -1].double().numpy()
        np.testing.assert_allclose(out["kin3"][b].double().numpy(), now[:3], rtol=2e-6, atol=2e-6)
        assert torch.equal(out["p10_t"][b], out["p10f"][b, -1])
        np.testing.assert_allclose(out["base"][b].numpy(), base_fn(out["kin3"][b].double().numpy()[None], "kappa")[0], rtol=1e-4, atol=1e-4)
        # speed and acceleration obey the law at NOW
        v_now, a_now = float(batch["kin3"][b, 0]), float(batch["kin3"][b, 1])
        assert math.isclose(float(out["kin3"][b, 0]), s * v_now, rel_tol=1e-5, abs_tol=1e-5) and math.isclose(float(out["kin3"][b, 1]), a_now - d, rel_tol=1e-5, abs_tol=1e-5)
        assert 0.5 - 1e-9 <= s <= 1.0 and 0.0 <= d <= 1.5 + 1e-9
        # odometry: displacement from NOW scales by s, the heading history shifts by -n x age, NOW is the anchor; pose_t / pose_s are read from that history
        ps = out["pose"][b].double().numpy()
        po = batch["pose"][b].double().numpy()
        np.testing.assert_allclose(ps[-1], po[-1], rtol=0, atol=0)
        for f in range(F):
            if (t - (F - 1) + f) >= 1 and bool(batch["frame_ok"][b, f]):
                np.testing.assert_allclose(ps[f, :2] - po[-1, :2], s * (po[f, :2] - po[-1, :2]), rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(ps[f, 2], po[f, 2] - n * (F - 1 - f) * 0.2, rtol=1e-5, atol=1e-5)
        assert torch.equal(out["pose_t"][b], out["pose"][b, F - 1 - int(batch["a_t"][b])]) and torch.equal(out["pose_s"][b], out["pose"][b, F - 1 - int(batch["a_s"][b])])
        assert torch.equal(out["pose_now"][b], batch["pose_now"][b])


def test_with_the_lateral_lever_off_a_speed_perturbation_leaves_the_path_curvature_alone(batch):
    """kappa = omega / max(v, 2): the same path at the lower speed has omega x s, so kappa is unchanged away from the 2 m/s floor and the 0.3 cap (yawrate_std 0 = lateral off)."""
    ego = D.EgoPerturb(p=1.0, vscale=(0.95, 0.95), adrop=0.5, yawrate_std_deg=0.0)
    out = ego.apply(batch, 0)
    F = batch["p10f"].shape[1]
    n_checked = 0
    for b in torch.nonzero(ego.last["sel"]).flatten().tolist():
        t = int(batch["t"][b])
        for f in range(F):
            v = float(batch["p10f"][b, f, 0])
            if (t - (F - 1) + f) >= 1 and bool(batch["frame_ok"][b, f]) and 0.95 * v >= 2.0 and abs(float(batch["p10f"][b, f, 2])) < 0.29:
                assert math.isclose(float(out["p10f"][b, f, 2]), float(batch["p10f"][b, f, 2]), rel_tol=1e-4, abs_tol=1e-6)
                n_checked += 1
    assert n_checked > 20, "the fixture must exercise the invariant"


def test_the_lateral_lever_moves_the_yaw_rate_and_nothing_moves_it_when_off(batch):
    off = D.EgoPerturb(p=1.0, yawrate_std_deg=0.0, seed=1).apply(batch, 4)
    on = D.EgoPerturb(p=1.0, yawrate_std_deg=3.0, seed=1).apply(batch, 4)
    sel = batch["kin3"][:, 0] < 8.0
    moved = (on["p10f"][sel][..., 2] - off["p10f"][sel][..., 2]).abs().max()
    assert float(moved) > 1e-4, "with yawrate_std > 0 kappa moves relative to the lateral-off perturbation of the same draws"
    assert float((on["pose"][sel][..., 2] - off["pose"][sel][..., 2]).abs().max()) > 0.0, "and so does the heading history"
    assert torch.equal(on["kin3"][sel][:, 0], off["kin3"][sel][:, 0]), "the speed draws are independent of the yaw std (a fixed draw order)"


def test_vmax_is_strict_and_a_row_exactly_at_it_is_not_eligible(batch):
    i = int(torch.nonzero(batch["kin3"][:, 0] < 8.0).flatten()[0])
    vmax = float(batch["kin3"][i, 0])
    ego = D.EgoPerturb(p=1.0, vmax=vmax)
    out = ego.apply(batch, 0)
    assert not bool(ego.last["sel"][i]) and torch.equal(out["p10f"][i], batch["p10f"][i]), "logged v0 == vmax is not below it"
    assert int(ego.last["n_selected"]) == int((batch["kin3"][:, 0] < vmax).sum())


def test_a_draw_is_a_function_of_seed_and_step_only_and_never_touches_the_global_rng(batch):
    ego = D.EgoPerturb(p=0.5, seed=2)
    rng0 = torch.get_rng_state()
    a1 = ego.apply(batch, 7)
    last1 = copy_last(ego)
    assert torch.equal(rng0, torch.get_rng_state()), "the global RNG state is untouched"
    a2 = D.EgoPerturb(p=0.5, seed=2).apply(batch, 7)
    assert _same(a1, a2), "same (seed, step): the same perturbation"
    ego.apply(batch, 8)
    assert not bool((ego.last["scale"] == last1["scale"]).all()), "another step draws something else"
    ego3 = D.EgoPerturb(p=0.5, seed=3)
    ego3.apply(batch, 7)
    assert not bool((ego3.last["scale"] == last1["scale"]).all()), "another seed draws something else"


def copy_last(ego):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ego.last.items()}


def test_off_is_the_identity_and_counts_eligibility(batch):
    ego = D.EgoPerturb(p=0.0)
    rng0 = torch.get_rng_state()
    assert ego.apply(batch, 3) is batch and torch.equal(rng0, torch.get_rng_state())
    assert ego.last["n_selected"] == 0 and ego.last["n_eligible"] == int((batch["kin3"][:, 0] < 8.0).sum()) and ego.last["n_rows"] == 32
    assert D.EgoPerturb(p=1.0).apply(batch, 3, p=0.0) is batch, "the probe's p override of 0 is the identity too"


# ------------------------------------------------------------------------------------------------ nav: the route progress carries the travelled distance
def _nav_fixture(batch):
    """The fixture batch with HAND-MADE nav on four selected rows (all ten history frames real): A one token throughout; B the token changes at frame 5 (a manoeuvre boundary); C invalid for frames 0-2;
    D a progress value above 3 (the clip). Returns (batch, {name: row})."""
    sel = torch.nonzero((batch["kin3"][:, 0] < 8.0) & (batch["t"] >= 10)).flatten().tolist()
    assert len(sel) >= 4, "the fixture must hold four eligible windows with a full real history"
    rows = dict(zip("ABCD", sel[:4]))
    nav = batch["nav"].clone()

    def put(r, tok, valid, d3):
        for f in range(10):
            nav[r, f] = 0.0
            nav[r, f, tok[f]] = 1.0
            nav[r, f, 3], nav[r, f, 4] = d3[f], valid[f]
    put(rows["A"], [1] * 10, [1] * 10, [0.40 + 0.03 * (9 - f) for f in range(10)])
    put(rows["B"], [2] * 5 + [1] * 5, [1] * 10, [1.00 + 0.05 * (9 - f) for f in range(10)])
    put(rows["C"], [1] * 10, [0] * 3 + [1] * 7, [0.0] * 3 + [0.50 + 0.02 * (9 - f) for f in range(3, 10)])
    put(rows["D"], [1] * 10, [1] * 10, [3.5] * 9 + [2.9])
    return {**batch, "nav": nav}, rows


def _expected_nav3(nav, ok, s):
    """Independent recomputation for one row: nav [F, 5], ok [F] real-row mask, s the speed scale -> (new d3 [F], kept count)."""
    tok, valid = nav[:, :3].argmax(1), nav[:, 4]
    out, kept = nav[:, 3].astype(np.float64).copy(), 0
    for f in range(len(out)):
        if not ok[f]:
            continue
        if tok[f] == tok[-1] and valid[f] == valid[-1]:
            out[f] = min(max(float(nav[-1, 3]) + s * (float(nav[f, 3]) - float(nav[-1, 3])), 0.0), 3.0)
        else:
            kept += 1
    return out, kept


def test_nav_progress_is_rebuilt_on_exactly_the_frames_sharing_nows_token_and_validity(batch):
    b, rows = _nav_fixture(batch)
    ego = D.EgoPerturb(p=1.0, seed=2)
    out = ego.apply(b, 5)
    sel, F = ego.last["sel"], 10
    total_kept = 0
    for r in torch.nonzero(sel).flatten().tolist():
        s = float(ego.last["scale"][r])
        ok = [(int(b["t"][r]) - (F - 1) + f) >= 1 and bool(b["frame_ok"][r, f]) for f in range(F)]
        want, kept = _expected_nav3(b["nav"][r].numpy(), ok, s)
        total_kept += kept
        np.testing.assert_allclose(out["nav"][r, :, 3].double().numpy(), want, rtol=0, atol=1e-6, err_msg=f"row {r}")
        assert torch.equal(out["nav"][r, :, :3], b["nav"][r, :, :3]) and torch.equal(out["nav"][r, :, 4], b["nav"][r, :, 4]), "token and validity are never touched"
        assert float(out["nav"][r, -1, 3]) == float(b["nav"][r, -1, 3]), "NOW is the anchor"
    assert ego.last["ego_perturb_nav_kept"] == total_kept
    # the hand-made rows pin the three rules with literals
    sA, sB, sC, sD = (float(ego.last["scale"][rows[k]]) for k in "ABCD")
    A = out["nav"][rows["A"], :, 3].double().numpy()
    np.testing.assert_allclose(A, [0.40 + sA * 0.03 * (9 - f) for f in range(10)], atol=1e-6)                       # one token: the whole history scales by s
    Bn, Bb = out["nav"][rows["B"], :, 3], b["nav"][rows["B"], :, 3]
    assert torch.equal(Bn[:5], Bb[:5]), "frames across the manoeuvre boundary are left as logged"
    np.testing.assert_allclose(Bn[5:].double().numpy(), [1.0 + sB * 0.05 * (9 - f) for f in range(5, 10)], atol=1e-6)
    Cn, Cb = out["nav"][rows["C"], :, 3], b["nav"][rows["C"], :, 3]
    assert torch.equal(Cn[:3], Cb[:3]) and bool((Cn[:3] == 0.0).all()), "invalid frames (validity differs from NOW's) are left as logged"
    np.testing.assert_allclose(Cn[3:].double().numpy(), [0.50 + sC * 0.02 * (9 - f) for f in range(3, 10)], atol=1e-6)
    Dn = out["nav"][rows["D"], :, 3].double().numpy()
    np.testing.assert_allclose(Dn[:9], 3.0, atol=1e-6)                                                              # 2.9 + s x 0.6 >= 3.2 for every s >= 0.5: clipped back into [0, 3]
    assert Dn.max() <= 3.0 and Dn.min() >= 0.0 and sD >= 0.5
    nk = {k: int(_nav_kept_of(ego, b, rows[k])) for k in "BC"}
    assert nk == {"B": 5, "C": 3}, nk
    # nothing else moves in nav: unselected rows are byte-identical
    assert torch.equal(out["nav"][~sel], b["nav"][~sel])


def _nav_kept_of(ego, b, r):
    rr = torch.tensor([r])
    return D.EgoPerturb._nav_masks(b, rr)[1].sum()


def test_s_equal_1_leaves_nav_bit_identical_while_the_kinematic_tensors_still_move(batch):
    b, rows = _nav_fixture(batch)
    b["nav"][rows["D"]] = batch["nav"][rows["D"]]                                  # row D holds an out-of-range 3.5 (the clip test): the real nav never does, and s = 1 is the identity only inside [0, 3]
    ego = D.EgoPerturb(p=1.0, vscale=(1.0, 1.0), adrop=1.0, seed=1)
    out = ego.apply(b, 3)
    assert torch.equal(out["nav"], b["nav"]), "s = 1: the distance behind NOW is unchanged, so nav is bit-identical"
    assert bool(_changed(out["p10f"], b["p10f"])[ego.last["sel"]].all()), "the control: the acceleration drop still moved p10f"


def test_nav_rebuild_on_the_unmodified_fixture_follows_the_same_rule_and_counts_what_it_keeps(batch):
    ego = D.EgoPerturb(p=1.0, seed=7)
    out = ego.apply(batch, 1)
    F, kept, rebuilt_rows = 10, 0, 0
    for r in torch.nonzero(ego.last["sel"]).flatten().tolist():
        ok = [(int(batch["t"][r]) - (F - 1) + f) >= 1 and bool(batch["frame_ok"][r, f]) for f in range(F)]
        want, k = _expected_nav3(batch["nav"][r].numpy(), ok, float(ego.last["scale"][r]))
        kept += k
        np.testing.assert_allclose(out["nav"][r, :, 3].double().numpy(), want, atol=1e-6)
        rebuilt_rows += int(bool(_changed(out["nav"][r:r + 1], batch["nav"][r:r + 1])[0]))
    assert ego.last["ego_perturb_nav_kept"] == kept
    assert rebuilt_rows > 0, "the fixture exercises a changed nav row"


# ------------------------------------------------------------------------------------------------ (c) the loss
def _woken(model, seed=11):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = model.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=g) * 0.05)                          # zero-init residual: wake it so a change in the input can be seen at O's output
    return model


def _grads(model, b, seed=20261026):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in named:
        p.grad = None
    for p in model.prior.parameters():
        p.grad = None
    torch.manual_seed(seed)
    T = model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
    T["total"].backward()
    g = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
    gp = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in model.prior.parameters()])
    return float(T["total"].detach()), g, gp


def _gdiff(ga, gb, pre=""):
    return float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))


@pytest.mark.parametrize("wm", ["off", "b"])
def test_the_perturbed_input_reaches_the_loss_gradient_and_p0_reads_exactly_zero(world, wm):
    J = _J(world)
    hmax = 16 if wm != "off" else 10
    b, _, _ = make_batch(world, n=32, seed=0, hmax=hmax)
    model, cfg = tiny_model(J, wm=wm, hmax=hmax)
    model = _woken(model)
    model.train()
    ego = D.EgoPerturb(p=1.0, seed=1)
    t0, g0, gp0 = _grads(model, b)
    t0b, g0b, _ = _grads(model, b)
    assert t0 == t0b and _gdiff(g0, g0b) == 0.0, "the control: the same loss twice differs by exactly 0.0"
    pert = ego.apply(b, 1)
    t1, g1, gp1 = _grads(model, pert)
    assert t1 != t0 and _gdiff(g1, g0) > 0.0, "the perturbed batch moves the loss and its gradient"
    for grp in ("e_ego", "read_o", "head_o"):
        assert _gdiff(g1, g0, grp) > 0.0, f"the perturbation reaches {grp}"
    assert float((gp1 - gp0).abs().sum()) == 0.0, "the frozen prior's gradient difference is exactly 0.0"
    if wm != "off":
        assert _gdiff(g1, g0, "wm.") == 0.0, "act (the WM's transition record) is not perturbed: the WM auxiliary reads exactly 0.0"
    zero = ego.apply(b, 1, p=0.0)
    assert zero is b
    tz, gz, _ = _grads(model, zero)
    assert tz == t0 and _gdiff(gz, g0) == 0.0, "p = 0: exactly 0.0"
    for name in ("p10f", "base"):                                                # one rebuilt tensor at a time
        tn, _, _ = _grads(model, {**b, name: pert[name]})
        assert tn != t0, f"the model reads the rebuilt {name}"
    for name in ("kin3", "p10_t", "pose"):                                       # rebuilt for consistency, not read by the loss: exactly 0.0
        tn, _, _ = _grads(model, {**b, name: pert[name]})
        assert tn == t0, f"{name} is not read by the loss"


def _probe_parts(world, hmax=10, wm="off"):
    J = _J(world)
    b, _, _ = make_batch(world, n=32, seed=0, hmax=hmax)
    model, cfg = tiny_model(J, wm=wm, hmax=hmax)
    return _woken(model), b, cfg


def test_the_launch_probe_passes_and_records_the_controls(world):
    model, b, cfg = _probe_parts(world)
    res = TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.3, seed=1))
    assert res["pass"] is True and res["n_selected_at_p1"] == res["n_eligible"] > 0
    assert res["control_repeat_grad_diff"] == 0.0 and res["control_p0_grad_diff"] == 0.0 and res["control_p0_loss_diff"] == 0.0 and res["control_p0_same_object"] is True
    assert res["control_prior_grad_diff"] == 0.0 and res["grad_diff_all"] > 0.0 and res["live_perturbed"]["pass"] is True and res["live_perturbed"]["control_prior_grad_sum"] == 0.0
    assert res["loss_diff_replacing_only"]["p10f"] != 0.0 and res["loss_diff_replacing_only"]["base"] != 0.0
    assert TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.0))["pass"] is True, "off: nothing to prove, nothing runs"


def test_the_launch_probe_refuses_a_perturbation_that_changes_nothing(world, monkeypatch):
    """The deliberate-regression arm in-process: an `apply` whose rebuild is dropped (the mutation script deletes that line in the file) is refused by the probe."""
    model, b, cfg = _probe_parts(world)
    monkeypatch.setattr(D.EgoPerturb, "_rebuilt", staticmethod(lambda batch, rows, s, d, n: {}))
    with pytest.raises(AssertionError, match="unchanged .* selected rows: the rebuild did not reach it"):
        TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.3))


def test_the_launch_probe_refuses_a_rebuild_that_leaves_one_tensor_unperturbed(world, monkeypatch):
    model, b, cfg = _probe_parts(world)
    orig = D.EgoPerturb._rebuilt

    def keep_base(batch, rows, s, d, n):
        r = orig(batch, rows, s, d, n)
        r["base"] = batch["base"]
        return r
    monkeypatch.setattr(D.EgoPerturb, "_rebuilt", staticmethod(keep_base))
    with pytest.raises(AssertionError, match="base is unchanged"):
        TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.3))


def test_the_launch_probe_refuses_a_perturbation_that_touches_a_target(world, monkeypatch):
    model, b, cfg = _probe_parts(world)
    orig = D.EgoPerturb._rebuilt

    def touch_gt(batch, rows, s, d, n):
        r = orig(batch, rows, s, d, n)
        r["gt"] = batch["gt"] + 1.0
        return r
    monkeypatch.setattr(D.EgoPerturb, "_rebuilt", staticmethod(touch_gt))
    with pytest.raises(AssertionError, match="not the same objects"):
        TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.3))


def test_the_launch_probe_refuses_a_perturbation_outside_the_declared_law(world, monkeypatch):
    model, b, cfg = _probe_parts(world)
    ego = D.EgoPerturb(p=0.3, vscale=(0.5, 1.0))
    orig = D.EgoPerturb.apply

    def too_slow(self, batch, step, *, p=None):                                  # a draw outside [LO, HI]: the declared law is not the applied one
        self.vscale = (0.1, 0.2)
        try:
            return orig(self, batch, step, p=p)
        finally:
            self.vscale = (0.5, 1.0)
    monkeypatch.setattr(D.EgoPerturb, "apply", too_slow)
    with pytest.raises(AssertionError, match="speed ratio .* outside the declared"):
        TR.ego_perturb_probe(model, b, cfg, ego)


def test_the_launch_probe_refuses_an_input_the_perturbation_does_not_account_for(world, monkeypatch):
    model, b, cfg = _probe_parts(world)
    monkeypatch.setattr(D, "EGO_PERTURB_KEPT", tuple(k for k in D.EGO_PERTURB_KEPT if k != "limit"))
    with pytest.raises(AssertionError, match="neither rebuilt nor kept"):
        TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.3))


def test_the_launch_probe_refuses_a_wm_term_that_sees_the_perturbation(world, monkeypatch):
    """`act` must stay true: a rebuild that also perturbs it moves the WM auxiliary's gradient and the probe's exact-zero control goes RED."""
    model, b, cfg = _probe_parts(world, hmax=16, wm="b")
    orig = D.EgoPerturb._rebuilt

    def also_act(batch, rows, s, d, n):
        r = orig(batch, rows, s, d, n)
        act = batch["act"].clone()
        act[rows, :, 0] = act[rows, :, 0] - 1.0
        r["act"] = act
        return r
    monkeypatch.setattr(D.EgoPerturb, "_rebuilt", staticmethod(also_act))
    with pytest.raises(AssertionError, match="not the same objects"):
        TR.ego_perturb_probe(model, b, cfg, D.EgoPerturb(p=0.3))


# ------------------------------------------------------------------------------------------------ (d) G-DVB
def _gdvb_parts(world, argv, **over):
    a = _args(argv)
    cfg = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(_J(world))
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    ws = cfg.window_start_effective
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None, "window_start": ws, "window_t_min": ws, "ds_window_start": ws}
    return a, cfg, model, opt, info


def test_every_new_flag_has_a_declared_vs_built_entry_and_the_defaults_are_off():
    for k in (*FLAGS, "init_ckpt"):
        assert k in TR.FLAG_CHECKS, k
    a = _args()
    assert (a.ego_perturb_p, a.ego_perturb_vscale, a.ego_perturb_vmax, a.ego_perturb_adrop, a.ego_perturb_yawrate_std, a.ego_perturb_seed, a.init_ckpt) == (0.0, [0.5, 1.0], 8.0, 1.5, 0.0, 0, "")


@pytest.mark.parametrize("flag", list(FLAGS))
def test_g_dvb_passes_what_was_built_and_refuses_a_mismatch_for_each_flag(world, flag):
    argv = ["--ego-perturb-p", "0.3"] + ([] if flag == "ego_perturb_p" else FLAGS[flag])
    a, cfg, model, opt, info = _gdvb_parts(world, argv)
    ego = TR.make_ego_perturb(a)
    assert TR.declared_vs_built(a, cfg, model, opt, info, None, ego) == [], "the control: what argv declared is what was built"
    # the BUILT perturber is the pre-change default for this one flag: the declaration and the build disagree
    kw = dict(p=a.ego_perturb_p, vscale=a.ego_perturb_vscale, vmax=a.ego_perturb_vmax, adrop=a.ego_perturb_adrop, yawrate_std_deg=a.ego_perturb_yawrate_std, seed=a.ego_perturb_seed, kin_mode=a.kin_mode)
    dflt = {"ego_perturb_p": ("p", 0.5), "ego_perturb_vscale": ("vscale", (0.5, 1.0)), "ego_perturb_vmax": ("vmax", 8.0), "ego_perturb_adrop": ("adrop", 1.5),
            "ego_perturb_yawrate_std": ("yawrate_std_deg", 0.0), "ego_perturb_seed": ("seed", 0)}[flag]
    kw[dflt[0]] = dflt[1]
    bad = TR.declared_vs_built(a, cfg, model, opt, info, None, D.EgoPerturb(**kw))
    assert any(f"lever {flag}:" in m and "declared" in m and "built" in m for m in bad), bad


@pytest.mark.parametrize("flag", list(FLAGS))
def test_a_declared_flag_with_no_perturber_built_is_refused(world, flag):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--ego-perturb-p", "0.3"] + ([] if flag == "ego_perturb_p" else FLAGS[flag]))
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("no ego perturber was built" in m and flag in m for m in bad), bad
    a0, cfg0, model0, opt0, info0 = _gdvb_parts(world, [])
    assert TR.declared_vs_built(a0, cfg0, model0, opt0, info0) == [], "the control: all defaults, no perturber, passes (every existing caller)"


@pytest.mark.parametrize("flag", [*FLAGS, "init_ckpt"])
def test_a_flag_without_an_entry_is_refused(world, monkeypatch, flag):
    a, cfg, model, opt, info = _gdvb_parts(world, [])
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != flag})
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("flags without a declared-vs-built entry" in m and flag in m for m in bad), bad


@pytest.mark.parametrize("argv, what", [
    (["--ego-perturb-vmax", "5.0"], "does nothing"),                                          # a lever set while p is 0
    (["--ego-perturb-seed", "3"], "does nothing"),
    (["--ego-perturb-p", "0.3", "--ego-perturb-vscale", "1.0", "1.0", "--ego-perturb-adrop", "0.0"], "moves nothing"),
    (["--ego-perturb-p", "0.3", "--ego-perturb-vscale", "0.5", "1.5"], "ego_perturb_vscale"),   # speeds the car UP
    (["--ego-perturb-p", "0.3", "--ego-perturb-vscale", "0.9", "0.6"], "ego_perturb_vscale"),
    (["--ego-perturb-p", "1.5"], "not in [0, 1]"),
    (["--ego-perturb-p", "0.3", "--ego-perturb-adrop", "-1"], "ego_perturb_adrop"),
    (["--ego-perturb-p", "0.3", "--ego-perturb-vmax", "0"], "ego_perturb_vmax"),
    (["--ego-perturb-p", "0.3", "--ego-perturb-yawrate-std", "-0.5"], "ego_perturb_yawrate_std"),
    (["--ego-perturb-p", "0.3", "--kin-mode", "steer"], "needs --kin-mode yaw"),
    (["--ego-perturb-p", "nan"], "ego_perturb_p"),
])
def test_g_dvb_refuses_an_illegal_or_empty_declaration(world, argv, what):
    a, cfg, model, opt, info = _gdvb_parts(world, argv)
    bad = TR.declared_vs_built(a, cfg, model, opt, info, None, TR.make_ego_perturb(a))
    assert any(what in m for m in bad), bad


def test_g_dvb_refuses_init_ckpt_declared_but_not_loaded_and_loaded_but_not_declared(world, tmp_path):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--init-ckpt", str(tmp_path / "ck.pt")])
    assert any("declared but no checkpoint was loaded" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    sha = "ab" * 32
    got = {"path": os.path.abspath(tmp_path / "ck.pt"), "sha256": sha, "applied": True, "skipped": None, "n_tensors": 10, "n_mismatch_after_load": 0}
    assert TR.declared_vs_built(a, cfg, model, opt, info | {"init_ckpt": got}) == [], "the control: loaded and read back"
    assert any("tensors differ" in m for m in TR.declared_vs_built(a, cfg, model, opt, info | {"init_ckpt": got | {"n_mismatch_after_load": 2}})), "the load line dropped: the read-back differs"
    assert any("declared" in m and "built" in m for m in TR.declared_vs_built(a, cfg, model, opt, info | {"init_ckpt": got | {"path": "elsewhere.pt"}})), "argv path != built path"
    assert any("sha256 is not recorded" in m for m in TR.declared_vs_built(a, cfg, model, opt, info | {"init_ckpt": got | {"sha256": None}}))
    resumed = got | {"applied": False, "skipped": "resume", "n_mismatch_after_load": None}
    assert TR.declared_vs_built(a, cfg, model, opt, info | {"init_ckpt": resumed}) == [], "resume wins: the load is skipped, the record says so"
    assert any("neither applied nor skipped" in m for m in TR.declared_vs_built(a, cfg, model, opt, info | {"init_ckpt": resumed | {"skipped": None}}))
    a0, cfg0, model0, opt0, info0 = _gdvb_parts(world, [])
    assert any("not declared but a checkpoint was loaded" in m for m in TR.declared_vs_built(a0, cfg0, model0, opt0, info0 | {"init_ckpt": got}))


# ------------------------------------------------------------------------------------------------ --warmup read back from the built schedule
def _old_lr_at(a_lr, a_warmup, total, s):
    """The loop's closure as it stood at the tip, copied here as the literal the new class must reproduce bit for bit."""
    if s < a_warmup:
        return a_lr * (s + 1) / a_warmup
    return a_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - a_warmup) / max(total - a_warmup, 1), 1.0))))


@pytest.mark.parametrize("lr, warmup, total", [(3e-4, 200, 19164), (1e-4, 100, 3000), (3e-4, 2, 4), (3e-4, 0, 10), (1e-4, 5, 3)])
def test_the_schedule_class_reproduces_the_tip_closure_bit_for_bit(lr, warmup, total):
    sch = TR.LRSchedule(lr, warmup, total)
    for step in list(range(0, min(total, 40))) + [total // 2, total - 1, total, total + 5]:
        assert sch(step) == _old_lr_at(lr, warmup, total, step), step


def test_g_dvb_reads_warmup_back_from_the_built_schedule_and_refuses_a_lie(world):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--warmup", "100", "--lr", "1e-4"])
    good = TR.LRSchedule(a.lr, 100, 3000)
    assert TR.declared_vs_built(a, cfg, model, opt, info, None, None, good) == [], "the control: what argv declared is what was built"
    bad = TR.declared_vs_built(a, cfg, model, opt, info, None, None, TR.LRSchedule(a.lr, 50, 3000))
    assert any("lever warmup: declared 100 but the built schedule's warmup is 50" in m for m in bad), bad
    bad = TR.declared_vs_built(a, cfg, model, opt, info, None, None, TR.LRSchedule(3e-4, 100, 3000))
    assert any("lever lr: declared" in m for m in bad), bad

    class Liar(TR.LRSchedule):                                                    # carries the declared numbers but does not DO them: constant lr, no warmup
        def __call__(self, s):
            return self.lr
    bad = TR.declared_vs_built(a, cfg, model, opt, info, None, None, Liar(a.lr, 100, 3000))
    assert any("the built schedule reads" in m for m in bad), bad
    bad = TR.declared_vs_built(a, cfg, model, opt, info)
    assert any("no learning-rate schedule was built" in m and "100" in m for m in bad), bad
    a0, cfg0, model0, opt0, info0 = _gdvb_parts(world, [])
    assert TR.declared_vs_built(a0, cfg0, model0, opt0, info0) == [], "the control: the default warmup with no schedule passed (every existing caller)"
    assert TR.declared_vs_built(a0, cfg0, model0, opt0, info0, None, None, TR.LRSchedule(a0.lr, 200, 10)) == []


def test_a_real_run_builds_the_declared_warmup_and_a_dropped_one_is_refused(tmp_path, monkeypatch):
    out = tmp_path / "ok"
    assert TR.main(_run(out, ["--steps", "3", "--warmup", "3", "--lr", "1e-3"])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["levers"]["lr_schedule"] == {"lr": 1e-3, "warmup": 3, "total": 3}
    assert [r["lr"] for r in _log(out)] == [1e-3 * 1 / 3, 1e-3 * 2 / 3, 1e-3 * 3 / 3], "the loop ran the declared warmup (lr logged at step - 1)"
    orig = TR.LRSchedule

    class Dropped(orig):                                                          # the schedule that was built ignores --warmup
        def __init__(self, lr, warmup, total):
            super().__init__(lr, 1, total)
    monkeypatch.setattr(TR, "LRSchedule", Dropped)
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ declared-vs-built: .*lever warmup: declared 3 but the built schedule's warmup is 1"):
        TR.main(_run(tmp_path / "bad", ["--steps", "3", "--warmup", "3", "--lr", "1e-3"]))
    for f in ("selfcheck.json", "summary.json", "ckpt_last.pt", "train_log.jsonl"):
        assert not (tmp_path / "bad" / f).exists(), f"a refused launch writes no {f}"


# ------------------------------------------------------------------------------------------------ (e) the real caller
def _run(out, extra=(), common=COMMON):
    out = Path(out)
    argv = ["--out", str(out), *common, *extra]
    return argv


def _summary(out):
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def _log(out):
    return [json.loads(x) for x in (Path(out) / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def spy(monkeypatch):
    """Record the (p10f, base, gt, lab_t has) every call of the model's `loss` receives."""
    calls = []
    orig = M.V8FMini.loss

    def loss(self, batch, p_gt=None, gen=None):
        calls.append({k: batch[k].clone() for k in ("p10f", "base", "kin3", "pose_t", "gt", "s_h", "limit", "nav", "t", "frame_ok")} | {"has": batch["lab_t"]["has"].clone(), "feats": batch["feats"].clone()})
        return orig(self, batch, p_gt, gen)
    monkeypatch.setattr(M.V8FMini, "loss", loss)
    return calls


def test_a_real_run_applies_the_perturbation_to_the_batch_the_loss_receives_and_the_counts_are_the_controls(tmp_path, spy):
    steps = 4
    assert TR.main(_run(tmp_path / "p0", ["--steps", str(steps)])) == 0
    c0 = spy[-steps:]
    spy.clear()
    assert TR.main(_run(tmp_path / "p1", ["--steps", str(steps), "--ego-perturb-p", "1.0", "--ego-perturb-seed", "4"])) == 0
    c1 = spy[-steps:]
    assert len(c0) == len(c1) == steps
    s0, s1 = _summary(tmp_path / "p0"), _summary(tmp_path / "p1")
    # the coordinator's controls: p = 0 -> exactly 0 perturbed; p = 1 -> perturbed == eligible; eligibility is a function of the logged state, so it is the same in both runs
    assert s0["ego_perturb_n"] == 0 and s0["ego_perturb_eligible"] > 0 and s0["ego_perturb_rows"] == 8 * steps and s0["ego_perturb_nav_kept"] == 0
    assert s1["ego_perturb_n"] == s1["ego_perturb_eligible"] > 0 and s1["ego_perturb_rows"] == 8 * steps
    assert s0["ego_perturb_eligible"] == s1["ego_perturb_eligible"]
    l0, l1 = _log(tmp_path / "p0"), _log(tmp_path / "p1")
    assert [r["ego_perturb_n"] for r in l0] == [0] * steps
    assert sum(r["ego_perturb_n"] for r in l1) == s1["ego_perturb_n"] and sum(r["ego_perturb_eligible"] for r in l1) == s1["ego_perturb_eligible"]
    assert all(r["ego_perturb_n"] == r["ego_perturb_eligible"] for r in l1) and all(r["ego_perturb_rows"] == 8 for r in l1)
    assert sum(r["ego_perturb_nav_kept"] for r in l1) == s1["ego_perturb_nav_kept"] >= 0 and all(r["ego_perturb_nav_kept"] == 0 for r in l0)
    # reachability from the caller: the batch the loss received differs in the rebuilt tensors, and ONLY in them
    n_changed = n_nav_changed = 0
    for x0, x1 in zip(c0, c1):
        for k in ("gt", "s_h", "limit", "has", "feats"):
            assert torch.equal(x0[k], x1[k]), f"{k}: the same window stream, targets and scene untouched"
        for k in ("p10f", "base", "kin3"):
            n_changed += int(_changed(x1[k], x0[k]).sum())
        n_nav_changed += int(_changed(x1["nav"], x0["nav"]).sum())
        assert torch.equal(x1["nav"][..., :3], x0["nav"][..., :3]) and torch.equal(x1["nav"][..., 4], x0["nav"][..., 4])
    assert n_changed > 0, "the loss received perturbed ego tensors"
    expected_kept = 0                                                            # nav frames left as logged, recomputed from the batches the loss received at p = 0 (p = 1 selects exactly the eligible rows)
    for x0 in c0:
        eligible = D.EgoPerturb(p=1.0).eligible(x0)
        rows = torch.nonzero(eligible).flatten()
        expected_kept += int(D.EgoPerturb._nav_masks(x0, rows)[1].sum()) if rows.numel() else 0
    assert s1["ego_perturb_nav_kept"] == expected_kept > 0, "the run's nav_kept total is what the batches it saw imply (and the fixture has manoeuvre boundaries inside a history)"
    assert n_nav_changed > 0, "nav[:, 3] reached the loss changed"
    sc = json.loads((tmp_path / "p1" / "selfcheck.json").read_text(encoding="utf-8"))
    pr = sc["ego_perturb_probe"]                                                 # the record carries the EVIDENCE of the probe the trainer ran, not just a verdict
    assert pr["pass"] is True and pr["n_selected_at_p1"] > 0 and pr["grad_diff_all"] > 0.0 and pr["control_p0_grad_diff"] == 0.0 and pr["control_prior_grad_diff"] == 0.0
    assert pr["live_perturbed"]["pass"] is True and pr["live_perturbed"]["control_prior_grad_sum"] == 0.0 and pr["loss_diff_replacing_only"]["p10f"] != 0.0
    assert sc["levers"]["ego_perturb"]["ego_perturb_p"] == 1.0 and sc["levers"]["ego_perturb"]["ego_perturb_seed"] == 4
    assert s1["levers"]["ego_perturb"] == sc["levers"]["ego_perturb"]
    ck = torch.load(tmp_path / "p1" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["ego_perturb"]["ego_perturb_p"] == 1.0 and ck["ego_totals"]["n"] == s1["ego_perturb_n"] and ck["ego_totals"]["nav_kept"] == s1["ego_perturb_nav_kept"]
    assert json.loads((tmp_path / "p0" / "selfcheck.json").read_text(encoding="utf-8"))["ego_perturb_probe"]["pass"] is True


def test_the_check_that_the_loss_received_perturbed_tensors_can_see_a_missing_apply(tmp_path, spy, monkeypatch):
    """The deliberate-regression arm of (e) in-process: if the loop's `ego.apply` returned the batch unchanged, the predicate above reads zero changed tensors."""
    steps = 3
    assert TR.main(_run(tmp_path / "ref", ["--steps", str(steps)])) == 0
    c0 = spy[-steps:]
    spy.clear()
    orig = D.EgoPerturb.apply

    def pass_through(self, batch, step, *, p=None):
        out = orig(self, batch, step, p=p)
        return batch if step >= 0 else out                                       # the training loop's steps are >= 0; the launch probe (step -1) still sees the real apply
    monkeypatch.setattr(D.EgoPerturb, "apply", pass_through)
    assert TR.main(_run(tmp_path / "mut", ["--steps", str(steps), "--ego-perturb-p", "1.0"])) == 0
    c1 = spy[-steps:]
    assert sum(int(_changed(x1[k], x0[k]).sum()) for x0, x1 in zip(c0, c1) for k in ("p10f", "base", "kin3")) == 0
    s = _summary(tmp_path / "mut")
    assert s["ego_perturb_n"] == s["ego_perturb_eligible"] > 0, "the counts come from apply's own report: only the batch-level predicate can see a deleted application"


def test_the_log_interval_counts_add_up_to_the_totals(tmp_path):
    common = list(COMMON)
    common[common.index("--log-every") + 1] = "3"
    assert TR.main(_run(tmp_path / "r", ["--steps", "7", "--ego-perturb-p", "0.5"], common)) == 0
    s, log = _summary(tmp_path / "r"), _log(tmp_path / "r")
    assert [r["step"] for r in log] == [3, 6, 7]
    assert sum(r["ego_perturb_n"] for r in log) == s["ego_perturb_n"] and sum(r["ego_perturb_eligible"] for r in log) == s["ego_perturb_eligible"] and sum(r["ego_perturb_rows"] for r in log) == 56
    assert 0 < s["ego_perturb_n"] <= s["ego_perturb_eligible"]


def test_a_resumed_run_draws_what_an_uninterrupted_run_drew_and_keeps_counting(tmp_path, monkeypatch):
    common = [*COMMON, "--steps", "4", "--ckpt-every", "1", "--ego-perturb-p", "0.6", "--ego-perturb-seed", "9"]
    assert TR.main(["--out", str(tmp_path / "full"), *common]) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "2")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(tmp_path / "res"), *common])
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(["--out", str(tmp_path / "res"), *common]) == 0
    a = torch.load(tmp_path / "full" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "res" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "the final weights are those of the uninterrupted run"
    assert a["ego_totals"] == b["ego_totals"] and _summary(tmp_path / "full")["ego_perturb_n"] == _summary(tmp_path / "res")["ego_perturb_n"] > 0


@pytest.mark.parametrize("extra, msg", [
    (["--ego-perturb-vmax", "5.0"], "does nothing"),
    (["--ego-perturb-p", "0.3", "--ego-perturb-vscale", "1.0", "1.0", "--ego-perturb-adrop", "0.0"], "moves nothing"),
    (["--ego-perturb-p", "0.3", "--kin-mode", "steer"], "needs --kin-mode yaw"),
])
def test_a_refused_declaration_stops_the_real_trainer_before_it_writes_anything(tmp_path, extra, msg):
    common = [x for x in COMMON]
    if "--kin-mode" in extra:
        common = [x for i, x in enumerate(COMMON) if COMMON[max(i - 1, 0)] != "--kin-mode" and x != "--kin-mode"]
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ declared-vs-built: .*" + msg):
        TR.main(_run(tmp_path / "bad", ["--steps", "2", *extra], common))
    for f in ("selfcheck.json", "argv.json", "ckpt_last.pt", "summary.json", "train_log.jsonl"):
        assert not (tmp_path / "bad" / f).exists(), f"a refused launch writes no {f}"


# ------------------------------------------------------------------------------------------------ --init-ckpt (f)-(h)
@pytest.fixture
def adam_spy(monkeypatch):
    """Record, at every AdamW.step: the optimiser's state size and a copy of every parameter it holds (= the parameters the forward + backward just used)."""
    calls = []
    orig = torch.optim.AdamW.step

    def step(self, *a, **k):
        calls.append({"n_state": len(self.state), "params": [p.detach().clone() for g in self.param_groups for p in g["params"]]})
        return orig(self, *a, **k)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    return calls


def _digests(tensors):
    return {(tuple(t.shape), t.contiguous().numpy().tobytes()) for t in tensors}


def test_init_ckpt_starts_the_run_at_the_checkpoints_weights_with_an_empty_optimiser_at_step_0(tmp_path, adam_spy):
    """(f)"""
    import hashlib
    assert TR.main(_run(tmp_path / "src", ["--steps", "3", "--ckpt-every", "1"])) == 0
    src = torch.load(tmp_path / "src" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    want = _digests(src["model"].values())
    adam_spy.clear()
    common = [x for x in COMMON]
    common[common.index("--seed") + 1] = "99"                                       # another seed: a from-scratch init differs from the checkpoint's
    ck = str(tmp_path / "src" / "ckpt_last.pt")
    argv = ["--out", str(tmp_path / "ft"), *common, "--steps", "2", "--lr", "1e-4", "--warmup", "1", "--init-ckpt", ck, "--ego-perturb-p", "1.0"]
    assert TR.main(argv) == 0
    first = adam_spy[-2]                                                            # the first optimiser step of the loop (the launch probes' steps precede it, on copies)
    assert first["n_state"] == 0, "a FRESH optimiser: no state before the first step"
    assert len(first["params"]) > 50 and _digests(first["params"]) <= want, "every parameter the first forward used IS the checkpoint's, bit for bit"
    log = _log(tmp_path / "ft")
    assert [r["step"] for r in log] == [1, 2] and _summary(tmp_path / "ft")["steps"] == 2, "step 0 start: the source's 3 steps are not carried over"
    rec = json.loads((tmp_path / "ft" / "selfcheck.json").read_text(encoding="utf-8"))["info"]["init_ckpt"]
    assert rec["sha256"] == hashlib.sha256(Path(ck).read_bytes()).hexdigest() and rec["path"] == os.path.abspath(ck) and rec["applied"] is True
    assert rec["n_mismatch_after_load"] == 0 and rec["n_tensors"] == len(src["model"]) and rec["source_step"] == 3
    fin = torch.load(tmp_path / "ft" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert _summary(tmp_path / "ft")["levers"]["init_ckpt"] == rec and fin["init_ckpt"] == rec and fin["step"] == 2
    # the control: without --init-ckpt the same argv does NOT start at the checkpoint's weights
    adam_spy.clear()
    assert TR.main(["--out", str(tmp_path / "scratch"), *common, "--steps", "2", "--lr", "1e-4", "--warmup", "1"]) == 0
    assert not (_digests(adam_spy[-2]["params"]) <= want)
    assert json.loads((tmp_path / "scratch" / "selfcheck.json").read_text(encoding="utf-8"))["info"].get("init_ckpt") is None


def test_init_ckpt_is_refused_on_a_config_mismatch_that_a_strict_load_alone_would_accept(tmp_path):
    """(g)"""
    assert TR.main(_run(tmp_path / "src", ["--steps", "1", "--ckpt-every", "1"])) == 0
    ck = str(tmp_path / "src" / "ckpt_last.pt")
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    assert saved["cfg"]["hmax"] == 16
    # hmax 18 builds a model with IDENTICAL parameter shapes: the strict load alone passes it ...
    a = TR.build_parser().parse_args(["--out", "x", *COMMON[1:], "--hmax", "18"])
    cfg = TR.make_config(a)
    assert cfg.hmax == 18
    m = M.V8FMini(cfg, M.P10Prior.synthetic(a.seed), saved["cons_mu"], saved["cons_sd"])
    m.load_state_dict(saved["model"], strict=True)                                   # the control: the shapes match, so the strict load alone cannot tell (it raises on a mismatch)
    # ... and the config clause refuses it, naming the field
    with pytest.raises(SystemExit, match=r"ZZDVB-REFUSEDZZ init-ckpt .*config mismatch.*hmax: 16 != 18"):
        TR.main(_run(tmp_path / "bad_hmax", ["--steps", "1", "--hmax", "18", "--init-ckpt", ck]))
    kin = [x for x in COMMON]
    kin[kin.index("--kin-mode") + 1] = "steer"
    with pytest.raises(SystemExit, match=r"ZZDVB-REFUSEDZZ init-ckpt .*config mismatch.*kin_mode: 'yaw' != 'steer'"):
        TR.main(["--out", str(tmp_path / "bad_kin"), *kin, "--steps", "1", "--init-ckpt", ck])
    wide = [x for x in COMMON]
    wide[wide.index("--d") + 1] = "64"
    with pytest.raises(SystemExit, match=r"ZZDVB-REFUSEDZZ init-ckpt .*config mismatch.*d: 32 != 64"):
        TR.main(["--out", str(tmp_path / "bad_d"), *wide, "--steps", "1", "--init-ckpt", ck])
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ init-ckpt"):
        TR.main(_run(tmp_path / "bad_missing", ["--steps", "1", "--init-ckpt", str(tmp_path / "nope.pt")]))
    for d in ("bad_hmax", "bad_kin", "bad_d", "bad_missing"):
        for f in ("selfcheck.json", "summary.json", "ckpt_last.pt", "train_log.jsonl"):
            assert not (tmp_path / d / f).exists(), f"a refused launch writes no {f}"
    # training levers may differ: a fine-tune moves them on purpose
    assert TR.main(_run(tmp_path / "ok", ["--steps", "1", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--w-acc", "0.5", "--init-ckpt", ck])) == 0


def test_an_existing_ckpt_last_in_out_resumes_and_wins_over_init_ckpt(tmp_path):
    assert TR.main(_run(tmp_path / "src", ["--steps", "2", "--ckpt-every", "1"])) == 0
    ck = str(tmp_path / "src" / "ckpt_last.pt")
    argv = _run(tmp_path / "ft", ["--steps", "3", "--ckpt-every", "1", "--init-ckpt", ck])
    assert TR.main(argv) == 0
    fin = torch.load(tmp_path / "ft" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert TR.main(argv) == 0, "the same argv again: ckpt_last.pt (step 3) is in --out"
    rec = json.loads((tmp_path / "ft" / "selfcheck.json").read_text(encoding="utf-8"))["info"]["init_ckpt"]
    assert rec["applied"] is False and rec["skipped"] == "resume" and len(rec["sha256"]) == 64, "resume wins: hashed and config-checked, not loaded"
    again = torch.load(tmp_path / "ft" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert again["step"] == 3 and all(torch.equal(again["model"][k], fin["model"][k]) for k in fin["model"]), "the resumed weights are the run's own, not the init checkpoint's"
    with pytest.raises(SystemExit, match="ZZRESUME-REFUSEDZZ"):                    # a resume under another argv is still refused
        TR.main(_run(tmp_path / "ft", ["--steps", "3", "--ckpt-every", "1"]))


PLANNED = ["--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2"]


@pytest.mark.parametrize("p", ["0.3", "0.0"])
def test_the_planned_finetune_argv_is_accepted_and_every_lever_in_it_is_read_back(tmp_path, p):
    """M2b's argv + `--init-ckpt <ckpt> --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p {0.3|0}`: --steps and the p-gt flags are real levers of the trainer's own gate
    (G-DVB), not smoke-only: they are compared with / recorded from what was BUILT, and a lie is refused (next test)."""
    assert TR.main(_run(tmp_path / "src", ["--steps", "1", "--ckpt-every", "1"])) == 0
    ck = str(tmp_path / "src" / "ckpt_last.pt")
    out = tmp_path / "gate"
    assert TR.main(_run(out, [*PLANNED, "--init-ckpt", ck, "--ego-perturb-p", p, "--selfcheck-only"])) == 0
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["declared_vs_built"] == "PASS" and sc["steps_total"] == 3000, "--steps is THE step budget (selfcheck.json steps_total), not a smoke cap"
    lv = sc["levers"]
    assert (lv["p_gt_start"], lv["p_gt_end"]) == (0.2, 0.2) and lv["ego_perturb"]["ego_perturb_p"] == float(p) and lv["init_ckpt"]["applied"] is True
    assert sc["ego_perturb_probe"]["pass"] is True


def test_the_p_gt_flags_are_checked_against_the_built_model_and_a_lie_is_refused(world):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--p-gt-start", "0.2", "--p-gt-end", "0.2"])
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    assert (model.built_levers()["p_gt_start"], model.built_levers()["p_gt_end"]) == (0.2, 0.2) and TR.FLAG_CHECKS["p_gt_start"] == "cfg:p_gt_start" and TR.FLAG_CHECKS["steps"] == "run"
    a2, cfg2, model2, opt2, info2 = _gdvb_parts(world, ["--p-gt-start", "0.2", "--p-gt-end", "0.2"], p_gt_start=1.0)          # built the default schedule under a 0.2 declaration
    bad = TR.declared_vs_built(a2, cfg2, model2, opt2, info2)
    assert any("p_gt_start: declared 0.2 but built 1.0" in m for m in bad), bad


# ------------------------------------------------------------------------------------------------ (a) the default run IS the pre-change run
CHILD = r'''
import sys, json, importlib.machinery, importlib.util
repo, mode, out, extra, ref_data, ref_train = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), sys.argv[5], sys.argv[6]
for p in (repo + "/stack", repo + "/taniteval", repo + "/stack/scripts"):
    sys.path.insert(0, p)
import torch
torch.set_num_threads(1)
if mode == "ref":
    import tanitad.data
    def load(name, path):
        loader = importlib.machinery.SourceFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, loader)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        loader.exec_module(m)
        return m
    tanitad.data.v8f_mini_data = load("tanitad.data.v8f_mini_data", ref_data)
    TR = load("v8f_mini_train", ref_train)
else:
    import v8f_mini_train as TR
sys.exit(TR.main(["--out", out] + extra))
'''


def _child(mode, out, extra):
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, str(REPO), mode, str(out), json.dumps(extra), str(REF_DATA), str(REF_TRAIN)], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, (mode, r.stdout[-1500:], r.stderr[-2500:])


def _numbers(out):
    """The loss trace: every logged number except the wall clock and the new control counts."""
    return [{k: v for k, v in rec.items() if k != "sec" and not k.startswith("ego_perturb")} for rec in _log(out)]


@needs_ref
def test_defaults_reproduce_the_pre_change_trainer_bit_for_bit(tmp_path):
    """The control that must read exactly equal: the saved pre-change trainer + data module vs this tree's, same argv, 3 steps: every logged number and every final weight."""
    extra = ["--steps", "3", *[x for x in COMMON if x != "--synthetic"], "--synthetic"]
    _child("ref", tmp_path / "ref", extra)
    _child("new", tmp_path / "new", extra)
    n_ref, n_new = _numbers(tmp_path / "ref"), _numbers(tmp_path / "new")
    assert len(n_ref) == 3 and n_ref == n_new, "the loss trace is bit-identical (json round-trips floats exactly)"
    a = torch.load(tmp_path / "ref" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "new" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "the final weights are bit-identical"
    assert _summary(tmp_path / "new")["ego_perturb_n"] == 0
    # the positive control: the comparison CAN see a difference -- the same argv with the lever on
    _child("new", tmp_path / "on", [*extra, "--ego-perturb-p", "1.0"])
    assert _numbers(tmp_path / "on") != n_ref, "p = 1 changes the loss trace: the identity above is not vacuous"


@needs_ref
def test_the_saved_references_are_the_tip_files_with_nothing_of_the_lever_in_them():
    for p in (REF_TRAIN, REF_DATA):
        s = p.read_text(encoding="utf-8")
        assert "ego_perturb" not in s and "EgoPerturb" not in s and "init_ckpt" not in s, p.name


# ------------------------------------------------------------------------------------------------ the mutation anchors
def test_the_single_applying_lines_exist_exactly_once():
    """`raw/2026-10-26-orecover/mutate_orecover.py` deletes these; an anchor that is not found exactly once would make a mutation test vacuous."""
    assert DATA_P.read_text(encoding="utf-8").count("            out.update(self._rebuilt(batch, rows, s[r].to(dev), d[r].to(dev), n[r].to(dev)))") == 1
    assert TRAIN_P.read_text(encoding="utf-8").count("            b = ego.apply(b, step) ") == 1
