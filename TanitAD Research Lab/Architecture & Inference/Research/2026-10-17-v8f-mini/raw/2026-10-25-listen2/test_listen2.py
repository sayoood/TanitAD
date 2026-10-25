"""Tests for listen2.py / read2.py (PREREG_V8F_LISTEN2). CPU only, synthetic world, random-init tiny model made SENSITIVE (every input moves every layer).

What is pinned, and the control that must go red for each:
  E1  (a) the donor nav reaches S's reader and NOTHING else of T's and O's own inputs: T's and O's ego / nav / limit tokens, O's age token and both scene
          paths are bit-identical to the clean chain's (checked here from the readers' captured inputs, independently of the module's own assertions);
          T's and O's S-checkpoint tokens and O's code token DO move (the change flows down); S's checkpoints, T's logits and O's trajectory move;
      (b) the arm equals LISTEN's Q-NAV 'nav@S' arm; donor = self reproduces the clean chain exactly;
      (c) MUTATION: injection at T, at S+T, or at S+O instead of S only => the module's per-window assertion goes RED.
  E3  the joint triple hands O the SAME donor's ego token, S checkpoints and T code and nothing else (scene, nav, limit, age and the kinematic PRIOR stay the
      window's own); it equals a hand-built shadow of the same three; donor = self is the clean pass; MUTATION: code dropped / prior swapped => RED.
  E2  the MEANS equal an independent per-window computation (including the padded-frame masking and the per-layer tick); every mean cell hands the reader
      exactly the mean token (and the unpadded mean scene) and nothing else; the window's OWN value through the same path is the clean pass bit for bit
      (a mean cell then reads exactly 0.0); MUTATION: an override that never reaches the layer => RED.
  plus: the three outcome tables of ``read2`` are applied verbatim to synthetic inputs, and the whole pipeline runs on the synthetic eval split.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("LISTEN_CODE", str(HERE / "code"))
sys.path.insert(0, str(HERE))
from test_listen import IDXS, _moved, _tree_equal, eval_world, sensitive_model, world  # noqa: E402,F401  (fixtures + helpers of LISTEN's test module)


def _load2(path, name="listen2_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


L2 = _load2(os.environ.get("LISTEN2_PY", str(HERE / "listen2.py")))
LM = L2.LS                                                                       # LISTEN's module, the instance listen2 uses
from tanitad.data import v8f_mini_data as D  # noqa: E402,F401

torch.set_num_threads(1)
_eq = torch.equal
capture = L2.capture


@pytest.fixture(scope="module")
def s2(world):
    model = sensitive_model(world["J"])
    perm = LM.donor_perm(len(world["wins"]), world["eid"], 0)
    P, Dn, ages_d = LM.collate_window_batch(model, world["ds"], world["wins"], perm, IDXS)
    return {"model": model, "perm": perm, "P": P, "Dn": Dn, "ages_d": ages_d, **world}


# ------------------------------------------------------------------------------------------------ E1
def check_e1_scope(L2mod, model, P, Dn, ages_d):
    """The independent scope check: what each Reader is handed when the donor's nav is injected at S only."""
    LMod = L2mod.LS
    with capture(model) as clean:
        LMod.run_chain(model, P)
    with capture(model) as donor:
        LMod.run_chain(model, Dn)
    r, n = L2mod.e1_arm(model, P, Dn, ages_d)                                      # raises when the per-window assertions fail
    assert n == len(IDXS)
    with capture(model) as cp:
        r2 = LMod.run_chain(model, P, Dn, {"S": ("nav",)}, ages=ages_d)
    assert _eq(r["traj60"], r2["traj60"])
    pos = LMod.TOKEN_POS
    (cs, ss, ps), (c0s, s0s, p0s), (cds, _, _) = cp["read_s"][0], clean["read_s"][0], donor["read_s"][0]
    assert _eq(cs[:, pos["S"]["nav"]], cds[:, pos["S"]["nav"]]), "S's nav token is not the donor's"
    assert _eq(cs[:, pos["S"]["ego"]], c0s[:, pos["S"]["ego"]]) and _eq(cs[:, pos["S"]["limit"]], c0s[:, pos["S"]["limit"]])
    assert _eq(ss, s0s) and _eq(ps, p0s)
    for L, flows in (("T", ("ck",)), ("O", ("ck", "code"))):
        rd = LMod._READER[L]
        c, sc, pd = cp[rd][0]
        c0, s0, p0 = clean[rd][0]
        assert _eq(sc, s0) and _eq(pd, p0), f"{L}: the scene path moved"
        for nm in ("ego", "nav", "limit") + (("age",) if L == "O" else ()):
            assert _eq(c[:, pos[L][nm]], c0[:, pos[L][nm]]), f"{L}'s own {nm} token is not the clean pass's"
        for nm in flows:
            assert not _eq(c[:, pos[L][nm]], c0[:, pos[L][nm]]), f"{L}.{nm}: the change did not flow down (a dead probe)"
    assert _moved(r["ckpt"], P.out["ckpt"]) and _moved(r["tout"]["lat_logits"], P.out["lat_logits"]) and _moved(r["tout"]["lon_logits"], P.out["lon_logits"])
    assert _moved(r["traj60"], P.out["traj60"])
    return True


def test_e1_injects_the_donor_nav_at_S_only_and_the_change_flows_down(s2):
    check_e1_scope(L2, s2["model"], s2["P"], s2["Dn"], s2["ages_d"])


def test_e1_arm_equals_listens_qnav_nav_at_S_arm_and_the_S_cell(s2):
    model, P, Dn, ages_d = s2["model"], s2["P"], s2["Dn"], s2["ages_d"]
    r, _ = L2.e1_arm(model, P, Dn, ages_d)
    q = LM.run_chain(model, P, Dn, LM.QNAV_ARMS["nav@S"], ages=ages_d)
    assert _eq(r["traj60"], q["traj60"]) and _eq(r["ckpt"], q["ckpt"])
    assert _eq(r["ckpt"], LM.cell(model, P, Dn, "S", "nav"))                         # S's step IS the S x nav cell


def test_e1_identity_donor_is_the_clean_chain_exactly(s2):
    model = s2["model"]
    ident = np.arange(len(s2["wins"]))
    P, Dn, ages_d = LM.collate_window_batch(model, s2["ds"], s2["wins"], ident, IDXS)
    r, _ = L2.e1_arm(model, P, Dn, ages_d)
    assert _eq(r["ckpt"], P.out["ckpt"]) and _eq(r["traj60"], P.out["traj60"])
    for k in ("lat_logits", "lon_logits", "goal_logits", "cons_std"):
        assert _eq(r["tout"][k], P.out[k]), k


@pytest.mark.parametrize("ov", [{"T": ("nav",)}, {"S": ("nav",), "T": ("nav",)}, {"S": ("nav",), "O": ("nav",)}, {"O": ("nav",)}])
def test_mutation_e1_injected_anywhere_but_S_only_goes_red(s2, monkeypatch, ov):
    orig = LM.run_chain

    def wrong(model, P, Dn=None, o=None, ages=None):
        return orig(model, P, Dn, ov, ages=ages)
    monkeypatch.setattr(LM, "run_chain", wrong)
    with pytest.raises(AssertionError):
        L2.e1_arm(s2["model"], s2["P"], s2["Dn"], s2["ages_d"])


def test_mutation_e1_check_catches_a_t_nav_token_that_moved(s2):
    """The assertion itself is able to fail: a pre-hook (registered BEFORE the capture hook) moves the nav token T's reader is handed."""
    model = s2["model"]

    def mut(mod, args):
        cond, scene, pad = args
        cond = cond.clone()
        cond[:, 1] = cond[:, 1] + 1.0
        return (cond, scene, pad)
    h = model.read_t.register_forward_pre_hook(mut)
    try:
        with pytest.raises(AssertionError, match="nav"):
            L2.e1_arm(model, s2["P"], s2["Dn"], s2["ages_d"])
    finally:
        h.remove()


# ------------------------------------------------------------------------------------------------ E3
def check_e3_scope(L2mod, model, P, Dn):
    LMod = L2mod.LS
    with capture(model) as clean:
        LMod.run_chain(model, P)
    with capture(model) as donor:
        LMod.run_chain(model, Dn)
    with capture(model) as cp:
        traj, p30 = L2mod.e3_joint(model, P, Dn)
    assert list(cp) == ["read_o"], f"only O's reader may run, got {list(cp)}"
    c, sc, pd = cp["read_o"][0]
    c0, s0, p0 = clean["read_o"][0]
    cd = donor["read_o"][0][0]
    pos = LMod.TOKEN_POS["O"]
    for nm in ("ego", "ck", "code"):
        assert _eq(c[:, pos[nm]], cd[:, pos[nm]]), f"the joint triple's {nm} is not the donor's"
    for nm in ("nav", "limit", "age"):
        assert _eq(c[:, pos[nm]], c0[:, pos[nm]]), f"the joint triple moved O's own {nm} token"
    assert _eq(sc, s0) and _eq(pd, p0), "the scene path moved"
    assert _eq(p30, P.out["p10"]), "the kinematic prior moved"
    assert _moved(traj, P.out["traj60"])
    return traj


def test_e3_joint_triple_hands_O_one_donors_ego_ck_and_code_only(s2):
    traj = check_e3_scope(L2, s2["model"], s2["P"], s2["Dn"])
    model, P, Dn = s2["model"], s2["P"], s2["Dn"]
    # a hand-built shadow of the same three (not through LISTEN's override machinery)
    tok = model._ego(model._at(Dn.b["p10f"], Dn.last_now))
    object.__setattr__(model, "_ego", lambda *a, **k: tok)
    try:
        with torch.no_grad():                                                        # the harness runs under no_grad (TransformerEncoder's inference fast path differs by ~1e-6 from the grad path)
            hand, _ = model.run_o(P.b, P.adapt, P.last_now, Dn.ck_for_o, P.ck_ok, Dn.code)
    finally:
        model.__dict__.pop("_ego")
    assert _eq(traj, hand)
    for x in L2.E3_SINGLES:                                                          # the joint is a different state from each single swap
        single, _ = LM.cell(model, P, Dn, "O", x)
        assert _moved(traj, single), f"the joint triple equals the single swap {x}"


def test_e3_identity_donor_is_the_clean_pass_exactly(s2):
    model = s2["model"]
    P, Dn, _ = LM.collate_window_batch(model, s2["ds"], s2["wins"], np.arange(len(s2["wins"])), IDXS)
    traj, p30 = L2.e3_joint(model, P, Dn)
    assert _eq(traj, P.out["traj60"]) and _eq(p30, P.out["p10"])


def test_mutation_e3_code_dropped_goes_red(s2, monkeypatch):
    monkeypatch.setattr(L2, "E3_TRIPLE", ("ego", "ck"))
    with pytest.raises(AssertionError):
        check_e3_scope(L2, s2["model"], s2["P"], s2["Dn"])


def test_mutation_e3_ck_dropped_goes_red(s2, monkeypatch):
    monkeypatch.setattr(L2, "E3_TRIPLE", ("ego", "code"))
    with pytest.raises(AssertionError):
        check_e3_scope(L2, s2["model"], s2["P"], s2["Dn"])


def test_mutation_e3_prior_swapped_goes_red(s2, monkeypatch):
    orig = LM.run_O

    def swapped(model, P, ov=(), Dn=None, **kw):
        b2 = dict(P.b)
        p10f = P.b["p10f"].clone()
        idx = torch.arange(p10f.shape[0])
        p10f[idx, P.last_now] = Dn.b["p10f"][idx, Dn.last_now]
        b2["p10f"] = p10f
        import copy
        P2 = copy.copy(P)
        P2.b = b2
        return orig(model, P2, ov, Dn, **kw)
    monkeypatch.setattr(LM, "run_O", swapped)
    with pytest.raises(AssertionError):
        L2.e3_joint(s2["model"], s2["P"], s2["Dn"])


# ------------------------------------------------------------------------------------------------ E2: the means
MEAN_IDX = sorted(set(IDXS))


@torch.no_grad()
def _independent_means(model, ds, wins, idxs):
    """A second implementation: one window at a time, numpy float64."""
    acc = {}
    scene_s, scene_c = {}, {}
    pad_seen = 0
    n = 0
    for i in idxs:
        P = L2.collate_own(model, ds, wins, [i])
        for L in LM.LAYERS:
            t = LM._tick(P, L)
            b = P.b
            toks = {"ego": model._ego(model._at(b["p10f"], t)), "nav": model.e_nav(model._at(b["nav"], t)),
                    "limit": model.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1))}
            for nm, v in toks.items():
                acc[("tok", L, nm)] = acc.get(("tok", L, nm), 0) + v[0].double().numpy()
            tok, pad = model._scene(P.adapt, b["frame_ok"], t)
            ok = ~pad[0].numpy()
            pad_seen += int(pad.any())
            scene_s[L] = scene_s.get(L, 0) + tok[0].double().numpy() * ok[:, None]
            scene_c[L] = scene_c.get(L, 0) + ok[:, None].astype(np.float64)
        acc[("ck", "T")] = acc.get(("ck", "T"), 0) + P.ck_for_t[0].double().numpy()
        acc[("ck", "O")] = acc.get(("ck", "O"), 0) + P.ck_for_o[0].double().numpy()
        acc[("code",)] = acc.get(("code",), 0) + P.code[0].double().numpy()
        acc[("age",)] = acc.get(("age",), 0) + np.array([float(b["a_t"][0]), float(b["a_s"][0])])
        n += 1
    out = {k: v / n for k, v in acc.items()}
    for L in scene_s:
        out[("scene", L)] = scene_s[L] / scene_c[L]
    return out, n, pad_seen


@pytest.fixture(scope="module")
def means(s2):
    model, ds, wins = s2["model"], s2["ds"], s2["wins"]
    acc = L2.MeanAccumulator(model)
    for b0 in range(0, len(MEAN_IDX), 16):
        acc.add(L2.collate_own(model, ds, wins, MEAN_IDX[b0:b0 + 16]))
    return acc.finish(), acc


def check_means_equal_independent(model, ds, wins, mn, acc):
    ref, n, pad_seen = _independent_means(model, ds, wins, MEAN_IDX)
    assert mn.n == n == len(MEAN_IDX)
    assert pad_seen > 0 and acc.pad_any["S"] > 0, "the test must exercise padded frames"
    for (L, nm), v in mn.tok.items():
        np.testing.assert_allclose(v.double().numpy(), ref[("tok", L, nm)], atol=1e-5, rtol=1e-5, err_msg=f"mean {L}/{nm}")
    for L, v in mn.scene.items():
        np.testing.assert_allclose(v.double().numpy(), ref[("scene", L)], atol=1e-5, rtol=1e-5, err_msg=f"mean scene {L}")
    for L, v in mn.ck.items():
        np.testing.assert_allclose(v.double().numpy(), ref[("ck", L)], atol=1e-4, rtol=1e-5, err_msg=f"mean ck {L}")
    np.testing.assert_allclose(mn.code.double().numpy(), ref[("code",)], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mn.age.double().numpy(), ref[("age",)], atol=1e-6, rtol=1e-6)
    assert len(mn.tok) == 9 and set(mn.scene) == {"S", "T", "O"} and set(mn.ck) == {"T", "O"}


def test_means_equal_an_independent_per_window_computation(s2, means):
    mn, acc = means
    check_means_equal_independent(s2["model"], s2["ds"], s2["wins"], mn, acc)


def test_means_differ_across_layers_that_read_at_different_ticks(s2, means):
    """The per-layer tick matters (a_t, a_s > 0 occur): S's, T's and O's mean ego tokens are not one tensor."""
    mn, _ = means
    assert not torch.allclose(mn.tok[("S", "ego")], mn.tok[("O", "ego")]) and not torch.allclose(mn.tok[("T", "ego")], mn.tok[("O", "ego")])


def test_means_round_trip_through_the_npz(means, tmp_path):
    mn, _ = means
    L2.save_means(mn, tmp_path / "m.npz")
    m2 = L2.load_means(tmp_path / "m.npz")
    assert m2.n == mn.n and _eq(m2.code, mn.code) and _eq(m2.age, mn.age)
    assert all(_eq(m2.tok[k], mn.tok[k]) for k in mn.tok) and all(_eq(m2.scene[k], mn.scene[k]) for k in mn.scene) and all(_eq(m2.ck[k], mn.ck[k]) for k in mn.ck)


def test_mutation_mean_ignores_the_pad_mask_goes_red(s2, monkeypatch):
    """A mean taken over padded frames too differs from the independent masked mean."""
    model, ds, wins = s2["model"], s2["ds"], s2["wins"]
    orig = LM.donor_token

    def unmasked(model_, Dn, layer, name):
        if name == "scene":
            tok, pad = orig(model_, Dn, layer, name)
            return tok, torch.zeros_like(pad)
        return orig(model_, Dn, layer, name)
    monkeypatch.setattr(LM, "donor_token", unmasked)
    acc = L2.MeanAccumulator(model)
    for b0 in range(0, len(MEAN_IDX), 16):
        acc.add(L2.collate_own(model, ds, wins, MEAN_IDX[b0:b0 + 16]))
    with pytest.raises(AssertionError):
        check_means_equal_independent(model, ds, wins, acc.finish(), acc)


def test_mutation_mean_taken_at_the_wrong_tick_goes_red(s2, monkeypatch):
    model, ds, wins = s2["model"], s2["ds"], s2["wins"]
    orig = LM.donor_token
    monkeypatch.setattr(LM, "donor_token", lambda m, Dn, layer, name: orig(m, Dn, "O", name))
    acc = L2.MeanAccumulator(model)
    for b0 in range(0, len(MEAN_IDX), 16):
        acc.add(L2.collate_own(model, ds, wins, MEAN_IDX[b0:b0 + 16]))
    with pytest.raises(AssertionError):
        check_means_equal_independent(model, ds, wins, acc.finish(), acc)


# ------------------------------------------------------------------------------------------------ E2: the replacement cells
def _mean_cells(model):
    return [(L, x) for (L, x) in LM.CELLS if x == "all" or x in LM.reads(model, L)]


def check_mean_scope(L2mod, model, P, mn):
    LMod = L2mod.LS
    src = L2mod.MeanSrc(mn)
    B = P.b["feats"].shape[0]
    with capture(model) as clean:
        LMod.run_chain(model, P)
    n_checked = 0
    for (L, x) in _mean_cells(model):
        ov = LMod.cell_inputs(model, L, x)
        with capture(model) as cp:
            got = L2mod.repl_cell(model, P, src, L, x)
        assert list(cp) == [LMod._READER[L]], f"{L}/{x} ran readers {list(cp)}"
        cond, scene, pad = cp[LMod._READER[L]][0]
        c0, s0, p0 = clean[LMod._READER[L]][0]
        exp = c0.clone()
        pos = LMod.TOKEN_POS[L]
        for nm in ov:
            if nm in ("ego", "nav", "limit"):
                exp[:, pos[nm]] = mn.tok[(L, nm)][None].expand(B, -1)
            elif nm == "ck":
                exp[:, pos["ck"]] = model._ck_cond(mn.ck[L][None].expand(B, -1, -1), P.ck_ok)
            elif nm == "code":
                exp[:, pos["code"]] = model.e_code(mn.code[None].expand(B, -1))
            elif nm == "age":
                exp[:, pos["age"]] = model.e_age(torch.stack([mn.age[0].expand(B) * LMod.DT, mn.age[1].expand(B) * LMod.DT], -1))
        assert _eq(cond, exp), f"{L}/{x}: conditioning tokens differ from 'own except the mean of {ov}'"
        if "scene" in ov:
            assert _eq(scene, mn.scene[L][None].expand(B, -1, -1)) and not bool(pad.any()), f"{L}/{x}: not the (unpadded) mean scene"
        else:
            assert _eq(scene, s0) and _eq(pad, p0), f"{L}/{x}: the scene path moved"
        assert _moved(got, LMod.clean_cell(model, P, L)), f"{L}/{x}: the mean cell did not move the layer's output (a dead probe)"
        if L == "O":
            assert _eq(got[1], P.out["p10"]), f"O/{x}: the kinematic prior moved"
        n_checked += 1
    assert n_checked == 5 + 6 + 8
    return n_checked


def test_every_mean_cell_hands_the_reader_the_mean_and_nothing_else(s2, means):
    check_mean_scope(L2, s2["model"], s2["P"], means[0])


def test_the_windows_own_value_through_the_mean_path_is_the_clean_pass_bit_for_bit(s2):
    model, P = s2["model"], s2["P"]
    own = L2.OwnSrc()
    for (L, x) in _mean_cells(model):
        got, clean = L2.repl_cell(model, P, own, L, x), LM.clean_cell(model, P, L)
        if L == "T":
            got = {k: got[k] for k in clean}
        assert _tree_equal(got, clean), f"the window's own value moved cell {L}/{x} (a mean cell would not read 0.0)"


def test_mutation_mean_override_never_reaches_the_layer_goes_red(s2, means, monkeypatch):
    @contextlib.contextmanager
    def noop(model, repl):
        yield
    monkeypatch.setattr(LM, "_patched", noop)
    with pytest.raises(AssertionError):
        check_mean_scope(L2, s2["model"], s2["P"], means[0])


def test_mutation_mean_code_not_replaced_goes_red(s2, means, monkeypatch):
    class Src(L2.MeanSrc):
        def code(self, model, P):
            return P.code
    monkeypatch.setattr(L2, "MeanSrc", Src)
    with pytest.raises(AssertionError):
        check_mean_scope(L2, s2["model"], s2["P"], means[0])


def test_mutation_own_value_in_the_wrong_frame_goes_red(s2):
    """The own-value control is able to fail: T's own S-checkpoints re-expressed in O's frame are not T's own."""
    class Wrong(L2.OwnSrc):
        def ck(self, model, P, layer):
            return P.ck_for_o if layer == "T" else P.ck_for_t
    model, P = s2["model"], s2["P"]
    got, clean = L2.repl_cell(model, P, Wrong(), "T", "ck"), LM.clean_cell(model, P, "T")
    assert not _tree_equal({k: got[k] for k in clean}, clean)


# ------------------------------------------------------------------------------------------------ the tables, verbatim
def _flags(bar, sep, delta=0.1):
    return {"delta": delta, "separated": sep, "bar_met": bar, "lo": 0, "hi": 0, "floor": 0, "positive": True, "above_floor": True}


def test_e1_outcome_table_is_applied_verbatim():
    import read2 as R2
    both = {"s0": _flags(True, True), "s1": _flags(True, True)}
    assert R2.e1_outcome(both)["headline"] == "ROUTE FLOWS DOWN"
    assert R2.e1_outcome({"s0": _flags(False, False), "s1": _flags(False, False)})["headline"] == "PARALLEL PATH"
    assert R2.e1_outcome({"s0": _flags(True, True), "s1": _flags(False, False)})["headline"] == "SEEDS DISAGREE"
    assert R2.e1_outcome({"s0": _flags(False, True), "s1": _flags(False, True)})["headline"] == "SEEDS DISAGREE"       # separated, but below the floor / negative
    assert R2.e1_outcome({"s0": _flags(True, True), "s1": _flags(False, True)})["headline"] == "SEEDS DISAGREE"


def test_e1_seed_flags_apply_the_bar_and_the_floor():
    import read2 as R2
    blk = lambda d, sep: {"rows": {"T-lat": {"delta": d, "lo": d - 0.1, "hi": d + 0.1, "separated": sep}}}              # noqa: E731
    fl = {"T-lat": 0.05}
    assert R2.e1_seed_flags(blk(0.2, True), fl)["bar_met"]
    assert not R2.e1_seed_flags(blk(0.04, True), fl)["bar_met"]            # below the donor-seed floor
    assert not R2.e1_seed_flags(blk(0.2, False), fl)["bar_met"]            # not separated
    assert not R2.e1_seed_flags(blk(-0.2, True), fl)["bar_met"]            # separated but the wrong sign
    f = R2.donor_floor_e1({"rows": {"T-lat": {"delta": 0.30}}}, {"rows": {"T-lat": {"delta": 0.26}}})
    assert abs(f["T-lat"] - 0.04) < 1e-12


def _rec(argmax_by_row, deltas=None):
    import listen_read as LR
    rec = {"rows": {}}
    for row in LR.ROWS:
        cells = {}
        for col in LR.COLS:
            if (row in ("S",) and col in ("ck", "code", "age")) or (row in ("T-lat", "T-lon") and col in ("code", "age")):
                cells[col] = None
            else:
                d = 1.0 if col == argmax_by_row[row] else 0.1
                cells[col] = {"delta": d, "lo": d - 0.05, "hi": d + 0.05, "separated": True}
        rec["rows"][row] = {"cells": cells}
    return rec


def test_e2_verdict_table_is_applied_verbatim():
    import read2 as R2
    garg = {s: dict(R2.PREREG_GRID_ARGMAX) for s in ("s0", "s1")}
    same = dict(R2.PREREG_GRID_ARGMAX)
    assert R2.e2_verdict({"s0": _rec(same), "s1": _rec(same)}, garg)["headline"] == "ROBUST"
    two = {**same, "S": "scene", "O": "code"}
    v = R2.e2_verdict({"s0": _rec(two), "s1": _rec(two)}, garg)
    assert v["headline"] == "PARTLY ROBUST" and v["k_rows_agreeing_at_both_seeds"] == 2
    three = {**same, "O": "code"}
    assert R2.e2_verdict({"s0": _rec(three), "s1": _rec(three)}, garg)["headline"] == "PARTLY ROBUST"
    one = {"S": "scene", "T-lat": "ego", "T-lon": "scene", "O": "ego"}
    assert R2.e2_verdict({"s0": _rec(one), "s1": _rec(one)}, garg)["headline"] == "NOT ROBUST"
    # a row counts only when it agrees at BOTH seeds (M4): all 4 at s0, 3 at s1 -> 3 rows -> PARTLY ROBUST
    v = R2.e2_verdict({"s0": _rec(same), "s1": _rec({**same, "O": "code"})}, garg)
    assert v["headline"] == "PARTLY ROBUST" and v["k_per_seed"] == {"s0": 4, "s1": 3} and v["k_rows_agreeing_at_both_seeds"] == 3


def test_e3_outcome_table_is_applied_verbatim():
    import read2 as R2
    b = lambda d, sep: {"E3": {"delta": d, "separated": sep}}                                                         # noqa: E731
    assert R2.e3_outcome({"s0": b(-0.3, True), "s1": b(-0.2, True)})["headline"] == "INCOHERENCE"
    assert R2.e3_outcome({"s0": b(-0.3, True), "s1": b(-0.2, False)})["headline"] == "NOT SHOWN"
    assert R2.e3_outcome({"s0": b(-0.3, True), "s1": b(0.2, True)})["headline"] == "NOT SHOWN"
    assert R2.e3_outcome({"s0": b(0.0, False), "s1": b(0.0, False)})["headline"] == "NOT SHOWN"


# ------------------------------------------------------------------------------------------------ the pipeline on the synthetic eval split
def test_pipeline_runs_and_the_identity_and_own_controls_read_exactly_zero(eval_world, s2, means):
    import listen_read as LR
    import read2 as R2
    import atexit

    import run_listen as RL_
    import run_listen2 as RL2
    atexit.unregister(RL_._peak_report)                                              # the tests must not write into the real runs' peakmem.log
    X, ds, wins = eval_world["X"], eval_world["ds"], eval_world["wins"]
    N = len(wins)
    model = sensitive_model(eval_world["J"], seed=0)
    mn = means[0]                                                                    # token SHAPES are all the eval model needs from these means
    # donor run, identity run, mean run, own run
    res_d, n_assert = L2.collect2(model, ds, wins, X["gt"], LM.donor_perm(N, X["eid"], 0), "donor", batch=5)
    assert n_assert == N
    res_d1, _ = L2.collect2(model, ds, wins, X["gt"], LM.donor_perm(N, X["eid"], 1), "donor", batch=5)
    res_id, n_id = L2.collect2(model, ds, wins, X["gt"], np.arange(N), "donor", batch=5)
    res_m, _ = L2.collect2(model, ds, wins, X["gt"], None, "mean", means=mn, batch=5)
    res_o, _ = L2.collect2(model, ds, wins, X["gt"], None, "own", batch=5)
    assert n_id == N
    for res in (res_id, res_o):
        c = RL2.control_identical_to_clean(res)
        assert c["pass"], c
        assert RL2.control_delta_zero(res, X)["pass"]
    # the real arms move
    assert not RL2.control_identical_to_clean(res_d)["pass"] and not RL2.control_identical_to_clean(res_m)["pass"]
    blk, blk1 = R2.e1_block(res_d, X, B=40), R2.e1_block(res_d1, X, B=40)
    floor = R2.donor_floor_e1(blk, blk1)
    assert set(floor) == {"S", "T-lat", "T-lon", "O", "O-ade2"}
    e3 = R2.e3_block(res_d, X, B=40)
    assert e3["comparator"] in L2.E3_SINGLES and e3["E3"]["delta"] == pytest.approx(e3["mean_ade6"]["joint"] - e3["mean_ade6"][e3["comparator"]])
    rec = LR.run_record(res_m, X, B=40)
    for row in LR.ROWS:                                                              # hatched cells = not an input of the LAYER
        for col in LR.COLS:
            assert (rec["rows"][row]["cells"][col] is None) == (col not in LM._READS[LR.LAYER_OF[row]]), (row, col)
    LR.add_n(rec)
    assert R2.e2_argmax(rec, "O") in LR.COLS
    # the E1 S-row equals the S x nav cell exactly (S's step of the chain IS that cell)
    assert np.array_equal(res_d[L2.E1_NAME]["ck_now"], res_d["S/nav"]["ck_now"])
    # the collector's stored E1 / E3 rows equal an independent recompute of the first batch (a collector that stored the wrong arm would go RED)
    idx5 = list(range(5))
    P5, D5, a5 = LM.collate_window_batch(model, ds, wins, LM.donor_perm(N, X["eid"], 0), idx5)
    r5 = LM.run_chain(model, P5, D5, {"S": ("nav",)}, ages=a5)
    ade = lambda t60: np.linalg.norm(LM.M.subsample_02(t60).numpy().astype(np.float64) - X["gt"][idx5], axis=-1).mean(1)      # noqa: E731
    np.testing.assert_allclose(res_d[L2.E1_NAME]["ade6"][:5], ade(r5["traj60"]), atol=1e-6)
    assert np.array_equal(res_d[L2.E1_NAME]["lat_pred"][:5], r5["tout"]["lat_logits"].argmax(-1).numpy())
    j5, _ = LM.run_O(model, P5, ("ego", "ck", "code"), D5)
    np.testing.assert_allclose(res_d[L2.E3_NAME]["ade6"][:5], ade(j5), atol=1e-6)
    assert not np.allclose(res_d[L2.E3_NAME]["ade6"][:5], res_d["O/ego"]["ade6"][:5])
