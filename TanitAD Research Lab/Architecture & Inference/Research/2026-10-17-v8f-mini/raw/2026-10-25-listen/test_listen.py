"""Tests for listen.py (PREREG_V8F_LISTEN). CPU only, synthetic world, random-init tiny model made SENSITIVE (every input moves every layer).

What is pinned, and the control that must go red for each:
  (a) IDENTITY: donor = self gives outputs BIT-IDENTICAL to ``forward_infer`` for every cell, every row reference, every Q-NAV arm and the
      whole chain; the collector's per-window arrays equal the clean ones exactly (so every Delta is exactly 0.0).
  (b) SCOPE: each override changes ONLY the intended conditioning token (or the scene tokens + pad), read at the layer's Reader through a
      pre-hook and compared with the clean pass and with the DONOR's own pass; only that layer runs; the output of every cell MOVES (a zero below
      is a measurement, not a dead probe). O x ego leaves the PRIOR on the window's own p10f. The Q-NAV chain arms change only the intended
      layer's nav token and let the change flow downstream (and nowhere else).
  (c) STRUCTURAL ZEROS: a copy whose nav token is a constant reads O x nav EXACTLY 0.0; a copy whose T builds no scene path reads T x scene
      EXACTLY 0.0 -- while the same cells on the unregressed model read non-zero.
  (d) MUTATION: breaking an override (never reaches the layer / wrong donor tick / prior swapped with the ego / donor ages = own / own ck)
      makes the checks go RED; the unbroken module is green.
  plus: the donor-cycle age clamp equals the dataset's; non-inputs are refused; the shadows are removed and an override that does not reach
  its layer is refused; the collector reproduces the evaluator's checkpoint re-expression.
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


def _load(path, name="listen_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


LM = _load(os.environ.get("LISTEN_PY", str(HERE / "listen.py")))
from tanitad.data import v8f_mini_data as D  # noqa: E402  (listen.py has put the exported tip first on sys.path)
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, d_feat=64)


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    w = D.make_synthetic_world(root, n_train=5, n_eval=2, T=56, seed=0, d_feat=64)
    J = D.load_join(Path(root) / "join_train.npz")
    J["limit"] = np.eye(4, dtype=np.float32)[np.arange(len(J["clip"])) % 4]   # the synthetic world has ONE limit; a limit cell needs donors that differ
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"], start=0)          # from t = 0: early windows have PADDED frames (the pad mask matters)
    ds = D.V8FMiniWindows(store, J, wins, hmax=10, window_start=0)
    eid = np.array([ci for ci, _, _, _ in wins])
    gt = np.stack([ds.item(ci, t, 0, 0)["gt"] for ci, t, _, _ in wins]).astype(np.float64)
    return {"root": root, "J": J, "ds": ds, "wins": wins, "eid": eid, "gt": gt}


def sensitive_model(J, seed=0, scale=0.3, **kw):
    cfg = M.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not n.startswith("prior."):
                p.add_(torch.randn(p.shape, generator=g) * scale)
    return model.eval()


# 26 windows per clip (t = 0..25). Every 4th window (i % 3 cycles, so a_t = 0, 1, 2 all occur: T's tick differs from NOW) plus ranks 0-2 of every clip
# (padded frames, clamped ages), whose donors are also early windows.
IDXS = sorted(set(range(0, 130, 4)) | {26 * k + r for k in range(5) for r in range(3)})


@pytest.fixture(scope="module")
def setup(world):
    model = sensitive_model(world["J"])
    perm = LM.donor_perm(len(world["wins"]), world["eid"], 0)
    P, Dn, ages_d = LM.collate_window_batch(model, world["ds"], world["wins"], perm, IDXS)
    return {"model": model, "perm": perm, "P": P, "Dn": Dn, "ages_d": ages_d, **world}


@contextlib.contextmanager
def capture(model):
    """Record what each Reader is handed: (cond, scene, pad)."""
    cap, hs = {}, []
    for nm in ("read_s", "read_t", "read_o"):
        def hook(mod, args, nm=nm):
            cap.setdefault(nm, []).append(tuple(None if a is None else a.detach().clone() for a in args))
        hs.append(getattr(model, nm).register_forward_pre_hook(hook))
    try:
        yield cap
    finally:
        for h in hs:
            h.remove()


def _eq(a, b):
    return torch.equal(a, b)


def _tree_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return _eq(a, b)


def _moved(a, b):
    if isinstance(a, dict):
        return any(_moved(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return any(_moved(x, y) for x, y in zip(a, b))
    return float((a - b).abs().max()) > 1e-6


# ------------------------------------------------------------------------------------------------ the checks (also used by the mutation tests)
def check_identity(LMod, model, world):
    """(a) donor = self."""
    ds, wins = world["ds"], world["wins"]
    ident = np.arange(len(wins))
    P, Dn, ages_d = LMod.collate_window_batch(model, ds, wins, ident, IDXS)
    assert _eq(ages_d[0], P.b["a_t"]) and _eq(ages_d[1], P.b["a_s"]), "identity donor must carry the window's own effective ages"
    with torch.no_grad():                                              # the evaluator's mode (grad mode may pick other kernels)
        ref = model.forward_infer(P.b)
    ch = LMod.run_chain(model, P)
    for k in ("ckpt", "code", "traj60", "p10"):
        assert _eq(ch[k], ref[k]), f"chain({k}) != forward_infer"
    for k in ("goal_logits", "lat_logits", "lon_logits", "cons_std", "goal_prob", "cons_pred"):
        assert _eq(ch["tout"][k], ref[k]), f"chain T({k}) != forward_infer"
    for (L, x) in LMod.CELLS:
        if x != "all" and x not in LMod.reads(model, L):
            continue
        got, clean = LMod.cell(model, P, Dn, L, x, ages=ages_d), LMod.clean_cell(model, P, L)
        if L == "T":
            got = {k: got[k] for k in clean}
        assert _tree_equal(got, clean), f"identity donor moved cell {L}/{x}"
    for arm, ov in LMod.QNAV_ARMS.items():
        r = LMod.run_chain(model, P, Dn, ov, ages=ages_d)
        assert _eq(r["traj60"], ref["traj60"]), f"identity donor moved {arm}"
    return P, Dn, ages_d


def _expected_reader_inputs(LMod, model, L, ov, clean, donor, ages_d):
    """What the layer's Reader must be handed when ``ov`` come from the donor and nothing else does."""
    rd = LMod._READER[L]
    c0, s0, p0 = clean[rd][0]
    cd, sd, pd = donor[rd][0]
    cond, scene, pad = c0.clone(), s0, p0
    for x in ov:
        if x == "scene":
            scene, pad = sd, pd
        elif x == "age":
            cond[:, LMod.TOKEN_POS[L]["age"]] = model.e_age(torch.stack([ages_d[0].float() * LMod.DT, ages_d[1].float() * LMod.DT], -1))
        else:
            cond[:, LMod.TOKEN_POS[L][x]] = cd[:, LMod.TOKEN_POS[L][x]]
    return cond, scene, pad


def check_scope(LMod, model, P, Dn, ages_d):
    """(b) each cell changes only what it should; the layer's output moves; the prior stays on the window's own p10f."""
    with capture(model) as clean:
        LMod.run_chain(model, P)
    with capture(model) as donor:
        LMod.run_chain(model, Dn)
    for (L, x) in LMod.CELLS:
        if x != "all" and x not in LMod.reads(model, L):
            continue
        ov = LMod.cell_inputs(model, L, x)
        with capture(model) as cp:
            got = LMod.cell(model, P, Dn, L, x, ages=ages_d)
        assert list(cp) == [LMod._READER[L]], f"{L}/{x} ran readers {list(cp)}: only {LMod._READER[L]} may run (upstream is not recomputed)"
        cond, scene, pad = cp[LMod._READER[L]][0]
        ec, es, ep = _expected_reader_inputs(LMod, model, L, ov, clean, donor, ages_d)
        assert _eq(cond, ec), f"{L}/{x}: conditioning tokens differ from 'own except {ov}'"
        assert _eq(scene, es) and _eq(pad, ep), f"{L}/{x}: scene tokens / pad differ from expectation"
        assert _moved(got, LMod.clean_cell(model, P, L)), f"{L}/{x} did not move the layer's output (a dead probe reads 0 for the wrong reason)"
    # O x ego (and the O row reference) keep the PRIOR on the window's own p10f
    for x in ("ego", "all"):
        traj, p30 = LMod.cell(model, P, Dn, "O", x, ages=ages_d)
        assert _eq(p30, P.out["p10"]), f"O/{x} swapped the kinematic prior (it must keep the window's own p10f)"
        assert not _eq(traj, P.out["traj60"])
    # Q-NAV: only the named layers' nav token changes, the change flows downstream and nowhere else
    for arm, ovs in LMod.QNAV_ARMS.items():
        with capture(model) as cp:
            r = LMod.run_chain(model, P, Dn, ovs, ages=ages_d)
        for L in LMod.LAYERS:
            rd = LMod._READER[L]
            cond, scene, pad = cp[rd][0]
            c0, s0, p0 = clean[rd][0]
            cd = donor[rd][0][0]
            pos_nav = LMod.TOKEN_POS[L]["nav"]
            assert _eq(scene, s0) and _eq(pad, p0), f"{arm}: {L} scene moved"
            for name, pos in LMod.TOKEN_POS[L].items():
                if name == "nav" and L in ovs:
                    assert _eq(cond[:, pos], cd[:, pos]), f"{arm}: {L} nav token is not the donor's"
                elif name == "nav":
                    assert _eq(cond[:, pos], c0[:, pos]), f"{arm}: {L} nav token moved though no nav was injected there"
                elif name in ("ck", "code"):                           # held upstream outputs: they move iff an upstream layer got the donor nav
                    upstream_moved = ("S" in ovs) if name == "ck" else ("S" in ovs or "T" in ovs)
                    assert _eq(cond[:, pos], c0[:, pos]) != upstream_moved, f"{arm}: {L}.{name} flow is wrong (moved={upstream_moved})"
                else:
                    assert _eq(cond[:, pos], c0[:, pos]), f"{arm}: {L}.{name} token moved"
        assert _moved(r["traj60"], P.out["traj60"]), f"{arm} did not move O"
    return True


# ------------------------------------------------------------------------------------------------ (a) identity
def test_identity_donor_is_bit_exact_for_every_cell_arm_and_chain(world):
    model = sensitive_model(world["J"])
    check_identity(LM, model, world)


def test_identity_donor_collector_arrays_equal_clean_exactly(world):
    model = sensitive_model(world["J"])
    wins = world["wins"]
    res = LM.collect(model, world["ds"], wins, world["gt"], np.arange(len(wins)), batch=17, n_use=60)
    clean = res["clean/S"]["ck_now"], res["clean/T"]["lat_pred"], res["clean/T"]["lon_pred"], res["clean/O"]["ade6"], res["clean/O"]["ade2"]
    n = 0
    for name, r in res.items():
        if name.startswith("clean/"):
            continue
        L = name.split("/")[0]
        if name.startswith("nav@"):
            assert np.array_equal(r["ade6"], clean[3]) and np.array_equal(r["ade2"], clean[4]), name
        elif L == "S":
            assert np.array_equal(r["ck_now"], clean[0]), name
        elif L == "T":
            assert np.array_equal(r["lat_pred"], clean[1]) and np.array_equal(r["lon_pred"], clean[2]), name
        else:
            assert np.array_equal(r["ade6"], clean[3]) and np.array_equal(r["ade2"], clean[4]), name
        n += 1
    assert n == 5 + 6 + 8 + 4, n                                       # S 4+all, T 5+all, O 7+all, Q-NAV 4 arms


# ------------------------------------------------------------------------------------------------ (b) scope
def test_each_override_changes_only_the_intended_input(setup):
    check_scope(LM, setup["model"], setup["P"], setup["Dn"], setup["ages_d"])


def test_donor_is_from_another_episode_and_ages_differ_somewhere(setup):
    eid, perm = setup["eid"], setup["perm"]
    assert all(eid[perm[i]] != eid[i] for i in range(len(perm)))
    a_own = setup["P"].b["a_t"] * 10 + setup["P"].b["a_s"]
    a_don = setup["ages_d"][0] * 10 + setup["ages_d"][1]
    assert bool((a_own != a_don).any()), "the age cell needs windows whose donor-cycle ages differ"


def test_o_ego_cell_reads_the_ego_token_of_the_donor_but_keeps_the_prior(setup):
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    with capture(model) as cp:
        traj, p30 = LM.cell(model, P, Dn, "O", "ego")
    cond = cp["read_o"][0][0]
    i = Dn.last_now
    assert _eq(cond[:, 0], model._ego(model._at(Dn.b["p10f"], i))), "ego token is not the donor's"
    assert _eq(p30, P.out["p10"]), "the prior moved"
    assert not bool(torch.equal(model._at(P.b["p10f"], P.last_now), model._at(Dn.b["p10f"], Dn.last_now)))


# ------------------------------------------------------------------------------------------------ (c) structural zeros
def test_structural_zero_o_x_nav_with_constant_nav_token(setup):
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    assert _moved(LM.cell(model, P, Dn, "O", "nav"), LM.clean_cell(model, P, "O")), "control: the unregressed model must listen to nav"
    m = LM.regress_copy(model, "nav_const")
    P1, D1 = LM.make_pass(m, P.b), LM.make_pass(m, Dn.b)
    for L in LM.LAYERS:
        got, clean = LM.cell(m, P1, D1, L, "nav"), LM.clean_cell(m, P1, L)
        if L == "T":
            got = {k: got[k] for k in clean}
        assert _tree_equal(got, clean), f"{L} x nav must read EXACTLY 0.0 on a constant nav token"
    assert _moved(LM.cell(m, P1, D1, "O", "ego"), LM.clean_cell(m, P1, "O")), "the regressed copy must still listen to other inputs"


def test_structural_zero_t_x_scene_when_t_builds_no_scene_path(setup):
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    assert _moved(LM.cell(model, P, Dn, "T", "scene"), LM.clean_cell(model, P, "T")), "control: T must listen to the scene"
    m = LM.regress_copy(model, "tscene_off")
    P1, D1 = LM.make_pass(m, P.b), LM.make_pass(m, Dn.b)
    got, clean = LM.cell(m, P1, D1, "T", "scene"), LM.clean_cell(m, P1, "T")
    assert _tree_equal({k: got[k] for k in clean}, clean), "T x scene must read EXACTLY 0.0 when T has no scene path"
    assert _moved(LM.cell(m, P1, D1, "T", "ego"), LM.clean_cell(m, P1, "T")), "the regressed copy must still listen to the ego token"


# ------------------------------------------------------------------------------------------------ (d) mutations
def _fresh_checks(LMod, setup):
    model, P, Dn, ages_d = setup["model"], setup["P"], setup["Dn"], setup["ages_d"]
    check_scope(LMod, model, P, Dn, ages_d)


def test_unbroken_module_passes_the_checks(setup):
    _fresh_checks(LM, setup)


def test_mutation_override_never_reaches_the_layer_goes_red(setup, monkeypatch):
    @contextlib.contextmanager
    def noop(model, repl):
        yield
    monkeypatch.setattr(LM, "_patched", noop)
    with pytest.raises((AssertionError, RuntimeError)):
        _fresh_checks(LM, setup)


def test_mutation_donor_token_read_at_the_wrong_tick_goes_red(setup, monkeypatch):
    orig = LM.donor_token

    def wrong(model, Dn, layer, name):
        if name in ("nav", "ego"):                                      # the donor's token at the NOW frame instead of its own tick
            i = Dn.last_now
            return model.e_nav(model._at(Dn.b["nav"], i)) if name == "nav" else model._ego(model._at(Dn.b["p10f"], i))
        return orig(model, Dn, layer, name)
    monkeypatch.setattr(LM, "donor_token", wrong)
    with pytest.raises((AssertionError, RuntimeError)):
        _fresh_checks(LM, setup)


def test_mutation_donor_ages_equal_own_goes_red(setup, monkeypatch):
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    own = (P.b["a_t"], P.b["a_s"])
    with pytest.raises(AssertionError):                                # the cell is run with the window's OWN ages: not the donor-cycle token
        _scope_with_ages(model, P, Dn, own, setup["ages_d"])


def _scope_with_ages(model, P, Dn, used_ages, true_ages):
    """The age cell is run with ``used_ages`` but the expectation is built from the donor-cycle ages ``true_ages``."""
    with capture(model) as clean:
        LM.run_chain(model, P)
    with capture(model) as donor:
        LM.run_chain(model, Dn)
    with capture(model) as cp:
        LM.cell(model, P, Dn, "O", "age", ages=used_ages)
    cond = cp["read_o"][0][0]
    ec, _, _ = _expected_reader_inputs(LM, model, "O", ("age",), clean, donor, true_ages)
    assert _eq(cond, ec), "age token is not the donor-cycle token"


def test_age_cell_is_alive_with_the_true_donor_cycle_ages(setup):
    _scope_with_ages(setup["model"], setup["P"], setup["Dn"], setup["ages_d"], setup["ages_d"])


def test_mutation_ego_swap_also_swaps_the_prior_goes_red(setup, monkeypatch):
    """The failure class the PREREG warns about: swapping p10f in the batch swaps the prior too."""
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    b2 = dict(P.b)
    p10f = P.b["p10f"].clone()
    idx = torch.arange(p10f.shape[0])
    p10f[idx, P.last_now] = Dn.b["p10f"][idx, Dn.last_now]
    b2["p10f"] = p10f
    traj, p30 = model.run_o(b2, P.adapt, P.last_now, P.ck_for_o, P.ck_ok, P.code)
    assert not _eq(p30, P.out["p10"]), "the broken (batch-swap) variant moves the prior, so the pin on O/ego is able to fail"


def test_mutation_ck_override_uses_own_checkpoints_goes_red(setup, monkeypatch):
    model, P, Dn, ages_d = setup["model"], setup["P"], setup["Dn"], setup["ages_d"]
    own_ck = P.ck_for_t
    orig = LM.run_T

    def broken(model, P_, ov=(), Dn_=None, ck_t=None):
        if "ck" in tuple(ov):                                          # 'donor value left unused': the clean ck is read
            return orig(model, P_, tuple(x for x in ov if x != "ck"), Dn_, ck_t=own_ck)
        return orig(model, P_, ov, Dn_, ck_t)
    monkeypatch.setattr(LM, "run_T", broken)
    with pytest.raises(AssertionError):
        _fresh_checks(LM, setup)


# ------------------------------------------------------------------------------------------------ plumbing
def test_donor_cycle_ages_equal_the_datasets_clamp(world):
    ds, wins = world["ds"], world["wins"]
    n_clamped = 0
    for j in range(len(wins)):
        ci, t, _, _ = wins[j]
        a_t, a_s = LM.cycle_ages(j)
        it = ds.item(ci, t, a_t, a_s)
        assert LM.donor_cycle_ages(ds, wins, j) == (it["a_t"], it["a_s"]), j
        n_clamped += (it["a_t"], it["a_s"]) != (a_t, a_s)
    assert n_clamped > 0, "the test must exercise the clamp"


def test_non_inputs_are_refused(setup):
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    for L, x in (("S", "ck"), ("S", "code"), ("S", "age"), ("T", "code"), ("T", "age")):
        with pytest.raises(ValueError, match="not an input"):
            LM.cell(model, P, Dn, L, x)
    with pytest.raises(ValueError):
        LM.run_O(model, P, ("age",), Dn)                                # no ages supplied
    nav_only = sensitive_model(setup["J"], navonly=True)
    P1, D1 = LM.make_pass(nav_only, P.b), LM.make_pass(nav_only, Dn.b)
    for L in ("S", "T"):
        with pytest.raises(ValueError, match="not an input"):
            LM.cell(nav_only, P1, D1, L, "scene")
    assert "scene" in LM.reads(nav_only, "O") and "scene" not in LM.reads(nav_only, "S")


def test_shadows_are_removed_and_an_unreached_override_is_refused(setup):
    model, P, Dn = setup["model"], setup["P"], setup["Dn"]
    before = set(model.__dict__)
    LM.cell(model, P, Dn, "O", "all", ages=setup["ages_d"])
    assert set(model.__dict__) == before, "a shadow leaked"
    assert isinstance(model.e_nav, torch.nn.Linear) and model._ego.__func__ is M.V8FMini._ego
    with pytest.raises(ValueError):
        LM.cell(model, P, Dn, "S", "ck")
    assert set(model.__dict__) == before
    # a layer that ignores a shadowed producer: the call-count check refuses the cell

    def skips_nav(b, adapt, last_s, age_rel=None):                                 # reads no nav token at all
        cond = torch.stack([model._ego(model._at(b["p10f"], last_s)), model.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1))], 1)
        return model.read_s(cond, None, None)
    model.run_s = skips_nav
    try:
        with pytest.raises(RuntimeError, match="did not reach the layer"):
            LM.run_S(model, P, ("nav",), Dn)
    finally:
        del model.__dict__["run_s"]
    assert set(model.__dict__) == before


def test_collector_clean_checkpoints_equal_the_evaluators_reexpression(world):
    """``v8f_mini_eval._ckpt_to_now`` (per window, ``ds.item``) == the collector's batch re-expression."""
    import v8f_mini_eval as EV
    model = sensitive_model(world["J"])
    ds, wins = world["ds"], world["wins"]
    n = 40
    res = LM.collect(model, ds, wins, world["gt"], LM.donor_perm(len(wins), world["eid"], 0), batch=16, qnav=False, n_use=n)
    ck = []
    for i in range(n):
        ci, t, _, _ = wins[i]
        a_t, a_s = LM.cycle_ages(i)
        b = D.collate([ds.item(ci, t, a_t, a_s)])
        with torch.no_grad():
            ck.append(model.forward_infer(b)["ckpt"].numpy())
    ck = np.concatenate(ck)
    ref = EV._ckpt_to_now(ck, ds, wins[:n], "cycle")
    np.testing.assert_allclose(res["clean/S"]["ck_now"], ref, rtol=1e-4, atol=1e-3)


# ------------------------------------------------------------------------------------------------ the reading (listen_read) on the synthetic EVAL split
@pytest.fixture(scope="module")
def eval_world(world):
    import json
    root = Path(world["root"])
    grid = json.load(open(root / "eval_grid_windows.json"))
    ej = D.load_join(root / "join_eval.npz")
    X = D.eval_context(grid, root / "eval_kin.npz", ej, kin_mode="steer")
    store = D.ClipStore(root, "eval", lru=8)
    wins = D.eval_grid_windows(root / "eval_grid_windows.json", ej)
    ds = D.V8FMiniWindows(store, ej, wins, hmax=10)
    return {"X": X, "ds": ds, "wins": wins, "J": ej}


def test_reading_pipeline_runs_and_identity_reads_exactly_zero(eval_world):
    import listen_read as LR
    X, ds, wins = eval_world["X"], eval_world["ds"], eval_world["wins"]
    N = len(wins)
    models = {s: sensitive_model(eval_world["J"], seed=k) for k, s in enumerate(("s0", "s1"))}
    recs = {}
    for key, (s, dseed) in {"s0": ("s0", 0), "s1": ("s1", 0), "s0_d1": ("s0", 1)}.items():
        perm = LM.donor_perm(N, X["eid"], dseed)
        res = LM.collect(models[s], ds, wins, X["gt"], perm, batch=5)
        recs[key] = LR.run_record(res, X, B=40)
    out = LR.analyse(recs)
    assert out["outcome"]["headline"] in ("DIVISION OF LABOUR SHOWN", "PARTIAL", "NOT SHOWN", "SEEDS DISAGREE")
    assert out["qnav"]["headline"] in ("THROUGH T", "AROUND T", "BOTH / NOT ESTABLISHED")
    for row in LR.ROWS:
        cells = recs["s0"]["rows"][row]["cells"]
        assert cells["all"] is not None
        for col in LR.COLS:                                              # hatched = not an input of the LAYER
            assert (cells[col] is None) == (col not in LM._READS[LR.LAYER_OF[row]]), (row, col)
    assert set(recs["s0"]["qnav"]) == {"nav@S", "nav@T", "nav@O", "nav@STO", "D_T_minus_O"}
    # identity: every cell's delta is exactly 0.0 and nothing is separated
    perm = np.arange(N)
    res_id = LM.collect(models["s0"], ds, wins, X["gt"], perm, batch=5)
    rid = LR.run_record(res_id, X, B=40)
    for part in (rid["rows"], rid["beside"]):
        for row, ent in part.items():
            for col, c in ent["cells"].items():
                if c is not None:
                    assert c["delta"] == 0.0 and not c["separated"], (row, col, c)
    for arm, c in rid["qnav"].items():
        assert c["delta"] == 0.0, arm


def test_outcome_and_qnav_tables_are_applied_verbatim():
    import listen_read as LR
    hold = lambda u0, u1, l0, l1: {"P-UPPER": {"s0": u0, "s1": u1}, "P-LOWER": {"s0": l0, "s1": l1}}   # noqa: E731
    assert LR.outcome(hold(True, True, True, True))["headline"] == "DIVISION OF LABOUR SHOWN"
    assert LR.outcome(hold(True, True, False, False))["headline"] == "PARTIAL"
    assert LR.outcome(hold(False, False, True, True))["headline"] == "PARTIAL"
    assert LR.outcome(hold(False, False, False, False))["headline"] == "NOT SHOWN"
    assert LR.outcome(hold(True, False, False, False))["headline"] == "SEEDS DISAGREE"
    o = LR.outcome(hold(True, True, True, False))                       # one holds at both, the other at one seed only
    assert o["headline"] == "PARTIAL" and o["seeds_disagree_on"] == ["P-LOWER"]
    D_ = lambda d, sep: {"D_T_minus_O": {"delta": d, "separated": sep}}                                  # noqa: E731
    assert LR.qnav_outcome({"s0": D_(0.5, True), "s1": D_(0.4, True)})["headline"] == "THROUGH T"
    assert LR.qnav_outcome({"s0": D_(-0.5, True), "s1": D_(-0.4, True)})["headline"] == "AROUND T"
    assert LR.qnav_outcome({"s0": D_(0.5, True), "s1": D_(-0.4, True)})["headline"] == "BOTH / NOT ESTABLISHED"
    assert LR.qnav_outcome({"s0": D_(0.5, True), "s1": D_(0.4, False)})["headline"] == "BOTH / NOT ESTABLISHED"
