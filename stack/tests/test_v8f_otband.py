"""PREREG_V8F_OTBAND (SPEC_HWM section 9.1, PI ruling): per-band trajectory loss and metrics on O's ONE 30-knot decoder, and the strategic token into T.
``--w-op`` (knots 1-10), ``--w-tac-traj`` (knots 11-30), ``--t-stok-dir``, ``--t-stok-shuffle``, ``--freeze``, ``--v10-md5`` in ``stack/tanitad/models/v8f_mini.py``, ``stack/tanitad/data/v8f_mini_data.py`` and
``stack/scripts/v8f_mini_train.py``. Every flag defaults to the earlier behaviour (bit for bit). No layer reads a state planned by another layer: S, T and O read the measured ego state; T's goals and constraints reach O through ``e_code``
as built. The PI's rule that binds every edge: (1) a test that changing the upper output changes the lower one, (2) a control that cutting the edge goes red (``mutate_otband.py`` on a COPY of the tree), (3) with the code detached
(``--detach-upper``, named in every arm's argv), a test that the lower level's loss puts EXACTLY 0.0 gradient on the upper level's parameters, paired with a mutation that removes the detach.

  1. OFF == TIP: a frozen snapshot of the ladder commit's model module (sha256 pinned) and the new one, same seed: state dict, tensors, loss terms, total, forward_infer, gradients -- bit for bit; no new key anywhere.
  2. Each band term moves the total by exactly its weight; the band metrics average to the 30-knot ADE and are pure in their band.
  3. Edges: T's code -> O's knots (changing it, zeroing it), strategic token -> T -> code -> O, the shuffle control, the token is read at S's tick (sample-and-hold).
  4. Purity under --detach-upper: O's terms put exactly 0.0 gradient on S and T; without it they reach T (the earlier behaviour).
  5. The launch gate refuses a lever that is declared and builds nothing, and a thing built that was not declared.
  6. Eval-only top-K decode; end to end on the trainer's synthetic world: checkpoint round trip, --init-ckpt, freeze identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import types
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
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
SNAP = Path(__file__).with_name("v8f_mini_tip_snapshot_otband.txt")
SNAP_SHA = "a6546109afa3297216b44431f9d69a2b3e4d355a54917e6e5f8ca50710e5108d"      # sha256 of the ladder commit's (1faa1a06) stack/tanitad/models/v8f_mini.py
RUN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
BAND = dict(w_traj=0.0, w_op=1.0 / 3.0, w_tac_traj=2.0 / 3.0)
BAND_ARGV = ["--w-traj", "0", "--w-op", "0.3333333", "--w-tac-traj", "0.6666667", "--detach-upper"]
STOK_D = 6
UPPER = ("read_s.", "head_s.", "read_t.", "trunk_t.", "h_goal.", "h_lat.", "h_lon.", "h_cons.", "e_stok.")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("otworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def write_stok(root, clips, T=60, dim=STOK_D, seed=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / D.STOK_META).write_text(json.dumps({"stok_dim": dim, "keys": [f"k{i}" for i in range(dim)]}), encoding="utf-8")
    rng = np.random.RandomState(seed)
    for c in clips:
        np.savez(root / f"{c}.npz", stok=rng.randn(T, dim).astype(np.float32), ok=np.ones(T, bool))
    return root


def make_batch(world, n=8, seed=0, hmax=10, stok=None, ages=(1, 3)):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax, stok=stok)
    idx = np.random.default_rng(seed).choice(len(wins), n, replace=False)
    b = D.collate([ds.item(wins[i][0], wins[i][1], *ages) for i in idx])
    b["gt_ok"] = torch.ones_like(b["gt_ok"])
    return b, J


def build(J, mod=M, seed=0, wake=True, **kw):
    cfg = mod.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)
    if wake:                                                   # the zero-init head hides every edge below it: wake it so a change at an input can be seen at the output
        with torch.no_grad():
            m.head_o[2].weight.copy_(torch.randn(m.head_o[2].weight.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    return m


def tip_module():
    name = "v8f_mini_tip_snapshot_otband"
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = str(SNAP)
    sys.modules[name] = mod
    exec(compile(SNAP.read_text(encoding="utf-8"), str(SNAP), "exec"), mod.__dict__)
    return mod


def run_loss(m, b, seed=3, p_gt=0.5):
    torch.manual_seed(seed)
    return m.loss(b, p_gt=p_gt, gen=torch.Generator().manual_seed(seed))


def gsum(term, params):
    gr = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return float(sum(float(x.abs().sum()) for x in gr if x is not None))


def named(m, *prefixes):
    return [p for n, p in m.named_parameters() if n.startswith(prefixes)]


# ------------------------------------------------------------------------------------------------ 1. off == the tip
def test_the_snapshot_is_the_tip_module():
    assert hashlib.sha256(SNAP.read_bytes()).hexdigest() == SNAP_SHA


@pytest.mark.parametrize("wm", ["off", "b"])
def test_off_is_the_tip_bit_for_bit(world, wm):
    tip = tip_module()
    b, J = make_batch(world, hmax=16)
    mt, mn = build(J, tip, wm=wm, hmax=16), build(J, M, wm=wm, hmax=16)
    assert list(mt.state_dict()) == list(mn.state_dict())
    for k, v in mt.state_dict().items():
        assert torch.equal(v, mn.state_dict()[k]), k
    ot, on = run_loss(mt, b), run_loss(mn, b)
    assert set(ot["terms"]) == set(on["terms"]), "no new T key with the lever off"
    for k in ot["terms"]:
        assert torch.equal(ot["terms"][k], on["terms"][k]), k
    ot["terms"]["total"].backward()
    on["terms"]["total"].backward()
    for (n, p), (_, q) in zip(mt.named_parameters(), mn.named_parameters()):
        assert (p.grad is None) == (q.grad is None) and (p.grad is None or torch.equal(p.grad, q.grad)), n
    mt.eval(), mn.eval()
    with torch.no_grad():
        it, inn = mt.forward_infer(b), mn.forward_infer(b)
    assert set(it) == set(inn) and all(torch.equal(it[k], inn[k]) for k in it if torch.is_tensor(it[k]))
    assert set(mt.built_levers()) == set(mn.built_levers())


def test_off_model_has_no_new_module_or_group(world):
    _, J = make_batch(world)
    m = build(J)
    assert m.e_stok is None and not any(k.startswith(M.NEW_LEVER_KEY_PREFIXES) for k in m.state_dict()) and "STOK" not in m.trainable_groups()


# ------------------------------------------------------------------------------------------------ 2. the band terms
@pytest.mark.parametrize("w", ["w_op", "w_tac_traj"])
def test_each_band_term_moves_the_total_by_exactly_its_weight(world, w):
    b, J = make_batch(world)
    key = {"w_op": "op", "w_tac_traj": "tac"}[w]
    other = "w_tac_traj" if w == "w_op" else "w_op"
    m = build(J, w_traj=0.0, **{other: 1.0})                      # the other band term on: both are computed
    t0 = float(run_loss(m, b)["terms"]["total"].detach())
    for wv in (0.5, 2.0):
        setattr(m.cfg, w, wv)
        o = run_loss(m, b)
        term = float(o["terms"][key].detach())
        assert term > 0.0 and math.isclose(float(o["terms"]["total"].detach()) - t0, wv * term, rel_tol=1e-4, abs_tol=1e-4), (w, wv)
        setattr(m.cfg, w, 0.0)


def test_with_a_band_weight_the_band_terms_are_logged_and_without_one_they_are_not(world):
    b, J = make_batch(world)
    on = run_loss(build(J, **BAND), b)["terms"]
    assert {"ade_op", "fde_op", "ade_tac", "fde_tac", "op", "tac"} <= set(on)
    assert not {"ade_op", "op", "tac"} & set(run_loss(build(J), b)["terms"])


def test_band_metrics_average_to_the_30_knot_ade_and_are_pure_in_their_band(world):
    b, J = make_batch(world)
    o = build(J).forward(b, p_gt=0.0)
    gt, gok = b["gt"], b["gt_ok"].float()
    bm = M.band_metrics(o["traj30"], gt, gok)
    ade, fde = M.traj_terms(o["traj30"], gt, gok)
    assert float(bm["ade_op"]) * 10 / 30 + float(bm["ade_tac"]) * 20 / 30 == pytest.approx(float(ade), rel=1e-5)
    assert float(bm["fde_tac"]) == pytest.approx(float(fde), rel=1e-6)
    t2 = o["traj30"].clone()
    t2[:, 10:] += 5.0                                                     # moving the tactical knots leaves the operative metrics untouched, and the reverse
    bm2 = M.band_metrics(t2, gt, gok)
    assert torch.equal(bm2["ade_op"], bm["ade_op"]) and torch.equal(bm2["fde_op"], bm["fde_op"]) and float(bm2["ade_tac"]) != float(bm["ade_tac"])
    t3 = o["traj30"].clone()
    t3[:, :10] += 5.0
    bm3 = M.band_metrics(t3, gt, gok)
    assert torch.equal(bm3["ade_tac"], bm["ade_tac"]) and torch.equal(bm3["fde_tac"], bm["fde_tac"]) and float(bm3["ade_op"]) != float(bm["ade_op"])


def test_the_band_terms_gradient_reaches_only_their_knots(world):
    b, J = make_batch(world)
    o = build(J, **BAND).forward(b, p_gt=0.0)
    t = M.band_metrics(o["traj30"], b["gt"], b["gt_ok"].float())
    g = torch.autograd.grad(t["ade_op"], o["traj30"], retain_graph=True)[0]
    assert float(g[:, 10:].abs().sum()) == 0.0 and float(g[:, :10].abs().sum()) > 0.0
    g = torch.autograd.grad(t["ade_tac"], o["traj30"])[0]
    assert float(g[:, :10].abs().sum()) == 0.0 and float(g[:, 10:].abs().sum()) > 0.0


def test_gt_validity_mask_is_respected_by_the_band_terms(world):
    b, J = make_batch(world)
    o = build(J).forward(b, p_gt=0.0)
    assert all(float(v) == 0.0 for v in M.band_metrics(o["traj30"], b["gt"], torch.zeros(len(b["gt"]))).values())


# ------------------------------------------------------------------------------------------------ 3. the edges
def test_changing_t_code_changes_o_knots_in_both_bands(world):
    b, J = make_batch(world)
    m = build(J)
    m.eval()
    with torch.no_grad():
        base = m.forward_infer(b)
        alt = m.forward_diag(b, base["code"] + 0.5)
    assert float((base["traj30"][:, :10] - alt["traj30"][:, :10]).abs().max()) > 0.0 and float((base["traj30"][:, 10:] - alt["traj30"][:, 10:]).abs().max()) > 0.0


def test_zeroing_the_code_changes_o_knots(world):
    b, J = make_batch(world)
    m = build(J)
    m.eval()
    with torch.no_grad():
        base = m.forward_infer(b)
        zero = m.forward_diag(b, torch.zeros_like(base["code"]))
    assert float((base["traj30"] - zero["traj30"]).abs().max()) > 0.0, "zeroing the code must change O's plan (the T -> O wire through e_code is live)"


def test_stok_reaches_t_output_code_and_o_and_the_shuffle_changes_it(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    st, st_sh = D.StokStore(root, world["names"]["train"]), D.StokStore(root, world["names"]["train"], shuffle_seed=3)
    b, J = make_batch(world, stok=st)
    b_sh, _ = make_batch(world, stok=st_sh)
    assert b["stok"].shape[-1] == STOK_D and not torch.equal(b["stok"], b_sh["stok"])
    m = build(J, stok_dim=STOK_D)
    m.eval()
    with torch.no_grad():
        a, c = m.forward_infer(b), m.forward_infer(b_sh)
        z = m.forward_infer({**b, "stok": torch.zeros_like(b["stok"]), "stok_ok": torch.zeros_like(b["stok_ok"])})
    assert float((a["goal_logits"] - c["goal_logits"]).abs().max()) > 0.0, "the shuffled token changes T's output"
    assert float((a["code"] - c["code"]).abs().max()) > 0.0, "T's code carries the token down"
    assert float((a["traj30"] - c["traj30"]).abs().max()) > 0.0, "and the code carries it to O's plan"
    assert float((a["goal_logits"] - z["goal_logits"]).abs().max()) > 0.0
    assert M.STOK_KEYS <= m.forward_infer(b)["_keys_read"]
    assert M.assert_no_situation_classifier_input(m, b)["read"]


def test_a_stok_model_refuses_a_batch_without_the_token(world):
    b, J = make_batch(world)
    m = build(J, stok_dim=STOK_D)
    with pytest.raises(KeyError):
        m.forward_infer(b)


def test_without_the_token_the_model_never_reads_it(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    b, J = make_batch(world, stok=D.StokStore(root, world["names"]["train"]))
    m = build(J)
    m.eval()
    with torch.no_grad():
        out = m.forward_infer(b)
    assert not (M.STOK_KEYS & out["_keys_read"])


def test_the_token_is_read_at_s_tick_frame_sample_and_hold(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    b, J = make_batch(world, stok=D.StokStore(root, world["names"]["train"]), ages=(1, 3))
    m = build(J, stok_dim=STOK_D)
    m.eval()
    last_s = (b["feats"].shape[1] - 1) - b["a_s"]
    other = torch.ones_like(b["stok"][:, :, 0], dtype=torch.bool)
    for r in range(len(last_s)):
        other[r, last_s[r]] = False
    with torch.no_grad():
        base = m.forward_infer(b)
        s2 = b["stok"].clone()
        s2[other] += 3.0                                                                 # change every frame EXCEPT S's tick frame
        same = m.forward_infer({**b, "stok": s2})
        s3 = b["stok"].clone()
        s3[~other] += 3.0                                                                # change S's tick frame only
        diff = m.forward_infer({**b, "stok": s3})
    assert torch.equal(base["goal_logits"], same["goal_logits"]) and float((base["goal_logits"] - diff["goal_logits"]).abs().max()) > 0.0


def test_stokstore_shuffle_is_a_seeded_derangement_and_the_dimension_is_the_files(world, tmp_path):
    clips = world["names"]["train"]
    root = write_stok(tmp_path / "stok", clips, dim=9)
    assert D.stok_dim_of(root) == 9
    s0, s1, s1b = D.StokStore(root, clips), D.StokStore(root, clips, 1), D.StokStore(root, clips, 1)
    assert s0.built_levers()["n_self_sourced"] == len(clips) and s1.built_levers()["n_self_sourced"] == 0
    assert s1.source == s1b.source and D.StokStore(root, clips, 2).source != s1.source and sorted(s1.source.values()) == sorted(clips)
    idx = np.arange(5, 15)
    v0, _ = s0.get(clips[0], idx)
    v1, _ = s1.get(clips[0], idx)
    assert v0.shape == (10, 9) and not np.array_equal(v0, v1)
    with pytest.raises(FileNotFoundError):
        D.StokStore(root, clips + ["no-such-clip"])
    with pytest.raises(ValueError):
        D.stok_dim_of(tmp_path)


def test_attach_stok_puts_the_key_in_the_batch_for_an_eval_tool(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=8)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    assert "stok" not in D.collate([ds.item(*wins[0])])
    D.attach_stok(ds, root, shuffle_seed=4)
    bb = D.collate([ds.item(*wins[0])])
    assert bb["stok"].shape == (1, 10, STOK_D) and bb["stok_ok"].dtype == torch.bool


# ------------------------------------------------------------------------------------------------ 4. purity under --detach-upper
@pytest.mark.parametrize("key", ["op", "tac"])
def test_with_detach_upper_o_terms_put_exactly_zero_gradient_on_s_and_t(world, key):
    b, J = make_batch(world)
    m = build(J, detach_upper=True, **BAND, stok_dim=STOK_D)
    b = {**b, "stok": torch.randn(len(b["gt"]), 10, STOK_D), "stok_ok": torch.ones(len(b["gt"]), 10, dtype=torch.bool)}
    o = run_loss(m, b)
    assert gsum(o["terms"][key], named(m, *UPPER)) == 0.0, "with the code detached, O's band term must not reach S, T or the token embedding"
    assert gsum(o["terms"][key], named(m, "head_o.")) > 0.0


def test_without_detach_upper_the_o_term_reaches_t_through_the_code_the_earlier_behaviour(world):
    b, J = make_batch(world)
    m = build(J, **BAND)
    o = run_loss(m, b)
    assert gsum(o["terms"]["op"], named(m, "read_t.", "trunk_t.", "h_goal.")) > 0.0


# ------------------------------------------------------------------------------------------------ 5. the launch gate
def ns(**kw):
    a = TR.build_parser().parse_args(["--arm", "M2b", "--out", "unused"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def built_for(world, **cfgkw):
    _, J = make_batch(world)
    return build(J, wake=False, **cfgkw)


def test_every_flag_has_a_declared_vs_built_entry():
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert {"w_op", "w_tac_traj", "t_stok_dir", "t_stok_shuffle", "freeze", "v10_md5"} <= dests and dests <= set(TR.FLAG_CHECKS)
    assert not {"o_split", "w_cont"} & dests, "no tactical head, no hand-over, no continuity term (PI ruling)"


def test_the_defaults_pass_bad_oband(world):
    assert TR.bad_oband(ns(), M.V8FMiniConfig(**TINY), built_for(world), {}) == []


@pytest.mark.parametrize("declared,cfgkw,match", [
    (dict(w_op=1.0, w_traj=0.0), {}, "cfg.w_op"),                                                              # declared weight, built weight 0
    (dict(w_tac_traj=1.0, w_traj=0.0), {}, "cfg.w_tac_traj"),
    (dict(w_op=1.0, w_traj=1.0), dict(w_op=1.0, w_traj=1.0), "twice"),                                         # the 30-knot term plus a band term
    (dict(t_stok_shuffle=3), {}, "shuffle"),                                                                   # a shuffle with no token
    (dict(freeze=["S"]), {}, "freeze"),                                                                        # declared frozen, built trainable
    (dict(v10_md5="0" * 32, v9_train_md5="0" * 32), {}, "ACCEPTED_V10"),                                     # an md5 nobody named
])
def test_the_gate_refuses_a_lever_declared_and_not_built(world, declared, cfgkw, match):
    bad = TR.bad_oband(ns(**declared), M.V8FMiniConfig(**TINY), built_for(world, **cfgkw), {})
    assert bad and any(match in x for x in bad), bad


def test_the_gate_refuses_a_token_the_dataset_does_not_hold_and_a_wrong_dimension(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    a = ns(t_stok_dir=str(root))
    m = built_for(world, stok_dim=STOK_D)
    cfg = M.V8FMiniConfig(**TINY)
    assert any("StokStore" in x for x in TR.bad_oband(a, cfg, m, {}))
    good = {"stok": {"stok_dim": STOK_D, "stok_shuffle": -1, "n_self_sourced": 6}, "stok_attached": True}
    assert TR.bad_oband(a, cfg, m, good) == []
    assert any("reaches nothing" in x for x in TR.bad_oband(a, cfg, m, {**good, "stok_attached": False}))
    assert any("stok_dim" in x for x in TR.bad_oband(a, cfg, built_for(world, stok_dim=STOK_D + 1), good))
    assert any("no --t-stok-dir" in x for x in TR.bad_oband(ns(), cfg, m, {}))


def test_a_shuffle_declared_must_be_a_derangement_in_the_built_store(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    a = ns(t_stok_dir=str(root), t_stok_shuffle=2)
    m, cfg = built_for(world, stok_dim=STOK_D), M.V8FMiniConfig(**TINY)
    info = {"stok": D.StokStore(root, world["names"]["train"], 2).built_levers(), "stok_attached": True}
    assert TR.bad_oband(a, cfg, m, info) == []
    assert any("derangement" in x for x in TR.bad_oband(a, cfg, m, {**info, "stok": {**info["stok"], "n_self_sourced": 2}}))
    assert any("shuffle" in x for x in TR.bad_oband(a, cfg, m, {**info, "stok": {**info["stok"], "stok_shuffle": -1}}))


def test_the_v10_release_is_accepted_by_md5_with_a_synthetic_release(world, monkeypatch):
    monkeypatch.setattr(D, "ACCEPTED_V10_TRAIN_MD5", dict(D.ACCEPTED_V10_TRAIN_MD5))
    md5 = hashlib.md5(b"a synthetic v10 release").hexdigest()
    assert D.v10_md5_problems(md5) and D.v10_md5_problems("") == []
    with pytest.raises(ValueError):
        D.register_v10_md5("nothex")
    D.register_v10_md5(md5, "synthetic")
    assert D.v10_md5_problems(md5) == []
    m, cfg = built_for(world), M.V8FMiniConfig(**TINY)
    assert TR.bad_oband(ns(v10_md5=md5, v9_train_md5=md5), cfg, m, {"v9_md5": md5}) == []
    assert any("loader hashed" in x for x in TR.bad_oband(ns(v10_md5=md5, v9_train_md5=md5), cfg, m, {"v9_md5": "f" * 32}))
    assert any("loader hashed" in x for x in TR.bad_oband(ns(v10_md5=md5, v9_train_md5=md5), cfg, m, {}))
    assert any("--v9-train-md5" in x for x in TR.bad_oband(ns(v10_md5=md5), cfg, m, {"v9_md5": md5}))


def test_init_ckpt_monotone_names_the_token_dimension():
    cfg = M.V8FMiniConfig(**TINY)
    assert set(TR.INIT_CKPT_MONOTONE) == {"stok_dim"}
    assert TR.monotone_diff({"stok_dim": 0}, M.V8FMiniConfig(**TINY, stok_dim=3)) == {} and TR.monotone_diff({}, cfg) == {}
    assert set(TR.monotone_diff({"stok_dim": 6}, cfg)) == {"stok_dim"} and set(TR.monotone_diff({"stok_dim": 6}, M.V8FMiniConfig(**TINY, stok_dim=7))) == {"stok_dim"}
    assert "e_stok." in M.NEW_LEVER_KEY_PREFIXES


def test_the_new_group_is_in_the_live_check_groups(world):
    _, J = make_batch(world)
    assert "STOK" in build(J, stok_dim=STOK_D).trainable_groups()
    assert TR.band_expect(M.V8FMiniConfig(**TINY)) == set() and {"op", "tac", "ade_op", "fde_tac"} <= TR.band_expect(M.V8FMiniConfig(**TINY, **BAND))


def test_freeze_groups_reads_back_from_the_built_model(world):
    _, J = make_batch(world)
    m = build(J)
    assert TR.frozen_groups(m) == []
    TR.freeze_groups(m, ["S"])
    assert TR.frozen_groups(m) == ["S"] and all(p.requires_grad for p in m.read_t.parameters())
    TR.freeze_groups(m, ["T", "adapter", "cond"])
    assert set(TR.frozen_groups(m)) == {"S", "T", "adapter", "cond"}
    assert not any(p.requires_grad for p in (m.pos, m.time)) and all(p.requires_grad for p in m.read_o.parameters())


def test_the_probe_is_green_with_detach_and_token_and_red_when_the_code_wire_is_cut(world, tmp_path):
    root = write_stok(tmp_path / "stok", world["names"]["train"])
    b, J = make_batch(world, stok=D.StokStore(root, world["names"]["train"]))
    m = build(J, wake=False, detach_upper=True, **BAND, stok_dim=STOK_D)
    rec = TR.oband_probe(m, b, m.cfg)
    assert rec["pass"] and rec["purity"]["upper_grad_abs_sum"] == 0.0 and rec["code_edge_max_dknot"] > 0.0 and rec["token_edge"]["max_dlogit"] > 0.0
    assert set(rec["contribution"]) == {"w_op", "w_tac_traj"}
    m.e_code.forward = lambda x: torch.zeros(x.shape[0], m.cfg.d)                      # the wire is cut: e_code returns zeros whatever the code is
    with pytest.raises(AssertionError, match="edge is cut"):
        TR.oband_probe(m, b, m.cfg)


def test_the_off_probe_is_a_noop(world):
    b, J = make_batch(world)
    assert TR.oband_probe(build(J), b, M.V8FMiniConfig(**TINY))["otband"] == "off"


# ------------------------------------------------------------------------------------------------ 6. eval-only top-K and end to end
def test_topk_decodes_o_once_per_tactical_code_and_has_no_parameter(world):
    b, J = make_batch(world)
    m = build(J)
    n_before = sum(p.numel() for p in m.parameters())
    m.eval()
    with torch.no_grad():
        o = m.forward_topk(b, K=3)
        base = m.forward_infer(b)
    B = len(b["gt"])
    assert o["traj30"].shape == (B, 3, 30, 2) and o["prob"].shape == (B, 3) and torch.equal(o["traj30_top1"], base["traj30"])
    assert bool((o["prob"][:, :-1] >= o["prob"][:, 1:]).all()) and sum(p.numel() for p in m.parameters()) == n_before
    assert float((o["traj30"][:, 0] - o["traj30"][:, 1]).abs().max()) > 0.0, "different manoeuvre codes give different plans"
    codes, _ = M.topk_codes(base["code"], 2)
    assert torch.equal(codes[0][:, :M.N_GOAL], base["code"][:, :M.N_GOAL]) and torch.equal(codes[0][:, -M.N_CONS:], base["code"][:, -M.N_CONS:])
    assert bool((codes[0][:, M.N_GOAL:M.N_GOAL + M.N_LAT].sum(1) == 1).all())


def test_topk_codes_follow_the_joint_rank():
    a, c = M.N_GOAL, M.N_GOAL + M.N_LAT
    code = torch.zeros(1, M.CODE_DIM)
    code[0, a:c] = torch.tensor([0.3, 0.4, 0.1, 0.1, 0.1])[:M.N_LAT]
    code[0, c:c + M.N_LON] = torch.tensor([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])[:M.N_LON]
    outs, pr = M.topk_codes(code, 3)
    lat = [int(o[0, a:c].argmax()) for o in outs]
    lon = [int(o[0, c:c + M.N_LON].argmax()) for o in outs]
    assert (lat[0], lon[0]) == (1, 0) and float(pr[0, 0]) == pytest.approx(0.36)
    assert len({(x, y) for x, y in zip(lat, lon)}) == 3 and bool((pr[:, :-1] >= pr[:, 1:]).all())
    assert (lat[1], lon[1]) == (0, 0), "rank 2 is the other lat with the same lon"
    assert (lat[1], lon[1]) != (lat[0], lon[0]) and (lat[2], lon[2]) != (lat[0], lon[0])


def test_hacc_post_map_and_the_prior_path_work_with_the_band_loss(world):
    b, J = make_batch(world)
    m = build(J, **BAND)
    m.eval()
    with torch.no_grad():
        raw = m.forward_infer(b)
        m.set_hacc_weights(np.full((30, 2), 0.5))
        post = m.forward_infer(b)
    assert torch.allclose(post["traj60"], 0.5 * raw["traj60"] + 0.5 * M.upsample_prior(raw["p10"]), atol=1e-5) and torch.equal(post["traj30_o"], raw["traj30"])


def _sc(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def _lg(out):
    return [json.loads(x) for x in (Path(out) / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def refused(argv, tmp_path, match):
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "refused"), *RUN, *argv])
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and match in str(e.value), str(e.value)[:600]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ottrain")
    names = D.make_synthetic_world(root / "w", n_train=6, n_eval=2, T=60, seed=5)["names"]["train"]
    stok = write_stok(root / "stok", names)
    off = root / "off"
    assert TR.main(["--out", str(off), *RUN, "--steps", "2"]) == 0
    ck = str(off / "ckpt_last.pt")
    band = root / "band"
    assert TR.main(["--out", str(band), *RUN, "--steps", "3", *BAND_ARGV, "--freeze", "S", "--init-ckpt", ck, "--lr", "1e-3"]) == 0
    tok = root / "tok"
    assert TR.main(["--out", str(tok), *RUN, "--steps", "3", *BAND_ARGV, "--freeze", "S", "--t-stok-dir", str(stok), "--init-ckpt", ck, "--lr", "1e-3"]) == 0
    fro = root / "fro"
    assert TR.main(["--out", str(fro), *RUN, "--steps", "3", *BAND_ARGV, "--freeze", "S", "T", "adapter", "cond", "--init-ckpt", ck, "--lr", "1e-3"]) == 0
    shuf = root / "shuf"
    assert TR.main(["--out", str(shuf), *RUN, "--steps", "2", *BAND_ARGV, "--t-stok-dir", str(stok), "--t-stok-shuffle", "7"]) == 0
    return types.SimpleNamespace(root=root, off=off, band=band, tok=tok, fro=fro, shuf=shuf, stok=stok)


def test_init_from_an_unbanded_checkpoint_loads_every_old_key_and_only_the_token_keys_are_fresh(trained):
    old = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    rb = _sc(trained.band)["levers"]["init_ckpt"]
    assert rb["applied"] and rb["n_mismatch_after_load"] == 0 and not rb.get("fresh_keys"), "a band-loss arm adds no parameter"
    rt = _sc(trained.tok)["levers"]["init_ckpt"]
    assert {k.split(".")[0] for k in rt["fresh_keys"]} == {"e_stok"} and set(rt["fresh_keys"]) == set(rt["declared_new_keys"])
    new = torch.load(trained.tok / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    assert set(old) | set(rt["fresh_keys"]) == set(new)


def test_with_s_t_adapter_and_cond_frozen_t_and_s_are_the_init_checkpoints_bit_for_bit_and_o_moves(trained):
    old = torch.load(trained.off / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    new = torch.load(trained.fro / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    held = ("read_s.", "head_s.", "read_t.", "trunk_t.", "h_goal.", "h_lat.", "h_lon.", "h_cons.", "adapter.", "pos", "time", "e_ego.", "e_nav.", "e_lim.", "e_age.", "e_ck.", "e_code.")
    keys = [k for k in old if k.startswith(held)]
    assert keys and all(torch.equal(old[k], new[k]) for k in keys), [k for k in keys if not torch.equal(old[k], new[k])][:4]
    assert any(not torch.equal(old[k], new[k]) for k in old if k.startswith(("read_o.", "head_o.")))
    assert _sc(trained.fro)["levers"]["init_ckpt"]["n_mismatch_after_load"] == 0


def test_the_log_carries_the_per_band_metrics_and_the_checkpoint_round_trips(trained):
    for r in _lg(trained.band):
        assert all(math.isfinite(r[k]) and r[k] > 0.0 for k in ("ade_op", "fde_op", "ade_tac", "fde_tac", "op", "tac"))
    ck = torch.load(trained.tok / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["w_op"] > 0.0 and ck["cfg"]["stok_dim"] == STOK_D and ck["cfg"]["detach_upper"] is True
    cfg = M.V8FMiniConfig(**ck["cfg"])                                                 # G-EVAL: the evaluators build V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior.synthetic(5), ck["cons_mu"], ck["cons_sd"])
    model.load_state_dict(ck["model"], strict=True)
    sc = _sc(trained.tok)
    assert model.e_stok is not None and sc["otband_probe"]["pass"] and sc["stok"]["stok_dim"] == STOK_D and sc["otband_probe"]["purity"]["upper_grad_abs_sum"] == 0.0


def test_the_shuffle_control_arm_built_a_derangement_and_flag_off_records_keep_their_key_sets(trained):
    sc = _sc(trained.shuf)
    assert sc["stok"]["stok_shuffle"] == 7 and sc["stok"]["n_self_sourced"] == 0
    off = _sc(trained.off)
    assert "otband_probe" not in off and "stok" not in off and "stok_attached" not in off["info"] and "ade_op" not in _lg(trained.off)[0]


def test_a_token_checkpoint_into_a_build_without_it_is_refused(trained, tmp_path):
    refused(["--init-ckpt", str(trained.tok / "ckpt_last.pt")], tmp_path, "stok_dim")


def test_the_launch_refuses_through_main(tmp_path, trained):
    refused(["--w-op", "1.0"], tmp_path, "twice")
    refused(["--t-stok-shuffle", "3"], tmp_path, "shuffle")
    refused(["--t-stok-dir", str(trained.stok), "--wm-only"], tmp_path, "wm")
    refused(["--v10-md5", "0" * 32, "--v9-train-md5", "0" * 32], tmp_path, "ACCEPTED_V10")
