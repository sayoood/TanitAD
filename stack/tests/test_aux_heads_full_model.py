"""refcv9 AUX-FULL -- the TRAINING-ONLY aux heads (B1 depth / B2 persp / B2c bev2) trained INSIDE THE FULL MODEL, dormant.

THE CLASS ("built, tested, unreachable from its caller", CLAUDE.md): `box_depth_aux_flag.pin_args` / `persp3d_aux_flag.pin_args` refused the heads
without `--r9-perc-rig`, and `loss_term` was called ONLY inside `compute_losses_perc_rig` -- so if E1 crowned a head, E2 / E3 (the full model) could
not train it. This file proves the path, hop by hop, on the tiny FULL model (`test_map_only_ft._build()`: trunk + s8 tap + 10 cm map + the pooled-BEV
perception branch; NO rig):

  LIVE       full model + each flag on -> the term is in the loss dict, finite, > 0, and EVERY head parameter gets a NON-ZERO gradient from `loss`;
             CONTROL: the planner's own term has EXACTLY 0.0 gradient on the head, a head whose term is dropped reads EXACTLY 0.0, weight 0 is refused
  EQUALITY   the term the full model logs == the term the perc-rig logs (same keys; same function on the same inputs is `torch.equal`)
  OFF        flags off -> the full model's loss dict and EVERY gradient are bit-identical to the loss function without the two calls (torch.equal);
             the tip-vs-hold form of this check is the cross-tree run recorded in the hold's evidence/
  PIN / GATE pin_args admits the full model when every input is present and still refuses each missing one; REFC_WEIGHT_GATES do not name the rig
  WARM START the full model's `--init-from` refuses a head the refcv8 source cannot carry; `_AuxHeadsHeldOut` is the (tested) way around it
  MUTATION   `test_red_arm_*` below; the real one (the two calls deleted from compute_losses_v3) is recorded in the hold's HANDOFF.md
"""
from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_box_depth_aux as C3T  # noqa: E402   (the C3 helpers: agent block, unzero-BN, the per-item target build)
import test_persp3d_aux as C4T  # noqa: E402   (the C4 helpers)
import test_map_only_ft as R  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import box_depth_aux_flag as BF  # noqa: E402
from tanitad.train import persp3d_aux_flag as PF  # noqa: E402
from tanitad.train import refcv8_train as r8train  # noqa: E402

TRAINER_SRC = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")

#: kind -> (flag module, loss key, weight attribute, head attribute, state_dict prefix, weight dest on the namespace, flag dest)
KINDS = {
    "depth": (BF, "box_depth", BF.WEIGHT_ATTR, BF.HEAD_ATTR, BF.KEY_PREFIX, "w_box_depth", "r9_box_depth_aux"),
    "persp": (PF, "box_persp", PF.WEIGHT_ATTR_P, PF.HEAD_P, PF.KEY_PREFIX_P, "w_box_persp", "r9_box_persp_aux"),
    "bev2": (PF, "box_bev2", PF.WEIGHT_ATTR_B, PF.HEAD_B, PF.KEY_PREFIX_B, "w_box_bev2", "r9_box_bev2_aux"),
}
ALL_KINDS = tuple(KINDS)
_ORIG = C3T._ORIG_MODEL


def _ns(*kinds, w=0.1, **over):
    a = dict(r9_box_depth_aux=False, w_box_depth=0.0, r9_box_persp_aux=False, w_box_persp=0.0, r9_box_bev2_aux=False, w_box_bev2=0.0)
    for k in kinds:
        a[KINDS[k][6]], a[KINDS[k][5]] = True, w
    a.update(over)
    return argparse.Namespace(**a)


def _attach(model, args):
    """What train() does after the perception branch is built (both calls, in the trainer's order)."""
    model._rig_camera = R.RigCamera.nominal(model._lift_bank_hires.frame, height_m=1.5, x_m=1.5)
    BF.attach(model, args)
    PF.attach(model, args)


def _full(*kinds, w=0.1):
    """The tiny FULL model (no rig) with the heads attached; (model, cfg, args)."""
    model, cfg = R._build()
    assert not getattr(model.core, "_perc_rig", False) and hasattr(model.core, "decoder")
    args = _ns(*kinds, w=w)
    _attach(model, args)
    return model, cfg, args


def _rig(monkeypatch, *kinds, w=0.1):
    """The tiny perception RIG (the path the heads already ran in), same seed / build recipe / heads."""
    with monkeypatch.context() as mp:
        mp.setattr(v3, "RefCV3Model", lambda cfg: _ORIG(cfg, perc_rig=True))
        model, cfg = R._build()
    assert getattr(model.core, "_perc_rig", False)
    args = _ns(*kinds, w=w)
    _attach(model, args)
    return model, cfg, args


def _batch(cfg, model, *kinds):
    """R._batch (the full model's batch) + two scenes' cuboids + the per-item worker-built targets of every head in play."""
    batch = R._batch(cfg)
    rows = [[(7.0, 0.0, 1.5, 4.5, 1.9, 1.5, 0.3)], [(6.0, 0.5, 1.2, 4.0, 1.8, 1.6, -0.4), (12.0, -3.0, 1.0, 4.5, 2.0, 1.5, 1.2)]]
    batch.update(C3T._agent_block(rows))
    batch["agent_ep"] = batch["map_ep"].clone()
    for k in kinds:
        batch = C3T._with_targets(batch, model) if k == "depth" else C4T._with_targets(batch, model, k)
    return batch


def _head(model, kind):
    return getattr(model._perception, KINDS[kind][3], None)


def _loss(model, batch, seed=5):
    model.train()
    torch.manual_seed(seed)
    return T.compute_losses_v3(model, batch, "cpu", log_metrics=False)


def _is_zero(g):
    return g is None or float(g.abs().max()) == 0.0


def _grads(loss, params):
    return torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)


def _tree_equal(a, b) -> bool:
    return C3T._tree_equal(a, b)


# ===================================================================================================================== #
# LIVE                                                                                                                  #
# ===================================================================================================================== #
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_live_full_model_term_is_in_the_loss_and_every_head_parameter_gets_a_nonzero_gradient(kind):
    mod, key, wattr, hattr, pfx, wdest, _ = KINDS[kind]
    model, cfg, args = _full(kind)
    head = _head(model, kind)
    assert head is not None and float(getattr(model, wattr)) == 0.1
    C3T._unzero_bn(model)
    out = _loss(model, _batch(cfg, model, kind))
    assert "traj" in out and "cls" in out, "this is the FULL model's loss dict (planner terms present), not the perception rig's"
    assert key in out and math.isfinite(float(out[key])) and float(out[key]) > 0.0
    params = list(head.parameters())
    gs = _grads(out["loss"], params)                                                      # the TOTAL loss reaches every head tensor
    assert len(params) >= 12 and all(not _is_zero(g) for g in gs), [n for (n, _), g in zip(head.named_parameters(), gs) if _is_zero(g)]
    # the head's own G-DVB checks agree on the full model (declared == built, weight == the attribute the loss multiplies by)
    checks = {"depth": (BF.dvb_check, BF.dvb_check_weight), "persp": (PF.dvb_check_persp, PF.dvb_check_weight_persp),
              "bev2": (PF.dvb_check_bev2, PF.dvb_check_weight_bev2)}[kind]
    assert checks[0](model, args) == [] and checks[1](model, args) == []
    # the trunk's s8 stage is trained BY the term in the full model too (s8 -> layer2): a dropped hop upstream of the head would read 0 here
    gd = _grads(float(getattr(model, wattr)) * out[key], [p for n, p in model.named_parameters() if n.startswith("core.encoder.net.layer2.")])
    assert any(not _is_zero(g) for g in gd)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_control_the_planner_term_has_exactly_zero_gradient_on_the_head_and_a_dropped_term_reads_exactly_zero(kind, monkeypatch):
    mod, key, wattr, hattr, pfx, wdest, _ = KINDS[kind]
    model, cfg, _ = _full(kind)
    batch = _batch(cfg, model, kind)
    out = _loss(model, batch)
    params = list(_head(model, kind).parameters())
    assert all(_is_zero(g) for g in _grads(out["traj"], params))                           # the zero control: the planner never touches the head ...
    assert any(not _is_zero(g) for g in _grads(out[key], params))                          # ... and the same probe reads non-zero where it must
    # the head is BUILT but its term is dropped from the loss (the mutant): every head parameter reads exactly 0.0 (None = no graph)
    monkeypatch.setattr(mod, "loss_term", lambda *a, **k: None)
    out2 = _loss(model, batch)
    assert key not in out2
    assert all(_is_zero(g) for g in _grads(out2["loss"], params))


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_control_weight_zero_is_refused_by_pin_args_and_an_orphan_weight_too(kind):
    mod, key, wattr, hattr, pfx, wdest, fdest = KINDS[kind]
    full_argv = dict(r9_perc_rig=False, w_box3d=1.0, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics")
    mod.pin_args(_ns(kind, **full_argv))                                                   # the complete full-model argv is admitted
    with pytest.raises(SystemExit, match="positive weight"):
        mod.pin_args(_ns(kind, w=0.0, **full_argv))                                        # weight 0 with the flag: refused
    with pytest.raises(SystemExit, match="dead flag"):
        mod.pin_args(_ns(**{wdest: 0.2}, **full_argv))                                     # a weight with no flag: refused


# ===================================================================================================================== #
# EQUALITY with the perc-rig path                                                                                       #
# ===================================================================================================================== #
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_equality_the_full_models_term_is_the_rigs_term_same_keys_same_function(kind, monkeypatch):
    mod, key, wattr, hattr, pfx, wdest, _ = KINDS[kind]
    full, cfg, _a = _full(kind)
    rigm, rcfg, _r = _rig(monkeypatch, kind)
    sd = {k: v for k, v in full.state_dict().items() if not rig.is_skipped_key(k)}
    rigm.load_state_dict(sd, strict=True)                                                   # one weight set, head included
    batch = _batch(cfg, full, kind)
    spy = {}
    real = mod.loss_term

    def tap(model, out, b, device, extra):
        spy["out"] = out
        return real(model, out, b, device, extra)
    monkeypatch.setattr(mod, "loss_term", tap)
    fo = _loss(full, batch)
    monkeypatch.setattr(mod, "loss_term", real)
    ro = _loss(rigm, batch)
    # (1) the SAME logged keys, with the same finite values to float tolerance (the two forwards batch the trunk differently)
    aux = lambda d: {k for k in d if k == key or k.startswith(key + "_")}  # noqa: E731
    assert aux(fo) == aux(ro) and key in aux(fo)
    assert float(fo[key]) == pytest.approx(float(ro[key]), rel=1e-4, abs=1e-6)
    # (2) the SAME function: feeding the full model's own `out` to the rig model's head (identical weights) is EXACTLY the full model's term
    extra = {}
    term = mod.loss_term(rigm, spy["out"], batch, "cpu", extra)
    assert torch.equal(extra[key].detach(), fo[key].detach())
    assert torch.equal(term.detach(), float(getattr(full, wattr)) * fo[key].detach())          # ... times the SAME weight attribute, in float32


def test_the_full_loss_is_the_off_loss_plus_the_weighted_terms_in_the_rigs_order():
    on, cfg, _ = _full("depth", "persp")                                                    # B1 + B2 may coexist (pin_args forbids only B2 + B2c)
    off, _, _ = _full()
    batch = _batch(cfg, on, "depth", "persp")
    lo, lf = _loss(on, batch), _loss(off, batch)
    assert float(lo["loss"]) == pytest.approx(float(lf["loss"]) + 0.1 * float(lo["box_depth"]) + 0.1 * float(lo["box_persp"]), rel=1e-6, abs=1e-7)
    for k, v in lf.items():
        if k != "loss" and torch.is_tensor(v) and v.numel() == 1:
            assert float(lo[k]) == float(v), k                                              # every other loss scalar is bit-identical


# ===================================================================================================================== #
# OFF == the loss function without the calls                                                                            #
# ===================================================================================================================== #
def test_off_flags_off_the_full_loss_dict_and_every_gradient_are_bit_identical_to_the_function_without_the_calls(monkeypatch):
    m1, cfg, _ = _full()
    assert not any(".box_depth_aux." in k or ".box_persp_aux." in k or ".box_bev2_aux." in k for k in m1.state_dict())     # OFF builds nothing
    m2, _, _ = _full()
    batch = _batch(cfg, m1)
    calls = []
    for mod in (BF, PF):
        real = mod.loss_term
        monkeypatch.setattr(mod, "loss_term", (lambda r: lambda *a, **k: (calls.append(1), r(*a, **k))[1])(real))
    o1 = _loss(m1, batch)
    assert len(calls) == 2 and not any(k.startswith(("box_depth", "box_persp", "box_bev2")) for k in o1)      # both ran, both returned None
    o1["loss"].backward()
    for mod in (BF, PF):                                                                     # the TIP: the two calls do not exist
        monkeypatch.setattr(mod, "loss_term", lambda *a, **k: None)
    o2 = _loss(m2, batch)
    o2["loss"].backward()
    assert set(o1) == set(o2) and _tree_equal({k: v.detach() if torch.is_tensor(v) else v for k, v in o1.items()},
                                              {k: v.detach() if torch.is_tensor(v) else v for k, v in o2.items()})
    g1, g2 = dict(m1.named_parameters()), dict(m2.named_parameters())
    assert set(g1) == set(g2) and all(torch.equal(g1[n].grad, g2[n].grad) if g1[n].grad is not None else g2[n].grad is None for n in g1)
    assert sum(g1[n].grad is not None for n in g1) > 100                                   # the comparison covered the real gradient set, not an empty one


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_eval_mode_never_calls_the_head_in_the_full_model(kind):
    model, cfg, _ = _full(kind)
    batch = _batch(cfg, model, kind)
    model.eval()
    out = T.compute_losses_v3(model, batch, "cpu", log_metrics=False)
    assert not any(k.startswith(KINDS[kind][1]) for k in out)                                # training-only: the eval forward is the OFF forward


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_a_built_head_with_no_targets_in_the_batch_refuses_not_skips(kind):
    model, cfg, _ = _full(kind)
    batch = _batch(cfg, model)                                                               # the dataset hook did not run: no bd_* / p3_* / b2_*
    with pytest.raises(SystemExit, match="precomputed"):
        _loss(model, batch)


# ===================================================================================================================== #
# PERC-RIG UNCHANGED                                                                                                    #
# ===================================================================================================================== #
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_perc_rig_with_each_flag_on_is_the_off_rig_loss_plus_the_weighted_term(kind, monkeypatch):
    mod, key, wattr, hattr, pfx, wdest, _ = KINDS[kind]
    on, cfg, _ = _rig(monkeypatch, kind)
    off, _, _ = _rig(monkeypatch)
    batch = _batch(cfg, on, kind)
    lo, lf = _loss(on, batch), _loss(off, batch)
    assert "traj" not in lo and key in lo                                                    # the rig's dict: no planner term
    assert float(lo["loss"]) == pytest.approx(float(lf["loss"]) + 0.1 * float(lo[key]), rel=1e-6, abs=1e-7)
    for k, v in lf.items():
        if k != "loss" and torch.is_tensor(v) and v.numel() == 1:
            assert float(lo[k]) == float(v), k


def test_the_calls_sit_after_the_10cm_map_term_in_both_losses_and_in_the_same_order():
    k, i = TRAINER_SRC.index("def compute_losses_v3("), TRAINER_SRC.index("def compute_losses_perc_rig(")
    j = TRAINER_SRC.index("def _lan_arm_preflight(")
    for body in (TRAINER_SRC[k:i], TRAINER_SRC[i:j]):
        a = body.index("loss = loss + _mh_term")
        b = body.index("_bd_term = _bda.loss_term(model, out, batch, device, extra)")
        c = body.index("_p3_term = _p3f.loss_term(model, out, batch, device, extra)")
        assert a < b < body.index("loss = loss + _bd_term") < c < body.index("loss = loss + _p3_term")


# ===================================================================================================================== #
# pin_args and the weight gates                                                                                         #
# ===================================================================================================================== #
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_pin_args_admits_the_full_model_when_every_input_is_present_and_refuses_each_missing_one(kind):
    mod, key, wattr, hattr, pfx, wdest, fdest = KINDS[kind]
    base = dict(r9_perc_rig=False, w_box3d=1.0, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics")
    mod.pin_args(_ns(kind, **base))
    cases = [("w_box3d", 0.0, "w-box3d|r9-perc-rig"), ("map_hires", "off", "map-hires"), ("join3d", "", "join3d"), ("agent_join", "", "agent-join")]
    if kind != "bev2":                                                                       # B2c reads the pooled BEV, not the camera
        cases.append(("agent_rig_camera", "off", "agent-rig-camera"))
    for k, v, frag in cases:
        with pytest.raises(SystemExit, match=frag):
            mod.pin_args(_ns(kind, **{**base, k: v}))
    with pytest.raises(SystemExit, match="MUTUALLY EXCLUSIVE"):                              # B2 + B2c at once: refused in the full model too
        PF.pin_args(_ns("persp", "bev2", **base))
    # the perception rig's own admission is unchanged: its full argv passes (B1 needs no --w-box3d there), and B2 / B2c still need --w-box3d
    rigb = {**base, "r9_perc_rig": True}
    mod.pin_args(_ns(kind, **rigb))
    if kind == "depth":
        mod.pin_args(_ns(kind, **{**rigb, "w_box3d": 0.0}))
    else:
        with pytest.raises(SystemExit, match="w-box3d"):
            mod.pin_args(_ns(kind, **{**rigb, "w_box3d": 0.0}))


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_weight_gates_do_not_name_the_rig(kind):
    mod, key, wattr, hattr, pfx, wdest, fdest = KINDS[kind]
    gate = T.REFC_WEIGHT_GATES[wdest]["gate"]
    ok = _ns(kind, r9_perc_rig=False, w_box3d=1.0, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics")
    met, why = gate(ok)
    assert met, why
    assert ("--r9-perc-rig" in why) == (kind == "depth")                                     # only B1 still names the rig (as an alternative to --w-box3d)
    assert not gate(_ns(kind, r9_perc_rig=False, w_box3d=0.0, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics"))[0]
    assert not gate(_ns(r9_perc_rig=False, w_box3d=1.0, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics"))[0]   # no flag, no head
    if kind == "depth":                                                                      # the rig's B1 needs no --w-box3d (unchanged)
        assert gate(_ns(kind, r9_perc_rig=True, w_box3d=0.0, map_hires="on", join3d="j3", agent_join="aj", agent_rig_camera="extrinsics"))[0]


# ===================================================================================================================== #
# the dataset / build hops in full-model mode                                                                           #
# ===================================================================================================================== #
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_attach_dataset_accepts_the_full_model_and_hands_the_worker_its_source(kind):
    mod, key, wattr, hattr, pfx, wdest, fdest = KINDS[kind]
    model, cfg, args = _full(kind)
    ds = types.SimpleNamespace(agent_join=object(), join3d=object(), episodes=[])
    rep = mod.attach_dataset(ds, model, args)
    assert rep and getattr(ds, "_r9_bd" if kind == "depth" else "_r9_p3", None) is not None
    assert mod.attach_dataset(types.SimpleNamespace(), model, _ns()) == {}                    # OFF: touches nothing


def test_the_trainer_attaches_the_heads_in_full_mode_too_no_rig_gate_on_the_build_or_the_dataset_hooks():
    for needle in ("_bda.attach(model, args)", "_p3f.attach(model, args)", "_bda.attach_dataset(ds, model, args)", "_p3f.attach_dataset(ds, model, args)"):
        assert TRAINER_SRC.count(needle) == 1, needle
        line = [ln for ln in TRAINER_SRC.split("\n") if needle in ln][0]
        assert "_r9rig_on" not in line and "r9_perc_rig" not in line
    assert "if getattr(self, \"_r9_bd\", None) is not None:" in TRAINER_SRC and "if getattr(self, \"_r9_p3\", None) is not None:" in TRAINER_SRC


# ===================================================================================================================== #
# the full model's --init-from warm start                                                                               #
# ===================================================================================================================== #
def _source(model, path):
    R._save_source(model, path)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_warm_start_the_full_models_key_check_refuses_a_head_and_the_hold_is_the_way_around_it(kind, tmp_path):
    mod, key, wattr, hattr, pfx, wdest, fdest = KINDS[kind]
    src_model, _ = R._build()                                                                # the refcv8-like source: no aux head
    p = tmp_path / "src.pt"
    _source(src_model, p)
    model, cfg, args = _full(kind)
    # CONTROL: unprotected, the full model's warm start refuses the head's keys (they are not r8_* seams)
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        r8train.warm_start_from(model, str(p))
    before = {n: t.detach().clone() for n, t in model.state_dict().items() if n.startswith(pfx)}
    order = list(model.state_dict())
    head = _head(model, kind)
    with T._AuxHeadsHeldOut(model, enabled=True) as hold:
        assert hold.held == [hattr] and _head(model, kind) is None and not any(n.startswith(pfx) for n in model.state_dict())
        rec = r8train.warm_start_from(model, str(p))
    assert rec["n_new_refcv8_keys"] == 0 and rec["n_loaded"] == len(src_model.state_dict())
    assert _head(model, kind) is head                                                        # the SAME module is back ...
    assert list(model.state_dict()) == order                                                 # ... in the SAME place (checkpoint key order unchanged)
    after = {n: t for n, t in model.state_dict().items() if n.startswith(pfx)}
    assert set(after) == set(before) and all(torch.equal(before[n], after[n]) for n in before)   # ... with its own init untouched by the warm start
    sd = model.state_dict()
    src_sd = src_model.state_dict()
    assert all(torch.equal(sd[n], src_sd[n]) for n in src_sd)                                # everything the source carries WAS loaded


def test_warm_start_hold_is_a_noop_under_the_rig_without_a_head_and_it_restores_on_an_exception(tmp_path):
    plain, _ = R._build()
    order = list(plain.state_dict())
    with T._AuxHeadsHeldOut(plain, enabled=True) as h:
        assert h.held == []
    assert list(plain.state_dict()) == order
    model, _, _ = _full("depth")
    with T._AuxHeadsHeldOut(model, enabled=False) as h:                                      # the rig names its own exemptions: nothing held
        assert h.held == [] and _head(model, "depth") is not None
    pre = list(model.state_dict())
    with pytest.raises(RuntimeError, match="boom"):
        with T._AuxHeadsHeldOut(model, enabled=True):
            assert _head(model, "depth") is None
            raise RuntimeError("boom")                                                       # a refusal inside the warm start still puts the head back
    assert _head(model, "depth") is not None and list(model.state_dict()) == pre


def test_the_hold_wraps_the_tips_warm_start_statement_and_stamps_it():
    stmt = ("_init_stamp = dict((_r9rig.warm_start_perc_rig if _r9rig_on else r8train.warm_start_from)"
            "(model, args.init_from),")
    i = TRAINER_SRC.index(stmt)                                                              # the statement the tip's pin reads is unchanged
    w = TRAINER_SRC.rindex("with _AuxHeadsHeldOut(model, enabled=not _r9rig_on) as _aux_hold:", 0, i)
    assert 0 < i - w < 400 and TRAINER_SRC.count("_AuxHeadsHeldOut(") == 1                    # the one call site (the class is `class _AuxHeadsHeldOut:`)
    assert '_init_stamp["aux_heads_fresh_init"] = list(_aux_hold.held)' in TRAINER_SRC


# ===================================================================================================================== #
# the launch gate keeps forbidding the levers until the MM flips the profile                                            #
# ===================================================================================================================== #
def test_the_launch_gate_profile_still_forbids_the_aux_levers_for_the_full_model():
    import launch_gate as LG
    for prof in ("refcv9-e2", "refcv9-full", "refcv9-full-d1"):
        forb = [f for ent in LG.PROFILES[prof]["forbidden_levers"] for f in ((ent[0],) if isinstance(ent[0], str) else tuple(ent[0]))]
        for f in ("--r9-box-depth-aux", "--r9-box-persp-aux", "--r9-box-bev2-aux", "--w-box-depth", "--w-box-persp", "--w-box-bev2"):
            assert f in forb, (prof, f)
    # and G-LIVE already has a row per weight (the rows the C12-FULL handoff says exist)
    for dest in ("w_box_depth", "w_box_persp", "w_box_bev2"):
        assert dest in LG.LIVE_WEIGHT_RULES and dest in T.REFC_WEIGHT_GATES
