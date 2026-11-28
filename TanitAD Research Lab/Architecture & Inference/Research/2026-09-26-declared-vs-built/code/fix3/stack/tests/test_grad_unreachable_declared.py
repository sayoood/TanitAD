"""⛔⛔ BUILT, BYPASSED BY DESIGN -> FROZEN AND DECLARED (batch 3 (a), 2026-09-27).

MEASURED by the refcv7 launch gate's G-LIVE smoke on the refcv7 argv (and by NEW-1, its F-2):
``core.decoder.control_head``, ``core.decoder.offset_head`` and ``scorer.goal_point`` receive ZERO
gradient. SPEC_REFCV7 §2's G-LIVE requires "every declared-trainable group gets a non-zero
gradient", and the refcv7 launch was blocked on it. Each of the three is bypassed BY
CONSTRUCTION, not by a wiring defect:

* ``offset_head``, on a sampler build: the classifier pass writes ``x = bank + offset`` and
  ``_sample`` REPLACES ``x``; ``out["offset"]`` is read by no loss of ``compute_losses_v3``;
* ``control_head``, under F3: ``_decode_ctrl`` returns the CASCADE's last stage whenever
  ``self.cascade`` is set (DD's per-layer heads generalise it);
* ``scorer.goal_point``: E9 always passes the structured goal ``g2``; the free decode is
  returned only as ``goal_point_free``.

They are therefore FROZEN and DECLARED with ``tanitad.models._gradreach.declare_grad_unreachable``
(the tensors stay in ``state_dict``, so every checkpoint still loads strictly), and G-DVB holds
the declared set against ARGV (``declared_vs_built.check_grad_unreachable``). Pinned here:

1. the declared set on a refcv7-shaped build is the LITERAL three, nothing else is frozen, and
   the state dict still carries all six tensors;
2. the declaration is MEASURED TRUE: re-opened for one backward of the trainer's OWN loss, none
   of the three receives a gradient, while the backward reached the model (positive control);
3. G-LIVE's dead-group scan (the gate's rule: a leaf group whose trainable tensors never carry a
   non-zero gradient) differs between the declared and the undeclared build by EXACTLY the three;
4. the freeze is CONDITIONAL, and each condition is shown from the other side: on a classifier
   build ``offset_head`` IS the refinement and learns; on a sampler build without F3
   ``control_head`` IS the head and learns;
5. RED arms: a live module frozen without a declaration; a declared module made trainable
   again; a declaration argv does not justify; a bypass argv implies that the build did not
   declare; a declared module a loss DOES reach (the probe names it); a probe whose closure ran
   no backward;
6. the REAL ``train()`` records the declared set in config.json and passes G-DVB.
Every expectation is a literal. CPU, smoke width.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402
from tanitad.models import _gradreach as gr  # noqa: E402
from tanitad.refs import anchor_meta as am  # noqa: E402
from tanitad.refs import refc_sampler as rs  # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402

#: the refcv6 live run's diffusion flags, F7-F9 excepted (as in
#: `test_refcv6_f3_cascade_reaches_loss.py` and `test_residual_prior.py`)
F1_F6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
         "--f5-emitting-conf", "--f6-w-u0-zero"]
#: the refcv7 core on the smoke rig: NEW-1's `_arm` recipe with SPEC_REFCV7 §10's prior
R7_CORE = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
           "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
           "--residual-prior", "ha0_ext_pose"]
#: ⛔ THE declared set on that build -- a LITERAL, never read from the rule table under test
DECLARED = {"core.decoder.control_head", "core.decoder.offset_head", "scorer.goal_point"}
FROZEN_TENSORS = sorted(f"{m}.{t}" for m in DECLARED for t in ("weight", "bias"))
A_LON = (-2.0, -1.0, 0.0, 1.0, 2.0)
A_LAT = (-1.5, -0.5, 0.0, 0.5)


def _grid() -> "torch.Tensor":
    return torch.cartesian_prod(torch.tensor(A_LON), torch.tensor(A_LAT))


def _build(argv, seed: int = 0):
    args = T.build_parser().parse_args(list(argv) + ["--out", "z"])
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(args.arm == "hier"), args)
    torch.manual_seed(seed)
    model = T.v3.RefCV3Model(cfg)
    dec = model.core.decoder
    if bool(getattr(dec, "anchor_v0_cond", False)):
        with torch.no_grad():                   # a real control ladder (the F9 tensor check)
            dec.anchor_controls.copy_(_grid().to(dec.anchor_controls.dtype))
    return args, cfg, model


def _batch(cfg):
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    ds.ego_history = cfg.core.ego_history is not None
    return torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])


def _loss_backward(model, batch, seed: int = 1):
    model.train()
    torch.manual_seed(seed)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    losses["loss"].backward()
    return losses


def _frozen(model) -> list[str]:
    return sorted(n for n, p in model.named_parameters() if not p.requires_grad)


def _dead_groups(model) -> set[str]:
    """The launch gate's G-LIVE rule (`launch_gate.grad_table`): a LEAF group -- a parameter's
    parent path -- whose every TRAINABLE tensor carries no non-zero gradient."""
    live: dict[str, bool] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = n.rsplit(".", 1)[0]
        live[g] = live.get(g, False) or (p.grad is not None and bool((p.grad != 0).any()))
    return {g for g, ok in live.items() if not ok}


@pytest.fixture(scope="module")
def r7():
    return _build(R7_CORE + F1_F6)


# ---- 1. the literal set --------------------------------------------------------------------- #
def test_1_the_declared_set_is_the_LITERAL_three_and_nothing_else_is_frozen(r7):
    args, _cfg, model = r7
    decl = gr.grad_unreachable_prefixes(model)
    assert set(decl) == DECLARED
    assert all(len(why) > 20 for why in decl.values()), decl        # each carries its REASON
    assert dvb.declared_grad_unreachable(model) == dict(sorted(decl.items()))
    assert _frozen(model) == FROZEN_TENSORS
    assert set(dvb.expected_grad_unreachable(args)) == DECLARED
    assert {p for p, _rule, _why in dvb.GRAD_UNREACHABLE_RULES} == DECLARED
    assert dvb.check_grad_unreachable(model, args) == []
    # DECLARED, NOT DELETED: the six tensors stay in the state dict, so a banked checkpoint loads
    # strictly into this build (and this build's into a fresh one)
    sd = model.state_dict()
    assert set(FROZEN_TENSORS) <= set(sd)
    _a2, _c2, fresh = _build(R7_CORE + F1_F6, seed=5)
    fresh.load_state_dict(sd, strict=True)


# ---- 2. the declaration, MEASURED ----------------------------------------------------------- #
def test_2_the_declaration_is_MEASURED_true_on_the_trainers_own_loss(r7):
    _args, cfg, model = r7
    batch = _batch(cfg)
    got = dvb.probe_grad_unreachable(model, lambda: _loss_backward(model, batch))
    assert got == [], [str(m) for m in got]
    assert _frozen(model) == FROZEN_TENSORS                          # the probe restored it
    assert all(p.grad is None for p in model.parameters())           # ...and cleared the grads


# ---- 3. G-LIVE's view: the declaration removes EXACTLY the three ---------------------------- #
def test_3_G_LIVE_dead_groups_differ_by_EXACTLY_the_three(monkeypatch):
    argv = R7_CORE + F1_F6
    _a1, cfg1, declared = _build(argv)
    _loss_backward(declared, _batch(cfg1))
    dead_declared = _dead_groups(declared)
    # the deliberate regression: the constructors' declarations removed (the pre-batch-3 build)
    monkeypatch.setattr(gr, "declare_grad_unreachable", lambda module, why: module)
    a2, cfg2, undeclared = _build(argv)
    assert gr.grad_unreachable_prefixes(undeclared) == {} and _frozen(undeclared) == []
    _loss_backward(undeclared, _batch(cfg2))
    dead_undeclared = _dead_groups(undeclared)
    assert dead_undeclared - dead_declared == DECLARED, sorted(dead_undeclared - dead_declared)
    assert dead_declared <= dead_undeclared
    assert not (dead_declared & DECLARED)
    # ...and G-DVB names all three on the undeclared build, from argv alone
    got = sorted((m.lever, m.built) for m in dvb.check_grad_unreachable(undeclared, a2))
    assert got == sorted((f"grad_unreachable[{p}]", "not declared") for p in DECLARED)


# ---- 4. the freeze is CONDITIONAL: the other side of each condition ------------------------- #
def test_4a_on_a_CLASSIFIER_build_offset_head_IS_the_refinement_and_LEARNS():
    args, cfg, model = _build(["--arm", "hier", "--n-anchors", "20"])
    assert model.core.decoder.control_head is None
    assert set(gr.grad_unreachable_prefixes(model)) == {"scorer.goal_point"}
    assert dvb.check_grad_unreachable(model, args) == []
    _loss_backward(model, _batch(cfg))
    g = model.core.decoder.offset_head.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0


def test_4b_on_a_sampler_build_WITHOUT_F3_control_head_IS_the_head_and_LEARNS():
    args, cfg, model = _build(R7_CORE + ["--w-u0", "0.5"])
    assert model.core.decoder.cascade is None
    assert set(gr.grad_unreachable_prefixes(model)) == {"core.decoder.offset_head",
                                                        "scorer.goal_point"}
    assert dvb.check_grad_unreachable(model, args) == []
    _loss_backward(model, _batch(cfg))
    g = model.core.decoder.control_head.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0


# ---- 5. RED arms ---------------------------------------------------------------------------- #
def test_5a_RED_a_LIVE_module_frozen_without_a_declaration_is_named(r7):
    args, _cfg, model = r7
    head = model.core.decoder.conf_head
    head.requires_grad_(False)
    try:
        got = dvb.check_grad_unreachable(model, args)
        assert [(m.lever, m.read_from) for m in got] == [("frozen parameters",
                                                          "model.named_parameters()")]
        assert "core.decoder.conf_head.weight" in got[0].why
    finally:
        head.requires_grad_(True)


def test_5b_RED_a_declared_module_made_trainable_again_is_named(r7):
    args, _cfg, model = r7
    model.core.decoder.offset_head.requires_grad_(True)
    try:
        got = [(m.lever, m.read_from) for m in dvb.check_grad_unreachable(model, args)]
        assert got == [("grad_unreachable[core.decoder.offset_head]",
                        "core.decoder.offset_head")]
    finally:
        model.core.decoder.offset_head.requires_grad_(False)


def test_5c_RED_a_declaration_argv_does_NOT_justify_is_named(r7):
    args, _cfg, model = r7
    no_f3 = argparse.Namespace(**{**vars(args), "f3_per_layer": False})
    got = [(m.lever, m.declared, m.built) for m in dvb.check_grad_unreachable(model, no_f3)]
    assert got == [("grad_unreachable[core.decoder.control_head]",
                    "not declared (argv: reached)", "declared")]
    flat = argparse.Namespace(**{**vars(args), "arm": "flat"})
    assert [m.lever for m in dvb.check_grad_unreachable(model, flat)] == \
        ["grad_unreachable[scorer.goal_point]"]


def test_5d_RED_a_bypass_argv_implies_that_the_build_did_NOT_declare():
    args, _cfg, model = _build(R7_CORE + ["--w-u0", "0.5"])          # no F3 built
    with_f3 = argparse.Namespace(**{**vars(args), "f3_per_layer": True})
    got = [(m.lever, m.declared, m.built) for m in dvb.check_grad_unreachable(model, with_f3)]
    assert got == [("grad_unreachable[core.decoder.control_head]",
                    "declared (argv: bypassed by design)", "not declared")]


def test_5e_RED_the_probe_names_a_declared_module_a_loss_DOES_reach(r7):
    _args, cfg, model = r7
    batch = _batch(cfg)

    def reached():                     # the defect: a loss wired onto a declared-dead head
        model.train()
        torch.manual_seed(1)
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        (losses["loss"] + model.core.decoder.offset_head.weight.sum()).backward()
    got = [(m.lever, m.read_from) for m in dvb.probe_grad_unreachable(model, reached)]
    assert got == [("grad_unreachable[core.decoder.offset_head]", "core.decoder.offset_head")]
    assert _frozen(model) == FROZEN_TENSORS


def test_5f_RED_a_probe_whose_closure_ran_NO_backward_certifies_nothing(r7):
    _args, _cfg, model = r7
    assert [m.lever for m in dvb.probe_grad_unreachable(model, lambda: None)] == \
        ["grad_unreachable probe"]


# ---- 6. the REAL train() --------------------------------------------------------------------- #
def _anchor_file(tmp: Path) -> Path:
    """`test_refcv6_f3_cascade_reaches_loss.py`'s 20-anchor v0-conditioned alat vocabulary."""
    ctrl = _grid()                                                      # [20, 2]
    n, s = ctrl.shape[0], len(V3_HORIZONS)
    u = ctrl[None, :, None, :].expand(1, n, s, 2).contiguous()
    anchors = rs.roll_controls(u, torch.tensor([10.0]), tuple(V3_HORIZONS),
                               control_units="alat")[0]
    art = am.build_anchor_artifact(anchors, ctrl, control_units="alat", horizons=V3_HORIZONS,
                                   dt=0.1, ref_speed_ms=10.0, kappa_cap=0.12, alat_v_floor=4.0,
                                   builder=None)
    p = tmp / "anchors_v0cond_alat_20.pt"
    torch.save(art, p)
    return p


def test_6_the_REAL_train_passes_G_DVB_and_RECORDS_the_declared_set(tmp_path):
    out = tmp_path / "run"
    argv = ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2",
            "--steps", "1", "--batch", "2", "--device", "cpu", "--save-every", "100",
            "--log-every", "1", "--sampler", "ddim", "--anchors", str(_anchor_file(tmp_path)),
            "--anchor-v0-conditioned", "--n-anchors", "20"] + F1_F6
    T.train(T.build_parser().parse_args(argv))
    rec = json.loads((out / "config.json").read_text(encoding="utf-8"))["declared_vs_built"]
    assert rec["mismatches"] == 0
    assert set(rec["grad_unreachable"]) == DECLARED
    assert all(isinstance(v, str) and v for v in rec["grad_unreachable"].values())
