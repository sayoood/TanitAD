"""``--map-hires-trunk-grad-scale S`` (K2DC's next lever: PREREG_K2DC02 section 4 step 2, "a stop-gradient from the map head into the trunk").

The class this file exists for is "built, tested, and UNREACHABLE FROM ITS CALLER" (CLAUDE.md 2026-09-10/11). So it proves reachability, not just that the op works:

  1. HOPS: argparse -> namespace -> ``map_trunk_gs.attach`` (right after the branch is built, in train() AND both replays) -> the branch attribute
     ``trunk_grad_scale`` -> ``MapHiresBranch._gs`` on the banded head's inputs -> the ``map_hires`` loss term -> ``.backward()``;
  2. ANALYTIC IDENTITY on a tiny REAL RefCV3Model (resnet18, the banded 10 cm branch, the trainer's own ``compute_losses_v3``): with ONLY the map term
     backpropagated, trunk-parameter gradients at S = 0.5 are exactly 0.5 x those at S = 1.0, at S = 0 exactly 0.0 (while the banded head's parameters keep
     S = 1.0's non-zero gradient); a CONTROL term (``traj``) gives identical trunk gradients at every S; the forward loss is unchanged;
  3. OFF (S = 1.0) is bit-identical to the tip's graph: no op node, ``grad_scale`` never called, same loss and every gradient bit for bit, no attribute,
     the state_dict keys unchanged;
  4. MUTATION: with the one line that applies the scale replaced by ``return x`` the identity tests go RED (the run is in the HANDOFF; here the same
     mutation is re-applied to a monkeypatched op and must fail the same assertions);
  5. G-DVB: the registry entry, both directions, the refusals; 6. the replays (box-overfit harness build, eval loader) carry the flag.
"""
from __future__ import annotations

import argparse
import json
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
import test_map_banded_wiring as W  # noqa: E402  (the tiny REAL RefCV3Model with the banded branch + the trainer's loss)
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_trunk_gs as G  # noqa: E402

FLAG = "--map-hires-trunk-grad-scale"
TRUNK = "core.encoder."
MAPHEAD = ("_map_hires.banded.", "_map_hires.refine.", "_map_hires.near.", "_map_hires.near_refine.")
SHARED = ("_map_hires.lift.", "_map_hires.encoder.")          # upstream of the op under the shared build: they get the map term's gradient scaled too


def _a(s=None, **over):
    a = dict(r9_map_banded=True, map_hires="on", w_map_hires=1.0)
    if s is not None:
        a["map_hires_trunk_grad_scale"] = s
    a.update(over)
    return argparse.Namespace(**a)


def _model(s: float):
    model, cfg = W._build(banded=True)
    G.attach(model, _a(s))
    return model, cfg


def _grads(s: float, term: str):
    """The model at scale ``s``, ``term`` (a key of the trainer's loss dict) backpropagated ALONE -> ({name: grad}, forward value of the term)."""
    model, cfg = _model(s)
    losses = W._losses(model, cfg)
    model.zero_grad(set_to_none=True)
    losses[term].backward()
    out = {n: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}
    return out, float(losses[term].detach())


def _sel(g, prefixes):
    return {n: v for n, v in g.items() if n.startswith(prefixes)}


# =========================================================================== #
# 2. the analytic identity                                                     #
# =========================================================================== #
def test_IDENTITY_map_term_trunk_gradient_is_exactly_S_times_the_unscaled_one():
    g1, v1 = _grads(1.0, "map_hires")
    g5, v5 = _grads(0.5, "map_hires")
    g0, v0 = _grads(0.0, "map_hires")
    assert v1 == v5 == v0                                                             # forward = identity: the same loss value, bit for bit
    t1, t5, t0 = _sel(g1, TRUNK), _sel(g5, TRUNK), _sel(g0, TRUNK)
    assert t1 and sum(float(v.abs().sum()) for v in t1.values()) > 0.0                # the probe is not vacuous: the map term DOES reach the trunk at S = 1
    assert [n for n in t1 if not torch.equal(t5[n], 0.5 * t1[n])] == []               # S = 0.5: exactly half (a power of two commutes with float backprop)
    assert [n for n in t1 if not torch.equal(t0[n], torch.zeros_like(t1[n]))] == []   # S = 0: EXACTLY 0.0 on every trunk parameter
    for pref in SHARED:                                                               # the shared lift / encoder sit upstream of the op: scaled too
        a, b, c = _sel(g1, pref), _sel(g5, pref), _sel(g0, pref)
        assert a and sum(float(v.abs().sum()) for v in a.values()) > 0.0
        assert [n for n in a if not torch.equal(b[n], 0.5 * a[n])] == []
        assert [n for n in a if float(c[n].abs().sum()) != 0.0] == []
    h1, h5, h0 = _sel(g1, MAPHEAD), _sel(g5, MAPHEAD), _sel(g0, MAPHEAD)              # downstream of the op: the FULL gradient at every S
    assert h1 and [n for n in h1 if float(h1[n].abs().sum()) == 0.0] == []            # non-zero (the head learns) ...
    assert [n for n in h1 if not (torch.equal(h5[n], h1[n]) and torch.equal(h0[n], h1[n]))] == []      # ... and equal to S = 1.0's at S = 0.5 and 0


def test_CONTROL_another_loss_term_gives_identical_trunk_gradients_at_every_S():
    ref, vref = _grads(1.0, "traj")
    t_ref = _sel(ref, TRUNK)
    assert t_ref and sum(float(v.abs().sum()) for v in t_ref.values()) > 0.0
    for s in (0.5, 0.0):
        g, v = _grads(s, "traj")
        assert v == vref
        assert [n for n in t_ref if not torch.equal(_sel(g, TRUNK)[n], t_ref[n])] == [], s     # no other term sees the op


def test_the_planner_BEV_output_is_the_unscaled_feats():
    """``out["map_hires_bev"]`` (what the planner pool / box read) is the branch's UNSCALED encoder output: a term on it gets the full gradient at S = 0."""
    def trunk_grad(s):
        model, cfg = _model(s)
        model.train()
        batch = W.R._batch(cfg)
        out_holder = {}
        orig = H.MapHiresBranch.forward

        def spy(self, *a, **k):
            r = orig(self, *a, **k)
            out_holder["bev"] = r["map_hires_bev"]
            return r
        H.MapHiresBranch.forward = spy
        try:
            T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        finally:
            H.MapHiresBranch.forward = orig
        model.zero_grad(set_to_none=True)
        out_holder["bev"].square().sum().backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if n.startswith(TRUNK) and p.grad is not None}
    g1, g0 = trunk_grad(1.0), trunk_grad(0.0)
    assert g1 and sum(float(v.abs().sum()) for v in g1.values()) > 0.0
    assert [n for n in g1 if not torch.equal(g1[n], g0[n])] == []


# =========================================================================== #
# 3. OFF is the tip's graph                                                    #
# =========================================================================== #
def _count_nodes(t, name="_GradScaleBackward"):
    seen, stack, n = set(), [t.grad_fn], 0
    while stack:
        f = stack.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        n += type(f).__name__ == name
        stack.extend(x for x, _ in f.next_functions)
    return n


def test_OFF_inserts_no_op_sets_nothing_and_is_bit_identical_to_the_tip(monkeypatch):
    calls = []
    real = G.grad_scale
    monkeypatch.setattr(G, "grad_scale", lambda x, s: (calls.append(s), real(x, s))[1])
    torch.manual_seed(0)
    off, cfg = W._build(banded=True)
    G.attach(off, _a())                                                              # the flag unset
    G.attach(off, _a(1.0))                                                           # the flag at its default
    assert not hasattr(off, G.MODEL_ATTR) and off._map_hires.trunk_grad_scale == 1.0 and "trunk_grad_scale" not in vars(off._map_hires)
    ref = W._build(banded=True)[0]
    assert sorted(off.state_dict()) == sorted(ref.state_dict())
    lo = W._losses(off, cfg)
    assert calls == [] and _count_nodes(lo["map_hires"]) == 0                        # the op is never even called, and not in the graph
    off.zero_grad(set_to_none=True)
    lo["loss"].backward()
    g_off = {n: p.grad.clone() for n, p in off.named_parameters() if p.grad is not None}
    # THE TIP: the branch with `_gs` removed (a no-op method == the tip's forward, which had no such call)
    monkeypatch.setattr(H.MapHiresBranch, "_gs", lambda self, x: x)
    tip, cfg_t = W._build(banded=True)
    lt = W._losses(tip, cfg_t)
    tip.zero_grad(set_to_none=True)
    lt["loss"].backward()
    g_tip = {n: p.grad.clone() for n, p in tip.named_parameters() if p.grad is not None}
    assert torch.equal(lo["loss"], lt["loss"]) and torch.equal(lo["map_hires"], lt["map_hires"])
    assert sorted(g_off) == sorted(g_tip) and [n for n in g_off if not torch.equal(g_off[n], g_tip[n])] == []     # every gradient, bit for bit


def test_ON_inserts_the_op_and_adds_no_parameter_or_buffer():
    off, _ = W._build(banded=True)
    on, cfg = _model(0.5)
    assert sorted(on.state_dict()) == sorted(off.state_dict()) and sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
    assert getattr(on, G.MODEL_ATTR) == 0.5 and on._map_hires.trunk_grad_scale == 0.5
    lo = W._losses(on, cfg)
    assert _count_nodes(lo["map_hires"]) == 2                                        # the near lift's trunk read (fmap_s8) and the banded head's BEV input (feats)


# =========================================================================== #
# 4. mutation                                                                  #
# =========================================================================== #
def test_MUTATION_without_the_scaling_line_the_identity_goes_RED(monkeypatch):
    monkeypatch.setattr(G, "grad_scale", lambda x, s: x)                              # "delete the line that applies the scale"
    with pytest.raises(AssertionError):
        test_IDENTITY_map_term_trunk_gradient_is_exactly_S_times_the_unscaled_one()
    monkeypatch.undo()
    test_IDENTITY_map_term_trunk_gradient_is_exactly_S_times_the_unscaled_one()        # restored: GREEN


def test_the_single_applying_line_exists_once():
    src = (ROOT / "tanitad" / "train" / "map_trunk_gs.py").read_text(encoding="utf-8")
    assert src.count("return _GradScale.apply(x, float(s))") == 1


# =========================================================================== #
# 1 / 5. hops, argv, G-DVB                                                     #
# =========================================================================== #
def test_HOPS_the_flag_reaches_the_banded_heads_inputs_from_every_caller():
    parser = T.build_parser()
    assert parser.parse_args(["--arm", "hier", "--out", "z"]).map_hires_trunk_grad_scale == 1.0                   # the DEFAULT is OFF
    assert parser.parse_args(["--arm", "hier", "--out", "z", FLAG, "0.25"]).map_hires_trunk_grad_scale == 0.25
    srcs = {p: (ROOT / p).read_text(encoding="utf-8") for p in ("scripts/refc_v3_train.py", "tanitad/eval/refcv7_loader.py", "scripts/g_box_overfit.py",
                                                                 "tanitad/models/map_head_hires.py")}
    tr = srcs["scripts/refc_v3_train.py"]
    for needle in ("_mgs.register_dvb_levers(_dvb.register)", "_mgs.pin_args(args)", "_mgs.attach(model, args)", "_mgs.add_arguments(g8)"):
        assert tr.count(needle) == 1, needle
    assert tr.index("model._w_map_hires = float(args.w_map_hires)") < tr.index("_mgs.attach(model, args)")      # right after the branch is built (the gate's capture point follows)
    assert srcs["tanitad/eval/refcv7_loader.py"].count("tr._mgs.attach(model, args)") == 1
    assert srcs["scripts/g_box_overfit.py"].count("tr._mgs.attach(m, a)") == 1
    head = srcs["tanitad/models/map_head_hires.py"]
    assert head.count("self._gs(fmap_s8)") == 1 and head.count("self._gs(feats)") == 1 and head.count("self._gs(bev)") == 1
    assert head.count("self._gs(") == 3                                              # the banded head's inputs, and nowhere else
    assert "_gs(" not in head[head.index("        if near_block_zeros and self.near_refine is None:"):]       # the 8-way (non-banded) path is untouched


def test_ARGV_refusals_name_the_flag_and_OFF_checks_nothing():
    G.pin_args(argparse.Namespace())                                                  # no attribute at all
    G.pin_args(_a(1.0, r9_map_banded=False, map_hires="off", w_map_hires=0.0))        # OFF: nothing checked
    G.pin_args(_a(0.0))
    G.pin_args(_a(0.5))
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(SystemExit, match=FLAG):
            G.pin_args(_a(bad))
    for over, why in ((dict(r9_map_banded=False), "r9-map-banded"), (dict(map_hires="off"), "map-hires is not"), (dict(w_map_hires=0.0), "w-map-hires is 0")):
        with pytest.raises(SystemExit, match=why):
            G.pin_args(_a(0.5, **over))


def test_GDVB_registered_and_reads_what_the_forward_reads_both_ways():
    parser = T.build_parser()
    assert D.coverage(parser) == [] and D.REGISTRY["map_hires_trunk_grad_scale"].kind == "built"
    chk = D.REGISTRY["map_hires_trunk_grad_scale"].check
    on, _ = _model(0.5)
    off, _ = W._build(banded=True)
    assert chk(on, _a(0.5)) == [] and chk(off, _a()) == [] and chk(off, _a(1.0)) == []
    declared_not_built = chk(off, _a(0.5))                                            # a build that LOST the attach line
    assert [m.read_from for m in declared_not_built] == ["model._map_hires.trunk_grad_scale", "model._map_trunk_gs"]
    assert chk(on, _a(0.25)) and chk(on, _a())                                        # a different value / built but not declared
    on._map_hires.trunk_grad_scale = 1.0                                              # the branch's own read lost: only the forward-read entry names it
    assert [m.read_from for m in chk(on, _a(0.5))] == ["model._map_hires.trunk_grad_scale"]
    nb, _ = W._build(banded=False)
    with pytest.raises(SystemExit, match=FLAG):
        G.attach(nb, _a(0.5))                                                         # no banded head: refused, never silently a no-op


# =========================================================================== #
# 6. the replays                                                               #
# =========================================================================== #
def test_the_box_overfit_replay_builds_the_scale_train_declares(tmp_path):
    import launch_gate as LG
    import test_g_box_overfit_banded as GB
    import test_g_box_overfit_near_lift as NL
    d = tmp_path / "rig"
    d.mkdir()
    NL._synth_inputs(d)
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    (d / "cw.json").write_text(json.dumps({"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES), "weights": [1.0, 1.5, 3.0, 2.0, 4.0, 1.2, 2.5, 1.1],
                                           "dry_run": False, "extent": {"x_max_m": float(GB.X_MAX), "y_half_m": float(GB.Y_HALF)}}), encoding="utf-8")
    argv_on = GB._banded_argv(d, "--r9-map-banded", FLAG, "0.5")
    argv_off = GB._banded_argv(d, "--r9-map-banded")
    Tm = NL._load("refc_v3_train_mtgs_ref", ROOT / "scripts" / "refc_v3_train.py")
    ref = LG.run_trainer_until(Tm, argv_on, "model")["model"]                         # the REAL train(): attaches the scale
    assert ref._map_hires.trunk_grad_scale == 0.5 and getattr(ref, G.MODEL_ATTR) == 0.5
    Gm = NL._load("g_box_overfit_mtgs_t", ROOT / "scripts" / "g_box_overfit.py")
    m, ad = NL._harness_model(Gm, Tm, argv_on)
    assert ad.dvb_mismatches == []                                                    # G-DVB clean WITH the flag in argv
    assert m._map_hires.trunk_grad_scale == 0.5 and getattr(m, G.MODEL_ATTR) == 0.5
    assert sorted(m.state_dict()) == sorted(ref.state_dict())                         # parameter-free: no mirror of any state
    m0, ad0 = NL._harness_model(Gm, Tm, argv_off)
    assert ad0.dvb_mismatches == [] and not hasattr(m0, G.MODEL_ATTR) and m0._map_hires.trunk_grad_scale == 1.0
    # RED: the harness without its attach line refuses the same argv (the K2DC replay defect, caught here)
    src = (ROOT / "scripts" / "g_box_overfit.py").read_bytes().decode("utf-8")
    line = "            tr._mgs.attach(m, a)                  # train(): --map-hires-trunk-grad-scale (1.0: nothing set)\n"
    assert src.count(line) == 1
    p = tmp_path / "gbo_no_attach.py"
    p.write_text(src.replace(line, ""), encoding="utf-8", newline="\n")
    G0 = NL._load("g_box_overfit_mtgs_red_t", p)
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, Tm, argv_on)
    assert "NOT the declared launch model" in str(e.value) and "map_hires_trunk_grad_scale: declared 0.5 but BUILT 1.0" in str(e.value)
    assert NL._harness_model(G0, Tm, argv_off)[1].dvb_mismatches == []                # control: OFF is unaffected by the missing line


def test_the_eval_loader_attaches_the_scale_like_train():
    """The loader's replay build (``_build_model``) calls the same ``attach`` right after ``build_map_hires_block`` and before ``tr._bw.attach``
    (the block itself holds no attach: test_refcv7_loader_aux_stamp; the scale is parameter-free, so no stamp moves), so G-DVB at the
    loader sees the declared scale; and a loader WITHOUT the line is refused by the same G-DVB entry (the model the loader would build has no attribute)."""
    src = (ROOT / "tanitad" / "eval" / "refcv7_loader.py").read_bytes().decode("utf-8")
    needle = "    tr._mgs.attach(model, args)"
    i = src.index(needle)
    assert src.index('    rec["map_hires"] = build_map_hires_block(tr, model, args, device)') < i < src.index("    tr._bw.attach(model, args)")
    b0 = src.index("def build_map_hires_block(")
    assert "attach(" not in src[b0:src.index("\ndef ", b0 + 1)]                     # the block itself stays attach-free
    on = _model(0.5)[0]
    lost, _ = W._build(banded=True)                                                   # what the loader would build with the line removed
    chk = D.REGISTRY["map_hires_trunk_grad_scale"].check
    assert chk(on, _a(0.5)) == [] and len(chk(lost, _a(0.5))) == 2
