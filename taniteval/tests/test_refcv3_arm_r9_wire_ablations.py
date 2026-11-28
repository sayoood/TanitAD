"""refcv9 planner-wire knockouts in `taniteval/tools/refcv3_arm.py`: `r9_map_chan_zero` (W-MAP-2) and `r9_box_tok_zero` (W-BOX).

PURPOSE. A zero-training diagnosis: refcv9 arm K2 (trained with `--r9-plan-map-channels` and `--r9-plan-box-tokens`) is laterally
noisier than K0, so K2 is rolled with each planner wire switched OFF at eval time.

THE CLASS THIS FILE GUARDS (CLAUDE.md, "built, tested and unreachable from its caller"): an ablation that parses and changes nothing
reads like a knockout and is a copy of the FULL arm. So the proofs run on a SMALL REAL `RefCV3Model` built by the same rig and the
same `plan_wires_flag.attach` the trainer uses (`stack/tests/test_plan_wires_r9_model.py::_rig`), driven by the trainer's own
`compute_losses_v3`, through the REAL `apply_ablations`:

  (a) REACHABILITY: wire ON with a non-zero weight, ablated output != un-ablated output;
  (b) IDENTITY: on a model whose weights are still at their zero init, ablating changes NOTHING, bit for bit (and says INERT);
  (c) REFUSAL: a model built without the wire (K0-like), or with only the OTHER wire, raises SystemExit, never a silent no-op;
  (d) THE REST IS UNTOUCHED: some outputs stay bit-identical to the un-ablated run, and EVERY output equals the model that NEVER BUILT
      the wire (this is the "off == not built" proof: the OFF model shares the base weights);
  (e) the `--ablate` CLI choice lists both.

The deliberate regressions (delete the hook line / the gate zeroing) were run by hand and each turns (a) and (d) RED; see the hand-back.
ASCII-only assertion text by choice (cp1252 console on the dev box).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

_TREE = Path(__file__).resolve().parents[2]
TOOL = _TREE / "taniteval" / "tools" / "refcv3_arm.py"
for _p in (str(_TREE / "stack"), str(_TREE / "stack" / "scripts"), str(_TREE / "stack" / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_plan_wires_r9_model as RG  # noqa: E402  the rig (not its tests: only the module object is bound)

T = RG.T
rc_spec = importlib.util.spec_from_file_location("_refcv3_arm_r9wire_ut", TOOL)
rc = importlib.util.module_from_spec(rc_spec)
sys.modules["_refcv3_arm_r9wire_ut"] = rc
rc_spec.loader.exec_module(rc)

MAP, BOX = "r9_map_chan_zero", "r9_box_tok_zero"
COUNTER = "tacv6_n_scene_mean"      # telemetry count of scene tokens: moves by exactly the 32 box tokens when W-BOX is built


# --------------------------------------------------------------------------- #
def _models(map_on=True, box_on=True):
    """-> (cfg, on, off): `on` has the wires, `off` never built them; both carry the SAME base weights."""
    cfg, off, _ = RG._rig(attach=False, ns=RG._ns(False, False))
    _, on, _ = RG._rig(ns=RG._ns(map_on, box_on))
    on.load_state_dict(off.state_dict(), strict=False)
    return cfg, on, off


def _ev(model, batch):
    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        return RG._scalars(T.compute_losses_v3(model, batch, "cpu", mode="diffusion"))


def _moved(a, b, ignore=()):
    assert a.keys() == b.keys() and len(a) > 10
    return sorted(k for k in a if k not in ignore and not torch.equal(a[k], b[k]))


def _open_map(model):
    with torch.no_grad():
        model._perception.bev_pool.map_chan_proj.proj.weight.fill_(0.5)


def _open_box(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.rsplit(".", 1)[-1] == "box_gate":
                p.fill_(0.5)


def _ablate(model, cfg, name):
    return rc.apply_ablations(model, cfg, [name], steps=0, feed_ego=False)


# --------------------------------------------------------------------------- #
# (e) the registry and the CLI                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [MAP, BOX])
def test_registry_entry_and_cli_choice(name):
    spec = rc.ABLATIONS[name]
    for key in ("prereg_arm", "mechanism", "seen_in_training", "tests", "applies_at", "requires"):
        assert spec.get(key), name + " carries no " + key
    assert spec["applies_at"] == "model"
    with pytest.raises(SystemExit) as ei:           # parsed by argparse, then dies on the NEXT required thing
        rc.main(["--ablate", name, "--action-units", "steer"])
    assert "--out is required" in str(ei.value), repr(ei.value)


def test_cli_accepts_both_together_and_help_lists_both(capsys):
    with pytest.raises(SystemExit) as ei:
        rc.main(["--ablate", MAP, BOX, "--action-units", "steer"])
    assert "--out is required" in str(ei.value)
    with pytest.raises(SystemExit):
        rc.main(["--help"])
    out = capsys.readouterr().out
    assert MAP in out and BOX in out and "R9-MAP2-OFF" in out and "R9-BOX-OFF" in out


# --------------------------------------------------------------------------- #
# (c) refusal on a build without the wire                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [MAP, BOX])
def test_refuses_on_a_K0_like_model_without_either_wire(name):
    cfg, _on, off = _models()
    with pytest.raises(SystemExit) as ei:
        _ablate(off, cfg, name)
    assert "needs a build with" in str(ei.value) and "Refusing" in str(ei.value)


@pytest.mark.parametrize("name,only", [(MAP, "box"), (BOX, "map")])
def test_refuses_when_only_the_OTHER_wire_is_built(name, only):
    cfg, on, _off = _models(map_on=(only == "map"), box_on=(only == "box"))
    with pytest.raises(SystemExit) as ei:
        _ablate(on, cfg, name)
    assert "needs a build with" in str(ei.value)


# --------------------------------------------------------------------------- #
# W-MAP-2                                                                      #
# --------------------------------------------------------------------------- #
def test_map_reachability_off_equals_not_built_and_the_rest_is_untouched():
    cfg, on, off = _models()
    batch = RG._batch(cfg)
    _open_map(on)
    full, base = _ev(on, batch), _ev(off, batch)
    state, rec = _ablate(on, cfg, MAP)
    abl = _ev(on, batch)
    # (a) the wire is reachable: opening it moved the output, and ablating it moved the output back
    moved_by_wire = _moved(full, base, ignore=(COUNTER,))
    assert moved_by_wire, "a non-zero map_chan_proj must move the planner/tactical output (else the test proves nothing)"
    moved_by_ablation = _moved(full, abl)
    assert moved_by_ablation and set(moved_by_ablation) <= set(moved_by_wire) | {COUNTER}, moved_by_ablation
    assert "INERT" not in rec["per_ablation"][MAP]
    # (d) off == not built: every output equals the model that never built W-MAP-2, bit for bit
    # (the scene-token COUNT differs only because this model still builds W-BOX, whose zero-gate tokens are appended)
    assert _moved(abl, base, ignore=(COUNTER,)) == [], _moved(abl, base, ignore=(COUNTER,))
    # (d) the rest is untouched: outputs the wire does not reach are bit-identical to the un-ablated run
    unmoved = sorted(set(full) - set(moved_by_ablation))
    assert unmoved, "every output moved: the ablation reaches more than the wire"
    print("[r9_map_chan_zero] moved:", moved_by_ablation, "untouched:", unmoved[:8])


def test_map_identity_at_the_zero_init_changes_nothing_bit_for_bit():
    cfg, on, _off = _models()
    batch = RG._batch(cfg)
    before = _ev(on, batch)
    _state, rec = _ablate(on, cfg, MAP)
    assert "INERT" in rec["per_ablation"][MAP]
    assert _moved(before, _ev(on, batch)) == []


# --------------------------------------------------------------------------- #
# W-BOX                                                                        #
# --------------------------------------------------------------------------- #
def test_box_reachability_off_equals_not_built_and_the_rest_is_untouched():
    cfg, on, off = _models()
    batch = RG._batch(cfg)
    _open_box(on)
    full, base = _ev(on, batch), _ev(off, batch)
    state, rec = _ablate(on, cfg, BOX)
    abl = _ev(on, batch)
    moved_by_wire = _moved(full, base, ignore=(COUNTER,))
    assert moved_by_wire, "non-zero box gates must move the planner/tactical output (else the test proves nothing)"
    moved_by_ablation = _moved(full, abl)
    assert moved_by_ablation, "ablating the open gates must move the output"
    assert len(rec["per_ablation"][BOX]["gates_after"]) == 3 and "INERT" not in rec["per_ablation"][BOX]
    # off == not built: only the scene-token COUNT differs (the tokens are still appended, they just carry weight 0)
    assert _moved(abl, base, ignore=(COUNTER,)) == [], _moved(abl, base, ignore=(COUNTER,))
    unmoved = sorted(set(full) - set(moved_by_ablation))
    assert unmoved, "every output moved: the ablation reaches more than the wire"
    print("[r9_box_tok_zero] moved:", moved_by_ablation, "untouched:", unmoved[:8])


def test_box_identity_at_the_zero_init_changes_nothing_bit_for_bit():
    cfg, on, _off = _models()
    batch = RG._batch(cfg)
    before = _ev(on, batch)
    _state, rec = _ablate(on, cfg, BOX)
    assert "INERT" in rec["per_ablation"][BOX]
    assert _moved(before, _ev(on, batch)) == []


def test_each_ablation_leaves_the_other_wire_alone():
    cfg, on, off = _models()
    batch = RG._batch(cfg)
    _open_map(on)
    _open_box(on)
    _ablate(on, cfg, BOX)                       # box off, map still open: must still differ from the not-built model
    assert _moved(_ev(on, batch), _ev(off, batch), ignore=(COUNTER,)), "box ablation also killed W-MAP-2"
    cfg, on, off = _models()
    _open_map(on)
    _open_box(on)
    _ablate(on, cfg, MAP)                       # map off, box still open
    assert _moved(_ev(on, batch), _ev(off, batch), ignore=(COUNTER,)), "map ablation also killed W-BOX"


# --------------------------------------------------------------------------- #
# r9_drv_critic_zero (the refcv8 drivable critic's selection weight)           #
# --------------------------------------------------------------------------- #
DRV = "r9_drv_critic_zero"


class _DrvStandIn(torch.nn.Module):
    """Stand-in carrying a REAL DrivableCritic as `r8_drv` (the only thing the knockout reads)."""

    def __init__(self, w=0.21958, with_critic=True):
        super().__init__()
        if with_critic:
            from tanitad.refs import refcv8_conditioning as r8c
            self.r8_drv = r8c.DrivableCritic()
            with torch.no_grad():
                self.r8_drv.w.fill_(w)


def test_drv_registry_entry_and_enumeration():
    spec = rc.ABLATIONS[DRV]
    for key in ("prereg_arm", "mechanism", "seen_in_training", "tests", "applies_at", "requires"):
        assert spec.get(key), DRV + " carries no " + key
    assert spec["prereg_arm"] == "R9-DRV-OFF" and spec["applies_at"] == "model"
    assert rc.R9_WIRE_KNOCKOUTS == (MAP, BOX, DRV)
    with pytest.raises(SystemExit) as ei:
        rc.main(["--ablate", DRV, "--action-units", "steer"])
    assert "--out is required" in str(ei.value)


def test_drv_refuses_on_a_model_without_r8_drv():
    for m in (_DrvStandIn(with_critic=False), RG._rig(attach=False, ns=RG._ns(False, False), critic=False)[1]):    # the second is a REAL model built without the critic
        with pytest.raises(SystemExit) as ei:
            rc.apply_r9_wire_knockout(m, DRV)
        assert "needs a build with" in str(ei.value) and "Refusing" in str(ei.value)


def test_drv_zeroes_w_and_records_it():
    m = _DrvStandIn(0.21958)
    ev = rc.apply_r9_wire_knockout(m, DRV)
    assert float(m.r8_drv.w) == 0.0 and ev["w_after"] == 0.0
    assert abs(ev["w_before"] - 0.21958) < 1e-6 and "INERT" not in ev and "effect" in ev


def test_drv_inert_when_w_is_already_zero():
    m = _DrvStandIn(0.0)
    ev = rc.apply_r9_wire_knockout(m, DRV)
    assert "INERT" in ev and float(m.r8_drv.w) == 0.0


def test_drv_analytic_identity_selection_is_the_no_critic_selection():
    """w == 0 => dterm is exactly zeros for finite inputs => `blended + dterm` is bit-equal to `blended` and the argmax is unchanged."""
    m = _DrvStandIn(0.21958)
    g = torch.Generator().manual_seed(0)
    feat = torch.rand(3, 7, m.r8_drv.n_feats, generator=g)
    paths = torch.randn(3, 7, 12, 2, generator=g) * 10.0
    blended = torch.randn(3, 7, generator=g)
    _, dterm_on = m.r8_drv(feat, paths)
    assert not torch.equal(blended + dterm_on, blended)          # the critic term is live before the knockout
    rc.apply_r9_wire_knockout(m, DRV)
    _, dterm = m.r8_drv(feat, paths)
    assert torch.equal(dterm, torch.zeros_like(dterm)) and bool((dterm == 0).all())
    assert torch.equal(blended + dterm, blended)
    assert torch.equal((blended + dterm).argmax(1), blended.argmax(1))
