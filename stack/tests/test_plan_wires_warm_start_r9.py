"""refcv9 FULL-WARM, C8 -- the FULL model's ``--init-from`` admits the plan wires' DECLARED new keys through the ONE exemption list.

THE DEFECT (MEASURED 2026-11-03 on Thor CPU, ``warmstart_probe.py`` at 23fbf8f, E2 source ckpt_ema_48852.pt): ``plan_wires_flag.attach`` runs BEFORE the
warm start (``refc_v3_train.py`` attach :10093, ``warm_start_from`` :11157), so K0 + banded + ``--r9-plan-map-channels`` was REFUSED on the 2
``_perception.bev_pool.map_chan_proj.*`` keys, and K0 + ``--r9-plan-box-tokens`` on 13 (``core.box_tok_r9.*`` + one zero-init ``box_gate`` per consuming
planner / tactical layer). THE FIX: ``refcv9_perc_rig.plan_wires_exemptions`` in ``refcv9_warm_exemptions`` (full model only; the rig builds no planner).

Proved on the tiny REAL planner model of ``test_plan_wires_r9_model._rig`` (agent head, cross-agent decoder, tactical v6 decoder, 10 cm banded map, perception
branch on the pooled BEV):
  ADMIT   per lever (map, box, both): the lever-OFF build's state warm-starts the lever-ON build; the record names exactly the C8 keys; every other key is
          the source's bytes; the new keys keep their own init (and the gates / projection are ZERO, so step 0 is the source's planner)
  RED     the C8 entry deleted from the list (MUTATION) -> REFUSED; one gate withheld -> REFUSED; an undeclared extra module with C8 on -> REFUSED
  OFF     both wires off: ``plan_wires_exemptions`` is ``()`` and the shared list is the banded head's alone
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_map_only_ft as R  # noqa: E402
import test_plan_wires_r9_model as PWM  # noqa: E402     (_rig / _ns: the tiny REAL model with C8's wires)
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import refcv8_train as r8train  # noqa: E402

WIRES = {"map": dict(map_on=True, box_on=False), "box": dict(map_on=False, box_on=True), "both": dict(map_on=True, box_on=True)}
BANDED = ("_map_hires.banded.",)                      # built in source AND build here (the rig's map is banded): admitted, nothing missing under it

_SRC = {}


@pytest.fixture(scope="module", autouse=True)
def _one_source_checkpoint_for_the_module():
    """ONE lever-OFF source file for the module (deleted at teardown)."""
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="c8warm_src_"))
    _, off, _ = PWM._rig(attach=False, seed=0)
    R._save_source(off, d / "ema.pt")
    _SRC["path"], _SRC["state"] = d / "ema.pt", {k: v.detach().clone() for k, v in off.state_dict().items()}
    yield
    _SRC.clear()
    shutil.rmtree(d, ignore_errors=True)


def _on(wires: str, seed: int = 1):
    """The lever-ON build, a DIFFERENT init from the source's (nothing is shared by luck)."""
    _, model, _ = PWM._rig(ns=PWM._ns(**WIRES[wires]), seed=seed)
    return model


def _expected(model, wires: str) -> set:
    """The C8 keys, spelled from the STRUCTURE (independently of the function under test)."""
    out = set()
    if WIRES[wires]["map_on"]:
        out.add("_perception.bev_pool.map_chan_proj.")
    if WIRES[wires]["box_on"]:
        out.add("core.box_tok_r9.")
        out |= {f"core.decoder.layers.{i}.box_gate" for i, ly in enumerate(model.core.decoder.layers) if getattr(ly, "cross_agent", None) is not None}
        out |= {f"tac_decoder_v6.layers.{i}.box_gate" for i in range(len(model.tac_decoder_v6.layers))}
    return out


def _under(k, prefixes):
    return any(k.startswith(q) for q in prefixes)


@pytest.mark.parametrize("wires", sorted(WIRES))
def test_ADMIT_the_wires_on_full_model_warm_starts_from_the_wires_off_state(wires):
    on = _on(wires)
    want = _expected(on, wires)
    assert set(rig.plan_wires_exemptions(on)) == want
    pre = {k: v.detach().clone() for k, v in on.state_dict().items()}
    rec = r8train.warm_start_from(on, str(_SRC["path"]))
    d = rec["r9_declared_new"]
    assert set(d["new_prefixes"]) == want | set(BANDED)
    assert d["dropped_source_keys"] == []
    src, post = _SRC["state"], on.state_dict()
    n_new = 0
    for k, v in post.items():
        if _under(k, want):
            n_new += 1
            assert k not in src and torch.equal(v, pre[k]), k                  # a declared new key: absent from the source, untouched by the warm start
        else:
            assert k in src and torch.equal(v, src[k]), k                      # everything else is the source's bytes
    assert d["n_new_keys"] == n_new > 0 and rec["n_loaded"] + n_new == len(post) and rec["n_new_refcv8_keys"] == 0
    if WIRES[wires]["box_on"]:                                                 # the zero gates: step 0 reads the source's planner
        gates = [post[k] for k in want if k.endswith(".box_gate")]
        assert gates and all(float(g.abs().sum()) == 0.0 for g in gates)
    if WIRES[wires]["map_on"]:                                                 # the zero-init 1x1: nothing is added to the pool at step 0
        assert float(sum(post[k].abs().sum() for k in post if k.startswith("_perception.bev_pool.map_chan_proj."))) == 0.0


@pytest.mark.parametrize("wires", sorted(WIRES))
def test_RED_MUTATION_the_c8_entry_deleted_from_the_list_refuses(wires, monkeypatch):
    real = rig.plan_wires_exemptions
    monkeypatch.setattr(rig, "plan_wires_exemptions", lambda m: ())
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        r8train.warm_start_from(_on(wires), str(_SRC["path"]))
    monkeypatch.setattr(rig, "plan_wires_exemptions", real)                    # the control: un-mutated, the same call is ADMITTED
    assert r8train.warm_start_from(_on(wires), str(_SRC["path"]))["r9_declared_new"]["n_new_keys"] > 0


def test_RED_MUTATION_one_gate_withheld_refuses(monkeypatch):
    real = rig.plan_wires_exemptions
    on = _on("box")
    victim = sorted(k for k in real(on) if k.endswith(".box_gate"))[-1]
    monkeypatch.setattr(rig, "plan_wires_exemptions", lambda m: tuple(q for q in real(m) if q != victim))
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        r8train.warm_start_from(on, str(_SRC["path"]))


@pytest.mark.parametrize("wires", sorted(WIRES))
def test_RED_an_undeclared_extra_module_is_refused_with_the_wires_on(wires):
    on = _on(wires)
    on.core.r9x_undeclared = torch.nn.Linear(2, 2)
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        r8train.warm_start_from(on, str(_SRC["path"]))


def test_OFF_both_wires_off_is_empty_and_the_shared_list_is_the_banded_heads_alone():
    _, off, _ = PWM._rig(attach=False, seed=0)
    assert rig.plan_wires_exemptions(off) == ()
    new, drop, widen = rig.refcv9_warm_exemptions(off)
    assert new == BANDED and drop == ("_map_hires.refine.cls.",) and widen is None
    rec = r8train.warm_start_from(off, str(_SRC["path"]))
    assert rec["r9_declared_new"]["n_new_keys"] == 0 and rec["n_loaded"] == len(off.state_dict())


def test_ONE_the_shared_list_consults_the_c8_function():
    assert "plan_wires_exemptions(model)" in inspect.getsource(rig.refcv9_warm_exemptions)
