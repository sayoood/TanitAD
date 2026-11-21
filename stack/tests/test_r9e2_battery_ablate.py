"""R9E2_ABLATE: the E2 per-frame battery rolls a checkpoint with ONE refcv9 planner-wire knockout (r9_map_chan_zero / r9_box_tok_zero).

THE CLASS (CLAUDE.md "built, tested and unreachable from its caller"): the knockouts were landed in `taniteval/tools/refcv3_arm.py` but the E2 roll never goes through
that tool.  Every proof here runs on a SMALL REAL `RefCV3Model` (the `test_plan_wires_r9_model` rig) through the REAL `Stager.stage` -- the call the child makes before
every inner forward -- and the REAL `refcv3_arm.apply_r9_wire_knockout` (loaded by file path; nothing re-implemented):

  (a) R9E2_ABLATE unset => the Stager does today's thing: outputs bit-identical to a never-staged model, census carries no `ablation` key.
  (b) set => the knockout is applied exactly ONCE (5 stage calls, one hook / one zeroing), the output moves when the wire weights are non-zero, and equals the model that
      NEVER BUILT the wire (the existing identity proof); INERT is recorded for a zero-init wire.
  (c) unknown name / list / a model without the wire / a tag without the ablation name => REFUSED.
  (d) the census (what the child writes as `r9e2_stage`) records name + the ablation's record.
  (e) `install_staging` (the child's hook) reads the env.
MUTATION (run by hand): delete the `apply_r9_wire_knockout` call in `Stager.stage` => (b), (d) go RED (see the hand-back).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
R9DIR = ROOT / "ops" / "r9e2_read"
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests"), str(R9DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_battery_r9_e2 as R9  # noqa: E402
import test_plan_wires_r9_model as RG  # noqa: E402  the rig

T = RG.T
MAP, BOX = "r9_map_chan_zero", "r9_box_tok_zero"
COUNTER = "tacv6_n_scene_mean"


def _models():
    cfg, off, _ = RG._rig(attach=False, ns=RG._ns(False, False))
    _, on, _ = RG._rig(ns=RG._ns(True, True))
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


def _open_map(m):
    with torch.no_grad():
        m._perception.bev_pool.map_chan_proj.proj.weight.fill_(0.5)


def _open_box(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.rsplit(".", 1)[-1] == "box_gate":
                p.fill_(0.5)


def _ctx():
    return types.SimpleNamespace(item={"pose_hist": torch.zeros(4, 3)}, index=None)


def _stage(st, model, n_calls=3):
    for _ in range(n_calls):                               # the proxy makes up to four inner forwards per window; each calls stage()
        st.stage(model, _ctx(), torch.zeros(2, 3, 8, 8))


def _mk(ablate):
    return R9.Stager([], episodes_of=None, ablate=ablate)


# (a) unset == today ------------------------------------------------------------------------------------------------------------------------------------------
def test_unset_is_todays_behaviour_bit_identical_and_census_unchanged(monkeypatch):
    monkeypatch.delenv(R9.ABLATE_ENV, raising=False)
    cfg, on, _ = _models()
    _open_map(on)
    _open_box(on)
    batch = RG._batch(cfg)
    before = _ev(on, batch)
    st = _mk(R9.ablate_from_env())
    assert st.ablate is None
    _stage(st, on)
    assert _moved(before, _ev(on, batch)) == []
    c = st.census()
    assert "ablation" not in c and set(c) == {"n_forwards", "n_stage_calls", "batch_sizes", "mechanisms_built", "order"} and c["n_forwards"] == 3
    monkeypatch.setenv(R9.ABLATE_ENV, "   ")
    assert R9.ablate_from_env() is None


# (b) set: applied once, through Stager.stage, moves the output, equals not-built ------------------------------------------------------------------------------
@pytest.mark.parametrize("name,opener", [(MAP, _open_map), (BOX, _open_box)])
def test_knockout_applied_once_through_stage_moves_output_and_equals_not_built(name, opener):
    cfg, on, off = _models()
    opener(on)
    batch = RG._batch(cfg)
    full, base = _ev(on, batch), _ev(off, batch)
    assert _moved(full, base, ignore=(COUNTER,)), "the open wire must move the output (else the test proves nothing)"
    st = _mk(name)
    st.stage(on, _ctx(), torch.zeros(2, 3, 8, 8))
    assert st.ablation is not None
    rec1 = st.ablation
    _stage(st, on, 4)                                      # 5 stage calls in all
    assert st.ablation is rec1 and st.n_forwards == 5       # applied exactly once: the record object is the first call's
    if name == MAP:
        assert len(on._perception.bev_pool.map_chan_proj._forward_hooks) == 1
    else:
        assert all(float(p.detach().reshape(-1)[0]) == 0.0 for n, p in on.named_parameters() if n.rsplit(".", 1)[-1] == "box_gate")
    abl = _ev(on, batch)
    assert _moved(full, abl), "the knockout must move the output of a model whose wire weights are non-zero"
    assert _moved(abl, base, ignore=(COUNTER,)) == []       # off == never built, bit for bit


@pytest.mark.parametrize("name", [MAP, BOX])
def test_zero_init_wire_is_recorded_INERT_and_changes_nothing(name):
    cfg, on, _ = _models()
    batch = RG._batch(cfg)
    before = _ev(on, batch)
    st = _mk(name)
    _stage(st, on, 1)
    assert "INERT" in st.ablation["record"]
    assert _moved(before, _ev(on, batch)) == []


# (c) refusals ------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["nope", "frames_blind", f"{MAP},{BOX}", f"{MAP} {BOX}"])
def test_unknown_name_or_list_refuses(bad):
    with pytest.raises(SystemExit) as ei:
        R9.ablate_from_env({R9.ABLATE_ENV: bad})
    assert "refusing" in str(ei.value)


@pytest.mark.parametrize("name", [MAP, BOX])
def test_model_without_the_wire_refuses_at_the_first_stage(name):
    _, _on, off = _models()
    with pytest.raises(SystemExit) as ei:
        _mk(name).stage(off, _ctx(), torch.zeros(2, 3, 8, 8))
    assert "needs a build with" in str(ei.value) and "Refusing" in str(ei.value)


@pytest.mark.parametrize("name,tag", [(MAP, "E2_K2_pf"), (BOX, "E2_K2abl_mapzero_pf"), (MAP, None), (BOX, "")])
def test_tag_without_the_ablation_name_refuses(name, tag):
    with pytest.raises(SystemExit) as ei:
        R9.ablate_tag_ok(name, tag)
    assert "baseline tag" in str(ei.value)


def test_matching_tag_and_unset_pass():
    R9.ablate_tag_ok(MAP, "E2_K2abl_mapzero_pf")
    R9.ablate_tag_ok(BOX, "E2_K2abl_boxzero_pf")
    R9.ablate_tag_ok(None, "E2_K2_pf")


def test_parent_main_refuses_a_baseline_tag_before_anything_runs(monkeypatch):
    for k in R9.ENV_PINS:
        monkeypatch.setenv(k, "x")
    monkeypatch.delenv("R8_NO_POSE_SYNC", raising=False)
    monkeypatch.setenv(R9.ABLATE_ENV, MAP)
    monkeypatch.setattr(R9, "install_thor_gate", lambda argv: None)
    with pytest.raises(SystemExit) as ei:
        R9.main(["--no-baselines", "--tag", "E2_K2_pf"])
    assert "baseline tag" in str(ei.value)


# (d) census --------------------------------------------------------------------------------------------------------------------------------------------------
def test_census_records_name_and_record_and_is_json_serialisable():
    cfg, on, _ = _models()
    _open_map(on)
    st = _mk(MAP)
    _stage(st, on, 2)
    c = json.loads(json.dumps(st.census()))                 # what roll_seed_r9_e2 writes into --out-json as r9e2_stage
    a = c["ablation"]
    assert a["requested"] == MAP and a["applied"]["name"] == MAP
    rec = a["applied"]["record"]
    assert rec["weight_abs_sum_before"] > 0.0 and "INERT" not in rec and "effect" in rec
    assert _mk(MAP).census()["ablation"] == {"requested": MAP, "applied": None}   # requested but never staged: visible, not silent


# (e) the child's hook reads the env --------------------------------------------------------------------------------------------------------------------------
def test_install_staging_reads_the_env(monkeypatch):
    ds = types.SimpleNamespace(episodes=[types.SimpleNamespace(episode_id=7)])

    def mk():
        return types.SimpleNamespace(L=types.SimpleNamespace(build_eval_dataset=lambda *a, **k: (ds, 0, {})), Refcv6Proxy=object)
    monkeypatch.delenv(R9.ABLATE_ENV, raising=False)
    assert R9.install_staging(mk(), mods=[]).ablate is None
    monkeypatch.setenv(R9.ABLATE_ENV, BOX)
    assert R9.install_staging(mk(), mods=[]).ablate == BOX
    monkeypatch.setenv(R9.ABLATE_ENV, "bogus")
    with pytest.raises(SystemExit):
        R9.install_staging(mk(), mods=[])


def test_one_implementation_apply_ablations_calls_the_shared_function():
    import inspect
    arm = R9.load_arm_tool()
    src = inspect.getsource(arm.apply_ablations)
    assert "apply_r9_wire_knockout" in src
    assert arm.R9_WIRE_KNOCKOUTS == (MAP, BOX, "r9_drv_critic_zero") and set(arm.R9_WIRE_KNOCKOUTS) <= set(arm.ABLATIONS)


# --------------------------------------------------------------------------- #
# r9_drv_critic_zero through the battery                                       #
# --------------------------------------------------------------------------- #
DRV = "r9_drv_critic_zero"


class _DrvModel(torch.nn.Module):
    def __init__(self, w=0.21958, with_critic=True):
        super().__init__()
        if with_critic:
            from tanitad.refs import refcv8_conditioning as r8c
            self.r8_drv = r8c.DrivableCritic()
            with torch.no_grad():
                self.r8_drv.w.fill_(w)


def test_drv_knockout_applied_once_through_stage_and_lands_in_census():
    m = _DrvModel()
    st = _mk(DRV)
    st.stage(m, _ctx(), torch.zeros(2, 3, 8, 8))
    rec1 = st.ablation
    _stage(st, m, 4)
    assert st.ablation is rec1 and st.n_forwards == 5            # applied exactly once
    assert float(m.r8_drv.w) == 0.0
    c = json.loads(json.dumps(st.census()))
    a = c["ablation"]
    assert a["requested"] == DRV and a["applied"]["name"] == DRV
    rec = a["applied"]["record"]
    assert abs(rec["w_before"] - 0.21958) < 1e-6 and rec["w_after"] == 0.0 and "INERT" not in rec


def test_drv_zero_w_is_recorded_INERT():
    st = _mk(DRV)
    _stage(st, _DrvModel(0.0), 1)
    assert "INERT" in st.ablation["record"]


def test_drv_model_without_critic_refuses_at_the_first_stage():
    with pytest.raises(SystemExit) as ei:
        _mk(DRV).stage(_DrvModel(with_critic=False), _ctx(), torch.zeros(2, 3, 8, 8))
    assert "needs a build with" in str(ei.value) and "Refusing" in str(ei.value)
    off = RG._rig(attach=False, ns=RG._ns(False, False), critic=False)[1]
    with pytest.raises(SystemExit):
        _mk(DRV).stage(off, _ctx(), torch.zeros(2, 3, 8, 8))


def test_drv_env_name_accepted_and_tag_guard():
    assert R9.ablate_from_env({R9.ABLATE_ENV: DRV}) == DRV
    for tag in ("E2_K2abl_drvzero_pf", "E2_K2abl_drv_critic_zero_pf"):
        R9.ablate_tag_ok(DRV, tag)
    for tag in ("E2_K2_pf", "E2_K2abl_mapzero_pf", None, ""):
        with pytest.raises(SystemExit) as ei:
            R9.ablate_tag_ok(DRV, tag)
        assert "baseline tag" in str(ei.value)
