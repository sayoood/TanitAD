"""Standalone tests for the bake-off harness (Arch backlog #2).

Fast + CPU-only + deterministic. The logic tests use a synthetic ``run_fn`` that
returns crafted ``GateReport``s (so matrix / CI / table logic is validated in
milliseconds); one heavier test drives a REAL ``WorldModel(smoke_config)`` latent
path through the actual D1-D3 gate runner to prove end-to-end wiring. No test
asserts a gate PASS on untrained latents — mechanics only (P8).

Run: ``pytest <package>/tests``  (needs an editable ``tanitad`` install).
"""

from __future__ import annotations

import math

import pytest
import torch

from tanitad.config import StackConfig, base250_config, smoke_config
from tanitad.eval.gates import GateReport
from tanitad.models.fourbrain import WorldModel

# Import the harness module (co-located next to this tests/ dir on the author
# machine; ``tanitad.eval.bakeoff`` after integration).
try:
    from tanitad.eval import bakeoff as bk
except Exception:                                             # pragma: no cover
    import importlib.util
    import pathlib
    import sys
    _p = pathlib.Path(__file__).resolve().parents[1] / "tanitad_bakeoff.py"
    _spec = importlib.util.spec_from_file_location("tanitad_bakeoff", _p)
    bk = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = bk          # dataclass resolves __module__ here
    _spec.loader.exec_module(bk)


# --------------------------------------------------------------------------- #
# Synthetic gate reports                                                       #
# --------------------------------------------------------------------------- #
def _report(gate: str, admissible: bool, passed: bool, **metrics) -> GateReport:
    rows = [{"row": "I1", "pass": admissible, "value": 0.99}]
    return GateReport(gate, f"{gate} claim", admissible, passed and admissible,
                      rows, dict(metrics), {}, {}, [] if admissible else ["I1"],
                      f"{gate} synthetic")


def _stub_run_fn(reports_by_gate):
    """Return a run_fn that always yields the given per-gate reports + fixed params."""
    def _fn(cfg, seed):
        return bk.RunOutput(reports=dict(reports_by_gate), n_params=1234)
    return _fn


# --------------------------------------------------------------------------- #
# G-AI1 registry integrity                                                     #
# --------------------------------------------------------------------------- #
def test_registry_g_ai1_every_lever_names_a_gate_and_hypothesis():
    for lever in bk.default_levers() + bk.planned_levers():
        assert lever.gates, f"{lever.name} names no falsifying gate (G-AI1)"
        assert all(g in bk.KNOWN_GATES for g in lever.gates), \
            f"{lever.name} targets an unknown gate {lever.gates}"
        assert lever.hypothesis and lever.hypothesis != "—", \
            f"{lever.name} names no hypothesis"
        assert lever.isolates, f"{lever.name} has no one-factor description"
        assert lever.rationale, f"{lever.name} has no rationale/evidence"


def test_lever_names_unique():
    names = [l.name for l in bk.default_levers() + bk.planned_levers()]
    assert len(names) == len(set(names)), f"duplicate lever names in {names}"


# --------------------------------------------------------------------------- #
# One-factor (OFAT) isolation                                                  #
# --------------------------------------------------------------------------- #
def test_baseline_is_identity():
    cfg = base250_config()
    assert bk.lever_diff(cfg, bk.baseline_lever().build(cfg)) == []


def test_default_levers_are_one_factor():
    # base250 has tactical_pred + h15 ON, so every default lever really changes
    # something. Each variant must differ from base in exactly its declared field(s).
    base = base250_config()
    for lever in bk.default_levers():
        variant = lever.build(base)
        diff = bk.lever_diff(base, variant)
        assert diff, f"{lever.name} changed nothing on base250"
        for path in diff:
            assert any(path == f or path.startswith(f + ".") for f in lever.fields), \
                f"{lever.name} changed undeclared field {path} (declared {lever.fields})"
        for f in lever.fields:
            assert any(path == f or path.startswith(f + ".") for path in diff), \
                f"{lever.name} declared {f} but did not change it"


def test_lever_diff_detects_nested_and_none():
    base = base250_config()
    v = base250_config()
    v.predictor.window = 4                                    # nested scalar
    assert bk.lever_diff(base, v) == ["predictor.window"]
    v2 = base250_config()
    v2.tactical_pred = None                                   # dataclass -> None
    assert bk.lever_diff(base, v2) == ["tactical_pred"]


def test_build_does_not_mutate_base():
    base = base250_config()
    before = base.predictor.residual
    bk.default_levers()[0].build(base)                        # residual_off
    assert base.predictor.residual == before, "build mutated the shared base cfg"


# --------------------------------------------------------------------------- #
# Planned levers must refuse to run                                            #
# --------------------------------------------------------------------------- #
def test_planned_levers_raise_with_pointer():
    for lever in bk.planned_levers():
        assert lever.implemented is False
        with pytest.raises(bk.PlannedLeverError) as ei:
            lever.build(StackConfig())
        assert lever.name in str(ei.value) or "not in the stack" in str(ei.value)


def test_applied_default_configs_build_a_real_model():
    # Every runnable lever must yield a config that actually instantiates
    # (catches e.g. an invalid readout grid). Use smoke-scale to stay fast;
    # skip levers that are no-ops on the smoke base (tactical already None).
    base = smoke_config()
    for lever in bk.default_levers():
        variant = lever.build(base)
        if bk.lever_diff(base, variant):                     # only if it changed smth
            WorldModel(variant)                              # must not raise


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def test_mean_ci95_basic():
    mean, ci = bk.mean_ci95([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert ci == pytest.approx(1.96 * 1.0 / math.sqrt(3), rel=1e-6)


def test_mean_ci95_single_seed_has_nan_ci():
    mean, ci = bk.mean_ci95([5.0])
    assert mean == 5.0 and math.isnan(ci)


def test_mean_ci95_drops_nonfinite_and_empty():
    mean, ci = bk.mean_ci95([1.0, float("nan"), 3.0])
    assert mean == pytest.approx(2.0)
    assert math.isnan(bk.mean_ci95([])[0])


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def test_run_bakeoff_aggregates_and_flags_blocked():
    levers = [bk.baseline_lever()]
    # baseline: D1 admissible+FAIL, D2 BLOCKED, D3 admissible+PASS
    run_fn = _stub_run_fn({
        "D1": _report("D1", True, False, **{"ade@1s": 2.0}),
        "D2": _report("D2", False, False, direction_acc=0.9),   # BLOCKED
        "D3": _report("D3", True, True, ratio=1.1),
    })
    res = bk.run_bakeoff(levers, run_fn, seeds=(0, 1, 2))
    assert len(res) == 1
    r = res[0]
    assert r.status["D1"] == "FAIL"
    assert r.status["D2"] == "BLOCKED"        # instruments failed -> no claim
    assert r.status["D3"] == "PASS"
    assert r.admissible["D2"] is False
    assert r.n_params == 1234
    assert r.metric["D1"][0] == pytest.approx(2.0)
    assert math.isfinite(r.metric["D1"][1])   # CI over 3 identical seeds -> 0.0


def test_run_bakeoff_records_planned_rows_without_running():
    ran = {"n": 0}

    def run_fn(cfg, seed):
        ran["n"] += 1
        return bk.RunOutput({"D1": _report("D1", True, True, **{"ade@1s": 0.3})}, 10)

    res = bk.run_bakeoff([bk.baseline_lever()] + bk.planned_levers(), run_fn,
                         seeds=(0,), gates=("D1",))
    planned = [r for r in res if r.planned]
    assert len(planned) == len(bk.planned_levers())
    assert all(p.status["D1"] == "PLANNED" and p.n_params is None for p in planned)
    # only the baseline ran (planned levers never invoke run_fn)
    assert ran["n"] == 1


def test_run_bakeoff_rejects_a_lever_that_lies_about_its_fields():
    def _two_factor(c):
        c.predictor.residual = False
        c.predictor.window = 2               # a SECOND, undeclared change
        return c
    bad = bk.Lever("bad", "H4", ("D1",), "claims one factor, changes two",
                   _two_factor, fields=("predictor.residual",), rationale="x")
    with pytest.raises(ValueError, match="not one-factor"):
        bk.run_bakeoff([bad], _stub_run_fn({}), seeds=(0,),
                       base_cfg=base250_config(), gates=("D1",))


# --------------------------------------------------------------------------- #
# Table                                                                        #
# --------------------------------------------------------------------------- #
def test_render_table_has_levers_doctrine_and_planned_section():
    run_fn = _stub_run_fn({
        "D1": _report("D1", True, False, **{"ade@1s": 2.0}),
        "D2": _report("D2", False, False, direction_acc=0.9),
        "D3": _report("D3", True, True, ratio=1.1),
    })
    res = bk.run_bakeoff([bk.baseline_lever()] + bk.planned_levers(), run_fn,
                         seeds=(0, 1))
    table = bk.render_table(res)
    assert "baseline" in table
    assert "BLOCKED" in table                       # D2 cell
    assert "measured" in table.lower()              # G-AI2 label
    assert "necessary, not sufficient" in table     # instrument doctrine
    assert "Planned levers" in table                # planned section
    assert "adaln_conditioning" in table
    # markdown header row present
    assert table.count("| lever |") >= 1


# --------------------------------------------------------------------------- #
# End-to-end: real WorldModel latent path through the actual gate runner        #
# --------------------------------------------------------------------------- #
def _smoke_run_fn(cfg, seed):
    """Build a real smoke WorldModel, encode a tiny synthetic batch, and score
    D1-D3 via the real gate runner. Untrained latents => gates BLOCKED/FAIL is
    EXPECTED; this proves wiring, not competence (P8)."""
    from tanitad.eval.gates import I2Input, run_d1, run_d2, run_d3

    torch.manual_seed(seed)
    model = WorldModel(cfg).eval()
    n, w = 8, cfg.predictor.window
    c, hw = cfg.encoder.in_channels, cfg.encoder.image_size
    frames_win = torch.randn(n, w, c, hw, hw)
    actions = torch.randn(n, w, cfg.predictor.action_dim) * 0.1
    next_frame = torch.randn(n, c, hw, hw)
    episode_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    disp = torch.randn(n, 2)

    with torch.no_grad():
        states_win = model.encode_window(frames_win)          # [N, W, S]
        preds = model.imagine(states_win, actions)            # {1:.., 2:..}
        z_prev = states_win[:, -1]
        z_true = model.encode(next_frame)                     # [N, S]
    z_imag = preds[1]
    i2 = I2Input(model.encode, frames_win[:, -1])

    r1 = run_d1(z_prev, disp, episode_ids, unit="camera", i2=i2)
    r2 = run_d2(z_prev, z_true, z_imag, disp, episode_ids, i2=i2)
    r3 = run_d3(z_prev, z_true, preds[2], disp, episode_ids, i2=i2)
    n_params = sum(p.numel() for p in model.parameters())
    return bk.RunOutput({"D1": r1, "D2": r2, "D3": r3}, n_params)


def test_end_to_end_smoke_real_model_and_gates():
    residual_off = next(l for l in bk.default_levers() if l.name == "residual_off")
    res = bk.run_bakeoff([bk.baseline_lever(), residual_off], _smoke_run_fn,
                         seeds=(0,), base_cfg=smoke_config())
    assert len(res) == 2
    for r in res:
        assert r.n_params and r.n_params > 0                  # MEASURED (G-AI2)
        for g in ("D1", "D2", "D3"):
            assert r.status[g] in ("PASS", "FAIL", "BLOCKED", "MIXED")
    # the harness verified residual_off is genuinely one-factor on the smoke base
    off = next(r for r in res if r.lever == "residual_off")
    assert off.fields_changed == ["predictor.residual"]
    # and the table renders over real reports
    assert "residual_off" in bk.render_table(res)
