"""EVERY residual predictor must start at a sane SCALE, in every frontier model.

WHY THIS EXISTS -- MEASURED, 2026-08-22.

`OperativePredictor.forward` computes ``out[k] = z_t + delta`` with ``delta``
from ``self.norm(...)`` -- a LayerNorm, so its output is O(1) PER DIM whatever
the latent's scale. v6's operative latent has ``mean|z| = 0.015581`` and moves
``0.000892`` per tick (stride-1 latents at the true dt=0.1s tick). A default-init
head therefore starts emitting a delta ~1000x larger than the movement it must
predict. v6F-SW-30k's own ``o5_step1`` was still **535x WORSE than predicting NO
CHANGE** at step 20,000, and rescaling the TRAINED heads could not rescue it --
the error fell monotonically to alpha=0, i.e. the learned delta direction carried
no usable signal at any scale.

WHY DOWN-SCALED AND NOT ZERO-INIT -- the correction the suite forced.
Zeroing an OUTPUT head sets ``dL/dh_last = W^T . dL/dout = 0``, stopping gradient
to the ENTIRE predictor body. 18 tests failed on the zero-init cut, and
``test_v6_staged.py::test_planner_surface_is_total`` named it exactly:
``predictor_op.intent_proj`` went invisible to the gradient probe. FiLM is
zero-init a few lines away because it is an INTERNAL modulation whose main path
is untouched; an output head is not the same object.

SCOPE: initialisation only. ``state_dict`` shapes are unchanged, so every
existing checkpoint loads byte-identically and no banked result moves.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.config import PredictorConfig
from tanitad.models.predictor import (RESIDUAL_HEAD_INIT_SCALE,
                                      OperativePredictor)

STATE_DIM = 2048
#: v6's MEASURED operative-latent scale and per-tick movement, so the test runs
#: in the regime the defect actually bit in rather than a convenient unit one.
LATENT_MAD = 0.015581
MOVEMENT_MAD = 0.000892


def _cfg(residual: bool = True, horizons=(1, 2, 4)) -> PredictorConfig:
    return PredictorConfig(d_model=1024, depth=2, n_heads=4, window=6,
                           horizons=tuple(horizons), action_dim=3,
                           residual=residual)


def _inputs(b: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    return (torch.randn(b, 6, STATE_DIM) * LATENT_MAD,
            torch.randn(b, 6, 3))


def test_initial_delta_is_comparable_to_the_movement_not_1000x_it():
    """THE INVARIANT. At init the residual correction must be the same ORDER as
    the movement it predicts -- not three orders above it."""
    p = OperativePredictor(_cfg(), STATE_DIM).eval()
    z, a = _inputs()
    with torch.no_grad():
        out = p(z, a)
    for k, v in out.items():
        d = float((v - z[:, -1]).abs().mean())
        assert d < 20 * MOVEMENT_MAD, (
            f"horizon {k}: initial |delta| = {d:.3e}, which is "
            f"{d / MOVEMENT_MAD:.0f}x the per-tick movement "
            f"({MOVEMENT_MAD:.3e}). A residual head that starts far above the "
            f"movement scale spends the run shrinking itself -- that left v6 "
            f"535x worse than the hold baseline after 20,000 steps.")


def test_gradient_still_reaches_the_predictor_body():
    """THE CORRECTION THE SUITE FORCED. Down-scaling must not become zeroing:
    a zeroed output head stalls gradient to blocks/in_proj/act_emb/intent_proj,
    which is what broke 18 tests on the first cut."""
    p = OperativePredictor(_cfg(), STATE_DIM)
    z, a = _inputs()
    p(z, a)[1].pow(2).mean().backward()
    for name, mod in (("in_proj", p.in_proj),
                      ("heads.1", p.heads["1"]),
                      ("blocks.0", p.blocks[0])):
        g = [q.grad for q in mod.parameters() if q.grad is not None]
        assert g and any(float(x.abs().sum()) > 0 for x in g), (
            f"no gradient reaches {name} -- the head is effectively zeroed and "
            f"the body cannot train")


def test_scale_constant_is_actually_applied():
    """A guard that cannot fail teaches nothing: compare against an unscaled
    head built from the same seed."""
    torch.manual_seed(7)
    scaled = OperativePredictor(_cfg(horizons=(1,)), STATE_DIM)
    torch.manual_seed(7)
    plain = OperativePredictor(_cfg(residual=False, horizons=(1,)), STATE_DIM)
    r = (float(scaled.heads["1"].weight.abs().mean())
         / float(plain.heads["1"].weight.abs().mean()))
    assert abs(r - RESIDUAL_HEAD_INIT_SCALE) < 1e-6, (
        f"residual head scale is {r:.2e}, expected "
        f"{RESIDUAL_HEAD_INIT_SCALE:.2e}")


def test_non_residual_path_is_untouched():
    """`residual=False` predicts the latent DIRECTLY -- shrinking that head
    would cripple it, so the scaling must be conditional on `residual`."""
    p = OperativePredictor(_cfg(residual=False, horizons=(1,)),
                           STATE_DIM).eval()
    z, a = _inputs()
    with torch.no_grad():
        out = p(z, a)[1]
    assert float(out.abs().mean()) > 0.01,         "non-residual head was down-scaled -- it predicts the latent directly"


@pytest.mark.parametrize("horizons", [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)])
def test_holds_for_every_horizon_set(horizons):
    p = OperativePredictor(_cfg(horizons=horizons), STATE_DIM).eval()
    z, a = _inputs(2)
    with torch.no_grad():
        out = p(z, a)
    assert set(out) == set(horizons)
    for k in horizons:
        d = float((out[k] - z[:, -1]).abs().mean())
        assert d < 20 * MOVEMENT_MAD, f"horizon {k}: |delta| {d:.3e}"


# --------------------------------------------------------------------------- #
# The same invariant, across the frontier models the audit covered.
# `stack/scripts/residual_scale_audit.py` enumerates the seams; these pin the
# ones whose `base` is a predicted STATE rather than an internal activation.
# --------------------------------------------------------------------------- #

#: MEASURED against the real banked DINOv3 field: mean|x| 0.2060, and it moves
#: 0.1021 per frame (49.5% of its magnitude). v6's operative latent moves only
#: 1.9%, which is why the SAME defect measured 580x there and 4.4x here --
#: severity scales with how STATIC the base is.
DINO_FIELD_MAD = 0.2060
DINO_FIELD_MOVEMENT = 0.1021


def test_refa_v1_token_field_predictor_starts_near_identity():
    """REF-A v1's TACTICAL predictor. Its docstring CLAIMED "a zero-action step
    is near-identity at init" while MEASURING 0.4497 -- 2.2x the field and 4.4x
    the movement. A docstring asserting an init property is not evidence."""
    import torch
    from tanitad.refs.refa_v1 import RefAV1Config, TokenFieldPredictor
    cfg = RefAV1Config()
    torch.manual_seed(0)
    p = TokenFieldPredictor(cfg, d=cfg.d_state, layers=2).eval()
    field = torch.randn(2, 32, cfg.d_state) * DINO_FIELD_MAD
    with torch.no_grad():
        d = float((p.step(field, torch.zeros(2, cfg.a_dim)) - field)
                  .abs().mean())
    assert d < 0.1 * DINO_FIELD_MOVEMENT, (
        f"zero-action |delta| = {d:.4f} = {d / DINO_FIELD_MOVEMENT:.1f}x the "
        f"movement it predicts. The docstring's near-identity claim is only "
        f"true if the head is down-scaled.")


def test_refa_v1_strategic_predictor_starts_near_identity():
    import torch
    from tanitad.refs.refa_v1 import RefAV1Config, StrategicSubspacePredictor
    cfg = RefAV1Config()
    torch.manual_seed(0)
    p = StrategicSubspacePredictor(cfg).eval()
    s0 = torch.randn(2, cfg.str_dim) * DINO_FIELD_MAD
    with torch.no_grad():
        out = p.rollout(s0, torch.zeros(2, 3, cfg.a_dim))
    d = float((out[:, 0] - s0).abs().mean())
    assert d < 0.05 * DINO_FIELD_MAD, (
        f"strategic rollout |delta| at init = {d:.6f} = "
        f"{d / DINO_FIELD_MAD:.3f}x the state")


def test_gradient_reaches_every_fixed_predictor_body():
    """The correction the v6 suite forced, held for the REF-A modules too:
    down-scaling must never become zeroing."""
    import torch
    from tanitad.refs.refa_v1 import (RefAV1Config, StrategicSubspacePredictor,
                                      TokenFieldPredictor)
    cfg = RefAV1Config()
    tf = TokenFieldPredictor(cfg, d=cfg.d_state, layers=2)
    f = torch.randn(2, 16, cfg.d_state) * DINO_FIELD_MAD
    tf.step(f, torch.randn(2, cfg.a_dim)).pow(2).mean().backward()
    assert float(tf.mix.weight.grad.abs().sum()) > 0,         "no gradient reaches TokenFieldPredictor.mix"

    sp = StrategicSubspacePredictor(cfg)
    s0 = torch.randn(2, cfg.str_dim) * DINO_FIELD_MAD
    sp.rollout(s0, torch.randn(2, 2, cfg.a_dim)).pow(2).mean().backward()
    assert float(sp.act.weight.grad.abs().sum()) > 0,         "no gradient reaches StrategicSubspacePredictor.act"


def test_run_config_records_the_init_scale():
    """⛔ The init regime arrives by ENV VAR, so nothing in ``args`` records it.

    MEASURED 2026-08-22: the v7-tiny `fixed` and `regress` arms differed ONLY in
    ``TANITAD_RESIDUAL_INIT_SCALE`` and their ``config.json`` files were
    identical on it — the single variable of a two-arm ablation was absent from
    both arms' own records and had to be reconstructed from the launch script.
    An env-var input that changes the model is a RUN FACT.
    """
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / \
        "scripts" / "train_v6_staged.py"
    tree = ast.parse(src.read_text(encoding="utf-8-sig"))
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "_run_config"), None)
    assert fn is not None, "_run_config vanished from train_v6_staged.py"
    keys = {k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
            for k in n.keys if isinstance(k, ast.Constant)}
    assert "residual_head_init_scale" in keys, (
        "_run_config no longer records `residual_head_init_scale`; a two-arm "
        "init ablation would again produce indistinguishable config.json files")


def test_banner_names_the_regression_arm():
    """The banner must SAY when the defect is deliberately reintroduced, so a
    regression arm is never mistaken for a normal run in a log."""
    import importlib
    import os
    import tanitad.models.predictor as P
    prev = os.environ.get("TANITAD_RESIDUAL_INIT_SCALE")
    try:
        os.environ["TANITAD_RESIDUAL_INIT_SCALE"] = "1.0"
        importlib.reload(P)
        assert P.RESIDUAL_HEAD_INIT_SCALE == 1.0
        assert "DEFECT" in P.residual_init_scale_banner().upper(), (
            "the banner does not flag a reintroduced defect")
        os.environ["TANITAD_RESIDUAL_INIT_SCALE"] = "1e-3"
        importlib.reload(P)
        assert "DEFECT" not in P.residual_init_scale_banner().upper()
    finally:
        if prev is None:
            os.environ.pop("TANITAD_RESIDUAL_INIT_SCALE", None)
        else:
            os.environ["TANITAD_RESIDUAL_INIT_SCALE"] = prev
        importlib.reload(P)


def test_rank_gate_capacity_arithmetic():
    """⛔ The O6 rank criterion must be able to RULE, or it is decoration.

    MEASURED 2026-08-22: v6F@20k has an effective rank of 5.86 against an
    `absolute_floor` of 64 — a COLLAPSED representation the gate would have
    FAILED. It reported INCONCLUSIVE for nine days instead, because
    `--spectrum-accum` defaults to 1 and one spectrum call sees only
    `batch * window` rows, so `rank_ceiling` was 47 against a `ceiling_min` of
    1024. A criterion that cannot rule at the configured settings is worse than
    no criterion, because the gate report looks populated.
    """
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "scripts"))
    import train_v6_staged as T

    assert T.O6_ADMISSIBLE_CEILING == 1024
    # the live v6F geometry: batch 8, window 6 -> 48 rows per spectrum call
    r = T.rank_gate_capacity(8, 6)
    assert r["rows_per_call"] == 48
    assert r["required_spectrum_accum"] == 22, (
        "v6F needed --spectrum-accum 22 to let the rank gate rule; it ran with 1")
    # v7-tiny's geometry
    assert T.rank_gate_capacity(4, 6)["required_spectrum_accum"] == 43
    # the default is what made every run blind on this criterion
    assert 1 < r["required_spectrum_accum"], (
        "if the default ever reaches the ceiling, drop this guard")


def test_rank_gate_warning_fires_on_the_default():
    """The blindness must be LOUD at startup, not discoverable in a post-mortem."""
    import io
    import sys
    import pathlib
    import contextlib
    from types import SimpleNamespace
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "scripts"))
    import train_v6_staged as T

    stack = SimpleNamespace(cfg=SimpleNamespace(
        predictor=SimpleNamespace(window=6), d_op=2048))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T._warn_rank_gate_unrulable(SimpleNamespace(batch=8, spectrum_accum=1),
                                    stack)
    out = buf.getvalue()
    assert "CANNOT RULE" in out, "the default must warn"
    assert "--spectrum-accum 22" in out, "the warning must name the fix"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T._warn_rank_gate_unrulable(SimpleNamespace(batch=8, spectrum_accum=22),
                                    stack)
    assert "CAN rule" in buf.getvalue(), "a sufficient setting must not warn"


def test_gate_receives_the_pooled_spectrum_not_one_batch():
    """⛔ --spectrum-accum must reach the GATE, not only the log.

    MEASURED 2026-08-22 on Thor: the `lewm` arm ran with --spectrum-accum 43
    and its stage_gate.json still reported ``n: 24`` /
    ``rank_ceiling 23 < 1024`` / INCONCLUSIVE — because the pooled reading was
    written to the log record while ``run_stage_gate`` was handed the
    SINGLE-BATCH ``spectrum_last``. Every run in this programme therefore
    reported INCONCLUSIVE on the one criterion that would have caught the
    representation collapse, no matter what the flag said.
    """
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "train_v6_staged.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "run_stage_gate"]
    assert calls, "run_stage_gate call vanished"
    real = [c for c in calls
            for kw in c.keywords if kw.arg == "spectrum"
            and "pooled" in ast.unparse(kw.value)]
    assert real, (
        "run_stage_gate is not handed the POOLED spectrum — the rank criterion "
        "cannot rule and will report INCONCLUSIVE forever")


def test_row_bank_renormalises_so_lambda_is_not_silently_changed():
    """⛔ Pooling rows must not change the EFFECTIVE loss weight.

    The Epps-Pulley statistic is deliberately not n-normalised, so its value
    scales with the row count. MEASURED 2026-08-22: 24 rows -> o6_sigreg 3.10,
    192 rows -> 46.3 (~15x). Without compensation, a row-count experiment is
    also a lambda experiment, and the lambda sweep is already known flat-to-
    harmful — which is exactly how the bank8/sub16 arms read WORSE than
    baseline for a reason that had nothing to do with estimator power.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "train_v6_staged.py").read_text(encoding="utf-8-sig")
    assert "o6_row_renorm" in src, "the renormalisation factor is not logged"
    assert "base_rows / float(z6.shape[0])" in src, (
        "the o6 term is not rescaled when the row bank enlarges n; the row "
        "count would silently multiply the effective lambda")
