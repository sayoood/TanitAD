"""K2DC (PREREG_K2DC_2026-11-12) -- the map-dose SHARE CONTROLLER (`tanitad/train/map_dose_ctl.py`), its trainer wiring and its
launch-gate rows. Expectations are LITERALS or independent recomputations from raw vectors, never expressions over the code
under test; every guard has a deliberate-regression arm that must go RED.

  A  ANALYTIC IDENTITY  synthetic gradient vectors: (norm, total, proj) -> A2 / AB / B2 EXACT; the solved k gives s(k) = target to 1e-9
  B  CLAMP              no root inside [LO, HI] -> the end whose share is nearest, `clamped` True (both ends)
  C  REFUSALS           every `pin_args` refusal; `_pin_refcv8` reaches it; OFF + a stray controller flag refuses
  D  RESUME             w_eff saved in EVERY ckpt writer and restored; ON + a ckpt without it REFUSES; OFF + one with it REFUSES
  E  TINY LOOP          the REAL tiny RefCV3 model + `compute_losses_v3`: a firing changes `model._w_map_hires`, the map term's weight on
                        the NEXT step equals it (loss(w_new) - loss(w_old) == (w_new - w_old) * map term), the share re-read at the new
                        weight equals the target; CONTROL (target OFF): the weight stays EXACTLY the initial value
  F  gs_* UNTOUCHED     the gradient-share rows are identical with the controller ON vs OFF up to the first firing (and differ after it)
  G  G-DVB              built == argv for all four flags; unbuilt / mismatched / stray controller each REFUSE by flag name
  H  WIRING             argparse -> pin -> build -> hop 1 / 2 / 3 -> checkpoint, pinned in the trainer source, each with a mutant
  I  LAUNCH GATE        static refusals, smoke_argv (exactly one firing), G-LIVE rows (no `dc_*` row = FAIL), cost exclusion
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import launch_gate as LG  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402
from tanitad.train import grad_share as GS  # noqa: E402
from tanitad.train import map_dose_ctl as M  # noqa: E402

TRAINER_SRC = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")


# =========================================================================== #
# helpers                                                                      #
# =========================================================================== #
def _dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def _vecs(seed, dim=9, cos=0.3):
    rnd = random.Random(seed)
    a1 = [rnd.gauss(0, 1) for _ in range(dim)]
    b = [cos * x + rnd.gauss(0, 1.5) for x in a1]
    return a1, b


def _probe_from(a1, b, w, group="trunk", prefix=M.PROBE_PREFIX):
    """What `grad_share.measure` would report on gradients a = w * a1 and b: |a|, |g|, <a,g> / |g|^2."""
    a = [w * x for x in a1]
    g = [x + y for x, y in zip(a, b)]
    t2 = _dot(g, g)
    return {f"{prefix}{group}_norm_map_hires": math.sqrt(_dot(a, a)), f"{prefix}{group}_norm_total": math.sqrt(t2),
            f"{prefix}{group}_proj_map_hires": _dot(a, g) / t2}


def _share_direct(a1, b, k):
    a = [k * x for x in a1]
    g = [x + y for x, y in zip(a, b)]
    return _dot(a, g) / _dot(g, g)


def _ns(**over):
    base = dict(map_hires="on", w_map_hires=25.0, grad_share_terms="tac_v6,map_hires", grad_share_every=250, log_every=50,
                steps=6000, map_hires_share_target=0.10, map_hires_share_every=250, map_hires_share_group="trunk",
                map_hires_w_bounds="1,25")
    base.update(over)
    return argparse.Namespace(**base)


# =========================================================================== #
# A  the analytic identity                                                     #
# =========================================================================== #
@pytest.mark.parametrize("seed,w0", [(0, 25.0), (1, 4.0), (2, 1.0), (3, 11.5)])
def test_A_recovering_A2_AB_B2_from_norm_total_proj_is_exact(seed, w0):
    a1, b = _vecs(seed)
    p = _probe_from(a1, b, w0)
    r = M.recover(p["dcp_trunk_norm_map_hires"], p["dcp_trunk_norm_total"], p["dcp_trunk_proj_map_hires"], w0)
    a = [w0 * x for x in a1]
    assert abs(r["A2"] - _dot(a, a)) < 1e-9 * _dot(a, a)
    assert abs(r["AB"] - _dot(a, b)) < 1e-9 * max(1.0, abs(_dot(a, b)))
    assert abs(r["B2"] - _dot(b, b)) < 1e-9 * _dot(b, b)
    assert abs(r["a1sq"] - _dot(a1, a1)) < 1e-9 * _dot(a1, a1) and abs(r["a1b"] - _dot(a1, b)) < 1e-9 * max(1.0, abs(_dot(a1, b)))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("target", [0.05, 0.10, 0.30, 0.60])
def test_A_the_solved_k_gives_the_target_share_to_1e9(seed, target):
    a1, b = _vecs(seed)
    w0 = 25.0
    p = _probe_from(a1, b, w0)
    ctl = M.MapDoseController(target, 250, "trunk", 1e-3, 1e3, w0)         # bounds wide enough to hold every root
    row = ctl.fire(p, 250)
    k = ctl.w_eff
    assert row["dc_clamped"] == 0.0 and 1e-3 < k < 1e3
    assert abs(_share_direct(a1, b, k) - target) < 1e-9, (seed, target, k)         # an independent recomputation from the raw vectors
    assert abs(row["dc_share_pred"] - target) < 1e-9
    assert row["dc_w_before"] == w0 and row["dc_w_eff"] == k and abs(row["dc_share_read"] - _share_direct(a1, b, w0)) < 1e-12


def test_A_the_share_read_at_the_current_weight_is_the_probes_proj():
    a1, b = _vecs(5)
    ctl = M.MapDoseController(0.1, 250, "trunk", 1, 25, 7.0)
    row = ctl.fire(_probe_from(a1, b, 7.0), 250)
    assert abs(row["dc_share_read"] - _share_direct(a1, b, 7.0)) < 1e-12
    # a second firing at the NEW weight reads the share the first predicted (the loop closes)
    row2 = ctl.fire(_probe_from(a1, b, ctl.w_eff), 500)
    assert abs(row2["dc_share_read"] - row["dc_share_pred"]) < 1e-9 and abs(ctl.w_eff - row["dc_w_eff"]) < 1e-9


# =========================================================================== #
# B  clamping                                                                  #
# =========================================================================== #
def test_B_clamps_to_the_hi_end_when_even_HI_is_below_the_target():
    a1, b = _vecs(0)
    hi_share = _share_direct(a1, b, 2.0)
    ctl = M.MapDoseController(min(0.99, hi_share + 0.2), 250, "trunk", 1.0, 2.0, 2.0)
    row = ctl.fire(_probe_from(a1, b, 2.0), 250)
    assert row["dc_clamped"] == 1.0 and ctl.w_eff == 2.0 and abs(row["dc_share_pred"] - hi_share) < 1e-9


def test_B_clamps_to_the_lo_end_when_even_LO_is_above_the_target():
    a1, b = _vecs(0)
    lo_share = _share_direct(a1, b, 5.0)
    assert lo_share > 0.0
    ctl = M.MapDoseController(lo_share / 2.0, 250, "trunk", 5.0, 50.0, 25.0)
    row = ctl.fire(_probe_from(a1, b, 25.0), 250)
    assert row["dc_clamped"] == 1.0 and ctl.w_eff == 5.0 and abs(row["dc_share_pred"] - lo_share) < 1e-9


def test_B_a_root_inside_is_not_clamped_and_a_non_finite_reading_holds_the_weight_and_reads_nan():
    a1, b = _vecs(1)
    ctl = M.MapDoseController(0.2, 250, "trunk", 1e-3, 1e3, 3.0)
    assert ctl.fire(_probe_from(a1, b, 3.0), 250)["dc_clamped"] == 0.0
    ctl2 = M.MapDoseController(0.2, 250, "trunk", 1.0, 25.0, 3.0)
    bad = {"dcp_trunk_norm_map_hires": 0.0, "dcp_trunk_norm_total": 1.0, "dcp_trunk_proj_map_hires": 0.0}   # a map term with a zero gradient
    row = ctl2.fire(bad, 250)
    assert ctl2.w_eff == 3.0 and math.isnan(row["dc_share_pred"]) and row["dc_clamped"] == 1.0              # the gate then FAILS it (pred != target)


def test_B_a_probe_row_without_the_map_term_refuses_rather_than_holding_a_dose_on_nothing():
    ctl = M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0)
    with pytest.raises(SystemExit, match="refusing to hold a dose on nothing"):
        ctl.fire({"dcp_trunk_norm_total": 1.0}, 250)


# =========================================================================== #
# C  refusals                                                                  #
# =========================================================================== #
def test_C_a_coherent_declaration_and_OFF_pass():
    M.pin_args(_ns())
    M.pin_args(_ns(map_hires_share_target=None))
    M.pin_args(_ns(grad_share_terms=None))                                  # None = every term, map_hires included
    M.pin_args(_ns(map_hires_share_group="layer2", map_hires_w_bounds="1,25"))
    M.pin_args(_ns(w_map_hires=1.0)); M.pin_args(_ns(w_map_hires=25.0, map_hires_w_bounds="25,25"))   # the bounds are inclusive


@pytest.mark.parametrize("over,needle", [
    (dict(map_hires="off"), "needs --map-hires on"),
    (dict(w_map_hires=0.0), "--w-map-hires > 0"),
    (dict(grad_share_terms="tac_v6"), "map_hires` in --grad-share-terms"),
    (dict(map_hires_share_target=0.0), "0 < target < 1"),
    (dict(map_hires_share_target=1.0), "0 < target < 1"),
    (dict(map_hires_share_target=float("nan")), "0 < target < 1"),
    (dict(map_hires_share_every=0), "must be >= 1"),
    (dict(map_hires_share_every=-5), "must be >= 1"),
    (dict(map_hires_share_every=7), "multiple of --log-every"),
    (dict(map_hires_share_every=6050), "never fire"),
    (dict(map_hires_w_bounds="0,25"), "map-hires-w-bounds"),
    (dict(map_hires_w_bounds="25,1"), "map-hires-w-bounds"),
    (dict(map_hires_w_bounds="1"), "map-hires-w-bounds"),
    (dict(map_hires_w_bounds="a,b"), "map-hires-w-bounds"),
    (dict(w_map_hires=30.0), "outside --map-hires-w-bounds"),
    (dict(w_map_hires=0.5), "outside --map-hires-w-bounds"),
    (dict(map_hires_share_group="layer9"), "not in"),
])
def test_C_every_refusal(over, needle):
    with pytest.raises(SystemExit, match=re.escape(needle)):
        M.pin_args(_ns(**over))


def test_C_a_controller_flag_without_the_target_refuses_and_the_defaults_do_not():
    M.pin_args(_ns(map_hires_share_target=None, map_hires_share_every=250, map_hires_share_group="trunk", map_hires_w_bounds="1,25"))
    for over in (dict(map_hires_share_every=100), dict(map_hires_share_group="layer2"), dict(map_hires_w_bounds="1,10")):
        with pytest.raises(SystemExit, match="declared in argv and never built"):
            M.pin_args(_ns(map_hires_share_target=None, **over))


def test_C_the_trainers_pin_reaches_pin_args_and_the_flags_parse_to_the_documented_defaults():
    p = T.build_parser()
    a = p.parse_args(["--arm", "hier", "--out", "x"])
    assert (a.map_hires_share_target, a.map_hires_share_every, a.map_hires_share_group, a.map_hires_w_bounds) == (None, 250, "trunk", "1,25")
    bad = p.parse_args(["--arm", "hier", "--out", "x", "--steps", "6000", "--map-hires-share-target", "0.1"])      # --map-hires off
    with pytest.raises(SystemExit, match=r"\[k2dc\]"):
        T._pin_refcv8(None, bad)                                             # the real entry point refuses through `_mdc.pin_args`


# =========================================================================== #
# D  resume                                                                    #
# =========================================================================== #
def _tiny_model_with_ctl(ctl):
    m = torch.nn.Module()
    m.core = torch.nn.Module()
    m.core.encoder = torch.nn.Linear(2, 2)
    m._w_map_hires = 25.0
    if ctl is not None:
        m._map_dose_ctl = ctl
    return m


def test_D_ckpt_extra_carries_w_eff_and_is_empty_when_OFF():
    assert M.ckpt_extra(_tiny_model_with_ctl(None)) == {}
    c = M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0)
    c.w_eff, c.n_fired = 7.25, 3
    ex = M.ckpt_extra(_tiny_model_with_ctl(c))
    assert set(ex) == {"map_dose_ctl"} and ex["map_dose_ctl"]["w_eff"] == 7.25 and ex["map_dose_ctl"]["n_fired"] == 3


def test_D_resume_restores_w_eff_and_refuses_a_missing_or_a_changed_state():
    c = M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0)
    c.w_eff, c.n_fired = 7.25, 3
    saved = {"step": 750, "model": {}, "map_dose_ctl": c.state_dict()}
    fresh = M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0)
    M.resume_guard(saved, fresh)
    assert fresh.w_eff == 7.25 and fresh.n_fired == 3                        # restored, not re-initialised to 25
    with pytest.raises(SystemExit, match="carries no 'map_dose_ctl'"):
        M.resume_guard({"step": 750, "model": {}}, M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0))     # ON + a ckpt without it
    with pytest.raises(SystemExit, match="no --map-hires-share-target"):
        M.resume_guard(saved, None)                                                                          # OFF + a ckpt with it
    M.resume_guard({"step": 1, "model": {}}, None)                                                           # OFF + OFF
    for other in (M.MapDoseController(0.2, 250, "trunk", 1, 25, 25.0), M.MapDoseController(0.1, 500, "trunk", 1, 25, 25.0),
                  M.MapDoseController(0.1, 250, "layer2", 1, 25, 25.0), M.MapDoseController(0.1, 250, "trunk", 1, 20, 20.0)):
        with pytest.raises(SystemExit, match="resume REFUSED"):
            M.resume_guard(saved, other)
    bad = dict(saved, map_dose_ctl=dict(saved["map_dose_ctl"], w_eff=99.0))
    with pytest.raises(SystemExit, match="outside"):
        M.resume_guard(bad, M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0))


def test_D_the_trainers_checkpoint_payload_carries_the_state_and_an_OFF_payload_is_the_tips_key_for_key():
    model, _cfg = _build_rig()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    args = argparse.Namespace(seed=0, batch=2)
    off = T._ckpt_payload(model, opt, 5, {"epoch": 0, "batch": 5}, 10, args, None, None)
    assert "map_dose_ctl" not in off
    ctl = M.MapDoseController(0.1, 250, "trunk", 1, 25, 25.0)
    ctl.w_eff = 3.5
    model._map_dose_ctl = ctl
    on = T._ckpt_payload(model, opt, 5, {"epoch": 0, "batch": 5}, 10, args, None, None)
    assert set(on) - set(off) == {"map_dose_ctl"} and on["map_dose_ctl"]["w_eff"] == 3.5
    assert TRAINER_SRC.count("**_mdc.ckpt_extra(model)") == 2                # the inline loop save + `_ckpt_payload` (which serves the re-save and --save-at)


# =========================================================================== #
# E / F  the tiny loop on the REAL tiny RefCV3 model                           #
# =========================================================================== #
def _build_rig():
    import test_map_only_ft as R
    return R._build()


def _batch(cfg):
    import test_map_only_ft as R
    return R._batch(cfg)


_BUF0 = {}


def _losses(model, batch, seed=7):
    """`compute_losses_v3` from a FIXED buffer state. MEASURED here: two identical calls differ by ~0.012 in the total because the training forward
    updates the in-place buffers `core.lat_log_prior` / `core.lon_log_prior` (every term this test reads is unchanged). The buffers are put back to
    their build-time values BEFORE the forward (in place, before the graph exists: after it the in-place write would invalidate the graph the caller
    differentiates), so every forward -- the probe's replay, the step's own, a re-read -- sees the same state."""
    st = _BUF0.setdefault(id(model), {n: b.detach().clone() for n, b in model.named_buffers()})
    with torch.no_grad():
        for n, b in model.named_buffers():
            if n in st:
                b.copy_(st[n])
    torch.manual_seed(seed)
    return T.compute_losses_v3(model, batch, "cpu", log_metrics=False)


def _run_loop(ctl, steps, *, w0=2.0, every_rows=(), lr=0.0, gs_rows=(), seed=7):
    """The trainer's own three hops around the REAL `compute_losses_v3`, on the real tiny model. -> (per-step records, the model)."""
    model, cfg = _build_rig()
    batch = _batch(cfg)
    model._w_map_hires = float(w0)
    model.train()
    if ctl is not None:
        model._map_dose_ctl = ctl
    groups = M.groups_for(model, ctl) if ctl is not None else {}
    gs_groups = GS.param_groups(model)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr) if lr else None
    recs = []
    for step in range(steps):
        replay = M.probe_before_forward(ctl, model, step + 1, groups, lambda: _losses(model, batch, seed), T.TRAJ_WEIGHT)
        losses = _losses(model, batch, seed)
        w_used = float(model._w_map_hires)
        gs = GS.measure(GS.term_tensors(losses, GS.weights_of(model, T.TRAJ_WEIGHT), ("map_hires",)), losses["loss"], gs_groups) \
            if (step + 1) in gs_rows else {}
        row = M.read_after_losses(ctl, model, losses, step + 1, groups, T.TRAJ_WEIGHT, replay)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        M.apply_weight(ctl, model, row)
        if opt is not None:
            opt.step()
        recs.append({"step": step + 1, "w_used": w_used, "w_after": float(model._w_map_hires), "row": row, "gs": gs,
                     "loss": float(losses["loss"]), "map": float(losses["map_hires"])})
    return recs, model, batch


def test_E_a_firing_changes_the_weight_and_the_map_term_reads_it_on_the_next_step():
    ctl = M.MapDoseController(0.30, 2, "trunk", 1e-3, 1e3, 2.0)
    recs, model, batch = _run_loop(ctl, 5, w0=2.0)                 # no optimiser: the model is the same at every step
    # rows 2 and 4 fire; steps 1, 3, 5 do not
    assert [bool(r["row"]) for r in recs] == [False, True, False, True, False]
    assert recs[0]["w_used"] == 2.0 and recs[1]["w_used"] == 2.0         # the firing step itself still ran at the OLD weight (acts from the next step)
    w1 = recs[1]["row"]["dc_w_eff"]
    assert w1 != 2.0 and recs[1]["w_after"] == w1 and recs[2]["w_used"] == w1          # the attribute the loss reads == the solved weight
    assert recs[2]["w_after"] == w1                                                      # a non-firing step leaves it exactly alone
    # the LOSS multiplies by it: total(w_new) - total(w_old) == (w_new - w_old) * (the unweighted map term), same params / same RNG
    model._w_map_hires = 2.0
    old = _losses(model, batch)
    model._w_map_hires = w1
    new = _losses(model, batch)
    d_loss = float(new["loss"]) - float(old["loss"])
    want = (w1 - 2.0) * float(new["map_hires"])
    assert abs(d_loss - want) < 1e-5 * max(1.0, abs(want)), (d_loss, want)
    # the share RE-READ at the new weight is the target (no optimiser step in between, so the graph is the one the controller solved on)
    losses = _losses(model, batch)
    row = GS.measure(GS.term_tensors(losses, GS.weights_of(model, T.TRAJ_WEIGHT), ("map_hires",)), losses["loss"], GS.param_groups(model))
    assert abs(row["gs_trunk_proj_map_hires"] - 0.30) < 2e-4, row["gs_trunk_proj_map_hires"]
    assert abs(recs[1]["row"]["dc_share_read"] - 0.30) > 1e-3                           # non-vacuous: the share at the INITIAL weight was not already the target
    assert abs(recs[1]["row"]["dc_share_pred"] - 0.30) < 1e-9


def test_E_CONTROL_target_OFF_the_weight_is_exactly_the_initial_value_at_every_step():
    recs, model, _ = _run_loop(None, 6, w0=2.0, lr=1e-4)
    assert all(r["w_used"] == 2.0 and r["w_after"] == 2.0 and r["row"] == {} for r in recs)
    assert model._w_map_hires == 2.0 and not hasattr(model, "_map_dose_ctl")


def test_E_a_firing_row_carries_the_five_logged_keys_and_the_probe_under_its_own_prefix():
    ctl = M.MapDoseController(0.30, 1, "trunk", 1e-3, 1e3, 2.0)
    recs, _, _ = _run_loop(ctl, 1, w0=2.0)
    row = recs[0]["row"]
    assert {"dc_w_before", "dc_w_eff", "dc_share_read", "dc_share_pred", "dc_clamped"} <= set(row)
    assert {"dcp_trunk_norm_map_hires", "dcp_trunk_norm_total", "dcp_trunk_proj_map_hires"} <= set(row)
    assert not any(k.startswith("gs_") for k in row)


def test_F_gs_rows_are_identical_with_the_controller_ON_vs_OFF_up_to_the_first_firing_and_differ_after_it():
    timing = {"gs_measure_s", "gs_pass_s_max"}
    on, _, _ = _run_loop(M.MapDoseController(0.30, 2, "trunk", 1e-3, 1e3, 2.0), 3, w0=2.0, lr=1e-4, gs_rows=(1, 3))
    off, _, _ = _run_loop(None, 3, w0=2.0, lr=1e-4, gs_rows=(1, 3))
    strip = lambda r: {k: ("nan" if v != v else v) for k, v in r.items() if k not in timing}   # noqa: E731  (a NaN group reads NaN in both arms)
    assert strip(on[0]["gs"]) == strip(off[0]["gs"]) and on[0]["gs"]                 # row 1: before the first firing (row 2): bit-identical
    assert on[0]["loss"] == off[0]["loss"] and on[1]["loss"] == off[1]["loss"]       # the firing step itself trained identically
    assert strip(on[2]["gs"]) != strip(off[2]["gs"])                                 # row 3: after it the weight differs, so the reading does (non-vacuous)
    assert on[2]["gs"]["gs_trunk_proj_map_hires"] != off[2]["gs"]["gs_trunk_proj_map_hires"]


def test_E_a_model_weight_that_drifted_from_the_controllers_w_eff_refuses_at_the_next_reading():
    ctl = M.MapDoseController(0.30, 1, "trunk", 1e-3, 1e3, 2.0)
    model, cfg = _build_rig()
    model._w_map_hires = 3.0                                                         # NOT w_eff (2.0)
    model.train()
    losses = _losses(model, _batch(cfg))
    with pytest.raises(SystemExit, match="probe would be read at one weight"):
        M.read_after_losses(ctl, model, losses, 1, M.groups_for(model, ctl), T.TRAJ_WEIGHT, None)


def test_E_the_controller_group_must_exist_on_the_built_model():
    model, _ = _build_rig()
    ctl = M.MapDoseController(0.1, 250, "layer2", 1, 25, 2.0)
    assert M.groups_for(model, ctl)["layer2"]                                        # the tiny model is a timm resnet18: it has net.layer2
    plain = _tiny_model_with_ctl(None)
    with pytest.raises(SystemExit, match="no trainable parameter"):
        M.groups_for(plain, ctl)


# =========================================================================== #
# G  G-DVB                                                                     #
# =========================================================================== #
def test_G_the_four_flags_are_registered_built_levers_and_the_registry_covers_the_parser():
    for d in M.FLAGS:
        assert dvb.REGISTRY[d].kind == "built", d
    assert dvb.coverage(T.build_parser()) == []


def test_G_built_equals_argv_and_each_mismatch_refuses_by_flag_name():
    args = _ns()
    good = _tiny_model_with_ctl(M.controller_for(args))
    for d in M.FLAGS:
        assert dvb.REGISTRY[d].check(good, args) == [], d
    # declared, never built (the attach hop deleted) -> every flag's entry reports; the target's names itself
    bare = _tiny_model_with_ctl(None)
    got = dvb.REGISTRY["map_hires_share_target"].check(bare, args)
    assert len(got) == 1 and "never built" in got[0].why and got[0].lever == "--map-hires-share-target"
    # a controller built with another value: ONLY that flag's entry reports
    for over, dest in ((dict(map_hires_share_every=500), "map_hires_share_every"), (dict(map_hires_share_target=0.2), "map_hires_share_target"),
                       (dict(map_hires_share_group="layer2"), "map_hires_share_group"), (dict(map_hires_w_bounds="1,20", w_map_hires=20.0), "map_hires_w_bounds")):
        m = _tiny_model_with_ctl(M.controller_for(_ns(**over)))
        for d in M.FLAGS:
            n = len(dvb.REGISTRY[d].check(m, args))
            assert n == (1 if d == dest else 0), (over, d, n)
    # argv OFF but a controller built -> refused
    assert len(dvb.REGISTRY["map_hires_share_target"].check(good, _ns(map_hires_share_target=None))) == 1
    assert all(dvb.REGISTRY[d].check(bare, _ns(map_hires_share_target=None)) == [] for d in M.FLAGS)           # OFF + nothing built: clean


# =========================================================================== #
# H  the wiring, pinned in the trainer source (+ mutants)                      #
# =========================================================================== #
WIRING = [
    ("import", r"from tanitad\.train import map_dose_ctl as _mdc"),
    ("registry", r"_mdc\.register_dvb_levers\(_dvb\.register\)"),
    ("parser", r"_mdc\.add_arguments\(g8\)"),
    ("pin", r"_mdc\.pin_args\(args\)"),
    ("build", r"_dose = _mdc\.controller_for\(args\)"),
    ("attach", r"model\._map_dose_ctl = _dose"),
    ("groups", r"_dose_groups = _mdc\.groups_for\(model, _dose\)"),
    ("resume", r"_mdc\.resume_guard\(state, _dose\)"),
    ("post_dvb_apply", r"_dvb\.refuse_on_mismatch\(model, args, build_parser\(\), where=\"train\"\)\n    if _dose is not None:[^\n]*\n        model\._w_map_hires = float\(_dose\.w_eff\)"),
    ("hop1", r"_dose_replay = _mdc\.probe_before_forward\("),
    ("hop2", r"_dose_row = _mdc\.read_after_losses\(_dose, model, losses, step \+ 1, _dose_groups, TRAJ_WEIGHT, _dose_replay\)"),
    ("hop3", r"_mdc\.apply_weight\(_dose, model, _dose_row\)"),
    ("row", r"if _dose_row:[^\n]*\n\s+row\.update\(_dose_row\)"),
    ("ckpt_inline", r"_ema_ckpt_extra\(_ema, _b1\), \*\*_mdc\.ckpt_extra\(model\), \*\*_p10_ckpt_extra\(model\), \*\*_rng_ckpt_extra\(model\)\}, ck\)"),
    ("ckpt_payload", r"_ema_ckpt_extra\(ema, b1\), \*\*_mdc\.ckpt_extra\(model\), \*\*_p10_ckpt_extra\(model\), \*\*_rng_ckpt_extra\(model\)\}\n"),
]


def wiring_problems(src):
    return [name for name, pat in WIRING if not re.search(pat, src)]


def test_H_the_trainer_wires_every_hop():
    assert wiring_problems(TRAINER_SRC) == []
    i1, i2, i3 = (TRAINER_SRC.index(x) for x in ("_dose_replay = _mdc.probe_before_forward(", "_dose_row = _mdc.read_after_losses(", "_mdc.apply_weight(_dose, model, _dose_row)"))
    fwd, bwd = TRAINER_SRC.index("losses = compute_losses_v3(model, batch, device, mode=args.mode,\n                                   ablate_frames=args.ablate_frames,\n                                   log_metrics=_logged_after"), \
        TRAINER_SRC.index("losses[\"loss\"].backward()\n        if _cd_sub:")
    assert i1 < fwd < i2 < bwd < i3            # replay BEFORE the training forward; the reading BEFORE the backward; the weight applied AFTER it


@pytest.mark.parametrize("name,pat", WIRING)
def test_H_MUTANT_deleting_any_hop_from_a_copy_of_the_trainer_goes_RED(name, pat):
    mut = re.sub(pat, "# MUTANT: removed", TRAINER_SRC, count=1)
    assert mut != TRAINER_SRC and wiring_problems(mut) == [name]


def test_H_the_weight_reaches_the_loss_through_model_attr_w_map_hires():
    # hop 3 writes `model._w_map_hires`; both the plain and the banded map term multiply by `float(getattr(model, "_w_map_hires"...))` at call time
    assert TRAINER_SRC.count('w = float(getattr(model, "_w_map_hires", 0.0) or 0.0)') >= 2
    assert re.search(r"return w \* row\[\"loss\"\]", TRAINER_SRC)
    src = (ROOT / "tanitad" / "train" / "map_dose_ctl.py").read_text(encoding="utf-8")
    assert src.count("model._w_map_hires = float(ctl.w_eff)") == 1


def test_H_MUTANT_apply_weight_not_assigning_the_attribute_makes_the_loop_test_RED(monkeypatch):
    def no_apply(ctl, model, row):
        return None
    monkeypatch.setattr(M, "apply_weight", no_apply)
    ctl = M.MapDoseController(0.30, 2, "trunk", 1e-3, 1e3, 2.0)
    recs, _, _ = _run_loop(ctl, 3, w0=2.0)
    assert recs[1]["row"]["dc_w_eff"] != 2.0 and recs[2]["w_used"] == 2.0            # the controller solved a weight and the loss never saw it


# =========================================================================== #
# I  the launch gate                                                           #
# =========================================================================== #
K2DC = json.loads((ROOT / "ops" / "runs.d" / "r9e2_K2DC.argv.json").read_text(encoding="utf-8"))
K2M1N20 = json.loads((ROOT / "ops" / "runs.d" / "r9e2_K2M1N20.argv.json").read_text(encoding="utf-8"))
DC_TAIL = ["--map-hires-share-target", "0.10", "--map-hires-share-every", "250", "--map-hires-share-group", "trunk", "--map-hires-w-bounds", "1,25"]


def test_I_the_K2DC_argv_is_K2M1N20s_plus_the_four_flags_and_the_renames_only():
    assert K2DC["arm"] == "r9e2_K2DC" and K2M1N20["arm"] == "r9e2_K2M1N20"
    a, b = K2DC["argv"], K2M1N20["argv"]
    assert a[-8:] == DC_TAIL and a[:-8] != b                                          # only --out differs in the shared part:
    diff = [(x, y) for x, y in zip(a[:-8], b) if x != y]
    assert diff == [("/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC", "/home/nvidia/r8ns/r9e2/runs/r9e2-K2M1N20")]
    skip = ("arm", "argv", "what", "lever")      # MM 2026-11-13: K2DC is COMPOSED (r9_e2_make_argv.py), which adds the controller to `lever` and `what`, nothing else
    assert {k: v for k, v in K2DC.items() if k not in skip} == {k: v for k, v in K2M1N20.items() if k not in skip}
    assert {k: v for k, v in K2DC["lever"].items() if not k.startswith("dose_ctl")} == K2M1N20["lever"] and K2DC["lever"]["dose_ctl"] == DC_TAIL
    assert K2DC["what"].startswith(K2M1N20["what"].replace("K2M1N20", "K2DC", 1))
    assert LG.flag_values(a, "--w-map-hires") == ["25.0"] and LG.flag_values(a, "--grad-share-terms") == ["tac_v6,map_hires"]


def test_I_static_reasons_inert_without_the_flags_pass_on_K2DC_and_refuse_each_defect():
    assert LG.dose_controller_static_reasons(K2M1N20["argv"])[0] == []
    r, d = LG.dose_controller_static_reasons(K2DC["argv"])
    assert r == [] and d["bounds"] == [1.0, 25.0]
    # DELIBERATE REGRESSION: the target unset on an argv that declares the controller elsewhere
    no_t = LG.set_flag(K2DC["argv"], "--map-hires-share-target", None)
    r, _ = LG.dose_controller_static_reasons(no_t)
    assert len(r) == 1 and "declared but --map-hires-share-target is not passed" in r[0]
    for flag, val, needle in (("--map-hires-share-target", "1.5", "0 < target < 1"), ("--map-hires-w-bounds", "30,1", "LO,HI"),
                              ("--map-hires-w-bounds", "1,20", "outside --map-hires-w-bounds"), ("--grad-share-terms", "tac_v6", "does not separate")):
        r, _ = LG.dose_controller_static_reasons(LG.set_flag(K2DC["argv"], flag, [val]))
        assert len(r) == 1 and needle in r[0], (flag, r)


def test_I_the_static_gate_entry_carries_the_dose_check():
    prof = LG.PROFILES["refcv9-e2"]
    assert [x for x in LG.profile_argv_rules(prof, K2DC["argv"])[0] if "DOSE-CTL" in x] == []
    bad = LG.set_flag(K2DC["argv"], "--map-hires-share-target", None)                  # DELIBERATE REGRESSION: target unset, controller declared elsewhere
    ref = [x for x in LG.profile_argv_rules(prof, bad)[0] if "DOSE-CTL" in x]
    assert len(ref) == 1 and "never built" in ref[0]


def test_I_smoke_argv_makes_exactly_one_firing_inside_the_smoke():
    for steps, L in ((100, 50), (150, 50), (120, 50), (100, 25)):
        run, dep = LG.smoke_argv(K2DC["argv"], steps, L)
        every = int(LG.flag_values(run, "--map-hires-share-every")[0])
        assert every % L == 0 and every <= steps and 2 * every > steps, (steps, L, every)
        assert len(LG.dc_reading_rows(run, steps)) == 1 and "--map-hires-share-every" in dep
        assert LG.flag_values(run, "--map-hires-share-target") == ["0.10"]
    run, dep = LG.smoke_argv(K2M1N20["argv"], 100, 50)                          # OFF: nothing rewritten, nothing departed
    assert not LG.has_flag(run, "--map-hires-share-every") and "--map-hires-share-every" not in dep
    assert LG.dc_reading_rows(K2M1N20["argv"]) == [] and LG.dc_reading_rows(K2DC["argv"]) == list(range(250, 6001, 250))


def _good_row(step=100, w=12.0, pred=0.10, clamped=0.0):
    return {"step": step, "loss": 1.0, "dc_w_before": 25.0, "dc_w_eff": w, "dc_share_read": 0.104, "dc_share_pred": pred, "dc_clamped": clamped}


def test_I_G_LIVE_rows_pass_clamped_and_FAIL_without_a_dc_row_or_with_a_bad_one():
    A = K2DC["argv"]
    assert LG.dose_controller_live_reasons(K2M1N20["argv"], [{"step": 50, "loss": 1.0}])[0] == []        # inert without the flags
    r, d = LG.dose_controller_live_reasons(A, [{"step": 50, "loss": 1.0}, _good_row()])
    assert r == [] and d["n_firings"] == 1
    assert LG.dose_controller_live_reasons(A, [_good_row(pred=0.10 + 5e-7)])[0] == []                      # within 1e-6
    assert LG.dose_controller_live_reasons(A, [_good_row(w=25.0, pred=0.07, clamped=1.0)])[0] == []        # clamped: pred != target is allowed
    r, _ = LG.dose_controller_live_reasons(A, [{"step": 50, "loss": 1.0}, {"step": 100, "loss": 1.0}])       # controller ON, no dc_* row
    assert len(r) == 1 and "never fired inside G-LIVE" in r[0]
    for bad, needle in ((_good_row(pred=0.12), "!= target"), (_good_row(w=26.0), "inside [1, 25]"), (_good_row(w=0.5), "inside [1, 25]"),
                        (_good_row(w=float("nan")), "inside [1, 25]"), (_good_row(pred=float("nan"), clamped=1.0), "!= target")):
        r, _ = LG.dose_controller_live_reasons(A, [bad])
        assert len(r) == 1 and needle in r[0], (bad, r)
    r, _ = LG.dose_controller_live_reasons(LG.set_flag(A, "--map-hires-share-target", None), [_good_row()])
    assert len(r) == 1 and "declared but" in r[0]


def test_I_full_profile_live_reasons_carries_the_dose_check_and_the_share_alarm_is_unchanged():
    prof = LG.PROFILES["refcv9-full"]
    A = K2DC["argv"]
    rows = [{"step": 50, "loss": 1.0, "lr": 1e-4, "gs_trunk_proj_map_hires": 0.104}]
    reasons, det = LG.full_profile_live_reasons(dict(prof, lr_schedule_rule=None), A, A, rows, None)
    assert [x for x in reasons if x.startswith("SHARE-ALARM")] == [] and any(x.startswith("DOSE-CTL") and "never fired" in x for x in reasons)   # FIRING LINE DELETED
    reasons, det = LG.full_profile_live_reasons(dict(prof, lr_schedule_rule=None), A, A, rows + [_good_row()], None)
    assert reasons == [] and det["dose_controller"]["n_firings"] == 1
    low = [{"step": 50, "loss": 1.0, "lr": 1e-4, "gs_trunk_proj_map_hires": 0.04}, _good_row()]
    reasons, _ = LG.full_profile_live_reasons(dict(prof, lr_schedule_rule=None), A, A, low, None)
    assert len(reasons) == 1 and reasons[0].startswith("SHARE-ALARM")                    # K2's own floor, UNCHANGED, reads step 50 at w 25
