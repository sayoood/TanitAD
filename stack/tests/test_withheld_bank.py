"""H-EGO-LIT-4 — the `--withheld-bank {fixed,pred,random,none}` switch.

Pre-registration: `Project Steering/GOALS_AND_CLAIMS.md` row `H-EGO-LIT-4`;
design: `TanitAD Research Lab/Architecture & Inference/Research/
2026-09-05-ego-input-literature/RESULT.md` §3.6; panel:
`.../2026-09-05-withheld-bank-panel/`.

⛔ THE POINT OF THIS FILE IS T1. The pre-registration says the `fixed` mode is
"bit-identical to the pre-flag trainer"; that is a claim about ARITHMETIC and it
is pinned here against the pre-flag expression written out by hand — not
against a re-run of the same code. Every other mode is then checked to move
ONLY the rows it is allowed to move (withheld ones), never a kept row and never
the eval regime.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.models.kinematic import rollout_unicycle          # noqa: E402
from tanitad.refs import refc                                   # noqa: E402
from tanitad.refs import refc_v3 as v3                          # noqa: E402

REF = 10.0
SPEED_MAX = 35.0


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _v4_v0cond_cfg(**kw):
    """The tiny CPU pair, as v4 (ego admitted, X15 on, NO echo-base — the live
    refcv4b levers) with a v0-CONDITIONED `alat` vocabulary."""
    cfg = v3.refc_v3_smoke_config(True)
    cfg.ego_state_inject = True
    cfg.echo_base = False
    cfg.core.ego_valid_channel = True
    cfg.core.anchors.v0_conditioned = True
    cfg.core.anchors.ref_speed_ms = REF
    cfg.core.anchors.control_units = "alat"
    for k, val in kw.items():
        setattr(cfg, k, val)
    return cfg


def _controls(n: int) -> torch.Tensor:
    """[N, 2] (a_lon, a_lat) with the {0, 0} control present (the trainer's
    own refusal otherwise)."""
    g = torch.Generator().manual_seed(7)
    c = torch.rand(n, 2, generator=g) * torch.tensor([6.0, 6.0]) - 3.0
    c[0] = 0.0
    return c


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = _v4_v0cond_cfg()
    m = v3.RefCV3Model(cfg)
    dec = m.core.decoder
    n = dec.anchors.shape[0]
    dec.load_anchors(dec.anchors.clone(), _controls(n))
    m.eval()
    return m, cfg


def _frames(core, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    h, w = core.encoder.image_hw()
    return torch.rand(n, core.window, core.encoder.in_channels, h, w,
                      generator=g)


def _ego(v, keep):
    """[B, 5] = (v0, a_long, yaw_rate, curvature, keep)."""
    v = torch.as_tensor(v, dtype=torch.float32)
    k = torch.as_tensor(keep, dtype=torch.float32)
    a = torch.full_like(v, 0.4)
    kap = torch.full_like(v, 0.02)
    return torch.stack([v, a, v * kap, kap, k], dim=1)


def _reference_roll(dec, v: torch.Tensor) -> torch.Tensor:
    """THE PRE-FLAG ARITHMETIC, written out by hand: roll `anchor_controls`
    from speed `v` [B] through `rollout_unicycle` exactly as `roll_bank` did
    before 2026-09-05 (float32, alat -> kappa per window, clamp, slot pick)."""
    n = dec.anchors.shape[0]
    b = v.shape[0]
    h = dec.anchor_roll_steps
    ctrl = dec.anchor_controls.to(torch.float32)
    vv = v.clamp_min(dec.anchor_alat_v_floor) ** 2
    kap = (ctrl[None, :, 1] / vv[:, None]).clamp(-dec.anchor_kappa_cap,
                                                 dec.anchor_kappa_cap)
    ctrl = torch.stack([ctrl[None, :, 0].expand(b, n), kap], dim=-1)
    ctrl = ctrl[:, :, None, :].expand(b, n, h, 2).reshape(-1, h, 2)
    s0 = torch.zeros(b * n, 4, dtype=torch.float32)
    s0[:, 3] = v[:, None].expand(b, n).reshape(-1)
    path = rollout_unicycle(s0, ctrl, dt=dec.anchor_dt)[..., :2]
    return path[:, dec.anchor_slots].reshape(b, n, dec.n_steps, 2)


V = torch.tensor([3.0, 14.0, 0.0, 27.5])
KEEP = torch.tensor([True, False, False, True])


# --------------------------------------------------------------------------- #
# T1 — `fixed` IS the pre-flag arithmetic, bit for bit                        #
# --------------------------------------------------------------------------- #
def test_T1_fixed_is_bit_identical_to_the_pre_flag_roll():
    m, _ = _model()
    dec = m.core.decoder
    assert dec.anchor_withheld_bank == "fixed"            # the default
    got = dec.roll_bank(V, KEEP, 4, torch.float32)
    v_pre = torch.where(KEEP, V, torch.full_like(V, REF))  # the OLD expression
    want = _reference_roll(dec, v_pre)
    assert torch.equal(got, want)
    # the new argument is INERT under `fixed`
    got2 = dec.roll_bank(V, KEEP, 4, torch.float32,
                         withheld_speed=torch.tensor([9.0, 9.0, 9.0, 9.0]))
    assert torch.equal(got2, want)
    # and the v_ms=None (eval / no-speed) path is the reference roll
    assert torch.equal(dec.roll_bank(None, None, 2, torch.float32),
                       _reference_roll(dec, torch.full((2,), REF)))


def test_T1b_kept_rows_never_move_under_fixed_pred_random():
    m, _ = _model()
    dec = m.core.decoder
    base = dec.roll_bank(V, KEEP, 4, torch.float32)
    dec.anchor_random_speed_pool = torch.tensor([1.0, 8.0, 22.0])
    for mode in ("fixed", "pred", "random"):
        dec.anchor_withheld_bank = mode
        got = dec.roll_bank(V, KEEP, 4, torch.float32,
                            withheld_speed=torch.tensor([5.0, 6.0, 7.0, 8.0]))
        assert torch.equal(got[KEEP], base[KEEP]), mode


# --------------------------------------------------------------------------- #
# T2 — `pred`: withheld rows at the DETACHED, CLAMPED predicted speed         #
# --------------------------------------------------------------------------- #
def test_T2_pred_rolls_withheld_rows_at_the_clamped_prediction():
    m, _ = _model()
    dec = m.core.decoder
    dec.anchor_withheld_bank = "pred"
    ws = torch.tensor([99.0, 50.0, -3.0, 99.0])           # rows 1, 2 withheld
    got = dec.roll_bank(V, KEEP, 4, torch.float32, withheld_speed=ws)
    v_eff = torch.where(KEEP, V, ws.clamp(0.0, SPEED_MAX))
    assert torch.equal(got, _reference_roll(dec, v_eff))
    assert float(v_eff[1]) == SPEED_MAX and float(v_eff[2]) == 0.0
    # without a prediction (warm-up / eval) it is the fixed roll
    fixed = _reference_roll(dec, torch.where(KEEP, V, torch.full_like(V, REF)))
    assert torch.equal(dec.roll_bank(V, KEEP, 4, torch.float32), fixed)


def test_T2b_pred_is_detached_from_the_speed_source():
    m, _ = _model()
    dec = m.core.decoder
    dec.anchor_withheld_bank = "pred"
    ws = torch.tensor([5.0, 6.0, 7.0, 8.0], requires_grad=True)
    got = dec.roll_bank(V, KEEP, 4, torch.float32, withheld_speed=ws)
    assert not got.requires_grad
    assert got.grad_fn is None


# --------------------------------------------------------------------------- #
# T3 — `random`: a draw from the POOL, independent of the row                  #
# --------------------------------------------------------------------------- #
def test_T3_random_draws_from_the_pool_and_refuses_without_one():
    m, _ = _model()
    dec = m.core.decoder
    dec.anchor_withheld_bank = "random"
    with pytest.raises(ValueError, match="random_speed_pool"):
        dec.roll_bank(V, KEEP, 4, torch.float32)
    pool = torch.tensor([1.5, 8.0, 22.0])
    dec.anchor_random_speed_pool = pool
    cands = {float(p): _reference_roll(dec, torch.full((1,), float(p)))[0]
             for p in pool}
    torch.manual_seed(3)
    seen = set()
    for _ in range(12):
        got = dec.roll_bank(V, KEEP, 4, torch.float32)
        for r in torch.nonzero(~KEEP).reshape(-1).tolist():
            hits = [p for p, c in cands.items() if torch.equal(got[r], c)]
            assert len(hits) == 1, "withheld row not rolled at a pool speed"
            seen.add(hits[0])
    assert len(seen) >= 2, "the draw never varied — not a marginal"


# --------------------------------------------------------------------------- #
# T4 — `none`: the speed-blind vocabulary, every row, kept or not             #
# --------------------------------------------------------------------------- #
def test_T4_none_ignores_speed_on_every_row():
    m, _ = _model()
    dec = m.core.decoder
    dec.anchor_withheld_bank = "none"
    ref = _reference_roll(dec, torch.full((4,), REF))
    assert torch.equal(dec.roll_bank(V, KEEP, 4, torch.float32), ref)
    assert torch.equal(dec.roll_bank(V, None, 4, torch.float32), ref)
    assert torch.equal(dec.roll_bank(None, None, 4, torch.float32), ref)


def test_T4b_unknown_mode_is_refused():
    m, _ = _model()
    dec = m.core.decoder
    dec.anchor_withheld_bank = "bucketed"
    with pytest.raises(ValueError, match="anchor_withheld_bank"):
        dec.roll_bank(V, KEEP, 4, torch.float32)


# --------------------------------------------------------------------------- #
# T5 — end to end: the hook's 2 s speed reaches the bank, detached; the       #
#      override (the eval-time SHUFFLE control) wins over the hook             #
# --------------------------------------------------------------------------- #
def test_T5_end_to_end_pred_bank_is_the_hooks_2s_speed():
    m, cfg = _model()
    dec = m.core.decoder
    fr = _frames(cfg.core, 2)
    v0 = torch.tensor([12.0, 12.0])
    ego = _ego([12.0, 12.0], [1.0, 0.0])                   # row 1 withheld
    with torch.no_grad():
        out_fixed = m(fr, v0=v0, ego_state=ego, steps=0)
        dec.anchor_withheld_bank = "pred"
        out_pred = m(fr, v0=v0, ego_state=ego, steps=0)
    assert "bank_speed_pred" in out_pred and out_pred["bank_speed_pred"].shape == (2,)
    assert "ego_keep" in out_pred
    assert torch.equal(out_pred["ego_keep"], torch.tensor([1.0, 0.0]))
    slot = cfg.goal_tau_steps.index(20)
    assert torch.equal(out_pred["bank_speed_pred"],
                       out_pred["g_tac"][:, slot, 3].detach())
    # kept row: bank unchanged between modes; withheld row: rolled at the
    # clamped prediction, and it DIFFERS from the fixed roll
    assert torch.equal(out_pred["anchor_bank"][0], out_fixed["anchor_bank"][0])
    ws = out_pred["bank_speed_pred"][1].clamp(0.0, SPEED_MAX)
    want = _reference_roll(dec, torch.stack([v0[0], ws]))
    assert torch.equal(out_pred["anchor_bank"].float(), want)
    assert torch.equal(out_fixed["anchor_bank"][1].float(),
                       _reference_roll(dec, torch.tensor([12.0, REF]))[1])
    # the override (shuffle control) wins over the hook
    with torch.no_grad():
        out_ovr = m(fr, v0=v0, ego_state=ego, steps=0,
                    withheld_speed=torch.tensor([3.0, 3.0]))
    assert torch.equal(out_ovr["anchor_bank"].float(),
                       _reference_roll(dec, torch.tensor([12.0, 3.0])))


def test_T5b_eval_regime_is_byte_identical_across_fixed_pred_random():
    """At eval nothing is withheld (keep = 1 everywhere), so the three
    training-time modes must emit the SAME bank and the SAME plan."""
    m, cfg = _model()
    dec = m.core.decoder
    dec.anchor_random_speed_pool = torch.tensor([1.0, 30.0])
    fr = _frames(cfg.core, 2)
    v0 = torch.tensor([6.0, 21.0])
    ego = _ego([6.0, 21.0], [1.0, 1.0])
    outs = {}
    with torch.no_grad():
        for mode in ("fixed", "pred", "random"):
            dec.anchor_withheld_bank = mode
            outs[mode] = m(fr, v0=v0, ego_state=ego, steps=0)
    for mode in ("pred", "random"):
        assert torch.equal(outs[mode]["anchor_bank"], outs["fixed"]["anchor_bank"])
        assert torch.equal(outs[mode]["traj"], outs["fixed"]["traj"])


def test_T5c_state_dict_is_unchanged_by_the_flag():
    """Plain attributes, never buffers: a checkpoint written before the flag
    loads strictly after it, and vice versa."""
    m, _ = _model()
    keys = set(m.state_dict())
    assert not [k for k in keys if "withheld" in k or "random_speed" in k]
    m.core.decoder.anchor_withheld_bank = "pred"
    assert set(m.state_dict()) == keys


# --------------------------------------------------------------------------- #
# T6 — the trainer STAMPS the policy (a run must say which arm it was)        #
# --------------------------------------------------------------------------- #
def test_T6_trainer_flag_and_seam_stamp():
    import refc_v3_train as tr
    ap = argparse.ArgumentParser()
    # the real parser is built inside main(); exercise the choices contract
    assert refc.WITHHELD_BANK_MODES == ("fixed", "pred", "random", "none")
    args = argparse.Namespace(goal_str=False, graft_lan=False,
                              withheld_bank="pred", withheld_bank_warmup=300,
                              withheld_speed_max=35.0)
    cfg = _v4_v0cond_cfg()
    st = tr._seam_stamp(cfg, args)
    assert st["withheld_bank"] == "pred"
    assert st["withheld_bank_warmup"] == 300
    assert st["withheld_speed_max_ms"] == 35.0
    # an args namespace WITHOUT the flag reads as the shipped default
    st0 = tr._seam_stamp(cfg, argparse.Namespace(goal_str=False,
                                                 graft_lan=False))
    assert st0["withheld_bank"] == "fixed" and st0["withheld_bank_warmup"] == 0
    del ap


def test_T6b_apply_refuses_pred_on_a_fixed_vocabulary_and_builds_the_pool():
    import refc_v3_train as tr
    m, _ = _model()
    dec = m.core.decoder

    class _Ep:
        def __init__(self, v):
            self.poses = torch.stack([torch.zeros(len(v)), torch.zeros(len(v)),
                                      torch.zeros(len(v)),
                                      torch.as_tensor(v, dtype=torch.float32)],
                                     dim=1)

    eps = [_Ep([0.0, 5.0, 10.0]), _Ep([20.0, 25.0])]
    args = argparse.Namespace(withheld_bank="random", withheld_bank_warmup=0,
                              withheld_speed_max=35.0, seed=0)
    st = tr._apply_withheld_bank(m, args, eps, "cpu")
    assert st["random_pool"]["n"] == 5 and st["random_pool"]["mean_ms"] == 12.0
    assert dec.anchor_withheld_bank == "random"
    assert dec.anchor_random_speed_pool.numel() == 5
    # warm-up > 0 starts on the FIXED roll (the loop flips it)
    args.withheld_bank, args.withheld_bank_warmup = "pred", 100
    st = tr._apply_withheld_bank(m, args, eps, "cpu")
    assert dec.anchor_withheld_bank == "fixed" and st["warmup_steps"] == 100
    # a fixed-path vocabulary has nothing to roll: refused
    dec.anchor_v0_cond = False
    with pytest.raises(SystemExit, match="v0-CONDITIONED"):
        tr._apply_withheld_bank(m, args, eps, "cpu")
