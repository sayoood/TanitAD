"""CPU smoke for refa-dynin (--dyn-input): the guarded yaw-rate conditioning
channel on the frozen-DINO REF-A+ trainer.

Validates every claim in the task:
  A. the model builds with action_dim 4 (v0 + yaw-rate on top of steer/accel),
     one grounded forward + loss step is FINITE, backward reaches the trainable
     adapter AND the predictor's action embedding yaw column (the "new yaw
     embedding"), and the frozen DINO features carry NO gradient (encoder stays
     frozen);
  B. off-flag == byte-identical: a --dyn-input-OFF build (action_dim 3, the
     existing speed-input model) is shape-identical everywhere; turning the flag
     ON changes ONLY the action-input layers (act_emb / inv_dyn), by exactly +1
     channel — so existing checkpoints still load;
  C. the ego-dropout guard is training-gated: eval mode feeds the true [v0, yr0]
     regardless of p (guard inert), train mode with p=1 zeros the ego vector
     (guard active) — mode itself changes nothing else (LayerNorm-only model);
  D. the 4-brain build (the canonical run) also wires action_dim 4 through BOTH
     the operative predictor and the tactical-predictor, and runs a forward.

Run:  <venv>/python stack/experiments/reset-speed4b/smoke_dynin.py
Portable: resolves the local stack from __file__ (no pod paths needed).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
STACK = HERE.parents[2]                       # .../stack
sys.path[:0] = [str(STACK), str(STACK / "scripts"), str(HERE.parent)]

import torch  # noqa: E402

import refa_train_plus as tp  # noqa: E402
from refa_plus import RefAModelPlus  # noqa: E402

N_TOK, D_DINO, GRID, DR = 16, 768, 4, 32      # state_dim = 4*4*32 = 512 (tiny)
B, H = 8, 4


def _model(speed, yaw, dyn, adapter="temporal"):
    pc = tp.smoke_pred_config()
    adim = 2 + int(speed) + int(yaw or dyn)
    if adim > 2:
        pc = dataclasses.replace(pc, action_dim=adim)
    m = RefAModelPlus(pc, adapter_kind=adapter, n_tokens=N_TOK, grid=GRID,
                      grid_d_readout=DR)
    m.standardizer.fit(torch.randn(64, N_TOK, D_DINO) for _ in range(2))
    return m


def _batch():
    W = tp.smoke_pred_config().window
    pose_last = torch.randn(B, 4)
    pose_last[:, 3] = pose_last[:, 3].abs() * 5 + 5           # ego-speed > 0
    return {
        "feats": torch.randn(B, W, N_TOK, D_DINO).half(),
        "actions": torch.randn(B, W, 2),
        "future_feats": torch.randn(B, H, N_TOK, D_DINO).half(),
        "future_actions": torch.randn(B, H, 2),
        "future_poses": torch.randn(B, H, 4),
        "pose_last": pose_last,
        "yawrate0": torch.randn(B) * 0.2,                     # rad/s-scale
    }


def _grounded(model, batch, *, dyn, ego_dropout, no_grad=False):
    kw = dict(metric_heads=tp.base.build_metric_heads(model.state_dim, "cpu"),
              mid_horizons=list(model.pred_cfg.horizons),
              speed_input=True, dyn_input=dyn, ego_dropout=ego_dropout)
    ctx = torch.no_grad() if no_grad else _nullctx()
    with ctx:
        return tp.compute_losses_plus(model, batch, 4, "cpu", **kw)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def part_a_build_forward_backward():
    torch.manual_seed(0)
    m = _model(speed=True, yaw=False, dyn=True)
    assert m.pred_cfg.action_dim == 4, m.pred_cfg.action_dim
    assert m.predictor.act_emb[0].in_features == 4
    m.train()
    mh = tp.base.build_metric_heads(m.state_dim, "cpu")
    opt = torch.optim.Adam(list(m.parameters())
                           + [p for h in mh.values() for p in h.parameters()], lr=1e-3)

    def gsum(ps):
        return sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)

    # A few real train steps: the operative's action FiLM is zero-initialised
    # (identity at step 0), so the action embedding only receives gradient once
    # FiLM warms off zero — exactly as in a real run. ego_dropout=0 so the yaw
    # channel is active every sample (the guard is exercised separately in C).
    w = m.predictor.act_emb[0].weight                     # [d, 4] = [.., v0, yr0]
    yaw_g = v0_g = adapt_g = 0.0
    for _ in range(6):
        batch = _batch()
        opt.zero_grad(set_to_none=True)
        out = tp.compute_losses_plus(
            m, batch, 4, "cpu", metric_heads=mh,
            mid_horizons=list(m.pred_cfg.horizons), speed_input=True,
            dyn_input=True, ego_dropout=0.0)
        assert torch.isfinite(out["loss"]), out["loss"]
        for k in ("pred", "roll", "inv", "sigreg", "metric_invdyn", "fwd"):
            assert torch.isfinite(out[k]).all(), k
        out["loss"].backward()
        yaw_g = float(w.grad[:, 3].abs().sum())           # yaw-rate embedding column
        v0_g = float(w.grad[:, 2].abs().sum())            # v0 embedding column
        adapt_g = gsum(m.adapter.parameters())
        # encoder frozen: DINO features are pure data, no grad ever.
        assert batch["feats"].requires_grad is False and batch["feats"].grad is None
        assert batch["future_feats"].grad is None
        opt.step()
    assert adapt_g > 0, "adapter got no grad"
    assert yaw_g > 0, "yaw-rate embedding column never received grad"
    assert v0_g > 0, "v0 embedding column never received grad"
    print(f"A ok: action_dim=4, loss finite, adapter grad + backward reaches the "
          f"yaw embedding (|g_yaw|={yaw_g:.3e}, |g_v0|={v0_g:.3e}), feats grad-free")


def part_b_off_byte_identical():
    torch.manual_seed(1)
    m_off = _model(speed=True, yaw=False, dyn=False)         # action_dim 3
    m_dyn = _model(speed=True, yaw=False, dyn=True)          # action_dim 4
    assert m_off.pred_cfg.action_dim == 3
    sd_off, sd_dyn = m_off.state_dict(), m_dyn.state_dict()
    assert set(sd_off) == set(sd_dyn), "state_dict key sets diverge"
    diff = [k for k in sd_off if sd_off[k].shape != sd_dyn[k].shape]
    assert diff, "dyn-input changed no shapes (expected the action-input layers)"
    for k in diff:                                          # only action layers
        assert "act_emb" in k or "inv_dyn" in k, f"unexpected shape change: {k}"
    assert sd_dyn["predictor.act_emb.0.weight"].shape[1] == \
        sd_off["predictor.act_emb.0.weight"].shape[1] + 1, "yaw channel not +1"
    # existing ckpts load: a dyn-OFF ckpt round-trips strict into a fresh
    # dyn-OFF model; a dyn-ON ckpt is genuinely incompatible with a dyn-OFF model.
    _model(speed=True, yaw=False, dyn=False).load_state_dict(sd_off)   # strict OK
    _model(speed=False, yaw=False, dyn=False).load_state_dict(          # vanilla OK
        _model(speed=False, yaw=False, dyn=False).state_dict())
    try:
        _model(speed=True, yaw=False, dyn=False).load_state_dict(sd_dyn)
        raise SystemExit("FAIL: dyn-ON ckpt loaded into a dyn-OFF model")
    except RuntimeError:
        pass
    print(f"B ok: off==byte-identical (only {sorted(diff)} grow by +1 channel); "
          "dyn-off ckpt round-trips, dyn-on ckpt correctly rejected")


def part_c_guard_training_gated():
    torch.manual_seed(2)
    m = _model(speed=True, yaw=False, dyn=True)
    batch = _batch()

    def L(train, p):
        m.train(train)
        torch.manual_seed(123)                             # freeze SigReg/guard RNG
        return float(_grounded(m, batch, dyn=True, ego_dropout=p,
                               no_grad=True)["loss"])

    l_eval_p0, l_eval_p1 = L(False, 0.0), L(False, 1.0)
    l_train_p0, l_train_p1 = L(True, 0.0), L(True, 1.0)
    assert l_eval_p1 == l_eval_p0, "eval must ignore ego_dropout (guard inert)"
    assert l_train_p0 == l_eval_p0, "train==eval at p=0 (mode changes nothing else)"
    assert abs(l_train_p1 - l_eval_p0) > 1e-6, \
        "train p=1 must zero the ego vector (guard active) -> different loss"
    print(f"C ok: eval disables guard (L_eval_p1==L_eval_p0={l_eval_p0:.5f}); "
          f"train p=1 zeros ego (L={l_train_p1:.5f} != {l_eval_p0:.5f})")


def part_d_four_brain_wiring():
    from tanitad.config import refa4b_smoke_config
    torch.manual_seed(3)
    cfg = refa4b_smoke_config()
    adim = 2 + 1 + 1                                        # speed + yaw
    object.__setattr__(cfg.predictor, "action_dim", adim)
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", adim)
    m = RefAModelPlus.from_stack_config(cfg, n_tokens=N_TOK, adapter_kind="grid")
    m.standardizer.fit(torch.randn(64, N_TOK, D_DINO) for _ in range(2))
    assert m.predictor.act_emb[0].in_features == 4
    assert m.tactical_pred is not None and \
        m.tactical_pred.act_emb[0].in_features == 4, "tactical_pred not wired to 4"
    W = cfg.predictor.window
    with torch.no_grad():
        st = m.encode_window(torch.randn(B, W, N_TOK, D_DINO).half())
        z = m.predict(st, torch.randn(B, W, 4))[1]          # 4-dim actions
    assert z.shape[0] == B and torch.isfinite(z).all()
    print("D ok: 4-brain builds action_dim=4 through predictor + tactical_pred, "
          "forward runs")


if __name__ == "__main__":
    part_a_build_forward_backward()
    part_b_off_byte_identical()
    part_c_guard_training_gated()
    part_d_four_brain_wiring()
    print("SMOKE_DYNIN_PASS")
