"""⛔ SPEC_REFCV7 A2 / PI R1: the fed set speed is a HARD cap on the EMITTED plan, not only on the
decoder's internal argmax.

MEASURED 2026-09-28 (refcv7 step 1,500, NavSim bridge): the plans were bit-identical with the ceiling
filter ON and OFF on 204/204 warmup scenes, and 2 emitted plans exceeded the ceiling on CUDA. The
decoder masks only its LOCAL `rank` and returns `score` unmasked. `RefCV3Model`'s E9 goal selection
re-ranked the fan from that score with the reach mask only, so the ceiling never reached
`out["traj"]`. The existing `test_speed_ceiling_inference_only.py` pins the DECODER (`RefCModel`),
where the filter does bite, which is exactly why this gap was invisible.

Pinned here:
* `e9_rank` with LITERAL inputs: the ceiling moves the argmax; an unsatisfiable row keeps its
  reach-only ranking and is counted;
* the DECODER exports `ceil_keep` at eval when the filter bites, and never in training;
* the REAL `RefCV3Model` (smoke rig, CPU): a ceiling that excludes E9's natural pick moves the
  EMITTED `traj` / `sel_idx` onto a compliant candidate;
* RED ARM: the pre-fix E9 (reach mask only) makes that assertion fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tanitad.refs import refc                                  # noqa: E402
from tanitad.refs import refc_v3 as v3                         # noqa: E402
from tanitad.refs.refc_agents import AgentSeamConfig           # noqa: E402
from tanitad.refs.refcv6_selection import planned_max_speed    # noqa: E402

INF = float("-inf")


# ------------------------------------------------------------------ e9_rank, literal ----
def test_e9_rank_without_masks_is_the_score():
    s = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    rank, tele = v3.e9_rank(s, {})
    assert torch.equal(rank, s) and tele == {}


def test_e9_rank_the_ceiling_moves_the_argmax():
    s = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    ck = torch.tensor([[False, True, True, True]])
    rank, tele = v3.e9_rank(s, {"ceil_keep": ck})
    assert int(rank.argmax(dim=1)) == 1
    assert torch.equal(rank, torch.tensor([[INF, 2.0, 1.0, 0.0]]))
    assert float(tele["e9_ceil_dead_frac"]) == 0.0


def test_e9_rank_reach_and_ceiling_combine():
    s = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    rk = torch.tensor([[True, False, True, True]])
    ck = torch.tensor([[False, True, True, True]])
    rank, _ = v3.e9_rank(s, {"reach_keep": rk, "ceil_keep": ck})
    assert torch.equal(rank, torch.tensor([[INF, INF, 1.0, 0.0]]))
    assert int(rank.argmax(dim=1)) == 2


def test_e9_rank_an_unsatisfiable_row_keeps_reach_only_and_is_counted():
    s = torch.tensor([[3.0, 2.0, 1.0, 0.0],
                      [3.0, 2.0, 1.0, 0.0]])
    rk = torch.tensor([[True, True, False, False],
                       [True, True, True, True]])
    ck = torch.tensor([[False, False, True, True],       # row 0: no candidate under both
                       [False, False, True, True]])
    rank, tele = v3.e9_rank(s, {"reach_keep": rk, "ceil_keep": ck})
    assert torch.equal(rank[0], torch.tensor([3.0, 2.0, INF, INF]))    # reach-only kept
    assert torch.equal(rank[1], torch.tensor([INF, INF, 1.0, 0.0]))
    assert rank.argmax(dim=1).tolist() == [0, 2]
    assert float(tele["e9_ceil_dead_frac"]) == 0.5


# ------------------------------------------------------------------ the decoder export ----
B = 8


def _core():
    cfg = refc.refc_smoke_config()
    cfg.agents = AgentSeamConfig(enable=True)
    cfg.decoder.cross_agent = True
    cfg.speed_ceiling_filter = True
    torch.manual_seed(0)
    return refc.RefCModel(cfg), cfg


def _core_run(model, cfg, v_lim):
    g = torch.Generator().manual_seed(1)
    frames = torch.rand(B, cfg.window, 1, 64, 64, generator=g)
    v0 = torch.linspace(4.0, 18.0, B)
    torch.manual_seed(7)
    with torch.no_grad():
        return model(frames, v0=v0, steps=0, scene_hook=lambda *a, **k: {"v_limit_ms": v_lim})


def _biting_limit(model, out):
    dec = model.decoder
    x, idx = out["anchor_traj"], out["sel_idx"]
    vmax = planned_max_speed(x, horizons=dec.anchor_horizons, tick_s=dec.anchor_dt)
    ar = torch.arange(x.shape[0])
    win = vmax[ar, idx]
    slower = torch.where(vmax < win[:, None] - 1e-3, vmax, torch.full_like(vmax, -1.0))
    nxt = slower.max(dim=1).values
    return torch.where(nxt >= 0.0, nxt, torch.full_like(nxt, float("inf")))


def test_the_decoder_EXPORTS_ceil_keep_at_eval():
    model, cfg = _core()
    model.eval()
    ref = _core_run(model, cfg, torch.full((B,), float("inf")))
    got = _core_run(model, cfg, _biting_limit(model, ref))
    assert "ceil_keep" in got, "the decoder must export the mask its argmax used"
    ck = got["ceil_keep"]
    assert ck.dtype == torch.bool and tuple(ck.shape) == tuple(got["anchor_traj"].shape[:2])
    ar = torch.arange(B)
    assert bool(ck[ar, got["sel_idx"]].all()), "the decoder's own pick must be inside its mask"


def test_the_decoder_exports_NOTHING_in_training():
    model, cfg = _core()
    model.train()
    ref = _core_run(model, cfg, torch.full((B,), float("inf")))
    got = _core_run(model, cfg, _biting_limit(model, ref))
    assert "ceil_keep" not in got, "the filter is inference-only (PI 2026-09-26 A2)"


# ------------------------------------------------------------------ the REAL RefCV3 E9 path ----
def _v3():
    cfg = v3.refc_v3_smoke_config(hier=True)
    torch.manual_seed(0)
    return cfg, v3.RefCV3Model(cfg).eval()


def _frames(cfg, b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.core.encoder.image_hw()
    return torch.rand(b, cfg.core.window, cfg.core.encoder.in_channels, h, w, generator=g)


V0 = torch.tensor([3.0, 7.0, 5.0, 9.0])


def _run_with_ceiling(m, cfg, ceil_keep=None):
    """Forward through the REAL model; `ceil_keep` (if given) is injected into the core's output
    exactly where the decoder now exports it, so E9 sees it as it would at inference."""
    orig = m.core.forward

    def fwd(*a, **k):
        out = orig(*a, **k)
        if ceil_keep is not None:
            out["ceil_keep"] = ceil_keep
        return out
    m.core.forward = fwd
    try:
        torch.manual_seed(3)
        with torch.no_grad():
            return m(_frames(cfg), v0=V0, steps=2)
    finally:
        del m.core.forward


def _assert_E9_obeys(m, cfg):
    ref = _run_with_ceiling(m, cfg)
    assert "sel_idx_base" in ref, "E9 did not run on this rig -- the test proves nothing"
    nat = ref["sel_idx"]
    n = ref["anchor_traj"].shape[1]
    ck = torch.ones(nat.shape[0], n, dtype=torch.bool)
    ck[torch.arange(nat.shape[0]), nat] = False          # the ceiling excludes E9's natural pick
    if "reach_keep" in ref:
        ok = (ref["reach_keep"] & ck).any(dim=1)
        assert bool(ok.all()), "every row needs a reachable compliant candidate for this test"
    got = _run_with_ceiling(m, cfg, ck)
    ar = torch.arange(nat.shape[0])
    assert bool((got["sel_idx"] != nat).all()), \
        "the EMITTED pick must leave the candidate the ceiling excludes (SPEC_REFCV7 A2)"
    assert bool(ck[ar, got["sel_idx"]].all())
    assert torch.equal(got["traj"], got["anchor_traj"][ar, got["sel_idx"]])
    assert float(got["e9_ceil_dead_frac"]) == 0.0


def test_the_EMITTED_plan_obeys_the_ceiling_on_the_real_v3_model():
    cfg, m = _v3()
    _assert_E9_obeys(m, cfg)


def test_RED_ARM_the_pre_fix_E9_goes_RED(monkeypatch):
    """E9 ranking with the reach mask only (the launch code): the emitted pick ignores the
    ceiling and the obedience assertion must FAIL."""
    def old_rank(blended, out):
        rank = blended
        if "reach_keep" in out:
            rank = blended.masked_fill(~out["reach_keep"], float("-inf"))
        return rank, {"e9_ceil_dead_frac": torch.tensor(0.0)}
    monkeypatch.setattr(v3, "e9_rank", old_rank)
    cfg, m = _v3()
    with pytest.raises(AssertionError, match="must leave the candidate"):
        _assert_E9_obeys(m, cfg)
