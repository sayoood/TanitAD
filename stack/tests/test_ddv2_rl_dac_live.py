"""The RL reward's DAC term is REACHABLE, and its dead default is pinned as dangerous.

⛔ WHY THIS FILE EXISTS. `H-DDV2RL-2` came out FAIL-HARM with a reward that had **no
road-boundary term at all**, and three independent layers had to hold for that to hide:

1. `score_candidates` takes `dac_cand` / `dac_human` as **optional keywords defaulting to
   ones**;
2. **no caller anywhere passed them** — `dac_from_drivable` had never had a production
   caller;
3. the trainer's telemetry key list was `("nc", "ep", "ttc", "comfort")` — **`"dac"` was
   omitted**, so 600 steps x 3 arms logged nothing about the term. MEASURED: no
   `cand_dac_mean` key exists in any banked L1 metrics file, while `cand_nc_mean`,
   `cand_ep_mean` and `cand_ttc_mean` are all present and varying.

⇒ A sub-score that is not logged cannot be seen to be constant. These tests pin the
contract at each layer, and `test_MUTATION_*` reproduces the shipped shape.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.rl import pdm_proxy as P                        # noqa: E402

CART_H, CART_W, CELL, Y_HALF = 120, 64, 0.5, 16.0


def _map(drivable_everywhere: bool = True):
    """(drivable_frac, seen) on the 120x64 @ 0.5 m rig grid."""
    frac = torch.ones((CART_H, CART_W)) if drivable_everywhere else torch.zeros((CART_H, CART_W))
    return frac, torch.ones((CART_H, CART_W), dtype=torch.bool)


def _straight(n_ticks: int, v: float = 10.0, y: float = 0.0):
    """One candidate driving straight ahead at ``v``, offset ``y`` metres to the LEFT."""
    t = torch.arange(n_ticks + 1, dtype=torch.float32) * P.PROXY.dt
    st = torch.zeros((1, n_ticks + 1, 4))
    st[0, :, 0] = v * t
    st[0, :, 1] = y
    st[0, :, 3] = v
    return st


# --------------------------------------------------------------------------- #
# layer 1: the function itself discriminates                                   #
# --------------------------------------------------------------------------- #

def test_dac_is_1_on_drivable_ground_and_0_off_it():
    st = _straight(P.PROXY.n_ticks)
    frac_ok, seen = _map(True)
    frac_no, _ = _map(False)
    assert float(P.dac_from_drivable(st, frac_ok, seen)[0]) == 1.0
    assert float(P.dac_from_drivable(st, frac_no, seen)[0]) == 0.0


def test_unseen_cells_carry_no_evidence_so_dac_is_an_UPPER_bound():
    """A cell that was never observed must not be scored as a violation."""
    st = _straight(P.PROXY.n_ticks)
    frac_no, seen = _map(False)
    assert float(P.dac_from_drivable(st, frac_no, torch.zeros_like(seen))[0]) == 1.0


# --------------------------------------------------------------------------- #
# layer 2: the DEFAULT is dangerous, and that is pinned                        #
# --------------------------------------------------------------------------- #

def _score(st, human, dac_kw):
    tracks = P.AgentTracks.from_frames([[] for _ in range(P.PROXY.n_ticks + 1 + 9)],
                                       torch.zeros((P.PROXY.n_ticks + 1 + 9, 4)))
    route = human[:, :2].clone()
    return P.score_candidates(st, human, tracks, route, **dac_kw)


def test_MUTATION_the_shipped_default_scores_a_FULLY_OFFROAD_plan_as_compliant():
    """⛔ The defect's exact shape. Called WITHOUT `dac_cand`, a plan driving entirely
    on non-drivable ground still reads `dac == 1` and a NON-ZERO `pdms`.

    If this ever fails, the default stopped being ones — which would be an
    improvement, but it means the surrounding comments and the H-DDV2RL-2 account are
    stale and must be rewritten rather than quietly passing.
    """
    st, human = _straight(P.PROXY.n_ticks), _straight(P.PROXY.n_ticks)[0]
    out = _score(st, human, {})                       # <-- the shipped call shape
    assert float(out["dac"][0]) == 1.0
    assert float(out["pdms"][0]) > 0.0


def test_passing_dac_makes_the_SAME_offroad_plan_score_zero():
    st, human = _straight(P.PROXY.n_ticks), _straight(P.PROXY.n_ticks)[0]
    frac_no, seen = _map(False)
    dac_c = P.dac_from_drivable(st, frac_no, seen)
    dac_h = P.dac_from_drivable(human[None], frac_no, seen)[0]
    out = _score(st, human, {"dac_cand": dac_c, "dac_human": dac_h})
    assert float(out["dac"][0]) == 0.0
    assert float(out["pdms"][0]) == 0.0, (
        "DAC is a MULTIPLIER on pdms; a zero must annihilate the score, not scale it")


def test_dac_also_gates_EP_not_only_the_final_product():
    """⚠️ `multi = nc * dac` feeds `raw = ego_progress(...) * multi`, so a dead DAC also
    removes drivability from the PROGRESS normalisation — an off-road candidate that
    does not collide would otherwise earn FULL progress credit."""
    st, human = _straight(P.PROXY.n_ticks), _straight(P.PROXY.n_ticks)[0]
    frac_no, seen = _map(False)
    live = _score(st, human, {"dac_cand": P.dac_from_drivable(st, frac_no, seen),
                              "dac_human": P.dac_from_drivable(human[None], frac_no, seen)[0]})
    dead = _score(st, human, {})
    assert float(live["ep"][0]) < float(dead["ep"][0]) or float(live["ep"][0]) == 0.0


# --------------------------------------------------------------------------- #
# layer 3: the telemetry must be able to SEE the term                          #
# --------------------------------------------------------------------------- #

def test_the_trainer_logs_cand_dac_mean():
    """⛔ The literal key list, asserted as a LITERAL. Until 2026-09-17 it read
    `("nc", "ep", "ttc", "comfort")` and no banked L1 metrics file carries a
    `cand_dac_mean` at all."""
    src = (ROOT / "scripts" / "ddv2_rl_refcv5.py").read_text(encoding="utf-8")
    assert '"nc", "dac", "ep", "ttc", "comfort"' in src, (
        "the telemetry key list must include 'dac': a sub-score that is not logged "
        "cannot be seen to be constant")


def test_score_candidates_returns_dac_so_the_key_list_is_satisfiable():
    """The contract behind the line above — tested functionally, not by grep."""
    st, human = _straight(P.PROXY.n_ticks), _straight(P.PROXY.n_ticks)[0]
    out = _score(st, human, {})
    for k in ("nc", "dac", "ep", "ttc", "comfort"):
        assert k in out, f"score_candidates must return {k!r} for the telemetry to log it"
        assert torch.is_tensor(out[k])


# --------------------------------------------------------------------------- #
# layer 4: the trainer plumbs it, and on the RAW frame                         #
# --------------------------------------------------------------------------- #

def test_the_map_is_keyed_on_the_RAW_frame_and_the_option_exists():
    src = (ROOT / "scripts" / "ddv2_rl_refcv5.py").read_text(encoding="utf-8")
    assert "self.map_for(cid, r0)" in src, (
        "the map must be keyed on r0 (the RAW frame), not on the stacked-row t0: "
        "t0 would label every window raw_off frames early, silently")
    assert "--map-gt-root" in src


def test_FUNCTIONAL_score_batch_actually_PASSES_dac_to_the_scorer():
    """⛔ THE ARM THAT CATCHES THE REAL DEFECT, and the first version of it did not.

    A source check for the strings `dac_cand` / `dac_human` **passes on the broken
    code**, because the names still appear in the `dk = {...}` construction even when
    `**dk` is never handed to `score_candidates`. MEASURED by mutation 2026-09-17:
    deleting `**dk` left all eight tests green. ⇒ **the call must be EXERCISED, not
    grepped.**

    The map here is non-drivable EVERYWHERE, so every candidate is off-road whatever
    the integrator produces — `dac` must read 0. It reads 1 the moment the plumbing
    stops passing it.
    """
    import ddv2_rl_refcv5 as R                                # noqa: E402

    n = P.PROXY.n_ticks
    human = _straight(n)[0]
    tracks = P.AgentTracks.from_frames([[] for _ in range(n + 1 + 9)],
                                       torch.zeros((n + 1 + 9, 4)))
    frac_no, seen = _map(False)
    item = {"human": human, "tracks": tracks, "route": human[:, :2].clone(),
            "map_drivable": frac_no, "map_seen": seen}

    class _Dec:
        anchor_control_units = "alat"
        anchor_alat_v_floor = 4.0
        anchor_kappa_cap = 0.12

    class _Ctx:
        # must reach the scorer's n_ticks (40) or `ego_states_from_controls` refuses
        horizons = (10, 20, 30, 40)
        dec = _Dec()

    u = torch.zeros((1, 1, len(_Ctx.horizons), 2))            # straight, no lateral accel
    v = torch.full((1,), 10.0)
    outs, _ = R.score_batch(_Ctx(), u, [item], v)
    assert float(outs[0]["dac"][0]) == 0.0, (
        "score_batch did not pass dac_cand: the scorer defaulted the road-boundary "
        "multiplier to ONES on a map that is non-drivable everywhere. That is exactly "
        "the H-DDV2RL-2 configuration.")

    # and the control: with NO map on the item AND no map root configured, the same
    # call must fall back to 1.0 -- so the test measures the plumbing and not something
    # that is always 0.
    item_nomap = dict(item, map_drivable=None, map_seen=None)
    outs2, _ = R.score_batch(_Ctx(), u, [item_nomap], v)
    assert float(outs2[0]["dac"][0]) == 1.0


def test_a_map_root_with_NO_MAPS_ON_THE_ITEMS_is_REFUSED():
    """⛔ The 'artifact supplied but read by nothing' refusal.

    MEASURED by mutation 2026-09-17: silently dropping the map in `fetch`
    (`"map_drivable": None`) left every unit test green, because the unit tests build
    their items by hand and never exercise `fetch`. The escape is closed here, at the
    one place both paths meet: if a map root is configured and NOT ONE item carries a
    map, the run refuses rather than quietly scoring with DAC \u2261 1 under a config that
    names a SAM3 corpus.
    """
    import ddv2_rl_refcv5 as R                                # noqa: E402

    n = P.PROXY.n_ticks
    human = _straight(n)[0]
    tracks = P.AgentTracks.from_frames([[] for _ in range(n + 1 + 9)],
                                       torch.zeros((n + 1 + 9, 4)))
    item = {"human": human, "tracks": tracks, "route": human[:, :2].clone(),
            "map_drivable": None, "map_seen": None}

    class _Dec:
        anchor_control_units = "alat"
        anchor_alat_v_floor = 4.0
        anchor_kappa_cap = 0.12

    class _CtxWithRoot:
        horizons = (10, 20, 30, 40)
        dec = _Dec()
        map_root = "D:/some/sam3/root"          # declared...

    u = torch.zeros((1, 1, 4, 2))
    v = torch.full((1,), 10.0)
    with pytest.raises(SystemExit, match="NOT ONE item in this batch carries a map"):
        R.score_batch(_CtxWithRoot(), u, [item], v)           # ...but never read
