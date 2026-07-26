"""PC1 wiring tests — the v2.1 ADAPTIVE-ARC route labels in the flagship path.

The v2.1 label FUNCTIONS are pinned by ``tests/test_refb_labels_v21.py``; this
file pins their INTEGRATION: the ``labels_v21`` gate on
``FailLoudWindowDataset`` / ``FlagshipWindowDataset``, the
``cfg.v21_route_labels`` flag, and the ``--labels-v21`` /
``--v2-route-from-vision`` CLI threading.

WHY THIS EXISTS (HPP-0 §1, corrected 2026-07-26). The HPP-0 audit's PC1 item #1
reads *"trainer flag `--labels-v2` (already exists)"* and the 4-Brain brief
carried that forward as *"`--labels-v2` => `route_target_v21`, coverage 27 % ->
80.4 %"*. **Both are wrong about the wiring**, and this file makes the
correction executable:

* ``--labels-v2`` selects the **v2** labeler (``route_from_future``), which
  keeps v1's fixed 15 s/25 s lookahead. On 17 100 real PhysicalAI trainer
  windows its coverage is **0.2307 vs v1's 0.2456** — marginally *worse*, not
  3x better (``verify_pc1_labels.py``).
* The route TARGET is a deterministic function of the fed route COMMAND on
  every CE-eligible window under **v1, v2 AND v2.1 alike** (echo 1.0000), for a
  structural reason no labeler swap can fix: the command is minted from the
  same ``route_from_future*`` call as the target, and ``_ROUTE_TO_NAV`` is a
  bijection. :func:`test_the_echo_survives_every_labeler_swap` pins that, so the
  next reader cannot re-acquire the belief that a label change alone fixes PC1.

What v2.1 DOES buy — pinned below — is coverage and honesty: ~3x the judgeable
windows, and ``ROUTE_UNKNOWN`` instead of a silent ``straight`` on the ones it
cannot judge. Both matter because LEVER A (``v2_route_from_vision``), the only
non-circular route gradient, trains **only on the valid mask**.

CPU-only, synthetic contract episodes (shares the roster with
``test_labels_v2_wiring``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R  # noqa: E402
import train_flagship4b as T4  # noqa: E402
from refb_train import FailLoudWindowDataset  # noqa: E402
from train_flagship4b import FlagshipWindowDataset, _wrap  # noqa: E402

from tanitad.config import flagship4b_smoke_config  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights, build_grounding,  # noqa: E402
                                           flagship_loss, horizon_plan)

from test_labels_v2_wiring import (FAST, _ep_windows, _eps_list,  # noqa: E402
                                   _sample)


def _ds(eps, cfg, plan, labels_v2=False, labels_v21=False):
    return FlagshipWindowDataset(
        eps, window=cfg.predictor.window, max_horizon=plan.max_horizon,
        maneuver_h=plan.maneuver_h, channels=cfg.encoder.in_channels,
        labels_v2=labels_v2, labels_v21=labels_v21)


def _spread(ds, per_ep=20):
    """Evenly-spaced window indices across the WHOLE of each episode.

    ``test_labels_v2_wiring._sample`` takes the FIRST k windows per episode,
    which is exactly the region where v1's 15 s-lookahead guard still passes —
    sampling that way hides the coverage collapse this file measures. The
    coverage question is a statement about a whole clip, so sample like one."""
    out = []
    for e in sorted({e for e, _t in ds.index}):
        ii = [i for i, (a, _t) in enumerate(ds.index) if a == e]
        out += ii[::max(1, len(ii) // per_ep)]
    return out


# --------------------------------------------------------------------------- #
# (a) OFF is the default and is byte-identical to the pre-PC1 behaviour         #
# --------------------------------------------------------------------------- #
def test_default_off_and_byte_identical():
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    assert cfg.v21_route_labels is False, "v2.1 route labels must default OFF"

    bare = FlagshipWindowDataset(                # no labels_v21 kwarg at all
        eps, window=cfg.predictor.window, max_horizon=plan.max_horizon,
        maneuver_h=plan.maneuver_h, channels=cfg.encoder.in_channels)
    assert bare.labels_v21 is False
    explicit = _ds(eps, cfg, plan, labels_v21=False)
    for i in _sample(bare, per_ep=6):
        a, b = bare[i], explicit[i]
        assert set(a) == set(b)
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k]), (i, k)

    # ... and the shared REF-B dataset keeps its own default OFF too.
    shared = FailLoudWindowDataset(eps, window=cfg.predictor.window,
                                   max_horizon=plan.max_horizon,
                                   channels=cfg.encoder.in_channels)
    assert shared.labels_v2 is False and shared.labels_v21 is False


# --------------------------------------------------------------------------- #
# (b) ON drives the v2.1 derivation, and buys COVERAGE                          #
# --------------------------------------------------------------------------- #
def test_on_matches_a_direct_v21_recompute():
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    ds = _ds(eps, cfg, plan, labels_v2=True, labels_v21=True)
    w = cfg.predictor.window
    for i in _sample(ds, per_ep=6):
        e_i, t = ds.index[i]
        poses = eps[e_i].poses
        t_last = t + w - 1
        cmd, valid = R.nav_command_v21(poses, t_last)
        tgt, tvalid = R.route_target_v21(poses, t_last)
        item = ds[i]
        assert int(item["nav_cmd"]) == int(cmd)
        assert bool(item["nav_valid"]) == bool(valid) == bool(tvalid)
        assert int(item["route_target"]) == int(tgt)


def test_v21_raises_coverage_over_v1_and_v2():
    """The whole point of the flag. v1/v2 gate on 15 s of REMAINING CLIP; v2.1
    gates on ARC TRAVELLED, so short/late windows become judgeable."""
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    idx = _spread(_ds(eps, cfg, plan))

    def cov(**kw):
        ds = _ds(eps, cfg, plan, **kw)
        return sum(bool(ds[i]["nav_valid"]) for i in idx) / len(idx)

    c1, c2 = cov(), cov(labels_v2=True)
    c21 = cov(labels_v2=True, labels_v21=True)
    assert c2 <= c1, ("v2 does not raise coverage over v1 — the correction "
                      "this file exists for", c1, c2)
    assert c21 > c1 and c21 > c2, (c1, c2, c21)
    # not a knife-edge: on real PhysicalAI windows the gap is 0.2456 -> 0.7546.
    assert c21 - max(c1, c2) > 0.2, (c1, c2, c21)


def test_unjudgeable_is_route_unknown_never_a_silent_straight():
    """v1/v2 emit ROUTE_STRAIGHT for "I cannot judge this" — the same class as
    "the road goes straight". v2.1 emits the out-of-range sentinel instead, so
    a consumer that drops the mask crashes rather than learning a wrong prior."""
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    ds21 = _ds(eps, cfg, plan, labels_v2=True, labels_v21=True)
    ds1 = _ds(eps, cfg, plan)

    seen_unknown = 0
    for i in _spread(ds21):
        it = ds21[i]
        if not bool(it["nav_valid"]):
            assert int(it["route_target"]) == R.ROUTE_UNKNOWN
            seen_unknown += 1
        else:
            assert int(it["route_target"]) < 3
    assert seen_unknown > 0, "fixture produced no unjudgeable window"
    # the v1 path does the opposite on its own invalid windows: "I cannot
    # judge" and "the road goes straight" are the SAME emitted class.
    v1_silent_straight = sum(
        int(ds1[i]["route_target"]) == R.ROUTE_STRAIGHT
        and not bool(ds1[i]["nav_valid"]) for i in _spread(ds1))
    assert v1_silent_straight > 0
    assert not any(int(ds1[i]["route_target"]) == R.ROUTE_UNKNOWN
                   for i in _spread(ds1))


def test_v21_stops_feeding_follow_through_a_turn():
    """HPP-0 §1.2, pinned: v1 hands the strategic level ``follow`` on windows
    where the vehicle is about to turn, because its `valid` guard fails on
    lookahead while the INPUT command is fed regardless."""
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    ds1 = _ds(eps, cfg, plan)
    ds21 = _ds(eps, cfg, plan, labels_v2=True, labels_v21=True)
    idx = _spread(ds21, per_ep=30)
    # the reference set: every window v2.1 is willing to call a route TURN.
    turning = [i for i in idx
               if bool(ds21[i]["nav_valid"])
               and int(ds21[i]["route_target"]) != R.ROUTE_STRAIGHT]
    assert turning, "fixture produced no judgeable turn"
    fed_follow_v1 = sum(int(ds1[i]["nav_cmd"]) == R.NAV_FOLLOW for i in turning)
    fed_follow_v21 = sum(int(ds21[i]["nav_cmd"]) == R.NAV_FOLLOW for i in turning)
    assert fed_follow_v1 > 0, "fixture does not exercise the v1 defect"
    assert fed_follow_v1 > fed_follow_v21
    assert fed_follow_v21 == 0        # v2.1 never says follow through its own turn


# --------------------------------------------------------------------------- #
# (c) THE CORRECTION — no labeler swap breaks the echo                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [dict(), dict(labels_v2=True),
                                dict(labels_v2=True, labels_v21=True)])
def test_the_echo_survives_every_labeler_swap(kw):
    """``route_target == _NAV_TO_ROUTE[nav_cmd]`` on EVERY CE-eligible window,
    under all three labelers. This is why PC1 needs LEVER A and not merely a
    better label: the command and the target are the same derivation, and
    ``_ROUTE_TO_NAV`` is a bijection, so the route head can always reach CE 0 by
    copying its own conditioning embedding to its logits.

    MEASURED equivalent on real data: echo rate on the nav_valid subset =
    1.0000 for v1, v2 and v2.1 (100-ep PhysicalAI cache, 17 100 windows)."""
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    ds = _ds(eps, cfg, plan, **kw)
    n = 0
    for i in _sample(ds, per_ep=12):
        it = ds[i]
        if not bool(it["nav_valid"]):
            continue
        assert int(it["route_target"]) == R._NAV_TO_ROUTE[int(it["nav_cmd"])]
        n += 1
    assert n >= 5, "need CE-eligible windows for the claim to mean anything"


# --------------------------------------------------------------------------- #
# (d) the loss runs finite on a v2.1 batch, mask honoured, LEVER A alive        #
# --------------------------------------------------------------------------- #
def test_flagship_loss_finite_on_v21_batch_with_lever_a():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_labels = True
    cfg.v21_route_labels = True
    cfg.v2_route_from_vision = True             # LEVER A — the real PC1 fix
    plan = horizon_plan(cfg, **FAST)
    eps = _eps_list()
    ds = _wrap(eps, cfg, plan, cfg.encoder.in_channels)
    assert ds.labels_v21 is True

    pick = (_ep_windows(ds, 4, 3) + _ep_windows(ds, 5, 3)
            + _ep_windows(ds, 0, 2) + _ep_windows(ds, 1, 2))
    batch = default_collate([ds[i] for i in pick])
    assert 0 < int(batch["nav_valid"].sum()) <= len(pick)
    # ROUTE_UNKNOWN really is present in the raw targets and really is masked
    if int(batch["nav_valid"].sum()) < len(pick):
        assert int(batch["route_target"].max()) == R.ROUTE_UNKNOWN
    assert int(batch["route_target"][batch["nav_valid"]].max()) < 3

    m = WorldModel(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    total, log, parts = flagship_loss(
        m, grounding, batch, states, fut_states, plan, cfg,
        weights=LossWeights(), sigreg_variant="full_relaxed",
        sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
        fwd_step_weight=0.5, device="cpu")
    assert torch.isfinite(total)
    assert math.isfinite(log["route"]) and math.isfinite(log["route_vis"])
    assert log["route_vis"] > 0.0            # the non-circular aux really ran
    total.backward()
    # LEVER A's gradient must reach the route head (the seam PC1 depends on)
    g = [p.grad for n, p in m.strategic_policy.named_parameters()
         if "route_head" in n and p.grad is not None]
    assert g and any(float(x.abs().sum()) > 0 for x in g)


def test_masked_in_route_unknown_fails_loud():
    """The PC1 guard in flagship_losses: if nav_valid and route_target ever
    disagree, the loss must ASSERT, never coerce class 3 into a 3-way CE."""
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_labels = True
    cfg.v21_route_labels = True
    plan = horizon_plan(cfg, **FAST)
    eps = _eps_list()
    ds = _wrap(eps, cfg, plan, cfg.encoder.in_channels)
    pick = _ep_windows(ds, 4, 4) + _ep_windows(ds, 0, 4)
    batch = default_collate([ds[i] for i in pick])
    batch["route_target"] = torch.full_like(batch["route_target"],
                                            R.ROUTE_UNKNOWN)
    batch["nav_valid"] = torch.ones_like(batch["nav_valid"])   # broken mask

    m = WorldModel(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    with pytest.raises(AssertionError, match="route_target contains class"):
        flagship_loss(m, grounding, batch, states, fut_states, plan, cfg,
                      weights=LossWeights(), sigreg_variant="full_relaxed",
                      sigreg_free_dims=cfg.loss.sigreg.free_dims,
                      pose_scale=10.0, fwd_step_weight=0.5, device="cpu")


# --------------------------------------------------------------------------- #
# (e) CLI threading                                                             #
# --------------------------------------------------------------------------- #
def _run_main_cfg(tmp_path, name, extra):
    out = tmp_path / name
    T4.main(["--data", "toy", "--config", "smoke", "--out", str(out),
             "--episodes", "6", "--steps", "0", "--batch-size", "4",
             "--op-fwd-k", "2", "--tac-fwd-k", "3", "--str-fwd-k", "4",
             *extra])
    outer = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return json.loads(outer["cfg"])


def test_v21_defaults_off_even_under_v2(tmp_path):
    """--v2 must NOT silently turn v2.1 on: it changes what nav_valid means on
    ~75 % of windows, so every shipped --v2 arm would stop being comparable."""
    cfg = _run_main_cfg(tmp_path, "v2only", ["--v2", "--rollout-k", "2"])
    assert cfg["v2_labels"] is True
    assert cfg["v21_route_labels"] is False


def test_labels_v21_flag_threads(tmp_path):
    cfg = _run_main_cfg(tmp_path, "v21", ["--v2", "--rollout-k", "2",
                                          "--labels-v21"])
    assert cfg["v21_route_labels"] is True
    assert cfg["v2_route_from_vision"] is True       # LEVER A pairing enforced


def test_labels_v21_without_lever_a_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="half-fix"):
        _run_main_cfg(tmp_path, "v21_bad", ["--labels-v21"])


def test_labels_v21_labels_only_control_arm_is_allowed(tmp_path):
    """The explicit control arm (labels changed, LEVER A off) must still be
    runnable — it is the contrast that isolates the label effect — but only
    when the operator says so with --route-vis-weight 0."""
    cfg = _run_main_cfg(tmp_path, "v21_ctrl",
                        ["--labels-v21", "--route-vis-weight", "0"])
    assert cfg["v21_route_labels"] is True
    assert cfg["v2_route_from_vision"] is False


def test_lever_a_standalone_flag(tmp_path):
    cfg = _run_main_cfg(tmp_path, "levera", ["--v2-route-from-vision"])
    assert cfg["v2_route_from_vision"] is True
    assert cfg["v2_labels"] is False                 # nothing else came along


def test_no_lever_a_overrides_v2(tmp_path):
    cfg = _run_main_cfg(tmp_path, "nolevera",
                        ["--v2", "--rollout-k", "2",
                         "--no-v2-route-from-vision"])
    assert cfg["v2_route_from_vision"] is False
