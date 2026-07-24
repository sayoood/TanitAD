"""`--labels v3` wiring pins (scripts/refc_train.py RouteV3Dataset).

The whole safety argument for shipping a new label set mid-programme is that it
is ADDITIVE: a v3 run must differ from a v21 run only by EXTRA batch fields that
no loss reads yet. If that ever stops being true, a v3 run silently stops being
comparable to every published REF-C number. These tests are that guard.

  (1) `--labels v3` is selectable alongside v1 / v21.
  (2) RouteV3Dataset returns route_target / route_valid / nav_cmd / nav_valid
      BYTE-IDENTICAL to RouteV21Dataset on the same windows.
  (3) it ADDS exactly the documented fields, each a valid index into its token
      table (with the `unknown` sentinel at len(table)).
  (4) the 5-way maneuver target is UNCHANGED in every label set — the LAT x LON
      factorization is a MODEL change, not a label-set change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STACK / "scripts"))

import refb_labels as R                                          # noqa: E402
import refc_train as T                                           # noqa: E402
from tanitad.data.mixing import ToyEpisode                       # noqa: E402


def _episode(T_steps=140, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = 8.0 + torch.zeros(T_steps)
    yaw = torch.cumsum(0.02 * torch.sin(torch.linspace(0, 6, T_steps)), 0)
    x = torch.cumsum(v * torch.cos(yaw) * 0.1, 0)
    y = torch.cumsum(v * torch.sin(yaw) * 0.1, 0)
    poses = torch.stack([x, y, yaw, v], dim=1).float()
    return ToyEpisode(
        frames=torch.randint(0, 255, (T_steps, 3, 32, 32), dtype=torch.uint8,
                             generator=g),
        actions=torch.zeros(T_steps, 2), poses=poses, episode_id=seed)


def _pair():
    eps = [_episode(seed=s) for s in range(2)]
    kw = dict(window=4, max_horizon=25, channels=3)
    return (T.RouteV21Dataset([e for e in eps], **kw),
            T.RouteV3Dataset([e for e in eps], **kw))


def test_labels_v3_is_selectable():
    """The flag itself accepts v3 alongside v1/v21 (the `--label-set` contract:
    a new label set is SELECTABLE, never an in-place mutation of an old one)."""
    import inspect
    src = inspect.getsource(T.main)
    assert '"--labels", choices=("v1", "v21", "v3")' in src
    assert "RouteV3Dataset" in inspect.getsource(T.train)


def test_v3_ce_targets_are_byte_identical_to_v21():
    d21, d3 = _pair()
    assert len(d21) == len(d3) and len(d3) > 4
    for i in range(len(d3)):
        a, b = d21[i], d3[i]
        for k in ("route_target", "route_valid", "nav_cmd", "nav_valid"):
            assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), (i, k)


def test_v3_adds_exactly_the_documented_fields():
    d21, d3 = _pair()
    extra = set(d3[0]) - set(d21[0])
    assert extra == {"route_token_idx", "route_dist_idx", "lat_idx", "lon_idx",
                     "lon_active"}, extra
    for i in range(len(d3)):
        it = d3[i]
        assert 0 <= int(it["route_token_idx"]) <= len(R.ROUTE_V3_TOKENS)
        assert 0 <= int(it["route_dist_idx"]) < len(R.DIST_BAND_TOKENS)
        assert 0 <= int(it["lat_idx"]) <= len(R.LAT_KINEMATIC_TOKENS)
        assert 0 <= int(it["lon_idx"]) <= len(R.LON_KINEMATIC_TOKENS)
        assert it["lon_active"].dtype == torch.bool


def test_the_five_way_maneuver_target_is_unchanged_by_the_label_set():
    """The 5-way collapse is a MODEL defect; no label set silently re-derives it."""
    d21, d3 = _pair()
    for i in range(len(d3)):
        a, b = d21[i], d3[i]
        for k in ("pose_last", "future_poses"):
            assert torch.equal(a[k], b[k])
    pose_last = torch.stack([d3[i]["pose_last"] for i in range(4)])
    fut = torch.stack([d3[i]["future_poses"] for i in range(4)])
    m = R.window_maneuver_labels(pose_last, fut)
    assert m.shape == (4,) and int(m.max()) < 5


def test_v3_label_stats_reports_the_new_mix():
    _, d3 = _pair()
    st = d3.label_stats(n=32, seed=0)
    for k in ("route_counts", "v3_token", "v3_dist_band", "v3_lon", "v3_lat",
              "v3_upgraded", "v3_collapsed_by_5way"):
        assert k in st, (k, sorted(st))
    assert set(st["v3_token"]) <= set(R.ROUTE_V3_TOKENS) | {R.TOKEN_UNKNOWN}
    assert set(st["v3_dist_band"]) <= set(R.DIST_BAND_TOKENS)
