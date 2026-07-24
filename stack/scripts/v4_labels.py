"""flagship v4 label mint — the complete per-window target set the v4 heads train
on (V4_FLAGSHIP_DESIGN §6.2 factorised tactical, §4.3 / §7A.4 strategic scalars).

WHY THIS EXISTS. ``train_flagship_v4.v4_loss_step`` reads ``lat_target`` /
``lon_target`` / ``dist_target`` and the strategic goal scalars from the batch;
``FlagshipWindowDataset`` never emitted them, so the factorised CE and the
strategic-scalar loss trained on IGNORE_INDEX for every window — a v4 run would
have measured identically to a v4 *without* its two marquee heads, with no way to
tell why. This module MINTS those labels (kinematic, from poses only) and is the
single authority for the label→index maps the model widths (``flagship_v4.N_LAT``
/ ``N_LON`` / ``N_DIST``) are sized against.

DESIGN CONTRACT — masked, never faked (§6.5):
  * every slot carries a validity mask; an ``unknown`` / out-of-horizon window
    reaches the loss as **IGNORE_INDEX** (a class it never trains), NOT a wrong
    class. A logit no label can train is a dead parameter (spec §2.1);
  * only the KINEMATICALLY mintable tokens are targets. The lead-referenced
    LONMODE modes (``follow_lead`` / ``close_gap`` / ``open_gap``) and the
    context LATMANEUVER modes (``merge_in`` / ``yield_merge``) need agent state
    we do not have; they are never emitted and their windows fall to the ego-only
    modes. The map-dependent ROUTE tokens (``straight`` / robust ``exit``/
    ``merge``/``roundabout``) are minted only on a confirmed kinematic signature
    and otherwise stay the v2.1 token. See ``mintability_report()``.

PARITY. Labels are RE-DERIVED on the existing parity pose cache (same episode ids,
same order) — nothing re-selects, reorders or drops an episode. The v2.1 fields
(``route`` / ``route_graded`` / ``vt_band`` / ``vt_speed``) are minted by the same
calls ``v15_prep.build_labels`` uses, so a v4 cache reproduces them bit-identically
and only ADDS fields (verified by ``build`` against the existing v2.1 cache).

The label CACHE mirrors ``v15_prep`` (``labels_*.pt``: per-field lists of
per-episode tensors, aligned to a poses cache's ``eids``); ``build`` writes it +
a provenance JSON. The on-the-fly path is ``scripts/flagship_v4_data.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))          # refb_labels
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # tanitad

import refb_labels  # noqa: E402

from tanitad.lake.vocab import VTARGET_TOKENS, vtarget_band  # noqa: E402
from tanitad.lake.vtarget import savgol, vtarget_v2  # noqa: E402
from tanitad.train.v4_curriculum import IGNORE_INDEX  # noqa: E402

# --- window geometry: identical to v15_prep (WINDOW/K_MAX) and to the dataset
# --- index (EpisodeWindowDataset builds T - window - max_horizon windows). With
# --- window 8 / max_horizon 20 the label count == v15_prep's T - WINDOW - K_MAX,
# --- so a v4 cache aligns with both the v1.6 loader and the parity window count.
WINDOW = 8
MAX_HORIZON = 20
DT = refb_labels.DT_DEFAULT                       # 0.1 s (10 Hz contract)

# --- factorised tactical vocabularies (the KINEMATIC subsets; sizes pinned to
# --- flagship_v4.N_LAT/N_LON/N_DIST by tests/test_v4_labels.py). Index = the
# --- token's position in these tuples; an invalid/unknown window -> IGNORE_INDEX.
LAT_TOKENS = refb_labels.LAT_KINEMATIC_TOKENS     # 7
LON_TOKENS = refb_labels.LON_KINEMATIC_TOKENS     # 6
DIST_TOKENS = refb_labels.DIST_BAND_TOKENS        # 8 (d_unknown at index 7)
ROUTE_V3_TOKENS = refb_labels.ROUTE_V3_TOKENS     # 9 (strategic goal CE, P6)
LAT_IX = {t: i for i, t in enumerate(LAT_TOKENS)}
LON_IX = {t: i for i, t in enumerate(LON_TOKENS)}
DIST_IX = {t: i for i, t in enumerate(DIST_TOKENS)}
ROUTE_V3_IX = {t: i for i, t in enumerate(ROUTE_V3_TOKENS)}
DIST_UNKNOWN = "d_unknown"

# --- strategic goal scalars: fixed column order shared with the head
# --- (models.strategic_goal) and the loss (v4_curriculum.strategic_scalar_loss).
STRAT_SCALAR_NAMES = ("ttm", "curv_3s", "curv_5s", "tspeed_5s")
CURV_3S_STEPS = 30                                # 3 s @ 10 Hz
CURV_5S_STEPS = 50                                # 5 s @ 10 Hz
TSPEED_STEPS = 50                                 # target speed read at 5 s ahead


# ============================================================================
# per-window kinematic minters (pure; poses [T, 4] = x, y, yaw, v)
# ============================================================================

def lat_target(poses: Tensor, last: int) -> int:
    """LATMANEUVER class index (0..6) or IGNORE_INDEX. ``last`` is the window's
    last pose index (``t + WINDOW - 1``)."""
    r = refb_labels.latmaneuver_from_future(poses, last)
    if r["valid"] and r["token"] in LAT_IX:
        return LAT_IX[r["token"]]
    return IGNORE_INDEX


def lon_target(poses: Tensor, last: int) -> int:
    """LONMODE class index (0..5, incl. ``free_cruise``) or IGNORE_INDEX."""
    r = refb_labels.lonmode_from_future(poses, last)
    if r["valid"] and r["token"] in LON_IX:
        return LON_IX[r["token"]]
    return IGNORE_INDEX


def dist_target_from_route(route3: dict) -> int:
    """DIST band index of the distance to the next junction-scale route maneuver
    (``d_now``..``d_none`` = 0..6) or IGNORE_INDEX for ``d_unknown``. Fed the
    ``route_from_future_v3`` dict so the heavy call is made once per window.

    NOTE the DIST axis is the distance to the next MANEUVER (§4.3, ``route`` v3's
    ``dist_band``); the LON stop-distance (``stop_dist_band``) is minted too as a
    separate field (``stop_dist_target``) for a future tactical-distance A/B, but
    ``dist_target`` is the route maneuver distance."""
    band = route3["dist_band"]
    return IGNORE_INDEX if band == DIST_UNKNOWN else DIST_IX[band]


def stop_dist_target(poses: Tensor, last: int) -> int:
    """DIST band index of the distance to the next STOP point (from LONMODE) or
    IGNORE_INDEX. Extra field — not what ``dist_target`` uses; see above."""
    band = refb_labels.lonmode_from_future(poses, last)["stop_dist_band"]
    return IGNORE_INDEX if band == DIST_UNKNOWN else DIST_IX[band]


def route_token_target(route3: dict) -> int:
    """v3 ROUTE token index (0..8) for the strategic goal CE (§4.3, consumed by
    P6) or IGNORE_INDEX when the token is not a real judgement."""
    if route3.get("token_valid") and route3["token"] in ROUTE_V3_IX:
        return ROUTE_V3_IX[route3["token"]]
    return IGNORE_INDEX


def _future_arc(poses: Tensor, last: int, horizon: int) -> Tensor:
    """Cumulative arc length [h] of the available future from ``last`` (h steps)."""
    T = poses.shape[0]
    h = max(0, min(int(horizon), T - 1 - last))
    if h < 1:
        return poses.new_zeros(0)
    seg = poses[last:last + h + 1, :2]
    return torch.cumsum((seg[1:] - seg[:-1]).norm(dim=-1), 0)


def time_to_maneuver(poses: Tensor, last: int, route3: dict,
                     horizon: int = refb_labels.NAV_HORIZON_STEPS
                     ) -> tuple[float, bool]:
    """Seconds until the ego reaches the START of the next junction-scale route
    maneuver. Derived from the route mint's arc distance ``dist_m`` and the
    realised future arc (time = when cumulative arc first reaches ``dist_m``), so
    it is speed-aware without a divide-by-zero. ``(0.0, False)`` when no maneuver
    is in range (masked)."""
    dm = route3.get("dist_m")
    if dm is None:
        return 0.0, False
    cum = _future_arc(poses, last, horizon)
    if cum.numel() == 0:
        return 0.0, False
    reach = (cum >= float(dm)).nonzero().flatten()
    if reach.numel() == 0:                        # start not actually reached
        return 0.0, False
    return (int(reach[0]) + 1) * DT, True


def mean_curvature(poses: Tensor, last: int, horizon: int) -> tuple[float, bool]:
    """Signed mean path curvature over the next ``horizon`` steps =
    net-heading-change / arc-length (1/m, +left; speed-invariant). Valid iff there
    are ``horizon`` future steps AND enough road travelled to define curvature
    (``MIN_ARC_ROUTE_M``); otherwise ``(0.0, False)`` (masked)."""
    T = poses.shape[0]
    if T - 1 - last < horizon:
        return 0.0, False
    seg = poses[last:last + horizon + 1]
    ds = (seg[1:, :2] - seg[:-1, :2]).norm(dim=-1)
    arc = float(ds.sum())
    if arc < refb_labels.MIN_ARC_ROUTE_M:
        return 0.0, False
    net_dyaw = float(refb_labels.wrap_to_pi(seg[1:, 2] - seg[:-1, 2]).sum())
    return net_dyaw / arc, True


def target_speed_at(poses: Tensor, last: int, horizon: int = TSPEED_STEPS,
                    v_smoothed: np.ndarray | None = None) -> tuple[float, bool]:
    """Smoothed ego speed (m/s) ``horizon`` steps ahead — the ego's own realised
    free-flow speed at 5 s, the honest kinematic target-speed (a lead-limited
    speed is indistinguishable from a chosen one without lead state, so this is
    "what the ego does", not "what it wants"). Valid iff ``horizon`` future steps
    exist. ``v_smoothed`` may be passed to avoid re-smoothing per window."""
    T = poses.shape[0]
    if T - 1 - last < horizon:
        return 0.0, False
    if v_smoothed is None:
        v_smoothed = savgol(poses[:, 3].numpy().astype(np.float64))
    return float(v_smoothed[last + horizon]), True


def strategic_scalars(poses: Tensor, last: int, route3: dict,
                      v_smoothed: np.ndarray | None = None
                      ) -> tuple[list[float], list[bool]]:
    """The four goal scalars + their masks, in ``STRAT_SCALAR_NAMES`` order."""
    ttm, ttm_ok = time_to_maneuver(poses, last, route3)
    c3, c3_ok = mean_curvature(poses, last, CURV_3S_STEPS)
    c5, c5_ok = mean_curvature(poses, last, CURV_5S_STEPS)
    ts, ts_ok = target_speed_at(poses, last, TSPEED_STEPS, v_smoothed)
    return [ttm, c3, c5, ts], [ttm_ok, c3_ok, c5_ok, ts_ok]


# ============================================================================
# per-episode mint (aligned to the dataset window index) + coverage
# ============================================================================

_VT_MEMO: dict[float, int] = {}
_N_VT = len(VTARGET_TOKENS)


def _vt_band_ix(x: float) -> int:
    k = round(float(x), 3)
    if k not in _VT_MEMO:
        _VT_MEMO[k] = list(VTARGET_TOKENS).index(vtarget_band(k))
    return _VT_MEMO[k]


# every per-window field the mint produces (the dataset + the cache read these)
FIELDS_LONG = ("lat_target", "lon_target", "dist_target", "stop_dist_target",
               "route", "route_token", "vt_band")
FIELDS_FLOAT = ("route_graded", "vt_speed")
FIELDS_BOOL = ("route_valid", "vt_valid")


def mint_episode(poses: Tensor, window: int = WINDOW,
                 max_horizon: int = MAX_HORIZON, min_lookahead: int = 50,
                 use_net_dyaw: bool = False) -> dict:
    """Every per-window label for one episode, as tensors of length
    ``n = T - window - max_horizon`` (0 if the episode is too short), aligned to
    the dataset's window index (``t = 0..n-1``, last pose ``t + window - 1``).

    Returns a dict with the tactical targets, the v2.1 conditioning fields
    (bit-identical to ``v15_prep``), the v3 route token, and ``strat_scalars``
    [n, 4] / ``strat_scalar_mask`` [n, 4]."""
    T = poses.shape[0]
    n = T - window - max_horizon
    cols: dict[str, list] = {k: [] for k in
                             FIELDS_LONG + FIELDS_FLOAT + FIELDS_BOOL}
    scal, smask = [], []
    if n <= 0:
        empty_l = {k: torch.zeros(0, dtype=torch.long) for k in FIELDS_LONG}
        empty_f = {k: torch.zeros(0, dtype=torch.float32) for k in FIELDS_FLOAT}
        empty_b = {k: torch.zeros(0, dtype=torch.bool) for k in FIELDS_BOOL}
        return {**empty_l, **empty_f, **empty_b,
                "strat_scalars": torch.zeros(0, 4, dtype=torch.float32),
                "strat_scalar_mask": torch.zeros(0, 4, dtype=torch.bool)}
    v = poses[:, 3].numpy().astype(np.float64)
    vs = savgol(v)
    last_ix = np.arange(window - 1, window - 1 + n, dtype=np.int64)
    vt2, vt_ok, _look, _ = vtarget_v2(v, last_ix, min_lookahead=min_lookahead)
    for i, L in enumerate(last_ix.tolist()):
        r3 = refb_labels.route_from_future_v3(poses, L, use_net_dyaw=use_net_dyaw)
        cols["lat_target"].append(lat_target(poses, L))
        cols["lon_target"].append(lon_target(poses, L))
        cols["dist_target"].append(dist_target_from_route(r3))
        cols["stop_dist_target"].append(stop_dist_target(poses, L))
        cols["route"].append(int(r3["route"]))            # v2.1 class 0..3
        cols["route_token"].append(route_token_target(r3))
        cols["route_graded"].append(float(r3["graded_route"]))
        cols["route_valid"].append(bool(r3["valid"]))
        ok = bool(vt_ok[i])
        cols["vt_band"].append(_vt_band_ix(vt2[i]) if ok else _N_VT)  # 23=DROPPED
        cols["vt_speed"].append(float(vt2[i]))
        cols["vt_valid"].append(ok)
        sc, sm = strategic_scalars(poses, L, r3, v_smoothed=vs)
        scal.append(sc)
        smask.append(sm)
    out = {k: torch.tensor(cols[k], dtype=torch.long) for k in FIELDS_LONG}
    out.update({k: torch.tensor(cols[k], dtype=torch.float32) for k in FIELDS_FLOAT})
    out.update({k: torch.tensor(cols[k], dtype=torch.bool) for k in FIELDS_BOOL})
    out["strat_scalars"] = torch.tensor(scal, dtype=torch.float32)
    out["strat_scalar_mask"] = torch.tensor(smask, dtype=torch.bool)
    return out


def mint_window(poses: Tensor, last: int,
                v_smoothed: np.ndarray | None = None,
                min_lookahead: int = 50, use_net_dyaw: bool = False) -> dict:
    """One window's labels as scalars/tensors — the on-the-fly path
    (``FlagshipV4Dataset``). Bit-identical to the corresponding row of
    :func:`mint_episode` (same calls, same savgol-of-the-whole-track for vt)."""
    r3 = refb_labels.route_from_future_v3(poses, last, use_net_dyaw=use_net_dyaw)
    sc, sm = strategic_scalars(poses, last, r3, v_smoothed=v_smoothed)
    v = poses[:, 3].numpy().astype(np.float64)
    vt2, vt_ok, _l, _ = vtarget_v2(v, np.array([last]), min_lookahead=min_lookahead)
    ok = bool(vt_ok[0])
    return {
        "lat_target": lat_target(poses, last),
        "lon_target": lon_target(poses, last),
        "dist_target": dist_target_from_route(r3),
        "stop_dist_target": stop_dist_target(poses, last),
        "route": int(r3["route"]),
        "route_token": route_token_target(r3),
        "route_graded": float(r3["graded_route"]),
        "route_valid": bool(r3["valid"]),
        "vt_band": _vt_band_ix(vt2[0]) if ok else _N_VT,
        "vt_speed": float(vt2[0]),
        "vt_valid": ok,
        "strat_scalars": torch.tensor(sc, dtype=torch.float32),
        "strat_scalar_mask": torch.tensor(sm, dtype=torch.bool),
    }


def _coverage(ep_fields: dict[str, list]) -> dict:
    """Per-slot coverage = fraction of windows carrying a NON-ignored label — the
    real acceptance metric (a head at 2 % coverage is nearly as dead as IGNORE)."""
    def frac_long(name: str) -> tuple[int, int]:
        tot = valid = 0
        for t in ep_fields[name]:
            tot += t.numel()
            valid += int((t != IGNORE_INDEX).sum())
        return valid, tot

    def frac_bool(name: str) -> tuple[int, int]:
        tot = valid = 0
        for t in ep_fields[name]:
            tot += t.numel()
            valid += int(t.sum())
        return valid, tot

    nw = sum(t.numel() for t in ep_fields["lat_target"]) or 1
    cov = {"n_windows": nw}
    for name in ("lat_target", "lon_target", "dist_target", "stop_dist_target",
                 "route_token"):
        v, _ = frac_long(name)
        cov[name] = round(v / nw, 4)
    for name in ("route_valid", "vt_valid"):
        v, _ = frac_bool(name)
        cov[name] = round(v / nw, 4)
    # strategic-scalar coverage, per column
    sm = ep_fields["strat_scalar_mask"]
    col = np.zeros(4, dtype=np.int64)
    for t in sm:
        if t.numel():
            col += t.sum(dim=0).numpy()
    cov["strat_scalars"] = {STRAT_SCALAR_NAMES[c]: round(int(col[c]) / nw, 4)
                            for c in range(4)}
    # active-mode rates (the tighter number the design quotes: LON != free_cruise)
    lon_active = 0
    for t in ep_fields["lon_target"]:
        lon_active += int(((t != IGNORE_INDEX) & (t != LON_IX["free_cruise"])).sum())
    cov["lon_active_rate"] = round(lon_active / nw, 4)
    lat_active = 0
    for t in ep_fields["lat_target"]:
        lat_active += int(((t != IGNORE_INDEX) & (t != LAT_IX["lane_keep"])).sum())
    cov["lat_active_rate"] = round(lat_active / nw, 4)
    return cov


def mintability_report() -> dict:
    """Honest map of what v4 CAN and CANNOT mint (the task's final ask)."""
    return {
        "fully_minted_kinematic": {
            "lat_target": list(LAT_TOKENS),
            "lon_target": list(LON_TOKENS),
            "dist_target": list(DIST_TOKENS[:-1]),   # all but d_unknown (masked)
            "route (v2.1 class)": ["left", "straight", "right", "unknown(masked)"],
            "route_token (v3)": ["follow", "turn_left", "turn_right",
                                 "u_turn*", "exit_left*", "exit_right*",
                                 "merge*", "roundabout*"],
            "strategic_scalars": list(STRAT_SCALAR_NAMES),
        },
        "not_mintable_needs_data": {
            "LONMODE follow_lead/close_gap/open_gap": "lead_state is a None stub "
                "-> these windows fall to free_cruise/coast; the lead modes are "
                "never emitted (LON = 'what the ego does', not 'why')",
            "LATMANEUVER merge_in/yield_merge": "need another agent's track",
            "ROUTE straight": "asserts a junction exists = a MAP fact; never minted",
            "ROUTE exit/merge/roundabout (*)": "minted only on a confirmed "
                "kinematic signature (low coverage); u_turn is roundabout-"
                "confounded without a map",
            "TACPOINT name (stop_line/crossing/queue)": "position minted "
                "(stop_dist), NAME needs vision/map -> tacpoint stays unknown",
        },
        "masking_rule": "unknown/out-of-horizon -> IGNORE_INDEX, never a class",
    }


# ============================================================================
# corpus cache builder — mirrors v15_prep.build_labels, ADD-ONLY, parity-checked
# ============================================================================

def build(args) -> dict:
    """Mint the full v4 label set over a parity pose cache and write a v15-format
    cache + provenance JSON. If ``--v21-labels`` is given, VERIFY the v2.1 fields
    reproduce bit-identically (parity proof: only new fields are added)."""
    pd = torch.load(args.poses, weights_only=False)
    eids, poses_list = pd["eids"], pd["poses"]
    v21 = torch.load(args.v21_labels, weights_only=False) if args.v21_labels else None
    if v21 is not None and list(v21["eids"]) != list(eids):
        raise SystemExit("REFUSING: v21 label cache eids != poses eids — a "
                         "re-selected/reordered corpus breaks parity.")

    fields: dict[str, list] = {k: [] for k in
                               FIELDS_LONG + FIELDS_FLOAT + FIELDS_BOOL}
    fields["strat_scalars"], fields["strat_scalar_mask"] = [], []
    t0, n_windows = time.time(), 0
    mism = {"route": 0, "route_graded": 0, "vt_band": 0}
    for e, p in enumerate(poses_list):
        po = torch.as_tensor(p, dtype=torch.float32)
        m = mint_episode(po, min_lookahead=args.min_lookahead,
                         use_net_dyaw=bool(args.use_net_dyaw))
        for k in fields:
            fields[k].append(m[k])
        n_windows += int(m["lat_target"].numel())
        if v21 is not None and m["route"].numel():           # parity cross-check
            if not torch.equal(m["route"], v21["route_v21"][e].long()):
                mism["route"] += 1
            if not torch.allclose(m["route_graded"], v21["route_graded"][e], atol=0):
                mism["route_graded"] += 1
            if not torch.equal(m["vt_band"], v21["vt_band_v2"][e].long()):
                mism["vt_band"] += 1
        if e % 200 == 0:
            print(f"  {e}/{len(poses_list)}  {time.time() - t0:.0f}s  "
                  f"windows={n_windows}", flush=True)

    cov = _coverage(fields)
    prov = {
        "artifact": "flagship v4 label cache",
        "poses_source": args.poses,
        "v21_labels_source": args.v21_labels,
        "n_episodes": len(eids),
        "n_windows": n_windows,
        "window": WINDOW, "max_horizon": MAX_HORIZON,
        "min_lookahead": args.min_lookahead, "use_net_dyaw": bool(args.use_net_dyaw),
        "coverage": cov,
        "v21_parity": ("N/A (no --v21-labels)" if v21 is None else
                       {"episodes_with_mismatch": mism,
                        "bit_identical": all(v == 0 for v in mism.values())}),
        "mintability": mintability_report(),
        "parity_key": "physicalai-train-e438721ae894",
        "skip_hash": "f09e44db (unchanged — labels re-derived on the existing "
                     "pose cache; no episode re-selection)",
        "seconds": round(time.time() - t0, 1),
    }
    if v21 is not None and not prov["v21_parity"]["bit_identical"]:
        raise SystemExit(f"PARITY FAILURE: v2.1 fields did not reproduce "
                         f"bit-identically: {mism}. Refusing to write a cache "
                         f"whose v2.1 fields diverge from the shipped v1.6 cache.")
    torch.save({"eids": list(eids), **fields, "provenance": prov}, args.out)
    prov_path = args.provenance or (os.path.splitext(args.out)[0] + "_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps({"saved": args.out, "provenance": prov_path,
                      "n_windows": n_windows, "coverage": cov,
                      "v21_parity": prov["v21_parity"]}, indent=2), flush=True)
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("v4_labels", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", required=True, help="parity pose cache (poses_*.pt)")
    ap.add_argument("--out", required=True, help="output v4 label cache (labels_*_v4.pt)")
    ap.add_argument("--v21-labels", help="existing v2.1 labels_*.pt — enables the "
                    "bit-identity parity check on route/route_graded/vt_band")
    ap.add_argument("--provenance", help="provenance JSON path (default: <out>_provenance.json)")
    ap.add_argument("--min-lookahead", type=int, default=50)
    ap.add_argument("--use-net-dyaw", type=int, default=0,
                    help="match v15_prep default (0 = v2 junction-only semantics)")
    build(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
