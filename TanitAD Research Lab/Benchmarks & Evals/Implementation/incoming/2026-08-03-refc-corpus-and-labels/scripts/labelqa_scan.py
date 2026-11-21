"""LABEL QUALITY SCAN — every label REF-C would train on, counted, with its denominator.

WHY THIS EXISTS
===============
Three separate briefs have asserted label quality ("the 5-way target is lossy", "the nav
label is degenerate", "route yields only road_following") without a population number
behind them. The programme's own operating standard says a claim that decides a GPU-day
must be MEASURED. This is the instrument that makes the corpus-and-label recommendation
MEASURED rather than INHERITED: it walks an epcache split, re-derives every label family
from the cached ``poses`` [T, 4] and reports, per family:

    * n windows evaluated (THE DENOMINATOR — printed on every line)
    * the class balance
    * the DEGENERACY rate (how often the label is a sentinel / fallback / unusable)
    * the LOSSY rate where one representation destroys information another keeps

It reads ONLY ``poses`` out of each ``ep_*.pt``, so it is CPU-only, needs no GPU and
never touches a training pod.

WHAT IT DOES **NOT** CLAIM
--------------------------
The split it is pointed at is whatever is on disk. It stamps the resolved corpus key and
the parity verdict (``tanitad.data.parity.corpus_key_of``) into the output so a number
produced on a NON-PARITY sample can never be quoted as a parity number by accident.

FAMILIES (the four the PI made binding, plus the route/strategic label)
----------------------------------------------------------------------
TACTICAL     5-way ``man5`` (v1 endpoint gate and v2 curvature gate) vs the FACTORED
             (lat x lon) label. The lossy rate is the fraction of windows whose lateral
             class is a turn AND whose longitudinal class is live — those windows carry
             a longitudinal manoeuvre that the 5-way target CANNOT represent, so no
             decode rule recovers them. This is the number that argues for a label
             change rather than a head change.
STRATEGIC    ``route_from_future_v21`` reason histogram + class share + valid rate, and
             the NAV-INPUT degeneracy: how much of ``nav == follow`` is a real
             road-following judgement and how much is ROUTE_UNKNOWN collapsed onto
             ``follow`` by ``nav_command_v21``.
LONGITUDINAL the target-speed label (dv over the horizon) and its class balance; the
             headway/TTC family is reported ABSENT WITH ITS REASON (the epcache carries
             no lead-agent track — see ``physicalai.py``'s stored fields).
LATERAL      heading-change, curvature and yaw-rate label distributions over the same
             windows.
SITUATION    lane-change / intersection event counts from ``tanitad.data.situations``,
             AND the CAUSALITY AUDIT: the centred-difference ``omega_pre`` / ``alon_pre``
             against their strictly-causal replacements, so the size of the leak the
             docstring admits to is a number rather than an adjective.

USAGE
-----
    python labelqa_scan.py --cache-dir <epcache split dir> [--cache-dir ...] \
        --out results/labelqa_<tag>.json [--stride 8] [--max-episodes N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# --- import the programme's own labelers; never re-implement a threshold here -----
_REPO = Path(__file__).resolve()
while _REPO.name != "TanitAD" and _REPO.parent != _REPO:
    _REPO = _REPO.parent
_STACK = _REPO / "stack"
for p in (str(_STACK), str(_STACK / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import refb_labels as RL                                    # noqa: E402
from tanitad.data import parity as PARITY                   # noqa: E402
from tanitad.data import situations as SIT                  # noqa: E402
from tanitad.refs import refc_tactical as TAC               # noqa: E402

HZ = 10.0
DT = 1.0 / HZ
MAN5 = RL.LANE_KEEP, RL.TURN_LEFT, RL.TURN_RIGHT, RL.ACCELERATE, RL.BRAKE_STOP
MAN5_NAMES = TAC.MAN5_NAMES
ROUTE_NAMES = {RL.ROUTE_LEFT: "left", RL.ROUTE_STRAIGHT: "straight",
               RL.ROUTE_RIGHT: "right", RL.ROUTE_UNKNOWN: "unknown"}
NAV_NAMES = {RL.NAV_FOLLOW: "follow", RL.NAV_LEFT: "left",
             RL.NAV_RIGHT: "right", RL.NAV_STRAIGHT: "straight"}


# --------------------------------------------------------------------------- #
# causal ego channels — the fix under audit (mirrors situations_causal.py)      #
# --------------------------------------------------------------------------- #
def causal_pre_channels(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strictly-causal ``(alon_pre, omega_pre)`` — BACKWARD differences.

    ``situations.kinematics`` builds both on ``np.gradient``, which is a CENTRED
    difference: ``g[t] = (x[t+1] - x[t-1]) / (2 dt)``. The trailing moving average
    applied afterwards cannot undo that — the value at ``t`` still contains
    ``x[t+1]``. A backward difference ``(x[t] - x[t-1]) / dt`` reads nothing past
    ``t``, which is what "strictly causal" means.
    """
    psi = np.unwrap(P[:, 2].astype(np.float64))
    v = P[:, 3].astype(np.float64)
    ns = max(2, int(round(SIT.SMOOTH_S * SIT.HZ)))

    def _bwd(x: np.ndarray) -> np.ndarray:
        d = np.empty_like(x)
        d[1:] = (x[1:] - x[:-1]) / SIT.DT
        d[0] = d[1] if len(x) > 1 else 0.0
        return d

    def _trail(x: np.ndarray) -> np.ndarray:
        return np.convolve(np.pad(x, (ns - 1, 0), mode="edge"),
                           np.ones(ns) / ns, mode="valid")

    return _trail(_bwd(v)), _trail(_bwd(psi))


# --------------------------------------------------------------------------- #
# per-episode scan                                                              #
# --------------------------------------------------------------------------- #
def scan_episode(poses_t: torch.Tensor, stride: int, horizon: int,
                 acc: dict) -> None:
    """Accumulate every family's counters for one episode's poses [T, 4]."""
    T = int(poses_t.shape[0])
    if T <= horizon + 1:
        acc["episodes_too_short"] += 1
        return
    acc["episodes"] += 1
    acc["frames"] += T
    acc["clip_len_s"].append(T / HZ)

    # ---- window grid: every t with a full 2 s future ----------------------- #
    ts = list(range(0, T - horizon, stride))
    if not ts:
        return
    idx = torch.tensor(ts, dtype=torch.long)
    pose_last = poses_t[idx]                                    # [B, 4]
    fut = torch.stack([poses_t[t + 1: t + 1 + horizon] for t in ts])  # [B,H,4]
    acc["windows"] += len(ts)

    # ---- TACTICAL --------------------------------------------------------- #
    m5_v1 = RL.window_maneuver_labels(pose_last, fut, horizon)
    m5_v2 = RL.window_maneuver_labels_v2(pose_last, fut, horizon)
    lat1, lon1 = TAC.window_factored_labels(pose_last, fut, horizon)
    lat2, lon2 = TAC.window_factored_labels_v2(pose_last, fut, horizon)

    for name, arr in (("man5_v1", m5_v1), ("man5_v2", m5_v2)):
        for c, n in Counter(arr.tolist()).items():
            acc[name][MAN5_NAMES[c]] += n
    for name, arr, names in (("lat_v1", lat1, TAC.LAT_CLASSES),
                             ("lat_v2", lat2, TAC.LAT_CLASSES),
                             ("lon_v1", lon1, TAC.LON_CLASSES),
                             ("lon_v2", lon2, TAC.LON_CLASSES)):
        for c, n in Counter(arr.tolist()).items():
            acc[name][names[c]] += n

    # THE LOSSY WINDOWS: a live longitudinal class destroyed by the turn priority
    for tag, lat, lon in (("v1", lat1, lon1), ("v2", lat2, lon2)):
        turn = lat != TAC.LAT_LANE_KEEP
        live = lon != TAC.LON_STEADY
        acc[f"lossy_{tag}"]["turn_and_live_lon"] += int((turn & live).sum())
        acc[f"lossy_{tag}"]["turn"] += int(turn.sum())
        acc[f"lossy_{tag}"]["live_lon"] += int(live.sum())
        acc[f"lossy_{tag}"]["n"] += int(turn.numel())
        # collapse must reproduce the shipped 5-way label exactly (self-check)
        ref = m5_v1 if tag == "v1" else m5_v2
        acc[f"lossy_{tag}"]["collapse_mismatch"] += int(
            (TAC.collapse(lat, lon) != ref).sum())

    # ---- LONGITUDINAL / LATERAL raw targets ------------------------------- #
    dv = (fut[:, horizon - 1, 3] - pose_last[:, 3]).numpy()
    dyaw = RL.wrap_to_pi(fut[:, horizon - 1, 2] - pose_last[:, 2]).numpy()
    seg = (fut[:, 1:horizon, :2] - fut[:, :horizon - 1, :2]).norm(dim=-1)
    arc = (seg.sum(dim=1) + (fut[:, 0, :2] - pose_last[:, :2]).norm(dim=-1)).numpy()
    acc["long"]["dv_abs_sum"] += float(np.abs(dv).sum())
    acc["long"]["v0_sum"] += float(pose_last[:, 3].sum())
    acc["long"]["n"] += len(ts)
    acc["long"]["moving"] += int((pose_last[:, 3].numpy() >= 1.0).sum())
    acc["lat"]["dyaw_abs_sum"] += float(np.abs(dyaw).sum())
    acc["lat"]["kappa_abs_sum"] += float(
        np.abs(dyaw / np.maximum(arc, RL.MIN_ARC_M)).sum())
    acc["lat"]["yawrate_abs_sum"] += float(np.abs(dyaw / (horizon * DT)).sum())
    acc["lat"]["n"] += len(ts)
    acc["_dv"].append(dv.astype(np.float32))
    acc["_v0"].append(pose_last[:, 3].numpy().astype(np.float32))
    acc["_dyaw"].append(dyaw.astype(np.float32))
    acc["_kappa"].append((dyaw / np.maximum(arc, RL.MIN_ARC_M)).astype(np.float32))

    # ---- STRATEGIC (route / nav) ------------------------------------------ #
    for t in ts:
        r = RL.route_from_future_v21(poses_t, int(t))
        acc["route_reason"][r["reason"]] += 1
        acc["route_class"][ROUTE_NAMES[r["route"]]] += 1
        acc["route_valid"] += int(bool(r["valid"]))
        acc["route_n"] += 1
        nav, nvalid = RL.nav_command_v21(poses_t, int(t))
        acc["nav_class"][NAV_NAMES[nav]] += 1
        if nav == RL.NAV_FOLLOW:
            # the degeneracy: is this a JUDGEMENT or a collapsed sentinel?
            key = ("from_road_following" if r["route"] == RL.ROUTE_STRAIGHT
                   else "from_UNKNOWN_sentinel")
            acc["nav_follow_provenance"][key] += 1
        acc["_graded"].append(float(r["mean_curv"]))
        acc["_arc"].append(float(r["arc_m"]))

    # ---- SITUATION + causality audit -------------------------------------- #
    P = poses_t.numpy().astype(np.float64)
    K = SIT.kinematics(P)
    lc = SIT.detect_lane_change(K)
    ev, turns, _x = SIT.detect_intersection(K, cross=None)
    rb = SIT.detect_roundabout(K)
    acc["sit"]["lane_change_events"] += len(lc)
    acc["sit"]["intersection_events"] += len(ev)
    acc["sit"]["turn_events"] += len(turns)
    acc["sit"]["roundabout_events"] += len(rb)
    acc["sit"]["episodes_with_lane_change"] += int(bool(lc))
    acc["sit"]["episodes_with_intersection"] += int(bool(ev))

    a_c, o_c = causal_pre_channels(P)
    acc["_causal"].append(np.stack([
        K["alon_pre"] - a_c, K["omega_pre"] - o_c,
        np.abs(K["alon_pre"]), np.abs(K["omega_pre"])], 1).astype(np.float64))


def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 4) if b else float("nan")


def _hist(x: np.ndarray, edges) -> dict:
    h, _ = np.histogram(x, bins=edges)
    return {f"[{edges[i]},{edges[i+1]})": int(h[i]) for i in range(len(h))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8,
                    help="window stride (8 = the canonical TanitEval grid)")
    ap.add_argument("--horizon", type=int, default=RL.LABEL_HORIZON)
    ap.add_argument("--max-episodes", type=int, default=0)
    a = ap.parse_args()

    acc: dict = {
        "episodes": 0, "episodes_too_short": 0, "frames": 0, "windows": 0,
        "clip_len_s": [], "route_valid": 0, "route_n": 0,
        "man5_v1": Counter(), "man5_v2": Counter(),
        "lat_v1": Counter(), "lat_v2": Counter(),
        "lon_v1": Counter(), "lon_v2": Counter(),
        "lossy_v1": Counter(), "lossy_v2": Counter(),
        "route_reason": Counter(), "route_class": Counter(),
        "nav_class": Counter(), "nav_follow_provenance": Counter(),
        "long": Counter(), "lat": Counter(), "sit": Counter(),
        "_dv": [], "_v0": [], "_dyaw": [], "_kappa": [], "_graded": [],
        "_arc": [], "_causal": [],
    }

    splits = []
    t0 = time.time()
    for cd in a.cache_dir:
        d = Path(cd)
        files = sorted(d.glob("ep_*.pt"))
        if a.max_episodes:
            files = files[: a.max_episodes]
        key = PARITY.corpus_key_of(d)
        splits.append({
            "cache_dir": str(d), "n_episode_files": len(sorted(d.glob("ep_*.pt"))),
            "n_scanned": len(files), "resolved_corpus_key": key,
            "is_registered_parity_corpus": key is not None,
            "parity_note": (
                "REGISTERED" if key else
                "NOT a registered parity corpus (tanitad.data.parity.corpus_key_of "
                "returned None) — numbers off it are NOT cross-arm comparable with "
                "the parity arms and must never be quoted as parity numbers."),
            "skip_markers_present": len(PARITY.scan_skip_markers(d)),
        })
        for f in files:
            d_ = torch.load(f, map_location="cpu", weights_only=False)
            scan_episode(d_["poses"].to(torch.float32), a.stride, a.horizon, acc)

    n = acc["windows"]
    dv = np.concatenate(acc["_dv"]) if acc["_dv"] else np.zeros(0)
    v0 = np.concatenate(acc["_v0"]) if acc["_v0"] else np.zeros(0)
    dyaw = np.concatenate(acc["_dyaw"]) if acc["_dyaw"] else np.zeros(0)
    kap = np.concatenate(acc["_kappa"]) if acc["_kappa"] else np.zeros(0)
    C = np.concatenate(acc["_causal"]) if acc["_causal"] else np.zeros((0, 4))

    out = {
        "tool": "labelqa_scan.py",
        "evidence_class": "MEASURED",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "splits": splits,
        "grid": {"stride": a.stride, "horizon_steps": a.horizon,
                 "horizon_s": a.horizon / HZ,
                 "note": "one window per stride-th frame that has a FULL 2 s future"},
        "DENOMINATORS": {
            "episodes_scanned": acc["episodes"],
            "episodes_too_short_for_a_window": acc["episodes_too_short"],
            "frames": acc["frames"],
            "windows": n,
            "median_clip_len_s": round(float(np.median(acc["clip_len_s"])), 2)
            if acc["clip_len_s"] else None,
            "min_clip_len_s": round(float(np.min(acc["clip_len_s"])), 2)
            if acc["clip_len_s"] else None,
            "max_clip_len_s": round(float(np.max(acc["clip_len_s"])), 2)
            if acc["clip_len_s"] else None,
        },
    }

    # ---- TACTICAL --------------------------------------------------------- #
    def _bal(c: Counter, names) -> dict:
        tot = sum(c.values())
        return {k: {"n": c.get(k, 0), "pct": _pct(c.get(k, 0), tot)}
                for k in names} | {"_n": tot}

    lossy = {}
    for tag in ("v1", "v2"):
        L = acc[f"lossy_{tag}"]
        lossy[tag] = {
            "n_windows": L["n"],
            "n_turn": L["turn"], "pct_turn": _pct(L["turn"], L["n"]),
            "n_live_longitudinal": L["live_lon"],
            "pct_live_longitudinal": _pct(L["live_lon"], L["n"]),
            "n_LOSSY_turn_and_live_lon": L["turn_and_live_lon"],
            "PCT_LOSSY": _pct(L["turn_and_live_lon"], L["n"]),
            "pct_of_live_longitudinal_destroyed":
                _pct(L["turn_and_live_lon"], L["live_lon"]),
            "collapse_self_check_mismatches": L["collapse_mismatch"],
            "collapse_self_check":
                "PASS — collapse(lat,lon) == the shipped 5-way label on every window"
                if L["collapse_mismatch"] == 0 else "FAIL",
        }
    out["TACTICAL"] = {
        "man5_v1_balance": _bal(acc["man5_v1"], MAN5_NAMES),
        "man5_v2_balance": _bal(acc["man5_v2"], MAN5_NAMES),
        "lat_v1_balance": _bal(acc["lat_v1"], TAC.LAT_CLASSES),
        "lat_v2_balance": _bal(acc["lat_v2"], TAC.LAT_CLASSES),
        "lon_v1_balance": _bal(acc["lon_v1"], TAC.LON_CLASSES),
        "lon_v2_balance": _bal(acc["lon_v2"], TAC.LON_CLASSES),
        "LOSSY_RATE": lossy,
        "interpretation": (
            "PCT_LOSSY is the fraction of windows on which the 5-way target CANNOT "
            "represent the true state: the lateral class is a turn, so the priority "
            "collapse discards a live longitudinal class. No decode rule recovers "
            "them — the factored (lat x lon) target does, at the same thresholds."),
    }

    # ---- STRATEGIC -------------------------------------------------------- #
    rn = acc["route_n"]
    navf = acc["nav_follow_provenance"]
    out["STRATEGIC"] = {
        "n_windows": rn,
        "route_valid_rate": _pct(acc["route_valid"], rn),
        "route_reason_share": {k: {"n": v, "pct": _pct(v, rn)}
                               for k, v in acc["route_reason"].most_common()},
        "route_class_share": {k: {"n": v, "pct": _pct(v, rn)}
                              for k, v in acc["route_class"].most_common()},
        "nav_input_class_share": {k: {"n": v, "pct": _pct(v, rn)}
                                  for k, v in acc["nav_class"].most_common()},
        "NAV_FOLLOW_DEGENERACY": {
            "n_nav_follow": sum(navf.values()),
            "from_road_following": navf.get("from_road_following", 0),
            "from_UNKNOWN_sentinel": navf.get("from_UNKNOWN_sentinel", 0),
            "pct_of_follow_that_is_a_collapsed_sentinel":
                _pct(navf.get("from_UNKNOWN_sentinel", 0), sum(navf.values())),
            "what_this_measures": (
                "`route_target_v21` refuses to say `straight` when it does not know "
                "(ROUTE_UNKNOWN), but `nav_command_v21` maps ROUTE_UNKNOWN back onto "
                "NAV_FOLLOW so the model INPUT can never tell the two apart. This is "
                "the v2.1 D2 silent-straight-fallback, still alive on the input side."),
        },
        "graded_mean_curv": {
            "n": len(acc["_graded"]),
            "abs_mean": round(float(np.mean(np.abs(acc["_graded"]))), 6)
            if acc["_graded"] else None,
            "note": "threshold-free target; defined on every window the CE must mask",
        },
        "observed_arc_m": {
            "median": round(float(np.median(acc["_arc"])), 2) if acc["_arc"] else None,
            "pct_below_MIN_ARC_ROUTE_M":
                _pct(int(np.sum(np.array(acc["_arc"]) < RL.MIN_ARC_ROUTE_M)),
                     len(acc["_arc"])) if acc["_arc"] else None,
            "pct_below_TRANSIENCE_MIN_ARC_M":
                _pct(int(np.sum(np.array(acc["_arc"]) < RL.TRANSIENCE_MIN_ARC_M)),
                     len(acc["_arc"])) if acc["_arc"] else None,
            "note": ("below TRANSIENCE_MIN_ARC_M the concentration gate is NOT applied "
                     "(route_from_future_v21 R1), so tightness alone decides"),
        },
    }

    # ---- LONGITUDINAL ----------------------------------------------------- #
    out["LONGITUDINAL"] = {
        "n_windows": int(len(dv)),
        "v0_mps": {"mean": round(float(v0.mean()), 4) if len(v0) else None,
                   "median": round(float(np.median(v0)), 4) if len(v0) else None,
                   "pct_stopped_lt_1mps": _pct(int((v0 < 1.0).sum()), len(v0))},
        "dv_2s_mps": {"mean_abs": round(float(np.abs(dv).mean()), 4) if len(dv) else None,
                      "hist": _hist(dv, [-100, -3, -1, -0.3, 0.3, 1, 3, 100])},
        "target_speed_label": {
            "available": True,
            "definition": "v(t+2s) — a pure function of the cached pose track",
            "n": int(len(dv))},
        "distance_keeping_headway_TTC": {
            "available": False,
            "reason": ("the epcache episode record stores only "
                       "{frames_u8, actions, poses, episode_id, maneuvers} — there is "
                       "NO lead-agent track in it, so headway / time-gap / TTC cannot "
                       "be computed from this cache. The 3D agent tracks exist in the "
                       "source corpus as `obstacle.offline` (97.44 % of clips) and "
                       "would have to be joined in at build time."),
            "n": 0},
    }

    # ---- LATERAL ---------------------------------------------------------- #
    out["LATERAL"] = {
        "n_windows": int(len(dyaw)),
        "dyaw_2s_rad": {"mean_abs": round(float(np.abs(dyaw).mean()), 5) if len(dyaw) else None,
                        "hist": _hist(dyaw, [-4, -0.5, -0.15, -0.05, 0.05, 0.15, 0.5, 4])},
        "curvature_1_per_m": {
            "mean_abs": round(float(np.abs(kap).mean()), 6) if len(kap) else None,
            "pct_above_junction_gate_CURV_TURN_PER_M":
                _pct(int((np.abs(kap) >= RL.CURV_TURN_PER_M).sum()), len(kap)),
            "pct_below_road_gate_CURV_ROAD_PER_M":
                _pct(int((np.abs(kap) <= RL.CURV_ROAD_PER_M).sum()), len(kap))},
        "yaw_rate_rad_s": {
            "mean_abs": round(float(np.abs(dyaw / (a.horizon * DT)).mean()), 5)
            if len(dyaw) else None},
    }

    # ---- SITUATION + causality audit -------------------------------------- #
    caus = {}
    if len(C):
        d_a, d_o, m_a, m_o = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
        caus = {
            "n_frames": int(len(C)),
            "alon_pre": {
                "max_abs_diff_mps2": round(float(np.abs(d_a).max()), 6),
                "mean_abs_diff_mps2": round(float(np.abs(d_a).mean()), 6),
                "mean_abs_value_mps2": round(float(m_a.mean()), 6),
                "relative_error_pct": round(
                    100.0 * float(np.abs(d_a).mean()) /
                    max(float(m_a.mean()), 1e-9), 3),
                "pct_frames_differing_gt_1e-6":
                    _pct(int((np.abs(d_a) > 1e-6).sum()), len(C))},
            "omega_pre": {
                "max_abs_diff_rad_s": round(float(np.abs(d_o).max()), 6),
                "mean_abs_diff_rad_s": round(float(np.abs(d_o).mean()), 6),
                "mean_abs_value_rad_s": round(float(m_o.mean()), 6),
                "relative_error_pct": round(
                    100.0 * float(np.abs(d_o).mean()) /
                    max(float(m_o.mean()), 1e-9), 3),
                "pct_frames_differing_gt_1e-6":
                    _pct(int((np.abs(d_o) > 1e-6).sum()), len(C))},
            "what_this_measures": (
                "situations.kinematics builds alon_pre/omega_pre on np.gradient, a "
                "CENTRED difference, then applies a TRAILING mean and calls the result "
                "STRICTLY CAUSAL. The value at t therefore contains the sample at t+1. "
                "The columns above are |centred - strictly-causal-backward-difference| "
                "over every frame of every scanned episode."),
        }
    out["SITUATION"] = {
        "n_episodes": acc["episodes"],
        "events": dict(acc["sit"]),
        "episodes_with_lane_change_pct":
            _pct(acc["sit"]["episodes_with_lane_change"], acc["episodes"]),
        "episodes_with_intersection_pct":
            _pct(acc["sit"]["episodes_with_intersection"], acc["episodes"]),
        "CAUSALITY_AUDIT": caus,
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("DENOMINATORS", "TACTICAL", "STRATEGIC")},
                     indent=1)[:4000])
    print(f"\n[labelqa] wrote {a.out}  ({acc['episodes']} episodes, "
          f"{n} windows, {round(time.time() - t0, 1)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
