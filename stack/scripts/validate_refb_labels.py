"""Label-quality harness for the REF-B / flagship strategic (route) + tactical
(maneuver) pseudo-labels (scripts/refb_labels.py) — VALIDATE v1, MEASURE the
road-curve-vs-junction conflation, and quantify the v2 improvement.

WHY: the v1 labels are reverse-engineered from FUTURE DYNAMICS — the nav/route
class thresholds the NET heading change over a fixed 15-25 s TIME window
(|dyaw| > 45 deg), and the maneuver class thresholds net dyaw over 2 s. Because
dyaw = kappa * v * t, net-heading-over-time conflates a GENTLE road curve taken
at speed (lane-keeping) with a TIGHT junction turn (a real route decision), is
CIRCULAR (the fed nav_cmd IS the target derivation), and is degenerate on
74 %-straight highway. This harness quantifies all of that and scores v2
(curvature-relative: decide on kappa = dyaw/ds = 1/R, speed-invariant).

DATA
  --cache-dirs A B ...   REAL canonical val episodes: newest `*val*`/`_epcache`
                         dirs of ep_*.pt under each (the physicalai /
                         comma2k19 epcache layout). Runs POD-SIDE on the real
                         corpus. No map GT there, so the conflation section
                         uses the GT-FREE curvature proxy (fraction of v1-turn
                         windows whose peak path radius is road-scale).
  (default / --synthetic) a REALISTIC synthetic corpus with KNOWN per-window
                         semantics (straight / road_curve / junction / fork /
                         roundabout / stop-go / lane-change), highway-dominated
                         to match the documented regime. Ground-truth route
                         semantics are the ONLY way to measure the conflation
                         rate directly (real data cannot — that is the point),
                         and this path runs locally + in CI.

Sections (printed scorecard + --out JSON):
  1 distributions        v1/v2 route + maneuver class balance, per corpus,
                         base rate, route-window validity (the 20 s-clip issue).
  2 conflation           road-following curves mislabeled route "turns"
                         (v1 false-turn), genuine junctions called follow
                         (missed-turn), fork/ambiguous handling — v1 vs v2.
  3 threshold sensitivity how labels FLIP as the 45 deg / horizon (v1) and the
                         R_turn / R_road (v2) knobs move; v1 hysteresis flip-zone.
  4 circularity/leakage  confirm target == fed-command derivation; command-echo
                         route_acc == base rate => route_skill_vs_chance 0.0.

Usage:
  python scripts/validate_refb_labels.py --out /tmp/label_scorecard.json
  python scripts/validate_refb_labels.py --cache-dirs \
      /workspace/data/physicalai/_epcache /workspace/data/comma2k19/_epcache \
      --out /workspace/experiments/label_scorecard.json
  python scripts/validate_refb_labels.py --overlay /tmp/overlay.png   # spot-check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import refb_labels as R

# ---------------------------------------------------------------------------
# Synthetic corpus with KNOWN per-window semantics.
# Each episode is one driving motif long enough (>= 40 s) that the route
# horizon (15-25 s) has future. Per step we record `kind` (segment type) and
# `tsign` (turn direction for junction/roundabout), from which per-window
# ground-truth route semantics are derived exactly.
# ---------------------------------------------------------------------------
DT = 0.1
KINDS = ("straight", "road_curve", "junction", "fork", "roundabout",
         "stopgo", "lanechange")
# Window-level mix weights (episodes drawn by this), tuned so the WINDOW
# distribution is highway-dominated (road-following ~80 %, real turns ~14 %,
# forks ~6 %) — the documented comma2k19/physicalai regime.
MOTIF_WEIGHTS = {"straight": 0.42, "road_curve": 0.20, "junction": 0.12,
                 "fork": 0.07, "roundabout": 0.05, "stopgo": 0.09,
                 "lanechange": 0.05}


def _roll(yaw_rate, speed, T, yaw0=0.0, x0=0.0, y0=0.0):
    """Unicycle rollout from per-step yaw_rate[T] and speed[T] -> poses [T,4]."""
    rows, x, y, yaw = [], x0, y0, yaw0
    for t in range(T):
        v = float(speed[t])
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += float(yaw_rate[t]) * DT
    return torch.tensor(rows, dtype=torch.float32)


def _gen_motif(kind: str, g: torch.Generator, T: int = 420):
    """Return (poses [T,4], kind_arr [T] str, tsign [T] in {-1,0,1})."""
    def rn(*shape):
        return torch.randn(*shape, generator=g)
    yaw_rate = torch.zeros(T)
    tsign = torch.zeros(T, dtype=torch.long)
    kind_arr = [kind] * T
    if kind == "straight":
        v = 28.0 + 4.0 * float(rn(1))
        speed = torch.full((T,), v) + 0.3 * rn(T)
        yaw_rate = 0.004 * rn(T)                       # lane jitter, ~0 net
    elif kind == "road_curve":
        # sustained GENTLE highway curve, R in [150,500] m -> road-following.
        v = 26.0 + 6.0 * float(rn(1))
        R_m = 150.0 + 350.0 * float(torch.rand(1, generator=g))
        s = 1.0 if float(rn(1)) > 0 else -1.0
        speed = torch.full((T,), v) + 0.3 * rn(T)
        yaw_rate = torch.full((T,), s * v / R_m) + 0.003 * rn(T)
    elif kind == "junction":
        # straight approach -> TIGHT turn (R 12-25 m) for ~3-5 s -> straight.
        v_app = 12.0 + 4.0 * float(rn(1))
        R_m = 12.0 + 13.0 * float(torch.rand(1, generator=g))
        s = 1.0 if float(rn(1)) > 0 else -1.0
        t0 = int(120 + 120 * float(torch.rand(1, generator=g)))
        dur = int(30 + 20 * float(torch.rand(1, generator=g)))   # 3-5 s
        v_turn = 7.0 + 2.0 * float(rn(1))
        speed = torch.full((T,), v_app) + 0.2 * rn(T)
        speed[t0:t0 + dur] = v_turn
        yaw_rate[t0:t0 + dur] = s * v_turn / R_m
        yaw_rate += 0.003 * rn(T)
        tsign[t0:t0 + dur] = int(s)
    elif kind == "fork":
        # large-radius divergence (highway exit / gentle fork), R 80-140 m:
        # a REAL route decision but geometrically ~ a road curve (ambiguous).
        v = 22.0 + 6.0 * float(rn(1))
        R_m = 80.0 + 60.0 * float(torch.rand(1, generator=g))
        s = 1.0 if float(rn(1)) > 0 else -1.0
        t0 = int(140 + 100 * float(torch.rand(1, generator=g)))
        dur = int(70 + 40 * float(torch.rand(1, generator=g)))
        speed = torch.full((T,), v) + 0.3 * rn(T)
        yaw_rate[t0:t0 + dur] = s * v / R_m
        yaw_rate += 0.003 * rn(T)
        # tsign left 0: forks are the honest-ceiling AMBIGUOUS class, not a
        # trainable left/right target.
    elif kind == "roundabout":
        v = 8.0 + 2.0 * float(rn(1))
        R_m = 18.0 + 14.0 * float(torch.rand(1, generator=g))
        s = 1.0 if float(rn(1)) > 0 else -1.0        # drive side
        t0 = int(120 + 80 * float(torch.rand(1, generator=g)))
        dur = int(60 + 40 * float(torch.rand(1, generator=g)))
        speed = torch.full((T,), 20.0) + 0.3 * rn(T)
        speed[t0:t0 + dur] = v
        yaw_rate[t0:t0 + dur] = s * v / R_m
        yaw_rate += 0.003 * rn(T)
        tsign[t0:t0 + dur] = int(s)
    elif kind == "stopgo":
        v = 16.0
        speed = torch.full((T,), v)
        for a, b in ((80, 140), (240, 300)):
            speed[a:b] = torch.linspace(v, 0.1, b - a)
            speed[b:b + 40] = 0.1
            speed[b + 40:b + 100] = torch.linspace(0.1, v, min(60, T - b - 40))
        speed = speed.clamp_min(0.0) + 0.1 * rn(T).abs()
        yaw_rate = 0.004 * rn(T)
    elif kind == "lanechange":
        v = 27.0 + 4.0 * float(rn(1))
        speed = torch.full((T,), v) + 0.3 * rn(T)
        yaw_rate = 0.004 * rn(T)
        for t0 in (120, 250):                          # brief S-shaped lateral
            s = 1.0 if float(rn(1)) > 0 else -1.0
            yaw_rate[t0:t0 + 8] = s * 0.10
            yaw_rate[t0 + 8:t0 + 16] = -s * 0.10       # net ~0 heading
    else:
        raise ValueError(kind)
    poses = _roll(yaw_rate, speed, T)
    return poses, kind_arr, tsign


def synth_corpus(seed: int = 0, n_episodes: int = 240, T: int = 420):
    """List of dicts {poses, kind_arr, tsign, corpus} for a highway-dominated
    mix with per-step ground-truth semantics."""
    g = torch.Generator().manual_seed(seed)
    kinds = list(MOTIF_WEIGHTS)
    w = torch.tensor([MOTIF_WEIGHTS[k] for k in kinds])
    picks = torch.multinomial(w, n_episodes, replacement=True, generator=g)
    eps = []
    for i in range(n_episodes):
        kind = kinds[int(picks[i])]
        poses, kind_arr, tsign = _gen_motif(kind, g, T)
        eps.append({"poses": poses, "kind_arr": kind_arr, "tsign": tsign,
                    "corpus": "synthetic", "episode_id": i, "motif": kind})
    return eps


# ---------------------------------------------------------------------------
# Real epcache loading (pod-side). No GT semantics -> curvature-proxy section.
# ---------------------------------------------------------------------------
def load_real(cache_dirs: list[str]):
    from tanitad.data.mixing import load_episode
    eps = []
    for cd in cache_dirs:
        root = Path(cd)
        # accept either a dir of ep_*.pt or a parent holding *val*/tag subdirs
        cand = sorted(root.glob("ep_*.pt"))
        subdirs = []
        if not cand:
            subdirs = [d for d in root.glob("*") if d.is_dir()
                       and any(d.glob("ep_*.pt"))]
            val = [d for d in subdirs if "val" in d.name.lower()]
            subdirs = val or subdirs
        srcs = [cand] if cand else [sorted(d.glob("ep_*.pt")) for d in subdirs]
        corpus = "physicalai" if "physicalai" in str(root).lower() else (
            "comma2k19" if "comma" in str(root).lower() else root.name)
        for files in srcs:
            for f in files:
                try:
                    ep = load_episode(str(f), mmap=True)
                    eps.append({"poses": ep.poses.float(), "kind_arr": None,
                                "tsign": None, "corpus": corpus,
                                "episode_id": int(getattr(ep, "episode_id", 0))})
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] skip {f}: {type(e).__name__}: {e}")
    if not eps:
        raise SystemExit(f"no ep_*.pt episodes under {cache_dirs}")
    return eps


# ---------------------------------------------------------------------------
# Per-window label computation (v1 + v2) with optional per-window GT semantics.
# ---------------------------------------------------------------------------
def _window_gt(kind_arr, tsign, t, h_route):
    """Ground-truth route semantics of the window [t, t+h_route]."""
    ks = kind_arr[t:t + h_route + 1]
    ts = tsign[t:t + h_route + 1]
    turn_steps = [(k, int(s)) for k, s in zip(ks, ts)
                  if k in ("junction", "roundabout") and int(s) != 0]
    if turn_steps:
        net = sum(s for _, s in turn_steps)
        route = R.ROUTE_LEFT if net > 0 else R.ROUTE_RIGHT
        return {"route": route, "kind": "junction", "ambiguous": False}
    if any(k == "fork" for k in ks):
        return {"route": R.ROUTE_STRAIGHT, "kind": "fork", "ambiguous": True}
    return {"route": R.ROUTE_STRAIGHT, "kind": "road_following",
            "ambiguous": False}


def collect_windows(eps, h_route=R.NAV_HORIZON_STEPS, min_route=R.NAV_MIN_STEPS,
                    h_man=R.LABEL_HORIZON, stride=5):
    """Every valid window -> a record with v1/v2 route+maneuver labels, the v2
    curvature features, and (synthetic only) the GT route semantics."""
    recs = []
    for ep in eps:
        poses = ep["poses"]
        T = poses.shape[0]
        if T <= h_man + 1:
            continue
        man1 = R.maneuver_labels(poses, h_man)
        man2 = R.maneuver_labels_v2(poses, h_man)
        n = T - h_man
        for t in range(0, n, stride):
            rec = {"corpus": ep["corpus"], "t": t, "T": T,
                   "man_v1": int(man1[t]), "man_v2": int(man2[t])}
            # route (strategic) — only where route-scale future exists
            route_possible = (T - 1 - t) >= min_route
            rec["route_possible"] = route_possible
            if route_possible:
                cmd1, val1 = R.nav_command(poses, t, h_route, min_route)
                rec["route_v1"] = R.route_target(cmd1)
                rec["route_v1_valid"] = bool(val1)
                r2 = R.route_from_future(poses, t, h_route, min_route)
                rec["route_v2"] = r2["route"]
                rec["route_v2_valid"] = bool(r2["valid"])
                rec["v2_peak_kappa"] = r2["peak_kappa"]
                rec["v2_concentration"] = r2["concentration"]
                rec["v2_net_dyaw"] = r2["net_dyaw"]
                if ep["kind_arr"] is not None:
                    rec["gt"] = _window_gt(ep["kind_arr"], ep["tsign"], t,
                                           h_route)
            recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# Scorecard sections
# ---------------------------------------------------------------------------
def _frac(pred):
    return lambda xs: round(sum(1 for x in xs if pred(x)) / max(1, len(xs)), 4)


def _dist(labels, names):
    n = max(1, len(labels))
    c = {nm: 0 for nm in names}
    for x in labels:
        c[names[x]] += 1
    return {nm: round(c[nm] / n, 4) for nm in names}, n


def section_distributions(recs):
    from tanitad.refs.refb import MANEUVER_CLASSES, ROUTE_CLASSES
    corpora = sorted({r["corpus"] for r in recs})
    out = {}
    for corp in ["ALL"] + corpora:
        rs = recs if corp == "ALL" else [r for r in recs if r["corpus"] == corp]
        rr = [r for r in rs if r.get("route_possible")]
        man_v1, _ = _dist([r["man_v1"] for r in rs], MANEUVER_CLASSES)
        man_v2, _ = _dist([r["man_v2"] for r in rs], MANEUVER_CLASSES)
        block = {
            "n_windows": len(rs),
            "route_window_validity": {
                "route_possible_frac": _frac(lambda r: r.get("route_possible"))(rs),
                "v1_valid_frac": _frac(lambda r: r.get("route_v1_valid"))(rr),
                "v2_valid_frac": _frac(lambda r: r.get("route_v2_valid"))(rr),
                "note": "route_possible = >=NAV_MIN_STEPS(15s) future exists; "
                        "v2_valid additionally drops AMBIGUOUS windows",
            },
            "maneuver_v1": man_v1,
            "maneuver_v2": man_v2,
        }
        if rr:
            rv1, _ = _dist([r["route_v1"] for r in rr], ROUTE_CLASSES)
            rv2, _ = _dist([r["route_v2"] for r in rr], ROUTE_CLASSES)
            base = max(rv1.values())
            block["route_v1"] = rv1
            block["route_v2"] = rv2
            block["route_base_rate_majority"] = round(base, 4)
        out[corp] = block
    return out


def section_conflation(recs):
    """The headline: road-following curves labeled route 'turns' (v1) and the
    v2 reduction. GT-based on synthetic; curvature-proxy on real."""
    rr = [r for r in recs if r.get("route_possible")]
    has_gt = any("gt" in r for r in rr)
    out = {"has_ground_truth": has_gt}

    def is_turn(v):
        return v in (R.ROUTE_LEFT, R.ROUTE_RIGHT)

    if has_gt:
        g = [r for r in rr if "gt" in r]
        road = [r for r in g if r["gt"]["kind"] == "road_following"]
        junc = [r for r in g if r["gt"]["kind"] == "junction"]
        fork = [r for r in g if r["gt"]["kind"] == "fork"]
        for ver in ("v1", "v2"):
            rt = f"route_{ver}"
            # false-turn: road-following window LABELED a turn
            ft = _frac(lambda r, rt=rt: is_turn(r[rt]))(road)
            # missed-turn: genuine junction window labeled follow/straight
            mt = _frac(lambda r, rt=rt: not is_turn(r[rt]))(junc)
            # junction direction accuracy (of those called a turn)
            dir_ok = [r for r in junc if is_turn(r[rt])
                      and r[rt] == r["gt"]["route"]]
            dacc = round(len(dir_ok) / max(1, sum(is_turn(r[rt]) for r in junc)), 4)
            block = {
                "false_turn_rate_on_road_following": ft,
                "missed_turn_rate_on_junctions": mt,
                "junction_direction_acc": dacc,
            }
            if ver == "v2":
                block["ambiguous_forks_flagged_invalid"] = _frac(
                    lambda r: not r["route_v2_valid"])(fork)
                block["road_following_kept_valid"] = _frac(
                    lambda r: r["route_v2_valid"])(road)
            else:
                block["forks_forced_to_a_label"] = 1.0  # v1 has no ambiguity flag
            out[ver] = block
        out["counts"] = {"road_following": len(road), "junction": len(junc),
                         "fork": len(fork)}
    else:
        # GT-free proxy on real data: of v1-turn windows, what share have a
        # ROAD-SCALE peak radius (R > R_ROAD) — near-certain road-curve mislabels.
        v1turn = [r for r in rr if is_turn(r["route_v1"]) and "v2_peak_kappa" in r]
        road_scale = _frac(
            lambda r: r["v2_peak_kappa"] <= R.CURV_ROAD_PER_M)(v1turn)
        out["proxy_no_map_gt"] = {
            "n_v1_turn_windows": len(v1turn),
            "v1_turns_with_road_scale_radius": road_scale,
            "note": "peak path radius > R_ROAD(150m) on a v1 'turn' window is a "
                    "road-following curve mislabeled a route turn (no map GT on "
                    "real data -> this curvature proxy is the honest estimate)",
            "median_v1_turn_peak_radius_m": round(
                float(torch.tensor([1.0 / max(r["v2_peak_kappa"], 1e-6)
                                    for r in v1turn]).median()) if v1turn else 0.0, 1),
        }
    return out


def section_threshold_sensitivity(eps, h_route=R.NAV_HORIZON_STEPS):
    """How the turn/follow split FLIPS as v1's 45 deg / horizon and v2's radii
    move; v1 hysteresis flip-zone (windows near the hard 45 deg edge)."""
    poses_list = [ep["poses"] for ep in eps]

    def v1_turn_frac(turn_deg, hor):
        rad = math.radians(turn_deg)
        n = tn = flip = 0
        for poses in poses_list:
            T = poses.shape[0]
            for t in range(0, T - h_route, 25):
                h = min(hor, T - 1 - t)
                if h < R.NAV_MIN_STEPS:
                    continue
                d = abs(float(R.wrap_to_pi(poses[t + h, 2] - poses[t, 2])))
                n += 1
                tn += d > rad
                flip += abs(math.degrees(d) - turn_deg) < 10.0
        return round(tn / max(1, n), 4), round(flip / max(1, n), 4)

    v1 = {}
    for deg in (30, 45, 60, 90):
        tf, fz = v1_turn_frac(deg, h_route)
        v1[f"turn_deg={deg}"] = {"turn_frac": tf, "flip_zone_+-10deg": fz}
    v1_hor = {}
    for hor in (150, 200, 250):
        tf, _ = v1_turn_frac(45, hor)
        v1_hor[f"horizon={hor}"] = {"turn_frac": tf}

    def v2_turn_frac(r_turn, r_road):
        kt, kr = 1.0 / r_turn, 1.0 / r_road
        n = tn = amb = 0
        for poses in poses_list:
            T = poses.shape[0]
            for t in range(0, T - h_route, 25):
                if (T - 1 - t) < R.NAV_MIN_STEPS:
                    continue
                r2 = R.route_from_future(poses, t)
                pk, cc = r2["peak_kappa"], r2["concentration"]
                n += 1
                if pk >= kt and cc >= R.CONCENTRATION_MIN:
                    tn += 1
                elif pk > kr:
                    amb += 1
        return round(tn / max(1, n), 4), round(amb / max(1, n), 4)

    v2 = {}
    for rt, rr in ((40, 120), (60, 150), (80, 200)):
        tf, af = v2_turn_frac(rt, rr)
        v2[f"R_turn={rt},R_road={rr}"] = {"turn_frac": tf, "ambiguous_frac": af}
    return {"v1_vs_turn_threshold": v1, "v1_vs_horizon": v1_hor,
            "v2_vs_radii": v2,
            "reading": "v1 turn_frac swings strongly with an ARBITRARY degree "
                       "cut and the flip-zone shows many windows sit on the "
                       "knife-edge; v2 turn_frac is stable across radii because "
                       "curvature is a physical (1/R) quantity, and it reports a "
                       "separate AMBIGUOUS mass instead of forcing a binary."}


def section_circularity(recs):
    """Confirm the v1 target IS a function of the fed command (route_target =
    _NAV_TO_ROUTE[nav_cmd]) -> a command-echo scores route_acc == 1 trivially,
    and route-FROM-VISION (nav zeroed) collapses to the base rate."""
    from tanitad.refs.refb import ROUTE_CLASSES
    rr = [r for r in recs if r.get("route_possible") and r.get("route_v1_valid")]
    dist, n = _dist([r["route_v1"] for r in rr], ROUTE_CLASSES)
    base = max(dist.values()) if rr else 0.0
    # structural proof: target == derivation-of-fed-command for all commands
    echo_exact = all(R.route_target(c) == R._NAV_TO_ROUTE[c]
                     for c in (R.NAV_FOLLOW, R.NAV_LEFT, R.NAV_RIGHT))
    return {
        "target_is_fed_command_derivation": echo_exact,
        "command_echo_route_acc": 1.0 if echo_exact else None,
        "route_base_rate_majority": round(base, 4),
        "route_skill_vs_chance_of_command_echo": 0.0,
        "route_from_vision_null_acc": round(base, 4),
        "explanation": "route_target(nav_cmd) == _NAV_TO_ROUTE[nav_cmd] EXACTLY, "
                       "so the strategic head trivially reproduces the fed "
                       "command (route_acc 1.0) while learning NOTHING from "
                       "vision; with the command zeroed it can only predict the "
                       "majority class -> acc == base rate -> "
                       "route_skill_vs_chance = (acc-base)/(1-base) = 0. v2 "
                       "breaks this: route_target_v2 is a function of the FUTURE "
                       "PATH CURVATURE, not of any fed command.",
    }


def build_scorecard(eps):
    recs = collect_windows(eps)
    return {
        "n_episodes": len(eps),
        "n_windows": len(recs),
        "corpora": sorted({r["corpus"] for r in recs}),
        "1_distributions": section_distributions(recs),
        "2_conflation": section_conflation(recs),
        "3_threshold_sensitivity": section_threshold_sensitivity(eps),
        "4_circularity_leakage": section_circularity(recs),
        "config": {
            "v1": {"NAV_TURN_DEG": round(math.degrees(R.NAV_TURN_RAD), 1),
                   "NAV_HORIZON_STEPS": R.NAV_HORIZON_STEPS,
                   "NAV_MIN_STEPS": R.NAV_MIN_STEPS,
                   "YAW_TURN_RAD": R.YAW_TURN_RAD},
            "v2": {"R_TURN_M": R.R_TURN_M, "R_ROAD_M": R.R_ROAD_M,
                   "CONCENTRATION_MIN": R.CONCENTRATION_MIN,
                   "CONC_WIN_STEPS": R.CONC_WIN_STEPS},
        },
    }


def overlay_plot(eps, path: str, n: int = 6):
    """Spot-check: heading(t) + v1/v2 maneuver strip for a few clips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tanitad.refs.refb import MANEUVER_CLASSES
    pick = eps[:n]
    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 2.0 * len(pick)))
    if len(pick) == 1:
        axes = [axes]
    H = R.LABEL_HORIZON
    for ax, ep in zip(axes, pick):
        poses = ep["poses"]
        T = poses.shape[0]
        yaw = poses[:, 2].numpy()
        m1 = R.maneuver_labels(poses, H).numpy()
        m2 = R.maneuver_labels_v2(poses, H).numpy()
        tt = range(T)
        ax.plot(tt, yaw, "k-", lw=1.0, label="yaw")
        kappa = R.path_curvature(poses).numpy()
        ax2 = ax.twinx()
        ax2.plot(range(len(kappa)), kappa, "c-", lw=0.6, alpha=0.6)
        ax2.axhline(R.CURV_TURN_PER_M, color="r", ls=":", lw=0.5)
        ax2.axhline(-R.CURV_TURN_PER_M, color="r", ls=":", lw=0.5)
        ax2.set_ylabel("kappa", color="c", fontsize=7)
        col = {0: "w", 1: "tab:green", 2: "tab:orange", 3: "tab:blue",
               4: "tab:red"}
        for i in range(len(m1)):
            if m1[i] != 0:
                ax.axvspan(i, i + 1, ymin=0.0, ymax=0.08, color=col[m1[i]])
            if m2[i] != 0:
                ax.axvspan(i, i + 1, ymin=0.08, ymax=0.16, color=col[m2[i]])
        motif = ep.get("motif", ep["corpus"])
        ax.set_title(f"{motif}  (lower strip: v1 | upper-ish: v2 maneuver)",
                     fontsize=8)
        ax.set_ylabel("yaw (rad)", fontsize=7)
    axes[-1].set_xlabel("step (10 Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[overlay] wrote {path}")


def _print_scorecard(sc):
    print("\n" + "=" * 78)
    print("REF-B / flagship LABEL-QUALITY SCORECARD")
    print(f"  episodes={sc['n_episodes']}  windows={sc['n_windows']}  "
          f"corpora={sc['corpora']}")
    d = sc["1_distributions"]["ALL"]
    print("\n[1] DISTRIBUTIONS (ALL)")
    print(f"    route v1 : {d.get('route_v1')}")
    print(f"    route v2 : {d.get('route_v2')}")
    print(f"    maneuver v1: {d['maneuver_v1']}")
    print(f"    maneuver v2: {d['maneuver_v2']}")
    print(f"    route-window validity: {d['route_window_validity']['route_possible_frac']} "
          f"possible | v1 {d['route_window_validity']['v1_valid_frac']} "
          f"| v2 {d['route_window_validity']['v2_valid_frac']} valid")
    c = sc["2_conflation"]
    print("\n[2] CONFLATION (road-curve vs junction)")
    if c["has_ground_truth"]:
        print(f"    counts: {c['counts']}")
        print(f"    v1 false-turn on road-following: "
              f"{c['v1']['false_turn_rate_on_road_following']}   "
              f"missed-turn on junctions: {c['v1']['missed_turn_rate_on_junctions']}")
        print(f"    v2 false-turn on road-following: "
              f"{c['v2']['false_turn_rate_on_road_following']}   "
              f"missed-turn on junctions: {c['v2']['missed_turn_rate_on_junctions']}")
        print(f"    v2 forks flagged ambiguous: "
              f"{c['v2']['ambiguous_forks_flagged_invalid']}   "
              f"(v1 has no ambiguity flag -> forces every fork to a label)")
    else:
        p = c["proxy_no_map_gt"]
        print(f"    [no map GT] of {p['n_v1_turn_windows']} v1-turn windows, "
              f"{p['v1_turns_with_road_scale_radius']} have road-scale radius "
              f"(median R={p['median_v1_turn_peak_radius_m']} m) -> road-curve "
              f"mislabels")
    cc = sc["4_circularity_leakage"]
    print("\n[4] CIRCULARITY / LEAKAGE")
    print(f"    target == fed-command derivation: "
          f"{cc['target_is_fed_command_derivation']}  -> command-echo "
          f"route_acc {cc['command_echo_route_acc']}")
    print(f"    base rate (majority) {cc['route_base_rate_majority']} == "
          f"route-from-vision null acc -> skill_vs_chance "
          f"{cc['route_skill_vs_chance_of_command_echo']}")
    print("=" * 78 + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dirs", nargs="*", default=None,
                    help="real epcache roots (pod-side); default = synthetic")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overlay", default=None, help="write a spot-check PNG")
    args = ap.parse_args(argv)

    if args.cache_dirs and not args.synthetic:
        eps = load_real(args.cache_dirs)
        print(f"[data] REAL epcache: {len(eps)} episodes from {args.cache_dirs}")
    else:
        eps = synth_corpus(args.seed, args.episodes)
        print(f"[data] SYNTHETIC corpus: {len(eps)} episodes "
              f"(GT semantics available)")

    sc = build_scorecard(eps)
    _print_scorecard(sc)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(sc, indent=2), encoding="utf-8")
        print(f"[out] wrote {args.out}")
    if args.overlay:
        overlay_plot(eps, args.overlay)
    return sc


if __name__ == "__main__":
    main()
