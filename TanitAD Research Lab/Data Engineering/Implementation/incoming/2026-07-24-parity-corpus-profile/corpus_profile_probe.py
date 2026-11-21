"""Parity-corpus profiler (CPU-only, READ-ONLY) — TanitAD 2026-07-24.

Profiles the canonical WM-training parity set physicalai-train-e438721ae894
(2376 clips, skip-hash f09e44db) to produce decision-grade real numbers for a
data-enlargement decision: driving hours, timestep count, epoch math, maneuver
histogram (headline scenario distribution), nav/strategic histogram, speed
distribution, turn/stop/junction rarity, and the under-representation ranking.

All numbers are MEASURED by loading each cached ToyEpisode's poses[T,4] and the
PRE-COMPUTED per-timestep maneuvers[T] (v1 kinematic labeler, LABEL_HORIZON=20,
sentinel -1 for the unlabelable tail). Frames stay disk-backed (mmap) and are
never materialized, so RAM stays ~one episode.

Reads NOTHING but the cache; writes ONLY the output JSON. Never modifies the set.
"""
from __future__ import annotations
import glob, json, math, os, sys, time

import torch

CACHE = "/workspace/pai_epcache/physicalai-train-e438721ae894"
OUT = "/workspace/tmp/corpus_profile.json"
HZ = 10.0
WINDOW = 8            # trainer cfg.predictor.window
MAX_HORIZON = 16      # trainer plan.max_horizon (tactical farthest = 16)
EFF_BATCH = 64        # batch-size 16 x accum 4
STEPS = 30000
LABEL_HORIZON = 20    # refb_labels.LABEL_HORIZON (2 s @ 10 Hz) — stored-maneuver horizon
ROUTE_STRIDE = 10     # v2.1 route sampled every 1 s (label is a 15-25 s integral)

# refb_labels lives under stack/scripts
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.append("/workspace/TanitAD/stack/scripts")
import refb_labels as rl  # noqa: E402

MAN_NAMES = ["lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop"]
ROUTE_NAMES = {rl.ROUTE_LEFT: "left", rl.ROUTE_STRAIGHT: "straight",
               rl.ROUTE_RIGHT: "right", rl.ROUTE_UNKNOWN: "unknown"}
NAV_NAMES = {rl.NAV_FOLLOW: "follow", rl.NAV_LEFT: "left", rl.NAV_RIGHT: "right"}


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    files = sorted(glob.glob(os.path.join(CACHE, "ep_*.pt")))
    assert files, f"no ep_*.pt under {CACHE}"
    n_ep = len(files)
    n_skip = len(glob.glob(os.path.join(CACHE, "skip_*")))
    print(f"[profile] {n_ep} episodes, {n_skip} skip files under {CACHE}", flush=True)

    Ts = []
    man_v1 = torch.zeros(5, dtype=torch.long)        # stored (v1 labeler, h=20)
    man_v2 = torch.zeros(5, dtype=torch.long)        # recomputed curvature-gated, h=20
    man_v1_sentinel = 0
    v_all = []                                        # every timestep's speed (m/s)

    # per-clip presence flags (from stored v1 maneuvers)
    clip_has = {k: 0 for k in ["turn_left", "turn_right", "turn_any",
                               "accelerate", "brake_stop"]}
    # nav / route distributions accumulated per-timestep
    nav_v1 = torch.zeros(3, dtype=torch.long)        # follow/left/right (valid only)
    nav_v1_invalid = 0
    route_v21 = torch.zeros(4, dtype=torch.long)     # left/straight/right/unknown
    route_v21_reason = {}                            # reason -> count
    clip_junction = 0                                # >=1 tight_transient turn (v2.1)
    net_head_deg = []                                # |net heading change| per clip (deg)
    cum_abs_head_deg = []                            # total |dyaw| per clip (deg)

    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        poses = d["poses"].float().clone()           # [T,4] copy off mmap
        man = d["maneuvers"].long().clone()          # [T] stored v1 labels
        del d
        T = poses.shape[0]
        Ts.append(T)
        yaw = poses[:, 2]
        v = poses[:, 3]
        v_all.append(v.clone())

        # --- maneuver histogram: stored v1 (headline) ---
        valid = man >= 0
        man_v1_sentinel += int((~valid).sum())
        if valid.any():
            man_v1 += torch.bincount(man[valid], minlength=5)
            present = set(man[valid].tolist())
            if rl.TURN_LEFT in present:
                clip_has["turn_left"] += 1
            if rl.TURN_RIGHT in present:
                clip_has["turn_right"] += 1
            if rl.TURN_LEFT in present or rl.TURN_RIGHT in present:
                clip_has["turn_any"] += 1
            if rl.ACCELERATE in present:
                clip_has["accelerate"] += 1
            if rl.BRAKE_STOP in present:
                clip_has["brake_stop"] += 1

        # --- maneuver histogram: v2 curvature-gated recompute (what the trainer
        #     window label uses; gentle highway curves -> lane_keep) ---
        if T > LABEL_HORIZON:
            m2 = rl.maneuver_labels_v2(poses, horizon=LABEL_HORIZON)
            man_v2 += torch.bincount(m2, minlength=5)

        # --- per-clip heading change (junction/curviness proxy) ---
        dyaw_step = wrap_pi(yaw[1:] - yaw[:-1])
        net_head_deg.append(abs(math.degrees(float(wrap_pi(yaw[-1] - yaw[0])))))
        cum_abs_head_deg.append(math.degrees(float(dyaw_step.abs().sum())))

        # --- nav v1 (refb_labels.nav_command: 15-25 s future heading) ---
        #     VECTORIZED, exact: h=min(NAV_HORIZON_STEPS, T-1-t); end index = t+h;
        #     valid iff h>=NAV_MIN_STEPS; dyaw=wrap(yaw[end]-yaw[t]); >NAV_TURN_RAD
        #     -> left, <-NAV_TURN_RAD -> right, else follow. Matches nav_command().
        tt = torch.arange(T)
        end = torch.minimum(tt + rl.NAV_HORIZON_STEPS,
                            torch.full_like(tt, T - 1))
        h = end - tt
        valid = h >= rl.NAV_MIN_STEPS
        dyaw = wrap_pi(yaw[end] - yaw[tt])
        cmd = torch.zeros(T, dtype=torch.long)                 # follow
        cmd[dyaw > rl.NAV_TURN_RAD] = rl.NAV_LEFT
        cmd[dyaw < -rl.NAV_TURN_RAD] = rl.NAV_RIGHT
        nav_v1 += torch.bincount(cmd[valid], minlength=3)
        nav_v1_invalid += int((~valid).sum())

        # --- route v2.1 (adaptive-horizon, never-straight-by-default) ---
        #     STRIDED (every ROUTE_STRIDE steps) — the label is a ~15-25 s future
        #     integral, so adjacent timesteps are near-identical; a 1 s stride
        #     gives statistically identical fractions at ~STRIDE x less compute.
        clip_is_junction = False
        for t in range(0, T, ROUTE_STRIDE):
            r = rl.route_from_future_v21(poses, t)
            route_v21[r["route"]] += 1
            route_v21_reason[r["reason"]] = route_v21_reason.get(r["reason"], 0) + 1
            if r["reason"] == "tight_transient":
                clip_is_junction = True
        if clip_is_junction:
            clip_junction += 1

        if i % 200 == 0:
            print(f"[profile] {i}/{n_ep}  elapsed {time.time()-t0:.0f}s", flush=True)

    Ts = torch.tensor(Ts, dtype=torch.long)
    total_frames = int(Ts.sum())
    total_windows = int((Ts - WINDOW - MAX_HORIZON).clamp_min(0).sum())
    total_hours = total_frames / HZ / 3600.0
    presentations = STEPS * EFF_BATCH
    epochs = presentations / total_windows

    v_all = torch.cat(v_all)
    n_v = v_all.numel()
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    v_pct = {f"p{p}": float(torch.quantile(v_all, p / 100.0)) for p in pcts}
    v_bins = {
        "stopped_lt1": float((v_all < 1.0).float().mean()),
        "city_1_12": float(((v_all >= 1.0) & (v_all <= 12.0)).float().mean()),
        "highway_gt12": float((v_all > 12.0).float().mean()),
    }
    # finer speed histogram (m/s edges)
    edges = [0, 1, 2, 5, 8, 12, 16, 20, 25, 30, 100]
    v_hist = {}
    for a, b in zip(edges[:-1], edges[1:]):
        v_hist[f"{a}-{b}"] = float(((v_all >= a) & (v_all < b)).float().mean())

    man_v1_total = int(man_v1.sum())
    man_v2_total = int(man_v2.sum())
    man_v1_frac = {MAN_NAMES[k]: float(man_v1[k]) / man_v1_total for k in range(5)}
    man_v2_frac = {MAN_NAMES[k]: float(man_v2[k]) / man_v2_total for k in range(5)}
    nav_v1_total = int(nav_v1.sum())
    nav_v1_frac = {NAV_NAMES[k]: float(nav_v1[k]) / nav_v1_total for k in range(3)}
    route_total = int(route_v21.sum())
    route_v21_frac = {ROUTE_NAMES[k]: float(route_v21[k]) / route_total for k in range(4)}
    route_v21_judgeable = 1.0 - route_v21_frac["unknown"]

    net = torch.tensor(net_head_deg)
    cum = torch.tensor(cum_abs_head_deg)

    out = {
        "corpus": "physicalai-train-e438721ae894",
        "skip_hash": "f09e44db",
        "cache_path": f"tanitad-pod3:{CACHE}",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler_note": ("maneuver headline = STORED per-timestep v1 kinematic "
                         "labeler (refb_labels.maneuver_labels, horizon 20 = 2 s); "
                         "man_v2 = curvature-gated recompute. KINEMATIC ONLY — no "
                         "semantics (lights/signs/peds/roundabouts unlabeled)."),
        "size": {
            "usable_clips": n_ep, "skipped_clips": n_skip,
            "total_frames": total_frames,
            "total_hours": total_hours,
            "total_minutes": total_hours * 60,
            "clip_len_frames_mean": float(Ts.float().mean()),
            "clip_len_frames_median": float(Ts.median()),
            "clip_len_frames_min": int(Ts.min()),
            "clip_len_frames_max": int(Ts.max()),
            "clip_len_s_mean": float(Ts.float().mean()) / HZ,
            "clip_len_s_median": float(Ts.median()) / HZ,
        },
        "epochs": {
            "window": WINDOW, "max_horizon": MAX_HORIZON,
            "windows_per_clip_formula": "max(0, T - window - max_horizon)",
            "total_unique_windows": total_windows,
            "registry_windows_ref": 406099,
            "effective_batch": EFF_BATCH, "steps": STEPS,
            "window_presentations": presentations,
            "epochs": epochs,
        },
        "maneuver_v1_stored": {
            "counts": {MAN_NAMES[k]: int(man_v1[k]) for k in range(5)},
            "fractions": man_v1_frac,
            "valid_timesteps": man_v1_total,
            "sentinel_timesteps": man_v1_sentinel,
        },
        "maneuver_v2_curvaturegated": {
            "counts": {MAN_NAMES[k]: int(man_v2[k]) for k in range(5)},
            "fractions": man_v2_frac,
            "valid_timesteps": man_v2_total,
        },
        "nav_v1_follow_left_right": {
            "counts": {NAV_NAMES[k]: int(nav_v1[k]) for k in range(3)},
            "fractions_over_valid": nav_v1_frac,
            "valid_timesteps": nav_v1_total,
            "invalid_timesteps": nav_v1_invalid,
            "coverage": nav_v1_total / (nav_v1_total + nav_v1_invalid),
        },
        "route_v21": {
            "counts": {ROUTE_NAMES[k]: int(route_v21[k]) for k in range(4)},
            "fractions_all": route_v21_frac,
            "judgeable_fraction": route_v21_judgeable,
            "reason_counts": route_v21_reason,
            "sampled_at_stride": ROUTE_STRIDE,
            "n_samples": route_total,
        },
        "speed_mps": {
            "n_timesteps": n_v,
            "percentiles": v_pct,
            "mean": float(v_all.mean()),
            "regime_fractions": v_bins,
            "histogram": v_hist,
        },
        "turn_stop_junction": {
            "clips_with_turn_left": clip_has["turn_left"],
            "clips_with_turn_right": clip_has["turn_right"],
            "clips_with_any_turn": clip_has["turn_any"],
            "clips_with_any_turn_frac": clip_has["turn_any"] / n_ep,
            "clips_with_accelerate": clip_has["accelerate"],
            "clips_with_accelerate_frac": clip_has["accelerate"] / n_ep,
            "clips_with_brake_stop": clip_has["brake_stop"],
            "clips_with_brake_stop_frac": clip_has["brake_stop"] / n_ep,
            "clips_with_v21_junction_turn": clip_junction,
            "clips_with_v21_junction_turn_frac": clip_junction / n_ep,
            "net_heading_deg_mean": float(net.mean()),
            "net_heading_deg_median": float(net.median()),
            "net_heading_deg_p90": float(torch.quantile(net, 0.90)),
            "net_heading_deg_max": float(net.max()),
            "clips_net_heading_gt45deg": int((net > 45).sum()),
            "clips_net_heading_gt45deg_frac": float((net > 45).float().mean()),
            "clips_net_heading_gt90deg": int((net > 90).sum()),
            "cum_abs_heading_deg_median": float(cum.median()),
        },
        "params_used": {
            "hz": HZ, "window": WINDOW, "max_horizon": MAX_HORIZON,
            "label_horizon": LABEL_HORIZON, "eff_batch": EFF_BATCH, "steps": STEPS,
            "nav_horizon_steps": rl.NAV_HORIZON_STEPS,
            "nav_min_steps": rl.NAV_MIN_STEPS,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print("WROTE", OUT, flush=True)
    print(json.dumps({
        "usable_clips": n_ep, "total_hours": round(total_hours, 3),
        "total_frames": total_frames, "total_windows": total_windows,
        "epochs": round(epochs, 3),
        "man_v1_frac": {k: round(x, 4) for k, x in man_v1_frac.items()},
        "man_v2_frac": {k: round(x, 4) for k, x in man_v2_frac.items()},
        "speed_regime": {k: round(x, 4) for k, x in v_bins.items()},
        "nav_v1_frac": {k: round(x, 4) for k, x in nav_v1_frac.items()},
        "nav_v1_cov": round(nav_v1_total / (nav_v1_total + nav_v1_invalid), 4),
        "route_v21_frac": {k: round(x, 4) for k, x in route_v21_frac.items()},
        "clips_any_turn_frac": round(clip_has["turn_any"] / n_ep, 4),
        "clips_brake_stop_frac": round(clip_has["brake_stop"] / n_ep, 4),
        "clips_junction_frac": round(clip_junction / n_ep, 4),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
