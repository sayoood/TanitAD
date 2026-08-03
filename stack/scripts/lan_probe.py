#!/usr/bin/env python3
"""LAN probes — two measurements, both cheap, both pre-registered.

``--agreement`` (0 GPU, seconds, runs on the dev box)
    Does the MAP-FREE route supplier (S1: the ego's own future path, arc-length
    resampled) agree with the MAP-BASED one (S2: lane centrelines + the lane
    graph)? S1 is the only supplier available on the parity corpus
    ``physicalai-train-e438721ae894``, which has no map. It is admissible as a
    stand-in for a real lane-graph route ONLY if the two agree — so this is the
    measurement that licenses the whole LAN label, and it runs on artifacts
    already banked in the repo (the NuRec ``map.xodr`` probe output).

``--navcf`` (minutes of GPU, forward passes only, NO training)
    The cheapest discriminating experiment for the C6 confound: sweep the
    existing 4-way ``nav_cmd`` over the val windows on a DEPLOYED REF-C
    checkpoint and measure how far the decoded trajectory moves. See
    ``tanitad.eval.route_cf.nav_cmd_sensitivity`` for the two pre-registered
    readings.

Usage
-----
    python scripts/lan_probe.py --agreement \
        --centerlines <lane_centerlines.json> --edges <lane_graph_edges.json> \
        --track <ego_track_map_frame.json> --out agreement.json

    python scripts/lan_probe.py --navcf --ckpt <refc.pt> --cache <val cache> \
        --out navcf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.data.lan import (LaneCorridor, LanConfig,  # noqa: E402
                              horizon_lead_m, lan_from_future_path,
                              lan_from_polyline, route_agreement, yaw_from_path)


def _load_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``ego_track_map_frame.json`` -> (xy [T, 2], yaw [T] rad).

    ⚠️ Read with ``[]``, not ``.get()`` — a missing stamp must raise here, not
    silently become a default that a downstream metric then averages.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["track"] if isinstance(d, dict) else d
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    if rows and "yaw_deg" in rows[0]:
        yaw = np.radians([float(r["yaw_deg"]) for r in rows])
    else:
        yaw = np.array([yaw_from_path(xy, i) for i in range(len(rows))])
    return xy, yaw


def agreement(centerlines: Path, edges: Path, track: Path,
              cfg: LanConfig, stride: int = 1,
              max_snap_m: float = 5.0,
              max_heading_dev_deg: float | None = 60.0,
              hysteresis_m: float = 1.0) -> dict:
    """S1 (ego future) vs S2 (lane graph) LAN corridors, window by window."""
    corr = LaneCorridor.from_json(centerlines, edges)
    xy, yaw = _load_track(track)
    poly, seq, stats = corr.route_polyline(
        xy, max_snap_m=max_snap_m, heading=yaw,
        max_heading_dev_deg=max_heading_dev_deg, hysteresis_m=hysteresis_m)
    rows, per = [], []
    for i in range(0, len(xy) - 1, stride):
        # Leak guard uses the ego's own 2 s path length, exactly as the trainer
        # would: the same guard must apply to both suppliers or the comparison
        # is between different arc-length sets.
        fut = xy[i:]
        speed = float(np.linalg.norm(xy[min(i + 1, len(xy) - 1)] - xy[i])) * 10.0
        lead = horizon_lead_m(v0=speed, t_pred_s=2.0, cfg=cfg)
        s1 = lan_from_future_path(fut, xy[i], float(yaw[i]), cfg, lead)
        s2 = lan_from_polyline(poly, xy[i], float(yaw[i]), cfg, lead)
        a = route_agreement(s1, s2)
        a["i"] = int(i)
        rows.append(a)
        if a["n_compared"]:
            per.append(a)

    def _agg(key: str) -> dict:
        v = np.array([r[key] for r in per], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not v.size:
            return {"n": 0}
        return {"n": int(v.size), "median": round(float(np.median(v)), 4),
                "mean": round(float(v.mean()), 4),
                "p90": round(float(np.percentile(v, 90)), 4),
                "max": round(float(v.max()), 4)}

    return {"probe": "lan_s1_vs_s2_agreement",
            "n_samples": len(rows),
            "n_compared_samples": len(per),
            "arclengths_m": list(cfg.arclengths_m),
            "min_lead_m": cfg.min_lead_m,
            "snap": {"max_snap_m": max_snap_m,
                     "max_heading_dev_deg": max_heading_dev_deg,
                     "hysteresis_m": hysteresis_m},
            "corridor": {"n_lanes": corr.n_lanes, "n_edges": len(corr.edges)},
            "route": {"lane_sequence_len": len(seq), **stats},
            "pos_l2_m": _agg("pos_l2_m"),
            "lat_delta_m": _agg("lat_delta_m"),
            "bearing_deg": _agg("bearing_deg"),
            "side_agree": _agg("side_agree"),
            "s1_valid_frac": _agg("a_valid_frac"),
            "s2_valid_frac": _agg("b_valid_frac"),
            "both_valid_frac": _agg("both_valid_frac"),
            "per_sample": rows}


def _assert_geometry(cfg, frames_hw, ckpt_path) -> dict:
    """⛔ REF-C is trained at ONE raster. Scoring it at another is a RETRACTION.

    MEASURED 2026-08-02 (`~/refc_thor_eval.json` on Thor): REF-C-XL fed the
    256x640 cylindrical val died with ``shape '[8, 512, 8, 8]' is invalid for
    input of size 491520`` — 491520 = 8*512*8*15, i.e. a 8x15 token grid from a
    256x640 frame where the decoder expects the SQUARE 8x8. REF-C-base did NOT
    crash on the same cache and its four-family numbers were published from it;
    a silent wrong-raster number is worse than a crash. This check runs BEFORE
    any forward pass and refuses rather than reports.
    """
    want = int(cfg.encoder.image_size)
    h, w = int(frames_hw[0]), int(frames_hw[1])
    ok = (h == want and w == want)
    info = {"ckpt": str(ckpt_path), "encoder_image_size": want,
            "cache_frame_hw": [h, w], "square": bool(h == w),
            "grid_shape": [want // 32, want // 32],
            "matches_training_raster": ok}
    if not ok:
        raise SystemExit(
            f"⛔ GEOMETRY REFUSED: checkpoint trained at {want}x{want} "
            f"(grid {want // 32}x{want // 32}) but the cache frames are {h}x{w}. "
            f"Scoring REF-C on a raster it was not trained at was retracted "
            f"2026-08-02. Point --val-dir at the SQUARE 256px cache.")
    return info


def _load_val_episodes(val_dir: Path, limit: int | None = None
                       ) -> tuple[list, dict]:
    """``ep_*.pt`` dicts -> objects with ``.feats/.poses/.episode_id/.maneuvers``.

    Same on-disk contract ``taniteval.loaders`` uses for the parity val set
    (`physicalai-val-0c5f7dac3b11`): keys ``frames_u8`` [T,9,S,S] uint8,
    ``poses`` [T,4] float32 (x, y, yaw, v), ``episode_id``, ``maneuvers`` [T].

    ⛔ An unreadable episode is RECORDED, not silently skipped and not fatal.
    Silently skipping would change the denominator of every metric downstream
    without saying so; dying would throw away 39 good episodes over one bad
    transfer. The returned ``coverage`` block travels into the result JSON so a
    reader can never see the ADE without seeing which episodes produced it.
    """
    import torch

    class _Ep:
        __slots__ = ("feats", "poses", "episode_id", "maneuvers", "path")

    files = sorted(Path(val_dir).glob("ep_*.pt"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"⛔ no ep_*.pt under {val_dir}")
    eps, unreadable = [], []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:                 # truncated / corrupt transfer
            unreadable.append({"file": f.name, "bytes": f.stat().st_size,
                               "error": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        e = _Ep()
        e.feats = d["frames_u8"]
        e.poses = d["poses"].float()
        e.episode_id = d["episode_id"]
        e.maneuvers = d.get("maneuvers")
        e.path = str(f)
        eps.append(e)
    if not eps:
        raise SystemExit(f"⛔ every ep_*.pt under {val_dir} failed to load")
    cov = {"val_dir": str(val_dir), "n_files_seen": len(files),
           "n_episodes_loaded": len(eps), "n_unreadable": len(unreadable),
           "unreadable": unreadable}
    if unreadable:
        cov["⚠️"] = (f"{len(unreadable)} of {len(files)} val episodes are "
                     f"unreadable, so this run is NOT window-for-window "
                     f"comparable with a full-set published number. The "
                     f"episode-cluster bootstrap is still valid — its unit is "
                     f"the episode and it resamples the episodes present.")
        print(f"[lan_probe] ⚠️  {cov['⚠️']}", flush=True)
    return eps, cov


def _epre(args) -> dict:                        # pragma: no cover - needs a cache
    """E-pre (PREREG_lan_refc.md §6.0) — the 0-GPU LAUNCH GATE.

    ⛔ **If LAN's ``any_valid_frac`` is not >= 0.50 on a real corpus, DO NOT
    LAUNCH.** A route input that is absent as often as the 4-way ``nav_cmd`` it
    replaces has not fixed the defect and does not deserve a GPU.

    The bar it must clear is ``nav_valid_frac`` = 0.21-0.25 (RETRACTION_LOG
    2026-07-21). ⭐ That historical number is re-MEASURED here **on the same
    windows** rather than quoted, because a coverage comparison across two
    different window sets is not a comparison. Both labels are computed from
    ``poses`` alone, so this costs no forward pass and no GPU.
    """
    import torch

    from tanitad.data.lan import lan_window_features

    import refb_labels as rl

    eps, coverage = _load_val_episodes(args.val_dir, args.max_episodes)
    cfg = LanConfig(
        arclengths_m=tuple(args.arclengths) if args.arclengths
        else LanConfig().arclengths_m, min_lead_m=args.min_lead_m)
    window = args.window
    k_max = 20
    k = cfg.k
    per_anchor = np.zeros(k)
    lan_any = nav_valid = route_valid = n = 0
    speeds = []
    for ep in eps:
        T = int(ep.poses.shape[0])
        for t0 in range(0, T - window - k_max, args.eval_stride):
            t = t0 + window - 1
            v = lan_window_features(ep.poses, t, cfg).reshape(k, -1)[:, 3]
            per_anchor += v
            lan_any += int(v.any())
            nav_valid += int(bool(rl.nav_command(ep.poses, int(t))[1]))
            route_valid += int(bool(rl.route_from_future_v21(
                ep.poses, int(t))["valid"]))
            speeds.append(float(ep.poses[t, 3]))
            n += 1
    n = max(n, 1)
    lan_frac = lan_any / n
    nav_frac = nav_valid / n
    gate = lan_frac >= 0.50
    return {
        "probe": "lan_e_pre_launch_gate",
        "prereg": "PREREG_lan_refc.md §6.0",
        "coverage": coverage,
        "n_windows": n, "window": window, "eval_stride": args.eval_stride,
        "arclengths_m": list(cfg.arclengths_m), "min_lead_m": cfg.min_lead_m,
        "lan_any_valid_frac": round(lan_frac, 4),
        "lan_per_anchor_valid_frac": [round(float(x) / n, 4) for x in per_anchor],
        "nav_cmd_valid_frac_SAME_WINDOWS": round(nav_frac, 4),
        "route_v21_valid_frac_SAME_WINDOWS": round(route_valid / n, 4),
        "ratio_lan_over_nav": (round(lan_frac / nav_frac, 3) if nav_frac else None),
        "ego_speed_mps": {"median": round(float(np.median(speeds)), 3),
                          "mean": round(float(np.mean(speeds)), 3),
                          "p90": round(float(np.percentile(speeds, 90)), 3)},
        "gate_threshold": 0.50,
        "GATE": "PASS — LAN may launch" if gate else
                "⛔ FAIL — DO NOT LAUNCH (any_valid_frac < 0.50)",
        "_note": ("the leak guard masks an anchor unless its arc-length exceeds "
                  "max(2 s GT path, v0*2 s) + min_lead_m, so coverage FALLS as "
                  "ego speed rises; the speed block is reported so a coverage "
                  "number is never read without the speed distribution that "
                  "produced it."),
    }


def _navcf(args) -> dict:                       # pragma: no cover - needs a ckpt
    """E0 (PREREG_lan_refc.md §7) — the cheapest discriminating experiment.

    Sweep the EXISTING 4-way ``nav_cmd`` over the val windows on a DEPLOYED
    REF-C checkpoint. Forward passes only; no training, no new labels.

    Two input paths:

    ``--val-dir``  the canonical parity val cache (40 ``ep_*.pt``). Windows are
        built EXACTLY as ``taniteval.refc_eval.collect`` builds them (window =
        ``cfg.window``, stride, ``K_MAX = max(WP_STEPS)``), so the ADE column is
        the same surface every published REF-C number sits on, and every window
        carries its ``eid`` for the paired episode-cluster bootstrap.
    ``--cache``    a single pre-stacked frame tensor (the stub path).
    """
    import torch

    from tanitad.eval.route_cf import CONTROL_TOL_M, nav_cmd_sensitivity
    from tanitad.refs.refc import NAV_COMMANDS, RefCModel, refc_config

    from tanitad.refs import refc as _refc
    presets = {"base": refc_config, "small": _refc.refc_small_config,
               "xl": _refc.refc_xl_config, "smoke": _refc.refc_smoke_config}
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = presets[args.preset]() if args.preset else refc_config()
    model = RefCModel(cfg)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    dev = args.device
    model.to(dev).eval()
    steps = (args.steps if args.steps is not None
             else int(cfg.decoder.diffusion_steps))

    meta = {"ckpt": str(args.ckpt),
            "sd_missing": len(missing), "sd_unexpected": len(unexpected),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
            "diffusion_steps": steps,
            "nav_commands": list(NAV_COMMANDS)}

    if args.val_dir:
        from driving_diagnostic import (WP_STEPS, baseline_waypoints,
                                        gt_ego_waypoints)
        eps, coverage = _load_val_episodes(args.val_dir, args.max_episodes)
        meta["geometry"] = _assert_geometry(cfg, eps[0].feats.shape[-2:],
                                            args.ckpt)
        meta["val_dir"] = str(args.val_dir)
        meta["coverage"] = coverage
        window = int(cfg.window)
        k_max = max(WP_STEPS)
        # --- build the window index ONCE; every arm scores the same windows ---
        plan, EID, GT, CV, SPD, MAN_GT = [], [], [], [], [], []
        for e_i, ep in enumerate(eps):
            T = int(ep.feats.shape[0])
            starts = list(range(0, T - window - k_max, args.eval_stride))
            for i in range(0, len(starts), args.batch):
                ch = starts[i:i + args.batch]
                last = torch.tensor([t + window - 1 for t in ch])
                plan.append((e_i, ch, last))
                EID.extend([ep.episode_id] * len(ch))
                GT.append(gt_ego_waypoints(ep.poses, last))
                CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
                SPD.append(ep.poses[last, 3])
                MAN_GT.append(ep.maneuvers[last] if ep.maneuvers is not None
                              else torch.full((len(ch),), -1, dtype=torch.long))
        gt = torch.cat(GT).float()
        cv = torch.cat(CV).float()
        n_win = int(gt.shape[0])
        meta.update(n_windows=n_win, n_episodes=len(set(EID)),
                    window=window, stride=args.eval_stride,
                    wp_steps=list(WP_STEPS))
        aux = {}

        def predict(nav_idx: int):
            wps, mans, routes = [], [], []
            with torch.no_grad():
                for e_i, ch, last in plan:
                    ep = eps[e_i]
                    fw = torch.stack([torch.as_tensor(ep.feats[t:t + window])
                                      for t in ch]).to(dev).float().div_(255.0)
                    v0 = None if args.no_v0 else ep.poses[last, 3].to(dev)
                    nav = torch.full((len(ch),), int(nav_idx),
                                     dtype=torch.long, device=dev)
                    out = model(fw, nav_cmd=nav, v0=v0, steps=steps)
                    wps.append(torch.stack([out["waypoints"][k]
                                            for k in WP_STEPS], dim=1)
                               .float().cpu().numpy())
                    mans.append(out["maneuver_logits"].float().argmax(-1).cpu())
                    routes.append(out["route_logits"].float().argmax(-1).cpu())
            aux[int(nav_idx)] = {"maneuver_pred": torch.cat(mans),
                                 "route_pred": torch.cat(routes)}
            return np.concatenate(wps, axis=0)
    else:
        frames = torch.load(args.cache, map_location="cpu", weights_only=False)
        if isinstance(frames, dict):
            frames = frames["frames"]
        meta["geometry"] = _assert_geometry(cfg, frames.shape[-2:], args.ckpt)
        v0t = None if args.no_v0 else torch.zeros(frames.shape[0])
        n_win = int(frames.shape[0])
        gt = cv = None
        EID, MAN_GT, aux = [0] * n_win, [], {}

        def predict(nav_idx: int):
            outs = []
            with torch.no_grad():
                for s in range(0, frames.shape[0], args.batch):
                    fr = frames[s:s + args.batch].to(dev)
                    nav = torch.full((fr.shape[0],), int(nav_idx),
                                     dtype=torch.long, device=dev)
                    ev = None if v0t is None else v0t[s:s + args.batch].to(dev)
                    outs.append(model(fr, nav_cmd=nav, v0=ev,
                                      steps=steps)["traj"].float().cpu().numpy())
            return np.concatenate(outs, axis=0)

    # ---- cache every arm so the four-family block reuses the SAME decode ----
    trajs: dict[int, np.ndarray] = {}

    def cached_predict(nav_idx: int):
        y = predict(int(nav_idx))
        trajs.setdefault(int(nav_idx), y)        # first call wins == the control
        return y

    res = nav_cmd_sensitivity(cached_predict, n_win, len(NAV_COMMANDS))
    pw = res.pop("per_window_max_m")
    res["per_window_max_m_stats"] = {
        "median": round(float(np.median(pw)), 6),
        "p90": round(float(np.percentile(pw, 90)), 6),
        "max": round(float(pw.max()), 6)}
    res.update(meta)
    res["control_tol_m"] = CONTROL_TOL_M
    res["reachable"] = _reachable_only(res["pairwise_mean_m"], NAV_COMMANDS)

    if args.val_dir:
        res["arms"] = _navcf_families(trajs, aux, gt, cv, EID,
                                      torch.cat(MAN_GT) if MAN_GT else None,
                                      eps, plan, list(NAV_COMMANDS), args)
        res["per_window"] = {"eid": [str(e) for e in EID],
                             "max_disp_m": [round(float(x), 8) for x in pw]}
    return res


def _navcf_families(trajs, aux, gt, cv, eid, man_gt, eps, plan, nav_names,
                    args) -> dict:               # pragma: no cover - needs a ckpt
    """⛔ BINDING (Sayed 2026-08-02): ADE + the FOUR FAMILIES, per arm, per family.

    Every interval is the paired episode-cluster bootstrap over the val episodes
    (`taniteval.ci`), unit = episode. ``overlapping_holdout_se`` is never called.
    """
    import torch

    from taniteval.ci import (episode_cluster_bootstrap,
                              paired_episode_cluster_bootstrap)
    from taniteval.four_families import all_families

    from driving_diagnostic import WP_STEPS

    # STRATEGIC ground truth — the v2.1 route label, the same one REF-C trains on
    import refb_labels as rl
    route_gt, route_valid = [], []
    for e_i, ch, last in plan:
        for t in last.tolist():
            r = rl.route_from_future_v21(eps[e_i].poses, int(t))
            route_gt.append(int(r["route"]))
            route_valid.append(bool(r["valid"]))
    route_gt_t = torch.tensor(route_gt, dtype=torch.long)
    valid_t = torch.tensor(route_valid, dtype=torch.bool)

    def ade(y):                                  # [N] per-window mean-over-h L2
        return np.linalg.norm(y - gt.numpy(), axis=-1).mean(axis=-1)

    cv_ade = np.linalg.norm(cv.numpy() - gt.numpy(), axis=-1).mean(axis=-1)
    base = ade(trajs[0])
    out = {"_estimator": ("paired episode-cluster bootstrap, taniteval/ci.py, "
                          f"n_boot={args.n_boot}, unit = episode. "
                          "overlapping_holdout_se is NEVER used (it biases the "
                          "point estimate, CLAUDE.md)."),
           "cv_baseline": {"ade_0_2s": episode_cluster_bootstrap(
               cv_ade, eid, n_boot=args.n_boot)}}
    gt_np = gt.numpy()
    eid_v = [e for e, v in zip(eid, valid_t.tolist()) if v]
    mv = valid_t.numpy()

    def score_arm(y, man_pred, route_pred, paired_against=None):
        """ADE + the FOUR FAMILIES + per-family CIs for ONE arm.

        Used for BOTH the constant-command arms and the assembled route arms, so
        the two are scored by the same code on the same windows — the only way a
        cross-arm delta means anything.
        """
        a = ade(y)
        fam = all_families({"pred": torch.from_numpy(y).float(), "gt": gt,
                            "wp_steps": list(WP_STEPS), "dt_s": 0.1,
                            "maneuver_pred": man_pred, "maneuver_gt": man_gt,
                            "route_pred": route_pred, "route_gt": route_gt_t})
        # ⛔ per-family components, never pooled into one score
        lon_err = np.abs(y[:, :, 0] - gt_np[:, :, 0]).mean(axis=-1)
        lat_err = np.abs(y[:, :, 1] - gt_np[:, :, 1]).mean(axis=-1)
        head_err, head_keep = _abs_heading_err_deg(y, gt_np)
        curv_err, curv_keep = _abs_curv_err(y, gt_np)
        spd_err = _abs_speed_err(y, gt_np)
        eid_h = [e for e, k in zip(eid, head_keep.tolist()) if k]
        eid_c = [e for e, k in zip(eid, curv_keep.tolist()) if k]
        tac_hit = (man_pred == man_gt).numpy().astype(np.float64)
        str_hit = (route_pred == route_gt_t).numpy().astype(np.float64)
        row = {
            "ade_0_2s": episode_cluster_bootstrap(a, eid, n_boot=args.n_boot),
            "four_families": fam,
            "family_ci": {
                "LONGITUDINAL_along_mae_m": episode_cluster_bootstrap(
                    lon_err, eid, n_boot=args.n_boot),
                "LONGITUDINAL_speed_mae_mps": episode_cluster_bootstrap(
                    spd_err, eid, n_boot=args.n_boot),
                "LATERAL_cross_mae_m": episode_cluster_bootstrap(
                    lat_err, eid, n_boot=args.n_boot),
                "LATERAL_heading_mae_deg": episode_cluster_bootstrap(
                    head_err[head_keep], eid_h, n_boot=args.n_boot),
                "LATERAL_curvature_mae_1pm": episode_cluster_bootstrap(
                    curv_err[curv_keep], eid_c, n_boot=args.n_boot),
                "TACTICAL_maneuver_accuracy": episode_cluster_bootstrap(
                    tac_hit, eid, n_boot=args.n_boot),
                "STRATEGIC_route_accuracy": episode_cluster_bootstrap(
                    str_hit, eid, n_boot=args.n_boot),
                "STRATEGIC_route_accuracy_valid_only": episode_cluster_bootstrap(
                    str_hit[mv], eid_v, n_boot=args.n_boot),
            },
            "head_share": {
                "maneuver": {n: int(c) for n, c in zip(
                    _maneuver_names(),
                    torch.bincount(man_pred, minlength=5).tolist())},
                "route": {n: int(c) for n, c in zip(
                    ("route_left", "route_straight", "route_right"),
                    torch.bincount(route_pred, minlength=3).tolist())},
            },
        }
        if paired_against is not None:
            pa = paired_against
            # ⛔ a paired test must run on windows BOTH arms scored. The MIN_DS
            # gate depends on the PREDICTED path, so two arms can gate different
            # windows; the intersection is the only aligned set.
            hk = head_keep & pa["head_keep"]
            ck = curv_keep & pa["curv_keep"]
            eid_hp = [e for e, k in zip(eid, hk.tolist()) if k]
            eid_cp = [e for e, k in zip(eid, ck.tolist()) if k]
            row["paired_vs_follow_constant"] = {
                "ade_0_2s": paired_episode_cluster_bootstrap(
                    a, pa["ade"], eid, n_boot=args.n_boot),
                "LONGITUDINAL_along_mae_m": paired_episode_cluster_bootstrap(
                    lon_err, pa["lon"], eid, n_boot=args.n_boot),
                "LONGITUDINAL_speed_mae_mps": paired_episode_cluster_bootstrap(
                    spd_err, pa["spd"], eid, n_boot=args.n_boot),
                "LATERAL_cross_mae_m": paired_episode_cluster_bootstrap(
                    lat_err, pa["lat"], eid, n_boot=args.n_boot),
                "LATERAL_heading_mae_deg": paired_episode_cluster_bootstrap(
                    head_err[hk], pa["head"][hk], eid_hp, n_boot=args.n_boot),
                "LATERAL_curvature_mae_1pm": paired_episode_cluster_bootstrap(
                    curv_err[ck], pa["curv"][ck], eid_cp, n_boot=args.n_boot),
                "_n_paired_windows": {"heading": int(hk.sum()),
                                      "curvature": int(ck.sum()),
                                      "ade_and_positional": int(a.size)},
                "_reading": ("delta = ARM - follow_constant on an ERROR metric, "
                             "so delta > 0 means the arm is WORSE. TACTICAL and "
                             "STRATEGIC are omitted from the paired block on "
                             "purpose: both heads read `pooled` only, so they "
                             "are BIT-IDENTICAL across every nav arm and a "
                             "paired delta on them is structurally 0."),
            }
        return row, {"ade": a, "lon": lon_err, "lat": lat_err,
                     "head": head_err, "curv": curv_err, "spd": spd_err,
                     "head_keep": head_keep, "curv_keep": curv_keep}

    base_comp = None
    for i, name in enumerate(nav_names):
        if i not in trajs:
            continue
        row, comp = score_arm(trajs[i], aux[i]["maneuver_pred"],
                              aux[i]["route_pred"],
                              paired_against=base_comp)
        if i == 0:
            base_comp = comp
            row["paired_vs_follow_constant"] = None    # it IS follow_constant
        out[name] = row
    # ------------------------------------------------------------------
    # E0-B — the follow-up PREREG §7 commits to when E0 returns RESPONSIVE:
    #   "the cheap fix is then to SUPPLY THE LABEL AT EVAL and re-score every
    #    REF-C row — a bigger, cheaper correction than LAN".
    # Costs ZERO extra forward passes: `nav_cmd` enters only through the
    # `measurement` MLP, so the four constant-command decodes already contain
    # every trajectory a per-window route assignment could select. We assemble
    # rather than re-decode, which also makes the three arms bit-comparable.
    # ------------------------------------------------------------------
    from taniteval.refc_eval import ROUTE_TO_NAV
    r_pred = aux[0]["route_pred"].numpy()        # image-only -> same for all arms
    r_gt = route_gt_t.numpy()

    # ⛔ route class 3 is UNKNOWN, not a fourth route. `route_from_future_v21`
    # emits it whenever there is not enough future to derive a route (15.37 % of
    # windows here). The deployable behaviour there is `follow` — the same thing
    # the model does with no route — so UNKNOWN maps to nav 0 and the count is
    # reported. Silently dropping those windows would shrink the denominator of
    # the oracle arm and flatter it.
    _R2N = dict(ROUTE_TO_NAV)
    _R2N[3] = 0

    def _assemble(sel_route):
        idx = np.array([_R2N[int(r)] for r in sel_route])
        return np.stack([trajs[int(idx[n])][n] for n in range(idx.shape[0])]), idx

    assembled = {}
    for nm, sel in (("produced_route", r_pred), ("oracle_route", r_gt)):
        y, idx = _assemble(sel)
        row, comp = score_arm(y, aux[0]["maneuver_pred"], aux[0]["route_pred"],
                              paired_against=base_comp)
        row["nav_fed_hist"] = {n: int(c) for n, c in zip(
            nav_names, np.bincount(idx, minlength=len(nav_names)).tolist())}
        row["frac_windows_not_follow"] = round(float((idx != 0).mean()), 4)
        # the same ADE contrast restricted to windows whose route label is VALID
        # — on ~15 % of windows there is no future to derive a route from, and an
        # oracle that is really "follow" there dilutes its own effect.
        row["paired_ade_vs_follow_constant_route_valid_only"] = \
            paired_episode_cluster_bootstrap(comp["ade"][mv], base_comp["ade"][mv],
                                             eid_v, n_boot=args.n_boot)
        row["n_route_valid"] = int(mv.sum())
        assembled[nm] = row
    assembled["_reading"] = (
        "delta > 0 means the ASSEMBLED arm is WORSE than constant-follow (ADE is "
        "an error). `oracle_route` is future-derived and is an UPPER BOUND only, "
        "never a leaderboard number; `produced_route` is the deployable fix — the "
        "model's own image-only route_head, no future, no label.")
    out["E0B_supply_the_label_at_eval"] = assembled

    # ------------------------------------------------------------------
    # NEGATIVE CONTROL #2 — do the per-window components this file reduces for
    # the bootstrap actually reproduce the family means `four_families` reports?
    # If they do not, every CI here describes a different quantity from the
    # family table beside it. This control CAUGHT A REAL BUG on 2026-08-03 (see
    # _row_mean): heading read 4.0181 vs 1.1486 deg and curvature 42.1365 vs
    # 0.00788 1/m. It runs on every future invocation and reports PASS/FAIL.
    # ------------------------------------------------------------------
    # ⛔ Two classes of metric, and conflating them would either hide a real bug
    # or manufacture a fake one:
    #   EXACT   — every window contributes the same number of steps, so the
    #             family's pooled step-mean and the per-window cluster mean are
    #             the SAME number. These must match to float precision; any gap
    #             is a bug.
    #   GATED   — heading/curvature are averaged over VALID steps only
    #             (MIN_DS_MPS 0.5 x dt), and the valid COUNT varies per window
    #             (curvature has only 2 pair-steps on a 4-waypoint view). The
    #             family pools over (window, step); a cluster bootstrap must
    #             reduce to ONE value per window. Those are two different, both
    #             correct, estimators and they legitimately differ. Reported
    #             with the measured gap rather than forced to agree.
    _EXACT = ("LATERAL_cross_mae_m", "LONGITUDINAL_along_mae_m",
              "LONGITUDINAL_speed_mae_mps", "TACTICAL_maneuver_accuracy",
              "STRATEGIC_route_accuracy")
    checks, worst_exact, worst_gated = [], 0.0, 0.0
    for name in nav_names:
        a = out.get(name)
        if not a:
            continue
        ff, ci = a["four_families"], a["family_ci"]
        for key, fv in (("LATERAL_cross_mae_m", ff["lateral"]["cross_mae_m"]),
                        ("LATERAL_heading_mae_deg",
                         ff["lateral"]["heading_mae_deg"]),
                        ("LATERAL_curvature_mae_1pm",
                         ff["lateral"]["curvature_mae_1pm"]),
                        ("LONGITUDINAL_along_mae_m",
                         ff["longitudinal"]["along_mae_m"]),
                        ("LONGITUDINAL_speed_mae_mps",
                         ff["longitudinal"]["speed_mae_mps"]),
                        ("TACTICAL_maneuver_accuracy", ff["tactical"]["accuracy"]),
                        ("STRATEGIC_route_accuracy", ff["strategic"]["accuracy"])):
            cv_ = ci[key]["mean"]
            rel = abs(cv_ - fv) / max(abs(fv), 1e-9)
            exact = key in _EXACT
            tol = 1e-3 if exact else 0.25
            if exact:
                worst_exact = max(worst_exact, rel)
            else:
                worst_gated = max(worst_gated, rel)
            checks.append({"arm": name, "metric": key, "class":
                           "EXACT" if exact else "GATED-ESTIMATOR-DIFFERENCE",
                           "family_mean": fv, "component_mean": cv_,
                           "rel_err": round(rel, 6), "tolerance": tol,
                           "ok": bool(rel <= tol)})
    failed = [c for c in checks if not c["ok"]]
    out["_negative_control_component_vs_family"] = {
        "worst_rel_err_EXACT": round(worst_exact, 9),
        "worst_rel_err_GATED": round(worst_gated, 6),
        "n_checks": len(checks), "n_failed": len(failed),
        "VERDICT": ("PASS — the five EXACT metrics reproduce their family means "
                    "to <1e-3 relative, and the two GATED metrics differ only by "
                    "the pooled-step vs per-window-cluster estimator gap, which "
                    "is bounded and explained below"
                    if not failed else
                    "⛔ FAIL — a bootstrapped component does NOT reduce to its "
                    "family mean beyond the allowed estimator gap. The CIs "
                    "describe a different quantity from the family table. Do "
                    "not quote them."),
        "failed": failed,
        "checks": checks,
        "_why_gated_differs": (
            "`four_families` reports x[mask].mean() — pooled over (window, step). "
            "A cluster bootstrap needs one value per EPISODE-clusterable window, "
            "so this file reduces per window first. The two agree only when every "
            "window contributes equally many valid steps. MEASURED gap on this "
            "run: heading 0.83 %, curvature 7.9-19.7 % (curvature has just 2 "
            "pair-steps per window, so a window with 1 valid pair reweights). "
            "Both are correct; the PAIRED deltas use the per-window components "
            "for BOTH arms, so the contrast is unaffected either way."),
        "_history": (
            "This control caught a REAL bug on 2026-08-03 before any number was "
            "quoted: windows with no valid step were back-filled with their "
            "UNMASKED row mean, which imported the crawling/stopped windows the "
            "gate exists to remove — heading read 4.0181 vs 1.1486 deg and "
            "curvature 42.1365 vs 0.00788 1/m (5 348x). See _row_mean."),
    }

    out["label_prevalence"] = {
        "maneuver_gt": {n: int(c) for n, c in zip(
            _maneuver_names(),
            torch.bincount(man_gt.clamp(min=0), minlength=5).tolist())},
        "route_gt": {n: int(c) for n, c in zip(
            ("route_left", "route_straight", "route_right", "UNKNOWN"),
            torch.bincount(route_gt_t, minlength=4).tolist())},
        "route_valid_frac": round(float(valid_t.float().mean()), 4),
        "n_windows": int(route_gt_t.numel()),
    }
    return out


def _geom(y: np.ndarray, dt: float = 0.5):
    """[N,H,2] ego-frame waypoints -> per-step speed / heading / curvature.

    Mirrors ``taniteval.four_families._seq_geometry`` — same origin prepend, same
    ``MIN_DS_MPS = 0.5`` validity gate scaled by ``dt`` — so the per-window
    components fed to the bootstrap reduce to the SAME family means the JSON
    reports. ``dt = 0.5 s`` because the scored surface is the sparse 4-waypoint
    view at WP_STEPS (5, 10, 15, 20); passing 0.1 here is the defect corrected
    on 2026-08-03 (speeds x5, accels x25).
    """
    p = np.concatenate([np.zeros_like(y[:, :1]), y], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    heading = np.arctan2(d[..., 1], d[..., 0])
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    ds_mid = 0.5 * (ds[:, 1:] + ds[:, :-1])
    valid = ds > 0.5 * dt
    return {"speed": ds / dt, "heading": heading, "valid": valid,
            "curvature": dh / (ds_mid + 1e-8),
            "pair_valid": valid[:, 1:] & valid[:, :-1]}


def _row_mean(x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (per-window mean over the VALID steps, keep-mask of windows with any).

    ⛔ **THE BUG THIS SIGNATURE EXISTS TO PREVENT — caught 2026-08-03 by the
    self-consistency control, before any number was quoted.** The first version
    fell back to the UNMASKED row mean for windows where every step was gated
    out. Those are exactly the crawling/stopped windows the ``MIN_DS`` gate
    exists to remove — heading is undefined and curvature is a division by ~0 —
    so the fallback imported the garbage the mask was there to exclude:
    ``heading_mae`` read **4.0181 deg** against ``four_families``' **1.1486**,
    and ``curvature_mae`` **42.1365 1/m** against **0.00788** (a 5 348x
    inflation). Windows with no valid step are now DROPPED, and the caller drops
    their ``eid`` with them so the bootstrap clusters stay aligned.
    """
    n = m.sum(axis=1)
    keep = n > 0
    safe = np.where(keep, n, 1)
    return ((x * m).sum(axis=1) / safe).astype(np.float64), keep


def _abs_heading_err_deg(y: np.ndarray, gt: np.ndarray):
    P, G = _geom(y), _geom(gt)
    dh = P["heading"] - G["heading"]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    return _row_mean(np.degrees(np.abs(dh)), P["valid"] & G["valid"])


def _abs_curv_err(y: np.ndarray, gt: np.ndarray):
    P, G = _geom(y), _geom(gt)
    return _row_mean(np.abs(P["curvature"] - G["curvature"]),
                     P["pair_valid"] & G["pair_valid"])


def _abs_speed_err(y: np.ndarray, gt: np.ndarray) -> np.ndarray:
    P, G = _geom(y), _geom(gt)
    return np.abs(P["speed"] - G["speed"]).mean(axis=1).astype(np.float64)


def _reachable_only(pairwise: dict, nav_names) -> dict:
    """⛔ One of the four nav commands can NEVER be produced by the label pipeline.

    ``refc_eval.ROUTE_TO_NAV = {0: 1, 1: 0, 2: 2}`` maps the 3-class route label
    (left / straight / right) onto nav indices **{1, 0, 2}** — so nav index 3,
    whose name happens to be ``"straight"``, is an embedding row the model has
    never been trained on and that no eval mode can emit. Feeding it is an
    out-of-distribution probe, not a route counterfactual, and MEASURED on the
    smoke run it dominated the sweep (0v3 = 2.157 m vs 0v1 = 0.259 m).

    Reporting only the raw 4-way max would therefore overstate route sensitivity
    with an artefact. This block reports the sweep restricted to the REACHABLE
    commands, and the verdict is taken from THAT.
    """
    from taniteval.refc_eval import ROUTE_TO_NAV
    reach = sorted(set(int(v) for v in ROUTE_TO_NAV.values()))
    keep = {k: v for k, v in pairwise.items()
            if all(int(x) in reach for x in k.split("v"))}
    mx = max(keep.values()) if keep else float("nan")
    return {
        "reachable_nav_indices": reach,
        "reachable_nav_names": [nav_names[i] for i in reach],
        "unreachable_nav_indices": [i for i in range(len(nav_names))
                                    if i not in reach],
        "unreachable_note": (
            "nav index 3 is unreachable: refc_eval.ROUTE_TO_NAV maps the 3-class "
            "route label onto nav {0,1,2} only. Its embedding row is untrained; "
            "differences involving it are OOD, not route responses."),
        "pairwise_mean_m": keep,
        "max_pairwise_mean_m": round(float(mx), 6),
        "verdict": ("INERT (reachable commands only)" if mx <= 1e-6 else
                    "RESPONSIVE (reachable commands only)"),
    }


def _maneuver_names():                          # pragma: no cover - trivial
    try:
        from tanitad.refs.refb import MANEUVER_CLASSES
        return list(MANEUVER_CLASSES)
    except Exception:
        return ["c%d" % i for i in range(5)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agreement", action="store_true")
    p.add_argument("--navcf", action="store_true")
    p.add_argument("--epre", action="store_true",
                   help="§6.0 launch gate: LAN route-input coverage, 0 GPU")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--centerlines", type=Path)
    p.add_argument("--edges", type=Path)
    p.add_argument("--track", type=Path)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-snap-m", type=float, default=5.0)
    p.add_argument("--max-heading-dev-deg", type=float, default=60.0)
    p.add_argument("--hysteresis-m", type=float, default=1.0)
    p.add_argument("--arclengths", type=float, nargs="+", default=None)
    p.add_argument("--min-lead-m", type=float, default=5.0)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--val-dir", type=Path, default=None,
                   help="parity val cache of ep_*.pt (canonical E0 surface)")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=8,
                   help="window stride for --val-dir; 8 is the canonical REF-C "
                        "eval surface (881 windows / 40 val episodes). ⛔ do NOT "
                        "reuse --stride here: that one belongs to --agreement "
                        "and defaults to 1, which silently produces a 8x denser "
                        "and NON-comparable window set")
    p.add_argument("--preset", default=None, help="refc_config preset, e.g. base")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=None,
                   help="denoise steps; default = cfg.decoder.diffusion_steps")
    p.add_argument("--no-v0", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    cfg = LanConfig(
        arclengths_m=tuple(a.arclengths) if a.arclengths else LanConfig().arclengths_m,
        min_lead_m=a.min_lead_m)

    if a.agreement:
        if not (a.centerlines and a.edges and a.track):
            p.error("--agreement needs --centerlines --edges --track")
        res = agreement(a.centerlines, a.edges, a.track, cfg,
                        stride=a.stride, max_snap_m=a.max_snap_m,
                        max_heading_dev_deg=a.max_heading_dev_deg,
                        hysteresis_m=a.hysteresis_m)
    elif a.navcf:
        if not (a.ckpt and (a.cache or a.val_dir)):
            p.error("--navcf needs --ckpt and one of --val-dir / --cache")
        res = _navcf(a)
    elif a.epre:
        if not a.val_dir:
            p.error("--epre needs --val-dir")
        res = _epre(a)
    else:
        p.error("pick a probe: --agreement, --navcf or --epre")

    slim = {k: v for k, v in res.items()
            if k not in ("per_sample", "per_window")}
    print(json.dumps(slim, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[lan_probe] wrote {a.out}")
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
