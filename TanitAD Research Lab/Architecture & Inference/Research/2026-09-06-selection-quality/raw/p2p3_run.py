"""P2 + P3 driver on the LOCAL refcv4b step-40284 stride-1 dump.

P2  temporal SELECTION STABILITY, with its GT ceiling and its random floor.
P3  valid SELECTION REGRET against the best candidate the fan actually
    contained, with the ceiling construction proven not to be the invalid
    `a_star` one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import _env  # noqa: F401
import selstab as S

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_dump(dump):
    fs = sorted(f for f in os.listdir(dump)
                if f.startswith("ep") and f.endswith(".npz"))
    eps = []
    for f in fs:
        z = np.load(os.path.join(dump, f), allow_pickle=True)
        d = np.load(os.path.join(dump, "decisions", f), allow_pickle=True)
        e = {k: z[k] for k in z.files}
        e.update({k: d[k] for k in d.files if k != "ws"})
        e["clip"] = None
        eps.append(e)
    return fs, eps


def clip_map(arm_json):
    """file_index -> clip_id, plus the PROVIDER->RAW frame offset.

    Never guessed from directory order: `refcv3_arm.py` filters the corpus, so
    `clip_index` in the npz indexes the FILTERED list.  The run's own manifest
    is the only honest source, and `provider_to_raw_frame_offset` is the reason
    `poses[ws]` is the WRONG row (v2_dataset stores poses[n_stack-1:]).
    """
    with open(arm_json, encoding="utf-8") as fh:
        m = json.load(fh)["refcv3"]["manifest"]
    ids = {int(e["file_index"]): e["clip_id"] for e in m["episodes"]}
    off = int(m["corpus"]["frames"]["provider_to_raw_frame_offset"])
    return ids, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--eps-dir", default=S.EPS)
    a = ap.parse_args()

    from taniteval import ci as CI

    ctrl, ckey = S.anchor_controls()
    files, eps = load_dump(a.dump)
    clips, frame_off = clip_map(a.arm_json)
    print(f"[dump] {a.dump}")
    print(f"[dump] provider->raw frame offset = {frame_off} "
          f"(read from the run's own manifest, never assumed)")
    print(f"[dump] {len(eps)} episodes, "
          f"{sum(len(e['ws']) for e in eps)} windows, "
          f"anchor_controls from {ckey} n={ctrl.shape[0]}")

    # ---------------- C2: v0 round-trip against the episode poses -----------
    # the dump's episode files are written in sorted clip order by refcv3_arm
    v0_err = 0.0
    poses_by_ep = []
    for i, e in enumerate(eps):
        p = torch.load(os.path.join(a.eps_dir, clips[i] + ".v2ep.pt"),
                       map_location="cpu", weights_only=False)["poses"].numpy()
        poses_by_ep.append(p.astype(np.float64))
        v0_err = max(v0_err, float(
            np.abs(p[e["ws"] + frame_off, 3] - e["v0"]).max()))
    print(f"[C2] v0 round-trip max|dump v0 - poses[ws+{frame_off}, 3]| = "
          f"{v0_err:.3e}  (must be < 1e-5)")
    assert v0_err < 1e-4, "window->pose mapping is WRONG; every transform void"

    # ---------------- assemble ----------------------------------------------
    rows = []
    for i, e in enumerate(eps):
        n = len(e["ws"])
        rows.append({"ep": i, "n": n, "ws": e["ws"].astype(int),
                     "v0": e["v0"].astype(np.float64),
                     "g": e["g"].astype(np.float64),
                     "os": e["os"].astype(np.float64),
                     "sel": e["sel_idx"].astype(int),
                     "a_star_fixedbank": e["a_star"].astype(int),
                     "lat": e["lat_pred_nav_true"].astype(int),
                     "lon": e["lon_pred_nav_true"].astype(int),
                     "poses": poses_by_ep[i]})

    # ---------------- the emitted fan, per window ---------------------------
    all_v0 = np.concatenate([r["v0"] for r in rows])
    eid = np.concatenate([np.full(r["n"], r["ep"]) for r in rows])
    F = np.empty((len(all_v0), ctrl.shape[0], 4, 2), dtype=np.float64)
    B = 4000
    for s in range(0, len(all_v0), B):
        F[s:s + B] = S.fan(ctrl, all_v0[s:s + B],
                           slots=S.SLOTS_2S).numpy().astype(np.float64)[..., :2]
    G = np.concatenate([r["g"] for r in rows])
    OS = np.concatenate([r["os"] for r in rows])
    SEL = np.concatenate([r["sel"] for r in rows])
    ASTAR = np.concatenate([r["a_star_fixedbank"] for r in rows])
    LAT = np.concatenate([r["lat"] for r in rows])
    LON = np.concatenate([r["lon"] for r in rows])
    N = len(all_v0)
    print(f"[fan] emitted fan rebuilt model-free: {F.shape}")

    # C1 straight control
    i0 = int(np.argmin(np.abs(ctrl[:, 1].numpy())))
    print(f"[C1] straight candidate #{i0}: max|y| over all windows/slots = "
          f"{np.abs(F[:, i0, :, 1]).max():.3e} m (must be 0)")

    # ---------------- P3: cost of every candidate, four families ------------
    costs = S.family_costs(F, G[:, None, :, :], all_v0)
    fams = ("ade_m", "along_mae_m", "cross_mae_m", "speed_mae_mps",
            "curv_mae_1pm", "heading_mae_deg")
    sel_cost, best_cost, best_idx, regret, scored = {}, {}, {}, {}, {}
    for f in fams:
        c = costs[f]                                   # [N, 117]
        fin = np.isfinite(c)
        cc = np.where(fin, c, np.inf)
        bi = cc.argmin(1)
        best_idx[f] = bi
        best_cost[f] = cc[np.arange(N), bi]
        sel_cost[f] = cc[np.arange(N), SEL]
        # a window is SCOREABLE for this family only where BOTH the selected
        # and the best candidate produced a finite cost.  `curv_mae_1pm` is
        # undefined where no segment clears the 0.25 m minimum arc, and an
        # inf-minus-inf there would read as a silent nan, not as a work item.
        ok = np.isfinite(best_cost[f]) & np.isfinite(sel_cost[f])
        scored[f] = ok
        regret[f] = np.where(ok, sel_cost[f] - best_cost[f], np.nan)

    # the ceiling-validity proof: our argmin is NOT the dumped a_star
    agree = float((best_idx["ade_m"] == ASTAR).mean())
    print(f"\n[P3 ceiling proof] argmin over the PER-WINDOW v0-conditioned fan "
          f"vs the dumped `a_star` (argmin over model.core.decoder.anchors, the "
          f"FIXED ref-speed bank): identical on {agree*100:.2f} % of "
          f"{N} windows")
    dstar = S.family_costs(F[np.arange(N), ASTAR], G, all_v0)["ade_m"]
    print(f"[P3 ceiling proof] ADE of the a_star candidate  "
          f"{np.nanmean(dstar):.4f} m")
    print(f"[P3 ceiling proof] ADE of OUR best-in-fan       "
          f"{np.nanmean(best_cost['ade_m']):.4f} m  "
          f"(a valid ceiling must be <= every arm it bounds)")
    print(f"[P3 ceiling proof] ADE of the SELECTED candidate "
          f"{np.nanmean(sel_cost['ade_m']):.4f} m")
    os_ade = np.linalg.norm(OS - G, axis=-1).mean(-1)
    print(f"[P3 ceiling proof] ADE of the model's REFINED output `os` "
          f"{os_ade.mean():.4f} m  (refinement sits on top of the selection)")

    rng = np.random.default_rng(0)
    rnd = rng.integers(0, ctrl.shape[0], size=N)
    # C5 INDEX-MAPPING CONTROL.  `sel_idx` must index the SAME ordering as
    # `anchor_controls`.  If it did not, F[sel] would score like a random draw.
    c_ade = np.where(np.isfinite(costs["ade_m"]), costs["ade_m"], np.inf)
    print(f"\n[C5] index-mapping control: ADE(F[sel_idx]) = "
          f"{c_ade[np.arange(N), SEL].mean():.4f} m  vs  ADE(F[random]) = "
          f"{c_ade[np.arange(N), rnd].mean():.4f} m  vs  ADE(F[best]) = "
          f"{best_cost['ade_m'].mean():.4f} m  -- a wrong mapping would read "
          f"as the random draw")

    print(f"\n[P3] SELECTION REGRET  = cost(selected candidate) - "
          f"cost(best candidate in the SAME emitted fan), per family")
    print(f"  {'family':<18}{'n':>7}{'selected':>11}{'best-in-fan':>13}"
          f"{'regret':>10}{'p50':>9}{'p90':>9}{'p99':>9}"
          f"{'frac=0':>9}{'RND floor':>11}")
    p3, p3_floor = {}, {}
    for f in fams:
        ok = scored[f]
        r = regret[f][ok]
        c = np.where(np.isfinite(costs[f]), costs[f], np.inf)
        rf = c[np.arange(N), rnd] - best_cost[f]
        rf = rf[np.isfinite(rf)]
        p3_floor[f] = float(rf.mean()) if len(rf) else float("nan")
        p3[f] = {
            "n_scoreable": int(ok.sum()), "n_total": int(N),
            "selected": float(np.mean(sel_cost[f][ok])),
            "best_in_fan": float(np.mean(best_cost[f][ok])),
            "regret_mean": float(r.mean()),
            "regret_p50": float(np.percentile(r, 50)),
            "regret_p90": float(np.percentile(r, 90)),
            "regret_p99": float(np.percentile(r, 99)),
            "frac_optimal": float((r <= 1e-9).mean()),
            "uniform_random_pick_floor": p3_floor[f]}
        print(f"  {f:<18}{p3[f]['n_scoreable']:>7}{p3[f]['selected']:>11.4f}"
              f"{p3[f]['best_in_fan']:>13.4f}{p3[f]['regret_mean']:>10.4f}"
              f"{p3[f]['regret_p50']:>9.4f}{p3[f]['regret_p90']:>9.4f}"
              f"{p3[f]['regret_p99']:>9.4f}{p3[f]['frac_optimal']:>9.4f}"
              f"{p3_floor[f]:>11.4f}")
        p3[f]["ci"] = CI.episode_cluster_bootstrap(r, eid[ok],
                                                   n_boot=a.n_boot, dp=6)

    # ---- TACTICAL family: does the SELECTED candidate carry the right
    # manoeuvre when the fan contained one that did?  Labelled by the
    # programme's OWN canonical gate, never a second detector.
    # ⛔ THE GATE IS THE v2 CURVATURE GATE, NOT `|dyaw| > 0.15`.  The v1 yaw
    # threshold is the one the published `four_families.tactical_from_trajectory`
    # block uses (kappa=None) and the one the programme has RULED OUT as a
    # decision gate -- the recorded human path fails it.  The v2 branch
    # (`|kappa| >= 1/60`) is the one used here; the v1 read is printed BESIDE it
    # only so the published block stays comparable.
    from tanitad.refs.refc_tactical import (factor_from_kinematics,
                                            MIN_ARC_M)
    from taniteval.four_families import maneuver_kinematics

    def _arc(p):
        q = torch.cat([torch.zeros_like(p[:, :1]), p], 1)
        return (q[:, 1:] - q[:, :-1]).norm(dim=-1).sum(1)

    def _classes(p, gate):
        dy, dv, va, vb, _ = maneuver_kinematics(p, S.DT_S)
        kap = None if gate == "v1" else dy / _arc(p).clamp_min(MIN_ARC_M)
        la, lo = factor_from_kinematics(dy, dv, va, vb, kappa=kap)
        return la.numpy(), lo.numpy()

    Gt = torch.as_tensor(G, dtype=torch.float32)
    Ff = torch.as_tensor(F.reshape(-1, 4, 2), dtype=torch.float32)
    lat_g, lon_g = _classes(Gt, "v2")
    lat_c, lon_c = _classes(Ff, "v2")
    lat_c = lat_c.reshape(N, -1)
    lon_c = lon_c.reshape(N, -1)
    lat_g1, lon_g1 = _classes(Gt, "v1")
    lat_c1, lon_c1 = _classes(Ff, "v1")
    lat_c1, lon_c1 = lat_c1.reshape(N, -1), lon_c1.reshape(N, -1)
    print(f"\n[gate] TACTICAL read on the v2 CURVATURE gate (|kappa| >= 1/60). "
          f"GT lateral class counts v2 "
          f"{np.bincount(lat_g, minlength=3).tolist()} vs v1 (the published "
          f"block's gate, NOT used for decisions here) "
          f"{np.bincount(lat_g1, minlength=3).tolist()}")
    tac = {"_gate": "v2 curvature (|kappa| >= 1/60); v1 |dyaw| > 0.15 printed "
                    "beside it for comparability with the published block only",
           "_gt_lateral_counts_v2": np.bincount(lat_g, minlength=3).tolist(),
           "_gt_lateral_counts_v1": np.bincount(lat_g1, minlength=3).tolist()}
    for nm, cc, gg in (("tac_lat_wrong", lat_c, lat_g),
                       ("tac_lon_wrong", lon_c, lon_g),
                       ("tac_lat_wrong_v1gate", lat_c1, lat_g1),
                       ("tac_lon_wrong_v1gate", lon_c1, lon_g1)):
        wrong = (cc != gg[:, None]).astype(float)          # [N, 117]
        sel_w = wrong[np.arange(N), SEL]
        best_w = wrong.min(1)
        tac[nm] = {
            "selected_wrong": float(sel_w.mean()),
            "best_in_fan_wrong": float(best_w.mean()),
            "regret": float((sel_w - best_w).mean()),
            "fan_contained_a_correct_candidate": float(1.0 - best_w.mean()),
            "wrong_although_available":
                int(((sel_w == 1) & (best_w == 0)).sum()),
            "n": int(N),
            "n_candidates_correct_median": float(
                np.median((1.0 - wrong).sum(1)))}
        print(f"\n[P3 TACTICAL] {nm}: the SELECTED candidate is WRONG on "
              f"{tac[nm]['selected_wrong']*100:.2f} % of windows; the fan "
              f"contained a CORRECT candidate on "
              f"{tac[nm]['fan_contained_a_correct_candidate']*100:.2f} %; "
              f"WRONG ALTHOUGH a correct one was available: "
              f"{tac[nm]['wrong_although_available']} / {N} "
              f"({100.0*tac[nm]['wrong_although_available']/N:.2f} %)")
        print(f"              median number of the 117 candidates carrying "
              f"the correct class: "
              f"{tac[nm]['n_candidates_correct_median']:.0f}")
    p3["_tactical"] = tac
    p3["_strategic"] = {
        "status": "UNAVAILABLE",
        "reason": ("no per-window strategic label exists on this grid: the v7.2 "
                   "record carries ONE g_str token per CLIP, not per window, and "
                   "the nav command is a MODEL INPUT (PI: it simulates the "
                   "vehicle's nav system), never a target. A strategic regret "
                   "needs a per-window route/goal label - a WORK ITEM, not a "
                   "pass."), "n": 0}
    p3["_distance_keeping"] = {
        "status": "UNAVAILABLE",
        "reason": ("the banked b1 eval lead block is per-FRAME on a 0.2 s "
                   "10-step grid while the scored fan is on the 0.5 s 4-step "
                   "grid, and `taniteval.lead_metrics.distance_keeping` loops "
                   "per window in Python, so 117 candidates x 24k windows is "
                   "not runnable as-is. Lead coverage on the eval corpus is "
                   "8341/29556 frames (28.22 %). A WORK ITEM, not a pass."),
        "n": 0}

    # ---------------- P2: temporal stability --------------------------------
    print(f"\n[P2] TEMPORAL SELECTION STABILITY")
    # per-episode a_gt on the v0-conditioned fan = the CEILING's selection
    AGT = best_idx["ade_m"]
    off = np.cumsum([0] + [r["n"] for r in rows])
    per_ep = []
    sw_arm, sw_gt, sw_shuf, sw_rand = [], [], [], []
    lat_sw, lon_sw = [], []
    wp_arm, wp_gt, wp_rand = [], [], []
    pair_eid, pair_idx = [], []
    dwell = []
    for r in rows:
        i, j = off[r["ep"]], off[r["ep"] + 1]
        sel = SEL[i:j]
        agt = AGT[i:j]
        n = j - i
        if n < 2:
            continue
        ws = r["ws"]
        assert np.all(np.diff(ws) == 1), "stability needs a stride-1 grid"
        sw_arm.append(sel[1:] != sel[:-1])
        sw_gt.append(agt[1:] != agt[:-1])
        sh = np.random.default_rng(r["ep"]).permutation(sel)
        sw_shuf.append(sh[1:] != sh[:-1])
        rd = np.random.default_rng(1000 + r["ep"]).integers(0, ctrl.shape[0], n)
        sw_rand.append(rd[1:] != rd[:-1])
        lat_sw.append(LAT[i:j][1:] != LAT[i:j][:-1])
        lon_sw.append(LON[i:j][1:] != LON[i:j][:-1])
        pair_eid.append(np.full(n - 1, r["ep"]))
        pair_idx.append(np.arange(i + 1, j))     # the SECOND window of each pair
        # dwell times of the selection
        ch = np.flatnonzero(np.diff(sel)) + 1
        seg = np.diff(np.concatenate([[0], ch, [n]]))
        dwell.append(seg)
        # waypoint stability, exact rigid transform between consecutive frames.
        # The NEXT plan is brought into THIS frame carrying its own origin knot
        # at its own time (+stride*0.1 s), then resampled onto THIS plan's four
        # instants, so the two are compared at the SAME absolute times.
        P, Gg, Fr = r["os"], r["g"], F[i:j]
        for t in range(n - 1):
            pt = r["poses"][ws[t] + frame_off]
            pn = r["poses"][ws[t + 1] + frame_off]
            sh = (ws[t + 1] - ws[t]) * 0.1
            for src, cur, sink in ((P[t + 1], P[t], wp_arm),
                                   (Gg[t + 1], Gg[t], wp_gt),
                                   (Fr[t + 1, rd[t + 1]], Fr[t, rd[t]],
                                    wp_rand)):
                q, ts = S.plan_in_frame(src, pt, pn, sh)
                nx = S.resample(q, ts, S.PLAN_T)
                sink.append(float(np.linalg.norm(nx - cur, axis=-1).mean()))

    cat = np.concatenate
    peid = cat(pair_eid)
    pidx = cat(pair_idx)
    arm, gtc, shuf, rnd_ = (cat(sw_arm), cat(sw_gt), cat(sw_shuf),
                            cat(sw_rand))
    latc, lonc = cat(lat_sw), cat(lon_sw)
    wa, wg, wr = np.array(wp_arm), np.array(wp_gt), np.array(wp_rand)
    dw = cat(dwell)

    def row(name, v, unit=""):
        c = CI.episode_cluster_bootstrap(v.astype(float), peid,
                                         n_boot=a.n_boot, dp=6)
        print(f"  {name:<52}{c['mean']:>10.4f}{unit}  "
              f"[{c['lo']:.4f}, {c['hi']:.4f}]  n={c['n_windows']}")
        return c

    print(f"  DEFINITION: over CONSECUTIVE windows of the same clip "
          f"(stride 1 = 0.1 s apart), the fraction of adjacent pairs at which "
          f"the quantity CHANGES.  Lower is more stable.")
    p2 = {}
    p2["sel_switch_arm"] = row("SELECTION switch rate  -- refcv4b (the arm)",
                               arm)
    p2["sel_switch_gt_ceiling"] = row(
        "   CEILING: the GT's own best-in-fan choice", gtc)
    p2["sel_switch_shuffled_floor"] = row(
        "   FLOOR-A: the arm's own picks SHUFFLED in time", shuf)
    p2["sel_switch_uniform_floor"] = row(
        "   FLOOR-B: a uniform random pick from the fan", rnd_)
    p2["lat_token_switch"] = row("TACTICAL LATERAL token switch rate", latc)
    p2["lon_token_switch"] = row("TACTICAL LONGITUDINAL token switch rate",
                                 lonc)
    p2["waypoint_arm_m"] = row("WAYPOINT disagreement, refcv4b", wa, " m")
    p2["waypoint_gt_ceiling_m"] = row(
        "   CEILING/C3: the GT path against itself", wg, " m")
    p2["waypoint_rand_floor_m"] = row(
        "   FLOOR: an independent random candidate each frame", wr, " m")
    print(f"\n  SELECTION DWELL TIMES (consecutive frames holding one "
          f"candidate), n_runs={len(dw)}")
    print(f"    mean {dw.mean():.2f} frames = {dw.mean()*0.1:.2f} s   "
          f"median {np.median(dw):.0f}   p90 {np.percentile(dw,90):.0f}   "
          f"max {dw.max()}")
    for k in (1, 2, 3, 5, 10):
        print(f"    runs of <= {k:>2} frames ({k*0.1:.1f} s): "
              f"{int((dw<=k).sum())} / {len(dw)}  "
              f"({100.0*(dw<=k).sum()/len(dw):.2f} %)")
    p2["dwell"] = {"n_runs": int(len(dw)), "mean_frames": float(dw.mean()),
                   "median_frames": float(np.median(dw)),
                   "p90_frames": float(np.percentile(dw, 90)),
                   "max_frames": int(dw.max()),
                   "frac_le_1_frame": float((dw <= 1).mean()),
                   "frac_le_2_frames": float((dw <= 2).mean())}

    # does a switch move the DECISION or only the waypoints?
    both = arm & (latc | lonc)
    print(f"\n  DECOMPOSITION of the {int(arm.sum())} selection switches:")
    print(f"    move a TACTICAL TOKEN too : {int(both.sum())} "
          f"({100.0*both.sum()/max(1,arm.sum()):.2f} % of switches)")
    print(f"    waypoints only            : {int((arm & ~(latc|lonc)).sum())}")
    print(f"    token moved with NO selection switch: "
          f"{int(((latc|lonc) & ~arm).sum())}")
    p2["switch_decomposition"] = {
        "n_switches": int(arm.sum()),
        "switch_moves_token": int(both.sum()),
        "switch_waypoints_only": int((arm & ~(latc | lonc)).sum()),
        "token_moves_without_switch": int(((latc | lonc) & ~arm).sum())}

    # is a switching window WORSE?  (paired within episodes, so the shared
    # per-episode difficulty cancels; this is an ASSOCIATION, not a cause)
    sw_ade = os_ade[pidx][arm]
    st_ade = os_ade[pidx][~arm]
    reg_sw = regret["ade_m"][pidx][arm]
    reg_st = regret["ade_m"][pidx][~arm]
    print(f"\n  ADE of `os` at a window that SWITCHED its selection: "
          f"{sw_ade.mean():.4f} m (n={len(sw_ade)})  vs one that HELD: "
          f"{st_ade.mean():.4f} m (n={len(st_ade)})")
    print(f"  SELECTION REGRET (ADE) at a switching window: "
          f"{np.nanmean(reg_sw):.4f} m  vs at a holding window: "
          f"{np.nanmean(reg_st):.4f} m")
    p2["ade_at_switch"] = {"switch_mean": float(sw_ade.mean()),
                           "hold_mean": float(st_ade.mean()),
                           "n_switch": int(len(sw_ade)),
                           "n_hold": int(len(st_ade)),
                           "regret_switch": float(np.nanmean(reg_sw)),
                           "regret_hold": float(np.nanmean(reg_st))}

    # ---------------- P3b: WHICH AXIS does the selector get wrong? ----------
    a_lon = ctrl[:, 0].numpy()
    a_lat = ctrl[:, 1].numpy()
    lon_g_ = np.unique(a_lon)
    lat_g_ = np.unique(a_lat)
    ix_lon = {v: i for i, v in enumerate(lon_g_)}
    ix_lat = {v: i for i, v in enumerate(lat_g_)}
    ilon = np.array([ix_lon[v] for v in a_lon])
    ilat = np.array([ix_lat[v] for v in a_lat])
    bi = best_idx["ade_m"]
    dlon = np.abs(ilon[SEL] - ilon[bi])
    dlat = np.abs(ilat[SEL] - ilat[bi])
    print(f"\n[P3b] WHICH AXIS OF THE 13 x 9 (a_lon x a_lat) GRID IS MISSED?")
    print(f"  mean |grid steps| off the best candidate: "
          f"a_lon {dlon.mean():.3f} of 12 max, a_lat {dlat.mean():.3f} of 8 max")
    print(f"  exactly right on a_lon: {int((dlon==0).sum())} / {N} "
          f"({100.0*(dlon==0).mean():.2f} %)   "
          f"on a_lat: {int((dlat==0).sum())} / {N} "
          f"({100.0*(dlat==0).mean():.2f} %)   "
          f"on BOTH: {int(((dlon==0)&(dlat==0)).sum())} / {N} "
          f"({100.0*((dlon==0)&(dlat==0)).mean():.2f} %)")
    # what would fixing ONE axis buy?  (an ORACLE-on-one-axis counterfactual)
    fix_lon = np.array([np.flatnonzero((ilon == ilon[bi[i]]) &
                                       (ilat == ilat[SEL[i]]))[0]
                        for i in range(N)])
    fix_lat = np.array([np.flatnonzero((ilon == ilon[SEL[i]]) &
                                       (ilat == ilat[bi[i]]))[0]
                        for i in range(N)])
    c = np.where(np.isfinite(costs["ade_m"]), costs["ade_m"], np.inf)
    print(f"  ADE if the a_lon index were ORACLE and a_lat kept: "
          f"{c[np.arange(N), fix_lon].mean():.4f} m")
    print(f"  ADE if the a_lat index were ORACLE and a_lon kept: "
          f"{c[np.arange(N), fix_lat].mean():.4f} m")
    print(f"  ADE selected {sel_cost['ade_m'].mean():.4f} m -> "
          f"best-in-fan {best_cost['ade_m'].mean():.4f} m")
    p2["axis_decomposition"] = {
        "mean_grid_steps_off_a_lon": float(dlon.mean()),
        "mean_grid_steps_off_a_lat": float(dlat.mean()),
        "frac_a_lon_exact": float((dlon == 0).mean()),
        "frac_a_lat_exact": float((dlat == 0).mean()),
        "frac_both_exact": float(((dlon == 0) & (dlat == 0)).mean()),
        "ade_oracle_a_lon_only": float(c[np.arange(N), fix_lon].mean()),
        "ade_oracle_a_lat_only": float(c[np.arange(N), fix_lat].mean()),
        "ade_selected": float(sel_cost["ade_m"].mean()),
        "ade_best_in_fan": float(best_cost["ade_m"].mean())}

    # ---------------- P2c: THE SELECTION-LEVEL LEVER -- a grid DEADBAND -----
    # Deployable by construction: it reads only the arm's own selection history
    # and the (a_lon, a_lat) grid coordinates, never the ground truth.
    #   sel_h(t) = sel_h(t-1)  if the new pick is within K grid steps of the
    #                             held one on BOTH axes  (a small move = noise)
    #            = sel(t)      otherwise                 (a big move = a
    #                                                     decision, let it pass)
    # Scored at the SELECTION level (cost of the chosen CANDIDATE), because
    # that is the level the rule acts on; the decoder's refinement is not
    # re-run and no claim is made about it.
    # PRE-REGISTERED, both outcomes committed: if the small moves are noise the
    # deadband LOWERS every family; if they are the model tracking the scene it
    # RAISES them.  Reported either way.
    print(f"\n[P2c] SELECTION DEADBAND (deployable, no GT): hold the previous "
          f"pick while the new one is within K grid steps on both axes")
    a_lon0 = ctrl[:, 0].numpy()
    a_lat0 = ctrl[:, 1].numpy()
    gl = {v: i for i, v in enumerate(np.unique(a_lon0))}
    ga = {v: i for i, v in enumerate(np.unique(a_lat0))}
    IL = np.array([gl[v] for v in a_lon0])
    IA = np.array([ga[v] for v in a_lat0])
    dead = {}
    for K in (1, 2):
        selh = SEL.copy()
        for r in rows:
            i, j = off[r["ep"]], off[r["ep"] + 1]
            held = SEL[i]
            for t in range(i, j):
                if (abs(IL[SEL[t]] - IL[held]) <= K
                        and abs(IA[SEL[t]] - IA[held]) <= K):
                    selh[t] = held
                else:
                    held = SEL[t]
                    selh[t] = held
        sw_h = []
        for r in rows:
            i, j = off[r["ep"]], off[r["ep"] + 1]
            sw_h.append(selh[i + 1:j] != selh[i:j - 1])
        sw_h = cat(sw_h)
        row_ = {"switch_rate": float(sw_h.mean()),
                "switch_rate_baseline": float(arm.mean())}
        print(f"  K={K}: selection switch rate {sw_h.mean():.4f} "
              f"(baseline {arm.mean():.4f}, GT ceiling {gtc.mean():.4f}); "
              f"changed the pick on "
              f"{int((selh != SEL).sum())} / {N} windows")
        for f in ("ade_m", "along_mae_m", "cross_mae_m", "speed_mae_mps",
                  "heading_mae_deg"):
            c = np.where(np.isfinite(costs[f]), costs[f], np.nan)
            aa = c[np.arange(N), selh]
            bb = c[np.arange(N), SEL]
            ok = np.isfinite(aa) & np.isfinite(bb)
            d = CI.paired_episode_cluster_bootstrap(aa[ok], bb[ok], eid[ok],
                                                    n_boot=a.n_boot)
            row_[f] = d
            print(f"      {f:<18} {aa[ok].mean():>9.4f} vs {bb[ok].mean():>9.4f}"
                  f"  delta {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                  f"  separated={d['separated']}  "
                  f"{'BETTER' if d['delta'] < 0 else 'WORSE'}")
        dead[str(K)] = row_
    p2["selection_deadband_lever"] = dead

    # ---------------- P2b: THE NEXT LEVER -- does temporal smoothing help? --
    # PRE-REGISTERED, both outcomes committed BEFORE the numbers were read:
    #   if the frame-to-frame movement is NOISE, blending the previous plan
    #      forward REDUCES the four-family errors (or is neutral);
    #   if it is the model REACTING to new observation, blending REINTRODUCES
    #      stale information and every family gets WORSE.
    # Either way it is reported plainly.  This costs no GPU: it is a filter on
    # the banked output.
    print(f"\n[P2b] NEXT LEVER (pre-registered, zero GPU): blend the PREVIOUS "
          f"plan forward into the current frame, os_a = (1-alpha)*os(t) + "
          f"alpha*T[os(t-1)]")
    smooth = {}
    base_fam = S.family_costs(OS, G, all_v0)
    for alpha in (0.25, 0.5, 0.75):
        SM = OS.copy()
        for r in rows:
            i, j = off[r["ep"]], off[r["ep"] + 1]
            ws = r["ws"]
            P = r["os"]
            prev = None
            for t in range(1, j - i):
                pt = r["poses"][ws[t] + frame_off]
                pp = r["poses"][ws[t - 1] + frame_off]
                q, ts = S.plan_in_frame(P[t - 1], pt, pp,
                                        -(ws[t] - ws[t - 1]) * 0.1)
                back = S.resample(q, ts, S.PLAN_T)
                SM[i + t] = (1 - alpha) * P[t] + alpha * back
        fam = S.family_costs(SM, G, all_v0)
        r4 = {}
        for f in ("ade_m", "along_mae_m", "cross_mae_m", "speed_mae_mps",
                  "curv_mae_1pm", "heading_mae_deg"):
            ok = np.isfinite(fam[f]) & np.isfinite(base_fam[f])
            d = CI.paired_episode_cluster_bootstrap(
                fam[f][ok], base_fam[f][ok], eid[ok], n_boot=a.n_boot)
            r4[f] = d
            print(f"    alpha={alpha}  {f:<18} {fam[f][ok].mean():>9.4f} vs "
                  f"{base_fam[f][ok].mean():>9.4f}  delta "
                  f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                  f"separated={d['separated']}  "
                  f"{'BETTER' if d['delta'] < 0 else 'WORSE'}")
        smooth[str(alpha)] = r4
        smooth[str(alpha)]["_smoothed_paths"] = SM
    p2["smoothing_lever"] = {k: {kk: vv for kk, vv in v.items()
                                 if kk != "_smoothed_paths"}
                             for k, v in smooth.items()}

    # ⛔ the LATERAL family must be read on the PROGRAMME'S OWN estimator with
    # the straight-line floor beside it, not on selstab's fan-comparison
    # geometry.  `ha0` (a = 0, kappa = 0 at the measured v0) IS that floor.
    from taniteval import four_families as FF
    HA0 = np.concatenate([r_["ha0"] for r_ in
                          [{"ha0": e["ha0"].astype(np.float64)} for e in eps]])
    gt_t = torch.as_tensor(G, dtype=torch.float32)
    print(f"\n[P2b/LATERAL on the PROGRAMME'S estimator, with its floor]")
    lat_rows = {}
    for nm, arr in (("ha0 (straight-line FLOOR)", HA0),
                    ("os  (the arm)", OS),
                    ("os smoothed alpha=0.25", smooth["0.25"]["_smoothed_paths"]),
                    ("os smoothed alpha=0.5", smooth["0.5"]["_smoothed_paths"])):
        b = FF.lateral(torch.as_tensor(arr, dtype=torch.float32), gt_t,
                       dt=S.DT_S, eid=eid.tolist(), n_boot=200)
        lat_rows[nm] = {k: b.get(k) for k in
                        ("curvature_mae_1pm", "heading_mae_deg",
                         "yaw_rate_mae_degps", "cross_mae_m",
                         "cross_final_mae_m", "n_steps_curvature")}
        print(f"  {nm:<26} curvature_mae {b['curvature_mae_1pm']:.6f}  "
              f"heading {b['heading_mae_deg']:.4f} deg  "
              f"cross {b['cross_mae_m']:.4f} m  "
              f"n_steps_curv {b['n_steps_curvature']}")
    p2["lateral_programme_estimator"] = lat_rows

    # ⭐ CROSS-VALIDATION AGAINST THE PUBLISHED ARTIFACT.  The published
    # refcv4b T1 record scores the SAME arm on the stride-5 grid (ws = 7, 12,
    # 17, ...), and its LATERAL row reads curvature_mae_1pm `os` 0.008097 /
    # `ha0` 0.006802.  Restricting this stride-1 roll to ws % 5 == 2 recovers
    # exactly those windows, so the two must agree -- and if they do, this
    # locally-rolled dump IS the published arm.
    ws_all = np.concatenate([r["ws"] for r in rows])
    sub = (ws_all % 5) == (7 % 5)
    print(f"\n[X-VAL vs the PUBLISHED refcv4b T1 record] stride-5 subset: "
          f"{int(sub.sum())} windows (published grid = 4823)")
    xv = {}
    for nm, arr, pub in (("os", OS, 0.008097), ("ha0", HA0, 0.006802)):
        b = FF.lateral(torch.as_tensor(arr[sub], dtype=torch.float32),
                       gt_t[sub], dt=S.DT_S, eid=eid[sub].tolist(), n_boot=200)
        xv[nm] = {"curvature_mae_1pm": b["curvature_mae_1pm"],
                  "published": pub,
                  "delta": round(b["curvature_mae_1pm"] - pub, 6)}
        print(f"  {nm:<4} curvature_mae_1pm  local {b['curvature_mae_1pm']:.6f}"
              f"   published {pub:.6f}   delta "
              f"{b['curvature_mae_1pm'] - pub:+.6f}")
    p2["xval_vs_published"] = xv

    out = {"_tool": "selstab / p2p3_run.py",
           "_tier": "T1 (self-action OPEN loop)",
           "_evidence_class": "MEASURED (ours)",
           "dump": a.dump, "ckpt": S.CKPT, "anchor_controls_key": ckey,
           "n_windows": int(N), "n_episodes": int(len(rows)),
           "controls": {"v0_roundtrip_max_abs": v0_err,
                        "straight_candidate_max_abs_y_m":
                            float(np.abs(F[:, i0, :, 1]).max())},
           "P3_regret": p3, "P3_regret_uniform_floor": p3_floor,
           "P3_ceiling_proof": {
               "our_best_equals_dumped_a_star_frac": agree,
               "a_star_fixedbank_ade_m": float(np.nanmean(dstar)),
               "our_best_in_fan_ade_m": float(np.nanmean(best_cost["ade_m"])),
               "selected_candidate_ade_m": float(np.nanmean(sel_cost["ade_m"])),
               "os_refined_ade_m": float(os_ade.mean())},
           "P2_stability": p2}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
