"""Pre-registered gate: does LEAD STATE add information about the ego's future 2 s
longitudinal displacement, over and above ego state alone?

WHY THIS EXISTS
---------------
`obstacle.offline` (3D agent tracks, 96.90 % of our corpus) was never ingested;
`lead_state` is a `None` stub, and headway/TTC metrics were refused in
`TANITEVAL_V2_METRIC_SUITE.md` for exactly that reason. Before spending 12.4 GB
and 2-3 eng-days on an ingest, we test the PREMISE the ingest rests on:

    "agent state unblocks tactics" ==> knowing the lead vehicle must measurably
    predict how far the ego travels along-track over the next 2 s.

It targets the program's MEASURED dominant residual: 83 % of the 2 s error is
along-track. A negative result therefore cannot be blamed on a training bug --
it is a statement about the data, not about a model.

PRE-REGISTERED DECISION RULE (committed BEFORE the numbers were read)
--------------------------------------------------------------------
Primary statistic: paired held-out RELATIVE reduction in mean-absolute error of
the 2 s longitudinal displacement, arm A (ego only) -> arm B (ego + lead), with a
PAIRED EPISODE-CLUSTER BOOTSTRAP over held-out episodes (taniteval.ci). The
legacy `overlapping_holdout_se` is 1.28-2.06x too narrow and may not decide this.

    >= 15 %                          -> PASS   : ingest all 197 chunks
    <= 5 %, OR the CI spans 0        -> FAIL   : premise falsified, do NOT ingest
    5-15 %                           -> AMBIGUOUS: report the number, escalate

Do NOT tune the experiment until it agrees.

DESIGN NOTES THAT MATTER
------------------------
* STRICTLY CAUSAL lead features. For a window at time t only obstacle rows with
  `timestamp_us <= t` are used (nearest sample, staleness <= MAX_STALE_S; closing
  speed from the two most recent samples of that track). Interpolating a track
  ACROSS t would import the lead's future position, which is correlated with the
  ego's future displacement through shared traffic dynamics -- a class-C6
  confound that would inflate arm B.
* Boxes are already in the **rig frame at their own timestamp**
  (`reference_frame = rig`), so a backward difference of `center_x` IS the
  relative closing rate; no ego-motion compensation is needed or wanted.
* NEGATIVE CONTROL arm `B_shuf`: identical lead features, permuted across
  EPISODES. If the shuffled arm also improves, the pipeline leaks and the result
  is void.
* Identical regressor + identical hyper-parameters for every arm; arm B's extra
  columns can only help it through held-out generalisation.
* Episode-disjoint split. Primary = the program's own canonical split
  (sorted clip ids -> torch.randperm(seed 0) -> first 20 % val), so the held-out
  episodes are episodes no TanitAD arm has ever trained on. Secondary = 5-fold
  episode-level out-of-fold over every clip, for power.

PARITY
------
Read-only. This script never writes to `_epcache`, never re-selects clips and
never touches `r0_selection.parquet`. It only reads label zips.

Usage
-----
    python scripts/lead_state_gate.py build --out <dir> [--chunks 36 170 ...]
    python scripts/lead_state_gate.py gate  --out <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants -- fixed before any number was read                               #
# --------------------------------------------------------------------------- #
ROOT_DEFAULT = r"C:\Users\Admin\tanitad-data\physicalai"
TARGET_HZ = 10.0
HORIZON_S = 2.0            # the 2 s the program's ADE is measured over
HIST_S = 1.0               # ego history depth -> grid starts at t = HIST_S
CLIP_END_S = 20.0          # dense egomotion + obstacle coverage is 0..20 s
MAX_STALE_S = 0.5          # doc's association gap guard
DIFF_BASE_S = 0.5          # nominal backward baseline for the closing rate
MIN_DIFF_S = 0.15          # shorter baselines differentiate box noise, not motion
MAX_DIFF_S = 1.5           # oldest sample admissible for a backward difference
CLOSING_CLIP_MS = 20.0     # |closing| beyond this is a track switch, not a vehicle
LEAD_MAX_GAP_M = 80.0
LEAD_LAT_M = 2.0
VEHICLE_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer")
BIG_CLASSES = ("heavy_truck", "bus", "trailer")
TTC_CAP_S = 30.0

SEED = 0
N_BOOT = 2000
PASS_THRESHOLD = 0.15
FAIL_THRESHOLD = 0.05

EGO_COLS = ["v", "ax", "ay", "curv", "yawrate", "dv_0p5", "dv_1p0",
            "v_lag_0p5", "v_lag_1p0", "abs_curv"]
LEAD_COLS = ["lead_present", "gap_m", "closing_ms", "ttc_s", "inv_ttc",
             "lead_lat_m", "lead_is_big"]
DENS_COLS = ["n_ahead_50m", "n_vru_near"]


# --------------------------------------------------------------------------- #
# Geometry / signal helpers                                                    #
# --------------------------------------------------------------------------- #
def quaternion_yaw(qx, qy, qz, qw):
    """Same convention as `tanitad.data.physicalai.quaternion_yaw`."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _interp(t_q, t, v):
    return np.interp(t_q, t, v)


def ego_frame(ego: pd.DataFrame, t_grid_s: np.ndarray, horizons=()) -> dict:
    """Ego features + the 2 s along-track target on `t_grid_s` (seconds).

    `horizons` (POST-HOC only) adds `y_long_h{H}` / `dv_h{H}` columns; the
    pre-registered `y_long` at HORIZON_S is unaffected by their presence.
    """
    t = ego["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    t = t[o]
    col = {c: ego[c].to_numpy(np.float64)[o] for c in
           ("qx", "qy", "qz", "qw", "x", "y", "vx", "vy", "ax", "ay", "curvature")}
    yaw_u = np.unwrap(quaternion_yaw(col["qx"], col["qy"], col["qz"], col["qw"]))

    speed = np.hypot(col["vx"], col["vy"])

    def at(ts, c):
        return _interp(ts, t, col[c])

    v = _interp(t_grid_s, t, speed)
    yaw = _interp(t_grid_s, t, yaw_u)
    x0, y0 = at(t_grid_s, "x"), at(t_grid_s, "y")
    x1 = at(t_grid_s + HORIZON_S, "x")
    y1 = at(t_grid_s + HORIZON_S, "y")
    dx, dy = x1 - x0, y1 - y0
    s_long = dx * np.cos(yaw) + dy * np.sin(yaw)
    s_lat = -dx * np.sin(yaw) + dy * np.cos(yaw)

    # yaw rate from a symmetric window entirely in the PAST
    yaw_b = _interp(t_grid_s - 0.2, t, yaw_u)
    v_l05 = _interp(t_grid_s - 0.5, t, speed)
    v_l10 = _interp(t_grid_s - 1.0, t, speed)
    curv = at(t_grid_s, "curvature")
    extra = {}
    t_end = float(t[-1])
    for h in horizons:
        xh, yh = at(t_grid_s + h, "x"), at(t_grid_s + h, "y")
        s = (xh - x0) * np.cos(yaw) + (yh - y0) * np.sin(yaw)
        vh = _interp(t_grid_s + h, t, speed)
        bad = (t_grid_s + h) > min(t_end, CLIP_END_S)
        extra[f"y_long_h{h}"] = np.where(bad, np.nan, s)
        extra[f"dv_h{h}"] = np.where(bad, np.nan, vh - v)
    return {**extra, **{
        "v": v,
        "ax": at(t_grid_s, "ax"),
        "ay": at(t_grid_s, "ay"),
        "curv": curv,
        "abs_curv": np.abs(curv),
        "yawrate": (yaw - yaw_b) / 0.2,
        "dv_0p5": v - v_l05,
        "dv_1p0": v - v_l10,
        "v_lag_0p5": v_l05,
        "v_lag_1p0": v_l10,
        "y_long": s_long,
        "y_lat": s_lat,
    }}


def lead_frame(obs: pd.DataFrame, t_grid_s: np.ndarray) -> dict:
    """STRICTLY CAUSAL lead + density features on `t_grid_s`."""
    n = t_grid_s.size
    out = {c: np.full(n, np.nan) for c in LEAD_COLS + DENS_COLS}
    out["lead_present"] = np.zeros(n)
    out["n_ahead_50m"] = np.zeros(n)
    out["n_vru_near"] = np.zeros(n)
    if obs is None or obs.empty:
        return out

    ts = obs["timestamp_us"].to_numpy(np.float64) / 1e6
    tid = obs["track_id"].to_numpy(str)
    cx = obs["center_x"].to_numpy(np.float64)
    cy = obs["center_y"].to_numpy(np.float64)
    sx = obs["size_x"].to_numpy(np.float64)
    cls = obs["label_class"].to_numpy(str)

    best_gap = np.full(n, np.inf)
    for track in np.unique(tid):
        m = tid == track
        tt = ts[m]
        o = np.argsort(tt)
        tt = tt[o]
        xx, yy, ss = cx[m][o], cy[m][o], sx[m][o]
        klass = cls[m][o][0]
        # last sample at or before each grid time
        j = np.searchsorted(tt, t_grid_s, side="right") - 1
        ok = j >= 0
        if not ok.any():
            continue
        jj = np.clip(j, 0, len(tt) - 1)
        stale = t_grid_s - tt[jj]
        ok &= (stale >= 0) & (stale <= MAX_STALE_S)
        if not ok.any():
            continue
        gap = xx[jj] - ss[jj] / 2.0
        lat = yy[jj]

        is_veh = klass in VEHICLE_CLASSES
        is_vru = klass in ("person", "rider", "stroller", "animal")
        if is_veh:
            out["n_ahead_50m"] += (ok & (gap > 0) & (gap < 50.0)
                                   & (np.abs(lat) < 8.0)).astype(float)
        if is_vru:
            out["n_vru_near"] += (ok & (gap > -5.0) & (gap < 50.0)
                                  & (np.abs(lat) < 8.0)).astype(float)
        if not is_veh:
            continue

        cand = ok & (gap >= 0.0) & (gap <= LEAD_MAX_GAP_M) & (np.abs(lat) < LEAD_LAT_M)
        if not cand.any():
            continue
        # CAUSAL backward difference over a ~DIFF_BASE_S baseline. A two-sample
        # difference over dt as small as 20 ms differentiates box noise, not
        # motion (it produced 28 m/s "closing rates" in the smoke run), so the
        # older endpoint is the last sample at or before t - DIFF_BASE_S.
        k0 = np.searchsorted(tt, t_grid_s - DIFF_BASE_S, side="right") - 1
        k0 = np.clip(k0, 0, len(tt) - 1)
        dt = tt[jj] - tt[k0]
        gap_prev = xx[k0] - ss[k0] / 2.0
        good_d = (dt >= MIN_DIFF_S) & ((t_grid_s - tt[k0]) <= MAX_DIFF_S)
        closing = np.where(good_d, -(gap - gap_prev) / np.where(dt > 0, dt, 1.0),
                           np.nan)
        closing = np.clip(closing, -CLOSING_CLIP_MS, CLOSING_CLIP_MS)

        win = cand & (gap < best_gap)
        if not win.any():
            continue
        best_gap = np.where(win, gap, best_gap)
        out["lead_present"] = np.where(win, 1.0, out["lead_present"])
        out["gap_m"] = np.where(win, gap, out["gap_m"])
        out["lead_lat_m"] = np.where(win, lat, out["lead_lat_m"])
        out["lead_is_big"] = np.where(win, float(klass in BIG_CLASSES),
                                      out["lead_is_big"])
        out["closing_ms"] = np.where(win, closing, out["closing_ms"])

    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(out["closing_ms"] > 0.1, out["gap_m"] / out["closing_ms"],
                       TTC_CAP_S)
    ttc = np.clip(ttc, 0.0, TTC_CAP_S)
    ttc[np.isnan(out["gap_m"])] = np.nan
    out["ttc_s"] = ttc
    out["inv_ttc"] = np.where(np.isnan(ttc), np.nan, 1.0 / np.maximum(ttc, 0.1))
    return out


# --------------------------------------------------------------------------- #
# BUILD                                                                        #
# --------------------------------------------------------------------------- #
def build(args):
    root = Path(args.root)
    sel = pd.read_parquet(root / "r0" / "phase0_selection.parquet")
    sel["clip_id"] = sel["clip_id"].astype(str)
    chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))
    country_of = dict(zip(sel["clip_id"], sel["country"].astype(str)))

    # canonical program split: sorted ids -> randperm(seed 0) -> first 20 % val
    import torch
    ids = sorted(sel["clip_id"].tolist())
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(ids), generator=g).tolist()
    n_val = max(1, int(len(ids) * 0.2))
    val_ids = {ids[i] for i in perm[:n_val]}

    obs_dir = root / "labels" / "obstacle.offline"
    ego_dir = root / "labels" / "egomotion"
    if args.chunks:
        chunks = [int(c) for c in args.chunks]
    else:
        chunks = sorted(int(p.name.split("_")[-1].split(".")[0])
                        for p in obs_dir.glob("obstacle.offline.chunk_*.zip"))
    print(f"[build] {len(chunks)} chunks", flush=True)

    hz = [int(h) for h in (args.horizons or [])]
    # POST-HOC builds need room for the LONGEST horizon; the pre-registered build
    # (no --horizons) keeps its original t in [1.0, 18.0] grid exactly.
    last_t = CLIP_END_S - (max(hz) if hz else HORIZON_S)
    t_grid = np.arange(HIST_S, last_t + 1e-9, 1.0 / TARGET_HZ)
    rows = []
    n_clip = 0
    for ci, c in enumerate(chunks):
        ozp, ezp = (obs_dir / f"obstacle.offline.chunk_{c:04d}.zip",
                    ego_dir / f"egomotion.chunk_{c:04d}.zip")
        if not (ozp.exists() and ezp.exists()):
            print(f"[build] chunk {c:04d}: missing zip, skip", flush=True)
            continue
        with zipfile.ZipFile(ozp) as oz, zipfile.ZipFile(ezp) as ez:
            omap = {n.split(".")[0]: n for n in oz.namelist()}
            emap = {n.split(".")[0]: n for n in ez.namelist()}
            want = [cid for cid in omap if chunk_of.get(cid) == c]
            for cid in sorted(want):
                if cid not in emap:
                    continue
                ego = pd.read_parquet(io.BytesIO(ez.read(emap[cid])))
                obs = pd.read_parquet(io.BytesIO(oz.read(omap[cid])))
                tmax = float(ego["timestamp"].max()) / 1e6
                if tmax < CLIP_END_S - 0.05:
                    continue
                ef = ego_frame(ego, t_grid, horizons=hz)
                lf = lead_frame(obs, t_grid)
                d = {**ef, **lf}
                d["t_s"] = t_grid
                d["clip_id"] = cid
                d["chunk"] = c
                d["country"] = country_of.get(cid, "?")
                d["split"] = "val" if cid in val_ids else "train"
                rows.append(pd.DataFrame(d))
                n_clip += 1
        print(f"[build] chunk {c:04d} ({ci+1}/{len(chunks)}): "
              f"{n_clip} clips cumulative", flush=True)

    df = pd.concat(rows, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "lead_gate_windows_h.parquet" if hz else "lead_gate_windows.parquet"
    df.to_parquet(out / name, index=False)
    print(f"[build] {len(df)} windows / {df.clip_id.nunique()} clips "
          f"-> {out/name}", flush=True)
    print(f"[build] lead present on {100*df.lead_present.mean():.2f} % of windows; "
          f"{100*df.groupby('clip_id').lead_present.max().mean():.1f} % of clips "
          f"have a lead at some point", flush=True)


# --------------------------------------------------------------------------- #
# GATE                                                                         #
# --------------------------------------------------------------------------- #
def _fit_predict(Xtr, ytr, Xte, kind, seed=SEED):
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if kind == "ridge":
        mu = np.nanmean(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), mu, Xtr)
        Xte = np.where(np.isnan(Xte), mu, Xte)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        Ztr = np.column_stack([Ztr, np.ones(len(Ztr))])
        Zte = np.column_stack([Zte, np.ones(len(Zte))])
        lam = 1.0 * np.eye(Ztr.shape[1])
        lam[-1, -1] = 0.0
        w = np.linalg.solve(Ztr.T @ Ztr + lam, Ztr.T @ ytr)
        return Zte @ w
    raise KeyError(kind)


def _arms(df, rng):
    """feature-column sets for every arm, incl. the shuffled negative control."""
    lead = df[LEAD_COLS].to_numpy(np.float64)
    # permute lead block across EPISODES (keeps within-episode structure intact)
    eps = df["clip_id"].to_numpy(str)
    uniq = np.unique(eps)
    order = rng.permutation(len(uniq))
    remap = {u: uniq[order[i]] for i, u in enumerate(uniq)}
    idx_by_ep = {u: np.flatnonzero(eps == u) for u in uniq}
    shuf = np.empty_like(lead)
    for u in uniq:
        src = idx_by_ep[remap[u]]
        dst = idx_by_ep[u]
        take = np.resize(src, len(dst))
        shuf[dst] = lead[take]
    ego = df[EGO_COLS].to_numpy(np.float64)
    dens = df[DENS_COLS].to_numpy(np.float64)
    return {
        "A": ego,
        "B": np.column_stack([ego, lead]),
        "B_shuf": np.column_stack([ego, shuf]),
        "Bplus": np.column_stack([ego, lead, dens]),
    }


def _folds_oof(df, y, arms, kind, k=5, seed=SEED):
    eps = df["clip_id"].to_numpy(str)
    uniq = np.unique(eps)
    rng = np.random.default_rng(seed)
    fold_of = {u: i for u, i in zip(uniq, rng.permutation(len(uniq)) % k)}
    fid = np.array([fold_of[e] for e in eps])
    pred = {a: np.full(len(y), np.nan) for a in arms}
    for f in range(k):
        te, tr = fid == f, fid != f
        for a, X in arms.items():
            pred[a][te] = _fit_predict(X[tr], y[tr], X[te], kind)
        print(f"    fold {f+1}/{k} done", flush=True)
    return pred


def gate(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))
    from taniteval.ci import (episode_cluster_bootstrap,
                              paired_episode_cluster_bootstrap)

    out = Path(args.out)
    df = pd.read_parquet(out / "lead_gate_windows.parquet")
    df = df[np.isfinite(df["y_long"]) & np.isfinite(df["v"])].reset_index(drop=True)
    y = df["y_long"].to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    arms = _arms(df, rng)

    corpus = {
        "n_windows": int(len(df)),
        "n_clips": int(df.clip_id.nunique()),
        "n_chunks": int(df.chunk.nunique()),
        "n_countries": int(df.country.nunique()),
        "lead_present_frac_windows": round(float(df.lead_present.mean()), 4),
        "clips_with_any_lead_frac": round(
            float(df.groupby("clip_id").lead_present.max().mean()), 4),
        "mean_agents_ahead_50m": round(float(df.n_ahead_50m.mean()), 3),
        "y_long_mean_m": round(float(y.mean()), 3),
        "y_long_std_m": round(float(y.std()), 3),
    }
    print(json.dumps(corpus, indent=2), flush=True)

    results = {"corpus": corpus, "preregistered": {
        "primary": "relative reduction in MAE of 2 s along-track displacement, "
                   "arm A (ego) -> arm B (ego+lead), paired episode-cluster "
                   "bootstrap over held-out episodes",
        "pass": ">= 15 %", "fail": "<= 5 % or CI spans 0",
        "ambiguous": "5-15 % -> escalate",
        "n_boot": N_BOOT, "seed": SEED,
        "estimator": "paired_episode_cluster_bootstrap (taniteval.ci)"}}

    for kind in args.models:
        for design in args.designs:
            tag = f"{kind}|{design}"
            print(f"\n=== {tag} ===", flush=True)
            if design == "canonical":
                tr = (df["split"] == "train").to_numpy()
                te = ~tr
                if te.sum() == 0 or tr.sum() == 0:
                    print("  empty split, skip", flush=True)
                    continue
                pred = {a: _fit_predict(X[tr], y[tr], X[te], kind)
                        for a, X in arms.items()}
                sub = df[te].reset_index(drop=True)
                ytrue = y[te]
            else:
                p = _folds_oof(df, y, arms, kind, k=5, seed=SEED)
                pred = p
                sub = df
                ytrue = y

            eid = sub["clip_id"].to_numpy(str)
            ae = {a: np.abs(pred[a] - ytrue) for a in pred}
            se = {a: (pred[a] - ytrue) ** 2 for a in pred}
            entry = {"n_test_windows": int(len(ytrue)),
                     "n_test_episodes": int(len(np.unique(eid)))}
            for a in ("A", "B", "B_shuf", "Bplus"):
                entry[f"mae_{a}"] = episode_cluster_bootstrap(
                    ae[a], eid, reduce="mean", n_boot=N_BOOT, seed=SEED)
                entry[f"rmse_{a}"] = episode_cluster_bootstrap(
                    se[a], eid, reduce="rms", n_boot=N_BOOT, seed=SEED)

            for a in ("B", "B_shuf", "Bplus"):
                d = paired_episode_cluster_bootstrap(
                    ae["A"], ae[a], eid, n_boot=N_BOOT, seed=SEED, reduce="mean")
                base = entry["mae_A"]["mean"]
                # relative reduction, bootstrapped on the SAME episode draws
                rel = _rel_reduction(ae["A"], ae[a], eid, N_BOOT, SEED)
                entry[f"paired_mae_A_minus_{a}"] = d
                entry[f"rel_reduction_{a}"] = rel
                print(f"  MAE A {base:.4f} -> {a} {entry[f'mae_{a}']['mean']:.4f} "
                      f"| abs delta {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] "
                      f"| rel {100*rel['point']:.2f} % "
                      f"[{100*rel['lo']:.2f},{100*rel['hi']:.2f}] "
                      f"separated={d['separated']}", flush=True)

            # lead-present subgroup (diagnostic only, NOT the gate)
            mask = sub["lead_present"].to_numpy() > 0.5
            if mask.sum() > 100:
                entry["lead_present_subgroup"] = {
                    "n_windows": int(mask.sum()),
                    "n_episodes": int(len(np.unique(eid[mask]))),
                    "rel_reduction_B": _rel_reduction(
                        ae["A"][mask], ae["B"][mask], eid[mask], N_BOOT, SEED),
                }
            results[tag] = entry

    # verdict on the PRIMARY cell
    primary = f"{args.models[0]}|{args.designs[0]}"
    rel = results[primary]["rel_reduction_B"]
    sep = results[primary]["paired_mae_A_minus_B"]["separated"]
    if (not sep) or rel["point"] <= FAIL_THRESHOLD:
        verdict = "FAIL"
    elif rel["point"] >= PASS_THRESHOLD:
        verdict = "PASS"
    else:
        verdict = "AMBIGUOUS"
    results["verdict"] = {
        "cell": primary, "verdict": verdict,
        "rel_reduction": rel, "ci_excludes_zero": sep,
        "rule": "PASS >=15 % | FAIL <=5 % or CI spans 0 | else AMBIGUOUS"}
    print(f"\nVERDICT [{primary}] {verdict}: "
          f"{100*rel['point']:.2f} % [{100*rel['lo']:.2f}, {100*rel['hi']:.2f}]",
          flush=True)

    p = out / "lead_gate_result.json"
    p.write_text(json.dumps(results, indent=2))
    print(f"[gate] -> {p}", flush=True)


def _rel_reduction(ae_a, ae_b, eid, n_boot, seed):
    """Bootstrap CI on (MAE_A - MAE_B)/MAE_A with EPISODE resampling."""
    from taniteval.ci import episode_index
    uniq, idx_by_ep = episode_index(eid)
    point = float((np.mean(ae_a) - np.mean(ae_b)) / np.mean(ae_a))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_ep[p] for p in pick])
        a, b = np.mean(ae_a[sel]), np.mean(ae_b[sel])
        vals[i] = (a - b) / a
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": round(point, 5), "lo": round(float(lo), 5),
            "hi": round(float(hi), 5), "n_boot": int(n_boot),
            "n_episodes": int(len(uniq)),
            "estimator": "episode_cluster_bootstrap(relative)"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default=ROOT_DEFAULT)
    b.add_argument("--out", required=True)
    b.add_argument("--chunks", nargs="*", default=None)
    b.add_argument("--horizons", nargs="*", default=None,
                   help="POST-HOC only: extra y_long_h{H}/dv_h{H} target columns; "
                        "writes lead_gate_windows_h.parquet, leaving the "
                        "pre-registered lead_gate_windows.parquet untouched")
    b.set_defaults(fn=build)
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    g.add_argument("--models", nargs="*", default=["gbm", "ridge"])
    g.add_argument("--designs", nargs="*", default=["canonical", "oof5"])
    g.set_defaults(fn=gate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
