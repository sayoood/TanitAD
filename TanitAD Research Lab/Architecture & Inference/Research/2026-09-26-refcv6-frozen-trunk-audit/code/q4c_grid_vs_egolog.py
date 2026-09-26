"""Q4c -- the decisive time-base identity: recover each cache row's TRUE TIMESTAMP from the
100 Hz egomotion log itself, then read the row step and the grid origin off it.

Why this is independent of everything under test: the cache's (x, y) per row were produced by
`np.interp(t_query, t_ego, x_ego)` (physicalai.signals_at, physicalai.py:596-649). Inverting that
map -- projecting each row's (x, y) onto the log's own polyline and reading the log's own clock
at the projection -- returns t_query without assuming ANY cadence, stride, n_stack or dt.
Then

    t_row(j) = a + b * j        (j = RAW index = provider row + n_stack - 1)

b is the true row step; (a - t_ego[0]) is where the grid starts on the LABEL timeline, which
`egomotion_source.load` defines as `ts - ts[0]` of the same log (egomotion_source.py:123-124).

CONTROLS (literals):
  K1 the fit residual must be ~0 (the grid IS a linspace) -- reported in ms; a non-linear grid
     would show here and void the linear model;
  K2 a synthetic row set built by interpolating the log at a KNOWN linspace (dt 0.1 exactly,
     start +0.037 s) must be recovered to < 0.1 ms / < 1e-5 relative -- the inversion works;
  K3 the recovered row times must reproduce the row's logged speed: interp(t_row, t_ego,
     hypot(vx, vy)) vs the cache's poses[:, 3] (float32), max abs diff reported.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import pandas as pd  # noqa: E402
import torch  # noqa: E402

EGO_DIR = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
SCRATCH = C.SCRATCH
MANIFESTS = {"eval139": C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
             "train": SCRATCH / "refcv6-b1-416x1024-train___v2manifest.pt"}
N_TRAIN_SAMPLE = int(__import__("os").environ.get("Q4C_N_TRAIN", "400"))
V_MIN = 2.0


def invert_rows(xy_rows: np.ndarray, t: np.ndarray, x: np.ndarray, y: np.ndarray):
    """For each row point, the log time of its projection onto the log polyline (+ distance)."""
    ax, ay = x[:-1], y[:-1]
    bx, by = x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    out_t = np.full(len(xy_rows), np.nan)
    out_d = np.full(len(xy_rows), np.nan)
    k_prev = 0
    for i, (px, py) in enumerate(xy_rows):
        lo = max(k_prev - 50, 0)
        hi = min(k_prev + 3000, len(ax))
        sx, sy, sdx, sdy, sl2 = ax[lo:hi], ay[lo:hi], dx[lo:hi], dy[lo:hi], L2[lo:hi]
        u = np.where(sl2 > 0, ((px - sx) * sdx + (py - sy) * sdy) / np.where(sl2 > 0, sl2, 1), 0.0)
        u = np.clip(u, 0.0, 1.0)
        qx, qy = sx + u * sdx, sy + u * sdy
        d = np.hypot(qx - px, qy - py)
        k = int(np.argmin(d))
        out_d[i] = d[k]
        kk = lo + k
        out_t[i] = t[kk] + u[k] * (t[kk + 1] - t[kk])
        k_prev = kk
    return out_t, out_d


def fit_clip(p_prov: np.ndarray, ns: int, ego: pd.DataFrame):
    t = ego["timestamp"].to_numpy(np.float64)
    o = np.argsort(t)
    t = t[o]
    x = ego["x"].to_numpy(np.float64)[o]
    y = ego["y"].to_numpy(np.float64)[o]
    v_log = np.hypot(ego["vx"].to_numpy(np.float64)[o], ego["vy"].to_numpy(np.float64)[o])
    unit = 1e6 if (t[-1] - t[0]) / 1e6 > 1.0 else 1.0
    tr, dist = invert_rows(p_prov[:, :2], t, x, y)
    j = np.arange(len(p_prov)) + (ns - 1)                     # RAW index of provider row
    mv = p_prov[:, 3] > V_MIN
    if mv.sum() < 20:
        return None
    A = np.stack([np.ones(mv.sum()), j[mv]], 1)
    coef, *_ = np.linalg.lstsq(A, tr[mv], rcond=None)
    a, b = coef
    resid = tr[mv] - (a + b * j[mv])
    # K3: the recovered times reproduce the logged speed
    t_fit = a + b * j
    v_rec = np.interp(t_fit, t, v_log)
    k3 = float(np.max(np.abs(v_rec[mv] - p_prov[mv, 3])))
    return {"dt_s": b / unit, "grid_start_on_label_timeline_s": (a - t[0]) / unit,
            "fit_resid_rms_ms": float(np.sqrt(np.mean(resid ** 2))) / unit * 1e3,
            "fit_resid_max_ms": float(np.max(np.abs(resid))) / unit * 1e3,
            "proj_dist_max_m": float(np.nanmax(dist[mv])),
            "K3_speed_maxabs_ms": k3, "n_moving_rows": int(mv.sum()),
            "anchor_row_true_time_s": float((a + b * (80 + ns - 1) - t[0]) / unit),
            "span_log_s": float((t[-1] - t[0]) / unit)}


def control_K2(ego: pd.DataFrame) -> dict:
    t = ego["timestamp"].to_numpy(np.float64)
    o = np.argsort(t)
    t = t[o]
    x = ego["x"].to_numpy(np.float64)[o]
    y = ego["y"].to_numpy(np.float64)[o]
    vx, vy = ego["vx"].to_numpy(np.float64)[o], ego["vy"].to_numpy(np.float64)[o]
    start, dt = t[0] + 0.037e6, 0.1e6
    tq = start + dt * np.arange(190)
    tq = tq[tq < t[-1]]
    prov = np.stack([np.interp(tq, t, x), np.interp(tq, t, y), np.zeros_like(tq),
                     np.interp(tq, t, np.hypot(vx, vy))], 1).astype(np.float32).astype(np.float64)
    f = fit_clip(prov, 1, ego)          # ns=1 -> j = provider index
    if f is None:
        return {"skipped": "not moving"}
    return {"dt_known_s": 0.1, "dt_recovered_s": f["dt_s"],
            "rel_err": f["dt_s"] / 0.1 - 1.0,
            "start_known_s": 0.037, "start_recovered_s": f["grid_start_on_label_timeline_s"],
            "start_err_ms": (f["grid_start_on_label_timeline_s"] - 0.037) * 1e3,
            "resid_rms_ms": f["fit_resid_rms_ms"]}


def main():
    if C.ram_available_gb() < 1.5:
        raise SystemExit("[audit:RAM] < 1.5 GB available even for a light job")
    avail = {p.stem for p in EGO_DIR.glob("*.parquet")}
    out = {"what": "Q4c: each cache row's true timestamp recovered from the 100 Hz egomotion log",
           "evidence_class": "MEASURED (ours)", "ego_dir_n_files": len(avail), "splits": {},
           "controls": {}}
    rng = np.random.default_rng(0)
    for split, mp in MANIFESTS.items():
        m = torch.load(str(mp), map_location="cpu", weights_only=False)
        idx = list(range(len(m["poses"])))
        if split == "train":
            idx = sorted(rng.choice(idx, size=min(N_TRAIN_SAMPLE, len(idx)), replace=False).tolist())
        rows, miss = [], 0
        for i in idx:
            cid = m["clip_id"][i]
            if cid not in avail:
                miss += 1
                continue
            ego = pd.read_parquet(EGO_DIR / f"{cid}.parquet")
            if split == "eval139" and not out["controls"]:
                out["controls"]["K2_known_linspace"] = control_K2(ego)
                print("K2", out["controls"]["K2_known_linspace"], flush=True)
            f = fit_clip(m["poses"][i].double().numpy(), int(m["n_stack"][i]), ego)
            if f is None:
                continue
            f["clip"] = C.sha12(cid)
            f["T_out"] = int(m["poses"][i].shape[0])
            rows.append(f)

        def q(k):
            v = np.asarray([r[k] for r in rows])
            return {"median": float(np.median(v)), "p05": float(np.quantile(v, .05)),
                    "p95": float(np.quantile(v, .95)), "min": float(v.min()), "max": float(v.max())}
        out["splits"][split] = {
            "n_clips_requested": len(idx), "n_clips_no_egolog": miss, "n_clips_fit": len(rows),
            "dt_s": q("dt_s"), "grid_start_on_label_timeline_s": q("grid_start_on_label_timeline_s"),
            "anchor_row_true_time_s (trainer calls it 8.0)": q("anchor_row_true_time_s"),
            "K1_fit_resid_rms_ms": q("fit_resid_rms_ms"), "K1_fit_resid_max_ms": q("fit_resid_max_ms"),
            "K3_speed_maxabs_ms": q("K3_speed_maxabs_ms"), "proj_dist_max_m": q("proj_dist_max_m"),
            "per_clip": rows}
        s = out["splits"][split]
        print(split, {k: v for k, v in s.items() if k != "per_clip"}, flush=True)
    C.write_json(__import__("os").environ.get("Q4C_OUT", "q4c_grid_vs_egolog.json"), out)


if __name__ == "__main__":
    main()
