"""Find REAL over-boundary events — by mechanisms independent of the rules under test.

The adjudication returned 60/60 on-surface, so the pack contained no positives and no candidate
can be ranked on the error that matters: MISSING a departure. This searches for positives three
ways, and reports how many each finds — including zero, which is itself a result about the corpus.

  ROUTE 1 — EGO DYNAMICS. A kerb strike or a mounting event is a VERTICAL shock: it lives in the
    100 Hz egomotion (az, vz, z, quaternion), a sensor the DAC rules never touch. The selection
    threshold is distributional and fixed BEFORE the overlap with any rule is looked at.
    ⚠️ What it MISSES, stated first: a car that rolls smoothly onto a flush driveway, a paved
    verge or a dropped kerb produces no shock. Route 1 finds MOUNTING EVENTS, not all departures.

  ROUTE 2 — WIDE MARGIN. Windows where all three candidates fire AND the corner is far enough
    past the mapped edge that no read-out defect explains it. ⛔ This route is NOT independent of
    the rules; it is reported as a margin CURVE so the dependence is visible.

  ROUTE 3 — the draw for a human sweep of stratum D (no rule fires; the only place a false-pass
    can live). ⛔ Drawn BY CLIP FIRST, then window within clip, with the cluster count printed
    beside every n — the sampling unit the last pack got wrong.

⛔ Read-only, CPU only, no GPU. Clip ids appear only as sha12.

    python find_positives.py --raw <raw dir>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dac_anatomy import DEFAULTS, THR, _import_harness

EGO_DIR = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
CAM_DIR = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
V0 = "V0 current rule (any corner, any tick, drivable < 0.50)"
P1 = "P1 road surface (drivable + paint) < 0.50"
P2 = "P2 explicit off-road (edge + hatched + sidewalk) >= 0.50"
ROBUST_Z = 6.0          # fixed before the overlap is inspected
SEED = 20260921


def rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.full_like(x, np.median(x) if len(x) else 0.0)
    v = np.lib.stride_tricks.sliding_window_view(x, w)
    m = np.median(v, axis=1)
    pad = w // 2
    return np.concatenate([np.full(pad, m[0]), m, np.full(len(x) - len(m) - pad, m[-1])])


def euler_from_quat(qx, qy, qz, qw):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return roll, pitch


def window_features(ego: dict, t0_us: float, t1_us: float) -> dict | None:
    m = (ego["t"] >= t0_us) & (ego["t"] <= t1_us)
    n = int(m.sum())
    if n < 50:                                   # < 0.5 s of samples: refuse rather than guess
        return None
    az, vz, z, t = ego["az"][m], ego["vz"][m], ego["z"][m], ego["t"][m]
    roll, pitch = ego["roll"][m], ego["pitch"][m]
    dt = np.diff(t) / 1e6
    dt[dt <= 0] = np.nan
    w = min(51, (n // 2) * 2 + 1)
    az_hf = az - rolling_median(az, w)
    zz = z - np.polyval(np.polyfit(t - t[0], z, 1), t - t[0])     # de-hill
    return {
        "n_samples": n,
        "az_hf_max": float(np.max(np.abs(az_hf))),
        "az_sd": float(np.std(az)),
        "vz_p2p": float(vz.max() - vz.min()),
        "z_detrended_p2p": float(zz.max() - zz.min()),
        "roll_rate_max_deg_s": float(np.nanmax(np.abs(np.diff(roll) / dt)) * 180 / np.pi),
        "pitch_rate_max_deg_s": float(np.nanmax(np.abs(np.diff(pitch) / dt)) * 180 / np.pi),
    }


def load_ego(cid: str) -> dict | None:
    p = EGO_DIR / f"{cid}.parquet"
    if not p.is_file():
        return None
    d = pq.ParquetFile(p).read().to_pandas()
    roll, pitch = euler_from_quat(*(d[c].to_numpy() for c in ("qx", "qy", "qz", "qw")))
    return {"t": d["timestamp"].to_numpy(np.float64), "az": d["az"].to_numpy(np.float64),
            "vz": d["vz"].to_numpy(np.float64), "z": d["z"].to_numpy(np.float64),
            "roll": roll, "pitch": pitch}


def injection_control(ego: dict, t0: float, t1: float) -> dict:
    """⛔ A search that finds nothing must prove it could have. Inject a 10 m/s^2, 60 ms vertical
    pulse — a kerb strike's order of magnitude — into a copy of the quietest window and require
    the statistic to cross the threshold."""
    m = (ego["t"] >= t0) & (ego["t"] <= t1)
    az = ego["az"][m].copy()
    k = len(az) // 2
    az[k:k + 6] += 10.0
    w = min(51, (len(az) // 2) * 2 + 1)
    return {"az_hf_max_before": float(np.max(np.abs(ego["az"][m] - rolling_median(ego["az"][m], w)))),
            "az_hf_max_after_10ms2_pulse": float(np.max(np.abs(az - rolling_median(az, w))))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--raw", required=True)
    a = ap.parse_args()
    raw = Path(a.raw)
    S1, P, SMG = _import_harness()
    rows = {(r["sha12"], r["t0"]): r for r in
            (json.loads(l) for l in (raw / "dac_windows.jsonl").read_text(encoding="utf-8").splitlines())}
    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)

    # ── ROUTE 1 ───────────────────────────────────────────────────────────────────────────
    ego_cache: dict = {}
    grid_cache: dict = {}
    out_rows, no_ego, no_span = [], 0, 0
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        cid = str(corp.clip_ids[e_i])
        t0 = int(t + corp.W - 1)
        s12 = S1.sha12(cid)
        if cid not in ego_cache:
            ego_cache[cid] = load_ego(cid)
            tsp = CAM_DIR / f"{cid}.timestamps.parquet"
            grid_cache[cid] = (SMG.episode_frame_times_us(
                pq.ParquetFile(tsp).read().to_pandas()["timestamp"].to_numpy())
                if tsp.is_file() else None)
        ego, grid = ego_cache[cid], grid_cache[cid]
        if ego is None or grid is None:
            no_ego += 1
            continue
        r0 = t0 + corp.raw_off
        ti = grid["t_img_us"]
        if r0 >= len(ti):
            no_span += 1
            continue
        t_start = float(ti[r0])
        t_end = float(ti[min(r0 + P.PROXY.n_ticks, len(ti) - 1)])
        f = window_features(ego, t_start, t_end)
        if f is None:
            no_span += 1
            continue
        k = (s12, t0)
        out_rows.append({"sha12": s12, "t0": t0, "span_s": round((t_end - t_start) / 1e6, 2),
                         **f,
                         "V0_fires": rows[k]["variants"][V0] == 0.0,
                         "P1_fires": rows[k]["variants"][P1] == 0.0,
                         "P2_fires": rows[k]["variants"][P2] == 0.0})
    az = np.array([r["az_hf_max"] for r in out_rows])
    med, mad = float(np.median(az)), float(np.median(np.abs(az - np.median(az))))
    thr = med + ROBUST_Z * 1.4826 * mad
    hits = [r for r in out_rows if r["az_hf_max"] >= thr]
    top20 = sorted(out_rows, key=lambda r: -r["az_hf_max"])[:20]
    quiet = min(out_rows, key=lambda r: r["az_hf_max"])
    qcid = next(c for c in ego_cache if S1.sha12(c) == quiet["sha12"])
    qg = grid_cache[qcid]["t_img_us"]
    ctrl = injection_control(ego_cache[qcid], float(qg[quiet["t0"] + corp.raw_off]),
                             float(qg[min(quiet["t0"] + corp.raw_off + P.PROXY.n_ticks, len(qg) - 1)]))
    ctrl["threshold"] = thr
    ctrl["pass"] = ctrl["az_hf_max_after_10ms2_pulse"] >= thr > ctrl["az_hf_max_before"]

    # ── ROUTE 2 ───────────────────────────────────────────────────────────────────────────
    kd = [json.loads(l) for l in (raw / "kerb_depth_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    per_w = defaultdict(float)
    for s in kd:
        k = (s["sha12"], s["t0"])
        per_w[k] = max(per_w[k], s["distance_to_mapped_drivable_m"])
    curve = {f"{m:.1f} m": sum(1 for v in per_w.values() if v >= m)
             for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)}

    # ── ROUTE 3 ───────────────────────────────────────────────────────────────────────────
    import random
    rng = random.Random(SEED)
    pool_d = [k for k, r in rows.items()
              if r["variants"][V0] == 1.0 and r["variants"][P1] == 1.0 and r["variants"][P2] == 1.0]
    by_clip = defaultdict(list)
    for k in pool_d:
        by_clip[k[0]].append(k)
    clips = sorted(by_clip)
    rng.shuffle(clips)
    draw3 = [rng.choice(sorted(by_clip[c])) for c in clips]          # ONE window per clip, by clip first
    rep = {
        "_what": "the search for REAL over-boundary events, by routes independent of the rules",
        "_evidence_class": "MEASURED (ours; CPU, read-only)",
        "route_1_ego_dynamics": {
            "source": "100 Hz egomotion (az, vz, z, quaternion) aligned to each window through "
                      "semantic_map_gt.episode_frame_times_us — the corpus's own v2ep grid",
            "windows_scored": len(out_rows), "windows_without_egomotion_or_grid": no_ego,
            "windows_without_a_usable_span": no_span,
            "statistic": "max |az - rolling median(az, 0.5 s)| over the window's 4 s",
            "threshold_rule": f"median + {ROBUST_Z} x 1.4826 x MAD, fixed before any overlap was read",
            "median_az_hf": round(med, 3), "mad": round(mad, 3), "threshold": round(thr, 3),
            "distribution_m_s2": {p: round(float(np.percentile(az, p)), 3)
                                  for p in (50, 90, 99, 100)},
            "n_over_threshold": len(hits),
            "hits": [{k: r[k] for k in ("sha12", "t0", "az_hf_max", "vz_p2p", "z_detrended_p2p",
                                        "pitch_rate_max_deg_s", "V0_fires", "P1_fires", "P2_fires")}
                     for r in sorted(hits, key=lambda r: -r["az_hf_max"])],
            "hits_clusters": len({r["sha12"] for r in hits}),
            "overlap_with_rules": {
                "hits_where_V0_fires": sum(1 for r in hits if r["V0_fires"]),
                "hits_where_P1_fires": sum(1 for r in hits if r["P1_fires"]),
                "hits_where_P2_fires": sum(1 for r in hits if r["P2_fires"]),
            },
            "top20_az_hf_max": [{k: r[k] for k in ("sha12", "t0", "az_hf_max", "V0_fires",
                                                   "P1_fires", "P2_fires")} for r in top20],
            "injection_control": ctrl,
            "what_it_misses": "a smooth mount onto a flush driveway, a paved verge or a dropped "
                              "kerb makes no shock; route 1 finds MOUNTING EVENTS, not departures",
        },
        "route_2_wide_margin": {
            "windows_with_an_explicitly_off_road_corner": len(per_w),
            "clusters": len({k[0] for k in per_w}),
            "count_at_margin": curve,
            "max_margin_m": round(max(per_w.values()), 3),
            "not_independent": "this route uses the rules' own firing set; reported as a curve so "
                               "the dependence is visible",
        },
        "route_3_stratum_D_sweep": {
            "population": len(pool_d), "clusters": len(by_clip),
            "windows_per_clip": dict(Counter(len(v) for v in by_clip.values())),
            "draw_rule": f"BY CLIP FIRST: shuffle clips (seed {SEED}), one window per clip",
            "drawn": len(draw3),
            "ids": [{"sha12": s, "t0": t} for s, t in draw3[:40]],
        },
    }
    (raw / "positive_search.json").write_text(json.dumps(rep, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    with (raw / "route1_ego_windows.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in sorted(out_rows, key=lambda r: -r["az_hf_max"]):
            fh.write(json.dumps(r) + "\n")
    r1 = rep["route_1_ego_dynamics"]
    print(json.dumps({k: r1[k] for k in ("windows_scored", "median_az_hf", "threshold",
                                         "distribution_m_s2", "n_over_threshold",
                                         "hits_clusters", "overlap_with_rules",
                                         "injection_control")}, indent=1))
    print("route 2:", json.dumps(rep["route_2_wide_margin"]["count_at_margin"]),
          "max margin", rep["route_2_wide_margin"]["max_margin_m"], "m")
    print("route 3 pool:", rep["route_3_stratum_D_sweep"]["population"], "windows in",
          rep["route_3_stratum_D_sweep"]["clusters"], "clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
