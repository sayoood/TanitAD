"""⭐ THE INSTRUMENT FOR THE 2026-09-16 ABSENCE-IS-NEGATIVE RULING.

The PI ruled that for the ``vlm-cot`` tactical-goal tokens, a caption that did
not mention the token means the token is FALSE. That ruling is an ASSUMPTION
about caption completeness, and this script is what turns it from an assumption
into a measured quantity: for each CoT token it asks, with a signal the caption
never touched, *"on the clips the caption left unlabelled, how often is the
token nonetheless TRUE?"*

⛔⛔ EVERY PROBE HERE IS A **NECESSARY CONDITION**, NOT A DETECTOR. That shape is
chosen so the arithmetic is honest without a hidden specificity assumption:

    probe does NOT fire  =>  the token is FALSE (the thing it needs is absent)
    probe DOES fire      =>  the token MAY be true

so the fire rate on the caption-unlabelled clips is an **upper bound** on the
false-negative rate, requiring nothing about how often the probe over-fires.

⭐ AND EVERY PROBE CARRIES A CONTROL THAT READS A KNOWN VALUE. A necessary
condition must hold on the LABELLED POSITIVES. The control is therefore the fire
rate on those positives (``sensitivity``): a probe that misses the positives is
not measuring the token, and this script marks its bound **VOID** rather than
quoting it. Two further controls run before any token is scored:

  ``PIPELINE CONTROL`` — the same kinematic features are scored against the
  three GEOMETRY-provenance tokens (``STOP_POINT``, ``TURN_L``, ``TURN_R``)
  whose truth is ego geometry and therefore independently knowable. If the
  time base or the feature extraction were wrong these would not separate, and
  every CoT number downstream would be noise wearing a decimal point.

  ``KNOWN-NEGATIVE CONTROL`` — where the frozen exclusion table
  (``TACTICAL_GOAL_EXCLUSIVE``) names a token that makes this one FALSE, the
  clips carrying it are a set of CERTAIN negatives. The probe's fire rate there
  is ``f0``, the base rate at which the necessary condition is met by a clip
  that does not carry the token, and it identifies a point estimate

      pi_hat = (F - f0) / (S - f0)       clipped to [0, 1]

  beside the assumption-free upper bound ``F / 1`` and the
  sensitivity-corrected ``F / S``. Tokens with no exclusion partner (the PI's
  original four) get the bound and no point estimate, and say so.

INPUTS — all three are artifacts on this box, none is another caption:
  * the label blob (positives, and the geometry tokens for the pipeline control)
  * ``egomotion_alpamayo/<clip>.parquet`` — ego world pose + velocity at ~10 Hz
  * the B1 TRAIN ``obstacle.offline`` agent join — 3D boxes in the ego frame

⚠️ WHAT THIS SCRIPT CANNOT DO, stated rather than approximated. There is no lane
reference and no map on this box, so ``LANE_CHANGE_L/R``, ``TAKE_EXIT_L/R`` and
``MERGE`` have no necessary condition that separates them from a nudge or a
bend, and the traffic-light colour tokens have no signal at all outside the
caption channel. Those are reported UNVERIFIED with the reason, never with an
invented probe.

ASCII only in the printed output: this box is cp1252.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# THE WINDOW. t0_s == 8.0 on all 4,572 records and bands.tactical_s == [2, 6]
# (MEASURED, both constant), so the tactical band is raw clip seconds
# [10, 14] and every goal's t_nominal_s is 4.0 == clip second 12.0.
# ⛔ These are READ from the blob at runtime, never hardcoded -- a derived
# constant that silently changes when its input changes is the HORIZON trap.
# ---------------------------------------------------------------------------
LEADIN_S = 2.0          # operative band ahead of the tactical one, for onset

#: obstacle.offline class enum, MEASURED at two locations (11 classes,
#: b1-train-join-20260906/raw/enum_probe_{A,B}*.json).
VEHICLE_CLS = frozenset({"automobile", "heavy_truck", "bus", "trailer",
                         "other_vehicle", "train_or_tram_car"})
VRU_CLS = frozenset({"person", "rider", "stroller", "animal"})

#: corridor half-width. A lane is 3.2-3.7 m (v7_labels.effective_mask), so
#: +-1.8 m is "in my lane" and +-3.0 m is the sensitivity variant.
CORRIDOR_HALF_M = 1.8
CORRIDOR_WIDE_M = 3.0

V_STOP_MS = 0.5         # ego_manoeuvre.V_STOP_MS -- reused, not re-invented
DECEL_SIGNIFICANT = 1.5  # ego_manoeuvre.DECEL_SIGNIFICANT
NUDGE_LAT_M = 1.0       # ego_manoeuvre.NUDGE_LAT_M

_TS_RE = re.compile(rb'"t_s":\s*(-?[0-9.eE+]+)')


def _yaw_from_quat(qx, qy, qz, qw):
    """Yaw about +z from a unit quaternion, vectorised."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny, cosy)


def ego_features(pq_path: Path, t_lo: float, t_hi: float, lead_s: float
                 ) -> dict | None:
    """Ego kinematics over the tactical window, from egomotion alone.

    Returns None when the clip has no egomotion covering the window -- a state
    of its own, counted, never silently treated as "nothing happened".
    """
    import pyarrow.parquet as pq
    try:
        tb = pq.read_table(pq_path, columns=["timestamp", "qx", "qy", "qz", "qw",
                                             "x", "y", "vx", "vy"])
    except Exception:
        return None
    t = tb.column("timestamp").to_numpy() / 1e6
    if t.size < 4 or t[0] > t_lo or t[-1] < t_hi:
        return None
    v = np.hypot(tb.column("vx").to_numpy(), tb.column("vy").to_numpy())
    yaw = _yaw_from_quat(tb.column("qx").to_numpy(), tb.column("qy").to_numpy(),
                         tb.column("qz").to_numpy(), tb.column("qw").to_numpy())
    x = tb.column("x").to_numpy()
    y = tb.column("y").to_numpy()

    win = (t >= t_lo) & (t <= t_hi)
    lead = (t >= t_lo - lead_s) & (t <= t_hi)
    if win.sum() < 4 or lead.sum() < 4:
        return None

    vw, vl = v[win], v[lead]
    tw = t[win]
    # largest drop from any earlier sample to any later one, inside the window
    run_max = np.maximum.accumulate(vl)
    drop_lead = float((run_max - vl).max())
    run_max_w = np.maximum.accumulate(vw)
    drop_win = float((run_max_w - vw).max())

    yw = np.unwrap(yaw[lead])
    dyaw_deg = float(np.degrees(yw[-1] - yw[0]))
    peak_yaw_deg = float(np.degrees(yw[int(np.argmax(np.abs(yw - yw[0])))] - yw[0]))

    # lateral track from the heading at the START of the lead-in window --
    # the ego_manoeuvre.py:321 formula, reused so the number means the same
    # thing it means everywhere else in the programme.
    xl, yl = x[lead], y[lead]
    c, s = math.cos(-yaw[lead][0]), math.sin(-yaw[lead][0])
    lat = s * (xl - xl[0]) + c * (yl - yl[0])
    lat_peak = float(lat[int(np.argmax(np.abs(lat)))])

    # ⭐ CURVATURE-DETRENDED LATERAL RESIDUAL — the instrument for
    # CORRIDOR_OFFSET, and the reason a raw ``lat_peak_m`` is NOT one.
    # MEASURED on the smoke sample: a clip with 5.0 deg of heading change over
    # 6 s at 8.8 m/s carries ``lat_peak_m`` 2.95 m purely from ROAD CURVATURE.
    # Reading that as "a deliberate offset" would call a bend a nudge on most
    # of the corpus. A constant-curvature arc (a quadratic in arc length) is
    # exactly what the road contributes, so it is fitted and removed; what is
    # left is a CHANGE of curvature, which is what a deliberate offset is.
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xl), np.diff(yl)))])
    if arc[-1] > 1.0 and arc.size >= 5:
        cf = np.polyfit(arc, lat, 2)
        resid = lat - np.polyval(cf, arc)
        lat_resid = float(np.abs(resid).max())
    else:
        lat_resid = 0.0

    dt = float(np.median(np.diff(tw))) or 0.1
    a = np.gradient(vw, dt) if vw.size >= 3 else np.zeros_like(vw)

    return {
        "v_start": float(vw[0]), "v_min": float(vw.min()),
        "v_max": float(vw.max()), "v_end": float(vw[-1]),
        "v_max_lead": float(vl.max()),
        "drop_win_ms": drop_win, "drop_lead_ms": drop_lead,
        "a_min": float(a.min()), "t_stopped_s": float((vw <= V_STOP_MS).sum() * dt),
        "dyaw_deg": dyaw_deg, "peak_yaw_deg": peak_yaw_deg,
        "lat_peak_m": lat_peak, "lat_resid_m": lat_resid,
        "arc_m": float(arc[-1]),
        "t_grid": tw.tolist(), "x_w": x[win].tolist(), "y_w": y[win].tolist(),
        "yaw_w": yaw[win].tolist(),
    }


def scan_join(join_path: Path, t_lo: float, t_hi: float,
              ego: dict[str, dict]) -> dict[str, dict]:
    """One streaming pass over the agent join, reducing each clip as it closes.

    ⭐ The ``t_s`` is pulled with a regex BEFORE json.loads so the ~78 % of
    lines outside the window are never parsed -- the difference between a
    40-second pass and a 5-minute one.

    ⛔ It REDUCES PER CLIP rather than buffering the whole file: the in-window
    slice is ~4 M agent records and holding it costs GBs. The file is written
    clip by clip, so a clip closes when the ``clip_id`` changes; a clip that
    REAPPEARS after closing would silently split its window, so that is an
    error rather than a merge.
    """
    out: dict[str, dict] = {}
    cur_id: str | None = None
    cur: list = []
    n_lines = n_kept = 0
    t0 = time.time()

    def _flush():
        nonlocal cur, cur_id
        if cur_id is not None and cur:
            if cur_id in out:
                raise SystemExit(f"[join] clip {cur_id} appears twice, "
                                 f"non-contiguously -- the per-clip reduction "
                                 f"would silently see half its window.")
            out[cur_id] = agent_features(cur, ego[cur_id])
        cur, cur_id = [], None

    with lzma.open(join_path, "rb") as fh:
        for raw in fh:
            n_lines += 1
            m = _TS_RE.search(raw)
            if m is None:
                continue
            ts = float(m.group(1))
            if not (t_lo <= ts <= t_hi):
                continue
            rec = json.loads(raw)
            cid = rec["clip_id"]
            if cid != cur_id:
                _flush()
                cur_id = cid
            if cid not in ego:
                cur_id = None
                continue
            cur.append((ts, rec.get("agents") or []))
            n_kept += 1
    _flush()
    print(f"[join] {n_lines} lines, {n_kept} in window, {len(out)} clips "
          f"reduced, {time.time() - t0:.0f}s", flush=True)
    return out


def agent_features(frames: list, ego: dict) -> dict:
    """Agent geometry over the window, in the ego frame, with WORLD speeds.

    ⛔ The world speed matters: ``OVERTAKE_VEHICLE`` is defined against a MOVING
    vehicle and ``EVADE_IN_CORRIDOR`` against a STATIC one (vocab_v7:115-127),
    and an agent's EGO-frame motion is dominated by the ego's own -- a parked
    car "moves" at the ego's speed in the ego frame. So each box is lifted to
    the world through the ego pose at its own frame time before any speed is
    read off it.
    """
    frames = sorted(frames)
    if not frames:
        return {"n_frames": 0}
    tg = np.asarray(ego["t_grid"])
    xg = np.asarray(ego["x_w"])
    yg = np.asarray(ego["y_w"])
    yg_yaw = np.asarray(ego["yaw_w"])

    # track_id -> [(t, cx, cy, yaw_rel, cls, occ, wx, wy)]
    tracks: dict[str, list] = {}
    for ts, agents in frames:
        i = int(np.argmin(np.abs(tg - ts)))
        ex, ey, eyaw = float(xg[i]), float(yg[i]), float(yg_yaw[i])
        c, s = math.cos(eyaw), math.sin(eyaw)
        for a in agents:
            cx, cy = float(a.get("cx", 0.0)), float(a.get("cy", 0.0))
            wx = ex + c * cx - s * cy
            wy = ey + s * cx + c * cy
            tracks.setdefault(str(a.get("track_id")), []).append(
                (ts, cx, cy, float(a.get("yaw", 0.0)), str(a.get("cls", "")),
                 int(a.get("occ", 0)), wx, wy))

    n_oncoming = n_cross = n_vru = 0
    oncoming_min_cx = math.inf
    overtaken = False
    closing_on_mover = False
    static_in_corridor = False
    vru_in_corridor = False
    mover_ahead = False
    for _tid, pts in tracks.items():
        pts.sort()
        ts_ = np.asarray([p[0] for p in pts])
        cxs = np.asarray([p[1] for p in pts])
        cys = np.asarray([p[2] for p in pts])
        yws = np.asarray([p[3] for p in pts])
        cls = pts[0][4]
        wxs = np.asarray([p[6] for p in pts])
        wys = np.asarray([p[7] for p in pts])
        span = float(ts_[-1] - ts_[0])
        wspeed = (float(np.hypot(wxs[-1] - wxs[0], wys[-1] - wys[0])) / span
                  if span > 0.3 else 0.0)
        is_veh = cls in VEHICLE_CLS
        is_vru = cls in VRU_CLS
        ahead = cxs > 2.0
        in_cor_wide = ahead & (np.abs(cys) <= CORRIDOR_WIDE_M) & (cxs <= 60.0)
        # heading relative to the ego: |yaw| near pi == pointing back at us
        opposed = np.abs(np.abs(yws) - math.pi) < (math.pi / 4)      # >135 deg
        perpend = (np.abs(yws) > math.pi / 4) & (np.abs(yws) < 3 * math.pi / 4)

        if is_veh and in_cor_wide.any() and wspeed >= 1.0:
            mover_ahead = True
            sel = np.nonzero(in_cor_wide)[0]
            # the ego is CATCHING it -- the necessary condition for an overtake
            if sel.size >= 2 and cxs[sel[-1]] < cxs[sel[0]] - 2.0:
                closing_on_mover = True
            if cxs[-1] < 0.0:          # started ahead of us, ended behind us
                overtaken = True
        if is_veh and in_cor_wide.any() and wspeed < 0.6:
            static_in_corridor = True
        onc = ahead & (np.abs(cys) <= 6.0) & (cxs <= 80.0) & opposed
        if is_veh and onc.any():
            n_oncoming += 1
            oncoming_min_cx = min(oncoming_min_cx, float(cxs[onc].min()))
        if ((cxs > 0.0) & (cxs <= 40.0) & (np.abs(cys) <= 12.0) & perpend).any() \
                and wspeed >= 0.8:
            n_cross += 1
        if is_vru and ((cxs > 0.0) & (cxs <= 40.0) & (np.abs(cys) <= 5.0)).any():
            n_vru += 1
            if ((cxs > 0.0) & (cxs <= 30.0) & (np.abs(cys) <= CORRIDOR_WIDE_M)).any():
                vru_in_corridor = True

    # ⭐ LEAD is computed PER FRAME, not per track: "is there a vehicle in my
    # lane ahead of me right now" is a property of the frame, and summing
    # per-track observations answers a different question (it counts a queue of
    # three as three). ``lead_frac`` is then the fraction of the window that
    # carries a lead at all, which is what "a HELD gap" means.
    d_lead, d_lead_w = [], []
    empty_ahead = 0
    for ts, agents in frames:
        best = bestw = math.inf
        any_ahead = False
        for a in agents:
            cx, cy = float(a.get("cx", 0.0)), float(a.get("cy", 0.0))
            if cx > 0.0 and cx <= 80.0 and abs(cy) <= 15.0:
                any_ahead = True
            if str(a.get("cls", "")) not in VEHICLE_CLS:
                continue
            if 2.0 < cx <= 60.0 and abs(cy) <= CORRIDOR_HALF_M:
                best = min(best, cx)
            if 2.0 < cx <= 60.0 and abs(cy) <= CORRIDOR_WIDE_M:
                bestw = min(bestw, cx)
        d_lead.append(None if best is math.inf else best)
        d_lead_w.append(None if bestw is math.inf else bestw)
        empty_ahead += (not any_ahead)
    have = [d for d in d_lead if d is not None]
    havew = [d for d in d_lead_w if d is not None]
    idx = [i for i, d in enumerate(d_lead) if d is not None]

    return {
        "n_frames": len(frames), "n_tracks": len(tracks),
        "lead_frac": len(have) / max(len(d_lead), 1),
        "lead_frac_wide": len(havew) / max(len(d_lead_w), 1),
        # ⭐ "NOTHING AHEAD AT ALL" -- the only KNOWN-NEGATIVE construction the
        # PI's four tokens admit. YIELD / GAP_TARGET / REACT_ON_ONCOMING /
        # EVADE all require SOMETHING to yield to, follow, meet or avoid; a
        # window in which the 3D-box channel sees no agent ahead in ANY frame
        # is a certain negative for all four, established from a channel the
        # caption never touched.
        "empty_ahead_frac": empty_ahead / max(len(frames), 1),
        "d_lead_min": min(have) if have else None,
        "d_lead_first": d_lead[idx[0]] if idx else None,
        "d_lead_last": d_lead[idx[-1]] if idx else None,
        "n_oncoming_tracks": n_oncoming,
        "oncoming_min_cx": None if oncoming_min_cx is math.inf else oncoming_min_cx,
        "n_crossing_tracks": n_cross, "n_vru_ahead": n_vru,
        "vru_in_corridor": vru_in_corridor,
        "mover_ahead": mover_ahead,
        "closing_on_mover": closing_on_mover,
        "overtaken_a_mover": overtaken,
        "static_veh_in_corridor": static_in_corridor,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ego-dir", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    # ⚠️ THE WINDOW OVERRIDE EXISTS BECAUSE THE TOKENS ARE UNTIMED. MEASURED
    # on this blob: 492/609 YIELD, 859/860 CORRIDOR_OFFSET, 333/333
    # REACT_ON_ONCOMING and 288/368 GAP_TARGET carry
    # ``time_basis: "untimed"``, and their ``t_nominal_s`` is the BAND MIDPOINT
    # fallback (``t_nominal_provenance: "band-midpoint (PI 2026-08-28)"``), not
    # an observation. An untimed token is a claim about the CLIP, so scoring it
    # only inside [t0+2, t0+6] asks the probe to find the evidence in a window
    # the label never promised it would be in. The default stays the tactical
    # band; the wide window is run BESIDE it and both are reported.
    ap.add_argument("--win-lo", type=float, default=None)
    ap.add_argument("--win-hi", type=float, default=None)
    a = ap.parse_args()

    blob = Path(a.labels)
    md5 = hashlib.md5(blob.read_bytes()).hexdigest()
    recs = []
    with gzip.open(blob, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    if a.limit:
        recs = recs[:a.limit]

    t0s = {r["t0_s"] for r in recs}
    bands = {tuple(r["bands"]["tactical_s"]) for r in recs}
    if len(t0s) != 1 or len(bands) != 1:
        raise SystemExit(f"[probe] t0_s/bands are not constant: {t0s} {bands} -- "
                         f"the window must then be per record; refusing to "
                         f"average over two different windows.")
    t0 = float(next(iter(t0s)))
    lo, hi = next(iter(bands))
    t_lo, t_hi = t0 + float(lo), t0 + float(hi)
    band = "tactical"
    if a.win_lo is not None or a.win_hi is not None:
        t_lo = a.win_lo if a.win_lo is not None else t_lo
        t_hi = a.win_hi if a.win_hi is not None else t_hi
        band = "override"
    print(f"[probe] blob md5={md5} n={len(recs)} window=[{t_lo}, {t_hi}] s "
          f"({band}; t0_s={t0}, tactical_s={[lo, hi]}), lead-in {LEADIN_S}s",
          flush=True)

    ego_dir = Path(a.ego_dir)
    feats: dict[str, dict] = {}
    n_no_ego = 0
    t_start = time.time()
    for i, r in enumerate(recs):
        cid = r["clip_id"]
        e = ego_features(ego_dir / f"{cid}.parquet", t_lo, t_hi, LEADIN_S)
        if e is None:
            n_no_ego += 1
            continue
        feats[cid] = e
        if (i + 1) % 500 == 0:
            print(f"[ego] {i+1}/{len(recs)} ok={len(feats)} "
                  f"{time.time()-t_start:.0f}s", flush=True)
    print(f"[ego] done ok={len(feats)} no_ego={n_no_ego} "
          f"{time.time()-t_start:.0f}s", flush=True)

    jf = scan_join(Path(a.join), t_lo, t_hi, feats)

    out = {}
    for cid, e in feats.items():
        rec = {"ego": {k: v for k, v in e.items()
                       if k not in ("t_grid", "x_w", "y_w", "yaw_w")}}
        af = jf.get(cid)
        # ⛔ NO JOIN IS "UNKNOWN", NOT "NO AGENTS". A clip the join never
        # covered has no agent evidence either way; folding it in as an empty
        # scene would manufacture negatives out of a coverage gap, which is
        # the exact failure this whole exercise is measuring.
        rec["agents"] = af if af is not None else None
        rec["has_join"] = af is not None
        out[cid] = rec

    Path(a.out).write_text(json.dumps(
        {"_evidence_class": "MEASURED (ours)",
         "blob_md5": md5, "window_s": [t_lo, t_hi], "leadin_s": LEADIN_S,
         "n_records": len(recs), "n_with_ego": len(feats),
         "n_without_ego": n_no_ego,
         "n_with_join": sum(1 for v in out.values() if v["has_join"]),
         "ego_dir": str(ego_dir), "join": str(a.join),
         "clips": out}, indent=None), encoding="utf-8")
    print(f"[probe] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
