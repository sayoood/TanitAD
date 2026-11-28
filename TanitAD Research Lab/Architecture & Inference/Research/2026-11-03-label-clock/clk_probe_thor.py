#!/usr/bin/env python
"""r9clk -- label clock diagnosis, Thor CPU probe (READ-ONLY; no label file, join or trainer is touched).

QUESTION: do the agent / 3-D box labels the refcv8 trainer pairs with an image row describe the scene
~0.2 s BEFORE that image (LAG), or do two clock conventions disagree while the labels are aligned (CONVENTION)?

Run on Thor:  OMP_NUM_THREADS=1 /home/nvidia/venvs/tanitad-train/bin/python clk_probe_thor.py --n-clips 200 --out OUT.json

STAGES
  A  clock trace. Per clip: the join's t_s, the CAMERA-timestamp grid T_i (the image time, from the corpus's own
     timestamps parquet via the physicalai episode_grid rule), the clip-clock sidecar, and the egomotion log's own first
     timestamp ts0. G2's number is recomputed with G2's own formula FIRST (probe-defect elimination), then explained.
  C0 pose <-> grid identity: the v2ep payload poses[i] must equal egomotion interpolated at T_i (and NOT at T_{i+k}).
  B  static-label scan. For STATIC raw tracks (world radius95 <= R) the label (cx, cy) in the ego frame of record i is
     put in the world with the ego pose at (T_i + delta); error e_i(delta) = |W_i(delta) - w_track|. delta* = argmin is
     the labels' time relative to the image. KNOWN-ANSWER controls: labels shifted by a known +0.1 s in software, and a
     deliberate mis-key by +/-2 frames. Also a model-free dispersion statistic.
  C  the 3-D join is the 2-D join annotated: t_s / frame / cx / cy identical line for line.
"""
import argparse
import hashlib
import io
import json
import lzma
import math
import os
import tarfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

D = "/home/nvidia/data"
JOIN2D = f"{D}/joins/b1_train_plus_eval_agents.jsonl.xz"
JOIN3D = f"{D}/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
SIDE = f"{D}/refcv8_clip_clock_sidecar_ts33.jsonl"
EGO = f"{D}/obstacle/egomotion_alpamayo"
OBS = f"{D}/obstacle/obstacle_offline_b1train"
TAR = f"{D}/b1-bundle/timestamps/timestamps.tar"
V2EP = f"{D}/refcv8-b1-416x1024-train-4572"
TARGET_HZ = 10.0


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def sid_of(cid):  # v2_dataset.stable_episode_id
    return int.from_bytes(hashlib.blake2b(cid.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def yaw_q(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class Ego:
    """Raw-clock egomotion (timestamp/1e6, NOT zero-referenced) -- the join builder's EgoTrack convention."""

    def __init__(self, df):
        ts = df["timestamp"].to_numpy(np.float64)
        o = np.argsort(ts)
        self.ts0_us = float(ts[o][0])
        self.t = ts[o] / 1e6
        self.x = df["x"].to_numpy(np.float64)[o]
        self.y = df["y"].to_numpy(np.float64)[o]
        self.yaw = np.unwrap(yaw_q(*(df[c].to_numpy(np.float64)[o] for c in ("qx", "qy", "qz", "qw"))))

    def at(self, ts):
        ts = np.asarray(ts, np.float64)
        return (np.interp(ts, self.t, self.x), np.interp(ts, self.t, self.y), np.interp(ts, self.t, self.yaw))


def episode_grid(t_frames):  # taniteval/tools/build_lead_block_b1.episode_grid (physicalai.build_episode 711-718)
    t_frames = np.asarray(t_frames, np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), 4)
    return np.linspace(t_frames[0], t_frames[-1], n_target), unit, n_target


def ego_to_world(ex, ey, eyaw, cx, cy):  # inverse of bev_raster.ego_frame_agents
    c, s = np.cos(eyaw), np.sin(eyaw)
    return ex + cx * c - cy * s, ey + cx * s + cy * c


def world_to_ego(ex, ey, eyaw, wx, wy):
    c, s = np.cos(eyaw), np.sin(eyaw)
    dx, dy = wx - ex, wy - ey
    return dx * c + dy * s, -dx * s + dy * c


def read_join(path, n_clips, need):
    """First n_clips (that satisfy need(cid)) of a join, in file order -> {cid: [records]}."""
    out, order, skipped = {}, [], set()
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cid = r["clip_id"]
            if cid in skipped:
                continue
            if cid not in out:
                if len(out) >= n_clips:
                    break
                if not need(cid):
                    skipped.add(cid)
                    continue
                out[cid] = []
                order.append(cid)
            out[cid].append(r)
    return out, order


def parabola_min(xs, ys):
    """Vertex of the parabola through the grid minimum and its two neighbours (uniform grid)."""
    i = int(np.argmin(ys))
    if 0 < i < len(xs) - 1:
        h = float(xs[i + 1] - xs[i])
        den = float(ys[i - 1] - 2 * ys[i] + ys[i + 1])
        if den > 0:
            return float(xs[i] - 0.5 * h * float(ys[i + 1] - ys[i - 1]) / den)
    return float(xs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--radius95", type=float, default=0.30)
    ap.add_argument("--min-ego-speed", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n3d-lines", type=int, default=60000)
    a = ap.parse_args()

    side = {}
    for l in open(SIDE):
        r = json.loads(l)
        side[int(r["sid"])] = (float(r["grid_start_s"]), float(r["dt_s"]))
    tf = tarfile.open(TAR)
    tnames = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.name.endswith(".parquet")}

    def need(cid):
        return (sid_of(cid) in side and f"{cid}.timestamps.parquet" in tnames
                and os.path.isfile(f"{EGO}/{cid}.parquet") and os.path.isfile(f"{OBS}/{cid}.parquet")
                and os.path.isfile(f"{V2EP}/{cid}.v2ep.pt"))

    join, order = read_join(JOIN2D, a.n_clips, need)
    print("clips", len(order), "records", sum(len(v) for v in join.values()), flush=True)
    import torch
    torch.set_num_threads(1)

    deltas = np.round(np.arange(-0.40, 0.2001, 0.01), 4)
    res = dict(n_clips=len(order), radius95=a.radius95, min_ego_speed=a.min_ego_speed)
    A_rows, C0_rows = [], []
    g2_all, d1_all, ds_all = [], [], []
    inst = []  # static label instances: dicts of arrays per clip

    for ci, cid in enumerate(order):
        recs = join[cid]
        ego = Ego(pd.read_parquet(f"{EGO}/{cid}.parquet"))
        tdf = pd.read_parquet(io.BytesIO(tf.extractfile(tnames[f"{cid}.timestamps.parquet"]).read()))
        tcol = next(c for c in tdf.columns if "time" in c.lower())
        tq, unit, n_target = episode_grid(tdf[tcol].to_numpy(np.float64))
        T = tq / unit                                    # IMAGE TIME of raw frame i, raw clock [s]
        g0, dt = side[sid_of(cid)]
        fr = np.array([r["frame"] for r in recs])
        fi = np.array([r["frame_idx"] for r in recs])
        ts = np.array([r["t_s"] for r in recs])
        ok = fr < n_target
        fr, fi, ts = fr[ok], fi[ok], ts[ok]
        # ---- Stage A -------------------------------------------------------------------------------------
        g2 = ts - (g0 + fr * dt)                         # G2's formula, verbatim
        d1 = ts - T[fr]                                  # join t_s vs the CAMERA grid (same raw clock)
        sc_raw = ego.t[0] + g0 + fr * dt                 # sidecar clock moved onto the raw clock
        ds = sc_raw - T[fr]
        g2_all += list(g2)
        d1_all += list(d1)
        ds_all += list(ds)
        A_rows.append(dict(clip=sha12(cid), n=int(len(fr)), ts0_s=ego.t[0], n_target=int(n_target),
                           frame_minus_frame_idx=int(np.median(fr - fi)), g2_med=float(np.median(g2)),
                           join_vs_camgrid_med=float(np.median(d1)), join_vs_camgrid_absmax=float(np.max(np.abs(d1))),
                           sidecar_rawclock_vs_camgrid_med=float(np.median(ds)), dt=dt, g0=g0,
                           g2_minus_ts0=float(np.median(g2) - ego.t[0])))
        # ---- C0: payload poses vs egomotion at T_{i+k} ------------------------------------------------------
        d = torch.load(f"{V2EP}/{cid}.v2ep.pt", map_location="cpu", weights_only=False, mmap=True)
        P = d["poses"].float().numpy().astype(np.float64)
        n_stack = int(d["n_stack"])
        row = dict(clip=sha12(cid), n_poses=int(len(P)), n_target=int(n_target), n_stack=n_stack)
        for k in range(-3, 4):
            idx = np.arange(max(0, -k), min(len(P), n_target - k))
            ex, ey, _ = ego.at(T[idx + k])
            row[f"k{k:+d}"] = float(np.median(np.hypot(ex - P[idx, 0], ey - P[idx, 1])))
        C0_rows.append(row)
        # ---- Stage B: static raw tracks -------------------------------------------------------------------
        obs = pd.read_parquet(f"{OBS}/{cid}.parquet")
        to = obs["timestamp_us"].to_numpy(np.float64) / 1e6
        ox, oy, oyaw = ego.at(to)
        wx, wy = ego_to_world(ox, oy, oyaw, obs["center_x"].to_numpy(np.float64), obs["center_y"].to_numpy(np.float64))
        tid = obs["track_id"].astype(str).to_numpy()
        spd = np.hypot(np.gradient(ego.x, ego.t), np.gradient(ego.y, ego.t))
        static = {}
        for t_ in np.unique(tid):
            m = tid == t_
            if m.sum() < 25:
                continue
            tt = to[m]
            if tt.max() - tt.min() < 2.5:
                continue
            mx, my = np.median(wx[m]), np.median(wy[m])
            rad = np.hypot(wx[m] - mx, wy[m] - my)
            if np.quantile(rad, 0.95) > a.radius95:
                continue
            sm = (ego.t >= tt.min()) & (ego.t <= tt.max())
            if sm.sum() == 0 or spd[sm].mean() < a.min_ego_speed:
                continue
            static[t_] = (mx, my)
        if not static:
            continue
        rows = []
        for r, f_ok in zip(recs, ok):
            if not f_ok:
                continue
            for ag in r["agents"]:
                if ag["track_id"] in static:
                    rows.append((r["frame"], ag["cx"], ag["cy"], static[ag["track_id"]][0], static[ag["track_id"]][1],
                                 hash(ag["track_id"]) & 0x7FFFFFFF))
        if rows:
            R = np.array(rows, np.float64)
            inst.append(dict(clip=cid, frame=R[:, 0].astype(int), cx=R[:, 1], cy=R[:, 2], wtx=R[:, 3], wty=R[:, 4],
                             trk=R[:, 5].astype(int), T=T, ego=ego, dt=dt, n_static_tracks=len(static)))
        if (ci + 1) % 50 == 0:
            print("  clip", ci + 1, "static-instances so far", sum(len(i["frame"]) for i in inst), flush=True)

    # ---------------- Stage A summary ------------------------------------------------------------------------
    def q(v):
        v = np.asarray(v)
        return dict(min=float(v.min()), p05=float(np.quantile(v, .05)), med=float(np.median(v)),
                    p95=float(np.quantile(v, .95)), max=float(v.max()), n=int(len(v)))

    res["A"] = dict(
        G2_formula_join_ts_minus_sidecar_clock=q(g2_all),
        join_ts_minus_CAMERA_grid=q(d1_all),
        sidecar_clock_on_raw_clock_minus_CAMERA_grid=q(ds_all),
        ego_log_first_timestamp_s=q([r["ts0_s"] for r in A_rows]),
        per_clip_g2_minus_ts0=q([r["g2_minus_ts0"] for r in A_rows]),
        frame_minus_frame_idx_counts={str(k): int(v) for k, v in zip(*np.unique(
            [r["frame_minus_frame_idx"] for r in A_rows], return_counts=True))},
        corr_g2_med_vs_ts0=float(np.corrcoef([r["g2_med"] for r in A_rows], [r["ts0_s"] for r in A_rows])[0, 1]),
        two_dt_median=float(2 * np.median([r["dt"] for r in A_rows])),
        n_clips=len(A_rows))
    res["A_rows"] = A_rows
    res["C0"] = dict(rows=C0_rows, median_by_k={k: float(np.median([r[k] for r in C0_rows]))
                                                for k in C0_rows[0] if k.startswith("k")},
                     n_nstack3=int(sum(r["n_stack"] == 3 for r in C0_rows)),
                     n_poses_eq_ntarget=int(sum(r["n_poses"] == r["n_target"] for r in C0_rows)))

    # ---------------- Stage B scan -------------------------------------------------------------------------
    def scan(transform, assumed_shift_frames=0):
        """mean |e| (m) per delta, per-clip and pooled. transform(inst)-> (cx, cy) used as the label."""
        per_clip = []
        for it in inst:
            cx, cy = transform(it)
            ego, T, fr = it["ego"], it["T"], it["frame"]
            Tk = T[np.clip(fr + assumed_shift_frames, 0, len(T) - 1)]
            e, disp = [], []
            for dl in deltas:
                ex, ey, eyaw = ego.at(Tk + dl)
                wx, wy = ego_to_world(ex, ey, eyaw, cx, cy)
                e.append(np.hypot(wx - it["wtx"], wy - it["wty"]).mean())
                # model-free dispersion: per track RMS about its own mean world position
                s2, n2 = 0.0, 0
                for t_ in np.unique(it["trk"]):
                    mm = it["trk"] == t_
                    if mm.sum() >= 5:
                        s2 += ((wx[mm] - wx[mm].mean()) ** 2 + (wy[mm] - wy[mm].mean()) ** 2).sum()
                        n2 += int(mm.sum())
                disp.append(math.sqrt(s2 / n2) if n2 else float("nan"))
            per_clip.append((np.array(e), np.array(disp), len(fr)))
        return per_clip

    def summarize(per_clip, name, rng):
        E = np.array([p[0] for p in per_clip])
        Dp = np.array([p[1] for p in per_clip])
        W = np.array([p[2] for p in per_clip], float)
        pooled = (E * W[:, None]).sum(0) / W.sum()
        dstar = parabola_min(deltas, pooled)
        boots = []
        for _ in range(1000):
            ix = rng.integers(0, len(E), len(E))
            pb = (E[ix] * W[ix, None]).sum(0) / W[ix].sum()
            boots.append(parabola_min(deltas, pb))
        i0 = int(np.argmin(np.abs(deltas - 0.0)))
        i2 = int(np.argmin(np.abs(deltas + 0.20)))
        dm = np.nanmean(Dp, 0)
        return dict(name=name, delta_star_s=dstar, ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                    mean_abs_err_m_at_0=float(pooled[i0]), mean_abs_err_m_at_minus_0p20=float(pooled[i2]),
                    min_mean_abs_err_m=float(pooled.min()), disp_rms_m_at_0=float(dm[i0]),
                    disp_delta_star_s=parabola_min(deltas, np.nan_to_num(dm, nan=1e9)),
                    curve={f"{d_:+.2f}": float(v) for d_, v in zip(deltas, pooled) if abs(round(d_ * 100) % 5) < 1e-9},
                    n_clips=len(E), n_instances=int(W.sum()), n_tracks=int(sum(i["n_static_tracks"] for i in inst)))

    rng = np.random.default_rng(0)
    res["B_n_clips_with_static"] = len(inst)
    res["B"] = {}
    ident = lambda it: (it["cx"], it["cy"])
    res["B"]["TRUE_labels"] = summarize(scan(ident), "TRUE (join labels as banked)", rng)

    # KNOWN-ANSWER 1: labels re-anchored 0.1 s LATER than the image (software shift) -> delta* must be +0.1
    def shifted(sh):
        def f(it):
            ego, T, fr = it["ego"], it["T"], it["frame"]
            e0 = ego.at(T[fr])
            wx, wy = ego_to_world(*e0, it["cx"], it["cy"])
            e1 = ego.at(T[fr] + sh)
            return world_to_ego(*e1, wx, wy)
        return f

    for sh in (0.10, -0.10, -0.20):
        res["B"][f"CONTROL_shift_{sh:+.2f}s"] = summarize(scan(shifted(sh)), f"known shift {sh:+.2f} s", rng)
    # KNOWN-ANSWER 2: deliberate mis-key (assumed image time = T_{i+k})
    for k in (2, -2, 1):
        res["B"][f"CONTROL_miskey_{k:+d}frames"] = summarize(scan(ident, assumed_shift_frames=k), f"mis-key {k:+d} frames", rng)
    res["B"]["expected"] = {"shift+0.10": 0.10, "shift-0.10": -0.10, "shift-0.20": -0.20, "miskey+2": -2 * float(np.median([r["dt"] for r in A_rows])),
                            "miskey-2": 2 * float(np.median([r["dt"] for r in A_rows]))}

    # ---------------- Stage C: 3-D join is the 2-D join annotated ---------------------------------------------
    n_cmp = n_eq_t = n_eq_xy = n_agents = n_cz = 0
    with lzma.open(JOIN2D, "rt", encoding="utf-8") as f2, lzma.open(JOIN3D, "rt", encoding="utf-8") as f3:
        for i, (l2, l3) in enumerate(zip(f2, f3)):
            if i >= a.n3d_lines:
                break
            r2, r3 = json.loads(l2), json.loads(l3)
            n_cmp += 1
            same_key = (r2["clip_id"] == r3["clip_id"] and r2["frame"] == r3["frame"])
            n_eq_t += int(same_key and r2["t_s"] == r3["t_s"])
            ok_xy = same_key and len(r2["agents"]) == len(r3["agents"]) and all(
                x["cx"] == y["cx"] and x["cy"] == y["cy"] and x["track_id"] == y["track_id"]
                for x, y in zip(r2["agents"], r3["agents"]))
            n_eq_xy += int(ok_xy)
            n_agents += len(r3["agents"])
            n_cz += sum(("cz" in y) for y in r3["agents"])
    res["C_join3d_vs_join2d"] = dict(lines_compared=n_cmp, same_key_and_t_s=n_eq_t, same_xy_and_track_ids=n_eq_xy,
                                     agents=n_agents, agents_with_cz=n_cz)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    b = res["B"]
    print(json.dumps(res["A"], indent=1, default=float))
    print("C0 median |poses - ego(T_{i+k})| by k:", res["C0"]["median_by_k"])
    for k in b:
        if k != "expected":
            print(k, "delta*", round(b[k]["delta_star_s"], 4), "CI", [round(x, 4) for x in b[k]["ci95"]],
                  "err@0", round(b[k]["mean_abs_err_m_at_0"], 3), "err@-0.2", round(b[k]["mean_abs_err_m_at_minus_0p20"], 3),
                  "n_inst", b[k]["n_instances"], "n_clips", b[k]["n_clips"])
    print("expected", b["expected"])
    print("C", res["C_join3d_vs_join2d"])


if __name__ == "__main__":
    main()
