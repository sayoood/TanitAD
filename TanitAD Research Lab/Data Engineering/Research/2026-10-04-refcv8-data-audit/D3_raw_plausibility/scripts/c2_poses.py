#!/usr/bin/env python3
"""D3 check 2 (+ check 6 window balance) -- ego-pose plausibility of the v2ep poses that the trainer reads.

Input  : _v2manifest.pt of the TRAIN view (4,369 clips; copied from Thor) and of the EVAL view (139 clips).
         poses[t] = (x_east_m, y_north_m, yaw_rad, v_mps), 10 Hz, provider rows AFTER the n_stack-1 drop
         (199 rows/clip), actions[t] = (steer_road_rad, accel_mps2)  [stack/tanitad/data/physicalai.py:144,635-641].
         dt per clip from the clock sidecar (dt_s, MEASURED median 0.100667), nominal 0.1007 when absent.
Every threshold below is a LITERAL from vehicle / road physics (not imported from the builder).
Controls run FIRST on synthetic tracks with analytically known answers and must read their known values.
Ids are reported as sha12 only.
"""
from __future__ import annotations
import gzip, hashlib, json, math, os, sys
import numpy as np
import torch

OUT = os.environ.get("D3_OUT", "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility/raw")
TRAIN_MAN = os.environ["D3_TRAIN_MAN"]
EVAL_MAN = "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"
SIDECAR = "D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl"
LAB_TRAIN = "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz"
LAB_EVAL = "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"

# ---- literal thresholds (physics) --------------------------------------------------------------------
T_JUMP_LITERAL_M = 3.0        # brief: |dxy| > 3 m in one 0.1 s step  (legit above 29 m/s = 104 km/h)
T_JUMP_SPEEDCONS_M = 0.5      # |dxy - vbar*dt| > 0.5 m : position channel disagrees with the speed channel
T_VMAX = 60.0                 # m/s = 216 km/h
T_ACC = 8.0                   # m/s^2 longitudinal (hard braking ~ 1 g limit is 9.8; > 8 is at the tyre limit)
T_JERK = 30.0                 # m/s^3 sustained
JERK_RUN = 3                  # consecutive samples (0.3 s)
T_YAWRATE = 1.2               # rad/s at v > 5 m/s
V_YAW = 5.0
T_ALAT = 8.0                  # m/s^2 lateral (mu 0.8 friction circle) at v > 5
T_HEAD_DEG = 20.0             # heading vs velocity direction while v > 3 m/s
V_HEAD = 3.0
T_REV_DEG = 150.0             # reversing: velocity direction opposite to heading while v > 1 m/s
V_REV = 1.0
STAT_VMAX, STAT_PATH = 0.2, 1.0   # fully stationary clip
DT_NOM = 0.1007
TURN_DEG, HIGH_SPEED, STOP_V = 30.0, 25.0, 0.5   # check-6 literals
WINDOW, MAXH = 8, 20           # window=8 (cfg predictor window), max_horizon=20 -> windows t in range(T-28)
HORIZON_STEPS = 60             # 6 s


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_clip(poses, accel_prov, dt):
    """Return dict of per-row flag arrays and scalars for one clip. poses [T,4] float64."""
    x, y, yaw, v = (poses[:, i].astype(np.float64) for i in range(4))
    T = len(x)
    dx, dy = np.diff(x), np.diff(y)
    d = np.hypot(dx, dy)
    vbar = 0.5 * (v[1:] + v[:-1])
    a = np.diff(v) / dt
    jerk = np.diff(a) / dt
    w = wrap(np.diff(yaw)) / dt
    yawm = np.arctan2(np.sin(yaw[1:]) + np.sin(yaw[:-1]), np.cos(yaw[1:]) + np.cos(yaw[:-1]))
    thv = np.arctan2(dy, dx)
    mis = np.degrees(np.abs(wrap(thv - yawm)))
    mov3 = (vbar > V_HEAD) & (d > 1e-6)
    out = {
        "finite": bool(np.isfinite(poses).all()),
        "jump_literal": d > T_JUMP_LITERAL_M,
        "jump_speedcons": np.abs(d - vbar * dt) > T_JUMP_SPEEDCONS_M,
        "vmax": v > T_VMAX,
        "acc": np.abs(a) > T_ACC,
        "yawrate": (np.abs(w) > T_YAWRATE) & (vbar > V_YAW),
        "alat": (np.abs(vbar * w) > T_ALAT) & (vbar > V_YAW),
        "head": (mis > T_HEAD_DEG) & mov3,
        "rev": (mis > T_REV_DEG) & (vbar > V_REV) & (d > 1e-6),
        "mov3": mov3,
        "moving": vbar > 0.5,
    }
    # sustained = |jerk| > T_JERK with the SAME SIGN for >= JERK_RUN consecutive samples (an acceleration RAMP).
    # A one-sample speed glitch makes a (+,-,+) jerk triplet, which is NOT sustained (it is caught by `acc`).
    j = np.abs(jerk) > T_JERK
    sg = np.sign(jerk)
    run = np.zeros_like(j)
    i = 0
    while i < len(j):
        if j[i]:
            k = i
            while k < len(j) and j[k] and sg[k] == sg[i]:
                k += 1
            if k - i >= JERK_RUN:
                run[i:k] = True
            i = k
        else:
            i += 1
    out["jerk_sustained"] = run
    out["stationary_clip"] = bool(v.max() < STAT_VMAX and d.sum() < STAT_PATH)
    out["pose_dt"] = float(d[vbar > 2.0].sum() / (vbar[vbar > 2.0].sum()) if (vbar > 2.0).sum() >= 10 else np.nan)
    out["mis_med_mov3"] = float(np.median(mis[mov3])) if mov3.any() else np.nan
    out["acc_prov_vs_fd"] = float(np.median(np.abs(accel_prov[1:-1] - 0.5 * (a[1:] + a[:-1])))) if accel_prov is not None and len(a) > 2 else np.nan
    out["v"], out["w"], out["a"], out["d"], out["mis"], out["vbar"] = v, w, a, d, mis, vbar
    return out


# ---- CONTROLS ----------------------------------------------------------------------------------------
def controls():
    dt = 0.1
    res = {}
    t = np.arange(199) * dt
    R, V = 50.0, 10.0
    om = V / R
    th = om * t
    circ = np.stack([R * np.sin(th), R * (1 - np.cos(th)), th, np.full_like(t, V)], 1)
    r = check_clip(circ, None, dt)
    res["circle_R50_v10"] = {
        "yawrate_median_expected_0.2": float(np.median(np.abs(r["w"]))),
        "alat_expected_2.0": float(np.median(np.abs(r["vbar"] * r["w"]))),
        "mismatch_deg_expected_0": float(np.median(r["mis"])),
        "n_flags_expected_0": int(sum(int(r[k].sum()) for k in ("jump_literal", "jump_speedcons", "vmax", "acc", "yawrate", "alat", "head", "rev", "jerk_sustained"))),
    }
    rev = circ.copy(); rev[:, 2] = wrap(rev[:, 2] + np.pi)
    r = check_clip(rev, None, dt)
    res["circle_reversed"] = {"rev_rows_expected_198": int(r["rev"].sum()), "head_rows_expected_198": int(r["head"].sum())}
    j = circ.copy(); j[100:, 0] += 5.0
    r = check_clip(j, None, dt)
    res["jump_5m_at_step_100"] = {"jump_literal_expected_1": int(r["jump_literal"].sum()), "jump_speedcons_expected_1": int(r["jump_speedcons"].sum())}
    f = circ.copy(); f[:, 3] = 70.0
    r = check_clip(f, None, dt)
    res["speed_70"] = {"vmax_rows_expected_199": int(r["vmax"].sum())}
    # sustained jerk: acceleration ramp a(t)=60*(t-t0) for 12 steps (jerk 60 m/s^3) => v quadratic
    v = np.full(199, 10.0)
    for i in range(80, 92):
        v[i] = v[i - 1] + 60.0 * ((i - 80) * dt) * dt
    for i in range(92, 199):
        v[i] = v[91]
    q = np.stack([np.cumsum(np.r_[0, v[:-1] * dt]), np.zeros(199), np.zeros(199), v], 1)
    r = check_clip(q, None, dt)
    res["jerk_ramp_60"] = {"jerk_sustained_rows_expected_ge_3": int(r["jerk_sustained"].sum())}
    # single-step spike must NOT be 'sustained'
    v2 = np.full(199, 10.0); v2[100] = 12.0
    q2 = np.stack([np.cumsum(np.r_[0, v2[:-1] * dt]), np.zeros(199), np.zeros(199), v2], 1)
    r = check_clip(q2, None, dt)
    res["jerk_single_spike"] = {"jerk_sustained_rows_expected_0": int(r["jerk_sustained"].sum()), "acc_rows_expected_2": int(r["acc"].sum())}
    st = np.zeros((199, 4)); r = check_clip(st, None, dt)
    res["stationary"] = {"stationary_clip_expected_True": r["stationary_clip"]}
    ok = (abs(res["circle_R50_v10"]["yawrate_median_expected_0.2"] - 0.2) < 1e-6
          and abs(res["circle_R50_v10"]["alat_expected_2.0"] - 2.0) < 1e-2
          and res["circle_R50_v10"]["mismatch_deg_expected_0"] < 1.0
          and res["circle_R50_v10"]["n_flags_expected_0"] == 0
          and res["circle_reversed"]["rev_rows_expected_198"] == 198
          and res["jump_5m_at_step_100"]["jump_literal_expected_1"] == 1
          and res["jump_5m_at_step_100"]["jump_speedcons_expected_1"] == 1
          and res["speed_70"]["vmax_rows_expected_199"] == 199
          and res["jerk_ramp_60"]["jerk_sustained_rows_expected_ge_3"] >= 3
          and res["jerk_single_spike"]["jerk_sustained_rows_expected_0"] == 0
          and res["stationary"]["stationary_clip_expected_True"])
    res["ALL_CONTROLS_READ_KNOWN_VALUE"] = bool(ok)
    return res


def load_sidecar():
    out = {}
    with open(SIDECAR, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[int(r["sid"])] = r
    return out


def load_strata(path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = r.get("strata", {})
            rows[r["clip_id"]] = {"country": s.get("country"), "daynight": s.get("daynight_clock"), "road": s.get("road_class"),
                                  "span": r.get("horizon", {}).get("recording_span_s"), "nav": r.get("nav_command", {}).get("token"),
                                  "stopped_frac": s.get("stopped_frame_frac_20s")}
    return rows


def pct(n, d):
    return round(100.0 * n / d, 3) if d else None


def run_split(name, man, sidecar, strata):
    n = len(man["clip_id"])
    agg = {k: 0 for k in ("jump_literal", "jump_speedcons", "vmax", "acc", "yawrate", "alat", "head", "rev", "jerk_sustained")}
    clips_with = {k: 0 for k in agg}
    rows_total = 0
    mov3_rows = 0
    clips_nonfinite = 0
    stationary = []
    ex = {k: [] for k in agg}
    pose_dt = []; sc_dt = []; dt_src = {"sidecar": 0, "nominal": 0}
    lens = []
    v_all = []; w_all = []
    sys_head = []
    acc_prov = []
    # check-6 window stats
    win = {"n": 0, "full6": 0, "turn_full6": 0, "turn_any": 0, "stop_now": 0, "highspeed": 0, "stop_in_6s_full6": 0,
           "night": 0, "day": 0, "unk_dn": 0}
    by_country = {}
    spans = []
    for i in range(n):
        cid = man["clip_id"][i]
        P = man["poses"][i].numpy().astype(np.float64)
        A = man["actions"][i].numpy().astype(np.float64)
        sid = int(man["episode_uid"][i])
        sc = sidecar.get(sid)
        if sc is not None:
            dt = float(sc["dt_s"]); dt_src["sidecar"] += 1
        else:
            dt = DT_NOM; dt_src["nominal"] += 1
        r = check_clip(P, A[:, 1], dt)
        lens.append(len(P))
        if not r["finite"]:
            clips_nonfinite += 1
            continue
        for k in agg:
            c = int(r[k].sum())
            agg[k] += c
            if c:
                clips_with[k] += 1
                if len(ex[k]) < 3:
                    ex[k].append(sha12(cid))
        rows_total += len(r["d"])
        mov3_rows += int(r["mov3"].sum())
        if r["stationary_clip"]:
            stationary.append(sha12(cid))
        pose_dt.append(r["pose_dt"]); sc_dt.append(dt if sc is not None else np.nan)
        acc_prov.append(r["acc_prov_vs_fd"])
        v_all.append(P[:, 3]); w_all.append(np.where(r["vbar"] > 2.0, np.abs(r["w"]), np.nan))
        if not np.isnan(r["mis_med_mov3"]) and r["mis_med_mov3"] > T_HEAD_DEG:
            sys_head.append(sha12(cid))
        # ---- windows ----
        T = len(P)
        L = strata.get(cid, {})
        yaw_un = np.unwrap(P[:, 2])
        dn = L.get("daynight"); ctry = L.get("country") or "UNK"
        spans.append(L.get("span"))
        for t in range(max(0, T - WINDOW - MAXH)):
            now = t + WINDOW - 1
            win["n"] += 1
            vnow = P[now, 3]
            full = now + HORIZON_STEPS <= T - 1
            end = min(now + HORIZON_STEPS, T - 1)
            dyaw = abs(math.degrees(yaw_un[end] - yaw_un[now]))
            if full:
                win["full6"] += 1
                if dyaw >= TURN_DEG:
                    win["turn_full6"] += 1
                if P[now:end + 1, 3].min() < STOP_V:
                    win["stop_in_6s_full6"] += 1
            if dyaw >= TURN_DEG:
                win["turn_any"] += 1
            if vnow < STOP_V:
                win["stop_now"] += 1
            if vnow > HIGH_SPEED:
                win["highspeed"] += 1
            if dn == "night":
                win["night"] += 1
            elif dn == "day":
                win["day"] += 1
            else:
                win["unk_dn"] += 1
            b = by_country.setdefault(ctry, {"n": 0, "turn_full6": 0, "full6": 0, "stop_now": 0, "highspeed": 0, "night": 0})
            b["n"] += 1; b["stop_now"] += int(vnow < STOP_V); b["highspeed"] += int(vnow > HIGH_SPEED); b["night"] += int(dn == "night")
            if full:
                b["full6"] += 1; b["turn_full6"] += int(dyaw >= TURN_DEG)
    V = np.concatenate(v_all); Wr = np.concatenate(w_all); Wr = Wr[~np.isnan(Wr)]
    vb = [0, 0.5, 2, 5, 10, 15, 20, 25, 30, 35, 40, 60, 1e9]
    vh = np.histogram(V, bins=vb)[0]
    wb = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1e9]
    wh = np.histogram(Wr, bins=wb)[0]
    pdt = np.array(pose_dt, dtype=float); sdt = np.array(sc_dt, dtype=float)
    both = ~np.isnan(pdt) & ~np.isnan(sdt)
    ratio = pdt[both] / sdt[both]
    res = {
        "split": name, "n_clips": n, "n_unreadable_nonfinite": clips_nonfinite, "rows_pairs_total": rows_total,
        "dt_source": dt_src, "n_rows_per_clip": {"min": int(min(lens)), "max": int(max(lens)), "median": float(np.median(lens))},
        "flag_rows": {k: {"rows": agg[k], "pct_rows": pct(agg[k], rows_total), "clips": clips_with[k], "pct_clips": pct(clips_with[k], n), "examples_sha12": ex[k]} for k in agg},
        "moving_gt3_rows": mov3_rows,
        "heading_flag_pct_of_moving_gt3": pct(agg["head"], mov3_rows),
        "clips_median_heading_mismatch_gt20deg": {"n": len(sys_head), "examples_sha12": sys_head[:3]},
        "stationary_clips": {"n": len(stationary), "pct": pct(len(stationary), n), "examples_sha12": stationary[:3]},
        "speed_hist_rows": {"bins_ms": vb[:-1] + ["inf"], "pct": [round(100 * x / V.size, 3) for x in vh], "n": int(V.size)},
        "yawrate_hist_abs_v_gt2": {"bins_rads": wb[:-1] + ["inf"], "pct": [round(100 * x / Wr.size, 3) for x in wh], "n": int(Wr.size)},
        "speed_pctiles": {q: round(float(np.quantile(V, q)), 3) for q in (0.5, 0.9, 0.99, 0.999)} | {"max": round(float(V.max()), 3)},
        "pose_dt_vs_sidecar_dt_ratio": {"n": int(both.sum()), "median": float(np.median(ratio)) if both.any() else None,
                                          "p01": float(np.quantile(ratio, 0.01)) if both.any() else None, "p99": float(np.quantile(ratio, 0.99)) if both.any() else None,
                                          "n_abs_dev_gt_2pct": int((np.abs(ratio - 1) > 0.02).sum()) if both.any() else None},
        "actions_accel_vs_finite_diff_median_abs_diff": {"median_over_clips": float(np.nanmedian(acc_prov)), "p99": float(np.nanquantile(acc_prov, 0.99))},
        "windows": win | {"pct_turn_of_full6": pct(win["turn_full6"], win["full6"]), "pct_stop_now": pct(win["stop_now"], win["n"]),
                          "pct_highspeed": pct(win["highspeed"], win["n"]), "pct_night": pct(win["night"], win["n"]),
                          "pct_full6_future": pct(win["full6"], win["n"]), "pct_stop_in_6s_of_full6": pct(win["stop_in_6s_full6"], win["full6"])},
        "by_country": {k: {**v, "pct_turn": pct(v["turn_full6"], v["full6"]), "pct_stop": pct(v["stop_now"], v["n"]),
                           "pct_high": pct(v["highspeed"], v["n"]), "pct_night": pct(v["night"], v["n"])} for k, v in sorted(by_country.items())},
    }
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    ctl = controls()
    print("CONTROLS", json.dumps(ctl, indent=1))
    if not ctl["ALL_CONTROLS_READ_KNOWN_VALUE"]:
        raise SystemExit("controls failed -- refusing to run")
    side = load_sidecar()
    strata = {}
    strata.update(load_strata(LAB_TRAIN)); strata.update(load_strata(LAB_EVAL))
    out = {"controls": ctl, "thresholds_literal": {k: v for k, v in globals().items() if k.startswith(("T_", "V_", "JERK", "STAT", "TURN", "HIGH", "STOP"))}}
    for name, path in (("eval139", EVAL_MAN), ("train4369", TRAIN_MAN)):
        man = torch.load(path, map_location="cpu", weights_only=False)
        n_lab = sum(1 for c in man["clip_id"] if c in strata)
        r = run_split(name, man, side, strata)
        r["clips_with_label_record"] = n_lab
        out[name] = r
        print(name, "done", flush=True)
    json.dump(out, open(f"{OUT}/c2_poses_result.json", "w"), indent=1)
    print("written")


if __name__ == "__main__":
    main()
