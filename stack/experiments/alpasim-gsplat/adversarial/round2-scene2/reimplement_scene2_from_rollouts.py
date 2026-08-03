"""INDEPENDENT re-derivation of the scene-2 panel numbers.

Written from the rollout dumps ONLY (no import of cl_metrics), to check the staged
metrics files. Uses taniteval.ci for the paired episode-cluster bootstrap.
"""
import json, math, sys
from pathlib import Path
import numpy as np

RES = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/scene2-realclose")
DT = 0.1
WP = (5, 10, 15, 20)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def gt_arrays(gt):
    xy = np.array([[g["x"], g["y"]] for g in gt], float)
    yaw = np.array([g["yaw"] for g in gt], float)
    v = np.zeros(len(gt))
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT
    v[:-1] = d
    v[-1] = d[-1] if len(d) else 0.0
    return xy, yaw, v


def ego_frame(d, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def xtrack(p, xy):
    seg = xy[1:] - xy[:-1]
    L2 = (seg ** 2).sum(1).clip(1e-9)
    t = (((p[None, :2] - xy[:-1]) * seg).sum(1) / L2).clip(0, 1)
    proj = xy[:-1] + t[:, None] * seg
    d = np.linalg.norm(proj - p[None, :2], axis=1)
    i = int(np.argmin(d))
    n = seg[i] / math.sqrt(L2[i])
    r = p[:2] - proj[i]
    return float(n[0] * r[1] - n[1] * r[0])


def plan_poses(plan, v0, n=21):
    knots = np.vstack([[0.0, 0.0], np.asarray(plan, float)])
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tq = np.arange(n) * DT
    x = np.interp(tq, ts, knots[:, 0])
    y = np.interp(tq, ts, knots[:, 1])
    d = np.diff(np.stack([x, y], 1), axis=0, prepend=np.zeros((1, 2)))
    yaw = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
    v = np.linalg.norm(d, axis=1) / DT
    v[0] = v0
    return np.stack([x, y, yaw, v], 1)


def score(path):
    d = json.loads(Path(path).read_text())
    gt = d["gt"]
    xy, gyaw, gv = gt_arrays(gt)
    N = len(gt)
    rows, eid = [], []
    for r in d["rollouts"]:
        for k, st in enumerate(r["steps"]):
            i = int(st["i_gt"])
            plan = np.array(st["plan"], float)
            gtc = np.stack([ego_frame(xy[min(i + h, N - 1)] - xy[i], gyaw[i]) for h in WP])
            de = np.linalg.norm(plan - gtc, axis=1)
            lon = np.abs(plan[:, 0] - gtc[:, 0])
            lat = np.abs(plan[:, 1] - gtc[:, 1])
            j = min(i + WP[-1], N - 1)
            dyaw_gt = wrap(gyaw[j] - gyaw[i])
            arc_gt = float(np.linalg.norm(np.diff(xy[i:j + 1], axis=0), axis=1).sum())
            pp = plan_poses(plan, st["v"])
            dyaw_pl = wrap(float(pp[-1, 2]))
            arc_pl = float(np.linalg.norm(np.diff(pp[:, :2], axis=0), axis=1).sum())
            ct = xtrack(np.array(st["ego"][:2]), xy)
            yr_exec = st["v"] / 2.7 * math.tan(st["steer"])
            yr_gt = wrap(gyaw[min(i + 1, N - 1)] - gyaw[i]) / DT
            rows.append({
                "ade": float(de.mean()), "de2s": float(de[-1]),
                "lon_ade": float(lon.mean()), "lat_ade": float(lat.mean()),
                "abs_speed_err": abs(float(st["v_target"] - gv[i])),
                "speed_err": float(st["v_target"] - gv[i]),
                "exec_speed_err": float(st["v"] - gv[i]),
                "heading_err": abs(float(wrap(dyaw_pl - dyaw_gt))),
                "curv_err": abs(dyaw_pl / max(arc_pl, 1.0) - dyaw_gt / max(arc_gt, 1.0)),
                "yawrate_err": abs(yr_exec - yr_gt),
                "cross_track_abs": abs(ct),
                "corridor_dep": float(abs(ct) > 2.0),
            })
            eid.append(r["start_frame"])
    return rows, eid, d


def boot_paired(a, b, eid, nboot=2000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); eid = np.asarray(eid)
    ue = np.unique(eid)
    idx = {e: np.where(eid == e)[0] for e in ue}
    rng = np.random.default_rng(seed)
    d0 = float(np.mean(a - b))
    out = []
    for _ in range(nboot):
        pick = rng.choice(ue, size=len(ue), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        out.append(float(np.mean(a[sel] - b[sel])))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return d0, float(lo), float(hi), bool(lo > 0 or hi < 0)


def boot_mean(a, eid, nboot=2000, seed=0):
    a = np.asarray(a, float); eid = np.asarray(eid)
    ue = np.unique(eid); idx = {e: np.where(eid == e)[0] for e in ue}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        pick = rng.choice(ue, size=len(ue), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        out.append(float(np.mean(a[sel])))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.mean(a)), float(lo), float(hi)


KEYS = ["ade", "de2s", "lon_ade", "lat_ade", "abs_speed_err", "exec_speed_err",
        "heading_err", "curv_err", "yawrate_err", "cross_track_abs", "corridor_dep"]

if __name__ == "__main__":
    data = {}
    for arm in ["flagship-v1", "refc-base"]:
        for cond in ["objects", "empty"]:
            data[(arm, cond)] = score(RES / "rollouts" / f"rollouts_{arm}_{cond}.json")

    print("=== SINGLE-ARM MEANS (episode-cluster bootstrap over 9 starts) ===")
    for key in ["ade", "de2s", "abs_speed_err", "heading_err", "curv_err",
                "yawrate_err", "cross_track_abs"]:
        line = [key.ljust(16)]
        for arm in ["flagship-v1", "refc-base"]:
            rows, eid, _ = data[(arm, "objects")]
            m, lo, hi = boot_mean([r[key] for r in rows], eid)
            line.append(f"{arm}/objects {m:8.4f} [{lo:7.4f},{hi:7.4f}]")
        print("  ".join(line))

    print()
    print("=== PAIRED CONTRASTS ===")
    for label, (A, B) in {
        "flagship objects-empty": (("flagship-v1", "objects"), ("flagship-v1", "empty")),
        "refc     objects-empty": (("refc-base", "objects"), ("refc-base", "empty")),
        "flag-refc in objects":   (("flagship-v1", "objects"), ("refc-base", "objects")),
        "flag-refc in empty":     (("flagship-v1", "empty"), ("refc-base", "empty")),
    }.items():
        ra, ea, _ = data[A]; rb, eb, _ = data[B]
        assert ea == eb, (A, B)
        nsep = 0
        det = []
        for key in KEYS:
            d0, lo, hi, sep = boot_paired([r[key] for r in ra], [r[key] for r in rb], ea)
            nsep += sep
            det.append(f"{key}={d0:+.4f}[{lo:+.4f},{hi:+.4f}]{'*' if sep else ''}")
        print(f"{label}: SEPARATED {nsep}/{len(KEYS)}")
        for x in det:
            print("     ", x)
