"""town_read: the statistics of PREREG_V8F_TOWN.md for ONE run's twin file (implementation choices frozen in IMPL_CHOICES.md before any number).
usage: town_read.py --twin twin_<tag>.npz --out READ_<tag>.json [--n-boot 2000]
Every interval is the episode-cluster bootstrap (episode = clip index; B 2,000, seed 0) of the program's resampler (taniteval/ci.py: episode_index, _draws, paired_episode_cluster_bootstrap)."""
import os
import sys
import json
import argparse

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
TREE = "F:/TanitAD-scratch/motion_c/ri/tree_ri"
for p in (f"{TREE}/stack", f"{TREE}/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
from taniteval import ci as CI                                              # noqa: E402
from taniteval import v8f_consistency as C                                  # noqa: E402
from tanitad.data import v8f_mini_data as D                                 # noqa: E402

RANK = np.array([0, 2, 1, 4, 3, 6, 5], np.float64)                          # slowest -> fastest; index = LON9 class (HOLD, CREEP, STOP, FOLLOW, DECELERATE, ACCELERATE, KEEP)
RANK_FK = np.array([0, 2, 1, 5, 3, 6, 4], np.float64)                       # beside: FOLLOW and KEEP swapped
RANK_3G = np.array([0, 0, 0, 1, 0, 2, 1], np.float64)                       # beside: slow / middle / fast
DV_MIN = 0.5
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ols(x, y, origin=False):
    if origin:
        return float((x * y).sum() / (x * x).sum())
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())


def slope_ci(dv, d, eid, n_boot, seed=0, origin=False):
    uniq, idx = CI.episode_index(eid)
    pt = ols(dv, d, origin)
    bs = np.array([ols(dv[s], d[s], origin) for s in CI._draws(uniq, idx, n_boot, seed)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"slope": pt, "lo": float(lo), "hi": float(hi), "n_rows": int(len(dv)), "n_episodes": int(len(uniq)), "n_boot": n_boot,
            "below0_separated": bool(hi < 0), "above0_separated": bool(lo > 0), "not_separated": bool(lo <= 0 <= hi), "estimator": "episode_cluster_bootstrap, slope refitted per draw"}


def f1_paired(gt, po, pl, n, eid, n_boot):
    m = gt >= 0
    a, b, g = po[m], pl[m], gt[m]
    red = lambda v: C.macro_f1(v.astype(np.int64) // (n + 1), v.astype(np.int64) % (n + 1), n)["macro_f1"]
    ca, cb = g * (n + 1) + a, g * (n + 1) + b
    r = CI.paired_episode_cluster_bootstrap(ca, cb, eid[m], n_boot=n_boot, seed=0, reduce=red)
    r.update({"own": float(red(ca)), "logged": float(red(cb)), "worse_separated": bool(r["delta"] < 0 and r["hi"] < 0), "n_labelled": int(m.sum())})
    return r


def rms_paired(se_own, se_log, eid, n_boot):
    m = np.isfinite(se_own) & np.isfinite(se_log)
    if m.sum() == 0:
        return {"n": 0}
    r = CI.paired_episode_cluster_bootstrap(se_own[m], se_log[m], eid[m], n_boot=n_boot, seed=0, reduce="rms")
    r.update({"own_rmse": float(np.sqrt(se_own[m].mean())), "logged_rmse": float(np.sqrt(se_log[m].mean())), "n": int(m.sum())})
    return r


def read_twin(path, n_boot=2000):
    z = np.load(path, allow_pickle=True)
    ci, tick = z["ci"], z["tick"]
    eid = ci.astype(str)
    dv = (z["v_log"] - z["v_own"]).astype(np.float64)
    dv_T = (z["v_log_T"] - z["v_own_T"]).astype(np.float64)
    po, pl = softmax(z["own_lon"]), softmax(z["twin_lon"])
    out = {"n_rows": int(len(ci)), "n_episodes": int(len(np.unique(ci))), "rows_per_tick": [int((tick == k).sum()) for k in range(int(tick.max()) + 1)],
           "rank_order_slowest_to_fastest": ["HOLD", "STOP", "CREEP", "DECELERATE", "FOLLOW", "KEEP", "ACCELERATE"]}
    for nm, rk in (("committed", RANK), ("beside_follow_keep_swapped", RANK_FK), ("beside_three_group", RANK_3G)):
        d = po @ rk - pl @ rk
        sel = np.abs(dv) >= DV_MIN
        r = {"d_mean_all": float(d.mean()), "d_mean_tick0_max_abs": float(np.abs(d[tick == 0]).max()), "n_slope_rows": int(sel.sum())}
        if sel.sum() > 20:
            r["slope"] = slope_ci(dv[sel], d[sel], eid[sel], n_boot)
            if nm == "committed":
                r["slope_through_origin"] = slope_ci(dv[sel], d[sel], eid[sel], n_boot, origin=True)
                sel_T = np.abs(dv_T) >= DV_MIN
                r["slope_dv_at_T_frame"] = slope_ci(dv_T[sel_T], d[sel_T], eid[sel_T], n_boot) if sel_T.sum() > 20 else None
        out["shift_" + nm] = r
    out["deficit"] = {"dv_mean": float(dv.mean()), "dv_mean_tick1": float(dv[tick == 1].mean()) if (tick == 1).any() else None, "dv_mean_tick2": float(dv[tick == 2].mean()) if (tick == 2).any() else None,
                      "share_abs_dv_ge_0.5": float((np.abs(dv) >= DV_MIN).mean()), "limit_bucket_differs_share": float((z["limit_own"] != z["limit_log"]).mean())}
    # quality
    has = z["gt_has"].astype(bool)
    lat_gt = np.where(has & (z["gt_lat"] >= 0), z["gt_lat"], -100)
    lon_gt = np.where(has & (z["gt_lon"] >= 0), z["gt_lon"], -100)
    q = {}
    for pop, mk in (("all_replans", np.ones(len(ci), bool)), ("ticks_1_2", tick >= 1)):
        q[pop] = {"lat_macro_f1": f1_paired(lat_gt[mk], z["own_lat"][mk].argmax(-1), z["twin_lat"][mk].argmax(-1), 5, eid[mk], n_boot),
                  "lon_macro_f1": f1_paired(lon_gt[mk], z["own_lon"][mk].argmax(-1), z["twin_lon"][mk].argmax(-1), 7, eid[mk], n_boot)}
    out["quality"] = q
    cons_gt = z["gt_cons"].astype(np.float64)
    se_o, se_l = (z["own_cons"] - cons_gt) ** 2, (z["twin_cons"] - cons_gt) ** 2
    se_o[~has], se_l[~has] = np.nan, np.nan
    ix = [D.CONS_IDX[c] for c in SPEED_CELLS]
    def rowmean(se):
        with np.errstate(all="ignore"):
            return np.nanmean(se[:, ix], axis=1)
    out["constraint_rmse"] = {"speed_kind_m_per_s_per_row_mean": rms_paired(rowmean(se_o), rowmean(se_l), eid, n_boot),
                              "per_cell": {c: rms_paired(se_o[:, i], se_l[:, i], eid, n_boot) for c, i in D.CONS_IDX.items()}}
    # beside: share of replans whose lon argmax is slower than GT's
    ok = lon_gt >= 0
    so = (RANK[z["own_lon"].argmax(-1)] < RANK[np.where(ok, lon_gt, 0)]).astype(np.float64)
    sl = (RANK[z["twin_lon"].argmax(-1)] < RANK[np.where(ok, lon_gt, 0)]).astype(np.float64)
    out["share_argmax_slower_than_gt"] = {**CI.paired_episode_cluster_bootstrap(so[ok], sl[ok], eid[ok], n_boot=n_boot, seed=0), "own": float(so[ok].mean()), "logged": float(sl[ok].mean()), "n": int(ok.sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    r = read_twin(a.twin, a.n_boot)
    json.dump(r, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    s = r["shift_committed"].get("slope")
    print("READ", a.out, "slope", None if s is None else (round(s["slope"], 5), round(s["lo"], 5), round(s["hi"], 5)), "n_rows", r["n_rows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
