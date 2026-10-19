"""Zero-compute natural experiment on the banked step40000ema_unc battery dumps.
Rows of ONE batched forward differ ONLY in the fed nav token (same frames, v0, ego, r8_nav=None).
Where the banked SHUFFLED / FLIPPED token happens to equal the v9 PER-FRAME token at a window whose clip-level
token (os) is wrong, that row IS the per-frame-correct roll. Control: windows where shuf == fed (identical token,
different DDIM row) must read ~0."""
import json, sys, importlib.util
import numpy as np
from pathlib import Path

V9, ROOT, OUT, CI = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location("tci", CI); tci = importlib.util.module_from_spec(spec); spec.loader.exec_module(tci)
z = np.load(V9, allow_pickle=False)
by_sid = {int(s): i for i, s in enumerate(z["clip__sid"])}
row0, nr, k0, kk, tok = z["clip__row0"], z["clip__n_rows"], z["clip__k0"], z["row__k"], z["row__nav_token"]
SLOT_IDX = [4, 9, 14, 19, 29, 39, 49, 59]        # model horizons 5..60 steps @0.1 s -> gt_future_ext index h-1

def v9_at(s, k):
    i = by_sid[int(s)]; r = int(row0[i]) + int(k) - int(k0[i])
    assert 0 <= r - int(row0[i]) < int(nr[i]) and int(kk[r]) == int(k)
    return int(tok[r])

def ego_frame(gt, pl):
    c, s = np.cos(pl[:, 2]), np.sin(pl[:, 2])
    dx, dy = gt[..., 0] - pl[:, None, 0], gt[..., 1] - pl[:, None, 1]
    return np.stack([c[:, None] * dx + s[:, None] * dy, -s[:, None] * dx + c[:, None] * dy], -1)

res = {}
def load(dump, arms):
    man = json.load(open(dump / "manifest.json", encoding="utf-8"))
    rows = {k: [] for k in ("eid", "fed", "shuf", "flip", "v9")}
    met = {}
    maxdiff_g = 0.0
    for e in man["episodes"]:
        fi, s = int(e["file_index"]), int(e["episode_id"])
        ep = np.load(dump / f"ep{fi:03d}.npz"); d = np.load(dump / "decisions" / f"ep{fi:03d}.npz")
        n = len(d["ws"])
        gt = ego_frame(d["gt_future_ext"][:, SLOT_IDX, :2].astype(np.float64), d["pose_last"].astype(np.float64))
        gv = d["gt_future_valid_ext"][:, SLOT_IDX] > 0.5
        maxdiff_g = max(maxdiff_g, float(np.abs(gt[:, :4] - ep["g"]).max()))   # independent frame check vs banked g
        for arm, key in arms:
            p = d[key].astype(np.float64)
            err = np.linalg.norm(p - gt, axis=-1)
            ade2 = err[:, :4].mean(1)
            ade6 = np.where(gv, err, 0).sum(1) / np.maximum(gv.sum(1), 1)
            last = np.where(gv.all(1))[0]
            lat6 = np.full(n, np.nan); lon6 = np.full(n, np.nan)
            lat6[last] = np.abs(p[last, -1, 1] - gt[last, -1, 1]); lon6[last] = np.abs(p[last, -1, 0] - gt[last, -1, 0])
            for mname, v in (("ade_2s", ade2), ("ade_6s", ade6), ("abs_lat_err_6s", lat6), ("abs_lon_err_6s", lon6)):
                met.setdefault((arm, mname), []).append(v)
        rows["eid"] += [s] * n
        rows["fed"] += list(d["nav_cmd"].astype(int)); rows["shuf"] += list(d["nav_cmd_shuf"].astype(int))
        rows["flip"] += list(d["nav_cmd_flip"].astype(int)) if "nav_cmd_flip" in d.files else [-9] * n
        rows["v9"] += [v9_at(s, int(w) + 2) for w in d["ws"]]
    return {k: np.asarray(v) for k, v in rows.items()}, {k: np.concatenate(v) for k, v in met.items()}, maxdiff_g
if True:
    R, M, maxdiff_g = load(ROOT / "dump_s0", (("os", "plan_full_nav_true"), ("shuf", "plan_full_nav_shuffled"), ("flip", "plan_full_nav_flipped")))
    R1, M1, _ = load(ROOT / "dump_s1", (("os", "plan_full_nav_true"),))
    assert all(np.array_equal(R[k], R1[k]) for k in ("eid", "fed", "v9"))      # same windows, same fed token
    for kk_ in list(M1): M[("os_s1",) + kk_[1:]] = M1[kk_]
    subsets = {
        "FIX_shuf (fed!=v9, shuf==v9): shuf row = per-frame-correct": (R["fed"] != R["v9"]) & (R["shuf"] == R["v9"]),
        "FIX_flip (fed!=v9, flip==v9): flip row = per-frame-correct": (R["fed"] != R["v9"]) & (R["flip"] == R["v9"]),
        "CTRL (shuf==fed): same token, other DDIM row -> must read ~0": (R["shuf"] == R["fed"]),
        "BREAK (fed==v9, shuf!=v9): shuf row = a wrong token": (R["fed"] == R["v9"]) & (R["shuf"] != R["v9"]),
    }
    out = {"frame_check_max_abs_m_vs_banked_g": round(maxdiff_g, 6)}
    for name, m in subsets.items():
        arm0 = "flip" if name.startswith("FIX_flip") else "shuf"
        rec = {"n_windows": int(m.sum()), "n_episodes": int(len(set(R["eid"][m].tolist())))}
        for arm, mname in [(a_, m_) for a_ in (arm0, "os_s1") for m_ in ("ade_2s", "ade_6s", "abs_lat_err_6s", "abs_lon_err_6s")]:
            a, b = M[(arm, mname)][m], M[("os", mname)][m]
            mname = mname if arm == arm0 else "SEEDFLOOR_os_s1_minus_os_s0." + mname
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10:
                rec[mname] = {"n": int(ok.sum()), "note": "too few"}; continue
            bt = tci.paired_episode_cluster_bootstrap(a[ok], b[ok], R["eid"][m][ok], n_boot=2000, seed=0)
            rec[mname] = {"n": int(ok.sum()), "arm_mean": round(float(a[ok].mean()), 4), "os_mean": round(float(b[ok].mean()), 4),
                          "delta_arm_minus_os": round(bt["point"] if "point" in bt else float(a[ok].mean() - b[ok].mean()), 4),
                          "ci95": [round(float(x), 4) for x in (bt.get("lo"), bt.get("hi"))], "separated": bool(bt.get("separated")),
                          "estimator": bt.get("estimator", "paired_episode_cluster_bootstrap")}
        out[name] = rec
    res["dump_s0 (+ dump_s1 os as inference-seed floor)"] = out
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
