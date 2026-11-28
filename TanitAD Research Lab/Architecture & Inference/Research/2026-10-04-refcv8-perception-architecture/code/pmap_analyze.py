"""WP-D P-MAP analysis (CPU): the registered metrics, bars and must-fails of PREREG_WPD_PROBES.md §P-MAP from the arms'
per-episode tolerant counts (`pmap_arms.py` -> <arm>/eval_acc.pt; EVAL-DIAG, thresholds fitted per arm on TRAIN-DIAG).

Per class (lane, edge, crosswalk, drivable) x band (0-20, 20-40 m): IoU_0, IoU_2 (0.2 m Chebyshev tolerance), P@1 m and
R@1 m (the diagnostics' M-b definitions, `diag_metrics.tolerant_summary`), and the share of GT-edge cells whose
nearest PREDICTED drivable boundary is within 0.2 m (M-f). Estimator: paired episode-cluster bootstrap over the EVAL
episodes, B = 1,000, 95 % percentile ("another draw of EPISODES"), plus the replicate floor |PM0r - PM0|.
Run: python pmap_analyze.py --arms-dir <dir with one sub-dir per arm> --diag-code <diagnostics code dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
CLS = {"drivable": 1, "lane": 2, "crosswalk": 3, "edge": 5}
BANDS = ["0_20", "20_40"]


def metrics(acc, w=None):
    """acc: stat -> [n_ep, 8, nb] ; w: episode weights -> {class: {band: {IoU0, IoU2, P10, R10}}}."""
    def pooled(s, c, bi):
        a = acc[s][:, c, bi]
        return float((a * w).sum() if w is not None else a.sum())
    out = {}
    for cn, c in CLS.items():
        out[cn] = {}
        for bi, b in enumerate(BANDS):
            P, G, I = pooled("pred", c, bi), pooled("gt", c, bi), pooled("inter", c, bi)
            u = P + G - I
            r = {"IoU0": I / u if u > 0 else None}
            for k in (2, 10):
                tpp, tpg = pooled(f"tpp{k}", c, bi), pooled(f"tpg{k}", c, bi)
                p = tpp / P if P > 0 else None
                rr = tpg / G if G > 0 else None
                f = (2 * p * rr / (p + rr) if (p is not None and rr is not None and p + rr > 0) else None)
                r[f"IoU{k}"] = None if f is None else f / (2 - f)
                r[f"P{k}"], r[f"R{k}"] = p, rr
            out[cn][b] = r
        # drivable 0-40 pooled IoU_0 (the PM3 non-inferiority read)
    P = sum(pooled("pred", 1, bi) for bi in range(len(BANDS)))
    G = sum(pooled("gt", 1, bi) for bi in range(len(BANDS)))
    I = sum(pooled("inter", 1, bi) for bi in range(len(BANDS)))
    out["drivable_0_40_IoU0"] = I / (P + G - I) if (P + G - I) > 0 else None
    return out


def get(m, path):
    v = m
    for p in path:
        v = v[p]
    return v


PATHS = {
    "edge_IoU2_0_20": ("edge", "0_20", "IoU2"), "edge_IoU2_20_40": ("edge", "20_40", "IoU2"),
    "lane_IoU2_0_20": ("lane", "0_20", "IoU2"), "lane_IoU2_20_40": ("lane", "20_40", "IoU2"),
    "edge_R10_0_20": ("edge", "0_20", "R10"), "lane_IoU0_0_20": ("lane", "0_20", "IoU0"),
    "edge_IoU0_0_20": ("edge", "0_20", "IoU0"), "crosswalk_IoU2_0_20": ("crosswalk", "0_20", "IoU2"),
    "drivable_0_40_IoU0": ("drivable_0_40_IoU0",),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.arms_dir)
    arms = sorted(p.name for p in d.iterdir() if (p / "eval_acc.pt").exists())
    R = {n: torch.load(d / n / "eval_acc.pt", map_location="cpu", weights_only=False) for n in arms}
    if "PM0" not in R:
        raise SystemExit("PM0 not banked")
    eps = R["PM0"]["episodes"]
    for n in arms:
        assert R[n]["episodes"] == eps, f"{n}: episode list differs"
        if R[n].get("eval_limit"):
            raise SystemExit(f"{n}: a SMOKE pack (eval_limit) is not a result")
    acc = {n: {s: R[n]["acc"][s].numpy() for s in R[n]["acc"]} for n in arms}
    rng = np.random.default_rng(0)
    E = len(eps)
    W = [np.bincount(rng.integers(0, E, E), minlength=E).astype(float) for _ in range(a.boot)]
    M = {n: metrics(acc[n]) for n in arms}
    for n in arms:
        reg = R[n]["reg_pred_drivable_vs_gt_edge"].numpy()             # [nb, 7] buckets 0,1,2,3-5,...
        M[n]["edge_cells_with_pred_drivable_boundary_within_0.2m"] = {
            b: float(reg[bi, :3].sum() / max(reg[bi].sum(), 1)) for bi, b in enumerate(BANDS)}
    paired = {}
    for n in arms:
        if n == "PM0":
            continue
        paired[n] = {}
        for key, path in PATHS.items():
            p1, p0 = get(M[n], path), get(M["PM0"], path)
            if p1 is None or p0 is None:
                paired[n][key] = None
                continue
            dd = []
            for w in W:
                x1, x0 = get(metrics(acc[n], w), path), get(metrics(acc["PM0"], w), path)
                if x1 is not None and x0 is not None:
                    dd.append(x1 - x0)
            dd = np.asarray(dd)
            paired[n][key] = {"point": p1 - p0, "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    floor = {k: (abs(v["point"]) if v else None) for k, v in paired.get("PM0r", {}).items()}

    def sep(n, k, thr):
        v = paired[n].get(k)
        if not v:
            return False
        fl = floor.get(k)
        return v["point"] >= thr and v["ci95"][0] > 0 and (fl is None or v["point"] > fl)

    verdict = {}
    if "PM1" in paired:
        verdict["PM1"] = {"PASS": bool(sep("PM1", "edge_IoU2_20_40", 0.03) or sep("PM1", "lane_IoU2_20_40", 0.03))}
    if "PM2" in paired:
        verdict["PM2"] = {"PASS": bool(sep("PM2", "edge_R10_0_20", 0.05))}
    if "PM3" in paired:
        ni = paired["PM3"]["drivable_0_40_IoU0"]
        verdict["PM3"] = {"PASS": bool(sep("PM3", "edge_IoU2_0_20", 0.03) and ni and ni["point"] >= -0.01)}
    if "PM4" in M:
        m4 = M["PM4"]
        verdict["PM4_dramatic"] = {"edge_IoU2_0_20_ge_0.33": (get(m4, PATHS["edge_IoU2_0_20"]) or 0) >= 0.33,
                                   "edge_IoU2_20_40_ge_0.18": (get(m4, PATHS["edge_IoU2_20_40"]) or 0) >= 0.18,
                                   "lane_IoU2_20_40_ge_0.38": (get(m4, PATHS["lane_IoU2_20_40"]) or 0) >= 0.38}
        verdict["PM4_dramatic"]["PASS"] = all(verdict["PM4_dramatic"].values())
    mf = {}
    if "R-s4zero" in paired and floor:
        bad = [k for k, v in paired["R-s4zero"].items() if v and floor.get(k) is not None and abs(v["point"]) > floor[k]]
        mf["R-s4zero"] = {"keys_outside_replicate_floor": bad, "failed_as_required": not bad}
    if "R-s4shuf" in paired:
        his = [paired["R-s4shuf"][k]["ci95"][1] for k in ("edge_IoU2_20_40", "lane_IoU2_20_40") if paired["R-s4shuf"][k]]
        # both registered keys must be readable -- an empty list would make all() pass vacuously
        mf["R-s4shuf"] = {"upper_ci": his, "failed_as_required": len(his) == 2 and all(h <= 0.01 for h in his)}
    if "R-gtshift" in paired and paired["R-gtshift"]["lane_IoU0_0_20"]:
        v = paired["R-gtshift"]["lane_IoU0_0_20"]["point"]
        mf["R-gtshift"] = {"d_lane_IoU0_0_20": v, "failed_as_required": v <= -0.05}
    void = [k for k, v in mf.items() if not v["failed_as_required"]]
    out = {"estimator": f"paired episode-cluster bootstrap B={a.boot}, 95 % percentile; + replicate floor |PM0r-PM0|",
           "arms": M, "paired_vs_PM0": paired, "replicate_floor": floor, "verdict": verdict, "must_fail": mf,
           "VOID": bool(void), "void_reasons": void, "n_episodes": E}
    p = Path(a.out) if a.out else PKG / "raw" / "pmap_result.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "must_fail": mf, "VOID": bool(void)}, indent=1, default=float))


if __name__ == "__main__":
    main()
