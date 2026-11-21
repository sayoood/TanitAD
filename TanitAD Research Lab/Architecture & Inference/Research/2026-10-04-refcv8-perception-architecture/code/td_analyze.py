"""WP-D priority 1 -- refcv7-r101-s0 training dynamics of the MAP and BOX heads from the run's own metrics.jsonl.

Input : the run record banked by the refcv8 data audit (D1)
        TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/raw/
        refcv7_run_records/metrics.jsonl  (6,150 rows: 1,008 train rows every 50 steps, 101 in-run eval rows every
        500 steps on a FIXED 128-window / 80-episode eval set, 5,039 conflict-detector rows every 10 steps).
Output: raw/td_curves.json (per-metric curves), raw/td_summary.json (window means, late slopes, gain fractions),
        printed tables. CPU only, stdlib + numpy.

Row kinds are identified by CONTENT (key presence), never by position: train = has 'traj' and 'lr' and no
'eval_loss'; eval = has 'eval_loss'; cd = has 'cd_ratio'.

Estimators (stated with every number):
  * window mean +- SD over in-run eval rows inside [c - 2,500, c + 2,500) steps (5 rows; FIXED windows, so the spread
    is MODEL-STATE fluctuation between checkpoints, not episode sampling);
  * late slope: OLS of metric on step over eval rows with step >= 30,000 (n = 41), reported per 10k steps with its
    OLS standard error (rows are serially correlated, so the SE is OPTIMISTIC -- stated);
  * gain fraction: (m_s - m_first) / (m_last - m_first) on the 5-row centred moving average.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TREE = PKG.parents[3]          # .../TanitAD
SRC = (TREE / "TanitAD Research Lab" / "Data Engineering" / "Research" / "2026-10-04-refcv8-data-audit" /
       "D1_label_census" / "raw" / "refcv7_run_records" / "metrics.jsonl")
OUT = PKG / "raw"

CLASSES = ["drivable", "sidewalk", "lane", "crosswalk", "edge", "arrow", "hatched", "nocls"]
BANDS = ["0_20", "20_40", "40_60", "60_80", "80_100"]
HEADS = ["box3d", "agent"]
RANGES = ["0_20", "20_40", "40_60", "all"]
THRS = ["0p5", "1", "2", "4"]
MODULES = ["trunk", "planner", "tac_decoder", "box_memory", "box_decoder", "bev_pool", "mh_lift", "mh_encoder",
           "mh_refine", "mh_near", "mh_near_refine", "mh_trunk_s8_stage"]
STAGES = ["stem", "stage_layer1", "stage_layer2", "stage_layer3", "stage_layer4", "fuse", "heads"]


def load(path: Path):
    tr, ev, cd = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "eval_loss" in r:
                ev.append(r)
            elif "cd_ratio" in r:
                cd.append(r)
            elif "traj" in r and "lr" in r:
                tr.append(r)
    for lst in (tr, ev, cd):
        lst.sort(key=lambda r: r["step"])
    return tr, ev, cd


def f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def pooled_iou(r, prefix, cls, bands, inter="inter", union="union"):
    i = sum(f(r.get(f"{prefix}map_hires_{inter}_{cls}_{b}", 0.0)) for b in bands)
    u = sum(f(r.get(f"{prefix}map_hires_{union}_{cls}_{b}", 0.0)) for b in bands)
    return i / u if u > 0 else float("nan")


def eval_metrics(r):
    m = {"step": r["step"]}
    for h in HEADS:
        p = f"eval_{h}_"
        m[f"{h}.ap2m"] = f(r.get(p + "ap2m"))
        for t in THRS:
            for rg in RANGES:
                m[f"{h}.mAP{t}.{rg}"] = f(r.get(f"{p}det_map{t}_{rg}"))
        for k in ("auroc_objectness", "auroc_matched", "centre_err_p50", "calib_pr_gate", "calib_prec",
                  "presence", "presence_layer0", "presence_layer1", "presence_layer2", "layer0", "layer1", "layer2",
                  "centre", "size", "yaw", "cls", "cls_acc_tp", "n_presence_exempt", "n_presence_matched",
                  "n_pos", "presence_frac_confident"):
            m[f"{h}.{k}"] = f(r.get(p + k))
    for c in CLASSES:
        m[f"map.iou_pc.{c}.all"] = pooled_iou(r, "eval_", c, BANDS)
        m[f"map.iou_raw.{c}.all"] = pooled_iou(r, "eval_", c, BANDS, "interraw", "unionraw")
        for b in BANDS:
            m[f"map.iou_raw.{c}.{b}"] = pooled_iou(r, "eval_", c, [b], "interraw", "unionraw")
    m["map.loss"] = f(r.get("eval_map_hires"))
    m["traj"] = f(r.get("eval_traj"))
    return m


def train_metrics(r):
    m = {"step": r["step"], "lr": f(r["lr"]), "loss": f(r["loss"]), "traj": f(r["traj"]),
         "box3d": f(r["box3d"]), "map_hires": f(r["map_hires"]), "data_epoch": f(r.get("data_epoch"))}
    for h in HEADS:
        for k in ("presence", "centre", "size", "yaw", "cls", "layer0", "layer1", "layer2"):
            m[f"{h}.{k}"] = f(r.get(f"{h}_{k}"))
    for mod in MODULES:
        s, n = f(r.get(f"ga_{mod}")), f(r.get(f"ga_{mod}_n"))
        m[f"ga.{mod}.sum"] = s
        m[f"ga.{mod}.per_param"] = s / n if n and n > 0 else float("nan")
    # map per-class loss share (summed over bands)
    for c in CLASSES:
        m[f"map.lshare.{c}"] = sum(f(r.get(f"map_hires_lshare_{c}_{b}", 0.0)) for b in BANDS)
        m[f"map.gn.{c}"] = math.sqrt(sum(f(r.get(f"map_hires_gn_{c}_{b}", 0.0)) ** 2 for b in BANDS))
    return m


def cd_metrics(r):
    m = {"step": r["step"]}
    for s in STAGES:
        pre = "cd_" if s == "trunk" else f"cd_{s}_"
        if s == "heads":
            pre = "cd_heads_"
        m[f"cd.{s}.cos"] = f(r.get(pre + "cos"))
        ga, gt = f(r.get(pre + "gn_aux")), f(r.get(pre + "gn_traj"))
        m[f"cd.{s}.gn_aux"] = ga
        m[f"cd.{s}.gn_traj"] = gt
        m[f"cd.{s}.aux_share"] = ga / (ga + gt) if (ga + gt) > 0 else float("nan")
    ga, gt = f(r.get("cd_gn_aux")), f(r.get("cd_gn_traj"))
    m["cd.trunk.cos"] = f(r.get("cd_cos"))
    m["cd.trunk.gn_aux"], m["cd.trunk.gn_traj"] = ga, gt
    m["cd.trunk.aux_share"] = ga / (ga + gt) if (ga + gt) > 0 else float("nan")
    return m


def window_mean(steps, vals, c, half=2500):
    sel = [v for s, v in zip(steps, vals) if c - half <= s < c + half and not math.isnan(v)]
    if not sel:
        return None
    a = np.asarray(sel)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def ols(steps, vals, lo, hi=10 ** 9):
    pts = [(s, v) for s, v in zip(steps, vals) if lo <= s <= hi and not math.isnan(v)]
    if len(pts) < 5:
        return None
    x = np.asarray([p[0] for p in pts], float) / 1e4
    y = np.asarray([p[1] for p in pts], float)
    X = np.stack([np.ones_like(x), x], 1)
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    s2 = float(((y - yhat) ** 2).sum() / (len(y) - 2))
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"slope_per_10k": float(beta[1]), "se": float(math.sqrt(cov[1, 1])), "n": len(y),
            "t": float(beta[1] / math.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")}


def movavg(vals, k=5):
    a = np.asarray(vals, float)
    out = np.full_like(a, np.nan)
    h = k // 2
    for i in range(len(a)):
        w = a[max(0, i - h): i + h + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return out


def gain_fraction(steps, vals):
    sm = movavg(vals)
    ok = ~np.isnan(sm)
    if ok.sum() < 3:
        return None
    first, last = sm[ok][0], sm[ok][-1]
    if abs(last - first) < 1e-12:
        return None
    out = {}
    for s in (5000, 10000, 20000, 30000, 40000):
        i = int(np.argmin(np.abs(np.asarray(steps) - s)))
        out[str(s)] = float((sm[i] - first) / (last - first))
    # first step at which the smoothed curve reaches 90 % of its total gain
    g = (sm - first) / (last - first)
    hit = [s for s, v in zip(steps, g) if not math.isnan(v) and v >= 0.9]
    out["step_90pct"] = int(hit[0]) if hit else None
    return out


def main():
    tr, ev, cd = load(SRC)
    print(f"rows: train {len(tr)} eval {len(ev)} cd {len(cd)}")
    E = [eval_metrics(r) for r in ev]
    T = [train_metrics(r) for r in tr]
    C = [cd_metrics(r) for r in cd]
    lr_at = {t["step"]: t["lr"] for t in T}
    curves = {"eval": E, "train": T, "cd_every50": [c for c in C if c["step"] % 50 == 0]}
    (OUT / "td_curves.json").write_text(json.dumps(curves, allow_nan=True), encoding="utf-8")

    centres = [5000, 10000, 20000, 30000, 40000, 47900]
    summ = {"source": str(SRC.relative_to(TREE)).replace("\\", "/"), "n_train_rows": len(tr), "n_eval_rows": len(ev),
            "n_cd_rows": len(cd), "estimators": __doc__.split("Estimators")[1].strip(), "eval": {}, "train": {},
            "cd": {}}
    es = [m["step"] for m in E]
    for k in E[0]:
        if k == "step":
            continue
        vals = [m[k] for m in E]
        summ["eval"][k] = {"windows": {str(c): window_mean(es, vals, c) for c in centres},
                           "late_slope_30k_50k": ols(es, vals, 30000),
                           "mid_slope_15k_30k": ols(es, vals, 15000, 30000),
                           "gain_fraction": gain_fraction(es, vals),
                           "first": vals[0], "last": vals[-1], "max": float(np.nanmax(vals)) if
                           not all(math.isnan(v) for v in vals) else None,
                           "argmax_step": int(es[int(np.nanargmax(vals))]) if not all(math.isnan(v) for v in vals)
                           else None}
    ts = [m["step"] for m in T]
    for k in T[0]:
        if k == "step":
            continue
        vals = [m[k] for m in T]
        summ["train"][k] = {"windows": {str(c): window_mean(ts, vals, c) for c in centres},
                            "late_slope_30k_50k": ols(ts, vals, 30000)}
    cs = [m["step"] for m in C]
    for k in C[0]:
        if k == "step":
            continue
        vals = [m[k] for m in C]
        summ["cd"][k] = {"windows": {str(c): window_mean(cs, vals, c) for c in centres}}
    summ["lr_at"] = {str(c): lr_at.get(c) for c in (2000, 5000, 10000, 20000, 30000, 35000, 40000, 45000, 50000,
                                                       50400)}
    (OUT / "td_summary.json").write_text(json.dumps(summ, indent=1, allow_nan=True), encoding="utf-8")
    print("wrote", OUT / "td_summary.json")


if __name__ == "__main__":
    sys.exit(main())
