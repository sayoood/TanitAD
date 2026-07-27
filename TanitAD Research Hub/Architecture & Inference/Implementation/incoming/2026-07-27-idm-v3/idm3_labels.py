"""IDM v3 — PHASE 2: fix the labels, and the 0-GPU test of the geometry hypothesis.

Runs four things, in the order the brief prioritises them:

  1. REPRODUCE A0 on the exact v2 substrate (68/36 episode split, 4,195 val
     windows). Nothing downstream may be quoted until this matches
     `IDM_V2_RESULTS.md` §3.
  2. MEASURE the comma2k19 heading defect as a function of SPEED, and pick the
     admissibility threshold `v_min` from the data instead of asserting one.
  3. REPAIR the heading (hold the last observable direction through standstill)
     and re-derive yaw_rate; report BEFORE/AFTER for EVERY channel, per corpus.
  4. The 0-GPU geometry test: does A0's per-clip speed bias — the 56.6 % of its
     MSE that `IDM_DIAGNOSIS.md` §5.3 shows is a per-clip level error — correlate
     with the clip's MEASURED camera height? Equation (1) of `idm3_geom.py` says
     v = (f*h) * PHI, so a head that does not know h under-predicts on tall
     mounts and over-predicts on short ones. This is the cheapest discriminating
     experiment for the PI's hypothesis and it costs no GPU.

Estimator: `taniteval.ci.(paired_)episode_cluster_bootstrap`, unit = episode,
B = 2000. `overlapping_holdout_se` is never called.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L          # noqa: E402
import idm_head as ih         # noqa: E402
import idm3_geom as GEO       # noqa: E402
from taniteval import ci as tci  # noqa: E402

DEV = "cuda"
KBUILD = 8
DT = ih.DT


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# 2/3. the comma2k19 heading repair                                            #
# --------------------------------------------------------------------------- #
def heading_repair(poses: torch.Tensor, v_min: float):
    """comma2k19 heading = arctan2(enu_v_north, enu_v_east) is UNDEFINED when the
    ENU velocity vanishes (`stack/tanitad/data/comma2k19.py:172`). Repair by
    holding the last OBSERVABLE direction through the standstill, operating on
    the unit direction vector so no 2*pi wrap can be introduced.

    Returns (yaw_fixed [T], observable [T] bool). Frames before the first
    observable sample inherit the first observable direction (back-fill)."""
    yaw = poses[:, 2].numpy().astype(np.float64).copy()
    v = poses[:, 3].numpy().astype(np.float64)
    obs = v >= v_min
    if not obs.any():
        return torch.from_numpy(yaw).float(), torch.from_numpy(obs)
    ux, uy = np.cos(yaw), np.sin(yaw)
    idx = np.where(obs, np.arange(len(yaw)), -1)
    np.maximum.accumulate(idx, out=idx)            # forward-fill index
    first = int(np.argmax(obs))
    idx[idx < 0] = first                           # back-fill the head
    return torch.from_numpy(np.arctan2(uy[idx], ux[idx])).float(), torch.from_numpy(obs)


def yaw_rate_from(yaw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ih.wrap_to_pi(yaw[t + 1] - yaw[t - 1]) / (2.0 * DT)


def speed_binned_yaw_audit(tags, edges=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0)):
    """Per speed bin: how noisy is the derived yaw_rate, and what fraction of
    frames is physically impossible? This is what SETS v_min."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w, imp, n = [], 0, 0
        for tag in tags:
            d = L.load_ep(tag)
            po = d["poses"].float()
            T = po.shape[0]
            t = torch.arange(1, T - 1)
            yr = yaw_rate_from(po[:, 2], t).numpy()
            v = po[t, 3].numpy()
            m = (v >= lo) & (v < hi)
            if m.sum() == 0:
                continue
            w.append(yr[m])
            imp += int((np.abs(yr[m]) > 1.5).sum())
            n += int(m.sum())
        if not w:
            continue
        a = np.concatenate(w)
        rows.append({"v_lo": lo, "v_hi": hi, "n": n,
                     "frac_of_corpus": None,
                     "yaw_std": float(a.std()),
                     "yaw_mad": float(np.median(np.abs(a - np.median(a)))),
                     "p999_abs": float(np.percentile(np.abs(a), 99.9)),
                     "max_abs": float(np.abs(a).max()),
                     "n_impossible_gt1p5": imp,
                     "frac_impossible": imp / max(n, 1)})
    tot = sum(r["n"] for r in rows)
    for r in rows:
        r["frac_of_corpus"] = r["n"] / max(tot, 1)
    return rows


# --------------------------------------------------------------------------- #
# 1. A0 reproduction                                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def a0_predict(va, batch=1024):
    d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
    h = ih.IDMHead(**d["config"]["head_kwargs"]).to(DEV)
    h.load_state_dict(d["state_dict"])
    h.eval()
    Z = va["Z"][:, KBUILD - 4:KBUILD + 5].to(DEV).float()
    S, T = [], []
    for i in range(0, Z.shape[0], batch):
        o = h(Z[i:i + batch])
        S.append(o["scalars"].cpu())
        T.append(o["traj"].cpu())
    del Z, h
    torch.cuda.empty_cache()
    return {"S": torch.cat(S).numpy().astype(np.float64),
            "Traj": torch.cat(T).numpy().astype(np.float64)}


def per_channel(P, G, dom, eid, tag=""):
    out = {}
    for j, nm in enumerate(L.SCALARS):
        m = L.chan_metrics(P[:, j], G[:, j])
        m["per_domain"] = {d: L.chan_metrics(P[dom == d, j], G[dom == d, j])
                           for d in ("pai", "cm")}
        out[nm] = m
    return out


# --------------------------------------------------------------------------- #
# 4. the 0-GPU geometry test                                                   #
# --------------------------------------------------------------------------- #
def clip_bias_vs_geometry(P, G, eid, dom):
    """Per clip: mean(pred_speed - gt_speed) — the level term — against the
    clip's MEASURED camera height. Equation (1): v = (f*h)*PHI, so a head blind
    to h emits v_hat = (f*h_bar)*PHI = v * h_bar/h  =>  the RELATIVE bias
    (v_hat-v)/v should fall as 1/h. Report both the absolute and the relative
    form, and the CV of h so the effect size is interpretable."""
    tab = GEO.load_table()
    rows = []
    for t in sorted(set(eid)):
        m = eid == t
        g = GEO.geom_for_tag(t, tab)
        vg, vp = G[m, 0], P[m, 0]
        mu = float(vg.mean())
        if mu < 1.0:                     # a parked clip has no scale to get wrong
            continue
        rows.append({"tag": t, "dom": t.split("_")[0], "n": int(m.sum()),
                     "v_mean": mu, "bias": float((vp - vg).mean()),
                     "rel_bias": float((vp - vg).mean() / mu),
                     "cam_h": g["cam_h_m"], "f_eff": g["f_eff_px"],
                     "metric_gain": g["f_eff_px"] * g["cam_h_m"],
                     "rig": g["rig"]})
    return rows


def corr_report(rows, xkey, ykey, sub=None):
    r = [x for x in rows if sub is None or x["dom"] == sub]
    if len(r) < 4:
        return None
    x = np.array([v[xkey] for v in r], float)
    y = np.array([v[ykey] for v in r], float)
    if x.std() < 1e-9:
        return {"n": len(r), "note": "x is constant in this subset"}
    sl, ic = np.polyfit(x, y, 1)
    pr = float(np.corrcoef(x, y)[0, 1])
    rs = L.spearman(x, y)
    # episode-cluster bootstrap over CLIPS on the Pearson r
    idx = np.arange(len(r), dtype=np.float64)

    def _r(i):
        i = i.astype(np.int64)
        return float(np.corrcoef(x[i], y[i])[0, 1]) if x[i].std() > 1e-9 else 0.0
    _r.__name__ = "pearson"
    ci = tci.episode_cluster_bootstrap(idx, np.array([v["tag"] for v in r]),
                                       reduce=_r, n_boot=2000, seed=0)
    return {"n": len(r), "pearson": pr, "spearman": rs,
            "slope": float(sl), "intercept": float(ic),
            "ci_lo": float(ci["lo"]), "ci_hi": float(ci["hi"]),
            "x_mean": float(x.mean()), "x_std": float(x.std()),
            "x_cv": float(x.std() / abs(x.mean())) if abs(x.mean()) > 0 else None,
            "y_mean": float(y.mean()), "y_std": float(y.std())}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/idm3/out/labels_v3.json")
    a = ap.parse_args()
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    tr_tags, va_tags = L.split_tags()
    va = L.build_set(va_tags, k=KBUILD, stride=2, want_seq=True)
    va["Akin"] = (va["Vseq"][:, KBUILD + 1] - va["Vseq"][:, KBUILD - 1]) / (2 * DT)
    G = va["S"].numpy().astype(np.float64)
    dom, eid = va["dom"], va["eid"]
    log(f"val {tuple(va['Z'].shape)} windows over {len(va_tags)} eps "
        f"(pai {int((dom=='pai').sum())} / cm {int((dom=='cm').sum())})")
    res["substrate"] = {"n_val_windows": int(G.shape[0]),
                        "n_val_eps": len(va_tags), "train_eps": len(tr_tags),
                        "n_pai": int((dom == "pai").sum()),
                        "n_cm": int((dom == "cm").sum())}

    # ---- 1. reproduce A0 ---------------------------------------------------
    P = a0_predict(va)
    a0 = per_channel(P["S"], G, dom, eid)
    res["A0_reproduction"] = a0
    log("A0 REPRO  speed R2 %+.4f MAE %.3f | yaw R2 %+.4f (pai %+.4f cm %+.4f) | "
        "steer %+.4f | accel %+.4f" % (
            a0["speed"]["r2"], a0["speed"]["mae"], a0["yaw_rate"]["r2"],
            a0["yaw_rate"]["per_domain"]["pai"]["r2"],
            a0["yaw_rate"]["per_domain"]["cm"]["r2"],
            a0["steer"]["r2"], a0["long_accel"]["r2"]))

    # ---- 2. the comma heading defect vs speed ------------------------------
    cm_tags = [t for t in va_tags + tr_tags if t.startswith("cm_")]
    pai_tags = [t for t in va_tags + tr_tags if t.startswith("pai_")]
    res["yaw_audit_by_speed"] = {"cm": speed_binned_yaw_audit(cm_tags),
                                 "pai": speed_binned_yaw_audit(pai_tags)}
    log("speed-binned yaw audit (comma):")
    for r in res["yaw_audit_by_speed"]["cm"]:
        log("   v [%4.1f,%5.1f)  n %6d (%.3f)  std %.4f  mad %.4f  "
            "p99.9 %.3f  max %.2f  impossible %.4f%%" % (
                r["v_lo"], r["v_hi"], r["n"], r["frac_of_corpus"], r["yaw_std"],
                r["yaw_mad"], r["p999_abs"], r["max_abs"],
                100 * r["frac_impossible"]))

    # ---- 3. repair, then BEFORE/AFTER on every channel ---------------------
    res["repair"] = {}
    for v_min in (0.0, 0.25, 0.5, 1.0, 2.0):
        Gf = G.copy()
        n_ch, n_unobs = 0, 0
        for tag in sorted(set(eid)):
            d = L.load_ep(tag)
            po = d["poses"].float()
            if tag.startswith("cm_") and v_min > 0:
                yaw_f, obs = heading_repair(po, v_min)
                n_unobs += int((~obs).sum())
            else:
                yaw_f = po[:, 2]
            m = eid == tag
            t = va["tcen"][torch.from_numpy(m)]
            yr = yaw_rate_from(yaw_f, t).numpy().astype(np.float64)
            n_ch += int((np.abs(yr - Gf[m, 1]) > 1e-9).sum())
            Gf[m, 1] = yr
        ch = per_channel(P["S"], Gf, dom, eid)
        # admissibility: physically impossible labels are inadmissible, always
        adm = np.abs(Gf[:, 1]) <= 1.5
        chA = {"n_admissible": int(adm.sum()), "n_dropped": int((~adm).sum())}
        chA["yaw_rate"] = L.chan_metrics(P["S"][adm, 1], Gf[adm, 1])
        chA["per_domain"] = {d_: L.chan_metrics(P["S"][adm & (dom == d_), 1],
                                                Gf[adm & (dom == d_), 1])
                             for d_ in ("pai", "cm")}
        res["repair"][f"v_min_{v_min}"] = {
            "n_windows_changed": n_ch, "n_frames_unobservable": n_unobs,
            "channels": ch, "admissible_only": chA}
        log("repair v_min=%.2f  changed %5d win  yaw R2 %+.4f -> adm %+.4f "
            "(drop %d) | cm %+.4f | pai %+.4f" % (
                v_min, n_ch, ch["yaw_rate"]["r2"], chA["yaw_rate"]["r2"],
                chA["n_dropped"], ch["yaw_rate"]["per_domain"]["cm"]["r2"],
                ch["yaw_rate"]["per_domain"]["pai"]["r2"]))

    # ---- long_accel: the CAN label vs the kinematic one --------------------
    Akin = va["Akin"].numpy().astype(np.float64)
    res["long_accel"] = {
        "vs_can_label": L.chan_metrics(P["S"][:, 3], G[:, 3]),
        "vs_kinematic": L.chan_metrics(P["S"][:, 3], Akin),
        "per_domain_can": {d: L.chan_metrics(P["S"][dom == d, 3], G[dom == d, 3])
                           for d in ("pai", "cm")},
        "per_domain_kin": {d: L.chan_metrics(P["S"][dom == d, 3], Akin[dom == d])
                           for d in ("pai", "cm")},
        "label_vs_kinematic_corr": {
            d: float(np.corrcoef(G[dom == d, 3], Akin[dom == d])[0, 1])
            for d in ("pai", "cm")},
        "target_stats": {d: {"std": float(G[dom == d, 3].std()),
                             "mad": float(np.median(np.abs(
                                 G[dom == d, 3] - np.median(G[dom == d, 3])))),
                             "mean": float(G[dom == d, 3].mean()),
                             "kurtosis": float(
                                 ((G[dom == d, 3] - G[dom == d, 3].mean()) ** 4).mean() /
                                 max(G[dom == d, 3].std() ** 4, 1e-12))}
                         for d in ("pai", "cm")},
    }

    # ---- 4. the 0-GPU geometry test ---------------------------------------
    rows = clip_bias_vs_geometry(P["S"], G, eid, dom)
    res["geometry_test"] = {
        "per_clip": rows,
        "ALL_bias_vs_cam_h": corr_report(rows, "cam_h", "bias"),
        "ALL_relbias_vs_cam_h": corr_report(rows, "cam_h", "rel_bias"),
        "PAI_bias_vs_cam_h": corr_report(rows, "cam_h", "bias", sub="pai"),
        "PAI_relbias_vs_cam_h": corr_report(rows, "cam_h", "rel_bias", sub="pai"),
        "PAI_relbias_vs_metric_gain": corr_report(rows, "metric_gain",
                                                  "rel_bias", sub="pai"),
        "PAI_relbias_vs_v_mean": corr_report(rows, "v_mean", "rel_bias", sub="pai"),
        "PAI_bias_vs_v_mean": corr_report(rows, "v_mean", "bias", sub="pai"),
    }
    for k in ("PAI_bias_vs_cam_h", "PAI_relbias_vs_cam_h", "PAI_relbias_vs_v_mean"):
        r = res["geometry_test"][k]
        if r and "pearson" in r:
            log("GEOM %-26s n=%2d  r=%+.3f [%.3f,%.3f]  rho=%+.3f  slope=%+.4f"
                % (k, r["n"], r["pearson"], r["ci_lo"], r["ci_hi"],
                   r["spearman"], r["slope"]))

    L.jdump(res, a.out)


if __name__ == "__main__":
    main()
