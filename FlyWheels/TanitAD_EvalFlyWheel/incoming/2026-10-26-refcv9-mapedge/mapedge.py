"""mapedge.py -- PREREG_MAPEDGE.md Q1: does the inner boundary of refcv8's OWN drivable plane recover the GT `edge` cells?  CPU, 0 GPU.

    python mapedge.py --rast F:/refcv9_mapsel/rast_s0 --out <dir>      # the sealed run
    python mapedge.py --selftest --out <dir>                           # K1 / K2 analytic worlds only (reads no eval raster)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_mapsel/code")
from mapsel_lib import dilate  # noqa: E402

EDGE, NOCLS, UNSEEN = 5, 0, 255
DRV_CODES = (1, 2, 3, 4, 6)
TOL = 3                      # 0.3 m (MAPSEL EDGE_DIL)
ROWS = 600                   # x in [0, 60) m
BANDS = {"0-20m": (0, 200), "20-40m": (200, 400), "40-60m": (400, 600)}
MIN_CELLS = 20
ARMS = ("OWN-CODE5", "OWN-BND", "DER-BND", "GT-BND")
NAME = re.compile(r"e(\d+)_w(\d+)\.npz$")


def boundary(d):
    """inner boundary of a bool plane: d & dilate(~d, 1), outermost raster row/col cleared."""
    d = torch.as_tensor(d).bool()
    b = d & dilate(~d, 1)
    b[0, :] = False
    b[-1, :] = False
    b[:, 0] = False
    b[:, -1] = False
    return b


def gt_planes(gt):
    g = torch.as_tensor(gt)
    J = (g != NOCLS) & (g != UNSEEN)
    E = g == EDGE
    G = torch.zeros_like(J)
    for c in DRV_CODES:
        G |= g == c
    gbnd = G & dilate(J & ~G, 1)          # GT-BND: drivable cells next to a JUDGED non-drivable cell (the sealed "restricted to J")
    gbnd[0, :] = gbnd[-1, :] = False
    gbnd[:, 0] = gbnd[:, -1] = False
    return J, E, gbnd


def window_stats(X, E, J, Edil):
    """per-window recall / precision of the bool plane X against GT edges E (region rows 0..ROWS-1); None where undefined."""
    Xr, Er, Jr, Edr = X[:ROWS], E[:ROWS], J[:ROWS], Edil[:ROWS]
    Xd = dilate(X, TOL)[:ROWS]
    ne = int(Er.sum())
    out = {"rec": (float((Er & Xd).sum()) / ne) if ne >= MIN_CELLS else None}
    xj = int((Xr & Jr).sum())
    out["prec"] = (float((Xr & Jr & Edr).sum()) / xj) if xj >= MIN_CELLS else None
    for nm, (a, b) in BANDS.items():
        nb = int(Er[a:b].sum())
        out[f"rec_{nm}"] = (float((Er[a:b] & Xd[a:b]).sum()) / nb) if nb >= MIN_CELLS else None
    nx = int(Xr.sum())
    out["n_x"] = nx
    out["share_on_nocls"] = None
    out["share_on_unseen"] = None
    return out


def boot_mean(vals, eps, n_boot=2000, seed=0):
    """pooled window mean, episode-cluster bootstrap (resample episodes with all their windows)."""
    v, e = np.asarray(vals, float), np.asarray(eps)
    ue, inv = np.unique(e, return_inverse=True)
    s, c = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    pt = s.sum() / c.sum()
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ii = rng.integers(0, len(ue), len(ue))
        bs[i] = s[ii].sum() / c[ii].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"point": float(pt), "lo": float(lo), "hi": float(hi), "n_windows": int(len(v)), "n_episodes": int(len(ue)),
            "separated": bool(lo > 0 or hi < 0)}


def paired(a, b, eps, n_boot=2000, seed=0):
    m = [(x - y, e) for x, y, e in zip(a, b, eps) if x is not None and y is not None]
    if not m:
        return None
    return boot_mean([d for d, _ in m], [e for _, e in m], n_boot, seed)


def selftest():
    H, W = 1000, 600
    ys = -30 + 0.1 * (np.arange(W) + 0.5)
    road = np.abs(ys) <= 3.5
    pdrv = np.where(road[None, :], 255, 0).astype(np.uint8).repeat(H, 0)
    jr = np.where(road)[0]
    lo, hi = jr.min() - 1, jr.max() + 1                     # first non-drivable column on each side
    gt = np.where(road[None, :], 1, 7).astype(np.uint8).repeat(H, 0)
    gt[:, [lo, hi]] = EDGE
    J, E, _ = gt_planes(gt)
    Edil = dilate(E, TOL)
    s1 = window_stats(boundary(pdrv >= 128), E, J, Edil)
    shift = 10                                               # 1.0 m lateral shift of the road against the GT
    s2 = window_stats(boundary(np.roll(pdrv, shift, axis=1) >= 128), E, J, Edil)
    k1 = bool(s1["rec"] == 1.0 and s1["prec"] == 1.0)
    k2 = bool(s2["rec"] == 0.0)
    k3c = paired([0.1, 0.4, 0.7], [0.1, 0.4, 0.7], [0, 1, 2])
    k3 = bool(k3c["point"] == 0 and k3c["lo"] == 0 and k3c["hi"] == 0)
    return {"K1_analytic_world": {"pass": k1, "rec": s1["rec"], "prec": s1["prec"]},
            "K2_shifted_world": {"pass": k2, "rec": s2["rec"]},
            "K3_identity_paired_zero": {"pass": k3}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rast", default="F:/refcv9_mapsel/rast_s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = hashlib.sha256((HERE / "PREREG_MAPEDGE.md").read_bytes()).hexdigest()
    want = (HERE / "PREREG_MAPEDGE.sha256").read_text().split()[0]
    am1 = hashlib.sha256((HERE / "PREREG_MAPEDGE_AMENDMENT1.md").read_bytes()).hexdigest()
    if am1 != (HERE / "PREREG_MAPEDGE_AMENDMENT1.sha256").read_text().split()[0]:
        raise SystemExit("[mapedge] PREREG_MAPEDGE_AMENDMENT1.md does not match its seal: refusing")
    ctl = selftest()
    res = {"tool": "mapedge.py", "prereg_sha256": pre, "amendment1_sha256": am1, "seal_ok": pre == want, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "controls": ctl, "estimator": "pooled window mean; episode-cluster bootstrap over episodes, B 2000, seed 0 -- 'another draw of EPISODES' only; "
                                         "one training seed; the map head is deterministic given the frames",
           "tier": "perception instrument (map head vs SAM3 GT raster), eval139 per-frame windows; not a planning or driving metric"}
    if a.selftest:
        json.dump(res, open(out / "MAPEDGE_SELFTEST.json", "w"), indent=1)
        print(json.dumps(ctl, indent=1))
        return 0 if all(v["pass"] for v in ctl.values()) else 1
    if pre != want:
        raise SystemExit("[mapedge] PREREG_MAPEDGE.md does not match its seal: refusing")
    files = sorted(Path(a.rast).glob("e*_w*.npz"))
    keys = []
    for f in files:
        m = NAME.search(f.name)
        keys.append((int(m.group(1)), int(m.group(2)), f))
    keys.sort()
    eps = sorted({k[0] for k in keys})
    by_ep = {e: [k for k in keys if k[0] == e] for e in eps}
    donor = {}
    for i, e in enumerate(eps):
        de = eps[(i + 1) % len(eps)]
        for p, k in enumerate(by_ep[e]):
            donor[(k[0], k[1])] = by_ep[de][min(p, len(by_ep[de]) - 1)]
    ctl["K4_donor_other_episode"] = {"pass": bool(len(eps) > 1 and all(d[0] != k[0] for k, d in donor.items())), "n_episodes": len(eps)}
    rows, n_bad, n_const, n_nogt, eps_nogt = [], 0, 0, 0, set()
    for (e, w, f) in keys:
        z = np.load(f)
        own_ok = all(p in z.files and z[p].shape == (1000, 600) and z[p].dtype == np.uint8 for p in ("pdrv", "pedge", "psw", "code"))
        if not own_ok or ("gt" in z.files and (z["gt"].shape != (1000, 600) or z["gt"].dtype != np.uint8)):
            n_bad += 1
            continue
        if "gt" not in z.files:                    # AMENDMENT 1 A1.1: no GT map for this window -> outside the population (still a donor)
            n_nogt += 1
            eps_nogt.add(e)
            continue
        if z["pdrv"].min() == z["pdrv"].max():
            n_const += 1
        dz = np.load(donor[(e, w)][2])
        J, E, gbnd = gt_planes(z["gt"])
        Edil = dilate(E, TOL)
        X = {"OWN-CODE5": torch.as_tensor(z["code"] == EDGE), "OWN-BND": boundary(z["pdrv"] >= 128), "DER-BND": boundary(dz["pdrv"] >= 128), "GT-BND": gbnd}
        r = {"ep": e, "ws": w}
        g = torch.as_tensor(z["gt"])[:ROWS]
        for nm, x in X.items():
            st = window_stats(x, E, J, Edil)
            xr = x[:ROWS]
            if st["n_x"]:
                st["share_on_nocls"] = float((xr & (g == NOCLS)).sum()) / st["n_x"]
                st["share_on_unseen"] = float((xr & (g == UNSEEN)).sum()) / st["n_x"]
            r[nm] = st
        rows.append(r)
    ctl["K5_decode"] = {"pass": bool(n_bad == 0 and len(rows) > 0 and n_const <= 0.05 * len(rows)), "n_windows": len(rows), "n_bad_shape": n_bad, "n_constant_pdrv": n_const,
                        "n_windows_without_gt": n_nogt, "n_episodes_touched_without_gt": len(eps_nogt)}
    ep_of = [r["ep"] for r in rows]
    summ = {}
    for nm in ARMS:
        summ[nm] = {}
        for met in ("rec", "prec", "rec_0-20m", "rec_20-40m", "rec_40-60m", "share_on_nocls", "share_on_unseen"):
            v = [(r[nm][met], r["ep"]) for r in rows if r[nm][met] is not None]
            summ[nm][met] = boot_mean([x for x, _ in v], [e for _, e in v], a.n_boot) if v else None
        p, rr = summ[nm]["prec"], summ[nm]["rec"]
        summ[nm]["F_tol_of_means"] = (2 * p["point"] * rr["point"] / (p["point"] + rr["point"])) if p and rr and (p["point"] + rr["point"]) > 0 else None
    res["summary"] = summ
    col = lambda nm, met: [r[nm][met] for r in rows]  # noqa: E731
    bars = {"B1_rec_OWNBND_minus_OWNCODE5": paired(col("OWN-BND", "rec"), col("OWN-CODE5", "rec"), ep_of, a.n_boot),
            "B2_rec_OWNBND_minus_DERBND": paired(col("OWN-BND", "rec"), col("DER-BND", "rec"), ep_of, a.n_boot),
            "B3_prec_OWNBND_minus_DERBND": paired(col("OWN-BND", "prec"), col("DER-BND", "prec"), ep_of, a.n_boot)}
    k3 = paired(col("OWN-BND", "rec"), col("OWN-BND", "rec"), ep_of, a.n_boot)
    ctl["K3_identity_paired_zero_real"] = {"pass": bool(k3 and k3["point"] == 0 and k3["lo"] == 0 and k3["hi"] == 0)}
    ok = {k: bool(v is not None and v["lo"] > 0) for k, v in bars.items()}
    res["bars"] = {k: {"cell": v, "pass": ok[k]} for k, v in bars.items()}
    ctl_fail = sorted(k for k, v in ctl.items() if not v["pass"])
    if ctl_fail or not res["seal_ok"]:
        verdict = f"VOID (control failed: {ctl_fail})"
        nxt = "fix the instrument; re-run (0 GPU)."
    elif all(ok.values()):
        verdict = "INSTRUMENT-PASS"
        nxt = "Q2: the OWN-BND-T12 selection arm through MAPSEL's scorer (sealed separately before it runs)."
    elif not ok["B1_rec_OWNBND_minus_OWNCODE5"]:
        verdict = "FAIL (B1)"
        nxt = "the boundary does not beat the decided class: the edge needs a TRAINED signal -> P3 thin-class recall (class-balanced / distance-transform edge loss)."
    else:
        verdict = "FAIL (" + ", ".join(k.split("_")[0] for k, v in ok.items() if not v) + ")"
        nxt = "the boundary is a geometric prior, not perception: Q2 runs with the PRIOR as the arm (labelled), and drivable precision at the curb is the perception lever."
    res["verdict"], res["next"], res["wall_s"] = verdict, nxt, round(time.time() - t0, 1)
    json.dump(res, open(out / "MAPEDGE_RESULT.json", "w"), indent=1)
    f3 = lambda c: f"{c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] (n_win {c['n_windows']}, n_ep {c['n_episodes']})" if c else "-"  # noqa: E731
    L = [verdict, "", f"# MAPEDGE Q1 (mechanical; PREREG_MAPEDGE sha256 {pre[:12]})", "", f"tier: {res['tier']}", f"estimator: {res['estimator']}", "",
         "| X | recall @0.3 m | precision @0.3 m | rec 0-20 | rec 20-40 | rec 40-60 | share on GT nocls | share on GT unseen |", "|---|---|---|---|---|---|---|---|"]
    for nm in ARMS:
        s = summ[nm]
        g = lambda m: f"{s[m]['point']:.3f}" if s[m] else "-"  # noqa: E731
        L.append(f"| {nm} | {f3(s['rec'])} | {f3(s['prec'])} | {g('rec_0-20m')} | {g('rec_20-40m')} | {g('rec_40-60m')} | {g('share_on_nocls')} | {g('share_on_unseen')} |")
    L += ["", "| bar | pass | paired cell |", "|---|---|---|"] + [f"| {k} | {res['bars'][k]['pass']} | {f3(res['bars'][k]['cell'])} |" for k in bars]
    L += ["", f"controls failed: {ctl_fail or 'none'}", "GT-BND = GT drivable cells next to a JUDGED non-drivable cell (reference ceiling, not an arm).", "", f"NEXT: {nxt}", "", "ZZMEVERDICTENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    print(verdict, json.dumps({k: v["pass"] for k, v in res["bars"].items()}), ctl_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
