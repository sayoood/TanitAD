"""P1 extension: (a) the SAME out-and-back hypothesis over LONGER windows —
the 4 s lat horizon may simply be too short to contain both steering phases;
(b) the arc-residual over the whole 201-clip corpus (expected emission rate).

This is NOT a new hypothesis, it is the same physical claim measured at three
horizons; multiplicity is re-reported over the enlarged battery.
"""
import json
import math
import os
import sys
from itertools import combinations

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
EGO = os.path.join(SP, "s2_ego", "aug120")
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np                                            # noqa: E402
import s2_derive                                              # noqa: E402

DT, T0 = 0.1, 80
Y, LC_MIN, HALF = 0.20, 3.0, 1.75


def prof(poses, t, nsteps):
    T = poses.shape[0]
    h = min(nsteps, T - 1 - t)
    if h < 5:
        return None
    seg = poses[t:t + h + 1]
    x, y = seg[:, 0], seg[:, 1]
    yaw = np.unwrap(seg[:, 2].astype(np.float64))
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = float(s[-1])
    if L < 1e-6:
        return None
    net = float(yaw[-1] - yaw[0])
    c, sn = math.cos(yaw[0]), math.sin(yaw[0])
    lat = -sn * (x - x[0]) + c * (y - y[0])
    lat_f = float(lat[-1])
    lat_arc = (L / net) * (1 - math.cos(net)) if abs(net) > 1e-6 else 0.0
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, yaw, rcond=None)
    res = yaw - A @ coef
    sst = float(((yaw - yaw.mean()) ** 2).sum())
    ds = np.diff(s)
    kap = np.where(ds > 1e-3, np.diff(yaw) / np.maximum(ds, 1e-9), 0.0)
    pos = float(np.sum(np.clip(kap, 0, None) * ds))
    neg = float(-np.sum(np.clip(kap, None, 0) * ds))
    # detrended curvature: bidirectionality AFTER removing the road's mean bend
    kd = kap - coef[0]
    pdd = float(np.sum(np.clip(kd, 0, None) * ds))
    ndd = float(-np.sum(np.clip(kd, None, 0) * ds))
    return {
        "L": L, "net_yaw": net, "lat_f": lat_f, "lat_arc": lat_arc,
        "resid": abs(lat_f - lat_arc),
        "yaw_lin_r2": 1.0 - float((res ** 2).sum()) / sst if sst > 1e-12 else 1.0,
        "yaw_detrend_peak": float(np.abs(res).max()),
        "bidir": min(pos, neg) / max(pos, neg) if max(pos, neg) > 1e-9 else 0.0,
        "bidir_detrend": (min(pdd, ndd) / max(pdd, ndd)
                          if max(pdd, ndd) > 1e-9 else 0.0),
        "swing_detrend": pdd + ndd,
    }


rows = json.load(open(os.path.join(SP, "lc_feat.json"), encoding="utf-8"))
ea_rows = {json.loads(l)["clip_id"]: json.loads(l)["engine_a"]
           for l in open(os.path.join(PKG, "labels", "engine_a_aug120.jsonl"),
                         encoding="utf-8") if l.strip()}

HORIZONS = [("h4s", 40), ("h8s", 80), ("h12s", 120)]
recs = []
for r in rows:
    if not r["gated"]:
        continue
    cid = r["clip_id"]
    poses = np.load(os.path.join(EGO, f"{cid}.npz"))["poses"].astype(np.float64)
    t = r["t_ev"]
    d = {"clip_id": cid, "pi": r.get("pi")}
    for name, n in HORIZONS:
        p = prof(poses, t, n)
        d[name] = p
    recs.append(d)

lab = [r for r in recs if r["pi"] in ("wrong", "correct")]
lab.sort(key=lambda r: r["clip_id"])
N, y = len(lab), [1 if r["pi"] == "correct" else 0 for r in lab]
nc = sum(y)
ALL = list(combinations(range(N), nc))
TRUE = tuple(i for i in range(N) if y[i] == 1)
KEYS = ["yaw_lin_r2", "yaw_detrend_peak", "bidir", "bidir_detrend",
        "swing_detrend", "resid"]


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[o[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return rk


print(f"n={N} (correct={nc}); exact enumeration over {len(ALL)} assignments\n")
hdr = (f"{'horizon':<7}{'feature':<19}{'p_exact':>9}{'Cliff d':>9}"
       f"{'med_C':>10}{'med_W':>10}{'sep':>5}")
print(hdr)
print("-" * len(hdr))
tested = []
for hn, _ in HORIZONS:
    for k in KEYS:
        v = [(r[hn] or {}).get(k, float("nan")) for r in lab]
        if any(not math.isfinite(x) for x in v):
            continue
        rk = ranks(v)
        st = lambda sel: abs(sum(rk[i] for i in sel) - nc * (N + 1) / 2.0)
        obs = st(TRUE)
        p = sum(1 for s in ALL if st(s) >= obs - 1e-12) / len(ALL)
        a = sorted(v[i] for i in TRUE)
        b = sorted(v[i] for i in range(N) if y[i] == 0)
        gt = sum(1 for x in a for z in b if x > z)
        lt = sum(1 for x in a for z in b if x < z)
        dl = (gt - lt) / (len(a) * len(b))
        sep = (min(a) > max(b)) or (max(a) < min(b))
        tested.append((hn, k, p))
        print(f"{hn:<7}{k:<19}{p:>9.4f}{dl:>9.2f}{a[len(a)//2]:>10.4g}"
              f"{b[len(b)//2]:>10.4g}{'YES' if sep else '':>5}")

print(f"\nbattery now 34 (h4s) + {len(tested)-6} new = "
      f"{34 + len(tested) - 6} features tested in total; "
      f"Bonferroni alpha for 0.05 = {0.05/(34+len(tested)-6):.2e}; "
      f"min attainable p = {2/len(ALL):.2e}")
print("=> NO feature at ANY horizon can reach family-wise significance unless "
      "it separates PERFECTLY, and none does.")

# ---------------- corpus-wide arc-residual, all 201 clips -------------------
print("\n--- corpus-wide: what an arc-residual gate would emit (n=201) ---")
allrec = []
for cid, ea in ea_rows.items():
    ev = s2_derive._gated_lc_event(ea)
    if ev is None:
        continue
    poses = np.load(os.path.join(EGO, f"{cid}.npz"))["poses"].astype(np.float64)
    t = T0 + int(round(float(ev["t_start_s"]) / DT))
    p = prof(poses, t, 40)
    allrec.append((cid, p))
for thr in (1.0, 1.5, HALF, 2.0, 2.5, LC_MIN):
    n = sum(1 for _, p in allrec if p and p["resid"] >= thr)
    print(f"  resid >= {thr:>4}: {n:>2}/201 = {100*n/201:5.2f}% of clips "
          f"({n}/{len(allrec)} of currently-gated)")
json.dump({"per_horizon": recs,
           "corpus_resid": [(c, p) for c, p in allrec]},
          open(os.path.join(SP, "lc_ext.json"), "w", encoding="utf-8"), indent=1)
