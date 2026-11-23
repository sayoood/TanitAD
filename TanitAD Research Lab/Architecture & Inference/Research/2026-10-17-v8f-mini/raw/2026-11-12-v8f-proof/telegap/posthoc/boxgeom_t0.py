"""boxgeom_t0.py -- tier T0 diagnostic on LABEL tracks only (no camera, no head, no card, no name): how much of the lead gap is fixed by where the lead's box sits in the image.
Maps are fitted on the pilot's train clips and read on its 141 evaluation clips (frames with a lead within 60 m). CPU, numpy only.
Usage: python boxgeom_t0.py <box dir> <train list> <eval list> <out.json>"""
import json
import os
import sys

import numpy as np

box, ftr, fev, out = sys.argv[1:5]
KEYS = ("gap_re", "ctr_az", "ctr_el", "bot_el", "ang_w", "ang_h", "has_lead")


def clips(path):
    return [l.split()[0] for l in open(path, encoding="utf-8") if l.strip()]


def load(cl):
    rows, cid, miss = {k: [] for k in KEYS}, [], 0
    for i, c in enumerate(cl):
        p = os.path.join(box, c + ".npz")
        if not os.path.isfile(p):
            miss += 1
            continue
        z = np.load(p)
        for k in KEYS:
            rows[k].append(np.asarray(z[k]))
        cid.append(np.full(len(z["gap_re"]), i))
    d = {k: np.concatenate(v) for k, v in rows.items()}
    d["clip"] = np.concatenate(cid)
    ok = d["has_lead"].astype(bool) & np.isfinite(d["gap_re"]) & (d["gap_re"] <= 60.0)
    for k in ("ctr_az", "ctr_el", "bot_el", "ang_w", "ang_h"):
        ok &= np.isfinite(d[k])
    ok &= (d["ang_w"] > 0) & (d["ang_h"] > 0)
    return {k: v[ok] for k, v in d.items()}, miss


def binmap(x, y, nb=60):
    q = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
    med = np.array([np.median(y[idx == b]) if (idx == b).any() else np.nan for b in range(len(q) - 1)])
    good = np.isfinite(med)
    cen = 0.5 * (q[:-1] + q[1:])
    return lambda xe: np.interp(xe, cen[good], med[good])


def knn(Xt, yt, Xe, k=25):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    A, B = (Xt - mu) / sd, (Xe - mu) / sd
    pred = np.empty(len(B))
    for s in range(0, len(B), 512):
        d2 = ((B[s:s + 512, None, :] - A[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        pred[s:s + 512] = np.median(yt[nn], axis=1)
    return pred


def report(name, pred, y):
    e = pred - y
    r = {"E": float(np.abs(e).mean()), "by_gap": {}}
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 60.01)):
        m = (y >= lo) & (y < hi)
        r["by_gap"]["%d-%d" % (lo, int(hi))] = {"n": int(m.sum()), "E": float(np.abs(e[m]).mean()), "signed": float(e[m].mean()), "short5": float((e[m] < -5).mean())}
    print(name, "E %.3f |" % r["E"], " ".join("%s: E %.2f signed %+.2f short>5 %.2f" % (k, v["E"], v["signed"], v["short5"]) for k, v in r["by_gap"].items()))
    return r


T, mt = load(clips(ftr))
E, me = load(clips(fev))
print("train frames %d (clips without a box file %d); evaluation frames %d (clips without a box file %d, clips with a frame %d)" % (len(T["gap_re"]), mt, len(E["gap_re"]), me, len(np.unique(E["clip"]))))
y = E["gap_re"]
res = {"note": "label tracks only; T0; names nothing", "n_train": int(len(T["gap_re"])), "n_eval": int(len(y)), "arms": {}}
res["arms"]["constant (train median)"] = report("constant (train median)   ", np.full(len(y), np.median(T["gap_re"])), y)
for k in ("bot_el", "ctr_el", "ang_h", "ang_w"):
    res["arms"][k] = report("%-25s" % ("map of " + k + " alone"), binmap(T[k], T["gap_re"])(E[k]), y)
f = lambda D, ks: np.stack([np.log(D[k]) if k.startswith("ang") else D[k] for k in ks], 1)
for ks in (("ang_h", "ang_w"), ("bot_el", "ang_h"), ("bot_el", "ang_h", "ang_w", "ctr_az")):
    nm = "nearest train frames on " + "+".join(ks)
    res["arms"][nm] = report(nm, knn(f(T, ks), T["gap_re"], f(E, ks)), y)
# how fine the direction has to be read: gap change per 0.0625 deg (one pixel of the wide camera) and per 0.0156 deg (one pixel of the tele camera) of bot_el, from the fitted map
m = binmap(T["bot_el"], T["gap_re"])
for g in (15, 30, 50):
    sel = np.abs(T["gap_re"] - g) < 2.5
    el0 = float(np.median(T["bot_el"][sel]))
    hmed = float(np.median(T["ang_h"][sel]))
    slope = (m(el0 + 0.25) - m(el0 - 0.25)) / 0.5
    print("at %d m: median bot_el %.2f deg, map slope %.1f m per deg -> %.2f m per wide pixel, %.2f m per tele pixel; median box height %.2f deg = %.0f wide px, %.0f tele px"
          % (g, el0, slope, abs(slope) * 0.0625, abs(slope) * 0.015625, hmed, hmed * 16, hmed * 64))
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
