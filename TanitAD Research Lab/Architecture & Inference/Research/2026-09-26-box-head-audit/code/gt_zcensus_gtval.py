#!/usr/bin/env python3
"""gt_zcensus_gtval.py -- the heavy_truck question on the GT-validation clips themselves (where the video agent's
pooled -0.63 m came from): absolute rig-frame bottom AND the same-frame neighbour-relative offset d (gt_zcensus.py's
definition), per class and per clip, over every labelled eval window of the 5 clips (the trainer's _agent_item).
sha12 only."""
import json
import math
import os
import sys

import numpy as np

CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person", "rider", "stroller",
           "animal", "protruding_object")
REF = (0, 5, 6)


def main():
    tree, code, out = sys.argv[1], sys.argv[2], sys.argv[3]
    clips = sys.argv[4].split(",")
    os.environ["REFCV6_REPO"] = tree
    sys.path.insert(0, code)
    import refcv6_loader as L
    L.bootstrap()
    import hashlib
    from pathlib import Path
    from tanitad.data import v2_dataset as v2d
    config = L.load_config("/home/nvidia/refcv6_run/runs/refcv6-r101-s0/config.json")
    model, cfg, targs, _ = L.build_model(config, "/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt.pt",
                                         device="cpu", remap={})
    del model
    table = json.load(open("/home/nvidia/data/refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
    by12 = {hashlib.sha256(c.encode()).hexdigest()[:12]: c for c in table}
    keep = {by12[s] for s in clips}

    def clip_of(ep):
        fp = ep.frames
        return Path(fp._cache.files[fp._clip]).name.split(".")[0]
    _o = v2d.build_v2_providers
    v2d.build_v2_providers = lambda *aa, **kk: [e for e in _o(*aa, **kk) if clip_of(e) in keep]
    try:
        e_ds, e_eps, _ = L.build_eval_dataset(None, cfg, targs, config, with_perception_targets=True)
    finally:
        v2d.build_v2_providers = _o
    Wn = int(cfg.core.window)
    res = {}
    for wi, (e_i, t) in enumerate(e_ds.index):
        ep = e_eps[e_i]
        s12 = hashlib.sha256(clip_of(ep).encode()).hexdigest()[:12]
        it = e_ds._agent_item(ep, t + Wn - 1)
        if not bool(it["agent_label"]):
            continue
        v = it["agent_valid"].numpy().astype(bool) & it["agent_zh_mask"].numpy().astype(bool)
        b = it["agent_box"].double().numpy()
        bot = (it["agent_cz"] - it["agent_h"] / 2).double().numpy()
        cl = it["agent_cls"].numpy().astype(int)
        ref = v & np.isin(cl, REF)
        for j in np.nonzero(v)[0]:
            if not (0 <= cl[j] < 10):
                continue
            d = np.hypot(b[:, 0] - b[j, 0], b[:, 1] - b[j, 1])
            nb = ref & (d <= 15.0)
            nb[j] = False
            rr = math.hypot(b[j, 0], b[j, 1])
            key = (s12, CLASSES[cl[j]])
            res.setdefault(key, []).append((bot[j], (bot[j] - float(np.median(bot[nb]))) if nb.sum() >= 2 else np.nan, rr))
    out_d = {}
    for (s12, c), v in sorted(res.items()):
        v = np.asarray(v, dtype=np.float64)
        dv = v[:, 1][np.isfinite(v[:, 1])]
        out_d[f"{s12}|{c}"] = {"n": int(len(v)), "median_bottom_rig_m": float(np.median(v[:, 0])),
                               "n_with_ref": int(dv.size),
                               "median_d_vs_neighbours_m": float(np.median(dv)) if dv.size else None,
                               "median_range_m": float(np.median(v[:, 2]))}
    pooled = {}
    for c in CLASSES:
        v = [x for (s, cc), rows in res.items() if cc == c for x in rows]
        if not v:
            continue
        v = np.asarray(v, dtype=np.float64)
        dv = v[:, 1][np.isfinite(v[:, 1])]
        pooled[c] = {"n": int(len(v)), "median_bottom_rig_m": float(np.median(v[:, 0])), "n_with_ref": int(dv.size),
                     "median_d_vs_neighbours_m": float(np.median(dv)) if dv.size else None,
                     "p10_d": float(np.percentile(dv, 10)) if dv.size else None,
                     "p90_d": float(np.percentile(dv, 90)) if dv.size else None}
    json.dump({"per_clip_class": out_d, "pooled": pooled}, open(out, "w"), indent=1)
    for c, v in pooled.items():
        print(c, v)
    for k, v in out_d.items():
        if "heavy_truck" in k or "bus" in k:
            print(k, v)


if __name__ == "__main__":
    main()
