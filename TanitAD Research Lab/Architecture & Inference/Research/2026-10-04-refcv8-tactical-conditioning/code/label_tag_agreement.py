"""DESIGN sec. 3.9.4 / SPEC_REFCV8 Q2 (owed): the v9 LABEL <-> plan TAG agreement on the [2, 6] s overlap.

The v9 tactical labels describe the band [NOW + 2 s, NOW + 8 s]; the plan, its tags (`refcv8_conditioning.tag_paths`)
and the plan-derived constraint targets live on [0, 6] s. Before the v9 labels supervise heads whose outputs condition
a [0, 6] s fan, measure how often the two agree -- per class, as confusion matrices, on the GT path itself (no model).

Two tags of the GT path per window (the tagger is the trainer's, unchanged):
* PLAN [0, 6] s: the 8 plan slots (0.5 ... 6 s) in the NOW frame, v0 = v(NOW) -- what the fan's candidates are tagged on;
* OVERLAP [2, 6] s: the path re-anchored at NOW + 2 s (its own frame and speed), slots 0.5 ... 4 s -- the window the
  label and the plan share.
Compared with the v9 class (frozen v7 ids, lat variant a), on EXACT rows (single class) and on PARTIAL rows (is the tag
inside the allowed set). FOLLOW is never a geometric tag (it needs a lead), so the lon agreement is also reported with
FOLLOW-labelled rows excluded. ⛔ No clip id is written -- counts only.

Run:  PYTHONPATH=<tree>/stack python code/label_tag_agreement.py --split train|eval139
Writes raw/label_tag_agreement_<split>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAN = {"train": os.environ.get("R8_TRAIN_V2MANIFEST",
                               "D:/Projects/TanitAD-artifacts/refcv8_wpb/inputs/train_v2manifest.pt"),
       "eval139": "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"}
V9 = {"train": "D:/Projects/TanitAD-artifacts/v9labels/v9_labels_train.npz",
      "eval139": "D:/Projects/TanitAD-artifacts/v9labels/v9_labels_eval139.npz"}
W, FUT = 8, 60
PLAN_H = (5, 10, 15, 20, 30, 40, 50, 60)            # the planner's slots (ticks), 0.5 ... 6 s
OVL_ANCHOR = 20                                     # NOW + 2 s
OVL_H = (5, 10, 15, 20, 30, 40)                     # 0.5 ... 4 s after the anchor = [2, 6] s after NOW


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def frame_slots(P, r, horizons):
    """[n, S, 2] slot positions of rows r + h in the frame of row r (x fwd, y left)."""
    x, y, yaw = P[:, 0], P[:, 1], P[:, 2]
    idx = r[:, None] + np.asarray(horizons)[None, :]
    c, s = np.cos(yaw[r])[:, None], np.sin(yaw[r])[:, None]
    dx, dy = x[idx] - x[r][:, None], y[idx] - y[r][:, None]
    return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)


def confusion(lab, tag, ids):
    m = np.zeros((len(ids), len(ids)), np.int64)
    pos = {v: i for i, v in enumerate(ids)}
    for a, b in zip(lab, tag):
        if a in pos and b in pos:
            m[pos[a], pos[b]] += 1
    return m


def block(lab, allowed_bits, tag, ids, names, exclude=()):
    """Agreement of the tag with the label. lab: v7 id (-100 = not exact); allowed_bits: v7 allowed mask bits."""
    ex = lab >= 0
    if exclude:
        ex &= ~np.isin(lab, exclude)
    part = (lab < 0) & (allowed_bits > 0)
    in_allowed = ((allowed_bits >> np.clip(tag, 0, 30)) & 1).astype(bool)
    cm = confusion(lab[ex], tag[ex], ids)
    per = {}
    for i, nm in enumerate(names):
        n = int(cm[i].sum())
        per[nm] = {"n_label": n, "agree": round(float(cm[i, i]) / n, 4) if n else None,
                   "n_tag": int(cm[:, i].sum()), "precision": round(float(cm[i, i]) / cm[:, i].sum(), 4)
                   if cm[:, i].sum() else None}
    nex = int(ex.sum())
    return {"n_exact": nex, "agree_exact": round(float(np.trace(cm)) / max(int(cm.sum()), 1), 4),
            "n_exact_in_space": int(cm.sum()),
            "n_partial": int(part.sum()), "tag_in_allowed_partial": round(float(in_allowed[part].mean()), 4)
            if part.any() else None,
            "per_class": per, "confusion_label_x_tag": {"ids": list(ids), "names": list(names), "m": cm.tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "eval139"), default="eval139")
    a = ap.parse_args()
    import torch
    import tanitad
    if str(Path(tanitad.__file__).resolve()).upper().startswith("G:"):
        raise SystemExit("tanitad imported from G:")
    from tanitad.data import v9_labels as V9L
    from tanitad.data.v2_dataset import stable_episode_id
    from tanitad.refs import refcv8_conditioning as r8c
    t0 = time.time()
    man = torch.load(MAN[a.split], map_location="cpu", weights_only=False)
    rel = V9L.load_v9_release(V9[a.split])
    R = rel.rows
    plans, ovls, v0p, v0o, rows = [], [], [], [], []
    for i, cid in enumerate(man["clip_id"]):
        P = man["poses"][i].numpy().astype(np.float64)
        T = len(P)
        n = T - (W + 20)                                       # the D4 / trainer window grid
        if n <= 0:
            continue
        r = np.arange(n) + W - 1
        ok = r + FUT <= T - 1                                  # a full 6-s GT future
        c = rel.clips.get(int(stable_episode_id(cid)))
        if c is None or not ok.any():
            continue
        r = r[ok]
        r0, nr, k0 = c
        j = r + 2 - k0
        inr = (j >= 0) & (j < nr)
        r, j = r[inr], j[inr]
        plans.append(frame_slots(P, r, PLAN_H))
        ovls.append(frame_slots(P, r + OVL_ANCHOR, OVL_H))
        v0p.append(P[r, 3])
        v0o.append(P[r + OVL_ANCHOR, 3])
        rows.append(r0 + j)
    plans, ovls = np.concatenate(plans), np.concatenate(ovls)
    v0p, v0o, rows = np.concatenate(v0p), np.concatenate(v0o), np.concatenate(rows)
    lat_ids, lon_ids = r8c.LAT3_V7_IDS, r8c.LON6_V7_IDS

    def tags(paths, v0, horizons):
        out_l, out_o = [], []
        for s in range(0, len(paths), 65536):
            tl, to = r8c.tag_paths(torch.tensor(paths[s:s + 65536], dtype=torch.float32)[:, None],
                                   torch.tensor(v0[s:s + 65536], dtype=torch.float32),
                                   [h * 0.1 for h in horizons])
            out_l.append(np.asarray(lat_ids)[tl[:, 0].numpy()])
            out_o.append(np.asarray(lon_ids)[to[:, 0].numpy()])
        return np.concatenate(out_l), np.concatenate(out_o)
    tl_p, to_p = tags(plans, v0p, PLAN_H)
    tl_o, to_o = tags(ovls, v0o, OVL_H)
    lat = np.asarray(R["lat_v7id_a"]).astype(np.int64)[rows]
    lat_al = np.asarray(R["lat_allowed_v7_a"]).astype(np.int64)[rows]
    lon = np.asarray(R["lon_v7id"]).astype(np.int64)[rows]
    lon_al = np.asarray(R["lon_allowed_v7"]).astype(np.int64)[rows]
    out = {"_evidence": "MEASURED (WP-B; the GT path only, no model; the trainer's tagger)", "split": a.split,
           "n_windows": int(len(rows)), "inputs": {"manifest_md5": md5(MAN[a.split]), "v9_md5": rel.md5},
           "tagger": {"turn_deg": r8c.TAG_TURN_DEG, "min_len_m": r8c.TAG_MIN_LEN_M, "dv_ms": r8c.TAG_DV_MS,
                      "crawl_ms": r8c.TAG_CRAWL_MS, "stop_ms": r8c.TAG_STOP_MS},
           "plan_0_6s": {"lat": block(lat, lat_al, tl_p, lat_ids, r8c.LAT3),
                         "lon": block(lon, lon_al, to_p, lon_ids, r8c.LON6),
                         "lon_excl_FOLLOW": block(lon, lon_al, to_p, lon_ids, r8c.LON6, exclude=(0,))},
           "overlap_2_6s": {"lat": block(lat, lat_al, tl_o, lat_ids, r8c.LAT3),
                            "lon": block(lon, lon_al, to_o, lon_ids, r8c.LON6),
                            "lon_excl_FOLLOW": block(lon, lon_al, to_o, lon_ids, r8c.LON6, exclude=(0,))},
           "wall_s": round(time.time() - t0, 1)}
    p = HERE.parent / "raw" / f"label_tag_agreement_{a.split}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for w in ("plan_0_6s", "overlap_2_6s"):
        for fam in ("lat", "lon", "lon_excl_FOLLOW"):
            b = out[w][fam]
            print(f"{w:13s} {fam:16s} exact n={b['n_exact']:7d} agree={b['agree_exact']}  partial n={b['n_partial']:7d} "
                  f"tag-in-allowed={b['tag_in_allowed_partial']}  per-class "
                  + " ".join(f"{k}:{v['agree']}" for k, v in b["per_class"].items()))
    print("wrote", p, out["wall_s"], "s")


if __name__ == "__main__":
    sys.exit(main())
