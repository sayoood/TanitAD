"""Per-step nav-text table aligned to the v8f-mini label joins (train4572, eval141).

Input : raw/nav_sub.npz (clips, clip_i, ts_us, text_i) cut from navigation.parquet on the pod; raw/texts.json (text_i -> text);
        raw/join_train.npz, raw/join_eval141.npz (the v8f-mini joins: one entry per cache step j, ``now_s`` = label-clock time of step j).
Time  : nav ``timestamp`` is us after clip start on the egomotion base (parquet schema metadata; egomotion t0 median -195,000 us,
        MEASURED navalign.json ``ego_t0_quantiles_us``).  The label timeline's zero is the egomotion log's first sample (clip_clock.py
        docstring), so  nav_ts_us(step) = round(now_s * 1e6) + T0_US  with T0_US = -195,000 (ESTIMATED per clip: the clip's own t0 is
        within [-199,566, -190,397] us for 90 % of clips, +-5 ms).  ``--t0-scan`` measures how the result moves with T0.
Rule  : the latest nav row with ts <= nav_ts_us (row semantics: "nav_text holds from timestamp until the clip's next row").
Flags : 0 ok | 1 step before the clip's first row | 2 clip has no nav rows | 3 step has no v9 row (now_s NaN: no time, no lookup)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navtok as N

T0_US = -195000


def load_nav(path=os.path.join(HERE, "raw", "nav_sub.npz"), texts_path=os.path.join(HERE, "raw", "texts.json")):
    z = np.load(path)
    texts = json.load(open(texts_path, encoding="utf-8"))
    parsed = [N.parse_full(t) for t in texts]
    cls_id = np.array([N.CLS_ID[p.cls] for p in parsed], np.int16)
    dist = np.array([np.nan if p.dist_m is None else p.dist_m for p in parsed], np.float32)
    clips = [str(c) for c in z["clips"]]
    ci, ts, ti = z["clip_i"], z["ts_us"], z["text_i"]
    start = np.searchsorted(ci, np.arange(len(clips)), "left")
    end = np.searchsorted(ci, np.arange(len(clips)), "right")
    return {"clips": {c: (int(start[i]), int(end[i])) for i, c in enumerate(clips)}, "ts": ts, "ti": ti, "cls_id": cls_id, "dist": dist, "texts": texts}


def build(J: dict, nav: dict, t0_us: int = T0_US) -> dict:
    N_ = int(J["n"].sum())
    out = {"cls_id": np.full(N_, -1, np.int16), "dist_m": np.full(N_, np.nan, np.float32), "flag": np.zeros(N_, np.uint8),
           "text_i": np.full(N_, -1, np.int32), "age_s": np.full(N_, np.nan, np.float32)}
    for ci, cid in enumerate(J["clip"]):
        o, n = int(J["off"][ci]), int(J["n"][ci])
        sl = slice(o, o + n)
        now = J["now_s"][sl]
        ok = np.isfinite(now)
        r = nav["clips"].get(str(cid))
        fl = np.zeros(n, np.uint8)
        fl[~ok] = 3
        if r is None:
            fl[ok] = 2
            out["flag"][sl] = fl
            continue
        a, b = r
        ts = nav["ts"][a:b]
        q = np.where(ok, np.round(np.where(ok, now, 0.0) * 1e6) + t0_us, 0).astype(np.int64)
        p = np.searchsorted(ts, q, "right") - 1
        before = ok & (p < 0)
        fl[before] = 1
        use = ok & ~before
        ti = nav["ti"][a:b][np.clip(p, 0, None)]
        out["text_i"][sl] = np.where(use, ti, -1)
        out["cls_id"][sl] = np.where(use, nav["cls_id"][ti], -1)
        out["dist_m"][sl] = np.where(use, nav["dist"][ti], np.nan)
        out["age_s"][sl] = np.where(use, (q - ts[np.clip(p, 0, None)]) / 1e6, np.nan)
        out["flag"][sl] = fl
    return out


def trainer_view(J: dict) -> dict:
    """What the v8f-mini model reads: token = argmax(nav[:, :3]) where a v9 row exists; d = nav_d (m, NaN if invalid)."""
    return {"tok": J["nav"][:, :3].argmax(1).astype(np.int8), "valid": J["nav"][:, 4] > 0, "d": J["nav_d"], "has": J["has"]}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "navtok_table.npz"))
    ap.add_argument("--t0-us", type=int, default=T0_US)
    a = ap.parse_args()
    nav = load_nav()
    pack, rep = {}, {"t0_us": a.t0_us, "vocab": list(N.VOCAB), "flag_legend": {"0": "ok", "1": "before the clip's first row", "2": "clip has no nav rows", "3": "step has no v9 row (now_s NaN)"}}
    for split, jf in (("train", "join_train.npz"), ("eval", "join_eval141.npz")):
        J = dict(np.load(os.path.join(HERE, "raw", jf)))
        T = build(J, nav, a.t0_us)
        for k in ("clip", "off", "n"):
            pack[f"{split}_{k}"] = J[k]
        for k, v in T.items():
            pack[f"{split}_{k}"] = v
        fl = T["flag"]
        rep[split] = {"clips": int(len(J["n"])), "steps": int(len(fl)), "flags": {str(i): int((fl == i).sum()) for i in range(4)},
                      "clips_without_nav_rows": int(sum(1 for c in J["clip"] if str(c) not in nav["clips"]))}
    pack["vocab"] = np.array(N.VOCAB)
    np.savez_compressed(a.out, **pack)
    rep["file"] = os.path.basename(a.out)
    rep["sha256"] = sha256(a.out)
    rep["bytes"] = os.path.getsize(a.out)
    json.dump(rep, open(os.path.splitext(a.out)[0] + "_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1)[:2500])


if __name__ == "__main__":
    main()
