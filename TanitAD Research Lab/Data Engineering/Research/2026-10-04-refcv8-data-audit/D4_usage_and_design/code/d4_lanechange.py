#!/usr/bin/env python3
"""D4: can LANE_CHANGE be separated from NUDGE / drift from EGO GEOMETRY ALONE? Two rules, one independent
referee: the Alpamayo chain-of-causation lane-change TEXT (`lane_change_text.status == EXECUTED`, side L/R,
104 train clips) -- a channel the geometry never reads. Writes raw/d4_lanechange.json. CPU only.

rule v1 = d4_lib.dense_tactical (end offset >= 2.5 m on a near-straight horizon, >= 3 deg of steering).
rule v2 = v1 + the heading must be STEADY for 2 s before NOW (|dpsi(NOW-2s..NOW)| < 2 deg), STEADY over
          the last 1 s of the horizon (< 2 deg), RETURNED to the NOW heading (|psi_end| < 3 deg), a
          steering bump >= 4 deg, and an end offset within ONE lane (2.5 .. 5.0 m).
Referee tests per text clip: (a) ANY window with a same-side LC, (b) the DOMINANT dense-LC side == text side.
Control: clips with NO lane-change text (NOT_CITED) -> rate of clips with any dense LC (lower = better).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ["D4_SCRATCH"]
LAB = "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz"
MAN = os.path.join(SCR, "train_v2manifest.pt")


def lc_v2(B):
    n = B["n"]
    valid = B["valid"]
    nfut = valid.sum(1) - 1
    ok = nfut >= 60
    r = B["r"]
    yu = B["yaw_u"]
    past = np.degrees(np.abs(yu[r] - yu[np.maximum(r - 20, 0)]))
    dps = np.degrees(B["dpsi"])
    end_steady = np.abs(dps[:, 60] - dps[:, 50]) < 2.0
    returned = np.abs(dps[:, 60]) < 3.0
    bump = np.abs(dps).max(1) >= 4.0
    y_end = B["Y"][:, 60]
    one_lane = (np.abs(y_end) >= 2.5) & (np.abs(y_end) <= 5.0)
    lc = ok & (past < 2.0) & (r >= 20) & end_steady & returned & bump & one_lane
    return np.where(lc, np.sign(y_end), 0).astype(int)


def main():
    txt = {}
    with gzip.open(LAB, "rt", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            lc = q.get("lane_change_text") or {}
            st, sd = lc.get("status"), lc.get("side")
            txt[q["clip_id"]] = (1 if sd == "L" else -1) if (st == "EXECUTED" and sd in ("L", "R")) else \
                (0 if st in (None, "NOT_CITED") else 9)
    man = torch.load(MAN, map_location="cpu", weights_only=False)
    rows = {"v1": [], "v2": []}
    tside = []
    for i, cid in enumerate(man["clip_id"]):
        if cid not in txt:
            continue
        P = man["poses"][i].numpy().astype(np.float64)
        B = L.window_block(P)
        lat, _, _ = L.dense_tactical(B)
        s1 = np.where(lat == 1, 1, np.where(lat == 2, -1, 0))
        s2 = lc_v2(B)
        for k, s in (("v1", s1), ("v2", s2)):
            rows[k].append(((s == 1).sum(), (s == -1).sum()))
        tside.append(txt[cid])
    tside = np.array(tside)
    out = {"_evidence": "MEASURED (ours), train 4,369 clips, referee = alpamayo lane_change_text EXECUTED",
           "n_text_L": int((tside == 1).sum()), "n_text_R": int((tside == -1).sum()),
           "n_no_text": int((tside == 0).sum())}
    for k in ("v1", "v2"):
        a = np.array(rows[k])
        nl, nr = a[:, 0], a[:, 1]
        anyc = (nl + nr) > 0
        dom = np.where(nl > nr, 1, np.where(nr > nl, -1, 0))
        res = {}
        for sd, nm in ((1, "L"), (-1, "R")):
            m = tside == sd
            same_any = (nl > 0) if sd == 1 else (nr > 0)
            opp_any = (nr > 0) if sd == 1 else (nl > 0)
            res[f"text_{nm}"] = {"n": int(m.sum()),
                                 "any_same_side": round(float(same_any[m].mean()), 4),
                                 "any_opposite_side": round(float(opp_any[m].mean()), 4),
                                 "dominant_side_correct": round(float((dom[m] == sd).mean()), 4),
                                 "dominant_side_wrong": round(float((dom[m] == -sd).mean()), 4)}
        m0 = tside == 0
        res["no_text_any_dense_LC"] = round(float(anyc[m0].mean()), 4)
        res["text_any_dense_LC"] = round(float(anyc[(tside == 1) | (tside == -1)].mean()), 4)
        res["windows_LC_per_clip_mean"] = round(float((nl + nr).mean()), 3)
        out[k] = res
    json.dump(out, open(os.path.join(HERE, "raw", "d4_lanechange.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
