#!/usr/bin/env python3
"""D4 L2 follow-up (Master Mind 2026-10-04, after D2 + route A5): the time-localised nav must use ANNOUNCED
entries only -- what a real nav announces (junction turns), not curves or obstacle passes the driven path shows.

announced(entry) = a NAV_TURN_x entry of nav_30s whose matching manoeuvre_sequence segment (|dt_start| <= 0.05 s)
carries the BUILDER's own `is_turn == True` and which is NOT a PI 2026-08-29 suppressed turn
(`turn_suppression.applied`, same side, its t_start_s inside the entry's [t_start, t_end]; the suppression time is
clipped to the band start 2.0 s when the turn began earlier -- MEASURED on the records). The builder's own flags are read, no
threshold is re-derived (A6's instruction).

Controls (must read known values first):
  K-A5  A5's nav_tl (ALL entries, H = 6 s, first unfinished entry) on the EVAL-DIAG grid must reproduce A5's table:
        GT-turn L/follow/R 12/55/40, GT-straight 4/572/12 (RESULT_A5.md sec. 2).
  G1    announced(entries[0]) side vs the shipped nav_command side (records), by nav_command reason.
Measures: the announced token (H = 6 s) on EVAL-DIAG GT-turn / GT-straight; TRAIN compliance-predicate correctness;
the NavSim-parity gap (the 20 m / 2 m driven-path rule fires while no announced entry is active).
Writes raw/d4_nav_announced.json. CPU only; ids never written.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ["D4_SCRATCH"]
LAB = {"train": "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz",
       "eval": "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"}
MAN = {"train": os.path.join(SCR, "train_v2manifest.pt"),
       "eval": "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"}
SIDECAR = "D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl"
SIDE = {"NAV_TURN_L": 1, "NAV_TURN_R": -1, "NAV_FOLLOW_ROAD": 0}
H = 6.0


def frac(a, b):
    return {"k": int(a), "n": int(b), "frac": (round(a / b, 4) if b else None)}


def entries_of(r):
    ents = (r.get("nav_30s") or {}).get("entries") or []
    seq = r.get("manoeuvre_sequence") or []
    ts = r.get("turn_suppression")
    sup = []
    if isinstance(ts, dict) and ts.get("applied"):
        sup.append((float(ts.get("t_start_s", -99)), ts.get("side")))
    out = []
    for e in ents:
        if e.get("t_start_s") is None:
            continue
        sd = SIDE.get(e.get("token"), 0)
        m = [s for s in seq if s.get("t_start_s") is not None and abs(s["t_start_s"] - e["t_start_s"]) <= 0.051]
        is_turn = bool(m and m[0].get("is_turn"))
        # the suppression's t_start_s is CLIPPED to the tactical band start (2.0 s) when the turn began earlier,
        # so match by SIDE and by the suppression time falling inside the entry's [t_start, t_end]
        t1e = float(e["t_end_s"]) if e.get("t_end_s") is not None else np.inf
        sname = {1: "left", -1: "right"}.get(sd)
        supp = any((side_ == sname) and (e["t_start_s"] - 0.051 <= t <= t1e + 0.051) for t, side_ in sup)
        out.append(dict(side=sd, t0=float(e["t_start_s"]),
                        t1=(float(e["t_end_s"]) if e.get("t_end_s") is not None else np.inf),
                        announced=bool(sd != 0 and is_turn and not supp), supp=supp, is_turn=is_turn))
    return sorted(out, key=lambda q: q["t0"])


def nav_tl(ents, trel_now, announced_only, skip_follow_entries):
    """per window: first unfinished (t1 > now) entry -> its side if it starts <= H ahead (or is under way)."""
    n = len(trel_now)
    side = np.zeros(n, np.int8)
    done = np.zeros(n, bool)
    for e in ents:
        if announced_only and not e["announced"]:
            continue
        if skip_follow_entries and e["side"] == 0:
            continue
        unf = (e["t1"] > trel_now) & ~done
        act = unf & ((e["t0"] - trel_now) <= H)
        side = np.where(act, e["side"], side)
        done = done | unf
    return side


def main():
    clock = {}
    for line in open(SIDECAR, encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            clock[int(q["sid"])] = (float(q["grid_start_s"]), float(q["dt_s"]))
    res = {"_evidence": "MEASURED (D4); labels b45377a1 / eefc38d1; manifests 3c9f8bc8 / 33433232", "H_s": H}
    for split in ("eval", "train"):
        recs = {}
        with gzip.open(LAB[split], "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["clip_id"]] = r
        # ---- G1 on records -------------------------------------------------------------------------
        g1 = {}
        for r in recs.values():
            nc = r.get("nav_command") or {}
            want = SIDE.get(nc.get("token"), 0)
            ents = entries_of(r)
            first = ents[0] if ents else None
            got_ann = (first["side"] if (first and first["announced"]) else 0)
            reason = (nc.get("reason") or "none")[:24]
            g = g1.setdefault(reason, [0, 0])
            g[0] += int(got_ann == want)
            g[1] += 1
        res[f"G1_{split}"] = {k: frac(v[0], v[1]) for k, v in g1.items()}
        res[f"G1_{split}_all"] = frac(sum(v[0] for v in g1.values()), sum(v[1] for v in g1.values()))
        # ---- windows ----------------------------------------------------------------------------------
        man = torch.load(MAN[split], map_location="cpu", weights_only=False)
        cols = {k: [] for k in ("sha", "t", "gt", "th", "a5", "a5skip", "ann", "path", "clip")}
        for i, cid in enumerate(man["clip_id"]):
            r = recs.get(cid)
            if r is None:
                continue
            P = man["poses"][i].numpy().astype(np.float64)
            g0, dt = clock.get(int(man["episode_uid"][i]), (0.0, 0.1))
            B = L.window_block(P)
            tnow = g0 + (B["r"] + L.RAW_OFF) * dt
            trel = tnow - float(r["t0_s"])
            ents = entries_of(r)
            gt, th = L.gt_class(B)
            path, _, okp = L.nav_path_rule_xy(P, B)
            cols["sha"] += [L.sha12(cid)] * B["n"]
            cols["t"].append(B["t"])
            cols["gt"].append(gt)
            cols["th"].append(th)
            cols["a5"].append(nav_tl(ents, trel, False, False))
            cols["a5skip"].append(nav_tl(ents, trel, False, True))
            cols["ann"].append(nav_tl(ents, trel, True, True))
            cols["path"].append(path)
            cols["clip"].append(np.full(B["n"], SIDE.get((r.get("nav_command") or {}).get("token"), 0), np.int8))
        Wd = {k: (np.concatenate(v) if k != "sha" else np.array(v)) for k, v in cols.items()}
        if split == "eval":
            # EVAL-DIAG grid (digest-checked in d4_measure.py)
            by = {}
            for j, (s, t) in enumerate(zip(Wd["sha"], Wd["t"])):
                by.setdefault(s, []).append((int(t), j))
            idx = []
            for s in sorted(by):
                ts = sorted(by[s])
                for q in range(8):
                    idx.append(ts[int((q + 0.5) * len(ts) / 8)][1])
            keys = [(Wd["sha"][j], int(Wd["t"][j])) for j in idx]
            res["evaldiag_digest16"] = hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:16]
            idx = np.array(idx)
            gt = Wd["gt"][idx]
            out = {}
            for name in ("clip", "a5", "a5skip", "ann", "path"):
                s = Wd[name][idx]
                row = {}
                for cls, gs in (("turnL", 1), ("turnR", -1)):
                    m = gt == cls
                    row[cls + "_L/F/R"] = [int((s[m] == 1).sum()), int((s[m] == 0).sum()), int((s[m] == -1).sum())]
                m = (gt == "turnL") | (gt == "turnR")
                gs = np.where(gt == "turnL", 1, -1)
                row["turn_side_correct"] = frac((s[m] == gs[m]).sum(), m.sum())
                m = gt == "straight"
                row["straight_L/F/R"] = [int((s[m] == 1).sum()), int((s[m] == 0).sum()), int((s[m] == -1).sum())]
                out[name] = row
            out["K_A5_known"] = {"turn_L/F/R_total": [12, 55, 40], "straight_L/F/R": [4, 572, 12]}
            res["evaldiag"] = out
        else:
            gt, th = Wd["gt"], Wd["th"]
            cls_ok = gt != "unclassified"
            tr = {}
            for name in ("clip", "a5", "ann"):
                s = Wd[name]
                inf = (s != 0) & cls_ok
                tr[name] = {"informative_windows": frac(inf.sum(), cls_ok.sum()),
                            "compliance_true": frac((L.compliance(th, s) & inf).sum(), inf.sum())}
            p = Wd["path"]
            tr["navsim_parity_gap"] = {
                "path_rule_fires_while_announced_follow": frac((((p == 1) | (p == -1)) & (Wd["ann"] == 0)).sum(), len(p)),
                "path_rule_fires": frac(((p == 1) | (p == -1)).sum(), len(p)),
                "announced_active": frac((Wd["ann"] != 0).sum(), len(p))}
            res["train_windows"] = tr
        print(split, "done", flush=True)
    json.dump(res, open(os.path.join(HERE, "raw", "d4_nav_announced.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
