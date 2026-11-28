#!/usr/bin/env python
"""build_dose_pack.py -- dose step of note PREREG_V8F_PROOF_ADD42_N2 section 3: the label pack of ALL training clips of the layout (4,572), same rule and same arrays as the banked
pack of the 1,200 probe clips (spatial/leadsup/read/leadsup_read.py build_train_pack, which uses spatial/leadhead_b/leadhead_b.py load_probe_meta).

  python build_dose_pack.py --layout <train_layout_0_4572.npz> --join C:/Users/Admin/lorapod/ship2/join_train.npz --out train_pack_all.npz [--states <ARM>_dose_train.npz]

Inputs: the layout file of pod_dose_layout.py (clips in manifest order, clip_idx, t, kin) and the label join (clip, off, n, lead_valid, lead_gap; the join of the banked pack, sha256 978f41bf...).
Row i of the pack is row i of the layout: clip_idx, t as in the state files of extract_states.py. Per row: pres = (lead_valid == 1); gap = lead_gap where lead_valid == 1, finite and
0 <= gap <= 60 m, else NaN (leadhead_b GAP_LO, GAP_HI); v = kin[:, 0] (ego speed of the cache). Refuses: a clip absent from the join, a join frame count different from the layout's,
frames of a clip that are not 0..T-1, an existing output file. Optional --states: the clips, clip_idx and t of a state file (small arrays only) must equal the layout's.
Exit 0 written, 2 refused. Tier T0 input preparation; no model, no GPU.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

GAP_LO, GAP_HI = 0.0, 60.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refuse(msg):
    raise SystemExit("REFUSED: " + msg)


def build_pack(layout, join):
    """layout: dict clips, clip_idx, t, kin. join: dict clip, off, n, lead_valid, lead_gap. -> dict of the pack arrays (without meta)."""
    clips = np.asarray(layout["clips"]).astype(str)
    ci = np.asarray(layout["clip_idx"]).astype(np.int64)
    t = np.asarray(layout["t"]).astype(np.int64)
    kin = np.asarray(layout["kin"], np.float64)
    if not (len(ci) == len(t) == len(kin)):
        refuse("layout arrays differ in length")
    nc = len(clips)
    if len(ci) and (ci.min() < 0 or ci.max() >= nc):
        refuse("layout clip_idx outside 0..%d" % (nc - 1))
    frames = np.bincount(ci, minlength=nc)
    first = np.r_[0, np.cumsum(frames)[:-1]]
    if not np.array_equal(t, np.arange(len(t)) - first[ci]) or not np.all(np.diff(ci) >= 0):
        refuse("layout rows are not clip by clip with frames 0..T-1")
    jc = {c: i for i, c in enumerate(np.asarray(join["clip"]).astype(str))}
    miss = [c for c in clips if c not in jc]
    if miss:
        refuse("%d clips of the layout are not in the join, first %s" % (len(miss), miss[:3]))
    ji = np.array([jc[c] for c in clips], np.int64)
    jn = np.asarray(join["n"]).astype(np.int64)[ji]
    if not np.array_equal(jn, frames):
        bad = int(np.flatnonzero(jn != frames)[0])
        refuse("join frame count differs from the layout for %d clips, first %s (join %d, layout %d)" % (int((jn != frames).sum()), clips[bad], jn[bad], frames[bad]))
    off = np.asarray(join["off"]).astype(np.int64)
    jidx = off[ji[ci]] + t
    lv = np.asarray(join["lead_valid"]).astype(np.int64)[jidx]
    g = np.asarray(join["lead_gap"])[jidx].astype(np.float64)
    ok = (lv == 1) & np.isfinite(g) & (g >= GAP_LO) & (g <= GAP_HI)
    return {"clips": clips, "clip_idx": ci, "t": t, "gap": np.where(ok, g, np.nan), "pres": (lv == 1).astype(np.float64), "v": kin[:, 0].copy()}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default="", help="optional state file whose clips, clip_idx and t must equal the layout's")
    a = ap.parse_args(argv)
    try:
        for p in (a.layout, a.join) + ((a.states,) if a.states else ()):
            if not os.path.isfile(p):
                refuse("no such file: " + p)
        if os.path.exists(a.out):
            refuse("%s exists (a pack is written once)" % a.out)
        L = np.load(a.layout, allow_pickle=False)
        layout = {k: L[k] for k in ("clips", "clip_idx", "t", "kin")}
        if a.states:
            S = np.load(a.states, allow_pickle=False)
            for k in ("clips", "clip_idx", "t"):
                if not np.array_equal(np.asarray(S[k]).astype(str if k == "clips" else np.int64), np.asarray(layout[k]).astype(str if k == "clips" else np.int64)):
                    refuse("%s of the state file differs from the layout" % k)
        z = np.load(a.join, allow_pickle=False)
        join = {k: z[k] for k in ("clip", "off", "n", "lead_valid", "lead_gap")}
        P = build_pack(layout, join)
    except SystemExit as e:
        print(json.dumps({"written": False, "error": str(e)}))
        return 2
    meta = {"join": a.join.replace("\\", "/"), "join_sha256": sha256_file(a.join), "layout_sha256": sha256_file(a.layout), "n_rows": int(len(P["t"])), "n_clips": int(len(P["clips"])),
            "gap_rule": "lead_valid == 1, finite, 0 <= gap <= 60 m (leadhead_b)", "ego_speed": "kin[:, 0] of the layout file (cache v, as the probe-set shards)",
            "n_v_nonfinite_rows": int((~np.isfinite(P["v"])).sum()), "states_checked": bool(a.states)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, clips=P["clips"], clip_idx=P["clip_idx"], t=P["t"], gap=P["gap"], pres=P["pres"], v=P["v"], meta=json.dumps(meta))
    print(json.dumps({"written": True, "out": a.out.replace("\\", "/"), "out_sha256": sha256_file(a.out), **meta}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
