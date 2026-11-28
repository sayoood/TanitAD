#!/usr/bin/env python3
"""build_strategic_tok.py -- per-frame strategic goal tokens over the PI's 6-30 s band (stream A, priority 3).

Re-emits ``g_str`` / ``a_str`` at every 0.2 s cache step with the v7 emitter's own ``strategic()``
(``s2_geom_emit_v7.py``, a verbatim copy, md5 in the manifest) run on a 10 Hz track that STARTS at NOW. Two runtime
changes to the emitter's module constants, both listed in SCHEMA.md and nowhere else:
  STRATEGIC_S = (8.0, 30.0) -> (6.0, 30.0)   (the PI's R-STR band; closes the 6-8 s span)
  GAP_S       = (6.0, 8.0)  -> (6.0, 6.0)    (no unassigned span is left)
TACTICAL_S stays (2, 6) and PLAN_S stays 6.
Layout is the same as ``strategic_geom_*.npz`` (same clip order, ``clip__off`` / ``clip__n``, the same N rows), so the
two files join by row index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_strategic_geom as SG  # noqa: E402
import build_v9_verbatim as B  # noqa: E402

SCHEMA = "tanitad.band_labels.strategic_tok/1"
BAND = (6.0, 30.0)
OBS_FULL_S = 29.95            # an absence claim (FOLLOW_ROUTE, HOLD_MAIN_ROAD) needs the whole band observed
OBS_MIN_S = 3.0               # v9 literal h_min_any_s: below this nothing is claimed


def emitter():
    S2 = B._builder()
    S2.STRATEGIC_S = BAND
    S2.GAP_S = (BAND[0], BAND[0])
    return S2


def vocab():
    from tanitad.models import vocab_v7 as V7
    return tuple(V7.STRATEGIC_GOAL_TOKENS_V7), tuple(V7.STRATEGIC_ACTION_TOKENS_V7)


def frame_tokens(S2, log, now, G, A):
    """One frame -> dict. ``G`` / ``A`` the token tuples."""
    out = dict(g=-1, a=-1, by=np.nan, within=np.nan, hold=np.nan, vt=np.nan, obs=0.0, gv=False, av=False)
    if now < -1e-9:
        return out
    obs = min(B.h_obs(log, now, horizon=BAND[1]), BAND[1])
    out["obs"] = obs
    if obs < OBS_MIN_S:
        return out
    tg = now + np.arange(0, int(np.floor(obs * 10 + 1e-9)) + 1) * 0.1
    x, y, yaw, v = B.sample(log, tg)
    p = np.stack([x, y, yaw, v], axis=1)
    seq = S2.manoeuvre_sequence(p, 0, hz=10.0)
    g, a = S2.strategic(p, 0, seq, 10.0)
    gi, ai = G.index(g["token"]), A.index(a["token"])
    args = g.get("args") or {}
    out.update(g=gi, a=ai, by=float(args.get("by_time_s", np.nan)), within=float(args.get("within_m", np.nan)),
               hold=float(args.get("hold_for_s", np.nan)),
               vt=float((a.get("args") or {}).get("v_target_ms", np.nan)))
    full = obs >= OBS_FULL_S
    out["gv"] = bool(g["token"] != "FOLLOW_ROUTE" or full)
    out["av"] = bool(a["token"] not in ("HOLD_MAIN_ROAD",) or full)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "eval"))
    ap.add_argument("--out-dir", default="F:/strat_out/band_labels")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", default=None)
    a = ap.parse_args(argv)
    B.assert_no_g_imports()
    S2 = emitter()
    G, A = vocab()
    U = SG.clip_universe(a.split, a.limit)
    keys = ("g_str_id", "a_str_id", "by_time_s", "within_m", "hold_for_s", "v_target_ms", "str_obs_s", "g_valid", "a_valid")
    cols = {k: [] for k in keys}
    ids, off, t0 = [], 0, time.time()
    for i, (cid, g0, dt, src, n, inm, inc) in enumerate(U):
        log = B.load_log(SG.EGO.format(cid))
        nows = g0 + 2 * np.arange(n) * dt
        rows = [frame_tokens(S2, log, float(t), G, A) for t in nows]
        for k, f in zip(keys, ("g", "a", "by", "within", "hold", "vt", "obs", "gv", "av")):
            cols[k].append(np.array([r[f] for r in rows]))
        ids.append(cid)
        off += n
        if (i + 1) % 100 == 0 or i + 1 == len(U):
            msg = f"[strategic_tok {a.split}] {i + 1}/{len(U)} {time.time() - t0:.0f}s"
            print(msg, flush=True)
            if a.progress:
                open(a.progress, "w", encoding="utf-8").write(msg + "\n")
    dt_ = {"g_str_id": np.int8, "a_str_id": np.int8, "by_time_s": np.float32, "within_m": np.float32,
           "hold_for_s": np.float32, "v_target_ms": np.float32, "str_obs_s": np.float32, "g_valid": bool, "a_valid": bool}
    out = {k: np.concatenate(v).astype(dt_[k]) for k, v in cols.items()}
    out.update({"clip_ids": np.array(ids), "g_tokens": np.array(G), "a_tokens": np.array(A), "band_s": np.array(BAND, np.float32),
                "clip__n": np.array([u[4] for u in U], np.int64)})
    out["clip__off"] = np.concatenate([[0], np.cumsum(out["clip__n"])[:-1]]).astype(np.int64)
    os.makedirs(a.out_dir, exist_ok=True)
    npz = os.path.join(a.out_dir, f"strategic_tok_{a.split}.npz")
    np.savez_compressed(npz, **out)
    man = {"schema": SCHEMA, "split": a.split, "n_clips": len(ids), "n_rows": int(off), "keys": sorted(out), "band_s": list(BAND),
           "module_overrides": {"STRATEGIC_S": [8.0, 30.0, "->", *BAND], "GAP_S": [6.0, 8.0, "->", BAND[0], BAND[0]]},
           "obs_full_s": OBS_FULL_S, "obs_min_s": OBS_MIN_S, "npz_md5": B.file_md5(npz), "wall_s": round(time.time() - t0, 1),
           "code": {"build_strategic_tok.py": B.file_md5(__file__), "s2_geom_emit_v7.py": B.file_md5(S2.__file__),
                    "build_v9_verbatim.py": B.file_md5(B.__file__)}}
    with open(os.path.join(a.out_dir, f"strategic_tok_{a.split}.manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"[strategic_tok {a.split}] done -> {npz}")


if __name__ == "__main__":
    main()
