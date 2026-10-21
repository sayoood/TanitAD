"""l7_targets_pai.py -- per-candidate PhysicalAI targets for the L7 scorer, read from a BANKED fan dump (scorer9).

INPUT (the fanceil package's dump format, ``fanceil/code/fan_roll_r8_pf.py``):
  <dump>/fan_extras.npz     clip_index, ws, anchor_traj [W,149,8,2], sel_score, refined_logits, sel_idx, traj, reach_keep
  <dump>/ep*.npz            g [n,4,2] (GT at 0.5/1/1.5/2 s), v0, os, ws, eid, clip_index
  --s6 <dir>/ep*.npz        g [n,6,2] (GT at 1/2/3/4/5/6 s = model slots 1,3,4,5,6,7), keyed (clip_index, ws)   [optional]
  --agents-npz <f>          r7_agent_fut [W,8,A,5] / _valid [W,8,A] / _label [W,8], keyed clip_index+ws           [optional;
                            written by the capture roll with ds.r7_agent_future=True, refc_v3_train.py:3743-3797]

OUTPUT (one npz + one json manifest; clip_index integers only, never a clip id):
  cost [W,N]          the imitation cost = ADE to GT over the window's valid slots of the 8-slot grid (m)
  ade_s2 [W,N]        ADE over the 2 s grid (slots 0..3) -- the registered S2 metric's per-candidate value
  along, cross [W,N]  mean |along-track| / |cross-track| error over the valid slots (fan_control_r8.py:254-266 convention)
  fde [W,N]           error at the last valid slot
  comf, ep (+ _mask)  ``refcv7_oracle.oracle_subscores`` UNCHANGED (on the valid grid)
  nc, ttc (+ _mask)   same, from --agents-npz; ABSTAIN (mask False) without it -- never a silent "no collision"
  dac (+ _mask)       ABSTAIN here (the SAM3 raster is label-side and lives on Thor; the capture roll dumps the crop)
  grid [W]            8 = full grid, 4 = 2 s only (S6 GT missing for that window)

CONTROLS written into the manifest (each must read its known value):
  C_S6  S6 slot 0 (1 s) / slot 1 (2 s) == 2 s dump slot 1 / slot 3 (same GT, two banks): max |diff| <= 1e-5 m
  C_OS  the fan at sel_idx == the bank's own plan ``os`` on the 2 s grid: max |diff| == 0 (the fanceil C4 form)
  C_GT  GT appended as an extra candidate has cost exactly 0 and ade_s2 exactly 0

    python l7_targets_pai.py --dump <fanceil/runs/s1/dump> [--s6 <bank>/s6_s1] [--agents-npz F] --limit 10 \
        --out-npz OUT.npz --out-json OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

SLOT_T = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
S6_TO_SLOT = (1, 3, 4, 5, 6, 7)          # S6 grid (1..6 s) -> model slots


def load_gt(dump: str, s6: str | None) -> dict:
    """{(clip_index, ws): {'g8': [8,2] (nan where missing), 'eid': str, 'v0': float, 'os': [4,2]}}"""
    tab = {}
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci = int(np.asarray(z["clip_index"]).ravel()[0])
        eid = str(int(np.asarray(z["eid"]).ravel()[0]))
        for j, w in enumerate(np.asarray(z["ws"]).ravel()):
            g8 = np.full((8, 2), np.nan)
            g8[:4] = np.asarray(z["g"][j], np.float64)
            tab[(ci, int(w))] = {"g8": g8, "eid": eid, "v0": float(z["v0"][j]), "os": np.asarray(z["os"][j], np.float64),
                                 "s6": None}
    if s6:
        for f in sorted(glob.glob(os.path.join(s6, "ep*.npz"))):
            z = np.load(f, allow_pickle=True)
            ci = int(np.asarray(z["clip_index"]).ravel()[0])
            for j, w in enumerate(np.asarray(z["ws"]).ravel()):
                k = (ci, int(w))
                if k in tab:
                    g6 = np.asarray(z["g"][j], np.float64)
                    tab[k]["s6"] = g6
                    for i, sl in enumerate(S6_TO_SLOT):
                        if sl >= 4:
                            tab[k]["g8"][sl] = g6[i]
    return tab


def along_cross(fan: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fan [W,N,S,2], gt [W,S,2] -> per-slot |along|, |cross| [W,N,S]; tangent = gt[j] - gt[j-1] (origin before slot 0),
    exactly fan_control_r8.py:254-266."""
    prev = np.concatenate([np.zeros_like(gt[:, :1]), gt[:, :-1]], 1)
    tan = gt - prev
    nrm = np.linalg.norm(tan, axis=-1, keepdims=True)
    tan = np.where(nrm > 1e-6, tan / np.maximum(nrm, 1e-12), np.array([1.0, 0.0]))
    nor = np.stack([-tan[..., 1], tan[..., 0]], -1)
    e = fan - gt[:, None]
    return np.abs((e * tan[:, None]).sum(-1)), np.abs((e * nor[:, None]).sum(-1))


def build(dump: str, s6: str | None, agents_npz: str | None, limit: int | None, windows: list | None = None) -> tuple:
    from tanitad.refs import refcv7_oracle as O          # imported UNCHANGED
    t_load = time.time()
    z = np.load(os.path.join(dump, "fan_extras.npz"), allow_pickle=True)
    ci = np.asarray(z["clip_index"]).astype(np.int64)
    ws = np.asarray(z["ws"]).astype(np.int64)
    gt_tab = load_gt(dump, s6)
    idx = np.arange(len(ci))
    if windows is not None:
        want = {(int(a), int(b)) for a, b in windows}
        idx = np.asarray([i for i in idx if (int(ci[i]), int(ws[i])) in want], np.int64)
    if limit:
        idx = idx[:int(limit)]
    fan_all = z["anchor_traj"]
    fan = np.asarray(fan_all[idx], np.float64)                  # [W,N,8,2]
    sel = np.asarray(z["sel_idx"][idx]).astype(np.int64)
    keys = [(int(ci[i]), int(ws[i])) for i in idx]
    miss = [k for k in keys if k not in gt_tab]
    if miss:
        raise SystemExit(f"[l7pai] {len(miss)} windows have no GT in the dump: {miss[:3]}")
    t_load = time.time() - t_load
    W, N = fan.shape[:2]
    g8 = np.stack([gt_tab[k]["g8"] for k in keys])              # [W,8,2]
    v0 = np.asarray([gt_tab[k]["v0"] for k in keys])
    valid = np.isfinite(g8).all(-1)                             # [W,8]
    grid = np.where(valid.all(1), 8, 4)
    if not valid[:, :4].all():
        raise SystemExit("[l7pai] a window lacks its 2 s GT")
    ctl = {}
    # C_S6 -- two banks agree on the shared slots
    d6 = [max(np.abs(gt_tab[k]["s6"][0] - gt_tab[k]["g8"][1]).max(), np.abs(gt_tab[k]["s6"][1] - gt_tab[k]["g8"][3]).max())
          for k in keys if gt_tab[k]["s6"] is not None]
    ctl["C_S6_shared_slots_max_abs_m"] = {"value": float(max(d6)) if d6 else None, "n": len(d6), "bar": 1e-5,
                                          "pass": (bool(max(d6) <= 1e-5) if d6 else None)}
    # C_OS -- the fan at sel_idx is the bank's own plan
    os_b = np.stack([gt_tab[k]["os"] for k in keys])
    dos = float(np.abs(fan[np.arange(W), sel, :4] - os_b).max())
    ctl["C_OS_fan_at_sel_equals_bank_os_max_abs_m"] = {"value": dos, "bar": 0.0, "pass": bool(dos == 0.0)}

    t0 = time.time()
    gz = np.where(valid[..., None], g8, 0.0)
    err = np.linalg.norm(fan - gz[:, None], axis=-1)            # [W,N,8]
    vm = valid[:, None, :].astype(np.float64)
    nval = vm.sum(-1)
    cost = (err * vm).sum(-1) / nval
    ade_s2 = err[..., :4].mean(-1)
    al, cr = along_cross(fan, gz)
    along = (al * vm).sum(-1) / nval
    cross = (cr * vm).sum(-1) / nval
    last = (valid * np.arange(1, 9)).argmax(1)                  # index of last valid slot
    fde = err[np.arange(W), :, last]
    # C_GT -- GT as a candidate costs exactly 0
    gtc = np.linalg.norm(gz - gz, axis=-1)
    ctl["C_GT_gt_as_candidate_cost_m"] = {"value": float(gtc.max()), "bar": 0.0, "pass": bool(gtc.max() == 0.0)}
    t_geom = time.time() - t0

    # refcv7 oracle sub-scores, per grid group (unchanged function; slot_t restricted to the valid grid)
    t1 = time.time()
    out = {k: np.zeros((W, N), np.float32) for k in ("comf", "ep", "nc", "ttc", "dac")}
    msk = {k + "_mask": np.zeros((W, N), bool) for k in ("comf", "ep", "nc", "ttc", "dac")}
    ag = None
    if agents_npz:
        A = np.load(agents_npz, allow_pickle=True)
        amap = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(A["clip_index"], A["ws"]))}
        ag = (A, amap)
    for gsz in (8, 4):
        rows = np.where(grid == gsz)[0]
        if rows.size == 0:
            continue
        sl = slice(0, gsz)
        st = torch.tensor(SLOT_T[:gsz], dtype=torch.float32)
        kw = {}
        if ag is not None:
            A, amap = ag
            ii = [amap.get(keys[r]) for r in rows]
            have = np.asarray([i is not None for i in ii])
            if have.all():
                ii = np.asarray(ii)
                kw["agents"] = torch.from_numpy(np.asarray(A["r7_agent_fut"][ii][:, :gsz], np.float32)).permute(0, 2, 1, 3)
                kw["agents_valid"] = torch.from_numpy(np.asarray(A["r7_agent_fut_valid"][ii][:, :gsz])).permute(0, 2, 1)
        o = O.oracle_subscores(torch.from_numpy(fan[rows][:, :, sl].astype(np.float32)), st,
                               torch.from_numpy(v0[rows].astype(np.float32)),
                               torch.from_numpy(gz[rows][:, sl].astype(np.float32)), **kw)
        for k in out:
            out[k][rows] = o[k].numpy()
            msk[k + "_mask"][rows] = o[k + "_mask"].numpy()
    t_oracle = time.time() - t1
    res = dict(clip_index=ci[idx], ws=ws[idx], eid=np.asarray([gt_tab[k]["eid"] for k in keys]), grid=grid,
               cost=cost.astype(np.float32), ade_s2=ade_s2.astype(np.float32), along=along.astype(np.float32),
               cross=cross.astype(np.float32), fde=fde.astype(np.float32), sel_idx=sel, v0=v0.astype(np.float32),
               **out, **msk)
    timing = {"windows": int(W), "candidates_per_window": int(N), "load_s": round(t_load, 3),
              "geometry_s": round(t_geom, 4), "oracle_s": round(t_oracle, 4),
              "targets_s_per_window": round((t_geom + t_oracle) / max(W, 1), 6),
              "targets_us_per_candidate": round(1e6 * (t_geom + t_oracle) / max(W * N, 1), 3)}
    return res, ctl, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--s6", default=None)
    ap.add_argument("--agents-npz", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--one-per-clip", action="store_true",
                    help="take the MIDDLE window of each clip, in clip order, up to --limit (spreads a smoke over episodes)")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    wins = None
    if a.one_per_clip:
        z = np.load(os.path.join(a.dump, "fan_extras.npz"), allow_pickle=True)
        ci, ws = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["ws"]).astype(np.int64)
        wins = []
        for c in dict.fromkeys(ci.tolist()):
            w = np.sort(ws[ci == c])
            wins.append((int(c), int(w[len(w) // 2])))
            if a.limit and len(wins) >= a.limit:
                break
    res, ctl, timing = build(a.dump, a.s6, a.agents_npz, a.limit, wins)
    np.savez_compressed(a.out_npz, **res)
    W = len(res["cost"])
    summ = {
        "producer": "l7_targets_pai.py (scorer9)", "dump": a.dump.replace("\\", "/"), "s6": a.s6, "agents_npz": a.agents_npz,
        "n_windows": W, "grid_counts": {"8": int((res["grid"] == 8).sum()), "4": int((res["grid"] == 4).sum())},
        "controls": ctl, "timing": timing, "wall_s": round(time.time() - t0, 3),
        "mask_counts": {k: int(res[k].sum()) for k in res if k.endswith("_mask")},
        "abstain_reasons": {"nc/ttc": None if a.agents_npz else "no --agents-npz: the banked fanceil dump carries no agent futures",
                            "dac": "the SAM3 drivable raster is label-side on Thor; not in the banked dump"},
        "per_window_summary": {
            "oracle_cost_m": [round(float(v), 4) for v in res["cost"].min(1)],
            "own_pick_cost_m": [round(float(res["cost"][i, res["sel_idx"][i]]), 4) for i in range(W)],
            "own_pick_ade_s2_m": [round(float(res["ade_s2"][i, res["sel_idx"][i]]), 4) for i in range(W)],
            "own_pick_cost_rank": [int((res["cost"][i] < res["cost"][i, res["sel_idx"][i]]).sum()) + 1 for i in range(W)],
            "comf_frac": [round(float(res["comf"][i].mean()), 4) for i in range(W)],
        },
    }
    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    bad = [k for k, v in ctl.items() if v.get("pass") is False]
    print("ZZL7PAI " + json.dumps({"windows": W, "controls_failed": bad, "timing": timing}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
