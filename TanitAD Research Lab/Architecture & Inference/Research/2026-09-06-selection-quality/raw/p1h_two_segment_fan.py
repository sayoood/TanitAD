"""P1h - WHAT WOULD A MINIMAL TWO-SEGMENT CANDIDATE FAMILY BUY?  (model-free)

RULE ZERO. D-SELQ-LC-VOCAB-4 showed the 117 candidates are constant-curvature
arcs whose yaw is monotone on 100 % of pairs, so none is an S-shape, and
D-SELQ-LC-DEMAND-8 showed the corpus demands a lane-change shape on 1,297 of
18,615 windows.  The decision the PI actually faces is not "is the vocabulary
short" but "what does the cheapest extension buy, and how many tokens does it
cost".  That is answerable with zero GPU.

THE EXTENSION, DEFINED BEFORE ANY NUMBER.  A two-segment candidate holds
`+a_lat` for `t_split` seconds and `-a_lat` for the rest of the 6 s horizon,
with `a_lon` held constant throughout -- i.e. exactly the existing control
alphabet, applied twice.  Nothing else changes: same integrator, same
`kappa = clamp(a_lat / max(v0, 4)^2, +-0.12)` map, same slots.

  n_new = |a_lat_used| x |t_split| x |a_lon_used|

SCORING.  Against the RECORDED 6 s ego path of each window (the same GT
D-SELQ-LC-DEMAND-8 read), by ADE over the eight model slots.  Reported on
  (a) the 1,297 STRICT lane-change windows -- the target,
  (b) ALL 18,615 scoreable windows      -- so a gain on the target is not
      bought with a loss everywhere else,
  (c) the 2,602 TURN windows            -- the CONTROL: the extension must not
      beat the existing fan there, since a turn is exactly what a
      constant-curvature arc is for.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import _env  # noqa: F401
import selstab as S

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = S.EPS
HZ = 10.0
T_S = 6.0
SLOT_TICKS = np.array(S.HORIZONS)          # 5,10,...,60
LANE_LO, LANE_HI = 2.5, 5.0
RET_NET_DEG, RET_PEAK_DEG, TURN_DEG = 10.0, 3.0, 30.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def roll2(a_lon, a_lat, t_split, v0, steps=60, dt=0.1):
    """Two-segment fan: [B, C, 8, 2] over the model's own slots."""
    from tanitad.models.kinematic import rollout_unicycle
    B = len(v0)
    C = len(a_lat)
    vv = np.maximum(v0, S.ALAT_V_FLOOR) ** 2
    kap = np.clip(a_lat[None, :] / vv[:, None], -S.KAPPA_CAP, S.KAPPA_CAP)
    k = np.arange(steps)[None, None, :] * dt                     # [1,1,T]
    sgn = np.where(k < t_split[None, :, None], 1.0, -1.0)        # [1,C,T]
    ctrl_k = kap[:, :, None] * sgn                               # [B,C,T]
    ctrl_a = np.broadcast_to(a_lon[None, :, None], ctrl_k.shape)
    c = torch.as_tensor(np.stack([ctrl_a, ctrl_k], -1),
                        dtype=torch.float32).reshape(-1, steps, 2)
    s0 = torch.zeros(B * C, 4, dtype=torch.float32)
    s0[:, 3] = torch.as_tensor(np.repeat(v0, C), dtype=torch.float32)
    st = rollout_unicycle(s0, c, dt=dt).reshape(B, C, steps, 4)
    return st[:, :, SLOT_TICKS - 1, :2].numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-windows", type=int, default=6000)
    a = ap.parse_args()

    with open(a.arm_json, encoding="utf-8") as fh:
        man = json.load(fh)["refcv3"]["manifest"]
    clip_of = {int(e["file_index"]): e["clip_id"] for e in man["episodes"]}
    off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
    n_fut = int(round(T_S * HZ))

    v0s, gts, tag = [], [], []
    for fi in sorted(clip_of):
        z = np.load(os.path.join(a.dump, f"ep{fi:03d}.npz"), allow_pickle=True)
        ws = z["ws"].astype(int)
        v0 = z["v0"].astype(np.float64)
        p = torch.load(os.path.join(EPS, clip_of[fi] + ".v2ep.pt"),
                       map_location="cpu",
                       weights_only=False)["poses"].numpy().astype(np.float64)
        for i, w in enumerate(ws):
            j0 = w + off
            if j0 + n_fut >= len(p):
                continue
            seg = p[j0:j0 + n_fut + 1]
            c, s = np.cos(-seg[0, 2]), np.sin(-seg[0, 2])
            dx, dy = seg[:, 0] - seg[0, 0], seg[:, 1] - seg[0, 1]
            lon = c * dx - s * dy
            lat = s * dx + c * dy
            yaw = wrap(seg[:, 2] - seg[0, 2])
            gts.append(np.stack([lon[SLOT_TICKS], lat[SLOT_TICKS]], -1))
            v0s.append(v0[i])
            dyt, dyp = np.degrees(yaw[-1]), np.degrees(np.abs(yaw).max())
            ret = abs(dyt) <= RET_NET_DEG and dyp >= RET_PEAK_DEG
            lc = (LANE_LO <= abs(lat[-1]) < LANE_HI) and ret
            tag.append((1 if lc else 0, 1 if abs(dyt) >= TURN_DEG else 0))
    GT = np.asarray(gts)
    V0 = np.asarray(v0s)
    TAG = np.asarray(tag)
    n = len(V0)
    print(f"[grid] {n} scoreable windows; strict lane-change "
          f"{int(TAG[:,0].sum())}; turn control {int(TAG[:,1].sum())}")

    # subsample for tractability, PRESERVING every lane-change window
    idx = np.flatnonzero(TAG[:, 0] == 1)
    rest = np.flatnonzero(TAG[:, 0] == 0)
    rng = np.random.default_rng(0)
    if len(rest) > a.max_windows - len(idx):
        rest = rng.choice(rest, a.max_windows - len(idx), replace=False)
    keep = np.sort(np.concatenate([idx, rest]))
    print(f"[grid] scoring {len(keep)} windows (all {len(idx)} lane-change "
          f"windows kept; the rest subsampled, seed 0)")
    GT, V0, TAG = GT[keep], V0[keep], TAG[keep]
    n = len(V0)

    ctrl, _ = S.anchor_controls()
    a_lon_grid = np.unique(ctrl[:, 0].numpy()).astype(np.float64)
    a_lat_grid = np.unique(ctrl[:, 1].numpy()).astype(np.float64)

    # existing 117-candidate fan, same slots
    base = np.empty((n, 117, 8, 2))
    for s0 in range(0, n, 2000):
        base[s0:s0 + 2000] = S.fan(ctrl, V0[s0:s0 + 2000]).numpy()[..., :2]
    ade_base = np.linalg.norm(base - GT[:, None], axis=-1).mean(-1)  # [n,117]
    best_base = ade_base.min(1)

    print(f"\n{'extension':<44}{'n_new':>7}{'LC ADE':>10}{'base':>9}"
          f"{'gain':>9}{'ALL ADE':>10}{'base':>9}{'TURN ADE':>11}{'base':>9}")
    res = {"n_windows_scored": int(n),
           "n_lane_change": int(TAG[:, 0].sum()),
           "n_turn_control": int(TAG[:, 1].sum()),
           "base_117": {
               "lane_change_ade": float(best_base[TAG[:, 0] == 1].mean()),
               "all_ade": float(best_base.mean()),
               "turn_ade": float(best_base[TAG[:, 1] == 1].mean())},
           "extensions": {}}
    for lat_sub, lat_name in ((a_lat_grid[np.abs(a_lat_grid) > 0], "all 8"),
                              (np.array([-1.5, -0.75, 0.75, 1.5]), "4 mid"),
                              (np.array([-0.75, 0.75]), "2 small")):
        for splits in (np.array([2.0, 3.0, 4.0]), np.array([3.0])):
            for lon_sub, lon_name in ((np.array([0.0]), "a_lon=0 only"),
                                      (a_lon_grid[::4], "a_lon every 4th")):
                A_lat = np.repeat(np.tile(lat_sub, len(splits)), len(lon_sub))
                A_spl = np.repeat(np.repeat(splits, len(lat_sub)),
                                  len(lon_sub))
                A_lon = np.tile(lon_sub, len(lat_sub) * len(splits))
                ext = np.empty((n, len(A_lat), 8, 2))
                for s0 in range(0, n, 1000):
                    ext[s0:s0 + 1000] = roll2(A_lon, A_lat, A_spl,
                                              V0[s0:s0 + 1000])
                ade_ext = np.linalg.norm(ext - GT[:, None], axis=-1).mean(-1)
                comb = np.minimum(best_base, ade_ext.min(1))
                lc = TAG[:, 0] == 1
                tn = TAG[:, 1] == 1
                nm = f"{lat_name} a_lat x {len(splits)} splits x {lon_name}"
                res["extensions"][nm] = {
                    "n_new_candidates": int(len(A_lat)),
                    "lane_change_ade": float(comb[lc].mean()),
                    "lane_change_ade_base": float(best_base[lc].mean()),
                    "all_ade": float(comb.mean()),
                    "all_ade_base": float(best_base.mean()),
                    "turn_ade": float(comb[tn].mean()),
                    "turn_ade_base": float(best_base[tn].mean()),
                    "lc_windows_where_ext_wins":
                        int((ade_ext.min(1) < best_base)[lc].sum()),
                    "all_windows_where_ext_wins":
                        int((ade_ext.min(1) < best_base).sum())}
                r = res["extensions"][nm]
                print(f"{nm:<44}{r['n_new_candidates']:>7}"
                      f"{r['lane_change_ade']:>10.4f}"
                      f"{r['lane_change_ade_base']:>9.4f}"
                      f"{r['lane_change_ade_base']-r['lane_change_ade']:>9.4f}"
                      f"{r['all_ade']:>10.4f}{r['all_ade_base']:>9.4f}"
                      f"{r['turn_ade']:>11.4f}{r['turn_ade_base']:>9.4f}")
    print(f"\n[CONTROL] the TURN columns must never IMPROVE by more than "
          f"rounding: a constant-curvature arc is exactly what a turn needs, "
          f"so a two-segment family that 'wins' there would mean the scorer, "
          f"not the geometry, is doing the work.")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
