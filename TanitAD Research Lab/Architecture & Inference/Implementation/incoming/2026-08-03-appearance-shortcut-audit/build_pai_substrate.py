"""D-APPEAR stage 1 — the PhysicalAI-AV substrate for the appearance-shortcut audit.

Pre-registration: ``Project Steering/PREREG_APPEARANCE_SHORTCUT.md``.

WHAT THIS BUILDS, AND WHY IT IS 0-GPU
    The comma2k19 finding (LATENT_BOTTLENECK.md Sec 0.0) needs the SAME contrast on a
    non-highway corpus. Two substrates on the SAME rows are needed:

      pixels   built here from the banked PhysicalAI episode caches
               ([T, 9, 256, 256] uint8 -- the D-015 3-frame RGB stack);
      latent   the FROZEN v1 encoder's 2048-d state, ALREADY BANKED by the sitclf
               stream in ``sitclf_b4_substrate.npz`` (500 clips x 199 frames).
               ⭐ Same checkpoint the comma latents came from
               (``v1_speedjerk_ckpt.pt`` step 29999) -- so the cross-corpus
               contrast is ENCODER-MATCHED and no GPU is needed here at all.

⛔ THE ALIGNMENT IS VERIFIED, NOT ASSUMED (this is the whole run's foundation)
    Three independent checks, all asserted at build time:
      A1  the npz's ``E[:, 0] * ego_scale[0]`` reproduces the episode cache's
          ``poses[:, 3]`` (true ego speed) to float32 rounding;
      A2  the episode index -> clip_id map is reconstructed from
          ``discover_r0_clips`` + ``split_clips`` and checked against the cache's
          own ``episode_id`` hash (``clip_id[:4]`` big-endian);
      A3  per-clip ``cy`` is read from the LOCAL intrinsics parquets and
          cross-checked against the independent ``2026-07-22-idm-proof/rig_table.json``.
    A1 is the one that matters: it proves latent row (cache_tag, clip, t) is the same
    physical frame as pixel row t of ``ep_%05d.pt``.

usage:
  python build_pai_substrate.py --n-episodes 240 --stride 6
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

PAI_ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
EPC_TRAIN = PAI_ROOT / "_epcache" / "physicalai-train-14231cd29c74"
EPC_VAL = PAI_ROOT / "_epcache" / "physicalai-val-bb543bdf7836"
SITCLF_NPZ = Path(r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
SITCLF_META = Path(r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.meta.json")
RIG_TABLE = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
             "Implementation" / "incoming" / "2026-07-22-idm-proof" / "rig_table.json")
OUT_CACHE = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_pai_substrate.pt")

K = 4                                        # 9-position NON-CAUSAL window (800 ms)
HORIZONS = (5, 10, 15, 20)
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
RIG_CY_SPLIT = 650.0                         # the empirical gap: rig A < 600 < 700 < rig B
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------------------- #
# pixel bases -- byte-identical semantics to run_temporal_falsifier.py         #
# --------------------------------------------------------------------------- #
LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)


def _gray_latest(fr_u8: torch.Tensor) -> torch.Tensor:
    """[T, 9, 256, 256] uint8 -> [T, 1, 256, 256] float32 luma of the LATEST triplet.

    Slicing BEFORE the float cast is a memory optimisation only: the arithmetic is
    identical to ``pool_gray``'s (cast-then-slice) to the last bit, because the cast
    is elementwise.
    """
    x = fr_u8[:, 6:9].float() / 255.0
    return (x * LUMA).sum(1, keepdim=True)


def pool_gray_latest(fr_u8: torch.Tensor, side: int) -> torch.Tensor:
    g = _gray_latest(fr_u8)
    return torch.nn.functional.avg_pool2d(g, g.shape[-1] // side).reshape(g.shape[0], -1)


def motion_energy(fr_u8: torch.Tensor, side: int) -> torch.Tensor:
    """avgpool(|I_{t+1} - I_t|) at FULL 256x256, pooled only AFTERWARDS.

    Pooling before differencing cancels opposing-sign gradients -- the defect that
    produced a false pixel null in the comma run's pass 1. Forward difference on
    purpose (the last row is zero and is never a window centre)."""
    g = _gray_latest(fr_u8)
    d = torch.zeros_like(g)
    d[:-1] = (g[1:] - g[:-1]).abs()
    return torch.nn.functional.avg_pool2d(d, g.shape[-1] // side).reshape(g.shape[0], -1)


def stack_gray(fr_u8: torch.Tensor, side: int) -> torch.Tensor:
    """[T, 9, 256, 256] -> [T, 3, side*side]: the 3 sub-frames of the D-015 stack at
    each index -- exactly what REF-C's kept ``fmap[:, -1]`` is computed from."""
    x = fr_u8.float() / 255.0
    k = x.shape[-1] // side
    subs = [(x[:, 3 * i:3 * i + 3] * LUMA).sum(1, keepdim=True) for i in range(3)]
    g = [torch.nn.functional.avg_pool2d(s, k).reshape(x.shape[0], -1) for s in subs]
    return torch.stack(g, dim=1)


PIX_GRIDS = (("pix32", 32), ("pix8", 8), ("pix1", 1))
MOT_GRIDS = (("mot16", 16), ("mot8", 8))
STK_GRIDS = (("stk32", 32),)


# --------------------------------------------------------------------------- #
# labels                                                                       #
# --------------------------------------------------------------------------- #
def build_targets(poses, actions, t):
    """(scalars [N,4], traj [N,H,2]).

    ⚠️ NO ``heading_repair`` here, and that is deliberate. comma2k19's heading is
    ``arctan2`` of ENU velocity and is UNDEFINED at standstill, which is why the comma
    panel repairs it. PhysicalAI's yaw comes from the ORIENTATION QUATERNION
    (``physicalai.py:637-641``: "yaw = orientation-quaternion heading
    (standstill-robust)"), so the repair would be a no-op at best and a corpus-specific
    distortion at worst. The rest of the recipe is byte-identical.
    """
    import idm_head as ih
    yaw = poses[:, 2]
    speed, steer, accel = poses[t, 3], actions[t, 0], actions[t, 1]
    yr = ih.wrap_to_pi(yaw[t + 1] - yaw[t - 1]) / (2.0 * ih.DT)
    scal = torch.stack([speed, yr, steer, accel], dim=-1)
    yaw0, xy0 = poses[t, 2], poses[t, :2]
    traj = torch.stack([ih.ego_frame(poses[t + h, :2] - xy0, yaw0) for h in HORIZONS], 1)
    return scal, traj


# --------------------------------------------------------------------------- #
def load_clip_table():
    """episode index -> clip_id -> cy -> rig, all reconstructed deterministically."""
    from tanitad.data.physicalai import discover_r0_clips, split_clips
    clips = discover_r0_clips(PAI_ROOT)
    tr, va = split_clips(clips, val_frac=0.2, seed=0)
    cy = {}
    for f in sorted(glob.glob(str(PAI_ROOT / "calibration" / "camera_intrinsics" / "*.parquet"))):
        import pandas as pd
        d = pd.read_parquet(f).reset_index()
        d = d[d["camera_name"] == "camera_front_wide_120fov"]
        for cid, c in zip(d["clip_id"].astype(str), d["cy"].astype(float)):
            cy[cid] = c
    # A3 -- second, independent source for the same fact
    n_agree = n_dis = 0
    if RIG_TABLE.exists():
        rt = json.load(open(RIG_TABLE))
        for v in rt.values():
            if v["clip_id"] in cy:
                if abs(v["cy"] - cy[v["clip_id"]]) < 1e-3:
                    n_agree += 1
                else:
                    n_dis += 1
    assert n_dis == 0, f"cy disagrees with rig_table.json on {n_dis} clips"
    log(f"A3 cy cross-check vs rig_table.json: {n_agree} agree, {n_dis} disagree")

    rows = []
    for tag, lst in ((0, tr), (1, va)):
        for i, c in enumerate(lst):
            cid = c["clip_id"]
            rows.append({"cache_tag": tag, "ep_index": i, "clip_id": cid,
                         "cy": cy.get(cid),
                         "rig": None if cid not in cy else
                               ("a" if cy[cid] < RIG_CY_SPLIT else "b"),
                         "epid": int.from_bytes(cid.encode()[:4].ljust(4, b"\0"), "big")})
    return rows, n_agree


def load_latents():
    z = np.load(SITCLF_NPZ, allow_pickle=True)
    F, cc, ct, tt, E = z["F"], z["clip_cluster"], z["cache_tag"], z["t"], z["E"]
    meta = json.load(open(SITCLF_META))
    ego_scale = meta["ego_scale"]
    by = {}
    order = np.lexsort((tt, cc, ct))
    ct, cc, tt = ct[order], cc[order], tt[order]
    F, E = F[order], E[order]
    starts = np.flatnonzero(np.r_[True, (cc[1:] != cc[:-1]) | (ct[1:] != ct[:-1])])
    ends = np.r_[starts[1:], len(cc)]
    for s, e in zip(starts, ends):
        by[(int(ct[s]), int(cc[s]))] = (F[s:e], tt[s:e], E[s:e, 0] * ego_scale[0])
    return by, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=240)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_CACHE))
    ap.add_argument("--rig", default="", help="restrict to rig 'a' or 'b' (P2)")
    a = ap.parse_args()

    import idm_head as ih
    rows, n_agree = load_clip_table()
    lat, meta = load_latents()
    log(f"latent bank: {len(lat)} clips, encoder {meta['trunk']['ckpt']} "
        f"step {meta['trunk']['step']}, frozen={meta['trunk']['frozen']}")

    # clip_cluster == train index for tag 0, 400 + val index for tag 1
    def cluster(r):
        return r["ep_index"] if r["cache_tag"] == 0 else 400 + r["ep_index"]

    sel = [r for r in rows if (not a.rig or r["rig"] == a.rig)]
    sel = sorted(sel, key=cluster)[:a.n_episodes]
    log(f"selected {len(sel)} episodes (rig filter={a.rig or 'none'})")

    offs = torch.arange(-K, K + 1)
    out = {"episodes": [], "k": K, "stride": a.stride,
           "grids": {**dict(PIX_GRIDS), **dict(MOT_GRIDS), **dict(STK_GRIDS)},
           "encoder": meta["trunk"]["ckpt"], "encoder_step": meta["trunk"]["step"],
           "caches": [str(EPC_TRAIN), str(EPC_VAL)],
           "corpus": "physicalai r0 500-clip build (NOT the canonical "
                     "physicalai-train-e438721ae894 training corpus)",
           "alignment_checks": {"A3_cy_vs_rig_table_agree": n_agree}}
    a1_max = 0.0
    skipped = []
    for i, r in enumerate(sel):
        base = EPC_TRAIN if r["cache_tag"] == 0 else EPC_VAL
        p = base / f"ep_{r['ep_index']:05d}.pt"
        e = torch.load(p, map_location="cpu", weights_only=False)
        # A2 -- the episode index -> clip_id map
        assert int(e["episode_id"]) == r["epid"], (
            f"{p}: episode_id {e['episode_id']} != clip_id hash {r['epid']} "
            f"for {r['clip_id']} -- the ep->clip map is WRONG")
        key = (r["cache_tag"], cluster(r))
        if key not in lat:
            skipped.append((r["clip_id"], "no latent"))
            continue
        Fz, tt, v_lat = lat[key]
        fr = e["frames_u8"]
        T = fr.shape[0]
        if len(tt) != T or int(tt[0]) != 0 or int(tt[-1]) != T - 1:
            skipped.append((r["clip_id"], f"latent rows {len(tt)} != T {T}"))
            continue
        poses, actions = e["poses"].float(), e["actions"].float()
        # ⭐ A1 -- THE decisive alignment check: the latent bank's own ego speed must be
        # the episode cache's pose speed, row for row.
        d1 = float(np.abs(v_lat - poses[:, 3].numpy()).max())
        a1_max = max(a1_max, d1)
        assert d1 < 1e-3, f"{r['clip_id']}: A1 speed mismatch {d1}"

        t = ih.valid_centers(T, K, HORIZONS, a.stride)
        if t.numel() == 0:
            skipped.append((r["clip_id"], "no valid centres"))
            continue
        rec = {"name": f"{r['cache_tag']}_{r['ep_index']:05d}", "clip_id": r["clip_id"],
               "n": int(t.numel()), "t": t.clone(), "rig": r["rig"], "cy": r["cy"]}
        Z = torch.from_numpy(np.asarray(Fz, dtype=np.float16))
        rec["Z"] = Z[t[:, None] + offs[None, :]]                   # [N, 9, 2048]
        for name, side in PIX_GRIDS:
            g = pool_gray_latest(fr, side)
            rec[name] = g[t[:, None] + offs[None, :]].half()
        for name, side in MOT_GRIDS:
            g = motion_energy(fr, side)
            rec[name] = g[t[:, None] + offs[None, :]].half()
        for name, side in STK_GRIDS:
            rec[name] = stack_gray(fr, side)[t].half()
        S, Tj = build_targets(poses, actions, t)
        rec["S"], rec["T"] = S, Tj
        rec["Q"] = ih.speed_seq_targets_at(poses, t, K)             # [N, 9] true speed
        rec["man"] = e["maneuvers"][t].clone()
        out["episodes"].append(rec)
        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(sel)} episodes, {sum(x['n'] for x in out['episodes'])} windows")
    out["alignment_checks"]["A1_max_abs_speed_delta_mps"] = a1_max
    out["skipped"] = skipped
    log(f"A1 max |v_latent_bank - v_episode_cache| = {a1_max:.2e} m/s over "
        f"{len(out['episodes'])} episodes")
    log(f"skipped {len(skipped)}: {skipped[:5]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    n_win = sum(x["n"] for x in out["episodes"])
    rigs = {}
    for x in out["episodes"]:
        rigs[x["rig"]] = rigs.get(x["rig"], 0) + 1
    log(f"saved {a.out}: {len(out['episodes'])} episodes, {n_win} windows, rigs {rigs}")


if __name__ == "__main__":
    main()
