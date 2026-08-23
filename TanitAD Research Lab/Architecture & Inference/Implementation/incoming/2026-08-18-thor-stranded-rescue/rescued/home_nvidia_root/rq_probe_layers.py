#!/usr/bin/env python3
"""Structural probe: what is actually IN dynamic_rigids / dynamic_deformables, and
what does the scene declare about the sky + the NRE up-axis.

Everything here is a MEASUREMENT of the file. No assumption is carried forward.
"""
import json
import sys
from pathlib import Path

import numpy as np

SCENE = Path(sys.argv[1]).expanduser()
sys.path.insert(0, "/home/nvidia/nurec-gsplat")
sys.path.insert(0, "/home/nvidia/tanitad_cl/stack/experiments/alpasim-gsplat")

from nurec_loader import (LAYERS, NuRecScene, RigTrajectories,  # noqa: E402
                          read_volume_nurec)

out = {"scene": str(SCENE)}
nre = read_volume_nurec(SCENE / "volume.nurec")
sc = NuRecScene(nre, quat_layout="wxyz")
rig = RigTrajectories(SCENE / "rig_trajectories.json")

out["config_layer_keys"] = sorted(sc.cfg["layers"].keys())
out["state_dict_layer_prefixes"] = sorted(
    {k.split(".")[2] for k in sc.sd if k.startswith(".gaussians_nodes.")})

per = {}
for L in LAYERS:
    if f".gaussians_nodes.{L}.positions" not in sc.sd:
        per[L] = {"present": False}
        continue
    n = sc.n_gaussians(L)
    g = sc.gaussians(L)
    r = sc.raw(L)
    d = {
        "present": True,
        "n_raw": int(n),
        "n_after_finite_filter": int(g.means.shape[0]),
        "n_dropped": int(g.n_dropped),
        "fourier_dim": int(sc.fourier_dim(L)),
        "activations": sc.activations(L),
        "pos_min": [float(x) for x in r["positions"].min(0)],
        "pos_max": [float(x) for x in r["positions"].max(0)],
        "pos_mean": [float(x) for x in np.nan_to_num(r["positions"]).mean(0)],
        "scale_median_m": [float(x) for x in np.median(np.exp(r["log_scales"]), 0)],
        "opacity_mean": float(np.nan_to_num(g.opacities).mean()),
    }
    # sub-keys the layer carries (cuboid ids, time embed, etc.)
    d["subkeys"] = sorted(k[len(f".gaussians_nodes.{L}."):] for k in sc.sd
                          if k.startswith(f".gaussians_nodes.{L}.") and not k.endswith(".shape"))
    ck = f".gaussians_nodes.{L}.gaussian_cuboid_ids"
    if ck in sc.sd:
        cid = np.frombuffer(sc.sd[ck], np.int32)
        d["n_cuboids"] = int(np.unique(cid).size)
        d["cuboid_id_range"] = [int(cid.min()), int(cid.max())]
    tk = f".gaussians_nodes.{L}.time_embed.timestamps_us_ranges"
    if tk in sc.sd:
        rr = np.frombuffer(sc.sd[tk], np.int64).reshape(-1, 2)
        d["n_layer_tracks"] = int(rr.shape[0])
        d["track_span_us_median"] = float(np.median(rr[:, 1] - rr[:, 0]))
    per[L] = d
out["layers"] = per

# --- sky ---------------------------------------------------------------------
try:
    cube = sc.sky_cubemap()
    out["sky"] = {"present": cube is not None}
    if cube is not None:
        out["sky"].update(shape=list(cube.shape),
                          mean=float(np.nan_to_num(cube).mean()),
                          per_face_mean=[float(np.nan_to_num(cube[i]).mean()) for i in range(6)],
                          p99=float(np.nanpercentile(np.nan_to_num(cube), 99)))
except Exception as e:  # noqa: BLE001
    out["sky"] = {"present": False, "error": repr(e)}
out["config_background"] = {k: v for k, v in sc.cfg.get("background", {}).items()
                            if not isinstance(v, (list, dict))}

# --- the NRE up-axis, MEASURED from the rig, not assumed -----------------------
cam = "camera_front_wide_120fov"
w2n = rig.world_to_nre
ups = []
for f in (0, 50, 100, 150, 200):
    try:
        T = rig.T_rig_world(cam, f, 1)
    except Exception:  # noqa: BLE001
        continue
    ups.append((w2n[:3, :3] @ T[:3, :3] @ np.array([0.0, 0.0, 1.0])))
ups = np.array(ups)
out["nre_up_from_rig_z"] = {"per_frame": ups.tolist(),
                            "mean": ups.mean(0).tolist(),
                            "max_spread": float(np.abs(ups - ups.mean(0)).max())}
# cross-check: the road layer's plane normal in NRE coords
rp = sc.raw("road")["positions"]
rp = rp[np.isfinite(rp).all(1)]
sub = rp[np.random.default_rng(0).choice(rp.shape[0], size=min(200000, rp.shape[0]), replace=False)]
c = sub - sub.mean(0)
_, _, vt = np.linalg.svd(c, full_matrices=False)
n_road = vt[2] * np.sign(vt[2] @ ups.mean(0))
out["nre_up_from_road_plane_pca"] = n_road.tolist()
out["up_agreement_dot"] = float(n_road @ (ups.mean(0) / np.linalg.norm(ups.mean(0))))

print(json.dumps(out, indent=1, default=float))
