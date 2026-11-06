"""Scene 7c72937c: what does the scene's own annotation say about the vehicle ahead around the human's launch?  CPU, local scene files only.
World-from-rig convention as in scene_leads.py (scene_geometry.py:17).  Lead = nearest vehicle-class track centre 0..40 m ahead, |lateral| < 1.8 m.
Speeds by central difference over +-0.3 s."""
import json, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R = "E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/7c72937c-c620-4776-9555-d57222c0081f/extracted/"
VEH = {"automobile", "bus", "heavy_truck", "trailer"}
st = json.load(open(R + "sequence_tracks.json"))["dummy_chunk_id"]["tracks_data"]
rg = json.load(open(R + "rig_trajectories.json"))["rig_trajectories"][0]
T = np.array(rg["T_rig_worlds"], float)
ts = np.array(rg["T_rig_world_timestamps_us"], float)
pos = T[:, :3, 3]
def spd(arr, t, i, w=0.3e6):
    a = int(np.argmin(np.abs(t - (t[i] - w)))); b = int(np.argmin(np.abs(t - (t[i] + w))))
    return float(np.linalg.norm(arr[b] - arr[a]) / max((t[b] - t[a]) * 1e-6, 1e-6)) if b != a else float("nan")
v_ego = np.array([spd(pos, ts, i) for i in range(len(ts))])
i_cross = int(np.argmax(v_ego >= 5.2))
rows = []
for i in range(max(0, i_cross - 30), min(len(ts), i_cross + 31), 5):
    ego = T[i]; Rinv = ego[:3, :3].T; best = None
    for j in range(len(st["tracks_id"])):
        if str(st["tracks_label_class"][j]) not in VEH:
            continue
        tt = np.array(st["tracks_timestamps_us"][j], float)
        if len(tt) < 3:
            continue
        k = int(np.argmin(np.abs(tt - ts[i])))
        if abs(tt[k] - ts[i]) > 1.5e5:
            continue
        P = np.array([p[:3] for p in st["tracks_poses"][j]], float)
        pe = Rinv @ (P[k] - ego[:3, 3])
        if 0 < pe[0] < 40 and abs(pe[1]) < 1.8 and (best is None or pe[0] < best[0]):
            best = (float(pe[0]), float(pe[1]), spd(P, tt, k), str(st["tracks_label_class"][j]))
    rows.append({"rig_tick": i, "t_rel_to_cross_s": round((ts[i] - ts[i_cross]) * 1e-6, 2), "v_ego": round(float(v_ego[i]), 2),
                 "lead_centre_dist_m": None if best is None else round(best[0], 2), "lead_speed": None if best is None else round(best[2], 2),
                 "lead_class": None if best is None else best[3]})
print(json.dumps({"i_cross_5p2": i_cross, "ego_speed_max": float(np.nanmax(v_ego)), "rows": rows}, indent=1))
