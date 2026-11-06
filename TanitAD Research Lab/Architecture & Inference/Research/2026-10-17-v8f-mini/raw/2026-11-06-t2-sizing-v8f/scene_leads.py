"""Real-lead presence in the two renderable NuRec scenes, from the scenes' own annotation (sequence_tracks.json) and rig poses.
CPU only; reads two local scene folders, nothing else.  Control: scene 7c72937c should reproduce the actor_map.py docstring
(135 close-following rows, min headway 3.27 m).  Lead = a vehicle-class track whose centre lies 0..40 m ahead and |lateral| < 1.8 m
in the ego frame at the same 10 Hz tick (nearest track timestamp within 0.15 s).
CONVENTION: stack/experiments/alpasim-gsplat/scene_geometry.py:17 reads the ego-frame position as inv(T_rig_world) @ p_world, i.e. T_rig_world is the rig pose in the world:
the `world_from_rig` entry is the one that applies; `rig_from_world` is the rejected alternative, kept so the choice is visible.
Control: scene 7c72937c with `world_from_rig` gives 136 lead ticks (docstring: 135 rows) and a min centre distance 8.92 m, i.e. about 3.3 m bumper-to-bumper after the ~5.6 m overhang+half-length
(docstring: 3.27 m) -- UNVERIFIED as an exact reproduction (the overhang constant was not re-read)."""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R = "E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/"
SC = {"00040136": ("00040136-e651-4abd-991d-0655ccda9430/extracted/sequence_tracks.json", "00040136-e651-4abd-991d-0655ccda9430/rig_trajectories.json"),
      "7c72937c": ("7c72937c-c620-4776-9555-d57222c0081f/extracted/sequence_tracks.json", "7c72937c-c620-4776-9555-d57222c0081f/extracted/rig_trajectories.json")}

def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

out = {}
for sc, (tp, rp) in SC.items():
    st = json.load(open(R + tp))["dummy_chunk_id"]
    td = st["tracks_data"]
    rg = json.load(open(R + rp))["rig_trajectories"][0]
    T = np.array(rg["T_rig_worlds"], float)          # [202,4,4]
    ts = np.array(rg["T_rig_world_timestamps_us"], float)
    labels = td["tracks_label_class"]
    res = {}
    for conv in ("rig_from_world", "world_from_rig"):
        P = np.linalg.inv(T) if conv == "rig_from_world" else T          # ego pose in world: [202,4,4]
        rows = []     # per rig tick: best (smallest) headway
        for i in range(len(ts)):
            ego = P[i]
            Rinv = ego[:3, :3].T
            best = None
            for j in range(len(td["tracks_id"])):
                tt = np.array(td["tracks_timestamps_us"][j], float)
                if len(tt) == 0:
                    continue
                k = int(np.argmin(np.abs(tt - ts[i])))
                if abs(tt[k] - ts[i]) > 1.5e5:
                    continue
                pose = td["tracks_poses"][j][k]
                pw = np.array(pose[:3], float)
                pe = Rinv @ (pw - ego[:3, 3])
                if 0.0 < pe[0] < 40.0 and abs(pe[1]) < 1.8:
                    if best is None or pe[0] < best[0]:
                        best = (pe[0], pe[1], str(labels[j]))
            rows.append(best)
        hw = [r[0] for r in rows if r is not None]
        res[conv] = {"ticks_with_lead_in_corridor": len(hw), "of": len(rows), "min_headway_m": (min(hw) if hw else None),
                     "ticks_with_lead_within_15m": sum(1 for h in hw if h <= 15)}
    out[sc] = {"n_tracks": len(td["tracks_id"]), "label_classes": sorted(set(map(str, labels))), "conventions": res}
print(json.dumps(out, indent=1))
