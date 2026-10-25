"""controls_v2_rerun.py -- RE-EXECUTES v2's model-level controls with v2's own harness (read-only on viz2: bytecode writing is off) and compares the regenerated arrays with the
data that v3 draws from (viz3/data/cf_windows.npz, a copy of viz2's).  CPU only, ONE python process.  Writes viz3/CONTROLS_V3_part2.json.
  control 1 (v2): the window's own nav injected at S, T and O (and no override) reproduces forward_infer EXACTLY (0.0), 4 windows x 2 seeds.
  regeneration: own plan, S's checkpoints and ALL 12 counterfactual plans (4 injection points x 3 commands) per window and seed equal the stored arrays (<= 1e-4 m; CPU float32).
  frame accessor: v3_lib.v2ep_frame is pixel-identical to v2_lib.v2ep_frame on the frames the video draws."""
from __future__ import annotations

import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["V2_THREADS"] = "2"
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
V2D, V3D = f"{SC}/viz2", f"{SC}/viz3"
sys.path.insert(0, V2D)
import numpy as np  # noqa: E402
from v2_lib import *  # noqa: E402,F401,F403  (v2's harness: model, EvalWorld, chain, own_forward, frames; read-only)
import v2_lib as L2  # noqa: E402

t0 = time.time()
cf3 = np.load(f"{V3D}/data/cf_windows.npz")
res = {"per_window": {}, "frames": {}}
tol_exact, tol_regen = 0.0, 1e-4
for seed in SEEDS:
    model, cfg = load_model(seed)
    ew = EvalWorld(cfg, lru=4)
    for name in KEYS:
        w = WINDOWS[name]["window"]
        b = ew.batch([w])
        o = own_forward(model, b)
        c0 = chain(model, b)
        c_own = chain(model, b, nav=b["nav"].clone(), where=("S", "T", "O"))
        P = f"{seed}/{name}"
        d = {"chain_vs_forward_traj30": float((c0["traj30"] - o["traj30"]).abs().max()), "own_nav_all_layers_vs_forward_traj30": float((c_own["traj30"] - o["traj30"]).abs().max()),
             "own_nav_all_layers_vs_forward_ckpt": float((c_own["ckpt_s"] - o["ckpt"]).abs().max()), "own_nav_all_layers_vs_forward_lat": float((c_own["lat_logits"] - o["lat_logits"]).abs().max()),
             "own_nav_all_layers_vs_forward_lon": float((c_own["lon_logits"] - o["lon_logits"]).abs().max())}
        d["regen_own_traj30_vs_stored"] = float(np.abs(o["traj30"][0].numpy() - cf3[f"{P}/own_traj30"]).max())
        d["regen_own_ckpt_s_vs_stored"] = float(np.abs(o["ckpt"][0].numpy() - cf3[f"{P}/own_ckpt_s"]).max())
        worst = 0.0
        for cfgn, where in CONFIGS.items():
            for cmd in CMDS:
                c = chain(model, b, nav=cmd_tensor(b, cmd), where=where)
                worst = max(worst, float(np.abs(c["traj30"][0].numpy() - cf3[f"{P}/cf/{cfgn}/{cmd}/traj30"]).max()), float(np.abs(c["ckpt_now"][0].numpy() - cf3[f"{P}/cf/{cfgn}/{cmd}/ckpt_now"]).max()))
        d["regen_all_12_counterfactual_traj30_and_ckpt_vs_stored"] = worst
        res["per_window"][P] = d
        print(P, {k: f"{v:.1e}" for k, v in d.items()}, f"{time.time() - t0:.0f}s", flush=True)
# ---- frame accessor: v3's copy == v2's on every frame the video draws (window t and the S-age frame t - a_s, both seeds' ages)
sys.path.insert(0, V3D)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("v3_lib_ctl", f"{V3D}/v3_lib.py")
L3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L3)
nfr, nbad = 0, 0
for name in KEYS:
    clip, t = WINDOWS[name]["clip"], WINDOWS[name]["t"]
    js = {t} | {t - int(cf3[f"{sd}/{name}/a_s"]) for sd in SEEDS}
    for j in sorted(js):
        a, bb = L2.v2ep_frame(clip, int(j)), L3.v2ep_frame(clip, int(j))
        nfr += 1
        nbad += int(not np.array_equal(a, bb))
res["frames"] = {"n_frames_compared": nfr, "n_not_identical": nbad}
pw = res["per_window"].values()
mx = lambda k: max(v[k] for v in pw)
res["control_1_v2_own_nav_all_layers_reproduces_forward"] = {k: mx(k) for k in ("chain_vs_forward_traj30", "own_nav_all_layers_vs_forward_traj30", "own_nav_all_layers_vs_forward_ckpt",
                                                                                   "own_nav_all_layers_vs_forward_lat", "own_nav_all_layers_vs_forward_lon")}
res["control_1_v2_own_nav_all_layers_reproduces_forward"].update({"tolerance": tol_exact, "n_window_seed_pairs": len(res["per_window"]),
                                                                   "PASS": all(v == 0.0 for k, v in res["control_1_v2_own_nav_all_layers_reproduces_forward"].items())})
res["regeneration_of_the_drawn_counterfactual_arrays"] = {"max_abs_m": max(mx("regen_own_traj30_vs_stored"), mx("regen_own_ckpt_s_vs_stored"), mx("regen_all_12_counterfactual_traj30_and_ckpt_vs_stored")),
                                                           "tolerance": tol_regen, "n_counterfactual_cells_regenerated": 12 * len(res["per_window"])}
res["regeneration_of_the_drawn_counterfactual_arrays"]["PASS"] = res["regeneration_of_the_drawn_counterfactual_arrays"]["max_abs_m"] <= tol_regen
res["frame_accessor_pixel_identical_to_v2"] = {**res["frames"], "PASS": nbad == 0 and nfr > 0}
res["seconds"] = time.time() - t0
res["ALL_PASS"] = all(res[k]["PASS"] for k in ("control_1_v2_own_nav_all_layers_reproduces_forward", "regeneration_of_the_drawn_counterfactual_arrays", "frame_accessor_pixel_identical_to_v2"))
json.dump(res, open(f"{V3D}/CONTROLS_V3_part2.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k not in ("per_window", "frames")}, indent=1))
