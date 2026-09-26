"""Can the N (batch) dimension score all proposals in ONE pass? 64x is the difference between a
15-minute bank and a 15-hour one. ⛔ A fast path that changes the answer is worthless, so the whole
point of this probe is the EQUALITY CHECK against the loop."""
import sys, time, copy, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP

def tile(sd, N):
    out = copy.deepcopy(sd)
    for f in list(vars(out).keys()) if hasattr(out, "__dict__") else []:
        if f.startswith("_"): continue
        try: v = getattr(out, f)
        except Exception: continue
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            try: setattr(out, f, v.repeat(N, *([1] * (v.dim() - 1))).contiguous())
            except Exception: pass
    acm = getattr(out, "agent_control_manager", None)
    if acm is not None:
        # controlled_mask / log_replay_mask are READ-ONLY properties derived from _control_types
        v = getattr(acm, "_control_types", None)
        if torch.is_tensor(v) and v.shape[0] == 1:
            acm._control_types = v.repeat(N, *([1] * (v.dim() - 1))).contiguous()
    if hasattr(out, "clean_polygon_cache"): out.clean_polygon_cache()
    return out

log = sys.argv[1]
cfg1, _ = G.build_engine_config(batch_size=1)
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(log, 40)
gxy, gyw = G.teacher_future(smps, step)
N = 8
props = [gxy.clone() for _ in range(N)]
for i in range(N): props[i][:, 1] += torch.linspace(0, i * 2.0, 20)
c1 = SP.build_calculators(cfg1)
t0 = time.time(); loop = [SP.score_proposal(sd, log_sd, c1, p, gyw) for p in props]
t_loop = time.time() - t0

cfgN, _ = G.build_engine_config(batch_size=N)
cN = SP.build_calculators(cfgN)
sdN, log_sdN = tile(sd, N), tile(log_sd, N)
print(f"tiled: agent_positions_all {tuple(sdN.agent_positions_all.shape)}  "
      f"lanes_points {tuple(sdN.lanes_points.shape)}  controlled {tuple(sdN.agent_control_manager.controlled_mask.shape)}")
t0 = time.time()
cand = copy.deepcopy(sdN)
pos = cand.agent_positions_all; T = 20
add = pos[:, :, -1:, :].repeat(1, 1, T, 1)
for i, p in enumerate(props): add[i, 0] = p
cand.agent_positions_all = torch.cat([pos, add], 2)
o = cand.agent_orientation_all; addo = o[:, :, -1:].repeat(1, 1, T)
for i in range(N): addo[i, 0] = gyw
cand.agent_orientation_all = torch.cat([o, addo], 2)
for nm in ("npc_mask_all", "agent_size_all", "agent_type_all", "agent_velocity_all"):
    t = getattr(cand, nm, None)
    if t is not None and t.dim() >= 3:
        setattr(cand, nm, torch.cat([t, t[:, :, -1:].repeat(*([1,1,T] + [1]*(t.dim()-3)))], 2))
cand.clean_polygon_cache()
try: cand.update_nearest_neighbors()
except Exception as e: print("  nearest_neighbors:", type(e).__name__, e)
shared = {}
for nm in SP.CALC_ORDER:
    try: cN[nm].forward(cand, log_sdN, shared)
    except Exception as e: print(f"  {nm} ERR {type(e).__name__}: {str(e)[:70]}")
t_batch = time.time() - t0
print(f"\ntiming: loop {t_loop*1000:.0f} ms   batched {t_batch*1000:.0f} ms   "
      f"speedup {t_loop/max(t_batch,1e-9):.1f}x")
print(f"\n{'key':40s} {'row':>4s} {'loop':>10s} {'batched':>10s}  match")
bad = 0
for key in ("OffRoad", "CrossLane", "CenterLine", "NuPlanCollision", "NuPlanTTC", "Comfort"):
    if key not in shared: continue
    d = shared[key]; v = d.get("info", d.get("reward")) if isinstance(d, dict) else d
    if not torch.is_tensor(v): continue
    for i in (0, N // 2, N - 1):
        lk = next((k for k in loop[i] if k.endswith(f"{key}.info")), None)
        if lk is None: continue
        lv = loop[i][lk]; bv = float(v.float()[i, 0]) if v.dim() >= 2 else float(v.float()[i])
        m = abs(lv - bv) < 1e-6; bad += (not m)
        print(f"{key + '.info':40s} {i:4d} {lv:10.4f} {bv:10.4f}  {'OK' if m else 'MISMATCH'}")
print(f"\n{'BATCHED PATH AGREES' if bad == 0 else f'BATCHED PATH DIVERGES ({bad} mismatches)'}")
