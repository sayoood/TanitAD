"""Local driver: RUNG 3 WM-MPPI/CEM (arm C) vs the single-step re-plan (arm B) and
single-shot open-loop (arm A) on the flagship v1 world model, RTX 4060, no pod.

Reuses the task-#21 no-renderer harness (taniteval.closedloop) + wm_mpc.py. Same ckpt
+ same 12 held-out val episodes as the -0.213 imagination proof, so C, B, A are paired
window-for-window. Writes wm_mpc_result.json.

  --smoke   2 episodes, headline config only (correctness + timing calibration)
  (default) 12 episodes, headline + pre-registered secondary grid + batch=1 timing
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

LOCAL = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
sys.path.insert(0, LOCAL + r"\stack")
sys.path.insert(0, LOCAL + r"\stack\scripts")
sys.path.insert(0, LOCAL + r"\taniteval")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import wm_mpc  # noqa: E402
from taniteval import data, loaders  # noqa: E402
from taniteval.registry import MODELS  # noqa: E402

_SCR = (r"C:\Users\Admin\AppData\Local\Temp\claude"
        r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
        r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
CKPT = _SCR + r"\ckpt\ckpt.pt"
VALDIR = _SCR + r"\valsub"
OUTDIR = HERE

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--episodes", type=int, default=12)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
EPISODES = 2 if args.smoke else args.episodes
BATCH = args.batch

assert os.path.exists(CKPT) and os.path.getsize(CKPT) > 3.2e9, f"ckpt missing: {CKPT}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[mpc] device={device} batch={BATCH} episodes={EPISODES}", flush=True)

entry = dict([m for m in MODELS if m["key"] == "flagship-30k"][0])
entry["ckpt"] = CKPT
t0 = time.time()
L = loaders.load(entry, device)
model = L["model"]
assert L["traj_capable"] and getattr(model, "tactical_policy", None) is not None
step_readout = L["step_readout"]
speed_input = bool(entry.get("speed_input"))
print(f"[mpc] loaded ckpt step={L['step']} state_dim={L['state_dim']} "
      f"speed_input={speed_input} ({time.time()-t0:.1f}s)", flush=True)

files = data.list_val_episodes(VALDIR, EPISODES)
eps = data.load_frames(files)
print(f"[mpc] {len(files)} val episodes", flush=True)


def run_cfg(cfg, tag):
    tc = time.time()
    win = wm_mpc.collect_all(model, step_readout, eps, device, speed_input, cfg, batch=BATCH)
    res = wm_mpc.analyze_mpc(win, cfg)
    res["n_windows"] = int(win["gt"].shape[0])
    res["n_episodes"] = len(set(win["eid"]))
    dec = win["_mpc_n_decisions"]
    res["compute"] = {
        "mpc_select_total_s": round(win["_mpc_select_s"], 3),
        "n_decisions": int(dec),
        "ms_per_decision_batched": round(1e3 * win["_mpc_select_s"] / max(1, dec), 3),
        "note": f"batched over up to {BATCH} windows/decision; K={cfg['K']} H={cfg['H']} "
                f"method={cfg['method']}; batch=1 real-time figure in timing_batch1",
    }
    res["wall_s"] = round(time.time() - tc, 1)
    cb = res["paired_delta_C_minus_B_ade@2s"]
    print(f"\n[{tag}] n={res['n_windows']}w/{res['n_episodes']}ep "
          f"| A={res['A_open_plan_bike_ade@2s']['mean']:.3f} "
          f"B={res['B_closed_bike_ade@2s']['mean']:.3f} "
          f"C={res['C_mpc_bike_ade@2s']['mean']:.3f}", flush=True)
    print(f"[{tag}] paired C-B ADE@2s = {cb['delta']:.3f} [{cb['lo']:.3f},{cb['hi']:.3f}] "
          f"sep={cb['separated']} p(C>B)={cb['p_delta_gt0']} "
          f"| {res['compute']['ms_per_decision_batched']:.2f} ms/dec(batched)", flush=True)
    print(f"[{tag}] reproduced B-A = {res['reproduced_A_vs_B']['paired_delta_B_minus_A_ade@2s']}"
          f" (proof -0.213) | {res['verdict'].split(':')[0]}", flush=True)
    return res


def timing_batch1(cfg, n_ticks=40):
    """Honest single-ego deploy-tick timing (batch=1): time n_ticks MPPI decisions."""
    ep0 = eps[0]
    w = wm_mpc.cl.WINDOW
    fw = torch.as_tensor(ep0.feats[0:w]).unsqueeze(0).to(device)
    if fw.dtype == torch.uint8:
        fw = fw.float().div_(255.0)
    elif fw.dtype == torch.float16:
        fw = fw.float()
    aw = ep0.actions[0:w].unsqueeze(0).to(device)
    v0 = ep0.poses[w - 1, 3].to(device).float().reshape(1)
    if speed_input:
        v0c = (v0 / wm_mpc.cl.SPEED_SCALE)[:, None, None]
        aw = torch.cat([aw, v0c.expand(-1, aw.shape[1], -1)], dim=-1)
    win_s = model.encode_window(fw)
    nav = torch.zeros(1, dtype=torch.long, device=device)
    ctx = model.strategic_policy(win_s, nav)["ctx"]
    wp = model.tactical_policy(win_s, ctx)["waypoints"]
    anchor = wm_mpc.cl.densify_plan(wp, cfg["H"])
    for _ in range(3):  # warmup
        wm_mpc.mppi_select(model, step_readout, win_s, aw, v0, anchor, speed_input, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_ticks):
        wm_mpc.mppi_select(model, step_readout, win_s, aw, v0, anchor, speed_input, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    return round(1e3 * (time.perf_counter() - t) / n_ticks, 3)


HEAD = dict(wm_mpc.MPC_DEFAULT)
out = {
    "experiment": "free inference-time floor rung 3 — WM-MPPI/CEM vs single-step re-plan",
    "date": "2026-07-23",
    "evidence_class": "MEASURED",
    "model": {"key": "flagship-30k", "alias": "flagship4b-speedjerk-30k (deployed v1)",
              "ckpt_step": L["step"], "hf": "Sayood/tanitad-flagship-4b-speedjerk"},
    "harness": "taniteval/taniteval/closedloop.py (task #21) + wm_mpc.py (rung 3)",
    "baseline_ref": {"proof": "incoming/2026-07-22-imagination-closedloop-proof",
                     "B_minus_A_ade@2s": -0.213, "B_ci": [-0.3413, -0.0527]},
    "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py)",
    "run_env": {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "batch": BATCH, "episodes": EPISODES,
                "val_subset": "physicalai-val-0c5f7dac3b11 ep_00000..ep_00011 (held-out)"},
    "headline_config": HEAD,
}

print("\n================= RUNG 3 — WM-MPPI HEADLINE =================", flush=True)
out["headline"] = run_cfg(HEAD, "headline mppi K8 H8 grnd")

if not args.smoke:
    grid = [
        ({**HEAD, "K": 16}, "mppi K16"),
        ({**HEAD, "K": 4}, "mppi K4"),
        ({**HEAD, "method": "cem"}, "cem K8"),
        ({**HEAD, "cost_path": "bike"}, "mppi K8 bike-cost"),
        ({**HEAD, "H": 12}, "mppi H12"),
    ]
    out["secondary_grid"] = []
    for cfg, tag in grid:
        out["secondary_grid"].append({"tag": tag, "result": run_cfg(cfg, tag)})
    print("\n[mpc] batch=1 real-time timing (K sweep, single ego)...", flush=True)
    out["timing_batch1"] = {
        "note": "single-ego deploy-tick: one MPPI decision, batch=1, 40-tick mean, "
                "RTX 4060; registry reference CEM ~20.8 ms @ K=8",
        "ms_per_decision": {
            "mppi_K8_H8": timing_batch1(HEAD),
            "mppi_K16_H8": timing_batch1({**HEAD, "K": 16}),
            "mppi_K32_H8": timing_batch1({**HEAD, "K": 32}),
            "cem_K8_H8": timing_batch1({**HEAD, "method": "cem"}),
        },
    }
    for kk, vv in out["timing_batch1"]["ms_per_decision"].items():
        print(f"[mpc]   {kk}: {vv:.2f} ms/decision (batch=1)", flush=True)

out["wall_s"] = round(time.time() - t0, 1)
outp = Path(OUTDIR) / ("wm_mpc_result_smoke.json" if args.smoke else "wm_mpc_result.json")
outp.write_text(json.dumps(out, indent=2, default=str))
print(f"\n[mpc] wrote {outp} (wall {out['wall_s']}s)", flush=True)
print("MPC_DONE", flush=True)
