"""Local driver: imagination-in-the-loop (B) vs single-shot open-loop (A) closed-loop
proof on the flagship v1 world model, RTX 4060. Reuses the taniteval closed-loop
harness (collect/analyze) with the arm-(A) addition; local paths + small batch."""
import json
import sys
import time
from pathlib import Path

LOCAL = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
sys.path.insert(0, LOCAL + r"\stack")
sys.path.insert(0, LOCAL + r"\stack\scripts")
sys.path.insert(0, LOCAL + r"\taniteval")          # the taniteval package parent

import torch  # noqa: E402

from taniteval import closedloop as cl  # noqa: E402
from taniteval import data, loaders     # noqa: E402
from taniteval.registry import MODELS   # noqa: E402

CKPT = (r"C:\Users\Admin\AppData\Local\Temp\claude"
        r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
        r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\ckpt\ckpt.pt")
VALDIR = (r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\valsub")
OUTDIR = (LOCAL + r"\TanitAD Research Hub\Architecture & Inference\Implementation"
          r"\incoming\2026-07-22-imagination-closedloop-proof")
EPISODES = 12
BATCH = 8          # 4060 has 8.6 GB; keep the frame batch small

import os  # noqa: E402
# Wait (in-process, no shell sleep) for the HF download to finalize ckpt.pt.
_t = time.time()
while not (os.path.exists(CKPT) and os.path.getsize(CKPT) > 3.2e9):
    if time.time() - _t > 420:
        raise SystemExit(f"[proof] ckpt not ready after 420s: {CKPT}")
    time.sleep(5)
print(f"[proof] ckpt ready {os.path.getsize(CKPT)/1e6:.0f}MB", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[proof] device={device} batch={BATCH}", flush=True)

entry = dict([m for m in MODELS if m["key"] == "flagship-30k"][0])
entry["ckpt"] = CKPT                 # override the pod path -> local download
print(f"[proof] arm={entry['key']} ({entry['name']}) speed_input={entry.get('speed_input')}",
      flush=True)

t0 = time.time()
L = loaders.load(entry, device)
model = L["model"]
assert L["traj_capable"] and getattr(model, "tactical_policy", None) is not None, \
    "flagship must expose tactical_policy + step_readout for the imagination harness"
print(f"[proof] loaded ckpt step={L['step']} state_dim={L['state_dim']} "
      f"({time.time()-t0:.1f}s)", flush=True)

files = data.list_val_episodes(VALDIR, EPISODES)
print(f"[proof] {len(files)} val episodes from {VALDIR}", flush=True)
eps = data.load_frames(files)

tcol = time.time()
win = cl.collect(model, L["step_readout"], eps, device,
                 speed_input=bool(entry.get("speed_input")), batch=BATCH)
print(f"[proof] collected {win['gt'].shape[0]} windows / {len(set(win['eid']))} "
      f"episodes ({time.time()-tcol:.1f}s)", flush=True)

res = cl.analyze(win)
res["model"] = {k: entry.get(k) for k in ("key", "name", "arch", "encoder", "speed_input")}
res["ckpt_step"] = L["step"]
res["ckpt_source"] = "HF Sayood/tanitad-flagship-4b-speedjerk/ckpt.pt (step 29999 = v1 FINAL)"
res["run_env"] = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  "batch": BATCH, "episodes_requested": EPISODES,
                  "val_subset": "physicalai-val-0c5f7dac3b11 ep_00000..ep_00011 (held-out)",
                  "note": "CHEAP no-renderer proof; 12-ep val subset, not the full 40"}
res["wall_s"] = round(time.time() - t0, 1)

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
outp = Path(OUTDIR) / "closedloop_flagship-30k_imagination-proof.json"
outp.write_text(json.dumps(res, indent=2, default=str))

ic = res["imagination_comparison"]
pa = ic["paired_delta_B_minus_A_ade@2s"]
A, B = ic["A_open_plan_bike_ade@2s"], ic["B_closed_bike_ade@2s"]
pl = ic["plan_direct_ade@2s_no_executor"]
print("\n================= IMAGINATION-IN-THE-LOOP PROOF =================", flush=True)
print(f"n = {res['n_windows']} windows / {res['n_episodes']} episodes | ckpt step {L['step']}",
      flush=True)
print(f"(A) single-shot open-loop  open_plan_bike ADE@2s = {A['mean']:.3f} "
      f"[{A['lo']:.3f}, {A['hi']:.3f}]", flush=True)
print(f"(B) imagination-in-the-loop closed_bike    ADE@2s = {B['mean']:.3f} "
      f"[{B['lo']:.3f}, {B['hi']:.3f}]", flush=True)
print(f"     (raw single-shot plan, no executor)   ADE@2s = {pl['mean']:.3f} "
      f"[{pl['lo']:.3f}, {pl['hi']:.3f}]", flush=True)
print(f"paired delta (B - A) ADE@2s = {pa['delta']:.3f} [{pa['lo']:.3f}, {pa['hi']:.3f}] "
      f"separated={pa['separated']} p(B>A)={pa['p_delta_gt0']}", flush=True)
print(f"VERDICT: {ic['verdict']}", flush=True)
print(f"\n[proof] wrote {outp} (wall {res['wall_s']}s)", flush=True)
print("PROOF_DONE", flush=True)
