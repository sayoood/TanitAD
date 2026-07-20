"""REF-A dyn-in overfitting curve driver (Task 2).

Non-destructive: does NOT edit registry.py. Clones the canonical refa-dynin-30k
registry entry per checkpoint (guaranteeing field parity: frozen DINOv2-B/14,
d_dino 768, temporal adapter, speed+dyn input, action_dim 4, four_brain), swaps
only the ckpt path/key, and runs the SAME rollout+bench protocol as the gate.

Loads the frozen-encoder val features ONCE (model-independent) and reuses them
across all checkpoints. Features are already disk-cached, so this is free.

Writes results/overfit_<key>.json + windows_overfit_<key>.pt per ckpt.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

import torch  # noqa: E402
from taniteval import bench, data, loaders, rollout  # noqa: E402
from taniteval.registry import MODELS  # noqa: E402

RES = Path("/root/taniteval/results")
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"

# ckpt-dir -> curve label; snap is byte-identical to 5k so we skip it.
CKPTS = [
    ("refa-dynin-5k", "/root/models/refa-dynin-5k/ckpt.pt"),
    ("refa-dynin-15k", "/root/models/refa-dynin-15k/ckpt.pt"),
    ("refa-dynin-20k", "/root/models/refa-dynin-20k/ckpt.pt"),
    ("refa-dynin-30k", "/root/models/refa-dynin-30k/ckpt.pt"),
]

BASE = next(m for m in MODELS if m["key"] == "refa-dynin-30k")


def main():
    device = "cuda"
    # Load frozen DINOv2 features ONCE (model-independent, disk-cached).
    files = data.list_val_episodes(VAL, 40)
    print(f"[driver] loading features for {len(files)} val episodes...",
          flush=True)
    eps = data.load_features(files, BASE["feat_kind"], device)
    print(f"[driver] features ready ({len(eps)} eps)", flush=True)

    curve = []
    for key, ckpt in CKPTS:
        t0 = time.time()
        entry = copy.deepcopy(BASE)
        entry["key"] = key
        entry["ckpt"] = ckpt
        L = loaders.load(entry, device)
        win = rollout.collect(
            L["model"], L["step_readout"], eps, device,
            speed_input=bool(entry.get("speed_input")),
            yaw_input=bool(entry.get("yaw_input")),
            dyn_input=bool(entry.get("dyn_input")))
        res = bench.run(win)
        res["model"] = {k: entry.get(k) for k in
                        ("key", "name", "arch", "encoder", "speed_input", "hf")}
        res["ckpt_step"] = L["step"]
        res["wall_s"] = round(time.time() - t0, 1)
        rollout.save_windows(win, RES / f"windows_overfit_{key}.pt")
        (RES / f"overfit_{key}.json").write_text(
            json.dumps(res, indent=2, default=str))

        hm = res["heldout"]["model"]
        fm = res["full_set"]["model"]
        cvm = res["heldout"]["cv"]
        row = dict(
            key=key, step=L["step"], n=res["n_windows"],
            ade2s_heldout=round(hm["ade_0_2s"]["mean"], 4),
            ade2s_ci95=round(hm["ade_0_2s"]["ci95"], 4),
            ade2s_full=round(fm["ade_0_2s"], 4),
            fde2s_heldout=round(hm["fde@2s"]["mean"], 4),
            miss2m=round(hm["miss_rate@2m"]["mean"], 4),
            cv_ade2s=round(cvm["ade_0_2s"]["mean"], 4),
            by_speed={s: round(res["by_speed"][s]["model_ade@2s"], 3)
                      for s in res["by_speed"]},
            by_curv={c: round(res["by_curvature"][c]["model_ade@2s"], 3)
                     for c in res["by_curvature"]},
            wall_s=res["wall_s"])
        curve.append(row)
        print(f"[curve] {key} step={L['step']} n={res['n_windows']} "
              f"ade@2s(heldout)={row['ade2s_heldout']}±{row['ade2s_ci95']} "
              f"full={row['ade2s_full']} fde@2s={row['fde2s_heldout']} "
              f"miss@2m={row['miss2m']} ({row['wall_s']}s)", flush=True)

    (RES / "overfit_curve.json").write_text(json.dumps(curve, indent=2))
    print("\n===== OVERFITTING CURVE (ADE@2s heldout) =====", flush=True)
    best = min(curve, key=lambda r: r["ade2s_heldout"])
    for r in curve:
        mark = "  <== BEST" if r["key"] == best["key"] else ""
        print(f"  {r['key']:18s} step={r['step']:>6} "
              f"ade@2s={r['ade2s_heldout']:.3f}±{r['ade2s_ci95']:.3f} "
              f"full={r['ade2s_full']:.3f} fde={r['fde2s_heldout']:.3f} "
              f"miss={r['miss2m']:.3f}{mark}", flush=True)
    print(f"\n[verdict] best refa-dynin ckpt = {best['key']} "
          f"(step {best['step']}) ade@2s {best['ade2s_heldout']:.3f}", flush=True)


if __name__ == "__main__":
    main()
