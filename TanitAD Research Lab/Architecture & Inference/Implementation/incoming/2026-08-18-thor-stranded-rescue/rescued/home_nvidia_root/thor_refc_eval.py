"""REF-C evaluated ON THOR with the route input EXERCISED — resolving the C6 confound.

⭐ WHY THIS RUN EXISTS. `taniteval/refc_eval.py` states in its own output dict:

    "`route_input_exercised` False means the decoder saw ONE constant command for every window,
     so it was compared on its marginal — the 07-21 C6 confound. Every REF-C number published
     before 2026-07-26 (base 0.4728, XL 0.4714) was collected that way."

Those are the numbers the programme quotes for REF-C. This re-collects them with
`nav_mode="produced"`, which runs one extra forward purely to READ the model's own **image-only**
`route_head` and feed its argmax back as `nav_cmd`. ⛔ Not the oracle mode — that reads the ego's
own future poses and is an upper bound, never a leaderboard number.

⛔ BINDING: reports the FOUR FAMILIES, not ADE alone.
⚠️ n is whatever has landed on Thor so far. The clip count is printed; a small-n run is a PLUMBING
and CONFOUND check, and the decision-grade number needs the full canonical 40.
"""
import dataclasses
import json
import os
import sys

for p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(p))

from types import SimpleNamespace

import torch

VAL = os.path.expanduser("~/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl")
OUT = os.path.expanduser("~/refc_thor_eval.json")
ARMS = [("refc-base", "base", os.path.expanduser("~/models/refc-base/ckpt.pt")),
        ("refc-xl", "xl", os.path.expanduser("~/models/refc-xl/ckpt.pt"))]

from tanitad.config import flagship4b_config
from tanitad.train.flagship_losses import horizon_plan
from train_flagship_v4 import resolve_v2_frames

from taniteval import four_families as FF
from taniteval import refc_eval

results = {"val_dir": VAL, "nav_mode": "produced",
           "why": "C6: pre-2026-07-26 REF-C numbers used constant follow (nav_cmd=None)"}


def load_episodes():
    """The v2 providers at REF-C's own geometry."""
    from tanitad.data.v2_dataset import build_v2_providers
    # ⛔ SELF-HEALING: a background relay is still delivering clips, and tar-over-ssh truncates
    # silently (exit 0). Verify by LOADING every clip and quarantine the unreadable ones, so this
    # eval is immune to in-flight corruption instead of racing a cleanup script.
    import glob, shutil
    q = os.path.join(VAL, "_quarantine"); os.makedirs(q, exist_ok=True)
    bad = 0
    for f in sorted(glob.glob(os.path.join(VAL, "*.v2ep.pt"))):
        try:
            torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            shutil.move(f, os.path.join(q, os.path.basename(f))); bad += 1
    for m in glob.glob(os.path.join(VAL, "_v2manifest*.pt")):
        if bad:
            os.remove(m)          # a manifest built over corrupt clips must not be reused
    print(f"[val] quarantined {bad} corrupt clip(s)", flush=True)

    # ⚠️ INTERFACE SHIM, not a data change. refc_eval.collect reads  (the raw-episode
    # interface); the v2 compressed providers expose the identical raster as . Alias it
    # so the SAME eval code path runs unmodified — no reformatting, no re-encoding.
    for e in eps:
        if not hasattr(e, "feats"):
            type(e).feats = property(lambda self: self.frames)
            break
    cfg = flagship4b_config()
    ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                         projection="cylindrical", v2_subframe="176x624", f_ref=None)
    cf, tf = resolve_v2_frames(ns, cfg, label="refc_thor")
    eps = build_v2_providers([VAL], lru_size=4,
                             frame=(None if tf == cf else tf), verbose=False)
    return eps, cfg


eps, cfg = load_episodes()
results["n_clips"] = len(eps)
print(f"[refc] {len(eps)} clips on Thor", flush=True)

for key, preset, ckpt in ARMS:
    if not os.path.exists(ckpt):
        results[key] = {"skipped": f"missing {ckpt}"}
        print(f"[refc] SKIP {key}: no ckpt", flush=True)
        continue
    try:
        # ⭐ Use loaders.py's OWN construction path: the preset dispatch table plus the run's
        # own config.json, so every gated graft and the anchor buffer are built at the TRAINED
        # shape and the state_dict loads STRICT. A hand-built config is a lookalike.
        from pathlib import Path as _P
        from taniteval.loaders import _apply_overrides
        from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                       refc_smoke_config, refc_xl_config)
        _presets = {"small": refc_small_config, "base": refc_config,
                    "xl": refc_xl_config, "smoke": refc_smoke_config}
        rcfg = _presets[preset]()
        cj = _P(ckpt).parent / "config.json"
        if cj.exists():
            _apply_overrides(rcfg, json.loads(cj.read_text()).get("cfg", {}))
        model = RefCModel(rcfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model = model.cuda().eval()
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[refc] {key} loaded {n_par:.1f}M  missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)

        win = refc_eval.collect(model, eps, "cuda", stride=8, batch=8,
                                speed_input=True, mode="diffusion",
                                nav_mode="produced")
        fam = FF.all_families(win)
        lo, la = fam["longitudinal"], fam["lateral"]
        results[key] = {
            "params_M": round(n_par, 2),
            "state_dict_missing": len(missing), "state_dict_unexpected": len(unexpected),
            "route_input_exercised": win.get("route_input_exercised"),
            "nav_note": win.get("nav_note"),
            "n_windows": int(torch.as_tensor(win["pred"]).shape[0]),
            "LONGITUDINAL": {k: lo[k] for k in
                             ("speed_mae_mps", "speed_bias_mps", "along_mae_m",
                              "along_bias_m", "along_final_bias_m")},
            "LATERAL": {k: la[k] for k in
                        ("heading_mae_deg", "yaw_rate_mae_degps",
                         "curvature_mae_1pm", "cross_mae_m", "cross_bias_m")},
            "TACTICAL": fam["tactical"].get("status"),
            "STRATEGIC": fam["strategic"].get("status"),
            "families_unavailable": fam["_families_unavailable"],
        }
        print(f"[{key}] exercised={results[key]['route_input_exercised']} "
              f"n={results[key]['n_windows']}", flush=True)
        print(f"[{key}] LON {results[key]['LONGITUDINAL']}", flush=True)
        print(f"[{key}] LAT {results[key]['LATERAL']}", flush=True)
    except Exception as e:
        import traceback
        results[key] = {"FAILED": f"{type(e).__name__}: {str(e)[:240]}"}
        print(f"[refc] {key} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc()

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("WROTE", OUT, flush=True)
