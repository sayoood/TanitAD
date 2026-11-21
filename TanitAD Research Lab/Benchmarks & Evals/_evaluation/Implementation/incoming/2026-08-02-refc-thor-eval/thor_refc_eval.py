"""REF-C evaluated ON THOR with the route input EXERCISED — resolving the C6 confound.

⭐ WHY. `taniteval/refc_eval.py` states in its own output dict:

    "`route_input_exercised` False means the decoder saw ONE constant command for every window,
     so it was compared on its marginal — the 07-21 C6 confound. Every REF-C number published
     before 2026-07-26 (base 0.4728, XL 0.4714) was collected that way."

Those are the numbers the programme quotes. This re-collects with `nav_mode="produced"`, which
spends one extra forward purely to READ the model's own **image-only** route head and feed its
argmax back as nav_cmd. ⛔ Not `oracle` — that reads the ego's own future poses and is an upper
bound, never a leaderboard number.

⛔ BINDING: reports the FOUR FAMILIES, never ADE alone.
⚠️ n is whatever has landed on Thor. A small-n run is a PLUMBING + CONFOUND check; the
decision-grade number needs the full canonical 40 clips and a paired episode-cluster bootstrap.
"""
import dataclasses
import glob
import json
import os
import shutil
import sys

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(_p))

from types import SimpleNamespace

import torch

VAL = os.path.expanduser("~/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl")
OUT = os.path.expanduser("~/refc_thor_eval.json")
ARMS = [("refc-base", "base", os.path.expanduser("~/models/refc-base/ckpt.pt")),
        ("refc-xl", "xl", os.path.expanduser("~/models/refc-xl/ckpt.pt"))]

from tanitad.config import flagship4b_config
from train_flagship_v4 import resolve_v2_frames

from taniteval import four_families as FF
from taniteval import refc_eval

results = {"val_dir": VAL, "nav_mode": "produced",
           "why": "C6 — pre-2026-07-26 REF-C numbers used constant follow (nav_cmd=None)"}


def load_episodes():
    from tanitad.data.v2_dataset import build_v2_providers

    # ⛔ SELF-HEALING: a background relay is still delivering clips and tar-over-ssh truncates
    # SILENTLY (exit 0). Verify by LOADING each clip; quarantine the unreadable ones so this run
    # is immune to in-flight corruption rather than racing a cleanup script.
    q = os.path.join(VAL, "_quarantine")
    os.makedirs(q, exist_ok=True)
    bad = 0
    for f in sorted(glob.glob(os.path.join(VAL, "*.v2ep.pt"))):
        try:
            torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            shutil.move(f, os.path.join(q, os.path.basename(f)))
            bad += 1
    if bad:
        for m in glob.glob(os.path.join(VAL, "_v2manifest*.pt")):
            os.remove(m)          # a manifest built over corrupt clips must not be reused
    print(f"[val] quarantined {bad} corrupt clip(s)", flush=True)

    # ⛔⛔ GEOMETRY. THE 2026-08-02 RUN WAS SCORED AT THE WRONG RASTER AND ITS NUMBERS ARE
    # RETRACTED. It fed v2_subframe="176x624" (a 6x20 = 120-token map) to REF-C arms that were
    # trained and canonically evaluated at 256 px SQUARE -> grid_shape (8,8) = 64 tokens
    # (refc.py:214-218; MODEL_REGISTRY §4.1/§4.3).
    #   - XL failed LOUDLY: `shape '[8,512,8,8]' invalid for input of size 491520`
    #     (= 8 x 512 x 120), raised in imagination.py:100-102 because graft_imagination=True.
    #   - base failed SILENTLY: it has graft_imagination=False, so nothing checks the grid, and
    #     `feat_proj(fmap.flatten(2).transpose(1,2))` (refc.py:577) accepts ANY token count. It
    #     cross-attended 120 tokens it had never seen. That is a silent instrument failure, not
    #     a model result — and it is why speed_mae 3.06 m/s / yaw_rate_mae 22.6 deg/s came out
    #     implausible for an arm whose registry ADE@2s is 0.4728.
    # ⇒ Feed each arm ITS OWN trained raster, and ASSERT the token count. A wrong-geometry eval
    #   must crash, never quietly produce a number.
    cfg = flagship4b_config()
    eps = build_v2_providers([VAL], lru_size=4, frame=None, verbose=False)

    # INTERFACE SHIM, not a data change: refc_eval.collect reads the raw-episode attribute
    # 'feats'; the v2 providers expose the identical raster under 'frames'. Aliasing lets the
    # SAME eval path run unmodified — no reformat, no re-encode.
    if eps and not hasattr(eps[0], "feats"):
        type(eps[0]).feats = property(lambda self: self.frames)
        print("[val] aliased LazyV2Episode.frames -> .feats", flush=True)
    return eps, cfg


eps, cfg = load_episodes()
results["n_clips"] = len(eps)
print(f"[refc] {len(eps)} clips on Thor", flush=True)

for key, preset, ckpt in ARMS:
    if not os.path.exists(ckpt):
        results[key] = {"skipped": f"missing {ckpt}"}
        continue
    try:
        # loaders.py's OWN construction path: preset dispatch + the run's own config.json, so
        # every gated graft and the anchor buffer are built at the TRAINED shape.
        from pathlib import Path as _P

        from taniteval.loaders import _apply_overrides
        from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                       refc_smoke_config, refc_xl_config)
        presets = {"small": refc_small_config, "base": refc_config,
                   "xl": refc_xl_config, "smoke": refc_smoke_config}
        rcfg = presets[preset]()
        cj = _P(ckpt).parent / "config.json"
        if cj.exists():
            _apply_overrides(rcfg, json.loads(cj.read_text()).get("cfg", {}))
        model = RefCModel(rcfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
        model = model.cuda().eval()
        n_par = sum(p.numel() for p in model.parameters()) / 1e6

        # ⛔ GEOMETRY GATE — the check whose ABSENCE produced the retracted numbers.
        # REF-C-base has graft_imagination=False, so NOTHING inside the model validates the
        # token count: feat_proj accepts any n and the arm scores silently on a raster it never
        # saw. Assert the fed raster against the arm's own declared grid_shape BEFORE scoring.
        gh, gw = rcfg.encoder.grid_shape
        exp_hw = (gh * 32, gw * 32)                      # refc.py:214-218 patch stride
        got_hw = tuple(eps[0].frames.shape[-2:])
        if got_hw != exp_hw:
            raise RuntimeError(
                f"GEOMETRY MISMATCH for {key}: val raster is {got_hw} but the arm declares "
                f"grid_shape {(gh, gw)} => it was trained at {exp_hw}. Scoring here is a SILENT "
                f"instrument failure (base accepts any token count). Re-cache val at {exp_hw} "
                f"or evaluate this arm on its canonical cache.")
        print(f"[{key}] geometry OK: raster {got_hw} matches grid {(gh, gw)}", flush=True)

        win = refc_eval.collect(model, eps, "cuda", stride=8, batch=8,
                                speed_input=True, mode="diffusion", nav_mode="produced")
        fam = FF.all_families(win)
        lo, la = fam["longitudinal"], fam["lateral"]
        results[key] = {
            "params_M": round(n_par, 2),
            "sd_missing": len(missing), "sd_unexpected": len(unexpected),
            # ⛔ READ A REQUIRED STAMP WITH [], NEVER .get(). The provenance is NESTED under
            # `nav_provenance` (refc_eval.py:177-190). The previous `win.get("route_input_exercised")`
            # looked at TOP level, where the key has never existed, so dict.get() silently returned
            # None — and a None from a BOOLEAN-valued field was misread as "the collector did not
            # report provenance". The collector always reported it; the CALLER read the wrong path.
            # A KeyError here is the correct behaviour: a missing required stamp must fail loudly.
            "nav_provenance": {k: win["nav_provenance"][k] for k in
                               ("nav_mode", "note", "fed_command_hist",
                                "route_input_exercised", "is_oracle")},
            "route_input_exercised": win["nav_provenance"]["route_input_exercised"],
            "nav_note": win["nav_provenance"]["note"],
            # ⚠️ `route_input_exercised` is (nav_mode != follow_constant) AND len(hist) > 1 — it
            # conflates EXERCISED with VARIED. A route head that collapses to one class reports
            # False even though the input genuinely was produced. Read the histogram, not the flag.
            "fed_command_hist": win["nav_provenance"]["fed_command_hist"],
            "n_windows": int(torch.as_tensor(win["pred"]).shape[0]),
            "LONGITUDINAL": {k: lo[k] for k in
                             ("speed_mae_mps", "speed_bias_mps", "along_mae_m",
                              "along_bias_m", "along_final_bias_m")},
            "LATERAL": {k: la[k] for k in
                        ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm",
                         "cross_mae_m", "cross_bias_m")},
            "TACTICAL": fam["tactical"].get("status"),
            "STRATEGIC": fam["strategic"].get("status"),
            "families_unavailable": fam["_families_unavailable"],
        }
        r = results[key]
        print(f"[{key}] {n_par:.1f}M exercised={r['route_input_exercised']} "
              f"n={r['n_windows']} | {r['nav_note']}", flush=True)
        print(f"[{key}] LON {r['LONGITUDINAL']}", flush=True)
        print(f"[{key}] LAT {r['LATERAL']}", flush=True)
    except Exception as e:
        results[key] = {"FAILED": f"{type(e).__name__}: {str(e)[:240]}"}
        print(f"[refc] {key} FAILED: {type(e).__name__}: {str(e)[:220]}", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("WROTE", OUT, flush=True)
