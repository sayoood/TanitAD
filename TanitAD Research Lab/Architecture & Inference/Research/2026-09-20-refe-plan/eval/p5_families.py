#!/usr/bin/env python3
"""P5 four families (eval/PREREG_P5_TRACKER_AWARE_TARGETS.md §3), DESCRIPTIVE ONLY: no verdict reads them.

Reads p5_target_labels.py's --npz and runs taniteval's NAVSIM adapter (families6 pattern, tier T1, cluster unit = log)
on the EXECUTED 8 poses against the INTENDED ones:
  vs T (the teacher label the arm was built from): T, Tc20, Tc8, Tw
  vs H (the logged human future):                  T, Tc20, Tc8, H-executed (Hc's tracker floor is Hc vs H)
STRATEGIC is UNAVAILABLE in NAVSIM and PAUSED by the PI's 2026-09-19 ruling.

  C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/p5_families.py --npz <p5_exec.npz> --out <p5_families.json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import numpy as np

ADAPTER = "E:/Projects/TanitAD/taniteval/adapters/navsim.py"
D_TANITEVAL = "E:/Projects/TanitAD/taniteval"
PAIRS = [("T", "intended_T"), ("Tc20", "intended_T"), ("Tc8", "intended_T"), ("Tw", "intended_T"),
         ("T", "intended_H"), ("Tc20", "intended_H"), ("Tc8", "intended_H"), ("H", "intended_H"),
         ("Hc", "intended_H")]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if D_TANITEVAL not in sys.path:
        sys.path.append(D_TANITEVAL)
    spec = importlib.util.spec_from_file_location("navsim_adapter6", ADAPTER)
    ad = importlib.util.module_from_spec(spec)
    sys.modules["navsim_adapter6"] = ad
    spec.loader.exec_module(ad)
    z = np.load(a.npz, allow_pickle=False)
    toks, logs, v0 = [str(t) for t in z["tokens"]], [str(g) for g in z["logs"]], z["v0"].astype(np.float64)
    out = {"prereg": "eval/PREREG_P5_TRACKER_AWARE_TARGETS.md", "descriptive_only": True, "tier": "T1",
           "n": len(toks), "n_logs": len(set(logs)),
           "strategic": "UNAVAILABLE in NAVSIM; PAUSED (PI ruling 2026-09-19)", "families": {}}
    for ref, it in PAIRS:
        name = f"exec_{ref}_vs_{it.split('_')[1]}"
        try:
            w = ad.scenes_to_win(z[f"exec_{ref}"].astype(np.float64), z[it].astype(np.float64), frame="ego",
                                 origin_included=False, dt_s=0.5, scene_tokens=toks, ego_speed_mps=v0, log_names=logs)
            out["families"][name] = ad.four_families_block(w, tier="T1", n_boot=a.n_boot)
        except Exception as e:                                                   # noqa: BLE001
            out["families"][name] = {"status": "RAISED", "error": f"{type(e).__name__}: {e}"}
        print(f"  {name}: {'RAISED' if 'status' in out['families'][name] else 'ok'}", flush=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    bad = [k for k, v in out["families"].items() if isinstance(v, dict) and v.get("status") == "RAISED"]
    print("ZZP5_FAMILIES " + ("DONE" if not bad else "RAISED " + ",".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
