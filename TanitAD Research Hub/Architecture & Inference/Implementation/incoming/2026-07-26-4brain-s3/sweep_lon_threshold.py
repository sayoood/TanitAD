"""Longitudinal manoeuvre-threshold sweep.

SELECTION RULE, declared BEFORE the sweep is read (outcome-independent — no
model exists, so nothing can be flattered): pick the SMALLEST (A_MAN, DV_MIN)
at which the longitudinal option set is NON-DEGENERATE, i.e.
  * all 5 ordered classes populated at >= 2 % each, AND
  * t_none >= 10 % (there must be a real "nothing coming" option), AND
  * majority_rate <= 0.60 (the majority baseline must not own the problem).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

S3DIR = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-s3")
sys.path.insert(0, str(S3DIR))
import s3_labels as S3

CACHE = Path(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74")
files = sorted(CACHE.glob("ep_*.pt"))[:60]
eps = []
for f in files:
    d = torch.load(f, weights_only=False, map_location="cpu")
    eps.append((f.stem, torch.as_tensor(d["poses"], dtype=torch.float32)))
print(f"loaded {len(eps)} episodes", flush=True)

GRID = [(0.5, 1.5), (0.8, 2.5), (1.0, 3.0), (1.0, 4.0),
        (1.5, 3.0), (1.5, 4.5), (2.0, 5.0), (2.5, 6.0)]
rows = []
for a_man, dv in GRID:
    S3.A_MAN_MS2, S3.DV_MIN_MS = a_man, dv
    bands, ttms, n_adm, n_m14, m2r, m3r = [], [], 0, 0, 0, 0
    for eid, poses in eps:
        for r in S3.mine_episode(poses, eid, horizon_s=12.0):
            if not (r["m1"] and r["m4"]):
                continue
            n_m14 += 1
            if not r["m2_lon"]:
                m2r += 1
                continue
            if not r["m3_lon"]:
                m3r += 1
                continue
            n_adm += 1
            bands.append(r["band_lon"])
            if r["ttm_lon_ok"] and np.isfinite(r["ttm_lon"]):
                ttms.append(r["ttm_lon"])
    c = np.bincount(np.array(bands, dtype=int), minlength=S3.N_BANDS)
    bal = c / max(1, c.sum())
    ok = bool((bal >= 0.02).all() and bal[S3.IX_NONE] >= 0.10
              and bal.max() <= 0.60)
    rows.append({"a_man": a_man, "dv_min": dv, "n_M1M4": n_m14,
                 "n_admissible": n_adm, "m2_rej": m2r, "m3_rej": m3r,
                 "adm_frac": round(n_adm / max(1, n_m14), 4),
                 "balance": {S3.BAND_NAMES[i]: round(float(bal[i]), 4)
                             for i in range(S3.N_BANDS)},
                 "majority_rate": round(float(bal.max()), 4),
                 "median_ttm_s": round(float(np.median(ttms)), 3) if ttms else None,
                 "p90_ttm_s": round(float(np.percentile(ttms, 90)), 3) if ttms else None,
                 "PASSES_RULE": ok})
    print(json.dumps(rows[-1]), flush=True)

winners = [r for r in rows if r["PASSES_RULE"]]
print("\nSELECTED:", json.dumps(winners[0] if winners else None, indent=1))
Path(r"C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sweep_lon.json").write_text(
    json.dumps({"n_episodes": len(eps), "rule": __doc__, "grid": rows,
                "selected": winners[0] if winners else None}, indent=2))
