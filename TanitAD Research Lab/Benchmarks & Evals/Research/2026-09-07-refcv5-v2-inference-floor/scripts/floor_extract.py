"""Tabulate the seven arms' interval-backed metrics for one inference seed.

⛔ THE CHECKPOINT IS ckpt_15000 OF 40,284 -- MID-TRAINING. No LEVEL printed here
is a capability number and none may be quoted as one. The deliverable is the
SPREAD ACROSS SEEDS, which is a property of the DDIM sampler, not of skill.

⭐ THE KNOWN-VALUE CONTROL IS FREE AND IT IS THE POINT: refcv3_arm.py's own
source says the model-free arms `ha` / `ha0` / `ha0_ext` READ NO FRAMES. They
are therefore deterministic BY CONSTRUCTION and MUST be identical across
inference seeds. If they move, the EXPERIMENT is broken -- not the arm.
"""
import json
import sys

METRICS = ["ade_dense_m", "fde_last_m", "LON_speed_mae_mps",
           "LON_along_mae_m", "LAT_cross_mae_m", "LAT_heading_mae_deg"]
MODEL_FREE = {"ha", "ha0", "ha0_ext"}

out = {}
for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    tag = path.rsplit("/", 1)[-1].replace(".json", "")
    row = {}
    for arm, v in d["arms"].items():
        iv = (v or {}).get("intervals") or {}
        mm = iv.get("metrics") or {}
        row[arm] = {k: (mm[k]["mean"] if k in mm else None) for k in METRICS}
        row[arm]["_n"] = iv.get("n")
    out[tag] = row
    print(f"\n=== {tag}   n_windows={d.get('n_windows')} n_episodes={d.get('n_episodes')} ===")
    print(f"{'arm':12s} " + " ".join(f"{k[:15]:>15s}" for k in METRICS))
    for arm in ["os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero", "oracle_sel"]:
        if arm not in row:
            continue
        flag = " (model-free: MUST NOT move across seeds)" if arm in MODEL_FREE else ""
        vals = " ".join(f"{row[arm][k]:>15}" if row[arm][k] is not None else f"{'-':>15}"
                        for k in METRICS)
        print(f"{arm:12s} {vals}{flag}")

json.dump(out, open("out/floor_table.json", "w", encoding="utf-8"), indent=1)
print("\nwrote out/floor_table.json")
