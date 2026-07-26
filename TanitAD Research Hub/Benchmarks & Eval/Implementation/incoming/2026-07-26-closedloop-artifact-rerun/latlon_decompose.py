#!/usr/bin/env python3
"""LATERAL / LONGITUDINAL decomposition of every recomputed closed-loop path.

Runs `taniteval.lateral` on the per-window closed-loop surfaces persisted by
`rerun_closedloop.py` (`clwin_<arm>.pt`).

⚠️ SURFACE. `closedloop.collect` keeps `[:, wp_idx]` — 4 knots at dense steps
[5,10,15,20] — so this is the **sparse_4wp** surface: a knot is **0.5 s**, not
0.1 s. That is precisely the surface on which `paired_cross_track` mislabelled
the horizon by **5x** until 2026-07-26. Every paired call below therefore passes
`knot_dt=0.5` EXPLICITLY rather than relying on inference, and the emitted
`horizon_provenance` is asserted to be `"explicit"` with `horizon_s == 2.0`. A
block that comes back with 0.4 s, or with no `horizon_provenance` at all, means
the module is stale and the run aborts instead of publishing a wrong timescale.

The decomposition is run per PATH (closed_bike / closed_grnd / open_grnd /
open_bike / cv) and the decision question — *is closed-loop drift lateral or
longitudinal, and does it differ from open loop?* — is answered by the PAIRED
cross-track contrast closed_bike vs open_grnd on the SAME windows.

Run:  OMP_NUM_THREADS=8 PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 latlon_decompose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")

from taniteval import lateral as LAT                # noqa: E402

OUT = Path("/root/cl_rerun_20260726")
ARMS = ["flagship-30k", "flagship-speed", "flagship-nospeed"]
PATHS = ["closed_bike", "closed_grnd", "open_grnd", "open_bike", "cv"]
KNOT_DT = 0.5          # MEASURED: wp_steps [5,10,15,20] at 10 Hz -> 0.5 s / knot


def guard(block_or_paired, where):
    """Refuse to emit a number whose horizon label we cannot vouch for."""
    hp = block_or_paired.get("horizon_provenance")
    hs = block_or_paired.get("horizon_s")
    if "horizon_provenance" in block_or_paired or "paired" in where:
        if hp is None:
            raise SystemExit(f"[{where}] STALE lateral.py — no horizon_provenance "
                             "stamp. The 5x sparse-horizon mislabelling fix is "
                             "not present. Refusing to emit.")
        if hp != "explicit":
            raise SystemExit(f"[{where}] horizon_provenance={hp!r}, expected "
                             "'explicit' (we pass knot_dt=0.5).")
    if hs is not None and abs(float(hs) - 2.0) > 1e-6:
        raise SystemExit(f"[{where}] horizon_s={hs}, expected 2.0 s on the "
                         "4-knot sparse surface. 0.4 s means stale code.")
    return block_or_paired


def main():
    out = {"_surface": "sparse_4wp — 4 knots at dense steps [5,10,15,20], "
                       "knot spacing 0.5 s (NOT 0.1 s)",
           "_knot_dt_s": KNOT_DT,
           "_estimator": "episode_cluster_bootstrap / "
                         "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                         "B=2000, over the val EPISODES",
           "_lateral_module_fix": "2026-07-26 horizon-labelling fix verified "
                                  "present (horizon_provenance stamped, "
                                  "horizon_s = 2.0 s)",
           "arms": {}}

    for arm in ARMS:
        wp = OUT / f"clwin_{arm}.pt"
        if not wp.exists():
            print(f"[latlon] SKIP {arm}: no {wp.name}")
            continue
        win = torch.load(wp, weights_only=False)
        eid = win["eid"]
        gt = win["gt"]
        rec = {"n_windows": int(gt.shape[0]), "n_episodes": len(set(map(str, eid))),
               "by_path": {}, "paired_cross_track": {}}

        for p in PATHS:
            blk = LAT.from_sparse_windows(
                {"pred": win[p], "gt": gt, "eid": eid,
                 "speed": win.get("speed"), "wp_steps": [5, 10, 15, 20]})
            guard(blk, f"{arm}/{p}")
            rec["by_path"][p] = {
                "surface": blk.get("surface"),
                "dt_s": blk.get("dt_s"),
                "horizon_s": blk.get("horizon_s"),
                "energy_share": blk.get("energy_share"),
                "growth": blk.get("growth"),
                "along": blk.get("along"),
                "cross": blk.get("cross"),
                "tail_cross": blk.get("tail_cross"),
                "verdict": blk.get("verdict"),
            }

        # ---- the decision contrast: is CLOSED-loop drift more lateral? ----- #
        for a, b, label in (("open_grnd", "closed_bike", "closed_bike_vs_open_grnd"),
                            ("open_bike", "closed_bike", "closed_bike_vs_open_bike"),
                            ("cv", "closed_bike", "closed_bike_vs_cv")):
            for red in ("mean", "p90"):
                d = LAT.paired_cross_track(
                    win[a], win[b], gt, eid, step=4,      # knot 4 == 2.0 s
                    knot_dt=KNOT_DT, reduce=red)
                guard(d, f"{arm}/paired/{label}/{red}")
                rec["paired_cross_track"][f"{label}.{red}"] = d
        out["arms"][arm] = rec
        es = rec["by_path"]["closed_bike"]["energy_share"]
        print(f"[latlon] {arm}: closed_bike energy_share={es} "
              f"horizon_s={rec['by_path']['closed_bike']['horizon_s']}", flush=True)

    (OUT / "latlon_decomposition.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote latlon_decomposition.json")


if __name__ == "__main__":
    main()
