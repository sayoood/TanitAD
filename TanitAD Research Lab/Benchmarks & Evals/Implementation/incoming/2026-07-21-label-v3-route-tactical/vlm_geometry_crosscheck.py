"""Cross-check the kinematic v3 mint against the VLM's `road_geometry` reads.

The brief's instruction was to treat the VLM as a CHECK on a kinematic mint, not
a source, and to report disagreement rather than paper over it. This script does
exactly that over the sibling agent's production artifacts in
``TanitAD Research Hub/Data Engineering/Implementation/incoming/
  2026-07-21-vlm-production-semantic/*.jsonl``.

It reports three things:
  1. the `road_geometry` distribution over EVERY VLM record (the enum is
     straight / curve_left / curve_right / junction / roundabout / merge / fork /
     unknown — see that run's enums.json);
  2. ENUM VIOLATIONS — values the VLM emitted that are not in its own enum;
  3. the `kin_v21` route-scale heading distribution over the distinct windows
     those records cover, i.e. how often the ROUNDABOUT-scale signature
     (|net Δheading| >= 135°) occurs at all on that corpus.

⚠️ THREE DIFFERENT VAL BUILDS are in play and none is a subset of another
(see that run's val_build_episode_map.json): the canonical 40-ep
`physicalai-val-0c5f7dac3b11` (eval pod), the VLM run's 80-ep
`physicalai-val-f1b378f295ae` (pod3), and the dev box's 100-ep
`physicalai-val-bb543bdf7836`. A per-window join between the VLM read and the
kinematic mint is therefore NOT possible offline; only rates are comparable.

  python vlm_geometry_crosscheck.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VLM = (Path(__file__).resolve().parents[4] / "Data Engineering" /
       "Implementation" / "incoming" / "2026-07-21-vlm-production-semantic")
ENUM = ("straight", "curve_left", "curve_right", "junction", "roundabout",
        "merge", "fork", "unknown")


def _walk(node, key, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, str):
                out.append(v)
            else:
                _walk(v, key, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, key, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    geom, per_file, kin = Counter(), {}, {}
    for f in sorted(VLM.glob("*.jsonl")):
        c = Counter()
        n = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            g = []
            _walk(r, "road_geometry", g)
            c.update(g)
            if r.get("kin_v21") is not None:
                kin[(r.get("episode"), r.get("t"))] = r["kin_v21"]
        geom.update(c)
        per_file[f.name] = {"records": n, "road_geometry": dict(c)}

    bands = Counter()
    big = []
    for (e, t), k in kin.items():
        deg = abs(k.get("net_dyaw_deg") or 0.0)
        bands[">=135" if deg >= 135 else ">=90" if deg >= 90
              else ">=45" if deg >= 45 else "<45"] += 1
        if deg >= 135:
            big.append({"episode": e, "t": t, "net_dyaw_deg": round(deg, 1),
                        "peak_kappa": k.get("peak_kappa"),
                        "arc_m": k.get("arc_m"), "reason": k.get("reason")})

    out = {
        "source_dir": str(VLM),
        "vlm_val_build": "physicalai-val-f1b378f295ae (80 eps, pod3)",
        "enum": list(ENUM),
        "road_geometry_total": dict(geom),
        "road_geometry_per_file": per_file,
        "enum_violations": {k: v for k, v in geom.items() if k not in ENUM},
        "kin_v21_windows": len(kin),
        "kin_v21_net_dyaw_bands": dict(bands),
        "roundabout_scale_windows": big,
        "note": "A per-window join with the kinematic v3 mint is impossible "
                "offline: three disjoint val builds, none a subset of another.",
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k != "road_geometry_per_file"}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
