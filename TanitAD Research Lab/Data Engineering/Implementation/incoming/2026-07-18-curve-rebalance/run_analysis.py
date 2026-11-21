"""Run the curve-rebalance analysis on the local epcache and print + save JSON.

    python run_analysis.py            # uses the default local epcache roots

Not a test (needs real bytes); the pure math is covered by tests/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curve_rebalance as cr

BASE = "C:/Users/Admin/tanitad-data"
SOURCES = {
    "comma2k19": [
        f"{BASE}/comma2k19/extracted/_epcache/comma2k19-train-5b8f2f7bbfc3",
        f"{BASE}/comma2k19/extracted/_epcache/comma2k19-train-63bbf03d23e7",
        f"{BASE}/comma2k19/extracted/_epcache/comma2k19-val-7d2568fc5e29",
        f"{BASE}/comma2k19/extracted/_epcache/comma2k19-val-bd37bf6709fd",
        f"{BASE}/eval/comma2k19-val-61c46fca8f7f",
    ],
    "physicalai": [
        f"{BASE}/physicalai/_epcache/physicalai-train-14231cd29c74",
        f"{BASE}/physicalai/_epcache/physicalai-val-bb543bdf7836",
    ],
}

if __name__ == "__main__":
    rep = cr.analyze(SOURCES, target_straight=0.575)
    # also report the 50/50 and current-mix combined straight-fractions
    dists = []
    for name, roots in SOURCES.items():
        d = cr.SourceDist(name=name)
        for r in roots:
            for poses in cr.iter_epcache_poses(r):
                d.add(poses)
        dists.append(d)
    rep["combined_5050_source"] = {
        k: round(v, 4)
        for k, v in cr.combine_sources(dists, {"comma2k19": 0.5, "physicalai": 0.5}).items()
    }
    print(json.dumps(rep, indent=2))
    out = Path(__file__).resolve().parent / "curve_rebalance_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nsaved -> {out}")
