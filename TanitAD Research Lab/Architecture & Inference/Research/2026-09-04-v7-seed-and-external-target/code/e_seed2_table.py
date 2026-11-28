"""Render the E-SEED-2 panel + contrasts as markdown tables (no re-computation)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BANK = Path(__file__).resolve().parent / "eseed2"


def table(path: Path, title: str) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    arms = list(d["arms"])
    tgts = list(next(iter(d["arms"].values()))["global"])
    print(f"\n## {title}   ({path.name})")
    m = d["meta"]
    print(f"n={next(iter(d['arms'].values()))['global'][tgts[0]]['n']} "
          f"clips={next(iter(d['arms'].values()))['global'][tgts[0]]['n_clusters']} "
          f"pca_k={m['pca_k']} k_outer={m['k_outer']} boot={m['n_boot']}")
    for space in ("global", "spatial4x4"):
        print(f"\n### {space}")
        print("| arm | " + " | ".join(tgts) + " |")
        print("|---|" + "---|" * len(tgts))
        for a in arms:
            cells = []
            for t in tgts:
                c = d["arms"][a][space][t]
                edge = " ⚠EDGE" if c["lambda_at_grid_edge"] else ""
                cells.append(f"{c['r2']:+.4f} [{c['ci95'][0]:+.3f}, "
                             f"{c['ci95'][1]:+.3f}]{edge}")
            print(f"| `{a}` | " + " | ".join(cells) + " |")
        print("| **constant-only** | " + " | ".join(["**0.0000** (exact)"] * len(tgts)) + " |")
    if "contrasts" in d:
        print("\n### paired contrasts (clip-cluster bootstrap)")
        print("| space::target | contrast | question | delta | CI95 | separated |")
        print("|---|---|---|---|---|---|")
        for k, v in d["contrasts"].items():
            sp, tg, pair = k.split("::")
            print(f"| {sp}::{tg} | `{pair}` | {v['question']} | "
                  f"{v['delta']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                  f"{'**YES**' if v['separated'] else 'no'} |")


if __name__ == "__main__":
    for nm, t in (("eseed2_panel.json", "E-SEED-2 pass 1 (raw all-cuboid n_agents)"),
                  ("eseed2b_panel.json", "E-SEED-2b (register's PSG n_agents)")):
        p = BANK / nm
        if p.exists():
            table(p, t)
        else:
            print(f"\n(missing {nm})", file=sys.stderr)
