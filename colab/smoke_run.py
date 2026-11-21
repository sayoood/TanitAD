"""CPU smoke driver — executes a SHIPPED notebook's code cells, no Jupyter.

    python colab/smoke_run.py colab/SAM3_BACKFILL_115.ipynb
    python colab/smoke_run.py colab/STRATEGIC_LABEL_LAB.ipynb

Sets S2_SMOKE=1 (unless already set) and execs the .ipynb's code cells in
order in ONE namespace — i.e. it runs the notebook itself, not a copy of its
logic, so what is smoke-tested is byte-for-byte what Colab will run. The
notebooks are magic-free (plain Python cells) precisely to keep this true.

What is REAL in smoke: auth, HF pulls, the gap/records derivation, the
ph1_fuse fusion, the VLM model-id resolution, banking to
`Sayood/tanitad-s2-lab/smoke/` with far-side byte verification, and the
resume listing. What is STUBBED: the GPU models (VLM inference, SAM3) and
the video frames. A notebook that has never executed is a hypothesis — this
driver is what turns these two into measurements.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    nb_path = Path(sys.argv[1]).resolve()
    root = nb_path.parent.parent
    os.environ.setdefault("S2_SMOKE", "1")
    os.environ.setdefault("S2_REPO_ROOT", str(root))
    os.chdir(root)
    for p in (root / "colab", root / "stack", root / "stack" / "scripts"):
        sys.path.insert(0, str(p))

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"=== smoke_run {nb_path.name}: {len(cells)} code cells, "
          f"S2_SMOKE={os.environ['S2_SMOKE']} ===")
    ns: dict = {"__name__": "__main__"}
    t00 = time.time()
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        head = next((ln for ln in src.splitlines() if ln.strip()), "")[:70]
        print(f"\n--- cell {i + 1}/{len(cells)}: {head}")
        t0 = time.time()
        try:
            exec(compile(src, f"<{nb_path.name} cell {i + 1}>", "exec"), ns)
        except Exception:
            print(f"!!! CELL {i + 1} FAILED after {time.time() - t0:.1f}s")
            raise
        print(f"    (cell {i + 1} ok, {time.time() - t0:.1f}s)")
    print(f"\n=== SMOKE PASS {nb_path.name} in {time.time() - t00:.0f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
