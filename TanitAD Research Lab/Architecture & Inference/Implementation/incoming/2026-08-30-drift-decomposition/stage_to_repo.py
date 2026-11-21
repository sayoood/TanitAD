"""Copy MM-E6 deliverables onto the G: mount, verifying by CONTENT.

⚠️ The Drive mount goes FULLY down (reads AND writes, Errno 22) for minutes at a
time and then recovers, and a partial write there looks like a success. So: write,
then re-READ and compare sha256. A copy that is not read back is not a copy.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
from pathlib import Path

DST = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
           r"\Architecture & Inference\Implementation\incoming"
           r"\2026-08-30-drift-decomposition")
SRC = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def put(src: Path, dst: Path, tries=8) -> bool:
    want = sha(src)
    for i in range(tries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if sha(dst) == want:
                print(f"  OK   {dst.name:<44} {want[:12]}  {src.stat().st_size:>9} B")
                return True
            print(f"  ⚠️  sha mismatch on {dst.name}, retry {i + 1}")
        except OSError as e:
            print(f"  ⚠️  {type(e).__name__} on {dst.name}, retry {i + 1}: "
                  f"{str(e)[:70]}")
        time.sleep(2 + 2 * i)
    print(f"  FAIL {dst.name}")
    return False


def main() -> int:
    items = [(SRC / n, DST / sub / n) for sub, n in
             [("", "RESULT.md"),
              ("", "mm_e6_drift_decompose.py"), ("", "mm_e6_dino_scene.py"),
              ("", "mm_e6_pdiag.py"), ("", "mm_e6_pctrl.py"),
              ("", "mm_e6_floor.py"),
              ("", "run_mm_e6.sh"), ("", "stage_to_repo.py")]]
    for p in sorted(SRC.glob("mm_e6_*.json")):
        items.append((p, DST / "raw" / p.name))
    for n in ("mm_e6_main.log", "mm_e6_repaired.log", "mm_e6_v2_16x40.log",
              "pdiag_full.log", "pctrl_full.log", "floor_postrain30k.log",
              "floor_grids_omp8.log", "floor_ranks.log", "dino_full.log",
              "dino_16x40.log", "mm_e6_smoke.log", "pdiag_smoke.log"):
        if (SRC / n).is_file():
            items.append((SRC / n, DST / "raw" / n))
    for p in sorted(SRC.glob("pctrl_*.log")):
        items.append((p, DST / "raw" / p.name))
    for g in ("dino_heldout_4x8", "dino_heldout_16x40"):
        if (SRC / g / "_meta.json").is_file():
            items.append((SRC / g / "_meta.json", DST / "raw" / f"{g}_meta.json"))
    ok = sum(put(s, d) for s, d in items if s.is_file())
    print(f"\n  staged {ok}/{len(items)} files -> {DST}")
    return 0 if ok == len(items) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
