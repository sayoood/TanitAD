#!/usr/bin/env python3
"""Pull the obstacle.offline chunks the canonical val40 episodes live in.

Uses the repo's own committed path (tanitad.keys.enable_tls/load_keys +
huggingface_hub), the same one `physicalai._calib_chunk_path` uses on demand.
Read-only w.r.t. the corpus: writes ONLY into the local data root's
labels/obstacle.offline/, never into _epcache and never near r0_selection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
MAP = Path(__file__).with_name("val40_map.json")


def main():
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    from huggingface_hub import hf_hub_download

    rows = json.loads(MAP.read_text())
    want = sorted({int(r["chunk"]) for r in rows})
    have = {int(p.name.split("_")[-1].split(".")[0])
            for p in (ROOT / "labels" / "obstacle.offline").glob("*.zip")}
    todo = [c for c in want if c not in have]
    print(f"[pull] {len(want)} val40 chunks, {len(todo)} missing", flush=True)
    ok, fail, mb = [], [], 0.0
    for i, c in enumerate(todo):
        rel = f"labels/obstacle.offline/obstacle.offline.chunk_{c:04d}.zip"
        t0 = time.time()
        try:
            p = hf_hub_download(HF_REPO, rel, repo_type="dataset",
                                local_dir=str(ROOT))
            s = Path(p).stat().st_size / 1e6
            mb += s
            ok.append(c)
            print(f"[pull] {i+1}/{len(todo)} chunk {c:04d} {s:.1f} MB "
                  f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            fail.append((c, repr(e)[:160]))
            print(f"[pull] {i+1}/{len(todo)} chunk {c:04d} FAILED {e!r}",
                  flush=True)
    print(json.dumps({"ok": ok, "failed": fail, "total_mb": round(mb, 1)},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
