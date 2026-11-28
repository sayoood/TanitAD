"""Acquire a small Cosmos-Drive-Dreams sample -- ANNOTATION TARS ONLY (no 43 GB video shards).

The custom metric suite is pixel-free, so the robustness first pass needs only the small per-clip
RDS-HQ annotation tars: vehicle_pose (~0.6 MB), all_object_info (1-31 MB), pinhole_intrinsic (~20 KB).
This pulls a spread of clips across object-richness (bigger all_object_info tar = more agents = more
occlusion content) so ``cosmos_telemetry.py`` has interesting geometry to score.

HF access on the dev box: repo is UNGATED (anonymous works), but we route TLS through the OS trust
store (intercepting proxy) via ``tanitad.keys.enable_tls`` and read the token from the git-ignored
Keys.txt if present. Uses the HF tree API (not list_repo_files, which hangs on this 5,843-clip repo).

Usage:
  python acquire_cosmos_sample.py --out C:/Users/Admin/tanitad-data/cosmos_bench3 --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_stack_on_path() -> None:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "stack" / "tanitad" / "keys.py").is_file():
            sys.path.insert(0, str(p / "stack"))
            return


COSMOS = "nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    _ensure_stack_on_path()
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    import httpx
    from huggingface_hub import hf_hub_download

    tok = os.environ.get("HF_TOKEN")
    H = {"Authorization": f"Bearer {tok}"} if tok else {}

    def tree(path=""):
        url = f"https://huggingface.co/api/datasets/{COSMOS}/tree/main"
        if path:
            url += "/" + path
        r = httpx.get(url, headers=H, params={"recursive": "false"}, timeout=60)
        r.raise_for_status()
        return r.json()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = [it for it in tree("all_object_info") if it["type"] == "file"]
    items.sort(key=lambda it: it.get("size", 0))
    n = len(items)
    # spread across the upper half of the object-richness distribution
    fracs = [0.55 + 0.44 * i / max(1, args.n - 1) for i in range(args.n)]
    picks = [items[min(n - 1, int(n * f))] for f in fracs]

    manifest = []
    for it in picks:
        clip = it["path"].split("/")[-1][:-4]
        rec = {"clip": clip, "obj_tar_bytes": it["size"]}
        try:
            for sub in ("vehicle_pose", "all_object_info", "pinhole_intrinsic"):
                hf_hub_download(COSMOS, f"{sub}/{clip}.tar", repo_type="dataset",
                                local_dir=str(out))
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["err"] = f"{type(e).__name__}: {e}"
        manifest.append(rec)
        print(f"  {'OK ' if rec['ok'] else 'ERR'} {clip[:40]}  obj={it['size']/1e6:.1f}MB")

    ok = sum(1 for r in manifest if r.get("ok"))
    json.dump(manifest, open(out / "acquire_manifest.json", "w"), indent=1)
    print(f"[acq] {ok}/{len(picks)} clips (annotation tars only) -> {out}")


if __name__ == "__main__":
    main()
