"""Build a corpus root the epcache builder can actually resolve.

Fixes the second preflight blocker: `_physicalai_root_of` recovers the root by
walking a clip path's parents for a dir named **`r0`** (physicalai.py:166). The
B1 camera bank lands at `<root>/camera/camera_front_wide_120fov/`, which has no
`r0` ancestor -> returns None -> intrinsics AND extrinsics both fall back
SILENTLY. And `_chunk_of_clip` reads `<root>/r0/r0_selection.parquet`, whose
local copy holds 500 rows covering 38 of the 4,719 B1 clips.

⛔ PARITY SAFETY: this writes a NEW root and never touches the canonical corpus
or its `r0_selection.parquet`. `physicalai-train-e438721ae894` / skip-hash
`f09e44db` are untouched; this corpus is separately identified and NON-PARITY.

The camera bank is attached by a DIRECTORY JUNCTION, not a copy: 47 GB stays
where the pull wrote it (and the pull may still be running).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path("C:/Users/Admin/tanitad-data/physicalai")
B1 = Path("C:/Users/Admin/tanitad-data/physicalai-b1")
CAM_SRC = SRC / "camera" / "camera_front_wide_120fov"
IDX = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus"
           "/index/clip_to_chunk.parquet")
LABELS = "C:/Users/Admin/tanitad-wt/_s2build/v7_sup/s2_labels_v7.jsonl"

(B1 / "r0").mkdir(parents=True, exist_ok=True)
(B1 / "calibration").mkdir(parents=True, exist_ok=True)


def junction(link: Path, target: Path):
    if link.exists():
        print(f"  junction exists: {link.name}")
        return
    # NOT text=True: mklink answers in the console codepage (localised), which
    # blows up cp1252 decoding inside subprocess's reader thread.
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       capture_output=True)
    if r.returncode or not link.exists():
        msg = (r.stdout + r.stderr).decode("utf-8", "replace")
        raise SystemExit(f"mklink failed for {link}: {msg}")
    print(f"  junction {link.name} -> {target}")


# --- 1. the r0 selection the chunk lookup needs ------------------------------
need = {json.loads(x)["clip_id"] for x in open(LABELS, encoding="utf-8") if x.strip()}
cix = pd.read_parquet(IDX).reset_index()
sel = cix[cix.clip_id.astype(str).isin(need)][["clip_id", "chunk"]].copy()
sel["clip_id"] = sel.clip_id.astype(str)
sel["chunk"] = sel.chunk.astype(int)
assert len(sel) == len(need) == 4719, (len(sel), len(need))
out = B1 / "r0" / "r0_selection.parquet"
sel.to_parquet(out, index=False)
print(f"wrote {out} — {len(sel)} clips, {sel.chunk.nunique()} chunks")

# --- 2. camera bank under an `r0` parent (junction, not a copy) --------------
junction(B1 / "r0" / "camera_front_wide_120fov", CAM_SRC)

# --- 3. calibration: the CSV table + both chunk dirs -------------------------
for kind in ("camera_intrinsics", "sensor_extrinsics"):
    junction(B1 / "calibration" / kind, SRC / "calibration" / kind)
csv = SRC / "calibration" / "physicalai_front_wide_intrinsics.csv"
dst = B1 / "calibration" / csv.name
if csv.exists() and not dst.exists():
    dst.write_bytes(csv.read_bytes())
    print(f"  copied {csv.name} ({dst.stat().st_size/1e6:.2f} MB)")

# --- 4. prove the fix by RESOLVING, not by asserting the layout looks right --
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data.physicalai import _physicalai_root_of  # noqa: E402

mp4 = next((B1 / "r0" / "camera_front_wide_120fov").glob("*.mp4"), None)
rec = _physicalai_root_of(mp4) if mp4 else None
ok = rec is not None and Path(rec).resolve() == B1.resolve()
print(f"\nroot recovery from a clip path: {rec} -> {'OK' if ok else 'STILL BROKEN'}")
print(f"NEXT: python preflight_epcache_build.py --root {B1}")
