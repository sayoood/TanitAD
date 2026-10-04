"""Write the S2 window list ({clip_id: [ws, ...]}) the battery rolls on -- the SAME function run_battery_r7
uses (`r7_roll.s2_window_list` over the banked refcv4b grid, restricted to the kit's 139 clips).

    python make_s2_windows.py D:/refcv7_eval_kit/windows/s2_windows.json

⛔ The file carries RAW clip ids (the join key): it lives in the dev-box kit and is never landed. Its
sha12-keyed twin (`<out>.sha12.json`) is safe to bank.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tanitad.eval import refcv7_loader as L  # noqa: E402

L.bootstrap()
import r7_roll as RR  # noqa: E402
import r7_panel as P  # noqa: E402
from tanitad.data.v2_dataset import load_or_build_manifest  # noqa: E402

out = sys.argv[1]
kit_eval = str(L.KIT / "data/refcv6-b1-416x1024-eval139")
ids = {str(c) for c in load_or_build_manifest(kit_eval, verbose=False)["clip_id"]}
w = RR.s2_window_list(P.BASELINES["b_refcv4b"], ids)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(w, open(out, "w", encoding="utf-8"))
tw = {L.sha12(c): v for c, v in w.items()}
json.dump(tw, open(out + ".sha12.json", "w", encoding="utf-8"))
print(json.dumps({"n_clips": len(w), "n_windows": sum(len(v) for v in w.values()),
                  "sha256_of_sha12_view": hashlib.sha256(json.dumps(tw, sort_keys=True).encode()).hexdigest()}))
