"""X3 LEAK check on the REAL artifacts (MM item 1, 2026-10-04): reproduce WP-A / D4's 3.46 % (N2) and 23.8 % (the v8
sidecar), then read the channel refcv8 actually FEEDS. Dev-box CPU, zero GPU. Writes raw/x3_leak.json.

Inputs (each md5-stamped into the output):
* the TRAIN v2ep manifest D4 measured on (md5 3c9f8bc8...), copied to D:/Projects/TanitAD-artifacts/refcv8_wpb/inputs/;
* WP-A's v9 TRAIN release (md5 f63ece41...);
* refcv7's v8 sidecar (D:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl, md5 bab41adf... = Thor's copy).

Run:  PYTHONPATH=<overlay>/stack python code/x3_leak_check.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAN = os.environ.get("R8_TRAIN_V2MANIFEST", "D:/Projects/TanitAD-artifacts/refcv8_wpb/inputs/train_v2manifest.pt")
V9 = os.environ.get("R8_V9_TRAIN", "D:/Projects/TanitAD-artifacts/v9labels/v9_labels_train.npz")
V8 = os.environ.get("R8_V8_SIDECAR", "D:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import tanitad
    if str(Path(tanitad.__file__).resolve()).upper().startswith("G:"):
        raise SystemExit("tanitad imported from G:")
    from tanitad.eval import speed_leak as SL
    t0 = time.time()
    out = SL.x3_leak_table(MAN, V9, V8)
    out["inputs"] = {"manifest_md5": md5(MAN), "v9_train_md5": md5(V9), "v8_sidecar_md5": md5(V8),
                     "tanitad": str(Path(tanitad.__file__).resolve().parent)}
    out["wall_s"] = round(time.time() - t0, 1)
    p = HERE.parent / "raw" / "x3_leak.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["rows"].items():
        print(f"{k:42s} {v}")
    print({k: v for k, v in out.items() if k not in ("rows",)})
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
