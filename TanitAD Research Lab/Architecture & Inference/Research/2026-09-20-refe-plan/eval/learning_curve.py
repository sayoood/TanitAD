"""The SPEC_NAVTEST E-3/E-4 learning curve: every checkpoint on the SAME tokens, each PAIRED with the
points before it, plus the proposal diagnostics that PDMS cannot show while the scorer is untrained.

  python eval/learning_curve.py --tokens <W3 A1_sub200_tokens.json> --prefix sub200 \
      --ckpt ep001=D:/.../snap_epoch001.pt --ckpt ep002=D:/.../snap_epoch002.pt [--wait-frames]
Each point is `eval_checkpoint.py` (seam -> PDMS -> floors -> families); a point whose
`points/<prefix>_<name>.json` already says OK is not recomputed. `--wait-frames` blocks until the
frame extractor wrote its end marker, and REFUSES to run on an incomplete extraction.
Prints the curve and ZZCURVE_OK <n points> (or ZZCURVE_FAIL <name>).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "D:/Projects/TanitAD/data/refe_navtest"


def point(name: str):
    p = os.path.join(DATA, "points", f"{name}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return d if d.get("verdict") == "OK" else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="name=path, in training order")
    ap.add_argument("--wait-frames", action="store_true")
    ap.add_argument("--frames-log", default=os.path.join(DATA, "build_frames.log"))
    a = ap.parse_args()
    if a.wait_frames:
        while True:
            txt = open(a.frames_log, encoding="utf-8", errors="replace").read() \
                if os.path.exists(a.frames_log) else ""
            if "ZZNAVTEST_FRAMES_OK" in txt:
                break
            if "ZZNAVTEST_FRAMES_INCOMPLETE" in txt:
                print("ZZCURVE_FAIL frames_incomplete"); return 1
            time.sleep(30)
    names = []
    for spec in a.ckpt:
        n, _, path = spec.partition("=")
        name = f"{a.prefix}_{n}"
        if point(name) is None:
            cmd = [sys.executable, os.path.join(HERE, "eval_checkpoint.py"), "--ckpt", path,
                   "--name", name, "--tokens", a.tokens]
            if names:
                cmd += ["--prev"] + names
            print(f"  -> {name}  ({path})", flush=True)
            subprocess.call(cmd, cwd=HERE)
            if point(name) is None:
                print(f"ZZCURVE_FAIL {name}"); return 1
        names.append(name)
    print(f"\n  {'point':18s} {'PDMS':>7s} {'vs STOP':>9s} {'vs prev':>9s} {'ADE sel':>8s} "
          f"{'random':>7s} {'oracle':>7s} {'spread':>7s} {'#best':>5s}")
    prev = None
    for name in names:
        d = point(name)
        fl = d.get("floors") or {}
        pdms = d["score"]["summary_x100_4dp"]["PDMS"]
        vs_stop = (fl.get("pairs") or {}).get("REFe__minus__STOP", {}).get("delta_x100")
        vs_prev = ((fl.get("pairs") or {}).get(f"REFe__minus__REFe_{prev}", {}).get("delta_x100")
                   if prev else None)
        pr = (d.get("seam") or {}).get("proposals") or {}
        f = lambda v, w=7, p=2: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}")
        print(f"  {name:18s} {f(pdms)} {f(vs_stop, 9)} {f(vs_prev, 9)} {f(pr.get('ade_selected_m'), 8, 3)} "
              f"{f(pr.get('ade_random_m'), 7, 3)} {f(pr.get('ade_oracle_m'), 7, 3)} "
              f"{f(pr.get('endpoint_spread_m'), 7, 3)} {pr.get('oracle_index_distinct', '-'):>5}")
        prev = name
    print(f"ZZCURVE_OK {len(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
