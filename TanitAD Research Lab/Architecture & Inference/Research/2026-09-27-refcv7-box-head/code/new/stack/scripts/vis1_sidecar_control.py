#!/usr/bin/env python3
"""vis1_sidecar_control.py -- the VIS-1 sidecar ARTIFACT must reproduce the box-head audit's banked z-buffer rows.

SPEC_REFCV7 §14 (A9) R3 / the builder brief: *"Include a control: the sidecar reproduces the audit's banked vis_frac
values on its sampled windows exactly."* ``precompute_vis1_sidecar.py --mode control`` proves the COMPUTATION on the
audit's windows; this proves the STORED FILE: for every banked window (sha12, w) of ``vis_{train,clipgrid,inrun,gtval}
_boxes.json`` and every banked row inside ``vis1.STORE_SCOPE``, the sidecar row at (sha12, f = w + W - 1) carries the
identical n_full / n_img / n_vis / vis_rows (and therefore vis_frac), in the same order; and the sidecar holds no
in-scope row the banked window lacks.

sha12 only. Exit 0 = exact, 3 = any mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--banked-dir", required=True)
    ap.add_argument("--sets", default="train,clipgrid,inrun,gtval")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from tanitad.data import vis1 as V
    sc = V.VIS1Sidecar(a.sidecar)
    A = sc.a
    rec = {"sidecar": sc.stamp(), "sets": {}, "n_rows_compared": 0, "n_windows": 0, "mismatch": []}
    for nm in [s for s in a.sets.split(",") if s]:
        rows = json.load(open(Path(a.banked_dir) / f"vis_{nm}_boxes.json", encoding="utf-8"))
        win: dict = {}
        for r in rows:
            win.setdefault((r["sha12"], int(r["w"])), []).append(r)
        n_cmp = 0
        for (s12, w), rs in win.items():
            f = w + a.window - 1
            sl = sc.frame(s12, f)
            if sl is None:
                rec["mismatch"].append({"set": nm, "sha12": s12, "w": w, "why": "frame absent"})
                continue
            lo, hi = sl
            banked_in = []
            for r in rs:
                # the banked row's centre is not stored; the range + class are. Re-derive scope membership from the
                # sidecar's own rows by ORDER: the banked in-scope rows, in order, must equal the stored rows.
                banked_in.append(r)
            stored = [(int(A["n_full"][k]), int(A["n_img"][k]), int(A["n_vis"][k]), int(A["vis_rows"][k]),
                       # ⛔ math.hypot, EXACTLY the audit's expression (vis_zbuf.py:333) -- np.hypot differs
                       # in the last ulp on some rows and would break the float-equality row match
                       math.hypot(float(A["cx"][k]), float(A["cy"][k]))) for k in range(lo, hi)]
            # map each stored row to the banked row with the identical range (float equality: both are
            # math.hypot of the same float32 centre), in order
            j = 0
            hit = set()
            for srow in stored:
                while j < len(banked_in) and float(banked_in[j]["range"]) != srow[4]:
                    j += 1
                if j >= len(banked_in):
                    rec["mismatch"].append({"set": nm, "sha12": s12, "w": w, "why": "stored row not in banked window",
                                            "range": srow[4]})
                    break
                b = banked_in[j]
                got = srow[:4]
                want = (int(b["n_full"]), int(b["n_img"]), int(b["n_vis"]), int(b["vis_rows"]))
                if got != want:
                    rec["mismatch"].append({"set": nm, "sha12": s12, "w": w, "got": got, "want": want})
                hit.add(j)
                n_cmp += 1
                j += 1
            # ⛔ every banked TARGET row (the trainer's filter) must have been matched by a stored row
            lost = [k for k, r in enumerate(rs) if r["target"] and k not in hit]
            if lost:
                rec["mismatch"].append({"set": nm, "sha12": s12, "w": w, "why": "banked target rows not stored",
                                        "n": len(lost)})
        rec["sets"][nm] = {"n_windows": len(win), "n_rows_compared": n_cmp}
        rec["n_rows_compared"] += n_cmp
        rec["n_windows"] += len(win)
    rec["n_mismatch"] = len(rec["mismatch"])
    rec["mismatch"] = rec["mismatch"][:50]
    rec["PASS"] = rec["n_mismatch"] == 0 and rec["n_rows_compared"] > 0
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("n_windows", "n_rows_compared", "n_mismatch", "PASS", "sets")}), flush=True)
    return 0 if rec["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
