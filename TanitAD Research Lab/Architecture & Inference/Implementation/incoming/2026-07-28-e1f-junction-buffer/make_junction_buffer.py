#!/usr/bin/env python3
"""E1f — filter the mined CL-SFT buffer to JUNCTION states only.

WHY: E1d MEASURED that the closed-loop primary's two halves behave differently —
junction recovery is cheap and monotone (better at EVERY alpha, never
separated-worse), while overall-corridor recovery is expensive and
barrier-crossing. The buffer currently supervises ALL recoverable pre-failure
states. E1c/E1d/E1e closed three one-dimensional levers (training time, weight
space, loss weighting); the remaining hypothesis is that the TARGET is wrong, not
its weight.

THRESHOLD: |dpsi| >= radians(10.0) — the evaluator's OWN `--junction-deg 10.0`.
Using the metric's own definition, not a new one invented for the arm.

UNITS: settled by measurement, not assumption — |dpsi| max is 0.8272 and ZERO
records exceed 10 in raw units, so dpsi is in RADIANS.

The output keeps the parent's `meta` verbatim and adds an `e1f` block recording
the filter, so the derived buffer can never be mistaken for the parent.
"""
import argparse, hashlib, json, math
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/e1b/mined_buffer.pt")
    ap.add_argument("--src-md5", default="a32cfe9bfea4b1b5c196d3bb7f71fa5f")
    ap.add_argument("--out", default="/workspace/e1f/junction_buffer.pt")
    ap.add_argument("--junction-deg", type=float, default=10.0)
    a = ap.parse_args()

    src = Path(a.src)
    got = hashlib.md5(src.read_bytes()).hexdigest()
    assert got == a.src_md5, f"source buffer md5 {got} != pinned {a.src_md5}"
    print(f"[e1f] source md5 VERIFIED {got}")

    d = torch.load(src, map_location="cpu", weights_only=False)
    rec = d["records"]
    thr = math.radians(a.junction_deg)

    keep = [r for r in rec if abs(float(r["dpsi"])) >= thr]
    n_src, n_keep = len(rec), len(keep)
    eps_src = {r["episode_id"] for r in rec}
    eps_keep = {r["episode_id"] for r in keep}
    assert n_keep > 0, "junction filter kept nothing"

    meta = dict(d.get("meta", {}))
    meta["e1f"] = {
        "_what": "JUNCTION-RESTRICTED subset of the E1b mined buffer",
        "parent_buffer": str(src),
        "parent_md5": got,
        "junction_deg": a.junction_deg,
        "threshold_rad": thr,
        "dpsi_units": "radians (measured: max 0.8272, 0 records >= 10 raw)",
        "n_records_parent": n_src,
        "n_records_kept": n_keep,
        "frac_kept": round(n_keep / n_src, 6),
        "n_episodes_parent": len(eps_src),
        "n_episodes_kept": len(eps_keep),
    }

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": keep, "meta": meta}, out)
    out_md5 = hashlib.md5(out.read_bytes()).hexdigest()

    print(f"[e1f] kept {n_keep}/{n_src} records ({100*n_keep/n_src:.1f}%) "
          f"across {len(eps_keep)}/{len(eps_src)} episodes")
    print(f"[e1f] wrote {out}")
    print(f"[e1f] OUT_MD5 {out_md5}")
    (out.parent / "junction_buffer_meta.json").write_text(
        json.dumps({**meta["e1f"], "out_md5": out_md5}, indent=1))


if __name__ == "__main__":
    main()
