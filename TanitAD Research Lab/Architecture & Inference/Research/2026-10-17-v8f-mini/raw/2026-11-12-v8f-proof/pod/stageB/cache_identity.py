"""Control r6 of HIERREP (note N1 to PREREG_V8F_PROOF_ADD11): the pod-built eval token cache against the box-2-equal eval cache
that was uploaded to the pod, clip by clip. It replaces the file-hash gate of the frame build by a measured bound.

  python cache_identity.py --pod <cache root with eval/clips> --ref <cache root with eval/clips> --out r6_result.json
Bars (written before the data): on EVERY common clip the kinematic fields are equal within 1e-4 (same logs, so a wrong clip,
a wrong timing or a wrong frame count shows here) and the pooled 4x10 tokens differ by rel-L2 <= 5e-2; the clip sets are equal.
rc 0 = PASS, 3 = FAIL, 2 = cannot run."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

KIN = ("v", "a", "yaw_rate", "pose_grid", "kin3")
TOK_BAR = 5e-2
KIN_BAR = 1e-4


def rel_l2(a, b):
    a, b = a.to(torch.float32), b.to(torch.float32)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fp = {os.path.basename(p): p for p in glob.glob(os.path.join(a.pod, "eval", "clips", "*.pt"))}
    fr = {os.path.basename(p): p for p in glob.glob(os.path.join(a.ref, "eval", "clips", "*.pt"))}
    res = {"control": "r6 cache identity", "tok_bar_rel_l2": TOK_BAR, "kin_bar_abs": KIN_BAR, "n_pod": len(fp), "n_ref": len(fr)}
    if not fp or not fr:
        res.update({"status": "CANNOT-RUN", "reason": "an eval cache is empty"})
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        print("r6", json.dumps(res))
        return 2
    common = sorted(set(fp) & set(fr))
    rows, bad = [], []
    for n in common:
        p = torch.load(fp[n], map_location="cpu", weights_only=False)
        r = torch.load(fr[n], map_location="cpu", weights_only=False)
        row = {"clip": n[:8]}
        if p["pool4x10"].shape != r["pool4x10"].shape:
            row["shape"] = [list(p["pool4x10"].shape), list(r["pool4x10"].shape)]
            bad.append(n[:8] + ":shape")
            rows.append(row)
            continue
        row["tok4x10"] = rel_l2(p["pool4x10"], r["pool4x10"])
        row["tok8x20"] = rel_l2(p["pool8x20"], r["pool8x20"]) if "pool8x20" in p and "pool8x20" in r else None
        row["mean"] = rel_l2(p["mean"], r["mean"]) if "mean" in p and "mean" in r else None
        kd = 0.0
        for k in KIN:
            if k in p and k in r and p[k].shape == r[k].shape:
                d = (p[k].to(torch.float64) - r[k].to(torch.float64)).abs()
                d = d[torch.isfinite(d)]
                kd = max(kd, float(d.max()) if d.numel() else 0.0)
            else:
                kd = float("inf")
        row["kin_max_abs"] = kd
        if not (row["tok4x10"] <= TOK_BAR):
            bad.append(n[:8] + ":tokens")
        if not (kd <= KIN_BAR):
            bad.append(n[:8] + ":kinematics")
        rows.append(row)
    t = np.array([x["tok4x10"] for x in rows if "tok4x10" in x], dtype=np.float64)
    k = np.array([x["kin_max_abs"] for x in rows if "kin_max_abs" in x], dtype=np.float64)
    sets_equal = set(fp) == set(fr)
    ok = bool(sets_equal and not bad and len(common) > 0)
    res.update({"status": "PASS" if ok else "FAIL", "n_common": len(common), "clip_sets_equal": sets_equal,
                "only_pod": sorted(x[:8] for x in set(fp) - set(fr))[:10], "only_ref": sorted(x[:8] for x in set(fr) - set(fp))[:10],
                "tok4x10_rel_l2": {"min": float(t.min()), "median": float(np.median(t)), "max": float(t.max())} if t.size else None,
                "kin_max_abs": {"median": float(np.median(k)), "max": float(k.max())} if k.size else None,
                "n_bad": len(bad), "bad": bad[:20], "rows": rows})
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("r6", json.dumps({x: res[x] for x in ("status", "n_common", "clip_sets_equal", "tok4x10_rel_l2", "kin_max_abs", "n_bad")}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
