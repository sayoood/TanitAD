#!/usr/bin/env python3
"""P3 - corpus-wide ORIENTATION / REGISTRATION check of every accepted P2 artifact.

For each clip: pooled hit rate of its `obstacle.offline` boxes (the B1 EVAL join -- an
INDEPENDENT source) on `cart_occ` over observed cells, the same for the MIRRORED boxes, and
the no-information marginal. Pooled over cells, never a mean of per-frame ratios.

⭐ The robust statistic is the RATIO real/mirror (09-11 §5.2: a wrong arm's absolute lift is a
statement about the scene). Reported per clip and pooled, with the count of clips where the
mirror beats the real boxes -- the per-clip failure a pooled number can hide.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bev_gt_loader as L  # noqa: E402

GT = Path(r"C:\Users\Admin\tanitad-caches\bevhead-20260913\bev_gt")
JOIN = (HERE.parents[3] / "Benchmarks & Evals" / "Research" / "2026-09-06-b1-agent-join"
        / "raw" / "b1eval_agents.jsonl.xz")


def main() -> int:
    join: dict = {}
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            k = hashlib.sha256(r["clip_id"].encode()).hexdigest()[:12]
            join.setdefault(k, []).append((float(r["t_s"]), r.get("agents", [])))
    man = {}
    for line in (GT / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        man[r["clip_sha12"]] = r
    per, tot = [], {"real": [0.0, 0], "mirror": [0.0, 0], "marg": [0.0, 0]}
    for s, r in sorted(man.items()):
        if not r.get("ok"):
            continue
        res = L.registration_check(GT / r["artifact"], join.get(s, []))
        res["clip_sha12"] = s
        per.append(res)
        for k in ("real", "mirror"):
            if res[f"{k}_n_cells"]:
                tot[k][0] += res[f"{k}_hit_rate"] * res[f"{k}_n_cells"]
                tot[k][1] += res[f"{k}_n_cells"]
        if res["n_observed_cells"]:
            tot["marg"][0] += res["marginal"] * res["n_observed_cells"]
            tot["marg"][1] += res["n_observed_cells"]
    real = tot["real"][0] / tot["real"][1]
    mirror = tot["mirror"][0] / tot["mirror"][1]
    marg = tot["marg"][0] / tot["marg"][1]
    with_boxes = [p for p in per if p["real_n_cells"] > 0 and p["mirror_n_cells"] > 0]
    out = {
        "schema": "tanitad.bevgt_orientation_all/1",
        "evidence_class": "MEASURED (ours; independent source = obstacle.offline boxes of the B1 EVAL join)",
        "n_clips": len(per), "n_clips_with_box_cells": len(with_boxes),
        "pooled": {"real_hit_rate": real, "mirror_hit_rate": mirror, "marginal": marg,
                   "real_over_mirror": real / mirror, "real_over_marginal": real / marg,
                   "real_cells": tot["real"][1], "mirror_cells": tot["mirror"][1],
                   "observed_cells": tot["marg"][1]},
        "per_clip_real_over_mirror": {
            "min": float(min(p["real_over_mirror"] for p in with_boxes)),
            "median": float(np.median([p["real_over_mirror"] for p in with_boxes])),
            "max_finite": float(max(p["real_over_mirror"] for p in with_boxes if math.isfinite(p["real_over_mirror"]))),
            "n_clips_mirror_hit_rate_zero": int(sum(not math.isfinite(p["real_over_mirror"]) for p in with_boxes)),
            "n_clips_mirror_beats_real": int(sum(p["real_over_mirror"] < 1.0 for p in with_boxes)),
        },
        "clips": per,
    }

    def finite(o):          # strict JSON: a non-finite ratio (mirror hit rate 0) is written as null
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [finite(v) for v in o]
        return o
    out = finite(out)
    (HERE.parent / "raw" / "p3_orientation_all.json").write_text(json.dumps(out, indent=1, allow_nan=False),
                                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "clips"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
