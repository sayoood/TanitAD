#!/usr/bin/env python3
"""Turn a ``build_frames.py`` output directory into a BANK the bridge will open.

``build_frames.py`` writes ``BUILD_REPORT.json``; ``model_arms.run_model_arms`` opens
``BUILD.json`` and hands its ``frame`` block to ``bridge.frame_tag_check``, which REFUSES a
geometry that is not the model's. This writes that ``BUILD.json`` from the report — ⛔ never
from a hand-typed constant, so the tag the bridge checks is the tag the builder actually used.

It also asserts the bank on CONTENT before declaring it usable (CLAUDE.md: a decode that raises
into a pre-allocated memmap leaves a full-size file of ZEROS and the job can still exit 0):
every sampled scene must be u8 [4,256,640,3], sha-match its provenance row, and have a NON-ZERO
mean. A floor arm of zeros scores at chance and makes everything look like a winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sample", type=int, default=24, help="scenes to verify by CONTENT")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import pandas as pd
    rep = json.load(open(os.path.join(a.bank, "BUILD_REPORT.json"), encoding="utf-8"))
    prov = pd.read_parquet(os.path.join(a.bank, "frames_provenance.parquet"))
    n = len(prov)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(n, size=min(a.sample, n), replace=False)
    bad, means = [], []
    for i in idx:
        row = prov.iloc[int(i)]
        p = os.path.join(a.bank, "frames", f"{row.scene_token}.npy")
        arr = np.load(p)
        sha = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        m = float(arr.mean())
        means.append(m)
        if arr.dtype != np.uint8 or arr.shape != (4, 256, 640, 3):
            bad.append({"scene_token": str(row.scene_token), "why": f"{arr.dtype}{arr.shape}"})
        elif sha != row.sha256:
            bad.append({"scene_token": str(row.scene_token), "why": f"sha {sha} != provenance {row.sha256}"})
        elif m < 1.0:
            bad.append({"scene_token": str(row.scene_token), "why": f"mean {m:.4f} — an all-black scene"})
    content = {"n_sampled": len(idx), "n_bad": len(bad), "bad": bad[:6],
               "mean_px_min": round(min(means), 3) if means else None,
               "mean_px_max": round(max(means), 3) if means else None,
               "pass": not bad,
               "what": ("u8 [4,256,640,3] + sha256[:16] == provenance + mean > 1.0 per sampled scene; "
                        "a bank of zeros would pass a file-size check and poison every arm that reads it")}
    build = {"built": __import__("time").strftime("%Y-%m-%d"), "split": a.split,
             "corpus_id": hashlib.sha256(
                 ("".join(sorted(prov.sha256.astype(str)))).encode()).hexdigest()[:16],
             "parity": "NON-PARITY (NavSim)", "n_scenes": int(n), "n_failed": int(rep["n_failed"]),
             "n_rigs": int(rep["n_rigs"]), "frame": rep["frame"],
             "cameras": ["cam_l0", "cam_f0", "cam_r0"],
             "stage": rep["stage"], "seconds": rep["seconds"],
             "observed_frac_min": float(prov.observed_frac.min()),
             "mean_px_mean": round(float(prov.mean_px.mean()), 3),
             "builder": ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/"
                         "code/build_frames.py (E2), reproduction-controlled against the DataFlyWheel "
                         "warmup corpus"),
             "content_assertion": content}
    with open(os.path.join(a.bank, "BUILD.json"), "w", encoding="utf-8") as fh:
        json.dump(build, fh, indent=1)
    print(json.dumps({k: build[k] for k in ("split", "n_scenes", "n_failed", "n_rigs", "frame",
                                            "observed_frac_min", "content_assertion")}, indent=1))
    return 0 if content["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
