#!/usr/bin/env python3
"""The FOUR METRIC FAMILIES for refcv6 on the NavSim scenes that carry a logged human future
(TANITAD VENV; the NavSim adapter is IMPORTED, not re-implemented).

Where a family is computable (SPEC §8):
* navhard STAGE 1 — 450 ORIGINAL scenes with real frames and the logged human future in the export
  (``human_future_poses``, the devkit's own ``Scene.get_future_trajectory`` = what ``HumanAgent``
  returns); the refcv6 row is the model's (not a stand-in) because the stage-1 frames exist;
* navtest — every token (single stage, original frames, human future exported by W3).
Warmup stage 2 and navhard stage 2 are synthetic starts with no human who drove them:
LONGITUDINAL / LATERAL / TACTICAL are UNAVAILABLE there with reason + n (not computed here).

``taniteval/adapters/navsim.py`` (the D: tree: it carries the ``log_names`` clustering the ev6 tip's
copy lacks) -> ``scenes_to_win(pred, human, frame="ego", origin_included=False, dt_s=0.5,
log_names=...)`` -> ``four_families_block(win, tier="T1")``. The devkit CV plan on the same scenes
(``cv_poses`` in the export) is computed beside it as the reference row. STRATEGIC: NavSim scores no
strategic decision and this arm has no strategic layer (``--no-strategic``) — the adapter's own
per-family refusal is kept.

    python code/families6.py --seam raw/bridge_navhard_s1000/seam_R6_A1.npz \
        --inputs <export.json> --stage 1 --label PIPELINE-VALIDATION --out raw/families_navhard_s1000.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys

import numpy as np

ADAPTER = "D:/Projects/TanitAD/taniteval/adapters/navsim.py"
D_TANITEVAL = "D:/Projects/TanitAD/taniteval"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--stage", type=int, default=1, help="scenes of this stage (navtest: 1)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if D_TANITEVAL not in sys.path:
        sys.path.append(D_TANITEVAL)            # AFTER the stack: the model tree is never shadowed
    spec = importlib.util.spec_from_file_location("navsim_adapter6", ADAPTER)
    ad = importlib.util.module_from_spec(spec)
    sys.modules["navsim_adapter6"] = ad
    spec.loader.exec_module(ad)
    doc = (json.load(gzip.open(a.inputs, "rt", encoding="utf-8")) if a.inputs.endswith(".gz")
           else json.load(open(a.inputs, encoding="utf-8")))
    toks_all = doc["tokens"]
    z = np.load(a.seam, allow_pickle=False)
    src = [str(s) for s in z["source"]] if "source" in z.files else ["precomputed"] * len(z["token"])
    idx = {str(t): i for i, t in enumerate(z["token"]) if src[i] != "cv_standin"}
    toks = sorted(t for t in idx if int(toks_all[t].get("stage", 1)) == a.stage
                  and toks_all[t].get("human_future_poses") is not None)
    out = {"_label": a.label, "seam": os.path.abspath(a.seam), "stage": a.stage, "n": len(toks),
           "tier": "T1-family (stage-1 loop OPEN; the plan is the model's single query)",
           "adapter": ADAPTER, "families": {}}
    if not toks:
        out["status"] = "UNAVAILABLE"
        out["reason"] = "no model row on a scene with a logged human future"
    else:
        human = np.asarray([toks_all[t]["human_future_poses"] for t in toks], np.float64)
        v0 = [float(np.hypot(*toks_all[t]["ego_statuses"][-1]["ego_velocity"])) for t in toks]
        logs = [toks_all[t]["log_name"] for t in toks]
        preds = {"refcv6": np.asarray([z["poses"][idx[t]] for t in toks], np.float64),
                 "CV_official": np.asarray([toks_all[t]["cv_poses"] for t in toks], np.float64)}
        for name, pp in preds.items():
            try:
                w = ad.scenes_to_win(pp, human, frame="ego", origin_included=False, dt_s=0.5,
                                     scene_tokens=toks, ego_speed_mps=v0, log_names=logs)
                out["families"][name] = ad.four_families_block(w, tier="T1", n_boot=a.n_boot)
            except Exception as ex:                                        # noqa: BLE001
                out["families"][name] = {"status": "RAISED", "error": f"{type(ex).__name__}: {ex}"}
        out["n_logs"] = len(set(logs))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({"n": out["n"], "n_logs": out.get("n_logs"),
                      "families": {k: {f: (v.get(f) or {}).get("status", "OK")
                                       for f in ("longitudinal", "lateral", "tactical", "strategic")}
                                   for k, v in out["families"].items() if isinstance(v, dict)}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
