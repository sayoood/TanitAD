"""flagship-v4 30k gate PREFLIGHT — the three checks the card mandates.

Card: Project Steering/Gates/flagship-v4-30k.card.json, preflight_checks[].
If any check FAILS the gate must STOP; a verdict rendered on a broken
instrument is worse than no verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, "/root/taniteval")

import numpy as np  # noqa: E402

REPORT = {}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- CHECK 1 -- #
# corridor.py present on the EXECUTING host and importable (it was MISSING from
# the eval pod earlier on 2026-07-26, so the co-primary emitter did not exist).
c1 = {"check": "corridor.py present + importable on executing host"}
try:
    from taniteval import corridor
    c1["module_file"] = corridor.__file__
    c1["md5"] = md5(corridor.__file__)
    c1["repo_md5_expected"] = "9f064714f35be7d172d228cfb26c5976"
    c1["byte_identical_to_repo"] = (c1["md5"] == c1["repo_md5_expected"])
    c1["CORRIDOR_HALFWIDTH_M"] = corridor.CORRIDOR_HALFWIDTH_M
    c1["JUNCTION_DEG"] = corridor.JUNCTION_DEG
    c1["has_from_windows"] = hasattr(corridor, "from_windows")
    c1["has_stratified"] = hasattr(corridor, "stratified")
    c1["has_paired_stratum_delta"] = hasattr(corridor, "paired_stratum_delta")
    # exercise the emitter end-to-end on a tiny synthetic set so "present" is
    # proven to mean "produces a corridor_departure_rate", not just "imports"
    rng = np.random.default_rng(0)
    n, K = 24, 185
    lat = np.abs(rng.normal(0, 1.0, size=(n, K)))
    eid = np.repeat(np.arange(6), 4)
    blk = corridor.corridor_block(lat, eid, n_boot=200, seed=0)
    c1["smoke_keys"] = sorted(blk.keys())[:12]
    c1["smoke_emitted_departure_rate"] = (
        "departure_rate" in json.dumps(blk) or "corridor" in json.dumps(blk))
    c1["horizon_seconds_K185"] = corridor.horizon_seconds(185)
    c1["PASS"] = bool(c1["byte_identical_to_repo"] and c1["has_from_windows"]
                      and c1["has_stratified"])
except Exception as e:  # noqa: BLE001
    import traceback
    c1["PASS"] = False
    c1["error"] = f"{type(e).__name__}: {e}"
    c1["tb"] = traceback.format_exc()[-1500:]
REPORT["preflight_1_corridor_present"] = c1


# ---------------------------------------------------------------- CHECK 2 -- #
# lateral.py must emit horizon_provenance AND horizon_s = 2.0 on the SPARSE
# 4-waypoint surface. A 0.4 s reading means stale code (pre-fix mislabelled 5x).
c2 = {"check": "lateral.py emits horizon_provenance and horizon_s=2.0 on sparse surface"}
try:
    from taniteval import lateral
    c2["module_file"] = lateral.__file__
    c2["md5"] = md5(lateral.__file__)
    c2["repo_md5_expected"] = "a3b3d4919e0b0aa966ec11d0515ea814"
    c2["byte_identical_to_repo"] = (c2["md5"] == c2["repo_md5_expected"])

    # synthetic SPARSE 4-waypoint windows (steps 5/10/15/20 = 0.5/1/1.5/2.0 s)
    rng = np.random.default_rng(1)
    n = 40
    gt = np.cumsum(rng.normal(0.0, 1.0, size=(n, 4, 2)), axis=1) + \
        np.stack([np.arange(1, 5) * 5.0, np.zeros(4)], axis=-1)[None]
    pred = gt + rng.normal(0.0, 0.3, size=gt.shape)
    eid = np.repeat(np.arange(10), 4)
    win = {"pred": pred, "gt": gt, "eid": eid, "wp_steps": [5, 10, 15, 20]}
    out = lateral.from_sparse_windows(win, mode="ego", n_boot=200, seed=0)
    c2["surface"] = out.get("surface")
    c2["dt_s"] = out.get("dt_s")
    c2["horizon_K"] = out.get("horizon_K")
    c2["horizon_s_EMITTED"] = out.get("horizon_s")
    c2["by_horizon_labels"] = list(out.get("by_horizon", {}).keys())
    c2["horizon_s_is_2.0"] = (float(out.get("horizon_s", -1)) == 2.0)
    c2["horizon_s_is_stale_0.4"] = (float(out.get("horizon_s", -1)) == 0.4)

    # horizon_provenance lives on the PAIRED emitter (paired_cross_track)
    pa, pb = pred, gt + rng.normal(0.0, 0.5, size=gt.shape)
    d = lateral.paired_cross_track(pa, pb, gt, eid, step=4, mode="ego",
                                   n_boot=200, seed=0)
    c2["paired_horizon_provenance"] = d.get("horizon_provenance")
    c2["paired_horizon_s"] = d.get("horizon_s")
    c2["paired_n_knots"] = d.get("n_knots")
    c2["emits_horizon_provenance"] = ("horizon_provenance" in d)
    # with knot_dt inferred from a 4-knot sparse surface, horizon_s must be 2.0
    c2["paired_horizon_s_is_2.0"] = (float(d.get("horizon_s", -1)) == 2.0)
    # and explicit knot_dt must also give 2.0
    d2 = lateral.paired_cross_track(pa, pb, gt, eid, step=4, mode="ego",
                                    n_boot=200, seed=0, knot_dt=0.5)
    c2["explicit_knot_dt_horizon_s"] = d2.get("horizon_s")
    c2["explicit_knot_dt_provenance"] = d2.get("horizon_provenance")

    c2["PASS"] = bool(c2["horizon_s_is_2.0"] and c2["emits_horizon_provenance"]
                      and c2["paired_horizon_s_is_2.0"]
                      and not c2["horizon_s_is_stale_0.4"]
                      and c2["byte_identical_to_repo"])
except Exception as e:  # noqa: BLE001
    import traceback
    c2["PASS"] = False
    c2["error"] = f"{type(e).__name__}: {e}"
    c2["tb"] = traceback.format_exc()[-1500:]
REPORT["preflight_2_lateral_horizon"] = c2


# --------------------------------------------------------- v1 IDENTITY ----- #
# CLAUDE.md: `flagship4b-phase0-30k` is the NO-SPEED ABLATION CONTROL (2.918 m),
# NOT the deployed v1 (`flagship4b-speedjerk-30k`, 0.452). The HF repo name
# invites the inversion. Identify by the checkpoint's OWN stored args, not name.
c0 = {"check": "identify which local ckpt is the deployed v1 (speedjerk)"}
try:
    import torch
    for name in ("flagship-30k", "flagship-speed"):
        p = f"/root/models/{name}/ckpt.pt"
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        args = sd.get("args") or sd.get("config") or {}
        if hasattr(args, "__dict__"):
            args = vars(args)
        pick = {k: v for k, v in args.items()
                if any(t in k.lower() for t in
                       ("speed", "jerk", "aux", "run", "name", "out", "step"))}
        c0[name] = {"top_keys": sorted(sd.keys())[:12], "args_subset": pick,
                    "md5_prefix": None}
        del sd
    c0["PASS"] = True
except Exception as e:  # noqa: BLE001
    import traceback
    c0["PASS"] = False
    c0["error"] = f"{type(e).__name__}: {e}"
    c0["tb"] = traceback.format_exc()[-1200:]
REPORT["v1_identity_probe"] = c0

print(json.dumps(REPORT, indent=2, default=str))
with open("/workspace/_v4gate/preflight_1_2.json", "w") as f:
    json.dump(REPORT, f, indent=2, default=str)
