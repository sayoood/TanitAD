"""progress_headroom.py -- how much of E1 is a PROGRESS-ALONG-THE-PATH problem, and how much of that does the frozen stage already
take?  (refcv9 L1 design annex, items 1 and 4.  CPU only, banked dumps, no model call.)

Every arm keeps refcv8's emitted PATH (the polyline through (0,0) and os's S2 points) and changes only the arc length at each instant
(RT.retime_* geometry = the shipped stage's geometry, S2 4-point form = RETIME_TRAINFIT on S2):
    os              s' = s_os                                       (identity; control)
    RETIME_TRAIN    s' = s_P + g_k (s_os - s_P), g = the frozen TRAIN table on S2 [0.967, 0.619, 0.551, 0.531]  (control: must
                    reproduce FORWARD_TRAINFIT_step30000ema_unc.json  -0.0318 / -0.0313 vs ha0_ext_pose)
    CONST_G05       g = 0.5                                         (control: -0.0308 / -0.0303)
    PROG_ORC        s' = s_orc  (the a_star candidate's progress, T0: a_star uses GT)   -> what a selector that picked the right
                    SPEED (not the right path) would buy on top of refcv8's path
    PROG_HUMAN      s' = s_H    (T0 ceiling of ANY re-timing of this path)
E1 form: S2 ADE(arm) - ADE(ha0_ext_pose), ceiling-compliant windows, paired episode-cluster bootstrap B 2000 (pair_read_r8.paired_cell
-> taniteval.ci), 'another draw of EPISODES' only, single training seed, inference seeds 0 / 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import along_track_localise as AL  # noqa: E402  (its loaders; imports RT from the battery package)

RT = AL.RT
TREE = "step30000ema_unc"
FROZEN = AL.BATT / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
TRAIN_S2 = json.loads(FROZEN.read_text(encoding="utf-8"))["surfaces"]["S2"]["all"]["RETIME"]   # full precision (the deployed table rounds)
BANKED = AL.BATT / "raw" / "retime_trainfit" / "FORWARD_TRAINFIT_step30000ema_unc.json"


def at_progress(O, s) -> np.ndarray:
    geo = RT.polyline_geometry(np.asarray(O, np.float64))
    s = np.maximum(np.asarray(s, np.float64), 0.0)
    return np.stack([RT.eval_polyline(geo, s[:, k]) for k in range(s.shape[1])], 1).astype(np.asarray(O).dtype)


def main() -> int:
    t0 = time.time()
    import compliant_reread_r8 as CR
    tools = RT.tools_load()
    PR, ra = tools["PR"], tools["ra"]
    banked = json.loads(BANKED.read_text(encoding="utf-8"))
    out = {"tool": "progress_headroom.py", "generated": time.strftime("%FT%T"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tree": TREE,
           "tier": "T1 open loop (os, RETIME_TRAIN, CONST_G05); PROG_ORC / PROG_HUMAN are T0 ceilings (use GT)",
           "estimator": "pair_read_r8.paired_cell (paired episode-cluster bootstrap B 2000, seed 0), episodes only, models fixed",
           "seeds": {}, "controls": {}}
    first = {}
    for seed in AL.SEEDS:
        ctx = CR.prepare_seed(TREE, seed, tools, tools["rc"])
        if ctx["sha"] != AL.MASK_SHA:
            raise SystemExit("mask sha mismatch")
        S = RT.surface_inputs(tools, ctx, "S2", first, TREE)
        D = AL.gather(tools, ctx, S, TREE)
        sel = D["comp"]
        O, HP, G, ORC = D["O"][sel], D["HP"][sel], D["G"][sel], D["ORC"][sel]
        eid = list(np.asarray(S["eid"])[sel])
        dt = S["dt"]
        arms = {"os": O,
                "RETIME_TRAIN": RT.retime_apply(O, HP, np.asarray(TRAIN_S2, np.float64)),
                "CONST_G05": RT.retime_apply(O, HP, np.full(4, 0.5)),
                "PROG_ORC": at_progress(O, AL.arc(ORC)),
                "PROG_HUMAN": at_progress(O, AL.arc(G)),
                "ORC_itself": ORC}
        ade = {k: PR.components(ra, v, G, dt)["ade_m"] for k, v in arms.items()}
        ade_hp = PR.components(ra, HP, G, dt)["ade_m"]
        cells = {k: PR.paired_cell(v, ade_hp, eid) for k, v in ade.items()}
        cells_vs_retime = {k: PR.paired_cell(ade[k], ade["RETIME_TRAIN"], eid) for k in ("PROG_ORC", "PROG_HUMAN", "CONST_G05", "os")}
        b = banked["seeds"][str(seed)]["surfaces"]["S2"]["sets"]["compliant"]
        out["controls"][f"s{seed}"] = {
            "os_minus_pose": [cells["os"]["delta"], b["os_minus_ha0_ext_pose"]["delta"]],
            "RETIME_TRAIN_minus_pose": [cells["RETIME_TRAIN"]["delta"], b["arms"]["RETIME_TRAINFIT"]["minus_ha0_ext_pose"]["delta"]],
            "CONST_G05_minus_pose": [cells["CONST_G05"]["delta"], b["arms"]["CONST_G05"]["minus_ha0_ext_pose"]["delta"]],
            "mean_ade_RETIME": [float(np.mean(ade["RETIME_TRAIN"])), b["arms"]["RETIME_TRAINFIT"]["mean_ade_m"]]}
        c = out["controls"][f"s{seed}"]
        c["PASS"] = bool(all(abs(round(x, 4) - y) < 1e-9 for x, y in (c["os_minus_pose"], c["RETIME_TRAIN_minus_pose"], c["CONST_G05_minus_pose"]))
                         and abs(c["mean_ade_RETIME"][0] - c["mean_ade_RETIME"][1]) < 1e-6)
        out["seeds"][str(seed)] = {"n_windows": int(sel.sum()), "minus_ha0_ext_pose": cells, "minus_RETIME_TRAIN": cells_vs_retime,
                                   "mean_ade_m": {k: float(np.mean(v)) for k, v in ade.items()} | {"ha0_ext_pose": float(np.mean(ade_hp))}}
        del ctx
    out["controls"]["ALL_PASS"] = bool(all(v["PASS"] for k, v in out["controls"].items() if k.startswith("s")))
    (HERE / "PROGRESS_HEADROOM.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[{time.time() - t0:.1f}s] WROTE PROGRESS_HEADROOM.json controls ALL_PASS={out['controls']['ALL_PASS']}")
    print("ZZHEADROOMDONEZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
