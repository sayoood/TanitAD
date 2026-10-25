#!/usr/bin/env python3
"""verify_sc.py -- the Master Mind's independent re-read of SPEEDCAL's PRIMARY / CEILING (descriptive; changes no row of the read).
Own ADE6 from the banked plans against HSPEC's eval GT (window order checked), own episode-cluster bootstrap (numpy, B = 2000, rng 20261026),
and a reachability check that the calibrated arm's T code actually moved (the lever ran)."""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
EV = "C:/Users/Admin/sc_run/eval"      # the run dir; its 12 eval npz are byte-identical to the banked copy (cmp, verify_sc.log header)
sys.path.insert(0, SCR + "/hspec")
import v8f_hspec as S  # noqa: E402


def boot(x, eid, nb=2000, seed=20261026):
    u, inv = np.unique(eid, return_inverse=True)
    s = np.bincount(inv, weights=x); c = np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(nb, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n_ep": int(len(u))}


out = {}
for k, seed in (("s0", "M2b-mixo_s0"), ("s1", "M2b-mixo_s1")):
    z = {n: np.load(os.path.join(EV, f"eval_{seed}-speedcal-{n}.npz")) for n in ("own", "cal", "ceiling", "identity")}
    eid = z["own"]["eid"]
    assert all((z[n]["eid"] == eid).all() for n in z)
    rec = {}
    try:
        W = S.load_world("M2b_" + k)                      # the GT is arm-independent; HSPEC's world is keyed by the B arm
        assert (np.asarray(W["eid"]) == eid).all(), "window order differs from HSPEC's world"
        gt = np.asarray(W["gt"], np.float64)
    except Exception as e:  # noqa: BLE001
        gt = None
        rec["gt_source"] = f"load_world unavailable for {seed}: {type(e).__name__}: {e}"[:300]
    for n in z:
        a = np.asarray(z[n]["ade6"], np.float64)
        if gt is not None and gt.shape == z[n]["pred"].shape:
            mine = np.linalg.norm(z[n]["pred"].astype(np.float64) - gt, axis=-1).mean(1)
            rec[f"{n}: |my ade6 - banked ade6| max"] = float(np.abs(mine - a).max())
    rec["identity == own (ade6 max abs)"] = float(np.abs(z["identity"]["ade6"] - z["own"]["ade6"]).max())
    rec["cal pred moved vs own (max abs m)"] = float(np.abs(z["cal"]["pred"] - z["own"]["pred"]).max())
    rec["cal tac_code_shift: frac > 0"] = float((z["cal"]["tac_code_shift"] > 0).mean())
    rec["PRIMARY ade6 cal-own"] = boot(z["cal"]["ade6"] - z["own"]["ade6"], eid)
    rec["CEILING ade6 ceiling-own (T0)"] = boot(z["ceiling"]["ade6"] - z["own"]["ade6"], eid)
    out[k] = rec
    print(k, json.dumps(rec, indent=1), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_sc.json"), "w"), indent=1)
