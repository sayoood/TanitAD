"""plans_pack3.py -- the banked per-window plans of the 4 pre-declared windows INCLUDING HACC (superset of v2's plans_pack.npz) + CONTROLS_V3_part1.json.
CPU only, numpy only, read-only on every source.  Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I viz3/plans_pack3.py"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_lib import *  # noqa

TOL = 1e-5
v2 = np.load(f"{V2}/data/plans_pack.npz")
cf = np.load(f"{OUT}/data/cf_windows.npz")
pack = {k: v2[k] for k in v2.files}                       # v2's arrays, unchanged (GT, B, P10, H, HS, ade6_*, ade_h_*)
ctl = {"per_window": {}, "eval_wide": {}}
for seed in SEEDS:
    hb = np.load(f"{HBLEND}/score_{seed}_perwindow.npz")
    hs = np.load(f"{HSPEC}/score_hs2_{seed}_perwindow.npz")
    hc = np.load(f"{HACC}/score_{seed}_perwindow.npz")
    Wj = json.load(open(f"{HACC}/weights_HACC_{seed}.json"))
    w = np.stack([Wj["w_along_by_step"], Wj["w_cross_by_step"]], 1)          # [30, 2]: axis 0 along (forward), axis 1 cross (left); HACC's own convention
    assert w.shape == (30, 2)
    assert np.array_equal(hb["eid"], hc["eid"]), "eid mismatch hblend vs hacc"
    # ---- eval-wide: banked HACC plan == P + w (.) (tau - P), all 2,399 windows
    formula_all = hb["P"].astype(np.float64) + w * (hb["tau"].astype(np.float64) - hb["P"].astype(np.float64))
    ew = {"HACC_banked_vs_formula_maxabs_m": float(np.abs(hc["HACC"] - formula_all).max()),
          "n_windows": int(len(hc["eid"])),
          "P_vs_P_file_maxabs_m_(info: the P10 file)": float(np.abs(hb["P"] - hb["P_file"]).max())}
    ctl["eval_wide"][seed] = ew
    for name in KEYS:
        wi = WINDOWS[name]["window"]
        P = f"{seed}/{name}"
        gt = pack[f"{P}/GT"].astype(np.float64)
        d = {"GT_pack_vs_cf_windows": float(np.abs(gt - cf[f"{seed}/{name}/gt"]).max())}
        arr = {"B": hb["tau"][wi], "P10": hb["P"][wi], "H": hb["H"][wi], "HS": hs["HS"][wi], "HACC": hc["HACC"][wi]}
        for k in ("B", "P10", "H", "HS"):
            d[f"v2_pack_{k}_vs_source"] = float(np.abs(pack[f"{P}/{k}"] - arr[k]).max())
        pack[f"{P}/HACC"] = arr["HACC"]
        formula = hb["P"][wi].astype(np.float64) + w * (hb["tau"][wi].astype(np.float64) - hb["P"][wi].astype(np.float64))
        d["HACC_drawn_vs_formula_P_plus_w_tau_minus_P"] = float(np.abs(pack[f"{P}/HACC"] - formula).max())
        d["HACC_drawn_vs_banked_perwindow_array"] = float(np.abs(pack[f"{P}/HACC"] - np.load(f"{HACC}/score_{seed}_perwindow.npz")["HACC"][wi]).max())
        for nm, key, src in (("B", "B__ade6", hb), ("H", "H__ade6", hb), ("P10", "P10__ade6", hb), ("HS", "HS__ade6", hs), ("HACC", "HACC__ade6", hc)):
            a6 = np.linalg.norm(arr[nm].astype(np.float64) - gt, axis=-1).mean()
            d[f"ade6_recomputed_vs_banked_{nm}"] = float(abs(a6 - src[key][wi]))
            pack[f"{P}/ade6_{nm}"] = a6
            pack[f"{P}/ade_h_{nm}"] = np.linalg.norm(arr[nm].astype(np.float64) - gt, axis=-1)
        ctl["per_window"][P] = d
        print(P, {k: f"{v:.1e}" for k, v in d.items()}, flush=True)
np.savez_compressed(f"{OUT}/data/plans_pack3.npz", **pack)
json.dump(ctl, open(f"{OUT}/CONTROLS_V3_part1.json", "w"), indent=1)
print("eval-wide", json.dumps(ctl["eval_wide"]))
print("ok")
