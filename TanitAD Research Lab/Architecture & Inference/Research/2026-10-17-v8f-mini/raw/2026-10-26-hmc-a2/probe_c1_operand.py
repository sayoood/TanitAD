"""C-1 diagnosis (after the A1 read REFUSED on C-1; no outcome row exists). Read-only, CPU.
For every MC run: d = (plan - P)_along - c (tau - P)_along at knots 1..12, exactly as hmc_controls.c1_reach computes it, divided by
 (a) spacing(float32(|plan|))                 -- Amendment 1's unit (reproduces the reader's number), and
 (b) spacing(float32(max(|P|, |tau|, |plan|))) -- the largest operand of the float32 blend.
At the argmax of (a) it prints the tick, the knot and the three operands. The deliberate regression (HACC's table) goes through (b) too."""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/hmc")
import hmc_controls as C  # noqa: E402

RUN = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/run"
BR = json.load(open("C:/Users/Admin/hmc/BAKE_REPORT_MC.json"))
NAME = {"MCH": "MC-H", "MCF": "MC-F"}
out = {}


def ratios(z, w, k=12):
    P, T, H = z["P"][:, :k, 0], z["tau"][:, :k, 0], z["plan"][:, :k, 0]
    wk = np.asarray(w, np.float64)[:k, 0]
    d = (H - P) - wk[None] * (T - P)
    up = np.spacing(np.abs(H).astype(np.float32)).astype(np.float64)
    M = np.maximum(np.maximum(np.abs(P), np.abs(T)), np.abs(H))
    uM = np.spacing(M.astype(np.float32)).astype(np.float64)
    ra, rb = np.abs(d) / up, np.abs(d) / uM
    i = np.unravel_index(int(ra.argmax()), ra.shape)
    return {"max_abs_d_m": float(np.abs(d).max()), "max_d_over_ulp_plan": float(ra.max()), "max_d_over_ulp_maxoperand": float(rb.max()),
            "n_over_1ulp_plan": int((ra > 1).sum()), "n_over_1ulp_maxoperand": int((rb > 1).sum()), "n_points": int(ra.size),
            "at_argmax_plan_ratio": {"tick": int(i[0]), "knot": int(i[1] + 1), "plan": float(H[i]), "P": float(P[i]), "tau": float(T[i]), "d": float(d[i])}}


for arm in ("MCH", "MCF"):
    for s in (0, 1):
        w = C.table_from_ckpt(BR["ckpts"][f"s{s}"]["arms"][NAME[arm]]["out"])
        for m in (10, 30):
            z = C.load_ticks(f"{RUN}/ticks/ticks_{arm}_s{s}_m{m}.npz")
            out[f"{arm}_s{s}_m{m}"] = r = ratios(z, w)
            print(f"{arm}_s{s}_m{m}", json.dumps(r))
for s in (0, 1):
    w = C.table_from_ckpt(f"E:/Projects/TanitAD-artifacts/v8f-mini-hacc/M2b_s{s}_hacc.pt")
    z = C.load_ticks(f"{RUN}/ticks/ticks_MCH_s{s}_m10.npz")
    out[f"REGRESSION_HACC_table_on_MCH_s{s}_m10"] = r = ratios(z, w)
    print(f"REGRESSION s{s}", json.dumps({k: r[k] for k in ("max_abs_d_m", "max_d_over_ulp_maxoperand", "n_over_1ulp_maxoperand")}))
json.dump(out, open("C:/Users/Admin/hmc/read/probe_c1_operand.json", "w"), indent=1)
