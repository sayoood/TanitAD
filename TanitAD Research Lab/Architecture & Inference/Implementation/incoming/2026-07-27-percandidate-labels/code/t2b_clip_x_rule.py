"""T1 x T2 — does the kinematic clip stop the RULE scorer from Goodharting?

T1 measured that the clip is FREE and INERT on ade_0_2s: it removes 72.08 % of
REF-C-XL's emitted fan, the ADE-oracle survives in 100 % of windows, and the
as-trained pick does not move in a single window (paired delta exactly 0.0).
So on the ADE surface the tail is not what the argmax was eating.

T2 measured the opposite on the RULE surface: the unconstrained PDMS-lite argmax
lands at 9.71 m ADE, because with no map, no route and no speed limit the Ego
Progress term is unbounded and rewards the fastest candidate in the fan. PDM
DROPS its speed-limit and no-progress terms with the explicit justification that
"the generator enforces them" — we have no such generator.

⇒ THE PREDICTION, stated before the number is computed: the clip should be inert
   for the ADE picker and LARGE for the rule picker. If it is, R3 is a
   PRECONDITION FOR R1 rather than a lever of its own, which is exactly how the
   research framed it — and it is now measured rather than argued.

Everything here recomputes from the staged T2 dump + anchor file. ZERO GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SEL_ACCEL_MAX, HORIZON_S = 2.5, 2.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    D = torch.load(a.dump, map_location="cpu", weights_only=False)
    anc = torch.load(a.anchors, map_location="cpu",
                     weights_only=False)["anchors"].numpy()
    v_term = np.linalg.norm(anc[:, -1, :], axis=-1) / HORIZON_S      # [C]

    FE = np.asarray(D["fan_err"], dtype=np.float64)
    NCF = np.asarray(D["nc_fault"], bool)
    TF = np.asarray(D["ttc_flag"], bool)
    CO = np.asarray(D["comfort_ok"], bool)
    W, C = FE.shape
    if "progress" in D:                                   # full dump layout
        PR = np.asarray(D["progress"], dtype=np.float64)
    else:                                                 # compacted layout
        p = np.asarray(D["progress_per_candidate"], dtype=np.float64)
        PR = np.broadcast_to(p[None] if p.ndim == 1 else p, (W, C))
    v0 = np.asarray(D["v0"], dtype=np.float64)

    reach = SEL_ACCEL_MAX * HORIZON_S
    keep = ((v_term[None] >= np.maximum(v0 - reach, 0.0)[:, None]) &
            (v_term[None] <= (v0 + reach)[:, None]))

    def arms(mask):
        pmax = np.maximum(np.where(mask, PR, -np.inf).max(1, keepdims=True), 1e-6)
        EP = np.where(PR <= 0.0, 0.0, np.clip(PR / pmax, 0.0, 1.0))
        pdms = (~NCF) * (5.0 * EP + 5.0 * (~TF) + 2.0 * CO) / 12.0
        big = np.where(mask, pdms, -np.inf)
        dead = ~mask.any(1)
        out = {}
        for name, sc in (("rule_pdms_lite", big),
                         ("rule_nc_ttc_only",
                          np.where(mask, (~NCF) * (5.0 * (~TF) + 2.0 * CO), -np.inf))):
            p = sc.argmax(1)
            p[dead] = pdms.argmax(1)[dead]
            out[name] = dict(
                ade_0_2s=round(float(np.take_along_axis(FE, p[:, None], 1).mean()), 4),
                mean_pdms_lite=round(
                    float(np.take_along_axis(pdms, p[:, None], 1).mean()), 4),
                at_fault_collision_rate=round(
                    float(np.take_along_axis(NCF, p[:, None], 1).mean()), 4),
                implied_mean_speed_kmh=round(
                    float((v_term[p] * 3.6).mean()), 2))
        m = np.where(mask, FE, np.inf)
        o = m.argmin(1)
        o[dead] = FE.argmin(1)[dead]
        out["oracle_in_fan"] = dict(
            ade_0_2s=round(float(np.take_along_axis(FE, o[:, None], 1).mean()), 4),
            at_fault_collision_rate=round(
                float(np.take_along_axis(NCF, o[:, None], 1).mean()), 4),
            implied_mean_speed_kmh=round(float((v_term[o] * 3.6).mean()), 2))
        out["_frac_windows_empty"] = round(float(dead.mean()), 5)
        return out

    full = np.ones_like(keep)
    res = dict(
        what="T1 x T2 — the kinematic clip applied to the RULE scorer",
        fan="256 FPS demonstration anchors (global vocabulary, no offset head)",
        windows=int(W), candidates=int(C),
        band="v_term in [max(0, v0 - 5.0), v0 + 5.0] m/s (the head's own clamp)",
        frac_candidates_removed=round(float(1.0 - keep.mean()), 4),
        oracle_survives_frac=round(
            float(np.take_along_axis(keep, FE.argmin(1)[:, None], 1).mean()), 4),
        unclipped=arms(full), clipped=arms(keep),
        ego_speed_kmh_mean=round(float((v0 * 3.6).mean()), 2))
    u, c = res["unclipped"], res["clipped"]
    res["verdict"] = dict(
        rule_pick_ade_before=u["rule_pdms_lite"]["ade_0_2s"],
        rule_pick_ade_after=c["rule_pdms_lite"]["ade_0_2s"],
        rule_pick_speed_kmh_before=u["rule_pdms_lite"]["implied_mean_speed_kmh"],
        rule_pick_speed_kmh_after=c["rule_pdms_lite"]["implied_mean_speed_kmh"],
        ade_oracle_pick_ade_before=u["oracle_in_fan"]["ade_0_2s"],
        ade_oracle_pick_ade_after=c["oracle_in_fan"]["ade_0_2s"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
