#!/usr/bin/env python3
"""exec_probe.py - descriptive: where does the first REPAIR-D shift get lost? For each token: requested target shift d = (ext0 + margin) * u at t_v (from the
primary trace, iteration 0) -> P5-R label shift (stage 1: target -> label) -> executed tracked path shift (stage 2: label -> executed), all projected on u.
Reuses n4_repair_trace (pinned run tree). Needs the scratch primary traces + anatomy_25.json. SEEN-ON-NAVTEST, descriptive only."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import n4_repair_trace as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    an = json.load(open(HERE + "/anatomy_25.json", encoding="utf-8"))
    toks, logs, plans, ix, rolls = T.load_inputs()
    si, sn = (int(x) for x in a.shard.split("/"))
    want = [t for i, t in enumerate(sorted(an["tokens"])) if i % sn == si]
    done = T.B.done_tokens(a.out)
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok, lg, cx in T.contexts({t: logs[ix[t]] for t in want if t not in done}):
            rec = {"token": tok}
            try:
                it0 = an["tokens"][tok]["iterations"][0]
                rv = it0["rv"]
                u, ext, t_v = np.asarray(rv["u_ego"]), rv["ext_m"], rv["t_v_s"]
                d = (ext + T.RP.MARGIN) * u
                t20 = rolls[tok]
                L0 = cx.labels(t20)["R"]
                tgt1 = T.RP.apply_shift(t20, d, t_v)
                L1 = cx.labels(tgt1)["R"]
                (r0, r1), (s0, s1) = (lambda res, sim: (res, sim))(*cx.score([L0, L1]))
                k20 = int(min(19, max(0, round(t_v / 0.2) - 1)))
                k41 = int(min(40, round(t_v / 0.1)))
                lab_shift = (L1[k20, :2] - L0[k20, :2]) @ u
                ego0 = cx.ego_frame(s0[k41, :2])
                ego1 = cx.ego_frame(s1[k41, :2])
                exec_shift = float((ego1 - ego0) @ u)
                tgt_shift_at = float((tgt1[k20, :2] - t20[k20, :2]) @ u)
                rec.update({"ext0_m": ext, "t_v_s": t_v, "req_m": float(np.linalg.norm(d)), "target_shift_along_u_m": tgt_shift_at,
                            "label_shift_along_u_m": float(lab_shift), "exec_shift_along_u_m": exec_shift,
                            "stage1_label_over_target": float(lab_shift / tgt_shift_at) if abs(tgt_shift_at) > 1e-9 else None,
                            "stage2_exec_over_label": float(exec_shift / lab_shift) if abs(lab_shift) > 1e-9 else None,
                            "exec_over_req": exec_shift / float(np.linalg.norm(d)),
                            "label_vs_target_gap_along_u_m": float((L0[k20, :2] - t20[k20, :2]) @ u),
                            "ext_after_m": float(cx.ext_dist(s1).max()), "ext_before_m": float(cx.ext_dist(s0).max()),
                            "dac_after": float(r1["dac"])})
            except Exception as e:                                       # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print("done", tok, rec.get("error", ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
