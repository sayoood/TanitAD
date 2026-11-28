"""REPRO 1/2: re-derive the banked HELD-OUT P1 (0-40 m) of the six S1 head arms from their trainer dumps with the programme's own readout, compare to S1_READ.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C
import shf_arrays as SA
S1 = json.load(open(os.path.join(C.PK, "raw", "2026-10-15-s1-read", "S1_READ.json"), encoding="utf-8"))
out = {}
D = SA.domain("heldout", arms=tuple(C.ARMS))
for a in C.ARMS:
    i, u, c = SA.frame_iou_parts(D, a)
    mine = SA.pooled_iou(i, u)
    banked = S1["outcome"]["per_arm"]["k2"][a]["p1_ho"]["iou_0_40"]
    # also via the programme's own function (RO.frame_stats + perception_reads), n_boot small: only the point estimate is compared
    p = C.load_heldout_pred(a)
    st = C.RO.frame_stats(p, C.open_train_labels())
    r = C.RO.perception_reads(p, st, None, n_boot=50)
    out[a] = dict(mine_arrays=mine, via_RO=r["p1"]["all"]["0-40"]["est"], banked=banked, abs_diff_arrays=abs(mine - banked), abs_diff_RO=abs(r["p1"]["all"]["0-40"]["est"] - banked),
                  n_frames=int(D["n"]))
    print(a, out[a], flush=True)
C.jdump(out, os.path.join(C.HERE, "b0_repro_heldout.json"))
print("ZZREPRO_HELDOUT max abs diff", max(max(v["abs_diff_arrays"], v["abs_diff_RO"]) for v in out.values()))
