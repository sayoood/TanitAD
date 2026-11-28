"""probe_check.py <probe g0 json> <step> -- print 1 iff the 1-seed / 1-batch probe produced a usable artifact:
an eval_traj for seed 0, the ckpt's step == <step>, a STRICT load with no missing / unexpected keys, an eval dataset with
the pose-sync sidecar attached (so the omission this package patches cannot recur silently), and the loader resolved
from the launch tree. Reads the final json, else the .partial.json; prints 0 otherwise. Used by chain_g0_r8.sh."""
import json
import os
import sys

p, step = sys.argv[1], int(sys.argv[2])
rp = p[:-5] + ".r8_repoint.json"
for q in (p, p[:-5] + ".partial.json"):
    if os.path.exists(q) and os.path.getsize(q) > 1000:
        g = json.load(open(q, encoding="utf-8"))
        sd = (g.get("model") or {}).get("state_dict") or {}
        row = ((g.get("by_seed") or {}).get("0") or {}).get("row") or {}
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        ps = rep.get("pose_sync_stats") or {}
        ok = (row.get("eval_traj") is not None and g.get("step") == step and not sd.get("missing")
              and not sd.get("unexpected") and rep.get("pose_sync_patch_enabled") is True
              and ps.get("n_covered") == ps.get("n_clips") and (ps.get("n_clips") or 0) > 0
              and "lgs" in str(g.get("tanitad_file", "")).replace("\\", "/"))
        print("1" if ok else "0")
        sys.exit(0)
print("0")
