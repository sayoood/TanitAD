#!/usr/bin/env python3
"""P6's on-policy PDM labels through the metric cache: the same scoring call as onpolicy_relabel_pdm.label_log, with the 6 scenario parts read from the
D9 cache that P6's own online labeller read (online_labeller.load_parts), so no scenario is rebuilt from the nuPlan DBs.
Same command line, same side files, same log lines as onpolicy_relabel_pdm.py: it patches label_log and _init, then calls that tool's main().
Run with cwd = the P6 package's refe/ directory and P6_METRIC_CACHE set. A set with no cache file counts as skipped, and pdm_label_complete's --expect catches it."""
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.getcwd())
import onpolicy_relabel_pdm as ORP  # noqa: E402

_C: dict = {}


def init_cached():
    import navsim_pdm_targets as PT
    import online_labeller as OL
    PT._setup()
    _C.update(PT=PT, OL=OL, root=os.environ["P6_METRIC_CACHE"])


def cached_label_log(job):
    log, rows = job
    out, failed, skipped = [], 0, 0
    PT, OL, root = _C["PT"], _C["OL"], _C["root"]
    for r in rows:
        path = OL.parts_path(root, r["log_name"], r["token"])
        if not path.is_file():
            skipped += 1
            continue
        try:
            props = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
            t = PT.score_plans(OL.load_parts(path), PT.executed_plans8(props, repair=True))
            rec = {"kind": "pdm_targets", "key": [r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))],
                   "ckpt_step": int(r["ckpt_step"]), "label_version": int(r.get("label_version", 1)),
                   "plans": "executed (A7 repair)"}
            rec.update({k: [round(float(x), 5) for x in v] for k, v in t.items()})
            out.append(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            failed += 1
            if failed <= 2:
                print(f"  {r.get('token')}: {traceback.format_exc()[-400:]}", flush=True)
    return out, failed, skipped, None


if __name__ == "__main__":
    if not os.environ.get("P6_METRIC_CACHE"):
        raise SystemExit("P6_METRIC_CACHE is not set")
    ORP.label_log = cached_label_log
    ORP._init = init_cached
    sys.exit(ORP.main())
