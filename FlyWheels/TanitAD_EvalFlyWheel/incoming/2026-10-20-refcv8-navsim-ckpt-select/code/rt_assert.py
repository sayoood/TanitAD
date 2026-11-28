#!/usr/bin/env python3
"""Assert the R8_RT arm ran on the REGISTERED per-frame RETIME table (2aacc9ff...) -- by CONTENT, not by a recorded field.

  (1) the frozen per-frame train-fit table hashes to 2aacc9ff... (the registered sha256, a LITERAL);
  (2) the stage-loadable deploy table hashes to its sidecar and to 3319b003... (a LITERAL) and its own `source_sha256`
      field names 2aacc9ff...;
  (3) INDEPENDENT re-derivation: the deploy table's 8 knots / gains equal, exactly, the frozen table's
      surfaces S2.all.RETIME at 0.5/1/1.5/2 s and S6.all.RETIME at 3/4/5/6 s (read off the frozen JSON, not the deploy one);
  (4) every seam_R8_RT*.manifest.json under the given bridge dirs records table_sha256 == 3319b003... and gains equal (3);
  (5) the stage ACTED: R8_RT knots differ from the source R8_LEGAL* knots on >= 1 token (a no-op stage would pass 1-4).

    python rt_assert.py <frozen.json> <deploy.json> <out.json> <bridge_dir> [<bridge_dir> ...]
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

REG_FROZEN = "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80"
REG_DEPLOY = "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"
frozen, deploy, out = sys.argv[1:4]
dirs = sys.argv[4:]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


rec = {"checks": {}}
fz, dp = sha(frozen), sha(deploy)
side = open(deploy + ".sha256", encoding="utf-8").read().split()[0].strip().lower()
F, D = json.load(open(frozen, encoding="utf-8")), json.load(open(deploy, encoding="utf-8"))
rec["checks"]["1_frozen_is_registered_2aacc9ff"] = (len(fz) == 64 and fz == REG_FROZEN)
rec["checks"]["2_deploy_sidecar_and_literal"] = (len(dp) == 64 and dp == side == REG_DEPLOY and D["source_sha256"] == REG_FROZEN)
g = {}
for t, v in zip(F["surfaces"]["S2"]["instants_s"], F["surfaces"]["S2"]["all"]["RETIME"]):
    g[float(t)] = float(v)
for t, v in zip(F["surfaces"]["S6"]["instants_s"], F["surfaces"]["S6"]["all"]["RETIME"]):
    if float(t) > 2.0:
        g[float(t)] = float(v)
kn = sorted(g)
gn = [g[t] for t in kn]
rec["rederived"] = {"knots_s": kn, "gains": gn}
rec["checks"]["3_deploy_equals_rederivation_exactly"] = (kn == [float(x) for x in D["knots_s"]] and gn == [float(x) for x in D["gains"]])
mans, acted = [], []
for d in dirs:
    for mp in sorted(glob.glob(os.path.join(d, "seam_R8_RT*.manifest.json"))):
        m = json.load(open(mp, encoding="utf-8"))
        r = m.get("retime") or {}
        ok = (r.get("table_sha256") == REG_DEPLOY and [float(x) for x in r.get("gains", [])] == gn
              and [float(x) for x in r.get("knots_s", [])] == kn and m.get("partial") is False)
        arm = m.get("arm")
        src_arm = {"R8_RT": "R8_LEGAL"}.get(arm, str(m.get("derived_from", "")).replace("rows_", "").replace(".jsonl", ""))
        a = np.load(os.path.join(d, f"seam_{arm}.npz"))
        b = np.load(os.path.join(d, f"seam_{src_arm}.npz"))
        ta, tb = list(a["token"]), list(b["token"])
        common = [t for t in ta if t in tb]
        ka = {t: a["knots"][i] for i, t in enumerate(ta)}
        kb = {t: b["knots"][i] for i, t in enumerate(tb)}
        dmax = max(float(np.abs(ka[t] - kb[t]).max()) for t in common) if common else None
        mans.append({"manifest": mp, "arm": arm, "source_arm": src_arm, "n_rows": m.get("n_rows"), "partial": m.get("partial"),
                     "table_sha256": r.get("table_sha256"), "stage_sha256": r.get("stage_sha256"), "ok": ok,
                     "max_abs_knot_change_vs_source_m": dmax})
        acted.append(dmax is not None and dmax > 0.0)
rec["manifests"] = mans
rec["checks"]["4_every_RT_manifest_on_registered_table"] = bool(mans) and all(x["ok"] for x in mans)
rec["checks"]["5_stage_acted_on_every_RT_seam"] = bool(acted) and all(acted)
rec["verdict"] = "PASS" if all(rec["checks"].values()) else "FAIL"
rec["frozen_sha256"], rec["deploy_sha256"] = fz, dp
json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
print("ZZRT " + json.dumps({"verdict": rec["verdict"], **rec["checks"], "n_manifests": len(mans)}))
