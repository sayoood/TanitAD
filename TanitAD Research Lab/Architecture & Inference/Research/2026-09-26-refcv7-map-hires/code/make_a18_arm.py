"""Generate the A18 G-MAP-OVERFIT spec (SPEC_REFCV7 §23, the PI's "Budget to 3,000 steps") from
the A17.1 arm's spec: steps 1,000 -> 3,000 and the A17.1 decay's start 900 -> 2,700 (lr 1e-3 for
steps 0-2,699, cosine to 0 over 2,700-3,000); A15's configuration (near lift 20 + one near refine
block + sqrt_mf); every bar, must-fail and other literal unchanged.
Usage: make_a18_arm.py <AID> <SEC> <landing sha>      (run: make_a18_arm.py A18 §23 37086c3)"""
import hashlib
import json
import os
import sys
from pathlib import Path

AID, SEC, SPEC_SHA = sys.argv[1:4]
_HERE = Path(__file__).resolve().parent                 # <repo>/<package>/code
TREE = Path(os.environ.get("A18_TREE", str(_HERE.parents[4])))
PKGRAW = TREE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw"
src = PKGRAW / "gmo_spec_A171.json"
b0 = src.read_bytes()
assert hashlib.md5(b0).hexdigest() == "5abd5b907738fc735a0def0c1e7fbb64", "not the A17.1 spec as landed"
s = json.loads(b0)
assert s["steps"] == 1000 and s["lr_decay"] == {"kind": "cosine_to_zero", "start_step": 900}
assert s["near_lift_m"] == 20.0 and s["near_refine_blocks"] == 1
out = {}
for k, v in s.items():
    if k == "registered":
        out[k] = (f"2026-09-27 ~16:55 Berlin: SPEC_REFCV7 {SEC} ({AID}, landed {SPEC_SHA}) -- the PI's "
                  f"'Budget to 3,000 steps': A15's configuration + the A17.1 decay moved to the "
                  f"final 10 % of 3,000 steps, registered BEFORE any {AID} number; every 1,000-step "
                  f"FAIL stays on the record")
        continue
    if k == "amends":
        out[k] = {"spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                          "2026-09-26-refcv7-map-hires/raw/gmo_spec_A171.json",
                  "spec_md5": hashlib.md5(b0).hexdigest(),
                  "amendment": f"Project Steering/SPEC_REFCV7.md {SEC} ({AID})",
                  "what_changes": ("steps 1,000 -> 3,000 (the read-out at 3,000) and lr_decay "
                                   "start_step 900 -> 2,700; every bar, must-fail and other "
                                   "literal unchanged")}
        continue
    if k == "steps":
        out[k] = 3000
        continue
    if k == "lr_decay":
        out[k] = {"kind": "cosine_to_zero", "start_step": 2700}
        continue
    out[k] = v
dst = PKGRAW / f"gmo_spec_{AID}.json"
b = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
dst.write_bytes(b)
md5 = hashlib.md5(b).hexdigest()
diff = sorted(k for k in set(s) | set(json.loads(b)) if s.get(k) != json.loads(b).get(k))
print(dst.name, md5, "diff vs A17.1:", diff)
