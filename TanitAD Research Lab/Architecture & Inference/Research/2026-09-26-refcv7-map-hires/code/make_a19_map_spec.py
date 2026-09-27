"""SPEC_REFCV7 §24 (A19, the PI 2026-09-27 "Binding = MAIN only, both"): the MAP binding's spec is
the A18 spec (md5 4eda0636) with its must-fail rows REMOVED -- nothing else changes.

WHY a file of its own: the frozen registered harness (`map_hires_overfit.py`, blob 9bec9e88,
main() lines 650-652) REFUSES a run whose spec names must-fail arms that `--arms` does not run
("the gated arm 'lane_w0' is not in --arms"), and the A11 closure judge requires that harness to
be the wrapped script -- so `--arms healthy` on the A18 spec itself cannot run. With no must-fail
rows the harness runs MAIN alone; its verdict then has no regression rows (G_MAP_OVERFIT = MAIN
PASS and C1-C3 and the 1 ms guard), and C1-C3 are still computed on MAIN's logits.

Removed: `must_fail`, `must_fail_all` (load_spec refuses a must_fail_all row without its must_fail
row). Kept: steps 3,000, lr_decay cosine from 2,700, the bars, the frames, the presence floor, the
controls, the informative arms (recorded NOT RUN), near lift 20 + one block (A15's config).
Usage: make_a19_map_spec.py <landing sha of §24>      (run: make_a19_map_spec.py 36cc332)"""
import hashlib
import json
import os
import sys
from pathlib import Path

SPEC_SHA = sys.argv[1]
_HERE = Path(__file__).resolve().parent                 # <repo>/<package>/code
TREE = Path(os.environ.get("A19_TREE", str(_HERE.parents[4])))
PKGRAW = TREE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw"
src = PKGRAW / "gmo_spec_A18.json"
b0 = src.read_bytes()
assert hashlib.md5(b0).hexdigest() == "4eda0636f1a41a9b60b59ccb6a4afca3", "not the A18 spec as registered"
s = json.loads(b0)
assert s["steps"] == 3000 and s["lr_decay"] == {"kind": "cosine_to_zero", "start_step": 2700}
assert set(s["must_fail"]) == {"lane_w0", "s8_zeros", "near_block_zeros"}
out = {}
for k, v in s.items():
    if k in ("must_fail", "must_fail_all"):
        continue                                        # A19: the binding runs MAIN only
    if k == "registered":
        out[k] = (f"2026-09-27 17:46 Berlin: SPEC_REFCV7 §24 (A19, landed {SPEC_SHA}) -- the MAP "
                  f"binding is MAIN-only on the A18 protocol: the A18 spec with its must-fail rows "
                  f"removed (the registered harness refuses a spec whose must-fail arms --arms does "
                  f"not run); registered BEFORE the binding run's first number")
        continue
    if k == "amends":
        out[k] = {"spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                          "2026-09-26-refcv7-map-hires/raw/gmo_spec_A18.json",
                  "spec_md5": hashlib.md5(b0).hexdigest(),
                  "amendment": "Project Steering/SPEC_REFCV7.md §24 (A19)",
                  "what_changes": ("must_fail and must_fail_all REMOVED (MAIN-only binding, the PI "
                                   "17:46 Berlin); every other literal unchanged. The must-fail "
                                   "evidence is INHERITED from A17.1 (A15 config + decay, 1,000 "
                                   "steps): s8_zeros, near_block_zeros and lane_w0 all held")}
        continue
    out[k] = v
out["a19_inherited_must_fail_record"] = (
    "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw/"
    "gmo_early/g_map_overfit_A171.EARLY_NONBINDING.json")
dst = PKGRAW / "gmo_spec_A19_MAP_MAIN.json"
b = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
dst.write_bytes(b)
diff = sorted(k for k in set(s) | set(json.loads(b)) if s.get(k) != json.loads(b).get(k))
print(dst.name, "md5", hashlib.md5(b).hexdigest(), "sha256", hashlib.sha256(b).hexdigest(),
      "diff vs A18:", diff)
