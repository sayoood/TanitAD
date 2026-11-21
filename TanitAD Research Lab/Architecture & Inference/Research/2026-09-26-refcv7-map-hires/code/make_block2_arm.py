"""PREPARED, NOT RUN (the Master Mind, 2026-09-27 after A17.1's edge FAIL: "prepare, but do not
run, the second near refine block arm. It goes to the PI."): generate the second-block arm's spec
and runner from the A17.1 arm's (A15's config + the lr decay):
  * ``near_refine_blocks`` 1 -> 2 (the flag already allows up to 4; NO code change);
  * must-fail unchanged: s8_zeros (both lifts; all five thin classes), near_block_zeros (zeros
    into BOTH blocks -- ``HiresRefine`` passes ``zero_input`` to every block -- so the arm is the
    A12 function plus constants: edge must fail); lane_w0 kept;
  * every other literal unchanged, the A17.1 decay included.
Without arguments it writes a DRAFT spec (``registered`` says so) for review; with the
registration's <AID> <SEC> <landing sha> it writes the arm's spec and runner.
``B2_BASE=A18`` builds it on the A18 protocol instead (SPEC_REFCV7 §23: 3,000 steps, the decay
from 2,700) -- the DRAFT is then ``gmo_spec_BLOCK2_A18_DRAFT.json`` (spec only; its early MAIN
runs with ``gmo_a18_main.py``, the spec path / md5 and ``near_refine_blocks`` swapped).
Usage: [B2_BASE=A18] make_block2_arm.py [<AID> <SEC> <landing sha>]"""
import hashlib
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                 # <repo>/<package>/code
TREE = Path(os.environ.get("B2_TREE", str(_HERE.parents[4])))
PKGRAW = TREE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw"
SHIP = Path(os.environ.get("B2_SHIP", str(_HERE)))
BASE = os.environ.get("B2_BASE", "A171")
BASES = {"A171": ("gmo_spec_A171.json", "5abd5b907738fc735a0def0c1e7fbb64", "A17.1", 1000, 900),
         "A18": ("gmo_spec_A18.json", "4eda0636f1a41a9b60b59ccb6a4afca3", "A18", 3000, 2700)}
SRC_NAME, SRC_MD5, BASE_ID, STEPS, START = BASES[BASE]
DRAFT = len(sys.argv) < 4
AID, SEC, SPEC_SHA = ((f"BLOCK2_DRAFT" if BASE == "A171" else f"BLOCK2_{BASE}_DRAFT"),
                      "<§ on registration>", "<sha>") if DRAFT else sys.argv[1:4]
src = PKGRAW / SRC_NAME
b0 = src.read_bytes()
assert hashlib.md5(b0).hexdigest() == SRC_MD5, f"not the {BASE_ID} spec as landed"
s = json.loads(b0)
assert s["near_refine_blocks"] == 1 and s["steps"] == STEPS
assert s["lr_decay"] == {"kind": "cosine_to_zero", "start_step": START}
out = {}
for k, v in s.items():
    if k == "registered":
        out[k] = (f"DRAFT -- NOT REGISTERED: the second near refine block arm on the {BASE_ID} "
                  f"protocol, prepared for the PI after an edge FAIL; any number from this file is "
                  f"inadmissible") if DRAFT else (
                  f"2026-09-27: SPEC_REFCV7 {SEC} ({AID}, landed {SPEC_SHA}) -- the second near refine "
                  f"block arm ({BASE_ID} + near_refine_blocks 2), registered BEFORE any {AID}-arm number")
        continue
    if k == "amends":
        out[k] = {"spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                          f"2026-09-26-refcv7-map-hires/raw/{SRC_NAME}",
                  "spec_md5": hashlib.md5(b0).hexdigest(),
                  "amendment": f"Project Steering/SPEC_REFCV7.md {SEC} ({AID})",
                  "what_changes": (f"MAIN = the {BASE_ID} arm with near_refine_blocks 2 (a second "
                                   "dilated 2/4 residual block on the near rows, zero-init last "
                                   "conv, built after the first); must-fails, the decay and every "
                                   "other literal unchanged")}
        continue
    if k == "near_refine_blocks":
        out[k] = 2
        continue
    out[k] = v
dst = PKGRAW / f"gmo_spec_{AID}.json"
b = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
dst.write_bytes(b)
md5 = hashlib.md5(b).hexdigest()
diff = sorted(k for k in set(s) | set(json.loads(b)) if s.get(k) != json.loads(b).get(k))
print(dst.name, md5, f"diff vs {BASE_ID}:", diff)
if DRAFT:
    raise SystemExit(0)
if BASE != "A171":
    raise SystemExit(f"spec written; the {BASE_ID}-protocol early MAIN runs with gmo_a18_main.py "
                     f"(swap its SPEC path + SPEC_MD5 = {md5} and near_refine_blocks 1 -> 2)")
run = (SHIP / "gmo_r5_runner.py").read_bytes().decode("utf-8")
R = [("A171", AID), ("A17.1 lr-decay arm (A15's config: near lift 20 m + 1 near refine ",
                     f"{AID} second-block arm (A17.1 + near_refine_blocks 2: near lift 20 m + 2 near refine "),
     ('SPEC_MD5 = "5abd5b907738fc735a0def0c1e7fbb64"', f'SPEC_MD5 = "{md5}"'),
     ('"--near-refine-blocks", "1",', '"--near-refine-blocks", "2",'),
     ("NONBINDING-EARLY-2ac0bfb+NEW2R5+A171", f"NONBINDING-EARLY-2ac0bfb+NEW2R5+{AID}")]
for old, new in R:
    assert run.count(old) >= 1, old[:60]
    run = run.replace(old, new)
(SHIP / f"gmo_{AID.lower()}_runner.py").write_bytes(run.encode("utf-8"))
print("runner written; its launcher = gmo_r5_launch.sh with A171 -> the new id and the runner name")
