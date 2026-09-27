"""Generate the lever-(4) arm's spec (in the R4 candidate tree), runner and launcher.
Usage: make_r4_arm.py <amendment id, e.g. A16> <spec section, e.g. §21> <landing sha of the SPEC>

Run for A16 as ``make_r4_arm.py A16 §21 879673c`` (spec md5 a4ef45d0). This packaged copy
resolves its defaults relative to ITSELF: the tree is the repo this package sits in, and the
A15 runner / launcher it rewrites are the package's own code/gmo_r3_runner.py and
code/gmo_r3_launch.sh (the build used a scratch copy of the same two files). R4_TREE /
R4_SHIP override both; a dry run points them at a temp copy.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

AID, SEC, SPEC_SHA = sys.argv[1], sys.argv[2], sys.argv[3]
_HERE = Path(__file__).resolve().parent                 # <repo>/<package>/code
TREE = Path(os.environ.get("R4_TREE", str(_HERE.parents[4])))
PKGRAW = TREE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw"
SHIP = Path(os.environ.get("R4_SHIP", str(_HERE)))
src = PKGRAW / "gmo_spec_A15.json"
b0 = src.read_bytes()
assert hashlib.md5(b0).hexdigest() == "20929e21a577a6374f31b3f754ddd4d4"
s = json.loads(b0)
out = {}
for k, v in s.items():
    if k == "registered":
        out[k] = (f"2026-09-27 ~15:20 Berlin: SPEC_REFCV7 {SEC} ({AID}, landed {SPEC_SHA}) -- the NEW-2 R4 weights "
                  f"lever's arm (mf, stacked on A15's near lift + near refine block), registered "
                  f"BEFORE any {AID}-arm number")
        continue
    if k == "amends":
        out[k] = {"spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                          "2026-09-26-refcv7-map-hires/raw/gmo_spec_A15.json",
                  "spec_md5": hashlib.md5(b0).hexdigest(),
                  "amendment": f"Project Steering/SPEC_REFCV7.md {SEC} ({AID})",
                  "what_changes": ("MAIN = the A15 arm (near lift 20 m + 1 near refine block) "
                                   "trained and decided with the TRAIN median-frequency weights "
                                   "(class_weights_definition mf); must-fail: edge_w0 (edge) "
                                   "replaces near_block_zeros; every other literal unchanged")}
        continue
    if k == "class_weights":
        out[k] = v
        out["class_weights_definition"] = "mf"
        continue
    out[k] = v
out["must_fail"] = {"lane_w0": ["lane"],
                    "s8_zeros": ["lane", "crosswalk", "arrow", "edge", "hatched"],
                    "edge_w0": ["edge"]}
dst = PKGRAW / f"gmo_spec_{AID}.json"
b = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
dst.write_bytes(b)
md5 = hashlib.md5(b).hexdigest()
diff = sorted(k for k in set(s) | set(json.loads(b)) if s.get(k) != json.loads(b).get(k))
print(dst.name, md5, "diff vs A15:", diff)

run = (SHIP / "gmo_r3_runner.py").read_bytes().decode("utf-8")
R = [
    ("NEW-2 R3 decoder arm (SPEC_REFCV7 §20, A15)", f"NEW-2 R4 weights arm (SPEC_REFCV7 {SEC}, {AID})"),
    ("Runs the R3 candidate's", "Runs the R4 candidate's"),
    ("REGISTERED A15 spec (raw/gmo_spec_A15.json: the A12 spec + near_refine_blocks 1 +\n"
     "must-fail near_block_zeros)",
     f"REGISTERED {AID} spec (raw/gmo_spec_{AID}.json: the A15 spec + mf weights + must-fail\n"
     "edge_w0)"),
    ("g_map_overfit_A15.EARLY_NONBINDING.json", f"g_map_overfit_{AID}.EARLY_NONBINDING.json"),
    ("Usage: gmo_r3_runner.py", "Usage: gmo_r4_runner.py"),
    ('"2026-09-26-refcv7-map-hires/raw/gmo_spec_A15.json")',
     f'"2026-09-26-refcv7-map-hires/raw/gmo_spec_{AID}.json")'),
    ('A15_MD5 = "20929e21a577a6374f31b3f754ddd4d4"', f'SPEC_MD5 = "{md5}"'),
    ('assert hashlib.md5(SPEC.read_bytes()).hexdigest() == A15_MD5, "the A15 spec is not as registered"',
     f'assert hashlib.md5(SPEC.read_bytes()).hexdigest() == SPEC_MD5, "the {AID} spec is not as registered"'),
    ('WHY = ("early read of the A15 decoder arm (stacked on A12) on candidate <base 2374cd2 + NEW-2 "\n'
     '       "R3 blobs>, not the launch commit")',
     f'WHY = ("early read of the {AID} weights arm (mf, stacked on A15) on candidate <base 1b8170e + "\n'
     '       "NEW-2 R4 blobs>, not the launch commit")'),
    ('"--launch-commit", "NONBINDING-EARLY-2374cd2+NEW2R3",', '"--launch-commit", "NONBINDING-EARLY-1b8170e+NEW2R4",'),
    ('"harness_argv": argv, "spec_md5": A15_MD5,', '"harness_argv": argv, "spec_md5": SPEC_MD5,'),
    ('"candidate": {"base_commit": "2374cd2cc36465d73b467ea955f273d459f7b928",',
     '"candidate": {"base_commit": "1b8170ea1ced89aefc15f7f202b0537cb888d78d",'),
    ('"shipped_tar_md5": "c9dff6147e058dd537be23339464a532",',
     '"shipped_tar_md5": "be33ddacdfc4590956b3901fa35be2de",'),
    ('"per_file_md5_manifest": "MD5SUMS_r3.txt (2,853 files, all OK on Thor)"},',
     f'"per_file_md5_manifest": "MD5SUMS_r4.txt (2,853 files, all OK on Thor) + '
     f'MD5SUMS_r4_post.txt (files shipped after the tar, md5-verified: the harness test '
     f'file and raw/gmo_spec_{AID}.json, whose md5 is also asserted above)"}},'),
    ('"runner": "gmo_r3_runner.py (stamps only; the harness is the candidate\'s, unmodified)"}',
     '"runner": "gmo_r4_runner.py (stamps only; the harness is the candidate\'s, unmodified)"}'),
    (f'dst = OUT / "g_map_overfit_A15.EARLY_NONBINDING.json"', f'dst = OUT / "g_map_overfit_{AID}.EARLY_NONBINDING.json"'),
]
for old, new in R:
    if old.startswith("dst = OUT"):
        continue
    assert run.count(old) >= 1, old[:60]
    run = run.replace(old, new)
(SHIP / "gmo_r4_runner.py").write_bytes(run.encode("utf-8"))
la = (SHIP / "gmo_r3_launch.sh").read_bytes().decode("utf-8")
L = [
    ("The A15 decoder arm's EARLY", f"The {AID} weights arm's EARLY"),
    ("on Thor (SPEC_REFCV7 §20; stacked on\n", f"on Thor (SPEC_REFCV7 {SEC}; the mf weights, stacked on\n"),
    ("writes $OUT/A15_DONE.\n", f"writes $OUT/{AID}_DONE.\n"),
    ("Held while $R/HOLD_A15 exists.\n", f"Held while $R/HOLD_{AID} exists.\n"),
    ("R=/home/nvidia/nb2r3_2374\n", "R=/home/nvidia/nb2r4_1b81\n"),
    ("OUT=$R/gmo_a15\n", f"OUT=$R/gmo_{AID.lower()}\n"),
    ("MARK=${1:-/home/nvidia/nb2r2_cef9/gmo_a12/A12_DONE}\n",
     "MARK=${1:-/home/nvidia/nb2r3_2374/gmo_a15/A15_DONE}\n"),
    ("DONE=$OUT/A15_DONE\n", f"DONE=$OUT/{AID}_DONE\n"),
    ('# the box builder\'s chain (PID 3676134, /home/nvidia/bx_anch_1116/chain4.sh) must have exited\n# too: it does not key on A12_DONE (box builder, ~12:20); explicit PID, read-only\nBOX=3676134\nwhile [ -d /proc/$BOX ]; do sleep 20; done\necho "[a15] box chain $BOX gone at $(date -u +%H:%M:%SZ)"\n',
     '# optional: an explicit PID to outwait (read-only), BOX_PID=<pid>; EMPTY = none. The box\n# builder finished 2026-09-27 ~13:00Z with no Thor process left, so the default waits for\n# nothing. (NOT a default of 1: /proc/1 always exists and the wait would never end.)\nBOX=${BOX_PID:-}\nif [ -n "$BOX" ]; then while [ -d /proc/$BOX ]; do sleep 20; done; fi\necho "[a15] box pid \'$BOX\' (empty = none) gone at $(date -u +%H:%M:%SZ)"\n'),
    ("# A12's near lift). Wakes on a done-marker (arg 1; default: my paused A12 run's A12_DONE, the\n# Master Mind's order: box G-BOX-OVERFIT -> A12 resumes and finishes -> A15). A STOPPED process\n",
     "# A15's near lift + near refine block). Wakes on a done-marker (arg 1; default: the A15 arm's\n# A15_DONE, which exists -- A15 finished 2026-09-27 ~12:28Z). A STOPPED process\n"),
    ("W=/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json",
     "W=/home/nvidia/gmo_early_0327/weights_mf/map_hires_class_weights_train_100x30_MF.json"),
    ('"d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67"',
     '"8ff4fd6d8798031a4af59991833ff724db251618f32c794beb2e5b8575702c98"'),
    ("while [ -e $R/HOLD_A15 ]", f"while [ -e $R/HOLD_{AID} ]"),
    ('nice -n 10 $PY $R/gmo_r3_runner.py', 'nice -n 10 $PY $R/gmo_r4_runner.py'),
    ("  --arms healthy,s8_zeros,near_block_zeros,lane_w0,s8_detached",
     "  --arms healthy,s8_zeros,edge_w0,lane_w0,s8_detached"),
    ('  "$OUT/g_map_overfit_A15.EARLY_NONBINDING.json" > "$DONE"',
     f'  "$OUT/g_map_overfit_{AID}.EARLY_NONBINDING.json" > "$DONE"'),
]
for old, new in L:
    assert la.count(old) >= 1, old[:60]
    la = la.replace(old, new)
la = la.replace("[a15]", f"[{AID.lower()}]").replace("A15 arm start", f"{AID} arm start")
(SHIP / "gmo_r4_launch.sh").write_bytes(la.encode("utf-8"))
print("runner + launcher written; spec md5", md5)
