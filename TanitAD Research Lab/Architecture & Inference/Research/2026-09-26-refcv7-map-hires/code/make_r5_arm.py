"""Generate the lr-decay arm's spec (in the R5 candidate tree) and runner: A15's config (sqrt_mf
launch weights, near lift 20 m + 1 near refine block) + ``lr_decay`` cosine to zero over the final
10 %; A15's must-fail set unchanged.
Usage: make_r5_arm.py <AID> <SEC> <landing sha> [--start 900]

Run for A17.1 as ``make_r5_arm.py A171 §22.1 2ac0bfb --start 900`` (spec md5 5abd5b90) with
R5_TREE = the candidate (tip 2ac0bfb + NEW-2 R5 blobs). The build then filled the runner's
candidate provenance (base 2ac0bfb, tar 87cecca4, MD5SUMS_r5.txt, launch tag
NONBINDING-EARLY-2ac0bfb+NEW2R5+A171) and wrote code/gmo_r5_launch.sh by hand. This packaged copy
resolves its defaults relative to ITSELF (the repo it sits in; the package's own
code/gmo_r3_runner.py); R5_TREE / R5_SHIP override both.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("aid")
ap.add_argument("sec")
ap.add_argument("sha")
ap.add_argument("--start", type=int, default=900)
a = ap.parse_args()
AID, SEC, SPEC_SHA = a.aid, a.sec, a.sha
_HERE = Path(__file__).resolve().parent                 # <repo>/<package>/code
TREE = Path(os.environ.get("R5_TREE", str(_HERE.parents[4])))
PKGRAW = TREE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw"
SHIP = Path(os.environ.get("R5_SHIP", str(_HERE)))
src = PKGRAW / "gmo_spec_A15.json"
b0 = src.read_bytes()
assert hashlib.md5(b0).hexdigest() == "20929e21a577a6374f31b3f754ddd4d4"
s = json.loads(b0)
out = {}
for k, v in s.items():
    if k == "registered":
        out[k] = (f"2026-09-27: SPEC_REFCV7 {SEC} ({AID}, landed {SPEC_SHA}) -- the NEW-2 R5 lr-decay "
                  f"arm (A15's config + cosine-to-zero over steps {a.start}-{s['steps']}), registered "
                  f"BEFORE any {AID}-arm number")
        continue
    if k == "amends":
        out[k] = {"spec": "TanitAD Research Lab/Architecture & Inference/Research/"
                          "2026-09-26-refcv7-map-hires/raw/gmo_spec_A15.json",
                  "spec_md5": hashlib.md5(b0).hexdigest(),
                  "amendment": f"Project Steering/SPEC_REFCV7.md {SEC} ({AID})",
                  "what_changes": (f"the A15 arm with the lr held at 1e-3 for steps 1-{a.start} and "
                                   f"cosine-decayed to 0 over steps {a.start}-{s['steps']} (the map "
                                   f"twin of A17); every other literal, arm and must-fail unchanged")}
        continue
    if k == "eval_every":
        out[k] = v
        out["lr_decay"] = {"kind": "cosine_to_zero", "start_step": int(a.start)}
        continue
    out[k] = v
dst = PKGRAW / f"gmo_spec_{AID}.json"
b = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
dst.write_bytes(b)
md5 = hashlib.md5(b).hexdigest()
diff = sorted(k for k in set(s) | set(json.loads(b)) if s.get(k) != json.loads(b).get(k))
print(dst.name, md5, "diff vs A15:", diff)

run = (SHIP / "gmo_r3_runner.py").read_bytes().decode("utf-8")
R = [
    ("NEW-2 R3 decoder arm (SPEC_REFCV7 §20, A15)", f"NEW-2 R5 lr-decay arm (SPEC_REFCV7 {SEC}, {AID})"),
    ("Runs the R3 candidate's", "Runs the R5 candidate's"),
    ("REGISTERED A15 spec (raw/gmo_spec_A15.json: the A12 spec + near_refine_blocks 1 +\n"
     "must-fail near_block_zeros)",
     f"REGISTERED {AID} spec (raw/gmo_spec_{AID}.json: the A15 spec + lr_decay cosine_to_zero\n"
     f"from step {a.start})"),
    ("g_map_overfit_A15.EARLY_NONBINDING.json", f"g_map_overfit_{AID}.EARLY_NONBINDING.json"),
    ("Usage: gmo_r3_runner.py", "Usage: gmo_r5_runner.py"),
    ('"2026-09-26-refcv7-map-hires/raw/gmo_spec_A15.json")',
     f'"2026-09-26-refcv7-map-hires/raw/gmo_spec_{AID}.json")'),
    ('A15_MD5 = "20929e21a577a6374f31b3f754ddd4d4"', f'SPEC_MD5 = "{md5}"'),
    ('assert hashlib.md5(SPEC.read_bytes()).hexdigest() == A15_MD5, "the A15 spec is not as registered"',
     f'assert hashlib.md5(SPEC.read_bytes()).hexdigest() == SPEC_MD5, "the {AID} spec is not as registered"'),
    ('WHY = ("early read of the A15 decoder arm (stacked on A12) on candidate <base 2374cd2 + NEW-2 "\n'
     '       "R3 blobs>, not the launch commit")',
     f'WHY = ("early read of the {AID} lr-decay arm (A15\'s config) on candidate <R5 tree>, not the "\n'
     '       "launch commit")'),
    ('"--launch-commit", "NONBINDING-EARLY-2374cd2+NEW2R3",', '"--launch-commit", "NONBINDING-EARLY-NEW2R5",'),
    ('"harness_argv": argv, "spec_md5": A15_MD5,', '"harness_argv": argv, "spec_md5": SPEC_MD5,'),
    ('"runner": "gmo_r3_runner.py (stamps only; the harness is the candidate\'s, unmodified)"}',
     '"runner": "gmo_r5_runner.py (stamps only; the harness is the candidate\'s, unmodified)"}'),
]
for old, new in R:
    assert run.count(old) >= 1, old[:60]
    run = run.replace(old, new)
(SHIP / "gmo_r5_runner.py").write_bytes(run.encode("utf-8"))
print("runner written (fill base_commit / shipped_tar_md5 / manifest before shipping)")
