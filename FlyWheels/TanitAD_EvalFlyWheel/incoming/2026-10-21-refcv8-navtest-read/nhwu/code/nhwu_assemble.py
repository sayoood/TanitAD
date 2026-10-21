#!/usr/bin/env python3
"""nhwu_assemble.py (dev box) -- lay the pulled Thor navhard arms out under the names parse8 reads.

finalize_navhard_arm.py writes ``scores_navhard_thor/<ARM>_THOR/score_<ARM>_THOR__navhard_two_stage.{csv,counts.json}`` +
``score_<ARM>_THOR__navhard_two_stage_wrapper/<ARM>_THOR__navhard_two_stage_final_scores_frame.csv`` (the refcv7 Thor
production layout).  parse8 reads ``scores_navhard/score_<ARM>__navhard_two_stage.{csv,counts.json}`` + the
``score_<ARM>__navhard_two_stage_wrapper/<ARM>__navhard_two_stage_final_scores_frame.csv``.  This copies BYTE-FOR-BYTE
(no parse, no rewrite) and asserts: the source counts.json reads PASS with backend thor, its recorded csv_sha256 equals the
pulled CSV's sha256, and every destination file's sha256 equals its source.  A ``<ARM>.thor.json`` provenance note
(backend, the THOR tag, the policy) is written beside each arm.  Writes --out-json with ``verdict``.
"""
import argparse
import hashlib
import json
import os
import shutil


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--full", required=True, help="pulled run dir holding scores_navhard_thor/ and scores_warmup/")
ap.add_argument("--arms", required=True, help="navhard arms, space separated")
ap.add_argument("--warms", default="", help="warmup arms, space separated (presence + PASS check only)")
ap.add_argument("--out-json", required=True)
a = ap.parse_args()
SFX = "__navhard_two_stage"
src_root = os.path.join(a.full, "scores_navhard_thor")
dst_root = os.path.join(a.full, "scores_navhard")
os.makedirs(dst_root, exist_ok=True)
bad, rec = [], {}
for arm in a.arms.split():
    tag = f"{arm}_THOR"
    sd = os.path.join(src_root, tag)
    pairs = [(os.path.join(sd, f"score_{tag}{SFX}.csv"), os.path.join(dst_root, f"score_{arm}{SFX}.csv")),
             (os.path.join(sd, f"score_{tag}{SFX}.counts.json"), os.path.join(dst_root, f"score_{arm}{SFX}.counts.json")),
             (os.path.join(sd, f"score_{tag}{SFX}_wrapper", f"{tag}{SFX}_final_scores_frame.csv"),
              os.path.join(dst_root, f"score_{arm}{SFX}_wrapper", f"{arm}{SFX}_final_scores_frame.csv"))]
    r = {"files": []}
    rec[arm] = r
    if not all(os.path.isfile(s) for s, _ in pairs):
        bad.append(f"{arm}: missing source {[s for s, _ in pairs if not os.path.isfile(s)]}")
        continue
    c = json.load(open(pairs[1][0], encoding="utf-8"))
    r["counts"] = {k: c.get(k) for k in ("status", "backend", "csv_valid_rows", "seam_sha256", "csv_sha256")}
    if c.get("status") != "PASS" or c.get("backend") != "thor" or c.get("csv_valid_rows") != 5912:
        bad.append(f"{arm}: counts {r['counts']}")
    if c.get("csv_sha256") != sha(pairs[0][0]):
        bad.append(f"{arm}: counts csv_sha256 != pulled CSV")
    for s, d in pairs:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        hs, hd = sha(s), sha(d)
        r["files"].append({"src": s, "dst": d, "sha256": hs, "dst_sha256": hd})
        if hs != hd or len(hs) != 64:
            bad.append(f"{arm}: copy sha mismatch {d}")
    json.dump({"backend": "thor", "label": tag, "arm": arm, "split": "navhard_two_stage", "thor_status": c.get("status"),
               "seam_sha256": c.get("seam_sha256"), "copied_from": tag,
               "fingerprint_fallback": "ACCEPTED (RULING_BACKEND_POLICY.md), atol 1e-9, ulp_reference_navhard_devbox.json",
               "policy": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend/RULING_BACKEND_POLICY.md",
               "cross_backend": None, "floors": "dev-box banks (refcv7 raw/floors) -> every R8-vs-floor pair is MIXED-backend"},
              open(os.path.join(dst_root, f"{arm}.thor.json"), "w", encoding="utf-8"), indent=1)
for arm in a.warms.split():
    p = os.path.join(a.full, "scores_warmup", f"score_{arm}.counts.json")
    c = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    rec[f"warmup/{arm}"] = {k: c.get(k) for k in ("status", "backend", "csv_valid_rows", "seam_sha256")}
    if c.get("status") != "PASS" or c.get("csv_valid_rows") != 220 or not os.path.isfile(p[:-12] + ".csv"):
        bad.append(f"warmup {arm}: {rec[f'warmup/{arm}']}")
json.dump({"verdict": "FAIL" if bad else "PASS", "bad": bad, "arms": rec}, open(a.out_json, "w", encoding="utf-8"), indent=1)
print(json.dumps({"verdict": "FAIL" if bad else "PASS", "bad": bad[:5]}))
