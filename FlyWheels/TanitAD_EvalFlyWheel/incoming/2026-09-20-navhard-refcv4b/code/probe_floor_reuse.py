#!/usr/bin/env python3
"""Exercise ``--reuse-floors`` against the REAL banked floors run, BEFORE a relaunch depends on it.

⭐ Yesterday's lesson, applied: a guard tested only on fixtures missed a tuple-vs-list mismatch that
the real artifact exposed in seconds. So the identity check is run here on the REAL source run
(``…06e257``) with this run's REAL inputs, and every mutation must go RED:

  POSITIVE  CV and STOP adopt; the adopted scores CSV is byte-identical to the source's.
  M1        wrong devkit sha                          -> REFUSED
  M2        one token DROPPED from the split yaml     -> REFUSED
  M3        one token's STAGE flipped (1 <-> 2)       -> REFUSED
  M4        a patch REMOVED from the patch set         -> REFUSED
  M5        a different agent-input export sha256     -> REFUSED
  M6        a STOP seam with ONE pose perturbed       -> REFUSED

    python probe_floor_reuse.py --src <floors run> --out raw/floor_reuse_probe.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from taniteval.bench.navsim import floor_reuse as FR, profiles as P, seams as SM
    from taniteval.bench import contract as C

    prof = P.SPLITS["navhard_two_stage"]
    y = P.read_split_yaml(prof.name)
    dk = P.devkit_sha_measured()
    patches = P.devkit_patches()
    pre = P.preflight(prof, need_logs=False, import_probe=False)
    src = Path(a.src)
    exp_sha = json.loads((src / "raw" / "export_record.json").read_text(encoding="utf-8"))["sha256"]
    exp_path = json.loads((src / "raw" / "export_record.json").read_text(encoding="utf-8"))["path"]
    doc = json.load(open(exp_path, encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="w7_floor_reuse_"))
    new_stop = SM.make_stop_seam(doc, tmp / "STOP.npz", arm="STOP")

    def fresh_run():
        return C.RunDir("navsim_v2", prof.name, f"probe-{os.urandom(3).hex()}",
                        root=tmp / "results", scratch=True).create()

    def attempt(arm, **over):
        kw = dict(arm=arm, src_run=src, run=fresh_run(), prof=prof, split_yaml=y, devkit_sha=dk["sha"],
                  patches=patches, preflight=pre, export_sha256=exp_sha,
                  new_seam=(new_stop if arm == "STOP" else None), log=lambda *_: None)
        kw.update(over)
        try:
            rep = FR.check_and_adopt(**kw)
            return {"verdict": "ADOPTED", "csv_sha256": hashlib.sha256(Path(rep["csv"]).read_bytes()).hexdigest()}
        except FR.FloorReuseRefused as e:
            return {"verdict": "REFUSED", "why": str(e)[:300]}

    out = {"src": str(src), "devkit": dk, "n_patches": len(patches), "export_sha256": exp_sha,
           "cases": {}}
    for arm in ("CV", "STOP"):
        r = attempt(arm)
        src_sha = hashlib.sha256((src / "scores" / f"{arm}.csv").read_bytes()).hexdigest()
        r["byte_identical_to_source_scores_csv"] = r.get("csv_sha256") == src_sha
        out["cases"][f"POSITIVE_{arm}"] = r

    y_drop = copy.deepcopy(y)
    y_drop["stage_two"] = y_drop["stage_two"][1:]
    y_flip = copy.deepcopy(y)
    t = y_flip["stage_two"].pop(0)
    y_flip["stage_one"].append(t)
    bad_stop = tmp / "STOP_perturbed.npz"
    z = dict(np.load(new_stop, allow_pickle=False))
    z["poses"] = z["poses"].copy()
    z["poses"][0, 0, 0] += 1e-3
    np.savez(bad_stop, **z)
    muts = {
        "M1_wrong_devkit_sha": ("CV", dict(devkit_sha="f" * 40)),
        "M2_token_dropped": ("CV", dict(split_yaml=y_drop)),
        "M3_stage_flipped": ("CV", dict(split_yaml=y_flip)),
        "M4_patch_removed": ("CV", dict(patches=patches[:-2] + patches[-1:])),
        "M5_other_export": ("CV", dict(export_sha256="0" * 64)),
        "M6_stop_seam_one_pose_perturbed": ("STOP", dict(new_seam=bad_stop)),
    }
    for k, (arm, over) in muts.items():
        out["cases"][k] = attempt(arm, **over)
    pos_ok = all(out["cases"][f"POSITIVE_{x}"]["verdict"] == "ADOPTED"
                 and out["cases"][f"POSITIVE_{x}"]["byte_identical_to_source_scores_csv"] for x in ("CV", "STOP"))
    mut_ok = all(out["cases"][k]["verdict"] == "REFUSED" for k in muts)
    out["verdict"] = ("PASS: both floors adopt byte-identically AND all %d mutations are REFUSED" % len(muts)
                      if (pos_ok and mut_ok) else
                      "FAIL: positives %s, mutations all refused %s" % (pos_ok, mut_ok))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["cases"].items():
        print(f"{k:38s} {v['verdict']:8s} {v.get('why', '')[:110]}")
    print(out["verdict"])
    return 0 if (pos_ok and mut_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
