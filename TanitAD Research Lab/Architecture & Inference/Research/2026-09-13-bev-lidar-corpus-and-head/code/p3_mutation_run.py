#!/usr/bin/env python3
"""P3 mutation run - point the WHOLE test suite at deliberately broken artifact directories and
record that it goes RED. A guard that has only ever been seen green has not been shown to work.

  control   : a copy of N real artifacts, unmodified            -> suite must PASS
  mirror    : the same N with every y / azimuth axis flipped     -> suite must FAIL
  cellsize  : the same N with meta_json cartesian.cell_m = 0.25  -> suite must FAIL

Writes `raw/p3_mutation_run.json` (pass/fail counts per directory, failing test names).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\Admin\tanitad-caches\bevhead-20260913\bev_gt")


def copy_set(dst: Path, rows: list[dict], mutate: str | None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for r in rows:
            src = SRC / r["artifact"]
            with np.load(src) as z:
                d = {k: z[k] for k in z.files}
            if mutate == "mirror":
                for k in list(d):
                    if d[k].ndim == 3 and (k.startswith("cart_") or k.startswith("polar")):
                        d[k] = d[k][:, :, ::-1].copy()
            elif mutate == "cellsize":
                meta = json.loads(str(d["meta_json"]))
                meta["cartesian"]["cell_m"] = 0.25
                d["meta_json"] = np.array(json.dumps(meta))
            np.savez_compressed(dst / r["artifact"], **d)
            mf.write(json.dumps(r) + "\n")


def run_suite(bevgt_dir: Path) -> dict:
    env = dict(os.environ, BEVGT_DIR=str(bevgt_dir), PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA",
                        str(HERE / "test_bev_gt_artifact.py")], capture_output=True, text=True,
                       env=env, cwd=str(HERE))
    out = p.stdout + p.stderr
    failed = sorted({ln.split("::")[1].split(" ")[0] for ln in out.splitlines()
                     if ln.startswith("FAILED ") and "::" in ln})
    passed = sorted({ln.split("::")[1].split(" ")[0] for ln in out.splitlines()
                     if ln.startswith("PASSED ") and "::" in ln})
    return {"returncode": p.returncode, "n_passed": len(passed), "n_failed": len(failed),
            "failed": failed, "tail": out.strip().splitlines()[-1] if out.strip() else ""}


def main() -> int:
    rows = {}
    for line in (SRC / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["clip_sha12"]] = r
    ok = [r for r in rows.values() if r.get("ok")][:6]
    res = {"schema": "tanitad.p3_mutation_run/1", "n_artifacts_per_dir": len(ok), "dirs": {}}
    with tempfile.TemporaryDirectory() as td:
        for name, mut in (("control", None), ("mirror", "mirror"), ("cellsize", "cellsize")):
            d = Path(td) / name
            copy_set(d, ok, mut)
            res["dirs"][name] = run_suite(d)
            print(name, json.dumps(res["dirs"][name]), flush=True)
    c, m, s = res["dirs"]["control"], res["dirs"]["mirror"], res["dirs"]["cellsize"]
    res["verdict"] = {
        "control_green": c["n_failed"] == 0 and c["n_passed"] > 0,
        "mirror_red": m["n_failed"] > 0 and "test_orientation_real_boxes_beat_mirror_and_MUTATION_goes_red" in m["failed"],
        "cellsize_red": s["n_failed"] > 0 and "test_every_ok_artifact_verifies_against_literals_and_builder" in s["failed"],
    }
    out = HERE.parent / "raw" / "p3_mutation_run.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["verdict"]))
    return 0 if all(res["verdict"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
