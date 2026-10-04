"""Assemble the upload bundle for the CPU smoke (dev box side) and the dev box's own JPEG decode hashes.

Everything in the bundle is either our code (W3's harness driver, E1's wrapper, the REFe seam modules, the Linux
launcher prototypes), the dev box's pin lists, or small slices of data the dev box already scored (the e2chk run:
one navtest log's navsim_logs pickle, its 25 metric-cache pickles, the seam npz, the token file and the landed CSV,
plus 4 camera JPEGs of one of those tokens). ~22 MB; no key, token or credential is in it.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe build_smoke_bundle.py <scratch_dir>
Writes <scratch_dir>/navplan_bundle.tgz and, beside this file, jpeg_devbox_hashes.json + smoke_bundle_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
W3C = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code"
E1 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/"
      "code/navsim_win.py")
DATA = "D:/Projects/TanitAD/data/refe_navtest"
MC = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
NL = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    scratch = sys.argv[1]
    B = os.path.join(scratch, "bundle")
    if os.path.exists(B):
        shutil.rmtree(B)
    toks = json.load(open(f"{DATA}/e2_tokens.json", encoding="utf-8"))
    log = sorted(set(toks["token_log"].values()))
    assert len(log) == 1, log
    log = log[0]
    files = {
        "w3code/run_v1.py": f"{W3C}/run_v1.py", "w3code/navsim_v1_win.py": f"{W3C}/navsim_v1_win.py",
        "w3code/w3_agents_v1.py": f"{W3C}/w3_agents_v1.py", "e1/navsim_win.py": E1,
        "linux_score_v1.py": f"{HERE}/linux_score_v1.py", "navsim_v1_linux.py": f"{HERE}/navsim_v1_linux.py",
        "jpeg_probe.py": f"{HERE}/jpeg_probe.py",
        "seam/refe_e2chk_newvenv.npz": f"{DATA}/seams/refe_e2chk_newvenv.npz",
        "seam/e2_tokens.json": f"{DATA}/e2_tokens.json",
        "seam/refe_e2chk_newvenv.csv": f"{DATA}/score/refe_e2chk_newvenv/refe_e2chk_newvenv.csv",
        f"navsim_logs/{log}.pkl": f"{NL}/{log}.pkl",
        "navsim_crun_venv_freeze.txt": f"{HERE}/navsim_crun_venv_freeze.txt",
        "driverl_eval_venv_freeze.txt": f"{HERE}/driverl_eval_venv_freeze.txt",
        "pkg/eval/refe_navtest_seam.py": f"{PKG}/eval/refe_navtest_seam.py",
        "pkg/code/augment_routes.py": f"{PKG}/code/augment_routes.py",
    }
    for m in ("planner", "model", "ckpt_io", "load_dinov3", "navtrain_scenarios", "calib_table"):
        files[f"pkg/refe/{m}.py"] = f"{PKG}/refe/{m}.py"
    n_mc = 0
    for dp, dn, fns in os.walk(os.path.join(MC, log)):
        if "metric_cache.pkl" in fns and os.path.basename(dp) in toks["tokens"]:
            rel = os.path.relpath(os.path.join(dp, "metric_cache.pkl"), MC).replace("\\", "/")
            files[f"exp/metric_cache_navtest/{rel}"] = os.path.join(dp, "metric_cache.pkl")
            n_mc += 1
    assert n_mc == len(toks["tokens"]) == 25, n_mc
    for dst, src in files.items():
        os.makedirs(os.path.dirname(os.path.join(B, dst)), exist_ok=True)
        shutil.copyfile(src, os.path.join(B, dst))
    # 4 camera JPEGs of one e2 token, and the dev box's own decode+resize hashes (the planner's exact preprocessing)
    rows = {r["token"]: r for r in json.load(open(f"{DATA}/frames/tokens_frames.json", encoding="utf-8"))}
    t0 = sorted(toks["tokens"])[0]
    os.makedirs(os.path.join(B, "frames"), exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(os.path.join(B, "frames", "probe.zip"), "w", zipfile.ZIP_STORED) as zout:
        for rel in rows[t0]["cams"]:
            cam = rel.replace("\\", "/").split("/")[-2]
            base = os.path.basename(rel)
            data = zipfile.ZipFile(f"{DATA}/frames/{rows[t0]['log']}_{cam}.zip").read(base)
            zout.writestr(base, data)
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            arr = np.ascontiguousarray(cv2.resize(im, (960, 512))[:, :, ::-1])
            hashes[base] = hashlib.sha256(arr.tobytes()).hexdigest()
    json.dump({"token": t0, "log": rows[t0]["log"], "opencv": cv2.__version__, "numpy": np.__version__,
               "what": "sha256 of the uint8 array after cv2.imdecode(IMREAD_COLOR) -> cv2.resize((960, 512)) -> "
                       "BGR->RGB, i.e. planner._image_for before the float cast, on the dev box (Windows)",
               "hashes": dict(sorted(hashes.items()))},
              open(os.path.join(HERE, "jpeg_devbox_hashes.json"), "w", encoding="utf-8"), indent=1)
    tgz = os.path.join(scratch, "navplan_bundle.tgz")
    with tarfile.open(tgz, "w:gz") as tf:
        for dst in sorted(os.listdir(B)):
            tf.add(os.path.join(B, dst), arcname=dst)
    man = {"bundle": tgz, "bytes": os.path.getsize(tgz), "sha256": sha(tgz), "n_files": len(files) + 1,
           "files": {k: {"src": v, "bytes": os.path.getsize(v)} for k, v in files.items()}}
    json.dump(man, open(os.path.join(HERE, "smoke_bundle_manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"ZZBUNDLE {tgz} {man['bytes']} B, {man['n_files']} files, sha256 {man['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
