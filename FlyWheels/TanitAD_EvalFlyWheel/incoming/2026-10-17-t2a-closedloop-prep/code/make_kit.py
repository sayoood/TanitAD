"""Assemble the T2-A box-2 kit (code tree + overlay + t2a code [+ assets]) into a directory.

    python make_kit.py <dest> [--assets] [--list]

  --list     print every item with its size and source and COPY NOTHING.
  --assets   also copy the big inputs (checkpoint, 2 scenes, P10 weights, the refcv6-kit data files the refcv7
             loader reads). Without it the kit is CODE ONLY (~30 MB) -- enough for the CPU dry path.

Layout produced:
    <dest>/tree/{stack,taniteval}          the LAUNCH tree fec3a0dc (pruned: no results/, tests/, __pycache__) + overlay
    <dest>/t2a/*.py                        this package's code
    <dest>/ckpt/{ckpt_50400.pt,config.json}
    <dest>/p10/p10_S2.pt
    <dest>/scenes/<scene-id>/{volume.nurec,rig_trajectories.json}
    <dest>/refcv6_eval_kit/data/...        only what ``refcv7_loader.build_model`` reads
    <dest>/KIT_MANIFEST.json               every file, size, md5 (md5 of assets > 100 MB is computed only with --md5)

The code tree is taken from the git object ``fec3a0dc`` (the refcv7 launch commit, the tree the checkpoint was
trained on) -- NOT from the working tree, whose branch lacks ``kinematic_prior.py`` and ``refcv7_loader`` (MEASURED).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

LAUNCH_COMMIT = "fec3a0dccfda7eeaf34b537337370f1659fd2429"
R4_COMMIT = "0fcb1c1c"          # refcv7 R4 restart commit: descendant of the launch commit; the ceiling reaches the EMITTED plan (SPEC_REFCV7 sec. 26.1)
REPO = Path(os.environ.get("T2A_REPO", "E:/Projects/TanitAD"))
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
KITSRC = Path(os.environ.get("REFCV6_KIT_SRC", "D:/refcv6_eval_kit")) / "data"
SCENE_ROOT = Path("E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release")
CKPT_DIR = Path("E:/refcv7_eval_kit/ckpt")
P10_SRC = PKG / "assets" / "p10_S2.pt"
SCENES = {"00040136-e651-4abd-991d-0655ccda9430": ("volume.nurec", "rig_trajectories.json", ""),
          "7c72937c-c620-4776-9555-d57222c0081f": ("volume.nurec", "rig_trajectories.json", "extracted/")}
# the refcv6-kit files the refcv7 loader's MODEL build reads (anchors, nav tau, map class weights; + vis1 if required)
KIT_FILES = ["anchors/refc_anchors_6s_v0cond_alat_117.pt",
             "refcv7/map_hires_class_weights_train_100x30.json",
             "refcv7/nav_compliance_tau_train.json",
             "refcv6_train_eval139_extrinsics.json"]      # the ORIGINAL 4,508-clip lift bank is built from it, then replaced
# MEASURED 2026-10-17: with exactly these 4 files + config.json the model build succeeds (strict, step 50400,
# param_breakdown == config). The 186 MB vis1 sidecar and the 703 MB joins are NOT read by build_model.
SKIP_PARTS = ("/results/", "/tests/", "__pycache__", "/.git/")
OVERLAY = [("stack/experiments/alpasim-gsplat/closedloop_drive.py", "stack/experiments/alpasim-gsplat/closedloop_drive.py"),
           ("taniteval/taniteval/floors_kin.py", "taniteval/taniteval/floors_kin.py")]


def md5(p: Path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


CONTROL = [("models/refc-base/ckpt.pt", Path("E:/thor_archive/2026-09-14/models/refc-base/ckpt.pt")),
           ("models/refc-base/config.json", Path("E:/thor_archive/2026-09-14/models/refc-base/config.json")),
           ("banked/HQ_refc-base_empty.json",
            REPO / "stack/experiments/alpasim-gsplat/results/closedloop-hq-render/rollouts/HQ_refc-base_empty.json")]


def items(with_assets: bool, control: bool = False, r4: bool = False):
    """-> list of (kind, src, dst_rel, size|None)."""
    out = [("code-tree", f"git:{LAUNCH_COMMIT}:stack,taniteval (pruned)", "tree/", None)]
    for s, d in OVERLAY:
        p = REPO / s
        out.append(("overlay", str(p), "tree/" + d, p.stat().st_size if p.exists() else None))
    if r4:
        out.append(("code-tree-r4", f"git:{R4_COMMIT}:stack,taniteval (pruned)", "tree_r4/", None))
        for s_, d in OVERLAY:
            p = REPO / s_
            out.append(("overlay-r4", str(p), "tree_r4/" + d, p.stat().st_size if p.exists() else None))
    for p in sorted(HERE.glob("*.py")) + sorted((HERE / "tests").glob("*.py")):
        out.append(("t2a-code", str(p), f"t2a/{p.relative_to(HERE).as_posix()}", p.stat().st_size))
    for p in sorted((PKG / "assets").glob("*")):                 # p10 weights + the known-value fixture (the kit's tests read them)
        out.append(("t2a-asset", str(p), f"assets/{p.name}", p.stat().st_size))
    if with_assets:
        for f in ("ckpt_50400.pt", "config.json"):
            p = CKPT_DIR / f
            out.append(("ckpt", str(p), f"ckpt/{f}", p.stat().st_size if p.exists() else None))
        out.append(("p10", str(P10_SRC), "p10/p10_S2.pt", P10_SRC.stat().st_size if P10_SRC.exists() else None))
        for sid, (vol, rig, sub) in SCENES.items():
            for f in (vol, rig):
                p = SCENE_ROOT / sid / (sub + f)
                out.append(("scene", str(p), f"scenes/{sid}/{f}", p.stat().st_size if p.exists() else None))
        for f in KIT_FILES:
            p = KITSRC / f
            out.append(("kit-data", str(p), f"refcv6_eval_kit/data/{f}", p.stat().st_size if p.exists() else None))
    if control:
        for rel, p in CONTROL:
            out.append(("control", str(p), rel, p.stat().st_size if p.exists() else None))
    return out


def copy_tree(dest: Path, commit: str = LAUNCH_COMMIT, sub: str = "tree"):
    tmp = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "stack", "taniteval"],
                         capture_output=True, check=True).stdout
    n = 0
    with tarfile.open(fileobj=io.BytesIO(tmp)) as tf:
        for m in tf:
            name = "/" + m.name
            if any(s in name for s in SKIP_PARTS) or m.isdir() and False:
                continue
            tf.extract(m, dest / sub, filter="data")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--assets", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="also the cross-host repro control: refc-base (1.25 GB) + its banked Thor panel (0.6 MB)")
    ap.add_argument("--md5", action="store_true", help="md5 the big assets too (slow)")
    ap.add_argument("--r4", action="store_true",
                    help="also tree_r4/ = git archive of the R4 restart commit 0fcb1c1c + the same overlay (+27 MB): the inference code on "
                         "which the speed ceiling reaches the emitted plan; enables the refcv7_r4code_vmax_log_s0 arm")
    a = ap.parse_args()
    its = items(a.assets or a.control, a.control, a.r4) if a.control and not a.assets else items(a.assets, a.control, a.r4)
    total = sum(s or 0 for _, _, _, s in its)
    if a.list:
        for k, s, d, sz in its:
            print(f"{k:10s} {('MISSING' if sz is None and not k.startswith('code-tree') else (sz if sz is not None else '-')):>14} "
                  f"{d:70s} <- {s}")
        print(f"TOTAL (excluding the pruned code tree) {total/1e6:.1f} MB")
        return 0
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    missing = [(k, s) for k, s, d, sz in its if sz is None and not k.startswith("code-tree")]
    if missing:
        raise SystemExit(f"missing sources: {missing}")
    n = copy_tree(dest)
    n4 = copy_tree(dest, R4_COMMIT, "tree_r4") if a.r4 else 0
    manifest = {"launch_commit": LAUNCH_COMMIT, "tree_files": n, "r4_commit": R4_COMMIT if a.r4 else None, "tree_r4_files": n4, "items": []}
    for k, s, d, sz in its:
        if k in ("code-tree", "code-tree-r4"):
            continue
        dst = dest / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst)
        e = {"kind": k, "dst": d, "size": dst.stat().st_size, "src": s}
        if dst.stat().st_size < 100e6 or a.md5:
            e["md5"] = md5(dst)
        manifest["items"].append(e)
    (dest / "KIT_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"kit at {dest}: {n} tree files + {len(manifest['items'])} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
