"""Dev-box scorer for the card's full 62-scene read (AGENT K): pools the STAGE-2 rollouts (42 launch scenes, idx 0..41) and the STAGE-3 rollouts (20 guard scenes, idx 42..61), per START CLASS.

    python -I t2v8f_stage3_score.py --root2 PULL2/rollouts --root3 PULL3/rollouts --sources2 stage2/SCENE_SOURCES.json --sources3 stage3/SCENE_SOURCES3.json --out-dir stage3/score [--pool-dir DIR]

Card section 3: M1 is read over the LAUNCH scenes; the four families are read over ALL 62. So two products, never mixed:
  * `M1_LAUNCH42.json/.md`  = `t2v8f_stage2_score.score` on the stage-2 root ALONE (the 42 launch scenes): the card's M1 contrast X1 / X2 and the lateral rows per start class.
  * `FAMILIES_ALL62.json/.md` = the same scorer on the POOL of both roots: every row per start class (launch / moving), never pooled across the classes. Its `*_POOLED.json` sibling is the reader's
    own all-start mean and is NOT a card quantity (it mixes the classes and, here, the scene sets).
The pool is a COPY of the small per-cell files (records.jsonl, scene_<idx>.npz) under one directory. It REFUSES a clash: the same cell path in both roots, or the same scene idx in both (a
stage-3 idx must be 42..61 and a stage-2 idx 0..41), so a scene can never be counted twice and an idx never means two scenes. Evidence class of anything this writes: MEASURED (ours, T2 rollouts).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import t2v8f_stage2_score as SC2  # noqa: E402

N_LAUNCH_SCENES, N_GUARD_SCENES = 42, 20


def cells(root):
    """{relative cell path: [absolute files to copy]} for every cell dir under root that has a records.jsonl."""
    out = {}
    for f in sorted(glob.glob(f"{root}/**/records.jsonl", recursive=True)):
        d = os.path.dirname(f).replace("\\", "/")
        out[os.path.relpath(d, root).replace("\\", "/")] = [f.replace("\\", "/")] + sorted(g.replace("\\", "/") for g in glob.glob(f"{d}/scene_*.npz"))
    return out


def scene_idx_of(files):
    return {int(os.path.basename(z)[6:-4]) for f in files for z in [f] if os.path.basename(z).startswith("scene_") and z.endswith(".npz")}


def pool(root2, root3, dest):
    """Copy both roots' per-cell files under `dest`. -> {"n_cells", "idx2", "idx3"}. Raises ValueError on a clash (same cell path, or an idx in both roots, or an idx outside its stage's range)."""
    c2, c3 = cells(root2), cells(root3)
    both = sorted(set(c2) & set(c3))
    if both:
        raise ValueError(f"POOL_CLASH the same cell path is in both roots: {both[:3]}")
    i2 = {i for fs in c2.values() for i in scene_idx_of(fs)}
    i3 = {i for fs in c3.values() for i in scene_idx_of(fs)}
    if i2 & i3:
        raise ValueError(f"POOL_CLASH scene idx in both roots: {sorted(i2 & i3)}")
    if any(not 0 <= i < N_LAUNCH_SCENES for i in i2) or any(not N_LAUNCH_SCENES <= i < N_LAUNCH_SCENES + N_GUARD_SCENES for i in i3):
        raise ValueError(f"POOL_RANGE stage-2 idx must be 0..{N_LAUNCH_SCENES - 1} and stage-3 idx {N_LAUNCH_SCENES}..{N_LAUNCH_SCENES + N_GUARD_SCENES - 1}: got {sorted(i2)[:3]}.. / {sorted(i3)[:3]}..")
    n = 0
    for tag, cs in (("s2", c2), ("s3", c3)):
        for rel, files in cs.items():
            d = f"{dest}/{tag}/{rel}"
            os.makedirs(d, exist_ok=True)
            for f in files:
                shutil.copyfile(f, f"{d}/{os.path.basename(f)}")
            n += 1
    return {"n_cells": n, "n_cells_stage2": len(c2), "n_cells_stage3": len(c3), "idx2": sorted(i2), "idx3": sorted(i3)}


def tracks_args(*sources):
    """`IDX=path` for every scene's sequence_tracks.json in the given SCENE_SOURCES files."""
    out = []
    for s in filter(None, sources):
        for e in json.load(open(s, encoding="utf-8")):
            if e["rel"] == "extracted/sequence_tracks.json":
                out.append(f"{int(e['idx'])}={e['source']}")
    return out


def run(root2, root3, sources2, sources3, out_dir, pool_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    pool_dir = pool_dir or f"{out_dir}/pool"
    assert not os.path.exists(pool_dir) or not os.listdir(pool_dir), f"{pool_dir} must be new and empty (nothing is overwritten)"
    info = pool(root2, root3, pool_dir)
    m1 = SC2.score(root2, f"{out_dir}/M1_LAUNCH42.json", sources2, (), f"{out_dir}/M1_LAUNCH42.md")
    fam = SC2.score(pool_dir, f"{out_dir}/FAMILIES_ALL62.json", None, tracks_args(sources2, sources3), f"{out_dir}/FAMILIES_ALL62.md")
    return {"pool": info, "m1_launch42": {"classes": {c: len(g) for c, g in m1["classes"].items()}, "starts": len(m1["starts"]), "checks": m1["checks"]},
            "families_all62": {"classes": {c: len(g) for c, g in fam["classes"].items()}, "starts": len(fam["starts"]), "checks": fam["checks"]}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root2", required=True)
    ap.add_argument("--root3", required=True)
    ap.add_argument("--sources2", default=None, help="stage 2's SCENE_SOURCES.json (tracks for the launch scenes)")
    ap.add_argument("--sources3", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool-dir", default=None)
    a = ap.parse_args(argv)
    r = run(a.root2, a.root3, a.sources2, a.sources3, a.out_dir, a.pool_dir)
    print(json.dumps(r))
    return 0 if r["m1_launch42"]["checks"]["ok"] and r["families_all62"]["checks"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
