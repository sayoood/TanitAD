"""RALIGN fit (PREREG_T2_RALIGN.md section 2 and 3): the cross-fitted per-cell affine map from render tokens to real-frame tokens. CPU only, dev box.

    python -I ralign_fit.py fit --tokens-dir <GAP48 pull>/tokens --out-dir <new dir> [--scene-list gap48/SCENES48.json] [--fold-digit 0|1] [--expect-scenes 48]
                                [--scene-key scene --render-key render --real-key real]
    python -I ralign_fit.py identity --out-dir <new dir> [--shape 40,1024]          (the G-IDMAP pair: a := 1, b := 0 on both folds; no tokens are read)

`fit` reads every `*.pt` in --tokens-dir ONCE (a torch dict with the scene id, `render` and `real` as [n, 40, 1024] fp16/fp32), assigns each scene to fold A or B by the first hex digit of
sha256(scene id) (even -> A, odd -> B; --fold-digit 1 is the section 6.1 replicate), accumulates the per-cell OLS statistics per fold and solves a, b per (position, channel).
Writes into the NEW --out-dir (nothing is overwritten): MAP_A.pt (fitted on fold A's pairs), MAP_B.pt, MAPS.json (shas, fold lists, status), FIT_RESULT.json.
The map APPLIED to a scene is the other fold's (`ralign_lib.other_fold`); this tool only fits.
Outcome NOT-EVALUABLE (exit 3): a fold has a cell with fewer than 200 pairs, or a fold is empty. No map file is written then.
Refusals (exit 4, named): RALIGN_IDENTITY_SCENE (a scene whose render tokens equal its real tokens bit for bit: the GAP48 harness encoded one frame twice), RALIGN_DUPLICATE_SCENE,
RALIGN_SCENE_COUNT, RALIGN_SCENE_LIST_MISMATCH, RALIGN_FILE_NAME (file stem is not the first 8 characters of its scene id), non-finite tokens, wrong shapes.
Reads no ego state, no label: the inputs are two token tensors per scene.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch  # noqa: E402

import ralign_lib as L  # noqa: E402

EXIT_OK, EXIT_NOT_EVALUABLE, EXIT_REFUSED = 0, 3, 4


def list_token_files(tokens_dir):
    fs = sorted(glob.glob(os.path.join(tokens_dir, "*.pt")))
    if not fs:
        raise ValueError(f"RALIGN_NO_TOKEN_FILES none in {tokens_dir}")
    return [f.replace("\\", "/") for f in fs]


def fit(tokens_dir, out_dir, *, digit=0, expect_scenes=48, scene_list=None, keys=None, shape=L.SHAPE, min_pairs=L.MIN_PAIRS, now=None):
    """-> (result dict, exit code). Raises ValueError(<NAME> ...) on a refusal; NOT-EVALUABLE is a RESULT (code 3), not an exception."""
    keys = keys or {}
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError(f"RALIGN_OUT_NOT_EMPTY {out_dir} (nothing is overwritten: move it aside)")
    files = list_token_files(tokens_dir)
    stats = {f: L.Stats(shape) for f in L.FOLDS}
    scenes, src = {f: [] for f in L.FOLDS}, []
    seen = set()
    for p in files:
        sid, R, X = L.load_pair_file(p, shape=shape, **keys)
        if os.path.splitext(os.path.basename(p))[0] != sid[:8]:
            raise ValueError(f"RALIGN_FILE_NAME {os.path.basename(p)} holds scene {sid!r}: the file stem must be its first 8 characters")
        if sid in seen:
            raise ValueError(f"RALIGN_DUPLICATE_SCENE {sid}")
        seen.add(sid)
        if not (bool(torch.isfinite(R).all()) and bool(torch.isfinite(X).all())):
            raise ValueError(f"RALIGN_NONFINITE_PAIRS scene {sid}")
        if torch.equal(R, X):
            raise ValueError(f"RALIGN_IDENTITY_SCENE {sid}: render tokens equal real tokens on every tick (the harness encoded one frame twice); fitting it would teach the identity")
        f = L.fold_of(sid, digit)
        stats[f].add(R, X)
        scenes[f].append(sid)
        src.append({"scene": sid, "file": os.path.basename(p), "sha256": L.sha256_file(p), "n_pairs": int(R.shape[0]), "fold": f})
    if expect_scenes is not None and len(seen) != expect_scenes:
        raise ValueError(f"RALIGN_SCENE_COUNT {len(seen)} scenes read, expected {expect_scenes}")
    if scene_list is not None:
        import json
        want = {s["scene"] for s in json.load(open(scene_list, encoding="utf-8"))["scenes"]}
        if want != seen:
            raise ValueError(f"RALIGN_SCENE_LIST_MISMATCH missing {sorted(want - seen)[:3]} extra {sorted(seen - want)[:3]} against {os.path.basename(scene_list)}")
    res = {"card": "PREREG_T2_RALIGN.md", "card_sha256": L.CARD_SHA256, "tool": "ralign_fit.py", "tool_sha256": L.sha256_file(os.path.abspath(__file__)), "lib_sha256": L.sha256_file(L.__file__),
           "built": now or time.strftime("%Y-%m-%d %H:%M:%S"), "tokens_dir": tokens_dir, "fold_digit": digit, "min_pairs": min_pairs, "shape": list(shape),
           "n_scenes": len(seen), "folds": {f: {"scenes": sorted(scenes[f]), "n_scenes": len(scenes[f]), "n_pairs": stats[f].n_pairs} for f in L.FOLDS}, "sources": src, "maps": {}}
    sol, why = {}, []
    for f in L.FOLDS:
        try:
            if not scenes[f]:
                raise ValueError(f"RALIGN_NOT_EVALUABLE fold {f} has no scene")
            sol[f] = L.solve(stats[f], min_pairs)
            res["folds"][f]["ols"] = sol[f][2]
        except ValueError as e:
            if not str(e).startswith("RALIGN_NOT_EVALUABLE"):
                raise
            why.append(f"fold {f}: {e}")
    if why:
        res.update(status="NOT-EVALUABLE", reasons=why)
        os.makedirs(out_dir, exist_ok=True)
        L.jdump(res, f"{out_dir}/FIT_RESULT.json")
        return res, EXIT_NOT_EVALUABLE
    os.makedirs(out_dir, exist_ok=True)
    for f in L.FOLDS:
        a, b, info = sol[f]
        meta = {"fold": f, "fold_scenes": sorted(scenes[f]), "n_pairs": stats[f].n_pairs, "fold_digit": digit, "card_sha256": L.CARD_SHA256, "min_count": info["min_count"],
                "n_degenerate": info["n_degenerate"], "status": "FITTED"}
        shas = L.save_map(f"{out_dir}/MAP_{f}.pt", a, b, meta)
        res["maps"][f] = {"file": f"MAP_{f}.pt", **shas, "fold_scenes": sorted(scenes[f]), "n_pairs": stats[f].n_pairs}
    res["status"] = "FITTED"
    L.jdump(res, f"{out_dir}/FIT_RESULT.json")
    L.jdump({"card_sha256": L.CARD_SHA256, "status": "FITTED", "fold_digit": digit, "shape": list(shape), "maps": res["maps"], "applied_map_for_scene_in_fold": {"A": "B", "B": "A"}}, f"{out_dir}/MAPS.json")
    return res, EXIT_OK


def write_identity(out_dir, shape=L.SHAPE, digit=0):
    """The G-IDMAP pair: a := 1, b := 0 (b stored as -0.0, see `ralign_lib.identity_pair`). Same file layout as a fit; status IDMAP."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError(f"RALIGN_OUT_NOT_EMPTY {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    a, b = L.identity_pair(shape)
    maps = {}
    for f in L.FOLDS:
        shas = L.save_map(f"{out_dir}/MAP_{f}.pt", a, b, {"fold": f, "fold_scenes": [], "n_pairs": 0, "fold_digit": digit, "card_sha256": L.CARD_SHA256, "status": "IDMAP"})
        maps[f] = {"file": f"MAP_{f}.pt", **shas, "fold_scenes": [], "n_pairs": 0}
    idx = {"card_sha256": L.CARD_SHA256, "status": "IDMAP", "fold_digit": digit, "shape": list(shape), "maps": maps, "applied_map_for_scene_in_fold": {"A": "B", "B": "A"}}
    L.jdump(idx, f"{out_dir}/MAPS.json")
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--tokens-dir", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--scene-list", default=None)
    f.add_argument("--fold-digit", type=int, default=0, choices=(0, 1))
    f.add_argument("--expect-scenes", type=int, default=48)
    f.add_argument("--scene-key", default="scene")
    f.add_argument("--render-key", default="render")
    f.add_argument("--real-key", default="real")
    f.add_argument("--shape", default=f"{L.SHAPE[0]},{L.SHAPE[1]}", help="only a test passes another shape")
    i = sub.add_parser("identity")
    i.add_argument("--out-dir", required=True)
    i.add_argument("--shape", default=f"{L.SHAPE[0]},{L.SHAPE[1]}")
    a = ap.parse_args(argv)
    shape = tuple(int(x) for x in a.shape.split(","))
    try:
        if a.cmd == "identity":
            print(f"identity pair written: {write_identity(a.out_dir.replace(chr(92), '/'), shape)['status']}")
            return EXIT_OK
        res, rc = fit(a.tokens_dir.replace("\\", "/"), a.out_dir.replace("\\", "/"), digit=a.fold_digit, expect_scenes=a.expect_scenes, scene_list=a.scene_list, shape=shape,
                      keys={"scene_key": a.scene_key, "render_key": a.render_key, "real_key": a.real_key})
    except ValueError as e:
        print(f"ZZ RALIGN-FIT-REFUSED {e} ZZ")
        return EXIT_REFUSED
    print(f"ZZ RALIGN-FIT {res['status']} folds A/B scenes {res['folds']['A']['n_scenes']}/{res['folds']['B']['n_scenes']} pairs {res['folds']['A']['n_pairs']}/{res['folds']['B']['n_pairs']} ZZ")
    if res["status"] != "FITTED":
        print("\n".join(res["reasons"]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
