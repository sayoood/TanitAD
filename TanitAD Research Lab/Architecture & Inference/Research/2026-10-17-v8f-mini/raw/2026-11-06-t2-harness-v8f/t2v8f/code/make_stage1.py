"""Builds the stage-1 package for box 2: ONE tar (`t2v8f_stage1.tar`, entries under `pkg/`) plus its `.sha256`. Dev box, CPU. Nothing is committed or staged.

    python -I make_stage1.py [--stage-dir C:/.../stage1_pkg] [--out-dir <t2v8f>/stage1] [--blind-src <blind_mean.pt>]

What it does, in order (every step asserts; a failed assertion stops the build, no tar is written):
  1 stage  code/ (7 files), tree/ (the tip tree, 479 files), v9/ (2 pinned files), refs/tool (the T1 tool, sha pinned), refs/blind_mean_eval141.pt, refs/SCREEN_TABLE_2.json,
           refs/scenes/<id>/{camera_front_wide_120fov.mp4, ref_frames.npz, extracted/sequence_tracks.json}, ckpt/<arm>.pt (4), jobs/stage1.cmd
  2 assert the BLIND mean is the eval-141 mean (sha256 of the fp16 tensor bytes = PINS_E4); the zeros stand-in must never ship
  3 assert each mp4 equals the `lfs_sha256` pinned in t2_scenes/raw/RELEASE_LISTING.json (rev 5fe7a149); the 4 checkpoints are 4 distinct files
  4 compute the manifest fields from the scenes' own logs: plumbing starts, gap ticks (+ the pre-extracted real frames), the actor falsifier's lead ticks (state LEAD, DK join,
    position at the tick), the kit manifest sha256
  5 write MANIFEST.json (path, size, sha256 of every other file), then the tar (sorted entries, fixed mtime / owner: the same inputs give the same bytes), then the `.sha256`
  6 report the byte size and the allocated footprint on box 2's 1 MiB-cluster exFAT D:
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
T2 = os.path.dirname(HERE)
MC = os.path.dirname(T2)
KIT = "C:/Users/Admin/t2a_scratch/box2_kit"
SRC_SCENES = "E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release"
TREE_SRC, V9_SRC = "F:/TanitAD-scratch/t2v8f/tree", "F:/TanitAD-scratch/t2v8f/agentA/v9"
TOOL_SRC = f"{MC}/lf6_build/tool/v8f_t1_eval_ri_lf6.py"
TOOL_SHA = "949c4b7a384943015a4b520987bc9c4267dee4a2160f566aa2225208ed3f7f01"
BLIND_DEFAULT = "C:/Users/Admin/AppData/Local/Temp/claude/rh_c1/blind_mean.pt"
BLIND_TENSOR_SHA = "cd6a7ab5e3c5450ec3528889b04e06e5605fea30ea9ff02a0e36d9ba47a2fcb9"           # PINS_E4: sha256 of the fp16 [40,1024] tensor bytes, eval-141 cache
BLIND_ZEROS_SHA = "fa569e2360c540e6280e34a4627516770f1a5f34d81d35689334a99cc1013357"           # the zeros stand-in of the test dir: must never ship
EVAL141_CLIPS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
RELEASE_LISTING = f"{MC}/t2_scenes/raw/RELEASE_LISTING.json"
SCREEN_TABLE = f"{MC}/t2_scenes/raw/SCREEN_TABLE.json"
SCENES = ["00040136-e651-4abd-991d-0655ccda9430", "7c72937c-c620-4776-9555-d57222c0081f"]
ACTOR_SCENE = SCENES[1]
PLUMBING_F0 = {SCENES[0]: 60}                      # MM ruling; 7c72937c: the first of all_starts (= 22)
CODE_FILES = ["t2v8f_run.py", "t2v8f_kit.py", "t2v8f_exec.py", "t2v8f_nocv2.py", "t2v8f_stage1.py", "t2a_v8f_policy.py", "t2a_v8f_encoder.py"]
STRIDE = 3                                         # camera frames (30 Hz) per rig tick (10 Hz)
N_GAP, LEAD_X = 8, (8.0, 30.0)
LEAD_ABS_Y_M = 1.0
MTIME = 1700000000
MIB = 1 << 20


def sha256_file(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def blind_mean_file(src, dst):
    """Copy the BLIND mean after asserting its tensor bytes are the eval-141 mean (rebuilt from the cache when `src` is gone)."""
    import torch
    if not os.path.isfile(src):
        import t2v8f_kit as K
        m, nf, nc = K.blind_mean(EVAL141_CLIPS)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(m, dst)
        got = hashlib.sha256(m.numpy().tobytes()).hexdigest()
        how = f"rebuilt from {EVAL141_CLIPS} ({nf} frames, {nc} clips)"
    else:
        m = torch.load(src, map_location="cpu")
        got = hashlib.sha256(m.numpy().tobytes()).hexdigest()
        put(src, dst)
        how = f"copied from {src}"
    assert got != BLIND_ZEROS_SHA and float(m.float().abs().max()) > 0.0, "REFUSED: the BLIND mean is the zeros stand-in"
    assert got == BLIND_TENSOR_SHA, f"REFUSED: BLIND mean tensor sha256 {got} is not the eval-141 mean {BLIND_TENSOR_SHA}"
    assert tuple(m.shape) == (40, 1024) and str(m.dtype) == "torch.float16", (tuple(m.shape), m.dtype)
    return {"tensor_sha256": got, "shape": [40, 1024], "dtype": "float16", "how": how}


def lead_ticks(rig_json, tracks_json, n_pick=3):
    """The actor falsifier's lead ticks for the scene: DK join state LEAD (the reader's own label, straight-driving gate included) at tick f, the lead's position at the tick in the
    RIG frame at f (x forward, y left; `lead_track_in_window` at t0), kept when LEAD_X[0] <= x <= LEAD_X[1] and |y| <= LEAD_ABS_Y_M, then `n_pick` spread through the candidates."""
    import numpy as np
    import t2v8f_lead as L
    gt, ts = L.load_rig(rig_json)
    assert abs(float(np.median(np.diff(ts))) - 1e5) < 50, "the rig is not 10 Hz"
    seq = L.load_tracks(tracks_json)
    sc = L.SceneLog(gt, ts, seq)
    n = len(ts)
    f0s = np.arange(18, n - 61)
    st = sc.jd(f0s)["state"]
    LS, ego, obs = L.init_lead()["LS"], sc.ego, sc.obs
    cands = []
    for f, s in zip(f0s, st):
        if s != "LEAD":
            continue
        t0 = float(ego["t"][f])
        trk, _, _ = LS.select_lead_causal(obs["t"], obs["track"], obs["center_x"], obs["center_y"], obs["size_x"], obs["is_vehicle"], t0)
        if trk is None:
            continue
        xy = LS.lead_track_in_window(obs["t"], obs["track"], obs["center_x"], obs["center_y"], trk, t0, np.array([t0]), ego["t"], ego["x"], ego["y"], ego["yaw"])[0]
        if np.all(np.isfinite(xy)) and LEAD_X[0] <= xy[0] <= LEAD_X[1] and abs(xy[1]) <= LEAD_ABS_Y_M:
            cands.append((int(f), float(xy[0]), float(xy[1]), str(trk)))
    assert len(cands) >= n_pick + 2, f"only {len(cands)} lead candidates"
    pick = [cands[i] for i in np.round(np.linspace(0, len(cands) - 1, n_pick + 2)).astype(int)[1:-1]]
    return {f: [round(x, 3), round(y, 3)] for f, x, y, _ in pick}, {"n_ticks": n, "n_candidates": len(cands), "candidates_first_last": [cands[0][:3], cands[-1][:3]], "picked_track_ids": [p[3] for p in pick]}


def build(stage, out_dir, blind_src):
    import numpy as np
    import t2v8f_kit as K
    import t2v8f_nocv2 as N
    t_start = time.time()
    assert not os.path.exists(stage) or not os.listdir(stage), f"{stage} must be new and empty (nothing is overwritten)"
    pkg = f"{stage}/pkg"
    prov = {"sources": {}}
    # ---- 1 code, tree, v9, tool, jobs
    for fn in CODE_FILES:
        s = open(f"{HERE}/{fn}", "rb").read()
        assert b"\r" not in s, f"{fn} has CR characters (LF only)"
        put(f"{HERE}/{fn}", f"{pkg}/code/{fn}")
    cmd = open(f"{T2}/stage1_src/jobs/stage1.cmd", "rb").read()
    assert b"\r" not in cmd, "stage1.cmd has CR characters (LF only)"
    put(f"{T2}/stage1_src/jobs/stage1.cmd", f"{pkg}/jobs/stage1.cmd")
    n_tree = 0
    for root, dirs, files in os.walk(TREE_SRC):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if not fn.endswith(".pyc"):
                put(os.path.join(root, fn), f"{pkg}/tree/" + os.path.relpath(os.path.join(root, fn), TREE_SRC).replace("\\", "/"))
                n_tree += 1
    for fn in sorted(os.listdir(V9_SRC)):
        if fn.endswith(".py"):
            put(f"{V9_SRC}/{fn}", f"{pkg}/v9/{fn}")
    assert sha256_file(TOOL_SRC) == TOOL_SHA, "REFUSED: the T1 tool is not the pinned file"
    put(TOOL_SRC, f"{pkg}/refs/tool/v8f_t1_eval_ri_lf6.py")
    prov["tree_files"] = n_tree
    # ---- 2 BLIND mean
    prov["blind_mean"] = blind_mean_file(blind_src, f"{pkg}/refs/blind_mean_eval141.pt")
    # ---- 3 checkpoints
    ck = K.arm_ckpts()
    arms = ["C0_s0", "C0_s1", "DAGOACC_s0", "DAGOACC_s1"]
    assert all(ck[a] for a in arms), f"a checkpoint is absent: { {a: ck[a] for a in arms} }"
    for a in arms:
        put(ck[a], f"{pkg}/ckpt/{a}.pt")
    shas = {a: sha256_file(f"{pkg}/ckpt/{a}.pt") for a in arms}
    assert len(set(shas.values())) == 4, f"two checkpoints are the same file: {shas}"
    prov["ckpt"] = {a: {"src": ck[a], "sha256": shas[a], "size": os.path.getsize(f"{pkg}/ckpt/{a}.pt")} for a in arms}
    # ---- 3 scenes: mp4 vs the pinned listing, tracks, real frames
    listing = json.load(open(RELEASE_LISTING, encoding="utf-8"))
    prov["release_revision"] = listing.get("revision")
    starts = K.all_starts()
    plumbing = dict(PLUMBING_F0)
    plumbing[ACTOR_SCENE] = int(starts[ACTOR_SCENE][0])
    assert plumbing[ACTOR_SCENE] == 22, f"the first start of {ACTOR_SCENE} is {plumbing[ACTOR_SCENE]}, not 22"
    gap_ticks, frames_info, leads, lead_info = {}, {}, None, None
    for sid in SCENES:
        src_dir = f"{SRC_SCENES}/{sid}"
        mp4 = f"{src_dir}/camera_front_wide_120fov.mp4"
        pinned = listing["scenes"][sid]["camera_front_wide_120fov.mp4"]["lfs_sha256"]
        got = sha256_file(mp4)
        assert got == pinned, f"REFUSED: {mp4} sha256 {got} != pinned lfs_sha256 {pinned}"
        put(mp4, f"{pkg}/refs/scenes/{sid}/camera_front_wide_120fov.mp4")
        put(f"{src_dir}/extracted/sequence_tracks.json", f"{pkg}/refs/scenes/{sid}/extracted/sequence_tracks.json")
        import t2v8f_lead as L
        gt, ts = L.load_rig(f"{KIT}/scenes/{sid}/rig_trajectories.json")
        n = len(ts)
        gt_ticks = sorted({int(round(v)) for v in np.linspace(18, n - 61, N_GAP)} | {plumbing[sid]})
        gap_ticks[sid] = gt_ticks
        cam = sorted({STRIDE * t for t in gt_ticks} | {0, 60, 120})                      # the gap job's frames, and the harness falsifier's (it reads camera frames 0, 60, 120)
        frames_info[sid] = N.extract_frames(mp4, cam, f"{pkg}/refs/scenes/{sid}/ref_frames.npz")
        prov["sources"][sid] = {"mp4": mp4, "mp4_sha256": got, "mp4_equals_pinned_lfs_sha256": True, "tracks_sha256": sha256_file(f"{pkg}/refs/scenes/{sid}/extracted/sequence_tracks.json"),
                                "n_ticks": n, "ref_frames": len(cam)}
        if sid == ACTOR_SCENE:
            leads, lead_info = lead_ticks(f"{KIT}/scenes/{sid}/rig_trajectories.json", f"{src_dir}/extracted/sequence_tracks.json")
    # ---- SCREEN_TABLE_2: the rows of the scenes that are in the table (00040136 is not)
    tab = json.load(open(SCREEN_TABLE, encoding="utf-8"))
    rows = [e for e in tab["scenes"] if e["scene"] in SCENES]
    assert [e["scene"] for e in rows] == [ACTOR_SCENE] and rows[0].get("selected"), "SCREEN_TABLE: expected exactly the row of 7c72937c, selected"
    os.makedirs(f"{pkg}/refs", exist_ok=True)
    with open(f"{pkg}/refs/SCREEN_TABLE_2.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "the rows of the 2 stage-1 scenes in t2_scenes/raw/SCREEN_TABLE.json (00040136 is not in the table)", "scenes": rows}, f, indent=1)
    # ---- 5 manifest, tar
    files = []
    for root, dirs, fs in os.walk(pkg):
        dirs.sort()
        for fn in sorted(fs):
            p = os.path.join(root, fn).replace("\\", "/")
            files.append({"path": os.path.relpath(p, pkg).replace("\\", "/"), "size": os.path.getsize(p), "sha256": sha256_file(p)})
    files.sort(key=lambda e: e["path"])
    man = {"package": "t2v8f_stage1", "built": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": "PLUMBING ONLY: stage 1 of the v8F T2 harness is never read for H1 or H2", "scenes": SCENES,
           "files": files, "plumbing_starts": plumbing, "gap_ticks": gap_ticks, "actor_leads": {ACTOR_SCENE: leads}, "stride": STRIDE,
           "lead_selection": {"rule": f"DK join state LEAD at the tick; position at the tick in the rig frame; {LEAD_X[0]} <= x <= {LEAD_X[1]} m, |y| <= {LEAD_ABS_Y_M} m; 3 spread through the candidates",
                              **lead_info},
           "kit_manifest_sha256": sha256_file(f"{KIT}/KIT_MANIFEST.json"), "provenance": prov}
    with open(f"{pkg}/MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    tar_path = f"{out_dir}/t2v8f_stage1.tar"
    assert not os.path.exists(tar_path), f"{tar_path} exists: move it aside first (nothing is overwritten)"
    names = sorted(os.path.relpath(os.path.join(r, f), stage).replace("\\", "/") for r, _, fs in os.walk(pkg) for f in fs)
    with tarfile.open(tar_path + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for nm in names:
            ti = tarfile.TarInfo(nm)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(f"{stage}/{nm}"), MTIME, 0o644, 0, 0, "", ""
            with open(f"{stage}/{nm}", "rb") as fh:
                tf.addfile(ti, fh)
    os.replace(tar_path + ".tmp", tar_path)
    tar_sha = sha256_file(tar_path)
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{tar_sha} *t2v8f_stage1.tar\n")
    # ---- 6 footprint on a 1 MiB-cluster exFAT volume
    all_files = names
    dirs_ = {os.path.dirname(n) for n in all_files}
    dirs_ |= {os.path.dirname(d) for d in dirs_ if os.path.dirname(d)}
    alloc = lambda sizes: sum(max(1, math.ceil(s / MIB)) for s in sizes)                 # noqa: E731
    sizes = [os.path.getsize(f"{stage}/{n}") for n in all_files]
    res = {"tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tar_sha, "n_files": len(all_files), "n_dirs": len(dirs_), "payload_bytes": int(sum(sizes)),
           "extracted_allocated_mib": alloc(sizes) + len(dirs_), "tar_allocated_mib": alloc([os.path.getsize(tar_path)]),
           "kit_install_allocated_mib": alloc([os.path.getsize(f"{stage}/pkg/refs/scenes/{s}/{n}") for s in SCENES for n in ("camera_front_wide_120fov.mp4", "ref_frames.npz", "extracted/sequence_tracks.json")]) + 2,
           "manifest_sha256": sha256_file(f"{pkg}/MANIFEST.json"), "blind_mean": prov["blind_mean"], "seconds": round(time.time() - t_start, 1)}
    res["allocated_mib_tar_plus_extracted_plus_install"] = res["tar_allocated_mib"] + res["extracted_allocated_mib"] + res["kit_install_allocated_mib"]
    with open(f"{out_dir}/BUILD_RESULT.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="C:/Users/Admin/t2a_scratch/stage1_pkg_build")
    ap.add_argument("--out-dir", default=f"{T2}/stage1")
    ap.add_argument("--blind-src", default=BLIND_DEFAULT)
    a = ap.parse_args(argv)
    res = build(a.stage_dir.replace("\\", "/"), a.out_dir.replace("\\", "/"), a.blind_src)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
