"""Builds the stage-2 OVERLAY for box 2: ONE small tar (`t2v8f_stage2_overlay.tar`, entries under `pkg/`, NEW files only) + `.sha256`, the dev-only scene source list, and the sftp batches
that copy the scene folders. Dev box, CPU. Nothing is committed or staged; nothing is overwritten.

    python -I make_stage2.py build  [--plan audit/stage2_plan.json] [--out-dir stage2] [--stage-dir C:/.../stage2_pkg_build] [--no-verify-src]
    python -I make_stage2.py resume --remote-sizes FILE [--out stage2/scenes_resume.sftp]      (FILE: lines `D:\\t2v8f\\scenes2\\<id>\\<rel>|<bytes>` from box 2)

The overlay never replaces a stage-1 file (stage 1's preflight keeps verifying its 471 files): it ADDS code/t2v8f_stage2.py, code/t2v8f_stage2_cell.py, code/t2v8f_exec_hold.py,
jobs/stage2.cmd and MANIFEST2.json (the plan: per scene the starts, start classes and the size + sha256 of every file the renderer and the scorer need).
Scenes travel separately, file by file (no tar of the scenes): 41 scene folders to D:/t2v8f/scenes2/<id>/{volume.nurec, rig_trajectories.json, extracted/sequence_tracks.json}.
7c72937c (a launch scene) is already in box 2's kit from stage 1: it is verified there by sha256 (where = "kit") and not copied again.
Every build step asserts: the plan's source files equal their sha256 (unless --no-verify-src), starts equal `t2v8f_kit.all_starts()`, the rig of every scene has the plan's n_ticks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
T2 = os.path.dirname(HERE)
KIT = "C:/Users/Admin/t2a_scratch/box2_kit"
CODE_FILES = ["t2v8f_stage2.py", "t2v8f_stage2_cell.py", "t2v8f_exec_hold.py"]
MTIME = 1700000000
MIB = 1 << 20
IN_KIT = {"7c72937c-c620-4776-9555-d57222c0081f"}                  # on box 2 since stage 1: verified there, not copied
REL = {"volume.nurec": "volume.nurec", "rig_trajectories.json": "rig_trajectories.json", "sequence_tracks.json": "extracted/sequence_tracks.json"}
CELL_MIB = 6                                                         # MEASURED on the stage-1 pull: cell dir + DONE + actor_attach.json + records.jsonl + rollouts_*.json + scene_N.npz, 1 MiB each
SCENE_DIRS_MIB = 4                                                   # <scene8>, objects, T2, RR (each directory takes one cluster)
FIXED_PULL_MIB = 24                                                  # markers, STAGE2_RESULT.json, GATE_SCENES.json, logs, the rollouts dir
FREE_AT_STAGE1_PREFLIGHT_GIB = 35.58                                 # MEASURED by stage 1's preflight on D:, BEFORE its install (160 MiB) and its pull (201 MiB)
STAGE1_AFTER_PREFLIGHT_MIB = 160 + 201                               # MEASURED allocations (BUILD_RESULT kit_install, stage-1 pull); the gsplat JIT cache is UNVERIFIED and not counted
FLOOR_GIB = 10.0
LAUNCH_V0 = 0.5                                                      # MM ruling 2026-11-06: launch = logged v0 at f0 < 0.5 m/s, moving = >= 0.5 m/s (the audit's stricter "standstill launch" is not used)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def alloc_mib(size):
    return max(1, math.ceil(size / MIB))


def start_class(gt_T, f0):
    """launch / moving for one start, from the LOGGED rig: v0 = the speed the policy is given at f0 (backward difference over the logged rows <= f0, as `set_t0` seeds it); the 6 s
    extent is reported beside. The scorer (`t2v8f_stage2_score.py`) applies the same rule to the rollouts' own logged track and cross-checks it against each record's `v0`."""
    import numpy as np
    p = np.array([[T[0, 3], T[1, 3]] for T in gt_T])
    v0 = float(np.linalg.norm(p[f0] - p[f0 - 1]) / 0.1)
    ext = float(np.linalg.norm(p[f0 + 60] - p[f0]))
    return ("launch" if v0 < LAUNCH_V0 else "moving"), round(v0, 4), round(ext, 3)


def footprint(scenes, tar_bytes):
    """Box-2 allocation on 1 MiB-cluster exFAT, in MiB. scenes: manifest scene dicts (where, files, starts)."""
    copied = [s for s in scenes if s["where"] == "scenes2"]
    scn = sum(alloc_mib(f["bytes"]) for s in copied for f in s["files"]) + 2 * len(copied) + 1
    overlay = alloc_mib(tar_bytes) + len(CODE_FILES) + 2                         # tar + 3 code files + stage2.cmd + MANIFEST2.json
    n = len(scenes)
    n_dec, n_bes = n * 10, n * 4
    pass_mib = {"deciding": n_dec * CELL_MIB + n * SCENE_DIRS_MIB + FIXED_PULL_MIB, "beside": n_bes * CELL_MIB + 2}
    out = {"scenes_copied": len(copied), "scenes_mib": scn, "overlay_mib": overlay, "pass_mib": pass_mib, "cells": {"deciding": n_dec, "beside": n_bes, "total": n_dec + n_bes},
           "cell_mib_measured_stage1": CELL_MIB}
    free0 = FREE_AT_STAGE1_PREFLIGHT_GIB - STAGE1_AFTER_PREFLIGHT_MIB / 1024
    out["free_gib_now_estimated"] = round(free0, 2)
    for tag, extra in (("after_scenes_and_overlay", 0), ("after_deciding", pass_mib["deciding"]), ("after_all", pass_mib["deciding"] + pass_mib["beside"])):
        out[f"free_gib_{tag}"] = round(free0 - (scn + overlay + extra) / 1024, 2)
    out["floor_gib"] = FLOOR_GIB
    out["shortfall_gib_deciding"] = round(max(0.0, FLOOR_GIB - out["free_gib_after_deciding"]), 2)
    out["shortfall_gib_all"] = round(max(0.0, FLOOR_GIB - out["free_gib_after_all"]), 2)
    out["note"] = ("free_gib_now_estimated = stage 1's preflight free (35.58 GiB, MEASURED) minus stage 1's own later writes (install 160 + pull 201 MiB, MEASURED); the gsplat JIT cache "
                   "and anything else written since are UNVERIFIED. The launch step re-probes. The driver refuses to START a pass that would end below the floor.")
    return out


def build(plan_path, out_dir, stage, verify_src=True, plan_root=None, fixture_tag=None):
    import numpy as np  # noqa: F401
    import t2v8f_stage2 as D
    t_start = time.time()
    assert not os.path.exists(stage) or not os.listdir(stage), f"{stage} must be new and empty (nothing is overwritten)"
    plan = json.load(open(plan_path, encoding="utf-8"))
    s1_tar = f"{T2}/stage1/t2v8f_stage1.tar"
    s1_sha = open(s1_tar + ".sha256").read().split()[0]
    assert sha256_file(s1_tar) == s1_sha, "the stage-1 tar does not match its .sha256"
    s1_build = json.load(open(f"{T2}/stage1/BUILD_RESULT.json", encoding="utf-8"))
    assert s1_build["tar_sha256"] == s1_sha, "BUILD_RESULT.json is not the current stage-1 tar"
    pkg = f"{stage}/pkg"
    for fn in CODE_FILES:
        s = open(f"{HERE}/{fn}", "rb").read()
        assert b"\r" not in s, f"{fn} has CR characters (LF only)"
        os.makedirs(f"{pkg}/code", exist_ok=True)
        with open(f"{pkg}/code/{fn}", "wb") as f:
            f.write(s)
    cmd = open(f"{T2}/stage2_src/jobs/stage2.cmd", "rb").read()
    assert b"\r" not in cmd, "stage2.cmd has CR characters (LF only)"
    os.makedirs(f"{pkg}/jobs", exist_ok=True)
    with open(f"{pkg}/jobs/stage2.cmd", "wb") as f:
        f.write(cmd)
    # ---- the scenes: sizes and sha256 of every file the renderer and the scorer need; the sources are verified by hash
    starts_kit = None
    if fixture_tag is None:
        import t2v8f_kit as K
        starts_kit = K.all_starts()
    scenes, sources, ids = [], [], set()
    for idx, sc in enumerate(plan["scenes"]):
        sid = sc["scene"]
        assert sid[:8] not in ids, f"scene id prefix {sid[:8]} is not unique (the cell directories use it)"
        ids.add(sid[:8])
        if starts_kit is not None:
            assert starts_kit[sid] == sc["starts"], f"{sid[:8]}: plan starts {sc['starts']} != t2v8f_kit.all_starts() {starts_kit[sid]}"
        files = []
        for f in sc["files"]:
            src = f["location"] if plan_root is None else f"{plan_root}/{sid}/{REL[f['member']]}"
            assert os.path.getsize(src) == f["bytes"], f"{src}: size {os.path.getsize(src)} != plan {f['bytes']}"
            if verify_src:
                assert sha256_file(src) == f["sha256"], f"{src}: sha256 differs from the plan (the download manifest)"
            files.append({"member": f["member"], "rel": REL[f["member"]], "bytes": f["bytes"], "sha256": f["sha256"]})
            sources.append({"idx": idx, "scene": sid, "rel": REL[f["member"]], "bytes": f["bytes"], "sha256": f["sha256"], "source": src.replace("\\", "/"), "where": "kit" if sid in IN_KIT else "scenes2"})
        assert sorted(x["member"] for x in files) == sorted(REL), f"{sid[:8]}: expected {sorted(REL)}, got {[x['member'] for x in files]}"
        rig = next(s["source"] for s in sources if s["scene"] == sid and s["rel"] == "rig_trajectories.json")
        gt, ts = D.load_rig(rig)
        assert len(ts) == sc["n_ticks"], f"{sid[:8]}: the rig has {len(ts)} ticks, the plan {sc['n_ticks']}"
        cls = [start_class(gt, f0) for f0 in sc["starts"]]
        scenes.append({"idx": idx, "scene": sid, "slot": sc.get("slot"), "t_c_s": sc.get("t_c_s"), "n_ticks": sc["n_ticks"], "starts": sc["starts"], "lead": sc.get("lead"),
                       "where": "kit" if sid in IN_KIT else "scenes2", "files": files, "start_class": [c[0] for c in cls], "start_v0_mps": [c[1] for c in cls], "start_extent_6s_m": [c[2] for c in cls]})
    n_starts = sum(len(s["starts"]) for s in scenes)
    # ---- the manifest of the overlay's own files, then MANIFEST2.json
    ov = []
    for root, dirs, fs in os.walk(pkg):
        dirs.sort()
        for fn in sorted(fs):
            p = os.path.join(root, fn).replace("\\", "/")
            ov.append({"path": os.path.relpath(p, pkg).replace("\\", "/"), "size": os.path.getsize(p), "sha256": sha256_file(p)})
    man = {"package": "t2v8f_stage2_overlay", "built": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": "STAGE 2: the deciding T2 rollouts (PREREG_T2_CRAWL sections 3 and 5); overlay on the stage-1 package",
           "stage1_manifest_sha256": s1_build["manifest_sha256"], "stage1_tar_sha256": s1_sha, "files": ov, "scene_root": "scenes2", "scenes": scenes,
           "arms": {"deciding": list(D.DECIDING), "beside": list(D.BESIDE)}, "t2_conds": ["T2", "RR"], "cell_priority": [list(x) for x in D.CELL_PRIORITY], "beside_order": [list(x) for x in D.BESIDE_ORDER],
           "totals": {"n_scenes": len(scenes), "n_starts": n_starts, "scene_bytes": sum(f["bytes"] for s in scenes for f in s["files"]), "copied_bytes": sum(f["bytes"] for s in scenes if s["where"] == "scenes2" for f in s["files"])},
           "start_classes": {c: sum(s["start_class"].count(c) for s in scenes) for c in ("launch", "moving")},
           "source_plan": os.path.basename(plan_path), "source_plan_sha256": sha256_file(plan_path)}
    probe = os.path.join(stage, "_probe.tar")
    man["footprint"] = footprint(scenes, 100_000)                                   # the tar is ~100 KB; the exact figure is written to BUILD_RESULT.json after the tar exists
    with open(f"{pkg}/MANIFEST2.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    name = f"t2v8f_stage2_overlay{fixture_tag or ''}.tar"
    tar_path = f"{out_dir}/{name}"
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
        f.write(f"{tar_sha} *{name}\n")
    # ---- dev-only: the scene sources and the sftp batches (one explicit put per file: only what the renderer and the scorer need)
    with open(f"{out_dir}/SCENE_SOURCES{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sources, f, indent=1)
    write_put_batch(sources, f"{out_dir}/scenes_put{fixture_tag or ''}.sftp", resume=False)
    fp = footprint(scenes, os.path.getsize(tar_path))
    res = {"tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tar_sha, "n_files": len(names), "manifest2_sha256": sha256_file(f"{pkg}/MANIFEST2.json"), "n_scenes": len(scenes),
           "n_starts": n_starts, "start_classes": man["start_classes"], "copied_scene_bytes": man["totals"]["copied_bytes"], "footprint": fp, "source_verified_by_sha256": bool(verify_src),
           "seconds": round(time.time() - t_start, 1)}
    with open(f"{out_dir}/BUILD_RESULT{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def write_put_batch(sources, path, resume=False, remote_sizes=None, root="/D:/t2v8f/scenes2"):
    """sftp batch: `-mkdir` lines (a leading `-` ignores `already exists`), then `put` per file. With `remote_sizes` ({norm remote path: bytes}): complete files are skipped, a shorter
    remote file is continued with `reput`, a missing or longer one is `put` again."""
    lines, made, n_skip, n_re, n_put = [], set(), 0, 0, 0
    for s in sources:
        if s["where"] != "scenes2":
            continue
        d = f"{root}/{s['scene']}"
        for dd in (d, f"{d}/extracted") if s["rel"].startswith("extracted/") else (d,):
            if dd not in made:
                made.add(dd)
                lines.append(f"-mkdir {dd}")
        rp = f"{d}/{s['rel']}"
        have = None if remote_sizes is None else remote_sizes.get(rp.lower())
        if have == s["bytes"]:
            n_skip += 1
            continue
        verb = "reput" if (have is not None and have < s["bytes"]) else "put"
        n_re += verb == "reput"
        n_put += verb == "put"
        lines.append(f"{verb} {s['source']} {rp}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("-mkdir /D:/t2v8f/scenes2\n" + "\n".join(lines) + "\n")
    return {"batch": path, "put": n_put, "reput": n_re, "skipped_complete": n_skip}


def resume(remote_sizes_file, sources_file, out):
    sources = json.load(open(sources_file, encoding="utf-8"))
    rs = {}
    for line in open(remote_sizes_file, encoding="utf-8-sig"):
        if "|" in line:
            p, n = line.strip().rsplit("|", 1)
            p = p.replace("\\", "/")
            rs["/" + p.lower() if not p.startswith("/") else p.lower()] = int(n)
    return write_put_batch(sources, out, resume=True, remote_sizes=rs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--plan", default=f"{T2}/audit/stage2_plan.json")
    b.add_argument("--out-dir", default=f"{T2}/stage2")
    b.add_argument("--stage-dir", default="C:/Users/Admin/t2a_scratch/stage2_pkg_build")
    b.add_argument("--no-verify-src", action="store_true")
    r = sub.add_parser("resume")
    r.add_argument("--remote-sizes", required=True)
    r.add_argument("--sources", default=f"{T2}/stage2/SCENE_SOURCES.json")
    r.add_argument("--out", default=f"{T2}/stage2/scenes_resume.sftp")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        res = build(a.plan, a.out_dir.replace("\\", "/"), a.stage_dir.replace("\\", "/"), verify_src=not a.no_verify_src)
    else:
        res = resume(a.remote_sizes, a.sources, a.out.replace("\\", "/"))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
