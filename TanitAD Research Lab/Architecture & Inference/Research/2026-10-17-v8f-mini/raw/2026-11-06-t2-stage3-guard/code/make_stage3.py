"""Builds the STAGE-3 package for box 2 (AGENT K): the 20 guard scenes of PREREG_T2_CRAWL section 3. Dev box, CPU. Nothing is committed, staged or overwritten.

    python -I make_stage3.py plan  [--starts stage3/GUARD_STARTS.json] [--out stage3/stage3_plan.json]
    python -I make_stage3.py build [--plan stage3/stage3_plan.json] [--refs stage3/ref_frames/REF_FRAMES_SOURCES3.json] [--out-dir stage3] [--stage-dir C:/.../stage3_pkg_build] [--no-verify-src]
    python -I make_stage3.py resume --remote-sizes FILE [--out stage3/scenes_resume3.sftp]     (FILE: lines `C:\\t2v8f\\scenes3\\<id>\\<rel>|<bytes>` from box 2)

The package is ONE small tar, `t2v8f_stage3_overlay.tar` (entries under `pkg/`, NEW files only; extracted `tar -xf ... -C D:\\t2v8f`): `pkg/code/t2v8f_stage3.py`, `pkg/jobs/stage3.cmd`, `pkg/MANIFEST3.json`.
It replaces nothing: stage 2 is running from `pkg/code/t2v8f_stage2.py` (sha256 33b65b19...) and this package only ADDS files beside it. MANIFEST3 `requires` pins the sha256 of every stage-2 file stage 3 imports
or spawns, taken from overlay 2's own MANIFEST2 and checked against the dev files; the stage-3 preflight refuses if box 2's copy differs.
The 20 scenes' files travel by sftp, file by file, into box 2's `C:\\t2v8f\\scenes3\\<id>\\{volume.nurec, rig_trajectories.json, extracted\\sequence_tracks.json, ref_frames.npz}` (`scenes_put3.sftp`);
MANIFEST3 lists every one with size + sha256 and the preflight verifies each. The mp4 files are NOT shipped.
Every build step asserts: the plan's source files equal their sha256 (unless --no-verify-src), starts equal `t2v8f_kit.all_starts()` and the hand derivation of `make_starts3.derive`, the rig of every scene
has the plan's n_ticks, every npz holds f0 / f60 / f120 as uint8 HxWx3.
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
RAW = f"{os.path.dirname(T2)}/t2_scenes/raw"
import make_overlay2 as MO  # noqa: E402  (npz_facts)
import make_stage2 as MK  # noqa: E402  (sha256_file, alloc_mib, REL, start_class, constants)

CODE_FILES = ["t2v8f_stage3.py"]
REQUIRES = ["code/t2v8f_stage2.py", "code/t2v8f_stage2_cell.py", "code/t2v8f_exec_hold.py"]           # what the stage-3 driver imports or spawns from stage 2's files
NPZ = "ref_frames.npz"
MIB = 1 << 20
NTFS_CLUSTER = 4096                                                                                  # box 2 C: (NTFS, 4 KiB clusters, MM-measured)
BOX2_SCENES = "C:/t2v8f/scenes3"
BOX2_SCENES_SFTP = "/C:/t2v8f/scenes3"
OVERLAY1_TAR = f"{T2}/stage2/t2v8f_stage2_overlay.tar"
OVERLAY2_TAR = f"{T2}/stage2/overlay2/t2v8f_stage2_overlay2.tar"


def ntfs_alloc(size):
    return math.ceil(size / NTFS_CLUSTER) * NTFS_CLUSTER


def footprint3(scenes, tar_bytes, n_pkg_files=3):
    """Box-2 allocation per drive, in MiB. C: = the scenes (NTFS, 4 KiB clusters, files only; directory entries are noise). D: = the overlay (exFAT, 1 MiB clusters: the tar + its extracted files)
    and the PULL, with the stage-2 driver's own per-cell figures (`make_stage2.CELL_MIB`, MEASURED on stage 1's pull) so `pass_need_mib` works unchanged."""
    n = len(scenes)
    c_bytes = sum(ntfs_alloc(f["bytes"]) for s in scenes for f in s["files"]) + sum(ntfs_alloc(s["ref_frames"]["bytes"]) for s in scenes)
    n_dec, n_bes = n * 10, n * 4
    pass_mib = {"deciding": n_dec * MK.CELL_MIB + n * MK.SCENE_DIRS_MIB + MK.FIXED_PULL_MIB, "beside": n_bes * MK.CELL_MIB + 2}
    overlay_d = MK.alloc_mib(tar_bytes) + n_pkg_files
    return {"cells": {"deciding": n_dec, "beside": n_bes, "total": n_dec + n_bes}, "pass_mib": pass_mib, "cell_mib_measured_stage1": MK.CELL_MIB,
            "floor_gib": MK.FLOOR_GIB,
            "box2_C_scenes_mib": round(c_bytes / MIB, 1), "box2_C_scenes_gib": round(c_bytes / 2**30, 3),
            "box2_D_overlay_mib": overlay_d, "box2_D_pull_deciding_mib": pass_mib["deciding"], "box2_D_pull_beside_mib": pass_mib["beside"],
            "box2_D_total_mib": overlay_d + pass_mib["deciding"] + pass_mib["beside"],
            "D_free_gib_needed_to_run_both_passes": round(MK.FLOOR_GIB + (overlay_d + pass_mib["deciding"] + pass_mib["beside"]) / 1024, 2),
            "D_free_gib_needed_for_deciding_pass_only": round(MK.FLOOR_GIB + (overlay_d + pass_mib["deciding"]) / 1024, 2),
            "note": "ESTIMATED from stage 1's MEASURED per-cell allocation (6 MiB on 1 MiB-cluster exFAT) and NTFS 4 KiB clusters; the launch step re-probes both drives. The driver checks the floor on the PULL drive "
                    "(D:) at preflight and again before each pass, never on the scenes drive."}


def plan(starts_path, out):
    """stage3_plan.json (the shape of stage 2's plan): per guard scene the starts (from GUARD_STARTS.json, re-checked against the kit and the hand derivation), the byte budget and the source files."""
    import make_starts3 as MS3
    import t2v8f_kit as K
    gs = json.load(open(starts_path, encoding="utf-8"))
    table = json.load(open(K.SCREEN_TABLE, encoding="utf-8"))["scenes"]
    kit, mine = K.all_starts(table), MS3.derive(table, 1)
    assert [d["scene"] for d in mine] == [s["scene"] for s in gs["scenes"]], "GUARD_STARTS.json lists other scenes than the screening table's guard slots"
    for d, s in zip(mine, gs["scenes"]):
        assert d["starts"] == s["starts"] == kit[d["scene"]], f"{d['slot']}: GUARD_STARTS {s['starts']}, hand derivation {d['starts']}, kit {kit[d['scene']]}"
    assert gs["starts_sha256"] == MS3.starts_sha256([(s["scene"], s["starts"]) for s in gs["scenes"]]), "GUARD_STARTS.json starts_sha256 is stale"
    dm = json.load(open(f"{RAW}/DOWNLOAD_MANIFEST.json", encoding="utf-8"))["files"]
    per = {}
    for v in dm.values():
        per.setdefault(v["scene"], []).append({"member": v["member"], "bytes": v["bytes"], "sha256": v["sha256"], "location": v["location"]})
    out_scenes = []
    for s in gs["scenes"]:
        fl = sorted(per.get(s["scene"], []), key=lambda f: ("volume.nurec", "rig_trajectories.json", "sequence_tracks.json").index(f["member"]))
        assert [f["member"] for f in fl] == ["volume.nurec", "rig_trajectories.json", "sequence_tracks.json"], f"{s['slot']}: download manifest members {[f['member'] for f in fl]}"
        out_scenes.append({"scene": s["scene"], "slot": s["slot"], "idx": s["idx"], "t_c_s": None, "n_ticks": s["n_ticks"], "stride": 3, "starts_raw": s["raw"], "starts": s["starts"], "lead": None,
                           "bytes": sum(f["bytes"] for f in fl), "n_files": len(fl), "files": fl})
    res = {"rule": gs["rule"], "starts_sha256": gs["starts_sha256"], "scenes": out_scenes,
           "totals": {"n_scenes": len(out_scenes), "n_starts": sum(len(s["starts"]) for s in out_scenes), "bytes": sum(s["bytes"] for s in out_scenes), "GiB": sum(s["bytes"] for s in out_scenes) / 2**30}}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res["totals"]


def stage2_requires():
    """{path: {size, sha256}} of the stage-2 files stage 3 relies on, from overlay 2's MANIFEST2 (the one on box 2 after the relaunch), asserted equal to the dev files."""
    with tarfile.open(OVERLAY2_TAR) as tf:
        m2 = json.loads(tf.extractfile("pkg/MANIFEST2.json").read())
    with tarfile.open(OVERLAY1_TAR) as tf:
        m1 = json.loads(tf.extractfile("pkg/MANIFEST2.json").read())
    by = {e["path"]: e for e in m1["files"]}
    by.update({e["path"]: e for e in m2["files"]})
    out = {}
    for p in REQUIRES:
        e = by[p]
        dev = f"{HERE}/{os.path.basename(p)}"
        assert MK.sha256_file(dev) == e["sha256"], f"{p}: the dev file {MK.sha256_file(dev)[:12]} is not the one the stage-2 overlay ships {e['sha256'][:12]}"
        out[p] = {"size": e["size"], "sha256": e["sha256"]}
    return out


def build(plan_path, refs_path, out_dir, stage, verify_src=True, plan_root=None, fixture_tag=None, stage1_dir=None, requires=None):
    import numpy as np  # noqa: F401
    import t2v8f_stage2 as D
    t_start = time.time()
    assert not os.path.exists(stage) or not os.listdir(stage), f"{stage} must be new and empty (nothing is overwritten)"
    plan = json.load(open(plan_path, encoding="utf-8"))
    refs = json.load(open(refs_path, encoding="utf-8"))
    s1_dir = stage1_dir or f"{T2}/stage1"
    s1_tar = f"{s1_dir}/t2v8f_stage1.tar"
    s1_sha = open(s1_tar + ".sha256").read().split()[0]
    assert MK.sha256_file(s1_tar) == s1_sha, "the stage-1 tar does not match its .sha256"
    s1_build = json.load(open(f"{s1_dir}/BUILD_RESULT.json", encoding="utf-8"))
    assert s1_build["tar_sha256"] == s1_sha, "BUILD_RESULT.json is not the current stage-1 tar"
    pkg = f"{stage}/pkg"
    for fn in CODE_FILES:
        s = open(f"{HERE}/{fn}", "rb").read()
        assert b"\r" not in s, f"{fn} has CR characters (LF only)"
        os.makedirs(f"{pkg}/code", exist_ok=True)
        with open(f"{pkg}/code/{fn}", "wb") as f:
            f.write(s)
    cmd = open(f"{T2}/stage3_src/jobs/stage3.cmd", "rb").read()
    assert b"\r" not in cmd, "stage3.cmd has CR characters (LF only)"
    os.makedirs(f"{pkg}/jobs", exist_ok=True)
    with open(f"{pkg}/jobs/stage3.cmd", "wb") as f:
        f.write(cmd)
    starts_kit = None
    if fixture_tag is None:
        import t2v8f_kit as K
        starts_kit = K.all_starts()
    scenes, sources, ids = [], [], set()
    for k, sc in enumerate(plan["scenes"]):
        sid = sc["scene"]
        assert sid[:8] not in ids, f"scene id prefix {sid[:8]} is not unique (the cell directories use it)"
        ids.add(sid[:8])
        assert sc["idx"] == 42 + k and str(sc["slot"]).startswith("guard"), f"{sid[:8]}: idx {sc['idx']} / slot {sc['slot']} (stage-3 scenes are guard_1.. with idx 42..)"
        if starts_kit is not None:
            assert starts_kit[sid] == sc["starts"], f"{sid[:8]}: plan starts {sc['starts']} != t2v8f_kit.all_starts() {starts_kit[sid]}"
        files = []
        for f in sc["files"]:
            src = f["location"] if plan_root is None else f"{plan_root}/{sid}/{MK.REL[f['member']]}"
            assert os.path.getsize(src) == f["bytes"], f"{src}: size {os.path.getsize(src)} != plan {f['bytes']}"
            if verify_src:
                assert MK.sha256_file(src) == f["sha256"], f"{src}: sha256 differs from the plan (the download manifest)"
            files.append({"member": f["member"], "rel": MK.REL[f["member"]], "bytes": f["bytes"], "sha256": f["sha256"]})
            sources.append({"idx": sc["idx"], "scene": sid, "rel": MK.REL[f["member"]], "bytes": f["bytes"], "sha256": f["sha256"], "source": src.replace("\\", "/"), "where": "scenes3"})
        assert sorted(x["member"] for x in files) == sorted(MK.REL), f"{sid[:8]}: expected {sorted(MK.REL)}, got {[x['member'] for x in files]}"
        rig = next(s["source"] for s in sources if s["scene"] == sid and s["rel"] == "rig_trajectories.json")
        gt, ts = D.load_rig(rig)
        assert len(ts) == sc["n_ticks"], f"{sid[:8]}: the rig has {len(ts)} ticks, the plan {sc['n_ticks']}"
        assert all(18 <= f0 and f0 + 60 < len(ts) for f0 in sc["starts"]), f"{sid[:8]}: a start leaves less than 6 s or has less than 18 frames of history"
        cls = [MK.start_class(gt, f0) for f0 in sc["starts"]]
        assert sid in refs, f"{sid[:8]}: no reference-frames source listed"
        p = refs[sid]["path"]
        facts = MO.npz_facts(p)
        nsize, nsha = os.path.getsize(p), MK.sha256_file(p)
        assert nsize == refs[sid]["bytes"] and nsha == refs[sid]["sha256"], f"{sid[:8]}: the npz is not the one the extraction recorded"
        rf = {"rel": NPZ, "bytes": nsize, "sha256": nsha, "shipped_by": "sftp", **facts}
        sources.append({"idx": sc["idx"], "scene": sid, "rel": NPZ, "bytes": nsize, "sha256": nsha, "source": p.replace("\\", "/"), "where": "scenes3"})
        scenes.append({"idx": sc["idx"], "scene": sid, "slot": sc["slot"], "t_c_s": None, "n_ticks": sc["n_ticks"], "starts": sc["starts"], "lead": None, "where": "scenes3", "files": files,
                       "ref_frames": rf, "start_class": [c[0] for c in cls], "start_v0_mps": [c[1] for c in cls], "start_extent_6s_m": [c[2] for c in cls]})
    n_starts = sum(len(s["starts"]) for s in scenes)
    ov = []
    for root, dirs, fs in os.walk(pkg):
        dirs.sort()
        for fn in sorted(fs):
            p = os.path.join(root, fn).replace("\\", "/")
            ov.append({"path": os.path.relpath(p, pkg).replace("\\", "/"), "size": os.path.getsize(p), "sha256": MK.sha256_file(p)})
    req = requires if requires is not None else stage2_requires()
    man = {"package": "t2v8f_stage3_overlay", "stage": 3, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "purpose": "STAGE 3: the 20 guard scenes, deciding T2 rollouts (PREREG_T2_CRAWL section 3); overlay on the stage-1 package, beside stage 2's files (nothing replaced)",
           "stage1_manifest_sha256": s1_build["manifest_sha256"], "stage1_tar_sha256": s1_sha, "files": ov, "requires": req, "scene_root": "scenes3", "scene_root_box2": BOX2_SCENES, "scenes": scenes,
           "arms": {"deciding": list(D.DECIDING), "beside": list(D.BESIDE)}, "t2_conds": ["T2", "RR"], "cell_priority": [list(x) for x in D.CELL_PRIORITY], "beside_order": [list(x) for x in D.BESIDE_ORDER],
           "totals": {"n_scenes": len(scenes), "n_starts": n_starts, "scene_bytes": sum(f["bytes"] for s in scenes for f in s["files"]), "ref_frames_bytes": sum(s["ref_frames"]["bytes"] for s in scenes)},
           "start_classes": {c: sum(s["start_class"].count(c) for s in scenes) for c in ("launch", "moving")}, "starts_sha256": plan.get("starts_sha256"),
           "source_plan": os.path.basename(plan_path), "source_plan_sha256": MK.sha256_file(plan_path), "frames": list(MO.REF_IDX)}
    man["footprint"] = footprint3(scenes, 100_000)
    with open(f"{pkg}/MANIFEST3.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    name = f"t2v8f_stage3_overlay{fixture_tag or ''}.tar"
    tar_path = f"{out_dir}/{name}"
    assert not os.path.exists(tar_path), f"{tar_path} exists: move it aside first (nothing is overwritten)"
    names = sorted(os.path.relpath(os.path.join(r, f), stage).replace("\\", "/") for r, _, fs in os.walk(pkg) for f in fs)
    with tarfile.open(tar_path + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for nm in names:
            ti = tarfile.TarInfo(nm)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(f"{stage}/{nm}"), MK.MTIME, 0o644, 0, 0, "", ""
            with open(f"{stage}/{nm}", "rb") as fh:
                tf.addfile(ti, fh)
    os.replace(tar_path + ".tmp", tar_path)
    tar_sha = MK.sha256_file(tar_path)
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{tar_sha} *{name}\n")
    with open(f"{out_dir}/SCENE_SOURCES3{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sources, f, indent=1)
    write_put_batch3(sources, f"{out_dir}/scenes_put3{fixture_tag or ''}.sftp")
    fp = footprint3(scenes, os.path.getsize(tar_path))
    res = {"tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tar_sha, "n_files": len(names), "manifest3_sha256": MK.sha256_file(f"{pkg}/MANIFEST3.json"), "n_scenes": len(scenes),
           "n_starts": n_starts, "start_classes": man["start_classes"], "starts_sha256": plan.get("starts_sha256"), "scene_bytes": man["totals"]["scene_bytes"], "ref_frames_bytes": man["totals"]["ref_frames_bytes"],
           "n_put_files": len(sources), "footprint": fp, "driver_sha256": MK.sha256_file(f"{pkg}/code/t2v8f_stage3.py"), "requires": req, "source_verified_by_sha256": bool(verify_src),
           "seconds": round(time.time() - t_start, 1)}
    with open(f"{out_dir}/BUILD_RESULT3{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def write_put_batch3(sources, path, remote_sizes=None, root=BOX2_SCENES_SFTP):
    """sftp batch: `-mkdir` lines (a leading `-` ignores `already exists`), then one `put` per file. With `remote_sizes` ({lower-case remote path: bytes}): complete files are skipped, a shorter remote
    file is continued with `reput`, a missing or longer one is `put` again. Small files first inside a scene, the volume last, so an interrupted copy leaves whole scenes."""
    lines, made, n_skip, n_re, n_put = [], set(), 0, 0, 0
    top = "/".join(root.split("/")[:3])                                   # /C:/t2v8f
    for d in (top, root):
        lines.append(f"-mkdir {d}")
    for s in sorted(sources, key=lambda s: (s["idx"], s["rel"] == "volume.nurec")):
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
        f.write("\n".join(lines) + "\n")
    return {"batch": path, "put": n_put, "reput": n_re, "skipped_complete": n_skip}


def resume(remote_sizes_file, sources_file, out):
    sources = json.load(open(sources_file, encoding="utf-8"))
    rs = {}
    for line in open(remote_sizes_file, encoding="utf-8-sig"):
        if "|" in line:
            p, n = line.strip().rsplit("|", 1)
            p = p.replace("\\", "/")
            rs["/" + p.lower() if not p.startswith("/") else p.lower()] = int(n)
    return write_put_batch3(sources, out, remote_sizes=rs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--starts", default=f"{T2}/stage3/GUARD_STARTS.json")
    p.add_argument("--out", default=f"{T2}/stage3/stage3_plan.json")
    b = sub.add_parser("build")
    b.add_argument("--plan", default=f"{T2}/stage3/stage3_plan.json")
    b.add_argument("--refs", default=f"{T2}/stage3/ref_frames/REF_FRAMES_SOURCES3.json")
    b.add_argument("--out-dir", default=f"{T2}/stage3")
    b.add_argument("--stage-dir", default="C:/Users/Admin/t2a_scratch/stage3_pkg_build")
    b.add_argument("--no-verify-src", action="store_true")
    r = sub.add_parser("resume")
    r.add_argument("--remote-sizes", required=True)
    r.add_argument("--sources", default=f"{T2}/stage3/SCENE_SOURCES3.json")
    r.add_argument("--out", default=f"{T2}/stage3/scenes_resume3.sftp")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        res = plan(a.starts, a.out.replace("\\", "/"))
    elif a.cmd == "build":
        res = build(a.plan, a.refs, a.out_dir.replace("\\", "/"), a.stage_dir.replace("\\", "/"), verify_src=not a.no_verify_src)
    else:
        res = resume(a.remote_sizes, a.sources, a.out.replace("\\", "/"))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
