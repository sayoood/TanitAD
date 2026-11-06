"""Builds the STAGE-4 package for box 2 (AGENT N): the composite arm C0C over the 62 scenes of stages 2 and 3. Dev box, CPU. Nothing is committed, staged, downloaded or overwritten.

    python -I make_stage4.py build [--out-dir stage4] [--stage-dir C:/.../stage4_pkg_build]

ONE small tar, `t2v8f_stage4_overlay.tar` (entries under `pkg/`, NEW files only; extracted `tar -xf ... -C D:\\t2v8f`):
  pkg/code/t2v8f_c0c.py  pkg/code/t2v8f_stage4.py  pkg/code/t2v8f_stage4_cell.py  pkg/jobs/stage4.cmd  pkg/refs/tool/v8f_t1_eval_ri.py (76,704 B, a BYTE COPY of the T1-kin RI tool, sha256 0417c838...)
  pkg/MANIFEST4.json  = the 62 scenes of MANIFEST2 (stage-2 overlay 2) and MANIFEST3 verbatim (files + sha256 + ref_frames + starts), the overlay's own files, `requires` (the stage-2 files this driver
  imports or spawns, sha256-pinned from overlay 2's own manifest), the RI tool pin, the three scene roots, the footprint.
NO scene file ships: the scenes are on box 2 already (`D:\\t2v8f\\scenes2`, the kit, `C:\\t2v8f\\scenes3`). The build asserts: both source manifests carry the stage-1 manifest sha256, 42 + 20 scenes
(or a fixture's counts), idx 0..41 then 42..61, 81 + 40 starts equal the manifests' own, the RI tool equals its pinned sha256, every text file is LF-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
T2 = os.path.dirname(HERE)
import make_stage2 as MK  # noqa: E402
import make_stage3 as M3  # noqa: E402
import t2v8f_c0c as C0C  # noqa: E402
import t2v8f_stage4 as D4  # noqa: E402

CODE_FILES = ["t2v8f_c0c.py", "t2v8f_stage4.py", "t2v8f_stage4_cell.py"]
OVERLAY2_TAR, OVERLAY3_TAR = M3.OVERLAY2_TAR, f"{T2}/stage3/t2v8f_stage3_overlay.tar"
MIB = 1 << 20


def read_member(tar, member):
    with tarfile.open(tar) as tf:
        b = tf.extractfile(member).read()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def load_real():
    (m2, h2), (m3, h3) = read_member(OVERLAY2_TAR, "pkg/MANIFEST2.json"), read_member(OVERLAY3_TAR, "pkg/MANIFEST3.json")
    return {"m2": m2, "h2": h2, "m3": m3, "h3": h3, "tar2": OVERLAY2_TAR, "tar3": OVERLAY3_TAR}


def footprint4(scenes, tar_bytes, n_pkg_files=5):
    """Box-2 writes per drive, MiB. D: (exFAT, 1 MiB clusters) = the overlay (tar + extracted files) and the PULL, with the stage-2 driver's own per-cell figures (`make_stage2.CELL_MIB`, MEASURED on stage 1's
    pull) so `pass_need_mib` works unchanged. C: = nothing new (the scenes3 root is READ only). The scene files are READ from D: (kit, scenes2) and C: (scenes3) and ALREADY occupy those drives."""
    n = len(scenes)
    n_dec = n_bes = n * 2
    pass_mib = {"deciding": n_dec * MK.CELL_MIB + n * MK.SCENE_DIRS_MIB + MK.FIXED_PULL_MIB, "beside": n_bes * MK.CELL_MIB + 2}
    overlay_d = MK.alloc_mib(tar_bytes) + n_pkg_files
    tot = overlay_d + pass_mib["deciding"] + pass_mib["beside"]
    return {"cells": {"deciding": n_dec, "beside": n_bes, "total": n_dec + n_bes}, "pass_mib": pass_mib, "cell_mib_measured_stage1": MK.CELL_MIB, "floor_gib": MK.FLOOR_GIB,
            "box2_C_new_writes_mib": 0, "box2_D_overlay_mib": overlay_d, "box2_D_pull_deciding_mib": pass_mib["deciding"], "box2_D_pull_beside_mib": pass_mib["beside"], "box2_D_total_mib": tot,
            "D_free_gib_needed_to_run_both_passes": round(MK.FLOOR_GIB + tot / 1024, 2), "D_free_gib_needed_for_deciding_pass_only": round(MK.FLOOR_GIB + (overlay_d + pass_mib["deciding"]) / 1024, 2),
            "note": "ESTIMATED from stage 1's MEASURED per-cell allocation (6 MiB on 1 MiB-cluster exFAT; a C0C record also carries plan_raw / plan_s, a few KB, inside the same cluster: the rehearsal asserts the stub cell "
                    "still costs CELL_MIB). The driver checks the floor on the PULL drive (D:) at preflight and before each pass."}


def build(src, out_dir, stage, n_launch=D4.N_LAUNCH, n_guard=D4.N_GUARD, fixture_tag=None, stage1_dir=None, requires=None, ri_tool=None, cmd_path=None):
    t_start = time.time()
    assert not os.path.exists(stage) or not os.listdir(stage), f"{stage} must be new and empty (nothing is overwritten)"
    m2, m3 = src["m2"], src["m3"]
    s1_dir = stage1_dir or f"{T2}/stage1"
    s1_tar = f"{s1_dir}/t2v8f_stage1.tar"
    s1_sha = open(s1_tar + ".sha256").read().split()[0]
    assert MK.sha256_file(s1_tar) == s1_sha, "the stage-1 tar does not match its .sha256"
    s1_build = json.load(open(f"{s1_dir}/BUILD_RESULT.json", encoding="utf-8"))
    assert s1_build["tar_sha256"] == s1_sha
    assert m2["stage1_manifest_sha256"] == m3["stage1_manifest_sha256"] == s1_build["manifest_sha256"], "MANIFEST2 / MANIFEST3 / stage 1 do not name one package"
    assert m3.get("stage") == 3 and len(m2["scenes"]) == n_launch and len(m3["scenes"]) == n_guard, (len(m2["scenes"]), len(m3["scenes"]))
    pkg = f"{stage}/pkg"
    os.makedirs(f"{pkg}/code", exist_ok=True)
    for fn in CODE_FILES:
        s = open(f"{HERE}/{fn}", "rb").read()
        assert b"\r" not in s, f"{fn} has CR characters (LF only)"
        open(f"{pkg}/code/{fn}", "wb").write(s)
    cmd = open(cmd_path or f"{T2}/stage4_src/jobs/stage4.cmd", "rb").read()
    assert b"\r" not in cmd, "stage4.cmd has CR characters (LF only)"
    os.makedirs(f"{pkg}/jobs", exist_ok=True)
    open(f"{pkg}/jobs/stage4.cmd", "wb").write(cmd)
    ri_src = ri_tool or C0C.RI_TOOL_DEV
    ri_b = open(ri_src, "rb").read()
    assert hashlib.sha256(ri_b).hexdigest() == C0C.RI_TOOL_SHA and len(ri_b) == D4.RI_BYTES, f"the RI tool at {ri_src} is not the pinned byte copy"
    os.makedirs(f"{pkg}/refs/tool", exist_ok=True)
    open(f"{pkg}/{D4.RI_REL}", "wb").write(ri_b)
    scenes = [dict(json.loads(json.dumps(s)), from_stage=2) for s in m2["scenes"]] + [dict(json.loads(json.dumps(s)), from_stage=3) for s in m3["scenes"]]
    assert [s["idx"] for s in scenes] == list(range(n_launch)) + [D4.IDX3_0 + k for k in range(n_guard)], "scene idx are not 0..n2-1 then 42.."
    assert len({s["scene"][:8] for s in scenes}) == len(scenes), "a scene id prefix is not unique"
    n2, n3 = sum(len(s["starts"]) for s in m2["scenes"]), sum(len(s["starts"]) for s in m3["scenes"])
    assert sum(len(s["starts"]) for s in scenes) == n2 + n3
    if fixture_tag is None:
        assert (n2, n3) == (81, 40), (n2, n3)
    ov = []
    for root, dirs, fs in os.walk(pkg):
        dirs.sort()
        for fn in sorted(fs):
            p = os.path.join(root, fn).replace("\\", "/")
            ov.append({"path": os.path.relpath(p, pkg).replace("\\", "/"), "size": os.path.getsize(p), "sha256": MK.sha256_file(p)})
    req = requires if requires is not None else M3.stage2_requires()
    man = {"package": "t2v8f_stage4_overlay", "stage": 4, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "purpose": "STAGE 4: the composite arm C0C (C0 + sg72 + kbox) at T2 (deciding) then RR (beside), 62 scenes, 121 starts (PREREG_T2_CRAWL section 6, outcome 1); overlay on the stage-1 package, nothing replaced",
           "stage1_manifest_sha256": s1_build["manifest_sha256"], "stage1_tar_sha256": s1_sha, "files": ov, "requires": req,
           "ri_tool": {"path": D4.RI_REL, "bytes": len(ri_b), "sha256": C0C.RI_TOOL_SHA, "source": ri_src.replace("\\", "/"), "role": "T1-kin smooth_plan / feasible_plan, loaded by file path and sha256-checked by t2v8f_c0c.load_ri_tool"},
           "scene_roots_box2": {"kit": "D:/t2a/box2_kit/scenes", "scenes2": "D:/t2v8f/scenes2", "scenes3": "C:/t2v8f/scenes3"}, "scenes": scenes,
           "arms": {"deciding": list(C0C.ARMS_C0C), "beside": list(C0C.ARMS_C0C)}, "t2_conds": ["T2", "RR"], "cell_priority": [list(x) for x in D4.DECIDING4], "beside_order": [list(x) for x in D4.BESIDE4],
           "composite": {"smooth": C0C.SMOOTH, "feasible": C0C.FEASIBLE, "ckpt_file": C0C.CKPT_FILE_C0C},
           "source_manifests": {"stage2": {"member": "pkg/MANIFEST2.json", "sha256": src["h2"], "tar": os.path.basename(src["tar2"])},
                                "stage3": {"member": "pkg/MANIFEST3.json", "sha256": src["h3"], "tar": os.path.basename(src["tar3"])}},
           "totals": {"n_scenes": len(scenes), "n_starts": n2 + n3, "n_starts_stage2": n2, "n_starts_stage3": n3, "scenes_by_where": {w: sum(1 for s in scenes if s["where"] == w) for w in ("kit", "scenes2", "scenes3")}},
           "start_classes": {c: sum(s["start_class"].count(c) for s in scenes) for c in ("launch", "moving")}}
    man["footprint"] = footprint4(scenes, 100_000, len(ov) + 1)
    with open(f"{pkg}/MANIFEST4.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    name = f"t2v8f_stage4_overlay{fixture_tag or ''}.tar"
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
    fp = footprint4(scenes, os.path.getsize(tar_path), len(names))
    res = {"tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tar_sha, "n_files": len(names), "members": names, "manifest4_sha256": MK.sha256_file(f"{pkg}/MANIFEST4.json"),
           "n_scenes": len(scenes), "n_starts": n2 + n3, "n_cells": fp["cells"], "footprint": fp, "driver_sha256": MK.sha256_file(f"{pkg}/code/t2v8f_stage4.py"), "cell_sha256": MK.sha256_file(f"{pkg}/code/t2v8f_stage4_cell.py"),
           "c0c_sha256": MK.sha256_file(f"{pkg}/code/t2v8f_c0c.py"), "ri_tool_sha256": C0C.RI_TOOL_SHA, "ri_tool_bytes": len(ri_b), "stage1_manifest_sha256": s1_build["manifest_sha256"],
           "source_manifests": man["source_manifests"], "requires": req, "seconds": round(time.time() - t_start, 1)}
    with open(f"{out_dir}/BUILD_RESULT4{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", default=f"{T2}/stage4")
    b.add_argument("--stage-dir", default="C:/Users/Admin/t2a_scratch/stage4_pkg_build")
    a = ap.parse_args(argv)
    print(json.dumps(build(load_real(), a.out_dir.replace("\\", "/"), a.stage_dir.replace("\\", "/")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
