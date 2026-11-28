"""Builds the STAGE-5 package for box 2 (AGENT R): ONE composite arm NAME (weights of your choice) over the 62 scenes of stages 2 to 4. Dev box, CPU. Nothing is committed, staged, downloaded or overwritten.

    python -I make_stage5.py build --arm EGAIN --tag a10 --ckpt-s0 <path> --ckpt-s1 <path> [--dose a10] [--out-dir stage5/EGAIN_a10] [--stage-dir C:/.../stage5_pkg_build]

ONE tar `t2v8f_stage5_overlay_<RUN>.tar` (RUN = NAME[_TAG]; entries under `pkg/`, NEW files only; extracted `tar -xf ... -C D:\\t2v8f`):
  pkg/code/t2v8f_stage5.py  pkg/code/t2v8f_stage5_cell.py  pkg/jobs/stage5_<RUN>.cmd  pkg/ckpt/<NAME>_s0.pt  pkg/ckpt/<NAME>_s1.pt  pkg/MANIFEST5.json
  MANIFEST5.json = the 62 scenes of MANIFEST4 verbatim (files + sha256 + ref_frames + starts), the overlay's own files (sha256 of EVERY file, the checkpoints included), `requires` (what stage 2's and
  stage 4's overlays already put on box 2, sha256-pinned), the RI tool pin, the arm table `arm5` (name, tag, dose, ckpt_file, ckpt_sha256, source paths), the footprint.
NO scene file ships (they are on box 2: kit, D:\\t2v8f\\scenes2, C:\\t2v8f\\scenes3). The pull directory, log and job name carry RUN, so two doses of one arm never share a directory (the resume rule would skip
the first dose's DONE cells). The build asserts: MANIFEST4 is the one in the stage-4 tar, 42 + 20 scenes (or a fixture's counts), the RI tool equals its pin, every text file is LF-only, no overlay path
replaces a stage-1/2/3/4 path, the checkpoints are real files and are not the stage-1 checkpoints (a stage-5 arm that is C0's own weights is a rehearsal and says so with `--allow-stage1-ckpt`).
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
import make_stage4 as M4  # noqa: E402
import t2v8f_c0c as C0C  # noqa: E402
import t2v8f_stage4 as D4  # noqa: E402
import t2v8f_stage5 as D5  # noqa: E402

CODE_FILES = ["t2v8f_stage5.py", "t2v8f_stage5_cell.py"]
TAR4 = f"{T2}/stage4/t2v8f_stage4_overlay.tar"
EGAIN_ROOT = "E:/Projects/TanitAD-artifacts/v8f-wspedit"
RUN_RE = D5.NAME_RE


def run_id(name, tag):
    return name + (f"_{tag}" if tag else "")


def cmd_text(name, tag):
    """The box-2 job (stage 4's `stage4.cmd`, with stage 5's driver, manifest and a pull directory that names the run). LF-only, no labels, no goto (see stage4.cmd)."""
    run = run_id(name, tag)
    return ("@echo off\n"
            f"rem T2 re-perception rollout, v8F-mini, STAGE 5 (the composite arm {name}, tag {tag or 'none'}, 62 scenes, 121 starts) on box 2. ONE job, safe mode. "
            f"Launch: powershell -NoProfile -File D:\\tanitad\\launch.ps1 -Job D:\\t2v8f\\pkg\\jobs\\stage5_{run}.cmd -Log D:\\t2v8f\\logs\\stage5_{run}.log\n"
            "rem No labels and no goto on purpose: this file is LF-only, and cmd.exe can mis-read labels in an LF-only batch file.\n"
            "call D:\\tanitad\\env.cmd\n"
            'set "PYTHONPATH="\n'
            'set "VCV=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"\n'
            'if exist "%VCV%" (call "%VCV%" >nul) else (echo ZZNO-VCVARS)\n'
            'set "CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8"\n'
            'set "CUDA_PATH=%CUDA_HOME%"\n'
            'set "PATH=%CUDA_HOME%\\bin;%PATH%"\n'
            'set "TORCH_CUDA_ARCH_LIST=8.9"\n'
            'set "T2A_KIT=D:/t2a/box2_kit"\n'
            'set "T2V8F_RI_TOOL=D:/t2v8f/pkg/refs/tool/v8f_t1_eval_ri.py"\n'
            f"if not exist D:\\t2v8f\\pull5_{run} mkdir D:\\t2v8f\\pull5_{run}\n"
            f"echo ===== STAGE5 {run} START %DATE% %TIME%\n"
            f"%RUNSAFE% %PY% -I -u D:\\t2v8f\\pkg\\code\\t2v8f_stage5.py all --pkg D:/t2v8f/pkg --kit D:/t2a/box2_kit --hf D:/tanitad/hf --pull D:/t2v8f/pull5_{run} "
            "--scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --manifest5 D:/t2v8f/pkg/MANIFEST5.json\n"
            "echo ===== ZZJOB-END %DATE% %TIME% rc=%ERRORLEVEL%\n")


def footprint5(scenes, tar_bytes, file_sizes):
    """Box-2 writes on D: (exFAT, 1 MiB clusters). The per-pass pull figures are stage 4's (`make_stage4.footprint4`, MEASURED per cell on stage 1); the overlay is the tar PLUS every extracted file, each at its
    own allocation (the two checkpoints are ~83 MiB each, so stage 4's '1 MiB per file' would understate it). C: gets no new write."""
    fp = M4.footprint4(scenes, tar_bytes, 0)
    overlay_d = MK.alloc_mib(tar_bytes) + sum(MK.alloc_mib(s) for s in file_sizes)
    tot = overlay_d + fp["pass_mib"]["deciding"] + fp["pass_mib"]["beside"]
    fp.update(box2_D_overlay_mib=overlay_d, box2_D_total_mib=tot, D_free_gib_needed_to_run_both_passes=round(MK.FLOOR_GIB + tot / 1024, 2),
              D_free_gib_needed_for_deciding_pass_only=round(MK.FLOOR_GIB + (overlay_d + fp["pass_mib"]["deciding"]) / 1024, 2),
              note="pull figures ESTIMATED from stage 1's MEASURED per-cell allocation (6 MiB on 1 MiB-cluster exFAT, as stage 4); the overlay figure is the tar plus each extracted file at ceil(size / 1 MiB). "
                   "The driver checks the floor on the PULL drive (D:) at preflight and before each pass.")
    return fp


def read_member(tar, member):
    with tarfile.open(tar) as tf:
        b = tf.extractfile(member).read()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def build(out_dir, stage, name, tag, ckpts, dose=None, src_tar4=TAR4, n_launch=D4.N_LAUNCH, n_guard=D4.N_GUARD, fixture_tag=None, allow_stage1_ckpt=False, stage1_dir=None):
    """ckpts = {0: path, 1: path}. -> the BUILD_RESULT5 dict."""
    t_start = time.time()
    assert RUN_RE.fullmatch(name) and name.upper() not in D5.NAME_RESERVED, f"bad arm name {name!r}"
    assert not tag or RUN_RE.fullmatch(tag), f"bad tag {tag!r}"
    run = run_id(name, tag)
    assert not os.path.exists(stage) or not os.listdir(stage), f"{stage} must be new and empty (nothing is overwritten)"
    m4, h4 = read_member(src_tar4, "pkg/MANIFEST4.json")
    assert m4.get("stage") == 4 and len(m4["scenes"]) == n_launch + n_guard, (m4.get("stage"), len(m4["scenes"]))
    s1_sha = m4["stage1_manifest_sha256"]
    s1_manifest, _ = read_member(f"{stage1_dir or T2 + '/stage1'}/t2v8f_stage1.tar", "pkg/MANIFEST.json")
    taken = {e["path"] for e in s1_manifest["files"]} | {e["path"] for e in m4["files"]} | set(m4["requires"])
    pkg = f"{stage}/pkg"
    names, files = [], []                                                                              # files: (tar member, source path)
    os.makedirs(f"{pkg}/code", exist_ok=True)
    os.makedirs(f"{pkg}/jobs", exist_ok=True)
    for fn in CODE_FILES:
        b = open(f"{HERE}/{fn}", "rb").read()
        assert b"\r" not in b, f"{fn} has CR characters (LF only)"
        open(f"{pkg}/code/{fn}", "wb").write(b)
        files.append((f"code/{fn}", f"{pkg}/code/{fn}"))
    cmd = cmd_text(name, tag).encode()
    assert b"\r" not in cmd
    open(f"{pkg}/jobs/stage5_{run}.cmd", "wb").write(cmd)
    files.append((f"jobs/stage5_{run}.cmd", f"{pkg}/jobs/stage5_{run}.cmd"))
    s1_ck = {e["path"]: e["sha256"] for e in s1_manifest["files"] if e["path"].startswith("ckpt/")}
    arms, ck_file, ck_sha, ck_src = D5.arm_names(name), {}, {}, {}
    for s, arm in enumerate(arms):
        src = ckpts[s].replace("\\", "/")
        assert os.path.isfile(src), f"checkpoint {src} is not a file"
        sha = MK.sha256_file(src)
        assert allow_stage1_ckpt or sha not in s1_ck.values(), f"{src} IS a stage-1 checkpoint (C0's or DAGOACC's own weights): pass allow_stage1_ckpt only for an identity rehearsal"
        fn = f"{arm}.pt"
        ck_file[arm], ck_sha[arm], ck_src[arm] = fn, sha, {"path": src, "bytes": os.path.getsize(src), "sha256": sha}
        files.append((f"ckpt/{fn}", src))
    assert ck_sha[arms[0]] != ck_sha[arms[1]], "the two seeds' checkpoints are the same file"
    ov = [{"path": rel, "size": os.path.getsize(src), "sha256": MK.sha256_file(src)} for rel, src in sorted(files)]
    assert not ({e["path"] for e in ov} & taken), sorted({e["path"] for e in ov} & taken)
    ri_b = open(C0C.RI_TOOL_DEV, "rb").read()
    assert hashlib.sha256(ri_b).hexdigest() == C0C.RI_TOOL_SHA and len(ri_b) == D4.RI_BYTES
    req = {p: {"size": e["size"], "sha256": e["sha256"]} for p, e in m4["requires"].items()}
    by4 = {e["path"]: e for e in m4["files"]}
    for p in D5.REQUIRED_STAGE4:
        req[p] = {"size": by4[p]["size"], "sha256": by4[p]["sha256"]}
    scenes = json.loads(json.dumps(m4["scenes"]))
    n_starts = sum(len(s["starts"]) for s in scenes)
    if fixture_tag is None:
        assert n_starts == 121, n_starts
    P = D5.passes5(name)
    man = {"package": "t2v8f_stage5_overlay", "stage": 5, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "purpose": f"STAGE 5: the composite arm {name} (tag {tag or 'none'}, dose {dose or 'n/a'}) at T2 (deciding) then RR (beside), 62 scenes, 121 starts; overlay on the stage-1 package, nothing replaced",
           "stage1_manifest_sha256": s1_sha, "stage1_tar_sha256": m4["stage1_tar_sha256"], "files": ov, "requires": req,
           "ri_tool": m4["ri_tool"], "scene_roots_box2": m4["scene_roots_box2"], "scenes": scenes,
           "arm5": {"name": name, "tag": tag, "dose": dose, "run": run, "arms": list(arms), "encoder": {"kind": D5.ENCODER_FROZEN, "note": "t2v8f_run.build_arm's InLoopEncoder: frozen DINOv3-L, bf16, as stages 2 to 4"}, "ckpt_file": ck_file, "ckpt_sha256": ck_sha, "ckpt_source": ck_src,
                    "composite": {"smooth": C0C.SMOOTH, "feasible": C0C.FEASIBLE, "same_as": "stage 4 C0C (t2v8f_c0c.CompositePolicy); only the weights differ"}},
           "arms": {"deciding": list(arms), "beside": list(arms)}, "t2_conds": ["T2", "RR"], "cell_priority": [list(x) for x in P["deciding"]], "beside_order": [list(x) for x in P["beside"]],
           "source_manifests": {"stage4": {"member": "pkg/MANIFEST4.json", "sha256": h4, "tar": os.path.basename(src_tar4)}}, "totals": m4["totals"], "start_classes": m4["start_classes"]}
    est_tar = sum(e["size"] for e in ov) + (1 << 20)                                                   # the tar is the files plus headers and the manifest: well under +1 MiB, and alloc_mib rounds up
    man["footprint"] = footprint5(scenes, est_tar, [e["size"] for e in ov] + [1 << 20])
    with open(f"{pkg}/MANIFEST5.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    name_tar = f"t2v8f_stage5_overlay_{run}{fixture_tag or ''}.tar"
    tar_path = f"{out_dir}/{name_tar}"
    assert not os.path.exists(tar_path), f"{tar_path} exists: move it aside first (nothing is overwritten)"
    members = sorted([("MANIFEST5.json", f"{pkg}/MANIFEST5.json")] + files)
    with tarfile.open(tar_path + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for rel, src in members:
            ti = tarfile.TarInfo(f"pkg/{rel}")
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(src), MK.MTIME, 0o644, 0, 0, "", ""
            with open(src, "rb") as fh:
                tf.addfile(ti, fh)
    os.replace(tar_path + ".tmp", tar_path)
    tar_sha = MK.sha256_file(tar_path)
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{tar_sha} *{name_tar}\n")
    assert os.path.getsize(tar_path) <= est_tar and MK.alloc_mib(os.path.getsize(tar_path)) <= MK.alloc_mib(est_tar), "the tar is larger than the footprint estimate in MANIFEST5"
    fp = man["footprint"]
    res = {"run": run, "arm": name, "tag": tag, "dose": dose, "tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tar_sha, "n_files": len(members), "members": [f"pkg/{r}" for r, _ in members],
           "manifest5_sha256": MK.sha256_file(f"{pkg}/MANIFEST5.json"), "n_scenes": len(scenes), "n_starts": n_starts, "n_cells": fp["cells"], "footprint": fp,
           "driver_sha256": MK.sha256_file(f"{pkg}/code/t2v8f_stage5.py"), "cell_sha256": MK.sha256_file(f"{pkg}/code/t2v8f_stage5_cell.py"), "cmd_sha256": MK.sha256_file(f"{pkg}/jobs/stage5_{run}.cmd"),
           "ckpt_sha256": ck_sha, "ckpt_bytes": {a: ck_src[a]["bytes"] for a in arms}, "ri_tool_sha256": C0C.RI_TOOL_SHA, "stage1_manifest_sha256": s1_sha,
           "requires": req, "source_manifests": man["source_manifests"], "overlay_files": ov, "seconds": round(time.time() - t_start, 1)}
    with open(f"{out_dir}/BUILD_RESULT5_{run}{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arm", required=True)
    b.add_argument("--tag", default="")
    b.add_argument("--dose", default=None)
    b.add_argument("--ckpt-s0", required=True)
    b.add_argument("--ckpt-s1", required=True)
    b.add_argument("--out-dir", default=None)
    b.add_argument("--stage-dir", default=None)
    b.add_argument("--allow-stage1-ckpt", action="store_true")
    a = ap.parse_args(argv)
    run = run_id(a.arm, a.tag)
    out = (a.out_dir or f"{T2}/stage5/{run}").replace("\\", "/")
    stg = (a.stage_dir or f"E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/build_{run}").replace("\\", "/")
    print(json.dumps(build(out, stg, a.arm, a.tag, {0: a.ckpt_s0, 1: a.ckpt_s1}, a.dose, allow_stage1_ckpt=a.allow_stage1_ckpt), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
