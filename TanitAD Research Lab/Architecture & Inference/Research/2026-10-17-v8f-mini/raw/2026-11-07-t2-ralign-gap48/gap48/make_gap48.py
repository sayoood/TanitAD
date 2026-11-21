"""Builds the GAP48 package for box 2. Dev box, CPU. Nothing is staged, committed, downloaded or overwritten (a new build refuses to write into a non-empty --out-dir).

    python -I make_gap48.py build [--scenes SCENES48.json] [--out-dir build] [--code-dir <t2v8f/code>]

Outputs in --out-dir:
  t2v8f_gap48.tar         entries under `gap48/` (extract with `tar -xf ... -C D:\\t2v8f`): code/t2v8f_gap48.py, jobs/gap48.cmd, MANIFEST_GAP48.json and the EMPTY directories mp4/<scene>/
  t2v8f_gap48.tar.sha256  sha256sum format
  put_mp4.sftp            48 `put` lines: the clips go from where they live on the dev box straight to box 2 (a second 2 GB copy on the dev box is never made)
  expect_files.txt        'relpath SHA256UPPER size' for the files the chain verifies on box 2 right after the extraction
  BUILD_RESULT_GAP48.json the build's own record (counts, shas, footprint)
MANIFEST_GAP48.json: the scene list with every scene file's size + sha256 (the box's preflight checks them), the clips (path, size, sha256), the shipped code, and `pinned_code` (the stage-1
package files this job imports, sha256 of the dev-box copies = the box-2 copies, MEASURED equal on 2026-11-07).
Asserts: 48 scenes (a fixture build says `fixture` and may differ), every clip hashes to the release listing, every text file is LF-only, no shipped path collides with the stage-1..5 packages (all live under gap48/).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import scenes48 as S  # noqa: E402

PINNED_CODE = ("t2v8f_stage1.py", "t2v8f_run.py", "t2v8f_nocv2.py", "t2a_v8f_encoder.py", "t2a_v8f_policy.py", "t2v8f_kit.py", "t2v8f_exec.py")
BOX_GAP = "D:/t2v8f/gap48"
MP4_NAME = "camera_front_wide_120fov.mp4"
JOB_CMD = ("@echo off\n"
           "rem GAP48 on box 2: the NuRec render gap on the 48 read scenes (descriptive). ONE job, safe mode. "
           "Launch: powershell -NoProfile -File D:\\tanitad\\launch.ps1 -Job D:\\t2v8f\\gap48\\jobs\\gap48.cmd -Log D:\\t2v8f\\logs\\gap48.log\n"
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
           "if not exist D:\\t2v8f\\pull_gap48 mkdir D:\\t2v8f\\pull_gap48\n"
           "echo ===== GAP48 START %DATE% %TIME%\n"
           "%RUNSAFE% %PY% -I -u D:\\t2v8f\\gap48\\code\\t2v8f_gap48.py all --pkg D:/t2v8f/pkg --gap D:/t2v8f/gap48 --kit D:/t2a/box2_kit --hf D:/tanitad/hf "
           "--scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --pull D:/t2v8f/pull_gap48\n"
           "echo ===== ZZJOB-END %DATE% %TIME% rc=%ERRORLEVEL%\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def build(scenes_json, out_dir, code_dir=f"{T2}/code", job_py=f"{HERE}/t2v8f_gap48.py", fixture=False, mp4_in_tar=False):
    sc = json.load(open(scenes_json, encoding="utf-8"))
    rows = sc["scenes"]
    if not fixture:
        assert (sc["n_all"], sc["n_excluded"], len(rows)) == (S.EXPECTED_ALL, S.EXPECTED_EXCLUDED, S.EXPECTED), f"REFUSED: {sc['n_all']} - {sc['n_excluded']} = {len(rows)} is not 62 - 14 = 48"
    os.makedirs(out_dir, exist_ok=True)
    assert not os.listdir(out_dir), f"REFUSED: {out_dir} is not empty (move the old build aside; a build never overwrites)"
    for r in rows:                                                                  # the clips: present and equal to what the scene list recorded
        assert os.path.getsize(r["mp4"]["path"]) == r["mp4"]["bytes"], f"clip size {r['scene']}"
        assert sha256_file(r["mp4"]["path"]) == r["mp4"]["sha256"], f"clip sha256 {r['scene']}"
    job_bytes = open(job_py, "rb").read()
    assert b"\r" not in job_bytes and b"\r" not in JOB_CMD.encode(), "a CR in a shipped text file (LF only)"
    cmd_bytes = JOB_CMD.encode()
    ids = sorted(r["scene"] for r in rows)
    own = [{"path": "code/t2v8f_gap48.py", "size": len(job_bytes), "sha256": sha256_bytes(job_bytes)},
           {"path": "jobs/gap48.cmd", "size": len(cmd_bytes), "sha256": sha256_bytes(cmd_bytes)}]
    own += [{"path": f"mp4/{r['scene']}/{MP4_NAME}", "size": r["mp4"]["bytes"], "sha256": r["mp4"]["sha256"]} for r in rows]
    man = {"stage": "gap48", "purpose": "NuRec render-vs-real gap on the read scenes; descriptive; decides nothing", "rule": sc.get("rule"), "fixture": bool(fixture), "n_scenes": len(rows),
           "n_all": sc["n_all"], "n_excluded": sc["n_excluded"], "excluded": sc["excluded"], "by_root": sc["by_root"], "scene_list_sha256": sha256_bytes(json.dumps(ids).encode()),
           "inputs": sc["inputs"], "tick_step": 10, "min_ticks_all": 20, "condition": "objects", "pinned_code": [{"file": f, "sha256": sha256_file(f"{code_dir}/{f}")} for f in PINNED_CODE],
           "own_files": own, "scenes": [{"scene": r["scene"], "idx": r["idx"], "where": r["where"], "files": r["files"], "mp4": {"bytes": r["mp4"]["bytes"], "sha256": r["mp4"]["sha256"]}} for r in rows],
           "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    man_bytes = (json.dumps(man, indent=1) + "\n").encode()
    assert b"\r" not in man_bytes
    tar_path = f"{out_dir}/t2v8f_gap48.tar"
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        def add_bytes(name, b):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(b), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(b))

        def add_dir(name):
            ti = tarfile.TarInfo(name)
            ti.type, ti.mtime, ti.mode = tarfile.DIRTYPE, int(time.time()), 0o755
            tf.addfile(ti)
        add_dir("gap48")
        add_dir("gap48/code")
        add_dir("gap48/jobs")
        add_dir("gap48/mp4")
        add_bytes("gap48/code/t2v8f_gap48.py", job_bytes)
        add_bytes("gap48/jobs/gap48.cmd", cmd_bytes)
        for r in rows:
            add_dir(f"gap48/mp4/{r['scene']}")
            if mp4_in_tar:
                tf.add(r["mp4"]["path"], arcname=f"gap48/mp4/{r['scene']}/{MP4_NAME}")
        add_bytes("gap48/MANIFEST_GAP48.json", man_bytes)
    tar_sha = sha256_file(tar_path)
    open(f"{tar_path}.sha256", "w", newline="\n").write(f"{tar_sha}  t2v8f_gap48.tar\n")
    open(f"{out_dir}/MANIFEST_GAP48.json", "wb").write(man_bytes)
    with open(f"{out_dir}/put_mp4.sftp", "w", newline="\n") as f:
        for r in rows:
            f.write(f"put {r['mp4']['path']} /{BOX_GAP}/mp4/{r['scene']}/{MP4_NAME}\n")
    with open(f"{out_dir}/expect_files.txt", "w", newline="\n") as f:                # what the chain verifies right after the extraction (clips are verified by the job's preflight)
        for e in own[:2]:
            f.write(f"{e['path']} {e['sha256'].upper()} {e['size']}\n")
        f.write(f"MANIFEST_GAP48.json {sha256_bytes(man_bytes).upper()} {len(man_bytes)}\n")
    mib = 1 << 20
    alloc = lambda sizes: sum(-(-s // mib) for s in sizes)                          # noqa: E731  exFAT 1 MiB clusters
    clip_mib = alloc([r["mp4"]["bytes"] for r in rows])
    res = {"tar": tar_path, "tar_sha256": tar_sha, "tar_bytes": os.path.getsize(tar_path), "manifest_sha256": sha256_bytes(man_bytes), "n_scenes": len(rows), "by_root": sc["by_root"],
           "clip_bytes": sum(r["mp4"]["bytes"] for r in rows), "scene_list_sha256": man["scene_list_sha256"], "job_sha256": own[0]["sha256"], "cmd_sha256": own[1]["sha256"],
           "pinned_code": man["pinned_code"], "fixture": bool(fixture), "mp4_in_tar": bool(mp4_in_tar),
           "footprint": {"clips_allocated_mib": clip_mib, "tokens_estimated_mib": len(rows) * 4 + 8, "d_free_gib_needed": round((clip_mib + len(rows) * 4 + 60) / 1024 + 3.0, 2),
                         "note": "clips 1.83 GiB + tokens ~0.2 GiB + the job's 3 GiB free floor; exFAT 1 MiB clusters counted"}}
    json.dump(res, open(f"{out_dir}/BUILD_RESULT_GAP48.json", "w", encoding="utf-8", newline="\n"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("--scenes", default=f"{HERE}/SCENES48.json")
    ap.add_argument("--out-dir", default=f"{HERE}/build")
    ap.add_argument("--code-dir", default=f"{T2}/code")
    a = ap.parse_args(argv)
    r = build(a.scenes, a.out_dir, a.code_dir)
    print(json.dumps({k: r[k] for k in ("tar", "tar_sha256", "tar_bytes", "manifest_sha256", "n_scenes", "clip_bytes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
