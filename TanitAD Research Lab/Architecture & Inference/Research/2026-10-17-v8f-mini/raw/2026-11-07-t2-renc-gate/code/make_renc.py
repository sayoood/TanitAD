"""Builds the RENC package for box 2 (PREREG_T2_RENC.md). Dev box, CPU. Nothing is staged, committed, downloaded or overwritten (a build refuses to write into a non-empty --out-dir).

    python -I make_renc.py build [--out-dir build]

Outputs in --out-dir:
  t2v8f_renc.tar          entries under `renc/` (extract with `tar -xf ... -C D:\\t2v8f`): code/{renc_lib,renc_box2,ralign_lib,t2v8f_lora_merge}.py, jobs/{renc_data,renc_train,renc_gate}.cmd, MANIFEST_RENC.json
  t2v8f_renc.tar.sha256   sha256sum format
  expect_files.txt        'relpath SHA256UPPER size' under D:/t2v8f/renc, verified on box 2 right after the extraction
  expect_external.txt     'relpath SHA256UPPER size' under D:/t2v8f, files this package imports from the GAP48 package that is already on box 2 (verified before any launch)
  BUILD_RESULT_RENC.json  the build's own record
REFUSES (named): the card's frozen prefix is not 5,544 bytes with sha256 425d3496..., a source file has a CR, a copied file does not hash equal to its source, the GAP48 inputs are missing.
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
CARD = os.path.join(os.path.dirname(T2), "t2_prereg", "PREREG_T2_RENC.md").replace("\\", "/")
CARD_PREFIX_BYTES, CARD_PREFIX_SHA256 = 5544, "425d3496eb52e30025c11c2609409ea1478aa2fe70ba6609c9f9477c128fd722"
SOURCES = {"renc_lib.py": f"{HERE}/renc_lib.py", "renc_box2.py": f"{HERE}/renc_box2.py", "ralign_lib.py": f"{T2}/ralign/ralign_lib.py", "t2v8f_lora_merge.py": f"{T2}/code_lc/t2v8f_lora_merge.py"}
EXTERNAL = {"gap48/code/t2v8f_gap48.py": f"{T2}/gap48/t2v8f_gap48.py", "gap48/MANIFEST_GAP48.json": f"{T2}/gap48/build/MANIFEST_GAP48.json"}
BOX = "D:/t2v8f/renc"
HEAD = ("@echo off\n"
        "rem RENC on box 2 (PREREG_T2_RENC.md), job: {job}. ONE job, safe mode. Launch: powershell -NoProfile -File D:\\tanitad\\launch.ps1 -Job D:\\t2v8f\\renc\\jobs\\{name}.cmd -Log D:\\t2v8f\\logs\\{name}.log\n"
        "rem No labels and no goto on purpose: this file is LF-only, and cmd.exe can mis-read labels in an LF-only batch file. A crash (no marker written) re-runs the job; every job resumes.\n"
        "call D:\\tanitad\\env.cmd\n"
        'set "PYTHONPATH="\n')
GPU_ENV = ('set "VCV=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"\n'
           'if exist "%VCV%" (call "%VCV%" >nul) else (echo ZZNO-VCVARS)\n'
           'set "CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8"\n'
           'set "CUDA_PATH=%CUDA_HOME%"\n'
           'set "PATH=%CUDA_HOME%\\bin;%PATH%"\n'
           'set "TORCH_CUDA_ARCH_LIST=8.9"\n'
           'set "T2A_KIT=D:/t2a/box2_kit"\n')
COMMON_ENV = 'set "RENC_PKG_CODE=D:/t2v8f/pkg/code"\nset "RENC_GAP_CODE=D:/t2v8f/gap48/code"\n'
PY = "%RUNSAFE% %PY% -I -u D:\\t2v8f\\renc\\code\\renc_box2.py"
RETRY = "for /L %%i in (1,1,{n}) do (\nif not exist {done} if not exist {fail} (\necho ===== {tag} ATTEMPT %%i %DATE% %TIME%\n{run}\n)\n)\n"


def job_cmds():
    d, o = f"{BOX}/data", f"{BOX}/out"
    dw, ow = d.replace("/", "\\"), o.replace("/", "\\")
    common = "--hf D:/tanitad/hf"
    data_run = (f"{PY} data --pkg D:/t2v8f/pkg --gap D:/t2v8f/gap48 --kit D:/t2a/box2_kit {common} --scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --out {d}")
    train_run = f"{PY} train --folds A,B --data {d} {common} --out {o}"
    gate_run = f"{PY} gate-encode --data {d} {common} --out {o}"
    return {
        "renc_data": HEAD.format(job="DATA (GAP48's render + decode + encode, stride 2, cylinders kept)", name="renc_data") + GPU_ENV + COMMON_ENV + f"if not exist {dw} mkdir {dw}\n"
        + RETRY.format(n=3, done=f"{dw}\\RENC_DATA.done", fail=f"{dw}\\RENC_DATA.fail", tag="RENC DATA", run=data_run) + "echo ===== ZZJOB-END %DATE% %TIME%\n",
        "renc_train": HEAD.format(job="TRAIN (E'_A then E'_B, exactly 10 epochs each, resumable by epoch)", name="renc_train") + 'set "T2A_KIT=D:/t2a/box2_kit"\n' + COMMON_ENV
        + f"if not exist {ow} mkdir {ow}\n" + RETRY.format(n=4, done=f"{ow}\\RENC_TRAIN.done", fail=f"{ow}\\RENC_TRAIN.fail", tag="RENC TRAIN", run=train_run) + "echo ===== ZZJOB-END %DATE% %TIME%\n",
        "renc_gate": HEAD.format(job="GATE-ENCODE (G-ID0 + the GAP48 ticks through each merged E')", name="renc_gate") + 'set "T2A_KIT=D:/t2a/box2_kit"\n' + COMMON_ENV
        + f"if not exist {ow} mkdir {ow}\n" + RETRY.format(n=2, done=f"{ow}\\RENC_GATE_ENCODE.done", fail=f"{ow}\\RENC_GATE_ENCODE.fail", tag="RENC GATE", run=gate_run) + "echo ===== ZZJOB-END %DATE% %TIME%\n",
    }


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha_f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build(out_dir, card=CARD, sources=None, external=None):
    sources, external = sources or SOURCES, external or EXTERNAL
    cb = open(card, "rb").read()
    assert len(cb) >= CARD_PREFIX_BYTES and sha_b(cb[:CARD_PREFIX_BYTES]) == CARD_PREFIX_SHA256, f"REFUSED: the card's first {CARD_PREFIX_BYTES} bytes do not hash to {CARD_PREFIX_SHA256[:8]}..."
    os.makedirs(out_dir, exist_ok=True)
    assert not os.listdir(out_dir), f"REFUSED: {out_dir} is not empty (move the old build aside; a build never overwrites)"
    own, blobs = [], {}
    for name, src in sources.items():
        b = open(src, "rb").read()
        assert b"\r" not in b, f"REFUSED: a CR in {src} (LF only)"
        blobs[f"code/{name}"] = b
        own.append({"path": f"code/{name}", "size": len(b), "sha256": sha_b(b), "source": src})
    cmds = job_cmds()
    for n, c in cmds.items():
        b = c.encode()
        assert b"\r" not in b
        blobs[f"jobs/{n}.cmd"] = b
        own.append({"path": f"jobs/{n}.cmd", "size": len(b), "sha256": sha_b(b)})
    ext = []
    for rel, src in external.items():
        assert os.path.isfile(src), f"REFUSED: GAP48 input missing {src}"
        ext.append({"path": rel, "size": os.path.getsize(src), "sha256": sha_f(src), "source": src})
    man = {"stage": "renc", "card": os.path.basename(card), "card_prefix_bytes": CARD_PREFIX_BYTES, "card_prefix_sha256": CARD_PREFIX_SHA256, "own_files": own, "external_files": ext, "box_dir": BOX,
           "data_dir": f"{BOX}/data", "out_dir": f"{BOX}/out", "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    mb = (json.dumps(man, indent=1) + "\n").encode()
    assert b"\r" not in mb
    tar_path = f"{out_dir}/t2v8f_renc.tar"
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        def add_bytes(name, b):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(b), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(b))
        for d in ("renc", "renc/code", "renc/jobs"):
            ti = tarfile.TarInfo(d)
            ti.type, ti.mtime, ti.mode = tarfile.DIRTYPE, int(time.time()), 0o755
            tf.addfile(ti)
        for rel, b in blobs.items():
            add_bytes(f"renc/{rel}", b)
        add_bytes("renc/MANIFEST_RENC.json", mb)
    for e in own:                                                                 # the copies in the tar hash equal to what the manifest says (read back, not trusted)
        with tarfile.open(tar_path) as tf:
            assert sha_b(tf.extractfile(f"renc/{e['path']}").read()) == e["sha256"], f"REFUSED: {e['path']} in the tar does not hash to the manifest"
    for name, src in sources.items():
        assert sha_f(src) == next(e["sha256"] for e in own if e["path"] == f"code/{name}"), f"REFUSED: {src} changed during the build"
    tsha = sha_f(tar_path)
    open(f"{tar_path}.sha256", "w", newline="\n").write(f"{tsha}  t2v8f_renc.tar\n")
    open(f"{out_dir}/MANIFEST_RENC.json", "wb").write(mb)
    with open(f"{out_dir}/expect_files.txt", "w", newline="\n") as f:
        for e in own:
            f.write(f"{e['path']} {e['sha256'].upper()} {e['size']}\n")
        f.write(f"MANIFEST_RENC.json {sha_b(mb).upper()} {len(mb)}\n")
    with open(f"{out_dir}/expect_external.txt", "w", newline="\n") as f:
        for e in ext:
            f.write(f"{e['path']} {e['sha256'].upper()} {e['size']}\n")
    res = {"tar": tar_path, "tar_sha256": tsha, "tar_bytes": os.path.getsize(tar_path), "manifest_sha256": sha_b(mb), "own_files": [{k: v for k, v in e.items() if k != "source"} for e in own],
           "external_files": [{k: v for k, v in e.items() if k != "source"} for e in ext], "card_prefix_sha256": CARD_PREFIX_SHA256,
           "footprint": {"data_gib_estimated": 4.7, "d_free_gib_needed": 6.0, "note": "ESTIMATED 9,554 gate-grid ticks / 2 = 4,777 pairs x 2 x 491,520 B; the job's preflight floor is 6 GiB free and it stops at 1.5 GiB"}}
    json.dump(res, open(f"{out_dir}/BUILD_RESULT_RENC.json", "w", encoding="utf-8", newline="\n"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("--out-dir", default=f"{HERE}/build")
    a = ap.parse_args(argv)
    r = build(a.out_dir)
    print(json.dumps({k: r[k] for k in ("tar", "tar_sha256", "tar_bytes", "manifest_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
