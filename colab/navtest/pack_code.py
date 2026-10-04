"""Pack the code bundle for a P0/P1 VM (dev box side): OUR code only, ~1 MB, uploaded with `colab upload`.

    python colab/navtest/pack_code.py --out <scratch>/p0_code.tgz --evidence colab/raw/<date>-navtest-p0

Carries: this directory's VM scripts; W3's harness driver + wrapper + agents and E1's wrapper (imported unchanged by
linux_run_v1.py); the REFe seam modules exactly as the dev box's eval runs them; the two venv pin lists; the public-
data fetch manifests; the JPEG decode probe (4 JPEGs of one public navtest token + the dev box's decode hashes);
signed_pull.py. Writes code_manifest.json (every file's source path, bytes, sha256) into --evidence, so the exact
code that ran on the VM is on record. No key, token, link or private data is in the bundle.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB = os.path.dirname(HERE)
PKG = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
W3C = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code"
E1 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/"
      "code/navsim_win.py")
PLAN = f"{COLAB}/raw/2026-09-27-navtest-plan"
FRAMES = "D:/Projects/TanitAD/data/refe_navtest/frames"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    files = {f"navtest/{n}": f"{HERE}/{n}" for n in ("vm_p0.py", "linux_run_v1.py", "navsim_v1_linux.py",
                                                      "e6_linux.py", "seam_shards.py", "zip_member_fetch.py")}
    files.update({f"w3code/{n}": f"{W3C}/{n}" for n in ("run_v1.py", "navsim_v1_win.py", "w3_agents_v1.py")})
    files["e1/navsim_win.py"] = E1
    for m in ("planner", "model", "ckpt_io", "load_dinov3", "navtrain_scenarios", "calib_table"):
        files[f"pkg/refe/{m}.py"] = f"{PKG}/refe/{m}.py"
    files["pkg/code/augment_routes.py"] = f"{PKG}/code/augment_routes.py"
    files["pkg/eval/refe_navtest_seam.py"] = f"{PKG}/eval/refe_navtest_seam.py"
    files["freeze/navsim_crun_venv_freeze.txt"] = f"{PLAN}/navsim_crun_venv_freeze.txt"
    files["freeze/driverl_eval_venv_freeze.txt"] = f"{PLAN}/driverl_eval_venv_freeze.txt"
    files["manifests/db_manifest_sub200.json"] = f"{a.evidence}/db_manifest_sub200.json"
    files["manifests/maps_manifest.json"] = f"{a.evidence}/maps_manifest.json"
    files["probe/jpeg_devbox_hashes.json"] = f"{PLAN}/jpeg_devbox_hashes.json"
    files["signed_pull.py"] = f"{COLAB}/signed_pull.py"
    man = {}
    with tarfile.open(a.out, "w:gz") as tf:
        for arc, src in sorted(files.items()):
            b = open(src, "rb").read()
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime = len(b), 0
            tf.addfile(ti, io.BytesIO(b))
            man[arc] = {"src": src, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
        # the JPEG probe: the 4 camera JPEGs of the token jpeg_devbox_hashes.json was computed on
        jh = json.load(open(f"{PLAN}/jpeg_devbox_hashes.json", encoding="utf-8"))
        rows = {r["token"]: r for r in json.load(open(f"{FRAMES}/tokens_frames.json", encoding="utf-8"))}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for rel in rows[jh["token"]]["cams"]:
                cam = rel.replace("\\", "/").split("/")[-2]
                base = os.path.basename(rel)
                z.writestr(base, zipfile.ZipFile(f"{FRAMES}/{jh['log']}_{cam}.zip").read(base))
        b = buf.getvalue()
        assert sorted(zipfile.ZipFile(io.BytesIO(b)).namelist()) == sorted(jh["hashes"]), "probe members != hashes"
        ti = tarfile.TarInfo("probe/probe.zip")
        ti.size, ti.mtime = len(b), 0
        tf.addfile(ti, io.BytesIO(b))
        man["probe/probe.zip"] = {"src": f"{FRAMES} (token {jh['token']})", "bytes": len(b),
                                  "sha256": hashlib.sha256(b).hexdigest()}
    json.dump({"bundle_bytes": os.path.getsize(a.out), "files": man},
              open(os.path.join(a.evidence, "code_manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"ZZCODE_OK {len(man)} files, {os.path.getsize(a.out):,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
