#!/usr/bin/env python3
"""VM-side runner for NAVSIM-on-Colab P0 (colab/NAVTEST_ON_COLAB_PLAN.md section f), launched DETACHED by
launch_p0.py so no `colab exec` timeout can interrupt it (RUNNER.md 9, trap 4). stdlib only (the system python).

    python3 vm_p0.py --stages setup_harness,gh,ge6,gh2            # CPU runtime
    python3 vm_p0.py --stages setup_harness,setup_seam,genv,seam,gp,rep,ge6 --gpu --e6-workers 10   # L4

Layout (/content/p0): code/ (the uploaded code bundle), src/ (public code, pinned SHAs), data/ (public data),
bundle/ (the relay tars, unpacked), exp/ (NAVSIM_EXP_ROOT), out/ (every result; status.json after EVERY step).
Public data is pulled from its public source and VERIFIED (CRC-32 per zip member against the dev box's copy of the
same zip; sha256 for the HF files). Private files arrive by signed links (signed_pull.py, md5-verified, no token on
this VM). Each stage records rc, seconds and log tails; a failed stage does not stop independent later stages, and
a stage whose inputs failed is SKIPPED with the reason -- never silently run on a partial input.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/content/p0"
CODE, SRC, DATA, BUNDLE, EXP, OUT = (f"{ROOT}/{d}" for d in ("code", "src", "data", "bundle", "exp", "out"))
RELAY = "/content/relay"
NV, DV = f"{ROOT}/venv_navsim", f"{ROOT}/venv_driverl"
NVPY, DVPY = f"{NV}/bin/python", f"{DV}/bin/python"
V11 = f"{SRC}/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
NUP = f"{SRC}/nuplan-devkit-ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
DZ = f"{SRC}/DriveZero-24959547edac549d64c34b2aab62d5c52599d078"
MAPS = f"{DATA}/maps/nuplan-maps-v1.0"
LOGS = f"{DATA}/openscene/navsim_logs/test"
DBS = f"{DATA}/nuplan/test"
TRUNK = f"{DATA}/backbones/dinov3-vitl16"
TOKENS = f"{BUNDLE}/tokens/A1_sub200_tokens.json"
EXPORT = f"{BUNDLE}/inputs/navtest_inputs_sub200.json.gz"
SNAP = f"{RELAY}/snap_epoch016.pt"
NAME = "sub200_ep016"
HF = "https://huggingface.co"
PUBLIC = {
    "navsim_logs": (f"{HF}/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz",
                    476034809, "871ba786da584e03f5151924674ab006e83782f796a979ee5dd144b5c6dca1ab"),
    "trunk": (f"{HF}/timm/vit_large_patch16_dinov3.lvd1689m/resolve/main/model.safetensors",
              1212347640, "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"),
}
CODELOAD = {
    "navsim": "autonomousvision/navsim/zip/3e8291bfa89ff247231e0227778840cd0a036896",
    "nuplan": "motional/nuplan-devkit/zip/ce3c323af01c0d7ec5672f7832ef53f9c679aab0",
    "drivezero": "XiaomiAutoL3/DriveZero/zip/24959547edac549d64c34b2aab62d5c52599d078",
}

os.makedirs(OUT, exist_ok=True)
ST = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}, "stages": {}}
LOCK = threading.Lock()


def save():
    with LOCK:
        tmp = f"{OUT}/status.json.tmp"
        with open(tmp, "w") as f:
            json.dump(ST, f, indent=1)
        os.replace(tmp, f"{OUT}/status.json")


def note(k, v):
    with LOCK:
        ST[k] = v
    save()


def sh(name, cmd, timeout=3600, env=None, cwd=None, keep=3000):
    t = time.time()
    log = f"{OUT}/logs/{name}.log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    try:
        with open(log, "w") as fh:
            p = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                               env=env, cwd=cwd, executable="/bin/bash" if isinstance(cmd, str) else None)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    txt = open(log, errors="replace").read()
    rec = {"rc": rc, "s": round(time.time() - t, 1), "tail": txt[-keep:]}
    with LOCK:
        ST["steps"][name] = rec
    save()
    print(f"ZZSTEP {name} rc={rc} s={rec['s']}", flush=True)
    return rc == 0, txt


def step_py(name, fn):
    t = time.time()
    try:
        info = fn()
        rec = {"rc": 0, "s": round(time.time() - t, 1), "info": info}
    except Exception as e:  # noqa: BLE001 -- recorded, the stage decides
        rec = {"rc": 1, "s": round(time.time() - t, 1), "error": f"{type(e).__name__}: {str(e)[:500]}"}
    with LOCK:
        ST["steps"][name] = rec
    save()
    print(f"ZZSTEP {name} rc={rec['rc']} s={rec['s']}", flush=True)
    return rec["rc"] == 0


def download(url, dst, size, sha):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h, n, t0 = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f:
        for b in iter(lambda: r.read(1 << 22), b""):
            f.write(b)
            h.update(b)
            n += len(b)
    dt = time.time() - t0
    if n != size or h.hexdigest() != sha:
        raise RuntimeError(f"{os.path.basename(dst)}: {n} B sha {h.hexdigest()[:16]} != {size} B {sha[:16]}")
    os.replace(dst + ".part", dst)
    return {"bytes": n, "sha256": h.hexdigest(), "s": round(dt, 1), "mb_s": round(n / 1e6 / max(dt, 1e-6), 1)}


def env_harness():
    e = dict(os.environ)
    e.update({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    return e


def env_seam():
    e = dict(os.environ)
    dzr = f"{DZ}/DriveRL"
    e.update({"DZ_ROOT": dzr, "NUPLAN_DATA_ROOT": f"{DATA}/nuplan", "NUPLAN_MAPS_ROOT": MAPS,
              "DRIVERL_EVAL_ROOT": dzr, "DRIVERL_EVAL_NUPLAN_ROOT": f"{dzr}/nuplan-devkit",
              "PYTHONPATH": f"{dzr}/nuplan-devkit:{dzr}/src", "REFE_BACKBONE_ROOT": f"{DATA}/backbones",
              "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
              "MPLBACKEND": "Agg"})
    return e


def paths_json():
    p = f"{OUT}/paths.json"
    json.dump({"w3_code": f"{CODE}/w3code", "e1_wrapper": f"{CODE}/e1/navsim_win.py", "nv_py": NVPY,
               "v11_tree": V11, "nuplan_tree": NUP, "exp": EXP, "openscene": f"{DATA}/openscene", "maps": MAPS},
              open(p, "w"), indent=1)
    return p


def harness(label, arm, out, cache="navtest", shim=True, cmd="score"):
    a = [NVPY, f"{CODE}/navtest/linux_run_v1.py", cmd, "--label", label, "--tokens", TOKENS, "--out", out,
         "--cache-name", cache, "--paths", paths_json()]
    if arm:
        a += ["--arm", arm]
    if shim:
        a.append("--windowspath-shim")
    return sh(label, a, timeout=3600, env=env_harness(), cwd=ROOT)


# ------------------------------------------------------------------------------------------------ setup stages
def relay_pull():
    ok, txt = sh("relay_pull", ["python3", f"{CODE}/signed_pull.py"], timeout=1800)
    ok = ok and "ZZPULL_OK" in txt
    if ok:
        for t in sorted(glob.glob(f"{RELAY}/*.tar")):
            with tarfile.open(t) as tf:
                tf.extractall(BUNDLE)
    note("relay", {"ok": ok, "pull": json.load(open("/content/relay_pull.json")) if os.path.exists(
        "/content/relay_pull.json") else None})
    return ok


def code_trees(names):
    os.makedirs(SRC, exist_ok=True)
    cmds = " && ".join(f"curl -sSL -o {n}.zip -w '{n} %{{size_download}} B %{{speed_download}} B/s\\n' "
                      f"https://codeload.github.com/{CODELOAD[n]} && unzip -q -o {n}.zip" for n in names)
    return sh("code_trees_" + "_".join(names), f"cd {SRC} && {cmds} && sha256sum *.zip", timeout=900)[0]


def navsim_venv():
    pins = f"{OUT}/navsim_pins_linux.txt"
    return sh("navsim_venv", f"grep -v -E '^-e |^(colorama|pywin32|pywinpty|torch)==' {CODE}/freeze/navsim_crun_venv_freeze.txt"
                             f" > {pins} && uv venv {NV} --python 3.9.25 && "
                             f"uv pip install --python {NVPY} --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.0.1+cpu' && "
                             f"uv pip install --python {NVPY} --no-deps -r {pins} && "
                             f"uv pip install --python {NVPY} --no-deps -e {NUP} && {NVPY} --version", timeout=1800)[0]


def seam_venv(gpu):
    pins = f"{OUT}/driverl_pins_linux.txt"
    torch = (f"uv pip install --python {DVPY} --index-url https://pypi.org/simple "
             f"--extra-index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1+cu128'" if gpu else
             f"uv pip install --python {DVPY} --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'")
    return sh("seam_venv", f"grep -v -E '^-e |^(colorama|torch)==' {CODE}/freeze/driverl_eval_venv_freeze.txt > {pins} && "
                           f"uv venv {DV} --python 3.11.13 && {torch} && "
                           f"uv pip install --python {DVPY} --no-deps -r {pins} && "
                           f"uv pip install --python {DVPY} --no-deps -e {DZ}/DriveRL -e {DZ}/DriveRL/nuplan-devkit && "
                           f"{DVPY} -c 'import torch; print(torch.__version__, torch.version.cuda)'", timeout=1800)[0]


def maps():
    ok, txt = sh("maps_fetch", ["python3", f"{CODE}/navtest/zip_member_fetch.py", "--manifest",
                                f"{CODE}/manifests/maps_manifest.json", "--out", f"{DATA}/maps", "--streams", "8",
                                "--report", f"{OUT}/maps_fetch.json"], timeout=1800)
    return ok and "ZZFETCH_OK" in txt and os.path.exists(f"{MAPS}/nuplan-maps-v1.0.json")


def navsim_logs():
    def f():
        tgz = f"{DATA}/openscene_metadata_test.tgz"
        url, size, sha = PUBLIC["navsim_logs"]
        info = download(url, tgz, size, sha)
        os.makedirs(LOGS, exist_ok=True)
        n = 0
        with tarfile.open(tgz, "r:gz") as tf:
            for m in tf:
                if m.isfile() and m.name.startswith("openscene-v1.1/meta_datas/test/") and m.name.endswith(".pkl"):
                    m.name = os.path.basename(m.name)
                    tf.extract(m, LOGS)
                    n += 1
        os.remove(tgz)
        info["n_pkl"] = n
        return info
    return step_py("navsim_logs", f)


def trunk():
    url, size, sha = PUBLIC["trunk"]
    return step_py("trunk", lambda: download(url, f"{TRUNK}/model.safetensors", size, sha))


def dbs():
    ok, txt = sh("db_fetch", ["python3", f"{CODE}/navtest/zip_member_fetch.py", "--manifest",
                              f"{CODE}/manifests/db_manifest_sub200.json", "--out", DBS, "--streams", "12",
                              "--report", f"{OUT}/db_fetch.json"], timeout=3600)
    return ok and "ZZFETCH_OK" in txt


def devbox_cache():
    def f():
        src, dst = f"{BUNDLE}/metric_cache_navtest", f"{EXP}/metric_cache_navtest"
        os.makedirs(EXP, exist_ok=True)
        if not os.path.exists(dst):
            shutil.move(src, dst)
        pk = sorted(glob.glob(f"{dst}/*/unknown/*/metric_cache.pkl"))
        os.makedirs(f"{dst}/metadata", exist_ok=True)
        with open(f"{dst}/metadata/metric_cache_navtest_metadata_node_0.csv", "w") as fh:
            fh.write("file_name\n" + "".join(p + "\n" for p in pk))
        return {"n_pkl": len(pk)}
    return step_py("devbox_cache", f)


def setup(stages, gpu):
    os.makedirs(DATA, exist_ok=True)
    want_h, want_s = "setup_harness" in stages, "setup_seam" in stages
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {"relay": ex.submit(relay_pull)}
        pys = " ".join((["3.9.25"] if want_h else []) + (["3.11.13"] if want_s else []))
        fut["python"] = ex.submit(lambda: sh("uv_python", f"uv python install {pys}", timeout=900)[0])
        trees = (["navsim", "nuplan"] if want_h else []) + (["drivezero"] if want_s else [])
        fut["trees"] = ex.submit(code_trees, trees)
        fut["maps"] = ex.submit(maps)
        if want_h:
            fut["logs"] = ex.submit(navsim_logs)
        if want_s:
            fut["dbs"] = ex.submit(dbs)
            fut["trunk"] = ex.submit(trunk)
        base_ok = fut["python"].result() and fut["trees"].result()
        if want_h:
            fut["navsim_venv"] = ex.submit(navsim_venv) if base_ok else None
        if want_s:
            fut["seam_venv"] = ex.submit(seam_venv, gpu) if base_ok else None
        for k, f in fut.items():
            res[k] = bool(f.result()) if f is not None else False
    if res.get("relay") and want_h:
        res["devbox_cache"] = devbox_cache()
    note("setup", res)
    return res


# ------------------------------------------------------------------------------------------------ gate stages
def genv():
    r = {}
    ok, txt = sh("genv_selftest", f"cd {CODE}/pkg/eval && {DVPY} refe_navtest_seam.py --selftest", env=env_seam())
    r["selftest"] = "ZZCONVERTER_EXACTZZ" in txt
    probe = (f"import cv2, numpy as np, hashlib, zipfile, json; z = zipfile.ZipFile('{CODE}/probe/probe.zip'); "
             "o = {n: hashlib.sha256(np.ascontiguousarray(cv2.resize(cv2.imdecode(np.frombuffer(z.read(n), np.uint8), "
             "cv2.IMREAD_COLOR), (960, 512))[:, :, ::-1]).tobytes()).hexdigest() for n in sorted(z.namelist())}; "
             "print('ZZJPEG ' + json.dumps(o))")
    ok, txt = sh("genv_jpeg", [DVPY, "-c", probe], env=env_seam())
    want = json.load(open(f"{CODE}/probe/jpeg_devbox_hashes.json"))["hashes"]
    got = json.loads(txt.split("ZZJPEG ", 1)[1].splitlines()[0]) if "ZZJPEG " in txt else {}
    r["jpeg_equal"] = f"{sum(1 for k in want if got.get(k) == want[k])}/{len(want)}"
    params = ("import sys, json; sys.path.insert(0, '../refe'); import torch, model, planner; "
              "m = model.REFe(model.REFeConfig.for_backbone('vitl16')); "
              "print('ZZPARAMS ' + json.dumps([sum(p.numel() for p in m.parameters()), "
              "sum(p.numel() for p in m.parameters() if p.requires_grad), planner.REFePlanner.DEFAULT_RULE, "
              "planner.REFePlanner.REPAIR_LAST_HEADING]))")
    ok, txt = sh("genv_params", f"cd {CODE}/pkg/eval && {DVPY} -c \"{params}\"", env=env_seam())
    r["params"] = json.loads(txt.split("ZZPARAMS ", 1)[1].splitlines()[0]) if "ZZPARAMS " in txt else None
    cuda = ("import torch, json; x = torch.randn(2, 3, 64, 64, device='cuda'); w = torch.randn(8, 3, 3, 3, "
            "device='cuda'); y = torch.nn.functional.conv2d(x, w); torch.cuda.synchronize(); "
            "print('ZZCUDA ' + json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
            "'cudnn': torch.backends.cudnn.version(), 'device': torch.cuda.get_device_name(0), "
            "'cap': torch.cuda.get_device_capability(0), 'conv_ok': bool(torch.isfinite(y).all()), "
            "'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32}))")
    ok, txt = sh("genv_cuda", [DVPY, "-c", cuda], env=env_seam())
    r["cuda"] = json.loads(txt.split("ZZCUDA ", 1)[1].splitlines()[0]) if "ZZCUDA " in txt else None
    ok, txt = sh("genv_nvidia_smi", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nproc; "
                                    "free -g | head -2")
    r["nvidia_smi"] = txt[-600:]
    note("genv", r)


def seam_single():
    os.makedirs(f"{OUT}/seam", exist_ok=True)
    ok, txt = sh("seam", f"cd {CODE}/pkg/eval && {DVPY} refe_navtest_seam.py --ckpt {SNAP} --frames {BUNDLE}/frames "
                         f"--db-dir {DBS} --export {EXPORT} --tokens {TOKENS} --out {OUT}/seam/refe_{NAME}_vm.npz "
                         f"--arm REFe_{NAME} --dump-proposals {OUT}/seam/proposals_vm.npz", env=env_seam(), timeout=5400)
    note("seam", {"ok": ok and "ZZSEAM_OK" in txt, "line": [x for x in txt.splitlines() if "ZZSEAM_" in x][-1:]})
    return ok and "ZZSEAM_OK" in txt


def rep(k):
    ok, txt = sh("rep_shards", [DVPY, f"{CODE}/navtest/seam_shards.py", "--k", str(k), "--seam-script",
                                f"{CODE}/pkg/eval/refe_navtest_seam.py", "--ckpt", SNAP, "--frames", f"{BUNDLE}/frames",
                                "--db-dir", DBS, "--export", EXPORT, "--tokens", TOKENS, "--work", f"{OUT}/rep",
                                "--out", f"{OUT}/rep/refe_{NAME}_k{k}.npz", "--dump-out", f"{OUT}/rep/proposals_k{k}.npz",
                                "--arm", f"REFe_{NAME}"], env=env_seam(), timeout=5400)
    cmp_ = ("import numpy as np, json, sys; A = np.load(sys.argv[1]); B = np.load(sys.argv[2]); C = np.load(sys.argv[3]); "
            "D = np.load(sys.argv[4]); r = {'tokens_equal': A['token'].tolist() == B['token'].tolist(), "
            "'poses_bitwise': bool(np.array_equal(A['poses'], B['poses'])), "
            "'poses_max_abs': float(np.abs(A['poses'].astype('f8') - B['poses'].astype('f8')).max()), "
            "'props_bitwise': bool(np.array_equal(C['proposals'], D['proposals'])), "
            "'logits_bitwise': bool(np.array_equal(C['logits'], D['logits'])), "
            "'pick_equal': bool(np.array_equal(C['pick'], D['pick']))}; print('ZZREP ' + json.dumps(r))")
    ok2, txt2 = sh("rep_compare", [DVPY, "-c", cmp_, f"{OUT}/seam/refe_{NAME}_vm.npz", f"{OUT}/rep/refe_{NAME}_k{k}.npz",
                                   f"{OUT}/seam/proposals_vm.npz", f"{OUT}/rep/proposals_k{k}.npz"], env=env_seam())
    note("rep", {"shards_ok": ok and "ZZSHARDS_OK" in txt,
                 "compare": json.loads(txt2.split("ZZREP ", 1)[1].splitlines()[0]) if "ZZREP " in txt2 else None})


def gh2():
    out = f"{OUT}/gh2"
    ok, txt = harness("refe_cache_vm", None, out, cache="navtest_vm", shim=False, cmd="cache")
    note("gh2_cache", {"ok": ok})
    if not ok:
        return
    harness(f"refe_{NAME}_vmcache", f"SEAM:{BUNDLE}/ref/refe_{NAME}.npz", out, cache="navtest_vm", shim=False)
    for arm in ("CV", "STOP", "HUMAN"):
        harness(f"refe_{arm.lower()}_vmcache", arm, out, cache="navtest_vm", shim=False)


def ge6(workers):
    ok, txt = sh("ge6", [NVPY, f"{CODE}/navtest/e6_linux.py", "--dump", f"{BUNDLE}/ref/proposals.npz", "--name", NAME,
                         "--tokens", TOKENS, "--work", f"{OUT}/ge6", "--paths", paths_json(), "--workers", str(workers),
                         "--windowspath-shim", "--landed-seam", f"{BUNDLE}/ref/refe_{NAME}.npz",
                         "--landed-csv", f"{BUNDLE}/ref/refe_{NAME}.csv"], env=env_harness(), timeout=7200)
    note("ge6", {"ok": ok and "ZZE6_OK" in txt, "line": [x for x in txt.splitlines() if "ZZE6_" in x][-1:]})


def pack():
    """Everything the dev box needs for the gates, small: no pickles, no per-proposal hooks/rows."""
    dst = f"{ROOT}/p0_results.tar.gz"
    with tarfile.open(dst, "w:gz") as tf:
        for p in sorted(glob.glob(f"{OUT}/**/*", recursive=True)):
            if os.path.isdir(p):
                continue
            b = os.path.basename(p)
            if "/ge6/score/" in p and not (b.endswith(".csv") or b.endswith(".counts.json")):
                continue
            if "/ge6/seams/" in p or "/rep/shard" in p and p.endswith(".npz"):
                continue
            if b.endswith("_hooks.json") and os.path.getsize(p) > 5_000_000:
                continue
            tf.add(p, arcname=os.path.relpath(p, ROOT))
    note("packed", {"path": dst, "bytes": os.path.getsize(dst)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--e6-workers", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--deadline-min", type=float, default=150.0)
    a = ap.parse_args()
    stages = a.stages.split(",")
    note("args", vars(a))
    t_end = time.time() + 60 * a.deadline_min
    s = {}
    if "setup_harness" in stages or "setup_seam" in stages:
        s = setup(stages, a.gpu)
    have_h = s.get("relay") and s.get("navsim_venv") and s.get("maps") and s.get("logs") and s.get("devbox_cache")
    have_s = s.get("relay") and s.get("seam_venv") and s.get("maps") and s.get("dbs") and s.get("trunk")
    order = [x for x in ("genv", "gh", "seam", "gp", "rep", "gh2", "ge6") if x in stages]
    seam_ok = False
    for st in order:
        if time.time() > t_end:
            ST["stages"][st] = "SKIPPED_DEADLINE"
            continue
        need_h = st in ("gh", "gp", "gh2", "ge6")
        need_s = st in ("genv", "seam", "rep")
        if (need_h and not have_h) or (need_s and not have_s) or (st in ("gp", "rep") and not seam_ok):
            ST["stages"][st] = "SKIPPED_INPUTS"
            save()
            continue
        t = time.time()
        ST["stages"][st] = "RUNNING"
        save()
        if st == "genv":
            genv()
        elif st == "gh":
            ok, _ = harness(f"refe_{NAME}", f"SEAM:{BUNDLE}/ref/refe_{NAME}.npz", f"{OUT}/gh")
        elif st == "seam":
            seam_ok = seam_single()
        elif st == "gp":
            harness(f"refe_{NAME}_vm", f"SEAM:{OUT}/seam/refe_{NAME}_vm.npz", f"{OUT}/gp")
        elif st == "rep":
            rep(a.k)
        elif st == "gh2":
            gh2()
        elif st == "ge6":
            ge6(a.e6_workers)
        ST["stages"][st] = f"DONE {time.time() - t:.0f} s"
        save()
        pack()
    ST["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    pack()
    print("ZZP0_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
