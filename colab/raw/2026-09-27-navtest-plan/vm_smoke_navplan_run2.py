# VM-side CPU smoke, RUN 2 -- the steps run 1 never reached (run 1's in-kernel CSV comparison raised KeyError on an
# all-invalid CSV after the harness had failed on the missing nuPlan maps; see NAVTEST_ON_COLAB_PLAN.md section c).
# The seam venv (Python 3.11, the dev box's pins), the seam's own converter selftest, the REFe import + ViT-L model
# build, cross-OS JPEG decode+resize parity (the planner's exact preprocessing), a CPU speed reference for that
# phase, and link speed from the VM (range reads <= 50 MiB each, discarded). Same bundle; persists after every step.
import hashlib
import json
import os
import subprocess
import time

W = "/content/navplan"
RES = f"{W}/results_run2.json"
os.makedirs(W, exist_ok=True)
res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
T_END = time.time() + 25 * 60


def save():
    with open(RES + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(RES + ".tmp", RES)


def sh(name, cmd, timeout=900, keep=3000):
    left = T_END - time.time()
    if left < 30:
        res["steps"][name] = {"rc": "SKIPPED_DEADLINE"}
        save()
        print(f"ZZSTEP {name} SKIPPED_DEADLINE", flush=True)
        return None
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=min(timeout, left),
                           executable="/bin/bash")
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "TIMEOUT", "", str(e)[:500]
    rec = {"rc": rc, "s": round(time.time() - t, 1), "stdout_tail": (out or "")[-keep:], "stderr_tail": (err or "")[-keep:]}
    res["steps"][name] = rec
    save()
    print(f"ZZSTEP {name} rc={rc} s={rec['s']}", flush=True)
    return rec


B = f"{W}/bundle"
SRC = f"{W}/src"
DV = f"{W}/venv_driverl"
DVPY = f"{DV}/bin/python"
DZ = f"{SRC}/DriveZero-24959547edac549d64c34b2aab62d5c52599d078"

sh("vm_facts", "nproc; free -g | head -2; df -h /content | tail -1; lscpu | grep -E 'Model name|^CPU\\(s\\)|MHz'; uv --version")
sh("unpack_bundle", f"mkdir -p {B} && tar -xzf /content/navplan_bundle.tgz -C {B} && find {B} -type f | wc -l")
sh("python_install", "uv python install 3.11.13 2>&1 | tail -2; uv python find 3.11", timeout=300)
sh("code_download", f"mkdir -p {SRC} && cd {SRC} && curl -sSL -o DriveZero.zip -w 'DriveZero %{{size_download}} B %{{speed_download}} B/s\\n' "
   "https://codeload.github.com/XiaomiAutoL3/DriveZero/zip/24959547edac549d64c34b2aab62d5c52599d078 && "
   "unzip -q -o DriveZero.zip && sha256sum DriveZero.zip", timeout=300)
sh("driverl_pins", f"grep -v -E '^-e |^(colorama|torch)==' {B}/driverl_eval_venv_freeze.txt > {W}/driverl_pins_linux.txt; "
                   f"wc -l {W}/driverl_pins_linux.txt")
sh("driverl_venv", f"uv venv {DV} --python 3.11.13 && "
                   f"uv pip install --python {DVPY} --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu' && "
                   f"uv pip install --python {DVPY} --no-deps -r {W}/driverl_pins_linux.txt && "
                   f"uv pip install --python {DVPY} --no-deps -e {DZ}/DriveRL -e {DZ}/DriveRL/nuplan-devkit && "
                   f"{DVPY} --version && du -sb {DV}", timeout=1200)
sh("seam_selftest", f"cd {B}/pkg/eval && {DVPY} refe_navtest_seam.py --selftest 2>&1 | tail -7")
sh("seam_imports", f"cd {B}/pkg/eval && {DVPY} -c \"import sys, time; sys.path.insert(0, '../refe'); sys.path.insert(0, '../code'); "
                   "import torch, planner, model, navtrain_scenarios, calib_table, ckpt_io, load_dinov3, augment_routes, cv2, numpy; "
                   "from nuplan.planning.script import driverl_runtime_map_features as M; "
                   "from driverl.datatypes.goal_position_utils import route_goal_positions; "
                   "t = time.time(); m = model.REFe(model.REFeConfig.for_backbone('vitl16')); bs = time.time() - t; "
                   "n = sum(p.numel() for p in m.parameters()); tr = sum(p.numel() for p in m.parameters() if p.requires_grad); "
                   "print('ZZSEAM_IMPORTS', torch.__version__, cv2.__version__, numpy.__version__, n, tr, round(bs, 1), "
                   "planner.REFePlanner.DEFAULT_RULE, planner.REFePlanner.REPAIR_LAST_HEADING, planner.__file__)\" 2>&1 | tail -4",
   timeout=600)
sh("jpeg_decode_parity", f"{DVPY} {B}/jpeg_probe.py {B}/frames/probe.zip 2>&1 | tail -2")
S3 = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip"
HFU = ("https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/"
       "openscene_sensor_test_camera/openscene_sensor_test_camera_0.tgz")
sh("s3_speed_1stream", f"curl -sS -r 0-52428799 -o /dev/null -w 'S3x1 %{{http_code}} %{{size_download}} B %{{speed_download}} B/s %{{time_total}} s\\n' {S3}",
   timeout=300)
def par_range_speed(url, start, n_streams=4, part=13107200):
    """4 parallel range reads of `part` bytes each (50 MiB in total), timed end to end; bytes discarded."""
    import concurrent.futures
    import urllib.request

    def one(i):
        a = start + i * part
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + part - 1}"})
        n = 0
        with urllib.request.urlopen(req, timeout=120) as r:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                n += len(b)
        return n
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(n_streams) as ex:
        got = list(ex.map(one, range(n_streams)))
    dt = time.time() - t
    return {"bytes": sum(got), "seconds": round(dt, 2), "MBps": round(sum(got) / dt / 1e6, 1), "parts": got}


try:
    res["s3_speed_4streams"] = par_range_speed(S3, 1073741824)
except Exception as e:  # noqa: BLE001
    res["s3_speed_4streams"] = {"error": repr(e)[:300]}
save()
print("ZZSTEP s3_speed_4streams " + json.dumps(res["s3_speed_4streams"]), flush=True)

sh("hf_speed_1stream", f"curl -sSL -r 0-52428799 -o /dev/null -w 'HFx1 %{{http_code}} %{{size_download}} B %{{speed_download}} B/s %{{time_total}} s\\n' {HFU}",
   timeout=300)
res["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("ZZSMOKE2_DONE " + RES, flush=True)
