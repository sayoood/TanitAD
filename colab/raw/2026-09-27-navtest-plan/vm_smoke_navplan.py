# VM-side CPU smoke for the REFe-navtest-on-Colab plan (run with `colab exec -f` on a CPU runtime).
# Expects /content/navplan_bundle.tgz (uploaded from the dev box by navplan_smoke.ps1). Persists results after EVERY
# step to /content/navplan/results.json (RUNNER.md 9 trap 4) and prints one `ZZSTEP` line per step.
# Downloads: pip/uv wheels for the two venvs (environment, not data), GitHub codeload zips of the three public code
# trees (<100 MB each), and two 50 MiB RANGE reads (S3 nuPlan test zip, HF OpenScene camera shard) for link speed.
# No token is used or placed anywhere; nothing is pushed anywhere.
import hashlib
import json
import os
import shutil
import subprocess
import time

W = "/content/navplan"
RES = f"{W}/results.json"
os.makedirs(W, exist_ok=True)
res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
T_END = time.time() + 30 * 60            # hard in-VM wall-clock cap for this exec


def save():
    with open(RES + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(RES + ".tmp", RES)


def sh(name, cmd, timeout=900, keep=2500):
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
        rc, out, err = "TIMEOUT", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), ""
    rec = {"rc": rc, "s": round(time.time() - t, 1), "stdout_tail": (out or "")[-keep:], "stderr_tail": (err or "")[-keep:]}
    res["steps"][name] = rec
    save()
    print(f"ZZSTEP {name} rc={rc} s={rec['s']}", flush=True)
    return rec


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


B = f"{W}/bundle"
SRC = f"{W}/src"
NV = f"{W}/venv_navsim"
DV = f"{W}/venv_driverl"
NVPY, DVPY = f"{NV}/bin/python", f"{DV}/bin/python"
V11 = f"{SRC}/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
NUP = f"{SRC}/nuplan-devkit-ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
DZ = f"{SRC}/DriveZero-24959547edac549d64c34b2aab62d5c52599d078"
UV = "uv"

# ---- 0. the machine ---------------------------------------------------------------------------------------------
sh("vm_facts", "set -x; nproc; free -g | head -2; df -h /content | tail -1; head -2 /etc/os-release; python3 --version; "
               "(which uv && uv --version) || echo NO_UV; lscpu | grep -E 'Model name|^CPU\\(s\\)'; "
               "(nvidia-smi -L 2>/dev/null || echo NO_GPU)")
sh("unpack_bundle", f"mkdir -p {B} && tar -xzf /content/navplan_bundle.tgz -C {B} && find {B} -type f | wc -l && du -sb {B}")
r = sh("uv_present", "which uv && uv --version")
if not r or r["rc"] != 0:
    sh("uv_install", "python3 -m pip install -q uv && which uv && uv --version", timeout=300)
sh("python_install", f"{UV} python install 3.9.25 3.11.13 2>&1 || {UV} python install 3.9 3.11 2>&1; "
                     f"{UV} python find 3.9; {UV} python find 3.11", timeout=600)

# ---- 1. the public code trees (GitHub codeload, pinned SHAs) ----------------------------------------------------
sh("code_download", f"mkdir -p {SRC} && cd {SRC} && for u in "
   "autonomousvision/navsim/zip/3e8291bfa89ff247231e0227778840cd0a036896 "
   "motional/nuplan-devkit/zip/ce3c323af01c0d7ec5672f7832ef53f9c679aab0 "
   "XiaomiAutoL3/DriveZero/zip/24959547edac549d64c34b2aab62d5c52599d078; do "
   "n=$(echo $u | cut -d/ -f2); curl -sSL -o $n.zip -w \"$n %{size_download} B %{speed_download} B/s\\n\" "
   "https://codeload.github.com/$u; done; for z in *.zip; do unzip -q -o $z; done; ls; sha256sum *.zip", timeout=600)

# ---- 2. the NAVSIM harness venv (Python 3.9, the dev box's 202 pins minus Windows-only + editables) ---------------
sh("navsim_pins", f"grep -v -E '^-e |^(colorama|pywin32|pywinpty|torch)==' {B}/navsim_crun_venv_freeze.txt > {W}/navsim_pins_linux.txt; "
                  f"wc -l {W}/navsim_pins_linux.txt")
sh("navsim_venv", f"{UV} venv {NV} --python 3.9 && "
                  f"{UV} pip install --python {NVPY} --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.0.1+cpu' && "
                  f"{UV} pip install --python {NVPY} --no-deps -r {W}/navsim_pins_linux.txt && "
                  f"{UV} pip install --python {NVPY} --no-deps -e {NUP} && {NVPY} --version && du -sb {NV}", timeout=1200)
sh("navsim_import", f"cd {W} && PYTHONPATH={V11}:{B}/w3code {NVPY} -c \"import navsim, nuplan, numpy, shapely, torch; "
                    "from navsim.planning.script import run_pdm_score; import w3_agents_v1; "
                    "print('ZZIMPORT', navsim.__file__, nuplan.__file__, numpy.__version__, shapely.__version__, "
                    "shapely.geos_version_string, torch.__version__)\"")

# ---- 3. the dev-box metric cache on Linux: the WindowsPath trap, then the shim ---------------------------------
one = f"{B}/exp/metric_cache_navtest"
sh("winpath_trap", f"P=$(find {one} -name metric_cache.pkl | head -1); PYTHONPATH={V11} {NVPY} -c \"import lzma,pickle,sys; "
                   "pickle.load(lzma.open(sys.argv[1],'rb')); print('ZZUNPICKLE_OK_WITHOUT_SHIM')\" $P 2>&1 | tail -3")
sh("winpath_shim", f"P=$(find {one} -name metric_cache.pkl | head -1); PYTHONPATH={V11} {NVPY} -c \"import pathlib; "
                   "pathlib.WindowsPath = pathlib.PureWindowsPath; import lzma,pickle,sys; "
                   "m = pickle.load(lzma.open(sys.argv[1],'rb')); print('ZZUNPICKLE_OK_WITH_SHIM', type(m.file_path).__name__, "
                   "type(m).__name__)\" $P 2>&1 | tail -3")

# ---- 4. a real harness run on Linux: W3's harness, the dev box's e2chk seam + metric cache, 25 tokens / 1 log -----
mc = f"{W}/exp/metric_cache_navtest"
sh("harness_prep", f"mkdir -p {W}/exp && cp -r {B}/exp/metric_cache_navtest {W}/exp/ && mkdir -p {mc}/metadata && "
                   f"(echo file_name; find {mc} -name metric_cache.pkl | sort) > {mc}/metadata/metric_cache_navtest_metadata_node_0.csv && "
                   f"wc -l {mc}/metadata/*.csv && mkdir -p {W}/openscene/navsim_logs/test && "
                   f"cp {B}/navsim_logs/*.pkl {W}/openscene/navsim_logs/test/ && ls -la {W}/openscene/navsim_logs/test")
sh("harness_run", f"cd {W} && {NVPY} {B}/linux_score_v1.py --label refe_navplan_e2chk_linux "
                  f"--seam {B}/seam/refe_e2chk_newvenv.npz --tokens {B}/seam/e2_tokens.json --out {W}/score "
                  f"--w3-code {B}/w3code --e1-wrapper {B}/e1/navsim_win.py --nv-py {NVPY} --v11-tree {V11} "
                  f"--nuplan-tree {NUP} --exp {W}/exp --openscene {W}/openscene --windowspath-shim 2>&1 | tail -5",
   timeout=900)
csv_vm = f"{W}/score/refe_navplan_e2chk_linux/refe_navplan_e2chk_linux.csv"
cmp = {"vm_csv_exists": os.path.exists(csv_vm)}
if os.path.exists(csv_vm):
    import csv
    a = {r["token"]: r for r in csv.DictReader(open(csv_vm)) if r["token"] != "average"}
    b = {r["token"]: r for r in csv.DictReader(open(f"{B}/seam/refe_e2chk_newvenv.csv")) if r["token"] != "average"}
    terms = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
             "comfort", "driving_direction_compliance", "score")
    diffs = {t: max(abs(float(a[t][k]) - float(b[t][k])) for k in terms) for t in a if t in b}
    cells_equal = sum(1 for t in a if t in b for k in terms if a[t][k] == b[t][k])
    cmp.update(n_vm=len(a), n_dev=len(b), same_tokens=set(a) == set(b), max_abs_diff=max(diffs.values()) if diffs else None,
               tokens_with_any_diff=sum(1 for v in diffs.values() if v > 0), cells_textually_equal=cells_equal,
               cells_total=len(a) * len(terms), vm_csv_sha256=sha256(csv_vm),
               dev_csv_sha256=sha256(f"{B}/seam/refe_e2chk_newvenv.csv"))
    cnt = f"{W}/score/refe_navplan_e2chk_linux/refe_navplan_e2chk_linux.counts.json"
    if os.path.exists(cnt):
        cj = json.load(open(cnt))
        cmp["counts"] = {k: cj.get(k) for k in ("status", "log_successful", "csv_valid_rows", "C1_max_abs_delta",
                                                "summary_x100_4dp", "wall_s", "failures", "imported_from", "patches",
                                                "resources", "agent_calls")}
res["harness_compare"] = cmp
save()
print("ZZHARNESS " + json.dumps({k: v for k, v in cmp.items() if k != "counts"}), flush=True)

# ---- 5. the seam venv (Python 3.11): the dev box's 117 pins, torch swapped to its CPU build FOR THIS SMOKE ONLY ----
sh("driverl_pins", f"grep -v -E '^-e |^(colorama|torch)==' {B}/driverl_eval_venv_freeze.txt > {W}/driverl_pins_linux.txt; "
                   f"wc -l {W}/driverl_pins_linux.txt")
sh("driverl_venv", f"{UV} venv {DV} --python 3.11 && "
                   f"{UV} pip install --python {DVPY} --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu' && "
                   f"{UV} pip install --python {DVPY} --no-deps -r {W}/driverl_pins_linux.txt && "
                   f"{UV} pip install --python {DVPY} --no-deps -e {DZ}/DriveRL -e {DZ}/DriveRL/nuplan-devkit && "
                   f"{DVPY} --version && du -sb {DV}", timeout=1200)
sh("seam_selftest", f"cd {B}/pkg/eval && {DVPY} refe_navtest_seam.py --selftest 2>&1 | tail -6")
sh("seam_imports", f"cd {B}/pkg/eval && {DVPY} -c \"import sys, os; sys.path.insert(0, '../refe'); sys.path.insert(0, '../code'); "
                   "import torch, planner, model, navtrain_scenarios, calib_table, ckpt_io, load_dinov3, augment_routes, cv2; "
                   "from nuplan.planning.script import driverl_runtime_map_features as M; "
                   "from driverl.datatypes.goal_position_utils import route_goal_positions; "
                   "m = model.REFe(model.REFeConfig.for_backbone('vitl16')); "
                   "n = sum(p.numel() for p in m.parameters()); t = sum(p.numel() for p in m.parameters() if p.requires_grad); "
                   "print('ZZSEAM_IMPORTS', torch.__version__, cv2.__version__, n, t, planner.REFePlanner.DEFAULT_RULE, "
                   "planner.REFePlanner.REPAIR_LAST_HEADING)\" 2>&1 | tail -4", timeout=600)
sh("jpeg_decode_parity", f"cd {B} && {DVPY} -c \"import cv2, numpy as np, hashlib, zipfile, json; "
                         "z = zipfile.ZipFile('frames/probe.zip'); out = {}; "
                         "[out.__setitem__(n, hashlib.sha256(np.ascontiguousarray(cv2.resize(cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR), (960, 512))[:, :, ::-1]).tobytes()).hexdigest()) for n in sorted(z.namelist())]; "
                         "print('ZZJPEG ' + json.dumps(out))\" 2>&1 | tail -2")

# ---- 6. link speed from THIS VM (each a 50 MiB range read, discarded) ---------------------------------------------
sh("s3_speed", "curl -sS -r 0-52428799 -o /dev/null -w 'S3 %{http_code} %{size_download} B %{speed_download} B/s %{time_total} s\\n' "
               "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip", timeout=300)
sh("hf_speed", "curl -sSL -r 0-52428799 -o /dev/null -w 'HF %{http_code} %{size_download} B %{speed_download} B/s %{time_total} s\\n' "
               "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_0.tgz",
   timeout=300)
res["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("ZZSMOKE_DONE " + RES, flush=True)
