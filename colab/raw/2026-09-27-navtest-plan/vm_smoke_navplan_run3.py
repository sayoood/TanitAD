# VM-side CPU smoke, RUN 3 -- run 2's `seam_imports` died on the Colab KERNEL's own environment:
# the kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline, every subprocess inherits it, and the seam
# venv has no matplotlib_inline (the NAVSIM venv does, which is why run 1's harness imports passed). This run repeats
# ONLY the seam-venv import test with MPLBACKEND=Agg, finds which import pulls matplotlib in, and builds the ViT-L REFe.
import json
import os
import subprocess
import time

W = "/content/navplan"
RES = f"{W}/results_run3.json"
os.makedirs(W, exist_ok=True)
res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {},
       "kernel_MPLBACKEND": os.environ.get("MPLBACKEND")}


def save():
    with open(RES + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(RES + ".tmp", RES)


def sh(name, cmd, timeout=900):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
    res["steps"][name] = {"rc": p.returncode, "s": round(time.time() - t, 1), "stdout_tail": p.stdout[-3000:],
                          "stderr_tail": p.stderr[-1500:]}
    save()
    print(f"ZZSTEP {name} rc={p.returncode} s={res['steps'][name]['s']}", flush=True)


B, SRC, DV = f"{W}/bundle", f"{W}/src", f"{W}/venv_driverl"
DVPY = f"{DV}/bin/python"
DZ = f"{SRC}/DriveZero-24959547edac549d64c34b2aab62d5c52599d078"
save()
print("ZZKERNEL_MPLBACKEND", res["kernel_MPLBACKEND"], flush=True)
sh("unpack_bundle", f"mkdir -p {B} && tar -xzf /content/navplan_bundle.tgz -C {B} && find {B} -type f | wc -l")
sh("python_install", "uv python install 3.11.13 2>&1 | tail -1")
sh("code_download", f"mkdir -p {SRC} && cd {SRC} && curl -sSL -o DriveZero.zip "
   "https://codeload.github.com/XiaomiAutoL3/DriveZero/zip/24959547edac549d64c34b2aab62d5c52599d078 && unzip -q -o DriveZero.zip")
sh("driverl_venv", f"grep -v -E '^-e |^(colorama|torch)==' {B}/driverl_eval_venv_freeze.txt > {W}/driverl_pins_linux.txt && "
                   f"uv venv {DV} --python 3.11.13 && "
                   f"uv pip install --python {DVPY} --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu' && "
                   f"uv pip install --python {DVPY} --no-deps -r {W}/driverl_pins_linux.txt && "
                   f"uv pip install --python {DVPY} --no-deps -e {DZ}/DriveRL -e {DZ}/DriveRL/nuplan-devkit", timeout=1200)
probe = r'''
import sys, time, importlib
sys.path.insert(0, "../refe"); sys.path.insert(0, "../code")
order = ["torch", "cv2", "model", "calib_table", "ckpt_io", "load_dinov3", "navtrain_scenarios", "planner",
         "nuplan.planning.script.driverl_runtime_map_features", "augment_routes", "driverl.datatypes.goal_position_utils",
         "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario"]
first_mpl = None
for m in order:
    importlib.import_module(m)
    if first_mpl is None and "matplotlib" in sys.modules:
        first_mpl = m
import torch, model, planner
t = time.time(); r = model.REFe(model.REFeConfig.for_backbone("vitl16")); bs = time.time() - t
n = sum(p.numel() for p in r.parameters()); tr = sum(p.numel() for p in r.parameters() if p.requires_grad)
import json
print("ZZSEAM_IMPORTS " + json.dumps({"torch": torch.__version__, "first_module_pulling_matplotlib": first_mpl,
      "params_total": n, "params_trainable": tr, "build_s": round(bs, 1), "rule": planner.REFePlanner.DEFAULT_RULE,
      "repair_last_heading": planner.REFePlanner.REPAIR_LAST_HEADING}))
'''
open(f"{W}/import_probe.py", "w").write(probe)
sh("seam_imports_default_env", f"cd {B}/pkg/eval && {DVPY} {W}/import_probe.py 2>&1 | tail -2")
sh("seam_imports_mpl_agg", f"cd {B}/pkg/eval && MPLBACKEND=Agg {DVPY} {W}/import_probe.py 2>&1 | tail -3", timeout=600)
res["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("ZZSMOKE3_DONE " + RES, flush=True)
