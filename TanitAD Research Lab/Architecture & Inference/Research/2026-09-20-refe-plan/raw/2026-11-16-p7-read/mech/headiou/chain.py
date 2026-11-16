"""Detached chain for the P7 head-IoU read.  Run with C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe chain.py   (log: chain.log, sentinel: chain.DONE or chain.FAIL).
Steps: commit gate -> head weight proof (CPU) -> determinism on 2 tokens (twice, GPU lock) -> full 1,058 forward (GPU lock, ctx cache) -> readout -> per-token IoU.
Each GPU step takes the lock itself (instrument's acquire_gpu_lock, released in its finally); exit 5 (GPU busy) is retried."""
import ctypes
import json
import os
import subprocess
import sys
import time

HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/headiou"
SCR = "F:/tanitad_scratch/p7headiou"
TREE = SCR + "/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
CK = "F:/TanitAD_p7read/run/model_final.pt"
sys.path.insert(0, TREE + "/eval")
import eval_checkpoint as EC  # noqa: E402

LOG = open(HOLD + "/chain.log", "a", encoding="utf-8", buffering=1)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)


def env():
    e = EC.env_driverl()
    for k, v in list(e.items()):
        if isinstance(v, str) and "D:/Projects/TanitAD" in v:
            e[k] = v.replace("D:/Projects/TanitAD", "E:/Projects/TanitAD")
    e["REFE_NUPLAN_DB_ROOT"] = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
    e["REFE_NAVTRAIN_YAML"] = "F:/Archive/devbox-C/navsim/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    e["PYTHONPATH"] = e["PYTHONPATH"] + ";" + HOLD
    return e


def avail_commit_gb():
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 2 ** 30


def wait_commit(need=6.0, tmax=4 * 3600):
    t0 = time.time()
    while avail_commit_gb() < need:
        if time.time() - t0 > tmax:
            fail("commit never reached %.1f GB" % need)
        say("waiting commit", round(avail_commit_gb(), 2))
        time.sleep(60)
    say("commit ok", round(avail_commit_gb(), 2))


def fail(msg):
    say("FAIL", msg)
    open(HOLD + "/chain.FAIL", "w").write(msg)
    sys.exit(1)


def run(cmd, tag, extra_env=None, ok=(0,), retry5=40):
    for att in range(retry5):
        wait_commit()
        e = env()
        e.update(extra_env or {})
        say("RUN", tag, "attempt", att + 1)
        with open(f"{HOLD}/logs/{tag}.log", "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, cwd=HOLD, env=e, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        say("RC", tag, rc)
        if rc in ok:
            return rc
        if rc == 5:
            time.sleep(120)
            continue
        fail(f"{tag} rc {rc}")
    fail(f"{tag}: GPU busy after {retry5} attempts")


os.makedirs(HOLD + "/logs", exist_ok=True)
fwd_common = ["--ckpt", CK, "--frames", "E:/Projects/TanitAD/data/refe_navtest/frames", "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
              "--wait-idle-timeout", "1500"]
heads = ["--heads", f"H1-s0={CK}", f"H1-s1=ZERO::{CK}", f"H1-s2=FRESH::{CK}"]

run([PY, HOLD + "/proof_head.py", HOLD + "/proof"], "proof_head", {"CUDA_VISIBLE_DEVICES": "-1"})
import shutil  # noqa: E402
for d in ("detA", "detB"):
    run([PY, HOLD + "/p7_adapter.py", "--out", f"{HOLD}/{d}", "--ctx-cache", f"{SCR}/{d}_ctx", "--limit-tokens", "2"] + fwd_common + heads, d, {"P7_PROOF_DIR": f"{HOLD}/{d}"}, ok=(0, 2))
run([PY, HOLD + "/compare_det.py", HOLD + "/detA", HOLD + "/detB", f"{SCR}/detA_ctx", f"{SCR}/detB_ctx", HOLD + "/proof/proof_determinism.json"], "compare_det", ok=(0,))
run([PY, HOLD + "/p7_adapter.py", "--out", HOLD + "/main", "--ctx-cache", SCR + "/ctx_full"] + fwd_common + heads, "main", {"P7_PROOF_DIR": HOLD + "/main"}, ok=(0, 2))
ro = [PY, TREE + "/eval/aux_s1_readout.py", "perception", "--pred"] + [f"H1-s{k}={HOLD}/main/head_pred_H1-s{k}.npz" for k in range(3)] + [
    "--pred-init", f"H1-s0={HOLD}/main/head_pred_H1-s2.npz", "--labels-set", "navtest", "--labels", "E:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest",
    "--train-store", "E:/Projects/TanitAD-artifacts/aux_drivable/label_store", "--dac-csv", TREE + "/raw/p2_r2_instrument/baseline/u400_navtest_full.csv",
    "--c3-table", TREE + "/raw/2026-10-11-c3-selection-headroom/table.npz", "--c3-tokens", TREE + "/raw/2026-10-11-c3-selection-headroom/tokens.json", "--out", HOLD + "/readout_perception.json"]
run(ro, "readout", {"CUDA_VISIBLE_DEVICES": "-1"})
run([PY, HOLD + "/token_iou.py", HOLD + "/token_iou.json"] + [f"H1-s{k}={HOLD}/main/head_pred_H1-s{k}.npz" for k in range(3)], "token_iou", {"CUDA_VISIBLE_DEVICES": "-1"})
open(HOLD + "/chain.DONE", "w").write("ok")
say("DONE")
