"""STEP 8 — record WHICH interpreter ran the 86, and prove the load-bearing venv
was not disturbed.

⛔ WHY THIS IS AN ARTIFACT AND NOT A NOTE. `CLAUDE.md`: *"`uv pip install
<anything>` CAN SILENTLY REPLACE TORCH WITH A WHEEL THE DRIVER CANNOT RUN"* —
MEASURED twice on pod4, where `accelerate` and then `compressed-tensors` each
dragged torch off the default PyPI index and broke CUDA for every job on the box.
Neither command named torch. The dev box's GPU is load-bearing (frozen-trunk
probes; the only GPU not running the live 30k on Thor), so "I was careful" is not
an admissible claim — the state of both interpreters has to be a MEASURED record
that a later reader can check.

⭐ THE ISOLATION CHOSEN, AND WHY. sam3's closure was installed into a SEPARATE
venv (`C:/Users/Admin/venvs/sam3run`), which reaches torch through a single
`.pth` line pointing at the tanitad venv's site-packages:

    sam3run/Lib/site-packages/_zz_tanitad_base.pth
        -> C:/Users/Admin/venvs/tanitad/Lib/site-packages

so torch/torchvision/numpy/av resolve to the SAME FILES the probes use, while
every `pip install` writes only into sam3run. That makes the protection
STRUCTURAL rather than procedural: it does not depend on anyone remembering
`--no-deps` on every future command, and `rm -rf` of one directory restores the
box exactly. (`--no-deps` was used on every install anyway — belt and braces.)

⚠️ Verification is a REAL `conv2d` on CUDA, in BOTH interpreters, because cuBLAS
can succeed while cuDNN/conv is broken — `import torch` and
`torch.cuda.is_available()` are not evidence.

usage:  python f8_env_probe.py --out raw/f7_env_local_gpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

TANITAD = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
SAM3RUN = r"C:\Users\Admin\venvs\sam3run\Scripts\python.exe"

#: What a healthy tanitad venv must still report. The dev box was MEASURED at
#: these values BEFORE anything was installed (2026-08-17), so a later drift is
#: detectable rather than merely suspected.
BASELINE = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
            "cudnn": 91900, "device": "NVIDIA GeForce RTX 4060"}

PROBE = r"""
import json, sys
out = {"executable": sys.executable, "python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_file"] = torch.__file__
    out["cuda_build"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version()
    out["is_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        out["sm"] = "sm_%d%d" % cap
        out["total_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
        # THE check: a real conv2d, not is_available()
        x = torch.nn.functional.conv2d(
            torch.randn(1, 3, 16, 16, device="cuda"),
            torch.randn(4, 3, 3, 3, device="cuda"))
        torch.cuda.synchronize()
        out["cuda_conv2d_ok"] = (tuple(x.shape) == (1, 4, 14, 14))
        # and a matmul, so a cuBLAS-only success cannot masquerade as health
        y = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
        torch.cuda.synchronize()
        out["cuda_matmul_ok"] = tuple(y.shape) == (64, 64)
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
try:
    import torchvision
    out["torchvision"] = torchvision.__version__
except Exception as e:
    out["torchvision"] = "ERR %s" % type(e).__name__
for mod in ("numpy", "sam3", "open_clip", "timm", "triton", "einops",
            "cv2", "decord", "av", "imageio_ffmpeg", "pycocotools"):
    try:
        m = __import__(mod)
        out.setdefault("modules", {})[mod] = {
            "version": getattr(m, "__version__", "n/a"),
            "from_tanitad": "venvs\\tanitad" in (getattr(m, "__file__", "") or "")}
    except Exception as e:
        out.setdefault("modules", {})[mod] = {"version": None,
                                              "error": type(e).__name__}
print("JSONSTART" + json.dumps(out) + "JSONEND")
"""


def probe(py: str) -> dict:
    if not os.path.exists(py):
        return {"executable": py, "error": "interpreter not found"}
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="6")
    p = subprocess.run([py, "-c", PROBE], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=600)
    txt = p.stdout or ""
    if "JSONSTART" not in txt:
        return {"executable": py, "error": "probe produced no JSON",
                "tail": (p.stdout + p.stderr)[-500:]}
    return json.loads(txt.split("JSONSTART", 1)[1].split("JSONEND", 1)[0])


def pip_list(py: str) -> list:
    if not os.path.exists(py):
        return []
    p = subprocess.run([py, "-m", "pip", "list", "--format=json"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600)
    try:
        return json.loads(p.stdout)
    except Exception:                                            # noqa: BLE001
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    tan, s3 = probe(TANITAD), probe(SAM3RUN)
    drift = {k: {"baseline": v, "now": tan.get(k)}
             for k, v in BASELINE.items() if tan.get(k) != v}

    out = {
        "class": "MEASURED", "ts": __import__("time").strftime("%Y-%m-%d"),
        "what": "which interpreter ran the 86, and whether the load-bearing "
                "tanitad venv was disturbed by installing sam3's closure",
        "isolation": {
            "strategy": "separate venv + one .pth back to tanitad's "
                        "site-packages, so torch is SHARED BY PATH and never "
                        "reinstalled; every pip write lands in sam3run only",
            "run_venv": SAM3RUN, "protected_venv": TANITAD,
            "pth": "sam3run/Lib/site-packages/_zz_tanitad_base.pth",
            "all_installs_no_deps": True,
            "reversible_by": "rm -rf C:/Users/Admin/venvs/sam3run"},
        "tanitad_baseline_before_any_install": BASELINE,
        "tanitad_now": tan,
        "tanitad_DRIFT": drift,
        "TANITAD_UNDISTURBED": bool(
            not drift and tan.get("cuda_conv2d_ok") and tan.get("cuda_matmul_ok")),
        "sam3run_now": s3,
        "SAM3RUN_READY": bool(
            s3.get("cuda_conv2d_ok")
            and (s3.get("modules", {}).get("sam3", {}).get("version"))),
        "sam3run_packages": pip_list(SAM3RUN),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[env] tanitad torch {tan.get('torch')} conv2d "
          f"{tan.get('cuda_conv2d_ok')} · UNDISTURBED "
          f"{out['TANITAD_UNDISTURBED']}")
    print(f"[env] sam3run torch {s3.get('torch')} (from "
          f"{'tanitad' if 'tanitad' in (s3.get('torch_file') or '') else '?'}) "
          f"· sam3 {s3.get('modules', {}).get('sam3', {}).get('version')} · "
          f"READY {out['SAM3RUN_READY']}")
    if drift:
        print("⛔ TORCH DRIFT DETECTED:", json.dumps(drift))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
