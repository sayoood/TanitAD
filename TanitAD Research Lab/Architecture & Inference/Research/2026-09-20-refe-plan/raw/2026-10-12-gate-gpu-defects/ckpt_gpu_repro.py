"""Reproduce G-CKPT's pod failure on the dev-box GPU, then test a deterministic-CUDA bootstrap for its three training runs.
Usage: python ckpt_gpu_repro.py <refe dir> <work root>"""
import json
import os
import subprocess
import sys
from pathlib import Path

REFE, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REFE))
os.chdir(REFE)
import gate_checks as C          # noqa: E402
import gate_fixture as GF        # noqa: E402

fx = GF.tiny_p2_fixture(ROOT / "fx")
argv = [a for a in GF.rehearsal_argv(fx, str(ROOT / "smoke"), steps=4) if a != "--cpu"] + ["--amp", "bf16"]
prof = {"ckpt_steps": 4}

DET_BOOT = ("import sys, runpy, torch\n"
            "torch.use_deterministic_algorithms(True)\n"
            "torch.backends.cudnn.deterministic = True\n"
            "torch.backends.cudnn.benchmark = False\n"
            "if torch.cuda.is_available():\n"
            "    torch.backends.cuda.enable_flash_sdp(False)\n"
            "    torch.backends.cuda.enable_mem_efficient_sdp(False)\n"
            "    torch.backends.cuda.enable_math_sdp(True)\n"
            "script = sys.argv[1]\n"
            "sys.argv = sys.argv[1:]\n"
            "runpy.run_path(script, run_name='__main__')\n")


def det_train_sub(args, log):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env.pop(C.ARM_ENV, None)
    r = subprocess.run([sys.executable, "-c", DET_BOOT, str(REFE / "train.py")] + args, cwd=str(REFE), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    Path(log).write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode


out = {}
v = C.check_ckpt(argv, ROOT / "plain", prof)
out["plain"] = {"verdict": v.get("verdict"), "reasons": v.get("reasons"), "loss_eq": v.get("details", {}).get("loss_trajectory_equal")}
orig = C._train_sub
C._train_sub = det_train_sub
v = C.check_ckpt(argv, ROOT / "det", prof)
out["det"] = {"verdict": v.get("verdict"), "reasons": v.get("reasons"), "loss_eq": v.get("details", {}).get("loss_trajectory_equal")}
v = C.check_ckpt(argv, ROOT / "det_dropopt", prof, arm="resume_drops_opt")
out["det_regression_resume_drops_opt"] = {"verdict": v.get("verdict"), "reasons": [r[:120] for r in v.get("reasons", [])]}
C._train_sub = orig
print(json.dumps(out, indent=1, default=str))
for tag in ("plain", "det"):
    for lg in ("U.log", "H1.log", "H2.log"):
        p = ROOT / tag / lg
        if p.exists():
            t = p.read_text(encoding="utf-8", errors="replace")
            bad = [l for l in t.splitlines() if "Error" in l or "error" in l][-2:]
            if bad:
                print(tag, lg, bad)
