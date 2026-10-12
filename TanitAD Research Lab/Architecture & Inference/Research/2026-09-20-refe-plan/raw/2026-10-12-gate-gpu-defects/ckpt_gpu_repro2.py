"""Variant: deterministic algorithms WITHOUT forcing the math SDPA path (PyTorch picks the attention kernel). Same fixture as repro 1."""
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


C._train_sub = det_train_sub
out = {}
for tag, arm in (("det_fast", None), ("det_fast_dropopt", "resume_drops_opt")):
    v = C.check_ckpt(argv, ROOT / tag, prof, arm=arm)
    out[tag] = {"reasons": [r[:140] for r in v.get("reasons", [])], "loss_eq": v.get("details", {}).get("loss_trajectory_equal")}
print(json.dumps(out, indent=1, default=str))
for lg in ("U.log", "H1.log", "H2.log"):
    p = ROOT / "det_fast" / lg
    if p.exists():
        t = p.read_text(encoding="utf-8", errors="replace")
        hits = [l[:200] for l in t.splitlines() if "determin" in l.lower() or "Error" in l][-3:]
        if hits:
            print(lg, hits)
