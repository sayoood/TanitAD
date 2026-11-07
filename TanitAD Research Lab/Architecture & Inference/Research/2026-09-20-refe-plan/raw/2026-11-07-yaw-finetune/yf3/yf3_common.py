"""YF3 shared constants and helpers. Read-only imports of YF2 (gate code), YT (shim) and the PK eval package; nothing outside YF3 is written."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
Y2 = Path("E:/Projects/TanitAD-artifacts/yawrate_wp/YF2")
PKG = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
EVAL = PKG / "eval"
D = Path("E:/Projects/TanitAD/data/refe_navtest")
YT_REFE = Path("C:/Users/Admin/pxyaw/refe")
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"      # torch, nuPlan, numpy: seam, scoring, training, analysis
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"            # families6 / taniteval adapter (as stage2_sft.py)
HARNESS = EVAL / "harness"
EV6E = HARNESS / "ev6e"
TABLE_TEST = Y2 / "pinned" / "ego_yaw_navtest.json.gz"
TABLE_TRAIN = Y2 / "pinned" / "ego_yaw_navtrain_devbox.json.gz"
TABLE_TEST_SHA = "f5ebe24fd05a8067215c22f08f873808a7020a97eca6940fdcc568f8b39e29a5"
TABLE_TRAIN_SHA = "517c6413fe3569534ec6ca05dff558b0ea7c2266d5ac7deef308a0bf5ceb528a"
PREREG = Y2 / "PREREG_YAWFT.md"
PREREG_SHA = "d28a630fa650a8fe50488dea82f01e1d61a5798347d7f522926b2b8215145b74"
U400 = Path("E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt")
A5_TOKENS = D / "a5_confirm_tokens.json"
BANKED_SEAM = D / "seams" / "stage2_sft4_B_u400.npz"
BANKED_CSV = D / "score" / "refe_stage2_sft4_B_u400" / "refe_stage2_sft4_B_u400.csv"
BANKED_FAMILIES = PKG / "raw" / "2026-10-05-sft-stage2" / "sft4_B_u400" / "sft4_B_u400_families.json"
BANKED_RESULT = PKG / "raw" / "2026-10-05-sft-stage2" / "sft4_B_u400" / "result.json"
BANKED_DEPLOYED_CSV = D / "score" / "refe_navtest_final_routefix" / "refe_navtest_final_routefix.csv"
CACHE = Y2 / "cache_u400_4k"
MANIFEST = Y2 / "manifest_yaw_4k.json"
BOOT_N, BOOT_SEED = 10000, 20260927
YAW_SCALE = 10.0


def navsim_root() -> str:
    r = os.environ.get("TANITAD_NAVSIM_ROOT")
    cands = [r.replace("\\", "/").rstrip("/")] if r else [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit("REFUSING: navsim archive root not found; set TANITAD_NAVSIM_ROOT")


def export_path() -> str:
    return navsim_root() + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(p, chunk: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tail(path, n=2500) -> str:
    try:
        return Path(path).read_bytes()[-n:].decode("utf-8", "replace")
    except OSError:
        return ""
