"""WP YAW / stream YF2: shared constants and small helpers (copies live here; PK and YT's work tree are only READ).

YAW_SCALE: MM decision 2026-11-06 `ego[4] = YAW_SCALE * angular_rate_z`, YAW_SCALE = 10.0, defined by stream YT in its work tree
(`C:/Users/Admin/pxyaw/refe/`). At the time of writing YT's `ego_yaw_table.py` / `ego_yaw.py` did NOT define it (MEASURED: grep -c YAW_SCALE = 0 in both),
so it was defined here with a TODO. YT BANKED IT at ~23:50 (ego_yaw_table.YAW_SCALE = 10.0, and a scaled EgoYawShim in ego_yaw.py): `import_yt_scale()` now finds theirs and `import_yt_scale()` prefers theirs and `yaw_scale_source()` says so; a DISAGREEING value raises.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
REFE_DIR = Path(os.environ.get("YF2_REFE_DIR", str(PK / "refe")))          # READ-ONLY source of model.py / train.py / ckpt_io.py / measures.py
YT_REFE = Path("C:/Users/Admin/pxyaw/refe")                                   # YT's work tree: READ-ONLY (ego_yaw_table.py, ego_yaw.py)
DATA = Path("E:/Projects/TanitAD/data")
BANK = DATA / "refe_navtrain10"
PIXELS = DATA / "refe_proxy" / "pixels"
CALIB = DATA / "refe_proxy" / "calib_table_pod_train_grow.json"
U400 = DATA / "refe_runs_eval" / "sft4_peek" / "model_sft4_B_u400.pt"
U400_SHA256 = "8b4dba48"                                                       # prefix, MM brief
PROXY_MANIFEST = PK / "raw" / "2026-09-27-training-measures" / "proxy_manifest_10k.json"
TABLE_NAVTRAIN = HERE / "pinned" / "ego_yaw_navtrain_devbox.json.gz"       # PINNED COPY of YT's table (YT rebuilt it twice in an hour: same values, new bytes)
TABLE_NAVTEST = HERE / "pinned" / "ego_yaw_navtest.json.gz"
A5_TOKENS = PK / "eval" / "raw" / "a5_confirm" / "a5_confirm_tokens.json"

YAW_SCALE = 10.0            # TODO(YT): import from C:/Users/Admin/pxyaw/refe/ego_yaw_table.py once YT defines it
YAW_TAG = "nuplan_angular_rate_z"
EGO_YAW_SLOT = 4
DEC = ("ego_enc.", "queries", "dec.", "traj_head.")                           # the proxy trainer's trainable decoder side


def import_yt_scale():
    """YT's YAW_SCALE if their module defines it, else None. A value different from ours is a hard error (two sources for one constant)."""
    v = None
    try:
        sys.path.insert(0, str(YT_REFE))
        import ego_yaw_table as EYT                                           # noqa: WPS433
        v = getattr(EYT, "YAW_SCALE", None)
    except Exception:
        v = None
    finally:
        if str(YT_REFE) in sys.path:
            sys.path.remove(str(YT_REFE))
    if v is not None and float(v) != YAW_SCALE:
        raise RuntimeError(f"YAW_SCALE disagreement: YT defines {v}, YF2 defines {YAW_SCALE}")
    return v


def yaw_scale_source() -> str:
    return "YT:ego_yaw_table.YAW_SCALE" if import_yt_scale() is not None else "local-definition (TODO import from YT; MM brief 2026-11-06 = 10.0)"


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def code_hashes(names=("model.py", "train.py", "ckpt_io.py", "measures.py")) -> dict:
    return {n: sha256_file(REFE_DIR / n) for n in names if (REFE_DIR / n).exists()}


def load_yaw_table(path):
    """{token: raw angular_rate_z rad/s}, meta, file sha256 -- through YT's loader (refuses a malformed file)."""
    sys.path.insert(0, str(YT_REFE))
    try:
        import ego_yaw_table as EYT
        return EYT.load(str(path))
    finally:
        sys.path.remove(str(YT_REFE))


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
