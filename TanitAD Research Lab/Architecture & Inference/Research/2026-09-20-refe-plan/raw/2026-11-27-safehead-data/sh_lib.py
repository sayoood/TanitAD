"""L2 SAFE-HEAD data cache -- shared constants and helpers (stdlib + numpy only at import).

Layout (all re-creatable except the final cache `out/`):
    W/frames/<log>/<CAM>/<hash>.jpg      extracted frames (deleted after a log's record is verified if F: is tight)
    W/plan/PLAN.json, need_<archive>.json, tokens_<log>.json
    W/dump/<log>.dump.npz                 GPU stage: proposals, s, logits, pick
    W/lab/<log>.lab.npz                   CPU stage: tracked PDM labels
    W/out/<log>.npz + W/index.csv         THE CACHE (contract: DATA_CONTRACT.md)
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
import zipfile

import numpy as np

W = os.environ.get("SH_WORK", "F:/safehead_cache")
TIP = "F:/TanitAD_p5cread_tip"
PK = f"{TIP}/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
NAV = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
PY_N = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
YAML = "F:/Archive/devbox-C/navsim/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
COMMANDS = "E:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json"
YAW_TRAIN = "E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtrain_devbox.json.gz"
YAW_TEST = "E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz"
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
N_PROP, N_STEP, S_DIM, N_HEAD = 64, 20, 256, 6
LAB_KEYS = ("tr_nc", "tr_dac", "tr_ddc", "tr_ttc", "tr_ep", "tr_c", "tr_pdms")
LAB_MAP = {"tr_nc": "nc", "tr_dac": "dac", "tr_ddc": "ddc", "tr_ttc": "ttc", "tr_ep": "ep", "tr_c": "c", "tr_pdms": "pdms"}
C1_SHA = "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4"
C1_CKPT = "F:/safehead_cache/ckpt/C1/model_final.pt"
PREFIX = "openscene-v1.1/sensor_blobs/trainval/"
HF_BASE = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/"
HIST_ROWS, FUT_ROWS = 40, 100            # the labeller's scenario window (onpolicy_relabel_pdm.label_log)


def wp(*a):
    return os.path.join(W, *a).replace("\\", "/")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_yaml_split(path=YAML):
    import re
    y = open(path, encoding="utf-8").read()
    i = y.index("\ntokens:")
    logs = re.findall(r"^  - '([^']+)'", y[:i], re.M)
    toks = re.findall(r"^  - '([^']+)'", y[i:], re.M)
    return logs, toks


def db_path(log, root=DB_ROOT):
    """Local nuPlan DB of a log: splits/<log>.db, then splits/{trainval,test,mini}/ (the yaw-table builder's order)."""
    for sub in ("", "trainval", "test", "mini"):
        p = os.path.join(root, sub, f"{log}.db")
        if os.path.exists(p):
            return p.replace("\\", "/")
    return None


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def lidar_rows(path):
    c = ro(path)
    try:
        return c.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc ORDER BY timestamp").fetchall()
    finally:
        c.close()


def cam_index(path, cams=CAMS):
    """{cam: (np.int64 timestamps, [filename_jpg])} from the DB image table (planner.FrameResolver._index's query)."""
    c = ro(path)
    try:
        out = {}
        for ch in cams:
            row = c.execute("SELECT token FROM camera WHERE channel=?", (ch,)).fetchone()
            rs = c.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token=? ORDER BY timestamp", (row[0],)).fetchall() if row else []
            out[ch] = (np.array([r[0] for r in rs], np.int64), [r[1] for r in rs])
        return out
    finally:
        c.close()


def frame_names(path, tokens, max_dt_ms=120.0):
    """{token: [rel jpg per cam] or None} -- nearest frame per camera within max_dt_ms of the lidar_pc timestamp (FrameResolver.resolve's rule)."""
    ts = dict(lidar_rows(path))
    ci = cam_index(path)
    res = {}
    for t in tokens:
        t0, rel = ts[t], []
        for ch in CAMS:
            a, names = ci[ch]
            if a.size == 0:
                rel = None
                break
            j = int(np.argmin(np.abs(a - t0)))
            if abs(int(a[j]) - t0) / 1000.0 > max_dt_ms:
                rel = None
                break
            rel.append(names[j])
        res[t] = rel
    return res


def guard_tokens(path, tokens, hist=HIST_ROWS, fut=FUT_ROWS):
    """Tokens whose row index leaves room for the labeller's window (NS.build_scenarios_for_log's margin guard, strict form)."""
    rows = lidar_rows(path)
    pos = {t: i for i, (t, _) in enumerate(rows)}
    n = len(rows)
    return [t for t in tokens if t in pos and pos[t] > hist and pos[t] + fut < n]


def save_npz_det(path, **arrays):
    """np.savez with FIXED zip timestamps, so two runs on equal arrays give identical BYTES (np.savez stamps the wall clock)."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            z.writestr(zi, buf.getvalue())
    os.replace(tmp, path)


def load_json_gz(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


def say(msg, f=None):
    import time
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    if f:
        with open(f, "a", encoding="utf-8") as fo:
            fo.write(line + "\n")
