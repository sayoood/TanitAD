"""D6 P4 common: paths, readers (fan, seam, masks, official frames) and the gate geometry.   numpy only (both venvs).

Every reader asserts the SHAPE of what it read; a missing or torn record is an error, never an absence.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKGD6 = os.path.dirname(HERE)
RAW = os.path.join(PKGD6, "raw")
NAV = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim"
M50 = f"{NAV}/raw/milestones/step50400"
SEAM_A1 = f"{M50}/bridge_navhard/seam_R7_A1.npz"
ROWS_A1 = f"{M50}/bridge_navhard/rows_R7_A1.jsonl"
FRAME = f"{M50}/scores_navhard/score_{{arm}}__navhard_two_stage_wrapper/{{arm}}__navhard_two_stage_final_scores_frame.csv"
PRE = f"{M50}/scores_navhard/score_{{arm}}__navhard_two_stage_wrapper/{{arm}}__navhard_two_stage_final_scores_frame_PRE_AGGREGATION.csv"
SUMMARY = f"{M50}/scores_navhard/score_{{arm}}__navhard_two_stage.csv"
COUNTS = f"{M50}/scores_navhard/score_{{arm}}__navhard_two_stage.counts.json"
INPUTS = "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"
FAN = os.path.join(RAW, "p1x_fan", "fan_R7_A1.jsonl")
N_FAN = 117
N_TOK = 5912

# ---- the map head window (raw/p4_impl_choices.md s2) ------------------------------------------
H, W = 1000, 600
CELL = 0.1
X_MAX, Y_HALF = 100.0, 30.0
LV_HW = (400, 240)
THRESH = 0.5
#: get_pacifica_parameters() (nuplan vehicle_parameters.py:125-138), the scorer's vehicle
VEH = {"width": 1.1485 * 2.0, "front_length": 4.049, "rear_length": 1.127}
HALF_LEN = (VEH["front_length"] + VEH["rear_length"]) / 2.0
HALF_WID = VEH["width"] / 2.0
REAR_AXLE_TO_CENTER = HALF_LEN - VEH["rear_length"]
#: BBCoordsIndex order (navsim pdm_enums.py:147-154)
PT_NAMES = ("FRONT_LEFT", "REAR_LEFT", "REAR_RIGHT", "FRONT_RIGHT", "CENTER")


def dec(b64, shape, dtype=np.float32):
    if b64 is None:
        return None
    return np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape).copy()


def read_jsonl_last(path: str, key: str = "token") -> dict:
    """``{key: last complete record}``; a torn line (killed writer) is skipped and COUNTED."""
    out, torn = {}, 0
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                torn += 1
                continue
            out[r[key]] = r
    out["__torn__"] = torn
    return out


def load_fan(path: str = FAN) -> dict:
    """token -> {poses [117,8,3] f32, s9 [117], rk [117], sel_idx, stage}."""
    recs = read_jsonl_last(path)
    torn = recs.pop("__torn__")
    out = {}
    for t, r in recs.items():
        n = int(r["n_cands"])
        if n != N_FAN or r["universe"] != "FAN117":
            raise ValueError(f"fan record universe {r['universe']} n {n}")
        out[t] = {"poses": dec(r["cands_poses_b64"], (n, 8, 3)), "s9": dec(r["sel_score_v3_b64"], (N_FAN,)),
                  "rk": dec(r["reach_keep_b64"], (N_FAN,)), "sel_idx": int(r["sel_idx"]), "stage": int(r["stage"]),
                  "emitted": np.asarray(r["poses_emitted"], np.float32)}
    return {"fan": out, "torn": torn}


def deployed_rank(s9: np.ndarray, rk: np.ndarray, extra_mask=None) -> int:
    """argmax over reach_keep (AND extra_mask) of sel_score_v3, first index on ties; -1 if the set is empty."""
    keep = rk > 0
    if extra_mask is not None:
        keep = keep & extra_mask
    if not keep.any():
        return -1
    r = np.where(keep, s9.astype(np.float64), -np.inf)
    return int(np.argmax(r))


def load_seam(path: str = SEAM_A1) -> dict:
    z = np.load(path, allow_pickle=False)
    toks = [str(x) for x in z["token"]]
    poses = np.asarray(z["poses"], np.float32)
    src = [str(x) for x in z["source"]]
    if len(toks) != N_TOK or poses.shape != (N_TOK, 8, 3):
        raise ValueError(f"seam shape {len(toks)} {poses.shape}")
    return {t: {"poses": poses[i], "source": src[i]} for i, t in enumerate(toks)}


# ---- masks -------------------------------------------------------------------------------------
class MaskStore:
    """Reader of ``raw/p4_mask/mask_R7_A1.{bin,index.jsonl}`` written by d6_p4_export.py (last record per token wins)."""

    def __init__(self, d: str, arm: str = "R7_A1"):
        self.bin = os.path.join(d, f"mask_{arm}.bin")
        idx = read_jsonl_last(os.path.join(d, f"mask_{arm}.index.jsonl"))
        self.torn = idx.pop("__torn__")
        self.idx = idx
        self._fh = open(self.bin, "rb")

    def tokens(self):
        return sorted(self.idx)

    def get(self, tok: str, what=("m_hat",)) -> dict:
        r = self.idx[tok]
        self._fh.seek(int(r["offset"]))
        comp = self._fh.read(int(r["nbytes"]))
        if hashlib.sha256(comp).hexdigest()[:16] != r["sha256_16"]:
            raise ValueError(f"mask record sha mismatch for a token (offset {r['offset']})")
        raw = zlib.decompress(comp)
        if len(raw) != int(r["raw_nbytes"]):
            raise ValueError("mask record length mismatch")
        nb = H * W // 8
        o = 0
        out = {}
        m_hat = np.unpackbits(np.frombuffer(raw, np.uint8, nb, o))[: H * W].reshape(H, W).astype(bool)
        o += nb
        m_raw = np.unpackbits(np.frombuffer(raw, np.uint8, nb, o))[: H * W].reshape(H, W).astype(bool)
        o += nb
        u8 = np.frombuffer(raw, np.uint8, H * W, o).reshape(H, W)
        o += H * W
        cls = np.frombuffer(raw, np.uint8, H * W, o).reshape(H, W)
        o += H * W
        nlv = LV_HW[0] * LV_HW[1] // 8
        lv = np.unpackbits(np.frombuffer(raw, np.uint8, nlv, o))[: LV_HW[0] * LV_HW[1]].reshape(LV_HW).astype(bool)
        loc = {"m_hat": m_hat, "m_raw": m_raw, "u8": u8, "cls": cls, "lv": lv}
        if int(m_hat.sum()) != int(r["n_drivable_hat"]) or int(m_raw.sum()) != int(r["n_drivable_raw"]):
            raise ValueError("mask record counts do not match the index")
        for k in what:
            out[k] = loc[k]
        out["sel_idx"] = int(r["sel_idx"])
        return out


# ---- the gate geometry -------------------------------------------------------------------------
def check_points(poses: np.ndarray) -> np.ndarray:
    """poses [..., 8, 3] rear-axle (x, y, heading) -> [..., 8, 5, 2] FRONT_LEFT, REAR_LEFT, REAR_RIGHT, FRONT_RIGHT, CENTER,
    the formula of navsim ``state_array_to_coords_array`` (pdm_array_representation.py:188-227) with the pacifica parameters."""
    p = np.asarray(poses, np.float64)
    h = p[..., 2]
    c, s = np.cos(h), np.sin(h)
    cx = p[..., 0] + REAR_AXLE_TO_CENTER * c
    cy = p[..., 1] + REAR_AXLE_TO_CENTER * s

    def tr(lon, lat):
        return np.stack([cx + lon * c - lat * s, cy + lon * s + lat * c], axis=-1)

    return np.stack([tr(HALF_LEN, HALF_WID), tr(-HALF_LEN, HALF_WID), tr(-HALF_LEN, -HALF_WID), tr(HALF_LEN, -HALF_WID),
                     np.stack([cx, cy], axis=-1)], axis=-2)


def cell_index(pts: np.ndarray, mirror_y: bool = False):
    """[..., 2] ego-frame points -> (i, j, inside). ``mirror_y`` is the K-FRAME control's wrong transform."""
    x = pts[..., 0]
    y = -pts[..., 1] if mirror_y else pts[..., 1]
    i = np.floor(x / CELL).astype(np.int64)
    j = np.floor((y + Y_HALF) / CELL).astype(np.int64)
    inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    return i, j, inside


def gate_from_mask(mask: np.ndarray, pts: np.ndarray, mirror_y: bool = False):
    """mask [H, W] bool; pts [N, 8, 5, 2] -> (pass [N] bool, checked [N, 8, 5] bool, drivable [N, 8, 5] bool).
    A candidate PASSES iff every CHECKED point is on a drivable cell; points outside the window are unchecked."""
    i, j, inside = cell_index(pts, mirror_y)
    drv = np.ones(inside.shape, bool)
    drv[inside] = mask[i[inside], j[inside]]
    ok = np.where(inside, drv, True)
    return ok.reshape(ok.shape[0], -1).all(axis=1), inside, drv


def ego_to_global(pts: np.ndarray, x0: float, y0: float, h0: float) -> np.ndarray:
    c, s = math.cos(h0), math.sin(h0)
    return np.stack([x0 + c * pts[..., 0] - s * pts[..., 1], y0 + s * pts[..., 0] + c * pts[..., 1]], axis=-1)


def derangement(n: int, seed: int = 20261004) -> np.ndarray:
    """raw/p4_impl_choices.md s5: rng.permutation re-drawn from the SAME rng until no fixed point."""
    rng = np.random.default_rng(seed)
    k = 0
    while True:
        k += 1
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p, k


def sha12(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()[:12]


def tokens_file(name: str) -> list:
    return [l.strip() for l in open(os.path.join(RAW, name), encoding="utf-8") if l.strip()]
