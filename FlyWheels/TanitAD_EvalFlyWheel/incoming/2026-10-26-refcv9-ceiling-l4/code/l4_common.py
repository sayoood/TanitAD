"""L4 common loaders. CPU only. Clip ids are used in memory only; outputs carry sha12 at most."""
from __future__ import annotations
import hashlib, json, numpy as np

ARCH = "C:/Users/Admin/refcv8_eval_archive/battery"
EVAL_TREE = "step48852ema_pf"            # the final refcv8 battery tree (per-frame nav feed); speed label identical across trees (checked)
SIDECAR_EVAL = "E:/refcv6_eval_kit/data/refcv6_speed_max_v8_eval.jsonl"
SIDECAR_TRAIN = "E:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl"
V9_EVAL = "E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz"
V9_TRAIN = "E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz"
COT_LIMITS = "F:/c_overflow_2026-10-26/tmp/l4_scratch/_limits_classified.json"   # git show of the 2026-09-06-speed-limit-source raw/limits_classified.json
#: LITERAL ladders (km/h). Never rounded; m/s derived by /3.6 in float64 at use.
L4_KMH = (30.0, 50.0, 100.0, 120.0)
L8_KMH = (20.0, 30.0, 50.0, 70.0, 80.0, 100.0, 120.0, 130.0)
HZ = 10
ANCHOR_RAW = 80          # egomotion_source.RAW_T0_S = 8.0 s * 10 Hz (s2 anchor); label band = [anchor+20, anchor+60]


def sha12(cid: str) -> str:
    return hashlib.sha256(cid.encode()).hexdigest()[:12]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snap_up_kmh(v_ms: np.ndarray, ladder_kmh=L4_KMH, clamp_value: bool = False) -> np.ndarray:
    """CONTAINING-WINDOW snap-up in km/h (the refcv6 rule): smallest step >= v; above the top step -> top step
    (clamp_value=False, refcv6/8 label rule) or the value itself (clamp_value=True, refcv8 'over' rule for N2)."""
    lad = np.asarray(ladder_kmh, np.float64)
    kmh = np.asarray(v_ms, np.float64) * 3.6
    idx = np.searchsorted(lad, kmh - 1e-9 * 3.6, side="left")      # first step >= v (1e-9 m/s tolerance as speed_max_bin)
    over = idx >= len(lad)
    out = lad[np.minimum(idx, len(lad) - 1)]
    if clamp_value:
        out = np.where(over, kmh, out)
    return out


def load_sidecar(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                out[r["clip_id"]] = r
    return out


class V9:
    """Per-clip contiguous v_now_ms by raw row k (NaN where the release has no row)."""
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.sid = z["clip__sid"]; self.row0 = z["clip__row0"]; self.n = z["clip__n_rows"]
        self.k = z["row__k"].astype(np.int64); self.v = z["row__v_now_ms"].astype(np.float64)
        self.vhi_pf = z["row__v_hi_ms"].astype(np.float64)        # v9 per-frame SPEED_BAND v_hi over [NOW+2, NOW+8]
        self.n2 = z["row__speed_n2_kmh"].astype(np.float64)
        self.idx = {int(s): i for i, s in enumerate(self.sid)}

    def clip(self, sid: int):
        i = self.idx.get(int(sid))
        if i is None:
            return None
        a, b = int(self.row0[i]), int(self.row0[i] + self.n[i])
        k = self.k[a:b]
        kmax = int(k.max()) + 1
        V = np.full(kmax + 1, np.nan); V[k] = self.v[a:b]
        H = np.full(kmax + 1, np.nan); H[k] = self.vhi_pf[a:b]
        N2 = np.full(kmax + 1, np.nan); N2[k] = self.n2[a:b]
        return {"k": k, "V": V, "VHI": H, "N2": N2}
