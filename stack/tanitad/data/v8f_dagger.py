"""PREREG_V8F_DAGGER (claim row H-V8F-DAGGER-1), the DATA side: the own-state bank, the matched route point r', and the swap of an own state into a training sample.

Source of every rule below: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-dagger-draft/PREREG_V8F_DAGGER.md`` (sections 1, 2, 6) and its
``probe/THRESHOLDS_FIXED.txt``. Nothing here is a free parameter: the constants are the PREREG's, and the probability ``p`` of a swap is NOT in this file (it is the trainer's ``--dagger-p``; the
arm's p is the one PREREG section 4 maps from O-RECOVER's read (one value after NO EFFECT, a smaller one after HARMS) -- fixed by ORC's outcome row, never by a number in this module).

WHAT A DAGGER SAMPLE IS (PREREG section 2). The LOGGED training sample at (clip, r') with its EGO tensors replaced by the model's OWN T1-kin state of a rollout of the frozen M2b. It is ORC's injection
(``EgoPerturb``, ``v8f_mini_data.py``) with the replacement drawn from a bank instead of from a law, so the three sets are ORC's, imported, not copied:
  REBUILT  ``EGO_PERTURB_REBUILT`` = p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav        (replaced by the own state)
  KEPT     ``EGO_PERTURB_KEPT``    = feats, frame_ok, limit, limit_ok, a_t, a_s, pose_now       (the same objects: the logged images at r', the clip's logged bucket, the sampler's ages, the anchor)
  RECORD   ``EGO_PERTURB_RECORD``  = act                                                        (the world model's transition record; own actions paired with logged frames would corrupt it)
  and the targets (gt, gt_ok, s_h, lead_gap_h, lab, lab_t) are the human's at r' -- the SAME objects. The ONE label tensor touched is ``lab_s.rc``, and only by a frame change (``rc_frame_change``).

THE BANK (section 1). One record per own state: ticks 1 and 2 of an m = 10 T1-kin rollout (j = t0 + 10, t0 + 20), TRAIN windows only. A record stores the own history rows j - hmax + 1 ... j (p10f, world
pose, travelled arc), the own kin3 at j, the matched row r', its lateral distance, heading error and the cut reason. Rows before the clip start are NaN. Cut records are STORED (with their reason): a cut is
counted, never silently dropped (section 2: "Every cut is counted by reason and reported"); only reason == OK is eligible.

THE MATCHER (section 2). ``match_route``: project the own (x, y) onto the logged 0.2 s polyline of rows [t0, T-1], segment by segment; e_lat = the smallest distance to a segment; r' = the row whose
cumulative logged arc is nearest the foot's arc; rows within ARC_TIE_M of that arc are tied (a stationary stretch), and among them the row nearest in time to the own clock row j wins, then the earlier row;
cuts, in this order: e_lat > 3.0 m (off_lat), |wrap(psi - psi_logged[r'])| > 30 deg (off_head), r' > T - 31 (end_edge).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tanitad.data import v8f_mini_data as D

# ------------------------------------------------------------------------------------------------ the PREREG's constants (probe/THRESHOLDS_FIXED.txt, sha256 438366d5...)
OFF_ROUTE_LAT_M = 3.0          # = t1hd_read.py OFF_ROUTE_M, here on the perpendicular distance to the logged polyline
OFF_ROUTE_HEAD_DEG = 30.0      # ESTIMATED (guards loops / crossings); fixed before data
ARC_TIE_M = 0.05               # rows whose cumulative logged arc lies within this of the nearest arc are tied
END_EDGE_ROWS = D.HORIZON + 1  # r' > T - 31: no complete 6 s logged future at r'
COVERAGE_MIN = 0.5             # section 6: the eligible fraction of TRAIN windows; below it the arm does not launch (BLOCKED)
TICKS = (1, 2)                 # ticks of an m = 10 rollout kept as own states (j = t0 + 10, t0 + 20); tick 0 is the logged state and is never an own state
REPLAN_STEPS = 10              # m
#: the decisive T1-kin configuration (section 1): v8f_t1_eval.py rollout flags
ROLLOUT = {"replan_steps": REPLAN_STEPS, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map": "none (PLAIN)"}
REASONS = ("ok", "off_lat", "off_head", "end_edge", "nonfinite")
OK, OFF_LAT, OFF_HEAD, END_EDGE, NONFINITE = range(5)
SCHEMA = "v8f-dagger-bank-v1"
#: ORC's partition, imported: the PREREG's table is "EGO_PERTURB_REBUILT / KEPT / RECORD" by name
REBUILT, KEPT, RECORD = D.EGO_PERTURB_REBUILT, D.EGO_PERTURB_KEPT, D.EGO_PERTURB_RECORD
LABELS_RE_EXPRESSED = ("lab_s.rc",)
#: AMBIGUITY (reported to the Master Mind). PREREG section 2 puts nav channel 3 at "nav[r', 3] + own arc behind NOW / 100 on frames sharing NOW's token and validity (ORC's rule)". Read to the letter, that
#: also writes a progress value onto frames whose route is INVALID (nav[:, 4] == 0), where the join stores exactly 0.0 and the model has never seen anything else -- ORC's rule leaves those at 0 (it scales a zero
#: difference), and the T1-kin dead reckoning counts down only ``if row[4] > 0``. False (the default) = progress is rebuilt only where the route is valid, i.e. ORC's and T1's behaviour. True = the letter.
NAV_REBUILD_INVALID = False
DAGGER_DEFAULTS = {"p": 0.0, "seed": 0}          # the lever's OFF state: p is NEVER given a non-zero default here


def constants() -> dict:
    """The matching constants as recorded in the bank header; ``constants_sha256`` is the hash of this dict's canonical JSON."""
    return {"OFF_ROUTE_LAT_M": OFF_ROUTE_LAT_M, "OFF_ROUTE_HEAD_DEG": OFF_ROUTE_HEAD_DEG, "ARC_TIE_M": ARC_TIE_M, "END_EDGE_ROWS": END_EDGE_ROWS,
            "TICKS": list(TICKS), "REPLAN_STEPS": REPLAN_STEPS, "COVERAGE_MIN": COVERAGE_MIN, "REASONS": list(REASONS), "ROLLOUT": ROLLOUT,
            "TIE_RULE": "among tied rows the one nearest in time to the own state's clock row j; then the earlier row",
            "CANDIDATE_ROWS": "[t0, T-1] of the same clip, 0.2 s pose grid, world xy"}


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_jsonable).encode("utf-8")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def constants_sha256() -> str:
    return sha256_bytes(canon(constants()))


def windows_sha256(clip_names, wins) -> str:
    """The TRAIN window list's digest: sha256 of ``clip:t`` per window, in list order (clip-name based, so it does not depend on the join's row order). The trainer has no window-list digest of its own
    (``build_data`` reports ``train_sha256`` over CLIPS); this is the one the bank header and the arm's gate compare (K-B3)."""
    names = np.asarray(clip_names)
    return sha256_bytes("\n".join(f"{names[int(w[0])]}:{int(w[1])}" for w in wins).encode("utf-8"))


class BankRefused(Exception):
    """A bank (or a bank request) that violates a PREREG control: the read is REFUSED, never repaired in place."""


# ------------------------------------------------------------------------------------------------ the matcher (section 2)
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Match(NamedTuple):
    r: int           # r': the matched logged row (-1 when none could be computed)
    e_lat: float     # the smallest perpendicular distance to the logged polyline [m]
    dpsi_deg: float  # |wrap(psi_own - psi_logged[r'])| [deg]
    reason: int      # index into REASONS


def match_route(pose_grid, t0: int, own, j: int) -> Match:
    """r' for an own pose ``own`` = (x, y, yaw) reached at clock row ``j`` of a rollout started at ``t0`` (see the module docstring for the rule). ``pose_grid`` [T, >=3] is the clip's LOGGED 0.2 s grid."""
    pg = np.asarray(pose_grid, np.float64)
    T = len(pg)
    own = np.asarray(own, np.float64)[:3]
    P = pg[int(t0):, :3]
    if len(P) == 0 or not np.isfinite(own).all() or not np.isfinite(P).all():
        return Match(-1, float("nan"), float("nan"), NONFINITE)
    xy = P[:, :2]
    seg = np.diff(xy, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(L)])
    q = own[:2]
    if len(seg):
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(L > 0, ((q - xy[:-1]) * seg).sum(1) / np.where(L > 0, L * L, 1.0), 0.0)
        u = np.clip(u, 0.0, 1.0)
        foot = xy[:-1] + u[:, None] * seg
        dist = np.hypot(*(foot - q).T)
        k = int(np.argmin(dist))                                   # first minimum (the earlier segment) on exact ties
        e_lat, s_star = float(dist[k]), float(arc[k] + u[k] * L[k])
    else:
        e_lat, s_star = float(np.hypot(*(xy[0] - q))), 0.0
    i0 = int(np.argmin(np.abs(arc - s_star)))                      # the row whose cumulative arc is nearest s* (the earlier one on an exact midpoint)
    tied = np.nonzero(np.abs(arc - arc[i0]) <= ARC_TIE_M)[0]       # rows within ARC_TIE_M of THAT nearest arc (a stationary stretch)
    rows = tied + int(t0)
    r = int(rows[np.lexsort((rows, np.abs(rows - int(j))))[0]])    # nearest in time to j, then the earlier row
    dpsi = math.degrees(abs(wrap(float(own[2]) - float(pg[r, 2]))))
    if e_lat > OFF_ROUTE_LAT_M:
        return Match(r, e_lat, dpsi, OFF_LAT)
    if dpsi > OFF_ROUTE_HEAD_DEG:
        return Match(r, e_lat, dpsi, OFF_HEAD)
    if r > T - END_EDGE_ROWS:
        return Match(r, e_lat, dpsi, END_EDGE)
    return Match(r, e_lat, dpsi, OK)


# ------------------------------------------------------------------------------------------------ re-anchoring (section 2, "poses re-anchored by the rigid motion ...")
def reanchor_rows(P, own_now, log_now):
    """Own world poses ``P`` [N, F, 3] -> the same poses after the rigid motion that maps ``own_now`` [N, 3] onto ``log_now`` [N, 3] (float64). Relative geometry is exact (a rigid motion). A row whose motion is the
    identity at the float32 resolution of the logged NOW pose (own NOW == logged NOW, as at tick 0) is returned UNCHANGED, so that the identity is bit-exact rather than 1 ulp off."""
    P = np.asarray(P, np.float64)
    own_now, log_now = np.asarray(own_now, np.float64), np.asarray(log_now, np.float64)
    c, s = np.cos(-own_now[:, 2])[:, None], np.sin(-own_now[:, 2])[:, None]
    dx, dy = P[..., 0] - own_now[:, None, 0], P[..., 1] - own_now[:, None, 1]
    rx, ry = dx * c - dy * s, dx * s + dy * c
    C, S_ = np.cos(log_now[:, 2])[:, None], np.sin(log_now[:, 2])[:, None]
    out = np.stack([log_now[:, None, 0] + rx * C - ry * S_, log_now[:, None, 1] + rx * S_ + ry * C, P[..., 2] - own_now[:, None, 2] + log_now[:, None, 2]], -1)
    ident = (own_now.astype(np.float32) == log_now.astype(np.float32)).all(1)
    return np.where(ident[:, None, None], P, out)


def reanchor(P, own_now, log_now):
    """``reanchor_rows`` for one state: ``P`` [..., 3]."""
    P = np.asarray(P, np.float64)
    return reanchor_rows(P.reshape(1, -1, 3), np.asarray(own_now, np.float64)[None, :3], np.asarray(log_now, np.float64)[None, :3]).reshape(P.shape)


def rc_frame_change(rc, rc_ok, src, dst):
    """``lab_s.rc`` re-expressed: the SAME world points (the human's route checkpoints at r' - a_s) moved from the logged frame at r' - a_s (pose ``src`` [B, 3]) into the sample's S frame (pose ``dst`` [B, 3]) by
    ``v8f_mini.transform_ckpt``. EXACTLY the identity for a row whose two frames coincide bit for bit (a_s = 0: the S frame IS the logged frame at r'); an entry that is not ``rc_ok`` is a zero placeholder (``rc *
    rc_ok`` in the join) with no world point behind it and is left as it is."""
    import torch
    from tanitad.models.v8f_mini import transform_ckpt
    moved = transform_ckpt(rc, src, dst)
    same = (src == dst).all(-1)[:, None, None]
    return torch.where(same | ~rc_ok[..., None], rc, moved)


# ------------------------------------------------------------------------------------------------ the bank on disk
def _save_npz_det(path, arrays: dict):
    """An uncompressed ``.npz`` with a FIXED zip timestamp and sorted member order: the same arrays give the same bytes (so the file's sha256 names its content)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for k in sorted(arrays):
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            with zf.open(zi, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.ascontiguousarray(arrays[k]), allow_pickle=False)


RECORD_FIELDS = {"ci": np.int32, "t0": np.int32, "tick": np.int8, "j": np.int32, "p10f": np.float32, "pose": np.float64, "arc": np.float64, "kin3": np.float64,
                 "r": np.int32, "e_lat": np.float64, "dpsi": np.float64, "reason": np.int8}


def save_shard(path, arrays: dict, header: dict):
    missing = sorted(set(RECORD_FIELDS) - set(arrays))
    if missing:
        raise ValueError(f"shard arrays lack {missing}")
    out = {k: np.asarray(arrays[k], RECORD_FIELDS[k]) for k in RECORD_FIELDS}
    n = len(out["ci"])
    if any(len(v) != n for v in out.values()):
        raise ValueError("shard arrays disagree on the record count")
    out["header_json"] = np.frombuffer(canon({**header, "n_records": n}), np.uint8)
    _save_npz_det(path, out)


def load_shard(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in RECORD_FIELDS}
        header = json.loads(bytes(z["header_json"]).decode("utf-8"))
    return arrays, header


def content_sha256(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(RECORD_FIELDS):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def check_train_only(*, bank_clips, train_clips, eval_clips, bank_window_list_sha256=None, train_window_list_sha256=None, complete=True):
    """K-B3. The bank's clips must be a subset of the arm's TRAIN clips and disjoint from eval-141 (``v8f_mini_data.assert_disjoint``: by clip id AND by stable sid), and -- for a COMPLETE bank -- its window-list
    sha256 must equal the trainer's. Raises :class:`BankRefused` naming the first violation."""
    b, tr, ev = set(map(str, bank_clips)), set(map(str, train_clips)), set(map(str, eval_clips))
    if b & ev:
        raise BankRefused(f"K-B3: the bank holds {len(b & ev)} eval clip(s), e.g. {sorted(b & ev)[:3]}: an eval clip may never enter the own-state bank")
    try:
        D.assert_disjoint(b, ev)
    except AssertionError as e:                                                        # a stable-sid collision across the two sets
        raise BankRefused(f"K-B3: {e}") from e
    if not b <= tr:
        raise BankRefused(f"K-B3: {len(b - tr)} bank clip(s) are not in the arm's TRAIN clip list, e.g. {sorted(b - tr)[:3]}")
    if complete and bank_window_list_sha256 != train_window_list_sha256:
        raise BankRefused(f"K-B3: the bank's window-list sha256 {str(bank_window_list_sha256)[:12]} != the trainer's {str(train_window_list_sha256)[:12]}")
    return {"n_bank_clips": len(b), "n_train_clips": len(tr), "n_eval_clips": len(ev), "overlap_with_eval": 0}


class DaggerBank:
    """The loaded own-state bank: record arrays + header, and the lookup ``(ci, r') -> eligible records`` the swap draws from. ``ci`` is the TRAINER's join index: pass ``clip_names`` (``J["clip"]``) and each
    record's clip is re-mapped BY NAME (a clip missing from the join is refused); without it the stored generator-side ``ci`` is used as it is."""

    def __init__(self, arrays: dict, header: dict, *, clip_names=None, manifest: dict | None = None, sha256: str | None = None, path=None):
        self.a, self.header, self.manifest, self.sha256, self.path = arrays, header, manifest, sha256, path
        self.ci = np.asarray(arrays["ci"], np.int64).copy()
        if clip_names is not None:
            idx = {str(c): i for i, c in enumerate(np.asarray(clip_names).tolist())}
            names = {int(k): v for k, v in header["clips"].items()}
            miss = sorted({v for v in names.values() if v not in idx})
            if miss:
                raise BankRefused(f"the bank holds {len(miss)} clip(s) that the join does not, e.g. {miss[:3]}")
            lut = np.full(max(names) + 1, -1, np.int64)
            for k, v in names.items():
                lut[k] = idx[v]
            self.ci = lut[self.ci]
        self.reason = np.asarray(arrays["reason"])
        ok = np.nonzero(self.reason == OK)[0]
        key = self.ci[ok] * (1 << 24) + np.asarray(arrays["r"], np.int64)[ok]
        order = np.argsort(key, kind="stable")                                         # stable: ties keep the bank's record order
        self._ok_idx, self._ok_key = ok[order], key[order]
        self._uniq, self._start, self._count = np.unique(self._ok_key, return_index=True, return_counts=True)

    # -- header facts ---------------------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(len(self.reason))

    @property
    def hmax(self) -> int:
        return int(self.a["p10f"].shape[1])

    def reason_counts(self) -> dict:
        return {REASONS[i]: int((self.reason == i).sum()) for i in range(len(REASONS))}

    # -- the lookup -----------------------------------------------------------------------------------------------
    def candidates(self, ci, t):
        """([B] count of eligible own states with r' == t in clip ci, [B] start offset into the sorted eligible list)."""
        key = np.asarray(ci, np.int64) * (1 << 24) + np.asarray(t, np.int64)
        if len(self._uniq) == 0:                                                       # every record was cut: nothing is eligible
            z = np.zeros(len(key), np.int64)
            return z, z.copy()
        pos_c = np.minimum(np.searchsorted(self._uniq, key), len(self._uniq) - 1)
        hit = self._uniq[pos_c] == key
        return np.where(hit, self._count[pos_c], 0), np.where(hit, self._start[pos_c], 0)

    def pick(self, ci, t, u):
        """One eligible record index per row, uniform among that (ci, t)'s candidates (``u`` in [0, 1) per row). Rows without a candidate return -1."""
        cnt, st = self.candidates(ci, t)
        if len(self._ok_idx) == 0:
            return np.full(len(cnt), -1, np.int64)
        k = np.minimum(np.floor(np.asarray(u, np.float64) * np.maximum(cnt, 1)).astype(np.int64), np.maximum(cnt - 1, 0))
        return np.where(cnt > 0, self._ok_idx[np.minimum(st + k, len(self._ok_idx) - 1)], -1)

    def gather(self, idx):
        idx = np.asarray(idx, np.int64)
        return {k: np.asarray(self.a[k])[idx] for k in RECORD_FIELDS}

    def coverage(self, wins) -> dict:
        """COVERAGE (section 6): the fraction of TRAIN windows (ci, t) that have at least one own state with r' == t that passed every cut."""
        w = np.asarray([(int(x[0]), int(x[1])) for x in wins], np.int64).reshape(-1, 2)
        cnt, _ = self.candidates(w[:, 0], w[:, 1])
        n_el, n = int((cnt > 0).sum()), int(len(w))
        frac = n_el / max(n, 1)
        return {"n_windows": n, "n_eligible": n_el, "fraction": frac, "min": COVERAGE_MIN, "blocked": bool(frac < COVERAGE_MIN)}

    # -- controls -------------------------------------------------------------------------------------------------
    def check_on_policy(self, init_ckpt_sha256: str) -> dict:
        """K-B2: the bank's generating checkpoint IS the arm's ``--init-ckpt``."""
        g = self.header.get("ckpt_sha256")
        if not g or str(g).lower() != str(init_ckpt_sha256).lower():
            raise BankRefused(f"K-B2: the bank was generated by checkpoint {str(g)[:12]}, the arm fine-tunes {str(init_ckpt_sha256)[:12]}: the states are not on-policy for this init")
        return {"ckpt_sha256": g}

    def check_train_only(self, *, train_clips, eval_clips, train_window_list_sha256=None, allow_partial: bool = False) -> dict:
        if self.header.get("partial", False) and not allow_partial:
            raise BankRefused("K-B3: this bank is PARTIAL (generated with --limit-clips): it covers only part of the arm's TRAIN windows and may not train an arm")
        names = self.header["clips"].values()
        return check_train_only(bank_clips=names, train_clips=train_clips, eval_clips=eval_clips, bank_window_list_sha256=self.header.get("window_list_sha256"),
                                train_window_list_sha256=train_window_list_sha256, complete=not self.header.get("partial", False))

    # -- loading --------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path, *, clip_names=None, verify: bool = True, expect_sha256: str | None = None):
        """``path`` = a ``bank.json`` manifest (or the directory holding it) or a single shard ``.npz``. ``verify`` re-hashes every shard file and its content against the manifest. ``expect_sha256`` is the
        arm's ``--dagger-bank-sha256``: the manifest's sha256 must equal it (a bank other than the declared one is refused)."""
        p = Path(path)
        if p.is_dir():
            p = p / "bank.json"
        if p.suffix == ".npz":
            arrays, header = load_shard(p)
            return cls(arrays, header, clip_names=clip_names, sha256=sha256_file(p), path=str(p))
        raw = p.read_bytes()
        sha = sha256_bytes(raw)
        if expect_sha256 and sha.lower() != str(expect_sha256).lower():
            raise BankRefused(f"the bank's sha256 {sha[:12]} != the declared {str(expect_sha256)[:12]}")
        man = json.loads(raw.decode("utf-8"))
        parts, headers = [], []
        for s in man["shards"]:
            f = p.parent / s["file"]
            if verify and sha256_file(f) != s["file_sha256"]:
                raise BankRefused(f"shard {s['file']}: sha256 differs from the manifest (a file with the same name but other bytes)")
            arrays, header = load_shard(f)
            if verify and content_sha256(arrays) != s["content_sha256"]:
                raise BankRefused(f"shard {s['file']}: content sha256 differs from the manifest")
            parts.append(arrays)
            headers.append(header)
        arrays = {k: np.concatenate([a[k] for a in parts]) for k in RECORD_FIELDS}
        clips = {}
        for h in headers:
            clips.update({int(k): v for k, v in h["clips"].items()})
        header = {**man["common"], "clips": {str(k): v for k, v in clips.items()}, "partial": man["partial"]}
        return cls(arrays, header, clip_names=clip_names, manifest=man, sha256=sha, path=str(p))


# ------------------------------------------------------------------------------------------------ the bank controls K-B4 / K-B5 (section 6), RECORDED per bank
# PREREG section 6 lists K-B4 and K-B5 as BANK controls "read before any training". They used to exist only as unit tests; the generator (``v8f_dagger_bank.py``) now RUNS them at generation time with THIS code (the
# unit tests call these same functions) and writes the result to each shard's provenance and, merged, to ``PROVENANCE.json`` ``controls`` -- the records ``dag_read.py`` (``g_kb4`` / ``g_kb5``) refuses without.
#: the reader's names (``dag_read.py`` KB4_CASES): the six synthetic cases of section 6
KB4_CASES = ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")
KB5_TOL_M = 1e-4               # section 6: the held-frame relative transforms are preserved to <= 1e-4 m (and so is pose_now against the logged pose)


def _kb4_straight(T: int = 80):
    pg = np.zeros((T, 4))
    pg[:, 0] = np.arange(T) * 2.0
    return pg


def kb4_cases() -> dict:
    """K-B4: the matcher's six synthetic cases, run NOW against ``match_route`` (looked up at CALL time: a bent matcher turns its case False). {case: bool} in ``KB4_CASES`` order. A case that raises is False.

    straight        own pose 0.4 m off a straight polyline: r' = the row at the foot, e_lat = 0.4, kept
    stationary_tie  rows 20..39 stationary (a human waiting): the row NEAREST IN TIME to j wins (35), not the first minimum (20)
    lateral_cut     3.2 m off the route: cut as off_lat
    heading_cut     40 deg off the route heading: cut as off_head
    end_edge        r' = 60 of T = 80: no complete 6 s logged future, cut as end_edge
    loop            a closed loop whose end is its start: the own state at row 10 matches row 10, not the coincident start"""
    pg = _kb4_straight()
    pg2 = pg.copy()
    pg2[20:40, 0] = 40.0
    pg2[40:, 0] = 40.0 + np.arange(80 - 40) * 2.0                                                   # stationary rows 20..40 at x = 40
    th = np.linspace(0, 2 * math.pi, 80)
    pg3 = np.zeros((80, 4))
    pg3[:, 0], pg3[:, 1], pg3[:, 2] = 30 * np.sin(th), 30 * (1 - np.cos(th)), th                       # the path returns to its start

    def straight():
        m = match_route(pg, 5, (40.9, 0.4, 0.0), 30)
        return m.r == 20 and abs(m.e_lat - 0.4) < 1e-9 and m.reason == OK

    def stationary_tie():
        m = match_route(pg2, 5, (40.0, 0.0, 0.0), 35)
        return m.r == 35 and m.r != 20 and m.reason == OK

    def lateral_cut():
        return match_route(pg, 5, (40.0, 3.2, 0.0), 30).reason == OFF_LAT

    def heading_cut():
        return match_route(pg, 5, (40.0, 0.0, math.radians(40.0)), 30).reason == OFF_HEAD

    def end_edge():
        m = match_route(pg, 5, (2.0 * 60, 0.0, 0.0), 70)
        return m.reason == END_EDGE and m.r == 60

    def loop():
        m = match_route(pg3, 0, (pg3[10, 0], pg3[10, 1], pg3[10, 2]), 10)
        return m.r == 10 and m.reason == OK
    fns = {"straight": straight, "stationary_tie": stationary_tie, "lateral_cut": lateral_cut, "heading_cut": heading_cut, "end_edge": end_edge, "loop": loop}
    out = {}
    for name in KB4_CASES:
        try:
            out[name] = bool(fns[name]())
        except Exception:                                                                            # noqa: BLE001  a case that raises did not pass
            out[name] = False
    return out


def kb4_record() -> dict:
    """``controls["K-B4"]`` of ONE generator run: ``{"PASS": bool, "cases": {case: bool}}``. PASS is DERIVED from the six cases (all True), never written on its own."""
    cases = kb4_cases()
    return {"PASS": all(cases.get(c) is True for c in KB4_CASES), "cases": cases}


def kb4_combine(shard_records) -> dict:
    """The bank's ``controls["K-B4"]``: the six cases run AGAIN now (at the merge) AND-ed with every shard's recorded case (a shard that recorded no case, or a False one, fails it). PASS is derived from the cases."""
    now, recs = kb4_cases(), list(shard_records)
    cases = {c: bool(now.get(c) is True and recs and all(isinstance(r, dict) and isinstance(r.get("cases"), dict) and r["cases"].get(c) is True for r in recs)) for c in KB4_CASES}
    return {"PASS": all(cases.values()), "cases": cases, "run_at": "generation (every shard) and at the merge", "n_shards_checked": len(recs)}


def _rel_all(Q):
    """Pose b in the frame of pose a for EVERY pair (a, b) of the history frames: ([N, F, F, 2] metres, [N, F, F] radians). ``Q`` is [N, F, 3]."""
    d = Q[:, None, :, :2] - Q[:, :, None, :2]
    c, s = np.cos(-Q[:, :, 2])[:, :, None], np.sin(-Q[:, :, 2])[:, :, None]
    return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], -1), Q[:, None, :, 2] - Q[:, :, None, 2]


def kb5_measure(own_pose, log_pose, block: int = 2048) -> dict:
    """K-B5 on bank records: ``own_pose`` [N, F, 3] (the records' own world-pose histories, NaN before the clip), ``log_pose`` [N, 3] (the LOGGED pose at each record's matched row r'). The records are re-anchored
    EXACTLY as the swap does (``reanchor_rows`` onto the float32 ``pose_now`` the sample keeps, then cast to the float32 tensor the model reads) and compared:

      max_pose_now_err_m           the re-anchored NOW against the LOGGED (float64) pose at r'   [m, x-y distance]  (= the float32 quantisation of ``pose_now`` + the re-anchoring)
      max_pose_now_yaw_err_rad     the same, the heading (wrapped)                                       [rad]
      max_reanchor_vs_pose_now_m   the re-anchored NOW against the sample's float32 ``pose_now``           [m]       (the re-anchoring alone: EXACTLY 0.0 for a correct rigid motion)
      max_rel_transform_err_m      the held-frame relative transforms, EVERY pair (a, b) of the history, BEFORE (the record's float64 poses) vs AFTER re-anchoring (the float32 tensor)    [m, x-y]
      max_rel_transform_yaw_err_rad   the same, the relative heading                                                                                                            [rad]
      max_rel_transform_err_float64_m the relative transforms before vs after in FLOAT64 (the re-anchoring's own arithmetic, ~1e-12): NOT gated; it tells a formula defect from the float32 floor (a pose 1 km from the
                                   origin cannot be stored in float32 to better than ~1e-4 m, whatever the re-anchoring does)
    A pair with a NaN frame (a row before the clip) is not a held frame and is skipped; a NaN that appears AFTER re-anchoring where the record was finite is NOT skipped (it propagates: PASS is False)."""
    P, L = np.asarray(own_pose, np.float64), np.asarray(log_pose, np.float64)[:, :3]
    out = {"n": int(len(P)), "n_pairs": 0, "max_pose_now_err_m": 0.0, "max_pose_now_yaw_err_rad": 0.0, "max_reanchor_vs_pose_now_m": 0.0, "max_rel_transform_err_m": 0.0, "max_rel_transform_yaw_err_rad": 0.0, "max_rel_transform_err_float64_m": 0.0}
    for s in range(0, len(P), block):
        p, lg = P[s:s + block], L[s:s + block]
        l32 = lg.astype(np.float32).astype(np.float64)                                               # the sample's pose_now: the logged pose at r' as float32 (KEPT, never swapped)
        R = reanchor_rows(p, p[:, -1], l32)                                                          # the swap's own call
        R32 = R.astype(np.float32).astype(np.float64)                                                # the tensor the model reads
        rb, yb = _rel_all(p)
        ra, ya = _rel_all(R32)
        rr, _ = _rel_all(R)
        ok = np.isfinite(rb).all(-1)                                                                 # held frames of the RECORD (before): both frames real
        out["n_pairs"] += int(ok.sum())
        e_rel = np.where(ok, np.hypot(*(ra - rb).transpose(3, 0, 1, 2)), 0.0)
        e_yaw = np.where(ok, np.abs(ya - yb), 0.0)
        e_rel64 = np.where(ok, np.hypot(*(rr - rb).transpose(3, 0, 1, 2)), 0.0)
        e_now = np.hypot(*(R32[:, -1, :2] - lg[:, :2]).T)
        e_re = np.hypot(*(R32[:, -1, :2] - l32[:, :2]).T)
        e_nyaw = np.abs(wrap(R32[:, -1, 2] - lg[:, 2]))
        for k, v in (("max_pose_now_err_m", e_now), ("max_pose_now_yaw_err_rad", e_nyaw), ("max_reanchor_vs_pose_now_m", e_re), ("max_rel_transform_err_m", e_rel), ("max_rel_transform_yaw_err_rad", e_yaw), ("max_rel_transform_err_float64_m", e_rel64)):
            out[k] = float(np.maximum(out[k], v.max())) if v.size else out[k]                          # np.maximum propagates a NaN
    return out


def kb5_pass(rec) -> bool:
    """K-B5 passes iff it measured at least one record and the gated numbers are finite and <= ``KB5_TOL_M``: the two the reader reads (``max_pose_now_err_m`` and ``max_rel_transform_err_m``, metres) and the
    heading of NOW (``max_pose_now_yaw_err_rad``: ``pose_now`` is a pose, not a point). Computed from the numbers, never from a stored flag."""
    try:
        n = int(rec.get("n_records") or 0)
        vals = [float(rec[k]) for k in ("max_pose_now_err_m", "max_rel_transform_err_m", "max_pose_now_yaw_err_rad")]
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return n > 0 and all(math.isfinite(v) and v <= KB5_TOL_M for v in vals)


_KB5_MAX = ("max_pose_now_err_m", "max_pose_now_yaw_err_rad", "max_reanchor_vs_pose_now_m", "max_rel_transform_err_m", "max_rel_transform_yaw_err_rad", "max_rel_transform_err_float64_m")
KB5_SAMPLE = "all records with a finite own pose and a logged row at r' (reason != nonfinite); no sampling, no seed"
KB5_MEASURED_ON = "the float32 tensors the swap emits (reanchor_rows, cast to float32), against the logged float64 poses"


class Kb5Accumulator:
    """K-B5 over a generator run, fed the snapshot dicts of every chunk (``update``): each needs ``pose`` [F, 3], ``log_now`` [3] (the logged pose at r') and ``reason``. ``record()`` is the shard's ``controls["K-B5"]``."""

    def __init__(self):
        self.n = self.n_skipped = self.n_pairs = 0
        self.max = {k: 0.0 for k in _KB5_MAX}
        self._h = hashlib.sha256()                                                                   # over the logged pose (float64 bytes) of every measured record, in record order: WHICH logged rows were compared

    def update(self, recs) -> None:
        keep = [r for r in recs if int(r["reason"]) != NONFINITE and np.isfinite(np.asarray(r["log_now"], np.float64)).all()]
        self.n_skipped += len(recs) - len(keep)
        if not keep:
            return
        m = kb5_measure(np.stack([np.asarray(r["pose"], np.float64) for r in keep]), np.stack([np.asarray(r["log_now"], np.float64) for r in keep]))
        for r in keep:
            self._h.update(np.ascontiguousarray(np.asarray(r["log_now"], np.float64)).tobytes())
        self.n += m["n"]
        self.n_pairs += m["n_pairs"]
        for k in _KB5_MAX:
            self.max[k] = float(np.maximum(self.max[k], m[k]))

    def record(self) -> dict:
        rec = {**self.max, "tol_m": KB5_TOL_M, "n_records": self.n, "n_skipped_nonfinite": self.n_skipped, "n_pairs": self.n_pairs, "sample": KB5_SAMPLE, "measured_on": KB5_MEASURED_ON,
               "log_pose_sha256": self._h.hexdigest()}
        return {"PASS": kb5_pass(rec), **rec}


def kb5_combine(shard_records) -> dict:
    """The bank's ``controls["K-B5"]`` from the shards' records: counts summed, every maximum the MAX over shards (a NaN propagates), PASS recomputed from the combined numbers."""
    recs = list(shard_records)
    out = {k: (float(np.max([float(r[k]) for r in recs])) if recs else float("nan")) for k in _KB5_MAX}
    out.update({"tol_m": KB5_TOL_M, "n_records": sum(int(r.get("n_records") or 0) for r in recs), "n_skipped_nonfinite": sum(int(r.get("n_skipped_nonfinite") or 0) for r in recs),
                "n_pairs": sum(int(r.get("n_pairs") or 0) for r in recs), "sample": KB5_SAMPLE, "n_shards": len(recs), "measured_on": KB5_MEASURED_ON,
                "log_pose_sha256_by_shard": [r.get("log_pose_sha256") for r in recs]})
    return {"PASS": kb5_pass(out), **out}


# ------------------------------------------------------------------------------------------------ the swap (section 2, the table)
def swap_rows(batch: dict, rows, rec: dict) -> dict:
    """The swapped tensors for ``rows`` (LongTensor [N] of batch positions) from the bank records ``rec`` (``DaggerBank.gather``, N records, aligned): a dict with the full ``REBUILT`` tensors (rows replaced, the
    others as they were) and ``lab_s`` (``rc`` re-expressed). ``batch`` is not modified. The rebuild is ORC's (``EgoPerturb._rebuilt``): the same P10 spelling for the base (``kdx_base`` of the kinematic triple in
    the yaw units), the same real-frame rule (``_hist_valid``: a padded or row-0 frame is not a row and keeps its logged value), the same nav rule (``_nav_masks``: token and validity equal NOW's) -- with the
    replacement taken from the own state instead of scaled from the logged one."""
    import torch
    f64 = torch.float64
    dev, F, N = batch["p10f"].device, int(batch["p10f"].shape[1]), int(rows.numel())
    if int(rec["p10f"].shape[1]) != F:
        raise ValueError(f"the bank's history length {rec['p10f'].shape[1]} != the batch's {F}: a bank of another hmax")
    P10 = torch.from_numpy(np.ascontiguousarray(rec["p10f"])).to(dev)                                    # own p10f [N, F, 8], NaN where the row does not exist
    own_idx = torch.from_numpy(rec["j"].astype(np.int64)).to(dev)[:, None] - (F - 1) + torch.arange(F, device=dev)[None]    # the cache step of each own history frame
    own_ok = (own_idx >= 1) & torch.isfinite(P10).all(-1)                                                # a REAL kinematic row (row 0 is NaN in the store; rows before the clip do not exist)
    if not bool(own_ok[:, -1].all()):
        raise ValueError("a bank record whose NOW frame is not a real kinematic row")
    new = {k: batch[k].clone() for k in REBUILT}
    new["p10f"][rows] = torch.where(own_ok[..., None], torch.nan_to_num(P10, 0.0), batch["p10f"][rows])
    new["p10_t"][rows] = P10[:, -1].to(batch["p10_t"].dtype)                                              # p10_t is the p10f row at NOW (ORC: feat2[:, -1])
    new["kin3"][rows] = torch.from_numpy(rec["kin3"]).to(dev, batch["kin3"].dtype)
    new["base"][rows] = torch.from_numpy(D.kdx_base(rec["kin3"], D.KIN_UNITS["yaw"])).to(dev, batch["base"].dtype)
    pose = batch["pose"]
    own_now = rec["pose"][:, -1]
    log_now = batch["pose_now"][rows].double().cpu().numpy()
    pose2 = torch.from_numpy(reanchor_rows(rec["pose"], own_now, log_now)).to(dev)                       # the own poses in the LOGGED frame at r'
    pose2 = torch.where(own_ok[..., None], pose2, pose[rows].to(f64)).to(pose.dtype)
    new["pose"][rows] = pose2
    k = torch.arange(N, device=dev)
    new["pose_t"][rows] = pose2[k, F - 1 - batch["a_t"][rows]]
    new["pose_s"][rows] = pose2[k, F - 1 - batch["a_s"][rows]]
    # nav: ORC's rule (channels 0-2 and 4 untouched; channel 3 on the frames sharing NOW's token and validity), the distance behind NOW taken from the OWN travelled arc
    nav_r = batch["nav"][rows]
    nav_same, nav_boundary = D.EgoPerturb._nav_masks(batch, rows)                                         # real logged frames sharing NOW's token and validity / across a boundary
    nav_ok = nav_same & own_ok & (torch.ones_like(own_ok) if NAV_REBUILD_INVALID else nav_r[..., 4] > 0)
    arc = torch.from_numpy(np.ascontiguousarray(rec["arc"])).to(dev)
    back = arc[:, -1:] - arc                                                                             # own arc from each history frame to NOW [m]
    nav3 = (nav_r[:, -1:, 3].to(f64) + back.to(f64) / 100.0).clamp(0.0, 3.0)                              # d_next grows backwards in time; NOW is the anchor
    new["nav"][rows, :, 3] = torch.where(nav_ok, nav3.to(nav_r.dtype), nav_r[..., 3])
    ls = batch["lab_s"]
    rc = ls["rc"].clone()
    rc[rows] = rc_frame_change(ls["rc"][rows], ls["rc_ok"][rows], batch["pose_s"][rows], new["pose_s"][rows])
    new["lab_s"] = {**ls, "rc": rc}
    new["_nav_kept"] = int(nav_boundary.sum() + (nav_same & ~own_ok).sum())                              # real frames left as logged: across a manoeuvre boundary (ORC's count), or with no own row
    return new


def swap_item(it: dict, rec1: dict) -> dict:
    """The swap of ONE own state (``rec1`` = ``DaggerBank.gather([i])``) into ONE logged training item (``V8FMiniWindows.item``): the item with the ``REBUILT`` ego tensors and ``lab_s.rc`` replaced. The same
    code as the batch path (``swap_rows``) on a batch of one."""
    import torch
    b = D.collate([it])
    upd = swap_rows(b, torch.zeros(1, dtype=torch.long), rec1)
    out = dict(it)
    for k in REBUILT:
        out[k] = upd[k][0].numpy()
    out["lab_s"] = {**it["lab_s"], "rc": upd["lab_s"]["rc"][0].numpy()}
    return out


class DaggerSwap:
    """The DAgger swap applied to a collated training batch just before ``model.loss`` (the trainer's ``b = dag.apply(b, step)``), with ``EgoPerturb``'s contract.

    A row is *eligible* when its drawn window (ci, t) has at least one bank own state with r' == t that passed every cut; an eligible row is *selected* with probability ``p``; a selected row's ego tensors are
    replaced by one eligible own state, drawn uniformly among that window's. The draws come from a private generator seeded by (``seed``, ``step``) -- ONE fixed draw order whatever the eligibility -- and never touch
    the global RNG, so a resumed run draws what an uninterrupted run drew. ``apply`` returns the input batch ITSELF (no copy, no draw) when the lever is off.

    p is a constructor argument with default 0.0 (OFF); this module contains no other value of it."""

    PROBE_STEP = -1

    def __init__(self, bank: DaggerBank | None = None, p: float = DAGGER_DEFAULTS["p"], seed: int = DAGGER_DEFAULTS["seed"], kin_mode: str = "yaw", bank_sha256: str | None = None):
        self.bank, self.p, self.seed, self.kin_mode = bank, float(p), int(seed), str(kin_mode)
        self.bank_sha256 = bank_sha256 if bank_sha256 is not None else (bank.sha256 if bank is not None else None)
        self.last = None

    @property
    def active(self) -> bool:
        return self.p > 0.0

    def built_levers(self) -> dict:
        """What the BUILT swap applies (the G-DVB 'built' side)."""
        return {"dagger_p": self.p, "dagger_seed": self.seed, "dagger_bank": getattr(self.bank, "path", None), "dagger_bank_sha256": self.bank_sha256, "kin_mode": self.kin_mode,
                "bank_hmax": None if self.bank is None else self.bank.hmax, "rebuilt": list(REBUILT), "kept": list(KEPT), "record": list(RECORD), "labels_re_expressed": list(LABELS_RE_EXPRESSED),
                "nav_rebuild_invalid": NAV_REBUILD_INVALID, "constants_sha256": constants_sha256()}

    def law_problems(self) -> list:
        bad = []
        if not (math.isfinite(self.p) and 0.0 <= self.p <= 1.0):
            bad.append(f"dagger_p {self.p!r} is not in [0, 1]")
        if self.seed < 0:
            bad.append(f"dagger_seed {self.seed!r} is negative")
        return bad

    def problems(self) -> list:
        """``law_problems`` plus the declaration's own: the swap needs the yaw representation and a bank; flags set while ``p`` is 0 are 'declared and do nothing'."""
        bad = self.law_problems()
        if self.active and self.kin_mode != "yaw":
            bad.append(f"dagger_p {self.p} needs --kin-mode yaw; got {self.kin_mode!r}")
        if self.active and self.bank is None:
            bad.append(f"dagger_p {self.p} > 0 but no bank is loaded")
        if not self.active:
            moved = [n for n, v in (("dagger_bank", self.bank), ("dagger_bank_sha256", self.bank_sha256), ("dagger_seed", self.seed if self.seed != DAGGER_DEFAULTS["seed"] else None)) if v is not None]
            if moved:
                bad.append(f"{moved} are set while dagger_p is 0: a lever that is declared and does nothing")
        return bad

    def eligible(self, batch: dict):
        """[B] bool: the rows whose (ci, t) has an eligible own state. A function of the batch's ids and the bank only: it does not depend on ``p`` or on any draw."""
        import torch
        if self.bank is None:
            return torch.zeros(int(batch["t"].shape[0]), dtype=torch.bool, device=batch["t"].device)
        cnt, _ = self.bank.candidates(batch["ci"].cpu().numpy(), batch["t"].cpu().numpy())
        return torch.from_numpy(cnt > 0).to(batch["t"].device)

    def _seed_of(self, step: int) -> int:
        return (self.seed * 1_000_003 + int(step)) % (1 << 62)

    def apply(self, batch: dict, step: int, *, p=None) -> dict:
        """The batch with the selected rows swapped for training step ``step`` (a shallow copy; ``batch`` is not modified), or ``batch`` itself when the lever is off. ``p`` overrides the probability for this call
        only (the launch gate's probes: 1.0 = every eligible row, 0.0 = the control)."""
        import torch
        p = self.p if p is None else float(p)
        elig = self.eligible(batch)
        if not p > 0.0:
            self.last = {"step": int(step), "p": 0.0, "n_rows": int(elig.numel()), "n_eligible": int(elig.sum()), "n_selected": 0, "dagger_nav_kept": 0}
            return batch
        bad = self.law_problems()
        if self.kin_mode != "yaw":
            bad.append(f"the swap needs --kin-mode yaw, got {self.kin_mode!r}")
        if self.bank is None:
            bad.append("no bank is loaded")
        if bad:
            raise ValueError("DAgger swap REFUSED: " + "; ".join(bad))
        dev, B = batch["p10f"].device, int(batch["p10f"].shape[0])
        g = torch.Generator().manual_seed(self._seed_of(step))
        u = torch.rand(2, B, generator=g, dtype=torch.float64)                       # selection, which of the window's own states
        sel = (u[0] < p).to(dev) & elig
        rows = sel.nonzero(as_tuple=True)[0]
        self.last = {"step": int(step), "p": p, "n_rows": B, "n_eligible": int(elig.sum()), "n_selected": int(rows.numel()), "dagger_nav_kept": 0, "sel": sel.detach().cpu().clone()}
        out = dict(batch)
        if rows.numel():
            r = rows.cpu()
            idx = self.bank.pick(batch["ci"][r].cpu().numpy(), batch["t"][r].cpu().numpy(), u[1][r].numpy())
            upd = swap_rows(batch, rows, self.bank.gather(idx))
            self.last["dagger_nav_kept"] = upd.pop("_nav_kept")
            self.last["record_idx"] = idx
            out.update(upd)                                                           # the ONE line that applies the swap: delete it and `out` is `batch`
        return out
