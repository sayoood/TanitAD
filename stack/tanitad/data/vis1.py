"""VIS-1 -- the camera-visibility supervision set, with IGNORE semantics (SPEC_REFCV7 §14, A9 R3).

PI, verbatim (2026-09-27): *"based on the compariosn with proven refernce heads, refine our bb head and
validate it"*. SPEC_REFCV7 §14 (A9) registers R3 before any refcv7 box number exists:

    POSITIVE  vis_frac >= 0.30 AND >= 100 visible px
    IGNORE    EVERY other real GT object -- vis 0.05-0.30, too few pixels, AND vis < 0.05
              (a DELIBERATE deviation from the audit, which made vis < 0.05 background:
              an existing object is never taught as "no object")

and SPEC_REFCV7 §15.1 (A10, binding) reconciles the scope: *"Rows removed by visible_target_filter (outside the
120 deg field or the decode box) stay outside the target set, as in A9 R3"*, and the loss mask and the DontCare
scoring use exactly the in-filter non-positives (the 77 of G-BOX-OVERFIT's 16 frames = 28 audit-IGNORE + 49
audit-DROPPED).

⭐ ONE FUNCTION serves the training targets AND the eval scoring: :func:`vis1_split`. The loss
(``tanitad.models.slot_presence``) and the detection metrics (``tanitad.eval.detection_metrics``) both call
it; neither re-spells the rule. It is applied AFTER ``refc_agents.visible_target_filter``: a POSITIVE must
first be a trainer target (120 deg field AND the decode box), exactly as before A9.

⛔⛔ THE SCOPE OF IGNORE (A10 §15.1, binding): IGNORE = the rows INSIDE ``visible_target_filter`` that are not
POSITIVE -- 0.05 <= vis < 0.30, too few pixels, vis < 0.05, no 3-D label, or no silhouette pixel. Excluded from
matching; an UNMATCHED slot whose BEV centre lies within :data:`VIS1_IGNORE_RADIUS_M` of one gets presence weight
0; at eval a detection greedy-matched to one leaves the PR count (DontCare). Rows the filter removes (outside the
120 deg field or the 60 m x 32 m decode box) are NEITHER: they stay outside the target set exactly as before A9.
⚠️ The first build of this module made those rows IGNORE too ("every other real GT row"); A10 reconciled the
scope before any number existed, and ``tests/test_refcv7_vis1.py`` pins the A10 reading with its red arm.

Where ``vis_frac`` comes from: the audit's exact per-pixel ray/cuboid z-buffer (``vis1_zbuffer``, vendored
verbatim), precomputed ONCE per (clip, NOW frame, track) by ``stack/scripts/precompute_vis1_sidecar.py`` into
a SIDECAR this module reads. A row the sidecar does not cover is not guessed: inside the store scope it is a
REFUSAL (:class:`VIS1SidecarError`), outside it the row cannot be a POSITIVE anyway (it fails the filter) and
is IGNORE by the rule above.

Units: ``vis_frac`` dimensionless in [0, 1] (NaN when the cuboid has no silhouette pixel), ``n_vis`` pixels of
the 416x1024 canonical cylinder, radii in metres in the ego BEV plane.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "VIS1_T_POS", "VIS1_PX_MIN", "VIS1_IGNORE_RADIUS_M", "STORE_SCOPE", "SIDECAR_SCHEMA",
    "vis1_positive", "vis1_split", "in_store_scope", "VIS1SidecarError", "VIS1Sidecar",
    "write_sidecar", "file_sha256", "vis1_block_for_rows", "vis1_rule_dict",
]

#: ⛔ THE REGISTERED LITERALS (SPEC_REFCV7 §14 R3). Changing one after a refcv7 box number exists is a
#: goalpost move and needs a dated amendment.
VIS1_T_POS: float = 0.30
VIS1_PX_MIN: int = 100
VIS1_IGNORE_RADIUS_M: float = 2.0

#: Which rows the sidecar STORES: every row with a 3-D label whose float32 centre lies inside the trainer's
#: target region WIDENED by a margin (1 m, 1 deg). ⭐ The margin exists so the sidecar and the trainer can
#: never disagree about a row on the boundary: the trainer's filter runs in float32 torch (``atan2`` on
#: ``|cy|``), the store test in float64 numpy, and a row the filter keeps must be in the store. Rows outside
#: the margined region can never be POSITIVE (they fail the filter), so they are IGNORE without a lookup.
STORE_SCOPE: dict = {"x_min_m": -1.0, "x_max_m": 61.0, "y_half_m": 17.0, "half_angle_deg": 61.0}

SIDECAR_SCHEMA = "tanitad.vis1_sidecar/1"

#: the arrays a sidecar carries, and their dtypes. Rows are sorted by (clip, frame, row).
_ROW_KEYS = {"row": np.int16, "track": np.int64, "n_full": np.int32, "n_img": np.int32,
             "n_vis": np.int32, "vis_rows": np.int32, "cx": np.float32, "cy": np.float32}
_FRAME_KEYS = {"frame_clip": np.int32, "frame_f": np.int32, "frame_ptr": np.int64,
               "frame_n_zh": np.int32, "frame_n_valid": np.int32}
_CLIP_KEYS = {"clip_sha12": "<U12", "clip_frame_ptr": np.int64, "clip_mask_ftheta": np.bool_,
              "clip_split": "<U5"}


def vis1_rule_dict() -> dict:
    """The rule as data, for ``config.json`` and every eval record."""
    return {"positive": f"in visible_target_filter AND vis_frac >= {VIS1_T_POS} AND n_vis >= {VIS1_PX_MIN} px",
            "ignore": "every other row INSIDE visible_target_filter, INCLUDING vis_frac < 0.05, rows with no 3-D "
                      "label and rows with no silhouette pixel (A10 §15.1); rows outside the filter are neither",
            "ignore_loss": f"excluded from matching; an UNMATCHED slot whose BEV centre is within "
                           f"{VIS1_IGNORE_RADIUS_M} m of an IGNORE row gets presence weight 0",
            "ignore_eval": "a detection greedy-matched (one-to-one, score order, over POSITIVE U IGNORE) to an "
                           "IGNORE row leaves the PR count (KITTI DontCare)",
            "t_pos": VIS1_T_POS, "px_min": VIS1_PX_MIN, "ignore_radius_m": VIS1_IGNORE_RADIUS_M,
            "source": "SPEC_REFCV7.md §14 (A9) R3"}


# --------------------------------------------------------------------------------------------------------- #
# the rule                                                                                                   #
# --------------------------------------------------------------------------------------------------------- #
def vis1_positive(vis_frac, n_vis):
    """``vis_frac >= 0.30 AND n_vis >= 100`` elementwise; NaN / unknown is never positive.

    Accepts numpy arrays or torch tensors (returns the same kind, bool)."""
    if torch.is_tensor(vis_frac):
        vf = vis_frac.to(torch.float64)
        nv = torch.as_tensor(n_vis, device=vf.device).to(torch.int64)
        return torch.isfinite(vf) & (vf >= VIS1_T_POS) & (nv >= VIS1_PX_MIN)
    vf = np.asarray(vis_frac, dtype=np.float64)
    nv = np.asarray(n_vis, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(vf) & (vf >= VIS1_T_POS) & (nv >= VIS1_PX_MIN)


def vis1_split(tgt: dict, *, n_full: Tensor, n_vis: Tensor, vis_known: Tensor,
               ranges=None, half_angle_rad: float | None = None) -> dict:
    """⭐ THE ONE FUNCTION: the VIS-1 POSITIVE targets and the IGNORE rows of a target block.

    ``tgt`` is the slot-target dict (``box`` [B, A, 4], ``valid`` [B, A] = the REAL rows, ...), exactly what the
    trainer builds from the batch. ``n_full`` / ``n_vis`` (integer pixel counts) and ``vis_known`` are [B, A],
    from the sidecar via the dataset (``agent_vis_full`` / ``agent_vis_px`` / ``agent_vis_known``).
    ``vis_frac = n_vis / n_full`` is formed HERE in float64 from the integers -- the audit's own arithmetic
    (``vis_zbuf.py:335``), so no float32 round trip can move a row across 0.30.

    Returns ``{"pos": <tgt copy with valid = POSITIVE>, "ignore": [B, A] bool, "in_filter": [B, A] bool,
    "counts": {...}}``. Every other tensor of ``tgt`` is shared, never copied.

    * POSITIVE = valid AND ``visible_target_filter`` AND known AND ``vis_frac >= 0.30`` AND ``n_vis >= 100``;
    * IGNORE   = valid AND ``visible_target_filter`` AND NOT POSITIVE (A10 §15.1 -- the module docstring's ⛔⛔);
    * rows outside the filter are neither (unchanged from before A9).

    ⛔ Nothing here reads a prediction: the split is a property of the LABELS, so train and eval cannot
    disagree about which rows exist.
    """
    from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD, visible_target_filter
    valid = tgt["valid"].to(torch.bool)
    for name, x in (("n_full", n_full), ("n_vis", n_vis), ("vis_known", vis_known)):
        if tuple(x.shape) != tuple(valid.shape):
            raise ValueError(f"[vis1] {name} {tuple(x.shape)} must match valid {tuple(valid.shape)} -- one "
                             f"visibility per target row, in the block's own row order")
    filt = visible_target_filter(
        tgt, ranges=ranges,
        half_angle_rad=(FOV_HALF_ANGLE_RAD if half_angle_rad is None else float(half_angle_rad)))
    in_filter = filt["valid"].to(torch.bool) & valid
    known = vis_known.to(device=valid.device, dtype=torch.bool)
    nf = n_full.to(device=valid.device, dtype=torch.float64)
    nv = n_vis.to(device=valid.device, dtype=torch.float64)
    vis_frac = torch.where(nf > 0, nv / nf.clamp_min(1.0), torch.full_like(nf, float("nan")))
    pos = in_filter & known & vis1_positive(vis_frac, n_vis.to(valid.device))
    ignore = in_filter & ~pos
    out = dict(tgt)
    out["valid"] = pos
    if out.get("zh_mask") is not None:
        out["zh_mask"] = out["zh_mask"].to(torch.bool) & pos
    counts = {"n_real": int(valid.sum()), "n_in_filter": int(in_filter.sum()),
              "n_positive": int(pos.sum()), "n_ignore": int(ignore.sum()),
              "n_outside_filter": int((valid & ~in_filter).sum()),
              "n_ignore_hidden": int((in_filter & known & (vis_frac < 0.05)).sum()),
              "n_in_filter_unknown_vis": int((in_filter & ~known).sum())}
    return {"pos": out, "ignore": ignore, "in_filter": in_filter, "vis_frac": vis_frac, "counts": counts}


def in_store_scope(cx, cy) -> np.ndarray:
    """float64 test of :data:`STORE_SCOPE` (the margined target region)."""
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    s = STORE_SCOPE
    az = np.degrees(np.arctan2(np.abs(cy), cx))
    return ((cx >= s["x_min_m"]) & (cx <= s["x_max_m"]) & (np.abs(cy) <= s["y_half_m"])
            & (az <= s["half_angle_deg"]))


# --------------------------------------------------------------------------------------------------------- #
# the sidecar                                                                                                #
# --------------------------------------------------------------------------------------------------------- #
class VIS1SidecarError(SystemExit):
    """A missing, mismatched or incomplete VIS-1 sidecar. A ``SystemExit`` on purpose: the trainer's refusal
    idiom, so a run cannot catch it by accident and train on guessed visibility."""


def file_sha256(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _npy_bytes(a: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.lib.format.write_array(bio, np.ascontiguousarray(a), allow_pickle=False)
    return bio.getvalue()


def write_sidecar(path, arrays: dict, meta: dict) -> dict:
    """Write a DETERMINISTIC ``.npz`` (fixed zip timestamps, stored, sorted keys) and return its digests.

    ``content_sha256`` is over (key, npy bytes) in sorted key order and is reproducible from the arrays alone;
    ``sha256`` is the file's own digest (what ``config.json`` records). The same inputs give the same bytes,
    so a rebuild on another host must reproduce BOTH.
    """
    missing = [k for k in list(_ROW_KEYS) + list(_FRAME_KEYS) + list(_CLIP_KEYS) if k not in arrays]
    if missing:
        raise ValueError(f"[vis1] write_sidecar: missing arrays {missing}")
    meta = dict(meta)
    meta["schema"] = SIDECAR_SCHEMA
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["meta_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    h = hashlib.sha256()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for k in sorted(payload):
            b = _npy_bytes(payload[k])
            h.update(k.encode("utf-8"))
            h.update(b)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, b)
    tmp.replace(p)
    return {"sha256": file_sha256(p), "content_sha256": h.hexdigest(), "bytes": p.stat().st_size}


class VIS1Sidecar:
    """Read-only VIS-1 sidecar: ``(clip sha12, NOW frame f, row) -> (n_full, n_img, n_vis, vis_rows)``.

    ``f`` is the 2-D agent join's ``frame_idx`` -- the index ``V3Dataset._agent_item(ep, f)`` receives, i.e.
    ``f = t + W - 1`` for a window starting at ``t``. ``row`` is the agent's position in that join record
    (= its row in the padded target block, before any truncation). ``track`` (the join's ``track_id`` as an
    integer) and the float32 centre are stored beside it so a lookup VERIFIES the row it is handed instead of
    trusting a position.

    Loaded arrays are plain numpy buffers, so forked data-loader workers share them without copying.
    """

    def __init__(self, path):
        p = Path(path)
        if not p.exists():
            raise VIS1SidecarError(f"[vis1] ⛔ sidecar not found: {p}. VIS-1 is ON, and a visibility that is "
                                   f"not in the sidecar is never guessed.")
        self.path = str(p)
        self.sha256 = file_sha256(p)
        with np.load(p, allow_pickle=False) as z:
            self.meta = json.loads(bytes(z["meta_json"].tobytes()).decode("utf-8"))
            if self.meta.get("schema") != SIDECAR_SCHEMA:
                raise VIS1SidecarError(f"[vis1] ⛔ {p.name} declares schema {self.meta.get('schema')!r}, "
                                       f"expected {SIDECAR_SCHEMA!r}")
            a = {k: z[k] for k in list(_ROW_KEYS) + list(_FRAME_KEYS) + list(_CLIP_KEYS)}
        for k, dt in {**_ROW_KEYS, **_FRAME_KEYS, **_CLIP_KEYS}.items():
            if np.dtype(a[k].dtype) != np.dtype(dt):
                raise VIS1SidecarError(f"[vis1] ⛔ {p.name}: array {k} is {a[k].dtype}, expected {np.dtype(dt)}")
        self.a = a
        self.clip_index = {str(s): i for i, s in enumerate(a["clip_sha12"].tolist())}
        if len(self.clip_index) != len(a["clip_sha12"]):
            raise VIS1SidecarError(f"[vis1] ⛔ {p.name}: duplicate clip sha12 entries")
        self.n_clips = int(len(a["clip_sha12"]))
        self.n_frames = int(len(a["frame_f"]))
        self.n_rows = int(len(a["row"]))
        if int(a["clip_frame_ptr"][-1]) != self.n_frames or int(a["frame_ptr"][-1]) != self.n_rows:
            raise VIS1SidecarError(f"[vis1] ⛔ {p.name}: pointer arrays do not close "
                                   f"({int(a['clip_frame_ptr'][-1])}/{self.n_frames} frames, "
                                   f"{int(a['frame_ptr'][-1])}/{self.n_rows} rows)")
        scope = self.meta.get("store_scope")
        if scope != STORE_SCOPE:
            raise VIS1SidecarError(f"[vis1] ⛔ {p.name} was stored over scope {scope}, this code expects "
                                   f"{STORE_SCOPE}: a row inside the expected scope could be absent")

    # -- identity ---------------------------------------------------------------------------------------- #
    def stamp(self) -> dict:
        """What ``config.json`` records (``vis1.sidecar``)."""
        return {"path": self.path, "sha256": self.sha256,
                "content_sha256": self.meta.get("content_sha256_of_arrays"),
                "schema": SIDECAR_SCHEMA, "n_clips": self.n_clips, "n_frames": self.n_frames,
                "n_rows": self.n_rows, "splits": self.meta.get("splits"),
                "sources": self.meta.get("sources"), "zbuffer": self.meta.get("zbuffer"),
                "store_scope": self.meta.get("store_scope"), "window": self.meta.get("window"),
                "n_stack": self.meta.get("n_stack")}

    def require_clips(self, sha12s, *, where: str) -> None:
        miss = sorted({str(s) for s in sha12s} - set(self.clip_index))
        if miss:
            raise VIS1SidecarError(
                f"[vis1] ⛔ {where}: {len(miss)} clip(s) of this dataset are not in the VIS-1 sidecar "
                f"{Path(self.path).name} (e.g. {miss[:4]}). A missing clip would train every one of its targets "
                f"as IGNORE while config.json states VIS-1 -- refusing rather than guessing.")
        nf = [s for s in {str(x) for x in sha12s}
              if not bool(self.a["clip_mask_ftheta"][self.clip_index[s]])]
        if nf and not bool(self.meta.get("allow_bounds_only_mask", False)):
            raise VIS1SidecarError(f"[vis1] ⛔ {where}: {len(nf)} clip(s) were z-buffered WITHOUT their f-theta "
                                   f"observed mask (e.g. {nf[:4]})")

    def frame(self, sha12: str, f: int):
        """``(lo, hi)`` row slice of ``(sha12, f)``, or ``None`` when the sidecar has no such frame."""
        ci = self.clip_index.get(str(sha12))
        if ci is None:
            return None
        a = self.a
        f0, f1 = int(a["clip_frame_ptr"][ci]), int(a["clip_frame_ptr"][ci + 1])
        ff = a["frame_f"][f0:f1]
        k = int(np.searchsorted(ff, int(f)))
        if k >= len(ff) or int(ff[k]) != int(f):
            return None
        fi = f0 + k
        return int(a["frame_ptr"][fi]), int(a["frame_ptr"][fi + 1])


def vis1_block_for_rows(sidecar: VIS1Sidecar, sha12: str, f: int, *, box, track_ids, zh_mask, pad: int,
                        order=None):
    """The dataset-side join: ``(n_full [pad] i32, n_vis [pad] i32, known [pad] bool)`` for ONE window.

    ``box`` [n, 4] float32 (cx, cy, l, w) and ``zh_mask`` [n] are the VALID rows of the target block exactly as
    ``V3Dataset._agent_item`` built them, ``track_ids`` the join's ids in the same order, ``order`` the
    truncation permutation ``_agent_item`` applied (``None`` = identity, the default ``--agent-pad``).

    ⛔ REFUSES (``VIS1SidecarError``) when the frame is absent, when an in-scope row with a 3-D label is absent,
    or when the stored track id / float32 centre disagree with the row handed in -- each of those means the
    sidecar was built on a different join, split or clock than the run, and a guessed visibility is exactly the
    silent label corruption VIS-1 exists to remove.
    """
    box = np.asarray(box, dtype=np.float32).reshape(-1, 4)
    n = int(box.shape[0])
    zh = np.asarray(zh_mask, dtype=bool).reshape(-1)[:n]
    n_full = np.zeros(int(pad), dtype=np.int32)
    n_vis = np.zeros(int(pad), dtype=np.int32)
    known = np.zeros(int(pad), dtype=bool)
    if n == 0:
        return n_full, n_vis, known
    sl = sidecar.frame(sha12, f)
    if sl is None:
        raise VIS1SidecarError(f"[vis1] ⛔ clip {sha12} frame {f}: labelled in the join but ABSENT from the "
                               f"VIS-1 sidecar {Path(sidecar.path).name}")
    lo, hi = sl
    a = sidecar.a
    rows = a["row"][lo:hi].astype(np.int64)
    pos = {int(r): lo + k for k, r in enumerate(rows.tolist())}
    src_row = (np.arange(n, dtype=np.int64) if order is None else np.asarray(order, dtype=np.int64)[:n])
    scope = in_store_scope(box[:, 0], box[:, 1])
    tids = list(track_ids)
    for j in range(n):
        if not (zh[j] and scope[j]):
            continue                     # outside the store scope, or no 3-D label: IGNORE, never positive
        k = pos.get(int(src_row[j]))
        if k is None:
            raise VIS1SidecarError(f"[vis1] ⛔ clip {sha12} frame {f} row {int(src_row[j])}: an in-scope row "
                                   f"with a 3-D label is ABSENT from the sidecar")
        tid = str(tids[j])
        if not tid.isdigit() or int(tid) != int(a["track"][k]):
            raise VIS1SidecarError(f"[vis1] ⛔ clip {sha12} frame {f} row {int(src_row[j])}: track "
                                   f"{tid!r} != sidecar {int(a['track'][k])} -- a different join")
        if float(a["cx"][k]) != float(box[j, 0]) or float(a["cy"][k]) != float(box[j, 1]):
            raise VIS1SidecarError(f"[vis1] ⛔ clip {sha12} frame {f} row {int(src_row[j])}: centre "
                                   f"({box[j, 0]}, {box[j, 1]}) != sidecar ({a['cx'][k]}, {a['cy'][k]})")
        n_full[j] = int(a["n_full"][k])
        n_vis[j] = int(a["n_vis"][k])
        known[j] = True
    return n_full, n_vis, known
