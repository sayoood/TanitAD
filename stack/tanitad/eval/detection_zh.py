"""refcv8 WP-C fix 5 -- the box head's z / h error BY RANGE, with the beyond-30-m bins flagged LOW-TRUST. OPT-IN, read-only.

Source: D3 ``TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility`` (RESULT.md,
"Boxes": base height off the ground plane by > 1.5 m for **0.9 % of boxes < 30 m, 11.7 % at 30-100 m, 25.9 % > 100 m** --
"treat z as near-field only"; ``raw/c3b_agents_detail.json`` ``base_face_by_range``).

THE DEFECT IS IN THE LABEL, NOT THE HEAD. The 3-D join's ``center_z`` / ``size_z`` are only trustworthy near the ego; a z or h
error measured against a GT whose own base face floats by > 1.5 m is a measurement of the LABEL. The trainer's z / h terms
(``loss_z`` / ``loss_h``) and the matched-pair ``pair_z_err`` average over every range, so a headline z number mixes a trusted
near field with an untrusted far field. This module reports z / h per range bin, derives the TRUST of each bin from D3's table
(not from a hand-set constant) and puts a near-field-only headline beside it.

* range = the matched ground truth's BEV distance ``hypot(x, y)`` from the ego (D3's bins are in the same metric);
* a D3 range bin is LOW-TRUST when more than ``low_trust_frac`` (5 %) of its boxes have |base height| > 1.5 m; ``near_field_m`` is the
  lowest lower edge of a low-trust D3 bin (= 30 m on D3's numbers); a report bin is TRUSTED iff its upper edge <= ``near_field_m``;
* packs must be built with ``detection_metrics.window_packs(.., with_zh_range=True)`` -- the pack then carries
  ``pair_zh_range`` and ``pair_h_err`` aligned with ``pair_z_err``. A pack without them is REFUSED, never read as "no z error".

DEFAULT: ``zh_range_keys(packs, head, None)`` is ``{}`` -- a default run emits no new key and the trainer's declared z / h terms
are untouched. ⛔ Nothing here is read by the planner.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np

__all__ = ["ZH_TRUST_SCHEMA", "RANGE_BINS_M", "ZH_KEYS", "bin_key", "load_zh_trust", "is_trusted_bin", "zh_range_keys",
           "zh_key_names"]

ZH_TRUST_SCHEMA = "tanitad.det_zh_trust/1"
#: report bins (metres of BEV range). The box head's decode box ends at 60 m; the last bin catches the corners.
RANGE_BINS_M: tuple = ((0.0, 30.0), (30.0, 60.0), (60.0, math.inf))
#: per-bin key suffixes, ``eval_{head}_zh_<bin>_<suffix>``
ZH_KEYS: tuple = ("n", "z_mae", "z_med", "h_mae", "h_med", "low_trust")
_NAN = float("nan")


def bin_key(lo: float, hi: float) -> str:
    """``0_30``, ``30_60``, ``60_inf``."""
    f = lambda v: "inf" if math.isinf(v) else f"{v:g}"          # noqa: E731
    return f"{f(lo)}_{f(hi)}"


def load_zh_trust(path) -> tuple[dict, dict]:
    """Read ``tanitad.det_zh_trust/1`` -> ``({"near_field_m", "low_trust_frac", "bins"}, stamp)``.

    The file holds D3's ``base_face_by_range`` rows (``lo_m``, ``hi_m``, ``n``, ``frac_abs_gt_1.5``) and ``low_trust_frac``.
    ``near_field_m`` is DERIVED here: the lowest ``lo_m`` of a bin with ``frac_abs_gt_1.5 > low_trust_frac`` (``inf`` if none).
    ⛔ REFUSES another schema, a missing/empty bins table, a fraction outside [0, 1], non-contiguous bins."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise ValueError(f"zh trust file {str(p)!r} does not exist")
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if d.get("schema") != ZH_TRUST_SCHEMA:
        raise ValueError(f"{p.name}: schema {d.get('schema')!r} != {ZH_TRUST_SCHEMA!r}")
    thr = float(d.get("low_trust_frac", float("nan")))
    if not (math.isfinite(thr) and 0.0 <= thr <= 1.0):
        raise ValueError(f"{p.name}: low_trust_frac {d.get('low_trust_frac')!r} must be in [0, 1]")
    bins = d.get("label_base_height_error")
    if not isinstance(bins, dict) or not bins:
        raise ValueError(f"{p.name}: no `label_base_height_error` table")
    rows = []
    for name, b in bins.items():
        lo = float(b["lo_m"])
        hi = float("inf") if b["hi_m"] is None else float(b["hi_m"])
        fr = float(b["frac_abs_gt_1.5"])
        if not (0.0 <= fr <= 1.0 and lo >= 0.0 and hi > lo):
            raise ValueError(f"{p.name}: bin {name!r} lo {lo} hi {hi} frac {fr} is not a valid range bin")
        rows.append((lo, hi, fr, name))
    rows.sort()
    for a, b in zip(rows, rows[1:]):
        if a[1] != b[0]:
            raise ValueError(f"{p.name}: bins {a[3]!r} and {b[3]!r} are not contiguous")
    low = [lo for lo, _hi, fr, _n in rows if fr > thr]
    near = min(low) if low else math.inf
    cfg = {"near_field_m": float(near), "low_trust_frac": thr,
           "bins": {n: {"lo_m": lo, "hi_m": hi, "frac_abs_gt_1.5": fr} for lo, hi, fr, n in rows}}
    return cfg, {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "schema": ZH_TRUST_SCHEMA,
                 "near_field_m": cfg["near_field_m"], "provenance": d.get("provenance")}


def is_trusted_bin(lo: float, hi: float, near_field_m: float) -> bool:
    """A report bin is trusted iff it lies wholly inside the trusted near field."""
    return float(hi) <= float(near_field_m)


def zh_key_names(head: str = "box3d") -> list:
    ks = []
    for lo, hi in RANGE_BINS_M:
        ks += [f"eval_{head}_zh_{bin_key(lo, hi)}_{k}" for k in ZH_KEYS]
    ks += [f"eval_{head}_zh_nearfield_{k}" for k in ("n", "z_mae", "z_med", "h_mae", "h_med", "range_m")]
    ks += [f"eval_{head}_zh_lowtrust_pair_frac"]
    return ks


def _stats(z: np.ndarray, h: np.ndarray) -> dict:
    n = int(z.size)
    hh = h[np.isfinite(h)]
    return {"n": float(n),
            "z_mae": float(z.mean()) if n else _NAN, "z_med": float(np.median(z)) if n else _NAN,
            "h_mae": float(hh.mean()) if hh.size else _NAN, "h_med": float(np.median(hh)) if hh.size else _NAN}


def zh_range_keys(packs: list, head: str = "box3d", trust: dict | None = None) -> dict:
    """``eval_{head}_zh_*`` keys over the matched, z/h-labelled pairs of ``packs`` (see :func:`zh_key_names`).

    ``trust=None`` returns ``{}`` (a default run emits nothing). Per range bin: ``n`` (pairs), mean / median |dz| and |dh|,
    ``low_trust`` (1.0 beyond the near field). The ``nearfield`` keys pool ONLY the trusted bins - that is the z / h number
    that may be quoted; ``lowtrust_pair_frac`` is the share of pairs the headline therefore excludes. NaN = undefined."""
    if trust is None:
        return {}
    if not packs:
        raise ValueError("no packs")
    for k in ("pair_z_err", "pair_zh_range", "pair_h_err"):
        if k not in packs[0]:
            raise ValueError(f"packs carry no {k!r}: build them with detection_metrics.window_packs(.., with_zh_range=True) "
                             f"(a pack without it must not be read as 'no z error')")
    z = np.concatenate([np.asarray(pk["pair_z_err"], np.float64) for pk in packs])
    r = np.concatenate([np.asarray(pk["pair_zh_range"], np.float64) for pk in packs])
    h = np.concatenate([np.asarray(pk["pair_h_err"], np.float64) for pk in packs])
    if not (z.shape == r.shape == h.shape):
        raise ValueError(f"misaligned pair arrays: z {z.shape}, range {r.shape}, h {h.shape}")
    near = float(trust["near_field_m"])
    out, trusted_mask = {}, np.zeros(z.shape, bool)
    for lo, hi in RANGE_BINS_M:
        m = (r >= lo) & (r < hi)
        s = _stats(z[m], h[m])
        lt = 0.0 if is_trusted_bin(lo, hi, near) else 1.0
        if lt == 0.0:
            trusted_mask |= m
        for k in ("n", "z_mae", "z_med", "h_mae", "h_med"):
            out[f"eval_{head}_zh_{bin_key(lo, hi)}_{k}"] = s[k]
        out[f"eval_{head}_zh_{bin_key(lo, hi)}_low_trust"] = lt
    s = _stats(z[trusted_mask], h[trusted_mask])
    for k in ("n", "z_mae", "z_med", "h_mae", "h_med"):
        out[f"eval_{head}_zh_nearfield_{k}"] = s[k]
    out[f"eval_{head}_zh_nearfield_range_m"] = near
    out[f"eval_{head}_zh_lowtrust_pair_frac"] = float((~trusted_mask).sum() / z.size) if z.size else _NAN
    return out
