#!/usr/bin/env python
"""refcv7 NEW-2 -- the 10 cm map loss's class weights, computed ONCE, from TRAIN GT.

``Project Steering/SPEC_REFCV7.md`` §6.2, item 3: *"median-frequency class
weights, computed once from the TRAIN split's 10 cm GT, then frozen and recorded.
Nothing is tuned on eval."* -- and §13 (Amendment A8, 2026-09-27, registered before
any NEW-2 number): **the weights are sqrt(median-frequency), id ``sqrt_mf``** (option
(c) removed the 0.5 m auxiliary that kept drivable's gradient; plain MF would leave each
big class 2.7 % of it at convergence, sqrt(MF) 10.4 %). This script is that
computation. Run it at launch prep ON THOR over the train cache's clips; the trainer
loads its JSON through ``--map-hires-class-weights`` and stamps the file's sha256 into
config.json.

DEFINITIONS (median-frequency balancing, Eigen & Fergus, ICCV 2015, as written)::

    f_c = n_c / S_c        n_c = seen 10 cm cells of class c
                           S_c = seen cells of the FRAMES in which c is present
    mf:      w_c = median_k(f_k) / f_c
    sqrt_mf: w_c = sqrt(median_k(f_k) / f_c)       <- REGISTERED (A8), the default
    then clipped to <= 25 (the pre-registered clip)

over 8 classes (codes 0..7; 255 = not seen is never counted). ``mf_global`` (``f_c =
n_c / all seen``) is named for reference. The JSON records ``definition_id`` and
``pre_registered`` (true only for ``sqrt_mf`` without a floor).

⭐ THE EXTENT (SPEC_REFCV7 §11.2 / §12, A6/A7): the counts are taken over the map's
DECLARED extent (``--x-max-m`` / ``--y-half-m``; default the A7 extent, 100 m x +-30 m,
read from ``/3`` GT), because class frequencies change with range. The JSON records it
and the trainer's loader REFUSES weights counted on another extent.

⭐ D4 (map-signal audit) -> A8: the analytic shares of every option are in the NEW-2
package (``code/d4_weighting_proposals.py``). ``--weight-floor`` stays available BY NAME
(never by default; a floored file is ``pre_registered: false``).

⛔ THE CLIP SET IS THE CACHE'S, never re-selected (parity): ``--v2-cache`` names the
TRAIN cache, its ``_v2manifest.pt`` is READ (never built -- a missing or stale
manifest refuses; this script must not write into a data directory) and every clip
is resolved in ``--gt-root`` with ``MapGTStore``'s three layouts and opened through
``semantic_map_gt_fine`` (schema, identity, 10 cm spec, code legend). Coverage below
``--min-coverage`` (default 0.90, the trainer's floor) refuses.

⛔ Clip ids never leave this process: the inputs list is hashed as sorted
``<sha12>\\t<bytes>`` lines, and nothing prints an id.

Also writes, with ``--prior-out``, the per-cell class counts of the same pass -- the
FIT side of the positional-prior control (``taniteval.map_hires_metrics.
PositionalPrior.load`` reads it; its ``fit_clips`` are the sha12s).

``--dry-run`` marks the JSON ``"dry_run": true`` with a note. The trainer's loader
REFUSES such a file: a dev-box run on eval files is a code check, not the weights.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                      # stack/

from tanitad.data import semantic_map_gt_fine as F            # noqa: E402

SCHEMA = "tanitad.map_hires_class_weights/1"
CLIP_MAX = 25.0
DEFINITIONS = {
    # ⭐ SPEC_REFCV7 §13 (A8): THE registered weights -- the square root of the Eigen &
    # Fergus median-frequency weights.
    "sqrt_mf": ("sqrt of median-frequency balancing (Eigen & Fergus, ICCV 2015): "
                "w_c = sqrt(median_k(f_k) / f_c), f_c = n_c / S_c, n_c = seen 10 cm cells "
                "of class c, S_c = seen cells of the frames in which c is present; "
                "clipped to <= 25; codes 0..7, 255 (not seen) never counted"),
    # the brief's "median-frequency balancing", as Eigen & Fergus (ICCV 2015) define it
    # (the audit's `MF_present`; the pre-A8 choice)
    "mf": ("median-frequency balancing (Eigen & Fergus, ICCV 2015): "
           "w_c = median_k(f_k) / f_c, f_c = n_c / S_c, n_c = seen 10 cm cells of "
           "class c, S_c = seen cells of the frames in which c is present; "
           "clipped to <= 25; codes 0..7, 255 (not seen) never counted"),
    # the common simplification (the audit's `MF_global`), named for reference
    "mf_global": ("median-frequency balancing, GLOBAL frequency: w_c = median_k(f_k) / f_c, "
                  "f_c = n_c / sum_k n_k over seen 10 cm cells; clipped to <= 25; codes "
                  "0..7, 255 (not seen) never counted"),
}
#: SPEC_REFCV7 §13 (A8): the id refcv7 uses
REGISTERED = "sqrt_mf"
DEFINITION = DEFINITIONS[REGISTERED]


def _read_manifest(cache_dir: Path) -> dict:
    """The cache's manifest, READ-ONLY. Refuses a missing / stale one."""
    import torch
    from tanitad.data.v2_dataset import MANIFEST_NAME, MANIFEST_VERSION, _list_clips
    mp = cache_dir / MANIFEST_NAME
    if not mp.is_file():
        raise SystemExit(f"⛔ no {MANIFEST_NAME} in {cache_dir}: this script READS the "
                         f"manifest and never builds one (it must not write into a "
                         f"data directory). Build it with the trainer's loader first.")
    man = torch.load(mp, map_location="cpu", weights_only=False)
    if man.get("version") != MANIFEST_VERSION:
        raise SystemExit(f"⛔ {mp.name} version {man.get('version')} != "
                         f"{MANIFEST_VERSION}")
    if man.get("files") != _list_clips(str(cache_dir)):
        raise SystemExit(f"⛔ {mp.name} is STALE (its file list is not the cache's): "
                         f"the clip set would not be the cache's clip set")
    cids = [str(c) for c in (man.get("clip_id") or [])]
    if not cids:
        raise SystemExit(f"⛔ {mp.name} carries no clip_id list")
    return {"clip_ids": cids, "n_stack": sorted({int(x) for x in man.get("n_stack") or []})}


def _count_clip(codes: np.ndarray, stride: int, want_prior: bool) -> dict:
    c = codes[::stride]
    T = c.shape[0]
    flat = c.reshape(T, -1)
    per = np.stack([np.bincount(flat[t], minlength=256) for t in range(T)])  # [T,256]
    cls = per[:, :F.N_FINE_CLASSES].astype(np.int64)                         # [T,8]
    seen = cls.sum(1)                                                        # [T]
    present = cls > 0
    out = {"n": cls.sum(0), "S": (present * seen[:, None]).sum(0),
           "present_frames": present.sum(0), "seen": int(seen.sum()),
           "frames": int(T),
           "bad": int(per[:, F.N_FINE_CLASSES:F.NOT_SEEN_CODE].sum())}
    if want_prior:
        out["cell"] = np.stack([(c == k).sum(0) for k in range(F.N_FINE_CLASSES)]
                               ).astype(np.int64)
    return out


def weights_from_counts(n, S, clip_max: float = CLIP_MAX,
                        definition: str = REGISTERED, floor: float = 0.0) -> dict:
    """The definition, on aggregated counts. Pure; the test pins it on literals.
    ``definition``: ``"sqrt_mf"`` (A8, the default), ``"mf"`` (Eigen & Fergus) or
    ``"mf_global"``; ``floor`` > 0 raises every weight to at least ``floor``, applied
    BEFORE the clip."""
    if definition not in DEFINITIONS:
        raise ValueError(f"definition {definition!r} not in {sorted(DEFINITIONS)}")
    n = np.asarray(n, np.float64)
    S = np.asarray(S, np.float64)
    if n.shape != (F.N_FINE_CLASSES,) or S.shape != n.shape:
        raise ValueError("need 8 class counts and 8 presence denominators")
    if np.any(n <= 0):
        miss = [F.FINE_CLASSES[i] for i in np.flatnonzero(n <= 0)]
        raise SystemExit(f"⛔ classes never seen in the input: {miss}. A median-"
                         f"frequency weight for an absent class is undefined; "
                         f"refusing rather than inventing one.")
    f = n / n.sum() if definition == "mf_global" else n / S
    med = float(np.median(f))
    w = med / f
    if definition == "sqrt_mf":
        w = np.sqrt(w)
    if floor > 0.0:
        w = np.maximum(w, float(floor))
    simple = n / n.sum()
    return {"freq": f.tolist(), "median_freq": med,
            "weights_unclipped": w.tolist(),
            "weights": np.minimum(w, clip_max).tolist(),
            "n_clipped": int((w > clip_max).sum()),
            "reference_simple_freq": simple.tolist(),
            "reference_simple_weights": (float(np.median(simple)) / simple).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v2-cache", required=True, type=Path,
                    help="the TRAIN v2 cache whose clips define the split")
    ap.add_argument("--gt-root", required=True, type=Path,
                    help="the SAM3 GT directory (canonical, flat clip id or flat sha12)")
    ap.add_argument("--split", required=True, choices=("train", "dry-run-eval"),
                    help="what the clips ARE; 'dry-run-eval' forces --dry-run")
    ap.add_argument("--out", required=True, type=Path, help="the weights JSON")
    ap.add_argument("--prior-out", type=Path, default=None,
                    help="optional: per-cell class counts (positional-prior FIT set)")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N clips by sha12 order (dry runs only; 0 = all)")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--clip-max", type=float, default=CLIP_MAX)
    ap.add_argument("--definition", choices=sorted(DEFINITIONS), default=REGISTERED,
                    help="'sqrt_mf' = sqrt(Eigen & Fergus median frequency), THE "
                         "registered weights (SPEC_REFCV7 §13, A8; the default); 'mf' = "
                         "median frequency itself; 'mf_global' = n_c / all seen cells. "
                         "Recorded with pre_registered true only for sqrt_mf.")
    ap.add_argument("--weight-floor", type=float, default=0.0,
                    help="raise every weight to at least this value (a named "
                         "alternative; 0 = none, the registered loss); recorded")
    ap.add_argument("--x-max-m", type=float, default=F.EXTENT_REFCV7.x_max_m,
                    help="the map extent ahead the counts cover (SPEC_REFCV7 §12: 100)")
    ap.add_argument("--y-half-m", type=float, default=F.EXTENT_REFCV7.y_half_m,
                    help="the map extent to each side (SPEC_REFCV7 §12: 30)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-run-note", default="")
    a = ap.parse_args(argv)
    if a.split == "dry-run-eval":
        a.dry_run = True
    if a.limit and not a.dry_run:
        raise SystemExit("⛔ --limit selects a SUBSET of the split: only a --dry-run "
                         "may do that (the launch weights use every train clip)")
    if abs(a.clip_max - CLIP_MAX) > 1e-12 and not a.dry_run:
        raise SystemExit(f"⛔ the pre-registered clip is {CLIP_MAX}")
    if a.frame_stride < 1:
        raise SystemExit("⛔ --frame-stride must be >= 1")
    if a.weight_floor < 0.0:
        raise SystemExit("⛔ --weight-floor must be >= 0")
    extent = F.MapExtent(float(a.x_max_m), float(a.y_half_m))
    t0 = time.time()
    man = _read_manifest(a.v2_cache)
    if len(man["n_stack"]) > 1:
        raise SystemExit(f"⛔ the cache mixes n_stack {man['n_stack']}")
    by12 = {F.sha12(c): c for c in man["clip_ids"]}
    if len(by12) != len(man["clip_ids"]):
        raise SystemExit("⛔ two clips of the cache share a sha12")
    order = sorted(by12)
    if a.limit:
        order = order[:int(a.limit)]
    store = F.FineMapGTStore(a.gt_root, max_open=1, extent=extent)
    tot = {"n": np.zeros(8, np.int64), "S": np.zeros(8, np.int64),
           "present_frames": np.zeros(8, np.int64), "seen": 0, "frames": 0, "bad": 0}
    cell = (np.zeros((8,) + tuple(extent.fine_shape), np.int64) if a.prior_out
            else None)
    lines, missing, used = [], [], []
    for i, s12 in enumerate(order):
        cid = by12[s12]
        path = store.resolve(cid)
        if path is None:
            missing.append(s12)
            continue
        gt = F.open_path_fine(path, cid, extent)       # every guard, extent, loud
        codes = gt.read(np.arange(gt.n_frames)).codes
        r = _count_clip(codes, a.frame_stride, cell is not None)
        if r["bad"]:
            raise SystemExit(f"⛔ clip sha12 {s12}: {r['bad']} cells with codes "
                             f"outside the legend")
        for k in ("n", "S", "present_frames"):
            tot[k] += r[k]
        tot["seen"] += r["seen"]
        tot["frames"] += r["frames"]
        if cell is not None:
            cell += r["cell"]
        lines.append(f"{s12}\t{os.path.getsize(path)}")
        used.append(s12)
        if (i + 1) % 200 == 0:
            print(f"[weights] {i + 1}/{len(order)} clips, {time.time() - t0:.0f} s",
                  flush=True)
    cov = len(used) / max(len(order), 1)
    if cov < a.min_coverage:
        raise SystemExit(f"⛔ 10 cm GT for {len(used)}/{len(order)} clips = {cov:.4f} "
                         f"< the floor {a.min_coverage}: refusing weights fitted on a "
                         f"biased subset")
    res = weights_from_counts(tot["n"], tot["S"], a.clip_max, a.definition,
                              float(a.weight_floor))
    inputs_blob = "\n".join(sorted(lines)).encode("utf-8")
    rec = {
        "schema": SCHEMA, "classes": list(F.FINE_CLASSES),
        "definition": DEFINITIONS[a.definition], "definition_id": a.definition,
        "weight_floor": float(a.weight_floor),
        "pre_registered": bool(a.definition == REGISTERED and a.weight_floor == 0.0),
        "extent": {"x_max_m": float(extent.x_max_m), "y_half_m": float(extent.y_half_m)},
        "weights": res["weights"], "weights_unclipped": res["weights_unclipped"],
        "clip_max": float(a.clip_max), "n_clipped": res["n_clipped"],
        "freq": res["freq"], "median_freq": res["median_freq"],
        "counts": tot["n"].tolist(), "presence_denominator": tot["S"].tolist(),
        "presence_frames": tot["present_frames"].tolist(),
        "seen_cells": int(tot["seen"]), "frames": int(tot["frames"]),
        "frame_stride": int(a.frame_stride),
        "share_of_seen": (tot["n"] / max(tot["seen"], 1)).tolist(),
        "reference_simple_freq": res["reference_simple_freq"],
        "reference_simple_weights": res["reference_simple_weights"],
        "split": a.split, "n_clips_in_cache": len(man["clip_ids"]),
        "n_clips_selected": len(order), "n_clips_used": len(used),
        "n_clips_missing_gt": len(missing), "coverage": cov,
        "inputs_sha256": hashlib.sha256(inputs_blob).hexdigest(),
        "inputs_rule": "sha256 of the sorted '<clip sha12>\\t<GT file bytes>' lines",
        "fine_spec": dict(extent.fine_spec),
        "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "dry_run": bool(a.dry_run),
    }
    if a.dry_run:
        rec["dry_run_note"] = (a.dry_run_note or
                               "DRY RUN -- not the launch weights")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if cell is not None:
        a.prior_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.prior_out, counts=cell, fit_clips=np.array(sorted(used)),
                            n_frames=np.int64(tot["frames"]),
                            extent=np.array([float(extent.x_max_m),
                                             float(extent.y_half_m)]))
    print(json.dumps({k: rec[k] for k in ("split", "dry_run", "n_clips_used",
                                          "frames", "weights", "n_clipped",
                                          "inputs_sha256", "elapsed_s")}), flush=True)
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
