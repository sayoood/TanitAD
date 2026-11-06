#!/usr/bin/env python3
"""r9_map_fit_thresholds.py -- the banded map head's DECISION RULE (REFCV9_DESIGN 2.1), fitted on a FIT split of TRAIN clips.

    python r9_map_fit_thresholds.py fit --rast-dir <FIT dump>/rast --dev-hold <DEV_HOLD_SPLIT.json> --arm M1 --out thr_M1.json
                                        [--planes psw,pedge] [--bands 0-10,10-40,40-100] [--population all|corridor]

REFCV9_DESIGN 2.1 (decision rule): *"per-channel, per-band thresholds fitted on a FIT split of TRAIN clips (``--map-hires-class-thresholds``
machinery); never on eval"*, and (scoring) *"per-class best threshold on FIT"*. It was never built: ``map_banded_flag.py`` refuses
``--map-hires-class-thresholds`` under the banded head, and the E1 reader cut every arm at a fixed ``psw >= 128``. MEASURED 2026-11-06
(``dagland/e1/m1_diag/RESULT_M1_CALIB.md``): at 128 the banded head over-flags (16.4 % of cells vs 4.6 %) and its threshold-free sidewalk AP is
BETTER than C0's -- the veto came from the cut.

THE CRITERION IS THE 8-WAY MACHINERY'S, NOT A NEW ONE (``refit_perception_thresholds.py``): per plane, ONE-vs-REST, the cut that maximises the
POOLED IoU of the mask ``{plane >= level}`` against the plane's GT class(es) (``refit_perception_thresholds.best_threshold`` itself is imported
and called: first level on ties), over GT-VALID cells (``gt != 255``) of the FIT rasters. The difference from the 8-way fit is the unit of the
cut only: the 8-way head's cut is on ``logit(p_hat_c)`` over a 4000-bin histogram, one per class; here it is on the uint8 level of the plane the
READER flags on (``psw`` / ``pedge`` / ``pdrv``), one per plane AND PER BAND, so it is exactly ``plane_u8 >= level`` at read time.
  plane -> GT class(es):  psw -> 7 (sidewalk), pedge -> 5 (edge), pdrv -> {1, 2, 3, 4, 6} (the drivable codes, ``map_banded.DRV_CODES``).
  bands (default 0-10 / 10-40 / 40-100 m = the banded head's own bands, so the read's 0-5/5-10/10-20/20-40 m bands nest in them): rows of the 10 cm raster.
  population: ``all`` (default = the 8-way machinery's: every GT-valid cell of the band) or ``corridor`` (|y| < 3 m only, the cells the near-field read scores).

THE FIT SPLIT. Rasters must come from ``r9_e1_dump.py --fit-k-per-ep K`` (a FIT dump): TRAIN clips, K windows per clip at the SAME evenly spaced
positions ``run_diag.py``'s train_diag split uses (``ts[int((j + 0.5) * len(ts) / K)]``), clips = the first N of the TRAIN cache by ``sha256(clip_id)``
that have a 10 cm GT file (``prep_train_diag8.py``). This tool REFUSES, loudly, any dump that is not a FIT dump, that is a smoke, whose raster set is
not the dump's own (``raster_names_sha256``), or that names a clip of DEV or HOLD of ``DEV_HOLD_SPLIT.json`` (``--dev-hold``; the check is run here AND in
the dump). Nothing is ever fitted on eval.

The output (``tanitad.r9_map_thresholds/1``) is read back with the evaluator's own loader before the tool reports success, carries the sha256 of every
input and the checkpoint's md5 from ``dump.json``, and is bound to THAT checkpoint: ``r9_e1_eval.py --map-thresholds`` refuses it against another one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_e1_eval as E                                              # noqa: E402  (the schema, the loader and the corridor live there)
from refit_perception_thresholds import best_threshold              # noqa: E402  (the 8-way machinery's criterion, called not copied)

SCHEMA = E.THR_SCHEMA
#: plane -> the GT class codes that are "positive" for it (MAPSEL: 0 nocls 1 drv 2 lane 3 xwalk 4 arrow 5 edge 6 hatch 7 sidewalk)
PLANE_GT = {"psw": (7,), "pedge": (5,), "pdrv": (1, 2, 3, 4, 6)}
DEFAULT_PLANES = ("psw", "pedge")
DEFAULT_BANDS = "0-10,10-40,40-100"
ROWS_PER_M = 10
WARN_MIN_POS = 1000                    # a (plane, band) with fewer positive cells than this is fitted, but flagged in the file
CRITERION = ("one-vs-rest, per plane and per band: the uint8 level that maximises the pooled IoU of {plane >= level} against the plane's GT class(es) "
             "over GT-valid cells (gt != 255); first level on ties (refit_perception_thresholds.best_threshold); level 0 is never emitted")


class FitRefused(RuntimeError):
    pass


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_bands(s: str) -> list:
    out = []
    for tok in str(s).split(","):
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", tok)
        if not m or int(m.group(2)) <= int(m.group(1)):
            raise FitRefused("--bands %r: want comma-separated lo-hi metres, e.g. 0-10,10-40,40-100" % s)
        out.append([int(m.group(1)), int(m.group(2))])
    if out[0][0] != 0 or any(out[i][1] != out[i + 1][0] for i in range(len(out) - 1)):
        raise FitRefused("--bands %r must start at 0 and be contiguous" % s)
    return out


def raster_hists(path, planes, bands, population="all") -> "np.ndarray | None":
    """[n_planes, n_bands, 2, 256]: per plane and band, the uint8-level histogram of GT-valid NEGATIVE ([..., 0, :]) and POSITIVE ([..., 1, :]) cells.
    None when the raster carries no ``gt`` plane (it cannot be scored, exactly as ``nearfield_per_file`` skips it)."""
    z = np.load(path)
    if "gt" not in z.files:
        return None
    gt = z["gt"]
    c0, c1 = E.CORRIDOR_COLS
    cols = slice(c0, c1) if population == "corridor" else slice(None)
    h = np.zeros((len(planes), len(bands), 2, 256))
    for pi, pl in enumerate(planes):
        if pl not in z.files:
            raise FitRefused("%s: no `%s` plane" % (Path(str(path)).name, pl))
        arr = z[pl]
        if arr.dtype != np.uint8:
            raise FitRefused("%s: plane %s is %s, not uint8" % (Path(str(path)).name, pl, arr.dtype))
        for bi, (lo, hi) in enumerate(bands):
            sl = (slice(lo * ROWS_PER_M, hi * ROWS_PER_M), cols)
            g, v = gt[sl], arr[sl]
            valid = g != 255
            pos = np.isin(g, PLANE_GT[pl]) & valid
            neg = valid & ~pos
            h[pi, bi, 1] = np.bincount(v[pos].astype(np.int64), minlength=256)[:256]
            h[pi, bi, 0] = np.bincount(v[neg].astype(np.int64), minlength=256)[:256]
    return h


def level_from_hist(neg, pos) -> tuple:
    """(level 1..255, IoU) via the 8-way ``best_threshold``. Level 0 (= flag every cell) is not a threshold: if it is the argmax because no cell sits at
    level 0 (the mask is the same as level 1's) the answer is level 1; if it wins on its own merit the plane carries no usable signal and the fit REFUSES."""
    if float(np.sum(pos)) <= 0:
        raise FitRefused("no GT-positive cell: the cut is not fittable -- refusing rather than shipping an arbitrary one")
    i, iou = best_threshold(neg, pos)
    if i == 0:
        if float(neg[0]) + float(pos[0]) == 0.0:
            i = 1
        else:
            raise FitRefused("the best cut is level 0 (flag every cell, IoU %.4f): the plane carries no usable signal on this FIT set" % iou)
    return int(i), float(iou)


def iou_at(neg, pos, level) -> float:
    inter = float(pos[level:].sum())
    pred = inter + float(neg[level:].sum())
    u = pred + float(pos.sum()) - inter
    return inter / u if u > 0 else 0.0


def _load_dump(rast_dir: Path, dump_json) -> tuple:
    dj = Path(dump_json) if dump_json else rast_dir.parent / "dump.json"
    if not dj.is_file():
        raise FitRefused("no dump.json (%s): the FIT split cannot be verified without the dump's own record" % dj)
    return dj, json.loads(dj.read_text(encoding="utf-8"))


def verify_fit_dump(d: dict, rast_dir: Path, dev_hold_path) -> dict:
    """Every REFUSAL that stands between a raster dir and a threshold file. Returns the fit record to stamp."""
    if (d.get("split") or {}).get("set") != "FIT" or not isinstance(d.get("fit"), dict):
        raise FitRefused("dump.json is not a FIT dump (split.set %r, fit block %s): thresholds are fitted only on rasters made by "
                         "`r9_e1_dump.py --fit-k-per-ep K` (TRAIN clips), never on a DEV / HOLD dump" % ((d.get("split") or {}).get("set"), "present" if d.get("fit") else "absent"))
    if d.get("SMOKE_RESTRICTED"):
        raise FitRefused("dump.json is SMOKE_RESTRICTED (a subsample): not a FIT set")
    names = sorted(p.name[:-4] for p in rast_dir.glob("*.npz"))
    if not names:
        raise FitRefused("%s holds no raster" % rast_dir)
    want = d.get("raster_names_sha256")
    got = hashlib.sha256("\n".join(sorted(n + ".npz" for n in names)).encode()).hexdigest()
    if want != got:
        raise FitRefused("the rasters under %s (%d files, names sha256 %s) are not the dump's own (dump.json raster_names_sha256 %s)" % (rast_dir, len(names), got[:12], str(want)[:12]))
    fit_clips = [str(x).lower() for x in d["fit"].get("clips_sha12") or []]
    if not fit_clips or len(set(fit_clips)) != len(fit_clips):
        raise FitRefused("dump.json fit.clips_sha12 is empty or has duplicates")
    dh = Path(dev_hold_path)
    if not dh.is_file():
        raise FitRefused("--dev-hold %s is missing: the FIT / eval disjointness cannot be checked" % dh)
    evalset = set(E.load_window_list(dh, "dev")) | set(E.load_window_list(dh, "hold"))
    clash = sorted(set(fit_clips) & evalset)
    if clash:
        raise FitRefused("%d FIT clip(s) are in DEV / HOLD of %s (e.g. %s): thresholds are never fitted on eval" % (len(clash), dh.name, clash[:3]))
    return {"n_clips": len(fit_clips), "k_per_ep": d["fit"].get("k_per_ep"), "clips_sha12_digest": hashlib.sha256(",".join(sorted(fit_clips)).encode()).hexdigest(),
            "dev_hold_file": dh.name, "dev_hold_sha256": _sha256(dh), "n_eval_clips_checked": len(evalset), "disjoint_from_dev_and_hold": True,
            "rule": d["fit"].get("rule")}


def fit(rast_dir, dev_hold, out, arm, planes=DEFAULT_PLANES, bands_s=DEFAULT_BANDS, population="all", dump_json=None) -> dict:
    rast_dir = Path(rast_dir)
    planes = tuple(planes)
    bad = [p for p in planes if p not in PLANE_GT]
    if bad or not planes:
        raise FitRefused("--planes %s: choose from %s" % (planes, sorted(PLANE_GT)))
    if population not in ("all", "corridor"):
        raise FitRefused("--population must be all | corridor")
    bands = parse_bands(bands_s)
    dj, d = _load_dump(rast_dir, dump_json)
    fitrec = verify_fit_dump(d, rast_dir, dev_hold)
    files = sorted(rast_dir.glob("*.npz"))
    H = None
    n_gt = 0
    eps = set()
    for f in files:
        h = raster_hists(f, planes, bands, population)
        if h is None:
            continue
        H = h if H is None else H + h
        n_gt += 1
        m = re.match(r"^e(\d+)_w", f.name)
        eps.add(m.group(1) if m else f.name)
    if H is None:
        raise FitRefused("no raster under %s carries a gt plane" % rast_dir)
    levels, detail, warns = {}, {}, []
    for pi, pl in enumerate(planes):
        levels[pl], detail[pl] = [], []
        for bi, (lo, hi) in enumerate(bands):
            neg, pos = H[pi, bi, 0], H[pi, bi, 1]
            try:
                lv, iou = level_from_hist(neg, pos)
            except FitRefused as e:
                raise FitRefused("plane %s band %d-%d m: %s" % (pl, lo, hi, e))
            levels[pl].append(lv)
            detail[pl].append({"band_m": [lo, hi], "level": lv, "iou_at_level": iou, "iou_at_128": iou_at(neg, pos, 128),
                               "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "flag_rate_at_level": float((pos[lv:].sum() + neg[lv:].sum()) / max(pos.sum() + neg.sum(), 1))})
            if pos.sum() < WARN_MIN_POS:
                warns.append("plane %s band %d-%d m: only %d GT-positive cells on FIT -- a noisy cut" % (pl, lo, hi, int(pos.sum())))
    obj = {"schema": SCHEMA, "planes": levels, "bands_m": bands,
           "rule": "flag = plane_u8 >= level, per plane and per band (r9_e1_eval.MapThresholds); a plane absent here keeps the fixed 128",
           "criterion": {"name": CRITERION, "population": population, "gt_classes": {p: list(PLANE_GT[p]) for p in planes}, "valid_cells": "gt != 255"},
           "provenance": {"arm": arm, "tool": "stack/scripts/r9_map_fit_thresholds.py", "tool_sha256": _sha256(__file__),
                          "fit_split": "TRAIN (FIT dump)", "fit": fitrec, "n_rasters_fitted": n_gt, "n_episodes_fitted": len(eps),
                          "rast_dir_names_sha256": d.get("raster_names_sha256"),
                          "dump_json": {"sha256": _sha256(dj), "ckpt_md5": d.get("ckpt_md5"), "ckpt_kind": d.get("ckpt_kind"), "step": d.get("step"),
                                        "weights_are_ema": d.get("weights_are_ema"), "config_md5": d.get("config_md5"), "order_digest": d.get("order_digest"),
                                        "tree_commit": d.get("tree_commit")},
                          "ckpt_md5": d.get("ckpt_md5"), "fit_detail": detail, "warnings": warns,
                          "caveat": "bound to THIS checkpoint's planes: valid for this model's rasters only; fitted on TRAIN clips (the arm may have trained on them: "
                                    "the FIT operating point is then optimistic, as for the 8-way refits); one seed"}}
    Path(out).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    back = E.load_map_thresholds(out)                      # the artifact, read back by the evaluator's own loader, is the evidence
    if back.planes != levels:
        raise FitRefused("the file written does not read back to the fitted levels")
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit per-plane, per-band uint8 cuts on a FIT dump's rasters")
    f.add_argument("--rast-dir", required=True, help="<FIT dump>/rast (e*_w*.npz; dump.json is read from its parent unless --dump-json)")
    f.add_argument("--dump-json")
    f.add_argument("--dev-hold", required=True, help="E0i DEV_HOLD_SPLIT.json: the FIT clips must be disjoint from DEV and HOLD")
    f.add_argument("--arm", required=True, help="a label for the provenance block (C0 / C1 / M1 ...)")
    f.add_argument("--planes", default=",".join(DEFAULT_PLANES), help="psw,pedge (the planes the near-field read flags on); pdrv optional")
    f.add_argument("--bands", default=DEFAULT_BANDS)
    f.add_argument("--population", default="all", choices=("all", "corridor"))
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        obj = fit(a.rast_dir, a.dev_hold, a.out, a.arm, tuple(x for x in a.planes.split(",") if x), a.bands, a.population, a.dump_json)
    except FitRefused as e:
        print("[r9-map-fit] REFUSED: %s" % e, file=sys.stderr)
        return 2
    print("wrote %s: %s over %d rasters / %d episodes; bands %s" % (a.out, json.dumps(obj["planes"]), obj["provenance"]["n_rasters_fitted"],
                                                                  obj["provenance"]["n_episodes_fitted"], obj["bands_m"]))
    for w in obj["provenance"]["warnings"]:
        print("[r9-map-fit] WARNING: %s" % w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
