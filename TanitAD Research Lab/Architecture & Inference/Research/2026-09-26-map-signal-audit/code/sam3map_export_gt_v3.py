"""`tanitad.sam3_map_gt/3` RE-EXPORT at the A6/A7 extent (SPEC_REFCV7 §12): 100 m ahead x +-30 m, 10 cm fine codes
[T, 1000, 600] and 0.5 m soft class fractions [T, 9, 200, 120]. Built from the STORED per-clip SAM3 world maps and
the poses/time axis of each clip's `/2` file. SAM3 is not re-run.

⭐ ANCHORED COORDINATES (the audit's convention, SPEC §12.3). Every lateral coordinate is written relative to the
/2 window's origin, so a cell inside the old 60 x 32 m window gets CHARACTER-IDENTICAL arithmetic to the /2
exporter (`sam3map_prod.export_v2ep` + `sam3map_export_gt`):
    10 cm cell (i, j):            x = (i + 0.5) * 0.1        y = -16 + (j - 140 + 0.5) * 0.1
    0.5 m cell (ix, iy), sub (sx, sy) = ((k + 0.5) / 5):
                                  x = (ix + sx) * 0.5        y = -16 + (iy - 28 + sy) * 0.5
    the old window = fine rows 0..599 x cols 140..459; cart rows 0..119 x cols 28..91.

⛔ THE CONTROL, ON EVERY FRAME OF EVERY CLIP: inside the old window the /3 `fine_codes` AND `cart_frac` must be
byte-identical to the clip's /2 arrays. A clip that fails is NOT written, and ANY failure makes the whole export
FAIL (`EXPORT_FAILED.json`; `EXPORT_OK.json` is written only when every clip passed). No silent partial.

Kept from /2, UNCHANGED: `t_query_us`, `t_img_us`, `cam_frame_idx` (the v2ep time axis), `T_world_rig` (the poses)
and the polar / polar48 grids (60 m, +-60 deg: their definitions do not change). Guards of the /2 reader
(`stack/tanitad/data/semantic_map_gt.py`) are copied, not imported (this runs under the production interpreter
`tanitad-edge`, whose environment need not import the stack): schema, rig frame, grid spec, channel order, fraction
scale, array names/shapes/dtypes, `source.clip_sha12 == sha256(clip_id)[:12]`, a non-decreasing `t_img_us`, a finite
`t_query_us`, finite poses; and the world map's own `clip_sha12` and `res`.

Scope: refcv6's TRAIN (`refcv6_train_clips.txt`) + EVAL (`eval139_ids.txt`) clips that have a /2 file in the
trainer's store (`/home/nvidia/data/sam3_corpus/semantic_maps/gt/`). Output: `<out>/<sha12>.sam3mapgt.npz` (the
flat sha12 layout `perception_targets.MapGTStore` resolves), written atomically (`.tmp` + rename), plus
`MANIFEST.jsonl` (one row per clip), `MANIFEST_SUMMARY.json` and the verdict marker. Clip ids never written.
Usage: sam3map_export_gt_v3.py --out DIR [--workers 6] [--limit N] [--clips-from FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GT2_ROOT = Path("/home/nvidia/data/sam3_corpus/semantic_maps/gt")
WM_DIR = Path("/home/nvidia/sam3map/corpus/out")
TRAIN_LIST = Path("/home/nvidia/data/refcv6_train_clips.txt")
EVAL_LIST = Path("/home/nvidia/data/eval139_ids.txt")
SCHEMA2, SCHEMA3 = "tanitad.sam3_map_gt/2", "tanitad.sam3_map_gt/3"
CHANNELS = ("seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text", "non-drivable edge",
            "hatched area", "sidewalk / verge", "not seen")
CART_SPEC2 = {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5, "shape": [120, 64]}
X_MAX, Y_HALF, Y_OLD = 100.0, 30.0, 16.0
FINE_CELL, CART_CELL, SUB = 0.1, 0.5, 5
FINE_SHAPE = (int(round(X_MAX / FINE_CELL)), int(round(2 * Y_HALF / FINE_CELL)))        # (1000, 600)
CART_SHAPE = (int(round(X_MAX / CART_CELL)), int(round(2 * Y_HALF / CART_CELL)))        # (200, 120)
FOFF = int(round((Y_HALF - Y_OLD) / FINE_CELL))                                         # 140
COFF = int(round((Y_HALF - Y_OLD) / CART_CELL))                                         # 28
REQ2 = {"cart_frac": (np.uint8, (None, 9, 120, 64)), "fine_codes": (np.uint8, (None, 600, 320)),
        "t_img_us": (np.int64, (None,)), "t_query_us": (np.float64, (None,)),
        "cam_frame_idx": (np.int32, (None,)), "T_world_rig": (np.float64, (None, 4, 4))}
_SELF_MD5 = None


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def file_hash(p, algo="sha256", chunk=1 << 22) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


# ----------------------------------------------------------- the /2 exporter's arithmetic, copied verbatim
def lookup(wm, xy_world):                                   # sam3map_export_gt.py:30-36
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    i = np.floor((xy_world[:, 0] - x0) / res).astype(np.int64); j = np.floor((xy_world[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    out = np.full(len(xy_world), 255, np.uint8); out[ok] = cls[i[ok], j[ok]]
    return out


def fractions(codes, n_cells):                              # sam3map_export_gt.py:39-43
    ch = np.where(codes == 255, 8, codes).astype(np.int64).reshape(n_cells, SUB * SUB)
    cnt = np.stack([(ch == k).sum(axis=1) for k in range(9)])
    return np.round(cnt * (255.0 / (SUB * SUB))).astype(np.uint8)


# ----------------------------------------------------------- the anchored /3 grids
def fine_xy_anchored():
    fx = (np.arange(FINE_SHAPE[0]) + 0.5) * FINE_CELL
    fy = -Y_OLD + (np.arange(FINE_SHAPE[1]) - FOFF + 0.5) * FINE_CELL
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.c_[FX.ravel(), FY.ravel()]


def cart_points_anchored():
    o = (np.arange(SUB) + 0.5) / SUB
    ix, iy, sx, sy = np.meshgrid(np.arange(CART_SHAPE[0]), np.arange(CART_SHAPE[1]) - COFF, o, o, indexing="ij")
    x = (ix + sx) * CART_CELL; y = -Y_OLD + (iy + sy) * CART_CELL
    return np.c_[x.ravel(), y.ravel()], CART_SHAPE[0] * CART_SHAPE[1]


# ----------------------------------------------------------- the /2 reader's guards, copied (semantic_map_gt.py)
def read_v2(path: Path, s12: str) -> tuple:
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        if "meta_json" not in names:
            raise ValueError("no meta_json")
        meta = json.loads(str(z["meta_json"]))
        arrs = {k: z[k] for k in z.files if k != "meta_json"}
    if meta.get("schema") != SCHEMA2:
        raise ValueError(f"schema {meta.get('schema')!r}")
    if meta.get("frame") != "rig":
        raise ValueError(f"frame {meta.get('frame')!r}")
    cart = meta.get("cartesian") or {}
    for k, v in CART_SPEC2.items():
        if cart.get(k) != v:
            raise ValueError(f"cartesian.{k} {cart.get(k)!r} != {v!r}")
    if tuple(meta.get("channels") or ()) != CHANNELS:
        raise ValueError("channel list/order")
    if meta.get("fraction_scale") != 255:
        raise ValueError("fraction_scale")
    if (meta.get("source") or {}).get("clip_sha12") != s12:
        raise ValueError("source.clip_sha12 != sha12(clip_id)")
    T = None
    for name, (dt, shape) in REQ2.items():
        a = arrs.get(name)
        if a is None:
            raise ValueError(f"missing {name}")
        if a.dtype != np.dtype(dt) or a.ndim != len(shape) or any(e is not None and g != e for g, e in zip(a.shape, shape)):
            raise ValueError(f"{name} {a.shape}/{a.dtype}")
        T = a.shape[0] if T is None else T
        if a.shape[0] != T:
            raise ValueError(f"{name} has {a.shape[0]} frames, expected {T}")
    if np.any(np.diff(arrs["t_img_us"]) < 0) or not np.all(np.isfinite(arrs["t_query_us"])):
        raise ValueError("time axis corrupt")
    if not np.all(np.isfinite(arrs["T_world_rig"])):
        raise ValueError("non-finite pose")
    return meta, arrs, int(T)


def export_clip(job: tuple) -> dict:
    cid, split, out = job
    s12 = sha12(cid)
    row = {"sha12": s12, "split": split}
    t0 = time.time()
    try:
        p2 = GT2_ROOT / f"{cid}.sam3mapgt.npz"
        wmp = WM_DIR / f"{s12}.worldmap.npz"
        if not p2.is_file():
            row["status"] = "NO_V2"
            return row
        if not wmp.is_file():
            row["status"] = "NO_WORLDMAP"
            return row
        meta2, a2, T = read_v2(p2, s12)
        wm = dict(np.load(wmp, allow_pickle=True))
        if str(wm.get("clip_sha12")) != s12:
            raise ValueError("world-map clip_sha12 mismatch")
        if abs(float(wm["res"]) - float((meta2.get("source") or {}).get("world_map_res_m", 0.1))) > 1e-12:
            raise ValueError("world-map res != /2 source.world_map_res_m")
        fxy = fine_xy_anchored()
        cpts, cn = cart_points_anchored()
        fine3 = np.empty((T,) + FINE_SHAPE, np.uint8)
        cart3 = np.empty((T, 9) + CART_SHAPE, np.uint8)
        for n, Tm in enumerate(a2["T_world_rig"]):
            Rm, t = Tm[:2, :2], Tm[:2, 3]
            fine3[n] = lookup(wm, fxy @ Rm.T + t).reshape(FINE_SHAPE)
            cart3[n] = fractions(lookup(wm, cpts @ Rm.T + t), cn).reshape(9, *CART_SHAPE)
        # ---- THE CONTROL: every frame, inside the old window, byte for byte ---------------------------------------
        fin_in = fine3[:, :600, FOFF:FOFF + 320]
        car_in = cart3[:, :, :120, COFF:COFF + 64]
        fdiff = (fin_in != a2["fine_codes"]).reshape(T, -1).sum(axis=1)
        cdiff = (car_in != a2["cart_frac"]).reshape(T, -1).sum(axis=1)
        row["T"] = T
        row["frames_identical_fine"] = int((fdiff == 0).sum())
        row["frames_identical_cart"] = int((cdiff == 0).sum())
        row["diff_bytes_fine"] = int(fdiff.sum())
        row["diff_bytes_cart"] = int(cdiff.sum())
        ok = row["diff_bytes_fine"] == 0 and row["diff_bytes_cart"] == 0
        # arithmetic invariant of the fractions over the WHOLE /3 grid
        sums = cart3.astype(np.int32).sum(axis=1)
        row["cart_bad_sum_cells"] = int((np.abs(sums - 255) > 5).sum())
        ok = ok and row["cart_bad_sum_cells"] == 0
        # information only (the /2 exporter's content stats, on the /3 grid)
        row["seen_share_fine"] = round(float((fine3 != 255).mean()), 4)
        row["drivable_share_cart"] = round(float(cart3[:, 1].mean() / 255.0), 4)
        if not ok:
            row["status"] = "FAIL_CONTROL"
            row["s"] = round(time.time() - t0, 2)
            return row
        # ---- meta /3 -------------------------------------------------------------------------------------------------
        meta3 = json.loads(json.dumps(meta2))
        meta3["schema"] = SCHEMA3
        meta3["cartesian"] = {"x_max_m": X_MAX, "y_half_m": Y_HALF, "cell_m": CART_CELL, "shape": list(CART_SHAPE),
                              "row0": "x in [0, cell_m); col0 is y = -y_half (RIGHT), +y is LEFT",
                              "anchored": "sub-sample (ix, iy, sx, sy), s in (k + 0.5)/5: x = (ix + sx) * 0.5, "
                                          f"y = -16 + (iy - {COFF} + sy) * 0.5"}
        meta3["fine"] = {"x_max_m": X_MAX, "y_half_m": Y_HALF, "cell_m": FINE_CELL, "shape": list(FINE_SHAPE),
                         "codes": "0-7 class, 255 not seen", "row0": "x in [0, 0.1)", "col0": "y = -30 m (RIGHT)",
                         "anchored": f"cell (i, j): x = (i + 0.5) * 0.1, y = -16 + (j - {FOFF} + 0.5) * 0.1"}
        meta3["extent"] = {"x_max_m": X_MAX, "y_half_m": Y_HALF,
                           "rule": "SPEC_REFCV7 §11.2 (A6) census -> §12 (A7): largest 10 m steps with >= 50 % of TRAIN "
                                   "frames seen; 4,369 clips / 78,321 frames"}
        meta3["old_window_v2"] = {"fine_rows": [0, 600], "fine_cols": [FOFF, FOFF + 320],
                                  "cart_rows": [0, 120], "cart_cols": [COFF, COFF + 64],
                                  "identity": "byte-identical to the clip's /2 fine_codes and cart_frac on every frame "
                                              "(checked at export)"}
        meta3["polar_note"] = "polar / polar48 grids unchanged from /2 (copied)"
        meta3["v3_provenance"] = {"from_v2_sha256": file_hash(p2), "worldmap_sha256": file_hash(wmp),
                                  "exporter": "sam3map_export_gt_v3.py (map-signal audit)", "exporter_md5": _SELF_MD5,
                                  "interpreter": sys.executable, "numpy": np.__version__}
        payload = {k: v for k, v in a2.items() if k not in ("fine_codes", "cart_frac")}
        final = Path(out) / f"{s12}.sam3mapgt.npz"
        tmp = Path(out) / f".{s12}.sam3mapgt.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, meta_json=json.dumps(meta3), fine_codes=fine3, cart_frac=cart3, **payload)
        os.replace(tmp, final)
        row["status"] = "PASS"
        row["bytes"] = final.stat().st_size
        row["sha256"] = file_hash(final)
        row["v2_sha256"] = meta3["v3_provenance"]["from_v2_sha256"]
        row["s"] = round(time.time() - t0, 2)
        return row
    except Exception as e:                                                      # noqa: BLE001 -- a FAIL row, never silent
        row["status"] = "FAIL_EXCEPTION"
        row["error"] = f"{type(e).__name__}: {str(e).replace(cid, '<clip>')[:300]}"
        row["trace_tail"] = traceback.format_exc().replace(cid, "<clip>")[-600:]
        row["s"] = round(time.time() - t0, 2)
        return row


def _init(md5):
    global _SELF_MD5
    _SELF_MD5 = md5
    try:
        os.nice(19)
    except OSError:
        pass


def main():
    global _SELF_MD5
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first N clips by sha12")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    a = ap.parse_args()
    _SELF_MD5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in ("EXPORT_OK.json", "EXPORT_FAILED.json"):
        if (out / m).exists():
            raise SystemExit(f"[v3] {out / m} exists: refusing to run into a finished export directory")
    if (out / "MANIFEST.jsonl").exists() and (out / "MANIFEST.jsonl").stat().st_size > 0:
        raise SystemExit(f"[v3] {out / 'MANIFEST.jsonl'} is not empty: an earlier (crashed?) run used this directory. "
                         f"Use a fresh directory; a manifest must describe exactly one run.")
    free = shutil.disk_usage(out).free / 1e9
    if free < a.min_free_gb:
        raise SystemExit(f"[v3] {free:.1f} GB free < {a.min_free_gb} GB")
    train = [l.strip() for l in open(TRAIN_LIST) if l.strip()]
    ev = [l.strip() for l in open(EVAL_LIST) if l.strip()]
    split = {c: "train" for c in train}
    for c in ev:
        if c in split:
            raise SystemExit("[v3] a clip is in both train and eval")
        split[c] = "eval"
    jobs = sorted(((c, s, str(out)) for c, s in split.items()), key=lambda j: sha12(j[0]))
    if a.limit:
        jobs = jobs[: a.limit]
    t0 = time.time()
    print(f"[v3] {len(jobs)} clips ({sum(1 for j in jobs if j[1] == 'train')} train, "
          f"{sum(1 for j in jobs if j[1] == 'eval')} eval); {free:.1f} GB free; exporter md5 {_SELF_MD5}; "
          f"workers {a.workers}", flush=True)
    rows = []
    with open(out / "MANIFEST.jsonl", "a", encoding="utf-8") as man, \
            Pool(int(a.workers), initializer=_init, initargs=(_SELF_MD5,)) as pool:
        for i, r in enumerate(pool.imap_unordered(export_clip, jobs, chunksize=2)):
            rows.append(r)
            man.write(json.dumps(r) + "\n")
            man.flush()
            if r["status"].startswith("FAIL"):
                print(f"[v3] ⛔ {r['sha12']} {r['status']} {r.get('error', '')} fine_diff {r.get('diff_bytes_fine')} "
                      f"cart_diff {r.get('diff_bytes_cart')}", flush=True)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                n_pass = sum(1 for x in rows if x["status"] == "PASS")
                print(f"[v3] {i + 1}/{len(jobs)} done, {n_pass} PASS, "
                      f"{sum(1 for x in rows if x['status'].startswith('FAIL'))} FAIL, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    passed = [r for r in rows if r["status"] == "PASS"]
    failed = [r for r in rows if r["status"].startswith("FAIL")]
    missing = [r for r in rows if r["status"] in ("NO_V2", "NO_WORLDMAP")]
    summ = {"schema": SCHEMA3, "extent": {"x_max_m": X_MAX, "y_half_m": Y_HALF},
            "fine_shape": list(FINE_SHAPE), "cart_shape": list(CART_SHAPE),
            "n_jobs": len(jobs), "n_pass": len(passed), "n_fail": len(failed), "n_missing": len(missing),
            "missing": [{"sha12": r["sha12"], "split": r["split"], "status": r["status"]} for r in missing],
            "failed": [{k: r.get(k) for k in ("sha12", "split", "status", "error", "diff_bytes_fine", "diff_bytes_cart")}
                       for r in failed],
            "n_pass_train": sum(1 for r in passed if r["split"] == "train"),
            "n_pass_eval": sum(1 for r in passed if r["split"] == "eval"),
            "n_frames": int(sum(r["T"] for r in passed)),
            "n_frames_control_identical": int(sum(min(r["frames_identical_fine"], r["frames_identical_cart"]) for r in passed)),
            "control_failures_frames": int(sum(r.get("T", 0) - min(r.get("frames_identical_fine", 0), r.get("frames_identical_cart", 0))
                                               for r in rows if "T" in r)),
            "bytes": int(sum(r["bytes"] for r in passed)), "hours": round((time.time() - t0) / 3600, 3),
            "exporter_md5": _SELF_MD5, "workers": a.workers, "interpreter": sys.executable, "numpy": np.__version__,
            "verdict": "PASS" if (not failed and passed) else "FAIL"}
    json.dump(summ, open(out / "MANIFEST_SUMMARY.json", "w", encoding="utf-8"), indent=1)
    json.dump({k: summ[k] for k in ("verdict", "n_pass", "n_fail", "n_missing", "n_frames", "bytes", "hours",
                                    "exporter_md5")},
              open(out / ("EXPORT_OK.json" if summ["verdict"] == "PASS" else "EXPORT_FAILED.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("missing", "failed")}), flush=True)
    print("ZZV3EXPORT-" + summ["verdict"] + "ZZ", flush=True)


if __name__ == "__main__":
    main()
