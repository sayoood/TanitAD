"""MEASURE every input the REFe navtest evaluation reads, on the dev box, for sub200 and for the full navtest.

Read-only. Writes input_sizes.json beside this file. Sizes are APPARENT bytes (what a transfer moves), not the
exFAT allocation (D: has a large cluster, so `du` without --apparent-size over-reports small files).

The input list is traced from the code, not assumed:
  eval/eval_checkpoint.py      env_driverl(): DZ_ROOT, NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT, REFE_BACKBONE_ROOT, ...
  eval/refe_navtest_seam.py    W3_EXPORT, TEST_DB_DIR (<log>.db), --frames (<log>_<CAM>.zip), --ckpt
  refe/planner.py              FrameResolver (DB image table + frame zips), calib_table.read_db (DB camera table),
                               _goal_for -> code/augment_routes.py -> nuplan...driverl_runtime_map_features (map API)
  refe/ckpt_io.py              partial snapshot -> load_dinov3 (REFE_BACKBONE_ROOT/dinov3-vitl16/model.safetensors)
  eval/score_navtest_refe.py   W3 run_v1.py -> navsim_v1_win.py (NAVSIM venv): metric_cache_navtest,
                               OPENSCENE_DATA_ROOT/navsim_logs/test/<log>.pkl, E1 navsim_win.py, v1.1 tree, nuplan-devkit
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
import zipfile

OUT = os.path.dirname(os.path.abspath(__file__))
PKG = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
W3 = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest"
SUB = f"{W3}/raw/A1_sub200_tokens.json"
FULL = "D:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json"
DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
TEST_ZIP = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1_test.zip"
MAPS = "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
MAPS_ZIP = "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.1.zip"
MAPS_HARNESS = "D:/Archive/devbox-C/navsim/data/maps"
FRAMES = "D:/Projects/TanitAD/data/refe_navtest/frames"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
NAVSIM_LOGS = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
META_TGZ = "D:/Archive/devbox-C/navsim/data/openscene_metadata_test.tgz"
MCACHE = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
CAM_TGZ = "D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera"
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch016.pt"
BACKBONE = "D:/Projects/TanitAD/data/backbones/dinov3-vitl16"
V11 = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
V11_ZIP = "D:/Archive/devbox-C/navsim/navsim-v1.1-3e8291bfa89ff247231e0227778840cd0a036896.zip"
NUPLAN_DEVKIT_NAVSIM = "C:/Users/Admin/navsim-crun/nuplan-devkit"
DZ = "C:/Users/Admin/dz/DriveZero"
E1 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
      "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")
EV6 = ("C:/Users/Admin/ev6/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code")
VENV_DRIVERL = "C:/Users/Admin/venvs/driverl-eval"
VENV_NAVSIM = "C:/Users/Admin/navsim-crun/venv"


def fsize(p):
    return os.path.getsize(p) if os.path.isfile(p) else None


def dsize(d, suffix=None, skip_dirs=("__pycache__", ".git", ".pytest_cache")):
    n = b = 0
    if not os.path.isdir(d):
        return {"bytes": None, "files": None, "exists": False}
    for dp, dn, fns in os.walk(d):
        dn[:] = [x for x in dn if x not in skip_dirs]
        for fn in fns:
            if suffix and not fn.endswith(suffix):
                continue
            try:
                b += os.path.getsize(os.path.join(dp, fn))
                n += 1
            except OSError:
                pass
    return {"bytes": b, "files": n, "exists": True}


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    sub = json.load(open(SUB, encoding="utf-8"))
    full = json.load(open(FULL, encoding="utf-8"))
    sets = {}
    for name, doc in (("sub200", sub), ("full", full)):
        toks = list(doc["tokens"])
        logs = sorted({doc["token_log"][t] for t in toks})
        sets[name] = {"tokens": toks, "logs": logs, "token_log": doc["token_log"]}
    R = {"measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": "dev box",
         "token_files": {"sub200": {"path": SUB, "bytes": fsize(SUB), "n_tokens": len(sets["sub200"]["tokens"]),
                                    "n_logs": len(sets["sub200"]["logs"])},
                         "full": {"path": FULL, "bytes": fsize(FULL), "n_tokens": len(sets["full"]["tokens"]),
                                  "n_logs": len(sets["full"]["logs"])}}}
    assert set(sets["sub200"]["logs"]) <= set(sets["full"]["logs"])

    # ---- 1. nuPlan test DBs (seam: scenario build, frame index, calibration, route) ------------------------------
    zi = {}
    if os.path.exists(TEST_ZIP):
        with zipfile.ZipFile(TEST_ZIP) as z:
            for i in z.infolist():
                if i.filename.endswith(".db"):
                    zi[os.path.basename(i.filename)[:-3]] = (i.compress_size, i.file_size, i.compress_type)
    dbs = {}
    for name, s in sets.items():
        sz = [fsize(f"{DB_DIR}/{lg}.db") for lg in s["logs"]]
        cz = [zi.get(lg, (None, None, None)) for lg in s["logs"]]
        dbs[name] = {"n_logs": len(s["logs"]), "n_present": sum(x is not None for x in sz),
                     "bytes_on_disk": sum(x for x in sz if x), "max_db_bytes": max(x for x in sz if x),
                     "zip_compressed_bytes": sum(c[0] for c in cz if c[0] is not None),
                     "zip_uncompressed_bytes": sum(c[1] for c in cz if c[1] is not None),
                     "n_in_zip": sum(c[0] is not None for c in cz),
                     "zip_methods": sorted({c[2] for c in cz if c[2] is not None})}
    R["nuplan_test_dbs"] = {"dir": DB_DIR, "zip": TEST_ZIP, "zip_bytes": fsize(TEST_ZIP),
                            "zip_db_members": len(zi), **dbs}

    # which map city does each log need (the DB's own log.map_version)
    city = {}
    for lg in sets["full"]["logs"]:
        c = sqlite3.connect(f"file:{DB_DIR}/{lg}.db?mode=ro", uri=True)
        try:
            city[lg] = c.execute("SELECT map_version FROM log").fetchone()[0]
        finally:
            c.close()
    maps = {"dir": MAPS, "total": dsize(MAPS), "per_city": {}, "zip": MAPS_ZIP, "zip_bytes": fsize(MAPS_ZIP),
            "harness_copy_dir": MAPS_HARNESS, "harness_copy_total": dsize(MAPS_HARNESS)}
    for d in sorted(os.listdir(MAPS)):
        if os.path.isdir(f"{MAPS}/{d}"):
            maps["per_city"][d] = dsize(f"{MAPS}/{d}")
    for name, s in sets.items():
        cs = sorted({city[lg] for lg in s["logs"]})
        maps[f"cities_{name}"] = cs
        maps[f"bytes_needed_{name}"] = sum(maps["per_city"][c]["bytes"] for c in cs if c in maps["per_city"]) + \
            (fsize(f"{MAPS}/nuplan-maps-v1.0.json") or 0)
    # the maps zip: compressed bytes per city (what a range fetch of only the needed cities would move)
    if os.path.exists(MAPS_ZIP):
        with zipfile.ZipFile(MAPS_ZIP) as z:
            per = {}
            for i in z.infolist():
                parts = i.filename.replace("\\", "/").split("/")
                key = next((p for p in parts if p in maps["per_city"]), "other")
                per.setdefault(key, [0, 0, 0])
                per[key][0] += i.compress_size
                per[key][1] += i.file_size
                per[key][2] += 1
            maps["zip_members_by_city"] = {k: {"compressed": v[0], "uncompressed": v[1], "members": v[2]}
                                           for k, v in per.items()}
    R["nuplan_maps"] = maps

    # ---- 2. camera frames (seam: planner._image_for) -------------------------------------------------------------
    rows = json.load(open(f"{FRAMES}/tokens_frames.json", encoding="utf-8"))
    by_tok = {r["token"]: r for r in rows}
    zip_members = {}
    zbytes = {}
    for fn in os.listdir(FRAMES):
        if fn.endswith(".zip"):
            p = f"{FRAMES}/{fn}"
            zbytes[fn] = fsize(p)
            with zipfile.ZipFile(p) as z:
                zip_members[fn[:-4]] = {i.filename: i.file_size for i in z.infolist()}
    fr = {"dir": FRAMES, "n_zips": len(zbytes), "zips_bytes_all": sum(zbytes.values()),
          "manifest": json.load(open(f"{FRAMES}/MANIFEST.json", encoding="utf-8"))}
    for name, s in sets.items():
        jpg_bytes = n_jpg = 0
        missing = 0
        seen = set()
        for t in s["tokens"]:
            r = by_tok.get(t)
            if r is None:
                missing += 1
                continue
            for rel in r["cams"]:
                cam = rel.replace("\\", "/").split("/")[-2]
                key = f"{r['log']}_{cam}"
                base = os.path.basename(rel)
                if (key, base) in seen:
                    continue
                seen.add((key, base))
                sz = zip_members.get(key, {}).get(base)
                if sz is None:
                    missing += 1
                    continue
                jpg_bytes += sz
                n_jpg += 1
        zb = sum(v for k, v in zbytes.items() if any(k.startswith(lg + "_CAM_") for lg in s["logs"]))
        fr[name] = {"n_jpg": n_jpg, "jpg_bytes": jpg_bytes, "missing": missing,
                    "zips_of_these_logs_bytes": zb, "mean_jpg_bytes": round(jpg_bytes / max(n_jpg, 1))}
    R["frames"] = fr
    # the public source on disk: OpenScene test camera shards (streamed ONCE by build_navtest_frames.py)
    shards = {fn: fsize(f"{CAM_TGZ}/{fn}") for fn in sorted(os.listdir(CAM_TGZ)) if fn.endswith(".tgz")}
    R["openscene_sensor_test_camera_local"] = {"dir": CAM_TGZ, "n": len(shards), "bytes": sum(shards.values()),
                                               "per_shard": shards}

    # ---- 3. W3's exported AgentInputs (seam: token list, fingerprints, human future; families) --------------------
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))
    ex = {"path": EXPORT, "bytes": fsize(EXPORT), "n_tokens": len(exp["tokens"]),
          "top_keys": sorted(exp.keys())}
    subset = {k: v for k, v in exp.items() if k != "tokens"}
    subset["tokens"] = {t: exp["tokens"][t] for t in sets["sub200"]["tokens"] if t in exp["tokens"]}
    blob = gzip.compress(json.dumps(subset).encode("utf-8"), compresslevel=9)
    ex["sub200_subset_gzip_bytes"] = len(blob)
    ex["sub200_subset_n_tokens"] = len(subset["tokens"])
    ex["per_token_keys"] = sorted(next(iter(exp["tokens"].values())).keys())
    R["w3_export"] = ex
    del exp, subset, blob

    # ---- 4. checkpoint + backbone -----------------------------------------------------------------------------------
    R["checkpoint"] = {"snapshot": SNAP, "snapshot_bytes": fsize(SNAP),
                       "model_final_bytes": None,
                       "model_final_note": "not yet written (training runs to ~Oct 1); ckpt_io.py docstring: "
                                           "FULL ~1.3 GB for ViT-L (322.07 M params, fp32) -- ESTIMATED"}
    bb = f"{BACKBONE}/model.safetensors"
    R["backbone"] = {"dir": BACKBONE, "model_safetensors_bytes": fsize(bb), "config_bytes": fsize(f"{BACKBONE}/config.json"),
                     "model_safetensors_sha256": sha256(bb),
                     "needed_for": "PARTIAL snapshots only (snap_epochNNN.pt rebuilds the frozen trunk); "
                                   "model_final.pt is FULL and self-contained"}

    # ---- 5. harness data: navsim_logs + metric cache ------------------------------------------------------------------
    nl = {}
    for name, s in sets.items():
        sz = [fsize(f"{NAVSIM_LOGS}/{lg}.pkl") for lg in s["logs"]]
        nl[name] = {"n_logs": len(s["logs"]), "n_present": sum(x is not None for x in sz),
                    "bytes": sum(x for x in sz if x)}
    R["navsim_logs_test"] = {"dir": NAVSIM_LOGS, "all": dsize(NAVSIM_LOGS, ".pkl"), "tgz": META_TGZ,
                             "tgz_bytes": fsize(META_TGZ), **nl}
    tok_pkl = {}
    for dp, dn, fns in os.walk(MCACHE):
        if "metric_cache.pkl" in fns:
            tok_pkl[os.path.basename(dp)] = os.path.getsize(os.path.join(dp, "metric_cache.pkl"))
    mc = {"dir": MCACHE, "n_pkl": len(tok_pkl), "bytes_all_pkl": sum(tok_pkl.values()),
          "metadata": dsize(f"{MCACHE}/metadata")}
    for name, s in sets.items():
        got = [tok_pkl.get(t) for t in s["tokens"]]
        mc[name] = {"n_tokens": len(got), "n_present": sum(x is not None for x in got),
                    "bytes": sum(x for x in got if x)}
    R["metric_cache_navtest"] = mc

    # ---- 6. code + environments ------------------------------------------------------------------------------------------
    seam_mods = ["planner.py", "model.py", "ckpt_io.py", "load_dinov3.py", "navtrain_scenarios.py", "calib_table.py"]
    R["code"] = {
        "refe_seam_modules": {m: fsize(f"{PKG}/refe/{m}") for m in seam_mods},
        "augment_routes": fsize(f"{PKG}/code/augment_routes.py"),
        "eval_dir_py": dsize(f"{PKG}/eval", ".py"),
        "refe_dir_py": dsize(f"{PKG}/refe", ".py"),
        "w3_code": dsize(f"{W3}/code", ".py"),
        "e1_wrapper": fsize(E1),
        "ev6_code": dsize(EV6, ".py"),
        "navsim_v11_tree": dsize(V11), "navsim_v11_zip_bytes": fsize(V11_ZIP),
        "nuplan_devkit_navsim": dsize(NUPLAN_DEVKIT_NAVSIM),
        "drivezero_checkout": {"src": dsize(f"{DZ}/DriveRL/src"), "nuplan_devkit_fork": dsize(f"{DZ}/DriveRL/nuplan-devkit"),
                               "release": dsize(f"{DZ}/DriveRL/release")},
    }
    R["venvs"] = {"driverl_eval": dsize(VENV_DRIVERL, skip_dirs=()), "navsim_crun": dsize(VENV_NAVSIM, skip_dirs=())}
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(f"{OUT}/input_sizes.json", "w", encoding="utf-8"), indent=1)
    print(f"ZZMEASURE_OK {R['seconds']} s -> {OUT}/input_sizes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
