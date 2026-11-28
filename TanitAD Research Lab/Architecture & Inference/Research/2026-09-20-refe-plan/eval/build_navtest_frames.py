"""Extract exactly the camera frames REFe reads for NAVSIM v1.1 `navtest` into per-log/camera zips.

REFe's eval reuses `planner.FrameResolver` UNCHANGED (SPEC_NAVTEST §1): it pairs each camera to the
token's lidar timestamp through the nuPlan DB's own `image` table (nearest, <= 120 ms, per camera --
the rule `build_targets.py` used for training) and reads `<images_root>/<log>_<CAM>.zip::<hash>.jpg`.
So the wanted set is computed with THAT resolver's `_index`, and the OpenScene test archives are
streamed ONCE, keeping only those members. ⛔ Loose JPEGs on exFAT D: would cost ~1 MiB per file
(memory: d-drive-is-exfat); one stored zip per log x camera costs its bytes.

  python eval/build_navtest_frames.py --out D:/Projects/TanitAD/data/refe_navtest/frames \
      [--tokens <W3 subset json>] [--archives "D:/Archive/.../openscene_sensor_test_camera_*.tgz"]
Prints ZZNAVTEST_FRAMES_OK <wanted> <extracted> only when every wanted frame was extracted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, REFE)

NAVTEST_YAML = ("D:/Archive/devbox-C/navsim/devkit/navsim/planning/script/config/common/"
                "train_test_split/scene_filter/navtest.yaml")
ARCHIVES = "D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera/*.tgz"
TEST_DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"


def wanted_frames(tokens_by_log: dict, db_dir: str):
    """{(log, cam): {basename: rel}} + per-token rows + misses, via planner.FrameResolver's own index."""
    import numpy as np
    import navtrain_scenarios as NS
    from planner import FrameResolver, CAMERAS
    fr = FrameResolver(db_dir, "", max_dt_ms=120.0)
    want: dict = {}
    rows, misses = [], []
    for log, toks in sorted(tokens_by_log.items()):
        db = os.path.join(db_dir, f"{log}.db")
        _map, lp = NS._log_rows(db)                 # [(token_hex, timestamp)] in time order
        ts_of = dict(lp)
        per = fr._index(log)
        for tok in toks:
            t_us = ts_of.get(tok)
            if t_us is None:
                misses.append((log, tok, "token not in lidar_pc"))
                continue
            cams = []
            for ch in CAMERAS:
                ts, names = per[ch]
                if ts.size == 0:
                    cams = None
                    misses.append((log, tok, f"{ch}: no image rows"))
                    break
                j = int(np.argmin(np.abs(ts - t_us)))
                dt = abs(int(ts[j]) - int(t_us)) / 1000.0
                if dt > 120.0:
                    cams = None
                    misses.append((log, tok, f"{ch}: nearest {dt:.1f} ms"))
                    break
                rel = names[j]
                want.setdefault((log, ch), {})[os.path.basename(rel)] = rel
                cams.append(rel)
            if cams:
                rows.append({"log": log, "token": tok, "timestamp_us": int(t_us), "cams": cams})
    return want, rows, misses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None, help="W3-format subset json ({'tokens': [...]})")
    ap.add_argument("--archives", default=ARCHIVES)
    ap.add_argument("--db-dir", default=TEST_DB_DIR)
    a = ap.parse_args()
    import navtrain_scenarios as NS
    logs, toks = NS.load_split(NAVTEST_YAML)
    if a.tokens:
        sub = json.load(open(a.tokens, encoding="utf-8"))
        want_toks = {t.lower() for t in (sub["tokens"] if isinstance(sub, dict) else sub)}
        toks = toks & want_toks
    # tokens -> logs through each DB's lidar_pc (the split file does not pair them)
    tokens_by_log: dict = {}
    for log in logs:
        db = os.path.join(a.db_dir, f"{log}.db")
        if not os.path.exists(db):
            continue
        _m, lp = NS._log_rows(db)
        hit = [t for t, _ts in lp if t in toks]
        if hit:
            tokens_by_log[log] = hit
    n_tok = sum(len(v) for v in tokens_by_log.values())
    print(f"  navtest tokens resolved to logs: {n_tok:,} over {len(tokens_by_log)} logs "
          f"(asked {len(toks):,})", flush=True)
    want, rows, misses = wanted_frames(tokens_by_log, a.db_dir)
    n_want = sum(len(v) for v in want.values())
    print(f"  frames wanted: {n_want:,} ({len(rows):,} tokens x 4 cameras, dedup); "
          f"misses at pairing: {len(misses)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    # member path in the archive ends with "<log>/<CAM>/<hash>.jpg"; key by basename (unique hash)
    by_base = {}
    for (log, ch), m in want.items():
        for base, rel in m.items():
            by_base[base] = (log, ch)
    got = {k: set() for k in want}
    zips: dict = {}
    t0 = time.time()
    # ⛔ A ZIP IS UNREADABLE UNTIL IT IS CLOSED -- its central directory is written by close(). The
    # first version closed every zip only after the LAST archive (MEASURED 2026-09-24: no frame was
    # readable for the whole 2 h 5 min run, and E-2 had to run on a separately extracted log). Now
    # every zip is closed at the end of EACH archive -- and in `finally` on a crash -- so a finished
    # archive leaves readable zips and an interrupted run resumes (append mode re-reads the
    # directory; frames already present are skipped). The progress line prints AFTER the close, so
    # "[n] archive done" means readable.
    try:
        for ai, arc in enumerate(sorted(glob.glob(a.archives))):
            n_arc = 0
            with tarfile.open(arc, mode="r|gz") as tf:
                for mem in tf:
                    if not mem.isfile():
                        continue
                    base = os.path.basename(mem.name)
                    key = by_base.get(base)
                    if key is None:
                        continue
                    log, ch = key
                    if not mem.name.replace("\\", "/").endswith(f"{log}/{ch}/{base}"):
                        continue
                    if base in got[key]:
                        continue
                    data = tf.extractfile(mem).read()
                    zf = zips.get(key)
                    if zf is None:
                        zp = os.path.join(a.out, f"{log}_{ch}.zip")
                        zf = zips[key] = zipfile.ZipFile(zp, "a", compression=zipfile.ZIP_STORED)
                        got[key] |= set(zf.namelist())
                        if base in got[key]:
                            continue
                    zf.writestr(base, data)
                    got[key].add(base)
                    n_arc += 1
            for zf in zips.values():
                zf.close()
            zips.clear()
            print(f"    [{ai + 1}] {os.path.basename(arc)}: +{n_arc:,} frames  "
                  f"({sum(len(v) for v in got.values()):,}/{n_want:,}, {time.time() - t0:.0f} s)",
                  flush=True)
    finally:
        for zf in zips.values():
            zf.close()
    n_got = sum(len(got[k] & set(want[k])) for k in want)
    missing = [(k, b) for k in want for b in want[k] if b not in got[k]]
    man = {"tokens": len(rows), "frames_wanted": n_want, "frames_extracted": n_got,
           "missing_frames": len(missing), "pairing_misses": misses[:50],
           "n_pairing_misses": len(misses), "archives": a.archives, "out": a.out,
           "missing_examples": [f"{k[0]}/{k[1]}/{b}" for k, b in missing[:10]]}
    json.dump(rows, open(os.path.join(a.out, "tokens_frames.json"), "w"))
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "pairing_misses"}, indent=1))
    if n_got == n_want and n_want > 0:
        print(f"ZZNAVTEST_FRAMES_OK {n_want} {n_got}")
        return 0
    print("ZZNAVTEST_FRAMES_" + "INCOMPLETE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
