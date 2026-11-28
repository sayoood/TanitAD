#!/usr/bin/env python3
"""run_all.py -- the AUCMAP sweep (PREREG_AUCMAP.md sections 5-9).  CEILING-TYPE where synthetic or logged-level; T1-kin; no driving claim.
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I run_all.py [--workers 4] [--probe]
Markers in raw/: AUCMAP.running, AUCMAP.done, AUCMAP.fail.  Per-task results in raw/tasks/<task>.json (banked as each finishes); merged into raw/AUCMAP_RESULT.json."""
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
RAW = f"{HERE}/raw"

_W = {}


def _setup(lever):
    import aucmap as A
    if "A" not in _W:
        A.init()
        _W["A"] = A
    A = _W["A"]
    if _W.get("lever") != lever:
        _W["rig"] = A.Rig(levers=[lever])
        _W["lever"] = lever
        _W["y"] = A.labels_from(_W["rig"].v0, _W["rig"].g)
    return A, _W["rig"], _W["y"]


def tasks():
    import aucmap as A
    out = []
    for lv in list(A.ADMISSIBLE) + list(A.CEILING):
        for sc in ("ANCHOR", "ALL"):
            out.append(("syn", lv, sc, "GO", False))
    out.append(("syn", "LF100", "ANCHOR", "GO", True))
    for lv in A.ADMISSIBLE:
        out.append(("syn", lv, "ANCHOR", "KEEP", False))
    for lv in A.ADMISSIBLE:
        out.append(("anchor", lv, "ANCHOR", "GO", False))
    return out


def name_of(t):
    kind, lv, sc, lab, perm = t
    return f"{kind}_{lv}_{sc}_{lab}{'_perm' if perm else ''}"


def run_one(t):
    kind, lv, sc, lab, perm = t
    nm = name_of(t)
    path = f"{RAW}/tasks/{nm}.json"
    if os.path.exists(path):
        return nm, "cached", 0.0
    t0 = time.time()
    try:
        A, rig, y = _setup(lv)
        import numpy as np
        scope = (rig.v0 < 8.0) if sc == "ANCHOR" else np.ones(len(rig.v0), bool)
        if kind == "syn":
            res = A.run_synthetic(rig, lv, sc, lab, scope, y[lab], permuted=perm, log=lambda s: None)
        else:
            post, gop, v0p = A.load_post(rig)
            res = A.anchor_curve(rig, lv, post, y["GO"])
            # realized AUC of the posterior on the three label sets (AMEND_1)
            res["auc_v0lt6_card"] = A.auc_mw(post[np.isfinite(post) & (rig.v0 < 6)], y["GO"][np.isfinite(post) & (rig.v0 < 6)])
            res["auc_v0lt8_card"] = A.auc_mw(post[np.isfinite(post)], y["GO"][np.isfinite(post)])
            res["auc_v0lt8_p1pgo"] = A.auc_mw(post[np.isfinite(post)], gop[np.isfinite(post)])
        res["task"] = nm
        res["wall_s"] = time.time() - t0
        os.makedirs(f"{RAW}/tasks", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="\n") as f:
            json.dump(res, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp, path)
        return nm, "ok", time.time() - t0
    except Exception:
        return nm, "FAIL " + traceback.format_exc()[-600:], time.time() - t0


def main():
    workers = 4
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    os.makedirs(f"{RAW}/tasks", exist_ok=True)
    for m in ("running", "done", "fail"):
        p = f"{RAW}/AUCMAP.{m}"
        if os.path.exists(p):
            os.replace(p, p + ".aside." + str(int(time.time())))
    open(f"{RAW}/AUCMAP.running", "w", newline="\n").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    ts = tasks()
    if "--probe" in sys.argv:
        ts = ts[:1]
    log = open(f"{RAW}/AUCMAP_run.log", "a", newline="\n")
    t0 = time.time()
    bad = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for nm, st, dt in pool.imap_unordered(run_one, ts, chunksize=1):
            log.write(f"{time.strftime('%H:%M:%S')} {nm} {st[:80]} {dt:.0f}s\n"); log.flush()
            if st.startswith("FAIL"):
                bad.append((nm, st))
    if bad:
        open(f"{RAW}/AUCMAP.fail", "w", newline="\n").write("\n".join(f"{n}: {s}" for n, s in bad) + "\n")
        os.replace(f"{RAW}/AUCMAP.running", f"{RAW}/AUCMAP.running.ended")
        return 1
    if "--probe" in sys.argv:
        os.replace(f"{RAW}/AUCMAP.running", f"{RAW}/AUCMAP.running.ended")
        return 0
    merged = {}
    for t in ts:
        nm = name_of(t)
        with open(f"{RAW}/tasks/{nm}.json") as f:
            merged[nm] = json.load(f)
    with open(f"{RAW}/AUCMAP_RESULT.json", "w", newline="\n") as f:
        json.dump({"tasks": merged, "wall_s": time.time() - t0, "workers": workers}, f)
    open(f"{RAW}/AUCMAP.done", "w", newline="\n").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    os.replace(f"{RAW}/AUCMAP.running", f"{RAW}/AUCMAP.running.ended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
