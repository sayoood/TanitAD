"""MEASURE the ORIGINAL 13.13 h parity corpus profile from its own epcache.

Same labeler, same thresholds, same aggregation as v2_corpus_qa_scan.py, so the
achieved-vs-target-vs-parity table is three MEASURED columns, not two measured
and one inherited from prose. Reads ONLY poses (mmap keeps the 111 MB frame
tensor off the page cache). READ-ONLY.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import torch

_STACK = os.environ.get("TANITAD_STACK", "/workspace/TanitAD/stack")
_QA_LIB = os.environ.get("QA_LIB", "")
if _QA_LIB:
    sys.path.insert(0, _QA_LIB)
sys.path.insert(1 if _QA_LIB else 0, _STACK)
sys.path.append(os.path.join(_STACK, "scripts"))
import refb_labels as rl                                       # noqa: E402
assert hasattr(rl, "route_from_future_v21"), rl.__file__

HZ = 10.0


def one(path):
    r = {"file": os.path.basename(path), "ok": False, "err": []}
    try:
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        p = d["poses"].float().clone()
        r["episode_id"] = int(d["episode_id"])
        r["frames_shape"] = list(d["frames_u8"].shape)
    except Exception as e:                                     # noqa: BLE001
        r["err"].append(f"LOAD:{type(e).__name__}:{str(e)[:120]}")
        return r
    T = int(p.shape[0])
    if p.ndim != 2 or p.shape[1] != 4 or T <= rl.LABEL_HORIZON:
        r["err"].append(f"SHAPE:{list(p.shape)}")
        return r
    v, yaw = p[:, 3], p[:, 2]
    h1 = torch.bincount(rl.maneuver_labels(p, rl.LABEL_HORIZON), minlength=5).tolist()
    h2 = torch.bincount(rl.maneuver_labels_v2(p, rl.LABEL_HORIZON), minlength=5).tolist()
    dyaw = yaw[1:] - yaw[:-1]
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
    junction = 0
    for tt in range(0, T, 5):
        if rl.route_from_future_v21(p, tt)["reason"] == "tight_transient":
            junction = 1
            break
    r.update(T_out=T, mean_v=float(v.mean()),
             stop_frac=float((v < 0.5).float().mean()),
             net_head=abs(math.degrees(float(((yaw[-1] - yaw[0] + math.pi)
                                              % (2 * math.pi)) - math.pi))),
             cum_head=math.degrees(float(dyaw.abs().sum())), junction=junction,
             stopped=float((v < 1.0).float().mean()),
             city=float(((v >= 1) & (v <= 12)).float().mean()),
             hw=float((v > 12).float().mean()),
             lk=h1[0], tl=h1[1], tr=h1[2], ac=h1[3], bs=h1[4],
             lk2=h2[0], tl2=h2[1], tr2=h2[2], ac2=h2[3], bs2=h2[4],
             nlab=int(sum(h1)), has_stop=int(bool((v < 0.5).any())),
             has_brake=int(h1[4] > 0), has_turn=int(h1[1] + h1[2] > 0), ok=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(1)
    files = sorted(glob.glob(os.path.join(a.cache, "ep_*.pt")))
    print(f"[parity] {len(files)} episodes", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, files, chunksize=8))
    good = [r for r in rows if r["ok"]]
    tot = lambda c: sum(int(r[c]) for r in good)               # noqa: E731
    agg = {"cache_dir": a.cache, "episodes": len(rows), "ok": len(good),
           "bad": [r for r in rows if not r["ok"]][:20],
           "refb_labels_from": rl.__file__,
           "scan_seconds": round(time.time() - t0, 1),
           "total_stacked_frames": tot("T_out"),
           "hours_stacked": round(tot("T_out") / HZ / 3600.0, 4),
           "total_label_steps": tot("nlab"),
           "man_v1_counts": {k: tot(k) for k in ("lk", "tl", "tr", "ac", "bs")},
           "man_v2_counts": {k: tot(k) for k in ("lk2", "tl2", "tr2", "ac2", "bs2")},
           "speed_steps": {k: sum(r[k] * r["T_out"] for r in good)
                           for k in ("stopped", "city", "hw")},
           "speed_clipmean": {k: sum(r[k] for r in good) / len(good)
                              for k in ("stopped", "city", "hw")},
           "presence": {k: sum(int(r[k]) for r in good) / len(good)
                        for k in ("junction", "has_stop", "has_brake", "has_turn")},
           "net_head_gt45": sum(1 for r in good if r["net_head"] > 45) / len(good),
           "net_head_gt90": sum(1 for r in good if r["net_head"] > 90) / len(good),
           "episode_id_unique": len({r["episode_id"] for r in good}),
           "T_out_min": min(r["T_out"] for r in good),
           "T_out_max": max(r["T_out"] for r in good)}
    json.dump(agg, open(a.out + ".json", "w"), indent=2)
    cols = ["file", "episode_id", "T_out", "nlab", "mean_v", "stop_frac", "net_head",
            "cum_head", "junction", "stopped", "city", "hw", "lk", "tl", "tr",
            "ac", "bs", "lk2", "tl2", "tr2", "ac2", "bs2", "has_stop", "has_brake",
            "has_turn"]
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in rows]
    print(json.dumps({k: v for k, v in agg.items() if k != "bad"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
