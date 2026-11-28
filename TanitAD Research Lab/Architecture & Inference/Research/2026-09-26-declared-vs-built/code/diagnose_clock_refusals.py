"""E2 (a): which of the clip-clock sidecar's 25 REFUSED clips fall in which split, and which
control refused each -- reported by sid and sha12 (sha256(clip_id)[:12]); no clip id is written.

Inputs (read-only): the sidecar meta (its `refused_by_sid` map), the eval-139 cache (file names
are the clip ids), the A6 train sub-cache on the dev box, and the refcv6 run's recorded
`label_clock` census (train 4,369 / eval 139). `sid = stable_episode_id(clip_id)` is
re-implemented from `tanitad/data/v2_dataset.py:94-95` (blake2b-8 >> 1) so no torch import.

    python diagnose_clock_refusals.py <out.json>
"""
import hashlib
import json
import pathlib
import sys
from collections import Counter

AUDIT = pathlib.Path("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/"
                     "Research/2026-09-26-refcv6-frozen-trunk-audit/raw")
META = AUDIT / "refcv6_clip_clock_sidecar.jsonl.meta.json"
SIDE = AUDIT / "refcv6_clip_clock_sidecar.jsonl"
EVAL = pathlib.Path("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139")
TRAIN_A6 = pathlib.Path("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6")
RUN_CFG = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                       "2026-09-23-refcv6-standard-tests/battery/raw/a5/"
                       "config_resume34500_20260926.json")


def sid_of(clip_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(),
                          "big") >> 1


def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


def clips(d: pathlib.Path) -> dict:
    return {sid_of(p.name[:-len(".v2ep.pt")]): sha12(p.name[:-len(".v2ep.pt")])
            for p in d.glob("*.v2ep.pt")}


def main(out: str) -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    refused = {int(k): v for k, v in meta["refused_by_sid"].items()}
    side_sids = {int(json.loads(l)["sid"]) for l in SIDE.read_text(encoding="utf-8").splitlines()
                 if l.strip()}
    ev, a6 = clips(EVAL), clips(TRAIN_A6)
    cls = lambda why: ("fewer than 20 moving rows" if why.startswith("fewer") else why.split()[0])
    rows = []
    for sid, why in sorted(refused.items()):
        split = "eval139" if sid in ev else ("train (A6 sub-cache)" if sid in a6 else
                                              "not in a dev-box cache (train, Thor-only)")
        rows.append({"sid": sid, "sha12": ev.get(sid) or a6.get(sid), "split": split,
                     "control": cls(why), "detail": why})
    run = json.loads(RUN_CFG.read_text(encoding="utf-8"))["label_clock"]
    ev_unverified = sorted(s for s in ev if s not in side_sids)
    res = {
        "what": "E2(a): the 25 clips the clip-clock sidecar REFUSED, by split and control",
        "evidence_class": "MEASURED (dev box; sidecar meta + eval-139 cache file names)",
        "sidecar_rows": len(side_sids), "sidecar_refused": len(refused),
        "by_control": dict(Counter(r["control"] for r in rows)),
        "eval139_clips_in_cache": len(ev),
        "eval139_unverified": len(ev_unverified),
        "eval139_unverified_are_all_refused": all(s in refused for s in ev_unverified),
        "run_label_clock": {k: {kk: run[k][kk] for kk in ("n_clips", "n_from_sidecar",
                                                          "n_pose_dt_grid_start_0",
                                                          "n_nominal_dt_grid_start_0")}
                            for k in ("train", "eval")},
        "refused": rows,
        "fixable_here": ("NO. 15 clips carry fewer than 20 moving rows (the log inversion needs "
                         "motion to align the grid; the cache stores no per-frame timestamps), "
                         "7 fail K1 (a single (grid_start, dt) does not fit: residual 29-433 ms, "
                         "i.e. a non-uniform grid), 3 fail K3 by 1.3-3.8x its 1e-3 m/s bound. "
                         "Relaxing K1/K3 after seeing these values would move a pre-set control; "
                         "the per-row camera timestamps (raw PhysicalAI, not on this box) would "
                         "clock all 25 exactly."),
    }
    pathlib.Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "refused"}, indent=1))
    for r in rows:
        print(f"  {r['split']:40s} sid {r['sid']:>20d} sha12 {r['sha12'] or '-':12s} {r['detail']}")


if __name__ == "__main__":
    main(sys.argv[1])
