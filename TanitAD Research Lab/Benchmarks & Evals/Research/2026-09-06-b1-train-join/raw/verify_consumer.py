"""Verify the B1 TRAIN join through the REAL consumer, and characterise v_rel_x.

The consumer is ``train_p8_occupancy.JoinFileReader`` -- what
``refc_v3_train.py --agent-join`` opens for ``--agents head``. Verification is by
LOADING with that class, never by inspecting the file.

v_rel_x is NOT a field in the join. It is computed by the reader
(``with_rates=True`` -> ``agent_slots.track_rates_from_join``) as the finite
difference of an agent's EGO-FRAME along-track coordinate ``cx`` with respect to
the records' own ``t_s``:

    v_rel_x = d(cx)/dt   [m/s]   (v_rel_y = d(cy)/dt, yaw_rate_rel = d(yaw)/dt)

so it is a RELATIVE (ego-frame) longitudinal rate -- the closing rate -- and its
unit is metres per second because ``cx`` is metres and ``t_s`` is seconds.
``dt`` is READ from the records (grid ~0.1007 s), never assumed to be 0.1.

Every print is ASCII (cp1252 box).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
for _p in (_REPO / "stack" / "scripts", _REPO / "stack", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate-clips", type=int, default=400,
                    help="how many clips to load a SECOND time with rates on "
                         "(a full rates load costs several GB RSS)")
    a = ap.parse_args(argv)

    from train_p8_occupancy import JoinFileReader
    try:
        from train_p8_occupancy import episode_uid_of_clip
    except ImportError:
        from tanitad.data.ids import episode_uid_of_clip  # type: ignore

    clips = sorted(json.load(open(a.clips, encoding="utf-8")))
    rep: dict = {"join": a.join}

    # ---- 1. the headline load, no rates ------------------------------------
    r = JoinFileReader(a.join)
    rep["reader"] = {
        "class": "train_p8_occupancy.JoinFileReader",
        "n_records": int(r.n_records),
        "n_clips": int(getattr(r, "n_clips", len(getattr(r, "_clip_of_uid", {})))),
        "max_agents_per_frame": int(r.max_agents_per_frame),
        "has_classes": bool(r.has_classes),
        "has_occlusion_flags": bool(r.has_occlusion_flags),
    }
    print("[verify] %s" % json.dumps(rep["reader"]), flush=True)

    # ---- 2. a HIT and a must-MISS control ----------------------------------
    hit = miss = None
    probe_clip = None
    for cid in clips:
        uid = episode_uid_of_clip(cid)
        g = r.lookup(uid, 10)
        if g is not None:
            probe_clip, hit = cid, g
            miss = r.lookup(uid, 100000)
            break
    rep["lookup"] = {
        "probe_clip": probe_clip,
        "lookup(uid, 10)": ("HIT: %d agents, array %s"
                            % (0 if hit is None else hit.shape[0],
                               None if hit is None else str(hit.shape))),
        "lookup(uid, 100000) MUST-MISS control": repr(r.lookup(
            episode_uid_of_clip(probe_clip), 100000)) if probe_clip else None,
        "must_miss_is_None": miss is None,
    }
    print("[verify] lookup %s" % json.dumps(rep["lookup"]), flush=True)
    if hit is None:
        raise SystemExit("[verify] REFUSING: no clip produced a lookup HIT")
    if miss is not None:
        raise SystemExit("[verify] REFUSING: the must-miss control RETURNED DATA")

    # ---- 3. v_rel_x, through the same reader --------------------------------
    sub = clips[:a.rate_clips]
    ids = [episode_uid_of_clip(c) for c in sub]
    rr = JoinFileReader(a.join, episode_ids=ids, with_rates=True)
    vx, vy, wz = [], [], []
    n_boxes = n_masked = n_frames = 0
    for cid in sub:
        uid = episode_uid_of_clip(cid)
        for f in range(0, 205):
            got = rr.lookup_rates(uid, f)
            if got is None:
                continue
            rates, mask = got
            n_frames += 1
            n_boxes += int(mask.shape[0])
            n_masked += int(mask.sum())
            if mask.any():
                vx.append(rates[mask, 0])
                vy.append(rates[mask, 1])
                wz.append(rates[mask, 2])
    vx = np.concatenate(vx) if vx else np.zeros(0)
    vy = np.concatenate(vy) if vy else np.zeros(0)
    wz = np.concatenate(wz) if wz else np.zeros(0)

    def dist(v, unit):
        if v.size == 0:
            return {"n": 0, "unit": unit}
        return {"n": int(v.size), "unit": unit,
                "mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max()),
                "p01": float(np.percentile(v, 1)),
                "p05": float(np.percentile(v, 5)),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99)),
                "frac_exactly_zero": float((v == 0.0).mean()),
                "frac_nonfinite": float((~np.isfinite(v)).mean())}

    rep["v_rel"] = {
        "present": True,
        "computed_by": "train_p8_occupancy.JoinFileReader(with_rates=True) -> "
                       "tanitad.models.agent_slots.track_rates_from_join",
        "definition": "v_rel_x = d(cx)/dt with cx the agent's along-track "
                      "coordinate in the per-frame EGO frame (+x forward) and "
                      "dt READ from the records' own t_s (grid ~0.1007 s, never "
                      "assumed 0.1); central difference when both neighbours "
                      "carry the track_id, one-sided when one does, MASKED "
                      "when neither (zero is a legitimate value -- a stationary "
                      "car -- so a missing rate is never zero-filled)",
        "units": {"v_rel_x": "m/s", "v_rel_y": "m/s", "yaw_rate_rel": "rad/s"},
        "sample": {"n_clips": len(sub), "n_frames_with_labels": n_frames,
                   "n_boxes": n_boxes, "n_boxes_with_rate": n_masked,
                   "coverage_frac": (round(n_masked / n_boxes, 4)
                                     if n_boxes else None)},
        "v_rel_x": dist(vx, "m/s"),
        "v_rel_y": dist(vy, "m/s"),
        "yaw_rate_rel": dist(wz, "rad/s"),
    }
    print("[verify] v_rel_x coverage %d/%d = %s"
          % (n_masked, n_boxes,
             rep["v_rel"]["sample"]["coverage_frac"]), flush=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print("VERIFY_DONE %s" % a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
