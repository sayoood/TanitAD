#!/usr/bin/env python3
"""refcv8 WP-C: mine the two agent-join label defects of D3 into a sha12-keyed frame list.

Streams the 3-D agent join (``*.jsonl.xz``, one line per labelled frame) ONCE, single process, < 1 GB, CPU only, and
writes ``tanitad.join_label_defects/1`` (``tanitad.data.join_label_hygiene``):

  (1) ``ego_footprint`` -- every frame holding a box whose centre is in x in (-1, 4) m, |y| < 1 m (D3 ``c3b`` rule 1);
  (2) ``track_jumps``   -- every (frame, track) whose WORLD displacement from the previous frame (poses from the v2
      manifest) exceeds 5 m (vehicles) / 2 m (person, stroller) / 3 m (others) -- D3 ``c3b`` rule 2 -- EXCEPT frames in
      which every track moves by the same vector (``median_all_disp >= 0.5 * max_jump``: a POSE defect).

The arithmetic is the D3 script's, line for line (``scripts/c3b_agents_detail.py``), so the aggregate counts it prints
are directly comparable with D3's banked ``raw/c3b_agents_detail.json``: 693 boxes / 18 clips / 84.27 % within 0.5 m of
(1.4, 0); 6,410 event frames / 887 clips / 853 pose-glitch-like / 5,822 single-track-like. ``--check-d3 <json>`` asserts
that equality and exits non-zero on any difference (the ACCEPTANCE of the miner).

No raw clip id or track id is written: clips are sha12 of the clip id, tracks sha12 of the track id.

    python mine_join_label_defects.py --join <join.jsonl.xz> --manifest <train _v2manifest.pt> \
        --manifest <eval _v2manifest.pt> --out <defects.json> [--check-d3 <c3b_agents_detail.json>]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import lzma
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanitad.data import join_label_hygiene as H  # noqa: E402


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def load_poses(manifests) -> dict:
    """``{clip_id: [T, >=3] float64}`` from the v2 manifests (clip_id list + poses list)."""
    import torch
    poses: dict = {}
    for mp in manifests:
        man = torch.load(mp, map_location="cpu", weights_only=False)
        for c, P in zip(man["clip_id"], man["poses"]):
            poses[c] = P.numpy().astype(np.float64)
    return poses


def mine(lines, poses: dict, *, progress: int = 0, max_lines: int = 10 ** 12) -> dict:
    """``lines``: an iterable of join lines (str). Returns the raw result dict (see ``main``)."""
    ego_frames = collections.defaultdict(set)
    ego_boxes = []                                        # (sha12, frame, cx, cy, l, w)
    jump_events = []                                      # D3's per-frame event rows
    jump_tracks = collections.defaultdict(set)            # sha12 -> {(frame, track sha12)} (pose-glitch frames excluded)
    n_lines = n_records = 0
    prev = None
    last_clip = None
    t0 = time.time()
    for line in lines:
        n_lines += 1
        if n_lines > max_lines:
            break
        r = json.loads(line)
        cid = r["clip_id"]
        if cid not in poses:
            continue
        n_records += 1
        fi = r["frame_idx"]
        ag = r["agents"]
        n = len(ag)
        if cid != last_clip:
            prev = None
            last_clip = cid
        if n == 0:
            prev = None
            continue
        P = poses[cid]
        cx = np.fromiter((a["cx"] for a in ag), np.float64, n)
        cy = np.fromiter((a["cy"] for a in ag), np.float64, n)
        cls = [a["cls"] for a in ag]
        tid = [a["track_id"] for a in ag]
        # (1) ego footprint
        eg = np.where(H.is_ego_footprint(cx, cy))[0]
        for k in eg:
            a = ag[k]
            ego_boxes.append((H.clip_key(cid), fi, float(a["cx"]), float(a["cy"]), float(a["l"]), float(a["w"]),
                              str(a["cls"])))
            ego_frames[cid].add(fi)
        # (2) world-frame continuity
        if 0 <= fi < len(P) - 1:
            x0, y0, ya = P[fi, 0], P[fi, 1], P[fi, 2]
            c_, s_ = math.cos(ya), math.sin(ya)
            X = x0 + cx * c_ - cy * s_
            Y = y0 + cx * s_ + cy * c_
            cur = {}
            for k in range(n):
                if tid[k] not in cur:
                    cur[tid[k]] = (X[k], Y[k], H.class_kind(cls[k]))
            if prev is not None and prev[0] == fi - 1:
                jl, jt = [], []
                for t_, (px, py, k_) in prev[1].items():
                    q = cur.get(t_)
                    if q is None:
                        continue
                    dx_, dy_ = q[0] - px, q[1] - py
                    if math.hypot(dx_, dy_) > H.JUMP_M[k_]:
                        jl.append((dx_, dy_))
                        jt.append(t_)
                if jl:
                    allv = [(q[0] - prev[1][t_][0], q[1] - prev[1][t_][1]) for t_, q in cur.items() if t_ in prev[1]]
                    mdx = float(np.median([a for a, b in allv]))
                    mdy = float(np.median([b for a, b in allv]))
                    med = round(math.hypot(mdx, mdy), 2)
                    mx = round(max(math.hypot(a, b) for a, b in jl), 1)
                    jump_events.append({"clip": H.clip_key(cid), "frame_idx": fi, "n_compared": len(allv),
                                        "n_jump": len(jl), "median_all_disp_m": med, "max_jump_m": mx})
                    if not (med >= H.POSE_GLITCH_FRAC * mx):          # D3's pose_glitch_like class is NOT listed
                        for t_ in jt:
                            jump_tracks[cid].add((fi, H.track_key(t_)))
            prev = (fi, cur)
        else:
            prev = None
        if progress and n_lines % progress == 0:
            print("lines", n_lines, round(time.time() - t0, 1), "s", flush=True)
    return {"n_lines": n_lines, "n_records": n_records, "ego_frames": ego_frames, "ego_boxes": ego_boxes,
            "jump_events": jump_events, "jump_tracks": jump_tracks}


def summarise(res: dict) -> dict:
    """The aggregates D3 banked in ``raw/c3b_agents_detail.json`` (same definitions, same rounding)."""
    eb = res["ego_boxes"]
    clips = sorted({c for c, *_ in eb})
    near = sum(1 for (_c, _f, cx, cy, *_r) in eb if math.hypot(cx - 1.4, cy) < 0.5)      # D3: strict < 0.5
    ev = res["jump_events"]
    return {
        "ego_footprint": {
            "n_boxes_listed": len(eb), "n_clips": len(res["ego_frames"]),
            "frames_total": sum(len(v) for v in res["ego_frames"].values()),
            "frac_within_0.5m_of_(1.4,0)": (near / len(eb)) if eb else None,
            "per_clip_frames": sorted((len(v) for v in res["ego_frames"].values()), reverse=True)[:20],
            "clip_sha12": clips[:20],
            "class_counts": dict(collections.Counter(c for *_a, c in eb))},
        "jump_events": {
            "n_event_frames": len(ev), "n_clips": len({e["clip"] for e in ev}),
            "pose_glitch_like": sum(1 for e in ev if e["median_all_disp_m"] >= H.POSE_GLITCH_FRAC * e["max_jump_m"]),
            "single_track_like": sum(1 for e in ev if e["n_jump"] == 1)}}


def check_against_d3(agg: dict, d3_path: Path) -> list:
    """Differences between this mining and D3's banked aggregates (empty list = identical)."""
    d3 = json.loads(Path(d3_path).read_text(encoding="utf-8"))
    ef, je = d3["ego_footprint"], d3["jump_events"]
    a, b = agg["ego_footprint"], agg["jump_events"]
    diffs = []
    for k in ("n_boxes_listed", "n_clips", "frames_total", "per_clip_frames", "clip_sha12"):
        if a[k] != ef[k]:
            diffs.append((f"ego_footprint.{k}", a[k], ef[k]))
    if abs(a["frac_within_0.5m_of_(1.4,0)"] - ef["frac_within_0.5m_of_(1.4,0)"]) > 1e-12:
        diffs.append(("ego_footprint.frac_within_0.5m", a["frac_within_0.5m_of_(1.4,0)"], ef["frac_within_0.5m_of_(1.4,0)"]))
    if a["class_counts"] != ef["class_counts"]:
        diffs.append(("ego_footprint.class_counts", a["class_counts"], ef["class_counts"]))
    for k in ("n_event_frames", "n_clips"):
        if b[k] != je[k]:
            diffs.append((f"jump_events.{k}", b[k], je[k]))
    if b["pose_glitch_like"] != je["pose_glitch_like(median_all_disp>=0.5*max_jump)"]:
        diffs.append(("jump_events.pose_glitch_like", b["pose_glitch_like"], je["pose_glitch_like(median_all_disp>=0.5*max_jump)"]))
    if b["single_track_like"] != je["single_track_like(n_jump==1)"]:
        diffs.append(("jump_events.single_track_like", b["single_track_like"], je["single_track_like(n_jump==1)"]))
    return diffs


def dumps_pretty(obj, indent: int = 1, _level: int = 0) -> str:
    """JSON with dicts indented and every LIST on one line (the frame lists are long; one element per line is 4x bigger)."""
    pad = " " * (indent * (_level + 1))
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad}{json.dumps(str(k))}: {dumps_pretty(v, indent, _level + 1)}" for k, v in obj.items()]
        nl = chr(10)
        return "{" + nl + ("," + nl).join(items) + nl + " " * (indent * _level) + "}"
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def build_file(res: dict, agg: dict, prov: dict) -> dict:
    ego = {H.clip_key(c): sorted(v) for c, v in sorted(res["ego_frames"].items(), key=lambda kv: H.clip_key(kv[0]))}
    ev = {H.clip_key(c): sorted([f, t] for f, t in v)
          for c, v in sorted(res["jump_tracks"].items(), key=lambda kv: H.clip_key(kv[0]))}
    n_ev = sum(len(v) for v in ev.values())
    return {
        "schema": H.SCHEMA,
        "ego_footprint": {
            "rule": {"x_open_interval_m": list(H.EGO_X_RANGE_M), "abs_y_lt_m": H.EGO_ABS_Y_M,
                     "frame": "rig, +x forward, +y left, rear-axle origin", "source": "D3 c3b_agents_detail.py rule (1)"},
            "n_boxes": agg["ego_footprint"]["n_boxes_listed"], "n_clips": agg["ego_footprint"]["n_clips"],
            "frames": ego},
        "track_jumps": {
            "rule": {"jump_m_per_step": dict(H.JUMP_M), "kinds": "v vehicles, p person+stroller, o others (rider included)",
                     "pose_glitch_excluded": f"median_all_disp >= {H.POSE_GLITCH_FRAC} * max_jump",
                     "meaning": "event [f, t]: track t jumped between frame f-1 and f; its rate rows at records f-1 and f "
                                "are corrupted", "source": "D3 c3b_agents_detail.py rule (2)"},
            "n_event_frames_all": agg["jump_events"]["n_event_frames"],
            "n_pose_glitch_like_excluded": agg["jump_events"]["pose_glitch_like"],
            "n_track_events_listed": n_ev, "events": ev},
        "provenance": prov}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--join", required=True)
    ap.add_argument("--manifest", action="append", required=True, help="v2 manifest .pt (repeat for train and eval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-d3", default=None, help="D3 raw/c3b_agents_detail.json: assert identical aggregates")
    ap.add_argument("--max-lines", type=int, default=10 ** 12)
    ap.add_argument("--progress", type=int, default=100000)
    a = ap.parse_args(argv)
    poses = load_poses(a.manifest)
    jp = Path(a.join)
    t0 = time.time()
    with lzma.open(jp, "rt", encoding="utf-8") as fh:
        res = mine(fh, poses, progress=a.progress, max_lines=a.max_lines)
    agg = summarise(res)
    print(json.dumps(agg["jump_events"]), flush=True)
    print(json.dumps({k: v for k, v in agg["ego_footprint"].items() if k != "clip_sha12"}), flush=True)
    prov = {"join_md5": _md5(jp), "join_bytes": jp.stat().st_size, "n_lines": res["n_lines"], "n_records": res["n_records"],
            "manifests": [{"md5": _md5(Path(m)), "bytes": Path(m).stat().st_size} for m in a.manifest],
            "tool": "stack/scripts/mine_join_label_defects.py", "wall_s": round(time.time() - t0, 1),
            "d3_aggregates": agg}
    out = build_file(res, agg, prov)
    Path(a.out).write_text(dumps_pretty(out) + chr(10), encoding="utf-8", newline="")
    print("[mine] wrote", a.out, "ego frames", sum(len(v) for v in out["ego_footprint"]["frames"].values()),
          "track events", out["track_jumps"]["n_track_events_listed"], flush=True)
    if a.check_d3:
        diffs = check_against_d3(agg, Path(a.check_d3))
        if diffs:
            for d in diffs:
                print("[mine] D3 MISMATCH", d, flush=True)
            return 2
        print("[mine] D3 aggregates reproduced EXACTLY", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
